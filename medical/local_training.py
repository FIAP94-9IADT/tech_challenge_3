"""LoRA local: uma época, métricas reais e adaptador integrado ao LangChain."""
import json
import os
from functools import lru_cache
import time

from medical.config import DATA, ROOT

os.environ.setdefault("HF_HOME", str(DATA / "hf-cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ADAPTER = DATA / "adapter"


def base_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return tokenizer, model.to(device)


def encode_records(tokenizer, split):
    import torch
    records = []
    truncated = 0
    for line in (DATA / "processed" / f"{split}.jsonl").read_text().splitlines():
        messages = json.loads(line)["messages"]
        prefix = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, return_dict=False)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        truncated += len(ids) > 384
        ids = ids[:384]
        if len(prefix) >= len(ids):
            raise ValueError("Prompt excede o contexto de treino; reduza o exemplo.")
        labels = [-100] * len(prefix) + ids[len(prefix):]
        mask = [1] * len(ids)
        pad = 384 - len(ids)
        records.append({"input_ids": torch.tensor(ids + [tokenizer.pad_token_id]*pad),
                        "attention_mask": torch.tensor(mask + [0]*pad),
                        "labels": torch.tensor(labels + [-100]*pad)})
    return records, truncated


def evaluate_loss(model, records):
    import torch
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(records, batch_size=4):
            batch = {k:v.to(model.device) for k,v in batch.items()}
            n = int((batch["labels"][:, 1:] != -100).sum())
            total += float(model(**batch).loss) * n
            count += n
    return total/count


def train():
    import torch
    from peft import LoraConfig, get_peft_model
    if (ADAPTER / "adapter_config.json").exists():
        raise ValueError("Adaptador já existe. Preserve-o antes de iniciar outro experimento.")
    torch.manual_seed(42)
    tokenizer, model = base_model()
    data = {s: encode_records(tokenizer, s) for s in ("train", "validation", "test")}
    before = {s: evaluate_loss(model, data[s][0]) for s in ("validation", "test")}
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    model.train()
    started = time.monotonic()
    losses = []
    loader = torch.utils.data.DataLoader(data["train"][0], batch_size=4, shuffle=True)
    for step, batch in enumerate(loader, 1):
        batch = {k:v.to(model.device) for k,v in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise ValueError("Loss não finita; treinamento interrompido.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 5 == 0 or step == len(loader):
            print(f"step={step}/{len(loader)} loss={losses[-1]:.4f}", flush=True)
    after = {s: evaluate_loss(model, data[s][0]) for s in ("validation", "test")}
    ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)
    report = {"base_model": BASE, "revision": getattr(model.config, "_commit_hash", None),
              "device": str(model.device), "epochs": 1, "seed": 42, "learning_rate": 2e-4,
              "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": ["q_proj", "v_proj"]},
              "max_length": 384, "batch_size": 4, "trainable_parameters": sum(p.numel() for p in parameters),
              "examples": {s:len(data[s][0]) for s in data}, "truncated": {s:data[s][1] for s in data},
              "loss_before": before, "loss_after": after, "training_losses": losses,
              "elapsed_seconds": round(time.monotonic()-started, 2), "api_tokens": 0,
              "note": "Loss teacher-forced nos tokens da resposta; não mede segurança ou acurácia clínica."}
    (ROOT / "docs" / "local_training.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "training_losses"}, indent=2))


@lru_cache(maxsize=1)
def trained_model():
    from peft import PeftModel
    tokenizer, model = base_model()
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return tokenizer, model


def runnable():
    import torch
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    def generate(prompt):
        tokenizer, model = trained_model()
        messages = [{"role": "system" if m.type == "system" else "user", "content": m.content} for m in prompt.to_messages()]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        if inputs["input_ids"].shape[1] > 7000:
            raise ValueError("Contexto local excedido.")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=220, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        n = inputs["input_ids"].shape[1]
        answer = tokenizer.decode(ids[0][n:], skip_special_tokens=True)
        finish = "length" if len(ids[0])-n >= 220 and int(ids[0][-1]) != tokenizer.eos_token_id else "stop"
        return AIMessage(content=answer, response_metadata={"finish_reason": finish},
                         usage_metadata={"input_tokens": n, "output_tokens": len(ids[0])-n, "total_tokens": len(ids[0])})
    return RunnableLambda(generate)


if __name__ == "__main__":
    train()
