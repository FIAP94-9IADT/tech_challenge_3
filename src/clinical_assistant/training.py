"""Experimento LoRA reproduzível em T5; teste usado somente após seleção."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]

def run(root=Path("."), epochs=6):
    root = Path(root).resolve()
    os.environ.setdefault("HF_HOME", str(root / ".hf-cache"))
    torch.set_num_threads(2)
    set_seed(42)
    model_id = "google/flan-t5-small"
    revision = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    base = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
    rows = {name: read_rows(root / f"data/processed/{name}.jsonl") for name in ("train", "validation", "test")}
    lengths = {name: {"max_input": max(len(tokenizer.encode(r["prompt"])) for r in values),
                     "max_target": max(len(tokenizer.encode(r["output"])) for r in values)}
               for name, values in rows.items()}
    if any(v["max_input"] > 512 or v["max_target"] > 256 for v in lengths.values()):
        raise ValueError("Comprimentos excedem os limites; revisar dados antes de treinar")
    def encoded(row):
        batch = tokenizer(row["prompt"], return_tensors="pt")
        batch["labels"] = tokenizer(text_target=row["output"], return_tensors="pt").input_ids
        return batch
    def loss_on(model, examples):
        model.eval()
        with torch.no_grad():
            return sum(model(**encoded(r)).loss.item() for r in examples) / len(examples)
    def predict(model, row):
        model.eval()
        with torch.no_grad():
            result = model.generate(**tokenizer(row["prompt"], return_tensors="pt"),
                                    max_new_tokens=160, do_sample=False)
        return tokenizer.decode(result[0], skip_special_tokens=True)
    # As respostas do modelo base são registradas sem orientar hiperparâmetros.
    base_predictions = [predict(base, r) for r in rows["test"]]
    base_test_loss = loss_on(base, rows["test"])
    model = get_peft_model(base, LoraConfig(task_type="SEQ_2_SEQ_LM", r=8,
                lora_alpha=16, lora_dropout=0.05, target_modules=["q", "v"]))
    trainable, total = model.get_nb_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    out = root / "models/clinical-t5-lora"
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(rows["train"])
        random.Random(42 + epoch).shuffle(order)
        losses = []
        optimizer.zero_grad()
        for i, row in enumerate(order):
            loss = model(**encoded(row)).loss
            losses.append(loss.item())
            group_size = min(4, len(order) - (i // 4) * 4)
            (loss / group_size).backward()
            if (i + 1) % 4 == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        val = loss_on(model, rows["validation"])
        entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation_loss": val}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if val < best:
            best = val
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
    from peft import PeftModel
    del optimizer, model, base
    import gc
    gc.collect()
    adapted = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision), out)
    comparisons = [{"id": row["id"], "reference": row["output"], "prompt": row["prompt"],
                    "base": before, "adapted": predict(adapted, row)}
                   for row, before in zip(rows["test"], base_predictions)]
    from .evaluation import lexical_recall
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": model_id,
        "model_revision": getattr(adapted.config, "_commit_hash", None),
        "technique": "LoRA sem quantização, CPU float32", "seed": 42, "epochs": epochs,
        "learning_rate": 2e-4, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
        "trainable_parameters": trainable, "total_parameters": total,
        "history": history, "selected_epoch": min(history, key=lambda e: e["validation_loss"])["epoch"],
        "base_test_loss": base_test_loss, "adapted_test_loss": loss_on(adapted, rows["test"]),
        "token_lengths": lengths, "comparisons": comparisons,
        "base_lexical_recall": sum(lexical_recall(c["reference"], c["base"]) for c in comparisons) / len(comparisons),
        "adapted_lexical_recall": sum(lexical_recall(c["reference"], c["adapted"]) for c in comparisons) / len(comparisons),
        "dataset_hashes": {n: hashlib.sha256((root / f"data/processed/{n}.jsonl").read_bytes()).hexdigest() for n in rows},
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft")},
    }
    evidence = root / "docs/results"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, label in (("train_loss", "Treino"), ("validation_loss", "Validação")):
        plt.plot([e["epoch"] for e in history], [e[key] for e in history], marker="o", label=label)
    plt.xlabel("Época")
    plt.ylabel("Perda média por exemplo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(evidence / "loss.png")
    plt.close()
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    run(epochs=args.epochs)
