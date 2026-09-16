"""Fine-tuning supervisionado econômico, retomável e sem submissão implícita."""
import argparse
import hashlib
import json

from openai import OpenAI

from medical.config import DATA, api_key

BASE = "gpt-4.1-nano-2025-04-14"
STATE = DATA / "finetune.json"


def plan():
    files = [DATA / "processed" / f"{split}.jsonl" for split in ("train", "validation")]
    summary = {"model": BASE, "epochs": 1, "files": {}}
    for path in files:
        content = path.read_bytes()
        records = [json.loads(line) for line in content.decode().splitlines()]
        if len(records) < 10:
            raise ValueError("Treino e validação devem ter pelo menos 10 exemplos.")
        if any([m["role"] for m in r["messages"]] != ["system", "user", "assistant"] for r in records):
            raise ValueError("Exemplo de treinamento inválido.")
        # UTF-8 bytes são um teto conservador para tokens de texto; inclui margem de formato.
        ceiling = sum(sum(len(m["content"].encode()) + 16 for m in r["messages"]) + 16 for r in records)
        summary["files"][path.stem] = {"examples": len(records), "token_upper_bound": ceiling,
                                       "sha256": hashlib.sha256(content).hexdigest()}
    if summary["files"]["train"]["token_upper_bound"] > 250_000:
        raise ValueError("Teto de 250 mil tokens/uma época excedido; reduza --limit em medical.dataset.")
    return summary


def save(state):
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATE)


def submit():
    summary = plan()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"plan": summary}
    if state["plan"] != summary:
        raise ValueError("Os dados mudaram. Preserve o estado anterior e revise antes de iniciar outro treino.")
    if state.get("job_id"):
        return status()
    client = OpenAI(api_key=api_key(), max_retries=0, timeout=60)
    save(state)
    for split in ("train", "validation"):
        if split not in state:
            with (DATA / "processed" / f"{split}.jsonl").open("rb") as f:
                uploaded = client.files.create(file=f, purpose="fine-tune")
            state[split] = uploaded.id
            save(state)
    # Uma chave estável permite retomar timeout de criação sem duplicar intencionalmente o job.
    key = "medassist-" + summary["files"]["train"]["sha256"]
    job = client.fine_tuning.jobs.create(model=BASE, training_file=state["train"],
        validation_file=state["validation"], seed=42, suffix="medassist",
        method={"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 1}}},
        extra_headers={"Idempotency-Key": key})
    state.update(job_id=job.id, status=job.status)
    save(state)
    return {"job_id": job.id, "status": job.status}


def status():
    state = json.loads(STATE.read_text())
    client = OpenAI(api_key=api_key(), max_retries=0, timeout=45)
    job = client.fine_tuning.jobs.retrieve(state["job_id"])
    state.update(status=job.status, fine_tuned_model=job.fine_tuned_model,
                 trained_tokens=job.trained_tokens, result_files=job.result_files,
                 error_code=getattr(job.error, "code", None))
    save(state)
    if job.status == "succeeded" and job.fine_tuned_model:
        (DATA / "model.txt").write_text(job.fine_tuned_model + "\n")
        for file_id in job.result_files:
            (DATA / "processed" / f"training-{file_id}.csv").write_bytes(client.files.content(file_id).content)
    return {k: v for k, v in state.items() if k not in ("plan", "train", "validation")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "submit", "status"], default="plan", nargs="?")
    args = parser.parse_args()
    try:
        print(json.dumps({"plan": plan, "submit": submit, "status": status}[args.action](), indent=2))
    except Exception as exc:
        print(f"Operação não concluída ({type(exc).__name__}). Verifique arquivos, acesso e saldo. Nenhuma chave foi exibida.")
        raise SystemExit(1)
