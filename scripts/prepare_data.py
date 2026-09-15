"""Curadoria do corpus sintético e partições fixas independentes."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clinical_assistant.anonymization import anonymize_text, contains_direct_identifier
from clinical_assistant.prompting import format_prompt

VALIDATION = {"FAQ-004", "PRO-002", "LAU-002", "REC-004", "SEC-002"}
TEST = {"FAQ-006", "PRO-004", "LAU-004", "REC-005", "SEC-004"}

def prepare(output_dir=None):
    raw_path = ROOT / "data/raw/internal_examples.jsonl"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"id", "category", "instruction", "input", "output", "source"}
    seen, signatures, clean = set(), set(), []
    for original in records:
        if not required.issubset(original):
            raise ValueError("Registro incompleto")
        if original["id"] in seen:
            raise ValueError("Identificador duplicado")
        seen.add(original["id"])
        row = dict(original)
        for field in ("instruction", "input", "output"):
            row[field] = anonymize_text(row[field]).text
            if contains_direct_identifier(row[field]):
                raise ValueError("Identificador detectável remanescente")
        signature = (row["instruction"].casefold(), row["input"].casefold())
        if signature in signatures:
            raise ValueError("Entrada duplicada")
        signatures.add(signature)
        if len(row["output"].split()) < 12:
            raise ValueError("Resposta insuficiente")
        row["prompt"] = format_prompt(row["instruction"], row["input"])
        row["text"] = row["prompt"] + row["output"]
        clean.append(row)
    if not (VALIDATION | TEST).issubset(seen) or VALIDATION & TEST:
        raise ValueError("Partições inválidas")
    splits = {
        "train": [r for r in clean if r["id"] not in VALIDATION | TEST],
        "validation": [r for r in clean if r["id"] in VALIDATION],
        "test": [r for r in clean if r["id"] in TEST],
    }
    assert any(r["source"] == "MODELO-PRESCRICAO" for r in splits["train"])
    output_dir = Path(output_dir or ROOT / "data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = {
        "synthetic_data": True, "total_examples": len(clean),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "partitions": {name: {"count": len(rows), "ids": [r["id"] for r in rows],
            "categories": dict(Counter(r["category"] for r in rows))} for name, rows in splits.items()},
        "detected_identifiers_after_processing": 0,
        "limitation": "Regex não comprova anonimização irreversível. Partições compartilham domínio e protocolos; medem tarefas novas no domínio conhecido.",
    }
    (output_dir / "curation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
