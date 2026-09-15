"""Prepara e valida o corpus institucional sintético.

Os registros representam protocolos, perguntas frequentes, laudos,
procedimentos, modelos de receita e mensagens de segurança. O objetivo é
demonstrar o pipeline sem expor dados de pacientes reais.
"""
from __future__ import annotations

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

def prepare(output_dir: Path | None = None) -> dict[str, object]:
    """Remove identificadores reconhecidos, verifica a curadoria e cria partições fixas."""
    raw_path = ROOT / "data/raw/internal_examples.jsonl"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"id", "category", "instruction", "input", "output", "source"}
    seen_ids, signatures, clean = set(), set(), []
    for original in records:
        if not required.issubset(original): raise ValueError("Registro incompleto.")
        if original["id"] in seen_ids: raise ValueError("Identificador duplicado.")
        seen_ids.add(original["id"]); row = dict(original)
        for field in ("instruction", "input", "output"):
            row[field] = anonymize_text(str(row[field])).text
            if contains_direct_identifier(row[field]): raise ValueError("Identificador direto remanescente após a anonimização.")
        signature = (row["instruction"].casefold(), row["input"].casefold())
        if signature in signatures: raise ValueError("Entrada duplicada.")
        signatures.add(signature)
        if len(row["output"].split()) < 12: raise ValueError("Resposta insuficiente para o treinamento.")
        row["origin"] = "institucional_sintetico"; row["prompt"] = format_prompt(row["instruction"], row["input"])
        clean.append(row)
    if not (VALIDATION | TEST).issubset(seen_ids) or VALIDATION & TEST: raise ValueError("Partições inválidas.")
    splits = {"train": [r for r in clean if r["id"] not in VALIDATION | TEST], "validation": [r for r in clean if r["id"] in VALIDATION], "test": [r for r in clean if r["id"] in TEST]}
    if not any(r["source"] == "MODELO-PRESCRICAO" for r in splits["train"]): raise ValueError("O modelo de receita deve permanecer no treinamento.")
    target = Path(output_dir or ROOT / "data/processed"); target.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (target / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
    report = {"synthetic_data": True, "total_examples": len(clean), "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "detected_identifiers_after_processing": 0,
        "partitions": {name: {"count": len(values), "ids": [row["id"] for row in values], "categories": dict(Counter(row["category"] for row in values))} for name, values in splits.items()},
        "limitation": "A anonimização baseada em padrões não comprova remoção irreversível de identificadores. Os exemplos são sintéticos e servem apenas à demonstração do pipeline."}
    (target / "curation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__": print(json.dumps(prepare(), ensure_ascii=False, indent=2))
