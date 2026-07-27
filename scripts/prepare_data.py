"""Prepara os exemplos sintéticos para fine-tuning supervisionado."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_assistant.anonymization import anonymize_text, contains_direct_identifier  # noqa: E402


RAW_PATH = ROOT / "data" / "raw" / "internal_examples.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed"


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def format_example(record: dict[str, str]) -> dict[str, str]:
    instruction = anonymize_text(record["instruction"]).text
    input_text = anonymize_text(record["input"]).text
    output = anonymize_text(record["output"]).text
    text = (
        "<s>[INST] Você é um assistente institucional de apoio clínico. "
        "Não diagnostique nem prescreva. Use somente o contexto informado e preserve a revisão humana.\n\n"
        f"Instrução: {instruction}\nContexto: {input_text} [/INST] {output}</s>"
    )
    return {
        "id": record["id"],
        "category": record["category"],
        "source": record["source"],
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": text,
    }


def validate(records: list[dict[str, str]]) -> None:
    required = {"id", "category", "source", "instruction", "input", "output", "text"}
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Há identificadores de exemplo duplicados.")
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Campos ausentes em {record.get('id')}: {sorted(missing)}")
        if any(contains_direct_identifier(record[field]) for field in ("instruction", "input", "output")):
            raise ValueError(f"Identificador direto remanescente em {record['id']}")
        if len(record["output"].split()) < 12:
            raise ValueError(f"Resposta curta demais em {record['id']}")


def stratified_split(records: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for record in records:
        grouped.setdefault(record["category"], []).append(record)
    train, test = [], []
    for category in sorted(grouped):
        ordered = sorted(grouped[category], key=lambda item: item["id"])
        test.append(ordered[-1])
        train.extend(ordered[:-1])
    return train, test


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_jsonl(RAW_PATH)
    processed = [format_example(record) for record in raw]
    validate(processed)
    train, test = stratified_split(processed)
    write_jsonl(OUTPUT_DIR / "train.jsonl", train)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test)
    report = {
        "synthetic_data": True,
        "total_examples": len(processed),
        "train_examples": len(train),
        "test_examples": len(test),
        "categories": dict(sorted(Counter(item["category"] for item in processed).items())),
        "sources": sorted({item["source"] for item in processed}),
        "direct_identifiers_detected_after_processing": 0,
        "split_strategy": "um exemplo por categoria no teste; demais exemplos no treino",
    }
    (OUTPUT_DIR / "curation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
