"""Auditoria JSONL sem persistência do texto clínico original."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, state: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        question = str(state.get("question", ""))
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "patient_id": state.get("patient_id"),
            "redaction_count": sum(state.get("redactions", {}).values()),
            "critical": bool(state.get("critical")),
            "alerts": state.get("alerts", []),
            "sources": [item.get("source_id") for item in state.get("sources", [])],
            "route": state.get("route"),
            "output_valid": bool(state.get("output_valid")),
            "validation_reasons": state.get("validation_reasons", []),
            "steps": state.get("steps", []),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
