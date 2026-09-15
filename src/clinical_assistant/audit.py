"""Registro local de metadados, inclusive falhas, sem texto livre."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

class AuditLogger:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {k: state.get(k) for k in ("request_id", "backend", "route", "output_valid",
                  "elapsed_seconds", "error_type", "steps", "critical", "alerts", "validation_reasons")}
        record.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            request_hash=hashlib.sha256(str(state.get("question", "")).encode()).hexdigest(),
            patient_ref_hash=hashlib.sha256(str(state.get("patient_id", "")).encode()).hexdigest(),
            redaction_count=sum(state.get("redactions", {}).values()),
            sources=[{"id": s["source_id"], "version": s.get("version")} for s in state.get("sources", [])])
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
