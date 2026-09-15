"""Indicadores operacionais; não equivalem a validação clínica."""
import re
from .safety import UNSAFE_OUTPUT

def token_set(text):
    return set(re.findall(r"[A-Za-zÀ-ÿ0-9-]{3,}", text.lower()))

def lexical_recall(reference, prediction):
    expected = token_set(reference)
    return len(expected & token_set(prediction)) / len(expected) if expected else 0.0

def evaluate_answers(items):
    rows = list(items)
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_lexical_recall": sum(lexical_recall(str(r.get("reference", "")), str(r["answer"])) for r in rows) / len(rows),
        "retrieval_presence_rate": sum(bool(r.get("sources")) for r in rows) / len(rows),
        "intervention_pattern_rate": sum(bool(UNSAFE_OUTPUT.search(str(r["answer"]))) for r in rows) / len(rows),
        "limitation": "Presença de documentos não mede suporte factual; padrões textuais não comprovam segurança.",
    }
