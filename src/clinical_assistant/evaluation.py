"""Métricas simples e interpretáveis para avaliação do protótipo."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .safety import HUMAN_REVIEW_NOTICE, UNSAFE_OUTPUT


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-zÀ-ÿ0-9-]{3,}", text.lower()))


def lexical_recall(reference: str, prediction: str) -> float:
    expected = token_set(reference)
    predicted = token_set(prediction)
    return len(expected & predicted) / len(expected) if expected else 1.0


def evaluate_answers(items: Iterable[dict[str, object]]) -> dict[str, float]:
    records = list(items)
    if not records:
        return {"lexical_recall": 0.0, "citation_rate": 0.0, "safety_rate": 0.0}
    recalls, citations, safety = [], [], []
    for item in records:
        answer = str(item["answer"])
        recalls.append(lexical_recall(str(item.get("reference", "")), answer))
        citations.append(bool(item.get("sources")))
        safety.append(not UNSAFE_OUTPUT.search(answer) and HUMAN_REVIEW_NOTICE in answer)
    total = len(records)
    return {
        "lexical_recall": round(sum(recalls) / total, 4),
        "citation_rate": round(sum(citations) / total, 4),
        "safety_rate": round(sum(safety) / total, 4),
    }
