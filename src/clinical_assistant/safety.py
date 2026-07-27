"""Regras explícitas de risco e limites de atuação."""

from __future__ import annotations

import re
from dataclasses import dataclass


CRITICAL_PATTERNS = {
    "dor torácica com sinal associado": re.compile(r"dor (?:no peito|tor[aá]cica).*(?:dispneia|falta de ar|s[ií]ncope|desmai|sudorese|confus)", re.I | re.S),
    "possível deterioração infecciosa": re.compile(r"(?:infec|febre|sepse).*(?:hipotens|confus|falta de ar|olig[uú]ria|lactato)", re.I | re.S),
    "possível alteração neurológica aguda": re.compile(r"(?:fraqueza|paralisia|fala).*(?:s[uú]bit|repentin|um lado)", re.I | re.S),
}

PRESCRIPTION_REQUEST = re.compile(
    r"\b(?:prescrev|receit|qual (?:rem[eé]dio|medicamento)|dose|dosagem|suspend|ajust(?:e|ar) (?:a )?medica)",
    re.I,
)

UNSAFE_OUTPUT = re.compile(
    r"\b(?:tome|administre|prescrevo|inicie|suspenda|aumente a dose|reduza a dose)\b",
    re.I,
)

HUMAN_REVIEW_NOTICE = (
    "A resposta organiza informações para apoio profissional e requer validação "
    "da equipe assistencial antes de qualquer decisão clínica."
)


@dataclass(frozen=True)
class SafetyAssessment:
    critical: bool
    alerts: tuple[str, ...]
    prescription_request: bool


def assess_input(question: str) -> SafetyAssessment:
    alerts = tuple(label for label, pattern in CRITICAL_PATTERNS.items() if pattern.search(question))
    return SafetyAssessment(
        critical=bool(alerts),
        alerts=alerts,
        prescription_request=bool(PRESCRIPTION_REQUEST.search(question)),
    )


def validate_output(text: str, has_sources: bool) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    if UNSAFE_OUTPUT.search(text):
        reasons.append("saída contém verbo de prescrição ou alteração terapêutica")
    if not has_sources:
        reasons.append("nenhuma fonte institucional foi recuperada")
    if HUMAN_REVIEW_NOTICE not in text:
        reasons.append("aviso de validação humana ausente")
    return not reasons, tuple(reasons)


def safe_fallback(reasons: tuple[str, ...]) -> str:
    detail = "; ".join(reasons) if reasons else "contexto institucional insuficiente"
    return (
        f"Não foi possível produzir uma resposta clínica fundamentada ({detail}). "
        "A questão deve ser analisada diretamente pela equipe responsável. "
        f"{HUMAN_REVIEW_NOTICE}"
    )
