"""Anonimização determinística de texto antes do uso por modelos."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnonymizationResult:
    text: str
    redactions: dict[str, int]

    @property
    def changed(self) -> bool:
        return any(self.redactions.values())


PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("cpf", re.compile(r"\b\d{3}\.?(?:\d{3})\.?(?:\d{3})-?\d{2}\b"), "[CPF_REMOVIDO]"),
    ("email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REMOVIDO]"),
    ("phone", re.compile(r"(?<!\d)(?:\+?55\s*)?\(?\d{2}\)?\s*9?\d{4}[-\s]?\d{4}(?!\d)"), "[TELEFONE_REMOVIDO]"),
    ("record", re.compile(r"\b(?:prontu[aá]rio|registro)\s*[:#-]?\s*\d{4,}\b", re.I), "[PRONTUARIO_REMOVIDO]"),
    (
        "name_label",
        re.compile(r"\b(?:nome|paciente)\s*:\s*[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç]+(?:\s+(?:de|da|do|das|dos|e|[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç]+)){1,5}", re.I),
        "nome: [NOME_REMOVIDO]",
    ),
    (
        "name_before_cpf",
        re.compile(r"\b[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+(?:de|da|do|das|dos|e|[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+)){1,5}(?=,?\s*CPF\b)"),
        "[NOME_REMOVIDO]",
    ),
)


def anonymize_text(text: str) -> AnonymizationResult:
    """Remove identificadores diretos e informa quantas substituições ocorreram."""
    cleaned = text.strip()
    counts: dict[str, int] = {}
    for label, pattern, replacement in PATTERNS:
        cleaned, count = pattern.subn(replacement, cleaned)
        counts[label] = count
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return AnonymizationResult(text=cleaned, redactions=counts)


def contains_direct_identifier(text: str) -> bool:
    """Retorna verdadeiro quando um padrão identificador ainda é detectado."""
    return any(pattern.search(text) for _, pattern, _ in PATTERNS[:4])
