"""Recuperação TF-IDF de protocolos com metadados de fonte."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ0-9-]{3,}")
STOPWORDS = {
    "para", "com", "sem", "uma", "das", "dos", "que", "por", "deve",
    "pode", "não", "nos", "nas", "como", "pela", "pelo", "este", "esta",
}


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    ]


@dataclass(frozen=True)
class RetrievedProtocol:
    source_id: str
    title: str
    excerpt: str
    score: float
    path: str


class ProtocolRetriever:
    """Índice local pequeno, adequado aos protocolos sintéticos do protótipo."""

    def __init__(self, protocols_path: str | Path):
        self.protocols_path = Path(protocols_path)
        self._documents = self._load_documents()
        self._document_frequency = self._build_document_frequency()

    def _load_documents(self) -> tuple[dict[str, object], ...]:
        documents = []
        for path in sorted(self.protocols_path.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            first_line = next((line for line in text.splitlines() if line.startswith("# ")), path.stem)
            title = first_line.removeprefix("# ").strip()
            documents.append({"source_id": path.stem, "title": title, "text": text, "path": str(path)})
        if not documents:
            raise FileNotFoundError(f"Nenhum protocolo .md encontrado em {self.protocols_path}")
        return tuple(documents)

    def _build_document_frequency(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for document in self._documents:
            frequencies.update(set(_tokens(str(document["text"]))))
        return frequencies

    def _vector(self, text: str) -> dict[str, float]:
        counts = Counter(_tokens(text))
        total = sum(counts.values()) or 1
        count_documents = len(self._documents)
        return {
            token: (count / total) * (math.log((1 + count_documents) / (1 + self._document_frequency[token])) + 1)
            for token, count in counts.items()
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        common = left.keys() & right.keys()
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @staticmethod
    def _best_excerpt(text: str, query_tokens: set[str], max_chars: int = 700) -> str:
        sections = [section.strip() for section in re.split(r"\n(?=## )", text) if section.strip()]
        best = max(sections, key=lambda section: len(set(_tokens(section)) & query_tokens))
        return best[:max_chars].rstrip()

    def retrieve(self, query: str, k: int = 2, minimum_score: float = 0.01) -> tuple[RetrievedProtocol, ...]:
        query_vector = self._vector(query)
        query_tokens = set(_tokens(query))
        ranked = []
        for document in self._documents:
            score = self._cosine(query_vector, self._vector(str(document["text"])))
            if score >= minimum_score:
                ranked.append(
                    RetrievedProtocol(
                        source_id=str(document["source_id"]),
                        title=str(document["title"]),
                        excerpt=self._best_excerpt(str(document["text"]), query_tokens),
                        score=round(score, 4),
                        path=str(document["path"]),
                    )
                )
        return tuple(sorted(ranked, key=lambda item: item.score, reverse=True)[:k])
