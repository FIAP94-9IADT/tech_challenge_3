"""Construção da aplicação a partir de variáveis de ambiente."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .audit import AuditLogger
from .data_access import ClinicalRepository
from .graph import ClinicalAssistantGraph
from .llm import DemoClinicalGenerator, build_huggingface_generator
from .retrieval import ProtocolRetriever


def build_application() -> ClinicalAssistantGraph:
    load_dotenv()
    backend = os.getenv("ASSISTANT_BACKEND", "demo").lower()
    if backend == "huggingface":
        generator = build_huggingface_generator(
            os.getenv("BASE_MODEL_ID", "meta-llama/Llama-2-7b-hf"),
            os.getenv("ADAPTER_PATH", "models/clinical-lora"),
        )
    elif backend == "demo":
        generator = DemoClinicalGenerator()
    else:
        raise ValueError("ASSISTANT_BACKEND deve ser 'demo' ou 'huggingface'.")
    return ClinicalAssistantGraph(
        repository=ClinicalRepository(os.getenv("DATABASE_PATH", "data/processed/hospital.db")),
        retriever=ProtocolRetriever(os.getenv("PROTOCOLS_PATH", "data/raw/protocols")),
        generator=generator,
        audit_logger=AuditLogger(os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")),
    )
