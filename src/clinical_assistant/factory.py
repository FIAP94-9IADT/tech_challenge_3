"""Configuração da aplicação com caminhos relativos à raiz do projeto."""
import os
from pathlib import Path
from dotenv import load_dotenv
from .audit import AuditLogger
from .data_access import ClinicalRepository
from .graph import ClinicalAssistantGraph
from .llm import DemoClinicalGenerator, T5Generator
from .retrieval import ProtocolRetriever

def build_application():
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    backend = os.getenv("ASSISTANT_BACKEND", "demo")
    if backend == "t5":
        generator = T5Generator(root / os.getenv("ADAPTER_PATH", "models/clinical-t5-lora"))
    elif backend == "demo":
        generator = DemoClinicalGenerator()
    else:
        raise ValueError("ASSISTANT_BACKEND: demo ou t5")
    return ClinicalAssistantGraph(
        ClinicalRepository(root / os.getenv("DATABASE_PATH", "data/processed/hospital.db")),
        ProtocolRetriever(root / "data/raw/protocols"), generator,
        AuditLogger(root / os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")))
