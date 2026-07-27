import json

from clinical_assistant.audit import AuditLogger
from clinical_assistant.data_access import ClinicalRepository
from clinical_assistant.graph import ClinicalAssistantGraph
from clinical_assistant.llm import DemoClinicalGenerator
from clinical_assistant.retrieval import ProtocolRetriever


def build_graph(root, log_path):
    return ClinicalAssistantGraph(
        ClinicalRepository(root / "data/processed/hospital.db"),
        ProtocolRetriever(root / "data/raw/protocols"),
        DemoClinicalGenerator(),
        AuditLogger(log_path),
    )


def test_end_to_end_with_sources_and_audit(root, tmp_path):
    log_path = tmp_path / "audit.jsonl"
    app = build_graph(root, log_path)
    result = app.invoke(
        "Há dor torácica e falta de ar. Quais exames estão pendentes?",
        "PAC-0001",
    )
    assert result["critical"] is True
    assert result["output_valid"] is True
    assert "ALERTA DE PRIORIZAÇÃO" in result["answer"]
    assert "Exames pendentes registrados" in result["answer"]
    assert "Não posso prescrever" not in result["answer"]
    assert "PROTO-DOR-TORACICA" in result["answer"]
    assert result["sources"][0]["source_id"] == "PROTO-DOR-TORACICA"
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert "question" not in record
    assert len(record["request_hash"]) == 64


def test_refuses_direct_prescription(root, tmp_path):
    app = build_graph(root, tmp_path / "audit.jsonl")
    result = app.invoke("Prescreva a melhor dose para hipertensão.", "PAC-0002")
    assert "Não posso prescrever" in result["answer"]
    assert result["prescription_request"] is True


def test_lists_protocol_options_without_prescribing(root, tmp_path):
    app = build_graph(root, tmp_path / "audit.jsonl")
    result = app.invoke(
        "Quais opções do protocolo podem ser discutidas no acompanhamento da hipertensão?",
        "PAC-0002",
    )
    assert "Opções registradas para discussão profissional" in result["answer"]
    assert "não corresponde a recomendação individual" in result["answer"]
    assert result["sources"][0]["source_id"] == "PROTO-HIPERTENSAO"
