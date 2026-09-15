import json
import pytest
from clinical_assistant.audit import AuditLogger
from clinical_assistant.data_access import ClinicalRepository
from clinical_assistant.graph import ClinicalAssistantGraph
from clinical_assistant.llm import DemoClinicalGenerator
from clinical_assistant.retrieval import ProtocolRetriever

class ExplodingGenerator:
    def invoke(self, prompt):
        raise RuntimeError("erro simulado")

def app(root, database, tmp_path, generator=None):
    return ClinicalAssistantGraph(ClinicalRepository(database),
        ProtocolRetriever(root / "data/raw/protocols"), generator or DemoClinicalGenerator(),
        AuditLogger(tmp_path / "audit.jsonl"))

def test_critical_bypasses_model(root, database, tmp_path):
    result = app(root,database,tmp_path,ExplodingGenerator()).invoke("Dor torácica e falta de ar","PAC-0001")
    assert result["route"] == "alerta_prioritario"
    assert "ALERTA" in result["answer"]

def test_refusal_bypasses_model(root, database, tmp_path):
    result = app(root,database,tmp_path,ExplodingGenerator()).invoke("Prescreva a dose","PAC-0001")
    assert result["route"] == "recusa"

def test_evidence_and_audit(root, database, tmp_path):
    result = app(root,database,tmp_path).invoke("Quais exames verificar no diabetes?", "PAC-0001")
    assert "creatinina" in result["answer"]
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert "question" not in record
    assert record["request_id"]
    assert record["sources"][0]["version"]

def test_failure_audited(root, database, tmp_path):
    with pytest.raises(RuntimeError):
        app(root,database,tmp_path,ExplodingGenerator()).invoke("Exames de diabetes", "PAC-0001")
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert record["error_type"] == "RuntimeError"
    assert record["route"] == "erro_operacional"

def test_missing_patient_blocks(root, database, tmp_path):
    result = app(root,database,tmp_path,ExplodingGenerator()).invoke("Exames de diabetes", "PAC-9999")
    assert result["route"] == "bloqueio_de_seguranca"

def test_out_of_scope_blocks(root, database, tmp_path):
    result = app(root,database,tmp_path,ExplodingGenerator()).invoke("Astronomia galáxias planetas", "PAC-0001")
    assert result["route"] == "bloqueio_de_seguranca"
