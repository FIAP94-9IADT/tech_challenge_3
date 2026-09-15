import pytest
from clinical_assistant.data_access import ClinicalRepository

def test_pending(database):
    patient = ClinicalRepository(database).get_patient_context("PAC-0001")
    assert {r["exam_name"] for r in patient.pending_exams} == {"creatinina", "eletrocardiograma"}

def test_invalid_id(database):
    with pytest.raises(ValueError):
        ClinicalRepository(database).get_patient_context("PAC-0001' OR 1=1 --")

def test_missing_patient(database):
    assert ClinicalRepository(database).get_patient_context("PAC-9999") is None
