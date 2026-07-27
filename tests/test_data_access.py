import pytest

from clinical_assistant.data_access import ClinicalRepository


def test_reads_pending_exams(root):
    repository = ClinicalRepository(root / "data/processed/hospital.db")
    context = repository.get_patient_context("PAC-0001")
    assert context is not None
    assert {item["exam_name"] for item in context.pending_exams} == {"creatinina", "eletrocardiograma"}


def test_rejects_invalid_identifier(root):
    repository = ClinicalRepository(root / "data/processed/hospital.db")
    with pytest.raises(ValueError):
        repository.get_patient_context("PAC-0001' OR 1=1 --")
