from clinical_assistant.safety import assess_input, validate_output, HUMAN_REVIEW_NOTICE


def test_flags_critical_combination():
    assessment = assess_input("Paciente com dor torácica e falta de ar.")
    assert assessment.critical


def test_flags_prescription_request():
    assessment = assess_input("Prescreva a dose do medicamento.")
    assert assessment.prescription_request


def test_rejects_unsafe_output():
    valid, reasons = validate_output(
        f"Tome o medicamento agora. {HUMAN_REVIEW_NOTICE}", has_sources=True
    )
    assert not valid
    assert reasons
