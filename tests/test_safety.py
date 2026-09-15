from clinical_assistant.safety import assess_input, validate_output

def test_alert_order():
    assert assess_input("Falta de ar associada a dor torácica").critical
    assert assess_input("Dor torácica e falta de ar").critical

def test_negated_symptoms():
    assert not assess_input("Sem dor torácica e sem falta de ar").critical

def test_dose_request():
    assert assess_input("Prescreva a dose").prescription_request

def test_nonliteral_output_rejected():
    valid, _ = validate_output("Diagnóstico: condição inventada", True, "Exame pendente")
    assert not valid

def test_exact_evidence_passes():
    assert validate_output("Exame pendente", True, "Exame pendente")[0]

def test_numeric_intervention_rejected():
    assert not validate_output("Medicamento 100 mg", True, "Medicamento 100 mg")[0]
