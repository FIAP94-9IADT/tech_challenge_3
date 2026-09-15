from clinical_assistant.anonymization import anonymize_text, contains_direct_identifier


def test_removes_direct_identifiers():
    original = "Nome: Maria de Souza, CPF 123.456.789-00, e-mail maria@example.com."
    result = anonymize_text(original)
    assert result.changed
    assert "Maria de Souza" not in result.text
    assert "123.456.789-00" not in result.text
    assert "maria@example.com" not in result.text
    assert not contains_direct_identifier(result.text)


def test_preserves_institutional_synthetic_id():
    result = anonymize_text("Consultar o paciente PAC-0001.")
    assert "PAC-0001" in result.text
