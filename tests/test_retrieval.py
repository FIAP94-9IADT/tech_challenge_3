from clinical_assistant.retrieval import ProtocolRetriever


def test_retrieves_traceable_protocol(root):
    retriever = ProtocolRetriever(root / "data/raw/protocols")
    results = retriever.retrieve("dor torácica com dispneia e eletrocardiograma", k=1)
    assert results
    assert results[0].source_id == "PROTO-DOR-TORACICA"
    assert results[0].score > 0
    assert results[0].path == "data/raw/protocols/PROTO-DOR-TORACICA.md"
    assert (root / results[0].path).is_file()
