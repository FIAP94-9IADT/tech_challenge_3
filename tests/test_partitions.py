import importlib.util

def test_partitions_and_prescription_coverage(root, tmp_path):
    spec = importlib.util.spec_from_file_location("prepare", root / "scripts/prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.prepare(tmp_path)
    sets = {name: set(values["ids"]) for name, values in report["partitions"].items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert "REC-007" in sets["train"]
    assert all(len(v["categories"]) == 5 for v in report["partitions"].values())
