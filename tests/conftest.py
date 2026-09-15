import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def root():
    return ROOT

@pytest.fixture
def database(tmp_path):
    spec = importlib.util.spec_from_file_location("init_database", ROOT / "scripts/init_database.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "hospital.db"
    module.create_database(path)
    return path
