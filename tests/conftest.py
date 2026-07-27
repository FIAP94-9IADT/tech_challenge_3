from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def prepared_assets():
    subprocess.run([sys.executable, "scripts/prepare_data.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/init_database.py"], cwd=ROOT, check=True)


@pytest.fixture()
def root() -> Path:
    return ROOT
