"""Cria a base SQLite sintética usada nas consultas estruturadas."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "hospital.db"

SCHEMA = """
CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY CHECK (patient_id GLOB 'PAC-[0-9][0-9][0-9][0-9]'),
    birth_year INTEGER NOT NULL,
    sex TEXT NOT NULL,
    conditions TEXT NOT NULL,
    allergies TEXT NOT NULL,
    last_visit TEXT NOT NULL
);
CREATE TABLE exams (
    exam_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    exam_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'completed')),
    requested_at TEXT NOT NULL,
    resulted_at TEXT,
    result TEXT
);
CREATE INDEX idx_exams_patient_status ON exams(patient_id, status);
"""


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    patients = rows(RAW_DIR / "patients.csv")
    exams = rows(RAW_DIR / "exams.csv")
    with sqlite3.connect(OUTPUT_PATH) as connection:
        connection.executescript(SCHEMA)
        connection.executemany(
            """INSERT INTO patients
               (patient_id, birth_year, sex, conditions, allergies, last_visit)
               VALUES (:patient_id, :birth_year, :sex, :conditions, :allergies, :last_visit)""",
            patients,
        )
        connection.executemany(
            """INSERT INTO exams
               (exam_id, patient_id, exam_name, status, requested_at, resulted_at, result)
               VALUES (:exam_id, :patient_id, :exam_name, :status, :requested_at, :resulted_at, :result)""",
            exams,
        )
    print(f"Base sintética criada em {OUTPUT_PATH} ({len(patients)} pacientes, {len(exams)} exames).")


if __name__ == "__main__":
    main()
