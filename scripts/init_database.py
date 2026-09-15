"""Inicializa SQLite a partir de CSV; preserva bases existentes."""
import csv
import sqlite3
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def create_database(path):
    path = Path(path)
    resolved = path.resolve()
    display_path = resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else path.name
    if path.exists():
        return {"created": False, "path": display_path}
    patients = list(csv.DictReader((ROOT / "data/raw/patients.csv").open(encoding="utf-8")))
    exams = list(csv.DictReader((ROOT / "data/raw/exams.csv").open(encoding="utf-8")))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY, birth_year INTEGER NOT NULL, sex TEXT,
            conditions TEXT, allergies TEXT, last_visit TEXT);
        CREATE TABLE exams (
            exam_id TEXT PRIMARY KEY, patient_id TEXT REFERENCES patients(patient_id),
            exam_name TEXT, status TEXT CHECK(status IN ('pending','completed')),
            requested_at TEXT, resulted_at TEXT, result TEXT);
        """)
        connection.executemany("INSERT INTO patients VALUES (:patient_id,:birth_year,:sex,:conditions,:allergies,:last_visit)", patients)
        connection.executemany("INSERT INTO exams VALUES (:exam_id,:patient_id,:exam_name,:status,:requested_at,:resulted_at,:result)", exams)
        connection.commit()
    finally:
        connection.close()
    return {"created": True, "path": display_path, "patients": len(patients), "exams": len(exams)}

if __name__ == "__main__":
    print(create_database(ROOT / "data/processed/hospital.db"))
