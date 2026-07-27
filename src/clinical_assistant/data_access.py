"""Acesso parametrizado e somente leitura aos registros sintéticos."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


PATIENT_ID_PATTERN = re.compile(r"^PAC-\d{4}$")


@dataclass(frozen=True)
class PatientContext:
    patient_id: str
    birth_year: int
    sex: str
    conditions: str
    allergies: str
    last_visit: str
    pending_exams: tuple[dict[str, str], ...]
    recent_results: tuple[dict[str, str], ...]

    def as_prompt_context(self) -> str:
        pending = "; ".join(
            f"{item['exam_name']} (solicitado em {item['requested_at']})"
            for item in self.pending_exams
        ) or "nenhum exame pendente registrado"
        results = "; ".join(
            f"{item['exam_name']}: {item['result']} ({item['resulted_at']})"
            for item in self.recent_results
        ) or "nenhum resultado recente registrado"
        return (
            f"Identificador institucional: {self.patient_id}\n"
            f"Ano de nascimento: {self.birth_year}; sexo registrado: {self.sex}\n"
            f"Condições registradas: {self.conditions}; alergias: {self.allergies}\n"
            f"Exames pendentes: {pending}\nResultados recentes: {results}"
        )


class ClinicalRepository:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).resolve()

    @staticmethod
    def validate_patient_id(patient_id: str) -> str:
        if not PATIENT_ID_PATTERN.fullmatch(patient_id):
            raise ValueError("Identificador institucional inválido; use PAC-0000.")
        return patient_id

    def _connect_read_only(self) -> sqlite3.Connection:
        if not self.database_path.exists():
            raise FileNotFoundError(
                f"Base não encontrada em {self.database_path}. Execute scripts/init_database.py."
            )
        uri = f"file:{self.database_path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def get_patient_context(self, patient_id: str) -> PatientContext | None:
        patient_id = self.validate_patient_id(patient_id)
        with closing(self._connect_read_only()) as connection:
            patient = connection.execute(
                """SELECT patient_id, birth_year, sex, conditions, allergies, last_visit
                   FROM patients WHERE patient_id = ?""",
                (patient_id,),
            ).fetchone()
            if patient is None:
                return None
            pending = connection.execute(
                """SELECT exam_name, requested_at FROM exams
                   WHERE patient_id = ? AND status = 'pending'
                   ORDER BY requested_at""",
                (patient_id,),
            ).fetchall()
            recent = connection.execute(
                """SELECT exam_name, resulted_at, result FROM exams
                   WHERE patient_id = ? AND status = 'completed'
                   ORDER BY resulted_at DESC LIMIT 5""",
                (patient_id,),
            ).fetchall()
        return PatientContext(
            patient_id=patient["patient_id"],
            birth_year=patient["birth_year"],
            sex=patient["sex"],
            conditions=patient["conditions"],
            allergies=patient["allergies"],
            last_visit=patient["last_visit"],
            pending_exams=tuple(dict(row) for row in pending),
            recent_results=tuple(dict(row) for row in recent),
        )
