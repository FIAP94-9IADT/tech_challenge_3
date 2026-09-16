"""SQLite: busca textual, auditoria e revisão humana persistente."""
import json
import re
import sqlite3
from datetime import datetime, timezone

from medical.config import DATA, DB
from medical.safety import normalize


def connect(path=DB):
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize(path=DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        db.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(
          id UNINDEXED, question, answer, focus, source UNINDEXED, split UNINDEXED,
          tokenize='unicode61 remove_diacritics 2');
        CREATE TABLE IF NOT EXISTS audit (
          sequence INTEGER PRIMARY KEY, at TEXT NOT NULL, run_id TEXT NOT NULL,
          event TEXT NOT NULL, details TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reviews (
          run_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected')), draft TEXT NOT NULL,
          reviewer TEXT, at TEXT);
        CREATE INDEX IF NOT EXISTS audit_run_id ON audit(run_id, sequence);
        CREATE INDEX IF NOT EXISTS reviews_status ON reviews(status, at DESC);
        """)
    path.chmod(0o600)


def index_corpus(path=DB):
    corpus = DATA / "processed" / "corpus.jsonl"
    if not corpus.exists():
        raise ValueError("Execute python -m medical.dataset antes de indexar.")
    initialize(path)
    with connect(path) as db:
        db.execute("DELETE FROM documents")
        for line in corpus.read_text().splitlines():
            r = json.loads(line)
            # Holdouts não entram no RAG: avaliação não consulta a resposta de teste.
            if r["split"] == "train":
                db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                           tuple(r[k] for k in ("id", "question", "answer", "focus", "source", "split")))


def search(question, path=DB, limit=3):
    # Busca lexical com glossário limitado; embeddings multilíngues podem ampliar a recuperação.
    aliases = {"diabetes": "diabetes", "hipertensao": "hypertension", "asma": "asthma",
               "cancer": "cancer", "sintomas": "symptoms", "tratamento": "treatment",
               "alergia": "allergy", "diagnostico": "diagnosis", "prevencao": "prevention"}
    stops = set("como qual quais que para uma um por com dos das the what are is of and do does sobre".split())
    tokens = [w for w in re.findall(r"\b\w+\b", normalize(question)) if len(w) > 2 and w not in stops][:20]
    tokens = list(dict.fromkeys(aliases.get(w, w) for w in tokens))
    if not tokens:
        return []
    query = " OR ".join('"' + t + '"' for t in tokens)
    with connect(path) as db:
        rows = db.execute("SELECT *, bm25(documents, 0, 5, 1, 3) AS rank FROM documents WHERE documents MATCH ? ORDER BY rank LIMIT ?", (query, limit)).fetchall()
    return [dict(r) for r in rows]


def audit(run_id, event, details, path=DB):
    with connect(path) as db:
        db.execute("INSERT INTO audit(at, run_id, event, details) VALUES (?, ?, ?, ?)",
                   (datetime.now(timezone.utc).isoformat(), run_id, event, json.dumps(details, ensure_ascii=False)))


def save_draft(run_id, draft, path=DB):
    with connect(path) as db:
        db.execute("INSERT INTO reviews(run_id,status,draft) VALUES (?, 'pending', ?)", (run_id, draft))


def review(run_id, approved, reviewer, path=DB):
    if type(approved) is not bool or not re.fullmatch(r"[A-Za-z0-9_-]{3,40}", reviewer):
        raise ValueError("Informe identificador de revisor com 3 a 40 letras, números, _ ou -.")
    status = "approved" if approved else "rejected"
    with connect(path) as db:
        cursor = db.execute("UPDATE reviews SET status=?,reviewer=?,at=? WHERE run_id=? AND status='pending'",
                            (status, reviewer, datetime.now(timezone.utc).isoformat(), run_id))
        if cursor.rowcount != 1:
            raise ValueError("Revisão inexistente ou já concluída.")
        db.execute("INSERT INTO audit(at,run_id,event,details) VALUES (?,?,?,?)",
                   (datetime.now(timezone.utc).isoformat(), run_id, "human_review", json.dumps({"status": status, "reviewer": reviewer})))
    return status


def audit_events(run_id, path=DB):
    with connect(path) as db:
        rows = db.execute("SELECT at, event, details FROM audit WHERE run_id=? ORDER BY sequence", (run_id,)).fetchall()
    return [dict(row) for row in rows]


def pending_reviews(path=DB, limit=25):
    with connect(path) as db:
        rows = db.execute("SELECT run_id, draft FROM reviews WHERE status='pending' ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def database_status(path=DB):
    with connect(path) as db:
        documents = db.execute("SELECT count(*) FROM documents").fetchone()[0]
        pending = db.execute("SELECT count(*) FROM reviews WHERE status='pending'").fetchone()[0]
    return {"documents": documents, "pending_reviews": pending}


if __name__ == "__main__":
    index_corpus()
    print("SQLite inicializado; corpus MedQuAD de treino indexado.")
