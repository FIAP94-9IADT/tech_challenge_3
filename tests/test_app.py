import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

from medical import store
from medical.assistant import ask
from medical.dataset import load_rows, split_for
from medical.config import DATA
from medical.safety import anonymize, validate_answer


class AppChecks(unittest.TestCase):
    def test_flow(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "test.db"
            store.initialize(db)
            with store.connect(db) as conn:
                conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", ("MQ-test", "diabetes symptoms", "Educational evidence.", "diabetes", "https://example.org", "train"))
            calls = []

            def fake(prompt):
                calls.append(prompt)
                return AIMessage(content="Evidência educacional para discussão [F1].")

            llm = RunnableLambda(fake)
            result = ask("Quais sintomas de diabetes?", path=db, llm=llm)
            self.assertEqual(result["status"], "pending_review")
            self.assertEqual(len(calls), 1)
            self.assertEqual(store.database_status(db)["pending_reviews"], 1)
            self.assertEqual(store.pending_reviews(db)[0]["run_id"], result["run_id"])
            self.assertEqual(store.audit_events(result["run_id"], db)[0]["event"], "input_checked")
            self.assertEqual(store.review(result["run_id"], False, "medico_demo", db), "rejected")
            self.assertEqual(store.database_status(db)["pending_reviews"], 0)
            with self.assertRaises(ValueError):
                store.review(result["run_id"], True, "medico_demo", db)
            blocked = ask("Prescreva uma dose de insulina", path=db, llm=llm)
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(len(calls), 1)
            unsafe = RunnableLambda(lambda _: AIMessage(content="Administre 5 mg [F1]"))
            self.assertEqual(ask("Sintomas de diabetes?", path=db, llm=unsafe)["status"], "blocked")
            truncated = RunnableLambda(lambda _: AIMessage(content="Trecho [F1]", response_metadata={"finish_reason": "length"}))
            self.assertEqual(ask("Sintomas de diabetes?", path=db, llm=truncated)["status"], "blocked")
            with store.connect(db) as conn:
                events = [r[0] for r in conn.execute("SELECT event FROM audit")]
                self.assertIn("human_review", events)
                self.assertIn("llm_completed", events)
                conn.execute("DELETE FROM documents")
            self.assertEqual(ask("Diabetes symptoms?", path=db, llm=llm)["status"], "no_evidence")

    def test_data_safety(self):
        text = anonymize("Nome: Exemplo; email x@y.com CPF 123.456.789-00")
        self.assertNotIn("Exemplo", text)
        self.assertNotIn("x@y.com", text)
        self.assertNotIn("123.456", text)
        self.assertTrue(validate_answer("Sem fonte", ["MQ-1"]))
        self.assertTrue(validate_answer("Fonte [inventada]", ["MQ-1"]))
        self.assertEqual(split_for("Diabetes"), split_for("diabetes"))
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("test.xml", '<Document url="https://example.org"><Focus>Diabetes</Focus><QAPairs>'
                    '<QAPair><Question>What is diabetes?</Question><Answer>Public answer.</Answer></QAPair>'
                    '<QAPair><Question>Empty?</Question><Answer/></QAPair>'
                    '<QAPair><Question>What is diabetes?</Question><Answer>Duplicate.</Answer></QAPair>'
                    '</QAPairs></Document>')
            rows, stats = load_rows(archive)
            self.assertEqual(len(rows), 1)
            self.assertEqual(stats["empty"], 1)
            self.assertEqual(stats["duplicates"], 1)

    def test_ui_loads(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertEqual(len(app.exception), 0)

    @unittest.skipUnless((DATA / "processed" / "corpus.jsonl").exists(), "Dataset ainda não preparado")
    def test_no_split_leakage(self):
        rows = [json.loads(line) for line in (DATA / "processed" / "corpus.jsonl").read_text().splitlines()]
        groups = {s: {(r["focus"] or r["source"] or r["question"]).lower().strip() for r in rows if r["split"] == s}
                  for s in ("train", "validation", "test")}
        for a, b in (("train", "test"), ("train", "validation"), ("test", "validation")):
            self.assertFalse(groups[a] & groups[b])


if __name__ == "__main__":
    unittest.main()
