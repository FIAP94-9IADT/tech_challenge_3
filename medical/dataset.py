"""Download, curadoria e separação por tópico, sem chamadas de LLM."""
import argparse
from collections import Counter
import hashlib
import html
import json
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from medical.config import DATA
from medical.safety import anonymize

URL = "https://codeload.github.com/abachaa/MedQuAD/zip/refs/heads/master"
TRAIN_SYSTEM = "Você é um assistente médico educacional. Responda à pergunta no idioma dela, citando o ID fornecido. Não prescreva nem invente protocolos internos. A resposta exige revisão médica."


def clean(text):
    return anonymize(" ".join(re.sub(r"<[^>]+>", " ", html.unescape(text or "")).split()))


def split_for(topic):
    bucket = int(hashlib.sha256(topic.lower().strip().encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def load_rows(archive):
    seen = set()
    stats = Counter()
    rows = []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".xml"):
                continue
            stats["xml"] += 1
            root = ET.fromstring(z.read(name))
            focus = clean(root.findtext("Focus"))
            url = root.attrib.get("url", "")
            for qa in root.findall(".//QAPair"):
                stats["raw_pairs"] += 1
                q, a = clean(qa.findtext("Question")), clean(qa.findtext("Answer"))
                if not q or not a:
                    stats["empty"] += 1
                    continue
                key = q.casefold()
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                # Mantém a resposta integral para consulta; amostra de treino filtra textos longos.
                rows.append({"id": "MQ-" + hashlib.sha256(key.encode()).hexdigest()[:12],
                             "question": q, "answer": a, "focus": focus,
                             "source": url, "file": name, "kind": "medquad",
                             "split": split_for(focus or url or q)})
    if not rows:
        raise ValueError("Arquivo não contém pares MedQuAD com respostas.")
    stats["retained"] = len(rows)
    return rows, dict(stats)


def training_example(row):
    return {"messages": [{"role": "system", "content": TRAIN_SYSTEM},
                         {"role": "user", "content": row["question"] + f'\nID da fonte: {row["id"]}'},
                         {"role": "assistant", "content": row["answer"] + f' [{row["id"]}]'}]}


def prepare(limit=100):
    if not 10 <= limit <= 1000:
        raise ValueError("Use entre 10 e 1000 exemplos de treino.")
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "medquad.zip"
    if not archive.exists():
        temp = archive.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=90) as response, temp.open("wb") as out:
            import shutil
            shutil.copyfileobj(response, out)
        temp.replace(archive)
    rows, stats = load_rows(archive)
    output = DATA / "processed"
    output.mkdir(exist_ok=True)
    (output / "corpus.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    rng = random.Random(42)
    samples = {}
    for split, n in (("train", limit), ("validation", 20), ("test", 20)):
        candidates = [r for r in rows if r["split"] == split and 80 <= len(r["answer"]) <= 1800]
        rng.shuffle(candidates)
        sample = candidates[:n]
        samples[split] = len(sample)
        (output / f"{split}.jsonl").write_text("".join(json.dumps(training_example(r), ensure_ascii=False) + "\n" for r in sample))
        (output / f"{split}_rows.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive) as z:
        for name in ("LICENSE.txt", "readme.txt"):
            (output / f"MedQuAD-{name}").write_bytes(z.read("MedQuAD-master/" + name))
    stats.update(samples=samples, split_counts=dict(Counter(r["split"] for r in rows)),
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), seed=42,
                 source=URL)
    (output / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    prepare(args.limit)
