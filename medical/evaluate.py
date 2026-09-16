"""Avalia base vs. fine-tuned no mesmo holdout, sem RAG nem juiz LLM pago."""
import argparse
from collections import Counter
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from medical.assistant import make_llm
from medical.config import DATA, ROOT, model_name
from medical.finetune import BASE
from medical.dataset import TRAIN_SYSTEM
from medical.safety import normalize


def token_f1(prediction, reference):
    def tokens(text):
        return Counter(re.findall(r"\w+", normalize(re.sub(r"\[[^]]+\]", "", text))))
    a, b = tokens(prediction), tokens(reference)
    common = sum((a & b).values())
    return 2 * common / (sum(a.values()) + sum(b.values())) if a and b else 0.0


def evaluate(limit=5, custom=False):
    if not 1 <= limit <= 20:
        raise ValueError("Use entre 1 e 20 exemplos para limitar custo.")
    rows = json.loads((DATA / "processed" / "test_rows.json").read_text())[:limit]
    models = [BASE] + ([model_name(True)] if custom else [])
    if any(m.startswith("local:") for m in models):
        raise ValueError("Use docs/local_training.json para comparar loss antes/depois do LoRA local.")
    report = {"method": "holdout por tópico, sem RAG; F1 lexical, não acurácia clínica", "results": []}
    for model in models:
        llm = make_llm(model)
        items = []
        for row in rows:
            response = llm.invoke([SystemMessage(content=TRAIN_SYSTEM), HumanMessage(content=row["question"] + f'\nID da fonte: {row["id"]}')])
            answer = response.content
            items.append({"id": row["id"], "question": row["question"], "reference": row["answer"],
                          "answer": answer, "token_f1": token_f1(answer, row["answer"]),
                          "usage": response.usage_metadata, "finish_reason": response.response_metadata.get("finish_reason")})
        report["results"].append({"model": model, "n": len(items),
                                  "mean_token_f1": sum(r["token_f1"] for r in items)/len(items), "items": items})
        # Salva o progresso para não perder a avaliação da base se a segunda chamada falhar.
        (ROOT / "docs" / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"results": [{k:v for k,v in r.items() if k != "items"} for r in report["results"]]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    try:
        evaluate(args.limit, args.compare)
    except Exception as exc:
        print(f"Avaliação interrompida ({type(exc).__name__}); resultados anteriores preservados.")
        raise SystemExit(1)
