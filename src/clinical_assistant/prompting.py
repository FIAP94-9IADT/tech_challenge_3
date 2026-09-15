"""Contrato textual compartilhado por treinamento e inferência."""
SYSTEM = (
    "Responda em português usando apenas as evidências fornecidas. "
    "Não confirme diagnóstico, não prescreva e não recomende alteração de tratamento. "
    "Quando faltarem evidências, declare a limitação."
)

def format_prompt(question: str, context: str) -> str:
    return f"{SYSTEM}\nPergunta: {question}\nEvidências: {context}\nResposta:"
