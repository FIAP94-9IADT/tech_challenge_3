"""Contrato textual compartilhado por treinamento e inferência."""
SYSTEM = (
    "Responda em português usando apenas as evidências. "
    "Não confirme diagnóstico nem prescreva. Se faltar evidência, declare insuficiência."
)

def format_prompt(question: str, context: str) -> str:
    return f"{SYSTEM}\nPergunta: {question}\nEvidências: {context}\nResposta:"
