"""Triagem textual conservadora e verificação de saídas; não é validação clínica."""
import re
import unicodedata
from dataclasses import dataclass

HUMAN_REVIEW_NOTICE = "Conteúdo para revisão do profissional responsável; nenhuma conduta foi autorizada."
CRITICAL_NOTICE = "ALERTA DE PRIORIZAÇÃO: sinais textuais requerem avaliação presencial pela equipe. O sistema não estabelece diagnóstico."
REFUSAL = "Solicitação fora do escopo: não serão fornecidos diagnóstico, prescrição, dose ou alteração terapêutica."
UNSAFE_OUTPUT = re.compile(r"\b(tome|administre|prescrevo|inicie|suspenda|diagn[oó]stico confirmado)\b|\d+\s*(mg|mcg|comprimidos)\b", re.I)

def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")

@dataclass(frozen=True)
class SafetyAssessment:
    critical: bool
    alerts: tuple[str, ...]
    prescription_request: bool

def assess_input(question):
    text = normalize(question)
    # Negações locais não ativam o sintoma; ambiguidades continuam limitadas por revisão humana.
    affirmed = re.sub(r"\b(?:sem|nega|nao apresenta|nao tem)\s+(?:dor toracica|dor no peito|falta de ar|dispneia|confusao|febre)", "", text)
    alerts = []
    if re.search(r"dor (toracica|no peito)", affirmed) and re.search(r"dispneia|falta de ar|sincope|sudorese|confusao", affirmed):
        alerts.append("dor torácica associada a outro sinal textual")
    if re.search(r"infecc|febre|sepse", affirmed) and re.search(r"hipotens|confusao|oliguria|falta de ar", affirmed):
        alerts.append("possível deterioração em contexto infeccioso")
    blocked = bool(re.search(r"prescrev|receit|\bdose\b|dosagem|suspend|ajust.*medica|confirm.*diagnost|qual (remedio|medicamento)", text))
    return SafetyAssessment(bool(alerts), tuple(alerts), blocked)

def validate_output(text, has_sources, evidence=None):
    reasons = []
    if not text.strip():
        reasons.append("saída vazia")
    if UNSAFE_OUTPUT.search(text):
        reasons.append("linguagem de intervenção detectada")
    if not has_sources:
        reasons.append("nenhuma fonte recuperada")
    # Modo conservador: somente trechos literais do contexto podem ser exibidos.
    if evidence is not None and text.strip() not in evidence:
        reasons.append("texto gerado não é um trecho literal das evidências")
    return not reasons, tuple(reasons)

def safe_fallback(reasons):
    return "Resposta retida para revisão: " + "; ".join(reasons) + ". " + HUMAN_REVIEW_NOTICE
