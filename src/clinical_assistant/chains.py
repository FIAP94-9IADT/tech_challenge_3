"""Pipeline LangChain para geração contextualizada."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from .llm import TextGenerator


PROMPT = PromptTemplate.from_template(
    """Você é um assistente institucional de apoio a profissionais de saúde.

Limites obrigatórios:
- não diagnostique, não prescreva e não determine alteração de tratamento;
- opções previstas no protocolo podem ser listadas para discussão profissional, sem recomendação individual;
- use somente o prontuário estruturado e os protocolos fornecidos;
- declare quando o contexto for insuficiente;
- apresente verificações e pendências, sempre com validação humana.

PERGUNTA ANONIMIZADA
{question}

CONTEXTO ESTRUTURADO DO PACIENTE
{patient_context}

PROTOCOLOS RECUPERADOS
{protocol_context}

Redija uma resposta curta, clara e rastreável. Não invente resultados ou fontes."""
)


def build_clinical_chain(generator: TextGenerator):
    """Compõe prompt e modelo no padrão de encadeamento do LangChain."""
    invoke_generator = RunnableLambda(lambda value: generator.invoke(value.to_string()))
    return PROMPT | invoke_generator
