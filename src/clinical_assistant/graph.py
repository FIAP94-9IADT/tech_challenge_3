"""Orquestração segura do atendimento com StateGraph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .anonymization import anonymize_text
from .audit import AuditLogger
from .chains import build_clinical_chain
from .data_access import ClinicalRepository
from .retrieval import ProtocolRetriever
from .safety import assess_input, safe_fallback, validate_output


class AssistantState(TypedDict, total=False):
    question: str
    anonymized_question: str
    patient_id: str
    redactions: dict[str, int]
    patient_context: str
    pending_exams: list[dict[str, str]]
    sources: list[dict[str, Any]]
    critical: bool
    alerts: list[str]
    prescription_request: bool
    draft: str
    answer: str
    output_valid: bool
    validation_reasons: list[str]
    route: str
    steps: list[str]


class ClinicalAssistantGraph:
    def __init__(self, repository, retriever, generator, audit_logger):
        self.repository: ClinicalRepository = repository
        self.retriever: ProtocolRetriever = retriever
        self.chain = build_clinical_chain(generator)
        self.audit_logger: AuditLogger = audit_logger
        self.graph = self._build()

    @staticmethod
    def _append_step(state: AssistantState, step: str) -> list[str]:
        return [*state.get("steps", []), step]

    def sanitize(self, state: AssistantState) -> dict[str, Any]:
        result = anonymize_text(state["question"])
        return {
            "anonymized_question": result.text,
            "redactions": result.redactions,
            "steps": self._append_step(state, "entrada anonimizada"),
        }

    def load_patient(self, state: AssistantState) -> dict[str, Any]:
        context = self.repository.get_patient_context(state["patient_id"])
        if context is None:
            patient_text = "identificador não encontrado na base estruturada"
            pending: list[dict[str, str]] = []
        else:
            patient_text = context.as_prompt_context()
            pending = list(context.pending_exams)
        return {
            "patient_context": patient_text,
            "pending_exams": pending,
            "steps": self._append_step(state, "prontuário estruturado consultado"),
        }

    def detect_risk(self, state: AssistantState) -> dict[str, Any]:
        assessment = assess_input(state["anonymized_question"])
        return {
            "critical": assessment.critical,
            "alerts": list(assessment.alerts),
            "prescription_request": assessment.prescription_request,
            "steps": self._append_step(state, "limites e sinais de alerta avaliados"),
        }

    def retrieve_protocols(self, state: AssistantState) -> dict[str, Any]:
        query = " ".join(
            [
                state["anonymized_question"],
                state["anonymized_question"],
                state["anonymized_question"],
                state.get("patient_context", ""),
            ]
        )
        retrieved = self.retriever.retrieve(query, k=2)
        sources = [
            {
                "source_id": item.source_id,
                "title": item.title,
                "excerpt": item.excerpt,
                "score": item.score,
                "path": item.path,
            }
            for item in retrieved
        ]
        return {
            "sources": sources,
            "steps": self._append_step(state, f"{len(sources)} protocolo(s) recuperado(s)"),
        }

    def generate(self, state: AssistantState) -> dict[str, Any]:
        if state.get("prescription_request"):
            question = (
                state["anonymized_question"]
                + "\nA solicitação envolve prescrição; recuse essa ação e ofereça apenas organização de dados."
            )
        else:
            question = state["anonymized_question"]
        protocol_context = "\n\n".join(
            f"[{source['source_id']}] {source['excerpt']}" for source in state.get("sources", [])
        ) or "Nenhum protocolo relevante localizado."
        draft = self.chain.invoke(
            {
                "question": question,
                "patient_context": state.get("patient_context", "não informado"),
                "protocol_context": protocol_context,
            }
        )
        return {"draft": str(draft), "steps": self._append_step(state, "resposta preliminar gerada")}

    def validate(self, state: AssistantState) -> dict[str, Any]:
        valid, reasons = validate_output(state["draft"], bool(state.get("sources")))
        return {
            "output_valid": valid,
            "validation_reasons": list(reasons),
            "steps": self._append_step(state, "resposta preliminar validada"),
        }

    @staticmethod
    def validation_route(state: AssistantState) -> Literal["finalize", "fallback"]:
        return "finalize" if state.get("output_valid") else "fallback"

    def finalize(self, state: AssistantState) -> dict[str, Any]:
        alert = ""
        if state.get("critical"):
            alert = (
                "ALERTA DE PRIORIZAÇÃO: há sinais informados que justificam avaliação presencial imediata. "
                "Não aguarde a resposta do sistema para acionar a equipe.\n\n"
            )
        citations = "\n".join(
            f"- {source['source_id']}: {source['title']} (relevância {source['score']:.3f})"
            for source in state.get("sources", [])
        )
        answer = f"{alert}{state['draft']}\n\nFontes institucionais:\n{citations}"
        return {
            "answer": answer,
            "route": "resposta_validada",
            "steps": self._append_step(state, "resposta final montada com fontes"),
        }

    def fallback(self, state: AssistantState) -> dict[str, Any]:
        return {
            "answer": safe_fallback(tuple(state.get("validation_reasons", []))),
            "route": "bloqueio_de_seguranca",
            "steps": self._append_step(state, "saída substituída por resposta segura"),
        }

    def audit(self, state: AssistantState) -> dict[str, Any]:
        self.audit_logger.write(dict(state))
        return {"steps": self._append_step(state, "evento registrado para auditoria")}

    def _build(self):
        workflow = StateGraph(AssistantState)
        workflow.add_node("sanitize", self.sanitize)
        workflow.add_node("load_patient", self.load_patient)
        workflow.add_node("detect_risk", self.detect_risk)
        workflow.add_node("retrieve_protocols", self.retrieve_protocols)
        workflow.add_node("generate", self.generate)
        workflow.add_node("validate", self.validate)
        workflow.add_node("finalize", self.finalize)
        workflow.add_node("fallback", self.fallback)
        workflow.add_node("audit", self.audit)
        workflow.add_edge(START, "sanitize")
        workflow.add_edge("sanitize", "load_patient")
        workflow.add_edge("load_patient", "detect_risk")
        workflow.add_edge("detect_risk", "retrieve_protocols")
        workflow.add_edge("retrieve_protocols", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_conditional_edges(
            "validate",
            self.validation_route,
            {"finalize": "finalize", "fallback": "fallback"},
        )
        workflow.add_edge("finalize", "audit")
        workflow.add_edge("fallback", "audit")
        workflow.add_edge("audit", END)
        return workflow.compile()

    def invoke(self, question: str, patient_id: str) -> AssistantState:
        return self.graph.invoke(
            {"question": question, "patient_id": patient_id, "steps": []}
        )
