"""Fluxo com triagem anterior ao modelo e registro de falhas."""
import time
import uuid
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from .anonymization import anonymize_text
from .chains import build_clinical_chain
from .safety import assess_input, validate_output, safe_fallback, HUMAN_REVIEW_NOTICE, CRITICAL_NOTICE, REFUSAL

class AssistantState(TypedDict, total=False):
    question: str
    anonymized_question: str
    patient_id: str
    patient_context: str
    pending_exams: list
    sources: list
    alerts: list
    critical: bool
    prescription_request: bool
    redactions: dict
    draft: str
    answer: str
    output_valid: bool
    validation_reasons: list
    route: str
    steps: list
    request_id: str
    backend: str

class ClinicalAssistantGraph:
    def __init__(self, repository, retriever, generator, audit_logger):
        self.repository, self.retriever = repository, retriever
        self.generator, self.audit_logger = generator, audit_logger
        self.chain = build_clinical_chain(generator)
        self.graph = self._build()

    def sanitize(self, state):
        clean = anonymize_text(state["question"])
        risk = assess_input(clean.text)
        return dict(anonymized_question=clean.text, redactions=clean.redactions,
            critical=risk.critical, alerts=list(risk.alerts), prescription_request=risk.prescription_request,
            steps=["entrada tratada", "triagem textual executada"])

    def route_input(self, state):
        if state["critical"]: return "alert"
        if state["prescription_request"]: return "refuse"
        return "load_patient"

    def alert(self, state):
        return dict(answer=CRITICAL_NOTICE + " " + HUMAN_REVIEW_NOTICE, route="alerta_prioritario",
            output_valid=False, steps=state["steps"] + ["alerta emitido sem chamar o modelo"])

    def refuse(self, state):
        return dict(answer=REFUSAL + " " + HUMAN_REVIEW_NOTICE, route="recusa",
            output_valid=False, steps=state["steps"] + ["solicitação recusada antes da geração"])

    def load_patient(self, state):
        patient = self.repository.get_patient_context(state["patient_id"])
        return dict(patient_context=patient.as_prompt_context() if patient else "",
            pending_exams=list(patient.pending_exams) if patient else [],
            steps=state["steps"] + ["consulta SQLite somente leitura"])

    def retrieve(self, state):
        # Apenas a pergunta define a cobertura; comorbidades não justificam fontes para perguntas alheias.
        docs = self.retriever.retrieve(state["anonymized_question"], k=2, minimum_score=0.08)
        return dict(sources=[vars(d) for d in docs], steps=state["steps"] + ["busca lexical dos protocolos"])

    def route_context(self, state):
        return "generate" if state["patient_context"] and state["sources"] else "fallback"

    def generate(self, state):
        protocol = "\n\n".join(s["excerpt"] for s in state["sources"])
        result = str(self.chain.invoke(dict(question=state["anonymized_question"],
            patient_context=state["patient_context"], protocol_context=protocol)))
        valid, reasons = validate_output(result, bool(state["sources"]), state["patient_context"] + "\n" + protocol)
        return dict(draft=result, output_valid=valid, validation_reasons=list(reasons),
            steps=state["steps"] + ["geração concluída", "verificação literal de evidências"])

    def finalize(self, state):
        sources = "\n".join(f"[{s['source_id']}] {s['title']} — versão {s['version']}" for s in state["sources"])
        return dict(answer=state["draft"] + "\n\nFontes recuperadas:\n" + sources + "\n" + HUMAN_REVIEW_NOTICE,
            route="trecho_para_revisao", steps=state["steps"] + ["trecho disponibilizado para revisão"])

    def fallback(self, state):
        reasons = state.get("validation_reasons") or ["paciente ausente ou contexto insuficiente"]
        return dict(answer=safe_fallback(reasons), output_valid=False, validation_reasons=reasons,
            route="bloqueio_de_seguranca", steps=state["steps"] + ["resposta retida"])

    def _build(self):
        graph = StateGraph(AssistantState)
        for name in ("sanitize", "alert", "refuse", "load_patient", "retrieve", "generate", "finalize", "fallback"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "sanitize")
        graph.add_conditional_edges("sanitize", self.route_input,
            {"alert": "alert", "refuse": "refuse", "load_patient": "load_patient"})
        graph.add_edge("load_patient", "retrieve")
        graph.add_conditional_edges("retrieve", self.route_context, {"generate": "generate", "fallback": "fallback"})
        graph.add_conditional_edges("generate", lambda s: "finalize" if s["output_valid"] else "fallback",
            {"finalize": "finalize", "fallback": "fallback"})
        for name in ("alert", "refuse", "finalize", "fallback"): graph.add_edge(name, END)
        return graph.compile()

    def invoke(self, question, patient_id):
        start = time.perf_counter()
        state = dict(question=question, patient_id=patient_id, steps=[], request_id=str(uuid.uuid4()),
                     backend=type(self.generator).__name__)
        try:
            for updates in self.graph.stream(state, stream_mode="updates"):
                for update in updates.values():
                    state.update(update)
            return state
        except Exception as exc:
            state.update(route="erro_operacional", error_type=type(exc).__name__, output_valid=False)
            raise
        finally:
            state["elapsed_seconds"] = round(time.perf_counter() - start, 4)
            self.audit_logger.write(state)
