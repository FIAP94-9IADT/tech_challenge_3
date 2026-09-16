import time

import streamlit as st

from medical import store
from medical.assistant import ask
from medical.config import DATA, DB, REQUEST_COOLDOWN_SECONDS, model_name
from medical.safety import NOTICE

st.set_page_config(page_title="MedAssist | Tech Challenge", page_icon="🩺", layout="wide")
st.title("MedAssist")
st.caption("Tech Challenge · Fase 3 · Assistente médico educacional com fontes rastreáveis")
st.warning(NOTICE + " Use apenas perguntas sem dados pessoais reais.")

if not DB.exists():
    st.info("Prepare a base: python -m medical.dataset e depois python -m medical.store")
    st.stop()

store.initialize()

available = (DATA / "model.txt").exists() or (DATA / "adapter" / "adapter_config.json").exists()
with st.sidebar:
    st.header("Configuração")
    custom = st.toggle("Usar modelo com fine-tuning", disabled=not available)
    st.caption("Modelo: " + model_name(custom))
    st.caption("Modelo LoRA local experimental, principalmente em inglês." if custom else "GPT-5 nano: uma chamada por consulta e busca local sem embeddings pagos.")
    status = store.database_status()
    col1, col2 = st.columns(2)
    col1.metric("Documentos", status["documents"])
    col2.metric("Revisões pendentes", status["pending_reviews"])
    st.divider()
    st.caption("MedQuAD é informação pública. Não há prontuários nem protocolos hospitalares nesta demonstração.")

consultation, review_queue = st.tabs(["Consulta", "Fila de revisão"])

with consultation:
    st.subheader("Consulta baseada em evidências")
    st.caption("A resposta só é liberada como rascunho se citar fontes recuperadas pela base.")
    with st.form("question"):
        question = st.text_area("Dúvida para discussão médica", placeholder="Ex.: Quais são os sintomas de diabetes?", max_chars=1200)
        submitted = st.form_submit_button("Consultar evidências", type="primary")

    if submitted:
        elapsed = time.monotonic() - st.session_state.get("last_request_at", 0)
        if elapsed < REQUEST_COOLDOWN_SECONDS:
            st.warning(f"Aguarde {REQUEST_COOLDOWN_SECONDS - elapsed:.0f}s antes de uma nova consulta.")
        else:
            st.session_state.last_request_at = time.monotonic()
            with st.spinner("Consultando evidências..."):
                try:
                    st.session_state.result = ask(question, custom)
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Consulta não concluída. O erro foi registrado sem conteúdo da pergunta.")

    if result := st.session_state.get("result"):
        st.divider()
        st.caption(f'Consulta {result["run_id"]} · Modelo {result["model"]} · Estado: {result["status"]}')
        st.subheader("Rascunho para revisão")
        st.write(result["answer"])
        with st.expander("Fontes e consumo de tokens", expanded=True):
            for source in result["sources"]:
                st.markdown(f'**[{source["id"]}] {source["question"]}**')
                st.caption(source["source"])
                st.write(source["answer"][:1800])
            st.json(result["usage"])
        if result["status"] == "pending_review":
            with st.form("review_current_" + result["run_id"]):
                reviewer = st.text_input("Identificador do revisor", placeholder="medico_demo", max_chars=40)
                approved = st.radio("Decisão", [False, True], format_func=lambda value: "Aprovar rascunho educacional" if value else "Rejeitar")
                if st.form_submit_button("Registrar revisão"):
                    try:
                        result["status"] = store.review(result["run_id"], approved, reviewer)
                        st.success("Revisão registrada.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        with st.expander("Auditoria desta consulta"):
            st.dataframe(store.audit_events(result["run_id"]), hide_index=True, use_container_width=True)

with review_queue:
    st.subheader("Rascunhos aguardando revisão")
    pending = store.pending_reviews()
    if not pending:
        st.success("Não há rascunhos pendentes.")
    for item in pending:
        with st.expander("Consulta " + item["run_id"]):
            st.write(item["draft"])
            with st.form("review_pending_" + item["run_id"]):
                reviewer = st.text_input("Identificador do revisor", key="reviewer_" + item["run_id"], max_chars=40)
                approved = st.radio("Decisão", [False, True], key="decision_" + item["run_id"], format_func=lambda value: "Aprovar" if value else "Rejeitar")
                if st.form_submit_button("Registrar", key="submit_" + item["run_id"]):
                    try:
                        store.review(item["run_id"], approved, reviewer)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
