"""Mesma estrutura de prompt no ajuste e na aplicação."""
from langchain_core.runnables import RunnableLambda
from .prompting import format_prompt

def build_clinical_chain(generator):
    prompt = RunnableLambda(lambda data: format_prompt(data["question"],
        data["patient_context"] + "\n" + data["protocol_context"]))
    return prompt | RunnableLambda(generator.invoke)
