"""Backends de geração: demonstração local e adaptador LoRA."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class TextGenerator(Protocol):
    def invoke(self, prompt: str) -> str: ...


class DemoClinicalGenerator:
    """Gerador determinístico para testes do pipeline sem inferência pesada."""

    def invoke(self, prompt: str) -> str:
        question = prompt
        if "PERGUNTA ANONIMIZADA" in prompt and "CONTEXTO ESTRUTURADO DO PACIENTE" in prompt:
            question = prompt.split("PERGUNTA ANONIMIZADA", 1)[1].split(
                "CONTEXTO ESTRUTURADO DO PACIENTE", 1
            )[0]
        lowered_question = question.lower()
        lowered_prompt = prompt.lower()
        if any(term in lowered_question for term in ("prescrev", "dose", "suspend", "receit")):
            body = (
                "Não posso prescrever, definir dose, iniciar ou suspender medicamentos. "
                "Posso organizar exames, registros e protocolos para a avaliação do profissional responsável."
            )
        elif "opç" in lowered_question or "discut" in lowered_question:
            if "hipertens" in lowered_question:
                options = (
                    "conferência da técnica de medida, educação sobre adesão, fatores de estilo de vida "
                    "e revisão profissional do plano terapêutico"
                )
            elif "diabet" in lowered_question:
                options = (
                    "educação para autocuidado, revisão de adesão, acompanhamento nutricional, "
                    "atividade física individualizada e revisão profissional do plano terapêutico"
                )
            else:
                options = "os pontos explicitamente descritos nos protocolos recuperados"
            body = (
                f"Opções registradas para discussão profissional: {options}. "
                "A lista não corresponde a recomendação individual ou prescrição."
            )
        elif "exames pendentes:" in lowered_prompt:
            pending = prompt.split("Exames pendentes:", 1)[-1].split("\n", 1)[0].strip()
            body = f"Exames pendentes registrados: {pending}. Conferir o status com a equipe responsável."
        else:
            body = (
                "Os dados e protocolos recuperados devem ser conferidos em conjunto. "
                "Considere as verificações descritas nas fontes e os sinais registrados, sem inferir diagnóstico."
            )
        return f"{body}\n\nA resposta organiza informações para apoio profissional e requer validação da equipe assistencial antes de qualquer decisão clínica."


def build_huggingface_generator(base_model_id: str, adapter_path: str):
    """Carrega o modelo base quantizado e conecta o adaptador treinado ao LangChain."""
    try:
        import torch
        from langchain_huggingface import HuggingFacePipeline
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    except ImportError as exc:
        raise RuntimeError("Instale as dependências opcionais com: pip install -e '.[training]'") from exc

    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adaptador LoRA não encontrado em {adapter_path}")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=os.getenv("HF_TOKEN") or None)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization,
        device_map="auto",
        token=os.getenv("HF_TOKEN") or None,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=320,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=text_pipeline)
