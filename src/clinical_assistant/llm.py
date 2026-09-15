"""Geradores locais: teste determinístico e T5-LoRA."""
import os
from pathlib import Path

class DemoClinicalGenerator:
    """Retorna um trecho literal para testar a orquestração; não representa uma LLM."""
    def invoke(self, prompt):
        context = prompt.split("Evidências:", 1)[-1].rsplit("\nResposta:", 1)[0]
        for line in context.splitlines():
            if line.startswith("Exames pendentes:"):
                return line
        return ""

class T5Generator:
    def __init__(self, adapter_path="models/clinical-t5-lora", base_model_id="google/flan-t5-small"):
        os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from peft import PeftModel
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.model = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(base_model_id,
            revision="0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"), adapter_path)
        self.model.eval()

    def invoke(self, prompt):
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if inputs.input_ids.shape[1] > 512:
            raise ValueError("Contexto excede 512 tokens; reduzir evidências antes da geração")
        with torch.no_grad():
            result = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        return self.tokenizer.decode(result[0], skip_special_tokens=True)
