import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "hospital.db"
REQUEST_COOLDOWN_SECONDS = 5
load_dotenv(ROOT / ".env")
# Tracing remoto pode enviar prompts: a auditoria deste projeto é local.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


def api_key():
    for name in ("OPENAI_API_KEY", "OPEN_AI_API_KEY", "OPEN_AI_API"):
        if value := os.getenv(name, "").strip():
            return value
    raise ValueError("Configure OPENAI_API_KEY no .env (chave nunca exibida).")


def model_name(custom=False):
    if custom:
        if (DATA / "adapter" / "adapter_config.json").exists():
            return "local:SmolLM2-135M-LoRA"
        path = DATA / "model.txt"
        if not path.exists() or not path.read_text().strip().startswith("ft:"):
            raise ValueError("Modelo customizado indisponível. Execute o pipeline de fine-tuning.")
        return path.read_text().strip()
    return os.getenv("OPENAI_MODEL", "gpt-5-nano")
