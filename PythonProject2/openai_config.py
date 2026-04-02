import os
from pathlib import Path

from openai import OpenAI


def load_local_env() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    for candidate_name in (".env.openai", ".env"):
        env_path = root_dir / candidate_name
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))


def require_openai_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OPENAI_API_KEY


def get_openai_client() -> OpenAI:
    require_openai_api_key()
    return OpenAI(api_key=OPENAI_API_KEY)
