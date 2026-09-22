from typing import Dict, Optional
from abstract_essentials import get_env_value
from hugpy_engine.llama.runners.src.imports.constants import LLAMA_HOST_DEFAULT, LLAMA_MODEL_PORTS
def load_llama_config(env_path: Optional[str] = None) -> Dict[str, str | int]:
    """Resolve host + per-model ports from env, with defaults as fallback.


    """
    cfg: Dict[str, str | int] = {}
    cfg["LLAMA_HOST"] = get_env_value("LLAMA_HOST", path=env_path) or LLAMA_HOST_DEFAULT

    for model_key, default_port in LLAMA_MODEL_PORTS.items():
        raw = get_env_value(f"{model_key}_PORT", path=env_path)
        cfg[model_key] = int(raw) if raw else default_port

    return cfg
