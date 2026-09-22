from abstract_essentials import get_env_value, is_number
from hugpy_engine.config.models.models_config import MODEL_REGISTRY
from hugpy_platform.utils import get_port
# ---------------------------------------------------------------------------
# Defaults — single source of truth for "what does the runner do when the
# request omits a value." Override per-runner via constructor or per-request
# via ChatRequest.
# ---------------------------------------------------------------------------

DEFAULT_N_CTX = 16384          # was 4096; small ctx silently truncated long outputs
DEFAULT_HTTP_TIMEOUT = 3600.0   # non-streaming HTTP only; streaming uses None




# ---------------------------------------------------------------------------
# Env / port wiring (host:port discovery for the HTTP runner)
# ---------------------------------------------------------------------------

LLAMA_HOST_DEFAULT = get_env_value("LLAMA_HOST") or "http://127.0.0.1"

def get_llama_ports():
    context_tokens = {}
    for module_key,values in MODEL_REGISTRY.items():
        if values.framework == "gguf":
            port = values.port
            if not is_number(port):
                port = get_port(module_key) or port
            context_tokens[module_key] = port
    return context_tokens
LLAMA_MODEL_PORTS: dict[str, int] = get_llama_ports()

# llama.cpp says 'length' / 'stop'; schema says 'max_tokens' / 'stop'.
FINISH_REASON_MAP = {
    "length": "max_tokens",
    "stop": "stop",
    None: "stop",
}
