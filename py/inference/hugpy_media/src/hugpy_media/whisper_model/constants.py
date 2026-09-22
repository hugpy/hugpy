import logging
import os

from hugpy_platform.constants import MODELS_HOME
logger = logging.getLogger(__name__)

# openai-whisper's load_model(name, download_root=...) looks for
# <download_root>/<name>.pt (e.g. base.pt). DEFAULT_PATHS[DEFAULT_WHISPER_MODEL]
# resolves to the HF *transformers* dir for whisper-large-v3-turbo, which holds
# safetensors, NOT the openai .pt weights — pointing the openai-whisper loader
# there makes it find no .pt and attempt a (slow / failing) network download.
# Point it at the cached openai .pt store (tiny/base/small.pt) instead.
# Override with WHISPER_MODEL_DIR (or WHISPER_DOWNLOAD_ROOT) on other boxes /
# the self-hosted product.
_CACHED_WHISPER_PT_DIR = "/mnt/llm_storage/legacy/models/whisper_base"


def _resolve_whisper_model_path() -> str:
    override = os.environ.get("WHISPER_MODEL_DIR") or os.environ.get("WHISPER_DOWNLOAD_ROOT")
    if override:
        return override
    if os.path.isdir(_CACHED_WHISPER_PT_DIR):
        return _CACHED_WHISPER_PT_DIR
    # Fresh box / self-hosted product: give openai-whisper a writable cache dir
    # under the models root (it will fetch the .pt there on first use) rather
    # than the HF transformers dir that has no .pt.
    try:
        fallback = os.path.join(MODELS_HOME, "whisper")
        os.makedirs(fallback, exist_ok=True)
        return fallback
    except Exception:
        return _CACHED_WHISPER_PT_DIR


DEFAULT_WHISPER_MODEL_PATH = _resolve_whisper_model_path()
