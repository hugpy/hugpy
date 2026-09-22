from abstract_essentials import SingletonMeta
from hugpy_platform.module_imports import get_whisper

from hugpy_media.whisper_model.constants import DEFAULT_WHISPER_MODEL_PATH
import threading

# Process-level guards.
#
# whisperManager is a SingletonMeta, so the *first* construction (and thus the
# first model load) is already serialized by the metaclass lock. We still take an
# explicit load lock with a double-check here so a swap to a different size/path
# can never be half-applied, and — more importantly — we expose a transcribe lock
# so concurrent /ml/transcribe calls single-flight through one CPU inference at a
# time instead of stacking several and saturating the box (no swap on this VM).
_WHISPER_LOAD_LOCK = threading.Lock()
WHISPER_TRANSCRIBE_LOCK = threading.Lock()


class whisperManager(metaclass=SingletonMeta):
    def __init__(
        self,
        module_size: str = "base",
        whisper_model_path: str | None = None,
    ):
        next_model_path = whisper_model_path or DEFAULT_WHISPER_MODEL_PATH

        if self._needs_load(module_size, next_model_path):
            with _WHISPER_LOAD_LOCK:
                # Re-check under the lock: another thread may have loaded the same
                # model while we were blocked.
                if self._needs_load(module_size, next_model_path):
                    self.whisper_model_path = next_model_path
                    self.module_size = module_size
                    self.whisper_model = get_whisper().load_model(
                        module_size,
                        download_root=next_model_path,
                    )
                    self.initialized = True

    def _needs_load(self, module_size: str, model_path: str) -> bool:
        return (
            not getattr(self, "initialized", False)
            or getattr(self, "module_size", None) != module_size
            or getattr(self, "whisper_model_path", None) != model_path
        )


def get_whisper_model(
    module_size: str = "base",
    whisper_model_path: str | None = None,
):
    whisper_mgr = whisperManager(
        module_size=module_size,
        whisper_model_path=whisper_model_path,
    )
    return whisper_mgr.whisper_model
