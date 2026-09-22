"""
Text generation manager (distilgpt2).

Singleton-loaded so repeated calls don't re-init the model.
"""

import logging

from abstract_essentials import SingletonMeta
from hugpy_platform.module_imports import get_torch, get_transformers

logger = logging.getLogger(__name__)


class GeneratorManager(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, "_ready"):
            # Spill seam (Slice C): distilgpt2 is a causal LM built by pipeline()
            # from a path/id, so the placement map rides model_kwargs + a
            # top-level device_map; device= is dropped when device_map is set
            # (mutually exclusive). None seam / no GPU / no accelerate ->
            # historical `device = 0 if cuda else -1`, byte-identical.
            pipe_kwargs = self._placement_kwargs()
            self.pipeline = get_transformers("pipeline")(
                "text-generation",
                model="distilgpt2",
                **pipe_kwargs,
            )
            self._ready = True

    @staticmethod
    def _placement_kwargs() -> dict:
        torch = get_torch()
        if not torch.cuda.is_available():
            return {"device": -1}
        try:
            from hugpy_engine.spill import transformers_max_memory
            mm = transformers_max_memory()
        except Exception as exc:  # noqa: BLE001 — no seam: today's path, logged
            logger.warning("generator spill seam unavailable (%s); loading on "
                           "the default device", exc)
            return {"device": 0}
        if not mm:
            return {"device": 0}
        try:
            import accelerate  # noqa: F401
        except ImportError:
            logger.warning("generator: spill seam produced a max_memory map but "
                           "accelerate is not installed — cannot honor the "
                           "allocation mode; loading on the default device")
            return {"device": 0}
        return {"device_map": "auto", "model_kwargs": {"max_memory": mm}}


def get_generator():
    return GeneratorManager().pipeline
