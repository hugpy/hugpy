"""hugpy-media: task plugins for the Hugpy engine.

Embeddings/similarity, summaries, keywords, speech recognition, TTS, vision
chat, vision analysis, image generation, ComfyUI and document/URL extraction.
Heavy frameworks are extras imported only inside the selected runner, so this
package imports without torch, diffusers, whisper, keybert,
sentence-transformers, OpenCV or llama.cpp installed.

Wiring: ``hugpy_media.plugin.register()`` (also the ``hugpy_engine.tasks``
entry point ``media``) puts every media task into ``hugpy_engine.tasks``;
``hugpy_media.hooks`` is where the fleet worker observes spawned processes.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "hooks", "plugin", "extract", "chunking"]


def __getattr__(name: str):
    # Submodules are exposed by name but imported on demand, keeping
    # ``import hugpy_media`` free of even the pydantic schema imports.
    if name in ("hooks", "plugin", "extract", "chunking"):
        import importlib

        return importlib.import_module(f"hugpy_media.{name}")
    raise AttributeError(f"module 'hugpy_media' has no attribute {name!r}")
