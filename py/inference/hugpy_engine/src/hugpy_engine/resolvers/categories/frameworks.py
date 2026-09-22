"""``(framework, task) -> Runner`` — a live view over :mod:`hugpy_engine.tasks`.

Historically this module held a static table that imported every runner in
the tree (torch, diffusers, whisper, the studio spine...). The table is now
the plugin registry: the engine registers its own chat runners here
(:func:`register_builtin_tasks`, lazily imported so a CPU-only host never pays
for llama_cpp/transformers at import), ``hugpy_media.plugin.register()`` /
``hugpy_video.plugin.register()`` add theirs, and ``hugpy_engine.tasks`` entry
points are loaded on first use. ``FRAMEWORK_RUNNERS`` / ``KNOWN_TASKS_REGISTRY``
keep their dict/set semantics for every existing caller.
"""
from __future__ import annotations

from hugpy_engine.tasks import RunnerTable, TaskNames, register_task

SOURCE = "hugpy_engine"


def _llama_chat_runner():
    from hugpy_engine.llama.runners.chat_runner import LlamaCppChatRunner

    return LlamaCppChatRunner


def _deepcoder_chat_runner():
    from hugpy_engine.generate.generate_runner import DeepCoderChatRunner

    return DeepCoderChatRunner


def register_builtin_tasks(*, replace: bool = False) -> None:
    """Register the engine-owned chat runners (idempotent).

    * ``("transformers", "text-generation")`` -> DeepCoderChatRunner
    * ``("gguf", "text-generation")``         -> LlamaCppChatRunner
    * ``("gguf", "image-text-to-text")``      -> LlamaCppChatRunner (a vision
      GGUF + mmproj rides the llama.cpp chat path; the image stays on
      ``ChatRequest.file`` — see builders._build_vision_chat_request)

    Request builders come from :mod:`.builders` (the chat-specific ones are
    engine-owned; ``("transformers", "image-text-to-text")`` belongs to media).
    """
    from hugpy_engine.resolvers.categories.builders import (
        _build_chat_request,
        _build_vision_chat_request,
    )

    register_task("text-generation", runner=_deepcoder_chat_runner,
                  build_request=_build_chat_request, frameworks=("transformers",),
                  extra="transformers", source=SOURCE, replace=replace)
    register_task("text-generation", runner=_llama_chat_runner,
                  build_request=_build_chat_request, frameworks=("gguf",),
                  extra="gguf", source=SOURCE, replace=replace)
    register_task("image-text-to-text", runner=_llama_chat_runner,
                  build_request=_build_vision_chat_request, frameworks=("gguf",),
                  extra="gguf", source=SOURCE, replace=replace)


register_builtin_tasks()

# (framework, task) -> runner class. Membership/lookup honour "*" wildcards.
FRAMEWORK_RUNNERS = RunnerTable()

# Every registered task name — derived from the registry so it can't drift.
KNOWN_TASKS_REGISTRY = TaskNames()

__all__ = ["FRAMEWORK_RUNNERS", "KNOWN_TASKS_REGISTRY", "register_builtin_tasks"]
