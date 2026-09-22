"""Engine plugin: register every media task with ``hugpy_engine.tasks``.

    from hugpy_media.plugin import register
    register()

is what the ``hugpy_engine.tasks`` entry point ``media`` runs (``pyproject``
``[project.entry-points."hugpy_engine.tasks"]``), and what the server and the
fleet worker entry points call explicitly. ``register()`` imports NO model
stack: each runner is a :class:`LazyRunner` proxy that resolves the real class
(and with it torch / diffusers / whisper / ...) only when first called or
inspected, and each request builder lives in ``hugpy_media.plugin_builders``
(pydantic schemas only).

Task keys are the dispatch keys of ``hugpy_engine.task_deps.TASK_DEPS`` plus
the two engine-table siblings media also serves (``image-to-image`` and the
``text2text-generation`` alias of summarization).
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple  # noqa: F401

from hugpy_media import __version__

__all__ = ["LazyRunner", "MEDIA_TASKS", "register", "task_keys", "task_pairs"]

SOURCE = f"hugpy_media {__version__}"


class LazyRunner:
    """A runner class that has not been imported yet — the engine's *zero-arg
    factory* shape: ``LazyRunner(...)()`` returns the runner class
    (``hugpy_engine.tasks.TaskSpec.runner_class`` calls it exactly so).

    Attribute access (``request_type``, ``result_type``) resolves the class
    first and :meth:`resolve` returns it. The import — and every heavy library
    behind it — happens on first use, never at registration time.
    """

    __slots__ = ("module", "qualname", "_cls", "_lock")

    def __init__(self, module: str, qualname: str) -> None:
        self.module = module
        self.qualname = qualname
        self._cls: Optional[type] = None
        self._lock = threading.Lock()

    def resolve(self) -> type:
        if self._cls is None:
            with self._lock:
                if self._cls is None:
                    mod = importlib.import_module(self.module)
                    self._cls = getattr(mod, self.qualname)
        return self._cls

    def __call__(self) -> type:
        return self.resolve()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def __repr__(self) -> str:
        state = "resolved" if self._cls is not None else "lazy"
        return f"LazyRunner({self.module}.{self.qualname}, {state})"


class _ExtractRunner:
    """Runner shape for the model-less extraction tasks.

    ``document-extraction`` and ``url-extraction`` need no weights; the request
    is the plain dict ``plugin_builders`` produces and the result is the dict
    ``hugpy_media.extract`` returns (``ok``/``error``/``text``/``pages``...).
    """

    request_type = dict
    result_type = dict

    def __init__(self, model_key: str = "", **_: Any) -> None:
        self.model_key = model_key

    async def stream(self, req, cancel_event=None):  # pragma: no cover - not streamable
        raise NotImplementedError("extraction tasks do not stream")


class DocumentExtractionRunner(_ExtractRunner):
    async def run(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        import asyncio

        from hugpy_media.extract import extract_document

        return await asyncio.to_thread(extract_document, req["path"])


class UrlExtractionRunner(_ExtractRunner):
    async def run(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        import asyncio

        from hugpy_media.extract import assess_url, fetch_url_text

        fn = assess_url if req.get("assess") else fetch_url_text
        return await asyncio.to_thread(fn, req["url"])


def _builder(name: str) -> Callable[[Mapping[str, Any], str], Any]:
    """A request builder resolved on first call (keeps ``register()`` free of
    even the pydantic schema imports)."""

    def build(kwargs: Mapping[str, Any], model_key: str):
        mod = importlib.import_module("hugpy_media.plugin_builders")
        return getattr(mod, name)(kwargs, model_key)

    build.__name__ = build.__qualname__ = name
    return build


#: (task, frameworks, runner, builder name, pip extra). Empty frameworks means
#: the engine's wildcard (the model-less extractors serve any registry row).
MEDIA_TASKS: Tuple[Tuple[str, Sequence[str], Any, str, Optional[str]], ...] = (
    ("automatic-speech-recognition", ("transformers",),
     LazyRunner("hugpy_media.whisper_model.src.runner", "WhisperRunner"),
     "build_whisper_request", "audio"),
    ("text-to-speech", ("transformers",),
     LazyRunner("hugpy_media.tts.tts_runner", "ChatterboxTtsRunner"),
     "build_tts_request", "tts"),
    ("text-summarization", ("transformers",),
     LazyRunner("hugpy_media.summarizers.summarize_runner", "SummarizeRunner"),
     "build_summarize_request", "transformers"),
    ("text2text-generation", ("transformers",),
     LazyRunner("hugpy_media.summarizers.summarize_runner", "SummarizeRunner"),
     "build_summarize_request", "transformers"),
    ("keyword-extraction", ("transformers",),
     LazyRunner("hugpy_media.keywords.keywords_runner", "KeywordRunner"),
     "build_keywords_request", "keywords"),
    ("feature-extraction", ("transformers",),
     LazyRunner("hugpy_media.embed.embed_runner", "FeatureExtractionRunner"),
     "build_embed_request", "embed"),
    ("sentence-similarity", ("transformers",),
     LazyRunner("hugpy_media.embed.embed_runner", "FeatureExtractionRunner"),
     "build_similarity_request", "embed"),
    # transformers vision chat. The GGUF variant (llama.cpp + mmproj) stays in
    # the engine: it rides the chat runner and the chat request.
    ("image-text-to-text", ("transformers",),
     LazyRunner("hugpy_media.vision.vision_runner", "VisionRunner"),
     "build_vl_text_or_vision_request", "vision"),
    ("text-to-image", ("transformers",),
     LazyRunner("hugpy_media.imagegen.imagegen_runner", "ImageGenRunner"),
     "build_imagegen_request", "imagegen"),
    ("image-to-image", ("transformers",),
     LazyRunner("hugpy_media.imagegen.imagegen_runner", "Img2ImgRunner"),
     "build_img2img_request", "imagegen"),
    # ComfyUI engine: a comfy row's checkpoint lives in the worker's own ComfyUI
    # install; the runner reuses the imagegen request/result shapes verbatim.
    ("text-to-image", ("comfy",),
     LazyRunner("hugpy_media.comfy.comfy_runner", "ComfyRunner"),
     "build_imagegen_request", "comfy"),
    ("image-to-image", ("comfy",),
     LazyRunner("hugpy_media.comfy.comfy_runner", "ComfyRunner"),
     "build_img2img_request", "comfy"),
    ("document-extraction", (), DocumentExtractionRunner,
     "build_document_extraction_request", "extract"),
    ("url-extraction", (), UrlExtractionRunner,
     "build_url_extraction_request", "extract"),
    ("depth-estimation", ("transformers",),
     LazyRunner("hugpy_media.vision_analysis.runner", "DepthEstimationRunner"),
     "build_vision_analysis_request", "transformers"),
    ("object-detection", ("transformers",),
     LazyRunner("hugpy_media.vision_analysis.runner", "ObjectDetectionRunner"),
     "build_vision_analysis_request", "transformers"),
    ("image-classification", ("transformers",),
     LazyRunner("hugpy_media.vision_analysis.runner", "ImageClassificationRunner"),
     "build_vision_analysis_request", "transformers"),
    ("image-segmentation", ("transformers",),
     LazyRunner("hugpy_media.vision_analysis.runner", "ImageSegmentationRunner"),
     "build_vision_analysis_request", "transformers"),
)


def task_keys() -> Tuple[str, ...]:
    """Distinct task keys media registers, in registration order."""
    return tuple(dict.fromkeys(task for task, *_ in MEDIA_TASKS))


def task_pairs() -> Tuple[Tuple[str, str], ...]:
    """Every (framework, task) pair media registers (``"*"`` = any framework)."""
    return tuple((fw, task) for task, fws, *_ in MEDIA_TASKS for fw in (fws or ("*",)))


def register(*, replace: bool = True) -> Tuple[str, ...]:
    """Register every media task with the engine's task table.

    Returns the task keys registered. Safe to call more than once. Imports no
    model stack (see module docstring).
    """
    from hugpy_engine.tasks import register_task

    for task, frameworks, runner, builder, extra in MEDIA_TASKS:
        register_task(
            task,
            runner=runner,
            build_request=_builder(builder),
            frameworks=frameworks,
            extra=extra,
            source=SOURCE,
            replace=replace,
        )
    return task_keys()
