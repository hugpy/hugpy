"""Behavioural smoke tests for the media package's public seams:

  * ``hugpy_media.plugin.register`` populates the engine task table with the
    expected task keys, lazily (no model stack imported by registering);
  * ``import hugpy_media`` + ``register()`` work with every heavy library
    blocked (CPU-only / base install);
  * ``hugpy_media.hooks`` defaults are no-ops and the worker can install
    callbacks.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from hugpy_engine import tasks as engine_tasks

from hugpy_media import hooks, plugin

EXPECTED_TASKS = {
    "automatic-speech-recognition",
    "text-to-speech",
    "text-summarization",
    "text2text-generation",
    "keyword-extraction",
    "feature-extraction",
    "sentence-similarity",
    "image-text-to-text",
    "text-to-image",
    "image-to-image",
    "document-extraction",
    "url-extraction",
    "depth-estimation",
    "object-detection",
    "image-classification",
    "image-segmentation",
}

HEAVY = ["torch", "diffusers", "whisper", "keybert", "sentence_transformers",
         "cv2", "onnxruntime", "llama_cpp", "transformers", "chatterbox",
         "accelerate", "bitsandbytes"]


@pytest.fixture
def clean_registry():
    engine_tasks.reset_tasks()
    yield
    engine_tasks.reset_tasks()


def test_register_populates_engine_task_table(clean_registry):
    keys = plugin.register()
    assert set(keys) == EXPECTED_TASKS
    registered = engine_tasks.registered_tasks()
    assert EXPECTED_TASKS <= set(registered)
    for task in EXPECTED_TASKS:
        spec = registered[task]
        assert spec.source.startswith("hugpy_media")
        assert callable(spec.build_request)
        # the engine resolves factories to the runner CLASS
        cls = engine_tasks.runner_for_task(task)
        assert isinstance(cls, type), (task, cls)
        assert cls.__name__.endswith("Runner")


def test_comfy_pairs_registered_by_framework(clean_registry):
    plugin.register()
    pairs = engine_tasks.registered_pairs()
    assert ("comfy", "text-to-image") in pairs and ("comfy", "image-to-image") in pairs
    assert engine_tasks.runner_for_task("text-to-image", "comfy").__name__ == "ComfyRunner"
    assert engine_tasks.runner_for_task("text-to-image", "transformers").__name__ == "ImageGenRunner"
    assert ("*", "document-extraction") in pairs


def test_task_keys_cover_task_deps(clean_registry):
    from hugpy_engine.task_deps import TASK_DEPS

    media_keys = set(plugin.task_keys())
    # every media-served task in the engine's dependency map is registered
    assert {k for k in TASK_DEPS if k != "text-generation"} <= media_keys


def test_register_is_lazy_and_idempotent(clean_registry):
    plugin.register()
    plugin.register()
    spec = engine_tasks.task_spec("automatic-speech-recognition")
    assert isinstance(spec.runner, plugin.LazyRunner)
    # (import laziness itself is proven by test_import_and_register_without_heavy_libraries,
    #  in a fresh interpreter; here the module-level proxies may already be resolved)
    assert spec.runner() is spec.runner.resolve(), "zero-arg factory returns the class"
    assert len(engine_tasks.registered_pairs()) == len(plugin.task_pairs())


def test_request_builders_build_typed_requests(clean_registry):
    plugin.register()
    build = engine_tasks.request_builder_for_task("feature-extraction")
    req = build({"text": "hello"}, "bge-small")
    assert req.texts == ["hello"] and req.model_key == "bge-small"

    build = engine_tasks.request_builder_for_task("automatic-speech-recognition")
    req = build({"file": "/tmp/a.wav", "translate": True}, "whisper-base")
    assert req.file_path == "/tmp/a.wav" and req.task == "translate"

    build = engine_tasks.request_builder_for_task("document-extraction")
    assert build({"file": "/tmp/x.pdf"}, "")["path"] == "/tmp/x.pdf"

    with pytest.raises(ValueError):
        engine_tasks.request_builder_for_task("text-to-image")({}, "sd15")


def test_lazy_runner_resolves_to_the_class():
    lazy = plugin.LazyRunner("hugpy_media.embed.embed_runner", "FeatureExtractionRunner")
    from hugpy_media.embed.embed_runner import FeatureExtractionRunner

    assert lazy.resolve() is FeatureExtractionRunner
    assert lazy.request_type is FeatureExtractionRunner.request_type


def test_import_and_register_without_heavy_libraries():
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None; "
        + "; ".join(f"sys.modules[{name!r}] = None" for name in HEAVY)
        + "; import hugpy_media; from hugpy_media.plugin import register; "
        "keys = register(); print(len(keys)); "
        "from hugpy_engine.tasks import request_builder_for_task; "
        "print(request_builder_for_task('text-summarization')({'text': 'x'}, 'flan').text)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.split() == [str(len(EXPECTED_TASKS)), "x"]


def test_entry_point_is_declared():
    from importlib.metadata import entry_points

    eps = [ep for ep in entry_points(group="hugpy_engine.tasks") if ep.name == "media"]
    assert eps and eps[0].value == "hugpy_media.plugin:register"


def test_hooks_default_is_noop():
    hooks.reset_process_hooks()
    hooks.on_process_spawned(12345, "comfy")
    hooks.on_foreign_call_started("comfy", "ckpt", "job-1")
    hooks.on_foreign_call_ended("comfy", "job-1")


def test_hooks_install_and_reset():
    seen = []
    previous = hooks.set_process_hooks(
        on_process_spawned=lambda pid, label: seen.append(("spawn", pid, label)),
        on_foreign_call_started=lambda s, mk, jid: seen.append(("start", s, mk, jid)),
        on_foreign_call_ended=lambda s, jid: seen.append(("end", s, jid)),
    )
    try:
        hooks.on_process_spawned(7, "tts")
        hooks.on_foreign_call_started("comfy", "ckpt", "j")
        hooks.on_foreign_call_ended("comfy", "j")
        assert seen == [("spawn", 7, "tts"), ("start", "comfy", "ckpt", "j"), ("end", "comfy", "j")]
    finally:
        hooks.set_process_hooks(previous)
    hooks.reset_process_hooks()


def test_hooks_swallow_callback_errors():
    def boom(*a):
        raise RuntimeError("telemetry bug")

    hooks.set_process_hooks(on_process_spawned=boom, on_foreign_call_started=boom,
                            on_foreign_call_ended=boom)
    try:
        hooks.on_process_spawned(1, "x")
        hooks.on_foreign_call_started("x")
        hooks.on_foreign_call_ended("x")
    finally:
        hooks.reset_process_hooks()


def test_comfy_runner_reports_through_hooks():
    from hugpy_media.comfy import comfy_runner

    seen = []
    hooks.set_process_hooks(
        on_foreign_call_started=lambda s, mk, jid: seen.append(("start", s, mk, jid)),
        on_foreign_call_ended=lambda s, jid: seen.append(("end", s, jid)),
    )
    try:
        comfy_runner._reg_comfy_call("ckpt", "job-9")
        comfy_runner._end_comfy_call("job-9")
    finally:
        hooks.reset_process_hooks()
    assert seen == [("start", "comfy", "ckpt", "job-9"), ("end", "comfy", "job-9")]
