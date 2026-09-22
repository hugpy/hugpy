"""Task-runner registry: how media/video plug runners into the engine.

Covers the registration API (``register_task``), framework-scoped and
wildcard lookups, lazy runner factories, the live ``FRAMEWORK_RUNNERS`` /
``MODEL_REQUEST_BUILDERS`` / ``KNOWN_TASKS_REGISTRY`` views (a plugin's own
builder wins, the engine's legacy chat builder is the fallback), the
"unknown task -> clear error, never an import" contract, and entry-point
isolation.
"""
from __future__ import annotations

import pytest

from hugpy_engine import tasks as T


class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Runner:
    request_type = _Req
    result_type = dict


@pytest.fixture(autouse=True)
def _isolated_registry():
    saved = T.registered_pairs()
    T.reset_tasks()
    from hugpy_engine.resolvers.categories.frameworks import register_builtin_tasks

    register_builtin_tasks()
    yield
    T.reset_tasks()
    for (fw, task), spec in saved.items():
        T.register_task(task, runner=spec.runner, build_request=spec.build_request,
                        frameworks=(fw,), extra=spec.extra, source=spec.source)


def test_builtin_chat_tasks_are_registered_lazily():
    pairs = T.registered_pairs()
    assert ("gguf", "text-generation") in pairs
    assert ("transformers", "text-generation") in pairs
    assert ("gguf", "image-text-to-text") in pairs
    spec = pairs[("gguf", "text-generation")]
    assert spec.source == "hugpy_engine" and spec.extra == "gguf"
    assert spec.build_request is not None
    # the runner is a factory; nothing heavy is imported until asked
    assert not isinstance(spec.runner, type)


def test_register_and_lookup_by_framework_then_wildcard():
    built = []

    def build(kwargs, model_key):
        built.append(model_key)
        return _Req(model_key=model_key, **kwargs)

    spec = T.register_task("automatic-speech-recognition", runner=_Runner, build_request=build,
                           frameworks=("transformers",), extra="audio", source="test")
    assert spec.runner_class() is _Runner
    assert T.runner_for_task("automatic-speech-recognition", "transformers") is _Runner
    assert T.runner_for_task("automatic-speech-recognition") is _Runner
    assert T.runner_for_task("automatic-speech-recognition", "gguf") is None  # no wildcard
    assert T.request_builder_for_task("automatic-speech-recognition", "transformers") is build
    assert T.runner_for("automatic-speech-recognition") is _Runner  # guide alias
    T.register_task("url-extraction", runner=lambda: _Runner, source="test")  # wildcard
    assert T.runner_for_task("url-extraction", "anything") is _Runner
    assert ("*", "url-extraction") in T.registered_pairs()


def test_replace_false_keeps_the_first_registration():
    first = T.register_task("keyword-extraction", runner=_Runner, frameworks=("transformers",))
    second = T.register_task("keyword-extraction", runner=lambda: dict, frameworks=("transformers",), replace=False)
    assert second is first
    T.unregister_task("keyword-extraction", "transformers")
    assert T.task_spec("keyword-extraction") is None


def test_live_views_follow_the_registry_and_plugins_win():
    from hugpy_engine.resolvers.categories import FRAMEWORK_RUNNERS, KNOWN_TASKS_REGISTRY, MODEL_REQUEST_BUILDERS
    from hugpy_engine.resolvers.categories.builders import _build_chat_request

    assert ("gguf", "text-generation") in FRAMEWORK_RUNNERS
    assert MODEL_REQUEST_BUILDERS[("gguf", "text-generation")] is _build_chat_request
    assert "text-generation" in KNOWN_TASKS_REGISTRY
    assert ("transformers", "text-to-speech") not in FRAMEWORK_RUNNERS
    assert MODEL_REQUEST_BUILDERS.get(("transformers", "text-to-speech")) is None
    assert "text-to-speech" not in KNOWN_TASKS_REGISTRY

    # a plugin registers the pair: runner AND its own builder become visible
    def tts_builder(kwargs, model_key):
        return _Req(model_key=model_key)

    T.register_task("text-to-speech", runner=_Runner, build_request=tts_builder, frameworks=("transformers",))
    assert FRAMEWORK_RUNNERS[("transformers", "text-to-speech")] is _Runner
    assert MODEL_REQUEST_BUILDERS[("transformers", "text-to-speech")] is tts_builder
    assert "text-to-speech" in KNOWN_TASKS_REGISTRY
    assert ("transformers", "text-to-speech") in sorted(FRAMEWORK_RUNNERS)

    # a plugin that registers text-generation for a NEW framework without a
    # builder inherits the engine's legacy chat builder for that task
    T.register_task("text-generation", runner=_Runner, frameworks=("mlx",))
    assert MODEL_REQUEST_BUILDERS[("mlx", "text-generation")] is _build_chat_request

    # plugin builder overrides the legacy one for the same pair
    def my_chat(kwargs, model_key):
        return _Req(model_key=model_key, custom=True)

    T.register_task("text-generation", runner=_Runner, build_request=my_chat, frameworks=("gguf",))
    assert MODEL_REQUEST_BUILDERS[("gguf", "text-generation")] is my_chat


def test_unknown_task_is_a_clear_error_not_an_import():
    from hugpy_engine.resolvers.categories import FRAMEWORK_RUNNERS, MODEL_REQUEST_BUILDERS

    assert T.runner_for_task("zz-no-such-task") is None
    assert T.request_builder_for_task("zz-no-such-task") is None
    with pytest.raises(KeyError):
        FRAMEWORK_RUNNERS[("transformers", "zz-no-such-task")]
    with pytest.raises(KeyError):
        MODEL_REQUEST_BUILDERS[("transformers", "zz-no-such-task")]


def test_broken_runner_factory_reads_as_not_installed():
    from hugpy_engine.resolvers.categories import FRAMEWORK_RUNNERS

    def boom():
        raise ImportError("torch not installed")

    T.register_task("depth-estimation", runner=boom, frameworks=("transformers",))
    assert ("transformers", "depth-estimation") not in FRAMEWORK_RUNNERS
    assert FRAMEWORK_RUNNERS.get(("transformers", "depth-estimation")) is None


def test_entry_point_failures_are_isolated(monkeypatch):
    import importlib.metadata as md

    class _EP:
        def __init__(self, name, fn):
            self.name, self._fn = name, fn

        def load(self):
            return self._fn

    def good():
        T.register_task("image-classification", runner=_Runner, frameworks=("transformers",))

    def bad():
        raise RuntimeError("plugin exploded")

    monkeypatch.setattr(md, "entry_points", lambda group=None: [_EP("bad", bad), _EP("good", good)])
    assert T.load_entry_points() == 1
    assert T.runner_for_task("image-classification") is _Runner


def test_validate_registry_tolerates_missing_plugins_but_not_broken_staples(monkeypatch):
    """A staple whose (framework, task) the ecosystem serves but no plugin
    registered here is "not installed" (warn); a staple declaring a pair the
    ecosystem does not know at all is still a code bug (raise)."""
    import copy
    import dataclasses

    from hugpy_engine.config.models import models_config as mc
    from hugpy_engine.resolvers import model_resolver as MR

    template = next(iter(mc.MODEL_REGISTRY.values()))

    def probe(key, fw, task):
        row = copy.deepcopy(template)
        for f, v in (("model_key", key), ("name", key), ("folder", key), ("hub_id", f"zz/{key}"),
                     ("framework", fw), ("primary_task", task), ("tasks", [task])):
            object.__setattr__(row, f, v)
        return row

    key = "zz-staple-missing-plugin"
    mc.MODELS[key] = {"hub_id": f"zz/{key}", "framework": "transformers",
                      "primary_task": "text-to-speech", "tasks": ["text-to-speech"]}
    mc.MODEL_REGISTRY[key] = probe(key, "transformers", "text-to-speech")
    mc.MODEL_REGISTRY_DICT[key] = dataclasses.asdict(mc.MODEL_REGISTRY[key])
    try:
        MR.validate_registry()          # media not registered here: warn, keep
        assert key in mc.MODEL_REGISTRY
    finally:
        mc.MODELS.pop(key, None); mc.MODEL_REGISTRY.pop(key, None); mc.MODEL_REGISTRY_DICT.pop(key, None)
