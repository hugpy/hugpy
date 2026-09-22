"""Media/video capabilities are optional plugins: absent packages are not
advertised, present media receives the worker's PID-registry callbacks via
``hugpy_media.hooks.set_process_hooks``."""

from __future__ import annotations

import sys
import types

import pytest

from hugpy_engine import tasks as T
from hugpy_fleet.worker import pid_registry, plugins


@pytest.fixture(autouse=True)
def _fresh():
    plugins.reset_for_tests()
    T.reset_tasks()
    yield
    plugins.reset_for_tests()
    T.reset_tasks()


def test_overlay_drops_media_tasks_without_media(monkeypatch):
    monkeypatch.setattr(plugins, "media_present", lambda: False)
    caps = {"text-to-image": True, "automatic-speech-recognition": True,
            "image-text-to-text": True, "text-summarization": False}
    out = plugins.overlay_task_capabilities(caps)
    assert out["text-to-image"] is False
    assert out["automatic-speech-recognition"] is False
    assert out["image-text-to-text"] is True      # engine-native: untouched
    assert out["text-summarization"] is False


def test_overlay_keeps_task_when_a_runner_registered(monkeypatch):
    monkeypatch.setattr(plugins, "media_present", lambda: False)
    T.register_task("text-to-image", runner=object, source="test")
    out = plugins.overlay_task_capabilities({"text-to-image": True, "text-to-speech": True})
    assert out["text-to-image"] is True
    assert out["text-to-speech"] is False


def test_boot_without_media_reports_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "hugpy_media", None)
    monkeypatch.setitem(sys.modules, "hugpy_video", None)
    summary = plugins.load_worker_capabilities(force=True)
    assert summary["media"] is False and summary["video"] is False
    assert summary["media_hooks"] == {"available": False, "installed": False}
    assert isinstance(summary["entry_points"], int)


def _fake_media(monkeypatch):
    """A stand-in ``hugpy_media.hooks`` exposing media's set_process_hooks contract."""
    pkg = types.ModuleType("hugpy_media")
    pkg.__path__ = []
    hooks = types.ModuleType("hugpy_media.hooks")
    installed = {}

    def set_process_hooks(hooks_obj=None, *, on_process_spawned=None,
                          on_foreign_call_started=None, on_foreign_call_ended=None):
        installed.update(spawned=on_process_spawned, started=on_foreign_call_started,
                         ended=on_foreign_call_ended)
    hooks.set_process_hooks = set_process_hooks
    monkeypatch.setitem(sys.modules, "hugpy_media", pkg)
    monkeypatch.setitem(sys.modules, "hugpy_media.hooks", hooks)
    monkeypatch.setattr(plugins, "media_present", lambda: True)
    return installed


def test_media_hooks_installed_when_media_present(monkeypatch):
    calls = []
    monkeypatch.setattr(pid_registry, "record_foreign_call",
                        lambda service, mk=None, job_id=None: calls.append(("start", service, mk, job_id)))
    monkeypatch.setattr(pid_registry, "end_foreign_call",
                        lambda service, job_id=None, model_key=None: calls.append(("end", service, job_id)))
    installed = _fake_media(monkeypatch)

    summary = plugins.load_worker_capabilities(force=True)
    assert summary["media"] is True
    assert summary["media_hooks"] == {"available": True, "installed": True}
    assert all(installed.get(k) for k in ("spawned", "started", "ended"))

    installed["spawned"](4242, "comfy")
    installed["started"]("comfy", "ckpt", "j1")
    installed["ended"]("comfy", "j1")
    assert ("start", "comfy", None, "pid:4242") in calls
    assert ("start", "comfy", "ckpt", "j1") in calls
    assert ("end", "comfy", "j1") in calls
    # memoised: a second call does not re-install
    assert plugins.load_worker_capabilities() == summary


def test_real_media_hooks_contract_if_installed():
    hooks = pytest.importorskip("hugpy_media.hooks")
    try:
        plugins.load_worker_capabilities(force=True)
        assert plugins._state["summary"]["media_hooks"]["installed"] is True
        current = hooks.get_process_hooks()
        assert type(current).__name__ != "_NoopHooks"
    finally:
        hooks.reset_process_hooks()
