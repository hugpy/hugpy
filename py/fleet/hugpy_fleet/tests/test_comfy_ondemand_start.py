"""On-demand comfy start on the /infer path (2026-09-24): the worker owns comfy,
so a placed comfy job runs evict-to-fit FIRST and THEN starts the managed comfy,
and a start failure becomes the SAME structured load_failure an LLM load ships
(so central's relay path records it via record_load_failure).

These drive the worker agent's ``_ensure_comfy_started`` with the manager,
framework lookup and headroom evictor stubbed — no GPU, no ComfyUI, no systemd.
"""
import pytest

from hugpy_fleet.worker import agent
from hugpy_engine.serve.load_failure import ModelLoadFailure


class FakeManager:
    def __init__(self, managed=True, running=False, start_result=None, order=None):
        self._managed = managed
        self._running = running
        self._start_result = start_result or {"ok": True, "note": "started"}
        self.spec = {"kind": "systemd-user", "unit": "comfyui.service"}
        self._order = order if order is not None else []

    @property
    def managed(self):
        return self._managed

    def is_running(self, timeout=2.0):
        return self._running

    def start(self, ready_timeout=None):
        self._order.append("start")
        return self._start_result


@pytest.fixture
def patched(monkeypatch):
    order = []

    def fake_headroom(state, model_key, job_id=None):
        order.append("evict")
        return {"evicted": []}

    monkeypatch.setattr(agent, "_worker_ensure_comfy_headroom", fake_headroom)
    monkeypatch.setattr(agent, "_model_framework", lambda mk: "comfy")
    return order


def test_evict_to_fit_runs_before_start(patched, monkeypatch):
    mgr = FakeManager(managed=True, running=False, order=patched)
    monkeypatch.setattr(agent, "_comfy_manager", lambda state: mgr)
    agent._ensure_comfy_started(object(), "sdxl-comfy")
    assert patched == ["evict", "start"]        # evict-to-fit FIRST, then start


def test_start_failure_raises_structured_load_failure(patched, monkeypatch):
    mgr = FakeManager(
        managed=True, running=False, order=patched,
        start_result={"ok": False, "note": "comfy did not become ready",
                      "load_class": "unreachable",
                      "loader_stderr": "CUDA error: out of memory"})
    monkeypatch.setattr(agent, "_comfy_manager", lambda state: mgr)
    with pytest.raises(ModelLoadFailure) as ei:
        agent._ensure_comfy_started(object(), "sdxl-comfy")
    exc = ei.value
    assert exc.load_class == "unreachable"
    lf = exc.load_failure
    assert lf["class"] == "unreachable"
    assert "out of memory" in lf["loader_stderr"]
    assert lf["model_key"] == "sdxl-comfy"


def test_external_launcher_never_starts_or_evicts(patched, monkeypatch):
    mgr = FakeManager(managed=False, running=False, order=patched)
    monkeypatch.setattr(agent, "_comfy_manager", lambda state: mgr)
    agent._ensure_comfy_started(object(), "sdxl-comfy")
    assert patched == []                        # unmanaged: unchanged, no-op


def test_already_running_skips_start_and_evict(patched, monkeypatch):
    mgr = FakeManager(managed=True, running=True, order=patched)
    monkeypatch.setattr(agent, "_comfy_manager", lambda state: mgr)
    agent._ensure_comfy_started(object(), "sdxl-comfy")
    assert patched == []                        # per-gen headroom hook handles it


def test_non_comfy_model_is_a_noop(patched, monkeypatch):
    monkeypatch.setattr(agent, "_model_framework", lambda mk: "gguf")
    mgr = FakeManager(managed=True, running=False, order=patched)
    monkeypatch.setattr(agent, "_comfy_manager", lambda state: mgr)
    agent._ensure_comfy_started(object(), "some-llm")
    assert patched == []
