"""Serve-path weights on a worker come from CENTRAL only (2026-09-23), and an
explicit ``alloc.worker`` pin is still admission-gated.

computron incident: LlamaCppChatRunner's slot path called
``hugpy_storage.download_models.ensure_model`` directly, which — with central
holding only two projector files — snapshot-downloaded a 134 GB repo from
Hugging Face onto an 8 GB-GPU worker. Every engine serve-path call site now goes
through ``hugpy_storage.provision.ensure_serving_weights`` (local-or-central on
a worker, raise if central holds no weights).
"""
from __future__ import annotations

import importlib

import pytest

SERVE_PATH_MODULES = (
    "hugpy_engine.llama.runners.get",
    "hugpy_engine.llama.runners.src.shard_server",
    "hugpy_engine.llama.runners.src.python_runner",
    "hugpy_engine.generate.config",
    "hugpy_engine.resolvers.model_resolver",
)


@pytest.mark.parametrize("modname", SERVE_PATH_MODULES)
def test_serve_path_modules_resolve_weights_through_central_first(modname):
    # Judged from the SOURCE, not the live attribute: other tests in this suite
    # rebind these module globals without restoring them.
    import inspect
    src = inspect.getsource(importlib.import_module(modname))
    assert "from hugpy_storage.provision import ensure_serving_weights" in src
    # The direct HF-capable entry point is no longer used in these modules.
    assert "download_models import ensure_model" not in src
    assert "ensure_model(" not in src.replace("ensure_model_", "")


def test_worker_serve_path_never_reaches_hf(monkeypatch):
    """Drive the real resolver the runners call, as a worker whose central
    holds no weights: it raises the precise reason; HF is never touched."""
    provision = importlib.import_module("hugpy_storage.provision")
    dm = importlib.import_module("hugpy_storage.download_models")
    monkeypatch.setenv("WORKER_CENTRAL_URL", "http://central")
    monkeypatch.setattr(dm, "_hf", lambda: pytest.fail("HF reached on a worker"))
    monkeypatch.setattr(provision, "ensure_model_present", lambda *a, **k: False)
    monkeypatch.setattr(provision, "model_is_local", lambda k: False)
    getmod = importlib.import_module("hugpy_engine.llama.runners.get")
    with pytest.raises(provision.CentralHoldsNoWeights,
                       match="central holds no weights for Qwen3.8-27B"):
        getmod.ensure_serving_weights("Qwen3.8-27B")


def test_staple_weights_failure_is_a_warning_not_a_crash(monkeypatch):
    mr = importlib.import_module("hugpy_engine.resolvers.model_resolver")
    key = next(iter(mr.MODELS), None)
    if key is None:
        pytest.skip("no staple models shipped")
    monkeypatch.setattr(mr, "HUGPY_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(mr, "_ensured_staples", set())

    def _refuse(k):
        raise RuntimeError("central holds no weights for " + k)

    monkeypatch.setattr(mr, "ensure_serving_weights", _refuse)
    assert mr.ensure_staple_weights(key) is None


# ── explicit worker pin admission ────────────────────────────────────────────
def _pinned(monkeypatch, gate):
    remote = importlib.import_module("hugpy_engine.resolvers.remote")
    w = {"id": "w-comp", "name": "computron"}
    monkeypatch.setattr(remote, "_worker_lookup_provider",
                        lambda want: w if want == "computron" else None)
    monkeypatch.setattr(remote, "_worker_pin_gate", gate)
    return remote, w


def test_pin_refused_when_gate_says_central_lacks_weights(monkeypatch):
    remote, _w = _pinned(monkeypatch, lambda w, mk: (
        f"central holds no weights for {mk} (mmproj/x-Q6_K.gguf)"))
    with pytest.raises(RuntimeError) as ei:
        remote._resolve_requested_worker("computron", "Qwen3.8", None, None)
    msg = str(ei.value)
    assert "requested worker 'computron' refused for Qwen3.8" in msg
    assert "central holds no weights for Qwen3.8 (mmproj/x-Q6_K.gguf)" in msg
    # 'requested worker' is a permanent-load marker: fail fast, never held.
    assert "requested worker" in remote._PERMANENT_LOAD_MARKERS


def test_pin_admitted_when_gate_has_no_objection(monkeypatch):
    remote, w = _pinned(monkeypatch, lambda w, mk: None)
    assert remote._resolve_requested_worker("computron", "m", None, None) is w


def test_broken_gate_never_invents_a_refusal(monkeypatch):
    def _boom(w, mk):
        raise ValueError("store down")
    remote, w = _pinned(monkeypatch, _boom)
    assert remote._resolve_requested_worker("computron", "m", None, None) is w


def test_no_gate_registered_is_the_old_behaviour(monkeypatch):
    remote, w = _pinned(monkeypatch, None)
    assert remote._resolve_requested_worker("computron", "m", None, None) is w
