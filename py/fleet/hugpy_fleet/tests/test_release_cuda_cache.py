"""_release_cuda_cache clears the cuBLAS workspaces only when no in-process
model still holds CUDA weights (2026-10-02 pinned-segment leak)."""
import sys
import types

from hugpy_fleet.worker import agent as A


def _fake_torch(calls):
    t = types.SimpleNamespace()
    t.cuda = types.SimpleNamespace(is_initialized=lambda: True,
                                   synchronize=lambda: calls.append("sync"),
                                   empty_cache=lambda: calls.append("empty"))
    t._C = types.SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append("cublas"))
    return t


def test_idle_clears_workspaces_then_empties(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    monkeypatch.setattr(A, "_inprocess_gpu_bytes", lambda: {})
    A._release_cuda_cache()
    assert calls == ["sync", "cublas", "empty"]


def test_sibling_on_gpu_keeps_workspaces(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    monkeypatch.setattr(A, "_inprocess_gpu_bytes", lambda: {"other": {"device": "cuda"}})
    A._release_cuda_cache()
    assert calls == ["empty"]
