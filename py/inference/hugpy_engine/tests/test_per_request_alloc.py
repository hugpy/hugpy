"""Regression coverage for per-request allocation placement on the relay."""
import os
import tempfile

os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-request-alloc-"))
os.environ.setdefault("HUGPY_COMMS_DB", "off")

from hugpy_engine.schemas.chat_schemas import ChatRequest
from hugpy_engine.resolvers import remote


def _request(alloc):
    return ChatRequest(
        model_key="Qwen3.5-0.8B",
        messages=[{"role": "user", "content": "hello"}],
        alloc=alloc,
    )


def test_per_request_gpu_only_uses_loader_placement_wire(monkeypatch):
    monkeypatch.setattr(remote, "_spill_for", lambda _wid, _mk: {})

    payload = remote._worker_payload(
        "text-generation", _request({"alloc_mode": "gpu_only"}),
        "Qwen3.5-0.8B", "worker-1", worker={"pkg_version": "0.1.266"})

    assert payload["spill"] == {"n_gpu_layers": -1}
    assert "alloc_mode" not in payload["spill"]


def test_per_request_ram_only_uses_loader_placement_wire(monkeypatch):
    monkeypatch.setattr(remote, "_spill_for", lambda _wid, _mk: {})

    payload = remote._worker_payload(
        "text-generation", _request({"alloc_mode": "ram_only"}),
        "Qwen3.5-0.8B", "worker-1", worker={"pkg_version": "0.1.266"})

    assert payload["spill"] == {"n_gpu_layers": "off"}
    assert "alloc_mode" not in payload["spill"]


def test_per_request_mode_composes_with_four_bit(monkeypatch):
    monkeypatch.setattr(remote, "_spill_for", lambda _wid, _mk: {})

    payload = remote._worker_payload(
        "text-generation",
        _request({"alloc_mode": "gpu_only", "bnb_4bit": True}),
        "Qwen3.5-0.8B", "worker-1", worker={"pkg_version": "0.1.266"})

    assert payload["spill"] == {"n_gpu_layers": -1, "bnb_4bit": True}


def test_explicit_worker_bypasses_automatic_candidate_filters(monkeypatch):
    named = {"id": "w-aeb", "name": "aeb", "url": "http://aeb:9200",
             "models_local": [], "task_capabilities": {"text-generation": False}}
    monkeypatch.setattr(remote, "_worker_lookup_provider",
                        lambda want: named if want in ("aeb", "w-aeb") else None)
    monkeypatch.setattr(remote, "_worker_candidates_provider", lambda *_args: [])

    assert remote._resolve_requested_worker(
        "aeb", "Qwen3.5-0.8B", "some-other-pool", "text-generation") is named
