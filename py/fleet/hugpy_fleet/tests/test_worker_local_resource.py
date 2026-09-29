"""Worker-local model availability must never depend on Central owning bytes."""
from __future__ import annotations

import importlib
import io
import tarfile
from pathlib import Path

import pytest

agent = importlib.import_module("hugpy_fleet.worker.agent")
provision = importlib.import_module("hugpy_storage.provision")


def test_live_resident_skips_central_registration_and_provision(monkeypatch):
    monkeypatch.setattr(agent, "loaded_model_keys", lambda: ["local-qwen"])
    monkeypatch.setattr(provision, "ensure_model_registered",
                        lambda *a, **k: pytest.fail("resident called Central metadata lookup"))
    monkeypatch.setattr(provision, "ensure_model_present",
                        lambda *a, **k: pytest.fail("resident tried Central transfer"))
    payload = {"model_key": "local-qwen"}

    assert agent._model_key_refusal(payload, "http://central") is None
    agent._ensure_present(payload, "http://central")


def test_pid_registry_resident_skips_central_even_outside_dispatch(monkeypatch):
    monkeypatch.setattr(agent, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(agent, "_slot_occupants", lambda: set())
    pidreg = importlib.import_module("hugpy_fleet.worker.pid_registry")
    monkeypatch.setattr(pidreg, "snapshot_for_heartbeat", lambda: {
        "models": [{"model_key": "external-qwen", "alive": True}],
    })
    assert agent._worker_has_resident("external-qwen") is True


def test_local_inventory_resolves_alias_before_central_transfer(monkeypatch):
    monkeypatch.setattr(agent, "_SYSTEM_MODELS", {
        "ready": True, "running": False, "at": 9999999999,
        "rows": {"Qwen/Qwen3.8-27B-FP8": {
            "model_key": "Qwen/Qwen3.8-27B-FP8",
            "name": "Qwen3.8-27B-FP8", "hub_id": "Qwen/Qwen3.8-27B-FP8",
        }},
    })
    assert agent._discover_local_model_key("Qwen3.8-27B-FP8") == "Qwen/Qwen3.8-27B-FP8"


def test_local_export_route_streams_only_discovered_model_files(tmp_path, monkeypatch):
    source = tmp_path / "resident-model"
    source.mkdir()
    (source / "weights.gguf").write_bytes(b"test weights")
    monkeypatch.setattr(agent, "_discover_local_model_key", lambda key: "local-model")
    monkeypatch.setattr(agent, "_system_models_snapshot", lambda: {
        "local-model": {"external_location": str(source), "dir": str(source)},
    })
    state = agent.WorkerState(name="t", url=None, worker_id="w-export", central_url=None)
    client = agent.build_app(state).test_client()

    response = client.get("/models/export/local-model")
    assert response.status_code == 200
    with tarfile.open(fileobj=io.BytesIO(response.data), mode="r:") as archive:
        assert archive.getnames() == ["weights.gguf"]
        assert archive.extractfile("weights.gguf").read() == b"test weights"
