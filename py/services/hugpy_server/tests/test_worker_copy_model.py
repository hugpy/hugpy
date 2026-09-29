"""Opt-in worker-to-Central model copy route."""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import tarfile

from flask import Flask

from worker_store_isolation import swap_worker_store

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")
workers = importlib.import_module("hugpy_fleet.central.workers")
worker_http = importlib.import_module("hugpy_fleet.central.worker_http")

OPERATOR = {"X-Operator-Token": "copy-secret"}


class _ArchiveResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def iter_raw(self):
        yield self.payload


def _tar_payload():
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:") as archive:
        data = b"gguf weights"
        info = tarfile.TarInfo("model.gguf")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def _client(monkeypatch):
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "copy-secret")
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    app.config["TESTING"] = True
    return app.test_client()


def test_copy_route_requires_operator_auth(monkeypatch, tmp_path):
    client = _client(monkeypatch)
    monkeypatch.setattr(wr, "get_worker", lambda _wid: None)
    with swap_worker_store(prefix="hugpy-copy-auth-"):
        response = client.post("/llm/workers/w1/copy-from-worker",
                              json={"model_key": "local-qwen"})
    assert response.status_code == 401


def test_copy_route_skips_model_already_on_central(monkeypatch, tmp_path):
    client = _client(monkeypatch)
    row = {"model_key": "local-qwen", "name": "local-qwen",
           "hub_id": "org/local-qwen", "framework": "gguf"}
    monkeypatch.setattr(wr, "get_worker", lambda _wid: {
        "id": "w1", "models_discovered": {
            "local-qwen": {"model_key": "local-qwen", "worker_location": "/worker/model"}}})
    monkeypatch.setattr(wr, "get_models_dict", lambda **_kw: {"local-qwen": row})
    monkeypatch.setattr(wr, "route_destination", lambda _row: str(tmp_path / "model"))
    monkeypatch.setattr("hugpy_storage.model_presence.model_looks_downloaded",
                        lambda *_a, **_kw: True)
    monkeypatch.setattr(worker_http, "stream",
                        lambda *_a, **_kw: (_ for _ in ()).throw(
                            AssertionError("must not copy a model already on Central")))

    with swap_worker_store(prefix="hugpy-copy-skip-"):
        response = client.post("/llm/workers/w1/copy-from-worker",
                               json={"model_key": "local-qwen"}, headers=OPERATOR)
    assert response.status_code == 200
    assert response.get_json()["already_on_central"] is True


def test_copy_route_streams_and_installs_missing_worker_model(monkeypatch, tmp_path):
    client = _client(monkeypatch)
    dest = tmp_path / "models" / "local-qwen"
    row = {"model_key": "local-qwen", "name": "local-qwen",
           "hub_id": "org/local-qwen", "framework": "gguf",
           "filename": "model.gguf", "tasks": ["text-generation"]}
    monkeypatch.setattr(wr, "get_worker", lambda _wid: {
        "id": "w1", "models_discovered": {
            "local-qwen": {"model_key": "local-qwen", "worker_location": "/worker/model"}}})
    monkeypatch.setattr(wr, "get_models_dict", lambda **_kw: {"local-qwen": row})
    monkeypatch.setattr(wr, "route_destination", lambda _row: str(dest))
    monkeypatch.setattr("hugpy_storage.model_presence.model_looks_downloaded",
                        lambda path, _row: os.path.isfile(os.path.join(path, "model.gguf")))

    seen = []

    @contextlib.contextmanager
    def stream(_method, _worker, path, call=None):
        seen.append((path, call))
        yield _ArchiveResponse(_tar_payload())

    monkeypatch.setattr(worker_http, "stream", stream)
    with swap_worker_store(prefix="hugpy-copy-install-"):
        response = client.post("/llm/workers/w1/copy-from-worker",
                               json={"model_key": "local-qwen"}, headers=OPERATOR)

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["bytes_copied"] == len(b"gguf weights")
    assert seen == [("/models/export/local-qwen", "transfer")]
    assert (dest / "model.gguf").read_bytes() == b"gguf weights"
    assert (dest / "hugpy.json").is_file()


def test_copy_route_refuses_unsafe_archive_path(monkeypatch, tmp_path):
    client = _client(monkeypatch)
    row = {"model_key": "local-qwen", "framework": "gguf"}
    monkeypatch.setattr(wr, "get_worker", lambda _wid: {
        "id": "w1", "models_discovered": {
            "local-qwen": {"model_key": "local-qwen", "worker_location": "/worker/model"}}})
    monkeypatch.setattr(wr, "get_models_dict", lambda **_kw: {"local-qwen": row})
    monkeypatch.setattr(wr, "route_destination", lambda _row: str(tmp_path / "model"))
    monkeypatch.setattr("hugpy_storage.model_presence.model_looks_downloaded",
                        lambda *_a, **_kw: False)

    bad = io.BytesIO()
    with tarfile.open(fileobj=bad, mode="w:") as archive:
        data = b"bad"
        info = tarfile.TarInfo("../outside")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    @contextlib.contextmanager
    def stream(*_a, **_kw):
        yield _ArchiveResponse(bad.getvalue())

    monkeypatch.setattr(worker_http, "stream", stream)
    with swap_worker_store(prefix="hugpy-copy-unsafe-"):
        response = client.post("/llm/workers/w1/copy-from-worker",
                               json={"model_key": "local-qwen"}, headers=OPERATOR)
    assert response.status_code == 502
    assert not (tmp_path / "outside").exists()
