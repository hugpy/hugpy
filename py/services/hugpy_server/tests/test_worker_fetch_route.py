"""Central fetch-to-disk route (operator ruling 2026-09-24).

POST /llm/workers/<id>/fetch downloads a model onto the worker's DISK without
loading it into VRAM — the fetch-only half of the old download+load /probe. It
relays to the worker agent's /models/fetch and NEVER loads. An older agent
without the verb (worker answers 404) must produce a clear 501 refusal, never a
silent load.

Sandbox note: like the other hugpy_server route tests, this needs the matching
abstract_flask/abstract_utilities in the venv; it runs under the fleet's own
test venv, not the bare miniconda used for the pure-python curation suite.
"""
from __future__ import annotations

import importlib

import pytest
from flask import Flask

from worker_store_isolation import swap_worker_store

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")
W = importlib.import_module("hugpy_fleet.central.workers")
wh = importlib.import_module("hugpy_fleet.central.worker_http")

OPERATOR = {"X-Operator-Token": "s3cret"}


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    # Isolate the two feasibility guards so the relay path itself is under test.
    monkeypatch.setattr(wr, "_central_missing_reason", lambda mk: None)
    monkeypatch.setattr(wr, "_archive_refusal", lambda mk: None)
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    app.config["TESTING"] = True
    with swap_worker_store(prefix="hugpy-fetch-workers-"):
        W.worker_store.register(name="box", url="http://192.0.2.9:9100", worker_id="wk-1")
        yield app.test_client()


def _relay_recorder(monkeypatch, resp):
    seen = []

    def _post(worker, path, json=None, call=None, **kw):
        seen.append({"path": path, "json": json, "call": call})
        return resp

    monkeypatch.setattr(wh, "post", _post)
    return seen


def test_fetch_is_operator_gated(client, monkeypatch):
    seen = _relay_recorder(monkeypatch, _Resp(200, {"ok": True}))
    r = client.post("/llm/workers/wk-1/fetch", json={"model_key": "org/m"})   # no token
    assert r.status_code == 401
    assert seen == []                        # gated before the relay


def test_fetch_relays_to_models_fetch_and_never_loads(client, monkeypatch):
    seen = _relay_recorder(monkeypatch, _Resp(200, {
        "ok": True, "model_key": "org/m", "already_local": False,
        "started": True, "provisioning": True}))
    r = client.post("/llm/workers/wk-1/fetch", json={"model_key": "org/m"}, headers=OPERATOR)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] and body["started"]
    # It relays to the FETCH verb, never /probe/ (the load path).
    assert len(seen) == 1 and seen[0]["path"] == "/models/fetch"
    assert seen[0]["json"] == {"model_key": "org/m"}
    assert not any("/probe" in s["path"] for s in seen)


def test_old_agent_without_the_verb_gets_a_501_refusal_not_a_load(client, monkeypatch):
    seen = _relay_recorder(monkeypatch, _Resp(404, {"error": "Not Found"}))
    r = client.post("/llm/workers/wk-1/fetch", json={"model_key": "org/m"}, headers=OPERATOR)
    assert r.status_code == 501
    body = r.get_json()
    assert body["ok"] is False and body["unsupported"] is True
    assert "fetch-to-disk" in body["error"]
    # It reached /models/fetch (and only that) — never fell back to a load.
    assert [s["path"] for s in seen] == ["/models/fetch"]


def test_fetch_refuses_when_central_lacks_the_files(client, monkeypatch):
    monkeypatch.setattr(wr, "_central_missing_reason", lambda mk: "no local dir")
    seen = _relay_recorder(monkeypatch, _Resp(200, {"ok": True}))
    r = client.post("/llm/workers/wk-1/fetch", json={"model_key": "org/m"}, headers=OPERATOR)
    assert r.status_code == 409
    assert r.get_json()["ok"] is False
    assert seen == []                        # never even reached the worker
