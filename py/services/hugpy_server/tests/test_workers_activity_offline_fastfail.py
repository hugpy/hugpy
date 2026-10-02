"""Activity is a console refresh path, not a recovery probe.

``GET /llm/workers/<id>/activity`` must NOT dial a worker central already knows
is offline. A black-holed box used to hold the request until nginx produced a
504, which made ONE dead worker look like the whole fleet was down. The fast
refusal reuses the 503 the ``WorkerUnreachable`` handler already returns, so the
console sees one contract either way; the explicit /health route (``force=True``)
remains the force-dial "is it back yet" check.

Sandbox note: like the other hugpy_server route tests, this needs the matching
abstract_flask/abstract_utilities in the venv; it runs under the fleet's own
test venv.
"""
from __future__ import annotations

import importlib

import pytest
from flask import Flask

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
wh = importlib.import_module("hugpy_fleet.central.worker_http")


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    app.config["TESTING"] = True
    return app.test_client()


def _dial_recorder(monkeypatch, resp=None):
    """Record every central->worker GET so a test can assert it never happened."""
    seen = []

    def _get(worker, path, call=None, read_timeout=None, **kw):
        seen.append({"path": path, "call": call, "read_timeout": read_timeout})
        return resp or _Resp(200, {"ok": True, "activity": []})

    monkeypatch.setattr(wh, "get", _get)
    return seen


def test_offline_worker_fast_fails_without_dialling(client, monkeypatch):
    monkeypatch.setattr(wr, "get_worker", lambda wid: {
        "id": wid, "status": "offline", "url": "http://10.99.0.1:9100"})
    seen = _dial_recorder(monkeypatch)

    r = client.get("/llm/workers/a-brain/activity")

    assert r.status_code == 503
    body = r.get_json()
    assert body["ok"] is False
    assert body["error"]["code"] == "WorkerUnreachable"
    assert body["error"]["url"] == "http://10.99.0.1:9100"
    # Callers render a list; the fast path must still hand them one.
    assert body["activity"] == []
    # The whole point: central never touched the dead box.
    assert seen == []


@pytest.mark.parametrize("status", ["unknown", "draining", "", None])
def test_any_non_online_status_fast_fails(client, monkeypatch, status):
    """The guard keys on `status != "online"`, not on the literal "offline"."""
    monkeypatch.setattr(wr, "get_worker", lambda wid: {
        "id": wid, "status": status, "url": "http://192.0.2.9:9100"})
    seen = _dial_recorder(monkeypatch)

    r = client.get("/llm/workers/wk-1/activity")

    assert r.status_code == 503
    assert seen == []


def test_online_worker_is_still_probed(client, monkeypatch):
    monkeypatch.setattr(wr, "get_worker", lambda wid: {
        "id": wid, "status": "online", "url": "http://192.0.2.9:9100",
        "models_local": []})
    seen = _dial_recorder(monkeypatch, _Resp(200, {
        "ok": True, "activity": [{"kind": "compute", "model_key": "org/m"}]}))

    r = client.get("/llm/workers/wk-1/activity")

    assert r.status_code == 200
    assert r.get_json()["activity"] == [{"kind": "compute", "model_key": "org/m"}]
    # Dialled once, on the short probe budget — never the long control timeout.
    assert len(seen) == 1
    assert seen[0]["path"] == "/ops/activity"
    assert seen[0]["call"] == "probe"


def test_unknown_worker_is_still_a_404(client, monkeypatch):
    """The fast-fail must not swallow the not-in-the-registry case."""
    monkeypatch.setattr(wr, "get_worker", lambda wid: None)
    seen = _dial_recorder(monkeypatch)

    r = client.get("/llm/workers/nope/activity")

    assert r.status_code == 404
    assert seen == []
