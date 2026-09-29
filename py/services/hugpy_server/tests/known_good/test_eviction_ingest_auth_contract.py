"""KNOWN-GOOD CONTRACT — worker eviction telemetry authenticates like a heartbeat.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/core-isolation-step2-2026-09-29.md (F5).
Source under test: hugpy_server/app/routes/eviction_routes.py
(_worker_authorized), hugpy_server/app/routes/worker_routes.py
(_enrollment_ok / _bearer_token), hugpy_fleet/worker/agent.py
(CentralClient.heartbeat / evictions_ingest).

LIVE CASE (ae-worker + computron, 2026-09-29 14:16): both relays logged
``eviction telemetry relay: 1 failed flush(es), last: HTTP Error 401``
while their heartbeats — same box, same Authorization header — were
accepted, so central's journal carried none of the fleet's evict.start/done
events. The ingest gate demanded a PRESENT token; the heartbeat gate followed
the gradual-rollout rule. One credential, one rule.
"""
from __future__ import annotations

import importlib
import json

import pytest

flask = pytest.importorskip("flask")

er = importlib.import_module("hugpy_server.app.routes.eviction_routes")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
A = importlib.import_module("hugpy_fleet.worker.agent")


@pytest.fixture
def app():
    return flask.Flask(__name__)


def _ctx(app, events, headers=None):
    return app.test_request_context(
        "/llm/evictions/ingest", method="POST",
        data=json.dumps({"events": events}), content_type="application/json",
        headers=headers or {})


def test_ingest_follows_the_heartbeat_gate_verbatim(app, monkeypatch):
    """INVARIANT: whatever verdict `_enrollment_ok` gives a heartbeat, the ingest
    gate gives a batch carrying a bearer token. A present-but-invalid token is
    denied on both; a valid one is accepted on both. Established: step 2 F5."""
    verdict = {"ok": True}
    monkeypatch.setattr(wr, "_enrollment_ok", lambda: verdict["ok"])
    monkeypatch.setattr(wr, "_bearer_token", lambda: "tok")
    with _ctx(app, [{"stage": "evict.done", "worker_id": "w1"}]):
        assert er._worker_authorized() is True
        verdict["ok"] = False                         # revoked / invalid / required+absent
        assert er._worker_authorized() is False


def test_tokenless_batch_is_bound_to_a_registered_worker(app, monkeypatch):
    """INVARIANT (gradual rollout, HUGPY_WORKER_ENROLL_REQUIRED off): a tokenless
    batch is accepted exactly when a tokenless heartbeat would be AND every
    worker_id it names is a worker central knows; a batch naming nobody, or an
    unknown box, is refused. Established: step 2 F5."""
    monkeypatch.setattr(wr, "_enrollment_ok", lambda: True)     # tokenless allowed
    monkeypatch.setattr(wr, "_bearer_token", lambda: None)
    W = importlib.import_module("hugpy_fleet.central.workers")
    known = {"w1": {"id": "w1", "name": "ae-worker"}}
    monkeypatch.setattr(W.worker_store, "get", lambda wid: known.get(wid))
    with _ctx(app, [{"stage": "evict.start", "worker_id": "w1"},
                    {"stage": "evict.done", "worker_id": "w1"}]):
        assert er._worker_authorized() is True
    with _ctx(app, [{"stage": "evict.done", "worker_id": "ghost"}]):
        assert er._worker_authorized() is False
    with _ctx(app, [{"stage": "evict.done"}]):                    # names nobody
        assert er._worker_authorized() is False
    monkeypatch.setattr(wr, "_enrollment_ok", lambda: False)     # enroll required
    with _ctx(app, [{"stage": "evict.done", "worker_id": "w1"}]):
        assert er._worker_authorized() is False


def test_ingest_route_answers_401_only_when_the_heartbeat_gate_would(app, monkeypatch):
    monkeypatch.setattr(wr, "_bearer_token", lambda: "tok")
    app.register_blueprint(er.eviction_bp)
    c = app.test_client()
    monkeypatch.setattr(wr, "_enrollment_ok", lambda: False)
    assert c.post("/llm/evictions/ingest", json={"events": []},
                  headers={"Authorization": "Bearer tok"}).status_code == 401
    monkeypatch.setattr(wr, "_enrollment_ok", lambda: True)
    monkeypatch.setattr(er.evictions_mod, "get_store",
                        lambda: type("S", (), {"append": staticmethod(lambda ev: len(ev))})())
    r = c.post("/llm/evictions/ingest",
               json={"events": [{"stage": "evict.done", "worker_id": "w1"}]},
               headers={"Authorization": "Bearer tok"})
    assert r.status_code == 200 and r.get_json()["received"] == 1


def test_worker_relay_sends_the_same_credential_as_its_heartbeat(monkeypatch):
    """INVARIANT (worker side): CentralClient.evictions_ingest carries byte-for-
    byte the Authorization header CentralClient.heartbeat carries, to the
    sibling /llm/evictions/ingest path under the same origin. Established:
    step 2 F5 (guards against the relay ever growing its own credential)."""
    import io
    import urllib.request
    seen = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen.append((req.full_url, dict(req.header_items())))
        return _Resp(b'{"ok": true}')
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(A, "_db_heartbeat_async", lambda wid, payload: None)
    c = A.CentralClient("http://central:7002", token="enroll-tok")
    c.heartbeat("w1", {"gpus": []})
    c.evictions_ingest([{"stage": "evict.done", "worker_id": "w1"}])
    (hb_url, hb_headers), (ev_url, ev_headers) = seen
    assert hb_url == "http://central:7002/api/llm/workers/w1/heartbeat"
    assert ev_url == "http://central:7002/api/llm/evictions/ingest"
    assert hb_headers.get("Authorization") == "Bearer enroll-tok"
    assert ev_headers.get("Authorization") == hb_headers.get("Authorization")
