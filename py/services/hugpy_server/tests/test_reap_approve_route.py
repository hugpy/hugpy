"""POST /llm/workers/<id>/reap-approve — central second-guard route regression.

Verifies the operator-approved eviction route wiring WITHOUT a live worker:
  * intersects approved keys with the freshly-recomputed proposal (drops keys no
    longer proposed — the render->approve race guard);
  * delegates the SURVIVORS to the same guarded reaper relay (/reap) that
    workers_reap uses, and folds approved/reaped/dropped into the result;
  * all-dropped -> early return (relay NEVER called, nothing deleted);
  * bad body -> 400; unknown worker -> 404.
"""
import pytest
from flask import Flask, jsonify

from hugpy_server.app.routes import worker_routes as wr

PROPOSAL = {
    "over_budget": True,
    "proposed_evictions": [
        {"model_key": "a", "bytes": 10, "last_picked": 100.0},
        {"model_key": "b", "bytes": 20, "last_picked": 200.0},
    ],
}


@pytest.fixture
def relay_calls(monkeypatch, tmp_path):
    """Stub the route's collaborators at the module level (the route calls bare
    names) and record every relay; audit writes stay out of the real projects tree."""
    monkeypatch.setenv("PROJECTS_HOME", str(tmp_path))
    calls = []

    def _fake_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
        calls.append({"op_path": op_path, "body": dict(body), "action": action})
        return jsonify({
            "ok": True, "freed_bytes": 123,
            "results": [{"model_key": k, "ok": True, "freed_bytes": 1}
                        for k in body["model_keys"]],
        }), 200

    monkeypatch.setattr(wr, "get_worker",
                        lambda wid: {"id": wid, "name": "box"} if wid == "wid" else None)
    monkeypatch.setattr(wr, "worker_storage_view", lambda wid: dict(PROPOSAL))
    monkeypatch.setattr(wr, "_relay_worker_op", _fake_relay)
    return calls


@pytest.fixture
def client(relay_calls):
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client()


def test_happy_path_intersects_and_relays(client, relay_calls):
    """'a' is proposed (kept), 'z' is not (dropped)."""
    r = client.post("/llm/workers/wid/reap-approve", json={"model_keys": ["a", "z"]})
    body = r.get_json()
    assert r.status_code == 200
    assert len(relay_calls) == 1
    # relay hits the SAME guarded /reap path
    assert relay_calls[0]["op_path"] == "/reap"
    assert relay_calls[0]["action"] == "reap-approve"
    # only the intersected key is relayed (z dropped centrally)
    assert relay_calls[0]["body"]["model_keys"] == ["a"]
    assert body["approved"] == ["a", "z"]
    assert body["reaped"] == ["a"]
    assert body["dropped"] == ["z"]          # render->approve race guard
    assert body["freed_bytes"] == 123        # reaper's typed freed_bytes passed through


def test_all_dropped_early_return_never_relays(client, relay_calls):
    r = client.post("/llm/workers/wid/reap-approve", json={"model_keys": ["z", "q"]})
    body = r.get_json()
    assert r.status_code == 200
    assert relay_calls == []                 # nothing deleted
    assert body["freed_bytes"] == 0
    assert set(body["dropped"]) == {"z", "q"} and body["reaped"] == []
    assert "note" in body


@pytest.mark.parametrize("payload", [{}, {"model_keys": []}])
def test_bad_body_400(client, payload):
    assert client.post("/llm/workers/wid/reap-approve", json=payload).status_code == 400


def test_unknown_worker_404(client):
    assert client.post("/llm/workers/nope/reap-approve",
                       json={"model_keys": ["a"]}).status_code == 404


def test_under_budget_at_approve_nothing_relayed(client, relay_calls, monkeypatch):
    """Proposal recomputed empty (worker back under budget) -> early return."""
    monkeypatch.setattr(wr, "worker_storage_view",
                        lambda wid: {"over_budget": False, "proposed_evictions": []})
    r = client.post("/llm/workers/wid/reap-approve", json={"model_keys": ["a"]})
    assert relay_calls == []
    assert r.get_json()["freed_bytes"] == 0
