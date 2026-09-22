"""POST /llm/workers/<id>/residency-all — bulk-residency route (todo t12).

The console's Serving table lets the operator multi-select models and change
their RESIDENCY tier in ONE action. That action must ride ONE /ops/config relay
(one agent re-exec), NOT N single-model POSTs — the same batching _relay_pin_all
does for pins. This regresses the route/relay wiring WITHOUT a live worker:

  * one relay call carrying the WHOLE selection as a single residency map;
  * on-demand normalizes to the null wire value (the agent's clears-the-override
    convention); "static" is stored verbatim; residency ONLY — never `pinned`;
  * off-worker keys in the selection are dropped (render->click staleness),
    reported as `skipped`, never written into the settings map;
  * per-model results / counts / mode / restarting surfaced like pin-all;
  * a relay failure still returns the full per-model map (never a bare 5xx);
  * bad body -> 400; bad mode -> 400; unknown worker -> 404.
"""
from __future__ import annotations

import importlib

import pytest
from flask import Flask, jsonify

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")

# The worker has models a,b,c designated; z is NOT designated to it.
WORKER = {"id": "wid", "name": "box", "models": ["a", "b", "c"]}


@pytest.fixture
def relay_calls(monkeypatch):
    """Fake the worker relay; records every call and replies like the agent
    does to /ops/config ({ok, settings, restarting})."""
    calls: list = []

    def _fake_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
        calls.append({"op_path": op_path, "body": dict(body),
                      "action": action, "retry": retry_on_connect})
        return jsonify({"ok": True, "settings": {}, "restarting": True}), 200

    monkeypatch.setattr(wr, "get_worker",
                        lambda wid: dict(WORKER) if wid == "wid" else None)
    monkeypatch.setattr(wr, "_relay_worker_op", _fake_relay)
    return calls


@pytest.fixture
def client(relay_calls):
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client()


def _post(client, path="/llm/workers/wid/residency-all", **json):
    return client.post(path, json=json)


def test_static_subset_one_relay_one_residency_map(client, relay_calls):
    r = _post(client, model_keys=["a", "b"], mode="static")
    body = r.get_json()
    assert r.status_code == 200
    assert len(relay_calls) == 1, "exactly ONE relay call (not one per model)"
    assert relay_calls[0]["op_path"] == "/ops/config"
    assert relay_calls[0]["retry"] is True, "config-style relay gets the apply-blip retry"
    assert relay_calls[0]["body"] == {"residency": {"a": "static", "b": "static"}}
    assert "pinned" not in relay_calls[0]["body"], "residency ONLY"
    assert body["mode"] == "static"
    assert body["results"] == {"a": "ok", "b": "ok"}
    assert body["counts"] == {"ok": 2, "error": 0, "total": 2}
    assert body["restarting"] is True, "restarting flag passed through"


def test_on_demand_normalizes_to_null_wire_value(client, relay_calls):
    r = _post(client, model_keys=["a", "c"], mode="on-demand")
    body = r.get_json()
    assert r.status_code == 200
    assert relay_calls[0]["body"] == {"residency": {"a": None, "c": None}}
    assert body["mode"] == "on-demand"
    assert body["counts"] == {"ok": 2, "error": 0, "total": 2}


def test_off_worker_key_dropped_reported_skipped(client, relay_calls):
    body = _post(client, model_keys=["a", "z"], mode="static").get_json()
    assert relay_calls[0]["body"] == {"residency": {"a": "static"}}
    assert body["skipped"] == ["z"]
    assert body["results"] == {"a": "ok"}


def test_all_keys_off_worker_no_relay_honest_note(client, relay_calls):
    r = _post(client, model_keys=["z", "q"], mode="static")
    body = r.get_json()
    assert r.status_code == 200
    assert relay_calls == [], "relay NEVER called"
    assert body["counts"] == {"ok": 0, "error": 0, "total": 0}
    assert "note" in body
    assert set(body["skipped"]) == {"z", "q"}


def test_relay_failure_returns_full_per_model_map(client, monkeypatch):
    def _fail_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
        return jsonify({"ok": False, "error": {
            "code": "AgentRestarting", "message": "worker agent is restarting"}}), 503
    monkeypatch.setattr(wr, "_relay_worker_op", _fail_relay)
    r = _post(client, model_keys=["a", "b"], mode="static")
    body = r.get_json()
    assert r.status_code == 200, "structured body carries the failure, never a bare 5xx"
    assert body["ok"] is False
    assert set(body["results"]) == {"a", "b"}
    assert all("restarting" in v for v in body["results"].values())
    assert body["counts"] == {"ok": 0, "error": 2, "total": 2}
    assert body.get("error", {}).get("code") == "AgentRestarting", "surfaced for fetchJson too"


@pytest.mark.parametrize("path,payload,status", [
    ("/llm/workers/wid/residency-all", {"mode": "static"}, 400),                    # missing model_keys
    ("/llm/workers/wid/residency-all", {"model_keys": [], "mode": "static"}, 400),  # empty
    ("/llm/workers/wid/residency-all", {"model_keys": ["a"], "mode": "bogus"}, 400),  # never silently clears
    ("/llm/workers/nope/residency-all", {"model_keys": ["a"], "mode": "static"}, 404),
])
def test_bad_body_bad_mode_unknown_worker(client, path, payload, status):
    assert client.post(path, json=payload).status_code == status


@pytest.mark.parametrize("mode", [None, "", "on-demand", "on_demand"])
def test_normalize_on_demand_synonyms_to_none(mode):
    assert wr._normalize_residency(mode) is None


def test_normalize_static_verbatim():
    assert wr._normalize_residency("static") == "static"


@pytest.mark.parametrize("mode", ["bogus", "serving", "warm"])
def test_normalize_garbage_to_sentinel(mode):
    """Anything that is not static / an on-demand spelling is the sentinel the
    route turns into a 400. "serving" / "warm" are NOT accepted synonyms any
    more (the normalizer only clears on null / "" / on-demand / on_demand)."""
    assert wr._normalize_residency(mode) == "__invalid__"


def test_route_is_operator_gated():
    """Same tier as config/pin-all: POST is in _SENSITIVE, GET is not (no such
    route, but never an open write)."""
    def _gated(path, method):
        return any(method in methods and rx.match(path)
                   for methods, rx in oa._SENSITIVE)
    assert _gated("/llm/workers/w1/residency-all", "POST")
    assert not _gated("/llm/workers/w1/residency-all", "GET")
