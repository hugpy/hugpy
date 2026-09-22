"""Per-worker WILDCARD routing opt-in (Slice A, operator doctrine 2026-07-23)
— central's flag store + routes.

Worker designations are a HARD routing scope. An UNDESIGNATED model "gets in
where it fits in" — but ONLY on workers that opted in as WILDCARD. Once
resident, NORMAL eviction rules apply — the flag is routing only, an explicit
per-worker opt-in, DEFAULT FALSE (no flags set == today's routing exactly).

THIS file covers:
  * state: set / get / clear / absent-is-False + legacy bare shape
    (engine-owned ``models_config`` flag store, exercised here because the
    routes below are its only caller);
  * routes: GET map open, POST operator-gated, ``wildcard`` field surfaced on
    /llm/workers and /llm/workers/<id> (response-copy only, never persisted
    onto the stored record), clear via the route.

The routing doctrine itself (``_match_keys`` "~"-tail unification,
eligibility, allocation hard scope, the blocked-sibling guard, star ranking)
lives in ``py/fleet/hugpy_fleet/tests/test_worker_wildcard.py``.
"""

from __future__ import annotations

import importlib
import os

import pytest
from flask import Flask

from worker_store_isolation import swap_worker_store

mc = importlib.import_module("hugpy_engine.config.models.models_config")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")
W = importlib.import_module("hugpy_fleet.central.workers")

OPERATOR = {"X-Operator-Token": "s3cret"}


@pytest.fixture
def isolated_flag_store(tmp_path, monkeypatch):
    """The flag file lives beside MODELS_DISCOVERY_PATH; redirect that module
    global so the test never touches a live store."""
    monkeypatch.setattr(mc, "MODELS_DISCOVERY_PATH",
                        os.path.join(str(tmp_path), "model_discovery.json"))
    yield


@pytest.fixture
def gated_client(isolated_flag_store, monkeypatch):
    """Worker blueprint + operator gate (open mode + a token => the token is
    REQUIRED on sensitive routes; fleet reads stay open) over an isolated
    registry with two workers (``wk-wild``, ``wk-plain``)."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    app.config["TESTING"] = True
    with swap_worker_store(prefix="hugpy-wc-workers-"):
        W.worker_store.register(name="box", url="http://192.0.2.9:9100", worker_id="wk-wild")
        W.worker_store.register(name="box2", url="http://192.0.2.10:9100", worker_id="wk-plain")
        yield app.test_client()


# ── state store ──────────────────────────────────────────────────────────────

def test_flag_store_set_get_clear_absent_is_false(isolated_flag_store):
    assert mc.worker_wildcard_state() == {}
    assert mc.worker_wildcard_state().get("wk-nobody", False) is False

    assert mc.set_worker_wildcard("wk-a", True) == {"worker_id": "wk-a", "wildcard": True}
    assert mc.worker_wildcard_state() == {"wk-a": True}
    mc.set_worker_wildcard("wk-b", True)
    assert mc.worker_wildcard_state() == {"wk-a": True, "wk-b": True}

    # cleared worker drops OUT of the map (absent, not stored-False); idempotent
    assert mc.set_worker_wildcard("wk-a", False) == {"worker_id": "wk-a", "wildcard": False}
    assert mc.worker_wildcard_state() == {"wk-b": True}
    mc.set_worker_wildcard("wk-never", False)
    assert mc.worker_wildcard_state() == {"wk-b": True}

    # legacy/bare on-disk shape tolerated; falsy entries read absent
    mc.safe_dump_to_file(data={"wk-legacy": True, "wk-off": False},
                         file_path=mc._worker_wildcard_path())
    assert mc.worker_wildcard_state() == {"wk-legacy": True}


# ── routes ───────────────────────────────────────────────────────────────────

def test_set_route_is_operator_gated_and_persists(gated_client):
    r = gated_client.post("/llm/workers/wk-wild/wildcard", json={"enabled": True})
    assert r.status_code == 401
    assert mc.worker_wildcard_state() == {}

    r = gated_client.post("/llm/workers/wk-wild/wildcard", json={"enabled": True},
                          headers=OPERATOR)
    assert r.status_code == 200
    assert mc.worker_wildcard_state() == {"wk-wild": True}

    # clear via the route
    r = gated_client.post("/llm/workers/wk-wild/wildcard", json={"enabled": False},
                          headers=OPERATOR)
    assert r.status_code == 200
    assert mc.worker_wildcard_state() == {}


def test_wildcard_is_surfaced_on_reads_never_persisted(gated_client):
    mc.set_worker_wildcard("wk-wild", True)

    # GET map stays open (read tier, like the roster it mirrors)
    r = gated_client.get("/llm/workers/wildcard")
    assert r.status_code == 200 and r.get_json() == {"wk-wild": True}

    rows = gated_client.get("/llm/workers").get_json()
    by_id = {w.get("id"): w for w in rows}
    assert by_id["wk-wild"].get("wildcard") is True
    assert by_id["wk-plain"].get("wildcard") is False

    one = gated_client.get("/llm/workers/wk-wild").get_json()
    assert one.get("wildcard") is True

    # response-copy only: the flag is never baked onto the stored record
    assert "wildcard" not in (W.worker_store._load().get("wk-wild") or {})
