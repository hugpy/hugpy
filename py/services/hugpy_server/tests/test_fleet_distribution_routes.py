"""FLEET DISTRIBUTION MODE (operator ruling 2026-09-24) — the state helper +
the GET/POST routes, plus the per-model ``strict`` round-trip through the
overrides layer that the PlacementControl checkbox drives.

Mode is ``feasible`` (the default) | ``designated``, stored beside the
per-worker wildcard map in the SAME ``worker_wildcard.json`` file (neither
writer clobbers the other). Env ``HUGPY_DISTRIBUTION`` overrides the store. The
GET is open (read tier, like the roster + wildcard map it sits beside); the POST
is operator-gated (operator_auth._SENSITIVE).

Every store here is redirected to a tmp path (isolated_flag_store /
isolated_overrides) so no test touches an operator's live state.

Run: venv/bin/python -m pytest tests/test_fleet_distribution_routes.py -q
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
OV = importlib.import_module("hugpy_engine.serve.overrides")
W = importlib.import_module("hugpy_fleet.central.workers")

OPERATOR = {"X-Operator-Token": "s3cret"}


@pytest.fixture
def isolated_flag_store(tmp_path, monkeypatch):
    """The distribution key + wildcard map live in worker_wildcard.json beside
    MODELS_DISCOVERY_PATH; redirect that module global so the test never touches
    a live store, and start from a clean env (no ambient HUGPY_DISTRIBUTION)."""
    monkeypatch.setattr(mc, "MODELS_DISCOVERY_PATH",
                        os.path.join(str(tmp_path), "model_discovery.json"))
    monkeypatch.delenv("HUGPY_DISTRIBUTION", raising=False)
    yield


@pytest.fixture
def isolated_overrides(tmp_path, monkeypatch):
    """A private serve_overrides.json for the strict round-trip."""
    monkeypatch.setattr(OV, "_OVERRIDES_PATH", str(tmp_path / "serve_overrides.json"))
    yield


@pytest.fixture
def gated_client(isolated_flag_store, monkeypatch):
    """Worker blueprint + operator gate (open mode + a token => the token is
    REQUIRED on sensitive routes; fleet reads stay open) over an isolated
    registry."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    app.config["TESTING"] = True
    with swap_worker_store(prefix="hugpy-dist-workers-"):
        yield app.test_client()


# ── state helper ─────────────────────────────────────────────────────────────

def test_default_absent_is_feasible(isolated_flag_store):
    assert mc.fleet_distribution_mode() == "feasible"
    assert mc.fleet_distribution_status() == {
        "mode": "feasible", "source": "default", "env_override": False,
        "stored": None}


def test_set_reports_store_source(isolated_flag_store):
    assert mc.set_fleet_distribution("designated") == {"distribution": "designated"}
    st = mc.fleet_distribution_status()
    assert st == {"mode": "designated", "source": "store", "env_override": False,
                  "stored": "designated"}
    assert mc.fleet_distribution_mode() == "designated"


def test_bad_mode_raises(isolated_flag_store):
    with pytest.raises(ValueError):
        mc.set_fleet_distribution("sideways")
    with pytest.raises(ValueError):
        mc.set_fleet_distribution("")


def test_env_overrides_store(isolated_flag_store, monkeypatch):
    mc.set_fleet_distribution("designated")
    monkeypatch.setenv("HUGPY_DISTRIBUTION", "feasible")
    st = mc.fleet_distribution_status()
    assert st["mode"] == "feasible"        # env wins over the store
    assert st["source"] == "env"
    assert st["env_override"] is True
    assert st["stored"] == "designated"    # the store still shows the preference
    assert mc.fleet_distribution_mode() == "feasible"


def test_distribution_and_wildcard_share_file_without_clobber(isolated_flag_store):
    mc.set_worker_wildcard("wk-a", True)
    mc.set_fleet_distribution("designated")
    assert mc.worker_wildcard_state() == {"wk-a": True}
    assert mc.fleet_distribution_mode() == "designated"
    # a later wildcard write must not drop the sibling distribution key
    mc.set_worker_wildcard("wk-b", True)
    assert mc.fleet_distribution_mode() == "designated"
    assert mc.worker_wildcard_state() == {"wk-a": True, "wk-b": True}


# ── routes ───────────────────────────────────────────────────────────────────

def test_get_route_open_and_default(gated_client):
    r = gated_client.get("/llm/fleet/distribution")
    assert r.status_code == 200
    assert r.get_json() == {"mode": "feasible", "source": "default",
                            "env_override": False, "stored": None}


def test_post_route_operator_gated_and_persists(gated_client):
    # unauthed -> refused, nothing stored
    r = gated_client.post("/llm/fleet/distribution", json={"mode": "designated"})
    assert r.status_code == 401
    assert mc.fleet_distribution_mode() == "feasible"

    r = gated_client.post("/llm/fleet/distribution", json={"mode": "designated"},
                          headers=OPERATOR)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["mode"] == "designated"
    assert body["source"] == "store"
    assert mc.fleet_distribution_mode() == "designated"

    # GET (open) reflects it
    assert gated_client.get("/llm/fleet/distribution").get_json()["mode"] == "designated"


def test_post_bad_mode_is_400(gated_client):
    r = gated_client.post("/llm/fleet/distribution", json={"mode": "sideways"},
                          headers=OPERATOR)
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert body["error"]["code"] == "BadValue"


def test_post_reports_env_override_but_still_stores(gated_client, monkeypatch):
    monkeypatch.setenv("HUGPY_DISTRIBUTION", "feasible")
    r = gated_client.post("/llm/fleet/distribution", json={"mode": "designated"},
                          headers=OPERATOR)
    assert r.status_code == 200
    body = r.get_json()
    assert body["env_override"] is True
    assert body["mode"] == "feasible"       # env pins the effective mode
    assert body["source"] == "env"
    assert body["stored"] == "designated"   # the write was stored anyway


# ── per-model strict (overrides round-trip; the PlacementControl checkbox) ────

def test_strict_round_trip_and_clears(isolated_overrides):
    assert OV.model_strict("DAN-8B") is False
    OV.set_override("DAN-8B", {"strict": True})
    assert OV.get_override("DAN-8B").get("strict") is True
    assert OV.model_strict("DAN-8B") is True
    # OFF clears the key (real fences only, like the no_evict sibling)
    OV.set_override("DAN-8B", {"strict": False})
    assert "strict" not in OV.get_override("DAN-8B")
    assert OV.model_strict("DAN-8B") is False


def test_strict_bare_vs_namespaced_key(isolated_overrides):
    # written on the bare key, read for the owner-qualified key (the ~ tolerance
    # model_strict shares with placement_policy)
    OV.set_override("DAN-8B", {"strict": True})
    assert OV.model_strict("owner~DAN-8B") is True
