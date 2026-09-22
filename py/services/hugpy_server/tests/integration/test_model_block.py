"""Model-level BLOCK from the serving pool (operator pool primitive).

An operator can BLOCK a model_key (and UNBLOCK it). A blocked model is removed
from the serving pool GLOBALLY: never a routing candidate, never (re)assigned,
never warmed/provisioned, never a fallback default — while its files and existing
designation rows are left untouched (inert). Block outranks pin (pin is routing
persistence; block is an operator override) and a direct call fails fast with a
distinct honest refusal. Persisted in the F4 settings store (survives restart).

This regresses every enforcement point WITHOUT a live worker/fleet:
  * the blocklist registry (block/is_blocked/blocked_keys/reason/info/unblock);
  * remote.py honest-refusal vocabulary (permanent marker + _blocked_reason);
  * routing candidate selection (workers_for_model / pick / candidates → []);
  * the no-worker diagnostic (explain_no_worker → distinct blocked reason);
  * reconcile warm-set ∩ not-blocked + _kick_warm provisioning filter;
  * the /assign route → 409; the /block + /unblock routes; operator-gating;
  * the placement/feasibility preview → "blocked" reason, not fake-infeasible;
  * /models + /v1/models listings mark blocked:true;
  * the brain/default fallback ladder (_reconcile_default) skips a blocked one.

Isolation: the blocklist persists through the F4 settings store, so
``HUGPY_SETTINGS_PATH`` is pointed at a tmp file per test. The engine reads the
blocklist through the ``hugpy_engine.placement`` seam (a NullBlocklist unless
the fleet is installed), so the fleet's ``FleetBlocklist`` adapter is installed
per test (conftest restores the seam). The /assign route reaches the REAL
module-level assign_model(), so the worker registry is swapped for a
tmpdir-backed store (see worker_store_isolation.py).
"""
import importlib

import pytest
from flask import Flask

from worker_store_isolation import swap_worker_store

bl = importlib.import_module("hugpy_fleet.central.blocklist")
placement = importlib.import_module("hugpy_engine.placement")
fleet_placement = importlib.import_module("hugpy_fleet.central.placement")
remote = importlib.import_module("hugpy_engine.resolvers.remote")
W = importlib.import_module("hugpy_fleet.central.workers")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
cr = importlib.import_module("hugpy_server.app.routes.comms_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")
lsr = importlib.import_module("hugpy_server.app.routes.llm_storage_routes")
v1 = importlib.import_module("hugpy_server.app.routes.v1_routes")
md = importlib.import_module("hugpy_engine.config.models.models_default")


def _reset():
    for k in list(bl.blocked_keys()):
        bl.unblock(k)


@pytest.fixture(autouse=True)
def blocklist(tmp_path, monkeypatch):
    """An empty blocklist on a tmp settings.json, wired into the engine seam."""
    monkeypatch.setenv("HUGPY_SETTINGS_PATH", str(tmp_path / "settings.json"))
    bl.settings_store._cache = None
    placement.set_blocklist(fleet_placement.FleetBlocklist())
    _reset()
    yield bl
    _reset()
    bl.settings_store._cache = None


# ── 1) the blocklist registry ────────────────────────────────────────────────

def test_registry_block_unblock_lifecycle():
    assert bl.is_blocked("M~1") is False
    assert bl.block_reason("M~1") is None
    rec = bl.block("M~1", note="dueling coder-next")
    assert bl.is_blocked("M~1") is True
    # record shape {blocked,by,ts,note} (extensible)
    assert rec["blocked"] is True and rec["by"] == "operator"
    assert isinstance(rec["ts"], float) and rec["note"] == "dueling coder-next"
    assert bl.blocked_keys() == {"M~1"}
    assert (bl.block_info("M~1") or {}).get("note") == "dueling coder-next"
    # block_reason is distinct + carries the marker
    assert bl.BLOCKED_MARKER in (bl.block_reason("M~1") or "")
    assert "by the operator" in bl.block_reason("M~1")
    # block is idempotent (still one key)
    assert (bl.block("M~1") or bl.is_blocked("M~1")) and bl.blocked_keys() == {"M~1"}
    # None/empty key never blocked
    assert bl.is_blocked("") is False and bl.is_blocked(None) is False
    assert bl.unblock("M~1") is True
    assert bl.is_blocked("M~1") is False
    assert bl.block_info("M~1") is None
    # unblock again is a no-op (was_blocked False)
    assert bl.unblock("M~1") is False


def test_registry_block_persists_across_fresh_store():
    """Persistence survives a fresh SettingsStore instance (== a central restart)."""
    from hugpy_control.settings import SettingsStore
    bl.block("Persist~Me")
    fresh = SettingsStore()
    assert bool(fresh.get(bl.NS, "Persist~Me", {}).get("blocked"))


# ── 2) remote.py honest-refusal vocabulary ───────────────────────────────────

def test_remote_marker_is_permanent_and_blocked_reason_flows():
    assert any("blocked from the serving pool" in m for m in remote._PERMANENT_LOAD_MARKERS)
    bl.block("M~2")
    # classified as PERMANENT (fail fast, no retry/hold)
    assert remote._is_permanent_load_error(bl.block_reason("M~2")) is True
    assert bool(remote._blocked_reason("M~2"))
    assert remote._blocked_reason("M~unblocked") is None


# ── 3) routing candidate selection + 4) no-worker diagnostic ─────────────────

SERVER = {"id": "w1", "name": "box", "status": "online", "admission": "approved",
          "models": ["M"], "loaded_models": [], "grants": {}, "pool": ""}


@pytest.fixture
def routing(monkeypatch):
    """A worker that WOULD serve "M" — gate helpers stubbed to pass so the ONLY
    variable under test is the operator block."""
    with swap_worker_store() as store:
        monkeypatch.setattr(store, "all", lambda: [dict(SERVER)])
        monkeypatch.setattr(W, "_engine_unusable", lambda w: False)
        monkeypatch.setattr(W, "_worker_env_tier", lambda w: "stable")
        monkeypatch.setattr(W, "env_tier_for_model", lambda mk: "stable")
        monkeypatch.setattr(W, "_task_capable", lambda w, t: True)
        yield store


def test_routing_block_removes_all_candidates_and_unblock_reverts(routing):
    store = routing
    assert [w["id"] for w in store.workers_for_model("M")] == ["w1"]
    assert (store.pick_for_model("M") or {}).get("id") == "w1"

    bl.block("M")
    assert store.workers_for_model("M") == []
    assert store.pick_for_model("M") is None
    assert store.candidates_for_model("M") == []
    # explain_no_worker returns the DISTINCT blocked reason
    assert bl.BLOCKED_MARKER in W.explain_no_worker("M")

    bl.unblock("M")
    assert [w["id"] for w in store.workers_for_model("M")] == ["w1"]
    # no manufactured reason after unblock
    assert W.explain_no_worker("M") == ""


# ── 5) reconcile warm-set ∩ not-blocked + _kick_warm provisioning filter ─────

# Warm set = (⭐ star ∪ 🔒static) ∩ on-disk − blocked (operator RULINGS 2026-07-23;
# warm_whitelist / task-defaults NO LONGER warm). Use 🔒static for both members so
# this exercises the ∩ not-blocked guard without needing to inject a star.
WK = {"id": "wk", "name": "box", "url": "http://192.0.2.1:9100",
      "models_local": ["A", "B"],
      "config": {"residency": {"A": "static", "B": "static"}}}


def test_reconcile_warm_set_and_kick_warm_filter_blocked():
    assert wr._reconcile_warm_set(WK) == ["A", "B"]
    bl.block("A")
    assert wr._reconcile_warm_set(WK) == ["B"]
    # _kick_warm filters blocked keys — all-blocked schedules NOTHING
    assert wr._kick_warm({"id": "wk", "url": "http://192.0.2.1:9100"}, ["A"], "test") == []
    bl.block("B")
    assert wr._kick_warm({"id": "wk", "url": "http://192.0.2.1:9100"}, ["A", "B"], "test") == []
    _reset()
    assert wr._reconcile_warm_set(WK) == ["A", "B"]


# ── 6) the routes: assign 409, block/unblock, placement, gating ──────────────

@pytest.fixture
def route_client(monkeypatch):
    monkeypatch.setattr(wr, "get_models_dict", lambda dict_return=False: {"M": {}})
    monkeypatch.setattr(wr, "_transfer_authorized", lambda: True)
    monkeypatch.setattr(cr, "audit", lambda *a, **k: None)
    monkeypatch.setattr(wr, "_central_missing_reason", lambda mk: None)
    monkeypatch.setattr(wr, "_disk_preflight_reason", lambda w, mk: None)
    monkeypatch.setattr(wr, "get_worker",
                        lambda wid: {"id": wid, "name": "box", "models_local": []})
    # The /assign route calls the REAL, unstubbed module-level assign_model()
    # once "M" is unblocked — swap the module singleton for an isolated store.
    with swap_worker_store():
        app = Flask(__name__)
        app.register_blueprint(wr.worker_bp)
        yield app.test_client()


def test_routes_block_assign_placement_unblock(route_client):
    client = route_client
    r = client.post("/llm/models/M/block", json={"note": "op says"})
    assert r.status_code == 200 and r.get_json()["blocked"] is True
    assert bl.is_blocked("M") is True
    # unknown key → 404
    assert client.post("/llm/models/Nope~Key/block", json={}).status_code == 404

    # assign of a blocked model → 409 with a clear blocked reason
    ra = client.post("/llm/workers/w1/assign", json={"model_key": "M"})
    assert ra.status_code == 409
    assert "blocked from the serving pool" in (ra.get_json() or {}).get("error", "")

    # placement/feasibility preview reports blocked (not fake-infeasible)
    rp = client.get("/llm/models/M/placement")
    pj = rp.get_json()
    assert rp.status_code == 200 and pj.get("blocked") is True
    assert "blocked from the serving pool" in (pj.get("winner_reason") or "")
    assert pj.get("workers") == []

    # /unblock reverts everything
    ru = client.post("/llm/models/M/unblock", json={})
    assert ru.status_code == 200 and ru.get_json()["was_blocked"] is True
    assert bl.is_blocked("M") is False
    # once unblocked, the route falls through to the REAL assign_model() — inside
    # the swapped, tmpdir-backed W.worker_store.
    assert client.post("/llm/workers/w1/assign",
                       json={"model_key": "M"}).status_code != 409


def _gated(path, method):
    p = path
    if p == "/api" or p.startswith("/api/"):
        p = p[len("/api"):] or "/"
    return any(method in methods and rx.match(p) for methods, rx in oa._SENSITIVE)


def test_gating_block_and_unblock_are_operator_gated():
    """Both verbs must be in _SENSITIVE (assign tier)."""
    assert _gated("/llm/models/Some~Key/block", "POST")
    assert _gated("/llm/models/Some~Key/unblock", "POST")
    # gate also matches the /api-mounted path
    assert _gated("/api/llm/models/o/r/g~k/block", "POST")
    # GET placement stays OPEN (read)
    assert _gated("/llm/models/k/placement", "GET") is False
    # slashed model_key still gated (path converter)
    assert _gated("/llm/models/owner/repo~q/block", "POST")


# ── 7) listings mark blocked:true (/models + /v1/models) ─────────────────────

def test_models_listing_marks_blocked_and_clears_stale_record(monkeypatch):
    # A SHARED manifest dict (mutated in place across calls) — mirrors the real
    # cached get_models_dict, so the stale-`block`-key-after-unblock regression
    # is actually reachable here.
    manifest = {"M": {"model_key": "M", "status": "installed"}}
    monkeypatch.setattr(lsr, "get_models_dict", lambda dict_return=False: manifest)
    monkeypatch.setattr(lsr, "update_model_status", lambda m: m)
    monkeypatch.setattr(lsr, "media_default_state", lambda: None)
    monkeypatch.setattr(lsr, "media_state", lambda mk: False)
    monkeypatch.setattr(lsr, "_annotate_gguf_size", lambda m, mk: None)
    monkeypatch.setattr(lsr, "_annotate_size", lambda m, mk: None)
    lapp = Flask(__name__)
    lapp.register_blueprint(lsr.llm_bp)
    lclient = lapp.test_client()

    row = (lclient.get("/models").get_json() or [{}])[0]
    assert row.get("blocked") is False
    bl.block("M")
    row = (lclient.get("/models").get_json() or [{}])[0]
    assert row.get("blocked") is True and (row.get("block") or {}).get("blocked") is True
    bl.unblock("M")
    row = (lclient.get("/models").get_json() or [{}])[0]
    # unblock clears blocked AND the stale block record (cached dict)
    assert row.get("blocked") is False and "block" not in row


def test_v1_models_listing_is_additive_blocked(monkeypatch):
    monkeypatch.setattr(v1, "get_models_dict", lambda dict_return=False: {
        "M": {"status": "installed", "primary_task": "text-generation"}})
    monkeypatch.setattr(v1, "update_model_status", lambda m: m)
    monkeypatch.setattr(v1, "media_default_state", lambda: None)
    monkeypatch.setattr(v1, "api_key_required", lambda: False)
    vapp = Flask(__name__)
    vapp.register_blueprint(v1.v1_bp)
    vclient = vapp.test_client()

    bl.block("M")
    d = vclient.get("/v1/models").get_json()["data"][0]
    assert d.get("blocked") is True and d["id"] == "M"
    bl.unblock("M")
    d = vclient.get("/v1/models").get_json()["data"][0]
    assert d.get("blocked") is False


# ── 8) the brain / default fallback ladder skips a blocked candidate ─────────

def test_brain_default_ladder_skips_blocked(monkeypatch):
    monkeypatch.setattr(md, "_on_disk", lambda cfg: True)   # both stand-ins count as installed
    reg = {"BRAIN": object(), "ALT": object()}
    assert md._reconcile_default("BRAIN", reg) == "BRAIN"
    bl.block("BRAIN")
    # BLOCKED brain is SKIPPED — stands in an installed non-blocked model
    assert md._reconcile_default("BRAIN", reg) == "ALT"
    bl.block("ALT")
    # both blocked → stand-in set empty → configured returned unchanged
    assert md._reconcile_default("BRAIN", reg) == "BRAIN"
    _reset()
    assert md._reconcile_default("BRAIN", reg) == "BRAIN"
