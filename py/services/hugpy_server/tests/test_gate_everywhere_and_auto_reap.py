"""Slice 8 — Part A (every transfer entry runs the budget gate) and Part B
(opt-in, heartbeat-driven auto-reap).

Part A field incident (0.1.187): unsloth~Qwen3-Coder-Next chunks were downloading
onto ae's over-budget store while a DIFFERENT pid was actively REFUSING the same
model (an ungated transfer entry — /redownload and the slot child's
_ensure_present pulled without a state, so the gate never ran). Fix: the gate is
now UNCONDITIONAL — ensure_model_present resolves a budget state itself (the
agent's live state in-process, else a minimal env/disk state) when the caller
passes state=None, so no path downloads atop the cap. Resume semantics: fit_plan
already counts on-disk bytes as headroom, so an admitted/partial pull asks only
for its remaining delta and never re-refuses itself into a wedge.

Part B (operator ask 2026-07-17 "there needs to be a way to auto approve this"):
per-worker auto_reap, default FALSE. On heartbeat ingest, an opted-in worker that
is over budget with a non-empty proposal fires EXACTLY the guarded reap-approve
flow (recompute → intersect → audit 'worker.auto-reap' → guarded relay), once per
cooldown. No new timer/daemon; the worker re-prove chain is untouched.

Run: venv/bin/python -m pytest tests/test_gate_everywhere_and_auto_reap.py -q
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-slice8-"))

from hugpy_storage import provision as P
from hugpy_server.app.routes import comms_routes as _comms_routes


# The reserve is carved OUT of disk_cache_gib since 2026-08-22 (effective =
# allocation - reserve). These tests reason in pure-allocation numbers, so pin
# the reserve to 0 here; the carve-out itself is covered by
# tests/test_budget_reserve_carveout.py.
@pytest.fixture(autouse=True)
def _zero_disk_reserve(monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_DISK_RESERVE_GIB", "0")



# ═══════════════════════ Part A — gate everywhere ═══════════════════════════
class _State:
    """Minimal explicit WorkerState for the gate."""
    def __init__(self, assigned=None, limits=None):
        self.assigned_models = list(assigned or [])
        self._provisioning = []
        self.model_last_picked = {}
        self.allocated = {}
        self.refused = {}
        self.limits = dict(limits or {})


@pytest.fixture
def gate_spy(monkeypatch):
    """Stub the transfer + gate collaborators so we can OBSERVE that the gate ran
    (evict_to_fit called) before any pull, without touching a disk or central."""
    calls = {"evict": [], "provision": []}

    monkeypatch.setattr(P, "ensure_model_registered", lambda mk, url: mk)
    monkeypatch.setattr(P, "model_is_local", lambda mk: False)
    monkeypatch.setattr(P, "central_total_bytes", lambda url, mk: 50 * (1 << 30))
    monkeypatch.setattr(P, "_provision_now",
                        lambda mk, url, progress=None: (calls["provision"].append(mk), True)[1])
    # A single-flight lock that always acquires.
    import threading
    monkeypatch.setattr(P, "_provision_lock", lambda mk: threading.Lock())

    from hugpy_fleet.worker import budget as B

    def _fake_evict(state, mk, need):
        calls["evict"].append({"mk": mk, "need": need,
                               "limits": dict(getattr(state, "limits", {}) or {})})
    monkeypatch.setattr(B, "evict_to_fit", _fake_evict)
    P.set_budget_state(None)                    # clean slate each test
    yield calls
    P.set_budget_state(None)


def test_explicit_state_caller_gates(gate_spy):
    """The already-gated path (demand) still gates."""
    P.ensure_model_present("M", "http://central", state=_State(limits={"disk_cache_gib": 400}))
    assert gate_spy["evict"] and gate_spy["evict"][0]["mk"] == "M"
    assert gate_spy["provision"] == ["M"]       # pull ran AFTER the gate


def test_stateless_caller_still_gates_via_registered_state(gate_spy):
    """The formerly-UNGATED path (state=None, e.g. /redownload before the fix):
    with the agent's live state registered, the gate runs against the REAL
    limits — no more download atop the cap."""
    P.set_budget_state(_State(assigned=["x"], limits={"disk_cache_gib": 400}))
    P.ensure_model_present("M", "http://central")   # NO state passed
    assert gate_spy["evict"], "the gate MUST run even without a caller state"
    assert gate_spy["evict"][0]["limits"] == {"disk_cache_gib": 400}


def test_stateless_caller_gates_via_minimal_state_when_none_registered(gate_spy, monkeypatch):
    """The slot child (separate process, no registered state) still gates through
    a minimal env/disk state — the cap comes from the env projection (slice 4)."""
    monkeypatch.setenv("_HUGPY_CENTRAL_DISK_CACHE_GIB", "250")
    P.set_budget_state(None)                    # nothing registered (fresh process)
    P.ensure_model_present("M", "http://central")
    assert gate_spy["evict"], "a state-less process must still gate"
    assert gate_spy["evict"][0]["limits"] == {"disk_cache_gib": 250.0}


def test_a_refusal_stops_the_pull(gate_spy, monkeypatch):
    """When the gate REFUSES (BudgetRefusal), the pull must never start."""
    from hugpy_fleet.worker import budget as B

    def _refuse(state, mk, need):
        raise B.BudgetRefusal({"state": "refused", "reason": "won't fit"})
    monkeypatch.setattr(B, "evict_to_fit", _refuse)
    with pytest.raises(B.BudgetRefusal):
        P.ensure_model_present("M", "http://central", state=_State(limits={"disk_cache_gib": 1}))
    assert gate_spy["provision"] == []          # NO bytes moved


def test_admitted_resume_does_not_self_refuse():
    """Resume semantics (real fit_plan, no stubs): a pull already ON DISK as a
    partial only needs its REMAINING delta, so it re-passes the gate for free and
    is never refused into a wedge.

    Setup isolates the have-credit: cap 60; a 30 GiB cold model + M half-present
    (25 of 50). The RESUME asks only for delta=25 (have 25 credited), so
    used(55)+25=80 needs 20 freed and the 30 GiB cold candidate covers it → the
    resume EVICTS and proceeds. A FRESH 50 GiB pull of a NOT-yet-present M would
    instead ask for the full 50 (used 30 + 50 = 80, still coverable here) — the
    point being that the partial's delta is smaller, so a resume is strictly
    EASIER to seat than its first admission and can never refuse itself."""
    from hugpy_fleet.worker import budget as B
    GIB = 1 << 30

    def _row(mk, gib, assigned=False):
        return {"model_key": mk, "bytes": gib * GIB, "protected": False,
                "why": "", "pinned": False, "loaded": False, "loading": False,
                "provisioning": False, "assigned": assigned}

    storage = {"cache_used_bytes": 55 * GIB, "disk_free": 100 * GIB,
               "models": [_row("M", 25, assigned=True), _row("cold", 30)]}
    plan = B.fit_plan("M", 50 * GIB, storage, {"disk_cache_gib": 60}, {"cold": 1})
    # delta = 50 - 25(have) = 25, NOT 50 — the partial bytes are credited so the
    # resume is seatable (evict the cold 30) instead of refusing into a wedge.
    assert plan["action"] == "evict"
    assert plan["evict"] == ["cold"]
    assert "M" not in plan["evict"]            # never evicts the keep-target

    # And a strictly-smaller remaining delta PROCEEDS with no eviction at all:
    storage2 = {"cache_used_bytes": 45 * GIB, "disk_free": 100 * GIB,
                "models": [_row("M", 45, assigned=True)]}   # 45 of 50 present
    plan2 = B.fit_plan("M", 50 * GIB, storage2, {"disk_cache_gib": 60}, {})
    # delta = 5; used(45)+5 = 50 <= 60 -> proceed, resume never self-refuses.
    assert plan2["action"] == "proceed"
    assert plan2["evict"] == []


# ═══════════════════════ Part B — auto-reap ═════════════════════════════════
@pytest.fixture
def wr(monkeypatch):
    """The worker_routes module with a Flask test client + stubbed collaborators
    so the heartbeat/auto-reap flow runs without a live worker or relay."""
    import importlib
    from flask import Flask, jsonify
    m = importlib.import_module(
        "hugpy_server.app.routes.worker_routes")

    relay = []
    audits = []
    stamps = []

    def _fake_relay(worker_id, op_path, body, timeout, action, retry_on_connect=False):
        relay.append({"op_path": op_path, "body": dict(body), "action": action})
        return jsonify({"ok": True, "freed_bytes": 999,
                        "results": [{"model_key": k, "ok": True}
                                    for k in body["model_keys"]]}), 200

    monkeypatch.setattr(m, "_relay_worker_op", _fake_relay)
    monkeypatch.setattr(m, "record_worker_auto_reap",
                        lambda wid, when: stamps.append((wid, when)))
    # audit lives in comms_routes; _execute_reap does `from .comms_routes import
    # audit` at call time, so patch the source attribute.
    monkeypatch.setattr(_comms_routes, "audit",
                        lambda ev, payload: audits.append((ev, payload)))

    app = Flask(__name__)
    app.register_blueprint(m.worker_bp)
    return type("H", (), {"m": m, "client": app.test_client(),
                          "relay": relay, "audits": audits, "stamps": stamps})()


PROPOSAL = {"over_budget": True,
            "proposed_evictions": [{"model_key": "a", "bytes": 10, "last_picked": 1.0},
                                   {"model_key": "b", "bytes": 20, "last_picked": 2.0}]}


def test_auto_reap_off_fires_nothing(wr, monkeypatch):
    """Default posture: auto_reap false -> the heartbeat check does nothing, even
    over budget with a proposal (today's behavior, byte-identical)."""
    monkeypatch.setattr(wr.m, "worker_storage_view", lambda wid: dict(PROPOSAL))
    worker = {"id": "w", "name": "box", "auto_reap": False}
    wr.m._maybe_auto_reap("w", worker)
    assert wr.relay == []
    assert wr.audits == []
    assert wr.stamps == []


def test_auto_reap_on_fires_the_guarded_flow_once(wr, monkeypatch):
    monkeypatch.setattr(wr.m, "worker_storage_view", lambda wid: dict(PROPOSAL))
    worker = {"id": "w", "name": "box", "auto_reap": True}
    wr.m._maybe_auto_reap("w", worker)
    # relayed to the SAME guarded /reap executor, with the proposal's own keys.
    assert len(wr.relay) == 1
    assert wr.relay[0]["op_path"] == "/reap"
    assert wr.relay[0]["body"]["model_keys"] == ["a", "b"]
    # audited distinctly as auto (not operator-approved).
    assert wr.audits and wr.audits[0][0] == "worker.auto-reap"
    assert wr.audits[0][1]["trigger"] == "auto"
    # cooldown stamped.
    assert wr.stamps and wr.stamps[0][0] == "w"


def test_auto_reap_cooldown_suppresses_immediate_refire(wr, monkeypatch):
    monkeypatch.setattr(wr.m, "worker_storage_view", lambda wid: dict(PROPOSAL))
    # last fire was 1s ago; default cooldown 300s -> suppressed.
    worker = {"id": "w", "name": "box", "auto_reap": True,
              "last_auto_reap_at": time.time() - 1.0}
    wr.m._maybe_auto_reap("w", worker)
    assert wr.relay == []


def test_auto_reap_fires_after_cooldown_elapses(wr, monkeypatch):
    monkeypatch.setattr(wr.m, "worker_storage_view", lambda wid: dict(PROPOSAL))
    worker = {"id": "w", "name": "box", "auto_reap": True,
              "last_auto_reap_at": time.time() - 10_000.0}   # long ago
    wr.m._maybe_auto_reap("w", worker)
    assert len(wr.relay) == 1


def test_auto_reap_not_over_budget_fires_nothing(wr, monkeypatch):
    monkeypatch.setattr(wr.m, "worker_storage_view",
                        lambda wid: {"over_budget": False, "proposed_evictions": []})
    worker = {"id": "w", "name": "box", "auto_reap": True}
    wr.m._maybe_auto_reap("w", worker)
    assert wr.relay == []


def test_auto_reap_empty_proposal_fires_nothing(wr, monkeypatch):
    monkeypatch.setattr(wr.m, "worker_storage_view",
                        lambda wid: {"over_budget": True, "proposed_evictions": []})
    worker = {"id": "w", "name": "box", "auto_reap": True}
    wr.m._maybe_auto_reap("w", worker)
    assert wr.relay == []


def test_auto_reap_intersects_stale_keys(wr, monkeypatch):
    """Blast-radius guard: if the live proposal no longer includes a key (since
    protected), the auto-fire drops it — never widens beyond the current need."""
    live = {"over_budget": True,
            "proposed_evictions": [{"model_key": "a", "bytes": 10, "last_picked": 1.0}]}
    monkeypatch.setattr(wr.m, "worker_storage_view", lambda wid: dict(live))
    worker = {"id": "w", "name": "box", "auto_reap": True}
    wr.m._maybe_auto_reap("w", worker)
    assert wr.relay[0]["body"]["model_keys"] == ["a"]   # only the still-proposed key


# ── the operator route still works via the shared core (no divergence) ──────
def test_operator_reap_approve_still_uses_the_shared_core(wr, monkeypatch):
    monkeypatch.setattr(wr.m, "get_worker",
                        lambda wid: {"id": wid, "name": "box"} if wid == "w" else None)
    monkeypatch.setattr(wr.m, "worker_storage_view", lambda wid: dict(PROPOSAL))
    r = wr.client.post("/llm/workers/w/reap-approve", json={"model_keys": ["a", "z"]})
    body = r.get_json()
    assert r.status_code == 200
    assert wr.relay[0]["body"]["model_keys"] == ["a"]   # z dropped (not proposed)
    assert body["reaped"] == ["a"] and body["dropped"] == ["z"]
    assert body["trigger"] == "operator"                # distinct from auto
    assert wr.audits[0][0] == "worker.reap-approve"


# ── the set-auto-reap route is operator-gated + persists ────────────────────
def test_set_auto_reap_route(wr, monkeypatch):
    stored = {}

    def _set(wid, enabled):
        if wid != "w":
            return None
        stored["auto_reap"] = enabled
        return {"id": wid, "auto_reap": enabled}
    monkeypatch.setattr(wr.m, "set_worker_auto_reap", _set)
    r = wr.client.post("/llm/workers/w/auto-reap", json={"enabled": True})
    assert r.status_code == 200
    assert stored["auto_reap"] is True
    assert r.get_json()["auto_reap"] is True
    # unknown worker -> 404
    assert wr.client.post("/llm/workers/nope/auto-reap",
                          json={"enabled": True}).status_code == 404


# ── the storage payload carries the mode (console visibility) ───────────────
def test_storage_proposal_carries_auto_reap_mode():
    from hugpy_fleet.central.workers import storage_proposal
    GIB = 1 << 30
    out = storage_proposal({
        "auto_reap": True, "last_auto_reap_at": 1234.5,
        "storage": {"cache_used_bytes": 500 * GIB, "disk_free": 100 * GIB,
                    "models": []},
        "disk": {"free_bytes": 100 * GIB}, "limits": {"disk_cache_gib": 400},
    })
    assert out["auto_reap"] is True
    assert out["last_auto_reap_at"] == 1234.5
    # default posture for a worker that never opted in
    out2 = storage_proposal({
        "storage": {"cache_used_bytes": 0, "disk_free": GIB, "models": []},
        "disk": {}, "limits": {}})
    assert out2["auto_reap"] is False
    assert out2["last_auto_reap_at"] is None
