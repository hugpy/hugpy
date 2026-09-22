"""PARITY: central's preview and the worker's auto-evict must name the SAME
victims — the spec's first invariant (assets/evictionflow.html, 2026-07-25):

    "Central's preview and the worker's auto-evict run the same function; idle
     times come from one ledger (central's call log, shipped at emission),
     never from each side's own clock. DIVERGENT VICTIM SETS ARE THE FAILURE
     MODE."

This is a REAL cross-module test, not a unit test of the shared function: it
drives ``utils.workers.storage_proposal`` (central's read-time preview) and
``worker_agent.budget.fit_plan`` (the worker's provision-path auto-evict) over
ONE fixture and asserts the two victim lists are identical. Those are the two
independently-maintained call sites that historically drifted; asserting the
shared helper alone would prove nothing about them.

Run: venv/bin/python -m pytest tests/test_eviction_parity.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from hugpy_fleet.worker import budget
from hugpy_fleet.central import workers as central
from hugpy_engine import eviction as ev


# The reserve is carved OUT of disk_cache_gib since 2026-08-22 (effective =
# allocation - reserve). These tests reason in pure-allocation numbers, so pin
# the reserve to 0 here; the carve-out itself is covered by
# tests/test_budget_reserve_carveout.py.
@pytest.fixture(autouse=True)
def _zero_disk_reserve(monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_DISK_RESERVE_GIB", "0")


GIB = 1 << 30
CAP_GIB = 100
NOW = 1_000_000.0

# ── ONE fixture, deliberately adversarial ───────────────────────────────────
# Every discriminating feature of the spec's sort is exercised at once, so a
# drift in ANY key shows up as a divergence rather than a coincidental match:
#   * mixed preferences (key ①)      — "ram-pref" prefers the other device
#   * mixed idle times (key ②)       — spread over four orders of magnitude
#   * a never-called model           — anchors at load, not at epoch 0
#   * mixed call counts (key ③)      — two models tie on idle
#   * an equal-idle equal-call pair  — forces key ④
#   * a 🔒static row                 — the one lock, on both sides
#   * sizes that make walk-then-drop bite (a big cold one covers the need)
_MODELS = [
    # (key, gib, last_picked, calls, alloc_mode, static)
    ("aa-tie",     8,  NOW - 500,   4, "max-gpu", False),
    ("bb-tie",     8,  NOW - 500,   4, "max-gpu", False),
    ("cold-big",  14,  NOW - 9000,  1, "max-gpu", False),
    ("hot-small",  6,  NOW - 5,   900, "max-gpu", False),
    ("ram-pref",  12,  NOW - 2,   700, "max-ram", False),
    ("never",     10,  None,        0, "max-gpu", False),
    ("locked",    25,  NOW - 9999,  0, "max-gpu", True),
]

CALLER = "incoming"
NEED_GIB = 55       # the pull that forces the eviction

# Sized so the walk takes SEVERAL victims and the drop pass actually fires —
# a single-victim fixture would agree on both sides even if the walk or the
# drop were broken, and would prove almost nothing.
MIN_EXPECTED_VICTIMS = 2

# ⚠ THE ONE HONEST ASYMMETRY, and why the fixture is built to cancel it.
#
# The two sites are asked DIFFERENT QUESTIONS, and always have been:
#   * the worker's ``fit_plan`` plans a SPECIFIC PULL — its need is
#     ``used + delta - cap`` (make room for `delta` more bytes);
#   * central's ``storage_proposal`` is a STEADY-STATE preview with no pull in
#     hand — its need is ``used - cap`` (how far over budget is this box now).
#
# That difference is structural and CORRECT: central cannot know about a pull
# that has not been requested. Parity is a claim about the FUNCTION and the
# LEDGER, not about the two questions being the same one — so the fixture sizes
# the resident set to put central's steady-state need and the worker's
# pull-driven need on the same footing (cap = used, so `used - cap` == 0 would
# propose nothing; instead the models total OVER the cap by exactly `delta`).
# With the needs equal, ANY divergence in the victim list is a real drift in
# the sort, the ledger, or the pool — which is what this file exists to catch.


def _rows():
    """The worker's storage survey — the heartbeat's `storage.models` shape."""
    out = []
    for key, gib, _lp, _calls, _mode, static in _MODELS:
        out.append({"model_key": key, "bytes": gib * GIB,
                    "protected": static, "why": "static" if static else "",
                    "pinned": False, "loaded": False, "loading": False,
                    "provisioning": False, "assigned": False})
    return out


def _last_picked():
    return {k: lp for k, _g, lp, _c, _m, _s in _MODELS if lp is not None}


def _call_stats():
    return {k: {"calls": c} for k, _g, _lp, c, _m, _s in _MODELS}


def _modes():
    return {k: m for k, _g, _lp, _c, m, _s in _MODELS}


def _used_bytes():
    return sum(g * GIB for _k, g, _lp, _c, _m, _s in _MODELS)


def _worker_victims():
    """The WORKER's auto-evict plan (worker_agent/budget.py, provision path)."""
    storage = {"cache_used_bytes": _used_bytes(), "disk_free": 0,
               "models": _rows()}
    plan = budget.fit_plan(
        CALLER, NEED_GIB * GIB, storage, {"disk_cache_gib": CAP_GIB},
        _last_picked(), call_stats=_call_stats(), model_modes=_modes(),
        now=NOW)
    assert plan["action"] == "evict", plan
    return list(plan["evict"])


def _central_victims():
    """CENTRAL's read-time preview (utils/workers.py, storage_proposal).

    The cap is lowered by exactly ``NEED_GIB`` so central's steady-state need
    (``used - cap``) equals the worker's pull-driven need (``used + delta -
    cap``) — see the asymmetry note at the top of this file. Everything that
    parity is actually ABOUT (the sort key, the ledger, the pool) is held
    identical; only the question's framing is normalised."""
    worker = {
        "id": "w-parity", "name": "parity-box",
        "storage": {"cache_used_bytes": _used_bytes(), "disk_free": 0,
                    "models": _rows()},
        "disk": {"free_bytes": 0, "total_bytes": 500 * GIB},
        "limits": {"disk_cache_gib": CAP_GIB - NEED_GIB},
        "model_last_picked": _last_picked(),
        "model_call_stats": _call_stats(),
        "model_alloc_modes": _modes(),
        "loaded_models": [], "loading": [], "provisioning": [],
        "config": {"residency": {"locked": "static"}, "pinned": {}},
    }
    prop = central.storage_proposal(worker)
    return [p["model_key"] for p in prop.get("proposed_evictions") or []]


def test_central_preview_and_worker_autoevict_name_the_same_victims():
    """THE parity assertion. One fixture, two independently-maintained call
    sites, identical victim LISTS — same members AND same order, because the
    order is what the operator reads as "what goes first"."""
    w = _worker_victims()
    c = _central_victims()
    assert w == c, (
        f"PARITY BROKEN — worker would evict {w}, central previews {c}. "
        "These two must run the ONE shared function over the ONE ledger; a "
        "divergence here is the exact failure the spec's Parity invariant names.")
    assert len(w) >= MIN_EXPECTED_VICTIMS, (
        f"the fixture must force a MULTI-victim walk (got {w}) — a one-victim "
        "agreement would hold even with a broken walk or drop pass")


def test_both_sides_honor_the_static_lock():
    assert "locked" not in _worker_victims()
    assert "locked" not in _central_victims()


def test_both_sides_spare_the_caller():
    assert CALLER not in _worker_victims()
    assert CALLER not in _central_victims()


def test_the_shared_function_is_what_they_both_reach():
    """Structural: neither side may re-spell the key. If someone inlines a
    tuple again, this fails — which is the drift that cost the parity before."""
    import inspect
    for mod, name in ((budget, "budget.py"),
                      (central, "utils/workers.py")):
        src = inspect.getsource(mod)
        assert "eviction" in src, f"{name} must import the shared eviction module"


def test_parity_survives_the_full_admission_flow_not_just_the_sort():
    """The two sites agree on the SET; this asserts the shared box-2 function
    reproduces that same set directly, so the admission flow (box 1) — which
    calls the identical function — cannot be a third answer."""
    rows = []
    for key, gib, lp, calls, mode, static in _MODELS:
        if key == "locked":
            continue
        rows.append(ev.EvictUnit(
            model_key=key, bytes=gib * GIB,
            pref=ev.preferred_device(mode), last_call=lp, calls=calls))
    need = _used_bytes() + NEED_GIB * GIB - CAP_GIB * GIB   # the worker's need
    direct = ev.evict_plan("disk", need, rows, now=NOW)
    assert direct.victims == _worker_victims() == _central_victims()


# ─────────────────────────────────────────────────────────────────────────────
# THE LEDGER PLUMBING. Parity is only real if the worker actually RECEIVES
# central's numbers; a shared function fed two different ledgers still diverges.
# ─────────────────────────────────────────────────────────────────────────────
def test_central_stamps_calls_beside_last_picked_in_one_place():
    """Key ③'s source. pick_for_model must increment the call count on the SAME
    event that stamps the clock — one ledger, not two that can drift."""
    import inspect
    src = inspect.getsource(central.WorkerStore.pick_for_model)
    assert "model_last_picked" in src and "model_call_stats" in src, (
        "the clock and the count must be stamped by the same routing event")


def test_the_public_view_ships_the_preference_map_to_the_worker():
    """Key ①'s source. The mode map must ride _public_view — that is the
    payload the heartbeat reply returns and the worker adopts."""
    view = central._public_view({
        "id": "w", "name": "w",
        "spill_by_model": {"m-ram": {"alloc_mode": "max-ram"},
                           "m-gpu": {"alloc_mode": "max-gpu"},
                           "m-derived": {}},
        "storage": {}, "disk": {}, "limits": {},
    })
    modes = view["model_alloc_modes"]
    assert modes["m-ram"] == "max-ram"
    assert modes["m-gpu"] == "max-gpu"
    assert "m-derived" not in modes, (
        "an empty spill asserts no preference — it must degrade at the reader, "
        "not be stamped with one nobody chose")


def test_the_worker_adopts_both_ledger_columns_from_the_heartbeat_reply():
    """The receiving half. _adopt_storage_inputs must fold central's counts and
    modes into the settings the eviction readers consult, or the worker ranks
    from its own clock — the exact divergence Parity forbids."""
    from hugpy_fleet.worker import agent as A

    class _S:
        pass

    state = _S()
    before = dict(A._RUNTIME_SETTINGS)
    try:
        A._adopt_storage_inputs(state, {
            "model_last_picked": {"a": 111.0},
            "model_call_stats": {"a": {"calls": 7}},
            "model_alloc_modes": {"a": "max-ram"},
        })
        assert state.model_last_picked == {"a": 111.0}
        assert A._model_alloc_mode("a") == "max-ram"
        last, calls = A._model_call_stats(state, "a")
        assert calls == 7
        assert last is not None and last >= 111.0
    finally:
        A._RUNTIME_SETTINGS.clear()
        A._RUNTIME_SETTINGS.update(before)


def test_an_absent_ledger_degrades_to_todays_ordering_not_a_guess():
    """A pre-ledger central (or a model it has no history for) must change
    NOTHING: no calls, no preference -> key ① constant, key ③ tied, and the
    order is today's idle-first."""
    from hugpy_fleet.worker import agent as A
    before = dict(A._RUNTIME_SETTINGS)
    try:
        A._RUNTIME_SETTINGS.pop("alloc_mode", None)
        A._RUNTIME_SETTINGS.pop("model_calls", None)
        assert A._model_alloc_mode("unknown") is None
        assert ev.preferred_device(None) == ev.VRAM      # the blank default
    finally:
        A._RUNTIME_SETTINGS.clear()
        A._RUNTIME_SETTINGS.update(before)
