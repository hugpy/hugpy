"""Slice 4 — per-drive min-wins budget (operator ruling, 2026-07-17).

"Worker designation beats out central, UNLESS central designation for that
worker's drive is lower than the worker designation." → the effective per-drive
storage budget is min(worker-declared, central-declared); the stricter bound
governs. Rationale: "the real issue for the workers is overcrowding of the HDD."

ae's shape: central disk_cache_gib=400, worker HUGPY_HOT_CACHE_GIB=1500, and the
hot root == the store root (same drive). Before this slice the pull gate read
only central's 400 while the hot tier admitted against 1500 on the SAME dir, so
promotes drove the drive to 702G. After: effective cap = 400 governs both, and
the heartbeat reports the number + its sources.

Same-drive detection is by st_dev (device id), the reliable signal — a symlinked
or bind-mounted hot root sharing the store's filesystem shares its st_dev, where
a realpath-prefix compare would miss it.

Run: venv/bin/python -m pytest tests/test_budget_min_wins.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import budget as B
from hugpy_engine.serve import hot_cache as _HC

GIB = 1 << 30

# Capture the GENUINE hot_cache budget functions, resilient to collection order.
# test_hot_cache.py is a script-style module that reassigns hot_cache globals at
# IMPORT time; if it is collected before us, _HC._budget_bytes is already a test
# lambda by the time we import. So resolve the real function robustly: prefer the
# live module attr when it is genuinely from hot_cache, else test_hot_cache's own
# import-time capture (_REAL_BUDGET), else re-exec the source function object.
import types as _types  # noqa: E402


# The reserve is carved OUT of disk_cache_gib since 2026-08-22 (effective =
# allocation - reserve). These tests reason in pure-allocation numbers, so pin
# the reserve to 0 here; the carve-out itself is covered by
# tests/test_budget_reserve_carveout.py.
@pytest.fixture(autouse=True)
def _zero_disk_reserve(monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_DISK_RESERVE_GIB", "0")



def _real_hc_budget():
    cur = _HC._budget_bytes
    if isinstance(cur, _types.FunctionType) and cur.__module__ == _HC.__name__:
        return cur
    try:
        import test_hot_cache as _THC       # its _REAL_BUDGET is the import-time real
        if isinstance(_THC._REAL_BUDGET, _types.FunctionType) \
                and _THC._REAL_BUDGET.__module__ == _HC.__name__:
            return _THC._REAL_BUDGET
    except Exception:  # noqa: BLE001
        pass
    return cur


_REAL_HC_OWN_BUDGET = _HC._own_budget_bytes


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every worker budget knob so each test declares only what it means to."""
    for k in ("HUGPY_HOT_CACHE_ROOT", "HUGPY_HOT_CACHE_GIB",
              "HUGPY_MODEL_CACHE", "HUGPY_MODEL_CACHE_MAX_GIB",
              "_HUGPY_CENTRAL_DISK_CACHE_GIB"):
        monkeypatch.delenv(k, raising=False)
    # Point HUGPY_MODEL_CACHE at a NONEXISTENT dir so its default tier stays OFF
    # unless a test opts in — the default /var/cache/hugpy-models may exist on the
    # runner and would otherwise inject a phantom 450 term.
    monkeypatch.setenv("HUGPY_MODEL_CACHE", "/nonexistent/hugpy-model-cache-xyz")
    # test_hot_cache.py is a script-style module that reassigns hot_cache module
    # globals (_budget_bytes, _free_bytes, _min_residency_s) at IMPORT time and
    # can leave them patched when collection ends, depending on order. Snapshot
    # this module's OWN import-time capture of the real _budget_bytes and restore
    # it so our hot-tier tests exercise the genuine function; monkeypatch reverts
    # after each test.
    from hugpy_engine.serve import hot_cache as HC
    monkeypatch.setattr(HC, "_budget_bytes", _real_hc_budget())
    monkeypatch.setattr(HC, "_own_budget_bytes", _REAL_HC_OWN_BUDGET)
    return monkeypatch


@pytest.fixture
def store(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    return str(d)


# ── the min is picked from whichever side is stricter ───────────────────────
def test_central_lower_than_worker_central_governs(clean_env, store):
    clean_env.setenv("HUGPY_HOT_CACHE_ROOT", store)     # same drive as store
    clean_env.setenv("HUGPY_HOT_CACHE_GIB", "1500")
    cap, src = B.resolve_effective_cap({"disk_cache_gib": 400}, store)
    assert cap == 400 * GIB                             # ae's exact case
    assert src["effective_source"] == "central_gib"
    assert src["central_gib"] == 400.0
    assert src["worker_hot_cache_gib"] == 1500.0
    assert src["effective_gib"] == 400.0


def test_worker_lower_than_central_worker_governs(clean_env, store):
    clean_env.setenv("HUGPY_HOT_CACHE_ROOT", store)
    clean_env.setenv("HUGPY_HOT_CACHE_GIB", "150")
    cap, src = B.resolve_effective_cap({"disk_cache_gib": 400}, store)
    assert cap == 150 * GIB
    assert src["effective_source"] == "worker_hot_cache_gib"


def test_only_central_declared(clean_env, store):
    cap, src = B.resolve_effective_cap({"disk_cache_gib": 400}, store)
    assert cap == 400 * GIB
    assert src["effective_source"] == "central_gib"
    assert "worker_hot_cache_gib" not in src           # hot root unset -> tier off


def test_only_worker_declared(clean_env, store):
    clean_env.setenv("HUGPY_HOT_CACHE_ROOT", store)
    clean_env.setenv("HUGPY_HOT_CACHE_GIB", "300")
    cap, src = B.resolve_effective_cap({}, store)       # no central limit
    assert cap == 300 * GIB
    assert src["effective_source"] == "worker_hot_cache_gib"
    assert "central_gib" not in src


def test_neither_declared_is_unmanaged(clean_env, store):
    cap, src = B.resolve_effective_cap({}, store)
    assert cap is None                                  # decision D — unmanaged
    assert "effective_gib" not in src


# ── same-drive vs different-drive ───────────────────────────────────────────
def test_worker_tier_on_a_different_drive_does_not_constrain(clean_env, store, monkeypatch):
    """A hot tier on a DIFFERENT drive has its own space and must NOT drag its
    budget into the store drive's min. Faked via st_dev so no second mount is
    needed."""
    other = tempfile.mkdtemp()
    clean_env.setenv("HUGPY_HOT_CACHE_ROOT", other)
    clean_env.setenv("HUGPY_HOT_CACHE_GIB", "50")       # would win IF same-drive
    # Force the hot root onto a different device id than the store root.
    real_drive_id = B._drive_id

    def _fake(path):
        if os.path.abspath(path).startswith(os.path.abspath(other)):
            return 999999                               # a distinct fake device
        return real_drive_id(path)
    monkeypatch.setattr(B, "_drive_id", _fake)
    cap, src = B.resolve_effective_cap({"disk_cache_gib": 400}, store)
    assert cap == 400 * GIB                             # central still governs
    assert "worker_hot_cache_gib" not in src            # different drive -> excluded


def test_worker_tier_on_the_same_drive_does_constrain(clean_env, store):
    clean_env.setenv("HUGPY_HOT_CACHE_ROOT", store)     # same fs as store
    clean_env.setenv("HUGPY_HOT_CACHE_GIB", "50")
    cap, src = B.resolve_effective_cap({"disk_cache_gib": 400}, store)
    assert cap == 50 * GIB
    assert src["effective_source"] == "worker_hot_cache_gib"


# ── fit_plan honours the effective cap + reports sources ────────────────────
def _storage(models, used=None, disk_free=10 * GIB):
    return {"cache_used_bytes": used if used is not None
            else sum(m["bytes"] for m in models),
            "models": models, "disk_free": disk_free}


def _m(key, gib, **f):
    row = {"model_key": key, "bytes": int(gib * GIB), "protected": False,
           "why": "", "pinned": False, "loaded": False, "loading": False,
           "provisioning": False, "assigned": False}
    row.update(f)
    return row


def test_fit_plan_uses_the_supplied_effective_cap_over_central():
    """fit_plan given effective_cap=400 must gate on 400 even though central's
    limits say 1500 — the min already resolved by the caller wins."""
    storage = _storage([_m("cold", 300)], used=420 * GIB)
    plan = B.fit_plan("newmodel", 30 * GIB, storage,
                      {"disk_cache_gib": 1500},          # central says 1500...
                      {"cold": 1},
                      effective_cap=400 * GIB,           # ...but min-wins is 400
                      budget_sources={"central_gib": 400, "worker_hot_cache_gib": 1500,
                                      "effective_gib": 400, "effective_source": "central_gib"})
    # 420 used + 30 delta = 450 > 400 -> over budget -> evict the cold candidate.
    assert plan["action"] == "evict"
    assert plan["evict"] == ["cold"]
    assert plan["budget_effective_bytes"] == 400 * GIB
    assert plan["budget_sources"]["effective_source"] == "central_gib"


def test_fit_plan_refusal_names_the_governing_source():
    storage = _storage([_m("static_big", 40, protected=True, why="static")],
                       used=40 * GIB)
    plan = B.fit_plan("huge", 500 * GIB, storage, {"disk_cache_gib": 1500},
                      {}, effective_cap=400 * GIB,
                      budget_sources={"central_gib": 400, "worker_hot_cache_gib": 1500,
                                      "effective_gib": 400, "effective_source": "central_gib"})
    assert plan["action"] == "refuse"
    r = plan["reason"]
    assert r["budget_effective_bytes"] == 400 * GIB
    assert r["budget_sources"]["effective_source"] == "central_gib"
    assert "effective cap" in r["reason"]               # names WHY the number governs


def test_fit_plan_without_effective_cap_is_byte_identical_to_before():
    """Backward-compat: no effective_cap -> falls back to cap_bytes(limits)."""
    storage = _storage([_m("cold", 40), _m("warm", 10)])
    a = B.fit_plan("caller", 30 * GIB, storage, {"disk_cache_gib": 50},
                   {"cold": 1, "warm": 999})
    assert a["action"] == "evict"
    assert a["evict"] == ["cold"]


# ── shared-store still exempt: min-wins must not resurrect a cap there ───────
def test_shared_store_skips_the_cap_regardless_of_min_wins():
    """On a shared/central store the cap is not-applicable (slice 2). Passing an
    effective_cap must NOT re-enable a cap gate there — proceed, no evict."""
    catalog = [_m("resident", 693, protected=True,
                  why="shared/central storage — never reaped")]
    storage = _storage(catalog, used=693 * GIB, disk_free=4800 * GIB)
    plan = B.fit_plan("newmodel", 91 * GIB, storage, {"disk_cache_gib": 400},
                      {}, shared_store=True,
                      effective_cap=400 * GIB,
                      budget_sources={"central_gib": 400, "effective_gib": 400})
    assert plan["action"] == "proceed"
    assert plan["evict"] == []
    assert "not applicable" in plan["note"]


# ── hot-tier admission consults the effective cap when same-drive ───────────
def test_hot_cache_budget_is_floored_by_central_when_same_drive(clean_env, store):
    from hugpy_engine.serve import hot_cache as HC
    clean_env.setenv("HUGPY_HOT_CACHE_ROOT", store)
    clean_env.setenv("HUGPY_HOT_CACHE_GIB", "1500")
    clean_env.setenv("_HUGPY_CENTRAL_DISK_CACHE_GIB", "400")
    # MODELS_HOME resolves under the store drive for the same-drive check.
    clean_env.setattr(HC, "_models_home", lambda: store)
    assert HC._own_budget_bytes() == 1500 * GIB
    assert HC._budget_bytes() == 400 * GIB              # floored by central 400


def test_hot_cache_budget_unfloored_on_a_different_drive(clean_env, store, monkeypatch):
    from hugpy_engine.serve import hot_cache as HC
    other = tempfile.mkdtemp()
    clean_env.setenv("HUGPY_HOT_CACHE_ROOT", other)
    clean_env.setenv("HUGPY_HOT_CACHE_GIB", "1500")
    clean_env.setenv("_HUGPY_CENTRAL_DISK_CACHE_GIB", "400")
    clean_env.setattr(HC, "_models_home", lambda: store)
    # Force different device ids for the two roots.
    def _fake_same(a, b):
        return False
    monkeypatch.setattr(HC, "_same_drive", _fake_same)
    assert HC._budget_bytes() == 1500 * GIB             # different drive -> own budget


def test_hot_cache_budget_own_when_no_central_term(clean_env, store):
    from hugpy_engine.serve import hot_cache as HC
    clean_env.setenv("HUGPY_HOT_CACHE_ROOT", store)
    clean_env.setenv("HUGPY_HOT_CACHE_GIB", "1500")
    clean_env.setattr(HC, "_models_home", lambda: store)
    # _HUGPY_CENTRAL_DISK_CACHE_GIB unset -> no floor.
    assert HC._budget_bytes() == 1500 * GIB


# ── central passes the effective-budget fields through verbatim ─────────────
def test_effective_budget_survives_storage_proposal():
    from hugpy_fleet.central.workers import storage_proposal
    out = storage_proposal({
        "storage": {"cache_used_bytes": 420 * GIB, "disk_free": 900 * GIB,
                    "models": [],
                    "budget_effective_bytes": 400 * GIB,
                    "budget_sources": {"central_gib": 400, "worker_hot_cache_gib": 1500,
                                       "effective_gib": 400, "effective_source": "central_gib"},
                    "budget_cap_not_applicable": False},
        "disk": {"free_bytes": 900 * GIB, "total_bytes": 1700 * GIB},
        "limits": {"disk_cache_gib": 400},
    })
    assert out["budget_effective_bytes"] == 400 * GIB
    assert out["budget_sources"]["effective_source"] == "central_gib"
    assert out["budget_cap_not_applicable"] is False


def test_effective_budget_degrades_for_a_pre_slice4_worker():
    from hugpy_fleet.central.workers import storage_proposal
    out = storage_proposal({
        "storage": {"cache_used_bytes": 0, "disk_free": GIB, "models": []},
        "disk": {}, "limits": {},
    })
    assert out["budget_effective_bytes"] is None
    assert out["budget_sources"] == {}
    assert out["budget_cap_not_applicable"] is False
