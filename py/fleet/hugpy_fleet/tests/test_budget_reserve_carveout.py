"""The reserve is carved OUT of disk_cache_gib (operator rule, 2026-08-22).

``disk_cache_gib`` IS the allocation = the total limit; the inference headroom
(``limits.disk_reserve_gib`` -> ``HUGPY_WORKER_DISK_RESERVE_GIB`` -> 50) is
portioned out of it, not added beside it. effective ceiling = allocation -
reserve on BOTH planners (worker fit_plan / resolve_effective_cap, central
storage_proposal) — Parity — and ``budget_sources`` names the split.
"""
import pytest

from hugpy_fleet.worker import budget as B
from hugpy_fleet.central import workers as W

GIB = 1 << 30


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for k in ("HUGPY_HOT_CACHE_ROOT", "HUGPY_HOT_CACHE_GIB", "HUGPY_MODEL_CACHE_MAX_GIB",
              "_HUGPY_CENTRAL_DISK_CACHE_GIB", "_HUGPY_CENTRAL_DISK_RESERVE_GIB",
              "HUGPY_WORKER_DISK_RESERVE_GIB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HUGPY_MODEL_CACHE", "/nonexistent/hugpy-model-cache-xyz")
    d = tmp_path / "store"
    d.mkdir()
    return str(d)


def _m(key, gib):
    return {"model_key": key, "bytes": int(gib * GIB), "protected": False, "why": "",
            "pinned": False, "loaded": False, "loading": False,
            "provisioning": False, "assigned": False}


# ── reserve resolution order ─────────────────────────────────────────────────
def test_reserve_default_is_50(clean_env):
    assert B.disk_reserve_bytes() == 50 * GIB
    assert W._disk_reserve_bytes() == 50 * GIB


def test_reserve_env_overrides_default(clean_env, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_DISK_RESERVE_GIB", "80")
    assert B.disk_reserve_bytes() == 80 * GIB
    assert W._disk_reserve_bytes() == 80 * GIB


def test_per_worker_limit_beats_env_on_both_sides(clean_env, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_DISK_RESERVE_GIB", "80")
    lim = {"disk_cache_gib": 1180, "disk_reserve_gib": 150}
    assert B.disk_reserve_bytes(lim) == 150 * GIB
    assert W._disk_reserve_bytes(lim) == 150 * GIB


# ── worker: allocation - reserve ─────────────────────────────────────────────
def test_resolve_effective_cap_carves_the_reserve_out(clean_env):
    cap, src = B.resolve_effective_cap({"disk_cache_gib": 1180, "disk_reserve_gib": 150},
                                       clean_env)
    assert cap == 1030 * GIB                       # ae: 1180 - 150
    assert src["allocation_gib"] == 1180.0
    assert src["allocation_source"] == "central_gib"
    assert src["reserve_gib"] == 150.0
    assert src["effective_gib"] == 1030.0
    assert src["effective_source"] == "central_gib"


def test_min_wins_then_carve_out(clean_env, monkeypatch):
    monkeypatch.setenv("HUGPY_HOT_CACHE_ROOT", clean_env)
    monkeypatch.setenv("HUGPY_HOT_CACHE_GIB", "400")
    cap, src = B.resolve_effective_cap({"disk_cache_gib": 1180}, clean_env)
    assert src["allocation_gib"] == 400.0 and src["allocation_source"] == "worker_hot_cache_gib"
    assert cap == 350 * GIB and src["effective_gib"] == 350.0


def test_reserve_larger_than_allocation_floors_at_zero(clean_env):
    cap, src = B.resolve_effective_cap({"disk_cache_gib": 20}, clean_env)
    assert cap == 0 and src["effective_gib"] == 0.0


def test_fit_plan_fallback_applies_the_same_carve_out(clean_env):
    # 100 allocated, 50 reserve -> 50 effective. used 40 + 20 = 60 > 50 -> evict.
    storage = {"cache_used_bytes": 40 * GIB, "disk_free": 500 * GIB,
               "models": [_m("cold", 40)]}
    plan = B.fit_plan("new", 20 * GIB, storage, {"disk_cache_gib": 100}, {"cold": 1})
    assert plan["action"] == "evict" and plan["evict"] == ["cold"]
    assert plan["budget_effective_bytes"] == 50 * GIB
    assert plan["budget_sources"]["allocation_gib"] == 100.0
    assert plan["budget_sources"]["reserve_gib"] == 50.0
    assert plan["budget_sources"]["effective_gib"] == 50.0


def test_fit_plan_disk_free_floor_drives_eviction(clean_env):
    # Under the ceiling (10 + 20 <= 50) but the real volume would dip below the
    # reserve. The CALLED MODEL WINS (operator, 2026-08-31): the disk-free
    # deficit routes into the SAME FIFO, so the oldest model is EVICTED to make
    # physical room rather than the pull being refused.
    #   cap = 100 - 50 = 50; cap_deficit = 10+20-50 = 0 (under ceiling)
    #   disk_deficit = (reserve 50 + delta 20) - free 60 = 10 GiB
    #   cold is 10 GiB -> freed 10 >= must_free 10 -> evict.
    storage = {"cache_used_bytes": 10 * GIB, "disk_free": 60 * GIB,
               "models": [_m("cold", 10)]}
    plan = B.fit_plan("new", 20 * GIB, storage, {"disk_cache_gib": 100}, {"cold": 1})
    assert plan["action"] == "evict" and plan["evict"] == ["cold"]


def test_fit_plan_refuses_only_when_disk_full_and_nothing_reclaimable(clean_env):
    # Same disk-free block, but the only model on disk is PROTECTED (loaded), so
    # a full FIFO frees nothing. THIS is the genuine physical floor — the Errno
    # 28 guard — and it still refuses honestly, naming the disk as the cause.
    hot = _m("hot", 10)
    hot["loaded"] = True
    storage = {"cache_used_bytes": 10 * GIB, "disk_free": 60 * GIB, "models": [hot]}
    plan = B.fit_plan("new", 20 * GIB, storage, {"disk_cache_gib": 100}, {"hot": 1})
    assert plan["action"] == "refuse" and plan["evict"] == []
    assert plan["reason"]["disk_reserve_bytes"] == 50 * GIB
    assert plan["reason"]["disk_deficit_bytes"] == 10 * GIB
    assert "volume physically full" in plan["reason"]["reason"]


# ── central: same ceiling, same sources ──────────────────────────────────────
def _worker(used_gib, free_gib, limits):
    return {"id": "w", "name": "w",
            "storage": {"cache_used_bytes": used_gib * GIB, "disk_free": free_gib * GIB,
                        "models": [_m("cold", used_gib)]},
            "disk": {"free_bytes": free_gib * GIB, "total_bytes": 1180 * GIB},
            "limits": limits, "model_last_picked": {"cold": 1.0},
            "loaded_models": [], "loading": [], "provisioning": [],
            "config": {"residency": {}, "pinned": {}}}


def test_central_budget_is_allocation_minus_reserve(clean_env):
    out = W.storage_proposal(_worker(1050, 130, {"disk_cache_gib": 1180, "disk_reserve_gib": 150}))
    assert out["budget_basis"] == "cap"
    assert out["allocation_bytes"] == 1180 * GIB
    assert out["reserve"] == 150 * GIB
    assert out["budget"] == 1030 * GIB
    assert out["over_budget"] is True and out["need_bytes"] == 20 * GIB
    assert [p["model_key"] for p in out["proposed_evictions"]] == ["cold"]
    src = out["budget_sources"]                  # central's own map (no worker report)
    assert (src["allocation_gib"], src["reserve_gib"], src["effective_gib"]) == (1180.0, 150.0, 1030.0)


def test_central_under_ceiling_is_not_over_budget(clean_env):
    out = W.storage_proposal(_worker(1000, 180, {"disk_cache_gib": 1180, "disk_reserve_gib": 150}))
    assert out["over_budget"] is False and out["proposed_evictions"] == []


def test_central_free_space_floor_still_trips_in_cap_mode(clean_env):
    # cache well under the ceiling, but something else ate the disk.
    out = W.storage_proposal(_worker(100, 10, {"disk_cache_gib": 1180, "disk_reserve_gib": 150}))
    assert out["budget_basis"] == "cap"
    assert out["over_budget"] is True and out["need_bytes"] == 140 * GIB


def test_stale_worker_report_never_governs_central_still_carves_out(clean_env):
    """The ae case: a pre-carve-out agent reports effective == allocation (1180).
    Central must still show/gate on 1180 - 150 = 1030."""
    w = _worker(1050, 130, {"disk_cache_gib": 1180, "disk_reserve_gib": 150})
    w["storage"]["budget_effective_bytes"] = 1180 * GIB
    w["storage"]["budget_sources"] = {"central_gib": 1180.0, "effective_gib": 1180.0,
                                      "effective_source": "central_gib"}
    out = W.storage_proposal(w)
    assert out["budget_effective_bytes"] == 1030 * GIB
    assert out["budget"] == 1030 * GIB
    assert out["budget_sources"]["effective_gib"] == 1030.0
    assert out["budget_sources"]["reserve_gib"] == 150.0
    assert out["over_budget"] is True and out["need_bytes"] == 20 * GIB
    assert [p["model_key"] for p in out["proposed_evictions"]] == ["cold"]
    assert out["worker_reported_budget_effective_bytes"] == 1180 * GIB   # kept for diagnosis


def test_central_folds_worker_declared_terms_into_the_min(clean_env):
    w = _worker(100, 500, {"disk_cache_gib": 1180, "disk_reserve_gib": 150})
    w["storage"]["budget_sources"] = {"central_gib": 1180.0, "worker_hot_cache_gib": 400.0,
                                      "effective_gib": 400.0, "effective_source": "worker_hot_cache_gib"}
    out = W.storage_proposal(w)
    assert out["allocation_bytes"] == 400 * GIB
    assert out["budget"] == 250 * GIB
    assert out["budget_sources"]["allocation_source"] == "worker_hot_cache_gib"
    assert out["budget_sources"]["worker_hot_cache_gib"] == 400.0


def test_parity_worker_and_central_agree_on_the_ceiling(clean_env):
    lim = {"disk_cache_gib": 1180, "disk_reserve_gib": 150}
    cap, _ = B.resolve_effective_cap(lim, clean_env)
    assert cap == W.storage_proposal(_worker(0, 1180, lim))["budget"]


# ── limits plumbing ──────────────────────────────────────────────────────────
def test_set_limits_accepts_disk_reserve_gib_and_clamp_passes_it(tmp_path, monkeypatch):
    assert "disk_reserve_gib" in W.WorkerStore._LIMIT_KEYS
    out = W._clamp_limits({"disk_cache_gib": 1180.0, "disk_reserve_gib": 150.0},
                          {"ram_max_gib": 64})
    assert out == {"disk_cache_gib": 1180.0, "disk_reserve_gib": 150.0}
    from hugpy_fleet.fleet_manager import templates as T
    assert "disk_reserve_gib" in T._LIMIT_KEYS
