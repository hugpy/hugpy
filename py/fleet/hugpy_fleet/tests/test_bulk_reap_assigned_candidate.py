"""Slice 7 — the bulk-reap path drops `assigned` as a protection reason.

Field incident (0.1.187): the operator approved central's 10-model eviction on
ae. 3 deleted, **7 refused worker-side "assigned"**. Central's proposal chain
already follows the 2026-07-17 ruling (assignment is routing/attribution, NEVER a
disk shield) and proposes assigned-but-cold models — but the worker's bulk reap
(_reap_scan's protected classification + _reap_reclaim's re-prove) still shielded
`assigned` from the old doctrine. So central proposed what the worker refused, and
an over-budget box whose cold models are ALL assigned (the normal case — ae
designates everything) could never be cleared.

This mirrors the call-driven path (budget._is_protected dropped `assigned` at
f1894b2). KEEP protecting: 🔒static, loaded/loading/provisioning (live-use), the
store-reapable gate, the shared/central sentinel gate, and the path jail.

Run: venv/bin/python -m pytest tests/test_bulk_reap_assigned_candidate.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import imports as WI
from hugpy_storage import provision as P


HOT = "/mnt/hot990/hugpy-worker/models/gguf/o/r"


def _cfg(mk):
    return SimpleNamespace(framework="gguf", hub_id="o/r", filename=None,
                           include=None, primary_task="text-generation",
                           tasks=["text-generation"], folder="gguf/o/r")


class _State:
    def __init__(self, assigned=None):
        self.assigned_models = list(assigned or [])
        self._provisioning = []


@pytest.fixture(autouse=True)
def _clear_models_local_cache():
    """_reap_reclaim/_models_local write a PROCESS-GLOBAL cache; clear it around
    every test in this file so a stubbed model here never leaks into another
    file's _reap_scan (local_keys union) via a stale cache entry."""
    A._MODELS_LOCAL_CACHE.update(at=0.0, value=[])
    yield
    A._MODELS_LOCAL_CACHE.update(at=0.0, value=[])


@pytest.fixture
def one_model(monkeypatch):
    """A single on-disk model on a reapable, non-shared store. Residency/loaded
    are overridable per-test via the returned knobs dict."""
    knobs = {"residency": "on-demand", "loaded": set(), "loading": set(),
             "store_reapable": True, "on_shared": False}
    # Neutralize the module-level _models_local cache so this file's keys never
    # leak into another file's scan via _reap_scan's local_keys union (the cache
    # is a process global that survives across tests).
    monkeypatch.setattr(A, "_models_local", lambda s: [])
    A._MODELS_LOCAL_CACHE.update(at=0.0, value=[])
    monkeypatch.setattr(WI, "get_models_dict", lambda: {"M": None})
    monkeypatch.setattr(WI, "get_model_config", _cfg)
    monkeypatch.setattr(WI, "get_model_path", lambda mk: HOT)
    monkeypatch.setattr(P, "model_is_local", lambda mk: True)
    monkeypatch.setattr(P, "_on_shared_model_store", lambda rp: knobs["on_shared"])
    monkeypatch.setattr(P, "_model_store_reapable", lambda rp: knobs["store_reapable"])
    monkeypatch.setattr(A, "_store_root_copy_path", lambda mk, c: HOT)
    monkeypatch.setattr(A, "_path_bytes", lambda p: 100)
    monkeypatch.setattr(A, "loaded_model_keys", lambda: sorted(knobs["loaded"]))
    monkeypatch.setattr(A, "_slot_occupants", lambda *a, **k: set())
    monkeypatch.setattr(A, "_loading_model_keys", lambda: sorted(knobs["loading"]))
    monkeypatch.setattr(A, "_residency", lambda mk: knobs["residency"])
    return knobs


# ── the fix: assigned-but-cold is PROPOSED (scan) and RECLAIMED (executor) ──
def test_assigned_cold_model_is_a_reclaimable_candidate(one_model):
    scan = A._reap_scan(_State(assigned=["M"]))
    assert [r["model_key"] for r in scan["reclaimable"]] == ["M"]
    assert scan["protected"] == []                 # assigned no longer protects


def test_assigned_cold_model_is_actually_reclaimed(one_model, monkeypatch):
    """The executor must delete it, not refuse "assigned". wipe_model stubbed so
    no real IO — the point is _reap_reclaim does NOT short-circuit on assigned."""
    wiped = []
    monkeypatch.setattr(P, "wipe_model",
                        lambda mk, path="": (wiped.append((mk, path)), True)[1])
    res = A._reap_reclaim(_State(assigned=["M"]), ["M"])
    assert res["ok"] is True
    assert res["results"][0]["ok"] is True         # deleted, not refused
    assert res["results"][0]["reason"] == ""
    assert wiped == [("M", HOT)]                    # hot copy targeted


def test_preview_and_executor_agree_on_assigned(one_model, monkeypatch):
    """The slice-7 invariant: what the scan proposes, the executor reclaims — no
    'central proposes what the worker refuses' gap for a bare-assigned model."""
    monkeypatch.setattr(P, "wipe_model", lambda mk, path="": True)
    scan = A._reap_scan(_State(assigned=["M"]))
    proposed = [r["model_key"] for r in scan["reclaimable"]]
    res = A._reap_reclaim(_State(assigned=["M"]), proposed)
    assert proposed == ["M"]
    assert all(r["ok"] for r in res["results"])


# ── KEEP: the live-use + durable-presence + gate protections still hold ─────
def test_static_still_protected_in_scan_and_reclaim(one_model, monkeypatch):
    one_model["residency"] = "static"
    scan = A._reap_scan(_State(assigned=["M"]))
    assert scan["reclaimable"] == []
    assert scan["protected"][0]["why"] == "static"
    monkeypatch.setattr(P, "wipe_model", lambda mk, path="": True)
    res = A._reap_reclaim(_State(assigned=["M"]), ["M"])
    assert res["results"][0]["ok"] is False
    assert res["results"][0]["reason"] == "static"


def test_loaded_still_protected_in_scan_and_reclaim(one_model, monkeypatch):
    one_model["loaded"] = {"M"}
    scan = A._reap_scan(_State(assigned=["M"]))
    assert scan["reclaimable"] == []
    assert scan["protected"][0]["why"] == "loaded"
    monkeypatch.setattr(P, "wipe_model", lambda mk, path="": True)
    res = A._reap_reclaim(_State(assigned=["M"]), ["M"])
    assert res["results"][0]["ok"] is False
    assert "loaded" in res["results"][0]["reason"]


def test_provisioning_still_refused_by_the_executor(one_model, monkeypatch):
    """provisioning is guarded at reclaim time (a mid-pull delete corrupts the
    fetch) — independent of assignment, must still refuse."""
    monkeypatch.setattr(P, "wipe_model", lambda mk, path="": True)
    st = _State(assigned=["M"])
    st._provisioning = ["M"]
    res = A._reap_reclaim(st, ["M"])
    assert res["results"][0]["ok"] is False
    assert res["results"][0]["reason"] == "provisioning"


def test_store_gated_still_protected(one_model):
    """model store not marked reapable -> still protected with an honest reason."""
    one_model["store_reapable"] = False
    one_model["on_shared"] = False
    scan = A._reap_scan(_State(assigned=["M"]))
    assert scan["reclaimable"] == []
    assert scan["protected"][0]["why"] == "model store not marked reapable"


def test_shared_sentinel_still_protected(one_model):
    """shared/central storage -> the sentinel gate still shields it (the NAS is
    the fleet SoT; never deletable regardless of assignment)."""
    one_model["store_reapable"] = False
    one_model["on_shared"] = True
    scan = A._reap_scan(_State(assigned=["M"]))
    assert scan["reclaimable"] == []
    assert scan["protected"][0]["why"] == "shared/central storage — never reaped"


# ── the scan's protected breakdown no longer counts bare `assigned` ─────────
def test_protected_breakdown_never_counts_bare_assigned(one_model, monkeypatch):
    """A whole box of assigned-but-cold models: ALL reclaimable, NONE protected,
    and the `why` vocabulary never emits a bare 'assigned' protection reason."""
    monkeypatch.setattr(WI, "get_models_dict",
                        lambda: {"A": None, "B": None, "C": None})
    scan = A._reap_scan(_State(assigned=["A", "B", "C"]))
    got = sorted(r["model_key"] for r in scan["reclaimable"])
    assert {"A", "B", "C"}.issubset(set(got))      # all three are candidates
    assert scan["protected"] == []                 # none protected
    whys = {r["why"] for r in scan["protected"]}
    assert "assigned" not in whys                  # vocabulary stays honest


# ── the heartbeat row reports assigned as attribution, not protection ───────
def test_storage_model_row_assigned_is_attribution_not_protected(monkeypatch):
    monkeypatch.setattr(A, "_pinned", lambda mk: False)
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    row = A._storage_model_row("m", 100, set(), set(), set(), {"m"})
    assert row["assigned"] is True             # attribution still reported
    assert row["protected"] is False           # but NOT a protection
    assert row["why"] == "assigned"            # attribution label
    # and budget agrees it's a candidate
    from hugpy_fleet.worker import budget
    assert budget._is_protected(row) == ""


def test_storage_model_row_store_gate_still_beats_assigned(monkeypatch):
    """The store gate is a hard filesystem fact and must still outrank the
    assigned attribution label (an assigned+store-gated row stays protected)."""
    monkeypatch.setattr(A, "_pinned", lambda mk: False)
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    row = A._storage_model_row("m", 100, set(), set(), set(), {"m"},
                               why_hint="model store not marked reapable")
    assert row["protected"] is True
    assert row["why"] == "model store not marked reapable"
