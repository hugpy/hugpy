"""k60 — the SHARED catalog is not in a worker's eviction economy (2026-07-31).

Operator alarm: ae's storage panel read "⚠ over budget · 2.0 TiB over" (2.8 TiB
used / 800 GB budget) on a box whose hot drive is 1.7 TiB. The survey was
counting the SHARED central catalog (/mnt/llm_storage, carrying the
.hugpy-central-catalog sentinel) as this worker's resident cache, pricing it
against the worker budget. Deletion was never actually possible (the store gate,
the sentinel veto, /reap's delete-time re-check) — but "over budget" reads to an
operator as "an auto-delete is coming", and it must never even LOOK that way.

The ruling, locked here:
  1. used / need_bytes count ONLY bytes on REAPABLE stores. A shared/unreapable
     row still SHIPS — tagged `shared` / `unreapable` — but contributes zero.
  2. Proposals may name only reapable rows (worker survey AND central chain).
  3. The disk row reports the drive the MODEL ROOT actually lives on; a second
     volume is a SECOND entry, never a relabel of the first.

NOT covered here on purpose: the delete-time guards. They were correct and are
untouched — see test_reap_store_root_classification.py, which still asserts that
wipe_model refuses a shared path no matter who supplies it.

Run: venv/bin/python -m pytest tests/test_storage_shared_store_accounting.py -q
"""
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import imports as WI
from hugpy_storage import provision as P

W = importlib.import_module(
    "hugpy_fleet.central.workers")

GIB = 1 << 30
HOT_ROOT = "/mnt/hot990/hugpy-worker/models"     # this box's own, reapable store
NAS_ROOT = "/mnt/llm_storage/models"             # the shared/central catalog

SHARED_WHY = "shared/central storage — never reaped"


class _State:
    def __init__(self, assigned=None):
        self.assigned_models = list(assigned or [])
        self._provisioning = []


def _cfg(framework="gguf"):
    return SimpleNamespace(framework=framework, hub_id="Owner/Repo",
                           filename=None, include=None,
                           primary_task="text-generation",
                           tasks=["text-generation"], folder="gguf/Owner/Repo")


# ── the scan fixture the ruling asks for: hot rows + sentinel-root rows ──────
# hot1/hot2 resolve under the worker's own reapable store; nas1/nas2 resolve
# under the sentinel-carrying shared root (a read-through with no hot copy).
_PATHS = {
    "hot1": f"{HOT_ROOT}/gguf/hot1",
    "hot2": f"{HOT_ROOT}/gguf/hot2",
    "nas1": f"{NAS_ROOT}/gguf/nas1",
    "nas2": f"{NAS_ROOT}/gguf/nas2",
}
_SIZES = {"hot1": 30 * GIB, "hot2": 20 * GIB, "nas1": 900 * GIB, "nas2": 700 * GIB}


@pytest.fixture
def two_tier(monkeypatch):
    """A box like ae: a reapable hot store root AND a mounted shared catalog."""
    keys = list(_PATHS)
    monkeypatch.setattr(WI, "get_models_dict", lambda: {k: _cfg() for k in keys})
    monkeypatch.setattr(WI, "get_model_config", lambda mk: _cfg())
    monkeypatch.setattr(WI, "get_model_path", lambda mk: _PATHS[mk])
    monkeypatch.setattr(P, "model_is_local", lambda mk: True)
    # The sentinel lives at/above the NAS root; the hot drive carries none.
    monkeypatch.setattr(P, "_on_shared_model_store",
                        lambda rp: str(rp).startswith("/mnt/llm_storage"))
    # ae's operator set HUGPY_MODEL_STORE_REAPABLE — so reapable iff not shared.
    monkeypatch.setattr(P, "_model_store_reapable",
                        lambda rp: not str(rp).startswith("/mnt/llm_storage"))
    # A hot copy exists only for the hot* keys; nas* are read-through only.
    monkeypatch.setattr(A, "_store_root_copy_path",
                        lambda mk, cfg: _PATHS[mk] if mk.startswith("hot") else "")
    monkeypatch.setattr(A, "_path_bytes",
                        lambda p: next((v for k, v in _SIZES.items()
                                        if p == _PATHS[k]), 0))
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_slot_occupants", lambda *a, **k: set())
    monkeypatch.setattr(A, "_loading_model_keys", lambda: [])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_models_local", lambda s: keys)
    return monkeypatch


def _by_key(rows):
    return {r["model_key"]: r for r in rows}


# ── ruling 1: the scan tags shared rows and prices them at zero ──────────────
def test_scan_tags_shared_rows_and_excludes_them_from_the_budget(two_tier):
    scan = A._reap_scan(_State(assigned=list(_PATHS)))

    # Both NAS rows still SHIP — the console must be able to show what occupies
    # the drive. They are protected, tagged, and NOT budget-bearing.
    prot = _by_key(scan["protected"])
    assert set(prot) == {"nas1", "nas2"}
    for row in prot.values():
        assert row["why"] == SHARED_WHY
        assert row["store"] == "shared"
        assert row["counts_toward_budget"] is False

    # Only the hot copies are candidates — ruling 2 at the worker.
    assert set(_by_key(scan["reclaimable"])) == {"hot1", "hot2"}

    # The budget sees the hot store ONLY: 50 GiB, not 1.6 TiB + 50 GiB.
    assert scan["budgeted_bytes"] == 50 * GIB
    assert scan["unbudgeted_bytes"] == 1600 * GIB
    assert scan["shared_bytes"] == 1600 * GIB
    assert scan["shared_count"] == 2
    # Every considered key still classified — the fix hides nothing.
    assert scan["scan_rows"] == 4


def test_unreapable_store_is_labeled_not_priced(two_tier):
    """The box never opted in (HUGPY_MODEL_STORE_REAPABLE unset). Nothing is
    deletable, so nothing is budget-bearing — but every row still ships, tagged
    `unreapable` rather than `shared` (it is not the central catalog)."""
    two_tier.setattr(P, "_model_store_reapable", lambda rp: False)
    two_tier.setattr(P, "_on_shared_model_store",
                     lambda rp: str(rp).startswith("/mnt/llm_storage"))
    scan = A._reap_scan(_State(assigned=list(_PATHS)))
    prot = _by_key(scan["protected"])
    assert set(prot) == set(_PATHS)
    assert prot["hot1"]["store"] == "unreapable"
    assert prot["hot1"]["why"] == "model store not marked reapable"
    assert prot["nas1"]["store"] == "shared"
    assert scan["reclaimable"] == []
    assert scan["budgeted_bytes"] == 0
    assert scan["unbudgeted_bytes"] == 1650 * GIB


# ── ruling 1: the heartbeat payload prices only the reapable store ───────────
def _storage_stubs(mp):
    mp.setattr(A, "_disk_status", lambda: {"root": HOT_ROOT,
                                           "free_bytes": 400 * GIB,
                                           "total_bytes": 1700 * GIB})
    mp.setattr(A, "_orphan_scan", lambda s, keys: {"items": [], "bytes": 0,
                                                   "count": 0})
    mp.setattr(A, "_hot_cache_status", lambda: {"enabled": False})
    mp.setattr(A, "_refused_snapshot", lambda s: {})
    A._STORAGE_CACHE["value"] = None
    A._STORE_MEASURE_CACHE["value"] = None


def test_worker_storage_used_counts_only_the_reapable_store(two_tier):
    _storage_stubs(two_tier)
    two_tier.setattr(A, "_models_store_root", lambda: HOT_ROOT)
    two_tier.setattr(A, "_measured_store_bytes", lambda: 50 * GIB)
    try:
        out = A._worker_storage(_State(assigned=list(_PATHS)))
    finally:
        A._STORAGE_CACHE["value"] = None

    assert out["cache_used_bytes"] == 50 * GIB          # NOT 1.65 TiB
    assert out["cache_used_model_sum_bytes"] == 50 * GIB
    assert out["unbudgeted_bytes"] == 1600 * GIB
    assert out["unbudgeted_count"] == 2
    assert out["store_root_budgeted"] is True
    # Every row is still on the wire, each carrying its store class.
    rows = _by_key(out["models"])
    assert set(rows) == set(_PATHS)
    assert rows["nas1"]["store"] == "shared"
    assert rows["nas1"]["counts_toward_budget"] is False
    assert rows["nas1"]["protected"] is True
    assert rows["hot1"]["store"] == "reapable"
    assert rows["hot1"]["counts_toward_budget"] is True


def test_shared_store_root_measurement_is_never_the_budget_number(two_tier):
    """The exact ae shape: the store root itself resolves onto the shared
    catalog, so the measured walk returns the WHOLE FLEET's catalog (2.8 TiB).
    That number must never become cache_used_bytes — it is reported under a name
    nothing prices, and used falls back to the reapable-row sum."""
    _storage_stubs(two_tier)
    two_tier.setattr(A, "_models_store_root", lambda: NAS_ROOT)
    two_tier.setattr(A, "_measured_store_bytes", lambda: 2800 * GIB)
    try:
        out = A._worker_storage(_State(assigned=list(_PATHS)))
    finally:
        A._STORAGE_CACHE["value"] = None

    assert out["store_root_budgeted"] is False
    assert out["store_root_shared"] is True
    assert out["cache_used_bytes"] == 50 * GIB          # the hot rows, NOT 2.8 TiB
    assert out["cache_used_measured_bytes"] is None
    assert out["store_root_measured_bytes"] == 2800 * GIB
    assert out["unbudgeted_bytes"] == 1600 * GIB


# ── ruling 3: the disk row names the MODEL ROOT's drive ─────────────────────
def test_disk_row_reports_the_model_root_drive(monkeypatch, tmp_path):
    """DEFAULT_ROOT is not necessarily where the models live. The first entry
    must be the MODEL ROOT's volume — on ae the hot drive, not the NAS the
    read-through happened to resolve to."""
    from hugpy_platform import constants as C
    model_root = tmp_path / "hot" / "models"
    other_root = tmp_path / "shared"
    model_root.mkdir(parents=True)
    other_root.mkdir(parents=True)
    monkeypatch.setattr(A, "_models_store_root", lambda: str(model_root))
    monkeypatch.setattr(C, "DEFAULT_ROOT", str(other_root))
    monkeypatch.setattr(A, "_path_on_shared_store",
                        lambda p: str(p) == str(other_root))

    disk = A._disk_status()
    assert disk["root"] == str(model_root)
    assert disk["tiers"][0]["label"] == "model root"
    assert disk["tiers"][0]["root"] == str(model_root)
    assert disk["shared"] is False
    assert disk["total_bytes"] > 0


def test_second_volume_is_a_second_entry_never_a_relabel(monkeypatch, tmp_path):
    """When DEFAULT_ROOT is on a DIFFERENT volume it is added as a second tier,
    tagged shared — the first entry keeps naming the model root."""
    from hugpy_platform import constants as C
    model_root = tmp_path / "hot" / "models"
    other_root = tmp_path / "shared"
    model_root.mkdir(parents=True)
    other_root.mkdir(parents=True)
    monkeypatch.setattr(A, "_path_device",
                        lambda p: 2 if str(p) == str(other_root) else 1)
    monkeypatch.setattr(A, "_models_store_root", lambda: str(model_root))
    monkeypatch.setattr(C, "DEFAULT_ROOT", str(other_root))
    monkeypatch.setattr(A, "_path_on_shared_store",
                        lambda p: str(p) == str(other_root))

    disk = A._disk_status()
    assert [t["label"] for t in disk["tiers"]] == ["model root", "default root"]
    assert disk["root"] == str(model_root)          # never mislabeled
    assert disk["tiers"][1]["shared"] is True


# ── central: the proposal chain prices its own copy the same way ────────────
def _worker(models, cache_used, cap_gib=800.0, **storage_extra):
    storage = {"reported": True, "cache_used_bytes": cache_used,
               "disk_free": 400 * GIB, "models": models}
    storage.update(storage_extra)
    return {"id": "w1", "name": "ae", "storage": storage,
            "disk": {"free_bytes": 400 * GIB, "total_bytes": 1700 * GIB},
            "limits": {"disk_cache_gib": cap_gib},
            "models": [], "model_last_picked": {}}


def _row(mk, b, **kw):
    row = {"model_key": mk, "bytes": b, "protected": False, "why": "",
           "pinned": False, "loaded": False, "loading": False,
           "provisioning": False, "assigned": True}
    row.update(kw)
    return row


def test_central_excludes_shared_rows_from_used_and_need():
    """The current-wheel shape: the worker already discounted the shared bytes
    and says so. Central must agree — 50 GiB against an 800 GiB cap FITS."""
    models = [_row("hot1", 30 * GIB), _row("hot2", 20 * GIB),
              _row("nas1", 900 * GIB, protected=True, why=SHARED_WHY,
                   store="shared", counts_toward_budget=False),
              _row("nas2", 700 * GIB, protected=True, why=SHARED_WHY,
                   store="shared", counts_toward_budget=False)]
    out = W.storage_proposal(_worker(models, 50 * GIB,
                                     unbudgeted_bytes=1600 * GIB,
                                     shared_bytes=1600 * GIB))
    assert out["cache_used_bytes"] == 50 * GIB
    assert out["over_budget"] is False
    assert out["need_bytes"] == 0
    assert out["proposed_evictions"] == []
    assert out["unbudgeted_bytes"] == 1600 * GIB
    assert out["shared_count"] == 2
    # The gauge must read the same discounted number, never the row sum.
    assert out["gauge_used_bytes"] == 50 * GIB
    assert out["hot_model_bytes"] == 50 * GIB


def test_central_discounts_a_released_workers_shared_bytes():
    """The live-fleet case: a released worker still prices the shared catalog
    into cache_used_bytes. Central discounts it from the store-gate `why` it has
    always sent, so the ae alarm clears at the next central restart rather than
    waiting on a wheel roll. This is the exact ae reading."""
    models = [_row("nas1", 1600 * GIB, protected=True, why=SHARED_WHY),
              _row("nas2", 1200 * GIB, protected=True, why=SHARED_WHY),
              _row("hot1", 84 * GIB)]
    out = W.storage_proposal(_worker(models, 2884 * GIB))
    assert out["cache_used_reported_bytes"] == 2884 * GIB   # what the wire said
    assert out["cache_used_bytes"] == 84 * GIB              # what may be priced
    assert out["over_budget"] is False                      # 84 GiB < 800 GiB
    assert out["need_bytes"] == 0
    assert out["unbudgeted_bytes"] == 2800 * GIB
    rows = _by_key(out["models"])
    assert rows["nas1"]["store"] == "shared"
    assert rows["nas1"]["counts_toward_budget"] is False
    assert rows["hot1"]["counts_toward_budget"] is True


def test_central_proposal_names_only_reapable_rows():
    """Genuinely over budget on the box's OWN store: the proposal fires, but it
    may only ever name hot rows — the shared catalog is not eviction inventory
    however large it is or however cold it looks."""
    models = [_row("hot_cold", 900 * GIB),
              _row("nas_colder", 1600 * GIB, protected=True, why=SHARED_WHY,
                   store="shared", counts_toward_budget=False)]
    w = _worker(models, 900 * GIB, unbudgeted_bytes=1600 * GIB)
    w["model_last_picked"] = {"hot_cold": 2_000_000.0}   # nas has none = colder
    out = W.storage_proposal(w)
    assert out["over_budget"] is True
    # 900 - (800 - 50 reserve carved out of the allocation), not 2.7 TiB
    assert out["need_bytes"] == 150 * GIB
    assert [p["model_key"] for p in out["proposed_evictions"]] == ["hot_cold"]


def test_shared_rows_are_protected_even_if_the_worker_forgot_to_say_so():
    """Belt to the worker's gate: a row tagged shared is protected centrally
    regardless of the flags on the wire, so a stale/rewritten `protected:false`
    can never put the central catalog into a proposal."""
    models = [_row("nas1", 1600 * GIB, store="shared",
                   counts_toward_budget=False),
              _row("hot1", 900 * GIB)]
    out = W.storage_proposal(_worker(models, 900 * GIB,
                                     unbudgeted_bytes=1600 * GIB))
    rows = _by_key(out["models"])
    assert rows["nas1"]["protected"] is True
    assert rows["nas1"]["why"] == SHARED_WHY
    assert all(p["model_key"] != "nas1" for p in out["proposed_evictions"])
