"""Slice 5 — the ae 2026-07-17 data-loss root cause: a hot root that OVERLAPS
MODELS_HOME ('cache of itself').

ae set HUGPY_HOT_CACHE_ROOT == MODEL_HOME (/mnt/hot990/hugpy-worker/models). In
that configuration:
  * hot_path(f) == f (identity), so is_complete is a no-op;
  * _rebuild_index adopts EVERY canonical model dir as a hot 'entry';
  * _make_room's LRU eviction then rmtree's real model weights out of MODELS_HOME
    to hit the budget — silently DELETING the store it was meant to accelerate.
slice-4's 400 GiB min-wins floor made that eviction aggressive against 672 GiB
present, and models_local fell 65 -> 0 at the 0.1.185 flip.

FIX: a degenerate hot root (overlaps MODELS_HOME) disables promotion+eviction —
the tier is INERT, never destructive. Serving is unaffected: use() returns
under-root paths unchanged, so loads still run off MODELS_HOME. status() names
the degenerate case so the heartbeat shows WHY the tier is off.

Also locks the slice-5 scan skip-reason histogram.

Run: venv/bin/python -m pytest tests/test_hot_root_degenerate.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine.serve import hot_cache as HC
from hugpy_fleet.worker import agent as A


def _mk_model(root, rel, name="model.gguf", size=5000):
    d = os.path.join(root, rel)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "wb") as fh:
        fh.write(b"x" * size)
    return d


@pytest.fixture
def hc_env(monkeypatch):
    """Isolated hot_cache module state + deterministic knobs."""
    monkeypatch.setattr(HC, "_free_bytes", lambda: 10 ** 15)
    monkeypatch.setattr(HC, "_min_residency_s", lambda: 0.0)
    with HC._INDEX_LOCK:
        HC._INDEX = {"version": 1, "entries": {}}
        HC._INDEX_LOADED = False
    with HC._STATE_LOCK:
        HC._QUEUED.clear()
    for k in ("HUGPY_HOT_CACHE_ROOT", "HUGPY_HOT_CACHE_GIB",
              "_HUGPY_CENTRAL_DISK_CACHE_GIB"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ── the degenerate configuration is detected and disables the tier ──────────
def test_hot_root_equal_to_models_home_is_degenerate(hc_env, tmp_path):
    store = str(tmp_path / "models")
    os.makedirs(store)
    hc_env.setenv("HUGPY_HOT_CACHE_ROOT", store)
    hc_env.setattr(HC, "_models_home", lambda: store)
    assert HC._degenerate_root() is True
    assert HC.enabled() is False               # promotion/eviction OFF


def test_hot_root_under_models_home_is_degenerate(hc_env, tmp_path):
    store = str(tmp_path / "models")
    hot = os.path.join(store, "hot")
    os.makedirs(hot)
    hc_env.setenv("HUGPY_HOT_CACHE_ROOT", hot)
    hc_env.setattr(HC, "_models_home", lambda: store)
    assert HC._degenerate_root() is True       # overlap (hot inside store)
    assert HC.enabled() is False


def test_separate_hot_root_is_not_degenerate(hc_env, tmp_path):
    store = str(tmp_path / "models")
    hot = str(tmp_path / "hot")
    os.makedirs(store)
    os.makedirs(hot)
    hc_env.setenv("HUGPY_HOT_CACHE_ROOT", hot)
    hc_env.setattr(HC, "_models_home", lambda: store)
    assert HC._degenerate_root() is False
    assert HC.enabled() is True                # a real separate tier still works


def test_sibling_dirs_do_not_falsely_overlap(hc_env, tmp_path):
    """/x/models and /x/models2 must NOT read as overlapping (component-aware)."""
    store = str(tmp_path / "models")
    hot = str(tmp_path / "models2")
    os.makedirs(store)
    os.makedirs(hot)
    hc_env.setenv("HUGPY_HOT_CACHE_ROOT", hot)
    hc_env.setattr(HC, "_models_home", lambda: store)
    assert HC._degenerate_root() is False


# ── THE DATA-LOSS GUARD: a degenerate root never deletes the canonical store ─
def test_promote_is_inert_and_never_evicts_the_store_when_degenerate(hc_env, tmp_path):
    """A promote under a tiny budget would normally _make_room -> rmtree LRU
    entries. When the hot root IS the store, those entries are real models — so
    the guard must make _promote a no-op and leave every model dir intact."""
    store = str(tmp_path / "models")
    keep = _mk_model(store, "gguf/unsloth/ModelA")
    victim = _mk_model(store, "gguf/unsloth/ModelB")
    hc_env.setenv("HUGPY_HOT_CACHE_ROOT", store)
    hc_env.setenv("HUGPY_HOT_CACHE_GIB", "0.000001")   # would force eviction
    hc_env.setattr(HC, "_models_home", lambda: store)
    # Directly invoke the promoter choke point.
    HC._promote(os.path.join(keep, "model.gguf"))
    assert os.path.isdir(keep)                 # the promoted model survives
    assert os.path.isdir(victim)               # and NOTHING was evicted


def test_use_serves_unchanged_and_does_not_enqueue_when_degenerate(hc_env, tmp_path):
    store = str(tmp_path / "models")
    m = _mk_model(store, "gguf/unsloth/ModelA")
    hc_env.setenv("HUGPY_HOT_CACHE_ROOT", store)
    hc_env.setattr(HC, "_models_home", lambda: store)
    src = os.path.join(m, "model.gguf")
    assert HC.use(src) == src                  # served straight off MODELS_HOME
    with HC._STATE_LOCK:
        assert src not in HC._QUEUED           # nothing scheduled for promotion


def test_status_names_the_degenerate_reason(hc_env, tmp_path):
    store = str(tmp_path / "models")
    os.makedirs(store)
    hc_env.setenv("HUGPY_HOT_CACHE_ROOT", store)
    hc_env.setattr(HC, "_models_home", lambda: store)
    st = HC.status()
    assert st["enabled"] is False
    assert st["disabled_reason"] == "degenerate_root"
    assert st["root"] == store
    assert "overlaps MODELS_HOME" in st["detail"]


def test_separate_root_status_is_a_normal_enabled_tier(hc_env, tmp_path):
    store = str(tmp_path / "models")
    hot = str(tmp_path / "hot")
    os.makedirs(store)
    os.makedirs(hot)
    hc_env.setenv("HUGPY_HOT_CACHE_ROOT", hot)
    hc_env.setattr(HC, "_models_home", lambda: store)
    st = HC.status()
    assert st["enabled"] is True
    assert "disabled_reason" not in st


# ── slice-5 scan skip-reason histogram ──────────────────────────────────────
class _State:
    def __init__(self, assigned=None):
        self.assigned_models = list(assigned or [])
        self._provisioning = []


def test_scan_skip_reasons_names_not_local(monkeypatch):
    """The ae fingerprint: N keys considered, all skipped not_local -> the
    histogram says so in one heartbeat, distinguishing it from no_config."""
    from hugpy_fleet.worker import imports as WI
    from hugpy_storage import provision as P
    from types import SimpleNamespace

    def _cfg(mk):
        return SimpleNamespace(framework="gguf", hub_id="o/r", filename=None,
                               include=None, primary_task="text-generation",
                               tasks=["text-generation"], folder="gguf/o/r")
    monkeypatch.setattr(WI, "get_models_dict", lambda: {"a": None, "b": None, "c": None})
    monkeypatch.setattr(WI, "get_model_config", _cfg)
    monkeypatch.setattr(WI, "get_model_path", lambda mk: "/nope")
    monkeypatch.setattr(P, "model_is_local", lambda mk: False)   # all absent
    monkeypatch.setattr(P, "_on_shared_model_store", lambda rp: False)
    monkeypatch.setattr(P, "_model_store_reapable", lambda rp: True)
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_slot_occupants", lambda *a, **k: set())
    monkeypatch.setattr(A, "_loading_model_keys", lambda: [])
    monkeypatch.setattr(A, "_models_local", lambda s: [])
    scan = A._reap_scan(_State(assigned=[]))
    assert scan["scan_rows"] == 0
    assert scan["scan_skip_reasons"].get("not_local") == 3
    assert "no_config" not in scan["scan_skip_reasons"]


def test_scan_skip_reasons_names_no_config(monkeypatch):
    from hugpy_fleet.worker import imports as WI
    from hugpy_storage import provision as P

    def _boom(mk):
        raise KeyError(mk)
    monkeypatch.setattr(WI, "get_models_dict", lambda: {"a": None, "b": None})
    monkeypatch.setattr(WI, "get_model_config", _boom)   # every key unresolvable
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_slot_occupants", lambda *a, **k: set())
    monkeypatch.setattr(A, "_loading_model_keys", lambda: [])
    monkeypatch.setattr(A, "_models_local", lambda s: [])
    scan = A._reap_scan(_State(assigned=[]))
    assert scan["scan_rows"] == 0
    assert scan["scan_skip_reasons"].get("no_config") == 2


def test_scan_skip_reasons_survive_storage_proposal():
    from hugpy_fleet.central.workers import storage_proposal
    out = storage_proposal({
        "storage": {"cache_used_bytes": 672 * (1 << 30), "disk_free": 900 * (1 << 30),
                    "models": [], "scan_keys_considered": 74, "scan_rows": 0,
                    "scan_skip_reasons": {"not_local": 74}},
        "disk": {"free_bytes": 900 * (1 << 30)},
        "limits": {"disk_cache_gib": 400},
    })
    assert out["scan_skip_reasons"] == {"not_local": 74}


def test_scan_skip_reasons_degrade_for_a_pre_slice5_worker():
    from hugpy_fleet.central.workers import storage_proposal
    out = storage_proposal({
        "storage": {"cache_used_bytes": 0, "disk_free": 1 << 30, "models": []},
        "disk": {}, "limits": {},
    })
    assert out["scan_skip_reasons"] == {}
