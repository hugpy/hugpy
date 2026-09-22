"""Slice-3 reaper hardening (ae 2026-07-17): store-root-copy classification,
honest scan diagnostics, and store-root-targeted wipe.

Context — ae's real topology:
  * worker store root = the HOT NVMe (DEFAULT_ROOT=/mnt/hot990/hugpy-worker);
    ~702G of models live there; the drive is disposable (re-promotable from the
    NAS), and the operator set HUGPY_MODEL_STORE_REAPABLE=true.
  * /mnt/llm_storage = the fleet's central catalog, ALSO mounted on ae, carrying
    the .hugpy-central-catalog sentinel → _on_shared_model_store True, never
    deletable — correct and must stay.

Two defects these tests lock against:
  B. A silent scan failure was indistinguishable from an empty store —
     _worker_storage dropped _reap_scan's error field, so central saw rows:0 and
     nothing else. The payload must now carry scan_error / scan_keys_considered /
     scan_rows so a broken scan can never masquerade as a clean empty store.
  C. get_model_path()'s read-through resolves ae's models to /mnt/llm_storage
     (the NAS) even though a re-promotable copy sits on the hot store root. Those
     NAS paths classify "shared/central — never reaped" → protected → the hot
     copies NEVER become candidates. The scan must classify the STORE-ROOT COPY;
     wipe must delete the store-root copy only, re-proving the shared gate on the
     resolved delete path.

These stub the resolution seams (get_model_config / model_is_local /
get_model_path / _store_root_copy_path / the shared gate) so behavior is asserted
deterministically without a real multi-mount filesystem.

Run: venv/bin/python -m pytest tests/test_reap_store_root_classification.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import imports as WI
from hugpy_storage import provision as P


class _State:
    def __init__(self, assigned=None):
        self.assigned_models = list(assigned or [])
        self._provisioning = []


def _cfg(framework="gguf", **kw):
    base = dict(framework=framework, hub_id="Owner/Repo", filename=None,
                include=None, primary_task="text-generation",
                tasks=["text-generation"], folder="gguf/Owner/Repo")
    base.update(kw)
    return SimpleNamespace(**base)


HOT = "/mnt/hot990/hugpy-worker/models/gguf/Owner/Repo"     # store-root copy
NAS = "/mnt/llm_storage/models/gguf/Owner/Repo"             # shared read-through


@pytest.fixture
def stub_scan(monkeypatch):
    """Wire a one-model world: model_is_local True, get_model_path -> the NAS
    read-through, and a shared gate that protects ONLY the NAS path. The
    store-root-copy resolver is injected per-test."""
    monkeypatch.setattr(WI, "get_models_dict", lambda: {"Owner~Repo": _cfg()})
    monkeypatch.setattr(WI, "get_model_config", lambda mk: _cfg())
    monkeypatch.setattr(WI, "get_model_path", lambda mk: NAS)
    monkeypatch.setattr(P, "model_is_local", lambda mk: True)
    # The NAS realpath is shared/central; the HOT realpath is not.
    monkeypatch.setattr(P, "_on_shared_model_store",
                        lambda rp: rp.startswith("/mnt/llm_storage"))
    # Reapable iff NOT shared AND the box flag is on (ae: flag on).
    monkeypatch.setattr(P, "_model_store_reapable",
                        lambda rp: (not rp.startswith("/mnt/llm_storage")))
    # Deterministic size + no in-process/slot residents + not static.
    monkeypatch.setattr(A, "_path_bytes", lambda p: 100 if p else 0)
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_slot_occupants", lambda *a, **k: set())
    monkeypatch.setattr(A, "_loading_model_keys", lambda: [])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    return monkeypatch


# ── C: a model with a hot store-root copy is a CANDIDATE on the hot path ────
def test_store_root_copy_makes_the_model_a_candidate(stub_scan):
    """get_model_path resolves to the NAS (shared → would protect), but a
    complete store-root copy exists → the row must classify RECLAIMABLE with the
    HOT path + bytes, not "shared/central — never reaped".

    (Uses assigned=[] to isolate the SHARED-gate → hot-copy behaviour cleanly;
    since slice 7 an assigned model is ALSO a reclaimable candidate, so the set
    membership no longer changes the classification here either way.)"""
    stub_scan.setattr(A, "_store_root_copy_path", lambda mk, cfg: HOT)
    scan = A._reap_scan(_State(assigned=[]))
    assert scan["protected"] == []            # NOT shared-protected anymore
    assert len(scan["reclaimable"]) == 1
    row = scan["reclaimable"][0]
    assert row["model_key"] == "Owner~Repo"
    assert row["path"] == HOT                 # the store-root copy, not the NAS
    assert row["bytes"] == 100
    assert scan["scan_rows"] == 1


# ── C: no hot copy + shared read-through path → protected as today ──────────
def test_no_store_root_copy_falls_back_to_shared_and_stays_protected(stub_scan):
    """Served straight from the NAS (no hot copy): _store_root_copy_path returns
    "" → the scan uses get_model_path's NAS path → shared gate protects it. This
    is the correct resting state — no evictable row for a NAS-only model."""
    stub_scan.setattr(A, "_store_root_copy_path", lambda mk, cfg: "")
    scan = A._reap_scan(_State(assigned=["Owner~Repo"]))
    assert scan["reclaimable"] == []
    assert len(scan["protected"]) == 1
    prot = scan["protected"][0]
    assert prot["why"] == "shared/central storage — never reaped"


# ── C: wipe targets the store-root copy, refuses the shared path ────────────
def test_reclaim_wipes_the_store_root_copy_only(stub_scan):
    """_reap_reclaim must call wipe_model with the HOT path (path= kwarg), never
    let it fall back to get_model_path's NAS resolution."""
    stub_scan.setattr(A, "_store_root_copy_path", lambda mk, cfg: HOT)
    wipe_calls = []

    def _fake_wipe(mk, path=""):
        wipe_calls.append((mk, path))
        return True                            # deletion 'succeeds'

    stub_scan.setattr(P, "wipe_model", _fake_wipe)
    res = A._reap_reclaim(_State(assigned=[]), ["Owner~Repo"])
    assert res["ok"] is True
    assert wipe_calls == [("Owner~Repo", HOT)]     # hot copy, explicit target
    assert res["results"][0]["ok"] is True


def test_wipe_model_reproves_the_shared_gate_on_the_supplied_path(monkeypatch):
    """Even handed the store-root path, wipe_model re-proves the jail + shared
    gate on the RESOLVED realpath — a caller-supplied NAS path is still refused.
    This is the invariant that keeps the NAS untouchable no matter who calls."""
    # NAS path: shared → not reapable → must refuse regardless of the path= arg.
    monkeypatch.setattr(P, "_model_store_reapable", lambda rp: False)
    assert P.wipe_model("Owner~Repo", path=NAS) is False


# ── B: scan diagnostics — a broken scan can't look like an empty store ──────
def test_scan_error_is_carried_and_never_a_bare_empty_list(stub_scan):
    """When get_models_dict() blows up (a discovery report unreadable for a
    process whose store root wasn't ready at import — the ae hypothesis), the
    scan must NOT abandon: it proceeds on assignment/slot/local keys AND records
    scan_error. Rows still surface from the assignment set."""
    def _boom():
        raise RuntimeError("discovery report unreadable")
    stub_scan.setattr(WI, "get_models_dict", _boom)
    stub_scan.setattr(A, "_store_root_copy_path", lambda mk, cfg: HOT)
    # _models_local (in agent) also names the on-disk copy — this is the A-defense
    # keys fallback: even with the registry dead, assignment/local keys drive the
    # scan so real on-disk models still classify.
    stub_scan.setattr(A, "_models_local", lambda s: ["Owner~Repo"])
    scan = A._reap_scan(_State(assigned=["Owner~Repo"]))
    assert scan["error"].startswith("get_models_dict:")
    # The registry blew up, but the assignment key still produced a real row.
    # (assigned is NOT a protection reason as of slice 7 — an assigned-but-cold
    # model is a reclaimable CANDIDATE; the point here is that a row exists at all
    # and the scan is not empty.)
    assert scan["scan_rows"] == 1
    assert scan["scan_keys_considered"] >= 1
    assert (scan["reclaimable"] or scan["protected"])      # a real row, not empty


def test_worker_storage_surfaces_scan_diagnostics(stub_scan):
    """_worker_storage must pass the scan's diagnostics into the heartbeat
    payload verbatim — so a rows:0 scan is distinguishable from a clean empty
    store. Here we force a scan error and assert the payload carries it."""
    A._STORAGE_CACHE["value"] = None           # bypass the 60s cache
    A._STORE_MEASURE_CACHE["value"] = None
    stub_scan.setattr(A, "_reap_scan", lambda s: {
        "reclaimable": [], "protected": [],
        "reclaimable_bytes": 0, "error": "get_models_dict: kaboom",
        "scan_keys_considered": 65, "scan_rows": 0, "scan_row_errors": 0})
    stub_scan.setattr(A, "_measured_store_bytes", lambda: 784 * (1 << 30))
    stub_scan.setattr(A, "_disk_status",
                      lambda: {"free_bytes": 900 * (1 << 30)})
    stub_scan.setattr(A, "_orphan_scan",
                      lambda s, keys: {"items": [], "bytes": 0, "count": 0})
    stub_scan.setattr(A, "_hot_cache_status", lambda: {"enabled": False})
    stub_scan.setattr(A, "_refused_snapshot", lambda s: {})
    out = A._worker_storage(_State(assigned=["a"]))
    assert out["scan_error"] == "get_models_dict: kaboom"
    assert out["scan_keys_considered"] == 65
    assert out["scan_rows"] == 0
    assert out["models"] == []
    A._STORAGE_CACHE["value"] = None           # don't poison other tests' cache


def test_clean_empty_store_has_no_scan_error(stub_scan):
    """The complement: a genuinely empty but healthy scan carries an empty
    scan_error and honest counts — so 'empty' and 'broken' are distinguishable."""
    A._STORAGE_CACHE["value"] = None
    A._STORE_MEASURE_CACHE["value"] = None
    stub_scan.setattr(A, "_reap_scan", lambda s: {
        "reclaimable": [], "protected": [],
        "reclaimable_bytes": 0,
        "scan_keys_considered": 0, "scan_rows": 0, "scan_row_errors": 0})
    stub_scan.setattr(A, "_measured_store_bytes", lambda: 0)
    stub_scan.setattr(A, "_disk_status", lambda: {"free_bytes": 0})
    stub_scan.setattr(A, "_orphan_scan",
                      lambda s, keys: {"items": [], "bytes": 0, "count": 0})
    stub_scan.setattr(A, "_hot_cache_status", lambda: {"enabled": False})
    stub_scan.setattr(A, "_refused_snapshot", lambda s: {})
    out = A._worker_storage(_State(assigned=[]))
    assert out["scan_error"] == ""
    assert out["scan_keys_considered"] == 0
    A._STORAGE_CACHE["value"] = None
