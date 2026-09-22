"""Operator ruling 2026-08-28 (coder-next × computron): allocation is HARD
dispatch scope, and permanent no-fit pairs are durable BLOCKS, not per-request
refusals.

The live incident: coder-next (45 GiB MoE) is allocated to ae. When ae hit its
in-process concurrency cap, the reroute walk offered wildcard computron — a
box whose combined 8+16 GiB can NEVER hold the file — and the caller received
computron's honest refusal instead of the honest worker_busy. Two rules close
it structurally:

  1. ALLOCATION IS HARD SCOPE — a model with designation rows anywhere lands
     ONLY on them; a wildcard box never catches an ALLOCATED model (initial
     pick AND the reroute set, since both share workers_for_model).
     Unallocated models keep the fleet-wide wildcard behavior unchanged.
  2. PAIR BLOCKS — a POSITIVE permanent-no worker_fit_verdict records a
     durable (model × worker) block (comms.blocklist), consulted when the
     verdict is unknowable (e.g. the BARE dispatch-cache key can't resolve a
     size), and CLEARED the moment the verdict flips to True. Missing data
     alone never records anything.

Run: venv/bin/python -m pytest tests/test_alloc_scope_pair_block.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-alloc-scope-test-"))
os.environ.setdefault("HUGPY_COMMS_DB", "off")

from hugpy_fleet.central import workers as W
from hugpy_fleet.central import blocklist as BL
from hugpy_engine.serve import overrides as OV

MK = "Qwen~Qwen3-Coder-Next-GGUF"
BARE = "Qwen3-Coder-Next-GGUF"
PAIR_KEY = BARE.lower()


class _FakeSettings:
    """In-memory stand-in for comms.settings.settings_store (test isolation)."""

    def __init__(self):
        self.d = {}

    def get(self, ns, key):
        return self.d.get((ns, str(key)))

    def set(self, ns, key, value):
        self.d[(ns, str(key))] = value


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    from hugpy_fleet.central.workers import WorkerStore
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    monkeypatch.setattr(W, "worker_store", s)
    monkeypatch.setattr(W, "_assign_memory_path",
                        lambda: str(tmp_path / "assign.json"))
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    monkeypatch.setattr(W, "_free_room_probe", None)
    monkeypatch.setattr(W, "_star_map", lambda: {})
    monkeypatch.setattr(OV, "_OVERRIDES_PATH",
                        str(tmp_path / "serve_overrides.json"))
    fake = _FakeSettings()
    monkeypatch.setattr(BL, "settings_store", fake)
    return s


def _worker(store, name, *, assign=False):
    w = store.register(name=name, url=f"http://{name}:9100")
    store.set_admission(w["id"], "approved")
    store.heartbeat(w["id"], pkg_version="0.1.241")
    if assign:
        store.assign_model(w["id"], MK)
    return w


def _names(rows):
    return sorted(r.get("name") for r in rows)


# ---------------------------------------------------------------------------
# Rule 1 — allocation is hard scope
# ---------------------------------------------------------------------------
def test_allocated_model_never_falls_to_a_wildcard(rig, monkeypatch):
    home = _worker(rig, "ae", assign=True)          # noqa: F841 — the allocation
    wild = _worker(rig, "computron", assign=False)
    monkeypatch.setattr(W, "_wildcard_map", lambda: {wild["id"]: True})
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: None)  # no fit data
    got = _names(rig.workers_for_model(MK))
    assert got == ["ae"], got                       # wildcard scoped out

    # the REROUTE set shares the filter: candidates_for_model excludes it too,
    # so a busy home worker yields worker_busy, never an off-scope refusal.
    got = _names(rig.candidates_for_model(MK))
    assert got == ["ae"], got

    # the bare dispatch-cache spelling scopes identically (alias-tolerant)
    got = _names(rig.workers_for_model(BARE))
    assert got == ["ae"], got


def test_unallocated_model_keeps_wildcard_behavior(rig, monkeypatch):
    wild = _worker(rig, "computron", assign=False)
    monkeypatch.setattr(W, "_wildcard_map", lambda: {wild["id"]: True})
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: None)
    got = _names(rig.workers_for_model("some-unallocated-model"))
    assert got == ["computron"], got                # byte-identical to before


# ---------------------------------------------------------------------------
# Rule 2 — permanent-no pairs become durable blocks
# ---------------------------------------------------------------------------
def test_permanent_no_records_a_pair_block_and_skips(rig, monkeypatch):
    wild = _worker(rig, "computron", assign=False)
    monkeypatch.setattr(W, "_wildcard_map", lambda: {wild["id"]: True})

    # A POSITIVE permanent-no verdict: skipped AND durably recorded.
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: False)
    assert _names(rig.workers_for_model(MK)) == []
    assert BL.pair_blocked(PAIR_KEY, wild["id"])
    rec = BL.pair_block_map(PAIR_KEY)[wild["id"]]
    assert rec["by"] == "auto" and rec["worker_name"] == "computron"

    # Later the verdict is UNKNOWABLE (bare-key reroute, missing size): the
    # durable block still excludes the pair — the live computron shape.
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: None)
    assert _names(rig.workers_for_model(BARE)) == []


def test_pair_block_clears_when_the_verdict_changes(rig, monkeypatch):
    wild = _worker(rig, "computron", assign=False)
    monkeypatch.setattr(W, "_wildcard_map", lambda: {wild["id"]: True})
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: False)
    assert _names(rig.workers_for_model(MK)) == []
    assert BL.pair_blocked(PAIR_KEY, wild["id"])

    # e.g. a per-worker quant pin shrinks the model: verdict True -> the box
    # serves again AND the stale block is dropped.
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: True)
    assert _names(rig.workers_for_model(MK)) == ["computron"]
    assert not BL.pair_blocked(PAIR_KEY, wild["id"])


def test_assigned_but_infeasible_worker_is_gated_and_blocked(rig, monkeypatch):
    """De-facto feasibility is MEASURED RESIDENCY, not assignment (2026-08-28,
    35B-Distill x computron): an ASSIGNED worker whose static verdict is a
    positive permanent-no is skipped AND pair-blocked — the ruling's "if
    allocated, should be blocked; the user will be forced to acknowledge it"."""
    small = _worker(rig, "computron", assign=True)   # assigned, can never fit
    _worker(rig, "ae", assign=True)                  # the feasible allocation
    monkeypatch.setattr(W, "_wildcard_map", lambda: {})
    monkeypatch.setattr(
        W, "worker_can_hold",
        lambda w, mk: False if w.get("name") == "computron" else True)
    got = _names(rig.workers_for_model(MK))
    assert got == ["ae"], got
    assert BL.pair_blocked(PAIR_KEY, small["id"])


def test_resident_copy_stays_exempt_from_the_verdict(rig, monkeypatch):
    """A box MEASURED holding the model is feasible by observation — the
    verdict (even a stale False) never route-refuses a live resident."""
    w = _worker(rig, "holder", assign=True)
    rig.heartbeat(w["id"], pkg_version="0.1.241", loaded_models=[MK])
    monkeypatch.setattr(W, "_wildcard_map", lambda: {})
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: False)
    got = _names(rig.workers_for_model(MK))
    assert got == ["holder"], got


def test_missing_data_alone_never_records(rig, monkeypatch):
    wild = _worker(rig, "computron", assign=False)
    monkeypatch.setattr(W, "_wildcard_map", lambda: {wild["id"]: True})
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: None)
    assert _names(rig.workers_for_model(MK)) == ["computron"]
    assert not BL.pair_block_map(PAIR_KEY)      # no evidence -> no block
