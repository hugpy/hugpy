"""Ensure comfy headroom before every ComfyUI gen (Fix B, 2026-07-15).

When hugpy drives a ComfyUI gen, ComfyUI (a SEPARATE process) silently grabs
VRAM or waits — never a first-class queue entry. Operator directive: "evict down
to free target vram always". Before every comfy gen commits, evict on-demand
managed models (LRU, via the same _evict_model mechanism the evict verb uses)
until real free VRAM reaches HUGPY_COMFY_TARGET_FREE_GIB (default 7.0 GiB).

Covered here:
  * agent._worker_ensure_comfy_headroom — evicts LRU on-demand models until free
    >= target; no-op when already above; honest-degrades (nothing evictable ->
    proceeds + warns; no-GPU -> no-op) — never blocks the gen. Target env knob +
    default. LRU order + the comfy model itself excluded.
  * comfy_runner hook indirection — calls the registered hook if present, no-ops
    if not (central-import-safe: no worker/GPU deps pulled).

Runs like the other tests here: venv/bin/python tests/test_comfy_headroom.py
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent
comfy_runner = importlib.import_module(
    "hugpy_media.comfy.comfy_runner")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


GIB = 2**30


# ===========================================================================
# Part 1 — the comfy_runner hook indirection (central-import-safe)
# ===========================================================================
# _COMFY_HEADROOM_HOOK defaults None; _ensure_comfy_headroom is a no-op then, and
# calls the hook (once, with model_key + job_id) when registered.
check("hook defaults to None (bare central / no-GPU: pre-gen step is a no-op)",
      comfy_runner._COMFY_HEADROOM_HOOK is None)

_saved_hook = comfy_runner._COMFY_HEADROOM_HOOK
try:
    comfy_runner.set_comfy_headroom_hook(None)
    # no-op path must not raise
    comfy_runner._ensure_comfy_headroom("m", "job1")
    check("_ensure_comfy_headroom with no hook is a silent no-op", True)

    calls = []
    comfy_runner.set_comfy_headroom_hook(lambda mk, job_id: calls.append((mk, job_id)))
    comfy_runner._ensure_comfy_headroom("ckpt-A", "job-42")
    check("_ensure_comfy_headroom calls the registered hook with (model_key, job_id)",
          calls == [("ckpt-A", "job-42")])

    # a raising hook must NOT propagate into the gen path (best-effort)
    def boom(mk, job_id):
        raise RuntimeError("nope")
    comfy_runner.set_comfy_headroom_hook(boom)
    raised = None
    try:
        comfy_runner._ensure_comfy_headroom("m", "j")
    except Exception as exc:  # noqa: BLE001
        raised = exc
    check("_ensure_comfy_headroom swallows a raising hook (never breaks the gen)",
          raised is None)
finally:
    comfy_runner._COMFY_HEADROOM_HOOK = _saved_hook


# ===========================================================================
# Part 2 — agent._worker_ensure_comfy_headroom (mock eviction + free_vram)
# ===========================================================================
# A fake fleet: models with sizes hold VRAM; evicting one hands its size back.
# _comfy_headroom_candidates is stubbed to return an LRU-ordered on-demand list;
# _evict_model is stubbed to free the model's size and report evicted=True (or
# gated=False for the in-flight/static case). _free_vram_bytes reads the live
# free counter that eviction raises.
class Fleet:
    def __init__(self, sizes, free, *, gated=None):
        self.sizes = dict(sizes)             # mk -> bytes it holds
        self.free = free
        self.resident = set(sizes)
        self.gated = set(gated or [])        # models the gate refuses to evict
        self.evict_calls = []

    def candidates(self, exclude):
        # LRU order is just the sizes-dict insertion order here; drop exclude.
        return [mk for mk in self.sizes if mk in self.resident and mk != exclude]

    def evict(self, state, mk, force=False):
        self.evict_calls.append(mk)
        if mk in self.gated:
            return {"model_key": mk, "evicted": False, "host_mode": "slot",
                    "reason": "eviction gated: in-flight"}
        # free the VRAM it held
        self.resident.discard(mk)
        self.free += self.sizes.get(mk, 0)
        return {"model_key": mk, "evicted": True, "host_mode": "slot",
                "reason": "freed"}


def install_fleet(fleet):
    saved = (agent._comfy_headroom_candidates, agent._evict_model,
             agent._free_vram_bytes, agent._comfy_need_detail,
             agent._COMFY_LEDGER)
    agent._comfy_headroom_candidates = fleet.candidates
    agent._evict_model = fleet.evict
    agent._free_vram_bytes = lambda: fleet.free
    # The per-checkpoint sizing (2026-09-22) is covered in test_comfy_ledger.py;
    # this file exercises the LEGACY constant-target path, which is what an
    # unsized checkpoint still gets. Pin that and a fresh ledger PER CASE (this
    # module runs at import, so a module-level patch would leak into every
    # suite collected after it) so nothing here touches the catalog or nvidia-smi.
    from hugpy_fleet.worker.comfy_ledger import ComfyLedger
    agent._COMFY_LEDGER = ComfyLedger()
    agent._comfy_need_detail = lambda state, mk: {
        "checkpoint": None, "checkpoint_bytes": None, "held": False, "need": None}
    def restore():
        (agent._comfy_headroom_candidates, agent._evict_model,
         agent._free_vram_bytes, agent._comfy_need_detail,
         agent._COMFY_LEDGER) = saved
    return restore


_state = object()   # _worker_ensure_comfy_headroom passes it straight to _evict_model

# --- target default + env knob -------------------------------------------------
os.environ.pop("HUGPY_COMFY_TARGET_FREE_GIB", None)
check("comfy target default is 7.0 GiB (observed ~6.5G ComfyUI peak + margin)",
      agent._comfy_target_free_bytes() == 7 * GIB)
os.environ["HUGPY_COMFY_TARGET_FREE_GIB"] = "9.5"
check("HUGPY_COMFY_TARGET_FREE_GIB override respected",
      agent._comfy_target_free_bytes() == int(9.5 * GIB))
os.environ["HUGPY_COMFY_TARGET_FREE_GIB"] = "garbage"
check("comfy target ignores garbage -> 7.0 GiB default",
      agent._comfy_target_free_bytes() == 7 * GIB)
os.environ.pop("HUGPY_COMFY_TARGET_FREE_GIB", None)

# --- (a) evicts LRU on-demand until free >= target ----------------------------
os.environ["HUGPY_COMFY_TARGET_FREE_GIB"] = "7.0"
# start 2 GiB free; three residents 3/3/3 GiB. Need to reach 7 -> evict 2 (LRU).
fleet = Fleet(sizes={"A": 3 * GIB, "B": 3 * GIB, "C": 3 * GIB}, free=2 * GIB)
restore = install_fleet(fleet)
try:
    res = agent._worker_ensure_comfy_headroom(_state, "COMFY-CKPT", "job-1")
    check("(a) evicted LRU-first until the target was reached (A then B)",
          fleet.evict_calls == ["A", "B"])
    check("(a) reached the target (free 2 -> 8 GiB >= 7)",
          res["reached"] is True and res["free_after"] == 8 * GIB)
    check("(a) did NOT over-evict (C survives)", "C" in fleet.resident)
finally:
    restore()

# --- (b) no-op when already above target --------------------------------------
fleet = Fleet(sizes={"A": 3 * GIB}, free=9 * GIB)   # already > 7 GiB
restore = install_fleet(fleet)
try:
    res = agent._worker_ensure_comfy_headroom(_state, "COMFY-CKPT")
    check("(b) already above target -> zero evictions (no-op)",
          fleet.evict_calls == [] and res["evicted"] == [])
    check("(b) reports reached=True", res["reached"] is True)
finally:
    restore()

# --- (c) honest-degrade: nothing evictable -> proceed + warn, never block -----
# No candidates at all: free stays below target, routine returns (never loops).
fleet = Fleet(sizes={}, free=1 * GIB)
restore = install_fleet(fleet)
try:
    res = agent._worker_ensure_comfy_headroom(_state, "COMFY-CKPT")
    check("(c) nothing evictable -> no evictions, returns (never hangs)",
          fleet.evict_calls == [] and res["evicted"] == [])
    check("(c) reports reached=False (proceeds anyway, honest-degrade)",
          res["reached"] is False)
finally:
    restore()

# --- (c2) honest-degrade: only a GATED candidate -> tries once, then proceeds --
# A single in-flight/static-gated model: _evict_model returns evicted=False. The
# routine must NOT loop forever on it (it advances `tried`) and must still return.
fleet = Fleet(sizes={"BUSY": 5 * GIB}, free=1 * GIB, gated=["BUSY"])
restore = install_fleet(fleet)
try:
    res = agent._worker_ensure_comfy_headroom(_state, "COMFY-CKPT")
    check("(c2) a gated candidate is tried exactly once (no infinite loop)",
          fleet.evict_calls == ["BUSY"])
    check("(c2) still short of target -> reached=False, proceeds",
          res["reached"] is False and res["evicted"] == [])
    check("(c2) the gated model was NOT freed", "BUSY" in fleet.resident)
finally:
    restore()

# --- (d) no-GPU / can't measure -> no-op (byte-identical to today) ------------
fleet = Fleet(sizes={"A": 3 * GIB}, free=0)
restore = install_fleet(fleet)
agent._free_vram_bytes = lambda: None      # can't read free VRAM
try:
    res = agent._worker_ensure_comfy_headroom(_state, "COMFY-CKPT")
    check("(d) no GPU / unmeasurable -> zero evictions (no-op, never blocks)",
          fleet.evict_calls == [] and res["evicted"] == [])
    check("(d) reports the no-GPU note", res.get("note") == "no GPU / unmeasurable")
finally:
    restore()

# --- (e) the comfy model itself is excluded from eviction candidates ----------
# _comfy_headroom_candidates is stubbed to honor `exclude`; assert the routine
# passes the comfy model_key as the exclusion.
seen_exclude = []
fleet = Fleet(sizes={"OTHER": 8 * GIB}, free=1 * GIB)
def spy_candidates(exclude):
    seen_exclude.append(exclude)
    return fleet.candidates(exclude)
restore = install_fleet(fleet)
agent._comfy_headroom_candidates = spy_candidates
try:
    agent._worker_ensure_comfy_headroom(_state, "COMFY-CKPT")
    check("(e) the comfy model_key is passed as the eviction exclusion",
          seen_exclude and all(x == "COMFY-CKPT" for x in seen_exclude))
finally:
    restore()

os.environ.pop("HUGPY_COMFY_TARGET_FREE_GIB", None)

print(f"\nall {ok} checks passed")
