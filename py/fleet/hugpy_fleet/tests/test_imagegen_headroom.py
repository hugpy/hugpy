"""Evict-to-fit before an in-process diffusers (t2i / img2img / studio frame) load.

Incident 2026-09-25 01:44:55: sd-turbo studio frame renders were placed on
computron while an IDLE flux2-klein slot child held 6.99 GiB of the 7.6 GiB card;
the diffusers path never evicted it and CUDA-OOM'd. The in-process image runners
now call ``imagegen_runner.set_imagegen_headroom_hook`` — the worker registers
``agent._worker_ensure_imagegen_headroom`` — BEFORE loading/generating, using the
SAME evict verb/policy comfy uses (``_evict_to_free_target`` -> the shared loop).

Covered here (mock eviction + free_vram, no GPU touched):
  * agent._worker_ensure_imagegen_headroom — target = need + gen cushion; evicts
    LRU eligible residents until free >= target; honest-degrades; names skipped
    (gated) residents for the honest refusal; unknown need -> legacy constant.
  * the SHARED policy is one policy — _evict_to_free_target is what both comfy and
    imagegen drive.

Runs like the sibling tests: venv/bin/python tests/test_imagegen_headroom.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


GIB = 2 ** 30


class Fleet:
    """models with sizes hold VRAM; evicting one hands its size back. Gated models
    refuse to evict (in-flight / static), returning a reason (named in `skipped`)."""

    def __init__(self, sizes, free, *, gated=None):
        self.sizes = dict(sizes)
        self.free = free
        self.resident = set(sizes)
        self.gated = set(gated or [])
        self.evict_calls = []

    def candidates(self, exclude):
        return [mk for mk in self.sizes if mk in self.resident and mk != exclude]

    def evict(self, state, mk, force=False):
        self.evict_calls.append(mk)
        if mk in self.gated:
            return {"model_key": mk, "evicted": False, "host_mode": "slot",
                    "reason": "eviction gated: in-flight"}
        self.resident.discard(mk)
        self.free += self.sizes.get(mk, 0)
        return {"model_key": mk, "evicted": True, "host_mode": "slot",
                "vram_freed": self.sizes.get(mk, 0), "reason": "freed"}


def install_fleet(fleet):
    saved = (agent._comfy_headroom_candidates, agent._evict_model,
             agent._free_vram_bytes)
    agent._comfy_headroom_candidates = fleet.candidates
    agent._evict_model = fleet.evict
    agent._free_vram_bytes = lambda: fleet.free

    def restore():
        (agent._comfy_headroom_candidates, agent._evict_model,
         agent._free_vram_bytes) = saved
    return restore


_state = object()


# --- cushion knob -------------------------------------------------------------
os.environ.pop("HUGPY_IMAGEGEN_GEN_CUSHION_GIB", None)
check("gen cushion default is 1.5 GiB",
      agent._imagegen_gen_cushion_bytes() == int(1.5 * GIB))
os.environ["HUGPY_IMAGEGEN_GEN_CUSHION_GIB"] = "2"
check("HUGPY_IMAGEGEN_GEN_CUSHION_GIB override respected",
      agent._imagegen_gen_cushion_bytes() == 2 * GIB)
os.environ.pop("HUGPY_IMAGEGEN_GEN_CUSHION_GIB", None)

# --- (a) THE incident: evict the idle squatter so the model fits the GPU -------
# 0.6 GiB free (card full behind a 6.99 GiB idle slot child); sd-turbo need 3 GiB
# + 1.5 cushion -> target 4.5 GiB. Evicting the squatter frees 6.99 -> fits.
os.environ["HUGPY_IMAGEGEN_GEN_CUSHION_GIB"] = "1.5"
fleet = Fleet(sizes={"flux2-klein": int(6.99 * GIB)}, free=int(0.6 * GIB))
restore = install_fleet(fleet)
try:
    tele = agent._worker_ensure_imagegen_headroom(_state, "sd-turbo", 3 * GIB)
    check("(a) the idle slot squatter was evicted via the SAME evict verb",
          fleet.evict_calls == ["flux2-klein"])
    check("(a) reached the target after reclaim",
          tele["reached"] is True and "flux2-klein" in tele["evicted"])
    check("(a) the squatter is no longer resident", "flux2-klein" not in fleet.resident)
finally:
    restore()
os.environ.pop("HUGPY_IMAGEGEN_GEN_CUSHION_GIB", None)

# --- (b) already enough room -> no eviction -----------------------------------
fleet = Fleet(sizes={"other": 3 * GIB}, free=10 * GIB)
restore = install_fleet(fleet)
try:
    tele = agent._worker_ensure_imagegen_headroom(_state, "sd-turbo", 3 * GIB)
    check("(b) enough free VRAM already -> zero evictions",
          fleet.evict_calls == [] and tele["evicted"] == [])
    check("(b) reached=True", tele["reached"] is True)
finally:
    restore()

# --- (c) honest-degrade: only a GATED (busy) resident -> named in skipped ------
fleet = Fleet(sizes={"BUSY": 6 * GIB}, free=1 * GIB, gated=["BUSY"])
restore = install_fleet(fleet)
try:
    tele = agent._worker_ensure_imagegen_headroom(_state, "sd-turbo", 4 * GIB)
    check("(c) the busy resident was tried once (no infinite loop)",
          fleet.evict_calls == ["BUSY"])
    check("(c) still short -> reached=False, proceeds (never blocks)",
          tele["reached"] is False and tele["evicted"] == [])
    check("(c) the busy resident is named in `skipped` with its gate reason (for "
          "the honest refusal)",
          any(s["model_key"] == "BUSY" and "gated" in (s.get("reason") or "")
              for s in tele["skipped"]))
    check("(c) the busy resident was NOT freed", "BUSY" in fleet.resident)
finally:
    restore()

# --- (d) unknown need -> legacy constant target -------------------------------
os.environ["HUGPY_COMFY_TARGET_FREE_GIB"] = "7.0"
fleet = Fleet(sizes={"A": 3 * GIB, "B": 3 * GIB, "C": 3 * GIB}, free=2 * GIB)
restore = install_fleet(fleet)
try:
    tele = agent._worker_ensure_imagegen_headroom(_state, "sd-turbo", None)
    check("(d) unknown need uses the legacy 7 GiB target",
          tele["target"] == 7 * GIB)
    check("(d) evicts toward that target (A then B -> 8 GiB)",
          fleet.evict_calls == ["A", "B"] and tele["reached"] is True)
finally:
    restore()
os.environ.pop("HUGPY_COMFY_TARGET_FREE_GIB", None)

# --- (e) no GPU / unmeasurable -> no-op ---------------------------------------
fleet = Fleet(sizes={"A": 3 * GIB}, free=0)
restore = install_fleet(fleet)
agent._free_vram_bytes = lambda: None
try:
    tele = agent._worker_ensure_imagegen_headroom(_state, "sd-turbo", 3 * GIB)
    check("(e) no GPU -> zero evictions, no-op",
          fleet.evict_calls == [] and tele["evicted"] == [])
finally:
    restore()

# --- (f) one shared policy: comfy + imagegen both drive _evict_to_free_target --
seen = []
_orig = agent._evict_to_free_target
def spy(state, subject, target, *, exclude=None, job_id=None, target_dev=None):
    seen.append(subject)
    return _orig(state, subject, target, exclude=exclude, job_id=job_id,
                 target_dev=target_dev)
fleet = Fleet(sizes={}, free=9 * GIB)
restore = install_fleet(fleet)
agent._evict_to_free_target = spy
# comfy target sizing stub (avoid catalog / nvidia-smi)
_saved = (agent._comfy_need_detail, agent._COMFY_LEDGER)
from hugpy_fleet.worker.comfy_ledger import ComfyLedger
agent._COMFY_LEDGER = ComfyLedger()
agent._comfy_need_detail = lambda state, mk: {
    "checkpoint": None, "checkpoint_bytes": None, "held": False, "need": None}
try:
    agent._worker_ensure_imagegen_headroom(_state, "img-model", 1 * GIB)
    agent._worker_ensure_comfy_headroom(_state, "comfy-model")
    check("(f) BOTH the imagegen and comfy headroom paths drive the ONE shared "
          "_evict_to_free_target policy",
          seen == ["img-model", "comfy-model"])
finally:
    agent._evict_to_free_target = _orig
    (agent._comfy_need_detail, agent._COMFY_LEDGER) = _saved
    restore()

print(f"\nall {ok} checks passed")
