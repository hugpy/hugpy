"""Real-VRAM ~90% ceiling gate on the slot load/evict path (Fix A, 2026-07-15).

ae's 3090 got topped out by a SEPARATE ComfyUI process (~5.5->6.5G) and the slot
scheduler never reacted, because SlotPool.endpoint_for decided "is there room?"
by slot-OCCUPANCY count, never by real device VRAM — so a 95%-full card with an
idle slot loaded happily, then the child under-offloaded or OOMed.

This slice adds a real-VRAM ceiling gate:
  * agent._worker_slot_fit_check(model_key) -> bool: True when loading leaves the
    card at/under the ~90% ceiling (>= (1-ceiling) of total VRAM free after the
    weights land), False when it would breach; degrades to True when free/total/
    need is unknown (no GPU / can't measure) — NEVER blocks a load because we
    couldn't read the card. HUGPY_VRAM_CEILING_FRAC overrides the 0.90 default.
  * slots.SlotPool.endpoint_for: with a registered fit-check that says "over
    ceiling", it asks the registered make-room hook (the worker's plan_fit
    admission) ONCE before loading — step 2 F1 (2026-09-29): the pool never
    evicts a seat itself on this path; no hook registered -> proceeds anyway
    (honest-degrade, nothing evicted outside plan_fit).
  * No fit-check registered (bare central / no-GPU) -> occupancy-only routing,
    byte-identical to before.

Runs like the other tests here: venv/bin/python tests/test_slot_vram_ceiling.py
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent
slots = importlib.import_module("hugpy_engine.serve.slots")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


GIB = 2**30

# ===========================================================================
# Part 1 — agent._worker_slot_fit_check semantics (mock free/total/need)
#
# THE DEFAULT CHANGED 2026-07-27 (see agent._vram_ceiling_reserve_bytes). The
# reserve used to be 10% of TOTAL card VRAM, which re-charged for KV that `need`
# already carries and refused models the box could genuinely run. It is now a
# bounded compute/activation cushion (spill._CTX_COMPUTE_RESERVE_BYTES, 512 MiB,
# measured 348 MiB) un-stacked against the external floor deducted from the
# free read (HUGPY_VRAM_RESERVE_GIB). THE FLOOR DEFAULTS TO 0 SINCE 2026-09-29
# (it was a flat 1.0 GiB): on a default box the budgetable figure IS the device
# free read and the reserve on it is the full 512 MiB cushion; the physical
# guarantee is still "raw free after the load >= max(external floor, cushion)".
# ===========================================================================
_fv_orig = agent._free_vram_bytes
_tv_orig = agent._total_vram_bytes
_need_orig = agent._incoming_need_bytes
_env_saved = {k: os.environ.get(k) for k in
              ("HUGPY_VRAM_CEILING_FRAC", "HUGPY_VRAM_CEILING_CUSHION_GIB",
               "HUGPY_VRAM_RESERVE_GIB")}
for _k in _env_saved:
    os.environ.pop(_k, None)                     # an unconfigured box
try:
    # 24 GiB card, default cushion, default floor 0 -> the reserve on the
    # budgetable figure is the whole 512 MiB cushion.
    agent._total_vram_bytes = lambda: 24 * GIB
    agent._incoming_need_bytes = lambda mk: 4 * GIB          # needs 4 GiB
    check("default reserve on a 24 GiB card is the 512 MiB cushion (no external "
          "floor to un-stack against since 2026-09-29)",
          agent._vram_ceiling_reserve_bytes(24 * GIB) == 512 * 2**20)

    # 10 GiB free now: after a 4 GiB load, 6 GiB budgetable free left -> fits.
    agent._free_vram_bytes = lambda: 10 * GIB
    check("fit_check: True when loading leaves headroom",
          agent._worker_slot_fit_check("m") is True)

    # THE REGRESSION, in miniature. 5 GiB budgetable free (= 6 GiB raw), need 4:
    # the OLD gate demanded 2.4 GiB free after the load and refused. 1 GiB of
    # budgetable free remains and 2 GiB of RAW device free — the load fits.
    agent._free_vram_bytes = lambda: 5 * GIB
    check("fit_check: True where the old 10%-of-card reserve wrongly refused "
          "(5 GiB free, 4 GiB need, 2 GiB raw free after the load)",
          agent._worker_slot_fit_check("m") is True)

    # Exactly at the boundary: free == need + cushion -> exactly the cushion of
    # RAW device headroom remains after the load (the floor is 0, so budgetable
    # free IS the device read).
    agent._free_vram_bytes = lambda: 4 * GIB + 512 * 2**20
    check("fit_check: True exactly at the boundary (>= is inclusive)",
          agent._worker_slot_fit_check("m") is True)

    # A load that would genuinely leave NO working room still REFUSES: one byte
    # into the compute cushion is past the last real device headroom.
    agent._free_vram_bytes = lambda: 4 * GIB + 512 * 2**20 - 1
    check("fit_check: False when the load would spend past the last headroom",
          agent._worker_slot_fit_check("m") is False)
    agent._free_vram_bytes = lambda: 0
    check("fit_check: False on a full card (never admit-then-OOM)",
          agent._worker_slot_fit_check("m") is False)

    # --- honest-degrade: unmeasurable -> True (never block) ---
    agent._free_vram_bytes = lambda: 0                       # would breach if measured
    agent._total_vram_bytes = lambda: None                   # can't read total
    check("fit_check: degrades to True when total VRAM unknown (no GPU)",
          agent._worker_slot_fit_check("m") is True)

    agent._total_vram_bytes = lambda: 24 * GIB
    agent._free_vram_bytes = lambda: None                    # can't read free
    check("fit_check: degrades to True when free VRAM unknown",
          agent._worker_slot_fit_check("m") is True)

    agent._free_vram_bytes = lambda: 0
    agent._incoming_need_bytes = lambda mk: None             # unknown weight size
    check("fit_check: degrades to True when the incoming need is unknown",
          agent._worker_slot_fit_check("m") is True)
    check("_vram_ceiling_reserve_bytes(None) is 0 (unmeasurable total)",
          agent._vram_ceiling_reserve_bytes(None) == 0)

    # --- HUGPY_VRAM_CEILING_FRAC keeps its ORIGINAL meaning when set ---------
    agent._incoming_need_bytes = lambda mk: 4 * GIB
    agent._free_vram_bytes = lambda: 5 * GIB
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "0.99"           # reserve = 0.24 GiB
    check("explicit frac 0.99 -> reserve is total*(1-frac), verbatim",
          agent._vram_ceiling_reserve_bytes(24 * GIB) == int(24 * GIB * 0.01))
    check("fit_check: HUGPY_VRAM_CEILING_FRAC override respected (0.99 -> fits)",
          agent._worker_slot_fit_check("m") is True)
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "0.90"           # the OLD default
    check("explicit frac 0.90 reproduces the OLD default reserve exactly "
          "(2.4 GiB on a 24 GiB card)",
          agent._vram_ceiling_reserve_bytes(24 * GIB) == int(24 * GIB * 0.10))
    check("fit_check: with the old ceiling asked for explicitly, the old "
          "refusal is back (5 GiB free, 4 GiB need)",
          agent._worker_slot_fit_check("m") is False)
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "0.50"           # reserve = 12 GiB
    # 5 GiB free < 12 GiB reserve even before the load -> breach.
    check("fit_check: a stricter ceiling (0.50) breaches",
          agent._worker_slot_fit_check("m") is False)
    check("_vram_ceiling_frac reads the env override", agent._vram_ceiling_frac() == 0.50)
    os.environ.pop("HUGPY_VRAM_CEILING_FRAC", None)
    check("_vram_ceiling_frac defaults to 0.90", agent._vram_ceiling_frac() == 0.90)
    check("_vram_ceiling_frac_explicit is None when unset",
          agent._vram_ceiling_frac_explicit() is None)
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "banana"
    check("_vram_ceiling_frac ignores garbage -> 0.90", agent._vram_ceiling_frac() == 0.90)
    check("garbage is NOT 'explicit' -> the default cushion governs",
          agent._vram_ceiling_frac_explicit() is None
          and agent._vram_ceiling_reserve_bytes(24 * GIB) == 512 * 2**20)
    os.environ["HUGPY_VRAM_CEILING_FRAC"] = "1.5"            # out of (0,1]
    check("_vram_ceiling_frac clamps out-of-range -> 0.90", agent._vram_ceiling_frac() == 0.90)
    check("out-of-range is NOT 'explicit' -> the default cushion governs",
          agent._vram_ceiling_reserve_bytes(24 * GIB) == 512 * 2**20)
    os.environ.pop("HUGPY_VRAM_CEILING_FRAC", None)

    # --- the cushion is the OOM guard when there is no external floor -------
    # (an EXPLICIT 0 behaves exactly like the default since 2026-09-29)
    os.environ["HUGPY_VRAM_RESERVE_GIB"] = "0"
    check("floor 0 -> the full 512 MiB cushion is the reserve",
          agent._vram_ceiling_reserve_bytes(24 * GIB) == 512 * 2**20)
    agent._free_vram_bytes = lambda: 4 * GIB + 512 * 2**20
    check("floor 0: a load leaving exactly the cushion is admitted",
          agent._worker_slot_fit_check("m") is True)
    agent._free_vram_bytes = lambda: 4 * GIB + 512 * 2**20 - 1
    check("floor 0: a load leaving less than the cushion is REFUSED",
          agent._worker_slot_fit_check("m") is False)
    os.environ.pop("HUGPY_VRAM_RESERVE_GIB", None)

    # --- operator cushion lever ---------------------------------------------
    os.environ["HUGPY_VRAM_CEILING_CUSHION_GIB"] = "3"
    check("HUGPY_VRAM_CEILING_CUSHION_GIB widens the cushion (clamped to the "
          "old 10%-of-card term; nothing to un-stack with the default floor 0)",
          agent._vram_ceiling_reserve_bytes(24 * GIB) == int(24 * GIB * 0.10))
    os.environ["HUGPY_VRAM_RESERVE_GIB"] = "1"
    check("...and an EXPLICIT 1 GiB floor still un-stacks against it",
          agent._vram_ceiling_reserve_bytes(24 * GIB)
          == int(24 * GIB * 0.10) - GIB)
    os.environ.pop("HUGPY_VRAM_RESERVE_GIB", None)
    os.environ["HUGPY_VRAM_CEILING_CUSHION_GIB"] = "nope"
    check("a garbage cushion is ignored -> the measured 512 MiB stands",
          agent._vram_ceiling_cushion_bytes() == 512 * 2**20)
    os.environ.pop("HUGPY_VRAM_CEILING_CUSHION_GIB", None)

    # --- NEVER STRICTER THAN THE OLD DEFAULT, on any card -------------------
    for _gib in (2, 4, 6, 8, 11.6, 12, 16, 23.6, 24, 48, 80):
        _tot = int(_gib * GIB)
        _old = int(_tot * 0.10)
        _new = agent._vram_ceiling_reserve_bytes(_tot)
        check(f"{_gib} GiB card: new default reserve ({_new}) <= old ({_old})",
              _new <= _old)
        check(f"{_gib} GiB card: post-load RAW headroom guarantee is "
              f"max(floor, cushion)",
              _new + agent._external_vram_floor_bytes()
              == max(agent._external_vram_floor_bytes(),
                     min(agent._vram_ceiling_cushion_bytes(), _old)))

    # --- THE LIVE ae REGRESSION (the refusal the operator quoted) -----------
    # "needs 21.1 GB, 21.3 GB free of 23.6 GB (2.4 GB ceiling reserve)".
    # _human_bytes labels 1024-based units "GB", so those are GiB. The 21.3 was
    # the BUDGETABLE figure under the old 1.0 GiB floor; the device read was
    # 22.3 GiB, which is what the default box (floor 0) now prices against.
    agent._total_vram_bytes = lambda: int(23.6 * GIB)
    agent._free_vram_bytes = lambda: int(22.3 * GIB)          # device free
    agent._incoming_need_bytes = lambda mk: int(21.1 * GIB)   # weights+KV
    check("LIVE ae REGRESSION: 21.1 GiB need, 22.3 GiB device free of 23.6 GiB "
          "is ADMITTED (1.2 GiB of raw device headroom remains >= the cushion)",
          agent._worker_slot_fit_check("m") is True)
    agent._incoming_need_bytes = lambda mk: int(22.6 * GIB)   # exceeds device free
    check("...but 22.6 GiB on the same card still REFUSES (more than the device "
          "has free)",
          agent._worker_slot_fit_check("m") is False)
finally:
    agent._free_vram_bytes = _fv_orig
    agent._total_vram_bytes = _tv_orig
    agent._incoming_need_bytes = _need_orig
    for _k, _v in _env_saved.items():
        os.environ.pop(_k, None)
        if _v is not None:
            os.environ[_k] = _v


# ===========================================================================
# Part 2 — slots.SlotPool.endpoint_for ceiling eviction (mock the pool I/O)
# ===========================================================================
# A fake pool: two slots, one holding an idle on-demand model ("A"), one idle.
# The ceiling gate says "over ceiling" until "A" is evicted (its /unload frees
# the card). We assert endpoint_for evicts A THEN loads the new model.
class FakePool(slots.SlotPool):
    def __init__(self, statuses):
        super().__init__(urls=[s["_control"] for s in statuses])
        self._statuses = statuses
        self.unloaded = []
        self.loaded = []

    def statuses(self):
        # return a shallow copy list of the live dicts (endpoint_for mutates none)
        return [dict(s) for s in self._statuses]

    def unload(self, control_url):
        self.unloaded.append(control_url)
        # free the seat: the occupant is gone
        for s in self._statuses:
            if s["_control"] == control_url:
                s["model_key"] = None
                s["healthy"] = True
        return {"ok": True}


def _post_recorder(pool):
    def fake_post(url, body, timeout):
        pool.loaded.append((url, body.get("model_key")))
        return {"endpoint": url.replace("/load", "") + "/infer"}
    return fake_post


_ep_saved = (slots._EVICTION_POLICY, slots._FIT_CHECK, slots._RESIDENCY_LOOKUP,
             slots._MAKE_ROOM, slots._post, slots._get)
try:
    # on-demand eviction policy (as the worker registers): A is on-demand.
    slots.set_eviction_policy(lambda mk: True)      # every occupant is on-demand
    slots.set_residency_lookup(lambda mk: "on-demand")
    slots._get = lambda url, timeout=3.0: {}        # unused (we override statuses)

    # (i) OVER CEILING then FITS after the ADMISSION evicts A ------------------
    # step 2 F1: the pool asks the registered make-room hook (the worker's
    # plan_fit admission) exactly once; THAT evicts A (through the worker's
    # eviction verb, here: the pool's unload) and answers "evicted".
    st = [
        {"_control": "http://s0", "model_key": "A", "healthy": True,
         "busy": False, "last_used": 100.0, "endpoint": "http://s0"},
        {"_control": "http://s1", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s1"},
    ]
    pool = FakePool(st)
    slots._post = _post_recorder(pool)
    admissions = []

    def admission_evicts_A(mk):
        admissions.append(mk)
        pool.unload("http://s0")
        return {"action": "evicted", "evicted": ["A"], "freed_bytes": 6 * GIB, "reason": None}
    slots.set_make_room(admission_evicts_A)

    # gate: over ceiling while A is resident; fits once A is gone.
    def gate_needs_A_gone(mk):
        a_resident = any(s.get("model_key") == "A" for s in pool._statuses)
        return not a_resident      # False (over ceiling) while A resident
    slots.set_fit_check(gate_needs_A_gone)

    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    check("(i) endpoint_for consulted the admission exactly once", admissions == ["NEW"])
    check("(i) the admission (not the pool) evicted A before loading",
          pool.unloaded == ["http://s0"])
    check("(i) then loaded NEW into a freed idle slot",
          any(mk == "NEW" for (_u, mk) in pool.loaded))
    check("(i) returned a usable endpoint", isinstance(ep, str) and ep)
    slots.set_make_room(None)

    # (ii) ALREADY UNDER CEILING -> no eviction, just load into the idle slot ---
    st = [
        {"_control": "http://s0", "model_key": "A", "healthy": True,
         "busy": False, "last_used": 100.0, "endpoint": "http://s0"},
        {"_control": "http://s1", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s1"},
    ]
    pool = FakePool(st)
    slots._post = _post_recorder(pool)
    slots.set_fit_check(lambda mk: True)            # always fits
    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    check("(ii) under ceiling: NO eviction",
          pool.unloaded == [])
    check("(ii) under ceiling: loaded NEW into the pre-existing idle slot",
          pool.loaded and pool.loaded[0][1] == "NEW")

    # (iii) HONEST-DEGRADE: over ceiling, nothing evictable -> proceed anyway ---
    # Only a static occupant + an idle slot; eviction policy rejects static, so
    # nothing is evictable, yet the load must still proceed (never hang).
    st = [
        {"_control": "http://s0", "model_key": "STAT", "healthy": True,
         "busy": False, "last_used": 100.0, "endpoint": "http://s0"},
        {"_control": "http://s1", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s1"},
    ]
    pool = FakePool(st)
    slots._post = _post_recorder(pool)
    slots.set_eviction_policy(lambda mk: mk != "STAT")   # STAT is not on-demand
    slots.set_fit_check(lambda mk: False)                # ALWAYS over ceiling
    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    check("(iii) honest-degrade: no admission registered -> the pool evicts nobody",
          pool.unloaded == [])
    check("(iii) honest-degrade: the load STILL proceeds (never hangs)",
          pool.loaded and pool.loaded[0][1] == "NEW" and isinstance(ep, str))

    # (iv) NO fit-check registered -> occupancy-only, byte-identical to before ---
    st = [
        {"_control": "http://s0", "model_key": "A", "healthy": True,
         "busy": False, "last_used": 100.0, "endpoint": "http://s0"},
        {"_control": "http://s1", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s1"},
    ]
    pool = FakePool(st)
    slots._post = _post_recorder(pool)
    slots.set_fit_check(None)                       # no ceiling gate
    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    check("(iv) no fit-check: NO ceiling eviction (occupancy-only path)",
          pool.unloaded == [])
    check("(iv) no fit-check: loaded straight into the idle slot",
          pool.loaded and pool.loaded[0][1] == "NEW")
finally:
    (slots._EVICTION_POLICY, slots._FIT_CHECK, slots._RESIDENCY_LOOKUP,
     slots._MAKE_ROOM, slots._post, slots._get) = _ep_saved


# ===========================================================================
# Part 3 — PARTIAL offload (t21 stage 2.5): the cross-tier make-room hook hands
# back an honest layers-that-fit plan for an oversize GGUF; endpoint_for must
# launch the child with --n-gpu-layers N (not the shard-blind autofit) and stop
# spinning the (full-need) ceiling loop.
# ===========================================================================
def _body_recorder(pool):
    def fake_post(url, body, timeout):
        pool.loaded.append((url, body))          # capture the WHOLE body
        return {"endpoint": url.replace("/load", "") + "/infer"}
    return fake_post


_p3_saved = (slots._EVICTION_POLICY, slots._FIT_CHECK, slots._RESIDENCY_LOOKUP,
             slots._MAKE_ROOM, slots._post, slots._get)
try:
    slots._get = lambda url, timeout=3.0: {}
    slots.set_eviction_policy(lambda mk: False)   # nothing slot-side is evictable
    slots.set_residency_lookup(lambda mk: "on-demand")
    slots.set_fit_check(lambda mk: False)         # ALWAYS over ceiling (full need)
    # make-room admits an honest partial offload of 17/48 layers.
    slots.set_make_room(lambda mk: {
        "action": "partial", "evicted": [], "n_gpu_layers": 17, "gpu_pct": 35,
        "partial": {"total_layers": 48}, "reason": None})

    st = [                                        # a single idle slot, nothing to evict
        {"_control": "http://s0", "model_key": None, "healthy": True,
         "busy": False, "last_used": 0.0, "endpoint": "http://s0"},
    ]
    pool = FakePool(st)
    slots._post = _body_recorder(pool)
    ep = pool.endpoint_for("Qwen~Qwen3-Coder-Next-GGUF", load_timeout=1.0)
    check("(v) partial: nothing slot-side evicted (hybrid, not eviction)",
          pool.unloaded == [])
    check("(v) partial: loaded the model into the idle slot",
          pool.loaded and pool.loaded[0][1].get("model_key")
          == "Qwen~Qwen3-Coder-Next-GGUF")
    check("(v) partial: launched the child with the honest --n-gpu-layers N",
          pool.loaded[0][1].get("n_gpu_layers") == 17)
    check("(v) partial: returned a usable endpoint (no hang on the ceiling loop)",
          isinstance(ep, str) and ep)
finally:
    (slots._EVICTION_POLICY, slots._FIT_CHECK, slots._RESIDENCY_LOOKUP,
     slots._MAKE_ROOM, slots._post, slots._get) = _p3_saved

print(f"\nall {ok} checks passed")
