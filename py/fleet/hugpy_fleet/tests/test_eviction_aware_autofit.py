"""Eviction-aware autofit — "size up for eviction; spill based on the theoretical
eviction's success" (operator, 2026-07-25).

THE MOTIVATING CASE, caught live: flux2-klein-9b sat at 21/36 layers on computron
with 3.1 GiB STILL FREE. Re-seating put all 36 on the card. The headroom had been
there the whole time — the seat was crippled not by a full card but by a
*momentarily* full one, and a slot child's layer count is fixed at spawn forever.
The cliff measured the same day makes that a ~4x loss: a dense GGUF runs ~135
tok/s fully resident and ~36 the moment ONE layer spills, and it looks completely
healthy from central.

The inversion under test:

    TODAY:  free VRAM now -> autofit N -> evict to fit N (trivially succeeds)
    WANTED: free + RECLAIMABLE -> autofit -> evict -> **RE-PLAN against what was
            actually freed**

The re-plan is the HARD REQUIREMENT: a resident can go static/busy between
planning and executing, so the eviction may under-deliver. Launching the
optimistic count against VRAM we did not get reproduces `vram-admission-no-evict`
(admit-then-crash). The optimistic number is a TARGET, never a promise.

Run: venv/bin/python -m pytest tests/test_eviction_aware_autofit.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import gen_gate
from hugpy_engine import spill as SP

D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30
MIB = 1 << 20


class _State:
    pass


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """A card with a mutable free cell, a resident set, and a GGUF subject whose
    geometry is stubbed. Eviction returns bytes to the free cell (the reclaim).

    The subject is a 36-layer GGUF standing in for flux2-klein-9b. autofit runs
    for REAL (spill.autofit_gpu_layers) against whatever free figure the code
    under test hands it — the seam this whole task turns on — so the layer counts
    asserted here are the ones the slot child would really compute.
    """
    card = {"total": 24 * GIB, "free": 0, "need": 0}
    residents = {}
    lru = {}
    static = set()
    replying = set()
    busy_slots = set()
    evicted_calls = []

    # A 36-layer, 6.0 GiB GGUF on disk (flux2-klein-9b shape).
    gguf = tmp_path / "flux2-klein-9b-Q4_K_M.gguf"
    gguf.write_bytes(b"\0")
    monkeypatch.setattr(A, "_served_gguf_geometry",
                        lambda mk: ((str(gguf), 36) if mk == "flux2-klein-9b"
                                    else (None, None)))
    import os as _os
    _real_getsize = _os.path.getsize
    monkeypatch.setattr(SP.os.path, "getsize",
                        lambda p: (6 * GIB if str(p) == str(gguf)
                                   else _real_getsize(p)))
    monkeypatch.setattr(SP, "_gguf_layer_count", lambda p: 36)
    monkeypatch.setattr(SP, "vision_projector_bytes", lambda p: 0)

    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: card["need"])
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v["vram_bytes"],
                                    "host_mode": v["host_mode"], "alive": True}
                                   for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency",
                        lambda mk: "static" if mk in static else "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set(busy_slots))
    monkeypatch.setattr(gen_gate, "in_flight",
                        lambda mk: 1 if mk in replying else 0)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(D, "last_used_snapshot", lambda: dict(lru))
    # No k37 alloc mode / no explicit ngl on the env wire by default.
    monkeypatch.delenv("HUGPY_N_GPU_LAYERS", raising=False)
    monkeypatch.delenv("HUGPY_GPU_MEM_GIB", raising=False)
    # The admission ceiling reads all three (2026-07-27 bounded cushion).
    for _c in ("HUGPY_VRAM_CEILING_FRAC", "HUGPY_VRAM_CEILING_CUSHION_GIB",
               "HUGPY_VRAM_RESERVE_GIB"):
        monkeypatch.delenv(_c, raising=False)

    hooks = {"on_evict": None}

    def _fake_evict(state, mk, force=False):
        evicted_calls.append(mk)
        if hooks["on_evict"] is not None:
            hooks["on_evict"](mk)
        row = residents.pop(mk, None)
        freed = row["vram_bytes"] if row else 0
        card["free"] += freed
        return {"model_key": mk, "evicted": bool(row),
                "vram_freed": freed if row else None,
                "host_mode": row["host_mode"] if row else "none"}
    monkeypatch.setattr(A, "_evict_model", _fake_evict)

    A._VRAM_EVICTIONS.update(count=0, last=None, last_at=0.0)
    A._PARTIAL_NGL.clear()

    return type("Rig", (), {
        "card": card, "residents": residents, "lru": lru, "static": static,
        "replying": replying, "busy_slots": busy_slots, "hooks": hooks,
        "evicted": evicted_calls, "gguf": str(gguf)})()


# ── THE flux2 CASE: 21/36 with evictable room -> plans all 36 ────────────────
def test_flux2_case_sizes_up_over_evictable_room(rig):
    # Card momentarily busy: an idle 4 GiB neighbour holds most of it, leaving a
    # sliver free. The subject's full need FITS under the ceiling (so today's
    # gate says "proceed" and nothing is evicted) but the child would autofit a
    # crippled partial seat against the sliver.
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 512 * MIB          # ceiling gate passes -> the 'ok' branch
    rig.residents["idle-neighbour"] = {"vram_bytes": 8 * GIB,
                                       "host_mode": "subprocess"}
    rig.lru["idle-neighbour"] = 100.0

    crippled = SP.autofit_gpu_layers(rig.gguf, free_vram=4 * GIB)
    assert 0 < crippled < 36, f"precondition: a partial seat, got {crippled}"

    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")

    assert plan["action"] == "partial"
    assert plan["evicted"] == ["idle-neighbour"]
    # -1 == every layer on the card (autofit's "the whole file fits") — the
    # re-seat result the operator measured by hand.
    assert plan["n_gpu_layers"] == -1
    su = plan["size_up"]
    assert su["layers_without_eviction"] == crippled
    assert su["reclaimable_bytes"] == 8 * GIB
    assert su["free_after_bytes"] == 12 * GIB


# ── THE HARD REQUIREMENT: eviction under-delivers -> re-plan, never the target ─
def test_under_delivered_eviction_replans_to_the_achievable_count(rig):
    # Two idle neighbours are counted as reclaimable at plan time (2 + 2 GiB), but
    # one of them goes STATIC the instant before it would be evicted. The
    # eviction therefore delivers HALF the theoretical figure. The emitted layer
    # count must be re-planned from what actually materialised — never the
    # optimistic target sized against VRAM we did not get.
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 256 * MIB
    rig.residents["idle-a"] = {"vram_bytes": 3 * GIB, "host_mode": "subprocess"}
    rig.residents["idle-b"] = {"vram_bytes": 3 * GIB, "host_mode": "subprocess"}
    rig.lru["idle-a"] = 100.0
    rig.lru["idle-b"] = 200.0

    optimistic = SP.autofit_gpu_layers(rig.gguf, free_vram=10 * GIB)  # 4 + 3 + 3

    # idle-a evicts fine; that turn also makes idle-b static (a resident can
    # become protected between planning and executing — the whole point).
    def _on_evict(mk):
        if mk == "idle-a":
            rig.static.add("idle-b")
    rig.hooks["on_evict"] = _on_evict

    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")

    assert plan["action"] == "partial"
    assert rig.evicted == ["idle-a"]                  # idle-b protected in time
    achievable = SP.autofit_gpu_layers(rig.gguf, free_vram=7 * GIB)
    assert plan["n_gpu_layers"] == achievable
    assert 0 < achievable < 36
    assert plan["n_gpu_layers"] != optimistic, (
        "must never launch the optimistic count against VRAM that did not "
        "materialise (vram-admission-no-evict / admit-then-crash)")
    assert optimistic == -1                            # the theoretical figure
    assert plan["size_up"]["target_layers"] == optimistic
    assert plan["size_up"]["free_after_bytes"] == 7 * GIB


def test_replan_never_emits_more_than_the_device_supports(rig):
    # Same shape, but the eviction frees NOTHING at all (the resident is gone /
    # the unload no-ops). The plan must collapse back to today's behaviour rather
    # than emit a number the card cannot hold.
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 256 * MIB
    rig.residents["ghost"] = {"vram_bytes": 8 * GIB, "host_mode": "subprocess"}
    rig.lru["ghost"] = 100.0

    def _on_evict(mk):
        rig.residents.pop(mk, None)        # vanishes -> _fake_evict frees 0
    rig.hooks["on_evict"] = _on_evict

    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan.get("n_gpu_layers") is None
    assert "under-delivered" in plan["note"]
    assert "flux2-klein-9b" not in A._PARTIAL_NGL


# ── nothing evictable -> byte-identical to today ────────────────────────────
def test_nothing_evictable_is_todays_behaviour(rig):
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 512 * MIB
    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan == {"action": "proceed", "evicted": [], "freed_bytes": 0,
                    "reason": None}
    assert rig.evicted == []


def test_only_protected_residents_is_todays_behaviour(rig):
    # A card held entirely by protected residents offers ZERO reclaimable, so the
    # planner must not size up at all — and must not touch them.
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 512 * MIB
    rig.residents["locked"] = {"vram_bytes": 8 * GIB, "host_mode": "in_process"}
    rig.static.add("locked")
    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan == {"action": "proceed", "evicted": [], "freed_bytes": 0,
                    "reason": None}
    assert rig.evicted == []


# ── unmeasurable -> byte-identical to today (degrade-not-guess) ─────────────
def test_unmeasurable_resident_footprint_does_not_size_up(rig):
    # A resident the worker could not join to nvidia-smi (vram_bytes 0) makes the
    # reclaimable estimate UNCONFIDENT. Doctrine: fall back to today's plan-
    # against-actual rather than size up on a guessed number.
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 512 * MIB
    rig.residents["unjoinable"] = {"vram_bytes": 0, "host_mode": "subprocess"}
    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan == {"action": "proceed", "evicted": [], "freed_bytes": 0,
                    "reason": None}
    assert rig.evicted == []


def test_unmeasurable_free_vram_fails_open_unchanged(rig, monkeypatch):
    # The device read is gone entirely. The admission fails OPEN exactly as today
    # (no size-up can be attempted without a free figure to plan against).
    rig.card["need"] = 512 * MIB
    rig.residents["idle"] = {"vram_bytes": 8 * GIB, "host_mode": "subprocess"}
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: None)
    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan["action"] == "proceed"
    assert "fail open" in plan["note"]
    assert rig.evicted == []


def test_reclaimable_none_when_no_candidates():
    assert A._reclaimable_vram_bytes([]) is None


def test_reclaimable_sums_only_measured_candidates():
    assert A._reclaimable_vram_bytes(
        [{"model_key": "a", "vram_bytes": 2 * GIB},
         {"model_key": "b", "vram_bytes": 1 * GIB}]) == 3 * GIB
    # one unmeasurable row poisons the whole estimate — never guess
    assert A._reclaimable_vram_bytes(
        [{"model_key": "a", "vram_bytes": 2 * GIB},
         {"model_key": "b", "vram_bytes": 0}]) is None


# ── protections are never counted as reclaimable ────────────────────────────
@pytest.mark.parametrize("kind", ["static", "replying", "busy_slot", "comfy",
                                  "queued_ahead"])
def test_protected_residents_are_never_reclaimable(rig, monkeypatch, kind):
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 512 * MIB
    host_mode = "comfy" if kind == "comfy" else "subprocess"
    rig.residents["neighbour"] = {"vram_bytes": 8 * GIB, "host_mode": host_mode}
    if kind == "static":
        rig.static.add("neighbour")
    elif kind == "replying":
        rig.replying.add("neighbour")
    elif kind == "busy_slot":
        rig.busy_slots.add("neighbour")
    elif kind == "queued_ahead":
        monkeypatch.setattr(A, "_queued_ahead_of", lambda subj: {"neighbour"})

    cands, prot = A._partition_residents(_State(), "flux2-klein-9b")
    assert cands == [], f"{kind} resident must not be a candidate"
    assert prot and prot[0]["model_key"] == "neighbour"
    assert A._reclaimable_vram_bytes(cands) is None

    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan["action"] == "proceed"
    assert rig.evicted == []                 # protections are INVIOLABLE


def test_subject_is_never_its_own_reclaimable(rig):
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 512 * MIB
    rig.residents["flux2-klein-9b"] = {"vram_bytes": 8 * GIB,
                                       "host_mode": "subprocess"}
    cands, _ = A._partition_residents(_State(), "flux2-klein-9b")
    assert cands == []


# ── an explicit operator n_gpu_layers still wins ────────────────────────────
@pytest.mark.parametrize("raw", ["-1", "0", "24", "cpu", "off"])
def test_explicit_ngl_is_unaffected(rig, monkeypatch, raw):
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", raw)
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 512 * MIB
    rig.residents["idle"] = {"vram_bytes": 8 * GIB, "host_mode": "subprocess"}
    rig.lru["idle"] = 100.0
    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    # Core verdict only: a CPU/RAM-only intent (0/cpu/off) legitimately adds
    # an informational "placement intent puts 0 B on the GPU" note (the
    # 2026-07-29 placement-intent re-price) — additive, so not asserted away.
    assert {k: plan.get(k) for k in ("action", "evicted", "freed_bytes",
                                     "reason")} == {
        "action": "proceed", "evicted": [], "freed_bytes": 0, "reason": None}
    assert rig.evicted == []                 # the AUTO path only


def test_planner_declines_on_explicit_intent(rig, monkeypatch):
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", "24")
    cands, _ = A._partition_residents(_State(), "flux2-klein-9b")
    assert A._plan_autofit_against_reclaimable(
        _State(), "flux2-klein-9b", cands, 4 * GIB) is None


# ── the MoE path is untouched ───────────────────────────────────────────────
def test_moe_split_verdict_is_not_a_size_up(rig, monkeypatch):
    # A MoE model's placement is decided by the expert split, not layer autofit.
    # When the admission re-targets to a MoE split it returns BEFORE the size-up
    # stage — assert the split verdict is emitted untouched (no size_up key, ngl
    # -1 + n_cpu_moe), so slots.py's size-up branch (which requires size_up)
    # cannot mistake it for one.
    rig.card["free"] = 6 * GIB
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: {
        "total": 40 * GIB, "weights": 40 * GIB, "kv": 0,
        "moe_split": {"path": rig.gguf, "n_cpu_moe": 99,
                      "gpu_total": 2 * GIB, "cpu_bytes": 38 * GIB,
                      "expert_count": 128, "expert_used_count": 8,
                      "sparsity": 0.06}})
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: 64 * GIB)
    # the expert guard prices cpu_bytes against the BOX (2026-08-28 mmap ruling),
    # so pin the host total too or a small CI runner refuses the split
    monkeypatch.setattr(A, "_ram_total_bytes", lambda: 128 * GIB)
    rig.residents["idle"] = {"vram_bytes": 1 * GIB, "host_mode": "subprocess"}

    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan["action"] == "partial"
    assert plan["n_gpu_layers"] == -1
    assert plan["n_cpu_moe"] == 99
    assert "size_up" not in plan
    assert rig.evicted == []


def test_non_gguf_subject_is_never_sized_up(rig):
    # _served_gguf_geometry answers (None, None) for anything but the GGUF —
    # a transformers subject keeps today's admission exactly.
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 512 * MIB
    rig.residents["idle"] = {"vram_bytes": 8 * GIB, "host_mode": "subprocess"}
    plan = A._vram_evict_to_fit(_State(), "some-transformers-model")
    assert plan == {"action": "proceed", "evicted": [], "freed_bytes": 0,
                    "reason": None}
    assert rig.evicted == []


def test_size_up_declines_when_eviction_buys_no_layers(rig):
    # The card is already roomy enough that autofit plans every layer — there is
    # nothing to size up TO, so no neighbour is disturbed.
    rig.card["free"] = 12 * GIB
    rig.card["need"] = 512 * MIB
    rig.residents["idle"] = {"vram_bytes": 1 * GIB, "host_mode": "subprocess"}
    assert SP.autofit_gpu_layers(rig.gguf, free_vram=12 * GIB) == -1
    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan == {"action": "proceed", "evicted": [], "freed_bytes": 0,
                    "reason": None}
    assert rig.evicted == []


def test_size_up_evicts_the_minimum_set(rig):
    # Three idle neighbours, but the target is reached after the first — the
    # size-up loop re-measures each round and must stop, exactly like the fit
    # loop. Coldest (lowest last_used) first.
    rig.card["free"] = 4 * GIB
    rig.card["need"] = 256 * MIB
    for i, mk in enumerate(("cold", "warm", "hot")):
        rig.residents[mk] = {"vram_bytes": 8 * GIB, "host_mode": "subprocess"}
        rig.lru[mk] = 100.0 * (i + 1)
    plan = A._vram_evict_to_fit(_State(), "flux2-klein-9b")
    assert plan["action"] == "partial"
    assert rig.evicted == ["cold"]           # minimum set, coldest-first
    assert plan["n_gpu_layers"] == -1


# ── the slot seam: the plan reaches the child's argv ────────────────────────
def test_slot_threads_the_size_up_layer_count_into_the_load(monkeypatch):
    """slots.endpoint_for must thread a size-up verdict's n_gpu_layers into the
    /load body even when the ceiling gate PASSES — the flux2 path, where nothing
    must be evicted and today's code therefore never consults make-room at all."""
    from hugpy_engine.serve import slots as S

    posted = {}
    monkeypatch.setattr(S, "_FIT_CHECK", lambda mk: True)     # ceiling passes
    monkeypatch.setattr(S, "_MAKE_ROOM", lambda mk: {
        "action": "partial", "n_gpu_layers": -1, "evicted": ["idle"],
        "freed_bytes": 4 << 30, "note": "eviction-aware autofit",
        "size_up": {"target_layers": -1, "layers_without_eviction": 21}})

    pool = S.SlotPool(urls=["http://x:8101"])
    monkeypatch.setattr(pool, "statuses",
                        lambda: [{"_control": "http://x:8101", "healthy": True,
                                  "model_key": None}])

    def _post(url, body, timeout):
        posted.update(body)
        return {"endpoint": "http://x:9101"}
    monkeypatch.setattr(S, "_post", _post)

    ep = pool.endpoint_for("flux2-klein-9b")
    assert ep == "http://x:9101"
    assert posted["n_gpu_layers"] == -1


def test_slot_leaves_the_load_untouched_without_a_size_up(monkeypatch):
    """A plain 'proceed' verdict (nothing to size up) must not inject any size-up
    n_gpu_layers into the /load body. The alloc-mismatch-provenance round
    (2026-09-23) makes every /load body additionally carry alloc_requested /
    alloc_source so a seat records what it was loaded FOR — with nothing
    requested those are the 'default' provenance stamp, and no reload_reason
    is present. The load-bearing invariant here is unchanged: NO size-up
    n_gpu_layers is injected."""
    from hugpy_engine.serve import slots as S

    posted = {}
    monkeypatch.setattr(S, "_FIT_CHECK", lambda mk: True)
    monkeypatch.setattr(S, "_MAKE_ROOM", lambda mk: {
        "action": "proceed", "evicted": [], "freed_bytes": 0, "reason": None})
    pool = S.SlotPool(urls=["http://x:8101"])
    monkeypatch.setattr(pool, "statuses",
                        lambda: [{"_control": "http://x:8101", "healthy": True,
                                  "model_key": None}])
    monkeypatch.setattr(S, "_post",
                        lambda url, body, timeout: (posted.update(body),
                                                    {"endpoint": "e"})[1])
    pool.endpoint_for("m")
    # the seat request itself: model_key only, NO size-up n_gpu_layers injected
    assert posted["model_key"] == "m"
    assert "n_gpu_layers" not in posted, "no size-up injected on a plain proceed"
    # provenance stamp: nothing was requested -> 'default' source, empty request
    assert posted["alloc_requested"] == {"alloc_mode": None, "n_gpu_layers": None}
    assert posted["alloc_source"] == {"kind": "default"}
    assert "reload_reason" not in posted, "no reload on a first seat"
    # exactly those keys — the body carries no other stowaways
    assert set(posted) == {"model_key", "alloc_requested", "alloc_source"}


def test_slot_explicit_ngl_in_opts_skips_the_size_up(monkeypatch):
    """An explicit operator n_gpu_layers in opts wins — make-room is not even
    consulted for a size-up, and the operator's number reaches the child."""
    from hugpy_engine.serve import slots as S

    called = []
    posted = {}
    monkeypatch.setattr(S, "_FIT_CHECK", lambda mk: True)
    monkeypatch.setattr(S, "_MAKE_ROOM", lambda mk: called.append(mk) or {
        "action": "partial", "n_gpu_layers": -1, "size_up": {}})
    pool = S.SlotPool(urls=["http://x:8101"])
    monkeypatch.setattr(pool, "statuses",
                        lambda: [{"_control": "http://x:8101", "healthy": True,
                                  "model_key": None}])
    monkeypatch.setattr(S, "_post",
                        lambda url, body, timeout: (posted.update(body),
                                                    {"endpoint": "e"})[1])
    pool.endpoint_for("m", opts={"n_gpu_layers": 12})
    assert called == []
    assert posted["n_gpu_layers"] == 12
