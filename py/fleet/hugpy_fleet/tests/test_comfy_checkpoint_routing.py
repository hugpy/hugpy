"""ComfyUI checkpoint-presence routing (2026-09-24, comfy-sd-turbo × computron).

Two live defects this covers:

  1. FEASIBLE-PRESENCE for comfy models did not count a worker's ADVERTISED
     checkpoint as "the model's files are present". Central usually cannot
     resolve a worker-synthesized ``comfy-<stem>`` row (it holds no comfy files),
     so ``_model_engine`` returned None, the whole comfy branch collapsed, and a
     box that actually advertises the checkpoint (ae-worker) was treated as a
     non-comfy on-disk miss. Fix: ``_is_comfy_model`` falls back to the
     ``comfy-`` key convention, and ``_comfy_has_checkpoint`` gains a STEM
     fallback so an advertised ``sd_turbo.safetensors`` satisfies
     ``comfy-sd-turbo`` even when central can't resolve the exact filename.

  2. A comfy worker whose advertised checkpoint list is KNOWN and does NOT carry
     the checkpoint was still routable when it was a DESIGNATED / home box (a
     stale ``worker_assignments`` row -> computron, which advertises 10 other
     checkpoints but not sd_turbo). ComfyUI then rejects the graph with
     "ckpt_name '<f>' not in [...]". Fix: a checkpoint-presence gate that applies
     to home boxes too, with a specific named reason in the no_worker
     diagnostics and a say-why WARNING.

Runs standalone (``python tests/test_comfy_checkpoint_routing.py``) and under
pytest (the verify pipeline).
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_store_isolation import isolated_worker_store, swap_worker_store

W = importlib.import_module("hugpy_fleet.central.workers")

MODEL = "comfy-sd-turbo"                       # worker-synthesized; central can't resolve it
CKPT = "sd_turbo.safetensors"
OTHER_CKPTS = [
    "DreamShaper_8_pruned.safetensors",
    "darkSushiMixMix_225D.safetensors",
    "ghostmix_v20Bakedvae.safetensors",
]
TASK = "text-to-image"


def _fresh_store():
    store, _tmp = isolated_worker_store(prefix="hugpy-comfy-ckpt-")
    return store


def _add_comfy(store, wid, checkpoints, *, models=(), available=True):
    """An APPROVED, online comfy worker advertising ``checkpoints``."""
    store.register(name=wid, url=f"http://{wid}:9100", worker_id=wid,
                   models=list(models))
    store.set_admission(wid, "approved")
    store.heartbeat(wid, comfy={"available": available,
                                "checkpoints": list(checkpoints)})


# ---------------------------------------------------------------------------
# Helper-level unit tests (hermetic; no store, no config resolution needed).
# ---------------------------------------------------------------------------

def test_is_comfy_model_falls_back_to_the_comfy_key_convention():
    # central can't resolve the synthesized row -> _model_engine is None -> the
    # explicit "comfy-" prefix is the honest marker.
    assert W._model_engine(MODEL) is None
    assert W._is_comfy_model(MODEL) is True
    # a plain (transformers) model that merely shares the tail is NOT comfy.
    assert W._is_comfy_model("sd-turbo") is False


def test_comfy_stem_normalization_matches_the_sweep_mint():
    assert W._comfy_key_stem(MODEL) == "sd-turbo"
    assert W._comfy_key_stem("sd-turbo") == ""     # not a comfy- key
    assert W._comfy_ckpt_stem("sd_turbo.safetensors") == "sd-turbo"
    assert W._comfy_ckpt_stem("checkpoints/SD_Turbo.ckpt") == "sd-turbo"


def test_comfy_has_checkpoint_stem_fallback_and_exact_basename():
    have = {"comfy": {"available": True, "checkpoints": [CKPT] + OTHER_CKPTS}}
    lack = {"comfy": {"available": True, "checkpoints": list(OTHER_CKPTS)}}
    # stem fallback: unresolved comfy-sd-turbo satisfied by advertised sd_turbo.
    assert W._comfy_has_checkpoint(have, MODEL) is True
    assert W._comfy_has_checkpoint(lack, MODEL) is False
    # exact-filename path (curated staple style) is unchanged/authoritative.
    assert W._comfy_has_checkpoint(have, "x", _want=CKPT) is True
    assert W._comfy_has_checkpoint(lack, "x", _want=CKPT) is False
    # an ABSENT advertised list is UNKNOWN, never a claimed presence.
    assert W._comfy_has_checkpoint({"comfy": {"available": True}}, MODEL) is False


# ---------------------------------------------------------------------------
# DEFECT 1 — a non-designated comfy box that advertises the checkpoint is feasible.
# ---------------------------------------------------------------------------

def test_advertised_checkpoint_makes_a_non_designated_comfy_box_feasible():
    store = _fresh_store()
    # No designation anywhere (empty alloc scope): ae-worker catches by comfy
    # presence alone (the 2026-09-24 de-facto placement), now via the stem
    # fallback for the synthesized key.
    _add_comfy(store, "aeworker", [CKPT] + OTHER_CKPTS)
    ids = {w["id"] for w in store.workers_for_model(MODEL, task=TASK)}
    assert ids == {"aeworker"}
    picked = store.pick_for_model(MODEL, task=TASK)
    assert picked and picked["id"] == "aeworker"


# ---------------------------------------------------------------------------
# DEFECT 2 — a DESIGNATED/home comfy box lacking the checkpoint is gated out,
# with a specific named reason, never routed to.
# ---------------------------------------------------------------------------

def test_designated_home_box_without_the_checkpoint_is_gated_out():
    store = _fresh_store()
    # computron carries a (stale) designation for the model but its advertised
    # checkpoints do NOT include sd_turbo — it would 500 with "ckpt_name not in".
    _add_comfy(store, "computron", OTHER_CKPTS, models=[MODEL])

    # capture the say-why WARNING off the workers module logger
    class _Cap(logging.Handler):
        def __init__(self):
            super().__init__()
            self.msgs = []
        def emit(self, record):
            self.msgs.append(record.getMessage())
    cap = _Cap()
    wlog = logging.getLogger(W.__name__)
    wlog.addHandler(cap)
    wlog.setLevel(logging.WARNING)
    try:
        res = store.workers_for_model(MODEL, task=TASK)
    finally:
        wlog.removeHandler(cap)

    assert res == []          # home box gated on checkpoint absence
    assert store.pick_for_model(MODEL, task=TASK) is None
    said = any("checkpoint not on any comfy worker" in m and MODEL in m
               for m in cap.msgs)
    assert said, cap.msgs


def test_designated_box_with_the_checkpoint_still_serves():
    """The gate is affirmative: a home box that DOES advertise the checkpoint is
    untouched (no regression for a correctly-placed comfy model)."""
    store = _fresh_store()
    _add_comfy(store, "computron", [CKPT] + OTHER_CKPTS, models=[MODEL])
    ids = {w["id"] for w in store.workers_for_model(MODEL, task=TASK)}
    assert ids == {"computron"}


def test_home_box_with_unknown_checkpoint_list_is_not_gated():
    """A live comfy box that advertises comfy.available but NO checkpoint list is
    UNKNOWN, not proven-absent — degrade-not-guess keeps it a candidate."""
    store = _fresh_store()
    store.register(name="c2", url="http://c2:9100", worker_id="c2", models=[MODEL])
    store.set_admission("c2", "approved")
    store.heartbeat("c2", comfy={"available": True})   # no 'checkpoints' key
    ids = {w["id"] for w in store.workers_for_model(MODEL, task=TASK)}
    assert ids == {"c2"}


# ---------------------------------------------------------------------------
# DEFECT 2 — the no_worker diagnostics name the specific gate reason.
# ---------------------------------------------------------------------------

def test_no_worker_diagnostics_name_the_checkpoint_absence():
    with swap_worker_store(prefix="hugpy-comfy-diag-") as store:
        _add_comfy(store, "computron", OTHER_CKPTS, models=[MODEL])

        skips = W.no_worker_skips(MODEL, task=TASK)
        reason = skips.get("computron", "")
        assert "advertised checkpoints" in reason or "ckpt_name" in reason, reason

        detail = W.explain_no_worker(MODEL, task=TASK)
        # a live comfy box without the checkpoint -> the honest "no live comfy
        # worker advertises its checkpoint" line, not "" and not a dead-backend
        # reason.
        assert "checkpoint" in detail and MODEL in detail, detail


def test_explain_transient_when_a_present_box_exists():
    with swap_worker_store(prefix="hugpy-comfy-diag2-") as store:
        _add_comfy(store, "aeworker", [CKPT] + OTHER_CKPTS)
        # a capable+present box exists -> the miss is transient, explain says "".
        assert W.explain_no_worker(MODEL, task=TASK) == ""


def test_prefs_scope_message_states_fact_not_a_contradicted_outcome():
    """The ordered-preference scope log must NOT assert "refusing" — under the
    feasible default the caller falls back, so the old wording contradicted the
    very next "falling back to the FEASIBLE set" line."""
    class _Cap(logging.Handler):
        def __init__(self):
            super().__init__()
            self.msgs = []
        def emit(self, record):
            self.msgs.append(record.getMessage())
    cap = _Cap()
    wlog = logging.getLogger(W.__name__)
    wlog.addHandler(cap)
    wlog.setLevel(logging.WARNING)
    try:
        # no candidate matches the preference -> the empty-scope warning fires
        kept = W._prefs_scope([], ["computron", "a-brain-Super-Server"], MODEL)
    finally:
        wlog.removeHandler(cap)
    assert kept == []
    joined = " ".join(cap.msgs)
    assert "refusing rather than landing off-list" not in joined, joined
    assert "NONE of them is an eligible candidate" in joined, joined


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok - {fn.__name__}")
    print(f"\nall {len(fns)} checks passed")
