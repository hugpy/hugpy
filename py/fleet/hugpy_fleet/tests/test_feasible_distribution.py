"""Feasibility-based distribution (operator ruling 2026-09-24) + the BUG-1 /
BUG-2 routing fixes.

BUG 1: a loaded+designated worker was refused because the model's ordered worker
preference (``worker_prefs``) named a token that matched no worker (the unix user
"aeb" written where the worker name "ae-worker" belonged). Under the "feasible"
distribution default, an unmet preference FALLS BACK to the feasible set instead
of refusing, so the loaded home routes.

The distribution mode / strict flag are monkeypatched at the module seam so these
tests never read the live serve-overrides / worker_wildcard stores.
"""

from __future__ import annotations

import pytest

from hugpy_fleet.central import workers as W

from worker_store_isolation import isolated_worker_store


def _store(prefix):
    store, _tmp = isolated_worker_store(prefix=prefix)

    def add(wid, models, **kw):
        store.register(name=wid, url=f"http://{wid}:9100", worker_id=wid,
                       models=models, **kw)
        store.set_admission(wid, "approved")
    return store, add


@pytest.fixture
def modes(monkeypatch):
    """Control distribution mode / strict / prefs at the workers.py seam."""
    state = {"feasible": True, "strict": False, "prefs": []}
    monkeypatch.setattr(W, "_distribution_feasible", lambda: state["feasible"])
    monkeypatch.setattr(W, "_model_strict", lambda mk: state["strict"])
    monkeypatch.setattr(W, "placement_policy",
                        lambda mk: (list(state["prefs"]), False, {}))
    return state


MODEL = "org~Coder-Next"


def test_bug1_unmet_preference_falls_back_to_loaded_home(modes):
    """The exact live bug: designated+loaded ae-worker, worker_prefs=['aeb'] (a
    token matching no worker). Feasible mode -> fallback -> routes to the home."""
    modes["prefs"] = ["aeb"]                      # matches neither name nor id
    store, add = _store("hugpy-feas-bug1-")
    add("ae-worker", [MODEL])
    store.heartbeat("ae-worker", loaded_models=[MODEL])
    # Sanity: the preference matches nobody.
    assert W._prefs_scope(store.workers_for_model(MODEL), ["aeb"], MODEL) == []
    picked = store.pick_for_model(MODEL)
    assert picked is not None and picked["id"] == "ae-worker"


def test_strict_model_keeps_hard_fence(modes):
    """A ``strict`` model still REFUSES when no preferred worker is eligible."""
    modes["prefs"] = ["aeb"]
    modes["strict"] = True
    store, add = _store("hugpy-feas-strict-")
    add("ae-worker", [MODEL])
    store.heartbeat("ae-worker", loaded_models=[MODEL])
    assert store.pick_for_model(MODEL) is None


def test_designated_mode_is_byte_identical_refusal(modes):
    """Legacy ``designated`` mode: an unmet preference REFUSES, as before."""
    modes["prefs"] = ["aeb"]
    modes["feasible"] = False
    store, add = _store("hugpy-feas-desig-")
    add("ae-worker", [MODEL])
    store.heartbeat("ae-worker", loaded_models=[MODEL])
    assert store.pick_for_model(MODEL) is None


def test_feasible_catch_requires_on_disk_presence(modes):
    """An UNDESIGNATED model reaches any worker holding its files ON DISK (feasible
    mode), and ONLY those — a box without the files on disk is not offered (no
    synchronous transfer in the route)."""
    STRAY = "org~Undesignated"
    store, add = _store("hugpy-feas-catch-")
    add("has-disk", [])                            # not designated for STRAY
    store.heartbeat("has-disk", models_local=[STRAY])
    add("no-disk", [])                             # nothing on disk
    cands = store.workers_for_model(STRAY)
    assert [w["id"] for w in cands] == ["has-disk"]
    assert cands[0].get("_feasible_catch") is True
    assert (store.pick_for_model(STRAY) or {}).get("id") == "has-disk"


def test_designated_mode_seals_undesignated_on_disk(modes):
    """The same on-disk box is SEALED OUT in designated mode (byte-identical to
    pre-feature: undesignated + non-wildcard never catches)."""
    STRAY = "org~Undesignated"
    modes["feasible"] = False
    store, add = _store("hugpy-feas-seal-")
    add("has-disk", [])
    store.heartbeat("has-disk", models_local=[STRAY])
    assert store.workers_for_model(STRAY) == []
    assert store.pick_for_model(STRAY) is None


def test_feasible_fallback_prefers_loaded_then_on_disk(modes):
    """When the preference is unmet the fallback order is loaded > on-disk."""
    modes["prefs"] = ["nobody"]
    store, add = _store("hugpy-feas-order-")
    add("loaded-box", [])
    store.heartbeat("loaded-box", models_local=[MODEL], loaded_models=[MODEL])
    add("disk-box", [])
    store.heartbeat("disk-box", models_local=[MODEL])
    ranked = [w["id"] for w in store.candidates_for_model(MODEL)]
    assert ranked[0] == "loaded-box"
    assert set(ranked) == {"loaded-box", "disk-box"}


def test_no_worker_skips_names_the_preference_gate(modes):
    """BUG 2: a strict model whose preference matches nobody names the ordered-
    preference gate for the eligible home, not a bare 'not a candidate'."""
    modes["prefs"] = ["aeb"]
    modes["strict"] = True
    store, add = _store("hugpy-feas-skip-")
    orig = W.worker_store
    W.worker_store = store
    try:
        add("ae-worker", [MODEL])
        store.heartbeat("ae-worker", loaded_models=[MODEL])
        skips = W.no_worker_skips(MODEL)
        reason = skips.get("ae-worker", "")
        assert "ordered worker preference" in reason
        assert "['aeb']" in reason
    finally:
        W.worker_store = orig


def test_no_worker_skips_names_presence_gate(modes):
    """A non-home box that lacks the files ON DISK gets the specific presence gate
    (feasible mode needs on-disk presence — routing does no synchronous transfer),
    not a bare 'not a candidate'."""
    STRAY = "org~Undesignated"
    store, add = _store("hugpy-feas-skip2-")
    orig = W.worker_store
    W.worker_store = store
    try:
        add("bare-box", [])                        # online, no files on disk
        skips = W.no_worker_skips(STRAY)
        reason = skips.get("bare-box", "").lower()
        assert "on-disk presence" in reason or "not on this box's disk" in reason
        assert "not a routing candidate" not in reason
    finally:
        W.worker_store = orig
