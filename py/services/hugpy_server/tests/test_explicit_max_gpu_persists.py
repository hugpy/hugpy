"""EXPLICIT max-gpu must be distinguishable from CLEAR (bugfix 2026-07-25).

THE BUG (reproduced live before this fix): max-gpu was the ONE allocation mode
that could not be saved. Its wire encoding was ``{}``, but ``{}`` is ALSO the
"clear this override" signal in ``assign_model`` — so selecting max-gpu in the
console DELETED the row instead of writing it, and ``/assign`` returned a
success either way. The operator could not tell their choice had no effect.

It went unnoticed for a long time because a cleared row falls through to the
DERIVED default, which for most models IS max-gpu — so it did the right thing by
accident. That stopped when derived defaults landed: Qwen2.5-VL-7B-Instruct-GGUF
(12.29 GiB) on computron's ~7.6 GiB card derives **max-ram**, so the operator's
explicit max-gpu silently became max-ram and the load refused.

THE FIX: an explicit max-gpu persists as ``{"alloc_mode": "max-gpu"}`` (non-empty
-> writable), while ``{}`` keeps meaning CLEAR. The persisted key is central-side
bookkeeping and is STRIPPED at the emission seam (``spill_for``), so the wire
stays byte-identical to a blank max-gpu on every worker version.

Covers, in order:
  1. explicit max-gpu round-trips: write -> read back -> still max-gpu;
  2. ``{}`` still CLEARS (the console's "↺ Auto — derived" control);
  3. the manifest-orphan cleanup path's safety argument still holds — an empty
     spill remains structurally incapable of writing a contract;
  4. a DERIVED max-gpu still does NOT persist (it must re-derive);
  5. the worker wire is unchanged, incl. a gated-DOWN (pre-0.1.203) worker;
  6. the response reflects what was actually persisted.
"""
from __future__ import annotations

import os

import pytest

from hugpy_engine.alloc_modes import (
    mode_to_spill,
    normalize_spill,
    derive_alloc_mode,
    gate_spill_for_worker,
    default_allocation,
)
from hugpy_fleet.central import workers as W
from hugpy_server.app.routes import worker_routes as wr
from worker_store_isolation import swap_worker_store

GIB = 2 ** 30

# THE LIVE REPRO's shape: a 12.29 GiB GGUF on a ~7.6 GiB card. Its DERIVED
# default is max-ram (too big for the card, fits RAM) — so a lost max-gpu does
# NOT silently land on max-gpu here. That is exactly what made the bug visible.
_SIZES = {"vl7b": int(12.29 * GIB), "small": 2 * GIB}
_ENGINES = {"vl7b": "gguf", "small": "gguf"}


def _persisted(store, wid, mk):
    return (store._load()[wid].get("spill_by_model") or {}).get(mk)


def _register(store, name, url, pkg_version):
    w = store.register(name=name, url=url, pkg_version=pkg_version)
    wid = w["id"]
    with store._transaction() as wk:
        wk[wid]["gpus"] = [{"name": "RTX 4060", "memory_total": int(7.6 * GIB),
                            "memory_free": int(7.0 * GIB)}]
        wk[wid]["ram_total"] = 64 * GIB
    return wid


@pytest.fixture
def fleet(monkeypatch, tmp_path):
    """An isolated worker store (swapped into the module singleton so the
    module-level ``derived_allocation_for`` / ``spill_for`` see it) with the
    model manifest lookups stubbed to the live-repro sizes. Yields
    ``(store, wid)`` for computron with vl7b designated (blank assign)."""
    monkeypatch.setattr(W, "_model_size_bytes", lambda mk: _SIZES.get(mk))
    monkeypatch.setattr(W, "_model_engine", lambda mk: _ENGINES.get(mk))
    monkeypatch.setattr(W, "_model_moe_detail", lambda mk: None)
    monkeypatch.setattr(W, "_assign_memory_path",
                        lambda: str(tmp_path / "worker_assignments.json"))
    with swap_worker_store(prefix="hugpy-maxgpu-test-") as store:
        wid = _register(store, "computron", "http://c:9100", "0.1.209")
        store.assign_model(wid, "vl7b")
        yield store, wid


# ── 1. the encoding is distinguishable at the pure layer ────────────────────
def test_encoding_distinguishable_at_pure_layer():
    explicit = mode_to_spill("max-gpu", explicit_pick=True)
    derived = mode_to_spill("max-gpu")
    assert explicit == {"alloc_mode": "max-gpu"}, "NON-EMPTY so assign_model writes it"
    assert derived == {}, "a derived max-gpu stays unpersisted"
    assert explicit != derived, "the two encodings are distinguishable — THE WHOLE BUG"
    # both still DERIVE back to max-gpu (same mode, different provenance)
    assert derive_alloc_mode({"alloc_mode": "max-gpu"}) == "max-gpu"
    assert derive_alloc_mode({}) == "max-gpu"
    assert normalize_spill({"alloc_mode": "max-gpu"})[0] == {"alloc_mode": "max-gpu"}, \
        "a console-sent max-gpu survives normalization instead of collapsing"
    assert normalize_spill({"alloc_mode": "autofit"})[0] == {"alloc_mode": "max-gpu"}, \
        "the legacy alias resolves to the same canonical encoding"


# ── 4. the DERIVED default must NOT persist ─────────────────────────────────
def test_derived_default_does_not_persist(fleet):
    """Asserted against derived_allocation_for (the ALLOCATION view — the one
    spill_for actually emits from), not derived_default_for (the NAME view).
    The two DISAGREE for a large dense GGUF: the name view short-circuits every
    dense GGUF to "max-gpu" while the allocation view walks the tree and
    returns "max-ram". That divergence is PRE-EXISTING and outside this bugfix
    — the emitting seam is what governs behavior, so that is what this pins."""
    store, wid = fleet
    # PRECONDITION (the live repro): this model's DERIVED allocation is max-ram,
    # NOT max-gpu — so a dropped max-gpu really does change behavior.
    assert W.derived_allocation_for(wid, "vl7b")["mode"] == "max-ram"
    assert not _persisted(store, wid, "vl7b"), "a blank assign persists NOTHING"
    assert store.assign_model(wid, "small") is not None
    assert not _persisted(store, wid, "small"), "derived max-gpu on a SMALL model persists nothing"
    # default_allocation still encodes a derived max-gpu as {} (unpersisted);
    # unknown size is the degrade-not-guess path that derives max-gpu since
    # fits-whole now derives gpu-only (operator default order 2026-07-31).
    assert default_allocation("gguf", None, 24 * GIB, 64 * GIB)["spill"] == {}


# ── 1. explicit max-gpu ROUND-TRIPS ─────────────────────────────────────────
def test_explicit_max_gpu_round_trips(fleet):
    store, wid = fleet
    store.assign_model(wid, "vl7b", spill={"alloc_mode": "max-gpu"})
    row = _persisted(store, wid, "vl7b")
    assert row == {"alloc_mode": "max-gpu"}, "THE FIX: actually WRITTEN to the registry"
    assert derive_alloc_mode(row) == "max-gpu"
    # it BEATS the derivation — the operator's choice is no longer overwritten
    # by the max-ram default that broke the live case
    assert derive_alloc_mode(row) != W.derived_allocation_for(wid, "vl7b")["mode"]


# ── 5. the WIRE is unchanged: the worker must not see the bookkeeping key ───
def test_wire_strips_bookkeeping_key(fleet):
    """Sending a literal HUGPY_ALLOC_MODE=max-gpu would SUPPRESS the worker's
    auto MoE split (slot_agent bails on any k37 alloc_mode), making an explicit
    max-gpu behave WORSE than a blank one. So the key is stripped at emission."""
    store, wid = fleet
    store.assign_model(wid, "vl7b", spill={"alloc_mode": "max-gpu"})
    assert store.spill_for(wid, "vl7b") == {}, "byte-identical to a blank max-gpu"
    assert store.assign_model(wid, "small", spill={"alloc_mode": "max-ram"}) is not None
    assert store.spill_for(wid, "small") == {"alloc_mode": "max-ram"}, \
        "the strip does not disturb a real mode contract"


def test_gated_down_worker_gets_same_behaviour(fleet):
    """the downgrade target IS max-gpu, so behavior is identical."""
    store, _ = fleet
    oid = _register(store, "oldbox", "http://o:9100", "0.1.150")
    store.assign_model(oid, "vl7b", spill={"alloc_mode": "max-gpu"})
    assert store.spill_for(oid, "vl7b") == {}
    assert _persisted(store, oid, "vl7b") == {"alloc_mode": "max-gpu"}, \
        "persisted row untouched (it applies on update)"
    # The strip runs BEFORE the version gate, so the gate never fires on a key
    # that carries no instruction — an old worker must not log a fictional
    # "max-gpu downgraded to max-gpu" note.
    assert gate_spill_for_worker({}, "0.1.150", "oldbox") == ({}, None)


# ── 2. {} still CLEARS ──────────────────────────────────────────────────────
def test_empty_spill_still_clears(fleet):
    store, wid = fleet
    store.assign_model(wid, "vl7b", spill={"alloc_mode": "max-gpu"})
    store.assign_model(wid, "small", spill={"alloc_mode": "max-ram"})
    store.assign_model(wid, "vl7b", spill={})
    assert not _persisted(store, wid, "vl7b"), "the console's '↺ Auto — derived'"
    # after the clear the model tracks the DERIVATION again (max-ram here)
    assert W.derived_allocation_for(wid, "vl7b")["mode"] == "max-ram"
    assert store.spill_for(wid, "vl7b") == {"alloc_mode": "max-ram"}
    store.assign_model(wid, "small", spill={})
    assert not _persisted(store, wid, "small"), "clearing a max-ram contract works the same way"


# ── 3. the manifest-orphan cleanup path's SAFETY ARGUMENT still holds ───────
def test_orphan_cleanup_safety_argument(fleet):
    """worker_routes relaxes the manifest gate for a CLEAR of an already-
    designated key. Its safety rests on: an empty spill cannot write a contract."""
    store, wid = fleet
    store.assign_model(wid, "orphan", spill={"alloc_mode": "explicit", "gpu_mem_gib": 4.0})
    assert _persisted(store, wid, "orphan"), "a stale contract exists to be cleaned"
    store.assign_model(wid, "orphan", spill={})
    assert not _persisted(store, wid, "orphan"), "the empty-spill clear removes the row"
    store.assign_model(wid, "phantom", spill={})
    assert not _persisted(store, wid, "phantom"), \
        "an empty spill remains STRUCTURALLY incapable of writing a contract"


# ── 6. the response reflects what was PERSISTED ─────────────────────────────
def test_response_reflects_persisted_explicit(fleet):
    store, wid = fleet
    store.assign_model(wid, "vl7b", spill={"alloc_mode": "max-gpu"})
    res = wr._assign_allocation_result(store.get(wid), "vl7b", {"alloc_mode": "max-gpu"})
    assert res["persisted"] is True and res["mode"] == "max-gpu"
    assert res["source"] == "operator", "attributed to the OPERATOR, not the derivation"
    assert res["honored"] is True and res["requested_mode"] == "max-gpu"
    assert "persisted" in res["note"], "the choice will survive derivation changes"


def test_response_reflects_clear(fleet):
    store, wid = fleet
    store.assign_model(wid, "vl7b", spill={"alloc_mode": "max-gpu"})
    store.assign_model(wid, "vl7b", spill={})
    res_clear = wr._assign_allocation_result(store.get(wid), "vl7b", {})
    # the API no longer claims a write that did not happen
    assert res_clear["persisted"] is False and res_clear["source"] == "derived"
    assert res_clear["mode"] == "max-ram", "a clear reports the DERIVED mode it now tracks"
    assert res_clear["honored"] is True and res_clear["requested_mode"] is None


def test_dropped_write_is_reported_not_honored():
    """The dishonesty detector itself: if a write were ever silently dropped
    again, the response must say so rather than report success (this is what
    the old {'admission':'approved'} hid)."""
    res_lie = wr._allocation_state(None, {"alloc_mode": "max-gpu"})
    assert res_lie["honored"] is False and res_lie["persisted"] is False
    assert "NOT applied" in res_lie["note"]
