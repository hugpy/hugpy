"""KNOWN-GOOD CONTRACT — no hollow residents: membership is never residency.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/core-isolation-step2-2026-09-29.md (F4).
Source under test: hugpy_fleet/worker/agent.py (loaded_model_keys,
_hollow_model_keys, _loaded_models_live_check, _forget_resident, _evict_model,
_allocations), hugpy_engine/serve/slots.py (_drop_resident).

LIVE CASES (2026-09-29): a hollow LlamaCppChatRunner left in dispatch
_INSTANCES after its slot seat was evicted made the worker report
"Qwen3-Coder-Next-GGUF" in loaded_models while the SAME heartbeat's allocation
row said materialized:false — central then said "loaded and idle, but the
request failed". Ceiling-evicted victims stayed in loaded_models as
kind=ram rss=None rows; the next plan named an already-gone victim from the
stale pid registry.
"""
from __future__ import annotations

import importlib

import pytest

A = importlib.import_module("hugpy_fleet.worker.agent")
D = importlib.import_module("hugpy_engine.dispatch.dispatch")
G = importlib.import_module("hugpy_engine.llama.runners.get")
slots = importlib.import_module("hugpy_engine.serve.slots")
pidreg = importlib.import_module("hugpy_fleet.worker.pid_registry")
from hugpy_engine.llama.runners.src.base_runner import LlamaCppBaseRunner  # noqa: E402

GIB = 1 << 30


class _HollowLlama(LlamaCppBaseRunner):
    """A lazy wrapper exactly like the real ones: constructing it loads nothing."""
    def __init__(self, mk):
        self.model_key = mk

    def _chat_complete(self, *a, **k):
        raise AssertionError("not reached")

    def _raw_complete(self, *a, **k):
        raise AssertionError("not reached")

    def _iter_stream(self, *a, **k):
        raise AssertionError("not reached")


class _Loaded:
    base_url = None
    llm = object()                              # a materialized Llama handle


class _Opaque:
    """A non-llama runner nothing can introspect: residency UNKNOWN."""
    def __init__(self, mk):
        self.model_key = mk


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(A, "_ollama_live_model_keys", lambda running: set())
    monkeypatch.setattr(A.ollama_adapter if hasattr(A, "ollama_adapter") else
                        importlib.import_module("hugpy_fleet.worker.ollama_adapter"),
                        "running", lambda: {})
    with A._MATERIALIZED_LOCK:
        A._MATERIALIZED.clear()
    A._HOLLOW_LOGGED.clear()
    for k in [k for k in list(D._INSTANCES) if k[0] in ("hollow", "live", "opaque", "victim")]:
        D._INSTANCES.pop(k, None)
    for k in ("hollow", "live", "opaque", "victim"):
        G._LLAMA_INSTANCES.pop(k, None)
    yield
    for k in [k for k in list(D._INSTANCES) if k[0] in ("hollow", "live", "opaque", "victim")]:
        D._INSTANCES.pop(k, None)
    for k in ("hollow", "live", "opaque", "victim"):
        G._LLAMA_INSTANCES.pop(k, None)
    pidreg.forget("victim")


def test_loaded_models_reports_only_measured_or_unknowable_residents(monkeypatch, caplog):
    """INVARIANT (F4a/F4c): a dispatch wrapper whose weights are measured ABSENT
    is not a loaded model; a materialized handle is; an unintrospectable
    runner stays reported (degrade-not-guess). The hollow key is logged once.
    Established: core isolation step 2 (2026-09-29)."""
    import logging
    monkeypatch.setitem(D._INSTANCES, ("hollow", "chat"), _HollowLlama("hollow"))
    monkeypatch.setitem(D._INSTANCES, ("live", "chat"), _HollowLlama("live"))
    monkeypatch.setitem(G._LLAMA_INSTANCES, "live", _Loaded())
    monkeypatch.setitem(D._INSTANCES, ("opaque", "chat"), _Opaque("opaque"))
    with caplog.at_level(logging.WARNING, logger=A.logger.name):
        first = A.loaded_model_keys()
        again = A.loaded_model_keys()
    assert first == ["live", "opaque"] and again == first
    assert A._hollow_model_keys(["hollow", "live", "opaque"]) == ["hollow"]
    warns = [r for r in caplog.records if "hollow runner(s) NOT reported" in r.getMessage()]
    assert len(warns) == 1 and "['hollow']" in warns[0].getMessage()


def test_heartbeat_live_check_names_every_entry_without_a_runner_or_slot_child(monkeypatch):
    """INVARIANT (F4c): every loaded_models entry has a live runner or a healthy
    slot child; the check returns the violators and the beat drops them.
    Established: step 2."""
    monkeypatch.setitem(D._INSTANCES, ("hollow", "chat"), _HollowLlama("hollow"))
    monkeypatch.setitem(D._INSTANCES, ("victim", "chat"), _HollowLlama("victim"))
    monkeypatch.setitem(D._INSTANCES, ("live", "chat"), _HollowLlama("live"))
    monkeypatch.setitem(G._LLAMA_INSTANCES, "live", _Loaded())
    slots_now = [{"model_key": "victim", "healthy": True, "child_pid": 4242},
                 {"model_key": None, "healthy": True, "child_pid": None}]
    bad = A._loaded_models_live_check(["hollow", "victim", "live", "unknown-key"], slots_now)
    assert bad == ["hollow"]                    # victim is seated; live has a handle; unknown is unknowable
    assert A._loaded_models_live_check(["victim"], [{"model_key": "victim", "healthy": False,
                                                     "child_pid": 4242}]) == ["victim"]
    assert A._loaded_models_live_check([], None) == []


def test_slot_eviction_forgets_the_resident_everywhere(monkeypatch):
    """INVARIANT (F4b): a successful _evict_model — the ONE eviction verb — drops
    the victim from dispatch _INSTANCES, the llama runner cache, the
    materialized flag and the pid registry at eviction time, on the SLOT path
    too (before: only the in-process branch cascaded). Established: step 2."""
    unloaded = []
    monkeypatch.setattr(A, "_model_framework", lambda mk: "llama_cpp")
    monkeypatch.setattr(A, "_evict_gate", lambda mk: (True, ""))
    monkeypatch.setattr(A, "_resolve_slot_handle",
                        lambda mk: {"control_url": "http://s0", "child_pid": 4242,
                                    "endpoint": "http://s0/v1"})
    monkeypatch.setattr(A, "_model_footprint_before_evict", lambda *a, **k: {"vram_bytes": GIB})
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 2 * GIB)
    monkeypatch.setattr(A, "_free_ram_bytes", lambda: 8 * GIB)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(slots.SlotPool, "unload", lambda self, url: unloaded.append(url) or {"ok": True})
    oa = importlib.import_module("hugpy_fleet.worker.ollama_adapter")
    monkeypatch.setattr(oa, "model_name", lambda mk: None)

    class _SlotBacked:
        base_url = "http://s0/v1"
        llm = None
    monkeypatch.setitem(D._INSTANCES, ("victim", "chat"), _HollowLlama("victim"))
    monkeypatch.setitem(G._LLAMA_INSTANCES, "victim", _SlotBacked())
    with A._MATERIALIZED_LOCK:
        A._MATERIALIZED.add("victim")
    pidreg.record_launch("victim", 4242, "subprocess")

    res = A._evict_model(A.WorkerState.__new__(A.WorkerState) if hasattr(A, "WorkerState") else object(),
                         "victim")
    assert res["evicted"] is True and res["host_mode"] == "slot" and unloaded == ["http://s0"]
    assert ("victim", "chat") not in D._INSTANCES
    assert "victim" not in G._LLAMA_INSTANCES
    with A._MATERIALIZED_LOCK:
        assert "victim" not in A._MATERIALIZED
    assert pidreg.verify("victim") is None
    assert "victim" not in res["loaded_models"]


def test_slot_pool_victim_drop_cascades_into_the_dispatch_cache(monkeypatch):
    """INVARIANT (F4b, engine side): a slot victim bumped by the pool leaves the
    dispatch runner cache as well as the llama runner cache (_drop_resident ⊇
    _drop_runner). Established: step 2."""
    class _SlotBacked:
        base_url = "http://s0/v1"
        llm = None
    monkeypatch.setitem(D._INSTANCES, ("victim", "chat"), _HollowLlama("victim"))
    monkeypatch.setitem(G._LLAMA_INSTANCES, "victim", _SlotBacked())
    slots._drop_resident("victim")
    assert ("victim", "chat") not in D._INSTANCES and "victim" not in G._LLAMA_INSTANCES
    slots._drop_resident(None)                   # never raises


def test_slot_allocation_rows_carry_measured_materialization(monkeypatch):
    """INVARIANT (F4): a slot row states `materialized` from the slot's own
    probe (healthy child = True, mid-load = False, unknown = omitted), the same
    key the ram rows carry, so central consults ONE field. Established: step 2."""
    monkeypatch.setattr(A, "_gpu_process_vram", lambda: {})
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_loaded_detail", lambda: {})
    monkeypatch.setattr(A, "_inprocess_gpu_bytes", lambda: {})
    monkeypatch.setattr(A, "_slot_total_layers_fallback", lambda mk: None)
    rows = A._allocations([
        {"model_key": "seated", "healthy": True, "child_pid": 1, "slot_id": 1, "_control": "http://s0"},
        {"model_key": "loading", "healthy": False, "child_pid": None, "slot_id": 2, "_control": "http://s1"},
        {"model_key": "old-slot", "slot_id": 3, "_control": "http://s2"},
        {"model_key": None, "healthy": True, "slot_id": 4, "_control": "http://s3"},
    ])
    by = {r["model_key"]: r for r in rows}
    assert by["seated"]["kind"] == "slot" and by["seated"]["materialized"] is True
    assert by["loading"]["materialized"] is False
    assert "materialized" not in by["old-slot"]
    assert None not in by
