"""KNOWN-GOOD CONTRACT — eviction protection + eviction telemetry (fleet side).

Catalogue: docs/KNOWN-GOOD-CORE.md (area "evict + fit / allocation").
Source under test: hugpy_fleet/worker/agent.py (_partition_residents,
_busy_slot_models, _actively_replying, _ctx_pct), hugpy_fleet/worker/flex.py
(ctx_band_bounds, kv_at_ctx_pct), hugpy_fleet/central/evictions.py
(EvictionStore, emit_eviction_event, install_store_sink).

Deterministic: residents/slots/gen-gate/GPU inventory are stubbed on the agent
module; the telemetry store writes to a tmp SQLite file or a deliberately
broken connection. No worker process, no GPU.
"""
from __future__ import annotations

import importlib
import time

import pytest

A = importlib.import_module("hugpy_fleet.worker.agent")
flex = importlib.import_module("hugpy_fleet.worker.flex")
ev = importlib.import_module("hugpy_fleet.central.evictions")

GIB = 1 << 30


class _State:
    pass


# ---------------------------------------------------------------------------
# Eviction is protected while a slot is busy (h24, 2026-09-28)
# ---------------------------------------------------------------------------
@pytest.fixture
def residents(monkeypatch):
    rows = [
        {"model_key": "busy-m", "vram_bytes": 8 * GIB, "host_mode": "slot"},
        {"model_key": "idle-m", "vram_bytes": 6 * GIB, "host_mode": "slot"},
        {"model_key": "static-m", "vram_bytes": 4 * GIB, "host_mode": "slot"},
        {"model_key": "subject", "vram_bytes": 2 * GIB, "host_mode": "slot"},
    ]
    monkeypatch.setattr(A, "_vram_residents", lambda state: [dict(r) for r in rows])
    monkeypatch.setattr(A, "_busy_slot_models", lambda: {"busy-m"})
    monkeypatch.setattr(A.gen_gate, "in_flight", lambda mk: 0)
    monkeypatch.setattr(A, "_residency",
                        lambda mk: "static" if mk == "static-m" else "on_demand")
    monkeypatch.setattr(A, "_queued_ahead_of", lambda subject: set())
    monkeypatch.setattr(A, "detect_gpus", lambda: [])
    monkeypatch.setattr(A, "_target_device_index", lambda: None)
    return rows


def test_busy_slot_resident_is_protected_from_eviction(residents):
    """INVARIANT: _partition_residents (THE single definition of "what may be
    evicted") protects a resident whose native slot is BUSY (actively
    replying) and a static resident, and never lists the subject itself; an
    idle on-demand resident is the only candidate. Each protected row says why.
    Established: eviction-protection-two-classes-only ruling; re-verified in
    toolserver handoff h24 (2026-09-28: 'eviction is protected while slot busy')."""
    cands, protected = A._partition_residents(_State(), "subject")
    assert [r["model_key"] for r in cands] == ["idle-m"]
    why = {r["model_key"]: r["why"] for r in protected}
    assert "actively replying" in why["busy-m"]
    assert "static" in why["static-m"]
    assert "subject" not in why and "subject" not in [r["model_key"] for r in cands]


def test_in_flight_gen_gate_protects_and_unknown_is_fail_safe(monkeypatch):
    """INVARIANT: _actively_replying is MEASURED: gen-gate in-flight > 0 OR the
    slot busy set; when the gate cannot be read it fails SAFE (protect).
    Established: eviction-protection ruling (operator: protect an in-flight reply)."""
    monkeypatch.setattr(A.gen_gate, "in_flight", lambda mk: 1)
    assert A._actively_replying("m", set()) is True
    monkeypatch.setattr(A.gen_gate, "in_flight", lambda mk: 0)
    assert A._actively_replying("m", {"m"}) is True
    assert A._actively_replying("m", {"other"}) is False

    def _boom(mk):
        raise RuntimeError("gate unreadable")
    monkeypatch.setattr(A.gen_gate, "in_flight", _boom)
    assert A._actively_replying("m", set()) is True


def test_leaked_busy_flag_does_not_protect_forever(monkeypatch):
    """INVARIANT: a slot whose ``busy`` flag is stale (last_used older than
    HUGPY_SLOT_BUSY_STALE_S, default 900s) is a LEAKED inflight counter, not a
    live reply — it is NOT reported busy, so an idle resident can still be
    evicted to fit. A fresh busy slot IS reported.
    Established: k67 item B (leaked inflight pinned residents as 'actively
    replying' forever)."""
    slots_mod = importlib.import_module("hugpy_engine.serve.slots")
    now = time.time()

    class _Pool:
        def statuses(self):
            return [{"model_key": "fresh", "busy": True, "last_used": now - 5},
                    {"model_key": "leaked", "busy": True, "last_used": now - 10 ** 6},
                    {"model_key": "idle", "busy": False, "last_used": now}]
    monkeypatch.setattr(slots_mod, "SlotPool", _Pool)
    assert A._busy_slot_models() == {"fresh"}


# ---------------------------------------------------------------------------
# ctx_pct is an allocation contract for the NEXT load/reseat
# ---------------------------------------------------------------------------
def test_ctx_pct_setting_is_clamped_and_flex_floor_wins(monkeypatch):
    """INVARIANT: the worker's per-model ctx_pct (1..100, percent of max
    context) is read from the runtime settings map and clamped; unset → None
    (today's default ctx, byte-identical); a committed flex ctx floor for the
    model wins over the setting so the served -c matches the KV the admission
    reserved. Established: t21 ctx band; frontier handoff 2026-09-28
    ('a context edit is an allocation contract for the next load/reseat')."""
    monkeypatch.setattr(A, "_RUNTIME_SETTINGS", {"ctx_pct": {"m": 40, "big": 500, "low": 0}})
    monkeypatch.setattr(A, "_FLEX_CTX_FLOOR", {})
    assert A._ctx_pct("m") == 40
    assert A._ctx_pct("big") == 100
    assert A._ctx_pct("low") == 1
    assert A._ctx_pct("unset") is None
    monkeypatch.setattr(A, "_FLEX_CTX_FLOOR", {"m": 25})
    assert A._ctx_pct("m") == 25


def test_ctx_band_and_kv_scale_are_linear_in_percent():
    """INVARIANT: ctx_band_bounds(target, deviation) is ± points clamped to
    [1, 100]; kv_at_ctx_pct scales a KV figure linearly (half ctx = half KV).
    No target → None (ctx band is opt-in). Established: t21 (flex.py)."""
    assert flex.ctx_band_bounds(None, 10) is None
    assert flex.ctx_band_bounds(40, 10) == (30, 50)
    assert flex.ctx_band_bounds(40, 0) == (40, 40)
    assert flex.ctx_band_bounds(5, 10) == (1, 15)
    assert flex.ctx_band_bounds(98, 10) == (88, 100)
    assert flex.kv_at_ctx_pct(1000, 100, 50) == 500
    assert flex.kv_at_ctx_pct(1000, 0, 50) == 1000      # degenerate → unchanged


# ---------------------------------------------------------------------------
# Eviction telemetry survives store errors (h23, 2026-09-28)
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_telemetry():
    ev.reset_for_tests()
    ev.set_store(None)
    yield
    ev.reset_for_tests()
    ev.set_store(None)


def test_eviction_event_emit_survives_a_failing_store(clean_telemetry, tmp_path, monkeypatch):
    """INVARIANT: emit_eviction_event is total — with the durable store sink
    installed and the store's connection FAILING, every emit still returns the
    event (ring + log + sinks are independently guarded), append() reports 0
    rows, and after MAX_FAILURES the store quarantines itself (health ok=False,
    retry_in>0) instead of taxing every eviction with a doomed write.
    Established: h23 (2026-09-28: 'model eviction telemetry/store resilience';
    verified live on Computron and AEB)."""
    store = ev.EvictionStore(path=str(tmp_path / "evictions.db"))
    ev.set_store(store)
    ev.install_store_sink()
    good = ev.emit_eviction_event("evict.done", model_key="m1", tier="vram")
    assert good is not None and store.max_id() >= 1
    assert store.health()["ok"] is True

    def _broken():
        raise RuntimeError("disk I/O error")
    monkeypatch.setattr(store, "_connect", _broken)
    for _ in range(ev.MAX_FAILURES + 2):
        assert ev.emit_eviction_event("evict.done", model_key="m1") is not None
    assert store.append([ev.build_event("evict.done", model_key="m1")]) == 0
    h = store.health()
    assert h["ok"] is False and h["retry_in"] > 0 and "disk I/O error" in h["last_error"]
    # The in-process ring still carried every event for the console.
    assert len(ev.recent(limit=50)) >= ev.MAX_FAILURES + 2


def test_eviction_store_unwritable_path_is_not_fatal(clean_telemetry):
    """INVARIANT: an unwritable store path degrades to 'no history' — append is
    0, emit still succeeds. Established: h23 (2026-09-28)."""
    store = ev.EvictionStore(path="/proc/definitely/not/writable/x.db")
    ev.set_store(store)
    ev.install_store_sink()
    assert ev.emit_eviction_event("makeroom.verdict", model_key="m1", action="refuse") is not None
    assert store.append([ev.build_event("evict.fail", model_key="m1")]) == 0


def test_serve_path_telemetry_wrappers_never_raise(clean_telemetry):
    """INVARIANT: the serve-path convenience emitters (emit_load_fail etc.) are
    wrapped by _never_raises — a broken sink or a bad field never reaches the
    load/evict caller. Established: h23 (2026-09-28)."""
    def boom(_ev):
        raise RuntimeError("sink down")
    ev.register_sink(boom)

    class Weird:
        def __repr__(self):
            raise RuntimeError("even repr is broken")
    assert ev.emit_load_fail("m1", engine="llama", error=Weird()) is None or True
    assert ev.emit_eviction_event("evict.done", model_key="m1", junk=Weird()) is not None
