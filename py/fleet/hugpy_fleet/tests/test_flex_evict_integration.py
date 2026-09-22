"""t21 — the flex stage wired INTO the VRAM admission engine
(worker_agent/agent._vram_evict_to_fit): flex-before-evict.

Drives the real engine with the seams stubbed (no GPU), asserting:
  * self-flex: compressing the SUBJECT's own ctx to its band floor makes an
    otherwise-too-big load FIT with NOTHING evicted, and commits the served ctx
    (_FLEX_CTX_FLOOR) so serving matches the KV admission reserved;
  * priority-ordered eviction: when flex can't fit, the LOWEST-priority evictable
    resident yields first;
  * uncontended == target: a load that fits at target proceeds untouched and
    leaves no ctx commitment;
  * a refusal voids any committed ctx flex.

Run: venv/bin/python -m pytest tests/test_flex_evict_integration.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import gen_gate
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30


class _State:
    pass


@pytest.fixture
def rig(monkeypatch):
    """A 24 GiB card with mutable free VRAM + a resident set, plus per-model KV /
    ctx / priority maps so the flex stage has real bands to reason about."""
    card = {"total": 24 * GIB, "free": 0}
    subject_det = {"total": 0, "weights": 0, "kv": 0, "ctx_pct": None,
                   "ctx_max": 32768, "geometry_source": "geometry"}
    residents = {}          # mk -> {vram_bytes, host_mode}
    resident_kv = {}        # mk -> (kv_bytes, {"ctx_pct": ...})
    lru = {}
    static = set()

    # The admission ceiling reads all three (2026-07-27 bounded cushion) — a
    # sibling suite leaking one of them silently re-prices every case here.
    for _c in ("HUGPY_VRAM_CEILING_FRAC", "HUGPY_VRAM_CEILING_CUSHION_GIB",
               "HUGPY_VRAM_RESERVE_GIB"):
        monkeypatch.delenv(_c, raising=False)
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: dict(subject_det))
    monkeypatch.setattr(A, "_kv_need_bytes",
                        lambda mk, cfg=None: resident_kv.get(mk, (0, {"ctx_pct": None})))
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v["vram_bytes"],
                                    "host_mode": v["host_mode"], "alive": True}
                                   for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency",
                        lambda mk: "static" if mk in static else "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(gen_gate, "in_flight", lambda mk: 0)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(D, "last_used_snapshot", lambda: dict(lru))
    monkeypatch.setattr(A, "_queued_ahead_of", lambda mk: set())

    def _fake_evict(state, mk, force=False):
        row = residents.pop(mk, None)
        freed = row["vram_bytes"] if row else 0
        card["free"] += freed
        A._FLEX_CTX_FLOOR.pop(mk, None)
        return {"model_key": mk, "evicted": bool(row),
                "vram_freed": freed if row else None,
                "host_mode": row["host_mode"] if row else "none"}
    monkeypatch.setattr(A, "_evict_model", _fake_evict)

    # fresh, inspectable module state
    monkeypatch.setattr(A, "_FLEX_CTX_FLOOR", {})
    monkeypatch.setattr(A, "_RUNTIME_SETTINGS",
                        {"ctx_deviation_pct": {}, "priority": {}})
    A._VRAM_EVICTIONS.update(count=0, last=None, last_at=0.0)

    return type("Rig", (), {
        "card": card, "subject_det": subject_det, "residents": residents,
        "resident_kv": resident_kv, "lru": lru, "static": static})()


# ── self-flex: compress the subject's OWN ctx to fit, evicting NOTHING ───────
def test_self_ctx_flex_fits_without_evicting(rig):
    # 24 GiB card. Since 2026-07-27 the DEFAULT admission reserve is a bounded
    # compute cushion un-stacked against the external floor already out of the
    # free read (agent._vram_ceiling_reserve_bytes), = 0 here — so the shortfall
    # this test needs has to come from the NEED, not from the reserve.
    rig.card["free"] = 13 * GIB
    # Subject: 8 GiB weights + 6 GiB KV @ ctx 50% -> 14 GiB at target, i.e. 1 GiB
    # MORE than the card has free -> does NOT fit; flex is the only way in.
    rig.subject_det.update(total=14 * GIB, weights=8 * GIB, kv=6 * GIB, ctx_pct=50)
    rig.static.add("bystander")
    rig.residents["bystander"] = {"vram_bytes": 5 * GIB, "host_mode": "in_process"}
    # ctx band ±20 -> floor 30% -> KV 3.6 GiB -> need 11.6 GiB -> fits (3.4 over).
    A._RUNTIME_SETTINGS["ctx_deviation_pct"]["subject"] = 20

    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "proceed"
    assert plan.get("evicted") == []
    assert "flex" in (plan.get("note") or "")
    assert A._FLEX_CTX_FLOOR.get("subject") == 30       # committed compressed ctx
    # nothing was evicted — the protected bystander is untouched
    assert "bystander" in rig.residents


def test_no_ctx_band_no_self_flex_still_evicts_as_before(rig):
    # Without a ctx band the flex stage is a no-op and the LRU evictor runs.
    rig.card["free"] = 1 * GIB
    rig.subject_det.update(total=10 * GIB, weights=10 * GIB, kv=0, ctx_pct=None)
    rig.residents["idle"] = {"vram_bytes": 20 * GIB, "host_mode": "subprocess"}
    rig.lru["idle"] = 100.0
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "evicted"
    assert plan["evicted"] == ["idle"]
    assert A._FLEX_CTX_FLOOR.get("subject") is None


# ── priority-ordered eviction: lowest priority yields first ──────────────────
def test_eviction_prefers_lowest_priority_neighbour(rig):
    rig.card["free"] = 1 * GIB
    rig.subject_det.update(total=10 * GIB, weights=10 * GIB, kv=0, ctx_pct=None)
    # Two idle residents, same LRU age; evicting EITHER frees enough. Priority
    # (not age) must decide: the priority-1 resident yields before priority-5.
    rig.residents["hi_prio"] = {"vram_bytes": 15 * GIB, "host_mode": "subprocess"}
    rig.residents["lo_prio"] = {"vram_bytes": 15 * GIB, "host_mode": "subprocess"}
    rig.lru["hi_prio"] = 100.0
    rig.lru["lo_prio"] = 100.0
    A._RUNTIME_SETTINGS["priority"] = {"hi_prio": 5, "lo_prio": 1}

    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "evicted"
    assert plan["evicted"] == ["lo_prio"]           # lowest priority first
    assert "hi_prio" in rig.residents               # higher priority spared


# ── uncontended == target: fits, no flex, no commitment ─────────────────────
def test_uncontended_fits_at_target_no_flex_commitment(rig):
    rig.card["free"] = 20 * GIB
    rig.subject_det.update(total=5 * GIB, weights=4 * GIB, kv=1 * GIB, ctx_pct=50)
    A._RUNTIME_SETTINGS["ctx_deviation_pct"]["subject"] = 20
    # seed a stale commitment to prove entry clears it when the load fits at target
    A._FLEX_CTX_FLOOR["subject"] = 30
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "proceed"
    assert "flex" not in (plan.get("note") or "")
    assert A._FLEX_CTX_FLOOR.get("subject") is None   # target restored


# ── a refusal voids any committed ctx flex ──────────────────────────────────
def test_refusal_clears_committed_flex(rig):
    rig.card["free"] = 3 * GIB
    # Even fully compressed the subject can't fit, and the only resident is
    # static (protected) -> honest refusal.
    rig.subject_det.update(total=30 * GIB, weights=25 * GIB, kv=5 * GIB, ctx_pct=60)
    A._RUNTIME_SETTINGS["ctx_deviation_pct"]["subject"] = 30
    rig.static.add("locked")
    rig.residents["locked"] = {"vram_bytes": 18 * GIB, "host_mode": "in_process"}

    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse"
    assert A._FLEX_CTX_FLOOR.get("subject") is None   # commitment voided
