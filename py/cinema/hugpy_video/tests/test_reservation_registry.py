"""p6 reservation REGISTRY — claim/release/refresh, lease expiry, double-claim,
admission-respect math, and the read-only listing.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_reservation_registry.py -q
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_video.intel.reservation.registry import ReservationRegistry

_GIB = 1024 ** 3


def _reg(ttl=100.0):
    d = tempfile.mkdtemp(prefix="hugpy-resv-reg-")
    return ReservationRegistry(path=os.path.join(d, "resv.db"), lease_ttl_s=ttl)


def test_claim_release_roundtrip():
    reg = _reg()
    assert reg.claim("run1", "ae", "ae", "studio_i2v", 20 * _GIB) is True
    active = reg.active("ae")
    assert [r["run_id"] for r in active] == ["run1"]
    assert active[0]["state"] == "active"
    assert active[0]["peak_bytes"] == 20 * _GIB
    assert reg.release("run1", reason="done") is True
    assert reg.active("ae") == []
    assert reg.reserved_bytes("ae") == 0


def test_reserved_bytes_is_admission_respect_input():
    reg = _reg()
    reg.claim("r1", "ae", "ae", "studio_i2v", 20 * _GIB)
    reg.claim("r2", "ae", "ae", "generate_movie", 7 * _GIB)
    # Two active claims on one card sum — the bytes another placement must NOT see.
    assert reg.reserved_bytes("ae") == 27 * _GIB
    # A different worker is unaffected.
    assert reg.reserved_bytes("computron") == 0
    # The subtraction a placement performs: effective free = max(0, phys - reserved).
    phys_free = 24 * _GIB
    assert max(0, phys_free - reg.reserved_bytes("ae")) == 0   # card fully reserved


def test_double_claim_same_run_is_upsert_not_duplicate():
    reg = _reg()
    reg.claim("run1", "ae", "ae", "studio_i2v", 20 * _GIB)
    # Re-claiming the SAME run_id refreshes/updates, never double-counts.
    reg.claim("run1", "ae", "ae", "studio_i2v", 18 * _GIB)
    active = reg.active("ae")
    assert len(active) == 1
    assert active[0]["peak_bytes"] == 18 * _GIB
    assert reg.reserved_bytes("ae") == 18 * _GIB


def test_lease_expiry_self_heals_orphaned_claim():
    reg = _reg(ttl=0.3)
    reg.claim("orphan", "ae", "ae", "generate_studio_movie", 20 * _GIB)
    assert reg.reserved_bytes("ae") == 20 * _GIB
    time.sleep(0.4)   # no refresh -> lease lapses
    # The read-side sweep flips it to expired; it no longer counts as reserved.
    assert reg.active("ae") == []
    assert reg.reserved_bytes("ae") == 0
    row = reg.get("orphan")
    assert row["state"] == "expired"
    assert "lease" in (row["reason"] or "").lower()


def test_refresh_advances_lease_anchor_and_noops_when_terminal():
    # Deterministic (no wall-clock race): a generous TTL, and we assert refresh
    # MOVES the lease anchor forward (that is what keeps a live run's claim from
    # self-expiring), and that refreshing a terminal claim is a clean no-op.
    reg = _reg(ttl=100.0)
    reg.claim("live", "ae", "ae", "studio_i2v", 20 * _GIB)
    hb0 = reg.get("live")["heartbeat_at"]
    time.sleep(0.05)
    assert reg.refresh("live") is True
    hb1 = reg.get("live")["heartbeat_at"]
    assert hb1 > hb0                       # anchor advanced -> lease renewed
    assert reg.reserved_bytes("ae") == 20 * _GIB
    reg.release("live")
    assert reg.refresh("live") is False    # terminal -> no-op


def test_release_is_idempotent():
    reg = _reg()
    reg.claim("run1", "ae", "ae", "studio_i2v", 20 * _GIB)
    assert reg.release("run1") is True
    assert reg.release("run1") is False    # already terminal -> no-op
    assert reg.release("never-existed") is False


def test_listing_shape_active_and_terminal():
    reg = _reg()
    reg.claim("a", "ae", "ae", "studio_i2v", 20 * _GIB)
    reg.note_make_room("a", ["Qwen~Qwen3-Coder-Next-GGUF"])
    reg.claim("b", "ae", "ae", "generate_movie", 7 * _GIB)
    reg.release("b", reason="done")
    # Default: active only.
    active = reg.listing()
    ids = {r["run_id"] for r in active}
    assert ids == {"a"}
    ra = next(r for r in active if r["run_id"] == "a")
    assert ra["made_room"] is True
    assert ra["evicted"] == ["Qwen~Qwen3-Coder-Next-GGUF"]
    assert ra["lease_expires_in_s"] is not None and ra["lease_expires_in_s"] > 0
    # include_terminal surfaces the released row too, active-first.
    allrows = reg.listing(include_terminal=True)
    assert {r["run_id"] for r in allrows} == {"a", "b"}
    assert allrows[0]["run_id"] == "a"     # active sorts ahead of terminal
    assert next(r for r in allrows if r["run_id"] == "b")["lease_expires_in_s"] is None


def test_store_failure_fails_open_not_closed():
    # A registry pointed at an unwritable path must degrade to "no reservation"
    # (reserved_bytes 0, empty listing) — never raise into placement/dispatch.
    reg = ReservationRegistry(path="/proc/nonexistent/cannot/reservations.db")
    assert reg.claim("x", "ae", "ae", "studio_i2v", 20 * _GIB) is False
    assert reg.reserved_bytes("ae") == 0
    assert reg.active("ae") == []
    assert reg.listing() == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
