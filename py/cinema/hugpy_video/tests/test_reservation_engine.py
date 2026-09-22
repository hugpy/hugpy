"""p6 reservation ENGINE — acquire/hold/release, make-room orchestration through
the EXISTING evict verbs (mocked), comfy-flush-first ordering, protection-respect
(force=false → the worker's gate keeps protected residents), refusal-on-timeout,
non-reservable + fail-open paths, and admission-respect accounting.

No GPU, no network: the fleet read (_list_workers) and the /ops/evict relay
(_evict) are stubbed with a mutable fake worker so the orchestration is asserted
behaviorally.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_reservation_engine.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Isolate the registry singleton + force estimates + make the bounded wait fast —
# ALL read live per-call, so setting them before import is enough.
_D = tempfile.mkdtemp(prefix="hugpy-resv-eng-")
os.environ["HUGPY_RESERVATIONS_DB"] = os.path.join(_D, "resv.db")
os.environ["HUGPY_RESERVATIONS_MEASURED"] = os.path.join(_D, "nope.json")
os.environ["HUGPY_RESERVATION_MAKEROOM_TIMEOUT_S"] = "2"
os.environ["HUGPY_RESERVATION_POLL_S"] = "0.05"
os.environ["HUGPY_RESERVATION_SETTLE_S"] = "0"
os.environ.pop("HUGPY_STUDIO_WORKER", None)
os.environ.pop("IDENTITY_RENDER_URL", None)

import pytest  # noqa: E402

from hugpy_video.intel.reservation import engine as E
from hugpy_video.intel.reservation.registry import reservation_registry as REG

# Robustly repoint the process-wide singleton at our scratch DB REGARDLESS of
# import order — the env var alone is fragile because a sibling test file may
# import (and thus construct) the singleton before this module's env-set runs,
# which would bind it to the real service DB. Repointing path + resetting the
# one-time init flag guarantees isolation.
REG.path = os.environ["HUGPY_RESERVATIONS_DB"]
REG._initialized = False

_GIB = 1024 ** 3


def _worker(free_gib, residents):
    """residents: [{model_key, host_mode, vram_bytes}] (host_mode comfy|subprocess|...)."""
    return {
        "id": "ae", "name": "ae", "status": "online", "url": "http://ae:9100",
        "gpus": [{"memory_total": 24 * _GIB, "memory_free": int(free_gib * _GIB)}],
        "pid_registry": {"models": [dict(alive=True, **r) for r in residents]},
    }


def _install(monkeypatch, worker, protected=()):
    """Stub the fleet read + the evict relay against a mutable fake worker. The
    fake _evict mutates the worker (removes the model, credits its VRAM back) —
    unless the model is in ``protected`` and force is False, which mirrors the
    WORKER-side gate refusing a static / actively-replying / queued-ahead resident."""
    calls = []

    def _fake_list():
        return [worker]

    def _fake_evict(w, mk, force=False):
        calls.append(mk)
        models = w["pid_registry"]["models"]
        row = next((m for m in models if m.get("model_key") == mk), None)
        if row is None:
            return {"evicted": False, "reason": "not resident"}
        if mk in protected and not force:
            return {"evicted": False, "reason": f"eviction gated: {mk} (protected)"}
        models.remove(row)
        w["gpus"][0]["memory_free"] += int(row.get("vram_bytes") or 0)
        return {"evicted": True, "vram_freed": row.get("vram_bytes"),
                "vram_free_after": w["gpus"][0]["memory_free"]}

    monkeypatch.setattr(E, "_list_workers", _fake_list)
    monkeypatch.setattr(E, "_evict", _fake_evict)
    return calls


def _cleanup(run_id):
    E.release(run_id)


def test_already_fits_holds_without_eviction(monkeypatch):
    w = _worker(free_gib=22, residents=[])
    calls = _install(monkeypatch, w)
    handle = E.acquire("studio_i2v", object(), "run-fits")
    assert handle is not None
    assert calls == []                                   # nothing evicted
    active = REG.active("ae")
    assert [r["run_id"] for r in active] == ["run-fits"]
    assert active[0]["peak_bytes"] == 20 * _GIB
    _cleanup("run-fits")
    assert REG.reserved_bytes("ae") == 0


def test_make_room_evicts_the_brain_squatter(monkeypatch):
    # The doc's collision class: 6 GB free, the 18 GB agent-brain squatting the card.
    w = _worker(free_gib=6, residents=[
        {"model_key": "Qwen~Qwen3-Coder-Next-GGUF", "host_mode": "subprocess",
         "vram_bytes": 18 * _GIB},
        {"model_key": "sd-turbo", "host_mode": "subprocess", "vram_bytes": 3 * _GIB},
    ])
    calls = _install(monkeypatch, w)
    handle = E.acquire("studio_i2v", object(), "run-brain")
    assert handle is not None
    # Largest-first: evict the brain (18 G) → 6+18=24 ≥ 20 → stop (sd-turbo untouched).
    assert calls == ["Qwen~Qwen3-Coder-Next-GGUF"]
    row = REG.get("run-brain")
    assert row["state"] == "active"
    assert bool(row["made_room"]) is True
    _cleanup("run-brain")


def test_comfy_flush_happens_first(monkeypatch):
    # A resident comfy checkpoint + the brain; peak needs both cleared.
    w = _worker(free_gib=1, residents=[
        {"model_key": "comfy-dreamshaper-8", "host_mode": "comfy", "vram_bytes": 5 * _GIB},
        {"model_key": "Qwen~Qwen3-Coder-Next-GGUF", "host_mode": "subprocess",
         "vram_bytes": 18 * _GIB},
    ])
    calls = _install(monkeypatch, w)
    handle = E.acquire("studio_i2v", object(), "run-comfy")
    assert handle is not None
    # Comfy flushed BEFORE the eviction engine touches the LLM residents.
    assert calls[0] == "comfy-dreamshaper-8"
    assert "Qwen~Qwen3-Coder-Next-GGUF" in calls
    _cleanup("run-comfy")


def test_default_best_effort_proceeds_on_envelope_shortfall(monkeypatch):
    # DEFAULT (refuse OFF): make-room can't reach the whole-GPU envelope because the
    # only resident is protected — but the render AUTOFITS/offloads, so the engine
    # PROCEEDS (holds the claim) rather than blocking a render that would succeed.
    w = _worker(free_gib=6, residents=[
        {"model_key": "Qwen~Qwen3-Coder-Next-GGUF", "host_mode": "subprocess",
         "vram_bytes": 18 * _GIB},
    ])
    _install(monkeypatch, w, protected={"Qwen~Qwen3-Coder-Next-GGUF"})
    handle = E.acquire("studio_i2v", object(), "run-besteffort")
    assert handle is not None                        # proceeds, does NOT raise
    assert REG.get("run-besteffort")["state"] == "active"   # claim held for the run
    # The protected resident was never removed.
    assert any(m["model_key"] == "Qwen~Qwen3-Coder-Next-GGUF"
               for m in w["pid_registry"]["models"])
    _cleanup("run-besteffort")


def test_refuses_honestly_when_shortfall_is_all_protected(monkeypatch):
    # OPT-IN (HUGPY_RESERVATION_REFUSE=on): the brain is PROTECTED (static / actively
    # replying) — force=false is refused. Nothing else to yield → the reservation
    # refuses honestly, never deadlocks, never force-evicts a protected resident.
    monkeypatch.setenv("HUGPY_RESERVATION_REFUSE", "on")
    w = _worker(free_gib=6, residents=[
        {"model_key": "Qwen~Qwen3-Coder-Next-GGUF", "host_mode": "subprocess",
         "vram_bytes": 18 * _GIB},
    ])
    _install(monkeypatch, w, protected={"Qwen~Qwen3-Coder-Next-GGUF"})
    with pytest.raises(E.ReservationRefused) as ei:
        E.acquire("studio_i2v", object(), "run-refuse")
    reason = ei.value.reason
    assert reason["peak_bytes"] == 20 * _GIB
    assert reason["short_by_bytes"] > 0
    assert any(r["model_key"] == "Qwen~Qwen3-Coder-Next-GGUF"
               for r in reason["remaining_residents"])
    # The protected resident was NEVER actually removed.
    assert any(m["model_key"] == "Qwen~Qwen3-Coder-Next-GGUF"
               for m in w["pid_registry"]["models"])
    # And the claim was released (no phantom reserved bytes left behind).
    assert REG.reserved_bytes("ae") == 0
    assert REG.get("run-refuse")["state"] == "released"


def test_refuses_when_peak_exceeds_all_evictable_headroom(monkeypatch):
    # OPT-IN refusal: everything evictable is cleared but the card still can't reach peak.
    monkeypatch.setenv("HUGPY_RESERVATION_REFUSE", "on")
    w = _worker(free_gib=6, residents=[
        {"model_key": "small-a", "host_mode": "subprocess", "vram_bytes": 2 * _GIB},
    ])
    calls = _install(monkeypatch, w)
    with pytest.raises(E.ReservationRefused) as ei:
        E.acquire("studio_i2v", object(), "run-toobig")
    assert calls == ["small-a"]                  # tried the one evictable resident
    assert ei.value.reason["free_bytes"] == 8 * _GIB   # 6 + 2, still < 20
    assert REG.reserved_bytes("ae") == 0


def test_non_reservable_task_makes_no_claim(monkeypatch):
    w = _worker(free_gib=6, residents=[])
    _install(monkeypatch, w)
    assert E.acquire("generate_image", object(), "run-light") is None
    assert REG.get("run-light") is None
    assert REG.reserved_bytes("ae") == 0


def test_unresolvable_fleet_fails_open(monkeypatch):
    monkeypatch.setattr(E, "_list_workers", lambda: [])   # can't see the fleet
    # Fail OPEN: proceed unreserved rather than block a render on a transient read.
    assert E.acquire("studio_i2v", object(), "run-blind") is None
    assert REG.get("run-blind") is None


def test_disabled_switch_is_inert(monkeypatch):
    monkeypatch.setenv("HUGPY_RESERVATIONS", "off")
    w = _worker(free_gib=6, residents=[])
    _install(monkeypatch, w)
    assert E.acquire("studio_i2v", object(), "run-off") is None
    assert REG.get("run-off") is None


def test_acquire_then_release_is_the_admission_respect_lifecycle(monkeypatch):
    w = _worker(free_gib=22, residents=[])
    _install(monkeypatch, w)
    E.acquire("generate_studio_movie", object(), "run-life")
    # While held, the card's reserved bytes are visible to admission-respect.
    assert REG.reserved_bytes("ae") == 20 * _GIB
    assert any(r["run_id"] == "run-life" for r in REG.listing())
    # Release on the terminal path clears it (abort/done/failed all route here).
    E.release("run-life")
    assert REG.reserved_bytes("ae") == 0
    assert REG.get("run-life")["state"] == "released"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
