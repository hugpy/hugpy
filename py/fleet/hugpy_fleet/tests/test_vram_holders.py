"""VRAM-HOLDER METER (incident 2026-08-05).

A studio render fork leaked 8GB, held it at 0% GPU util for minutes, central was
blind to it, and it wedged the card (new slot loads aborted SIGABRT). The
operator's principle: "if it's not doing work it's evictable; if it's NOT
evictable and NOT doing work, that must be EXPLICIT." This exercises the
VISIBILITY + EXPLICIT-FLAG half — the read-only ``_vram_holders`` meter and its
GET /ops/vram-holders route, plus central's ``_vram_squatters`` aggregate.

NOTHING here kills. ``_vram_holders`` MIRRORS the reaper's four gates (own-venv,
no live slot claim, holds VRAM, past min-age) using the SAME shared primitives
to LABEL the idle own-venv squatter — it never calls _reap_gpu_orphans and never
signals a pid.

Asserted (the operator's four cases):
  * a NON-serving own-venv holder past min-age  -> reapable=True, squatter=True
  * a SERVING slot child                        -> kind=slot, never a squatter
  * a comfy process                             -> kind=comfy, never reapable
  * a foreign process                           -> kind=foreign, never reapable

Run: venv/bin/python -m pytest tests/test_vram_holders.py -q
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import pid_registry as PR

MIB = 1 << 20
VENV = "/opt/hugpy/venv"                   # the worker's own venv marker
OLD = 10_000.0                             # comfortably past the 300s grace
YOUNG = 5.0                                # inside the mid-spawn grace


class _State:
    pass


@pytest.fixture
def rig(monkeypatch):
    """A fake card: an nvidia-smi per-process snapshot, per-card mem/util, a
    fake /proc, a fake slot pool. Mutate the dicts, then call ``holders()``."""
    gpu = {}          # pid -> {"name", "mib"}
    procs = {}        # pid -> {"cmdline", ...}
    ages = {}         # pid -> seconds (missing = unmeasurable)
    slots = {"rows": []}
    cards = [{"index": 0, "mem_total_bytes": 24000 * MIB,
              "mem_used_bytes": 20000 * MIB, "mem_free_bytes": 4000 * MIB,
              "util_pct": 0}]               # 0% util — the incident's fingerprint

    monkeypatch.setattr(A, "_gpu_process_vram", lambda: dict(gpu))
    monkeypatch.setattr(A, "_slot_statuses",
                        lambda: (None if slots["rows"] is None
                                 else list(slots["rows"])))
    monkeypatch.setattr(A, "_gpu_card_stats", lambda: [dict(c) for c in cards])
    monkeypatch.setattr(A, "_self_venv_marker", lambda: VENV)
    monkeypatch.setattr(A, "_reap_own_pids", lambda: {A.os.getpid()})
    monkeypatch.setattr(A, "_proc_age_s", lambda pid: ages.get(pid))
    monkeypatch.setattr(PR, "_default_proc_info", lambda pid: procs.get(pid))

    def _add(pid, mib=8000, name=f"{VENV}/bin/python", age=OLD, cmdline=None):
        gpu[pid] = {"name": name, "mib": mib}
        procs[pid] = {"cmdline": cmdline if cmdline is not None else name}
        if age is not None:
            ages[pid] = age

    def _add_slot(pid, model_key="Fable-Distill", mib=18000, busy=False,
                  last_used=None):
        gpu[pid] = {"name": f"{VENV}/bin/llama-server", "mib": mib}
        slots["rows"].append({"slot_id": "1", "model_key": model_key,
                              "child_pid": pid, "busy": busy,
                              "last_used": last_used})

    def _holders():
        return A._vram_holders(_State())

    return type("Rig", (), {
        "gpu": gpu, "procs": procs, "ages": ages, "slots": slots, "cards": cards,
        "add": staticmethod(_add), "add_slot": staticmethod(_add_slot),
        "holders": staticmethod(_holders)})()


def _h(out, pid):
    return next(h for h in out["holders"] if h["pid"] == pid)


# ── the squatter: non-serving own-venv holder past min-age ──────────────────
def test_idle_own_venv_holder_is_the_squatter(rig):
    rig.add(4242, mib=8000)                 # the leaked 8GB studio fork
    out = rig.holders()
    row = _h(out, 4242)
    assert row["kind"] == "own-orphan"
    assert row["reapable"] is True
    assert row["squatter"] is True
    assert row["work_state"] == "idle-squatter"
    assert row["vram_bytes"] == 8000 * MIB
    assert "not auto-reaped" in row["reason"].lower()
    # It rides the top-level squatters list — the EXPLICIT surfacing.
    assert [s["pid"] for s in out["squatters"]] == [4242]


# ── a serving slot is doing work: never a squatter ──────────────────────────
def test_serving_slot_is_not_a_squatter(rig):
    rig.add_slot(5001, model_key="Qwen3", busy=True)
    out = rig.holders()
    row = _h(out, 5001)
    assert row["kind"] == "slot"
    assert row["serving"] is True
    assert row["model_key"] == "Qwen3"
    assert row["reapable"] is False
    assert row["squatter"] is False
    assert out["squatters"] == []


# ── an IDLE slot holds VRAM but has a LIVE claim: reapable/squatter False ────
def test_idle_slot_holds_but_is_claimed(rig):
    rig.add_slot(5002, busy=False, last_used=None)   # seated, never served
    out = rig.holders()
    row = _h(out, 5002)
    assert row["kind"] == "slot"
    assert row["serving"] is False
    assert row["work_state"] == "idle"
    # Gate 2 (a live slot claims this pid) means it is NOT an orphan.
    assert row["reapable"] is False
    assert row["squatter"] is False


# ── comfy: external adopted service, never reapable ─────────────────────────
def test_comfy_process_never_reapable(rig):
    rig.add(777, name="/opt/ComfyUI/venv/bin/python (comfyui)")
    out = rig.holders()
    row = _h(out, 777)
    assert row["kind"] == "comfy"
    assert row["reapable"] is False
    assert row["squatter"] is False


# ── foreign: not our venv, out of scope ─────────────────────────────────────
def test_foreign_process_never_reapable(rig):
    rig.add(555, name="/usr/bin/some-daemon", cmdline="/usr/bin/some-daemon -x")
    out = rig.holders()
    row = _h(out, 555)
    assert row["kind"] == "foreign"
    assert row["reapable"] is False
    assert row["squatter"] is False


# ── the agent's own pid / infra: never reapable ─────────────────────────────
def test_agent_own_pid_is_infra(rig):
    me = A.os.getpid()
    rig.add(me)
    out = rig.holders()
    row = _h(out, me)
    assert row["kind"] == "infra"
    assert row["reapable"] is False
    assert row["squatter"] is False


# ── gate 4: a YOUNG own-venv holder is protected (mid-spawn race) ────────────
def test_young_own_venv_holder_not_yet_squatter(rig):
    rig.add(4243, age=YOUNG)
    out = rig.holders()
    row = _h(out, 4243)
    assert row["kind"] == "own-orphan"
    assert row["reapable"] is False        # inside the grace window
    assert row["squatter"] is False
    assert "too young" in row["reason"]


# ── gate 4 fail-closed: unmeasurable age is NOT reapable ─────────────────────
def test_unmeasurable_age_fails_closed(rig):
    rig.add(4244, age=None)
    out = rig.holders()
    row = _h(out, 4244)
    assert row["kind"] == "own-orphan"
    assert row["reapable"] is False
    assert "unmeasurable" in row["reason"]


# ── gate 1 fail-closed: no venv marker -> own-venv can't be proven -> foreign ─
def test_no_venv_marker_reads_as_foreign(rig, monkeypatch):
    monkeypatch.setattr(A, "_self_venv_marker", lambda: None)
    rig.add(4245)
    out = rig.holders()
    row = _h(out, 4245)
    assert row["kind"] == "foreign"        # ownership unprovable -> not ours
    assert row["reapable"] is False
    assert row["squatter"] is False


# ── the per-card meter + coarse work signal ─────────────────────────────────
def test_card_totals_and_coarse_util(rig):
    rig.cards[0]["util_pct"] = 0           # 0% — card doing NO work
    rig.add(4242)
    out = rig.holders()
    assert out["vram_total_bytes"] == 24000 * MIB
    assert out["vram_used_bytes"] == 20000 * MIB
    assert out["vram_free_bytes"] == 4000 * MIB
    assert out["gpu_util_pct"] == 0        # coarse whole-card signal
    assert "WHOLE-CARD" in out["gpu_util_source"]
    assert out["min_age_s"] == A._ORPHAN_MIN_AGE_S


def test_gpu_util_is_max_across_cards(rig):
    rig.cards.append({"index": 1, "mem_total_bytes": 24000 * MIB,
                      "mem_used_bytes": 1000 * MIB, "mem_free_bytes": 23000 * MIB,
                      "util_pct": 73})
    out = rig.holders()
    assert out["gpu_util_pct"] == 73       # coarse: MAX across the box
    assert out["vram_total_bytes"] == 48000 * MIB


# ── no GPU: empty holders, degrades exactly like a CPU box ───────────────────
def test_no_gpu_degrades_cleanly(rig, monkeypatch):
    monkeypatch.setattr(A, "_gpu_process_vram", lambda: {})
    monkeypatch.setattr(A, "_gpu_card_stats", lambda: [])
    out = rig.holders()
    assert out["ok"] is True
    assert out["holders"] == []
    assert out["squatters"] == []
    assert out["gpu_util_pct"] is None
    assert out["vram_total_bytes"] is None


# ── the beat wrapper never raises (a missed beat drops the worker) ──────────
def test_beat_wrapper_swallows_errors(monkeypatch):
    monkeypatch.setattr(A, "_vram_holders",
                        lambda state: (_ for _ in ()).throw(RuntimeError("boom")))
    assert A._vram_holders_beat(_State()) is None


# ── the route contract: GET /ops/vram-holders, read-only ────────────────────
def test_route_returns_meter(rig, monkeypatch):
    rig.add(4242)
    state = A.WorkerState(name="t", url=None, worker_id="w-vram")
    c = A.build_app(state).test_client()
    r = c.get("/ops/vram-holders")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert any(h["squatter"] for h in data["holders"])


def test_route_never_500s(monkeypatch):
    monkeypatch.setattr(A, "_vram_holders",
                        lambda state: (_ for _ in ()).throw(RuntimeError("boom")))
    state = A.WorkerState(name="t", url=None, worker_id="w-vram2")
    c = A.build_app(state).test_client()
    r = c.get("/ops/vram-holders")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is False and "boom" in data["error"]
