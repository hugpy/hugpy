"""p27 (2026-07-23) — the GPU orphan reaper verb + comfy display honesty.

THE k30 GAP: an orphaned llama-server child — slot claim cleared, child kept
VRAM — became ENUMERABLE with c34199e (reconcile tags it cuda_context with
model_key None) but stayed UNEVICTABLE: every eviction verb keys on model_key
and an orphan has none. _reap_gpu_orphans (POST /ops/reap-orphans) kills by
PID, so its four admission gates are FAIL-CLOSED: any gate unverifiable means
NOT reapable.

Gates, each individually broken here:
  1. OWN-VENV  — name/cmdline must contain this worker's venv marker; comfy
                 and foreign processes fail by construction; the agent's own
                 pid and its direct infra pids are excluded outright.
  2. NO CLAIM  — no current slot status references the pid as child_pid;
                 unreadable slot statuses -> nothing reapable.
  3. HOLDS GPU — pid in the nvidia-smi snapshot with mib > 0.
  4. MIN AGE   — older than HUGPY_ORPHAN_MIN_AGE_S (default 300s), closing the
                 mid-spawn race (child exists before its claim registers).

Also covered: dry_run DEFAULTS TRUE on the route; SIGTERM->wait->SIGKILL kill
discipline (mocked os.kill); and the p27 companion — comfy emitted as ONE
process-class row (display_label "comfy", is_process_row True, model_key as a
call-time LABEL only).

Run: venv/bin/python -m pytest tests/test_reap_gpu_orphans.py -q
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import pid_registry as PR

MIB = 1 << 20
GIB = 1 << 30

VENV = "/opt/hugpy/venv"                  # the worker's own venv marker
OLD = 10_000.0                            # comfortably past the 300s grace


class _State:
    pass


@pytest.fixture
def rig(monkeypatch):
    """A worker with a fake nvidia-smi snapshot, fake /proc, fake slot pool and
    a recorded (not executed) kill. Mutate the dicts per test, then call
    ``reap(dry_run=...)``."""
    gpu = {}          # pid -> {"name", "mib"}
    procs = {}        # pid -> {"starttime", "cmdline", "name"} (None = gone)
    ages = {}         # pid -> seconds (missing = unmeasurable)
    slots = {"rows": []}          # rows=None -> statuses unreadable
    kills = []        # (pid, signum) in order

    monkeypatch.setattr(A, "_gpu_process_vram", lambda: dict(gpu))
    monkeypatch.setattr(A, "_slot_statuses",
                        lambda: (None if slots["rows"] is None
                                 else list(slots["rows"])))
    monkeypatch.setattr(A, "_self_venv_marker", lambda: VENV)
    monkeypatch.setattr(A, "_reap_own_pids", lambda: {A.os.getpid()})
    monkeypatch.setattr(A, "_proc_age_s", lambda pid: ages.get(pid))
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)

    def _fake_proc_info(pid):
        return procs.get(pid)
    # _reap_gpu_orphans imports _default_proc_info FROM pid_registry — patch it
    # there (the import site), same seam the registry's own tests use.
    monkeypatch.setattr(PR, "_default_proc_info", _fake_proc_info)

    def _fake_kill(pid, signum):
        kills.append((pid, signum))
        info = procs.get(pid)
        if info is None:
            raise ProcessLookupError(pid)
        import signal
        if signum == signal.SIGTERM and not info.get("_term_immune"):
            procs[pid] = None           # exits promptly on SIGTERM
        elif signum == signal.SIGKILL:
            procs[pid] = None
    monkeypatch.setattr(A.os, "kill", _fake_kill)
    monkeypatch.setattr(time, "sleep", lambda s: None)  # no real waiting
    monkeypatch.setattr(A, "_ORPHAN_TERM_WAIT_S", 0.05)  # short SIGTERM window

    def _add_orphan(pid, mib=18000, name=f"{VENV}/bin/python", age=OLD,
                    starttime=111, term_immune=False):
        gpu[pid] = {"name": name, "mib": mib}
        procs[pid] = {"starttime": starttime, "cmdline": name,
                      "name": "python", "_term_immune": term_immune}
        if age is not None:
            ages[pid] = age

    def _reap(dry_run):
        return A._reap_gpu_orphans(_State(), dry_run=dry_run)

    return type("Rig", (), {
        "gpu": gpu, "procs": procs, "ages": ages, "slots": slots,
        "kills": kills, "add_orphan": staticmethod(_add_orphan),
        "reap": staticmethod(_reap)})()


def _row(out, pid):
    return next(r for r in out["results"] if r["pid"] == pid)


# ── happy path: all four gates hold -> reaped (SIGTERM honored) ─────────────
def test_happy_path_orphan_is_reaped(rig):
    rig.add_orphan(4242)
    out = rig.reap(dry_run=False)
    row = _row(out, 4242)
    assert row["action"] == "reaped"
    assert row["vram_bytes"] == 18000 * MIB
    assert out["reaped_count"] == 1
    import signal
    assert rig.kills == [(4242, signal.SIGTERM)]     # SIGTERM sufficed


def test_sigterm_survivor_gets_sigkill(rig):
    rig.add_orphan(4242, term_immune=True)
    out = rig.reap(dry_run=False)
    row = _row(out, 4242)
    assert row["action"] == "reaped"
    assert "SIGKILL" in row["reason"]
    import signal
    assert rig.kills == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]


# ── dry_run: reports, never signals ─────────────────────────────────────────
def test_dry_run_reports_without_killing(rig):
    rig.add_orphan(4242)
    out = rig.reap(dry_run=True)
    row = _row(out, 4242)
    assert row["action"] == "skipped"
    assert "would be reaped" in row["reason"]
    assert out["reapable_vram_bytes"] == 18000 * MIB
    assert rig.kills == []                            # nothing signalled
    assert rig.procs[4242] is not None                # still alive


# ── gate 1: OWN-VENV ────────────────────────────────────────────────────────
def test_foreign_process_never_reapable(rig):
    rig.add_orphan(555, name="/usr/bin/some-other-daemon")
    rig.procs[555]["cmdline"] = "/usr/bin/some-other-daemon --serve"
    out = rig.reap(dry_run=False)
    row = _row(out, 555)
    assert row["action"] == "skipped"
    assert "foreign" in row["reason"]
    assert rig.kills == []


def test_comfy_pid_never_reapable(rig):
    # Comfy's process_name is its OWN venv python — even if it contained our
    # marker, the comfy name check fires FIRST (fails own-venv by construction).
    rig.add_orphan(777, name="/opt/ComfyUI/venv/bin/python")
    out = rig.reap(dry_run=False)
    row = _row(out, 777)
    assert row["action"] == "skipped"
    assert "comfy" in row["reason"].lower()
    assert rig.kills == []


def test_agent_own_pid_never_reapable(rig):
    me = A.os.getpid()
    rig.add_orphan(me)
    out = rig.reap(dry_run=False)
    row = _row(out, me)
    assert row["action"] == "skipped"
    assert "own pid" in row["reason"]
    assert rig.kills == []


def test_no_venv_marker_fails_closed(rig, monkeypatch):
    monkeypatch.setattr(A, "_self_venv_marker", lambda: None)
    rig.add_orphan(4242)
    out = rig.reap(dry_run=False)
    row = _row(out, 4242)
    assert row["action"] == "skipped"
    assert "fail-closed" in row["reason"]
    assert rig.kills == []


# ── gate 2: NO LIVE CLAIM ───────────────────────────────────────────────────
def test_live_slot_claim_protects_pid(rig):
    rig.add_orphan(4242)
    rig.slots["rows"].append({"slot_id": "1", "model_key": "Fable-Distill",
                              "child_pid": 4242, "busy": False})
    out = rig.reap(dry_run=False)
    row = _row(out, 4242)
    assert row["action"] == "skipped"
    assert "live slot claims" in row["reason"]
    assert rig.kills == []


def test_unreadable_slot_statuses_fail_closed(rig):
    rig.add_orphan(4242)
    rig.slots["rows"] = None                          # pool unreadable
    out = rig.reap(dry_run=False)
    row = _row(out, 4242)
    assert row["action"] == "skipped"
    assert "fail-closed" in row["reason"]
    assert rig.kills == []


# ── gate 3: HOLDS GPU ───────────────────────────────────────────────────────
def test_zero_vram_pid_is_not_reapable(rig):
    rig.add_orphan(4242, mib=0)
    out = rig.reap(dry_run=False)
    row = _row(out, 4242)
    assert row["action"] == "skipped"
    assert "no VRAM" in row["reason"]
    assert rig.kills == []


# ── gate 4: MIN AGE (mid-spawn race) ────────────────────────────────────────
def test_young_process_protected_by_min_age(rig):
    rig.add_orphan(4242, age=5.0)                     # newborn slot child
    out = rig.reap(dry_run=False)
    row = _row(out, 4242)
    assert row["action"] == "skipped"
    assert "too young" in row["reason"]
    assert rig.kills == []


def test_unmeasurable_age_fails_closed(rig):
    rig.add_orphan(4242, age=None)                    # ages dict has no entry
    out = rig.reap(dry_run=False)
    row = _row(out, 4242)
    assert row["action"] == "skipped"
    assert "unmeasurable" in row["reason"]
    assert rig.kills == []


# ── recycled-pid guard at kill time ─────────────────────────────────────────
def test_recycled_pid_before_sigterm_is_not_signalled(rig):
    rig.add_orphan(4242, starttime=111)
    # Between the scan snapshot and the signal, the pid was recycled: same
    # number, different starttime. Identity re-check must refuse to signal.
    orig_info = rig.procs[4242]
    calls = {"n": 0}

    def _flip(pid):
        calls["n"] += 1
        if pid == 4242 and calls["n"] > 1:            # scan sees ours; kill-time sees stranger
            return {"starttime": 999, "cmdline": "stranger", "name": "x"}
        return orig_info if pid == 4242 else None
    import hugpy_fleet.worker.pid_registry as _pr
    _saved = _pr._default_proc_info
    _pr._default_proc_info = _flip
    try:
        out = rig.reap(dry_run=False)
    finally:
        _pr._default_proc_info = _saved
    row = _row(out, 4242)
    assert row["action"] == "skipped"
    assert "identity changed" in row["reason"]
    assert rig.kills == []


# ── the route contract: POST /ops/reap-orphans, dry_run defaults TRUE ───────
def _client(monkeypatch):
    state = A.WorkerState(name="t", url=None, worker_id="w-reap")
    return A.build_app(state).test_client()


def test_route_dry_run_defaults_true_when_absent(rig, monkeypatch):
    rig.add_orphan(4242)
    c = _client(monkeypatch)
    r = c.post("/ops/reap-orphans", json={})
    data = r.get_json()
    assert r.status_code == 200
    assert data["ok"] is True
    assert data["dry_run"] is True                    # ABSENT field -> preview
    assert rig.kills == []                            # nothing killed
    assert data["reapable_vram_bytes"] == 18000 * MIB


def test_route_explicit_dry_run_false_reaps(rig, monkeypatch):
    rig.add_orphan(4242)
    c = _client(monkeypatch)
    r = c.post("/ops/reap-orphans", json={"dry_run": False})
    data = r.get_json()
    assert data["ok"] is True and data["dry_run"] is False
    assert data["reaped_count"] == 1
    import signal
    assert (4242, signal.SIGTERM) in rig.kills


def test_route_never_500s(rig, monkeypatch):
    monkeypatch.setattr(A, "_reap_gpu_orphans",
                        lambda state, dry_run=True: (_ for _ in ()).throw(
                            RuntimeError("boom")))
    c = _client(monkeypatch)
    r = c.post("/ops/reap-orphans", json={"dry_run": True})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is False and "boom" in data["error"]


# ═══════════ p27 companion: comfy display honesty (ONE process-class row) ═══
def test_comfy_idle_row_is_a_labelled_process_row():
    reg = PR.PidRegistry(proc_info=lambda pid: None)
    out = reg.reconcile(
        gpu_procs={888: {"name": "/opt/ComfyUI/venv/bin/python", "mib": 9000}},
        inprocess_bytes={}, comfy_bytes=None)
    rows = [r for r in out["foreign"] if r["host_mode"] == "comfy"]
    assert len(rows) == 1                             # ONE row, never per-model
    row = rows[0]
    assert row["display_label"] == "comfy"
    assert row["is_process_row"] is True
    assert row["vram_bytes"] == 9000 * MIB            # measured process VRAM
    assert row["model_key"] is None                   # idle: no fake model row
    assert "idle" in (row["label"] or "")
    assert out["unattributed"] == []                  # recognized, not a squatter


def test_comfy_active_call_model_key_is_label_only():
    reg = PR.PidRegistry(proc_info=lambda pid: None)
    reg.record_foreign_call("comfy", "sdxl-base", job_id="j1")
    out = reg.reconcile(
        gpu_procs={888: {"name": "/opt/ComfyUI/venv/bin/python", "mib": 9000}},
        inprocess_bytes={}, comfy_bytes=None)
    rows = [r for r in out["foreign"] if r["host_mode"] == "comfy"]
    assert len(rows) == 1                             # still ONE process row
    row = rows[0]
    assert row["display_label"] == "comfy"            # class label unchanged
    assert row["is_process_row"] is True              # NOT a per-model resident
    assert row["model_key"] == "sdxl-base"            # the call-time LABEL
    assert row["job_id"] == "j1"


def test_comfy_row_rides_the_heartbeat_snapshot():
    reg = PR.PidRegistry(proc_info=lambda pid: None)
    reg.reconcile(
        gpu_procs={888: {"name": "/opt/ComfyUI/venv/bin/python", "mib": 9000}},
        inprocess_bytes={}, comfy_bytes=None)
    snap = reg.snapshot_for_heartbeat()
    comfy = [r for r in snap["models"] if r.get("host_mode") == "comfy"]
    assert len(comfy) == 1
    assert comfy[0]["display_label"] == "comfy"
    assert comfy[0]["is_process_row"] is True
