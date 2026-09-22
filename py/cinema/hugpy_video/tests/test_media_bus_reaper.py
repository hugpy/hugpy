"""k63 follow-up — the media-bus ORPHAN SWEEP (_reap_orphans).

The incident (2026-08-04): a gunicorn restart dropped the runner threads that were
supervising two in-flight studio_i2v renders. The rows kept their in-flight status,
a user Cancel flipped them to 'cancelling' — a flag only a LIVE runner honors — and
they sat "canceling" in the console for 134+ minutes. 19 zombie rows in total had to
be hand-terminalized. These tests pin the sweep that terminalizes them automatically:

  * the PID gate (provably-dead runner -> immediate reap, both statuses);
  * an ALIVE pid + recent movement is NEVER touched (the safety direction);
  * the movement gate's short 'cancelling' window vs the long claimed/running one;
  * the compare-and-swap: a row that moves under the sweep is skipped, not overwritten;
  * the throttle (one sweep per interval across the pool) and its env knob.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_media_bus_reaper.py -q
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from hugpy_video.intel import media_bus as MB


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _dead_pid() -> int:
    """A PID that is provably NOT alive on this host. Fabricated high and verified
    with the same signal-0 probe the reaper uses, so the test can never pass by
    accidentally naming a live process."""
    for cand in range(2 ** 22 - 1, 2 ** 22 - 400, -2):
        try:
            os.kill(cand, 0)
        except ProcessLookupError:
            return cand
        except Exception:  # noqa: BLE001 — exists (or unprobeable) -> try the next
            continue
    pytest.skip("no provably-dead pid available on this host")


@pytest.fixture
def bus(tmp_path, monkeypatch):
    """A private media-bus DB + a reset throttle clock, so a sweep in one test can
    never gate a sweep in the next."""
    monkeypatch.setattr(MB, "DB_PATH", str(tmp_path / "media_jobs.db"))
    monkeypatch.setattr(MB, "_initialized", False)
    monkeypatch.setattr(MB, "_last_reap_ts", 0.0)
    # Deterministic gates unless a test overrides them.
    monkeypatch.delenv("HUGPY_MEDIA_BUS_CANCEL_REAP_SECONDS", raising=False)
    monkeypatch.delenv("HUGPY_MEDIA_BUS_STALE_SECONDS", raising=False)
    monkeypatch.delenv("HUGPY_MEDIA_BUS_REAP_INTERVAL_SECONDS", raising=False)
    MB._ensure_db()
    yield


def _insert(job_id, status, claim_token, *, name="studio_i2v", age_s=0.0,
            progress=None, stage_log=None):
    """Insert one in-flight row directly (no runner, no spec deserialization).
    ``age_s`` backdates BOTH the movement clock sources (updated + the stage log)."""
    now = time.time() - age_s
    conn = MB._connect()
    try:
        conn.execute(
            "INSERT INTO media_jobs (job_id, name, status, spec_json, result_json, "
            "claim_token, created, updated, progress_json, stage_log_json) "
            "VALUES (?,?,?,?,NULL,?,?,?,?,?)",
            (job_id, name, status, "{}", claim_token, now - 5.0, now,
             json.dumps(progress if progress is not None
                        else {"stage": "generating", "done": 3, "total": 48}),
             json.dumps(stage_log) if stage_log is not None else json.dumps(
                 [{"stage": "generating", "ts": now, "ts_last": now, "count": 3}])),
        )
    finally:
        conn.close()
    return job_id


def _row(job_id):
    conn = MB._connect()
    try:
        r = conn.execute(
            "SELECT status, result_json, claim_token, progress_json, stage_log_json "
            "FROM media_jobs WHERE job_id=?", (job_id,)).fetchone()
    finally:
        conn.close()
    return {"status": r[0],
            "result": json.loads(r[1]) if r[1] else None,
            "claim_token": r[2],
            "progress_json": r[3],
            "stage_log": MB._load_stage_log(r[4])}


# --------------------------------------------------------------------------- #
# 1 — the restart case: 'cancelling' owned by a dead PID
# --------------------------------------------------------------------------- #
def test_cancelling_with_dead_pid_is_reaped_to_cancelled(bus):
    pid = _dead_pid()
    jid = _insert("j_cancel_dead", "cancelling", f"daemon-{pid}-r0-abc123")

    assert MB._reap_orphans() == 1

    row = _row(jid)
    assert row["status"] == "cancelled"
    assert row["result"]["ok"] is False
    assert row["result"]["error"]["code"] == "cancelled"
    assert row["result"]["error"]["retryable"] is False
    # claim_token NULLed is LOAD-BEARING: a late-finishing thread's terminal write is
    # gated `AND claim_token=?` and must MISS, so it cannot overwrite this reap.
    assert row["claim_token"] is None
    assert row["progress_json"] is None
    # the terminal timeline entry is retained and says WHY
    last = row["stage_log"][-1]
    assert last["stage"] == "cancelled"
    assert last["code"] == "cancelled"
    assert last["reap_gate"] == "dead_pid"
    assert last["prior_status"] == "cancelling"


# --------------------------------------------------------------------------- #
# 2 — a 'running' row whose runner process is gone
# --------------------------------------------------------------------------- #
def test_running_with_dead_pid_is_reaped_to_failed_runner_lost(bus):
    pid = _dead_pid()
    jid = _insert("j_run_dead", "running", f"worker-{pid}-deadbeef")

    assert MB._reap_orphans() == 1

    row = _row(jid)
    assert row["status"] == "failed"
    err = row["result"]["error"]
    assert err["code"] == "runner_lost"
    assert err["retryable"] is True
    assert "re-submit" in err["message"]
    assert row["claim_token"] is None
    assert row["progress_json"] is None


def test_worker_token_shape_also_parses_and_claimed_state_is_swept(bus):
    pid = _dead_pid()
    _insert("j_claimed_dead", "claimed", f"daemon-{pid}-r3-ffee00")
    assert MB._reap_orphans() == 1
    assert _row("j_claimed_dead")["status"] == "failed"


# --------------------------------------------------------------------------- #
# 3 — a LIVE runner is never touched (the safety direction)
# --------------------------------------------------------------------------- #
def test_live_pid_with_recent_movement_is_untouched(bus):
    jid = _insert("j_live", "running", f"daemon-{os.getpid()}-r0-cafe11")

    assert MB._reap_orphans() == 0

    row = _row(jid)
    assert row["status"] == "running"
    assert row["claim_token"] == f"daemon-{os.getpid()}-r0-cafe11"
    assert row["progress_json"] is not None
    assert row["result"] is None


def test_unparseable_token_never_uses_the_pid_gate(bus):
    """A token that is not one of our two shapes carries NO liveness evidence — it
    may only ever be reaped by the movement gate, never by the PID gate."""
    _insert("j_odd", "running", "some-external-token")
    assert MB._reap_orphans() == 0
    assert _row("j_odd")["status"] == "running"


# --------------------------------------------------------------------------- #
# 4 — the movement fallback (alive process, wedged job) and its two-tier window
# --------------------------------------------------------------------------- #
def test_cancelling_alive_pid_but_stale_movement_is_reaped(bus, monkeypatch):
    monkeypatch.setenv("HUGPY_MEDIA_BUS_CANCEL_REAP_SECONDS", "30")
    jid = _insert("j_cancel_wedged", "cancelling",
                  f"daemon-{os.getpid()}-r1-aa11bb", age_s=600.0)

    assert MB._reap_orphans() == 1

    row = _row(jid)
    assert row["status"] == "cancelled"
    assert row["result"]["error"]["code"] == "cancelled"
    assert row["stage_log"][-1]["reap_gate"] == "no_movement"
    assert "no movement for" in row["result"]["error"]["message"]


def test_two_tier_gate_running_survives_the_cancel_window(bus, monkeypatch):
    """The same 600s of silence that condemns a 'cancelling' row must NOT condemn a
    'running' one: a job legitimately HELD for GPU capacity idles its movement clock
    for hours while perfectly alive (hence the long stale window for claimed/running)."""
    monkeypatch.setenv("HUGPY_MEDIA_BUS_CANCEL_REAP_SECONDS", "30")
    _insert("j_cancel_old", "cancelling", f"daemon-{os.getpid()}-r1-aa11bb",
            age_s=600.0)
    _insert("j_run_old", "running", f"daemon-{os.getpid()}-r2-cc22dd", age_s=600.0)

    assert MB._reap_orphans() == 1
    assert _row("j_cancel_old")["status"] == "cancelled"
    assert _row("j_run_old")["status"] == "running"

    # ... until the long window is crossed too.
    monkeypatch.setenv("HUGPY_MEDIA_BUS_STALE_SECONDS", "300")
    assert MB._reap_orphans() == 1
    assert _row("j_run_old")["status"] == "failed"


def test_movement_clock_reads_the_stage_log_not_just_updated(bus, monkeypatch):
    """Same clock the stale filter uses: a fresh timeline entry keeps a row alive even
    when the row's own `updated` column is ancient."""
    monkeypatch.setenv("HUGPY_MEDIA_BUS_CANCEL_REAP_SECONDS", "30")
    now = time.time()
    _insert("j_fresh_log", "cancelling", f"daemon-{os.getpid()}-r0-eeff00",
            age_s=600.0,
            stage_log=[{"stage": "generating", "ts": now - 600, "ts_last": now,
                        "count": 9}])
    assert MB._reap_orphans() == 0
    assert _row("j_fresh_log")["status"] == "cancelling"


# --------------------------------------------------------------------------- #
# 5 — CAS safety: a row that moved under the sweep is skipped, never overwritten
# --------------------------------------------------------------------------- #
def test_cas_miss_skips_the_row_entirely(bus):
    """Simulates the SELECT->UPDATE race: the row reaches a real terminal (its actual
    runner got there first) before the reaper's write. The CAS on the OBSERVED status
    must match 0 rows -> no status overwrite, no result overwrite, no stage-log
    append, no bridge."""
    jid = _insert("j_raced", "running", "daemon-1-r0-zz")
    conn = MB._connect()
    try:
        conn.execute("UPDATE media_jobs SET status='done', result_json=? "
                     "WHERE job_id=?", (json.dumps({"ok": True}), jid))
        before = _row(jid)
        # observed status is the pre-race 'running' — exactly what the sweep held
        won = MB._reap_one(conn, jid, "studio_i2v", "running", "dead_pid",
                           "pid 1 is not alive", before["stage_log"])
    finally:
        conn.close()

    assert won is False
    after = _row(jid)
    assert after["status"] == "done"
    assert after["result"] == {"ok": True}
    assert after["stage_log"] == before["stage_log"]   # no terminal entry appended


def test_full_sweep_does_not_touch_queued_or_terminal_rows(bus):
    pid = _dead_pid()
    _insert("j_queued", "queued", None)
    _insert("j_done", "done", f"daemon-{pid}-r0-1111")
    assert MB._reap_orphans() == 0
    assert _row("j_queued")["status"] == "queued"
    assert _row("j_done")["status"] == "done"


# --------------------------------------------------------------------------- #
# 6 — the throttle
# --------------------------------------------------------------------------- #
def test_throttled_wrapper_sweeps_once_per_interval(bus, monkeypatch):
    monkeypatch.setenv("HUGPY_MEDIA_BUS_REAP_INTERVAL_SECONDS", "3600")
    pid = _dead_pid()
    _insert("j_first", "cancelling", f"daemon-{pid}-r0-aaaa")

    MB._maybe_reap_orphans()
    assert _row("j_first")["status"] == "cancelled"
    won_ts = MB._last_reap_ts
    assert won_ts > 0

    # A second immediate call is a NO-OP: inside the interval, nothing is scanned.
    _insert("j_second", "cancelling", f"daemon-{pid}-r0-bbbb")
    MB._maybe_reap_orphans()
    assert _row("j_second")["status"] == "cancelling"
    assert MB._last_reap_ts == won_ts

    # Once the interval lapses (clock rewound here rather than slept), it sweeps again.
    MB._last_reap_ts = time.time() - 7200.0
    MB._maybe_reap_orphans()
    assert _row("j_second")["status"] == "cancelled"


def test_reap_interval_env_is_honored(bus, monkeypatch):
    assert MB._reap_interval_seconds() == 60.0
    monkeypatch.setenv("HUGPY_MEDIA_BUS_REAP_INTERVAL_SECONDS", "5")
    assert MB._reap_interval_seconds() == 5.0
    for bad in ("", "   ", "nope", "0", "-3"):
        monkeypatch.setenv("HUGPY_MEDIA_BUS_REAP_INTERVAL_SECONDS", bad)
        assert MB._reap_interval_seconds() == 60.0


def test_cancel_reap_seconds_env_is_honored(bus, monkeypatch):
    assert MB._cancel_reap_seconds() == 1800.0
    monkeypatch.setenv("HUGPY_MEDIA_BUS_CANCEL_REAP_SECONDS", "45")
    assert MB._cancel_reap_seconds() == 45.0
    for bad in ("", "junk", "0", "-1"):
        monkeypatch.setenv("HUGPY_MEDIA_BUS_CANCEL_REAP_SECONDS", bad)
        assert MB._cancel_reap_seconds() == 1800.0


def test_sweep_never_raises_out_of_the_wrapper(bus, monkeypatch):
    """The janitor must never be able to kill a runner thread."""
    def _boom():
        raise RuntimeError("db exploded")
    monkeypatch.setattr(MB, "_reap_orphans", _boom)
    monkeypatch.setattr(MB, "_last_reap_ts", 0.0)
    MB._maybe_reap_orphans()   # no raise


# --------------------------------------------------------------------------- #
# pid-probe unit checks (the gate's bias is absolute: never call an alive pid dead)
# --------------------------------------------------------------------------- #
def test_token_pid_parsing():
    assert MB._token_pid("daemon-4242-r0-abc123") == 4242
    assert MB._token_pid("worker-77-deadbeef") == 77
    assert MB._token_pid("daemon-notapid-r0-x") is None
    assert MB._token_pid("something-4242-r0") is None
    assert MB._token_pid(None) is None
    assert MB._token_pid("") is None


def test_pid_alive_probe():
    assert MB._pid_alive(os.getpid()) is True
    assert MB._pid_alive(_dead_pid()) is False
    # pid 1 exists in every namespace; unprivileged signal-0 to it raises
    # PermissionError in some sandboxes — either way it must read ALIVE.
    assert MB._pid_alive(1) is True
