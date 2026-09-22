"""k117 — every job finishes or fails; the clock never counts forever.

The operator watched the Active-Processes feed show jobs at 875+ minutes in
"loading", 25h in "awaiting_capacity", and DONE rows whose timers kept ticking
under a stale stage label ("archiving"). Keeper triage found the bus rows were
CORRECT — a "done" row reading 875m had really run 9.4s — and that the plane was
missing three things. These tests pin all three:

  * TERMINAL SEMANTICS: a terminal row's duration is FROZEN (no function of
    `now`), split into queue-wait vs run, and labelled with the TRUE terminal
    stage rather than the last in-flight one. Rows that predate the sidecar
    (~1950 of them live) derive the same numbers from created/updated/stage_log.
  * STALL WATCHDOG: a per-stage progress deadline. Progress RESETS the deadline;
    no-progress past it FAILS the job typed, carrying the stage, the elapsed and
    the worker's own last reported error; a healthy job is never touched; a
    worker that finishes late has its result RECORDED, not dropped.
  * RE-QUOTE ON STALE ADMISSION: a job that starts long after it was quoted
    re-checks VRAM feasibility and never blind-runs.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_job_lifecycle.py -q
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from hugpy_video.intel import media_bus as MB
from hugpy_video.intel import job_lifecycle as JL


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def bus(tmp_path, monkeypatch):
    """A private bus DB + a reset watchdog clock, so a sweep in one test can never
    gate a sweep in the next (mirrors test_media_bus_reaper's fixture)."""
    monkeypatch.setattr(MB, "DB_PATH", str(tmp_path / "media_jobs.db"))
    monkeypatch.setattr(MB, "_initialized", False)
    monkeypatch.setattr(MB, "_last_reap_ts", 0.0)
    monkeypatch.setattr(JL, "_initialized", False)
    monkeypatch.setattr(JL, "_last_sweep_ts", 0.0)
    monkeypatch.setattr(JL, "_armed_logged", False)
    for var in ("HUGPY_JOB_WATCHDOG", "HUGPY_JOB_WATCHDOG_INTERVAL_S",
                "HUGPY_JOB_REQUOTE_AFTER_S", "HUGPY_JOB_REQUOTE_MAX",
                "HUGPY_JOB_LOADING_HISTORY_MULT",
                "HUGPY_JOB_LOADING_HISTORY_MIN_SAMPLES",
                "HUGPY_MEDIA_BUS_STALE_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    for stage in JL._DEADLINE_DEFAULTS:
        monkeypatch.delenv(f"HUGPY_JOB_DEADLINE_{stage.upper()}_S", raising=False)
    MB._ensure_db()
    JL._ensure()
    yield


def _insert(job_id, status, *, name="studio_i2v", claim_token=None,
            created_ago=60.0, updated_ago=0.0, progress=None, stage_log=None):
    """Insert one row directly (no runner, no spec deserialization)."""
    now = time.time()
    conn = MB._connect()
    try:
        conn.execute(
            "INSERT INTO media_jobs (job_id, name, status, spec_json, result_json, "
            "claim_token, created, updated, progress_json, stage_log_json) "
            "VALUES (?,?,?,?,NULL,?,?,?,?,?)",
            (job_id, name, status, "{}", claim_token, now - created_ago,
             now - updated_ago,
             json.dumps(progress) if progress is not None else None,
             json.dumps(stage_log) if stage_log is not None else None),
        )
    finally:
        conn.close()
    return job_id


def _stage(stage, ago, *, last_ago=None, count=1, **extra):
    now = time.time()
    e = {"stage": stage, "ts": now - ago,
         "ts_last": now - (ago if last_ago is None else last_ago), "count": count}
    e.update(extra)
    return e


def _row(job_id):
    conn = MB._connect()
    try:
        r = conn.execute(
            "SELECT status, result_json, claim_token, progress_json, stage_log_json "
            "FROM media_jobs WHERE job_id=?", (job_id,)).fetchone()
    finally:
        conn.close()
    return {"status": r[0], "result": json.loads(r[1]) if r[1] else None,
            "claim_token": r[2], "progress_json": r[3],
            "stage_log": MB._load_stage_log(r[4])}


def _listed(job_id, **kw):
    for row in MB.list_jobs(include_terminal=True, include_stale=True, **kw):
        if row["job_id"] == job_id:
            return row
    raise AssertionError(f"{job_id} not in listing")


# =========================================================================== #
# 1 — TERMINAL SEMANTICS: the clock stops
# =========================================================================== #
def test_done_row_duration_is_frozen_and_not_a_function_of_now(bus):
    """THE BUG, verbatim: a generate_scene that ran 9.4s, created 946 minutes ago,
    rendered as "946m · archiving". The row was always right; the wire shape had
    no frozen duration to render. Now it does — and it does not move."""
    ago = 946 * 60.0
    _insert("j_done", "done", name="generate_scene", created_ago=ago,
            updated_ago=ago - 9.4,
            stage_log=[_stage("loading", ago - 0.2),
                       _stage("generating", ago - 9.3),
                       _stage("archiving", ago - 9.3),
                       _stage("done", ago - 9.4, detail="completed",
                              at_stage="archiving")])

    first = _listed("j_done")
    time.sleep(0.05)
    second = _listed("j_done")

    assert first["terminal"] is True
    assert first["run_s"] == second["run_s"], "a finished job's clock must stop"
    assert first["terminal_at"] == second["terminal_at"]
    # 9.4s of work, not 946 minutes.
    assert 9.0 <= first["run_s"] <= 9.5
    assert first["total_s"] < 10.0
    # The TRUE terminal stage — not the last in-flight one.
    assert first["terminal_stage"] == "done"
    assert first["at_stage"] == "archiving"
    # A terminal row has no in-stage clock at all; there is no stage to be in.
    assert first["elapsed_in_stage_s"] is None


def test_terminal_queue_and_run_are_split_not_conflated(bus):
    """`now - created` conflated a queue wait with a runtime — the single number
    that made a 9s render read as 875 minutes. They are separate facts now."""
    _insert("j_split", "done", created_ago=3600.0, updated_ago=3600.0 - 1830.0,
            stage_log=[_stage("awaiting_capacity", 3600.0 - 5),
                       _stage("generating", 3600.0 - 1800.0),
                       _stage("done", 3600.0 - 1830.0)])

    row = _listed("j_split")
    # Waited ~30 min for capacity, then ran ~30s.
    assert 1750 <= row["queue_wait_s"] <= 1850
    assert 25 <= row["run_s"] <= 35
    assert row["queue_wait_s"] + row["run_s"] == pytest.approx(row["total_s"], abs=1.0)


def test_never_started_job_reports_zero_run_not_a_negative_or_a_guess(bus):
    """A job cancelled before it ever ran: the whole span is queue wait and the
    run really did take zero seconds. No fabricated runtime."""
    _insert("j_prestart", "cancelled", created_ago=500.0, updated_ago=100.0,
            stage_log=[_stage("cancelled", 100.0,
                              detail="cancelled before it started")])

    row = _listed("j_prestart")
    assert row["run_s"] == 0.0
    assert 395 <= row["queue_wait_s"] <= 405
    assert row["terminal_stage"] == "cancelled"


def test_row_with_no_recorded_start_says_total_not_a_fabricated_runtime(bus):
    """A pre-k117 row whose runner never wrote a stage entry (studio_tester on the
    live bus) knows its TOTAL span and nothing about the split. It must say so —
    asserting run_s = 0 for a job that plainly ran 17 minutes would be a new lie
    in place of the old one."""
    _insert("j_nosplit", "done", name="studio_tester", created_ago=1100.0,
            updated_ago=1100.0 - 1042.5,
            stage_log=[_stage("done", 1100.0 - 1042.5)])

    row = _listed("j_nosplit")
    assert row["run_s"] is None
    assert row["queue_wait_s"] is None
    assert 1040 <= row["total_s"] <= 1046
    assert row["terminal_stage"] == "done"


def test_running_row_reports_in_stage_elapsed_and_queue_wait_separately(bus):
    _insert("j_live", "running", claim_token="worker-1-abc", created_ago=900.0,
            progress={"stage": "generating", "done": 3, "total": 48},
            stage_log=[_stage("awaiting_capacity", 895.0),
                       _stage("loading", 600.0),
                       _stage("generating", 120.0, last_ago=2.0, count=3)])

    row = _listed("j_live")
    assert row["terminal"] is False
    assert row["terminal_at"] is None and row["terminal_stage"] is None
    assert row["at_stage"] == "generating"
    # In the CURRENT stage for ~2 min — not "running for 15 minutes".
    assert 115 <= row["elapsed_in_stage_s"] <= 125
    # Queue wait ended when the first live stage began.
    assert 290 <= row["queue_wait_s"] <= 310
    assert 590 <= row["run_s"] <= 610
    assert row["last_progress_at"] is not None


def test_terminal_stamp_is_computed_once_and_never_re_times(bus):
    _insert("j_once", "done", created_ago=30.0, updated_ago=0.0)
    JL.stamp_start("j_once")
    JL.stamp_terminal("j_once", "done", at_stage="archiving")
    first = JL.read_side("j_once")
    time.sleep(0.05)
    JL.stamp_terminal("j_once", "failed", at_stage="generating")
    second = JL.read_side("j_once")

    assert first["terminal_at"] == second["terminal_at"]
    assert second["terminal_status"] == "done"
    assert second["at_stage"] == "archiving"


def test_done_terminal_no_longer_mislabels_its_last_stage_as_failed_at_stage(bus):
    """261 live DONE rows carried `"failed_at_stage": "archiving"` — a render that
    finished perfectly, recorded as having failed in archiving. The neutral
    `at_stage` is always written; `failed_at_stage` only when it really failed."""
    from hugpy_video.intel.result_schema import JobError, JobResult

    _insert("j_ok", "done", stage_log=[_stage("archiving", 1.0)])
    MB._record_terminal_stage("j_ok", "done", JobResult(job_id="j_ok", ok=True))
    ok_entry = _row("j_ok")["stage_log"][-1]
    assert ok_entry["at_stage"] == "archiving"
    assert "failed_at_stage" not in ok_entry

    _insert("j_bad", "failed", stage_log=[_stage("loading", 1.0)])
    MB._record_terminal_stage("j_bad", "failed", JobResult(
        job_id="j_bad", ok=False,
        error=JobError(code="oom", message="CUDA OOM", retryable=True)))
    bad_entry = _row("j_bad")["stage_log"][-1]
    assert bad_entry["at_stage"] == "loading"
    assert bad_entry["failed_at_stage"] == "loading"
    # the failure envelope the panel renders still resolves the stage
    assert MB.build_failure_summary(
        {"ok": False, "error": {"code": "oom", "message": "CUDA OOM"}},
        _row("j_bad")["stage_log"])["stage"] == "loading"


# =========================================================================== #
# 2 — STALL WATCHDOG
# =========================================================================== #
@pytest.fixture
def released(monkeypatch):
    """Capture the reservation releases the watchdog performs — it must release
    through the EXISTING path, and it must never touch a worker process."""
    seen = []
    monkeypatch.setattr(MB, "_release_reservation", lambda jid: seen.append(jid))
    return seen


def test_awaiting_capacity_past_its_deadline_fails_typed(bus, released, monkeypatch):
    """25h in awaiting_capacity. The default deadline is 30 min; past it the job
    FAILS with a typed reason carrying the stage, the elapsed and the worker's own
    last reported error — and its reservation is released."""
    monkeypatch.setattr(
        "hugpy_video.intel.reservation.can_admit",
        lambda name, spec=None, run_id=None: (False, {"reason": "card still full",
                                                      "short_by_bytes": 3 << 30}))
    held = 25 * 3600.0
    _insert("j_held", "queued", name="generate_scene", created_ago=held + 60,
            progress={"phase": "awaiting_capacity", "held_since": time.time() - held,
                      "reason": {"short_by_bytes": 3 << 30},
                      "log_tail": ["worker ae: CUDA out of memory (20.1 GiB in use)"]},
            stage_log=[_stage("awaiting_capacity", held, last_ago=held - 1)])

    assert JL.sweep() == 1

    row = _row("j_held")
    assert row["status"] == "failed"
    err = row["result"]["error"]
    assert err["code"] == "stage_deadline_exceeded"
    assert "awaiting_capacity" in err["message"]
    assert "90000s" in err["message"] or "8999" in err["message"] or \
        str(int(held))[:3] in err["message"]
    # the worker's OWN last reported error is surfaced, not invented
    assert "CUDA out of memory" in err["message"]
    # the re-quote check the operator asked for on this stage specifically
    assert "still infeasible" in err["message"]
    last = row["stage_log"][-1]
    assert last["failed_by"] == "stall_watchdog"
    assert last["stage"] == "awaiting_capacity"
    assert last["worker_last_error"].startswith("worker ae: CUDA")
    assert last["requote"]["admit"] is False
    # claim_token NULLed so a late finisher's gated write MISSES (see late-result)
    assert row["claim_token"] is None and row["progress_json"] is None
    assert released == ["j_held"]


def test_loading_past_its_deadline_fails_typed(bus, released):
    """875 minutes in "loading". No history for this job kind, so the flat 15 min
    fallback applies."""
    _insert("j_load", "running", name="generate_scene", claim_token="worker-1-x",
            created_ago=875 * 60 + 10,
            progress={"stage": "loading", "label": "loading Wan2.1"},
            stage_log=[_stage("loading", 875 * 60, last_ago=1.0)])

    assert JL.sweep() == 1
    row = _row("j_load")
    assert row["status"] == "failed"
    assert row["result"]["error"]["code"] == "stage_deadline_exceeded"
    assert "'loading'" in row["result"]["error"]["message"]
    assert released == ["j_load"]


def test_loading_deadline_uses_this_job_kinds_own_history_times_three(bus):
    """The operator's rule: loading's deadline is the model's historical load time
    x3. A kind that has always taken ~20 min to load gets ~60 min, not 15."""
    for i in range(5):
        _insert(f"h{i}", "done", name="slowload", created_ago=10000.0,
                updated_ago=1000.0,
                stage_log=[_stage("loading", 5000.0),
                           _stage("generating", 5000.0 - 1200.0),
                           _stage("done", 1000.0)])
    assert JL.historical_stage_seconds("slowload", "loading") == pytest.approx(
        1200.0, abs=1.0)
    assert JL.deadline_for_stage("loading", "slowload") == pytest.approx(3600.0,
                                                                        abs=5.0)
    # no history -> the flat fallback, never a fabricated number
    assert JL.deadline_for_stage("loading", "neverseen") == 900.0


def test_healthy_jobs_are_never_touched(bus, released):
    """The reaper's whole safety property. None of these has passed a deadline."""
    _insert("h_render", "running", claim_token="worker-1-a", created_ago=300.0,
            progress={"stage": "generating", "done": 12, "total": 48},
            stage_log=[_stage("generating", 200.0, last_ago=1.0, count=12)])
    _insert("h_load", "running", claim_token="worker-1-b", created_ago=120.0,
            progress={"stage": "loading"},
            stage_log=[_stage("loading", 100.0, last_ago=1.0)])
    _insert("h_hold", "queued", created_ago=400.0,
            progress={"phase": "awaiting_capacity"},
            stage_log=[_stage("awaiting_capacity", 300.0, last_ago=2.0)])
    _insert("h_queued", "queued", created_ago=60.0)
    _insert("h_done", "done", created_ago=99999.0, updated_ago=99000.0)

    assert JL.sweep() == 0
    for jid in ("h_render", "h_load", "h_hold", "h_queued"):
        assert _row(jid)["status"] in ("running", "queued")
    assert _row(jid := "h_done")["status"] == "done"
    assert released == []


def test_unknown_stage_gets_no_invented_deadline(bus):
    """The watchdog never guesses. A stage it does not have a deadline for is
    left entirely to the existing 6h orphan sweep."""
    _insert("j_weird", "running", claim_token="worker-1-w", created_ago=99999.0,
            progress={"stage": "transmogrifying"},
            stage_log=[_stage("transmogrifying", 99000.0, last_ago=98999.0)])
    assert JL.deadline_for_stage("transmogrifying") is None
    assert JL.sweep() == 0
    assert _row("j_weird")["status"] == "running"


def test_progress_resets_the_no_progress_deadline(bus, released):
    """A long render that keeps ADVANCING is healthy no matter how long it takes.
    The signature moves -> the deadline restarts."""
    jid = "j_moving"
    _insert(jid, "running", claim_token="worker-1-m", created_ago=10000.0,
            progress={"stage": "generating", "done": 3, "total": 48},
            stage_log=[_stage("generating", 9000.0, last_ago=1.0)])

    assert JL.sweep() == 0                     # first pass records the signature
    side = JL.read_side(jid)
    wd = json.loads(side["watchdog_json"])
    assert wd["stage"] == "generating"

    # Backdate the recorded stall clock well past the 20 min deadline, then report
    # ADVANCED numbers: the signature changes, so the deadline resets.
    JL._upsert(jid, watchdog_json=json.dumps(
        {"stage": "generating", "sig": wd["sig"], "since": time.time() - 4000}))
    conn = MB._connect()
    try:
        conn.execute("UPDATE media_jobs SET progress_json=? WHERE job_id=?",
                     (json.dumps({"stage": "generating", "done": 9, "total": 48}),
                      jid))
    finally:
        conn.close()

    assert JL.sweep() == 0
    assert _row(jid)["status"] == "running"
    assert json.loads(JL.read_side(jid)["watchdog_json"])["sig"] != wd["sig"]
    assert released == []


def test_frozen_progress_past_the_deadline_fails_typed(bus, released):
    """Same render, same wall-clock — but the frame counter has not moved. THIS is
    what "stalled" means, and it is what gets failed."""
    jid = "j_frozen"
    _insert(jid, "running", claim_token="worker-1-f", created_ago=10000.0,
            progress={"stage": "generating", "done": 3, "total": 48},
            stage_log=[_stage("generating", 9000.0, last_ago=1.0)])

    assert JL.sweep() == 0                     # records the signature
    sig = json.loads(JL.read_side(jid)["watchdog_json"])["sig"]
    JL._upsert(jid, watchdog_json=json.dumps(
        {"stage": "generating", "sig": sig, "since": time.time() - 4000}))

    assert JL.sweep() == 1
    row = _row(jid)
    assert row["status"] == "failed"
    err = row["result"]["error"]
    assert err["code"] == "stage_deadline_exceeded"
    assert err["retryable"] is True
    last = row["stage_log"][-1]
    assert last["basis"] == "no frame/step advance"
    assert last["elapsed_s"] > 1200
    assert last["deadline_s"] == 1200.0
    assert released == [jid]


def test_watchdog_skips_a_row_that_moved_under_it(bus, released):
    """Compare-and-swap, exactly like the orphan sweep: a row a real runner
    terminalized between the scan and the write is left completely alone."""
    jid = "j_race"
    _insert(jid, "running", claim_token="worker-1-r", created_ago=99999.0,
            progress={"stage": "loading"},
            stage_log=[_stage("loading", 99000.0, last_ago=1.0)])
    # the row is 'running' in the scan but 'done' by the time we write
    assert JL._fail_typed(jid, "studio_i2v", "queued", code="stage_deadline_exceeded",
                          message="x", extra={"stage": "loading"}) is False
    assert _row(jid)["status"] == "running"
    assert released == []


def test_watchdog_can_be_disabled(bus, monkeypatch):
    monkeypatch.setenv("HUGPY_JOB_WATCHDOG", "0")
    _insert("j_off", "running", claim_token="worker-1-o", created_ago=99999.0,
            progress={"stage": "loading"},
            stage_log=[_stage("loading", 99000.0, last_ago=1.0)])
    assert JL.sweep() == 0
    assert _row("j_off")["status"] == "running"


def test_per_stage_deadline_is_env_tunable(bus, monkeypatch):
    monkeypatch.setenv("HUGPY_JOB_DEADLINE_ARCHIVING_S", "42")
    assert JL.deadline_for_stage("archiving") == 42.0
    monkeypatch.setenv("HUGPY_JOB_DEADLINE_ARCHIVING_S", "0")
    assert JL.deadline_for_stage("archiving") is None   # 0 = off, for one stage


def test_maybe_sweep_is_throttled_across_the_pool(bus, monkeypatch):
    calls = []
    monkeypatch.setattr(JL, "sweep", lambda: calls.append(1))
    JL.maybe_sweep()
    JL.maybe_sweep()
    JL.maybe_sweep()
    assert len(calls) == 1, "one sweep per interval across the whole runner pool"
    monkeypatch.setenv("HUGPY_JOB_WATCHDOG_INTERVAL_S", "0.01")
    time.sleep(0.02)
    JL.maybe_sweep()
    assert len(calls) == 2


def test_status_reports_the_armed_deadlines(bus):
    st = JL.status()
    assert st["armed"] is True
    assert st["deadlines"]["awaiting_capacity"] == 1800.0
    assert st["deadlines"]["archiving"] == 600.0
    assert "generating" in st["no_progress_stages"]


def test_sweep_leaves_a_durable_proof_of_life(bus):
    """`_log_armed` writes an INFO line, but this deployment runs the video_intel
    loggers above INFO — so "is the watchdog actually armed?" needs an answer that
    survives the log level. The heartbeat row is it."""
    assert JL.heartbeat() is None
    _insert("hb_job", "running", claim_token="worker-1-h", created_ago=60.0,
            progress={"stage": "generating", "done": 1, "total": 10},
            stage_log=[_stage("generating", 30.0, last_ago=1.0)])

    assert JL.sweep() == 0
    hb = JL.heartbeat()
    assert hb["scanned"] == 1 and hb["failed"] == 0
    assert hb["deadlines"]["loading"] == 900.0
    assert time.time() - hb["last_sweep"] < 5
    # the heartbeat key is reserved and can never be mistaken for a job row
    assert JL._HEARTBEAT_KEY not in {r["job_id"] for r in
                                     MB.list_jobs(include_terminal=True,
                                                  include_stale=True)}
    assert JL.status()["last_sweep"]["pid"] == hb["pid"]


# =========================================================================== #
# 3 — LATE RESULT: a worker that finishes after the deadline is not dropped
# =========================================================================== #
def test_late_worker_result_is_recorded_not_dropped(bus, released):
    """The watchdog NULLs claim_token so a late terminal write misses — that is
    what protects the honest terminal. But the worker's artifact is real, so its
    answer is kept as an addendum with the state conflict noted."""
    jid = "j_late"
    _insert(jid, "running", claim_token="worker-1-late", created_ago=99999.0,
            progress={"stage": "loading"},
            stage_log=[_stage("loading", 99000.0, last_ago=1.0)])
    assert JL.sweep() == 1
    frozen = JL.read_side(jid)["terminal_at"]

    JL.record_late_result(jid, "done", json.dumps(
        {"job_id": jid, "ok": True, "artifacts": ["/clips/late.mp4"]}),
        worker_token="worker-1-late")

    side = JL.read_side(jid)
    late = json.loads(side["late_json"])
    assert late["reported_status"] == "done"
    assert late["result"]["artifacts"] == ["/clips/late.mp4"]
    assert "central terminal" in late["note"]
    # the central terminal is untouched: still failed, still frozen at the same ts
    assert _row(jid)["status"] == "failed"
    assert side["terminal_at"] == frozen
    # and the feed carries it without the addendum masquerading as a live stage
    row = _listed(jid)
    assert row["terminal_stage"] == "failed"
    assert row["at_stage"] == "loading"
    assert row["late_result"]["reported_status"] == "done"


# =========================================================================== #
# 4 — RE-QUOTE ON STALE ADMISSION
# =========================================================================== #
def test_stale_admission_requeues_instead_of_blind_running(bus, monkeypatch):
    """CODE_GAPS: "queued 25h, then OOM'd into a different VRAM world than they
    were admitted in". A run that starts long after its quote re-checks first."""
    probes = []

    def _no(name, spec=None, run_id=None):
        probes.append(name)
        return (False, {"reason": "card committed to an active reservation"})

    monkeypatch.setattr("hugpy_video.intel.reservation.can_admit", _no)
    jid = "j_stale"
    _insert(jid, "claimed", name="generate_scene", claim_token="worker-1-s",
            created_ago=25 * 3600.0)

    assert JL.on_run_start(jid, "generate_scene", "worker-1-s") is False
    assert probes == ["generate_scene"]
    row = _row(jid)
    assert row["status"] == "queued", "back to the queue, never blind-run"
    assert row["claim_token"] is None
    quote = json.loads(JL.read_side(jid)["quote_json"])
    assert quote["admit"] is False and quote["requeues"] == 1
    # the hold is VISIBLE, with the fresh quote's reason
    assert json.loads(row["progress_json"])["phase"] == "awaiting_capacity"


def test_stale_admission_fails_typed_after_the_requeue_bound(bus, monkeypatch):
    """A job the fleet can never fit must still reach a terminal — it may not
    bounce between queue and claim for ever."""
    monkeypatch.setattr(
        "hugpy_video.intel.reservation.can_admit",
        lambda name, spec=None, run_id=None: (False, {"reason": "never fits here"}))
    monkeypatch.setenv("HUGPY_JOB_REQUOTE_MAX", "1")
    jid = "j_bounce"
    _insert(jid, "running", name="generate_scene", claim_token="worker-1-b",
            created_ago=25 * 3600.0)

    assert JL.on_run_start(jid, "generate_scene", "worker-1-b") is False
    assert _row(jid)["status"] == "queued"

    conn = MB._connect()
    try:
        conn.execute("UPDATE media_jobs SET status='running', claim_token=? "
                     "WHERE job_id=?", ("worker-1-b", jid))
    finally:
        conn.close()
    assert JL.on_run_start(jid, "generate_scene", "worker-1-b") is False
    row = _row(jid)
    assert row["status"] == "failed"
    err = row["result"]["error"]
    assert err["code"] == "gpu_unavailable"
    assert "re-quote failed" in err["message"]
    assert "never fits here" in err["message"]


def test_fresh_admission_is_not_re_probed(bus, monkeypatch):
    """A job that starts promptly after being quoted runs straight through — the
    re-quote is a staleness check, not a second admission gate on every run."""
    probes = []
    monkeypatch.setattr(
        "hugpy_video.intel.reservation.can_admit",
        lambda name, spec=None, run_id=None: (probes.append(name), (True, None))[1])
    jid = "j_fresh"
    _insert(jid, "running", name="generate_scene", claim_token="worker-1-f",
            created_ago=30.0)

    assert JL.on_run_start(jid, "generate_scene", "worker-1-f") is True
    assert probes == []
    assert _row(jid)["status"] == "running"
    assert isinstance(JL.read_side(jid)["started_at"], float)


def test_requote_that_still_fits_proceeds_and_records_the_fresh_quote(bus,
                                                                     monkeypatch):
    monkeypatch.setattr(
        "hugpy_video.intel.reservation.can_admit",
        lambda name, spec=None, run_id=None: (True, None))
    jid = "j_ok_now"
    _insert(jid, "running", name="generate_scene", claim_token="worker-1-n",
            created_ago=25 * 3600.0)

    assert JL.on_run_start(jid, "generate_scene", "worker-1-n") is True
    assert _row(jid)["status"] == "running"
    quote = json.loads(JL.read_side(jid)["quote_json"])
    assert quote["admit"] is True and quote["revalidated"] is True
    assert quote["prior_quote_age_s"] > 1000


def test_on_run_start_fails_open_when_the_probe_explodes(bus, monkeypatch):
    """An admission-check bug must never be the thing that wedges dispatch — the
    same rule media_bus's own admission seam is built on."""
    def _boom(*a, **kw):
        raise RuntimeError("engine on fire")

    monkeypatch.setattr("hugpy_video.intel.reservation.can_admit",
                        _boom)
    jid = "j_openfail"
    _insert(jid, "running", name="generate_scene", claim_token="worker-1-e",
            created_ago=25 * 3600.0)
    assert JL.on_run_start(jid, "generate_scene", "worker-1-e") is True
    assert _row(jid)["status"] == "running"


# =========================================================================== #
# 5 — projection survives a missing sidecar (every pre-k117 row)
# =========================================================================== #
def test_projection_works_with_no_sidecar_table_at_all(bus, monkeypatch):
    """The sidecar makes the numbers exact; the derivation makes them honest. If
    the sidecar cannot be read at all the feed still freezes terminal clocks."""
    _insert("j_derive", "done", created_ago=600.0, updated_ago=540.0,
            stage_log=[_stage("generating", 599.0), _stage("done", 540.0)])
    monkeypatch.setattr(JL, "read_side_many", lambda ids, conn=None: {})
    row = _listed("j_derive")
    assert row["terminal"] is True
    assert 55 <= row["run_s"] <= 62
    assert row["terminal_stage"] == "done"
