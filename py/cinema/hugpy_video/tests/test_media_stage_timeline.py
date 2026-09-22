"""Backend tests for the media-bus STAGE TIMELINE (the exhaustive per-process
telemetry that feeds the Studio "Active Processes" expandable).

Covers (GPU-free, script-style with a __main__ guard like the sibling tests):

  1. set_progress accumulates a DEDUPED stage timeline: same coarse stage across
     many calls (a frame loop) is ONE row with a running count, and a change of
     stage appends a new row — surfaced by media_bus.get() + list_jobs().
  2. The timeline AND a terminal FAILURE summary (stage/code/message/retryable)
     SURVIVE a terminal 'failed' write (which nulls the live progress blob), are
     returned by get() and the /video/jobs projection, and record the exact stage
     the job was in when it broke (failed_at_stage -> failure.stage).
  3. A stalled job exposes a last-movement ts (the honest "time since last
     movement" basis, tied to the current stage).
  4. The idempotent `ALTER TABLE ... ADD COLUMN stage_log_json` migration works on
     an EXISTING (pre-feature) DB and is safe to run repeatedly; a not-yet-migrated
     read is not crashed.

Isolation: media_bus.DB_PATH is repointed to a PRIVATE temp sqlite db (the
_selftest idiom) so the real job bus is never touched.

Run:
  abstract_hugpy_dev/venv/bin/python tests/test_media_stage_timeline.py
"""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _private_bus():
    from hugpy_video.intel import media_bus
    tmpdir = tempfile.mkdtemp(prefix="hugpy_test_stagelog_")
    media_bus.DB_PATH = os.path.join(tmpdir, "media_jobs.db")
    media_bus._initialized = False
    return media_bus, tmpdir


def _scene_spec():
    from hugpy_video.intel.scene_schema import make_generate_scene
    from hugpy_video.intel.gen_schema import text_part
    return make_generate_scene(
        parts=(text_part("a serene landscape"),),
        model_id="sd-turbo", width=256, height=256, steps=2, guidance=0.0,
        n_frames=3, fps=8, assemble=True, seed=1, chain=False,
    )


# --------------------------------------------------------------------------- #
# 1) DEDUP: same stage coalesces (count bumps), a new stage appends a row.
# --------------------------------------------------------------------------- #
def test_timeline_dedup():
    media_bus, _ = _private_bus()
    job_id = media_bus.enqueue("generate_scene", _scene_spec())

    media_bus.set_progress(job_id, {"stage": "loading"})
    # a 3-"frame" render loop, same coarse stage -> ONE row, count == 3
    for i in range(1, 4):
        media_bus.set_progress(
            job_id, {"stage": "generating", "done": i, "total": 3,
                     "label": f"frame {i}/3"})
    media_bus.set_progress(job_id, {"stage": "assembling"})

    view = media_bus.get(job_id)
    log = view["stage_log"]
    stages = [e["stage"] for e in log]
    assert stages == ["loading", "generating", "assembling"], stages
    gen = log[1]
    assert gen["count"] == 3, f"frame loop should coalesce to count=3: {gen}"
    assert gen["ts_last"] >= gen["ts"], gen
    assert "frame 3/3" in (gen.get("detail") or ""), gen
    assert view["current_stage"] == "assembling", view["current_stage"]

    # list_jobs (the /video/jobs projection) carries the same timeline while in-flight.
    rows = media_bus.list_jobs(include_terminal=False)
    mine = [r for r in rows if r["job_id"] == job_id]
    assert mine and mine[0]["stage_log"] == log, mine
    print("[1] PASS  timeline dedups a frame loop to one row + surfaces via get/list")


# --------------------------------------------------------------------------- #
# 2) SURVIVAL: timeline + failure summary persist through a terminal failure.
# --------------------------------------------------------------------------- #
def test_timeline_survives_failure():
    media_bus, _ = _private_bus()
    from hugpy_video.intel.result_schema import JobError, JobResult

    job_id = media_bus.enqueue("generate_scene", _scene_spec())
    media_bus.set_progress(job_id, {"stage": "loading"})
    media_bus.set_progress(job_id, {"stage": "generating", "done": 1, "total": 3})

    # Simulate the terminal-failure write EXACTLY as run_claimed does: null the live
    # progress blob, then record the retained terminal stage entry.
    result = JobResult(
        job_id=job_id, ok=False,
        error=JobError(code="oom", message="CUDA out of memory at frame 2",
                       retryable=True))
    conn = media_bus._connect()
    try:
        conn.execute(
            "UPDATE media_jobs SET status='failed', result_json=?, progress_json=NULL, "
            "updated=? WHERE job_id=?",
            (media_bus.serialize_result(result), 999.0, job_id),
        )
    finally:
        conn.close()
    media_bus._record_terminal_stage(job_id, "failed", result)

    view = media_bus.get(job_id)
    assert view["status"] == "failed", view
    assert view["progress"] is None, "live blob is nulled at terminal (unchanged)"
    # Timeline SURVIVED and carries the terminal entry.
    stages = [e["stage"] for e in view["stage_log"]]
    assert stages == ["loading", "generating", "failed"], stages
    # Terminal FAILURE summary with the exact 'where it's failing' (stage + code).
    fail = view["failure"]
    assert fail is not None, "failure summary must survive to terminal"
    assert fail["code"] == "oom", fail
    assert fail["retryable"] is True, fail
    assert "out of memory" in fail["message"], fail
    assert fail["stage"] == "generating", f"failed_at_stage should be the live stage: {fail}"

    # The /video/jobs projection (include_terminal) carries it too.
    rows = media_bus.list_jobs(include_terminal=True)
    mine = [r for r in rows if r["job_id"] == job_id]
    assert mine, "failed row must appear with include_terminal"
    assert mine[0]["failure"]["code"] == "oom", mine[0]["failure"]
    assert [e["stage"] for e in mine[0]["stage_log"]] == stages, mine[0]["stage_log"]

    # A successful terminal has NO failure summary.
    ok_job = media_bus.enqueue("generate_scene", _scene_spec())
    media_bus.set_progress(ok_job, {"stage": "generating"})
    ok_res = JobResult(job_id=ok_job, ok=True)
    conn = media_bus._connect()
    try:
        conn.execute("UPDATE media_jobs SET status='done', result_json=?, "
                     "progress_json=NULL, updated=? WHERE job_id=?",
                     (media_bus.serialize_result(ok_res), 1.0, ok_job))
    finally:
        conn.close()
    media_bus._record_terminal_stage(ok_job, "done", ok_res)
    ov = media_bus.get(ok_job)
    assert ov["failure"] is None, ov["failure"]
    assert ov["stage_log"][-1]["stage"] == "done", ov["stage_log"]
    print("[2] PASS  timeline + failure summary survive a terminal failure (get + list)")


# --------------------------------------------------------------------------- #
# 3) STALL: last_movement_ts is exposed and reflects the newest timeline movement.
# --------------------------------------------------------------------------- #
def test_stall_last_movement():
    media_bus, _ = _private_bus()
    job_id = media_bus.enqueue("generate_scene", _scene_spec())
    media_bus.set_progress(job_id, {"stage": "generating", "done": 1, "total": 3})

    view = media_bus.get(job_id)
    lm = view["last_movement_ts"]
    assert isinstance(lm, (int, float)), view
    # It equals the current stage row's ts_last (the honest stall basis).
    gen = view["stage_log"][-1]
    assert abs(lm - gen["ts_last"]) < 1e-6, (lm, gen)
    # A later movement advances it.
    import time as _t
    _t.sleep(0.01)
    media_bus.set_progress(job_id, {"stage": "generating", "done": 2, "total": 3})
    assert media_bus.get(job_id)["last_movement_ts"] >= lm
    print("[3] PASS  last_movement_ts exposed + advances with movement (stall basis)")


# --------------------------------------------------------------------------- #
# 4) idempotent ALTER migration on a pre-existing (pre-feature) DB.
# --------------------------------------------------------------------------- #
def test_migration_idempotent():
    from hugpy_video.intel import media_bus
    tmpdir = tempfile.mkdtemp(prefix="hugpy_test_stagelog_migrate_")
    db = os.path.join(tmpdir, "media_jobs.db")

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE media_jobs ("
        " job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT,"
        " result_json TEXT, claim_token TEXT, created REAL, updated REAL,"
        " progress_json TEXT)")
    conn.execute(
        "INSERT INTO media_jobs (job_id, name, status, created, updated) "
        "VALUES ('old-job', 'generate_scene', 'running', 1.0, 1.0)")
    conn.commit()
    conn.close()

    def _cols():
        c = sqlite3.connect(db)
        try:
            return {r[1] for r in c.execute("PRAGMA table_info(media_jobs)")}
        finally:
            c.close()

    assert "stage_log_json" not in _cols(), "precondition: old DB lacks the column"

    media_bus.DB_PATH = db
    media_bus._initialized = False
    media_bus._ensure_db()
    assert "stage_log_json" in _cols(), "migration did not add stage_log_json"

    # Re-run twice — the duplicate-column ALTER must be swallowed.
    for _ in range(2):
        media_bus._initialized = False
        media_bus._ensure_db()
    assert "stage_log_json" in _cols()

    # A pre-existing row now accumulates a timeline + reads back cleanly.
    media_bus.set_progress("old-job", {"stage": "generating", "done": 1, "total": 2})
    v = media_bus.get("old-job")
    assert v["stage_log"] and v["stage_log"][-1]["stage"] == "generating", v
    # Unknown id -> empty telemetry, never a crash.
    u = media_bus.get("no-such-id")
    assert u["stage_log"] == [] and u["failure"] is None, u
    print("[4] PASS  ALTER migration adds stage_log_json + idempotent + safe reads")


def _run_all():
    test_timeline_dedup()
    test_timeline_survives_failure()
    test_stall_last_movement()
    test_migration_idempotent()
    print("\nALL media-bus stage-timeline backend checks passed")


if __name__ == "__main__":
    _run_all()
