"""k57 — the Active-Processes feed: fast, read-only, informative.

The operator's report was "no logs are ever shown, the progress bar never
updates, the errors are never explicit". The live diagnosis was upstream of all
three: `GET /video/jobs` HUNG past 60s while single-job GETs returned instantly,
so the panel never received the rows that carry the logs/progress/errors. The
cause was per-row work in the listing — each row asked the reservation registry
for its placement, and that read took a WRITE lock (its lapsed-lease sweep) on a
store live renderers heartbeat into, plus re-read measured.json off shared
storage — on top of an unindexed full-table scan of the whole job history.

Covers (GPU-free):

  1. The listing does NO per-row blocking I/O: N=100 rows -> exactly ONE
     reservation-store read, ONE measured.json read, ZERO write-capable bus
     connections, and it completes fast.
  2. The reservation snapshot NEVER writes (the sweep is a predicate, not an
     UPDATE) and still hides a lapsed-lease claim.
  3. progress_ratio moves WITHIN a segment from the runner's step counter, and
     is null when nothing measurable is reported.
  4. A failed job's failure envelope reaches the listing VERBATIM (code +
     message), and its stage_log is complete while a live row carries a tail.
  5. Stale in-flight rows (progressed_at aged past the window) are hidden from
     the default view and reappear behind include_stale — without being mutated.
  6. The worker's render subprocess relays denoise-step progress to its sink.

"""
import json
import os
import tempfile
import time

import pytest


@pytest.fixture(autouse=True)
def _restore_bus_pointers():
    """``_private_bus`` repoints the process-wide media_bus at a temp DB; put it
    back afterwards so a sibling file (test_studio_offload's own private bus)
    never claims a queue this file left behind."""
    from hugpy_video.intel import media_bus, job_lifecycle
    saved = (media_bus.DB_PATH, media_bus._initialized, job_lifecycle._initialized)
    yield
    media_bus.DB_PATH, media_bus._initialized, job_lifecycle._initialized = saved


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _private_bus():
    """Repoint media_bus.DB_PATH to a private temp db + reset the one-time init
    flag (the _selftest_scene idiom), so these checks never touch the real bus."""
    from hugpy_video.intel import media_bus
    tmpdir = tempfile.mkdtemp(prefix="hugpy_test_listing_")
    media_bus.DB_PATH = os.path.join(tmpdir, "media_jobs.db")
    media_bus._initialized = False
    media_bus._ensure_db()
    # The k117 lifecycle sidecar table is created ONCE per process on its first
    # read (a single schema DDL on the write handle). Warm it here, as part of
    # the private-bus setup, so the listing under test measures only the
    # steady-state read path.
    from hugpy_video.intel import job_lifecycle
    job_lifecycle._initialized = False
    job_lifecycle._ensure()
    return media_bus, tmpdir


def step_progress_target(spec_dict, conn, cancel_event):
    """A FAKE render child — module-level so multiprocessing 'spawn' can pickle it
    by reference (the sibling watchdog test's idiom). Streams three denoise-step
    frames, then the settled payload."""
    from hugpy_fleet.worker import _studio_subproc
    for step in (1, 2, 3):
        conn.send({_studio_subproc._PROGRESS_KEY: {
            "phase": "rendering", "step": step, "steps": 3}})
    conn.send({"ok": True, "path": "/shared/clip.mp4", "frames": 81})
    conn.close()


def _insert(media_bus, job_id, *, name="studio_i2v", status="running",
            created=None, updated=None, progress=None, stage_log=None,
            result=None):
    """Write one row directly (the runners' side of the bus), so a test can shape
    a catalog no ordinary enqueue sequence would produce."""
    now = time.time()
    conn = media_bus._connect()
    try:
        conn.execute(
            "INSERT INTO media_jobs (job_id, name, status, spec_json, result_json, "
            "claim_token, created, updated, progress_json, stage_log_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job_id, name, status, "{}",
             json.dumps(result) if result is not None else None,
             None,
             created if created is not None else now,
             updated if updated is not None else now,
             json.dumps(progress) if progress is not None else None,
             json.dumps(stage_log) if stage_log is not None else None))
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
def test_listing_does_no_per_row_io():
    """[1] N=100 in-flight rows -> ONE registry read, ONE measured.json read, no
    write-capable bus connection, and a fast return. This is THE regression: the
    old path paid a sqlite write transaction + a file read PER ROW."""
    media_bus, _ = _private_bus()
    from hugpy_video.intel import placement
    from hugpy_video.intel.reservation import templates

    for i in range(100):
        _insert(media_bus, f"job{i:03d}", name="generate_studio_movie",
                progress={"stage": "generating", "segment_done": 0,
                          "segment_total": 2},
                stage_log=[{"stage": "generating", "ts": time.time(),
                            "ts_last": time.time(), "count": 3,
                            "detail": "segment 0/2 · t2v · worker rendering"}])

    calls = {"registry": 0, "measured": 0, "rw_connect": 0}

    def _snapshot():
        calls["registry"] += 1
        return {}

    def _measured(path):
        calls["measured"] += 1
        return {}

    def _rw_connect():
        calls["rw_connect"] += 1
        return real_connect()

    real_connect = media_bus._connect
    from hugpy_video.intel.reservation.registry import reservation_registry
    orig_snapshot = reservation_registry.active_snapshot
    orig_measured = templates._read_measured_uncached
    reservation_registry.active_snapshot = _snapshot
    templates._read_measured_uncached = _measured
    templates._measured_cache["key"] = None      # force a cold overlay read
    media_bus._connect = _rw_connect
    try:
        t0 = time.time()
        rows = media_bus.list_jobs(limit=200)
        snap = placement.PlacementSnapshot()
        for r in rows:
            r["placement"] = snap.placement_for(r["job_id"], r["name"])
        elapsed = time.time() - t0
    finally:
        reservation_registry.active_snapshot = orig_snapshot
        templates._read_measured_uncached = orig_measured
        media_bus._connect = real_connect

    assert len(rows) == 100, f"expected 100 rows, got {len(rows)}"
    assert calls["registry"] == 1, (
        f"reservation store read {calls['registry']}x — the listing must read it "
        "ONCE per page, never per row")
    assert calls["measured"] <= 1, (
        f"measured.json read {calls['measured']}x — the template overlay must be "
        "memoized, not re-read per row")
    assert calls["rw_connect"] == 0, (
        "the listing opened a WRITE-capable connection — the read path must use "
        "the mode=ro handle so it can never contend with a running render")
    assert elapsed < 1.0, f"listing took {elapsed:.2f}s for 100 rows"
    print(f"[1] PASS  100-row listing: 1 registry read, "
          f"{calls['measured']} overlay read, 0 rw connections, {elapsed*1000:.0f}ms")


def test_reservation_snapshot_is_read_only():
    """[2] active_snapshot() hides a lapsed-lease claim WITHOUT writing — the old
    get()/active() path expressed the same truth as an UPDATE, which is what put a
    write lock in the middle of a read."""
    from hugpy_video.intel.reservation.registry import ReservationRegistry
    tmpdir = tempfile.mkdtemp(prefix="hugpy_test_resv_")
    rr = ReservationRegistry(path=os.path.join(tmpdir, "reservations.db"),
                             lease_ttl_s=60.0)
    assert rr.claim("live-run", "w1", "ae", "generate_studio_movie", 1 << 30)
    assert rr.claim("dead-run", "w1", "ae", "studio_i2v", 1 << 30)
    # Age the second claim's lease out of the window (the orphaned-claim case).
    import sqlite3
    conn = sqlite3.connect(rr.path)
    try:
        conn.execute("UPDATE reservations SET heartbeat_at=? WHERE run_id=?",
                     (time.time() - 3600, "dead-run"))
        conn.commit()
    finally:
        conn.close()

    before = os.stat(rr.path).st_mtime_ns
    snap = rr.active_snapshot()
    after = os.stat(rr.path).st_mtime_ns

    assert "live-run" in snap, "a live claim must be in the snapshot"
    assert "dead-run" not in snap, "a lapsed-lease claim must not read as active"
    assert before == after, "active_snapshot() wrote to the store"
    # The lapsed row is still THERE (hidden, not swept) — the read path mutates nothing.
    conn = sqlite3.connect(rr.path)
    try:
        state = conn.execute(
            "SELECT state FROM reservations WHERE run_id='dead-run'").fetchone()[0]
    finally:
        conn.close()
    assert state == "active", f"the read path rewrote the row (state={state})"
    print("[2] PASS  reservation snapshot hides a lapsed lease without writing")


def test_progress_ratio_moves_within_a_segment():
    """[3] The bar MOVES during a long segment: segment 0 of 2 at denoise step
    15/30 is 25%, not 0%. And an unmeasurable blob yields null, never a fake 0."""
    from hugpy_video.intel import media_bus

    # A delegated movie segment: the worker's blob nests under current.worker.
    blob = {"stage": "generating", "segment_done": 0, "segment_total": 2,
            "current": {"segment_id": "seg_00", "capability": "t2v",
                        "worker": {"phase": "rendering", "step": 15, "steps": 30}}}
    ratio = media_bus._progress_ratio(blob)
    assert abs(ratio - 0.25) < 1e-6, f"expected 0.25 within segment 0/2, got {ratio}"
    detail = media_bus._progress_detail(blob)
    assert detail["segment_done"] == 0 and detail["segment_total"] == 2
    assert detail["step"] == 15 and detail["steps"] == 30

    # Same movie one segment later, mid-render.
    blob2 = dict(blob, segment_done=1)
    assert abs(media_bus._progress_ratio(blob2) - 0.75) < 1e-6

    # A frame-loop render reports done/total flat.
    assert abs(media_bus._progress_ratio({"done": 12, "total": 48}) - 0.25) < 1e-6

    # A HELD job has honestly made no progress.
    assert media_bus._progress_ratio(
        {"phase": "awaiting_capacity", "reason": {}}) == 0.0

    # Nothing measurable -> null (the panel then draws NO bar rather than 0%).
    assert media_bus._progress_ratio({"phase": "rendering"}) is None
    assert media_bus._progress_ratio(None) is None
    # A half-pair is not a fraction.
    assert media_bus._progress_ratio({"step": 4}) is None
    assert media_bus._progress_ratio({"segment_done": 1, "segment_total": 0}) is None
    print("[3] PASS  progress_ratio advances within a segment; null when unknown")


def test_failure_envelope_and_stage_log_reach_the_listing():
    """[4] A failed job's exact code+message ride the listing verbatim (never a
    generic "failed"), with its FULL timeline; a live row carries a bounded tail
    plus the untruncated count."""
    media_bus, _ = _private_bus()
    msg = ("wan i2v needs diffusers>=0.39 and ftfy — install with "
           "`pip install 'diffusers>=0.39' ftfy`")
    _insert(media_bus, "broken", name="studio_i2v", status="failed",
            result={"ok": False, "error": {"code": "deps_missing",
                                           "message": msg, "retryable": False}},
            stage_log=[{"stage": "loading", "ts": 1.0, "ts_last": 2.0, "count": 1},
                       {"stage": "failed", "ts": 3.0, "ts_last": 3.0, "count": 1,
                        "detail": msg, "code": "deps_missing", "message": msg,
                        "retryable": False, "failed_at_stage": "loading"}])
    now = time.time()
    live_log = [{"stage": "generating", "ts": now + i, "ts_last": now + i,
                 "count": 1, "detail": f"frame {i}/20"} for i in range(20)]
    _insert(media_bus, "live", name="generate_scene", status="running",
            progress={"stage": "generating", "done": 5, "total": 20},
            stage_log=live_log)

    rows = {r["job_id"]: r for r in media_bus.list_jobs(include_terminal=True)}

    failed = rows["broken"]
    assert failed["failure"]["code"] == "deps_missing"
    assert failed["failure"]["message"] == msg, "the message must be VERBATIM"
    assert failed["failure"]["stage"] == "loading", "where it broke"
    assert len(failed["stage_log"]) == 2, "a terminal row keeps its FULL timeline"

    live = rows["live"]
    assert len(live["stage_log"]) == media_bus._STAGE_TAIL, "a live row ships a tail"
    assert live["stage_log_total"] == 20, "…and says how much was elided"
    assert live["stage_log"][-1]["detail"] == "frame 19/20", "the tail is the LATEST"
    assert abs(live["progress_ratio"] - 0.25) < 1e-6
    print("[4] PASS  failure envelope verbatim + full timeline; live rows tail")


def test_stale_inflight_rows_are_hidden_not_mutated():
    """[5] A row abandoned by a dead process (progressed_at days old) drops out of
    the active view, comes back behind include_stale, and is never rewritten — a
    view does not get to mutate a renderer's store."""
    media_bus, _ = _private_bus()
    now = time.time()
    old = now - 3 * 86400
    _insert(media_bus, "ghost", name="studio_i2v", status="running",
            created=old, updated=old,
            stage_log=[{"stage": "rendering", "ts": old, "ts_last": old,
                        "count": 1}])
    _insert(media_bus, "alive", name="studio_i2v", status="running",
            created=now, updated=now,
            stage_log=[{"stage": "rendering", "ts": now, "ts_last": now,
                        "count": 1}])
    # A job HELD for capacity only rewrites its marker when the hold CHANGES, so
    # its movement clock idles while it is perfectly alive — the window must be
    # generous enough not to hide it.
    held = now - 3600
    _insert(media_bus, "held", name="generate_studio_movie", status="queued",
            created=held, updated=held,
            progress={"phase": "awaiting_capacity", "reason": {}},
            stage_log=[{"stage": "awaiting_capacity", "ts": held, "ts_last": held,
                        "count": 1}])

    before = os.stat(media_bus.DB_PATH).st_mtime_ns
    active = {r["job_id"] for r in media_bus.list_jobs()}
    assert active == {"alive", "held"}, f"active view showed {active}"

    with_stale = {r["job_id"]: r for r in media_bus.list_jobs(include_stale=True)}
    assert set(with_stale) == {"ghost", "alive", "held"}
    assert with_stale["ghost"]["stale"] is True
    assert with_stale["ghost"]["stale_for_s"] > 2 * 86400
    assert with_stale["alive"]["stale"] is False
    assert os.stat(media_bus.DB_PATH).st_mtime_ns == before, (
        "the listing wrote to the bus DB — the read path must never write")

    # The window is env-tunable, and a tighter one ages the same row out sooner.
    os.environ["HUGPY_MEDIA_BUS_STALE_SECONDS"] = "60"
    try:
        assert {r["job_id"] for r in media_bus.list_jobs()} == {"alive"}
    finally:
        del os.environ["HUGPY_MEDIA_BUS_STALE_SECONDS"]
    print("[5] PASS  stale in-flight rows hidden (not mutated); window tunable")


def test_render_subprocess_relays_step_progress():
    """[6] The render child streams its denoise step up the result pipe and the
    worker's sink sees it — without that hop the worker can only ever report
    "rendering" and the bar has nothing to move on."""
    from hugpy_fleet.worker import _studio_subproc

    seen = []
    payload = _studio_subproc.run_render_subprocess(
        {"prompt": "x"}, None, timeout_s=30.0,
        on_progress=seen.append, _target=step_progress_target, _poll_s=0.05)

    assert payload.get("ok") is True, f"the settled payload was lost: {payload}"
    assert [b["step"] for b in seen] == [1, 2, 3], (
        f"progress frames not relayed in order: {seen}")
    assert all(b["steps"] == 3 for b in seen)
    print("[6] PASS  render subprocess relays step progress; payload still settles")
