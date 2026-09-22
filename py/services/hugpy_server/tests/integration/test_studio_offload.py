"""Studio render OFFLOAD to a GPU worker (option a) — conformance.

Locks the offload seam as executable checks, in the same script style as
``test_studio_cancel.py`` / ``test_studio_t2v.py`` (plain python, ``__main__``
guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check
FAILED, every check independently run so a failing one never masks the rest).
pytest is NOT installed in this venv, so there are no fixtures.

What is under test:
  * DECISION rule (video_intel/runners/studio_i2v.should_delegate): synthetic ->
    local, real -> delegate WHEN a worker is set, worker unset -> local, and the
    TEST-ONLY HUGPY_STUDIO_FORCE_REMOTE=1 override -> delegate even synthetic.
  * PAYLOAD round-trip: a spec's asdict -> studio_i2v_from_dict -> the SAME
    CapabilityRequest the in-process path builds (identical delegated construction).
  * artifact_result_to_payload / _payload_to_job_result: Ok -> shared-path ingest,
    Err -> the SAME JobError mapping (incl. retryable classification), worker
    error -> JobError, malformed payload -> retryable internal error.
  * StudioRenderManager state machine: one render at a time (busy), idempotent
    re-POST, cancel/status of an unknown job.
  * E2E with NO GPU and NO release: an EPHEMERAL worker agent spun from THIS tree
    on a loopback port (werkzeug, in-process, synthetic render only — no model
    load, no systemd, not registered as a real worker). Prove enqueue -> delegate
    -> worker thread renders synthetic -> shared-path result -> central ingests ->
    job done in the media store; and a cancel mid-render settles as 'cancelled'
    with NO clip left behind. The ephemeral agent is torn down afterwards.

"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
import urllib.error
from dataclasses import asdict

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
# The offload suites PROVE THE PLUMBING with the synthetic prover as a stand-in for a
# GPU render (that is what HUGPY_STUDIO_FORCE_REMOTE exists for), so they must opt in
# to it explicitly — synthetic became opt-in on 2026-07-27 so a real render can never
# silently degrade into noise. This is a HARNESS opt-in, never a product default.
os.environ.setdefault("STUDIO_ALLOW_SYNTHETIC", "1")

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import studio_i2v as S
from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.enums import Capability
from hugpy_video.intel.studio.errors import Err, ErrorCode, Ok, StageError
from hugpy_video.intel.studio.job import make_studio_i2v, studio_i2v_from_dict
from hugpy_video.intel.studio.schemas import CapabilityRequest, Resolution
from hugpy_fleet.worker.studio_render import StudioRenderManager, register_studio_routes

# LIVE-DB SAFETY + ISOLATION (mirrors test_invariants_conformance): media_bus.DB_PATH
# is derived from DEFAULT_ROOT at import and points at the LIVE media_jobs.db, which
# the running dev central's worker DAEMON continuously drains. If our E2E enqueued
# there, that daemon could CLAIM our job first (and run it in-process, without our
# ephemeral worker), stealing the render. Repoint the bus at a throwaway db BEFORE
# any db op so our jobs are ours alone and the live queue is never touched. (Clip
# storage still lands under DEFAULT_ROOT — only the job ledger moves.)
import tempfile  # noqa: E402
_TMP_DB_DIR = tempfile.mkdtemp(prefix="hugpy_offload_test_")
media_bus.DB_PATH = os.path.join(_TMP_DB_DIR, "media_jobs.db")
media_bus._initialized = False

_FFMPEG = shutil.which("ffmpeg") is not None

# Env keys we mutate — captured so every check restores the process env it found.
_ENV_KEYS = (
    "HUGPY_STUDIO_WORKER",
    "HUGPY_STUDIO_FORCE_REMOTE",
    "HUGPY_STUDIO_POLL_INTERVAL_S",
    "HUGPY_STUDIO_DELEGATE_TIMEOUT_S",
    # k91: the PRIMARY deadline (no-forward-progress window). Listed here so every
    # check starts from a clean set of all THREE clocks — a leaked stall window would
    # silently change which one fires in the checks that follow it.
    "HUGPY_STUDIO_STALL_TIMEOUT_S",
    "HUGPY_STUDIO_OVERALL_CAP_S",
    "HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S",
    "HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S",
    "HUGPY_STUDIO_QUEUE_DEPTH",
)


def _clear_offload_env() -> None:
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def _clip_files_under(root: str) -> list:
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn == "clip.mp4":
                found.append(os.path.join(dirpath, fn))
    return found


def _synth_spec(out_root=None, *, cap="i2v", w=256, h=256, fps=24, seed=0):
    """A spec whose VRAM budget (0.5GB) is below every real model's floor, so the
    router binds a SYNTHETIC runner — a real render with NO GPU/weights."""
    return make_studio_i2v(
        capability=cap, width=w, height=h, fps=fps, vram_budget_gb=0.5,
        seed=seed, out_root=out_root)


def _real_spec():
    """A spec that resolves to a REAL model (wan2.1-t2v-1.3b @ 832x480, 8GB)."""
    return make_studio_i2v(
        capability="t2v", width=832, height=480, fps=16, vram_budget_gb=8.0, seed=0)


# --------------------------------------------------------------------------- #
# (1) DECISION rule matrix
# --------------------------------------------------------------------------- #
def test_decision_rule_matrix():
    _clear_offload_env()
    synth = _synth_spec()
    real = _real_spec()
    try:
        # No worker configured -> never delegate (in-process, unchanged).
        assert S.should_delegate(synth) is False, "no worker: synthetic must be local"
        assert S.should_delegate(real) is False, "no worker: real must be local"

        # Worker set -> synthetic local, real delegated (the core decision rule).
        os.environ["HUGPY_STUDIO_WORKER"] = "http://127.0.0.1:9"
        assert S.should_delegate(synth) is False, "worker set: synthetic must stay local"
        assert S.should_delegate(real) is True, "worker set: real must delegate"

        # Test-only force-remote -> delegate even a synthetic render.
        os.environ["HUGPY_STUDIO_FORCE_REMOTE"] = "1"
        assert S.should_delegate(synth) is True, "force-remote: synthetic must delegate"
        assert S.should_delegate(real) is True, "force-remote: real must delegate"

        # resolves_to_real_model is the pure discriminator (worker-independent).
        assert S.resolves_to_real_model(synth) is False
        assert S.resolves_to_real_model(real) is True
    finally:
        _clear_offload_env()


# --------------------------------------------------------------------------- #
# (2) PAYLOAD round-trip: spec -> asdict -> from_dict -> identical request
# --------------------------------------------------------------------------- #
def test_payload_request_round_trip():
    spec = make_studio_i2v(
        capability="i2v", width=512, height=384, fps=12, vram_budget_gb=8.0,
        seed=7, prompt="a cat", negative="blurry", source_video=None)
    d = asdict(spec)                         # the exact wire form central POSTs
    rebuilt = studio_i2v_from_dict(d)        # the exact spec the worker rebuilds

    # The request the worker builds must equal the request the in-process path
    # builds from the original spec — byte-for-byte (CapabilityRequest is frozen).
    req_direct = S.build_capability_request(spec)
    req_wire = S.build_capability_request(rebuilt)
    assert req_direct == req_wire, (
        f"delegated request must equal in-process request; "
        f"{req_direct} != {req_wire}")

    # And it must equal a hand-built CapabilityRequest with the spec's fields.
    expected = CapabilityRequest(
        capability=Capability("i2v"),
        target_resolution=Resolution(512, 384, 12),
        vram_budget_gb=8.0,
        source_video=None)
    assert req_wire == expected, f"request fields drifted: {req_wire} != {expected}"


# --------------------------------------------------------------------------- #
# (3) artifact_result_to_payload — Ok + Err (+ retryable classification)
# --------------------------------------------------------------------------- #
def test_artifact_result_to_payload():
    art = Artifact(path="/mnt/llm_storage/video_intel/studio/clips/abc/clip.mp4",
                   content_hash="abc", frames=48, width=256, height=256,
                   duration_s=2.0, resumed=False)
    ok = S.artifact_result_to_payload(Ok(art))
    assert ok["ok"] is True and ok["path"] == art.path, f"ok payload wrong: {ok}"
    assert ok["frames"] == 48 and ok["content_hash"] == "abc"

    # CANCELLED -> not retryable (intentional).
    canc = S.artifact_result_to_payload(
        Err(StageError(ErrorCode.CANCELLED, "stopped")))
    assert canc["ok"] is False, f"cancel payload must be ok=False: {canc}"
    assert canc["error"]["code"] == "cancelled", f"code: {canc}"
    assert canc["error"]["retryable"] is False, "cancel must not be retryable"

    # OOM -> retryable (transient/resource).
    oom = S.artifact_result_to_payload(Err(StageError(ErrorCode.OOM, "boom")))
    assert oom["error"]["code"] == "oom" and oom["error"]["retryable"] is True, oom

    # NO_GPU -> not retryable (policy/routing on this box).
    nogpu = S.artifact_result_to_payload(Err(StageError(ErrorCode.NO_GPU, "no cuda")))
    assert nogpu["error"]["code"] == "no_gpu" and nogpu["error"]["retryable"] is False, nogpu


# --------------------------------------------------------------------------- #
# (4) _payload_to_job_result — worker error -> JobError; malformed -> retryable
# --------------------------------------------------------------------------- #
def test_payload_to_job_result_errors():
    # A worker-side error payload rebuilds a JobError verbatim.
    jr = S._payload_to_job_result(
        {"ok": False, "error": {"code": "worker_lost", "message": "gone",
                                "retryable": True}}, "job-x")
    assert jr.ok is False and jr.error.code == "worker_lost", f"{jr}"
    assert jr.error.retryable is True and jr.error.message == "gone"
    assert jr.outputs == ()

    # A cancelled worker render maps to a 'cancelled' JobError (run_claimed will
    # then record a 'cancelled' terminal status, not 'failed').
    jc = S._payload_to_job_result(
        {"ok": False, "error": {"code": "cancelled", "message": "stopped",
                                "retryable": False}}, "job-c")
    assert jc.error.code == "cancelled" and jc.error.retryable is False, f"{jc}"

    # A malformed/empty payload is itself errors-as-data (retryable internal).
    jm = S._payload_to_job_result({}, "job-m")
    assert jm.ok is False and jm.error.retryable is True, f"{jm}"

    # ok=True but no path -> retryable internal (never a false success).
    jp = S._payload_to_job_result({"ok": True}, "job-p")
    assert jp.ok is False and jp.error.code == "internal", f"{jp}"


# --------------------------------------------------------------------------- #
# (5) StudioRenderManager — QUEUE: accept/position, idempotent, full->409,
#     queued-cancel, running-cancel, unknown status/cancel.
# --------------------------------------------------------------------------- #
def test_render_manager_state_machine():
    # A stub manager whose render thread never settles, so the FIRST job stays
    # 'running' and later jobs stay 'queued' — deterministic state observation.
    class _StubMgr(StudioRenderManager):
        def _run(self, job):
            return  # leave status 'running' + _active set (simulated in-flight)

    # Bounded depth 2 so we can drive the queue to FULL deterministically.
    mgr = _StubMgr(queue_depth=2)

    # First job -> starts immediately (no position).
    ok_a, why_a, pos_a = mgr.start("job-A", {"width": 1})
    assert ok_a and why_a == "started" and pos_a is None, (ok_a, why_a, pos_a)

    # A DIFFERENT job while one is in flight -> QUEUED (not busy), position 1.
    ok_b, why_b, pos_b = mgr.start("job-B", {"width": 1})
    assert ok_b is True and why_b == "queued" and pos_b == 1, (ok_b, why_b, pos_b)

    # Another -> queued at position 2.
    ok_c, why_c, pos_c = mgr.start("job-C", {"width": 1})
    assert ok_c is True and why_c == "queued" and pos_c == 2, (ok_c, why_c, pos_c)

    # status reflects the queued position (central forwards this to the console).
    assert mgr.status("job-B")["status"] == "queued" and mgr.status("job-B")["position"] == 1
    assert mgr.status("job-C")["status"] == "queued" and mgr.status("job-C")["position"] == 2

    # Queue is now FULL (depth 2) -> the ONLY 409 (start returns ok=False, "full").
    ok_d, why_d, pos_d = mgr.start("job-D", {"width": 1})
    assert ok_d is False and why_d == "full" and pos_d is None, (ok_d, why_d, pos_d)

    # A re-POST of a KNOWN job -> idempotent (exists), never re-queued.
    ok_a2, why_a2, pos_a2 = mgr.start("job-A", {"width": 1})
    assert ok_a2 is True and why_a2 == "exists" and pos_a2 is None, (ok_a2, why_a2, pos_a2)

    # cancel of a QUEUED job -> removed from the queue, settled cancelled WITHOUT
    # rendering; the response says 'cancelled' and the polled result carries the
    # cancelled error payload (so central records a 'cancelled' terminal).
    cq = mgr.cancel("job-B")
    assert cq["cancelled"] is True and cq["status"] == "cancelled", cq
    stq = mgr.status("job-B")
    assert stq["result"] and stq["result"]["ok"] is False, stq
    assert stq["result"]["error"]["code"] == "cancelled", stq

    # C moves up to position 1 now that B is gone, and there's room to queue again.
    assert mgr.status("job-C")["position"] == 1, mgr.status("job-C")
    ok_e, why_e, pos_e = mgr.start("job-E", {"width": 1})
    assert ok_e is True and why_e == "queued" and pos_e == 2, (ok_e, why_e, pos_e)

    # cancel of a RUNNING job -> cancelling (cooperative); unknown -> not cancelled.
    c = mgr.cancel("job-A")
    assert c["cancelled"] is True and c["status"] == "cancelling", c
    cu = mgr.cancel("nope")
    assert cu["cancelled"] is False and cu["status"] == "unknown", cu

    # status of an unknown job -> 'unknown' (central reads this as worker_lost).
    su = mgr.status("nope")
    assert su["status"] == "unknown" and su["result"] is None, su


# --------------------------------------------------------------------------- #
# (5b) StudioRenderManager — the FIFO drains SERIALLY (one render executes at a
#      time; each render's finally promotes the next queued job, in order).
# --------------------------------------------------------------------------- #
def test_render_manager_serial_drain_fifo():
    # A gated stub: each render blocks on a shared gate, then settles + promotes the
    # next queued job exactly as the real _run's finally does. Releasing the gate
    # lets the whole FIFO drain; we assert it drained in enqueue order, one at a time.
    class _GatedMgr(StudioRenderManager):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.gate = threading.Event()
            self.order = []
            self.max_concurrent = 0
            self._live = 0

        def _run(self, job):
            with self._lock:
                self._live += 1
                self.max_concurrent = max(self.max_concurrent, self._live)
            self.gate.wait(timeout=5.0)
            with self._lock:
                self._live -= 1
                job.status = "done"
                job.result = {"ok": True, "job": job.job_id}
                job.progress = None
                self.order.append(job.job_id)
                if self._active == job.job_id:
                    self._active = None
                self._start_next_locked()

    mgr = _GatedMgr(queue_depth=8)
    for jid in ("A", "B", "C", "D"):
        mgr.start(jid, {"width": 1})
    # A is running (blocked on the gate); B/C/D are queued behind it.
    assert mgr.status("A")["status"] == "running", mgr.status("A")
    assert [mgr.status(j)["position"] for j in ("B", "C", "D")] == [1, 2, 3]

    mgr.gate.set()  # release -> the FIFO drains serially
    assert _wait_until(lambda: len(mgr.order) == 4, 5.0), f"drain stalled: {mgr.order}"
    assert mgr.order == ["A", "B", "C", "D"], f"FIFO order violated: {mgr.order}"
    assert mgr.max_concurrent == 1, f"more than one render ran at once: {mgr.max_concurrent}"


# --------------------------------------------------------------------------- #
# ephemeral worker plumbing for the E2E checks
# --------------------------------------------------------------------------- #
class _EphemeralWorker:
    """A studio worker agent spun from THIS tree on a loopback port: a bare Flask
    app with ONLY the studio render routes (no central registration, no systemd,
    no model load). Torn down via stop()."""

    def __init__(self):
        from flask import Flask
        from werkzeug.serving import make_server

        app = Flask("ephemeral-studio-worker")
        self.manager = register_studio_routes(
            app, worker_id="ephemeral-test", worker_name="ephemeral")
        self._srv = make_server("127.0.0.1", 0, app, threaded=True)
        self.port = self._srv.server_port
        self._thread = threading.Thread(
            target=self._srv.serve_forever, name="ephemeral-worker", daemon=True)
        self._thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        try:
            self._srv.shutdown()
        except Exception:
            pass


def _wait_until(pred, timeout_s: float, interval_s: float = 0.05) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval_s)
    return False


# --------------------------------------------------------------------------- #
# (6) E2E happy path: enqueue -> delegate -> worker renders -> central ingests
# --------------------------------------------------------------------------- #
def test_e2e_delegate_synthetic_render():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping E2E synthetic render)")
        return
    _clear_offload_env()
    worker = _EphemeralWorker()
    out_root = os.path.join(DEFAULT_ROOT, "video_intel", "studio", "clips",
                            f"offload-e2e-{os.getpid()}")
    os.environ["HUGPY_STUDIO_WORKER"] = worker.base
    os.environ["HUGPY_STUDIO_FORCE_REMOTE"] = "1"     # delegate the synthetic render
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.1"
    try:
        spec = _synth_spec(out_root=out_root, seed=101)
        job_id = media_bus.enqueue("studio_i2v", spec)

        # work_once claims + run_claimed -> run_studio_i2v -> DELEGATE (blocks in
        # the poll loop until the remote synthetic render settles), then INGESTS
        # the clip from the shared path. Synchronous by design.
        processed = media_bus.work_once()
        assert processed == job_id, f"work_once should process our job; got {processed}"

        view = media_bus.get(job_id)
        assert view["status"] == "done", f"job must be done; got {view['status']}"
        result = view["result"]
        assert result and result["ok"] is True, f"delegated render must be ok: {result}"
        outs = result.get("outputs") or []
        assert len(outs) == 1, f"one video MediaRef expected; got {outs}"
        ref = outs[0]
        assert ref["kind"] == "video", f"output must be kind=video; got {ref['kind']}"
        uri = ref["uri"]
        # The clip was written by the worker under the SHARED media-store root and
        # ingested by central directly (no b64) — prove the file is really there.
        assert os.path.isfile(uri) and os.path.getsize(uri) > 0, (
            f"ingested clip must exist under the shared root: {uri}")
        assert os.path.commonpath([os.path.realpath(uri),
                                    os.path.realpath(DEFAULT_ROOT)]) == \
            os.path.realpath(DEFAULT_ROOT), f"clip must be under the shared root: {uri}"
    finally:
        worker.stop()
        _clear_offload_env()
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (7) E2E cancel mid-render: media_bus.cancel -> forwarded -> settles 'cancelled'
# --------------------------------------------------------------------------- #
def test_e2e_cancel_mid_render():
    if not _FFMPEG:
        print("      (ffmpeg unavailable — skipping E2E cancel)")
        return
    _clear_offload_env()

    # Widen the mid-render window DETERMINISTICALLY (test-only): a tiny per-frame
    # sleep in the SAME-PROCESS worker render thread, so the render is reliably
    # in flight when central forwards the cancel. This does NOT change production
    # behavior — it only slows THIS test's frame loop.
    from hugpy_video.intel.studio.runners import synthetic as _syn
    _orig_synth = _syn.synthesize_frame

    def _slow_synth(*a, **k):
        time.sleep(0.03)
        return _orig_synth(*a, **k)

    _syn.synthesize_frame = _slow_synth

    worker = _EphemeralWorker()
    out_root = os.path.join(DEFAULT_ROOT, "video_intel", "studio", "clips",
                            f"offload-cancel-{os.getpid()}")
    os.environ["HUGPY_STUDIO_WORKER"] = worker.base
    os.environ["HUGPY_STUDIO_FORCE_REMOTE"] = "1"
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.1"
    try:
        spec = _synth_spec(out_root=out_root, seed=202)
        job_id = media_bus.enqueue("studio_i2v", spec)

        # Run the full bus cycle in the background (it blocks until settle).
        done = {}

        def _work():
            done["job"] = media_bus.work_once()

        wt = threading.Thread(target=_work, name="e2e-cancel-work", daemon=True)
        wt.start()

        # Wait until the render is genuinely in flight (central flipped 'running'
        # and the worker has started rendering).
        assert _wait_until(lambda: media_bus.get(job_id)["status"] == "running", 10.0), (
            "job never reached 'running'")
        assert _wait_until(
            lambda: worker.manager.status(job_id).get("status") == "running", 10.0), (
            "worker never started the render")
        time.sleep(0.15)  # let a few frames render so this is truly mid-render

        # Cancel via the bus (the ONLY cancel channel) — central's poll loop must
        # detect is_cancelling and forward POST /studio/cancel to the worker.
        res = media_bus.cancel(job_id)
        assert res["cancelled"] is True, f"cancel should engage; got {res}"

        wt.join(timeout=15.0)
        assert not wt.is_alive(), "work thread did not settle after cancel"

        view = media_bus.get(job_id)
        assert view["status"] == "cancelled", (
            f"a cancelled delegated render must settle 'cancelled'; got {view['status']}")
        result = view["result"]
        assert result and result["ok"] is False, f"cancelled result ok=False: {result}"
        assert result["error"]["code"] == "cancelled", f"error code: {result}"
        # abort-before-replace guarantee holds across the offload seam too.
        leftovers = _clip_files_under(out_root)
        assert leftovers == [], f"a cancelled render must leave NO clip.mp4; {leftovers}"
    finally:
        worker.stop()
        _syn.synthesize_frame = _orig_synth
        _clear_offload_env()
        shutil.rmtree(out_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# central-side delegation tests (item 1 queue polling + item 2 kick-off retry).
# These mock the worker's HTTP surface (S._http_post_json / S._http_get_json) so the
# central _delegate_to_worker loop is exercised deterministically with NO worker, NO
# GPU. A terminal ERROR payload (oom) settles the job without needing a real clip
# ingest, so we can assert the poll/retry control flow in isolation.
# --------------------------------------------------------------------------- #
def _spec_for_delegate():
    """Any valid spec — these tests fully mock the HTTP calls, so the spec only has
    to ``asdict`` cleanly (no routing/GPU is exercised)."""
    return _real_spec()


# --------------------------------------------------------------------------- #
# (8) central QUEUED polling: 202 accepted="queued" -> keep polling through queued
#     -> running -> terminal, forwarding the queue position as progress.
# --------------------------------------------------------------------------- #
def test_central_queued_polling_forwards_position():
    _clear_offload_env()
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.01"
    os.environ["HUGPY_STUDIO_OVERALL_CAP_S"] = "10"
    script = [
        {"status": "queued", "position": 2},
        {"status": "queued", "position": 1},
        {"status": "running", "progress": {"phase": "rendering"}},
        {"status": "error", "result": {"ok": False, "error": {
            "code": "oom", "message": "boom", "retryable": True}}},
    ]
    seq = {"i": 0}
    prog_calls = []

    def fake_post(url, payload, timeout):
        return 202, {"ok": True, "accepted": "queued", "position": 2,
                     "pkg_version": S._pkg_version()}

    def fake_get(url, timeout):
        i = min(seq["i"], len(script) - 1)
        seq["i"] += 1
        return 200, dict(script[i])

    op, og, osp = S._http_post_json, S._http_get_json, media_bus.set_progress
    S._http_post_json, S._http_get_json = fake_post, fake_get
    media_bus.set_progress = lambda jid, prog: prog_calls.append(prog)
    try:
        jr = S._delegate_to_worker("http://worker.test", _spec_for_delegate(), "job-q")
        # Delegated (NOT fallen back in-process) and settled on the terminal error.
        assert jr is not None and jr.ok is False, jr
        assert jr.error.code == "oom", jr.error
        # Queue positions were forwarded to the console via set_progress, in order.
        queued = [p.get("position") for p in prog_calls if p.get("phase") == "queued"]
        assert queued == [2, 1], f"queued positions not forwarded in order: {queued}"
    finally:
        S._http_post_json, S._http_get_json, media_bus.set_progress = op, og, osp
        _clear_offload_env()


# --------------------------------------------------------------------------- #
# (9) central 409 (queue FULL) -> retry kick-off within the overall window, then
#     delegate once a slot frees (worker_busy means WAIT, not fail).
# --------------------------------------------------------------------------- #
def test_central_queue_full_retries_then_delegates():
    _clear_offload_env()
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.01"
    os.environ["HUGPY_STUDIO_OVERALL_CAP_S"] = "5"
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S"] = "0.01"
    posts = {"n": 0}

    def fake_post(url, payload, timeout):
        if url.endswith("/studio/render"):
            posts["n"] += 1
            if posts["n"] <= 2:                       # queue full twice, then room
                raise urllib.error.HTTPError(url, 409, "queue full", {}, None)
            return 202, {"ok": True, "accepted": "started",
                         "pkg_version": S._pkg_version()}
        return 200, {}

    def fake_get(url, timeout):
        return 200, {"status": "error", "result": {"ok": False, "error": {
            "code": "oom", "message": "x", "retryable": True}}}

    op, og = S._http_post_json, S._http_get_json
    S._http_post_json, S._http_get_json = fake_post, fake_get
    try:
        jr = S._delegate_to_worker("http://worker.test", _spec_for_delegate(), "job-f")
        assert posts["n"] == 3, f"expected 2x 409 then success; got {posts['n']} posts"
        assert jr is not None and jr.ok is False and jr.error.code == "oom", jr
    finally:
        S._http_post_json, S._http_get_json = op, og
        _clear_offload_env()


# --------------------------------------------------------------------------- #
# (10) central 409 that never clears within the window -> retryable worker_busy
#      (only fails AFTER the delegation window expires, never immediately).
# --------------------------------------------------------------------------- #
def test_central_queue_full_gives_up_worker_busy():
    _clear_offload_env()
    os.environ["HUGPY_STUDIO_OVERALL_CAP_S"] = "0.05"        # window expires fast
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S"] = "0.01"
    posts = {"n": 0}

    def fake_post(url, payload, timeout):
        posts["n"] += 1
        raise urllib.error.HTTPError(url, 409, "queue full", {}, None)

    op = S._http_post_json
    S._http_post_json = fake_post
    try:
        jr = S._delegate_to_worker("http://worker.test", _spec_for_delegate(), "job-fb")
        assert jr is not None and jr.ok is False, jr
        assert jr.error.code == "worker_busy" and jr.error.retryable is True, jr.error
        assert posts["n"] >= 2, f"must retry at least once before giving up; got {posts['n']}"
    finally:
        S._http_post_json = op
        _clear_offload_env()


# --------------------------------------------------------------------------- #
# (11) item 2 — a ConnectionReset at kick-off (post-restart socket window) retries
#      within the reset window, then delegates once the socket recovers.
# --------------------------------------------------------------------------- #
def test_kickoff_reset_retries_then_delegates():
    _clear_offload_env()
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.01"
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S"] = "2"
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S"] = "0.01"
    posts = {"n": 0}

    def fake_post(url, payload, timeout):
        if url.endswith("/studio/render"):
            posts["n"] += 1
            if posts["n"] <= 2:                       # socket window, then recovers
                raise ConnectionResetError("connection reset by peer")
            return 202, {"ok": True, "accepted": "started",
                         "pkg_version": S._pkg_version()}
        return 200, {}

    def fake_get(url, timeout):
        return 200, {"status": "done", "result": {"ok": False, "error": {
            "code": "oom", "message": "x", "retryable": True}}}

    op, og = S._http_post_json, S._http_get_json
    S._http_post_json, S._http_get_json = fake_post, fake_get
    try:
        jr = S._delegate_to_worker("http://worker.test", _spec_for_delegate(), "job-r")
        assert posts["n"] == 3, f"expected 2 resets then success; got {posts['n']}"
        # Delegated (NOT None), so it did NOT double-render via the in-process fallback.
        assert jr is not None and jr.ok is False and jr.error.code == "oom", jr
    finally:
        S._http_post_json, S._http_get_json = op, og
        _clear_offload_env()


# --------------------------------------------------------------------------- #
# (12) item 2 — a ConnectionReset that persists PAST the retry window falls back to
#      the in-process path (return None), the historical "worker unreachable" case.
# --------------------------------------------------------------------------- #
def test_kickoff_reset_falls_back_in_process():
    _clear_offload_env()
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S"] = "0.05"   # window expires fast
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S"] = "0.01"

    def fake_post(url, payload, timeout):
        raise ConnectionResetError("reset persists")

    op = S._http_post_json
    S._http_post_json = fake_post
    try:
        jr = S._delegate_to_worker("http://worker.test", _spec_for_delegate(), "job-rb")
        assert jr is None, f"a reset past the retry window must fall back in-process; got {jr}"
    finally:
        S._http_post_json = op
        _clear_offload_env()


# --------------------------------------------------------------------------- #
# (13-16) k91 — the PROGRESS-AWARE delegation deadline.
#
# THE INCIDENT THESE LOCK. ``HUGPY_STUDIO_DELEGATE_TIMEOUT_S`` used to be a pure WALL
# CLOCK on a running render (30 min), and on 2026-08-06 it cancelled a LIVE one: a
# 14-segment Cinema movie whose segment 0 was at denoise step 19/32 with the worker
# reporting ~86 s/step under GPU contention — roughly 46 minutes of honest work for
# that ONE segment. A clock that cannot tell WEDGED from SLOW answers the wrong
# question.
#
# So there are now THREE clocks, and these four checks pin which one fires when:
#   * the STALL window (``HUGPY_STUDIO_STALL_TIMEOUT_S``, primary) — no advance in the
#     worker's own progress fingerprint;
#   * the absolute RENDER ceiling (``HUGPY_STUDIO_DELEGATE_TIMEOUT_S``, retained and
#     demoted) — a backstop against endless plausible movement;
#   * the OVERALL cap — queue wait + render (already covered by check 10).
# The MESSAGES are asserted on too, because "timed out" without naming the clock is
# exactly the ambiguity that made the incident read as a healthy budget doing its job:
# an operator has to know whether to raise a knob or go look at the worker.
#
# Each check scripts the WORKER STATUS POLLS directly (``S._http_get_json``), so what
# is under test is the delegation loop's own bookkeeping — no GPU, and no waiting
# beyond a few hundred ms of poll ticks.
# --------------------------------------------------------------------------- #
def _stall_env(*, stall, ceiling, overall="60", poll="0.01"):
    """Delegation env with the three deadlines set INDEPENDENTLY, so each check can
    make exactly ONE of them reachable and thereby prove the other two stayed out of
    it (a shared/derived value would let a passing check hide the wrong clock firing)."""
    _clear_offload_env()
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = poll
    os.environ["HUGPY_STUDIO_STALL_TIMEOUT_S"] = str(stall)
    os.environ["HUGPY_STUDIO_DELEGATE_TIMEOUT_S"] = str(ceiling)
    os.environ["HUGPY_STUDIO_OVERALL_CAP_S"] = str(overall)


def _accepting_post(cancels):
    """A worker that accepts every render and RECORDS the cancels it is sent — the
    cancel is half the contract: a deadline that fires without stopping the worker
    leaves a render burning the card with nobody waiting on it."""
    def fake_post(url, payload, timeout):
        if url.endswith("/studio/render"):
            return 202, {"ok": True, "accepted": "started",
                         "pkg_version": S._pkg_version()}
        cancels.append(url)
        return 200, {"cancelled": True}
    return fake_post


def _run_delegate(fake_post, fake_get, job_id):
    op, og = S._http_post_json, S._http_get_json
    S._http_post_json, S._http_get_json = fake_post, fake_get
    try:
        return S._delegate_to_worker("http://worker.test", _spec_for_delegate(), job_id)
    finally:
        S._http_post_json, S._http_get_json = op, og


def _worker_done_payload():
    return {"ok": True, "path": "/x/clip.mp4", "content_hash": "h", "frames": 81,
            "width": 832, "height": 480, "duration_s": 5.0, "resumed": False}


# (13) SLOW BUT ADVANCING is never cancelled — the whole point of the change. The
#      render runs FAR past the stall window while its denoise step keeps ticking, and
#      settles on the worker's own terminal instead of a manufactured timeout.
def test_slow_but_advancing_render_is_not_cancelled():
    _stall_env(stall=0.2, ceiling=60)
    cancels = []
    step = {"n": 0}

    def fake_get(url, timeout):
        step["n"] += 1
        if step["n"] > 60:      # ~0.6s of ticking — 3x the 0.2s stall window
            return 200, {"status": "done", "result": _worker_done_payload()}
        # ONE step of forward progress per poll: slow, but moving.
        return 200, {"status": "running",
                     "progress": {"phase": "denoise", "step": step["n"], "steps": 100}}

    try:
        jr = _run_delegate(_accepting_post(cancels), fake_get, "job-slow")
        assert jr is not None and jr.ok is True, (
            f"a SLOW but ADVANCING render must never be cancelled; got {jr}")
        assert not cancels, f"nothing should have been cancelled; got {cancels}"
    finally:
        _clear_offload_env()


# (14) A render that STOPS MOVING trips the STALL window (the ceiling is set far out of
#      reach here), and the error NAMES the stall clock.
def test_stalled_render_trips_stall_window():
    _stall_env(stall=0.15, ceiling=60)
    cancels = []

    def fake_get(url, timeout):
        # Byte-identical every poll: the worker is answering, but the render is wedged.
        return 200, {"status": "running",
                     "progress": {"phase": "denoise", "step": 19, "steps": 32}}

    try:
        jr = _run_delegate(_accepting_post(cancels), fake_get, "job-stall")
        assert jr is not None and jr.ok is False, jr
        assert jr.error.code == "delegation_timeout", jr.error
        assert jr.error.retryable is True, "a stall timeout must stay retryable"
        assert "no forward progress" in jr.error.message, (
            f"the error must name the STALL clock; got {jr.error.message!r}")
        assert "HUGPY_STUDIO_STALL_TIMEOUT_S" in jr.error.message, jr.error.message
        assert cancels, "a stalled render must be cancelled on the worker"
    finally:
        _clear_offload_env()


# (15) The absolute RENDER CEILING is still a real backstop: a worker emitting endless
#      plausible movement is eventually stopped, and the message says WHICH knob to
#      raise — the one thing the pre-k91 error could not tell an operator.
def test_endless_movement_trips_absolute_ceiling():
    _stall_env(stall=60, ceiling=0.2)     # stall unreachable; the ceiling is the one
    cancels = []
    step = {"n": 0}

    def fake_get(url, timeout):
        step["n"] += 1
        return 200, {"status": "running",
                     "progress": {"phase": "denoise", "step": step["n"], "steps": 10 ** 9}}

    try:
        jr = _run_delegate(_accepting_post(cancels), fake_get, "job-ceil")
        assert jr is not None and jr.ok is False, jr
        assert jr.error.code == "delegation_timeout", jr.error
        assert "HUGPY_STUDIO_DELEGATE_TIMEOUT_S" in jr.error.message, (
            f"the error must name the absolute ceiling knob; got {jr.error.message!r}")
        assert "no forward progress" not in jr.error.message, (
            f"an ADVANCING render must never be reported as stalled; {jr.error.message!r}")
        assert cancels, "a render past the ceiling must be cancelled on the worker"
    finally:
        _clear_offload_env()


# (16) WAITING IS NOT STALLING: a job parked behind another at a STABLE queue position
#      is healthy. It sits there well past the stall window and still renders — the
#      queue wait stays bounded by the OVERALL cap alone, exactly as it was pre-k91.
def test_stable_queue_position_does_not_trip_stall():
    _stall_env(stall=0.15, ceiling=60)
    polls = {"n": 0}

    def fake_get(url, timeout):
        polls["n"] += 1
        if polls["n"] <= 40:           # ~0.4s parked at an UNCHANGING position
            return 200, {"status": "queued", "position": 3}
        return 200, {"status": "done", "result": _worker_done_payload()}

    try:
        jr = _run_delegate(_accepting_post([]), fake_get, "job-queued")
        assert jr is not None and jr.ok is True, (
            f"a long WAIT at a stable queue position must not read as a stall; got {jr}")
    finally:
        _clear_offload_env()
