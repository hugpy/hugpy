"""Studio movie SESSIONS (k91) — naming, the job heartbeat, the session list, and
pause/resume.

WHAT THIS SLICE IS FOR. A studio movie is ONE bus job that can run for HOURS across
many segments, and on 2026-08-06 a 14-segment Cinema movie proved how little of that
was durable: the job record was auto-expired as "wedged" while it was actively
rendering (the runner never heartbeated), a wall-clock delegation deadline then
cancelled a LIVE worker render, and when it was over the rendered segments sat on disk
with no id, no name, and no way to see or continue them. The fixes are a heartbeat
(the runner folds per-segment progress into the bus record), an optional ``title``, and
a movie DIR that is a resumable SESSION described by two sidecars — ``spec.json`` (what
was ASKED for) and ``movie.json`` (what has RENDERED).

Invariants under test:
  * TITLE — optional end to end: POST -> spec.json -> rehydrated spec -> movie.json.
    Absent/blank normalizes to None (UNNAMED), which is what the composer warns on.
  * HEARTBEAT — every movie progress tick carries a monotonic 0..1 ``progress`` and a
    human ``message`` ("segment 1/2 - step 7/12"), and ``job_bridge.on_progress``
    relays that message AND stamps ``progressed_at`` when it CHANGES (the movement
    signal ``expire_pending_orphans`` reads). A REPEATED message is not movement.
  * SESSION LIST — GET /video/studio/movies lists every movie dir with its counts,
    status, and per-segment rows, and a completed segment's clip is FETCHABLE OVER
    HTTP through the existing jailed /video/media route.
  * PAUSE — cancels the live job and records status "paused" (distinct from the
    "partial" a crash leaves), and the session stays resumable.
  * RESUME — re-enqueues the persisted spec as a NEW job; content-addressed reuse
    brings every completed segment back resumed=True. A session already in flight is
    a 409, and a traversal/unknown id is a 404.

Same plain-script style as the other studio suites (numbered ``[n] PASS`` / ``[n] FAIL``
lines, every check independent so a failing one never masks the rest, nonzero exit iff
any FAILED) — and every check is a plain ``test_*`` function, so pytest collects the
file as-is.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_movie_sessions.py
  venv/bin/python -m pytest tests/studio/test_studio_movie_sessions.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict

logging.disable(logging.INFO)  # silence the models_config registry chatter

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
# The delegated checks below prove PLUMBING with the synthetic prover standing in for a
# GPU render (that is what HUGPY_STUDIO_FORCE_REMOTE is for), so they opt in explicitly
# — synthetic became opt-in on 2026-07-27 so a real render can never silently degrade
# into noise. A HARNESS opt-in, never a product default.
os.environ.setdefault("STUDIO_ALLOW_SYNTHETIC", "1")
# These checks exercise the INLINE path unless a check installs its own fake worker;
# clear the ambient studio-worker env this box runs with so nothing reaches a live GPU.
os.environ.pop("HUGPY_STUDIO_WORKER", None)
os.environ.pop("HUGPY_STUDIO_FORCE_REMOTE", None)

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import importlib  # noqa: E402

from flask import Flask  # noqa: E402

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import job_bridge, media_bus
from hugpy_video.intel.runners import studio_i2v as S
from hugpy_video.intel.runners import studio_movie as SM
from hugpy_video.intel.runners.studio_movie import run_generate_studio_movie
from hugpy_video.intel.studio_movie_schema import (
    StudioMovieGoal,
    make_studio_movie,
    studio_movie_from_dict,
)

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None

_FPS = 12
_SEG_FRAMES = _FPS * 2   # 24 — the synthetic runner's fps*2 clip length
_W, _H = 320, 180

# --------------------------------------------------------------------------- #
# ISOLATION. Two things must move off the live box before anything runs:
#   * the media bus DB — enqueue / cancel / is_cancelling / set_progress must never
#     touch the running dev central's media_jobs.db;
#   * the STUDIO MOVIES ROOT — the session routes WALK it, and the real one holds
#     hundreds of the operator's movies. It is repointed at a temp dir UNDER
#     DEFAULT_ROOT, because a segment clip's playback url goes through /video/media,
#     whose jail is the storage roots: a root outside them would make the fetchability
#     check pass or fail for the wrong reason.
# The routes read ``STUDIO_MOVIE_ROOT`` off this module at CALL time (their studio
# imports are lazy, inside each function), so assigning the module attribute is enough.
# --------------------------------------------------------------------------- #
_TMP_DB_DIR = tempfile.mkdtemp(prefix="hugpy_movie_sessions_db_")
media_bus.DB_PATH = os.path.join(_TMP_DB_DIR, "media_jobs.db")
media_bus._initialized = False

_MOVIE_ROOT = tempfile.mkdtemp(prefix="movie-sessions-root-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
SM.STUDIO_MOVIE_ROOT = _MOVIE_ROOT

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


def _cleanup() -> None:
    shutil.rmtree(_TMP_DB_DIR, ignore_errors=True)
    shutil.rmtree(_MOVIE_ROOT, ignore_errors=True)


def _no_cancel():
    """Patch is_cancelling to False for the duration of a runner call; returns the
    original so the caller restores it (the runner reads the module attribute)."""
    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    return orig


def _submit(**body):
    """POST /video/studio/movie with sane geometry, returning (status, payload)."""
    payload = {"resolution": {"width": _W, "height": _H, "fps": _FPS},
               "vram_budget_gb": 0.5}
    payload.update(body)
    r = client.post("/video/studio/movie", json=payload)
    return r.status_code, r.get_json()


def _two_goals():
    return [{"prompt": "a lighthouse on a cliff"},
            {"prompt": "a storm rolls in", "branch_frame": 8}]


def _spec_sidecar(movie_id):
    with open(os.path.join(_MOVIE_ROOT, movie_id, "spec.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _manifest(movie_id):
    with open(os.path.join(_MOVIE_ROOT, movie_id, "movie.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _run_session(movie_id):
    """Rehydrate a session's persisted spec and run it under its CURRENT job id — the
    exact sequence a resume performs, so using it here means the checks exercise the
    real path rather than a test-only shortcut."""
    envelope = _spec_sidecar(movie_id)
    spec = studio_movie_from_dict(envelope["spec"])
    orig = _no_cancel()
    try:
        return run_generate_studio_movie(spec, job_id=envelope["job_id"])
    finally:
        media_bus.is_cancelling = orig


def _find(movies, movie_id):
    return next((m for m in movies if m["movie_id"] == movie_id), None)


# --------------------------------------------------------------------------- #
# [1] TITLE, end to end and OPTIONAL: POST -> spec.json -> rehydrated spec ->
#     movie.json. And an absent/blank title is None (UNNAMED), never "" or "   " —
#     the composer's warn-if-unnamed is a truthiness check, and a blank-but-present
#     title would defeat it while looking named in the session list.
# --------------------------------------------------------------------------- #
def test_title_round_trips_and_is_optional():
    st, body = _submit(goals=_two_goals(), title="  The Lighthouse Cut  ")
    assert st == 200, (st, body)
    movie_id = body["job_id"]

    env = _spec_sidecar(movie_id)
    assert env["job_id"] == movie_id, env
    assert env["spec"]["title"] == "The Lighthouse Cut", (
        f"the submitted title must be persisted (stripped); got {env['spec']['title']!r}")
    # ...and survives the rehydration a resume performs.
    assert studio_movie_from_dict(env["spec"]).title == "The Lighthouse Cut"

    if _FFMPEG and _FFPROBE:
        res = _run_session(movie_id)
        assert res.ok is True, f"the named movie must render; got {res.error}"
        assert _manifest(movie_id)["title"] == "The Lighthouse Cut", _manifest(movie_id)

    # UNNAMED (absent) and BLANK both normalize to None — backward compatible: every
    # pre-k91 caller omits the field and is unaffected.
    st2, body2 = _submit(goals=[{"prompt": "an unnamed take"}])
    assert st2 == 200, (st2, body2)
    assert _spec_sidecar(body2["job_id"])["spec"]["title"] is None

    st3, body3 = _submit(goals=[{"prompt": "a blank-named take"}], title="   ")
    assert st3 == 200, (st3, body3)
    assert _spec_sidecar(body3["job_id"])["spec"]["title"] is None, (
        "a whitespace-only title is not a name")


# --------------------------------------------------------------------------- #
# [2] HEARTBEAT — the runner's own half. Every movie progress tick carries a MONOTONIC
#     0..1 ``progress`` and a human ``message``; while a segment renders the message
#     counts DENOISE STEPS, so even a single 46-minute segment moves the number roughly
#     every poll instead of once per segment. Driven through a fake worker because the
#     step counter comes from the worker's own blob (the inline path has none).
# --------------------------------------------------------------------------- #
def _gray_clip(dst, n_frames, w, h, fps):
    """A solid-gray clip of exactly ``n_frames`` frames — the fake worker's render
    output. Built with ffmpeg's lavfi source so this suite needs no PIL."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=gray:s={w}x{h}:r={fps}",
         "-frames:v", str(n_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", dst],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


class _SteppingWorker:
    """A fake studio worker that reports a DENOISE STEP COUNTER while it renders — the
    signal the heartbeat message is built from. Renders a real clip on the shared store
    so the movie genuinely ingests and assembles it."""

    def __init__(self, steps=12):
        self.steps = steps
        self.renders = {}

    def post(self, url, payload, timeout):
        if url.endswith("/studio/render"):
            rid, spec = payload["job_id"], payload["spec"]
            clip = os.path.join(spec["out_root"], "_worker", "clip.mp4")
            _gray_clip(clip, _SEG_FRAMES, spec["width"], spec["height"], spec["fps"])
            self.renders[rid] = {"polls": 0, "done": {
                "ok": True, "path": clip, "content_hash": f"wc-{rid}",
                "frames": _SEG_FRAMES, "width": spec["width"], "height": spec["height"],
                "duration_s": _SEG_FRAMES / spec["fps"], "resumed": False}}
            return 202, {"ok": True, "accepted": "started", "pkg_version": S._pkg_version()}
        return 200, {"cancelled": True}

    def get(self, url, timeout):
        rid = url.rsplit("/", 1)[-1]
        r = self.renders.get(rid)
        if r is None:
            return 200, {"status": "unknown", "position": None, "progress": None}
        r["polls"] += 1
        if r["polls"] <= self.steps:
            return 200, {"status": "running", "progress": {
                "phase": "denoise", "step": r["polls"], "steps": self.steps}}
        return 200, {"status": "done", "result": r["done"]}


def test_movie_heartbeat_carries_progress_and_step_message():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="movie-heartbeat-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    fake = _SteppingWorker(steps=6)
    op, og = S._http_post_json, S._http_get_json
    S._http_post_json, S._http_get_json = fake.post, fake.get
    orig_cx, orig_sp = media_bus.is_cancelling, media_bus.set_progress
    media_bus.is_cancelling = lambda job_id: False
    blobs = []
    media_bus.set_progress = lambda jid, blob: blobs.append(blob)
    for k, v in (("HUGPY_STUDIO_WORKER", "http://worker.test"),
                 ("HUGPY_STUDIO_FORCE_REMOTE", "1"),
                 ("HUGPY_STUDIO_POLL_INTERVAL_S", "0.01")):
        os.environ[k] = v
    try:
        spec = make_studio_movie(
            goals=(StudioMovieGoal(segment_id="s0", prompt="dawn"),
                   StudioMovieGoal(segment_id="s1", prompt="dusk",
                                   parent_segment_id="s0", branch_frame=8)),
            width=_W, height=_H, fps=_FPS, vram_budget_gb=0.5, out_root=work)
        res = run_generate_studio_movie(spec, job_id="hb")
        assert res.ok is True, f"the delegated movie must render; got {res.error}"

        # EVERY tick carries both heartbeat fields — a tick that carries neither is a
        # tick job_bridge cannot turn into movement.
        assert blobs, "the movie must publish progress"
        for b in blobs:
            assert isinstance(b.get("progress"), float), f"no 0..1 progress in {b.get('stage')}"
            assert 0.0 <= b["progress"] <= 1.0, b["progress"]
            assert isinstance(b.get("message"), str) and b["message"], b

        # MONOTONIC — a number that goes backwards is not read as movement by the
        # JobStore's advance rule, i.e. it would stop heartbeating at the worst moment.
        fractions = [b["progress"] for b in blobs]
        assert all(a <= c for a, c in zip(fractions, fractions[1:])), fractions
        assert fractions[-1] == 1.0, f"a completed movie must end at 1.0; got {fractions[-1]}"

        # ...and WITHIN a segment the message counts denoise steps, which is the whole
        # point: one 46-minute segment must move the record many times, not once.
        step_msgs = [b["message"] for b in blobs if " - step " in b["message"]]
        assert step_msgs, f"no per-step heartbeat message; saw {[b['message'] for b in blobs]}"
        assert any(m.startswith("segment 1/2 - step ") for m in step_msgs), step_msgs
        assert any(m.startswith("segment 2/2 - step ") for m in step_msgs), step_msgs
        assert len(set(step_msgs)) > 2, (
            f"the message must CHANGE as steps advance (it is the movement signal); "
            f"got {sorted(set(step_msgs))}")
    finally:
        media_bus.is_cancelling, media_bus.set_progress = orig_cx, orig_sp
        S._http_post_json, S._http_get_json = op, og
        for k in ("HUGPY_STUDIO_WORKER", "HUGPY_STUDIO_FORCE_REMOTE",
                  "HUGPY_STUDIO_POLL_INTERVAL_S"):
            os.environ.pop(k, None)
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [3] HEARTBEAT — the BRIDGE half. ``job_bridge.on_progress`` relays a runner-authored
#     message into the comms Job AND stamps ``progressed_at`` when it CHANGES. That
#     stamp is the movement clock ``expire_pending_orphans`` ages its wedged decision
#     on, and the JobStore's own advance rule (status transition / numeric advance /
#     stage change) is blind to a job whose every tick happens inside ONE stage — which
#     is exactly how a live 14-segment movie got retired as wedged.
#
#     A REPEATED message must NOT count: "a wedged render still spews log lines" is the
#     guard that relaxation would have broken.
# --------------------------------------------------------------------------- #
class _FakeJob:
    def __init__(self):
        self.message = None
        self.progressed_at = 0.0

    def to_dict(self):
        return {"message": self.message, "progressed_at": self.progressed_at}


class _FakeStore:
    def __init__(self):
        self.job = _FakeJob()
        self.updates = []

    def get(self, job_id):
        return self.job

    def update(self, job_id, **changes):
        self.updates.append(changes)
        if "message" in changes:
            self.job.message = changes["message"]
        return self.job


def test_job_bridge_relays_message_and_marks_movement():
    store = _FakeStore()
    orig_store, orig_place = job_bridge._store, job_bridge._placement
    job_bridge._store = lambda: store
    job_bridge._placement = lambda *a, **k: None
    try:
        job_bridge.on_progress("mv1", {"stage": "generating", "progress": 0.5,
                                       "message": "segment 2/14 - step 18/32"})
        assert store.job.message == "segment 2/14 - step 18/32", store.updates
        first = store.job.progressed_at
        assert first > 0, "a NEW runner-authored message must stamp progressed_at"

        # The SAME message again is not movement (the wedged-render guard).
        job_bridge.on_progress("mv1", {"stage": "generating", "progress": 0.5,
                                       "message": "segment 2/14 - step 18/32"})
        assert store.job.progressed_at == first, (
            "a REPEATED message must not be read as forward progress")

        # A CHANGED message is.
        job_bridge.on_progress("mv1", {"stage": "generating", "progress": 0.5,
                                       "message": "segment 2/14 - step 19/32"})
        assert store.job.progressed_at > first, (
            "a CHANGED runner-authored message must stamp progressed_at")

        # The authored message WINS over the stage+percent summary the bridge derives —
        # that summary cannot say anything a one-stage job has not already said.
        assert store.job.message == "segment 2/14 - step 19/32"
    finally:
        job_bridge._store, job_bridge._placement = orig_store, orig_place


# --------------------------------------------------------------------------- #
# [4] SESSION LIST — GET /video/studio/movies projects every movie dir, and a completed
#     segment's clip is FETCHABLE OVER HTTP through the existing jailed /video/media
#     route (no new serving route: one jail, one place to get it wrong). This is what
#     makes segments watchable AS THEY LAND — the runner rewrites movie.json after
#     EVERY segment, so a poll mid-render already sees segment N while N+1 denoises.
# --------------------------------------------------------------------------- #
def test_movies_list_projects_sessions_and_serves_clips():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping)")
        return
    st, body = _submit(goals=_two_goals(), title="Session List Movie")
    assert st == 200, (st, body)
    movie_id = body["job_id"]
    res = _run_session(movie_id)
    assert res.ok is True, f"the movie must render; got {res.error}"

    r = client.get("/video/studio/movies")
    assert r.status_code == 200, (r.status_code, r.get_json())
    row = _find(r.get_json()["movies"], movie_id)
    assert row is not None, f"{movie_id} missing from the session list"

    assert row["title"] == "Session List Movie", row
    assert row["status"] == "done", row
    assert row["segments_total"] == 2 and row["segments_completed"] == 2, row
    assert row["job_id"] == movie_id, row
    assert row["resumable"] is False, "a finished movie is not waiting to be resumed"
    assert isinstance(row["updated"], float), row
    assert len(row["segments"]) == 2, row["segments"]

    # Per-segment status + clip availability, and the assembled movie's own url.
    for seg in row["segments"]:
        assert seg["status"] in ("done", "resumed"), seg
        assert seg["clip_available"] is True, seg
        assert seg["media"].startswith("/video/media?handle="), seg
        assert seg["frames"] == _SEG_FRAMES, seg
    assert row["movie"] and row["movie"].startswith("/video/media?handle="), row

    # THE FETCH: a completed segment's bytes really come back over HTTP, as video,
    # range-aware. A url that 404s would make "watchable as it lands" a claim, not a
    # feature.
    clip_r = client.get(row["segments"][0]["media"])
    assert clip_r.status_code == 200, (clip_r.status_code, clip_r.data[:200])
    assert clip_r.mimetype == "video/mp4", clip_r.mimetype
    assert len(clip_r.data) > 0, "the segment clip must have bytes"


# --------------------------------------------------------------------------- #
# [5] PAUSE — cancel the live job AND record "paused". Both halves matter: without the
#     cancel the work keeps burning the card; without the record a paused movie is
#     indistinguishable from one the operator gave up on, and "which of these is
#     waiting on me" is the question the session list exists to answer.
# --------------------------------------------------------------------------- #
def test_pause_cancels_and_marks_paused():
    st, body = _submit(goals=_two_goals(), title="Pausable")
    assert st == 200, (st, body)
    movie_id = body["job_id"]

    r = client.post(f"/video/studio/movie/{movie_id}/pause")
    assert r.status_code == 200, (r.status_code, r.get_json())
    out = r.get_json()
    assert out["ok"] is True and out["status"] == "paused", out
    assert out["cancelled"] is True, (
        f"a QUEUED movie job must actually be cancelled by pause; got {out}")
    assert out["job_id"] == movie_id, out
    # ...and the bus agrees the job is gone.
    assert media_bus.get(movie_id)["status"] == "cancelled", media_bus.get(movie_id)

    # PAUSED is written to the manifest even though this movie never rendered a
    # segment (no movie.json existed) — a movie paused during segment 0 is exactly the
    # run this feature exists for, and an invisible one would defeat the point.
    assert _manifest(movie_id)["status"] == "paused", _manifest(movie_id)

    row = _find(client.get("/video/studio/movies").get_json()["movies"], movie_id)
    assert row is not None and row["status"] == "paused", row
    assert row["resumable"] is True, "a paused session must be resumable"
    assert row["title"] == "Pausable", (
        "a session paused before its first segment must still show its name")


# --------------------------------------------------------------------------- #
# [6] RESUME — re-enqueue the persisted spec as a NEW job, and let content-addressed
#     reuse return every completed segment as resumed=True. THE CLIPS ARE THE
#     CHECKPOINT: there is no checkpoint format to desynchronize from what rendered.
# --------------------------------------------------------------------------- #
def test_resume_reenqueues_and_reuses_completed_segments():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping)")
        return
    st, body = _submit(goals=_two_goals(), title="Resumable")
    assert st == 200, (st, body)
    movie_id = body["job_id"]

    first = _run_session(movie_id)
    assert first.ok is True, f"first run must render; got {first.error}"
    assert all(s["resumed"] is False for s in first.movie["segments"]), first.movie

    # Park it, then resume.
    assert client.post(f"/video/studio/movie/{movie_id}/pause").status_code == 200

    r = client.post(f"/video/studio/movie/{movie_id}/resume")
    assert r.status_code == 200, (r.status_code, r.get_json())
    out = r.get_json()
    assert out["ok"] is True and out["status"] == "running", out
    assert out["previous_job_id"] == movie_id, out
    new_job = out["job_id"]
    assert new_job != movie_id, "a resume mints a NEW bus job"
    assert out["segments_total"] == 2, out
    assert media_bus.get(new_job)["status"] == "queued", media_bus.get(new_job)

    # The SESSION id is unchanged — keying these routes on the job id would have made
    # every resume a new, unrelated session.
    row = _find(client.get("/video/studio/movies").get_json()["movies"], movie_id)
    assert row is not None and row["job_id"] == new_job, row
    assert row["title"] == "Resumable", row

    # spec.json now points at the NEW job (so a subsequent pause cancels the job that
    # is actually running), and the persisted title/goals are untouched.
    env = _spec_sidecar(movie_id)
    assert env["job_id"] == new_job, env
    assert env["spec"]["title"] == "Resumable", env

    # THE PIN. Resume stamps ``session_id`` so the re-enqueued run resolves to THIS dir
    # rather than one named after its new job id. Without it an unnamed movie would
    # relocate on every resume, find none of its clips, and re-render the whole thing
    # while reporting success — a silent failure whose only symptom is the GPU bill.
    assert env["spec"]["session_id"] == movie_id, env["spec"]
    pinned = studio_movie_from_dict(env["spec"])
    assert SM.movie_root_for(pinned, "a-totally-different-job-id") == \
        os.path.join(_MOVIE_ROOT, movie_id), (
            "a pinned spec must resolve to the session dir under ANY job id")

    # Running the resumed job re-uses every completed segment — no re-render.
    second = _run_session(movie_id)
    assert second.ok is True, f"resumed run must succeed; got {second.error}"
    assert all(s["resumed"] is True for s in second.movie["segments"]), (
        f"every already-rendered segment must RESUME; got {second.movie['segments']}")
    assert _manifest(movie_id)["status"] == "done", _manifest(movie_id)


# --------------------------------------------------------------------------- #
# [7] REFUSALS — a session already IN FLIGHT is a 409 (two runners over one movie dir
#     would race on movie.json and interleave their segment records), and an unknown or
#     path-escaping id is a 404 from the same jail seam all three routes share.
# --------------------------------------------------------------------------- #
def test_resume_refuses_while_running_and_rejects_bad_ids():
    st, body = _submit(goals=_two_goals())
    assert st == 200, (st, body)
    movie_id = body["job_id"]   # freshly enqueued -> the job is QUEUED, i.e. in flight

    r = client.post(f"/video/studio/movie/{movie_id}/resume")
    assert r.status_code == 409, (r.status_code, r.get_json())
    assert "already running" in r.get_json()["error"], r.get_json()

    for bad in ("no-such-movie", "..", "%2e%2e%2fetc"):
        for verb in ("pause", "resume"):
            rr = client.post(f"/video/studio/movie/{bad}/{verb}")
            assert rr.status_code == 404, (bad, verb, rr.status_code, rr.get_json())

    # A dir with NO persisted spec (a session from before spec.json existed) cannot be
    # re-enqueued — an honest 409 that says so, not a 404 pretending it is gone.
    legacy = os.path.join(_MOVIE_ROOT, "legacy-no-spec")
    os.makedirs(legacy, exist_ok=True)
    rr = client.post("/video/studio/movie/legacy-no-spec/resume")
    assert rr.status_code == 409, (rr.status_code, rr.get_json())
    assert "no persisted spec.json" in rr.get_json()["error"], rr.get_json()

    # ``session_id`` becomes a DIRECTORY NAME, so the schema refuses anything that is
    # not a bare leaf — validated at construction, the one place the value can enter,
    # so no downstream code has to re-check a path it was handed.
    goals = (StudioMovieGoal(segment_id="s0", prompt="x"),)
    for bad in ("../escape", "a/b", "..", "", "   ", 7):
        try:
            make_studio_movie(goals=goals, width=_W, height=_H, fps=_FPS, session_id=bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"session_id={bad!r} must be rejected at construction")
    # ...and a legitimate leaf is accepted and survives the asdict -> json -> from_dict
    # round trip the bus and spec.json both put it through.
    ok = make_studio_movie(goals=goals, width=_W, height=_H, fps=_FPS,
                           session_id="my-session")
    assert ok.session_id == "my-session"
    assert studio_movie_from_dict(json.loads(json.dumps(asdict(ok)))).session_id == \
        "my-session"
    # A spec with NO session_id (every pre-k91 one, and every fresh submit) rehydrates
    # to None and derives its dir from the job id exactly as it always did.
    legacy_d = json.loads(json.dumps(asdict(ok)))
    legacy_d.pop("session_id")
    legacy_spec = studio_movie_from_dict(legacy_d)
    assert legacy_spec.session_id is None
    assert SM.movie_root_for(legacy_spec, "job-xyz") == \
        os.path.join(_MOVIE_ROOT, "job-xyz")


CHECKS = [
    ("title: optional, round-trips POST -> spec.json -> spec -> movie.json; blank = unnamed",
     test_title_round_trips_and_is_optional),
    ("heartbeat: every movie tick carries a monotonic 0..1 progress + a per-STEP message",
     test_movie_heartbeat_carries_progress_and_step_message),
    ("heartbeat: job_bridge relays the authored message + marks movement only on CHANGE",
     test_job_bridge_relays_message_and_marks_movement),
    ("sessions: GET /video/studio/movies projects counts/status/segments; clips fetch over HTTP",
     test_movies_list_projects_sessions_and_serves_clips),
    ("pause: cancels the live job AND records 'paused' (visible + resumable in the list)",
     test_pause_cancels_and_marks_paused),
    ("resume: re-enqueues the persisted spec as a NEW job; completed segments resume=True",
     test_resume_reenqueues_and_reuses_completed_segments),
    ("refusals: in-flight resume 409, unknown/traversal id 404, spec-less dir 409",
     test_resume_refuses_while_running_and_rejects_bad_ids),
]


def main() -> int:
    passed = 0
    failed = 0
    try:
        for i, (name, fn) in enumerate(CHECKS, 1):
            try:
                fn()
            except Exception as exc:  # surface EVERY divergence, not just the first
                failed += 1
                import traceback
                print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"[{i}] PASS  {name}")
    finally:
        _cleanup()
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
