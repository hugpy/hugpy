"""Studio MOVIE slice (B1) — the studio's ordered strip of clips conjoined at splice
points (an NLE row).

Locks the studio-movie slice as executable checks, in the same script style as
``test_studio_source_video.py`` / ``test_studio_prompt.py`` (plain python, ``__main__``
guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check FAILED,
every check independently run so a failing one never masks the rest). pytest is NOT
installed in this venv, so there are no fixtures.

Invariants under test:
  * SCHEMA — make_studio_movie builds a valid LINEAR take-tree; asdict/from_dict
    round-trips it; empty goals / negative branch_frame / a broken (non-linear)
    parent chain / duplicate segment_ids / a root-with-parent are rejected LOCALLY.
  * ROUTE — POST /video/studio/movie enqueues a valid job (auto-filling segment_id
    + the linear parent chain from a minimal body), and rejects a bad body / a
    jail-escaping start_image with a clean 4xx.
  * SYNTHETIC E2E (GPU-less) — a 2-goal movie whose SECOND goal branches from a MID
    frame of the first goal's synthetic clip: movie.mp4 is playable (ffprobe), the
    assembled frame count reflects the TRIM (parent shortened at the splice),
    movie.json records the per-joint {branch_frame, trim_frames}, the per-segment
    clip files stay WHOLE (non-destructive), and a re-run RESUMES every segment.
  * NULL BRANCH — a null branch_frame conditions on the parent's LAST frame, so the
    parent plays in FULL (no trim).
  * OUT-OF-RANGE — a branch_frame past the parent's real length is errors-as-data
    (JobResult(ok=False), never a crash).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_movie.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict

logging.disable(logging.INFO)  # silence the models_config registry chatter

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
# These checks exercise the INLINE / synthetic render + assembly path (no delegation).
# Clear any ambient studio-worker env so a movie whose segments bind a real model never
# tries to delegate to a live worker mid-test (this central runs with HUGPY_STUDIO_WORKER
# set); per-segment offload has its OWN suite (test_studio_movie_offload.py).
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
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners.studio_movie import run_generate_studio_movie
from hugpy_video.intel.studio_movie_schema import (
    StudioMovieGoal,
    make_studio_movie,
    studio_movie_from_dict,
)

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None

# Synthetic clip length is fps*2 frames (see studio/runners/synthetic._geometry),
# well under the synthetic model's 240 cap. fps=12 -> 24 frames per segment.
_FPS = 12
_SEG_FRAMES = _FPS * 2  # 24


# --------------------------------------------------------------------------- #
# Isolation: point the media bus at a TEMP DB so route enqueues + the runner's
# is_cancelling/set_progress never touch the real media_jobs.db.
# --------------------------------------------------------------------------- #
_TMP_DB = tempfile.mkstemp(prefix="studio-movie-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False  # force _ensure_db to re-init against the temp DB
with sqlite3.connect(_TMP_DB) as _c:
    _c.execute(
        "CREATE TABLE IF NOT EXISTS media_jobs ("
        "job_id TEXT PRIMARY KEY, name TEXT, status TEXT, spec_json TEXT, "
        "result_json TEXT, claim_token TEXT, created REAL, updated REAL, "
        "progress_json TEXT)")

vr = importlib.import_module("hugpy_server.app.routes.video_routes")
app = Flask(__name__)
app.register_blueprint(vr.video_bp)
client = app.test_client()


def _ffprobe_frames(path: str) -> int:
    """Count the DECODED frames in a video (the authoritative playability + length
    check). ``-count_frames`` decodes; ``csv=p=0`` prints the bare integer."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(out.stdout.strip())


def _movie_spec(goals, project=None, out_root=None, seed=0):
    return make_studio_movie(
        goals=tuple(goals), width=320, height=180, fps=_FPS,
        vram_budget_gb=0.5, seed=seed, project=project, out_root=out_root)


# --------------------------------------------------------------------------- #
# [1] Schema: a valid linear take-tree builds + round-trips; the load-bearing
#     invariants are rejected LOCALLY (never carried across the bus).
# --------------------------------------------------------------------------- #
def test_schema_builds_roundtrips_and_rejects():
    goals = (StudioMovieGoal(segment_id="s0", prompt="a lighthouse"),
             StudioMovieGoal(segment_id="s1", prompt="a storm",
                             parent_segment_id="s0", branch_frame=10))
    spec = _movie_spec(goals)
    assert len(spec.goals) == 2 and spec.goals[1].branch_frame == 10
    # asdict -> json -> from_dict round-trips to an EQUAL frozen spec.
    d = json.loads(json.dumps(asdict(spec)))
    assert studio_movie_from_dict(d) == spec, "spec must round-trip through JSON"

    rejects = {
        "empty goals": dict(goals=()),
        "root has parent": dict(goals=(StudioMovieGoal("s0", "a", parent_segment_id="x"),)),
        "negative branch": dict(goals=(StudioMovieGoal("s0", "a", branch_frame=-1),)),
        "duplicate ids": dict(goals=(StudioMovieGoal("s0", "a"),
                                     StudioMovieGoal("s0", "b", parent_segment_id="s0"))),
        "broken chain": dict(goals=(StudioMovieGoal("s0", "a"),
                                    StudioMovieGoal("s1", "b", parent_segment_id="sX"))),
        "blank prompt": dict(goals=(StudioMovieGoal("s0", "   "),)),
        "bad steps": dict(goals=(StudioMovieGoal("s0", "a", steps=999),)),
    }
    for desc, kw in rejects.items():
        try:
            make_studio_movie(width=320, height=180, fps=_FPS, **kw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"make_studio_movie must reject: {desc}")


# --------------------------------------------------------------------------- #
# [2] Route: a minimal 2-goal body (no segment_ids/parents) auto-forms a valid
#     linear chain -> 200 {job_id}; a bad body -> 400.
# --------------------------------------------------------------------------- #
def test_route_minimal_body_enqueues_200():
    r = client.post("/video/studio/movie", json={
        "resolution": {"width": 320, "height": 180, "fps": _FPS},
        "goals": [{"prompt": "a lighthouse"},
                  {"prompt": "a storm", "branch_frame": 10}]})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()

    # empty goals -> 400
    r2 = client.post("/video/studio/movie", json={"goals": []})
    assert r2.status_code == 400, (r2.status_code, r2.get_json())
    # a non-linear parent chain -> 400 (factory re-validates the explicit parent)
    r3 = client.post("/video/studio/movie", json={"goals": [
        {"segment_id": "a", "prompt": "x"},
        {"segment_id": "b", "prompt": "y", "parent_segment_id": "ghost"}]})
    assert r3.status_code == 400, (r3.status_code, r3.get_json())


# --------------------------------------------------------------------------- #
# [3] Route: a start_image that ESCAPES the storage jail -> 400 (never an
#     arbitrary-file read).
# --------------------------------------------------------------------------- #
def test_route_start_image_jail_escape_400():
    r = client.post("/video/studio/movie", json={
        "goals": [{"prompt": "a lighthouse"}],
        "start_image": "/etc/passwd"})
    assert r.status_code == 400, (r.status_code, r.get_json())


# --------------------------------------------------------------------------- #
# [4] Synthetic E2E: 2 goals, goal 2 branches from a MID frame (10) of goal 1's
#     synthetic clip. movie.mp4 is playable; the assembled length reflects the
#     TRIM; movie.json records the joint; the per-segment clips stay WHOLE.
# --------------------------------------------------------------------------- #
def test_e2e_mid_branch_trim():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping synthetic E2E)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-e2e-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a lighthouse on a cliff"),
                 StudioMovieGoal(segment_id="s1", prompt="a storm rolls in",
                                 parent_segment_id="s0", branch_frame=10))
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="e2e-mid")
        assert res.ok is True, f"movie job must be ok=True; got {res.error}"
        assert res.error is None

        movie_root = os.path.join(work, "e2e-mid")
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        assert os.path.isfile(movie_mp4) and os.path.getsize(movie_mp4) > 0, "movie.mp4 non-empty"

        # playable + TRIM-honored length: 11 (parent up to frame 10) + 24 (full leaf).
        expected = (10 + 1) + _SEG_FRAMES
        untrimmed = _SEG_FRAMES + _SEG_FRAMES
        frames = _ffprobe_frames(movie_mp4)
        assert frames == expected, f"assembled movie must be {expected} frames; got {frames}"
        assert frames < untrimmed, "the parent must be SHORTENED at the splice (trim honored)"

        # movie.json records the joint {branch_frame, trim_frames} + the node list.
        with open(os.path.join(movie_root, "movie.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        assert len(man["segments"]) == 2, man["segments"]
        joint = man["joints"][0]
        assert joint["branch_frame"] == 10 and joint["trim_frames"] == 11, joint
        assert joint["parent_segment_id"] == "s0" and joint["child_segment_id"] == "s1"
        assert man["assembly"]["total_frames"] == expected, man["assembly"]
        assert "VACE-extend" in man["drift"], "drift note must record the motion-not-carried caveat"

        # NON-DESTRUCTIVE: the per-segment clip files are WHOLE (24 frames each) —
        # the trim is metadata, never a re-render of the source clip.
        for rec in man["segments"]:
            assert _ffprobe_frames(rec["clip_path"]) == _SEG_FRAMES, (
                f"segment {rec['segment_id']} clip must stay whole ({_SEG_FRAMES} frames)")

        # outputs: 2 per-segment clip refs + the final movie ref (video, last).
        assert len(res.outputs) == 3, res.outputs
        assert res.outputs[-1].kind == "video", res.outputs[-1]
        assert res.outputs[-1].uri == movie_mp4, res.outputs[-1].uri
    finally:
        media_bus.is_cancelling = orig
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [5] Resume: re-running the SAME job (same out_root) RESUMES every segment
#     (content-addressed produce_clip), and still assembles the same movie.
# --------------------------------------------------------------------------- #
def test_e2e_resume():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping resume check)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-resume-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a quiet harbor"),
                 StudioMovieGoal(segment_id="s1", prompt="fireworks",
                                 parent_segment_id="s0", branch_frame=8))
        spec = _movie_spec(goals, out_root=work)
        r1 = run_generate_studio_movie(spec, job_id="e2e-resume")
        assert r1.ok, r1.error
        assert all(s["resumed"] is False for s in r1.movie["segments"]), (
            "first run must RENDER (not resume) every segment")

        r2 = run_generate_studio_movie(spec, job_id="e2e-resume")
        assert r2.ok, r2.error
        assert all(s["resumed"] is True for s in r2.movie["segments"]), (
            "a re-run must RESUME every segment (content-addressed reuse)")
        # same assembled length after resume (8+1 + 24 = 33).
        movie_mp4 = os.path.join(work, "e2e-resume", "movie.mp4")
        assert _ffprobe_frames(movie_mp4) == (8 + 1) + _SEG_FRAMES
    finally:
        media_bus.is_cancelling = orig
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [6] Null branch: branch_frame=None conditions on the parent's LAST frame, so the
#     parent plays in FULL (no trim) — total = 24 + 24 = 48.
# --------------------------------------------------------------------------- #
def test_e2e_null_branch_plays_full():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping null-branch check)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-null-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a sunrise"),
                 StudioMovieGoal(segment_id="s1", prompt="a sunset",
                                 parent_segment_id="s0", branch_frame=None))
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="e2e-null")
        assert res.ok, res.error
        movie_mp4 = os.path.join(work, "e2e-null", "movie.mp4")
        assert _ffprobe_frames(movie_mp4) == _SEG_FRAMES + _SEG_FRAMES, (
            "a null branch (last frame) must let the parent play in FULL")
        joint = res.movie["joints"][0]
        assert joint["branch_frame"] == _SEG_FRAMES - 1, joint  # resolved to last index
        assert joint["trim_frames"] == _SEG_FRAMES, joint
    finally:
        media_bus.is_cancelling = orig
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [7] Out-of-range branch: a branch_frame past the parent's real length is
#     errors-as-data (JobResult(ok=False), never a crash).
# --------------------------------------------------------------------------- #
def test_e2e_branch_out_of_range():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping out-of-range check)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-oor-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a field"),
                 StudioMovieGoal(segment_id="s1", prompt="a forest",
                                 parent_segment_id="s0", branch_frame=999))
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="e2e-oor")
        assert res.ok is False, "a branch past the parent's frames must fail as DATA"
        assert res.error is not None and res.error.code == "branch_frame_out_of_range", res.error
    finally:
        media_bus.is_cancelling = orig
        shutil.rmtree(work, ignore_errors=True)


CHECKS = [
    ("schema: build + json round-trip + reject empty/neg-branch/broken-chain/dup-id/root-parent",
     test_schema_builds_roundtrips_and_rejects),
    ("route: minimal 2-goal body auto-chains -> 200; empty/broken -> 400",
     test_route_minimal_body_enqueues_200),
    ("route: start_image escaping the jail -> 400",
     test_route_start_image_jail_escape_400),
    ("e2e: mid-branch trim — movie playable, length reflects TRIM, movie.json joint, clips whole",
     test_e2e_mid_branch_trim),
    ("e2e: re-run RESUMES every segment (content-addressed reuse)",
     test_e2e_resume),
    ("e2e: null branch conditions on the LAST frame -> parent plays FULL",
     test_e2e_null_branch_plays_full),
    ("e2e: branch_frame past the parent's length -> errors-as-data (ok=False)",
     test_e2e_branch_out_of_range),
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
                print(f"[{i}] FAIL  {name}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
                print(f"[{i}] PASS  {name}")
    finally:
        try:
            os.remove(_TMP_DB)
        except OSError:
            pass
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
