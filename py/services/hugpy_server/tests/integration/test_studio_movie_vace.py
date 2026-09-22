"""Studio-movie VACE-EXTEND joint mode (splice motion-carry).

Locks the ``joint_mode="vace_extend"`` upgrade as executable checks, in the same
script style as ``test_studio_movie.py`` (plain python, ``__main__`` guard, numbered
``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit iff any check FAILED, every check
run independently so a failing one never masks the rest). pytest is NOT installed in
this venv, so there are no fixtures.

Invariants under test:
  * SCHEMA — a vace_extend joint (on a NON-root goal) builds + round-trips; goal-0
    vace_extend, a bad joint_mode, and an out-of-range context_frames (movie-level +
    per-node) are rejected LOCALLY.
  * ROUTE — POST /video/studio/movie accepts joint_mode + context_frames; a goal-0
    vace_extend / a bad context_frames is a clean 400.
  * RUNNER (FAKE VACE SEAM) — with a numbered synthetic parent clip and the VACE
    render seam faked to a controlled clip:
      (a) the trailing-frames extraction picks EXACTLY [branch-K+1 .. branch];
      (b) assembly DROPS the child's re-rendered context overlap (no double-play) —
          total = (branch+1) + (child_frames - K);
      (c) movie.json labels the splice mode="vace_extend" + context_frames, and the
          vace segment carries context_drop=K.
  * RUNNER (REAL, GPU-LESS) — a real vace_extend movie on THIS box renders segment 0
    (synthetic) then fails segment 1 with the VACE runner's GRACEFUL Err
    (DEPS_MISSING), surfaced as per-segment DATA that names WHICH segment + mode —
    never a hang/raise, never a silent fallback to still-mode.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_movie_vace.py
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
# These checks exercise the INLINE render (fake VACE seam / real GPU-less degrade) + the
# assembly path — never delegation. Clear any ambient studio-worker env so a vace_extend
# segment (real budget) never delegates to a live worker mid-test (this central runs with
# HUGPY_STUDIO_WORKER set); per-segment offload has its OWN suite (test_studio_movie_offload.py).
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
from PIL import Image  # noqa: E402

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import studio_movie
from hugpy_video.intel.runners.studio_movie import run_generate_studio_movie
from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.errors import Ok
from hugpy_video.intel.studio_movie_schema import (
    StudioMovieGoal,
    make_studio_movie,
    studio_movie_from_dict,
)

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None

_FPS = 12
_SEG_FRAMES = _FPS * 2  # 24 synthetic frames per segment


# --------------------------------------------------------------------------- #
# Isolation: point the media bus at a TEMP DB so route enqueues + the runner's
# is_cancelling/set_progress never touch the real media_jobs.db.
# --------------------------------------------------------------------------- #
_TMP_DB = tempfile.mkstemp(prefix="studio-movie-vace-bus-", suffix=".db")[1]
media_bus.DB_PATH = _TMP_DB
media_bus._initialized = False
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
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(out.stdout.strip())


def _build_numbered_clip(dst: str, n_frames: int, w: int, h: int, fps: int) -> None:
    """A clip whose frame N is a SOLID gray of luma 10 + N*9 (each frame unique) — so
    an extracted frame's gray recovers its source INDEX (proves the extractor's
    indices, not just 'some frames')."""
    fdir = tempfile.mkdtemp(prefix=".numframes-", dir=os.path.dirname(dst))
    try:
        for n in range(n_frames):
            v = 10 + n * 9
            Image.new("RGB", (w, h), (v, v, v)).save(os.path.join(fdir, f"f_{n:04d}.png"))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(fdir, "f_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    finally:
        shutil.rmtree(fdir, ignore_errors=True)


def _build_solid_clip(dst: str, n_frames: int, w: int, h: int, fps: int, gray: int) -> None:
    """A solid-gray clip of exactly ``n_frames`` frames (the faked VACE child output)."""
    fdir = tempfile.mkdtemp(prefix=".solidframes-", dir=os.path.dirname(dst))
    try:
        for n in range(n_frames):
            Image.new("RGB", (w, h), (gray, gray, gray)).save(
                os.path.join(fdir, f"f_{n:04d}.png"))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(fdir, "f_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    finally:
        shutil.rmtree(fdir, ignore_errors=True)


def _png_gray(path: str) -> int:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.getpixel((w // 2, h // 2))[0]


def _movie_spec(goals, out_root=None, vram=0.5, context_frames=8):
    return make_studio_movie(
        goals=tuple(goals), width=320, height=180, fps=_FPS,
        vram_budget_gb=vram, seed=0, out_root=out_root, context_frames=context_frames)


# --------------------------------------------------------------------------- #
# [1] Schema: a vace_extend joint on a non-root goal builds + round-trips; the
#     load-bearing new invariants are rejected LOCALLY.
# --------------------------------------------------------------------------- #
def test_schema_vace_builds_and_rejects():
    goals = (StudioMovieGoal(segment_id="s0", prompt="a lighthouse"),
             StudioMovieGoal(segment_id="s1", prompt="a storm", parent_segment_id="s0",
                             branch_frame=10, joint_mode="vace_extend", context_frames=6))
    spec = _movie_spec(goals)
    assert spec.goals[1].joint_mode == "vace_extend"
    assert spec.goals[1].context_frames == 6
    assert spec.context_frames == 8
    # asdict -> json -> from_dict round-trips to an EQUAL frozen spec.
    d = json.loads(json.dumps(asdict(spec)))
    assert studio_movie_from_dict(d) == spec, "vace spec must round-trip through JSON"

    # a per-goal context_frames=None inherits the movie-level default (no crash).
    inherit = _movie_spec(
        (StudioMovieGoal("s0", "a"),
         StudioMovieGoal("s1", "b", parent_segment_id="s0", joint_mode="vace_extend")),
        context_frames=12)
    assert inherit.goals[1].context_frames is None and inherit.context_frames == 12

    rejects = {
        "root vace_extend": dict(goals=(
            StudioMovieGoal("s0", "a", joint_mode="vace_extend"),)),
        "bad joint_mode": dict(goals=(
            StudioMovieGoal("s0", "a"),
            StudioMovieGoal("s1", "b", parent_segment_id="s0", joint_mode="morph"))),
        "per-node ctx too big": dict(goals=(
            StudioMovieGoal("s0", "a"),
            StudioMovieGoal("s1", "b", parent_segment_id="s0",
                            joint_mode="vace_extend", context_frames=999))),
        "per-node ctx zero": dict(goals=(
            StudioMovieGoal("s0", "a"),
            StudioMovieGoal("s1", "b", parent_segment_id="s0", context_frames=0))),
    }
    for desc, kw in rejects.items():
        try:
            make_studio_movie(width=320, height=180, fps=_FPS, **kw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"make_studio_movie must reject: {desc}")

    # movie-level context_frames out of range is rejected too.
    for bad in (0, 33, 8.5, True):
        try:
            make_studio_movie(goals=(StudioMovieGoal("s0", "a"),),
                              width=320, height=180, fps=_FPS, context_frames=bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"movie-level context_frames must reject: {bad!r}")


# --------------------------------------------------------------------------- #
# [2] Route: a vace_extend body enqueues 200; a goal-0 vace_extend / bad
#     context_frames -> 400.
# --------------------------------------------------------------------------- #
def test_route_vace_body():
    r = client.post("/video/studio/movie", json={
        "resolution": {"width": 320, "height": 180, "fps": _FPS},
        "context_frames": 8,
        "goals": [{"prompt": "a lighthouse"},
                  {"prompt": "a storm", "branch_frame": 10,
                   "joint_mode": "vace_extend", "context_frames": 6}]})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()

    # goal 0 vace_extend -> 400 (root has no parent to extend from)
    r2 = client.post("/video/studio/movie", json={
        "goals": [{"prompt": "x", "joint_mode": "vace_extend"}]})
    assert r2.status_code == 400, (r2.status_code, r2.get_json())

    # out-of-range movie-level context_frames -> 400
    r3 = client.post("/video/studio/movie", json={
        "context_frames": 999,
        "goals": [{"prompt": "a"}, {"prompt": "b", "branch_frame": 5}]})
    assert r3.status_code == 400, (r3.status_code, r3.get_json())


# --------------------------------------------------------------------------- #
# [3] Runner (FAKE VACE SEAM): numbered parent + faked VACE child. Prove exact
#     context indices, overlap-drop assembly, and honest movie.json labels.
# --------------------------------------------------------------------------- #
def test_runner_extract_indices_and_overlap_drop():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping fake-seam runner check)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-vace-fake-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    F_PARENT, F_CHILD, BRANCH, K = 24, 20, 15, 8
    parent_clip = os.path.join(work, "parent_numbered.mp4")
    child_clip = os.path.join(work, "vace_child.mp4")
    _build_numbered_clip(parent_clip, F_PARENT, 320, 180, _FPS)
    _build_solid_clip(child_clip, F_CHILD, 320, 180, _FPS, gray=200)

    orig_rpc = studio_movie.run_produce_clip
    orig_cancel = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False

    def fake_rpc(spec, should_cancel):
        # The VACE segment carries vace_context_frames -> return the faked child clip;
        # segment 0 (no context) -> return the numbered parent clip. (We fake BOTH so the
        # parent's per-frame INDEX is known, letting the extraction test be exact.)
        if getattr(spec, "vace_context_frames", ()):
            return Ok(Artifact(path=child_clip, content_hash="fakechild",
                               frames=F_CHILD, width=320, height=180,
                               duration_s=F_CHILD / float(_FPS), resumed=False))
        return Ok(Artifact(path=parent_clip, content_hash="fakeparent",
                           frames=F_PARENT, width=320, height=180,
                           duration_s=F_PARENT / float(_FPS), resumed=False))

    studio_movie.run_produce_clip = fake_rpc
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a cliff"),
                 StudioMovieGoal(segment_id="s1", prompt="a storm rolls in",
                                 parent_segment_id="s0", branch_frame=BRANCH,
                                 joint_mode="vace_extend", context_frames=K))
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="vace-fake")
        assert res.ok is True, f"faked vace movie must be ok=True; got {res.error}"

        movie_root = os.path.join(work, "vace-fake")

        # (a) EXACT context indices: segment_01/context/ctx_000..ctx_007 must be the
        #     parent's frames [BRANCH-K+1 .. BRANCH] = [8..15]. Each numbered frame N has
        #     a UNIQUE gray (10+N*9), so the recovered grays prove the indices + order.
        ctx_dir = os.path.join(movie_root, "segment_01", "context")
        ctx_pngs = sorted(f for f in os.listdir(ctx_dir) if f.startswith("ctx_"))
        assert len(ctx_pngs) == K, f"must extract exactly K={K} context frames; got {ctx_pngs}"
        grays = [_png_gray(os.path.join(ctx_dir, p)) for p in ctx_pngs]
        expected_idx = list(range(BRANCH - K + 1, BRANCH + 1))   # [8..15]
        expected_gray = [10 + n * 9 for n in expected_idx]
        assert grays == sorted(grays), f"context frames must be oldest->newest; got {grays}"
        assert all(abs(g - e) <= 8 for g, e in zip(grays, expected_gray)), (
            f"extracted grays {grays} must match parent frames {expected_idx} "
            f"(expected ~{expected_gray})")

        # (b) OVERLAP DROP: the child's first K frames reconstruct the context and are
        #     dropped. total = (BRANCH+1) parent + (F_CHILD - K) child = 16 + 12 = 28.
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        expected_total = (BRANCH + 1) + (F_CHILD - K)
        got = _ffprobe_frames(movie_mp4)
        assert got == expected_total, (
            f"assembled movie must be {expected_total} frames (context overlap dropped); "
            f"got {got}")
        # sanity: WITHOUT the drop it would have been 16 + 20 = 36 (frames double-played).
        assert got < (BRANCH + 1) + F_CHILD, "the K context frames must NOT double-play"

        # (c) HONEST metadata: the joint labels the splice + kept-context length, and the
        #     vace segment carries context_drop=K.
        with open(os.path.join(movie_root, "movie.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        joint = man["joints"][0]
        assert joint["mode"] == "vace_extend", joint
        assert joint["context_frames"] == K, joint
        assert joint["branch_frame"] == BRANCH and joint["trim_frames"] == BRANCH + 1, joint
        seg1 = next(s for s in man["segments"] if s["segment_id"] == "s1")
        assert seg1["joint_mode"] == "vace_extend", seg1
        assert seg1["context_frames"] == K and seg1["context_drop"] == K, seg1
        assert seg1["capability"] == "v2v", seg1
        assert man["assembly"]["total_frames"] == expected_total, man["assembly"]
    finally:
        studio_movie.run_produce_clip = orig_rpc
        media_bus.is_cancelling = orig_cancel
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [4] Runner (REAL, GPU-LESS): a real vace_extend movie renders segment 0
#     (synthetic) then fails segment 1 with the VACE runner's GRACEFUL Err
#     (DEPS_MISSING), surfaced as per-segment DATA naming the segment + mode. No
#     monkeypatch — proves the extract -> route-to-VACE -> graceful-degrade path.
# --------------------------------------------------------------------------- #
def test_runner_real_vace_graceful_err():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping real graceful-Err check)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-vace-real-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    orig_cancel = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a quiet harbor"),
                 StudioMovieGoal(segment_id="s1", prompt="the tide surges",
                                 parent_segment_id="s0", branch_frame=None,
                                 joint_mode="vace_extend"))
        spec = _movie_spec(goals, out_root=work, vram=0.5)   # movie tier stays synthetic
        res = run_generate_studio_movie(spec, job_id="vace-real")

        # honest per-segment Err — NOT a hang/raise/500, NOT a silent still fallback.
        assert res.ok is False, "a vace_extend segment that can't run must fail as DATA"
        assert res.error is not None
        assert res.error.code == "deps_missing", (
            f"this GPU-less box must surface the VACE runner's graceful Err; got "
            f"{res.error.code}: {res.error.message}")
        # the JobError names WHICH segment + mode failed.
        assert "segment 1" in res.error.message and "joint_mode=vace_extend" in res.error.message, (
            f"error must name the failing segment + joint mode; got {res.error.message!r}")

        # movie.json: segment 0 rendered (synthetic), segment 1 recorded FAILED + its mode.
        with open(os.path.join(work, "vace-real", "movie.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        assert man["segments_completed"] == 1, man
        s0 = next(s for s in man["segments"] if s["segment_id"] == "s0")
        s1 = next(s for s in man["segments"] if s["segment_id"] == "s1")
        assert s0["status"] in ("done", "resumed") and s0["joint_mode"] == "still", s0
        assert s1["status"] == "failed" and s1["joint_mode"] == "vace_extend", s1
        assert s1["capability"] == "v2v", s1
        assert s1["error"]["code"] == "deps_missing", s1
        # the vace segment's budget was BUMPED to reach the VACE model (not left at 0.5).
        assert s1["vram_budget_gb"] >= 6.0, s1
        return res  # surfaced for the smoke printout
    finally:
        media_bus.is_cancelling = orig_cancel
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [5] Regression: an all-STILL 2-goal movie under the vace-aware code still
#     assembles with mode="still" joints + context_drop=0 (today's behavior).
# --------------------------------------------------------------------------- #
def test_still_mode_labeled_and_unchanged():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping still-mode regression)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-vace-still-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    orig_cancel = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a sunrise"),
                 StudioMovieGoal(segment_id="s1", prompt="a sunset",
                                 parent_segment_id="s0", branch_frame=10))
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="vace-still")
        assert res.ok is True, res.error
        # trim honored, no drop: 11 (parent up to 10) + 24 (full leaf) = 35.
        movie_mp4 = os.path.join(work, "vace-still", "movie.mp4")
        assert _ffprobe_frames(movie_mp4) == (10 + 1) + _SEG_FRAMES
        joint = res.movie["joints"][0]
        assert joint["mode"] == "still" and joint["context_frames"] is None, joint
        for s in res.movie["segments"]:
            assert s["joint_mode"] == "still" and s["context_drop"] == 0, s
    finally:
        media_bus.is_cancelling = orig_cancel
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [6] Installed-API idiom: the VACE runner's video+mask EXTEND channels have the
#     right SHAPE (grounded in diffusers 0.39 prepare_video_latents semantics:
#     black mask=keep/context, white mask=generate). Pure/PIL — no GPU needed.
# --------------------------------------------------------------------------- #
def test_extend_channels_shape():
    from hugpy_video.intel.studio.runners.wan_vace import _build_vace_extend_channels
    W, H, N = 32, 16, 8
    ctx = [Image.new("RGB", (W, H), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    K = len(ctx)
    video, mask = _build_vace_extend_channels(ctx, N, W, H)
    # both channels are exactly num_frames long.
    assert len(video) == N and len(mask) == N, (len(video), len(mask))
    # video: the K kept context frames are the PREFIX (identity-preserved), the rest black.
    for i in range(K):
        assert video[i] is ctx[i], f"video[{i}] must be the kept context frame"
    for i in range(K, N):
        assert video[i].getpixel((W // 2, H // 2)) == (0, 0, 0), f"video[{i}] must be black"
    # mask: black(keep) over the context, white(generate) over the rest.
    for i in range(K):
        assert mask[i].getpixel((W // 2, H // 2)) == (0, 0, 0), (
            f"mask[{i}] must be BLACK (keep the context — diffusers mask=0 'inactive')")
    for i in range(K, N):
        assert mask[i].getpixel((W // 2, H // 2)) == (255, 255, 255), (
            f"mask[{i}] must be WHITE (generate — diffusers mask=1 'reactive')")


CHECKS = [
    ("schema: vace_extend builds + round-trips; goal-0/bad-mode/ctx-range rejected",
     test_schema_vace_builds_and_rejects),
    ("route: vace_extend body -> 200; goal-0 vace_extend / bad context_frames -> 400",
     test_route_vace_body),
    ("runner: fake VACE seam — exact context indices + overlap-drop + honest movie.json",
     test_runner_extract_indices_and_overlap_drop),
    ("runner: REAL vace_extend on this box -> segment 0 renders, segment 1 graceful Err (mode named)",
     test_runner_real_vace_graceful_err),
    ("regression: all-still movie labeled mode=still, context_drop=0, unchanged length",
     test_still_mode_labeled_and_unchanged),
    ("installed-API: VACE video+mask extend channels shape (black=keep, white=generate)",
     test_extend_channels_shape),
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
