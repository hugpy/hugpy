"""Studio-movie IDENTITY LOCK (movie-level reference_images) + SCENE CUT joint.

Locks the id-movie + "cut" slice as executable checks, in the same script style as
``test_studio_movie_vace.py`` (plain python, ``__main__`` guard, numbered ``[n] PASS`` /
``[n] FAIL`` lines, nonzero exit iff any check FAILED, every check run independently so a
failing one never masks the rest). pytest is NOT installed in this venv, so no fixtures.

The operator's ask: "take that id [a locked character reference] and use it for a video —
her on the beach, then playing volleyball." A MULTI-SCENE movie where the IDENTITY carries
across scene changes: movie-level ``reference_images`` -> EVERY segment renders capability
id_lock (Wan-VACE reference-to-video); a ``joint_mode="cut"`` is a HARD scene cut (no frame
carry, parent plays FULL).

Invariants under test:
  * SCHEMA — reference_images validation; a "cut" node rejects branch_frame/context_frames;
    goal-0 cut is rejected (root MUST be "still").
  * ROUTE — POST /video/studio/movie accepts reference_images + a cut joint (200); a cut
    with a branch_frame, a jail-escaping / non-image / >4 reference, all 400/404.
  * RUNNER (FAKE RENDER SEAM) — with the produce_clip seam faked to controlled clips:
      (a) an id-movie passes the movie references + capability id_lock + the bumped VACE
          budget on EVERY segment spec;
      (b) a "cut" joint extracts NO frame (no branch.png / context dir), the parent plays
          FULL, and movie.json records the joint {mode:"cut", branch_frame:null,
          trim_frames = parent's full length};
      (c) an id-movie "still" joint passes BOTH the branch still AND the references (the
          documented precedence — references win the render; the still governs the trim).
  * ASSEMBLY MATH (REAL SYNTHETIC, done/ok) — a PLAIN 2-goal movie (no references) whose
    second goal is a "cut" renders synthetically end-to-end: total frames = parent FULL +
    child FULL (a cut trims nothing), movie.json labels the joint mode="cut".
  * RUNNER (REAL id-movie, GPU-LESS) — an id-movie on THIS box fails segment 0 with the
    VACE path's GRACEFUL Err (DEPS_MISSING), surfaced as per-segment DATA naming the
    segment + capability id_lock — never a hang/raise, never a synthetic fallback (there
    is no synthetic id_lock tier by design).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_movie_id_cut.py
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
# These checks exercise the INLINE render (fake render seam / real GPU-less degrade) + the
# assembly path — never delegation. Clear any ambient studio-worker env so an id-movie
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

try:
    from PIL import Image  # noqa: E402
    _PIL = True
except Exception:  # noqa: BLE001
    _PIL = False

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import studio_movie
from hugpy_video.intel.runners.studio_movie import run_generate_studio_movie, _VACE_MIN_BUDGET_GB
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
_TMP_DB = tempfile.mkstemp(prefix="studio-movie-idcut-bus-", suffix=".db")[1]
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

# In-jail reference PNGs (under DEFAULT_ROOT so the route's jail-resolve accepts them).
_WORK_REFS = tempfile.mkdtemp(prefix="studio-idcut-refs-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
_REF_PNG = os.path.join(_WORK_REFS, "subject.png")
_REF_PNG2 = os.path.join(_WORK_REFS, "subject2.png")
if _PIL:
    Image.new("RGB", (96, 96), (200, 120, 60)).save(_REF_PNG, "PNG")
    Image.new("RGB", (96, 96), (60, 120, 200)).save(_REF_PNG2, "PNG")


def _ffprobe_frames(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(out.stdout.strip())


def _build_solid_clip(dst: str, n_frames: int, w: int, h: int, fps: int, gray: int) -> None:
    """A solid-gray clip of exactly ``n_frames`` frames (a faked render output)."""
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


def _movie_spec(goals, out_root=None, vram=0.5, reference_images=None):
    return make_studio_movie(
        goals=tuple(goals), width=320, height=180, fps=_FPS,
        vram_budget_gb=vram, seed=0, out_root=out_root, reference_images=reference_images)


# --------------------------------------------------------------------------- #
# [1] Schema: reference_images validation; cut rejects branch_frame/context_frames;
#     goal-0 cut rejected; round-trip.
# --------------------------------------------------------------------------- #
def test_schema_refs_and_cut():
    goals = (StudioMovieGoal("s0", "on a sunny beach"),
             StudioMovieGoal("s1", "playing volleyball",
                             parent_segment_id="s0", joint_mode="cut"))
    spec = _movie_spec(goals, reference_images=("/a.png", "/b.png"))
    assert spec.reference_images == ("/a.png", "/b.png")
    assert spec.goals[1].joint_mode == "cut"
    # asdict -> json -> from_dict round-trips to an EQUAL frozen spec.
    d = json.loads(json.dumps(asdict(spec)))
    assert studio_movie_from_dict(d) == spec, "id-movie/cut spec must round-trip through JSON"

    rejects = {
        "cut + branch_frame": dict(goals=(
            StudioMovieGoal("s0", "a"),
            StudioMovieGoal("s1", "b", parent_segment_id="s0",
                            joint_mode="cut", branch_frame=5))),
        "cut + context_frames": dict(goals=(
            StudioMovieGoal("s0", "a"),
            StudioMovieGoal("s1", "b", parent_segment_id="s0",
                            joint_mode="cut", context_frames=4))),
        "goal-0 cut": dict(goals=(StudioMovieGoal("s0", "a", joint_mode="cut"),)),
        ">4 references": dict(goals=(StudioMovieGoal("s0", "a"),),
                              reference_images=("1", "2", "3", "4", "5")),
        "blank reference": dict(goals=(StudioMovieGoal("s0", "a"),),
                                reference_images=("",)),
        "reference not a list": dict(goals=(StudioMovieGoal("s0", "a"),),
                                     reference_images="x.png"),
    }
    for desc, kw in rejects.items():
        try:
            make_studio_movie(width=320, height=180, fps=_FPS, **kw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"make_studio_movie must reject: {desc}")


# --------------------------------------------------------------------------- #
# [2] Route: reference_images + a cut joint -> 200; the 4xx guards.
# --------------------------------------------------------------------------- #
def test_route_refs_and_cut():
    if not _PIL:
        print("      (PIL unavailable — skipping route id-movie/cut check)")
        return
    # id-movie with a cut second scene -> 200 (the operator's exact ask shape).
    r = client.post("/video/studio/movie", json={
        "resolution": {"width": 320, "height": 180, "fps": _FPS},
        "reference_images": [_REF_PNG, _REF_PNG2],
        "goals": [{"prompt": "on a sunny beach"},
                  {"prompt": "playing volleyball", "joint_mode": "cut"}]})
    assert r.status_code == 200, (r.status_code, r.get_json())
    assert isinstance(r.get_json().get("job_id"), str), r.get_json()

    # a cut with a branch_frame -> 400 (a cut carries no frame)
    r2 = client.post("/video/studio/movie", json={
        "goals": [{"prompt": "a"}, {"prompt": "b", "joint_mode": "cut", "branch_frame": 5}]})
    assert r2.status_code == 400, (r2.status_code, r2.get_json())

    # a jail-escaping reference -> 400
    r3 = client.post("/video/studio/movie", json={
        "goals": [{"prompt": "a"}], "reference_images": ["/etc/passwd"]})
    assert r3.status_code == 400, (r3.status_code, r3.get_json())

    # >4 references -> 400
    r4 = client.post("/video/studio/movie", json={
        "goals": [{"prompt": "a"}], "reference_images": [_REF_PNG] * 5})
    assert r4.status_code == 400, (r4.status_code, r4.get_json())

    # a reference pointing at a non-image (this .py file, in-jail? no — use a video-less
    # bad file under the jail): a text file -> 400 (classified non-image).
    bad = os.path.join(_WORK_REFS, "notimage.txt")
    with open(bad, "w") as fh:
        fh.write("not an image")
    r5 = client.post("/video/studio/movie", json={
        "goals": [{"prompt": "a"}], "reference_images": [bad]})
    assert r5.status_code == 400, (r5.status_code, r5.get_json())


# --------------------------------------------------------------------------- #
# [3] Runner (FAKE SEAM): id-movie + cut. Prove every segment spec carries the
#     references + id_lock + bumped budget; the cut extracts NO frame + parent full;
#     movie.json labels {mode:"cut"}.
# --------------------------------------------------------------------------- #
def test_runner_id_movie_cut_fake_seam():
    if not (_FFMPEG and _FFPROBE and _PIL):
        print("      (ffmpeg/ffprobe/PIL unavailable — skipping id-movie/cut fake seam)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-idcut-fake-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    F_PARENT, F_CHILD = 24, 20
    parent_clip = os.path.join(work, "parent.mp4")
    child_clip = os.path.join(work, "child.mp4")
    _build_solid_clip(parent_clip, F_PARENT, 320, 180, _FPS, gray=90)
    _build_solid_clip(child_clip, F_CHILD, 320, 180, _FPS, gray=200)

    seen: list = []  # (index-in-order) captured specs
    orig_rpc = studio_movie.run_produce_clip
    orig_cancel = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False

    def fake_rpc(spec, should_cancel):
        seen.append(spec)
        # First call = segment 0 -> parent clip; second = the cut child.
        if len(seen) == 1:
            return Ok(Artifact(path=parent_clip, content_hash=f"p{len(seen)}",
                               frames=F_PARENT, width=320, height=180,
                               duration_s=F_PARENT / float(_FPS), resumed=False))
        return Ok(Artifact(path=child_clip, content_hash=f"c{len(seen)}",
                           frames=F_CHILD, width=320, height=180,
                           duration_s=F_CHILD / float(_FPS), resumed=False))

    studio_movie.run_produce_clip = fake_rpc
    try:
        refs = (_REF_PNG, _REF_PNG2)
        goals = (StudioMovieGoal("s0", "on a sunny beach"),
                 StudioMovieGoal("s1", "playing volleyball",
                                 parent_segment_id="s0", joint_mode="cut"))
        spec = _movie_spec(goals, out_root=work, reference_images=refs)
        res = run_generate_studio_movie(spec, job_id="idcut-fake")
        assert res.ok is True, f"faked id-movie/cut must be ok=True; got {res.error}"

        # (a) EVERY segment spec carries the movie references + capability id_lock + the
        #     bumped VACE budget (id_lock routes through VACE).
        assert len(seen) == 2, f"expected 2 segment renders; got {len(seen)}"
        for i, sp in enumerate(seen):
            assert tuple(sp.reference_images) == refs, (
                f"segment {i} spec must carry the movie references; got {sp.reference_images}")
            assert sp.capability == "id_lock", (
                f"segment {i} spec must render capability id_lock; got {sp.capability!r}")
            assert sp.vram_budget_gb >= _VACE_MIN_BUDGET_GB, (
                f"segment {i} budget must be bumped to the VACE floor; got {sp.vram_budget_gb}")
        # The CUT child carries NO branch still + NO context frames.
        assert seen[1].start_image in (None, ""), (
            f"a cut child conditions on NO branch still; got start_image={seen[1].start_image!r}")
        assert not seen[1].vace_context_frames, (
            f"a cut child carries NO context frames; got {seen[1].vace_context_frames}")

        movie_root = os.path.join(work, "idcut-fake")
        # (b) the CUT extracts nothing: no branch.png, no context dir under segment_01.
        assert not os.path.isfile(os.path.join(movie_root, "segment_01", "branch.png")), (
            "a cut joint must NOT extract a branch frame")
        assert not os.path.isdir(os.path.join(movie_root, "segment_01", "context")), (
            "a cut joint must NOT extract context frames")

        # (b) the parent plays in FULL (no trim): total = F_PARENT + F_CHILD.
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        expected_total = F_PARENT + F_CHILD
        got = _ffprobe_frames(movie_mp4)
        assert got == expected_total, (
            f"a cut assembles parent FULL + child FULL = {expected_total}; got {got}")

        # (b) movie.json: the joint is a cut (branch_frame null, trim = parent full),
        #     and the id-movie is recorded.
        with open(os.path.join(movie_root, "movie.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        joint = man["joints"][0]
        assert joint["mode"] == "cut", joint
        assert joint["branch_frame"] is None, joint
        assert joint["trim_frames"] == F_PARENT, joint
        assert man["id_lock"] is True and man["reference_images"] == list(refs), man
        s1 = next(s for s in man["segments"] if s["segment_id"] == "s1")
        assert s1["joint_mode"] == "cut" and s1["resolved_branch"] is None, s1
        assert s1["capability"] == "id_lock", s1
        assert man["assembly"]["total_frames"] == expected_total, man["assembly"]
    finally:
        studio_movie.run_produce_clip = orig_rpc
        media_bus.is_cancelling = orig_cancel
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [4] Runner (FAKE SEAM): id-movie + STILL joint. The still segment carries BOTH
#     the branch still AND the references (documented precedence: refs win the
#     render, the still governs the parent trim).
# --------------------------------------------------------------------------- #
def test_runner_id_movie_still_carries_both():
    if not (_FFMPEG and _FFPROBE and _PIL):
        print("      (ffmpeg/ffprobe/PIL unavailable — skipping id-movie/still-both check)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-idstill-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    F_PARENT, F_CHILD, BRANCH = 24, 20, 10
    parent_clip = os.path.join(work, "parent.mp4")
    child_clip = os.path.join(work, "child.mp4")
    _build_solid_clip(parent_clip, F_PARENT, 320, 180, _FPS, gray=90)
    _build_solid_clip(child_clip, F_CHILD, 320, 180, _FPS, gray=200)

    seen: list = []
    orig_rpc = studio_movie.run_produce_clip
    orig_cancel = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False

    def fake_rpc(spec, should_cancel):
        seen.append(spec)
        if len(seen) == 1:
            return Ok(Artifact(path=parent_clip, content_hash="p", frames=F_PARENT,
                               width=320, height=180, duration_s=F_PARENT / float(_FPS),
                               resumed=False))
        return Ok(Artifact(path=child_clip, content_hash="c", frames=F_CHILD,
                           width=320, height=180, duration_s=F_CHILD / float(_FPS),
                           resumed=False))

    studio_movie.run_produce_clip = fake_rpc
    try:
        refs = (_REF_PNG,)
        goals = (StudioMovieGoal("s0", "on a sunny beach"),
                 StudioMovieGoal("s1", "still on the beach", parent_segment_id="s0",
                                 branch_frame=BRANCH))  # joint_mode default "still"
        spec = _movie_spec(goals, out_root=work, reference_images=refs)
        res = run_generate_studio_movie(spec, job_id="idstill")
        assert res.ok is True, f"faked id-movie/still must be ok=True; got {res.error}"

        # The STILL child carries BOTH the branch still (start_image) AND the references,
        # under capability id_lock (references win the render; the still governs the trim).
        child_spec = seen[1]
        assert child_spec.start_image and os.path.isfile(child_spec.start_image), (
            f"a still child must carry the extracted branch still; got {child_spec.start_image!r}")
        assert tuple(child_spec.reference_images) == refs, (
            f"a still id-movie child must ALSO carry the references; got {child_spec.reference_images}")
        assert child_spec.capability == "id_lock", child_spec.capability

        # The still trim is honored (references do not change the assembly math):
        # parent trimmed to BRANCH+1, child full -> (BRANCH+1) + F_CHILD.
        movie_mp4 = os.path.join(work, "idstill", "movie.mp4")
        assert _ffprobe_frames(movie_mp4) == (BRANCH + 1) + F_CHILD
        joint = res.movie["joints"][0]
        assert joint["mode"] == "still" and joint["branch_frame"] == BRANCH, joint
        assert joint["trim_frames"] == BRANCH + 1, joint
    finally:
        studio_movie.run_produce_clip = orig_rpc
        media_bus.is_cancelling = orig_cancel
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [5] Assembly math (REAL SYNTHETIC, done/ok): a PLAIN movie (no references) whose
#     second goal is a "cut" renders end-to-end on this GPU-less box (a cut child is
#     a fresh t2v render the synthetic tier serves) — total = parent FULL + child
#     FULL, movie.json labels mode="cut". This exercises the whole spec/runner/
#     assembly path for the cut with NO GPU.
# --------------------------------------------------------------------------- #
def test_assembly_cut_full_plus_full_synthetic():
    if not (_FFMPEG and _FFPROBE):
        print("      (ffmpeg/ffprobe unavailable — skipping synthetic cut assembly)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-cut-synth-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        goals = (StudioMovieGoal("s0", "a lighthouse on a cliff"),
                 StudioMovieGoal("s1", "a busy market square",
                                 parent_segment_id="s0", joint_mode="cut"))
        spec = _movie_spec(goals, out_root=work)   # NO references -> plain movie, synthetic
        res = run_generate_studio_movie(spec, job_id="cut-synth")
        assert res.ok is True, f"a plain cut movie must render synthetically; got {res.error}"

        movie_root = os.path.join(work, "cut-synth")
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        # a cut trims NOTHING: total = both segments in FULL.
        assert _ffprobe_frames(movie_mp4) == _SEG_FRAMES + _SEG_FRAMES, (
            "a cut assembles parent FULL + child FULL (no trim)")
        joint = res.movie["joints"][0]
        assert joint["mode"] == "cut" and joint["branch_frame"] is None, joint
        assert joint["trim_frames"] == _SEG_FRAMES, joint
        # a plain-movie cut child is a fresh t2v render (no identity, no start image).
        s1 = next(s for s in res.movie["segments"] if s["segment_id"] == "s1")
        assert s1["capability"] == "t2v" and s1["joint_mode"] == "cut", s1
        assert res.movie["id_lock"] is False and res.movie["reference_images"] == [], res.movie
        # per-segment clips stay WHOLE (non-destructive).
        for rec in res.movie["segments"]:
            assert _ffprobe_frames(rec["clip_path"]) == _SEG_FRAMES, rec
    finally:
        media_bus.is_cancelling = orig
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# [6] Runner (REAL id-movie, GPU-LESS): an id-movie on THIS box fails segment 0 with
#     a GRACEFUL Err naming the segment + capability id_lock. No monkeypatch — proves the
#     real route-to-VACE + graceful-degrade path (there is NO synthetic id_lock tier; a
#     graceful Err is the honest result).
#
# RULING UPDATE (2026-09-24): video is placed like an LLM. A REAL id_lock render (a real
# Wan-VACE model at the bumped VACE budget) is NO LONGER run on central's GPU-less
# in-process path — which only ever produced deps_missing. On a box with NO studio worker
# it is a NAMED placement refusal (``no_studio_worker``) that STILL names the segment +
# id_lock and STILL records the segment as failed with the bumped budget. The old
# deps_missing/no_gpu/weights_missing codes now come only FROM a worker that took the job.
# --------------------------------------------------------------------------- #
def test_real_id_movie_graceful_err():
    if not (_FFMPEG and _FFPROBE and _PIL):
        print("      (ffmpeg/ffprobe/PIL unavailable — skipping real id-movie graceful Err)")
        return
    work = tempfile.mkdtemp(prefix="studio-movie-idreal-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    try:
        refs = (_REF_PNG, _REF_PNG2)
        goals = (StudioMovieGoal("s0", "on a sunny beach"),
                 StudioMovieGoal("s1", "playing volleyball",
                                 parent_segment_id="s0", joint_mode="cut"))
        spec = _movie_spec(goals, out_root=work, vram=0.5, reference_images=refs)
        res = run_generate_studio_movie(spec, job_id="idreal")

        # honest per-segment Err — NOT a hang/raise/500, NOT a synthetic fallback.
        assert res.ok is False, "an id-movie segment that can't run VACE must fail as DATA"
        assert res.error is not None
        # RULING (2026-09-24): a real id_lock render with no studio worker refuses by name
        # (placement), never the GPU-less in-process deps_missing.
        assert res.error.code in (
            "no_studio_worker", "studio_model_not_on_worker", "no_feasible_studio_worker"), (
            f"this box has no studio worker, so a real id_lock render must surface a NAMED "
            f"placement refusal; got {res.error.code}: {res.error.message}")
        # the JobError names segment 0 + capability id_lock.
        assert "segment 0" in res.error.message and "id_lock" in res.error.message, (
            f"error must name the failing segment + capability; got {res.error.message!r}")

        # movie.json: segment 0 recorded FAILED under id_lock with the bumped budget.
        with open(os.path.join(work, "idreal", "movie.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        assert man["id_lock"] is True, man
        s0 = next(s for s in man["segments"] if s["segment_id"] == "s0")
        assert s0["status"] == "failed" and s0["capability"] == "id_lock", s0
        assert s0["vram_budget_gb"] >= _VACE_MIN_BUDGET_GB, s0
    finally:
        media_bus.is_cancelling = orig
        shutil.rmtree(work, ignore_errors=True)


CHECKS = [
    ("schema: reference_images + cut build/round-trip; cut+branch/cut+ctx/goal0-cut/>4-refs rejected",
     test_schema_refs_and_cut),
    ("route: reference_images + cut joint -> 200; cut+branch / jail-escape / >4 / non-image -> 4xx",
     test_route_refs_and_cut),
    ("runner: fake seam id-movie/cut — every seg carries refs+id_lock+bumped budget, cut extracts nothing, parent full, movie.json cut",
     test_runner_id_movie_cut_fake_seam),
    ("runner: fake seam id-movie/still — child carries BOTH branch still AND references (refs win render, still governs trim)",
     test_runner_id_movie_still_carries_both),
    ("assembly: plain cut movie renders SYNTHETICALLY done/ok — parent FULL + child FULL, movie.json cut",
     test_assembly_cut_full_plus_full_synthetic),
    ("runner: REAL id-movie, no worker -> segment 0 NAMED placement refusal (no_studio_worker), names segment + id_lock",
     test_real_id_movie_graceful_err),
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
        shutil.rmtree(_WORK_REFS, ignore_errors=True)
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
