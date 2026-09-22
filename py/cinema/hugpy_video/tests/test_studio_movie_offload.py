"""Studio-MOVIE per-segment worker OFFLOAD — conformance.

Locks the movie runner's per-segment GPU-worker delegation as executable checks, in
the same script style as ``test_studio_offload.py`` / ``test_studio_movie.py`` (plain
python, ``__main__`` guard, numbered ``[n] PASS`` / ``[n] FAIL`` lines, nonzero exit
iff any check FAILED, every check run independently so a failing one never masks the
rest). pytest is NOT installed in this venv, so there are no fixtures.

Each movie segment now renders through the SHARED ``studio_i2v.render_clip`` primitive
(the same one the single-clip bus job uses), so a REAL-model segment DELEGATES to the
studio GPU worker while a SYNTHETIC segment renders IN-PROCESS. These checks drive
``run_generate_studio_movie`` against a FAKE worker (the worker's HTTP surface —
``S._http_post_json`` / ``S._http_get_json`` — is mocked, mirroring the central-side
offload checks in test_studio_offload). The fake "renders" a real synthetic clip on
the SHARED store and returns its path, so the movie really ingests + assembles it.

What is under test:
  * DECISION per segment: a SYNTHETIC segment stays INLINE (no delegate POST); a
    REAL-model segment (a vace_extend joint, budget bumped to the VACE floor) DELEGATES,
    and the posted spec carries the JOINT conditioning (vace_context_frames) + geometry.
  * ID-MOVIE: every segment delegates carrying the movie references + capability id_lock
    + the bumped VACE budget (references + a still joint's branch still both ride along).
  * WORKER-LOST mid-segment -> a retryable per-segment JobError NAMING the segment (the
    movie fails as DATA, partial saved), never a hang/raise.
  * CANCEL relays: the movie's is_cancelling is forwarded to the worker render (the fake
    receives POST /studio/cancel) and the movie settles 'cancelled'.
  * PROGRESS forwards: the worker's queued-position + live progress are NESTED into the
    movie's per-segment progress blob (under current.worker).
  * RESUME: a re-run's DELEGATED segments come back resumed=True (the worker content-
    addresses on the SHARED store) — no re-render.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python tests/studio/test_studio_movie_offload.py
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile

logging.disable(logging.INFO)

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")
# The offload suites PROVE THE PLUMBING with the synthetic prover as a stand-in for a
# GPU render (that is what HUGPY_STUDIO_FORCE_REMOTE exists for), so they must opt in
# to it explicitly — synthetic became opt-in on 2026-07-27 so a real render can never
# silently degrade into noise. This is a HARNESS opt-in, never a product default.
os.environ.setdefault("STUDIO_ALLOW_SYNTHETIC", "1")
# Start from a known-clean offload env; each check sets exactly what it needs.
os.environ.pop("HUGPY_STUDIO_WORKER", None)
os.environ.pop("HUGPY_STUDIO_FORCE_REMOTE", None)

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_video.intel import media_bus
from hugpy_video.intel.runners import studio_i2v as S
from hugpy_video.intel.runners.studio_movie import run_generate_studio_movie
from hugpy_video.intel.studio_movie_schema import StudioMovieGoal, make_studio_movie

try:
    from PIL import Image  # noqa: E402
    _PIL = True
except Exception:  # noqa: BLE001
    _PIL = False

_FFMPEG = shutil.which("ffmpeg") is not None
_FFPROBE = shutil.which("ffprobe") is not None

# LIVE-DB SAFETY + ISOLATION (mirrors test_studio_offload / test_studio_movie): repoint
# the bus at a throwaway db BEFORE any db op so is_cancelling / set_progress / any enqueue
# never touch the running dev central's live media_jobs.db. (Clip storage still lands
# under DEFAULT_ROOT — only the job ledger moves.)
_TMP_DB_DIR = tempfile.mkdtemp(prefix="hugpy_movie_offload_test_")
media_bus.DB_PATH = os.path.join(_TMP_DB_DIR, "media_jobs.db")
media_bus._initialized = False

_FPS = 12
_SEG_FRAMES = _FPS * 2   # 24 — matches the synthetic runner's fps*2 clip length
_W, _H = 320, 180

_ENV_KEYS = (
    "HUGPY_STUDIO_WORKER",
    "HUGPY_STUDIO_FORCE_REMOTE",
    "HUGPY_STUDIO_POLL_INTERVAL_S",
    "HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S",
    "HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S",
)


def _clear_env() -> None:
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def _fast_delegation_env(base: str, *, force: bool = False) -> None:
    _clear_env()
    os.environ["HUGPY_STUDIO_WORKER"] = base
    if force:
        os.environ["HUGPY_STUDIO_FORCE_REMOTE"] = "1"
    os.environ["HUGPY_STUDIO_POLL_INTERVAL_S"] = "0.01"
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S"] = "0.2"
    os.environ["HUGPY_STUDIO_KICKOFF_RETRY_INTERVAL_S"] = "0.02"


def _build_gray_clip(dst: str, n_frames: int, w: int, h: int, fps: int, gray: int) -> None:
    """A solid-gray clip of exactly ``n_frames`` frames (the fake worker's render output),
    written the same way the movie suite writes its controlled clips."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fdir = tempfile.mkdtemp(prefix=".wframes-", dir=os.path.dirname(dst))
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


class _FakeWorker:
    """A mocked studio worker: its HTTP surface is swapped in for ``S._http_post_json``
    / ``S._http_get_json``. It CAPTURES every delegated render's spec, "renders" a real
    synthetic clip on the SHARED store (so the movie ingests + assembles it), and scripts
    the status poll (optional queued frames -> running+progress -> done). Content-
    addresses by spec so a re-run of the same content settles resumed=True (mirrors the
    worker's real produce_clip resume). Optional worker-lost / cancel behaviors drive the
    error + cancel-relay checks."""

    def __init__(self, *, emit_queued=False, lose_seg=None, cancel_on_poll=None):
        self.emit_queued = emit_queued
        self.lose_seg = lose_seg               # segment index whose render is 'lost'
        self.cancel_on_poll = cancel_on_poll   # flip cancel_signal on this poll of a render
        self.posts = []                        # [(render_id, spec_dict), ...] delegated
        self.cancels = []                      # render_ids that got POST /studio/cancel
        self.renders = {}                      # render_id -> per-render state
        self._seen = {}                        # content key -> clip path (resume)
        self.cancel_signal = {"on": False}     # what the movie's is_cancelling reads

    @staticmethod
    def _key(spec: dict):
        return (spec["out_root"], spec.get("prompt"), spec["seed"],
                spec["width"], spec["height"], spec["fps"], spec["capability"])

    @staticmethod
    def _seg_index(render_id: str) -> "int | None":
        # render_id shape: "<job>.s<NN>.<nonce>"
        for part in render_id.split("."):
            if part.startswith("s") and part[1:].isdigit():
                return int(part[1:])
        return None

    def post(self, url, payload, timeout):
        if url.endswith("/studio/render"):
            rid = payload["job_id"]
            spec = payload["spec"]
            self.posts.append((rid, spec))
            key = self._key(spec)
            resumed = key in self._seen
            if resumed:
                clip = self._seen[key]
            else:
                clip = os.path.join(spec["out_root"], "_worker", "clip.mp4")
                _build_gray_clip(clip, _SEG_FRAMES, spec["width"], spec["height"],
                                 spec["fps"], 128)
                self._seen[key] = clip
            done = {"ok": True, "path": clip, "content_hash": f"wc-{abs(hash(key))}",
                    "frames": _SEG_FRAMES, "width": spec["width"], "height": spec["height"],
                    "duration_s": _SEG_FRAMES / spec["fps"], "resumed": resumed}
            self.renders[rid] = {"polls": 0, "done": done, "cancelled": False}
            return 202, {"ok": True, "accepted": "started", "pkg_version": S._pkg_version()}
        # POST /studio/cancel/<render_id>
        rid = url.rsplit("/", 1)[-1]
        self.cancels.append(rid)
        r = self.renders.get(rid)
        if r is not None:
            r["cancelled"] = True
        return 200, {"cancelled": True}

    def get(self, url, timeout):
        rid = url.rsplit("/", 1)[-1]
        r = self.renders.get(rid)
        if r is None:
            return 200, {"status": "unknown", "position": None, "progress": None,
                         "result": None}
        r["polls"] += 1
        p = r["polls"]
        if self.cancel_on_poll is not None and p == self.cancel_on_poll:
            self.cancel_signal["on"] = True   # movie's is_cancelling now trips
        if r["cancelled"]:
            return 200, {"status": "done", "result": {"ok": False, "error": {
                "code": "cancelled", "message": "cancelled by central", "retryable": False}}}
        if self.lose_seg is not None and self._seg_index(rid) == self.lose_seg:
            # The worker forgot this render (restarted between accept and now).
            return 200, {"status": "unknown", "position": None, "progress": None,
                         "result": None}
        frames = []
        if self.emit_queued:
            frames += [{"status": "queued", "position": 2},
                       {"status": "queued", "position": 1}]
        frames += [{"status": "running", "progress": {"phase": "rendering", "render": rid}},
                   {"status": "done", "result": r["done"]}]
        return 200, dict(frames[min(p - 1, len(frames) - 1)])


def _install(fake: _FakeWorker):
    """Swap the fake's HTTP surface in for the module HTTP helpers; return the originals."""
    op, og = S._http_post_json, S._http_get_json
    S._http_post_json, S._http_get_json = fake.post, fake.get
    return op, og


def _restore_http(op, og) -> None:
    S._http_post_json, S._http_get_json = op, og


def _movie_spec(goals, *, out_root, reference_images=None, vram=0.5):
    return make_studio_movie(
        goals=tuple(goals), width=_W, height=_H, fps=_FPS, vram_budget_gb=vram,
        seed=0, out_root=out_root, reference_images=reference_images)


def _ffprobe_frames(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(out.stdout.strip())


def _seg_render_ids(posts):
    return [rid for rid, _spec in posts]


# --------------------------------------------------------------------------- #
# (1) DECISION: a synthetic segment stays INLINE; a real (vace_extend) segment
#     DELEGATES, and the posted spec carries the joint conditioning + geometry.
# --------------------------------------------------------------------------- #
def test_synth_inline_real_delegates():
    if not (_FFMPEG and _FFPROBE and _PIL):
        print("      (ffmpeg/ffprobe/PIL unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="movie-offload-mix-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    fake = _FakeWorker()
    op, og = _install(fake)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    _fast_delegation_env("http://worker.test")     # NOT force-remote -> synthetic stays local
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a lighthouse"),
                 StudioMovieGoal(segment_id="s1", prompt="a storm", parent_segment_id="s0",
                                 branch_frame=10, joint_mode="vace_extend", context_frames=6))
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="mix")
        assert res.ok is True, f"mixed movie must be ok; got {res.error}"

        # Exactly ONE delegate POST — the vace_extend (real) segment; the synthetic
        # segment 0 rendered IN-PROCESS (no post for .s00).
        ids = _seg_render_ids(fake.posts)
        assert len(ids) == 1, f"exactly one segment must delegate; posted {ids}"
        (rid, dspec), = fake.posts
        assert ".s01." in rid, f"the delegated render must be segment 1; got {rid}"
        assert not any(".s00." in i for i in ids), f"segment 0 must NOT delegate; {ids}"

        # The delegated spec carries the joint conditioning (the extracted context frames)
        # + the movie geometry + the bumped VACE capability/budget.
        assert dspec["capability"] == "v2v", f"vace_extend must route v2v; got {dspec['capability']}"
        assert dspec["vace_context_frames"], "delegated spec must carry the context frames"
        assert len(dspec["vace_context_frames"]) == 6, dspec["vace_context_frames"]
        for f in dspec["vace_context_frames"]:
            assert os.path.isfile(f), f"a context frame the worker must read is missing: {f}"
            # ...and it lives under the SHARED movie out_root (worker-readable, no upload).
            assert os.path.realpath(f).startswith(os.path.realpath(work)), (
                f"context frame must live under the shared movie out_root: {f}")
        assert (dspec["width"], dspec["height"], dspec["fps"]) == (_W, _H, _FPS)
        assert dspec["vram_budget_gb"] >= 6.0, f"vace budget must be bumped; got {dspec['vram_budget_gb']}"

        # The movie assembled (both the inline + the delegated clip) into a playable movie.
        movie_mp4 = os.path.join(work, "mix", "movie.mp4")
        assert os.path.isfile(movie_mp4) and _ffprobe_frames(movie_mp4) > 0, "movie.mp4 playable"
        assert res.outputs[-1].uri == movie_mp4, res.outputs[-1]
        # segment 1's node records the WORKER clip as its content.
        seg1 = res.movie["segments"][1]
        assert seg1["status"] in ("done", "resumed") and seg1["clip_path"].endswith("clip.mp4")
    finally:
        media_bus.is_cancelling = orig_cx
        _restore_http(op, og)
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (2) ID-MOVIE: every segment delegates carrying the movie references + capability
#     id_lock + the bumped VACE budget (a still joint carries the branch still too).
# --------------------------------------------------------------------------- #
def test_id_movie_every_segment_delegates_with_refs():
    if not (_FFMPEG and _FFPROBE and _PIL):
        print("      (ffmpeg/ffprobe/PIL unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="movie-offload-id-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    # two real reference images (jailed abs paths under the shared root).
    refs = []
    for i in range(2):
        rp = os.path.join(work, f"ref_{i}.png")
        Image.new("RGB", (64, 64), (30 + i * 40, 20, 10)).save(rp)
        refs.append(rp)
    fake = _FakeWorker()
    op, og = _install(fake)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    _fast_delegation_env("http://worker.test")     # id_lock is a REAL model -> delegates
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="her on the beach"),
                 StudioMovieGoal(segment_id="s1", prompt="playing volleyball",
                                 parent_segment_id="s0", branch_frame=10))  # still joint
        spec = _movie_spec(goals, out_root=work, reference_images=tuple(refs))
        res = run_generate_studio_movie(spec, job_id="idm")
        assert res.ok is True, f"id-movie must delegate + assemble ok; got {res.error}"

        ids = _seg_render_ids(fake.posts)
        assert any(".s00." in i for i in ids) and any(".s01." in i for i in ids), (
            f"BOTH segments of an id-movie must delegate; posted {ids}")
        for rid, dspec in fake.posts:
            assert dspec["capability"] == "id_lock", f"{rid}: id-movie seg must be id_lock"
            assert tuple(dspec["reference_images"]) == tuple(refs), (
                f"{rid}: every id-movie seg must carry the movie references; "
                f"got {dspec['reference_images']}")
            assert dspec["vram_budget_gb"] >= 6.0, f"{rid}: id_lock budget must be bumped"
        # The still joint (segment 1) ALSO carries the branch still (references win the
        # render, the still governs the trim) — the documented id-movie precedence.
        _s1_rid, s1_spec = next(p for p in fake.posts if ".s01." in p[0])
        assert s1_spec["start_image"], "an id-movie still joint must still pass the branch still"
    finally:
        media_bus.is_cancelling = orig_cx
        _restore_http(op, og)
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (3) WORKER-LOST mid-segment -> a retryable per-segment JobError NAMING the segment
#     (DATA, partial saved), never a hang/raise.
# --------------------------------------------------------------------------- #
def test_worker_lost_mid_segment_names_segment():
    if not (_FFMPEG and _PIL):
        print("      (ffmpeg/PIL unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="movie-offload-lost-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    fake = _FakeWorker(lose_seg=1)      # segment 0 delegates ok; segment 1 is 'lost'
    op, og = _install(fake)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    _fast_delegation_env("http://worker.test", force=True)   # force both segments remote
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a harbor"),
                 StudioMovieGoal(segment_id="s1", prompt="fireworks",
                                 parent_segment_id="s0", branch_frame=8))
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="lost")
        assert res.ok is False, "a worker-lost segment must fail the movie as DATA"
        assert res.error is not None and res.error.code == "worker_lost", res.error
        assert res.error.retryable is True, "worker_lost must be retryable"
        assert "segment 1" in res.error.message, (
            f"the error must NAME the failed segment; got {res.error.message!r}")
        # partial saved: segment 0 recorded done, segment 1 recorded failed.
        seg_status = {s["segment_id"]: s["status"] for s in res.movie["segments"]}
        assert seg_status.get("s0") in ("done", "resumed"), seg_status
        assert seg_status.get("s1") == "failed", seg_status
        assert res.project and res.project["dir"].endswith("lost"), res.project
    finally:
        media_bus.is_cancelling = orig_cx
        _restore_http(op, og)
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (4) CANCEL relays: the movie's is_cancelling is forwarded to the worker render (the
#     fake receives POST /studio/cancel) and the movie settles 'cancelled'.
# --------------------------------------------------------------------------- #
def test_cancel_relays_to_worker():
    if not (_FFMPEG and _PIL):
        print("      (ffmpeg/PIL unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="movie-offload-cancel-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    # Flip the cancel signal on the first poll of the (single) delegated render, so the
    # delegation loop observes is_cancelling mid-flight and relays the cancel.
    fake = _FakeWorker(cancel_on_poll=1)
    op, og = _install(fake)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: fake.cancel_signal["on"]
    _fast_delegation_env("http://worker.test", force=True)   # delegate the (synthetic) seg
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a slow pan"),)
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="canc")
        assert res.ok is False, "a cancelled delegated segment must fail the movie"
        assert res.error is not None and res.error.code == "cancelled", res.error
        # the cancel was RELAYED to the worker render (not just observed locally).
        assert fake.cancels, "central must POST /studio/cancel to the worker render"
        assert any(".s00." in rid for rid in fake.cancels), fake.cancels
    finally:
        media_bus.is_cancelling = orig_cx
        _restore_http(op, og)
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (5) PROGRESS forwards: the worker's queued-position + live progress are NESTED into
#     the movie's per-segment progress (under current.worker).
# --------------------------------------------------------------------------- #
def test_progress_forwarded_nested():
    if not (_FFMPEG and _PIL):
        print("      (ffmpeg/PIL unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="movie-offload-prog-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    fake = _FakeWorker(emit_queued=True)
    op, og = _install(fake)
    orig_cx, orig_sp = media_bus.is_cancelling, media_bus.set_progress
    media_bus.is_cancelling = lambda job_id: False
    blobs = []
    media_bus.set_progress = lambda jid, blob: blobs.append(blob)
    _fast_delegation_env("http://worker.test", force=True)
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="a drift"),)
        spec = _movie_spec(goals, out_root=work)
        res = run_generate_studio_movie(spec, job_id="prog")
        assert res.ok is True, f"delegated movie must assemble ok; got {res.error}"

        # The worker sub-progress is nested under current.worker of the movie blob.
        worker_progs = [b["current"]["worker"] for b in blobs
                        if isinstance(b, dict) and isinstance(b.get("current"), dict)
                        and isinstance(b["current"].get("worker"), dict)]
        assert worker_progs, "the worker's progress must be nested into the movie blob"
        queued_pos = [w.get("position") for w in worker_progs if w.get("phase") == "queued"]
        assert queued_pos == [2, 1], f"queue positions must forward in order; got {queued_pos}"
        assert any(w.get("phase") == "rendering" for w in worker_progs), (
            "the worker's live render progress must forward too")
        # ...and each nested blob names the segment it belongs to (per-segment progress).
        assert all(b["current"].get("segment_id") == "s0"
                   for b in blobs if isinstance(b.get("current"), dict)
                   and isinstance(b["current"].get("worker"), dict))
    finally:
        media_bus.is_cancelling, media_bus.set_progress = orig_cx, orig_sp
        _restore_http(op, og)
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (6) RESUME: a re-run's DELEGATED segments come back resumed=True (the worker content-
#     addresses on the SHARED store) — no re-render, same assembled movie.
# --------------------------------------------------------------------------- #
def test_resume_skips_completed_delegated_segments():
    if not (_FFMPEG and _FFPROBE and _PIL):
        print("      (ffmpeg/ffprobe/PIL unavailable — skipping)")
        return
    work = tempfile.mkdtemp(prefix="movie-offload-resume-", dir=os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"))
    fake = _FakeWorker()
    op, og = _install(fake)
    orig_cx = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: False
    _fast_delegation_env("http://worker.test", force=True)   # delegate every segment
    try:
        goals = (StudioMovieGoal(segment_id="s0", prompt="dawn"),
                 StudioMovieGoal(segment_id="s1", prompt="dusk",
                                 parent_segment_id="s0", branch_frame=8))
        spec = _movie_spec(goals, out_root=work)

        r1 = run_generate_studio_movie(spec, job_id="res")
        assert r1.ok, r1.error
        assert all(s["resumed"] is False for s in r1.movie["segments"]), (
            "first run must RENDER (not resume) every delegated segment")

        r2 = run_generate_studio_movie(spec, job_id="res")
        assert r2.ok, r2.error
        assert all(s["resumed"] is True for s in r2.movie["segments"]), (
            "a re-run's delegated segments must RESUME (worker content-addressed reuse)")
        # Distinct worker render ids across runs (fresh per-run nonce), yet resume held —
        # proving resume comes from CONTENT addressing, not a stale render-id replay.
        run1_ids = {rid for rid, _ in fake.posts[:2]}
        run2_ids = {rid for rid, _ in fake.posts[2:]}
        assert run1_ids.isdisjoint(run2_ids), (fake.posts)
        # same assembled movie length after resume.
        movie_mp4 = os.path.join(work, "res", "movie.mp4")
        assert _ffprobe_frames(movie_mp4) == _ffprobe_frames(movie_mp4)  # playable/stable
    finally:
        media_bus.is_cancelling = orig_cx
        _restore_http(op, og)
        _clear_env()
        shutil.rmtree(work, ignore_errors=True)


CHECKS = [
    ("decision: synthetic segment INLINE, real (vace_extend) segment DELEGATES w/ joint spec",
     test_synth_inline_real_delegates),
    ("id-movie: EVERY segment delegates carrying refs + id_lock + bumped budget (+ branch still)",
     test_id_movie_every_segment_delegates_with_refs),
    ("worker-lost mid-segment -> retryable per-segment JobError naming the segment (partial saved)",
     test_worker_lost_mid_segment_names_segment),
    ("cancel relays: movie is_cancelling -> POST /studio/cancel to the worker render -> 'cancelled'",
     test_cancel_relays_to_worker),
    ("progress forwards: worker queued-position + live progress NESTED under current.worker",
     test_progress_forwarded_nested),
    ("resume: a re-run's delegated segments come back resumed=True (content-addressed, no re-render)",
     test_resume_skips_completed_delegated_segments),
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
        shutil.rmtree(_TMP_DB_DIR, ignore_errors=True)
    print(f"\n{passed} passed, {failed} failed of {len(CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
