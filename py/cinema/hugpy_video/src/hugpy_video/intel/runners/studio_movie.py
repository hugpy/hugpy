"""Pure ``(studio, generate_studio_movie)`` runner — an ordered strip of REAL studio
clips conjoined at splice points, like a single NLE timeline ROW.

``run_generate_studio_movie(spec, job_id) -> JobResult``. A FAT orchestrator that
sequences the movie's segments INLINE (NOT an orchestrator-of-child-jobs, which
would deadlock the single-daemon bus — it follows ``runners.movie``'s inline
pattern). media_jobs.db stays single-writer in the bus; this runner is pure
``(spec, job_id) -> JobResult``.

NLE-ROW SEMANTICS. A studio movie is ``[segment 0 | segment 1 | …]``: an ordered
strip of studio ``produce_clip`` outputs conjoined at splice points. Per segment,
in timeline order:
  a. Segment 0 renders t2v (or i2v when a movie-level ``start_image`` is given).
     Each LATER segment is spliced onto its parent per its ``joint_mode``:
       * "still" (default, backward-compatible): render i2v, conditioned on ONE still
         — the **branch frame** of the PREVIOUS segment's clip (``branch_frame:
         int | null``; null ⇒ the parent's LAST frame). The still is extracted with
         ffmpeg (a frame-accurate ``select=eq(n\\,B)`` pluck) and handed to
         ``produce_clip`` as the i2v ``start_image``. Motion is NOT carried.
       * "vace_extend": carry MOTION across the splice — extract the parent's TRAILING
         ``context_frames`` frames (``[branch-K+1 .. branch]``, clamped) and route the
         render through the VACE path (capability "v2v" -> Task.VACE_CONTROL) with those
         frames as the temporal conditioning. The VACE runner builds the diffusers
         video+mask extend idiom (kept context prefix + generated tail), so the segment
         CONTINUES the parent's motion instead of restarting from one frame. The child's
         first K output frames RECONSTRUCT the context and are DROPPED at assembly (see
         ASSEMBLY below) so no frame double-plays. A vace_extend segment routes to a real
         VACE model, so its per-segment vram budget is raised to the VACE floor
         (``_VACE_MIN_BUDGET_GB``); on a GPU-less box it returns the VACE runner's
         graceful Err (NO_GPU/DEPS_MISSING/WEIGHTS_MISSING) — an HONEST per-segment error,
         NEVER a silent fallback to still-mode.
       * "cut": a HARD SCENE CUT — NO frame carry at all. No branch still / context window
         is extracted; the child is a FRESH render of its own prompt. The parent is NOT
         trimmed (it plays in FULL) and assembly records the joint as ``{mode:"cut"}``.

     IDENTITY MOVIE (movie-level ``reference_images`` set) — the operator's "take that id
     and use it for a video: her on the beach, then playing volleyball". When the movie
     carries reference image(s), EVERY segment (segment 0 included) renders capability
     ``id_lock`` (Wan-VACE reference-to-video) with those references, so the locked SUBJECT
     carries across every scene change. The per-segment budget is raised to the shared
     ``_VACE_MIN_BUDGET_GB`` floor (id_lock routes through the VACE path, exactly like
     vace_extend). The joint behavior is UNCHANGED per mode — a "still" joint still extracts
     the branch frame and trims the parent, a "cut" joint still carries no frame — but the
     RENDER of each segment is now reference-conditioned. On the VACE path the i2v
     ``start_image`` (a "still" joint's branch frame) is ACCEPTED but UNUSED (the runner
     conditions on the references, not a single still) — so for an identity movie the
     REFERENCES win the conditioning; the branch frame governs only the parent TRIM at
     assembly. There is no synthetic id_lock tier, so an identity movie on a GPU-less box
     surfaces the VACE runner's graceful per-segment Err (see ``_VACE_MIN_BUDGET_GB``).
  b. The render goes through the SAME studio boundary the single-clip bus job uses:
     ``runners.studio_i2v.run_produce_clip`` (router -> manifest -> runner ->
     content-addressed clip). RESUME is content-addressed INSIDE ``produce_clip``:
     an identical segment spec re-run returns the existing clip (``resumed=True``),
     no regeneration — so a re-enqueue of the same movie skips/reuses every segment.
     Each segment renders under its OWN ``out_root`` subtree
     (``<movie_root>/segment_NN``) because ``start_image`` is NOT part of the studio
     content_hash (only prompt/seed/geometry/source_video/… are); isolating the
     out_root guarantees two segments never collide on a shared hash and each
     resumes independently.
  c. Between segments: honor ``is_cancelling(job_id)`` AND ``spec.time_budget_s``
     (this runner owns its OWN wall-clock — the bus has no timeout/reaper). The
     cancel probe is also threaded DOWN into ``produce_clip`` so a mid-render cancel
     aborts before a clip is written (Err(CANCELLED), errors-as-data).
  d. Emit NESTED movie progress via ``media_bus.set_progress`` — which is ALSO this
     job's HEARTBEAT (k91). The blob carries a monotonic 0..1 ``progress`` (completed
     segments + the in-flight segment's denoise step) and a human ``message``
     ("segment 2/14 - step 18/32"), which ``job_bridge.on_progress`` relays into the
     comms Job record. Without them a movie's bus row showed no forward progress for
     its whole run and ``comms/jobs.py::expire_pending_orphans`` retired a LIVE render
     as wedged. See ``_emit``.

SESSIONS (k91). A movie DIR is a resumable session, described by two sidecars:
``spec.json`` (the SUBMITTED spec + the CURRENT job id, written at enqueue and at every
resume — see ``write_movie_spec``) and ``movie.json`` (the manifest: what has actually
rendered, plus a coarse ``status`` of running/partial/paused/done). ``movie_root_for``
is the single definition of where that dir is, shared with the submit route so the two
cannot drift. Resume is a plain RE-ENQUEUE of the persisted spec: content-addressed
reuse inside ``produce_clip`` returns every completed segment as ``resumed=True``
without re-rendering, so the clips ARE the checkpoint and there is no separate
checkpoint format to keep honest.

That reuse is a lookup BY PATH, which makes the dir load-bearing: a resume mints a NEW
bus job id, and an UNNAMED movie's dir leaf IS its job id, so the dir would move on
every resume and every segment would re-render while the run reported success. The
resume route therefore stamps ``spec.session_id`` — a pin on the leaf — and
``movie_root_for`` prefers it over the derived name. See that function.

NON-DESTRUCTIVE TRIM (metadata, honored at ASSEMBLY only). The per-segment clip
files stay WHOLE — they are content-addressed and never modified. A mid-frame
branch means the assembled movie uses the PARENT clip only UP TO the branch frame;
that trim is applied only when building ``movie.mp4`` (by re-encoding a trimmed
COPY of the parent into a work dir), never by re-rendering. The retained parent
length is ``trim_frames = branch_frame + 1`` (frames ``[0, branch_frame]``
inclusive); a null branch resolves to the parent's LAST frame, so the parent plays
in FULL. The LEAF segment (no child) plays in full. Concat is ffmpeg's concat
DEMUXER over the (uniformly re-encoded) contribution clips.

VACE-EXTEND OVERLAP (assembly). The parent-trim math above is UNCHANGED: the parent
still plays ``[0 .. branch]`` and the child still "starts at the branch frame". But a
``vace_extend`` child's output INCLUDES its first K frames as a RECONSTRUCTION of the
parent's trailing context (the kept mask=0 prefix). Those K frames overlap the parent's
tail, so they are DROPPED from the CHILD's head at concat (``context_drop = K`` frames):
the child contributes ``[K .. end]``, its first NEWLY-generated frame (index K) splices
directly onto the parent's branch frame — no frame double-plays. (A "still" child has
``context_drop = 0`` — nothing dropped, today's behavior byte-identical.)

``movie.json`` sidecar records the full node list + per-joint
``{branch_frame, trim_frames, mode, context_frames}`` (``mode`` labels each splice
"still" vs "vace_extend" so the UI can show it honestly) + a DRIFT note.

PER-SEGMENT WORKER OFFLOAD. Each segment renders through the SHARED ``render_clip``
primitive (the same one the single-clip bus job uses), so a REAL-model segment is
DELEGATED to the studio GPU worker (``HUGPY_STUDIO_WORKER``) — kicked off, polled,
its progress nested into this movie's per-segment progress, its cancel relayed from
the MOVIE's ``is_cancelling``, then INGESTED from the SHARED content-addressed path —
exactly like a single clip, while a SYNTHETIC segment keeps rendering IN-PROCESS
(cheap, correct, no model load). This is what makes a REAL id-movie possible from a
GPU-less central: the segments execute on the worker's GPU. Each delegated segment
uses a DISTINCT worker render id (``<job>.s<NN>.<run-nonce>``) since a movie posts
many renders under its one bus job (the worker keys by that id; a per-run nonce also
avoids replaying a stale worker-side terminal across a bus retry). RESUME is
unchanged: content-addressed reuse still lives inside ``produce_clip`` — for a
delegated segment it runs ON the worker over the SHARED store, so a re-run's completed
segments come back ``resumed=True`` fast without re-rendering. The per-segment
branch-frame / context-frame stills are written under this movie's ``out_root`` on the
SHARED volume, so the worker reads them directly (no upload).

DRIFT / v0 simplifications (also in ``studio_movie_schema``'s header + the report):
  * one movie-level tier (``vram_budget_gb``) applied to every segment;
  * geometry (w/h/fps) uniform across segments (they concat into one row).
    On a GPU-less box with NO worker configured the synthetic tier renders every
    segment fine; a REAL segment there surfaces the graceful per-segment Err.

Pure discipline (map §6): EXPECTED failures are returned as
``JobResult(ok=False, JobError(...))`` — DATA, never a raise. The studio-spine
import stays LAZY (``run_produce_clip`` does its own lazy studio imports); nothing
heavy is pulled at this module's import time, so it can never break app boot.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, replace

from hugpy_platform.binaries import resolve_bin
from hugpy_platform.constants import DEFAULT_ROOT
from hugpy_platform.utils import slugify

from hugpy_video.intel.media_store import ingest
from hugpy_video.intel.result_schema import JobError, JobResult
from hugpy_video.intel.studio.job import make_studio_i2v
# PER-SEGMENT capability resolution (k58): the ONE definition of "what does segment N
# render" plus the movie-pin-binds-only-what-it-serves rule. Read from there rather
# than restated here, so the submit-time preflight (video_routes) and this render can
# never disagree about what will be asked for. studio.movie_plan is registry-only
# (no torch/numpy), so this import stays boot-safe.
from hugpy_video.intel.studio.movie_plan import resolve_segment_model, segment_capability
from hugpy_video.intel.studio_movie_schema import StudioMovieSpec
# The studio-spine boundary. studio_i2v's module top is dependency-light (its
# studio/numpy imports are lazy INSIDE its functions), so importing these here — and
# thus at app boot via runners/__init__ — can never break boot.
#   * render_clip is the SHARED render primitive: the delegate-or-inline decision ladder
#     + the delegation loop (kickoff, progress forward, cancel relay, timeout, shared-path
#     ingest, error translation) the single-clip bus job uses, so a REAL movie segment
#     delegates to the GPU worker (HUGPY_STUDIO_WORKER) exactly like a single clip while a
#     SYNTHETIC segment renders inline. It returns a normalized ClipOutcome (Artifact
#     fields on Ok, an already-translated bus JobError on Err).
#   * run_produce_clip stays imported (and is passed to render_clip as its inline
#     ``produce``) because it is THIS module's patchable render seam — the movie tests
#     fake ``studio_movie.run_produce_clip`` to drive controlled clips through the inline
#     path with no GPU. It builds env+seeds from a StudioI2VSpec and calls produce_clip
#     (content-addressed resume inside).
from hugpy_video.intel.runners.studio_i2v import render_clip, run_produce_clip

logger = logging.getLogger(__name__)

# Studio movies land under the media-store root (inside ingest's storage jail) so
# every segment clip + the final movie.mp4 is cataloged like any other media output.
STUDIO_MOVIE_ROOT = os.path.join(DEFAULT_ROOT, "video_intel", "studio_movies")

_DRIFT_NOTE = ("still-mode splices condition each segment on ONE frame of its parent — "
               "motion is NOT carried across the splice. A joint's "
               "joint_mode='vace_extend' carries motion via VACE-extend (conditioning on "
               "the parent's trailing context_frames through the diffusers video+mask "
               "extend idiom) instead of a single still. A joint_mode='cut' is a HARD "
               "scene cut: no frame carry, the parent plays in full. With movie-level "
               "reference_images set (an identity movie) every segment renders id_lock so "
               "the subject carries across scene changes even though no pixels do — on the "
               "VACE path the references win the conditioning (an i2v branch still is "
               "accepted but unused; it governs only the parent trim).")

# The SHARED Wan-VACE budget floor. Two segment kinds route through the VACE path
# (Task.VACE_CONTROL), which is served ONLY by real Wan-VACE models — the cheapest,
# wan2.1-vace-1.3b, needs ~6GB (INT8 @ <=480p):
#   * a ``vace_extend`` splice (motion-carry across a join), and
#   * EVERY segment of an IDENTITY MOVIE (movie-level ``reference_images`` -> capability
#     ``id_lock`` -> Task.VACE_CONTROL reference-to-video).
# A plain still/i2v/t2v segment stays on the movie's (often tiny/synthetic) budget, so a
# VACE-bound segment RAISES its own per-segment budget to this floor to actually REACH the
# VACE model — never a silent downgrade (that dishonesty is banned). On a GPU-less box the
# render then returns the VACE runner's graceful DEPS_MISSING/NO_GPU/WEIGHTS_MISSING (an
# honest per-segment Err), not a synthetic clip — IDENTICAL to the single-clip id_lock
# path. There is NO synthetic id_lock/VACE tier BY DESIGN (no synthetic model declares
# id_lock), so an identity movie is a real-box render; the honest GPU-less result is a
# graceful Err. NOTE: vace-1.3b tops out at 480p, so a movie wider/taller than 832x480
# surfaces an honest VRAM_EXCEEDED (a bigger VACE model needs a bigger movie budget).
_VACE_MIN_BUDGET_GB = 6.0

# The two sidecars that make a movie DIR a resumable session (k91). ``movie.json`` is
# the manifest the runner (re)writes after every segment — what HAS been rendered.
# ``spec.json`` is the SUBMITTED REQUEST, persisted at enqueue time — what was ASKED
# for. They answer different questions and only the pair is enough to resume: the
# manifest alone cannot rebuild a spec (it records outcomes, not the goal tree), and
# the spec alone cannot say what is already done. Content-addressed reuse inside
# ``produce_clip`` then makes a re-enqueue of the persisted spec skip every completed
# segment (``resumed=True``), which is what "resume" physically is here — there is no
# checkpoint file, the clips ARE the checkpoint.
MOVIE_MANIFEST_FILENAME = "movie.json"
MOVIE_SPEC_FILENAME = "spec.json"

# Manifest ``status`` vocabulary (a projection of the run, NOT a second source of
# truth — segments_completed/segments_total stay authoritative for counts):
#   * "running"  — a job is in flight over this dir (the runner set it at start);
#   * "partial"  — the run stopped with some segments done (cancel/error/pause);
#   * "paused"   — an operator PAUSED it (POST .../pause): the in-flight job was
#                  cancelled cooperatively and the dir is waiting to be resumed;
#   * "done"     — every segment rendered and the movie assembled.
# "paused" is deliberately distinct from "partial": both are resumable, but only one
# was the operator's choice, and a session list that cannot tell them apart cannot
# say which movies are waiting on a person.
MOVIE_STATUS_RUNNING = "running"
MOVIE_STATUS_PARTIAL = "partial"
MOVIE_STATUS_PAUSED = "paused"
MOVIE_STATUS_DONE = "done"


def _ensure_shared_dir(path: str) -> None:
    """makedirs + the GROUP-WRITABLE (2775/setgid) mode the shared studio tree needs.

    Factored out of the runner's inline hotfix so the SUBMIT-time spec.json write lands
    with the same permissions: central (uid 1000) creates these dirs but DELEGATED
    segments are written into them by the WORKER's uid through the shared llmstorage
    group, and a default-umask 0755 dir EPERMs that write. A chmod refusal (tests, tmp,
    a non-shared root) is harmless and swallowed."""
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o2775)
    except OSError:
        pass


def movie_root_for(spec: StudioMovieSpec, job_id: str) -> str:
    """The directory THIS movie's segments, sidecars and assembled mp4 live in.

    The ONE definition of that path. The submit route persists ``spec.json`` here
    BEFORE the runner ever claims the job, and the session/pause/resume routes read
    both sidecars back out of it, so "where does this movie live" must not be an
    expression that exists in two places waiting to drift.

    The DIR is: an explicit ``out_root`` else the default studio-movies root, with a
    per-movie LEAF resolved in this order:

      1. ``spec.session_id`` — the RESUME pin. A resume mints a NEW bus job id, and for
         an UNNAMED movie the leaf would otherwise BE that job id, so every resume would
         silently relocate the dir: ``produce_clip`` looks for a completed segment's clip
         by PATH (``<dir>/segment_NN/<content_hash>/clip.mp4``), so a moved dir finds
         nothing and re-renders the entire movie while reporting success. Pinning the
         leaf is what makes "the clips are the checkpoint" actually true (k91). Validated
         as a bare directory leaf at spec construction, never a caller-shaped path.
      2. the slugified project NAME, when the caller named the movie — so a named
         session is findable by name (the historical rule, unchanged).
      3. the bus job id — the historical fallback for an unnamed, never-resumed movie.
    """
    leaf = spec.session_id or (slugify(spec.project) if spec.project else job_id)
    return os.path.join(
        os.path.abspath(spec.out_root) if spec.out_root else STUDIO_MOVIE_ROOT,
        leaf)


def write_movie_spec(movie_root: str, spec: StudioMovieSpec, job_id: str) -> "str | None":
    """Persist the FULL submitted spec as ``<movie_root>/spec.json`` and return its path
    (None if it could not be written).

    Called at SUBMIT time (the route, before the job is claimed) and again by the runner
    at start, so a movie enqueued by any path is resumable — including one that dies
    during segment 0, which is precisely the run that has no manifest yet and is
    therefore invisible to every recovery that reads only ``movie.json``.

    The envelope wraps the spec rather than dumping it bare so the file can carry the
    CURRENT job id: pause has to cancel the job that is actually running, and after a
    resume that is a NEW job id — a bare spec dump would leave the only recorded id
    pointing at a terminal run. ``job_id`` is rewritten on every submit/resume for
    exactly that reason. Best-effort — a sidecar write must never fail an enqueue."""
    payload = {
        "kind": "studio_movie_spec",
        "version": 1,
        "job_id": job_id,
        "submitted_at": time.time(),
        "spec": asdict(spec),
    }
    path = os.path.join(movie_root, MOVIE_SPEC_FILENAME)
    try:
        _ensure_shared_dir(movie_root)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path
    except Exception as exc:  # noqa: BLE001 — best-effort sidecar, never fatal
        logger.warning("studio movie %s: spec.json write FAILED (non-fatal): %s: %s",
                       job_id, type(exc).__name__, exc)
        return None


def read_movie_spec(movie_root: str) -> "dict | None":
    """The persisted submit envelope (``{kind, version, job_id, submitted_at, spec}``)
    for a movie dir, or None when there is none / it is unreadable. Total by
    construction: the session list calls this for every dir on the root and a single
    corrupt sidecar must not 500 the listing."""
    try:
        with open(os.path.join(movie_root, MOVIE_SPEC_FILENAME), "r",
                  encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("spec"), dict) else None


def read_movie_manifest(movie_root: str) -> "dict | None":
    """The ``movie.json`` manifest for a movie dir, or None (missing/unreadable). Same
    total-by-construction contract as ``read_movie_spec`` — a movie whose first segment
    has not landed yet simply has no manifest, which is a normal state, not an error."""
    try:
        with open(os.path.join(movie_root, MOVIE_MANIFEST_FILENAME), "r",
                  encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def mark_movie_status(movie_root: str, status: str, **extra) -> "dict | None":
    """Read-modify-write ``movie.json``'s ``status`` (plus a ``<status>_at`` stamp and
    any ``extra`` fields) and return the updated manifest, or None if it could not be
    written.

    Used by PAUSE/RESUME, which change what a session IS without re-rendering anything,
    so they must not touch the segment records the runner owns. When there is no
    manifest yet — a movie paused during segment 0 — a MINIMAL one is created rather
    than silently doing nothing: a paused session that does not appear as paused is the
    same invisible-work failure this whole slice exists to end. The runner's next
    ``_write_movie_json`` overwrites the stub wholesale, so the stub can never outlive
    the real thing."""
    manifest = read_movie_manifest(movie_root)
    if manifest is None:
        manifest = {"kind": "studio_movie", "segments": [], "joints": [],
                    "assembly": {"movie": None, "total_frames": 0},
                    "segments_completed": 0, "segments_total": None, "partial": True}
    manifest["status"] = status
    manifest[f"{status}_at"] = time.time()
    manifest.update(extra)
    try:
        _ensure_shared_dir(movie_root)
        with open(os.path.join(movie_root, MOVIE_MANIFEST_FILENAME), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
    except Exception as exc:  # noqa: BLE001 — best-effort, mirrors _write_movie_json
        logger.warning("studio movie: marking %s status=%s FAILED (non-fatal): %s: %s",
                       movie_root, status, type(exc).__name__, exc)
        return None
    return manifest


def _seg_budget(movie_budget, floor: float):
    """Per-segment vram budget for a VACE-path segment (id_lock / vace_extend).

    * An EXPLICIT movie budget is FLOORED to the VACE minimum so a real VACE model can
      actually bind (today's behavior, byte-identical) — never a silent downgrade.
    * A BLANK (None) movie budget stays None: AUTOFIT flows through to ``render_clip``,
      which sizes it to the serving worker's MEASURED free VRAM. The floor is moot there —
      a real box's free VRAM already clears 6GB, and a too-small box honestly Errs rather
      than pretending the floor is available (the exact guaranteed-fail the operator hates).
    """
    return None if movie_budget is None else max(movie_budget, floor)


def _bound_model_id(clip_path: "str | None") -> "str | None":
    """The model a finished segment ACTUALLY bound, read from the ``manifest.json``
    sidecar every studio runner writes beside its content-addressed clip.

    The point is attribution (k58): when a movie-level pin does not serve a segment's
    capability the router picks the model, and "the router picked something" is not an
    answer — movie.json should name WHICH. Best-effort by construction: a missing or
    unreadable sidecar simply leaves the requested pin in place rather than failing a
    rendered segment over a metadata read."""
    if not clip_path:
        return None
    try:
        with open(os.path.join(os.path.dirname(clip_path), "manifest.json"),
                  "r", encoding="utf-8") as fh:
            mid = json.load(fh).get("model_id")
    except (OSError, ValueError):
        return None
    return mid if isinstance(mid, str) and mid else None


# --------------------------------------------------------------------------- #
# ffmpeg helpers — a frame-accurate branch pluck, a frame-accurate trim, and the
# concat-demux stitch. Each is errors-as-data (returns (ok, stderr_tail)) except
# the concat, which RAISES so the caller wraps it into a movie_assembly_failed
# JobError (never a raise across the job boundary).
# --------------------------------------------------------------------------- #
def _extract_frame_at(clip_path: str, frame_index: int, dest_png: str) -> "tuple[bool, str]":
    """Pluck the ``frame_index``-th (0-based) frame of ``clip_path`` to ``dest_png``.

    Frame-accurate via the ``select=eq(n\\,IDX)`` filter + ``-frames:v 1`` (the
    existing ``ffmpeg_frames`` helper is an fps RESAMPLE, not an index pluck, so it
    is unsuitable — this is the small dedicated extractor the header points at).
    Never raises on a plain ffmpeg failure (errors-as-data). Returns
    (ok, stderr_tail)."""
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(os.path.dirname(dest_png), exist_ok=True)
    # -vsync 0 / -frames:v 1 with a select that keeps only frame IDX yields exactly
    # that one still. -q:v 2 = high-quality mjpeg/png quantizer.
    cmd = [
        ffmpeg, "-y",
        "-i", clip_path,
        "-vf", f"select=eq(n\\,{int(frame_index)})",
        "-vsync", "0",
        "-frames:v", "1",
        "-q:v", "2",
        dest_png,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = (result.returncode == 0 and os.path.isfile(dest_png)
          and os.path.getsize(dest_png) > 0)
    return ok, (result.stderr or "")[-500:]


def _extract_context_frames(
    clip_path: str, branch_index: int, k: int, dest_dir: str
) -> "tuple[list[str] | None, str, list[int]]":
    """Pluck the parent clip's TRAILING context window for a ``vace_extend`` splice:
    frames ``[branch_index-k+1 .. branch_index]`` (inclusive, oldest -> newest), CLAMPED
    at 0. Writes them as ordered PNGs (``ctx_000.png`` = oldest) into ``dest_dir`` via the
    same frame-accurate ``select=eq(n,IDX)`` pluck the branch still uses.

    The number extracted is ``min(k, branch_index + 1)`` — a branch too early to have k
    frames behind it yields the fewer frames actually available (never a crash). Returns
    ``(paths | None, stderr_tail, indices)``; None signals an errors-as-data ffmpeg
    failure. ``indices`` is the exact 0-based source frame indices extracted (in order),
    so the caller/manifest records precisely which parent frames carried the motion."""
    start = max(0, int(branch_index) - int(k) + 1)
    indices = list(range(start, int(branch_index) + 1))   # inclusive, oldest -> newest
    os.makedirs(dest_dir, exist_ok=True)
    paths: "list[str]" = []
    for pos, idx in enumerate(indices):
        dest = os.path.join(dest_dir, f"ctx_{pos:03d}.png")
        ok, tail = _extract_frame_at(clip_path, idx, dest)
        if not ok:
            return None, tail, indices
        paths.append(dest)
    return paths, "", indices


def _trim_clip(src_mp4: str, dst_mp4: str, n_frames: int, fps: int) -> "tuple[bool, str]":
    """Re-encode the FIRST ``n_frames`` frames of ``src_mp4`` into ``dst_mp4``.

    Frame-accurate via ``-frames:v n`` on decoded output. Re-encodes with the SAME
    house H.264/yuv420p invocation the synthetic runner uses, so EVERY contribution
    clip (trimmed parents AND the untrimmed leaf, which is also routed through here
    with n = its full length) carries identical codec params — the concat DEMUXER's
    ``-c copy`` is then valid. Never raises on a plain ffmpeg failure. Returns
    (ok, stderr_tail)."""
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-i", src_mp4,
        "-frames:v", str(int(n_frames)),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(int(fps)),
        "-movflags", "+faststart",
        dst_mp4,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = (result.returncode == 0 and os.path.isfile(dst_mp4)
          and os.path.getsize(dst_mp4) > 0)
    return ok, (result.stderr or "")[-500:]


def _slice_clip(src_mp4: str, dst_mp4: str, start_frame: int, n_frames: int,
                fps: int) -> "tuple[bool, str]":
    """Re-encode a WINDOW ``[start_frame, start_frame+n_frames)`` of ``src_mp4`` into
    ``dst_mp4`` — the vace_extend head-drop path (drop a child's reconstructed context
    prefix). Frame-accurate via a ``select='gte(n,START)'`` filter + ``setpts`` reset,
    then ``-frames:v n`` on the kept stream. Same house H.264/yuv420p invocation as
    ``_trim_clip`` so every contribution shares codec params (concat ``-c copy`` stays
    valid). For ``start_frame <= 0`` it is exactly ``_trim_clip`` (first-n), so the
    still-mode path never changes. Never raises on a plain ffmpeg failure."""
    if int(start_frame) <= 0:
        return _trim_clip(src_mp4, dst_mp4, n_frames, fps)
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    s = int(start_frame)
    # select drops the leading window, setpts rebases timestamps to 0; -frames:v then
    # counts the KEPT frames. NOTE: no ``-vsync 0`` here — it conflicts with the CFR
    # ``-r`` on this ffmpeg (6.1) and aborts ("Invalid argument"); the default vsync +
    # ``-r`` gives a clean CFR clip whose codec params match _trim_clip's (concat-safe).
    vf = (f"select='gte(n\\,{s})',setpts=PTS-STARTPTS,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    cmd = [
        ffmpeg, "-y",
        "-i", src_mp4,
        "-vf", vf,
        "-frames:v", str(int(n_frames)),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(int(fps)),
        "-movflags", "+faststart",
        dst_mp4,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = (result.returncode == 0 and os.path.isfile(dst_mp4)
          and os.path.getsize(dst_mp4) > 0)
    return ok, (result.stderr or "")[-500:]


def _concat_clips(contribution_mp4s: "list[str]", movie_mp4: str, work_dir: str) -> None:
    """Stitch the contribution mp4s into ``movie_mp4`` via ffmpeg's concat DEMUXER
    (stream-copy — all inputs went through ``_trim_clip`` so they share codec params,
    making ``-c copy`` valid + fast). Mirrors ``runners.movie._concat_movie``. RAISES
    on failure — the caller wraps it into a movie_assembly_failed JobError."""
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    os.makedirs(work_dir, exist_ok=True)
    list_path = os.path.join(work_dir, "concat_list.txt")
    with open(list_path, "w") as fh:
        for p in contribution_mp4s:
            safe = p.replace("'", "'\\''")   # concat-demux single-quote escaping
            fh.write(f"file '{safe}'\n")
    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        movie_mp4,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.isfile(movie_mp4):
        raise RuntimeError(
            f"ffmpeg concat failed rc={result.returncode}: {(result.stderr or '')[-500:]}")


# --------------------------------------------------------------------------- #
# assembly — trim each parent at its child's branch point (metadata honored at
# concat time only), leaf plays full, concat -> movie.mp4. Best-effort / never
# raises across the job boundary (a stitch hiccup can only lose the partial, never
# mask the caller's real error).
# --------------------------------------------------------------------------- #
def _assemble_movie(movie_root: str, work_dir: str, seg_records: "list[dict]",
                    fps: int, job_id: str) -> "dict":
    """(Re)stitch whatever segments are complete SO FAR into
    ``<movie_root>/movie.mp4``, honoring each parent's TRIM at its child's branch
    point. Returns an ``assembly`` dict ``{movie, total_frames, joints}`` (movie is
    None when nothing could be stitched). NEVER raises — a stitch failure is logged
    and swallowed so a partial save is additive only."""
    completed = [r for r in seg_records if r["status"] in ("done", "resumed")]
    assembly = {"movie": None, "total_frames": 0, "joints": []}
    if not completed:
        return assembly

    # Per-joint trim record + each segment's contribution WINDOW [head_drop, tail_end).
    #   * tail_end (the PARENT trim): a segment with a NEXT completed segment (its child)
    #     plays to child.resolved_branch + 1 for a still/vace_extend child; a "cut" child
    #     carries NO frame, so its parent plays in FULL (tail_end = the parent's frames — no
    #     trim). The last (leaf) segment plays FULL.
    #   * head_drop (vace_extend only): a segment RENDERED via vace_extend has its first
    #     ``context_drop`` frames as a RECONSTRUCTION of its parent's context (the kept
    #     mask=0 prefix), which overlaps the parent's tail — so drop them from this segment's
    #     head (context_drop=0 for a still/cut segment, so it plays [0, tail_end) exactly as
    #     before). The joint records the CHILD's splice ``mode`` so the UI can label it
    #     "still" / "vace_extend" / "cut" honestly.
    # (segment clip path, head_drop, n_frames)
    contributions: "list[tuple[str, int, int]]" = []
    joints: "list[dict]" = []
    for p, rec in enumerate(completed):
        head_drop = int(rec.get("context_drop") or 0)   # vace_extend reconstructed prefix
        if p + 1 < len(completed):
            child = completed[p + 1]
            child_mode = child.get("joint_mode", "still")
            if child_mode == "cut":
                # SCENE CUT: no frame carry -> the parent plays in FULL (no trim). The joint
                # records mode="cut" with branch_frame=None (a cut conditions on no frame) and
                # trim_frames = the parent's full length (the spliced-row math reads a joint
                # whose trim == full as an untrimmed block; see movieTimeline.deriveRow).
                tail_end = int(rec["frames"])
                joints.append({
                    "parent_segment_id": rec["segment_id"],
                    "child_segment_id": child["segment_id"],
                    "branch_frame": None,
                    "trim_frames": tail_end,
                    "mode": "cut",
                    "context_frames": None,
                })
            else:
                rb = child["resolved_branch"]          # branch INTO this segment
                tail_end = int(rb) + 1
                joints.append({
                    "parent_segment_id": rec["segment_id"],
                    "child_segment_id": child["segment_id"],
                    "branch_frame": int(rb),
                    "trim_frames": tail_end,
                    # SPLICE MODE (child's): "still" or "vace_extend"; context_frames is the
                    # child's kept-context length (None for a still splice). The UI labels the
                    # splice from these — motion-carry is never silent.
                    "mode": child_mode,
                    "context_frames": (child.get("context_frames")
                                       if child_mode == "vace_extend" else None),
                })
        else:
            tail_end = int(rec["frames"])          # leaf: full clip
        contributions.append((rec["clip_path"], head_drop, tail_end - head_drop))

    # Materialize each contribution as a uniformly re-encoded WINDOW in the work dir,
    # then concat. Non-destructive: the source clips are never touched. A head_drop>0
    # (vace_extend) window goes through _slice_clip; head_drop==0 is the historical
    # first-n _trim_clip (byte-identical still-mode path).
    os.makedirs(work_dir, exist_ok=True)
    contrib_paths: "list[str]" = []
    total = 0
    try:
        for i, (src, head_drop, n) in enumerate(contributions):
            dst = os.path.join(work_dir, f"contrib_{i:02d}.mp4")
            ok, tail = _slice_clip(src, dst, head_drop, n, fps)
            if not ok:
                logger.warning("studio movie %s: contribution slice %d FAILED "
                               "(non-fatal): %s", job_id, i, tail)
                # keep whatever we could stitch before this failure
                break
            contrib_paths.append(dst)
            total += n
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        if len(contrib_paths) == 1:
            shutil.copyfile(contrib_paths[0], movie_mp4)
            assembly["movie"] = "movie.mp4"
            assembly["total_frames"] = total
        elif len(contrib_paths) >= 2:
            _concat_clips(contrib_paths, movie_mp4, work_dir)
            assembly["movie"] = "movie.mp4"
            assembly["total_frames"] = total
    except Exception as exc:  # a stitch failure must NEVER mask the caller's error
        logger.warning("studio movie %s: assembly FAILED (non-fatal): %s: %s",
                       job_id, type(exc).__name__, exc)
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        if os.path.isfile(movie_mp4):
            assembly["movie"] = "movie.mp4"     # keep an earlier good stitch if any
    assembly["joints"] = joints
    return assembly


# --------------------------------------------------------------------------- #
# k120 slice 2 — PRODUCER CONTINUITY REFRESH (advisory, opt-in).
#
# Between segments, when spec.continuity_refresh is set: describe the frame the
# previous segment ACTUALLY ended on (the vision plane, movie.py's judge idiom),
# then have a text model rewrite the next segment's prompt so it opens from that
# real state, keeping the authored action. Advisory discipline throughout: any
# failure returns (None, why) and the authored prompt renders unchanged — the
# refresh may improve a movie, it must never lose one.
#
# RESUME DETERMINISM (k91: "the clips are the checkpoint"): the refreshed prompt
# is a content-hash input of the clip, so a re-enqueued movie must REUSE the
# persisted rewrite from movie.json rather than re-rolling a different one —
# otherwise every refreshed segment re-renders from scratch on resume.
# --------------------------------------------------------------------------- #
_REFRESH_TEXT_MODEL = "Qwen2.5-7B-Instruct-GGUF"   # explicit — never the silent 3B task default
_REFRESH_DESC_TOKENS = 140
_REFRESH_REWRITE_TOKENS = 260


def _result_text(res) -> str:
    """Best-effort reply text from an execute_prompt result (movie.py idiom)."""
    txt = getattr(res, "text", None)
    if txt:
        return txt
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(res, attr, None)
        if callable(fn):
            try:
                d = fn()
            except TypeError:
                continue
            if isinstance(d, dict) and d.get("text"):
                return d["text"]
    return str(res)


def _persisted_refresh(movie_root: str, segment_id: str) -> "str | None":
    """The rewrite movie.json already recorded for this segment, or None."""
    try:
        with open(os.path.join(movie_root, "movie.json"), encoding="utf-8") as f:
            data = json.load(f)
        for rec in (data.get("segments") or []):
            if (rec.get("segment_id") == segment_id and rec.get("refresh_note")
                    and rec.get("prompt")):
                return str(rec["prompt"])
    except Exception:  # noqa: BLE001 — no movie.json / unreadable == no persisted rewrite
        return None
    return None


def _continuity_refresh(prev_frame_png: str, prev_prompt: "str | None",
                        next_prompt: str, model: "str | None" = None,
                        ) -> "tuple[str | None, str]":
    """(refreshed_prompt, note) — or (None, why-not). Never raises, never blocks."""
    try:
        from hugpy_video.intel.plane import execute_prompt
        from hugpy_platform.async_runtime import run
        from hugpy_engine.utils.no_think import with_no_think, strip_think

        desc_ask = (
            "This is the final frame of a film shot. Describe it in 2-3 factual "
            "sentences for the director of the NEXT shot: the subject(s) and their "
            "exact appearance, their position/pose, the setting, the lighting, and "
            "the camera framing. Only the description."
        )
        res = run(execute_prompt(task="image-text-to-text", file=prev_frame_png,
                                 prompt=with_no_think(desc_ask),
                                 max_new_tokens=_REFRESH_DESC_TOKENS))
        if not getattr(res, "ok", True):
            return None, f"vision describe not-ok: {getattr(res, 'error', None)}"
        desc, _ = strip_think(_result_text(res))
        desc = (desc or "").strip()
        if not desc:
            return None, "vision describe returned no text"

        rewrite_ask = (
            "You are a film director keeping continuity between two shots.\n"
            "The previous shot ACTUALLY ended like this:\n" + desc + "\n\n"
            + ("Previous shot's prompt:\n" + prev_prompt + "\n\n" if prev_prompt else "")
            + "The next shot's PLANNED prompt:\n" + next_prompt + "\n\n"
            "Rewrite the next shot's prompt so it opens exactly from the state the "
            "previous shot ended in (same subject appearance, position, setting, "
            "light) while keeping the planned action and intent. Self-contained, "
            "visual, present tense. Return ONLY the rewritten prompt text."
        )
        res2 = run(execute_prompt(task="text-generation",
                                  model_key=(model or _REFRESH_TEXT_MODEL),
                                  prompt=with_no_think(rewrite_ask),
                                  max_new_tokens=_REFRESH_REWRITE_TOKENS))
        if not getattr(res2, "ok", True):
            return None, f"rewrite not-ok: {getattr(res2, 'error', None)}"
        refreshed, _ = strip_think(_result_text(res2))
        refreshed = (refreshed or "").strip().strip('"')
        # Sanity: an empty or wildly bloated rewrite is worse than the authored
        # prompt — keep the author's text and say why.
        if len(refreshed) < 20:
            return None, f"rewrite too short ({len(refreshed)} chars) — kept authored prompt"
        if len(refreshed) > max(1200, 4 * len(next_prompt)):
            return None, "rewrite implausibly long — kept authored prompt"
        return refreshed, "opened from previous frame: " + desc[:200]
    except Exception as exc:  # noqa: BLE001 — advisory: any plane trouble keeps the authored prompt
        return None, f"refresh skipped ({type(exc).__name__}: {exc})"


def _write_movie_json(movie_root: str, spec: StudioMovieSpec, seg_records: "list[dict]",
                      assembly: "dict", job_id: str, partial: bool) -> "dict":
    """(Over)write ``<movie_root>/movie.json`` — the full node list + per-joint
    ``{branch_frame, trim_frames, mode, context_frames}`` (``mode`` labels each splice
    "still" vs "vace_extend"; ``context_frames`` is the vace kept-context length) +
    assembly + drift note. Returns the manifest dict (for ``JobResult.movie``).
    Best-effort — never raises across the job boundary.

    Also carries the three SESSION fields the movies list + pause/resume read (k91):
    the human ``title``, the ``job_id`` of the run that wrote this manifest, and a
    coarse ``status``. ``status`` is DERIVED from ``partial`` here rather than tracked
    separately — the runner only ever writes running/partial/done — while PAUSED is
    written exclusively by ``mark_movie_status`` from the pause route. That split keeps
    one writer per state: the runner cannot accidentally un-pause a session, and pause
    cannot invent a completion."""
    manifest = {
        "kind": "studio_movie",
        "drift": _DRIFT_NOTE,
        # SESSION identity: the operator's own name for this work (None when unnamed —
        # the UI warns and offers to name it) and the bus job that produced this state.
        "title": spec.title,
        "project": spec.project,
        "job_id": job_id,
        "status": (MOVIE_STATUS_PARTIAL if partial else MOVIE_STATUS_DONE),
        "updated_at": time.time(),
        "fps": spec.fps,
        "width": spec.width,
        "height": spec.height,
        "vram_budget_gb": spec.vram_budget_gb,
        # IDENTITY LOCK: the movie-level subject references (empty for a plain movie). When
        # non-empty every segment rendered capability id_lock (see each segment's capability).
        "id_lock": bool(spec.reference_images),
        "reference_images": list(spec.reference_images),
        "segments": seg_records,
        "joints": assembly.get("joints", []),
        "assembly": {
            "movie": assembly.get("movie"),
            "total_frames": assembly.get("total_frames", 0),
        },
        "partial": partial,
        "segments_completed": len([r for r in seg_records if r["status"] in ("done", "resumed")]),
        "segments_total": len(spec.goals),
    }
    try:
        with open(os.path.join(movie_root, "movie.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
    except Exception as exc:  # best-effort — never raise across the job boundary
        logger.warning("studio movie %s: movie.json write FAILED (non-fatal): %s: %s",
                       job_id, type(exc).__name__, exc)
    return manifest


# --------------------------------------------------------------------------- #
# the orchestrator
# --------------------------------------------------------------------------- #
def run_generate_studio_movie(spec: StudioMovieSpec, job_id: str) -> JobResult:
    started_at = time.time()
    from hugpy_video.intel.media_bus import is_cancelling, set_progress

    # Per-RUN nonce for the worker-side render ids of DELEGATED segments: a movie posts
    # many renders under its one bus job, so each segment needs a distinct worker key,
    # and a fresh nonce per invocation stops a bus RETRY of the movie from replaying a
    # prior attempt's stale worker-side terminal (resume still comes from produce_clip's
    # content-addressing on the SHARED store, NOT from the render id). Irrelevant to the
    # in-process path (synthetic segments never reach the worker).
    run_nonce = uuid.uuid4().hex[:8]

    seg_total = len(spec.goals)
    movie_root = movie_root_for(spec, job_id)
    work_dir = os.path.join(movie_root, "_assembly")
    # GROUP-WRITABLE out_root (2026-07-12 hotfix, first delegated id-movie):
    # central (uid 1000) creates these dirs, but DELEGATED segments are written
    # by the WORKER's uid (ae runs as 988) through the shared llmstorage group.
    # Default umask 022 yields 0755 dirs -> the worker EPERMs writing its clip
    # ("[Errno 13] Permission denied ... segment_00"). The canonical studio
    # tree is 2775/setgid by design; make what we create match it, so cross-box
    # writes work regardless of the service's umask. (_ensure_shared_dir.)
    _ensure_shared_dir(movie_root)
    _ensure_shared_dir(work_dir)
    # RESUMABILITY (k91): stamp the submitted spec + THIS job's id into the dir before
    # a single frame is rendered. The submit route already wrote this at enqueue; the
    # runner rewrites it because a movie can be enqueued by paths that are not that
    # route (a bus retry, a headless caller), and a session that cannot be re-enqueued
    # is a session that dies with whatever killed its job.
    write_movie_spec(movie_root, spec, job_id)
    mark_movie_status(movie_root, MOVIE_STATUS_RUNNING, job_id=job_id,
                      title=spec.title, project=spec.project, segments_total=seg_total)

    _tier = ("autofit" if spec.vram_budget_gb is None
             else f"<={spec.vram_budget_gb:.2f}GB")
    logger.info("studio movie %s: %d segment(s), %dx%d @ %dfps, tier vram %s",
                job_id, seg_total, spec.width, spec.height, spec.fps, _tier)

    seg_records: "list[dict]" = []   # movie.json / JobResult.movie segment nodes
    seg_refs: "list" = []            # per-segment clip MediaRefs, in order
    prev_clip_path: "str | None" = None   # the parent clip the next segment branches from
    prev_frames: int = 0                   # the parent clip's frame count
    prev_prompt: "str | None" = None       # the previous segment's EFFECTIVE prompt (k120)

    # live per-segment meta (for the nested progress blob)
    segments_meta = [
        {"index": i, "segment_id": g.segment_id, "prompt": g.prompt,
         "status": "pending", "resumed": None}
        for i, g in enumerate(spec.goals)
    ]

    # ---- BUS JOB HEARTBEAT (k91) ---------------------------------------------- #
    # A movie is ONE bus job that can run for HOURS across many segments, and it used
    # to publish a blob whose only /llm/jobs-visible field was stage="generating" for
    # the entire run. comms' movement clock (``Job.progressed_at``) advances on a status
    # transition, a numeric progress ADVANCE, or a stage change — none of which a
    # mid-movie tick produced. So ``comms/jobs.py::expire_pending_orphans`` did exactly
    # what it says on the tin ("no forward progress for 900s -> wedged") and RETIRED the
    # bus record of a LIVE 14-segment render. The runner kept rendering into a job row
    # nobody believed in.
    #
    # Every _emit now also carries:
    #   * ``progress`` — a MONOTONIC 0..1 fraction folding completed segments AND the
    #     in-flight segment's own denoise step, so even a single 46-minute segment moves
    #     the number roughly every poll instead of once per segment;
    #   * ``message``  — "segment 2/14 - step 18/32", the human line /llm/jobs shows.
    # ``job_bridge.on_progress`` relays both and stamps ``progressed_at`` when the
    # authored message changes, so a movie that is rendering can no longer be mistaken
    # for a movie that is wedged. The nested per-segment blob below is UNCHANGED — these
    # are additive keys, so every existing console reader is untouched.
    #
    # Monotonicity is enforced by a floor rather than trusted: the fraction is derived
    # from a worker-reported step counter that can restart (a segment retried, a resumed
    # segment reporting nothing), and a progress number that goes BACKWARDS is not read
    # as movement by the JobStore's advance rule — it would silently stop heartbeating
    # at the worst possible moment.
    progress_floor = [0.0]

    def _fraction_in_segment(current: "dict | None") -> float:
        """0..1 through the IN-FLIGHT segment, from whatever counter is visible in the
        nested worker/render blob (``step``/``steps``). 0.0 when nothing is countable —
        queued, resumed, or a runner that reports no step — which is honest: no step
        information means no evidence of within-segment movement."""
        if not isinstance(current, dict):
            return 0.0
        blob = current.get("worker")
        if not isinstance(blob, dict):
            return 0.0
        step, steps = blob.get("step"), blob.get("steps")
        if not isinstance(step, (int, float)) or not isinstance(steps, (int, float)):
            return 0.0
        if steps <= 0:
            return 0.0
        return max(0.0, min(1.0, float(step) / float(steps)))

    def _human_message(stage: str, seg_done: int, current: "dict | None") -> str:
        """The operator-facing one-liner for /llm/jobs: "segment 2/14 - step 18/32"
        while a segment renders, the queue position while it waits, and the coarse
        stage + completed count otherwise. Segment numbers are 1-BASED here (and only
        here) because this string is read by a person counting shots, not by code."""
        idx = None
        if isinstance(current, dict) and isinstance(current.get("index"), int):
            idx = current["index"]
        if idx is None:
            return f"{stage} - {seg_done}/{seg_total} segment(s) complete"
        head = f"segment {idx + 1}/{seg_total}"
        blob = current.get("worker") if isinstance(current, dict) else None
        if isinstance(blob, dict):
            step, steps = blob.get("step"), blob.get("steps")
            if isinstance(step, (int, float)) and isinstance(steps, (int, float)) and steps:
                return f"{head} - step {int(step)}/{int(steps)}"
            if blob.get("phase") == "queued":
                pos = blob.get("position")
                return f"{head} - queued" + (f" (position {pos})" if pos is not None else "")
        return head

    def _emit(stage: str, current: "dict | None" = None) -> None:
        """Build + persist the NESTED movie progress blob (best-effort)."""
        seg_done = sum(1 for s in segments_meta if s["status"] in ("done", "resumed"))
        elapsed = time.time() - started_at
        eta = round((elapsed / seg_done) * (seg_total - seg_done), 2) if seg_done > 0 else None
        # Fold completed segments + the in-flight segment's step fraction into ONE
        # 0..1 number, clamped to a never-decreasing floor (see the note above).
        raw = (seg_done + _fraction_in_segment(current)) / seg_total if seg_total else 1.0
        fraction = max(progress_floor[0], min(1.0, raw))
        progress_floor[0] = fraction
        blob = {
            "stage": stage, "segment_done": seg_done, "segment_total": seg_total,
            "segments": segments_meta, "current": current,
            "started_at": started_at, "eta_s": eta,
            # The two heartbeat fields job_bridge relays into the comms Job record.
            "progress": fraction,
            "message": _human_message(stage, seg_done, current),
        }
        try:
            set_progress(job_id, blob)
        except Exception:
            logger.debug("studio movie %s: set_progress failed (non-fatal)", job_id, exc_info=True)

    def _save(partial: bool) -> "dict":
        """Best-effort iterative finalize: (re)stitch completed segments + rewrite
        movie.json, so an unfinishable movie still leaves a watchable partial +
        fresh manifest on disk."""
        assembly = _assemble_movie(movie_root, work_dir, seg_records, spec.fps, job_id)
        return _write_movie_json(movie_root, spec, seg_records, assembly, job_id, partial)

    def _partial_return(result: JobResult) -> JobResult:
        """Iterative partial save, THEN attach project + the partial manifest to a
        failure/cancel JobResult so the operator learns WHERE the partial landed.
        The ORIGINAL error is preserved verbatim (additive only)."""
        manifest = _save(partial=True)
        return replace(
            result,
            project={"name": spec.project, "uuid": job_id, "dir": movie_root},
            movie=manifest)

    should_cancel = lambda: is_cancelling(job_id)  # noqa: E731

    _emit("loading", None)

    for seg_i, goal in enumerate(spec.goals):
        # ---- between-segment cancel + time-budget checks (the bus won't) ----
        if is_cancelling(job_id):
            return _partial_return(JobResult(job_id, ok=False, error=JobError(
                code="cancelled",
                message=f"cancelled after {seg_i} of {seg_total} segment(s)",
                retryable=False)))
        if spec.time_budget_s is not None and (time.time() - started_at) > spec.time_budget_s:
            return _partial_return(JobResult(job_id, ok=False, error=JobError(
                code="time_budget_exceeded",
                message=(f"studio movie time budget {spec.time_budget_s}s exceeded after "
                         f"{seg_i} of {seg_total} segment(s)"),
                retryable=True)))

        seg_out_root = os.path.join(movie_root, f"segment_{seg_i:02d}")
        # Delegated segments are WRITTEN INTO this dir by the worker's uid —
        # same group-writability requirement as movie_root above (2775).
        os.makedirs(seg_out_root, exist_ok=True)
        try:
            os.chmod(seg_out_root, 0o2775)
        except OSError:
            pass

        # ---- decide this segment's conditioning + capability + joint mode ----
        # IDENTITY MOVIE (movie-level reference_images set): EVERY segment renders capability
        # id_lock (Wan-VACE reference-to-video) so the locked SUBJECT carries across scene
        # changes; the per-segment budget is raised to the shared VACE floor (id_lock routes
        # through the VACE path, exactly like vace_extend). The per-mode JOINT behavior is
        # UNCHANGED (a still joint still extracts + trims, a cut carries no frame) — only the
        # RENDER is now reference-conditioned. On the VACE path an i2v start_image is ACCEPTED
        # but UNUSED (the runner conditions on the references), so for an id-movie the
        # REFERENCES win; a still joint's branch frame governs only the parent TRIM at assembly.
        # Otherwise (PLAIN movie): segment 0 is i2v (movie start_image) else t2v; a later
        # segment splices onto its parent per goal.joint_mode:
        #   * "still": i2v conditioned on ONE branch frame (start_image). No motion carry.
        #   * "vace_extend": v2v (VACE) conditioned on the parent's TRAILING context frames.
        #   * "cut": a HARD scene cut — no frame carry; a FRESH render, parent plays in FULL.
        # PER-GOAL id_lock DNA (IDENTITY-3D-CONTINUITY-PLAN.md S2-movie): prefer THIS
        # segment's own references (a specific turntable VIEW of the identity the route
        # resolved from the goal's ``view``) over the movie-level set, so a ``cut`` into a
        # new scene can hold the SAME person while re-framing the camera per shot. A goal
        # with no per-goal refs (reference_images None) inherits the movie-level set —
        # byte-identical to today's every-segment behavior.
        #
        # THE CAPABILITY ITSELF is decided by ``studio.movie_plan.segment_capability``
        # (k58) — the SAME call the submit-time preflight walks the take-tree with, so a
        # movie that passed POST asks for exactly the capabilities that were checked. The
        # branch below still owns the CONDITIONING (branch still / context frames / budget
        # floor); it no longer re-derives the capability alongside it.
        id_refs = tuple(goal.reference_images or spec.reference_images or ())
        is_id_movie = bool(id_refs)
        capability = segment_capability(spec, goal, seg_i)
        resolved_branch = None
        start_image = None
        vace_context_frames = None
        seg_joint_mode = "still"
        seg_context_frames = 0          # how many parent frames carried the motion (K)
        seg_context_drop = 0            # frames DROPPED from THIS segment's head at assembly
        # An id-movie segment (id_lock) routes through VACE -> raise to the shared VACE floor
        # when the budget is EXPLICIT; a BLANK (None) movie budget flows through as autofit
        # (render_clip sizes it to the worker's free VRAM — the floor is moot there).
        seg_budget = (_seg_budget(spec.vram_budget_gb, _VACE_MIN_BUDGET_GB)
                      if is_id_movie else spec.vram_budget_gb)
        if seg_i == 0:
            # Root: id_lock in an id-movie (the references define the render); else i2v from
            # the movie start_image, else t2v. In an id-movie the movie start_image is
            # ACCEPTED but the VACE runner ignores it (references win) — carried for provenance.
            start_image = spec.start_image.uri if spec.start_image is not None else None
        elif goal.joint_mode == "cut":
            # SCENE CUT: no frame carry at all — no branch resolve / extraction. The child is a
            # FRESH render (id_lock in an id-movie so the subject carries, else t2v). The parent
            # plays in FULL: resolved_branch stays None so assembly does NOT trim it.
            seg_joint_mode = "cut"
            _emit("branching", {"segment_id": goal.segment_id, "index": seg_i,
                                "mode": "cut"})
        else:
            # still / vace_extend splice onto the parent at a branch frame.
            # branch_frame null -> the parent's LAST frame (prev_frames - 1).
            raw = goal.branch_frame
            resolved_branch = (prev_frames - 1) if raw is None else int(raw)
            # RUN-time bound check (the schema can't know the parent's real length):
            # a branch past the parent's frames is errors-as-data, never a crash.
            if resolved_branch < 0 or resolved_branch >= prev_frames:
                return _partial_return(JobResult(job_id, ok=False, error=JobError(
                    code="branch_frame_out_of_range",
                    message=(f"segment {seg_i} ({goal.segment_id!r}) branch_frame "
                             f"{raw!r} -> resolved {resolved_branch} is outside the "
                             f"parent clip's [0, {prev_frames}) frames"),
                    retryable=False)))
            seg_joint_mode = goal.joint_mode
            if seg_joint_mode == "vace_extend":
                # VACE-EXTEND: extract the parent's trailing K frames [branch-K+1 .. branch]
                # (clamped) and route the segment through the VACE path (capability "v2v" ->
                # Task.VACE_CONTROL). The child's first K output frames RECONSTRUCT this
                # context and are dropped at assembly (context_drop) so no frame double-plays.
                # RAISE the per-segment budget to the VACE floor so a real VACE model
                # actually binds (a still segment stays on the movie's tiny/synthetic
                # budget) — this is REQUIRED to reach the VACE path, NOT a silent downgrade.
                # In an id-movie the references ALSO ride along (identity + motion-carry both).
                k = goal.context_frames if goal.context_frames is not None else spec.context_frames
                ctx_dir = os.path.join(seg_out_root, "context")
                _emit("branching", {"segment_id": goal.segment_id, "index": seg_i,
                                    "branch_frame": resolved_branch,
                                    "mode": "vace_extend", "context_frames": k})
                paths, tail, idxs = _extract_context_frames(
                    prev_clip_path, resolved_branch, k, ctx_dir)
                if paths is None:
                    return _partial_return(JobResult(job_id, ok=False, error=JobError(
                        code="context_frame_extract_failed",
                        message=(f"segment {seg_i} ({goal.segment_id!r}, joint_mode=vace_extend): "
                                 f"could not extract context frames {idxs} from the parent "
                                 f"clip: {tail}"),
                        retryable=False)))
                vace_context_frames = tuple(paths)
                seg_context_frames = len(paths)     # actual K extracted = min(k, branch+1)
                seg_context_drop = len(paths)        # the child reconstructs these -> drop at assembly
                # EXPLICIT budget -> floored to the VACE minimum; BLANK (None) -> autofit
                # flows through (render_clip sizes it to the worker's free VRAM).
                seg_budget = _seg_budget(spec.vram_budget_gb, _VACE_MIN_BUDGET_GB)
            else:
                # STILL (default, backward-compatible): condition on ONE branch frame. In an
                # id-movie the render is id_lock (the references) + this branch still, which the
                # VACE runner ACCEPTS but IGNORES (references win) — the still governs only the
                # parent TRIM at assembly. In a plain movie it is the historical i2v.
                branch_png = os.path.join(seg_out_root, "branch.png")
                _emit("branching", {"segment_id": goal.segment_id, "index": seg_i,
                                    "branch_frame": resolved_branch, "mode": "still"})
                ok, tail = _extract_frame_at(prev_clip_path, resolved_branch, branch_png)
                if not ok:
                    return _partial_return(JobResult(job_id, ok=False, error=JobError(
                        code="branch_frame_extract_failed",
                        message=(f"segment {seg_i} ({goal.segment_id!r}): could not extract "
                                 f"branch frame {resolved_branch} from the parent clip: {tail}"),
                        retryable=False)))
                start_image = branch_png

        # ---- k120 slice 2: producer continuity refresh (opt-in, advisory) ----
        # The frame the previous segment ACTUALLY ended on describes itself to a
        # vision model; a text model then rewrites THIS segment's prompt to open
        # from that real state. still/vace_extend already extracted the frame; a
        # cut carries none, so extract one here — a cut is precisely where the
        # authored prompt drifts furthest from what got rendered.
        prompt_authored = goal.prompt
        refresh_note: "str | None" = None
        if (getattr(spec, "continuity_refresh", False) and seg_i > 0
                and prev_clip_path is not None and prev_frames > 0):
            prev_png: "str | None" = None
            if seg_joint_mode == "still":
                prev_png = start_image                     # branch.png, extracted above
            elif seg_joint_mode == "vace_extend" and vace_context_frames:
                prev_png = vace_context_frames[-1]         # newest context frame
            else:                                          # cut — no frame carried
                cut_png = os.path.join(seg_out_root, "refresh_prev.png")
                ok, _tail = _extract_frame_at(prev_clip_path, prev_frames - 1, cut_png)
                prev_png = cut_png if ok else None
            if prev_png is None:
                refresh_note = "no previous frame available — kept authored prompt"
            else:
                persisted = _persisted_refresh(movie_root, goal.segment_id)
                if persisted is not None:
                    refreshed: "str | None" = persisted
                    refresh_note = "reused persisted rewrite (resume determinism)"
                else:
                    _emit("refreshing", {"segment_id": goal.segment_id, "index": seg_i})
                    refreshed, refresh_note = _continuity_refresh(
                        prev_png, prev_prompt, goal.prompt,
                        model=getattr(spec, "continuity_model", None))
                if refreshed is not None:
                    goal = replace(goal, prompt=refreshed)
                    segments_meta[seg_i].update(prompt=refreshed,
                                                prompt_authored=prompt_authored)
            if refresh_note is not None:
                segments_meta[seg_i].update(refresh_note=refresh_note)
                logger.info("studio movie %s: segment %d (%s) continuity refresh — %s",
                            job_id, seg_i, goal.segment_id, refresh_note)

        # ---- deterministic per-segment seed (node override wins) ----
        seg_seed = goal.seed if goal.seed is not None else (spec.seed + seg_i)

        # ---- which MODEL this segment asks for (k58) ----
        # A movie-level pin binds ONLY the segments whose capability it serves; the rest
        # resolve their own capable model through the router, and the substitution is
        # ATTRIBUTED (seg_model.as_record() rides into movie.json + the JobResult, and the
        # note into the live progress blob -> the stage log). An EXPLICIT per-goal model_id
        # is authoritative and never substituted — the route's preflight already refused
        # the movie if it cannot serve this segment, so reaching a render with one means it
        # can. This is what stops the mid-movie pinned_model_unavailable that burned real
        # GPU minutes on segment 0 before segment 1's i2v splice was ever considered.
        seg_model = resolve_segment_model(capability, goal.model_id, spec.model_id)
        if seg_model.note is not None:
            logger.info("studio movie %s: segment %d (%s) capability=%s — %s",
                        job_id, seg_i, goal.segment_id, capability, seg_model.note)

        # ---- build the per-segment studio spec + render through the SAME spine ----
        # (validate-at-construction; a bad geometry/override raises LOCALLY here, which
        # is a programmer error since the movie spec was already validated — geometry
        # is movie-level and in range.)
        seg_spec = make_studio_i2v(
            capability=capability,
            width=spec.width, height=spec.height, fps=spec.fps,
            vram_budget_gb=seg_budget,   # bumped to the VACE floor for a vace_extend joint
            seed=seg_seed,
            out_root=seg_out_root,
            start_image=start_image,
            negative=(goal.negative if goal.negative is not None else spec.negative),
            prompt=goal.prompt,
            project=spec.project,
            steps=(goal.steps if goal.steps is not None else spec.steps),
            cfg=(goal.cfg if goal.cfg is not None else spec.cfg),
            # CLIP LENGTH (2026-08-13, threaded together with the schema field —
            # see studio_movie_schema's header rule): per-goal frames, else the
            # movie-level default, else None -> the bound model's own default.
            # getattr keeps an old pickled/dict spec without the field working.
            requested_frames=(getattr(goal, "frames", None)
                              if getattr(goal, "frames", None) is not None
                              else getattr(spec, "frames", None)),
            # PER-CAPABILITY pin (k58): the explicit per-goal choice, else the movie-level
            # pin ONLY where it serves this segment's capability, else None (unpinned —
            # the router resolves a capable model for THIS capability).
            model_id=seg_model.model_id,
            # VACE-EXTEND temporal conditioning (None for a still/i2v/t2v segment).
            vace_context_frames=vace_context_frames,
            # IDENTITY LOCK: the movie-level subject references, passed on EVERY segment of an
            # id-movie (capability id_lock) so the locked subject carries across scene changes.
            # None for a plain movie.
            reference_images=(id_refs if is_id_movie else None),
        )

        segments_meta[seg_i].update(status="generating")
        # The model attribution rides in the LIVE blob too (and therefore into the bus
        # stage log's detail line), so a capability fallback is visible WHILE it renders,
        # not only in movie.json after the fact.
        _emit("generating", {"segment_id": goal.segment_id, "index": seg_i,
                             "prompt": goal.prompt, "capability": capability,
                             **seg_model.as_record()})

        # RENDER via the SHARED render_clip: a REAL-model segment DELEGATES to the studio
        # GPU worker (HUGPY_STUDIO_WORKER) — progress nested into THIS movie's per-segment
        # blob, cancel relayed from the MOVIE's is_cancelling, ingested from the SHARED
        # content-addressed path — while a synthetic segment renders IN-PROCESS. The inline
        # execution point is passed as ``produce=run_produce_clip`` (this module's global,
        # which the movie tests patch to a fake render seam). ``render_id`` is a DISTINCT
        # per-segment worker key (the worker keys renders by it; a movie posts many).
        seg_render_id = f"{job_id}.s{seg_i:02d}.{run_nonce}"

        def _seg_progress_sink(worker_blob, _sid=goal.segment_id, _idx=seg_i,
                               _prompt=goal.prompt, _cap=capability,
                               _model=seg_model.as_record()):
            # Nest the DELEGATED segment's worker progress (queue position / render
            # progress) under the movie's ``current.worker`` so the console shows the
            # in-flight segment without flattening the movie-level nested blob.
            _emit("generating", {"segment_id": _sid, "index": _idx, "prompt": _prompt,
                                 "capability": _cap, "delegated": True,
                                 "worker": worker_blob, **_model})

        outcome = render_clip(
            seg_spec, render_id=seg_render_id, should_cancel=should_cancel,
            progress_sink=_seg_progress_sink, produce=run_produce_clip)
        if not outcome.ok:
            # An expected per-segment failure (unroutable, mid-render CANCELLED, IO, a
            # vace_extend/id_lock segment's graceful NO_GPU/DEPS_MISSING/WEIGHTS_MISSING on
            # a GPU-less box, or a delegation error — worker_lost/delegation_timeout/
            # worker_busy — from the worker seam) fails the whole movie: DATA, never a
            # raise, NEVER a silent fallback to still-mode. ``outcome.error`` is already a
            # bus JobError (translated in-process OR on the worker), ENRICHED here with
            # WHICH segment + joint mode failed; the failed node is recorded (status="failed").
            segments_meta[seg_i].update(status="failed")
            je = outcome.error
            seg_records.append({
                "index": seg_i,
                "segment_id": goal.segment_id,
                "parent_segment_id": goal.parent_segment_id,
                "prompt": goal.prompt,
                "prompt_authored": prompt_authored,
                "refresh_note": refresh_note,
                "capability": capability,
                # k58 attribution: which model this segment ASKED for and why (the
                # movie pin, an explicit per-segment choice, or a capability fallback).
                **seg_model.as_record(),
                "joint_mode": seg_joint_mode,
                "context_frames": seg_context_frames,
                "resolved_branch": resolved_branch,
                # RESOLVED effective budget + how it was chosen (honesty in the artifact):
                # the autofit-resolved number when the movie budget was blank, else the
                # explicit/floored seg_budget. render_clip stamps these on failure too.
                "vram_budget_gb": (outcome.effective_budget_gb
                                   if outcome.effective_budget_gb is not None else seg_budget),
                "budget_source": outcome.budget_source,
                "status": "failed",
                "error": {"code": je.code, "message": je.message},
            })
            return _partial_return(JobResult(job_id, ok=False, error=JobError(
                code=je.code,
                message=(f"segment {seg_i} ({goal.segment_id!r}, joint_mode={seg_joint_mode}, "
                         f"capability={capability}): {je.message}"),
                retryable=je.retryable)))

        # Ok: the clip exists on the SHARED store (rendered in-process OR by the worker).
        # Catalog the WHOLE clip (never trimmed) as a video MediaRef on outputs — the
        # ingest is identical for a local or a worker-written clip (same shared path).
        ref = ingest(outcome.path, kind_hint="video")
        seg_refs.append(ref)

        seg_records.append({
            "index": seg_i,
            "segment_id": goal.segment_id,
            "parent_segment_id": goal.parent_segment_id,
            "prompt": goal.prompt,
            "prompt_authored": prompt_authored,
            "refresh_note": refresh_note,
            "capability": capability,
            # k58 attribution: how the model was chosen (``model_source``) and WHICH one
            # actually rendered — the manifest sidecar's bound model when it is readable,
            # so a capability fallback names its substitute instead of a bare null.
            **{**seg_model.as_record(),
               "model_id": (_bound_model_id(outcome.path) or seg_model.model_id)},
            "seed": seg_seed,
            "branch_frame": goal.branch_frame,          # the AUTHORED value (may be null)
            "resolved_branch": resolved_branch,          # frame index into the PARENT (None for root)
            # JOINT MODE honesty: how this segment was spliced onto its parent + the
            # motion-carry conditioning. context_frames = parent frames KEPT as the VACE
            # extend prefix (0 for still); context_drop = frames the assembler drops from
            # THIS segment's head so the reconstructed context never double-plays (0 for
            # still). vram_budget_gb = the EFFECTIVE per-segment budget (bumped for vace).
            "joint_mode": seg_joint_mode,
            "context_frames": seg_context_frames,
            "context_drop": seg_context_drop,
            # RESOLVED effective per-segment budget + its source ("explicit" |
            # "autofit:<worker>" | "unresolved"): the number the router actually used
            # (autofit-sized to the worker's GPU CAPACITY when the movie budget was blank —
            # the reservation engine evicts to free it — else the explicit/floored
            # seg_budget). Honesty in movie.json.
            "vram_budget_gb": (outcome.effective_budget_gb
                               if outcome.effective_budget_gb is not None else seg_budget),
            "budget_source": outcome.budget_source,
            "clip_path": outcome.path,
            "clip_uri": ref.uri,
            "frames": outcome.frames,
            "width": outcome.width,
            "height": outcome.height,
            "duration_s": outcome.duration_s,
            "content_hash": outcome.content_hash,
            "resumed": outcome.resumed,
            "status": "resumed" if outcome.resumed else "done",
        })
        segments_meta[seg_i].update(status=("resumed" if outcome.resumed else "done"),
                                    resumed=outcome.resumed)

        prev_clip_path = outcome.path
        prev_frames = outcome.frames
        prev_prompt = goal.prompt            # EFFECTIVE (post-refresh) prompt (k120)
        logger.info("studio movie %s: segment %d (%s) %s — %s frames @ %s",
                    job_id, seg_i, goal.segment_id,
                    "RESUMED" if outcome.resumed else "rendered", outcome.frames, outcome.path)

        _emit("generating", None)
        # ITERATIVE save after every segment (watchable partial + fresh manifest).
        _save(partial=True)

    # ---- FINAL assembly + manifest ----
    _emit("assembling", None)
    movie_manifest = _save(partial=False)

    # Ingest the stitched movie.mp4 LAST so outputs[-1] is the assembled video.
    all_refs = list(seg_refs)
    if movie_manifest.get("assembly", {}).get("movie"):
        movie_mp4 = os.path.join(movie_root, "movie.mp4")
        if os.path.isfile(movie_mp4):
            all_refs.append(ingest(movie_mp4, kind_hint="video"))

    _emit("archiving", None)
    return JobResult(
        job_id, ok=True, outputs=tuple(all_refs),
        project={"name": spec.project, "uuid": job_id, "dir": movie_root},
        movie=movie_manifest)
