"""LAST-RESORT ffmpeg enhancers (slice b, §6) — the studio's REAL frame-
interpolation and spatial-upscale executors on a GPU-less box TODAY, with ZERO new
deps (they shell out to the system ``ffmpeg`` binary, house style).

    run_ffmpeg_interpolate(manifest, out_root, start_image=None, should_cancel=None)
    run_ffmpeg_upscale    (manifest, out_root, start_image=None, should_cancel=None)
        -> Result[Artifact, StageError]

Both are ENHANCEMENT runners: they transform an EXISTING clip (``manifest.source_
video``, part of the content_hash — B-2), so they follow the v2v discipline of
``wan_vace``: SOURCE-FIRST preflight (a source-less enhance is a SPEC error that is
malformed on ANY box), then the box capability check (here just: is ``ffmpeg`` /
``ffprobe`` on PATH). Same content-addressed atomic layout as every other runner
(``<out_root>/<content_hash>/clip.mp4`` + ``manifest.json`` + ``provenance.json``),
same resume-on-hash (INV-6), same errors-as-data discipline (INV-3), same
cooperative-cancel contract (T1). Only a genuine programmer error (a
non-RenderManifest) raises.

THE TWO CAPABILITIES (slice b):
  * INTERP  (Task.INTERPOLATE) — ffmpeg ``minterpolate`` MOTION-COMPENSATED
    interpolation (``mi_mode=mci``, genuinely synthesized in-between frames, not
    duplicates) resampling the source to the manifest's TARGET FPS. FPS MAPPING:
    the target fps is ``manifest.resolution_ladder[0].fps`` (the top rung's cadence,
    which the router sets from ``CapabilityRequest.target_resolution.fps``); the
    SPATIAL resolution is left at the source's native size (interpolation adds
    frames in time, it does not rescale — that is upscale's job).
  * UPRES   (Task.UPSCALE) — ffmpeg ``scale=<W>:<H>:flags=lanczos`` to the manifest
    resolution (``manifest.resolution_ladder[0]`` width/height), Lanczos resampling.
    The temporal cadence (fps + frame count) is left as the source's (upscale is
    spatial only — fps conversion is interpolation's job).

DETERMINISM (DeterminismClass.EXACT): the ffmpeg transform is a pure function of
the source bytes + the filter string. ``-threads 1`` fixes the encoder/filter
thread count so the same source+params yield the same output bits on a given ffmpeg
build (libx264 sliced-threading is the only nondeterminism knob these paths touch).
The manifest's ``prompt`` rides in the content_hash (and the manifest.json sidecar)
for provenance but is NEVER passed to ffmpeg, so it can address a distinct output
without ever altering a pixel — mirroring the synthetic t2v prompt discipline.

IMPORT SAFETY: stdlib only at module top (plus the reused synthetic sidecar
helpers, which pull numpy/PIL — already house deps, NOT the GPU stack). No torch /
diffusers / bitsandbytes anywhere, lazily or otherwise.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Callable

from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.errors import Err, ErrorCode, Ok, Result, StageError
from hugpy_video.intel.studio.manifest import render_manifest_to_dict
from hugpy_video.intel.studio.schemas import RenderManifest
from hugpy_video.intel.studio.storage import atomic_write_text
# Reuse the synthetic runner's atomic/content-addressed plumbing so an enhanced clip
# lands in the IDENTICAL on-disk layout + shares the single ffmpeg-subprocess
# semaphore (house convention). These names are module-level in ``synthetic`` and
# pull only numpy/PIL (house deps) — NOT the heavy GPU stack.
from hugpy_video.intel.studio.runners.synthetic import (
    _ASSEMBLY_SEM,
    _CLIP_NAME,
    _MANIFEST_NAME,
    _PROVENANCE_NAME,
    _provenance_dict,
)


# --------------------------------------------------------------------------- #
# Source-clip resolution (pure) — the enhancement INPUT
# --------------------------------------------------------------------------- #
def _resolve_source(manifest: RenderManifest) -> str | None:
    """The absolute path of the clip this enhance transforms, or None if the
    manifest carries none. ``source_video`` is part of the content_hash (B-2), so an
    enhance is deterministically keyed on the clip it interpolates/upscales."""
    src = getattr(manifest, "source_video", "") or ""
    return src or None


# --------------------------------------------------------------------------- #
# Preflight — errors as data (returns a StageError to wrap in Err, or None)
# --------------------------------------------------------------------------- #
def _preflight(manifest: RenderManifest, what: str) -> StageError | None:
    """Gate the real path. ORDER: source (spec) -> tools (box capability). The
    SOURCE check is FIRST because an enhance with no source clip is a SPEC error
    (malformed on any box) and must surface as SOURCE_MISSING rather than be masked
    by a missing-ffmpeg box's DEPS_MISSING — mirrors ``wan_vace._preflight``."""
    source = _resolve_source(manifest)
    if source is None:
        return StageError(
            ErrorCode.SOURCE_MISSING,
            f"ffmpeg {what} render carries no source_video — an enhancement is "
            f"defined by the clip it transforms; supply source_video",
            (("model_id", manifest.model_id), ("capability", manifest.capability.value)),
        )
    if not os.path.isfile(source):
        return StageError(
            ErrorCode.SOURCE_MISSING,
            f"ffmpeg {what} source_video not found on disk: {source}",
            (("source_video", source), ("model_id", manifest.model_id)),
        )

    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        return StageError(
            ErrorCode.DEPS_MISSING,
            f"ffmpeg {what} needs the system tools that are not on PATH: "
            + ", ".join(missing)
            + ". Install ffmpeg (provides ffmpeg + ffprobe).",
            (("missing", ",".join(missing)),),
        )
    return None


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe plumbing (house style: shutil.which, PIPE, returncode check,
# never raises on a plain tool failure — errors-as-data)
# --------------------------------------------------------------------------- #
def _run_ffmpeg(source: str, vf: str, tmp_mp4: str) -> tuple[bool, str]:
    """Single-shot video->video transform ``source`` --(-vf ``vf``)--> ``tmp_mp4``
    (H.264, house encode settings mirrored from ``synthetic._assemble_mp4``).
    ``-threads 1`` fixes thread count for deterministic output bits. Serialized on
    the shared ffmpeg semaphore. Returns (ok, stderr_tail); never raises."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y",
        "-i", source,
        "-vf", vf,
        "-an",                       # enhancement is a video op; drop any audio
        "-threads", "1",             # deterministic (no sliced-thread nondeterminism)
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_mp4,
    ]
    with _ASSEMBLY_SEM:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = (result.returncode == 0
          and os.path.isfile(tmp_mp4) and os.path.getsize(tmp_mp4) > 0)
    return ok, (result.stderr or "")[-500:]


def _parse_fps(rate: str) -> float:
    """ffprobe ``r_frame_rate`` arrives as a rational string (e.g. "16/1"). Parse it
    to a float; a malformed value degrades to 0.0 (errors-as-data, never raises)."""
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _probe(video: str) -> "tuple[int, int, float, int] | None":
    """(width, height, fps, n_frames) of ``video`` via ffprobe, or None on any
    failure (errors-as-data). ``nb_frames`` from the container is used when present;
    when the muxer reports "N/A" we DECODE-count (``-count_frames``) so the reported
    frame count is exact (the enhance already re-encoded the whole clip, so the extra
    pass is affordable). Reads the FIRST video stream only."""
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    base = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", video,
    ]
    try:
        res = subprocess.run(base, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return None
        streams = json.loads(res.stdout or "{}").get("streams") or []
        if not streams:
            return None
        s = streams[0]
        width = int(s.get("width") or 0)
        height = int(s.get("height") or 0)
        fps = _parse_fps(str(s.get("r_frame_rate") or "0"))
        raw_n = s.get("nb_frames")
        if raw_n in (None, "N/A", ""):
            cnt = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames", "-of", "json", video],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            cstreams = json.loads(cnt.stdout or "{}").get("streams") or []
            n_frames = int((cstreams[0].get("nb_read_frames") if cstreams else 0) or 0)
        else:
            n_frames = int(raw_n)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height, fps, n_frames


# --------------------------------------------------------------------------- #
# Shared enhance core (both runners differ ONLY in the filter they build)
# --------------------------------------------------------------------------- #
def _enhance(
    manifest: RenderManifest,
    out_root: str,
    *,
    what: str,
    build_vf: "Callable[[RenderManifest, str], tuple[str, float]]",
    should_cancel: "Callable[[], bool] | None",
) -> Result[Artifact, StageError]:
    """The shared spine for both ffmpeg enhancers. ``build_vf(manifest, source)`` ->
    (ffmpeg -vf string, fallback_fps) is the only per-operation piece. The reported
    Artifact geometry is PROBED from the produced clip (the honest, exact answer),
    falling back to a computed value only if ffprobe can't read the output.

    Resume-on-hash FIRST (INV-6), then SOURCE-first preflight, then the single-shot
    ffmpeg run (checked for cancel BEFORE it starts — an ffmpeg run is not
    cooperatively interruptible; checking before start is the accepted T1 semantics
    for a single external process), then atomic content-addressed promotion +
    sidecars. Errors ride back as data; only a non-RenderManifest raises."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(
            f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    content_hash = manifest.content_hash()
    out_dir = os.path.join(os.path.abspath(out_root), content_hash)
    clip_path = os.path.join(out_dir, _CLIP_NAME)

    # INV-6 resume FIRST: an existing non-empty clip is served as-is, no ffmpeg, no
    # source touched. Geometry re-derived from the served clip (best-effort probe;
    # falls back to the manifest's top rung if the probe fails).
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        w, h, fps, n = _resumed_geometry(manifest, clip_path)
        return Ok(Artifact(
            path=clip_path, content_hash=content_hash, frames=n,
            width=w, height=h, duration_s=(n / fps if fps > 0 else 0.0),
            resumed=True))

    pf = _preflight(manifest, what)
    if pf is not None:
        return Err(pf)

    source = _resolve_source(manifest)          # preflight proved it exists
    vf, fallback_fps = build_vf(manifest, source)

    tmp_mp4 = None
    try:
        os.makedirs(out_dir, exist_ok=True)

        # Cooperative cancel BEFORE the single-shot ffmpeg run (nothing written yet).
        if should_cancel is not None and should_cancel():
            return Err(StageError(
                ErrorCode.CANCELLED, f"cancelled before ffmpeg {what}",
                (("content_hash", content_hash), ("model_id", manifest.model_id))))

        tmp_mp4 = os.path.join(out_dir, f".clip-tmp-{os.getpid()}.mp4")
        ok, stderr_tail = _run_ffmpeg(source, vf, tmp_mp4)
        if not ok:
            return Err(StageError(
                ErrorCode.ASSEMBLY_FAILED,
                f"ffmpeg {what} failed: {stderr_tail}",
                (("content_hash", content_hash), ("source_video", source))))

        os.replace(tmp_mp4, clip_path)          # atomic promotion of the clip
        tmp_mp4 = None

        atomic_write_text(
            os.path.join(out_dir, _MANIFEST_NAME),
            json.dumps(render_manifest_to_dict(manifest), indent=2, sort_keys=True))
        atomic_write_text(
            os.path.join(out_dir, _PROVENANCE_NAME),
            json.dumps(_provenance_dict(manifest), indent=2, sort_keys=True))
    except OSError as exc:                       # unwritable out_root, disk full, ...
        return Err(StageError(
            ErrorCode.IO_ERROR, f"ffmpeg {what} IO failure: {exc}",
            (("out_dir", out_dir),)))
    finally:
        if tmp_mp4 is not None and os.path.isfile(tmp_mp4):
            try:
                os.remove(tmp_mp4)
            except OSError:
                pass

    # Report EXACT geometry from the produced clip; fall back to the manifest's top
    # rung geometry + the operation's fallback fps if ffprobe can't read the output.
    probed = _probe(clip_path)
    if probed is not None:
        w, h, fps, n = probed
        if fps <= 0:
            fps = fallback_fps
    else:
        top = manifest.resolution_ladder[0]
        w, h, fps, n = top.width, top.height, fallback_fps, 0
    return Ok(Artifact(
        path=clip_path, content_hash=content_hash, frames=n,
        width=w, height=h, duration_s=(n / fps if fps > 0 else 0.0),
        resumed=False))


def _resumed_geometry(manifest: RenderManifest, clip_path: str) -> tuple[int, int, float, int]:
    """Best-effort geometry for a resumed clip: probe it, else the manifest top rung."""
    probed = _probe(clip_path)
    if probed is not None and probed[2] > 0:
        return probed
    top = manifest.resolution_ladder[0]
    return top.width, top.height, float(top.fps), (probed[3] if probed else 0)


# --------------------------------------------------------------------------- #
# The two runners
# --------------------------------------------------------------------------- #
def _interp_vf(manifest: RenderManifest, source: str) -> tuple[str, float]:
    """minterpolate to the manifest's TARGET fps (top rung's cadence), motion-
    compensated (``mi_mode=mci`` — real synthesized in-betweens, not dup'd frames).
    Spatial resolution untouched (interpolation is temporal)."""
    target_fps = int(manifest.resolution_ladder[0].fps)
    vf = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=obmc:me_mode=bilat"
    return vf, float(target_fps)


def _upscale_vf(manifest: RenderManifest, source: str) -> tuple[str, float]:
    """Lanczos spatial upscale to the manifest resolution (top rung W x H). Temporal
    cadence untouched (fps conversion is interpolation's job); the fallback fps is
    the top rung's nominal fps if the produced clip can't be probed."""
    top = manifest.resolution_ladder[0]
    vf = f"scale={int(top.width)}:{int(top.height)}:flags=lanczos"
    return vf, float(top.fps)


def run_ffmpeg_interpolate(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a MOTION-INTERPOLATED clip for ``manifest`` under
    ``out_root`` via ffmpeg ``minterpolate`` (the LAST-RESORT INTERP capability).

    Returns ``Ok(Artifact)`` with the interpolated clip, or ``Err(StageError)`` on
    any expected failure (SOURCE_MISSING when no source clip, DEPS_MISSING when
    ffmpeg/ffprobe are off PATH, ASSEMBLY_FAILED on an ffmpeg error, CANCELLED via
    ``should_cancel``). Only a non-RenderManifest raises. ``start_image`` is part of
    the uniform runner signature but UNUSED (interpolation conditions on the whole
    source clip, not a still). See the module docstring for the fps mapping."""
    return _enhance(
        manifest, out_root, what="interpolate",
        build_vf=_interp_vf, should_cancel=should_cancel)


def run_ffmpeg_upscale(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a SPATIALLY-UPSCALED clip for ``manifest`` under
    ``out_root`` via ffmpeg ``scale=<W>:<H>:flags=lanczos`` (the LAST-RESORT UPRES
    capability).

    Returns ``Ok(Artifact)`` with the upscaled clip, or ``Err(StageError)`` on any
    expected failure (SOURCE_MISSING / DEPS_MISSING / ASSEMBLY_FAILED / CANCELLED).
    Only a non-RenderManifest raises. ``start_image`` is part of the uniform runner
    signature but UNUSED (upscale conditions on the whole source clip)."""
    return _enhance(
        manifest, out_root, what="upscale",
        build_vf=_upscale_vf, should_cancel=should_cancel)
