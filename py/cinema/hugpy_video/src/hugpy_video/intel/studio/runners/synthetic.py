"""SYNTHETIC i2v runner (P0-B1) — the no-model executor that proves the spine.

Given a ``RenderManifest`` it produces a real, playable H.264 clip with NO GPU and
NO weights: deterministic frames synthesized from ``manifest.seeds.global_seed`` +
the top ``Resolution`` in the ladder, assembled to mp4 via ffmpeg (the house
invocation mirrored from ``video_intel/runners/scene.py``).

Invariants honored:
  * INV-3  errors as data      — expected failures return ``Err(StageError)``;
                                 only genuine programmer error would raise.
  * INV-6  content-addressed    — output at ``<out_root>/<content_hash>/clip.mp4``
           + atomic + resumable   with ``manifest.json`` / ``provenance.json``
                                   sidecars; writes go temp -> ``os.replace``; an
                                   existing non-empty clip SKIPS regeneration.
  * Determinism  — same manifest ⇒ identical frame bytes (``synthesize_frame`` is
                   a pure function of seed + geometry + frame index).

Clip length is a function of the MANIFEST only — ``manifest.requested_frames`` when
the caller asked for a length, else the bound model's default, clamped to that
model's real ceiling (``resolve_frames`` below) — so identical manifests still yield
identical-length clips, and the requested length is part of the content hash (INV-6)
rather than divorced from it. A request's ``min_frames`` is still honored upstream by
the ROUTER (it rejects models whose ``max_frames`` is below the floor), never by
silently lengthening the clip here.

That length REQUEST now has a real path to get here (2026-07-27):
``StudioI2VSpec.requested_frames`` -> ``produce_clip(requested_frames=)`` ->
``make_render_manifest(requested_frames=)`` -> ``RenderManifest.requested_frames``
-> ``resolve_frames``. Before that the field existed and was hashed but no caller
could set it, so every real render on this fleet was forced to the default — the
lever the docstrings advertised did not exist.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Callable

import numpy as np
from PIL import Image

from hugpy_video.intel.studio.artifacts import Artifact
from hugpy_video.intel.studio.enums import Framework, Task
from hugpy_video.intel.studio.errors import Err, ErrorCode, Ok, Result, StageError
from hugpy_video.intel.studio.manifest import render_manifest_to_dict
from hugpy_video.intel.studio.registry import MODEL_REGISTRY
# CLIP-LENGTH POLICY comes from ``schemas`` (the leaf module), NOT from a literal
# here — the wire (``studio/presets.py`` -> GET /video/render/presets) must publish
# the SAME numbers this runner renders at, and it cannot import this module without
# dragging numpy/PIL into app boot. See the long WHY on schemas' clip-length section.
# Re-exported by this import (``runners.synthetic.DEFAULT_FRAMES_REAL`` resolves) so
# either import site works and there is still exactly one literal.
from hugpy_video.intel.studio.schemas import (
    DEFAULT_FRAMES_REAL,
    ProvenanceStub,
    RenderManifest,
    WAN_FRAME_CADENCE,
    WAN_MAX_FRAMES,
    snap_wan_frames,
)
from hugpy_video.intel.studio.storage import atomic_write_text

logger = logging.getLogger(__name__)

# Serialize ONLY the mp4 assembly subprocess (house convention, mirrors
# scene.py's _SCENE_SEM). Frame synthesis is pure-python/numpy and not gated.
_ASSEMBLY_SEM = threading.BoundedSemaphore(1)

_CLIP_NAME = "clip.mp4"
_MANIFEST_NAME = "manifest.json"
_PROVENANCE_NAME = "provenance.json"
# Sidecar for the B2 "extend the movie" conditioning still: the source clip's last
# frame, extracted next to the content-addressed clip it conditions (content_hash
# already keys on manifest.source_video, so this path is deterministic + resume-safe).
_SOURCE_LASTFRAME_NAME = "source_lastframe.png"


# --------------------------------------------------------------------------- #
# Pure, deterministic frame synthesis (testable in isolation)
# --------------------------------------------------------------------------- #
def _frame_params(seed: int) -> dict:
    """Seed-derived, resolution-independent pattern parameters. A fixed RNG stream
    off ``global_seed`` so the same seed always yields the same look."""
    rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
    return {
        "fx": float(rng.uniform(2.0, 6.0)),
        "fy": float(rng.uniform(2.0, 6.0)),
        "fd": float(rng.uniform(2.0, 6.0)),
        "px": float(rng.uniform(0.0, 2.0 * math.pi)),
        "py": float(rng.uniform(0.0, 2.0 * math.pi)),
        "pd": float(rng.uniform(0.0, 2.0 * math.pi)),
        # per-channel colour phase offsets
        "cr": float(rng.uniform(0.0, 2.0 * math.pi)),
        "cg": float(rng.uniform(0.0, 2.0 * math.pi)),
        "cb": float(rng.uniform(0.0, 2.0 * math.pi)),
        # start-image tint gains + pan direction
        "gr": float(rng.uniform(0.85, 1.15)),
        "gg": float(rng.uniform(0.85, 1.15)),
        "gb": float(rng.uniform(0.85, 1.15)),
        "panx": float(rng.uniform(-1.0, 1.0)),
        "pany": float(rng.uniform(-1.0, 1.0)),
    }


def synthesize_frame(
    seed: int,
    width: int,
    height: int,
    frame_idx: int,
    n_frames: int,
    start_arr: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Deterministic HxWx3 uint8 frame. Pure function of its arguments — same
    inputs ⇒ byte-identical output. With ``start_arr`` it does a seeded tint +
    slow zoom-pan of the still; otherwise a seed-driven procedural plasma."""
    p = _frame_params(seed)
    denom = max(1, n_frames - 1)
    t = frame_idx / denom                     # 0..1 progress
    phase = 2.0 * math.pi * frame_idx / max(1, n_frames)  # loops smoothly

    if start_arr is not None:
        # --- seeded tint + slow zoom-pan of a still image ---
        img = Image.fromarray(start_arr)
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        zoom = 1.0 + 0.12 * t                 # zoom in up to 12%
        cw = max(2, int(round(width / zoom)))
        ch = max(2, int(round(height / zoom)))
        max_dx = width - cw
        max_dy = height - ch
        # deterministic pan sweeping from center outward along the seeded vector
        cx = (width - cw) / 2.0 + p["panx"] * (max_dx / 2.0) * t
        cy = (height - ch) / 2.0 + p["pany"] * (max_dy / 2.0) * t
        left = int(min(max(0, round(cx)), max_dx))
        top = int(min(max(0, round(cy)), max_dy))
        crop = img.crop((left, top, left + cw, top + ch)).resize(
            (width, height), Image.LANCZOS)
        arr = np.asarray(crop, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr[:, :, :3]
        # slow oscillating tint so motion is visible even on a flat still
        osc = 0.15 * math.sin(phase)
        gains = np.array([p["gr"] + osc, p["gg"] - osc, p["gb"] + osc], np.float32)
        out = np.clip(arr * gains, 0.0, 255.0).astype(np.uint8)
        return out

    # --- procedural plasma (no start image) ---
    yy, xx = np.mgrid[0:height, 0:width]
    xn = xx.astype(np.float32) / float(width)
    yn = yy.astype(np.float32) / float(height)
    v1 = np.sin(p["fx"] * xn * 2.0 * math.pi + phase + p["px"])
    v2 = np.sin(p["fy"] * yn * 2.0 * math.pi + phase * 1.3 + p["py"])
    v3 = np.sin(p["fd"] * (xn + yn) * math.pi * 2.0 + phase * 0.7 + p["pd"])
    base = (v1 + v2 + v3) / 3.0               # -1..1
    r = 0.5 + 0.5 * np.sin(base * math.pi + p["cr"])
    g = 0.5 + 0.5 * np.sin(base * math.pi + p["cg"])
    b = 0.5 + 0.5 * np.sin(base * math.pi + p["cb"])
    frame = np.stack([r, g, b], axis=-1)
    return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Manifest-derived geometry (clip length is a pure function of the manifest)
# --------------------------------------------------------------------------- #
# ``WAN_FRAME_CADENCE`` (4), ``WAN_MAX_FRAMES`` (81), ``DEFAULT_FRAMES_REAL`` (81)
# and ``snap_wan_frames`` are IMPORTED from ``..schemas`` (see the import block).
# They used to be private literals here, which is how the wire and the renderer
# drifted 2.8x apart: ``GET /video/render/presets`` published 29 frames while this
# module rendered 81. One literal, two importers, no drift — do NOT restate them.
#
# THE DEFAULT IS 81 FRAMES for a real model: Wan's own reference length, the
# ``max_frames`` every Wan row declares, and MEASURED at ~352s wall-clock for
# wan2.1-t2v-1.3b @ 832x480 x 81 frames on ae's 3090 (2026-07-27). 81 @ 16fps =
# 5.0625s. A caller who wants a cheap preview asks for fewer via
# ``manifest.requested_frames`` — which, since 2026-07-27, they can actually do.
#
# The 4k+1 snap happens HERE (``resolve_frames``) and ``wan_i2v._wan_geometry``
# snaps AGAIN on top of it. That is idempotent by construction (snapping an
# already-4k+1 value is a no-op), which is the property that keeps the resume check
# and the generate call agreeing on one exact frame count.

# DEFAULT clip length for the SYNTHETIC prover: the historical ``fps * 2`` (~2s).
#
# ⚠ DELIBERATELY DIFFERENT FROM ``DEFAULT_FRAMES_REAL``, and that divergence is the
# one place two clip-length numbers are allowed to coexist. The reason they differ is
# that they are not the same KIND of number:
#   * 81 is a PRODUCT PROMISE. It costs ~352s of a 3090 and it is what a caller who
#     asked for a clip actually wants, so the wire publishes it and the renderer
#     honours it.
#   * fps*2 is a TEST COST. The synthetic runner is a SPINE PROVER — no GPU, no
#     weights, one pure-numpy frame + one PNG write per frame — so its output is
#     noise that proves the plumbing, never a product surface. 81 frames of plasma is
#     ~3.4x the PNG encoding of a 24-frame prover clip and proves nothing the 2s clip
#     does not, and keeping the historical multiplier leaves every synthetic clip
#     already on disk the length it has always been.
# Because the prover is not a product surface, this number is NOT published on the
# wire (no preset binds a synthetic model unless STUDIO_ALLOW_SYNTHETIC=1), so there
# is nothing here for the wire to disagree with. It stays module-private for exactly
# that reason: nothing outside this runner has any business reading it.
_DEFAULT_SYNTHETIC_FPS_MULT = 2


def resolve_frames(manifest: RenderManifest) -> tuple[int, str]:
    """The clip's frame count PLUS the one-line REASON it ended up there.

    THE one place clip length is decided (``_geometry`` below and, through it,
    ``wan_i2v._wan_geometry`` / the VACE runner all land here). Resolution order:

      1. ``manifest.requested_frames`` when set, else the model-aware default
         (``DEFAULT_FRAMES_REAL`` for a real model, ``fps * 2`` for the prover);
      2. the ``max(1, n)`` floor (unchanged behaviour);
      3. CLAMP to the model's real ceiling — ``cfg.max_frames`` from MODEL_REGISTRY,
         and additionally ``WAN_MAX_FRAMES`` for a Wan binding;
      4. SNAP down to the 4k+1 temporal cadence for a real pipeline.

    An out-of-range request CLAMPS and records why — it is NEVER an error. That is
    INV-3 in spirit: a caller asking a 3-second model for 500 frames wants a clip,
    not a 500, so they get the 81 the model can actually deliver and a reason string
    saying so. The reason is logged at INFO whenever the resolved count differs from
    what was asked for, so a short clip is always traceable to a decision.

    NOTE the ceiling is frames-only: ``cfg.max_duration_s`` is DELIBERATELY not a
    clamp input. The Wan rows declare 5.0s while their measured-good 81 frames run
    5.0625s @ 16fps — clamping on the rounded planning figure would give 5.0*16 = 80
    frames, snapped to 77, cutting the MEASURED default for no physical reason. This
    is the seconds-vs-frames trap in miniature: 4k+1 is odd, so no on-cadence count
    lands on a whole second at an even fps, and a seconds-shaped ceiling can only ever
    round the real one DOWN.

    RETURNS the count the runner will actually render, which is also what the caller
    is told: ``Artifact.duration_s`` is computed from THIS number (resolved / fps),
    never from the request — a clamped or snapped ask reports its TRUE duration."""
    top = manifest.resolution_ladder[0]
    cfg = MODEL_REGISTRY.get(manifest.model_id)
    # The prover is identified by its FRAMEWORK, not by ``cfg.synthetic`` (which is
    # also True on the last-resort ffmpeg enhancer rows — different runners entirely,
    # they never reach here) and not by a registry lookup (an unknown model_id must
    # still resolve to the real default, never to the prover's cheap one).
    is_prover = manifest.framework is Framework.SYNTHETIC

    requested = manifest.requested_frames
    if requested is None:
        if is_prover:
            n = top.fps * _DEFAULT_SYNTHETIC_FPS_MULT
            reason = f"default: synthetic prover fps*{_DEFAULT_SYNTHETIC_FPS_MULT}"
        else:
            n = DEFAULT_FRAMES_REAL
            reason = f"default: {DEFAULT_FRAMES_REAL} frames (model capability)"
    else:
        n = int(requested)
        reason = f"requested {n}"

    if n < 1:                                  # unchanged max(1, n) floor behaviour
        reason += f"; floored {n} -> 1"
        n = 1

    ceiling = cfg.max_frames if (cfg is not None and cfg.max_frames) else None
    if manifest.framework is Framework.WAN:
        ceiling = WAN_MAX_FRAMES if ceiling is None else min(ceiling, WAN_MAX_FRAMES)
    if ceiling is not None and n > ceiling:
        reason += f"; clamped {n} -> {ceiling} (model ceiling)"
        n = ceiling

    if not is_prover:
        # 4k+1 cadence: a real latent pipeline REQUIRES it. The prover has no VAE, so
        # snapping its noise would shorten every existing synthetic clip to buy nothing.
        # ``snap_wan_frames`` is the SHARED implementation (schemas) — the presets layer
        # calls the same function to show a caller the TRUE length before spending ~6
        # minutes of denoise, so the preview and the render cannot disagree.
        snapped = snap_wan_frames(n)
        if snapped != n:
            reason += (f"; snapped {n} -> {snapped} "
                       f"({WAN_FRAME_CADENCE}k+1 temporal cadence)")
            n = snapped

    if requested is not None and n != requested:
        # RECORD the clamp/snap rather than failing it (and rather than silently
        # handing back a different length than was asked for).
        logger.info(
            "studio clip length: %s -> %d frames [model=%s]", reason, n, manifest.model_id)
    return n, reason


def _geometry(manifest: RenderManifest) -> tuple[int, int, int, int]:
    """(width, height, fps, n_frames) from the top ladder rung + the RESOLVED clip
    length (``resolve_frames``: the manifest's request, or the bound model's default,
    clamped to that model's ceiling). Signature deliberately unchanged — ``wan_i2v``
    imports this exact function, so real Wan renders pick the lever up for free."""
    top = manifest.resolution_ladder[0]
    n, _reason = resolve_frames(manifest)
    return top.width, top.height, top.fps, n


def _assemble_mp4(frame_dir: str, tmp_mp4: str, fps: int) -> tuple[bool, str]:
    """Mux frame_%05d.png -> H.264 mp4 (house invocation from scene.py). Returns
    (ok, stderr_tail); never raises on a plain ffmpeg failure."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-start_number", "0",
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_mp4,
    ]
    with _ASSEMBLY_SEM:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = result.returncode == 0 and os.path.isfile(tmp_mp4) and os.path.getsize(tmp_mp4) > 0
    return ok, (result.stderr or "")[-500:]


def _extract_last_frame(source_video: str, dest_png: str) -> tuple[bool, str]:
    """Extract the LAST frame of ``source_video`` to ``dest_png`` (a PNG still) —
    the B2 "extend the movie" conditioning frame. Mirrors ``_assemble_mp4``'s house
    ffmpeg invocation (shutil.which, PIPE, returncode check) and never raises on a
    plain ffmpeg failure (errors-as-data, INV-3). The ``-sseof -3 -update 1`` idiom
    seeks ~3s before the end and OVERWRITES the single output for every remaining
    frame, so the final write is the clip's last frame (a clip shorter than 3s just
    decodes from its start and still ends on the final frame). Reused by the Wan i2v
    runner (like the other helpers here) so both extract identically. Returns
    (ok, stderr_tail)."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y",
        "-sseof", "-3",
        "-i", source_video,
        "-update", "1",
        "-q:v", "2",
        dest_png,
    ]
    with _ASSEMBLY_SEM:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = result.returncode == 0 and os.path.isfile(dest_png) and os.path.getsize(dest_png) > 0
    return ok, (result.stderr or "")[-500:]


def _provenance_dict(manifest: RenderManifest) -> dict:
    prov = manifest.provenance
    if prov is None:
        from datetime import datetime, timezone
        prov = ProvenanceStub(
            operator="synthetic-runner",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    return {
        "operator": prov.operator,
        "created_at": prov.created_at,
        "tool": prov.tool,
        "c2pa_pending": prov.c2pa_pending,
    }


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_synthetic_i2v(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a synthetic clip for ``manifest`` under ``out_root``.

    Returns ``Ok(Artifact)`` on success, ``Err(StageError)`` on any expected
    failure (unwritable output tree, bad start image, ffmpeg mux failure). Only a
    genuine programmer error (e.g. a non-RenderManifest) raises.

    ``should_cancel`` is an OPTIONAL cooperative-cancel probe (Task 1): a zero-arg
    callable polled at the TOP of every frame. When it returns True the loop aborts
    BEFORE the atomic ``os.replace`` — so no clip.mp4 lands at the content-addressed
    path — and returns ``Err(StageError(CANCELLED, ...))``. The temp frame dir is
    cleaned by the existing ``finally``. None (default) = never cancel."""
    if not isinstance(manifest, RenderManifest):
        raise TypeError(f"manifest must be a RenderManifest; got {type(manifest).__name__}")

    content_hash = manifest.content_hash()
    width, height, fps, n_frames = _geometry(manifest)
    duration_s = n_frames / float(fps)
    out_dir = os.path.join(os.path.abspath(out_root), content_hash)
    clip_path = os.path.join(out_dir, _CLIP_NAME)

    # INV-6 resume: an existing non-empty clip is returned as-is, no regeneration.
    if os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0:
        return Ok(Artifact(
            path=clip_path, content_hash=content_hash, frames=n_frames,
            width=width, height=height, duration_s=duration_s, resumed=True))

    # B2 chain — "extend the movie": for an i2v render given a source clip but NO
    # start_image, condition on the clip's LAST FRAME (extracted via ffmpeg). The
    # source_video is in the manifest (and thus content_hash), so this is
    # deterministic + resume-safe — the resume check above already served an existing
    # clip WITHOUT touching the source. Placed AFTER resume so a re-run never
    # re-extracts. t2v is text-only (manifest.task != I2V) -> the source is CARRIED
    # in the manifest for provenance but never alters a frame here.
    src_video = getattr(manifest, "source_video", "") or ""
    if start_image is None and src_video and manifest.task == Task.I2V:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            return Err(StageError(
                ErrorCode.IO_ERROR, f"could not create out_dir: {exc}",
                (("out_dir", out_dir),)))
        last_frame = os.path.join(out_dir, _SOURCE_LASTFRAME_NAME)
        ok, stderr_tail = _extract_last_frame(src_video, last_frame)
        if not ok:
            return Err(StageError(
                ErrorCode.IO_ERROR,
                f"could not extract last frame from source_video: {stderr_tail}",
                (("source_video", src_video),)))
        start_image = last_frame

    # Load the start image up front (an unreadable one is DATA, not a crash).
    start_arr = None
    if start_image is not None:
        try:
            with Image.open(start_image) as im:
                start_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception as exc:  # unreadable/corrupt still -> errors-as-data
            return Err(StageError(
                ErrorCode.IO_ERROR,
                f"could not read start_image: {exc}",
                (("start_image", str(start_image)),),
            ))

    frame_dir = None
    tmp_mp4 = None
    try:
        os.makedirs(out_dir, exist_ok=True)
        frame_dir = tempfile.mkdtemp(prefix=".frames-", dir=out_dir)
        for i in range(n_frames):
            # Cooperative mid-render cancel: honored BETWEEN frames. Aborting here
            # (before the atomic os.replace below) guarantees NO clip lands at the
            # content-addressed path; the finally cleans up frame_dir (and there is
            # no tmp_mp4 yet). Mirrors scene.py's per-frame is_cancelling() poll.
            if should_cancel is not None and should_cancel():
                return Err(StageError(
                    ErrorCode.CANCELLED,
                    f"cancelled mid-render after {i} of {n_frames} frame(s)",
                    (("content_hash", content_hash), ("frames", str(n_frames))),
                ))
            frame = synthesize_frame(
                manifest.seeds.global_seed, width, height, i, n_frames, start_arr)
            Image.fromarray(frame).save(
                os.path.join(frame_dir, f"frame_{i:05d}.png"), "PNG")

        # NOTE: the temp name keeps a .mp4 extension so ffmpeg infers the mp4
        # muxer (it keys off the extension); it stays in out_dir for an atomic
        # same-filesystem os.replace onto clip.mp4.
        tmp_mp4 = os.path.join(out_dir, f".clip-tmp-{os.getpid()}.mp4")
        ok, stderr_tail = _assemble_mp4(frame_dir, tmp_mp4, fps)
        if not ok:
            return Err(StageError(
                ErrorCode.ASSEMBLY_FAILED,
                f"ffmpeg mux failed: {stderr_tail}",
                (("content_hash", content_hash), ("frames", str(n_frames))),
            ))

        os.replace(tmp_mp4, clip_path)        # atomic promotion of the clip
        tmp_mp4 = None

        # Sidecars (INV-1/INV-7): the manifest that defines this render + the
        # provenance stub, both written atomically alongside the pixels.
        atomic_write_text(
            os.path.join(out_dir, _MANIFEST_NAME),
            json.dumps(render_manifest_to_dict(manifest), indent=2, sort_keys=True))
        atomic_write_text(
            os.path.join(out_dir, _PROVENANCE_NAME),
            json.dumps(_provenance_dict(manifest), indent=2, sort_keys=True))
    except OSError as exc:                     # unwritable out_root, disk full, ...
        return Err(StageError(
            ErrorCode.IO_ERROR,
            f"synthetic runner IO failure: {exc}",
            (("out_dir", out_dir),),
        ))
    finally:
        if tmp_mp4 is not None and os.path.isfile(tmp_mp4):
            try:
                os.remove(tmp_mp4)
            except OSError:
                pass
        if frame_dir is not None and os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir, ignore_errors=True)

    return Ok(Artifact(
        path=clip_path, content_hash=content_hash, frames=n_frames,
        width=width, height=height, duration_s=duration_s, resumed=False))


# --------------------------------------------------------------------------- #
# SYNTHETIC t2v (Task 3b) — the no-model TEXT-to-video prover
# --------------------------------------------------------------------------- #
def run_synthetic_t2v(
    manifest: RenderManifest,
    out_root: str,
    start_image: str | None = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> Result[Artifact, StageError]:
    """Produce (or resume) a synthetic TEXT-to-video clip for ``manifest``.

    A thin, DETERMINISTIC delegation to ``run_synthetic_i2v`` with
    ``start_image`` forced to None: text-to-video has no conditioning still, so a
    supplied start_image is DELIBERATELY IGNORED (never tints/pans a frame). The
    frames are a pure function of ``manifest.seeds.global_seed`` + geometry — the
    PROMPT rides in the manifest (and thus the content_hash + ``manifest.json``
    sidecar) for provenance, but never alters a pixel, so t2v stays
    byte-deterministic. Identical content-addressed atomic layout / resume /
    errors-as-data as the i2v path (it IS the i2v path). Only a genuine programmer
    error (a non-RenderManifest) raises — inherited from ``run_synthetic_i2v``."""
    return run_synthetic_i2v(
        manifest, out_root, start_image=None, should_cancel=should_cancel)
