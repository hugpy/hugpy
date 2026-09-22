"""Pure `(ffmpeg, crop)` runner — map §6 dispatch row 3.

`run_crop(spec, job_id) -> JobResult`. Branches on which axes the CropSpec
carries:
    spatial present  -> ffmpeg  -vf crop=w:h:x:y
    temporal present -> ffmpeg  -ss start -t (end-start)   (trim)
    both (video)     -> both

Pure discipline (map §6): EXPECTED failures (bad bbox, ffmpeg nonzero, missing
output) are returned as JobResult(ok=False, error=JobError(...)) — data, never a
raise. Only the worker loop is allowed to catch an UNEXPECTED raise.

The output is written under DEFAULT_ROOT/video_intel/crops/<uuid>.<ext> and then
re-ingested so its dimensions/duration are authoritatively resolved.
"""
from __future__ import annotations

import os
import subprocess
from uuid import uuid4

from hugpy_platform.binaries import resolve_bin
from hugpy_platform.constants import DEFAULT_ROOT

from hugpy_video.intel.crop_schema import CropSpec
from hugpy_video.intel.media_store import ingest
from hugpy_video.intel.result_schema import JobError, JobResult

_CROPS_DIR = os.path.join(DEFAULT_ROOT, "video_intel", "crops")


def _fmt_num(x) -> str:
    """Compact numeric string for ffmpeg (avoid '1.5000000000001' noise)."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return repr(x) if isinstance(x, float) else str(x)


def run_crop(spec: CropSpec, job_id: str) -> JobResult:
    src = spec.source
    in_path = src.uri

    if not os.path.isfile(in_path):
        return JobResult(job_id, ok=False, error=JobError(
            code="missing_input",
            message=f"source file does not exist: {in_path}",
            retryable=False,
        ))

    # ---- validate the spatial bbox against known dims (expected failure = data) ----
    if spec.spatial is not None:
        r = spec.spatial
        if r.w <= 0 or r.h <= 0 or r.x < 0 or r.y < 0:
            return JobResult(job_id, ok=False, error=JobError(
                code="region_out_of_bounds",
                message=f"spatial region must be non-negative with positive size; got {r}",
                retryable=False,
            ))
        if src.width is not None and src.height is not None:
            if r.x + r.w > src.width or r.y + r.h > src.height:
                return JobResult(job_id, ok=False, error=JobError(
                    code="region_out_of_bounds",
                    message=(
                        f"spatial region {r} exceeds source "
                        f"{src.width}x{src.height}"
                    ),
                    retryable=False,
                ))

    # ---- validate the temporal interval (expected failure = data) ----
    if spec.temporal is not None:
        t = spec.temporal
        if t.end_s <= t.start_s or t.start_s < 0:
            return JobResult(job_id, ok=False, error=JobError(
                code="region_out_of_bounds",
                message=f"temporal region must satisfy 0 <= start < end; got {t}",
                retryable=False,
            ))

    # ---- build the output path (keep the source container/extension) ----
    ext = os.path.splitext(in_path)[1] or ".bin"
    os.makedirs(_CROPS_DIR, exist_ok=True)
    out_path = os.path.join(_CROPS_DIR, uuid4().hex + ext)

    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    command = [ffmpeg, "-y"]
    # input-side seek is accurate for these containers and cheap
    if spec.temporal is not None:
        command += ["-ss", _fmt_num(spec.temporal.start_s)]
    command += ["-i", in_path]
    if spec.temporal is not None:
        command += ["-t", _fmt_num(spec.temporal.end_s - spec.temporal.start_s)]
    if spec.spatial is not None:
        r = spec.spatial
        command += ["-vf", f"crop={r.w}:{r.h}:{r.x}:{r.y}"]
    command += [out_path]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return JobResult(job_id, ok=False, error=JobError(
            code="ffmpeg_failed",
            message=(
                f"ffmpeg exited {result.returncode}.\n"
                f"cmd: {' '.join(command)}\n"
                f"stderr:\n{result.stderr}"
            ),
            retryable=False,
        ))

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        return JobResult(job_id, ok=False, error=JobError(
            code="missing_output",
            message=f"ffmpeg reported success but produced no output at {out_path}",
            retryable=False,
        ))

    # Re-ingest so the output's dims/duration are authoritative (map §9.2).
    out_ref = ingest(out_path)
    return JobResult(job_id, ok=True, outputs=(out_ref,))
