"""Pure `(ffmpeg, frame_extract)` runner — map §4.3 / §6.

`run_frame_extract(spec, job_id) -> JobResult` producing MANY MediaRef outputs
(one per extracted frame). A single ffmpeg invocation with the `fps` filter does
the sampling (simplest — one subprocess); an optional temporal window is applied
input-side (`-ss`) + output-length (`-t`), mirroring ffmpeg_crop.py.

Pure discipline (map §6): EXPECTED failures (missing input, non-video source,
the LOUD max_frames cap, ffmpeg nonzero, no output) are returned as
JobResult(ok=False, JobError(...)) — DATA, never a raise. Only the worker loop
may catch an UNEXPECTED raise.

max_frames is a LOUD cap (map §4.3): we compute the expected frame count up
front and REFUSE the whole job if it exceeds the cap — we never silently
truncate.

CONCURRENCY BOUND: a prior run flagged heavy frame_extract fan-outs starving
other jobs. A module-level BoundedSemaphore(1) serializes the ffmpeg subprocess
so at most one frame_extract fan-out runs at a time per process. It is
deliberately minimal/additive; the surrounding validation stays outside the
semaphore so cheap rejections never wait on a running extraction.
"""
from __future__ import annotations

import glob
import math
import os
import subprocess
import threading
from typing import Optional

from hugpy_platform.binaries import resolve_bin
from hugpy_platform.constants import DEFAULT_ROOT

from hugpy_video.intel.frame_schema import FrameExtractSpec
from hugpy_video.intel.media_store import ingest
from hugpy_video.intel.result_schema import JobError, JobResult

_FRAMES_DIR = os.path.join(DEFAULT_ROOT, "video_intel", "frames")

# See module docstring: serialize the (heavy) ffmpeg fan-out to one at a time
# per process so a big extraction can't starve other queued jobs.
_FRAME_SEM = threading.BoundedSemaphore(1)

# fmt -> file extension for the output pattern.
_EXT = {"jpg": "jpg", "png": "png", "webp": "webp"}


from hugpy_platform.formatting import ffmpeg_num as _fmt_num


def _mjpeg_qscale(quality: int) -> int:
    """Map the schema's 1..100 (higher=better) to ffmpeg's -qscale:v 2..31
    (lower=better) used by mjpeg/webp. quality=100 -> 2 (best), 1 -> 31."""
    q = max(1, min(100, int(quality)))
    return int(round(31 - (q - 1) / 99.0 * 29))


def run_frame_extract(spec: FrameExtractSpec, job_id: str) -> JobResult:
    src = spec.source
    in_path = src.uri

    if not os.path.isfile(in_path):
        return JobResult(job_id, ok=False, error=JobError(
            code="missing_input",
            message=f"source file does not exist: {in_path}",
            retryable=False,
        ))
    if src.kind != "video":
        return JobResult(job_id, ok=False, error=JobError(
            code="not_a_video",
            message=f"frame_extract requires a video source; got kind={src.kind!r}",
            retryable=False,
        ))

    # ---- LOUD max_frames cap (map §4.3: refuse, don't truncate) ----
    # window given -> its length; else the full source duration.
    duration: Optional[float]
    if spec.window is not None:
        duration = spec.window.end_s - spec.window.start_s
    else:
        duration = src.duration_s
    if spec.max_frames is not None and duration is not None and duration > 0:
        expected = math.ceil(spec.fps * duration)
        if expected > spec.max_frames:
            return JobResult(job_id, ok=False, error=JobError(
                code="frame_cap_exceeded",
                message=(
                    f"frame_extract would produce ~{expected} frames "
                    f"(fps={spec.fps} * {duration:.3f}s) > cap {spec.max_frames}; "
                    "refusing (raise max_frames or narrow the window)"
                ),
                retryable=False,
            ))

    # ---- build the single ffmpeg invocation ----
    ext = _EXT[spec.fmt]
    out_dir = os.path.join(_FRAMES_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, f"frame_%05d.{ext}")

    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    command = [ffmpeg, "-y"]
    # input-side seek (accurate + cheap for these containers), then output length
    if spec.window is not None:
        command += ["-ss", _fmt_num(spec.window.start_s)]
    command += ["-i", in_path]
    if spec.window is not None:
        command += ["-t", _fmt_num(spec.window.end_s - spec.window.start_s)]
    command += ["-vf", f"fps={_fmt_num(spec.fps)}"]
    # per-fmt quality flag (see make_frame_extract for the validated ranges)
    if spec.fmt == "png":
        command += ["-compression_level", str(spec.quality)]
    else:  # jpg / webp -> qscale:v (mapped from 1..100 to mjpeg's 2..31)
        command += ["-qscale:v", str(_mjpeg_qscale(spec.quality))]
    command += [out_pattern]

    with _FRAME_SEM:  # serialize the heavy fan-out (see module docstring)
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

    frames = sorted(glob.glob(os.path.join(out_dir, f"frame_*.{ext}")))
    if not frames:
        return JobResult(job_id, ok=False, error=JobError(
            code="missing_output",
            message=(
                f"ffmpeg reported success but produced no frames in {out_dir}"
            ),
            retryable=False,
        ))

    # Re-ingest each frame so its dims/mime are authoritatively resolved (§9.2).
    refs = tuple(ingest(p) for p in frames)
    return JobResult(job_id, ok=True, outputs=refs)
