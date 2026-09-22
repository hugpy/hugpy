"""Pure `(ffmpeg, audio_extract)` runner — map §4.4 / §6.

`run_audio_extract(spec, job_id) -> JobResult` producing exactly ONE MediaRef
output (a standalone audio file). A single ffmpeg invocation drops the video
stream (`-vn`) and re-encodes the audio into the requested container/codec.

Pure discipline (map §6): EXPECTED failures (missing input, non-video source,
no audio track, ffmpeg nonzero, no output) are returned as
JobResult(ok=False, JobError(...)) — DATA, never a raise. Only the worker loop
may catch an UNEXPECTED raise.

Audio extract is light (no fan-out, one short subprocess) so — unlike
ffmpeg_frames — it needs no concurrency semaphore. The output is written under
DEFAULT_ROOT/video_intel/audio/<uuid>.<fmt> and then re-ingested so its
duration/sample_rate/channels are authoritatively resolved (map §9.2).
"""
from __future__ import annotations

import os
import subprocess
from uuid import uuid4

from hugpy_platform.binaries import resolve_bin
from hugpy_platform.constants import DEFAULT_ROOT

from hugpy_video.intel.audio_schema import AudioExtractSpec
from hugpy_video.intel.media_store import ingest
from hugpy_video.intel.result_schema import JobError, JobResult

_AUDIO_DIR = os.path.join(DEFAULT_ROOT, "video_intel", "audio")

# fmt -> ffmpeg audio codec (map's frozen codec map).
_CODEC = {"wav": "pcm_s16le", "mp3": "libmp3lame", "m4a": "aac"}


def run_audio_extract(spec: AudioExtractSpec, job_id: str) -> JobResult:
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
            message=f"audio_extract requires a video source; got kind={src.kind!r}",
            retryable=False,
        ))
    # No audio track: ingest leaves sample_rate/channels None for a silent video
    # (map §9.2 — the MediaRef is authoritative). Refuse loudly rather than emit
    # an empty file.
    if src.sample_rate is None and src.channels is None:
        return JobResult(job_id, ok=False, error=JobError(
            code="no_audio_track",
            message=f"source video has no audio stream: {in_path}",
            retryable=False,
        ))

    # ---- build the single ffmpeg invocation ----
    codec = _CODEC[spec.fmt]
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    out_path = os.path.join(_AUDIO_DIR, uuid4().hex + "." + spec.fmt)

    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    command = [ffmpeg, "-y", "-i", in_path, "-vn", "-acodec", codec, out_path]

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

    # Re-ingest so the output's duration/sample_rate/channels are authoritative.
    out_ref = ingest(out_path)
    return JobResult(job_id, ok=True, outputs=(out_ref,))
