"""Media-bus entrypoint for Chatterbox TTS.

The synthesis adapter itself (``TtsSpec``, ``make_tts``, ``probe``,
``synthesize`` and the ``TtsError`` family) lives in
``hugpy_media.tts.chatterbox_runner`` — media owns text-to-speech. This module
is the bus glue video owns: it turns expected failures into ``JobResult`` /
``JobError`` data and ingests the wav into the media store. It keeps the
historical module path so the dispatch table and the oracle tests keep
addressing ``hugpy_video.intel.runners.tts_chatterbox``.
"""

from __future__ import annotations

import os
from typing import Any

from hugpy_media.tts.chatterbox_runner import (  # noqa: F401 - re-exported surface
    BACKEND_PACKAGE,
    BACKEND_PIP,
    JOB_NAME,
    JOB_QUEUE,
    JOB_TIMEOUT_S,
    MODEL_ID,
    MTL_T3_MODEL,
    PITCH_ANALYSIS_FORBIDDEN,
    RUNNER_KEY,
    TASK,
    ReferenceVoiceUnauthorized,
    TtsBackendUnavailable,
    TtsError,
    TtsSpec,
    TtsSpecError,
    _flatten,
    _load_backend,
    _pcm_levels,
    _to_pcm16,
    _write_wav,
    make_tts,
    probe,
    synthesize,
)


def run_tts_chatterbox(spec: Any, job_id: str):
    """``(spec, job_id) -> JobResult``, the media-bus runner signature.

    Accepts a ``TtsSpec`` or a plain dict (the bus rehydrates specs from JSON).
    Every expected failure — unauthorized reference, absent backend, malformed
    spec, empty output — returns ``JobResult(ok=False, JobError(...))``; only a
    genuine programmer error escapes as a raise for the worker loop to catch."""
    from hugpy_video.intel.media_store import ingest
    from hugpy_video.intel.result_schema import JobError, JobResult

    try:
        if isinstance(spec, dict):
            spec = make_tts(**spec)
        elif not isinstance(spec, TtsSpec):
            raise TtsSpecError(
                f"run_tts_chatterbox needs a TtsSpec or dict; got "
                f"{type(spec).__name__}")
        result = synthesize(spec)
    except TtsError as exc:
        return JobResult(job_id, ok=False, error=JobError(**exc.error_fields()))
    except (OSError, ValueError) as exc:
        return JobResult(job_id, ok=False, error=JobError(
            code="io_error", message=f"{type(exc).__name__}: {exc}",
            retryable=False))

    audio_path = result["audio_path"]
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        return JobResult(job_id, ok=False, error=JobError(
            code="missing_output",
            message=f"synthesis reported success but produced no audio at "
                    f"{audio_path}",
            retryable=False))
    return JobResult(job_id, ok=True, outputs=(ingest(audio_path),))


__all__ = [
    "BACKEND_PACKAGE", "BACKEND_PIP", "JOB_NAME", "JOB_QUEUE", "JOB_TIMEOUT_S",
    "MODEL_ID", "PITCH_ANALYSIS_FORBIDDEN", "RUNNER_KEY", "TASK",
    "ReferenceVoiceUnauthorized", "TtsBackendUnavailable", "TtsError",
    "TtsSpec", "TtsSpecError", "make_tts", "probe", "run_tts_chatterbox",
    "synthesize",
]
