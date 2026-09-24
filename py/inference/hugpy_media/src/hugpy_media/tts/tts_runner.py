"""Text-to-speech runner — serves ("transformers", "text-to-speech").

WHAT IT IS. The in-fleet dispatch seam for the oracle capability ``audio.tts``:
``POST /oracle/route`` -> ``runtime.execute_route`` -> ``execute_prompt`` ->
this runner on whichever worker advertises ``task_capabilities['text-to-speech']``
-> the k98 adapter (``hugpy_media.tts.chatterbox_runner``) -> a wav.

IT OWNS NO SYNTHESIS LOGIC. Every decision about how a line becomes audio —
the authority refusal, the backend load, the waveform write, the sidecar — is
k98's adapter, unchanged. This module is the three things the adapter
deliberately does NOT do: (1) speak the fleet's Runner protocol, (2) resolve
WHICH interpreter holds the backend (``managers.tts.seat``), and (3) resolve
the registry row's own weights (``ensure_model``) so the bytes that speak are
the bytes the catalog bound, not a second copy the backend downloads for itself.

THE PROCESS SEAM. ``chatterbox-tts`` pins torch 2.6 / transformers 5.2; a
worker's venv serves every other model on the box on a different torch. So when
the backend is not importable HERE, synthesis runs as a child of the fleet's
per-model env-PROFILE venv (``managers/serve/profiles.py``) via
``_backend_main.py``. Same adapter file, different interpreter. When the backend
IS importable here, it runs in-process and no child is spawned — the two paths
produce the identical manifest because they call the identical function.

DURATION IS MEASURED, NEVER CLAIMED (doc invariant 11 / k102 rule 1): the value
returned is read back out of the wav header that was actually written.

AUTHORITY. A ``reference_audio`` without ``authorized=True`` is refused by
``make_tts`` before a backend is loaded, and surfaces here as
``ok=False, error_code="missing_consent"`` — error-as-data, which
``oracle.runtime.execute_route`` classifies as a RUNNER_ERROR receipt rather
than an artifact. It never downgrades to the default voice.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import time
import wave
from typing import Any

from hugpy_media.tts.schemas import SynthesizedAudio, TtsRequest, TtsResult
from hugpy_media.tts import seat as _seat

logger = logging.getLogger(__name__)

#: The k98 adapter — the ONE implementation, addressed as a module here and as a
#: file path in the child.
ADAPTER_MODULE = "hugpy_media.tts.chatterbox_runner"

#: How long one synthesis child may run before it is killed. A cold seat pays
#: for a multi-GB checkpoint load; the media-bus job budget for this runner
#: (JOB_TIMEOUT_S) is 1800s and this mirrors it, so the two never disagree about
#: what "too long" means. The oracle's own, much shorter, HTTP deadline
#: (ORACLE_SYNC_DEADLINE_S) bounds the CALLER independently — that is the point
#: of k101b's bound, and it is not this runner's job to duplicate it.
CHILD_TIMEOUT_S = 1800.0


def _adapter():
    """The k98 adapter module (lazy: importing this runner must not cost it)."""
    from hugpy_media.tts import chatterbox_runner
    return chatterbox_runner


def _adapter_path() -> str:
    """The adapter's FILE path — what the profile-venv child loads, since the
    package itself is not installed there."""
    return _adapter().__file__


from hugpy_media.tts.paths import output_dir as _output_dir


def _weights_dir(model_key: str) -> str | None:
    """The registry row's own checkpoint dir, provisioned if absent — the same
    ``ensure_model`` chokepoint every other transformers runner loads through.

    None (with a logged warning) when the row cannot be resolved on this box:
    the adapter then falls back to the backend's own default download and SAYS
    SO in the sidecar (``weights_source``), instead of quietly serving different
    bytes under this model_id."""
    try:
        from hugpy_storage.download_models import ensure_model
        directory = ensure_model(model_key)
    except Exception as exc:  # noqa: BLE001 — resolution failure is not fatal
        logger.warning("tts: could not resolve weights for %s (%s: %s); the "
                       "backend will load its own default checkpoint",
                       model_key, type(exc).__name__, exc)
        return None
    return directory if directory and os.path.isdir(directory) else None


def _wav_facts(path: str) -> tuple[int, float]:
    """(sample_rate, duration_s) READ BACK from the written wav. The backend's
    claim is never trusted for either — a duration that came from anywhere but
    the file is the exact failure mode invariant 11 exists to stop."""
    with wave.open(path, "rb") as fh:
        rate = fh.getframerate()
        frames = fh.getnframes()
    return int(rate), (frames / float(rate) if rate else 0.0)


def _spec_kwargs(req: TtsRequest, weights_dir: str | None) -> dict[str, Any]:
    """The adapter's ``make_tts`` kwargs, 1:1 with the request fields it mirrors."""
    return {
        "text": req.text,
        "reference_audio": req.reference_audio,
        "authorized": bool(req.authorized),
        "voice_style": req.voice_style,
        "seed": req.seed,
        "language": req.language,
        "device": req.device,
        "model_id": req.model_key,
        "weights_dir": weights_dir,
    }


def _run_in_child(python: str, spec_kwargs: dict[str, Any],
                  out_dir: str) -> dict[str, Any]:
    """Synthesize in the profile venv. Returns the child's payload dict."""
    child = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_backend_main.py")
    job = json.dumps({"module_path": _adapter_path(), "out_dir": out_dir,
                      "spec": spec_kwargs})
    # The child writes into the SHARED output dir; it inherits nothing from this
    # process's environment that could re-shadow its own torch (no PYTHONPATH —
    # see _backend_main's module docstring).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([python, child], input=job, capture_output=True,
                          text=True, timeout=CHILD_TIMEOUT_S, env=env)
    marker = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("@@TTS_RESULT@@"):
            marker = line[len("@@TTS_RESULT@@"):]
    if marker is None:
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        return {"ok": False, "error_code": "deps_missing",
                "error": (f"the chatterbox child ({python}) produced no result "
                          f"(rc={proc.returncode}): {tail}")}
    return json.loads(marker)


def _run_in_process(spec_kwargs: dict[str, Any], out_dir: str) -> dict[str, Any]:
    """Synthesize here — the same adapter calls, no child."""
    adapter = _adapter()
    started = time.monotonic()
    try:
        spec = adapter.make_tts(**spec_kwargs)
        manifest = adapter.synthesize(spec, out_dir)
    except Exception as exc:  # noqa: BLE001 — errors as data, like the child
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "error_code": getattr(exc, "code", None) or "tts_error",
                "elapsed_s": round(time.monotonic() - started, 3)}
    payload: dict[str, Any] = {"ok": True, "manifest": manifest,
                               "elapsed_s": round(time.monotonic() - started, 3)}
    try:
        import torch
        if torch.cuda.is_available():
            payload["vram_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            payload["vram_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 — a measurement never fails the work
        pass
    return payload


from hugpy_platform.runner_config import RunnerConfig


class ChatterboxTtsRunner(RunnerConfig):
    """Runner for the chatterbox TTS backend.

    Stateless by design: the backend lives in a child process whose lifetime is
    ONE synthesis, so there is no pipeline singleton to cache here (the imagegen
    precedent caches because its pipeline shares this process; this one cannot).
    That costs a checkpoint load per call — recorded honestly in the result's
    timings, and the first thing a warm-sidecar follow-up should fix."""

    request_type = TtsRequest
    result_type = TtsResult


    # --- synthesis ----------------------------------------------------------

    def _synthesize(self, req: TtsRequest) -> TtsResult:
        model_key = req.model_key or self.model_key
        placement = _seat.resolve()
        if not placement.get("available"):
            return TtsResult(
                request_id=req.request_id, model_key=model_key, ok=False,
                error_code="deps_missing",
                error=(f"this box does not seat the chatterbox TTS backend: "
                       f"{placement.get('reason')}"))

        out_dir = _output_dir()
        os.makedirs(out_dir, exist_ok=True)
        spec_kwargs = _spec_kwargs(req, _weights_dir(model_key))

        if placement["mode"] == "in-process":
            payload = _run_in_process(spec_kwargs, out_dir)
        else:
            payload = _run_in_child(placement["python"], spec_kwargs, out_dir)

        if not payload.get("ok"):
            return TtsResult(
                request_id=req.request_id, model_key=model_key, ok=False,
                error=str(payload.get("error") or "synthesis failed"),
                error_code=str(payload.get("error_code") or "tts_error"),
                vram_peak_bytes=payload.get("vram_peak_reserved_bytes"))

        manifest = dict(payload.get("manifest") or {})
        audio_path = manifest.get("audio_path") or ""
        if not audio_path or not os.path.isfile(audio_path) or \
                os.path.getsize(audio_path) == 0:
            return TtsResult(
                request_id=req.request_id, model_key=model_key, ok=False,
                error_code="missing_output",
                error=(f"synthesis reported success but produced no audio at "
                       f"{audio_path!r}"))

        sample_rate, duration_s = _wav_facts(audio_path)
        b64 = None
        if req.return_b64:
            with open(audio_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")

        sidecar = {k: v for k, v in manifest.items()
                   if k not in ("audio_path", "sidecar_path")}
        sidecar["synthesis_elapsed_s"] = payload.get("elapsed_s")
        sidecar["seat_mode"] = placement["mode"]
        for key in ("vram_peak_reserved_bytes", "vram_peak_allocated_bytes",
                    "vram_device_total_bytes", "torch_version"):
            if payload.get(key) is not None:
                sidecar[key] = payload[key]

        clip = SynthesizedAudio(
            path=audio_path, b64=b64, sample_rate=sample_rate,
            duration_s=round(duration_s, 6), seed=req.seed,
            reference_used=bool(req.reference_audio), sidecar=sidecar)
        return TtsResult(
            request_id=req.request_id, model_key=model_key, ok=True,
            audio=[clip],
            text=f"[{duration_s:.2f}s @ {sample_rate} Hz] {req.text}",
            vram_peak_bytes=payload.get("vram_peak_reserved_bytes"))

    # --- public API ---------------------------------------------------------

    async def run(self, req: TtsRequest) -> TtsResult:
        try:
            return await asyncio.to_thread(self._synthesize, req)
        except Exception as exc:  # noqa: BLE001 — the boundary returns data
            logger.warning("tts runner failed for %s: %s: %s",
                           req.model_key, type(exc).__name__, exc)
            return TtsResult(
                request_id=req.request_id,
                model_key=req.model_key or self.model_key, ok=False,
                error=f"{type(exc).__name__}: {exc}", error_code="tts_error")


__all__ = ["ADAPTER_MODULE", "CHILD_TIMEOUT_S", "ChatterboxTtsRunner"]
