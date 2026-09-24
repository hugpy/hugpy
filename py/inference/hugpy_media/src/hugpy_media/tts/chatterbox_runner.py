"""Chatterbox TTS runner (k98) — reference-conditioned speech, as an adapter.

WHAT THIS IS. The media-bus runner behind the oracle capability ``audio.tts``
(alias ``voice.synthesize.reference_conditioned``, doc §4). It turns a locked
line of dialogue into ONE wav artifact plus a sidecar JSON that records exactly
how it was made. Chatterbox (resemble-ai, MIT) is the doc §17 reference target;
its multilingual V3 weights already sit on the shared store as the registry row
``Viral2AI~chatterbox`` — what is missing is the ``chatterbox`` PYTHON PACKAGE
and a GPU, neither of which exists on central. Hence ``probe()``.

INVARIANT — CHATTERBOX IS NEVER A PITCH ANALYSER (doc Stage 3, stated verbatim:
"Chatterbox or another reference-conditioned TTS/voice-conversion backend may
synthesize authorized speech. It must not be used as the pitch-analysis
component."). This module therefore exposes exactly one verb — SYNTHESIZE — and
has no analysis entrypoint, no f0/prosody return value and no place to add one.
Pitch contour, prosody, cadence and speaker embeddings come from the analysis
capabilities (``voice.analyze.prosody`` / ``audio.speaker_similarity``), which
are separate rows in the catalog for exactly this reason. If a later slice needs
prosody, it adds an analyser; it does not add a method here.

AUTHORITY — REFERENCE VOICE REQUIRES ``authorized=True``. Cloning a specific
person's voice is a rights decision, not a rendering parameter. The TYPED gate
is k97's (``oracle.authority``: ``AuthorityKind.VOICE`` over a ``RightsManifest``,
refusing the route before a model is picked). This module is the LAST line, not
the first: a spec carrying ``reference_audio`` without ``authorized=True`` is
refused HERE TOO, before the backend is even imported, so no call path — a
direct runner call, a test, a future orchestrator that forgets the gate — can
reach reference-conditioned synthesis without an explicit authorization flag.
Defence in depth on a legal invariant is not redundancy.

ERRORS. ``ReferenceVoiceUnauthorized`` (policy) and ``TtsBackendUnavailable``
(the standard "backend unavailable": the tree spells it ``deps_missing`` —
``studio.errors.ErrorCode.DEPS_MISSING`` for the studio spine, ``JobError(code=
"deps_missing")`` for the media bus) are RAISED by the low-level ``synthesize``,
because that function is called directly by orchestrators for whom a silent
degradation would be catastrophic. The bus entrypoint ``run_tts_chatterbox``
converts both into ``JobResult(ok=False, error=JobError(...))`` — map §6's law
that an expected failure crosses a module boundary as DATA, never a raise.

IMPORT DISCIPLINE, STRICTER THAN ITS SIBLINGS. This module's own top level is
STDLIB ONLY: even ``result_schema`` / ``media_store`` / ``DEFAULT_ROOT`` are
imported lazily inside the functions that need them (``ffmpeg_audio`` imports
all three at module top; this one deliberately does not). Reason: the ORACLE
CATALOG imports this module on every ``GET /oracle/capabilities`` merely to call
``probe()``, so ``probe()`` must cost a ``find_spec`` and nothing else. (The
package ``__init__`` on the import PATH still pulls its siblings — that is the
bus's dispatch table, not this module's doing, and it is already paid for in any
process that has the app loaded.) The ``chatterbox`` backend itself is imported
lazily inside ``synthesize``, never at module top, exactly like the Wan runners
import torch.

REGISTRATION. This module is NOT yet wired into the bus dispatch table:
``runners/__init__.py`` and ``video_intel/job_schema.py`` were dirty with
another agent's work when k98 landed. The two one-line edits are recorded in the
k98 dispatch record and in ``RUNNER_KEY`` / ``JOB_NAME`` below. Until they land,
``audio.tts`` is reachable through this module's functions and is listed by the
catalog as INELIGIBLE with the precise reason — which is the honest state.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import struct
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

# --------------------------------------------------------------------------- #
# Registration facts (registries over globals — the values the two one-line
# edits below need). Kept as constants so the dispatch table and the oracle
# catalog read the SAME strings instead of retyping them.
#
#   runners/__init__.py :  from .tts_chatterbox import run_tts_chatterbox
#                          (hugpy_video wraps this module's make_tts/synthesize)
#                          DISPATCH[("chatterbox", "tts")] = run_tts_chatterbox
#   job_schema.py       :  "tts_chatterbox": JobSpec("tts_chatterbox", TtsSpec,
#                              ("chatterbox", "tts"), "gpu", 1800)
# --------------------------------------------------------------------------- #

RUNNER_KEY: tuple[str, str] = ("chatterbox", "tts")
JOB_NAME: str = "tts_chatterbox"
JOB_QUEUE: str = "gpu"
JOB_TIMEOUT_S: int = 1800

#: The pip distribution and the import name (they differ — a classic probe bug).
BACKEND_PIP: str = "chatterbox-tts"
BACKEND_PACKAGE: str = "chatterbox"

#: The legacy-registry model key whose weights back this runner, and the legacy
#: dispatch task string the fleet advertises for it. The oracle catalog binds
#: ``audio.tts`` to exactly this pair.
MODEL_ID: str = "Viral2AI~chatterbox"
TASK: str = "text-to-speech"

#: Chatterbox emits 24 kHz mono. Only a FALLBACK: the sidecar always records
#: ``model.sr``, the backend's own authoritative rate.
FALLBACK_SAMPLE_RATE: int = 24000

#: Multilingual V3 checkpoint tag (README "Multilingual Quickstart"). Used only
#: when a ``language`` is requested — English takes the single-language class.
MTL_T3_MODEL: str = "v3"

#: Loud, greppable statement of the doc Stage 3 invariant.
PITCH_ANALYSIS_FORBIDDEN: str = (
    "Chatterbox is a SYNTHESIS backend only; pitch/prosody analysis must come "
    "from the analysis capabilities (voice.analyze.prosody / "
    "audio.speaker_similarity), never from this runner (doc Stage 3)")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TtsError(RuntimeError):
    """Base for this runner's raised failures. ``code`` is the media-bus
    ``JobError.code`` the bus entrypoint reports, so a raise and a JobResult
    always name the same failure."""

    code: str = "tts_error"
    retryable: bool = False

    def error_fields(self) -> dict:
        """This exception as error-as-data: the ``code`` / ``message`` /
        ``retryable`` triple the media bus's ``JobError`` is built from
        (``hugpy_video.intel.runners.tts_chatterbox`` does the wrapping — this
        package never imports the bus)."""
        return {"code": self.code, "message": str(self),
                "retryable": self.retryable}


class TtsBackendUnavailable(TtsError):
    """The standard "backend unavailable" for this runner: the ``chatterbox``
    package (or its torch stack) is not importable in THIS process.

    ``code`` is ``deps_missing`` — the tree's existing spelling for it
    (``studio.errors.ErrorCode.DEPS_MISSING``, which the Wan runners return as
    ``Err`` data, and which the media bus classifies as not-retryable because a
    re-run on the same box fails identically). Retryable is False for exactly
    that reason: seating the runner on a worker is an operator action, not a
    backoff."""

    code = "deps_missing"
    retryable = False


class ReferenceVoiceUnauthorized(TtsError):
    """A reference voice was supplied without ``authorized=True``.

    ``code`` is ``missing_consent`` — the studio's existing LEGAL-1 code
    (``ErrorCode.MISSING_CONSENT``). Never retryable: the fix is an
    authorization, not a repeat."""

    code = "missing_consent"
    retryable = False


class TtsSpecError(TtsError):
    """A malformed spec (no text, negative seed, missing reference file). Local
    to construction, like every other ``make_*`` factory in video_intel."""

    code = "bad_spec"
    retryable = False


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TtsSpec:
    """One line of speech to synthesize.

        text             the line to speak (required, non-blank)
        reference_audio  path to the authorized reference voice, or None for
                         the model's default (non-identifying) voice
        authorized       explicit authorization for ``reference_audio``. False
                         with a reference set is a REFUSAL, never a downgrade
                         to the default voice (a silent substitution would be a
                         worse lie than an error — doc invariant 12)
        voice_style      free-text delivery note recorded in the sidecar and
                         passed to the backend when it accepts one
        seed             determinism seed; None means the backend's own
        language         BCP-47-ish language id ("fr", "zh"); None = English
                         single-language model
        device           "cuda" / "cpu" / None (auto-detect)
        model_id         registry key recorded in the sidecar
        weights_dir      the LOCAL checkpoint directory for ``model_id`` (the
                         registry row's own weights, as resolved by whoever
                         seats this runner). None means "let the backend fetch
                         its own default checkpoint" — which is a DIFFERENT set
                         of bytes than the row the catalog bound, so the sidecar
                         records which of the two was used and never conflates
                         them.
    """

    text: str
    reference_audio: str | None = None
    authorized: bool = False
    voice_style: str | None = None
    seed: int | None = None
    language: str | None = None
    device: str | None = None
    model_id: str = MODEL_ID
    weights_dir: str | None = None


def make_tts(text: str, reference_audio: str | None = None,
             authorized: bool = False, voice_style: str | None = None,
             seed: int | None = None, language: str | None = None,
             device: str | None = None, model_id: str = MODEL_ID,
             weights_dir: str | None = None) -> TtsSpec:
    """Validate + build a TtsSpec. Raises are fine here: construction-time and
    local, never across a boundary (map §4/§6). Also the reconstruction path a
    bus would use to rehydrate the spec from JSON — kwargs line up 1:1 with the
    fields.

    The authorization refusal is checked HERE as well as in ``synthesize`` so a
    caller that builds a spec at plan time learns about the missing release
    before a GPU is ever reserved."""
    if not str(text or "").strip():
        raise TtsSpecError("tts spec needs non-blank 'text'")
    if seed is not None and int(seed) < 0:
        raise TtsSpecError(f"seed must be non-negative; got {seed!r}")
    if reference_audio is not None:
        if not authorized:
            raise ReferenceVoiceUnauthorized(_UNAUTHORIZED_MESSAGE)
        if not os.path.isfile(reference_audio):
            raise TtsSpecError(
                f"reference_audio does not exist: {reference_audio}")
    if weights_dir is not None and not os.path.isdir(weights_dir):
        # A named-but-absent checkpoint dir is a spec fault, not a silent
        # downgrade: falling back to the backend's own download would serve
        # DIFFERENT bytes than the registry row this spec names.
        raise TtsSpecError(f"weights_dir does not exist: {weights_dir}")
    return TtsSpec(text=str(text), reference_audio=reference_audio,
                   authorized=bool(authorized), voice_style=voice_style,
                   seed=None if seed is None else int(seed),
                   language=language, device=device, model_id=model_id,
                   weights_dir=weights_dir)


_UNAUTHORIZED_MESSAGE = (
    "reference-conditioned synthesis refused: reference_audio was supplied "
    "without authorized=True. Cloning a specific voice requires an explicit "
    "voice authorization (oracle.authority: AuthorityKind.VOICE over the "
    "request's RightsManifest). Supply the authorization, or omit "
    "reference_audio to use the model's default non-identifying voice — this "
    "runner will NOT silently downgrade one to the other.")


# --------------------------------------------------------------------------- #
# Registration probe — what the oracle catalog consults
# --------------------------------------------------------------------------- #


def probe() -> dict[str, Any]:
    """Can this process actually run the chatterbox backend?

    ``{"importable": bool, "reason": str, ...}``. ``find_spec`` ONLY — never an
    import — for the same reason ``managers/task_deps.have`` and ``ml_routes._have``
    use it: a capability listing must be cheap, must not load a multi-GB torch
    stack, and must not crash when the package is half-installed. ``reason`` is
    empty exactly when ``importable`` is True.

    This is the "adapter registration and health probe" of doc §4 step 3. It is
    a fact about THIS process: a worker that has the package answers True from
    its own probe; central answers False and the catalog says so out loud."""
    importable = False
    reason = ""
    try:
        importable = importlib.util.find_spec(BACKEND_PACKAGE) is not None
    except (ImportError, ValueError) as exc:      # half-installed / namespace mess
        importable = False
        reason = (f"python package {BACKEND_PACKAGE!r} is present but not "
                  f"loadable ({type(exc).__name__}: {exc})")
    if not importable and not reason:
        reason = (f"python package {BACKEND_PACKAGE!r} is not installed in this "
                  f"interpreter (pip install {BACKEND_PIP}); central has no GPU "
                  f"and does not seat this runner — a worker must")
    return {
        "importable": importable,
        "reason": reason,
        "package": BACKEND_PACKAGE,
        "pip": BACKEND_PIP,
        "runner_key": RUNNER_KEY,
        "job_name": JOB_NAME,
        "model_id": MODEL_ID,
        "task": TASK,
        # Stated in the probe payload so any consumer that only reads this dict
        # still learns the invariant.
        "pitch_analysis": False,
        "pitch_analysis_note": PITCH_ANALYSIS_FORBIDDEN,
    }


# --------------------------------------------------------------------------- #
# Waveform -> wav (stdlib only: no numpy/soundfile/torchaudio requirement)
# --------------------------------------------------------------------------- #


def _flatten(value: Any) -> list[float]:
    """Any backend waveform -> a flat list of python floats.

    Handles, in order: torch tensors (``detach``/``cpu``/``tolist``), numpy
    arrays (``tolist``), nested sequences (mono channel inside a batch dim, the
    shape chatterbox returns), and flat sequences. Deliberately duck-typed —
    this module must not import torch or numpy to normalize a buffer."""
    for attr in ("detach", "cpu"):
        if hasattr(value, attr):
            try:
                value = getattr(value, attr)()
            except Exception:  # noqa: BLE001 — a buffer we cannot narrow is fine
                pass
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        return [float(s) for s in
                struct.unpack(f"<{len(bytes(value)) // 2}h", bytes(value))]
    out: list[float] = []
    stack = [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        else:
            out.append(float(item))
    return out


def _to_pcm16(samples: list[float]) -> bytes:
    """Float [-1, 1] or already-int16 samples -> little-endian PCM16 bytes.

    The scale is inferred from what the numbers ARE, not from how loud they are:
    a buffer whose every value is a whole number AND whose peak exceeds unity is
    already integer PCM and is only clipped; everything else is float audio and
    is scaled by 32767. Float overshoot past 1.0 is CLIPPED (the pack below
    clamps), never re-read as integer PCM.

    WHY NOT ``peak <= 1.0`` — the bug this replaced, measured live on this fleet
    2026-08-21. Chatterbox returns float32 that OVERSHOOTS unity: the PerTh
    watermarker adds energy to an already near-full-scale waveform, so a real
    line ("speak line l1 as ana", seed 2733992527) came back with peak
    1.011981 on TWO samples out of 58 560. The old peak test therefore declared
    the whole buffer "already integer PCM", multiplied it by 1.0, and every
    float in [-1, 1] rounded to -1/0/1 — a valid PCM16 wav of exactly the right
    duration whose peak amplitude was 1 and whose RMS was -117 dBFS. Digital
    silence, on roughly one line in four, scoring hard_pass. Clipping two
    samples costs nothing audible; misreading them costs the whole utterance."""
    if not samples:
        return b""
    integral = all(float(s).is_integer() for s in samples)
    scale = 1.0 if integral and max(abs(s) for s in samples) > 1.0 else 32767.0
    frames = bytearray()
    for sample in samples:
        value = int(round(sample * scale))
        frames += struct.pack("<h", max(-32768, min(32767, value)))
    return bytes(frames)


def _pcm_levels(pcm: bytes) -> tuple[int, float]:
    """(peak, rms) of PCM16 bytes in int16 units — the two numbers that tell a
    spoken wav from a correctly-shaped silence."""
    count = len(pcm) // 2
    if not count:
        return 0, 0.0
    values = struct.unpack(f"<{count}h", pcm[:count * 2])
    peak = max(max(values), -min(values))
    rms = math.sqrt(sum(float(v) * v for v in values) / count)
    return peak, rms


def _write_wav(path: str, samples: list[float], sample_rate: int
               ) -> tuple[float, int, float]:
    """Write mono PCM16; return (duration_s, peak, rms), ALL THREE measured off
    the bytes that were actually written (invariant 11, applied to loudness as
    well as to duration)."""
    pcm = _to_pcm16(samples)
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm)
    peak, rms = _pcm_levels(pcm)
    duration = (len(pcm) // 2) / float(sample_rate) if sample_rate else 0.0
    return duration, peak, rms


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


from hugpy_media.tts.paths import output_dir as _output_dir


def _resolve_device(requested: str | None) -> str:
    """Requested device, or auto: cuda when torch reports one, else cpu. Torch
    is imported inside a guard — a box without it is a cpu box, not a crash."""
    if requested:
        return requested
    try:
        import torch  # noqa: PLC0415 — deliberately lazy
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _seed_backend(seed: int | None) -> None:
    """Best-effort determinism. Seeding is advisory: a backend that ignores it
    still records the seed in the sidecar, so a reader can tell what was asked
    for even when the result is not bit-reproducible."""
    if seed is None:
        return
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(int(seed))
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:  # noqa: BLE001
        pass
    try:
        import random
        random.seed(int(seed))
    except Exception:  # noqa: BLE001
        pass


def _load_backend(spec: TtsSpec, device: str):
    """Import chatterbox LAZILY and build the model for ``spec``.

    Raises ``TtsBackendUnavailable`` on ANY import failure — that is the whole
    "backend unavailable" contract, and it is raised for a missing torch just as
    much as for a missing chatterbox, because the caller's question is "can this
    box speak", not "which wheel is absent"."""
    module_name = ("chatterbox.mtl_tts" if spec.language
                   else "chatterbox.tts")
    class_name = ("ChatterboxMultilingualTTS" if spec.language
                  else "ChatterboxTTS")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TtsBackendUnavailable(
            f"{module_name} is not importable ({exc}). "
            f"{probe()['reason'] or 'pip install ' + BACKEND_PIP}") from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise TtsBackendUnavailable(
            f"{module_name} has no {class_name} — the installed "
            f"{BACKEND_PACKAGE!r} does not expose the expected interface")

    # LOCAL WEIGHTS FIRST when the spec names a checkpoint dir. ``from_pretrained``
    # downloads the backend's OWN default repo, which is not the registry row the
    # catalog bound this capability to — on a fleet whose shared store already
    # holds those weights, that would be both a needless multi-GB fetch and a
    # provenance lie (the sidecar names ``model_id``). ``from_local`` is
    # chatterbox's own loader for exactly this; a build that lacks it falls
    # through to ``from_pretrained`` and the caller learns which was used from
    # the sidecar's ``weights_source``.
    if spec.weights_dir:
        local = getattr(cls, "from_local", None)
        if callable(local):
            try:
                return local(spec.weights_dir, device)
            except ImportError as exc:
                raise TtsBackendUnavailable(
                    f"{class_name}.from_local failed to import its own "
                    f"dependencies ({exc})") from exc

    kwargs: dict[str, Any] = {"device": device}
    if spec.language:
        kwargs["t3_model"] = MTL_T3_MODEL
    try:
        return cls.from_pretrained(**kwargs)
    except TypeError:
        # Older/newer signatures: fall back to positional device only.
        return cls.from_pretrained(device)
    except ImportError as exc:      # a lazy sub-import inside the backend
        raise TtsBackendUnavailable(
            f"{class_name}.from_pretrained failed to import its own "
            f"dependencies ({exc})") from exc


def _generate(model: Any, spec: TtsSpec) -> Any:
    """Call the backend's ``generate`` with only the kwargs it accepts.

    The multilingual class takes ``language_id``; the single-language one does
    not; both take ``audio_prompt_path``. Unsupported kwargs are dropped rather
    than guessed at, and a TypeError falls back to the minimal call so a version
    bump cannot break synthesis outright."""
    kwargs: dict[str, Any] = {}
    if spec.reference_audio:
        kwargs["audio_prompt_path"] = spec.reference_audio
    if spec.language:
        kwargs["language_id"] = spec.language
    try:
        return model.generate(spec.text, **kwargs)
    except TypeError:
        return model.generate(spec.text)


def synthesize(spec: TtsSpec, out_dir: str | None = None) -> dict[str, Any]:
    """Synthesize ``spec`` into a wav + sidecar JSON; return the manifest dict.

    ORDER MATTERS and is part of the contract:

      1. AUTHORITY first — a reference voice without ``authorized=True`` is
         refused BEFORE the backend is imported, so an unauthorized request
         never loads a voice-cloning model, never touches a GPU and never
         produces a file to leak.
      2. Backend availability second — ``TtsBackendUnavailable`` when the
         chatterbox package is absent.
      3. Only then: seed, load, generate, write.

    Returns ``{"audio_path", "sidecar_path", **sidecar}``. The sidecar records
    ``model_id``, ``sample_rate``, ``duration_s``, ``reference_used`` and
    ``seed`` — the five facts a downstream evaluator (speech_scorecard) and a
    lineage reader both need, plus the language/style/device context.
    """
    if spec.reference_audio and not spec.authorized:
        raise ReferenceVoiceUnauthorized(_UNAUTHORIZED_MESSAGE)
    if not str(spec.text or "").strip():
        raise TtsSpecError("tts spec needs non-blank 'text'")

    probed = probe()
    if not probed["importable"]:
        raise TtsBackendUnavailable(probed["reason"])

    device = _resolve_device(spec.device)
    _seed_backend(spec.seed)
    model = _load_backend(spec, device)
    wav = _generate(model, spec)

    sample_rate = int(getattr(model, "sr", 0) or FALLBACK_SAMPLE_RATE)
    samples = _flatten(wav)
    if not samples:
        raise TtsSpecError(
            "backend returned an empty waveform (nothing was synthesized)")

    target_dir = out_dir or _output_dir()
    os.makedirs(target_dir, exist_ok=True)
    stem = uuid4().hex
    audio_path = os.path.join(target_dir, stem + ".wav")
    sidecar_path = os.path.join(target_dir, stem + ".json")

    duration_s, peak_amplitude, rms = _write_wav(audio_path, samples, sample_rate)
    sidecar = {
        "model_id": spec.model_id,
        "sample_rate": sample_rate,
        "duration_s": round(duration_s, 6),
        "reference_used": bool(spec.reference_audio),
        "seed": spec.seed,
        # Context beyond the five required facts — cheap to record, impossible
        # to reconstruct later.
        "backend": BACKEND_PACKAGE,
        "runner_key": list(RUNNER_KEY),
        "language": spec.language,
        "voice_style": spec.voice_style,
        "device": device,
        # WHICH bytes spoke: the registry row's own checkpoint dir, or the
        # backend's default download. Never inferred later from a path guess.
        "weights_dir": spec.weights_dir,
        "weights_source": "local" if spec.weights_dir else "backend-default",
        "authorized": bool(spec.authorized),
        "reference_audio": spec.reference_audio,
        "text": spec.text,
        "n_samples": len(samples),
        # LOUDNESS, MEASURED off the written PCM. Recorded at the source so a
        # downstream guard corroborates a number instead of guessing, and so a
        # silent line is legible in the sidecar without reopening the wav.
        # ``source_peak`` is the backend's own float peak: > 1.0 means the
        # waveform overshot unity and was clipped (see _to_pcm16).
        "peak_amplitude": peak_amplitude,
        "rms_dbfs": (round(20 * math.log10(rms / 32768.0), 2) if rms > 0
                     else None),
        "source_peak": round(max(abs(s) for s in samples), 6),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pitch_analysis": False,
    }
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, ensure_ascii=False)

    return {"audio_path": audio_path, "sidecar_path": sidecar_path, **sidecar}


# --------------------------------------------------------------------------- #
# Bus entrypoint: ``hugpy_video.intel.runners.tts_chatterbox.run_tts_chatterbox``
# wraps ``make_tts`` + ``synthesize`` into the media bus's JobResult/JobError and
# ingests the wav — that glue is video's; this module stops at the wav.
# --------------------------------------------------------------------------- #


__all__ = [
    "BACKEND_PACKAGE", "BACKEND_PIP", "JOB_NAME", "JOB_QUEUE", "JOB_TIMEOUT_S",
    "MODEL_ID", "PITCH_ANALYSIS_FORBIDDEN", "RUNNER_KEY", "TASK",
    "ReferenceVoiceUnauthorized", "TtsBackendUnavailable", "TtsError",
    "TtsSpec", "TtsSpecError", "make_tts", "probe", "synthesize",
]
