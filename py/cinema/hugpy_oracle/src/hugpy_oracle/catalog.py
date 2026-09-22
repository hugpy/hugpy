"""Unified READ-ONLY capability catalog (k90a): the bridge over both registries.

Two disjoint model registries exist today (documented in studio/tester.py):

  * STUDIO — video_intel/studio: typed Capability/ModelConfig zoo with its own
    router, runner gates (``runner_gate_reason``) and viability facts
    (``ZERO_BYTE_MODELS`` / ``STUB_RUNNER_MODULES`` in presets).
  * TASKS  — imports/config/models: the legacy tasks-string registry
    (text/vision/ASR/imagegen rows carrying ``tasks: [str]``), dispatched
    through ``execute_prompt`` and gated at request time by worker heartbeat
    ``task_capabilities`` + central's own import probes (managers/task_deps).

This module COMPOSES the two behind one namespaced capability vocabulary and
joins the health signals each side already computes, so GET /oracle/capabilities
can explain WHY a capability is ineligible BEFORE anything executes (the
Phase-1 done-criterion). It never mutates either registry — pure reads.

The mapping tables are the contract k90b's router will dispatch on:
``LEGACY_TASK_CAPABILITY`` (task string -> namespaced name) and
``STUDIO_CAPABILITY_NAME`` (studio Capability -> namespaced name), with
explicit EXCLUDED tables for members that deliberately do not become routable
capabilities. ``tests/test_oracle_catalog.py`` proves both maps total.

k98 adds a THIRD family that belongs to neither table: the SPEECH capabilities
(``audio.tts``, ``audio.transcribe.word_timestamps``, ``audio.speaker_similarity``
— doc §4 names). They are declared in ``SPEECH_CAPABILITY_TASK`` /
``_SPEECH_IO`` and built by ``_speech_views()`` rather than through
``LEGACY_TASK_CAPABILITY``, for two reasons that are worth stating because the
shortcut is tempting:

  * the generic legacy path judges worker health with ``_task_capable``, which
    is LEGACY-PERMISSIVE (a task a worker does not enumerate reads as capable).
    For a brand-new capability that is backwards: "unknown" must mean "not
    yet". ``_worker_seats_task`` is the strict/affirmative twin, modelled on
    ``workers._comfy_id_lock_capable``, and ``audio.tts`` is judged on it;
  * ``router.CAPABILITY_TASK`` is proven set-EQUAL to
    ``LEGACY_TASK_CAPABILITY.values()`` by ``tests/test_oracle_route.py``.
    Adding rows to that table from here would break a test in another agent's
    file. The speech rows therefore ship as a SEPARATE, published mapping the
    router folds in when it is next opened (see the k98 dispatch record).

k101 turns each row into the doc §3.2 DESCRIPTOR: a semver, declared param /
result schemas, declared limits, host access, authority, license, evaluation
suite and model fingerprint ride on the same ``CapabilityView`` the route
already serializes, plus a REGISTRATION PROBE (``probes.py``) run at catalog
build. Three rules keep it honest and are worth stating because the temptation
runs the other way every time:

  * a field nobody can answer stays empty. Most ``limits`` are ``{}`` and most
    ``param_schema``s are ``{}`` because neither registry records that fact
    today; the two that ARE knowable (chatterbox's spec fields, whisper's
    ``word_timestamps`` flag, the studio's ``max_duration_s``/resolutions/
    license/weight_hash) are read from the declaration that already exists,
    never retyped and never guessed;
  * a FAILING probe makes the capability ineligible with the probe's own words
    as the reason (``CapabilityView.with_probe``) — doc §3.2's "ineligible until
    its descriptor and probe agree";
  * ``registry_version()`` digests the ROUTING SNAPSHOT (legacy row projections
    + the studio zoo's pinned facts + every descriptor version). It rides on
    every view and belongs in every ``ExecutionReceipt``: "this ran on
    whisper-x" is only reproducible next to "…out of THIS registry".

Import discipline (same as studio/tester.py): module top level is
dependency-light — contracts + probes + the studio ENUMS only (all plain
stdlib). Every registry/worker read is LAZY inside a provider function, and
those providers are module-level seams (``_legacy_registry_rows`` /
``_online_workers`` / …) precisely so tests monkeypatch them and need no live
workers, GPU or network. Worker/blocklist reads are guarded fail-open: a
telemetry read must never turn the catalog itself into the failure. (``probes``
imports ``catalog`` lazily, INSIDE its checks, so the pair does not cycle;
``authority`` and ``evaluation`` are read lazily for the same reason —
``evaluation`` imports ``router``, which imports this module.)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Any, Callable, Mapping

from hugpy_video.intel.studio.enums import Capability

from hugpy_oracle import probes
from hugpy_oracle.contracts import (
    DEFAULT_CAPABILITY_VERSION,
    AccessKind,
    ArtifactKind,
    AuthorityKind,
    CapabilityView,
    Eligibility,
    Provenance,
    ResourceHints,
    SourceRegistry,
    canonical_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The mapping tables — defined ONCE, exported, proven total by tests.
# ---------------------------------------------------------------------------

# Legacy tasks-string registry: task string -> namespaced capability name.
# Covers every ML_TASKS dispatch task plus the registry task strings the
# models_default.py groupings key on (chat/embed aliases). A discovery row
# carrying a task string absent from BOTH tables is simply not catalogued —
# unknown discovery noise must not invent capabilities (see unmapped_tasks()).
LEGACY_TASK_CAPABILITY: dict[str, str] = {
    "text-generation":               "text.chat",
    "text-generation-inference":     "text.chat",
    "text2text-generation":          "text.chat",
    "text-summarization":            "text.summarize",
    "keyword-extraction":            "text.keywords",
    "feature-extraction":            "text.embed",
    "sentence-transformers":         "text.embed",
    "sentence-similarity":           "text.similarity",
    "automatic-speech-recognition":  "audio.transcribe",
    "speech-recognition":            "audio.transcribe",
    "image-text-to-text":            "image.understand",
    "text-to-image":                 "image.generate",
    "image-to-image":                "image.transform",
    "depth-estimation":              "image.depth",
    "object-detection":              "image.detect",
    "image-classification":          "image.classify",
    "image-segmentation":            "image.segment",
    "document-extraction":           "doc.extract",
    "url-extraction":                "web.fetch",
}

# Legacy tasks deliberately NOT catalogued as capabilities, with the reason.
# Every ML_TASKS member maps above; these are the OTHER task strings observed
# on live registry rows (discovery/classifier output). An exclusion is a
# recorded decision, never a silent gap — the completeness test enforces
# membership in exactly one of the two tables, and ``unmapped_tasks()`` reports
# any live task string that reaches neither.
LEGACY_TASK_EXCLUDED: dict[str, str] = {
    "text-to-video": (
        "video generation is owned by the STUDIO registry (video.generate.*); "
        "legacy diffusers video rows are reached through the studio/movie "
        "pipeline, not routed as a second t2v capability"),
    "image-to-video": (
        "video generation is owned by the STUDIO registry (video.generate.*); "
        "same ownership rule as text-to-video"),
    "adapter": (
        "classifier marker for LoRA/adapter weight dirs (model_classifier."
        "ADAPTER_TASK) — an adapter conditions another model, it is not an "
        "executable capability"),
    "needs-classification": (
        "classifier marker for un-classified discovery rows (model_classifier."
        "NEEDS_CLASSIFICATION_TASK) — unknown weights are not a capability"),
    "pipeline-component": (
        "classifier marker for partial pipeline dirs (VAE/text-encoder/…) — a "
        "component of a model, not an executable capability"),
}

# Deterministic ingest amenities (ml_routes._DETERMINISTIC_ML): they run as
# thin LOCAL handlers on central — no model row, no worker dispatch. A
# capability whose tasks are all deterministic is judged ONLY on central's
# dependency probe; "no model registered" would be a false refusal.
DETERMINISTIC_TASKS: frozenset[str] = frozenset(
    {"document-extraction", "url-extraction"})

# Studio registry: Capability enum member -> namespaced capability name.
STUDIO_CAPABILITY_NAME: dict[Capability, str] = {
    Capability.T2V:      "video.generate.t2v",
    Capability.I2V:      "video.generate.i2v",
    Capability.V2V:      "video.generate.v2v",
    Capability.KEYFRAME: "video.generate.keyframe",
    Capability.ID_LOCK:  "video.generate.id_lock",
    Capability.MOTION:   "video.generate.motion",
    Capability.STREAM:   "video.generate.stream",
    Capability.INPAINT:  "video.generate.inpaint",
    Capability.OUTPAINT: "video.generate.outpaint",
    Capability.RETAKE:   "video.generate.retake",
    Capability.AUDIO:    "video.generate.audio",
    Capability.LIPSYNC:  "video.generate.lipsync",
    Capability.UPRES:    "video.enhance.upres",
    Capability.INTERP:   "video.enhance.interp",
    Capability.RESTORE:  "video.enhance.restore",
}

# Studio Capability members that are NOT routable capabilities, with the why.
STUDIO_CAPABILITY_EXCLUDED: dict[Capability, str] = {
    Capability.ASSEMBLE: (
        "orchestration stage, not a model-served capability (studio "
        "PLANNED_CAPABILITIES): multi-shot assembly is a composition node the "
        "oracle plans, never a route it resolves to one model"),
}


# ---------------------------------------------------------------------------
# SPEECH capabilities (k98) — the voice vertical's first slice.
# ---------------------------------------------------------------------------

# Capability -> the legacy dispatch task string it executes on. Published for
# the router to fold into CAPABILITY_TASK (one line; see the k98 dispatch
# record) — it is NOT merged here, because that table is proven set-equal to
# LEGACY_TASK_CAPABILITY.values() by a test in another agent's file.
# ``audio.speaker_similarity`` is absent ON PURPOSE: there is no dispatch task
# for speaker embeddings on this fleet and inventing one would be the exact
# phantom this module exists to delete.
SPEECH_CAPABILITY_TASK: dict[str, str] = {
    "audio.tts":                        "text-to-speech",
    "audio.transcribe.word_timestamps": "automatic-speech-recognition",
}

# Fixed parameters a capability implies for its dispatch. The word-timestamp
# capability IS the base transcription capability plus one flag — the whole
# reason it is a separate name rather than an argument the planner has to
# remember (doc §4: "It must not request a checkpoint path and loosely
# structured arguments"). ``capability_params()`` is the read accessor; the
# router merges these into the dispatch kwargs (see the k98 dispatch record for
# the passthrough chain this flag still needs).
CAPABILITY_FIXED_PARAMS: dict[str, dict[str, Any]] = {
    "audio.transcribe.word_timestamps": {"word_timestamps": True},
}

# Doc §4 spells the TTS capability ``voice.synthesize.reference_conditioned``;
# the fleet's dispatch vocabulary spells it ``audio.tts``. Rather than publish
# two views of one thing (which would let them drift into disagreeing about
# eligibility), the doc name is an ALIAS resolved by ``get_capability`` /
# ``resolve_owners``. ``list_capabilities`` lists canonical names only.
# k97's authority table already knows BOTH names, and correctly treats them
# differently: the doc name is always voice-conditioned, while plain
# ``audio.tts`` needs a voice authorization only when a reference is supplied.
CAPABILITY_ALIASES: dict[str, str] = {
    "voice.synthesize.reference_conditioned": "audio.tts",
}

# The task string the TTS row declares (or should declare — see TTS_MODEL_MARKERS).
TTS_TASK: str = "text-to-speech"

#: MEASURED peak VRAM for one chatterbox synthesis, in GiB — not an estimate.
#: Observed 2026-08-21 on a-brain-Super-Server (RTX 3090, torch 2.6.0+cu124,
#: chatterbox-tts 0.1.7, fp32 English checkpoint loaded from the registry row's
#: own weights): ``torch.cuda.max_memory_reserved()`` = 3.22 GiB inside the
#: synthesis child, and a device-level delta of 3.54 GiB across the whole call
#: (nvidia-smi 10650 MiB -> 14276 MiB), the difference being the child's own
#: CUDA context. The DEVICE delta is published because that is the number a
#: placement decision actually spends. k98 expected "~6 GiB free" from the
#: checkpoint sizes; the measurement is what the catalog publishes now
#: (Provenance.MEASURED), and it is lower than the guess — which is exactly why
#: the guess was never published.
TTS_MEASURED_VRAM_GIB: float = 3.5

# The runner adapter + registration probe the catalog consults for audio.tts.
# Overridden by managers.resolvers.model_resolver.EXTERNAL_TASK_RUNNERS when
# that declaration is readable; this constant is the fail-open default so a
# resolver import problem degrades the REASON, never the catalog.
TTS_RUNNER_MODULE: str = "hugpy_media.tts.chatterbox_runner"

# Row markers that identify the chatterbox weights when the row's own ``tasks``
# do not. This is a NAMED RESCUE of one known-misclassified row, not a fuzzy
# search: model_discovery recorded ``Viral2AI~chatterbox`` as text-generation
# (its HF card carries ``library_name: chatterbox`` and ``pipeline_tag:
# text-to-speech``, but it has no transformers config the classifier could read
# — the same repo shape that produced the 2026-07-03 import failure documented
# in model_resolver.validate_registry). The weights are REAL (≈13.9 GB on the
# shared store), so refusing to bind them would be its own dishonesty. The
# mismatch is reported as an advisory reason on the view.
TTS_MODEL_MARKERS: tuple[str, ...] = ("chatterbox",)

# License markers that REFUSE a voice model outright. A reference-conditioned
# voice model is a rights instrument, so a declared non-commercial/no-derivative
# license is a hard gate. UNKNOWN is NOT a refusal (the legacy registry rows
# carry no license field at all — refusing on absence would refuse everything)
# — it becomes an advisory reason instead. Chatterbox is MIT.
TTS_REFUSED_LICENSE_MARKERS: tuple[str, ...] = (
    "-nc", "noncommercial", "non-commercial", "-nd", "noderiv", "research-only")

# Row markers for a speaker-embedding model. Searched, found nothing on this
# fleet (2026-08-20), and that is precisely what ``audio.speaker_similarity``
# reports — the capability is DECLARED with no binding rather than bound to an
# invented model.
SPEAKER_EMBEDDING_MARKERS: tuple[str, ...] = (
    "speechbrain", "ecapa", "wavlm", "resemblyzer", "titanet", "pyannote",
    "x-vector", "xvector", "speaker-embedding", "speaker_embedding")

# The exact edits ``audio.transcribe.word_timestamps`` is waiting on. Stated as
# a constant so the refusal text and the dispatch record cannot drift, and so a
# reader of GET /oracle/capabilities learns what to change rather than only that
# something is missing. Verified 2026-08-20 by reading the chain end to end.
WORD_TIMESTAMPS_PASSTHROUGH_GAP: str = (
    "word-level timing is not wired through the whisper dispatch yet: "
    "normalize_ml_kwargs passes word_timestamps through, but "
    "builders._build_whisper_request forwards only an allow-list of keys, "
    "TranscribeRequest has no word_timestamps field, and "
    "whisper_model/src/model/execute.whisper_transcribe never sets "
    "options['word_timestamps'] — so whisper returns segments with an EMPTY "
    "words list. Four one-line edits (whisper_schemas.TranscribeRequest, "
    "builders._build_whisper_request key tuple, whisper runner forward, "
    "execute.whisper_transcribe option) close it; until then this capability "
    "refuses rather than promise timing it cannot produce")


# What each capability accepts/produces (artifact kinds). Declared here because
# neither registry states IO kinds today — the legacy registry has only task
# strings, and the studio's contract is capability-shaped, not artifact-shaped.
_K = ArtifactKind
_LEGACY_IO: dict[str, tuple[tuple[ArtifactKind, ...], tuple[ArtifactKind, ...]]] = {
    "text.chat":        ((_K.TEXT,), (_K.TEXT,)),
    "text.summarize":   ((_K.TEXT,), (_K.TEXT,)),
    "text.keywords":    ((_K.TEXT,), (_K.JSON,)),
    "text.embed":       ((_K.TEXT,), (_K.EMBEDDING,)),
    "text.similarity":  ((_K.TEXT,), (_K.JSON,)),
    "audio.transcribe": ((_K.AUDIO, _K.VIDEO), (_K.TEXT, _K.JSON)),
    "image.understand": ((_K.IMAGE, _K.TEXT), (_K.TEXT,)),
    "image.generate":   ((_K.TEXT,), (_K.IMAGE,)),
    "image.transform":  ((_K.IMAGE, _K.TEXT), (_K.IMAGE,)),
    "image.depth":      ((_K.IMAGE,), (_K.IMAGE,)),
    "image.detect":     ((_K.IMAGE,), (_K.JSON,)),
    "image.classify":   ((_K.IMAGE,), (_K.JSON,)),
    "image.segment":    ((_K.IMAGE,), (_K.IMAGE, _K.JSON)),
    "doc.extract":      ((_K.DOCUMENT,), (_K.TEXT,)),
    "web.fetch":        ((_K.URL,), (_K.TEXT,)),
}
_STUDIO_IO: dict[Capability, tuple[tuple[ArtifactKind, ...], tuple[ArtifactKind, ...]]] = {
    Capability.T2V:      ((_K.TEXT,), (_K.VIDEO,)),
    Capability.I2V:      ((_K.IMAGE, _K.TEXT), (_K.VIDEO,)),
    Capability.V2V:      ((_K.VIDEO, _K.TEXT), (_K.VIDEO,)),
    Capability.KEYFRAME: ((_K.IMAGE,), (_K.VIDEO,)),
    Capability.ID_LOCK:  ((_K.TEXT, _K.IMAGE), (_K.VIDEO,)),
    Capability.MOTION:   ((_K.TEXT, _K.IMAGE), (_K.VIDEO,)),
    Capability.STREAM:   ((_K.TEXT, _K.IMAGE), (_K.VIDEO,)),
    Capability.INPAINT:  ((_K.VIDEO, _K.TEXT), (_K.VIDEO,)),
    Capability.OUTPAINT: ((_K.VIDEO, _K.TEXT), (_K.VIDEO,)),
    Capability.RETAKE:   ((_K.VIDEO, _K.TEXT), (_K.VIDEO,)),
    Capability.AUDIO:    ((_K.TEXT, _K.IMAGE), (_K.VIDEO, _K.AUDIO)),
    Capability.LIPSYNC:  ((_K.VIDEO, _K.AUDIO), (_K.VIDEO,)),
    Capability.UPRES:    ((_K.VIDEO,), (_K.VIDEO,)),
    Capability.INTERP:   ((_K.VIDEO,), (_K.VIDEO,)),
    Capability.RESTORE:  ((_K.VIDEO,), (_K.VIDEO,)),
}
# Speech side (k98). ``audio.tts`` accepts AUDIO as well as TEXT because the
# reference voice IS an audio input; it produces AUDIO plus the JSON sidecar the
# runner writes (model_id/sample_rate/duration_s/reference_used/seed).
# ``audio.speaker_similarity`` takes two audio refs and produces a score.
_SPEECH_IO: dict[str, tuple[tuple[ArtifactKind, ...], tuple[ArtifactKind, ...]]] = {
    "audio.tts":                        ((_K.TEXT, _K.AUDIO), (_K.AUDIO, _K.JSON)),
    "audio.transcribe.word_timestamps": ((_K.AUDIO, _K.VIDEO), (_K.TEXT, _K.JSON)),
    "audio.speaker_similarity":         ((_K.AUDIO,), (_K.JSON,)),
}


# ---------------------------------------------------------------------------
# Descriptor declarations (k101) — doc §3.2. EMPTY WHERE UNKNOWN.
# ---------------------------------------------------------------------------

# Per-capability semver of the DESCRIPTOR (not of the model, not of the runner).
# It changes when the CONTRACT changes: a new parameter, a narrowed limit, a
# different produced artifact. Everything not listed is
# DEFAULT_CAPABILITY_VERSION — "0.1.0", the honest starting point for a
# descriptor that has never been revised. Bump a row here and every receipt's
# registry_version changes with it, which is the point.
CAPABILITY_VERSION: dict[str, str] = {}

# JSON-Schema-ish parameter declarations. Only where the fleet ACTUALLY declares
# the surface somewhere a reader can go check:
#   * audio.tts            -> runners/tts_chatterbox.TtsSpec (the same names the
#                             registration probe compares against);
#   * word_timestamps      -> CAPABILITY_FIXED_PARAMS + whisper_schemas.
# Every other capability declares ``{}``: neither registry records parameter
# schemas today, and a plausible-looking invented schema is worse than an
# absent one — the planner would trust it.
CAPABILITY_PARAM_SCHEMA: dict[str, dict[str, Any]] = {
    "audio.tts": {
        "type": "object",
        "required": ["text"],
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "minLength": 1,
                     "description": "the line to speak"},
            "reference_audio": {"type": ["string", "null"],
                                "description": "path to an AUTHORIZED reference "
                                               "voice; requires authorized=true"},
            "authorized": {"type": "boolean", "default": False,
                           "description": "explicit authorization for "
                                          "reference_audio (k97 AuthorityKind."
                                          "VOICE is the typed gate; this is the "
                                          "runner's last line of defence)"},
            "voice_style": {"type": ["string", "null"]},
            "seed": {"type": ["integer", "null"], "minimum": 0},
            "language": {"type": ["string", "null"],
                         "description": "BCP-47-ish; null = English"},
            "device": {"type": ["string", "null"], "enum": ["cuda", "cpu", None]},
            "model_id": {"type": "string"},
        },
    },
    "audio.transcribe.word_timestamps": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "word_timestamps": {
                "type": "boolean", "const": True,
                "description": "fixed by the capability (CAPABILITY_FIXED_"
                               "PARAMS); the whole reason this is a separate "
                               "capability name rather than a flag the planner "
                               "has to remember"},
        },
    },
}

# What a capability RETURNS. Same rule: declared only where the shape is pinned
# by a schema module or a runner's documented sidecar.
CAPABILITY_RESULT_SCHEMA: dict[str, dict[str, Any]] = {
    "audio.tts": {
        "type": "object",
        "required": ["audio_path", "sidecar_path", "model_id", "sample_rate",
                     "duration_s", "reference_used", "seed"],
        "properties": {
            "audio_path": {"type": "string"},
            "sidecar_path": {"type": "string"},
            "model_id": {"type": "string"},
            "sample_rate": {"type": "integer"},
            "duration_s": {"type": "number"},
            "reference_used": {"type": "boolean"},
            "seed": {"type": ["integer", "null"]},
        },
    },
    "audio.transcribe": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "segments": {"type": "array", "items": {"type": "object"}},
        },
    },
    "audio.transcribe.word_timestamps": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "words": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"word": {"type": "string"},
                                       "start": {"type": "number"},
                                       "end": {"type": "number"},
                                       "probability": {"type": "number"}}}}}},
            },
        },
    },
}

# Declared limits (formats / languages / durations / resolutions / context).
# The STUDIO side is computed per-view from the zoo's own ModelConfig
# (max_duration_s, max_frames, resolutions) — real numbers, so they are not
# retyped here. This table carries only what a constant in the tree states.
CAPABILITY_LIMITS: dict[str, dict[str, Any]] = {
    "audio.tts": {"formats": ["wav"]},
}

# Host access a capability's implementation TOUCHES (doc §3.2). DELIBERATELY
# TINY: "writes its own artifact into the media store" is what every runner
# does and declaring it everywhere would make the field meaningless. What is
# declared here is access BEYOND the capability's own workspace — reaching the
# public internet, or reading a path the OPERATOR named. See authority.py for
# why declaring is not yet the same as enforcing.
CAPABILITY_ACCESS_DECL: dict[str, tuple[AccessKind, ...]] = {
    "web.fetch": (AccessKind.NETWORK, AccessKind.EXTERNAL),
    "doc.extract": (AccessKind.FILESYSTEM,),
}

# The canonical evaluation suite per capability (doc §3.2). Derived, not
# retyped: the judge rubrics live in oracle/evaluation.py and the speech checks
# in oracle/speech.py, so this maps to what actually exists — see
# ``capability_eval_suite``.
SPEECH_EVAL_SUITE: str = "oracle.speech:speech_scorecard"
SPEECH_EVALUATED: frozenset[str] = frozenset({
    "audio.tts", "audio.transcribe.word_timestamps", "audio.speaker_similarity"})

# Version of the registry-snapshot PAYLOAD shape. Bumping it changes every
# registry_version by construction — which is correct: a snapshot that means
# something different is a different snapshot.
REGISTRY_SNAPSHOT_SCHEMA: int = 1


def capability_version(name: str) -> str:
    """The declared descriptor version for ``name`` (aliases resolve)."""
    return CAPABILITY_VERSION.get(canonical_name(name), DEFAULT_CAPABILITY_VERSION)


def capability_authority(name: str) -> tuple[AuthorityKind, ...]:
    """The authority this capability needs BY CONSTRUCTION (k97's tables, read
    rather than re-declared — one source of truth, two readers).

    ``audio.tts`` is deliberately absent: k97 classes it ``VOICE_ON_REFERENCE``
    — it needs a voice authorization only when the REQUEST carries a reference
    voice, and a descriptor field cannot express "only when". That conditional
    stays in ``authority.required_authorities`` rule 2, where it belongs; a
    licensed synthetic voice is not a rights question and must not be gated as
    if it were."""
    from hugpy_oracle import authority
    target = canonical_name(name)
    kinds: list[AuthorityKind] = []
    if target in authority.IDENTITY_CONDITIONED:
        kinds.append(AuthorityKind.LIKENESS)
    if target in authority.VOICE_CONDITIONED:
        kinds.append(AuthorityKind.VOICE)
    return tuple(kinds)


def capability_eval_suite(name: str) -> str | None:
    """The canonical evaluation suite for ``name``, or None when nothing
    evaluates it yet. Read from the evaluator modules (lazy import: evaluation
    imports router, which imports this module)."""
    target = canonical_name(name)
    if target in SPEECH_EVALUATED:
        return SPEECH_EVAL_SUITE
    try:
        from hugpy_oracle.evaluation import RUBRICS
    except Exception as exc:  # noqa: BLE001 — a listing must not need the judge
        logger.debug("oracle catalog: rubrics unreadable (%s)", exc)
        return None
    rubric = RUBRICS.get(target)
    return f"oracle.evaluation:{rubric.name}" if rubric is not None else None


def _descriptor_fields(name: str, *, license: str | None = None,
                       limits: Mapping[str, Any] | None = None,
                       model_fingerprint: str | None = None,
                       adapter_version: str | None = None) -> dict[str, Any]:
    """The doc §3.2 fields every view carries, assembled from the declaration
    tables + whatever the caller MEASURED for this particular row set."""
    declared = dict(CAPABILITY_LIMITS.get(name) or {})
    declared.update({k: v for k, v in dict(limits or {}).items() if v is not None})
    return {
        "version": capability_version(name),
        "param_schema": CAPABILITY_PARAM_SCHEMA.get(name) or {},
        "result_schema": CAPABILITY_RESULT_SCHEMA.get(name) or {},
        "limits": declared,
        "authority_required": capability_authority(name),
        "access": CAPABILITY_ACCESS_DECL.get(name, ()),
        "license": license,
        "eval_suite": capability_eval_suite(name),
        "adapter_version": adapter_version,
        "model_fingerprint": model_fingerprint,
    }


# ---------------------------------------------------------------------------
# Provider seams — lazy reads, monkeypatchable in tests.
# ---------------------------------------------------------------------------


def _legacy_registry_rows() -> dict[str, dict[str, Any]]:
    """The legacy registry as plain row dicts (model_key -> row). Served from
    models_config's cached build — a read, never a rebuild."""
    from hugpy_engine.config.models.models_config import get_model_registry
    return dict(get_model_registry(dict_return=True))


def _online_workers() -> list[dict[str, Any]] | None:
    """Online workers per the heartbeat registry, or None when the worker
    plane is unreadable (signal UNKNOWN — the catalog then judges on central's
    own capabilities rather than inventing a fleet)."""
    try:
        from hugpy_engine.placement import get_worker_registry
        return [dict(w) for w in get_worker_registry().list_workers(online_only=True)]
    except Exception as exc:  # noqa: BLE001 — telemetry must not break the catalog
        logger.debug("oracle catalog: worker registry unreadable (%s)", exc)
        return None


def _worker_task_capable(worker: dict[str, Any], task: str) -> bool:
    """The fleet's own capability-honesty rule (affirmative-deny only,
    legacy-permissive), reached through the ``TaskCapabilityGate`` provider so
    the fleet's carve-outs stay where they live."""
    try:
        from hugpy_oracle.providers import get_task_capability_gate
        return get_task_capability_gate().task_capable(worker, task)
    except Exception:  # noqa: BLE001
        return True


def _central_task_available(task: str) -> bool | None:
    """Can CENTRAL run ``task`` in-process? True/False from the canonical
    task->dependency probe (managers/task_deps, find_spec only); None when the
    task has no dependency entry there (no gate — e.g. chat, which central
    serves through its own engine path)."""
    from hugpy_engine.task_deps import TASK_DEPS, have
    dep = TASK_DEPS.get(task)
    if dep is None:
        return None
    return have(dep[0])


def _blocked_model_keys() -> set[str]:
    """Operator-blocked model keys, fail-open like every blocklist read."""
    try:
        from hugpy_engine.placement import get_blocklist
        return set(get_blocklist().blocked_keys())
    except Exception:  # noqa: BLE001
        return set()


# --- speech seams (k98) ----------------------------------------------------


def _worker_seats_task(worker: dict[str, Any], task: str) -> bool:
    """Does ``worker`` AFFIRMATIVELY advertise ``task``?

    STRICT / affirmative-only, DELIBERATELY UNLIKE ``_worker_task_capable``
    (which wraps ``workers._task_capable``: legacy-permissive, so a task a
    worker does not enumerate reads as capable). The permissive default exists
    to protect a pre-feature fleet's EXISTING routes; it is exactly wrong for a
    brand-new capability, where "unknown" must mean "not yet" — the same
    reasoning ``workers._comfy_id_lock_capable`` already applies to id_lock.

    A worker qualifies only when its heartbeat ``task_capabilities`` carries
    ``task`` set truthy. No dict, no key, or a falsy value: not seated."""
    caps = worker.get("task_capabilities")
    if not isinstance(caps, dict):
        return False
    return bool(caps.get(task))


def _tts_runner_module_name() -> str:
    """Where the TTS runner adapter lives — read from the resolver's
    ``EXTERNAL_TASK_RUNNERS`` declaration (the model_resolver hook k98 added so
    a row whose task has no in-process runner can still say WHERE its runner
    is), falling back to this module's constant. Fail-open: a resolver import
    problem must degrade the reason text, never the catalog."""
    try:
        from hugpy_engine.resolvers.model_resolver import external_runner_for
        declared = external_runner_for(TTS_TASK)
        if declared:
            return declared[0]
    except Exception as exc:  # noqa: BLE001
        logger.debug("oracle catalog: external runner declaration unreadable (%s)", exc)
    return TTS_RUNNER_MODULE


def _tts_runner_probe() -> dict[str, Any]:
    """The doc §4 step-3 "adapter registration and health probe" for audio.tts.

    Two distinct facts, never conflated:
      ``runner_registered`` — the ADAPTER module imports and exposes ``probe``;
      ``importable``        — its BACKEND (the ``chatterbox`` package) is
                              usable in THIS process.
    Central answers registered=True / importable=False, which is the honest
    state: the adapter exists, the box cannot run it, a worker must."""
    module = _tts_runner_module_name()
    try:
        import importlib
        mod = importlib.import_module(module)
        result = dict(mod.probe())
    except Exception as exc:  # noqa: BLE001 — an unimportable adapter is the finding
        return {
            "importable": False, "runner_registered": False, "module": module,
            "reason": (f"runner adapter {module!r} is not importable "
                       f"({type(exc).__name__}: {exc})")}
    result.setdefault("runner_registered", True)
    result.setdefault("module", module)
    return result


def _word_timestamps_wired() -> bool:
    """Can a ``word_timestamps=True`` request actually reach whisper?

    Probes the ONE structural fact the whole passthrough chain hangs on: does
    ``TranscribeRequest`` carry the field? The builder allow-list, the runner
    forward and the whisper option are all downstream of it, and none of them
    can be added without it. False here (2026-08-20) — see
    ``WORD_TIMESTAMPS_PASSTHROUGH_GAP``. Fail-CLOSED on an unreadable schema:
    an unverifiable passthrough is not a passthrough."""
    try:
        from hugpy_media.schemas.whisper_schemas import TranscribeRequest
        fields = getattr(TranscribeRequest, "model_fields", None)
        if fields is None:                      # pydantic v1 shape
            fields = getattr(TranscribeRequest, "__fields__", {})
        return "word_timestamps" in fields
    except Exception as exc:  # noqa: BLE001
        logger.debug("oracle catalog: whisper schema unreadable (%s)", exc)
        return False


# ---------------------------------------------------------------------------
# View construction — TASKS side
# ---------------------------------------------------------------------------


def _capability_task_groups() -> dict[str, tuple[str, ...]]:
    """Invert LEGACY_TASK_CAPABILITY: capability name -> its task strings,
    sorted for stable output."""
    groups: dict[str, list[str]] = {}
    for task, cap in LEGACY_TASK_CAPABILITY.items():
        groups.setdefault(cap, []).append(task)
    return {cap: tuple(sorted(tasks)) for cap, tasks in groups.items()}


def _legacy_views() -> list[CapabilityView]:
    rows = _legacy_registry_rows()
    blocked = _blocked_model_keys()
    workers = _online_workers()

    views: list[CapabilityView] = []
    for cap_name, tasks in sorted(_capability_task_groups().items()):
        model_ids = tuple(sorted(
            key for key, row in rows.items()
            if any(t in (row.get("tasks") or ()) for t in tasks)))
        usable_ids = tuple(m for m in model_ids if m not in blocked)
        task_list = ", ".join(tasks)
        deterministic = all(t in DETERMINISTIC_TASKS for t in tasks)

        # Central-local signal (canonical task->dependency probe).
        central_flags = {t: _central_task_available(t) for t in tasks}
        central_ok = any(flag is not False for flag in central_flags.values())

        reasons: list[str] = []
        worker_ok: bool | None = None
        if deterministic:
            # No model, no worker dispatch — a thin local handler on central
            # (ml_routes._DETERMINISTIC_ML). Only the dependency probe gates.
            eligible = central_ok
            if not central_ok:
                reasons.append(
                    f"central cannot run deterministic task(s) {task_list} "
                    f"(dependency module not importable — see managers/task_deps)")
        else:
            if not model_ids:
                reasons.append(f"no model registered for task(s) {task_list}")
            elif not usable_ids:
                reasons.append(
                    f"every registered model for task(s) {task_list} is "
                    f"operator-blocked from the serving pool")

            # Worker signal (heartbeat task_capabilities, affirmative-deny only).
            if workers is not None:
                if not workers:
                    worker_ok = False
                    reasons.append("no online worker registered")
                else:
                    worker_ok = any(_worker_task_capable(w, t)
                                    for w in workers for t in tasks)
                    if not worker_ok:
                        reasons.append(
                            f"no online worker advertises task(s) {task_list} "
                            f"(heartbeat task_capabilities)")

            if not central_ok:
                missing = ", ".join(
                    t for t, f in sorted(central_flags.items()) if f is False)
                reasons.append(
                    f"central cannot serve task(s) {missing} in-process "
                    f"(dependency module not importable — see managers/task_deps)")

            eligible = bool(usable_ids) and (worker_ok is True or central_ok)
            if not eligible and not reasons:  # defensive: Eligibility requires reasons
                reasons.append("no execution path (no capable worker and no "
                               "central-local fallback)")

        frameworks = tuple(sorted({
            str(rows[m].get("framework")) for m in model_ids
            if rows.get(m, {}).get("framework")}))
        notes = "deterministic local amenity (no model)" if deterministic else ""
        accepts, produces = _LEGACY_IO[cap_name]
        views.append(CapabilityView(
            name=cap_name,
            source=SourceRegistry.TASKS,
            accepts=accepts,
            produces=produces,
            model_ids=usable_ids,
            eligibility=Eligibility(eligible=eligible, reasons=tuple(reasons)),
            resources=ResourceHints(frameworks=frameworks, notes=notes),
            **_descriptor_fields(
                cap_name,
                license=_declared_license(rows, usable_ids),
                limits=_row_limits(rows, usable_ids)),
        ))
    return views


def _declared_license(rows: dict[str, dict[str, Any]],
                      model_ids: tuple[str, ...]) -> str | None:
    """The license of the bound models, or None when NO row records one.

    Several models with several licenses is a real state on this fleet, so the
    field reports the set rather than picking a winner — a capability whose
    implementations disagree about licensing is exactly what an operator needs
    to see. ``None`` means "not recorded", never "permissive"."""
    declared = sorted({lic for m in model_ids
                       if (lic := _row_license(rows.get(m) or {}))})
    if not declared:
        return None
    return declared[0] if len(declared) == 1 else ", ".join(declared)


def _row_limits(rows: dict[str, dict[str, Any]],
                model_ids: tuple[str, ...]) -> dict[str, Any]:
    """Limits READ off the bound legacy rows. Today exactly one is recorded
    there: ``model_max_length`` -> ``max_context_tokens``, the largest context
    any bound model offers (the planner asks "can this capability hold my
    document"; the answer is the best model's window, and the router picks it).
    A row without the field contributes nothing rather than a zero."""
    windows: list[int] = []
    for model_id in model_ids:
        raw = (rows.get(model_id) or {}).get("model_max_length")
        try:
            value = int(raw)                       # rows carry it as str|int
        except (TypeError, ValueError):
            continue
        if value > 0:
            windows.append(value)
    return {"max_context_tokens": max(windows)} if windows else {}


# ---------------------------------------------------------------------------
# View construction — STUDIO side
# ---------------------------------------------------------------------------


def _studio_views() -> list[CapabilityView]:
    """The studio side, through its own typed API only: capability_verdict for
    the servability answer + wording (two gates, one wording — this catalog is
    a third consumer of the SAME wording, never a rival derivation),
    capable_model_ids for the honest model set (synthetic excluded), and the
    registry/preset gate facts for per-model reasons."""
    # Importing the package registers the zoo (models_seed side effect) —
    # reads only; validate_registry() is deliberately NOT called (it enforces
    # weight-pinning policy, a serve-path concern, not a catalog concern).
    from hugpy_video.intel.studio.presets import (
        STUB_RUNNER_MODULES,
        ZERO_BYTE_MODELS,
        capability_verdict,
    )
    from hugpy_video.intel.studio.registry import MODEL_REGISTRY, model_gate_reasons, runner_for
    from hugpy_video.intel.studio.router import capable_model_ids

    views: list[CapabilityView] = []
    for cap, name in sorted(STUDIO_CAPABILITY_NAME.items(),
                            key=lambda item: item[1]):
        verdict = capability_verdict(cap)
        ids = capable_model_ids(cap)

        reasons: list[str] = []
        if not verdict.servable:
            reasons.append(verdict.reason)
        elif not ids:
            # The verdict says servable but no REAL model is capable: the
            # capability is carried by the last-resort tier (the ffmpeg
            # enhancer / synthetic rows rank below every real model but DO
            # render — e.g. interp/upres on a GPU-less fleet). Honor the
            # studio's own verdict; say honestly which tier serves it.
            ids = capable_model_ids(cap, include_synthetic=True)
            if ids:
                reasons.append(
                    "served only by the last-resort tier (synthetic/ffmpeg "
                    "fallback — every real model for it is gated or absent)")

        # Per-model "declared it but cannot run it" reasons, from the same
        # facts the router rejects on (gate + viability), so the catalog and a
        # routing refusal can never tell different stories.
        declared = [m for m in MODEL_REGISTRY.values()
                    if cap in m.capabilities and not m.synthetic]
        for cfg in sorted(declared, key=lambda m: m.model_id):
            if cfg.model_id in ids:
                continue
            if cfg.model_id in ZERO_BYTE_MODELS:
                reasons.append(f"{cfg.model_id}: weights absent "
                               f"(0 bytes on the shared store)")
                continue
            stub_tasks = [
                t.value for t in cfg.tasks
                if (spec := runner_for(cfg.family, t)) is not None
                and spec.entrypoint.split(":", 1)[0] in STUB_RUNNER_MODULES]
            if stub_tasks:
                reasons.append(f"{cfg.model_id}: stub runner "
                               f"({', '.join(stub_tasks)} — every path returns Err)")
                continue
            gates = model_gate_reasons(cfg.model_id)
            if gates:
                gate_text = "; ".join(f"{t}: {why}" for t, why in sorted(gates.items()))
                reasons.append(f"{cfg.model_id}: runner gated ({gate_text})")

        eligible = verdict.servable and bool(ids)
        if not eligible and not reasons:
            reasons.append("no capable model on this fleet")

        bound = [cfg for cfg in MODEL_REGISTRY.values() if cfg.model_id in ids]
        vram_mins = [cfg.vram.min_gb() for cfg in bound]
        frameworks = tuple(sorted({cfg.family.value for cfg in bound}))
        accepts, produces = _STUDIO_IO[cap]
        views.append(CapabilityView(
            name=name,
            source=SourceRegistry.STUDIO,
            accepts=accepts,
            produces=produces,
            model_ids=ids,
            eligibility=Eligibility(eligible=eligible, reasons=tuple(reasons)),
            resources=ResourceHints(
                vram_gib=min(vram_mins) if vram_mins else None,
                # The zoo's VramEnvelope is a DECLARED envelope from
                # models_seed, not a number this box measured. Saying so is the
                # whole point of the provenance flag — k105 replaces it with
                # MEASURED per model as seatings report real figures.
                vram_provenance=(Provenance.DECLARED if vram_mins
                                 else Provenance.UNKNOWN),
                frameworks=frameworks),
            **_descriptor_fields(
                name,
                license=_studio_license(bound),
                limits=_studio_limits(bound),
                model_fingerprint=_studio_fingerprint(bound)),
        ))
    return views


def _studio_license(bound: list[Any]) -> str | None:
    """The bound models' declared ``LicenseClass``. The studio zoo records one
    per model, so unlike the legacy rows this is never a guess."""
    declared = sorted({cfg.license.value for cfg in bound})
    if not declared:
        return None
    return declared[0] if len(declared) == 1 else ", ".join(declared)


def _studio_limits(bound: list[Any]) -> dict[str, Any]:
    """Limits READ off the zoo's ModelConfig: the longest clip any bound model
    renders, the most frames, and the largest native resolution (by area). The
    best bound model's envelope, because the router will pick it."""
    if not bound:
        return {}
    limits: dict[str, Any] = {
        "max_duration_s": max(float(cfg.max_duration_s) for cfg in bound),
        "max_frames": max(int(cfg.max_frames) for cfg in bound),
    }
    best = None
    for cfg in bound:
        for res in cfg.resolutions:
            if best is None or res.area > best.area:
                best = res
    if best is not None:
        limits["max_resolution"] = f"{best.width}x{best.height}"
        limits["nominal_fps"] = best.fps
    return limits


def _studio_fingerprint(bound: list[Any]) -> str | None:
    """A content fingerprint of the bound model set — the digest of every
    model's PINNED ``weight_hash``.

    None when ANY bound model is unpinned, which is most of them today
    (``studio.registry.unpinned_models()`` is the zoo's own report on this). A
    fingerprint that silently skipped the unpinned rows would claim to identify
    weights it cannot identify; the honest answer is no fingerprint. Weight
    hashes for the LEGACY registry are k105's — discovery records no checksum
    today, and hashing 13.9 GB is not something a capability listing may do
    (``model_dir_fingerprint`` is the opt-in, one-model version)."""
    if not bound:
        return None
    pins = {cfg.model_id: cfg.weight_hash for cfg in bound}
    if any(not h for h in pins.values()):
        return None
    return _digest({"weights": {k: pins[k] for k in sorted(pins)}})


# ---------------------------------------------------------------------------
# View construction — SPEECH side (k98)
# ---------------------------------------------------------------------------


def _row_license(row: dict[str, Any]) -> str | None:
    """The row's declared license, or None when it declares none. Looks in the
    row and in its ``extra`` bag (discovery writes the HF card's license there
    on some paths). None means UNKNOWN — never "permissive"."""
    for bag in (row, row.get("extra") or {}, (row.get("extra") or {}).get("extra") or {}):
        if not isinstance(bag, dict):
            continue
        value = bag.get("license")
        if value:
            return str(value)
    return None


def _row_blob(key: str, row: dict[str, Any]) -> str:
    """The identifying strings of a row, lowered, for marker matching."""
    parts = [key, str(row.get("model_key") or ""), str(row.get("name") or ""),
             str(row.get("hub_id") or ""), str(row.get("folder") or "")]
    return " ".join(parts).lower()


def _tts_rows(rows: dict[str, dict[str, Any]]) -> dict[str, str]:
    """model_key -> advisory note, for every row that backs audio.tts.

    A row qualifies by DECLARING ``text-to-speech`` (note "") or by carrying a
    TTS_MODEL_MARKERS identifier while declaring something else (note explains
    the misclassification). Both are recorded; neither is invented."""
    hits: dict[str, str] = {}
    for key, row in rows.items():
        tasks = tuple(row.get("tasks") or ())
        if TTS_TASK in tasks:
            hits[key] = ""
            continue
        if any(m in _row_blob(key, row) for m in TTS_MODEL_MARKERS):
            hits[key] = (
                f"row declares tasks={list(tasks)} but its identifier matches a "
                f"known reference-conditioned TTS backend; the model card "
                f"declares pipeline_tag text-to-speech and the classifier could "
                f"not read it (no transformers config). Bound by name — re-run "
                f"discovery to record tasks=['{TTS_TASK}'] on the row")
    return hits


def _speaker_embedding_rows(rows: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    """Registry rows that look like a speaker-embedding model. Empty on this
    fleet — the capability then declares itself with NO binding."""
    return tuple(sorted(
        key for key, row in rows.items()
        if any(m in _row_blob(key, row) for m in SPEAKER_EMBEDDING_MARKERS)))


def _tts_view(rows: dict[str, dict[str, Any]], blocked: set[str],
              workers: list[dict[str, Any]] | None) -> CapabilityView:
    """``audio.tts`` — the doc §4 ``voice.synthesize.reference_conditioned``.

    Eligibility is the conjunction of the three gates the task names, resolved
    in doc §4's own order (compatibility -> policy -> adapter probe -> worker
    health):

      (a) the RUNNER ADAPTER is registered and its backend is reachable —
          either here (``probe()['importable']``) or on a worker that seats it;
      (b) a WORKER SEAT affirmatively advertises the task (strict, see
          ``_worker_seats_task``) — or central can run it itself;
      (c) the model's LICENSE is not a declared refusal.

    On this fleet (b) fails and central cannot run it either, so the view is
    INELIGIBLE with a reason naming both the runner and the missing seat. That
    is the whole point: never fake eligibility for a capability no box can
    execute."""
    hits = _tts_rows(rows)
    model_ids = tuple(sorted(k for k in hits if k not in blocked))
    reasons: list[str] = []

    if not hits:
        reasons.append(f"no model registered for task(s) {TTS_TASK}")
    elif not model_ids:
        reasons.append(
            f"every registered model for task(s) {TTS_TASK} is "
            f"operator-blocked from the serving pool")
    for key in model_ids:
        if hits[key]:
            reasons.append(f"{key}: {hits[key]}")

    # (c) license
    license_ok = True
    for key in model_ids:
        declared = _row_license(rows.get(key) or {})
        if declared is None:
            reasons.append(
                f"{key}: license not recorded on the registry row — verify it "
                f"before a worker seats this model (the upstream card declares "
                f"MIT); unknown is reported, never assumed permissive")
            continue
        low = declared.lower()
        if any(m in low for m in TTS_REFUSED_LICENSE_MARKERS):
            license_ok = False
            reasons.append(
                f"{key}: license {declared!r} refuses reference-conditioned "
                f"voice synthesis on this fleet")

    # (a) adapter registration + backend health probe
    probed = _tts_runner_probe()
    runner_registered = bool(probed.get("runner_registered"))
    backend_here = bool(probed.get("importable"))
    module = probed.get("module") or TTS_RUNNER_MODULE
    if not runner_registered:
        reasons.append(f"{module}: {probed.get('reason') or 'runner adapter unavailable'}")
    elif not backend_here:
        reasons.append(
            f"{module}: runner adapter registered but its backend is "
            f"unavailable here ({probed.get('reason') or 'backend not importable'})")

    # (b) worker seat — strict/affirmative
    seat_ok = False
    if workers is None:
        reasons.append(
            "worker registry unreadable — no worker seat can be confirmed for "
            "the chatterbox TTS runner")
    elif not workers:
        reasons.append("no online worker registered")
    else:
        seat_ok = any(_worker_seats_task(w, TTS_TASK) for w in workers)
        if not seat_ok:
            reasons.append(
                f"no online worker seats the chatterbox TTS runner: none of the "
                f"{len(workers)} online worker(s) advertises "
                f"task_capabilities[{TTS_TASK!r}] (STRICT/affirmative — unlike "
                f"the legacy-permissive task gate, an un-enumerated task on a "
                f"brand-new capability means 'not yet', never 'assume yes')")

    eligible = bool(model_ids) and runner_registered and license_ok and (
        seat_ok or backend_here)
    if not eligible and not reasons:  # defensive: Eligibility requires reasons
        reasons.append("no execution path for audio.tts on this fleet")

    frameworks = tuple(sorted({
        str(rows[m].get("framework")) for m in model_ids
        if rows.get(m, {}).get("framework")}))
    accepts, produces = _SPEECH_IO["audio.tts"]
    return CapabilityView(
        name="audio.tts", source=SourceRegistry.TASKS,
        accepts=accepts, produces=produces, model_ids=model_ids,
        eligibility=Eligibility(eligible=eligible, reasons=tuple(reasons)),
        resources=ResourceHints(
            # MEASURED on the first seating (see TTS_MEASURED_VRAM_GIB) — the
            # catalog publishes a measurement over an estimate, and publishes
            # nothing at all when it has neither. k98 deliberately left this
            # None rather than guess; the guess it declined to make (~6 GiB)
            # was high.
            vram_gib=TTS_MEASURED_VRAM_GIB,
            vram_provenance=Provenance.MEASURED,
            frameworks=frameworks,
            notes=(f"reference-conditioned TTS; adapter {module} "
                   f"(pip {probed.get('pip') or 'chatterbox-tts'}); a reference "
                   f"voice requires an explicit voice authorization; every "
                   f"generated file carries Resemble's PerTh watermark "
                   f"(the backend's own, not optional); VRAM measured "
                   f"2026-08-21 on an RTX 3090")),
        **_descriptor_fields(
            "audio.tts",
            license=_declared_license(rows, model_ids),
            # Forward-compatible: k98's probe() does not report a version yet.
            # When the adapter declares one it appears here with no edit.
            adapter_version=probed.get("adapter_version")))


def _word_timestamps_view(base: CapabilityView) -> CapabilityView:
    """``audio.transcribe.word_timestamps`` — the SAME whisper backend as
    ``audio.transcribe``, plus one flag and a stronger promise.

    Eligibility TRACKS the base capability (same models, same workers, same
    dependency probe — it is the same execution) and additionally requires the
    ``word_timestamps`` passthrough to exist. Without it whisper returns
    segments whose ``words`` list is empty, so declaring the capability eligible
    would promise timing the fleet cannot produce: ineligible beats faked, and
    the reason names the four edits that close it."""
    reasons = list(base.eligibility.reasons)
    wired = _word_timestamps_wired()
    if not wired:
        reasons.append(WORD_TIMESTAMPS_PASSTHROUGH_GAP)
    eligible = base.eligibility.eligible and wired
    if not eligible and not reasons:
        reasons.append("audio.transcribe is not eligible on this fleet")
    accepts, produces = _SPEECH_IO["audio.transcribe.word_timestamps"]
    return CapabilityView(
        name="audio.transcribe.word_timestamps", source=SourceRegistry.TASKS,
        accepts=accepts, produces=produces, model_ids=base.model_ids,
        eligibility=Eligibility(eligible=eligible, reasons=tuple(reasons)),
        resources=ResourceHints(
            vram_gib=base.resources.vram_gib,
            vram_provenance=base.resources.vram_provenance,
            frameworks=base.resources.frameworks,
            notes=("same backend as audio.transcribe, dispatched with "
                   "word_timestamps=True; returns TranscribeSegment.words "
                   "(word/start/end/probability — whisper_schemas.TranscribeWord)")),
        # The base capability's descriptor facts carry over unchanged — it IS
        # the same execution, so a different license or context window here
        # would be a contradiction, not a refinement.
        **_descriptor_fields("audio.transcribe.word_timestamps",
                             license=base.license,
                             limits=dict(base.limits.to_dict())))


def _speaker_similarity_view(rows: dict[str, dict[str, Any]],
                             blocked: set[str]) -> CapabilityView:
    """``audio.speaker_similarity`` — embedding-based, DECLARED, unbound.

    The planner needs the name to exist (doc §9 Identity: "face and speaker
    embeddings plus an independent VLM rubric"; ``speech.check_speaker_similarity``
    consumes its output). No speaker-embedding model is registered on this
    fleet, so the view carries no model and refuses by name. k98 does not invent
    one — an unbound declaration is the honest encoding of a gap."""
    found = tuple(k for k in _speaker_embedding_rows(rows) if k not in blocked)
    if not found:
        reasons = (
            f"no speaker-embedding model registered (searched the legacy "
            f"registry for {', '.join(SPEAKER_EMBEDDING_MARKERS[:6])}, … "
            f"markers): the capability is declared so a plan can name it, but "
            f"nothing on this fleet implements it — register a speaker-embedding "
            f"model and an adapter for it",)
    else:
        reasons = (
            f"model(s) {list(found)} match a speaker-embedding marker but no "
            f"runner adapter is registered for audio.speaker_similarity "
            f"(declare one the way runners/tts_chatterbox declares probe())",)
    accepts, produces = _SPEECH_IO["audio.speaker_similarity"]
    return CapabilityView(
        name="audio.speaker_similarity", source=SourceRegistry.TASKS,
        accepts=accepts, produces=produces, model_ids=found,
        eligibility=Eligibility(eligible=False, reasons=reasons),
        resources=ResourceHints(
            notes=("cosine similarity between two speaker embeddings; consumed "
                   "by oracle.speech.check_speaker_similarity "
                   "(VOICE_SIMILARITY_LOW)")),
        **_descriptor_fields("audio.speaker_similarity",
                             license=_declared_license(rows, found)))


def _speech_views(legacy: list[CapabilityView] | None = None) -> list[CapabilityView]:
    """The three k98 speech capabilities, sorted by name. ``legacy`` is the
    already-built legacy view list (so ``audio.transcribe`` is not derived
    twice); it is rebuilt when absent."""
    rows = _legacy_registry_rows()
    blocked = _blocked_model_keys()
    workers = _online_workers()
    views = [_tts_view(rows, blocked, workers),
             _speaker_similarity_view(rows, blocked)]

    base = next((v for v in (legacy or _legacy_views())
                 if v.name == "audio.transcribe"), None)
    if base is not None:
        views.append(_word_timestamps_view(base))
    return sorted(views, key=lambda v: v.name)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def _digest(payload: Any) -> str:
    """``sha256:<16 hex>`` over the canonical JSON of ``payload``. Truncated
    because this is an identity for humans to compare in a receipt, not a
    security primitive — 64 bits of collision resistance over a registry
    snapshot is plenty, and a full digest makes receipts unreadable."""
    encoded = canonical_json(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()[:16]


def _row_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """The ROUTING-RELEVANT projection of a registry row.

    Deliberately NOT the whole row: discovery writes mtimes, byte counts and
    scan bookkeeping that churn without changing a single routing decision, and
    a snapshot digest that moved every time someone re-ran discovery would be
    noise nobody could act on. These five facts are exactly what the catalog
    and the router read."""
    tasks = row.get("tasks") or ()
    if isinstance(tasks, str):                    # some rows carry it as a str
        tasks = (tasks,)
    return {
        "tasks": sorted(str(t) for t in tasks),
        "framework": str(row.get("framework") or ""),
        "hub_id": str(row.get("hub_id") or ""),
        "license": _row_license(row) or "",
        "context_tokens": str(row.get("model_max_length") or ""),
    }


def _studio_snapshot() -> dict[str, Any] | None:
    """The studio zoo's routing-relevant facts, or None when it is unreadable
    (an unreadable half IS a different snapshot — the digest says so rather
    than pretending the zoo was empty)."""
    try:
        from hugpy_video.intel.studio.registry import MODEL_REGISTRY
    except Exception as exc:  # noqa: BLE001
        logger.debug("oracle catalog: studio registry unreadable (%s)", exc)
        return None
    return {
        cfg.model_id: {
            "vram_gib": cfg.vram.min_gb(),
            "license": cfg.license.value,
            "weight_hash": cfg.weight_hash or "",
            "capabilities": sorted(c.value for c in cfg.capabilities),
            "synthetic": bool(cfg.synthetic),
        }
        for cfg in sorted(MODEL_REGISTRY.values(), key=lambda c: c.model_id)
    }


def declared_capability_names() -> tuple[str, ...]:
    """Every CANONICAL capability name the catalog can publish, from the tables
    alone — no registry read, no worker read. The snapshot needs this without
    recursing into ``list_capabilities`` (which reads the snapshot)."""
    names = (set(LEGACY_TASK_CAPABILITY.values())
             | set(STUDIO_CAPABILITY_NAME.values())
             | set(_SPEECH_IO))
    return tuple(sorted(names))


def registry_snapshot(rows: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The exact payload ``registry_version`` digests.

    Exposed because a version string nobody can DIFF is a version string nobody
    trusts: when two receipts disagree, this is what an operator compares to
    find out which row moved."""
    rows = _legacy_registry_rows() if rows is None else rows
    return {
        "schema": REGISTRY_SNAPSHOT_SCHEMA,
        "rows": {key: _row_snapshot(row) for key, row in sorted(rows.items())},
        "studio": _studio_snapshot(),
        "capabilities": {name: capability_version(name)
                         for name in declared_capability_names()},
    }


def registry_version(rows: dict[str, dict[str, Any]] | None = None) -> str:
    """The deterministic digest of the routing snapshot: legacy row projections
    + the studio zoo's pinned facts + every descriptor version (k101; k105 adds
    measured VRAM to the same payload).

    Two identical fleets produce the same string; ONE changed task string,
    license, context window, weight pin or descriptor version changes it. It
    rides on every ``CapabilityView`` and belongs in every ``ExecutionReceipt``
    — a receipt without it records what ran but not what it was chosen FROM."""
    return _digest(registry_snapshot(rows))


def model_dir_fingerprint(directory: str) -> str | None:
    """Digest of a model directory's top-level MANIFEST (names + sizes), or
    None when the directory is unreadable.

    OPT-IN and one directory at a time: this is NOT called during a capability
    listing (117 rows × a scandir on a shared store is not a page-load cost),
    it is the honest cheap fingerprint for a caller that wants one — k105's
    registry snapshot is its first real consumer. It fingerprints the FILE
    MANIFEST, never the bytes: hashing 13.9 GB of weights is a different
    operation with a different cost, and conflating the two would put a
    reassuring hash next to weights nobody verified."""
    try:
        entries = sorted(probes._scandir_entries(directory))
    except OSError as exc:
        logger.debug("oracle catalog: %s unreadable (%s)", directory, exc)
        return None
    return _digest({"manifest": [[name, size] for name, size in entries]})


def _probe_views(views: list[CapabilityView], *,
                 rows: dict[str, dict[str, Any]],
                 workers: list[dict[str, Any]] | None,
                 version: str) -> list[CapabilityView]:
    """Stamp the snapshot version on every view and run each capability's
    REGISTRATION PROBE (doc §4 step 3, run at catalog build).

    A capability with no registered probe keeps ``probe=None`` — "nobody
    declared one", which is not the same claim as "probed and inconclusive". A
    FAILING probe makes the view ineligible with the probe's own detail as the
    reason; an ``unknown`` probe is advisory only."""
    out: list[CapabilityView] = []
    for view in views:
        stamped = replace(view, registry_version=version)
        result = probes.probe_capability(
            stamped.name, rows=rows, model_ids=stamped.model_ids,
            workers=workers, registry_version=version)
        out.append(stamped.with_probe(result))
    return out


def list_capabilities() -> list[CapabilityView]:
    """Every namespaced capability from BOTH registries, sorted by name. The
    two vocabularies are disjoint by construction (the mapping tables share no
    names — proven by tests AND re-checked here, because a silent collision
    would mean one registry's view shadows the other's).

    k98's speech views join the same list and the same collision check. Only
    CANONICAL names are listed; ``CAPABILITY_ALIASES`` resolve through
    ``get_capability`` so one capability is never published twice.

    k101: every view leaves here as a full DESCRIPTOR — stamped with the
    ``registry_version`` it was built from and carrying its registration probe
    (TTL-cached, hard-budgeted; see probes.py). ``GET /oracle/capabilities``
    serializes all of it with no route change, because the route serializes
    whatever the catalog returns."""
    legacy = _legacy_views()
    views = _studio_views() + legacy + _speech_views(legacy)
    by_name: dict[str, CapabilityView] = {}
    for view in views:
        if view.name in by_name:
            raise RuntimeError(
                f"capability name collision across registries: {view.name!r} "
                f"({by_name[view.name].source.value} vs {view.source.value})")
        by_name[view.name] = view
    rows = _legacy_registry_rows()
    return _probe_views(sorted(by_name.values(), key=lambda v: v.name),
                        rows=rows, workers=_online_workers(),
                        version=registry_version(rows))


def canonical_name(name: str) -> str:
    """``name`` with ``CAPABILITY_ALIASES`` applied (identity for a canonical or
    unknown name). One hop only — aliases of aliases are not a thing."""
    return CAPABILITY_ALIASES.get(name, name)


def capability_params(name: str) -> dict[str, Any]:
    """The fixed dispatch parameters ``name`` implies (a COPY, so a caller
    cannot mutate the table). ``{"word_timestamps": True}`` for the
    word-timestamp capability; empty for everything else."""
    return dict(CAPABILITY_FIXED_PARAMS.get(canonical_name(name), {}))


def get_capability(name: str) -> CapabilityView | None:
    """The single view for ``name``, or None when no such capability exists.
    Aliases resolve (``voice.synthesize.reference_conditioned`` -> the
    ``audio.tts`` view), so a planner may use the doc's vocabulary."""
    target = canonical_name(name)
    for view in list_capabilities():
        if view.name == target:
            return view
    return None


def resolve_owners(name: str) -> tuple[SourceRegistry, tuple[str, ...]] | None:
    """Which registry owns ``name`` and which model ids implement it — the
    lookup k90b's router starts from. None for an unknown name; aliases resolve
    through ``get_capability``."""
    view = get_capability(name)
    if view is None:
        return None
    return (view.source, view.model_ids)


def unmapped_tasks() -> tuple[str, ...]:
    """Reporting hook (mirrors studio ``unpinned_models()``): task strings
    present on legacy registry rows but absent from BOTH mapping tables —
    discovery noise the catalog is deliberately not inventing capabilities
    for. Empty when the tables cover the live registry."""
    known = (set(LEGACY_TASK_CAPABILITY) | set(LEGACY_TASK_EXCLUDED)
             | set(SPEECH_CAPABILITY_TASK.values()))
    seen: set[str] = set()
    for row in _legacy_registry_rows().values():
        for task in (row.get("tasks") or ()):
            if task and task not in known:
                seen.add(str(task))
    return tuple(sorted(seen))


__all__ = [
    "LEGACY_TASK_CAPABILITY", "LEGACY_TASK_EXCLUDED",
    "STUDIO_CAPABILITY_NAME", "STUDIO_CAPABILITY_EXCLUDED",
    # speech (k98)
    "CAPABILITY_ALIASES", "CAPABILITY_FIXED_PARAMS", "SPEECH_CAPABILITY_TASK",
    "SPEAKER_EMBEDDING_MARKERS", "TTS_MODEL_MARKERS", "TTS_RUNNER_MODULE",
    "TTS_TASK", "WORD_TIMESTAMPS_PASSTHROUGH_GAP",
    # descriptor + registry snapshot (k101)
    "CAPABILITY_ACCESS_DECL", "CAPABILITY_LIMITS", "CAPABILITY_PARAM_SCHEMA",
    "CAPABILITY_RESULT_SCHEMA", "CAPABILITY_VERSION",
    "REGISTRY_SNAPSHOT_SCHEMA", "capability_authority", "capability_eval_suite",
    "capability_version", "declared_capability_names", "model_dir_fingerprint",
    "registry_snapshot", "registry_version",
    "canonical_name", "capability_params",
    "list_capabilities", "get_capability", "resolve_owners", "unmapped_tasks",
]
