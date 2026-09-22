"""k106 — ``video.performance``: the FAT orchestrator for the first audio-first
video slice (roadmap Wave 3; doc Phase 5 over Stages 1, 4, 8, 11, 14, 15, 17,
18, 19).

WHAT THIS IS. One resumable function, :func:`run_performance`, that wires the
Wave-1/2 pieces into the doc's ordered recipe and ends honestly:

    1. authority   k97 ``authority.check`` for every identity/voice ref
    2. snapshot    k104 ``GenerationSnapshot`` + ``RunPromptLedger``
    3. audio       k102 ``build_audio_master`` (N candidates per line)
    4. lock        k104 shot windows -> ``ShotPlan`` -> ``ProductionLock.lock``
    5. segments    k104 ``compile_segments`` -> ``to_plan_graph`` -> k103 validate
    6. keyframes   Stage 15 identity-first stills, judged before any video
    7. clips       Stage 17 per-shot candidates, judged, ONE bounded repair
    8. assembly    Stage 18/19 concat + final round-trip ASR over every line

It is deliberately a FAT ORCHESTRATOR and not a DAG runtime: the roadmap's
ground rule is "FAT orchestrator before DAG runtime", and k111 is where the
generalization lands. What this module owes the future runtime is the SHAPE —
a ``PlanGraph`` is still emitted and still statically validated at stage 5, so
the plan k111 will execute is the plan this function executes by hand.

THE FIVE RULES THIS MODULE ENFORCES, none of them decorative:

1. **Nothing is faked.** Every backend is an INJECTED SEAM
   (:class:`PerformanceSeams`). A seam that is not bound on this fleet does not
   get simulated: the stage that needs it returns a typed ``CAPABILITY_GAP``
   naming the capability AND the operator step that would seat it, and no later
   stage runs. Today that is ``synth`` — no worker seats chatterbox — so a LIVE
   run refuses at stage 3, by design, with the seat requirement in the gap.
2. **The generator never grades itself.** Keyframes and clips are judged by
   ``seams.judge_image`` / ``seams.judge_clip``; a judge that returns nothing is
   recorded as UNSCORED (``speech.UNSCORED_PREFIX``), which lowers the card's
   ``confidence`` and adds a line to ``limitations`` — it never silently
   becomes a pass with full confidence.
3. **Repairs are bounded and targeted** (Stage 17: "a targeted failure does not
   rerun unrelated accepted nodes"). An identity failure re-runs the KEYFRAME
   and only the keyframe — never transcription, never audio, never a sibling
   segment. A clip failure gets exactly ONE repair round, decided by
   ``repair.attempt_repair`` semantics (a decision object, then at most one
   execution, then the second card stands whatever it says). There is no loop
   anywhere in this module.
4. **Segments are siblings.** Stage 5 goes through k104's compiler, which makes
   invariant 9 structural; this module adds the ``RunPromptLedger`` k104 could
   not supply, so a prompt minted DURING the run can never be laundered back
   into the immutable snapshot.
5. **Resume is cheap and verified.** Every stage journals its artifact digests
   to ``runs/performance/<run_id>/state.json`` (mirroring where
   ``runners/movie`` writes its work dir). ``resume=<run_id>`` re-derives each
   recorded artifact from its payload and compares DIGESTS before skipping the
   stage; a mismatch, a missing file, or a different goal digest re-executes
   rather than trusting the journal. The AUTHORITY gate is the one stage that
   is NEVER resumed: a grant revoked between runs must stop the resumed run,
   and re-checking it costs nothing (it calls no seam).

WHAT IS NOT HERE, said out loud (these ride on every result's ``limitations``):
no lip-sync evaluator (k121), no speaker-similarity backend on this fleet, no
spatial conditioning (``SegmentSpec.spatial_ref`` is None until Wave 5), no
continuity-with-neighbours check (k107), no director/mix/colour finishing pass
(k108, doc Stage 18 steps 2-4), no screenplay artifact (k110).

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from hugpy_oracle import authority as oracle_authority
from hugpy_oracle import repair as oracle_repair
from hugpy_oracle import speech
from hugpy_oracle.audio_master import (
    AudioBuildResult,
    AudioMaster,
    DialogueTimeline,
    Line,
    SpeechPolicy,
    VoiceKind,
    VoiceProfile,
    build_audio_master,
)
from hugpy_oracle.contracts import (
    ArtifactKind,
    ArtifactRef,
    AuthorityKind,
    Check,
    CheckKind,
    ExecutionReceipt,
    FailureClass,
    GoalSpec,
    InputKind,
    InputRef,
    JudgeResult,
    RepairCode,
    Scorecard,
)
from hugpy_oracle.evaluation import THRESHOLDS
from hugpy_oracle.plan import PlanGraph
from hugpy_oracle.production import (
    ContinuityBible,
    ContinuityState,
    GenerationSnapshot,
    LockRefused,
    ProductionError,
    ProductionLock,
    RunPromptLedger,
    ShotPlan,
    digest_payload,
)
from hugpy_oracle.router import RouteDecision
from hugpy_oracle.segments import (
    SEGMENT_CAPABILITY,
    CompileRefused,
    SegmentSpec,
    SiblingViolation,
    compile_segments,
    default_prompt_writer,
    execution_order,
    segment_seed,
    shot_plan_from_windows,
    shot_windows_from_audio,
    to_plan_graph,
)
from hugpy_oracle.validator import ValidationReport, validate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The recipe itself. It has no ``CapabilityDescriptor`` in the catalog (k101
#: describes IMPLEMENTATIONS; a recipe is a composition of them), so this name
#: appears on receipts and nowhere in ``GET /oracle/capabilities``.
PERFORMANCE_CAPABILITY = "video.performance"

TTS_CAPABILITY = "audio.tts"
TRANSCRIBE_CAPABILITY = "audio.transcribe.word_timestamps"
TRANSCRIBE_FALLBACK_CAPABILITY = "audio.transcribe"
SIMILARITY_CAPABILITY = "audio.speaker_similarity"
KEYFRAME_CAPABILITY = "image.generate"
KEYFRAME_JUDGE_CAPABILITY = "image.understand"
CLIP_CAPABILITY = SEGMENT_CAPABILITY          # "video.generate.i2v"
ASSEMBLE_CAPABILITY = "video.assemble"

#: The stages, in the ONE order they run. The tuple is the source of truth for
#: the journal, for ``stop_after``, and for "later stages untouched".
STAGES: tuple[str, ...] = (
    "authority", "snapshot", "audio", "lock", "segments",
    "keyframes", "clips", "assembly",
)

#: Which capability each stage's receipt names. Stages that are pure
#: composition over already-produced artifacts name the recipe.
STAGE_CAPABILITY: dict[str, str] = {
    "authority": PERFORMANCE_CAPABILITY,
    "snapshot": PERFORMANCE_CAPABILITY,
    "audio": TTS_CAPABILITY,
    "lock": PERFORMANCE_CAPABILITY,
    "segments": PERFORMANCE_CAPABILITY,
    "keyframes": KEYFRAME_CAPABILITY,
    "clips": CLIP_CAPABILITY,
    "assembly": ASSEMBLE_CAPABILITY,
}

#: The authority gate is never resumed — see rule 5 in the module docstring.
RESUMABLE_STAGES: tuple[str, ...] = tuple(s for s in STAGES if s != "authority")

#: The capabilities whose k97 requirements are collected at stage 1. The
#: pipeline really does route through all four, so the gate is asked about all
#: four BEFORE anything is synthesized.
GATED_CAPABILITIES: tuple[str, ...] = (
    TTS_CAPABILITY, TRANSCRIBE_CAPABILITY, KEYFRAME_CAPABILITY, CLIP_CAPABILITY,
)

DEFAULT_TTS_CANDIDATES = 3
DEFAULT_KEYFRAME_CANDIDATES = 3
DEFAULT_CLIP_CANDIDATES = 3

#: One bounded repair round, and its deterministic seed salt. Different takes,
#: still exactly reproducible — the k102/k104 idiom.
KEYFRAME_REPAIR_SALT = "k106:keyframe-repair:1"
CLIP_REPAIR_SALT = "k106:clip-repair:1"

#: Stage 9 gives every shot a rubric; a shot with none cannot be judged and
#: therefore cannot be accepted (invariant 11). k110 authors the real ones —
#: until then this is the honest minimum, and it is DATA on the goal, not a
#: constant buried in the compiler.
DEFAULT_RUBRIC: tuple[str, ...] = (
    "the shot shows the speaking character clearly enough to read identity",
    "the shot matches the written action and setting",
    "the shot holds for the whole of its audio window",
)

STATE_VERSION = 1
STATE_FILENAME = "state.json"
RUN_ROOT_ENV = "HUGPY_PERFORMANCE_RUN_ROOT"


class PerformanceError(ValueError):
    """A caller/wiring fault — raised at the boundary, never a run outcome."""


class SeamUnavailable(RuntimeError):
    """A bound seam could not reach its backend. Raised by the LIVE seam
    implementations so the orchestrator turns it into a typed gap with the
    reason intact, instead of a bare exception crossing a stage boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _q(value: Any) -> float:
    return round(float(value), 6)


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------

SynthFn = Callable[[Line, VoiceProfile, int], Any]
TranscribeFn = Callable[[str], Any]
SimilarityFn = Callable[[str, VoiceProfile], "float | None"]
GenImageFn = Callable[[str, "tuple[str, ...]", int], Any]
JudgeImageFn = Callable[[str, SegmentSpec], Any]
GenClipFn = Callable[[str, SegmentSpec], Any]
JudgeClipFn = Callable[[str, SegmentSpec], Any]
ConcatFn = Callable[["tuple[str, ...]", AudioMaster], Any]

#: Every seam name, in pipeline order. Used to validate overrides and to keep
#: ``unbound`` honest.
SEAM_NAMES: tuple[str, ...] = (
    "synth", "transcribe", "similarity", "gen_image", "judge_image",
    "gen_clip", "judge_clip", "concat",
)

#: seam -> the capability it stands for. A gap names the CAPABILITY, because
#: that is what an operator seats; the seam name is this module's plumbing.
SEAM_CAPABILITY: dict[str, str] = {
    "synth": TTS_CAPABILITY,
    "transcribe": TRANSCRIBE_CAPABILITY,
    "similarity": SIMILARITY_CAPABILITY,
    "gen_image": KEYFRAME_CAPABILITY,
    "judge_image": KEYFRAME_JUDGE_CAPABILITY,
    "gen_clip": CLIP_CAPABILITY,
    "judge_clip": CLIP_CAPABILITY,
    "concat": ASSEMBLE_CAPABILITY,
}


@dataclass(frozen=True, slots=True)
class SeamGap:
    """An UNBOUND seam, with the operator step that would bind it.

    ``requirement`` is the whole value of this type: "audio.tts is unavailable"
    is a status line; "pip install chatterbox-tts on a GPU worker and heartbeat
    task_capabilities['text-to-speech']" is an instruction. A gap without an
    instruction is a shrug with a schema."""

    seam: str
    capability: str
    requirement: str

    def __post_init__(self) -> None:
        if self.seam not in SEAM_NAMES:
            raise PerformanceError(
                f"SeamGap.seam must be one of {list(SEAM_NAMES)}, got "
                f"{self.seam!r}")
        if not str(self.capability).strip():
            raise PerformanceError("SeamGap.capability must be non-empty")
        if not str(self.requirement).strip():
            raise PerformanceError(
                f"SeamGap({self.seam}) carries no requirement — an unbound "
                f"seam with no operator step is not a usable gap")

    def to_dict(self) -> dict[str, Any]:
        return {"seam": self.seam, "capability": self.capability,
                "requirement": self.requirement}


@dataclass(frozen=True, slots=True)
class PerformanceSeams:
    """Everything the orchestrator calls OUT to, as injectables.

    This module imports no backend and no runner. A test binds fakes; a live
    caller uses :func:`default_seams`, which binds what this fleet can actually
    do today and records a :class:`SeamGap` for everything it cannot.

    Seam contracts (all deliberately small — the orchestrator adapts, the
    backend does not have to):

        ``synth(line, voice, seed)``      -> ``(audio_ref, duration_s)``
        ``transcribe(audio_ref)``         -> word items (k102 coerces any shape)
        ``similarity(audio_ref, voice)``  -> ``float | None``
        ``gen_image(prompt, identity_refs, seed)`` -> image ref (or a mapping
                                             carrying ``ref``/``uri``)
        ``judge_image(image_ref, spec)``  -> verdict mapping (see
                                             :func:`coerce_verdict`)
        ``gen_clip(keyframe_ref, spec)``  -> ``clip_ref`` or
                                             ``(clip_ref, duration_s)``
        ``judge_clip(clip_ref, spec)``    -> verdict mapping
        ``concat(clip_refs, audio_master)`` -> assembled video ref

    ``registry_version`` and ``catalog_view`` are seams too, for the same
    reason: reading the live catalog is a fleet call, and a test must be able to
    pin both without a registry."""

    synth: SynthFn | None = None
    transcribe: TranscribeFn | None = None
    similarity: SimilarityFn | None = None
    gen_image: GenImageFn | None = None
    judge_image: JudgeImageFn | None = None
    gen_clip: GenClipFn | None = None
    judge_clip: JudgeClipFn | None = None
    concat: ConcatFn | None = None
    registry_version: Callable[[], "str | None"] | None = None
    catalog_view: Callable[[], Mapping[str, Any]] | None = None
    tts_candidates: int = DEFAULT_TTS_CANDIDATES
    keyframe_candidates: int = DEFAULT_KEYFRAME_CANDIDATES
    clip_candidates: int = DEFAULT_CLIP_CANDIDATES
    unbound: tuple[SeamGap, ...] = ()
    run_root: str | None = None

    def __post_init__(self) -> None:
        for name in ("tts_candidates", "keyframe_candidates",
                     "clip_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PerformanceError(
                    f"PerformanceSeams.{name} must be an int >= 1, got "
                    f"{value!r} — a fan-out of zero candidates produces "
                    f"nothing to judge")
        object.__setattr__(self, "unbound", tuple(self.unbound))
        for gap in self.unbound:
            if not isinstance(gap, SeamGap):
                raise PerformanceError(
                    f"PerformanceSeams.unbound takes SeamGap, got "
                    f"{type(gap).__name__}")
            if getattr(self, gap.seam) is not None:
                raise PerformanceError(
                    f"seam {gap.seam!r} is BOUND but also listed as unbound — "
                    f"a gap that is not a gap is worse than no gap at all")

    def bound(self, seam: str) -> bool:
        if seam not in SEAM_NAMES:
            raise PerformanceError(f"unknown seam {seam!r}")
        return getattr(self, seam) is not None

    def gap_for(self, seam: str) -> SeamGap:
        """The gap for an unbound seam. Never None: a stage that needs an
        unbound seam always has something to say.

        Four sources, in order: this seam set's OWN record; the fleet's
        standing table (:data:`LIVE_SEAM_GAPS`), so a caller that assembled a
        bare ``PerformanceSeams`` still gets the real operator step instead of
        a shrug; :data:`LIVE_SEAM_BINDINGS`, for a seam this fleet CAN do that
        this seam set chose to unbind (saying "no implementation is wired"
        there would be a lie); and finally a generic one naming the
        capability."""
        for gap in self.unbound:
            if gap.seam == seam:
                return gap
        for gap in LIVE_SEAM_GAPS:
            if gap.seam == seam:
                return gap
        live = LIVE_SEAM_BINDINGS.get(seam)
        if live:
            return SeamGap(seam=seam, capability=SEAM_CAPABILITY[seam],
                           requirement=live)
        return SeamGap(
            seam=seam, capability=SEAM_CAPABILITY[seam],
            requirement=(f"bind PerformanceSeams.{seam} (capability "
                         f"{SEAM_CAPABILITY[seam]}); no implementation is "
                         f"wired for it on this fleet"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bound": [n for n in SEAM_NAMES if self.bound(n)],
            "unbound": [self.gap_for(n).to_dict() for n in SEAM_NAMES
                        if not self.bound(n)],
            "tts_candidates": self.tts_candidates,
            "keyframe_candidates": self.keyframe_candidates,
            "clip_candidates": self.clip_candidates,
        }


# --- the live bindings ------------------------------------------------------
#
# Every function below is LAZY about its imports and RAISES SeamUnavailable
# with the reason intact. None of them is called at import time.


def _seam_goal(objective: str, prompt: str, capability: str,
               inputs: Sequence[InputRef] = (),
               rights: Any = None) -> GoalSpec:
    """A minimal ``GoalSpec`` for one seam dispatch. The k97 gate reads this
    goal, so the identity/voice references a seam is conditioning on must ride
    in the text it can scan — see ``_live_gen_image``."""
    return GoalSpec(objective=objective[:400] or capability,
                    raw_prompt=prompt[:4000] or objective[:400] or capability,
                    capability=capability, inputs=tuple(inputs), rights=rights)


def _executable_route(goal: GoalSpec, capability: str) -> Any:
    """Resolve a seam call's route. k113a: the per-call selector (matrix +
    reliability ledger + VRAM/quality/latency) proposes ``requested_model``;
    the router keeps the authority gate and the catalog authoritative. A
    selector with no opinion (no evidence, disabled, gap) leaves the router's
    default untouched — and says so on the decision it returns."""
    from hugpy_oracle import router, selection
    g = replace(goal, capability=capability)
    requested, decision = selection.requested_model_for(g, capability)
    try:
        route = router.resolve_route(g, requested)
    except router.RouteRefusal:
        # the selector and the catalog disagreed about eligibility; the catalog wins
        route = router.resolve_route(g)
    if decision is not None:
        try:
            route = replace(route, reasons=tuple(route.reasons) + (
                f"selection: {decision.get('rationale')} -> {decision.get('model_id') or requested}"
                f" (fallback {decision.get('fallback')})",))
        except Exception:  # noqa: BLE001
            pass
    if route.execution != "execute":
        raise SeamUnavailable(
            f"{capability}: route is {route.execution!r}, not executable — "
            + ("; ".join(route.reasons) or "no reason recorded"))
    return route


def _first_artifact(artifacts: Sequence[Mapping[str, Any]],
                    kind: ArtifactKind) -> str | None:
    for art in artifacts:
        if art.get("kind") == kind.value and art.get("uri"):
            return str(art["uri"])
    return None


def _measured_duration(audio_ref: str, fallback: float) -> float:
    """Seconds of audio, READ OFF THE FILE. Invariant 11 / k102 rule 1: a
    timeline built on the synthesizer's own claim is a timeline built on a
    promise. ``fallback`` (the runner's own read-back, carried on the artifact)
    is used only when the file is not readable from HERE — which is itself a
    fact worth not hiding, so it is logged."""
    import wave
    try:
        with wave.open(audio_ref, "rb") as fh:
            rate = fh.getframerate()
            frames = fh.getnframes()
        if rate:
            return frames / float(rate)
    except Exception as exc:  # noqa: BLE001 — unreadable is a fallback, not a fault
        logger.info("performance: could not measure %s here (%s: %s); using the "
                    "runner's recorded duration", audio_ref, type(exc).__name__, exc)
    return float(fallback)


def make_live_synth(rights: Any = None) -> Any:
    """Build the ``synth`` seam: ``(line, voice, seed) -> (audio_ref, duration_s)``.

    Routes ``audio.tts`` through the SAME router + runtime every other live seam
    uses, so the chatterbox seat is reached by the fleet's ordinary dispatch (a
    worker advertising ``task_capabilities['text-to-speech']``) rather than by a
    private path this module owns.

    ON THE REFERENCE VOICE. ``authorized`` on the dispatched body is not "the
    route looked fine" — it is a typed fact restated. A ``VoiceProfile`` of kind
    REFERENCE CANNOT BE CONSTRUCTED without ``authorized=True`` and a
    ``reference_ref`` (audio_master's gate), and ``run_performance`` stage 1
    re-checks every REFERENCE voice in the casting against the run's
    ``RightsManifest`` BEFORE any seam is called. So a reference voice arriving
    here has already passed both. Pass ``rights=`` to this factory and the
    manifest rides on the seam goal as well, so k97's gate re-validates it at
    DISPATCH time too — defence in depth, and the reason this is a factory
    rather than a bare function. With no rights supplied the seam still refuses
    to invent one: it forwards only what the profile already proves.

    DURATION IS MEASURED off the returned wav, never taken from the backend."""
    from hugpy_oracle import runtime

    def _synth(line: Line, voice: VoiceProfile, seed: int) -> tuple[str, float]:
        inputs: list[InputRef] = []
        overrides: dict[str, Any] = {"seed": int(seed)}
        reference = getattr(voice, "reference_ref", None)
        if voice.kind is VoiceKind.REFERENCE and reference:
            inputs.append(InputRef(kind=InputKind.AUDIO, ref=reference,
                                   label=f"authorized voice reference "
                                         f"({voice.voice_id})"))
            overrides["reference_audio"] = reference
            overrides["authorized"] = bool(voice.authorized)
        style = voice.style_dict() if hasattr(voice, "style_dict") else {}
        note = ", ".join(f"{k}: {v}" for k, v in sorted(style.items()))
        if line.emotion:
            note = f"{line.emotion}{', ' + note if note else ''}"
        if note:
            overrides["voice_style"] = note

        goal = _seam_goal(f"speak line {line.line_id} as {line.speaker}",
                          line.text, TTS_CAPABILITY, inputs=tuple(inputs),
                          rights=rights)
        route = _executable_route(goal, TTS_CAPABILITY)
        artifacts, receipt = runtime.execute_route(goal, route,
                                                   overrides=overrides)
        if receipt.failure is not None:
            raise SeamUnavailable(
                f"{TTS_CAPABILITY}: {receipt.failure.value} — "
                + ("; ".join(receipt.log_excerpt) or "no log"))
        uri = _first_artifact(artifacts, ArtifactKind.AUDIO)
        if not uri:
            raise SeamUnavailable(
                f"{TTS_CAPABILITY} produced no audio artifact "
                f"({len(artifacts)} artifact(s))")
        recorded = next((a.get("duration_s") for a in artifacts
                         if a.get("uri") == uri and a.get("duration_s")), 0.0)
        from hugpy_oracle import selection
        selection.remember_producer(uri, TTS_CAPABILITY, receipt.model_id)
        return uri, _measured_duration(uri, recorded or 0.0)

    return _synth


#: The rights-free binding ``default_seams()`` installs. A caller running a
#: REFERENCE-voice performance should bind ``make_live_synth(rights=manifest)``
#: so the dispatch-time gate sees the same manifest stage 1 checked.
_live_synth = make_live_synth()


def _live_transcribe(audio_ref: str) -> tuple[Any, ...]:
    """Round-trip ASR over a produced artifact.

    Routes ``audio.transcribe.word_timestamps`` (eligible on this fleet since
    k98b wired the passthrough — whisper-large-v3-turbo), falling back to plain
    ``audio.transcribe`` if that view ever goes ineligible.

    WHAT COMES BACK TODAY IS BARE WORD STRINGS EITHER WAY, and it is worth
    being exact about why: ``runtime.extract_artifacts`` flattens every
    text-producing capability — transcription included — to ONE inline text
    artifact, so the per-word times the whisper runner does produce do not
    survive the artifact read. This function therefore splits that text and
    hands k102 untimed words, which it counts and WARNS about rather than
    placing on a timeline. Consequence, recorded on every result's
    ``limitations``: line presence is verified, word timing is not, and shot
    boundaries fall on line edges only. The fix is one branch in
    ``runtime.extract_artifacts`` (emit a JSON artifact carrying ``words``
    alongside the text) — this function already reads that shape, so it needs
    no edit when it lands."""
    from hugpy_oracle import runtime

    goal = _seam_goal("transcribe the produced speech",
                      "round-trip transcription of generated audio",
                      TRANSCRIBE_CAPABILITY,
                      inputs=(InputRef(kind=InputKind.AUDIO, ref=audio_ref,
                                       label="produced audio"),))
    try:
        route = _executable_route(goal, TRANSCRIBE_CAPABILITY)
    except SeamUnavailable:
        goal = replace(goal, capability=TRANSCRIBE_FALLBACK_CAPABILITY)
        route = _executable_route(goal, TRANSCRIBE_FALLBACK_CAPABILITY)

    artifacts, receipt = runtime.execute_route(goal, route)
    if receipt.failure is not None:
        raise SeamUnavailable(
            f"{route.capability}: {receipt.failure.value} — "
            + ("; ".join(receipt.log_excerpt) or "no log"))
    words: list[Any] = []
    for art in artifacts:
        payload = art.get("data")
        if isinstance(payload, Mapping) and payload.get("words"):
            words.extend(payload["words"])
            continue
        text = str(art.get("text") or "").strip()
        if text:
            words.extend(text.split())
    return tuple(words)


def _live_gen_image(prompt: str, identity_refs: tuple[str, ...],
                    seed: int) -> str:
    """One keyframe candidate through the existing ``image.generate`` route.

    The identity refs are appended to the goal's RAW PROMPT rather than passed
    as conditioning, because ``image.generate`` on this fleet takes no identity
    pack (``image.identity_reference_pack`` is unregistered). Two consequences,
    both recorded: the k97 gate sees the refs and can refuse; and the keyframe
    is NOT identity-conditioned, which rides on every result's
    ``limitations`` — a named ref that only reaches the text is a description
    of a person, not a lock on one."""
    from hugpy_oracle import runtime

    text = prompt
    if identity_refs:
        text = f"{prompt}\n[identity: {', '.join(identity_refs)}]"
    goal = _seam_goal(prompt, text, KEYFRAME_CAPABILITY)
    route = _executable_route(goal, KEYFRAME_CAPABILITY)
    artifacts, receipt = runtime.execute_route(goal, route,
                                               overrides={"seed": int(seed)})
    if receipt.failure is not None:
        raise SeamUnavailable(
            f"{KEYFRAME_CAPABILITY}: {receipt.failure.value} — "
            + ("; ".join(receipt.log_excerpt) or "no log"))
    uri = _first_artifact(artifacts, ArtifactKind.IMAGE)
    if not uri:
        raise SeamUnavailable(
            f"{KEYFRAME_CAPABILITY} produced no image artifact "
            f"({len(artifacts)} artifact(s))")
    from hugpy_oracle import selection
    selection.remember_producer(uri, KEYFRAME_CAPABILITY, receipt.model_id)
    return uri


def _live_judge_image(image_ref: str, spec: SegmentSpec) -> dict[str, Any]:
    """The vision judge over one keyframe — the same path
    ``runners/movie._score_keyframe`` uses, reached through k90c's evaluator so
    the rubric, the no-think handling and the tolerant verdict parsing are the
    fleet's existing ones rather than a second copy."""
    from hugpy_oracle.evaluation import RUBRICS, run_judge

    goal = _seam_goal(spec.prompt, spec.prompt, KEYFRAME_CAPABILITY)
    result = run_judge(RUBRICS[KEYFRAME_CAPABILITY], goal,
                       [{"kind": ArtifactKind.IMAGE.value, "uri": image_ref}])
    if result is None:
        return {"verdict": None, "score": None,
                "why": "nothing judgeable in the keyframe artifact"}
    return {"judge": result.judge, "verdict": result.verdict,
            "score": result.score, "why": result.rationale}


def _live_registry_version() -> str | None:
    from hugpy_oracle import catalog
    try:
        return catalog.registry_version()
    except Exception as exc:                       # noqa: BLE001
        logger.warning("performance: registry_version unreadable (%s: %s); "
                       "recording None", type(exc).__name__, exc)
        return None


def _live_catalog_view() -> dict[str, Any]:
    from hugpy_oracle import catalog
    return {v.name: v for v in catalog.list_capabilities()}


def _ffmpeg() -> str:
    from hugpy_platform.binaries import resolve_bin
    return resolve_bin("ffmpeg") or "ffmpeg"


def _live_concat(clip_refs: tuple[str, ...], master: AudioMaster) -> str:
    """Assembly cut (doc Stage 18 step 1) — concat the accepted takes and lay
    the locked audio under them.

    Reuses ``runners.movie._concat_movie`` for the concat-demux (the clips are
    encoded by the same studio path, so ``-c copy`` is valid there) and then
    places each line's wav at its LOCKED start with ``adelay`` + ``amix``,
    which is the audio-first rule made literal: the picture is cut to the
    timeline, the timeline is never retimed to the picture.

    What it deliberately does NOT do: bounded retiming, dialogue alignment,
    ambience/foley/music, colour finishing. Those are doc Stage 18 steps 2-4
    and belong to k108; the result's ``limitations`` says so rather than the
    output implying it happened."""
    if not clip_refs:
        raise SeamUnavailable("assembly: no accepted clips to concatenate")
    missing = [c for c in clip_refs if not os.path.isfile(c)]
    if missing:
        raise SeamUnavailable(f"assembly: clip file(s) missing: {missing}")

    from hugpy_video.intel.runners.movie import _concat_movie

    work_dir = os.path.join(default_run_root(), "assembly",
                            master.digest[:16])
    os.makedirs(work_dir, exist_ok=True)
    silent = os.path.join(work_dir, "picture.mp4")
    try:
        _concat_movie(list(clip_refs), silent, work_dir)
    except Exception as exc:                       # noqa: BLE001
        raise SeamUnavailable(f"assembly: ffmpeg concat failed: {exc}") from exc

    tracks = [(master.timing(line_id).start_s, ref)
              for line_id, ref in master.tracks]
    absent = [ref for _s, ref in tracks if not os.path.isfile(ref)]
    if absent:
        raise SeamUnavailable(f"assembly: audio track file(s) missing: {absent}")

    out = os.path.join(work_dir, "performance.mp4")
    cmd = [_ffmpeg(), "-y", "-i", silent]
    for _start, ref in tracks:
        cmd += ["-i", ref]
    filters = []
    for index, (start, _ref) in enumerate(tracks, start=1):
        delay_ms = int(round(max(0.0, start) * 1000))
        filters.append(f"[{index}:a]adelay={delay_ms}|{delay_ms}[a{index}]")
    mix_inputs = "".join(f"[a{i}]" for i in range(1, len(tracks) + 1))
    filters.append(f"{mix_inputs}amix=inputs={len(tracks)}:normalize=0[aout]")
    cmd += ["-filter_complex", ";".join(filters),
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", out]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.isfile(out):
        raise SeamUnavailable(
            f"assembly: ffmpeg mux failed rc={result.returncode}: "
            f"{(result.stderr or '')[-500:]}")
    return out


#: The gaps ``default_seams()`` records today, each naming the operator step.
#: Editing this table is how a newly-seated capability stops being a gap.
#: ``synth`` is NOT in this table any more: chatterbox was seated on
#: a-brain-Super-Server (2026-08-21) — env-PROFILE venv ``chatterbox-tts``,
#: heartbeat ``task_capabilities['text-to-speech'] = true``, ``audio.tts``
#: ELIGIBLE — so a gap for it would be a gap that is not a gap, which
#: ``PerformanceSeams`` refuses at construction. It is bound by
#: ``make_live_synth`` above.
LIVE_SEAM_GAPS: tuple[SeamGap, ...] = (
    SeamGap(
        seam="similarity", capability=SIMILARITY_CAPABILITY,
        requirement=(
            "no speaker-embedding backend is registered on this fleet, so "
            "voice similarity is UNSCORED rather than passed. Operator: "
            "register a speaker-embedding model for audio.speaker_similarity; "
            "until then keep SpeechPolicy.require_similarity=False, or every "
            "line gaps with CAPABILITY_GAP (which is honest, not useful).")),
    SeamGap(
        seam="gen_clip", capability=CLIP_CAPABILITY,
        requirement=(
            "video.* is DEFERRED by the oracle route on purpose (k90b): clips "
            "execute through the studio job pipeline, not POST /oracle/route, "
            "and central has no GPU. Operator: seat i2v on a GPU worker and "
            "bind gen_clip to the studio produce_clip spine "
            "(video_intel/runners/studio_i2v.py) — the seam signature is "
            "(keyframe_ref, SegmentSpec) -> (clip_ref, duration_s).")),
    SeamGap(
        seam="judge_clip", capability=CLIP_CAPABILITY,
        requirement=(
            "there is no clip-level evaluator on this fleet: the vision judge "
            "grades STILLS (image.understand). Operator: bind judge_clip to a "
            "frame-sampling wrapper over image.understand, or wait for the "
            "temporal/lip-sync evaluators (k119/k121). Leaving it unbound is "
            "why stage 7 refuses instead of accepting unjudged video.")),
)


#: Seams ``default_seams()`` BINDS live on this fleet. A hand-built seam set
#: that leaves one of these unbound is NOT facing a fleet gap — it unbound
#: something that works — and ``gap_for`` must say that instead of the generic
#: "no implementation is wired", which would be false. This table is the third
#: source it consults, after the seam set's own record and the standing gap
#: table.
LIVE_SEAM_BINDINGS: dict[str, str] = {
    "synth": (
        "this seam set left PerformanceSeams.synth unbound, but audio.tts IS "
        "seated on this fleet: chatterbox on a GPU worker advertising "
        "task_capabilities['text-to-speech'] (a-brain-Super-Server, seated "
        "2026-08-21, env-profile venv 'chatterbox-tts'). default_seams() binds "
        "it through make_live_synth(); bind it here too — and pass "
        "make_live_synth(rights=<manifest>) when the casting includes a "
        "REFERENCE voice, so k97's gate re-validates at dispatch."),
    "transcribe": (
        "this seam set left PerformanceSeams.transcribe unbound; "
        "audio.transcribe.word_timestamps is eligible on this fleet and "
        "default_seams() binds it through _live_transcribe."),
    "gen_image": (
        "this seam set left PerformanceSeams.gen_image unbound; image.generate "
        "is eligible and default_seams() binds it through _live_gen_image."),
    "judge_image": (
        "this seam set left PerformanceSeams.judge_image unbound; "
        "image.understand is eligible and default_seams() binds it through "
        "_live_judge_image (k90c's evaluator)."),
    "concat": (
        "this seam set left PerformanceSeams.concat unbound; default_seams() "
        "binds the ffmpeg assembly cut through _live_concat."),
}


def default_seams(**overrides: Any) -> PerformanceSeams:
    """The seams this fleet can honour TODAY, plus a typed gap for each it
    cannot.

    BOUND LIVE (2026-08-21): ``synth`` (audio.tts -> the chatterbox seat on
    a-brain-Super-Server, through the same router + runtime; duration MEASURED
    off the wav), ``transcribe`` (audio.transcribe.word_timestamps,
    falling back to audio.transcribe — see ``_live_transcribe``), ``gen_image``
    (image.generate through the existing router + runtime), ``judge_image``
    (image.understand through k90c's evaluator — the same path
    ``runners/movie`` judges keyframes with), ``concat`` (ffmpeg concat-demux
    via ``runners.movie._concat_movie`` plus an adelay/amix audio bed),
    ``registry_version`` and ``catalog_view`` (the live k101 catalog).

    UNBOUND, each with an operator step (:data:`LIVE_SEAM_GAPS`):
    ``similarity`` (no speaker-embedding backend), ``gen_clip`` and
    ``judge_clip`` (video executes through the studio job pipeline on a GPU
    worker, and there is no clip evaluator).

    An override BINDS its seam and therefore removes its recorded gap — that is
    the whole reason ``unbound`` is data rather than a constant."""
    unknown = sorted(set(overrides) - {f.name for f in
                                       PerformanceSeams.__dataclass_fields__.values()})
    if unknown:
        raise PerformanceError(
            f"default_seams() got unknown seam(s) {unknown}; the seam set is "
            f"{list(PerformanceSeams.__dataclass_fields__)}")

    bound: dict[str, Any] = {
        "synth": _live_synth,
        "transcribe": _live_transcribe,
        "similarity": None,
        "gen_image": _live_gen_image,
        "judge_image": _live_judge_image,
        "gen_clip": None,
        "judge_clip": None,
        "concat": _live_concat,
        "registry_version": _live_registry_version,
        "catalog_view": _live_catalog_view,
    }
    bound.update(overrides)
    if "unbound" not in overrides:
        bound["unbound"] = tuple(g for g in LIVE_SEAM_GAPS
                                 if bound.get(g.seam) is None)
    return PerformanceSeams(**bound)


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerformanceGoal:
    """One "two characters, three lines" request, fully typed.

    A ``GoalSpec`` alone cannot express this: it carries the operator's words,
    the budget and the rights manifest, but a performance also needs the LOCKED
    dialogue, the casting, and the direction knobs the compiler reads. Those
    live here, next to the goal rather than inside it, because ``GoalSpec`` is
    the shared oracle currency and a video recipe must not colonise it.

    ``dialogue`` is LOCKED at construction (``.lock()``): every downstream
    digest is taken against a locked timeline, so a goal that carried a draft
    one would produce artifacts nobody can re-derive."""

    goal: GoalSpec
    dialogue: DialogueTimeline
    casting: tuple[tuple[str, VoiceProfile], ...]
    raw_request_ref: str
    identity_refs: tuple[str, ...] = ()
    voice_refs: tuple[str, ...] = ()
    deliverable: str = ""
    prompts_before_run: tuple[str, ...] = ()
    operator_refs: tuple[str, ...] = ()
    acquisition_refs: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    continuity: ContinuityBible | None = None
    speech_policy: SpeechPolicy = field(default_factory=SpeechPolicy)
    tone: float = 0.0
    rubric: tuple[str, ...] = DEFAULT_RUBRIC
    camera: Mapping[str, Any] | None = None
    blocking: str | None = None
    lighting: str | None = None
    negative_prompt: str | None = None
    min_shot_s: float = 1.0
    max_shot_s: float = 8.0
    pad_s: float = 0.0
    segment_prefix: str = "s"
    seed_salt: int = 0
    prompt_writer: Any = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.goal, GoalSpec):
            raise PerformanceError(
                f"PerformanceGoal.goal takes a GoalSpec, got "
                f"{type(self.goal).__name__}")
        if not isinstance(self.dialogue, DialogueTimeline):
            raise PerformanceError(
                f"PerformanceGoal.dialogue takes a DialogueTimeline, got "
                f"{type(self.dialogue).__name__}")
        object.__setattr__(self, "dialogue", self.dialogue.lock())
        normalized = _normalize_casting(self.casting)
        for speaker, profile in normalized:
            if not isinstance(profile, VoiceProfile):
                raise PerformanceError(
                    f"casting for {speaker!r} must be a VoiceProfile, got "
                    f"{type(profile).__name__}")
        object.__setattr__(self, "casting", normalized)
        cast = {s for s, _v in normalized}
        uncast = [s for s in self.dialogue.speakers if s not in cast]
        if uncast:
            raise PerformanceError(
                f"speaker(s) {uncast} have no voice cast (casting has "
                f"{sorted(cast)}) — a line nobody can speak is not a "
                f"performance request")
        if not str(self.raw_request_ref).strip():
            raise PerformanceError(
                "PerformanceGoal.raw_request_ref must be non-empty (invariant "
                "1: the operator's raw request rides along by reference)")
        for name in ("identity_refs", "voice_refs", "prompts_before_run",
                     "operator_refs", "acquisition_refs", "exclusions",
                     "rubric"):
            object.__setattr__(self, name,
                               tuple(str(v) for v in getattr(self, name) or ()))
        if not self.rubric:
            raise PerformanceError(
                "PerformanceGoal.rubric is empty; Stage 9 gives every shot an "
                "acceptance rubric and an unjudgeable shot cannot be accepted "
                "(invariant 11)")
        if not 0.0 <= float(self.tone) <= 1.0:
            raise PerformanceError(
                f"PerformanceGoal.tone must be in [0, 1], got {self.tone}")
        object.__setattr__(self, "tone", float(self.tone))
        if not isinstance(self.speech_policy, SpeechPolicy):
            raise PerformanceError(
                f"PerformanceGoal.speech_policy takes a SpeechPolicy, got "
                f"{type(self.speech_policy).__name__}")
        if self.continuity is not None and \
                not isinstance(self.continuity, ContinuityBible):
            raise PerformanceError(
                f"PerformanceGoal.continuity takes a ContinuityBible, got "
                f"{type(self.continuity).__name__}")

    # -- reading -----------------------------------------------------------

    @property
    def voices(self) -> dict[str, VoiceProfile]:
        """The casting table ``build_audio_master`` takes."""
        return {speaker: profile for speaker, profile in self.casting}

    @property
    def lines(self) -> tuple[Line, ...]:
        return self.dialogue.lines

    @property
    def dialogue_map(self) -> dict[str, str]:
        """``line_id -> text`` — what k104's compiler needs so a locked brief
        carries the WORDS, not just the window."""
        return {ln.line_id: ln.text for ln in self.dialogue.lines}

    @property
    def effective_deliverable(self) -> str:
        return (self.deliverable or self.goal.objective).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.to_dict(),
            "dialogue": self.dialogue.to_dict(),
            "casting": [[speaker, profile.to_dict()]
                        for speaker, profile in self.casting],
            "raw_request_ref": self.raw_request_ref,
            "identity_refs": list(self.identity_refs),
            "voice_refs": list(self.voice_refs),
            "deliverable": self.effective_deliverable,
            "prompts_before_run": list(self.prompts_before_run),
            "operator_refs": list(self.operator_refs),
            "acquisition_refs": list(self.acquisition_refs),
            "exclusions": list(self.exclusions),
            "continuity": self.continuity.to_dict() if self.continuity else None,
            "speech_policy": self.speech_policy.to_dict(),
            "tone": self.tone,
            "rubric": list(self.rubric),
            "camera": dict(self.camera or {}),
            "blocking": self.blocking,
            "lighting": self.lighting,
            "negative_prompt": self.negative_prompt,
            "min_shot_s": self.min_shot_s,
            "max_shot_s": self.max_shot_s,
            "pad_s": self.pad_s,
            "segment_prefix": self.segment_prefix,
            "seed_salt": self.seed_salt,
            "created_at": self.created_at,
        }

    @property
    def digest(self) -> str:
        """Content address of the REQUEST. A resumed run whose goal digest
        differs from the journal's is a different request wearing the same run
        id, and the journal is not trusted for it."""
        return digest_payload(self.to_dict())


def _normalize_casting(casting: Any) -> tuple[tuple[str, VoiceProfile], ...]:
    """``casting`` -> sorted ``((speaker, VoiceProfile), …)``.

    Accepts the three shapes a caller actually has: a ``{speaker: profile}``
    mapping, a sequence of ``(speaker, profile)`` pairs (what this dataclass
    stores, so a round trip is free), or a bare iterable of profiles keyed by
    their own ``voice_id`` — the same three ``audio_master._casting_table``
    accepts, so the two never disagree about who speaks what. Sorted because a
    frozen dataclass's digest must not depend on dict insertion order."""
    if isinstance(casting, Mapping):
        pairs = [(str(k), v) for k, v in casting.items()]
    else:
        pairs = []
        for item in casting or ():
            if isinstance(item, VoiceProfile):
                pairs.append((item.voice_id, item))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append((str(item[0]), item[1]))
            else:
                raise PerformanceError(
                    f"PerformanceGoal.casting takes a mapping, "
                    f"(speaker, VoiceProfile) pairs, or VoiceProfiles; got "
                    f"{item!r}")
    return tuple(sorted(pairs, key=lambda kv: kv[0]))


@dataclass(frozen=True, slots=True)
class PerformanceBudget:
    """What the run is allowed to spend.

    ``max_seconds`` is checked BETWEEN stages (and between segments), the way
    ``runners/movie`` owns its own wall clock: the bus has no reaper and a
    deadline nobody enforces is a comment. ``keyframe_repair_rounds`` and
    ``clip_repair_rounds`` are capped at 1 by construction — "bounded, one
    round" is the doc's rule, so a 2 here would not be a tuning choice, it
    would be a different design."""

    max_seconds: float | None = None
    keyframe_repair_rounds: int = 1
    clip_repair_rounds: int = 1

    def __post_init__(self) -> None:
        if self.max_seconds is not None and float(self.max_seconds) <= 0:
            raise PerformanceError(
                f"PerformanceBudget.max_seconds must be positive when set, got "
                f"{self.max_seconds}")
        for name in ("keyframe_repair_rounds", "clip_repair_rounds"):
            value = getattr(self, name)
            if value not in (0, 1):
                raise PerformanceError(
                    f"PerformanceBudget.{name} must be 0 or 1: the doc's repair "
                    f"rule is ONE bounded round, and a loop is the thing this "
                    f"orchestrator exists to not be (got {value!r})")

    def to_dict(self) -> dict[str, Any]:
        return {"max_seconds": self.max_seconds,
                "keyframe_repair_rounds": self.keyframe_repair_rounds,
                "clip_repair_rounds": self.clip_repair_rounds}


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One stage's line in the run's own journal.

    ``digests`` is the point: every stage says what it PRODUCED by content
    address, so a reader can tie the final video back through the clips, the
    keyframes, the specs, the lock, the master and the snapshot without holding
    any of them."""

    stage: str
    index: int
    ok: bool
    resumed: bool = False
    digests: tuple[tuple[str, str], ...] = ()
    artifact_refs: tuple[str, ...] = ()
    detail: str = ""
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise PerformanceError(
                f"StageRecord.stage must be one of {list(STAGES)}, got "
                f"{self.stage!r}")
        object.__setattr__(self, "digests",
                           tuple((str(k), str(v)) for k, v in self.digests))
        object.__setattr__(self, "artifact_refs",
                           tuple(str(r) for r in self.artifact_refs))

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "index": self.index, "ok": self.ok,
                "resumed": self.resumed,
                "digests": {k: v for k, v in self.digests},
                "artifact_refs": list(self.artifact_refs),
                "detail": self.detail, "duration_s": self.duration_s}


@dataclass(frozen=True, slots=True)
class PerformanceGap:
    """Why the run stopped, named by STAGE and by REPAIR CODE.

    ``capability`` and ``requirement`` are filled when the stop is a capability
    gap, so the answer to "what do I do about it" is in the object rather than
    in a log. ``evidence`` carries the scorecards / validation errors behind
    the codes."""

    stage: str
    diagnosis: str
    repair_codes: tuple[RepairCode, ...] = ()
    capability: str | None = None
    requirement: str = ""
    segment_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise PerformanceError(
                f"PerformanceGap.stage must be one of {list(STAGES)}, got "
                f"{self.stage!r}")
        if not str(self.diagnosis).strip():
            raise PerformanceError(
                "PerformanceGap.diagnosis must be non-empty — a stop with no "
                "explanation is the failure mode this whole module exists to "
                "avoid")
        object.__setattr__(self, "repair_codes", tuple(self.repair_codes))
        object.__setattr__(self, "segment_ids", tuple(self.segment_ids))
        object.__setattr__(self, "evidence", tuple(str(e) for e in self.evidence))

    @property
    def primary_code(self) -> RepairCode | None:
        return self.repair_codes[0] if self.repair_codes else None

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "diagnosis": self.diagnosis,
                "repair_codes": [c.value for c in self.repair_codes],
                "primary_code": (self.primary_code.value
                                 if self.primary_code else None),
                "capability": self.capability,
                "requirement": self.requirement,
                "segment_ids": list(self.segment_ids),
                "evidence": list(self.evidence)}


@dataclass(frozen=True, slots=True)
class ShotResult:
    """One shot's whole story: which keyframe, which clip, how many candidates
    each cost, whether a bounded repair happened, and the card that decided it.

    ``keyframe_candidates``/``clip_candidates`` are what "3 candidates per shot"
    actually cost on this run, per Stage 16's "generate several short candidate
    clips per shot rather than committing immediately to one long result"."""

    segment_id: str
    index: int
    accepted: bool
    keyframe_ref: str | None = None
    keyframe_seed: int | None = None
    keyframe_candidates: int = 0
    keyframe_repaired: bool = False
    keyframe_scorecard: Scorecard | None = None
    clip_ref: str | None = None
    clip_seconds: float | None = None
    clip_candidates: int = 0
    clip_repaired: bool = False
    scorecard: Scorecard | None = None
    repair_codes: tuple[RepairCode, ...] = ()
    diagnosis: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "repair_codes", tuple(self.repair_codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id, "index": self.index,
            "accepted": self.accepted,
            "keyframe_ref": self.keyframe_ref,
            "keyframe_seed": self.keyframe_seed,
            "keyframe_candidates": self.keyframe_candidates,
            "keyframe_repaired": self.keyframe_repaired,
            "keyframe_scorecard": (self.keyframe_scorecard.to_dict()
                                   if self.keyframe_scorecard else None),
            "clip_ref": self.clip_ref, "clip_seconds": self.clip_seconds,
            "clip_candidates": self.clip_candidates,
            "clip_repaired": self.clip_repaired,
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
            "repair_codes": [c.value for c in self.repair_codes],
            "diagnosis": self.diagnosis,
        }


@dataclass(frozen=True, slots=True)
class PerformanceResult:
    """The whole run, as one readable answer.

    ``ok`` is TRUE only when an assembled deliverable exists AND its
    whole-result scorecard hard-passes. A stopped run (``stop_after``) is not
    ok and carries no gap — it did not fail, it did not finish."""

    run_id: str
    ok: bool
    goal_digest: str
    stages: tuple[StageRecord, ...] = ()
    gap: PerformanceGap | None = None
    stopped_after: str | None = None
    registry_version: str | None = None
    snapshot: GenerationSnapshot | None = None
    audio_master: AudioMaster | None = None
    lock: ProductionLock | None = None
    segments: tuple[SegmentSpec, ...] = ()
    plan_graph: PlanGraph | None = None
    validation: ValidationReport | None = None
    shots: tuple[ShotResult, ...] = ()
    video_ref: str | None = None
    scorecard: Scorecard | None = None
    receipts: tuple[ExecutionReceipt, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    state_path: str = ""

    def __post_init__(self) -> None:
        for name in ("stages", "segments", "shots", "receipts",
                     "artifact_refs", "limitations", "warnings"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.ok and self.gap is not None:
            raise PerformanceError(
                f"an ok PerformanceResult cannot carry a gap "
                f"({self.gap.stage}: {self.gap.diagnosis})")
        if self.ok and not self.video_ref:
            raise PerformanceError(
                "an ok PerformanceResult must carry a video_ref — 'ok' means a "
                "deliverable exists, never 'nothing went wrong'")

    # -- reading -----------------------------------------------------------

    @property
    def lock_digest(self) -> str | None:
        return self.lock.digest if self.lock is not None else None

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(s.stage for s in self.stages)

    def stage(self, name: str) -> StageRecord | None:
        for record in self.stages:
            if record.stage == name:
                return record
        return None

    def ran(self, name: str) -> bool:
        """Did ``name`` EXECUTE on this run (as opposed to being resumed or
        never reached)? The "later stages untouched" assertion, as a method."""
        record = self.stage(name)
        return record is not None and not record.resumed

    @property
    def digests(self) -> dict[str, str]:
        """Every artifact digest this run journalled, flattened."""
        out: dict[str, str] = {}
        for record in self.stages:
            for key, value in record.digests:
                out[f"{record.stage}.{key}"] = value
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "ok": self.ok,
            "goal_digest": self.goal_digest,
            "stages": [s.to_dict() for s in self.stages],
            "gap": self.gap.to_dict() if self.gap else None,
            "stopped_after": self.stopped_after,
            "registry_version": self.registry_version,
            "snapshot_digest": (self.snapshot.digest if self.snapshot else None),
            "audio_master_digest": (self.audio_master.digest
                                    if self.audio_master else None),
            "lock_digest": self.lock_digest,
            "segment_digests": [s.digest for s in self.segments],
            "validation": (self.validation.to_dict()
                           if self.validation else None),
            "shots": [s.to_dict() for s in self.shots],
            "video_ref": self.video_ref,
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
            "receipts": [r.to_dict() for r in self.receipts],
            "artifact_refs": list(self.artifact_refs),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "state_path": self.state_path,
        }


# ---------------------------------------------------------------------------
# The run journal (resume)
# ---------------------------------------------------------------------------


def default_run_root() -> str:
    """Where performance runs write, mirroring where movie runs write.

    ``runners/movie`` puts its work dir at
    ``<DEFAULT_ROOT>/video_intel/movies/<job_id>``; this puts the run journal at
    ``<DEFAULT_ROOT>/video_intel/runs/performance/<run_id>/state.json``. The env
    override exists so a test (and an operator with a full disk) can point it
    somewhere else without monkeypatching a constant."""
    from hugpy_oracle.config import performance_run_root
    return performance_run_root()


def run_dir(run_id: str, root: str | None = None) -> str:
    return os.path.join(root or default_run_root(), "runs", "performance",
                        str(run_id))


def state_path(run_id: str, root: str | None = None) -> str:
    return os.path.join(run_dir(run_id, root), STATE_FILENAME)


def derive_run_id(pgoal: PerformanceGoal) -> str:
    """A deterministic run id from the REQUEST. Two identical requests share a
    run dir, which is what makes ``resume`` useful without an id registry; a
    changed request gets a different id and never silently reuses the other's
    artifacts."""
    return f"perf-{pgoal.digest[:16]}"


class RunState:
    """The stage journal: ``{stage: {digest, payload, payload_digest, refs}}``.

    Deliberately dumb. It is not a database and it is not the source of truth
    about the fleet — it is a note this run leaves for its own resume, and every
    value it hands back is re-verified by digest before anything is skipped."""

    def __init__(self, run_id: str, goal_digest_: str, path: str,
                 stages: dict[str, Any] | None = None,
                 version: int = STATE_VERSION) -> None:
        self.run_id = str(run_id)
        self.goal_digest = str(goal_digest_)
        self.path = str(path)
        self.stages: dict[str, Any] = dict(stages or {})
        self.version = int(version)
        self.write_errors: list[str] = []

    # -- io ---------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "RunState | None":
        """The journal at ``path``, or None when there is none / it is not
        readable / it is not this version. Never raises: an unreadable journal
        means "no resume", which is a correct answer."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.debug("performance: no readable run state at %s (%s)",
                         path, exc)
            return None
        if not isinstance(payload, Mapping):
            return None
        if int(payload.get("version", 0)) != STATE_VERSION:
            logger.info("performance: run state at %s is version %s, not %s — "
                        "not resuming", path, payload.get("version"),
                        STATE_VERSION)
            return None
        return cls(run_id=payload.get("run_id", ""),
                   goal_digest_=payload.get("goal_digest", ""),
                   path=path, stages=dict(payload.get("stages") or {}),
                   version=STATE_VERSION)

    def flush(self) -> None:
        """Atomically persist. BEST EFFORT — a journal write failure records a
        warning and never fails the run: resumability is an optimisation, and
        losing it must not lose the work it was protecting."""
        payload = {"version": self.version, "run_id": self.run_id,
                   "goal_digest": self.goal_digest, "stages": self.stages,
                   "updated_at": _utc_now()}
        tmp = f"{self.path}.tmp"
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError as exc:
            note = (f"run state not persisted to {self.path} "
                    f"({type(exc).__name__}: {exc}); this run is not resumable")
            if note not in self.write_errors:
                self.write_errors.append(note)
            logger.warning("performance: %s", note)

    # -- stage entries ----------------------------------------------------

    def record(self, stage: str, *, digest: str, payload: Any,
               refs: Sequence[str] = ()) -> None:
        if stage not in STAGES:
            raise PerformanceError(f"unknown stage {stage!r}")
        self.stages[stage] = {
            "digest": digest,
            "payload": payload,
            "payload_digest": digest_payload(payload),
            "refs": [str(r) for r in refs],
            "at": _utc_now(),
        }
        self.flush()

    def entry(self, stage: str) -> dict[str, Any] | None:
        entry = self.stages.get(stage)
        return dict(entry) if isinstance(entry, Mapping) else None

    def intact(self, stage: str) -> bool:
        """Is this entry internally consistent and are its file-backed refs
        still on disk? The cheap half of "verify by digest"; the stage-specific
        half (re-derive the artifact and compare ITS digest) lives in the
        resume helpers, because only they know the artifact type."""
        entry = self.entry(stage)
        if entry is None:
            return False
        if digest_payload(entry.get("payload")) != entry.get("payload_digest"):
            logger.info("performance: run state stage %r payload digest "
                        "mismatch — re-running it", stage)
            return False
        for ref in entry.get("refs") or ():
            if not _ref_present(ref):
                logger.info("performance: run state stage %r references %r "
                            "which is gone — re-running it", stage, ref)
                return False
        return True

    def drop_from(self, stage: str) -> None:
        """Forget ``stage`` and everything after it. Called the moment a stage
        is re-executed, so a later entry can never be resumed on top of an
        input that changed underneath it."""
        index = STAGES.index(stage)
        for name in STAGES[index:]:
            self.stages.pop(name, None)


def _ref_present(ref: str) -> bool:
    """Is an artifact reference still real? An absolute path must still be a
    file; anything else (an opaque handle, a url, a test fixture's
    ``audio://l1#0``) is taken at face value, because this module cannot verify
    a namespace it does not own and must not pretend to."""
    text = str(ref)
    if os.path.isabs(text):
        return os.path.isfile(text)
    return True


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    """One judge's answer, normalized out of whatever the seam returned.

    ``scored`` is the honesty bit: False means nobody actually graded this, and
    the caller must lower confidence and say so rather than treat silence as
    approval."""

    passed: bool
    scored: bool
    score: float | None = None
    judge: str = ""
    why: str = ""
    codes: tuple[RepairCode, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "scored": self.scored,
                "score": self.score, "judge": self.judge, "why": self.why,
                "codes": [c.value for c in self.codes]}


_CODE_ALIASES: dict[str, RepairCode] = {c.value: c for c in RepairCode}


def _codes_from(raw: Any) -> tuple[RepairCode, ...]:
    out: list[RepairCode] = []
    values: list[Any] = []
    if isinstance(raw, Mapping):
        single = raw.get("repair_code")
        if single is not None:
            values.append(single)
        values.extend(raw.get("repair_codes") or ())
    for value in values:
        code = value if isinstance(value, RepairCode) else \
            _CODE_ALIASES.get(str(value).strip().lower())
        if code is not None and code not in out:
            out.append(code)
    return tuple(out)


def coerce_verdict(raw: Any, *, threshold: float,
                   default_code: RepairCode,
                   judge: str = "") -> Verdict:
    """Whatever a judge seam returned -> a :class:`Verdict`.

    Tolerant in exactly the way ``movie.parse_vision_verdict`` and
    ``evaluation.parse_judge_verdict`` are tolerant, and honest in exactly the
    way they are honest:

    * ``None`` / an empty mapping / ``verdict`` in (None, "unscored",
      "unavailable") with no score -> UNSCORED, and unscored KEEPS the take
      (the fleet's vision plane being down must not fail a shot) while marking
      ``scored=False`` so the card's confidence drops and ``limitations`` says
      a judge was missing.
    * an explicit ``passed`` / ``ok`` boolean wins outright.
    * otherwise a numeric ``score`` under ``threshold`` fails, and a
      ``verdict`` of NO fails.
    * repair codes come from the judge when it names them, else
      ``default_code``; ``identity``/``identity_ok`` False adds IDENTITY_DRIFT,
      because Stage 15 routes that failure to the keyframe and nowhere else."""
    if raw is None:
        return Verdict(passed=True, scored=False, judge=judge,
                       why="no judge verdict (unscored)")
    if isinstance(raw, bool):
        return Verdict(passed=raw, scored=True, judge=judge,
                       why="boolean verdict",
                       codes=() if raw else (default_code,))
    if isinstance(raw, Verdict):
        return raw
    if not isinstance(raw, Mapping):
        raw = {"verdict": str(raw)}

    judge_name = str(raw.get("judge") or judge or "")
    why = str(raw.get("why") or raw.get("rationale") or raw.get("detail") or "")
    score = raw.get("score")
    score = None if score is None else float(score)
    verdict = raw.get("verdict")
    verdict = None if verdict is None else str(verdict).strip().upper()
    explicit = raw.get("passed", raw.get("ok", raw.get("hard_pass")))

    identity_flag = raw.get("identity", raw.get("identity_ok"))
    identity_bad = identity_flag is False or (
        isinstance(identity_flag, (int, float))
        and not isinstance(identity_flag, bool)
        and float(identity_flag) < threshold / 100.0)

    codes = _codes_from(raw)
    if identity_bad and RepairCode.IDENTITY_DRIFT not in codes:
        codes = (RepairCode.IDENTITY_DRIFT,) + codes

    if explicit is not None:
        passed = bool(explicit)
        return Verdict(passed=passed, scored=True, score=score,
                       judge=judge_name, why=why,
                       codes=() if passed else (codes or (default_code,)))

    if verdict in (None, "", "UNSCORED", "UNAVAILABLE") and score is None \
            and not codes and not identity_bad:
        return Verdict(passed=True, scored=False, score=None, judge=judge_name,
                       why=why or "no judge verdict (unscored)")

    passed = True
    if verdict == "NO":
        passed = False
    if score is not None and score < float(threshold):
        passed = False
    if identity_bad or codes:
        passed = False
    return Verdict(passed=passed, scored=True, score=score, judge=judge_name,
                   why=why, codes=() if passed else (codes or (default_code,)))


def _judge_check(name: str, kind: CheckKind, verdict: Verdict,
                 threshold: float) -> Check:
    if not verdict.scored:
        return Check(name=name, kind=kind, value=None, threshold=None,
                     passed=True,
                     detail=(speech.UNSCORED_PREFIX + (verdict.why or
                             "no judge was reachable for this artifact")))
    return Check(name=name, kind=kind,
                 value=verdict.score if verdict.score is not None
                 else ("pass" if verdict.passed else "fail"),
                 threshold=threshold if verdict.score is not None else "pass",
                 passed=verdict.passed,
                 detail=verdict.why or ("judge accepted" if verdict.passed
                                        else "judge rejected"))


def _card(checks: Sequence[Check], *, judge_results: Sequence[JudgeResult] = (),
          codes: Sequence[RepairCode] = (),
          recommended: str | None = None) -> Scorecard:
    """Fold checks into a ``Scorecard`` with the tree's existing discipline:
    ``hard_pass`` is the conjunction, ``confidence`` is the SCORED fraction
    (``speech.speech_scorecard``'s rule, so an unscored judge shows up as a
    number rather than as nothing), and a ``repair_code`` is set only on a
    failing card."""
    checks = tuple(checks)
    hard_pass = all(c.passed for c in checks)
    scored = [c for c in checks if not speech.is_unscored(c)]
    confidence = round(len(scored) / len(checks), 3) if checks else 1.0
    diagnoses = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    unscored = [c.name for c in checks if speech.is_unscored(c)]
    if unscored:
        diagnoses.append("unscored (no evidence): " + ", ".join(unscored))
    code = None if hard_pass else (tuple(codes)[0] if codes else None)
    return Scorecard(
        hard_pass=hard_pass, checks=checks,
        judge_results=tuple(judge_results), confidence=confidence,
        diagnosis="; ".join(diagnoses) or None,
        repair_code=code,
        recommended_repair=None if hard_pass else recommended)


def keyframe_scorecard(spec: SegmentSpec, verdict: Verdict, *,
                       image_ref: str | None,
                       threshold: float) -> Scorecard:
    """Stage 15 — the card that decides whether a still may authorize video.

    Two checks, both outside the generator: the artifact exists (technical) and
    the judge accepted it (identity/intent). Composition, geometry, costume,
    props and setting are Stage 15's other axes and are NOT checked here — they
    need evaluators this fleet does not have, and inventing a passing check for
    them would be worse than the missing check, which at least shows up in
    ``limitations``."""
    checks = [
        Check(name="keyframe.produced", kind=CheckKind.TECHNICAL,
              value=bool(image_ref), threshold=True, passed=bool(image_ref),
              detail=(f"keyframe artifact {image_ref}" if image_ref
                      else "the image seam produced no artifact")),
        _judge_check(f"keyframe.judge.{spec.segment_id}", CheckKind.IDENTITY,
                     verdict, threshold),
    ]
    codes = verdict.codes if not verdict.passed else ()
    if not image_ref:
        codes = (RepairCode.EMPTY_OUTPUT,) + tuple(codes)
    judge_results = ()
    if verdict.scored and verdict.judge:
        judge_results = (JudgeResult(
            judge=verdict.judge,
            verdict="YES" if verdict.passed else "NO",
            score=verdict.score, rationale=verdict.why),)
    return _card(checks, judge_results=judge_results, codes=codes,
                 recommended=("regenerate THIS KEYFRAME only (Stage 15: an "
                              "identity failure repairs the identity pack, "
                              "spatial conditioning or keyframe node — never "
                              "the transcription or the audio)"))


def clip_scorecard(spec: SegmentSpec, verdict: Verdict, *,
                   clip_ref: str | None, clip_seconds: float | None,
                   threshold: float,
                   duration_tolerance: float) -> Scorecard:
    """Stage 17 — the per-shot card.

    Evidence: the artifact exists; the judge accepted it; and k98's
    ``check_duration_fit`` compares the segment's LOCKED audio window against
    the rendered clip, in Stage 8's direction (audio is authoritative, a clip
    that is shorter than its audio is ``SHOT_TOO_SHORT``, never "speak
    faster")."""
    duration_check = speech.check_duration_fit(
        audio_seconds=spec.duration_s, shot_seconds=clip_seconds,
        tolerance=duration_tolerance)
    checks = [
        Check(name="clip.produced", kind=CheckKind.TECHNICAL,
              value=bool(clip_ref), threshold=True, passed=bool(clip_ref),
              detail=(f"clip artifact {clip_ref}" if clip_ref
                      else "the clip seam produced no artifact")),
        _judge_check(f"clip.judge.{spec.segment_id}", CheckKind.INTENT,
                     verdict, threshold),
        duration_check,
    ]
    codes: list[RepairCode] = []
    if not clip_ref:
        codes.append(RepairCode.EMPTY_OUTPUT)
    if not duration_check.passed:
        codes.append(RepairCode.SHOT_TOO_SHORT)
    for code in (verdict.codes if not verdict.passed else ()):
        if code not in codes:
            codes.append(code)
    judge_results = ()
    if verdict.scored and verdict.judge:
        judge_results = (JudgeResult(
            judge=verdict.judge,
            verdict="YES" if verdict.passed else "NO",
            score=verdict.score, rationale=verdict.why),)
    return _card(checks, judge_results=judge_results, codes=codes,
                 recommended=("re-render THIS SHOT only; a targeted failure "
                              "must not rerun unrelated accepted shots "
                              "(Stage 17)"))


# ---------------------------------------------------------------------------
# Bounded repair (repair.py semantics, extended to the video codes)
# ---------------------------------------------------------------------------

#: Codes this orchestrator can repair by re-rolling the SAME node once. Every
#: other code either belongs to a different node (SHOT_TOO_SHORT is a lock
#: revision, Stage 10) or has no bounded repair defined, and both of those end
#: the shot honestly rather than spinning.
_RESEEDABLE: tuple[RepairCode, ...] = (
    RepairCode.INTENT_MISMATCH, RepairCode.IDENTITY_DRIFT,
    RepairCode.ACTION_MISSING, RepairCode.TEMPORAL_ARTIFACT,
)


def _repair_route(capability: str) -> RouteDecision:
    """A minimal ``RouteDecision`` so ``repair.attempt_repair`` can answer for
    the codes IT owns (worker/timeout/empty/decode). It is a deferred decision
    because that is the truth: video does not execute through
    ``POST /oracle/route``."""
    return RouteDecision(capability=capability, execution="deferred",
                         model_rationale="k106 performance orchestrator")


def repair_decision(goal: GoalSpec, capability: str, card: Scorecard,
                    ) -> oracle_repair.RepairDecision:
    """ONE bounded decision for a failing shot card — never an execution, never
    a loop (``repair.attempt_repair``'s discipline, reused).

    k90c's policy is asked FIRST, so the fleet-level codes (WORKER_UNAVAILABLE,
    TIMEOUT, EMPTY_OUTPUT, DECODE_FAILED) keep exactly the handling they have
    everywhere else. Only when it has nothing to say does this add the video
    codes k90c never had a node for."""
    decision = oracle_repair.attempt_repair(goal, _repair_route(capability), card)
    if decision.action != "none":
        return decision
    code = card.repair_code
    if code is None or card.hard_pass:
        return decision
    if code in _RESEEDABLE:
        return oracle_repair.RepairDecision(
            "reseed",
            f"{code.value} on {capability}: ONE bounded re-roll of this node "
            f"at a repair salt; the accepted shots around it are untouched")
    if code is RepairCode.SHOT_TOO_SHORT:
        return oracle_repair.RepairDecision(
            "none",
            f"{code.value}: the locked audio is authoritative, so the fix is a "
            f"LONGER SHOT WINDOW — a post-lock production revision "
            f"(ProductionLock.revise, Stage 10), not a re-roll of this render")
    return oracle_repair.RepairDecision(
        "none", f"no bounded repair defined for {code.value} on {capability}")


# ---------------------------------------------------------------------------
# The run context
# ---------------------------------------------------------------------------


class _Run:
    """Mutable working state for ONE call. Deliberately a plain object: the
    frozen contracts are the currency, this is the scratchpad the FAT
    orchestrator writes down as it goes."""

    def __init__(self, pgoal: PerformanceGoal, seams: PerformanceSeams,
                 budget: PerformanceBudget, run_id: str, state: RunState,
                 resuming: bool) -> None:
        self.pgoal = pgoal
        self.seams = seams
        self.budget = budget
        self.run_id = run_id
        self.state = state
        self.resuming = resuming
        self.started = time.monotonic()

        self.records: list[StageRecord] = []
        self.receipts: list[ExecutionReceipt] = []
        self.warnings: list[str] = []
        self.limitations: list[str] = []
        self.artifact_refs: list[str] = []

        self.registry_version: str | None = None
        self.snapshot: GenerationSnapshot | None = None
        self.ledger = RunPromptLedger()
        self.audio: AudioBuildResult | None = None
        self.master: AudioMaster | None = None
        self.shot_plan: ShotPlan | None = None
        self.continuity: ContinuityBible | None = None
        self.lock: ProductionLock | None = None
        self.specs: tuple[SegmentSpec, ...] = ()
        self.graph: PlanGraph | None = None
        self.validation: ValidationReport | None = None
        self.keyframes: dict[str, dict[str, Any]] = {}
        self.clips: dict[str, dict[str, Any]] = {}
        self.shots: list[ShotResult] = []
        self.video_ref: str | None = None
        self.final_card: Scorecard | None = None

    # -- bookkeeping -------------------------------------------------------

    def note(self, text: str) -> None:
        if text and text not in self.warnings:
            self.warnings.append(text)

    def limit(self, text: str) -> None:
        if text and text not in self.limitations:
            self.limitations.append(text)

    def refs(self, *values: str) -> None:
        for value in values:
            if value and value not in self.artifact_refs:
                self.artifact_refs.append(value)

    def receipt(self, stage: str, *, started_at: str, duration_s: float,
                failure: FailureClass | None = None,
                warnings: Sequence[str] = (),
                log: Sequence[str] = (),
                artifacts: Sequence[ArtifactRef] = (),
                request: Mapping[str, Any] | None = None) -> ExecutionReceipt:
        receipt = ExecutionReceipt(
            request=ExecutionReceipt.normalize_request(dict(request or {
                "stage": stage, "run_id": self.run_id,
                "recipe": PERFORMANCE_CAPABILITY})),
            capability=STAGE_CAPABILITY[stage],
            model_id="", worker=None,
            started_at=started_at, ended_at=_utc_now(),
            duration_s=max(0.0, _q(duration_s)),
            failure=failure, artifacts=tuple(artifacts),
            warnings=tuple(warnings), log_excerpt=tuple(log),
            registry_version=self.registry_version)
        self.receipts.append(receipt)
        return receipt

    def record(self, stage: str, *, ok: bool, resumed: bool = False,
               digests: Sequence[tuple[str, str]] = (),
               artifact_refs: Sequence[str] = (), detail: str = "",
               duration_s: float = 0.0) -> StageRecord:
        record = StageRecord(stage=stage, index=STAGES.index(stage), ok=ok,
                             resumed=resumed, digests=tuple(digests),
                             artifact_refs=tuple(artifact_refs), detail=detail,
                             duration_s=max(0.0, _q(duration_s)))
        self.records.append(record)
        return record

    def over_budget(self) -> str | None:
        if self.budget.max_seconds is None:
            return None
        spent = time.monotonic() - self.started
        if spent > float(self.budget.max_seconds):
            return (f"performance budget {self.budget.max_seconds:g}s exceeded "
                    f"after {spent:.1f}s")
        return None

    # -- outcomes ----------------------------------------------------------

    def result(self, *, ok: bool, gap: PerformanceGap | None = None,
               stopped_after: str | None = None) -> PerformanceResult:
        return PerformanceResult(
            run_id=self.run_id, ok=ok, goal_digest=self.pgoal.digest,
            stages=tuple(self.records), gap=gap, stopped_after=stopped_after,
            registry_version=self.registry_version, snapshot=self.snapshot,
            audio_master=self.master, lock=self.lock,
            segments=self.specs, plan_graph=self.graph,
            validation=self.validation, shots=tuple(self.shots),
            video_ref=self.video_ref, scorecard=self.final_card,
            receipts=tuple(self.receipts),
            artifact_refs=tuple(self.artifact_refs),
            limitations=tuple(self.limitations),
            warnings=tuple(self.warnings + self.state.write_errors),
            state_path=self.state.path)

    def fail(self, stage: str, gap: PerformanceGap, *,
             started_at: str | None = None,
             duration_s: float = 0.0,
             failure: FailureClass = FailureClass.CAPABILITY_GAP,
             ) -> PerformanceResult:
        """Stop at ``stage`` with a typed gap. Later stages are NOT recorded:
        "later stages untouched" has to be visible in the answer, not just true
        in the implementation."""
        self.receipt(stage, started_at=started_at or _utc_now(),
                     duration_s=duration_s, failure=failure,
                     log=(gap.diagnosis,))
        self.record(stage, ok=False, detail=gap.diagnosis,
                    duration_s=duration_s)
        self.state.drop_from(stage)
        self.state.flush()
        _finish_limitations(self)
        return self.result(ok=False, gap=gap)


# ---------------------------------------------------------------------------
# Stage 1 — authority
# ---------------------------------------------------------------------------


def authority_requirements(pgoal: PerformanceGoal,
                           ) -> tuple[tuple[AuthorityKind, str], ...]:
    """Every ``(kind, subject)`` this performance needs a grant for.

    Three sources, unioned in a stable order: k97's capability+request table for
    each capability the pipeline routes through; the identity refs the request
    names explicitly (a ref that never appears in the prompt text is still a
    person); and every REFERENCE voice in the casting, because cloning a
    specific voice is a rights decision that must not depend on where the ref
    happened to be spelled."""
    out: list[tuple[AuthorityKind, str]] = []

    def _add(kind: AuthorityKind, subject: str) -> None:
        pair = (kind, str(subject))
        if pair not in out:
            out.append(pair)

    for capability in GATED_CAPABILITIES:
        for kind, subject in oracle_authority.required_authorities(
                capability, pgoal.goal):
            _add(kind, subject)
    for ref in pgoal.identity_refs:
        _add(AuthorityKind.LIKENESS, ref)
    for ref in pgoal.voice_refs:
        _add(AuthorityKind.VOICE, ref)
    for _speaker, profile in pgoal.casting:
        if profile.kind is not VoiceKind.REFERENCE:
            continue
        subject = profile.reference_ref or ""
        found = oracle_authority.find_subject_refs(subject)
        _add(AuthorityKind.VOICE,
             found[0][1] if found else f"voice_profile:{profile.voice_id}")
    return tuple(out)


def check_authority(pgoal: PerformanceGoal
                    ) -> oracle_authority.AuthorityDecision:
    """The stage-1 gate. Grants come ONLY from ``goal.rights`` — absence is
    never consent (doc §11), which is why a request with no manifest and any
    requirement at all is refused by name."""
    required = authority_requirements(pgoal)
    if not required:
        return oracle_authority.AuthorityDecision(
            ok=True, required=(),
            reason="no typed authority required by this performance")
    rights = pgoal.goal.rights
    missing = tuple((kind, subject) for kind, subject in required
                    if rights is None or not rights.covers(kind, subject))
    if not missing:
        return oracle_authority.AuthorityDecision(
            ok=True, required=required,
            reason=("authorized by the request's RightsManifest: "
                    + ", ".join(f"{k.value} for {s}" for k, s in required)))
    named = ", ".join(f"{k.value} for {s}" for k, s in missing)
    why = (f"{PERFORMANCE_CAPABILITY}: no RightsManifest on the request — "
           f"absence is not consent; missing {named}" if rights is None else
           f"{PERFORMANCE_CAPABILITY}: the request's RightsManifest does not "
           f"cover {named}")
    return oracle_authority.AuthorityDecision(ok=False, missing=missing,
                                              reason=why, required=required)


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


def run_performance(goal: PerformanceGoal, *,
                    seams: PerformanceSeams | None = None,
                    budget: PerformanceBudget | None = None,
                    resume: str | None = None,
                    run_id: str | None = None,
                    stop_after: str | None = None) -> PerformanceResult:
    """Run the audio-first performance recipe end to end, or stop honestly.

    ``resume`` is a run id: its journal is loaded and every stage whose
    recorded artifacts still verify BY DIGEST is skipped (the authority gate
    excepted — see the module docstring, rule 5). ``stop_after`` is the
    checkpoint affordance: run through that stage, persist the journal, and
    return without a gap — a dry run that a later ``resume`` continues.

    Returns a :class:`PerformanceResult` in every case and raises only for a
    caller fault (a bad goal, a bad seam set, an unknown stage name)."""
    if not isinstance(goal, PerformanceGoal):
        raise PerformanceError(
            f"run_performance takes a PerformanceGoal, got "
            f"{type(goal).__name__}")
    seams = seams if seams is not None else default_seams()
    if not isinstance(seams, PerformanceSeams):
        raise PerformanceError(
            f"seams must be a PerformanceSeams, got {type(seams).__name__}")
    budget = budget if budget is not None else PerformanceBudget()
    if not isinstance(budget, PerformanceBudget):
        raise PerformanceError(
            f"budget must be a PerformanceBudget, got {type(budget).__name__}")
    if stop_after is not None and stop_after not in STAGES:
        raise PerformanceError(
            f"stop_after must be one of {list(STAGES)}, got {stop_after!r}")

    identifier = str(resume or run_id or derive_run_id(goal))
    path = state_path(identifier, seams.run_root)
    state = RunState.load(path)
    resuming = False
    stale_note = ""
    if resume is not None:
        if state is None:
            stale_note = (f"resume={resume!r}: no readable run state at {path} "
                          f"— every stage runs")
        elif state.goal_digest != goal.digest:
            stale_note = (f"resume={resume!r}: the journal was written for goal "
                          f"{state.goal_digest[:12]}… but this request is "
                          f"{goal.digest[:12]}… — nothing is resumed")
            state = None
        else:
            resuming = True
    if state is None:
        state = RunState(run_id=identifier, goal_digest_=goal.digest, path=path)

    run = _Run(goal, seams, budget, identifier, state, resuming)
    if stale_note:
        run.note(stale_note)

    for stage in STAGES:
        over = run.over_budget()
        if over is not None:
            return run.fail(stage, PerformanceGap(
                stage=stage, diagnosis=over,
                repair_codes=(RepairCode.TIMEOUT,),
                capability=PERFORMANCE_CAPABILITY,
                requirement=("raise PerformanceBudget.max_seconds, or reduce "
                             "the candidate fan-out")),
                failure=FailureClass.TIMEOUT)

        outcome = _STAGE_FN[stage](run)
        if outcome is not None:
            return outcome
        if stop_after == stage:
            run.state.flush()
            _finish_limitations(run)
            return run.result(ok=False, stopped_after=stage)

    _finish_limitations(run)
    ok = bool(run.video_ref) and bool(run.final_card
                                      and run.final_card.hard_pass)
    if ok:
        return run.result(ok=True)
    card = run.final_card
    codes = (card.repair_code,) if card and card.repair_code else ()
    return run.result(ok=False, gap=PerformanceGap(
        stage="assembly",
        diagnosis=((card.diagnosis if card and card.diagnosis else None)
                   or "the assembled performance did not pass its whole-result "
                      "scorecard"),
        repair_codes=tuple(c for c in codes if c is not None),
        capability=ASSEMBLE_CAPABILITY,
        requirement=("re-run the failing shot(s) named on the scorecard; the "
                     "locked audio and the accepted shots are untouched"),
        evidence=tuple(c.detail for c in (card.checks if card else ())
                       if not c.passed)))


# --- stage 1 ---------------------------------------------------------------


def _stage_authority(run: _Run) -> PerformanceResult | None:
    started_at, t0 = _utc_now(), time.monotonic()
    decision = check_authority(run.pgoal)
    spent = time.monotonic() - t0
    if not decision.ok:
        card = oracle_authority.refusal_scorecard(decision)
        run.final_card = card
        # registry_version is None ON PURPOSE and stays that way for the whole
        # of stage 1: the gate runs BEFORE the catalog is read, so a refusal
        # honestly records "no registry was consulted" rather than stamping a
        # snapshot id this decision never depended on (k97's own rule for
        # refusal_receipt, kept here).
        receipt = oracle_authority.refusal_receipt(
            run.pgoal.goal, PERFORMANCE_CAPABILITY, decision,
            registry_version=None)
        run.receipts.append(receipt)
        run.record("authority", ok=False, detail=decision.reason,
                   duration_s=spent)
        run.state.drop_from("authority")
        run.state.flush()
        _finish_limitations(run)
        return run.result(ok=False, gap=PerformanceGap(
            stage="authority", diagnosis=decision.reason,
            repair_codes=(RepairCode.SOURCE_AUTHORITY_MISSING,),
            capability=PERFORMANCE_CAPABILITY,
            requirement=(card.recommended_repair or
                         "supply a RightsManifest covering the missing grants"),
            evidence=tuple(f"{k.value}:{s}" for k, s in decision.missing)))

    run.receipt("authority", started_at=started_at, duration_s=spent,
                request={"stage": "authority", "run_id": run.run_id,
                         "required": [f"{k.value}:{s}"
                                      for k, s in decision.required]})
    run.record("authority", ok=True, duration_s=spent,
               detail=(decision.reason if decision.required else
                       "no typed authority required"),
               digests=(("required",
                         digest_payload([[k.value, s]
                                         for k, s in decision.required])),))
    return None


# --- stage 2 ---------------------------------------------------------------


def _stage_snapshot(run: _Run) -> PerformanceResult | None:
    started_at, t0 = _utc_now(), time.monotonic()
    pgoal = run.pgoal

    if run.seams.registry_version is not None:
        try:
            run.registry_version = run.seams.registry_version()
        except Exception as exc:                   # noqa: BLE001
            run.note(f"registry_version seam raised ({type(exc).__name__}: "
                     f"{exc}); recording None rather than a guess")
            run.registry_version = None
    if run.registry_version is None:
        run.limit("registry_version is unrecorded: this run cannot be tied to "
                  "a routing-registry snapshot (k105)")

    resumed = _resume_snapshot(run)
    if not resumed:
        try:
            run.snapshot = GenerationSnapshot(
                raw_request_ref=pgoal.raw_request_ref,
                prompts_before_run=pgoal.prompts_before_run,
                operator_refs=pgoal.operator_refs,
                acquisition_refs=pgoal.acquisition_refs,
                identity_refs=pgoal.identity_refs,
                voice_refs=pgoal.voice_refs,
                deliverable=pgoal.effective_deliverable,
                exclusions=pgoal.exclusions,
                registry_version=run.registry_version,
                created_at=pgoal.created_at)
        except (ValueError, TypeError) as exc:
            return run.fail("snapshot", PerformanceGap(
                stage="snapshot",
                diagnosis=f"the generation snapshot is not constructible: {exc}",
                capability=PERFORMANCE_CAPABILITY,
                requirement=("fix the request: Stage 4 needs a raw request ref "
                             "and a named deliverable")),
                started_at=started_at,
                duration_s=time.monotonic() - t0,
                failure=FailureClass.RUNNER_ERROR)
        run.state.drop_from("snapshot")
        run.state.record("snapshot", digest=run.snapshot.digest,
                         payload=run.snapshot.to_dict())

    spent = time.monotonic() - t0
    run.receipt("snapshot", started_at=started_at, duration_s=spent)
    run.record("snapshot", ok=True, resumed=bool(resumed), duration_s=spent,
               digests=(("snapshot", run.snapshot.digest),),
               detail=("resumed from the run journal (digest verified)"
                       if resumed else
                       f"{len(run.snapshot.prompt_digests)} pre-run prompt(s) "
                       f"recorded; the run ledger starts empty"))
    return None


def _resume_snapshot(run: _Run) -> bool:
    if not run.resuming or not run.state.intact("snapshot"):
        return False
    entry = run.state.entry("snapshot") or {}
    try:
        snapshot = GenerationSnapshot.from_dict(entry["payload"])
    except (KeyError, TypeError, ValueError):
        return False
    if snapshot.digest != entry.get("digest"):
        return False
    if snapshot.registry_version != run.registry_version:
        run.note("run state snapshot carries registry_version "
                 f"{snapshot.registry_version!r} but this fleet now reports "
                 f"{run.registry_version!r} — re-running the snapshot stage")
        return False
    run.snapshot = snapshot
    return True


# --- stage 3 ---------------------------------------------------------------


def _seam_from_failure(exc: Exception) -> str:
    """WHICH seam raised a ``SeamUnavailable``.

    Every live seam raises with its CAPABILITY as the message prefix
    (``_executable_route`` and the seams below it), so the capability names the
    seam. Before this, stage 3 attributed EVERY audio-stage SeamUnavailable to
    ``synth`` — harmless while synth was the only unbound seam, actively
    misleading the moment it was seated: a transcribe failure came back quoting
    the chatterbox seating step (observed live 2026-08-21, when the round trip
    died on a worker with no ffmpeg). Defaults to ``synth`` — the seam this
    stage exists for — when the message says nothing recognisable."""
    text = str(exc)
    for seam in ("synth", "transcribe", "similarity"):
        capability = SEAM_CAPABILITY.get(seam)
        if capability and text.startswith(f"{capability}:"):
            return seam
    if text.startswith(f"{TRANSCRIBE_FALLBACK_CAPABILITY}:"):
        return "transcribe"                    # the word_timestamps fallback
    return "synth"


def _stage_audio(run: _Run) -> PerformanceResult | None:
    started_at, t0 = _utc_now(), time.monotonic()
    pgoal, seams = run.pgoal, run.seams

    if _resume_audio(run):
        spent = time.monotonic() - t0
        run.receipt("audio", started_at=started_at, duration_s=spent,
                    warnings=("resumed from the run journal; no take was "
                              "re-synthesized",))
        run.record("audio", ok=True, resumed=True, duration_s=spent,
                   digests=(("audio_master", run.master.digest),
                            ("dialogue_timeline", run.master.timeline_digest)),
                   artifact_refs=tuple(ref for _l, ref in run.master.tracks),
                   detail="resumed from the run journal (digest verified)")
        run.refs(*(ref for _l, ref in run.master.tracks))
        _audio_limitations(run)
        return None

    for seam in ("synth", "transcribe"):
        if seams.bound(seam):
            continue
        gap = seams.gap_for(seam)
        return run.fail("audio", PerformanceGap(
            stage="audio",
            diagnosis=(f"stage 3 needs the {seam!r} seam ({gap.capability}) and "
                       f"nothing on this fleet provides it, so no audio was "
                       f"produced and no later stage ran"),
            repair_codes=(RepairCode.CAPABILITY_GAP,),
            capability=gap.capability, requirement=gap.requirement),
            started_at=started_at, duration_s=time.monotonic() - t0)

    policy = pgoal.speech_policy
    if policy.registry_version is None and run.registry_version is not None:
        policy = replace(policy, registry_version=run.registry_version)
    elif policy.registry_version is not None and \
            run.registry_version is not None and \
            policy.registry_version != run.registry_version:
        return run.fail("audio", PerformanceGap(
            stage="audio",
            diagnosis=(f"the speech policy pins registry version "
                       f"{policy.registry_version!r} but this fleet reports "
                       f"{run.registry_version!r} — Stage 11 locks ONE accepted "
                       f"registry version"),
            capability=PERFORMANCE_CAPABILITY,
            requirement="clear SpeechPolicy.registry_version, or re-pin it"),
            started_at=started_at, duration_s=time.monotonic() - t0,
            failure=FailureClass.RUNNER_ERROR)

    try:
        result = build_audio_master(
            pgoal.dialogue, pgoal.voices,
            synth=seams.synth, transcribe=seams.transcribe,
            similarity=seams.similarity, candidates=seams.tts_candidates,
            policy=policy)
    except SeamUnavailable as exc:
        seam_name = _seam_from_failure(exc)
        if seams.bound(seam_name):
            # BOUND but failing is a different fact from unbound, and the gap
            # table has nothing true to say about it: quoting a seating step for
            # a seam that is already seated sends the reader to the wrong box.
            capability = SEAM_CAPABILITY[seam_name]
            requirement = (
                f"the {seam_name!r} seam IS bound to {capability} and the "
                f"dispatch reached the fleet — the CALL failed: {exc}. This is "
                f"not a missing seat: fix the backend the message names (a "
                f"worker missing a binary or a model, not a capability nobody "
                f"has seated).")
        else:
            gap = seams.gap_for(seam_name)
            capability, requirement = gap.capability, gap.requirement
        return run.fail("audio", PerformanceGap(
            stage="audio",
            diagnosis=(f"the {seam_name!r} seam could not reach its backend: "
                       f"{exc}"),
            repair_codes=(RepairCode.CAPABILITY_GAP,),
            capability=capability, requirement=requirement),
            started_at=started_at, duration_s=time.monotonic() - t0)
    except (TypeError, ValueError) as exc:
        return run.fail("audio", PerformanceGap(
            stage="audio",
            diagnosis=f"the audio fan-out refused its inputs: {exc}",
            capability=TTS_CAPABILITY,
            requirement=("fix the dialogue/casting: build_audio_master raises "
                         "only for a caller fault")),
            started_at=started_at, duration_s=time.monotonic() - t0,
            failure=FailureClass.RUNNER_ERROR)

    run.audio = result
    for warning in result.warnings:
        run.note(f"audio: {warning}")
    if not result.ok:
        return run.fail("audio", PerformanceGap(
            stage="audio",
            diagnosis=("no take of " +
                       ", ".join(g.line_id for g in result.gaps) +
                       f" passed the round-trip judge across "
                       f"{result.candidates_considered} candidate(s): " +
                       "; ".join(g.diagnosis for g in result.gaps)),
            repair_codes=result.repair_codes,
            capability=TTS_CAPABILITY,
            requirement=("re-synthesize the named line(s) at a different "
                         "SpeechPolicy.seed_salt, or fix the voice casting; "
                         "the locked dialogue is authoritative and is never "
                         "rewritten to match a take"),
            evidence=tuple(g.line_id for g in result.gaps)),
            started_at=started_at, duration_s=time.monotonic() - t0,
            failure=FailureClass.EMPTY_OUTPUT)

    run.master = result.master
    run.refs(*(ref for _l, ref in run.master.tracks))
    run.state.drop_from("audio")
    run.state.record("audio", digest=run.master.digest,
                     payload=run.master.to_dict(),
                     refs=[ref for _l, ref in run.master.tracks])

    spent = time.monotonic() - t0
    run.receipt(
        "audio", started_at=started_at, duration_s=spent,
        artifacts=tuple(ArtifactRef(kind=ArtifactKind.AUDIO, uri=ref)
                        for _l, ref in run.master.tracks),
        warnings=result.warnings)
    run.record("audio", ok=True, duration_s=spent,
               digests=(("audio_master", run.master.digest),
                        ("dialogue_timeline", run.master.timeline_digest)),
               artifact_refs=tuple(ref for _l, ref in run.master.tracks),
               detail=(f"{len(run.master.line_timings)} line(s), "
                       f"{run.master.total_seconds:.3f}s, "
                       f"{result.candidates_considered} candidate(s) judged"))
    _audio_limitations(run)
    return None


def _resume_audio(run: _Run) -> bool:
    if not run.resuming or not run.state.intact("audio"):
        return False
    entry = run.state.entry("audio") or {}
    try:
        master = AudioMaster.from_dict(entry["payload"])
    except (KeyError, TypeError, ValueError):
        return False
    if master.digest != entry.get("digest"):
        return False
    if master.timeline_digest != run.pgoal.dialogue.digest:
        run.note("run state audio master realizes a different dialogue lock — "
                 "re-synthesizing")
        return False
    if not master.locked:
        return False
    run.master = master
    return True


def _audio_limitations(run: _Run) -> None:
    if run.seams.similarity is None:
        gap = run.seams.gap_for("similarity")
        run.limit(f"voice similarity is UNSCORED (not passed): {gap.requirement}")
    if run.master is not None and not any(t.words for t in
                                          run.master.line_timings):
        run.limit("no per-word timings were measured: the round-trip transcript "
                  "carried none, so shot boundaries fall on line edges only "
                  "(a long line is never split mid-word)")


# --- stage 4 ---------------------------------------------------------------


def _stage_lock(run: _Run) -> PerformanceResult | None:
    started_at, t0 = _utc_now(), time.monotonic()
    pgoal = run.pgoal

    if not _resume_lock(run):
        try:
            windows = shot_windows_from_audio(
                run.master, min_shot_s=pgoal.min_shot_s,
                max_shot_s=pgoal.max_shot_s, pad_s=pgoal.pad_s)
            shot_plan = shot_plan_from_windows(
                windows, rubric=pgoal.rubric, camera=pgoal.camera,
                blocking=pgoal.blocking, lighting=pgoal.lighting,
                prefix=pgoal.segment_prefix)
        except (TypeError, ValueError) as exc:
            return run.fail("lock", PerformanceGap(
                stage="lock",
                diagnosis=f"shot windows could not be derived: {exc}",
                capability=PERFORMANCE_CAPABILITY,
                requirement=("check min_shot_s/max_shot_s/pad_s against the "
                             "locked audio")),
                started_at=started_at, duration_s=time.monotonic() - t0,
                failure=FailureClass.RUNNER_ERROR)

        continuity = pgoal.continuity or _continuity_shell(pgoal, shot_plan)
        try:
            lock = ProductionLock.lock(
                run.snapshot, audio_master=run.master, continuity=continuity,
                shot_plan=shot_plan,
                identity_refs=pgoal.identity_refs or None,
                registry_version=run.registry_version,
                locked_at=pgoal.created_at, run_prompts=run.ledger)
        except (LockRefused, ProductionError, TypeError) as exc:
            return run.fail("lock", PerformanceGap(
                stage="lock",
                diagnosis=f"the production lock refused: {exc}",
                capability=PERFORMANCE_CAPABILITY,
                requirement=("fix the artifact the refusal names; a lock is "
                             "the Stage 11 approval transition, not a "
                             "formality"),
                evidence=(str(exc),)),
                started_at=started_at, duration_s=time.monotonic() - t0,
                failure=FailureClass.RUNNER_ERROR)

        run.shot_plan, run.continuity, run.lock = shot_plan, continuity, lock
        run.state.drop_from("lock")
        run.state.record("lock", digest=lock.digest, payload={
            "lock": lock.to_dict(), "shot_plan": shot_plan.to_dict(),
            "continuity": continuity.to_dict()})
        resumed = False
    else:
        resumed = True

    spent = time.monotonic() - t0
    run.receipt("lock", started_at=started_at, duration_s=spent)
    run.record("lock", ok=True, resumed=resumed, duration_s=spent,
               digests=(("production_lock", run.lock.digest),
                        ("shot_plan", run.shot_plan.digest),
                        ("continuity", run.continuity.digest),
                        ("snapshot", run.lock.snapshot_digest)),
               detail=("resumed from the run journal (all four digests "
                       "verified against the lock)" if resumed else
                       f"{len(run.shot_plan.entries)} shot(s) over "
                       f"{run.shot_plan.total_seconds:.3f}s of locked audio"))
    if run.lock.screenplay_digest is None:
        run.limit("no screenplay artifact (k110): the lock records "
                  "screenplay_digest=None, which means there is none — not "
                  "that one existed and went unrecorded")
    if pgoal.continuity is None:
        run.limit("the continuity bible is an EMPTY SHELL: no before/after "
                  "state was authored for any segment (k110), so no "
                  "continuity-with-neighbours check can be made (k107)")
    return None


def _continuity_shell(pgoal: PerformanceGoal, shot_plan: ShotPlan
                      ) -> ContinuityBible:
    """The minimum Stage 7 shape the lock will accept: one explicit (empty)
    state per segment, plus the cast as the standing character inventory. It
    asserts nothing about wardrobe, props or location — k110 authors those, and
    an invented inventory would be indistinguishable from a real one."""
    return ContinuityBible(
        entries=tuple(ContinuityState(segment_id=sid)
                      for sid in shot_plan.segment_ids),
        characters=pgoal.dialogue.speakers,
        notes=("k106 shell: no continuity content authored (k110 fills this); "
               "each segment carries an explicit empty before/after state"))


def _resume_lock(run: _Run) -> bool:
    if not run.resuming or not run.state.intact("lock"):
        return False
    entry = run.state.entry("lock") or {}
    payload = entry.get("payload") or {}
    try:
        lock = ProductionLock.from_dict(payload["lock"])
        shot_plan = ShotPlan.from_dict(payload["shot_plan"])
        continuity = ContinuityBible.from_dict(payload["continuity"])
    except (KeyError, TypeError, ValueError):
        return False
    if lock.digest != entry.get("digest"):
        return False
    # The payoff of content addressing: the lock itself says what it locked, so
    # every rehydrated artifact is checked against IT, not against the journal.
    if shot_plan.digest != lock.shot_plan_digest:
        return False
    if continuity.digest != lock.continuity_digest:
        return False
    if run.snapshot is None or lock.snapshot_digest != run.snapshot.digest:
        return False
    if run.master is None or lock.audio_master_digest != run.master.digest:
        return False
    run.lock, run.shot_plan, run.continuity = lock, shot_plan, continuity
    return True


# --- stage 5 ---------------------------------------------------------------


def _stage_segments(run: _Run) -> PerformanceResult | None:
    started_at, t0 = _utc_now(), time.monotonic()
    pgoal = run.pgoal
    resumed = _resume_segments(run)

    if not resumed:
        writer = pgoal.prompt_writer or default_prompt_writer

        def _ledgered_writer(context: Any, index: int) -> str:
            """Every prompt minted DURING the run goes into the ledger. k104
            built the mechanism and could not supply the ledger; this is the
            supply. Without it, Stage 4's rule is a mechanism nobody runs."""
            written = writer(context, index)
            if isinstance(written, str) and written.strip():
                run.ledger.record(written)
            return written

        try:
            specs = compile_segments(
                run.lock, snapshot=run.snapshot, audio_master=run.master,
                continuity=run.continuity, shot_plan=run.shot_plan,
                tone=pgoal.tone, identity_refs=pgoal.identity_refs or None,
                prompt_writer=_ledgered_writer,
                negative_prompt=pgoal.negative_prompt,
                dialogue=pgoal.dialogue_map, seed_salt=pgoal.seed_salt)
        except (CompileRefused, SiblingViolation, ProductionError,
                TypeError, ValueError) as exc:
            return run.fail("segments", PerformanceGap(
                stage="segments",
                diagnosis=f"segment compilation refused: {exc}",
                capability=PERFORMANCE_CAPABILITY,
                requirement=("compile against exactly the artifacts the lock "
                             "locked (invariant 4/9)"),
                evidence=(str(exc),)),
                started_at=started_at, duration_s=time.monotonic() - t0,
                failure=FailureClass.RUNNER_ERROR)

        # Stage 4's rule, run at the one place it can actually catch something:
        # a writer that minted a prompt the immutable snapshot also claims as
        # pre-run.
        try:
            run.snapshot.assert_pre_run(run.ledger)
        except ProductionError as exc:
            return run.fail("segments", PerformanceGap(
                stage="segments",
                diagnosis=(f"invariant 9: a prompt minted during this run is "
                           f"also claimed as pre-run by the immutable "
                           f"snapshot — {exc}"),
                capability=PERFORMANCE_CAPABILITY,
                requirement=("remove the prompt from "
                             "GenerationSnapshot.prompts_before_run, or stop "
                             "the writer from minting it")),
                started_at=started_at, duration_s=time.monotonic() - t0,
                failure=FailureClass.RUNNER_ERROR)

        run.specs = specs
        run.state.drop_from("segments")
        run.state.record("segments",
                         digest=digest_payload([s.digest for s in specs]),
                         payload=[s.to_dict() for s in specs])

    try:
        run.graph = to_plan_graph(run.specs, run.lock, goal=pgoal.goal)
    except (SiblingViolation, CompileRefused) as exc:
        return run.fail("segments", PerformanceGap(
            stage="segments",
            diagnosis=f"the plan graph violates Stage 14: {exc}",
            capability=PERFORMANCE_CAPABILITY,
            requirement="segments must hang off the LOCKED artifacts",
            evidence=(str(exc),)),
            started_at=started_at, duration_s=time.monotonic() - t0,
            failure=FailureClass.RUNNER_ERROR)

    view: Mapping[str, Any] = {}
    if run.seams.catalog_view is not None:
        try:
            view = run.seams.catalog_view() or {}
        except Exception as exc:                   # noqa: BLE001
            run.note(f"catalog view unreadable ({type(exc).__name__}: {exc}); "
                     f"validating against an EMPTY catalog, which will report "
                     f"every capability unknown")
            view = {}
    run.validation = validate(run.graph, view, pgoal.goal)
    if not run.validation.ok:
        return run.fail("segments", PerformanceGap(
            stage="segments",
            diagnosis=("the plan does not statically validate against this "
                       "fleet: " +
                       "; ".join(e.message for e in run.validation.errors)),
            repair_codes=_validation_codes(run.validation),
            capability=PERFORMANCE_CAPABILITY,
            requirement=("fix or seat what the validation errors name; a plan "
                         "that does not validate is not 'probably fine'"),
            segment_ids=tuple(dict.fromkeys(
                e.node_id for e in run.validation.errors if e.node_id)),
            evidence=tuple(f"{e.code.value}: {e.message}"
                           for e in run.validation.errors)),
            started_at=started_at, duration_s=time.monotonic() - t0)
    for warning in run.validation.warnings:
        run.note(f"plan validation: {warning.message}")

    spent = time.monotonic() - t0
    order = execution_order(run.specs, "sequential", graph=run.graph)
    run.receipt("segments", started_at=started_at, duration_s=spent,
                warnings=tuple(w.message for w in run.validation.warnings))
    run.record("segments", ok=True, resumed=resumed, duration_s=spent,
               digests=tuple(("segment:" + s.segment_id, s.digest)
                             for s in run.specs)
               + (("plan_graph", run.graph.structure_digest()),),
               detail=(f"{len(run.specs)} sibling segment(s); "
                       f"{len(order)} sequential batch(es); "
                       f"{len(run.ledger)} run prompt(s) ledgered"
                       + (" (resumed)" if resumed else "")))
    without = [s.segment_id for s in run.specs if s.spatial_ref is None]
    if len(without) == len(run.specs):
        run.limit("no spatial conditioning: no segment carries a spatial_ref "
                  "(the goal supplied no scene source — glTF/USDA/pose track "
                  "— for any shot), so geometry is unconstrained and unscored "
                  "rather than constrained and passing")
    elif without:
        run.limit(f"partial spatial conditioning: {len(without)} of "
                  f"{len(run.specs)} segment(s) carry no spatial_ref "
                  f"({', '.join(without)}); their geometry is unconstrained "
                  f"and unscored")
    return None


def _validation_codes(report: ValidationReport) -> tuple[RepairCode, ...]:
    out: list[RepairCode] = []
    for error in report.errors:
        code = {"authority_missing": RepairCode.SOURCE_AUTHORITY_MISSING,
                "capability_gap": RepairCode.CAPABILITY_GAP,
                "unknown_capability": RepairCode.CAPABILITY_GAP,
                }.get(error.code.value)
        if code is not None and code not in out:
            out.append(code)
    return tuple(out)


def _resume_segments(run: _Run) -> bool:
    if not run.resuming or not run.state.intact("segments"):
        return False
    entry = run.state.entry("segments") or {}
    try:
        specs = tuple(SegmentSpec.from_dict(d) for d in entry["payload"])
    except (KeyError, TypeError, ValueError):
        return False
    if not specs:
        return False
    if digest_payload([s.digest for s in specs]) != entry.get("digest"):
        return False
    if run.lock is None or any(s.lock_digest != run.lock.digest for s in specs):
        return False
    run.specs = specs
    return True


# --- stage 6 ---------------------------------------------------------------


def _stage_keyframes(run: _Run) -> PerformanceResult | None:
    started_at, t0 = _utc_now(), time.monotonic()
    seams = run.seams

    if _resume_keyframes(run):
        spent = time.monotonic() - t0
        refs = tuple(str(v["ref"]) for v in run.keyframes.values())
        run.refs(*refs)
        run.receipt("keyframes", started_at=started_at, duration_s=spent,
                    warnings=("resumed from the run journal; no keyframe was "
                              "regenerated",))
        run.record("keyframes", ok=True, resumed=True, duration_s=spent,
                   digests=tuple((sid, digest_payload(v))
                                 for sid, v in sorted(run.keyframes.items())),
                   artifact_refs=refs,
                   detail="resumed from the run journal (digest verified)")
        _keyframe_limitations(run)
        return None

    if not seams.bound("gen_image"):
        gap = seams.gap_for("gen_image")
        return run.fail("keyframes", PerformanceGap(
            stage="keyframes",
            diagnosis=(f"stage 6 needs the 'gen_image' seam ({gap.capability}); "
                       f"Stage 15 puts a judged keyframe BEFORE any video, so "
                       f"no clip was generated"),
            repair_codes=(RepairCode.CAPABILITY_GAP,),
            capability=gap.capability, requirement=gap.requirement),
            started_at=started_at, duration_s=time.monotonic() - t0)

    threshold = float(THRESHOLDS[run.pgoal.goal.quality])
    failures: list[tuple[SegmentSpec, Scorecard, int, bool]] = []

    for spec in run.specs:
        over = run.over_budget()
        if over is not None:
            return run.fail("keyframes", PerformanceGap(
                stage="keyframes", diagnosis=over,
                repair_codes=(RepairCode.TIMEOUT,),
                capability=PERFORMANCE_CAPABILITY,
                requirement="raise the budget or lower keyframe_candidates",
                segment_ids=(spec.segment_id,)),
                started_at=started_at, duration_s=time.monotonic() - t0,
                failure=FailureClass.TIMEOUT)

        chosen, card, considered, repaired, error = _keyframe_for(
            run, spec, threshold)
        if error is not None:
            return run.fail("keyframes", error, started_at=started_at,
                            duration_s=time.monotonic() - t0)
        if chosen is None:
            failures.append((spec, card, considered, repaired))
            continue
        run.keyframes[spec.segment_id] = {
            "ref": chosen["ref"], "seed": chosen["seed"],
            "candidates": considered, "repaired": repaired,
            "scorecard": card.to_dict()}
        run.refs(str(chosen["ref"]))

    if failures:
        codes: list[RepairCode] = []
        for _spec, card, _n, _r in failures:
            if card.repair_code is not None and card.repair_code not in codes:
                codes.append(card.repair_code)
        return run.fail("keyframes", PerformanceGap(
            stage="keyframes",
            diagnosis=("no keyframe candidate passed for segment(s) " +
                       ", ".join(s.segment_id for s, _c, _n, _r in failures) +
                       "; Stage 15 forbids authorizing video off a rejected "
                       "still — " +
                       "; ".join(c.diagnosis or "no diagnosis"
                                 for _s, c, _n, _r in failures)),
            repair_codes=tuple(codes),
            capability=KEYFRAME_CAPABILITY,
            requirement=("repair the identity pack / spatial conditioning / "
                         "keyframe node for the named segment(s) only; the "
                         "audio and the accepted shots stay as they are"),
            segment_ids=tuple(s.segment_id for s, _c, _n, _r in failures)),
            started_at=started_at, duration_s=time.monotonic() - t0,
            failure=FailureClass.EMPTY_OUTPUT)

    run.state.drop_from("keyframes")
    run.state.record(
        "keyframes",
        digest=digest_payload({k: v for k, v in sorted(run.keyframes.items())}),
        payload={k: v for k, v in sorted(run.keyframes.items())},
        refs=[str(v["ref"]) for v in run.keyframes.values()])

    spent = time.monotonic() - t0
    refs = tuple(str(v["ref"]) for v in run.keyframes.values())
    run.receipt("keyframes", started_at=started_at, duration_s=spent,
                artifacts=tuple(ArtifactRef(kind=ArtifactKind.IMAGE, uri=r)
                                for r in refs))
    run.record("keyframes", ok=True, duration_s=spent,
               digests=tuple((sid, digest_payload(v))
                             for sid, v in sorted(run.keyframes.items())),
               artifact_refs=refs,
               detail=(f"{len(refs)} accepted keyframe(s); "
                       f"{sum(v['candidates'] for v in run.keyframes.values())} "
                       f"candidate(s) judged; "
                       f"{sum(1 for v in run.keyframes.values() if v['repaired'])}"
                       f" repaired"))
    _keyframe_limitations(run)
    return None


def _keyframe_for(run: _Run, spec: SegmentSpec, threshold: float):
    """Up to N candidates for ONE segment, then at most ONE bounded repair
    round when the failure is an IDENTITY failure.

    Returns ``(chosen | None, card, candidates_considered, repaired, error)``.
    The repair round re-runs THIS KEYFRAME and nothing else — no transcription,
    no audio, no sibling segment. Stage 15 states that as a rule about what a
    repair must NOT touch, so it is enforced by never calling anything else
    from in here rather than by a comment asking nicely.

    Seeds are ``segment_seed(lock_digest, segment_id [+ repair salt]) +
    attempt``: deterministic, reproducible, and different every candidate — the
    k102/k104 idiom, where the salt IS the repair dial."""
    seams = run.seams
    considered = 0
    repaired = False
    last: tuple[dict[str, Any], Scorecard] | None = None

    rounds = (0, 1) if run.budget.keyframe_repair_rounds else (0,)
    plan = _context_plan(run, spec, seams.keyframe_candidates)
    n_candidates = max(int(seams.keyframe_candidates), plan.candidates if plan else 1)
    for round_index in rounds:
        key = (f"{spec.segment_id}:{KEYFRAME_REPAIR_SALT}" if round_index
               else spec.segment_id)
        base = segment_seed(spec.lock_digest, key, 0)
        failed_models: list[str] = []
        for attempt in range(n_candidates):
            seed = (base + attempt) % (2 ** 32)
            prompt, angle = _variant_prompt(spec.prompt, plan, attempt)
            pick, decision = _candidate_pick(run, KEYFRAME_CAPABILITY, attempt, n_candidates, failed_models)
            try:
                with _pin(KEYFRAME_CAPABILITY, pick, decision):
                    produced = seams.gen_image(prompt, spec.identity_refs, seed)
            except SeamUnavailable as exc:
                gap = seams.gap_for("gen_image")
                return None, None, considered, repaired, PerformanceGap(
                    stage="keyframes",
                    diagnosis=(f"the keyframe seam could not reach its backend "
                               f"for {spec.segment_id}: {exc}"),
                    repair_codes=(RepairCode.CAPABILITY_GAP,),
                    capability=gap.capability, requirement=gap.requirement,
                    segment_ids=(spec.segment_id,))
            ref = _ref_of(produced)
            considered += 1
            # The DEFAULT code for a rejected still is INTENT_MISMATCH, not
            # IDENTITY_DRIFT: only a judge that actually says "identity" buys
            # the repair round below. Defaulting to IDENTITY_DRIFT would let
            # every generic rejection spend a second round under a code nobody
            # measured, which is a loop wearing a diagnosis.
            verdict = _judge(run, "judge_image", ref, spec, threshold,
                             RepairCode.INTENT_MISMATCH)
            card = keyframe_scorecard(spec, verdict, image_ref=ref,
                                      threshold=threshold)
            _attribute(ref, KEYFRAME_CAPABILITY, pick)
            _note_verdict(ref, card)
            producer = _producer_model(ref) or pick
            if not card.hard_pass and producer:
                failed_models.append(producer)   # next candidate goes to a different model when one exists
            last = ({"ref": ref, "seed": seed, "angle": angle, "model_id": producer,
                     "selection": decision,
                     "difficulty": (plan.difficulty if plan else None)}, card)
            if card.hard_pass:
                return last[0], card, considered, repaired, None

        if round_index == 0 and len(rounds) == 2:
            # ONLY an identity failure buys the repair round. Anything else
            # ends here with the card it earned: Stage 15 routes identity
            # failures to this node, and re-rolling for a code that belongs
            # elsewhere is the loop this module refuses to be.
            card = last[1] if last else None
            if card is None or card.repair_code is not RepairCode.IDENTITY_DRIFT:
                break
            repaired = True

    card = last[1] if last else _card(
        (Check(name="keyframe.produced", kind=CheckKind.TECHNICAL, value=False,
               threshold=True, passed=False,
               detail="no candidate was produced"),),
        codes=(RepairCode.EMPTY_OUTPUT,))
    return None, card, considered, repaired, None


def _keyframe_limitations(run: _Run) -> None:
    if not run.seams.bound("judge_image"):
        run.limit("keyframes were accepted with NO independent judge: "
                  "identity, composition, geometry, costume, props and setting "
                  "are all unscored (Stage 15's evidence axes)")
    else:
        run.limit("keyframe judging covers identity/intent only: composition, "
                  "geometry, costume, props and setting have no evaluator on "
                  "this fleet (Stage 15)")
    if run.seams.gen_image is _live_gen_image:
        run.limit("keyframes are NOT identity-conditioned: image.generate takes "
                  "no identity pack on this fleet, so the identity refs reach "
                  "the model as text and the k97 gate, not as a lock")


def _resume_keyframes(run: _Run) -> bool:
    if not run.resuming or not run.state.intact("keyframes"):
        return False
    entry = run.state.entry("keyframes") or {}
    payload = entry.get("payload")
    if not isinstance(payload, Mapping) or not payload:
        return False
    if set(payload) != {s.segment_id for s in run.specs}:
        return False
    if digest_payload({k: payload[k] for k in sorted(payload)}) != \
            entry.get("digest"):
        return False
    run.keyframes = {k: dict(v) for k, v in payload.items()}
    return True


# --- stage 7 ---------------------------------------------------------------


def _stage_clips(run: _Run) -> PerformanceResult | None:
    started_at, t0 = _utc_now(), time.monotonic()
    seams = run.seams

    if _resume_clips(run):
        spent = time.monotonic() - t0
        _rebuild_shots(run)
        refs = tuple(str(v["ref"]) for v in run.clips.values())
        run.refs(*refs)
        run.receipt("clips", started_at=started_at, duration_s=spent,
                    warnings=("resumed from the run journal; no clip was "
                              "re-rendered",))
        run.record("clips", ok=True, resumed=True, duration_s=spent,
                   digests=tuple((sid, digest_payload(v))
                                 for sid, v in sorted(run.clips.items())),
                   artifact_refs=refs,
                   detail="resumed from the run journal (digest verified)")
        _clip_limitations(run)
        return None

    if not seams.bound("gen_clip"):
        gap = seams.gap_for("gen_clip")
        return run.fail("clips", PerformanceGap(
            stage="clips",
            diagnosis=(f"stage 7 needs the 'gen_clip' seam ({gap.capability}); "
                       f"{len(run.keyframes)} keyframe(s) were accepted and no "
                       f"video was generated"),
            repair_codes=(RepairCode.CAPABILITY_GAP,),
            capability=gap.capability, requirement=gap.requirement),
            started_at=started_at, duration_s=time.monotonic() - t0)

    threshold = float(THRESHOLDS[run.pgoal.goal.quality])
    tolerance = float(run.pgoal.speech_policy.duration_tolerance)
    failed: list[ShotResult] = []

    for spec in run.specs:
        over = run.over_budget()
        if over is not None:
            return run.fail("clips", PerformanceGap(
                stage="clips", diagnosis=over,
                repair_codes=(RepairCode.TIMEOUT,),
                capability=PERFORMANCE_CAPABILITY,
                requirement="raise the budget or lower clip_candidates",
                segment_ids=(spec.segment_id,)),
                started_at=started_at, duration_s=time.monotonic() - t0,
                failure=FailureClass.TIMEOUT)

        keyframe = run.keyframes.get(spec.segment_id) or {}
        shot, error = _clip_for(run, spec, str(keyframe.get("ref") or ""),
                                threshold, tolerance, keyframe)
        if error is not None:
            return run.fail("clips", error, started_at=started_at,
                            duration_s=time.monotonic() - t0)
        run.shots.append(shot)
        if shot.accepted:
            run.clips[spec.segment_id] = {
                "ref": shot.clip_ref, "seconds": shot.clip_seconds,
                "candidates": shot.clip_candidates,
                "repaired": shot.clip_repaired,
                "scorecard": shot.scorecard.to_dict() if shot.scorecard else None}
            run.refs(str(shot.clip_ref))
        else:
            failed.append(shot)

    if failed:
        codes: list[RepairCode] = []
        for shot in failed:
            for code in shot.repair_codes:
                if code not in codes:
                    codes.append(code)
        return run.fail("clips", PerformanceGap(
            stage="clips",
            diagnosis=("shot(s) " + ", ".join(s.segment_id for s in failed) +
                       " did not pass after one bounded repair round: " +
                       "; ".join(s.diagnosis or "no diagnosis" for s in failed)),
            repair_codes=tuple(codes),
            capability=CLIP_CAPABILITY,
            requirement=("re-render the named shot(s) only — the accepted "
                         "shots and the locked audio are untouched (Stage 17)"),
            segment_ids=tuple(s.segment_id for s in failed),
            evidence=tuple(s.diagnosis for s in failed if s.diagnosis)),
            started_at=started_at, duration_s=time.monotonic() - t0,
            failure=FailureClass.EMPTY_OUTPUT)

    run.state.drop_from("clips")
    run.state.record(
        "clips",
        digest=digest_payload({k: v for k, v in sorted(run.clips.items())}),
        payload={k: v for k, v in sorted(run.clips.items())},
        refs=[str(v["ref"]) for v in run.clips.values()])

    spent = time.monotonic() - t0
    refs = tuple(str(v["ref"]) for v in run.clips.values())
    run.receipt("clips", started_at=started_at, duration_s=spent,
                artifacts=tuple(ArtifactRef(kind=ArtifactKind.VIDEO, uri=r)
                                for r in refs))
    run.record("clips", ok=True, duration_s=spent,
               digests=tuple((sid, digest_payload(v))
                             for sid, v in sorted(run.clips.items())),
               artifact_refs=refs,
               detail=(f"{len(refs)} accepted clip(s); "
                       f"{sum(s.clip_candidates for s in run.shots)} "
                       f"candidate(s) judged; "
                       f"{sum(1 for s in run.shots if s.clip_repaired)} "
                       f"repaired"))
    _clip_limitations(run)
    return None


def _clip_for(run: _Run, spec: SegmentSpec, keyframe_ref: str,
              threshold: float, tolerance: float,
              keyframe: Mapping[str, Any]):
    """One shot: N candidates, judged; then at most ONE bounded repair round
    whose action comes from :func:`repair_decision`. Whatever the second card
    says, STANDS — that is the whole difference between a bounded repair and a
    retry loop, and it is why there is no ``while`` in this function.

    ``gen_clip(keyframe_ref, spec)`` carries no seed argument, so the candidate
    dial is the SPEC's own ``seed_base``: each candidate gets a spec whose
    seed_base is ``segment_seed(lock_digest, segment_id [+ repair salt]) +
    attempt``. Everything else about the spec — window, rubric, parents, lock
    digest — is untouched, so a candidate is the same shot at a different
    roll, never a different shot."""
    seams = run.seams
    considered = 0
    repaired = False
    last: tuple[str, "float | None", Scorecard] | None = None
    decision: "oracle_repair.RepairDecision | None" = None

    rounds = (0, 1) if run.budget.clip_repair_rounds else (0,)
    plan = _context_plan(run, spec, seams.clip_candidates)
    n_candidates = max(int(seams.clip_candidates), plan.candidates if plan else 1)
    for round_index in rounds:
        key = (f"{spec.segment_id}:{CLIP_REPAIR_SALT}" if round_index
               else spec.segment_id)
        base = segment_seed(spec.lock_digest, key, 0)
        failed_models: list[str] = []
        for attempt in range(n_candidates):
            prompt, _angle = _variant_prompt(spec.prompt, plan, attempt)
            take = replace(spec, seed_base=(base + attempt) % (2 ** 32), prompt=prompt)
            pick, sel_decision = _candidate_pick(run, CLIP_CAPABILITY, attempt, n_candidates, failed_models)
            try:
                with _pin(CLIP_CAPABILITY, pick, sel_decision):
                    produced = seams.gen_clip(keyframe_ref, take)
            except SeamUnavailable as exc:
                gap = seams.gap_for("gen_clip")
                return None, PerformanceGap(
                    stage="clips",
                    diagnosis=(f"the clip seam could not reach its backend for "
                               f"{spec.segment_id}: {exc}"),
                    repair_codes=(RepairCode.CAPABILITY_GAP,),
                    capability=gap.capability, requirement=gap.requirement,
                    segment_ids=(spec.segment_id,))
            ref, seconds = _clip_of(produced)
            considered += 1
            verdict = _judge(run, "judge_clip", ref, take, threshold,
                             RepairCode.INTENT_MISMATCH)
            card = clip_scorecard(take, verdict, clip_ref=ref,
                                  clip_seconds=seconds, threshold=threshold,
                                  duration_tolerance=tolerance)
            _attribute(ref, CLIP_CAPABILITY, pick)
            _note_verdict(ref, card)
            producer = _producer_model(ref) or pick
            if not card.hard_pass and producer:
                failed_models.append(producer)
            if repaired and decision is not None:
                # The reader of the SECOND card must see that one bounded
                # repair already happened — pass or fail (repair.py's rule).
                card = oracle_repair.annotate_repaired(card, decision)
            last = (ref, seconds, card)
            if card.hard_pass:
                return _shot(spec, True, keyframe, ref, seconds, considered,
                             repaired, card), None

        if round_index == 0 and len(rounds) == 2:
            card = last[2] if last else None
            if card is None:
                break
            decision = repair_decision(run.pgoal.goal, CLIP_CAPABILITY, card)
            if decision.action == "none":
                run.note(f"{spec.segment_id}: no bounded repair — "
                         f"{decision.rationale}")
                break
            repaired = True

    if last is not None:
        ref, seconds, card = last
    else:
        ref, seconds = None, None
        card = _card(
            (Check(name="clip.produced", kind=CheckKind.TECHNICAL, value=False,
                   threshold=True, passed=False,
                   detail="no candidate was produced"),),
            codes=(RepairCode.EMPTY_OUTPUT,))
    return _shot(spec, False, keyframe, ref, seconds, considered, repaired,
                 card), None


def _shot(spec: SegmentSpec, accepted: bool, keyframe: Mapping[str, Any],
          ref: str | None, seconds: float | None, considered: int,
          repaired: bool, card: Scorecard) -> ShotResult:
    keyframe_card = keyframe.get("scorecard")
    return ShotResult(
        segment_id=spec.segment_id, index=spec.index, accepted=accepted,
        keyframe_ref=keyframe.get("ref"), keyframe_seed=keyframe.get("seed"),
        keyframe_candidates=int(keyframe.get("candidates") or 0),
        keyframe_repaired=bool(keyframe.get("repaired")),
        keyframe_scorecard=(Scorecard.from_dict(keyframe_card)
                            if isinstance(keyframe_card, Mapping) else None),
        clip_ref=ref, clip_seconds=seconds, clip_candidates=considered,
        clip_repaired=repaired, scorecard=card,
        repair_codes=((card.repair_code,) if card.repair_code else ()),
        diagnosis=card.diagnosis or "")


def _rebuild_shots(run: _Run) -> None:
    """Reconstruct the per-shot scorecards from a resumed journal, so a resumed
    result reads exactly like a fresh one instead of losing its evidence."""
    run.shots = []
    for spec in run.specs:
        clip = run.clips.get(spec.segment_id) or {}
        card = clip.get("scorecard")
        run.shots.append(_shot(
            spec, True, run.keyframes.get(spec.segment_id) or {},
            clip.get("ref"), clip.get("seconds"),
            int(clip.get("candidates") or 0), bool(clip.get("repaired")),
            Scorecard.from_dict(card) if isinstance(card, Mapping)
            else _card((Check(name="clip.resumed", kind=CheckKind.TECHNICAL,
                              value=True, threshold=True, passed=True,
                              detail="resumed from the run journal"),))))


def _clip_limitations(run: _Run) -> None:
    if not run.seams.bound("judge_clip"):
        run.limit("clips were accepted with NO independent judge: action, "
                  "temporal artefacts, object mutation and continuity are all "
                  "unscored (Stage 17's evidence axes)")
    run.limit("no lip-sync evaluator on this fleet: mouth/audio "
              "synchronization is unscored, never verified (k121)")
    run.limit("no continuity-with-neighbours check: adjacent shots are not "
              "compared against each other's canonical state (k107)")


def _resume_clips(run: _Run) -> bool:
    if not run.resuming or not run.state.intact("clips"):
        return False
    entry = run.state.entry("clips") or {}
    payload = entry.get("payload")
    if not isinstance(payload, Mapping) or not payload:
        return False
    if set(payload) != {s.segment_id for s in run.specs}:
        return False
    if digest_payload({k: payload[k] for k in sorted(payload)}) != \
            entry.get("digest"):
        return False
    run.clips = {k: dict(v) for k, v in payload.items()}
    return True


# --- stage 8 ---------------------------------------------------------------


def _stage_assembly(run: _Run) -> PerformanceResult | None:
    started_at, t0 = _utc_now(), time.monotonic()
    seams = run.seams

    for seam in ("concat", "transcribe"):
        if seams.bound(seam):
            continue
        gap = seams.gap_for(seam)
        return run.fail("assembly", PerformanceGap(
            stage="assembly",
            diagnosis=(f"stage 8 needs the {seam!r} seam ({gap.capability}); "
                       f"{len(run.clips)} accepted clip(s) were not assembled"),
            repair_codes=(RepairCode.CAPABILITY_GAP,),
            capability=gap.capability, requirement=gap.requirement),
            started_at=started_at, duration_s=time.monotonic() - t0)

    clip_refs = tuple(str(run.clips[s.segment_id]["ref"]) for s in run.specs
                      if s.segment_id in run.clips)
    resumed = _resume_assembly(run)
    if not resumed:
        try:
            produced = seams.concat(clip_refs, run.master)
        except SeamUnavailable as exc:
            gap = seams.gap_for("concat")
            return run.fail("assembly", PerformanceGap(
                stage="assembly",
                diagnosis=f"the assembly seam failed: {exc}",
                repair_codes=(RepairCode.CAPABILITY_GAP,),
                capability=gap.capability, requirement=gap.requirement),
                started_at=started_at, duration_s=time.monotonic() - t0)
        run.video_ref = _ref_of(produced)
        if not run.video_ref:
            return run.fail("assembly", PerformanceGap(
                stage="assembly",
                diagnosis="the assembly seam produced no artifact",
                repair_codes=(RepairCode.EMPTY_OUTPUT,),
                capability=ASSEMBLE_CAPABILITY,
                requirement="check the ffmpeg concat/mux output path"),
                started_at=started_at, duration_s=time.monotonic() - t0,
                failure=FailureClass.EMPTY_OUTPUT)

    # Stage 19 — re-transcribe the FINAL output and check every locked line.
    try:
        words = tuple(seams.transcribe(run.video_ref) or ())
    except SeamUnavailable as exc:
        gap = seams.gap_for("transcribe")
        return run.fail("assembly", PerformanceGap(
            stage="assembly",
            diagnosis=(f"the final round-trip transcription failed, so line "
                       f"coverage is UNVERIFIED and the deliverable is not "
                       f"accepted: {exc}"),
            repair_codes=(RepairCode.CAPABILITY_GAP,),
            capability=gap.capability, requirement=gap.requirement),
            started_at=started_at, duration_s=time.monotonic() - t0)

    lines_check = speech.check_lines_present(
        [line.text for line in run.pgoal.lines], words)
    total_clip_seconds = sum(
        float(v["seconds"]) for v in run.clips.values()
        if v.get("seconds") is not None) or None
    duration_check = speech.check_duration_fit(
        audio_seconds=run.master.total_seconds, shot_seconds=total_clip_seconds,
        tolerance=float(run.pgoal.speech_policy.duration_tolerance))

    checks: list[Check] = [
        Check(name="assembly.produced", kind=CheckKind.TECHNICAL,
              value=bool(run.video_ref), threshold=True,
              passed=bool(run.video_ref),
              detail=f"assembled deliverable {run.video_ref}"),
        lines_check,
        duration_check,
    ]
    for shot in run.shots:
        checks.append(Check(
            name=f"shot.{shot.segment_id}", kind=CheckKind.INTENT,
            value="accepted" if shot.accepted else "rejected",
            threshold="accepted", passed=shot.accepted,
            detail=(shot.diagnosis or
                    f"clip {shot.clip_ref} accepted after "
                    f"{shot.clip_candidates} candidate(s)")))

    codes: list[RepairCode] = []
    if not lines_check.passed:
        codes.append(RepairCode.LINE_OMITTED)
    if not duration_check.passed:
        codes.append(RepairCode.SHOT_TOO_SHORT)
    for shot in run.shots:
        for code in shot.repair_codes:
            if code not in codes:
                codes.append(code)

    judge_results = tuple(j for shot in run.shots
                          for j in (shot.scorecard.judge_results
                                    if shot.scorecard else ()))
    run.final_card = _card(
        checks, judge_results=judge_results, codes=codes,
        recommended=("re-render only the shot(s) whose check failed; the "
                     "locked audio timeline is authoritative and is never "
                     "retimed to match the picture"))

    if not resumed:
        run.state.drop_from("assembly")
        run.state.record("assembly", digest=digest_payload(
            {"video_ref": run.video_ref,
             "clips": [str(c) for c in clip_refs]}),
            payload={"video_ref": run.video_ref,
                     "clips": [str(c) for c in clip_refs]},
            refs=[run.video_ref])

    run.refs(run.video_ref)
    spent = time.monotonic() - t0
    run.receipt("assembly", started_at=started_at, duration_s=spent,
                artifacts=(ArtifactRef(kind=ArtifactKind.VIDEO,
                                       uri=run.video_ref),),
                failure=None if run.final_card.hard_pass
                else FailureClass.EMPTY_OUTPUT,
                log=((run.final_card.diagnosis,)
                     if run.final_card.diagnosis else ()))
    run.record("assembly", ok=run.final_card.hard_pass, resumed=resumed,
               duration_s=spent,
               digests=(("deliverable", digest_payload(
                   {"video_ref": run.video_ref,
                    "clips": [str(c) for c in clip_refs]})),
                        ("scorecard", digest_payload(
                            run.final_card.to_dict()))),
               artifact_refs=(run.video_ref,),
               detail=(f"{lines_check.value}/{lines_check.threshold} locked "
                       f"line(s) present in the final round-trip transcript"))
    if not words:
        run.limit("the final round-trip transcript carried no words, so line "
                  "coverage FAILED rather than passing unverified")
    run.limit("assembly is the Stage 18 CUT only: no director/editor pass, no "
              "bounded retiming, no mix, no colour finishing (k108)")
    return None


def _resume_assembly(run: _Run) -> bool:
    if not run.resuming or not run.state.intact("assembly"):
        return False
    entry = run.state.entry("assembly") or {}
    payload = entry.get("payload")
    if not isinstance(payload, Mapping):
        return False
    ref = payload.get("video_ref")
    if not ref or digest_payload(dict(payload)) != entry.get("digest"):
        return False
    expected = [str(run.clips[s.segment_id]["ref"]) for s in run.specs
                if s.segment_id in run.clips]
    if [str(c) for c in (payload.get("clips") or ())] != expected:
        return False
    run.video_ref = str(ref)
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ref_of(produced: Any) -> str:
    """Whatever a generation seam returned -> its artifact reference. Accepts a
    bare string, a mapping with ``ref``/``uri``/``path``, or an object with one
    of those attributes; anything else is a wiring bug and reads as empty,
    which fails the card's ``produced`` check rather than crashing a stage."""
    if produced is None:
        return ""
    if isinstance(produced, str):
        return produced
    if isinstance(produced, (tuple, list)) and produced:
        return _ref_of(produced[0])
    if isinstance(produced, Mapping):
        for key in ("ref", "uri", "path", "file"):
            if produced.get(key):
                return str(produced[key])
        return ""
    for attr in ("ref", "uri", "path", "file"):
        value = getattr(produced, attr, None)
        if value:
            return str(value)
    return ""


def _clip_of(produced: Any) -> tuple[str, float | None]:
    """``(clip_ref, duration_s | None)``. A seam that reports no duration is
    UNSCORED on duration fit, never assumed to fit."""
    seconds: float | None = None
    if isinstance(produced, (tuple, list)) and len(produced) >= 2:
        try:
            seconds = float(produced[1])
        except (TypeError, ValueError):
            seconds = None
    elif isinstance(produced, Mapping):
        for key in ("seconds", "duration_s", "duration"):
            if produced.get(key) is not None:
                try:
                    seconds = float(produced[key])
                except (TypeError, ValueError):
                    seconds = None
                break
    else:
        for attr in ("seconds", "duration_s", "duration"):
            value = getattr(produced, attr, None)
            if value is not None:
                try:
                    seconds = float(value)
                except (TypeError, ValueError):
                    seconds = None
                break
    return _ref_of(produced), seconds


def segment_context(spec: SegmentSpec, pgoal: "PerformanceGoal | None" = None) -> dict[str, Any]:
    """Adapt a locked SegmentSpec into the mapping ``prompt_compiler`` reads.
    Everything comes from the spec (and the goal's static production notes);
    nothing from a sibling."""
    cont = spec.continuity
    shot = spec.shot
    before = dict(cont.state_before) if cont is not None else {}
    after = dict(cont.state_after) if cont is not None else {}
    chars = before.get("characters") or after.get("characters") or ()
    if isinstance(chars, Mapping):
        chars = tuple(chars.keys())
    return {
        "segment_id": spec.segment_id,
        "characters": list(chars) if isinstance(chars, (list, tuple)) else ([chars] if chars else []),
        "scene": spec.prompt,
        "blocking": shot.blocking or (pgoal.blocking if pgoal else None) or "",
        "lighting": shot.lighting or (pgoal.lighting if pgoal else None) or "",
        "camera": dict(shot.camera) if shot.camera else (dict(pgoal.camera) if (pgoal and pgoal.camera) else {}),
        "props": before.get("props") or after.get("props") or (),
        "state_before": before,
        "state_after": after,
        "identity_constraints": ", ".join(spec.identity_refs) if spec.identity_refs else "",
        "negative_constraints": spec.negative_prompt or "",
        "dialogue": list(spec.line_ids),
        "audio_window": {"start_s": spec.start_s, "end_s": spec.end_s, "lines": list(spec.line_ids)},
        "duration_s": spec.duration_s,
        "tone": _tone_operator(spec.tone),
        "spatial_manifest": _spatial_manifest_payload(spec),
        "spatial_ref": spec.spatial_ref,
    }


def _spatial_manifest_payload(spec: SegmentSpec) -> Any:
    """The Fold-1 payload the prompt compiler reads: the resolved
    ``SpatialSource`` dict (manifest + camera track + entity tracks) when the
    ref resolves, else the bare ref string (truthy: there IS an authority, we
    just could not expand it here), else ``None``."""
    if not spec.spatial_ref:
        return None
    try:
        from hugpy_oracle.segments import spatial_source_for
        src = spatial_source_for(spec.spatial_ref, spec.segment_id)
    except Exception:  # noqa: BLE001 — a stale file is reported by the recipe's spatial node, not here
        src = None
    return src.to_dict() if src is not None else spec.spatial_ref


def _tone_operator(tone_unit: float) -> float:
    """The ONE crossing from the locked unit-scale tone to the 0–10 operator
    scale the compiler/spatial layer speak (oracle/tone_scale.py)."""
    from hugpy_oracle.tone_scale import to_operator
    return to_operator(tone_unit)


def _context_plan(run: "_Run", spec: SegmentSpec, floor: int) -> Any:
    """The per-segment context plan (difficulty -> candidates/angles). Never
    raises; a compiler fault means 'no plan' and the seam floor applies."""
    try:
        from hugpy_oracle.prompt_compiler import compile_context
        quality = getattr(run.pgoal.goal.quality, "value", "balanced")
        cap = max(int(floor), {"preview": 1, "balanced": 4, "best": 8}.get(quality, 3))
        return compile_context(segment_context(spec, run.pgoal), goal=run.pgoal.goal,
                               max_candidates=cap, eligible_models=_eligible_model_count(run))
    except Exception as exc:  # noqa: BLE001
        run.limit(f"prompt compiler unavailable for {spec.segment_id}: {exc}")
        return None


def _eligible_model_count(run: "_Run") -> int:
    try:
        view = (run.seams.catalog_view() if run.seams.catalog_view else {}) or {}
        entry = view.get(KEYFRAME_CAPABILITY) if isinstance(view, Mapping) else None
        ids = entry.get("model_ids") if isinstance(entry, Mapping) else getattr(entry, "model_ids", None)
        return max(1, len(ids or ()))
    except Exception:  # noqa: BLE001
        return 1


def _variant_prompt(base_prompt: str, plan: Any, attempt: int) -> tuple[str, str | None]:
    """Candidate 0 is the locked prompt verbatim (the sibling-invariant
    artifact). Later candidates append the plan's angle emphasis — derived from
    the same locked spec, never from another take or segment."""
    if plan is None or attempt == 0 or not plan.variants:
        return base_prompt, None
    variant = plan.variants[attempt % len(plan.variants)]
    return f"{base_prompt}\n[{variant.angle.upper()} PRIORITY] {variant.emphasis}", variant.angle


def _candidate_pick(run: "_Run", capability: str, attempt: int, candidates: int,
                    exclude: Sequence[str]) -> tuple[str | None, dict[str, Any] | None]:
    """Per-candidate model selection for a live seam call: spread across
    eligible models, exclude the ones that already failed this shot. (None,
    None) = no opinion, the seam's router default stands."""
    try:
        from hugpy_oracle import selection
        return selection.requested_model_for(run.pgoal.goal, capability, candidate_index=attempt,
                                             candidates=candidates, exclude=tuple(exclude))
    except Exception:  # noqa: BLE001
        return None, None


def _pin(capability: str, model_id: str | None, decision: Mapping[str, Any] | None):
    try:
        from hugpy_oracle import selection
        return selection.pinned(capability, model_id, decision)
    except Exception:  # noqa: BLE001
        import contextlib
        return contextlib.nullcontext()


def _attribute(ref: str | None, capability: str, pick: str | None) -> None:
    """If the seam did not register the producer (fakes / legacy seams), the
    pinned pick is the best available attribution — recorded as such."""
    try:
        from hugpy_oracle import selection
        if ref and pick and selection.producer_of(ref) is None:
            selection.remember_producer(ref, capability, pick)
    except Exception:  # noqa: BLE001
        pass


def _producer_model(ref: str | None) -> str | None:
    try:
        from hugpy_oracle import selection
        prod = selection.producer_of(ref)
        return prod[1] if prod else None
    except Exception:  # noqa: BLE001
        return None


def _note_verdict(ref: str, card: Scorecard) -> None:
    try:
        from hugpy_oracle import selection
        selection.note_verdict_for_ref(ref, hard_pass=bool(card.hard_pass),
                                       repair_code=card.repair_code)
    except Exception:  # noqa: BLE001
        pass


def _judge(run: _Run, seam: str, ref: str, spec: SegmentSpec,
           threshold: float, default_code: RepairCode) -> Verdict:
    """Run a judge seam and normalize its answer. A judge that RAISES is
    recorded as unavailable/unscored (movie semantics: the vision plane being
    down must not fail a shot) — it never becomes a silent pass with full
    confidence, because ``scored=False`` lowers the card and adds a
    limitation."""
    fn = getattr(run.seams, seam)
    if fn is None:
        return coerce_verdict(None, threshold=threshold,
                              default_code=default_code,
                              judge=SEAM_CAPABILITY[seam])
    try:
        raw = fn(ref, spec)
    except Exception as exc:                       # noqa: BLE001
        run.note(f"{seam} raised for {spec.segment_id} "
                 f"({type(exc).__name__}: {exc}); the candidate is UNSCORED")
        return coerce_verdict(None, threshold=threshold,
                              default_code=default_code,
                              judge=SEAM_CAPABILITY[seam])
    return coerce_verdict(raw, threshold=threshold, default_code=default_code,
                          judge=SEAM_CAPABILITY[seam])


def _finish_limitations(run: _Run) -> None:
    """The limitations that are about the RUN as a whole rather than one
    stage. Always non-empty: a slice this early with nothing to disclose would
    be the least believable result this module could return."""
    run.limit("this is a FAT orchestrator, not the k111 DAG runtime: resume is "
              "per-stage, there are no leases, and a cancelled run resumes from "
              "its last completed stage, not from mid-stage")
    if run.registry_version is None:
        run.limit("no registry_version was recorded, so these receipts say what "
                  "ran but not what it was chosen from (k105)")
    if run.seams.judge_image is not None and \
            run.seams.gen_image is not None and \
            run.seams.judge_image is run.seams.gen_image:
        run.limit("the keyframe judge and the keyframe generator are the SAME "
                  "seam — a generator grading its own work (invariant 11, k115)")


_STAGE_FN: dict[str, Callable[[_Run], "PerformanceResult | None"]] = {
    "authority": _stage_authority,
    "snapshot": _stage_snapshot,
    "audio": _stage_audio,
    "lock": _stage_lock,
    "segments": _stage_segments,
    "keyframes": _stage_keyframes,
    "clips": _stage_clips,
    "assembly": _stage_assembly,
}


__all__ = [
    # vocabulary
    "ASSEMBLE_CAPABILITY", "CLIP_CAPABILITY", "DEFAULT_CLIP_CANDIDATES",
    "DEFAULT_KEYFRAME_CANDIDATES", "DEFAULT_RUBRIC", "DEFAULT_TTS_CANDIDATES",
    "GATED_CAPABILITIES", "KEYFRAME_CAPABILITY", "LIVE_SEAM_BINDINGS",
    "LIVE_SEAM_GAPS", "make_live_synth",
    "PERFORMANCE_CAPABILITY", "RESUMABLE_STAGES", "SEAM_CAPABILITY",
    "SEAM_NAMES", "STAGES", "STAGE_CAPABILITY", "TTS_CAPABILITY",
    # errors
    "PerformanceError", "SeamUnavailable",
    # seams + request
    "PerformanceBudget", "PerformanceGoal", "PerformanceSeams", "SeamGap",
    "default_seams",
    # results
    "PerformanceGap", "PerformanceResult", "ShotResult", "StageRecord",
    "Verdict",
    # journal
    "RunState", "default_run_root", "derive_run_id", "run_dir", "state_path",
    # the run + its pieces
    "authority_requirements", "check_authority", "clip_scorecard",
    "coerce_verdict", "keyframe_scorecard", "repair_decision",
    "run_performance",
]
