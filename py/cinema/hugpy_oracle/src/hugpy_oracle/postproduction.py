"""Post-production plan + footage-vs-spec evaluation (k108) — doc Stage 18/19.

The script-first brief's Phase 3, as typed artifacts::

    Create a structured post-production plan containing: assembly order,
    preferred takes, edit decision list, transition instructions, pacing and
    timing, sound effects and ambience, dialogue alignment, music cues,
    color-grading targets, continuity corrections, required regeneration notes.

    Use vision-language evaluation where available to compare generated footage
    against its shot specification, screenplay event, and continuity state. A
    rejected shot should be regenerated from the same canonical segment
    specification with documented corrections — not from the rejected prompt or
    footage description alone.

That last sentence is the reason :class:`RegenerationNote` carries a
``spec_digest`` and a ``correction`` and only ever carries the rejected take as
EVIDENCE (``rejected_take_ref``). The regeneration source is the canonical
``SegmentSpec`` that produced the take in the first place; the rejected footage
is what the correction is written ABOUT, never what it is written FROM.

WHAT IS HERE

``Take`` / ``EditDecision`` / ``EditDecisionList``
    The assembly cut as content-addressed artifacts. One preferred take per
    segment, gapless order, a closed transition vocabulary, in/out points
    bounded by the take's own measured duration.

``PostProductionPlan``
    Every artifact the brief's Phase 3 list names, in one immutable object,
    plus ``regeneration_notes`` for the segments that have no accepted take.
    A segment with no footage is NEVER silently dropped from the plan: it
    appears as an explicit note naming a ``RepairCode`` and a correction.

``build_postproduction_plan``
    Deterministic assembly from a shot plan (or compiled segment specs, or a
    run state that carries them) plus the takes that came back. Same inputs,
    same plan, same digest.

``evaluate_take``
    ONE take against ITS shot specification: the k90c judge-rubric pattern
    (``evaluation.py`` — imported and called, never re-implemented) widened
    with a shot-spec rubric built from the requested action / setting /
    characters and the continuity ``state_before``/``state_after``, k98 speech
    evidence when a round-trip transcript is available, and the duration fit
    that Stage 8 fixes the direction of. Failures map to ``RepairCode``s.

``bind_live_judge``
    The live judge over the catalog's own resolution — ``image.understand``
    for stills, ``video.*`` attempted first for clips (and honestly
    unavailable on a fleet with no clip-level evaluator, per k106's seam
    table), with a ``frames`` seam for sampling a still out of a clip.

``final_consistency_report``
    Stage 19's whole-result pass: does the cut cover every locked segment, did
    the dialogue survive the round trip, does any shot's screen time deviate
    from its locked window, and is there a color target for every scene whose
    shot plan declared one.

INVARIANTS KEPT FROM THE REST OF THE TREE

* **The generator never judges itself.** ``evaluate_take`` raises
  :class:`JudgeConflict` when the judge model it is told about equals the
  take's ``generator_model`` (doc §9: "do not ... let the generator approve
  itself"). It is a typed refusal, not a warning: a self-graded take is worse
  than an ungraded one because it looks like evidence.
* **Unscored is not passed.** A judge that is unreachable, a transcript that
  was never taken, a duration nobody measured — each is recorded with
  ``speech.UNSCORED_PREFIX`` and counted OUT of ``Scorecard.confidence``
  (k98's rule, k90c's degradation: "unscored, keep").
* **Canonical JSON, content addressing, frozen slots.** Every artifact here
  extends ``production.ContentAddressed``, so a plan digest is comparable to a
  lock digest without a conversion step.
* **No new ``RepairCode`` members.** Everything emitted here already exists in
  ``contracts.RepairCode``.

Offline by construction: stdlib + this package. The judge and the transcriber
are injected callables, so the whole module is testable without a GPU, a
worker, a registry or a network.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from hugpy_oracle import evaluation, speech
from hugpy_oracle.audio_master import AudioMaster, scorecard_digest
from hugpy_oracle.contracts import (
    Check,
    CheckKind,
    GoalSpec,
    JudgeResult,
    QualityProfile,
    RepairCode,
    Scorecard,
)
from hugpy_oracle.plan import FrozenParams
from hugpy_oracle.production import (
    ContentAddressed,
    ContinuityState,
    ProductionError,
    ProductionLock,
    ShotPlan,
    ShotPlanEntry,
)
# BORROWED, not re-copied — the same discipline k110 wrote down: ``production``
# already owns the exact versions of these helpers that k104's artifacts
# validate with, and a private copy in yet another module is how a tree ends up
# with five subtly different "non-empty string" rules. A test asserts these are
# the same objects.
from hugpy_oracle.production import _EPS as EPS
from hugpy_oracle.spatial import (
    CANONICAL,
    CameraSpec,
    CoordinateSystem,
    DriftThresholds,
    convert_points,
)
from hugpy_oracle.production import _q as quantize
from hugpy_oracle.production import _require_non_negative as require_non_negative
from hugpy_oracle.production import _require_text as require_text
from hugpy_oracle.production import _str_tuple as str_tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Closed vocabularies and tunables — named, never magic.
# ---------------------------------------------------------------------------

#: The transition instructions an ``EditDecision`` may carry. Closed on purpose:
#: an editor's vocabulary that grows by typing is a vocabulary the assembler
#: cannot execute. ``continuous`` is not a transition between two shots so much
#: as the ABSENCE of a cut — the screenplay's ``CONTINUOUS:`` carried into the
#: cut (k110's ``CONTINUOUS_TRANSITION``), which is why it takes no duration.
TRANSITIONS: tuple[str, ...] = ("cut", "dissolve", "fade", "continuous")

#: The transitions that occupy no time. A ``cut`` with a duration is a dissolve
#: nobody named; a ``dissolve`` with no duration is a cut nobody meant.
INSTANT_TRANSITIONS: tuple[str, ...] = ("cut", "continuous")

#: doc Stage 18 step 3 — "ambience, foley, music". Sound effects, ambience and
#: foley are the three the brief names; music has its own cue type because it
#: is mixed, licensed and graded differently.
SOUND_KINDS: tuple[str, ...] = ("sfx", "ambience", "foley")

#: How a music cue functions in the mix. ``source`` is diegetic (it exists in
#: the world of the film), ``score`` is not, ``sting`` is a punctuation hit.
MUSIC_ROLES: tuple[str, ...] = ("score", "source", "sting")

#: Whole-token cues that mean a shot DECLARED a look, i.e. that the finishing
#: pass owes this scene a color-grading target. Matched against the shot plan's
#: own ``lighting`` line and acceptance ``rubric`` — never against free prose
#: elsewhere, for k110's reason: a bible that invented a fact because a verb
#: suggested it is fabricating evidence for the judge.
COLOR_CUES: frozenset[str] = frozenset({
    "color", "colors", "colour", "colours", "grade", "graded", "grading",
    "palette", "lut", "warm", "cool", "teal", "amber", "golden", "sepia",
    "monochrome", "desaturated", "saturation", "tint", "tinted", "contrast",
})

#: Seconds a shot's screen time may deviate from its locked window before the
#: whole-result pacing check calls it an outlier. Larger than k98's retiming
#: slack (``speech.DEFAULT_DURATION_TOLERANCE``, 0.15 s) on purpose: that one
#: bounds a SYNC decision on one shot, this one bounds an EDITORIAL judgement
#: over the whole cut, where a quarter second is still invisible.
DEFAULT_PACING_TOLERANCE_S: float = 0.25

#: The check name -> repair code table for ``evaluate_take``. Same shape as
#: ``speech.SPEECH_REPAIR`` and ``scorecard._FAILURE_REPAIR``: one table, one
#: story. Every code already exists in ``contracts.RepairCode``; k108 defines
#: no new enum member.
TAKE_REPAIR: dict[str, RepairCode] = {
    "shot.intent":          RepairCode.INTENT_MISMATCH,
    "shot.action":          RepairCode.ACTION_MISSING,
    "shot.identity":        RepairCode.IDENTITY_DRIFT,
    "shot.temporal":        RepairCode.TEMPORAL_ARTIFACT,
    "speech.lines_present": RepairCode.LINE_OMITTED,
    "sync.duration_fit":    RepairCode.SHOT_TOO_SHORT,
}

# Which failure wins when several fire at once. "Repair the largest thing
# first" (k98's rule): a shot that is not the requested shot invalidates
# everything downstream of it; a missing ACTION is the next largest; identity
# and dialogue are inside an otherwise correct shot; a temporal artifact is a
# reroll; a duration miss is a WINDOW fix, not a footage fix.
_TAKE_PRIORITY: tuple[str, ...] = (
    "shot.intent", "shot.action", "shot.identity", "speech.lines_present",
    "shot.temporal", "sync.duration_fit")

#: The check name -> repair code table for ``final_consistency_report``.
REPORT_REPAIR: dict[str, RepairCode] = {
    "assembly.coverage":            RepairCode.EMPTY_OUTPUT,
    "assembly.regeneration_open":   RepairCode.EMPTY_OUTPUT,
    "speech.lines_present":         RepairCode.LINE_OMITTED,
    "pacing.window_fit":            RepairCode.SHOT_TOO_SHORT,
    "audio.bed_fit":                RepairCode.SHOT_TOO_SHORT,
    "color.targets":                RepairCode.INTENT_MISMATCH,
}

_REPORT_PRIORITY: tuple[str, ...] = (
    "assembly.coverage", "assembly.regeneration_open", "speech.lines_present",
    "pacing.window_fit", "audio.bed_fit", "color.targets")

#: What an operator should DO about each code, in this module's terms. The
#: wording matters: none of these says "re-prompt from the footage".
RECOMMENDED_REPAIR: dict[RepairCode, str] = {
    RepairCode.INTENT_MISMATCH: (
        "regenerate this segment from its canonical SegmentSpec with the "
        "documented correction; do not re-prompt from the rejected footage"),
    RepairCode.ACTION_MISSING: (
        "regenerate from the same SegmentSpec with the requested action stated "
        "explicitly in the shot prose (the spec's blocking is authoritative)"),
    RepairCode.IDENTITY_DRIFT: (
        "regenerate from the same SegmentSpec with the authorized identity "
        "references; never accept a lookalike"),
    RepairCode.TEMPORAL_ARTIFACT: (
        "reroll this shot from the same SegmentSpec at a bumped seed — a "
        "temporal artifact is a sampling failure, not a specification error"),
    RepairCode.LINE_OMITTED: (
        "re-synthesize/re-render the omitted line(s) only — the locked "
        "dialogue is authoritative; do not rewrite the script to match"),
    RepairCode.SHOT_TOO_SHORT: (
        "extend the shot window to the locked audio (doc Stage 8: the audio "
        "timeline precedes shot timing) via ProductionLock.revise, then "
        "re-render that shot only"),
    RepairCode.EMPTY_OUTPUT: (
        "this segment has no usable footage: regenerate it from its canonical "
        "SegmentSpec before the cut can be called complete"),
}

_UNSCORED = speech.UNSCORED_PREFIX

# Whole-token matcher, the ``shot_intent._has_cue`` idiom k110 mirrors: cues are
# recognized as WORDS, so "discolored" never reads as "color".
_TOKENS = re.compile(r"[A-Za-z][A-Za-z'-]*")


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class PostProductionError(ProductionError):
    """Base for every refusal in this module (a ``ValueError`` by inheritance,
    so a caller that only catches ValueError still catches these)."""


class EDLRefused(PostProductionError):
    """The edit decision list does not describe an executable assembly."""


class PlanRefused(PostProductionError):
    """The post-production plan cannot be built from what was supplied."""


class JudgeConflict(PostProductionError):
    """The judge is the generator. doc §9: "do not reduce quality to one VLM
    opinion or let the generator approve itself." A self-graded take is worse
    than an ungraded one — it looks like evidence."""

    def __init__(self, message: str, model: str = "",
                 segment_id: str = "") -> None:
        super().__init__(message)
        self.model = model
        self.segment_id = segment_id


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(m.group(0).casefold() for m in _TOKENS.finditer(str(text or "")))


def has_color_cue(*texts: Any) -> bool:
    """Whether any of ``texts`` declares a look this cut owes a grade target.

    Whole-token match against :data:`COLOR_CUES`. A shot that says "warm
    practicals, amber spill" declares one; a shot that says "she is
    discolored" does not."""
    for text in texts:
        if not text:
            continue
        if isinstance(text, (list, tuple)):
            if has_color_cue(*text):
                return True
            continue
        if any(tok in COLOR_CUES for tok in _tokens(text)):
            return True
    return False


def _opt_text(value: Any, what: str) -> str | None:
    if value is None:
        return None
    return require_text(value, what)


def _opt_seconds(value: Any, what: str) -> float | None:
    if value is None:
        return None
    return quantize(require_non_negative(value, what))


def _coerce_repair(value: Any, what: str) -> RepairCode:
    if isinstance(value, RepairCode):
        return value
    try:
        return RepairCode(str(value))
    except ValueError as exc:
        raise PostProductionError(
            f"{what} must be a contracts.RepairCode (or its value), got "
            f"{value!r}") from exc


def _mapping_of_text(value: Any) -> dict[str, str]:
    """``{line_id: text}`` from a mapping, a sequence of k102 ``Line``s, a
    sequence of pairs, or an object carrying ``.lines`` (a ``DialogueTimeline``
    or a ``Screenplay`` scene). Anything unrecognized reads as empty rather
    than raising: a missing line text is evidence that is absent, not a crash."""
    if value is None:
        return {}
    lines = getattr(value, "lines", None)
    if lines is not None and not isinstance(value, (Mapping, str)):
        value = lines
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    out: dict[str, str] = {}
    for item in value or ():
        line_id = getattr(item, "line_id", None)
        text = getattr(item, "text", None)
        if line_id is not None and text is not None:
            out[str(line_id)] = str(text)
            continue
        if isinstance(item, Mapping):
            if "line_id" in item:
                out[str(item["line_id"])] = str(item.get("text", ""))
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2:
            out[str(item[0])] = str(item[1])
    return out


# ---------------------------------------------------------------------------
# ShotBrief — the ONE shape this module reads a shot specification through.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShotBrief:
    """What post-production needs to know about one planned shot.

    A normalizer, not a new artifact: :func:`shot_brief` builds it from a k104
    ``SegmentSpec``, a k104 ``ShotPlanEntry``, a k110 ``ShotDesign`` or a plain
    mapping, so every function here reads ONE shape and none of them grows a
    second opinion about where ``blocking`` lives.

    ``spec_digest`` is the canonical segment specification a rejected shot is
    regenerated FROM — present only when the brief came from a ``SegmentSpec``,
    because that is the only one of the four that IS the canonical spec."""

    segment_id: str
    index: int = 0
    scene_ref: str | None = None
    line_ids: tuple[str, ...] = ()
    start_s: float = 0.0
    end_s: float = 0.0
    camera: Mapping[str, Any] = field(default_factory=FrozenParams)
    blocking: str = ""
    lighting: str = ""
    rubric: tuple[str, ...] = ()
    prompt: str = ""
    identity_refs: tuple[str, ...] = ()
    spec_digest: str | None = None
    continuity: ContinuityState | None = None
    spatial_ref: str | None = None

    def __post_init__(self) -> None:
        require_text(self.segment_id, "ShotBrief.segment_id")
        object.__setattr__(self, "line_ids",
                           str_tuple(self.line_ids, "ShotBrief.line_ids"))
        object.__setattr__(self, "start_s", quantize(self.start_s))
        object.__setattr__(self, "end_s", quantize(self.end_s))
        if self.end_s < self.start_s:
            raise PostProductionError(
                f"ShotBrief({self.segment_id}) ends before it starts: "
                f"{self.end_s} < {self.start_s}")
        object.__setattr__(self, "camera", FrozenParams(self.camera))
        for name in ("blocking", "lighting", "prompt"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        object.__setattr__(self, "rubric",
                           str_tuple(self.rubric, "ShotBrief.rubric"))
        object.__setattr__(self, "identity_refs",
                           str_tuple(self.identity_refs,
                                     "ShotBrief.identity_refs"))
        object.__setattr__(self, "scene_ref",
                           _opt_text(self.scene_ref, "ShotBrief.scene_ref"))
        object.__setattr__(self, "spec_digest",
                           _opt_text(self.spec_digest, "ShotBrief.spec_digest"))
        object.__setattr__(self, "spatial_ref",
                           _opt_text(self.spatial_ref, "ShotBrief.spatial_ref"))

    @property
    def duration_s(self) -> float:
        """The LOCKED window length — what this shot owes the timeline."""
        return quantize(self.end_s - self.start_s)

    @property
    def scene_key(self) -> str:
        """The key a color target is filed under: the scene when the shot knows
        one, the segment otherwise. A grade is a SCENE decision (doc Stage 18
        step 4: "enforce cross-shot consistency"), so shots that share a scene
        share a target."""
        return self.scene_ref or self.segment_id

    @property
    def declares_color(self) -> bool:
        """Whether the shot plan asked for a look (see :func:`has_color_cue`)."""
        return has_color_cue(self.lighting, self.rubric)

    @property
    def characters(self) -> tuple[str, ...]:
        """Who this shot is supposed to contain, from the continuity state's
        ``present``/``characters`` (k110's ``STATE_KEYS``) and falling back to
        the identity references the spec locked."""
        for key in ("present", "characters"):
            value = (self.continuity.state_before.get(key)
                     if self.continuity else None)
            if isinstance(value, str) and value.strip():
                return (value,)
            if isinstance(value, (list, tuple)) and value:
                return tuple(str(v) for v in value)
        return self.identity_refs

    @property
    def action(self) -> str:
        """The requested ACTION: Stage 9's *block* step, falling back to the
        production prompt when a shot carries no blocking line."""
        return self.blocking or self.prompt

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id, "index": self.index,
            "scene_ref": self.scene_ref, "line_ids": list(self.line_ids),
            "start_s": self.start_s, "end_s": self.end_s,
            "camera": FrozenParams(self.camera).to_dict(),
            "blocking": self.blocking, "lighting": self.lighting,
            "rubric": list(self.rubric), "prompt": self.prompt,
            "identity_refs": list(self.identity_refs),
            "spec_digest": self.spec_digest,
            "continuity": (self.continuity.to_dict()
                           if self.continuity is not None else None),
            "spatial_ref": self.spatial_ref,
        }


def shot_brief(obj: Any, index: int = 0) -> ShotBrief:
    """Normalize one shot-spec-like object into a :class:`ShotBrief`.

    Accepts (duck-typed, so this module imports nothing it does not already
    depend on): a ``ShotBrief`` (returned as-is), a k104 ``SegmentSpec``, a
    k110 ``ShotDesign`` (anything with ``to_entry`` and ``scene_id``), a k104
    ``ShotPlanEntry``, or a mapping of the same field names."""
    if isinstance(obj, ShotBrief):
        return obj
    if isinstance(obj, Mapping):
        data = dict(obj)
        data.setdefault("index", index)
        known = {f for f in ShotBrief.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise PostProductionError(
                f"shot_brief mapping carries unknown key(s) {unknown}; a key "
                f"nobody reads is a silently dropped direction")
        return ShotBrief(**data)

    shot = getattr(obj, "shot", None)          # SegmentSpec
    if isinstance(shot, ShotPlanEntry):
        window = getattr(obj, "audio_window", None) or shot.window
        return ShotBrief(
            segment_id=obj.segment_id,
            index=int(getattr(obj, "index", index) or 0),
            scene_ref=getattr(obj, "scene_ref", None),
            line_ids=tuple(window[2]), start_s=window[0], end_s=window[1],
            camera=shot.camera, blocking=shot.blocking or "",
            lighting=shot.lighting or "",
            rubric=tuple(getattr(obj, "rubric", ()) or shot.rubric),
            prompt=str(getattr(obj, "prompt", "") or ""),
            identity_refs=tuple(getattr(obj, "identity_refs", ()) or ()),
            spec_digest=getattr(obj, "digest", None),
            continuity=getattr(obj, "continuity", None),
            spatial_ref=getattr(obj, "spatial_ref", None))

    to_entry = getattr(obj, "to_entry", None)  # k110 ShotDesign
    if callable(to_entry) and hasattr(obj, "scene_id"):
        entry = to_entry()
        brief = shot_brief(entry, index)
        return replace(brief, scene_ref=str(obj.scene_id))

    if isinstance(obj, ShotPlanEntry):
        return ShotBrief(
            segment_id=obj.segment_id, index=index,
            line_ids=obj.line_ids, start_s=obj.start_s, end_s=obj.end_s,
            camera=obj.camera, blocking=obj.blocking or "",
            lighting=obj.lighting or "", rubric=obj.rubric)

    raise PostProductionError(
        f"shot_brief cannot read a {type(obj).__name__}: pass a SegmentSpec, a "
        f"ShotPlanEntry, a k110 ShotDesign, a ShotBrief or a mapping")


def shot_briefs(source: Any) -> tuple[ShotBrief, ...]:
    """Every shot of a production, in ASSEMBLY ORDER.

    The order is the shot plan's own order, which k104 already guarantees is
    timeline order (``ShotPlan`` refuses a list out of order) — this function
    re-sorts nothing, because a plan whose order is wrong is a bug upstream,
    not a sorting problem here."""
    if source is None:
        return ()
    if isinstance(source, ProductionLock):
        raise PlanRefused(
            "a ProductionLock carries DIGESTS, not shots — pass the ShotPlan "
            "it locked, the compiled SegmentSpecs, or a run state carrying "
            "them (and pass the lock itself as lock=... to bind the plan to it)")
    if isinstance(source, ShotPlan):
        return tuple(shot_brief(e, i) for i, e in enumerate(source.entries))

    segments = getattr(source, "segments", None)          # PerformanceResult
    if segments:
        return tuple(shot_brief(s, i) for i, s in enumerate(segments))
    plan = getattr(source, "shot_plan", None)             # a state carrying one
    if plan is not None:
        return shot_briefs(plan)
    designs = getattr(source, "designs", None)            # k110 ShotPlanDraft
    if designs:
        return tuple(shot_brief(d, i) for i, d in enumerate(designs))
    entries = getattr(source, "entries", None)            # a ShotPlan-alike
    if entries is not None and not isinstance(source, (str, bytes)):
        return tuple(shot_brief(e, i) for i, e in enumerate(entries))

    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return tuple(shot_brief(s, i) for i, s in enumerate(source))
    raise PlanRefused(
        f"cannot read a shot plan out of a {type(source).__name__}; pass a "
        f"ShotPlan, a sequence of SegmentSpecs/ShotPlanEntries, a k110 "
        f"ShotPlanDraft, or a run state carrying one of those")


def declared_color_scenes(source: Any) -> tuple[str, ...]:
    """The scene keys whose shot plan declared a look, in first-seen order."""
    out: list[str] = []
    for brief in shot_briefs(source):
        if brief.declares_color and brief.scene_key not in out:
            out.append(brief.scene_key)
    return tuple(out)


# ---------------------------------------------------------------------------
# Takes
# ---------------------------------------------------------------------------


TAKE_KINDS: tuple[str, ...] = ("video", "image")


@dataclass(frozen=True, slots=True)
class Take(ContentAddressed):
    """One attempt at one segment, as a record — never as a verdict.

    ``scorecard_digest`` points at the card that DECIDED this take (k102's
    ``scorecard_digest`` over the canonical JSON, so the same card digests
    identically wherever it is stored). ``accepted`` is the caller's reading of
    that card, not an opinion this class forms: a ``Take`` is evidence, and the
    judging lives in :func:`evaluate_take`.

    ``generator_model`` is what makes the "no self-judging" rule checkable
    rather than merely intended — see :class:`JudgeConflict`.

    ``duration_s`` is the MEASURED length of the produced artifact (k102 rule
    1: measured off the file, never the generator's claim). ``None`` means
    nobody measured it, which the duration check reports as unscored rather
    than assuming a fit."""

    segment_id: str
    attempt_id: str
    artifact_ref: str
    scorecard_digest: str | None = None
    preferred: bool = False
    accepted: bool = True
    duration_s: float | None = None
    generator_model: str | None = None
    seed: int | None = None
    kind: str = "video"
    repair_codes: tuple[RepairCode, ...] = ()
    diagnosis: str = ""

    def __post_init__(self) -> None:
        require_text(self.segment_id, "Take.segment_id")
        require_text(self.attempt_id, "Take.attempt_id")
        require_text(self.artifact_ref, "Take.artifact_ref")
        object.__setattr__(self, "scorecard_digest",
                           _opt_text(self.scorecard_digest,
                                     "Take.scorecard_digest"))
        object.__setattr__(self, "generator_model",
                           _opt_text(self.generator_model,
                                     "Take.generator_model"))
        for name in ("preferred", "accepted"):
            if not isinstance(getattr(self, name), bool):
                raise PostProductionError(
                    f"Take.{name} must be a bool, got "
                    f"{type(getattr(self, name)).__name__}")
        object.__setattr__(self, "duration_s",
                           _opt_seconds(self.duration_s, "Take.duration_s"))
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int) \
                    or self.seed < 0:
                raise PostProductionError(
                    f"Take.seed must be a non-negative int when set, got "
                    f"{self.seed!r}")
        if self.kind not in TAKE_KINDS:
            raise PostProductionError(
                f"Take.kind must be one of {list(TAKE_KINDS)}, got "
                f"{self.kind!r}")
        object.__setattr__(self, "repair_codes", tuple(
            _coerce_repair(c, "Take.repair_codes") for c in self.repair_codes))
        object.__setattr__(self, "diagnosis", str(self.diagnosis or ""))
        if self.preferred and not self.accepted:
            raise PostProductionError(
                f"Take({self.attempt_id}) is preferred but not accepted — the "
                f"preferred take of a segment is the one that PASSED; a "
                f"rejected take in the cut is the failure this type exists to "
                f"prevent")
        if not self.accepted and not self.repair_codes and not self.diagnosis:
            raise PostProductionError(
                f"Take({self.attempt_id}) was rejected with no repair code and "
                f"no diagnosis — a rejection nobody wrote down cannot become a "
                f"documented correction (brief Phase 3)")

    @classmethod
    def judged(cls, segment_id: str, attempt_id: str, artifact_ref: str,
               card: Scorecard, **kwargs: Any) -> "Take":
        """A take carrying the verdict of ``card``: its digest, its
        acceptance, its repair codes and its diagnosis. The one bridge from
        :func:`evaluate_take` to the assembly, so no caller has to re-derive
        "was this accepted" from a scorecard by hand."""
        codes = repair_codes_from(card)
        return cls(segment_id=segment_id, attempt_id=attempt_id,
                   artifact_ref=artifact_ref,
                   scorecard_digest=scorecard_digest(card),
                   accepted=bool(card.hard_pass), repair_codes=codes,
                   diagnosis=card.diagnosis or "", **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id, "attempt_id": self.attempt_id,
            "artifact_ref": self.artifact_ref,
            "scorecard_digest": self.scorecard_digest,
            "preferred": self.preferred, "accepted": self.accepted,
            "duration_s": self.duration_s,
            "generator_model": self.generator_model, "seed": self.seed,
            "kind": self.kind,
            "repair_codes": [c.value for c in self.repair_codes],
            "diagnosis": self.diagnosis,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Take":
        return cls(
            segment_id=d["segment_id"], attempt_id=d["attempt_id"],
            artifact_ref=d["artifact_ref"],
            scorecard_digest=d.get("scorecard_digest"),
            preferred=bool(d.get("preferred", False)),
            accepted=bool(d.get("accepted", True)),
            duration_s=d.get("duration_s"),
            generator_model=d.get("generator_model"), seed=d.get("seed"),
            kind=d.get("kind", "video"),
            repair_codes=tuple(d.get("repair_codes", ())),
            diagnosis=d.get("diagnosis", ""))


def group_takes(takes: Any) -> dict[str, tuple[Take, ...]]:
    """``{segment_id: takes}`` from a flat sequence or an already-grouped
    mapping, preserving the order the takes were produced in (attempt order is
    provenance: take 2 exists because take 1 was rejected)."""
    if takes is None:
        return {}
    out: dict[str, list[Take]] = {}
    if isinstance(takes, Mapping):
        items: Iterable[Any] = [t for group in takes.values()
                                for t in (group if isinstance(group, (list, tuple))
                                          else (group,))]
    else:
        items = takes
    for take in items:
        if not isinstance(take, Take):
            raise PostProductionError(
                f"takes must be Take objects, got {type(take).__name__}")
        out.setdefault(take.segment_id, []).append(take)
    return {k: tuple(v) for k, v in out.items()}


def takes_from_run_state(state: Any) -> tuple[Take, ...]:
    """The takes a k106 ``PerformanceResult`` already holds, as ``Take``s.

    Duck-typed on purpose (this module imports no orchestrator): anything whose
    ``shots`` carry ``segment_id`` / ``accepted`` / ``clip_ref`` reads. A shot
    with no ``clip_ref`` produced no artifact and yields NO take — which is
    exactly how it becomes a regeneration note instead of a silent gap."""
    out: list[Take] = []
    for shot in getattr(state, "shots", ()) or ():
        ref = getattr(shot, "clip_ref", None)
        segment_id = getattr(shot, "segment_id", None)
        if not ref or not segment_id:
            continue
        card = getattr(shot, "scorecard", None)
        codes = tuple(getattr(shot, "repair_codes", ()) or ())
        accepted = bool(getattr(shot, "accepted", False))
        diagnosis = str(getattr(shot, "diagnosis", "") or "")
        if not accepted and not codes and not diagnosis:
            diagnosis = "rejected by the orchestrator with no code recorded"
        out.append(Take(
            segment_id=str(segment_id),
            attempt_id=f"{segment_id}:clip",
            artifact_ref=str(ref),
            scorecard_digest=scorecard_digest(card) if card is not None else None,
            accepted=accepted,
            duration_s=getattr(shot, "clip_seconds", None),
            repair_codes=codes, diagnosis=diagnosis))
    return tuple(out)


# ---------------------------------------------------------------------------
# The edit decision list
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EditDecision(ContentAddressed):
    """One row of the cut: which take, trimmed where, joined how.

    ``in_s``/``out_s`` are TAKE-LOCAL times (where to trim the clip), not
    timeline positions — the timeline position is the sum of everything before
    it, which is what makes an EDL re-orderable without rewriting every row.
    ``None`` on either means "no trim on that side"."""

    order: int
    segment_id: str
    take: Take
    in_s: float | None = None
    out_s: float | None = None
    transition: str = "cut"
    transition_duration_s: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) \
                or self.order < 0:
            raise EDLRefused(
                f"EditDecision.order must be a non-negative int, got "
                f"{self.order!r}")
        require_text(self.segment_id, "EditDecision.segment_id")
        if not isinstance(self.take, Take):
            raise EDLRefused(f"EditDecision.take takes a Take, got "
                             f"{type(self.take).__name__}")
        if self.take.segment_id != self.segment_id:
            raise EDLRefused(
                f"EditDecision({self.order}) for segment {self.segment_id!r} "
                f"carries a take of {self.take.segment_id!r} — a cut row that "
                f"plays another segment's footage is how a chain starts")
        if self.transition not in TRANSITIONS:
            raise EDLRefused(
                f"EditDecision({self.order}).transition must be one of "
                f"{list(TRANSITIONS)}, got {self.transition!r}")
        object.__setattr__(self, "transition_duration_s",
                           quantize(require_non_negative(
                               self.transition_duration_s,
                               "EditDecision.transition_duration_s")))
        if self.transition in INSTANT_TRANSITIONS and \
                self.transition_duration_s > EPS:
            raise EDLRefused(
                f"EditDecision({self.order}) is a {self.transition!r} lasting "
                f"{self.transition_duration_s}s; {list(INSTANT_TRANSITIONS)} "
                f"occupy no time — a timed cut is a dissolve nobody named")
        if self.transition not in INSTANT_TRANSITIONS and \
                self.transition_duration_s <= EPS:
            raise EDLRefused(
                f"EditDecision({self.order}) is a {self.transition!r} of zero "
                f"length — a dissolve/fade with no duration is a cut nobody "
                f"meant; give it a duration or make it a cut")
        object.__setattr__(self, "in_s",
                           _opt_seconds(self.in_s, "EditDecision.in_s"))
        object.__setattr__(self, "out_s",
                           _opt_seconds(self.out_s, "EditDecision.out_s"))
        if self.in_s is not None and self.out_s is not None and \
                self.out_s + EPS < self.in_s:
            raise EDLRefused(
                f"EditDecision({self.order}) goes out at {self.out_s}s before "
                f"it comes in at {self.in_s}s")
        take_duration = self.take.duration_s
        if take_duration is not None:
            for name in ("in_s", "out_s"):
                value = getattr(self, name)
                if value is not None and value > take_duration + EPS:
                    raise EDLRefused(
                        f"EditDecision({self.order}).{name}={value}s is past "
                        f"the end of take {self.take.attempt_id!r} "
                        f"({take_duration}s) — a cut point outside the footage "
                        f"is black frames nobody asked for")
        object.__setattr__(self, "note", str(self.note or ""))

    @property
    def in_value(self) -> float:
        return 0.0 if self.in_s is None else self.in_s

    @property
    def out_value(self) -> float | None:
        """Where this row leaves the take: the explicit out point, else the
        take's measured end, else ``None`` (nobody measured it)."""
        return self.take.duration_s if self.out_s is None else self.out_s

    @property
    def duration_s(self) -> float | None:
        """Screen time of this row, or ``None`` when the take was never
        measured. ``None`` is never quietly treated as zero."""
        out = self.out_value
        return None if out is None else quantize(out - self.in_value)

    def to_dict(self) -> dict[str, Any]:
        return {"order": self.order, "segment_id": self.segment_id,
                "take": self.take.to_dict(), "in_s": self.in_s,
                "out_s": self.out_s, "transition": self.transition,
                "transition_duration_s": self.transition_duration_s,
                "note": self.note}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EditDecision":
        return cls(order=int(d["order"]), segment_id=d["segment_id"],
                   take=Take.from_dict(d["take"]), in_s=d.get("in_s"),
                   out_s=d.get("out_s"), transition=d.get("transition", "cut"),
                   transition_duration_s=d.get("transition_duration_s", 0.0),
                   note=d.get("note", ""))


@dataclass(frozen=True, slots=True)
class EditDecisionList(ContentAddressed):
    """The assembly order (brief Phase 3, doc Stage 18 step 1).

    The four validators, each a real failure mode of an assembly:

    1. ``order`` is gapless from zero and strictly increasing AS GIVEN. A gap
       means a row was dropped between planning and assembly; a repeat means
       two rows claim the same position and the assembler picks by luck.
    2. Every take in the list is ``preferred``. The EDL IS the preferred-take
       list, so "one preferred take per segment" is checkable here rather than
       being a convention somebody remembers.
    3. A segment appears with ONE take. Two rows for one segment are legal (an
       intercut plays the same shot twice) but they must play the same take —
       two different takes of one segment in one cut is two preferred takes.
    4. Order zero cannot be ``continuous``: there is no previous shot to
       continue from. (k104's index-0 ``joint_mode`` rule, one layer up.)"""

    decisions: tuple[EditDecision, ...] = ()
    fps: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(self.decisions))
        for decision in self.decisions:
            if not isinstance(decision, EditDecision):
                raise EDLRefused(
                    f"EditDecisionList.decisions takes EditDecision, got "
                    f"{type(decision).__name__}")
        for position, decision in enumerate(self.decisions):
            if decision.order != position:
                raise EDLRefused(
                    f"EditDecisionList is not gapless: row {position} carries "
                    f"order {decision.order}. Assembly order is 0..N-1 with no "
                    f"gaps and no repeats — a gap is a shot that fell out of "
                    f"the cut between planning and assembly")
        for decision in self.decisions:
            if not decision.take.preferred:
                raise EDLRefused(
                    f"EditDecisionList row {decision.order} plays take "
                    f"{decision.take.attempt_id!r}, which is not marked "
                    f"preferred; the EDL is the preferred-take list (one "
                    f"preferred take per segment)")
        chosen: dict[str, str] = {}
        for decision in self.decisions:
            previous = chosen.setdefault(decision.segment_id,
                                         decision.take.attempt_id)
            if previous != decision.take.attempt_id:
                raise EDLRefused(
                    f"segment {decision.segment_id!r} has two preferred takes "
                    f"in one cut ({previous!r} and "
                    f"{decision.take.attempt_id!r}) — pick one")
        if self.decisions and self.decisions[0].transition == "continuous":
            raise EDLRefused(
                "the first row of a cut cannot be 'continuous': there is no "
                "previous shot to continue from")
        if self.fps is not None:
            fps = float(self.fps)
            if fps <= 0:
                raise EDLRefused(f"EditDecisionList.fps must be positive when "
                                 f"set, got {self.fps!r}")
            object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "note", str(self.note or ""))

    def __len__(self) -> int:
        return len(self.decisions)

    def __iter__(self):
        return iter(self.decisions)

    @property
    def segment_ids(self) -> tuple[str, ...]:
        """Every segment the cut plays, in assembly order, deduped."""
        return tuple(dict.fromkeys(d.segment_id for d in self.decisions))

    @property
    def attempt_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(d.take.attempt_id for d in self.decisions))

    @property
    def takes(self) -> tuple[Take, ...]:
        return tuple(d.take for d in self.decisions)

    @property
    def has_unmeasured_rows(self) -> bool:
        return any(d.duration_s is None for d in self.decisions)

    @property
    def total_seconds(self) -> float | None:
        """The cut's screen time, or ``None`` when any row is unmeasured — a
        total that quietly skipped the rows it could not measure would be the
        most confidently wrong number in the file.

        Transitions are NOT subtracted: a dissolve's handle length is a mix
        decision the assembler owns, and guessing it here would fork the number
        the finishing pass computes."""
        if not self.decisions or self.has_unmeasured_rows:
            return None
        return quantize(sum(d.duration_s for d in self.decisions))

    def decision_for(self, segment_id: str) -> EditDecision:
        for decision in self.decisions:
            if decision.segment_id == segment_id:
                return decision
        raise KeyError(f"no edit decision for segment {segment_id!r}")

    def missing(self, segment_ids: Iterable[str]) -> tuple[str, ...]:
        """The requested segments this cut does NOT play, in the order asked."""
        covered = set(self.segment_ids)
        return tuple(s for s in dict.fromkeys(segment_ids) if s not in covered)

    def to_dict(self) -> dict[str, Any]:
        return {"decisions": [d.to_dict() for d in self.decisions],
                "fps": self.fps, "note": self.note}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EditDecisionList":
        return cls(decisions=tuple(EditDecision.from_dict(x)
                                   for x in d.get("decisions", ())),
                   fps=d.get("fps"), note=d.get("note", ""))


# ---------------------------------------------------------------------------
# Sound, music, regeneration notes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SoundCue(ContentAddressed):
    """One sound-effect / ambience / foley cue on the master timeline."""

    cue_id: str
    kind: str
    start_s: float
    end_s: float
    description: str
    segment_id: str | None = None
    asset_ref: str | None = None
    level_db: float | None = None

    def __post_init__(self) -> None:
        require_text(self.cue_id, "SoundCue.cue_id")
        if self.kind not in SOUND_KINDS:
            raise PostProductionError(
                f"SoundCue({self.cue_id}).kind must be one of "
                f"{list(SOUND_KINDS)}, got {self.kind!r}")
        object.__setattr__(self, "start_s", quantize(require_non_negative(
            self.start_s, "SoundCue.start_s")))
        object.__setattr__(self, "end_s", quantize(require_non_negative(
            self.end_s, "SoundCue.end_s")))
        if self.end_s + EPS < self.start_s:
            raise PostProductionError(
                f"SoundCue({self.cue_id}) ends before it starts: "
                f"{self.end_s} < {self.start_s}")
        require_text(self.description,
                     f"SoundCue({self.cue_id}).description")
        object.__setattr__(self, "segment_id",
                           _opt_text(self.segment_id, "SoundCue.segment_id"))
        object.__setattr__(self, "asset_ref",
                           _opt_text(self.asset_ref, "SoundCue.asset_ref"))
        if self.level_db is not None:
            object.__setattr__(self, "level_db", float(self.level_db))

    @property
    def duration_s(self) -> float:
        return quantize(self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return {"cue_id": self.cue_id, "kind": self.kind,
                "start_s": self.start_s, "end_s": self.end_s,
                "description": self.description, "segment_id": self.segment_id,
                "asset_ref": self.asset_ref, "level_db": self.level_db}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SoundCue":
        return cls(cue_id=d["cue_id"], kind=d["kind"],
                   start_s=d.get("start_s", 0.0), end_s=d.get("end_s", 0.0),
                   description=d["description"], segment_id=d.get("segment_id"),
                   asset_ref=d.get("asset_ref"), level_db=d.get("level_db"))


@dataclass(frozen=True, slots=True)
class MusicCue(ContentAddressed):
    """One music cue. ``role`` separates score from source music because they
    are mixed, licensed and graded differently."""

    cue_id: str
    start_s: float
    end_s: float
    description: str
    role: str = "score"
    segment_id: str | None = None
    asset_ref: str | None = None
    level_db: float | None = None

    def __post_init__(self) -> None:
        require_text(self.cue_id, "MusicCue.cue_id")
        if self.role not in MUSIC_ROLES:
            raise PostProductionError(
                f"MusicCue({self.cue_id}).role must be one of "
                f"{list(MUSIC_ROLES)}, got {self.role!r}")
        object.__setattr__(self, "start_s", quantize(require_non_negative(
            self.start_s, "MusicCue.start_s")))
        object.__setattr__(self, "end_s", quantize(require_non_negative(
            self.end_s, "MusicCue.end_s")))
        if self.end_s + EPS < self.start_s:
            raise PostProductionError(
                f"MusicCue({self.cue_id}) ends before it starts: "
                f"{self.end_s} < {self.start_s}")
        require_text(self.description, f"MusicCue({self.cue_id}).description")
        object.__setattr__(self, "segment_id",
                           _opt_text(self.segment_id, "MusicCue.segment_id"))
        object.__setattr__(self, "asset_ref",
                           _opt_text(self.asset_ref, "MusicCue.asset_ref"))
        if self.level_db is not None:
            object.__setattr__(self, "level_db", float(self.level_db))

    @property
    def duration_s(self) -> float:
        return quantize(self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return {"cue_id": self.cue_id, "start_s": self.start_s,
                "end_s": self.end_s, "description": self.description,
                "role": self.role, "segment_id": self.segment_id,
                "asset_ref": self.asset_ref, "level_db": self.level_db}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MusicCue":
        return cls(cue_id=d["cue_id"], start_s=d.get("start_s", 0.0),
                   end_s=d.get("end_s", 0.0), description=d["description"],
                   role=d.get("role", "score"), segment_id=d.get("segment_id"),
                   asset_ref=d.get("asset_ref"), level_db=d.get("level_db"))


@dataclass(frozen=True, slots=True)
class RegenerationNote(ContentAddressed):
    """A segment the cut cannot play yet, and what to do about it.

    This type IS the brief's rule::

        A rejected shot should be regenerated from the same canonical segment
        specification with documented corrections — not from the rejected
        prompt or footage description alone.

    ``spec_digest`` names that canonical specification (a ``SegmentSpec``
    digest). ``rejected_take_ref`` is EVIDENCE — the thing the correction is
    written about — and is never the thing it is written from. ``correction``
    must be non-empty: a regeneration note with no correction is a retry, and a
    retry that changes nothing is a loop wearing a diagnosis.

    Unpacks as ``(segment_id, repair_code, correction)`` so a caller can read
    it as the plain triple the brief describes."""

    segment_id: str
    repair_code: RepairCode
    correction: str
    spec_digest: str | None = None
    rejected_take_ref: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        require_text(self.segment_id, "RegenerationNote.segment_id")
        object.__setattr__(self, "repair_code",
                           _coerce_repair(self.repair_code,
                                          "RegenerationNote.repair_code"))
        require_text(self.correction,
                     f"RegenerationNote({self.segment_id}).correction")
        for name in ("spec_digest", "rejected_take_ref", "attempt_id"):
            object.__setattr__(self, name, _opt_text(
                getattr(self, name), f"RegenerationNote.{name}"))

    def __iter__(self):
        return iter(self.as_tuple)

    @property
    def as_tuple(self) -> tuple[str, RepairCode, str]:
        return (self.segment_id, self.repair_code, self.correction)

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id,
                "repair_code": self.repair_code.value,
                "correction": self.correction, "spec_digest": self.spec_digest,
                "rejected_take_ref": self.rejected_take_ref,
                "attempt_id": self.attempt_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RegenerationNote":
        return cls(segment_id=d["segment_id"], repair_code=d["repair_code"],
                   correction=d["correction"], spec_digest=d.get("spec_digest"),
                   rejected_take_ref=d.get("rejected_take_ref"),
                   attempt_id=d.get("attempt_id"))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PostProductionPlan(ContentAddressed):
    """Brief Phase 3's artifact list, as one immutable, content-addressed plan.

    Field-by-field against the brief: assembly order and preferred takes and
    transition instructions live in :attr:`edl`; "pacing and timing" is
    :attr:`pacing_notes`; "sound effects and ambience" is :attr:`sound`;
    "music cues" is :attr:`music`; "dialogue alignment" is
    :attr:`dialogue_alignment`; "color-grading targets" is
    :attr:`color_targets`; "continuity corrections" is
    :attr:`continuity_corrections`; "required regeneration notes" is
    :attr:`regeneration_notes`.

    ``locked_segments`` is the coverage denominator — every segment the locked
    shot plan declared. It is what makes "the EDL covers every locked segment"
    a checkable statement instead of a comparison against whatever happens to
    be in the EDL.

    ``locked`` is Stage 18's finishing gate: a plan cannot be locked while any
    locked segment is missing from the cut or any regeneration note is still
    open. You do not lock the cut while a shot is out for a re-shoot."""

    edl: EditDecisionList
    pacing_notes: tuple[str, ...] = ()
    sound: tuple[SoundCue, ...] = ()
    music: tuple[MusicCue, ...] = ()
    dialogue_alignment: tuple[tuple[str, float], ...] = ()
    color_targets: Mapping[str, Any] = field(default_factory=FrozenParams)
    continuity_corrections: tuple[tuple[str, str], ...] = ()
    regeneration_notes: tuple[RegenerationNote, ...] = ()
    locked: bool = False
    locked_segments: tuple[str, ...] = ()
    lock_digest: str | None = None
    audio_master_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.edl, EditDecisionList):
            raise PlanRefused(f"PostProductionPlan.edl takes an "
                              f"EditDecisionList, got {type(self.edl).__name__}")
        object.__setattr__(self, "pacing_notes",
                           str_tuple(self.pacing_notes,
                                     "PostProductionPlan.pacing_notes"))
        for name, cls_ in (("sound", SoundCue), ("music", MusicCue)):
            values = tuple(getattr(self, name))
            for cue in values:
                if not isinstance(cue, cls_):
                    raise PlanRefused(
                        f"PostProductionPlan.{name} takes {cls_.__name__}, got "
                        f"{type(cue).__name__}")
            ids = [c.cue_id for c in values]
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            if duplicates:
                raise PlanRefused(
                    f"PostProductionPlan.{name} has duplicate cue id(s) "
                    f"{duplicates} — a cue sheet with two rows under one id "
                    f"cannot be mixed")
            object.__setattr__(self, name, values)

        alignment: list[tuple[str, float]] = []
        seen: set[str] = set()
        for item in self.dialogue_alignment:
            line_id, offset = item
            line_id = require_text(line_id,
                                   "PostProductionPlan.dialogue_alignment "
                                   "line id")
            if line_id in seen:
                raise PlanRefused(
                    f"dialogue_alignment carries two offsets for line "
                    f"{line_id!r} — a line has one place in the mix")
            seen.add(line_id)
            alignment.append((line_id, quantize(float(offset))))
        object.__setattr__(self, "dialogue_alignment", tuple(alignment))

        object.__setattr__(self, "color_targets",
                           FrozenParams(self.color_targets))

        corrections: list[tuple[str, str]] = []
        for item in self.continuity_corrections:
            segment_id, note = item
            corrections.append((
                require_text(segment_id,
                             "PostProductionPlan.continuity_corrections "
                             "segment id"),
                require_text(note,
                             "PostProductionPlan.continuity_corrections note")))
        object.__setattr__(self, "continuity_corrections", tuple(corrections))

        notes = tuple(self.regeneration_notes)
        for note in notes:
            if not isinstance(note, RegenerationNote):
                raise PlanRefused(
                    f"PostProductionPlan.regeneration_notes takes "
                    f"RegenerationNote, got {type(note).__name__}")
        keys = [(n.segment_id, n.repair_code) for n in notes]
        duplicates = sorted({f"{s}/{c.value}" for s, c in keys
                             if keys.count((s, c)) > 1})
        if duplicates:
            raise PlanRefused(
                f"PostProductionPlan.regeneration_notes repeats "
                f"{duplicates} — one code per segment, with one correction")
        object.__setattr__(self, "regeneration_notes", notes)

        object.__setattr__(self, "locked_segments",
                           str_tuple(self.locked_segments,
                                     "PostProductionPlan.locked_segments"))
        object.__setattr__(self, "lock_digest",
                           _opt_text(self.lock_digest,
                                     "PostProductionPlan.lock_digest"))
        object.__setattr__(self, "audio_master_digest",
                           _opt_text(self.audio_master_digest,
                                     "PostProductionPlan.audio_master_digest"))
        if not isinstance(self.locked, bool):
            raise PlanRefused("PostProductionPlan.locked must be a bool")
        if self.locked:
            if self.uncovered_segments:
                raise PlanRefused(
                    f"cannot lock a post-production plan while segment(s) "
                    f"{list(self.uncovered_segments)} are missing from the cut "
                    f"— locking a partial assembly is how a segment gets "
                    f"silently dropped")
            if self.open_segments:
                raise PlanRefused(
                    f"cannot lock a post-production plan with open "
                    f"regeneration note(s) for segment(s) "
                    f"{list(self.open_segments)}: those shots are out for "
                    f"re-generation and the cut is not final")

    # -- reading -----------------------------------------------------------

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return self.edl.segment_ids

    @property
    def uncovered_segments(self) -> tuple[str, ...]:
        """Locked segments the cut does not play. Empty is the goal."""
        return self.edl.missing(self.locked_segments)

    @property
    def open_segments(self) -> tuple[str, ...]:
        """Segments carrying a regeneration note that are NOT in the cut — the
        work still outstanding. A segment can carry a note AND be in the cut
        (an accepted take with a documented reservation); that is not open."""
        covered = set(self.edl.segment_ids)
        return tuple(dict.fromkeys(n.segment_id for n in self.regeneration_notes
                                   if n.segment_id not in covered))

    @property
    def regeneration_codes(self) -> tuple[RepairCode, ...]:
        return tuple(dict.fromkeys(n.repair_code
                                   for n in self.regeneration_notes))

    def note_for(self, segment_id: str) -> RegenerationNote | None:
        for note in self.regeneration_notes:
            if note.segment_id == segment_id:
                return note
        return None

    def lock(self) -> "PostProductionPlan":
        """The finishing gate. Refuses (via ``__post_init__``) unless the cut
        is complete and nothing is out for regeneration."""
        return self if self.locked else replace(self, locked=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edl": self.edl.to_dict(),
            "pacing_notes": list(self.pacing_notes),
            "sound": [c.to_dict() for c in self.sound],
            "music": [c.to_dict() for c in self.music],
            "dialogue_alignment": [[line_id, offset]
                                   for line_id, offset in self.dialogue_alignment],
            "color_targets": FrozenParams(self.color_targets).to_dict(),
            "continuity_corrections": [[s, n]
                                       for s, n in self.continuity_corrections],
            "regeneration_notes": [n.to_dict() for n in self.regeneration_notes],
            "locked": self.locked,
            "locked_segments": list(self.locked_segments),
            "lock_digest": self.lock_digest,
            "audio_master_digest": self.audio_master_digest,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PostProductionPlan":
        return cls(
            edl=EditDecisionList.from_dict(d["edl"]),
            pacing_notes=tuple(d.get("pacing_notes", ())),
            sound=tuple(SoundCue.from_dict(c) for c in d.get("sound", ())),
            music=tuple(MusicCue.from_dict(c) for c in d.get("music", ())),
            dialogue_alignment=tuple(
                (a[0], float(a[1])) for a in d.get("dialogue_alignment", ())),
            color_targets=FrozenParams(d.get("color_targets") or {}),
            continuity_corrections=tuple(
                (c[0], c[1]) for c in d.get("continuity_corrections", ())),
            regeneration_notes=tuple(RegenerationNote.from_dict(n)
                                     for n in d.get("regeneration_notes", ())),
            locked=bool(d.get("locked", False)),
            locked_segments=tuple(d.get("locked_segments", ())),
            lock_digest=d.get("lock_digest"),
            audio_master_digest=d.get("audio_master_digest"))


# ---------------------------------------------------------------------------
# Spatial continuity (or-k13) — adjacent shots must agree about the world
# ---------------------------------------------------------------------------
#
# k110's continuity state is free-form, but when a shot plan went through the
# k116 spatial fold its ``state_before``/``state_after`` carry WHERE things are:
#
#     camera_pose        {"position": [x,y,z], "forward": [x,y,z]?,
#                         "track_uri": str?}      — or a CameraSpec dict under
#     camera_spec        {"track_uri": ..., ...}    (spatial.CameraSpec shape)
#     entity_positions   {entity_id: [x,y,z]}       (alias: "positions")
#     coordinate_system  spatial.CoordinateSystem dict; canonical when absent
#
# plus ``SegmentSpec.spatial_ref`` (the locked scene manifest) on the brief.
# Across a cut the camera MAY move — but not when both shots are locked to the
# SAME camera track: then shot B opens where shot A's track left it. Entities
# are world state and must agree at the cut whenever the two shots share a
# world (same spatial_ref, or neither names one). A contradiction is filed as a
# continuity correction on the LATER shot and as a RegenerationNote carrying
# the spatial repair code, so ``repair_controller`` re-renders the bounded
# subgraph instead of the film.

#: Entities further apart than this at a cut contradict each other (metres).
SPATIAL_ENTITY_TOLERANCE_M: float = 0.25

#: Camera look direction off the locked track across a cut (degrees).
SPATIAL_CAMERA_ANGLE_DEG_MAX: float = 2.0

_SPATIAL_STATE_KEYS: frozenset[str] = frozenset({
    "camera_pose", "camera_spec", "entity_positions", "positions"})


@dataclass(frozen=True, slots=True)
class SpatialContradiction:
    """Two adjacent shots that cannot both be true about the world at the cut.

    ``segment_id`` is the LATER shot — the one regenerated — and
    ``previous_segment_id`` the authority it contradicts. Unpacks as
    ``(segment_id, repair_code, note)`` like a correction row with a code."""

    segment_id: str
    previous_segment_id: str
    repair_code: RepairCode
    note: str
    evidence: Mapping[str, Any] = field(default_factory=FrozenParams)

    def __post_init__(self) -> None:
        require_text(self.segment_id, "SpatialContradiction.segment_id")
        require_text(self.previous_segment_id,
                     "SpatialContradiction.previous_segment_id")
        object.__setattr__(self, "repair_code",
                           _coerce_repair(self.repair_code,
                                          "SpatialContradiction.repair_code"))
        require_text(self.note, "SpatialContradiction.note")
        object.__setattr__(self, "evidence", FrozenParams(self.evidence))

    def __iter__(self):
        yield self.segment_id
        yield self.repair_code
        yield self.note

    @property
    def correction(self) -> tuple[str, str]:
        return (self.segment_id, self.note)

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id,
                "previous_segment_id": self.previous_segment_id,
                "repair_code": self.repair_code.value, "note": self.note,
                "evidence": FrozenParams(self.evidence).to_dict()}


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        value = [value.get(k) for k in ("x", "y", "z")]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _coord_system(state: Mapping[str, Any]) -> CoordinateSystem:
    cs = state.get("coordinate_system")
    if isinstance(cs, Mapping):
        try:
            return CoordinateSystem.from_dict(cs)
        except (KeyError, ValueError, TypeError):
            return CANONICAL
    return CANONICAL


def _canonical(point: tuple[float, float, float],
               system: CoordinateSystem) -> tuple[float, float, float]:
    return convert_points([point], system, CANONICAL)[0]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def _angle_deg(a: Sequence[float], b: Sequence[float]) -> float | None:
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(x) ** 2 for x in b))
    if na < 1e-12 or nb < 1e-12:
        return None
    cos = sum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _camera_of(state: Mapping[str, Any]
               ) -> tuple[tuple[float, float, float] | None,
                          tuple[float, float, float] | None, str | None]:
    """(position, forward, track_uri) in CANONICAL metres, from a state."""
    pose = state.get("camera_pose")
    spec = state.get("camera_spec")
    track: str | None = None
    if isinstance(spec, Mapping) and spec.get("track_uri"):
        try:
            track = CameraSpec.from_dict(spec).track_uri
        except (KeyError, ValueError, TypeError):
            track = str(spec.get("track_uri"))
    if not isinstance(pose, Mapping):
        return None, None, track
    system = _coord_system(state)
    pos = _vec3(pose.get("position"))
    fwd = _vec3(pose.get("forward") or pose.get("look_dir"))
    if pos is not None:
        pos = _canonical(pos, system)
    if fwd is not None:
        fwd = _canonical(fwd, system)
        # direction: undo the unit scale convert_points applied
        scale = system.scale_to_m / CANONICAL.scale_to_m
        if scale and scale != 1.0:
            fwd = (fwd[0] / scale, fwd[1] / scale, fwd[2] / scale)
    if pose.get("track_uri"):
        track = str(pose["track_uri"])
    return pos, fwd, track


def _entities_of(state: Mapping[str, Any]
                 ) -> dict[str, tuple[float, float, float]]:
    raw = state.get("entity_positions")
    if not isinstance(raw, Mapping):
        raw = state.get("positions")
    if not isinstance(raw, Mapping):
        return {}
    system = _coord_system(state)
    out: dict[str, tuple[float, float, float]] = {}
    for name, value in raw.items():
        p = _vec3(value)
        if p is not None:
            out[str(name)] = _canonical(p, system)
    return out


def _is_spatial(brief: ShotBrief) -> bool:
    if brief.spatial_ref:
        return True
    if brief.continuity is None:
        return False
    keys = set(brief.continuity.state_before) | set(brief.continuity.state_after)
    return bool(keys & _SPATIAL_STATE_KEYS)


def _fmt3(p: Sequence[float] | None) -> str:
    return ("(" + ", ".join(f"{float(v):.3f}" for v in p) + ")"
            if p is not None else "?")


def spatial_continuity_checks(
        briefs: Sequence[ShotBrief], *,
        thresholds: DriftThresholds | None = None,
        entity_tolerance_m: float = SPATIAL_ENTITY_TOLERANCE_M,
        camera_angle_deg_max: float = SPATIAL_CAMERA_ANGLE_DEG_MAX,
) -> tuple[SpatialContradiction, ...]:
    """Compare every ADJACENT pair of briefs that carry spatial information:
    ``A.state_after`` is the world at the cut as A leaves it, ``B.state_before``
    the world as B assumes it. Pairs with no spatial refs/camera info are
    skipped in silence — they were never spatially locked, so there is nothing
    to contradict. Pure: no I/O, deterministic order."""
    th = thresholds or DriftThresholds()
    out: list[SpatialContradiction] = []
    for prev, cur in zip(briefs, briefs[1:]):
        if not (_is_spatial(prev) and _is_spatial(cur)):
            continue
        after = (prev.continuity.state_after
                 if prev.continuity is not None else FrozenParams())
        before = (cur.continuity.state_before
                  if cur.continuity is not None else FrozenParams())
        same_ref = (prev.spatial_ref is not None
                    and prev.spatial_ref == cur.spatial_ref)
        different_ref = (prev.spatial_ref is not None
                         and cur.spatial_ref is not None
                         and prev.spatial_ref != cur.spatial_ref)

        # -- camera: only a SHARED track is authoritative across a cut
        a_pos, a_fwd, a_track = _camera_of(after)
        b_pos, b_fwd, b_track = _camera_of(before)
        shared_track = ((a_track is not None and a_track == b_track)
                        or (same_ref and a_track is None and b_track is None))
        if shared_track and a_pos is not None and b_pos is not None:
            dist = _distance(a_pos, b_pos)
            angle = (_angle_deg(a_fwd, b_fwd)
                     if a_fwd is not None and b_fwd is not None else None)
            off_pos = dist > th.camera_drift_m_max + EPS
            off_dir = angle is not None and angle > camera_angle_deg_max + EPS
            if off_pos or off_dir:
                what = []
                if off_pos:
                    what.append(f"position {_fmt3(a_pos)} -> {_fmt3(b_pos)} "
                                f"({dist:.3f} m > {th.camera_drift_m_max:g} m)")
                if off_dir:
                    what.append(f"look direction off by {angle:.1f} deg "
                                f"(> {camera_angle_deg_max:g} deg)")
                track = a_track or prev.spatial_ref or "shared camera track"
                out.append(SpatialContradiction(
                    segment_id=cur.segment_id,
                    previous_segment_id=prev.segment_id,
                    repair_code=RepairCode.CAMERA_PATH_MISMATCH,
                    note=(f"camera contradicts {prev.segment_id} at the cut on "
                          f"{track}: " + "; ".join(what) +
                          f". {prev.segment_id}'s track end is authoritative; "
                          f"re-render {cur.segment_id} with the camera pass "
                          f"enforced from that pose."),
                    evidence={"check": "camera_pose", "track": track,
                              "previous_position": list(a_pos),
                              "position": list(b_pos),
                              "distance_m": quantize(dist),
                              "angle_deg": (None if angle is None
                                            else round(angle, 3)),
                              "camera_drift_m_max": th.camera_drift_m_max}))

        # -- entities: world state, compared whenever the two shots share a world
        if not different_ref:
            a_ent = _entities_of(after)
            b_ent = _entities_of(before)
            moved: list[tuple[str, float]] = []
            detail: dict[str, Any] = {}
            for name in sorted(set(a_ent) & set(b_ent)):
                d = _distance(a_ent[name], b_ent[name])
                if d > entity_tolerance_m + EPS:
                    moved.append((name, d))
                    detail[name] = {"previous": list(a_ent[name]),
                                    "position": list(b_ent[name]),
                                    "distance_m": quantize(d)}
            if moved:
                listing = ", ".join(f"{n} moved {d:.2f} m" for n, d in moved)
                out.append(SpatialContradiction(
                    segment_id=cur.segment_id,
                    previous_segment_id=prev.segment_id,
                    repair_code=RepairCode.GEOMETRY_DRIFT,
                    note=(f"entity placement contradicts {prev.segment_id} at "
                          f"the cut: {listing} (tolerance "
                          f"{entity_tolerance_m:g} m). {prev.segment_id}'s "
                          f"state_after is authoritative; re-render "
                          f"{cur.segment_id} from the locked geometry with "
                          f"those placements."),
                    evidence={"check": "entity_positions", "entities": detail,
                              "entity_tolerance_m": entity_tolerance_m}))
    return tuple(out)


def _spatial_regeneration_notes(
        briefs: Sequence[ShotBrief],
        contradictions: Sequence[SpatialContradiction],
        existing: Sequence[RegenerationNote],
) -> tuple[RegenerationNote, ...]:
    """One note per (segment, code): a segment contradicting both its
    neighbours on the same axis gets ONE note whose correction lists both."""
    by_id = {b.segment_id: b for b in briefs}
    taken = {(n.segment_id, n.repair_code) for n in existing}
    grouped: dict[tuple[str, RepairCode], list[SpatialContradiction]] = {}
    for c in contradictions:
        grouped.setdefault((c.segment_id, c.repair_code), []).append(c)
    notes: list[RegenerationNote] = []
    for (segment_id, code), items in grouped.items():
        if (segment_id, code) in taken:
            continue                     # the take's own diagnosis already routes it
        brief = by_id[segment_id]
        source = (f"segment spec {brief.spec_digest[:12]}"
                  if brief.spec_digest else "the segment's canonical specification")
        notes.append(RegenerationNote(
            segment_id=segment_id, repair_code=code,
            correction=(f"spatial continuity: regenerate from {source} — "
                        + " ".join(c.note for c in items) +
                        f" ({RECOMMENDED_REPAIR.get(code, 're-render')}). "
                        f"Do NOT re-prompt from the rejected footage or its "
                        f"description."),
            spec_digest=brief.spec_digest))
    return tuple(notes)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _regeneration_correction(brief: ShotBrief, rejected: Sequence[Take],
                            code: RepairCode) -> str:
    """The documented correction for a segment with no accepted take.

    Deterministic prose, built from the SPECIFICATION (window, rubric) and the
    rejections' own diagnoses — never from the rejected footage's description,
    which is the thing the brief forbids regenerating from."""
    source = (f"segment spec {brief.spec_digest[:12]}"
              if brief.spec_digest else "the segment's canonical specification")
    head = (f"no take was produced for {brief.segment_id}"
            if not rejected else
            f"{len(rejected)} take(s) rejected "
            f"({', '.join(t.attempt_id for t in rejected)})")
    reasons = [t.diagnosis for t in rejected if t.diagnosis]
    detail = ("; ".join(reasons) if reasons else
              RECOMMENDED_REPAIR.get(code, "regenerate this shot"))
    return (f"{head}: regenerate from {source} — the locked window "
            f"[{brief.start_s:g}s..{brief.end_s:g}s] and its acceptance "
            f"rubric are unchanged. Correction: {detail}. Do NOT re-prompt "
            f"from the rejected footage or its description.")


def _select_take(segment_id: str, candidates: Sequence[Take]
                 ) -> tuple[Take | None, tuple[Take, ...]]:
    """(chosen, rejected). The chosen take is the one the caller marked
    ``preferred`` when there is exactly one such accepted take, otherwise the
    FIRST accepted take — attempt order is provenance, and take 1 passing means
    takes 2..N were never needed."""
    accepted = tuple(t for t in candidates if t.accepted)
    rejected = tuple(t for t in candidates if not t.accepted)
    marked = tuple(t for t in accepted if t.preferred)
    if len(marked) > 1:
        raise PlanRefused(
            f"segment {segment_id!r} has {len(marked)} takes marked preferred "
            f"({', '.join(t.attempt_id for t in marked)}) — one preferred take "
            f"per segment; the plan will not pick for you")
    if marked:
        return marked[0], rejected
    return (accepted[0] if accepted else None), rejected


def build_postproduction_plan(
        lock_or_run_state: Any,
        takes: Any = (),
        audio_master: AudioMaster | None = None,
        *,
        lock: ProductionLock | None = None,
        transitions: Mapping[str, Any] | None = None,
        sound: Sequence[SoundCue] = (),
        music: Sequence[MusicCue] = (),
        color_targets: Mapping[str, Any] | None = None,
        continuity_corrections: Sequence[tuple[str, str]] = (),
        pacing_tolerance: float = DEFAULT_PACING_TOLERANCE_S,
        spatial_thresholds: DriftThresholds | None = None,
        spatial_checks: bool = True,
) -> PostProductionPlan:
    """Assemble the post-production plan. Deterministic: same inputs, same
    plan, same digest.

    ``lock_or_run_state`` is anything :func:`shot_briefs` can read a shot plan
    out of — a ``ShotPlan``, the compiled ``SegmentSpec``s, a k110
    ``ShotPlanDraft``, or a k106 ``PerformanceResult`` (whose own ``shots``
    also supply the takes when ``takes`` is left empty).

    THE RULE THIS FUNCTION EXISTS FOR: a segment with no accepted take becomes
    an explicit :class:`RegenerationNote` naming a ``RepairCode`` and a
    documented correction. It is never dropped from the assembly order and
    never left as an unexplained hole.

    ``transitions`` maps ``segment_id`` to a transition — either a name from
    :data:`TRANSITIONS` or a ``(name, duration_s)`` pair. Unmapped rows are
    straight cuts: inventing a dissolve is a directorial decision this function
    does not have the evidence to make.

    ``spatial_checks`` runs :func:`spatial_continuity_checks` over adjacent
    briefs that carry spatial refs / camera info: every contradiction becomes
    a continuity correction on the later shot AND a :class:`RegenerationNote`
    carrying ``CAMERA_PATH_MISMATCH`` / ``GEOMETRY_DRIFT`` so the repair
    controller routes the re-render. ``spatial_thresholds`` defaults to
    ``spatial.DriftThresholds()``."""
    briefs = shot_briefs(lock_or_run_state)
    if not briefs:
        raise PlanRefused(
            "cannot build a post-production plan for an empty shot plan — "
            "there is nothing to assemble")

    if not takes:
        takes = takes_from_run_state(lock_or_run_state)
    grouped = group_takes(takes)
    master = audio_master
    if master is None:
        candidate = getattr(lock_or_run_state, "audio_master", None)
        if isinstance(candidate, AudioMaster):
            master = candidate
    if lock is None:
        candidate = getattr(lock_or_run_state, "lock", None)
        if isinstance(candidate, ProductionLock):
            lock = candidate

    unknown_segments = sorted(set(grouped) - {b.segment_id for b in briefs})
    if unknown_segments:
        raise PlanRefused(
            f"takes name segment(s) {unknown_segments} that this shot plan "
            f"does not contain — the takes and the plan are from different "
            f"productions")

    decisions: list[EditDecision] = []
    notes: list[RegenerationNote] = []
    pacing: list[str] = []
    corrections: list[tuple[str, str]] = [
        (require_text(s, "continuity_corrections segment id"),
         require_text(n, "continuity_corrections note"))
        for s, n in continuity_corrections]
    derived_color: dict[str, Any] = {}
    picture_s = 0.0
    alignment: list[tuple[str, float]] = []

    for brief in briefs:
        if brief.declares_color and brief.scene_key not in derived_color:
            derived_color[brief.scene_key] = (
                brief.lighting or
                "; ".join(c for c in brief.rubric if has_color_cue(c)))

        chosen, rejected = _select_take(brief.segment_id,
                                        grouped.get(brief.segment_id, ()))
        if chosen is None:
            code = (rejected[0].repair_codes[0] if rejected
                    and rejected[0].repair_codes
                    else (RepairCode.INTENT_MISMATCH if rejected
                          else RepairCode.EMPTY_OUTPUT))
            notes.append(RegenerationNote(
                segment_id=brief.segment_id, repair_code=code,
                correction=_regeneration_correction(brief, rejected, code),
                spec_digest=brief.spec_digest,
                rejected_take_ref=rejected[0].artifact_ref if rejected else None,
                attempt_id=rejected[0].attempt_id if rejected else None))
            pacing.append(
                f"{brief.segment_id}: no footage in the cut; its locked "
                f"{brief.duration_s:.3f}s window is unfilled")
            continue

        name, duration = _transition_for(brief, transitions,
                                         first=not decisions)
        in_s: float | None = None
        out_s: float | None = None
        window = brief.duration_s
        if chosen.duration_s is not None and window > 0 and \
                chosen.duration_s > window + EPS:
            in_s, out_s = 0.0, window        # trim the take to its locked window
        decision = EditDecision(
            order=len(decisions), segment_id=brief.segment_id,
            take=replace(chosen, preferred=True), in_s=in_s, out_s=out_s,
            transition=name, transition_duration_s=duration,
            note=(f"assembly order {len(decisions)}; locked window "
                  f"[{brief.start_s:g}s..{brief.end_s:g}s]"))
        decisions.append(decision)

        screen = decision.duration_s
        if screen is None:
            pacing.append(
                f"{brief.segment_id}: take {chosen.attempt_id} was never "
                f"measured — its screen time against the "
                f"{window:.3f}s locked window is unknown")
        else:
            deviation = quantize(screen - window)
            if abs(deviation) > pacing_tolerance:
                pacing.append(
                    f"{brief.segment_id}: {screen:.3f}s of picture against a "
                    f"{window:.3f}s locked window ({deviation:+.3f}s, outside "
                    f"±{pacing_tolerance:g}s)")
        if chosen.diagnosis:
            corrections.append((
                brief.segment_id,
                f"accepted with a reservation: {chosen.diagnosis}"))

        if master is not None:
            # The offset is the DRIFT between where the master plays a line and
            # where the picture has got to by then: zero while every shot runs
            # its locked length, and exactly the accumulated shortfall after a
            # shot that does not. Lines the master does not carry are skipped
            # rather than aligned against a time nobody recorded.
            drift = quantize(brief.start_s - picture_s)
            for line_id in brief.line_ids:
                if line_id not in master.line_ids:
                    continue
                alignment.append((line_id, drift))
        if screen is not None:
            picture_s = quantize(picture_s + screen)

    if spatial_checks:
        contradictions = spatial_continuity_checks(
            briefs, thresholds=spatial_thresholds)
        corrections.extend(c.correction for c in contradictions)
        notes.extend(_spatial_regeneration_notes(briefs, contradictions, notes))

    edl = EditDecisionList(decisions=tuple(decisions))
    total = edl.total_seconds
    if master is not None and total is not None:
        pacing.append(
            f"cut runs {total:.3f}s against a {master.total_seconds:.3f}s "
            f"audio master ({quantize(total - master.total_seconds):+.3f}s)")
    if not pacing:
        pacing.append(
            f"pacing: all {len(decisions)} shot(s) sit within "
            f"±{pacing_tolerance:g}s of their locked windows")

    targets = dict(derived_color)
    targets.update(dict(color_targets or {}))

    return PostProductionPlan(
        edl=edl,
        pacing_notes=tuple(pacing),
        sound=tuple(sound),
        music=tuple(music),
        dialogue_alignment=tuple(alignment),
        color_targets=FrozenParams(targets),
        continuity_corrections=tuple(corrections),
        regeneration_notes=tuple(notes),
        locked_segments=tuple(b.segment_id for b in briefs),
        lock_digest=lock.digest if lock is not None else None,
        audio_master_digest=master.digest if master is not None else None)


def _transition_for(brief: ShotBrief, transitions: Mapping[str, Any] | None,
                    *, first: bool) -> tuple[str, float]:
    """``(name, duration_s)`` for one row. Straight cut unless told otherwise;
    a ``continuous`` on the first row is refused by the EDL, so it is corrected
    to a cut here with no drama — there is nothing before it to continue."""
    value = (transitions or {}).get(brief.segment_id)
    if value is None:
        return ("cut", 0.0)
    if isinstance(value, str):
        name, duration = value, 0.0
    else:
        name, duration = value[0], float(value[1])
    if name not in TRANSITIONS:
        raise PlanRefused(
            f"transitions[{brief.segment_id!r}]={name!r} is not in "
            f"{list(TRANSITIONS)}")
    if first and name == "continuous":
        return ("cut", 0.0)
    if name not in INSTANT_TRANSITIONS and duration <= EPS:
        raise PlanRefused(
            f"transitions[{brief.segment_id!r}] is a {name!r} with no "
            f"duration; give it one as (name, seconds)")
    return (name, 0.0 if name in INSTANT_TRANSITIONS else duration)


# ---------------------------------------------------------------------------
# Footage vs. specification — the judge seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """One rubric put to one judge about one artifact.

    ``repair_code`` travels WITH the request so the diagnosis a failure earns
    is fixed by the rubric that asked the question, not decided afterwards by
    whoever reads the reply."""

    name: str
    kind: CheckKind
    prompt: str
    artifact_ref: str
    artifact_kind: str = "video"
    segment_id: str = ""
    repair_code: RepairCode = RepairCode.INTENT_MISMATCH

    def __post_init__(self) -> None:
        require_text(self.name, "JudgeRequest.name")
        require_text(self.prompt, "JudgeRequest.prompt")
        require_text(self.artifact_ref, "JudgeRequest.artifact_ref")
        if self.artifact_kind not in TAKE_KINDS:
            raise PostProductionError(
                f"JudgeRequest.artifact_kind must be one of {list(TAKE_KINDS)}, "
                f"got {self.artifact_kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind.value,
                "prompt": self.prompt, "artifact_ref": self.artifact_ref,
                "artifact_kind": self.artifact_kind,
                "segment_id": self.segment_id,
                "repair_code": self.repair_code.value}


JudgeFn = Callable[[JudgeRequest], Any]
TranscribeFn = Callable[[str], Any]


def shot_spec_block(brief: ShotBrief,
                    continuity: ContinuityState | None = None) -> str:
    """The shot SPECIFICATION as the judge sees it — deterministic prose over
    the requested setting, characters, action, camera, look, the continuity
    state on either side of the shot, and the acceptance rubric.

    This is what "compare generated footage against its shot specification,
    screenplay event, and continuity state" reduces to when the comparison has
    to be reproducible: the same brief always produces the same block, so two
    judges are asked the same question and their disagreement means something."""
    state = continuity if continuity is not None else brief.continuity
    lines = [f"SHOT SPECIFICATION for segment {brief.segment_id}"
             + (f" (scene {brief.scene_ref})" if brief.scene_ref else "") + ":"]
    setting = None
    if state is not None:
        for key in ("location", "setting"):
            value = state.state_before.get(key)
            if value:
                setting = str(value)
                break
    if setting:
        lines.append(f"- setting: {setting}")
    if brief.characters:
        lines.append(f"- characters: {', '.join(brief.characters)}")
    if brief.action:
        lines.append(f"- requested action: {brief.action}")
    if brief.camera:
        lines.append("- camera: " + ", ".join(
            f"{k}={brief.camera[k]}" for k in sorted(brief.camera)))
    if brief.lighting:
        lines.append(f"- lighting/look: {brief.lighting}")
    lines.append(f"- duration: {brief.duration_s:g}s "
                 f"[{brief.start_s:g}s..{brief.end_s:g}s] on the audio master")
    if state is not None:
        lines.append("- state before: " + _state_line(state.state_before))
        lines.append("- state after: " + _state_line(state.state_after))
        if state.changed_keys:
            lines.append("- must change during the shot: "
                         + ", ".join(state.changed_keys))
    for criterion in brief.rubric:
        lines.append(f"- acceptance: {criterion}")
    return "\n".join(lines)


def _state_line(state: Mapping[str, Any]) -> str:
    if not state:
        return "(nothing declared)"
    return "; ".join(f"{k}={state[k]}" for k in sorted(state))


def take_rubrics(brief: ShotBrief, continuity: ContinuityState | None = None,
                 *, artifact_ref: str, artifact_kind: str = "video"
                 ) -> tuple[JudgeRequest, ...]:
    """The rubrics one take is judged against, in evidence-class order.

    ``shot.intent`` (semantic/intent) always runs: it is the k90c rubric widened
    from "does the image achieve the goal" to "is this the SPECIFIED shot".
    ``shot.action`` runs when the specification names an action. ``shot.identity``
    runs when it names who is in the frame. ``shot.temporal`` runs only for
    video — a still cannot flicker, and asking a judge whether it does invites
    an invented answer."""
    block = shot_spec_block(brief, continuity)
    fmt = evaluation._JUDGE_REPLY_FORMAT      # noqa: SLF001 — reuse, not copy
    subject = "video clip" if artifact_kind == "video" else "image"
    state = continuity if continuity is not None else brief.continuity

    out: list[JudgeRequest] = [JudgeRequest(
        name="shot.intent", kind=CheckKind.INTENT,
        prompt=f"{block}\n\nDoes this {subject} show the shot specified above "
               f"(setting, framing and story beat)? {fmt}",
        artifact_ref=artifact_ref, artifact_kind=artifact_kind,
        segment_id=brief.segment_id, repair_code=RepairCode.INTENT_MISMATCH)]

    if brief.action:
        changed = (", ".join(state.changed_keys)
                   if state is not None and state.changed_keys else "")
        tail = (f" By the end of the shot these must have changed: {changed}."
                if changed else "")
        out.append(JudgeRequest(
            name="shot.action", kind=CheckKind.SEMANTIC,
            prompt=f"{block}\n\nDoes the requested action actually occur in "
                   f"this {subject}: {brief.action!r}?{tail} {fmt}",
            artifact_ref=artifact_ref, artifact_kind=artifact_kind,
            segment_id=brief.segment_id, repair_code=RepairCode.ACTION_MISSING))

    if brief.characters:
        out.append(JudgeRequest(
            name="shot.identity", kind=CheckKind.IDENTITY,
            prompt=f"{block}\n\nAre exactly these characters present and "
                   f"consistent with their established appearance: "
                   f"{', '.join(brief.characters)}? {fmt}",
            artifact_ref=artifact_ref, artifact_kind=artifact_kind,
            segment_id=brief.segment_id, repair_code=RepairCode.IDENTITY_DRIFT))

    if artifact_kind == "video":
        out.append(JudgeRequest(
            name="shot.temporal", kind=CheckKind.TEMPORAL,
            prompt=f"{block}\n\nIs this clip free of temporal artifacts — "
                   f"flicker, morphing, mutating or disappearing anatomy, "
                   f"objects that pop in or out? {fmt}",
            artifact_ref=artifact_ref, artifact_kind=artifact_kind,
            segment_id=brief.segment_id,
            repair_code=RepairCode.TEMPORAL_ARTIFACT))
    return tuple(out)


def coerce_judge_result(request: JudgeRequest, raw: Any,
                        model: str | None = None) -> JudgeResult:
    """Whatever a judge returned -> a ``JudgeResult``. Never raises.

    Accepts a ``JudgeResult``, a reply string (parsed with k90c's own tolerant
    ``evaluation.parse_judge_verdict`` — imported, not re-implemented), a
    mapping (``verdict``/``score``/``why``, or a raw ``text`` to parse), or a
    bare bool. ``None`` and anything unrecognized become the honest
    ``verdict="unavailable"`` entry, which the caller records WITHOUT flipping
    ``hard_pass`` (k90c: "unscored, keep")."""
    name = f"{request.name}:{model}" if model else request.name
    if isinstance(raw, JudgeResult):
        return raw
    if raw is None:
        return JudgeResult(judge=name, verdict="unavailable", score=None,
                           rationale="judge returned nothing")
    if isinstance(raw, bool):
        return JudgeResult(judge=name, verdict="YES" if raw else "NO",
                           score=None, rationale="")
    if isinstance(raw, str):
        parsed = evaluation.parse_judge_verdict(raw)
        return JudgeResult(judge=name, verdict=parsed["verdict"] or "unscored",
                           score=(float(parsed["score"])
                                  if parsed["score"] is not None else None),
                           rationale=parsed["why"] or raw.strip()[:300])
    if isinstance(raw, Mapping):
        model_name = str(raw.get("model") or raw.get("judge") or model or "")
        name = f"{request.name}:{model_name}" if model_name else request.name
        verdict = raw.get("verdict", raw.get("passed", raw.get("ok")))
        text = raw.get("text") or raw.get("reply")
        if verdict is None and text:
            parsed = evaluation.parse_judge_verdict(str(text))
            verdict, score = parsed["verdict"], parsed["score"]
            why = parsed["why"]
        else:
            score = raw.get("score")
            why = str(raw.get("why") or raw.get("rationale")
                      or raw.get("detail") or "")
        return JudgeResult(
            judge=name, verdict=_verdict_word(verdict),
            score=None if score is None else float(score), rationale=why)
    return JudgeResult(judge=name, verdict="unavailable", score=None,
                       rationale=f"judge returned a {type(raw).__name__}, "
                                 f"which carries no verdict")


def _verdict_word(value: Any) -> str:
    if value is None:
        return "unscored"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    text = str(value).strip()
    lowered = text.casefold()
    if lowered in ("yes", "true", "pass", "passed", "ok"):
        return "YES"
    if lowered in ("no", "false", "fail", "failed"):
        return "NO"
    if lowered in ("unavailable", "unscored"):
        return lowered
    return text.upper() if lowered in ("y", "n") else text


def _check_from_judge(request: JudgeRequest, result: JudgeResult,
                      threshold: int) -> tuple[Check, str | None]:
    """(check, disagreement). Mirrors ``evaluation.evaluate``: an
    unavailable/unscored judge is recorded and does NOT fail the take; a
    verdict of NO over a passing score is a DISAGREEMENT and the score wins
    (movie semantics, kept identical so two evaluators never tell different
    stories about the same reply)."""
    if result.verdict in ("unavailable", "unscored"):
        return (Check(name=request.name, kind=request.kind, value=None,
                      threshold=threshold, passed=True,
                      detail=f"{_UNSCORED}{result.judge}: "
                             f"{result.rationale or 'no verdict'}"), None)
    failing = (result.score is not None and result.score < threshold) or \
              (result.score is None and result.verdict == "NO")
    disagreement = None
    if not failing and result.verdict == "NO" and result.score is not None:
        disagreement = (f"{result.judge}: verdict NO but score "
                        f"{result.score:g} >= threshold {threshold} — score "
                        f"wins (movie semantics)")
    scored = f"score {result.score:g}" if result.score is not None \
        else "no score"
    return (Check(name=request.name, kind=request.kind, value=result.score,
                  threshold=threshold, passed=not failing,
                  detail=(f"{result.judge}: verdict {result.verdict}, {scored} "
                          f"(threshold {threshold})"
                          + (f" — {result.rationale}" if result.rationale
                             else ""))), disagreement)


def take_repair_code(checks: Sequence[Check]) -> RepairCode | None:
    """The code of the highest-priority failing take check, or None."""
    failing = {c.name for c in checks if not c.passed}
    for name in _TAKE_PRIORITY:
        if name in failing:
            return TAKE_REPAIR[name]
    return None


def repair_codes_from(card: Scorecard) -> tuple[RepairCode, ...]:
    """Every repair code a card's failing checks diagnose, in priority order —
    the codes a ``RegenerationNote`` (or k107's mapping) needs, not just the
    single winner ``Scorecard.repair_code`` records."""
    failing = {c.name for c in card.checks if not c.passed}
    codes = [TAKE_REPAIR[n] for n in _TAKE_PRIORITY if n in failing]
    for name in sorted(failing - set(_TAKE_PRIORITY)):
        extra = REPORT_REPAIR.get(name) or speech.SPEECH_REPAIR.get(name)
        if extra is not None:
            codes.append(extra)
    if not codes and card.repair_code is not None:
        codes.append(card.repair_code)
    return tuple(dict.fromkeys(codes))


def _judge_model_of(judge: Any, judge_model: str | None) -> str | None:
    if judge_model is not None:
        return str(judge_model).strip() or None
    model = getattr(judge, "model", None)
    return str(model).strip() if model else None


def evaluate_take(take: Take, segment: Any,
                  continuity: ContinuityState | None = None, *,
                  judge: JudgeFn,
                  transcribe: TranscribeFn | None = None,
                  judge_model: str | None = None,
                  lines: Any = None,
                  quality: QualityProfile = QualityProfile.BALANCED,
                  duration_tolerance: float = speech.DEFAULT_DURATION_TOLERANCE,
                  ) -> Scorecard:
    """Compare ONE take against ITS shot specification. doc Stage 17.

    The card composes four evidence classes, all on the existing ``Scorecard``
    shape:

    * **Semantic / identity / temporal** — the shot-spec rubrics of
      :func:`take_rubrics`, judged by the injected ``judge`` and scored against
      k90c's own per-quality bar (``evaluation.THRESHOLDS``).
    * **Speech** — k98 ``check_lines_present`` over a round-trip transcript,
      when ``transcribe`` is supplied and the shot covers dialogue whose text
      can be resolved from ``lines``.
    * **Sync** — k98 ``check_duration_fit``, with Stage 8's direction: the
      LOCKED WINDOW is the authoritative audio and the take is the shot, so a
      take shorter than its window is ``SHOT_TOO_SHORT`` (extend/re-render the
      shot), never "trim the audio".

    ``confidence`` is the scored fraction: a card built with an unreachable
    judge is not as sure as one built with it, and says so.

    REFUSES, typed, when the judge model equals the take's ``generator_model``
    (doc §9). ``judge_model`` may be passed explicitly or read off a judge
    object's ``.model`` attribute (which :func:`bind_live_judge` sets)."""
    if not isinstance(take, Take):
        raise PostProductionError(f"evaluate_take takes a Take, got "
                                  f"{type(take).__name__}")
    if judge is None or not callable(judge):
        raise PostProductionError(
            "evaluate_take needs a judge callable: footage is not accepted on "
            "the generator's word (doc §9)")
    brief = shot_brief(segment)
    state = continuity if continuity is not None else brief.continuity

    model = _judge_model_of(judge, judge_model)
    if model and take.generator_model and \
            model.strip().casefold() == take.generator_model.strip().casefold():
        raise JudgeConflict(
            f"refusing to evaluate take {take.attempt_id!r} of segment "
            f"{take.segment_id!r} with {model!r}: that is the model that "
            f"GENERATED it. doc §9 — do not let the generator approve itself; "
            f"a self-graded take is worse than an ungraded one because it "
            f"looks like evidence. Bind a different judge model.",
            model=model, segment_id=take.segment_id)

    threshold = evaluation.THRESHOLDS[quality]
    checks: list[Check] = []
    judge_results: list[JudgeResult] = []
    disagreements: list[str] = []

    for request in take_rubrics(brief, state, artifact_ref=take.artifact_ref,
                                artifact_kind=take.kind):
        try:
            raw = judge(request)
        except Exception as exc:      # noqa: BLE001 — a judge fault degrades
            logger.info("k108 judge %s raised (%s: %s); recording unavailable",
                        request.name, type(exc).__name__, exc)
            raw = JudgeResult(judge=f"{request.name}:{model or 'judge'}",
                              verdict="unavailable", score=None,
                              rationale=f"{type(exc).__name__}: {exc}"[:300])
        result = coerce_judge_result(request, raw, model)
        judge_results.append(result)
        check, disagreement = _check_from_judge(request, result, threshold)
        checks.append(check)
        if disagreement:
            disagreements.append(disagreement)

    if transcribe is not None:
        checks.append(_speech_check(take, brief, transcribe, lines))

    checks.append(speech.check_duration_fit(
        audio_seconds=brief.duration_s if brief.end_s > brief.start_s else None,
        shot_seconds=take.duration_s, tolerance=duration_tolerance))

    hard_pass = all(c.passed for c in checks)
    scored = [c for c in checks if not speech.is_unscored(c)]
    confidence = round(len(scored) / len(checks), 3) if checks else 1.0
    code = None if hard_pass else take_repair_code(checks)
    diagnoses = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    unscored = [c.name for c in checks if speech.is_unscored(c)]
    if unscored:
        diagnoses.append("unscored (no evidence): " + ", ".join(unscored))
    return Scorecard(
        hard_pass=hard_pass,
        checks=tuple(checks),
        judge_results=tuple(judge_results),
        confidence=confidence,
        disagreements=tuple(disagreements),
        diagnosis="; ".join(diagnoses) or None,
        repair_code=code,
        recommended_repair=(None if hard_pass else RECOMMENDED_REPAIR.get(
            code, "regenerate this segment from its canonical SegmentSpec")))


def _speech_check(take: Take, brief: ShotBrief, transcribe: TranscribeFn,
                  lines: Any) -> Check:
    """k98's line check over a round trip of THIS take's audio window."""
    texts_by_id = _mapping_of_text(lines)
    expected = [texts_by_id[i] for i in brief.line_ids if i in texts_by_id]
    if not brief.line_ids:
        return Check(name="speech.lines_present", kind=CheckKind.SPEECH,
                     value=None, threshold=None, passed=True,
                     detail=f"{_UNSCORED}this shot covers no dialogue window "
                            f"— there is no line evidence to produce")
    if not expected:
        return Check(name="speech.lines_present", kind=CheckKind.SPEECH,
                     value=None, threshold=len(brief.line_ids), passed=True,
                     detail=f"{_UNSCORED}no text supplied for line(s) "
                            f"{list(brief.line_ids)}; pass lines= (the locked "
                            f"DialogueTimeline) to verify them")
    try:
        words = transcribe(take.artifact_ref)
    except Exception as exc:          # noqa: BLE001 — an ASR fault degrades
        logger.info("k108 transcribe raised (%s: %s); recording unscored",
                    type(exc).__name__, exc)
        return Check(name="speech.lines_present", kind=CheckKind.SPEECH,
                     value=None, threshold=len(expected), passed=True,
                     detail=f"{_UNSCORED}round-trip transcription failed "
                            f"({type(exc).__name__}: {exc})"[:400])
    return speech.check_lines_present(expected, words)


# ---------------------------------------------------------------------------
# The live judge — the catalog's own resolution, imported and called.
# ---------------------------------------------------------------------------


#: The capability that grades a still. Same one ``evaluation.RUBRICS`` uses for
#: ``image.generate``, so the judge model is the catalog's choice, never ours.
JUDGE_CAPABILITY: str = "image.understand"

#: The capability a CLIP would be graded by. k106's live seam table records the
#: truth here: "there is no clip-level evaluator: the vision judge grades
#: STILLS". Resolution is still ATTEMPTED — a fleet that seats one starts
#: working with no edit here — and its absence is reported, never papered over.
VIDEO_JUDGE_CAPABILITY: str = "video.understand"

_FRAME_GAP = ("no clip-level evaluator is registered on this fleet and no "
              "frames= sampler is bound: pass frames=<callable returning a "
              "still from a clip ref> (e.g. an ffmpeg mid-frame extract), or "
              "register a model serving video.understand")


class LiveJudge:
    """A ``JudgeFn`` over the oracle's own judge path.

    Everything model-facing is REUSED from k90c: ``evaluation._resolve_judge_
    route`` (the catalog's eligibility gates and default-model policy),
    ``evaluation.run_judge`` (the dispatch, the no-think wrapper, the tolerant
    verdict parse, the degrade-to-unavailable discipline). This class only
    decides WHICH capability and WHICH artifact — it re-implements none of it,
    which is why a monkeypatched ``evaluation`` seam drives it in tests exactly
    as it drives ``/oracle/route``.

    ``model`` is resolved at bind time so ``evaluate_take`` can refuse a
    self-judging configuration BEFORE spending a dispatch."""

    __slots__ = ("capability", "video_capability", "frames", "model", "reasons")

    def __init__(self, *, capability: str = JUDGE_CAPABILITY,
                 video_capability: str = VIDEO_JUDGE_CAPABILITY,
                 frames: Callable[[str], str | None] | None = None,
                 model: str | None = None) -> None:
        self.capability = capability
        self.video_capability = video_capability
        self.frames = frames
        route = evaluation._resolve_judge_route(capability)   # noqa: SLF001
        self.reasons: tuple[str, ...] = tuple(
            getattr(route, "reasons", ()) or ()) if route else (
            "judge route resolution raised",)
        resolved = (route.model_id if route is not None
                    and route.execution == "execute" else None)
        self.model = model or resolved

    def _unavailable(self, request: JudgeRequest, detail: str) -> JudgeResult:
        return JudgeResult(judge=f"{request.name}:{self.model or 'unbound'}",
                           verdict="unavailable", score=None, rationale=detail)

    def __call__(self, request: JudgeRequest) -> JudgeResult:
        capability = self.capability
        ref = request.artifact_ref
        if request.artifact_kind == "video":
            route = evaluation._resolve_judge_route(                # noqa: SLF001
                self.video_capability)
            if route is not None and route.execution == "execute":
                capability = self.video_capability
            elif self.frames is not None:
                try:
                    sampled = self.frames(request.artifact_ref)
                except Exception as exc:     # noqa: BLE001 — degrade honestly
                    return self._unavailable(
                        request, f"frame sampler raised "
                                 f"({type(exc).__name__}: {exc})"[:300])
                if not sampled:
                    return self._unavailable(
                        request, "frame sampler returned no still for this clip")
                ref = str(sampled)
            else:
                reasons = "; ".join(getattr(route, "reasons", ()) or ()) \
                    if route is not None else "route resolution raised"
                return self._unavailable(
                    request,
                    f"{self.video_capability} is not eligible ({reasons}); "
                    f"{_FRAME_GAP}")
        rubric = evaluation.Rubric(name=request.name, kind=request.kind,
                                   judge_capability=capability,
                                   judged_artifact="image")
        goal = GoalSpec(objective=request.prompt, raw_prompt=request.prompt,
                        capability=capability)
        result = evaluation.run_judge(rubric, goal, [{"kind": "image",
                                                      "uri": ref}])
        if result is None:
            return self._unavailable(
                request, f"nothing judgeable at {ref!r} (the judge grades a "
                         f"readable image file)")
        return result


def bind_live_judge(*, capability: str = JUDGE_CAPABILITY,
                    video_capability: str = VIDEO_JUDGE_CAPABILITY,
                    frames: Callable[[str], str | None] | None = None,
                    model: str | None = None) -> LiveJudge:
    """The live footage judge, bound through the catalog (see
    :class:`LiveJudge`). Binding never dispatches and never raises: an
    ineligible judge yields a ``LiveJudge`` whose ``model`` is ``None`` and
    whose ``reasons`` carry the catalog's own words, and every call it receives
    answers ``verdict="unavailable"`` — which ``evaluate_take`` records as
    unscored evidence instead of a passing take."""
    return LiveJudge(capability=capability, video_capability=video_capability,
                     frames=frames, model=model)


# ---------------------------------------------------------------------------
# Stage 19 — the whole-result pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinalReport:
    """Stage 19's answer: a machine-readable ``Scorecard`` plus the summary
    lines an operator actually reads. Both, always — the card is the gate and
    the summary is the explanation, and shipping only one of them is how a
    refusal becomes a shrug."""

    scorecard: Scorecard
    summary: tuple[str, ...] = ()
    uncovered_segments: tuple[str, ...] = ()
    omitted_line_ids: tuple[str, ...] = ()
    pacing_outliers: tuple[tuple[str, float], ...] = ()
    missing_color_scenes: tuple[str, ...] = ()
    open_segments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", tuple(str(s) for s in self.summary))
        object.__setattr__(self, "pacing_outliers",
                           tuple((str(s), quantize(d))
                                 for s, d in self.pacing_outliers))

    @property
    def ok(self) -> bool:
        return self.scorecard.hard_pass

    @property
    def repair_code(self) -> RepairCode | None:
        return self.scorecard.repair_code

    def to_dict(self) -> dict[str, Any]:
        return {"scorecard": self.scorecard.to_dict(),
                "summary": list(self.summary),
                "uncovered_segments": list(self.uncovered_segments),
                "omitted_line_ids": list(self.omitted_line_ids),
                "pacing_outliers": [[s, d] for s, d in self.pacing_outliers],
                "missing_color_scenes": list(self.missing_color_scenes),
                "open_segments": list(self.open_segments),
                "ok": self.ok}


def _unscored_check(name: str, kind: CheckKind, detail: str,
                    threshold: Any = None) -> Check:
    return Check(name=name, kind=kind, value=None, threshold=threshold,
                 passed=True, detail=f"{_UNSCORED}{detail}")


def final_consistency_report(
        plan: PostProductionPlan,
        takes: Any = (),
        transcribe: TranscribeFn | None = None,
        *,
        shots: Any = None,
        lines: Any = None,
        video_ref: str | None = None,
        audio_master: AudioMaster | None = None,
        pacing_tolerance: float = DEFAULT_PACING_TOLERANCE_S,
) -> FinalReport:
    """doc Stage 19 — evaluate the WHOLE result, not another shot.

    The checks, and why each is here rather than per-shot:

    ``assembly.coverage``  every locked segment is in the cut. A per-shot pass
        cannot see a segment that is absent — absence has no artifact to judge.
    ``assembly.regeneration_open``  no segment is still out for regeneration.
    ``speech.lines_present``  the LOCKED dialogue survived into the assembled
        result, verified by re-transcribing it (k98, per line, in order).
    ``pacing.window_fit``  no shot's screen time deviates from its locked
        window by more than ``pacing_tolerance``.
    ``audio.bed_fit``  the picture is at least as long as the audio master
        (k106's documented trap: audio-derived windows with ``pad_s=0`` do not
        cover the inter-line pauses, and the picture comes up short).
    ``color.targets``  every scene whose shot plan declared a look has a
        grading target in the plan.

    Every check whose input is absent is recorded UNSCORED and counted out of
    ``confidence`` — never as a pass."""
    if not isinstance(plan, PostProductionPlan):
        raise PostProductionError(
            f"final_consistency_report takes a PostProductionPlan, got "
            f"{type(plan).__name__}")
    briefs = shot_briefs(shots) if shots is not None else ()
    by_segment = {b.segment_id: b for b in briefs}
    supplied = group_takes(takes)
    if audio_master is None and shots is not None:
        candidate = getattr(shots, "audio_master", None)
        if isinstance(candidate, AudioMaster):
            audio_master = candidate

    checks: list[Check] = []
    summary: list[str] = []

    # 1 — coverage -------------------------------------------------------
    locked = tuple(plan.locked_segments) or tuple(b.segment_id for b in briefs)
    uncovered = plan.edl.missing(locked)
    if not locked:
        checks.append(_unscored_check(
            "assembly.coverage", CheckKind.TECHNICAL,
            "no locked segment list is known (the plan carries none and no "
            "shots= were supplied), so coverage cannot be verified"))
        summary.append("coverage: UNSCORED — nothing declared what the cut owed")
    else:
        covered = len(locked) - len(uncovered)
        checks.append(Check(
            name="assembly.coverage", kind=CheckKind.TECHNICAL,
            value=covered, threshold=len(locked), passed=not uncovered,
            detail=(f"the cut plays {covered}/{len(locked)} locked segment(s)"
                    + (f"; missing: {', '.join(uncovered)}" if uncovered
                       else ""))))
        summary.append(
            f"coverage: {covered}/{len(locked)} locked segment(s) in the cut"
            + (f" — MISSING {', '.join(uncovered)}" if uncovered else ""))

    # 2 — open regeneration ----------------------------------------------
    open_segments = plan.open_segments
    checks.append(Check(
        name="assembly.regeneration_open", kind=CheckKind.TECHNICAL,
        value=len(open_segments), threshold=0, passed=not open_segments,
        detail=("no open regeneration notes" if not open_segments else
                "; ".join(f"{n.segment_id} -> {n.repair_code.value}"
                          for n in plan.regeneration_notes
                          if n.segment_id in set(open_segments)))))
    for note in plan.regeneration_notes:
        summary.append(
            f"regeneration: {note.segment_id} -> {note.repair_code.value}"
            f"{' (open, not in the cut)' if note.segment_id in set(open_segments) else ''}"
            f" — {note.correction}")

    # 3 — dialogue round trip --------------------------------------------
    texts_by_id = _mapping_of_text(lines)
    expected_ids: list[str] = []
    for segment_id in plan.edl.segment_ids:
        brief = by_segment.get(segment_id)
        for line_id in (brief.line_ids if brief else ()):
            if line_id not in expected_ids:
                expected_ids.append(line_id)
    if not expected_ids:
        expected_ids = [i for i in texts_by_id]
    expected = [texts_by_id[i] for i in expected_ids if i in texts_by_id]
    omitted: tuple[str, ...] = ()

    if transcribe is None or not expected:
        reason = ("no round-trip transcriber was supplied"
                  if transcribe is None else
                  "no locked line text was supplied (pass lines=)")
        checks.append(_unscored_check("speech.lines_present", CheckKind.SPEECH,
                                      reason, threshold=len(expected) or None))
        summary.append(f"dialogue: UNSCORED — {reason}")
    elif not video_ref:
        checks.append(_unscored_check(
            "speech.lines_present", CheckKind.SPEECH,
            "no assembled artifact to re-transcribe (pass video_ref=)",
            threshold=len(expected)))
        summary.append("dialogue: UNSCORED — no assembled artifact to "
                       "re-transcribe")
    else:
        try:
            words = transcribe(video_ref)
        except Exception as exc:      # noqa: BLE001 — an ASR fault degrades
            checks.append(_unscored_check(
                "speech.lines_present", CheckKind.SPEECH,
                f"round-trip transcription failed ({type(exc).__name__}: "
                f"{exc})"[:400], threshold=len(expected)))
            summary.append("dialogue: UNSCORED — round-trip transcription "
                           "failed")
        else:
            check = speech.check_lines_present(expected, words)
            checks.append(check)
            # Per line, by NAME: the aggregate above is ordered and
            # authoritative (k98 matches with an advancing cursor); this second
            # pass is unordered and exists only to name which line went missing.
            present_ids = [i for i in expected_ids if i in texts_by_id]
            omitted = tuple(
                line_id for line_id in present_ids
                if not speech.check_lines_present([texts_by_id[line_id]],
                                                  words).passed)
            summary.append(
                f"dialogue: {check.value}/{check.threshold} locked line(s) "
                f"survive the round-trip transcript"
                + (f" — OMITTED {', '.join(omitted)}" if omitted else ""))

    # 4 — pacing ----------------------------------------------------------
    outliers: list[tuple[str, float]] = []
    measured = 0
    for decision in plan.edl.decisions:
        brief = by_segment.get(decision.segment_id)
        if brief is None or brief.duration_s <= 0:
            continue
        screen = decision.duration_s
        if screen is None:
            take = _find_take(supplied, decision)
            screen = take.duration_s if take is not None else None
        if screen is None:
            continue
        measured += 1
        deviation = quantize(screen - brief.duration_s)
        if abs(deviation) > pacing_tolerance:
            outliers.append((decision.segment_id, deviation))
    if not measured:
        checks.append(_unscored_check(
            "pacing.window_fit", CheckKind.SYNC,
            "no shot had both a measured screen time and a locked window "
            "(pass shots= and measured takes)", threshold=pacing_tolerance))
        summary.append("pacing: UNSCORED — nothing measurable")
    else:
        worst = max((abs(d) for _s, d in outliers), default=0.0)
        checks.append(Check(
            name="pacing.window_fit", kind=CheckKind.SYNC, value=worst,
            threshold=pacing_tolerance, passed=not outliers,
            detail=(f"{measured} shot(s) measured against their locked windows"
                    if not outliers else
                    "; ".join(f"{s} deviates {d:+.3f}s" for s, d in outliers))))
        for segment_id, deviation in outliers:
            summary.append(
                f"pacing: {segment_id} deviates {deviation:+.3f}s from its "
                f"locked window (tolerance ±{pacing_tolerance:g}s)")
        if not outliers:
            summary.append(f"pacing: {measured} shot(s) within "
                           f"±{pacing_tolerance:g}s of their locked windows")

    # 5 — the audio bed ---------------------------------------------------
    total = plan.edl.total_seconds
    if audio_master is None or total is None:
        checks.append(_unscored_check(
            "audio.bed_fit", CheckKind.SYNC,
            ("no AudioMaster was supplied" if audio_master is None else
             "the cut carries unmeasured rows, so its length is unknown")))
        summary.append("audio bed: UNSCORED")
    else:
        fit = speech.check_duration_fit(
            audio_seconds=audio_master.total_seconds, shot_seconds=total,
            tolerance=pacing_tolerance)
        checks.append(replace(fit, name="audio.bed_fit"))
        summary.append(
            f"audio bed: {total:.3f}s of picture against "
            f"{audio_master.total_seconds:.3f}s of audio master"
            + ("" if fit.passed else " — the picture comes up SHORT"))

    # 6 — color targets ---------------------------------------------------
    declared = tuple(b.scene_key for b in briefs if b.declares_color)
    declared = tuple(dict.fromkeys(declared))
    if not briefs:
        checks.append(_unscored_check(
            "color.targets", CheckKind.SEMANTIC,
            "no shot plan was supplied, so nothing declares a look (pass "
            "shots=)"))
        summary.append("color: UNSCORED — no shot plan supplied")
        missing_color: tuple[str, ...] = ()
    else:
        missing_color = tuple(s for s in declared
                              if s not in plan.color_targets)
        checks.append(Check(
            name="color.targets", kind=CheckKind.SEMANTIC,
            value=len(declared) - len(missing_color), threshold=len(declared),
            passed=not missing_color,
            detail=(f"{len(declared)} scene(s) declared a look"
                    + (f"; no grading target for {', '.join(missing_color)}"
                       if missing_color else "; all have grading targets"))))
        summary.append(
            f"color: {len(declared) - len(missing_color)}/{len(declared)} "
            f"declared scene(s) have a grading target"
            + (f" — MISSING {', '.join(missing_color)}" if missing_color
               else ""))

    hard_pass = all(c.passed for c in checks)
    scored = [c for c in checks if not speech.is_unscored(c)]
    confidence = round(len(scored) / len(checks), 3) if checks else 1.0
    code = None
    if not hard_pass:
        failing = {c.name for c in checks if not c.passed}
        for name in _REPORT_PRIORITY:
            if name in failing:
                code = REPORT_REPAIR[name]
                break
        if code is REPORT_REPAIR["assembly.coverage"] and uncovered:
            note = plan.note_for(uncovered[0])
            if note is not None:
                code = note.repair_code
    diagnoses = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    unscored_names = [c.name for c in checks if speech.is_unscored(c)]
    if unscored_names:
        diagnoses.append("unscored (no evidence): " + ", ".join(unscored_names))
    summary.append(f"VERDICT: {'PASS' if hard_pass else 'FAIL'}"
                   + (f" (repair: {code.value})" if code else "")
                   + f"; confidence {confidence:g}")

    card = Scorecard(
        hard_pass=hard_pass, checks=tuple(checks), judge_results=(),
        confidence=confidence, diagnosis="; ".join(diagnoses) or None,
        repair_code=code,
        recommended_repair=(None if hard_pass else RECOMMENDED_REPAIR.get(
            code, "regenerate the failing segment(s) from their canonical "
                  "SegmentSpecs")))
    return FinalReport(
        scorecard=card, summary=tuple(summary), uncovered_segments=uncovered,
        omitted_line_ids=omitted, pacing_outliers=tuple(outliers),
        missing_color_scenes=missing_color, open_segments=open_segments)


def _find_take(supplied: Mapping[str, tuple[Take, ...]],
               decision: EditDecision) -> Take | None:
    for take in supplied.get(decision.segment_id, ()):
        if take.attempt_id == decision.take.attempt_id:
            return take
    return None


__all__ = [
    "COLOR_CUES",
    "DEFAULT_PACING_TOLERANCE_S",
    "INSTANT_TRANSITIONS",
    "JUDGE_CAPABILITY",
    "MUSIC_ROLES",
    "RECOMMENDED_REPAIR",
    "REPORT_REPAIR",
    "SOUND_KINDS",
    "TAKE_KINDS",
    "TAKE_REPAIR",
    "TRANSITIONS",
    "VIDEO_JUDGE_CAPABILITY",
    "EDLRefused",
    "EditDecision",
    "EditDecisionList",
    "FinalReport",
    "JudgeConflict",
    "JudgeRequest",
    "LiveJudge",
    "MusicCue",
    "PlanRefused",
    "PostProductionError",
    "PostProductionPlan",
    "RegenerationNote",
    "SPATIAL_CAMERA_ANGLE_DEG_MAX",
    "SPATIAL_ENTITY_TOLERANCE_M",
    "ShotBrief",
    "SpatialContradiction",
    "SoundCue",
    "Take",
    "bind_live_judge",
    "build_postproduction_plan",
    "coerce_judge_result",
    "declared_color_scenes",
    "evaluate_take",
    "final_consistency_report",
    "group_takes",
    "has_color_cue",
    "repair_codes_from",
    "shot_brief",
    "shot_briefs",
    "spatial_continuity_checks",
    "shot_spec_block",
    "take_repair_code",
    "take_rubrics",
    "takes_from_run_state",
]
