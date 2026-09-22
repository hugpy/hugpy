"""Storyboards as RENDERED FRAMES (k124) — the second filmmaking delta.

The script-first brief puts storyboards in pre-production::

    Generate: shot list, STORYBOARD DESCRIPTIONS OR STORYBOARD-GENERATION
    PROMPTS, camera placement and movement, lens and framing guidance ...

k110 produced the prompts. This module produces the FRAMES — cheap, low-res
stills rendered at PRE-PRODUCTION time, judged against the shot's own
acceptance rubric, accepted or rejected by an operator who can see them, and
then LOCKED with the production so the shoot has a carried visual reference
rather than a paragraph of prose.

WHY THE FRAME IS THE POINT (the interim principle, 2026-08-21). "Between a
prompt and its deliverable there is a structured, persisted, inspectable
production process — never a black box." A storyboard description is a
description of an interim; a rendered, judged, accepted frame IS one. It is the
first artifact in the pipeline an operator can look at and say "no, not that"
before a single second of video is spent.

WHAT IS HERE

``StoryboardFrame``
    ``(shot_id, prompt, artifact_ref, scorecard_ref, accepted)`` plus the
    evidence that decided it: the seed, the generating model, the judging
    model, the verdict and the reason. A REJECTED frame must carry a reason —
    k108's rule for ``Take``, for the same cause: a rejection nobody wrote down
    cannot become a documented correction. An UNJUDGED frame is not accepted
    and says so; unscored never reads as passed.

``Storyboard``
    The board: every frame, ``per_shot``, the shot plan it was drawn against,
    and the lock gate. One ACCEPTED frame per shot, enforced — the reference
    the shoot conditions on has to be unambiguous.

``render_storyboards``
    ``per_shot`` candidates for every shot, judged against the shot rubric
    through k108's ``JudgeRequest`` / ``coerce_judge_result`` (imported and
    called, never re-implemented), first hard-pass wins.

``storyboard_prompt``
    The deterministic prompt: the shot specification (k108's ``shot_spec_block``
    for the parts a judge also reads) plus the world's palette and lighting
    when a k124 ``WorldBuild`` is supplied. Same shot, same prompt, same seed,
    same frame.

``bind_storyboard_image``
    The live LOW-RES binding over ``image.generate``, through the same two
    functions everything else in this tree dispatches with
    (``router.resolve_route`` + ``runtime.execute_route``). Cheap is not a
    comment here: it is ``width``/``height`` overrides on the route.

THE STYLE CHOICE, FLAGGED. The default style is ``color_key``, not the classic
line-art ``sketch``, and the reason is downstream: an accepted frame is passed
to the shoot as the keyframe generation's reference. A monochrome sketch used
as an init-image or style reference biases the keyframe toward line art, which
is exactly the wrong bias. ``sketch`` is available for boards that are only
ever going to be looked at.

Offline by construction: stdlib + this package. The image generator and the
judge are injected callables, so the whole module is testable without a GPU, a
worker, a registry or a network.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from hugpy_oracle.contracts import Check, CheckKind, RepairCode, Scorecard
from hugpy_oracle.audio_master import scorecard_digest
from hugpy_oracle.production import ContentAddressed, ContinuityState, ProductionError
from hugpy_oracle.postproduction import (
    JUDGE_CAPABILITY,
    JudgeConflict,
    JudgeRequest,
    ShotBrief,
    coerce_judge_result,
    shot_brief,
    shot_briefs,
    shot_spec_block,
)
from hugpy_oracle import speech

# BORROWED, not re-copied — k108's and k110's discipline (a test asserts these
# are the same objects as ``production``'s).
from hugpy_oracle.production import _require_text as require_text
from hugpy_oracle.production import _str_tuple as str_tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Closed vocabularies and tunables — named, never magic
# ---------------------------------------------------------------------------

#: How a storyboard frame is drawn. ``color_key`` is a flat, low-detail COLOR
#: composition frame; ``sketch`` is the classic monochrome line-art panel. See
#: the style note in the module docstring for why the default is the former.
STORYBOARD_STYLES: tuple[str, ...] = ("color_key", "sketch")

#: The style preamble each style puts at the head of its prompt. Deterministic
#: text, so the same shot always asks for the same picture.
STYLE_PREAMBLE: dict[str, str] = {
    "color_key": ("storyboard color key frame: flat, simplified color "
                  "composition, no lettering, no panel borders, no text"),
    "sketch": ("storyboard panel: rough monochrome marker sketch, clean line "
               "work, no lettering, no panel borders, no text"),
}

#: Who accepted a frame. ``""`` means nobody has — which is the state an
#: unjudged frame stays in, because silent success is forbidden.
ACCEPTED_BY: tuple[str, ...] = ("", "judge", "operator")

#: The default board size. LOW-RES is the point: a storyboard pass over twelve
#: shots must cost a fraction of one keyframe, or nobody will run it.
DEFAULT_WIDTH: int = 384
DEFAULT_HEIGHT: int = 384

#: Candidates per shot. One is the default because a storyboard is a
#: CONVERSATION with the operator, not a fan-out to be auto-selected from.
DEFAULT_PER_SHOT: int = 1

#: The judge's pass mark, on k90c's 0-10 scale — the same threshold
#: ``evaluation``/``performance`` use for a keyframe.
DEFAULT_THRESHOLD: int = 6

#: The seed namespace, so a storyboard seed and a keyframe seed for the same
#: shot never collide (they are different pictures of the same shot and must be
#: reproducible independently).
SEED_SALT: str = "k124:storyboard:1"

#: The rubric name a frame is judged under. Deliberately NOT ``shot.intent``:
#: a storyboard is judged on whether it is the SPECIFIED COMPOSITION, and
#: reusing k108's take rubric name would make a board frame and a finished take
#: indistinguishable in a scorecard.
FRAME_RUBRIC: str = "storyboard.frame"


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class StoryboardError(ProductionError):
    """Base for every refusal in this module (a ``ValueError`` by inheritance)."""


class StoryboardRefused(StoryboardError):
    """The board does not describe a usable set of references."""


GenImageFn = Callable[..., Any]
JudgeImageFn = Callable[..., Any]


def frame_seed(shot_id: str, index: int, salt: str = SEED_SALT) -> int:
    """A deterministic seed for one candidate of one shot, in ``[0, 2**32)``.

    Same derivation shape as ``segments.segment_seed`` (sha256 over a
    colon-joined key, first 8 hex digits) under this module's own salt, so a
    storyboard candidate is reproducible and can never draw the same number as
    the keyframe of the same shot."""
    payload = f"{salt}:{shot_id}:{int(index)}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# The artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryboardFrame(ContentAddressed):
    """One rendered storyboard frame, as a record with its evidence attached.

    ``scorecard_ref`` is the digest of the card that judged it (k102's
    ``scorecard_digest``, so the same card digests identically wherever it is
    stored). ``None`` means nobody judged this frame, which is why
    ``accepted`` cannot be True with ``accepted_by == "judge"`` and no card:
    an unscored frame that read as approved is the exact failure the interim
    principle names."""

    shot_id: str
    prompt: str
    artifact_ref: str
    scorecard_ref: str | None = None
    accepted: bool = False
    index: int = 0
    seed: int | None = None
    generator_model: str | None = None
    judge_model: str | None = None
    verdict: str = "unscored"
    score: float | None = None
    reason: str = ""
    accepted_by: str = ""
    style: str = "color_key"
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        require_text(self.shot_id, "StoryboardFrame.shot_id")
        require_text(self.prompt, "StoryboardFrame.prompt")
        require_text(self.artifact_ref, "StoryboardFrame.artifact_ref")
        for name in ("scorecard_ref", "generator_model", "judge_model"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_text(
                    value, f"StoryboardFrame.{name}"))
        if not isinstance(self.accepted, bool):
            raise StoryboardRefused("StoryboardFrame.accepted must be a bool")
        if isinstance(self.index, bool) or not isinstance(self.index, int) \
                or self.index < 0:
            raise StoryboardRefused(
                f"StoryboardFrame.index must be a non-negative int, got "
                f"{self.index!r}")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int) \
                    or self.seed < 0:
                raise StoryboardRefused(
                    f"StoryboardFrame.seed must be a non-negative int when "
                    f"set, got {self.seed!r}")
        for name in ("verdict", "reason"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        if self.accepted_by not in ACCEPTED_BY:
            raise StoryboardRefused(
                f"StoryboardFrame.accepted_by must be one of "
                f"{[a for a in ACCEPTED_BY]}, got {self.accepted_by!r}")
        if self.style not in STORYBOARD_STYLES:
            raise StoryboardRefused(
                f"StoryboardFrame.style must be one of "
                f"{list(STORYBOARD_STYLES)}, got {self.style!r}")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) \
                        or value <= 0:
                    raise StoryboardRefused(
                        f"StoryboardFrame.{name} must be a positive int when "
                        f"set, got {value!r}")
        if self.accepted and not self.accepted_by:
            raise StoryboardRefused(
                f"StoryboardFrame({self.frame_id}) is accepted by nobody — an "
                f"acceptance with no author cannot be reviewed or reversed")
        if not self.accepted and self.accepted_by:
            raise StoryboardRefused(
                f"StoryboardFrame({self.frame_id}) records accepted_by="
                f"{self.accepted_by!r} but is not accepted")
        if self.accepted and self.accepted_by == "judge" and \
                self.scorecard_ref is None:
            raise StoryboardRefused(
                f"StoryboardFrame({self.frame_id}) claims the JUDGE accepted "
                f"it but carries no scorecard — an unscored frame that reads "
                f"as approved is the failure this field exists to prevent")
        if not self.accepted and not self.reason:
            raise StoryboardRefused(
                f"StoryboardFrame({self.frame_id}) is not accepted and carries "
                f"no reason — a rejection nobody wrote down cannot become a "
                f"documented correction")

    @property
    def frame_id(self) -> str:
        return f"{self.shot_id}#{self.index}"

    @property
    def judged(self) -> bool:
        return self.verdict not in ("", "unscored", "unavailable")

    def accept(self, *, by: str = "operator", note: str = "") -> "StoryboardFrame":
        if by not in ("judge", "operator"):
            raise StoryboardRefused(f"accept(by=) must be 'judge' or "
                                    f"'operator', got {by!r}")
        return replace(self, accepted=True, accepted_by=by,
                       reason=note or self.reason)

    def reject(self, reason: str) -> "StoryboardFrame":
        text = str(reason or "").strip()
        if not text:
            raise StoryboardRefused(
                "reject() needs a reason: a rejected frame with nothing "
                "written down cannot become a documented correction")
        return replace(self, accepted=False, accepted_by="", reason=text)

    def to_dict(self) -> dict[str, Any]:
        return {"shot_id": self.shot_id, "prompt": self.prompt,
                "artifact_ref": self.artifact_ref,
                "scorecard_ref": self.scorecard_ref, "accepted": self.accepted,
                "index": self.index, "seed": self.seed,
                "generator_model": self.generator_model,
                "judge_model": self.judge_model, "verdict": self.verdict,
                "score": self.score, "reason": self.reason,
                "accepted_by": self.accepted_by, "style": self.style,
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StoryboardFrame":
        return cls(shot_id=d["shot_id"], prompt=d["prompt"],
                   artifact_ref=d["artifact_ref"],
                   scorecard_ref=d.get("scorecard_ref"),
                   accepted=bool(d.get("accepted", False)),
                   index=int(d.get("index", 0)), seed=d.get("seed"),
                   generator_model=d.get("generator_model"),
                   judge_model=d.get("judge_model"),
                   verdict=d.get("verdict", "unscored"), score=d.get("score"),
                   reason=d.get("reason", ""),
                   accepted_by=d.get("accepted_by", ""),
                   style=d.get("style", "color_key"), width=d.get("width"),
                   height=d.get("height"))


@dataclass(frozen=True, slots=True)
class Storyboard(ContentAddressed):
    """The board: every frame drawn for this production, and the lock gate.

    ``shot_ids`` is the COVERAGE DENOMINATOR — the shots the board was drawn
    against. It is what makes "every shot has an accepted frame" a checkable
    statement instead of a comparison against whatever happens to be on the
    board (k108's ``locked_segments`` idea, one artifact over).

    ``locked`` refuses two things: a board with no accepted frame at all (a
    lock over nothing), and TWO accepted frames for one shot — the reference
    the shoot conditions on has to be unambiguous."""

    frames: tuple[StoryboardFrame, ...] = ()
    shot_ids: tuple[str, ...] = ()
    per_shot: int = DEFAULT_PER_SHOT
    shot_plan_digest: str | None = None
    style: str = "color_key"
    note: str = ""
    locked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))
        for frame in self.frames:
            if not isinstance(frame, StoryboardFrame):
                raise StoryboardRefused(
                    f"Storyboard.frames takes StoryboardFrame, got "
                    f"{type(frame).__name__}")
        ids = [f.frame_id for f in self.frames]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise StoryboardRefused(
                f"Storyboard has two frames under id(s) {duplicates} — a frame "
                f"id is (shot, candidate index) and two of them cannot be the "
                f"same picture")
        object.__setattr__(self, "shot_ids",
                           str_tuple(self.shot_ids, "Storyboard.shot_ids"))
        if isinstance(self.per_shot, bool) or not isinstance(self.per_shot, int) \
                or self.per_shot < 1:
            raise StoryboardRefused(
                f"Storyboard.per_shot must be an int >= 1, got "
                f"{self.per_shot!r} — a board of zero candidates draws nothing")
        if self.shot_plan_digest is not None:
            object.__setattr__(self, "shot_plan_digest", require_text(
                self.shot_plan_digest, "Storyboard.shot_plan_digest"))
        if self.style not in STORYBOARD_STYLES:
            raise StoryboardRefused(
                f"Storyboard.style must be one of {list(STORYBOARD_STYLES)}, "
                f"got {self.style!r}")
        object.__setattr__(self, "note", str(self.note or ""))
        if not isinstance(self.locked, bool):
            raise StoryboardRefused("Storyboard.locked must be a bool")
        unknown = sorted({f.shot_id for f in self.frames} - set(self.shot_ids)) \
            if self.shot_ids else []
        if unknown:
            raise StoryboardRefused(
                f"Storyboard carries frame(s) for shot(s) {unknown} that are "
                f"not in its shot plan — the board and the plan are from "
                f"different productions")
        doubled = sorted({s for s in self.accepted_shots
                          if len(self.accepted_for(s)) > 1})
        if doubled:
            raise StoryboardRefused(
                f"shot(s) {doubled} have more than one ACCEPTED frame — the "
                f"reference the shoot conditions on has to be unambiguous")
        if self.locked and not self.accepted_frames:
            raise StoryboardRefused(
                "cannot lock a storyboard with no accepted frame: a lock over "
                "nothing is not a lock")

    # -- reading -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    @property
    def accepted_frames(self) -> tuple[StoryboardFrame, ...]:
        return tuple(f for f in self.frames if f.accepted)

    @property
    def accepted_shots(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.shot_id for f in self.accepted_frames))

    @property
    def artifact_refs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.artifact_ref for f in self.frames))

    def frames_for(self, shot_id: str) -> tuple[StoryboardFrame, ...]:
        return tuple(f for f in self.frames if f.shot_id == shot_id)

    def accepted_for(self, shot_id: str) -> tuple[StoryboardFrame, ...]:
        return tuple(f for f in self.frames_for(shot_id) if f.accepted)

    def reference_for(self, shot_id: str) -> str | None:
        """The artifact the SHOOT should condition on for this shot, or None.

        ``None`` is the honest answer for a shot whose frames were all
        rejected: conditioning a keyframe on a frame an operator turned down
        would carry the rejection forward as if it were a decision."""
        accepted = self.accepted_for(shot_id)
        return accepted[0].artifact_ref if accepted else None

    @property
    def references(self) -> dict[str, str]:
        """``{shot_id: artifact_ref}`` for every accepted frame — the mapping
        the keyframe stage takes."""
        return {f.shot_id: f.artifact_ref for f in self.accepted_frames}

    def missing(self, shot_ids: Sequence[str] | None = None) -> tuple[str, ...]:
        """Shots with no accepted frame, in the order asked."""
        wanted = tuple(shot_ids) if shot_ids is not None else self.shot_ids
        accepted = set(self.accepted_shots)
        return tuple(s for s in dict.fromkeys(wanted) if s not in accepted)

    # -- deciding ----------------------------------------------------------

    def accept(self, shot_id: str, index: int = 0, *,
               by: str = "operator", note: str = "") -> "Storyboard":
        """Accept one frame and UN-accept every sibling candidate of the same
        shot. Returns a NEW board — the one already locked into a production is
        never mutated."""
        if self.locked:
            raise StoryboardRefused(
                f"storyboard is LOCKED; accepting {shot_id}#{index} now would "
                f"change a reference the production already locked. Render a "
                f"new board and revise the lock (Stage 10)")
        frames = list(self.frames)
        target = None
        for position, frame in enumerate(frames):
            if frame.shot_id != shot_id:
                continue
            if frame.index == index:
                target = position
            elif frame.accepted:
                frames[position] = frame.reject(
                    f"superseded by {shot_id}#{index}")
        if target is None:
            raise StoryboardRefused(
                f"no storyboard frame {shot_id}#{index} on this board "
                f"(has: {[f.frame_id for f in self.frames]})")
        frames[target] = frames[target].accept(by=by, note=note)
        return replace(self, frames=tuple(frames))

    def reject(self, shot_id: str, index: int, reason: str) -> "Storyboard":
        if self.locked:
            raise StoryboardRefused(
                f"storyboard is LOCKED; rejecting {shot_id}#{index} now would "
                f"change a reference the production already locked")
        frames = list(self.frames)
        for position, frame in enumerate(frames):
            if frame.shot_id == shot_id and frame.index == index:
                frames[position] = frame.reject(reason)
                return replace(self, frames=tuple(frames))
        raise StoryboardRefused(
            f"no storyboard frame {shot_id}#{index} on this board")

    def lock(self, *, require_coverage: bool = False) -> "Storyboard":
        """The pre-production gate. ``require_coverage=True`` additionally
        refuses while any planned shot has no accepted frame — the right
        setting when a production intends every shot to be boarded, and the
        wrong default, because a partial board is a real and useful thing."""
        if require_coverage:
            missing = self.missing()
            if missing:
                raise StoryboardRefused(
                    f"cannot lock a storyboard while shot(s) {list(missing)} "
                    f"have no accepted frame (require_coverage=True)")
        return self if self.locked else replace(self, locked=True)

    def to_dict(self) -> dict[str, Any]:
        return {"frames": [f.to_dict() for f in self.frames],
                "shot_ids": list(self.shot_ids), "per_shot": self.per_shot,
                "shot_plan_digest": self.shot_plan_digest, "style": self.style,
                "note": self.note, "locked": self.locked}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Storyboard":
        return cls(frames=tuple(StoryboardFrame.from_dict(f)
                                for f in d.get("frames", ())),
                   shot_ids=tuple(d.get("shot_ids", ())),
                   per_shot=int(d.get("per_shot", DEFAULT_PER_SHOT)),
                   shot_plan_digest=d.get("shot_plan_digest"),
                   style=d.get("style", "color_key"), note=d.get("note", ""),
                   locked=bool(d.get("locked", False)))


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def storyboard_prompt(brief: Any, world: Any = None, *,
                      style: str = "color_key",
                      continuity: ContinuityState | None = None) -> str:
    """The deterministic storyboard prompt for one shot.

    Built from the SAME normalizer k108 judges against (``ShotBrief``), so the
    picture that is asked for and the specification it is graded against cannot
    drift apart. When a k124 ``WorldBuild`` is supplied, the shot's scene
    palette and the film's lighting style are appended — that is the whole
    reason the world is an artifact and not a paragraph: it reaches the
    renderer as text a model can act on."""
    if style not in STORYBOARD_STYLES:
        raise StoryboardRefused(
            f"storyboard_prompt(style=) must be one of "
            f"{list(STORYBOARD_STYLES)}, got {style!r}")
    normalized = brief if isinstance(brief, ShotBrief) else shot_brief(brief)
    parts = [STYLE_PREAMBLE[style], ""]
    if normalized.prompt:
        parts.append(normalized.prompt)
    parts.append(shot_spec_block(normalized, continuity))
    palette = _world_lines(world, normalized)
    if palette:
        parts.append("")
        parts.extend(palette)
    return "\n".join(p for p in parts if p is not None).strip()


def _world_lines(world: Any, brief: ShotBrief) -> list[str]:
    """The world's contribution to one shot's prompt, or an empty list.

    Duck-typed against k124's ``WorldBuild`` (``palette_for`` / ``card_for`` /
    ``lighting_style`` / ``tone_words``) rather than imported, so this module
    stays usable with a world-shaped mapping and neither module has to import
    the other."""
    if world is None:
        return []
    lines: list[str] = ["WORLD:"]
    entry = None
    getter = getattr(world, "palette_for", None)
    if callable(getter) and brief.scene_ref:
        entry = getter(brief.scene_ref)
    if entry is not None:
        lines.append(f"- palette ({entry.basis}): {entry.full_target}")
        card_getter = getattr(world, "card_for", None)
        card = card_getter(entry.location) if callable(card_getter) else None
        if card is not None and card.dressing:
            lines.append(f"- set dressing at {card.name}: "
                         + ", ".join(card.dressing))
    style = str(getattr(world, "lighting_style", "") or "")
    if style:
        lines.append(f"- lighting style: {style}")
    tone = tuple(getattr(world, "tone_words", ()) or ())
    if tone:
        lines.append(f"- tone: {', '.join(tone)}")
    return lines if len(lines) > 1 else []


def frame_rubric(brief: ShotBrief, artifact_ref: str,
                 continuity: ContinuityState | None = None) -> JudgeRequest:
    """The ONE rubric a storyboard frame is judged against.

    k108's ``JudgeRequest`` verbatim — same type, same repair code travel rule,
    same ``shot_spec_block`` — under this module's own check name so a board
    frame and a finished take are never confused in a scorecard. The question
    is composition, not photorealism: a storyboard that is the right shot is a
    good storyboard even when it looks like a storyboard."""
    block = shot_spec_block(brief, continuity)
    return JudgeRequest(
        name=FRAME_RUBRIC, kind=CheckKind.SEMANTIC,
        prompt=(f"{block}\n\nThis image is a STORYBOARD FRAME for the shot "
                f"above — a low-resolution planning drawing, not finished "
                f"footage. Judge COMPOSITION ONLY: is this the specified "
                f"framing, with the specified subject(s) doing the specified "
                f"action in the specified setting? Do not penalize sketch "
                f"quality, resolution, or missing photographic detail."),
        artifact_ref=artifact_ref, artifact_kind="image",
        segment_id=brief.segment_id,
        repair_code=RepairCode.INTENT_MISMATCH)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_storyboards(shot_plan: Any, *,
                       gen_image: GenImageFn,
                       judge_image: JudgeImageFn | None = None,
                       per_shot: int = DEFAULT_PER_SHOT,
                       world: Any = None,
                       style: str = "color_key",
                       continuity: Any = None,
                       width: int = DEFAULT_WIDTH,
                       height: int = DEFAULT_HEIGHT,
                       threshold: int = DEFAULT_THRESHOLD,
                       generator_model: str | None = None,
                       judge_model: str | None = None,
                       shot_plan_digest: str | None = None,
                       ) -> Storyboard:
    """Render (and judge) ``per_shot`` storyboard candidates for every shot.

    ``gen_image(prompt, identity_refs, seed)`` is the SAME seam shape k106's
    ``PerformanceSeams.gen_image`` declares, on purpose: a caller can hand this
    ``default_seams().gen_image`` unchanged, and :func:`bind_storyboard_image`
    is only that seam with the low-res overrides attached.

    ``judge_image(artifact_ref, brief)`` returns any verdict shape k108's
    ``coerce_judge_result`` accepts. It is OPTIONAL, and its absence is
    reported rather than papered over: with no judge, every frame comes back
    NOT accepted with an ``UNSCORED`` reason, and the operator accepts what
    they can see. Unscored never reads as passed.

    The generator never judges itself: when both models are known and equal,
    k108's :class:`JudgeConflict` is raised BEFORE any dispatch is spent."""
    if per_shot < 1:
        raise StoryboardRefused(
            f"render_storyboards(per_shot=) must be >= 1, got {per_shot}")
    if style not in STORYBOARD_STYLES:
        raise StoryboardRefused(
            f"render_storyboards(style=) must be one of "
            f"{list(STORYBOARD_STYLES)}, got {style!r}")
    if generator_model and judge_model and \
            str(generator_model).casefold() == str(judge_model).casefold():
        raise JudgeConflict(
            f"the storyboard generator and its judge are the same model "
            f"({generator_model!r}) — a self-graded frame is worse than an "
            f"ungraded one because it looks like evidence",
            model=str(generator_model))
    briefs = shot_briefs(shot_plan)
    if not briefs:
        raise StoryboardRefused(
            "cannot render a storyboard for an empty shot plan — there is "
            "nothing to draw")
    states = _continuity_map(continuity)

    frames: list[StoryboardFrame] = []
    for brief in briefs:
        state = states.get(brief.segment_id) or brief.continuity
        prompt = storyboard_prompt(brief, world, style=style,
                                   continuity=state)
        for index in range(per_shot):
            seed = frame_seed(brief.segment_id, index)
            produced = gen_image(prompt, brief.identity_refs, seed)
            ref = _ref_of(produced)
            if not ref:
                logger.warning("storyboard: %s#%d produced no artifact",
                               brief.segment_id, index)
                continue
            frame = _judge_frame(brief, state, prompt, ref, index=index,
                                 seed=seed, judge_image=judge_image,
                                 threshold=threshold, style=style,
                                 width=width, height=height,
                                 generator_model=generator_model,
                                 judge_model=judge_model)
            frames.append(frame)
            if frame.accepted:
                break                     # first hard-pass wins, k106's rule

    return Storyboard(
        frames=tuple(frames), shot_ids=tuple(b.segment_id for b in briefs),
        per_shot=per_shot, style=style,
        shot_plan_digest=(shot_plan_digest
                          if shot_plan_digest is not None
                          else _digest_of(shot_plan)),
        note=(f"{len(frames)} frame(s) over {len(briefs)} shot(s) at "
              f"{width}x{height}"
              + ("" if judge_image is not None
                 else "; NO judge was bound — every frame is UNSCORED and "
                      "waiting on an operator")))


def _judge_frame(brief: ShotBrief, state: ContinuityState | None,
                 prompt: str, ref: str, *, index: int, seed: int,
                 judge_image: JudgeImageFn | None, threshold: int,
                 style: str, width: int, height: int,
                 generator_model: str | None,
                 judge_model: str | None) -> StoryboardFrame:
    """One candidate through the judge, or honestly unscored without one."""
    common: dict[str, Any] = {
        "shot_id": brief.segment_id, "prompt": prompt, "artifact_ref": ref,
        "index": index, "seed": seed, "style": style, "width": width,
        "height": height, "generator_model": generator_model,
        "judge_model": judge_model,
    }
    if judge_image is None:
        return StoryboardFrame(
            **common, accepted=False, verdict="unscored",
            reason=(f"{speech.UNSCORED_PREFIX}no storyboard judge is bound; "
                    f"this frame is waiting on an operator decision"))
    request = frame_rubric(brief, ref, state)
    try:
        raw = judge_image(ref, brief)
    except Exception as exc:                       # noqa: BLE001
        return StoryboardFrame(
            **common, accepted=False, verdict="unavailable",
            reason=(f"{speech.UNSCORED_PREFIX}the storyboard judge raised "
                    f"{type(exc).__name__}: {exc}"))
    result = coerce_judge_result(request, raw, judge_model)
    card = frame_scorecard(request, result, threshold=threshold)
    if result.verdict in ("unavailable", "unscored"):
        return StoryboardFrame(
            **common, accepted=False, verdict=result.verdict,
            score=result.score, scorecard_ref=scorecard_digest(card),
            reason=(f"{speech.UNSCORED_PREFIX}{result.judge}: "
                    f"{result.rationale or 'no verdict'}"))
    if card.hard_pass:
        return StoryboardFrame(
            **common, accepted=True, accepted_by="judge",
            verdict=result.verdict, score=result.score,
            scorecard_ref=scorecard_digest(card),
            reason=(result.rationale
                    or f"{result.judge}: verdict {result.verdict}"))
    return StoryboardFrame(
        **common, accepted=False, verdict=result.verdict, score=result.score,
        scorecard_ref=scorecard_digest(card),
        reason=(f"{result.judge} rejected this frame: "
                f"{result.rationale or 'no rationale given'}"))


def frame_scorecard(request: JudgeRequest, result: Any, *,
                    threshold: int = DEFAULT_THRESHOLD) -> Scorecard:
    """The card one frame earned. ONE check, so a board's evidence is directly
    comparable frame to frame; ``confidence`` drops to 0 when the judge was
    unreachable, which is what keeps "nobody looked" from reading as "it
    passed"."""
    verdict = getattr(result, "verdict", "unscored")
    score = getattr(result, "score", None)
    judge = getattr(result, "judge", request.name)
    rationale = getattr(result, "rationale", "") or ""
    if verdict in ("unavailable", "unscored"):
        check = Check(name=request.name, kind=request.kind, value=None,
                      threshold=threshold, passed=True,
                      detail=(f"{speech.UNSCORED_PREFIX}{judge}: "
                              f"{rationale or 'no verdict'}"))
        return Scorecard(hard_pass=True, checks=(check,), confidence=0.0,
                         diagnosis=None, repair_code=None)
    failing = (score is not None and score < threshold) or \
              (score is None and verdict == "NO")
    check = Check(name=request.name, kind=request.kind, value=score,
                  threshold=threshold, passed=not failing,
                  detail=(f"{judge}: verdict {verdict}, "
                          + (f"score {score:g}" if score is not None
                             else "no score")
                          + f" (threshold {threshold})"
                          + (f" — {rationale}" if rationale else "")))
    return Scorecard(
        hard_pass=not failing, checks=(check,), confidence=1.0,
        diagnosis=(None if not failing else f"{request.name}: {check.detail}"),
        repair_code=(None if not failing else request.repair_code),
        recommended_repair=(None if not failing else
                            "re-render this storyboard frame at a bumped seed, "
                            "or edit the shot's blocking/camera and re-derive "
                            "the prompt — never accept a frame the judge "
                            "turned down"))


def _continuity_map(continuity: Any) -> dict[str, ContinuityState]:
    """``{segment_id: state}`` from a ``ContinuityBible``, a sequence of
    states, or a mapping. Empty for None — a shot with no declared state is
    judged on its specification alone, which is honest."""
    if continuity is None:
        return {}
    entries = getattr(continuity, "entries", None)
    if entries is None:
        entries = continuity.values() if isinstance(continuity, Mapping) \
            else continuity
    out: dict[str, ContinuityState] = {}
    for entry in entries or ():
        if isinstance(entry, ContinuityState):
            out[entry.segment_id] = entry
    return out


def _ref_of(produced: Any) -> str:
    """The artifact reference out of whatever the seam returned — a string, or
    a mapping carrying ``ref``/``uri``/``path``. Same tolerance k106's
    ``_ref_of`` has, so one seam implementation serves both."""
    if produced is None:
        return ""
    if isinstance(produced, str):
        return produced.strip()
    if isinstance(produced, Mapping):
        for key in ("ref", "uri", "path", "artifact_ref"):
            value = produced.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _digest_of(shot_plan: Any) -> str | None:
    """The digest of whatever shot plan this board was drawn against.

    A k110 ``ShotPlanDraft`` is asked for its ``plan`` first, because that is
    the ``ShotPlan`` the production lock actually records — comparing a board
    to a lock then needs no conversion step. A bare sequence of specs has no
    digest of its own and honestly returns None."""
    for candidate in (getattr(shot_plan, "plan", None), shot_plan):
        digest = getattr(candidate, "digest", None)
        if isinstance(digest, str) and digest:
            return digest
    return None


# ---------------------------------------------------------------------------
# The live low-res binding
# ---------------------------------------------------------------------------


def bind_storyboard_image(*, width: int = DEFAULT_WIDTH,
                          height: int = DEFAULT_HEIGHT,
                          capability: str = "image.generate",
                          steps: int | None = None) -> GenImageFn:
    """A live ``(prompt, identity_refs, seed) -> ref`` at STORYBOARD size.

    Everything is imported INSIDE the returned function — building the model
    registry is a two-second import and a contract module must not do it at
    import time (k110's rule, k106's rule). Cheap is expressed as ``width`` /
    ``height`` (and optionally ``steps``) overrides on the existing route, not
    as a second renderer.

    Raises ``RuntimeError`` with the route's own reasons when the capability is
    not executable here — the caller records that as the gap it is."""
    def _render(prompt: str, identity_refs: Sequence[str] = (),
                seed: int = 0) -> str:
        from hugpy_oracle.contracts import ArtifactKind, GoalSpec
        from hugpy_oracle.router import resolve_route
        from hugpy_oracle import runtime

        text = str(prompt)
        if identity_refs:
            text = f"{text}\n[identity: {', '.join(identity_refs)}]"
        goal = GoalSpec(objective="render a storyboard frame",
                        raw_prompt=text[:4000], capability=capability)
        from hugpy_oracle import selection as _selection
        requested, _decision = _selection.requested_model_for(goal, capability)
        try:
            route = resolve_route(goal, requested)
        except Exception:  # noqa: BLE001 — selector/catalog disagreement: catalog wins
            route = resolve_route(goal)
        if route.execution != "execute":
            raise RuntimeError(
                f"{capability}: route is {route.execution!r}, not executable — "
                + ("; ".join(route.reasons) or "no reason recorded"))
        overrides: dict[str, Any] = {"seed": int(seed), "width": int(width),
                                     "height": int(height)}
        if steps is not None:
            overrides["steps"] = int(steps)
        artifacts, receipt = runtime.execute_route(goal, route,
                                                   overrides=overrides)
        if receipt.failure is not None:
            raise RuntimeError(
                f"{capability}: {receipt.failure.value} — "
                + ("; ".join(receipt.log_excerpt) or "no log"))
        for artifact in artifacts or ():
            kind = artifact.get("kind") if isinstance(artifact, Mapping) else None
            if kind == ArtifactKind.IMAGE.value:
                uri = artifact.get("uri") or artifact.get("path")
                if uri:
                    _selection.remember_producer(str(uri), capability, receipt.model_id)
                    return str(uri)
        raise RuntimeError(f"{capability} produced no image artifact "
                           f"({len(artifacts or ())} artifact(s))")
    return _render


class _Thresholds(dict):
    """evaluation.THRESHOLDS, read lazily (import-light module)."""
    def get(self, key, default=None):  # type: ignore[override]
        try:
            from hugpy_oracle.evaluation import THRESHOLDS
            return THRESHOLDS.get(key, default)
        except Exception:  # noqa: BLE001
            return default


THRESHOLDS_FOR_JUDGE = _Thresholds()


def bind_storyboard_judge(*, capability: str = JUDGE_CAPABILITY) -> JudgeImageFn:
    """A live ``(artifact_ref, brief) -> verdict mapping`` over k90c's own
    evaluator — the SAME path ``runners/movie._score_keyframe`` and k106's
    ``_live_judge_image`` use. No second rubric, no second verdict parser."""
    def _judge(artifact_ref: str, brief: Any) -> dict[str, Any]:
        from hugpy_oracle.contracts import ArtifactKind, GoalSpec
        from hugpy_oracle.evaluation import RUBRICS, run_judge

        normalized = brief if isinstance(brief, ShotBrief) else shot_brief(brief)
        request = frame_rubric(normalized, artifact_ref)
        goal = GoalSpec(objective=request.prompt[:400],
                        raw_prompt=request.prompt[:4000], capability=capability)
        from hugpy_oracle import selection as _selection
        prod = _selection.producer_of(artifact_ref)
        result = run_judge(RUBRICS[capability], goal,
                           [{"kind": ArtifactKind.IMAGE.value,
                             "uri": artifact_ref}],
                           generator_model=prod[1] if prod else None)
        if result is None:
            return {"verdict": None, "score": None,
                    "why": "nothing judgeable in the storyboard artifact"}
        if result.verdict not in (None, "unavailable", "unscored"):
            passed = (result.score is not None and result.score >= THRESHOLDS_FOR_JUDGE.get(goal.quality, 70)) \
                if result.score is not None else str(result.verdict).upper().startswith("Y")
            _selection.note_verdict_for_ref(artifact_ref, hard_pass=bool(passed))
        return {"judge": result.judge, "verdict": result.verdict,
                "score": result.score, "why": result.rationale}
    return _judge


__all__ = [
    "ACCEPTED_BY",
    "DEFAULT_HEIGHT",
    "DEFAULT_PER_SHOT",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WIDTH",
    "FRAME_RUBRIC",
    "SEED_SALT",
    "STORYBOARD_STYLES",
    "STYLE_PREAMBLE",
    "Storyboard",
    "StoryboardError",
    "StoryboardFrame",
    "StoryboardRefused",
    "bind_storyboard_image",
    "bind_storyboard_judge",
    "frame_rubric",
    "frame_scorecard",
    "frame_seed",
    "render_storyboards",
    "storyboard_prompt",
]
