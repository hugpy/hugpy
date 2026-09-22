"""Screenplay / PlotSpec / ContinuityBible content / ShotPlan (k110) —
doc Stages 5, 6, 7 and 9, as LOCKED TYPED ARTIFACTS.

k104 shipped the production LOCK and three shells it locks: ``ContinuityState``
/ ``ContinuityBible`` (Stage 7), ``ShotPlanEntry`` / ``ShotPlan`` (Stage 9), and
a ``ProductionLock.screenplay_digest`` field that was always ``None`` because a
screenplay did not exist yet. This module writes what goes in them.

    Stage 5   PlotSpec      premise, genre, tone, characters, causal beats,
                            turning points, ending — validated, not asserted
    Stage 6   Screenplay    ordered Scenes: heading, action, staging, dialogue,
                            entrances/exits, transitions, a/v events, time
    Stage 7   build_continuity()   the before/after log, derived MECHANICALLY
    Stage 9   build_shot_plan()    block / light / rehearse / tweak / shoot

THE ONE IDEA. An LLM is very good at writing a scene and very bad at keeping a
promise about structure. So the LLM never produces an artifact here: it produces
TEXT, the text is parsed, and the parsed object is handed to the SAME frozen
constructor a hand-built artifact goes through. If it validates, you have an
artifact; if it does not, you get the validator's errors back in ONE bounded
reprompt; if it still does not, you get a typed :class:`AuthoringGap` carrying
the errors AND the raw reply. There is no third branch where a field is quietly
coerced to something plausible — a screenplay whose speaker list was "repaired"
by dropping the name that did not match is exactly the artifact that produces a
shot of someone talking to an empty room, forty minutes of GPU time later.

WHAT IS DETERMINISTIC ON PURPOSE. Stage 7 is not an authoring step. Who is in
the room, where the room is, when it is and which props are in play are all
FUNCTIONS of the screenplay; asking a model to restate them is asking it to
disagree with the script. :func:`build_continuity` therefore takes no ``llm``
and never will. The same goes for :func:`build_shot_plan`'s timing when an
``AudioMaster`` is present: Stage 8 makes the audio authoritative, so the
windows are READ off the master, and only a shot with no dialogue behind it
carries ``estimated=True``.

THE CARRIED-STATE CHOICE (read this before the continuity tests). A continuity
state has two kinds of key and they behave differently across a cut:

  * FRAME coordinates — ``scene``, ``location``, ``time_of_day``,
    ``story_time_s``, ``present``, ``props``, ``weather``, ``lighting``,
    ``wardrobe``. These are the scene's own and are RE-DECLARED at every cut.
    A hard cut from a kitchen at night to a car at dawn changes all of them,
    and a chain check over ``location`` would fire on every well-formed film.
  * CARRIED world facts — :data:`CARRIED_KEYS`: ``props_seen`` and
    ``characters_seen``, the cumulative record of what the story has
    ESTABLISHED. Those chain EXACTLY: ``after(scene N)[k] == before(scene
    N+1)[k]``, by construction, and :func:`chain_breaks` returns every
    violation (empty for anything this module builds).

  ``present`` additionally chains across a ``"CONTINUOUS:"`` transition, which
  is what "entrances and exits tracked scene to scene" buys: a continuous scene
  inherits the previous scene's closing cast instead of re-declaring it, and
  its speakers are checked against the INHERITED cast.

REUSE, NOT A SECOND DEFINITION. A screenplay's dialogue line IS k102's
``audio_master.Line`` — same class, not a parallel one. That makes "every Line
maps 1:1 onto a k102 Line id" structural rather than a convention somebody has
to maintain, and :meth:`Screenplay.to_dialogue_timeline` is then a collection,
not a translation. Camera vocabulary is k104's closed ``CAMERA_KEYS`` /
``SHOT_SIZES`` / ``CAMERA_MOVES`` (which already mirror the studio's), the view
hint comes from ``video_intel.shot_intent`` through k104's lazy wrapper, and
the dramatic beat comes from ``video_intel.prompt_seeds.beat_for_index`` — both
imported LAZILY, at the call site, exactly as k104's dispatch record asked, so
that importing a contract never builds the model registry.

NO CLOCK, NO REGISTRY, NO DISK, NO NETWORK. Every artifact here is a frozen
slotted dataclass with canonical-JSON identity byte-identical to
``production``/``audio_master``/``mct.manifest``. The only outward call is the
injected ``llm: (prompt) -> str``; :func:`bind_llm` produces one from the
existing oracle route + dispatch and degrades to a typed ``AuthoringGap`` when
no text model is eligible on this fleet.

No pathlib anywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from hugpy_oracle.audio_master import AudioMaster, DialogueTimeline, Line
from hugpy_oracle.plan import FrozenParams
from hugpy_oracle.production import (
    CAMERA_KEYS,
    CAMERA_MOVES,
    CAMERA_VIEWS,
    SHOT_SIZES,
    ContentAddressed,
    ContinuityBible,
    ContinuityState,
    GenerationSnapshot,
    LockRefused,
    ProductionError,
    ProductionLock,
    ShotPlan,
    ShotPlanEntry,
    camera_view_from_prompt,
)
# BORROWED, not re-copied. ``production`` already owns the exact versions of
# these four helpers that k104's artifacts validate with; a fourth private copy
# in a fifth module is how a tree ends up with five subtly different
# "non-empty string" rules. A test asserts these are the same objects.
from hugpy_oracle.production import _EPS as EPS
from hugpy_oracle.production import _q as quantize
from hugpy_oracle.production import _require_text as require_text
from hugpy_oracle.production import _str_tuple as str_tuple
from hugpy_oracle.schema_export import json_schema_for


# ---------------------------------------------------------------------------
# Errors + the typed authoring gap
# ---------------------------------------------------------------------------


class ScreenplayError(ProductionError):
    """Base for every refusal in the k110 family.

    Subclasses ``ProductionError`` (itself a ``ValueError``) so a caller that
    already catches the production-lock family catches these too — a screenplay
    that will not build and a lock that will not close are the same class of
    event to whoever is holding the request."""

    def __init__(self, message: str, errors: Sequence[str] = ()) -> None:
        super().__init__(message)
        #: EVERY problem found, not just the first. A constructor that reports
        #: one error per attempt turns a bounded reprompt into a slow ladder;
        #: reporting all of them means one repair round can fix all of them.
        self.errors: tuple[str, ...] = tuple(str(e) for e in errors) or (message,)


class PlotRefused(ScreenplayError):
    """A :class:`PlotSpec` does not describe a causally connected story."""


class ScreenplayRefused(ScreenplayError):
    """A :class:`Screenplay` contradicts itself (presence, time, or ids)."""


class ShotPlanRefused(ScreenplayError):
    """A shot plan cannot be built from this screenplay + audio master."""


class LlmUnavailable(ScreenplayError):
    """The bound live model could not answer. Raised by the callable
    :func:`bind_llm` returns, and converted to an ``AuthoringGap`` by the
    ``author_*`` functions — an authoring call never raises for a model
    problem, because "the fleet was busy" is an answer the caller has to be
    able to read, not an exception it has to guess about."""


#: Why an ``AuthoringGap`` exists. Each is a genuinely different operator
#: action: fix the model output, seat a text model, or look at the fleet.
GAP_CODES: tuple[str, ...] = (
    "AUTHORING_INVALID",   # two attempts, still not a valid artifact
    "AUTHORING_UNPARSED",  # two attempts, no JSON object in the reply at all
    "CAPABILITY_GAP",      # no eligible text model on this fleet
    "LLM_ERROR",           # the callable raised / the dispatch failed
)


@dataclass(frozen=True, slots=True)
class AuthoringGap(ContentAddressed):
    """The honest outcome when LLM-assisted authoring did not produce an
    artifact. NEVER a half-built one.

    ``errors`` is what the validators said, verbatim — the same strings that
    were embedded in the reprompt, so an operator reading a gap sees exactly
    what the model was told. ``raw`` is the LAST reply, preserved untouched:
    the text is usually 90% of a usable screenplay and throwing it away to
    return a tidy error is the expensive kind of tidiness. ``raw_attempts``
    keeps both tries for a benchmark (k109) that wants to score the repair
    round separately from the first."""

    errors: tuple[str, ...] = ()
    raw: str = ""
    stage: str = "plot"
    code: str = "AUTHORING_INVALID"
    attempts: int = 0
    raw_attempts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors",
                           tuple(str(e) for e in self.errors))
        if not self.errors:
            raise ValueError("AuthoringGap needs at least one error — a gap "
                             "with no reason is the silent failure this type "
                             "exists to prevent")
        object.__setattr__(self, "raw", str(self.raw or ""))
        object.__setattr__(self, "stage", require_text(self.stage,
                                                       "AuthoringGap.stage"))
        if self.code not in GAP_CODES:
            raise ValueError(f"AuthoringGap.code {self.code!r} is not one of "
                             f"{list(GAP_CODES)}")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) \
                or self.attempts < 0:
            raise ValueError(f"AuthoringGap.attempts must be a non-negative "
                             f"int, got {self.attempts!r}")
        object.__setattr__(self, "raw_attempts",
                           tuple(str(r) for r in self.raw_attempts))

    def to_dict(self) -> dict[str, Any]:
        return {"errors": list(self.errors), "raw": self.raw,
                "stage": self.stage, "code": self.code,
                "attempts": self.attempts,
                "raw_attempts": list(self.raw_attempts)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AuthoringGap":
        return cls(errors=tuple(d.get("errors", ())), raw=d.get("raw", ""),
                   stage=d.get("stage", "plot"),
                   code=d.get("code", "AUTHORING_INVALID"),
                   attempts=int(d.get("attempts", 0)),
                   raw_attempts=tuple(d.get("raw_attempts", ())))


# ---------------------------------------------------------------------------
# Stage 5 — plot construction
# ---------------------------------------------------------------------------

#: Doc Stage 5's three supported input shapes, as a closed vocabulary. The mode
#: chooses ONE guidance paragraph in the authoring prompt and is recorded on the
#: artifact as provenance. It changes NOTHING about validation: all three route
#: through the same constructor, which is the whole point of Stage 5 — "little
#: or no narrative input" must not produce a laxer artifact than a full outline.
INPUT_MODES: tuple[str, ...] = ("complete", "partial", "minimal")

#: Under this many words, the request carries "little or no narrative input".
MINIMAL_WORDS: int = 25


@dataclass(frozen=True, slots=True)
class Character(ContentAddressed):
    """One character, with the three things Stage 5 names: a goal, a conflict
    and an arc.

    All three are required text. A character with no goal is a prop with a
    name, and the beats that "feature" them will read as furniture — this is
    the cheapest place in the whole pipeline to catch that."""

    name: str
    goal: str
    conflict: str
    arc: str
    description: str = ""

    def __post_init__(self) -> None:
        for name in ("name", "goal", "conflict", "arc"):
            require_text(getattr(self, name), f"Character.{name}")
        object.__setattr__(self, "description", str(self.description or ""))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "goal": self.goal,
                "conflict": self.conflict, "arc": self.arc,
                "description": self.description}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Character":
        return cls(name=d["name"], goal=d.get("goal", ""),
                   conflict=d.get("conflict", ""), arc=d.get("arc", ""),
                   description=d.get("description", ""))


@dataclass(frozen=True, slots=True)
class Beat(ContentAddressed):
    """One story beat and what CAUSED it.

    ``causes`` names the beats this one follows FROM — the doc's "causal
    progression", stored on the effect rather than the cause because that is
    the direction a reader asks the question in ("why does this happen?").
    A cause must appear EARLIER in ``PlotSpec.beats``: that tuple is the causal
    order, and screen order (which may flash back) is the Screenplay's problem,
    not the plot's.

    ``turning_point`` is a flag on the beat rather than a separate list, so a
    beat cannot be a turning point in one field and absent from the other."""

    beat_id: str
    summary: str
    characters: tuple[str, ...] = ()
    causes: tuple[str, ...] = ()
    turning_point: bool = False
    location: str | None = None
    time_of_day: str | None = None

    def __post_init__(self) -> None:
        require_text(self.beat_id, "Beat.beat_id")
        require_text(self.summary, "Beat.summary")
        object.__setattr__(self, "characters",
                           str_tuple(self.characters, "Beat.characters"))
        object.__setattr__(self, "causes",
                           str_tuple(self.causes, "Beat.causes"))
        if not isinstance(self.turning_point, bool):
            raise TypeError(f"Beat({self.beat_id}).turning_point must be a "
                            f"bool, got {type(self.turning_point).__name__}")
        for name in ("location", "time_of_day"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name,
                                   require_text(value, f"Beat.{name}"))

    def to_dict(self) -> dict[str, Any]:
        return {"beat_id": self.beat_id, "summary": self.summary,
                "characters": list(self.characters),
                "causes": list(self.causes),
                "turning_point": self.turning_point,
                "location": self.location, "time_of_day": self.time_of_day}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Beat":
        return cls(beat_id=d["beat_id"], summary=d.get("summary", ""),
                   characters=tuple(d.get("characters", ())),
                   causes=tuple(d.get("causes", ())),
                   turning_point=bool(d.get("turning_point", False)),
                   location=d.get("location"),
                   time_of_day=d.get("time_of_day"))


@dataclass(frozen=True, slots=True)
class PlotSpec(ContentAddressed):
    """Doc Stage 5 — the constructed plot, validated as a connected story.

    THE THREE REFUSALS, and why each one is a real failure of this pipeline:

    1. **A beat that names no character.** "The storm arrives" is a weather
       report. Every segment prompt this pipeline compiles is about somebody
       doing something; a beat with nobody in it produces a shot with nobody in
       it, and identity conditioning has nothing to hold onto.
    2. **A causal link to a beat that does not exist** (or to itself, or
       forward). The model that invents ``"b7"`` in a five-beat plot has lost
       the thread, and a dangling cause is not detectable any later: by Stage 9
       it is just a shot list with a hole in the middle.
    3. **An orphan character** — declared in ``characters`` and named by no
       beat. Either they belong in the story and a beat is missing, or they do
       not and they are about to be cast, voiced and identity-locked for
       nothing. Both are worth one exception here.

    Every problem found is reported, not just the first: ``PlotRefused.errors``
    carries them all, which is what makes ONE bounded reprompt enough."""

    premise: str
    genre: str
    tone: str
    characters: tuple[Character, ...] = ()
    beats: tuple[Beat, ...] = ()
    ending: str = ""
    pacing: str = ""
    input_mode: str = "complete"
    notes: str = ""

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in ("premise", "genre", "tone"):
            try:
                require_text(getattr(self, name), f"PlotSpec.{name}")
            except ValueError as exc:
                problems.append(str(exc))
        try:
            require_text(self.ending, "PlotSpec.ending")
        except ValueError:
            problems.append("PlotSpec.ending must be non-empty — Stage 5 asks "
                            "for the ending, and a plot that stops is not one")
        object.__setattr__(self, "characters", tuple(self.characters))
        object.__setattr__(self, "beats", tuple(self.beats))
        for item in self.characters:
            if not isinstance(item, Character):
                raise TypeError(f"PlotSpec.characters takes Character, got "
                                f"{type(item).__name__}")
        for item in self.beats:
            if not isinstance(item, Beat):
                raise TypeError(f"PlotSpec.beats takes Beat, got "
                                f"{type(item).__name__}")
        if not self.characters:
            problems.append("PlotSpec has no characters")
        if not self.beats:
            problems.append("PlotSpec has no beats")
        if self.input_mode not in INPUT_MODES:
            problems.append(f"PlotSpec.input_mode {self.input_mode!r} is not "
                            f"one of {list(INPUT_MODES)}")
        object.__setattr__(self, "pacing", str(self.pacing or ""))
        object.__setattr__(self, "notes", str(self.notes or ""))

        names = [c.name for c in self.characters]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            problems.append(f"PlotSpec declares character(s) {dupes} twice")
        beat_ids = [b.beat_id for b in self.beats]
        dupe_beats = sorted({b for b in beat_ids if beat_ids.count(b) > 1})
        if dupe_beats:
            problems.append(f"PlotSpec declares beat(s) {dupe_beats} twice")

        known_names = set(names)
        seen_beats: set[str] = set()
        used_names: set[str] = set()
        for beat in self.beats:
            # [1] every beat names at least one character
            if not beat.characters:
                problems.append(
                    f"beat {beat.beat_id!r} names no character — a beat with "
                    f"nobody in it compiles to a shot with nobody in it")
            unknown = [n for n in beat.characters if n not in known_names]
            if unknown:
                problems.append(
                    f"beat {beat.beat_id!r} names character(s) {unknown} that "
                    f"the plot does not declare (has: {sorted(known_names)})")
            used_names.update(n for n in beat.characters if n in known_names)
            # [2] causal links reference existing, EARLIER beats
            for cause in beat.causes:
                if cause == beat.beat_id:
                    problems.append(f"beat {beat.beat_id!r} causes itself")
                elif cause not in beat_ids:
                    problems.append(
                        f"beat {beat.beat_id!r} is caused by {cause!r}, which "
                        f"is not a beat in this plot (has: {beat_ids})")
                elif cause not in seen_beats:
                    problems.append(
                        f"beat {beat.beat_id!r} is caused by {cause!r}, which "
                        f"comes AFTER it — PlotSpec.beats is the causal order "
                        f"(screen order, including flashbacks, is the "
                        f"screenplay's)")
            seen_beats.add(beat.beat_id)
        # [3] no orphan characters
        orphans = sorted(known_names - used_names)
        if orphans:
            problems.append(
                f"character(s) {orphans} appear in no beat — either a beat is "
                f"missing or the character is, and both are cheaper to fix "
                f"here than after they have been cast and voiced")

        if problems:
            raise PlotRefused("; ".join(problems), errors=problems)

    # -- reading -----------------------------------------------------------

    @property
    def character_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.characters)

    @property
    def beat_ids(self) -> tuple[str, ...]:
        return tuple(b.beat_id for b in self.beats)

    @property
    def turning_points(self) -> tuple[str, ...]:
        """The ids of the beats flagged as turning points, in beat order."""
        return tuple(b.beat_id for b in self.beats if b.turning_point)

    @property
    def roots(self) -> tuple[str, ...]:
        """Beats nothing causes — where the story starts. More than one is
        legal (parallel threads); zero in a multi-beat plot is impossible,
        because a cause must precede its effect."""
        return tuple(b.beat_id for b in self.beats if not b.causes)

    def character(self, name: str) -> Character:
        for item in self.characters:
            if item.name == name:
                return item
        raise KeyError(f"no character named {name!r}")

    def beat(self, beat_id: str) -> Beat:
        for item in self.beats:
            if item.beat_id == beat_id:
                return item
        raise KeyError(f"no beat {beat_id!r}")

    def beats_for(self, name: str) -> tuple[str, ...]:
        return tuple(b.beat_id for b in self.beats if name in b.characters)

    def to_dict(self) -> dict[str, Any]:
        return {"premise": self.premise, "genre": self.genre,
                "tone": self.tone,
                "characters": [c.to_dict() for c in self.characters],
                "beats": [b.to_dict() for b in self.beats],
                "ending": self.ending, "pacing": self.pacing,
                "input_mode": self.input_mode, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PlotSpec":
        return cls(premise=d.get("premise", ""), genre=d.get("genre", ""),
                   tone=d.get("tone", ""),
                   characters=tuple(Character.from_dict(c)
                                    for c in d.get("characters", ())),
                   beats=tuple(Beat.from_dict(b) for b in d.get("beats", ())),
                   ending=d.get("ending", ""), pacing=d.get("pacing", ""),
                   input_mode=d.get("input_mode", "complete"),
                   notes=d.get("notes", ""))


# ---------------------------------------------------------------------------
# Stage 6 — the screenplay
# ---------------------------------------------------------------------------

#: Slugline prefixes. A scene heading that does not start with one of these is
#: not a scene heading, and the breakdown reads ``location`` off it.
SCENE_PREFIXES: tuple[str, ...] = ("INT.", "EXT.", "INT./EXT.", "EXT./INT.",
                                   "I/E.")

#: The transition OUT of a scene, into the next one — closed, because two of
#: these carry rules (see below) and a typo'd ``"FLASBACK TO:"`` that silently
#: became free text would disable one of them.
TRANSITIONS: frozenset[str] = frozenset({
    "CUT TO:", "SMASH CUT TO:", "MATCH CUT TO:", "TIME CUT TO:",
    "DISSOLVE TO:", "FADE TO BLACK.", "FADE OUT.", "INTERCUT WITH:",
    "CONTINUOUS:", "FLASHBACK TO:", "BACK TO PRESENT:", "END OF FLASHBACK.",
})

#: The only transitions that may move story time BACKWARDS. Everything else is
#: held to a monotonic clock.
FLASHBACK_TRANSITIONS: frozenset[str] = frozenset({
    "FLASHBACK TO:", "BACK TO PRESENT:", "END OF FLASHBACK.",
})

#: The transition that says "no time passes and nobody moves between these two
#: scenes" — the next scene INHERITS this one's closing cast.
CONTINUOUS_TRANSITION: str = "CONTINUOUS:"

#: Doc Stage 6's "required audio/visual events", as a closed kind vocabulary.
AV_EVENT_KINDS: frozenset[str] = frozenset({
    "sfx", "music", "ambience", "voiceover", "title_card", "silence",
    "foley", "on_screen_text",
})


@dataclass(frozen=True, slots=True)
class AVEvent(ContentAddressed):
    """One required audio or visual event (Stage 6's last item).

    ``cue`` anchors it in the scene in the script's own words ("on JAMIE's
    entrance", "under the last line") rather than in seconds: the seconds do
    not exist until Stage 8 has locked the audio, and a cue written in seconds
    before then is a guess wearing a number."""

    kind: str
    description: str
    cue: str = ""

    def __post_init__(self) -> None:
        require_text(self.kind, "AVEvent.kind")
        if self.kind not in AV_EVENT_KINDS:
            raise ValueError(f"AVEvent.kind {self.kind!r} is not in the "
                             f"vocabulary {sorted(AV_EVENT_KINDS)}")
        require_text(self.description, "AVEvent.description")
        object.__setattr__(self, "cue", str(self.cue or ""))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "description": self.description,
                "cue": self.cue}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AVEvent":
        return cls(kind=d.get("kind", ""), description=d.get("description", ""),
                   cue=d.get("cue", ""))


def make_heading(interior: str, location: str, time_of_day: str) -> str:
    """A slugline from its parts: ``make_heading("INT.", "KITCHEN", "NIGHT")``
    -> ``"INT. KITCHEN - NIGHT"``. Deterministic, uppercased, so two callers
    cannot produce two headings that differ only in whitespace and fork a
    digest over nothing."""
    prefix = str(interior or "").strip().upper()
    if prefix not in SCENE_PREFIXES:
        raise ValueError(f"make_heading(interior={interior!r}) must be one of "
                         f"{list(SCENE_PREFIXES)}")
    place = require_text(location, "make_heading(location)").strip().upper()
    when = require_text(time_of_day, "make_heading(time_of_day)").strip().upper()
    return f"{prefix} {place} - {when}"


@dataclass(frozen=True, slots=True)
class Scene(ContentAddressed):
    """One scene: doc Stage 6's unit.

    ``present_at_open`` is who is already in the room when the scene starts.
    It may be EMPTY, and empty means two different things depending on the
    previous scene's transition: after ``CONTINUOUS:`` it means "inherit", and
    otherwise it means "nobody is here yet, they walk in". That distinction is
    resolved by :class:`Screenplay`, which is the only object that can see the
    previous scene — a lone ``Scene`` does not know what it follows.

    ``story_time_s`` is the NARRATIVE clock in seconds (the world's time, not
    the film's runtime). It is what the monotonic check reads, because
    ``time_of_day`` is a label a screenplay is allowed to write as "MOMENTS
    LATER" and no ordering can be derived from that honestly.

    ``transition`` is the transition OUT of this scene, into the next — where a
    screenplay actually writes it. Two of them carry rules: a
    ``FLASHBACK_TRANSITIONS`` member licenses the NEXT scene to move story time
    backwards, and ``CONTINUOUS:`` makes the next scene inherit this one's
    closing cast."""

    scene_id: str
    heading: str
    location: str
    time_of_day: str
    action: str = ""
    staging: str = ""
    present_at_open: tuple[str, ...] = ()
    entrances: tuple[str, ...] = ()
    exits: tuple[str, ...] = ()
    dialogue: tuple[Line, ...] = ()
    av_events: tuple[AVEvent, ...] = ()
    props: tuple[str, ...] = ()
    wardrobe: tuple[str, ...] = ()
    weather: str = ""
    lighting: str = ""
    transition: str = "CUT TO:"
    story_time_s: float = 0.0
    beat_id: str | None = None

    def __post_init__(self) -> None:
        require_text(self.scene_id, "Scene.scene_id")
        heading = require_text(self.heading, "Scene.heading").strip()
        if not any(heading.upper().startswith(p) for p in SCENE_PREFIXES):
            raise ValueError(
                f"Scene({self.scene_id}).heading {heading!r} is not a slugline: "
                f"it must start with one of {list(SCENE_PREFIXES)} — the "
                f"breakdown reads interior/exterior off the heading")
        object.__setattr__(self, "heading", heading)
        require_text(self.location, "Scene.location")
        require_text(self.time_of_day, "Scene.time_of_day")
        for name in ("action", "staging", "weather", "lighting"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        for name in ("present_at_open", "entrances", "exits", "props",
                     "wardrobe"):
            object.__setattr__(self, name,
                               str_tuple(getattr(self, name), f"Scene.{name}"))
        object.__setattr__(self, "dialogue", tuple(self.dialogue))
        for line in self.dialogue:
            if not isinstance(line, Line):
                raise TypeError(f"Scene.dialogue takes audio_master.Line, got "
                                f"{type(line).__name__}")
        ids = [ln.line_id for ln in self.dialogue]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"Scene({self.scene_id}) repeats line id(s) "
                             f"{dupes} — a line id is the handle every later "
                             f"artifact refers to and must be unique")
        object.__setattr__(self, "av_events", tuple(self.av_events))
        for event in self.av_events:
            if not isinstance(event, AVEvent):
                raise TypeError(f"Scene.av_events takes AVEvent, got "
                                f"{type(event).__name__}")
        if self.transition not in TRANSITIONS:
            raise ValueError(
                f"Scene({self.scene_id}).transition {self.transition!r} is not "
                f"in the vocabulary {sorted(TRANSITIONS)}")
        object.__setattr__(self, "story_time_s", quantize(self.story_time_s))
        if self.story_time_s < 0:
            raise ValueError(f"Scene({self.scene_id}).story_time_s must be "
                             f"non-negative, got {self.story_time_s}")
        if self.beat_id is not None:
            object.__setattr__(self, "beat_id",
                               require_text(self.beat_id, "Scene.beat_id"))

        already = [n for n in self.entrances if n in self.present_at_open]
        if already:
            raise ValueError(
                f"Scene({self.scene_id}): {already} are listed both as present "
                f"at the open and as entering — a character cannot enter a "
                f"room they are already in")
        # Scene-local presence checks. Only meaningful when the scene DECLARES
        # its own opening cast: a scene with an empty ``present_at_open`` may
        # be inheriting one from a ``CONTINUOUS:`` predecessor, and only the
        # Screenplay can see that. Screenplay re-runs both checks against the
        # RESOLVED cast for every scene, so nothing is skipped — it is only
        # deferred to the object that has the information.
        if self.present_at_open:
            self.assert_presence(self.present_at_open,
                                 where=f"Scene({self.scene_id})")

    # -- presence ----------------------------------------------------------

    @property
    def speakers(self) -> tuple[str, ...]:
        """Distinct speakers in first-line order."""
        out: list[str] = []
        for line in self.dialogue:
            if line.speaker not in out:
                out.append(line.speaker)
        return tuple(out)

    @property
    def line_ids(self) -> tuple[str, ...]:
        return tuple(ln.line_id for ln in self.dialogue)

    @property
    def names(self) -> tuple[str, ...]:
        """Every character this scene mentions in a structural field — the set
        the screenplay's cast must contain."""
        return str_tuple(list(self.present_at_open) + list(self.entrances)
                         + list(self.exits) + list(self.speakers),
                         "Scene.names")

    def cast_after_open(self, opened: Sequence[str]) -> tuple[str, ...]:
        """Everyone who appears at ANY point, given the resolved opening cast:
        the open plus the entrances. This is what a speaker must belong to —
        somebody who walks in halfway through may of course speak."""
        return str_tuple(list(opened) + list(self.entrances),
                         f"Scene({self.scene_id}).cast")

    def cast_at_close(self, opened: Sequence[str]) -> tuple[str, ...]:
        """Who is still in the room when the scene ends."""
        leaving = set(self.exits)
        return tuple(n for n in self.cast_after_open(opened)
                     if n not in leaving)

    def assert_presence(self, opened: Sequence[str], *,
                        where: str = "") -> None:
        """Refuse a line whose speaker is not in the room, and an exit by
        somebody who never was — both against the RESOLVED opening cast.

        The speaker check is the one that stops the pipeline from spending a
        GPU hour on a shot of somebody talking to an empty room: the speaker
        list and the presence list are written by different parts of a model's
        answer, and disagreeing is the single most common way an LLM
        screenplay is wrong. Every problem is reported at once, so one repair
        round can fix them all."""
        label = where or f"Scene({self.scene_id})"
        cast = set(self.cast_after_open(opened))
        problems: list[str] = []
        ghosts = [n for n in self.exits if n not in cast]
        if ghosts:
            problems.append(
                f"{label}: {ghosts} exit without ever being present "
                f"(present: {sorted(cast)})")
        missing = [ln.line_id for ln in self.dialogue if ln.speaker not in cast]
        if missing:
            names = sorted({ln.speaker for ln in self.dialogue
                            if ln.speaker not in cast})
            problems.append(
                f"{label}: {names} speak line(s) {missing} but are not in the "
                f"scene (present: {sorted(cast)}) — a speaker must be in the "
                f"room, or enter it")
        if problems:
            raise ScreenplayRefused("; ".join(problems), errors=problems)

    # -- wire shape --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id, "heading": self.heading,
            "location": self.location, "time_of_day": self.time_of_day,
            "action": self.action, "staging": self.staging,
            "present_at_open": list(self.present_at_open),
            "entrances": list(self.entrances), "exits": list(self.exits),
            "dialogue": [ln.to_dict() for ln in self.dialogue],
            "av_events": [e.to_dict() for e in self.av_events],
            "props": list(self.props), "wardrobe": list(self.wardrobe),
            "weather": self.weather, "lighting": self.lighting,
            "transition": self.transition, "story_time_s": self.story_time_s,
            "beat_id": self.beat_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Scene":
        return cls(
            scene_id=d["scene_id"], heading=d.get("heading", ""),
            location=d.get("location", ""),
            time_of_day=d.get("time_of_day", ""),
            action=d.get("action", ""), staging=d.get("staging", ""),
            present_at_open=tuple(d.get("present_at_open", ())),
            entrances=tuple(d.get("entrances", ())),
            exits=tuple(d.get("exits", ())),
            dialogue=tuple(Line.from_dict(x) for x in d.get("dialogue", ())),
            av_events=tuple(AVEvent.from_dict(x)
                            for x in d.get("av_events", ())),
            props=tuple(d.get("props", ())),
            wardrobe=tuple(d.get("wardrobe", ())),
            weather=d.get("weather", ""), lighting=d.get("lighting", ""),
            transition=d.get("transition", "CUT TO:"),
            story_time_s=d.get("story_time_s", 0.0),
            beat_id=d.get("beat_id"),
        )


@dataclass(frozen=True, slots=True)
class Screenplay(ContentAddressed):
    """Doc Stage 6 — ONE ordered screenplay covering the entire requested video.

    THE FOUR SEQUENCE REFUSALS (a lone ``Scene`` cannot make any of them,
    because each one is about what a scene FOLLOWS):

    1. **A speaker who is not in the room**, checked against the RESOLVED
       opening cast — the declared one, or the previous scene's closing cast
       when the transition was ``CONTINUOUS:``.
    2. **Story time running backwards** without a flashback transition to
       declare it. Time going backwards by accident is how a continuity bible
       ends up asserting that a character is both dead and driving.
    3. **A duplicate line id** anywhere in the screenplay. Line ids are the
       1:1 join onto k102's ``DialogueTimeline``, k104's shot windows, and
       every repair target; two lines sharing one id silently overwrite each
       other three artifacts downstream.
    4. **A character the cast does not declare.** ``characters`` may be left
       empty and is then DERIVED (first appearance order); when it is supplied
       it is authoritative and a name outside it is refused.

    ``locked`` is the Stage 11 gate, the same shape ``DialogueTimeline.locked``
    and ``AudioMaster.locked`` already use: :func:`lock_production` refuses an
    unlocked screenplay, because locking a production against a script that may
    still change is the Stage 8 mistake one stage earlier."""

    title: str
    scenes: tuple[Scene, ...]
    logline: str = ""
    characters: tuple[str, ...] = ()
    plot_digest: str | None = None
    locked: bool = False

    def __post_init__(self) -> None:
        require_text(self.title, "Screenplay.title")
        object.__setattr__(self, "scenes", tuple(self.scenes))
        for scene in self.scenes:
            if not isinstance(scene, Scene):
                raise TypeError(f"Screenplay.scenes takes Scene, got "
                                f"{type(scene).__name__}")
        object.__setattr__(self, "logline", str(self.logline or ""))
        if self.plot_digest is not None:
            object.__setattr__(self, "plot_digest",
                               require_text(self.plot_digest,
                                            "Screenplay.plot_digest"))
        if not isinstance(self.locked, bool):
            raise TypeError(f"Screenplay.locked must be a bool, got "
                            f"{type(self.locked).__name__}")

        problems: list[str] = []
        if not self.scenes:
            raise ScreenplayRefused(
                "Screenplay has no scenes — Stage 6 asks for one ordered "
                "screenplay covering the ENTIRE requested video",
                errors=("Screenplay has no scenes",))

        ids = [s.scene_id for s in self.scenes]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            problems.append(f"Screenplay repeats scene id(s) {dupes}")

        # [3] line ids are unique across the WHOLE screenplay
        line_ids: list[str] = []
        for scene in self.scenes:
            line_ids.extend(scene.line_ids)
        line_dupes = sorted({i for i in line_ids if line_ids.count(i) > 1})
        if line_dupes:
            problems.append(
                f"line id(s) {line_dupes} appear in more than one scene — a "
                f"line id is the 1:1 join onto the dialogue timeline, the "
                f"audio master and every shot window")

        # [4] the cast
        appearing: list[str] = []
        for scene in self.scenes:
            for name in scene.names:
                if name not in appearing:
                    appearing.append(name)
        declared = str_tuple(self.characters, "Screenplay.characters")
        if declared:
            unknown = [n for n in appearing if n not in declared]
            if unknown:
                problems.append(
                    f"character(s) {unknown} appear in the scenes but not in "
                    f"Screenplay.characters (has: {list(declared)})")
            object.__setattr__(self, "characters", declared)
        else:
            object.__setattr__(self, "characters", tuple(appearing))

        # [1] + [2] the sequence walk
        opened: tuple[str, ...] = ()
        previous: Scene | None = None
        for index, scene in enumerate(self.scenes):
            if previous is not None and \
                    previous.transition == CONTINUOUS_TRANSITION:
                carried = previous.cast_at_close(opened)
                if scene.present_at_open and \
                        set(scene.present_at_open) != set(carried):
                    problems.append(
                        f"scene {scene.scene_id!r} follows {previous.scene_id!r} "
                        f"on {CONTINUOUS_TRANSITION} but opens with "
                        f"{list(scene.present_at_open)} instead of the cast "
                        f"that scene closed on ({list(carried)}) — nothing may "
                        f"change across a continuous join")
                resolved = scene.present_at_open or carried
            else:
                resolved = scene.present_at_open
            try:
                scene.assert_presence(resolved, where=f"scene {scene.scene_id!r}")
            except ScreenplayRefused as exc:
                problems.extend(exc.errors)

            if previous is not None:
                backwards = scene.story_time_s + EPS < previous.story_time_s
                if backwards and previous.transition not in FLASHBACK_TRANSITIONS:
                    problems.append(
                        f"scene {scene.scene_id!r} is at story time "
                        f"{scene.story_time_s}s, BEFORE {previous.scene_id!r} "
                        f"at {previous.story_time_s}s, and "
                        f"{previous.scene_id!r} ends on "
                        f"{previous.transition!r} — only "
                        f"{sorted(FLASHBACK_TRANSITIONS)} may move story time "
                        f"backwards")
            opened = scene.cast_at_close(resolved)
            previous = scene

        if problems:
            raise ScreenplayRefused("; ".join(problems), errors=problems)

    # -- presence, resolved -------------------------------------------------

    def presence_chain(self) -> tuple[tuple[str, tuple[str, ...],
                                            tuple[str, ...]], ...]:
        """``((scene_id, cast_at_open, cast_at_close), …)`` — the entrance/exit
        tracking, resolved scene to scene. This is the one place presence is
        computed; :meth:`open_at`, :meth:`close_at` and
        :func:`build_continuity` all read it, so they cannot drift apart."""
        out: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        previous: Scene | None = None
        closed: tuple[str, ...] = ()
        for scene in self.scenes:
            if previous is not None and \
                    previous.transition == CONTINUOUS_TRANSITION:
                opened = scene.present_at_open or closed
            else:
                opened = scene.present_at_open
            closed = scene.cast_at_close(opened)
            out.append((scene.scene_id, tuple(opened), closed))
            previous = scene
        return tuple(out)

    def open_at(self, index: int) -> tuple[str, ...]:
        return self.presence_chain()[index][1]

    def close_at(self, index: int) -> tuple[str, ...]:
        return self.presence_chain()[index][2]

    # -- reading -----------------------------------------------------------

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(s.scene_id for s in self.scenes)

    @property
    def lines(self) -> tuple[Line, ...]:
        """Every dialogue line, in screenplay order. THE source of the
        dialogue timeline — same objects, not copies."""
        out: list[Line] = []
        for scene in self.scenes:
            out.extend(scene.dialogue)
        return tuple(out)

    @property
    def line_ids(self) -> tuple[str, ...]:
        return tuple(ln.line_id for ln in self.lines)

    @property
    def speakers(self) -> tuple[str, ...]:
        out: list[str] = []
        for line in self.lines:
            if line.speaker not in out:
                out.append(line.speaker)
        return tuple(out)

    @property
    def locations(self) -> tuple[str, ...]:
        return str_tuple([s.location for s in self.scenes],
                         "Screenplay.locations")

    @property
    def prop_inventory(self) -> tuple[str, ...]:
        """Every prop any scene declares, first-appearance order. This is the
        vocabulary :func:`build_continuity` scans the action text against —
        props are recognized, never invented from prose."""
        out: list[str] = []
        for scene in self.scenes:
            for prop in scene.props:
                if prop not in out:
                    out.append(prop)
        return tuple(out)

    @property
    def wardrobe(self) -> tuple[str, ...]:
        out: list[str] = []
        for scene in self.scenes:
            for note in scene.wardrobe:
                if note not in out:
                    out.append(note)
        return tuple(out)

    def scene(self, scene_id: str) -> Scene:
        for item in self.scenes:
            if item.scene_id == scene_id:
                return item
        raise KeyError(f"no scene {scene_id!r}")

    def scene_of_line(self, line_id: str) -> str:
        for item in self.scenes:
            if line_id in item.line_ids:
                return item.scene_id
        raise KeyError(f"no scene carries line {line_id!r}")

    def line_index(self) -> dict[str, tuple[str, int]]:
        """``line_id -> (scene_id, position in the screenplay)``. The 1:1 map
        onto k102's ``DialogueTimeline``, made explicit so a caller does not
        have to re-derive it (and get it subtly wrong)."""
        out: dict[str, tuple[str, int]] = {}
        position = 0
        for scene in self.scenes:
            for line in scene.dialogue:
                out[line.line_id] = (scene.scene_id, position)
                position += 1
        return out

    # -- Stage 8's hand-off -------------------------------------------------

    def to_dialogue_timeline(self, *, locked: bool | None = None
                             ) -> DialogueTimeline:
        """k102's ``DialogueTimeline``, in screenplay order.

        A COLLECTION, not a translation: ``Scene.dialogue`` already holds k102
        ``Line`` objects, so the ids, speakers, emotions and per-line budgets
        arrive unchanged and the 1:1 mapping is structural. ``locked`` defaults
        to this screenplay's own flag — a locked script yields a locked
        dialogue timeline, which is precisely what ``build_audio_master``
        demands before it will synthesize anything."""
        lines = self.lines
        if not lines:
            raise ScreenplayRefused(
                f"screenplay {self.title!r} has no dialogue, so there is no "
                f"dialogue timeline to build — a silent film is a legitimate "
                f"artifact, but Stage 8 has nothing to lock and this call is "
                f"the wrong one to make",
                errors=("screenplay has no dialogue lines",))
        want = self.locked if locked is None else bool(locked)
        return DialogueTimeline(lines=lines, locked=want)

    def lock(self) -> "Screenplay":
        """The locked twin (self if already locked)."""
        return self if self.locked else replace(self, locked=True)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title,
                "scenes": [s.to_dict() for s in self.scenes],
                "logline": self.logline,
                "characters": list(self.characters),
                "plot_digest": self.plot_digest,
                "locked": self.locked}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Screenplay":
        return cls(title=d.get("title", ""),
                   scenes=tuple(Scene.from_dict(s) for s in d.get("scenes", ())),
                   logline=d.get("logline", ""),
                   characters=tuple(d.get("characters", ())),
                   plot_digest=d.get("plot_digest"),
                   locked=bool(d.get("locked", False)))


# ---------------------------------------------------------------------------
# Stage 7 — continuity, derived mechanically
# ---------------------------------------------------------------------------

#: Every key :func:`build_continuity` writes into a ``ContinuityState``.
STATE_KEYS: tuple[str, ...] = (
    "scene", "location", "time_of_day", "story_time_s", "weather", "lighting",
    "present", "props", "wardrobe", "props_seen", "characters_seen",
)

#: The keys that CARRY across a cut and therefore chain exactly:
#: ``after(N)[k] == before(N+1)[k]``. See the module docstring for why
#: ``location`` / ``present`` are deliberately not among them.
CARRIED_KEYS: tuple[str, ...] = ("props_seen", "characters_seen")


def _mentions(text: str, phrase: str) -> bool:
    """Whole-token containment, the same matcher idiom
    ``video_intel.shot_intent._has_cue`` uses (mirrored, not imported — that
    one is private and is about view cues, this one is about props).

    Whole-token matters: a scene mentioning "the knifemaker" has not put a
    knife on the table."""
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase.lower())
                     + r"(?![a-z0-9])", text.lower()) is not None


def props_in_play(scene: Scene, inventory: Sequence[str]) -> tuple[str, ...]:
    """The props this scene actually has on screen: the ones it declares, plus
    any item of the screenplay's ``inventory`` its action, staging or dialogue
    NAMES. Sorted, so a state is comparable without the caller remembering to.

    Deterministic keyword recognition over a CLOSED inventory, never extraction
    of nouns from prose: a continuity bible that invented "a chair" because the
    action said "she sat" would be fabricating evidence for the judge."""
    text = " ".join([scene.action, scene.staging]
                    + [ln.text for ln in scene.dialogue])
    found = set(scene.props)
    for item in inventory:
        if item not in found and _mentions(text, item):
            found.add(item)
    return tuple(sorted(found))


def _state(scene: Scene, *, present: Sequence[str], props: Sequence[str],
           props_seen: Sequence[str], characters_seen: Sequence[str]
           ) -> FrozenParams:
    return FrozenParams({
        "scene": scene.scene_id,
        "location": scene.location,
        "time_of_day": scene.time_of_day,
        "story_time_s": scene.story_time_s,
        "weather": scene.weather,
        "lighting": scene.lighting,
        "present": sorted(present),
        "props": sorted(props),
        "wardrobe": sorted(scene.wardrobe),
        "props_seen": sorted(props_seen),
        "characters_seen": sorted(characters_seen),
    })


def build_continuity(screenplay: Screenplay,
                     shots: "ShotPlanDraft | None" = None,
                     *, notes: str = "") -> ContinuityBible:
    """Doc Stage 7, derived — never authored.

    Fills k104's ``ContinuityBible`` shell: one ``ContinuityState`` per SEGMENT
    with an explicit ``state_before`` and ``state_after``, plus the standing
    inventories (characters, wardrobe, props, locations) the breakdown draws
    from and a ``notes`` block carrying the per-scene time/weather/lighting
    line Stage 7 also asks for.

    WITHOUT ``shots`` the segments are the SCENES (``segment_id == scene_id``).
    WITH a :class:`ShotPlanDraft` they are the shots, and a scene's shots are
    chained: the first shot opens on the scene's opening state, the last closes
    on the scene's closing state, and everything between sits on the mid-scene
    state (entrances applied, exits not yet). That attribution is a documented
    CHOICE, not a measurement — a screenplay says a character enters during the
    scene, not in which shot — and it is made in the only direction that keeps
    the chain sound: somebody who speaks is present from the first shot they
    could be, and somebody who leaves is gone only after the last.

    No LLM, no clock, no randomness. Same screenplay, same bible, same digest."""
    if not isinstance(screenplay, Screenplay):
        raise TypeError(f"build_continuity takes a Screenplay, got "
                        f"{type(screenplay).__name__}")
    inventory = screenplay.prop_inventory
    chain = screenplay.presence_chain()
    by_scene: dict[str, list[str]] = {}
    if shots is not None:
        if not isinstance(shots, ShotPlanDraft):
            raise TypeError(f"build_continuity(shots=) takes a ShotPlanDraft, "
                            f"got {type(shots).__name__}")
        for design in shots.designs:
            by_scene.setdefault(design.scene_id, []).append(design.segment_id)
        unknown = sorted(set(by_scene) - set(screenplay.scene_ids))
        if unknown:
            raise ShotPlanRefused(
                f"the shot plan names scene(s) {unknown} that this screenplay "
                f"does not have — the plan is stale")

    entries: list[ContinuityState] = []
    props_seen: list[str] = []
    chars_seen: list[str] = []
    note_lines: list[str] = []

    for index, scene in enumerate(screenplay.scenes):
        _, opened, closed = chain[index]
        mid = scene.cast_after_open(opened)
        scene_props = props_in_play(scene, inventory)

        seen_props_before = tuple(props_seen)
        seen_chars_before = tuple(chars_seen)
        for prop in scene_props:
            if prop not in props_seen:
                props_seen.append(prop)
        for name in mid:
            if name not in chars_seen:
                chars_seen.append(name)
        seen_props_after = tuple(props_seen)
        seen_chars_after = tuple(chars_seen)

        established = tuple(p for p in scene_props if p in seen_props_before)
        before = _state(scene, present=opened, props=established,
                        props_seen=seen_props_before,
                        characters_seen=seen_chars_before)
        after = _state(scene, present=closed, props=scene_props,
                       props_seen=seen_props_after,
                       characters_seen=seen_chars_after)

        note_lines.append(
            f"{scene.scene_id}: {scene.heading} | story_time={scene.story_time_s}s"
            f" | weather={scene.weather or '-'} | lighting={scene.lighting or '-'}"
            f" | screen direction={scene.staging or '-'}")

        segment_ids = by_scene.get(scene.scene_id, []) if shots is not None \
            else [scene.scene_id]
        if not segment_ids:
            continue
        if len(segment_ids) == 1:
            entries.append(ContinuityState(segment_id=segment_ids[0],
                                           state_before=before,
                                           state_after=after))
            continue
        middle = _state(scene, present=mid, props=scene_props,
                        props_seen=seen_props_after,
                        characters_seen=seen_chars_after)
        for position, segment_id in enumerate(segment_ids):
            first = position == 0
            last = position == len(segment_ids) - 1
            entries.append(ContinuityState(
                segment_id=segment_id,
                state_before=before if first else middle,
                state_after=after if last else middle))

    return ContinuityBible(
        entries=tuple(entries),
        characters=screenplay.characters,
        wardrobe=screenplay.wardrobe,
        props=inventory,
        locations=screenplay.locations,
        notes=notes or "\n".join(note_lines))


def chain_breaks(bible: ContinuityBible) -> tuple[tuple[str, str, str], ...]:
    """``((earlier_segment, later_segment, key), …)`` for every
    :data:`CARRIED_KEYS` value that changed between one segment's
    ``state_after`` and the next segment's ``state_before``.

    Empty for anything :func:`build_continuity` produces. Non-empty means an
    artifact was hand-edited or spliced, which is exactly the case where a
    continuity error is invisible by inspection."""
    out: list[tuple[str, str, str]] = []
    for first, second in zip(bible.entries, bible.entries[1:]):
        for key in CARRIED_KEYS:
            if first.state_after.get(key) != second.state_before.get(key):
                out.append((first.segment_id, second.segment_id, key))
    return tuple(out)


# ---------------------------------------------------------------------------
# Stage 9 — direction and visual conception
# ---------------------------------------------------------------------------

#: Doc Stage 9's ordered production subplan, verbatim. ``shoot`` is the
#: compiled ``ShotPlanEntry`` itself — the step that becomes a
#: ``SegmentSpec``, which is k104's job, not this module's.
SUBPLAN_STEPS: tuple[str, ...] = ("block", "light", "rehearse", "tweak",
                                  "shoot")

#: Speaking rate used ONLY when there is no ``AudioMaster``. Every window it
#: produces is flagged ``estimated=True``; it is a placeholder for scheduling,
#: never a timing the audio is later forced onto (Stage 8 forbids exactly that).
WORDS_PER_SECOND: float = 2.6
#: The floor for an estimated dialogue window, and the beat held after a line.
MIN_LINE_S: float = 1.2
LINE_PAUSE_S: float = 0.4
#: An estimated window for a scene with no dialogue at all.
SILENT_SCENE_S: float = 3.0

#: shot_size -> a focal length that is not a lie about the framing. A table
#: rather than a formula so a director can read and argue with it.
LENS_FOR_SIZE: dict[str, float] = {
    "extreme_wide": 18.0, "wide": 24.0, "full": 28.0, "medium_wide": 35.0,
    "medium": 50.0, "medium_close": 85.0, "close": 85.0,
    "extreme_close": 100.0, "insert": 100.0, "two_shot": 35.0,
    "over_shoulder": 50.0,
}


def estimated_line_seconds(line: Line) -> float:
    """A duration ESTIMATE for one line, from its word count.

    Bounded below by :data:`MIN_LINE_S` and by the line's own ``max_seconds``
    above when the operator declared one. Deliberately crude: its only job is
    to let a shot plan exist before Stage 8 has run, and every window it
    produces says so."""
    words = len([w for w in str(line.text).split() if w.strip()])
    seconds = max(MIN_LINE_S, words / WORDS_PER_SECOND)
    if line.max_seconds is not None:
        seconds = min(seconds, float(line.max_seconds))
    return quantize(seconds)


@dataclass(frozen=True, slots=True)
class ShotDesign(ContentAddressed):
    """One shot, with Stage 9's ordered subplan encoded as FIELDS.

    ``blocking`` is *block*, ``lighting`` is *light*, ``rehearse`` and ``tweak``
    are the two steps that are CHECKS rather than decisions — so they are
    written as acceptance criteria and ride into the ``ShotPlanEntry``'s rubric,
    which is what k104 turns into the plan graph's ``AcceptanceTest``s. The
    fifth step, *shoot*, is :meth:`to_entry` itself.

    ``estimated`` is the honesty flag. False means the window was READ off an
    ``AudioMaster`` (Stage 8: the audio is authoritative and the shot is cut to
    it). True means nobody has recorded the audio yet and this duration is a
    guess — which is fine for planning and is never allowed to become the thing
    the audio is retimed to fit."""

    segment_id: str
    scene_id: str
    line_ids: tuple[str, ...] = ()
    start_s: float = 0.0
    end_s: float = 0.0
    estimated: bool = False
    camera: Mapping[str, Any] = field(default_factory=FrozenParams)
    blocking: str = ""
    lighting: str = ""
    rehearse: str = ""
    tweak: str = ""
    rubric: tuple[str, ...] = ()
    beat: str = ""

    def __post_init__(self) -> None:
        require_text(self.segment_id, "ShotDesign.segment_id")
        require_text(self.scene_id, "ShotDesign.scene_id")
        object.__setattr__(self, "line_ids",
                           str_tuple(self.line_ids, "ShotDesign.line_ids"))
        object.__setattr__(self, "start_s", quantize(self.start_s))
        object.__setattr__(self, "end_s", quantize(self.end_s))
        if self.start_s < 0:
            raise ValueError(f"ShotDesign({self.segment_id}).start_s must be "
                             f"non-negative, got {self.start_s}")
        if self.end_s < self.start_s:
            raise ValueError(f"ShotDesign({self.segment_id}) ends before it "
                             f"starts: {self.end_s} < {self.start_s}")
        if not isinstance(self.estimated, bool):
            raise TypeError(f"ShotDesign.estimated must be a bool, got "
                            f"{type(self.estimated).__name__}")
        object.__setattr__(self, "camera", FrozenParams(self.camera))
        unknown = sorted(set(self.camera) - CAMERA_KEYS)
        if unknown:
            raise ValueError(
                f"ShotDesign({self.segment_id}).camera has unknown key(s) "
                f"{unknown}; the Stage 9 vocabulary is {sorted(CAMERA_KEYS)}")
        for name in ("blocking", "lighting", "rehearse", "tweak", "beat"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        object.__setattr__(self, "rubric",
                           str_tuple(self.rubric, "ShotDesign.rubric"))

    @property
    def duration_s(self) -> float:
        return quantize(self.end_s - self.start_s)

    @property
    def subplan(self) -> tuple[tuple[str, str], ...]:
        """Stage 9's five steps IN ORDER, each with what this shot says for it.
        The order is the contract — "rehearse then tweak" is the whole reason
        the doc numbers them — so it is returned as a sequence rather than left
        for a reader to reassemble from field names."""
        return (("block", self.blocking), ("light", self.lighting),
                ("rehearse", self.rehearse), ("tweak", self.tweak),
                ("shoot", f"{self.segment_id} "
                          f"[{self.start_s}s..{self.end_s}s]"
                          f"{' (estimated)' if self.estimated else ''}"))

    def to_entry(self) -> ShotPlanEntry:
        """The *shoot* step: k104's ``ShotPlanEntry``, ready for the lock.

        ``rehearse`` and ``tweak`` join the rubric (in that order, after the
        shot's own criteria) because that is where an unmet one becomes a
        judgeable failure instead of a note nobody reads."""
        rubric = list(self.rubric)
        for text in (self.rehearse, self.tweak):
            if text and text not in rubric:
                rubric.append(text)
        return ShotPlanEntry(
            segment_id=self.segment_id, line_ids=self.line_ids,
            start_s=self.start_s, end_s=self.end_s, camera=self.camera,
            blocking=self.blocking or None, lighting=self.lighting or None,
            rubric=tuple(rubric))

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "scene_id": self.scene_id,
                "line_ids": list(self.line_ids), "start_s": self.start_s,
                "end_s": self.end_s, "estimated": self.estimated,
                "camera": FrozenParams(self.camera).to_dict(),
                "blocking": self.blocking, "lighting": self.lighting,
                "rehearse": self.rehearse, "tweak": self.tweak,
                "rubric": list(self.rubric), "beat": self.beat}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ShotDesign":
        return cls(segment_id=d["segment_id"], scene_id=d["scene_id"],
                   line_ids=tuple(d.get("line_ids", ())),
                   start_s=d.get("start_s", 0.0), end_s=d.get("end_s", 0.0),
                   estimated=bool(d.get("estimated", False)),
                   camera=FrozenParams(d.get("camera") or {}),
                   blocking=d.get("blocking", ""),
                   lighting=d.get("lighting", ""),
                   rehearse=d.get("rehearse", ""), tweak=d.get("tweak", ""),
                   rubric=tuple(d.get("rubric", ())),
                   beat=d.get("beat", ""))


@dataclass(frozen=True, slots=True)
class ShotPlanDraft(ContentAddressed):
    """The Stage 9 shot list with its design detail attached.

    ``plan`` is the k104 ``ShotPlan`` that goes into ``ProductionLock.lock()``;
    this wrapper is what carries the two things a ``ShotPlanEntry`` has no
    field for and should not grow one for — the ``estimated`` flag and the
    scene each shot belongs to. Keeping them here rather than widening k104's
    shipped artifact means the lock's shape is unchanged and a plan built
    before Stage 8 is still distinguishable from one built after it."""

    designs: tuple[ShotDesign, ...] = ()
    audio_first: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "designs", tuple(self.designs))
        for design in self.designs:
            if not isinstance(design, ShotDesign):
                raise TypeError(f"ShotPlanDraft.designs takes ShotDesign, got "
                                f"{type(design).__name__}")
        ids = [d.segment_id for d in self.designs]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ShotPlanRefused(f"ShotPlanDraft has two shots for segment(s) "
                                  f"{dupes}")
        if not isinstance(self.audio_first, bool):
            raise TypeError("ShotPlanDraft.audio_first must be a bool")

    @property
    def plan(self) -> ShotPlan:
        """k104's ``ShotPlan`` — the artifact the production lock takes."""
        return ShotPlan(entries=tuple(d.to_entry() for d in self.designs))

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return tuple(d.segment_id for d in self.designs)

    @property
    def estimated_ids(self) -> tuple[str, ...]:
        """Every shot whose window is a guess. Empty is the Stage 8 condition:
        the whole plan is cut to recorded audio."""
        return tuple(d.segment_id for d in self.designs if d.estimated)

    def design(self, segment_id: str) -> ShotDesign:
        for item in self.designs:
            if item.segment_id == segment_id:
                return item
        raise KeyError(f"no shot design for segment {segment_id!r}")

    def designs_for_scene(self, scene_id: str) -> tuple[ShotDesign, ...]:
        return tuple(d for d in self.designs if d.scene_id == scene_id)

    def to_dict(self) -> dict[str, Any]:
        return {"designs": [d.to_dict() for d in self.designs],
                "audio_first": self.audio_first}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ShotPlanDraft":
        return cls(designs=tuple(ShotDesign.from_dict(x)
                                 for x in d.get("designs", ())),
                   audio_first=bool(d.get("audio_first", False)))


def _beat_for(index: int, total: int) -> str:
    """``prompt_seeds.beat_for_index``, LAZILY. Called at this call site rather
    than inside a contract module (k104's dispatch record asked for exactly
    that) and degrading to "" when the studio side is not installed — an
    absent dramatic-beat hint is a missing flourish, never a failure."""
    try:
        from hugpy_video.intel.prompt_seeds import beat_for_index
    except Exception:      # pragma: no cover - depends on install layout
        return ""
    try:
        return str(beat_for_index(index, max(total, 1)) or "")
    except Exception:      # pragma: no cover - defensive
        return ""


def _shot_size(*, first_in_scene: bool, line: Line | None,
               present_count: int) -> str:
    """The framing table. Deterministic and small enough to argue with:

        first shot, no dialogue        -> wide           (establish the place)
        first shot, dialogue           -> medium_wide    (establish, then talk)
        dialogue, two or more present  -> medium_close   (play the reaction)
        dialogue, one present          -> medium
        no dialogue                    -> medium
    """
    if first_in_scene and line is None:
        return "wide"
    if first_in_scene:
        return "medium_wide"
    if line is not None and present_count >= 2:
        return "medium_close"
    return "medium"


def _camera_for(scene: Scene, *, size: str, line: Line | None,
                present: Sequence[str], turning_point: bool) -> FrozenParams:
    """One shot's camera block, in k104's CLOSED vocabulary.

    ``view`` comes from ``shot_intent`` via k104's lazy wrapper and is OMITTED
    when the prose carries no orientation cue — the segment then inherits the
    movie-level DNA, which is exactly what ``derive_view_from_prompt`` promises
    for an unrecognizable prompt. Nothing here invents a view."""
    camera: dict[str, Any] = {
        "shot_size": size,
        "movement": "push_in" if turning_point else "static",
        "lens_mm": LENS_FOR_SIZE.get(size, 50.0),
        "angle": "eye",
        "height": "eye height",
        "framing": (f"{line.speaker} centre frame, {scene.location}"
                    if line is not None
                    else f"{scene.location}, {scene.time_of_day}"),
        "eyeline": (" to ".join(present[:2]) if len(present) >= 2
                    else (present[0] if present else "off-screen")),
        "focus": (f"hold focus on {line.speaker}" if line is not None
                  else "deep focus"),
    }
    text = " ".join([scene.action, scene.staging]
                    + ([line.text] if line is not None else []))
    view = camera_view_from_prompt(text)
    if view in CAMERA_VIEWS:
        camera["view"] = view
    if camera["movement"] not in CAMERA_MOVES:      # pragma: no cover - guard
        camera["movement"] = "static"
    if camera["shot_size"] not in SHOT_SIZES:       # pragma: no cover - guard
        camera["shot_size"] = "medium"
    return FrozenParams(camera)


def _rubric_for(scene: Scene, *, present: Sequence[str], line: Line | None,
                size: str) -> tuple[str, ...]:
    """The per-shot acceptance rubric Stage 9 requires. Every criterion is
    CHECKABLE against the produced clip; none of them is a compliment."""
    out = [
        f"the shot is {scene.location} at {scene.time_of_day} "
        f"({'exterior' if scene.heading.upper().startswith('EXT') else 'interior'})",
        f"exactly these characters are on screen: "
        f"{', '.join(present) if present else '(none)'}",
        f"framing is {size}",
    ]
    if line is not None:
        out.append(f"{line.speaker} is speaking line {line.line_id} on camera"
                   + (f", {line.emotion}" if line.emotion else ""))
    if scene.lighting:
        out.append(f"lighting reads as: {scene.lighting}")
    if scene.weather:
        out.append(f"weather reads as: {scene.weather}")
    return tuple(out)


def build_shot_plan(screenplay: Screenplay,
                    audio_master: AudioMaster | None = None,
                    *, plot: PlotSpec | None = None) -> ShotPlanDraft:
    """Doc Stage 9 — screenplay (+ locked audio) -> the shot list.

    AUDIO-FIRST when ``audio_master`` is given: one shot per LINE, its window
    read straight off the master (``[start_s, next_start_s]``, so consecutive
    shots abut exactly and the lock's partition check passes), ``estimated``
    False. A scene with NO dialogue takes the audio's own gap around it — the
    seconds between the previous line's end and the next line's start — and is
    still flagged ``estimated`` because its length was not set by dialogue.
    When there is no gap at all the shot is emitted with a zero-length window
    rather than by inventing time and pushing every later shot off the audio:
    that is k107's ``SHOT_TOO_SHORT``, honestly reported, not silently padded.

    WITHOUT an audio master every window is an estimate laid end to end from
    zero and every design carries ``estimated=True``. The plan is usable for
    scheduling and feasibility; it must not be locked as a timeline, which is
    why ``ProductionLock.lock()`` demands a LOCKED ``AudioMaster`` anyway.

    ``plot`` is optional and buys exactly one thing: a scene whose ``beat_id``
    names a Stage 5 TURNING POINT gets ``movement="push_in"`` instead of
    ``"static"``. That is the whole of the plot's influence on the camera, and
    it is opt-in because a shot plan must be buildable from the screenplay
    alone — the screenplay is the artifact that covers the entire video.

    Refuses (``ShotPlanRefused``) a master that does not carry a line the
    screenplay speaks, or whose line order disagrees with the screenplay's —
    both mean the plan would be built against a different script."""
    if not isinstance(screenplay, Screenplay):
        raise TypeError(f"build_shot_plan takes a Screenplay, got "
                        f"{type(screenplay).__name__}")
    if audio_master is not None and not isinstance(audio_master, AudioMaster):
        raise TypeError(f"build_shot_plan(audio_master=) takes an AudioMaster, "
                        f"got {type(audio_master).__name__}")
    if plot is not None and not isinstance(plot, PlotSpec):
        raise TypeError(f"build_shot_plan(plot=) takes a PlotSpec, got "
                        f"{type(plot).__name__}")
    turning_points = set(plot.turning_points) if plot is not None else set()

    if audio_master is not None:
        missing = [l for l in screenplay.line_ids
                   if l not in set(audio_master.line_ids)]
        if missing:
            raise ShotPlanRefused(
                f"the audio master does not carry line(s) {missing} that the "
                f"screenplay speaks (has: {list(audio_master.line_ids)}) — the "
                f"master was built from a different script")
        order = [l for l in audio_master.line_ids if l in set(screenplay.line_ids)]
        if order != list(screenplay.line_ids):
            raise ShotPlanRefused(
                f"the audio master plays the lines in a different order than "
                f"the screenplay writes them ({order} vs "
                f"{list(screenplay.line_ids)}) — shots cut to this master "
                f"would tell a different story")

    chain = screenplay.presence_chain()
    designs: list[ShotDesign] = []
    cursor = 0.0
    total_shots = sum(max(len(s.dialogue), 1) for s in screenplay.scenes)
    shot_number = 0

    for index, scene in enumerate(screenplay.scenes):
        _, opened, _closed = chain[index]
        cast = scene.cast_after_open(opened)
        turning = scene.beat_id is not None and scene.beat_id in turning_points
        lines = list(scene.dialogue)
        units: list[Line | None] = list(lines) if lines else [None]

        for position, line in enumerate(units):
            segment_id = f"{scene.scene_id}-{position + 1}"
            size = _shot_size(first_in_scene=position == 0, line=line,
                              present_count=len(cast))
            camera = _camera_for(scene, size=size, line=line, present=cast,
                                 turning_point=turning)

            if audio_master is not None and line is not None:
                timing = audio_master.timing(line.line_id)
                # Clamp to the master: a shot is cut TO the audio, never past
                # it (the same rule ProductionLock.lock enforces, applied here
                # so a trailing pause that runs off the end of a hand-built
                # master produces a short shot rather than an unlockable plan).
                start = timing.start_s
                end = max(start, min(timing.next_start_s,
                                     audio_master.total_seconds))
                estimated = False
            elif audio_master is not None:
                start, end = _silent_window(screenplay, audio_master, index)
                estimated = True
            else:
                span = (estimated_line_seconds(line) + LINE_PAUSE_S
                        if line is not None else SILENT_SCENE_S)
                start, end = cursor, quantize(cursor + span)
                estimated = True
            cursor = max(cursor, end)

            duration = quantize(end - start)
            blocking = (
                f"{', '.join(cast) if cast else 'empty frame'} in "
                f"{scene.location}; {scene.staging or scene.action or 'as written'}")
            lighting = (scene.lighting
                        or f"motivated by {scene.time_of_day.lower()} in "
                           f"{scene.location}")
            rehearse = (
                f"the {duration}s window carries "
                + (f"line {line.line_id} end to end with no truncation and no "
                   f"dead air" if line is not None
                   else f"the action beat without stalling")
                + (" (window is an ESTIMATE — re-rehearse against the locked "
                   "audio master before the production lock)" if estimated
                   else ""))
            tweak = (
                f"framing, eyelines and {size} hold across the whole window; "
                f"no identity drift, no wardrobe or prop change from the "
                f"continuity state for {segment_id}")

            designs.append(ShotDesign(
                segment_id=segment_id, scene_id=scene.scene_id,
                line_ids=(line.line_id,) if line is not None else (),
                start_s=start, end_s=end, estimated=estimated, camera=camera,
                blocking=blocking, lighting=lighting, rehearse=rehearse,
                tweak=tweak,
                rubric=_rubric_for(scene, present=cast, line=line, size=size),
                beat=_beat_for(shot_number, total_shots)))
            shot_number += 1

    return ShotPlanDraft(designs=tuple(designs),
                         audio_first=audio_master is not None)


def _silent_window(screenplay: Screenplay, master: AudioMaster,
                   scene_index: int) -> tuple[float, float]:
    """The audio gap a dialogue-free scene sits in.

    From the end of the last line spoken BEFORE this scene (or 0.0) to the
    start of the first line spoken after it (or the master's total). A scene
    wedged between two lines with no silence between them gets a zero-length
    window — the honest statement that the recorded audio left it no room."""
    before: float = 0.0
    for scene in screenplay.scenes[:scene_index]:
        for line_id in scene.line_ids:
            before = max(before, master.timing(line_id).next_start_s)
    after: float | None = None
    for scene in screenplay.scenes[scene_index + 1:]:
        for line_id in scene.line_ids:
            start = master.timing(line_id).start_s
            after = start if after is None else min(after, start)
    end = master.total_seconds if after is None else after
    if end < before:
        end = before
    return quantize(before), quantize(end)


# ---------------------------------------------------------------------------
# Stage 11 — the lock, with a screenplay in it
# ---------------------------------------------------------------------------


def lock_production(snapshot: GenerationSnapshot, *,
                    screenplay: Screenplay,
                    audio_master: AudioMaster,
                    continuity: ContinuityBible | None = None,
                    shots: ShotPlanDraft | None = None,
                    storyboard: Any = None,
                    **kwargs: Any) -> ProductionLock:
    """``ProductionLock.lock()`` with the Stage 6 artifact actually in it.

    k104 left ``ProductionLock.screenplay_digest`` as ``str | None`` and always
    passed None, which honestly meant "there is no screenplay". There is one
    now, and NOTHING in ``production.py`` had to change to accept it: the field
    and the ``lock(screenplay_digest=…)`` keyword already exist, and the digest
    joins ``parent_digests`` on its own. What this wrapper adds is the two
    checks that only make sense once a screenplay exists:

    1. **The screenplay must be LOCKED.** Same rule Stage 8 puts on the audio
       master, one stage earlier: locking a production against a script that
       may still change is how a shot plan ends up cut to dialogue nobody
       kept. ``Screenplay.lock()`` is the one call.
    2. **The audio master must realize THIS screenplay's dialogue.**
       ``AudioMaster.timeline_digest`` points at the locked
       ``DialogueTimeline``; this screenplay produces exactly one of those, so
       the two digests must agree. A mismatch is a master built from an older
       draft — invisible by inspection, and caught here in one comparison.

    ``continuity`` and ``shots`` default to the derived ones, so the ordinary
    call is ``lock_production(snapshot, screenplay=s, audio_master=m)``.

    ``storyboard`` (k124, optional) is a ``storyboard.Storyboard``. It adds a
    THIRD check of the same kind as the two above, and for the same reason:

    3. **The board must be LOCKED and must be drawn against THIS shot plan.**
       An unlocked board can still gain and lose accepted frames, so a
       production locked against one would be locked against a reference that
       may still change; and a board whose ``shot_plan_digest`` names a
       different plan is a set of references for another cut of this film,
       which is invisible by inspection and caught here in one comparison.
       The board is duck-typed (``locked`` / ``digest`` /
       ``shot_plan_digest``) so this module does not import ``storyboard`` —
       the dependency runs the other way."""
    if not isinstance(screenplay, Screenplay):
        raise TypeError(f"lock_production(screenplay=) takes a Screenplay, "
                        f"got {type(screenplay).__name__}")
    if not screenplay.locked:
        raise LockRefused(
            f"screenplay {screenplay.title!r} is not locked: a production "
            f"cannot lock against a script that may still change (Stage 11 "
            f"versions and locks the screenplay, the continuity bible, the "
            f"audio master and the shot plan together). Call "
            f"Screenplay.lock() on the accepted draft first")
    if isinstance(audio_master, AudioMaster):
        expected = screenplay.to_dialogue_timeline(locked=True).digest
        if audio_master.timeline_digest != expected:
            raise LockRefused(
                f"the audio master realizes dialogue timeline "
                f"{audio_master.timeline_digest[:12]}… but this screenplay's "
                f"locked timeline is {expected[:12]}… — the master was built "
                f"from a different draft of the script")
    if shots is None:
        shots = build_shot_plan(screenplay, audio_master)
    if continuity is None:
        continuity = build_continuity(screenplay, shots)
    storyboard_digest = None
    if storyboard is not None:
        storyboard_digest = _storyboard_digest_for(storyboard, shots)
    return ProductionLock.lock(snapshot, audio_master=audio_master,
                               continuity=continuity, shot_plan=shots.plan,
                               screenplay_digest=screenplay.digest,
                               storyboard_digest=storyboard_digest, **kwargs)


def _storyboard_digest_for(storyboard: Any, shots: ShotPlanDraft) -> str:
    """The digest a locked, matching storyboard contributes, or a refusal.

    Duck-typed on purpose (see :func:`lock_production`'s docstring): the board
    lives in ``oracle/storyboard.py``, which imports ``postproduction``, which
    imports this module's siblings. Naming the type here would close a cycle
    for a three-attribute check."""
    locked = getattr(storyboard, "locked", None)
    digest = getattr(storyboard, "digest", None)
    if locked is None or not isinstance(digest, str) or not digest:
        raise TypeError(
            f"lock_production(storyboard=) takes a Storyboard (locked/digest/"
            f"shot_plan_digest), got {type(storyboard).__name__}")
    if not locked:
        raise LockRefused(
            "the storyboard is not locked: an unlocked board can still gain "
            "and lose accepted frames, so a production locked against one is "
            "locked against a reference that may still change. Call "
            "Storyboard.lock() on the accepted board first")
    if not getattr(storyboard, "accepted_frames", ()):
        raise LockRefused(
            "the storyboard has no accepted frame — locking a board nobody "
            "accepted anything on records a reference the shoot cannot use")
    drawn_against = getattr(storyboard, "shot_plan_digest", None)
    if drawn_against and drawn_against != shots.plan.digest:
        raise LockRefused(
            f"the storyboard was drawn against shot plan "
            f"{str(drawn_against)[:12]}… but this production locks "
            f"{shots.plan.digest[:12]}… — those references are for another "
            f"cut of this film")
    unknown = tuple(s for s in getattr(storyboard, "accepted_shots", ())
                    if s not in shots.segment_ids)
    if unknown:
        raise LockRefused(
            f"the storyboard accepts frame(s) for shot(s) {list(unknown)} that "
            f"this shot plan does not contain (has: "
            f"{list(shots.segment_ids)})")
    return digest


# ---------------------------------------------------------------------------
# LLM-assisted authoring — prompts, parsing, ONE bounded repair
# ---------------------------------------------------------------------------

#: The injected model. One string in, one string out — no client, no session,
#: no provider. Tests pass a function; production passes :func:`bind_llm`.
Llm = Callable[[str], str]

#: How much of a rejected reply travels back into the repair prompt. Bounded so
#: a model that answered with a novel cannot push the schema out of context.
MAX_ECHO_CHARS: int = 4000
#: How many validator errors the repair prompt carries. All of them, in
#: practice; the cap exists so a pathological reply cannot do the same thing.
MAX_REPAIR_ERRORS: int = 25

PLOT_SYSTEM: str = (
    "You are a story editor constructing the PLOT for one short film that will "
    "be generated shot by shot.\n"
    "\n"
    "RULES:\n"
    "1. Every beat must name at least one character who is IN it. A beat with "
    "nobody in it cannot be filmed.\n"
    "2. Every character you declare must appear in at least one beat. Do not "
    "invent a character you then never use.\n"
    "3. `causes` on a beat lists the ids of EARLIER beats that caused it. "
    "Every id must be a beat you declared, and it must come before this one in "
    "the list. The first beat has no causes.\n"
    "4. Give every character a goal, a conflict and an arc. All three are "
    "required and all three must be specific to this story.\n"
    "5. Mark the beats that turn the story with `turning_point: true`.\n"
    "6. Write the ending. A plot that stops is not a plot.\n"
    "7. Never invent identity attributes (age, gender, ethnicity, build) for a "
    "character unless the source material states them.\n"
)

SCREENPLAY_SYSTEM: str = (
    "You are a screenwriter turning a locked plot into ONE ordered screenplay "
    "covering the ENTIRE film. Every scene will be shot as written.\n"
    "\n"
    "RULES:\n"
    "1. Scene headings are sluglines: they must start with one of "
    f"{list(SCENE_PREFIXES)} — for example \"INT. KITCHEN - NIGHT\".\n"
    "2. `present_at_open` lists who is already in the room when the scene "
    "starts; `entrances` lists who walks in during it; `exits` lists who "
    "leaves. A character cannot enter a room they are already in, and cannot "
    "exit one they were never in.\n"
    "3. EVERY line of dialogue must be spoken by someone who is present at the "
    "open or who enters. This is the rule most often broken; check it twice.\n"
    "4. Every `line_id` must be unique across the whole screenplay. Use short "
    "stable ids like \"l1\", \"l2\".\n"
    "5. `story_time_s` is the time in the STORY, in seconds from the film's "
    "earliest moment. It must never go backwards from one scene to the next "
    "unless the previous scene's `transition` is one of "
    f"{sorted(FLASHBACK_TRANSITIONS)}.\n"
    "6. `transition` is written on the scene it leads OUT of, and must be one "
    f"of {sorted(TRANSITIONS)}. Use \"CONTINUOUS:\" only when no time passes "
    "and nobody moves between the two scenes — the next scene then inherits "
    "this one's cast exactly.\n"
    f"7. `av_events[].kind` must be one of {sorted(AV_EVENT_KINDS)}.\n"
    "8. List every physical object that matters in `props`, on the first scene "
    "it appears in. The continuity breakdown is built from that list.\n"
    "9. Never invent identity attributes for a character that the plot does "
    "not state.\n"
)

#: Doc Stage 5's three input shapes -> the one paragraph of guidance that
#: differs between them. NOTHING else differs: all three produce the same
#: artifact, checked by the same constructor.
MODE_GUIDANCE: dict[str, str] = {
    "complete": (
        "The source material below is a screenplay or a full outline. "
        "EXTRACT the plot that is already there — premise, characters, beats, "
        "causality, ending. Do not replace the writer's story with a better "
        "one. Add only what is genuinely absent, and mark nothing as a "
        "turning point that the source does not turn on."),
    "partial": (
        "The source material below is partial: notes, fragments, a character "
        "sketch, some scenes. COMPLETE it. Keep every element that is there, "
        "in the form it is there, and construct the connective causality, the "
        "missing beats and the ending around it."),
    "minimal": (
        "There is little or no narrative input below. CONSTRUCT the whole "
        "plot from the request. Keep it small enough to film: two or three "
        "characters, four to six beats, one clear turning point, one ending."),
}


def plot_input_mode(input_text: str) -> str:
    """Which of doc Stage 5's three input shapes this request is, decided
    deterministically.

    Under :data:`MINIMAL_WORDS` words is "little or no narrative input"; text
    carrying sluglines is a screenplay fragment and therefore "complete"
    material to extract from; everything else is "partial".

    The mode selects ONE guidance paragraph and is recorded as provenance on
    the artifact. It never relaxes a validator — an artifact built from a
    one-line request is held to exactly the rules an artifact built from a
    finished outline is."""
    text = str(input_text or "").strip()
    words = [w for w in text.split() if w.strip()]
    if len(words) < MINIMAL_WORDS:
        return "minimal"
    upper = text.upper()
    if any(prefix in upper for prefix in SCENE_PREFIXES):
        return "complete"
    return "partial"


def schema_block(dataclass_type: type) -> str:
    """The JSON Schema for one artifact, as the exact text a prompt embeds.

    Generated from the dataclass by k101's ``json_schema_for`` — so the schema
    the model is shown and the constructor the reply is checked against are the
    SAME definition, and a field added to the artifact appears in the prompt
    with no second edit anywhere. Sorted keys, indent 2: a prompt that changed
    every time it was built could not be cached or diffed."""
    return json.dumps(json_schema_for(dataclass_type), sort_keys=True,
                      indent=2)


def build_plot_prompt(input_text: str, mode: str = "complete") -> str:
    """The Stage 5 authoring prompt: rules, the mode's guidance, the source
    material verbatim, and the generated schema."""
    if mode not in INPUT_MODES:
        raise ValueError(f"mode {mode!r} is not one of {list(INPUT_MODES)}")
    source = str(input_text or "").strip() or "(no narrative input supplied)"
    return (
        f"{PLOT_SYSTEM}\n"
        f"INPUT MODE: {mode}\n{MODE_GUIDANCE[mode]}\n\n"
        f"SOURCE MATERIAL:\n{source}\n\n"
        f"JSON SCHEMA — your object MUST validate against this exactly:\n"
        f"{schema_block(PlotSpec)}\n\n"
        f"Return ONLY the JSON object. No markdown, no commentary, no "
        f"explanation, no code fence.")


def build_screenplay_prompt(plot: PlotSpec) -> str:
    """The Stage 6 authoring prompt: rules, the LOCKED plot as canonical JSON,
    and the generated schema.

    The plot travels as its own canonical JSON rather than as prose, because
    the screenplay's beat ids, character names and causality have to match it
    EXACTLY and a paraphrase is where they stop matching."""
    if not isinstance(plot, PlotSpec):
        raise TypeError(f"build_screenplay_prompt takes a PlotSpec, got "
                        f"{type(plot).__name__}")
    return (
        f"{SCREENPLAY_SYSTEM}\n"
        f"THE LOCKED PLOT (digest {plot.digest[:12]}…) — use these exact "
        f"character names and beat ids:\n"
        f"{json.dumps(plot.to_dict(), sort_keys=True, indent=2)}\n\n"
        f"JSON SCHEMA — your object MUST validate against this exactly:\n"
        f"{schema_block(Screenplay)}\n\n"
        f"Write one scene per beat, in beat order, unless the story needs "
        f"more. Set `beat_id` on every scene to the beat it dramatizes.\n"
        f"Return ONLY the JSON object. No markdown, no commentary, no "
        f"explanation, no code fence.")


def build_repair_prompt(original: str, raw: str,
                        errors: Sequence[str]) -> str:
    """The ONE bounded reprompt.

    It carries the original instructions (so the schema is still in front of
    the model), the rejected reply, and EVERY validator error verbatim. Verbatim
    matters: "beat 'b3' is caused by 'b9', which is not a beat in this plot" is
    a repairable instruction, and "invalid plot" is not."""
    listed = "\n".join(f"{i + 1}. {e}"
                       for i, e in enumerate(list(errors)[:MAX_REPAIR_ERRORS]))
    echo = str(raw or "")[:MAX_ECHO_CHARS]
    return (
        f"{original}\n\n"
        f"--- YOUR PREVIOUS REPLY WAS REJECTED ---\n"
        f"{echo}\n\n"
        f"--- VALIDATION ERRORS ---\n"
        f"{listed}\n\n"
        f"Fix EVERY error listed above. Change nothing else. Return ONLY the "
        f"corrected JSON object.")


_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """``(object, "")`` or ``(None, why-not)``.

    Tolerant of exactly two things a small instruct model does anyway — a code
    fence, and prose either side of the object — and tolerant of nothing else.
    The brace scanner is string-aware so a ``"}"`` inside a line of dialogue
    cannot end the object early. A reply with no object is DATA (it becomes the
    error in the repair prompt), never an exception."""
    raw = str(text or "")
    stripped = _FENCE.sub("", raw.strip())
    start = stripped.find("{")
    if start < 0:
        return None, ("the reply contains no JSON object at all (expected a "
                      "single object starting with '{')")
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        return None, ("the JSON object in the reply is never closed (unbalanced "
                      "braces) — the reply was probably truncated")
    try:
        parsed = json.loads(stripped[start:end])
    except ValueError as exc:
        return None, f"the reply is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, (f"the reply parsed to a {type(parsed).__name__}, not a "
                      f"JSON object")
    return parsed, ""


def _errors_from(exc: BaseException) -> tuple[str, ...]:
    """Every problem the constructor found, or the one the interpreter did.

    ``ScreenplayError`` carries a full list (that is why it exists); a
    ``KeyError`` from a missing required field carries a name and needs a
    sentence around it, or the model reads ``'premise'`` and learns nothing."""
    if isinstance(exc, ScreenplayError):
        return exc.errors
    if isinstance(exc, KeyError):
        return (f"required field {exc.args[0]!r} is missing from the object",)
    return (f"{type(exc).__name__}: {exc}",)


def _author(stage: str, prompt: str, llm: Llm,
            build: Callable[[Mapping[str, Any]], Any]) -> Any:
    """Ask, validate, ONE repair, then a typed gap. The whole authoring
    contract in one place so ``author_plot`` and ``author_screenplay`` cannot
    drift apart.

    Two attempts, never three: a model that has been handed the schema AND its
    own errors and still cannot produce the artifact is not going to on the
    fourth try, and an unbounded repair ladder is a hang with good manners."""
    if not callable(llm):
        raise TypeError(f"llm must be callable (prompt) -> str, got "
                        f"{type(llm).__name__}")
    current = prompt
    raws: list[str] = []
    errors: tuple[str, ...] = ()
    unparsed = False

    for attempt in (0, 1):
        try:
            reply = llm(current)
        except Exception as exc:  # noqa: BLE001 — a model fault is a GAP
            return AuthoringGap(
                errors=(f"{type(exc).__name__}: {exc}",),
                raw=raws[-1] if raws else "", stage=stage, code="LLM_ERROR",
                attempts=attempt + 1, raw_attempts=tuple(raws))
        raw = "" if reply is None else str(reply)
        raws.append(raw)

        parsed, why = parse_json_object(raw)
        if parsed is None:
            errors, unparsed = (why,), True
        else:
            unparsed = False
            try:
                return build(parsed)
            except Exception as exc:  # noqa: BLE001 — validated, never coerced
                errors = _errors_from(exc)
        if attempt == 0:
            current = build_repair_prompt(prompt, raw, errors)

    return AuthoringGap(
        errors=errors, raw=raws[-1] if raws else "", stage=stage,
        code="AUTHORING_UNPARSED" if unparsed else "AUTHORING_INVALID",
        attempts=len(raws), raw_attempts=tuple(raws))


def author_plot(input_text: str, llm: Llm, *, mode: str | None = None
                ) -> "PlotSpec | AuthoringGap":
    """Doc Stage 5, LLM-assisted and hard-validated.

    ``mode`` defaults to :func:`plot_input_mode` of the request, covering the
    doc's three shapes (complete / partial / minimal). All three route through
    the SAME ``PlotSpec`` constructor: an artifact built from "make me a film"
    is held to every rule an artifact built from a finished outline is, which
    is the only reading of Stage 5 that does not quietly make the hardest case
    the laxest one.

    Returns a ``PlotSpec`` or an :class:`AuthoringGap`. Never a coerced
    artifact, and never an exception for a model problem."""
    picked = plot_input_mode(input_text) if mode is None else mode
    if picked not in INPUT_MODES:
        raise ValueError(f"mode {picked!r} is not one of {list(INPUT_MODES)}")
    prompt = build_plot_prompt(input_text, picked)

    def build(obj: Mapping[str, Any]) -> PlotSpec:
        # input_mode is PROVENANCE and belongs to the caller, not the model:
        # whatever it answered there is overwritten with what actually
        # happened. Re-running __post_init__ via replace() keeps the artifact
        # validated after the overwrite.
        return replace(PlotSpec.from_dict(obj), input_mode=picked)

    return _author("plot", prompt, llm, build)


def author_screenplay(plot: PlotSpec, llm: Llm) -> "Screenplay | AuthoringGap":
    """Doc Stage 6, LLM-assisted and hard-validated.

    The returned screenplay's ``plot_digest`` is set HERE, from the plot that
    was actually shown to the model — never read from the reply. A model that
    invents a digest is claiming provenance it cannot have, and provenance is
    the one thing in this pipeline that must not be authored."""
    if not isinstance(plot, PlotSpec):
        raise TypeError(f"author_screenplay takes a PlotSpec, got "
                        f"{type(plot).__name__}")
    prompt = build_screenplay_prompt(plot)

    def build(obj: Mapping[str, Any]) -> Screenplay:
        return replace(Screenplay.from_dict(obj), plot_digest=plot.digest)

    return _author("screenplay", prompt, llm, build)


# ---------------------------------------------------------------------------
# The live binding — the existing oracle route + dispatch, or a typed gap
# ---------------------------------------------------------------------------

#: What a live authoring call routes as. ``text.chat`` is the catalog name for
#: instruct/chat generation on this fleet (``router.CAPABILITY_TASK`` maps it
#: to the ``text-generation`` dispatch task).
AUTHORING_CAPABILITY: str = "text.chat"


def bind_llm(capability: str = AUTHORING_CAPABILITY, *,
             deadline_s: float | None = None,
             objective: str = "author a screenplay artifact",
             requested_model: str | None = None,
             ) -> "Llm | AuthoringGap":
    """A live ``(prompt) -> str`` through the EXISTING oracle route + dispatch,
    or an :class:`AuthoringGap` when this fleet cannot serve it.

    No new inference machinery, exactly like ``runtime.py``: the prompt becomes
    a ``GoalSpec``, ``router.resolve_route`` picks the model (and runs the
    authority gate), and ``runtime.execute_route`` runs it under the oracle's
    own deadline. Everything is imported INSIDE this function — building the
    model registry is a two-second import, and a typed contract has to stay
    importable from a route, a test or the agent without paying it.

    ``requested_model`` (k114's follow-up, k109's matrix landed) is passed
    straight through to ``router.resolve_route`` — the ONE place a caller can
    pin a specific model instead of the catalog's own default pick, e.g. a
    k109 routing-matrix winner. ``None`` (the default) is unchanged behaviour:
    the catalog default, exactly as before this parameter existed. A
    ``requested_model`` outside the capability's eligible set is a
    ``RouteRefusal`` inside ``resolve_route`` — caught below, same as every
    other routing fault — so an ineligible pin degrades to a typed
    ``AuthoringGap``, never a silent substitution.

    The degrade is the point. When no text model is eligible, the route comes
    back ``"gap"`` (or ``"refused"``) and this returns an ``AuthoringGap`` with
    ``code="CAPABILITY_GAP"`` carrying the catalog's own reasons — the caller
    gets the same typed shape it would get from a model that answered badly,
    and nothing in the pipeline has to distinguish "we tried and failed" from
    "we could not try". A gap is a value here, not an exception, so an
    ``author_*`` caller can write ``llm = bind_llm(); if isinstance(llm,
    AuthoringGap): return llm``."""
    try:
        from hugpy_oracle.contracts import GoalSpec
        from hugpy_oracle.router import resolve_route
        from hugpy_oracle.runtime import execute_route
    except Exception as exc:  # pragma: no cover - depends on install layout
        return AuthoringGap(
            errors=(f"the oracle runtime is not importable here: "
                    f"{type(exc).__name__}: {exc}",),
            stage="bind", code="CAPABILITY_GAP")

    probe = GoalSpec(objective=objective, raw_prompt="(probe)",
                     capability=capability)
    try:
        route = resolve_route(probe, requested_model)
    except Exception as exc:  # noqa: BLE001 — a routing fault is a GAP
        return AuthoringGap(errors=(f"{type(exc).__name__}: {exc}",),
                            stage="bind", code="CAPABILITY_GAP")
    if route.execution != "execute":
        return AuthoringGap(
            errors=((f"capability {capability!r} is not executable on this "
                     f"fleet (route: {route.execution})",) + tuple(route.reasons)),
            stage="bind", code="CAPABILITY_GAP")

    def call(prompt: str) -> str:
        goal = GoalSpec(objective=objective, raw_prompt=str(prompt),
                        capability=capability)
        decision = resolve_route(goal, requested_model)
        if decision.execution != "execute":
            raise LlmUnavailable(
                f"capability {capability!r} stopped being executable mid-run "
                f"(route: {decision.execution}): "
                f"{'; '.join(decision.reasons) or 'no reason given'}")
        artifacts, receipt = execute_route(goal, decision,
                                           deadline_s=deadline_s)
        if receipt.failure is not None:
            raise LlmUnavailable(
                f"{capability} failed ({receipt.failure.value}): "
                f"{'; '.join(receipt.log_excerpt) or 'no detail recorded'}")
        for artifact in artifacts:
            text = artifact.get("text")
            if text:
                return str(text)
        raise LlmUnavailable(
            f"{capability} returned no text artifact — the model answered "
            f"with nothing, which is not a screenplay")

    return call


__all__ = [
    "AUTHORING_CAPABILITY",
    "AVEvent",
    "AV_EVENT_KINDS",
    "AuthoringGap",
    "Beat",
    "CARRIED_KEYS",
    "CONTINUOUS_TRANSITION",
    "Character",
    "FLASHBACK_TRANSITIONS",
    "GAP_CODES",
    "INPUT_MODES",
    "LENS_FOR_SIZE",
    "Llm",
    "LlmUnavailable",
    "MODE_GUIDANCE",
    "PLOT_SYSTEM",
    "PlotRefused",
    "PlotSpec",
    "SCENE_PREFIXES",
    "SCREENPLAY_SYSTEM",
    "STATE_KEYS",
    "SUBPLAN_STEPS",
    "Scene",
    "Screenplay",
    "ScreenplayError",
    "ScreenplayRefused",
    "ShotDesign",
    "ShotPlanDraft",
    "ShotPlanRefused",
    "TRANSITIONS",
    "author_plot",
    "author_screenplay",
    "bind_llm",
    "build_continuity",
    "build_plot_prompt",
    "build_repair_prompt",
    "build_screenplay_prompt",
    "build_shot_plan",
    "chain_breaks",
    "estimated_line_seconds",
    "lock_production",
    "make_heading",
    "parse_json_object",
    "plot_input_mode",
    "props_in_play",
    "schema_block",
]
