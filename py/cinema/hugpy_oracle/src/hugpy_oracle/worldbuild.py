"""World building (k124) — the third filmmaking delta, doc Phase 1's
"rules of the world" / "production design" / "color palette" as a locked artifact.

The script-first brief asks pre-production for a *visual conception*::

    Generate: shot list, storyboard descriptions or storyboard-generation
    prompts, camera placement and movement, lens and framing guidance,
    blocking, lighting plans, COLOR PALETTE, PRODUCTION DESIGN, composition
    and motion rules.

k110 produced the shot list, the camera and the lighting per shot. Two of that
list had no artifact until now: the palette (a per-SCENE grading target, not a
per-shot one — a grade is a scene decision) and the production design (what a
location IS: how it is dressed and which props it has established). This module
is those two, plus the tone words and the lighting style that hold them
together, as ONE content-addressed artifact that locks with the production.

WHAT IS HERE

``LocationCard``
    One SET. Its name, whether it is interior, the scenes that play there, the
    times of day it is seen at, how it is dressed, and the props it has
    ESTABLISHED (the ones whose first appearance in screen order is here).
    ``established_props`` is the continuity-bearing field: a prop in play on a
    set that never established it is a set-consistency finding, not a detail.

``ScenePalette``
    One scene's color target and the BASIS it was derived from — a declared
    lighting line, the scene's time of day, a carry from the previous scene
    (``CONTINUOUS:``/"MOMENTS LATER" declare no clock of their own), an
    operator edit, or an LLM enrichment. The basis is a field because "warm
    amber falloff" derived from a slugline and the same words typed by an
    operator are different evidence, and a grading pass that cannot tell them
    apart cannot be argued with.

``WorldBuild``
    ``palette`` + ``sets`` + ``tone_words`` + ``lighting_style``, locked with
    the production. :meth:`WorldBuild.color_targets` returns exactly the
    mapping k108's ``PostProductionPlan.color_targets`` takes, keyed by scene,
    so post-production consumes the world rather than re-deriving a second
    opinion about it from the shot plan's prose.

``build_world``
    Deterministic derivation from a k110 ``Screenplay`` (+ optional
    ``ShotPlanDraft`` and ``PlotSpec``). No model, ever, for the facts: which
    sets exist, which props they establish and what time of day a scene plays
    at are FUNCTIONS of the screenplay, and asking a model to restate them is
    asking it to disagree with the script (k110's rule for ``build_continuity``,
    kept).

``enrich_world``
    The OPTIONAL model pass, hard-validated: it may widen ``dressing``,
    ``tone_words``, ``lighting_style`` and a palette target for a scene that
    declared no lighting of its own. It may NOT invent a location, a scene, or
    an ``established_props`` entry — those are continuity facts and a model
    that adds one is fabricating evidence for the judge. Failures land as
    k110's typed ``AuthoringGap``, never as a half-built world.

``set_consistency``
    The continuity check this artifact makes possible: for every segment, the
    props the continuity bible puts in play against the props the location card
    says that set has established. An unestablished prop is exactly the "where
    did that come from" error a continuity bible alone cannot see, because the
    bible tracks the STORY's inventory and the card tracks the SET's.

DERIVATION IS OVER DECLARED FIELDS ONLY. Time of day, weather, lighting, props
and wardrobe are ``Scene`` FIELDS. Nothing here reads a noun out of prose; the
one place free text is consulted is ``screenplay.props_in_play``, which
recognizes items of the screenplay's own CLOSED inventory and is k110's
function, imported rather than re-implemented.

Offline by construction: stdlib + this package. The LLM is an injected
``(prompt) -> str``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from hugpy_oracle.contracts import Check, CheckKind
from hugpy_oracle.plan import FrozenParams
from hugpy_oracle.production import ContentAddressed, ContinuityBible, ProductionError
from hugpy_oracle.screenplay import (
    AuthoringGap,
    Screenplay,
    ShotPlanDraft,
    build_repair_prompt,
    parse_json_object,
    props_in_play,
)

# BORROWED, not re-copied — k108's and k110's discipline: ``production`` owns
# the exact versions of these helpers every k104 artifact validates with, and a
# private copy in yet another module is how a tree ends up with five subtly
# different "non-empty string" rules. A test asserts they are the same objects.
from hugpy_oracle.production import _require_text as require_text
from hugpy_oracle.production import _str_tuple as str_tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Closed vocabularies — named, never magic
# ---------------------------------------------------------------------------

#: Where a :class:`ScenePalette` target came from. Closed, and ORDERED by
#: authority: an operator's edit outranks an enrichment, which outranks the
#: screenplay's own declared lighting, which outranks a time-of-day derivation,
#: which outranks a carry from the previous scene.
PALETTE_BASES: tuple[str, ...] = (
    "operator", "llm", "declared_lighting", "time_of_day", "carried",
)

#: Time-of-day label -> the grading target it implies. Matched on the
#: NORMALIZED label (upper-cased, punctuation stripped), never on prose. The
#: wording of each target is deliberately a colorist's instruction rather than
#: an adjective: "cool" is a mood, "cool blue shadows, crushed blacks, low key"
#: is something a LUT can be argued with about.
TIME_PALETTE: dict[str, str] = {
    "DAY":        "neutral daylight, balanced whites, mild contrast",
    "MORNING":    "cool pale light warming toward neutral, soft contrast",
    "DAWN":       "cool pale light warming toward neutral, soft contrast",
    "SUNRISE":    "cool pale light warming toward neutral, soft contrast",
    "AFTERNOON":  "warm neutral daylight, long shadows, mild contrast",
    "DUSK":       "warm amber falloff into cool shadow, high contrast",
    "SUNSET":     "warm amber falloff into cool shadow, high contrast",
    "MAGIC HOUR": "warm amber falloff into cool shadow, high contrast",
    "TWILIGHT":   "desaturated blue-violet, lifted blacks, low contrast",
    "EVENING":    "desaturated blue-violet, lifted blacks, low contrast",
    "NIGHT":      "cool blue shadows, crushed blacks, low key",
    "LATE NIGHT": "cool blue shadows, crushed blacks, low key",
    "MIDNIGHT":   "cool blue shadows, crushed blacks, low key",
}

#: Labels that declare no clock of their own. A scene headed ``CONTINUOUS`` is
#: the same minute as the one before it, so its palette is CARRIED rather than
#: derived — and the first scene of a screenplay cannot carry from anything,
#: which is why :func:`build_world` falls back to ``DAY`` and says so.
CARRY_LABELS: frozenset[str] = frozenset({
    "CONTINUOUS", "MOMENTS LATER", "LATER", "SAME", "SAME TIME", "MOMENTS",
})

#: Whole-token weather cues -> the modifier they add to a scene's target. The
#: cue is matched against ``Scene.weather``, a DECLARED field — never against
#: the action, because a screenplay that says "his mood darkened" has not
#: declared an overcast sky.
WEATHER_MODIFIERS: dict[str, str] = {
    "rain":     "rain: specular highlights, raised blacks, cyan cast",
    "rainy":    "rain: specular highlights, raised blacks, cyan cast",
    "storm":    "storm: high contrast, desaturated midtones",
    "stormy":   "storm: high contrast, desaturated midtones",
    "snow":     "snow: cool whites, low saturation, protected highlights",
    "snowy":    "snow: cool whites, low saturation, protected highlights",
    "fog":      "fog: lifted blacks, compressed contrast, low saturation",
    "foggy":    "fog: lifted blacks, compressed contrast, low saturation",
    "mist":     "fog: lifted blacks, compressed contrast, low saturation",
    "overcast": "overcast: flat contrast, neutral-cool whites",
    "cloudy":   "overcast: flat contrast, neutral-cool whites",
    "clear":    "clear: full saturation, clean whites",
    "sunny":    "clear: full saturation, clean whites",
    "wind":     "wind: no grade change (motion cue only)",
    "windy":    "wind: no grade change (motion cue only)",
}

#: Whole-token atmosphere cues recognized in the DECLARED lighting/weather
#: fields, contributed to :attr:`WorldBuild.tone_words`. Closed on purpose: a
#: tone vocabulary that grows by typing is a vocabulary no two scenes share.
ATMOSPHERE_CUES: frozenset[str] = frozenset({
    "bleak", "bright", "clinical", "cold", "cosy", "cozy", "dim", "gloomy",
    "golden", "hard", "harsh", "intimate", "moody", "muted", "naturalistic",
    "neon", "noir", "practical", "romantic", "shadowy", "soft", "stark",
    "sterile", "sunlit", "tense", "warm",
})

#: What ``interior`` means for each slugline prefix, k110's ``SCENE_PREFIXES``
#: read the one way a location card cares about. A location seen both ways
#: records ``None`` — it is both, and pretending otherwise would dress it wrong.
_INTERIOR_BY_PREFIX: dict[str, bool | None] = {
    "INT.": True, "EXT.": False, "INT./EXT.": None, "EXT./INT.": None,
    "I/E.": None,
}

#: The one place this module falls back when a screenplay declares a time of
#: day nothing recognizes. Recorded on the palette entry as its own basis so an
#: unrecognized label never reads as a derived one.
UNKNOWN_TIME_TARGET: str = ("no palette derived: the scene's time of day is "
                            "not one this module recognizes")

#: The enrichment patch's permitted keys. Anything else in the reply is a
#: refusal with the key named, which is the difference between "the model
#: cannot invent a set" as a rule and as a hope.
ENRICHABLE_KEYS: tuple[str, ...] = (
    "tone_words", "lighting_style", "sets", "palette",
)

#: The per-set keys enrichment may touch. ``established_props`` is deliberately
#: absent: which props a set has established is a CONTINUITY fact derived from
#: screen order, and a model that adds one has invented a prop the script never
#: put there.
ENRICHABLE_SET_KEYS: tuple[str, ...] = ("dressing", "notes")

WORLD_STAGE: str = "world"


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class WorldError(ProductionError):
    """Base for every refusal in this module (a ``ValueError`` by inheritance,
    so a caller that only catches ValueError still catches these)."""

    def __init__(self, message: str, errors: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.errors = tuple(str(e) for e in errors) or (str(message),)


class WorldRefused(WorldError):
    """The world cannot be built or locked from what was supplied."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def normalize_time_of_day(label: Any) -> str:
    """The comparable form of a time-of-day label: upper-cased, punctuation and
    runs of whitespace collapsed. ``"magic hour."`` and ``"MAGIC  HOUR"`` are
    the same clock and must select the same target."""
    text = str(label or "").upper()
    kept = [c if (c.isalnum() or c.isspace() or c == "/") else " " for c in text]
    return " ".join("".join(kept).split())


def _tokens(text: Any) -> tuple[str, ...]:
    """Whole-token lowercase words. The ``shot_intent._has_cue`` idiom k108 and
    k110 both mirror: cues are recognized as WORDS, so "discolored" never reads
    as "color" and "unclear" never reads as "clear"."""
    out: list[str] = []
    word: list[str] = []
    for char in str(text or "").lower():
        if char.isalpha() or char == "'":
            word.append(char)
        else:
            if word:
                out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return tuple(out)


def interior_of(heading: Any) -> bool | None:
    """``True`` interior, ``False`` exterior, ``None`` both/unknown — read off
    the slugline prefix, which k110's ``Scene`` already guarantees is one of
    ``SCENE_PREFIXES``."""
    text = str(heading or "").strip().upper()
    for prefix in ("INT./EXT.", "EXT./INT.", "I/E.", "INT.", "EXT."):
        if text.startswith(prefix):
            return _INTERIOR_BY_PREFIX[prefix]
    return None


def weather_modifiers(weather: Any) -> tuple[str, ...]:
    """Every modifier the DECLARED weather field earns, deduped, in the order
    :data:`WEATHER_MODIFIERS` is read. Empty when the field is empty or names
    nothing recognized — never a guess."""
    found: list[str] = []
    tokens = set(_tokens(weather))
    for cue in sorted(WEATHER_MODIFIERS):
        if cue in tokens:
            modifier = WEATHER_MODIFIERS[cue]
            if modifier not in found:
                found.append(modifier)
    return tuple(found)


def atmosphere_words(*texts: Any) -> tuple[str, ...]:
    """Atmosphere cues present in the DECLARED text supplied, sorted."""
    tokens: set[str] = set()
    for text in texts:
        tokens.update(_tokens(text))
    return tuple(sorted(tokens & ATMOSPHERE_CUES))


# ---------------------------------------------------------------------------
# The artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenePalette(ContentAddressed):
    """One scene's color target, and the evidence it rests on.

    ``target`` is what a colorist (or k108's ``color_targets`` consumer) is
    told to hit. ``basis`` is WHY, from the closed :data:`PALETTE_BASES` — the
    field that keeps a derived target and a typed one distinguishable forever.
    ``modifiers`` are the weather clauses layered on top, kept separate so the
    base target stays comparable across scenes that share a clock."""

    scene_id: str
    target: str
    basis: str = "time_of_day"
    location: str = ""
    time_of_day: str = ""
    modifiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.scene_id, "ScenePalette.scene_id")
        require_text(self.target, "ScenePalette.target")
        if self.basis not in PALETTE_BASES:
            raise WorldRefused(
                f"ScenePalette({self.scene_id}).basis must be one of "
                f"{list(PALETTE_BASES)}, got {self.basis!r} — an unlabelled "
                f"target cannot be told apart from a derived one")
        for name in ("location", "time_of_day"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        object.__setattr__(self, "modifiers",
                           str_tuple(self.modifiers, "ScenePalette.modifiers"))

    @property
    def full_target(self) -> str:
        """The target with its weather modifiers, as one grading instruction —
        the string k108's ``color_targets`` mapping actually carries."""
        return "; ".join((self.target,) + self.modifiers)

    @property
    def derived(self) -> bool:
        """Whether nobody typed this: it came out of the screenplay."""
        return self.basis in ("declared_lighting", "time_of_day", "carried")

    def to_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "target": self.target,
                "basis": self.basis, "location": self.location,
                "time_of_day": self.time_of_day,
                "modifiers": list(self.modifiers)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScenePalette":
        return cls(scene_id=d["scene_id"], target=d["target"],
                   basis=d.get("basis", "time_of_day"),
                   location=d.get("location", ""),
                   time_of_day=d.get("time_of_day", ""),
                   modifiers=tuple(d.get("modifiers", ())))


@dataclass(frozen=True, slots=True)
class LocationCard(ContentAddressed):
    """One SET: the brief's "production design", per location.

    ``dressing`` is what is ON the set — every prop and wardrobe item any scene
    playing here declares or names out of the screenplay's own inventory.
    ``established_props`` is the smaller, load-bearing set: the props whose
    FIRST appearance in screen order is here. A prop that is in play at a
    location that never established it is what :func:`set_consistency` reports,
    and it is the error a continuity bible cannot see on its own — the bible
    tracks the STORY's inventory, a card tracks the SET's."""

    name: str
    interior: bool | None = None
    scene_ids: tuple[str, ...] = ()
    times_of_day: tuple[str, ...] = ()
    dressing: tuple[str, ...] = ()
    established_props: tuple[str, ...] = ()
    lighting: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        require_text(self.name, "LocationCard.name")
        if self.interior is not None and not isinstance(self.interior, bool):
            raise WorldRefused(
                f"LocationCard({self.name}).interior must be a bool or None "
                f"(None = seen both ways), got {type(self.interior).__name__}")
        for name in ("scene_ids", "times_of_day", "dressing",
                     "established_props"):
            object.__setattr__(self, name,
                               str_tuple(getattr(self, name),
                                         f"LocationCard.{name}"))
        for name in ("lighting", "notes"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        unknown = tuple(p for p in self.established_props
                        if p not in self.dressing)
        if unknown:
            raise WorldRefused(
                f"LocationCard({self.name}) establishes prop(s) "
                f"{list(unknown)} that are not in its dressing — a set cannot "
                f"establish something it is not dressed with")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "interior": self.interior,
                "scene_ids": list(self.scene_ids),
                "times_of_day": list(self.times_of_day),
                "dressing": list(self.dressing),
                "established_props": list(self.established_props),
                "lighting": self.lighting, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LocationCard":
        return cls(name=d["name"], interior=d.get("interior"),
                   scene_ids=tuple(d.get("scene_ids", ())),
                   times_of_day=tuple(d.get("times_of_day", ())),
                   dressing=tuple(d.get("dressing", ())),
                   established_props=tuple(d.get("established_props", ())),
                   lighting=d.get("lighting", ""), notes=d.get("notes", ""))


@dataclass(frozen=True, slots=True)
class WorldBuild(ContentAddressed):
    """The world of the film, as one locked artifact.

    ``palette`` is per SCENE (a grade is a scene decision, k108's rule);
    ``sets`` is per LOCATION; ``tone_words`` and ``lighting_style`` are the
    whole-film constants a segment prompt and a grading pass both read.

    ``locked`` is the Stage 11 gate, the same shape ``Screenplay.locked`` and
    ``AudioMaster.locked`` use: :func:`screenplay.lock_production` takes a
    LOCKED world, because locking a production against a palette that may still
    change is the same mistake as locking against an unlocked script."""

    palette: tuple[ScenePalette, ...] = ()
    sets: tuple[LocationCard, ...] = ()
    tone_words: tuple[str, ...] = ()
    lighting_style: str = ""
    lighting_basis: str = "derived"
    screenplay_digest: str | None = None
    shot_plan_digest: str | None = None
    notes: str = ""
    locked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "palette", tuple(self.palette))
        for entry in self.palette:
            if not isinstance(entry, ScenePalette):
                raise WorldRefused(f"WorldBuild.palette takes ScenePalette, "
                                   f"got {type(entry).__name__}")
        ids = [p.scene_id for p in self.palette]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise WorldRefused(
                f"WorldBuild.palette carries two targets for scene(s) "
                f"{duplicates} — a scene is graded one way")
        object.__setattr__(self, "sets", tuple(self.sets))
        for card in self.sets:
            if not isinstance(card, LocationCard):
                raise WorldRefused(f"WorldBuild.sets takes LocationCard, got "
                                   f"{type(card).__name__}")
        names = [c.name for c in self.sets]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise WorldRefused(
                f"WorldBuild.sets carries two cards for location(s) "
                f"{duplicates} — a set is dressed one way")
        object.__setattr__(self, "tone_words",
                           str_tuple(self.tone_words, "WorldBuild.tone_words"))
        for name in ("lighting_style", "lighting_basis", "notes"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        for name in ("screenplay_digest", "shot_plan_digest"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name,
                                   require_text(value, f"WorldBuild.{name}"))
        if not isinstance(self.locked, bool):
            raise WorldRefused("WorldBuild.locked must be a bool")
        if self.locked and not self.palette:
            raise WorldRefused(
                "cannot lock a world with no palette: a locked world with no "
                "scene in it is a lock over nothing")

    # -- reading -----------------------------------------------------------

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(p.scene_id for p in self.palette)

    @property
    def location_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.sets)

    def palette_for(self, scene_id: str) -> ScenePalette | None:
        for entry in self.palette:
            if entry.scene_id == scene_id:
                return entry
        return None

    def card_for(self, location: str) -> LocationCard | None:
        for card in self.sets:
            if card.name == location:
                return card
        return None

    @property
    def color_targets(self) -> dict[str, str]:
        """``{scene_id: grading target}`` — exactly the shape k108's
        ``PostProductionPlan.color_targets`` takes, so post-production consumes
        the world instead of forming a second opinion from the shot prose."""
        return {p.scene_id: p.full_target for p in self.palette}

    @property
    def established_props(self) -> tuple[str, ...]:
        """Every prop any set has established, sorted — the world's inventory."""
        out: set[str] = set()
        for card in self.sets:
            out.update(card.established_props)
        return tuple(sorted(out))

    def location_of(self, scene_id: str) -> str:
        entry = self.palette_for(scene_id)
        return entry.location if entry is not None else ""

    def lock(self) -> "WorldBuild":
        return self if self.locked else replace(self, locked=True)

    def to_dict(self) -> dict[str, Any]:
        return {"palette": [p.to_dict() for p in self.palette],
                "sets": [c.to_dict() for c in self.sets],
                "tone_words": list(self.tone_words),
                "lighting_style": self.lighting_style,
                "lighting_basis": self.lighting_basis,
                "screenplay_digest": self.screenplay_digest,
                "shot_plan_digest": self.shot_plan_digest,
                "notes": self.notes, "locked": self.locked}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WorldBuild":
        return cls(
            palette=tuple(ScenePalette.from_dict(p)
                          for p in d.get("palette", ())),
            sets=tuple(LocationCard.from_dict(c) for c in d.get("sets", ())),
            tone_words=tuple(d.get("tone_words", ())),
            lighting_style=d.get("lighting_style", ""),
            lighting_basis=d.get("lighting_basis", "derived"),
            screenplay_digest=d.get("screenplay_digest"),
            shot_plan_digest=d.get("shot_plan_digest"),
            notes=d.get("notes", ""),
            locked=bool(d.get("locked", False)))


# ---------------------------------------------------------------------------
# Derivation — deterministic, model-free
# ---------------------------------------------------------------------------


def build_world(screenplay: Screenplay,
                shots: ShotPlanDraft | None = None,
                *, plot: Any = None) -> WorldBuild:
    """Derive the world from the screenplay. Same script, same world, same
    digest (a test asserts it).

    Takes no ``llm`` and never will for the FACTS. Which sets exist, which
    props they establish and what clock a scene runs on are functions of the
    screenplay; a model asked to restate them can only disagree with the
    script. :func:`enrich_world` is where a model is allowed to help, and it
    is allowed to help with the things the script does not say.

    ``shots`` is optional and is used for one thing: recording the shot plan's
    digest on the world, so a world locked beside a plan can be told apart from
    one derived before it. ``plot`` (a k110 ``PlotSpec``) contributes its
    DECLARED ``tone`` and ``genre`` to :attr:`WorldBuild.tone_words`."""
    if not isinstance(screenplay, Screenplay):
        raise WorldRefused(
            f"build_world takes a Screenplay, got {type(screenplay).__name__}")
    if not screenplay.scenes:
        raise WorldRefused(
            "cannot build a world from a screenplay with no scenes — there is "
            "no set to dress and no scene to grade")
    if shots is not None and not isinstance(shots, ShotPlanDraft):
        raise WorldRefused(
            f"build_world(shots=) takes a ShotPlanDraft, got "
            f"{type(shots).__name__}")

    inventory = _screenplay_inventory(screenplay)

    palette: list[ScenePalette] = []
    previous: ScenePalette | None = None
    for scene in screenplay.scenes:
        entry = _palette_for_scene(scene, previous)
        palette.append(entry)
        previous = entry

    sets = _location_cards(screenplay, inventory)
    tone = _tone_words(screenplay, plot)
    style, basis = _lighting_style(screenplay)

    notes = (f"derived from screenplay {screenplay.title!r}: "
             f"{len(palette)} scene palette(s) over {len(sets)} set(s); "
             f"{len(inventory)} declared prop/wardrobe item(s)")
    return WorldBuild(
        palette=tuple(palette), sets=tuple(sets), tone_words=tone,
        lighting_style=style, lighting_basis=basis,
        screenplay_digest=screenplay.digest,
        shot_plan_digest=(shots.digest if shots is not None else None),
        notes=notes)


def _screenplay_inventory(screenplay: Screenplay) -> tuple[str, ...]:
    """The screenplay's own CLOSED inventory of props and wardrobe, in
    declaration order. k110's rule: recognition happens against this, never
    against nouns pulled out of prose."""
    out: list[str] = []
    for scene in screenplay.scenes:
        for item in tuple(scene.props) + tuple(scene.wardrobe):
            if item not in out:
                out.append(item)
    return tuple(out)


def _palette_for_scene(scene: Any,
                       previous: ScenePalette | None) -> ScenePalette:
    """One scene's target, by the authority order in :data:`PALETTE_BASES`.

    A DECLARED ``Scene.lighting`` wins outright: the screenplay named the look
    and inventing a second one from the clock would put two instructions on one
    scene. Otherwise the clock decides; a clock that declares nothing
    (``CONTINUOUS``) carries the previous scene's target, which is what
    ``CONTINUOUS:`` means; and a label nothing recognizes gets
    :data:`UNKNOWN_TIME_TARGET` under its own basis rather than a guess."""
    modifiers = weather_modifiers(scene.weather)
    label = normalize_time_of_day(scene.time_of_day)
    if str(scene.lighting or "").strip():
        return ScenePalette(scene_id=scene.scene_id,
                            target=str(scene.lighting).strip(),
                            basis="declared_lighting",
                            location=scene.location,
                            time_of_day=scene.time_of_day,
                            modifiers=modifiers)
    if label in TIME_PALETTE:
        return ScenePalette(scene_id=scene.scene_id, target=TIME_PALETTE[label],
                            basis="time_of_day", location=scene.location,
                            time_of_day=scene.time_of_day, modifiers=modifiers)
    if label in CARRY_LABELS and previous is not None:
        return ScenePalette(scene_id=scene.scene_id, target=previous.target,
                            basis="carried", location=scene.location,
                            time_of_day=scene.time_of_day, modifiers=modifiers)
    return ScenePalette(scene_id=scene.scene_id, target=UNKNOWN_TIME_TARGET,
                        basis="time_of_day", location=scene.location,
                        time_of_day=scene.time_of_day, modifiers=modifiers)


def _location_cards(screenplay: Screenplay,
                    inventory: Sequence[str]) -> tuple[LocationCard, ...]:
    """One card per distinct location, in first-appearance order.

    ``dressing`` accumulates every prop/wardrobe item any scene at this
    location declares or NAMES out of the closed inventory (k110's
    ``props_in_play``, imported). ``established_props`` is the subset whose
    first appearance in SCREEN order is here — computed by walking the scenes
    once, in order, and claiming each prop for the first set that shows it."""
    order: list[str] = []
    scenes_at: dict[str, list[Any]] = {}
    for scene in screenplay.scenes:
        if scene.location not in scenes_at:
            scenes_at[scene.location] = []
            order.append(scene.location)
        scenes_at[scene.location].append(scene)

    claimed: dict[str, str] = {}
    for scene in screenplay.scenes:
        for item in _scene_items(scene, inventory):
            claimed.setdefault(item, scene.location)

    cards: list[LocationCard] = []
    for location in order:
        scenes = scenes_at[location]
        interiors = {interior_of(s.heading) for s in scenes}
        interior = interiors.pop() if len(interiors) == 1 else None
        dressing: list[str] = []
        times: list[str] = []
        lighting: list[str] = []
        for scene in scenes:
            for item in _scene_items(scene, inventory):
                if item not in dressing:
                    dressing.append(item)
            label = normalize_time_of_day(scene.time_of_day)
            if label and label not in times:
                times.append(label)
            declared = str(scene.lighting or "").strip()
            if declared and declared not in lighting:
                lighting.append(declared)
        established = tuple(i for i in dressing if claimed.get(i) == location)
        cards.append(LocationCard(
            name=location, interior=interior,
            scene_ids=tuple(s.scene_id for s in scenes),
            times_of_day=tuple(times), dressing=tuple(dressing),
            established_props=established, lighting="; ".join(lighting),
            notes=(f"{len(scenes)} scene(s) play here"
                   + ("" if interior is None
                      else f"; {'interior' if interior else 'exterior'}"))))
    return tuple(cards)


def _scene_items(scene: Any, inventory: Sequence[str]) -> tuple[str, ...]:
    """Everything on the set for one scene: the props it has in play (k110's
    closed-inventory recognizer) plus the wardrobe it declares. Sorted, so a
    card is comparable without a caller remembering to sort."""
    return tuple(sorted(set(props_in_play(scene, inventory))
                        | set(scene.wardrobe)))


def _tone_words(screenplay: Screenplay, plot: Any) -> tuple[str, ...]:
    """The film's tone words: the plot's own DECLARED ``tone``/``genre``, then
    the atmosphere cues present in the DECLARED lighting and weather fields.

    Nothing is read out of action prose. A tone word that came from a verb
    somebody wrote is not a tone the production agreed on."""
    words: list[str] = []
    for field_name in ("tone", "genre", "pacing"):
        value = getattr(plot, field_name, None)
        for token in _tokens(value):
            if len(token) > 2 and token not in words:
                words.append(token)
    declared: list[Any] = []
    for scene in screenplay.scenes:
        declared.extend((scene.lighting, scene.weather))
    for token in atmosphere_words(*declared):
        if token not in words:
            words.append(token)
    return tuple(words)


def _lighting_style(screenplay: Screenplay) -> tuple[str, str]:
    """``(style, basis)``. The most-declared ``Scene.lighting`` line wins, ties
    broken by first appearance. When NO scene declares one the style is derived
    from the interior/exterior and time-of-day mix and says so — an invented
    lighting plan that read as a declared one is the failure this pair of
    return values exists to prevent."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for scene in screenplay.scenes:
        declared = str(scene.lighting or "").strip()
        if not declared:
            continue
        if declared not in counts:
            counts[declared] = 0
            order.append(declared)
        counts[declared] += 1
    if counts:
        best = max(order, key=lambda text: (counts[text], -order.index(text)))
        return (best, "declared")
    interiors = sum(1 for s in screenplay.scenes
                    if interior_of(s.heading) is True)
    exteriors = sum(1 for s in screenplay.scenes
                    if interior_of(s.heading) is False)
    nights = sum(1 for s in screenplay.scenes
                 if normalize_time_of_day(s.time_of_day) in
                 ("NIGHT", "LATE NIGHT", "MIDNIGHT", "EVENING", "TWILIGHT"))
    where = "interior" if interiors >= exteriors else "exterior"
    when = "night" if nights * 2 >= len(screenplay.scenes) else "day"
    return (f"predominantly {where} {when}: "
            + ("low-key, practical-motivated" if when == "night"
               else "naturalistic, motivated by available light"),
            "derived")


# ---------------------------------------------------------------------------
# Set consistency — the check this artifact makes possible
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SetInconsistency:
    """One segment's set disagreeing with its continuity state.

    ``unestablished`` is the finding that matters: a prop the bible puts in
    play at a location whose card never established it. ``undressed`` is the
    softer half — the set is dressed with something this scene does not use,
    which is normal and is reported only for completeness."""

    segment_id: str
    location: str
    unestablished: tuple[str, ...] = ()
    undressed: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "location": self.location,
                "unestablished": list(self.unestablished),
                "undressed": list(self.undressed), "detail": self.detail}


def set_consistency(world: WorldBuild,
                    continuity: ContinuityBible) -> tuple[SetInconsistency, ...]:
    """Every segment whose continuity state puts a prop on a set that never
    established it, in bible order.

    Reads each entry's OWN ``state_before['location']`` (k110's ``STATE_KEYS``)
    so no shot plan is needed and no second opinion about which scene a segment
    belongs to is formed here. A segment whose state declares no location, or
    a location with no card, is SKIPPED rather than reported: an unknown set
    cannot disagree with anything, and reporting it as a failure would make a
    thin bible look like a continuity error."""
    if not isinstance(world, WorldBuild):
        raise WorldRefused(f"set_consistency takes a WorldBuild, got "
                           f"{type(world).__name__}")
    if not isinstance(continuity, ContinuityBible):
        raise WorldRefused(f"set_consistency takes a ContinuityBible, got "
                           f"{type(continuity).__name__}")
    findings: list[SetInconsistency] = []
    for entry in continuity.entries:
        location = str(entry.state_before.get("location") or "")
        if not location:
            continue
        card = world.card_for(location)
        if card is None:
            continue
        in_play = _props_of(entry.state_before) | _props_of(entry.state_after)
        dressed = set(card.dressing)
        unestablished = tuple(sorted(in_play - dressed))
        undressed = tuple(sorted(dressed - in_play))
        if not unestablished:
            continue
        findings.append(SetInconsistency(
            segment_id=entry.segment_id, location=location,
            unestablished=unestablished, undressed=undressed,
            detail=(f"{entry.segment_id} plays at {location!r} with prop(s) "
                    f"{list(unestablished)} that this set is not dressed with "
                    f"(dressing: {sorted(dressed)})")))
    return tuple(findings)


def _props_of(state: Mapping[str, Any]) -> set[str]:
    """The props a continuity state names, from ``props`` only.

    ``props_seen`` is deliberately NOT read: it is k110's CUMULATIVE record of
    what the story has established anywhere, so checking it against one set's
    dressing would report every prop from every other location as an
    inconsistency here."""
    value = state.get("props")
    if isinstance(value, str):
        return {value} if value.strip() else set()
    if isinstance(value, (list, tuple)):
        return {str(v) for v in value if str(v).strip()}
    return set()


def set_consistency_check(world: WorldBuild,
                          continuity: ContinuityBible) -> Check:
    """The findings as a :class:`Check`, ready to fold into a scorecard beside
    k108's report checks. Named ``continuity.set_consistency`` so it sorts with
    the rest of the continuity evidence rather than the color evidence."""
    findings = set_consistency(world, continuity)
    checked = sum(1 for e in continuity.entries
                  if world.card_for(str(e.state_before.get("location") or ""))
                  is not None)
    if not checked:
        from hugpy_oracle import speech
        return Check(
            name="continuity.set_consistency", kind=CheckKind.SEMANTIC,
            value=None, threshold=len(continuity.entries), passed=True,
            detail=(f"{speech.UNSCORED_PREFIX}no continuity entry names a "
                    f"location this world has a set card for; set consistency "
                    f"cannot be verified"))
    return Check(
        name="continuity.set_consistency", kind=CheckKind.SEMANTIC,
        value=checked - len(findings), threshold=checked, passed=not findings,
        detail=(f"{checked - len(findings)}/{checked} segment(s) play on a set "
                f"dressed for them"
                + ("" if not findings else
                   "; " + "; ".join(f.detail for f in findings))))


# ---------------------------------------------------------------------------
# Optional LLM enrichment — hard-validated, never a half-built world
# ---------------------------------------------------------------------------


def build_enrichment_prompt(world: WorldBuild) -> str:
    """The enrichment prompt: the derived world verbatim, the permitted keys,
    and the four things the model may not do. Deterministic — a prompt that
    changed every time it was built could not be cached or diffed."""
    payload = json.dumps(world.to_dict(), sort_keys=True, indent=2)
    return "\n".join([
        "You are a production designer enriching an ALREADY DERIVED world "
        "bible for a film. The facts below came from the locked screenplay "
        "and are not yours to change.",
        "",
        "THE DERIVED WORLD:",
        payload,
        "",
        "Reply with ONE JSON object — a PATCH, not a replacement — using only "
        f"these keys: {list(ENRICHABLE_KEYS)}.",
        "",
        f'  "tone_words": [...]        extra tone words (strings)',
        f'  "lighting_style": "..."    one replacement lighting style line',
        f'  "sets": {{"<location name>": {{"dressing": [...], "notes": "..."}}}}',
        f'  "palette": {{"<scene id>": "<grading target>"}}',
        "",
        "RULES — each of these is checked and a violation is refused:",
        "1. Every location name in `sets` must already exist in `sets` above. "
        "You may not invent a set.",
        "2. Every scene id in `palette` must already exist in `palette` above. "
        "You may not invent a scene.",
        "3. You may NOT set `established_props` on any set. Which props a set "
        "established is a continuity fact derived from screen order.",
        "4. You may not re-grade a scene whose basis is `declared_lighting` — "
        "the screenplay already named that look.",
        "5. Every string must be non-empty.",
        "",
        "Return the JSON object and nothing else.",
    ])


def apply_enrichment(world: WorldBuild,
                     patch: Mapping[str, Any],
                     *, basis: str = "llm") -> tuple[WorldBuild | None,
                                                     tuple[str, ...]]:
    """``(enriched_world, ())`` or ``(None, errors)``.

    EVERY problem is reported at once — k110's rule, and what makes ONE bounded
    reprompt sufficient instead of a slow ladder. Also the validator an
    operator's hand-typed patch goes through, which is the only reading of
    "allow the user to review and edit" that does not make the edit path a way
    around the rules."""
    if basis not in ("llm", "operator"):
        raise WorldRefused(f"apply_enrichment(basis=) must be 'llm' or "
                           f"'operator', got {basis!r}")
    if not isinstance(patch, Mapping):
        return None, (f"the enrichment must be a JSON object, got "
                      f"{type(patch).__name__}",)

    errors: list[str] = []
    unknown = sorted(set(patch) - set(ENRICHABLE_KEYS))
    if unknown:
        errors.append(f"key(s) {unknown} are not enrichable; only "
                      f"{list(ENRICHABLE_KEYS)} may be patched")

    tone = list(world.tone_words)
    raw_tone = patch.get("tone_words", ())
    if raw_tone:
        if not isinstance(raw_tone, (list, tuple)):
            errors.append("tone_words must be a list of strings")
        else:
            for word in raw_tone:
                text = str(word or "").strip()
                if not text:
                    errors.append("tone_words carries an empty string")
                elif text not in tone:
                    tone.append(text)

    style = world.lighting_style
    style_basis = world.lighting_basis
    if "lighting_style" in patch:
        text = str(patch.get("lighting_style") or "").strip()
        if not text:
            errors.append("lighting_style must be a non-empty string")
        else:
            style, style_basis = text, basis

    cards = list(world.sets)
    raw_sets = patch.get("sets") or {}
    if raw_sets and not isinstance(raw_sets, Mapping):
        errors.append("sets must be a JSON object keyed by location name")
    elif raw_sets:
        for name in sorted(raw_sets):
            card = world.card_for(name)
            if card is None:
                errors.append(
                    f"sets[{name!r}] names a location this world does not "
                    f"have (has: {list(world.location_names)}) — a set cannot "
                    f"be invented here")
                continue
            body = raw_sets[name]
            if not isinstance(body, Mapping):
                errors.append(f"sets[{name!r}] must be a JSON object")
                continue
            extra = sorted(set(body) - set(ENRICHABLE_SET_KEYS))
            if extra:
                errors.append(
                    f"sets[{name!r}] key(s) {extra} may not be enriched; only "
                    f"{list(ENRICHABLE_SET_KEYS)} may be")
                continue
            dressing = list(card.dressing)
            for item in body.get("dressing", ()) or ():
                text = str(item or "").strip()
                if not text:
                    errors.append(f"sets[{name!r}].dressing carries an empty "
                                  f"string")
                elif text not in dressing:
                    dressing.append(text)
            notes = str(body.get("notes") or card.notes)
            cards[cards.index(card)] = replace(
                card, dressing=tuple(dressing), notes=notes)

    palette = list(world.palette)
    raw_palette = patch.get("palette") or {}
    if raw_palette and not isinstance(raw_palette, Mapping):
        errors.append("palette must be a JSON object keyed by scene id")
    elif raw_palette:
        for scene_id in sorted(raw_palette):
            entry = world.palette_for(scene_id)
            if entry is None:
                errors.append(
                    f"palette[{scene_id!r}] names a scene this world does not "
                    f"have (has: {list(world.scene_ids)}) — a scene cannot be "
                    f"invented here")
                continue
            if entry.basis == "declared_lighting":
                errors.append(
                    f"palette[{scene_id!r}] re-grades a scene whose look the "
                    f"screenplay DECLARED ({entry.target!r}); the script wins")
                continue
            text = str(raw_palette[scene_id] or "").strip()
            if not text:
                errors.append(f"palette[{scene_id!r}] is empty")
                continue
            palette[palette.index(entry)] = replace(entry, target=text,
                                                    basis=basis)

    if errors:
        return None, tuple(errors)
    return (replace(world, palette=tuple(palette), sets=tuple(cards),
                    tone_words=tuple(tone), lighting_style=style,
                    lighting_basis=style_basis), ())


def enrich_world(world: WorldBuild, llm: Any) -> "WorldBuild | AuthoringGap":
    """Enrich the derived world with a model, or return k110's typed
    :class:`AuthoringGap`. There is no third branch: a test asserts the return
    is either a ``WorldBuild`` or a gap, never a coerced world.

    Parse -> validate -> ONE bounded reprompt carrying the validator errors
    VERBATIM -> still bad -> gap. k110's ``build_repair_prompt`` is imported so
    the repair wording is the tree's one wording."""
    if not isinstance(world, WorldBuild):
        raise WorldRefused(f"enrich_world takes a WorldBuild, got "
                           f"{type(world).__name__}")
    prompt = build_enrichment_prompt(world)
    raws: list[str] = []
    errors: tuple[str, ...] = ()
    code = "AUTHORING_INVALID"
    for attempt in (0, 1):
        try:
            reply = str(llm(prompt))
        except Exception as exc:                   # noqa: BLE001
            return AuthoringGap(
                errors=(f"the world enrichment model failed: "
                        f"{type(exc).__name__}: {exc}",),
                raw="\n\n".join(raws), stage=WORLD_STAGE, code="LLM_ERROR",
                attempts=attempt + 1, raw_attempts=tuple(raws))
        raws.append(reply)
        parsed, why = parse_json_object(reply)
        if parsed is None:
            errors, code = (why,), "AUTHORING_UNPARSED"
        else:
            enriched, errors = apply_enrichment(world, parsed)
            if enriched is not None:
                return enriched
            code = "AUTHORING_INVALID"
        if attempt == 0:
            prompt = build_repair_prompt(prompt, reply, errors)
    return AuthoringGap(errors=errors, raw=raws[-1] if raws else "",
                        stage=WORLD_STAGE, code=code, attempts=len(raws),
                        raw_attempts=tuple(raws))


__all__ = [
    "ATMOSPHERE_CUES",
    "CARRY_LABELS",
    "ENRICHABLE_KEYS",
    "ENRICHABLE_SET_KEYS",
    "PALETTE_BASES",
    "TIME_PALETTE",
    "UNKNOWN_TIME_TARGET",
    "WEATHER_MODIFIERS",
    "WORLD_STAGE",
    "LocationCard",
    "ScenePalette",
    "SetInconsistency",
    "WorldBuild",
    "WorldError",
    "WorldRefused",
    "apply_enrichment",
    "atmosphere_words",
    "build_enrichment_prompt",
    "build_world",
    "enrich_world",
    "interior_of",
    "normalize_time_of_day",
    "set_consistency",
    "set_consistency_check",
    "weather_modifiers",
]
