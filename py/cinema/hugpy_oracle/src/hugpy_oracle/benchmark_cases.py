"""k109 case suites — the FIXED inputs the model evaluation runs on.

Data only: no dispatch, no clock, no disk, no registry. Importing this module
builds three suites and one fixture screenplay and touches nothing else, which
is what lets a test, a runner and a report all read the SAME cases.

THREE TRACKS, from the doc's "Full-scale model evaluation":

  A  screenplay completion   build a coherent screenplay from partial,
                             disconnected or incomplete material
  B  plot construction       the doc's SIX input conditions, verbatim
  C  filmmaking workflow     screenplay -> actionable production artifacts
  D  spatial adherence       (k119) a locked scene manifest + a rendered
                             observation set -> the right repair code; cases
                             are SYNTHETIC geometry with one injected fault,
                             scored by ``spatial_eval`` — no model, no GPU

WHY A CASE CARRIES ITS OWN CHECKLIST. Every case names the properties its
output must have (:class:`Expectation`), and each expectation says which LAYER
owns it: ``deterministic`` (a validator or a countable check answers it) or
``judge`` (only a rubric can). That split is the whole scoring design written
down where the case is defined, so a reader can see — before any model runs —
exactly which claims about a model are measured and which are opinion.

WHY THE SUPPLIED MATERIAL IS LISTED SEPARATELY. ``supplied_lines`` and
``supplied_beats`` are the phrases the operator actually handed the model. The
doc asks for "preservation of supplied material" as a measure; a benchmark can
only measure it if it knows, per case, what was supplied — so it is DATA on the
case rather than something a scorer re-derives from the prompt by guessing.

WHY CONTRADICTIONS ARE PAIRS. A contradiction is not a word, it is two claims
that cannot both stand ("the radio is smashed" / "she raises the radio and
calls the coastguard"). Each :class:`Contradiction` names both sides, and the
scorer's soft check fires only when BOTH sides survive into the same scene —
documented as a heuristic, next to the HARD contradictions the k110 validators
find (a speaker who is not in the room, story time running backwards, a
continuity chain break), which are not heuristics at all.

The Track C fixture is a real, VALIDATED :class:`~.screenplay.Screenplay`. That
matters: the ground truth for continuity extraction and the shot list is then
``build_continuity`` / ``build_shot_plan`` of the same object — a DERIVED
answer key, not a hand-typed one that can drift from the artifact it describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hugpy_oracle.audio_master import Line
from hugpy_oracle.screenplay import AVEvent, Scene, Screenplay

#: The three tracks, as a closed vocabulary.
TRACKS: tuple[str, ...] = ("A", "B", "C", "D")

#: Which layer owns an expectation.
LAYERS: tuple[str, ...] = ("deterministic", "judge")

#: The constraint kinds a case may declare. Closed, because a constraint whose
#: kind the scorer does not recognize would be silently "adhered to" — the
#: exact failure mode a constraint-adherence measure exists to catch.
CONSTRAINT_KINDS: tuple[str, ...] = (
    "max_scenes", "min_scenes", "max_locations", "min_beats", "max_beats",
    "max_characters", "requires_location", "requires_character",
    "requires_transition", "requires_time_of_day", "forbidden_term",
    "requires_term",
)


# ---------------------------------------------------------------------------
# The case shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Expectation:
    """One property the output must have, and who can answer it."""
    key: str
    description: str
    layer: str = "deterministic"

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Expectation.key must be non-empty")
        if self.layer not in LAYERS:
            raise ValueError(f"Expectation.layer {self.layer!r} is not one of "
                             f"{list(LAYERS)}")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "description": self.description,
                "layer": self.layer}


@dataclass(frozen=True, slots=True)
class Constraint:
    """One checkable instruction the operator attached to the request.

    ``value`` is an int for the counting kinds and a string for the rest; the
    scorer dispatches on ``kind`` and never on the value's type."""
    kind: str
    value: Any
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(f"Constraint.kind {self.kind!r} is not one of "
                             f"{list(CONSTRAINT_KINDS)}")

    @property
    def key(self) -> str:
        return f"constraint:{self.kind}={self.value}"

    def sentence(self) -> str:
        """The constraint as the instruction the prompt actually carries."""
        if self.description:
            return self.description
        readable = {
            "max_scenes": f"use at most {self.value} scenes",
            "min_scenes": f"use at least {self.value} scenes",
            "max_locations": f"use at most {self.value} distinct locations",
            "min_beats": f"write at least {self.value} beats",
            "max_beats": f"write at most {self.value} beats",
            "max_characters": f"use at most {self.value} characters",
            "requires_location": f"the story must visit {self.value!r}",
            "requires_character": f"{self.value!r} must appear",
            "requires_transition": f"use the transition {self.value!r}",
            "requires_time_of_day": f"at least one scene is at {self.value!r}",
            "forbidden_term": f"never mention {self.value!r}",
            "requires_term": f"mention {self.value!r}",
        }
        return readable[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value,
                "description": self.sentence()}


@dataclass(frozen=True, slots=True)
class Contradiction:
    """Two claims from the supplied material that cannot both stand."""
    key: str
    left: tuple[str, ...]
    right: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", tuple(str(s) for s in self.left))
        object.__setattr__(self, "right", tuple(str(s) for s in self.right))
        if not self.left or not self.right:
            raise ValueError(f"Contradiction({self.key}) needs both sides — a "
                             f"one-sided contradiction is just a phrase")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "left": list(self.left),
                "right": list(self.right), "description": self.description}


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One routing-matrix key: an artifact operation a model can be best at.

    ``container``/``row_fields``/``top_fields`` describe the JSON shape a Track
    C answer must have; Track A and B answers are k110 ARTIFACTS instead and
    leave those empty (their shape is the dataclass, and the dataclass is the
    validator). ``coverage`` names which id set the answer has to cover, which
    is what makes completeness a measurement rather than an impression."""
    operation: str
    track: str
    artifact: str
    instruction: str = ""
    container: str = ""
    top_fields: tuple[str, ...] = ()
    row_fields: tuple[str, ...] = ()
    id_field: str = ""
    coverage: str = ""          # "" | scene_ids | segment_ids | line_ids
    example: str = ""

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise ValueError(f"OperationSpec.track {self.track!r} is not one "
                             f"of {list(TRACKS)}")

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "track": self.track,
                "artifact": self.artifact, "container": self.container,
                "top_fields": list(self.top_fields),
                "row_fields": list(self.row_fields),
                "id_field": self.id_field, "coverage": self.coverage}


@dataclass(frozen=True, slots=True)
class BenchCase:
    """One benchmark case: an input, an operation, and a checklist."""
    case_id: str
    track: str
    operation: str
    condition: str
    input_text: str
    expectations: tuple[Expectation, ...] = ()
    supplied_lines: tuple[str, ...] = ()
    supplied_beats: tuple[str, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    plot_mode: str | None = None
    uses_fixture: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise ValueError(f"BenchCase.track {self.track!r} is not one of "
                             f"{list(TRACKS)}")
        if not self.case_id.strip():
            raise ValueError("BenchCase.case_id must be non-empty")
        for name in ("expectations", "supplied_lines", "supplied_beats",
                     "constraints", "contradictions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.expectations:
            raise ValueError(f"BenchCase({self.case_id}) has no expectations — "
                             f"a case with no checklist scores nothing")

    @property
    def spec(self) -> OperationSpec:
        return OPERATIONS[self.operation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "track": self.track,
            "operation": self.operation, "condition": self.condition,
            "input_text": self.input_text,
            "expectations": [e.to_dict() for e in self.expectations],
            "supplied_lines": list(self.supplied_lines),
            "supplied_beats": list(self.supplied_beats),
            "constraints": [c.to_dict() for c in self.constraints],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "plot_mode": self.plot_mode, "uses_fixture": self.uses_fixture,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# The Track C fixture — a real, validated screenplay
# ---------------------------------------------------------------------------


def _fixture_screenplay() -> Screenplay:
    """The one screenplay every Track C case is asked about.

    Small on purpose (3 scenes, 4 lines): Track C measures whether a model can
    TRANSLATE a script into production artifacts, and a 40-scene fixture would
    measure context length instead."""
    s1 = Scene(
        scene_id="s1",
        heading="INT. LIGHTHOUSE KEEPER'S ROOM - NIGHT",
        location="LIGHTHOUSE KEEPER'S ROOM", time_of_day="NIGHT",
        action=("MARA hunches over the logbook. The lamp above her gutters and "
                "steadies. Outside, the sea works at the rocks."),
        staging="MARA at the desk, back to the window; camera left of the desk.",
        present_at_open=("MARA",),
        dialogue=(Line(line_id="l1", speaker="MARA",
                       text="Third night the light has failed. Someone is "
                            "cutting the cable."),),
        av_events=(AVEvent(kind="ambience", description="storm surf, constant",
                           cue="under the whole scene"),),
        props=("logbook", "lamp"), wardrobe=("oilskin coat",),
        weather="storm", lighting="single failing lamp",
        transition="CUT TO:", story_time_s=0.0, beat_id="b1")
    s2 = Scene(
        scene_id="s2",
        heading="EXT. CLIFF PATH - NIGHT",
        location="CLIFF PATH", time_of_day="NIGHT",
        action=("DEV works along the cable run with a torch. MARA comes up "
                "the path behind him."),
        staging="DEV downhill of MARA; the drop is camera right.",
        present_at_open=("DEV",), entrances=("MARA",),
        dialogue=(Line(line_id="l2", speaker="DEV",
                       text="It is cut clean. That is a blade, not weather."),
                  Line(line_id="l3", speaker="MARA",
                       text="Then whoever cut it is still on this rock.")),
        av_events=(AVEvent(kind="sfx", description="torch click",
                           cue="on DEV's first move"),),
        props=("torch", "cable"), wardrobe=("oilskin coat",),
        weather="storm", lighting="torchlight only",
        transition="CUT TO:", story_time_s=600.0, beat_id="b2")
    s3 = Scene(
        scene_id="s3",
        heading="INT. LIGHTHOUSE KEEPER'S ROOM - DAWN",
        location="LIGHTHOUSE KEEPER'S ROOM", time_of_day="DAWN",
        action=("The lamp burns steady. MARA closes the logbook. DEV sets the "
                "spliced cable on the desk."),
        staging="Both at the desk; the window is behind them, now grey.",
        present_at_open=("MARA", "DEV"),
        dialogue=(Line(line_id="l4", speaker="MARA",
                       text="We log it, and we keep the light. That is the "
                            "whole of the job."),),
        av_events=(AVEvent(kind="music", description="low resolving cue",
                           cue="on the last line"),),
        props=("logbook", "cable"), wardrobe=("oilskin coat",),
        weather="clearing", lighting="dawn through the window",
        transition="FADE OUT.", story_time_s=25200.0, beat_id="b3")
    return Screenplay(
        title="KEEP THE LIGHT",
        logline=("Two keepers find their cable cut and choose the work over "
                 "the hunt."),
        scenes=(s1, s2, s3), characters=("MARA", "DEV"))


#: The Track C source artifact. Built once at import; frozen and digestible.
FIXTURE_SCREENPLAY: Screenplay = _fixture_screenplay()


# ---------------------------------------------------------------------------
# The operations — the routing matrix's keys
# ---------------------------------------------------------------------------

OPERATIONS: dict[str, OperationSpec] = {
    "screenplay.complete": OperationSpec(
        operation="screenplay.complete", track="A",
        artifact="Screenplay (k110)",
        instruction=("Complete the supplied material into ONE ordered "
                     "screenplay covering the whole film."),
        coverage=""),
    "plot.construct": OperationSpec(
        operation="plot.construct", track="B",
        artifact="PlotSpec (k110)",
        instruction="Construct a filmable plot from the source material.",
        coverage=""),
    "breakdown.script": OperationSpec(
        operation="breakdown.script", track="C",
        artifact="script breakdown",
        instruction=("Break the screenplay down for production: one row per "
                     "scene with everything the floor has to bring."),
        container="scenes",
        row_fields=("scene_id", "interior_exterior", "location",
                    "time_of_day", "cast", "props", "wardrobe", "sound"),
        id_field="scene_id", coverage="scene_ids",
        example=('{"scenes": [{"scene_id": "s1", "interior_exterior": "INT", '
                 '"location": "...", "time_of_day": "NIGHT", "cast": ["..."], '
                 '"props": ["..."], "wardrobe": ["..."], '
                 '"sound": ["..."]}]}')),
    "continuity.extract": OperationSpec(
        operation="continuity.extract", track="C",
        artifact="continuity state per segment",
        instruction=("Read the continuity state off the screenplay: for every "
                     "scene, who is present and where it is, BEFORE it starts "
                     "and AFTER it ends."),
        container="segments",
        row_fields=("segment_id", "state_before", "state_after"),
        id_field="segment_id", coverage="scene_ids",
        example=('{"segments": [{"segment_id": "s1", "state_before": '
                 '{"location": "...", "time_of_day": "NIGHT", '
                 '"present": ["..."]}, "state_after": {"location": "...", '
                 '"time_of_day": "NIGHT", "present": ["..."]}}]}')),
    "shotlist.build": OperationSpec(
        operation="shotlist.build", track="C",
        artifact="shot list",
        instruction=("Turn the screenplay into a shot list: at least one shot "
                     "per scene, one per line of dialogue."),
        container="shots",
        row_fields=("segment_id", "scene_id", "shot_size", "camera_move",
                    "line_ids", "description"),
        id_field="segment_id", coverage="shot_ids",
        example=('{"shots": [{"segment_id": "s1-1", "scene_id": "s1", '
                 '"shot_size": "medium", "camera_move": "static", '
                 '"line_ids": ["l1"], "description": "..."}]}')),
    "storyboard.prompts": OperationSpec(
        operation="storyboard.prompts", track="C",
        artifact="storyboard image prompts",
        instruction=("Write one storyboard image prompt per shot, describing "
                     "the frame — not the story."),
        container="frames",
        row_fields=("segment_id", "prompt", "negative_prompt"),
        id_field="segment_id", coverage="shot_ids",
        example=('{"frames": [{"segment_id": "s1-1", "prompt": "...", '
                 '"negative_prompt": "..."}]}')),
    "segment.prompts": OperationSpec(
        operation="segment.prompts", track="C",
        artifact="production-ready segment prompts",
        instruction=("Compile the production prompt for every segment. Each "
                     "prompt is compiled from the SCREENPLAY, independently — "
                     "a segment must never describe another segment's "
                     "location, and must never refer to a previous clip."),
        container="segments",
        row_fields=("segment_id", "prompt", "duration_s"),
        id_field="segment_id", coverage="shot_ids",
        example=('{"segments": [{"segment_id": "s1-1", "prompt": "...", '
                 '"duration_s": 4.5}]}')),
    "assembly.plan": OperationSpec(
        operation="assembly.plan", track="C",
        artifact="assembly plan",
        instruction=("Lay the segments end to end into an assembly plan: "
                     "windows in order, no gaps, no overlaps, plus the "
                     "transition out of each one."),
        container="timeline",
        top_fields=("timeline",),
        row_fields=("segment_id", "start_s", "end_s", "transition"),
        id_field="segment_id", coverage="shot_ids",
        example=('{"timeline": [{"segment_id": "s1-1", "start_s": 0.0, '
                 '"end_s": 4.5, "transition": "CUT TO:"}]}')),
    "spatial.evaluate": OperationSpec(
        operation="spatial.evaluate", track="D",
        artifact="SpatialEvalReport (k119)",
        instruction=("Measure a rendered observation set against its locked "
                     "SpatialSceneManifest and emit the repair code that "
                     "names what to regenerate — or none."),
        coverage=""),
}


# ---------------------------------------------------------------------------
# Track A — screenplay completion (6 conditions)
# ---------------------------------------------------------------------------

_A_COMMON = (
    Expectation("validates", "the reply builds a valid k110 Screenplay "
                             "(headings, cast, presence, monotonic time)"),
    Expectation("preserves_supplied",
                "every supplied line of dialogue survives into the output"),
    Expectation("no_hard_contradictions",
                "no speaker outside the room, no story time running backwards"),
    Expectation("complete_artifact",
                "title, scenes, cast and per-scene action are all present"),
    Expectation("narrative_coherence",
                "the completed screenplay reads as one story, not a stapled "
                "set of fragments", layer="judge"),
    Expectation("transition_quality",
                "the joins between supplied and generated material are "
                "motivated, not abrupt", layer="judge"),
)

TRACK_A: tuple[BenchCase, ...] = (
    BenchCase(
        case_id="A1-partial",
        track="A", operation="screenplay.complete",
        condition="partial screenplay (a real opening, no ending)",
        input_text=(
            "PARTIAL SCREENPLAY — the first two scenes exist. The film has no "
            "ending yet.\n\n"
            "INT. NIGHT BUS - NIGHT\n"
            "RUTH sits with a cake box on her knees. The driver watches her in "
            "the mirror.\n"
            "RUTH\n"
            "Just to the depot. I know the way back.\n\n"
            "EXT. DEPOT FORECOURT - NIGHT\n"
            "The bus pulls out without her. RUTH holds the box level.\n"
            "RUTH\n"
            "It only has to last until morning.\n"),
        supplied_lines=("Just to the depot. I know the way back.",
                        "It only has to last until morning."),
        supplied_beats=("night bus", "depot forecourt", "cake box"),
        expectations=_A_COMMON + (
            Expectation("causal_completion",
                        "the added scenes follow causally from the supplied "
                        "ones", layer="judge"),),
        notes="Baseline: material is coherent, only unfinished."),
    BenchCase(
        case_id="A2-disconnected",
        track="A", operation="screenplay.complete",
        condition="disconnected segments (three fragments, no connective tissue)",
        input_text=(
            "THREE DISCONNECTED FRAGMENTS. They belong to one film. Nothing "
            "joins them yet.\n\n"
            "FRAGMENT 1\n"
            "INT. SORTING OFFICE - DAWN\n"
            "OKON stamps the last sack and sets it aside.\n"
            "OKON\n"
            "That one does not go out today.\n\n"
            "FRAGMENT 2\n"
            "EXT. RIVER STEPS - DAY\n"
            "A woman, PRIYA, counts stamps into a tin.\n"
            "PRIYA\n"
            "Forty. He said there would be sixty.\n\n"
            "FRAGMENT 3\n"
            "INT. SORTING OFFICE - NIGHT\n"
            "The sack is gone. OKON stands where it was.\n"
            "OKON\n"
            "I only had to not look at it.\n"),
        supplied_lines=("That one does not go out today.",
                        "Forty. He said there would be sixty.",
                        "I only had to not look at it."),
        supplied_beats=("sorting office", "river steps", "the sack"),
        expectations=_A_COMMON + (
            Expectation("connects_fragments",
                        "the three fragments end up causally connected rather "
                        "than merely adjacent", layer="judge"),),
        notes="The hard case for causal reasoning: order is not given."),
    BenchCase(
        case_id="A3-missing-middle",
        track="A", operation="screenplay.complete",
        condition="missing middle (opening and ending supplied, middle absent)",
        input_text=(
            "OPENING AND ENDING ARE FIXED. The middle of the film is missing "
            "and must be written. Do not change the two supplied scenes.\n\n"
            "OPENING\n"
            "INT. WORKSHOP - DAY\n"
            "TOMAS unwraps a violin with a split back.\n"
            "TOMAS\n"
            "Six weeks. Maybe seven.\n\n"
            "ENDING\n"
            "EXT. CONCERT HALL STEPS - NIGHT\n"
            "TOMAS listens from outside as the violin is played within.\n"
            "TOMAS\n"
            "Seven, then.\n"),
        supplied_lines=("Six weeks. Maybe seven.", "Seven, then."),
        supplied_beats=("workshop", "split back", "concert hall steps"),
        expectations=_A_COMMON + (
            Expectation("bridges_middle",
                        "the written middle earns the supplied ending",
                        layer="judge"),),
        notes="Tests interpolation, not extrapolation."),
    BenchCase(
        case_id="A4-constrained",
        track="A", operation="screenplay.complete",
        condition="constraint-laden (hard production limits attached)",
        input_text=(
            "FRAGMENT\n"
            "INT. SERVER ROOM - NIGHT\n"
            "ADA watches a rack light go amber, then red.\n"
            "ADA\n"
            "That is not a fan. That is somebody in the building.\n"),
        supplied_lines=("That is not a fan. That is somebody in the building.",),
        supplied_beats=("server room", "rack light"),
        constraints=(
            Constraint("max_scenes", 4,
                       "the finished screenplay uses AT MOST 4 scenes"),
            Constraint("max_characters", 3,
                       "at most 3 characters appear anywhere"),
            Constraint("requires_location", "SERVER ROOM",
                       "the SERVER ROOM must appear in the film"),
            Constraint("requires_transition", "FADE OUT.",
                       "the last scene ends on 'FADE OUT.'"),
            Constraint("forbidden_term", "gun",
                       "no firearm appears or is mentioned"),
            Constraint("forbidden_term", "blood",
                       "nothing bloody appears or is mentioned"),
        ),
        expectations=_A_COMMON + (
            Expectation("constraints_met",
                        "every declared production constraint is honoured"),),
        notes="Constraint adherence is the measured axis here."),
    BenchCase(
        case_id="A5-dialogue-heavy",
        track="A", operation="screenplay.complete",
        condition="dialogue-heavy (a long exchange supplied verbatim)",
        input_text=(
            "SUPPLIED SCENE — the dialogue is FINAL and must appear verbatim, "
            "in this order, attributed to these speakers. Build the rest of "
            "the film around it.\n\n"
            "INT. KITCHEN - EVENING\n"
            "JUNE\n"
            "You kept the letter.\n"
            "HAL\n"
            "I kept all of them. That is not the same as reading them.\n"
            "JUNE\n"
            "Read this one out loud.\n"
            "HAL\n"
            "No. You know what it says.\n"
            "JUNE\n"
            "Then say it in your own words, and I will go.\n"
            "HAL\n"
            "In my own words it is worse.\n"),
        supplied_lines=("You kept the letter.",
                        "I kept all of them. That is not the same as reading "
                        "them.",
                        "Read this one out loud.",
                        "No. You know what it says.",
                        "Then say it in your own words, and I will go.",
                        "In my own words it is worse."),
        supplied_beats=("kitchen", "the letter"),
        expectations=_A_COMMON + (
            Expectation("dialogue_consistency",
                        "the added dialogue sounds like the same two people",
                        layer="judge"),),
        notes="Preservation is the measured axis: six lines, verbatim."),
    BenchCase(
        case_id="A6-contradiction",
        track="A", operation="screenplay.complete",
        condition="contradiction-bearing (the fragments disagree)",
        input_text=(
            "FRAGMENTS FROM TWO DRAFTS. They contradict each other. Resolve "
            "the contradictions in the finished screenplay — do not carry both "
            "versions.\n\n"
            "DRAFT A\n"
            "INT. RADIO HUT - NIGHT\n"
            "The radio is smashed on the floor. NELL turns it over with her "
            "boot.\n"
            "NELL\n"
            "Nothing is coming out of that again.\n\n"
            "DRAFT B\n"
            "INT. RADIO HUT - NIGHT\n"
            "NELL raises the radio and calls the coastguard.\n"
            "NELL\n"
            "Station four, we have one man down on the north face.\n\n"
            "DRAFT A\n"
            "EXT. NORTH FACE - NIGHT\n"
            "IVO is dead in the snow. NELL covers his face.\n\n"
            "DRAFT B\n"
            "EXT. NORTH FACE - NIGHT\n"
            "IVO\n"
            "Leave the pack. Just walk.\n"),
        supplied_lines=("Nothing is coming out of that again.",),
        supplied_beats=("radio hut", "north face"),
        contradictions=(
            Contradiction(
                key="radio", left=("smashed", "nothing is coming out"),
                right=("calls the coastguard", "raises the radio",
                       "station four"),
                description="the radio is destroyed AND used to call for help"),
            Contradiction(
                key="ivo", left=("ivo is dead", "covers his face"),
                right=("ivo\nleave the pack", "leave the pack"),
                description="IVO is dead AND speaking"),
        ),
        expectations=_A_COMMON + (
            Expectation("contradictions_resolved",
                        "the output commits to ONE version of each "
                        "contradicted fact"),
            Expectation("resolution_is_motivated",
                        "the chosen resolution is dramatized, not silently "
                        "dropped", layer="judge"),),
        notes="Contradiction rate is the measured axis here."),
)


# ---------------------------------------------------------------------------
# Track B — plot construction (the doc's six input conditions, in order)
# ---------------------------------------------------------------------------

_B_COMMON = (
    Expectation("validates", "the reply builds a valid k110 PlotSpec "
                             "(no orphan characters, no dangling causes)"),
    Expectation("complete_artifact",
                "premise, genre, tone, characters, beats and an ending are "
                "all present"),
    Expectation("causal_logic",
                "every beat after the first names a cause that exists and "
                "comes earlier"),
    Expectation("no_hard_contradictions",
                "no beat is caused by itself or by a later beat"),
    Expectation("character_motivation",
                "each character's goal, conflict and arc are specific to this "
                "story", layer="judge"),
    Expectation("screenplay_suitability",
                "the plot could be dramatized as scenes without further "
                "invention", layer="judge"),
)

TRACK_B: tuple[BenchCase, ...] = (
    BenchCase(
        case_id="B1-detailed-premise",
        track="B", operation="plot.construct",
        condition="a detailed partial premise",
        input_text=(
            "A municipal archivist in a flooded coastal town is told to "
            "destroy forty years of parish records before the building is "
            "condemned. She begins photographing them at night. Her brother, "
            "who signed the condemnation order, notices the power draw. The "
            "town's claim to the harbour rests on a document in the third "
            "box. She has three nights and no legal standing. The ending "
            "should turn on what she chooses to photograph LAST."),
        supplied_beats=("archivist", "parish records", "harbour", "third box",
                        "three nights"),
        expectations=_B_COMMON + (
            Expectation("preserves_supplied",
                        "the named premise elements survive into the plot"),),
        plot_mode="partial",
        notes="Rich input: the risk is drift, not invention."),
    BenchCase(
        case_id="B2-sparse-notes",
        track="B", operation="plot.construct",
        condition="sparse notes",
        input_text=(
            "notes:\n"
            "- ferry, off season\n"
            "- somebody is riding it back and forth all day\n"
            "- the ticket inspector knows and says nothing\n"
            "- ends at the last crossing\n"),
        supplied_beats=("ferry", "ticket inspector", "last crossing"),
        expectations=_B_COMMON + (
            Expectation("preserves_supplied",
                        "the notes' elements survive into the plot"),),
        plot_mode="partial",
        notes="Notes, not prose: tests structure from fragments."),
    BenchCase(
        case_id="B3-characters-no-plot",
        track="B", operation="plot.construct",
        condition="characters without a plot",
        input_text=(
            "Three characters, no story yet.\n\n"
            "BASIL, 60s, repairs church organs; has not spoken to his "
            "daughter in nine years; refuses to work on anything electric.\n"
            "SHIREEN, 30s, tunes concert pianos; owes money to the wrong "
            "person; unfailingly polite.\n"
            "TEO, 17, sweeps the workshop; can hear a quarter-tone; lies about "
            "everything small and nothing large.\n\n"
            "Give them a plot."),
        supplied_beats=("BASIL", "SHIREEN", "TEO"),
        expectations=_B_COMMON + (
            Expectation("preserves_supplied",
                        "all three supplied characters appear, with their "
                        "supplied attributes intact"),
            Expectation("character_arcs",
                        "each supplied character gets an arc the plot actually "
                        "uses", layer="judge"),),
        plot_mode="partial",
        notes="Characters are supplied; causality is not."),
    BenchCase(
        case_id="B4-setting-no-characters",
        track="B", operation="plot.construct",
        condition="a setting without characters",
        input_text=(
            "Setting only. No characters, no events.\n\n"
            "A decommissioned funicular railway on a steep town, kept running "
            "one day a week by volunteers. Winter. The lower station is a cafe "
            "now. The upper station is locked and has been since the last "
            "accident, which nobody in the town describes the same way twice."),
        supplied_beats=("funicular", "lower station", "upper station",
                        "the accident"),
        expectations=_B_COMMON + (
            Expectation("preserves_supplied",
                        "the supplied setting elements are used, not replaced"),
            Expectation("invents_characters",
                        "characters are invented and each one is motivated by "
                        "the setting", layer="judge"),),
        plot_mode="partial",
        notes="The setting must survive; the people must be invented."),
    BenchCase(
        case_id="B5-minimal",
        track="B", operation="plot.construct",
        condition="minimal input",
        input_text="A short film about a lighthouse.",
        supplied_beats=("lighthouse",),
        expectations=_B_COMMON + (
            Expectation("preserves_supplied",
                        "the one supplied element (the lighthouse) is in the "
                        "plot"),
            Expectation("originality",
                        "the plot is not the most obvious lighthouse story",
                        layer="judge"),),
        plot_mode="minimal",
        notes="Under MINIMAL_WORDS: k110 routes this to the minimal guidance."),
    BenchCase(
        case_id="B6-no-input",
        track="B", operation="plot.construct",
        condition="no meaningful prior narrative input",
        input_text="video",
        expectations=_B_COMMON + (
            Expectation("originality",
                        "a specific story, not a genre summary", layer="judge"),
            Expectation("thematic_consistency",
                        "the invented plot has one theme rather than three",
                        layer="judge"),),
        plot_mode="minimal",
        notes="The doc's hardest Stage 5 case: the artifact is held to the "
              "SAME validator as B1."),
)


# ---------------------------------------------------------------------------
# Track C — filmmaking workflow generation (one case per operation)
# ---------------------------------------------------------------------------

_C_COMMON = (
    Expectation("validates", "the reply is one JSON object of the requested "
                             "shape"),
    Expectation("complete_artifact",
                "every required field is present on every row"),
    Expectation("covers_source",
                "every id the screenplay defines is covered"),
    Expectation("no_hard_contradictions",
                "no row refers to an id the screenplay does not have"),
)

TRACK_C: tuple[BenchCase, ...] = (
    BenchCase(
        case_id="C1-breakdown",
        track="C", operation="breakdown.script",
        condition="script breakdown from a locked screenplay",
        input_text="", uses_fixture=True,
        expectations=_C_COMMON + (
            Expectation("preserves_supplied",
                        "props and cast named in the screenplay appear in the "
                        "breakdown"),
            Expectation("production_actionability",
                        "a first AD could order the day from this row",
                        layer="judge"),),
        constraints=(
            Constraint("requires_term", "logbook",
                       "the logbook is a named prop and must be broken out"),),
        notes="Coverage = every scene id."),
    BenchCase(
        case_id="C2-continuity",
        track="C", operation="continuity.extract",
        condition="continuity extraction checked against the derived bible",
        input_text="", uses_fixture=True,
        expectations=_C_COMMON + (
            Expectation("continuity_accuracy",
                        "the extracted present/location state matches the "
                        "state derived from the screenplay"),
            Expectation("continuity_judgement",
                        "the state reads as a usable continuity log",
                        layer="judge"),),
        notes="Scored against build_continuity() — a DERIVED answer key."),
    BenchCase(
        case_id="C3-shotlist",
        track="C", operation="shotlist.build",
        condition="shot list from a locked screenplay",
        input_text="", uses_fixture=True,
        expectations=_C_COMMON + (
            Expectation("shot_grammar",
                        "shot sizes and moves come from a real vocabulary",
                        layer="judge"),),
        notes="Coverage = the shot ids build_shot_plan() derives."),
    BenchCase(
        case_id="C4-storyboard",
        track="C", operation="storyboard.prompts",
        condition="storyboard prompts per shot",
        input_text="", uses_fixture=True,
        expectations=_C_COMMON + (
            Expectation("frame_specificity",
                        "each prompt describes a FRAME (subject, framing, "
                        "light) and not the plot", layer="judge"),),
        notes="Each prompt must name its own scene's people or place."),
    BenchCase(
        case_id="C5-segment-prompts",
        track="C", operation="segment.prompts",
        condition="production-ready segment prompt compilation",
        input_text="", uses_fixture=True,
        expectations=_C_COMMON + (
            Expectation("no_cross_segment_leak",
                        "no segment prompt names another scene's location or "
                        "refers to a previous clip"),
            Expectation("render_readiness",
                        "each prompt could be handed to a video model as-is",
                        layer="judge"),),
        constraints=(
            Constraint("forbidden_term", "previous clip",
                       "a segment prompt must not chain off a sibling"),
            Constraint("forbidden_term", "as before",
                       "a segment prompt must not chain off a sibling"),),
        notes="The pipeline's no-chaining rule, measured."),
    BenchCase(
        case_id="C6-assembly",
        track="C", operation="assembly.plan",
        condition="assembly plan over the shot list",
        input_text="", uses_fixture=True,
        expectations=_C_COMMON + (
            Expectation("timeline_partition",
                        "windows are ordered, non-overlapping and gapless"),
            Expectation("assembly_judgement",
                        "the cut order and transitions serve the story",
                        layer="judge"),),
        notes="Deterministic: the timeline is a partition or it is not."),
)


# ---------------------------------------------------------------------------
# Track D — spatial adherence (k119 / or-k7): one case per injected fault
# ---------------------------------------------------------------------------
#
# A Track D case is not a prompt. ``input_text`` is the JSON a runner hands to
# ``spatial_eval.run_track_d_case``: which perturbation to inject into the
# shared synthetic scene, how hard, and which repair code the evaluators must
# emit (``null`` for the clean control). Every expectation is deterministic —
# geometry is measured, never judged — and the whole track runs in-process on
# numpy, which is what lets the evaluators be benchmarked on a host with no GPU.

_D_COMMON = (
    Expectation("measured",
                "every metric the observation set can answer is measured, "
                "none is reported as passed while unmeasured"),
    Expectation("right_code",
                "the first emitted repair code equals expect_code (and none "
                "is emitted for the clean control)"),
    Expectation("evidence",
                "a failing metric names the worst frames / entities so the "
                "repair controller can bound the re-render"),
)


def _d_case(case_id: str, perturbation: str, magnitude: float,
            expect_code: str | None, condition: str, notes: str = "") -> BenchCase:
    import json
    return BenchCase(
        case_id=case_id, track="D", operation="spatial.evaluate",
        condition=condition,
        input_text=json.dumps({"perturbation": perturbation,
                               "magnitude": magnitude,
                               "expect_code": expect_code}, sort_keys=True),
        expectations=_D_COMMON, notes=notes)


TRACK_D: tuple[BenchCase, ...] = (
    _d_case("D0-clean", "none", 0.0, None,
            "perfect render: every metric within DriftThresholds",
            "The control. A gate that fails this has no threshold, only a veto."),
    _d_case("D1-reprojection", "landmark_shift", 10.0, "geometry_drift",
            "landmarks 10 px off the locked geometry -> GEOMETRY_DRIFT",
            "DriftThresholds.landmark_reprojection_px = 6."),
    _d_case("D2-silhouette", "silhouette_erode", 6.0, "geometry_drift",
            "silhouette slid 6 px -> IoU below silhouette_iou_min"),
    _d_case("D3-depth", "depth_scale", 0.2, "geometry_drift",
            "depth scaled by 1.2 -> depth_rel_error_max exceeded"),
    _d_case("D4-normals", "normal_tilt", 20.0, "geometry_drift",
            "normals tilted 20 deg -> normal_angle_deg_max exceeded"),
    _d_case("D5-flow", "flow_shift", 5.0, "geometry_drift",
            "flow off by 5 px -> flow_warp_error_max exceeded"),
    _d_case("D6-camera-offset", "camera_offset", 0.2, "camera_path_mismatch",
            "camera path 20 cm off the locked track -> CAMERA_PATH_MISMATCH"),
    _d_case("D7-camera-yaw", "camera_yaw", 5.0, "camera_path_mismatch",
            "camera yawed 5 deg -> CAMERA_PATH_MISMATCH on rotation alone",
            "Translation stays within camera_drift_m_max; rotation is the tell."),
    _d_case("D8-sink", "sink_into_ground", 0.1, "collision_violation",
            "hero 10 cm into the ground the simulation rested it on"),
    _d_case("D9-float", "float_off_ground", 0.1, "collision_violation",
            "hero floating 10 cm above a resting contact"),
)


#: Every case, in track order.
ALL_CASES: tuple[BenchCase, ...] = TRACK_A + TRACK_B + TRACK_C + TRACK_D

#: Track -> cases.
SUITES: dict[str, tuple[BenchCase, ...]] = {
    "A": TRACK_A, "B": TRACK_B, "C": TRACK_C, "D": TRACK_D,
}


def cases_for(tracks: str | None = None,
              limit_per_track: int | None = None) -> tuple[BenchCase, ...]:
    """The selected cases, in a STABLE order.

    ``tracks`` is a string of track letters ("AB", "ABCD"); None means all
    (the k109 runner's default is "ABC": Track D is scored in-process by
    ``spatial_eval``, not by a model).
    ``limit_per_track`` takes the first N of each suite — that is what the
    polite pilot uses, and taking the FIRST N rather than a sample keeps two
    pilot runs comparable."""
    picked = [t for t in TRACKS if tracks is None or t in tracks.upper()]
    out: list[BenchCase] = []
    for track in picked:
        suite = SUITES[track]
        out.extend(suite if limit_per_track is None
                   else suite[:max(0, int(limit_per_track))])
    return tuple(out)


def case(case_id: str) -> BenchCase:
    for item in ALL_CASES:
        if item.case_id == case_id:
            return item
    raise KeyError(f"no benchmark case {case_id!r}")



# ---------------------------------------------------------------------------
# k109b — the STATIONARY operations and their cases
# ---------------------------------------------------------------------------
#
# k109 asked eighteen different questions of a fleet. k109b asks ONE brief of
# every model at every lifecycle point, so the six operations below exist for
# the six lifecycle steps k109 had no matrix key for. They are ADDITIVE: the
# eight k109 operations above keep their names, their shapes and their rows, so
# a matrix derived from a k109 run and one derived from a k109b run are the
# same file format and ``routing_matrix.best_route`` reads both unchanged.
#
# All six are Track "C" shaped — one JSON object, one named container, one row
# per id — because that is the shape ``workflow_errors`` validates and
# ``author_workflow`` repairs, and inventing a seventh answer shape would mean
# inventing a seventh validator to go with it.
#
# TWO of them carry no id set to cover (``correction.notes`` covers the FAILING
# CHECKS of a fixed rejection report, ``postproduction.plan`` covers the fixed
# segment durations). Their ``coverage`` is deliberately empty here and their
# real answer keys live in ``stationary_scenario.validate_correction_notes`` /
# ``validate_timeline`` — a ``coverage`` value invented to make them look like
# the others would put a wrong id set in the prompt.

STATIONARY_OPERATION_SPECS: dict[str, OperationSpec] = {
    "continuity.bible": OperationSpec(
        operation="continuity.bible", track="C",
        artifact="continuity bible (lifecycle step 5)",
        instruction=("Read the continuity state off the screenplay and assert "
                     "the standing facts. For EVERY scene id give the state "
                     "BEFORE it starts and AFTER it ends: who is present, "
                     "where it is, and what time of day it is."),
        container="segments",
        row_fields=("segment_id", "state_before", "state_after"),
        id_field="segment_id", coverage="scene_ids",
        example=('{"segments": [{"segment_id": "s1", "state_before": '
                 '{"location": "...", "time_of_day": "DUSK", "present": '
                 '["..."]}, "state_after": {"location": "...", '
                 '"time_of_day": "DUSK", "present": ["..."]}}]}')),
    "screenplay.breakdown": OperationSpec(
        operation="screenplay.breakdown", track="C",
        artifact="screenplay breakdown (lifecycle step 6)",
        instruction=("Break this screenplay down for production. One row per "
                     "scene id carrying everything the floor has to bring on "
                     "the day."),
        container="scenes",
        row_fields=("scene_id", "interior_exterior", "location",
                    "time_of_day", "cast", "props", "wardrobe", "sound"),
        id_field="scene_id", coverage="scene_ids",
        example=('{"scenes": [{"scene_id": "s1", "interior_exterior": "EXT", '
                 '"location": "...", "time_of_day": "DUSK", "cast": ["..."], '
                 '"props": ["..."], "wardrobe": ["..."], '
                 '"sound": ["..."]}]}')),
    "shots.design": OperationSpec(
        operation="shots.design", track="C",
        artifact="shot design: storyboard, blocking, camera, lighting "
                 "(lifecycle step 7)",
        instruction=("Design the shots. One row per shot id you are given, "
                     "with framing, camera move, lighting and the line ids "
                     "that shot plays. Use those ids and no others."),
        container="shots",
        row_fields=("segment_id", "scene_id", "shot_size", "camera_move",
                    "lighting", "line_ids", "description"),
        id_field="segment_id", coverage="shot_ids",
        example=('{"shots": [{"segment_id": "s1-1", "scene_id": "s1", '
                 '"shot_size": "wide", "camera_move": "static", '
                 '"lighting": "overcast dusk, available light", '
                 '"line_ids": ["l1"], "description": "..."}]}')),
    "segment.compile-prompt": OperationSpec(
        operation="segment.compile-prompt", track="C",
        artifact="compiled segment specs (lifecycle step 12)",
        instruction=("Compile the production prompt for EVERY segment id you "
                     "are given. Each prompt is compiled from the SCREENPLAY "
                     "independently: it must never name another segment's "
                     "location, never say 'as before' or 'the previous clip', "
                     "and never assume a viewer saw any other segment. A "
                     "segment spec is a SIBLING of the locked artifacts, never "
                     "a child of the segment before it."),
        container="segments",
        row_fields=("segment_id", "prompt", "duration_s"),
        id_field="segment_id", coverage="shot_ids",
        example=('{"segments": [{"segment_id": "s1-1", "prompt": "...", '
                 '"duration_s": 5.0}]}')),
    "correction.notes": OperationSpec(
        operation="correction.notes", track="C",
        artifact="correction data for a rejected result (lifecycle step 15)",
        instruction=("Attempt 1 of the shot was REJECTED. Author the "
                     "correction data for attempt 2 FROM the locked shot spec "
                     "and character sheets. One row per FAILING check and none "
                     "for the checks that passed. Each row states the LOCKED "
                     "value it is restoring. You do not have the rejected "
                     "attempt's prompt and must not ask for it."),
        container="corrections",
        row_fields=("check", "locked_value", "correction", "source"),
        id_field="check", coverage="",
        example=('{"corrections": [{"check": "adherence.time_of_day", '
                 '"locked_value": "...", "correction": "...", '
                 '"source": "shot_spec"}]}')),
    "postproduction.plan": OperationSpec(
        operation="postproduction.plan", track="C",
        artifact="post-production assembly + export plan (lifecycle step 16)",
        instruction=("Lay the segments end to end over the FIXED durations you "
                     "are given: windows in order, no gaps, no overlaps, with "
                     "the transition out of each one. Then give the export "
                     "block for the one delivery target."),
        container="timeline",
        top_fields=("timeline", "export"),
        row_fields=("segment_id", "start_s", "end_s", "transition"),
        id_field="segment_id", coverage="",
        example=('{"timeline": [{"segment_id": "s1-1", "start_s": 0.0, '
                 '"end_s": 5.0, "transition": "CUT TO:"}], "export": '
                 '{"container": "mp4", "video_codec": "h264", "fps": 24, '
                 '"resolution": "1920x1080", "loudness_lufs": -16}}')),
}

# ADDITIVE merge. Asserted rather than assumed: an operation name colliding
# with a k109 key would silently redefine that key's shape and every k109 row
# in an old run dir would then be scored against the wrong validator.
for _name, _spec in STATIONARY_OPERATION_SPECS.items():
    if _name in OPERATIONS:
        raise RuntimeError(
            f"k109b operation {_name!r} collides with a k109 operation — "
            f"routing-matrix keys must be unique across waves")
    OPERATIONS[_name] = _spec
del _name, _spec


def _stationary_cases() -> tuple[BenchCase, ...]:
    """The eight LLM cases of the stationary sweep, in LIFECYCLE ORDER.

    Every input is derived from ``stationary_scenario`` and from nothing else —
    that is the whole point of the wave. The import is function-local so
    ``benchmark_cases`` stays importable by anything that only wants k109's
    suites (``stationary_scenario`` imports the k110 constructors and builds a
    Screenplay at module scope)."""
    from hugpy_oracle.stationary_scenario import (
        LIFECYCLE_POINTS,
        PREMISE_FRAGMENT,
        SCREENPLAY_EXCERPT,
        SUPPLIED_BEATS,
        SUPPLIED_LINES,
    )

    common = (
        Expectation("structured", "the reply is ONE JSON object of the "
                                  "declared shape, with no prose around it"),
        Expectation("scenario_grounded",
                    "the answer is about SALT LINE and invents no character, "
                    "location or id that is not in the brief"),
    )
    by_op = {op: point for point in LIFECYCLE_POINTS
             for op in point.operations}

    cases: list[BenchCase] = []
    for operation, point in by_op.items():
        if point.kind != "llm":
            continue
        judged = tuple(
            Expectation(e.key, e.detail, layer="judge")
            for e in point.expectations if e.layer == "judge")
        checked = tuple(
            Expectation(e.key, e.detail)
            for e in point.expectations
            if e.layer == "deterministic" and e.key not in
            {c.key for c in common})
        if operation == "plot.construct":
            cases.append(BenchCase(
                case_id="SP03-plot", track="B", operation=operation,
                condition="stationary premise fragment -> PlotSpec",
                input_text=PREMISE_FRAGMENT,
                supplied_beats=SUPPLIED_BEATS,
                expectations=common + checked + judged,
                notes=f"lifecycle step {point.step}: {point.name}"))
        elif operation == "screenplay.complete":
            cases.append(BenchCase(
                case_id="SP04-screenplay", track="A", operation=operation,
                condition="stationary partial screenplay -> full Screenplay",
                input_text=SCREENPLAY_EXCERPT,
                supplied_lines=SUPPLIED_LINES, supplied_beats=SUPPLIED_BEATS,
                expectations=common + checked + judged,
                notes=f"lifecycle step {point.step}: {point.name}"))
        else:
            cases.append(BenchCase(
                case_id=f"SP{point.step:02d}-{operation.split('.')[-1]}",
                track="C", operation=operation,
                condition=f"stationary scenario -> {OPERATIONS[operation].artifact}",
                input_text="", uses_fixture=True,
                expectations=common + checked + judged,
                notes=f"lifecycle step {point.step}: {point.name}"))
    order = {op: i for i, op in enumerate(by_op)}
    return tuple(sorted(cases, key=lambda c: order[c.operation]))


#: The stationary suite: one case per LLM lifecycle point, lifecycle-ordered.
STATIONARY_CASES: tuple[BenchCase, ...] = _stationary_cases()

#: case_id -> case, for the sweep's resume path.
STATIONARY_BY_ID: dict[str, BenchCase] = {c.case_id: c
                                          for c in STATIONARY_CASES}


def stationary_case_for(operation: str) -> BenchCase:
    """The stationary case that measures ``operation``. Raises for an
    operation this wave does not measure — a sweep that silently skipped a
    point would report a gap that is really a typo."""
    for item in STATIONARY_CASES:
        if item.operation == operation:
            return item
    raise KeyError(f"no stationary case for operation {operation!r}; "
                   f"measured: {sorted(c.operation for c in STATIONARY_CASES)}")


__all__ = [
    "ALL_CASES", "BenchCase", "CONSTRAINT_KINDS", "Constraint",
    "Contradiction", "Expectation", "FIXTURE_SCREENPLAY", "LAYERS",
    "OPERATIONS", "OperationSpec", "STATIONARY_BY_ID", "STATIONARY_CASES",
    "STATIONARY_OPERATION_SPECS", "SUITES", "TRACKS", "TRACK_A", "TRACK_B",
    "TRACK_C", "TRACK_D", "case", "cases_for", "stationary_case_for",
]
