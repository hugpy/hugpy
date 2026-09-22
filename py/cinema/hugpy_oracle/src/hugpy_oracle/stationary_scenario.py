"""k109b — THE STATIONARY PROMPT: one canonical brief, every model, every point.

k109 asked "which model is best at operation X" with eighteen different cases.
This module asks the harder and more useful question the operator actually
posed — *"which model is capable of each point of the lifecycle, so they can
all actively contribute their strengths to the ultimate realized professional
piece"* — and it can only be answered if **every model at every point sees
inputs derived from ONE brief**. Comparability is the entire point. A sweep
where the 3B model got the easy case and the 27B model got the hard one
measures the case assignment, not the fleet.

So: one two-character, three-line, two-location micro-film — ``SALT LINE`` —
defined ONCE, here, as seven pieces the doc's lifecycle actually consumes:

    PREMISE_FRAGMENT      the seed for step 3   (plot construction)
    SCREENPLAY_EXCERPT    the partial for step 4 (screenplay completion)
    CHARACTER_SHEETS      the identities steps 5-7 and 12 must hold
    CONTINUITY_FACTS      the fixed fact set step 5's bible must not break
    SHOT_REQUEST          the ONE shot every visual model renders and judges
    TONE                  the value step 13 would style to
    ANSWER_KEY            expected properties per lifecycle point

Everything downstream is DERIVED from those: the keyframe prompt, the clip
prompt, the TTS line, the six reference frames (with their planted
violations), and the k110 ``Screenplay`` that gives steps 5-7 a machine-derived
answer key instead of a hand-typed one.

DETERMINISM IS THE CONTRACT. :func:`scenario_digest` is a sha256 over the
canonical JSON of every piece, sorted, with no clock and no environment in it.
Two processes on two days compute the same digest or the scenario changed, and
:data:`SCENARIO_VERSION` is recorded in EVERY result row so a matrix can never
be compared against evidence gathered under a different brief.

THE HONESTY LINE THIS MODULE WILL NOT CROSS. The six reference frames used to
score VLM judging are RENDERS OF PROMPTS, and their answer key is derived from
**the prompt each frame was rendered from, not from a human looking at the
pixels**. A renderer that ignored a planted violation makes that frame's key
wrong. That limitation is spelled out in
:data:`REFERENCE_FRAME_KEY_BASIS`, carried onto every VLM row, and is why the
VLM stage reports a SECOND, key-independent axis (grounding) beside it.

No pathlib anywhere. No fleet, no clock, no I/O: importing this module touches
nothing but the k110 artifact constructors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from hugpy_oracle.screenplay import AVEvent, Line, Scene, Screenplay

#: Bump when ANY piece below changes. Recorded in every result row and in the
#: serialized routing matrix, because a route measured under a different brief
#: is evidence about a different question.
SCENARIO_VERSION: str = "salt-line/1"

#: The scenario's own name, used in run dirs and report headings.
SCENARIO_TITLE: str = "SALT LINE"


# ---------------------------------------------------------------------------
# Piece 1 — the premise fragment (lifecycle step 3 input)
# ---------------------------------------------------------------------------

PREMISE_FRAGMENT: str = (
    "PREMISE FRAGMENT — a two-hander, one night, one working harbour.\n\n"
    "A harbour pilot, NIA, logs that the number 7 channel buoy has moved a "
    "hundred metres off its charted position. Tides do not move a buoy that "
    "far. The only other person on the wall is TEODOR, a commercial diver who "
    "was in that water the night before and who has not said why.\n\n"
    "The film is two locations and one decision: whether to report the "
    "position they can prove, or the position they believe. It ends when the "
    "log is written."
)


# ---------------------------------------------------------------------------
# Piece 3 — the character sheets
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CharacterSheet:
    """One identity every downstream point has to hold constant.

    ``visual_anchor`` is the ONE thing an image/video model and a VLM judge can
    both check without ambiguity, which is why every reference frame and the
    keyframe prompt are written around it."""
    name: str
    role: str
    visual_anchor: str
    wardrobe: tuple[str, ...]
    props: tuple[str, ...]
    voice: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role,
                "visual_anchor": self.visual_anchor,
                "wardrobe": list(self.wardrobe), "props": list(self.props),
                "voice": self.voice}


CHARACTER_SHEETS: tuple[CharacterSheet, ...] = (
    CharacterSheet(
        name="NIA", role="harbour pilot, twenty years on this water",
        visual_anchor="a bright yellow foul-weather jacket, hood down",
        wardrobe=("yellow foul-weather jacket", "black work trousers"),
        props=("pilot's logbook", "handheld VHF radio"),
        voice="low, flat, unhurried; states facts and stops"),
    CharacterSheet(
        name="TEODOR", role="commercial diver, contracted, not local",
        visual_anchor="a grey neoprene hood pushed back and a white dive slate "
                      "on a lanyard",
        wardrobe=("grey neoprene hood", "dark drysuit, unzipped to the waist"),
        props=("white dive slate on a lanyard",),
        voice="quieter, careful; answers a question with a smaller question"),
)


# ---------------------------------------------------------------------------
# Piece 4 — the continuity fact set (lifecycle step 5's answer key)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContinuityFact:
    """One fact the continuity bible must assert and must never contradict.

    ``checkable`` marks the facts a DETERMINISTIC validator can test against a
    model's answer (a substring/id test); the rest are judge-only and are
    labelled so, rather than being scored by a regex that would pretend."""
    fact_id: str
    statement: str
    scope: str                       # "film" | scene id
    checkable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"fact_id": self.fact_id, "statement": self.statement,
                "scope": self.scope, "checkable": self.checkable}


CONTINUITY_FACTS: tuple[ContinuityFact, ...] = (
    ContinuityFact("cf1", "NIA wears the yellow foul-weather jacket in every "
                          "scene of the film", "film"),
    ContinuityFact("cf2", "TEODOR carries the white dive slate on a lanyard in "
                          "every scene of the film", "film"),
    ContinuityFact("cf3", "the stone of the harbour wall is WET in every "
                          "exterior scene; the rain has stopped but nothing "
                          "has dried", "film"),
    ContinuityFact("cf4", "the number 7 channel buoy is GREEN and carries the "
                          "numeral 7", "film"),
    ContinuityFact("cf5", "the pilot hut has exactly one window and one desk "
                          "lamp; the lamp is the only practical light in s2",
                  "s2"),
    ContinuityFact("cf6", "only NIA and TEODOR ever appear; there is no third "
                          "person anywhere in the film", "film"),
    ContinuityFact("cf7", "story time runs forward only: DUSK then NIGHT, "
                          "never back", "film"),
    ContinuityFact("cf8", "the logbook is open in s2 and closed in s3; it is "
                          "never open on the wall", "film", checkable=False),
)


# ---------------------------------------------------------------------------
# Piece 6 — the tone value (what step 13 would style to)
# ---------------------------------------------------------------------------

TONE: str = (
    "cold, procedural, unsentimental. Documentary handheld realism: available "
    "light, desaturated blue-grey palette, no lens flare, no score under "
    "dialogue, no slow motion. The camera observes and does not comment."
)


# ---------------------------------------------------------------------------
# The canonical screenplay — the artifact every derived answer key comes from
# ---------------------------------------------------------------------------
#
# THREE lines of dialogue, TWO characters, TWO locations, three scenes. Small
# on purpose: this benchmark measures whether a model can hold ONE brief across
# sixteen different jobs, and a forty-scene fixture would measure context
# length instead. Every id here is stable and is what the answer keys quote.

LINE_1: str = "The number seven buoy has moved a hundred metres. Tides do not do that."
LINE_2: str = "Then somebody walked it. In that water, at night."
LINE_3: str = "We log the position we can prove, and we sail the line we logged."


def _build_screenplay() -> Screenplay:
    """The canonical ``SALT LINE`` screenplay, built through the SAME k110
    constructors the pipeline uses — so "the scenario is valid" means the
    pipeline would actually have accepted it, not that it reads well."""
    s1 = Scene(
        scene_id="s1",
        heading="EXT. HARBOUR WALL - DUSK",
        location="HARBOUR WALL", time_of_day="DUSK",
        action=("NIA walks the wet stone of the wall with the logbook under "
                "her arm, watching the water. Out in the channel the green "
                "number 7 buoy sits where nothing has ever sat. TEODOR comes "
                "up the steps behind her, hood pushed back, the dive slate "
                "swinging at his chest."),
        staging=("NIA at the seaward edge, camera left; TEODOR arrives from "
                 "frame right at the steps. The buoy is in the water behind "
                 "them, camera right of centre."),
        present_at_open=("NIA",), entrances=("TEODOR",),
        dialogue=(Line(line_id="l1", speaker="NIA", text=LINE_1),
                  Line(line_id="l2", speaker="TEODOR", text=LINE_2)),
        av_events=(AVEvent(kind="ambience",
                           description="harbour swell against stone, constant",
                           cue="under the whole scene"),),
        props=("pilot's logbook", "white dive slate", "green number 7 buoy"),
        wardrobe=("yellow foul-weather jacket", "grey neoprene hood"),
        weather="rain stopped, overcast, everything wet",
        lighting="last daylight, flat and blue-grey, no sun",
        transition="CUT TO:", story_time_s=0.0, beat_id="b1")
    s2 = Scene(
        scene_id="s2",
        heading="INT. PILOT HUT - NIGHT",
        location="PILOT HUT", time_of_day="NIGHT",
        action=("The logbook is open under the desk lamp. NIA writes the "
                "charted position, then the measured one, then rules a line "
                "between them. TEODOR stands at the one window with his back "
                "to her."),
        staging=("NIA seated at the desk, camera right; TEODOR at the window, "
                 "camera left, in silhouette against the black glass."),
        present_at_open=("NIA", "TEODOR"),
        dialogue=(Line(line_id="l3", speaker="NIA", text=LINE_3),),
        av_events=(AVEvent(kind="sfx", description="pen on paper, single ruled line",
                           cue="on NIA's last word"),),
        props=("pilot's logbook", "desk lamp", "white dive slate"),
        wardrobe=("yellow foul-weather jacket", "grey neoprene hood"),
        weather="dry indoors, rain stopped outside",
        lighting="one desk lamp, the only practical in the room",
        transition="CUT TO:", story_time_s=1800.0, beat_id="b2")
    s3 = Scene(
        scene_id="s3",
        heading="EXT. HARBOUR WALL - NIGHT",
        location="HARBOUR WALL", time_of_day="NIGHT",
        action=("The logbook is closed. NIA and TEODOR stand at the seaward "
                "edge and watch the green light of the number 7 buoy flash "
                "where it should not be. Neither of them says anything more "
                "about it."),
        staging=("Both at the seaward edge, backs to camera, the buoy light "
                 "small in the distance, camera right of centre."),
        present_at_open=("NIA", "TEODOR"),
        av_events=(AVEvent(kind="ambience",
                           description="swell against stone, further off now",
                           cue="under the whole scene"),),
        props=("pilot's logbook", "white dive slate", "green number 7 buoy"),
        wardrobe=("yellow foul-weather jacket", "grey neoprene hood"),
        weather="dry, overcast, everything still wet",
        lighting="harbour sodium behind, the buoy's green flash in front",
        transition="FADE OUT.", story_time_s=2400.0, beat_id="b3")
    return Screenplay(
        title=SCENARIO_TITLE,
        logline=("A harbour pilot and a diver find a channel buoy a hundred "
                 "metres off station and choose what to write down."),
        scenes=(s1, s2, s3), characters=("NIA", "TEODOR"))


#: The canonical artifact. Built once at import, frozen, content-addressed.
SCENARIO_SCREENPLAY: Screenplay = _build_screenplay()


# ---------------------------------------------------------------------------
# Piece 2 — the partial screenplay excerpt (lifecycle step 4 input)
# ---------------------------------------------------------------------------
#
# Scene 1 ONLY, written out as a human would hand it over. The model is asked
# to complete the film; the two supplied lines must survive verbatim, which is
# a countable check, and the two locations bound what "complete" may mean.

SCREENPLAY_EXCERPT: str = (
    "PARTIAL SCREENPLAY — the first scene exists. The film has no middle and "
    "no ending yet. Do not change the supplied scene.\n\n"
    "EXT. HARBOUR WALL - DUSK\n"
    "NIA walks the wet stone of the wall with the logbook under her arm, "
    "watching the water. Out in the channel the green number 7 buoy sits where "
    "nothing has ever sat. TEODOR comes up the steps behind her, hood pushed "
    "back, the dive slate swinging at his chest.\n"
    "NIA\n"
    f"{LINE_1}\n"
    "TEODOR\n"
    f"{LINE_2}\n"
    "CUT TO:\n"
)

#: The lines a completion MUST preserve verbatim. Counted, not judged.
SUPPLIED_LINES: tuple[str, ...] = (LINE_1, LINE_2)

#: The beats a completion must still be about.
SUPPLIED_BEATS: tuple[str, ...] = ("harbour wall", "pilot hut",
                                   "number 7 buoy", "the logbook")


# ---------------------------------------------------------------------------
# Piece 5 — the shot request (the ONE shot every visual model renders)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShotRequest:
    """The single shot the image, video and VLM stages all work on.

    ONE shot, not a set: an image model, a clip model and a judge scored on
    three different frames are three unrelated measurements wearing one
    heading. ``required_elements`` / ``forbidden_elements`` are what the VLM
    judge is asked about and what the answer key is written in terms of."""
    shot_id: str
    scene_id: str
    heading: str
    shot_size: str
    camera: str
    lighting: str
    subject: str
    action: str
    required_elements: tuple[str, ...]
    forbidden_elements: tuple[str, ...]
    duration_s: float = 4.0

    def to_dict(self) -> dict[str, Any]:
        return {"shot_id": self.shot_id, "scene_id": self.scene_id,
                "heading": self.heading, "shot_size": self.shot_size,
                "camera": self.camera, "lighting": self.lighting,
                "subject": self.subject, "action": self.action,
                "required_elements": list(self.required_elements),
                "forbidden_elements": list(self.forbidden_elements),
                "duration_s": self.duration_s}

    @property
    def spec_text(self) -> str:
        """The shot spec as a judge is shown it — one block, no JSON, because
        a VLM asked to hold a schema AND look at a picture does neither."""
        return (
            f"SHOT {self.shot_id} ({self.scene_id})\n"
            f"  slugline: {self.heading}\n"
            f"  framing: {self.shot_size}\n"
            f"  camera: {self.camera}\n"
            f"  light: {self.lighting}\n"
            f"  subject: {self.subject}\n"
            f"  action: {self.action}\n"
            f"  MUST SHOW: {'; '.join(self.required_elements)}\n"
            f"  MUST NOT SHOW: {'; '.join(self.forbidden_elements)}")


SHOT_REQUEST: ShotRequest = ShotRequest(
    shot_id="s1-2", scene_id="s1",
    heading="EXT. HARBOUR WALL - DUSK",
    shot_size="medium two-shot, both figures from the knees up",
    camera="static, eye level, 35mm, no movement",
    lighting="last daylight under heavy overcast; flat, blue-grey, no sun, "
             "no shadows",
    subject="NIA and TEODOR facing each other on the wall",
    action=("TEODOR has just come up the steps; NIA has turned from the water "
            "to face him. Neither is moving."),
    required_elements=(
        "exactly two people and no more",
        "a woman in a bright yellow foul-weather jacket",
        "a man in a dark drysuit with a white dive slate on a lanyard",
        "wet stone harbour wall underfoot",
        "a green channel buoy in the water behind them",
        "overcast dusk light, no sun",
    ),
    forbidden_elements=(
        "a third person",
        "sunshine, blue sky or visible sun",
        "an interior room",
        "a sailing boat under sail",
        "any text, caption, watermark or subtitle",
    ),
    duration_s=4.0)


# ---------------------------------------------------------------------------
# Derived prompts — every visual/audio stage's ONE input string
# ---------------------------------------------------------------------------

KEYFRAME_PROMPT: str = (
    "Documentary film still, medium two-shot: a woman in a bright yellow "
    "foul-weather jacket faces a man in a dark drysuit with a white dive slate "
    "on a lanyard, standing on a wet stone harbour wall at dusk under heavy "
    "overcast. A green channel buoy floats in the water behind them. Flat "
    "blue-grey available light, no sun, 35mm, eye level, static camera."
)

KEYFRAME_NEGATIVE_PROMPT: str = (
    "third person, crowd, sunshine, blue sky, lens flare, interior, sailing "
    "boat, text, caption, watermark, subtitle, cartoon, illustration"
)

CLIP_PROMPT: str = (
    KEYFRAME_PROMPT + " The two figures hold still and look at each other; "
    "only the water moves. No camera movement."
)

#: The one line every TTS candidate speaks. NIA's, because it is the longest
#: fully-declarative line and therefore the one round-trip ASR can actually
#: score without a coin flip on a two-word utterance.
TTS_LINE: str = LINE_3
TTS_SPEAKER: str = "NIA"

#: Fixed render geometry for the sweep. Low on purpose — the question is
#: "can this model produce the shot at all", and a 1024px sweep across 25 image
#: models is an hour of GPU spent proving the same thing at more pixels.
KEYFRAME_WIDTH: int = 512
KEYFRAME_HEIGHT: int = 512
KEYFRAME_SEED: int = 109_1102
CLIP_WIDTH: int = 512
CLIP_HEIGHT: int = 320
CLIP_FPS: int = 16
CLIP_SEED: int = 109_1103


# ---------------------------------------------------------------------------
# The six reference frames — the VLM stage's fixed input set
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    """One frame every VLM judge is shown, and what a good judge should say.

    ``violations`` is the answer key: the labels a judge that is doing its job
    must raise. An EMPTY tuple means the frame adheres and the judge must NOT
    invent a fault — a judge that fails every frame is not a strict judge, it
    is a useless one, and the compliant frames are what catch that."""
    frame_id: str
    prompt: str
    violations: tuple[str, ...]
    expected_verdict: str            # "YES" (adheres) | "NO" (violates)
    rationale: str = ""

    def __post_init__(self) -> None:
        expected = "NO" if self.violations else "YES"
        if self.expected_verdict != expected:
            raise ValueError(
                f"ReferenceFrame({self.frame_id}): expected_verdict "
                f"{self.expected_verdict!r} disagrees with violations "
                f"{self.violations!r} — the key must be derivable from the "
                f"violation list, not typed twice")

    def to_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "prompt": self.prompt,
                "violations": list(self.violations),
                "expected_verdict": self.expected_verdict,
                "rationale": self.rationale}


#: THE LIMITATION, stated once and carried onto every VLM result row.
REFERENCE_FRAME_KEY_BASIS: str = (
    "The answer key for each reference frame is derived from the PROMPT the "
    "frame was rendered from, not from a human inspecting the pixels. A "
    "renderer that silently ignored a planted violation makes that frame's key "
    "wrong, and a judge would then be marked down for being correct. This is a "
    "known heuristic and is why the VLM stage also reports GROUNDING — a "
    "key-independent measure of whether the judge answered about the shot spec "
    "at all — beside key agreement, and why the two are never summed into one "
    "number without both being visible."
)

REFERENCE_FRAMES: tuple[ReferenceFrame, ...] = (
    ReferenceFrame(
        frame_id="rf1-compliant",
        prompt=KEYFRAME_PROMPT,
        violations=(), expected_verdict="YES",
        rationale="the shot as specified; a judge must pass it"),
    ReferenceFrame(
        frame_id="rf2-compliant-wide",
        prompt=("Documentary film still, medium two-shot on a wet stone "
                "harbour wall at dusk under heavy overcast: a woman in a "
                "bright yellow foul-weather jacket turned toward a man in a "
                "dark drysuit wearing a white dive slate on a lanyard. A green "
                "channel buoy on the water behind. Flat grey available light, "
                "35mm, static."),
        violations=(), expected_verdict="YES",
        rationale="the same shot from a second framing; still compliant"),
    ReferenceFrame(
        frame_id="rf3-wrong-time",
        prompt=("Documentary film still, medium two-shot: a woman in a bright "
                "yellow foul-weather jacket faces a man in a dark drysuit with "
                "a white dive slate on a stone harbour wall at NOON in "
                "BRILLIANT SUNSHINE under a CLEAR BLUE SKY, hard shadows, sun "
                "flare. A green channel buoy behind them. 35mm, static."),
        violations=("time_of_day", "lighting"), expected_verdict="NO",
        rationale="planted: bright noon sun and blue sky against a specified "
                  "overcast dusk with no sun"),
    ReferenceFrame(
        frame_id="rf4-third-person",
        prompt=("Documentary film still on a wet stone harbour wall at dusk "
                "under heavy overcast: THREE PEOPLE standing together — a "
                "woman in a bright yellow foul-weather jacket, a man in a dark "
                "drysuit with a white dive slate, and a THIRD MAN in a navy "
                "harbour-master's uniform. A green channel buoy behind. Flat "
                "grey light, 35mm, static."),
        violations=("cast_count", "third_person"), expected_verdict="NO",
        rationale="planted: a third person in a two-hander"),
    ReferenceFrame(
        frame_id="rf5-wrong-wardrobe",
        prompt=("Documentary film still, medium two-shot on a wet stone "
                "harbour wall at dusk under heavy overcast: a woman in a "
                "BRIGHT RED raincoat faces a man in a dark drysuit with "
                "NOTHING around his neck and no slate. A green channel buoy "
                "behind. Flat grey light, 35mm, static."),
        violations=("wardrobe", "prop"), expected_verdict="NO",
        rationale="planted: NIA's jacket is red not yellow, and TEODOR's dive "
                  "slate — his visual anchor — is gone"),
    ReferenceFrame(
        frame_id="rf6-wrong-location",
        prompt=("Documentary film still: a woman in a bright yellow "
                "foul-weather jacket and a man in a dark drysuit with a white "
                "dive slate stand INSIDE a small wooden hut at a desk under a "
                "single desk lamp, one black window behind them. Warm lamp "
                "light, 35mm, static."),
        violations=("location", "interior_exterior", "lighting"),
        expected_verdict="NO",
        rationale="planted: the interior pilot hut standing in for the "
                  "exterior harbour wall"),
)

#: Every violation label the key uses, for the report's own vocabulary check.
VIOLATION_LABELS: tuple[str, ...] = tuple(sorted(
    {v for f in REFERENCE_FRAMES for v in f.violations}))


# ---------------------------------------------------------------------------
# Fixed inputs for the two points that need a PRIOR result to work on
# ---------------------------------------------------------------------------
#
# Step 15 corrects FROM the locked spec with explicit correction data — never
# from the rejected result's own prompt. To measure that honestly the model has
# to be handed a rejected result, and it has to be the SAME rejected result for
# every model. So it is written down here, once, and it is deliberately a
# report about frames rf3 and rf5 — the violations this scenario already
# planted, so the whole brief stays one brief.

REJECTED_VALIDATION_REPORT: str = (
    "VALIDATION REPORT — attempt 1 of shot s1-2 was REJECTED.\n"
    "  adherence.time_of_day   FAIL  frame reads as bright midday sun under a "
    "clear blue sky; the locked shot specifies overcast dusk with no sun\n"
    "  adherence.wardrobe      FAIL  the woman's jacket renders RED; the locked "
    "character sheet specifies a bright yellow foul-weather jacket\n"
    "  adherence.prop          FAIL  no white dive slate is visible on the "
    "man; it is his locked visual anchor\n"
    "  adherence.cast_count    PASS  exactly two figures\n"
    "  adherence.location      PASS  wet stone harbour wall\n"
    "  tone                    FAIL  hard sunlight and saturated colour "
    "contradict the locked tone value\n"
    "The rejected attempt's own prompt is NOT supplied and must not be asked "
    "for: the correction is authored FROM the locked segment spec."
)

#: Step 16 needs durations to lay a timeline over. Fixed, so every model plans
#: the same edit rather than one they invented a length for.
SEGMENT_DURATIONS_S: tuple[tuple[str, float], ...] = (
    ("s1-1", 5.0), ("s1-2", 4.0), ("s2-1", 6.0), ("s3-1", 7.0),
)

#: The delivery target step 16 exports to. One target, so "export plan" is a
#: comparable answer and not a free-form essay about codecs.
DELIVERY_TARGET: str = (
    "single 1920x1080 H.264 MP4 at 24 fps, stereo 48 kHz AAC, "
    "R128 -16 LUFS integrated, no burned-in titles"
)


# ---------------------------------------------------------------------------
# Piece 7 — the validation answer key, per lifecycle point
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointExpectation:
    """One expected property of a good answer at one lifecycle point.

    ``layer`` is ``deterministic`` when a validator in this package actually
    tests it and ``judge`` when only a rubric can. Nothing is ever listed as
    deterministic that no code checks — the whole value of an answer key is
    that it is not aspirational."""
    key: str
    detail: str
    layer: str = "deterministic"

    def __post_init__(self) -> None:
        if self.layer not in ("deterministic", "judge"):
            raise ValueError(
                f"PointExpectation({self.key}).layer must be 'deterministic' "
                f"or 'judge'; got {self.layer!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "detail": self.detail, "layer": self.layer}


# ---------------------------------------------------------------------------
# The lifecycle point map — all sixteen steps, honestly
# ---------------------------------------------------------------------------

#: The kinds a point can be. ``pipeline`` points are executed by this codebase,
#: not by a model, and saying "no model is capable of step 2" would be a
#: category error; ``gap`` points are ones the doc asks for and this fleet
#: cannot serve at all. Both are reported, and they are NOT the same finding.
POINT_KINDS: tuple[str, ...] = ("llm", "vlm", "image", "video", "tts",
                                "pipeline", "gap")


@dataclass(frozen=True, slots=True)
class LifecyclePoint:
    """One of the sixteen ordered steps, mapped to what the sweep can measure.

    ``operations`` are the routing-matrix keys measured AT this point (empty
    for ``pipeline`` and ``gap`` points). ``capability`` is the catalog
    capability the candidates are enumerated from. ``missing_capability``
    names, for a ``gap`` point, exactly what this fleet does not have — the gap
    IS the data, and naming it is the deliverable."""
    step: int
    point_id: str
    name: str
    kind: str
    operations: tuple[str, ...] = ()
    capability: str = ""
    expectations: tuple[PointExpectation, ...] = ()
    missing_capability: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in POINT_KINDS:
            raise ValueError(f"LifecyclePoint({self.point_id}).kind "
                             f"{self.kind!r} is not one of {list(POINT_KINDS)}")
        if self.kind in ("pipeline", "gap") and self.operations:
            raise ValueError(
                f"LifecyclePoint({self.point_id}) is a {self.kind} point and "
                f"must declare no operations; got {list(self.operations)}")
        if self.kind not in ("pipeline", "gap") and not self.operations:
            raise ValueError(
                f"LifecyclePoint({self.point_id}) is a {self.kind} point and "
                f"must declare at least one operation")
        if self.kind == "gap" and not self.missing_capability:
            raise ValueError(
                f"LifecyclePoint({self.point_id}) is a gap point and must NAME "
                f"the missing capability — an unnamed gap is not evidence")

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "point_id": self.point_id,
                "name": self.name, "kind": self.kind,
                "operations": list(self.operations),
                "capability": self.capability,
                "expectations": [e.to_dict() for e in self.expectations],
                "missing_capability": list(self.missing_capability),
                "note": self.note}


_LLM_COMMON: tuple[PointExpectation, ...] = (
    PointExpectation("structured", "the reply is ONE JSON object of the "
                                   "declared shape, with no prose around it"),
    PointExpectation("scenario_grounded", "the answer is about SALT LINE — it "
                                          "names this film's people, places "
                                          "and ids, and invents no others"),
)

LIFECYCLE_POINTS: tuple[LifecyclePoint, ...] = (
    LifecyclePoint(
        step=1, point_id="p01-qualify-route",
        name="Qualify and route models -> versioned routing registry",
        kind="pipeline",
        note=("This sweep IS step 1. It is executed by "
              "oracle/benchmark.run_stationary_sweep against the live catalog, "
              "not by a model, and its output is the registry this run "
              "writes. A model verdict here would be circular.")),
    LifecyclePoint(
        step=2, point_id="p02-run-snapshot",
        name="Immutable run snapshot (pre-existing prompts/refs/routes only)",
        kind="pipeline",
        note=("Executed by oracle/script_first.GenerationSnapshot (k104/k114). "
              "Structural, model-free: a snapshot is captured and asserted, "
              "never authored.")),
    LifecyclePoint(
        step=3, point_id="p03-plot",
        name="Construct the plot",
        kind="llm", operations=("plot.construct",), capability="text.chat",
        expectations=_LLM_COMMON + (
            PointExpectation("plotspec_valid",
                             "the reply builds a valid k110 PlotSpec: every "
                             "beat names a character, every causal link "
                             "resolves backwards to a real beat"),
            PointExpectation("premise_preserved",
                             "the buoy, the two-hander and the log decision "
                             "all survive from the premise fragment"),
            PointExpectation("filmable",
                             "the plot could be shot in two locations with two "
                             "actors", layer="judge"),
        )),
    LifecyclePoint(
        step=4, point_id="p04-screenplay",
        name="Write the complete screenplay (before any generation prompts)",
        kind="llm", operations=("screenplay.complete",), capability="text.chat",
        expectations=_LLM_COMMON + (
            PointExpectation("screenplay_valid",
                             "the reply builds a valid k110 Screenplay: real "
                             "sluglines, no speaker outside the room, story "
                             "time monotonic, unique line ids"),
            PointExpectation("lines_verbatim",
                             "both supplied lines appear VERBATIM, attributed "
                             "to the speaker they were given to"),
            PointExpectation("two_locations",
                             "the completed film still lives in the harbour "
                             "wall and the pilot hut"),
            PointExpectation("earns_the_ending",
                             "the written material earns the log decision "
                             "rather than announcing it", layer="judge"),
        )),
    LifecyclePoint(
        step=5, point_id="p05-continuity",
        name="Continuity bible",
        kind="llm", operations=("continuity.bible",), capability="text.chat",
        expectations=_LLM_COMMON + (
            PointExpectation("covers_every_scene",
                             "one entry per scene id, none invented"),
            PointExpectation("state_matches_derived",
                             "the before/after presence and location match the "
                             "state build_continuity() derives from the same "
                             "screenplay — a DERIVED key, never hand-typed"),
            PointExpectation("holds_the_facts",
                             "the standing fact set (yellow jacket, dive "
                             "slate, wet stone, green buoy 7, two people only) "
                             "is asserted and never contradicted"),
        )),
    LifecyclePoint(
        step=6, point_id="p06-breakdown",
        name="Screenplay breakdown",
        kind="llm", operations=("screenplay.breakdown",), capability="text.chat",
        expectations=_LLM_COMMON + (
            PointExpectation("covers_every_scene",
                             "one breakdown row per scene id, none invented"),
            PointExpectation("floor_actionable",
                             "cast, props, wardrobe and sound are all present "
                             "on every row"),
            PointExpectation("orderable",
                             "a first AD could order the shooting day off "
                             "these rows", layer="judge"),
        )),
    LifecyclePoint(
        step=7, point_id="p07-shot-design",
        name="Shot design (storyboard, blocking, camera, lighting, timing)",
        kind="llm", operations=("shots.design",), capability="text.chat",
        expectations=_LLM_COMMON + (
            PointExpectation("covers_derived_shot_ids",
                             "the shot ids build_shot_plan() derives are all "
                             "covered and none is invented"),
            PointExpectation("lines_attached",
                             "every line id is carried by exactly the shot "
                             "that plays it"),
            PointExpectation("real_grammar",
                             "shot sizes, moves and lighting come from a real "
                             "vocabulary, not adjectives", layer="judge"),
        )),
    LifecyclePoint(
        step=8, point_id="p08-spatial-feasibility",
        name="Spatial feasibility planning (geometry informs PRE-lock revision)",
        kind="gap",
        missing_capability=("scene.geometry.solve", "scene.collision.check",
                            "camera.path.feasibility"),
        note=("NO capability on this fleet answers 'can this blocking exist in "
              "this space'. The catalog serves image.depth / image.detect / "
              "image.segment (one model each) — SINGLE-FRAME 2D inference, "
              "which is not a feasibility solve over a staged volume and must "
              "not be reported as one. Wave 5.")),
    LifecyclePoint(
        step=9, point_id="p09-lock",
        name="Lock the production plan",
        kind="pipeline",
        note=("Executed by oracle/production.lock_production (k106) and "
              "oracle/script_first (k114): parent digests are collected and "
              "frozen. Model-free by construction — a lock a model could "
              "author would not be a lock.")),
    LifecyclePoint(
        step=10, point_id="p10-fold1-capture",
        name="Fold 1: spatial data capture (transforms, rigs, mocap, sims)",
        kind="gap",
        missing_capability=("motion.capture", "rig.transform.export",
                            "camera.tracking", "physics.sim.bake"),
        note=("NOTHING on this fleet captures spatial data: there is no mocap "
              "source, no rig exporter, no camera tracker and no simulation "
              "bake. This is not a model shortage, it is an absent pipeline "
              "limb. Wave 5.")),
    LifecyclePoint(
        step=11, point_id="p11-fold2-inference",
        name="Fold 2: geometric inference (pose, depth/normals, flow, scene state)",
        kind="gap",
        missing_capability=("pose.estimate", "optical.flow", "surface.normals",
                            "scene.state.neural", "temporal.correspondence"),
        note=("PARTIAL AND HONESTLY SO. This fleet DOES serve image.depth, "
              "image.detect and image.segment — one model each — so per-frame "
              "boxes, a depth map and a silhouette are obtainable. What is "
              "missing is everything that makes it a FOLD: pose, surface "
              "normals, optical flow, neural scene state, and any temporal "
              "binding across frames. A depth map per frame is not geometric "
              "inference over a shot, and scoring it as if it were would be "
              "the exact overclaim this map exists to prevent.")),
    LifecyclePoint(
        step=12, point_id="p12-segment-specs",
        name="Compile ALL segment specs as siblings of the locked artifacts",
        kind="llm", operations=("segment.compile-prompt",),
        capability="text.chat",
        expectations=_LLM_COMMON + (
            PointExpectation("covers_derived_shot_ids",
                             "one compiled prompt per derived shot id"),
            PointExpectation("sibling_invariant",
                             "no prompt refers to a previous clip, to 'as "
                             "before', or to another segment's location — "
                             "spec[i+1] is never derived from spec[i]"),
            PointExpectation("self_contained",
                             "each prompt could be handed to a render model "
                             "with no other context", layer="judge"),
        )),
    LifecyclePoint(
        step=12, point_id="p12r-keyframe",
        name="Segment render seed: the keyframe every segment spec implies",
        kind="image", operations=("keyframe.render",),
        capability="image.generate",
        expectations=(
            PointExpectation("produced_a_frame",
                             "a readable image file of the requested geometry "
                             "came back"),
            PointExpectation("not_blank",
                             "the frame carries pixel variance — a flat or "
                             "black image scores 0 with EMPTY_OUTPUT"),
            PointExpectation("adheres_to_shot",
                             "a VLM judge that is never the candidate reads "
                             "the shot's required elements off the frame",
                             layer="judge"),
        )),
    LifecyclePoint(
        step=13, point_id="p13-render",
        name="Fold 3: stylistic rendering — the clip itself",
        kind="video", operations=("clip.render",),
        capability="video.generate.t2v",
        expectations=(
            PointExpectation("produced_a_clip",
                             "a readable video file with more than one frame "
                             "came back"),
            PointExpectation("geometry_honoured",
                             "the clip's resolution and frame count are the "
                             "ones that were requested"),
            PointExpectation("adheres_to_shot",
                             "a VLM judge reads the shot's required elements "
                             "off a frame lifted from the clip", layer="judge"),
        )),
    LifecyclePoint(
        step=13, point_id="p13t3-style-controls",
        name="Fold 3 Tier 3: CFG schedule, geometric-control strength, temporal controls",
        kind="gap",
        missing_capability=("render.control.geometric", "render.control.temporal",
                            "render.schedule.cfg", "render.material.override"),
        note=("The studio spine renders a clip from a prompt, a seed and a "
              "geometry. It exposes NO controlnet-class geometric conditioning, "
              "NO temporal control (no motion module route that is not "
              "weights-absent), NO per-step CFG schedule and NO material or "
              "lighting override. video.generate.motion and "
              "video.generate.keyframe both enumerate models whose weights are "
              "0 bytes on the shared store, so the tier is unserved rather "
              "than merely unwired.")),
    LifecyclePoint(
        step=14, point_id="p14-validate",
        name="Validate every result (adherence, identity, tone, violations)",
        kind="vlm", operations=("frame.validate",),
        capability="image.understand",
        expectations=(
            PointExpectation("answers_the_form",
                             "the judge returns a parseable "
                             "VERDICT/SCORE/WHY, once, per frame"),
            PointExpectation("grounded",
                             "the answer is about THIS shot spec — it names "
                             "the elements it was asked about rather than "
                             "describing a picture in general"),
            PointExpectation("catches_planted",
                             "the four frames with planted violations are "
                             "failed and the two compliant frames are passed "
                             "(key basis: the render PROMPT — see "
                             "REFERENCE_FRAME_KEY_BASIS)"),
        )),
    LifecyclePoint(
        step=15, point_id="p15-correct",
        name="Correct or regenerate FROM the locked spec with correction data",
        kind="llm", operations=("correction.notes",), capability="text.chat",
        expectations=_LLM_COMMON + (
            PointExpectation("one_note_per_failure",
                             "every FAIL row in the rejected validation report "
                             "gets a correction entry, and the PASS rows get "
                             "none"),
            PointExpectation("references_locked_spec",
                             "each correction quotes the LOCKED shot/character "
                             "value it is restoring, not the rejected output"),
            PointExpectation("no_prompt_chaining",
                             "the answer never asks for, quotes or builds on "
                             "the rejected attempt's own prompt"),
            PointExpectation("actionable",
                             "a regeneration driven by these notes would "
                             "plausibly fix the frame", layer="judge"),
        )),
    LifecyclePoint(
        step=16, point_id="p16-postproduction",
        name="Post-production and export",
        kind="llm", operations=("postproduction.plan",), capability="text.chat",
        expectations=_LLM_COMMON + (
            PointExpectation("timeline_partition",
                             "the timeline is ordered, gapless and "
                             "non-overlapping over the fixed durations"),
            PointExpectation("covers_every_segment",
                             "every supplied segment id appears exactly once"),
            PointExpectation("names_the_target",
                             "the export block answers the one delivery target "
                             "it was given"),
            PointExpectation("deliverable",
                             "an editor could conform this plan as written",
                             layer="judge"),
        )),
    LifecyclePoint(
        step=16, point_id="p16a-speak",
        name="Post-production audio: speak the locked line",
        kind="tts", operations=("line.speak",), capability="audio.tts",
        expectations=(
            PointExpectation("produced_audio",
                             "a readable wav came back"),
            PointExpectation("carries_sound",
                             "the wav's peak amplitude clears the silence "
                             "floor — a silent wav scores 0 with EMPTY_OUTPUT "
                             "no matter how correct its duration is"),
            PointExpectation("says_the_line",
                             "a round-trip ASR pass finds the locked line's "
                             "words, in order, within the miss budget"),
        ),
        note=("The lifecycle's sixteen steps do not name TTS as a step of its "
              "own; dialogue synthesis is where post-production begins and is "
              "ALSO the step-9 lock prerequisite (k114's lock refuses without "
              "an audio master). It is filed under 16 and this note is the "
              "disclosure that the filing is a judgement call.")),
)

#: point_id -> point.
POINTS_BY_ID: dict[str, LifecyclePoint] = {p.point_id: p
                                           for p in LIFECYCLE_POINTS}

#: The routing-matrix operation keys this scenario introduces or reuses, in
#: lifecycle order. ``derive_matrix`` keys on exactly these strings.
STATIONARY_OPERATIONS: tuple[str, ...] = tuple(
    op for point in LIFECYCLE_POINTS for op in point.operations)

#: operation -> the point that owns it. One owner per operation, asserted.
POINT_FOR_OPERATION: dict[str, str] = {}
for _point in LIFECYCLE_POINTS:
    for _op in _point.operations:
        if _op in POINT_FOR_OPERATION:
            raise RuntimeError(
                f"operation {_op!r} is claimed by both "
                f"{POINT_FOR_OPERATION[_op]} and {_point.point_id} — a "
                f"routing-matrix key with two owners cannot produce a "
                f"per-point verdict")
        POINT_FOR_OPERATION[_op] = _point.point_id
del _point, _op


def points_for_kind(kind: str) -> tuple[LifecyclePoint, ...]:
    """Every point of one kind, in lifecycle order."""
    return tuple(p for p in LIFECYCLE_POINTS if p.kind == kind)


def gap_points() -> tuple[LifecyclePoint, ...]:
    """The points this fleet cannot serve at all. The gap IS the data."""
    return points_for_kind("gap")


# ---------------------------------------------------------------------------
# Digests — the comparability guarantee, computable with nothing running
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str, ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scenario_parts() -> dict[str, Any]:
    """Every piece of the stationary prompt, as plain JSON-safe data.

    This is what the digest is taken over and what a run dir serializes, so an
    operator reading an old run can reconstruct the exact brief every model
    saw without this module being importable at all."""
    return {
        "version": SCENARIO_VERSION,
        "title": SCENARIO_TITLE,
        "premise_fragment": PREMISE_FRAGMENT,
        "screenplay_excerpt": SCREENPLAY_EXCERPT,
        "supplied_lines": list(SUPPLIED_LINES),
        "supplied_beats": list(SUPPLIED_BEATS),
        "character_sheets": [c.to_dict() for c in CHARACTER_SHEETS],
        "continuity_facts": [f.to_dict() for f in CONTINUITY_FACTS],
        "shot_request": SHOT_REQUEST.to_dict(),
        "tone": TONE,
        "keyframe_prompt": KEYFRAME_PROMPT,
        "keyframe_negative_prompt": KEYFRAME_NEGATIVE_PROMPT,
        "clip_prompt": CLIP_PROMPT,
        "tts_line": TTS_LINE,
        "tts_speaker": TTS_SPEAKER,
        "geometry": {"keyframe_width": KEYFRAME_WIDTH,
                     "keyframe_height": KEYFRAME_HEIGHT,
                     "keyframe_seed": KEYFRAME_SEED,
                     "clip_width": CLIP_WIDTH, "clip_height": CLIP_HEIGHT,
                     "clip_fps": CLIP_FPS, "clip_seed": CLIP_SEED},
        "reference_frames": [f.to_dict() for f in REFERENCE_FRAMES],
        "reference_frame_key_basis": REFERENCE_FRAME_KEY_BASIS,
        "rejected_validation_report": REJECTED_VALIDATION_REPORT,
        "segment_durations_s": [list(row) for row in SEGMENT_DURATIONS_S],
        "delivery_target": DELIVERY_TARGET,
        "screenplay": SCENARIO_SCREENPLAY.to_dict(),
        "screenplay_digest": SCENARIO_SCREENPLAY.digest,
        "lifecycle_points": [p.to_dict() for p in LIFECYCLE_POINTS],
    }


def scenario_digest() -> str:
    """``sha256:<16 hex>`` over every piece of the brief.

    Deterministic across processes and days: no clock, no environment, no set
    iteration order (everything is a tuple or is sorted by ``_canonical``).
    Recorded in every result row — two rows with different scenario digests are
    answers to different questions and the reports never mix them."""
    return "sha256:" + _sha256(_canonical(scenario_parts()))[:16]


def part_digests() -> dict[str, str]:
    """A per-piece digest, so a report can say WHICH piece moved when the
    scenario digest changes. Cheap, and the alternative is a diff by eye."""
    parts = scenario_parts()
    return {name: "sha256:" + _sha256(_canonical(value))[:12]
            for name, value in sorted(parts.items())}


# ---------------------------------------------------------------------------
# Derived answer keys — computed from the artifact, never typed
# ---------------------------------------------------------------------------


def derived_shot_ids() -> tuple[str, ...]:
    """The shot ids k110's own ``build_shot_plan`` derives from the canonical
    screenplay. The answer key for steps 7 and 12 is THIS, not a list a human
    wrote next to the screenplay and forgot to update."""
    from hugpy_oracle.screenplay import build_shot_plan
    return tuple(build_shot_plan(SCENARIO_SCREENPLAY).segment_ids)


def derived_continuity() -> Any:
    """The ``ContinuityBible`` k110 derives from the canonical screenplay — the
    answer key step 5 is scored against."""
    from hugpy_oracle.screenplay import build_continuity
    return build_continuity(SCENARIO_SCREENPLAY)


def shot_request_is_derivable() -> tuple[bool, str]:
    """Is :data:`SHOT_REQUEST`'s ``shot_id`` a shot the pipeline would actually
    produce from this screenplay?

    Returns ``(ok, detail)`` rather than raising: this is asserted by a test,
    and a scenario whose hero shot does not exist in its own shot plan is a
    scenario bug that should fail loudly in CI, not at 3am inside a sweep."""
    ids = derived_shot_ids()
    if SHOT_REQUEST.shot_id in ids:
        return True, (f"{SHOT_REQUEST.shot_id} is shot "
                      f"{ids.index(SHOT_REQUEST.shot_id) + 1} of "
                      f"{len(ids)} derived from the canonical screenplay")
    return False, (f"SHOT_REQUEST.shot_id {SHOT_REQUEST.shot_id!r} is not in "
                   f"the derived shot ids {list(ids)} — the hero shot every "
                   f"visual model renders does not exist in this scenario's "
                   f"own shot plan")


def segment_duration_ids() -> tuple[str, ...]:
    return tuple(row[0] for row in SEGMENT_DURATIONS_S)


# ---------------------------------------------------------------------------
# The prompt builders — one per operation, all derived from the SAME pieces
# ---------------------------------------------------------------------------


def _character_block() -> str:
    lines = ["CHARACTER SHEETS:"]
    for sheet in CHARACTER_SHEETS:
        lines.append(f"  {sheet.name} — {sheet.role}")
        lines.append(f"    always visible: {sheet.visual_anchor}")
        lines.append(f"    wardrobe: {', '.join(sheet.wardrobe)}")
        lines.append(f"    props: {', '.join(sheet.props)}")
        if sheet.voice:
            lines.append(f"    voice: {sheet.voice}")
    return "\n".join(lines)


def _facts_block() -> str:
    return "CONTINUITY FACTS (standing, for the whole film):\n" + "\n".join(
        f"  {f.fact_id} [{f.scope}] {f.statement}" for f in CONTINUITY_FACTS)


#: The extra inputs each stationary operation needs BEYOND the locked
#: screenplay, keyed by operation. The instruction, the row shape and the JSON
#: example are NOT here: they live on the ``OperationSpec`` in
#: ``benchmark_cases.py``, which is what ``build_workflow_prompt`` already
#: renders. Duplicating them here would give the sweep a second, drifting copy
#: of the question — the exact fault k109's "reuse the pipeline's own prompt"
#: rule exists to prevent.
_PREAMBLE_EXTRAS: dict[str, str] = {}


def stationary_preamble(operation: str) -> str:
    """The scenario block prepended to one operation's prompt.

    EVERY operation gets the same three pieces — tone, character sheets,
    continuity facts — so no model is ever asked a better-briefed question than
    another. Two operations get one extra fixed input each, because they cannot
    be asked at all without it: step 15 needs the rejected validation report
    (the SAME report for every model, or "who corrects best" measures who got
    the easier rejection), and step 16 needs the fixed segment durations and
    the one delivery target.

    Raises ``KeyError`` for an operation this scenario does not brief — a
    silent empty preamble would let a new operation be swept against a
    different question than its peers and nothing would say so."""
    if operation not in STATIONARY_BRIEFED_OPERATIONS:
        raise KeyError(
            f"operation {operation!r} has no stationary preamble; briefed: "
            f"{sorted(STATIONARY_BRIEFED_OPERATIONS)}")
    blocks = [f"THE FILM: {SCENARIO_TITLE} (stationary scenario "
              f"{SCENARIO_VERSION}). Everything you need is below. Do not "
              f"invent a character, a location or an id that is not here.",
              "", f"TONE (fixed for the whole film): {TONE}", "",
              _character_block(), "", _facts_block()]
    if operation == "correction.notes":
        blocks += ["", "LOCKED SHOT SPEC (the only thing you correct FROM):",
                   SHOT_REQUEST.spec_text, "", REJECTED_VALIDATION_REPORT]
    if operation == "postproduction.plan":
        blocks += ["", "SEGMENT DURATIONS (fixed — plan over exactly these):",
                   "\n".join(f"  {sid}: {dur}s"
                             for sid, dur in SEGMENT_DURATIONS_S),
                   "", f"DELIVERY TARGET: {DELIVERY_TARGET}"]
    return "\n".join(blocks)


#: The operations :func:`stationary_preamble` will brief. Every LLM operation
#: in the point map, and only those: the media operations are not prompted
#: through the workflow prompt builder at all.
STATIONARY_BRIEFED_OPERATIONS: frozenset = frozenset({
    "plot.construct", "screenplay.complete", "continuity.bible",
    "screenplay.breakdown", "shots.design", "segment.compile-prompt",
    "correction.notes", "postproduction.plan",
})


def build_frame_judge_prompt(frame: ReferenceFrame | None = None) -> str:
    """What every VLM judge is asked about every reference frame.

    IDENTICAL for all sixteen judges and all six frames — the frame varies, the
    question does not. ``frame`` is accepted only so a caller can be explicit;
    the prompt deliberately does NOT mention which frame it is, because a judge
    told 'this one has a planted fault' would find one."""
    del frame  # the question is frame-independent, on purpose
    return (
        "You are validating ONE frame against ONE locked shot specification "
        "from a film. Look at the image.\n\n"
        f"{SHOT_REQUEST.spec_text}\n\n"
        "Decide whether the frame ADHERES to that specification. Name every "
        "element from MUST SHOW that is absent, and every element from MUST "
        "NOT SHOW that is present. Judge only what is in the picture.\n\n"
        "Reply exactly: VERDICT=YES|NO; SCORE=0-100; WHY=<one sentence naming "
        "the specific elements you checked>."
    )


def build_keyframe_judge_prompt() -> str:
    """What the VLM judge is asked about a CANDIDATE's rendered keyframe.

    Same shot spec, same reply discipline as the frame-validation prompt — the
    judge must not be able to tell a reference frame from a candidate render,
    or its strictness becomes a function of what it thinks it is grading."""
    return build_frame_judge_prompt()


def build_clip_judge_prompt() -> str:
    """What the VLM judge is asked about a frame lifted out of a rendered clip.

    The extra sentence is the ONLY difference, and it exists because a frame
    pulled from a 4-second generated clip is legitimately softer than a still
    and a judge that does not know that marks every clip model down for being
    a clip model."""
    return build_frame_judge_prompt() + (
        "\n\nThis frame was lifted from the middle of a short generated video "
        "clip, so it may be softer or noisier than a still. Judge the CONTENT "
        "against the specification, not the sharpness."
    )


# ---------------------------------------------------------------------------
# Deterministic validators for the six new operations
# ---------------------------------------------------------------------------
#
# Each returns ``(errors, facts)``. ``errors`` empty means the artifact is
# structurally what the point asked for; ``facts`` are the countable
# observations the scorer turns into axes. Nothing here asks a model anything.


def _rows(obj: Any, container: str) -> list[Mapping[str, Any]]:
    if not isinstance(obj, Mapping):
        return []
    value = obj.get(container)
    if not isinstance(value, list):
        return []
    return [r for r in value if isinstance(r, Mapping)]


_CHAIN_MARKERS: tuple[str, ...] = (
    "previous clip", "previous segment", "previous shot", "as before",
    "the last shot", "continuing from", "same as the", "earlier clip",
    "as in the previous", "from the prior",
)


def check_no_prompt_chaining(texts: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    """How many of ``texts`` chain off a sibling, and which markers fired.

    A SUBSTRING test, and named as one. It cannot catch a prompt that chains
    implicitly ("she turns back to him") and it is not claimed to: it catches
    the explicit forms, which is what the sibling invariant is most often
    broken by and is the only part that is honestly countable."""
    hits = 0
    seen: list[str] = []
    for text in texts:
        low = str(text or "").lower()
        fired = [m for m in _CHAIN_MARKERS if m in low]
        if fired:
            hits += 1
            seen.extend(fired)
    return hits, tuple(sorted(set(seen)))


def validate_correction_notes(obj: Any) -> tuple[tuple[str, ...],
                                                 dict[str, Any]]:
    """Step 15: one correction per FAILING check, none for the passing ones."""
    failing = ("adherence.time_of_day", "adherence.wardrobe",
               "adherence.prop", "tone")
    passing = ("adherence.cast_count", "adherence.location")
    rows = _rows(obj, "corrections")
    errors: list[str] = []
    if not rows:
        errors.append("no 'corrections' array of objects in the reply")
    named = [str(r.get("check") or "").strip().lower() for r in rows]
    covered = [c for c in failing if any(c in n or n in c for n in named if n)]
    spurious = [c for c in passing if any(c in n for n in named if n)]
    for check in failing:
        if check not in covered:
            errors.append(f"no correction for failing check {check!r}")
    for check in spurious:
        errors.append(f"a correction was written for {check!r}, which PASSED")
    empty = [r for r in rows if not str(r.get("correction") or "").strip()]
    if empty:
        errors.append(f"{len(empty)} correction row(s) carry no correction text")
    chained, markers = check_no_prompt_chaining(
        [str(r.get("correction") or "") for r in rows])
    if chained:
        errors.append(f"{chained} correction(s) refer to the rejected attempt "
                      f"or a sibling ({', '.join(markers)})")
    locked = sum(1 for r in rows if str(r.get("locked_value") or "").strip())
    return tuple(errors), {
        "rows": len(rows), "covered_failing": len(covered),
        "failing_total": len(failing), "spurious": len(spurious),
        "locked_value_cited": locked, "chained": chained,
        "accuracy": (len(covered) - len(spurious)) / len(failing)
        if failing else None,
    }


def validate_timeline(obj: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Step 16: the timeline is a partition of the fixed durations, or it is
    not. Gaps and overlaps are arithmetic, not opinion."""
    rows = _rows(obj, "timeline")
    errors: list[str] = []
    if not rows:
        errors.append("no 'timeline' array of objects in the reply")
        return tuple(errors), {"rows": 0, "accuracy": 0.0}

    expected = dict(SEGMENT_DURATIONS_S)
    seen: list[str] = []
    cursor = 0.0
    gaps = overlaps = bad_length = 0
    for row in rows:
        sid = str(row.get("segment_id") or "")
        seen.append(sid)
        try:
            start = float(row.get("start_s"))
            end = float(row.get("end_s"))
        except (TypeError, ValueError):
            errors.append(f"segment {sid!r} has an unreadable window")
            continue
        if end <= start:
            errors.append(f"segment {sid!r} window is not forward "
                          f"({start} -> {end})")
            continue
        if abs(start - cursor) > 0.051:
            if start > cursor:
                gaps += 1
            else:
                overlaps += 1
        want = expected.get(sid)
        if want is not None and abs((end - start) - want) > 0.051:
            bad_length += 1
        cursor = max(cursor, end)

    missing = [sid for sid in expected if sid not in seen]
    invented = [sid for sid in seen if sid and sid not in expected]
    duplicated = [sid for sid in set(seen) if seen.count(sid) > 1]
    if missing:
        errors.append(f"segment(s) missing from the timeline: {missing}")
    if invented:
        errors.append(f"invented segment id(s): {invented[:4]}")
    if duplicated:
        errors.append(f"segment(s) laid twice: {sorted(duplicated)[:4]}")
    if gaps:
        errors.append(f"{gaps} gap(s) between windows")
    if overlaps:
        errors.append(f"{overlaps} overlapping window(s)")
    if bad_length:
        errors.append(f"{bad_length} window(s) do not match the fixed duration")

    export = obj.get("export") if isinstance(obj, Mapping) else None
    export_ok = isinstance(export, Mapping) and bool(export)
    if not export_ok:
        errors.append("no 'export' block answering the delivery target")

    total = max(1, len(expected))
    hits = len([s for s in expected if s in seen])
    return tuple(errors), {
        "rows": len(rows), "covered": hits, "expected": total,
        "gaps": gaps, "overlaps": overlaps, "bad_length": bad_length,
        "invented": len(invented), "export_present": export_ok,
        "accuracy": max(0.0, (hits - gaps - overlaps - bad_length
                              - len(invented)) / total),
    }


__all__ = [
    "CHARACTER_SHEETS", "CLIP_FPS", "CLIP_HEIGHT", "CLIP_PROMPT", "CLIP_SEED",
    "CLIP_WIDTH", "CONTINUITY_FACTS", "CharacterSheet", "ContinuityFact",
    "DELIVERY_TARGET", "KEYFRAME_HEIGHT", "KEYFRAME_NEGATIVE_PROMPT",
    "KEYFRAME_PROMPT", "KEYFRAME_SEED", "KEYFRAME_WIDTH", "LIFECYCLE_POINTS",
    "LINE_1", "LINE_2", "LINE_3", "LifecyclePoint",
    "POINTS_BY_ID", "POINT_FOR_OPERATION", "POINT_KINDS", "PREMISE_FRAGMENT",
    "PointExpectation", "REFERENCE_FRAMES", "REFERENCE_FRAME_KEY_BASIS",
    "REJECTED_VALIDATION_REPORT", "ReferenceFrame", "SCENARIO_SCREENPLAY",
    "SCENARIO_TITLE", "SCENARIO_VERSION", "SCREENPLAY_EXCERPT",
    "SEGMENT_DURATIONS_S", "SHOT_REQUEST", "STATIONARY_OPERATIONS",
    "STATIONARY_BRIEFED_OPERATIONS", "SUPPLIED_BEATS", "SUPPLIED_LINES",
    "ShotRequest", "TONE", "TTS_LINE",
    "TTS_SPEAKER", "VIOLATION_LABELS", "build_clip_judge_prompt",
    "build_frame_judge_prompt", "build_keyframe_judge_prompt",
    "check_no_prompt_chaining", "derived_continuity",
    "derived_shot_ids", "gap_points", "part_digests", "points_for_kind",
    "scenario_digest", "scenario_parts", "segment_duration_ids",
    "shot_request_is_derivable", "stationary_preamble",
    "validate_correction_notes",
    "validate_timeline",
]
