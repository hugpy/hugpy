"""k110 — PlotSpec / Screenplay artifacts, mechanical continuity extraction,
shot-plan authoring, and LLM-assisted construction with hard validation.

Everything here is OFFLINE and deterministic: no catalog, no worker, no GPU, no
network, no clock, no disk. The "LLM" is a function that returns strings from a
list and records the prompts it was handed, which is the whole point of the
injected-callable seam — the authoring contract is testable without an
inference stack, and the live binding is one monkeypatched route away.

Locks:
  [1] wire shape: every artifact round-trips through to_dict/from_dict without
      loss, digests are stable across rebuilds and byte-compatible with k104's
      canonical JSON.
  [2] Stage 5: a plot is refused unless every beat names a character, every
      causal link points at an existing EARLIER beat, and no character is an
      orphan — and ALL the problems are reported, not just the first.
  [3] Stage 6: a speaker must be in the room (with entrances and exits tracked
      scene to scene, including inheritance across CONTINUOUS:), story time may
      not run backwards without a flashback transition, and a line id is unique
      across the whole screenplay. to_dialogue_timeline is a 1:1 collection.
  [4] Stage 7: continuity is DERIVED, never authored — same screenplay, same
      bible, same digest — and the carried world facts chain exactly from one
      segment's state_after to the next one's state_before.
  [5] Stage 9: audio-first windows are read off the AudioMaster; without one
      every window is flagged estimated=True and none of them is ever allowed
      to become the timing the audio is retimed to fit.
  [6] authoring: valid -> artifact; invalid -> ONE reprompt carrying the
      validator's own words -> artifact; still invalid -> a typed AuthoringGap
      with the raw reply preserved. Never a coerced artifact.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_screenplay.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

logging.disable(logging.WARNING)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import production as production_mod
from hugpy_oracle.audio_master import AudioMaster, DialogueTimeline, Line, LineTiming
from hugpy_oracle.production import (
    CAMERA_KEYS,
    CAMERA_MOVES,
    CAMERA_VIEWS,
    SHOT_SIZES,
    ContinuityBible,
    GenerationSnapshot,
    LockRefused,
    ProductionLock,
    ShotPlan,
    ShotPlanEntry,
)
from hugpy_oracle.schema_export import json_schema_for
from hugpy_oracle.screenplay import (
    AUTHORING_CAPABILITY,
    AVEvent,
    AV_EVENT_KINDS,
    AuthoringGap,
    Beat,
    CARRIED_KEYS,
    CONTINUOUS_TRANSITION,
    Character,
    FLASHBACK_TRANSITIONS,
    INPUT_MODES,
    PlotRefused,
    PlotSpec,
    SCENE_PREFIXES,
    STATE_KEYS,
    SUBPLAN_STEPS,
    Scene,
    Screenplay,
    ScreenplayRefused,
    ShotDesign,
    ShotPlanDraft,
    ShotPlanRefused,
    TRANSITIONS,
    author_plot,
    author_screenplay,
    bind_llm,
    build_continuity,
    build_plot_prompt,
    build_repair_prompt,
    build_screenplay_prompt,
    build_shot_plan,
    chain_breaks,
    estimated_line_seconds,
    lock_production,
    make_heading,
    parse_json_object,
    plot_input_mode,
    props_in_play,
    quantize,
    require_text,
    schema_block,
    str_tuple,
)


# ---------------------------------------------------------------------------
# Fixtures — one small film, hand-built, used by most of the file.
# ---------------------------------------------------------------------------


def make_plot(**overrides) -> PlotSpec:
    base = dict(
        premise="Two salvagers disagree about a relic they should not have found.",
        genre="science fiction",
        tone="tense, close, unglamorous",
        pacing="three beats, no slack",
        characters=(
            Character(name="MARA", goal="sell the relic and clear her debt",
                      conflict="Bo wants it returned", arc="greed to doubt"),
            Character(name="BO", goal="put the relic back",
                      conflict="Mara will not let go of it",
                      arc="doubt to resolve"),
        ),
        beats=(
            Beat(beat_id="b1", summary="They lift the relic out of the crate.",
                 characters=("MARA", "BO"), location="SALVAGE HOLD"),
            Beat(beat_id="b2", summary="The argument turns.",
                 characters=("MARA", "BO"), causes=("b1",), turning_point=True),
            Beat(beat_id="b3", summary="Bo walks off the ship.",
                 characters=("BO",), causes=("b2",)),
        ),
        ending="Bo leaves. Mara keeps the relic and is alone with it.",
    )
    base.update(overrides)
    return PlotSpec(**base)


def make_scenes() -> tuple[Scene, Scene, Scene]:
    first = Scene(
        scene_id="sc1", heading="INT. SALVAGE HOLD - NIGHT",
        location="SALVAGE HOLD", time_of_day="NIGHT",
        action="MARA lifts the relic out of the crate.",
        staging="Mara camera left, Bo camera right, the crate between them.",
        present_at_open=("MARA", "BO"),
        dialogue=(
            Line(line_id="l1", speaker="MARA", text="Look at this thing.",
                 emotion="awed"),
            Line(line_id="l2", speaker="BO",
                 text="Put it back where you found it.", emotion="wary"),
        ),
        av_events=(AVEvent(kind="ambience", description="hull creak",
                           cue="under the whole scene"),),
        props=("relic", "crate"), wardrobe=("MARA: grey coveralls",),
        lighting="one work lamp, hard from above", weather="",
        transition=CONTINUOUS_TRANSITION, story_time_s=0.0, beat_id="b1")
    second = Scene(
        scene_id="sc2", heading="INT. SALVAGE HOLD - NIGHT",
        location="SALVAGE HOLD", time_of_day="NIGHT",
        action="The relic sits on the deck between them.",
        exits=("BO",),
        dialogue=(Line(line_id="l3", speaker="BO",
                       text="I am not doing this again.", emotion="flat"),),
        lighting="one work lamp, hard from above",
        transition="CUT TO:", story_time_s=40.0, beat_id="b2")
    third = Scene(
        scene_id="sc3", heading="EXT. DOCK RAMP - NIGHT", location="DOCK RAMP",
        time_of_day="NIGHT",
        action="BO walks away down the ramp; the relic is still in the hold.",
        entrances=("BO",), transition="FADE OUT.", story_time_s=120.0,
        beat_id="b3")
    return first, second, third


def make_screenplay(**overrides) -> Screenplay:
    base = dict(title="SALVAGE", scenes=make_scenes(),
                logline="Two salvagers, one relic, no good options.")
    base.update(overrides)
    return Screenplay(**base)


def make_master(play: Screenplay, *, locked: bool = True) -> AudioMaster:
    timeline = play.to_dialogue_timeline(locked=True)
    timings = (
        LineTiming(line_id="l1", start_s=0.5, end_s=2.0, pause_after_s=0.3),
        LineTiming(line_id="l2", start_s=2.3, end_s=4.1, pause_after_s=0.4),
        LineTiming(line_id="l3", start_s=4.5, end_s=6.0, pause_after_s=0.5),
    )
    master = AudioMaster(
        timeline_digest=timeline.digest, line_timings=timings,
        tracks=(("l1", "/a/1.wav"), ("l2", "/a/2.wav"), ("l3", "/a/3.wav")),
        total_seconds=8.0, candidates_considered=3)
    return master.lock() if locked else master


class FakeLlm:
    """An injected model: canned replies out, prompts recorded in.

    Records every prompt so the reprompt assertions can read what the model was
    actually told, which is the only way to prove the repair round carries the
    validator's words rather than a summary of them."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("the fake llm was called more times than it "
                                 "has replies — authoring is supposed to be "
                                 "bounded at two attempts")
        return self.replies.pop(0)


def plot_json(**overrides) -> str:
    payload = make_plot().to_dict()
    payload.update(overrides)
    return json.dumps(payload)


def screenplay_json(**overrides) -> str:
    payload = make_screenplay().to_dict()
    payload.update(overrides)
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# [1] wire shape, digests, and the borrowed helpers
# ---------------------------------------------------------------------------


def test_plot_round_trips_without_loss():
    plot = make_plot()
    assert PlotSpec.from_dict(plot.to_dict()) == plot
    assert PlotSpec.from_dict(plot.to_dict()).digest == plot.digest


def test_screenplay_round_trips_without_loss():
    play = make_screenplay()
    assert Screenplay.from_dict(play.to_dict()) == play
    assert Screenplay.from_dict(play.to_dict()).digest == play.digest


def test_shot_plan_draft_round_trips_without_loss():
    draft = build_shot_plan(make_screenplay())
    assert ShotPlanDraft.from_dict(draft.to_dict()) == draft
    assert ShotPlanDraft.from_dict(draft.to_dict()).digest == draft.digest


def test_authoring_gap_round_trips_and_needs_a_reason():
    gap = AuthoringGap(errors=("nope",), raw="{", stage="plot",
                       code="AUTHORING_UNPARSED", attempts=2,
                       raw_attempts=("{", "{"))
    assert AuthoringGap.from_dict(gap.to_dict()) == gap
    with pytest.raises(ValueError, match="at least one error"):
        AuthoringGap(errors=(), raw="x")
    with pytest.raises(ValueError, match="not one of"):
        AuthoringGap(errors=("x",), code="MADE_UP")


def test_digests_are_clock_free_and_rebuild_identical():
    """An id that changed because you built the artifact twice is not an id."""
    assert make_plot().digest == make_plot().digest
    assert make_screenplay().digest == make_screenplay().digest
    assert build_shot_plan(make_screenplay()).digest == \
        build_shot_plan(make_screenplay()).digest


def test_canonical_bytes_are_k104_canonical_json():
    play = make_screenplay()
    assert play.canonical_bytes == production_mod.canonical_json(play.to_dict())


def test_validation_helpers_are_k104s_and_not_a_fourth_copy():
    """One tree, one "non-empty string" rule. If this fails, two modules have
    started disagreeing about what an empty field is."""
    assert require_text is production_mod._require_text
    assert str_tuple is production_mod._str_tuple
    assert quantize is production_mod._q


def test_screenplay_lock_is_a_new_artifact_not_a_mutation():
    play = make_screenplay()
    locked = play.lock()
    assert locked.locked is True and play.locked is False
    assert locked.digest != play.digest
    assert locked.lock() is locked


# ---------------------------------------------------------------------------
# [2] Stage 5 — plot validation
# ---------------------------------------------------------------------------


def test_valid_plot_reports_its_structure():
    plot = make_plot()
    assert plot.character_names == ("MARA", "BO")
    assert plot.beat_ids == ("b1", "b2", "b3")
    assert plot.turning_points == ("b2",)
    assert plot.roots == ("b1",)
    assert plot.beats_for("BO") == ("b1", "b2", "b3")
    assert plot.beat("b2").turning_point is True
    assert plot.input_mode == "complete"


def test_beat_with_no_character_is_refused():
    with pytest.raises(PlotRefused) as excinfo:
        make_plot(beats=(
            Beat(beat_id="b1", summary="A storm arrives.", characters=()),
            Beat(beat_id="b2", summary="They argue.",
                 characters=("MARA", "BO"), causes=("b1",)),
        ))
    assert any("names no character" in e and "b1" in e
               for e in excinfo.value.errors)


def test_causal_link_to_a_beat_that_does_not_exist_is_refused():
    with pytest.raises(PlotRefused) as excinfo:
        make_plot(beats=(
            Beat(beat_id="b1", summary="They find it.",
                 characters=("MARA", "BO")),
            Beat(beat_id="b2", summary="They argue.",
                 characters=("MARA", "BO"), causes=("b7",)),
            Beat(beat_id="b3", summary="Bo leaves.", characters=("BO",),
                 causes=("b2",)),
        ))
    assert any("'b7'" in e and "not a beat in this plot" in e
               for e in excinfo.value.errors)


def test_causal_link_to_a_later_beat_is_refused():
    """PlotSpec.beats is the CAUSAL order. Screen order, flashbacks included,
    is the screenplay's business."""
    with pytest.raises(PlotRefused) as excinfo:
        make_plot(beats=(
            Beat(beat_id="b1", summary="They find it.",
                 characters=("MARA", "BO"), causes=("b2",)),
            Beat(beat_id="b2", summary="They argue.",
                 characters=("MARA", "BO")),
            Beat(beat_id="b3", summary="Bo leaves.", characters=("BO",),
                 causes=("b2",)),
        ))
    assert any("comes AFTER it" in e for e in excinfo.value.errors)


def test_a_beat_cannot_cause_itself():
    with pytest.raises(PlotRefused) as excinfo:
        make_plot(beats=(
            Beat(beat_id="b1", summary="They find it.",
                 characters=("MARA", "BO"), causes=("b1",)),
            Beat(beat_id="b2", summary="Bo leaves.", characters=("BO",),
                 causes=("b1",)),
        ))
    assert any("causes itself" in e for e in excinfo.value.errors)


def test_orphan_character_is_refused():
    with pytest.raises(PlotRefused) as excinfo:
        make_plot(characters=(
            Character(name="MARA", goal="sell it", conflict="Bo",
                      arc="greed to doubt"),
            Character(name="BO", goal="return it", conflict="Mara",
                      arc="doubt to resolve"),
            Character(name="TESS", goal="collect the debt", conflict="time",
                      arc="patient to done"),
        ))
    assert any("TESS" in e and "appear in no beat" in e
               for e in excinfo.value.errors)


def test_beat_naming_an_undeclared_character_is_refused():
    with pytest.raises(PlotRefused) as excinfo:
        make_plot(beats=(
            Beat(beat_id="b1", summary="They find it.",
                 characters=("MARA", "BO")),
            Beat(beat_id="b2", summary="Tess calls.",
                 characters=("MARA", "TESS"), causes=("b1",)),
            Beat(beat_id="b3", summary="Bo leaves.", characters=("BO",),
                 causes=("b2",)),
        ))
    assert any("TESS" in e and "does not declare" in e
               for e in excinfo.value.errors)


def test_plot_reports_every_problem_at_once():
    """One repair round has to be able to fix everything, so one refusal has to
    be able to name everything."""
    with pytest.raises(PlotRefused) as excinfo:
        make_plot(
            ending="",
            beats=(Beat(beat_id="b1", summary="A storm.", characters=()),),
            characters=(Character(name="MARA", goal="g", conflict="c",
                                  arc="a"),))
    errors = excinfo.value.errors
    assert len(errors) >= 3
    assert any("ending" in e for e in errors)
    assert any("names no character" in e for e in errors)
    assert any("appear in no beat" in e for e in errors)


def test_character_needs_a_goal_a_conflict_and_an_arc():
    with pytest.raises(ValueError, match="Character.goal"):
        Character(name="MARA", goal="", conflict="c", arc="a")


def test_plot_refuses_an_unknown_input_mode():
    with pytest.raises(PlotRefused, match="input_mode"):
        make_plot(input_mode="vibes")


# ---------------------------------------------------------------------------
# [3] Stage 6 — screenplay validation
# ---------------------------------------------------------------------------


def test_scene_refuses_a_speaker_who_is_not_in_the_room():
    with pytest.raises(ScreenplayRefused) as excinfo:
        Scene(scene_id="sc1", heading="INT. HOLD - NIGHT", location="HOLD",
              time_of_day="NIGHT", present_at_open=("MARA",),
              dialogue=(Line(line_id="l1", speaker="BO", text="Hello."),))
    assert "BO" in str(excinfo.value) and "l1" in str(excinfo.value)


def test_screenplay_refuses_a_speaker_who_is_not_in_an_inherited_room():
    """A scene with an empty present_at_open may be INHERITING one, so only the
    screenplay can check it — and it does."""
    first = Scene(scene_id="sc1", heading="INT. HOLD - NIGHT", location="HOLD",
                  time_of_day="NIGHT", present_at_open=("MARA",),
                  dialogue=(Line(line_id="l1", speaker="MARA", text="Hey."),),
                  transition=CONTINUOUS_TRANSITION)
    second = Scene(scene_id="sc2", heading="INT. HOLD - NIGHT", location="HOLD",
                   time_of_day="NIGHT",
                   dialogue=(Line(line_id="l2", speaker="BO", text="Hey."),),
                   story_time_s=10.0)
    with pytest.raises(ScreenplayRefused) as excinfo:
        Screenplay(title="X", scenes=(first, second))
    assert any("BO" in e and "l2" in e for e in excinfo.value.errors)


def test_scene_refuses_an_exit_by_someone_who_was_never_there():
    with pytest.raises(ScreenplayRefused, match="exit without ever being"):
        Scene(scene_id="sc1", heading="INT. HOLD - NIGHT", location="HOLD",
              time_of_day="NIGHT", present_at_open=("MARA",), exits=("BO",))


def test_scene_refuses_an_entrance_by_someone_already_present():
    with pytest.raises(ValueError, match="cannot enter a room they are already"):
        Scene(scene_id="sc1", heading="INT. HOLD - NIGHT", location="HOLD",
              time_of_day="NIGHT", present_at_open=("MARA",),
              entrances=("MARA",))


def test_story_time_may_not_run_backwards_without_a_flashback():
    first, second, third = make_scenes()
    # sc2 sits at 40s and ends on a plain "CUT TO:"; sc3 lands at 5s.
    earlier = Scene.from_dict({**third.to_dict(), "story_time_s": 5.0})
    with pytest.raises(ScreenplayRefused) as excinfo:
        Screenplay(title="X", scenes=(first, second, earlier))
    assert any("BEFORE" in e and "backwards" in e for e in excinfo.value.errors)


def test_a_flashback_transition_licenses_going_backwards():
    first, second, third = make_scenes()
    declaring = Scene.from_dict({**second.to_dict(),
                                 "transition": "FLASHBACK TO:"})
    long_ago = Scene.from_dict({**third.to_dict(), "story_time_s": 5.0})
    play = Screenplay(title="X", scenes=(first, declaring, long_ago))
    assert play.scene("sc3").story_time_s == 5.0
    assert declaring.transition in FLASHBACK_TRANSITIONS


def test_duplicate_line_ids_across_scenes_are_refused():
    first, second, third = make_scenes()
    collide = Scene.from_dict({
        **second.to_dict(),
        "dialogue": [{"line_id": "l1", "speaker": "BO", "text": "Again."}]})
    with pytest.raises(ScreenplayRefused) as excinfo:
        Screenplay(title="X", scenes=(first, collide, third))
    assert any("l1" in e and "more than one scene" in e
               for e in excinfo.value.errors)


def test_a_character_outside_the_declared_cast_is_refused():
    with pytest.raises(ScreenplayRefused) as excinfo:
        make_screenplay(characters=("MARA",))
    assert any("BO" in e and "Screenplay.characters" in e
               for e in excinfo.value.errors)


def test_cast_is_derived_when_not_declared():
    play = make_screenplay()
    assert play.characters == ("MARA", "BO")


def test_a_scene_heading_must_be_a_slugline():
    with pytest.raises(ValueError, match="not a slugline"):
        Scene(scene_id="sc1", heading="The salvage hold, at night",
              location="HOLD", time_of_day="NIGHT")
    assert make_heading("INT.", "kitchen", "night") == "INT. KITCHEN - NIGHT"
    with pytest.raises(ValueError, match="must be one of"):
        make_heading("INSIDE", "kitchen", "night")


def test_closed_vocabularies_refuse_an_invented_token():
    with pytest.raises(ValueError, match="transition"):
        Scene(scene_id="sc1", heading="INT. HOLD - NIGHT", location="HOLD",
              time_of_day="NIGHT", transition="WIPE TO:")
    with pytest.raises(ValueError, match="AVEvent.kind"):
        AVEvent(kind="explosion", description="boom")
    assert "CUT TO:" in TRANSITIONS
    assert FLASHBACK_TRANSITIONS <= TRANSITIONS
    assert CONTINUOUS_TRANSITION in TRANSITIONS
    assert "sfx" in AV_EVENT_KINDS
    assert all(p.endswith(".") for p in SCENE_PREFIXES)


def test_a_continuous_join_may_not_change_the_cast():
    """sc1 ends on CONTINUOUS:, so sc2 may not open on a different room. The
    scene below is internally consistent (TESS is present, so nobody speaks or
    exits out of nowhere) — the ONLY thing wrong with it is the join, which is
    exactly what this check exists to catch."""
    first, second, third = make_scenes()
    reopened = Scene.from_dict({**second.to_dict(),
                                "present_at_open": ["MARA", "BO", "TESS"]})
    with pytest.raises(ScreenplayRefused) as excinfo:
        Screenplay(title="X", scenes=(first, reopened, third))
    assert any("CONTINUOUS" in e and "nothing may change" in e
               for e in excinfo.value.errors)


def test_a_continuous_join_carries_the_cast_forward():
    play = make_screenplay()
    chain = play.presence_chain()
    assert chain[0] == ("sc1", ("MARA", "BO"), ("MARA", "BO"))
    # sc2 declares nobody and inherits sc1's closing cast across CONTINUOUS:,
    # which is what lets BO speak in it.
    assert chain[1] == ("sc2", ("MARA", "BO"), ("MARA",))
    # sc3 is a hard cut: nothing is inherited, BO walks in.
    assert chain[2] == ("sc3", (), ("BO",))
    assert play.open_at(1) == ("MARA", "BO")
    assert play.close_at(1) == ("MARA",)


def test_screenplay_needs_at_least_one_scene():
    with pytest.raises(ScreenplayRefused, match="no scenes"):
        Screenplay(title="X", scenes=())


# ---------------------------------------------------------------------------
# [3b] the 1:1 map onto k102
# ---------------------------------------------------------------------------


def test_to_dialogue_timeline_is_a_one_to_one_collection():
    play = make_screenplay()
    timeline = play.to_dialogue_timeline()
    assert isinstance(timeline, DialogueTimeline)
    assert timeline.line_ids == play.line_ids == ("l1", "l2", "l3")
    # SAME objects, not copies: the ids, speakers, emotions and per-line
    # budgets cannot drift because there is nothing to drift between.
    assert all(a is b for a, b in zip(timeline.lines, play.lines))
    assert timeline.speakers == play.speakers == ("MARA", "BO")
    index = play.line_index()
    assert index == {"l1": ("sc1", 0), "l2": ("sc1", 1), "l3": ("sc2", 2)}
    assert play.scene_of_line("l1") == "sc1"


def test_dialogue_timeline_lock_mirrors_the_screenplay():
    play = make_screenplay()
    assert play.to_dialogue_timeline().locked is False
    assert play.lock().to_dialogue_timeline().locked is True
    assert play.to_dialogue_timeline(locked=True).locked is True


def test_a_silent_screenplay_has_no_dialogue_timeline_to_build():
    silent = Screenplay(title="SILENT", scenes=(
        Scene(scene_id="sc1", heading="EXT. RIDGE - DAWN", location="RIDGE",
              time_of_day="DAWN", action="Wind over stone."),))
    with pytest.raises(ScreenplayRefused, match="no dialogue"):
        silent.to_dialogue_timeline()


# ---------------------------------------------------------------------------
# [4] Stage 7 — continuity, derived
# ---------------------------------------------------------------------------


def test_continuity_covers_every_scene_with_a_before_and_an_after():
    bible = build_continuity(make_screenplay())
    assert isinstance(bible, ContinuityBible)
    assert bible.segment_ids == ("sc1", "sc2", "sc3")
    assert bible.missing(("sc1", "sc2", "sc3")) == ()
    for entry in bible.entries:
        assert set(entry.state_before) == set(STATE_KEYS)
        assert set(entry.state_after) == set(STATE_KEYS)


def test_continuity_tracks_entrances_and_exits():
    bible = build_continuity(make_screenplay())
    assert bible.state("sc1").state_after["present"] == ("BO", "MARA")
    # BO exits during sc2: present at the open, gone at the close.
    assert bible.state("sc2").state_before["present"] == ("BO", "MARA")
    assert bible.state("sc2").state_after["present"] == ("MARA",)
    assert "present" in bible.state("sc2").changed_keys
    # BO enters sc3 (a hard cut into a new location).
    assert bible.state("sc3").state_before["present"] == ()
    assert bible.state("sc3").state_after["present"] == ("BO",)


def test_continuity_tracks_props_including_ones_only_mentioned():
    bible = build_continuity(make_screenplay())
    assert bible.props == ("relic", "crate")
    # sc1 DECLARES both; nothing was established before it.
    assert bible.state("sc1").state_before["props"] == ()
    assert bible.state("sc1").state_after["props"] == ("crate", "relic")
    # sc3 declares no props but its action names the relic, so the relic is in
    # play there — recognized against the closed inventory, never invented.
    assert bible.state("sc3").state_after["props"] == ("relic",)
    assert props_in_play(make_scenes()[2], ("relic", "crate")) == ("relic",)


def test_prop_recognition_is_whole_token():
    """"the knifemaker" has not put a knife on the table."""
    scene = Scene(scene_id="sc1", heading="INT. SHOP - DAY", location="SHOP",
                  time_of_day="DAY", action="She visits the knifemaker.")
    assert props_in_play(scene, ("knife",)) == ()


def test_carried_state_chains_exactly_from_after_to_before():
    bible = build_continuity(make_screenplay())
    for first, second in zip(bible.entries, bible.entries[1:]):
        for key in CARRIED_KEYS:
            assert first.state_after[key] == second.state_before[key], key
    assert chain_breaks(bible) == ()


def test_chain_breaks_finds_a_reordered_bible():
    """A continuity bible whose entries were shuffled reads as perfectly
    well-formed; the carried world facts are what give it away."""
    bible = build_continuity(make_screenplay())
    shuffled = ContinuityBible(entries=tuple(reversed(bible.entries)))
    breaks = chain_breaks(shuffled)
    assert breaks and all(b[2] in CARRIED_KEYS for b in breaks)
    assert ("sc2", "sc1", "props_seen") in breaks


def test_continuity_is_deterministic_and_takes_no_llm():
    play = make_screenplay()
    assert build_continuity(play).digest == build_continuity(play).digest


def test_continuity_carries_the_standing_inventories_and_notes():
    bible = build_continuity(make_screenplay())
    assert bible.characters == ("MARA", "BO")
    assert bible.wardrobe == ("MARA: grey coveralls",)
    assert bible.locations == ("SALVAGE HOLD", "DOCK RAMP")
    assert "sc1: INT. SALVAGE HOLD - NIGHT" in bible.notes
    assert "lighting=one work lamp, hard from above" in bible.notes


def test_continuity_keyed_by_shots_chains_within_a_scene_too():
    play = make_screenplay()
    draft = build_shot_plan(play)
    bible = build_continuity(play, draft)
    assert bible.segment_ids == draft.segment_ids
    assert bible.missing(draft.plan.segment_ids) == ()
    assert chain_breaks(bible) == ()
    # sc1 has two shots: the first opens on the scene's opening state and the
    # second closes on its closing state.
    assert bible.state("sc1-1").state_before == \
        build_continuity(play).state("sc1").state_before
    assert bible.state("sc1-2").state_after == \
        build_continuity(play).state("sc1").state_after


def test_continuity_refuses_a_shot_plan_from_another_screenplay():
    other = Screenplay(title="OTHER", scenes=(
        Scene(scene_id="zz1", heading="EXT. VOID - DAY", location="VOID",
              time_of_day="DAY", action="Nothing."),))
    with pytest.raises(ShotPlanRefused, match="stale"):
        build_continuity(make_screenplay(), build_shot_plan(other))


def test_build_continuity_refuses_the_wrong_types():
    with pytest.raises(TypeError, match="Screenplay"):
        build_continuity("not a screenplay")
    with pytest.raises(TypeError, match="ShotPlanDraft"):
        build_continuity(make_screenplay(), ShotPlan(entries=()))


# ---------------------------------------------------------------------------
# [5] Stage 9 — the shot plan
# ---------------------------------------------------------------------------


def test_shot_plan_without_audio_is_entirely_estimated():
    play = make_screenplay()
    draft = build_shot_plan(play)
    assert draft.audio_first is False
    assert draft.estimated_ids == draft.segment_ids
    assert all(d.estimated for d in draft.designs)
    # laid end to end from zero, in order, with no overlap
    assert draft.designs[0].start_s == 0.0
    for first, second in zip(draft.designs, draft.designs[1:]):
        assert second.start_s == first.end_s
    assert draft.plan.overlaps() == ()


def test_shot_plan_with_audio_reads_its_windows_off_the_master():
    play = make_screenplay()
    draft = build_shot_plan(play, make_master(play))
    assert draft.audio_first is True
    by_id = {d.segment_id: d for d in draft.designs}
    assert (by_id["sc1-1"].start_s, by_id["sc1-1"].end_s) == (0.5, 2.3)
    assert (by_id["sc1-2"].start_s, by_id["sc1-2"].end_s) == (2.3, 4.5)
    assert (by_id["sc2-1"].start_s, by_id["sc2-1"].end_s) == (4.5, 6.5)
    assert by_id["sc1-1"].estimated is False
    assert draft.plan.overlaps() == ()


def test_only_a_scene_without_dialogue_stays_estimated_under_audio():
    """Stage 8: dialogue windows come from the recorded audio. A silent scene's
    length was never set by dialogue, so it says so."""
    play = make_screenplay()
    draft = build_shot_plan(play, make_master(play))
    assert draft.estimated_ids == ("sc3-1",)
    silent = draft.design("sc3-1")
    assert (silent.start_s, silent.end_s) == (6.5, 8.0)
    assert silent.line_ids == ()


def test_shot_plan_refuses_a_master_that_lost_a_line():
    play = make_screenplay()
    master = make_master(play)
    short = AudioMaster.from_dict({
        **master.to_dict(),
        "line_timings": [t for t in master.to_dict()["line_timings"]
                         if t["line_id"] != "l3"],
        "tracks": [t for t in master.to_dict()["tracks"] if t[0] != "l3"]})
    with pytest.raises(ShotPlanRefused, match="different script"):
        build_shot_plan(play, short)


def test_shot_plan_refuses_a_master_that_reorders_the_lines():
    play = make_screenplay()
    master = make_master(play)
    payload = master.to_dict()
    payload["line_timings"] = [payload["line_timings"][i] for i in (1, 0, 2)]
    payload["tracks"] = [payload["tracks"][i] for i in (1, 0, 2)]
    with pytest.raises(ShotPlanRefused, match="different order"):
        build_shot_plan(play, AudioMaster.from_dict(payload))


def test_every_shot_carries_stage_9s_ordered_subplan():
    draft = build_shot_plan(make_screenplay())
    for design in draft.designs:
        steps = design.subplan
        assert tuple(s for s, _ in steps) == SUBPLAN_STEPS
        assert all(text.strip() for _, text in steps)


def test_rehearse_and_tweak_become_rubric_criteria_on_the_entry():
    """The two subplan steps that are CHECKS have to end up somewhere a judge
    reads, which is the rubric k104 turns into AcceptanceTests."""
    design = build_shot_plan(make_screenplay()).design("sc1-1")
    entry = design.to_entry()
    assert isinstance(entry, ShotPlanEntry)
    assert design.rehearse in entry.rubric
    assert design.tweak in entry.rubric
    assert entry.rubric[:len(design.rubric)] == design.rubric


def test_every_shot_has_a_non_empty_rubric():
    """Unjudgeable is unacceptable (invariant 11); SegmentSpec refuses an empty
    rubric, so a plan that produced one would fail three stages later."""
    for entry in build_shot_plan(make_screenplay()).plan.entries:
        assert entry.rubric


def test_estimated_windows_say_so_in_the_rehearsal_criterion():
    draft = build_shot_plan(make_screenplay())
    assert "ESTIMATE" in draft.design("sc1-1").rehearse
    audio = build_shot_plan(make_screenplay(), make_master(make_screenplay()))
    assert "ESTIMATE" not in audio.design("sc1-1").rehearse


def test_camera_blocks_stay_inside_k104s_closed_vocabularies():
    for design in build_shot_plan(make_screenplay()).designs:
        camera = dict(design.camera)
        assert set(camera) <= CAMERA_KEYS
        assert camera["shot_size"] in SHOT_SIZES
        assert camera["movement"] in CAMERA_MOVES
        assert camera.get("view", "front") in CAMERA_VIEWS
        assert camera["lens_mm"] > 0
        # every entry it compiles to must survive k104's own validation
        design.to_entry()


def test_the_view_hint_comes_from_the_prose_or_not_at_all():
    """shot_intent derives a view from an orientation cue and NOTHING from an
    unremarkable line — the segment then inherits the movie-level DNA."""
    draft = build_shot_plan(make_screenplay())
    # sc3's action says "BO walks away down the ramp"
    assert dict(draft.design("sc3-1").camera).get("view") == "back"
    assert "view" not in dict(draft.design("sc1-1").camera)


def test_a_turning_point_beat_pushes_in():
    play = make_screenplay()
    plot = make_plot()
    plain = build_shot_plan(play)
    directed = build_shot_plan(play, plot=plot)
    assert dict(plain.design("sc2-1").camera)["movement"] == "static"
    assert dict(directed.design("sc2-1").camera)["movement"] == "push_in"
    assert dict(directed.design("sc1-1").camera)["movement"] == "static"


def test_estimated_line_seconds_respects_the_declared_budget():
    long_line = Line(line_id="x", speaker="MARA",
                     text=" ".join(["word"] * 40))
    assert estimated_line_seconds(long_line) > 10
    budgeted = Line(line_id="x", speaker="MARA",
                    text=" ".join(["word"] * 40), max_seconds=4.0)
    assert estimated_line_seconds(budgeted) == 4.0


def test_shot_plan_draft_refuses_duplicate_segments():
    design = build_shot_plan(make_screenplay()).designs[0]
    with pytest.raises(ShotPlanRefused, match="two shots"):
        ShotPlanDraft(designs=(design, design))


def test_shot_design_refuses_an_unknown_camera_key():
    with pytest.raises(ValueError, match="unknown key"):
        ShotDesign(segment_id="s1", scene_id="sc1", camera={"lense_mm": 50})


# ---------------------------------------------------------------------------
# [5b] Stage 11 — the lock, with a screenplay in it
# ---------------------------------------------------------------------------


def test_lock_production_records_the_screenplay_digest():
    play = make_screenplay().lock()
    master = make_master(play)
    snapshot = GenerationSnapshot(raw_request_ref="req-1",
                                  deliverable="a 15s scene")
    lock = lock_production(snapshot, screenplay=play, audio_master=master)
    assert isinstance(lock, ProductionLock)
    assert lock.screenplay_digest == play.digest
    assert play.digest in lock.parent_digests
    assert lock.parent_digests[0] == lock.digest


def test_lock_production_refuses_an_unlocked_screenplay():
    play = make_screenplay()
    with pytest.raises(LockRefused, match="not locked"):
        lock_production(GenerationSnapshot(raw_request_ref="req-1",
                                          deliverable="a 15s scene"),
                        screenplay=play, audio_master=make_master(play))


def test_lock_production_refuses_a_master_from_another_draft():
    play = make_screenplay().lock()
    master = make_master(play)
    edited = play.to_dict()
    edited["scenes"][0]["dialogue"][0]["text"] = "Look at this thing, Bo."
    redraft = Screenplay.from_dict(edited)
    with pytest.raises(LockRefused, match="different draft"):
        lock_production(GenerationSnapshot(raw_request_ref="req-1",
                                          deliverable="a 15s scene"),
                        screenplay=redraft, audio_master=master)


def test_lock_production_refuses_a_draft_audio_master():
    play = make_screenplay().lock()
    with pytest.raises(LockRefused, match="audio master is not locked"):
        lock_production(GenerationSnapshot(raw_request_ref="req-1",
                                          deliverable="a 15s scene"),
                        screenplay=play,
                        audio_master=make_master(play, locked=False))


# ---------------------------------------------------------------------------
# [6] LLM-assisted authoring
# ---------------------------------------------------------------------------


def test_a_valid_reply_becomes_an_artifact():
    llm = FakeLlm(plot_json())
    plot = author_plot("two salvagers and a relic they should not have", llm)
    assert isinstance(plot, PlotSpec)
    assert plot.digest == make_plot(input_mode=plot.input_mode).digest
    assert len(llm.prompts) == 1


def test_the_prompt_embeds_the_generated_schema_verbatim():
    """One definition of the truth: the schema the model is shown and the
    constructor the reply is checked against are the same dataclass."""
    prompt = build_plot_prompt("anything at all", "complete")
    assert schema_block(PlotSpec) in prompt
    assert json.dumps(json_schema_for(PlotSpec), sort_keys=True,
                      indent=2) in prompt
    assert schema_block(Screenplay) in build_screenplay_prompt(make_plot())


def test_the_screenplay_prompt_carries_the_locked_plot_and_its_vocabularies():
    prompt = build_screenplay_prompt(make_plot())
    assert make_plot().digest[:12] in prompt
    assert '"b2"' in prompt and "MARA" in prompt
    assert "FLASHBACK TO:" in prompt and "CONTINUOUS:" in prompt


def test_one_bounded_reprompt_carries_the_validators_own_words():
    broken = json.loads(plot_json())
    broken["beats"][1]["causes"] = ["b7"]
    llm = FakeLlm(json.dumps(broken), plot_json())
    plot = author_plot("two salvagers", llm)
    assert isinstance(plot, PlotSpec)
    assert len(llm.prompts) == 2
    repair = llm.prompts[1]
    assert "'b7'" in repair and "not a beat in this plot" in repair
    assert "VALIDATION ERRORS" in repair
    assert json.dumps(broken) in repair          # the rejected reply, echoed
    assert schema_block(PlotSpec) in repair      # and the schema, still there


def test_twice_invalid_is_a_typed_gap_with_the_raw_reply_preserved():
    broken = json.loads(plot_json())
    broken["beats"][0]["characters"] = []
    first, second = json.dumps(broken), json.dumps({**broken, "tone": "grim"})
    llm = FakeLlm(first, second)
    gap = author_plot("two salvagers", llm)
    assert isinstance(gap, AuthoringGap)
    assert gap.code == "AUTHORING_INVALID"
    assert gap.stage == "plot" and gap.attempts == 2
    assert gap.raw == second
    assert gap.raw_attempts == (first, second)
    assert any("names no character" in e for e in gap.errors)


def test_a_reply_with_no_json_at_all_is_a_typed_gap():
    llm = FakeLlm("Sure! Here is a great plot for you.",
                  "I would be delighted to help.")
    gap = author_plot("two salvagers", llm)
    assert isinstance(gap, AuthoringGap)
    assert gap.code == "AUTHORING_UNPARSED"
    assert gap.raw == "I would be delighted to help."


def test_a_model_that_raises_is_a_typed_gap_not_an_exception():
    class Angry:
        def __call__(self, prompt: str) -> str:
            raise RuntimeError("worker unavailable")

    gap = author_plot("two salvagers", Angry())
    assert isinstance(gap, AuthoringGap)
    assert gap.code == "LLM_ERROR"
    assert "worker unavailable" in gap.errors[0]


def test_authoring_never_returns_a_coerced_artifact():
    """The point of the whole module: there is no third branch between a valid
    artifact and a typed gap."""
    broken = json.loads(plot_json())
    del broken["ending"]
    gap = author_plot("x", FakeLlm(json.dumps(broken), json.dumps(broken)))
    assert isinstance(gap, AuthoringGap)
    assert not isinstance(gap, PlotSpec)


def test_a_missing_required_field_is_reported_by_name():
    broken = json.loads(plot_json())
    del broken["beats"][0]["beat_id"]
    gap = author_plot("x", FakeLlm(json.dumps(broken), json.dumps(broken)))
    assert isinstance(gap, AuthoringGap)
    assert any("beat_id" in e for e in gap.errors)


def test_author_screenplay_sets_provenance_itself():
    plot = make_plot()
    payload = json.loads(screenplay_json())
    payload["plot_digest"] = "0" * 64          # a digest the model invented
    llm = FakeLlm(json.dumps(payload))
    play = author_screenplay(plot, llm)
    assert isinstance(play, Screenplay)
    assert play.plot_digest == plot.digest     # overwritten, never trusted


def test_author_screenplay_refuses_a_non_plot():
    with pytest.raises(TypeError, match="PlotSpec"):
        author_screenplay({"premise": "x"}, FakeLlm(""))


def test_author_screenplay_gaps_on_a_speaker_who_is_not_in_the_room():
    payload = json.loads(screenplay_json())
    payload["scenes"][0]["dialogue"][0]["speaker"] = "TESS"
    llm = FakeLlm(json.dumps(payload), json.dumps(payload))
    gap = author_screenplay(make_plot(), llm)
    assert isinstance(gap, AuthoringGap)
    assert gap.stage == "screenplay"
    assert any("TESS" in e for e in gap.errors)
    assert "TESS" in llm.prompts[1]


def test_all_three_input_modes_route_through_the_same_validators():
    broken = json.loads(plot_json())
    broken["beats"][0]["characters"] = []
    results = {}
    for mode in INPUT_MODES:
        gap = author_plot("anything", FakeLlm(json.dumps(broken),
                                              json.dumps(broken)), mode=mode)
        assert isinstance(gap, AuthoringGap)
        results[mode] = gap.errors
    assert len({tuple(v) for v in results.values()}) == 1


def test_the_mode_only_changes_the_guidance_paragraph():
    prompts = {m: build_plot_prompt("some notes about a relic", m)
               for m in INPUT_MODES}
    assert len(set(prompts.values())) == 3
    for prompt in prompts.values():
        assert schema_block(PlotSpec) in prompt


def test_input_mode_is_classified_deterministically_and_recorded():
    assert plot_input_mode("") == "minimal"
    assert plot_input_mode("make me a film about two salvagers") == "minimal"
    assert plot_input_mode(" ".join(["a story about salvagers"] * 12)) == "partial"
    assert plot_input_mode("INT. HOLD - NIGHT\n" + " " .join(["x"] * 40)) \
        == "complete"
    llm = FakeLlm(plot_json())
    plot = author_plot("make me a film about two salvagers", llm)
    assert plot.input_mode == "minimal"
    assert "INPUT MODE: minimal" in llm.prompts[0]


def test_an_unknown_mode_is_a_caller_error_not_a_gap():
    with pytest.raises(ValueError, match="not one of"):
        author_plot("x", FakeLlm(""), mode="cinematic")


def test_author_plot_refuses_a_non_callable_model():
    with pytest.raises(TypeError, match="callable"):
        author_plot("x", "gpt-please")


def test_repair_prompt_bounds_what_it_echoes():
    prompt = build_repair_prompt("ORIGINAL", "x" * 10_000,
                                 [f"error {i}" for i in range(50)])
    assert len(prompt) < 12_000
    assert "error 0" in prompt and "error 49" not in prompt


# ---------------------------------------------------------------------------
# [6b] parsing — tolerant of exactly two things, and nothing else
# ---------------------------------------------------------------------------


def test_parse_tolerates_a_code_fence_and_surrounding_prose():
    obj, why = parse_json_object('Here you go:\n```json\n{"a": 1}\n```\nEnjoy!')
    assert obj == {"a": 1} and why == ""


def test_parse_is_string_aware_so_dialogue_cannot_close_the_object():
    obj, why = parse_json_object('{"text": "he said }", "n": 2}')
    assert obj == {"text": "he said }", "n": 2} and why == ""
    obj, why = parse_json_object(r'{"text": "a backslash \" and a }", "n": 3}')
    assert obj is not None and obj["n"] == 3


def test_parse_reports_why_instead_of_raising():
    for text, marker in (("no object here", "no JSON object"),
                         ('{"a": 1', "never closed"),
                         ('{"a": }', "not valid JSON"),
                         ("[1, 2, 3]", "no JSON object")):
        obj, why = parse_json_object(text)
        assert obj is None and marker in why


# ---------------------------------------------------------------------------
# [6c] the live binding — degrades to a typed gap, never a fake answer
# ---------------------------------------------------------------------------


def test_bind_llm_degrades_to_a_capability_gap_when_nothing_is_eligible(
        monkeypatch):
    from hugpy_oracle import router

    class Gap:
        execution = "gap"
        reasons = ("no text model is eligible on this host",)

    monkeypatch.setattr(router, "resolve_route", lambda goal, *a, **k: Gap())
    result = bind_llm()
    assert isinstance(result, AuthoringGap)
    assert result.code == "CAPABILITY_GAP"
    assert any("not executable" in e for e in result.errors)
    assert "no text model is eligible on this host" in result.errors


def test_bind_llm_returns_a_callable_that_reads_the_text_artifact(monkeypatch):
    from hugpy_oracle import router, runtime

    class Ok:
        execution = "execute"
        reasons = ()

    class Receipt:
        failure = None
        log_excerpt = ()

    monkeypatch.setattr(router, "resolve_route", lambda goal, *a, **k: Ok())
    monkeypatch.setattr(runtime, "execute_route",
                        lambda goal, route, **k: (
                            [{"kind": "text", "uri": "inline:",
                              "text": plot_json()}], Receipt()))
    llm = bind_llm()
    assert callable(llm)
    plot = author_plot("two salvagers", llm)
    assert isinstance(plot, PlotSpec)


def test_bind_llm_failures_become_a_gap_not_a_hang(monkeypatch):
    from hugpy_oracle import router, runtime

    class Ok:
        execution = "execute"
        reasons = ()

    class Failed:
        class failure:            # noqa: N801 - a stand-in for FailureClass
            value = "TIMEOUT"
        log_excerpt = ("the oracle stopped waiting",)

    monkeypatch.setattr(router, "resolve_route", lambda goal, *a, **k: Ok())
    monkeypatch.setattr(runtime, "execute_route",
                        lambda goal, route, **k: ([], Failed()))
    gap = author_plot("two salvagers", bind_llm())
    assert isinstance(gap, AuthoringGap)
    assert gap.code == "LLM_ERROR"
    assert "TIMEOUT" in gap.errors[0]


def test_the_authoring_capability_is_a_real_catalog_name():
    from hugpy_oracle.router import CAPABILITY_TASK
    assert AUTHORING_CAPABILITY in CAPABILITY_TASK


# ---------------------------------------------------------------------------
# [6d] k114's follow-up, landed: ``bind_llm(requested_model=...)`` — an
# additive passthrough so a caller (k114's ``resolve_authoring_model``) can
# pin k109's routing-matrix winner instead of the catalog default.
# ---------------------------------------------------------------------------


def test_bind_llm_forwards_requested_model_to_every_resolve_route_call(
        monkeypatch):
    from hugpy_oracle import router, runtime

    seen: list[str | None] = []

    class Ok:
        execution = "execute"
        reasons = ()

    class Receipt:
        failure = None
        log_excerpt = ()

    def fake_resolve(goal, requested_model=None):
        seen.append(requested_model)
        return Ok()

    monkeypatch.setattr(router, "resolve_route", fake_resolve)
    monkeypatch.setattr(runtime, "execute_route",
                        lambda goal, route, **k: (
                            [{"kind": "text", "uri": "inline:",
                              "text": plot_json()}], Receipt()))
    llm = bind_llm(requested_model="Qwen3.8_4B_Distilled_GGUF")
    assert callable(llm)
    plot = author_plot("two salvagers", llm)
    assert isinstance(plot, PlotSpec)
    # the probe call AND the real dispatch call both carry the pin
    assert seen == ["Qwen3.8_4B_Distilled_GGUF", "Qwen3.8_4B_Distilled_GGUF"]


def test_bind_llm_requested_model_defaults_to_none_unchanged_behaviour(
        monkeypatch):
    """Omitted, behaviour is byte-identical to before this parameter
    existed: ``resolve_route`` sees ``requested_model=None`` and picks the
    catalog default exactly as it always has."""
    from hugpy_oracle import router

    seen: list[str | None] = []

    class Gap:
        execution = "gap"
        reasons = ()

    def fake_resolve(goal, requested_model=None):
        seen.append(requested_model)
        return Gap()

    monkeypatch.setattr(router, "resolve_route", fake_resolve)
    result = bind_llm()
    assert isinstance(result, AuthoringGap)
    assert seen == [None]


def test_bind_llm_ineligible_requested_model_is_a_gap_not_a_raise(
        monkeypatch):
    """A ``RouteRefusal`` (a requested model outside the capability's
    eligible set — a race between the matrix and the live fleet) is caught
    exactly like every other routing fault: a typed gap, never a silent
    substitution and never an uncaught exception."""
    from hugpy_oracle import router

    def refusing(goal, requested_model=None):
        assert requested_model == "some-evicted-model"
        raise router.RouteRefusal(
            f"model {requested_model!r} does not serve capability "
            f"'text.chat' on this fleet; eligible: ['fixture-llm']")

    monkeypatch.setattr(router, "resolve_route", refusing)
    result = bind_llm(requested_model="some-evicted-model")
    assert isinstance(result, AuthoringGap)
    assert result.code == "CAPABILITY_GAP"
    assert "some-evicted-model" in result.errors[0]


# ---------------------------------------------------------------------------
# end-to-end: authored plot -> authored screenplay -> continuity -> shots -> lock
# ---------------------------------------------------------------------------


def test_the_whole_stage_5_to_11_path_runs_offline():
    plot = author_plot("two salvagers argue over a relic", FakeLlm(plot_json()))
    assert isinstance(plot, PlotSpec)
    play = author_screenplay(plot, FakeLlm(screenplay_json()))
    assert isinstance(play, Screenplay)
    locked = play.lock()
    master = make_master(locked)
    draft = build_shot_plan(locked, master, plot=plot)
    bible = build_continuity(locked, draft)
    lock = lock_production(GenerationSnapshot(raw_request_ref="req-1",
                                              deliverable="a 15s scene"),
                           screenplay=locked, audio_master=master,
                           continuity=bible, shots=draft)
    assert lock.screenplay_digest == locked.digest
    assert set(lock.parent_digests) >= {locked.digest, bible.digest,
                                        draft.plan.digest, master.digest}
    assert chain_breaks(bible) == ()
    assert draft.estimated_ids == ("sc3-1",)
