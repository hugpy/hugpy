"""k108 — PostProductionPlan / EDL, footage-vs-spec evaluation, final report.

Everything here is OFFLINE and deterministic: no catalog, no worker, no GPU, no
network, no clock, no disk (the one file on disk is a pytest ``tmp_path`` image
the live-judge binding is pointed at, because k90c's judge body checks that its
evidence is a readable file). The "judge" is a function that returns canned
verdicts and records the requests it was handed; the "transcriber" is a
function that returns a word list. That is the whole point of the injected
seams: the acceptance contract for generated footage is testable without an
inference stack, and the live binding is two monkeypatched module attributes
away.

Locks:
  [1] wire shape: every artifact round-trips through to_dict/from_dict without
      loss and digests over k104's canonical JSON, byte-compatible with the
      lock it is derived from.
  [2] the EDL describes an EXECUTABLE assembly: gapless order, one preferred
      take per segment, a closed transition vocabulary, cut points inside the
      footage.
  [3] a segment with no accepted take becomes an explicit RegenerationNote
      naming a repair code and a documented correction — never a silent hole,
      and never an instruction to re-prompt from the rejected footage.
  [4] evaluate_take composes the k90c judge rubric, the k98 speech checks and
      the k98 duration fit into ONE Scorecard, and maps each failure to the
      repair code that names what to regenerate.
  [5] the generator never judges itself: a judge model equal to the take's
      generator is a typed refusal BEFORE a dispatch is spent.
  [6] the final report catches what no per-shot pass can see: an uncovered
      segment, an omitted line, a pacing outlier, a scene with no grade target.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_postproduction.py -q
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import replace

import pytest

logging.disable(logging.WARNING)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import evaluation, router, speech
from hugpy_oracle import postproduction as pp
from hugpy_oracle import production as production_mod
from hugpy_oracle.audio_master import AudioMaster, LineTiming
from hugpy_oracle.contracts import (
    Check,
    CheckKind,
    JudgeResult,
    QualityProfile,
    RepairCode,
    Scorecard,
)
from hugpy_oracle.plan import FrozenParams
from hugpy_oracle.production import (
    ContinuityState,
    ProductionLock,
    ShotPlan,
    ShotPlanEntry,
    canonical_json,
)
from hugpy_oracle.postproduction import (
    EDLRefused,
    EditDecision,
    EditDecisionList,
    JudgeConflict,
    JudgeRequest,
    MusicCue,
    PlanRefused,
    PostProductionError,
    PostProductionPlan,
    RegenerationNote,
    ShotBrief,
    SoundCue,
    Take,
    bind_live_judge,
    build_postproduction_plan,
    evaluate_take,
    final_consistency_report,
    shot_brief,
    shot_briefs,
)
from hugpy_oracle.segments import SegmentSpec

BALANCED = evaluation.THRESHOLDS[QualityProfile.BALANCED]


# ---------------------------------------------------------------------------
# Fixtures — a tiny three-shot production
# ---------------------------------------------------------------------------


def _state(segment_id, before=None, after=None):
    return ContinuityState(segment_id=segment_id,
                           state_before=FrozenParams(before or {}),
                           state_after=FrozenParams(after or {}))


def _entry(segment_id, start, end, *, line_ids=(), lighting="soft key light",
           blocking="ALEX crosses to the window", rubric=("ALEX is visible",)):
    return ShotPlanEntry(segment_id=segment_id, line_ids=tuple(line_ids),
                         start_s=start, end_s=end,
                         camera={"shot_size": "medium"}, blocking=blocking,
                         lighting=lighting, rubric=tuple(rubric))


def _spec(segment_id, index, start, end, *, line_ids=(), scene="sc_01",
          lighting="soft key light", blocking="ALEX crosses to the window",
          before=None, after=None, rubric=("ALEX is visible",)):
    entry = _entry(segment_id, start, end, line_ids=line_ids,
                   lighting=lighting, blocking=blocking, rubric=rubric)
    return SegmentSpec(
        segment_id=segment_id, index=index, scene_ref=scene,
        continuity=_state(segment_id,
                          before or {"location": "KITCHEN", "present": ["ALEX"],
                                     "coat": "on"},
                          after or {"location": "KITCHEN", "present": ["ALEX"],
                                    "coat": "off"}),
        audio_window=(start, end, tuple(line_ids)), shot=entry,
        spatial_ref=None, tone=0.5, rubric=tuple(rubric),
        prompt=f"{segment_id}: medium shot, {blocking}",
        negative_prompt=None, identity_refs=("ref_alex",),
        joint_mode="cut", seed_base=1234 + index,
        lock_digest="lock-digest-abc", parents=("lock-digest-abc",))


@pytest.fixture()
def specs():
    return (
        _spec("seg_01", 0, 0.0, 3.0, line_ids=("L1",)),
        _spec("seg_02", 1, 3.0, 6.0, line_ids=("L2",),
              lighting="warm amber practicals", scene="sc_02"),
        _spec("seg_03", 2, 6.0, 9.0),
    )


@pytest.fixture()
def lines():
    return {"L1": "Hello there friend", "L2": "The kettle is boiling"}


@pytest.fixture()
def master():
    timings = (LineTiming(line_id="L1", start_s=0.0, end_s=2.5,
                          pause_after_s=0.5),
               LineTiming(line_id="L2", start_s=3.0, end_s=5.5,
                          pause_after_s=0.5))
    return AudioMaster(timeline_digest="timeline-digest",
                       line_timings=timings,
                       tracks=(("L1", "/tmp/l1.wav"), ("L2", "/tmp/l2.wav")),
                       total_seconds=9.0, locked=True)


def _take(segment_id, attempt="a1", *, duration=3.0, accepted=True,
          preferred=False, **kw):
    return Take(segment_id=segment_id, attempt_id=f"{segment_id}:{attempt}",
                artifact_ref=f"/clips/{segment_id}_{attempt}.mp4",
                duration_s=duration, accepted=accepted, preferred=preferred,
                **kw)


def _takes(specs, **overrides):
    out = []
    for spec in specs:
        out.append(overrides.get(spec.segment_id,
                                 _take(spec.segment_id,
                                       duration=spec.duration_s)))
    return tuple(t for t in out if t is not None)


class FakeJudge:
    """Records every request and answers from a per-rubric table."""

    def __init__(self, verdicts=None, model="judge-vlm", raise_on=()):
        self.verdicts = dict(verdicts or {})
        self.model = model
        self.raise_on = set(raise_on)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if request.name in self.raise_on:
            raise RuntimeError("vision plane is down")
        return self.verdicts.get(request.name,
                                 {"verdict": "YES", "score": 92,
                                  "why": "matches the specification"})

    @property
    def names(self):
        return [r.name for r in self.requests]


def _words(text):
    return [{"word": w} for w in text.split()]


def _transcribe_of(text):
    def transcribe(ref):
        assert ref
        return _words(text)
    return transcribe


# ---------------------------------------------------------------------------
# [1] wire shape and digests
# ---------------------------------------------------------------------------


def test_take_round_trips_and_digests_stably():
    take = _take("seg_01", generator_model="wan2.1", seed=7,
                 scorecard_digest="card-digest")
    assert Take.from_dict(take.to_dict()) == take
    assert take.digest == _take("seg_01", generator_model="wan2.1", seed=7,
                                scorecard_digest="card-digest").digest


def test_artifact_digest_is_canonical_json_over_to_dict():
    """Byte-compatible with k104: one encoding across the tree, so a plan
    digest is comparable to a lock digest without a conversion step."""
    take = _take("seg_01")
    import hashlib
    expected = hashlib.sha256(canonical_json(take.to_dict())).hexdigest()
    assert take.digest == expected


def test_borrowed_production_helpers_are_the_same_objects():
    assert pp.require_text is production_mod._require_text
    assert pp.str_tuple is production_mod._str_tuple
    assert pp.quantize is production_mod._q
    assert pp.EPS is production_mod._EPS


def test_edit_decision_and_edl_round_trip():
    take = _take("seg_01", preferred=True)
    decision = EditDecision(order=0, segment_id="seg_01", take=take, in_s=0.0,
                            out_s=2.5, transition="fade",
                            transition_duration_s=0.5, note="open on black")
    edl = EditDecisionList(decisions=(decision,), fps=24.0, note="v1")
    assert EditDecision.from_dict(decision.to_dict()) == decision
    assert EditDecisionList.from_dict(edl.to_dict()) == edl
    assert edl.digest == EditDecisionList.from_dict(edl.to_dict()).digest


def test_plan_round_trips_with_every_phase_three_artifact():
    take = _take("seg_01", preferred=True)
    plan = PostProductionPlan(
        edl=EditDecisionList(decisions=(EditDecision(
            order=0, segment_id="seg_01", take=take),)),
        pacing_notes=("holds a beat too long on the exit",),
        sound=(SoundCue(cue_id="sfx_01", kind="ambience", start_s=0.0,
                        end_s=3.0, description="kitchen room tone"),),
        music=(MusicCue(cue_id="m_01", start_s=0.0, end_s=3.0,
                        description="low piano bed", role="score"),),
        dialogue_alignment=(("L1", -0.25),),
        color_targets={"sc_01": "warm 3200K, lifted blacks"},
        continuity_corrections=(("seg_01", "coat must read as ON at the cut"),),
        regeneration_notes=(RegenerationNote(
            segment_id="seg_09", repair_code=RepairCode.ACTION_MISSING,
            correction="regenerate from spec abc with the crossing action "
                       "stated explicitly"),),
        locked_segments=("seg_01", "seg_09"), lock_digest="lock-digest-abc",
        audio_master_digest="master-digest")
    assert PostProductionPlan.from_dict(plan.to_dict()) == plan
    assert PostProductionPlan.from_dict(plan.to_dict()).digest == plan.digest


def test_regeneration_note_unpacks_as_the_brief_s_triple():
    note = RegenerationNote(segment_id="seg_02",
                            repair_code=RepairCode.TEMPORAL_ARTIFACT,
                            correction="reroll at a bumped seed",
                            spec_digest="spec-digest")
    segment_id, code, correction = note
    assert (segment_id, code) == ("seg_02", RepairCode.TEMPORAL_ARTIFACT)
    assert correction == "reroll at a bumped seed"
    assert RegenerationNote.from_dict(note.to_dict()) == note


def test_regeneration_note_refuses_an_empty_correction():
    # ValueError, because the artifact validators BORROWED from k104 raise it
    # (PostProductionError is a ValueError subclass, so this is the superset).
    with pytest.raises(ValueError, match="correction must be non-empty"):
        RegenerationNote(segment_id="seg_02",
                         repair_code=RepairCode.ACTION_MISSING, correction="  ")


def test_cues_refuse_a_backwards_window_and_an_unknown_kind():
    with pytest.raises(PostProductionError, match="ends before it starts"):
        SoundCue(cue_id="s1", kind="sfx", start_s=3.0, end_s=1.0,
                 description="door")
    with pytest.raises(PostProductionError, match="must be one of"):
        SoundCue(cue_id="s1", kind="explosion", start_s=0.0, end_s=1.0,
                 description="door")
    with pytest.raises(PostProductionError, match="must be one of"):
        MusicCue(cue_id="m1", start_s=0.0, end_s=1.0, description="bed",
                 role="jingle")


# ---------------------------------------------------------------------------
# [2] Take and EDL validators — positive and negative
# ---------------------------------------------------------------------------


def test_take_refuses_preferred_but_rejected():
    with pytest.raises(PostProductionError, match="preferred but not accepted"):
        Take(segment_id="seg_01", attempt_id="a1", artifact_ref="/c.mp4",
             accepted=False, preferred=True,
             repair_codes=(RepairCode.INTENT_MISMATCH,))


def test_take_refuses_a_rejection_nobody_wrote_down():
    with pytest.raises(PostProductionError, match="no repair code"):
        Take(segment_id="seg_01", attempt_id="a1", artifact_ref="/c.mp4",
             accepted=False)


def test_take_judged_bridges_a_scorecard_into_acceptance():
    card = Scorecard(hard_pass=False,
                     checks=(Check(name="shot.action", kind=CheckKind.SEMANTIC,
                                   value=10, threshold=60, passed=False,
                                   detail="no crossing"),),
                     diagnosis="shot.action: no crossing",
                     repair_code=RepairCode.ACTION_MISSING)
    take = Take.judged("seg_01", "seg_01:a1", "/clips/a1.mp4", card,
                       duration_s=3.0)
    assert take.accepted is False
    assert take.repair_codes == (RepairCode.ACTION_MISSING,)
    assert take.scorecard_digest and take.diagnosis.startswith("shot.action")


def test_edl_accepts_a_gapless_cut_and_reports_its_length():
    takes = [_take(f"seg_0{i}", preferred=True, duration=2.0)
             for i in (1, 2, 3)]
    edl = EditDecisionList(decisions=tuple(
        EditDecision(order=i, segment_id=t.segment_id, take=t)
        for i, t in enumerate(takes)))
    assert edl.segment_ids == ("seg_01", "seg_02", "seg_03")
    assert edl.total_seconds == 6.0
    assert len(edl) == 3


def test_edl_refuses_a_gap_in_the_assembly_order():
    takes = [_take(f"seg_0{i}", preferred=True) for i in (1, 2)]
    with pytest.raises(EDLRefused, match="not gapless"):
        EditDecisionList(decisions=(
            EditDecision(order=0, segment_id="seg_01", take=takes[0]),
            EditDecision(order=2, segment_id="seg_02", take=takes[1])))


def test_edl_refuses_a_repeated_order():
    takes = [_take(f"seg_0{i}", preferred=True) for i in (1, 2)]
    with pytest.raises(EDLRefused, match="not gapless"):
        EditDecisionList(decisions=(
            EditDecision(order=0, segment_id="seg_01", take=takes[0]),
            EditDecision(order=0, segment_id="seg_02", take=takes[1])))


def test_edl_refuses_a_take_that_is_not_the_preferred_one():
    take = _take("seg_01", preferred=False)
    with pytest.raises(EDLRefused, match="preferred"):
        EditDecisionList(decisions=(
            EditDecision(order=0, segment_id="seg_01", take=take),))


def test_edl_refuses_two_preferred_takes_for_one_segment():
    first = _take("seg_01", "a1", preferred=True)
    second = _take("seg_01", "a2", preferred=True)
    with pytest.raises(EDLRefused, match="two preferred takes"):
        EditDecisionList(decisions=(
            EditDecision(order=0, segment_id="seg_01", take=first),
            EditDecision(order=1, segment_id="seg_01", take=second)))


def test_edl_allows_one_take_twice_because_an_intercut_is_legal():
    take = _take("seg_01", preferred=True, duration=2.0)
    other = _take("seg_02", preferred=True, duration=2.0)
    edl = EditDecisionList(decisions=(
        EditDecision(order=0, segment_id="seg_01", take=take),
        EditDecision(order=1, segment_id="seg_02", take=other),
        EditDecision(order=2, segment_id="seg_01", take=take)))
    assert edl.segment_ids == ("seg_01", "seg_02")
    assert len(edl.decisions) == 3


def test_decision_refuses_a_transition_outside_the_vocabulary():
    take = _take("seg_01", preferred=True)
    with pytest.raises(EDLRefused, match="must be one of"):
        EditDecision(order=0, segment_id="seg_01", take=take,
                     transition="wipe", transition_duration_s=1.0)


def test_decision_refuses_a_timed_cut_and_a_zero_length_dissolve():
    take = _take("seg_01", preferred=True)
    with pytest.raises(EDLRefused, match="occupy no time"):
        EditDecision(order=0, segment_id="seg_01", take=take,
                     transition="cut", transition_duration_s=0.5)
    with pytest.raises(EDLRefused, match="zero"):
        EditDecision(order=0, segment_id="seg_01", take=take,
                     transition="dissolve", transition_duration_s=0.0)


def test_decision_refuses_a_cut_point_past_the_end_of_the_footage():
    take = _take("seg_01", preferred=True, duration=2.0)
    with pytest.raises(EDLRefused, match="past the end of take"):
        EditDecision(order=0, segment_id="seg_01", take=take, in_s=0.0,
                     out_s=5.0)


def test_decision_refuses_an_out_point_before_its_in_point():
    take = _take("seg_01", preferred=True, duration=5.0)
    with pytest.raises(EDLRefused, match="before it comes in"):
        EditDecision(order=0, segment_id="seg_01", take=take, in_s=3.0,
                     out_s=1.0)


def test_decision_refuses_another_segments_take():
    take = _take("seg_02", preferred=True)
    with pytest.raises(EDLRefused, match="carries a take of"):
        EditDecision(order=0, segment_id="seg_01", take=take)


def test_edl_refuses_a_continuous_first_row():
    take = _take("seg_01", preferred=True)
    with pytest.raises(EDLRefused, match="first row"):
        EditDecisionList(decisions=(
            EditDecision(order=0, segment_id="seg_01", take=take,
                         transition="continuous"),))


def test_edl_total_is_none_rather_than_wrong_when_a_row_is_unmeasured():
    measured = _take("seg_01", preferred=True, duration=2.0)
    unmeasured = _take("seg_02", preferred=True, duration=None)
    edl = EditDecisionList(decisions=(
        EditDecision(order=0, segment_id="seg_01", take=measured),
        EditDecision(order=1, segment_id="seg_02", take=unmeasured)))
    assert edl.has_unmeasured_rows is True
    assert edl.total_seconds is None


# ---------------------------------------------------------------------------
# [3] build_postproduction_plan
# ---------------------------------------------------------------------------


def test_plan_assembles_in_shot_plan_order_and_is_gapless(specs, master, lines):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    assert plan.edl.segment_ids == ("seg_01", "seg_02", "seg_03")
    assert [d.order for d in plan.edl.decisions] == [0, 1, 2]
    assert all(d.take.preferred for d in plan.edl.decisions)
    assert plan.locked_segments == ("seg_01", "seg_02", "seg_03")
    assert plan.regeneration_notes == ()
    assert plan.audio_master_digest == master.digest


def test_plan_is_deterministic(specs, master):
    first = build_postproduction_plan(specs, _takes(specs), master)
    second = build_postproduction_plan(specs, _takes(specs), master)
    assert first.digest == second.digest
    assert first == second


def test_a_missing_take_becomes_a_regeneration_note_never_silence(specs, master):
    takes = [t for t in _takes(specs) if t.segment_id != "seg_02"]
    plan = build_postproduction_plan(specs, takes, master)
    assert plan.edl.segment_ids == ("seg_01", "seg_03")
    assert "seg_02" in plan.locked_segments        # still declared
    assert plan.uncovered_segments == ("seg_02",)
    note = plan.note_for("seg_02")
    assert note is not None and note.repair_code is RepairCode.EMPTY_OUTPUT
    assert note.spec_digest == specs[1].digest     # the CANONICAL spec
    assert "Do NOT re-prompt from the rejected footage" in note.correction


def test_a_rejected_take_keeps_its_code_and_its_diagnosis_in_the_note(specs,
                                                                     master):
    rejected = Take(segment_id="seg_02", attempt_id="seg_02:a1",
                    artifact_ref="/clips/seg_02_a1.mp4", duration_s=3.0,
                    accepted=False,
                    repair_codes=(RepairCode.ACTION_MISSING,),
                    diagnosis="ALEX never crosses to the window")
    takes = [t for t in _takes(specs) if t.segment_id != "seg_02"] + [rejected]
    plan = build_postproduction_plan(specs, takes, master)
    note = plan.note_for("seg_02")
    assert note.repair_code is RepairCode.ACTION_MISSING
    assert "never crosses" in note.correction
    assert note.rejected_take_ref == "/clips/seg_02_a1.mp4"   # evidence only
    assert note.spec_digest == specs[1].digest                # regenerate FROM


def test_plan_honours_an_explicitly_preferred_take(specs, master):
    first = _take("seg_01", "a1", duration=3.0)
    second = _take("seg_01", "a2", duration=3.0, preferred=True)
    takes = [first, second] + [t for t in _takes(specs)
                               if t.segment_id != "seg_01"]
    plan = build_postproduction_plan(specs, takes, master)
    assert plan.edl.decision_for("seg_01").take.attempt_id == "seg_01:a2"


def test_plan_refuses_two_preferred_takes_rather_than_picking(specs, master):
    takes = [_take("seg_01", "a1", duration=3.0, preferred=True),
             _take("seg_01", "a2", duration=3.0, preferred=True)]
    takes += [t for t in _takes(specs) if t.segment_id != "seg_01"]
    with pytest.raises(PlanRefused, match="preferred"):
        build_postproduction_plan(specs, takes, master)


def test_plan_falls_back_to_the_first_accepted_take(specs, master):
    rejected = Take(segment_id="seg_01", attempt_id="seg_01:a1",
                    artifact_ref="/clips/a1.mp4", duration_s=3.0,
                    accepted=False, repair_codes=(RepairCode.TEMPORAL_ARTIFACT,))
    accepted = _take("seg_01", "a2", duration=3.0)
    takes = [rejected, accepted] + [t for t in _takes(specs)
                                    if t.segment_id != "seg_01"]
    plan = build_postproduction_plan(specs, takes, master)
    assert plan.edl.decision_for("seg_01").take.attempt_id == "seg_01:a2"
    assert plan.regeneration_notes == ()


def test_a_long_take_is_trimmed_to_its_locked_window(specs, master):
    long_take = _take("seg_01", duration=5.0)
    takes = [long_take] + [t for t in _takes(specs)
                           if t.segment_id != "seg_01"]
    plan = build_postproduction_plan(specs, takes, master)
    decision = plan.edl.decision_for("seg_01")
    assert (decision.in_s, decision.out_s) == (0.0, 3.0)
    assert decision.duration_s == 3.0


def test_dialogue_alignment_is_flat_until_a_shot_runs_short(specs, master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    assert plan.dialogue_alignment == (("L1", 0.0), ("L2", 0.0))

    short = _take("seg_01", duration=2.0)
    drifted = build_postproduction_plan(
        specs, [short] + [t for t in _takes(specs)
                          if t.segment_id != "seg_01"], master)
    assert drifted.dialogue_alignment == (("L1", 0.0), ("L2", 1.0))


def test_color_targets_are_derived_for_the_scenes_that_declared_a_look(specs,
                                                                      master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    assert "sc_02" in plan.color_targets            # "warm amber practicals"
    assert "sc_01" not in plan.color_targets        # "soft key light" declares no look
    assert pp.declared_color_scenes(specs) == ("sc_02",)


def test_an_operator_color_target_wins_over_the_derived_one(specs, master):
    plan = build_postproduction_plan(specs, _takes(specs), master,
                                     color_targets={"sc_02": "teal/orange, "
                                                             "crushed blacks"})
    assert plan.color_targets["sc_02"] == "teal/orange, crushed blacks"


def test_pacing_notes_name_the_outlier_and_are_never_empty(specs, master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    assert plan.pacing_notes                          # never empty
    short = _take("seg_02", duration=1.0)
    noisy = build_postproduction_plan(
        specs, [short] + [t for t in _takes(specs)
                          if t.segment_id != "seg_02"], master)
    assert any("seg_02" in note and "-2.000s" in note
               for note in noisy.pacing_notes)


def test_transitions_are_straight_cuts_unless_told_otherwise(specs, master):
    plan = build_postproduction_plan(
        specs, _takes(specs), master,
        transitions={"seg_02": ("dissolve", 0.75), "seg_03": "cut"})
    assert plan.edl.decisions[0].transition == "cut"
    assert plan.edl.decisions[1].transition == "dissolve"
    assert plan.edl.decisions[1].transition_duration_s == 0.75


def test_takes_from_another_production_are_refused(specs, master):
    stray = _take("seg_99", duration=1.0)
    with pytest.raises(PlanRefused, match="different productions"):
        build_postproduction_plan(specs, list(_takes(specs)) + [stray], master)


def test_a_bare_production_lock_is_refused_with_the_fix(specs):
    lock = ProductionLock(snapshot_digest="s", screenplay_digest=None,
                          continuity_digest="c", audio_master_digest="a",
                          shot_plan_digest="p")
    with pytest.raises(PlanRefused, match="carries DIGESTS, not shots"):
        build_postproduction_plan(lock, _takes(specs))


def test_plan_builds_from_a_shot_plan_and_binds_to_a_lock(specs, master):
    plan_source = ShotPlan(entries=tuple(s.shot for s in specs))
    lock = ProductionLock(snapshot_digest="s", screenplay_digest=None,
                          continuity_digest="c", audio_master_digest="a",
                          shot_plan_digest=plan_source.digest)
    plan = build_postproduction_plan(plan_source, _takes(specs), master,
                                     lock=lock)
    assert plan.lock_digest == lock.digest
    assert plan.edl.segment_ids == ("seg_01", "seg_02", "seg_03")


def test_plan_reads_a_run_state_including_its_own_takes(specs, master):
    class FakeShot:
        def __init__(self, segment_id, ref, seconds, accepted=True):
            self.segment_id = segment_id
            self.clip_ref = ref
            self.clip_seconds = seconds
            self.accepted = accepted
            self.scorecard = None
            self.repair_codes = ()
            self.diagnosis = ""

    class FakeRun:
        segments = ()
        shots = ()
        audio_master = None
        lock = None

    run = FakeRun()
    run.segments = specs
    run.audio_master = master
    run.shots = (FakeShot("seg_01", "/c1.mp4", 3.0),
                 FakeShot("seg_02", "/c2.mp4", 3.0),
                 FakeShot("seg_03", "/c3.mp4", 3.0))
    plan = build_postproduction_plan(run)
    assert plan.edl.segment_ids == ("seg_01", "seg_02", "seg_03")
    assert plan.audio_master_digest == master.digest


def test_an_empty_shot_plan_is_refused():
    with pytest.raises(PlanRefused, match="empty shot plan"):
        build_postproduction_plan((), ())


def test_a_plan_cannot_be_locked_while_a_segment_is_out_for_regeneration(specs,
                                                                        master):
    takes = [t for t in _takes(specs) if t.segment_id != "seg_02"]
    plan = build_postproduction_plan(specs, takes, master)
    assert plan.open_segments == ("seg_02",)
    with pytest.raises(PlanRefused, match="missing from the cut"):
        plan.lock()

    complete = build_postproduction_plan(specs, _takes(specs), master)
    assert complete.lock().locked is True


# ---------------------------------------------------------------------------
# [4] evaluate_take — rubric + speech + duration, composed
# ---------------------------------------------------------------------------


def test_evaluate_take_composes_rubric_speech_and_duration(specs, lines):
    judge = FakeJudge()
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0],
                         judge=judge,
                         transcribe=_transcribe_of("Hello there friend"),
                         lines=lines)
    names = [c.name for c in card.checks]
    assert names == ["shot.intent", "shot.action", "shot.identity",
                     "shot.temporal", "speech.lines_present",
                     "sync.duration_fit"]
    assert card.hard_pass is True
    assert card.confidence == 1.0
    assert len(card.judge_results) == 4


def test_the_judge_prompt_carries_the_shot_specification(specs):
    judge = FakeJudge()
    evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    intent = judge.requests[0].prompt
    assert "SHOT SPECIFICATION for segment seg_01 (scene sc_01)" in intent
    assert "crosses to the window" in intent
    assert "state before: coat=on" in intent
    assert "state after: coat=off" in intent
    assert "must change during the shot: coat" in intent
    assert "ALEX is visible" in intent          # the acceptance rubric
    action = next(r for r in judge.requests if r.name == "shot.action")
    assert "must have changed: coat" in action.prompt


def test_an_image_take_is_never_asked_about_flicker(specs):
    judge = FakeJudge()
    take = replace(_take("seg_01", duration=3.0), kind="image")
    card = evaluate_take(take, specs[0], judge=judge)
    assert "shot.temporal" not in judge.names
    assert "shot.temporal" not in [c.name for c in card.checks]


def test_a_failing_intent_rubric_maps_to_intent_mismatch(specs):
    judge = FakeJudge({"shot.intent": {"verdict": "NO", "score": 12,
                                       "why": "this is a street, not a kitchen"}})
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.INTENT_MISMATCH
    assert "street" in card.diagnosis
    assert "canonical SegmentSpec" in card.recommended_repair


def test_a_failing_action_rubric_maps_to_action_missing(specs):
    judge = FakeJudge({"shot.action": {"verdict": "NO", "score": 20}})
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    assert card.repair_code is RepairCode.ACTION_MISSING


def test_a_failing_identity_rubric_maps_to_identity_drift(specs):
    judge = FakeJudge({"shot.identity": {"verdict": "NO", "score": 5}})
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    assert card.repair_code is RepairCode.IDENTITY_DRIFT


def test_a_failing_temporal_rubric_maps_to_temporal_artifact(specs):
    judge = FakeJudge({"shot.temporal": {"verdict": "NO", "score": 30,
                                         "why": "the hand mutates"}})
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    assert card.repair_code is RepairCode.TEMPORAL_ARTIFACT
    assert "bumped seed" in card.recommended_repair


def test_a_take_shorter_than_its_locked_window_is_shot_too_short(specs):
    card = evaluate_take(_take("seg_01", duration=1.0), specs[0],
                         judge=FakeJudge())
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.SHOT_TOO_SHORT
    assert "the locked audio timeline is authoritative" in card.diagnosis


def test_an_omitted_line_is_line_omitted(specs, lines):
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0],
                         judge=FakeJudge(),
                         transcribe=_transcribe_of("Hello friend"),
                         lines=lines)
    # "there" was dropped; a 3-token line has a zero miss budget (k98).
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.LINE_OMITTED


def test_speech_evidence_is_only_produced_when_a_transcriber_is_given(specs,
                                                                     lines):
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0],
                         judge=FakeJudge(), lines=lines)
    assert "speech.lines_present" not in [c.name for c in card.checks]


def test_a_shot_with_no_dialogue_records_unscored_speech_evidence(specs):
    card = evaluate_take(_take("seg_03", duration=3.0), specs[2],
                         judge=FakeJudge(), transcribe=_transcribe_of("x"))
    check = next(c for c in card.checks if c.name == "speech.lines_present")
    assert speech.is_unscored(check) and check.passed is True
    assert card.hard_pass is True
    assert card.confidence < 1.0


def test_an_unreachable_judge_is_unscored_and_never_fails_the_take(specs):
    judge = FakeJudge({"shot.intent": None})
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    check = next(c for c in card.checks if c.name == "shot.intent")
    assert speech.is_unscored(check) and check.passed is True
    assert card.hard_pass is True
    assert card.confidence < 1.0
    assert card.judge_results[0].verdict == "unavailable"


def test_a_judge_that_raises_degrades_instead_of_exploding(specs):
    judge = FakeJudge(raise_on=("shot.action",))
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    check = next(c for c in card.checks if c.name == "shot.action")
    assert speech.is_unscored(check)
    assert "vision plane is down" in check.detail
    assert card.hard_pass is True


def test_a_verdict_of_no_over_a_passing_score_is_a_disagreement(specs):
    judge = FakeJudge({"shot.intent": {"verdict": "NO", "score": 88}})
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    assert card.hard_pass is True
    assert card.disagreements and "score wins" in card.disagreements[0]


def test_a_plain_reply_string_is_parsed_with_the_k90c_parser(specs):
    judge = FakeJudge({"shot.intent": "VERDICT=NO; SCORE=11; WHY=wrong room"})
    card = evaluate_take(_take("seg_01", duration=3.0), specs[0], judge=judge)
    assert card.repair_code is RepairCode.INTENT_MISMATCH
    assert card.judge_results[0].rationale == "wrong room"


def test_the_quality_bar_is_k90c_s_own(specs):
    judge = FakeJudge({"shot.intent": {"verdict": "YES", "score": 50}})
    balanced = evaluate_take(_take("seg_01", duration=3.0), specs[0],
                             judge=judge, quality=QualityProfile.BALANCED)
    preview = evaluate_take(_take("seg_01", duration=3.0), specs[0],
                            judge=judge, quality=QualityProfile.PREVIEW)
    assert balanced.hard_pass is False and balanced.repair_code is \
        RepairCode.INTENT_MISMATCH
    assert preview.hard_pass is True
    assert BALANCED == 60


def test_intent_outranks_a_duration_miss_when_both_fail(specs):
    judge = FakeJudge({"shot.intent": {"verdict": "NO", "score": 3}})
    card = evaluate_take(_take("seg_01", duration=0.5), specs[0], judge=judge)
    assert card.repair_code is RepairCode.INTENT_MISMATCH
    assert pp.repair_codes_from(card) == (RepairCode.INTENT_MISMATCH,
                                          RepairCode.SHOT_TOO_SHORT)


# ---------------------------------------------------------------------------
# [5] the generator never judges itself
# ---------------------------------------------------------------------------


def test_evaluate_take_refuses_when_the_judge_is_the_generator(specs):
    take = _take("seg_01", duration=3.0, generator_model="Wan2.1-I2V-14B")
    judge = FakeJudge(model="wan2.1-i2v-14b")     # same model, different case
    with pytest.raises(JudgeConflict, match="GENERATED it") as caught:
        evaluate_take(take, specs[0], judge=judge)
    assert caught.value.segment_id == "seg_01"
    assert judge.requests == []                   # refused BEFORE a dispatch


def test_the_refusal_also_fires_on_an_explicit_judge_model(specs):
    take = _take("seg_01", duration=3.0, generator_model="sd-turbo")
    with pytest.raises(JudgeConflict):
        evaluate_take(take, specs[0], judge=FakeJudge(model=None),
                      judge_model="sd-turbo")


def test_a_different_judge_model_is_allowed(specs):
    take = _take("seg_01", duration=3.0, generator_model="wan2.1")
    card = evaluate_take(take, specs[0], judge=FakeJudge(model="qwen2.5-vl"))
    assert card.hard_pass is True


def test_an_unknown_generator_cannot_be_refused_but_is_still_judged(specs):
    take = _take("seg_01", duration=3.0, generator_model=None)
    card = evaluate_take(take, specs[0], judge=FakeJudge(model="qwen2.5-vl"))
    assert card.hard_pass is True


# ---------------------------------------------------------------------------
# [6] bind_live_judge — the catalog's own resolution, reused
# ---------------------------------------------------------------------------


@pytest.fixture()
def still(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return str(path)


def _route(capability, execution="execute", model_id="qwen2.5-vl",
           reasons=()):
    return router.RouteDecision(capability=capability, execution=execution,
                                task="image-text-to-text", model_id=model_id,
                                reasons=tuple(reasons))


def test_live_judge_runs_through_the_k90c_judge_path(monkeypatch, still):
    seen = {}

    def fake_resolve(capability):
        seen.setdefault("capabilities", []).append(capability)
        return _route(capability) if capability == "image.understand" else None

    class Reply:
        ok = True
        text = "VERDICT=YES; SCORE=81; WHY=matches the shot"

    monkeypatch.setattr(evaluation, "_resolve_judge_route", fake_resolve)
    def fake_dispatch(task, body):
        seen["task"] = task
        seen["body"] = body
        return Reply()

    monkeypatch.setattr(evaluation, "_judge_dispatch", fake_dispatch)

    judge = bind_live_judge()
    assert judge.model == "qwen2.5-vl"
    request = JudgeRequest(name="shot.intent", kind=CheckKind.INTENT,
                           prompt="SHOT SPECIFICATION ...", artifact_ref=still,
                           artifact_kind="image", segment_id="seg_01")
    result = judge(request)
    assert result.verdict == "YES" and result.score == 81.0
    assert "SHOT SPECIFICATION" in seen["body"]["prompt"]
    assert seen["body"]["file"] == still
    assert seen["body"]["model_key"] == "qwen2.5-vl"    # the catalog's choice
    assert seen["capabilities"] == ["image.understand", "image.understand"]


def test_live_judge_reports_the_missing_clip_evaluator_honestly(monkeypatch,
                                                               still):
    def fake_resolve(capability):
        if capability == "image.understand":
            return _route(capability)
        return _route(capability, execution="gap", model_id=None,
                      reasons=("no model serves video.understand",))

    monkeypatch.setattr(evaluation, "_resolve_judge_route", fake_resolve)
    judge = bind_live_judge()
    result = judge(JudgeRequest(name="shot.temporal", kind=CheckKind.TEMPORAL,
                                prompt="p", artifact_ref="/clips/a.mp4",
                                artifact_kind="video"))
    assert result.verdict == "unavailable"
    assert "video.understand is not eligible" in result.rationale
    assert "frames=" in result.rationale


def test_live_judge_samples_a_still_when_a_frames_seam_is_bound(monkeypatch,
                                                               still):
    def fake_resolve(capability):
        return _route(capability) if capability == "image.understand" else None

    class Reply:
        ok = True
        text = "VERDICT=NO; SCORE=22; WHY=the hand mutates"

    monkeypatch.setattr(evaluation, "_resolve_judge_route", fake_resolve)
    monkeypatch.setattr(evaluation, "_judge_dispatch", lambda task, body: Reply())
    judge = bind_live_judge(frames=lambda ref: still)
    result = judge(JudgeRequest(name="shot.temporal", kind=CheckKind.TEMPORAL,
                                prompt="p", artifact_ref="/clips/a.mp4",
                                artifact_kind="video"))
    assert result.verdict == "NO" and result.score == 22.0


def test_an_unbound_live_judge_answers_unavailable_rather_than_raising(
        monkeypatch):
    monkeypatch.setattr(evaluation, "_resolve_judge_route", lambda cap: None)
    judge = bind_live_judge()
    assert judge.model is None
    result = judge(JudgeRequest(name="shot.intent", kind=CheckKind.INTENT,
                                prompt="p", artifact_ref="/clips/a.png",
                                artifact_kind="image"))
    assert result.verdict == "unavailable"


# ---------------------------------------------------------------------------
# [7] final_consistency_report — the whole result
# ---------------------------------------------------------------------------


def _report(plan, specs, lines, master, **kw):
    kw.setdefault("shots", specs)
    kw.setdefault("lines", lines)
    kw.setdefault("audio_master", master)
    return final_consistency_report(plan, **kw)


def test_a_complete_cut_passes_the_final_report(specs, lines, master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    report = _report(plan, specs, lines, master,
                     takes=_takes(specs),
                     transcribe=_transcribe_of("Hello there friend "
                                               "The kettle is boiling"),
                     video_ref="/out/final.mp4")
    assert report.ok is True
    assert report.scorecard.confidence == 1.0
    assert report.uncovered_segments == ()
    assert any(line.startswith("VERDICT: PASS") for line in report.summary)


def test_the_report_catches_an_uncovered_segment_and_names_its_code(specs,
                                                                   lines,
                                                                   master):
    rejected = Take(segment_id="seg_02", attempt_id="seg_02:a1",
                    artifact_ref="/clips/seg_02_a1.mp4", duration_s=3.0,
                    accepted=False, repair_codes=(RepairCode.ACTION_MISSING,),
                    diagnosis="ALEX never crosses")
    takes = [t for t in _takes(specs) if t.segment_id != "seg_02"] + [rejected]
    plan = build_postproduction_plan(specs, takes, master)
    report = _report(plan, specs, lines, master, takes=takes)
    assert report.ok is False
    assert report.uncovered_segments == ("seg_02",)
    assert report.repair_code is RepairCode.ACTION_MISSING   # from the note
    assert any("MISSING seg_02" in line for line in report.summary)
    assert report.open_segments == ("seg_02",)


def test_the_report_catches_an_omitted_line_by_name(specs, lines, master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    report = _report(plan, specs, lines, master, takes=_takes(specs),
                     transcribe=_transcribe_of("Hello there friend "
                                               "The kettle is"),
                     video_ref="/out/final.mp4")
    assert report.ok is False
    assert report.repair_code is RepairCode.LINE_OMITTED
    assert report.omitted_line_ids == ("L2",)
    assert any("OMITTED L2" in line for line in report.summary)


def test_the_report_catches_a_pacing_outlier(specs, lines, master):
    short = _take("seg_02", duration=1.0)
    takes = [short] + [t for t in _takes(specs) if t.segment_id != "seg_02"]
    plan = build_postproduction_plan(specs, takes, master)
    report = _report(plan, specs, lines, master, takes=takes)
    assert report.ok is False
    assert report.pacing_outliers == (("seg_02", -2.0),)
    assert report.repair_code is RepairCode.SHOT_TOO_SHORT
    assert any("deviates -2.000s" in line for line in report.summary)


def test_the_report_catches_a_scene_with_no_grading_target(specs, lines,
                                                          master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    stripped = replace(plan, color_targets=FrozenParams({}))
    report = _report(stripped, specs, lines, master, takes=_takes(specs))
    assert report.ok is False
    assert report.missing_color_scenes == ("sc_02",)
    assert report.repair_code is RepairCode.INTENT_MISMATCH
    assert any("MISSING sc_02" in line for line in report.summary)


def test_the_report_catches_a_picture_shorter_than_its_audio_bed(specs, lines,
                                                                master):
    """k106's documented trap: audio-derived windows with pad_s=0 do not cover
    the inter-line pauses, so the picture comes up short against the master."""
    trimmed = tuple(_spec(s.segment_id, s.index, s.start_s, s.end_s - 0.5,
                          line_ids=s.line_ids,
                          scene=s.scene_ref,
                          lighting=s.shot.lighting)
                    for s in specs)
    takes = tuple(_take(s.segment_id, duration=s.duration_s) for s in trimmed)
    plan = build_postproduction_plan(trimmed, takes, master)
    report = _report(plan, trimmed, lines, master, takes=takes)
    bed = next(c for c in report.scorecard.checks if c.name == "audio.bed_fit")
    assert bed.passed is False
    assert any("comes up SHORT" in line for line in report.summary)


def test_missing_evidence_is_unscored_rather_than_a_pass(specs, lines, master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    report = final_consistency_report(plan)     # no shots, no lines, no ASR
    names = {c.name for c in report.scorecard.checks}
    assert {"speech.lines_present", "pacing.window_fit", "color.targets"} <= names
    assert all(c.passed for c in report.scorecard.checks)
    assert report.scorecard.confidence < 1.0
    assert report.ok is True
    assert any("UNSCORED" in line for line in report.summary)


def test_a_transcriber_with_nothing_to_transcribe_is_unscored(specs, lines,
                                                              master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    report = _report(plan, specs, lines, master,
                     transcribe=_transcribe_of("anything"))   # no video_ref
    check = next(c for c in report.scorecard.checks
                 if c.name == "speech.lines_present")
    assert speech.is_unscored(check)
    assert "re-transcribe" in check.detail


def test_the_report_round_trips_to_a_dict(specs, lines, master):
    plan = build_postproduction_plan(specs, _takes(specs), master)
    report = _report(plan, specs, lines, master, takes=_takes(specs))
    data = report.to_dict()
    assert data["ok"] is report.ok
    assert Scorecard.from_dict(data["scorecard"]) == report.scorecard
    assert data["summary"] == list(report.summary)


def test_final_report_refuses_something_that_is_not_a_plan():
    with pytest.raises(PostProductionError, match="PostProductionPlan"):
        final_consistency_report({"edl": []})


# ---------------------------------------------------------------------------
# [8] ShotBrief — the one shape everything reads a shot through
# ---------------------------------------------------------------------------


def test_shot_brief_reads_a_segment_spec_a_plan_entry_and_a_mapping(specs):
    from_spec = shot_brief(specs[0])
    assert from_spec.spec_digest == specs[0].digest
    assert from_spec.scene_ref == "sc_01"
    assert from_spec.characters == ("ALEX",)
    assert from_spec.duration_s == 3.0

    from_entry = shot_brief(specs[0].shot)
    assert from_entry.segment_id == "seg_01"
    assert from_entry.spec_digest is None      # an entry is not the spec

    from_mapping = shot_brief({"segment_id": "seg_09", "start_s": 0.0,
                               "end_s": 1.0})
    assert from_mapping.segment_id == "seg_09"
    assert shot_brief(from_mapping) is from_mapping


def test_shot_brief_refuses_an_unknown_shape_and_an_unknown_key():
    with pytest.raises(PostProductionError, match="cannot read"):
        shot_brief(object())
    with pytest.raises(PostProductionError, match="unknown key"):
        shot_brief({"segment_id": "seg_01", "colour": "warm"})


def test_shot_briefs_preserve_the_shot_plans_own_order(specs):
    briefs = shot_briefs(ShotPlan(entries=tuple(s.shot for s in specs)))
    assert [b.segment_id for b in briefs] == ["seg_01", "seg_02", "seg_03"]


def test_color_cues_are_whole_tokens(specs):
    assert pp.has_color_cue("warm amber practicals") is True
    assert pp.has_color_cue("she is discolored") is False
    assert pp.has_color_cue(("soft key",), "a graded look") is True


# ---------------------------------------------------------------------------
# 7. Spatial continuity across cuts (or-k13)
# ---------------------------------------------------------------------------
#
# [7] adjacent shots that share a locked camera track / world must agree about
#     camera pose and entity placement at the cut; a contradiction is a
#     continuity correction on the later shot AND a RegenerationNote carrying
#     CAMERA_PATH_MISMATCH / GEOMETRY_DRIFT for the repair controller.


def _spatial_spec(segment_id, index, start, end, *, before, after,
                  spatial_ref="scene://kitchen/v3", scene="sc_01"):
    spec = _spec(segment_id, index, start, end, scene=scene,
                 before=before, after=after)
    return replace(spec, spatial_ref=spatial_ref)


_CAM_A = {"position": [0.0, 1.6, 3.0], "forward": [0.0, 0.0, -1.0],
          "track_uri": "track://kitchen/dolly"}
_ENTS = {"ALEX": [0.5, 0.0, 0.0], "TABLE": [-1.0, 0.0, -0.5]}


def _pair(cam_b=None, ents_b=None, *, ref_a="scene://kitchen/v3",
          ref_b="scene://kitchen/v3", cs_b=None, extra_b=None):
    before_b = {"location": "KITCHEN", "present": ["ALEX"],
                "camera_pose": cam_b if cam_b is not None else dict(_CAM_A),
                "entity_positions": ents_b if ents_b is not None else dict(_ENTS)}
    if cs_b is not None:
        before_b["coordinate_system"] = cs_b
    before_b.update(extra_b or {})
    return (
        _spatial_spec("seg_01", 0, 0.0, 3.0, spatial_ref=ref_a,
                      before={"location": "KITCHEN", "present": ["ALEX"]},
                      after={"location": "KITCHEN", "present": ["ALEX"],
                             "camera_pose": dict(_CAM_A),
                             "entity_positions": dict(_ENTS)}),
        _spatial_spec("seg_02", 1, 3.0, 6.0, spatial_ref=ref_b,
                      before=before_b,
                      after={"location": "KITCHEN", "present": ["ALEX"]}),
    )


def test_shot_brief_carries_spatial_ref_from_the_segment_spec():
    spec = _spatial_spec("seg_01", 0, 0.0, 3.0, before={}, after={})
    brief = shot_brief(spec)
    assert brief.spatial_ref == "scene://kitchen/v3"
    assert brief.to_dict()["spatial_ref"] == "scene://kitchen/v3"
    assert shot_brief({"segment_id": "x", "spatial_ref": "scene://y"}).spatial_ref == "scene://y"
    assert shot_brief(_spec("seg_01", 0, 0.0, 3.0)).spatial_ref is None


def test_consistent_adjacent_shots_raise_no_spatial_contradiction():
    briefs = shot_briefs(_pair())
    assert pp.spatial_continuity_checks(briefs) == ()
    plan = build_postproduction_plan(_pair(), _takes(_pair()))
    assert plan.continuity_corrections == ()
    assert plan.regeneration_notes == ()
    assert plan.lock().locked


def test_camera_jump_on_a_shared_track_is_a_camera_path_mismatch():
    cam_b = dict(_CAM_A, position=[0.0, 1.6, 1.0])          # 2 m off the track end
    specs = _pair(cam_b=cam_b)
    (c,) = pp.spatial_continuity_checks(shot_briefs(specs))
    seg, code, note = c
    assert (seg, code) == ("seg_02", RepairCode.CAMERA_PATH_MISMATCH)
    assert c.previous_segment_id == "seg_01"
    assert "2.000 m" in note and "track://kitchen/dolly" in note
    assert c.evidence["distance_m"] == 2.0
    plan = build_postproduction_plan(specs, _takes(specs))
    assert plan.continuity_corrections == (("seg_02", note),)
    (n,) = plan.regeneration_notes
    assert n.segment_id == "seg_02" and n.repair_code is RepairCode.CAMERA_PATH_MISMATCH
    assert n.spec_digest == specs[1].digest
    assert "camera pass" in n.correction and "Do NOT re-prompt" in n.correction
    assert plan.open_segments == ()            # the take is in the cut, with a reservation
    assert plan.regeneration_codes == (RepairCode.CAMERA_PATH_MISMATCH,)


def test_camera_look_direction_alone_trips_the_track():
    cam_b = dict(_CAM_A, forward=[0.2, 0.0, -1.0])           # ~11 deg yaw, no translation
    (c,) = pp.spatial_continuity_checks(shot_briefs(_pair(cam_b=cam_b)))
    assert c.repair_code is RepairCode.CAMERA_PATH_MISMATCH
    assert "look direction" in c.note and c.evidence["angle_deg"] > 10


def test_camera_cut_to_a_different_track_is_not_a_contradiction():
    cam_b = {"position": [5.0, 1.6, -2.0], "forward": [-1.0, 0.0, 0.0],
             "track_uri": "track://kitchen/reverse"}
    assert pp.spatial_continuity_checks(shot_briefs(_pair(cam_b=cam_b))) == ()


def test_camera_spec_dict_names_the_track_like_camera_pose():
    cam_b = {"position": [0.0, 1.6, 0.0], "forward": [0.0, 0.0, -1.0]}
    specs = _pair(cam_b=cam_b, extra_b={"camera_spec": {"track_uri": "track://kitchen/dolly"}})
    (c,) = pp.spatial_continuity_checks(shot_briefs(specs))
    assert c.repair_code is RepairCode.CAMERA_PATH_MISMATCH


def test_camera_drift_threshold_comes_from_drift_thresholds():
    from hugpy_oracle.spatial import DriftThresholds
    cam_b = dict(_CAM_A, position=[0.0, 1.6, 2.9])            # 10 cm
    briefs = shot_briefs(_pair(cam_b=cam_b))
    assert len(pp.spatial_continuity_checks(briefs)) == 1    # default 5 cm
    assert pp.spatial_continuity_checks(
        briefs, thresholds=DriftThresholds(camera_drift_m_max=0.5)) == ()
    plan = build_postproduction_plan(_pair(cam_b=cam_b), _takes(_pair(cam_b=cam_b)),
                                     spatial_thresholds=DriftThresholds(camera_drift_m_max=0.5))
    assert plan.regeneration_notes == ()


def test_entity_teleport_across_the_cut_is_geometry_drift():
    ents_b = dict(_ENTS, TABLE=[2.0, 0.0, -0.5])             # table moved 3 m
    specs = _pair(ents_b=ents_b)
    (c,) = pp.spatial_continuity_checks(shot_briefs(specs))
    assert c.repair_code is RepairCode.GEOMETRY_DRIFT
    assert "TABLE moved 3.00 m" in c.note and "ALEX" not in c.evidence["entities"]
    assert c.evidence["entities"]["TABLE"]["distance_m"] == 3.0
    plan = build_postproduction_plan(specs, _takes(specs))
    assert plan.regeneration_codes == (RepairCode.GEOMETRY_DRIFT,)
    assert plan.note_for("seg_02").repair_code is RepairCode.GEOMETRY_DRIFT
    assert ("seg_02", c.note) in plan.continuity_corrections


def test_entity_positions_are_compared_in_the_canonical_frame():
    # seg_02 authored in a Z-up, centimetre system: (50, 0, 0) cm == (0.5, 0, 0) m
    z_up_cm = {"handedness": "right", "up_axis": "Z", "forward_axis": "Y",
               "world_units": "centimeters"}
    ents_b = {"ALEX": [50.0, 0.0, 0.0], "TABLE": [-100.0, 50.0, 0.0]}
    assert pp.spatial_continuity_checks(
        shot_briefs(_pair(ents_b=ents_b, cs_b=z_up_cm,
                          cam_b={"position": [0.0, -300.0, 160.0],
                                 "forward": [0.0, 1.0, 0.0],
                                 "track_uri": "track://kitchen/dolly"}))) == ()
    ents_b["TABLE"] = [-100.0, -50.0, 0.0]                   # 1 m off once converted
    (c,) = pp.spatial_continuity_checks(
        shot_briefs(_pair(ents_b=ents_b, cs_b=z_up_cm,
                          cam_b={"position": [0.0, -300.0, 160.0],
                                 "forward": [0.0, 1.0, 0.0],
                                 "track_uri": "track://kitchen/dolly"})))
    assert c.repair_code is RepairCode.GEOMETRY_DRIFT
    assert c.evidence["entities"]["TABLE"]["distance_m"] == 1.0


def test_different_spatial_refs_are_different_worlds():
    ents_b = dict(_ENTS, TABLE=[9.0, 0.0, 0.0])
    cam_b = dict(_CAM_A, position=[9.0, 1.6, 0.0], track_uri="track://other")
    specs = _pair(ents_b=ents_b, cam_b=cam_b, ref_b="scene://garden/v1")
    assert pp.spatial_continuity_checks(shot_briefs(specs)) == ()


def test_shots_without_spatial_information_are_skipped(specs):
    assert pp.spatial_continuity_checks(shot_briefs(specs)) == ()
    plan = build_postproduction_plan(specs, _takes(specs))
    assert plan.regeneration_notes == () and plan.continuity_corrections == ()


def test_both_axes_fail_gives_two_codes_and_one_note_each():
    specs = _pair(cam_b=dict(_CAM_A, position=[3.0, 1.6, 3.0]),
                  ents_b=dict(_ENTS, ALEX=[4.0, 0.0, 0.0]))
    found = pp.spatial_continuity_checks(shot_briefs(specs))
    assert [c.repair_code for c in found] == [RepairCode.CAMERA_PATH_MISMATCH,
                                              RepairCode.GEOMETRY_DRIFT]
    plan = build_postproduction_plan(specs, _takes(specs))
    assert len(plan.continuity_corrections) == 2
    assert set(plan.regeneration_codes) == {RepairCode.CAMERA_PATH_MISMATCH,
                                            RepairCode.GEOMETRY_DRIFT}
    for note in plan.regeneration_notes:
        assert note.segment_id == "seg_02" and note.spec_digest == specs[1].digest


def test_spatial_note_does_not_duplicate_a_take_that_already_carries_the_code():
    specs = _pair(ents_b=dict(_ENTS, ALEX=[4.0, 0.0, 0.0]))
    rejected = _take("seg_02", accepted=False, diagnosis="geometry off",
                     repair_codes=(RepairCode.GEOMETRY_DRIFT,))
    plan = build_postproduction_plan(specs, _takes(specs, seg_02=rejected))
    notes = [n for n in plan.regeneration_notes if n.segment_id == "seg_02"]
    assert len(notes) == 1 and notes[0].repair_code is RepairCode.GEOMETRY_DRIFT
    assert notes[0].rejected_take_ref == rejected.artifact_ref   # the take's own note won
    assert any(s == "seg_02" and "entity placement" in n
               for s, n in plan.continuity_corrections)
    assert plan.open_segments == ("seg_02",)


def test_spatial_checks_can_be_switched_off_and_round_trip():
    specs = _pair(cam_b=dict(_CAM_A, position=[3.0, 1.6, 3.0]))
    plan = build_postproduction_plan(specs, _takes(specs), spatial_checks=False)
    assert plan.regeneration_notes == ()
    plan = build_postproduction_plan(specs, _takes(specs))
    again = PostProductionPlan.from_dict(plan.to_dict())
    assert again.digest == plan.digest
    assert again.regeneration_codes == (RepairCode.CAMERA_PATH_MISMATCH,)
    (c,) = pp.spatial_continuity_checks(shot_briefs(specs))
    d = c.to_dict()
    assert d["repair_code"] == "camera_path_mismatch" and d["evidence"]["check"] == "camera_pose"
    assert pp.SpatialContradiction(**{**d, "evidence": d["evidence"]}).note == c.note
