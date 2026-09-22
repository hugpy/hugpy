"""k106 — the ``video.performance`` FAT orchestrator, end to end under fakes.

Everything here is OFFLINE and deterministic: every backend is an injected
seam, so the doc's Phase-5 recipe runs with no GPU, no TTS seat, no worker, no
network and no clock dependency. The run journal is the only disk touched, and
it lands under a pytest ``tmp_path`` via ``PerformanceSeams.run_root``.

Locks:
  [1] the happy path really is end to end: two identities, three lines, sibling
      parents, per-shot scorecards, and every locked line accounted for in the
      FINAL round-trip transcript.
  [2] a missing TTS seat stops at stage 3 with a typed CAPABILITY_GAP naming
      ``audio.tts`` and the operator step — and NOTHING after stage 3 runs.
  [3] the authority gate is stage 1: a request without the grants never reaches
      a synthesizer, and a resumed run re-checks it rather than trusting the
      journal.
  [4] repairs are bounded and TARGETED: an identity failure re-runs the
      keyframe and only the keyframe (counted at the seams), a clip failure
      gets exactly ONE repair round and then fails honestly.
  [5] resume skips completed stages only after re-deriving their artifacts and
      comparing DIGESTS; a tampered journal, a moved artifact or a different
      goal re-executes instead.
  [6] a plan that does not statically validate aborts with the report — never
      "probably fine".
  [7] ``limitations`` is always populated, and says the true things: no
      lip-sync evaluator, similarity unscored, no spatial conditioning.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_performance.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import pytest

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import performance as perf
from hugpy_oracle.audio_master import DialogueTimeline, Line, SpeechPolicy, VoiceProfile
from hugpy_oracle.contracts import (
    ArtifactKind,
    AuthorityKind,
    Authorization,
    CapabilityView,
    Eligibility,
    GoalSpec,
    QualityProfile,
    RepairCode,
    RightsManifest,
    SourceRegistry,
)
from hugpy_oracle.production import ContinuityState


# ---------------------------------------------------------------------------
# Fixtures — two identities, three lines
# ---------------------------------------------------------------------------

LINES = (
    Line(line_id="l1", speaker="ana", text="We should not have come here.",
         max_seconds=4.0),
    Line(line_id="l2", speaker="ben", text="The map said the road was clear.",
         max_seconds=4.0),
    Line(line_id="l3", speaker="ana", text="The map was drawn before the flood.",
         max_seconds=4.0),
)
TEXTS = {line.line_id: line.text for line in LINES}
ANA = "identity_profile:ana"
BEN = "identity_profile:ben"


def rights(*subjects: str, kinds=(AuthorityKind.LIKENESS, AuthorityKind.VOICE)
           ) -> RightsManifest:
    return RightsManifest(authorizations=tuple(
        Authorization(kind=kind, subject=subject, scope="one performance",
                      evidence="release-2026-08", granted_by="operator")
        for subject in subjects for kind in kinds))


def goal_spec(*, manifest: RightsManifest | None = None, **kw) -> GoalSpec:
    base = dict(
        objective="two characters, three lines, 15-30 seconds",
        raw_prompt=(f"ana ({ANA}) and ben ({BEN}) argue on a flooded road; "
                    f"three lines, hand-held, overcast"),
        rights=manifest if manifest is not None else rights(ANA, BEN),
    )
    base.update(kw)
    return GoalSpec(**base)


def performance_goal(**kw) -> perf.PerformanceGoal:
    base = dict(
        goal=goal_spec(),
        dialogue=DialogueTimeline(lines=LINES),
        casting={"ana": VoiceProfile.synthetic("ana_voice", timbre="dry"),
                 "ben": VoiceProfile.synthetic("ben_voice", timbre="warm")},
        raw_request_ref="mct://request/k106",
        identity_refs=(ANA, BEN),
        deliverable="a 15-30 second two-hander",
        speech_policy=SpeechPolicy(pause_after_s=0.35),
        pad_s=0.35,           # windows absorb the pauses -> they partition
        created_at="2026-08-20T00:00:00+00:00",
    )
    base.update(kw)
    return perf.PerformanceGoal(**base)


def catalog_view(*, eligible: bool = True) -> dict:
    return {
        "video.generate.i2v": CapabilityView(
            name="video.generate.i2v", source=SourceRegistry.STUDIO,
            accepts=(ArtifactKind.JSON, ArtifactKind.IMAGE),
            produces=(ArtifactKind.VIDEO,), model_ids=("wan-i2v",),
            eligibility=Eligibility(
                eligible=eligible,
                reasons=() if eligible else ("no worker seats i2v",))),
    }


class Fakes:
    """Counting fakes for every seam. The counters ARE the assertions for
    "a targeted failure does not rerun unrelated nodes"."""

    def __init__(self, *, keyframe_verdicts=None, clip_verdicts=None,
                 clip_seconds=None, synth_delay=0.0):
        self.calls = {name: 0 for name in perf.SEAM_NAMES}
        self.video = "video://performance/final.mp4"
        self.keyframe_verdicts = keyframe_verdicts
        self.clip_verdicts = clip_verdicts
        self.clip_seconds = clip_seconds
        self.synth_delay = synth_delay
        self.seen_keyframe_seeds: list[int] = []
        self.seen_clip_seeds: list[int] = []

    # -- seams ------------------------------------------------------------

    def synth(self, line, voice, seed):
        self.calls["synth"] += 1
        if self.synth_delay:
            time.sleep(self.synth_delay)
        return (f"audio://{line.line_id}#{seed}", 2.5)

    def transcribe(self, ref):
        self.calls["transcribe"] += 1
        text = str(ref)
        if text.startswith("audio://"):
            line_id = text.split("//", 1)[1].split("#", 1)[0]
            return TEXTS[line_id].split()
        return [word for line in LINES for word in line.text.split()]

    def gen_image(self, prompt, identity_refs, seed):
        self.calls["gen_image"] += 1
        self.seen_keyframe_seeds.append(seed)
        return f"image://keyframe#{seed}"

    def judge_image(self, ref, spec):
        self.calls["judge_image"] += 1
        if self.keyframe_verdicts is None:
            return {"verdict": "YES", "score": 88, "why": "identity reads"}
        index = min(self.calls["judge_image"] - 1,
                    len(self.keyframe_verdicts) - 1)
        return self.keyframe_verdicts[index]

    def gen_clip(self, keyframe_ref, spec):
        self.calls["gen_clip"] += 1
        self.seen_clip_seeds.append(spec.seed_base)
        seconds = (self.clip_seconds if self.clip_seconds is not None
                   else spec.duration_s)
        return (f"clip://{spec.segment_id}#{spec.seed_base}", seconds)

    def judge_clip(self, ref, spec):
        self.calls["judge_clip"] += 1
        if self.clip_verdicts is None:
            return {"verdict": "YES", "score": 81, "why": "action present"}
        index = min(self.calls["judge_clip"] - 1, len(self.clip_verdicts) - 1)
        return self.clip_verdicts[index]

    def concat(self, clip_refs, master):
        self.calls["concat"] += 1
        return self.video

    # -- wiring -----------------------------------------------------------

    def seams(self, root, **overrides) -> perf.PerformanceSeams:
        base = dict(
            synth=self.synth, transcribe=self.transcribe,
            gen_image=self.gen_image, judge_image=self.judge_image,
            gen_clip=self.gen_clip, judge_clip=self.judge_clip,
            concat=self.concat,
            registry_version=lambda: "rv-k106-test",
            catalog_view=catalog_view,
            run_root=str(root))
        base.update(overrides)
        return perf.PerformanceSeams(**base)


@pytest.fixture()
def fakes():
    return Fakes()


@pytest.fixture()
def happy(tmp_path, fakes):
    """A completed happy-path run plus the fakes that produced it."""
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    return result, fakes


# ---------------------------------------------------------------------------
# 1. Happy path, end to end
# ---------------------------------------------------------------------------


def test_happy_path_produces_a_deliverable(happy):
    result, _fakes = happy
    assert result.ok is True
    assert result.gap is None
    assert result.video_ref == "video://performance/final.mp4"
    assert result.stopped_after is None


def test_happy_path_runs_every_stage_once_and_none_resumed(happy):
    result, _fakes = happy
    assert result.stage_names == perf.STAGES
    assert all(record.ok for record in result.stages)
    assert not any(record.resumed for record in result.stages)


def test_happy_path_two_identities_three_lines_three_shots(happy):
    result, _fakes = happy
    assert len(result.audio_master.line_timings) == 3
    assert len(result.segments) == 3
    assert len(result.shots) == 3
    assert set(result.segments[0].identity_refs) == {ANA, BEN}


def test_segments_are_siblings_not_a_chain(happy):
    result, _fakes = happy
    parents = set(result.lock.parent_digests)
    spec_digests = {spec.digest for spec in result.segments}
    for spec in result.segments:
        assert set(spec.parents) == parents
        assert not spec_digests & set(spec.parents)
        assert spec.lock_digest == result.lock.digest


def test_every_shot_carries_its_own_scorecard(happy):
    result, _fakes = happy
    assert [shot.segment_id for shot in result.shots] == ["s1", "s2", "s3"]
    for shot in result.shots:
        assert shot.accepted is True
        assert shot.scorecard is not None and shot.scorecard.hard_pass
        assert shot.keyframe_scorecard is not None
        assert shot.keyframe_ref and shot.clip_ref


def test_final_transcript_accounts_for_every_locked_line(happy):
    result, _fakes = happy
    check = next(c for c in result.scorecard.checks
                 if c.name == "speech.lines_present")
    assert check.passed is True
    assert (check.value, check.threshold) == (3, 3)


def test_whole_result_scorecard_names_every_shot(happy):
    result, _fakes = happy
    names = {c.name for c in result.scorecard.checks}
    assert {"shot.s1", "shot.s2", "shot.s3"} <= names
    assert result.scorecard.hard_pass is True


def test_candidate_fan_out_is_three_per_line_and_per_shot(happy):
    result, fakes = happy
    assert fakes.calls["synth"] == 3 * len(LINES)          # tts_candidates
    assert fakes.calls["gen_image"] == 3                   # 1 accepted per shot
    assert fakes.calls["gen_clip"] == 3
    assert fakes.calls["concat"] == 1
    assert result.audio_master.candidates_considered == 9


def test_registry_version_rides_on_the_result_and_every_receipt(happy):
    result, _fakes = happy
    assert result.registry_version == "rv-k106-test"
    assert result.lock.registry_version == "rv-k106-test"
    assert result.audio_master.registry_version == "rv-k106-test"
    # Stage 1 runs BEFORE the catalog is read, so its receipt honestly records
    # no registry version; every later stage carries one.
    assert result.receipts[0].registry_version is None
    assert {r.registry_version for r in result.receipts[1:]} == \
        {"rv-k106-test"}


def test_each_stage_journals_its_artifact_digests(happy):
    result, _fakes = happy
    digests = result.digests
    assert digests["snapshot.snapshot"] == result.snapshot.digest
    assert digests["audio.audio_master"] == result.audio_master.digest
    assert digests["lock.production_lock"] == result.lock.digest
    assert digests["segments.segment:s1"] == result.segments[0].digest
    assert "assembly.deliverable" in digests


def test_one_receipt_per_stage_naming_the_right_capability(happy):
    result, _fakes = happy
    assert len(result.receipts) == len(perf.STAGES)
    assert [r.capability for r in result.receipts] == [
        perf.STAGE_CAPABILITY[s] for s in perf.STAGES]
    assert all(r.failure is None for r in result.receipts)


def test_artifact_refs_carry_audio_keyframes_clips_and_the_deliverable(happy):
    result, _fakes = happy
    refs = set(result.artifact_refs)
    assert any(r.startswith("audio://") for r in refs)
    assert any(r.startswith("image://") for r in refs)
    assert any(r.startswith("clip://") for r in refs)
    assert result.video_ref in refs


def test_the_run_is_deterministic(tmp_path):
    first = perf.run_performance(performance_goal(),
                                 seams=Fakes().seams(tmp_path / "a"))
    second = perf.run_performance(performance_goal(),
                                  seams=Fakes().seams(tmp_path / "b"))
    assert first.lock.digest == second.lock.digest
    assert [s.digest for s in first.segments] == \
           [s.digest for s in second.segments]
    assert first.run_id == second.run_id


def test_limitations_are_populated_and_honest(happy):
    result, _fakes = happy
    blob = " | ".join(result.limitations)
    assert result.limitations
    assert "lip-sync" in blob
    assert "similarity is UNSCORED" in blob
    assert "spatial" in blob
    assert "k111 DAG runtime" in blob


def test_the_run_journal_lands_under_the_movie_run_convention(happy, tmp_path):
    result, _fakes = happy
    assert result.state_path.endswith(
        os.path.join("runs", "performance", result.run_id, "state.json"))
    assert os.path.isfile(result.state_path)
    with open(result.state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    assert set(state["stages"]) == set(perf.RESUMABLE_STAGES)
    assert state["goal_digest"] == result.goal_digest


# ---------------------------------------------------------------------------
# 2. No TTS seat — the honest refusal this fleet gives today
# ---------------------------------------------------------------------------


def test_missing_tts_seat_gaps_at_stage_three(tmp_path, fakes):
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path, synth=None))
    assert result.ok is False
    assert result.gap is not None
    assert result.gap.stage == "audio"
    assert result.gap.repair_codes == (RepairCode.CAPABILITY_GAP,)
    assert result.gap.capability == "audio.tts"


def test_missing_tts_seat_runs_nothing_after_stage_three(tmp_path, fakes):
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path, synth=None))
    assert result.stage_names == ("authority", "snapshot", "audio")
    assert fakes.calls["gen_image"] == 0
    assert fakes.calls["gen_clip"] == 0
    assert fakes.calls["concat"] == 0
    assert result.lock is None and result.segments == ()


def test_an_unbound_synth_seam_still_names_the_chatterbox_seat(tmp_path, fakes):
    """The seat EXISTS now (chatterbox on a-brain, 2026-08-21), so a seam set
    that leaves ``synth`` unbound is not facing a fleet gap — it unbound
    something that works. The requirement says so, and still names the seat, so
    the reader learns what to bind rather than "no implementation is wired"
    (which would now be false)."""
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path, synth=None))
    requirement = result.gap.requirement
    assert "chatterbox" in requirement
    assert "text-to-speech" in requirement
    assert "make_live_synth" in requirement
    assert all(gap.seam != "synth" for gap in perf.LIVE_SEAM_GAPS)


def test_missing_tts_seat_receipt_is_classified(tmp_path, fakes):
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path, synth=None))
    audio_receipt = result.receipts[-1]
    assert audio_receipt.capability == "audio.tts"
    assert audio_receipt.failure is not None
    assert audio_receipt.failure.value == "capability_gap"


def test_default_seams_reports_the_live_bindings_and_gaps():
    """``synth`` moved from the gap list to the bound list when chatterbox was
    seated (2026-08-21). The other three gaps are untouched — a seating is not
    a licence to quietly declare the rest fixed."""
    seams = perf.default_seams()
    assert [name for name in perf.SEAM_NAMES if seams.bound(name)] == \
        ["synth", "transcribe", "gen_image", "judge_image", "concat"]
    assert {gap.seam for gap in seams.unbound} == \
        {"similarity", "gen_clip", "judge_clip"}
    assert seams.synth is perf._live_synth


def test_binding_a_seam_clears_its_recorded_gap():
    seams = perf.default_seams(synth=lambda line, voice, seed: ("a", 1.0))
    assert seams.bound("synth")
    assert all(gap.seam != "synth" for gap in seams.unbound)


def test_audio_gap_when_every_take_is_rejected(tmp_path):
    class Deaf(Fakes):
        def transcribe(self, ref):
            self.calls["transcribe"] += 1
            return []                      # round trip hears nothing

    deaf = Deaf()
    result = perf.run_performance(performance_goal(),
                                  seams=deaf.seams(tmp_path))
    assert result.ok is False
    assert result.gap.stage == "audio"
    assert RepairCode.LINE_OMITTED in result.gap.repair_codes
    assert deaf.calls["gen_image"] == 0


# ---------------------------------------------------------------------------
# 3. Authority — stage 1, before anything is synthesized
# ---------------------------------------------------------------------------


def test_no_rights_manifest_refuses_at_stage_one(tmp_path, fakes):
    goal = performance_goal(goal=GoalSpec(
        objective="two characters, three lines",
        raw_prompt=f"ana ({ANA}) and ben ({BEN}) argue on a flooded road"))
    result = perf.run_performance(goal, seams=fakes.seams(tmp_path))
    assert result.ok is False
    assert result.gap.stage == "authority"
    assert result.gap.repair_codes == (RepairCode.SOURCE_AUTHORITY_MISSING,)
    assert "absence is not consent" in result.gap.diagnosis


def test_authority_refusal_runs_no_seam_at_all(tmp_path, fakes):
    goal = performance_goal(goal=GoalSpec(
        objective="two characters", raw_prompt=f"{ANA} and {BEN} talk"))
    result = perf.run_performance(goal, seams=fakes.seams(tmp_path))
    assert result.stage_names == ("authority",)
    assert set(fakes.calls.values()) == {0}
    assert result.snapshot is None and result.audio_master is None


def test_partial_rights_name_the_missing_subject(tmp_path, fakes):
    goal = performance_goal(goal=goal_spec(manifest=rights(ANA)))
    result = perf.run_performance(goal, seams=fakes.seams(tmp_path))
    assert result.ok is False
    assert result.gap.stage == "authority"
    assert any(BEN in item for item in result.gap.evidence)
    assert not any(ANA in item for item in result.gap.evidence)


def test_authority_refusal_carries_the_k97_scorecard(tmp_path, fakes):
    goal = performance_goal(goal=GoalSpec(
        objective="two characters", raw_prompt=f"{ANA} and {BEN} talk"))
    result = perf.run_performance(goal, seams=fakes.seams(tmp_path))
    assert result.scorecard is not None
    assert result.scorecard.hard_pass is False
    assert result.scorecard.repair_code is RepairCode.SOURCE_AUTHORITY_MISSING
    assert result.receipts[-1].failure.value == "refused"


def test_authority_requirements_cover_both_identities():
    required = perf.authority_requirements(performance_goal())
    assert (AuthorityKind.LIKENESS, ANA) in required
    assert (AuthorityKind.LIKENESS, BEN) in required


def test_a_reference_voice_needs_a_voice_grant():
    goal = performance_goal(casting={
        "ana": VoiceProfile(voice_id="ana_voice", kind="reference",
                            reference_ref="voice_profile:ana",
                            authorized=True),
        "ben": VoiceProfile.synthetic("ben_voice")})
    required = perf.authority_requirements(goal)
    assert (AuthorityKind.VOICE, "voice_profile:ana") in required


# ---------------------------------------------------------------------------
# 4. Stage 15 — identity failure repairs ONLY the keyframe
# ---------------------------------------------------------------------------


def test_identity_failure_reruns_only_the_keyframe(tmp_path):
    # segment 1: three identity rejections, then the repair round passes.
    verdicts = ([{"verdict": "NO", "score": 20,
                  "repair_code": "identity_drift", "why": "not her face"}] * 3
                + [{"verdict": "YES", "score": 90, "why": "identity reads"}])
    fakes = Fakes(keyframe_verdicts=verdicts)
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    assert result.ok is True
    assert result.shots[0].keyframe_repaired is True
    assert result.shots[0].keyframe_candidates == 4      # 3 + 1 repair take
    # The targeted-repair rule, counted: audio was NOT re-synthesized and no
    # extra clip was rendered for the repaired shot.
    assert fakes.calls["synth"] == 9
    assert fakes.calls["gen_clip"] == 3
    assert result.audio_master.candidates_considered == 9


def test_identity_repair_uses_different_deterministic_seeds(tmp_path):
    verdicts = ([{"verdict": "NO", "repair_code": "identity_drift"}] * 3
                + [{"verdict": "YES", "score": 90}])
    fakes = Fakes(keyframe_verdicts=verdicts)
    perf.run_performance(performance_goal(), seams=fakes.seams(tmp_path))
    first_round = fakes.seen_keyframe_seeds[:3]
    repair_take = fakes.seen_keyframe_seeds[3]
    assert len(set(first_round)) == 3
    assert repair_take not in first_round


def test_permanent_identity_failure_gaps_before_any_video(tmp_path):
    fakes = Fakes(keyframe_verdicts=[
        {"verdict": "NO", "score": 10, "repair_code": "identity_drift"}])
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    assert result.ok is False
    assert result.gap.stage == "keyframes"
    assert RepairCode.IDENTITY_DRIFT in result.gap.repair_codes
    assert fakes.calls["gen_clip"] == 0
    assert fakes.calls["concat"] == 0


def test_permanent_identity_failure_names_every_failing_segment(tmp_path):
    fakes = Fakes(keyframe_verdicts=[
        {"verdict": "NO", "score": 10, "repair_code": "identity_drift"}])
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    assert result.gap.segment_ids == ("s1", "s2", "s3")
    # three segments, each 3 candidates + one repair round of 3
    assert fakes.calls["gen_image"] == 3 * 6


def test_a_non_identity_keyframe_rejection_buys_no_repair_round(tmp_path):
    fakes = Fakes(keyframe_verdicts=[{"verdict": "NO", "score": 30,
                                      "why": "wrong setting"}])
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    assert result.ok is False
    assert result.gap.stage == "keyframes"
    assert RepairCode.INTENT_MISMATCH in result.gap.repair_codes
    assert fakes.calls["gen_image"] == 3 * 3      # no fourth round anywhere


def test_keyframe_candidate_count_is_a_seam_knob(tmp_path, fakes):
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path, keyframe_candidates=1))
    assert result.ok is True
    assert fakes.calls["gen_image"] == 3


def test_an_unbound_keyframe_seam_gaps_before_any_video(tmp_path, fakes):
    result = perf.run_performance(
        performance_goal(),
        seams=fakes.seams(tmp_path, gen_image=None,
                          unbound=(perf.SeamGap(
                              seam="gen_image", capability="image.generate",
                              requirement="seat a diffusers image model"),)))
    assert result.gap.stage == "keyframes"
    assert result.gap.capability == "image.generate"
    assert fakes.calls["gen_clip"] == 0


def test_an_unscored_keyframe_judge_keeps_the_take_but_lowers_confidence(
        tmp_path):
    fakes = Fakes()
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path, judge_image=None))
    assert result.ok is True
    card = result.shots[0].keyframe_scorecard
    assert card.hard_pass is True
    assert card.confidence < 1.0
    assert any("NO independent judge" in item for item in result.limitations)


# ---------------------------------------------------------------------------
# 5. Stage 17 — ONE bounded clip repair, then honest failure
# ---------------------------------------------------------------------------


def test_clip_judge_failure_gets_one_bounded_repair_then_fails(tmp_path):
    fakes = Fakes(clip_verdicts=[{"verdict": "NO", "score": 12,
                                  "why": "the action never happens"}])
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    assert result.ok is False
    assert result.gap.stage == "clips"
    assert RepairCode.INTENT_MISMATCH in result.gap.repair_codes
    # exactly TWO rounds of three candidates per shot — never a loop
    assert fakes.calls["gen_clip"] == 3 * (3 + 3)
    assert fakes.calls["concat"] == 0


def test_the_repaired_clip_card_says_a_repair_happened(tmp_path):
    fakes = Fakes(clip_verdicts=[{"verdict": "NO", "score": 12}])
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    shot = result.shots[0]
    assert shot.clip_repaired is True
    assert shot.accepted is False
    assert "after one bounded repair" in shot.scorecard.diagnosis


def test_a_repair_round_that_succeeds_is_accepted(tmp_path):
    verdicts = ([{"verdict": "NO", "score": 12}] * 3
                + [{"verdict": "YES", "score": 85}])
    fakes = Fakes(clip_verdicts=verdicts)
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    assert result.ok is True
    assert result.shots[0].clip_repaired is True
    assert result.shots[0].clip_candidates == 4


def test_clip_candidates_use_distinct_deterministic_seeds(tmp_path):
    fakes = Fakes(clip_verdicts=[{"verdict": "NO", "score": 12}])
    perf.run_performance(performance_goal(), seams=fakes.seams(tmp_path))
    first_shot_seeds = fakes.seen_clip_seeds[:6]
    assert len(set(first_shot_seeds)) == 6


def test_a_short_shot_is_not_reseeded_but_diagnosed(tmp_path):
    # The clip is far shorter than its locked audio window: Stage 8 says the
    # fix is a longer WINDOW (a lock revision), not another roll of the dice.
    fakes = Fakes(clip_seconds=0.2)
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path))
    assert result.ok is False
    assert result.gap.stage == "clips"
    assert RepairCode.SHOT_TOO_SHORT in result.gap.repair_codes
    assert fakes.calls["gen_clip"] == 3 * 3       # no repair round was bought


def test_repair_decision_policy_is_bounded_and_named():
    from hugpy_oracle.contracts import Check, CheckKind, Scorecard
    failing = Scorecard(
        hard_pass=False,
        checks=(Check(name="clip.judge", kind=CheckKind.INTENT, value=10,
                      threshold=60, passed=False, detail="rejected"),),
        repair_code=RepairCode.INTENT_MISMATCH)
    decision = perf.repair_decision(goal_spec(), perf.CLIP_CAPABILITY, failing)
    assert decision.action == "reseed"

    short = Scorecard(
        hard_pass=False,
        checks=(Check(name="sync.duration_fit", kind=CheckKind.SYNC, value=3.0,
                      threshold=1.0, passed=False, detail="overrun"),),
        repair_code=RepairCode.SHOT_TOO_SHORT)
    decision = perf.repair_decision(goal_spec(), perf.CLIP_CAPABILITY, short)
    assert decision.action == "none"
    assert "ProductionLock.revise" in decision.rationale


def test_an_unbound_clip_seam_gaps_after_the_keyframes(tmp_path, fakes):
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path, gen_clip=None))
    assert result.gap.stage == "clips"
    assert result.gap.capability == "video.generate.i2v"
    assert fakes.calls["gen_image"] == 3           # keyframes DID run
    assert result.stage_names == perf.STAGES[:7]


def test_default_seams_clip_gap_names_the_studio_spine():
    requirement = perf.default_seams().gap_for("gen_clip").requirement
    assert "studio" in requirement
    assert "GPU worker" in requirement


# ---------------------------------------------------------------------------
# 6. Resume
# ---------------------------------------------------------------------------


def test_stop_after_lock_persists_four_stages(tmp_path, fakes):
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path),
                                  stop_after="lock")
    assert result.ok is False
    assert result.gap is None
    assert result.stopped_after == "lock"
    assert result.stage_names == ("authority", "snapshot", "audio", "lock")
    assert fakes.calls["gen_image"] == 0
    with open(result.state_path, encoding="utf-8") as handle:
        assert set(json.load(handle)["stages"]) == {"snapshot", "audio", "lock"}


def test_resume_skips_the_completed_stages(tmp_path):
    first = Fakes()
    stopped = perf.run_performance(performance_goal(),
                                   seams=first.seams(tmp_path),
                                   stop_after="lock")

    second = Fakes()
    resumed = perf.run_performance(performance_goal(),
                                   seams=second.seams(tmp_path),
                                   resume=stopped.run_id)
    assert resumed.ok is True
    # stages 1-4 spent NOTHING: no take was re-synthesized, and the only
    # transcription is the final round trip in stage 8.
    assert second.calls["synth"] == 0
    assert second.calls["transcribe"] == 1
    assert resumed.stage("snapshot").resumed is True
    assert resumed.stage("audio").resumed is True
    assert resumed.stage("lock").resumed is True
    assert resumed.lock.digest == stopped.lock.digest


def test_resume_re_checks_authority_rather_than_trusting_the_journal(tmp_path):
    first = Fakes()
    stopped = perf.run_performance(performance_goal(),
                                   seams=first.seams(tmp_path),
                                   stop_after="lock")
    revoked = performance_goal(
        goal=goal_spec(manifest=RightsManifest(denied=(ANA,))))
    second = Fakes()
    result = perf.run_performance(revoked, seams=second.seams(tmp_path),
                                  resume=stopped.run_id)
    assert result.ok is False
    assert result.gap.stage == "authority"
    assert result.stage("authority").resumed is False
    assert second.calls["synth"] == 0


def test_resume_of_a_complete_run_regenerates_nothing(tmp_path):
    first = Fakes()
    done = perf.run_performance(performance_goal(), seams=first.seams(tmp_path))
    second = Fakes()
    again = perf.run_performance(performance_goal(),
                                 seams=second.seams(tmp_path),
                                 resume=done.run_id)
    assert again.ok is True
    assert second.calls["gen_image"] == 0
    assert second.calls["gen_clip"] == 0
    assert second.calls["concat"] == 0
    assert second.calls["transcribe"] == 1        # only the final round trip
    assert again.video_ref == done.video_ref
    assert all(again.stage(name).resumed for name in perf.RESUMABLE_STAGES)


def test_a_resumed_run_keeps_its_per_shot_evidence(tmp_path):
    first = Fakes()
    done = perf.run_performance(performance_goal(), seams=first.seams(tmp_path))
    again = perf.run_performance(performance_goal(),
                                 seams=Fakes().seams(tmp_path),
                                 resume=done.run_id)
    assert [s.segment_id for s in again.shots] == ["s1", "s2", "s3"]
    assert all(s.scorecard is not None and s.accepted for s in again.shots)


def test_a_tampered_journal_is_re_executed_not_trusted(tmp_path):
    first = Fakes()
    stopped = perf.run_performance(performance_goal(),
                                   seams=first.seams(tmp_path),
                                   stop_after="lock")
    with open(stopped.state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["stages"]["audio"]["payload"]["total_seconds"] = 99.0
    with open(stopped.state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)

    second = Fakes()
    result = perf.run_performance(performance_goal(),
                                  seams=second.seams(tmp_path),
                                  resume=stopped.run_id)
    assert result.ok is True
    assert second.calls["synth"] == 9              # audio really was re-run
    assert result.stage("audio").resumed is False
    assert result.stage("lock").resumed is False   # and everything after it


def test_a_missing_artifact_file_forces_a_rerun(tmp_path):
    """A journal whose refs point at absolute paths that are gone must not be
    resumed: the digest still matches, but the bytes are not there."""
    real = tmp_path / "take.wav"
    real.write_bytes(b"RIFF")

    class FileBacked(Fakes):
        def synth(self, line, voice, seed):
            self.calls["synth"] += 1
            return (str(real), 2.5)

        def transcribe(self, ref):
            self.calls["transcribe"] += 1
            return [w for line in LINES for w in line.text.split()]

    first = FileBacked()
    stopped = perf.run_performance(performance_goal(),
                                   seams=first.seams(tmp_path),
                                   stop_after="audio")
    os.remove(real)
    second = FileBacked()
    result = perf.run_performance(performance_goal(),
                                  seams=second.seams(tmp_path),
                                  resume=stopped.run_id, stop_after="audio")
    assert second.calls["synth"] == 9
    assert result.stage("audio").resumed is False


def test_resuming_a_different_goal_resumes_nothing(tmp_path):
    first = Fakes()
    stopped = perf.run_performance(performance_goal(),
                                   seams=first.seams(tmp_path),
                                   stop_after="lock")
    other = performance_goal(deliverable="a different deliverable entirely")
    second = Fakes()
    result = perf.run_performance(other, seams=second.seams(tmp_path),
                                  resume=stopped.run_id)
    assert second.calls["synth"] == 9
    assert not any(record.resumed for record in result.stages)
    assert any("nothing is resumed" in w for w in result.warnings)


def test_resuming_an_unknown_run_id_says_so_and_runs_everything(tmp_path,
                                                                fakes):
    result = perf.run_performance(performance_goal(),
                                  seams=fakes.seams(tmp_path),
                                  resume="perf-does-not-exist")
    assert result.ok is True
    assert fakes.calls["synth"] == 9
    assert any("no readable run state" in w for w in result.warnings)


def test_run_dir_mirrors_the_movie_work_dir_convention(monkeypatch, tmp_path):
    monkeypatch.setenv(perf.RUN_ROOT_ENV, str(tmp_path))
    assert perf.default_run_root() == str(tmp_path)
    assert perf.run_dir("perf-1") == os.path.join(
        str(tmp_path), "runs", "performance", "perf-1")
    assert perf.state_path("perf-1").endswith("state.json")


def test_derive_run_id_is_a_function_of_the_request():
    assert perf.derive_run_id(performance_goal()) == \
        perf.derive_run_id(performance_goal())
    assert perf.derive_run_id(performance_goal()) != \
        perf.derive_run_id(performance_goal(deliverable="something else"))


# ---------------------------------------------------------------------------
# 7. Static validation — never "probably fine"
# ---------------------------------------------------------------------------


def test_an_empty_catalog_aborts_at_stage_five_with_the_report(tmp_path,
                                                              fakes):
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path,
                                              catalog_view=lambda: {}))
    assert result.ok is False
    assert result.gap.stage == "segments"
    assert result.validation is not None and result.validation.ok is False
    assert any("unknown_capability" in item for item in result.gap.evidence)
    assert fakes.calls["gen_image"] == 0


def test_an_ineligible_capability_aborts_with_capability_gap(tmp_path, fakes):
    result = perf.run_performance(
        performance_goal(),
        seams=fakes.seams(tmp_path,
                          catalog_view=lambda: catalog_view(eligible=False)))
    assert result.gap.stage == "segments"
    assert RepairCode.CAPABILITY_GAP in result.gap.repair_codes
    assert any("no worker seats i2v" in item for item in result.gap.evidence)


def test_a_validation_abort_names_the_offending_nodes(tmp_path, fakes):
    result = perf.run_performance(
        performance_goal(), seams=fakes.seams(tmp_path,
                                              catalog_view=lambda: {}))
    assert set(result.gap.segment_ids) == {"segment:s1", "segment:s2",
                                           "segment:s3"}


def test_the_emitted_plan_graph_is_the_sibling_shape(happy):
    result, _fakes = happy
    graph = result.plan_graph
    assert graph is not None
    assert set(graph.segment_node_ids()) == {"segment:s1", "segment:s2",
                                             "segment:s3"}
    assert {edge.src_node for edge in graph.edges} == {"production_lock"}


def test_the_run_prompt_ledger_is_supplied_and_used(happy):
    """k104 built Stage 4's mechanism and could not supply the ledger; k106
    supplies it. A snapshot claiming a run-minted prompt as pre-run is caught."""
    result, _fakes = happy
    minted = result.segments[0].prompt
    poisoned = performance_goal(prompts_before_run=(minted,))
    outcome = perf.run_performance(
        poisoned, seams=Fakes().seams(os.path.dirname(result.state_path)))
    assert outcome.ok is False
    assert outcome.gap.stage == "segments"
    assert "invariant 9" in outcome.gap.diagnosis


# ---------------------------------------------------------------------------
# 8. Contract units
# ---------------------------------------------------------------------------


def test_coerce_verdict_treats_silence_as_unscored_not_approval():
    verdict = perf.coerce_verdict(None, threshold=60,
                                  default_code=RepairCode.INTENT_MISMATCH)
    assert verdict.passed is True
    assert verdict.scored is False
    assert verdict.codes == ()


def test_coerce_verdict_reads_a_no_and_a_low_score():
    no = perf.coerce_verdict({"verdict": "NO"}, threshold=60,
                             default_code=RepairCode.INTENT_MISMATCH)
    assert no.passed is False and no.codes == (RepairCode.INTENT_MISMATCH,)
    low = perf.coerce_verdict({"score": 12}, threshold=60,
                              default_code=RepairCode.INTENT_MISMATCH)
    assert low.passed is False and low.scored is True


def test_coerce_verdict_routes_an_identity_flag_to_identity_drift():
    verdict = perf.coerce_verdict({"identity": False, "verdict": "NO"},
                                  threshold=60,
                                  default_code=RepairCode.INTENT_MISMATCH)
    assert verdict.codes[0] is RepairCode.IDENTITY_DRIFT


def test_seams_refuse_a_zero_candidate_fan_out():
    with pytest.raises(perf.PerformanceError):
        perf.PerformanceSeams(tts_candidates=0)


def test_seams_refuse_a_gap_for_a_bound_seam():
    with pytest.raises(perf.PerformanceError):
        perf.PerformanceSeams(
            synth=lambda line, voice, seed: ("a", 1.0),
            unbound=(perf.SeamGap(seam="synth", capability="audio.tts",
                                  requirement="seat it"),))


def test_a_seam_gap_must_carry_an_operator_step():
    with pytest.raises(perf.PerformanceError):
        perf.SeamGap(seam="synth", capability="audio.tts", requirement="  ")


def test_budget_refuses_more_than_one_repair_round():
    with pytest.raises(perf.PerformanceError):
        perf.PerformanceBudget(clip_repair_rounds=2)


def test_the_goal_locks_its_dialogue_and_refuses_an_uncast_speaker():
    goal = performance_goal(dialogue=DialogueTimeline(lines=LINES))
    assert goal.dialogue.locked is True
    with pytest.raises(perf.PerformanceError):
        performance_goal(casting={"ana": VoiceProfile.synthetic("ana_voice")})


def test_a_result_cannot_claim_ok_without_a_deliverable():
    with pytest.raises(perf.PerformanceError):
        perf.PerformanceResult(run_id="perf-x", ok=True, goal_digest="d")


def test_a_result_cannot_be_ok_and_carry_a_gap():
    with pytest.raises(perf.PerformanceError):
        perf.PerformanceResult(
            run_id="perf-x", ok=True, goal_digest="d", video_ref="v",
            gap=perf.PerformanceGap(stage="clips", diagnosis="broken"))


def test_a_gap_must_explain_itself():
    with pytest.raises(perf.PerformanceError):
        perf.PerformanceGap(stage="clips", diagnosis="   ")


def test_a_supplied_continuity_bible_is_used_over_the_shell(tmp_path, fakes):
    from hugpy_oracle.production import ContinuityBible
    bible = ContinuityBible(
        entries=tuple(ContinuityState(segment_id=sid,
                                      state_before={"coat": "on"},
                                      state_after={"coat": "off"})
                      for sid in ("s1", "s2", "s3")),
        characters=("ana", "ben"), wardrobe=("coat",))
    result = perf.run_performance(performance_goal(continuity=bible),
                                  seams=fakes.seams(tmp_path))
    assert result.ok is True
    assert result.segments[0].continuity.changed_keys == ("coat",)
    assert not any("EMPTY SHELL" in item for item in result.limitations)


def test_a_wall_clock_budget_ends_the_run_with_a_typed_timeout(tmp_path):
    slow = Fakes(synth_delay=0.02)
    result = perf.run_performance(
        performance_goal(), seams=slow.seams(tmp_path),
        budget=perf.PerformanceBudget(max_seconds=0.01))
    assert result.ok is False
    assert result.gap.repair_codes == (RepairCode.TIMEOUT,)
    assert result.receipts[-1].failure.value == "timeout"


def test_quality_profile_moves_the_judge_bar(tmp_path):
    fakes = Fakes(clip_verdicts=[{"score": 65, "why": "adequate"}])
    ok = perf.run_performance(
        performance_goal(goal=goal_spec(quality=QualityProfile.BALANCED)),
        seams=fakes.seams(tmp_path / "balanced"))
    assert ok.ok is True

    strict = Fakes(clip_verdicts=[{"score": 65, "why": "adequate"}])
    result = perf.run_performance(
        performance_goal(goal=goal_spec(quality=QualityProfile.BEST)),
        seams=strict.seams(tmp_path / "best"))
    assert result.ok is False
    assert result.gap.stage == "clips"


def test_stop_after_rejects_an_unknown_stage(tmp_path, fakes):
    with pytest.raises(perf.PerformanceError):
        perf.run_performance(performance_goal(), seams=fakes.seams(tmp_path),
                             stop_after="render")


def test_run_performance_refuses_a_bare_goalspec(tmp_path, fakes):
    with pytest.raises(perf.PerformanceError):
        perf.run_performance(goal_spec(), seams=fakes.seams(tmp_path))




# ---------------------------------------------------------------------------
# 9. The bus relay — a socket, not a second implementation
# ---------------------------------------------------------------------------

from hugpy_oracle.relay import performance_relay as relay


def relay_spec(**kw):
    goal = performance_goal()
    base = dict(
        goal=goal.goal.to_dict(),
        dialogue=goal.dialogue.to_dict(),
        casting=[[speaker, profile.to_dict()]
                 for speaker, profile in goal.casting],
        raw_request_ref=goal.raw_request_ref,
        identity_refs=list(goal.identity_refs),
        deliverable=goal.deliverable,
        pad_s=0.35,
        created_at=goal.created_at,
    )
    base.update(kw)
    return relay.make_performance(
        base.pop("goal"), base.pop("dialogue"), base.pop("casting"),
        base.pop("raw_request_ref"), **base)


def test_the_relay_is_registered_in_the_bus_dispatch_table():
    """Video never imports the oracle: the relay reaches the bus's dispatch
    table through ``hugpy_oracle.install_hooks`` -> ``hugpy_video.jobs``."""
    import hugpy_oracle
    from hugpy_video import jobs as vjobs
    from hugpy_video.intel.runners import DISPATCH
    assert relay.RUNNER_KEY == ("oracle", "performance")
    try:
        assert hugpy_oracle.install_hooks()["performance_runner"] == relay.JOB_NAME
        assert DISPATCH[relay.RUNNER_KEY] is relay.run_video_performance
        assert vjobs.registered_jobs()[relay.JOB_NAME].spec_type is relay.PerformanceSpec
    finally:
        vjobs.unregister_job(relay.JOB_NAME)


def test_the_relay_module_top_is_stdlib_only():
    """``runners/__init__`` imports this at app boot; importing the oracle
    there would build the model registry on every boot."""
    import ast
    source = open(relay.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            top.append((node.module or "").split(".")[0])
    assert set(top) <= {"__future__", "logging", "os", "dataclasses", "typing"}


def test_the_relay_probe_is_honest_about_the_remaining_gaps():
    """Still not ready — but for the VIDEO seams now, not the audio one. The
    probe names what is actually missing, which after the chatterbox seating is
    gen_clip (and judge_clip), never synth."""
    probe = relay.probe()
    assert probe["runner_registered"] is True
    assert probe["importable"] is True
    assert probe["ready"] is False
    assert "gen_clip" in probe["reason"]
    assert "synth" not in probe["reason"]
    assert all(g["seam"] != "synth" for g in probe["unbound"])
    step = next(g for g in probe["unbound"] if g["seam"] == "gen_clip")
    assert "studio" in step["requirement"]
    assert list(probe["stages"]) == list(perf.STAGES)


def test_the_relay_spec_refuses_a_request_it_cannot_run():
    with pytest.raises(relay.PerformanceSpecError):
        relay.make_performance({}, {"lines": [{}]}, [["ana", {}]], "ref")
    with pytest.raises(relay.PerformanceSpecError):
        relay.make_performance({"raw_prompt": "x"}, {"lines": []},
                               [["ana", {}]], "ref")
    with pytest.raises(relay.PerformanceSpecError):
        relay.make_performance({"raw_prompt": "x"}, {"lines": [{}]}, [], "ref")
    with pytest.raises(relay.PerformanceSpecError):
        relay.make_performance({"raw_prompt": "x"}, {"lines": [{}]},
                               [["ana", {}]], "  ")


def test_the_relay_spec_round_trips_through_json():
    import dataclasses
    spec = relay_spec()
    payload = json.loads(json.dumps(dataclasses.asdict(spec)))
    assert relay.performance_from_dict(payload) == spec


def test_the_relay_spec_refuses_an_unknown_key():
    import dataclasses
    payload = dataclasses.asdict(relay_spec())
    payload["render_forever"] = True
    with pytest.raises(relay.PerformanceSpecError):
        relay.performance_from_dict(payload)


def test_the_relay_returns_a_typed_job_error_when_the_seat_is_missing(
        monkeypatch, tmp_path):
    """An UNBOUND synth seam becomes a typed capability_gap on the bus.

    The seam set is injected rather than taken from ``default_seams()``: since
    the chatterbox seating (2026-08-21) that function binds ``synth`` LIVE, so
    the un-injected version of this test stopped testing the gap and started
    dispatching real synthesis to a GPU worker — 19s of fleet work inside a unit
    test, passing for a reason its name does not describe. A test must not need
    a worker to prove what a refusal looks like."""
    fakes = Fakes()
    monkeypatch.setattr(relay, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(perf, "default_seams",
                        lambda **kw: fakes.seams(tmp_path, synth=None, **{
                            k: v for k, v in kw.items()
                            if k.endswith("_candidates")}))
    monkeypatch.setenv(perf.RUN_ROOT_ENV, str(tmp_path))
    result = relay.run_video_performance(relay_spec(), "job-k106-1")
    assert result.ok is False
    assert result.error.code == "capability_gap"
    assert "audio.tts" in result.error.message
    assert "chatterbox" in result.error.message
    assert result.error.retryable is False
    assert result.movie["gap"]["stage"] == "audio"


@pytest.fixture()
def offline_process_selector(tmp_path, monkeypatch):
    """The relay binds ``selection.process_selector()`` — the LIVE catalog —
    to the DAG. Off the fleet that catalog has no model for any visual
    capability and every keyframe fan-out gaps, so the selector is bound to
    the same fake view the seams use (every capability seated by one fake
    model) and a ledger under tmp_path. Nothing here touches ~/.hugpy."""
    from hugpy_oracle import selection

    fake_views = catalog_view()

    def get_view(capability: str):
        view = fake_views.get(capability)
        if view is not None:
            return view
        return CapabilityView(
            name=capability, source=SourceRegistry.STUDIO,
            accepts=(ArtifactKind.JSON,), produces=(ArtifactKind.JSON,),
            model_ids=(f"fake-{capability}",),
            eligibility=Eligibility(eligible=True, reasons=()))

    ledger = selection.ReliabilityLedger(str(tmp_path / "relay-ledger.sqlite"))
    sel = selection.Selector(ledger=ledger, get_view=get_view, get_matrix=lambda: None)
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", sel)
    yield sel
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", None)
    ledger.close()


def test_the_relay_carries_the_whole_manifest_on_success(monkeypatch,
                                                         tmp_path,
                                                         offline_process_selector):
    fakes = Fakes()
    monkeypatch.setattr(relay, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(perf, "default_seams",
                        lambda **kw: fakes.seams(tmp_path, **{
                            k: v for k, v in kw.items()
                            if k.endswith("_candidates")}))
    result = relay.run_video_performance(relay_spec(), "job-k106-2")
    assert result.ok is True
    assert result.error is None
    assert result.movie["ok"] is True
    assert result.movie["video_ref"] == fakes.video
    assert result.movie["limitations"]
    assert result.project["uuid"] == "job-k106-2"
    assert result.outputs == ()      # opaque refs are never ingested or faked


def test_the_relay_reports_a_deliberate_checkpoint_as_a_checkpoint(
        monkeypatch, tmp_path):
    fakes = Fakes()
    monkeypatch.setattr(relay, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(perf, "default_seams",
                        lambda **kw: fakes.seams(tmp_path))
    result = relay.run_video_performance(relay_spec(stop_after="lock"),
                                         "job-k106-3")
    assert result.ok is False
    assert result.error.code == "stopped_after_stage"
    assert result.error.retryable is True
    assert result.movie["stopped_after"] == "lock"


def test_the_relay_turns_a_bad_spec_into_data_not_a_raise(monkeypatch,
                                                          tmp_path):
    monkeypatch.setattr(relay, "_progress", lambda *a, **k: None)
    broken = relay_spec(casting=[["nobody", VoiceProfile.synthetic(
        "nobody_voice").to_dict()]])
    result = relay.run_video_performance(broken, "job-k106-4")
    assert result.ok is False
    assert result.error.code == "bad_spec"
    assert "no voice cast" in result.error.message
