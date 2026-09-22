"""k102 — audio-first timing artifacts + the TTS candidate fan-out.

Everything here runs on a GPU-less box with no TTS and no ASR backend: the unit
under test (``oracle/audio_master.py``) is a PURE orchestrator over three
injected callables, so the whole Stage 8 rule set — dialogue lock, N candidates
per line, round-trip judging, best-take selection, sequential assembly, typed
gaps — is exercised against a ``FakeTts`` that never leaves the process.

The fake is deliberately dumb: it hands back a made-up audio ref plus a duration
and, on ``transcribe``, the words it decides the take "said" (which is NOT
necessarily the line — that is how LINE_OMITTED gets tested).

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_audio_master.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import audio_master as am
from hugpy_oracle.contracts import RepairCode


# ===========================================================================
# Fakes
# ===========================================================================


class FakeTts:
    """A TTS + ASR pair with no backend behind it.

    ``duration_fn(line, index)`` decides how long take ``index`` of a line is;
    ``spoken_fn(line, index)`` decides what the round-trip transcript says the
    take contained. Word times are spread evenly over the take, which is enough
    for the timing assertions and never pretends to be phoneme alignment.
    """

    def __init__(self, duration_fn=None, spoken_fn=None, sim_fn=None,
                 word_times=True):
        self.duration_fn = duration_fn or (lambda line, index: 1.0)
        self.spoken_fn = spoken_fn or (lambda line, index: line.text)
        self.sim_fn = sim_fn
        self.word_times = word_times
        self.synth_calls: list[tuple[str, str, int]] = []
        self.transcribe_calls: list[str] = []
        self._made: dict[str, tuple] = {}
        self._per_line: dict[str, int] = {}

    def synth(self, line, voice, seed):
        index = self._per_line.get(line.line_id, 0)
        self._per_line[line.line_id] = index + 1
        duration = float(self.duration_fn(line, index))
        ref = f"/audio/{line.line_id}-{index}-{seed:08x}.wav"
        self._made[ref] = (line, index, duration)
        self.synth_calls.append((line.line_id, voice.voice_id, seed))
        return ref, duration

    def transcribe(self, audio_ref):
        self.transcribe_calls.append(audio_ref)
        line, index, duration = self._made[audio_ref]
        words = str(self.spoken_fn(line, index)).split()
        if not words:
            return []
        step = duration / len(words)
        out = []
        for i, word in enumerate(words):
            item = {"word": word, "probability": 0.9}
            if self.word_times:
                item["start"] = round(i * step, 6)
                item["end"] = round((i + 1) * step, 6)
            out.append(item)
        return out

    def similarity(self, audio_ref, voice):
        if self.sim_fn is None:
            return None
        line, index, _ = self._made[audio_ref]
        return self.sim_fn(line, index, voice)


def two_line_timeline(**kw):
    return am.DialogueTimeline(lines=(
        am.Line("l1", "MIRA", "Hello there my friend", **kw),
        am.Line("l2", "JON", "We should go now"),
    )).lock()


CAST = {"MIRA": am.VoiceProfile.synthetic("mira_default"),
        "JON": am.VoiceProfile.style_fallback("jon_style", cadence="slow",
                                              energy="low")}
POLICY = am.SpeechPolicy(pause_after_s=0.5)


def build(timeline, fake, **kw):
    kw.setdefault("policy", POLICY)
    return am.build_audio_master(
        timeline, CAST, synth=fake.synth, transcribe=fake.transcribe,
        similarity=(fake.similarity if fake.sim_fn is not None else None), **kw)


# ===========================================================================
# Artifacts — Line / DialogueTimeline
# ===========================================================================


def test_line_roundtrip_and_digest_is_value_derived():
    line = am.Line("l1", "MIRA", "Hello there.", emotion="warm", max_seconds=2.0)
    assert am.Line.from_dict(line.to_dict()) == line
    assert am.Line.from_dict(json.loads(line.canonical_bytes)) == line
    # the digest is a function of the VALUES, not of construction
    assert line.digest == am.Line("l1", "MIRA", "Hello there.", emotion="warm",
                                  max_seconds=2.0).digest
    assert line.digest != am.Line("l1", "MIRA", "Hello there!",
                                  emotion="warm", max_seconds=2.0).digest
    assert len(line.digest) == 64


@pytest.mark.parametrize("kwargs", [
    {"line_id": "", "speaker": "M", "text": "hi"},
    {"line_id": "l", "speaker": "  ", "text": "hi"},
    {"line_id": "l", "speaker": "M", "text": "   "},
])
def test_line_refuses_structurally_empty_fields(kwargs):
    with pytest.raises(ValueError):
        am.Line(**kwargs)


def test_line_max_seconds_must_be_positive_when_set():
    with pytest.raises(ValueError, match="max_seconds"):
        am.Line("l", "M", "hi", max_seconds=0)
    assert am.Line("l", "M", "hi", max_seconds=None).max_seconds is None


def test_timeline_roundtrip_lock_and_lookup():
    timeline = am.DialogueTimeline(lines=[am.Line("l1", "MIRA", "one"),
                                          am.Line("l2", "JON", "two")])
    assert not timeline.locked
    assert timeline.line_ids == ("l1", "l2")
    assert timeline.texts == ("one", "two")
    assert timeline.speakers == ("MIRA", "JON")
    assert timeline.line("l2").text == "two"
    with pytest.raises(KeyError):
        timeline.line("nope")

    locked = timeline.lock()
    assert locked.locked and locked.lock() is locked
    # locking CHANGES the digest: a locked timeline is a different artifact
    assert locked.digest != timeline.digest
    assert am.DialogueTimeline.from_dict(locked.to_dict()) == locked


def test_timeline_refuses_duplicate_ids_and_emptiness():
    with pytest.raises(ValueError, match="duplicate line_id"):
        am.DialogueTimeline(lines=(am.Line("l1", "M", "a"),
                                   am.Line("l1", "M", "b")))
    with pytest.raises(ValueError, match="at least one line"):
        am.DialogueTimeline(lines=())


# ===========================================================================
# Artifacts — VoiceProfile (the authority gate)
# ===========================================================================


def test_reference_voice_without_authorization_cannot_be_constructed():
    with pytest.raises(ValueError, match="requires authorized=True"):
        am.VoiceProfile("mira", kind=am.VoiceKind.REFERENCE,
                        reference_ref="/voices/mira.wav")
    ok = am.VoiceProfile("mira", kind="reference",
                         reference_ref="/voices/mira.wav", authorized=True)
    assert ok.kind is am.VoiceKind.REFERENCE and ok.authorized


def test_reference_voice_needs_a_reference():
    with pytest.raises(ValueError, match="needs a reference_ref"):
        am.VoiceProfile("mira", kind="reference", authorized=True)


def test_non_reference_kind_may_not_smuggle_a_reference():
    for kind in ("synthetic", "style"):
        with pytest.raises(ValueError, match="must not carry reference_ref"):
            am.VoiceProfile("x", kind=kind, reference_ref="/voices/mira.wav",
                            authorized=True, style={"cadence": "slow"})


def test_style_fallback_must_describe_a_delivery():
    with pytest.raises(ValueError, match="must describe a delivery"):
        am.VoiceProfile("x", kind="style", style={"celebrity": "somebody"})
    voice = am.VoiceProfile.style_fallback("x", cadence="slow", energy="low")
    assert voice.style_dict() == {"cadence": "slow", "energy": "low"}


def test_voice_style_is_order_independent_and_json_scalar_only():
    a = am.VoiceProfile.synthetic("v", energy="low", cadence="slow")
    b = am.VoiceProfile.synthetic("v", cadence="slow", energy="low")
    assert a.style == b.style and a.digest == b.digest
    assert am.VoiceProfile.from_dict(a.to_dict()) == a
    with pytest.raises(ValueError, match="JSON scalar"):
        am.VoiceProfile.synthetic("v", cadence=["slow", "fast"])


# ===========================================================================
# Artifacts — timings, candidates, master
# ===========================================================================


def test_word_timing_roundtrip_shift_and_ordering_rule():
    word = am.WordTiming("hello", 0.25, 0.75, prob=0.9)
    assert am.WordTiming.from_dict(word.to_dict()) == word
    assert word.duration_s == pytest.approx(0.5)
    assert word.shifted(1.0) == am.WordTiming("hello", 1.25, 1.75, prob=0.9)
    with pytest.raises(ValueError, match="ends before it starts"):
        am.WordTiming("hello", 1.0, 0.5)
    with pytest.raises(ValueError, match="prob"):
        am.WordTiming("hello", 0.0, 1.0, prob=1.5)


def test_line_timing_roundtrip_and_next_start():
    timing = am.LineTiming("l1", 0.0, 1.5,
                           words=(am.WordTiming("hi", 0.0, 0.4),),
                           pause_after_s=0.5)
    assert am.LineTiming.from_dict(timing.to_dict()) == timing
    assert timing.duration_s == pytest.approx(1.5)
    assert timing.next_start_s == pytest.approx(2.0)
    with pytest.raises(ValueError, match="non-negative"):
        am.LineTiming("l1", 0.0, 1.0, pause_after_s=-0.1)
    with pytest.raises(TypeError, match="WordTiming"):
        am.LineTiming("l1", 0.0, 1.0, words=({"word": "hi"},))


def test_speech_candidate_roundtrip_and_seed_rules():
    cand = am.SpeechCandidate("c1", "l1", "v1", "/a.wav", 1.25, 42,
                              scorecard_digest="ab" * 32, accepted=True)
    assert am.SpeechCandidate.from_dict(cand.to_dict()) == cand
    with pytest.raises(ValueError, match="seed must be non-negative"):
        am.SpeechCandidate("c1", "l1", "v1", "/a.wav", 1.0, -1)
    with pytest.raises(ValueError, match="audio_ref"):
        am.SpeechCandidate("c1", "l1", "v1", "", 1.0, 1)


def test_audio_master_roundtrip_windows_and_consistency():
    timings = (am.LineTiming("l1", 0.0, 1.0, pause_after_s=0.5),
               am.LineTiming("l2", 1.5, 2.5, pause_after_s=0.5))
    master = am.AudioMaster(timeline_digest="d" * 64, line_timings=timings,
                            tracks=(("l1", "/a1.wav"), ("l2", "/a2.wav")),
                            total_seconds=3.0, candidates_considered=6,
                            locked=True)
    assert am.AudioMaster.from_dict(master.to_dict()) == master
    assert master.line_ids == ("l1", "l2")
    assert master.audio_ref("l2") == "/a2.wav"
    assert master.windows() == (("l1", 0.0, 1.0), ("l2", 1.5, 2.5))
    assert master.windows(include_pause=True)[0] == ("l1", 0.0, 1.5)
    assert master.timing("l1").pause_after_s == pytest.approx(0.5)
    with pytest.raises(KeyError):
        master.audio_ref("l9")


def test_audio_master_refuses_tracks_that_disagree_with_timings():
    timings = (am.LineTiming("l1", 0.0, 1.0),)
    with pytest.raises(ValueError, match="timings and tracks disagree"):
        am.AudioMaster(timeline_digest="d" * 64, line_timings=timings,
                       tracks=(("l2", "/a.wav"),), total_seconds=1.0)
    with pytest.raises(ValueError, match="shorter than its own last line end"):
        am.AudioMaster(timeline_digest="d" * 64, line_timings=timings,
                       tracks=(("l1", "/a.wav"),), total_seconds=0.5)


# ===========================================================================
# Policy + seeds + transcript coercion
# ===========================================================================


def test_policy_pause_default_and_per_line_override():
    policy = am.SpeechPolicy(pause_after_s=0.4,
                             pause_overrides={"l2": 1.25})
    assert policy.pause_for("l1") == pytest.approx(0.4)
    assert policy.pause_for("l2") == pytest.approx(1.25)
    assert policy.to_dict()["pause_overrides"] == {"l2": 1.25}
    with pytest.raises(ValueError, match="non-negative"):
        am.SpeechPolicy(pause_after_s=-1)
    with pytest.raises(ValueError, match="similarity_threshold"):
        am.SpeechPolicy(similarity_threshold=1.5)


def test_candidate_seeds_are_deterministic_and_input_derived():
    line = am.Line("l1", "MIRA", "Hello there")
    other = am.Line("l1", "MIRA", "Hello there!")
    voice = am.VoiceProfile.synthetic("v")
    voice2 = am.VoiceProfile.synthetic("w")

    seeds = [am.candidate_seed(line, voice, i) for i in range(3)]
    assert seeds == [am.candidate_seed(line, voice, i) for i in range(3)]
    assert len(set(seeds)) == 3                      # takes differ
    assert all(0 <= s < am.SEED_MODULUS for s in seeds)
    assert am.candidate_seed(other, voice, 0) != seeds[0]      # text re-rolls
    assert am.candidate_seed(line, voice2, 0) != seeds[0]      # voice re-rolls
    assert am.candidate_seed(line, voice, 0, "repair:1") != seeds[0]  # salt
    with pytest.raises(ValueError):
        am.candidate_seed(line, voice, -1)


def test_coerce_word_timings_accepts_every_transcript_shape():
    class Pydanticish:
        word, start, end, probability = "there", 0.5, 0.9, 0.8

    words, untimed = am.coerce_word_timings([
        {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.99},
        Pydanticish(),
        am.WordTiming("friend", 1.0, 1.4),
        {"word": "  "},                       # blank: not a word at all
    ])
    assert [w.word for w in words] == ["Hello", "there", "friend"]
    assert words[0].prob == pytest.approx(0.99)
    assert untimed == 0


def test_coerce_word_timings_counts_untimed_words_instead_of_inventing_times():
    words, untimed = am.coerce_word_timings([{"word": "hello"},
                                             {"word": "there"}])
    assert words == () and untimed == 2


# ===========================================================================
# The fan-out — happy path
# ===========================================================================


def test_fan_out_assembles_sequential_timings_with_pauses():
    fake = FakeTts(duration_fn=lambda line, index: 1.0 if line.line_id == "l1"
                   else 2.0)
    result = build(two_line_timeline(), fake)

    assert result.ok and result.gaps == ()
    master = result.master
    assert master.locked and master.timeline_digest == two_line_timeline().digest
    # l1: 0.0 -> 1.0, pause 0.5; l2 starts at 1.5 and runs 2.0s; total 4.0
    assert master.windows() == (("l1", 0.0, 1.0), ("l2", 1.5, 3.5))
    assert master.total_seconds == pytest.approx(4.0)
    assert all(t.pause_after_s == pytest.approx(0.5) for t in master.line_timings)
    assert [ref for _, ref in master.tracks] == [
        master.audio_ref("l1"), master.audio_ref("l2")]


def test_candidates_considered_counts_every_take_synthesized_and_judged():
    fake = FakeTts()
    result = build(two_line_timeline(), fake, candidates=3)
    assert len(fake.synth_calls) == 6
    assert len(fake.transcribe_calls) == 6          # every take is round-tripped
    assert result.candidates_considered == 6
    assert result.master.candidates_considered == 6
    assert len(result.accepted()) == 6
    assert len({c.candidate_id for c in result.candidates}) == 6
    assert all(c.scorecard_digest and len(c.scorecard_digest) == 64
               for c in result.candidates)


def test_word_timings_come_from_the_round_trip_not_the_synth_claim():
    # The take "says" something the SCRIPT does not contain; the timeline must
    # carry what the transcript heard, offset onto the master timeline.
    fake = FakeTts(duration_fn=lambda line, index: 2.0,
                   spoken_fn=lambda line, index: line.text + " indeed")
    result = build(two_line_timeline(), fake)
    assert result.ok
    l2 = result.master.timing("l2")
    assert [w.word for w in l2.words] == ["We", "should", "go", "now", "indeed"]
    assert l2.words[0].start_s == pytest.approx(l2.start_s)
    assert l2.words[-1].end_s == pytest.approx(l2.end_s)
    assert all(l2.start_s - 1e-9 <= w.start_s <= w.end_s <= l2.end_s + 1e-9
               for w in l2.words)


def test_missing_word_times_are_reported_not_fabricated():
    fake = FakeTts(word_times=False)
    result = build(two_line_timeline(), fake)
    assert result.ok
    assert all(t.words == () for t in result.master.line_timings)
    assert any("no timing" in w for w in result.warnings)


def test_lead_in_and_per_line_pause_override_move_the_timeline():
    fake = FakeTts(duration_fn=lambda line, index: 1.0)
    policy = am.SpeechPolicy(pause_after_s=0.25, lead_in_s=0.5,
                             pause_overrides={"l1": 2.0})
    result = build(two_line_timeline(), fake, policy=policy)
    assert result.master.windows() == (("l1", 0.5, 1.5), ("l2", 3.5, 4.5))
    assert result.master.total_seconds == pytest.approx(4.75)


def test_registry_version_rides_from_policy_into_provenance():
    fake = FakeTts()
    policy = am.SpeechPolicy(registry_version="registry-2026-08-20")
    assert build(two_line_timeline(), fake,
                 policy=policy).master.registry_version == "registry-2026-08-20"
    assert build(two_line_timeline(), FakeTts()).master.registry_version is None


# ===========================================================================
# The fan-out — selection
# ===========================================================================


def test_best_candidate_is_the_accepted_take_closest_to_the_budget():
    # takes: 1.0s, 1.9s, 3.5s against a 2.0s budget -> the 3.5s take fails the
    # duration check, and 1.9 beats 1.0 on closeness.
    durations = {0: 1.0, 1: 1.9, 2: 3.5}
    fake = FakeTts(duration_fn=lambda line, index: durations[index]
                   if line.line_id == "l1" else 1.0)
    timeline = am.DialogueTimeline(lines=(
        am.Line("l1", "MIRA", "Hello there my friend", max_seconds=2.0),
        am.Line("l2", "JON", "We should go now"))).lock()
    result = build(timeline, fake, candidates=3)

    assert result.ok
    assert result.master.timing("l1").duration_s == pytest.approx(1.9)
    l1_takes = [c for c in result.candidates if c.line_id == "l1"]
    assert [c.accepted for c in l1_takes] == [True, True, False]
    assert result.master.audio_ref("l1") == l1_takes[1].audio_ref


def test_a_rejected_take_never_outranks_an_accepted_one():
    # take 0 omits a word (rejected) but is dead on the budget; take 1 is a
    # clean read that is further from it. The clean read must win.
    fake = FakeTts(
        duration_fn=lambda line, index: 2.0 if index == 0 else 1.2,
        spoken_fn=lambda line, index: ("Hello there my" if index == 0
                                       else line.text))
    timeline = am.DialogueTimeline(lines=(
        am.Line("l1", "MIRA", "Hello there my friend", max_seconds=2.0),)).lock()
    result = build(timeline, fake, candidates=2)
    assert result.ok
    assert result.master.timing("l1").duration_s == pytest.approx(1.2)


def test_higher_measured_similarity_breaks_a_tie():
    fake = FakeTts(duration_fn=lambda line, index: 1.0,
                   sim_fn=lambda line, index, voice: 0.80 + 0.05 * index)
    timeline = am.DialogueTimeline(
        lines=(am.Line("l1", "MIRA", "Hello there my friend"),)).lock()
    result = build(timeline, fake, candidates=3)
    assert result.ok
    takes = [c for c in result.candidates if c.line_id == "l1"]
    assert result.master.audio_ref("l1") == takes[2].audio_ref


# ===========================================================================
# The fan-out — typed gaps (never silently accept)
# ===========================================================================


def test_line_omitted_when_the_round_trip_drops_a_word():
    fake = FakeTts(spoken_fn=lambda line, index: " ".join(
        line.text.split()[:-1]) if line.line_id == "l1" else line.text)
    result = build(two_line_timeline(), fake, candidates=2)

    assert not result.ok and result.master is None
    assert [g.line_id for g in result.gaps] == ["l1"]
    gap = result.gaps[0]
    assert gap.repair_codes == (RepairCode.LINE_OMITTED,)
    assert gap.primary_code is RepairCode.LINE_OMITTED
    assert gap.candidates_considered == 2 and len(gap.scorecards) == 2
    assert all(not s.hard_pass for s in gap.scorecards)
    assert "friend" in gap.scorecards[0].diagnosis
    # l2 was still judged: gaps are reported for the WHOLE timeline in one pass
    assert len(result.candidates) == 4
    assert [c.accepted for c in result.candidates if c.line_id == "l2"] == \
        [True, True]


def test_shot_too_short_when_every_take_overruns_the_budget():
    fake = FakeTts(duration_fn=lambda line, index: 5.0)
    timeline = am.DialogueTimeline(
        lines=(am.Line("l1", "MIRA", "Hello there my friend",
                       max_seconds=1.0),)).lock()
    result = build(timeline, fake, candidates=3)
    assert not result.ok
    assert result.repair_codes == (RepairCode.SHOT_TOO_SHORT,)
    assert result.gaps[0].candidate_ids == tuple(
        c.candidate_id for c in result.candidates)
    assert not any(c.accepted for c in result.candidates)


def test_voice_similarity_low_gaps_the_line():
    fake = FakeTts(sim_fn=lambda line, index, voice: 0.20)
    result = build(two_line_timeline(), fake, candidates=2)
    assert not result.ok
    assert result.repair_codes == (RepairCode.VOICE_SIMILARITY_LOW,)
    assert len(result.gaps) == 2          # both lines, in one pass


def test_multiple_codes_are_all_reported_in_repair_first_order():
    fake = FakeTts(
        duration_fn=lambda line, index: 9.0 if index == 0 else 0.5,
        spoken_fn=lambda line, index: (line.text if index == 0
                                       else "totally different words"))
    timeline = am.DialogueTimeline(
        lines=(am.Line("l1", "MIRA", "Hello there my friend",
                       max_seconds=1.0),)).lock()
    gap = build(timeline, fake, candidates=2).gaps[0]
    assert gap.repair_codes == (RepairCode.LINE_OMITTED,
                                RepairCode.SHOT_TOO_SHORT)
    assert gap.primary_code is RepairCode.LINE_OMITTED
    assert gap.to_dict()["primary_code"] == "line_omitted"


def test_unmeasured_similarity_is_unscored_and_still_selectable():
    fake = FakeTts()                       # sim_fn None -> no similarity seam
    result = build(two_line_timeline(), fake)
    assert result.ok
    card = am.judge_candidate(am.Line("l1", "MIRA", "Hello there my friend"),
                              1.0, [{"word": w, "start": 0.0, "end": 0.1}
                                    for w in "Hello there my friend".split()],
                              None, POLICY)
    assert card.hard_pass
    assert card.confidence < 1.0           # honest: nothing measured the voice
    assert any(am.speech.is_unscored(c) for c in card.checks)


def test_require_similarity_turns_no_evidence_into_a_capability_gap():
    fake = FakeTts()
    policy = am.SpeechPolicy(require_similarity=True)
    result = build(two_line_timeline(), fake, candidates=2, policy=policy)
    assert not result.ok
    assert result.repair_codes == (RepairCode.CAPABILITY_GAP,)
    assert "no speaker-embedding backend" in result.gaps[0].scorecards[0].diagnosis
    # ... and with a measured score the same policy accepts
    scored = FakeTts(sim_fn=lambda line, index, voice: 0.9)
    assert build(two_line_timeline(), scored, policy=policy).ok


def test_result_cannot_carry_a_master_and_a_gap_at_once():
    master = am.AudioMaster(timeline_digest="d" * 64, line_timings=(),
                            tracks=(), total_seconds=0.0)
    with pytest.raises(ValueError, match="master AND gaps"):
        am.AudioBuildResult(master=master, gaps=(am.AudioGap("l1"),))


# ===========================================================================
# Determinism + programmer-error refusals
# ===========================================================================


def test_same_inputs_produce_the_same_master_digest():
    first = build(two_line_timeline(), FakeTts())
    second = build(two_line_timeline(), FakeTts())
    assert first.master.digest == second.master.digest
    assert [c.digest for c in first.candidates] == \
        [c.digest for c in second.candidates]
    assert [c.seed for c in first.candidates] == [c.seed for c in second.candidates]
    # a different pacing policy is a different timeline, and says so
    other = build(two_line_timeline(), FakeTts(),
                  policy=am.SpeechPolicy(pause_after_s=0.9))
    assert other.master.digest != first.master.digest


def test_unlocked_dialogue_is_refused():
    unlocked = am.DialogueTimeline(lines=(am.Line("l1", "MIRA", "hi"),))
    with pytest.raises(ValueError, match="LOCKED DialogueTimeline"):
        build(unlocked, FakeTts())


def test_uncast_speaker_and_bad_fan_out_width_are_refused():
    with pytest.raises(ValueError, match="no voice cast"):
        am.build_audio_master(two_line_timeline(), {"MIRA": CAST["MIRA"]},
                              synth=FakeTts().synth,
                              transcribe=FakeTts().transcribe, policy=POLICY)
    with pytest.raises(ValueError, match="candidates must be >= 1"):
        build(two_line_timeline(), FakeTts(), candidates=0)
    with pytest.raises(TypeError, match="VoiceProfile"):
        am.build_audio_master(two_line_timeline(), {"MIRA": "mira", "JON": "jon"},
                              synth=FakeTts().synth,
                              transcribe=FakeTts().transcribe, policy=POLICY)


def test_a_synth_seam_with_the_wrong_shape_is_a_loud_wiring_bug():
    with pytest.raises(TypeError, match="must return"):
        am.build_audio_master(two_line_timeline(), CAST,
                              synth=lambda line, voice, seed: "/a.wav",
                              transcribe=lambda ref: [], policy=POLICY)


def test_repair_table_covers_every_speech_check_and_adds_only_capability_gap():
    assert set(am.AUDIO_REPAIR) == set(am.speech.SPEECH_REPAIR) | {
        am.SIMILARITY_EVIDENCE_CHECK}
    assert set(am.AUDIO_REPAIR) == set(am._REPAIR_PRIORITY)
    assert am.AUDIO_REPAIR[am.SIMILARITY_EVIDENCE_CHECK] is RepairCode.CAPABILITY_GAP
    for name, code in am.speech.SPEECH_REPAIR.items():
        assert am.AUDIO_REPAIR[name] is code


def test_words_running_past_their_own_audio_are_clamped_and_reported():
    # An ASR that reports 3s of words over a 1s take is reporting about
    # nothing; the words stay inside the line window and the operator is told.
    timeline = am.DialogueTimeline(
        lines=(am.Line("l1", "MIRA", "Hello there"),)).lock()
    result = am.build_audio_master(
        timeline, {"MIRA": am.VoiceProfile.synthetic("v")},
        synth=lambda line, voice, seed: ("/a.wav", 1.0),
        transcribe=lambda ref: [{"word": "Hello", "start": 0.0, "end": 0.5},
                                {"word": "there", "start": 2.0, "end": 3.0}],
        candidates=1, policy=POLICY)
    assert result.ok
    timing = result.master.timing("l1")
    assert timing.end_s == pytest.approx(1.0)
    assert all(w.end_s <= timing.end_s + 1e-9 for w in timing.words)
    assert any("clamped" in w for w in result.warnings)


def test_every_result_shape_is_json_serializable_for_the_route_layer():
    ok = build(two_line_timeline(), FakeTts())
    payload = json.loads(json.dumps(ok.to_dict()))
    assert payload["ok"] is True and payload["candidates_considered"] == 6
    assert payload["master"]["locked"] is True
    assert payload["repair_codes"] == []

    gapped = build(two_line_timeline(max_seconds=1.0),
                   FakeTts(duration_fn=lambda l, i: 9.0), candidates=1)
    assert not gapped.ok
    gap_payload = json.loads(json.dumps(gapped.to_dict()))
    assert gap_payload["master"] is None and gap_payload["gaps"]
    assert gap_payload["gaps"][0]["scorecards"][0]["hard_pass"] is False


def test_a_passing_card_diagnoses_no_repair():
    timeline = two_line_timeline()
    card = am.judge_candidate(timeline.lines[0], 1.0,
                              [{"word": w, "start": 0.0, "end": 0.1}
                               for w in timeline.lines[0].text.split()],
                              0.9, POLICY)
    assert card.hard_pass and card.repair_code is None
    assert am.candidate_repair_codes(card) == ()
    assert am.AudioGap("l1").primary_code is None
