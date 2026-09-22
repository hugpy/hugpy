"""k98 — voice vertical I: speech evidence producers + the Chatterbox runner.

Two units, both testable on a GPU-less box with no chatterbox installed:

  * ``oracle/speech.py`` — pure functions over data structures. No fixtures, no
    fakes, no I/O: the normalization rule, the line-match miss budget, the
    similarity threshold edges, the unscored path and the scorecard fold.
  * ``video_intel/runners/tts_chatterbox.py`` — probe honesty, the
    authorization refusal (which must happen BEFORE any import), the standard
    "backend unavailable" raise, and a full synthesis against a FAKE
    ``chatterbox`` module installed in ``sys.modules``. The fake proves the
    adapter's contract (wav bytes + sidecar facts) without a 13.9 GB checkpoint
    or a GPU.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_speech.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys
import types
import wave

import pytest

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_oracle import speech
from hugpy_oracle.contracts import CheckKind, RepairCode
from hugpy_video.intel.runners import tts_chatterbox as tts


# ===========================================================================
# speech.py — normalization
# ===========================================================================


def test_normalization_folds_punctuation_case_and_typography():
    assert speech.normalize_tokens("Hello, there!") == ("hello", "there")
    assert speech.normalize_tokens("HELLO   there") == ("hello", "there")
    # curly apostrophe folds to straight and stays INSIDE the word
    assert speech.normalize_tokens("Don’t") == ("don't",)
    # edge apostrophes (quoting) are stripped, intra-word ones are not
    assert speech.normalize_tokens("'quote' don't") == ("quote", "don't")
    # em-dashes / ellipses / brackets are separators, not tokens
    assert speech.normalize_tokens("a — b… [c]") == ("a", "b", "c")
    assert speech.normalize_tokens("") == ()
    assert speech.normalize_tokens("!!! ...") == ()


def test_token_stream_flattens_multi_token_asr_words():
    words = [{"word": " New York,"}, {"word": "please."}]
    assert speech.transcript_token_stream(words) == ("new", "york", "please")


def test_word_shapes_all_read(monkeypatch):
    """A TranscribeWord, a raw whisper dict and a bare string all work."""
    from hugpy_media.schemas.whisper_schemas import TranscribeWord
    typed = TranscribeWord(word="hello", start=0.0, end=0.4, probability=0.9)
    assert speech.transcript_token_stream(
        [typed, {"word": "there"}, {"text": "friend"}, "again"]
    ) == ("hello", "there", "friend", "again")


def test_allowed_misses_is_one_per_eight_tokens():
    assert speech.allowed_misses(0) == 0
    assert speech.allowed_misses(7) == 0     # a short line must match every token
    assert speech.allowed_misses(8) == 1
    assert speech.allowed_misses(15) == 1
    assert speech.allowed_misses(16) == 2


# ===========================================================================
# speech.py — LINE_OMITTED
# ===========================================================================


def _words(*tokens: str):
    return [{"word": t} for t in tokens]


def test_lines_present_passes_on_exact_round_trip():
    check = speech.check_lines_present(
        ["Hello there.", "How are you?"],
        _words("Hello", "there", "how", "are", "you"))
    assert check.passed
    assert check.kind is CheckKind.SPEECH
    assert check.value == 2 and check.threshold == 2


def test_lines_present_ignores_punctuation_and_case():
    check = speech.check_lines_present(
        ["We're done — really."], _words("We’re", "DONE...", "(really!)"))
    assert check.passed, check.detail


def test_lines_present_fails_when_a_content_word_is_omitted():
    check = speech.check_lines_present(
        ["Bring me the hammer"], _words("bring", "me", "the"))
    assert not check.passed
    assert "hammer" in check.detail
    assert check.value == 0 and check.threshold == 1


def test_lines_present_tolerates_one_miss_per_eight_tokens():
    line = "one two three four five six seven eight"          # 8 tokens -> budget 1
    ok = speech.check_lines_present(
        [line], _words("one", "two", "three", "four", "five", "six", "seven"))
    assert ok.passed, ok.detail
    # a second miss on the same line exceeds the budget
    bad = speech.check_lines_present(
        [line], _words("one", "two", "three", "four", "five", "six"))
    assert not bad.passed


def test_lines_present_requires_order_within_and_across_lines():
    scrambled = speech.check_lines_present(
        ["alpha bravo charlie delta"],
        _words("delta", "charlie", "bravo", "alpha"))
    assert not scrambled.passed
    # every word of both lines is present, but line 2 is spoken first
    swapped = speech.check_lines_present(
        ["alpha bravo", "charlie delta"],
        _words("charlie", "delta", "alpha", "bravo"))
    assert not swapped.passed


def test_lines_present_empty_transcript_fails_loudly():
    check = speech.check_lines_present(["anything at all"], [])
    assert not check.passed
    assert "no words" in check.detail
    assert not speech.is_unscored(check)   # a silent take is a FAILURE, not a gap


def test_lines_present_with_no_expected_lines_is_unscored():
    check = speech.check_lines_present([], _words("whatever"))
    assert check.passed and speech.is_unscored(check)
    assert check.value is None


# ===========================================================================
# speech.py — VOICE_SIMILARITY_LOW
# ===========================================================================


def test_similarity_threshold_edges():
    below = speech.check_speaker_similarity(0.7499)
    at = speech.check_speaker_similarity(speech.DEFAULT_SIMILARITY_THRESHOLD)
    above = speech.check_speaker_similarity(0.9)
    assert not below.passed
    assert at.passed, "the threshold value itself must PASS"
    assert above.passed
    assert below.threshold == speech.DEFAULT_SIMILARITY_THRESHOLD
    assert below.kind is CheckKind.IDENTITY


def test_similarity_custom_threshold():
    assert speech.check_speaker_similarity(0.6, threshold=0.5).passed
    assert not speech.check_speaker_similarity(0.6, threshold=0.9).passed


def test_similarity_none_is_unscored_not_passed_or_failed():
    check = speech.check_speaker_similarity(None)
    assert speech.is_unscored(check)
    assert check.value is None
    assert "no registered backend" in check.detail
    assert check.passed  # unscored must not FAIL the artifact for the fleet's gap


def test_similarity_nan_is_unscored():
    assert speech.is_unscored(speech.check_speaker_similarity(float("nan")))


# ===========================================================================
# speech.py — SHOT_TOO_SHORT
# ===========================================================================


def test_duration_fit_passes_inside_the_shot_and_its_slack():
    assert speech.check_duration_fit(2.0, 3.0).passed
    assert speech.check_duration_fit(3.0, 3.0).passed
    # exactly at shot + tolerance still fits (bounded retiming)
    assert speech.check_duration_fit(
        3.0 + speech.DEFAULT_DURATION_TOLERANCE, 3.0).passed


def test_duration_fit_fails_when_audio_outruns_the_shot():
    check = speech.check_duration_fit(4.0, 3.0)
    assert not check.passed
    assert check.kind is CheckKind.SYNC
    assert "extend the shot" in check.detail
    assert check.threshold == pytest.approx(3.0 + speech.DEFAULT_DURATION_TOLERANCE)


def test_duration_fit_unknown_durations_are_unscored():
    assert speech.is_unscored(speech.check_duration_fit(None, 3.0))
    assert speech.is_unscored(speech.check_duration_fit(3.0, None))


def test_duration_fit_refuses_negative_input():
    with pytest.raises(ValueError):
        speech.check_duration_fit(-1.0, 3.0)


# ===========================================================================
# speech.py — the scorecard fold
# ===========================================================================


def test_scorecard_all_green():
    card = speech.speech_scorecard(
        expected_lines=["hello there"], transcript_words=_words("hello", "there"),
        similarity=0.91, audio_seconds=1.2, shot_seconds=2.0)
    assert card.hard_pass
    assert card.repair_code is None
    assert card.confidence == 1.0
    assert card.judge_results == ()       # deterministic checks invent no judge
    assert {c.name for c in card.checks} == {
        "speech.lines_present", "speech.speaker_similarity", "sync.duration_fit"}


def test_scorecard_names_line_omitted_first():
    card = speech.speech_scorecard(
        expected_lines=["bring me the hammer"], transcript_words=_words("bring"),
        similarity=0.1, audio_seconds=9.0, shot_seconds=1.0)
    assert not card.hard_pass
    assert card.repair_code is RepairCode.LINE_OMITTED   # priority: biggest first
    assert "do not rewrite the script" in card.recommended_repair


def test_scorecard_names_voice_similarity_low():
    card = speech.speech_scorecard(
        expected_lines=["hello"], transcript_words=_words("hello"),
        similarity=0.2, audio_seconds=1.0, shot_seconds=5.0)
    assert card.repair_code is RepairCode.VOICE_SIMILARITY_LOW


def test_scorecard_names_shot_too_short():
    card = speech.speech_scorecard(
        expected_lines=["hello"], transcript_words=_words("hello"),
        similarity=0.99, audio_seconds=6.0, shot_seconds=1.0)
    assert card.repair_code is RepairCode.SHOT_TOO_SHORT


def test_scorecard_confidence_is_the_scored_fraction():
    card = speech.speech_scorecard(
        expected_lines=["hello"], transcript_words=_words("hello"),
        similarity=None, audio_seconds=1.0, shot_seconds=5.0)
    assert card.hard_pass                       # unscored does not fail the card
    assert card.confidence == pytest.approx(2 / 3, abs=1e-3)
    assert "unscored (no evidence): speech.speaker_similarity" in card.diagnosis


def test_scorecard_round_trips_through_the_wire_shape():
    card = speech.speech_scorecard(
        expected_lines=["hello"], transcript_words=_words("nope"),
        similarity=0.5, audio_seconds=1.0, shot_seconds=5.0)
    from hugpy_oracle.contracts import Scorecard
    assert Scorecard.from_dict(json.loads(json.dumps(card.to_dict()))) == card


def test_speech_repair_table_uses_only_existing_repair_codes():
    for name, code in speech.SPEECH_REPAIR.items():
        assert isinstance(code, RepairCode), name


# ===========================================================================
# tts_chatterbox — probe + refusals (no backend on this box)
# ===========================================================================


def test_probe_is_honest_here():
    probed = tts.probe()
    assert probed["importable"] is False
    assert "chatterbox" in probed["reason"]
    assert tts.BACKEND_PIP in probed["reason"]
    assert probed["package"] == "chatterbox"
    assert probed["runner_key"] == ("chatterbox", "tts")


def test_probe_states_the_pitch_analysis_invariant():
    """doc Stage 3: Chatterbox must NEVER be the pitch-analysis component."""
    probed = tts.probe()
    assert probed["pitch_analysis"] is False
    assert "pitch" in probed["pitch_analysis_note"]
    # and the module exposes no analysis verb at all
    assert not [n for n in dir(tts)
                if any(k in n.lower() for k in ("pitch", "prosody", "f0"))
                and callable(getattr(tts, n))]


def test_reference_without_authorization_is_refused_before_any_import(monkeypatch, tmp_path):
    """The refusal must beat the backend probe — an unauthorized voice request
    never loads a cloning model."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    called = []
    monkeypatch.setattr(tts, "probe", lambda: called.append(1) or {})
    monkeypatch.setattr(tts, "_load_backend",
                        lambda *a, **k: pytest.fail("backend loaded on a refusal"))
    with pytest.raises(tts.ReferenceVoiceUnauthorized) as exc:
        tts.synthesize(tts.TtsSpec(text="hello", reference_audio=str(ref)))
    assert exc.value.code == "missing_consent"
    assert "authorized=True" in str(exc.value)
    assert not called, "probe() ran before the authorization refusal"


def test_make_tts_refuses_the_same_thing_at_construction(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    with pytest.raises(tts.ReferenceVoiceUnauthorized):
        tts.make_tts("hello", reference_audio=str(ref))
    spec = tts.make_tts("hello", reference_audio=str(ref), authorized=True)
    assert spec.reference_audio == str(ref) and spec.authorized


def test_make_tts_validates_its_spec(tmp_path):
    with pytest.raises(tts.TtsSpecError):
        tts.make_tts("   ")
    with pytest.raises(tts.TtsSpecError):
        tts.make_tts("hi", seed=-1)
    with pytest.raises(tts.TtsSpecError):
        tts.make_tts("hi", reference_audio=str(tmp_path / "nope.wav"),
                     authorized=True)


def test_backend_unavailable_is_raised_here():
    with pytest.raises(tts.TtsBackendUnavailable) as exc:
        tts.synthesize(tts.TtsSpec(text="hello", device="cpu"))
    assert exc.value.code == "deps_missing"
    assert exc.value.retryable is False


def test_bus_entrypoint_returns_failures_as_data(tmp_path):
    """map §6: an expected failure crosses the boundary as DATA, never a raise."""
    result = tts.run_tts_chatterbox({"text": "hello", "device": "cpu"}, "job-1")
    assert result.ok is False and result.error.code == "deps_missing"
    assert result.error.retryable is False

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    refused = tts.run_tts_chatterbox(
        {"text": "hello", "reference_audio": str(ref)}, "job-2")
    assert refused.ok is False and refused.error.code == "missing_consent"

    bad = tts.run_tts_chatterbox({"text": "  "}, "job-3")
    assert bad.ok is False and bad.error.code == "bad_spec"

    wrong = tts.run_tts_chatterbox(object(), "job-4")
    assert wrong.ok is False and wrong.error.code == "bad_spec"


# ===========================================================================
# tts_chatterbox — synthesis against a FAKE backend
# ===========================================================================


class _FakeModel:
    """Stands in for ChatterboxTTS / ChatterboxMultilingualTTS."""

    sr = 24000
    instances: list = []

    def __init__(self, **init):
        self.init = init
        self.calls: list[dict] = []
        self.samples = 2400          # 0.1 s at 24 kHz
        self.as_numpy = False

    @classmethod
    def from_pretrained(cls, device=None, **kwargs):
        model = cls(device=device, **kwargs)
        cls.instances.append(model)
        return model

    def generate(self, text, audio_prompt_path=None, **kwargs):
        self.calls.append({"text": text, "audio_prompt_path": audio_prompt_path,
                           **kwargs})
        # a small triangle wave in float [-1, 1], shaped (1, N) like the real one
        wave_ = [[(i % 200) / 100.0 - 1.0 for i in range(self.samples)]]
        if self.as_numpy:
            import numpy
            return numpy.array(wave_, dtype="float32")
        return wave_


@pytest.fixture()
def fake_chatterbox(monkeypatch):
    """Install a fake ``chatterbox`` package so ``find_spec`` and
    ``import_module`` both resolve without the real 13.9 GB backend."""
    from importlib.machinery import ModuleSpec

    _FakeModel.instances = []
    pkg = types.ModuleType("chatterbox")
    pkg.__spec__ = ModuleSpec("chatterbox", loader=None, is_package=True)
    pkg.__path__ = []
    single = types.ModuleType("chatterbox.tts")
    single.__spec__ = ModuleSpec("chatterbox.tts", loader=None)
    single.ChatterboxTTS = _FakeModel
    multi = types.ModuleType("chatterbox.mtl_tts")
    multi.__spec__ = ModuleSpec("chatterbox.mtl_tts", loader=None)
    multi.ChatterboxMultilingualTTS = _FakeModel
    pkg.tts, pkg.mtl_tts = single, multi

    monkeypatch.setitem(sys.modules, "chatterbox", pkg)
    monkeypatch.setitem(sys.modules, "chatterbox.tts", single)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", multi)
    yield _FakeModel


def _read_wav(path):
    with wave.open(path, "rb") as fh:
        return (fh.getnchannels(), fh.getsampwidth(), fh.getframerate(),
                fh.getnframes())


def test_probe_sees_the_fake_backend(fake_chatterbox):
    probed = tts.probe()
    assert probed["importable"] is True
    assert probed["reason"] == ""


def test_synthesis_writes_a_wav_and_a_sidecar(fake_chatterbox, tmp_path):
    result = tts.synthesize(
        tts.TtsSpec(text="Hello there.", device="cpu", seed=7,
                    voice_style="wry"),
        out_dir=str(tmp_path))

    channels, width, rate, frames = _read_wav(result["audio_path"])
    assert (channels, width, rate) == (1, 2, 24000)
    assert frames == 2400
    assert os.path.getsize(result["audio_path"]) > 44   # header + real frames

    sidecar = json.load(open(result["sidecar_path"], encoding="utf-8"))
    assert sidecar["model_id"] == tts.MODEL_ID
    assert sidecar["sample_rate"] == 24000
    assert sidecar["duration_s"] == pytest.approx(0.1, abs=1e-6)
    assert sidecar["reference_used"] is False
    assert sidecar["seed"] == 7
    assert sidecar["voice_style"] == "wry"
    assert sidecar["pitch_analysis"] is False


def test_synthesis_accepts_a_numpy_buffer(fake_chatterbox, tmp_path):
    """The adapter must not care whether the backend hands back numpy/torch."""
    pytest.importorskip("numpy")
    fake = fake_chatterbox
    original = fake.generate

    def _numpy_generate(self, text, audio_prompt_path=None, **kwargs):
        self.as_numpy = True
        return original(self, text, audio_prompt_path=audio_prompt_path, **kwargs)

    fake.generate = _numpy_generate
    try:
        result = tts.synthesize(tts.TtsSpec(text="hi", device="cpu"),
                                out_dir=str(tmp_path))
    finally:
        fake.generate = original
    assert _read_wav(result["audio_path"])[3] == 2400


def test_authorized_reference_reaches_the_backend(fake_chatterbox, tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    result = tts.synthesize(
        tts.make_tts("Say it", reference_audio=str(ref), authorized=True,
                     device="cpu"),
        out_dir=str(tmp_path))
    call = fake_chatterbox.instances[-1].calls[-1]
    assert call["audio_prompt_path"] == str(ref)
    assert result["reference_used"] is True
    assert result["authorized"] is True


def test_language_takes_the_multilingual_class(fake_chatterbox, tmp_path):
    tts.synthesize(tts.TtsSpec(text="Bonjour", language="fr", device="cpu"),
                   out_dir=str(tmp_path))
    model = fake_chatterbox.instances[-1]
    assert model.init.get("t3_model") == tts.MTL_T3_MODEL
    assert model.calls[-1]["language_id"] == "fr"


def test_empty_waveform_is_refused(fake_chatterbox, tmp_path, monkeypatch):
    monkeypatch.setattr(_FakeModel, "generate",
                        lambda self, text, **kw: [], raising=True)
    with pytest.raises(tts.TtsSpecError):
        tts.synthesize(tts.TtsSpec(text="hi", device="cpu"), out_dir=str(tmp_path))


def test_pcm_scaling_infers_float_vs_int16():
    """Silent-or-shattered-wav guard: float audio is scaled, int PCM is not."""
    assert tts._to_pcm16([1.0, -1.0]) == b"\xff\x7f\x01\x80"
    assert tts._to_pcm16([32767.0, -32768.0]) == b"\xff\x7f\x00\x80"
    assert tts._to_pcm16([]) == b""


def test_flatten_handles_nested_and_tensor_like():
    class _Tensorish:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[0.5, -0.5]]

    assert tts._flatten([[0.5, -0.5]]) == [0.5, -0.5]
    assert tts._flatten(_Tensorish()) == [0.5, -0.5]


# ===========================================================================
# tts silence (2026-08-21) — the float->PCM16 scale fault and the content guard
#
# The fault, measured live on a-brain: chatterbox returns float32 that
# OVERSHOOTS unity (the PerTh watermarker adds energy to a near-full-scale
# waveform), so a real line came back with peak 1.011981 on two samples out of
# 58 560. The old ``peak <= 1.0`` scale test read that as "already integer PCM",
# multiplied by 1.0, and every float in [-1, 1] rounded to -1/0/1 — a valid
# PCM16 wav of exactly the right duration, peak amplitude 1, RMS -117 dBFS.
# Two tests: the scaler must survive overshoot, and a silent wav must never
# reach hard_pass again.
# ===========================================================================


def _samples(peak, n=2000):
    """A synthetic waveform whose peak is exactly ``peak`` — a sine, so that a
    scaling collapse is visible as "almost everything rounds to zero"."""
    import math as _math
    return [peak * _math.sin(2 * _math.pi * 3 * i / n) for i in range(n)]


def test_pcm_scaling_survives_a_float_waveform_that_overshoots_unity():
    """THE REGRESSION. Float audio peaking just past 1.0 is still float audio."""
    pcm = tts._to_pcm16(_samples(1.011981))
    peak, rms = tts._pcm_levels(pcm)
    assert peak >= 32767                      # the overshoot is CLIPPED...
    assert rms > 20000                        # ...and the rest is still loud
    # under the old rule this whole buffer collapsed to peak 1 / RMS ~0.05
    from hugpy_oracle.scorecard import SILENT_AUDIO_PEAK_FLOOR
    assert peak >= SILENT_AUDIO_PEAK_FLOOR


def test_pcm_scaling_still_passes_integer_pcm_through_unscaled():
    """The other half of the inference must not regress (the shattered-wav side
    of the same coin): whole numbers louder than unity are already integer PCM
    and are only clipped. ``test_pcm_scaling_infers_float_vs_int16`` pins the
    byte-level edges; this pins a realistic buffer."""
    int_pcm = tts._to_pcm16([float(v) for v in (0, 12000, -12000, 30000)])
    assert tts._pcm_levels(int_pcm) == (30000, pytest.approx(17234, abs=1))


def _write_wav_file(path, samples, sample_rate=24000):
    dur, peak, rms = tts._write_wav(str(path), samples, sample_rate)
    return dur, peak, rms


def test_written_wav_records_its_own_measured_level(tmp_path):
    """``_write_wav`` measures what it wrote — loudness, like duration, is never
    the backend's claim."""
    loud = tmp_path / "loud.wav"
    dur, peak, rms = _write_wav_file(loud, _samples(0.8, n=24000))
    assert dur == 1.0 and peak == pytest.approx(26214, abs=2) and rms > 15000
    with wave.open(str(loud), "rb") as fh:
        assert fh.getframerate() == 24000 and fh.getnframes() == 24000


def test_scorecard_fails_a_digitally_silent_wav(tmp_path):
    """THE CONTENT GUARD. A valid, right-duration, non-zero-byte wav that
    contains nothing must FAIL empty_output with EMPTY_OUTPUT, and the card must
    name the level it measured."""
    from hugpy_oracle import router, scorecard
    from hugpy_oracle.contracts import ArtifactKind, ExecutionReceipt, GoalSpec

    def _card(path):
        route = router.RouteDecision(
            capability="audio.tts", execution="execute", task="text-to-speech",
            produces=(ArtifactKind.AUDIO,))
        receipt = ExecutionReceipt(
            request=ExecutionReceipt.normalize_request({"task": route.task}),
            capability=route.capability, model_id="Viral2AI~chatterbox",
            worker=None, started_at="2026-08-21T00:00:00+00:00",
            ended_at="2026-08-21T00:00:20+00:00", duration_s=20.0)
        arts = [{"kind": "audio", "uri": str(path), "sha256": "0" * 64,
                 "duration_s": 1.0, "sample_rate": 24000}]
        goal = GoalSpec(objective="speak", raw_prompt="speak",
                        capability="audio.tts")
        return scorecard.build_technical_scorecard(goal, route, arts, receipt)

    # The exact silent buffer the bug produced: float audio multiplied by 1.0.
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(b"".join(
            __import__("struct").pack("<h", int(round(s)))
            for s in _samples(0.8, n=24000)))
    assert os.path.getsize(silent) > 40000        # a real, non-empty file
    card = _card(silent)
    assert card.hard_pass is False
    assert card.repair_code is RepairCode.EMPTY_OUTPUT
    assert {c.name for c in card.checks if not c.passed} == {"empty_output"}
    empty = [c for c in card.checks if c.name == "empty_output"][0]
    assert "peak 1/32767" in empty.detail and "dBFS" in empty.detail
    assert "digital silence" in (card.diagnosis or "")

    # ...and real speech-level audio still passes, level named on the card.
    loud = tmp_path / "loud.wav"
    _, loud_peak, _ = _write_wav_file(loud, _samples(0.8, n=24000))
    ok_card = _card(loud)
    assert ok_card.hard_pass is True
    detail = [c for c in ok_card.checks if c.name == "empty_output"][0].detail
    assert f"peak {loud_peak}/32767" in detail and loud_peak > 26000


def test_unmeasurable_audio_is_never_called_silent(tmp_path):
    """A format this guard cannot read is reported as unmeasurable and PASSES —
    the check refuses to convict on a number it does not have."""
    from hugpy_oracle import scorecard
    mp3ish = tmp_path / "clip.mp3"
    mp3ish.write_bytes(b"ID3\x03\x00\x00\x00" + b"\x00" * 128)
    ok, detail = scorecard._audio_substance(str(mp3ish))
    assert ok is True and "unmeasurable" in detail
