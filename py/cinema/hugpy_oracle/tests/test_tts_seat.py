"""The chatterbox SEAT — how a worker comes to be able to speak, and how the
fleet learns it can. Every test runs on a GPU-less box with no ``chatterbox``
installed; nothing here reaches a worker, a network or a checkpoint.

What is covered, and why each one exists:

  * ``managers/tts/seat.py`` — the ONE answer the runner and the heartbeat both
    read. The drift this prevents is the whole reason it is a module and not two
    probes: a worker that advertises a task it cannot run, or runs one it never
    advertised.
  * The dispatch REGISTRATION — ``("transformers","text-to-speech")`` present in
    the runner table, the builder table, ``RUNNER_PAIRS`` and ``TASK_DEPS``, all
    four agreeing. A row missing from any one of them is the "no runner
    registered" class of failure this seating exists to close.
  * The AUTHORITY refusal at the runner, which must happen before a backend is
    loaded (k98's last line of defence, behind k97's typed gate).
  * ``imports/src/model_classifier`` — a speech checkpoint is recognised from
    the weight files its own loader reads, never from a name (k61's rule,
    applied to audio), and does NOT fire on an image/chat model dir.
  * ``oracle/runtime.extract_artifacts`` — relayed wav BYTES become a real,
    hashable file on the box that answers, instead of a path that only existed
    on a worker.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_tts_seat.py -q
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import sys
import types
import wave

import pytest

logging.disable(logging.INFO)   # silence the registry chatter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from hugpy_media.tts import seat
from hugpy_media.tts import tts_runner
from hugpy_media.tts.schemas import TtsRequest, TtsResult
from hugpy_video.intel.runners import tts_chatterbox


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fresh_seat(monkeypatch):
    """Clear the seat's TTL cache so each test measures its own world."""
    monkeypatch.setattr(seat, "_CACHE", {"at": 0.0, "seat": None})


def _wav_bytes(seconds: float = 0.25, rate: int = 24000) -> bytes:
    import io
    frames = int(seconds * rate)
    pcm = b"".join(struct.pack("<h", (i % 200) - 100) for i in range(frames))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# seat resolution — the single source of truth
# ---------------------------------------------------------------------------


def test_no_backend_anywhere_is_unavailable_with_the_remedy(monkeypatch):
    """Nothing importable and no profile venv: unavailable, and the reason names
    BOTH the pip distribution and the profile to materialize. An unavailable
    seat that does not say what to do is a shrug."""
    _fresh_seat(monkeypatch)
    monkeypatch.setattr(seat, "_importable_here", lambda: False)
    monkeypatch.setattr(seat, "_profile_python", lambda: None)
    out = seat.resolve()
    assert out["available"] is False
    assert out["mode"] == ""
    assert seat.BACKEND_PIP in out["reason"]
    assert seat.DEFAULT_PROFILE in out["reason"]
    assert seat.available() is False


def test_in_process_backend_needs_no_child(monkeypatch):
    """A box whose own interpreter holds the backend seats it in-process — no
    profile venv, no subprocess."""
    _fresh_seat(monkeypatch)
    monkeypatch.setattr(seat, "_importable_here", lambda: True)
    monkeypatch.setattr(seat, "_profile_python",
                        lambda: pytest.fail("must not look for a profile"))
    out = seat.resolve()
    assert out == {"available": True, "mode": "in-process", "python": "",
                   "reason": "", "profile": seat.DEFAULT_PROFILE}


def test_profile_venv_seat_is_proved_by_asking_that_interpreter(monkeypatch):
    """The profile seat is affirmed by the interpreter that would do the work,
    not by the agent's own find_spec — which answers "no" for a seat that works
    perfectly. This IS the whisper lesson, mirrored."""
    _fresh_seat(monkeypatch)
    monkeypatch.setattr(seat, "_importable_here", lambda: False)
    monkeypatch.setattr(seat, "_profile_python", lambda: "/envs/cb/bin/python")
    monkeypatch.setattr(seat, "_child_can_import", lambda python: (True, ""))
    out = seat.resolve()
    assert out["available"] is True
    assert out["mode"] == "profile-venv"
    assert out["python"] == "/envs/cb/bin/python"
    assert out["reason"] == ""


def test_a_profile_venv_without_the_package_is_not_a_seat(monkeypatch):
    """A venv that exists but does not hold the backend is NOT a seat. STRICT
    and affirmative: presence of a directory proves nothing."""
    _fresh_seat(monkeypatch)
    monkeypatch.setattr(seat, "_importable_here", lambda: False)
    monkeypatch.setattr(seat, "_profile_python", lambda: "/envs/cb/bin/python")
    monkeypatch.setattr(seat, "_child_can_import",
                        lambda python: (False, "no chatterbox there"))
    out = seat.resolve()
    assert out["available"] is False
    assert out["mode"] == ""
    assert "no chatterbox there" in out["reason"]


def test_the_seat_is_cached_so_a_heartbeat_never_spawns_a_probe(monkeypatch):
    """The heartbeat consults this every beat; the expensive half (spawning the
    profile python) must not run every beat."""
    _fresh_seat(monkeypatch)
    calls = {"n": 0}

    def _child(python):
        calls["n"] += 1
        return True, ""

    monkeypatch.setattr(seat, "_importable_here", lambda: False)
    monkeypatch.setattr(seat, "_profile_python", lambda: "/envs/cb/bin/python")
    monkeypatch.setattr(seat, "_child_can_import", _child)
    seat.resolve()
    seat.resolve()
    seat.resolve()
    assert calls["n"] == 1
    seat.resolve(refresh=True)
    assert calls["n"] == 2


def test_the_seat_names_match_the_adapter_constants():
    """Two modules naming the same package must not drift; seat.py keeps its own
    copies to stay import-cheap, so a test holds them equal instead."""
    assert seat.BACKEND_PACKAGE == tts_chatterbox.BACKEND_PACKAGE
    assert seat.BACKEND_PIP == tts_chatterbox.BACKEND_PIP
    assert seat.TASK == tts_chatterbox.TASK


# ---------------------------------------------------------------------------
# registration — four tables, one truth
# ---------------------------------------------------------------------------


def test_text_to_speech_is_registered_in_every_table_that_gates_it():
    from hugpy_engine.resolvers.categories.frameworks import FRAMEWORK_RUNNERS, KNOWN_TASKS_REGISTRY
    from hugpy_engine.resolvers.categories.builders import MODEL_REQUEST_BUILDERS
    from hugpy_engine.categories import HF_TASK_TO_TASKS, RUNNER_PAIRS
    from hugpy_engine.task_deps import TASK_DEPS

    key = ("transformers", "text-to-speech")
    assert FRAMEWORK_RUNNERS[key] is tts_runner.ChatterboxTtsRunner
    assert key in MODEL_REQUEST_BUILDERS
    assert key in RUNNER_PAIRS            # else the registry row is unserveable
    assert "text-to-speech" in KNOWN_TASKS_REGISTRY
    assert HF_TASK_TO_TASKS["text-to-speech"] == ["text-to-speech"]
    assert TASK_DEPS["text-to-speech"] == (seat.BACKEND_PACKAGE, "tts")


def test_the_builder_takes_the_line_from_text_or_prompt():
    from hugpy_media.plugin_builders import _build_tts_request
    req = _build_tts_request({"text": "hello"}, "Viral2AI~chatterbox")
    assert isinstance(req, TtsRequest)
    assert req.text == "hello" and req.model_key == "Viral2AI~chatterbox"
    assert _build_tts_request({"prompt": "hi"}, "m").text == "hi"
    with pytest.raises(ValueError):
        _build_tts_request({"text": "   "}, "m")


def test_the_builder_never_invents_an_authorization():
    """``authorized`` is forwarded, never defaulted to True — it means k97's gate
    granted a VOICE authorization, which no builder can know."""
    from hugpy_media.plugin_builders import _build_tts_request
    plain = _build_tts_request({"text": "x", "reference_audio": "/v.wav"}, "m")
    assert plain.authorized is False
    granted = _build_tts_request(
        {"text": "x", "reference_audio": "/v.wav", "authorized": True}, "m")
    assert granted.authorized is True


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


def _run(runner, req):
    return asyncio.run(runner.run(req))


def test_an_unseated_box_refuses_with_deps_missing(monkeypatch):
    """No seat -> error-as-data with the tree's existing code, never a raise and
    never a silent empty result."""
    monkeypatch.setattr(tts_runner._seat, "resolve",
                        lambda **kw: {"available": False, "mode": "",
                                      "python": "", "profile": "cb",
                                      "reason": "nothing installed"})
    runner = tts_runner.ChatterboxTtsRunner(
        types.SimpleNamespace(model_key="Viral2AI~chatterbox"))
    res = _run(runner, TtsRequest(request_id="r1", model_key="Viral2AI~chatterbox",
                                  text="hello"))
    assert isinstance(res, TtsResult)
    assert res.ok is False
    assert res.error_code == "deps_missing"
    assert "nothing installed" in res.error
    assert res.audio == []


def test_a_reference_voice_without_authorization_is_refused_before_the_backend(
        monkeypatch, tmp_path):
    """k98's last line of defence, reached through the fleet's runner: the
    refusal happens at spec construction, so no backend is loaded, no GPU is
    touched, and there is no file to leak. It never downgrades to the default
    voice."""
    reference = tmp_path / "voice.wav"
    reference.write_bytes(_wav_bytes())
    monkeypatch.setattr(tts_runner._seat, "resolve",
                        lambda **kw: {"available": True, "mode": "in-process",
                                      "python": "", "profile": "cb",
                                      "reason": ""})
    monkeypatch.setattr(tts_runner, "_weights_dir", lambda key: None)
    monkeypatch.setattr(tts_runner, "_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        tts_chatterbox, "synthesize",
        lambda *a, **k: pytest.fail("the backend must never be reached"))

    runner = tts_runner.ChatterboxTtsRunner(
        types.SimpleNamespace(model_key="Viral2AI~chatterbox"))
    res = _run(runner, TtsRequest(request_id="r2", model_key="Viral2AI~chatterbox",
                                  text="say this", reference_audio=str(reference),
                                  authorized=False))
    assert res.ok is False
    assert res.error_code == "missing_consent"
    assert res.audio == []


def test_a_successful_run_measures_the_wav_it_wrote(monkeypatch, tmp_path):
    """Duration and sample rate are READ BACK from the file (invariant 11), the
    sidecar travels with the bytes, and b64 is present when asked for."""
    out = tmp_path / "out"
    out.mkdir()
    audio_path = out / "take.wav"
    audio_path.write_bytes(_wav_bytes(seconds=0.5))

    monkeypatch.setattr(tts_runner._seat, "resolve",
                        lambda **kw: {"available": True, "mode": "in-process",
                                      "python": "", "profile": "cb",
                                      "reason": ""})
    monkeypatch.setattr(tts_runner, "_weights_dir", lambda key: "/weights/cb")
    monkeypatch.setattr(tts_runner, "_output_dir", lambda: str(out))
    monkeypatch.setattr(
        tts_runner, "_run_in_process",
        lambda spec_kwargs, out_dir: {
            "ok": True, "elapsed_s": 1.5,
            "vram_peak_reserved_bytes": 3_458_203_648,
            "manifest": {"audio_path": str(audio_path),
                         "sidecar_path": str(out / "take.json"),
                         "model_id": "Viral2AI~chatterbox",
                         "weights_source": "local",
                         # a LYING claim, deliberately: the runner must ignore it
                         "duration_s": 99.0, "sample_rate": 8000}})

    runner = tts_runner.ChatterboxTtsRunner(
        types.SimpleNamespace(model_key="Viral2AI~chatterbox"))
    res = _run(runner, TtsRequest(request_id="r3", model_key="Viral2AI~chatterbox",
                                  text="measured, not claimed"))
    assert res.ok is True and len(res.audio) == 1
    clip = res.audio[0]
    assert clip.sample_rate == 24000            # from the wav, not the claim
    assert abs(clip.duration_s - 0.5) < 0.01    # from the wav, not the claim
    assert clip.b64 and base64.b64decode(clip.b64)[:4] == b"RIFF"
    assert clip.sidecar["weights_source"] == "local"
    assert clip.sidecar["seat_mode"] == "in-process"
    assert res.vram_peak_bytes == 3_458_203_648


def test_a_missing_output_is_reported_not_assumed(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_runner._seat, "resolve",
                        lambda **kw: {"available": True, "mode": "in-process",
                                      "python": "", "profile": "cb",
                                      "reason": ""})
    monkeypatch.setattr(tts_runner, "_weights_dir", lambda key: None)
    monkeypatch.setattr(tts_runner, "_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        tts_runner, "_run_in_process",
        lambda spec_kwargs, out_dir: {
            "ok": True, "manifest": {"audio_path": str(tmp_path / "gone.wav")}})
    runner = tts_runner.ChatterboxTtsRunner(
        types.SimpleNamespace(model_key="m"))
    res = _run(runner, TtsRequest(request_id="r4", model_key="m", text="x"))
    assert res.ok is False and res.error_code == "missing_output"


# ---------------------------------------------------------------------------
# the adapter's local-weights preference (k98 + the seat)
# ---------------------------------------------------------------------------


def test_the_spec_refuses_a_weights_dir_that_is_not_there(tmp_path):
    """A named-but-absent checkpoint is a spec fault, not a silent fallback to
    the backend's own download: that would serve DIFFERENT bytes under this
    model_id."""
    with pytest.raises(tts_chatterbox.TtsSpecError):
        tts_chatterbox.make_tts("hello", weights_dir=str(tmp_path / "nope"))
    spec = tts_chatterbox.make_tts("hello", weights_dir=str(tmp_path))
    assert spec.weights_dir == str(tmp_path)


def test_local_weights_are_preferred_over_the_backends_own_download(monkeypatch,
                                                                    tmp_path):
    """``from_local`` wins when the spec names a checkpoint dir — the registry
    row's own bytes, not a second copy fetched from the hub."""
    seen = {}

    class _Model:
        sr = 24000

        @classmethod
        def from_local(cls, ckpt_dir, device):
            seen["from_local"] = (ckpt_dir, device)
            return cls()

        @classmethod
        def from_pretrained(cls, *a, **k):
            seen["from_pretrained"] = True
            return cls()

    module = types.ModuleType("chatterbox.tts")
    module.ChatterboxTTS = _Model
    monkeypatch.setitem(sys.modules, "chatterbox.tts", module)

    spec = tts_chatterbox.make_tts("hi", weights_dir=str(tmp_path))
    tts_chatterbox._load_backend(spec, "cuda")
    assert seen["from_local"] == (str(tmp_path), "cuda")
    assert "from_pretrained" not in seen


# ---------------------------------------------------------------------------
# the classifier: a speech checkpoint speaks for itself
# ---------------------------------------------------------------------------


def _speech_dir(root) -> str:
    d = root / "chatterbox"
    d.mkdir()
    for name in ("ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors"):
        (d / name).write_bytes(b"\0" * 16)
    (d / "tokenizer.json").write_text("{}")
    return str(d)


def test_a_chatterbox_checkpoint_dir_classifies_as_text_to_speech(tmp_path):
    from hugpy_engine import model_classifier as mc
    directory = _speech_dir(tmp_path)
    assert mc.is_speech_checkpoint_dir(directory) is True
    assert mc.classify_model_dir(directory) == {
        "tasks": ["text-to-speech"], "primary_task": "text-to-speech",
        "source": "speech_checkpoint"}


def test_the_speech_arm_does_not_fire_on_a_transformers_model(tmp_path):
    """A dir that declares a model_type is something else and is left alone —
    the arm speaks only for the shape that declares nothing."""
    from hugpy_engine import model_classifier as mc
    directory = _speech_dir(tmp_path)
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump({"model_type": "llama"}, fh)
    assert mc.is_speech_checkpoint_dir(directory) is False


def test_a_tts_row_stops_advertising_text_generation(tmp_path):
    """The registry corrector: two witnesses, either of which is the model
    speaking for itself. Fires only over the "nobody classified this" verdicts."""
    from hugpy_engine.config.models import models_config as mcfg
    directory = _speech_dir(tmp_path)
    assert mcfg._correct_speech_task(
        "transformers", ["text-generation"], {"dir": directory}) == \
        ["text-to-speech"]
    assert mcfg._correct_speech_task(
        "transformers", ["text-generation"],
        {"pipeline_tag": "text-to-speech"}) == ["text-to-speech"]
    # a row that already declares a real task of its own is never re-labelled
    assert mcfg._correct_speech_task(
        "transformers", ["text-to-image"], {"dir": directory}) == \
        ["text-to-image"]


# ---------------------------------------------------------------------------
# the artifact seam: relayed bytes become a real file
# ---------------------------------------------------------------------------


def test_relayed_audio_is_materialized_and_hashed(monkeypatch, tmp_path):
    """The worker's path is not readable here, so the inline bytes are written
    where the caller can actually read them — and the artifact hashes what was
    written rather than carrying a null sha for a path that does not exist."""
    from hugpy_oracle import runtime
    # The REAL _materialize_audio runs; only the root it writes under is moved
    # into the tmp dir (the oracle's own runs root, ``hugpy_oracle.config``),
    # so this exercises the write, not a stand-in for it.
    monkeypatch.setenv("HUGPY_ORACLE_RUNS_ROOT", str(tmp_path / "video_intel"))
    payload_b64 = base64.b64encode(_wav_bytes(seconds=0.25)).decode("ascii")
    arts = runtime.extract_artifacts("audio.tts", {"audio": [{
        "path": "/on/another/box/take.wav", "b64": payload_b64,
        "duration_s": 0.25, "sample_rate": 24000, "reference_used": False}]})
    assert len(arts) == 1
    art = arts[0]
    assert art["kind"] == "audio"
    assert art["uri"] == str(tmp_path / "video_intel" / "tts" / "take.wav")
    assert os.path.isfile(art["uri"])
    assert art["sha256"] and len(art["sha256"]) == 64
    assert art["duration_s"] == 0.25 and art["sample_rate"] == 24000


def test_a_readable_worker_path_is_used_as_is(tmp_path):
    """When the file IS readable here (in-process seat, or a shared store), it is
    hashed in place — no needless copy."""
    from hugpy_oracle import runtime
    path = tmp_path / "take.wav"
    path.write_bytes(_wav_bytes())
    arts = runtime.extract_artifacts("audio.tts", {"audio": [{
        "path": str(path), "b64": None, "duration_s": 0.25,
        "sample_rate": 24000}]})
    assert arts[0]["uri"] == str(path)
    assert arts[0]["sha256"]
