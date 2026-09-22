# k120 slice 2 — producer continuity refresh: schema round-trip, the advisory
# refresh helper's failure discipline, and the persisted-rewrite resume seam.
#
# House landmine respected (k53 card: "Do NOT patch managers.dispatch at the
# outer name only — mispatched tests run REAL inference", and it is measurably
# true): ``import abstract_hugpy_dev.managers.dispatch`` binds a module object
# that is NOT ``sys.modules["hugpy_engine.dispatch"]`` — the
# entry the helper's own lazy ``from … import execute_prompt`` resolves
# against. Patch the sys.modules entry, nothing else intercepts.
import json
import os

import pytest

from hugpy_video.intel.studio_movie_schema import (
    StudioMovieGoal,
    make_studio_movie,
    studio_movie_from_dict,
)
from hugpy_video.intel.runners.studio_movie import _continuity_refresh, _persisted_refresh
from dataclasses import asdict


def _two_goals():
    return (
        StudioMovieGoal(segment_id="s1", prompt="a cat"),
        StudioMovieGoal(segment_id="s2", prompt="the cat jumps",
                        parent_segment_id="s1", joint_mode="vace_extend"),
    )


# ---------------------------------------------------------------------------
# Schema: the new fields (and the frames regression) survive the bus rehydrate.
# ---------------------------------------------------------------------------

def test_continuity_fields_roundtrip_through_from_dict():
    spec = make_studio_movie(goals=_two_goals(), width=480, height=480, fps=24,
                             frames=57, continuity_refresh=True,
                             continuity_model="Qwen2.5-7B-Instruct-GGUF")
    back = studio_movie_from_dict(asdict(spec))
    assert back.continuity_refresh is True
    assert back.continuity_model == "Qwen2.5-7B-Instruct-GGUF"
    # The pre-existing regression k120 recon caught: movie-level frames used to
    # be dropped by studio_movie_from_dict on every rehydrate/resume.
    assert back.frames == 57


def test_continuity_defaults_off_and_prek120_specs_rehydrate():
    spec = make_studio_movie(goals=_two_goals(), width=480, height=480, fps=24)
    assert spec.continuity_refresh is False
    assert spec.continuity_model is None
    d = asdict(spec)
    # A pre-k120 spec.json has neither key — .get defaults must hold.
    d.pop("continuity_refresh", None)
    d.pop("continuity_model", None)
    back = studio_movie_from_dict(d)
    assert back.continuity_refresh is False
    assert back.continuity_model is None


def test_continuity_model_must_be_a_string():
    with pytest.raises(ValueError):
        make_studio_movie(goals=_two_goals(), width=480, height=480, fps=24,
                          continuity_model=7)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The refresh helper: advisory — (None, why) on any trouble, rewrite on success.
# ---------------------------------------------------------------------------

class _FakeRes:
    def __init__(self, text, ok=True):
        self.text = text
        self.ok = ok
        self.error = None if ok else "boom"


def _dispatch_module():
    """The one seam video drives the inference plane through (the runner
    binds ``from hugpy_video.intel.plane import execute_prompt`` lazily)."""
    from hugpy_video.intel import plane

    return plane


def _patch_execute(monkeypatch, replies):
    """Feed execute_prompt one canned _FakeRes per call, in order."""
    calls = []

    async def fake_execute_prompt(**kwargs):
        calls.append(kwargs)
        return replies[len(calls) - 1]

    monkeypatch.setattr(_dispatch_module(), "execute_prompt", fake_execute_prompt)
    return calls


def test_refresh_plane_raise_keeps_authored_prompt(monkeypatch):
    async def raising(**kwargs):
        raise RuntimeError("no worker")

    monkeypatch.setattr(_dispatch_module(), "execute_prompt", raising)
    refreshed, note = _continuity_refresh("/nonexistent.png", "prev", "next shot prompt")
    assert refreshed is None
    assert "refresh skipped" in note


def test_refresh_vision_not_ok_is_advisory(monkeypatch):
    _patch_execute(monkeypatch, [_FakeRes("", ok=False)])
    refreshed, note = _continuity_refresh("/f.png", None, "next shot prompt")
    assert refreshed is None
    assert "vision describe not-ok" in note


def test_refresh_success_rewrites_and_notes_the_frame(monkeypatch):
    desc = "A robotic jaguar stands centered, glowing blue, night alley, low key light."
    rewrite = ("A robotic jaguar with glowing blue seams climbs the spiral staircase "
               "to the lamp room, low-key light, present tense motion.")
    calls = _patch_execute(monkeypatch, [_FakeRes(desc), _FakeRes(rewrite)])
    refreshed, note = _continuity_refresh("/f.png", "prev prompt", "the cat climbs")
    assert refreshed == rewrite
    assert note.startswith("opened from previous frame: ")
    assert desc[:50] in note
    # First call is the vision describe (image task + the frame file), second the
    # text rewrite with the EXPLICIT model (never the silent task default).
    assert calls[0]["task"] == "image-text-to-text" and calls[0]["file"] == "/f.png"
    assert calls[1]["task"] == "text-generation"
    assert calls[1]["model_key"] == "Qwen2.5-7B-Instruct-GGUF"


def test_refresh_model_pin_rides_to_the_rewrite(monkeypatch):
    calls = _patch_execute(
        monkeypatch,
        [_FakeRes("a plain frame description of reasonable length"),
         _FakeRes("a rewritten prompt of a perfectly reasonable length for a shot")])
    refreshed, _ = _continuity_refresh("/f.png", None, "next", model="My-Model")
    assert refreshed is not None
    assert calls[1]["model_key"] == "My-Model"


def test_refresh_rejects_degenerate_rewrites(monkeypatch):
    _patch_execute(monkeypatch, [_FakeRes("a frame description"), _FakeRes("ok")])
    refreshed, note = _continuity_refresh("/f.png", None, "next shot prompt")
    assert refreshed is None
    assert "too short" in note


# ---------------------------------------------------------------------------
# Resume determinism: the persisted rewrite is preferred over a fresh roll.
# ---------------------------------------------------------------------------

def test_persisted_refresh_reads_movie_json(tmp_path):
    root = str(tmp_path)
    with open(os.path.join(root, "movie.json"), "w", encoding="utf-8") as f:
        json.dump({"segments": [
            {"segment_id": "s1", "prompt": "authored one", "refresh_note": None},
            {"segment_id": "s2", "prompt": "the persisted rewrite",
             "refresh_note": "opened from previous frame: ..."},
        ]}, f)
    assert _persisted_refresh(root, "s2") == "the persisted rewrite"
    # A segment without a recorded rewrite (note None/absent) yields None.
    assert _persisted_refresh(root, "s1") is None
    # No movie.json at all — no persisted rewrite, never a raise.
    assert _persisted_refresh(str(tmp_path / "missing"), "s2") is None
