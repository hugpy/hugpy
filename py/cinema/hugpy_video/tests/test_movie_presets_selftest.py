"""Movie-template registry conformance (converted from
``video_intel/_selftest_movie_presets.py``; pure in-process, no GPU/bus).

  1) every MOVIE_PRESETS id is unique and the ordered accessor agrees;
  2) each template's goal timeline is CONTIGUOUS and tiles [0, total) with
     non-empty prompts — proven by feeding it through ``make_movie``;
  3) each model_key is a non-empty string;
  4) ``apply()["request"]`` round-trips into the SAME MovieSpec.
"""
from __future__ import annotations

import pytest

from hugpy_video.intel.movie_schema import GoalInterval, make_movie, total_frames
from hugpy_video.intel.presets import (
    MOVIE_PRESETS,
    available_movie_presets,
    get_movie_preset,
)


def _spec_direct(preset):
    return make_movie(
        goals=tuple(GoalInterval(g["start_frame"], g["end_frame"], g["prompt"])
                    for g in preset.goals),
        model_id=preset.model_key, width=preset.width, height=preset.height,
        steps=preset.steps, guidance=preset.guidance, fps=preset.fps,
        assemble=True, chain=preset.chain, vision_enabled=preset.vision_enabled,
        score_threshold=preset.score_threshold,
    )


def _spec_from_request(req: dict):
    intervals = tuple(
        GoalInterval(start_frame=g["start_frame"], end_frame=g["end_frame"], prompt=g["prompt"])
        for g in req["goals"])
    return make_movie(
        goals=intervals, model_id=req["model_id"], width=req["width"], height=req["height"],
        steps=req["steps"], guidance=req["guidance"], fps=req["fps"], assemble=req["assemble"],
        chain=req["chain"], vision_enabled=req["vision_enabled"],
        score_threshold=req["score_threshold"],
    )


def test_ids_unique_and_registry_agrees():
    presets = available_movie_presets()
    ids = [p.id for p in presets]
    assert presets, "no movie templates registered"
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"
    assert set(ids) == set(MOVIE_PRESETS)
    assert len(ids) == len(MOVIE_PRESETS)


@pytest.mark.parametrize("preset", available_movie_presets(), ids=lambda p: p.id)
def test_template_is_sound(preset):
    assert isinstance(preset.model_key, str) and preset.model_key.strip()
    for i, g in enumerate(preset.goals):
        assert isinstance(g.get("prompt"), str) and g["prompt"].strip(), f"goals[{i}].prompt"

    spec_direct = _spec_direct(preset)          # raises on gap/overlap/bad range
    total = total_frames(spec_direct)
    assert total == preset.total_frames()
    cursor = 0
    for i, g in enumerate(preset.goals):
        assert g["start_frame"] == cursor, f"goals[{i}] not contiguous"
        cursor = g["end_frame"]
    assert cursor == total, "goals do not tile [0, total)"

    env = preset.apply()
    assert env.get("ok") is True and env.get("id") == preset.id
    req = env["request"]
    assert req["model_id"] == preset.model_key
    assert _spec_from_request(req) == spec_direct
    assert get_movie_preset(preset.id) is preset
