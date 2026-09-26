"""A model sweep must run with the same sampler and input settings as the clip form."""

import pytest

from hugpy_video.intel.studio.tester import (
    make_studio_tester,
    run_tester,
    studio_tester_from_dict,
)


def test_settings_survive_bus_roundtrip_and_each_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_MODEL_BATTERY", "off")
    spec = make_studio_tester(
        category="clip", prompt="turn toward camera", models=["first", "second"],
        start_image="/tmp/start.png", steps=28, cfg=5.5,
        requested_frames=41, negative="blur", out_root=str(tmp_path),
    )
    restored = studio_tester_from_dict(vars(spec))
    calls = []

    def generate(model, prompt, **settings):
        calls.append((model, prompt, settings))
        return False, "", "model unavailable"

    result = run_tester(restored.category, restored.prompt, list(restored.models),
                        start_image=restored.start_image, steps=restored.steps,
                        cfg=restored.cfg, requested_frames=restored.requested_frames,
                        negative=restored.negative, out_root=str(tmp_path),
                        video_generator=generate)
    assert result["count"] == 2 and result["ok_count"] == 0
    assert [c[0] for c in calls] == ["first", "second"]
    assert all(c[2]["start_image"] == "/tmp/start.png" and
               c[2]["steps"] == 28 and c[2]["cfg"] == 5.5 and
               c[2]["requested_frames"] == 41 and
               c[2]["negative"] == "blur" for c in calls)


@pytest.mark.parametrize("field,value", [
    ("steps", 0), ("cfg", 21), ("requested_frames", 0),
    ("negative", ["blur"]),
])
def test_invalid_settings_refuse_before_queue(field, value):
    with pytest.raises((ValueError, TypeError)):
        make_studio_tester(category="clip", prompt="test", **{field: value})


def test_empty_model_roster_is_a_gap(tmp_path, monkeypatch):
    monkeypatch.setattr("hugpy_video.intel.studio.tester.enumerate_models", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="no servable models"):
        run_tester("clip", "test", out_root=str(tmp_path))
