"""VIDEO GEN (2026-08-28) — registry seat for the studio spine.

Covers the two-vocabulary bridge (registry model keys -> studio zoo ids), the
transport schema, and — the actual regression guard — that the
text-to-video / image-to-video pairs resolve in both registries so
validate_registry stops flagging the Wan rows. Rendering itself is the studio
spine's and exercised on dev.

    ./venv/bin/pytest tests/test_video_gen_runner.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VG = importlib.import_module("hugpy_video.video_gen")


def test_studio_model_id_bridges_the_two_vocabularies():
    # The four Wan rows registered on this fleet, as discovered.
    assert VG.studio_model_id("Wan2.1-T2V-1.3B") == "wan2.1-t2v-1.3b"
    assert VG.studio_model_id("Wan-AI~Wan2.1-VACE-1.3B") == "wan2.1-vace-1.3b"
    assert VG.studio_model_id("Wan2.1-VACE-1.3B-diffusers") == "wan2.1-vace-1.3b"
    # No match in the studio zoo -> passed through for the router to refuse
    # as err-as-data, never silently rewritten to a different model.
    assert VG.studio_model_id("VACE-Wan2.1-1.3B-Preview") == \
        "vace-wan2.1-1.3b-preview"


def test_request_defaults_are_the_wan_reference_geometry():
    req = VG.VideoGenRequest(request_id="r", model_key="m", prompt="p")
    assert (req.width, req.height, req.fps) == (832, 480, 16)
    assert req.requested_frames is None      # None = bound model's default (81)
    assert req.vram_budget_gb is None        # None = AUTOFIT, not a low guess
    with pytest.raises(Exception):
        VG.VideoGenRequest(request_id="r", model_key="m", prompt="p",
                           steps=101)


def test_registry_pairs_resolve():
    """The plugin registers both tasks with the engine's task registry
    (``hugpy_engine.tasks``) — runner + builder — under framework transformers,
    and the lazy runner factory resolves to the real class without importing
    the studio spine at registration time."""
    from hugpy_engine import tasks as T
    from hugpy_video import plugin

    plugin.register()
    try:
        for task in ("text-to-video", "image-to-video"):
            spec = T.task_spec(task)
            assert spec is not None and spec.source == "hugpy_video", task
            assert "transformers" in spec.frameworks
            assert spec.build_request is plugin.build_videogen_request
            assert spec.runner is plugin.studio_video_runner       # lazy factory
            assert T.runner_for_task(task) is VG.StudioVideoRunner  # resolved class
        assert plugin.studio_video_runner() is VG.StudioVideoRunner
    finally:
        plugin.unregister()
    assert T.task_spec("text-to-video") is None


def test_builder_refuses_an_unconditioned_ask():
    from hugpy_video.plugin_builders import build_videogen_request as build
    with pytest.raises(ValueError):
        build({}, "Wan2.1-T2V-1.3B")
    req = build({"prompt": "a hillside at dawn", "requested_frames": 81},
                "Wan2.1-T2V-1.3B")
    assert req.model_key == "Wan2.1-T2V-1.3B"
    assert req.requested_frames == 81
