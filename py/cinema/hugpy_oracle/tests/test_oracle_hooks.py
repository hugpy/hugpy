"""The oracle's side of the video hook seam (``hugpy_oracle.relay.hooks``)."""

from __future__ import annotations

import types

import pytest

from hugpy_oracle.relay import hooks


def test_install_hooks_is_a_no_op_without_the_video_seams(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name in ("hugpy_video.hooks", "hugpy_video.jobs"):
            raise ModuleNotFoundError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake)
    assert hooks.install_hooks() == {"video_hooks": False, "prompt_coordinator": None,
                                     "performance_runner": None}


def test_install_hooks_wires_the_coordinator_and_registers_the_job():
    wired = {}
    fake_video_hooks = types.SimpleNamespace(
        set_prompt_coordinator=lambda impl: wired.__setitem__("coordinator", impl))
    fake_video_jobs = types.SimpleNamespace(
        register_job=lambda name, **kw: wired.__setitem__("job", (name, kw)))
    report = hooks.install_hooks(fake_video_hooks, fake_video_jobs)
    assert report == {"video_hooks": True, "prompt_coordinator": "set_prompt_coordinator",
                      "performance_runner": "video_performance"}
    assert isinstance(wired["coordinator"], hooks.OraclePromptCoordinator)
    from hugpy_oracle.relay import performance_relay
    name, kw = wired["job"]
    assert name == performance_relay.JOB_NAME == "video_performance"
    assert kw["runner_key"] == performance_relay.RUNNER_KEY == ("oracle", "performance")
    assert kw["runner"] is performance_relay.run_video_performance
    assert kw["spec_type"] is performance_relay.PerformanceSpec
    assert kw["from_dict"] is performance_relay.performance_from_dict
    assert kw["queue"] == "gpu" and kw["timeout_s"] == 14400


def test_install_hooks_against_the_real_video_seams():
    """The live contract: ``hugpy_video.hooks`` takes the coordinator and
    ``hugpy_video.jobs.register_job`` fills the bus tables."""
    pytest.importorskip("hugpy_video.hooks")
    from hugpy_video import hooks as vhooks, jobs as vjobs
    from hugpy_oracle.relay import performance_relay
    try:
        report = hooks.install_hooks()
        assert report["prompt_coordinator"] == "set_prompt_coordinator"
        assert report["performance_runner"] == "video_performance"
        assert isinstance(vhooks.get_prompt_coordinator(), hooks.OraclePromptCoordinator)
        assert isinstance(vhooks.get_prompt_coordinator(), vhooks.PromptCoordinator)
        spec = vjobs.registered_jobs()["video_performance"]
        assert spec.runner_key == performance_relay.RUNNER_KEY
        from hugpy_video.intel.runners import DISPATCH
        assert DISPATCH[performance_relay.RUNNER_KEY] is performance_relay.run_video_performance
    finally:
        vjobs.unregister_job("video_performance")
        vhooks.reset_hooks()


def test_the_coordinator_reviews_and_applies_like_prompt_spread_did():
    from hugpy_oracle.relay import prompt_coordination as PC

    coord = hooks.OraclePromptCoordinator()
    rows = [
        {"segment_id": "s1", "prompt": "A woman walks into a bar.", "joint_mode": "cut"},
        {"segment_id": "s2", "prompt": "Continuous: she orders a drink.", "joint_mode": "cut"},
    ]
    report = coord.review(rows, notes="")
    assert isinstance(report, PC.CoordinationReport)
    applied_rows, applied = coord.apply_decisions(rows, report)
    assert [r["segment_id"] for r in applied_rows] == ["s1", "s2"]
    assert coord.block_confidence == PC.BLOCK_CONFIDENCE
    assert isinstance(coord.blocking_mismatches(report), list)
    rows2, applied2, as_dict = coord.coordinate(rows)
    assert as_dict == report.as_dict()
    assert len(applied2) == len(applied)


def test_package_level_install_hooks_delegates():
    import hugpy_oracle
    fake = types.SimpleNamespace(set_prompt_coordinator=lambda impl: None)
    assert hugpy_oracle.install_hooks(fake)["prompt_coordinator"] == "set_prompt_coordinator"
