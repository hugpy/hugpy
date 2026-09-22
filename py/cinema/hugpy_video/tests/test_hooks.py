"""hugpy_video.hooks: the PromptCoordinator seam the oracle plugs into.

The default must be an honest no-op (nothing reviewed, nothing applied,
nothing blocks) and both call sites (spread coordination, movie preflight)
must route through whatever is installed.
"""
from __future__ import annotations

import pytest

from hugpy_video import hooks


@pytest.fixture(autouse=True)
def _clean_hooks():
    hooks.reset_hooks()
    yield
    hooks.reset_hooks()


def test_default_is_the_null_coordinator():
    pc = hooks.get_prompt_coordinator()
    assert isinstance(pc, hooks.NullPromptCoordinator)
    assert isinstance(pc, hooks.PromptCoordinator)          # runtime-checkable
    rows = [{"segment_id": "s0", "prompt": "a", "locked": False}]
    report = pc.review(rows, notes="n", context={}, llm=None)
    d = report.as_dict()
    assert d["reviewed"] is False and d["decisions"] == [] and d["segments"] == []
    applied_rows, applied = pc.apply_decisions(rows, report)
    assert applied_rows == rows and applied_rows is not rows and applied == []
    assert pc.blocking_mismatches(report) == []
    assert pc.blocking_mismatches(pc.review_goals(object()), 0.1) == []


def test_installed_coordinator_is_used_by_movie_plan_and_spread():
    calls = []

    class _Report:
        decisions = ()
        def as_dict(self):
            return {"reviewed": True, "coordinator": "fake"}

    class _Decision:
        def __init__(self, sid, knob):
            self.segment_id, self.knob = sid, knob

    class Fake:
        block_confidence = 0.5
        def review(self, rows, *, notes="", context=None, llm=None):
            calls.append(("review", [r["segment_id"] for r in rows], notes))
            return _Report()
        def review_goals(self, spec, *, context=None):
            calls.append(("review_goals", spec))
            return _Report()
        def apply_decisions(self, rows, report):
            out = [dict(r) for r in rows]
            out[0]["seed"] = 7
            return out, [_Decision(out[0]["segment_id"], "seed")]
        def blocking_mismatches(self, report, threshold=None):
            calls.append(("blocking", threshold))
            return [{"index": 0, "segment_id": "s0", "reason": "mismatch", "detail": "x"}]

    hooks.set_prompt_coordinator(Fake())
    from hugpy_video.intel.studio import movie_plan
    assert movie_plan.coordination_report("SPEC") == {"reviewed": True, "coordinator": "fake"}
    assert movie_plan.preflight_coordination("SPEC")[0]["segment_id"] == "s0"
    assert ("blocking", 0.5) in calls                       # block_confidence honoured

    from hugpy_video.intel import prompt_spread as PS
    req = PS.SpreadRequest.__new__(PS.SpreadRequest)       # only the fields coordinate_spread reads
    object.__setattr__(req, "fixed_segments", ())
    object.__setattr__(req, "target_segments", ({"segment_id": "s0", "prompt": "old", "locked": False},))
    object.__setattr__(req, "context", {})
    object.__setattr__(req, "hint", "hint!")
    object.__setattr__(req, "steering_seed", None)
    parsed = {"segments": [{"segment_id": "s0", "prompt": "new"}]}
    out = PS.coordinate_spread(req, parsed)
    assert out["coordination"] == {"reviewed": True, "coordinator": "fake"}
    assert out["segments"][0]["knobs"] == {"seed": 7}
    assert ("review", ["s0"], "hint!") in calls

    hooks.set_prompt_coordinator(None)
    assert isinstance(hooks.get_prompt_coordinator(), hooks.NullPromptCoordinator)
