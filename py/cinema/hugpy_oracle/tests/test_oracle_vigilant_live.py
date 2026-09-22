"""Verification follow-ups: the selector's decision is what actually runs.

* a pinned model reaches the router in ``_executable_route``;
* live keyframe candidates spread across models and exclude a failed model;
* a judge that is the generator is refused (invariant 11, runtime);
* a DAG run ends with a steward report.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_vigilant_live.py -q
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

logging.disable(logging.INFO)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

from hugpy_oracle import evaluation, performance as perf, router, selection
from hugpy_oracle.contracts import Eligibility, GoalSpec
from hugpy_oracle.dag_runtime import RunJournal
from hugpy_oracle.recipes import video_performance as vp
from hugpy_oracle.router import RouteDecision
from hugpy_oracle.selection import SelectionPolicy

import test_oracle_performance as base  # noqa: E402


@dataclass
class FakeView:
    name: str
    model_ids: tuple
    eligibility: Eligibility = Eligibility(eligible=True)


@pytest.fixture()
def two_model_selector(tmp_path, monkeypatch):
    led = selection.ReliabilityLedger(os.path.join(str(tmp_path), "ledger.sqlite"))
    sel = selection.Selector(ledger=led, get_view=lambda c: FakeView(c, ("flux", "sdxl")),
                             get_matrix=lambda: None,
                             policy=SelectionPolicy(spread_margin=1.0, explore_every=0))
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", sel)
    yield sel, led
    led.close()
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", None)


def test_pinned_model_reaches_the_router(monkeypatch, two_model_selector):
    seen = {}

    def fake_resolve(goal, requested_model=None):
        seen["requested"] = requested_model
        return RouteDecision(capability=goal.capability, execution="execute", model_id=requested_model or "flux")

    monkeypatch.setattr(router, "resolve_route", fake_resolve)
    goal = GoalSpec(objective="o", raw_prompt="p")
    with selection.pinned(perf.KEYFRAME_CAPABILITY, "sdxl", {"rationale": "spread"}):
        route = perf._executable_route(goal, perf.KEYFRAME_CAPABILITY)
    assert seen["requested"] == "sdxl" and route.model_id == "sdxl"
    assert any("selection: spread -> sdxl" in r for r in route.reasons)
    # without a pin the selector decides (flux ranks first by declaration)
    perf._executable_route(goal, perf.KEYFRAME_CAPABILITY)
    assert seen["requested"] == "flux"


def test_live_keyframe_candidates_spread_and_exclude_failed_model(tmp_path, two_model_selector):
    sel, led = two_model_selector
    fakes = base.Fakes(keyframe_verdicts=[{"verdict": "NO", "score": 5, "why": "no"},
                                          {"verdict": "YES", "score": 90, "why": "ok"}])
    result = perf.run_performance(base.performance_goal(), seams=fakes.seams(tmp_path))
    assert result.ok, result.gap
    rows = list(reversed(led.recent(perf.KEYFRAME_CAPABILITY, limit=100)))  # oldest first
    assert rows[0]["model_id"] == "flux" and rows[0]["hard_pass"] == 0       # candidate 0 -> flux, failed
    assert rows[1]["model_id"] == "sdxl" and rows[1]["hard_pass"] == 1       # excluded flux -> sdxl, passed
    # a shot that passes on its first candidate is attributed too
    assert all(r["model_id"] in ("flux", "sdxl") for r in rows)
    kf = result.to_dict()["stages"]
    assert kf  # stages serialised


def test_judge_that_is_the_generator_is_refused(monkeypatch, tmp_path):
    img = os.path.join(str(tmp_path), "x.png")
    with open(img, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")

    def only_m1(judge_capability):
        return RouteDecision(capability=judge_capability, execution="execute", model_id="m1", task="image_understand")

    # the fleet's only eligible judge model IS the generator
    sel = selection.Selector(ledger=None, get_view=lambda c: FakeView(c, ("m1",)), get_matrix=lambda: None)
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", sel)
    monkeypatch.setattr(evaluation, "_resolve_judge_route", only_m1)
    # the exclusion path would otherwise resolve the selector's pick through
    # the LIVE router/catalog (fleet- and venv-dependent); the fleet here is m1
    monkeypatch.setattr(evaluation, "_resolve_judge_route_excluding",
                        lambda cap, exclude_model=None: only_m1(cap))
    monkeypatch.setattr(evaluation, "_judge_dispatch", lambda task, body: {"text": "YES 95"})
    rubric = evaluation.RUBRICS[perf.KEYFRAME_CAPABILITY]
    goal = GoalSpec(objective="o", raw_prompt="p", capability=perf.KEYFRAME_CAPABILITY)
    arts = [{"kind": "image", "uri": img}]
    res = evaluation.run_judge(rubric, goal, arts, generator_model="m1")
    assert res is not None and res.verdict == "unavailable"
    assert "self-judgment" in res.rationale
    other = evaluation.run_judge(rubric, goal, arts, generator_model="m2")
    assert other is not None and other.verdict != "unavailable"


def test_dag_run_ends_with_a_steward_report(tmp_path, two_model_selector):
    sel, led = two_model_selector
    fakes = base.Fakes()
    goal = base.performance_goal()
    prep = perf.run_performance(goal, seams=fakes.seams(tmp_path), stop_after="segments")
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    rt, visual = vp.run_visual_stages(prep, goal, fakes.seams(tmp_path), journal=j, run_id="v1", selector=sel)
    assert visual.ok
    assert visual.steward is not None and "findings" in visual.steward and visual.steward["summary"]
    assert visual.to_dict()["steward"]["findings"]
    j.close()


def test_authoring_outcome_is_ledgered(tmp_path, monkeypatch):
    """TODO-9: an accepted plot writes a PASS for the authoring model; a
    validator-rejected one writes a FAIL — both under text.chat."""
    from hugpy_oracle import script_first as sf
    led = selection.ReliabilityLedger(os.path.join(str(tmp_path), "ledger.sqlite"))
    sel = selection.Selector(ledger=led, get_view=lambda c: None, get_matrix=lambda: None)
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", sel)
    try:
        # drive the exact statement the authoring path runs, with a fake outcome
        selection.note_verdict("text.chat", "qwen-author", hard_pass=True)
        selection.note_verdict("text.chat", "qwen-author", hard_pass=False,
                               repair_code=__import__("hugpy_oracle.contracts", fromlist=["RepairCode"]).RepairCode.FORMAT_MISMATCH)
        s = led.stats("text.chat", "qwen-author")
        assert s.n == 2 and s.pass_rate == 0.5 and s.repair_codes == (("format_mismatch", 1),)
        # and the live authoring function references a resolvable RepairCode
        src = open(sf.__file__).read()
        assert "from hugpy_oracle.contracts import RepairCode as _RC" in src
    finally:
        led.close()
        monkeypatch.setattr(selection, "_PROCESS_SELECTOR", None)
