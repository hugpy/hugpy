"""k113a — per-call model selection + reliability ledger + runtime loop.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_selection.py -q
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

logging.disable(logging.INFO)
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle.contracts import (
    BudgetHints,
    Eligibility,
    GoalSpec,
    QualityProfile,
    RepairCode,
    Scorecard,
)
from hugpy_oracle.dag_runtime import DagRuntime, NodeState, RunJournal, RunState
from hugpy_oracle.plan import NodeKind, PlanGraph, PlanNode, Port, ResourceRequest, RetryPolicy
from hugpy_oracle.selection import ReliabilityLedger, SelectionPolicy, Selector, select


# --------------------------------------------------------------------------- #
# fakes for the catalog / matrix seams
# --------------------------------------------------------------------------- #


@dataclass
class FakeView:
    name: str
    model_ids: tuple
    eligibility: Eligibility = Eligibility(eligible=True)


@dataclass
class FakeCand:
    model: str
    ok_rate: float = 1.0
    quality: float = 0.5
    latency_s: float | None = 5.0


@dataclass
class FakeEntry:
    operation: str
    primary: str | None
    fallback: str | None
    candidates: tuple


class FakeMatrix:
    def __init__(self, *entries):
        self._e = {e.operation: e for e in entries}

    def entry(self, op):
        return self._e.get(op)


def goal(quality=QualityProfile.BALANCED, max_seconds=None, max_vram=None):
    return GoalSpec(objective="o", raw_prompt="p", quality=quality,
                    budget=BudgetHints(max_seconds=max_seconds, max_vram_gb=max_vram))


VIEW = FakeView("image.generate", ("flux", "sdxl", "tiny"))
MATRIX = FakeMatrix(FakeEntry("image.generate", "flux", "sdxl", (
    FakeCand("flux", 1.0, 0.9, 20.0), FakeCand("sdxl", 0.95, 0.7, 8.0), FakeCand("tiny", 1.0, 0.3, 1.0))))


# --------------------------------------------------------------------------- #
# pure selection
# --------------------------------------------------------------------------- #


def test_unknown_capability_is_a_gap_not_a_default():
    d = select("nope", view=None)
    assert d.gap and d.model_id is None and d.rationale == "capability_gap"


def test_ineligible_view_is_a_gap_with_reasons():
    v = FakeView("x", ("m",), Eligibility(eligible=False, reasons=("no worker seats x",)))
    d = select("x", view=v)
    assert d.gap and "no worker seats x" in d.steps[0]


def test_matrix_quality_wins_under_best_profile():
    d = select("image.generate", view=VIEW, matrix=MATRIX, goal=goal(QualityProfile.BEST))
    assert d.model_id == "flux"
    assert [v.model_id for v in d.ranked] == ["flux", "sdxl", "tiny"]
    assert any("matrix primary" in r for r in d.ranked[0].reasons)


def test_preview_profile_tilts_to_speed():
    d = select("image.generate", view=VIEW, matrix=MATRIX, goal=goal(QualityProfile.PREVIEW))
    assert d.model_id == "tiny"


def test_latency_budget_rejects_slow_model_with_reason():
    d = select("image.generate", view=VIEW, matrix=MATRIX, goal=goal(max_seconds=10))
    assert d.model_id == "sdxl"
    rej = {v.model_id: v for v in d.rejected}
    assert rej["flux"].rejected_at == "7.latency"


def test_vram_cap_rejects_model_that_does_not_fit():
    vram = {"flux": 24.0, "sdxl": 10.0, "tiny": 2.0}
    d = select("image.generate", view=VIEW, matrix=MATRIX, goal=goal(max_vram=12),
               model_vram_gib=vram.get)
    assert d.model_id == "sdxl"
    assert {v.model_id for v in d.rejected} == {"flux"}
    assert d.rejected[0].rejected_at == "5.resources"


def test_node_resource_request_also_caps_vram():
    node = PlanNode(node_id="kf", kind=NodeKind.TASK, capability="image.generate",
                    resources=ResourceRequest(vram_gib=4))
    d = select("image.generate", view=VIEW, matrix=MATRIX, node=node,
               model_vram_gib={"flux": 24.0, "sdxl": 10.0, "tiny": 2.0}.get)
    assert d.model_id == "tiny"


def test_health_probe_rejects():
    d = select("image.generate", view=VIEW, matrix=MATRIX, model_health=lambda m: m != "flux")
    assert d.model_id == "sdxl"
    assert d.rejected[0].rejected_at == "4.health"


def test_excluded_models_are_skipped():
    d = select("image.generate", view=VIEW, matrix=MATRIX, exclude=("flux",))
    assert d.model_id == "sdxl"


def test_all_excluded_is_a_gap():
    d = select("image.generate", view=VIEW, matrix=MATRIX, exclude=("flux", "sdxl", "tiny"))
    assert d.gap


def test_candidate_spread_sends_siblings_to_distinct_models():
    pol = SelectionPolicy(spread_margin=0.5)
    picks = [select("image.generate", view=VIEW, matrix=MATRIX, policy=pol,
                    candidate_index=i, candidates=3).model_id for i in range(3)]
    assert picks == ["flux", "sdxl", "tiny"]
    assert select("image.generate", view=VIEW, matrix=MATRIX, policy=pol,
                  candidate_index=1, candidates=3).spread is True


def test_spread_does_not_reach_models_far_below_best():
    pol = SelectionPolicy(spread_margin=0.01)
    picks = {select("image.generate", view=VIEW, matrix=MATRIX, policy=pol,
                    candidate_index=i, candidates=3).model_id for i in range(3)}
    assert picks == {"flux"}


def test_decision_serializes():
    import json
    d = select("image.generate", view=VIEW, matrix=MATRIX)
    j = json.loads(json.dumps(d.to_dict()))
    assert j["model_id"] == "flux" and j["fallback"] == "sdxl" and len(j["steps"]) >= 9


# --------------------------------------------------------------------------- #
# reliability ledger changes the answer
# --------------------------------------------------------------------------- #


@pytest.fixture()
def ledger(tmp_path):
    l = ReliabilityLedger(os.path.join(str(tmp_path), "ledger.sqlite"))
    yield l
    l.close()


def test_measured_failures_demote_the_matrix_primary(ledger):
    for _ in range(5):
        ledger.record("image.generate", "flux", ok=True, hard_pass=False,
                      repair_code=RepairCode.IDENTITY_DRIFT, latency_s=20)
    for _ in range(5):
        ledger.record("image.generate", "sdxl", ok=True, hard_pass=True, latency_s=8)
    d = select("image.generate", view=VIEW, matrix=MATRIX, ledger=ledger)
    assert d.model_id == "sdxl"
    rej = {v.model_id: v for v in d.rejected}
    assert "flux" in rej and rej["flux"].rejected_at == "8.reliability"
    assert rej["flux"].evidence["repair_codes"][0][0] == "identity_drift"


def test_thin_evidence_is_advisory_not_decisive(ledger):
    ledger.record("image.generate", "flux", ok=True, hard_pass=False, repair_code=RepairCode.IDENTITY_DRIFT)
    d = select("image.generate", view=VIEW, matrix=MATRIX, ledger=ledger,
               policy=SelectionPolicy(min_samples=3))
    assert d.model_id == "flux"  # one bad take does not dethrone the primary
    assert any("ledger: pass=0.00" in r for r in d.ranked[0].reasons)


def test_all_models_measured_bad_is_reported_not_defaulted(ledger):
    for m in ("flux", "sdxl", "tiny"):
        for _ in range(4):
            ledger.record("image.generate", m, ok=False)
    d = select("image.generate", view=VIEW, matrix=MATRIX, ledger=ledger)
    assert d.gap and d.rationale == "all_models_below_reliability_floor"
    assert len(d.rejected) == 3


def test_ledger_stats_window_and_counts(ledger):
    for i in range(60):
        ledger.record("audio.tts", "cb", ok=i % 2 == 0, hard_pass=True, latency_s=1.0)
    s = ledger.stats("audio.tts", "cb", window=50)
    assert s.n == 50 and 0.45 <= s.ok_rate <= 0.55 and s.mean_latency_s == 1.0
    assert ledger.count() == 60


# --------------------------------------------------------------------------- #
# closed loop through the DAG runtime
# --------------------------------------------------------------------------- #


def _graph(candidates=1, kind=NodeKind.TASK, retry=1):
    kf = PlanNode(node_id="kf", kind=kind, capability="image.generate",
                  outputs=(Port("out", "image"),), candidates=candidates,
                  retry=RetryPolicy(max_attempts=retry))
    return PlanGraph(graph_id="g", goal_digest="d", nodes=(kf,))


def _selector(ledger, **kw):
    return Selector(ledger=ledger, get_view=lambda cap: VIEW, get_matrix=lambda: MATRIX, **kw)


def test_runtime_routes_each_call_and_records_outcomes(tmp_path, ledger):
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    seen = []

    def ex(node, inputs, ctx):
        seen.append(ctx.model_id)
        return {"out": f"img-by-{ctx.model_id}"}

    rt = DagRuntime(j, ex, owner="t", selector=_selector(ledger),
                    evaluator=lambda n, o, c: Scorecard(hard_pass=True))
    rt.start(_graph(), "r1")
    assert rt.run("r1").state is RunState.COMPLETED
    assert seen == ["flux"]
    rec = j.node("r1", "kf")
    assert rec.receipt["model_id"] == "flux"
    assert rec.receipt["selection"]["rationale"] == "evidence-ranked"
    assert rec.receipt["selection"]["ranked"][0]["selected"] is True
    s = ledger.stats("image.generate", "flux")
    assert s.n == 1 and s.pass_rate == 1.0
    j.close()


def test_fanout_spreads_candidates_across_models_and_ledgers_each(tmp_path, ledger):
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    seen = []

    def ex(node, inputs, ctx):
        seen.append((ctx.candidate, ctx.model_id))
        return {"out": ctx.model_id}

    sel = _selector(ledger, policy=SelectionPolicy(spread_margin=0.5))
    rt = DagRuntime(j, ex, owner="t", selector=sel,
                    evaluator=lambda n, o, c: Scorecard(hard_pass=True))
    rt.start(_graph(candidates=3, kind=NodeKind.FANOUT), "r1")
    assert rt.run("r1").state is RunState.COMPLETED
    assert seen == [(0, "flux"), (1, "sdxl"), (2, "tiny")]
    rec = j.node("r1", "kf")
    assert rec.receipt["models"] == ["flux", "sdxl", "tiny"]
    assert {m: ledger.stats("image.generate", m).n for m in ("flux", "sdxl", "tiny")} == {"flux": 1, "sdxl": 1, "tiny": 1}
    j.close()


def test_failed_attempt_excludes_that_model_on_retry(tmp_path, ledger):
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    seen = []

    def ex(node, inputs, ctx):
        seen.append(ctx.model_id)
        if ctx.model_id == "flux":
            raise RuntimeError("flux OOM")
        return {"out": ctx.model_id}

    rt = DagRuntime(j, ex, owner="t", selector=_selector(ledger))
    rt.start(_graph(retry=2), "r1")
    assert rt.run("r1").state is RunState.COMPLETED
    assert seen == ["flux", "sdxl"]
    assert j.node("r1", "kf").receipt["model_id"] == "sdxl"
    assert ledger.stats("image.generate", "flux").ok_rate == 0.0
    j.close()


def test_judge_failure_is_written_back_as_evidence(tmp_path, ledger):
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    rt = DagRuntime(j, lambda n, i, c: {"out": c.model_id}, owner="t", selector=_selector(ledger),
                    evaluator=lambda n, o, c: Scorecard(hard_pass=False, diagnosis="drift",
                                                        repair_code=RepairCode.IDENTITY_DRIFT))
    rt.start(_graph(), "r1")
    assert rt.run("r1").state is RunState.FAILED
    s = ledger.stats("image.generate", "flux")
    assert s.n == 1 and s.pass_rate == 0.0 and s.repair_codes == (("identity_drift", 1),)
    j.close()


def test_selection_gap_is_a_typed_failure_on_the_node(tmp_path, ledger):
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    empty = Selector(ledger=ledger, get_view=lambda cap: FakeView(cap, ()), get_matrix=lambda: None)
    rt = DagRuntime(j, lambda n, i, c: {"out": 1}, owner="t", selector=empty)
    rt.start(_graph(), "r1")
    assert rt.run("r1").state is RunState.FAILED
    rec = j.node("r1", "kf")
    assert rec.state is NodeState.FAILED
    assert rec.repair_code is RepairCode.CAPABILITY_GAP
    assert "no selectable model" in rec.failure
    j.close()
