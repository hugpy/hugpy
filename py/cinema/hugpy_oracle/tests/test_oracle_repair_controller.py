"""k112 — repair controller: repair code -> smallest invalid subgraph.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_repair_controller.py -q
"""
from __future__ import annotations

import logging
import os
import sys

logging.disable(logging.INFO)
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle.contracts import RepairCode, Scorecard
from hugpy_oracle.dag_runtime import DagRuntime, NodeState, RunJournal, RunState
from hugpy_oracle.plan import Edge, NodeKind, PlanGraph, PlanNode, Port, RetryPolicy
from hugpy_oracle.repair_controller import POLICY, RepairController, repair_path, responsible_root


def _n(nid, cap, *, inputs=(), depends=(), params=None, retry=1):
    return PlanNode(
        node_id=nid, kind=NodeKind.TASK, capability=cap,
        inputs=tuple(Port(i, "any") for i in inputs), outputs=(Port("out", "any"),),
        depends_on=tuple(depends), params=params or {}, retry=RetryPolicy(max_attempts=retry),
    )


def video_graph(revision: int = 0) -> PlanGraph:
    """transcribe -> tts -> assemble ; identity -> kf1 -> clip1 -> assemble
                                        identity -> kf2 -> clip2 -> assemble
       (clips also consume tts for their audio window)."""
    nodes = (
        _n("transcribe", "audio.transcribe.word_timestamps"),
        _n("tts", "voice.synthesize.reference_conditioned", inputs=("lines",), depends=("transcribe",)),
        _n("identity", "image.identity_reference_pack"),
        _n("kf1", "image.keyframe", inputs=("identity",), depends=("identity",), params={"seed": 10}),
        _n("kf2", "image.keyframe", inputs=("identity",), depends=("identity",), params={"seed": 20}),
        _n("clip1", "video.generate.identity_conditioned", inputs=("kf", "audio"),
           depends=("kf1", "tts"), params={"geometry_strength": 0.7, "seed": 1}),
        _n("clip2", "video.generate.identity_conditioned", inputs=("kf", "audio"),
           depends=("kf2", "tts"), params={"geometry_strength": 0.7, "seed": 2}),
        _n("assemble", "media.assemble.timeline", inputs=("a", "b", "audio"),
           depends=("clip1", "clip2", "tts")),
    )
    edges = (
        Edge("transcribe", "out", "tts", "lines"),
        Edge("identity", "out", "kf1", "identity"), Edge("identity", "out", "kf2", "identity"),
        Edge("kf1", "out", "clip1", "kf"), Edge("tts", "out", "clip1", "audio"),
        Edge("kf2", "out", "clip2", "kf"), Edge("tts", "out", "clip2", "audio"),
        Edge("clip1", "out", "assemble", "a"), Edge("clip2", "out", "assemble", "b"),
        Edge("tts", "out", "assemble", "audio"),
    )
    return PlanGraph(graph_id="vid", goal_digest="g", revision=revision, nodes=nodes, edges=edges)


class Recorder:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, node, inputs, ctx):
        self.calls.append(node.node_id)
        return {"out": f"{node.node_id}@{ctx.attempt}:{dict(node.params)}"}

    def count(self, nid):
        return self.calls.count(nid)


class FailOnce:
    """Evaluator that fails ``node`` with ``code`` exactly once."""
    def __init__(self, node: str, code: RepairCode):
        self.node, self.code, self.fired = node, code, False

    def __call__(self, node, outputs, ctx):
        if node.node_id == self.node and not self.fired:
            self.fired = True
            return Scorecard(hard_pass=False, diagnosis=f"{self.code.value} on {node.node_id}",
                             repair_code=self.code)
        return Scorecard(hard_pass=True)


@pytest.fixture()
def journal(tmp_path):
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    yield j
    j.close()


# --------------------------------------------------------------------------- #
# pure policy
# --------------------------------------------------------------------------- #


def test_policy_covers_every_repair_code():
    assert set(POLICY) == set(RepairCode)


def test_identity_drift_root_is_keyframe_not_audio():
    g = video_graph()
    pol = POLICY[RepairCode.IDENTITY_DRIFT]
    assert responsible_root(g, "clip1", pol) == "kf1"
    assert repair_path(g, "kf1", "clip1", pol) == ("kf1", "clip1")


def test_temporal_artifact_root_is_the_clip_itself():
    g = video_graph()
    pol = POLICY[RepairCode.TEMPORAL_ARTIFACT]
    assert responsible_root(g, "clip1", pol) == "clip1"
    assert repair_path(g, "clip1", "clip1", pol) == ("clip1",)


def test_voice_similarity_root_is_tts_and_never_picture():
    g = video_graph()
    pol = POLICY[RepairCode.VOICE_SIMILARITY_LOW]
    assert responsible_root(g, "assemble", pol) == "tts"
    path = repair_path(g, "tts", "assemble", pol)
    assert "tts" in path and "assemble" in path
    assert not any(n.startswith(("kf", "clip", "identity")) for n in path)


def test_authority_missing_is_not_repairable():
    g = video_graph()
    assert responsible_root(g, "clip1", POLICY[RepairCode.SOURCE_AUTHORITY_MISSING]) is None


# --------------------------------------------------------------------------- #
# end to end against the runtime
# --------------------------------------------------------------------------- #


def _run_until_failure(journal, evaluator, retry=1):
    ex = Recorder()
    rt = DagRuntime(journal, ex, evaluator=evaluator, owner="t")
    rt.start(video_graph(), "r1")
    final = rt.run("r1")
    assert final.state is RunState.FAILED
    return rt, ex


def test_identity_drift_repairs_keyframe_and_clip_only(journal):
    rt, ex = _run_until_failure(journal, FailOnce("clip1", RepairCode.IDENTITY_DRIFT))
    # sibling clip2 was produced even though clip1 failed (siblings are independent)
    assert journal.node("r1", "clip2").state is NodeState.SUCCEEDED
    before = dict((n, ex.count(n)) for n in ("transcribe", "tts", "identity", "kf2", "clip2"))
    ctl = RepairController(rt)
    plan = ctl.diagnose("r1", "clip1")
    assert plan.strategy == "retry_nodes" and plan.root == "kf1"
    assert plan.path == ("kf1", "clip1") and plan.repairable
    reset = ctl.apply(plan)
    assert set(reset) == {"kf1", "clip1"}
    assert journal.node("r1", "kf1").state is NodeState.PENDING
    assert journal.node("r1", "tts").state is NodeState.SUCCEEDED
    final = rt.run("r1")
    assert final.state is RunState.COMPLETED
    after = dict((n, ex.count(n)) for n in before)
    assert after == before, "unrelated accepted nodes were re-executed"
    assert ex.count("kf1") == 2 and ex.count("clip1") == 2
    assert journal.run("r1").revision == 0  # no graph change for a reseed-style repair
    kinds = [c["kind"] for c in journal.controls("r1")]
    assert kinds == ["repair", "repair_retry"]


def test_geometry_drift_replans_with_stronger_geometry(journal):
    rt, ex = _run_until_failure(journal, FailOnce("clip1", RepairCode.GEOMETRY_DRIFT))
    ctl = RepairController(rt)
    plan = ctl.diagnose("r1", "clip1")
    assert plan.strategy == "replan" and plan.root == "clip1"
    assert plan.param_changes == {"geometry_strength": "+0.1"}
    ctl.apply(plan)
    assert journal.run("r1").revision == 1
    g1 = journal.graph("r1")
    assert g1.node("clip1").params["geometry_strength"] == pytest.approx(0.8)
    assert g1.node("clip1").params["seed"] == 2  # bumped
    assert g1.node("clip2").params["geometry_strength"] == 0.7  # untouched
    final = rt.run("r1")
    assert final.state is RunState.COMPLETED
    assert ex.count("clip2") == 1 and ex.count("kf1") == 1 and ex.count("clip1") == 2
    # assemble is downstream of clip1 so it legitimately re-ran (its input changed)
    assert ex.count("assemble") == 1  # never succeeded the first time


def test_repair_budget_exhaustion_is_reported_not_silent(journal):
    ex = Recorder()
    rt = DagRuntime(journal, ex, evaluator=FailOnce("clip1", RepairCode.SHOT_TOO_SHORT), owner="t")
    rt.start(video_graph(), "r1", repair_budget=0)
    assert rt.run("r1").state is RunState.FAILED
    plan = RepairController(rt).diagnose("r1", "clip1")
    assert plan.strategy == "none" and not plan.repairable
    assert "budget exhausted" in plan.rationale
    assert RepairController(rt).apply(plan) == ()


def test_authority_failure_is_never_regenerated(journal):
    rt, ex = _run_until_failure(journal, FailOnce("identity", RepairCode.SOURCE_AUTHORITY_MISSING))
    plan = RepairController(rt).diagnose("r1", "identity")
    assert plan.strategy == "none" and not plan.repairable
    assert RepairController(rt).apply(plan) == ()
    assert journal.run("r1").state is RunState.FAILED
    assert ex.count("identity") == 1


def test_diagnose_on_succeeded_node_is_a_noop(journal):
    ex = Recorder()
    rt = DagRuntime(journal, ex, owner="t")
    rt.start(video_graph(), "r1")
    rt.run("r1")
    plan = RepairController(rt).diagnose("r1", "clip1")
    assert plan.strategy == "none" and not plan.repairable


def test_repair_plan_round_trips_json(journal):
    import json
    rt, _ = _run_until_failure(journal, FailOnce("clip2", RepairCode.ACTION_MISSING))
    plan = RepairController(rt).diagnose("r1", "clip2")
    d = json.loads(json.dumps(plan.to_dict()))
    assert d["code"] == "action_missing" and d["path"] == ["clip2"]
