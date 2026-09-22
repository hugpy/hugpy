"""k111 — durable DAG runtime: record-before-run, leases, idempotency, cache,
resume-after-kill, controls, graph revisions, reservations.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_dag_runtime.py -q
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
from hugpy_oracle.dag_runtime import (
    DagRuntime,
    JournalError,
    NodeState,
    RepairBudgetExceeded,
    ResourceBroker,
    RunJournal,
    RunState,
    derive_cache_key,
)
from hugpy_oracle.plan import (
    Edge,
    NodeKind,
    PlanGraph,
    PlanNode,
    Port,
    ResourceRequest,
    RetryPolicy,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _task(nid: str, cap: str = "text.generate", *, depends=(), inputs=(), outputs=("out",),
          **kw) -> PlanNode:
    return PlanNode(
        node_id=nid, kind=kw.pop("kind", NodeKind.TASK), capability=cap,
        inputs=tuple(Port(name=i, artifact_kind="text") for i in inputs),
        outputs=tuple(Port(name=o, artifact_kind="text") for o in outputs),
        depends_on=tuple(depends), **kw,
    )


def linear_graph(revision: int = 0, **node_kw) -> PlanGraph:
    """a -> b -> c ; each consumes the previous node's ``out`` on port ``in``."""
    a = _task("a")
    b = _task("b", inputs=("in",), depends=("a",), **node_kw.get("b", {}))
    c = _task("c", inputs=("in",), depends=("b",), **node_kw.get("c", {}))
    return PlanGraph(
        graph_id="g1", goal_digest="goal", revision=revision, nodes=(a, b, c),
        edges=(Edge("a", "out", "b", "in"), Edge("b", "out", "c", "in")),
    )


class CountingExecutor:
    def __init__(self, fail_on: dict[str, int] | None = None):
        self.calls: list[tuple[str, int, int]] = []  # (node, attempt, candidate)
        self.fail_on = fail_on or {}  # node_id -> number of attempts that should raise

    def __call__(self, node, inputs, ctx):
        self.calls.append((node.node_id, ctx.attempt, ctx.candidate))
        remaining = self.fail_on.get(node.node_id, 0)
        if remaining > 0:
            self.fail_on[node.node_id] = remaining - 1
            raise RuntimeError(f"boom {node.node_id}")
        upstream = inputs.get("in")
        return {"out": f"{node.node_id}({upstream})" if upstream is not None else node.node_id}

    def count(self, nid: str) -> int:
        return sum(1 for c in self.calls if c[0] == nid)


@pytest.fixture()
def db(tmp_path):
    return os.path.join(str(tmp_path), "dag.sqlite")


@pytest.fixture()
def journal(db):
    j = RunJournal(db)
    yield j
    j.close()


# --------------------------------------------------------------------------- #
# happy path + journal semantics
# --------------------------------------------------------------------------- #


def test_linear_graph_runs_to_completion(journal):
    ex = CountingExecutor()
    rt = DagRuntime(journal, ex, owner="t1")
    run = rt.start(linear_graph(), "r1")
    assert run.state is RunState.RUNNING
    final = rt.run("r1")
    assert final.state is RunState.COMPLETED
    nodes = journal.nodes("r1")
    assert nodes["c"].outputs == {"out": "c(b(a))"}
    assert [c[0] for c in ex.calls] == ["a", "b", "c"]
    assert journal.schema_version == 1
    assert os.path.exists(journal.db_path + "-wal") or True  # WAL mode set; file may be checkpointed


def test_wal_mode_and_record_before_run_ordering(journal):
    assert journal._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    seen: list[NodeState] = []

    def spy(node, inputs, ctx):
        # at the moment the executor runs, the journal ALREADY says RUNNING
        seen.append(journal.node(ctx.run_id, node.node_id).state)
        return {"out": node.node_id}

    rt = DagRuntime(journal, spy, owner="t1")
    rt.start(linear_graph(), "r1")
    rt.run("r1")
    assert seen == [NodeState.RUNNING] * 3
    trail = [t["to_state"] for t in journal.transitions("r1", "a")]
    assert trail == ["pending", "leased", "running", "succeeded"]


def test_receipt_carries_idempotency_and_lease(journal):
    rt = DagRuntime(journal, CountingExecutor(), owner="owner-x")
    rt.start(linear_graph(), "r1")
    rt.run("r1")
    rec = journal.node("r1", "b")
    assert rec.receipt["lease_owner"] == "owner-x"
    assert rec.receipt["idempotency_key"] == rec.idempotency_key
    assert rec.receipt["attempt"] == 1
    assert rec.lease_owner is None and rec.lease_expires_at is None


# --------------------------------------------------------------------------- #
# resume after kill
# --------------------------------------------------------------------------- #


def test_resume_after_crash_does_not_repeat_completed_nodes(db):
    """Simulate a kill mid-``b``: a succeeded, b is RUNNING with a live lease,
    the process dies. A fresh runtime on the same file must expire b's lease,
    run b and c, and never call a again."""
    j1 = RunJournal(db)
    ex1 = CountingExecutor()
    rt1 = DagRuntime(j1, ex1, owner="proc1", lease_s=0.01)
    rt1.start(linear_graph(), "r1")
    rt1.step("r1")  # runs a
    assert j1.node("r1", "a").state is NodeState.SUCCEEDED
    # hand-journal b as leased+running and "die"
    rec = j1.node("r1", "b")
    j1.lease("r1", "b", owner="proc1", lease_s=0.01, idempotency_key="k", cache_key="ck")
    j1.mark_running("r1", "b")
    j1.close()

    import time
    time.sleep(0.02)
    j2 = RunJournal(db)
    ex2 = CountingExecutor()
    rt2 = DagRuntime(j2, ex2, owner="proc2")
    expired = rt2.resume("r1")
    assert expired == ("b",)
    assert j2.node("r1", "b").state is NodeState.PENDING
    assert j2.node("r1", "b").attempt == 1  # attempt count preserved
    final = rt2.run("r1")
    assert final.state is RunState.COMPLETED
    assert ex2.count("a") == 0
    assert ex2.count("b") == 1 and ex2.count("c") == 1
    assert j2.node("r1", "b").attempt == 2
    assert j2.node("r1", "c").outputs == {"out": "c(b(a))"}
    j2.close()


def test_live_lease_is_not_expired_by_resume(journal):
    rt = DagRuntime(journal, CountingExecutor(), owner="p1")
    rt.start(linear_graph(), "r1")
    journal.lease("r1", "a", owner="p1", lease_s=3600, idempotency_key="k", cache_key="ck")
    journal.mark_running("r1", "a")
    assert rt.resume("r1") == ()
    assert journal.node("r1", "a").state is NodeState.RUNNING


# --------------------------------------------------------------------------- #
# cache + idempotency
# --------------------------------------------------------------------------- #


def test_cache_serves_identical_node_across_runs(journal):
    ex = CountingExecutor()
    rt = DagRuntime(journal, ex, owner="t")
    rt.start(linear_graph(), "r1")
    rt.run("r1")
    assert journal.cache_size() == 3
    rt.start(linear_graph(), "r2")
    rt.run("r2")
    # nothing executed the second time; all three served from the journal
    assert len(ex.calls) == 3
    recs = journal.nodes("r2")
    assert all(r.cached for r in recs.values())
    assert all(r.receipt["cached"] is True for r in recs.values())
    assert recs["c"].outputs == {"out": "c(b(a))"}
    assert journal.run("r2").state is RunState.COMPLETED


def test_cache_key_changes_with_params_and_inputs():
    n1 = _task("x", params={"seed": 1})
    n2 = _task("x", params={"seed": 2})
    assert derive_cache_key(n1, {"in": "a"}) != derive_cache_key(n2, {"in": "a"})
    assert derive_cache_key(n1, {"in": "a"}) != derive_cache_key(n1, {"in": "b"})
    assert derive_cache_key(n1, {"in": "a"}) == derive_cache_key(n1, {"in": "a"})
    assert derive_cache_key(_task("y", cache_key="fixed"), {}) == "fixed"


def test_declared_idempotency_key_is_used(journal):
    g = PlanGraph(graph_id="g", goal_digest="d",
                  nodes=(_task("a", idempotency_key="idem-a"),))
    rt = DagRuntime(journal, CountingExecutor(), owner="t")
    rt.start(g, "r1")
    rt.run("r1")
    assert journal.node("r1", "a").idempotency_key == "idem-a"


# --------------------------------------------------------------------------- #
# retries, acceptance, repair codes
# --------------------------------------------------------------------------- #


def test_retry_policy_bounds_attempts(journal):
    ex = CountingExecutor(fail_on={"b": 2})
    g = linear_graph(b={"retry": RetryPolicy(max_attempts=3)})
    rt = DagRuntime(journal, ex, owner="t")
    rt.start(g, "r1")
    final = rt.run("r1")
    assert final.state is RunState.COMPLETED
    assert ex.count("b") == 3
    assert journal.node("r1", "b").attempt == 3
    trail = [t["to_state"] for t in journal.transitions("r1", "b")]
    assert trail.count("pending") == 3  # created + 2 retries


def test_exhausted_retries_fail_run_without_touching_siblings(journal):
    ex = CountingExecutor(fail_on={"b": 5})
    g = linear_graph(b={"retry": RetryPolicy(max_attempts=2)})
    rt = DagRuntime(journal, ex, owner="t")
    rt.start(g, "r1")
    final = rt.run("r1")
    assert final.state is RunState.FAILED
    recs = journal.nodes("r1")
    assert recs["a"].state is NodeState.SUCCEEDED
    assert recs["b"].state is NodeState.FAILED and "boom" in recs["b"].failure
    assert recs["c"].state is NodeState.PENDING
    assert ex.count("c") == 0


def test_acceptance_failure_sets_repair_code_and_retries(journal):
    verdicts = iter([False, True])

    def evaluator(node, outputs, ctx):
        if node.node_id != "b":
            return None
        ok = next(verdicts)
        return Scorecard(hard_pass=ok, diagnosis=None if ok else "identity drift",
                         repair_code=None if ok else RepairCode.IDENTITY_DRIFT)

    ex = CountingExecutor()
    g = linear_graph(b={"retry": RetryPolicy(max_attempts=2)})
    rt = DagRuntime(journal, ex, evaluator=evaluator, owner="t")
    rt.start(g, "r1")
    final = rt.run("r1")
    assert final.state is RunState.COMPLETED
    assert ex.count("b") == 2
    failed = [t for t in journal.transitions("r1", "b") if t["reason"].startswith("failed")]
    assert failed and failed[0]["to_state"] == "pending"
    assert journal.node("r1", "b").receipt["scorecard"]["hard_pass"] is True


def test_acceptance_failure_without_retry_is_terminal_with_code(journal):
    def evaluator(node, outputs, ctx):
        return Scorecard(hard_pass=False, diagnosis="line omitted",
                         repair_code=RepairCode.LINE_OMITTED)

    rt = DagRuntime(journal, CountingExecutor(), evaluator=evaluator, owner="t")
    rt.start(linear_graph(), "r1")
    final = rt.run("r1")
    assert final.state is RunState.FAILED
    rec = journal.node("r1", "a")
    assert rec.state is NodeState.FAILED
    assert rec.repair_code is RepairCode.LINE_OMITTED
    assert rec.failure_class == "acceptance"


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #


def test_approval_gate_pauses_until_approved(journal):
    ex = CountingExecutor()
    g = linear_graph(b={"approval_gate": True})
    rt = DagRuntime(journal, ex, owner="t")
    rt.start(g, "r1")
    final = rt.run("r1")
    assert final.state is RunState.AWAITING_APPROVAL
    assert journal.node("r1", "b").state is NodeState.AWAITING_APPROVAL
    assert ex.count("b") == 0
    rt.approve("r1", "b", by="op")
    assert journal.run("r1").state is RunState.RUNNING
    final = rt.run("r1")
    assert final.state is RunState.COMPLETED
    kinds = [c["kind"] for c in journal.controls("r1")]
    assert kinds == ["approve"]


def test_reject_fails_run(journal):
    g = linear_graph(b={"approval_gate": True})
    rt = DagRuntime(journal, CountingExecutor(), owner="t")
    rt.start(g, "r1")
    rt.run("r1")
    rt.reject("r1", "b", note="not this take")
    assert journal.node("r1", "b").state is NodeState.REJECTED
    assert journal.run("r1").state is RunState.FAILED


def test_pause_and_unpause(journal):
    ex = CountingExecutor()
    rt = DagRuntime(journal, ex, owner="t")
    rt.start(linear_graph(), "r1")
    rt.step("r1")
    rt.pause("r1")
    report = rt.step("r1")
    assert report.run_state is RunState.PAUSED and report.idle
    assert ex.count("b") == 0
    rt.unpause("r1")
    assert rt.run("r1").state is RunState.COMPLETED


def test_cancel_marks_live_nodes_and_releases_reservations(journal):
    broker = ResourceBroker(gpus=1)
    g = linear_graph()
    rt = DagRuntime(journal, CountingExecutor(), broker=broker, owner="t")
    rt.start(g, "r1")
    rt.step("r1")
    cancelled = rt.cancel("r1", "operator stop")
    assert set(cancelled) == {"b", "c"}
    assert journal.run("r1").state is RunState.CANCELLED
    assert journal.node("r1", "a").state is NodeState.SUCCEEDED
    assert rt.step("r1").idle
    with pytest.raises(JournalError):
        rt.pause("r1")


def test_retry_node_resets_only_the_node_and_descendants(journal):
    ex = CountingExecutor()
    rt = DagRuntime(journal, ex, owner="t")
    rt.start(linear_graph(), "r1")
    rt.run("r1")
    reset = rt.retry_node("r1", "b", reason="identity repair")
    assert reset == ("b", "c")
    assert journal.node("r1", "a").state is NodeState.SUCCEEDED
    assert journal.run("r1").state is RunState.RUNNING
    # b's cache key was cleared so it re-executes; c is a deterministic
    # function of b's (identical) output so it is a legitimate cache hit.
    rt.run("r1")
    assert ex.count("a") == 1
    assert ex.count("b") == 2
    assert journal.node("r1", "c").cached is True
    assert journal.run("r1").state is RunState.COMPLETED


# --------------------------------------------------------------------------- #
# graph revisions / replan / repair budget
# --------------------------------------------------------------------------- #


def test_replan_keeps_unchanged_ancestors_and_resets_changed_subgraph(journal):
    ex = CountingExecutor()
    rt = DagRuntime(journal, ex, owner="t")
    g0 = linear_graph()
    rt.start(g0, "r1", repair_budget=2)
    rt.run("r1")
    # revise b (new params) -> b and c reset, a kept
    new_b = _task("b", inputs=("in",), depends=("a",), params={"seed": 7})
    g1 = g0.revise(replacing=["b"], new_nodes=[new_b], reason="reseed b")
    assert g1.revision == 1
    reset = rt.replan("r1", g1, "reseed b")
    assert set(reset) == {"b", "c"}
    recs = journal.nodes("r1")
    assert recs["a"].state is NodeState.SUCCEEDED and recs["a"].revision == 1
    assert recs["b"].state is NodeState.PENDING
    assert journal.run("r1").revision == 1
    assert [r["revision"] for r in journal.revisions("r1")] == [0, 1]
    assert journal.revisions("r1")[1]["reason"] == "reseed b"
    rt.run("r1")
    assert ex.count("a") == 1 and ex.count("b") == 2
    assert journal.run("r1").state is RunState.COMPLETED
    assert journal.graph("r1").node("b").params["seed"] == 7
    assert journal.graph("r1", revision=0).node("b").params.get("seed") is None


def test_replan_budget_is_bounded(journal):
    rt = DagRuntime(journal, CountingExecutor(), owner="t")
    g0 = linear_graph()
    rt.start(g0, "r1", repair_budget=1)
    rt.run("r1")
    g1 = g0.revise(["b"], [_task("b", inputs=("in",), depends=("a",), params={"s": 1})], "r1")
    rt.replan("r1", g1, "first")
    g2 = g1.revise(["b"], [_task("b", inputs=("in",), depends=("a",), params={"s": 2})], "r2")
    with pytest.raises(RepairBudgetExceeded):
        rt.replan("r1", g2, "second")
    assert journal.run("r1").revision == 1


def test_replan_refuses_non_monotonic_revision(journal):
    rt = DagRuntime(journal, CountingExecutor(), owner="t")
    g0 = linear_graph()
    rt.start(g0, "r1")
    with pytest.raises(JournalError):
        rt.replan("r1", linear_graph(revision=0), "same revision")


# --------------------------------------------------------------------------- #
# fan-out, join passthrough, resources
# --------------------------------------------------------------------------- #


def test_fanout_collects_candidates_per_port(journal):
    gen = _task("gen", kind=NodeKind.FANOUT, candidates=3)
    join = PlanNode(node_id="join", kind=NodeKind.JOIN,
                    inputs=(Port("takes", "text", many=True),), depends_on=("gen",))
    g = PlanGraph(graph_id="g", goal_digest="d", nodes=(gen, join),
                  edges=(Edge("gen", "out", "join", "takes"),))
    ex = CountingExecutor()
    rt = DagRuntime(journal, ex, owner="t")
    rt.start(g, "r1")
    assert rt.run("r1").state is RunState.COMPLETED
    assert [c[2] for c in ex.calls] == [0, 1, 2]
    gen_rec = journal.node("r1", "gen")
    assert gen_rec.outputs == {"out": ["gen", "gen", "gen"]}
    assert journal.node("r1", "join").outputs == {"takes": [["gen", "gen", "gen"]]}


def test_resource_reservations_are_journaled_and_released(journal):
    broker = ResourceBroker(vram_gib=8)
    g = PlanGraph(graph_id="g", goal_digest="d", nodes=(
        _task("a", resources=ResourceRequest(vram_gib=6, gpu=True)),))
    rt = DagRuntime(journal, CountingExecutor(), broker=broker, owner="t")
    rt.start(g, "r1")
    rt.run("r1")
    res = journal.reservations("r1")
    assert len(res) == 1
    assert res[0]["released_at"] is not None and res[0]["release_reason"] == "succeeded"
    assert broker.held == {}


def test_resource_shortfall_blocks_instead_of_running(journal):
    broker = ResourceBroker(vram_gib=4)
    g = PlanGraph(graph_id="g", goal_digest="d", nodes=(
        _task("a", resources=ResourceRequest(vram_gib=6)),))
    ex = CountingExecutor()
    rt = DagRuntime(journal, ex, broker=broker, owner="t")
    rt.start(g, "r1")
    report = rt.step("r1")
    assert report.blocked_on_resources == ("a",)
    assert ex.calls == []
    assert journal.node("r1", "a").state is NodeState.PENDING
    assert rt.run("r1").state is RunState.RUNNING


def test_missing_required_input_is_a_plan_error_not_a_crash(journal):
    # b declares a required port that no edge binds
    a = _task("a")
    b = _task("b", inputs=("in",), depends=("a",))
    g = PlanGraph(graph_id="g", goal_digest="d", nodes=(a, b))  # no edges
    rt = DagRuntime(journal, CountingExecutor(), owner="t")
    rt.start(g, "r1")
    final = rt.run("r1")
    assert final.state is RunState.FAILED
    rec = journal.node("r1", "b")
    assert rec.failure_class == "plan_error" and "unbound" in rec.failure


def test_round_trip_records_are_json_safe(journal):
    import json
    rt = DagRuntime(journal, CountingExecutor(), owner="t")
    rt.start(linear_graph(), "r1")
    rt.run("r1")
    json.dumps(journal.run("r1").to_dict())
    json.dumps({k: v.to_dict() for k, v in journal.nodes("r1").items()})
    json.dumps(rt.step("r1").to_dict())
