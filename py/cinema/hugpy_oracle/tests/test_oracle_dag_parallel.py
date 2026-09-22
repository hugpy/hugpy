"""or-k12 — DagRuntime.step runs the sliced ready set on a bounded thread pool.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_dag_parallel.py -q
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle.dag_runtime import DagRuntime, NodeState, RunJournal, RunState
from hugpy_oracle.plan import Edge, NodeKind, PlanGraph, PlanNode, Port

SLEEP = 0.25


def _task(nid, *, depends=(), inputs=()):
    return PlanNode(node_id=nid, kind=NodeKind.TASK, capability="text.generate",
                    inputs=tuple(Port(name=i, artifact_kind="text") for i in inputs),
                    outputs=(Port(name="out", artifact_kind="text"),), depends_on=tuple(depends),
                    params={"tag": nid})   # distinct cache keys: no cross-chain cache hits


def three_chains() -> PlanGraph:
    """a1->a2, b1->b2, c1->c2 : three independent chains, two layers."""
    nodes, edges = [], []
    for ch in "abc":
        nodes.append(_task(f"{ch}1"))
        nodes.append(_task(f"{ch}2", depends=(f"{ch}1",), inputs=("in",)))
        edges.append(Edge(f"{ch}1", "out", f"{ch}2", "in"))
    return PlanGraph(graph_id="g3", goal_digest="goal", revision=0, nodes=tuple(nodes), edges=tuple(edges))


class SleepingRunner:
    def __init__(self, sleep=SLEEP, fail=()):
        self.sleep, self.fail = sleep, set(fail)
        self.threads: dict[str, int] = {}
        self.states_seen: dict[str, NodeState] = {}
        self.journal: RunJournal | None = None

    def __call__(self, node, inputs, ctx):
        self.threads[node.node_id] = threading.get_ident()
        if self.journal is not None:
            self.states_seen[node.node_id] = self.journal.node(ctx.run_id, node.node_id).state
        time.sleep(self.sleep)
        if node.node_id in self.fail:
            raise RuntimeError(f"boom {node.node_id}")
        up = inputs.get("in")
        return {"out": f"{node.node_id}({up})" if up is not None else node.node_id}


@pytest.fixture()
def journal(tmp_path):
    j = RunJournal(os.path.join(str(tmp_path), "dag.sqlite"))
    yield j
    j.close()


def test_three_chains_complete_in_about_one_x_wall_time(journal):
    runner = SleepingRunner()
    runner.journal = journal
    rt = DagRuntime(journal, runner, owner="par", max_parallel=3)
    rt.start(three_chains(), "r1")
    t0 = time.monotonic()
    run = rt.run("r1")
    elapsed = time.monotonic() - t0
    assert run.state is RunState.COMPLETED
    # two layers of three nodes: ~2*SLEEP in parallel vs ~6*SLEEP sequentially
    assert elapsed < 4 * SLEEP, f"took {elapsed:.2f}s — nodes ran sequentially"
    assert len({runner.threads[n] for n in ("a1", "b1", "c1")}) == 3
    # record-before-run held on every worker thread
    assert set(runner.states_seen.values()) == {NodeState.RUNNING}
    recs = journal.nodes("r1")
    assert all(r.state is NodeState.SUCCEEDED for r in recs.values())
    assert recs["a2"].outputs["out"] == "a2(a1)"
    for nid in recs:
        assert [t["to_state"] for t in journal.transitions("r1", nid)] == \
            ["pending", "leased", "running", "succeeded"]
    assert journal.open_reservations("r1") == ()
    assert len(journal.reservations("r1")) == 6
    assert journal._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_sequential_baseline_is_slow(journal):
    rt = DagRuntime(journal, SleepingRunner(), owner="seq", max_parallel=1)
    rt.start(three_chains(), "r1")
    t0 = time.monotonic()
    run = rt.run("r1")
    assert run.state is RunState.COMPLETED
    assert time.monotonic() - t0 >= 6 * SLEEP * 0.9


def test_parallel_failure_fails_only_its_node(journal):
    runner = SleepingRunner(sleep=0.02, fail={"b1"})
    rt = DagRuntime(journal, runner, owner="par", max_parallel=3)
    rt.start(three_chains(), "r1")
    run = rt.run("r1")
    recs = journal.nodes("r1")
    assert recs["b1"].state is NodeState.FAILED
    assert recs["b2"].state is NodeState.PENDING   # never ran: its dep failed
    for n in ("a1", "a2", "c1", "c2"):
        assert recs[n].state is NodeState.SUCCEEDED, n
    assert run.state is RunState.FAILED
    assert journal.open_reservations("r1") == ()


def test_max_parallel_one_transitions_match_parallel_transitions(tmp_path):
    """The journal trail per node is identical whichever mode produced it."""
    trails = {}
    for mp in (1, 3):
        j = RunJournal(os.path.join(str(tmp_path), f"dag{mp}.sqlite"))
        rt = DagRuntime(j, SleepingRunner(sleep=0.01), owner="o", max_parallel=mp)
        rt.start(three_chains(), "r")
        assert rt.run("r").state is RunState.COMPLETED
        trails[mp] = {n: [(t["from_state"], t["to_state"], t["reason"]) for t in j.transitions("r", n)]
                      for n in j.nodes("r")}
        j.close()
    assert trails[1] == trails[3]


def test_step_report_counts_parallel_batch(journal):
    rt = DagRuntime(journal, SleepingRunner(sleep=0.01), owner="par", max_parallel=2)
    rt.start(three_chains(), "r1")
    rep = rt.step("r1")
    assert len(rep.executed) == 2 and len(rep.succeeded) == 2
    assert rep.ready_remaining == 3   # the 3rd root + 2 unblocked second-layer nodes
