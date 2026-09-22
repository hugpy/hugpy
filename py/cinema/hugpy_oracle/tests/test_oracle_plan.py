"""k103 — the PlanGraph contract: typed ports, frozen params, deterministic
digests, traversal, controlled graph revisions, and the Stage 14 sibling
invariant.

Everything here is offline and deterministic: the plan is a data structure, not
an engine, so no catalog, no workers, no GPU, no clock.

Locks:
  [1] wire shape: every contract round-trips through to_dict/from_dict without
      loss, and plan_digest is stable across dict ordering + round-trips while
      changing when the plan changes.
  [2] frozen params: a node carries a mapping, stays hashable, freezes nested
      structures, and refuses non-JSON values.
  [3] structural refusals at construction: a task without a capability, a join
      WITH one, a self-loop, duplicate ports — programmer error, raised, never
      carried downstream.
  [4] traversal: topological order is deterministic and honours both edges and
      depends_on; a cycle raises CycleError naming the tangle; ancestors /
      descendants terminate on a cyclic graph.
  [5] revise() = a controlled graph revision: revision bumps, parent_revision
      is set, untouched nodes keep their IDENTITY (same object), replaced nodes
      swap in place, dropped nodes take their edges with them, and an
      unexplained revision is refused.
  [6] invariant 9 / Stage 14: siblings may share a locked parent and must not
      depend on each other.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_plan.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys

logging.disable(logging.INFO)  # silence the models_config registry chatter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle.contracts import (
    ArtifactKind,
    AuthorityKind,
    BudgetHints,
    CheckKind,
    GoalSpec,
    InputKind,
    InputRef,
    PlannerMode,
    RepairCode,
)
from hugpy_oracle.plan import (
    AcceptanceTest,
    CycleError,
    Edge,
    FrozenParams,
    NodeKind,
    PlanGraph,
    PlanNode,
    Port,
    ResourceRequest,
    RetryPolicy,
    canonical_json,
    coerce_artifact_kind,
    goal_digest,
    sibling_check,
    sibling_violations,
)

GOAL = GoalSpec(objective="transcribe the interview",
                raw_prompt="transcribe the interview please",
                inputs=(InputRef(kind=InputKind.AUDIO, ref="/tmp/a.wav"),))
DIGEST = goal_digest(GOAL)


def _task(node_id: str, capability: str = "audio.transcribe", **kw) -> PlanNode:
    kw.setdefault("outputs", (Port("out", ArtifactKind.JSON),))
    return PlanNode(node_id=node_id, kind=NodeKind.TASK,
                    capability=capability, **kw)


def _graph(nodes, edges=(), **kw) -> PlanGraph:
    return PlanGraph(graph_id="g1", goal_digest=DIGEST, nodes=tuple(nodes),
                     edges=tuple(edges), **kw)


# ---------------------------------------------------------------------------
# [1] ports and kinds
# ---------------------------------------------------------------------------


def test_port_kind_coerces_known_names_to_the_enum():
    assert coerce_artifact_kind("audio") is ArtifactKind.AUDIO
    assert Port("a", "video").artifact_kind is ArtifactKind.VIDEO


def test_port_kind_keeps_unknown_logical_types_as_strings():
    """The enum is media kinds; a plan also moves artifacts it will never
    enumerate. Inventing an enum member would be fabrication."""
    port = Port("t", "dialogue_timeline")
    assert port.artifact_kind == "dialogue_timeline"
    assert port.kind_value == "dialogue_timeline"


def test_port_rejects_empty_name_and_empty_kind():
    with pytest.raises(ValueError):
        Port("", ArtifactKind.TEXT)
    with pytest.raises(ValueError):
        Port("a", "   ")


def test_port_roundtrips():
    port = Port("clips", ArtifactKind.VIDEO, required=False, many=True)
    assert Port.from_dict(port.to_dict()) == port


# ---------------------------------------------------------------------------
# [2] FrozenParams
# ---------------------------------------------------------------------------


def test_frozen_params_reads_like_a_mapping_and_is_hashable():
    params = FrozenParams({"segment": True, "seed": 7})
    assert params["segment"] is True
    assert params.get("missing") is None
    assert "seed" in params
    assert len(params) == 2
    assert hash(params) == hash(FrozenParams({"seed": 7, "segment": True}))


def test_frozen_params_freezes_nested_structures():
    params = FrozenParams({"shot": {"a": [1, 2]}})
    assert isinstance(params["shot"], FrozenParams)
    assert params["shot"]["a"] == (1, 2)
    with pytest.raises(TypeError):
        params["shot"]["a"] = 3


def test_frozen_params_iterates_sorted_and_round_trips_to_plain_json():
    params = FrozenParams({"z": 1, "a": {"b": [1, {"c": 2}]}})
    assert list(params) == ["a", "z"]
    assert params.to_dict() == {"a": {"b": [1, {"c": 2}]}, "z": 1}
    assert json.loads(canonical_json(params.to_dict())) == params.to_dict()


def test_frozen_params_refuses_values_that_are_not_json_safe():
    with pytest.raises(TypeError):
        FrozenParams({"handle": object()})


def test_frozen_params_compares_equal_to_a_plain_dict():
    assert FrozenParams({"a": 1}) == {"a": 1}


# ---------------------------------------------------------------------------
# [3] construction-time refusals
# ---------------------------------------------------------------------------


def test_task_node_must_name_a_capability():
    with pytest.raises(ValueError, match="must name a capability"):
        PlanNode(node_id="n", kind=NodeKind.TASK)


def test_structural_node_must_not_name_a_capability():
    with pytest.raises(ValueError, match="must NOT name a capability"):
        PlanNode(node_id="j", kind=NodeKind.JOIN, capability="audio.transcribe")
    assert PlanNode(node_id="j", kind=NodeKind.JOIN).capability is None


def test_capability_must_be_namespaced():
    with pytest.raises(ValueError, match="namespaced"):
        PlanNode(node_id="n", kind=NodeKind.TASK, capability="transcribe")


def test_node_refuses_duplicate_ports_self_dependency_and_zero_candidates():
    with pytest.raises(ValueError, match="duplicate inputs port"):
        _task("n", inputs=(Port("a", ArtifactKind.TEXT),
                           Port("a", ArtifactKind.TEXT)))
    with pytest.raises(ValueError, match="cannot depend on itself"):
        _task("n", depends_on=("n",))
    with pytest.raises(ValueError, match="candidates"):
        _task("n", candidates=0)


def test_ports_must_be_tuples_not_a_lone_port():
    """The missing-comma bug, caught where it happens instead of three frames
    deep in a traversal."""
    with pytest.raises(TypeError, match="missing-comma"):
        _task("n", inputs=(Port("a", ArtifactKind.TEXT)))


def test_edge_refuses_self_loops_and_blanks():
    with pytest.raises(ValueError, match="self-loop"):
        Edge("a", "out", "a", "in")
    with pytest.raises(ValueError, match="src_port"):
        Edge("a", "", "b", "in")


def test_retry_policy_bounds_and_namespaced_fallback():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(timeout_s=0)
    with pytest.raises(ValueError, match="namespaced"):
        RetryPolicy(fallback="tts")
    assert RetryPolicy(max_attempts=3, timeout_s=90.0,
                       fallback="audio.tts").fallback == "audio.tts"


def test_resource_request_refuses_negative_values():
    with pytest.raises(ValueError, match="vram_gib"):
        ResourceRequest(vram_gib=-1)


def test_acceptance_test_kind_coerces_or_stays_free_text():
    assert AcceptanceTest(kind="speech").kind is CheckKind.SPEECH
    free = AcceptanceTest(kind="duration_within_window", threshold=0.25,
                          repair_code=RepairCode.SHOT_TOO_SHORT)
    assert free.kind == "duration_within_window"
    assert AcceptanceTest.from_dict(free.to_dict()) == free


def test_graph_refuses_blank_ids_and_impossible_parent_revisions():
    with pytest.raises(ValueError, match="graph_id"):
        PlanGraph(graph_id=" ", goal_digest=DIGEST)
    with pytest.raises(ValueError, match="goal_digest"):
        PlanGraph(graph_id="g", goal_digest="")
    with pytest.raises(ValueError, match="must precede"):
        PlanGraph(graph_id="g", goal_digest=DIGEST, revision=1,
                  parent_revision=1)


# ---------------------------------------------------------------------------
# [1] round-trip + digests
# ---------------------------------------------------------------------------


def _rich_graph() -> PlanGraph:
    gen = PlanNode(
        node_id="gen", kind=NodeKind.FANOUT, capability="video.generate.id_lock",
        inputs=(Port("keyframe", ArtifactKind.IMAGE),),
        outputs=(Port("clips", ArtifactKind.VIDEO, many=True),),
        params={"seed": 11, "shot": {"lens": "35mm"}},
        candidates=3, approval_gate=True,
        retry=RetryPolicy(max_attempts=2, timeout_s=120.0,
                          fallback="video.generate.t2v"),
        resources=ResourceRequest(vram_gib=12.5, ram_gib=32, gpu=True,
                                  est_seconds=90),
        acceptance=(AcceptanceTest(CheckKind.IDENTITY, 0.7,
                                   RepairCode.IDENTITY_DRIFT),),
        cache_key="ck", idempotency_key="ik",
        authority_required=((AuthorityKind.LIKENESS, "identity_profile:mira"),),
    )
    judge = PlanNode(node_id="judge", kind=NodeKind.JUDGE,
                     capability="video.evaluate.action_alignment",
                     inputs=(Port("clips", ArtifactKind.VIDEO, many=True),),
                     outputs=(Port("card", "scorecard"),),
                     depends_on=("gen",))
    return PlanGraph(graph_id="g1", goal_digest=DIGEST, revision=0,
                     nodes=(gen, judge),
                     edges=(Edge("gen", "clips", "judge", "clips"),),
                     budgets=BudgetHints(max_seconds=600, max_vram_gb=24),
                     planner_mode=PlannerMode.LOCAL_ONLY, recipe="video.performance")


def test_graph_round_trips_losslessly():
    graph = _rich_graph()
    assert PlanGraph.from_dict(graph.to_dict()) == graph


def test_graph_round_trip_preserves_the_digest():
    graph = _rich_graph()
    assert PlanGraph.from_dict(json.loads(json.dumps(graph.to_dict()))).plan_digest() \
        == graph.plan_digest()


def test_digest_is_stable_across_param_insertion_order():
    a = _graph((_task("n", params={"a": 1, "b": {"x": 1, "y": 2}}),))
    b = _graph((_task("n", params={"b": {"y": 2, "x": 1}, "a": 1}),))
    assert a.plan_digest() == b.plan_digest()


def test_digest_changes_when_anything_about_the_plan_changes():
    base = _graph((_task("n", params={"seed": 1}),))
    other = _graph((_task("n", params={"seed": 2}),))
    assert base.plan_digest() != other.plan_digest()


def test_structure_digest_ignores_identity_and_provenance():
    a = _rich_graph()
    b = PlanGraph(graph_id="other", goal_digest=DIGEST, revision=4,
                  nodes=a.nodes, edges=a.edges, parent_revision=3,
                  recipe=None, revision_reason="whatever")
    assert a.structure_digest() == b.structure_digest()
    assert a.plan_digest() != b.plan_digest()


def test_goal_digest_survives_a_goal_round_trip():
    assert goal_digest(GoalSpec.from_dict(GOAL.to_dict())) == DIGEST


def test_nodes_are_hashable_so_repair_can_hold_a_set_of_them():
    graph = _rich_graph()
    assert len(set(graph.nodes)) == 2
    assert len({graph, _rich_graph()}) == 1


# ---------------------------------------------------------------------------
# [4] traversal
# ---------------------------------------------------------------------------


def _chain() -> PlanGraph:
    a = _task("a", outputs=(Port("out", ArtifactKind.JSON),))
    b = _task("b", inputs=(Port("in", ArtifactKind.JSON),),
              outputs=(Port("out", ArtifactKind.JSON),))
    c = _task("c", inputs=(Port("in", ArtifactKind.JSON),))
    return _graph((c, b, a), (Edge("a", "out", "b", "in"),
                              Edge("b", "out", "c", "in")))


def test_topological_order_follows_edges_not_declaration_order():
    assert _chain().topological_order() == ("a", "b", "c")


def test_topological_order_honours_depends_on_without_an_edge():
    graph = _graph((_task("second", depends_on=("first",)), _task("first")))
    assert graph.topological_order() == ("first", "second")


def test_topological_order_is_deterministic_across_ties():
    graph = _graph((_task("x"), _task("y"), _task("z")))
    assert graph.topological_order() == ("x", "y", "z")
    assert graph.topological_order() == graph.topological_order()


def test_cycle_raises_and_names_the_tangle():
    graph = _graph((_task("a", depends_on=("b",)), _task("b", depends_on=("a",)),
                    _task("c")))
    with pytest.raises(CycleError) as exc:
        graph.topological_order()
    assert exc.value.nodes == ("a", "b")


def test_ancestors_and_descendants_are_transitive():
    graph = _chain()
    assert graph.ancestors("c") == frozenset({"a", "b"})
    assert graph.descendants("a") == frozenset({"b", "c"})
    assert graph.ancestors("a") == frozenset()


def test_traversal_terminates_on_a_cyclic_graph():
    graph = _graph((_task("a", depends_on=("b",)), _task("b", depends_on=("a",))))
    assert graph.ancestors("a") == frozenset({"b"})
    assert graph.descendants("b") == frozenset({"a"})


def test_roots_leaves_and_incident_edges():
    graph = _chain()
    assert graph.roots() == ("a",)
    assert graph.leaves() == ("c",)
    assert graph.incoming("b") == (Edge("a", "out", "b", "in"),)
    assert graph.outgoing("b") == (Edge("b", "out", "c", "in"),)


def test_unknown_node_references_do_not_break_traversal():
    """Traversal must not require a valid graph — the VALIDATOR reports the
    dangling reference, the graph still walks."""
    graph = _graph((_task("a"),), (Edge("ghost", "out", "a", "in"),))
    assert graph.topological_order() == ("a",)
    assert graph.ancestors("a") == frozenset()


# ---------------------------------------------------------------------------
# subgraph
# ---------------------------------------------------------------------------


def test_subgraph_keeps_node_identity_and_only_internal_edges():
    graph = _chain()
    frag = graph.subgraph(("b", "c"))
    assert frag.node_ids == ("c", "b")     # declaration order is preserved
    assert frag.node("b") is graph.node("b")
    assert frag.edges == (Edge("b", "out", "c", "in"),)


def test_subgraph_refuses_unknown_ids():
    with pytest.raises(ValueError, match="unknown node ids"):
        _chain().subgraph(("b", "nope"))


# ---------------------------------------------------------------------------
# [5] revise
# ---------------------------------------------------------------------------


def test_revise_bumps_the_revision_and_records_the_parent_and_reason():
    graph = _chain()
    revised = graph.revise({"b"}, (_task("b", inputs=(Port("in", ArtifactKind.JSON),),
                                        outputs=(Port("out", ArtifactKind.JSON),),
                                        params={"attempt": 2}),),
                           reason="identity_drift on b")
    assert (revised.revision, revised.parent_revision) == (1, 0)
    assert revised.revision_reason == "identity_drift on b"
    assert graph.revision == 0            # the original is untouched


def test_revise_keeps_untouched_nodes_identical_and_swaps_the_replacement_in_place():
    graph = _chain()
    replacement = _task("b", inputs=(Port("in", ArtifactKind.JSON),),
                        outputs=(Port("out", ArtifactKind.JSON),),
                        params={"attempt": 2})
    revised = graph.revise({"b"}, (replacement,), reason="retry b")
    assert revised.node_ids == graph.node_ids
    assert revised.node("a") is graph.node("a")
    assert revised.node("c") is graph.node("c")
    assert revised.node("b") is replacement
    assert revised.edges == graph.edges


def test_revise_can_add_nodes_and_drop_replaced_ones_with_their_edges():
    graph = _chain()
    added = _task("b2", inputs=(Port("in", ArtifactKind.JSON),))
    revised = graph.revise({"c"}, (added,), reason="replace the tail")
    assert revised.node_ids == ("b", "a", "b2")
    assert revised.edges == (Edge("a", "out", "b", "in"),)


def test_revise_refuses_an_unexplained_revision_and_unknown_or_colliding_ids():
    graph = _chain()
    with pytest.raises(ValueError, match="reason"):
        graph.revise({"b"}, (), reason="  ")
    with pytest.raises(ValueError, match="unknown node ids"):
        graph.revise({"ghost"}, (), reason="x")
    with pytest.raises(ValueError, match="collides"):
        graph.revise({"b"}, (_task("a"),), reason="x")


def test_revise_chains_keep_the_lineage():
    r1 = _chain().revise({"c"}, (), reason="drop the tail")
    r2 = r1.revise({"b"}, (), reason="drop the middle")
    assert (r2.revision, r2.parent_revision) == (2, 1)
    assert r2.node_ids == ("a",)


# ---------------------------------------------------------------------------
# [6] Stage 14 — the sibling invariant
# ---------------------------------------------------------------------------


def _segment(node_id: str, **kw) -> PlanNode:
    return PlanNode(node_id=node_id, kind=NodeKind.TASK,
                    capability="video.generate.id_lock",
                    inputs=(Port("spec", ArtifactKind.JSON),),
                    outputs=(Port("clip", ArtifactKind.VIDEO),),
                    params={"segment": True}, **kw)


def _fanout_from_lock(*extra_edges: Edge) -> PlanGraph:
    lock = _task("lock", capability="scene.plan.shots",
                 outputs=(Port("locked", ArtifactKind.JSON),))
    nodes = [lock] + [_segment(f"s{i}") for i in (1, 2, 3)]
    edges = [Edge("lock", "locked", f"s{i}", "spec") for i in (1, 2, 3)]
    return _graph(nodes, tuple(edges) + extra_edges)


def test_siblings_may_share_a_locked_parent():
    graph = _fanout_from_lock()
    assert graph.segment_node_ids() == ("s1", "s2", "s3")
    assert sibling_check(graph, graph.segment_node_ids()) is True
    assert sibling_violations(graph, graph.segment_node_ids()) == ()


def test_a_segment_chain_is_the_prohibited_relationship():
    lock = _task("lock", capability="scene.plan.shots",
                 outputs=(Port("locked", ArtifactKind.JSON),))
    graph = _graph((lock, _segment("s1"), _segment("s2", depends_on=("s1",)),
                    _segment("s3", depends_on=("s2",))),
                   (Edge("lock", "locked", "s1", "spec"),))
    assert sibling_check(graph, ("s1", "s2", "s3")) is False
    assert sibling_violations(graph, ("s1", "s2", "s3")) == (
        ("s2", "s1"), ("s3", "s1"), ("s3", "s2"))


def test_sibling_check_refuses_unknown_node_ids():
    with pytest.raises(ValueError, match="unknown node ids"):
        sibling_check(_fanout_from_lock(), ("s1", "ghost"))


def test_a_single_segment_cannot_violate_the_invariant():
    graph = _graph((_segment("s1"),))
    assert graph.segment_node_ids() == ("s1",)
    assert sibling_check(graph, graph.segment_node_ids()) is True


# ---------------------------------------------------------------------------
# k113 — the planner-mode gate (POLICY-rights-consent-disclosure §3.1-3.2)
# ---------------------------------------------------------------------------

from hugpy_oracle.plan import (
    FRONTIER_ENABLED_ENV,
    PlannerModeRefusal,
    check_planner_mode,
    effective_planner_mode,
    frontier_enabled,
    frontier_nodes,
    is_frontier_capability,
)
from hugpy_oracle.contracts import FailureClass, PlannerMode


def test_frontier_capability_is_named_by_prefix_or_param():
    assert is_frontier_capability("frontier.plan")
    assert is_frontier_capability("text.chat", {"frontier": True})
    assert not is_frontier_capability("text.chat", {"frontier": "yes"})
    assert not is_frontier_capability("text.chat")
    assert not is_frontier_capability(None)


def test_frontier_enabled_is_an_operator_switch_default_off(monkeypatch):
    monkeypatch.delenv(FRONTIER_ENABLED_ENV, raising=False)
    assert frontier_enabled() is False
    assert effective_planner_mode("frontier") is PlannerMode.LOCAL_ONLY
    assert effective_planner_mode("telepathy") is PlannerMode.LOCAL_ONLY
    assert effective_planner_mode(None) is PlannerMode.LOCAL_ONLY
    monkeypatch.setenv(FRONTIER_ENABLED_ENV, "true")
    assert effective_planner_mode("frontier") is PlannerMode.FRONTIER
    assert effective_planner_mode(PlannerMode.LOCAL_ONLY) is PlannerMode.LOCAL_ONLY


def test_local_only_plan_with_a_frontier_node_is_a_typed_refusal(monkeypatch):
    monkeypatch.setenv(FRONTIER_ENABLED_ENV, "1")
    g = _graph([_task("t"), _task("f", "frontier.plan"),
                _task("p", "text.chat", params={"frontier": True})])
    assert g.planner_mode is PlannerMode.LOCAL_ONLY
    assert frontier_nodes(g) == (("f", "frontier.plan"), ("p", "text.chat"))
    r = check_planner_mode(g)
    assert isinstance(r, PlannerModeRefusal)
    assert r.failure is FailureClass.REFUSED
    assert r.nodes == (("f", "frontier.plan"), ("p", "text.chat"))
    d = r.to_dict()
    assert d["outcome"] == "refused" and d["planner_mode"] == "local_only"
    assert [n["node_id"] for n in d["nodes"]] == ["f", "p"]
    json.dumps(d)
    # zero frontier nodes -> no refusal
    assert check_planner_mode(_graph([_task("t")])) is None


def test_frontier_plan_on_a_frontier_disabled_fleet_is_refused(monkeypatch):
    g = _graph([_task("t"), _task("f", "frontier.plan")],
               planner_mode=PlannerMode.FRONTIER)
    monkeypatch.delenv(FRONTIER_ENABLED_ENV, raising=False)
    r = check_planner_mode(g)
    assert r is not None and FRONTIER_ENABLED_ENV in r.reason
    assert r.to_dict()["frontier_enabled"] is False
    monkeypatch.setenv(FRONTIER_ENABLED_ENV, "1")
    assert check_planner_mode(g) is None


def test_planner_mode_refusal_must_carry_evidence():
    with pytest.raises(ValueError):
        PlannerModeRefusal(planner_mode=PlannerMode.LOCAL_ONLY, nodes=(), reason="")
