"""k103 — the STATIC plan validator: doc §6 step 7, run before a plan is quoted
and long before anything executes.

The catalog is a plain dict of hand-built ``CapabilityView``s: validation is
pure and offline by construction, so these tests need no registry, no workers,
no GPU and no monkeypatching.

Locks:
  [1] a well-formed plan over eligible capabilities validates clean — no
      errors, no warnings (a validator that always complains gets ignored).
  [2] structure: DUPLICATE_NODE, DANGLING_EDGE (node, port and depends_on),
      CYCLE.
  [3] capabilities: UNKNOWN_CAPABILITY for one the catalog never heard of,
      CAPABILITY_GAP for one it knows and cannot run — CARRYING the catalog's
      own reasons, per §4 ("never hallucinate an adapter").
  [4] types: PORT_TYPE_MISMATCH on kind and on cardinality (many -> one), while
      the legitimate fan-in (one -> many) passes; a node's ports are also
      checked against the capability's declared I/O.
  [5] MISSING_REQUIRED_INPUT unless an edge, a params binding or a supplied
      goal input feeds the port; MAP_WITHOUT_MANY on a map that maps over
      nothing.
  [6] AUTHORITY_MISSING through k97's gate — from the goal, from the node's own
      params, and from a node's declared authority_required — cleared by a
      RightsManifest that covers it.
  [7] BUDGET_EXCEEDED on estimated seconds (candidates included) and on peak
      VRAM, with a BUDGET_UNKNOWN warning when the estimate is missing.
  [8] SIBLING_VIOLATION (invariant 9) and JUDGE_IS_GENERATOR (invariant 11 —
      error when the model is pinned to the generator's, warning when no model
      is pinned yet).
  [9] the report is deterministic, never mutates the plan and never auto-fixes.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_validator.py -q
"""
from __future__ import annotations

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
    Authorization,
    AuthorityKind,
    BudgetHints,
    CapabilityView,
    Eligibility,
    GoalSpec,
    InputKind,
    InputRef,
    RightsManifest,
    SourceRegistry,
)
from hugpy_oracle.plan import (
    Edge,
    NodeKind,
    PlanGraph,
    PlanNode,
    Port,
    ResourceRequest,
    goal_digest,
)
from hugpy_oracle.validator import ErrorCode, ValidationError, ValidationReport, validate


# ---------------------------------------------------------------------------
# Fixtures: a three-row fake catalog and the goal it answers
# ---------------------------------------------------------------------------


def _view(name: str, accepts=(), produces=(ArtifactKind.JSON,),
          eligible: bool = True, reasons=()) -> CapabilityView:
    return CapabilityView(
        name=name, source=SourceRegistry.TASKS, accepts=tuple(accepts),
        produces=tuple(produces), model_ids=("fake-model",),
        eligibility=Eligibility(eligible=eligible, reasons=tuple(reasons)))


def _catalog(*views: CapabilityView) -> dict[str, CapabilityView]:
    return {v.name: v for v in views}


CATALOG = _catalog(
    _view("audio.transcribe", accepts=(ArtifactKind.AUDIO,),
          produces=(ArtifactKind.JSON,)),
    _view("text.summarize", accepts=(ArtifactKind.JSON, ArtifactKind.TEXT),
          produces=(ArtifactKind.TEXT,)),
)

GOAL = GoalSpec(objective="transcribe and summarize the interview",
                raw_prompt="transcribe and summarize the interview",
                inputs=(InputRef(kind=InputKind.AUDIO, ref="/tmp/a.wav"),))


def _goal(**kw) -> GoalSpec:
    base = dict(objective=GOAL.objective, raw_prompt=GOAL.raw_prompt,
                inputs=GOAL.inputs)
    base.update(kw)
    return GoalSpec(**base)


def _graph(nodes, edges=(), goal: GoalSpec = GOAL, **kw) -> PlanGraph:
    return PlanGraph(graph_id="g1", goal_digest=goal_digest(goal),
                     nodes=tuple(nodes), edges=tuple(edges), **kw)


def _asr(node_id: str = "asr", **kw) -> PlanNode:
    kw.setdefault("inputs", (Port("audio", ArtifactKind.AUDIO),))
    kw.setdefault("outputs", (Port("transcript", ArtifactKind.JSON),))
    return PlanNode(node_id=node_id, kind=NodeKind.TASK,
                    capability="audio.transcribe", **kw)


def _sum(node_id: str = "sum", **kw) -> PlanNode:
    kw.setdefault("inputs", (Port("doc", ArtifactKind.JSON),))
    kw.setdefault("outputs", (Port("summary", ArtifactKind.TEXT),))
    return PlanNode(node_id=node_id, kind=NodeKind.TASK,
                    capability="text.summarize", **kw)


def _happy() -> PlanGraph:
    return _graph((_asr(), _sum()),
                  (Edge("asr", "transcript", "sum", "doc"),))


def _codes(report: ValidationReport) -> list[str]:
    return [e.code.value for e in report.errors]


# ---------------------------------------------------------------------------
# [1] the clean plan
# ---------------------------------------------------------------------------


def test_a_well_formed_plan_validates_with_no_findings_at_all():
    report = validate(_happy(), CATALOG, GOAL)
    assert report.ok is True
    assert report.errors == ()
    assert report.warnings == ()


def test_report_ok_cannot_contradict_its_errors():
    err = ValidationError(ErrorCode.CYCLE, None, "boom")
    with pytest.raises(ValueError, match="contradicts"):
        ValidationReport(ok=True, errors=(err,))
    with pytest.raises(ValueError, match="message"):
        ValidationError(ErrorCode.CYCLE, None, "  ")


def test_report_round_trips_and_is_deterministic():
    graph = _graph((_asr("a", depends_on=("b",)), _asr("b", depends_on=("a",))))
    first = validate(graph, CATALOG, GOAL)
    second = validate(graph, CATALOG, GOAL)
    assert first.to_dict() == second.to_dict()
    assert ValidationReport.from_dict(first.to_dict()) == first


def test_validation_never_touches_the_plan():
    graph = _happy()
    before = graph.plan_digest()
    validate(graph, CATALOG, GOAL)
    assert graph.plan_digest() == before


# ---------------------------------------------------------------------------
# [2] structure
# ---------------------------------------------------------------------------


def test_cycle_is_reported_with_the_tangled_nodes():
    graph = _graph((_asr("a", depends_on=("b",)), _asr("b", depends_on=("a",))))
    report = validate(graph, CATALOG, GOAL)
    cycles = [e for e in report.errors if e.code is ErrorCode.CYCLE]
    assert len(cycles) == 1
    assert cycles[0].node_id is None
    assert cycles[0].detail == ("a", "b")


def test_duplicate_node_ids_are_reported():
    graph = _graph((_asr("asr"), _asr("asr")))
    report = validate(graph, CATALOG, GOAL)
    dupes = [e for e in report.errors if e.code is ErrorCode.DUPLICATE_NODE]
    assert [e.node_id for e in dupes] == ["asr"]
    assert "declared 2 times" in dupes[0].message


def test_edge_to_an_unknown_node_is_dangling():
    graph = _graph((_asr(),), (Edge("asr", "transcript", "ghost", "doc"),))
    report = validate(graph, CATALOG, GOAL)
    codes = [e.code for e in report.errors]
    assert ErrorCode.DANGLING_EDGE in codes
    assert any("unknown destination node" in e.message for e in report.errors)


def test_edge_to_an_unknown_port_is_dangling():
    graph = _graph((_asr(), _sum()),
                   (Edge("asr", "nope", "sum", "doc"),))
    report = validate(graph, CATALOG, GOAL)
    dangling = [e for e in report.errors if e.code is ErrorCode.DANGLING_EDGE]
    assert [e.node_id for e in dangling] == ["asr"]
    assert "has no output port" in dangling[0].message


def test_depends_on_an_unknown_node_is_dangling():
    graph = _graph((_asr(depends_on=("ghost",)),))
    report = validate(graph, CATALOG, GOAL)
    assert [(e.code, e.node_id) for e in report.errors] == [
        (ErrorCode.DANGLING_EDGE, "asr")]


# ---------------------------------------------------------------------------
# [3] capabilities
# ---------------------------------------------------------------------------


def test_unknown_capability_is_refused_not_invented():
    graph = _graph((PlanNode(node_id="x", kind=NodeKind.TASK,
                             capability="video.hallucinate",
                             outputs=(Port("out", ArtifactKind.VIDEO),)),))
    report = validate(graph, CATALOG, GOAL)
    assert _codes(report) == ["unknown_capability"]
    assert "must not invent" in report.errors[0].message


def test_capability_gap_carries_the_catalogs_own_reasons():
    catalog = _catalog(_view("audio.tts", produces=(ArtifactKind.AUDIO,),
                             eligible=False,
                             reasons=("no online worker seats audio.tts",
                                      "chatterbox runner probe not run")))
    graph = _graph((PlanNode(node_id="tts", kind=NodeKind.TASK,
                             capability="audio.tts",
                             outputs=(Port("wav", ArtifactKind.AUDIO),)),))
    report = validate(graph, catalog, GOAL)
    gaps = [e for e in report.errors if e.code is ErrorCode.CAPABILITY_GAP]
    assert len(gaps) == 1
    assert gaps[0].node_id == "tts"
    assert gaps[0].detail == ("no online worker seats audio.tts",
                              "chatterbox runner probe not run")
    assert "no online worker seats audio.tts" in gaps[0].message


def test_a_structural_node_needs_no_capability_and_is_not_a_gap():
    join = PlanNode(node_id="join", kind=NodeKind.JOIN,
                    inputs=(Port("a", ArtifactKind.TEXT, required=False),),
                    outputs=(Port("out", ArtifactKind.TEXT),))
    report = validate(_graph((join,)), CATALOG, GOAL)
    assert report.ok is True


# ---------------------------------------------------------------------------
# [4] types
# ---------------------------------------------------------------------------


def test_port_kind_mismatch_across_an_edge():
    sink = PlanNode(node_id="sum", kind=NodeKind.TASK, capability="text.summarize",
                    inputs=(Port("doc", ArtifactKind.TEXT),),
                    outputs=(Port("summary", ArtifactKind.TEXT),))
    graph = _graph((_asr(), sink), (Edge("asr", "transcript", "sum", "doc"),))
    report = validate(graph, CATALOG, GOAL)
    mism = [e for e in report.errors if e.code is ErrorCode.PORT_TYPE_MISMATCH]
    assert [e.node_id for e in mism] == ["sum"]
    assert "produces 'json'" in mism[0].message


def test_many_into_one_is_a_cardinality_mismatch():
    src = PlanNode(node_id="fan", kind=NodeKind.FANOUT,
                   capability="audio.transcribe", candidates=3,
                   inputs=(Port("audio", ArtifactKind.AUDIO),),
                   outputs=(Port("transcripts", ArtifactKind.JSON, many=True),))
    graph = _graph((src, _sum()), (Edge("fan", "transcripts", "sum", "doc"),))
    report = validate(graph, CATALOG, GOAL)
    mism = [e for e in report.errors if e.code is ErrorCode.PORT_TYPE_MISMATCH]
    assert len(mism) == 1
    assert "map or reduce" in mism[0].message


def test_one_into_many_is_the_legitimate_fan_in():
    a = _asr("a")
    b = _asr("b")
    reduce_node = PlanNode(node_id="pick", kind=NodeKind.REDUCE,
                           inputs=(Port("candidates", ArtifactKind.JSON, many=True),),
                           outputs=(Port("best", ArtifactKind.JSON),))
    graph = _graph((a, b, reduce_node),
                   (Edge("a", "transcript", "pick", "candidates"),
                    Edge("b", "transcript", "pick", "candidates")))
    assert validate(graph, CATALOG, GOAL).ok is True


def test_node_ports_are_checked_against_the_capabilitys_declared_io():
    bad = PlanNode(node_id="asr", kind=NodeKind.TASK, capability="audio.transcribe",
                   inputs=(Port("clip", ArtifactKind.VIDEO),),
                   outputs=(Port("wav", ArtifactKind.AUDIO),))
    report = validate(_graph((bad,)), CATALOG, GOAL)
    messages = [e.message for e in report.errors
                if e.code is ErrorCode.PORT_TYPE_MISMATCH]
    assert any("does not accept 'video'" in m for m in messages)
    assert any("does not produce 'audio'" in m for m in messages)


def test_logical_artifact_types_are_not_second_guessed_by_the_catalog():
    """A port typed ``dialogue_timeline`` is not evidence of a mismatch — the
    catalog has no vocabulary for it yet."""
    node = PlanNode(node_id="asr", kind=NodeKind.TASK, capability="audio.transcribe",
                    inputs=(Port("audio", ArtifactKind.AUDIO),),
                    outputs=(Port("timeline", "dialogue_timeline"),))
    assert validate(_graph((node,)), CATALOG, GOAL).ok is True


# ---------------------------------------------------------------------------
# [5] inputs and map
# ---------------------------------------------------------------------------


def test_required_input_with_nothing_feeding_it():
    report = validate(_graph((_sum(),)), CATALOG, GOAL)
    assert _codes(report) == ["missing_required_input"]
    assert report.errors[0].node_id == "sum"


def test_a_required_input_may_be_satisfied_by_a_supplied_goal_input():
    assert validate(_graph((_asr(),)), CATALOG, GOAL).ok is True


def test_a_required_input_may_be_satisfied_by_a_params_binding():
    node = _sum(params={"doc": "inline text"})
    assert validate(_graph((node,)), CATALOG, GOAL).ok is True


def test_an_optional_input_needs_nothing():
    node = _sum(inputs=(Port("doc", ArtifactKind.JSON, required=False),))
    assert validate(_graph((node,)), CATALOG, GOAL).ok is True


def test_map_over_a_single_valued_port_is_refused():
    node = PlanNode(node_id="m", kind=NodeKind.MAP, capability="text.summarize",
                    inputs=(Port("scenes", ArtifactKind.JSON),),
                    outputs=(Port("out", ArtifactKind.TEXT),),
                    map_over="scenes", params={"scenes": []})
    report = validate(_graph((node,)), CATALOG, GOAL)
    assert _codes(report) == ["map_without_many"]
    assert "nothing to map over" in report.errors[0].message


def test_map_without_map_over_and_map_over_an_absent_port():
    node = PlanNode(node_id="m", kind=NodeKind.MAP, capability="text.summarize",
                    inputs=(Port("scenes", ArtifactKind.JSON, many=True,
                                 required=False),),
                    outputs=(Port("out", ArtifactKind.TEXT),))
    report = validate(_graph((node,)), CATALOG, GOAL)
    assert "must name the input port it maps over" in report.errors[0].message

    ghost = PlanNode(node_id="m", kind=NodeKind.MAP, capability="text.summarize",
                     inputs=(Port("scenes", ArtifactKind.JSON, many=True,
                                  required=False),),
                     outputs=(Port("out", ArtifactKind.TEXT),),
                     map_over="nope")
    report = validate(_graph((ghost,)), CATALOG, GOAL)
    assert "not an input port" in report.errors[0].message


def test_map_over_on_a_non_map_node_is_refused():
    node = _sum(map_over="doc", params={"doc": "x"})
    report = validate(_graph((node,)), CATALOG, GOAL)
    assert _codes(report) == ["map_without_many"]
    assert "only meaningful on a map node" in report.errors[0].message


def test_a_well_formed_map_node_validates():
    node = PlanNode(node_id="m", kind=NodeKind.MAP, capability="text.summarize",
                    inputs=(Port("scenes", ArtifactKind.JSON, many=True),),
                    outputs=(Port("out", ArtifactKind.TEXT, many=True),),
                    map_over="scenes", params={"scenes": [1, 2]})
    assert validate(_graph((node,)), CATALOG, GOAL).ok is True


# ---------------------------------------------------------------------------
# [6] authority (k97's gate, statically)
# ---------------------------------------------------------------------------


IDENTITY_CATALOG = _catalog(
    _view("video.generate.id_lock", accepts=(ArtifactKind.IMAGE,),
          produces=(ArtifactKind.VIDEO,)))


def _identity_node(**kw) -> PlanNode:
    kw.setdefault("inputs", (Port("ref", ArtifactKind.IMAGE, required=False),))
    return PlanNode(node_id="gen", kind=NodeKind.TASK,
                    capability="video.generate.id_lock",
                    outputs=(Port("clip", ArtifactKind.VIDEO),), **kw)


def test_identity_work_without_a_rights_manifest_is_authority_missing():
    goal = _goal(objective="a scene with identity_profile:mira",
                 raw_prompt="a scene with identity_profile:mira", inputs=())
    graph = _graph((_identity_node(),), goal=goal)
    report = validate(graph, IDENTITY_CATALOG, goal)
    missing = [e for e in report.errors if e.code is ErrorCode.AUTHORITY_MISSING]
    assert [e.detail for e in missing] == [("likeness:identity_profile:mira",)]
    assert "absence is not consent" in missing[0].message


def test_an_identity_conditioned_capability_is_gated_even_with_nobody_named():
    """The capability reproduces SOMEONE: an unnamed subject is a blanket need,
    not the absence of one."""
    goal = _goal(objective="a scene", raw_prompt="a scene", inputs=())
    report = validate(_graph((_identity_node(),), goal=goal),
                      IDENTITY_CATALOG, goal)
    assert [d for e in report.errors for d in e.detail] == ["likeness:*"]


def test_a_covering_rights_manifest_clears_the_gate():
    rights = RightsManifest(authorizations=(
        Authorization(kind=AuthorityKind.LIKENESS,
                      subject="identity_profile:mira",
                      evidence="release-forms/mira-2026-08.pdf"),))
    goal = _goal(objective="a scene with identity_profile:mira",
                 raw_prompt="a scene with identity_profile:mira", inputs=(),
                 rights=rights)
    graph = _graph((_identity_node(),), goal=goal)
    assert validate(graph, IDENTITY_CATALOG, goal).ok is True


def test_a_denial_beats_a_grant():
    rights = RightsManifest(
        authorizations=(Authorization(kind=AuthorityKind.LIKENESS, subject="*",
                                      evidence="blanket-contract"),),
        denied=("likeness:identity_profile:mira",))
    goal = _goal(objective="a scene with identity_profile:mira",
                 raw_prompt="a scene with identity_profile:mira", inputs=(),
                 rights=rights)
    report = validate(_graph((_identity_node(),), goal=goal),
                      IDENTITY_CATALOG, goal)
    assert ErrorCode.AUTHORITY_MISSING in [e.code for e in report.errors]


def test_a_node_that_names_an_identity_only_in_its_params_is_still_gated():
    """The goal text is not the only place a plan can point at a person."""
    node = _identity_node(params={"reference": "identity_profile:vin"})
    goal = _goal(objective="a scene", raw_prompt="a scene", inputs=())
    report = validate(_graph((node,), goal=goal), IDENTITY_CATALOG, goal)
    subjects = [d for e in report.errors for d in e.detail]
    assert "likeness:identity_profile:vin" in subjects


def test_a_declared_authority_requirement_on_a_node_is_checked():
    node = PlanNode(node_id="j", kind=NodeKind.JOIN,
                    outputs=(Port("out", ArtifactKind.TEXT),),
                    authority_required=((AuthorityKind.DISCLOSURE, "public"),))
    report = validate(_graph((node,)), CATALOG, GOAL)
    assert _codes(report) == ["authority_missing"]
    assert report.errors[0].detail == ("disclosure:public",)


def test_ordinary_work_needs_no_authority():
    assert validate(_happy(), CATALOG, GOAL).ok is True


# ---------------------------------------------------------------------------
# [7] budget
# ---------------------------------------------------------------------------


def test_estimated_seconds_over_the_goal_budget_counts_every_candidate():
    goal = _goal(budget=BudgetHints(max_seconds=10))
    fan = PlanNode(node_id="fan", kind=NodeKind.FANOUT,
                   capability="audio.transcribe", candidates=3,
                   inputs=(Port("audio", ArtifactKind.AUDIO),),
                   outputs=(Port("t", ArtifactKind.JSON, many=True),),
                   resources=ResourceRequest(est_seconds=4))
    graph = _graph((fan,), goal=goal)
    report = validate(graph, CATALOG, goal)
    over = [e for e in report.errors if e.code is ErrorCode.BUDGET_EXCEEDED]
    assert len(over) == 1
    assert over[0].node_id is None
    assert "estimated 12s" in over[0].message


def test_a_plan_inside_the_budget_passes():
    goal = _goal(budget=BudgetHints(max_seconds=100, max_vram_gb=24))
    nodes = (_asr(resources=ResourceRequest(est_seconds=8, gpu=True, vram_gib=6)),
             _sum(resources=ResourceRequest(est_seconds=2)))
    graph = _graph(nodes, (Edge("asr", "transcript", "sum", "doc"),), goal=goal)
    report = validate(graph, CATALOG, goal)
    assert report.ok is True
    assert report.warnings == ()


def test_peak_vram_over_the_budget_names_the_node():
    goal = _goal(budget=BudgetHints(max_vram_gb=16))
    node = _asr(resources=ResourceRequest(vram_gib=24, gpu=True))
    report = validate(_graph((node,), goal=goal), CATALOG, goal)
    over = [e for e in report.errors if e.code is ErrorCode.BUDGET_EXCEEDED]
    assert [e.node_id for e in over] == ["asr"]
    assert "24 GiB" in over[0].message


def test_vram_is_peak_not_sum_so_two_fitting_nodes_pass():
    goal = _goal(budget=BudgetHints(max_vram_gb=16))
    nodes = (_asr(resources=ResourceRequest(vram_gib=10, gpu=True)),
             _sum(params={"doc": "x"},
                  resources=ResourceRequest(vram_gib=10, gpu=True)))
    assert validate(_graph(nodes, goal=goal), CATALOG, goal).ok is True


def test_an_unestimated_node_warns_instead_of_passing_quietly():
    goal = _goal(budget=BudgetHints(max_seconds=100))
    nodes = (_asr(resources=ResourceRequest(est_seconds=8)),
             _sum(params={"doc": "x"}))
    report = validate(_graph(nodes, goal=goal), CATALOG, goal)
    assert report.ok is True
    assert [w.code for w in report.warnings] == [ErrorCode.BUDGET_UNKNOWN]
    assert report.warnings[0].detail == ("sum",)


def test_no_budget_hint_means_no_budget_finding():
    node = _asr(resources=ResourceRequest(vram_gib=48, est_seconds=9000, gpu=True))
    report = validate(_graph((node,)), CATALOG, GOAL)
    assert report.ok is True and report.warnings == ()


# ---------------------------------------------------------------------------
# [8] the two invariants
# ---------------------------------------------------------------------------


SEGMENT_CATALOG = _catalog(
    _view("scene.plan.shots", produces=(ArtifactKind.JSON,)),
    _view("video.generate.t2v", accepts=(ArtifactKind.JSON,),
          produces=(ArtifactKind.VIDEO,)))


def _segment(node_id: str, **kw) -> PlanNode:
    return PlanNode(node_id=node_id, kind=NodeKind.TASK,
                    capability="video.generate.t2v",
                    inputs=(Port("spec", ArtifactKind.JSON),),
                    outputs=(Port("clip", ArtifactKind.VIDEO),),
                    params={"segment": True}, **kw)


def _lock() -> PlanNode:
    return PlanNode(node_id="lock", kind=NodeKind.TASK,
                    capability="scene.plan.shots",
                    outputs=(Port("locked", ArtifactKind.JSON),))


def test_siblings_hanging_off_a_locked_parent_validate():
    nodes = [_lock()] + [_segment(f"s{i}") for i in (1, 2, 3)]
    edges = [Edge("lock", "locked", f"s{i}", "spec") for i in (1, 2, 3)]
    assert validate(_graph(nodes, edges), SEGMENT_CATALOG, GOAL).ok is True


def test_a_segment_depending_on_a_sibling_is_a_violation():
    nodes = [_lock(), _segment("s1"), _segment("s2", depends_on=("s1",))]
    edges = [Edge("lock", "locked", "s1", "spec"),
             Edge("lock", "locked", "s2", "spec")]
    report = validate(_graph(nodes, edges), SEGMENT_CATALOG, GOAL)
    viol = [e for e in report.errors if e.code is ErrorCode.SIBLING_VIOLATION]
    assert [e.node_id for e in viol] == ["s2"]
    assert "never off each other" in viol[0].message


def test_untagged_nodes_may_chain_freely():
    a = _asr("a")
    b = _sum("b")
    assert validate(_graph((a, b), (Edge("a", "transcript", "b", "doc"),)),
                    CATALOG, GOAL).ok is True


JUDGE_CATALOG = _catalog(
    _view("video.generate.t2v", produces=(ArtifactKind.VIDEO,)),
    _view("video.evaluate.action_alignment", produces=(ArtifactKind.JSON,)))


def _gen(**kw) -> PlanNode:
    return PlanNode(node_id="gen", kind=NodeKind.FANOUT,
                    capability="video.generate.t2v", candidates=3,
                    outputs=(Port("clips", ArtifactKind.VIDEO, many=True),), **kw)


def _judge(capability: str = "video.evaluate.action_alignment", **kw) -> PlanNode:
    kw.setdefault("inputs", (Port("clips", ArtifactKind.VIDEO, many=True),))
    return PlanNode(node_id="judge", kind=NodeKind.JUDGE, capability=capability,
                    outputs=(Port("card", "scorecard"),), **kw)


def test_an_independent_judge_is_clean():
    graph = _graph((_gen(), _judge()), (Edge("gen", "clips", "judge", "clips"),))
    report = validate(graph, JUDGE_CATALOG, GOAL)
    assert report.ok is True and report.warnings == ()


def test_a_judge_on_the_generators_capability_warns_when_no_model_is_pinned():
    graph = _graph((_gen(), _judge("video.generate.t2v")),
                   (Edge("gen", "clips", "judge", "clips"),))
    report = validate(graph, JUDGE_CATALOG, GOAL)
    assert report.ok is True                     # a suspicion, not a proof
    assert [w.code for w in report.warnings] == [ErrorCode.JUDGE_IS_GENERATOR]
    assert "no model pinned at plan time" in report.warnings[0].message


def test_a_judge_pinned_to_the_generators_model_is_an_error():
    graph = _graph((_gen(params={"model_id": "wan-vace"}),
                    _judge("video.generate.t2v",
                           params={"model_id": "wan-vace"})),
                   (Edge("gen", "clips", "judge", "clips"),))
    report = validate(graph, JUDGE_CATALOG, GOAL)
    errs = [e for e in report.errors if e.code is ErrorCode.JUDGE_IS_GENERATOR]
    assert [e.node_id for e in errs] == ["judge"]
    assert "may not grade its own work" in errs[0].message


def test_a_judge_pinned_to_a_different_model_is_clean():
    graph = _graph((_gen(params={"model_id": "wan-vace"}),
                    _judge("video.generate.t2v",
                           params={"model_id": "qwen-vl"})),
                   (Edge("gen", "clips", "judge", "clips"),))
    report = validate(graph, JUDGE_CATALOG, GOAL)
    assert report.ok is True and report.warnings == ()


def test_the_generator_is_found_through_a_structural_node():
    join = PlanNode(node_id="join", kind=NodeKind.JOIN,
                    inputs=(Port("clips", ArtifactKind.VIDEO, many=True),),
                    outputs=(Port("clips", ArtifactKind.VIDEO, many=True),))
    graph = _graph((_gen(), join, _judge("video.generate.t2v")),
                   (Edge("gen", "clips", "join", "clips"),
                    Edge("join", "clips", "judge", "clips")))
    report = validate(graph, JUDGE_CATALOG, GOAL)
    assert [w.node_id for w in report.warnings] == ["judge"]


# ---------------------------------------------------------------------------
# [9] several findings at once
# ---------------------------------------------------------------------------


def test_a_thoroughly_broken_plan_reports_every_finding_without_fixing_any():
    catalog = _catalog(_view("audio.transcribe", accepts=(ArtifactKind.AUDIO,),
                             produces=(ArtifactKind.JSON,)))
    goal = _goal(inputs=(), budget=BudgetHints(max_seconds=1))
    nodes = (_asr(resources=ResourceRequest(est_seconds=30)),
             _sum(),
             PlanNode(node_id="x", kind=NodeKind.TASK, capability="a.b",
                      outputs=(Port("o", ArtifactKind.TEXT),)))
    graph = _graph(nodes, (Edge("asr", "transcript", "sum", "nope"),), goal=goal)
    report = validate(graph, catalog, goal)
    found = set(_codes(report))
    assert {"dangling_edge", "unknown_capability", "missing_required_input",
            "budget_exceeded"} <= found
    assert report.ok is False
    assert graph.node_ids == ("asr", "sum", "x")     # nothing was rewritten
