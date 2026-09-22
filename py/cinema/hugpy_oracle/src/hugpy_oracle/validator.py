"""Static plan validation (k103): doc §6 step 7 — "validate every port, artifact,
parameter, authority gate, budget, resource request, and evaluator" — BEFORE
anything is quoted (step 8) and long before anything executes (step 9).

This is the cheap half of routing. It is PURE: no catalog build, no worker read,
no model call, no disk, no clock. The catalog is handed in as a
``Mapping[str, CapabilityView]`` (``{v.name: v for v in list_capabilities()}``
at the call site) precisely so this module cannot accidentally acquire an I/O
dependency and so tests can hand it three fake rows.

It NEVER auto-fixes. A validator that quietly rewrites a plan is a planner with
no audit trail; the operator gets a typed list of what is wrong, and the caller
decides whether to replan, refuse, or ask for a release form.

The finding vocabulary:

  CYCLE                  the plan is not a DAG
  DUPLICATE_NODE         two nodes share an id (identity is the id)
  DANGLING_EDGE          an edge/dependency names a node or port that is absent
  UNKNOWN_CAPABILITY     the capability is not in the catalog at all
  CAPABILITY_GAP         it IS in the catalog and has no eligible implementation
                         — carrying the catalog's own reasons, because §4's
                         rule is "return CAPABILITY_GAP, never hallucinate an
                         adapter", and a gap with no reason is a shrug
  PORT_TYPE_MISMATCH     an edge joins incompatible kinds/cardinality, or a
                         node's ports contradict its capability's declared I/O
  MISSING_REQUIRED_INPUT a required input port is fed by nothing
  MAP_WITHOUT_MANY       a map node that is not mapping over a collection
  AUTHORITY_MISSING      k97's gate says the request has no grant for a
                         (kind, subject) this node needs
  BUDGET_EXCEEDED        the plan's own estimates exceed the goal's budget
  BUDGET_UNKNOWN         (warning) a budget hint exists but nodes carry no
                         estimate for it — the check could not be made, and
                         saying so is better than passing quietly
  SIBLING_VIOLATION      invariant 9 / Stage 14: one segment depends on another
  JUDGE_IS_GENERATOR     invariant 11: the judge is the generator it judges

Order is deterministic: checks run in a fixed sequence and each iterates nodes
and edges in declaration order, so two runs over the same plan produce the same
report byte for byte.

No pathlib anywhere. os.path only (not that this module touches the disk).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from hugpy_oracle import authority
from hugpy_oracle.contracts import ArtifactKind, AuthorityKind, CapabilityView, GoalSpec
from hugpy_oracle.plan import (
    CycleError,
    NodeKind,
    PlanGraph,
    PlanNode,
    canonical_json,
    sibling_violations,
)


class ErrorCode(str, Enum):
    CYCLE = "cycle"
    DUPLICATE_NODE = "duplicate_node"
    DANGLING_EDGE = "dangling_edge"
    UNKNOWN_CAPABILITY = "unknown_capability"
    CAPABILITY_GAP = "capability_gap"
    PORT_TYPE_MISMATCH = "port_type_mismatch"
    MISSING_REQUIRED_INPUT = "missing_required_input"
    MAP_WITHOUT_MANY = "map_without_many"
    AUTHORITY_MISSING = "authority_missing"
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_UNKNOWN = "budget_unknown"
    SIBLING_VIOLATION = "sibling_violation"
    JUDGE_IS_GENERATOR = "judge_is_generator"


@dataclass(frozen=True, slots=True)
class ValidationError:
    """One finding. ``node_id`` is None for whole-graph findings (a cycle, a
    total-runtime budget). ``detail`` carries machine-readable evidence the
    message renders in prose — notably the CATALOG'S OWN reasons on a
    CAPABILITY_GAP, so the caller can show the operator why without re-deriving
    it."""
    code: ErrorCode
    node_id: str | None
    message: str
    detail: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("ValidationError.message must be non-empty — an "
                             "unexplained finding is not a finding")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "node_id": self.node_id,
                "message": self.message, "detail": list(self.detail)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ValidationError":
        return cls(code=ErrorCode(d["code"]), node_id=d.get("node_id"),
                   message=d["message"], detail=tuple(d.get("detail", ())))


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """``ok`` is derived, not asserted: a report that claims ok with errors (or
    the reverse) is incoherent and refused at construction, the same way
    ``AuthorityDecision`` refuses one."""
    ok: bool
    errors: tuple[ValidationError, ...] = ()
    warnings: tuple[ValidationError, ...] = ()

    def __post_init__(self) -> None:
        if self.ok != (not self.errors):
            raise ValueError(
                f"ValidationReport.ok={self.ok} contradicts "
                f"{len(self.errors)} error(s)")

    def codes(self) -> tuple[ErrorCode, ...]:
        return tuple(e.code for e in self.errors)

    def warning_codes(self) -> tuple[ErrorCode, ...]:
        return tuple(w.code for w in self.warnings)

    def for_node(self, node_id: str) -> tuple[ValidationError, ...]:
        return tuple(e for e in self.errors + self.warnings
                     if e.node_id == node_id)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok,
                "errors": [e.to_dict() for e in self.errors],
                "warnings": [w.to_dict() for w in self.warnings]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ValidationReport":
        return cls(ok=bool(d["ok"]),
                   errors=tuple(ValidationError.from_dict(e)
                                for e in d.get("errors", ())),
                   warnings=tuple(ValidationError.from_dict(w)
                                  for w in d.get("warnings", ())))


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class _Findings:
    __slots__ = ("errors", "warnings")

    def __init__(self) -> None:
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

    def err(self, code: ErrorCode, node_id: str | None, message: str,
            detail: Iterable[str] = ()) -> None:
        self.errors.append(ValidationError(code, node_id, message, tuple(detail)))

    def warn(self, code: ErrorCode, node_id: str | None, message: str,
             detail: Iterable[str] = ()) -> None:
        self.warnings.append(ValidationError(code, node_id, message, tuple(detail)))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_duplicates(graph: PlanGraph, out: _Findings) -> None:
    seen: dict[str, int] = {}
    for n in graph.nodes:
        seen[n.node_id] = seen.get(n.node_id, 0) + 1
    for node_id, count in seen.items():
        if count > 1:
            out.err(ErrorCode.DUPLICATE_NODE, node_id,
                    f"{node_id}: declared {count} times — a node's identity IS "
                    f"its id (revisions and caches depend on it)")


def _check_dangling(graph: PlanGraph, out: _Findings) -> None:
    known = {n.node_id: n for n in graph.nodes}
    for n in graph.nodes:
        for dep in n.depends_on:
            if dep not in known:
                out.err(ErrorCode.DANGLING_EDGE, n.node_id,
                        f"{n.node_id}: depends_on names unknown node {dep!r}")
    for e in graph.edges:
        src = known.get(e.src_node)
        dst = known.get(e.dst_node)
        if src is None:
            out.err(ErrorCode.DANGLING_EDGE, None,
                    f"edge {e.src_node}.{e.src_port} -> {e.dst_node}.{e.dst_port}: "
                    f"unknown source node {e.src_node!r}")
        elif src.output_port(e.src_port) is None:
            out.err(ErrorCode.DANGLING_EDGE, e.src_node,
                    f"{e.src_node} has no output port {e.src_port!r} "
                    f"(has: {[p.name for p in src.outputs]})")
        if dst is None:
            out.err(ErrorCode.DANGLING_EDGE, None,
                    f"edge {e.src_node}.{e.src_port} -> {e.dst_node}.{e.dst_port}: "
                    f"unknown destination node {e.dst_node!r}")
        elif dst.input_port(e.dst_port) is None:
            out.err(ErrorCode.DANGLING_EDGE, e.dst_node,
                    f"{e.dst_node} has no input port {e.dst_port!r} "
                    f"(has: {[p.name for p in dst.inputs]})")


def _check_acyclic(graph: PlanGraph, out: _Findings) -> bool:
    try:
        graph.topological_order()
    except CycleError as exc:
        out.err(ErrorCode.CYCLE, None,
                f"plan is not a DAG; cycle among nodes: {', '.join(exc.nodes)}",
                detail=exc.nodes)
        return False
    return True


def _check_capabilities(graph: PlanGraph,
                        catalog_view: Mapping[str, CapabilityView],
                        out: _Findings) -> None:
    for n in graph.nodes:
        if not n.capability:
            continue
        view = catalog_view.get(n.capability)
        if view is None:
            out.err(ErrorCode.UNKNOWN_CAPABILITY, n.node_id,
                    f"{n.node_id}: capability {n.capability!r} is not in the "
                    f"catalog — the planner must not invent one")
            continue
        if not view.eligibility.eligible:
            reasons = tuple(view.eligibility.reasons)
            out.err(ErrorCode.CAPABILITY_GAP, n.node_id,
                    f"{n.node_id}: {n.capability} has no eligible "
                    f"implementation — " + "; ".join(reasons),
                    detail=reasons)
        _check_node_io_against_view(n, view, out)


def _check_node_io_against_view(node: PlanNode, view: CapabilityView,
                                out: _Findings) -> None:
    """A node's declared ports must not contradict the capability's declared
    I/O. Only ``ArtifactKind`` ports are checked: a logical artifact type the
    catalog has no vocabulary for yet (``dialogue_timeline``) is not evidence
    of a mismatch, and an empty accepts/produces means "undeclared", not
    "nothing"."""
    if view.accepts:
        allowed = {k.value for k in view.accepts}
        for p in node.inputs:
            if isinstance(p.artifact_kind, ArtifactKind) and p.kind_value not in allowed:
                out.err(ErrorCode.PORT_TYPE_MISMATCH, node.node_id,
                        f"{node.node_id}.{p.name}: {view.name} does not accept "
                        f"{p.kind_value!r} (accepts: {sorted(allowed)})")
    if view.produces:
        allowed = {k.value for k in view.produces}
        for p in node.outputs:
            if isinstance(p.artifact_kind, ArtifactKind) and p.kind_value not in allowed:
                out.err(ErrorCode.PORT_TYPE_MISMATCH, node.node_id,
                        f"{node.node_id}.{p.name}: {view.name} does not produce "
                        f"{p.kind_value!r} (produces: {sorted(allowed)})")


def _check_edge_types(graph: PlanGraph, out: _Findings) -> None:
    known = {n.node_id: n for n in graph.nodes}
    for e in graph.edges:
        src = known.get(e.src_node)
        dst = known.get(e.dst_node)
        if src is None or dst is None:
            continue                      # already a DANGLING_EDGE
        sp = src.output_port(e.src_port)
        dp = dst.input_port(e.dst_port)
        if sp is None or dp is None:
            continue                      # already a DANGLING_EDGE
        if sp.kind_value != dp.kind_value:
            out.err(ErrorCode.PORT_TYPE_MISMATCH, e.dst_node,
                    f"{e.src_node}.{e.src_port} produces {sp.kind_value!r} but "
                    f"{e.dst_node}.{e.dst_port} accepts {dp.kind_value!r}")
        elif sp.many and not dp.many:
            # The reverse (single -> many) is the legitimate FAN-IN shape: N
            # candidate edges collecting into one judge/reduce port.
            out.err(ErrorCode.PORT_TYPE_MISMATCH, e.dst_node,
                    f"{e.src_node}.{e.src_port} produces MANY {sp.kind_value!r} "
                    f"but {e.dst_node}.{e.dst_port} takes one — go through a "
                    f"map or reduce node")


def _check_inputs(graph: PlanGraph, goal: GoalSpec, out: _Findings) -> None:
    """A required input port must be fed by an edge, or bound to something the
    request already carries: a supplied ``InputRef`` of the same kind, or a
    literal in the node's own params under the port's name."""
    goal_kinds = {i.kind.value for i in goal.inputs}
    for n in graph.nodes:
        fed = {e.dst_port for e in graph.incoming(n.node_id)}
        for p in n.inputs:
            if not p.required or p.name in fed:
                continue
            if p.name in n.params:
                continue
            if p.kind_value in goal_kinds:
                continue
            out.err(ErrorCode.MISSING_REQUIRED_INPUT, n.node_id,
                    f"{n.node_id}.{p.name} ({p.kind_value}) is required but "
                    f"nothing feeds it: no incoming edge, no params binding, "
                    f"and the goal supplies no {p.kind_value} input")


def _check_map(graph: PlanGraph, out: _Findings) -> None:
    for n in graph.nodes:
        if n.kind is NodeKind.MAP:
            if not n.map_over:
                out.err(ErrorCode.MAP_WITHOUT_MANY, n.node_id,
                        f"{n.node_id}: a map node must name the input port it "
                        f"maps over (map_over)")
                continue
            port = n.input_port(n.map_over)
            if port is None:
                out.err(ErrorCode.MAP_WITHOUT_MANY, n.node_id,
                        f"{n.node_id}: map_over names {n.map_over!r}, which is "
                        f"not an input port (has: {[p.name for p in n.inputs]})")
            elif not port.many:
                out.err(ErrorCode.MAP_WITHOUT_MANY, n.node_id,
                        f"{n.node_id}: map_over port {port.name!r} is not a "
                        f"collection (many=False) — there is nothing to map over")
        elif n.map_over:
            out.err(ErrorCode.MAP_WITHOUT_MANY, n.node_id,
                    f"{n.node_id}: map_over is only meaningful on a map node, "
                    f"not a {n.kind.value} node")


def _node_authority_requirements(
        node: PlanNode, goal: GoalSpec) -> tuple[tuple[AuthorityKind, str], ...]:
    """Everything this node needs a grant for: k97's capability+request table,
    plus any identity/voice reference the NODE ITSELF names in its params (the
    goal text is not the only place a plan can point at a person), plus the
    pairs the node declares outright."""
    pairs: list[tuple[AuthorityKind, str]] = []

    def _add(kind: AuthorityKind, subject: str) -> None:
        if (kind, subject) not in pairs:
            pairs.append((kind, subject))

    if node.capability:
        for kind, subject in authority.required_authorities(node.capability, goal):
            _add(kind, subject)
    for prefix, subject in authority.find_subject_refs(
            canonical_json(node.params.to_dict())):
        _add(AuthorityKind.LIKENESS if prefix == "identity_profile"
             else AuthorityKind.VOICE, subject)
    for kind, subject in node.authority_required:
        _add(kind, subject)
    return tuple(pairs)


def _check_authority(graph: PlanGraph, goal: GoalSpec, out: _Findings) -> None:
    rights = goal.rights
    for n in graph.nodes:
        for kind, subject in _node_authority_requirements(n, goal):
            if rights is not None and rights.covers(kind, subject):
                continue
            why = ("no RightsManifest on the request — absence is not consent"
                   if rights is None else
                   "the request's RightsManifest does not cover it")
            out.err(ErrorCode.AUTHORITY_MISSING, n.node_id,
                    f"{n.node_id}: needs {kind.value} authorization for "
                    f"{subject!r}; {why}",
                    detail=(f"{kind.value}:{subject}",))


def _check_budget(graph: PlanGraph, goal: GoalSpec, out: _Findings) -> None:
    """Against ``goal.budget`` (``BudgetHints``): total estimated seconds and
    PEAK vram. Peak, not sum, because this fleet runs one heavy node at a time —
    summing VRAM would refuse plans that fit. A fan-out of N candidates costs N
    times its estimate; a map node's fan width is unknown at plan time, so its
    contribution is a lower bound and the message says so.

    Naming note: ``ResourceRequest.vram_gib`` and ``BudgetHints.max_vram_gb``
    are compared numerically as the same unit (a k101 follow-up should unify
    the spelling)."""
    budget = goal.budget
    if budget.max_seconds is not None:
        known = [n for n in graph.nodes if n.resources.est_seconds is not None]
        unknown = [n.node_id for n in graph.nodes
                   if n.resources.est_seconds is None]
        total = sum(float(n.resources.est_seconds) * max(1, n.candidates)
                    for n in known)
        has_map = any(n.kind is NodeKind.MAP for n in graph.nodes)
        if total > budget.max_seconds:
            lower = " (a lower bound: map fan width is unknown at plan time)" \
                if has_map else ""
            out.err(ErrorCode.BUDGET_EXCEEDED, None,
                    f"estimated {total:g}s{lower} exceeds the goal's "
                    f"max_seconds={budget.max_seconds:g}")
        elif unknown:
            out.warn(ErrorCode.BUDGET_UNKNOWN, None,
                     f"max_seconds={budget.max_seconds:g} could not be fully "
                     f"checked: {len(unknown)} node(s) carry no est_seconds "
                     f"({', '.join(unknown)})",
                     detail=tuple(unknown))
    if budget.max_vram_gb is not None:
        over = [n for n in graph.nodes
                if n.resources.vram_gib is not None
                and n.resources.vram_gib > budget.max_vram_gb]
        for n in over:
            out.err(ErrorCode.BUDGET_EXCEEDED, n.node_id,
                    f"{n.node_id}: needs {n.resources.vram_gib:g} GiB VRAM, "
                    f"over the goal's max_vram_gb={budget.max_vram_gb:g}")
        if not over:
            unknown = [n.node_id for n in graph.nodes
                       if n.resources.gpu and n.resources.vram_gib is None]
            if unknown:
                out.warn(ErrorCode.BUDGET_UNKNOWN, None,
                         f"max_vram_gb={budget.max_vram_gb:g} could not be fully "
                         f"checked: {len(unknown)} GPU node(s) declare no "
                         f"vram_gib ({', '.join(unknown)})",
                         detail=tuple(unknown))


def _check_siblings(graph: PlanGraph, out: _Findings) -> None:
    segments = graph.segment_node_ids()
    if len(segments) < 2:
        return
    for child, parent in sibling_violations(graph, segments):
        out.err(ErrorCode.SIBLING_VIOLATION, child,
                f"{child} depends on sibling segment {parent} — segments must "
                f"hang off the LOCKED artifacts, never off each other "
                f"(Stage 14: locked -> S1, S2, S3; never S1 -> S2 -> S3)")


def _generator_sources(graph: PlanGraph, node_id: str) -> tuple[PlanNode, ...]:
    """The nearest capability-bearing nodes upstream of ``node_id``, looking
    THROUGH structural nodes (a judge fed by ``fanout -> join`` still judges the
    fanout). Visited-set walk: safe on a cyclic plan."""
    known = {n.node_id: n for n in graph.nodes}
    found: list[PlanNode] = []
    seen: set[str] = {node_id}
    frontier = [e.src_node for e in graph.incoming(node_id)]
    while frontier:
        cur = frontier.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        node = known.get(cur)
        if node is None:
            continue
        if node.capability:
            found.append(node)
        else:
            frontier.extend(e.src_node for e in graph.incoming(cur))
    return tuple(found)


def _check_judges(graph: PlanGraph, out: _Findings) -> None:
    """Invariant 11 — the judge must not be the generator.

    Same capability AND the same pinned model is a hard error. Same capability
    with the model unpinned is a WARNING, not an error: at plan time the router
    has not picked a model yet, so "same capability" is a suspicion, not a
    proof — and refusing a plan on a suspicion would make the honest thing
    (planning by capability, not by checkpoint) impossible."""
    for n in graph.nodes:
        if n.kind is not NodeKind.JUDGE:
            continue
        judge_model = n.model_hint()
        for src in _generator_sources(graph, n.node_id):
            src_model = src.model_hint()
            same_cap = src.capability == n.capability
            same_model = (judge_model is not None and src_model is not None
                          and judge_model == src_model)
            if same_model:
                out.err(ErrorCode.JUDGE_IS_GENERATOR, n.node_id,
                        f"{n.node_id} judges {src.node_id} with the same model "
                        f"{judge_model!r}" +
                        (f" and capability {n.capability!r}" if same_cap else "") +
                        " — a generator may not grade its own work")
            elif same_cap and (judge_model is None or src_model is None):
                out.warn(ErrorCode.JUDGE_IS_GENERATOR, n.node_id,
                         f"{n.node_id} judges {src.node_id} with the same "
                         f"capability {n.capability!r} and no model pinned at "
                         f"plan time — the router must not resolve both to the "
                         f"same model")


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def validate(graph: PlanGraph, catalog_view: Mapping[str, CapabilityView],
             goal: GoalSpec) -> ValidationReport:
    """Statically check ``graph`` against the catalog and the goal.

    Pure, offline, deterministic, and it never modifies anything. ``ok`` means
    every check passed; warnings are advisory and never flip it."""
    out = _Findings()
    _check_duplicates(graph, out)
    _check_dangling(graph, out)
    _check_acyclic(graph, out)
    _check_capabilities(graph, catalog_view, out)
    _check_edge_types(graph, out)
    _check_inputs(graph, goal, out)
    _check_map(graph, out)
    _check_authority(graph, goal, out)
    _check_budget(graph, goal, out)
    _check_siblings(graph, out)
    _check_judges(graph, out)
    return ValidationReport(ok=not out.errors, errors=tuple(out.errors),
                            warnings=tuple(out.warnings))


__all__ = [
    "ErrorCode",
    "ValidationError",
    "ValidationReport",
    "validate",
]
