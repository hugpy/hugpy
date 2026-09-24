"""PlanGraph (k103): the typed graph the oracle plans IN — and nothing that runs it.

Doc §3.3 asks for a typed DAG with ports, dependencies, map/fan-out/judge/
reduce/join/assemble nodes, approval gates, retry/timeout/fallback policy,
resource requests, cache + idempotency keys, per-node acceptance tests and
"bounded repair loops represented as controlled graph revisions". The keeper
assessment defers the *runtime* for that graph (Wave 4 / k111: leases,
heartbeats, resume) until a real recipe outgrows the FAT-orchestrator pattern.

Both of those are true at once, and this module is what falls out: the graph is
a CONTRACT before it is an engine. The Wave-3 FAT orchestrator
(``runners/performance.py``) can emit one and be statically checked; the
eventual DAG runtime consumes the same objects without a rewrite; the repair
controller gets ``revise()`` — a new revision that keeps every untouched node's
identity — instead of inventing its own mutation story later.

Everything here is pure, offline and deterministic, in the contracts idiom
(``oracle/contracts.py``): frozen slotted dataclasses, closed vocabularies as
str-Enums, tuples for collections, structurally-invalid values raised in
``__post_init__``, lossless ``to_dict``/``from_dict``. Two additions of its own:

  * ``FrozenParams`` — a hashable, JSON-safe, recursively frozen Mapping, so a
    node can carry ``params`` and still be a frozen dataclass you can put in a
    set. ``node.params["segment"]`` reads like a dict; ``hash(node)`` works.
  * ``plan_digest()`` — sha256 over the canonical JSON of the whole graph. Two
    structurally identical plans hash identically regardless of dict ordering,
    which is what makes a plan quotable (§6 step 8) and cacheable.

What is deliberately NOT here: execution, scheduling, state, journalling,
model selection, and any notion of "current" node. A PlanGraph never knows
whether it ran. Static checking lives next door in ``oracle/validator.py`` so
that the contract stays importable by anything (routes, tests, the agent)
without dragging the catalog in.

No pathlib anywhere. os.path only (not that this module touches the disk).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping

from hugpy_oracle.contracts import (
    ArtifactKind,
    AuthorityKind,
    BudgetHints,
    CheckKind,
    FailureClass,
    GoalSpec,
    PlannerMode,
    RepairCode,
)

# The params key that marks a node as one of Stage 14's sibling segments. It is
# a param and not a NodeKind on purpose: "is a segment" is a property of the
# work, orthogonal to whether the node is a task, a map or a fan-out.
SEGMENT_PARAM = "segment"

# ---------------------------------------------------------------------------
# Planner-mode gate (k113; POLICY-rights-consent-disclosure §3). A capability is
# FRONTIER-BOUND when its work is done by Frontier Keeper A rather than on this
# fleet: its catalog name lives under ``frontier.`` or the node's params say
# ``frontier: True``. Under ``PlannerMode.LOCAL_ONLY`` a plan may carry ZERO such
# nodes — not "they run locally instead", but a typed REFUSED outcome naming
# each one, because silently re-homing the work would make the echoed
# ``planner_mode`` a lie (invariant 8). Whether the fleet has a frontier at all
# is an OPERATOR switch (``HUGPY_FRONTIER_ENABLED``), default off: a frontier-
# disabled fleet cannot produce a ``frontier`` plan, whatever the request said.
# ---------------------------------------------------------------------------

FRONTIER_CAPABILITY_PREFIX = "frontier."
FRONTIER_PARAM = "frontier"
FRONTIER_ENABLED_ENV = "HUGPY_FRONTIER_ENABLED"
_TRUTHY = ("1", "true", "yes", "on")


def frontier_enabled() -> bool:
    """Is Frontier Keeper A wired into THIS fleet? Read from the environment on
    every call (no cache) so a test or an operator toggle takes effect at once.
    Default False — the truthful description of the fleet today."""
    return os.environ.get(FRONTIER_ENABLED_ENV, "").strip().lower() in _TRUTHY


def effective_planner_mode(requested: PlannerMode | str | None) -> PlannerMode:
    """The planner mode a response may TRUTHFULLY claim. ``frontier`` only when
    it was asked for AND the fleet has a frontier; an unknown or absent value,
    or a frontier request on a frontier-disabled fleet, is ``local_only``."""
    try:
        mode = PlannerMode(requested or PlannerMode.LOCAL_ONLY)
    except ValueError:
        return PlannerMode.LOCAL_ONLY
    if mode is PlannerMode.FRONTIER and not frontier_enabled():
        return PlannerMode.LOCAL_ONLY
    return mode


def is_frontier_capability(capability: str | None,
                           params: Mapping[str, Any] | None = None) -> bool:
    """Does this capability (or this node's params) name frontier work?"""
    if capability and capability.startswith(FRONTIER_CAPABILITY_PREFIX):
        return True
    return bool(params) and params.get(FRONTIER_PARAM) is True


# ---------------------------------------------------------------------------
# Canonical JSON + digests
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """The one canonical encoding used for every digest in this module: keys
    sorted, no insignificant whitespace. Same trick as
    ``ExecutionReceipt.normalize_request`` — identical content encodes
    identically no matter how the dicts were built."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def content_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def goal_digest(goal: GoalSpec) -> str:
    """The digest a ``PlanGraph`` pins itself to. A plan whose ``goal_digest``
    no longer matches the goal it is answering is stale by construction — the
    check is cheap and it is the only tie between the two contracts."""
    return content_digest(goal.to_dict())


# ---------------------------------------------------------------------------
# FrozenParams — a hashable, JSON-safe mapping
# ---------------------------------------------------------------------------


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenParams):
        return value
    if isinstance(value, _MappingABC):
        return FrozenParams(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, Enum):          # str-Enums serialize to their value
        return value.value
    raise TypeError(
        f"plan params must be JSON-safe (str/int/float/bool/None/list/dict), "
        f"got {type(value).__name__}: {value!r}")


def _thaw_value(value: Any) -> Any:
    if isinstance(value, FrozenParams):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw_value(v) for v in value]
    return value


class FrozenParams(_MappingABC):
    """An immutable, recursively frozen, hashable ``Mapping[str, JSON]``.

    ``collections.abc.Mapping`` sets ``__hash__ = None``; a frozen dataclass
    that carries params still has to be hashable (the repair controller wants
    node sets), so the hash is defined here over the canonical JSON. Reading is
    ordinary mapping access, iteration is key-sorted so anything built from it
    is deterministic without the caller remembering to sort."""

    __slots__ = ("_data", "_json")

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        items: dict[str, Any] = {}
        for key, value in dict(data or {}).items():
            if not isinstance(key, str):
                raise TypeError(f"param keys must be str, got {key!r}")
            items[key] = _freeze_value(value)
        object.__setattr__(self, "_data", items)
        object.__setattr__(
            self, "_json",
            canonical_json({k: _thaw_value(v) for k, v in items.items()}))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenParams is immutable")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FrozenParams):
            return self._json == other._json
        if isinstance(other, _MappingABC):
            return self.to_dict() == {k: _thaw_value(_freeze_value(v))
                                      for k, v in other.items()}
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._json)

    def __repr__(self) -> str:
        return f"FrozenParams({self._json})"

    def to_dict(self) -> dict[str, Any]:
        return {k: _thaw_value(v) for k, v in sorted(self._data.items())}


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


from hugpy_oracle.contracts import coerce_artifact_kind


from hugpy_oracle.contracts import kind_value


@dataclass(frozen=True, slots=True)
class Port:
    """One typed slot on a node. ``many=True`` marks a COLLECTION port: a
    fan-out's output, a judge's candidate input, a map's iterable input."""
    name: str
    artifact_kind: ArtifactKind | str
    required: bool = True
    many: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Port.name must be non-empty")
        object.__setattr__(self, "artifact_kind",
                           coerce_artifact_kind(self.artifact_kind))

    @property
    def kind_value(self) -> str:
        return kind_value(self.artifact_kind)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "artifact_kind": self.kind_value,
                "required": self.required, "many": self.many}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Port":
        return cls(name=d["name"], artifact_kind=d["artifact_kind"],
                   required=bool(d.get("required", True)),
                   many=bool(d.get("many", False)))


# ---------------------------------------------------------------------------
# Node vocabulary + policies
# ---------------------------------------------------------------------------


class NodeKind(str, Enum):
    """Doc §3.3 / §5's node vocabulary, and nothing beyond it."""
    TASK = "task"           # one capability, one execution
    MAP = "map"             # run the capability once per element of map_over
    FANOUT = "fanout"       # N candidates of the SAME work (candidates=N)
    JUDGE = "judge"         # score candidates — never the generator (inv. 11)
    REDUCE = "reduce"       # select/merge scored candidates down to one
    GATE = "gate"           # typed operator approval / conditional
    JOIN = "join"           # wait for several branches, pass them on
    ASSEMBLE = "assemble"   # build the deliverable from ACCEPTED artifacts


# Which kinds name a capability. A capability-less task node has nothing to
# route; a capability ON a join is a lie about where the work happens.
CAPABILITY_KINDS: frozenset[NodeKind] = frozenset(
    {NodeKind.TASK, NodeKind.MAP, NodeKind.FANOUT, NodeKind.JUDGE})
STRUCTURAL_KINDS: frozenset[NodeKind] = frozenset(
    {NodeKind.JOIN, NodeKind.GATE, NodeKind.REDUCE, NodeKind.ASSEMBLE})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded, declared up front. ``fallback`` names another CAPABILITY to try
    after the attempts are spent (doc §4 rule 9's "declared fallback policy") —
    never a model, which is the router's business."""
    max_attempts: int = 1
    timeout_s: float | None = None
    fallback: str | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")
        if self.fallback is not None and "." not in self.fallback:
            raise ValueError(
                f"RetryPolicy.fallback must be a namespaced capability name, "
                f"got {self.fallback!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"max_attempts": self.max_attempts, "timeout_s": self.timeout_s,
                "fallback": self.fallback}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RetryPolicy":
        return cls(max_attempts=int(d.get("max_attempts", 1)),
                   timeout_s=d.get("timeout_s"), fallback=d.get("fallback"))


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """What the node expects to need. None means UNKNOWN, never zero — the
    legacy registry carries no VRAM envelope and fabricating one would defeat
    the budget check that reads these."""
    vram_gib: float | None = None
    ram_gib: float | None = None
    gpu: bool = False
    est_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, val in (("vram_gib", self.vram_gib), ("ram_gib", self.ram_gib),
                          ("est_seconds", self.est_seconds)):
            if val is not None and val < 0:
                raise ValueError(f"ResourceRequest.{name} must be >= 0, got {val}")

    def to_dict(self) -> dict[str, Any]:
        return {"vram_gib": self.vram_gib, "ram_gib": self.ram_gib,
                "gpu": self.gpu, "est_seconds": self.est_seconds}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ResourceRequest":
        return cls(vram_gib=d.get("vram_gib"), ram_gib=d.get("ram_gib"),
                   gpu=bool(d.get("gpu", False)),
                   est_seconds=d.get("est_seconds"))


@dataclass(frozen=True, slots=True)
class AcceptanceTest:
    """A per-node gate declared AT PLAN TIME (doc §3.3 "per-node acceptance
    tests"): which evidence axis must hold, at what threshold, and which repair
    code a failure produces — so the repair controller has the mapping before
    anything runs instead of guessing from a stack trace.

    ``kind`` is a ``CheckKind`` when it names one of the evidence axes, else a
    free check name (``"duration_within_window"``) resolved by the evaluator."""
    kind: CheckKind | str
    threshold: float | str | None = None
    repair_code: RepairCode | None = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, CheckKind):
            return
        text = str(self.kind).strip()
        if not text:
            raise ValueError("AcceptanceTest.kind must be non-empty")
        try:
            object.__setattr__(self, "kind", CheckKind(text))
        except ValueError:
            object.__setattr__(self, "kind", text)

    @property
    def kind_value(self) -> str:
        return self.kind.value if isinstance(self.kind, CheckKind) else str(self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind_value, "threshold": self.threshold,
                "repair_code": self.repair_code.value if self.repair_code else None}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AcceptanceTest":
        code = d.get("repair_code")
        return cls(kind=d["kind"], threshold=d.get("threshold"),
                   repair_code=RepairCode(code) if code else None)


# ---------------------------------------------------------------------------
# Nodes and edges
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlanNode:
    """One unit of planned work.

    Identity is ``node_id`` and only ``node_id``: a revision that leaves a node
    untouched leaves the SAME id in place, which is how a cache hit, a receipt
    and a completed expensive node survive a bounded repair (doc §3.3, §5).

    ``depends_on`` carries ORDERING that no artifact flows along (a lock, an
    approval); real data flow is an ``Edge`` between typed ports. Both feed the
    topological order."""
    node_id: str
    kind: NodeKind
    capability: str | None = None
    inputs: tuple[Port, ...] = ()
    outputs: tuple[Port, ...] = ()
    params: Mapping[str, Any] = field(default_factory=FrozenParams)
    depends_on: tuple[str, ...] = ()
    map_over: str | None = None
    candidates: int = 1
    approval_gate: bool = False
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    acceptance: tuple[AcceptanceTest, ...] = ()
    cache_key: str | None = None
    idempotency_key: str | None = None
    authority_required: tuple[tuple[AuthorityKind, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("PlanNode.node_id must be non-empty")
        object.__setattr__(self, "params", FrozenParams(self.params))
        if self.kind in CAPABILITY_KINDS and not self.capability:
            raise ValueError(
                f"{self.node_id}: a {self.kind.value} node must name a "
                f"capability (that is the thing the router resolves)")
        if self.kind in STRUCTURAL_KINDS and self.capability:
            raise ValueError(
                f"{self.node_id}: a {self.kind.value} node must NOT name a "
                f"capability ({self.capability!r}) — structural nodes do no "
                f"model work")
        if self.capability is not None and "." not in self.capability:
            raise ValueError(
                f"{self.node_id}: capability must be a namespaced catalog name, "
                f"got {self.capability!r}")
        if self.candidates < 1:
            raise ValueError(f"{self.node_id}: candidates must be >= 1")
        for label, ports in (("inputs", self.inputs), ("outputs", self.outputs)):
            if not isinstance(ports, tuple) or not all(isinstance(p, Port)
                                                       for p in ports):
                raise TypeError(
                    f"{self.node_id}: {label} must be a TUPLE of Port "
                    f"(a lone Port is the missing-comma bug), got {ports!r}")
            names = [p.name for p in ports]
            if len(set(names)) != len(names):
                raise ValueError(f"{self.node_id}: duplicate {label} port names "
                                 f"{sorted(n for n in names if names.count(n) > 1)}")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"{self.node_id}: duplicate depends_on entries")
        if self.node_id in self.depends_on:
            raise ValueError(f"{self.node_id}: a node cannot depend on itself")

    # -- lookups -----------------------------------------------------------

    def input_port(self, name: str) -> Port | None:
        for p in self.inputs:
            if p.name == name:
                return p
        return None

    def output_port(self, name: str) -> Port | None:
        for p in self.outputs:
            if p.name == name:
                return p
        return None

    @property
    def is_segment(self) -> bool:
        """Stage 14 tag: this node produces one of the sibling segments that
        must not depend on each other."""
        return self.params.get(SEGMENT_PARAM) is True

    def model_hint(self) -> str | None:
        """The model this node is PINNED to at plan time, if any. Usually None:
        the router picks the model, and invariant 11 (judge != generator) can
        then only be enforced at execution. Recorded here when a plan does pin
        one, so the static validator can enforce it early."""
        for key in ("model_id", "model"):
            value = self.params.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    # -- wire shape --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "capability": self.capability,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "params": FrozenParams(self.params).to_dict(),
            "depends_on": list(self.depends_on),
            "map_over": self.map_over,
            "candidates": self.candidates,
            "approval_gate": self.approval_gate,
            "retry": self.retry.to_dict(),
            "resources": self.resources.to_dict(),
            "acceptance": [a.to_dict() for a in self.acceptance],
            "cache_key": self.cache_key,
            "idempotency_key": self.idempotency_key,
            "authority_required": [{"kind": k.value, "subject": s}
                                   for k, s in self.authority_required],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PlanNode":
        return cls(
            node_id=d["node_id"],
            kind=NodeKind(d["kind"]),
            capability=d.get("capability"),
            inputs=tuple(Port.from_dict(p) for p in d.get("inputs", ())),
            outputs=tuple(Port.from_dict(p) for p in d.get("outputs", ())),
            params=FrozenParams(d.get("params") or {}),
            depends_on=tuple(d.get("depends_on", ())),
            map_over=d.get("map_over"),
            candidates=int(d.get("candidates", 1)),
            approval_gate=bool(d.get("approval_gate", False)),
            retry=RetryPolicy.from_dict(d.get("retry") or {}),
            resources=ResourceRequest.from_dict(d.get("resources") or {}),
            acceptance=tuple(AcceptanceTest.from_dict(a)
                             for a in d.get("acceptance", ())),
            cache_key=d.get("cache_key"),
            idempotency_key=d.get("idempotency_key"),
            authority_required=tuple(
                (AuthorityKind(a["kind"]), a["subject"])
                for a in d.get("authority_required", ())),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    """One typed artifact flow: ``src_node.src_port -> dst_node.dst_port``."""
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    def __post_init__(self) -> None:
        for name in ("src_node", "src_port", "dst_node", "dst_port"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Edge.{name} must be non-empty")
        if self.src_node == self.dst_node:
            raise ValueError(f"Edge is a self-loop on {self.src_node!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"src_node": self.src_node, "src_port": self.src_port,
                "dst_node": self.dst_node, "dst_port": self.dst_port}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Edge":
        return cls(src_node=d["src_node"], src_port=d["src_port"],
                   dst_node=d["dst_node"], dst_port=d["dst_port"])


class CycleError(ValueError):
    """``topological_order`` on a graph that is not a DAG. ``nodes`` names the
    members of the strongly-tangled remainder, sorted, so the message is stable
    across runs (the validator turns this into a ``CYCLE`` finding)."""

    def __init__(self, nodes: Iterable[str]) -> None:
        self.nodes: tuple[str, ...] = tuple(sorted(nodes))
        super().__init__("plan is not a DAG; cycle among nodes: "
                         + ", ".join(self.nodes))


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlanGraph:
    """A typed plan: nodes, typed edges, budgets, and which revision it is.

    ``goal_digest`` pins the plan to the ``GoalSpec`` it answers (see
    ``goal_digest()``). ``revision``/``parent_revision`` are the graph-revision
    history doc §5 requires: a bounded repair does not mutate a plan, it
    produces revision N+1 whose untouched nodes are the SAME nodes.

    ``revision_reason`` is an addition to the doc's field list: a revision with
    no recorded reason is an unexplained plan change, and ``revise()`` refuses
    to make one."""
    graph_id: str
    goal_digest: str
    revision: int = 0
    nodes: tuple[PlanNode, ...] = ()
    edges: tuple[Edge, ...] = ()
    budgets: BudgetHints = field(default_factory=BudgetHints)
    planner_mode: PlannerMode = PlannerMode.LOCAL_ONLY
    recipe: str | None = None
    parent_revision: int | None = None
    revision_reason: str = ""

    def __post_init__(self) -> None:
        if not self.graph_id.strip():
            raise ValueError("PlanGraph.graph_id must be non-empty")
        if not self.goal_digest.strip():
            raise ValueError("PlanGraph.goal_digest must be non-empty — a plan "
                             "that is not pinned to a goal is not a plan")
        if self.revision < 0:
            raise ValueError(f"revision must be >= 0, got {self.revision}")
        if self.parent_revision is not None:
            if self.parent_revision < 0:
                raise ValueError(
                    f"parent_revision must be >= 0, got {self.parent_revision}")
            if self.parent_revision >= self.revision:
                raise ValueError(
                    f"parent_revision {self.parent_revision} must precede "
                    f"revision {self.revision}")

    # -- lookups -----------------------------------------------------------

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.node_id for n in self.nodes)

    def node(self, node_id: str) -> PlanNode | None:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def segment_node_ids(self) -> tuple[str, ...]:
        """Nodes tagged ``params['segment'] = True``, in declaration order."""
        return tuple(n.node_id for n in self.nodes if n.is_segment)

    def _index(self) -> dict[str, int]:
        index: dict[str, int] = {}
        for i, n in enumerate(self.nodes):
            index.setdefault(n.node_id, i)
        return index

    def predecessors(self) -> dict[str, set[str]]:
        """node -> the nodes it waits on (edges + depends_on). References to
        unknown nodes are DROPPED here and reported by the validator as
        DANGLING_EDGE: traversal must not depend on the graph being valid."""
        known = set(self.node_ids)
        preds: dict[str, set[str]] = {nid: set() for nid in known}
        for n in self.nodes:
            for dep in n.depends_on:
                if dep in known and dep != n.node_id:
                    preds[n.node_id].add(dep)
        for e in self.edges:
            if e.src_node in known and e.dst_node in known:
                preds[e.dst_node].add(e.src_node)
        return preds

    def successors(self) -> dict[str, set[str]]:
        succs: dict[str, set[str]] = {nid: set() for nid in self.node_ids}
        for dst, srcs in self.predecessors().items():
            for src in srcs:
                succs[src].add(dst)
        return succs

    # -- traversal ---------------------------------------------------------

    def topological_order(self) -> tuple[str, ...]:
        """Deterministic Kahn: ties broken by DECLARATION order, so the same
        graph always yields the same order (a plan quote must not shuffle).
        Raises ``CycleError`` naming the tangle."""
        index = self._index()
        preds = {k: set(v) for k, v in self.predecessors().items()}
        remaining = dict(preds)
        out: list[str] = []
        while remaining:
            ready = sorted((nid for nid, ps in remaining.items() if not ps),
                           key=lambda nid: index[nid])
            if not ready:
                raise CycleError(remaining)
            for nid in ready:
                out.append(nid)
                del remaining[nid]
            done = set(out)
            for ps in remaining.values():
                ps -= done
        return tuple(out)

    def _reachable(self, start: str, graph: dict[str, set[str]]) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(graph.get(start, ()))
        while stack:                      # visited-set walk: safe on a cycle
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(graph.get(cur, ()))
        seen.discard(start)
        return frozenset(seen)

    def ancestors(self, node_id: str) -> frozenset[str]:
        """Everything ``node_id`` transitively waits on. Cycle-safe."""
        return self._reachable(node_id, self.predecessors())

    def descendants(self, node_id: str) -> frozenset[str]:
        """Everything that transitively waits on ``node_id``. Cycle-safe."""
        return self._reachable(node_id, self.successors())

    def roots(self) -> tuple[str, ...]:
        preds = self.predecessors()
        return tuple(nid for nid in self.node_ids if not preds.get(nid))

    def leaves(self) -> tuple[str, ...]:
        succs = self.successors()
        return tuple(nid for nid in self.node_ids if not succs.get(nid))

    def incoming(self, node_id: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.dst_node == node_id)

    def outgoing(self, node_id: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.src_node == node_id)

    # -- derivation --------------------------------------------------------

    def subgraph(self, node_ids: Iterable[str]) -> "PlanGraph":
        """The induced VIEW over ``node_ids``: the same node objects (identity
        preserved), only the edges internal to the set. It is a fragment for
        inspection and digesting — the repair controller's "smallest invalid
        subgraph" — not a standalone runnable plan: a kept node's
        ``depends_on`` may still name a node outside the fragment."""
        wanted = set(node_ids)
        unknown = sorted(wanted - set(self.node_ids))
        if unknown:
            raise ValueError(f"subgraph(): unknown node ids {unknown}")
        nodes = tuple(n for n in self.nodes if n.node_id in wanted)
        edges = tuple(e for e in self.edges
                      if e.src_node in wanted and e.dst_node in wanted)
        return PlanGraph(
            graph_id=self.graph_id, goal_digest=self.goal_digest,
            revision=self.revision, nodes=nodes, edges=edges,
            budgets=self.budgets, planner_mode=self.planner_mode,
            recipe=self.recipe, parent_revision=self.parent_revision,
            revision_reason=self.revision_reason)

    def revise(self, replacing: Iterable[str], new_nodes: Iterable[PlanNode],
               reason: str) -> "PlanGraph":
        """Bounded repair as a CONTROLLED GRAPH REVISION (doc §3.3/§5).

        Returns revision N+1 with ``parent_revision=N``. Every node NOT named in
        ``replacing`` is carried over unchanged — same id, same object — so a
        completed expensive node is still the same node afterwards and its cache
        key still hits. A replaced id that no new node re-uses is DROPPED along
        with its edges. A new node whose id collides with an untouched node is
        refused: that is an unannounced replacement."""
        replacing = frozenset(replacing)
        new_list = tuple(new_nodes)
        known = set(self.node_ids)
        unknown = sorted(replacing - known)
        if unknown:
            raise ValueError(f"revise(): unknown node ids {unknown}")
        if not reason.strip():
            raise ValueError("revise() needs a reason — an unexplained plan "
                             "change is exactly what graph revisions prevent")
        by_id: dict[str, PlanNode] = {}
        for n in new_list:
            if n.node_id in by_id:
                raise ValueError(f"revise(): duplicate new node {n.node_id!r}")
            by_id[n.node_id] = n

        out: list[PlanNode] = []
        used: set[str] = set()
        for n in self.nodes:
            if n.node_id in replacing:
                repl = by_id.get(n.node_id)
                if repl is not None:
                    out.append(repl)
                    used.add(n.node_id)
                continue
            if n.node_id in by_id:
                raise ValueError(
                    f"revise(): new node {n.node_id!r} collides with an "
                    f"untouched node — list it in `replacing` to replace it")
            out.append(n)
        for n in new_list:
            if n.node_id not in used:
                out.append(n)

        surviving = {n.node_id for n in out}
        edges = tuple(e for e in self.edges
                      if e.src_node in surviving and e.dst_node in surviving)
        return PlanGraph(
            graph_id=self.graph_id, goal_digest=self.goal_digest,
            revision=self.revision + 1, nodes=tuple(out), edges=edges,
            budgets=self.budgets, planner_mode=self.planner_mode,
            recipe=self.recipe, parent_revision=self.revision,
            revision_reason=reason)

    # -- wire shape + digest ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "goal_digest": self.goal_digest,
            "revision": self.revision,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "budgets": self.budgets.to_dict(),
            "planner_mode": self.planner_mode.value,
            "recipe": self.recipe,
            "parent_revision": self.parent_revision,
            "revision_reason": self.revision_reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PlanGraph":
        return cls(
            graph_id=d["graph_id"],
            goal_digest=d["goal_digest"],
            revision=int(d.get("revision", 0)),
            nodes=tuple(PlanNode.from_dict(n) for n in d.get("nodes", ())),
            edges=tuple(Edge.from_dict(e) for e in d.get("edges", ())),
            budgets=BudgetHints.from_dict(d.get("budgets") or {}),
            planner_mode=PlannerMode(d.get("planner_mode", "local_only")),
            recipe=d.get("recipe"),
            parent_revision=d.get("parent_revision"),
            revision_reason=d.get("revision_reason", ""),
        )

    def plan_digest(self) -> str:
        """Content hash of the WHOLE plan (revision included). Stable across
        dict ordering and round-trips; changes when anything about the plan
        changes, which is what makes it quotable and cacheable."""
        return content_digest(self.to_dict())

    def structure_digest(self) -> str:
        """Content hash of the SHAPE only — nodes and edges, without graph_id,
        revision or provenance. Two plans that would execute identically share
        it, which is what a plan cache keys on."""
        return content_digest({"nodes": [n.to_dict() for n in self.nodes],
                               "edges": [e.to_dict() for e in self.edges]})


# ---------------------------------------------------------------------------
# Planner-mode refusal — the typed REFUSED outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannerModeRefusal:
    """A plan the planner-mode gate stopped. ``nodes`` is every ``(node_id,
    capability)`` that would have handed work to the frontier under a mode
    that forbids it; ``reason`` is operator-readable. Classified
    ``FailureClass.REFUSED`` like every other gate (invariant 12: a refusal is
    an outcome with evidence, never silence)."""
    planner_mode: PlannerMode
    nodes: tuple[tuple[str, str], ...]
    reason: str
    graph_id: str = ""
    failure: FailureClass = FailureClass.REFUSED

    def __post_init__(self) -> None:
        if not self.nodes and not self.reason:
            raise ValueError("a PlannerModeRefusal must name >=1 node or a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": "refused",
            "failure": self.failure.value,
            "planner_mode": self.planner_mode.value,
            "graph_id": self.graph_id,
            "nodes": [{"node_id": n, "capability": c} for n, c in self.nodes],
            "reason": self.reason,
            "frontier_enabled": frontier_enabled(),
        }


def frontier_nodes(graph: PlanGraph) -> tuple[tuple[str, str], ...]:
    """``((node_id, capability), …)`` for every node that names frontier work,
    in declaration order."""
    return tuple((n.node_id, n.capability or "")
                 for n in graph.nodes
                 if is_frontier_capability(n.capability, n.params))


def check_planner_mode(graph: PlanGraph) -> PlannerModeRefusal | None:
    """The gate. ``None`` means the plan is consistent with its declared
    planner mode and the fleet; otherwise a typed refusal that names exactly
    why:

      * ``local_only`` + any frontier-bound node  -> refused (policy §3.2);
      * ``frontier`` on a frontier-disabled fleet -> refused (policy §3.1),
        whether or not the plan actually uses a frontier node — the label
        itself would be the lie.
    """
    bound = frontier_nodes(graph)
    if graph.planner_mode is PlannerMode.LOCAL_ONLY:
        if not bound:
            return None
        named = ", ".join(f"{n}({c})" for n, c in bound)
        return PlannerModeRefusal(
            planner_mode=graph.planner_mode, nodes=bound, graph_id=graph.graph_id,
            reason=(f"planner_mode=local_only admits zero frontier capability "
                    f"nodes; refused: {named}"))
    if not frontier_enabled():
        return PlannerModeRefusal(
            planner_mode=graph.planner_mode, nodes=bound, graph_id=graph.graph_id,
            reason=(f"planner_mode=frontier requested but {FRONTIER_ENABLED_ENV} "
                    f"is not set on this fleet — no Frontier Keeper A is wired "
                    f"in, so no plan may claim its participation"))
    return None


# ---------------------------------------------------------------------------
# Stage 14 — the sibling invariant (invariant 9)
# ---------------------------------------------------------------------------


def sibling_violations(graph: PlanGraph,
                       segment_node_ids: Iterable[str]
                       ) -> tuple[tuple[str, str], ...]:
    """``((segment, ancestor_segment), …)`` — every place one segment
    transitively depends on another. Empty means the required Stage 14 shape
    holds: ``locked artifacts -> {S1, S2, S3}``, never ``S1 -> S2 -> S3``.

    Sharing LOCKED UPSTREAM PARENTS is not a violation and never should be —
    that is the required shape, not the prohibited one. Only segment-to-segment
    ancestry counts. Order is declaration order, so findings are stable."""
    ids = tuple(dict.fromkeys(segment_node_ids))
    known = set(graph.node_ids)
    unknown = sorted(set(ids) - known)
    if unknown:
        raise ValueError(f"sibling_violations(): unknown node ids {unknown}")
    others = set(ids)
    out: list[tuple[str, str]] = []
    for nid in ids:
        anc = graph.ancestors(nid) & (others - {nid})
        for parent in sorted(anc, key=lambda p: ids.index(p)):
            out.append((nid, parent))
    return tuple(out)


def sibling_check(graph: PlanGraph, segment_node_ids: Iterable[str]) -> bool:
    """True iff no segment node depends (transitively) on another segment node.
    They may share locked upstream parents."""
    return not sibling_violations(graph, segment_node_ids)


__all__ = [
    "AcceptanceTest",
    "CAPABILITY_KINDS",
    "CycleError",
    "Edge",
    "FRONTIER_CAPABILITY_PREFIX",
    "FRONTIER_ENABLED_ENV",
    "FRONTIER_PARAM",
    "FrozenParams",
    "NodeKind",
    "PlanGraph",
    "PlanNode",
    "PlannerModeRefusal",
    "Port",
    "ResourceRequest",
    "RetryPolicy",
    "SEGMENT_PARAM",
    "STRUCTURAL_KINDS",
    "canonical_json",
    "check_planner_mode",
    "coerce_artifact_kind",
    "content_digest",
    "effective_planner_mode",
    "frontier_enabled",
    "frontier_nodes",
    "goal_digest",
    "is_frontier_capability",
    "kind_value",
    "sibling_check",
    "sibling_violations",
]
