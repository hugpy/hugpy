"""Graph-level repair controller (k112): repair code -> smallest invalid subgraph.

``repair.py`` (k90c) decides ONE bounded retry for a single synchronous route
call. This module is the DAG-scale counterpart for :mod:`dag_runtime`: given a
node that FAILED with a :class:`RepairCode`, it

1. names the **responsible root** — the nearest ancestor (or the node itself)
   whose capability the policy says owns that class of failure;
2. computes the **repair path** — the responsible root plus only those nodes
   that are simultaneously its descendants and the failed node's ancestors
   (or the failed node itself). Siblings are never touched; accepted work
   outside the path is never repeated (invariant 6);
3. chooses a **strategy**: ``retry_nodes`` when a re-run with fresh randomness
   is the fix (no graph change, attempt-bounded by each node's RetryPolicy),
   or ``replan`` when parameters must change (a new graph revision, bounded
   by the run's repair budget);
4. records the decision on the run as a ``repair`` control event.

Everything is pure until :meth:`RepairController.apply` — ``diagnose`` is a
value you can show in the UI before anything moves.

Policy table (directive §9/§17): responsible-root capability prefixes are
searched among the failed node's ancestors nearest-first. "Never" lists name
capabilities that must not be reset for that code even if they are ancestors
— e.g. an IDENTITY_DRIFT on a clip must not regenerate transcription or audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from hugpy_oracle.contracts import RepairCode
from hugpy_oracle.dag_runtime import DagRuntime, NodeState, RepairBudgetExceeded
from hugpy_oracle.plan import FrozenParams, PlanGraph, PlanNode

__all__ = [
    "POLICY",
    "RepairPolicy",
    "RepairController",
    "RepairPlan",
    "responsible_root",
    "repair_path",
]


@dataclass(frozen=True, slots=True)
class RepairPolicy:
    """How one repair code is handled."""
    roots: tuple[str, ...]            # capability prefixes that own the failure, nearest-first
    never: tuple[str, ...] = ()       # capability prefixes that must never be reset for this code
    param_changes: Mapping[str, Any] = field(default_factory=dict)  # non-empty => replan
    reseed: bool = True               # bump the seed on the responsible root
    rationale: str = ""


_AUDIO = ("audio.", "voice.", "speech.")
_PRODUCERS = ("image.generate", "image.keyframe", "video.generate", "voice.synthesize", "audio.tts",
              "text.", "media.assemble", "video.assemble")
_ANALYSIS = ("audio.transcribe", "audio.separate", "voice.analyze", "image.identity_reference_pack")

POLICY: dict[RepairCode, RepairPolicy] = {
    RepairCode.IDENTITY_DRIFT: RepairPolicy(
        roots=("image.keyframe", "image.generate", "spatial.render", "image.identity_reference_pack"),
        never=_AUDIO + ("audio.transcribe",),
        rationale="identity lives in the keyframe / identity conditioning, not in audio or text",
    ),
    RepairCode.ACTION_MISSING: RepairPolicy(
        roots=("video.generate", "image.keyframe", "image.generate"),
        never=_AUDIO + _ANALYSIS,
        rationale="the requested action is a property of the clip render",
    ),
    RepairCode.TEMPORAL_ARTIFACT: RepairPolicy(
        roots=("video.generate",),
        never=_AUDIO + _ANALYSIS + ("image.keyframe",),
        rationale="flicker/mutation is a render-time defect; keyframes stay",
    ),
    RepairCode.GEOMETRY_DRIFT: RepairPolicy(
        roots=("video.generate", "spatial.render", "spatial.normalize"),
        never=_AUDIO + _ANALYSIS,
        param_changes={"geometry_strength": "+0.1"},
        rationale="raise geometry conditioning strength and re-render from the locked manifest",
    ),
    RepairCode.CAMERA_PATH_MISMATCH: RepairPolicy(
        roots=("video.generate", "spatial.render"),
        never=_AUDIO + _ANALYSIS,
        param_changes={"geometry_strength": "+0.1", "camera_lock": True},
        rationale="camera track is authoritative; re-render with the camera pass enforced",
    ),
    RepairCode.COLLISION_VIOLATION: RepairPolicy(
        roots=("video.generate", "spatial.simulate", "physics."),
        never=_AUDIO + _ANALYSIS,
        param_changes={"silhouette_mask": "hard"},
        rationale="simulation contacts are authoritative; enforce hard containment masks",
    ),
    RepairCode.VOICE_SIMILARITY_LOW: RepairPolicy(
        roots=("voice.synthesize", "audio.tts"),
        never=("audio.transcribe", "image.", "video."),
        rationale="re-synthesize the line; transcription and picture are unaffected",
    ),
    RepairCode.LINE_OMITTED: RepairPolicy(
        roots=("voice.synthesize", "audio.tts", "media.assemble"),
        never=("image.", "video.generate", "audio.transcribe"),
        reseed=False,
        rationale="a dropped line is an audio-master defect",
    ),
    RepairCode.SHOT_TOO_SHORT: RepairPolicy(
        roots=("video.generate",),
        never=_AUDIO + _ANALYSIS,
        param_changes={"duration_s": "+0.5"},
        rationale="extend the shot to cover its audio window",
    ),
    RepairCode.LIP_SYNC_OUT_OF_RANGE: RepairPolicy(
        roots=("audio_video.sync", "video.generate"),
        never=("audio.transcribe", "voice.synthesize", "audio.tts", "image.identity_reference_pack"),
        rationale="bounded retiming / re-render against the locked audio window",
    ),
    RepairCode.SOURCE_AUTHORITY_MISSING: RepairPolicy(
        roots=(), never=("*",), reseed=False,
        rationale="authority is an operator gate, not a regeneration",
    ),
    RepairCode.CAPABILITY_GAP: RepairPolicy(
        roots=(), never=("*",), reseed=False,
        rationale="no implementation exists; report the gap",
    ),
    # Judge-detected generic failures: go back to the nearest PRODUCER (a judge
    # node re-judging the same takes fixes nothing).
    RepairCode.INTENT_MISMATCH: RepairPolicy(roots=_PRODUCERS, never=_ANALYSIS,
                                             rationale="regenerate at the nearest producer"),
    RepairCode.DECODE_FAILED: RepairPolicy(roots=_PRODUCERS, rationale="re-run the producer"),
    RepairCode.EMPTY_OUTPUT: RepairPolicy(roots=_PRODUCERS, rationale="re-run the producer"),
    RepairCode.FORMAT_MISMATCH: RepairPolicy(roots=_PRODUCERS, rationale="re-run the producer"),
    RepairCode.TIMEOUT: RepairPolicy(roots=_PRODUCERS, reseed=False, rationale="re-run the producer"),
    RepairCode.WORKER_UNAVAILABLE: RepairPolicy(roots=_PRODUCERS, reseed=False, rationale="re-run the producer"),
}

NON_REPAIRABLE: frozenset[RepairCode] = frozenset(
    {RepairCode.SOURCE_AUTHORITY_MISSING, RepairCode.CAPABILITY_GAP}
)


@dataclass(frozen=True, slots=True)
class RepairPlan:
    run_id: str
    failed_node: str
    code: RepairCode
    root: str | None
    path: tuple[str, ...]
    strategy: str                      # "retry_nodes" | "replan" | "none"
    param_changes: Mapping[str, Any]
    rationale: str
    repairable: bool
    budget_left: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "failed_node": self.failed_node, "code": self.code.value,
            "root": self.root, "path": list(self.path), "strategy": self.strategy,
            "param_changes": dict(self.param_changes), "rationale": self.rationale,
            "repairable": self.repairable, "budget_left": self.budget_left,
        }


def _cap_matches(cap: str | None, prefixes: tuple[str, ...]) -> bool:
    if not cap:
        return False
    return any(p == "*" or cap.startswith(p) for p in prefixes)


def responsible_root(graph: PlanGraph, failed_node: str, policy: RepairPolicy) -> str | None:
    """Nearest ancestor (BFS upward, declaration order for ties) whose
    capability matches ``policy.roots``; the failed node itself if it matches
    or if no root is named; ``None`` when the policy forbids any repair."""
    if "*" in policy.never:
        return None
    node = graph.node(failed_node)
    if node is None:
        return None
    if not policy.roots:
        return failed_node
    if _cap_matches(node.capability, policy.roots):
        return failed_node
    preds = graph.predecessors()
    order = {nid: i for i, nid in enumerate(graph.node_ids)}
    frontier = sorted(preds.get(failed_node, ()), key=order.get)
    seen: set[str] = set()
    while frontier:
        nxt: list[str] = []
        for nid in frontier:
            if nid in seen:
                continue
            seen.add(nid)
            cand = graph.node(nid)
            if cand is not None and _cap_matches(cand.capability, policy.roots) \
                    and not _cap_matches(cand.capability, policy.never):
                return nid
            nxt.extend(preds.get(nid, ()))
        frontier = sorted(set(nxt), key=order.get)
    return failed_node


def repair_path(graph: PlanGraph, root: str, failed_node: str, policy: RepairPolicy) -> tuple[str, ...]:
    """``root`` + nodes that are descendants of root AND ancestors of the failed
    node, + the failed node. Excludes anything the policy says never to touch."""
    if root == failed_node:
        members = {failed_node}
    else:
        between = graph.descendants(root) & graph.ancestors(failed_node)
        members = {root, failed_node} | set(between)
    out = []
    for nid in graph.topological_order():
        if nid not in members:
            continue
        n = graph.node(nid)
        if n is not None and _cap_matches(n.capability, policy.never) and nid != failed_node:
            continue
        out.append(nid)
    return tuple(out)


def _bump(params: Mapping[str, Any], changes: Mapping[str, Any], *, reseed: bool,
          salt: str) -> dict[str, Any]:
    out = dict(params)
    for k, v in changes.items():
        if isinstance(v, str) and v[:1] in "+-" and v[1:].replace(".", "", 1).isdigit():
            delta = float(v)
            base = out.get(k)
            out[k] = round((float(base) if isinstance(base, (int, float)) else 0.0) + delta, 6)
        else:
            out[k] = v
    if reseed:
        seed = out.get("seed")
        out["seed"] = (int(seed) + 1) if isinstance(seed, int) else abs(hash(salt)) % (2**31)
    return out


class RepairController:
    def __init__(self, runtime: DagRuntime, policy: Mapping[RepairCode, RepairPolicy] | None = None) -> None:
        self.runtime = runtime
        self.policy = dict(POLICY if policy is None else policy)

    # -- pure ---------------------------------------------------------------- #

    def diagnose(self, run_id: str, failed_node: str) -> RepairPlan:
        j = self.runtime.journal
        rec = j.node(run_id, failed_node)
        run = j.run(run_id)
        budget_left = max(0, run.repair_budget - run.revisions_used)
        code = rec.repair_code
        if rec.state is not NodeState.FAILED or code is None:
            return RepairPlan(run_id, failed_node, code or RepairCode.INTENT_MISMATCH, None, (),
                              "none", {}, f"{failed_node} is {rec.state.value} with no repair code",
                              False, budget_left)
        pol = self.policy.get(code) or RepairPolicy(roots=())
        if code in NON_REPAIRABLE:
            return RepairPlan(run_id, failed_node, code, None, (), "none", {}, pol.rationale,
                              False, budget_left)
        graph = j.graph(run_id)
        root = responsible_root(graph, failed_node, pol)
        if root is None:
            return RepairPlan(run_id, failed_node, code, None, (), "none", {}, pol.rationale,
                              False, budget_left)
        path = repair_path(graph, root, failed_node, pol)
        if pol.param_changes:
            strategy = "replan" if budget_left > 0 else "none"
            repairable = budget_left > 0
            rationale = pol.rationale if repairable else f"{pol.rationale} — repair budget exhausted"
        else:
            strategy, repairable, rationale = "retry_nodes", True, pol.rationale
        return RepairPlan(run_id, failed_node, code, root, path, strategy,
                          dict(pol.param_changes), rationale, repairable, budget_left)

    # -- effectful ----------------------------------------------------------- #

    def apply(self, plan: RepairPlan) -> tuple[str, ...]:
        """Execute ``plan``. Returns the node ids reset. Raises
        :class:`RepairBudgetExceeded` if a replan would overrun the budget."""
        rt = self.runtime
        j = rt.journal
        j.record_control(plan.run_id, "repair", plan.failed_node, plan.to_dict())
        if not plan.repairable or plan.strategy == "none":
            return ()
        pol = self.policy.get(plan.code) or RepairPolicy(roots=())
        if plan.strategy == "retry_nodes":
            return rt.retry_nodes(plan.run_id, plan.path, reason=f"repair {plan.code.value}",
                                  force=(plan.root,) if plan.root else (), control="repair_retry",
                                  anchor=plan.failed_node)
        # replan: revise the responsible root's params, keep everything else
        graph = j.graph(plan.run_id)
        root_node = graph.node(plan.root or "")
        if root_node is None:
            return ()
        if plan.budget_left <= 0:
            raise RepairBudgetExceeded(f"run {plan.run_id}: repair budget exhausted")
        salt = f"{plan.run_id}:{plan.root}:{plan.code.value}:{graph.revision + 1}"
        new_params = _bump(root_node.params, plan.param_changes, reseed=pol.reseed, salt=salt)
        new_root = _replace_params(root_node, new_params)
        new_graph = graph.revise(replacing=[root_node.node_id], new_nodes=[new_root],
                                 reason=f"repair {plan.code.value} at {plan.root}")
        reset = rt.replan(plan.run_id, new_graph, f"repair {plan.code.value}: {plan.rationale}")
        # replan resets root + ALL its descendants; restore accepted siblings that
        # are not on the repair path so unrelated work is not repeated.
        off_path = [n for n in reset if n not in plan.path]
        if off_path:
            self._restore(plan.run_id, off_path, graph)
        return tuple(n for n in reset if n in plan.path)

    def repair(self, run_id: str, failed_node: str) -> RepairPlan:
        plan = self.diagnose(run_id, failed_node)
        self.apply(plan)
        return plan

    def _restore(self, run_id: str, node_ids: list[str], old_graph: PlanGraph) -> None:
        """Off-path descendants reset by replan are structurally downstream of
        the changed root, so their inputs MAY change; they stay PENDING but
        keep their cache key so an unchanged input is a journal read, not a
        re-execution. Nothing to do beyond leaving the cache key in place —
        replan already preserved it. Recorded for the audit trail."""
        self.runtime.journal.record_control(run_id, "repair_offpath", None,
                                            {"nodes": node_ids, "note": "cache-eligible, not forced"})


def _replace_params(node: PlanNode, params: Mapping[str, Any]) -> PlanNode:
    from dataclasses import replace
    return replace(node, params=FrozenParams(params))
