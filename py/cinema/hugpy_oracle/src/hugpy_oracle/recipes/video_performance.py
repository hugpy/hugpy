"""``video.performance`` on the durable DAG (unit A of METHOD-vigilant-inference).

The proven audio-first stages (authority → snapshot → audio → lock → segments)
keep running through ``performance.run_performance(stop_after="segments")``:
they are cheap, deterministic over their seams, and already journaled. What
they produce — the locked ``SegmentSpec`` siblings and the ``AudioMaster`` —
is compiled here into a typed ``PlanGraph`` for the EXPENSIVE visual stages
and executed under :class:`DagRuntime`:

    production_lock (gate)
        ├─ kf:<seg>  (FANOUT image.generate, N candidates from the prompt compiler)
        │     └─ kfjudge:<seg>  (JUDGE image.understand → chosen keyframe)
        │           └─ clip:<seg>  (FANOUT video.generate.i2v)
        │                 └─ clipjudge:<seg>  (JUDGE → accepted shot)
        └─ … one chain per segment, no edge between chains (sibling invariant)
    assemble (ASSEMBLE video.assemble ← every clipjudge)

What that buys a real build:

* **resume after kill** — a restart re-runs zero succeeded keyframes/clips;
* **per-candidate model routing** — each FANOUT candidate is selected and
  ledgered individually (``DagRuntime(selector=…)``);
* **targeted repair** — a judge's repair code resets the producer chain of
  THAT segment only (``RepairController``), bounded by the run's budget;
* **receipts** — every candidate carries model, selection decision, latency,
  angle and seed.

The seams are the SAME ``PerformanceSeams`` the linear recipe uses, so the
existing fakes drive this path unchanged. Unbound seams surface as typed
capability gaps on the node, never as a crash.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from hugpy_oracle.contracts import GoalSpec, RepairCode, Scorecard
from hugpy_oracle.dag_runtime import DagRuntime, NodeState, RunJournal, RunRecord, RunState
from hugpy_oracle.plan import Edge, NodeKind, PlanGraph, PlanNode, Port, RetryPolicy
from hugpy_oracle.repair_controller import RepairController, RepairPlan
from hugpy_oracle.validator import ErrorCode, validate as validate_plan
from hugpy_oracle import performance as perf
from hugpy_oracle.segments import SegmentSpec, segment_seed
from hugpy_oracle import spatial as sp

__all__ = [
    "SeamExecutor",
    "GraphRefused",
    "VisualResult",
    "build_visual_graph",
    "validate_visual_graph",
    "resume_visual_stages",
    "run_performance_on_dag",
    "run_visual_stages",
]

LOCK_NODE = "production_lock"
SPATIAL_CAPABILITY = "spatial.normalize.scene"
ASSEMBLE_NODE = "assemble"
JUDGE_CAPABILITY = perf.KEYFRAME_JUDGE_CAPABILITY     # "image.understand"
CLIP_JUDGE_CAPABILITY = "video.understand"


def _spatial(sid: str) -> str:
    return f"spatial:{sid}"


def _kf(sid: str) -> str:
    return f"kf:{sid}"


def _kfj(sid: str) -> str:
    return f"kfjudge:{sid}"


def _clip(sid: str) -> str:
    return f"clip:{sid}"


def _clipj(sid: str) -> str:
    return f"clipjudge:{sid}"


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #


def _plan_for(spec: SegmentSpec, pgoal: perf.PerformanceGoal, floor: int, eligible_models: int):
    try:
        from hugpy_oracle.prompt_compiler import compile_context
        quality = getattr(pgoal.goal.quality, "value", "balanced")
        cap = max(int(floor), {"preview": 1, "balanced": 4, "best": 8}.get(quality, 3))
        return compile_context(perf.segment_context(spec, pgoal), goal=pgoal.goal,
                               max_candidates=cap, eligible_models=eligible_models)
    except Exception:  # noqa: BLE001
        return None


def build_visual_graph(specs: tuple[SegmentSpec, ...], *, pgoal: perf.PerformanceGoal,
                       seams: perf.PerformanceSeams, lock_digest: str,
                       eligible_models: int = 1, revision: int = 0,
                       approval_before_render: bool = False,
                       manifests: Mapping[str, "sp.SpatialSceneManifest"] | None = None,
                       graph_id: str = "video.performance.visual") -> PlanGraph:
    """Compile the locked siblings into the visual DAG. Candidate counts per
    segment come from the prompt compiler (floored by the seams). A segment
    with a ``SpatialSceneManifest`` gets a ``spatial:<seg>`` node first: the
    manifest is validated against the lock and the Fold 1 → Fold 2
    ``ConditioningRequest`` is emitted; its keyframes depend on it."""
    manifests = dict(manifests or {})
    nodes: list[PlanNode] = [PlanNode(node_id=LOCK_NODE, kind=NodeKind.GATE,
                                      approval_gate=approval_before_render,
                                      params={"lock_digest": lock_digest})]
    edges: list[Edge] = []
    for spec in specs:
        sid = spec.segment_id
        plan = _plan_for(spec, pgoal, seams.keyframe_candidates, eligible_models)
        n_kf = max(int(seams.keyframe_candidates), plan.candidates if plan else 1)
        n_clip = max(int(seams.clip_candidates), plan.candidates if plan else 1)
        angles = [v.angle for v in plan.variants] if plan else []
        common = {"segment": sid, "lock_digest": lock_digest, "spec_digest": spec.digest,
                  "difficulty": plan.difficulty if plan else None, "angles": angles}
        kf_deps: tuple[str, ...] = (LOCK_NODE,)
        kf_inputs: tuple[Port, ...] = ()
        if sid in manifests:
            nodes.append(PlanNode(
                node_id=_spatial(sid), kind=NodeKind.TASK, capability=SPATIAL_CAPABILITY,
                outputs=(Port("conditioning", "json"), Port("manifest_digest", "json")),
                depends_on=(LOCK_NODE,),
                params={**common, "operation": "spatial.normalize", "model_free": True,
                        "manifest_digest": manifests[sid].digest,
                        "tier_profile": manifests[sid].tier_profile.to_dict()},
            ))
            kf_deps = (_spatial(sid),)
            kf_inputs = (Port("conditioning", "json", required=False),)
            edges.append(Edge(_spatial(sid), "conditioning", _kf(sid), "conditioning"))
        nodes.append(PlanNode(
            node_id=_kf(sid), kind=NodeKind.FANOUT, capability=perf.KEYFRAME_CAPABILITY,
            inputs=kf_inputs,
            outputs=(Port("image", "image"), Port("seed", "json"), Port("angle", "json")),
            depends_on=kf_deps, candidates=n_kf, params={**common, "operation": "image.generate"},
            retry=RetryPolicy(max_attempts=1 + int(perf.PerformanceBudget().keyframe_repair_rounds)),
        ))
        nodes.append(PlanNode(
            node_id=_kfj(sid), kind=NodeKind.JUDGE, capability=JUDGE_CAPABILITY,
            inputs=(Port("image", "image"), Port("seed", "json")),
            outputs=(Port("keyframe", "image"), Port("scorecard", "json")),
            depends_on=(_kf(sid),), params={**common, "operation": "image.judge"},
        ))
        edges += [Edge(_kf(sid), "image", _kfj(sid), "image"), Edge(_kf(sid), "seed", _kfj(sid), "seed")]
        nodes.append(PlanNode(
            node_id=_clip(sid), kind=NodeKind.FANOUT, capability=perf.CLIP_CAPABILITY,
            inputs=(Port("keyframe", "image"),),
            outputs=(Port("clip", "video"),),   # duration rides ON the clip artifact (seconds is not a separate artifact)
            depends_on=(_kfj(sid),), candidates=n_clip, params={**common, "operation": "video.generate"},
            retry=RetryPolicy(max_attempts=1 + int(perf.PerformanceBudget().clip_repair_rounds)),
        ))
        edges.append(Edge(_kfj(sid), "keyframe", _clip(sid), "keyframe"))
        nodes.append(PlanNode(
            node_id=_clipj(sid), kind=NodeKind.JUDGE, capability=CLIP_JUDGE_CAPABILITY,
            inputs=(Port("clip", "video"), Port("keyframe", "image")),
            outputs=(Port("shot", "json"),),
            depends_on=(_clip(sid),), params={**common, "operation": "video.judge"},
        ))
        edges += [Edge(_clip(sid), "clip", _clipj(sid), "clip"),
                  Edge(_kfj(sid), "keyframe", _clipj(sid), "keyframe")]
    # concat is deterministic model-free work but it IS a capability call
    # (ffmpeg behind video.assemble), so it is a TASK; plan.py reserves the
    # structural ASSEMBLE kind for capability-less joins.
    nodes.append(PlanNode(
        node_id=ASSEMBLE_NODE, kind=NodeKind.TASK, capability=perf.ASSEMBLE_CAPABILITY,
        inputs=(Port("shots", "json", many=True),), outputs=(Port("video", "video"), Port("shots", "json")),
        depends_on=tuple(_clipj(s.segment_id) for s in specs),
        params={"lock_digest": lock_digest, "order": [s.segment_id for s in specs], "model_free": True},
    ))
    edges += [Edge(_clipj(s.segment_id), "shot", ASSEMBLE_NODE, "shots") for s in specs]
    from hugpy_oracle.plan import goal_digest
    return PlanGraph(graph_id=graph_id, goal_digest=goal_digest(pgoal.goal), revision=revision,
                     nodes=tuple(nodes), edges=tuple(edges), recipe="video.performance")


class GraphRefused(ValueError):
    """Static validation refused the visual DAG before any node ran."""

    def __init__(self, report: Any) -> None:
        self.report = report
        codes = [e.code.value for e in report.errors]
        super().__init__(f"visual graph refused by static validation: {codes}")


# capability-presence errors are advisory for the visual DAG: seams may serve a
# capability the catalog view does not list (fakes; central in-process), and a
# missing backend surfaces as a typed gap on the node at run time anyway.
_STRUCTURAL_ONLY_ERRORS = frozenset({ErrorCode.UNKNOWN_CAPABILITY, ErrorCode.CAPABILITY_GAP,
                                     ErrorCode.BUDGET_UNKNOWN})


def validate_visual_graph(graph: PlanGraph, *, catalog_view: Mapping[str, Any] | None,
                          goal: GoalSpec) -> dict[str, Any]:
    """TODO-11: run the static validator on the recipe's graph. Structural
    faults (cycles, dangling/mismatched ports, missing required inputs, a
    judge pinned to its generator, sibling violations, authority, budget
    exceeded) REFUSE the graph; capability-presence findings are recorded as
    warnings on the result."""
    report = validate_plan(graph, dict(catalog_view or {}), goal)
    hard = tuple(e for e in report.errors if e.code not in _STRUCTURAL_ONLY_ERRORS)
    soft = tuple(e for e in report.errors if e.code in _STRUCTURAL_ONLY_ERRORS) + tuple(report.warnings)
    out = {"ok": not hard,
           "errors": [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in hard],
           "warnings": [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in soft]}
    if hard:
        class _R:  # minimal report carrier for GraphRefused
            errors = hard
        raise GraphRefused(_R())
    return out


# --------------------------------------------------------------------------- #
# executor + evaluator over the seams
# --------------------------------------------------------------------------- #


class SeamExecutor:
    """Node runner over ``PerformanceSeams``. Keeps the last scorecard per
    judge node so the evaluator can hand it to the runtime."""

    def __init__(self, specs: tuple[SegmentSpec, ...], *, pgoal: perf.PerformanceGoal,
                 seams: perf.PerformanceSeams, master: Any,
                 manifests: Mapping[str, "sp.SpatialSceneManifest"] | None = None,
                 production_fps: float | None = None,
                 asset_exists: Any = None, checksum_of: Any = None) -> None:
        self.manifests = dict(manifests or {})
        self.production_fps = production_fps
        self.asset_exists = asset_exists
        self.checksum_of = checksum_of
        self.specs = {s.segment_id: s for s in specs}
        self.order = [s.segment_id for s in specs]
        self.pgoal = pgoal
        self.seams = seams
        self.master = master
        self.cards: dict[str, Scorecard] = {}
        self.threshold = float(perf.THRESHOLDS[pgoal.goal.quality])
        self.tolerance = float(pgoal.speech_policy.duration_tolerance)
        self.calls: dict[str, int] = {"gen_image": 0, "judge_image": 0, "gen_clip": 0,
                                      "judge_clip": 0, "concat": 0}
        self._plans: dict[str, Any] = {}

    # -- helpers -------------------------------------------------------------- #

    def _spec(self, node: PlanNode) -> SegmentSpec:
        return self.specs[str(node.params["segment"])]

    def _plan(self, spec: SegmentSpec, floor: int):
        if spec.segment_id not in self._plans:
            self._plans[spec.segment_id] = _plan_for(spec, self.pgoal, floor, 1)
        return self._plans[spec.segment_id]

    def _seam(self, name: str):
        fn = getattr(self.seams, name, None)
        if fn is None:
            gap = self.seams.gap_for(name)
            raise perf.SeamUnavailable(f"{gap.capability}: {gap.requirement}")
        self.calls[name] = self.calls.get(name, 0) + 1
        return fn

    # -- nodes ---------------------------------------------------------------- #

    def __call__(self, node: PlanNode, inputs: dict[str, Any], ctx: Any) -> Mapping[str, Any]:
        nid = node.node_id
        if nid == LOCK_NODE:
            return {}
        if nid.startswith("spatial:"):
            return self._spatial(node)
        if nid.startswith("kf:"):
            return self._keyframe(node, ctx)
        if nid.startswith("kfjudge:"):
            return self._judge_keyframes(node, inputs)
        if nid.startswith("clip:"):
            return self._clip(node, inputs, ctx)
        if nid.startswith("clipjudge:"):
            return self._judge_clips(node, inputs)
        if nid == ASSEMBLE_NODE:
            return self._assemble(node, inputs)
        raise ValueError(f"unknown node {nid}")

    def _spatial(self, node: PlanNode) -> Mapping[str, Any]:
        """Validate the segment's manifest against the lock; emit the Fold 1 →
        Fold 2 payload. Any fault is a typed failure with every code listed."""
        spec = self._spec(node)
        m = self.manifests[spec.segment_id]
        known = {e.entity_id for e in m.entities}  # continuity cast when available
        before = dict(spec.continuity.state_before) if spec.continuity is not None else {}
        cast = before.get("characters")
        if isinstance(cast, Mapping):
            known = set(cast.keys()) | {e.entity_id for e in m.entities if e.entity_type != "character"}
        elif isinstance(cast, (list, tuple)) and cast:
            known = set(cast) | {e.entity_id for e in m.entities if e.entity_type != "character"}
        report = sp.validate_manifest(
            m, expected_fps=self.production_fps, known_entity_ids=known,
            asset_exists=self.asset_exists, checksum_of=self.checksum_of)
        if not report.ok:
            codes = ", ".join(f"{f.code.value}@{f.where}" for f in report.faults)
            exc = perf.SeamUnavailable(f"{SPATIAL_CAPABILITY}: manifest rejected ({codes})")
            exc.oracle_gap = True  # terminal: no render on unconstrained geometry
            raise exc
        req = sp.ConditioningRequest.from_manifest(m)
        return {"conditioning": req.to_dict(), "manifest_digest": m.digest,
                "validation": report.to_dict()}

    def _keyframe(self, node: PlanNode, ctx: Any) -> Mapping[str, Any]:
        spec = self._spec(node)
        plan = self._plan(spec, self.seams.keyframe_candidates)
        salt = "" if ctx.attempt <= 1 else f":{perf.KEYFRAME_REPAIR_SALT}:{ctx.attempt}"
        base = segment_seed(spec.lock_digest, spec.segment_id + salt, 0)
        seed = (base + ctx.candidate) % (2 ** 32)
        prompt, angle = perf._variant_prompt(spec.prompt, plan, ctx.candidate)
        from hugpy_oracle import selection
        with selection.pinned(perf.KEYFRAME_CAPABILITY, getattr(ctx, "model_id", None), getattr(ctx, "selection", None)):
            produced = self._seam("gen_image")(prompt, spec.identity_refs, seed)
        ref = perf._ref_of(produced)
        # the live seam registers the model that ACTUALLY produced the ref (from
        # its receipt); only when no seam did (fakes) fall back to the pin
        if selection.producer_of(ref) is None:
            selection.remember_producer(ref, perf.KEYFRAME_CAPABILITY, getattr(ctx, "model_id", None))
        prod = selection.producer_of(ref)
        return {"image": ref, "seed": seed, "angle": angle, "model_id": prod[1] if prod else None}

    def _judge_keyframes(self, node: PlanNode, inputs: dict[str, Any]) -> Mapping[str, Any]:
        spec = self._spec(node)
        refs = inputs.get("image") or []
        seeds = inputs.get("seed") or []
        refs = list(refs) if isinstance(refs, (list, tuple)) else [refs]
        seeds = list(seeds) if isinstance(seeds, (list, tuple)) else [seeds]
        last: Scorecard | None = None
        considered = 0
        for i, ref in enumerate(refs):
            considered += 1
            raw = None
            try:
                raw = self._seam("judge_image")(ref, spec)
            except perf.SeamUnavailable:
                raw = None
            verdict = perf.coerce_verdict(raw, threshold=self.threshold,
                                          default_code=RepairCode.INTENT_MISMATCH, judge="judge_image")
            card = perf.keyframe_scorecard(spec, verdict, image_ref=ref, threshold=self.threshold)
            perf._note_verdict(ref, card)
            last = card
            if card.hard_pass:
                self.cards[node.node_id] = card
                return {"keyframe": ref, "seed": seeds[i] if i < len(seeds) else None,
                        "scorecard": card.to_dict(), "considered": considered, "index": i}
        self.cards[node.node_id] = last if last is not None else Scorecard(
            hard_pass=False, diagnosis="no keyframe candidates", repair_code=RepairCode.EMPTY_OUTPUT)
        return {"keyframe": None, "seed": None, "scorecard": self.cards[node.node_id].to_dict(),
                "considered": considered, "index": None}

    def _clip(self, node: PlanNode, inputs: dict[str, Any], ctx: Any) -> Mapping[str, Any]:
        spec = self._spec(node)
        keyframe_ref = inputs.get("keyframe")
        if not keyframe_ref:
            raise perf.SeamUnavailable("no accepted keyframe to animate")
        plan = self._plan(spec, self.seams.clip_candidates)
        salt = "" if ctx.attempt <= 1 else f":{perf.CLIP_REPAIR_SALT}:{ctx.attempt}"
        base = segment_seed(spec.lock_digest, spec.segment_id + salt, 0)
        prompt, _angle = perf._variant_prompt(spec.prompt, plan, ctx.candidate)
        take = replace(spec, seed_base=(base + ctx.candidate) % (2 ** 32), prompt=prompt)
        from hugpy_oracle import selection
        with selection.pinned(perf.CLIP_CAPABILITY, getattr(ctx, "model_id", None), getattr(ctx, "selection", None)):
            produced = self._seam("gen_clip")(str(keyframe_ref), take)
        ref, seconds = perf._clip_of(produced)
        if selection.producer_of(ref) is None:
            selection.remember_producer(ref, perf.CLIP_CAPABILITY, getattr(ctx, "model_id", None))
        prod = selection.producer_of(ref)
        return {"clip": {"uri": ref, "seconds": seconds, "model_id": prod[1] if prod else None}}

    def _judge_clips(self, node: PlanNode, inputs: dict[str, Any]) -> Mapping[str, Any]:
        spec = self._spec(node)
        clips = inputs.get("clip") or []
        clips = list(clips) if isinstance(clips, (list, tuple)) else [clips]
        last: Scorecard | None = None
        considered = 0
        for i, item in enumerate(clips):
            considered += 1
            if isinstance(item, Mapping):
                ref, seconds = item.get("uri"), item.get("seconds")
            else:
                ref, seconds = item, None
            try:
                raw = self._seam("judge_clip")(ref, spec)
            except perf.SeamUnavailable:
                raw = None
            verdict = perf.coerce_verdict(raw, threshold=self.threshold,
                                          default_code=RepairCode.INTENT_MISMATCH, judge="judge_clip")
            card = perf.clip_scorecard(spec, verdict, clip_ref=ref, clip_seconds=seconds,
                                       threshold=self.threshold, duration_tolerance=self.tolerance)
            perf._note_verdict(ref, card)
            last = card
            if card.hard_pass:
                self.cards[node.node_id] = card
                return {"shot": {"segment_id": spec.segment_id, "clip_ref": ref, "seconds": seconds,
                                 "keyframe_ref": inputs.get("keyframe"), "scorecard": card.to_dict(),
                                 "considered": considered}}
        self.cards[node.node_id] = last if last is not None else Scorecard(
            hard_pass=False, diagnosis="no clip candidates", repair_code=RepairCode.EMPTY_OUTPUT)
        return {"shot": {"segment_id": spec.segment_id, "clip_ref": None, "seconds": None,
                         "keyframe_ref": inputs.get("keyframe"),
                         "scorecard": self.cards[node.node_id].to_dict(), "considered": considered}}

    def _assemble(self, node: PlanNode, inputs: dict[str, Any]) -> Mapping[str, Any]:
        shots = [s for s in (inputs.get("shots") or []) if isinstance(s, Mapping)]
        by_sid = {s["segment_id"]: s for s in shots}
        ordered = [by_sid[sid] for sid in node.params.get("order", self.order) if sid in by_sid]
        refs = [s["clip_ref"] for s in ordered if s.get("clip_ref")]
        if len(refs) != len(self.order):
            raise perf.SeamUnavailable(f"assembly needs {len(self.order)} accepted shots, has {len(refs)}")
        produced = self._seam("concat")(refs, self.master)
        return {"video": perf._ref_of(produced), "shots": ordered}

    # -- evaluator ------------------------------------------------------------ #

    def evaluate(self, node: PlanNode, outputs: Mapping[str, Any], ctx: Any) -> Scorecard | None:
        if node.kind is NodeKind.JUDGE:
            return self.cards.get(node.node_id)
        return None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VisualResult:
    run: RunRecord
    video_ref: str | None
    shots: tuple[Mapping[str, Any], ...]
    repairs: tuple[RepairPlan, ...]
    failed_nodes: tuple[str, ...]
    nodes: Mapping[str, str]                 # node_id -> state
    limitations: tuple[str, ...] = ()
    steward: Mapping[str, Any] | None = None  # HealthReport.to_dict() when a ledger was in play
    validation: Mapping[str, Any] | None = None  # static validation of the graph (TODO-11)

    @property
    def ok(self) -> bool:
        return self.run.state is RunState.COMPLETED and self.video_ref is not None

    def to_dict(self) -> dict[str, Any]:
        return {"run": self.run.to_dict(), "ok": self.ok, "video_ref": self.video_ref,
                "shots": [dict(s) for s in self.shots], "repairs": [r.to_dict() for r in self.repairs],
                "failed_nodes": list(self.failed_nodes), "nodes": dict(self.nodes),
                "limitations": list(self.limitations),
                "steward": dict(self.steward) if self.steward is not None else None,
                "validation": dict(self.validation) if self.validation is not None else None}


def _drive(rt: DagRuntime, run_id: str, *, max_repairs: int) -> tuple[list[RepairPlan], list[str]]:
    """run → repair the failed nodes → run, until complete or nothing repairable."""
    ctl = RepairController(rt)
    repairs: list[RepairPlan] = []
    for _ in range(max_repairs + 1):
        rec = rt.run(run_id)
        if rec.state is not RunState.FAILED:
            break
        failed = [nid for nid, r in rt.journal.nodes(run_id).items() if r.state is NodeState.FAILED]
        applied = False
        for nid in failed:
            plan = ctl.diagnose(run_id, nid)
            repairs.append(plan)
            if plan.repairable:
                try:
                    ctl.apply(plan)
                    applied = True
                except Exception:  # noqa: BLE001 — budget exhausted etc. is reported via the plan list
                    pass
        if not applied:
            break
    final_failed = [nid for nid, r in rt.journal.nodes(run_id).items() if r.state is NodeState.FAILED]
    return repairs, final_failed


def _steward_report(rt: DagRuntime) -> Mapping[str, Any] | None:
    """The system checks itself at the end of every run: calibration,
    streaks, starvation, gap rate, matrix staleness — and rebalances within
    bounds. Library-only stewards are worthless; this is the scheduling."""
    sel = rt.selector
    ledger = getattr(sel, "ledger", None)
    if ledger is None:
        return None
    try:
        from hugpy_oracle.steward import Steward
        return Steward(ledger, selector=sel, journal=rt.journal, matrix=getattr(sel, "matrix", lambda: None)()
                       ).check().to_dict()
    except Exception as exc:  # noqa: BLE001 — the audit must not take the deliverable down
        return {"ok": False, "summary": f"steward failed: {type(exc).__name__}: {exc}", "findings": []}


def _collect(rt: DagRuntime, run_id: str, repairs: list[RepairPlan], failed: list[str],
             limitations: tuple[str, ...] = (), validation: Mapping[str, Any] | None = None) -> VisualResult:
    nodes = rt.journal.nodes(run_id)
    asm = nodes.get(ASSEMBLE_NODE)
    video = asm.outputs.get("video") if (asm and asm.outputs) else None
    shots = tuple(asm.outputs.get("shots") or ()) if (asm and asm.outputs) else ()
    return VisualResult(rt.journal.run(run_id), video, shots, tuple(repairs), tuple(failed),
                        {nid: r.state.value for nid, r in nodes.items()}, limitations,
                        steward=_steward_report(rt), validation=validation)


def run_visual_stages(prep: perf.PerformanceResult, pgoal: perf.PerformanceGoal,
                      seams: perf.PerformanceSeams, *, journal: RunJournal, run_id: str,
                      selector: Any = None, repair_budget: int = 2, max_repairs: int = 4,
                      approval_before_render: bool = False, eligible_models: int = 1,
                      manifests: Mapping[str, "sp.SpatialSceneManifest"] | None = None,
                      production_fps: float | None = None,
                      asset_exists: Any = None, checksum_of: Any = None
                      ) -> tuple[DagRuntime, VisualResult]:
    specs = tuple(prep.segments or ())
    lock_digest = getattr(prep.lock, "digest", None) or (specs[0].lock_digest if specs else "")
    graph = build_visual_graph(specs, pgoal=pgoal, seams=seams, lock_digest=lock_digest,
                               eligible_models=eligible_models,
                               approval_before_render=approval_before_render, manifests=manifests)
    try:
        view = seams.catalog_view() if seams.catalog_view else {}
    except Exception:  # noqa: BLE001
        view = {}
    validation = validate_visual_graph(graph, catalog_view=view, goal=pgoal.goal)  # raises GraphRefused
    ex = SeamExecutor(specs, pgoal=pgoal, seams=seams, master=prep.audio_master, manifests=manifests,
                      production_fps=production_fps, asset_exists=asset_exists, checksum_of=checksum_of)
    rt = DagRuntime(journal, ex, evaluator=ex.evaluate, selector=selector, owner=f"video.performance:{run_id}")
    rt.start(graph, run_id, repair_budget=repair_budget)
    repairs, failed = _drive(rt, run_id, max_repairs=max_repairs)
    return rt, _collect(rt, run_id, repairs, failed, tuple(prep.limitations or ()), validation)


def resume_visual_stages(prep: perf.PerformanceResult, pgoal: perf.PerformanceGoal,
                         seams: perf.PerformanceSeams, *, journal: RunJournal, run_id: str,
                         selector: Any = None, max_repairs: int = 4,
                         manifests: Mapping[str, "sp.SpatialSceneManifest"] | None = None
                         ) -> tuple[DagRuntime, VisualResult]:
    """After a crash: reconcile leases and continue. Succeeded nodes are not
    re-run; their outputs come from the journal."""
    specs = tuple(prep.segments or ())
    ex = SeamExecutor(specs, pgoal=pgoal, seams=seams, master=prep.audio_master, manifests=manifests)
    rt = DagRuntime(journal, ex, evaluator=ex.evaluate, selector=selector, owner=f"video.performance:{run_id}:resumed")
    rt.resume(run_id)
    repairs, failed = _drive(rt, run_id, max_repairs=max_repairs)
    return rt, _collect(rt, run_id, repairs, failed, tuple(prep.limitations or ()))


def run_performance_on_dag(pgoal: perf.PerformanceGoal, *, seams: perf.PerformanceSeams,
                           journal: RunJournal, run_id: str | None = None, selector: Any = None,
                           budget: perf.PerformanceBudget | None = None, repair_budget: int = 2,
                           max_repairs: int = 4) -> tuple[perf.PerformanceResult, VisualResult | None, DagRuntime | None]:
    """The whole recipe: linear stages to the lock, visual stages on the DAG.
    Returns (prep, visual, runtime); ``visual`` is None when prep stopped
    honestly before the lock (authority refused, audio gap, …)."""
    prep = perf.run_performance(pgoal, seams=seams, budget=budget, run_id=run_id, stop_after="segments")
    if prep.gap is not None or not prep.segments:
        return prep, None, None
    rid = f"{prep.run_id}:visual"
    rt, visual = run_visual_stages(prep, pgoal, seams, journal=journal, run_id=rid,
                                   selector=selector, repair_budget=repair_budget, max_repairs=max_repairs)
    return prep, visual, rt
