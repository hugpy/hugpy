"""Oracle evaluator kernel (k90c): capability-aware judge rubrics on the Scorecard.

Generalizes the movie keyframe judge (``video_intel/runners/movie.py``:
``_score_keyframe`` / ``parse_vision_verdict``) into the oracle: the SAME
``VERDICT=YES|NO; SCORE=0-100; WHY=<one sentence>`` discipline, the SAME
tolerant parse, and the SAME degradation — a judge that raises, times out or
returns garbage yields an UNSCORED/UNAVAILABLE ``JudgeResult`` and NEVER flips
``hard_pass`` (movie: "unscored, keep"). Both callers import the same pure
parser from ``hugpy_platform.verdict``, so oracle has no video runtime import.

Rubrics are per-capability defaults (``RUBRICS``):

  image.generate / image.transform -> VLM "does this image satisfy the prompt"
                                      (kind=intent, judge via image.understand)
  text.summarize                   -> LLM faithfulness quick-check
                                      (kind=semantic, judge via text.chat)
  everything analytic              -> NO judge (embed/similarity/detect/
                                      classify/depth/segment/transcribe/
                                      understand/chat/keywords/extract/fetch)

The judge MODEL is resolved through the oracle's own router over the k90a
catalog (``resolve_route`` on the judge capability) — never hardcoded. The
pass bar is per-quality (``THRESHOLDS``: preview 40 / balanced 60 / best 75);
a judged score under it fails the card with ``RepairCode.INTENT_MISMATCH``
(the repair controller in repair.py maps that to one bounded retry).

Provider seams (``_resolve_judge_route`` / ``_resolve_judge_routes`` /
``_judge_dispatch``) are module-level and lazy so tests monkeypatch them and
no worker/GPU is touched.

k115 (or-k6): identity/semantic rubric classes are judged by a PANEL of
``JUDGE_PANEL_SIZE`` independent judges (``run_judges``). Independent means:
not the generator, not a prior pick, and not the generator's or a prior
pick's MODEL FAMILY (``model_family``: two quantizations / sizes / chat-vs-VL
variants of one lineage share training data and blind spots, so their
agreement is not evidence — operator pitfall). The panel's agreement rate
becomes ``Scorecard.confidence``; every dissenting ``judge: verdict`` lands in
``Scorecard.disagreements``. When only ONE independent judge exists the card
says so (``limitation:single_judge``, confidence ``SINGLE_JUDGE_CONFIDENCE``)
instead of faking a second opinion. ``run_judge`` is unchanged for callers.
"""

from __future__ import annotations

import logging
import re
import inspect
from dataclasses import dataclass, field, replace
from typing import Any

from hugpy_oracle.contracts import (
    CheckKind,
    ExecutionReceipt,
    GoalSpec,
    JudgeResult,
    QualityProfile,
    RepairCode,
    Scorecard,
)
from hugpy_oracle.router import RouteDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rubrics — which capabilities get judged, and how.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rubric:
    """One default judge rubric: what evidence axis it produces (``kind``),
    which catalog capability serves the judge, and whether the judged evidence
    is a produced image file or produced text."""
    name: str
    kind: CheckKind
    judge_capability: str
    judged_artifact: str    # "image" | "text"


RUBRICS: dict[str, Rubric] = {
    "image.generate":  Rubric("image_intent", CheckKind.INTENT,
                              "image.understand", "image"),
    "image.transform": Rubric("image_intent", CheckKind.INTENT,
                              "image.understand", "image"),
    "text.summarize":  Rubric("summary_faithfulness", CheckKind.SEMANTIC,
                              "text.chat", "text"),
}

# The capabilities POST /oracle/route evaluates by default (deliverable 3).
DEFAULT_EVALUATED: frozenset[str] = frozenset(RUBRICS)

# Judged-score pass bar per quality profile.
THRESHOLDS: dict[QualityProfile, int] = {
    QualityProfile.PREVIEW:  40,
    QualityProfile.BALANCED: 60,
    QualityProfile.BEST:     75,
}

#: Independent judges asked for per identity/semantic rubric (k115).
JUDGE_PANEL_SIZE = 2
#: Confidence recorded when only ONE independent judge could be seated: a lone
#: verdict has no agreement rate, so the card says "unconfirmed", not "unanimous".
SINGLE_JUDGE_CONFIDENCE = 0.5
#: Extra route resolutions allowed while hunting for a family-distinct judge.
_MAX_PANEL_ATTEMPTS = 8

_JUDGE_REPLY_FORMAT = ("Reply exactly: VERDICT=YES|NO; SCORE=0-100; "
                       "WHY=<one sentence>.")
_MAX_SOURCE_CHARS = 4000
_MAX_SUMMARY_CHARS = 2000
_ERROR_TAIL = 300


# ---------------------------------------------------------------------------
# Verdict parsing — lifted from movie.parse_vision_verdict (same tolerance).
# ---------------------------------------------------------------------------


from hugpy_platform.verdict import parse_verdict as parse_judge_verdict


def _reply_text(res: Any) -> str:
    """Best-effort reply text from a dispatch result (movie._vision_text)."""
    txt = getattr(res, "text", None)
    if txt:
        return txt
    if isinstance(res, dict):
        return str(res.get("text") or res.get("content") or res)
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(res, attr, None)
        if callable(fn):
            try:
                d = fn()
            except TypeError:
                continue
            if isinstance(d, dict) and d.get("text"):
                return d["text"]
    return str(res)


# ---------------------------------------------------------------------------
# Provider seams — lazy, monkeypatchable.
# ---------------------------------------------------------------------------


def _resolve_judge_route_excluding(judge_capability: str, exclude_model: str | None
                                   ) -> RouteDecision | None:
    """Judge selection with the generator excluded; falls back to the plain
    seam (which tests monkeypatch) when selection has no opinion."""
    try:
        from hugpy_oracle import selection as _selection, router
        goal = GoalSpec(objective=f"judge via {judge_capability}",
                        raw_prompt=f"judge via {judge_capability}", capability=judge_capability)
        requested, _d = _selection.requested_model_for(
            goal, judge_capability, exclude=(exclude_model,) if exclude_model else ())
        if requested:
            try:
                return router.resolve_route(goal, requested)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return _resolve_judge_route(judge_capability)


def _resolve_judge_route(judge_capability: str,
                         exclude_model: str | None = None) -> RouteDecision | None:
    """The judge's own route through the k90a catalog — the SAME resolution the
    judged execution used (eligibility gates, default-model policy), so the
    judge model is chosen, never hardcoded. None when resolution itself dies.
    ``exclude_model``: the generator — the selector is asked for a judge that
    is not it; the caller still verifies the outcome."""
    from hugpy_oracle import router
    try:
        goal = GoalSpec(objective=f"judge via {judge_capability}",
                        raw_prompt=f"judge via {judge_capability}",
                        capability=judge_capability)
        requested = None
        try:
            from hugpy_oracle import selection as _selection
            requested, _decision = _selection.requested_model_for(
                goal, judge_capability, exclude=(exclude_model,) if exclude_model else ())
        except Exception:  # noqa: BLE001
            requested = None
        try:
            return router.resolve_route(goal, requested)
        except Exception:  # noqa: BLE001 — selector/catalog disagreement: catalog wins
            return router.resolve_route(goal)
    except Exception as exc:  # noqa: BLE001 — an unresolvable judge degrades
        logger.info("oracle judge: route resolution failed (%s: %s)",
                    type(exc).__name__, exc)
        return None


_FAMILY_NOISE = re.compile(
    r"^(v?\d+(\.\d+)*[a-z]?|\d+[bm]|q\d[_a-z0-9]*|gguf|awq|gptq|fp\d+|bf\d+|int\d+|"
    r"instruct|chat|it|base|vl|vision|mini|small|medium|large|xl|xxl|turbo|"
    r"lightning|preview|latest|hf|ggml)$", re.I)


def model_family(model_id: str | None) -> str:
    """Family key of a model id: the lineage name with vendor path, size,
    quantization, version and variant suffixes stripped, lowercased.
    ``Qwen/Qwen2.5-7B-Instruct-Q4_K_M`` / ``qwen-vl`` / ``qwen2-chat`` ->
    ``qwen``; ``meta-llama/Llama-3-8b`` -> ``llama``; ``sdxl`` -> ``sdxl``.
    Two judges with the same key are NOT independent. Empty for None."""
    if not model_id:
        return ""
    base = str(model_id).strip().split("/")[-1].split(":")[0]
    tokens = [t for t in re.split(r"[-_.\s+]+", base) if t]
    for tok in tokens:
        if _FAMILY_NOISE.match(tok):
            continue
        stem = re.sub(r"\d+(\.\d+)*$", "", tok).lower()
        if stem:
            return stem
    return base.lower()


def _resolve_judge_routes(judge_capability: str, exclude_models: tuple[str, ...],
                          n: int) -> list[RouteDecision]:
    """Up to ``n`` judge routes, each a DIFFERENT model from a DIFFERENT family
    than every model in ``exclude_models`` (the generator) and every prior
    pick. First pick goes through the existing single-judge seams (so the
    monkeypatched tests keep working); further picks ask the selector with the
    growing exclusion set and fall back to the route's own eligible
    ``model_ids``. Bounded; never raises; may return fewer than ``n``."""
    generator = exclude_models[0] if exclude_models else None
    first = (_resolve_judge_route_excluding(judge_capability, generator)
             if generator else _resolve_judge_route(judge_capability))
    if first is None or first.execution != "execute":
        return [first] if first is not None else []

    excluded = {m for m in exclude_models if m}
    families = {model_family(m) for m in excluded}
    picks: list[RouteDecision] = []

    def _take(route: RouteDecision | None) -> bool:
        if route is None or route.execution != "execute" or not route.model_id:
            return False
        if route.model_id in excluded or model_family(route.model_id) in families:
            excluded.add(route.model_id)
            return False
        picks.append(route)
        excluded.add(route.model_id)
        families.add(model_family(route.model_id))
        return True

    if not _take(first):
        return [first]            # let run_judge name the self-judgment refusal

    attempts = 0
    while len(picks) < n and attempts < _MAX_PANEL_ATTEMPTS:
        attempts += 1
        candidate: RouteDecision | None = None
        try:
            from hugpy_oracle import router, selection as _selection
            goal = GoalSpec(objective=f"judge via {judge_capability}",
                            raw_prompt=f"judge via {judge_capability}",
                            capability=judge_capability)
            requested = None
            try:
                requested, _d = _selection.requested_model_for(
                    goal, judge_capability, exclude=tuple(excluded))
            except Exception:  # noqa: BLE001
                requested = None
            if requested and requested not in excluded:
                candidate = router.resolve_route(goal, requested)
            else:
                pool = [m for m in first.model_ids
                        if m not in excluded and model_family(m) not in families]
                if not pool:
                    break
                candidate = router.resolve_route(goal, pool[0])
        except Exception as exc:  # noqa: BLE001 — a second opinion is optional
            logger.info("oracle judge panel: extra route resolution failed (%s: %s)",
                        type(exc).__name__, exc)
            break
        _take(candidate)
    return picks


def _judge_dispatch(task: str, body: dict[str, Any]) -> Any:
    """The judge call through the same inference front door the runtime uses
    (normalize + execute_prompt). Kept as a SEPARATE seam from the runtime's
    so tests drive the judged execution and the judge independently."""
    from hugpy_oracle import runtime
    return runtime._dispatch(runtime._normalized_kwargs(task, body))


# ---------------------------------------------------------------------------
# Rubric prompt + evidence selection.
# ---------------------------------------------------------------------------


def _judged_image(artifacts: list[dict[str, Any]]) -> str | None:
    import os
    for art in artifacts:
        if art.get("kind") == "image" and os.path.isfile(art.get("uri", "")):
            return art["uri"]
    return None


def _judged_text(artifacts: list[dict[str, Any]]) -> str | None:
    for art in artifacts:
        text = str(art.get("text") or "").strip()
        if text:
            return text
    return None


def _rubric_body(rubric: Rubric, goal: GoalSpec,
                 artifacts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The pre-normalization judge request, or None when the artifacts carry
    nothing this rubric can judge (nothing to judge is not a judge failure —
    the technical checks already own that diagnosis)."""
    from hugpy_engine.utils.no_think import with_no_think

    if rubric.judged_artifact == "image":
        uri = _judged_image(artifacts)
        if uri is None:
            return None
        prompt = (f"GOAL: {goal.objective}.\n"
                  f"Does the image achieve this goal? {_JUDGE_REPLY_FORMAT}")
        return {"file": uri, "prompt": with_no_think(prompt),
                "max_new_tokens": 80}

    summary = _judged_text(artifacts)
    if summary is None:
        return None
    texts = [i.ref for i in goal.inputs if i.kind.value == "text"]
    source = "\n\n".join(texts) if texts else goal.raw_prompt
    prompt = (f"SOURCE:\n{source[:_MAX_SOURCE_CHARS]}\n\n"
              f"SUMMARY:\n{summary[:_MAX_SUMMARY_CHARS]}\n\n"
              "Is the summary faithful to the source (no invented claims) and "
              f"does it cover the main points? {_JUDGE_REPLY_FORMAT}")
    return {"prompt": with_no_think(prompt), "max_new_tokens": 80}


# ---------------------------------------------------------------------------
# The kernel.
# ---------------------------------------------------------------------------


def _unavailable(rubric: Rubric, model: str | None, detail: str) -> JudgeResult:
    return JudgeResult(
        judge=f"{rubric.name}:{model or rubric.judge_capability}",
        verdict="unavailable", score=None, rationale=detail[-_ERROR_TAIL:])


def _judge_once(rubric: Rubric, body: dict[str, Any],
                judge_route: RouteDecision | None,
                generator_model: str | None) -> JudgeResult:
    """One judge call on an already-resolved route: the self-judgment refusal,
    dispatch, degradation and parse — shared by run_judge and run_judges."""
    if judge_route is None or judge_route.execution != "execute":
        reasons = "; ".join(judge_route.reasons) if judge_route else \
            "judge route resolution raised"
        return _unavailable(
            rubric, None,
            f"no eligible judge for {rubric.judge_capability}: {reasons}")
    if generator_model and judge_route.model_id == generator_model:
        return _unavailable(
            rubric, judge_route.model_id,
            f"judge {generator_model!r} is the generator of the artifact; refusing self-judgment")

    model_id = judge_route.model_id
    body = dict(body)
    if model_id:
        body["model_key"] = model_id
    judge_name = f"{rubric.name}:{model_id or rubric.judge_capability}"

    try:
        res = _judge_dispatch(judge_route.task or "", body)
    except Exception as exc:  # noqa: BLE001 — judge fault degrades, never fails the route
        logger.info("oracle judge %s raised (%s: %s); degrading to unavailable",
                    judge_name, type(exc).__name__, exc)
        return _unavailable(rubric, model_id, f"{type(exc).__name__}: {exc}")
    if not getattr(res, "ok", True) or (isinstance(res, dict)
                                        and res.get("ok") is False):
        err = getattr(res, "error", None) or \
            (res.get("error") if isinstance(res, dict) else None)
        return _unavailable(rubric, model_id, f"judge not-ok: {err}")

    from hugpy_engine.utils.no_think import strip_think
    raw = _reply_text(res)
    scored, _reasoning = strip_think(raw)   # parse the prose, never the monologue
    parsed = parse_judge_verdict(scored)
    return JudgeResult(
        judge=judge_name,
        verdict=parsed["verdict"] or "unscored",
        score=float(parsed["score"]) if parsed["score"] is not None else None,
        rationale=parsed["why"] or raw.strip()[:_ERROR_TAIL])


def run_judge(rubric: Rubric, goal: GoalSpec,
              artifacts: list[dict[str, Any]],
              generator_model: str | None = None) -> JudgeResult | None:
    """One judge pass under ``rubric``. None when nothing is judgeable;
    otherwise ALWAYS a JudgeResult — judge faults become the honest
    ``verdict="unavailable"`` entry (movie: plane raise -> unscored, keep).

    ``generator_model`` (invariant 11): the model that produced the artifact
    may not judge it. The judge is selected with that model excluded; if the
    only eligible judge IS the generator the verdict is ``unavailable`` with
    that reason, never a self-approval."""
    body = _rubric_body(rubric, goal, artifacts)
    if body is None:
        return None
    judge_route = (_resolve_judge_route_excluding(rubric.judge_capability, generator_model)
                   if generator_model else _resolve_judge_route(rubric.judge_capability))
    return _judge_once(rubric, body, judge_route, generator_model)


_USABLE = ("YES", "NO")


@dataclass(frozen=True, slots=True)
class JudgePanel:
    """The folded opinion of N independent judges (k115).

    ``confidence`` is the agreement rate among judges that returned a usable
    YES/NO (unanimous 2/2 -> 1.0; split -> 0.5 for two, 0.67 for 2-of-3...);
    a lone usable judge gets ``SINGLE_JUDGE_CONFIDENCE``; no usable judge 0.0.
    ``disagreements`` are ``"<judge>: <verdict>"`` for every judge off the
    majority (both sides when tied). ``limitations`` names what the panel
    could NOT provide: ``single_judge`` (only one independent judge seated or
    answered), ``no_judge`` (none). ``verdict``/``score`` are the panel's
    majority verdict and mean score — what ``evaluate`` gates on."""
    results: tuple[JudgeResult, ...]
    confidence: float
    disagreements: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    verdict: str = "unavailable"
    score: float | None = None
    models: tuple[str, ...] = field(default=())

    @property
    def usable(self) -> tuple[JudgeResult, ...]:
        return tuple(r for r in self.results if r.verdict in _USABLE)


def fold_verdicts(results: tuple[JudgeResult, ...] | list[JudgeResult]) -> JudgePanel:
    """Pure fold of judge results into a JudgePanel (no I/O; unit-testable)."""
    results = tuple(results)
    usable = [r for r in results if r.verdict in _USABLE]
    limitations: list[str] = []
    if not usable:
        any_unscored = any(r.verdict == "unscored" for r in results)
        return JudgePanel(results, 0.0, (), ("no_judge",),
                          verdict="unscored" if any_unscored else "unavailable",
                          models=tuple(r.judge for r in results))
    yes = [r for r in usable if r.verdict == "YES"]
    no = [r for r in usable if r.verdict == "NO"]
    scores = [r.score for r in usable if r.score is not None]
    mean_score = sum(scores) / len(scores) if scores else None
    if len(usable) == 1:
        confidence = SINGLE_JUDGE_CONFIDENCE
        limitations.append("single_judge")
        majority, dissent = usable, []
    else:
        if len(yes) == len(no):
            majority, dissent = [], usable       # tie: nobody is the majority
        else:
            majority, dissent = (yes, no) if len(yes) > len(no) else (no, yes)
        confidence = len(majority) / len(usable) if majority else 0.5
    verdict = majority[0].verdict if majority else "SPLIT"
    return JudgePanel(
        results, round(confidence, 4),
        tuple(f"{r.judge}: {r.verdict}" + (f" ({r.score:g})" if r.score is not None else "")
              for r in dissent),
        tuple(limitations), verdict=verdict, score=mean_score,
        models=tuple(r.judge for r in results))


def run_judges(rubric: Rubric, goal: GoalSpec,
               artifacts: list[dict[str, Any]],
               generator_model: str | None = None,
               n: int = JUDGE_PANEL_SIZE) -> JudgePanel | None:
    """``n`` independent judge passes under ``rubric`` folded into a
    JudgePanel; None when nothing is judgeable. Each judge excludes the
    generator, every prior pick, and their model FAMILIES. When fewer than two
    independent judges can be seated the panel carries ``single_judge`` and a
    reduced confidence rather than a second opinion from the same lineage."""
    body = _rubric_body(rubric, goal, artifacts)
    if body is None:
        return None
    routes = _resolve_judge_routes(
        rubric.judge_capability, (generator_model,) if generator_model else (), max(1, n))
    if not routes:
        return fold_verdicts((_judge_once(rubric, body, None, generator_model),))
    results = tuple(_judge_once(rubric, body, r, generator_model) for r in routes[:max(1, n)])
    return fold_verdicts(results)


def _note_verdict(capability: str, model_id: str | None, *, hard_pass: bool,
                  confidence: float) -> None:
    """k113a ledger hook, forwarding the panel confidence when the ledger's
    signature has grown to take it (selection.py is another agent's file)."""
    try:
        from hugpy_oracle import selection as _selection
        fn = _selection.note_verdict
        try:
            takes_conf = "confidence" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            takes_conf = False
        if takes_conf:
            fn(capability, model_id, hard_pass=hard_pass, confidence=confidence)
        else:
            fn(capability, model_id, hard_pass=hard_pass)
    except Exception:  # noqa: BLE001
        pass


def evaluate(goal: GoalSpec, route: RouteDecision,
             artifacts: list[dict[str, Any]], receipt: ExecutionReceipt,
             scorecard: Scorecard) -> Scorecard:
    """The k90c kernel: fill ``judge_results`` on ``scorecard`` per the
    capability's default rubric and update hard_pass/diagnosis/repair_code
    when the judged score misses the quality bar. Analytic capabilities (no
    rubric), failed executions and unjudgeable artifacts return the card
    unchanged; an unavailable/unscored judge is recorded WITHOUT touching
    ``hard_pass`` (the fleet's vision plane being down must not fail routes)."""
    rubric = RUBRICS.get(route.capability)
    if rubric is None or receipt.failure is not None:
        return scorecard

    panel = run_judges(rubric, goal, artifacts, generator_model=route.model_id)
    if panel is None:
        return scorecard

    judge_results = scorecard.judge_results + panel.results
    disagreements = scorecard.disagreements + panel.disagreements
    limitations = tuple(f"limitation:{lim}" for lim in panel.limitations)
    confidence = min(scorecard.confidence, panel.confidence)
    if panel.verdict in ("unavailable", "unscored"):
        return replace(scorecard, judge_results=judge_results,
                       disagreements=disagreements + limitations)

    threshold = THRESHOLDS[goal.quality]
    failing = (panel.score is not None and panel.score < threshold) or \
              (panel.score is None and panel.verdict in ("NO", "SPLIT"))
    # k113a: the verdict is evidence about the PRODUCING model, weighted by
    # how much the judges agreed about it
    _note_verdict(route.capability, route.model_id, hard_pass=not failing,
                  confidence=panel.confidence)
    if not failing:
        if panel.verdict in ("NO", "SPLIT"):   # score cleared the bar but verdict says NO
            disagreements = disagreements + (
                f"{rubric.name} panel: verdict {panel.verdict} but mean score "
                f"{panel.score:g} >= threshold {threshold} — score wins (movie semantics)",)
        return replace(scorecard, judge_results=judge_results,
                       confidence=confidence,
                       disagreements=disagreements + limitations)

    scored = f"mean score {panel.score:g}" if panel.score is not None else "no score"
    rationale = "; ".join(r.rationale for r in panel.usable if r.rationale)
    diagnosis = (f"{rubric.name} judge panel ({len(panel.usable)}/{len(panel.results)} "
                 f"usable, agreement {panel.confidence:.2f}) failed the artifact "
                 f"({scored}, verdict {panel.verdict}, threshold {threshold} "
                 f"for quality={goal.quality.value})"
                 + (f": {rationale}" if rationale else ""))
    if scorecard.diagnosis:
        diagnosis = f"{scorecard.diagnosis}; {diagnosis}"
    return replace(
        scorecard,
        hard_pass=False,
        judge_results=judge_results,
        confidence=confidence,
        disagreements=disagreements + limitations,
        diagnosis=diagnosis,
        repair_code=scorecard.repair_code or RepairCode.INTENT_MISMATCH,
        recommended_repair=scorecard.recommended_repair or
            "regenerate with a bumped seed (bounded repair — see oracle/repair.py)")


__all__ = ["RUBRICS", "Rubric", "DEFAULT_EVALUATED", "THRESHOLDS",
           "JUDGE_PANEL_SIZE", "SINGLE_JUDGE_CONFIDENCE", "JudgePanel",
           "model_family", "fold_verdicts", "parse_judge_verdict",
           "run_judge", "run_judges", "evaluate"]
