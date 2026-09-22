"""Oracle router (k90b): goal -> capability -> best eligible route.

Three decisions live here, all DETERMINISTIC and all recorded on the returned
``RouteDecision`` so the receipt can say why:

  0. AUTHORITY (k97) — the typed gate, run FIRST, before the catalog is even
     read and long before a model is picked (doc §7 Stage 1; invariant 7).
     ``oracle.authority.check`` says which ``(kind, subject)`` permissions the
     route needs and whether ``GoalSpec.rights`` grants them; a shortfall ends
     the route at ``execution == "refused"`` naming exactly what is missing.
     It never downgrades: the doc's non-identifying fallback ("use a licensed
     synthetic voice") is a LATER task, and quietly substituting a lookalike
     would be a worse lie than a precise refusal.
  1. INTENT — when the goal carries no explicit capability, a small
     input-modality + explicit-ask table picks one (``infer_capability``).
     No model call, no scoring: the table IS the contract, and an explicit
     ``GoalSpec.capability`` always wins over it.
  2. MODEL — within the chosen capability, the model comes from the same
     defaults the existing task handlers use: an explicit request wins
     (``requested``), a single eligible model is taken as-is
     (``only-eligible``), otherwise the dispatcher's own task-default
     resolution (``default``; managers/resolvers TASK_DEFAULTS). No new
     scoring scheme.

Eligibility is the k90a catalog's verdict, never re-derived: an ineligible or
unknown capability routes to ``execution == "gap"`` (the typed CAPABILITY_GAP
shape, catalog reasons echoed — an unmapped task string must land here, never
a KeyError). STUDIO (video.*) capabilities route to ``execution == "deferred"``
— k90b explains the best route via the studio's own verdict but does not run the
render pipeline (that stays with the studio job routes until a later slice).

Import discipline mirrors catalog.py: module top level is contracts + catalog
only; every registry read is lazy inside a module-level provider seam
(``_get_capability`` / ``_task_default_model`` / ``_studio_menu`` /
``_placement_for``) so tests monkeypatch them and need no live fleet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hugpy_oracle import authority as oracle_authority
from hugpy_oracle import catalog
from hugpy_oracle.contracts import ArtifactKind, GoalSpec, InputKind

# ---------------------------------------------------------------------------
# Capability -> canonical dispatch task string (the exact key ml_routes.ML_TASKS
# dispatches on). The LEGACY rows are inverse-consistent with
# catalog.LEGACY_TASK_CAPABILITY by test: every one of THOSE values maps back
# to its key through that table. k98's SPEECH rows (folded in below via
# ``**catalog.SPEECH_CAPABILITY_TASK``) are a deliberately SEPARATE family —
# catalog.py's module docstring explains why they are not added to
# LEGACY_TASK_CAPABILITY itself — so they are exempt from that inverse check
# (tests/test_oracle_route.py carves them out explicitly) while still counting
# toward "every dispatchable capability has a CAPABILITY_TASK entry".
# ---------------------------------------------------------------------------

CAPABILITY_TASK: dict[str, str] = {
    "text.chat":        "text-generation",
    "text.summarize":   "text-summarization",
    "text.keywords":    "keyword-extraction",
    "text.embed":       "feature-extraction",
    "text.similarity":  "sentence-similarity",
    "audio.transcribe": "automatic-speech-recognition",
    "image.understand": "image-text-to-text",
    "image.generate":   "text-to-image",
    "image.transform":  "image-to-image",
    "image.depth":      "depth-estimation",
    "image.detect":     "object-detection",
    "image.classify":   "image-classification",
    "image.segment":    "image-segmentation",
    "doc.extract":      "document-extraction",
    "web.fetch":        "url-extraction",
    **catalog.SPEECH_CAPABILITY_TASK,
}

# ---------------------------------------------------------------------------
# Intent inference — the input-modality + explicit-ask table.
# ---------------------------------------------------------------------------

# Verbs that read as "change this image" rather than "tell me about this image".
_TRANSFORM_VERBS = (
    "transform", "restyle", "stylize", "repaint", "recolor", "redraw",
    "convert", "turn this", "turn it", "make it", "make this", "edit",
    "remix", "variation", "in the style of",
)
# Verbs that read as "shorten this text".
_SUMMARIZE_VERBS = ("summarize", "summarise", "summary", "tl;dr", "tldr",
                    "condense", "shorten this")
_QUESTION_WORDS = ("what", "who", "where", "when", "why", "how", "which",
                   "describe", "explain", "identify", "is there", "are there",
                   "does", "do you see", "can you see", "tell me")


def _question_like(prompt: str) -> bool:
    p = prompt.strip().lower()
    return "?" in p or any(p.startswith(w) for w in _QUESTION_WORDS)


def _has_any(prompt: str, verbs: tuple[str, ...]) -> bool:
    p = prompt.lower()
    return any(v in p for v in verbs)


def infer_capability(goal: GoalSpec) -> tuple[str, str]:
    """(capability, why) from the deterministic table. First match wins:

      image attached + transform verb, not question-like -> image.transform
      image attached (question-like or anything else)   -> image.understand
      audio attached                                    -> audio.transcribe
      video attached                                    -> audio.transcribe
      url attached                                      -> web.fetch
      no attachment + summarize verb                    -> text.summarize
      default                                           -> text.chat
    """
    prompt = goal.raw_prompt
    kinds = {i.kind for i in goal.inputs}
    if InputKind.IMAGE in kinds:
        if _has_any(prompt, _TRANSFORM_VERBS) and not _question_like(prompt):
            return ("image.transform",
                    "image attached + transform verb, not question-like")
        return ("image.understand",
                "image attached; question-like or descriptive ask")
    if InputKind.AUDIO in kinds:
        return ("audio.transcribe", "audio attached")
    if InputKind.VIDEO in kinds:
        return ("audio.transcribe",
                "video attached; transcription is the only synchronous "
                "capability that accepts video")
    # InputKind has no DOCUMENT member — doc.extract is reached by naming the
    # capability explicitly, never inferred.
    if InputKind.URL in kinds:
        return ("web.fetch", "url attached")
    if _has_any(prompt, _SUMMARIZE_VERBS):
        return ("text.summarize", "no attachment + summarize verb")
    return ("text.chat", "default: no attachment, no routing verb")


# ---------------------------------------------------------------------------
# Provider seams — lazy reads, monkeypatchable in tests.
# ---------------------------------------------------------------------------


def _get_capability(name: str):
    return catalog.get_capability(name)


def _select_model(goal: GoalSpec, capability: str, view: Any
                  ) -> tuple[str | None, str, tuple[str, ...]]:
    """Evidence-ranked selection over the view's eligible models
    (``selection.select``: VRAM/budget, quality, latency, reliability ledger,
    matrix). Never raises; (None, "", reasons) means "no opinion" and the
    caller falls back to the legacy default. The selector never calls the
    router, so there is no recursion."""
    try:
        from hugpy_oracle import selection
        sel = selection.process_selector()
        if sel is None:
            return None, "", ("selection disabled",)
        d = selection.select(capability, view=view, goal=goal, matrix=sel.matrix(),
                             ledger=sel.ledger, policy=sel.policy,
                             model_health=sel.model_health, model_vram_gib=sel.model_vram_gib)
    except Exception as exc:  # noqa: BLE001 — selection faults degrade to the legacy default
        return None, "", (f"selection unavailable: {type(exc).__name__}: {exc}",)
    if d.gap or not d.model_id:
        return None, "", tuple(f"selection: {step}" for step in d.steps[-2:])
    chosen = next((v for v in d.ranked if v.selected), None)
    why = "; ".join(chosen.reasons) if chosen else ""
    reasons = (f"selection: {d.rationale} -> {d.model_id} (fallback {d.fallback}); {why}",)
    if d.rejected:
        reasons += ("selection rejected: " + ", ".join(f"{r.model_id}@{r.rejected_at}" for r in d.rejected),)
    return d.model_id, f"selected:{d.rationale}", reasons


def _task_default_model(task: str) -> str | None:
    """The dispatcher's OWN default for a task-only request (TASK_DEFAULTS via
    resolve_model_key) — recorded, never re-invented. None when unresolvable."""
    try:
        from hugpy_engine.resolvers.model_resolver import resolve_model_key
        return resolve_model_key(task=task)
    except Exception:  # noqa: BLE001 — no default is data, not a fault
        return None


def _studio_menu() -> str:
    """available_menu() — what the studio CAN render, for deferred responses."""
    try:
        from hugpy_video.intel.studio.presets import available_menu
        return available_menu()
    except Exception:  # noqa: BLE001
        return ""


def _placement_for(model_id: str, task: str) -> str | None:
    """placement.json pin for (model, task), or None for the default route
    (DelegatingRunner: live worker if one serves it, else central-local)."""
    try:
        from hugpy_engine.resolvers.model_resolver import peer_for
        peer = peer_for(model_id, task)
        return getattr(peer, "name", None) if peer is not None else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# The decision.
# ---------------------------------------------------------------------------


class RouteRefusal(ValueError):
    """A typed request-shape refusal (e.g. a requested model that does not
    serve the capability) — the route answers 400, never a traceback."""


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Where a goal goes and why — the object runtime + scorecard consume.

    ``execution`` is the branch: "execute" (synchronous dispatch), "deferred"
    (video.*: explained, not run), "gap" (typed CAPABILITY_GAP), "refused"
    (k97: the typed authority gate said no — ``authority.missing`` names every
    permission the request did not bring). ``task`` is the legacy dispatch task
    string ("execute" only). ``placement`` names a placement.json peer pin, or
    "auto" for the DelegatingRunner default. ``dispatch_params`` is
    ``catalog.capability_params(capability)`` (k98) — fixed keyword arguments
    the capability itself implies (e.g. ``audio.transcribe.word_timestamps``
    always dispatches with ``word_timestamps=True``); it rides on an
    ``"execute"`` decision for whoever builds the actual dispatch kwargs
    (``runtime.execute_route``, not this module) to merge in, so the flag
    never has to be re-derived or re-typed downstream."""
    capability: str
    execution: str
    source: str = ""
    task: str | None = None
    model_id: str | None = None
    model_rationale: str = ""       # requested | only-eligible | default | deterministic-local
    placement: str = "auto"
    inferred: bool = False
    inference_reason: str = ""
    produces: tuple[ArtifactKind, ...] = ()
    reasons: tuple[str, ...] = ()   # gap reasons / deferred verdict / advisories
    alternatives: str = ""          # studio available_menu (deferred only)
    model_ids: tuple[str, ...] = field(default=())
    authority: oracle_authority.AuthorityDecision | None = None
    dispatch_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "execution": self.execution,
            "source": self.source,
            "task": self.task,
            "model_id": self.model_id,
            "model_rationale": self.model_rationale,
            "placement": self.placement,
            "inferred": self.inferred,
            "inference_reason": self.inference_reason,
            "produces": [k.value for k in self.produces],
            "reasons": list(self.reasons),
            "alternatives": self.alternatives,
            "model_ids": list(self.model_ids),
            "authority": self.authority.to_dict() if self.authority else None,
            "dispatch_params": dict(self.dispatch_params),
        }


def resolve_route(goal: GoalSpec, requested_model: str | None = None) -> RouteDecision:
    """Goal -> RouteDecision through the authority gate and the k90a catalog.
    Never raises for a missing/ineligible capability (that is the gap SHAPE) or
    for missing authority (that is the ``"refused"`` SHAPE); raises RouteRefusal
    only for a request-shape fault (a requested model outside the capability's
    eligible set)."""
    if goal.capability is not None:
        name, inferred, why = goal.capability, False, "explicit capability in request"
    else:
        name, why = infer_capability(goal)
        inferred = True

    # Stage 1 — the typed authority gate, BEFORE the catalog read and before
    # any model pick. An unauthorized identity/voice request must not learn a
    # route, reach a worker, or be quietly served by something else.
    decision = oracle_authority.check(goal, name)
    if not decision.ok:
        return RouteDecision(
            capability=name, execution="refused", inferred=inferred,
            inference_reason=why, authority=decision,
            reasons=(decision.reason,) + tuple(
                f"missing {kind.value} authorization for {subject}"
                for kind, subject in decision.missing))

    view = _get_capability(name)
    if view is None:
        return RouteDecision(
            capability=name, execution="gap", inferred=inferred,
            inference_reason=why, authority=decision,
            reasons=(f"unknown capability {name!r}: not in the unified "
                     f"catalog (GET /oracle/capabilities lists what exists)",))

    if name.startswith("video."):
        # k90b explains the best video route via the studio's own verdict but
        # does not execute it — the studio job routes own that pipeline.
        return RouteDecision(
            capability=name, execution="deferred", source=view.source.value,
            model_id=view.model_ids[0] if view.model_ids else None,
            model_ids=view.model_ids,
            model_rationale="studio-router" if view.model_ids else "",
            inferred=inferred, inference_reason=why, authority=decision,
            produces=view.produces,
            reasons=view.eligibility.reasons or (
                ("servable per the studio capability verdict",)
                if view.eligibility.eligible else ()),
            alternatives=_studio_menu())

    if not view.eligibility.eligible:
        return RouteDecision(
            capability=name, execution="gap", source=view.source.value,
            inferred=inferred, inference_reason=why, produces=view.produces,
            authority=decision, reasons=view.eligibility.reasons)

    task = CAPABILITY_TASK.get(name)
    if task is None:  # a legacy capability outside the dispatch table — drift alarm
        return RouteDecision(
            capability=name, execution="gap", source=view.source.value,
            inferred=inferred, inference_reason=why, produces=view.produces,
            authority=decision,
            reasons=(f"capability {name!r} has no dispatch task mapping "
                     f"(oracle/router.CAPABILITY_TASK drift)",))

    deterministic = task in catalog.DETERMINISTIC_TASKS
    extra_reasons: tuple[str, ...] = ()
    if requested_model is not None:
        if view.model_ids and requested_model not in view.model_ids:
            raise RouteRefusal(
                f"model {requested_model!r} does not serve capability {name!r} "
                f"on this fleet; eligible: {list(view.model_ids)}")
        model_id, rationale = requested_model, "requested"
    elif deterministic:
        model_id, rationale = None, "deterministic-local"
    elif len(view.model_ids) == 1:
        model_id, rationale = view.model_ids[0], "only-eligible"
    else:
        # TODO-4 / k113a: ONE selection policy. With more than one eligible
        # model the evidence-ranked selector decides (VRAM vs budget, quality
        # profile, latency budget, reliability ledger, routing matrix), and
        # its decision rides on the route's reasons. The legacy TASK_DEFAULTS
        # branch remains only as the fallback when selection has no opinion.
        model_id, rationale, sel_reasons = _select_model(goal, name, view)
        if model_id is None:
            model_id = _task_default_model(task)
            if model_id not in view.model_ids:
                # default resolution failed or landed outside the eligible set
                # (e.g. the default is operator-blocked) — take the first eligible.
                model_id = view.model_ids[0]
            rationale = "default"
        extra_reasons = tuple(sel_reasons)

    if deterministic:
        placement = "local"
    elif model_id:
        placement = _placement_for(model_id, task) or "auto"
    else:
        placement = "auto"
    return RouteDecision(
        capability=name, execution="execute", source=view.source.value,
        task=task, model_id=model_id, model_rationale=rationale,
        placement=placement, inferred=inferred, inference_reason=why,
        produces=view.produces, model_ids=view.model_ids, authority=decision,
        reasons=tuple(view.eligibility.reasons) + extra_reasons,  # advisory + selection reasons
        dispatch_params=catalog.capability_params(name))


__all__ = ["CAPABILITY_TASK", "RouteDecision", "RouteRefusal",
           "infer_capability", "resolve_route"]
