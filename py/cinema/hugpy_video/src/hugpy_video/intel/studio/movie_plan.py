"""PER-SEGMENT capability resolution for a studio movie (k58) — the movie-level pin
is a pin PER CAPABILITY, and the whole take-tree is preflighted at SUBMIT.

THE INCIDENT THIS DELETES (operator movie eb9dee56, 2026-07-31). A 2-segment movie
pinned ``wan2.1-t2v-1.3b``. Segment 0 (t2v) rendered fine on the 3090; segment 1
(``joint_mode="still"`` -> capability ``i2v``) then died mid-movie with
``[pinned_model_unavailable] pinned model 'wan2.1-t2v-1.3b' does not serve capability
'i2v'`` — while the refusal itself listed ``clip-i2v-480p`` as available. Real GPU
minutes were spent before a failure that was FULLY KNOWABLE at submit: the movie's
segment capabilities are a pure function of the spec (references / start_image /
joint_mode), and the pin's capabilities are a registry lookup. Nothing about that
answer needed a card.

Two rulings live here, and they are deliberately different rulings:

  1. A MOVIE-LEVEL pin BINDS ONLY THE SEGMENTS WHOSE CAPABILITY IT SERVES. It is the
     movie's *default* model, not a claim about every segment — a t2v pin on a movie
     that also has an i2v splice is a coherent request, not an error. The segments it
     cannot serve resolve their own capable model through the normal studio resolution
     (the router picks), and the substitution is ATTRIBUTED — per-segment ``model_id``,
     ``model_source="capability_fallback"`` and a note naming the pin that did not
     bind — in movie.json, the JobResult and the live progress blob (hence the stage
     log). A silent substitution would be the same betrayal in the other direction.

  2. An EXPLICIT PER-SEGMENT ``model_id`` is NEVER substituted. The caller named that
     model for that shot; if it cannot serve the shot's capability the answer is a
     refusal AT SUBMIT that NAMES THE SEGMENT, not a different model wearing the
     choice. Substitution is for the movie-level default only.

``preflight_movie`` is what makes rule 2 (and a dead capability, and an unknown pin)
land at POST instead of mid-render: it walks the WHOLE take-tree before a job_id
exists. After it passes, a capability-class failure is IMPOSSIBLE later — anything
that still breaks broke for a RUNTIME reason (OOM, weights, a lost worker), which is
exactly the line the operator drew.

DERIVED, NOT DUPLICATED. ``segment_capability`` is the ONE definition of "what does
segment N render"; ``runners/studio_movie.py`` calls it rather than keeping its own
copy, so the preflight and the render can never disagree about what will be asked
for. The capable-model question likewise goes to ``router.capable_model_ids``, which
reuses the router's own candidate gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from hugpy_video.intel.studio.enums import Capability
from hugpy_video.intel.studio.presets import available_menu, capability_verdict
from hugpy_video.intel.studio.registry import MODEL_REGISTRY
from hugpy_video.intel.studio.router import capable_model_ids

# ``model_source`` values recorded per segment (movie.json / JobResult / progress).
SOURCE_SEGMENT = "segment"              # an explicit per-goal model_id (authoritative)
SOURCE_MOVIE = "movie"                  # the movie-level pin, which serves this segment
SOURCE_FALLBACK = "capability_fallback"  # the pin does NOT serve it -> studio resolution
SOURCE_AUTO = "auto"                    # nothing pinned -> studio resolution

# ``reason`` values on a preflight refusal (stable keys for a console).
REASON_COORDINATION_MISMATCH = "coordination_mismatch"
REASON_CAPABILITY_UNSERVABLE = "capability_unservable"
REASON_PIN_UNKNOWN = "pinned_model_unknown"
REASON_PIN_WRONG_CAPABILITY = "pinned_model_unavailable"
REASON_NO_CAPABLE_MODEL = "no_capable_model"


@dataclass(frozen=True, slots=True)
class SegmentModel:
    """How ONE segment's model was decided.

    ``model_id`` is the pin handed to the studio resolution (``None`` = no pin, the
    router picks). ``pinned_model_id`` is the movie-level pin that did NOT bind, set
    only on a ``capability_fallback``, and ``note`` says so in one sentence — the
    attribution that keeps a substitution honest."""
    model_id: "str | None"
    source: str
    pinned_model_id: "str | None" = None
    note: "str | None" = None

    def as_record(self) -> dict:
        """The attribution fields as they land in a movie.json segment node / the
        JobResult (omitting the two that are only meaningful for a fallback)."""
        out = {"model_id": self.model_id, "model_source": self.source}
        if self.pinned_model_id is not None:
            out["pinned_model_id"] = self.pinned_model_id
        if self.note is not None:
            out["model_note"] = self.note
        return out


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """What segment ``index`` will ask the studio for, decided from the SPEC alone."""
    index: int
    segment_id: str
    capability: str
    model: SegmentModel


def segment_capability(spec, goal, index: int) -> str:
    """The capability segment ``index`` renders — a pure function of the spec.

    THE one definition (``runners/studio_movie.py`` reads it here):
      * IDENTITY MOVIE (per-goal or movie-level ``reference_images``): ``id_lock``,
        so the locked subject carries across every scene change — EXCEPT a
        ``vace_extend`` splice, which routes through the VACE path as ``v2v``.
      * segment 0 otherwise: ``i2v`` when the movie carries a ``start_image``, else
        ``t2v``.
      * a later segment otherwise: ``t2v`` for a ``cut`` (a fresh render, no frame
        carry), ``v2v`` for ``vace_extend``, else ``i2v`` (the "still" splice — the
        one that surprised the pinned movie).
    """
    is_id_movie = bool(goal.reference_images or spec.reference_images)
    if index == 0:
        return "id_lock" if is_id_movie else (
            "i2v" if spec.start_image is not None else "t2v")
    if goal.joint_mode == "cut":
        return "id_lock" if is_id_movie else "t2v"
    if goal.joint_mode == "vace_extend":
        return "v2v"
    return "id_lock" if is_id_movie else "i2v"


def _model_capabilities(model_id: str) -> "tuple[str, ...]":
    cfg = MODEL_REGISTRY.get(model_id)
    if cfg is None:
        return ()
    return tuple(sorted(c.value for c in cfg.capabilities))


def _serves(model_id: str, capability: str) -> bool:
    """Does ``model_id`` DECLARE ``capability``? (Registry membership only — whether
    it also fits the geometry/budget is the router's question, and a pin that fits
    badly must still surface the router's own sharpened reason, never a substitution.)"""
    return capability in _model_capabilities(model_id)


def resolve_segment_model(capability: str, goal_model_id: "str | None",
                          movie_model_id: "str | None") -> SegmentModel:
    """Which model this segment asks for, per the k58 ruling (see the header).

    An explicit per-goal ``model_id`` wins unconditionally — it is authoritative even
    when it cannot serve the capability, because the honest answer there is the
    submit-time refusal ``preflight_movie`` already raised, NOT a quiet swap."""
    if goal_model_id is not None:
        return SegmentModel(goal_model_id, SOURCE_SEGMENT)
    if movie_model_id is None:
        return SegmentModel(None, SOURCE_AUTO)
    if _serves(movie_model_id, capability):
        return SegmentModel(movie_model_id, SOURCE_MOVIE)
    serves = ", ".join(_model_capabilities(movie_model_id)) or "nothing on this fleet"
    return SegmentModel(
        None, SOURCE_FALLBACK, pinned_model_id=movie_model_id,
        note=(f"movie-level pin {movie_model_id!r} serves {serves}, not "
              f"{capability!r}; this segment resolved a capable model through the "
              f"studio router instead (the pin still binds the segments it serves)"))


def plan_segments(spec) -> "tuple[SegmentPlan, ...]":
    """The whole take-tree's per-segment (capability, model) decision, from the spec
    alone — no GPU, no geometry, no budget. Both the preflight and the runner walk
    this, so what is refused at submit is exactly what would have been rendered."""
    plans: list[SegmentPlan] = []
    for i, goal in enumerate(spec.goals):
        cap = segment_capability(spec, goal, i)
        plans.append(SegmentPlan(
            index=i, segment_id=goal.segment_id, capability=cap,
            model=resolve_segment_model(cap, goal.model_id, spec.model_id)))
    return tuple(plans)


def preflight_movie(spec) -> "list[dict]":
    """Every SUBMIT-TIME capability problem in the take-tree, one dict per offending
    segment (empty list = nothing of this class can fail later).

    The four refusals, all of them knowable without a card:
      * ``capability_unservable`` — no ratified preset covers what this segment needs
        (the same verdict ``/video/studio/i2v`` refuses on, one layer up);
      * ``pinned_model_unknown`` — the segment's own ``model_id``, or the movie-level
        pin, is not in the registry at all (a typo must not be absorbed by a
        substitution — that would hide it forever);
      * ``pinned_model_unavailable`` — the segment's EXPLICIT ``model_id`` does not
        serve the segment's capability (ruling 3: named, never substituted);
      * ``no_capable_model`` — nothing on this fleet serves the capability, so the
        movie-level fallback has nowhere to go either.

    Each dict carries WHAT IS AVAILABLE (both the capable model ids and the preset
    menu wording) so the caller is never told "no" without being told "instead".
    """
    problems: list[dict] = []
    # A movie-level typo is ONE fact about the movie, reported against the first
    # segment that would have used it — not once per segment.
    movie_pin_unknown = (spec.model_id is not None
                         and spec.model_id not in MODEL_REGISTRY)
    for plan in plan_segments(spec):
        goal = spec.goals[plan.index]
        base = {"index": plan.index, "segment_id": plan.segment_id,
                "capability": plan.capability}
        try:
            cap_enum = Capability(plan.capability)
        except ValueError:  # unreachable: segment_capability emits enum members
            problems.append({**base, "reason": REASON_CAPABILITY_UNSERVABLE,
                             "detail": f"unknown capability {plan.capability!r}",
                             "available": available_menu()})
            continue

        verdict = capability_verdict(cap_enum)
        if not verdict.servable:
            problems.append({**base, "reason": REASON_CAPABILITY_UNSERVABLE,
                             "detail": verdict.refusal,
                             "available": available_menu()})
            continue

        capable = capable_model_ids(cap_enum)

        # An EXPLICIT per-segment pin: authoritative, so it is CHECKED, not replaced.
        if goal.model_id is not None:
            if goal.model_id not in MODEL_REGISTRY:
                problems.append({**base, "reason": REASON_PIN_UNKNOWN,
                                 "model_id": goal.model_id,
                                 "model_source": SOURCE_SEGMENT,
                                 "detail": (f"segment {plan.segment_id!r} pins model_id "
                                            f"{goal.model_id!r}, which is not in the "
                                            f"studio registry"),
                                 "capable_models": list(capable),
                                 "available": available_menu()})
            elif not _serves(goal.model_id, plan.capability):
                problems.append({**base, "reason": REASON_PIN_WRONG_CAPABILITY,
                                 "model_id": goal.model_id,
                                 "model_source": SOURCE_SEGMENT,
                                 "model_capabilities": list(
                                     _model_capabilities(goal.model_id)),
                                 "detail": (f"segment {plan.segment_id!r} explicitly pins "
                                            f"{goal.model_id!r}, which does not serve this "
                                            f"segment's capability {plan.capability!r}. An "
                                            f"explicit per-segment model is never "
                                            f"substituted — repin it or drop it to let the "
                                            f"studio resolve one"),
                                 "capable_models": list(capable),
                                 "available": available_menu()})
            continue

        # The MOVIE-LEVEL pin: a capability it does not serve is a FALLBACK, not a
        # refusal (ruling 1) — but a pin that is not a model at all is a typo, and a
        # typo absorbed by a fallback is a typo that never gets fixed.
        if movie_pin_unknown:
            movie_pin_unknown = False
            problems.append({**base, "reason": REASON_PIN_UNKNOWN,
                             "model_id": spec.model_id,
                             "model_source": SOURCE_MOVIE,
                             "detail": (f"the movie-level model_id {spec.model_id!r} is "
                                        f"not in the studio registry"),
                             "capable_models": list(capable),
                             "available": available_menu()})
            continue

        if not capable:
            problems.append({**base, "reason": REASON_NO_CAPABLE_MODEL,
                             "model_id": plan.model.model_id,
                             "model_source": plan.model.source,
                             "detail": (f"segment {plan.segment_id!r} needs capability "
                                        f"{plan.capability!r} and no model on this fleet "
                                        f"both declares it and can run it"),
                             "capable_models": [],
                             "available": available_menu()})
    return problems


# --------------------------------------------------------------------------- #
# COORDINATION PREFLIGHT (k121) — the OTHER thing that is knowable at submit
#
# ``preflight_movie`` above deletes the capability-class mid-render failure. This
# deletes its cousin: a movie whose PROSE and whose KNOBS describe two different
# films. The operator's cinema session shipped four segments whose text declared
# continuous scenes, a scene drawn from the previous scene's render and a
# recurring character, with every joint left at ``cut``, no parent links, no
# lengthening and no identity — "not a single prompt was created with the proper
# knobs turned". Nothing failed. It rendered, correctly, the wrong film, and the
# GPU hours were spent before anyone could see it.
#
# Like the capability preflight this is a PURE function of the spec (the prose is
# in ``goal.prompt``; the knobs are the goal's own fields), so it belongs at
# submit and nowhere later. Unlike it, most findings are advisory: only a
# ``mismatch`` at or above ``BLOCK_CONFIDENCE`` — the words and the knob in
# outright contradiction, or a take-tree rule the factory will refuse anyway —
# is returned here. Everything else rides the full report, which the caller
# attaches so the operator sees what was set as well as what was refused.
# --------------------------------------------------------------------------- #
def preflight_coordination(spec, *, threshold: "float | None" = None) -> "list[dict]":
    """Every SUBMIT-BLOCKING words-vs-knobs contradiction in the take-tree.

    One dict per offending segment, in ``preflight_movie``'s own vocabulary
    (``index`` / ``segment_id`` / ``reason`` / ``detail``) so a console that
    already renders capability refusals renders these with no new code. An empty
    list means nothing of this class is wrong — NOT that nothing was reviewed;
    :func:`coordination_report` is where the full, never-silent review lives.
    """
    from hugpy_video.hooks import get_prompt_coordinator

    pc = get_prompt_coordinator()
    report = pc.review_goals(spec)
    limit = pc.block_confidence if threshold is None else float(threshold)
    return pc.blocking_mismatches(report, limit)


def coordination_report(spec) -> dict:
    """The FULL k121 review of a movie spec, as the wire dict.

    Always populated, one row per segment, including the segments whose knobs
    already matched their words — the operator's "explicitly reviewed" is a
    positive statement, and a report that only spoke on failure would leave
    "reviewed and fine" indistinguishable from "never reviewed"."""
    from hugpy_video.hooks import get_prompt_coordinator

    return get_prompt_coordinator().review_goals(spec).as_dict()
