"""ROUTE GLUE for the k121 coordination review — the knobs must match the words.

``video_intel/prompt_coordination.py`` is the kernel (pure, deterministic, no
Flask). This module is the three-line adapter each video prompt path needs, kept
OUT of ``video_routes.py`` on purpose: that file is 5.5k lines and carries other
agents' in-flight work, so every route touched here gets a ONE-LINE call instead
of thirty lines of inline glue.

Three entry points, one per path the operator's directive names:

  :func:`spread_coordination`  ``POST /video/prompt/assist`` mode="spread" — the
                               whole-movie generator. The review runs over the
                               written rows PLUS the locked ones and the
                               ratchet-safe knobs are applied to the result rows.
  :func:`assist_coordination`  the same endpoint's ``detail`` / ``generate``
                               modes — one enhanced or generated prompt, reviewed
                               against the neighbours the caller sent.
  :func:`movie_coordination`   ``POST /video/studio/movie`` — a blocking submit
                               preflight beside k58's capability preflight.

Every one of them returns DATA (a dict, or a (payload, status) tuple), never an
exception, and every one of them is safe to ignore: a caller that drops the
extra response key behaves exactly as it did before this file existed.

No pathlib. os.path only (there is none here).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Response/refusal key the UI reads. Stable — the prompt cards key their badges
#: off it.
COORDINATION_KEY = "coordination"
#: Refusal code for a submit blocked by a words-vs-knobs contradiction. Mirrors
#: the shape of ``movie_capability_preflight_failed`` so a console that already
#: renders one renders the other.
CODE_COORDINATION_FAILED = "movie_coordination_mismatch"
#: The body flag an operator sets to submit anyway. A mismatch is a JUDGEMENT,
#: not a law of physics — the operator can always be right about their own film,
#: so the refusal names its own override in the same breath.
OVERRIDE_KEY = "coordination_override"

#: Said when the caller sent no neighbours: a lone row has no timeline, and a
#: join reviewed against an unknown neighbour would invent a mismatch.
NO_NEIGHBOURS_NOTE = (
    "no neighbouring segments rode this request, so the joins were not reviewed "
    "— send context.previous_segment / context.next_segment to review them")


def _empty_report(note: str) -> Dict[str, Any]:
    """A report that says WHY it is empty. Never ``None`` on the wire: 'nothing
    was reviewed' and 'everything was fine' must not look the same."""
    from hugpy_oracle.relay import prompt_coordination as PC

    return {"version": PC.REPORT_VERSION, "status": PC.STATUS_OK,
            "counts": {s: 0 for s in PC.STATUS_ORDER} | {"segments": 0},
            "llm_used": False, "notes": [note], "segments": [],
            "expectations": [], "decisions": []}


# --------------------------------------------------------------------------- #
# mode="spread"
# --------------------------------------------------------------------------- #
def spread_coordination(req, parsed: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """Review + set knobs on a parsed spread reply, in place. Returns ``parsed``.

    Delegates to ``prompt_spread.coordinate_spread`` (which owns the row
    assembly, because it owns the request shape). Failures are SWALLOWED and
    reported as a note: a coordination review must never turn a spread that
    produced good prose into a 502 — the prose is the product, the review is the
    check on it.
    """
    try:
        from hugpy_video.intel import prompt_spread

        return prompt_spread.coordinate_spread(req, parsed, context)
    except Exception as exc:  # noqa: BLE001 — the prose is still good
        parsed[COORDINATION_KEY] = _empty_report(
            f"the coordination review failed and was skipped: {exc}")
        for seg in parsed.get("segments") or []:
            seg.setdefault("knobs", {})
        return parsed


# --------------------------------------------------------------------------- #
# mode="detail" / "generate"
# --------------------------------------------------------------------------- #
def assist_coordination(typed_ctx: Dict[str, Any], prompt_text: str,
                        ) -> Dict[str, Any]:
    """Review ONE generated/enhanced prompt against the neighbours it was given.

    The single-row path is where the operator's failure is easiest to miss: the
    UI hands the writer ``context.previous_segment`` with its join rendered as a
    sentence, the writer honours it, and the row's own ``joint_mode`` is whatever
    it already was. So the same kernel runs here, over the same typed context
    ``prompt_spread.render_context_preface`` renders for the model — what the
    writer was TOLD is exactly what the review checks.

    Returns the report dict, always. When the caller sent no neighbours the
    report is empty and SAYS SO (:data:`NO_NEIGHBOURS_NOTE`) rather than
    inventing a first-shot mismatch out of a missing field.
    """
    from hugpy_oracle.relay import prompt_coordination as PC

    previous = typed_ctx.get("previous_segment")
    nxt = typed_ctx.get("next_segment")
    if not previous and not nxt:
        return _empty_report(NO_NEIGHBOURS_NOTE)

    current = dict(typed_ctx.get("segment") or {})
    current.setdefault("segment_id", "current")
    # The row is reviewed on the text that was just WRITTEN, not on the draft it
    # replaced — the review's whole subject is the new prose.
    current["prompt"] = prompt_text or current.get("prompt") or ""
    current["locked"] = False

    rows: List[Dict[str, Any]] = []
    if previous:
        rows.append({**previous, "locked": True,
                     "segment_id": previous.get("segment_id") or "previous"})
    rows.append(current)
    if nxt:
        rows.append({**nxt, "locked": True,
                     "segment_id": nxt.get("segment_id") or "next"})
    for i, row in enumerate(rows):
        row["index"] = i

    ctx: Dict[str, Any] = {}
    ident = typed_ctx.get("identity_profile")
    if ident is not None:
        ctx["identity_profiles"] = ident
    # A two/three-row window is not a timeline: an identity CAPTURE proposal
    # needs the whole set to know where a character is born, so it is off here.
    ctx["allow_identity_capture"] = False
    try:
        return PC.review(rows, context=ctx).as_dict()
    except Exception as exc:  # noqa: BLE001
        return _empty_report(f"the coordination review failed and was skipped: {exc}")


# --------------------------------------------------------------------------- #
# POST /video/studio/movie
# --------------------------------------------------------------------------- #
def movie_coordination(spec, body: Dict[str, Any]
                       ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """``(refusal_body_or_None, report)`` for a studio-movie submit.

    Sits beside k58's ``preflight_movie`` and refuses in the same style: the
    whole take-tree is walked BEFORE a job_id exists, the refusal is PER SEGMENT
    and carries the operator's own words as evidence, and the caller is never
    told "no" without being told how to proceed — the refusal names
    :data:`OVERRIDE_KEY` in its own text.

    The report comes back either way. A movie that passes still gets its full
    review attached, because "reviewed and fine" has to be visible; that is the
    entire lesson of the incident this exists for.
    """
    from hugpy_video.intel.studio.movie_plan import coordination_report, preflight_coordination

    try:
        report = coordination_report(spec)
        problems = preflight_coordination(spec)
    except Exception as exc:  # noqa: BLE001 — never 500 a submit over a review
        return None, _empty_report(
            f"the coordination review failed and was skipped: {exc}")

    if not problems or bool(body.get(OVERRIDE_KEY)):
        if problems:
            report.setdefault("notes", []).append(
                f"{len(problems)} coordination mismatch(es) were OVERRIDDEN by the "
                f"operator at submit ({OVERRIDE_KEY}=true) and are recorded here")
        return None, report

    first = problems[0]
    return {
        "error": (f"this movie's words and its knobs disagree: "
                  f"{first.get('detail')}"
                  + (f" (and {len(problems) - 1} more segment(s))"
                     if len(problems) > 1 else "")
                  + f". Fix the knobs, change the prose, or resubmit with "
                    f"{OVERRIDE_KEY}=true to render it as written."),
        "code": CODE_COORDINATION_FAILED,
        "segments": problems,
        COORDINATION_KEY: report,
    }, report
