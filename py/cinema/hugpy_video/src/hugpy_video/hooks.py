"""Upper-layer callbacks video needs but must not import (EXTRACTION_GUIDE §3.3).

``hugpy_video`` executes immutable specs; ``hugpy_oracle`` decides. Two places
in video used to reach *up* into the oracle's ``prompt_coordination`` module:

* ``intel.prompt_spread.coordinate_spread`` — after a spread reply is parsed,
  review the rows' words against their knobs and apply the mechanical
  decisions (joint mode, parent pointer, seed, frame count, identity refs).
* ``intel.studio.movie_plan.preflight_coordination`` /
  ``coordination_report`` — the submit-time words-vs-knobs review of a
  ``StudioMovieSpec``.

Both now call the :class:`PromptCoordinator` installed here. The default is a
no-op coordinator whose report says *nothing was reviewed* (``reviewed: False``)
— never a silent "reviewed and fine". The oracle installs its implementation at
composition time (``hugpy_server`` wiring, or an oracle CLI)::

    from hugpy_video.hooks import set_prompt_coordinator
    from hugpy_oracle.relay import prompt_coordination as PC

    class OracleCoordinator:
        block_confidence = PC.BLOCK_CONFIDENCE
        def review(self, rows, *, notes="", context=None, llm=None):
            return PC.review(rows, notes=notes, context=context, llm=llm)
        def review_goals(self, spec, *, context=None):
            return PC.review_goals(spec, context=context)
        def apply_decisions(self, rows, report):
            return PC.apply_decisions(rows, report)
        def blocking_mismatches(self, report, threshold=None):
            return PC.blocking_mismatches(
                report, PC.BLOCK_CONFIDENCE if threshold is None else threshold)

    set_prompt_coordinator(OracleCoordinator())

Report objects are opaque to video: it only calls ``as_dict()`` on them and
reads ``segment_id`` / ``knob`` off the applied decisions. Invariant 9 (no
generated prompt text crosses rows) is the coordinator's to keep; the null
coordinator trivially keeps it by writing nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

__all__ = [
    "CoordinationReportLike",
    "KnobDecisionLike",
    "PromptCoordinator",
    "NullPromptCoordinator",
    "NullCoordinationReport",
    "get_prompt_coordinator",
    "set_prompt_coordinator",
    "reset_hooks",
]


@runtime_checkable
class CoordinationReportLike(Protocol):
    """What video reads off a review report: its wire shape."""

    def as_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class KnobDecisionLike(Protocol):
    """One applied decision: which row, which knob (the value is read off the row)."""

    segment_id: str
    knob: str


@runtime_checkable
class PromptCoordinator(Protocol):
    """Words-vs-knobs review for a prompt set. Implemented by ``hugpy_oracle``."""

    #: Confidence at or above which a ``mismatch`` blocks a submit.
    block_confidence: float

    def review(self, rows: Sequence[Mapping[str, Any]], *, notes: str = "",
               context: Optional[Mapping[str, Any]] = None,
               llm: Any = None) -> CoordinationReportLike:
        """Review generated/enhanced prompt rows (``segment_id``, ``prompt``,
        ``locked`` + current knobs). Every row must appear in the report."""
        ...

    def review_goals(self, spec: Any, *,
                     context: Optional[Mapping[str, Any]] = None) -> CoordinationReportLike:
        """:meth:`review` for a built ``StudioMovieSpec`` (submit preflight)."""
        ...

    def apply_decisions(self, rows: Sequence[Mapping[str, Any]],
                        report: CoordinationReportLike,
                        ) -> Tuple[List[Dict[str, Any]], List[KnobDecisionLike]]:
        """Apply the report's ``set`` decisions to copies of ``rows``; return
        ``(rows, applied)``. May write mechanics only, never ``prompt``."""
        ...

    def blocking_mismatches(self, report: CoordinationReportLike,
                            threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Submit-blocking rows in ``movie_plan.preflight_movie``'s vocabulary
        (``index`` / ``segment_id`` / ``reason`` / ``detail``)."""
        ...


class NullCoordinationReport:
    """The honest empty report: nothing reviewed, nothing decided."""

    decisions: Tuple[Any, ...] = ()
    segments: Tuple[Any, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"reviewed": False, "coordinator": "none",
                "segments": [], "decisions": [], "expectations": []}


class NullPromptCoordinator:
    """Default: no review, no decisions, nothing blocks (single-box / tests)."""

    block_confidence = 0.8

    def review(self, rows, *, notes: str = "", context=None, llm=None):
        return NullCoordinationReport()

    def review_goals(self, spec, *, context=None):
        return NullCoordinationReport()

    def apply_decisions(self, rows, report):
        return [dict(r) for r in rows], []

    def blocking_mismatches(self, report, threshold=None):
        return []


_providers: Dict[str, Any] = {}
_defaults: Dict[str, Any] = {"prompt_coordinator": NullPromptCoordinator()}


def get_prompt_coordinator() -> PromptCoordinator:
    return _providers.get("prompt_coordinator", _defaults["prompt_coordinator"])


def set_prompt_coordinator(impl: PromptCoordinator | None) -> None:
    """Install (or, with ``None``, uninstall) the coordinator implementation."""
    if impl is None:
        _providers.pop("prompt_coordinator", None)
    else:
        _providers["prompt_coordinator"] = impl


def reset_hooks() -> None:
    """Drop every installed hook (tests)."""
    _providers.clear()
