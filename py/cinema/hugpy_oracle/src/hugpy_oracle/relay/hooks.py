"""What the oracle plugs INTO the video package, without video importing it.

``hugpy_video`` executes; ``hugpy_oracle`` decides. Video must not import the
oracle (``PARTITION.md``: the cycle break), so wherever video used to call
``prompt_coordination`` or the ``(oracle, performance)`` bus runner directly,
it now calls a hook (``hugpy_video.hooks``) with a no-op default. This module
is the oracle's side of that seam:

* :class:`OraclePromptCoordinator` — the ``PromptCoordinator`` implementation
  over :mod:`hugpy_oracle.relay.prompt_coordination`, with exactly the
  methods ``prompt_spread.coordinate_spread`` and ``studio.movie_plan`` used
  (``review`` / ``apply_decisions`` / ``review_goals`` /
  ``blocking_mismatches``, plus the ``block_confidence`` threshold).
* :func:`performance_runner` — the ``(oracle, performance)`` media-bus
  runner and its spec type, for video's runner table.
* :func:`install_hooks` — installs the coordinator through
  ``hugpy_video.hooks.set_prompt_coordinator`` and registers the
  ``video_performance`` bus job through ``hugpy_video.jobs.register_job``
  (envelope + spec deserializer + runner), and reports what was wired. Safe
  to call against a video without either seam (nothing is wired, nothing
  raises).

The composition root (``hugpy_server.wsgi_app`` or ``hugpy-oracle
install-hooks``) calls :func:`install_hooks` once at startup.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["OraclePromptCoordinator", "performance_runner", "register_performance_job", "install_hooks"]


class OraclePromptCoordinator:
    """``hugpy_video.hooks.PromptCoordinator`` backed by the oracle's k121
    review. Every method defers the import so building the coordinator costs
    nothing and video's boot never pays for the oracle."""

    name = "hugpy_oracle.relay.prompt_coordination"

    @property
    def block_confidence(self) -> float:
        from hugpy_oracle.relay.prompt_coordination import BLOCK_CONFIDENCE
        return float(BLOCK_CONFIDENCE)

    def review(self, rows: Any, *, notes: str = "",
               context: Optional[Mapping[str, Any]] = None,
               llm: Optional[Callable[[str], str]] = None) -> Any:
        from hugpy_oracle.relay.prompt_coordination import review
        return review(rows, notes=notes, context=context, llm=llm)

    def apply_decisions(self, rows: Any, report: Any
                        ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        from hugpy_oracle.relay.prompt_coordination import apply_decisions
        return apply_decisions(rows, report)

    def review_goals(self, spec: Any, *,
                     context: Optional[Mapping[str, Any]] = None) -> Any:
        from hugpy_oracle.relay.prompt_coordination import review_goals
        return review_goals(spec, context=context)

    def blocking_mismatches(self, report: Any,
                            threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        from hugpy_oracle.relay.prompt_coordination import blocking_mismatches
        limit = self.block_confidence if threshold is None else float(threshold)
        return blocking_mismatches(report, limit)

    def coordinate(self, rows: Sequence[Mapping[str, Any]], *, notes: str = "",
                   context: Optional[Mapping[str, Any]] = None,
                   llm: Optional[Callable[[str], str]] = None
                   ) -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, Any]]:
        """``review`` then ``apply_decisions`` in one call:
        ``(applied_rows, applied_decisions, report_dict)``."""
        report = self.review(rows, notes=notes, context=context, llm=llm)
        applied_rows, applied = self.apply_decisions(rows, report)
        return applied_rows, applied, report.as_dict()

    def report_dict(self, report: Any) -> Dict[str, Any]:
        return report.as_dict()


def performance_runner() -> Tuple[Tuple[str, str], Callable[..., Any], type]:
    """``(runner_key, run_video_performance, PerformanceSpec)`` for video's
    job registry. The relay's module top is stdlib-only, so this is cheap."""
    from hugpy_oracle.relay.performance_relay import (
        RUNNER_KEY,
        PerformanceSpec,
        run_video_performance,
    )
    return RUNNER_KEY, run_video_performance, PerformanceSpec


def _import(name: str) -> Any:
    import importlib
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 — a video without this seam yet
        logger.debug("hugpy_oracle.install_hooks: %s unavailable (%s)", name, exc)
        return None


def register_performance_job(video_jobs: Any = None) -> Optional[str]:
    """Register the ``video_performance`` bus job with ``hugpy_video.jobs``
    (``register_job``: envelope, spec deserializer and runner in one call).
    Returns the registered job name, or ``None`` when video has no job
    registration seam."""
    jobs = video_jobs if video_jobs is not None else _import("hugpy_video.jobs")
    register = getattr(jobs, "register_job", None) if jobs is not None else None
    if not callable(register):
        return None
    from hugpy_oracle.relay.performance_relay import (
        JOB_NAME,
        JOB_QUEUE,
        JOB_TIMEOUT_S,
        performance_from_dict,
    )
    key, runner, spec_type = performance_runner()
    register(JOB_NAME, spec_type=spec_type, runner_key=key, queue=JOB_QUEUE,
             timeout_s=JOB_TIMEOUT_S, from_dict=performance_from_dict, runner=runner)
    return JOB_NAME


def install_hooks(video_hooks: Any = None, video_jobs: Any = None) -> Dict[str, Any]:
    """Install the oracle's implementations into video's seams.

    ``video_hooks`` / ``video_jobs`` may be passed explicitly (tests);
    otherwise ``hugpy_video.hooks`` / ``hugpy_video.jobs`` are imported
    lazily. Returns ``{"video_hooks": bool, "prompt_coordinator": <setter or
    None>, "performance_runner": <job name or None>}`` so the composition
    root can log what was and was not wired.
    """
    report: Dict[str, Any] = {"video_hooks": False, "prompt_coordinator": None,
                              "performance_runner": None}
    hooks = video_hooks if video_hooks is not None else _import("hugpy_video.hooks")
    if hooks is not None:
        report["video_hooks"] = True
        setter = getattr(hooks, "set_prompt_coordinator", None)
        if callable(setter):
            setter(OraclePromptCoordinator())
            report["prompt_coordinator"] = "set_prompt_coordinator"
    report["performance_runner"] = register_performance_job(video_jobs)
    return report
