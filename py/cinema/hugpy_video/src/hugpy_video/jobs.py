"""Public media-job-bus API (the surface upper layers and the CLI use).

Thin, lazy façade over ``hugpy_video.intel.media_bus`` / ``job_schema`` so
``import hugpy_video.jobs`` costs nothing until a function is called.

Upper layers (the oracle) plug their own bus jobs in with
:func:`register_job` instead of video importing them::

    from hugpy_video.jobs import register_job
    register_job("video_performance", spec_type=PerformanceSpec,
                 runner_key=("oracle", "performance"), queue="gpu",
                 timeout_s=14400, from_dict=performance_from_dict,
                 runner=run_video_performance)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

__all__ = [
    "register_job", "unregister_job", "registered_jobs",
    "enqueue", "get", "list_jobs", "cancel", "work_once", "set_progress",
]


def register_job(name: str, *, spec_type: Type, runner_key: Tuple[str, str],
                 queue: str, timeout_s: int,
                 from_dict: Callable[[dict], object],
                 runner: Callable[[Any, str], Any], replace: bool = True):
    from hugpy_video.intel.job_schema import register_job as _reg
    return _reg(name, spec_type=spec_type, runner_key=runner_key, queue=queue,
                timeout_s=timeout_s, from_dict=from_dict, runner=runner,
                replace=replace)


def unregister_job(name: str) -> None:
    from hugpy_video.intel.job_schema import unregister_job as _unreg
    _unreg(name)


def registered_jobs() -> Dict[str, Any]:
    from hugpy_video.intel.job_schema import JOB_REGISTRY
    return dict(JOB_REGISTRY)


def enqueue(name: str, spec, principal: Optional[str] = None,
            owner: Optional[str] = None, private: bool = False) -> str:
    from hugpy_video.intel import media_bus
    return media_bus.enqueue(name, spec, principal=principal, owner=owner, private=private)


def get(job_id: str) -> dict:
    from hugpy_video.intel import media_bus
    return media_bus.get(job_id)


def list_jobs(include_terminal: bool = False, limit: int = 50, **kw) -> List[dict]:
    from hugpy_video.intel import media_bus
    return media_bus.list_jobs(include_terminal=include_terminal, limit=limit, **kw)


def cancel(job_id: str) -> dict:
    from hugpy_video.intel import media_bus
    return media_bus.cancel(job_id)


def work_once(worker_token: Optional[str] = None) -> Optional[str]:
    from hugpy_video.intel import media_bus
    return media_bus.work_once(worker_token)


def set_progress(job_id: str, blob: dict) -> None:
    from hugpy_video.intel import media_bus
    media_bus.set_progress(job_id, blob)
