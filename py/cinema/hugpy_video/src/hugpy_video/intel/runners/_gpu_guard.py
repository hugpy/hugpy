"""Shared GPU-worker guard — extracted VERBATIM from runners/imagegen.py.

`guard_gpu_worker(model_id, job_id)` centralizes the 2026-07-03 central-meltdown
fix so every generation runner (generate_image, generate_scene, ...) shares ONE
policy. DelegatingRunner runs LOCAL unconditionally when the worker provider
returns no live worker (the HUGPY_LOCAL_FALLBACK gate only covers
selected-then-failed). During the worker's warm-up window that loaded a multi-GB
diffusion model onto central's CPU inside gunicorn (13 GB RSS, box-wide
timeouts). Policy here mirrors the plane's own: when a fleet EXISTS (provider
registered) but no live worker serves this model, refuse with retryable data
instead of melting central. A standalone single-box deploy (no provider
registered) keeps local generation — that posture is the product. Override with
HUGPY_VIDEOGEN_LOCAL=always.

The block below is byte-identical to imagegen's prior inline guard; only
`spec.model_id` became the `model_id` parameter.
"""
from __future__ import annotations

import os
from typing import Optional

from hugpy_video.intel.result_schema import JobError, JobResult


def guard_gpu_worker(model_id: str, job_id: str) -> Optional[JobResult]:
    """Return a refusal JobResult when no live worker serves `model_id` and this
    box may not serve it locally; else None (proceed to normal routing).

    ORDER MATTERS. Check for a LIVE WORKER first; apply the per-box
    no-local-serving policy only as the fallback when none is found. The prior
    version checked the policy FIRST, so on a central box (HUGPY_NO_LOCAL_SERVING)
    it refused EVERY generation before routing could ever select a capable
    worker — a model a worker was already serving (e.g. sd-turbo on a comfy box)
    got refused despite a healthy fleet. This mirrors DelegatingRunner, which
    enforces the same policy only AFTER a worker-selection attempt.
    """
    _videogen_local = (
        os.environ.get("HUGPY_VIDEOGEN_LOCAL", "").strip().lower()
        in ("always", "1", "true", "yes", "on"))

    # 1) Can the fleet serve this model? The worker provider is the SAME
    #    placement decision /v1 chat uses (WorkerStore.pick_for_model): it returns
    #    a worker that is already serving the model OR a placement-chosen capable
    #    box that will cold-load it just-in-time — for a comfy model, a live comfy
    #    box that advertises the checkpoint installed. If it returns a worker, never
    #    refuse here — let resolve()/DelegatingRunner relay the job so the worker's
    #    evict-to-fit makes room and comfy cold-loads the checkpoint, exactly the
    #    way a cold LLM is placed. Only when NO worker can serve it do the honest
    #    refusals below fire.
    try:
        from hugpy_engine.resolvers.remote import get_worker_provider
        provider = get_worker_provider()
    except ImportError:
        provider = None
    live_worker = None
    if provider is not None:
        try:
            try:
                live_worker = provider(model_id, None)
            except TypeError:   # provider may predate the pool arg
                live_worker = provider(model_id)
        except Exception:
            live_worker = None  # a broken provider must not crash the runner
    if live_worker is not None:
        return None

    # No worker CAN serve it (cold-placement found nothing). Pull the specific
    # reason from the SAME no-worker diagnostic the /v1 relay uses
    # (remote._no_worker_detail -> the fleet's explain_no_worker), so a comfy
    # model with no capable/present box says exactly why (comfy backend down, no
    # comfy box online, or the checkpoint installed nowhere) instead of only the
    # generic policy refusal. "" (unregistered seam / standalone box / a truly
    # transient miss) leaves the message byte-identical to before.
    detail = ""
    try:
        from hugpy_engine.resolvers.remote import _no_worker_detail
        detail = (_no_worker_detail(model_id) or "").strip()
    except Exception:  # noqa: BLE001 — a diagnostic must never crash the runner
        detail = ""
    _why = f" Reason: {detail}" if detail else ""

    # 2) No live worker. Per-box "never serve locally" policy: a central box runs
    #    no in-process generation at all, even in a standalone/no-provider
    #    posture. Refuse before a multi-GB diffusion model can land on this CPU.
    #    HUGPY_VIDEOGEN_LOCAL=always still overrides. See managers.serve.policy.
    try:
        from hugpy_engine.serve.policy import no_local_serving
        _policy_off = not no_local_serving()
    except Exception:
        _policy_off = True
    if not _policy_off and not _videogen_local:
        return JobResult(job_id, ok=False, error=JobError(
            code="local_serving_disabled",
            message=(
                f"local model serving is disabled on this box "
                f"(HUGPY_NO_LOCAL_SERVING); refusing in-process generation of "
                f"{model_id!r}. Bring a GPU worker online, or set "
                f"HUGPY_VIDEOGEN_LOCAL=always to permit local generation here."
                f"{_why}"
            ),
            retryable=True,
        ))

    # 3) A fleet exists (provider registered) but no live worker serves this
    #    model — refuse rather than melt central with a CPU load. A bare
    #    single-box deploy (no provider) keeps local generation — the product.
    if provider is not None and not _videogen_local:
        return JobResult(job_id, ok=False, error=JobError(
            code="no_live_gpu_worker",
            message=(
                f"no live GPU worker is serving {model_id!r} (worker "
                "offline or still warming); refusing local CPU generation "
                "on central. Retry shortly, or set HUGPY_VIDEOGEN_LOCAL="
                f"always to permit in-process generation.{_why}"
            ),
            retryable=True,
        ))
    return None
