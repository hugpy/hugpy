"""Remedy whitelist: implemented, tested, DISABLED by default.

Every remedy the sentinel could ever apply is a typed entry here — nothing
outside this table is executable, execution is gated on
settings.remedies_enabled (default OFF; the keeper flips it deliberately),
and the prod worker "ae" is excluded STRUCTURALLY: eligibility filters it
out and execute() re-raises even if a caller hand-crafts params. ae cases
are document+escalate only, always.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from hugpy_ops.sentinel.cases import Anomaly

# Workers remedies must NEVER touch. Membership is checked in BOTH
# eligibility and execution so no single call path can bypass it.
PROD_EXCLUDED_WORKERS = frozenset({"ae"})


class RemediesDisabled(RuntimeError):
    """Raised by execute() while the remedies_enabled setting is OFF."""


class DownloadsDisabled(RuntimeError):
    """Raised by execute() for gate="downloads" remedies while the
    downloads_enabled setting (default ON, k97) is explicitly OFF."""


class ProdWorkerExcluded(RuntimeError):
    """Raised by execute() when params target an excluded (prod) worker."""


def _target_worker(anomaly_or_params) -> str | None:
    if isinstance(anomaly_or_params, Anomaly):
        return anomaly_or_params.evidence.get("worker")
    return anomaly_or_params.get("worker")


@dataclass(frozen=True)
class Remedy:
    name: str
    method: str
    url_template: str                       # str.format over execute() params
    applies_to: Callable[[Anomaly], bool]
    reversible: bool = True                 # whitelist admits ONLY reversible
    required_params: tuple = field(default=())
    # Which settings gate governs execution: "remedies" (default OFF — the
    # keeper flips HUGPY_SENTINEL_REMEDIES deliberately) or "downloads"
    # (default ON per the k97 operator ruling; HUGPY_SENTINEL_DOWNLOADS=0
    # turns it off). Separate gates because enqueueing a download mutates
    # nothing a worker serves — it queues a reversible job on central.
    gate: str = "remedies"


def _has(anomaly: Anomaly, *keys: str) -> bool:
    return all(anomaly.evidence.get(k) for k in keys)


WHITELIST: tuple[Remedy, ...] = (
    # Evict a wedged model seat (the 2026-08-06 vision-wedge remedy): the
    # next request reseats it cleanly. Reversible — the model reloads.
    Remedy(name="worker_model_unload", method="POST",
           url_template="{worker_base}/models/unload",
           applies_to=lambda a: a.kind in ("scorecard_hard_fail",
                                           "capability_lost")
           and _has(a, "worker", "model_key"),
           required_params=("worker", "worker_base", "model_key")),
    # Unload / relaunch one slot when a specific seat is wedged.
    Remedy(name="worker_slot_unload", method="POST",
           url_template="{worker_base}/slots/{slot_id}/unload",
           applies_to=lambda a: a.kind in ("job_stalled",
                                           "scorecard_hard_fail")
           and _has(a, "worker", "slot_id"),
           required_params=("worker", "worker_base", "slot_id")),
    Remedy(name="worker_slot_relaunch", method="POST",
           url_template="{worker_base}/slots/{slot_id}/relaunch",
           applies_to=lambda a: a.kind in ("job_stalled",
                                           "scorecard_hard_fail")
           and _has(a, "worker", "slot_id"),
           required_params=("worker", "worker_base", "slot_id")),
    # Cancel one stuck request on central. Reversible — the caller retries.
    Remedy(name="central_chat_cancel", method="POST",
           url_template="{central}/llm/chat/cancel/{request_id}",
           applies_to=lambda a: a.kind in ("job_stalled", "job_expired")
           and _has(a, "request_id"),
           required_params=("central", "request_id")),
    # k97: enqueue a missing declared weight on central's transfer plane —
    # the SAME queue the console's add-models path uses (POST
    # /llm/repos/download -> downloader.queue.enqueue_download -> the
    # hugpy-downloader-dev daemon), so dedupe/progress/cancel/retry all
    # apply. Reversible: a queued download can be cancelled and its files
    # deleted. Applies only when the anomaly PROVES a source (hub_id) —
    # provisioner.wants() never guesses one. Own gate ("downloads", default
    # ON per the 2026-08-06 operator ruling). The ae prod-worker exclusion
    # is irrelevant here by construction — downloads land on CENTRAL's
    # shared storage, never on a worker, and weight_missing evidence carries
    # no worker; the structural check in execute() still runs and is simply
    # never tripped.
    Remedy(name="enqueue_download", method="POST",
           url_template="{central}/llm/repos/download",
           applies_to=lambda a: a.kind == "weight_missing"
           and _has(a, "hub_id"),
           gate="downloads",
           required_params=("central", "hub_id")),
)


def eligible(anomaly: Anomaly) -> list[Remedy]:
    """Whitelist entries applicable to this anomaly; [] for prod workers."""
    if _target_worker(anomaly) in PROD_EXCLUDED_WORKERS:
        return []
    return [r for r in WHITELIST if r.applies_to(anomaly)]


def execute(remedy: Remedy, params: dict, settings,
            http_post: Callable | None = None) -> dict:
    """Apply one whitelisted remedy. Gated; raises rather than half-runs.

    `http_post(url, body) -> dict` is injectable for tests; the default
    posts JSON with a short timeout.
    """
    if remedy not in WHITELIST:
        raise ValueError("remedy %r is not whitelisted" % (remedy.name,))
    if remedy.gate == "downloads":
        if not getattr(settings, "downloads_enabled", True):
            raise DownloadsDisabled(
                "downloads are disabled (HUGPY_SENTINEL_DOWNLOADS=0); "
                "case stays document-only")
    elif not getattr(settings, "remedies_enabled", False):
        raise RemediesDisabled(
            "remedies are disabled (HUGPY_SENTINEL_REMEDIES is not '1'); "
            "case stays document+escalate")
    if _target_worker(params) in PROD_EXCLUDED_WORKERS:
        raise ProdWorkerExcluded(
            "remedy %s targets prod worker %r — never remediated by the "
            "sentinel" % (remedy.name, params.get("worker")))
    missing = [k for k in remedy.required_params if not params.get(k)]
    if missing:
        raise ValueError("remedy %s missing params: %s"
                         % (remedy.name, ", ".join(missing)))
    url = remedy.url_template.format(**params)
    body = {k: v for k, v in params.items()
            if k not in ("worker", "worker_base", "central")}
    if http_post is None:
        http_post = _default_post
    return http_post(url, body)


def _default_post(url: str, body: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")
