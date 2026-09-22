"""review/download.py — fetch one quant THROUGH ``hugpy_storage``.

The review pipeline downloads the best quant of a survivor before it loads
it. Curation owns no transfer code: bytes in transit are ``hugpy_storage``'s
(``PARTITION.md``), and the review asks for them the same way the console's
add-models routes and the ops provisioner do —

* ``hugpy_storage.downloader.queue.enqueue_download`` — a queued job of kind
  ``download`` that the ``hugpy-downloader`` daemon claims and runs with its
  stall-killer / resume / backoff guard; progress, cancel and retry all show
  in the same ``/jobs`` view as a human-clicked download. The review polls
  ``get_download`` until the job is terminal.
* ``hugpy_storage.download_models.download_one`` — the daemon's own
  synchronous transfer (atomic staging, orphan adoption, provenance stamp),
  run in THIS process when no daemon is there to claim a job. The nightly
  review runs on a worker box (ae) where central's downloader does not; a job
  enqueued there would wait forever, and "queued, waiting for downloader" is
  not a review.

Both paths are storage's public API; there is no third one here. The queue is
injectable (:class:`DownloadQueue`) so the pipeline is testable without a
daemon, a mirror or the network, and ``inline`` is injectable for the same
reason.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "DownloadQueue",
    "StorageQueue",
    "TERMINAL_STATUSES",
    "download_quant",
    "locate",
    "review_model_spec",
]

#: Legacy ``/jobs`` wire statuses (what ``get_download`` returns) that end a wait.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "expired", "discarded"})

#: A run downloads tens of GB unattended; the systemd unit allows 4h in total.
DEFAULT_TIMEOUT_SECONDS = 4 * 3600
DEFAULT_POLL_SECONDS = 2.0


@runtime_checkable
class DownloadQueue(Protocol):
    """What the review needs from a download queue (storage's, or a fake)."""

    def available(self) -> bool:
        """True when a daemon is there to execute an enqueued job."""
        ...

    def enqueue(self, model_key: str, model: dict, total_bytes: Optional[int] = None) -> str:
        """Queue one download; the job id."""
        ...

    def get(self, job_id: str) -> Optional[Mapping[str, Any]]:
        """The job row in the legacy wire shape (``status``, ``error``,
        ``message``), or ``None`` when unknown."""
        ...


class StorageQueue:
    """:class:`DownloadQueue` over ``hugpy_storage.downloader.queue``.

    Imports lazily: the queue module opens the shared job mirror on import,
    which a screen-only run and every test must never do."""

    transport = "review"

    def available(self) -> bool:
        try:
            from hugpy_storage.downloader.presence import downloader_alive
            if not downloader_alive():
                return False                        # no daemon: never open the mirror
            from hugpy_storage.downloader.queue import queue_healthy
            return bool(queue_healthy())
        except Exception as exc:  # noqa: BLE001 — an unreadable queue is "no daemon"
            logger.debug("review download: queue unavailable (%s)", exc)
            return False

    def enqueue(self, model_key: str, model: dict, total_bytes: Optional[int] = None) -> str:
        from hugpy_storage.downloader.queue import enqueue_download
        job = enqueue_download(model_key, model, total_bytes=total_bytes,
                               transport=self.transport)
        return str(getattr(job, "id", job))

    def get(self, job_id: str) -> Optional[Mapping[str, Any]]:
        from hugpy_storage.downloader.queue import get_download
        return get_download(job_id)


def review_model_spec(hub_id: str, quant: str, files: list[str]) -> tuple[str, dict]:
    """The ``(model_key, model)`` pair storage's transfer functions take for
    one GGUF quant of ``hub_id``: a single file by name, a sharded quant by
    include pattern."""
    model: dict[str, Any] = {
        "hub_id": hub_id,
        "framework": "gguf",
        "primary_task": "text-generation",
    }
    if len(files) == 1:
        model["filename"] = files[0]
    else:
        model["include"] = [f"*{quant}*.gguf"]      # sharded quant
    key = f"{hub_id.split('/')[-1]}-{quant}"
    return key, model


def locate(model: dict) -> str:
    """Where storage put (or would put) ``model``; ``""`` when it cannot say."""
    from hugpy_storage.model_paths import resolve_model_dir, route_destination
    return resolve_model_dir(model) or route_destination(model) or ""


def _inline_download(model: dict, model_key: str) -> None:
    from hugpy_storage.download_models import download_one
    download_one(model, model_key=model_key)


def _failure_detail(row: Optional[Mapping[str, Any]]) -> str:
    if not row:
        return "job vanished from the queue"
    error = row.get("error")
    if isinstance(error, Mapping):
        error = error.get("message") or error.get("code")
    return str(error or row.get("message") or row.get("status") or "unknown")


def download_quant(hub_id: str, quant: str, files: list[str], *,
                   queue: Optional[DownloadQueue] = None,
                   inline: Optional[Callable[[dict, str], None]] = None,
                   timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                   poll_seconds: float = DEFAULT_POLL_SECONDS,
                   sleep: Callable[[float], None] = time.sleep,
                   log: Callable[[str], None] = logger.info) -> str:
    """Fetch one quant through storage and return the directory it landed in.

    With a live downloader daemon the job is enqueued and polled to a terminal
    state; without one storage's synchronous ``download_one`` runs here.
    Raises ``RuntimeError`` on a failed/cancelled job and ``TimeoutError``
    when the wait exceeds ``timeout_seconds``, so the caller records a
    'download failed' row exactly as it always did.
    """
    key, model = review_model_spec(hub_id, quant, files)
    q: DownloadQueue = queue if queue is not None else StorageQueue()

    if q.available():
        job_id = q.enqueue(key, model)
        log(f"download {key}: queued as job {job_id} on the storage daemon")
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        row = q.get(job_id)
        status = str((row or {}).get("status") or "")
        while status not in TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"download {key}: job {job_id} still {status or 'unknown'} "
                                   f"after {timeout_seconds:.0f}s")
            sleep(poll_seconds)
            row = q.get(job_id)
            status = str((row or {}).get("status") or "")
            if row is None:
                break
        if status != "completed":
            raise RuntimeError(f"download {status or 'lost'}: {_failure_detail(row)}")
    else:
        log(f"download {key}: no downloader daemon — transferring in-process via storage")
        (inline or _inline_download)(model, key)

    return locate(model)
