"""Route-facing shim over the downloader engine.

THE TRANSFER LIFECYCLE MOVED (2026-07-28) to ``hugpy_storage.downloader``
so it can run in its OWN process (``hugpy-downloader-dev``) instead of inside
gunicorn — see that package's docstring for why (operator: a download "pushes
off all of the workers"). Nothing in the console API starts a download any more.

What stays here:

  * ``update_model_status`` / ``update_model_sizes`` — the manifest-row stampers.
    They are a LISTING concern, not a download concern, and they read the
    flask-side model_physical module.
  * re-exports of the engine + queue names (``start_cancellable_download``,
    ``cancel_download``, ``retry_download``, ``invalidate_model_status_cache``,
    ``_progress_bytes``, …) so every existing caller and test keeps working
    against the import path it already uses.

Anything that needs to CHANGE download behaviour belongs in
``downloader/engine.py`` (how a transfer runs) or ``downloader/queue.py`` (how
one is requested), never here.
"""
import multiprocessing as mp
from datetime import datetime, timezone
# The persisted physical-state read path (comms/model_physical.py is the store).
from hugpy_storage.console.model_physical import (
    ASPECT_SIZE,
    ASPECT_STATUS,
    size_fields,
    stamp_fields,
    status_fields,
)

# ── the engine (flask-free, runs in the downloader daemon) ────────────────
from hugpy_storage.downloader import engine as engine
from hugpy_storage.downloader.engine import (
    MAX_ATTEMPTS,
    STALL_SECONDS,
    invalidate_model_status_cache,
    start_cancellable_download,
    run_download_job,
    _estimate_total_bytes,
    _estimate_total_bytes_bounded,
    _dir_bytes,
    _progress_bytes,
    _watch,
    _download_worker,
    _read_error,
    _write_error,
    _clear_error,
    _error_path,
    _is_cancelled,
)
# ── the queue (what a ROUTE is allowed to do) ─────────────────────────────
from hugpy_storage.downloader.queue import (
    cancel_download,
    retry_download,
    enqueue_download,
    list_downloads,
    get_download,
    queue_depth,
    queue_healthy,
)


def update_model_status(model: dict) -> dict:
    """Stamp status/destination onto a manifest row — THE listing hot path.

    Reads the PERSISTED physical state (comms/model_physical.py) instead of
    re-walking the store for every model on every request. Central downloaded
    these models; it already knows whether they are installed and where. Same
    keys, same values, same in-place mutation of the caller's dict — only the
    number of filesystem calls changes (a warm row: zero).

    A model with no persisted record is derived LIVE and written through, so a
    fresh install and a newly-added model are correct rather than zeroed.
    """
    return stamp_fields(model, status_fields(model), ASPECT_STATUS)


def update_model_sizes(model: dict, mk: str = None) -> dict:
    """Stamp the SIZE half (effective quant / variants / dir + effective bytes)
    onto a row — the extra half ``/models`` shows and ``/v1/models`` does not.

    Same contract as :func:`update_model_status`: persisted lookup, live derive
    + write-through on a miss."""
    mk = mk or model.get("model_key")
    return stamp_fields(model, size_fields(model, mk), ASPECT_SIZE)
