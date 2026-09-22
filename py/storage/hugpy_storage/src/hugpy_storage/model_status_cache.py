"""Per-model install-status memo — the /models + /v1/models availability fix.

WHY
---
``model_status(model)`` (flask_app/.../downloads/downloader.py) answers one
question — *is this model installed, partial, or absent, and where?* — by
walking the store: ``route_destination`` → ``resolve_model_dir`` →
``candidate_model_dirs`` globs FOUR runtime families' legacy task dirs, stats
every candidate, then ``model_looks_downloaded`` globs the winner for weights.
That is on the order of a hundred filesystem calls PER MODEL.

Every listing route loops the whole manifest and calls it once per model
(``/v1/models``, ``/models``). With ~107 models that is ~10^4 filesystem calls
per request, and central's store is a spinning array reached over **virtiofs**,
so each one is a FUSE round-trip to the host. Measured on the live box
(2026-07-27): ``/health`` 0.0009s, ``/llm/workers`` 2.0s, ``/v1/models`` 18.5s →
25.7s → 55.3s and DEGRADING, with established connections to :7002 climbing
47 → 140 against only 24 gunicorn slots (``--workers 3 --threads 8``). Thread
wchans across all three workers read ``request_wait_answer`` (FUSE),
``folio_wait_bit_common`` (disk) and ``locks_lock_inode_wait`` — the API was not
computing, it was queued on I/O. In isolation the walk costs ~0.5s; under
concurrency callers serialise on the same inodes and it collapses.

The manifest itself is already cached (``models_config.get_models_dict`` serves
``MODEL_REGISTRY_DICT``, advanced in place by ``refresh_registry``) — the
per-model *status stat* was the uncached half. And installation status does not
change between requests: it changes when a model is downloaded, deleted,
pruned, reconciled or re-discovered. Re-walking on every call was pure waste.

WHAT THIS IS
------------
A memo in front of the live stat, with three properties:

  * **TTL** (``HUGPY_MODEL_STATUS_TTL_S``, default 15s) — bounds staleness for
    changes nobody told us about (an operator ``mv`` on the store, files landing
    from another box). Set to 0 to disable the cache entirely and always stat.
  * **Event invalidation** — :func:`invalidate_model_status` is called from the
    events that actually change a status (download reached a terminal state,
    delete, prune, reconcile apply, discovery sweep, and every
    ``refresh_registry``). The TTL alone would be a band-aid; the events are the
    correct signal.
  * **Single-flight** — concurrent callers asking for the SAME model share one
    live walk instead of stampeding it. This is what makes a short TTL cheap:
    N threads across a burst cost one walk, not N.

CROSS-PROCESS CONSISTENCY
-------------------------
Central runs ``gunicorn --workers 3 --threads 8``: three separate PROCESSES.
The memo is a per-process dict (no shared memory, no lock on the store — we are
trying to *stop* touching virtiofs, not add a lock file on it). Threads within a
process are safe via a module lock; processes converge through a tiny **epoch
file** on LOCAL disk (``$TMPDIR/hugpy-model-status.epoch``, override with
``HUGPY_MODEL_STATUS_EPOCH_PATH``):

  * an invalidation writes a fresh random token there (atomic ``os.replace``,
    no locking — any change is the whole signal);
  * every process re-reads that token at most once per
    ``HUGPY_MODEL_STATUS_EPOCH_POLL_S`` (default 1s) and drops its entire memo
    when the token differs from the one it last saw.

So a delete in worker process #1 is visible in #2 and #3 within ~1s, not within
the TTL. That is deliberate: a wrongly-cached ``not_installed`` would HIDE a
real model, and the operator's standing rule is that correctness of the cached
value outranks the cache. One read of a ~32-byte local file per second per
process is nothing next to the ~10^4 virtiofs calls it replaces. If the epoch
file cannot be read or written we degrade to TTL-only convergence — never to a
crash and never to a made-up status.

DEGRADE-NOT-GUESS
-----------------
Any failure in the cache machinery falls back to the live stat. The cache never
invents a status, never caches a non-dict, and never swallows an error raised by
the live stat itself (that propagates exactly as it does today).

Stdlib-only and importing nothing from the rest of the package (the ``comms``
contract) — the live stat is injected by the caller, so this module has no
opinion about what a "status" is.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── tunables (env-overridable; read live so an operator can retune without a
#    code change — the values are plain os.environ reads, never env_value(),
#    which carries .env inline-comment pollution) ────────────────────────────
_DEFAULT_TTL_S = 15.0          # staleness bound for changes we get no event for
_DEFAULT_EPOCH_POLL_S = 1.0    # how often a process re-reads the epoch token
_DEFAULT_LOCK_WAIT_S = 20.0    # single-flight wait before degrading to our own walk
_MAX_KEY_LOCKS = 4096          # bound the per-key lock table (fleet has ~10^2 models)

# The fields the live stat's routing actually reads (candidate_model_dirs +
# _status_cfg). They ARE the cache key: two entries that route identically
# resolve identically, and any edit to a routing field yields a different key,
# so a re-keyed model can never be answered from the old entry.
_KEY_FIELDS = ("model_key", "hub_id", "name", "folder", "dir",
               "framework", "filename", "include",
               "primary_task", "task", "tasks")

_LOCK = threading.Lock()                       # guards every module global below
_CACHE: Dict[str, Tuple[float, dict]] = {}     # key -> (expires_monotonic, value)
_KEY_LOCKS: Dict[str, threading.Lock] = {}     # key -> single-flight lock
_EPOCH_SEEN: Optional[str] = None
_EPOCH_CHECKED_AT = 0.0
_STATS = {"hits": 0, "misses": 0, "live_calls": 0,
          "invalidations": 0, "epoch_clears": 0, "errors": 0}


# ──────────────────────────────────────────────────────────────────────────
# tunables
# ──────────────────────────────────────────────────────────────────────────
def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.debug("ignoring non-numeric %s=%r", name, raw)
        return default


def ttl_seconds() -> float:
    """Entry lifetime. ``<= 0`` disables the cache (every call stats live)."""
    return _env_float("HUGPY_MODEL_STATUS_TTL_S", _DEFAULT_TTL_S)


def epoch_poll_seconds() -> float:
    """Minimum gap between epoch-file reads. ``<= 0`` reads on every lookup."""
    return _env_float("HUGPY_MODEL_STATUS_EPOCH_POLL_S", _DEFAULT_EPOCH_POLL_S)


def lock_wait_seconds() -> float:
    return _env_float("HUGPY_MODEL_STATUS_LOCK_WAIT_S", _DEFAULT_LOCK_WAIT_S)


def epoch_path() -> str:
    """The cross-process invalidation token file — LOCAL disk on purpose.

    It must NOT live on the shared model store: the whole point of this module
    is to stop making virtiofs round-trips on the request path. gunicorn's
    workers are all on the same box, so the system temp dir is shared between
    them and local (the download error hand-off already uses it)."""
    explicit = (os.environ.get("HUGPY_MODEL_STATUS_EPOCH_PATH") or "").strip()
    return explicit or os.path.join(tempfile.gettempdir(),
                                    "hugpy-model-status.epoch")


# ──────────────────────────────────────────────────────────────────────────
# key
# ──────────────────────────────────────────────────────────────────────────
def status_key(model: Any, scope: str = "") -> str:
    """Stable cache key for ``model`` — a digest of its routing identity.

    ``scope`` folds in anything outside the model dict that changes the answer
    (a store root, in tests). Accepts a dict or any object exposing the fields.
    """
    get = model.get if isinstance(model, dict) else (
        lambda f, _m=model: getattr(_m, f, None))
    parts = []
    for field in _KEY_FIELDS:
        value = get(field)
        if isinstance(value, tuple):
            value = list(value)
        parts.append([field, value])
    try:
        raw = json.dumps(parts, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 — a weird value must not break the key
        raw = repr(parts)
    digest = hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()
    return f"{scope}:{digest}" if scope else digest


# ──────────────────────────────────────────────────────────────────────────
# epoch (cross-process invalidation)
# ──────────────────────────────────────────────────────────────────────────
def _read_epoch() -> Optional[str]:
    """Current token, ``""`` when no event has ever been published, ``None``
    when the file exists but cannot be read (unknown -> keep what we had)."""
    try:
        with open(epoch_path(), "r", encoding="utf-8") as fh:
            return fh.read(256).strip()
    except FileNotFoundError:
        return ""
    except OSError:
        logger.debug("model-status epoch unreadable at %s", epoch_path(),
                     exc_info=True)
        return None


def _write_epoch(token: str) -> None:
    path = epoch_path()
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(token)
        os.replace(tmp, path)
    except OSError:
        # TTL-only convergence for the other processes. Never fatal.
        logger.debug("could not publish the model-status epoch to %s", path,
                     exc_info=True)
        try:
            os.remove(tmp)
        except OSError:
            pass


def _sync_epoch(force: bool = False) -> None:
    """Adopt the published token; drop the whole memo when it changed."""
    global _EPOCH_SEEN, _EPOCH_CHECKED_AT
    now = time.monotonic()
    poll = epoch_poll_seconds()
    with _LOCK:
        if not force and poll > 0 and (now - _EPOCH_CHECKED_AT) < poll:
            return
        _EPOCH_CHECKED_AT = now
    token = _read_epoch()
    if token is None:
        return                                   # unknown — keep last-seen
    with _LOCK:
        if _EPOCH_SEEN is None:
            _EPOCH_SEEN = token                  # first read; memo is empty
            return
        if token != _EPOCH_SEEN:
            _EPOCH_SEEN = token
            _CACHE.clear()
            _STATS["epoch_clears"] += 1


# ──────────────────────────────────────────────────────────────────────────
# memo
# ──────────────────────────────────────────────────────────────────────────
def _get(key: str) -> Optional[dict]:
    now = time.monotonic()
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        expires, value = entry
        if expires <= now:
            _CACHE.pop(key, None)
            return None
        return dict(value)


def _put(key: str, value: dict, ttl: float) -> None:
    with _LOCK:
        _CACHE[key] = (time.monotonic() + ttl, value)


def _key_lock(key: str) -> threading.Lock:
    with _LOCK:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            if len(_KEY_LOCKS) >= _MAX_KEY_LOCKS:
                # Bounded table. Dropping locks can at worst let two threads
                # walk the same model once — never a wrong answer.
                _KEY_LOCKS.clear()
            lock = _KEY_LOCKS[key] = threading.Lock()
        return lock


def _bump(stat: str) -> None:
    with _LOCK:
        _STATS[stat] = _STATS.get(stat, 0) + 1


# ──────────────────────────────────────────────────────────────────────────
# public surface
# ──────────────────────────────────────────────────────────────────────────
def cached_model_status(model: Any,
                        live: Callable[[Any], dict],
                        scope: str = "") -> dict:
    """``live(model)``, memoized per model with a TTL and single-flight.

    ``live`` is the real stat (the caller owns what a status *is*). A returned
    dict is always a COPY, so a caller mutating it — every listing route does,
    via ``model.update(...)`` — can never poison the entry.

    Degrade-not-guess: any failure inside the cache machinery falls through to
    ``live(model)``; an exception raised by ``live`` itself propagates
    unchanged.
    """
    ttl = ttl_seconds()
    if ttl <= 0:
        _bump("live_calls")
        return live(model)

    try:
        key = status_key(model, scope)
        _sync_epoch()
        hit = _get(key)
    except Exception:  # noqa: BLE001
        logger.debug("model-status cache lookup failed; using the live stat",
                     exc_info=True)
        _bump("errors")
        _bump("live_calls")
        return live(model)

    if hit is not None:
        _bump("hits")
        return hit
    _bump("misses")

    try:
        lock = _key_lock(key)
    except Exception:  # noqa: BLE001
        logger.debug("model-status single-flight unavailable", exc_info=True)
        _bump("errors")
        _bump("live_calls")
        return live(model)

    # Single-flight: the first caller walks, the rest wait on the same lock and
    # read the result. Bounded — we never block a request forever on a lock; on
    # timeout we simply pay for our own walk (today's behaviour).
    acquired = lock.acquire(timeout=max(lock_wait_seconds(), 0.0))
    try:
        if acquired:
            try:
                hit = _get(key)
            except Exception:  # noqa: BLE001
                hit = None
                _bump("errors")
            if hit is not None:
                _bump("hits")
                return hit
        _bump("live_calls")
        value = live(model)
        if not isinstance(value, dict):
            return value                          # never memo a shape we can't copy
        try:
            _put(key, dict(value), ttl)
        except Exception:  # noqa: BLE001
            logger.debug("model-status cache store failed", exc_info=True)
            _bump("errors")
        return dict(value)
    finally:
        if acquired:
            lock.release()


def refresh_model_status(model: Any,
                         live: Callable[[Any], dict],
                         scope: str = "") -> dict:
    """Explicit refresh: always stat live, then seed the memo with the answer.

    The single-model detail route uses this — "tell me the truth about THIS
    model" stays a real read, and it repairs the entry the listings share."""
    value = live(model)
    try:
        ttl = ttl_seconds()
        if ttl > 0 and isinstance(value, dict):
            _sync_epoch()
            _put(status_key(model, scope), dict(value), ttl)
    except Exception:  # noqa: BLE001
        logger.debug("model-status refresh could not seed the cache",
                     exc_info=True)
        _bump("errors")
    return dict(value) if isinstance(value, dict) else value


def invalidate_model_status(reason: str = "") -> None:
    """Drop every memoized status, here and in the other gunicorn workers.

    Deliberately COARSE. The events that change a status are rare (a download
    finishing, a delete, a prune, a reconcile, a discovery sweep) and the cost
    of over-invalidating is one re-walk, whereas a mis-keyed targeted
    invalidation would leave a wrong value serving. Across processes only a
    whole-memo drop is expressible through the epoch token anyway.
    """
    global _EPOCH_SEEN, _EPOCH_CHECKED_AT
    token = uuid.uuid4().hex
    with _LOCK:
        _CACHE.clear()
        _EPOCH_SEEN = token
        _EPOCH_CHECKED_AT = time.monotonic()
        _STATS["invalidations"] += 1
    _write_epoch(token)
    logger.info("model-status cache invalidated (%s)", reason or "unspecified")


def cache_stats() -> dict:
    """Counters + config, for diagnostics and tests."""
    with _LOCK:
        stats = dict(_STATS)
        stats["entries"] = len(_CACHE)
    stats["ttl_s"] = ttl_seconds()
    stats["epoch_poll_s"] = epoch_poll_seconds()
    stats["epoch_path"] = epoch_path()
    return stats


def reset_model_status_cache(forget_epoch: bool = True) -> None:
    """Wipe this process's memo and counters WITHOUT publishing an event.

    Test/diagnostic hook — use :func:`invalidate_model_status` for a real
    change, which is what the other processes need to hear about."""
    global _EPOCH_SEEN, _EPOCH_CHECKED_AT
    with _LOCK:
        _CACHE.clear()
        _KEY_LOCKS.clear()
        for stat in _STATS:
            _STATS[stat] = 0
        if forget_epoch:
            _EPOCH_SEEN = None
            _EPOCH_CHECKED_AT = 0.0
