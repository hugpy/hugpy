"""Persisted PHYSICAL model state — central serves what it already knows.

WHY
---
The registry row (``models_config.get_models_dict``) persists a model's
IDENTITY: model_key, hub_id, folder, filename, framework, tasks, ports. It does
NOT persist its PHYSICAL state — whether the weights are on disk, where, and how
big they are. Every listing therefore RE-DERIVED those facts from the store on
every request:

    /v1/models   -> update_model_status()  -> model_status()      (~10^2 stats)
    /models      -> the same, PLUS gguf_variants_detail + a recursive
                    walk_listing/dir_size_bytes                   (~10^3 stats)
    /llm/workers -> _public_view -> _model_size_bytes /
                    _model_moe_detail / _model_marker_flag, PER DESIGNATED
                    MODEL (ae alone has 75, ~111 fleet-wide) and from FOUR
                    callers each — allocated_totals, planned_split,
                    derived_default_mode, derived_default_allocation

With ~107 models that is ~10^4-10^5 filesystem calls per request, and central's
store is a 16TB spinning array reached over **virtiofs** — every one of them a
FUSE round-trip to the host. Measured 2026-07-27: ``/models`` 40.4s,
``/v1/models`` 3.9s, ``list_workers()`` 110-230s (``GET /llm/workers`` 31s under
the console's polling, so the workers view never rendered at all); under
concurrency threads pile on ``request_wait_answer`` /
``folio_wait_bit_common`` / ``locks_lock_inode_wait`` and the site stops
answering. ``_public_view`` is also the HEARTBEAT REPLY, so the same walk was
starving the beats that decide whether a worker reads online.

But **central downloaded these models**. It knows when that finished and how big
they are. Re-discovering its own facts on every GET is pure waste. So: derive
once, at the moments the facts CHANGE, persist the answer beside the registry,
and serve reads as a plain dict lookup.

This module is the persistence half. The derivation half (which knows what a
"status" or an "effective quant" IS) lives with the code that already owned it —
``flask_app/app/functions/downloads/model_physical.py`` — and injects its result
here. Same contract as ``model_status_cache``: stdlib-only, no opinion about the
values it stores.

WHAT A RECORD IS
----------------
``{model_key: {"fields": {...}, <provenance>}}`` in one JSON file:

    fields      the physical facts, EXACTLY the keys the deriver produced (see
                ASPECT_STATUS / ASPECT_SIZE). Absent key != zero: a key the
                deriver did not produce is not stamped, ever.
    identity    digest of the row's ROUTING identity (``status_key``) at derive
                time. A re-keyed / re-routed model therefore reads as ABSENT
                (-> derive live), never as its predecessor's answer.
    aspects     which halves of the record are present ("status", "size").
                ``/v1/models`` needs only "status"; ``/models`` needs both. A
                status-only record still serves ``/v1/models`` for free instead
                of forcing the (much heavier) size walk it never reads.
    derived_at  unix seconds — the provenance stamp. Lets a reader tell fresh
                from stale and a repair pass find rows to redo.
    dir_mtime   mtime of the model dir as observed at derive time (diagnostic /
                repair input; deliberately NOT consulted on the hot path — see
                STALENESS below).
    source      which event wrote it ("listing", "detail", "discover", …).

ABSENT MEANS DERIVE
-------------------
:func:`lookup` returns ``(None, reason)`` for absent / identity-changed /
missing-aspect. There is no "default" record and no zero-filled row: the caller
MUST derive live and write through. A wrongly-persisted ``not_installed`` would
HIDE a real model, which is the failure this design exists to avoid.

STALENESS
---------
The store is SHARED and MUTABLE — another box writes to it, an operator can
``mv`` a directory, the reaper deletes — so a persisted value can go stale with
no event firing. Three answers, in order of importance:

  1. **Events.** Every change hugpy itself makes drops the affected row(s):
     ``refresh_registry``, download terminal states, DELETE, prune, reconcile
     apply, a ``gguf_file`` override change. Targeted where the key is known
     (the persisted store CAN express "forget one row" across processes, which
     the in-process memo could not).
  2. **Explicit repair.** ``/models/discover`` re-derives and rewrites the whole
     table in its background thread; ``GET /models/<key>`` is a live read that
     rewrites that one row (opening a row IS the force-refresh).
  3. **A bounded max age**, ``HUGPY_MODEL_PHYSICAL_MAX_AGE_S`` (default 6h, 0 =
     never expire). Expiry is JITTERED per model_key over a ±25% window so ~107
     rows written by one sweep do not all come due on the same request — an
     expired row is re-derived by whichever listing sees it next, a trickle
     rather than a cliff.

There is deliberately **no per-request freshness stat** (no dir-mtime compare on
the hot path). It would cost one virtiofs round-trip PER MODEL — reintroducing
the exact O(models) I/O this removes — and it would not even work: a directory's
mtime does not change when a file inside a subdirectory changes or grows, so it
would buy false confidence at real cost. Events + repair + max-age cover it
honestly.

CROSS-PROCESS
-------------
Central runs ``gunicorn --workers 3``. The file IS the shared state: writes take
an ``fcntl`` exclusive lock, read-modify-write, then ``os.replace`` (the atomic
idiom ``comms.settings`` already proved here), so concurrent writers can never
interleave a half-written table. Readers keep an in-memory snapshot revalidated
by (mtime_ns, size) at most once per ``HUGPY_MODEL_PHYSICAL_POLL_S`` (default
1s), so a delete in worker #1 is visible in #2 and #3 within ~1s while a warm
listing costs ZERO filesystem calls. A process that WRITES updates its own
snapshot immediately, so it is never stale about its own change.

DEGRADE-NOT-GUESS
-----------------
Any failure here — unreadable file, unparseable JSON, unwritable directory,
no ``fcntl`` — degrades to "absent", i.e. the caller derives live exactly as it
does today. The store never invents a field and never blocks the operation that
changed the store.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover — POSIX-only in practice
    fcntl = None

from hugpy_storage.model_status_cache import status_key

logger = logging.getLogger(__name__)

# Record schema. Bump when the MEANING of a stored field changes; a record
# written by another version reads as absent (-> derive), never as garbage.
RECORD_VERSION = 1

# ── the two halves of a physical record ───────────────────────────────────
# STATUS: what ``model_status`` answers — is it here, where, is the marker
# there. This is all ``/v1/models`` reads (it filters on ``status``).
ASPECT_STATUS = "status"
STATUS_FIELDS: Tuple[str, ...] = (
    "status", "destination", "installed_marker", "filename_warning",
)
# SIZE: what ``_annotate_gguf_size`` + ``_annotate_size`` answer — the effective
# quant, its variants, the projector, the dir footprint. Only ``/models`` reads
# these, and they cost an order of magnitude more to derive, so they are their
# own aspect rather than a tax on every /v1/models miss.
ASPECT_SIZE = "size"
SIZE_FIELDS: Tuple[str, ...] = (
    "dir_bytes", "size_bytes", "effective_bytes", "effective_gguf",
    "gguf_variants", "mmproj_bytes", "moe",
    # 2026-09-23: where size_bytes came from (marker/manifest/dir_walk/
    # gguf_effective), a qualifier, and — when it is None — WHY.
    "size_source", "size_note", "size_reason",
)
# MARKER: the model's ``hugpy.json`` declared-identity blob, read off disk. Its
# capability bools (moe_capable, bnb_capable) are what the WORKERS view asks per
# designated model — ae alone has 75 — and resolving the dir + reading the file
# is another per-model store round-trip. Its own aspect because no LISTING wants
# it: /models and /v1/models never request it, so it can never leak a key into
# their responses.
ASPECT_MARKER = "marker"
MARKER_FIELDS: Tuple[str, ...] = ("hugpy_marker",)
ASPECT_FIELDS: Dict[str, Tuple[str, ...]] = {
    ASPECT_STATUS: STATUS_FIELDS,
    ASPECT_SIZE: SIZE_FIELDS,
    ASPECT_MARKER: MARKER_FIELDS,
}
PHYSICAL_FIELDS: Tuple[str, ...] = STATUS_FIELDS + SIZE_FIELDS + MARKER_FIELDS

# ── tunables (plain os.environ reads — never env_value(), which carries .env
#    inline-comment pollution) ───────────────────────────────────────────────
_DEFAULT_MAX_AGE_S = 6 * 3600.0   # bounded staleness for changes nobody told us about
_DEFAULT_POLL_S = 1.0             # min gap between (mtime,size) revalidations


from hugpy_storage.env import env_float

def _env_float(name: str, default: float) -> float:
    return env_float(name, default, logger=logger)


def max_age_seconds() -> float:
    """Base record lifetime. ``<= 0`` = never expire (events + repair only)."""
    return _env_float("HUGPY_MODEL_PHYSICAL_MAX_AGE_S", _DEFAULT_MAX_AGE_S)


def poll_seconds() -> float:
    """Min gap between revalidations of the on-disk table. ``<= 0`` = always."""
    return _env_float("HUGPY_MODEL_PHYSICAL_POLL_S", _DEFAULT_POLL_S)


def max_age_for(model_key: str) -> float:
    """Per-key lifetime: the base age spread over a ±25% window.

    A rebuild sweep stamps every row within the same second; without jitter the
    whole table would come due on one unlucky request (a 40s cliff). Keyed on
    the model_key so it is deterministic — the same row always expires at the
    same offset, and processes agree."""
    base = max_age_seconds()
    if base <= 0:
        return 0.0
    h = int(hashlib.sha1(str(model_key).encode("utf-8", "replace")
                         ).hexdigest()[:8], 16) / 0xFFFFFFFF
    return base * (0.75 + 0.5 * h)


def identity_of(model: Any) -> str:
    """Digest of a model's ROUTING identity.

    Shared with ``model_status_cache.status_key`` on purpose: the two stores
    answer different questions but must agree on what "the same model" is, or a
    re-keyed row could be served one store's stale answer."""
    return status_key(model, scope=f"phys{RECORD_VERSION}")


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


class PhysicalStore:
    """The persisted physical-state table. One JSON file, fcntl-locked writes."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._cache: Dict[str, dict] = {}
        self._stamp: Optional[Tuple[int, int]] = None   # (mtime_ns, size)
        self._checked_at = 0.0
        self._loaded = False
        self._stats = {"hits": 0, "misses": 0, "expired": 0, "writes": 0,
                       "forgets": 0, "reloads": 0, "errors": 0}

    # -- path ---------------------------------------------------------------
    def path(self) -> str:
        """Beside the registry's other persisted artifacts (the discovery report
        and its siblings under ``$PROJECTS_HOME``) — this IS registry state."""
        if self._path:
            return self._path
        env = (os.environ.get("HUGPY_MODEL_PHYSICAL_PATH") or "").strip()
        if env:
            return env
        # Consolidated under HUGPY_HOME (~/.hugpy/state); migrates a legacy
        # ~/model_physical.json in place on first resolve.
        from hugpy_platform import app_dirs as _hp
        return _hp.model_physical_json()

    # -- io -----------------------------------------------------------------
    def _read_disk(self) -> dict:
        try:
            with open(self.path(), "r", encoding="utf-8") as f:
                raw = f.read()
        except FileNotFoundError:
            return {}
        except OSError:
            logger.debug("model-physical table unreadable at %s", self.path(),
                         exc_info=True)
            self._stats["errors"] += 1
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Loud + empty: a corrupt table must degrade to "derive live", never
            # take the listings down and never serve half a record.
            logger.error("model_physical.json unparseable at %s — treating as "
                         "empty (every row will be re-derived)", self.path())
            self._stats["errors"] += 1
            return {}
        if not isinstance(data, dict):
            return {}
        if data.get("version") != RECORD_VERSION:
            return {}
        models = data.get("models")
        return models if isinstance(models, dict) else {}

    def _snapshot(self) -> Dict[str, dict]:
        """The current table, revalidated by (mtime, size) at most once per
        ``poll_seconds``. Returns the LIVE dict — callers must not mutate it."""
        with self._lock:
            now = time.monotonic()
            poll = poll_seconds()
            if self._loaded and poll > 0 and (now - self._checked_at) < poll:
                return self._cache
            self._checked_at = now
            try:
                st = os.stat(self.path())
                stamp = (st.st_mtime_ns, st.st_size)
            except FileNotFoundError:
                stamp = None
            except OSError:
                # Unknown — keep what we have rather than dropping to "derive
                # everything" over a transient virtiofs hiccup.
                logger.debug("model-physical stat failed", exc_info=True)
                self._stats["errors"] += 1
                return self._cache
            if self._loaded and stamp == self._stamp:
                return self._cache
            self._cache = self._read_disk()
            self._stamp = stamp
            self._loaded = True
            self._stats["reloads"] += 1
            return self._cache

    def _transaction(self, mutate):
        """fcntl-locked read-modify-write + atomic replace.

        Three gunicorn workers and the reconcile path all write here; the lock
        makes a concurrent write a serialised read-modify-write instead of a
        last-writer-wins clobber, and ``os.replace`` means a reader never sees a
        partial file. Returns ``mutate``'s result, or None if we could not
        write (degrade: the value is simply not persisted, never lost data).

        The lock lives on a SIDECAR file (``<path>.lock``), not on the table
        itself, and the table is re-read BY PATH inside the lock. That is not
        fussiness: ``os.replace`` swaps the inode, so a writer holding a lock on
        the OLD inode is holding a lock nobody else takes, and the next writer
        reads the file it opened before the swap — stale. Measured here: four
        concurrent processes writing 25 rows each landed 76 of 100 rows with the
        lock on the table, 100 of 100 with it on the sidecar."""
        path = self.path()
        lock_path = f"{path}.lock"
        with self._lock:
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(lock_path, "a+", encoding="utf-8") as fh:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                    try:
                        models = self._read_disk()   # by PATH, inside the lock
                        result = mutate(models)
                        payload = {"version": RECORD_VERSION, "models": models}
                        tmp = f"{path}.tmp.{os.getpid()}"
                        with open(tmp, "w", encoding="utf-8") as out:
                            json.dump(payload, out)
                        os.replace(tmp, path)
                        # This process is never stale about its own write.
                        self._cache = models
                        self._loaded = True
                        self._checked_at = time.monotonic()
                        try:
                            st = os.stat(path)
                            self._stamp = (st.st_mtime_ns, st.st_size)
                        except OSError:
                            self._stamp = None
                        return result
                    finally:
                        if fcntl is not None:
                            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                logger.warning("could not persist model physical state to %s "
                               "(listings will keep deriving live)", path,
                               exc_info=True)
                self._stats["errors"] += 1
                return None

    # -- read ---------------------------------------------------------------
    def lookup(self, model_key: str, identity: str,
               aspect: str = ASPECT_STATUS) -> Tuple[Optional[dict], str]:
        """``(fields, state)`` for ``model_key``.

        ``state`` is one of ``fresh`` / ``expired`` / ``absent`` /
        ``identity-changed`` / ``aspect-missing``. ``fields`` is a COPY (callers
        stamp it onto registry rows they then mutate) and is None for every
        state except ``fresh`` and ``expired``:

          * ``absent`` / ``identity-changed`` / ``aspect-missing`` -> the caller
            MUST derive live. There is no invented value here.
          * ``expired`` -> real, dated fields the caller may re-derive now or
            serve while it re-derives something else; ``derived_at`` says how
            old they are.
        """
        try:
            rec = self._snapshot().get(str(model_key))
        except Exception:  # noqa: BLE001 — a lookup must never break a listing
            logger.debug("model-physical lookup failed", exc_info=True)
            self._stats["errors"] += 1
            return None, "absent"
        if not isinstance(rec, dict):
            self._stats["misses"] += 1
            return None, "absent"
        if rec.get("identity") != identity:
            self._stats["misses"] += 1
            return None, "identity-changed"
        if aspect not in (rec.get("aspects") or []):
            self._stats["misses"] += 1
            return None, "aspect-missing"
        fields = rec.get("fields")
        if not isinstance(fields, dict):
            self._stats["misses"] += 1
            return None, "absent"
        wanted = ASPECT_FIELDS.get(aspect, ())
        out = {k: v for k, v in fields.items() if k in wanted}
        age_limit = max_age_for(model_key)
        if age_limit > 0:
            try:
                age = time.time() - float(rec.get("derived_at") or 0.0)
            except (TypeError, ValueError):
                age = age_limit + 1.0
            if age > age_limit:
                self._stats["expired"] += 1
                return out, "expired"
        self._stats["hits"] += 1
        return out, "fresh"

    def record(self, model_key: str) -> Optional[dict]:
        """The whole stored record (provenance included), or None. Diagnostics."""
        rec = self._snapshot().get(str(model_key))
        return dict(rec) if isinstance(rec, dict) else None

    def keys(self) -> list:
        return sorted(self._snapshot().keys())

    # -- write --------------------------------------------------------------
    def put(self, model_key: str, identity: str, fields: dict, aspects,
            *, source: str = "", dir_mtime: Optional[float] = None) -> bool:
        """Persist ``fields`` for ``model_key``. MERGES aspects: writing the size
        aspect never drops a status aspect already stored (and vice versa)."""
        return bool(self.put_many(
            [(model_key, identity, fields, aspects, dir_mtime)], source=source))

    def put_many(self, entries: Iterable, *, source: str = "") -> int:
        """One transaction for N rows — the repair sweep writes ~107 at once and
        must not take the lock 107 times.

        ``entries``: ``(model_key, identity, fields, aspects, dir_mtime)``."""
        staged = []
        for model_key, identity, fields, aspects, dir_mtime in entries:
            if not isinstance(fields, dict):
                continue
            aspects = [a for a in (aspects or []) if a in ASPECT_FIELDS]
            if not aspects:
                continue
            allowed = set()
            for a in aspects:
                allowed.update(ASPECT_FIELDS[a])
            clean = {k: v for k, v in fields.items() if k in allowed}
            if not _json_safe(clean):
                # Never persist something we cannot read back — degrade to
                # "absent", i.e. this row keeps deriving live.
                logger.debug("model-physical: %s produced non-JSON fields; not "
                             "persisted", model_key)
                self._stats["errors"] += 1
                continue
            staged.append((str(model_key), identity, clean, aspects, dir_mtime))
        if not staged:
            return 0

        now = time.time()

        def _mut(models: dict):
            written = 0
            for model_key, identity, clean, aspects, dir_mtime in staged:
                prev = models.get(model_key)
                merged_fields, merged_aspects = {}, []
                if isinstance(prev, dict) and prev.get("identity") == identity:
                    # Same model, other half already known — keep it.
                    old_fields = prev.get("fields")
                    if isinstance(old_fields, dict):
                        merged_fields.update(old_fields)
                    merged_aspects = [a for a in (prev.get("aspects") or [])
                                      if a in ASPECT_FIELDS]
                merged_fields.update(clean)
                for a in aspects:
                    if a not in merged_aspects:
                        merged_aspects.append(a)
                # A re-derived aspect REPLACES its half wholesale: a key the
                # deriver stopped producing (e.g. filename_warning after the
                # pin was fixed) must disappear, not linger.
                for a in aspects:
                    for field in ASPECT_FIELDS[a]:
                        if field not in clean:
                            merged_fields.pop(field, None)
                models[model_key] = {
                    "fields": merged_fields,
                    "identity": identity,
                    "aspects": sorted(merged_aspects),
                    "derived_at": now,
                    "dir_mtime": dir_mtime,
                    "source": source or "",
                }
                written += 1
            return written

        result = self._transaction(_mut)
        if result:
            self._stats["writes"] += int(result)
        return int(result or 0)

    def forget(self, model_key: str, reason: str = "") -> bool:
        """Drop one row — the next read derives it live.

        Targeted invalidation is the whole point of persisting: a delete /
        download / prune knows WHICH model changed, so the other ~106 rows stay
        warm. (The in-process memo could only express a whole-table drop.)"""
        def _mut(models: dict):
            return models.pop(str(model_key), None) is not None
        result = self._transaction(_mut)
        self._stats["forgets"] += 1
        logger.info("model physical state dropped for %s (%s)", model_key,
                    reason or "unspecified")
        return bool(result)

    def forget_all(self, reason: str = "") -> int:
        """Drop every row — for events whose blast radius is the whole store
        (a discovery re-walk, an applied reconcile that moved weights)."""
        def _mut(models: dict):
            n = len(models)
            models.clear()
            return n
        result = self._transaction(_mut)
        self._stats["forgets"] += 1
        logger.info("model physical state dropped for ALL models (%s)",
                    reason or "unspecified")
        return int(result or 0)

    def reconcile_identities(self, registry: dict, reason: str = "") -> int:
        """Drop rows the registry no longer backs, or whose ROUTING IDENTITY
        moved (a re-keyed / re-routed model). Returns how many were dropped.

        Cheap and I/O-free: it compares digests, it does not touch the store."""
        try:
            wanted = {str(k): identity_of(v if isinstance(v, dict) else {})
                      for k, v in (registry or {}).items()}
        except Exception:  # noqa: BLE001
            logger.debug("model-physical identity reconcile skipped",
                         exc_info=True)
            return 0
        current = self._snapshot()
        doomed = [k for k, rec in current.items()
                  if k not in wanted
                  or not isinstance(rec, dict)
                  or rec.get("identity") != wanted[k]]
        if not doomed:
            return 0

        def _mut(models: dict):
            n = 0
            for k in doomed:
                if models.pop(k, None) is not None:
                    n += 1
            return n

        result = self._transaction(_mut)
        if result:
            self._stats["forgets"] += 1
            logger.info("model physical state dropped for %d re-keyed/removed "
                        "model(s) (%s)", int(result), reason or "unspecified")
        return int(result or 0)

    # -- diagnostics --------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            out = dict(self._stats)
            out["entries"] = len(self._cache) if self._loaded else None
        out["path"] = self.path()
        out["max_age_s"] = max_age_seconds()
        out["poll_s"] = poll_seconds()
        return out

    def reset(self) -> None:
        """Forget this process's snapshot + counters WITHOUT touching the file.
        Test/diagnostic hook — use :meth:`forget_all` for a real change."""
        with self._lock:
            self._cache = {}
            self._stamp = None
            self._checked_at = 0.0
            self._loaded = False
            for k in self._stats:
                self._stats[k] = 0


physical_store = PhysicalStore()


# ── module-level convenience (mirrors the model_status_cache surface) ──────
def lookup_physical(model_key: str, model: Any,
                    aspect: str = ASPECT_STATUS) -> Tuple[Optional[dict], str]:
    return physical_store.lookup(model_key, identity_of(model), aspect)


def record_physical(model_key: str, model: Any, fields: dict, aspects,
                    *, source: str = "",
                    dir_mtime: Optional[float] = None) -> bool:
    return physical_store.put(model_key, identity_of(model), fields, aspects,
                              source=source, dir_mtime=dir_mtime)


def forget_physical(model_key: Optional[str], reason: str = "") -> bool:
    """Drop one model's persisted physical state (all of it when key is None)."""
    if model_key is None:
        return bool(physical_store.forget_all(reason))
    return physical_store.forget(model_key, reason)


def forget_all_physical(reason: str = "") -> int:
    return physical_store.forget_all(reason)


def reconcile_physical_identities(registry: dict, reason: str = "") -> int:
    return physical_store.reconcile_identities(registry, reason)


def physical_stats() -> dict:
    return physical_store.stats()


def reset_physical_store() -> None:
    physical_store.reset()
