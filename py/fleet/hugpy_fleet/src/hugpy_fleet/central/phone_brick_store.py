"""Phone-brick registry + run store.

The console's *video-analytics pool* is a fleet of phones running the
``phone_brick`` ONNX-YOLO worker (``/queue`` + ``/results`` + ``/status``). This
module is the single source of truth for that pool and for the orchestration
runs fanned across it, mirroring the GPU :mod:`.workers` registry:

    * phones self-register and heartbeat, and are ``online`` only if seen
      recently (so the UI can show live status);
    * one orchestration *run* (fan one image across the chosen phones) is a
      record the UI polls until it finishes.

Both live in JSON files beside the model manifest and are mutated under an
exclusive ``fcntl`` lock, because the API runs as several gunicorn processes —
an in-memory dict would split-brain (a phone registered in one process would be
invisible to a heartbeat or a poll handled by another). See
:class:`.workers.WorkerStore` for the same reasoning in more depth.
"""
from __future__ import annotations

import os
import json
import time
import uuid
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

try:
    import fcntl  # POSIX advisory locks — cross-process coordination.
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from hugpy_fleet.central.config import settings


# A phone/run that hasn't checked in within this window is considered offline.
HEARTBEAT_TIMEOUT_SECONDS = 45.0


def _registry_dir() -> str:
    """Where phones.json / runs.json / run outputs live (…/projects/)."""
    return settings.state_dir


def output_dir() -> str:
    """Directory the orchestrator seeds images into and serves to phones.

    Override with ``PHONE_BRICK_OUTPUT_DIR``; defaults to a ``phone_brick_runs``
    dir beside the model manifest. Phones fetch the seeded image from here via
    the file-server route, and annotated results are written back here.
    """
    d = os.environ.get("PHONE_BRICK_OUTPUT_DIR") or \
        os.path.join(_registry_dir(), "phone_brick_runs")
    os.makedirs(d, exist_ok=True)
    return d


def _now() -> float:
    return time.time()


class _LockedJson:
    """Disk-authoritative, multi-process-safe JSON map keyed by ``id``.

    Stored as ``{"<collection>": [ {id, ...}, ... ]}``. Reads serve from a short
    TTL cache (polls don't hammer the disk / block on a slow mount); writes take
    an exclusive cross-process lock, reload, mutate, and persist atomically.
    """

    _READ_TTL = 3.0

    def __init__(self, path: str, collection: str) -> None:
        self._path = path
        self._collection = collection
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache_at = 0.0
        parent = os.path.dirname(path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass

    def _read_unlocked(self, fh=None) -> Dict[str, Dict[str, Any]]:
        try:
            if fh is not None:
                fh.seek(0)
                raw = fh.read()
            elif os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = f.read()
            else:
                return {}
            if not raw.strip():
                return {}
            data = json.loads(raw)
            if isinstance(data, dict):
                return {r["id"]: r for r in data.get(self._collection, []) if r.get("id")}
        except (OSError, ValueError, KeyError):
            return {}
        return {}

    def _write_unlocked(self, fh, items: Dict[str, Dict[str, Any]]) -> None:
        payload = json.dumps({self._collection: list(items.values())}, indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    def _load(self) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        with self._lock:
            if self._cache is not None and (now - self._cache_at) < self._READ_TTL:
                return self._cache
            data = self._read_unlocked()
            self._cache = data
            self._cache_at = now
            return data

    @contextmanager
    def _transaction(self):
        with self._lock:
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
            fh = os.fdopen(fd, "r+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                items = self._read_unlocked(fh)
                yield items
                self._write_unlocked(fh, items)
                self._cache = items
                self._cache_at = time.time()
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()


# ---------------------------------------------------------------------------
# Phone registry
# ---------------------------------------------------------------------------
def _phone_online(phone: Dict[str, Any]) -> bool:
    return (_now() - (phone.get("last_seen") or 0)) <= HEARTBEAT_TIMEOUT_SECONDS


def _phone_view(phone: Dict[str, Any]) -> Dict[str, Any]:
    return {**phone, "status": "online" if _phone_online(phone) else "offline"}


class PhoneStore(_LockedJson):
    """Registry of phone-brick worker phones."""

    def __init__(self, path: Optional[str] = None) -> None:
        super().__init__(path or os.path.join(_registry_dir(), "phones.json"), "phones")

    def register(
        self,
        *,
        name: str,
        host: str,
        port: int = 5002,
        color: str = "#58a6ff",
        url: Optional[str] = None,
        phone_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a phone, or re-register an existing one by id, else by host:port."""
        with self._transaction() as phones:
            existing = None
            if phone_id and phone_id in phones:
                existing = phones[phone_id]
            else:
                for p in phones.values():
                    if p.get("host") == host and int(p.get("port", 0)) == int(port):
                        existing = p
                        break
            if existing is not None:
                existing.update(
                    name=name or existing.get("name"),
                    host=host or existing.get("host"),
                    port=int(port),
                    color=color or existing.get("color", "#58a6ff"),
                    url=url or existing.get("url"),
                    last_seen=_now(),
                )
                return _phone_view(existing)

            pid = phone_id or uuid.uuid4().hex
            phone = {
                "id": pid,
                "name": name or pid,
                "host": host,
                "port": int(port),
                "color": color or "#58a6ff",
                "url": url or f"http://{host}:{int(port)}",
                "created_at": _now(),
                "last_seen": _now(),
            }
            phones[pid] = phone
            return _phone_view(phone)

    def heartbeat(self, phone_id: str, *, live: Optional[Dict[str, Any]] = None,
                  host: Optional[str] = None, port: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Mark a phone alive and refresh its live ``/status`` snapshot."""
        with self._transaction() as phones:
            phone = phones.get(phone_id)
            if phone is None:
                return None
            phone["last_seen"] = _now()
            if host:
                phone["host"] = host
            if port:
                phone["port"] = int(port)
                phone["url"] = f"http://{phone['host']}:{int(port)}"
            if live is not None:
                # Live fields the UI shows: model_loaded, classes, queue_size, etc.
                phone["live"] = live
            return _phone_view(phone)

    def remove(self, phone_id: str) -> bool:
        with self._transaction() as phones:
            return phones.pop(phone_id, None) is not None

    def get(self, phone_id: str) -> Optional[Dict[str, Any]]:
        phone = self._load().get(phone_id)
        return _phone_view(phone) if phone else None

    def all(self) -> List[Dict[str, Any]]:
        return [_phone_view(p) for p in self._load().values()]

    def online(self) -> List[Dict[str, Any]]:
        return [p for p in self.all() if p["status"] == "online"]


# ---------------------------------------------------------------------------
# Run store (one orchestration fan-out over the pool)
# ---------------------------------------------------------------------------
class RunStore(_LockedJson):
    """Registry of orchestration runs the UI polls until they finish."""

    def __init__(self, path: Optional[str] = None) -> None:
        super().__init__(path or os.path.join(_registry_dir(), "phone_brick_runs.json"), "runs")

    def create(self, *, image: str, phone_ids: List[str]) -> Dict[str, Any]:
        rid = uuid.uuid4().hex
        with self._transaction() as runs:
            run = {
                "id": rid,
                "status": "queued",        # queued | running | done | error | cancelled
                "image": image,
                "phone_ids": phone_ids,
                "phases": [],              # final per-phone verdicts (with consensus)
                "progress": [],            # incremental per-phone verdicts as they arrive
                "current_phone": None,     # phone currently being asked
                "cancel_requested": False, # set by the cancel route; polled by the run
                "output_rel": None,
                "error": None,
                "created_at": _now(),
                "finished_at": None,
            }
            runs[rid] = run
            return dict(run)

    def request_cancel(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Flag a run for cancellation (the running job polls this)."""
        with self._transaction() as runs:
            run = runs.get(run_id)
            if run is None:
                return None
            # Only an in-flight run can be cancelled.
            if run.get("status") in ("queued", "running"):
                run["cancel_requested"] = True
            return dict(run)

    def is_cancelled(self, run_id: str) -> bool:
        run = self._load().get(run_id)
        return bool(run and run.get("cancel_requested"))

    def update(self, run_id: str, **fields) -> Optional[Dict[str, Any]]:
        with self._transaction() as runs:
            run = runs.get(run_id)
            if run is None:
                return None
            run.update(fields)
            return dict(run)

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self._load().get(run_id)
        return dict(run) if run else None

    def all(self) -> List[Dict[str, Any]]:
        runs = list(self._load().values())
        runs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return runs


phone_store = PhoneStore()
run_store = RunStore()


# Module-level convenience wrappers (mirrors workers.py's plain-function style).
def register_phone(**kwargs) -> Dict[str, Any]:
    return phone_store.register(**kwargs)


def heartbeat_phone(phone_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    return phone_store.heartbeat(phone_id, **kwargs)


def remove_phone(phone_id: str) -> bool:
    return phone_store.remove(phone_id)


def list_phones() -> List[Dict[str, Any]]:
    return phone_store.all()


def get_phone(phone_id: str) -> Optional[Dict[str, Any]]:
    return phone_store.get(phone_id)


def online_phones() -> List[Dict[str, Any]]:
    return phone_store.online()
