"""CENTRAL TRANSFER LEDGER — central's own accounting of the weight bytes it
serves to workers (2026-09-23).

Central SERVES every central->worker pull itself: the worker fetches
``/llm/models/<key>/manifest`` (the file set + total), then ``/file`` with HTTP
Range segments (64 MB x 8 parallel streams per worker). So "is aeb downloading
this model, how far along, how fast" is a fact central already has — it does
not have to be estimated from heartbeats or activity. This module is that
fact, recorded at the source:

  per (model_key, worker): started_at, total_bytes + files (from the manifest),
  bytes_served (bytes actually written to the socket, per file capped at the
  file size so a retried segment is not double counted), segments_in_flight,
  last_request_at, finished_at, status:

    active    a segment is streaming, or one was requested < STALL_S ago
    stalled   no request for STALL_S (default 30 s) and not complete
    complete  every manifest file served in full
    aborted   stalled for ABORT_S (default 600 s)

In memory (the API runs one process); the bounded history of finished
transfers is appended to ``$PROJECTS_HOME/transfers.jsonl`` on completion or
abort. Never raises into a transfer: every hook is guarded by the callers.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

STALL_S = float(os.environ.get("HUGPY_TRANSFER_STALL_S", "30"))
ABORT_S = float(os.environ.get("HUGPY_TRANSFER_ABORT_S", "600"))
HISTORY = 200


def _jsonl_path() -> Optional[str]:
    env = os.environ.get("HUGPY_TRANSFERS_JSONL")
    if env:
        return env
    try:
        from hugpy_platform.constants import PROJECTS_HOME
        return os.path.join(str(PROJECTS_HOME), "transfers.jsonl")
    except Exception:  # noqa: BLE001
        return None


class TransferLedger:
    def __init__(self, *, stall_s: float = STALL_S, abort_s: float = ABORT_S, history: int = HISTORY,
                 jsonl_path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.stall_s, self.abort_s, self.history = stall_s, abort_s, history
        self.jsonl_path = jsonl_path
        self.clock = clock
        self._lock = threading.Lock()
        self._live: Dict[Tuple[str, str], dict] = {}
        self._done: "OrderedDict[Tuple[str, str, float], dict]" = OrderedDict()

    # -- writers (called from the transfer routes) --------------------------
    def _entry(self, model_key: str, worker: str, now: float) -> dict:
        key = (model_key, worker)
        e = self._live.get(key)
        if e is None:
            e = {"model_key": model_key, "worker_id": worker, "started_at": now, "total_bytes": None,
                 "files": {}, "manifest": False, "segments_in_flight": 0, "last_request_at": now,
                 "finished_at": None, "requests": 0}
            self._live[key] = e
        return e

    def manifest(self, model_key: str, worker: str, files: List[dict], total_bytes: Optional[int] = None) -> None:
        """The worker fetched the transfer manifest: the file set to expect.
        A new manifest for a finished pair starts a new transfer."""
        now = self.clock()
        with self._lock:
            e = self._live.get((model_key, worker))
            if e is not None and e.get("finished_at"):
                self._retire((model_key, worker), e, "complete")
                e = None
            e = self._entry(model_key, worker, now)
            e["manifest"] = True
            for f in files or []:
                p = str(f.get("path") or "")
                if p:
                    e["files"].setdefault(p, {"size": int(f.get("size") or 0), "served": 0})
            e["total_bytes"] = int(total_bytes) if total_bytes is not None else sum(
                v["size"] for v in e["files"].values())
            e["last_request_at"] = now

    def begin(self, model_key: str, worker: str, path: str, file_size: int) -> Tuple[str, str, str]:
        now = self.clock()
        with self._lock:
            e = self._entry(model_key, worker, now)
            e["files"].setdefault(path, {"size": int(file_size or 0), "served": 0})
            if not e["manifest"]:
                e["total_bytes"] = sum(v["size"] for v in e["files"].values())
            if e["requests"] == 0:
                # The transfer STARTS at the first byte-range request, not at the
                # manifest read (live 2026-09-23: 60 "stalled" transfers with 0
                # bytes right after a central restart — every worker had only
                # re-read manifests for its presence check).
                e["started_at"] = now
            e["segments_in_flight"] += 1
            e["requests"] += 1
            e["last_request_at"] = now
        return (model_key, worker, path)

    def served(self, token: Tuple[str, str, str], nbytes: int) -> None:
        model_key, worker, path = token
        now = self.clock()
        with self._lock:
            e = self._live.get((model_key, worker))
            if e is None:
                return
            f = e["files"].setdefault(path, {"size": 0, "served": 0})
            f["served"] = f["served"] + int(nbytes)
            if f["size"]:
                f["served"] = min(f["served"], f["size"])
            e["last_request_at"] = now

    def end(self, token: Tuple[str, str, str]) -> None:
        model_key, worker, _path = token
        now = self.clock()
        with self._lock:
            e = self._live.get((model_key, worker))
            if e is None:
                return
            e["segments_in_flight"] = max(0, e["segments_in_flight"] - 1)
            e["last_request_at"] = now
            if (not e.get("finished_at") and e["manifest"] and e["files"]
                    and all(v["served"] >= v["size"] for v in e["files"].values())):
                e["finished_at"] = now
                self._persist(e, "complete")

    # -- reads ---------------------------------------------------------------
    def _status(self, e: dict, now: float) -> str:
        if e.get("finished_at"):
            return "complete"
        if e["requests"] == 0:
            # manifest read only: nothing has been transferred; not a transfer yet
            return "expected"
        if e["segments_in_flight"] > 0 or now - e["last_request_at"] < self.stall_s:
            return "active"
        if now - e["last_request_at"] >= self.abort_s:
            return "aborted"
        return "stalled"

    def _view(self, e: dict, status: str, now: float) -> dict:
        files = e["files"]
        served = sum(v["served"] for v in files.values())
        total = e.get("total_bytes") or sum(v["size"] for v in files.values()) or None
        end = e.get("finished_at") or now
        elapsed = max(0.0, end - e["started_at"])
        return {"model_key": e["model_key"], "worker_id": e["worker_id"], "status": status,
                "started_at": e["started_at"], "last_request_at": e["last_request_at"],
                "finished_at": e.get("finished_at"), "elapsed_s": round(elapsed, 3),
                "bytes_served": served, "total_bytes": total,
                "pct": round(100.0 * served / total, 1) if total else None,
                "mb_per_s": round(served / elapsed / 1e6, 2) if elapsed > 0 else None,
                "segments_in_flight": e["segments_in_flight"], "requests": e["requests"],
                "files_done": sum(1 for v in files.values() if v["size"] and v["served"] >= v["size"]),
                "files_total": len(files), "manifest_seen": e["manifest"],
                "idle_s": round(now - e["last_request_at"], 1), "source": "central-ledger"}

    def snapshot(self, *, include_done: bool = True) -> List[dict]:
        now = self.clock()
        out = []
        with self._lock:
            for key, e in list(self._live.items()):
                st = self._status(e, now)
                if st == "expected":
                    # Not a transfer: drop silently once the manifest read is stale.
                    if now - e["last_request_at"] >= self.abort_s:
                        self._live.pop(key, None)
                    continue
                if st == "aborted":
                    self._retire(key, e, "aborted")
                    continue
                if st == "complete" and now - e["finished_at"] > self.abort_s:
                    self._retire(key, e, "complete")
                    continue
                out.append(self._view(e, st, now))
            if include_done:
                out += [dict(v) for v in reversed(self._done.values())]
        return out

    # -- history -------------------------------------------------------------
    def _retire(self, key, e: dict, status: str) -> None:
        self._live.pop(key, None)
        view = self._view(e, status, e.get("finished_at") or self.clock())
        self._done[(key[0], key[1], e["started_at"])] = view
        while len(self._done) > self.history:
            self._done.popitem(last=False)
        if status == "aborted":
            self._write(view)

    def _persist(self, e: dict, status: str) -> None:
        self._write(self._view(e, status, e.get("finished_at") or self.clock()))

    def _write(self, view: dict) -> None:
        path = self.jsonl_path if self.jsonl_path is not None else _jsonl_path()
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(view, default=str) + "\n")
        except OSError:
            pass


ledger = TransferLedger()


def transfer_worker_id(request) -> str:
    """Who is pulling: the worker's own id header, else its address."""
    wid = (request.headers.get("X-Worker-Id") or "").strip()
    if wid:
        return wid
    return str(request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
               or request.remote_addr or "unknown")


class CountingIter:
    """Wrap a response iterable: every chunk yielded is counted into the
    ledger; ``close()`` ends the segment (and forwards to the inner close)."""

    def __init__(self, inner, token, led: TransferLedger):
        self._inner, self._token, self._led = inner, token, led
        self._closed = False

    def __iter__(self):
        for chunk in self._inner:
            try:
                self._led.served(self._token, len(chunk))
            except Exception:  # noqa: BLE001 — accounting never breaks a transfer
                pass
            yield chunk

    def close(self):
        try:
            if hasattr(self._inner, "close"):
                self._inner.close()
        finally:
            if not self._closed:
                self._closed = True
                try:
                    self._led.end(self._token)
                except Exception:  # noqa: BLE001
                    pass


def ledger_for(worker: dict, transfers: List[dict]) -> List[dict]:
    """The ledger rows whose puller is this worker (id, name or url host)."""
    w = worker or {}
    ids = {str(x) for x in (w.get("id"), w.get("name")) if x}
    host = ""
    url = str(w.get("url") or "")
    if "://" in url:
        host = url.split("://", 1)[1].split("/", 1)[0].rsplit(":", 1)[0]
    return [t for t in transfers or [] if str(t.get("worker_id")) in ids or (host and t.get("worker_id") == host)]
