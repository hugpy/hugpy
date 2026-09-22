from __future__ import annotations

import os
import shutil
import socket
import threading
import time
from typing import Any

from hugpy_fleet.central.config import settings


# describe_self() is polled by the console every ~15s. It touches the storage
# mount (exists + disk_usage), which BLOCKS for the mount timeout on a degraded
# NFS/sshfs share — stalling gunicorn workers and making the whole console slow.
# So we (a) run the probe with a hard timeout in a thread and (b) cache the
# result briefly, so a sick mount degrades to "unknown disk" instantly instead
# of hanging every request.
_DISK_CACHE: dict[str, Any] = {"at": 0.0, "value": None, "mounted": None}
_DISK_CACHE_TTL = 30.0
_DISK_PROBE_TIMEOUT = 1.5
_DISK_LOCK = threading.Lock()


def _disk_raw(path: str) -> dict[str, int | None]:
    try:
        usage = shutil.disk_usage(path if os.path.exists(path) else "/")
        return {"total": usage.total, "used": usage.used, "free": usage.free}
    except OSError:
        return {"total": None, "used": None, "free": None}


def _probe_storage(path: str) -> tuple[dict[str, int | None], bool]:
    """Run the (potentially blocking) mount probe with a hard timeout.

    Returns (disk_info, mounted). On timeout/hang we return unknowns rather than
    blocking the request thread, so a degraded mount can't take the API down.
    """
    result: dict[str, Any] = {}

    def _work():
        result["disk"] = _disk_raw(path)
        result["mounted"] = os.path.exists(path)

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(_DISK_PROBE_TIMEOUT)
    if t.is_alive():
        # Probe is stuck on a hung mount — report unknown, don't wait for it.
        return {"total": None, "used": None, "free": None}, False
    return result.get("disk", {"total": None, "used": None, "free": None}), bool(result.get("mounted"))


def _disk_cached(path: str) -> tuple[dict[str, int | None], bool]:
    now = time.time()
    with _DISK_LOCK:
        if _DISK_CACHE["value"] is not None and (now - _DISK_CACHE["at"]) < _DISK_CACHE_TTL:
            return _DISK_CACHE["value"], _DISK_CACHE["mounted"]
    disk, mounted = _probe_storage(path)
    with _DISK_LOCK:
        _DISK_CACHE.update(at=now, value=disk, mounted=mounted)
    return disk, mounted


def _disk(path: str) -> dict[str, int | None]:
    
    return _disk_cached(path)[0]


def describe_self() -> dict[str, Any]:
    """Describe this node as a peer entry — the central registry/storage box."""
    hostname = socket.gethostname()
    role = os.environ.get("LLM_PEER_ROLE", "central")
    name = os.environ.get("LLM_PEER_NAME", hostname)

    disk, mounted = _disk_cached(str(settings.storage_root))

    return {
        "name": name,
        "host": hostname,
        "role": role,
        "storage_root": str(settings.storage_root),
        "manifest_path": str(settings.manifest_path),
        "storage_mounted": mounted,
        "disk": disk,
        "status": "online",
    }


def _worker_as_peer(worker: dict[str, Any]) -> dict[str, Any]:
    """Project a registered GPU worker into the peer shape PeersBar renders."""
    return {
        "name": worker.get("name") or worker.get("id"),
        "host": worker.get("url"),
        "role": worker.get("role", "worker"),
        "storage_root": worker.get("url", ""),
        "manifest_path": "",
        "storage_mounted": worker.get("status") == "online",
        "disk": None,
        "status": worker.get("status", "offline"),
        "gpus": worker.get("gpus") or [],
        "models": worker.get("models", []),
        "loaded_models": worker.get("loaded_models", []),
        "worker_id": worker.get("id"),
    }


def list_peers() -> list[dict[str, Any]]:
    # Central node first, then every GPU worker that has joined the pool.
    from hugpy_fleet.central.workers import list_workers

    peers = [describe_self()]
    peers.extend(_worker_as_peer(w) for w in list_workers())
    return peers

def execute(**kwargs):
    """Delegated module execution. Pure **kwargs so prune_inputs passes
    every field straight through — no positional reshaping of 'file'."""
    from hugpy_engine.dispatch.dispatch import execute_prompt
    delegated = kwargs.pop("delegated", False)
    if delegated:
        kwargs["_force_local"] = True      # loop guard, consumed by resolve()
    result = execute_prompt(**kwargs)
    return result.model_dump() if hasattr(result, "model_dump") else result
