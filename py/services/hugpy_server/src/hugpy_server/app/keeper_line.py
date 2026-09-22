"""THE DIRECT LINE to the hugpy VM's keeper (operator ruling 2026-08-20).

One primitive, two consumers: the help widget's Ask tab and the download
queue's diagnose verb. It execs the SAME B seat the station console and the
bridge-inbox watcher use — ``python3 -m hugpy_agent.mct.b_answer`` — as this
process's own user (the central already runs INSIDE the hugpy VM, so no lxc
hop is needed). No canned answers, no substitute model: when the keeper cannot
be reached the caller gets ``offline: True`` and B's deterministic state
readout, never a fabricated reply.

Grounding: the keeper reads its own MCT ledger/catalog state (b_answer builds
that itself); ``fleet_grounding()`` prepends what B cannot see from the ledger
— live central health, worker roster, and the download queue including the
persistent failure records.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request

KEEPER_TIMEOUT_S = float(os.environ.get("HUGPY_KEEPER_LINE_TIMEOUT", "180"))
_MCT_WORKSPACE = os.environ.get(
    "HUGPY_MCT_WORKSPACE", os.path.expanduser("~/.mct/repl"))
_SELF_BASE = os.environ.get("HUGPY_SELF_BASE", "http://127.0.0.1:7002")


def _self_get(path: str, timeout: float = 5.0):
    """A read from our own open GET surface — decoupled from route internals
    on purpose (grounding must never crash the line)."""
    try:
        with urllib.request.urlopen(_SELF_BASE + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None


def fleet_grounding(include_queue: bool = True) -> str:
    """A compact factual block the keeper can reason over. Facts only —
    the keeper draws the conclusions."""
    from hugpy_storage.downloader.presence import downloader_alive
    from hugpy_storage.downloader.queue import queue_depth, queue_healthy
    facts: dict = {
        "downloader_alive": bool(downloader_alive()),
        "download_queue_healthy": bool(queue_healthy()),
        "download_queue_depth": queue_depth(),
    }
    root = os.environ.get("DEFAULT_ROOT") or "/mnt/llm_storage"
    try:
        du = shutil.disk_usage(root)
        facts["store_free_gb"] = round(du.free / 2**30, 1)
    except OSError:
        facts["store_free_gb"] = None
    ready = _self_get("/readiness")
    if isinstance(ready, dict):
        facts["readiness"] = {k: ready.get(k) for k in
                              ("storage", "serving", "version") if k in ready}
    workers = _self_get("/llm/workers")
    if isinstance(workers, dict):
        facts["workers"] = workers.get("workers") or workers
    if include_queue:
        jobs = _self_get("/jobs")
        if isinstance(jobs, list):
            facts["download_jobs"] = [
                {k: d.get(k) for k in ("id", "model_key", "status", "error",
                                       "error_reason", "message", "attempt",
                                       "progress", "diagnosis")}
                for d in jobs][:40]
    return json.dumps(facts, default=str)[:8000]


def keeper_ask(text: str, history: list | None = None,
               timeout: float = KEEPER_TIMEOUT_S) -> dict:
    """Ask the hugpy VM's keeper. Returns ``{reply, offline}``.

    ``offline: True`` means B's brain was unreachable and the reply is its
    deterministic state readout — honest degradation, clearly labelled, never
    a pretend answer from some other model."""
    payload = json.dumps({
        "workspace": _MCT_WORKSPACE,
        "text": text,
        "history": list(history or [])[-20:],
    })
    try:
        proc = subprocess.run(
            ["python3", "-m", "hugpy_agent.mct.b_answer"],
            input=payload.encode("utf-8"),
            capture_output=True, timeout=timeout,
            env={**os.environ, "HOME": os.path.expanduser("~")},
        )
        out = json.loads((proc.stdout or b"{}").decode("utf-8", "replace"))
        reply = str(out.get("reply") or "").strip()
        if reply:
            return {"reply": reply, "offline": bool(out.get("offline"))}
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        return {"reply": f"keeper seat returned nothing"
                         f"{(' — ' + err[-400:]) if err else ''}",
                "offline": True}
    except subprocess.TimeoutExpired:
        return {"reply": f"keeper did not answer within {int(timeout)}s",
                "offline": True}
    except Exception as exc:  # noqa: BLE001
        return {"reply": f"keeper line failed: {exc}", "offline": True}
