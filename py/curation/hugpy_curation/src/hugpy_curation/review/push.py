"""review/push.py — ship a finished run from the box that ran it to central.

WHY THIS EXISTS
    The review pipeline runs where the GPU is (ae), so its rows land in that
    box's local sqlite (``REVIEW_DB``). Central's /llm/review/* routes read
    CENTRAL's DB. Left alone, the nightly timer reviews models all night and
    the operator's console shows nothing. This module closes that gap in the
    only direction that is safe: worker -> central, never back.

    Central's DB is the SOURCE OF TRUTH. The local DB stays the on-box record —
    it survives a network outage, and the pipeline's own "did I already screen
    this repo?" check reads it — so a failed push costs visibility, never data.

BEST-EFFORT, ALWAYS
    ``push_run`` never raises. A push failure logs ONE line and returns a dict
    saying so; the run that just completed is already durably recorded locally
    and must not be failed by a telemetry-shaped concern. Replay later with
    ``hugpy-curation review push --all``, which walks exactly the
    runs that have no ``pushed_at`` stamp.

CONFIGURATION (environment)
    REVIEW_CENTRAL_URL    central's API base, e.g. https://dev.hugpy.ai/api
                          Unset -> pushing is OFF and every call is a no-op.
    REVIEW_CENTRAL_TOKEN  the bearer credential. Optional: when unset this
                          falls back to WORKER_ENROLL_TOKEN (what the worker
                          agent already presents on register/heartbeat and on
                          /llm/evictions/ingest) and then HUGPY_OPERATOR_TOKEN.
                          On a worker box the enrollment token is the RIGHT
                          answer and needs no new secret — same credential,
                          same revocation.
    REVIEW_PUSH_HOST      the name rows are attributed to. Defaults to the
                          worker's WORKER_NAME, else the hostname.
    REVIEW_PUSH_TIMEOUT   seconds for the HTTP call (default 20).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request

from hugpy_curation.review import store

logger = logging.getLogger(__name__)

INGEST_PATH = "/llm/review/ingest"
DEFAULT_TIMEOUT = 20.0
# One POST carries one run. A run that screened a huge pool can still produce a
# few hundred rows, so the batch is chunked rather than sent as one giant body.
MAX_RESULTS_PER_POST = 250


def central_url() -> str | None:
    """Central's API base, or None when pushing is not configured."""
    return (os.environ.get("REVIEW_CENTRAL_URL") or "").strip().rstrip("/") or None


def central_token() -> str | None:
    """The bearer credential, in preference order (see the module docstring).

    Deliberately NOT a new secret: WORKER_ENROLL_TOKEN is the credential a
    worker already holds, and central's ingest gate is the same enrollment
    check that guards register/heartbeat — so revoking a worker stops its
    review pushes at the same instant it stops its heartbeat."""
    for var in ("REVIEW_CENTRAL_TOKEN", "WORKER_ENROLL_TOKEN",
                "HUGPY_OPERATOR_TOKEN"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    return None


def source_host() -> str:
    for var in ("REVIEW_PUSH_HOST", "WORKER_NAME"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    try:
        return socket.gethostname()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_payload(run_id: int, results: list[dict] | None = None) -> dict | None:
    """Read one run + its result rows out of the LOCAL db and shape the batch.

    Returns None when the run id doesn't exist locally. ``results`` may be
    supplied to skip the re-read (the in-line push right after a run already
    has them in hand — and passing them avoids racing a concurrent writer).

    Shape:
        {"host": "ae",
         "criteria": "nightly",
         "run": {"run_id": 12, "criteria": …, "started_at": …, "finished_at": …,
                 "screened": …, "passed": …, "downloaded": …, "smoked": …,
                 "error": null},
         "results": [{"run_id": 12, "criteria": …, "hub_id": …, "stage": …,
                      "passed": 1, "score": 8.5, "verdict": "trial",
                      "payload": {…full Review dict…}, "reviewed_at": …}, …]}

    ``run_id`` is this box's local id; central keys on (host, run_id), which is
    what makes a retried push an update instead of a duplicate."""
    row = store.get_run(run_id)
    if row is None:
        return None
    if results is None:
        results = store.results_for_run(run_id)
    run = {"run_id": int(row["id"]),
           "criteria": row.get("criteria"),
           "started_at": row.get("started_at"),
           "finished_at": row.get("finished_at"),
           "screened": row.get("screened"),
           "passed": row.get("passed"),
           "downloaded": row.get("downloaded"),
           "smoked": row.get("smoked"),
           "error": row.get("error")}
    out = []
    for r in results:
        out.append({"run_id": int(run_id),
                    "criteria": r.get("criteria") or row.get("criteria"),
                    "hub_id": r.get("hub_id"),
                    "stage": r.get("stage"),
                    "passed": r.get("passed"),
                    "score": r.get("score"),
                    "verdict": r.get("verdict"),
                    "payload": r.get("payload"),
                    "reviewed_at": r.get("reviewed_at")})
    return {"host": source_host(), "criteria": row.get("criteria"),
            "run": run, "results": out}


def _post(url: str, body: dict, token: str | None, timeout: float) -> dict:
    data = json.dumps(body, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8") or "{}"
    try:
        return json.loads(raw)
    except ValueError:
        return {}


def push_run(run_id: int, results: list[dict] | None = None,
             log=None) -> dict:
    """Best-effort: POST one run and its results to central. NEVER raises.

    Returns {"ok": bool, ...}. ``ok`` False with ``reason`` "not_configured"
    means no REVIEW_CENTRAL_URL — a local-only box, which is a normal state and
    not worth a warning at anything above debug."""
    say = log or (lambda m: logger.info("[review push] %s", m))
    base = central_url()
    if not base:
        return {"ok": False, "reason": "not_configured"}
    try:
        payload = build_payload(run_id, results=results)
    except Exception as exc:  # noqa: BLE001 — a local read fault is not the run's problem
        say(f"could not read run {run_id} for push: {type(exc).__name__}: {exc}")
        return {"ok": False, "reason": "unreadable", "error": str(exc)}
    if payload is None:
        say(f"run {run_id} not found locally — nothing to push")
        return {"ok": False, "reason": "no_such_run"}

    token = central_token()
    try:
        timeout = float(os.environ.get("REVIEW_PUSH_TIMEOUT") or DEFAULT_TIMEOUT)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    url = base + INGEST_PATH

    rows = payload["results"]
    # Chunk: the run header rides every chunk (upserting it repeatedly is a
    # no-op by design), so a partial delivery still lands a coherent run.
    chunks = [rows[i:i + MAX_RESULTS_PER_POST]
              for i in range(0, len(rows), MAX_RESULTS_PER_POST)] or [[]]
    accepted = rejected = 0
    try:
        for chunk in chunks:
            body = dict(payload, results=chunk)
            resp = _post(url, body, token, timeout)
            accepted += int(resp.get("accepted") or 0)
            rejected += int(resp.get("rejected") or 0)
    except urllib.error.HTTPError as exc:
        # ONE line. A push failure is a visibility loss, never a run failure.
        say(f"push of run {run_id} to {url} refused: HTTP {exc.code} "
            f"{getattr(exc, 'reason', '')}")
        return {"ok": False, "reason": "http", "status": exc.code}
    except Exception as exc:  # noqa: BLE001
        say(f"push of run {run_id} to {url} failed: {type(exc).__name__}: {exc}")
        return {"ok": False, "reason": "unreachable", "error": str(exc)}

    try:
        store.mark_pushed(run_id)
    except Exception as exc:  # noqa: BLE001 — the push DID land; the stamp is bookkeeping
        say(f"run {run_id} pushed but the pushed_at stamp failed: {exc}")
    say(f"run {run_id} -> {base}: {accepted} accepted, {rejected} rejected")
    return {"ok": True, "run_id": run_id, "accepted": accepted,
            "rejected": rejected, "url": url}


def push_pending(limit: int = 50, log=None) -> list[dict]:
    """Replay every locally-produced run that has no pushed_at stamp."""
    out = []
    try:
        pending = store.unpushed_runs(limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[review push] could not list pending runs: %s", exc)
        return out
    for row in pending:
        out.append(push_run(int(row["id"]), log=log))
    return out
