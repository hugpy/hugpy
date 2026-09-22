"""k118 — HTTP for the environment doctrine.

Thin, like ``script_first_routes``: every decision is already made in
``fleet_doctrine`` (the classification, the diff, the repair line). These routes
read the worker registry, read the doctrine, and serialize.

``GET /fleet/doctrine/status`` is the fleet-wide view an operator opens when a
task is mysteriously not routing anywhere. It answers from HEARTBEATS ONLY — no
outbound HTTP, so it stays a page-load-cheap read and a wedged worker cannot
hang it. A worker that has never sent ``doctrine_status`` is reported
``unknown``, explicitly, with the reason: "unknown" and "clean" must never look
alike, which is the same rule ``oracle.probes`` enforces on capabilities.

``GET /fleet/doctrine/status/<worker>?live=1`` is the one place that WILL go to
the box — it pulls ``/ops/environment`` through the sanctioned
``worker_http`` client and assesses it here, for a worker whose agent is too old
to self-assess.

Registration is one line in ``routes/__init__.py``: ``abstract_flask``'s
``_discover_blueprints`` registers every ``*_bp`` attribute of that module with
no url_prefix, which is why the paths are spelled in full below.
"""
from __future__ import annotations

import logging
from typing import Any

from flask import jsonify, request

from hugpy_fleet.central.workers import list_workers

logger = logging.getLogger(__name__)

try:  # the tree's blueprint factory (abstract_flask), same as fleet_routes
    from abstract_flask import get_bp
    fleet_doctrine_bp, logger = get_bp("fleet_doctrine_bp", __name__)  # noqa: F405
except Exception:  # pragma: no cover — plain Flask fallback keeps this importable
    from flask import Blueprint
    fleet_doctrine_bp = Blueprint("fleet_doctrine_bp", __name__)

BASE: str = "/fleet/doctrine"


def _doctrine():
    """The latest doctrine, imported LAZILY and never fatally.

    None means "central holds no doctrine", which every route below reports as
    such rather than as a clean fleet."""
    try:
        from hugpy_fleet.doctrine import doctrine as _d
        return _d.latest()
    except Exception as exc:  # noqa: BLE001
        logger.warning("fleet_doctrine: no loadable doctrine (%s: %s)",
                       type(exc).__name__, exc)
        return None


def _worker_row(worker: dict[str, Any]) -> dict[str, Any]:
    """One worker's doctrine line, from its heartbeat.

    The three states are kept DISTINCT on purpose:
      * a ``doctrine_status`` present -> the worker's own verdict, verbatim;
      * an ``environment_digest`` but no status -> the box reports its
        environment but holds no doctrine to judge it against;
      * neither -> an agent that predates k118. All three are ``unknown`` for
        eligibility purposes and none of them is ``ok``."""
    name = str(worker.get("name") or worker.get("id") or "?")
    status = worker.get("doctrine_status")
    digest = worker.get("environment_digest")
    if isinstance(status, dict) and status:
        row = dict(status)
        row.setdefault("verdict", "unknown")
    elif isinstance(digest, dict) and digest:
        row = {"verdict": "unknown", "blocked_tasks": [], "blockers": 0,
               "warnings": 0,
               "reason": ("this worker reports its environment but carries no "
                          "doctrine to assess it against — ask central with "
                          f"?live=1, or GET {BASE}/status/{name}?live=1")}
    else:
        row = {"verdict": "unknown", "blocked_tasks": [], "blockers": 0,
               "warnings": 0,
               "reason": ("no environment report on this worker's heartbeat — "
                          "its agent predates k118 (update the worker package)")}
    row["worker"] = name
    row["worker_id"] = worker.get("id")
    row["online"] = worker.get("status") == "online"
    row["pkg_version"] = worker.get("pkg_version")
    row["environment_digest"] = (digest or {}).get("digest") if isinstance(digest, dict) else None
    return row


@fleet_doctrine_bp.route(f"{BASE}/status", methods=["GET"])
def fleet_doctrine_status():
    """Every ONLINE worker's doctrine verdict, from heartbeats. ``?all=1``
    includes offline boxes."""
    include_all = str(request.args.get("all") or "").lower() in ("1", "true", "yes")
    try:
        workers = list_workers() or []
    except Exception as exc:  # noqa: BLE001 — an unreadable registry is data
        return jsonify({"ok": False, "error": {
            "code": type(exc).__name__,
            "message": f"worker registry unreadable: {exc}"}}), 503
    rows = [_worker_row(w) for w in workers
            if include_all or w.get("status") == "online"]
    current = _doctrine()
    blocked: dict[str, list[str]] = {}
    for row in rows:
        for task in (row.get("blocked_tasks") or []):
            blocked.setdefault(str(task), []).append(row["worker"])
    return jsonify({
        "ok": True,
        "doctrine": ({"version": current.version,
                      "reference": current.reference,
                      "provisional": current.provisional,
                      "pending": current.pending,
                      "created_at": current.created_at,
                      "entries": len(current.entries)}
                     if current is not None else None),
        "workers": rows,
        "counts": {
            "total": len(rows),
            "ok": sum(1 for r in rows if r.get("verdict") == "ok"),
            "warn": sum(1 for r in rows if r.get("verdict") == "warn"),
            "blocked": sum(1 for r in rows if r.get("verdict") == "blocked"),
            "unknown": sum(1 for r in rows if r.get("verdict") == "unknown"),
        },
        # task -> the workers where it is doctrine-blocked. The view an operator
        # actually wants: "why is nothing running text-to-image?"
        "blocked_tasks": {k: sorted(v) for k, v in sorted(blocked.items())},
    })


@fleet_doctrine_bp.route(f"{BASE}/versions", methods=["GET"])
def fleet_doctrine_versions():
    try:
        from hugpy_fleet.doctrine import doctrine as _d
        versions = _d.list_versions()
        directory = _d.doctrine_dir()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": {
            "code": type(exc).__name__, "message": str(exc)}}), 503
    return jsonify({"ok": True, "versions": versions, "directory": directory})


@fleet_doctrine_bp.route(f"{BASE}/status/<worker>", methods=["GET"])
def fleet_doctrine_worker(worker: str):
    """One worker in full. ``?live=1`` pulls ``/ops/environment`` from the box
    and assesses it HERE — the path for an agent too old to self-assess."""
    try:
        workers = list_workers() or []
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": {
            "code": type(exc).__name__, "message": str(exc)}}), 503
    match = [w for w in workers
             if str(w.get("name") or "") == worker or str(w.get("id") or "") == worker]
    if not match:
        return jsonify({"ok": False, "error": {
            "code": "UnknownWorker",
            "message": f"no worker named {worker!r} is registered"}}), 404
    row = match[0]
    if str(request.args.get("live") or "").lower() not in ("1", "true", "yes"):
        return jsonify({"ok": True, "worker": _worker_row(row)})

    current = _doctrine()
    if current is None:
        return jsonify({"ok": False, "error": {
            "code": "NoDoctrine",
            "message": "central holds no doctrine to assess against"}}), 503
    try:
        from hugpy_fleet.central import worker_http
        response = worker_http.get(row, "/ops/environment", call="status",
                                   read_timeout=20.0)
    except Exception as exc:  # noqa: BLE001 — unreachable is DATA, not a 500
        return jsonify({"ok": False, "error": {
            "code": "WorkerUnreachable", "message": str(exc)}}), 503
    if response.status_code == 404:
        return jsonify({"ok": False, "error": {
            "code": "NoEnvironmentEndpoint",
            "message": ("this worker agent predates GET /ops/environment; "
                        "update the worker package")}}), 501
    if response.status_code != 200:
        return jsonify({"ok": False, "error": {
            "code": "WorkerError",
            "message": f"/ops/environment answered {response.status_code}"}}), 502
    from hugpy_fleet.doctrine.doctor import assess
    assessment = assess(response.json(), current)
    payload = assessment.to_dict()
    payload["repair_plan"] = assessment.repair_plan()
    return jsonify({"ok": True, "worker": worker, "assessment": payload})


__all__ = ["BASE", "fleet_doctrine_bp"]
