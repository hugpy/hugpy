"""Central's link to the ONE toolserver on this host (abstract-toolserver
CENTRALIZATION.md, 2026-09-30).

Central never starts, embeds or re-implements a toolserver. At startup it asks
``abstract_toolserver.discovery.ensure_endpoint(start=False)`` which toolserver
is configured ($HUGPY_TOOLSERVER_URL) or ADVERTISED locally, and remembers the
answer; tool calls go through the shared client
(``abstract_toolserver.client``). The DB arm (``abstract_toolserver.metrics.
PgMetricsStore`` in model_metrics.py) is the same package.

Everything here is bounded and fail-open: a missing package or an unreachable
toolserver is logged once and central boots normally.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("hugpy_fleet.central.toolserver_link")

DISCOVERY_TIMEOUT_S = 8.0
_STATE: dict = {"url": "", "source": "unresolved"}
_LOCK = threading.Lock()


def discover(timeout: float = DISCOVERY_TIMEOUT_S) -> dict:
    """Resolve the toolserver endpoint (never starts one). Bounded by `timeout`
    via a daemon thread; returns {url, source[, reason]} and caches it."""
    box: dict = {}

    def _run():
        try:
            from abstract_toolserver.discovery import ensure_endpoint
            box["res"] = ensure_endpoint(start=False)
        except Exception as exc:  # noqa: BLE001 — package absent / probe failure
            box["res"] = {"url": "", "source": "none", "reason": str(exc)}

    t = threading.Thread(target=_run, name="toolserver-discovery", daemon=True)
    t.start()
    t.join(timeout)
    res = box.get("res") or {"url": "", "source": "none",
                             "reason": "discovery timed out after %.0fs" % timeout}
    res = {k: v for k, v in res.items() if k in ("url", "source", "reason")}
    with _LOCK:
        _STATE.clear()
        _STATE.update(res)
    if res.get("url"):
        logger.info("toolserver: %s (%s)", res["url"], res["source"])
    else:
        logger.warning("toolserver: none found (%s)", res.get("reason", "?"))
    return dict(res)


def endpoint() -> dict:
    with _LOCK:
        return dict(_STATE)


def client():
    """The shared abstract_toolserver client pointed at the discovered endpoint
    (explicit env config still wins inside the client's own resolution)."""
    from abstract_toolserver.client import ToolserverClient
    url = endpoint().get("url") or None
    return ToolserverClient(url=url)


def call(tool: str, **args):
    """Call a toolserver tool through the shared client."""
    return client().call(tool, args)
