"""k122 — HTTP for the Interim Ledger (``AH/oracle/interim_ledger.py``).

Three reads, no writes:

* ``GET /video/interim/entries`` — the filterable list.
* ``GET /video/interim/tree/<ref>`` — the family tree from any ref, walked
  across surfaces in both directions.
* ``GET /video/interim/stats``   — the honesty dashboard.

MOUNTED UNDER ``/video/`` ON PURPOSE. ``flask_app/app/video_auth.py`` installs
an app-wide ``before_request`` gate matching ``^/(video|movie)(/|$)`` with the
member-or-operator-session-or-share check, so every route here inherits the
video surface's existing credential check rather than growing a second one.
That is the same reasoning ``script_first_routes.py`` records, and it is pinned
by ``test_every_route_is_under_the_video_prefix_so_it_inherits_the_video_gate``
— a route added outside ``BASE`` fails that test rather than shipping unguarded.

A NEW blueprint file rather than a route in ``video_routes.py``: that module is
another task's dirty working file, so this follows the seam script_first
already cut — one file plus one import line in ``routes/__init__.py``, which
``abstract_flask._discover_blueprints`` picks up by the ``*_bp`` suffix.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

interim_bp = Blueprint("interim_bp", __name__)

#: The single mount point, so a route and its documentation cannot drift.
BASE: str = "/video/interim"

#: Hard ceiling on one page, whatever the caller asks for.
MAX_LIMIT: int = 1000


def _ledger_module():
    """Lazy import — the oracle package is heavy and must not load at blueprint
    import time (it pulls the model registry in)."""
    from hugpy_oracle import interim_ledger
    return interim_ledger


def _guard(fn: Callable[..., Any]):
    """Same shape as ``script_first_routes._guard``: refusals become their own
    JSON body with the ``error`` alias the React transport keeps, and anything
    unexpected becomes a 500 that names itself instead of a blank page."""
    def wrapped(*args: Any, **kwargs: Any):
        ledger_module = _ledger_module()
        try:
            return fn(ledger_module, *args, **kwargs)
        except ledger_module.LedgerRefused as exc:
            logger.info("interim: %s refused (%s): %s", request.path, exc.code,
                        exc.message)
            return jsonify(exc.to_dict()), exc.http_status
        except Exception as exc:                    # noqa: BLE001
            logger.warning("interim: %s failed: %s", request.path, exc,
                           exc_info=True)
            return jsonify({"ok": False, "code": "UNEXPECTED",
                            "message": f"{type(exc).__name__}: {exc}",
                            "errors": [str(exc)], "detail": {},
                            "error": f"UNEXPECTED: {type(exc).__name__}: {exc}"}), 500
    wrapped.__name__ = fn.__name__
    return wrapped


def _flag(name: str) -> bool | None:
    """A tri-state query flag: absent means "do not filter", NOT "false"."""
    raw = request.args.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def _str(name: str) -> str | None:
    raw = request.args.get(name)
    raw = (raw or "").strip()
    return raw or None


def _load(ledger_module):
    """One ledger per request, cached on disk between requests.

    ``rebuild=1`` forces a rescan — the cache is a convenience, never a source
    of truth, and an operator who just ran something must be able to say so.
    """
    return ledger_module.load_ledger(rebuild=bool(_flag("rebuild")))


@interim_bp.route(f"{BASE}/entries", methods=["GET"])
@_guard
def interim_entries(ledger_module):
    ledger = _load(ledger_module)
    limit = _int("limit", 200, low=1, high=MAX_LIMIT)
    offset = _int("offset", 0, low=0, high=10_000_000)
    entries = ledger.query(
        surface=_str("surface"), kind=_str("kind"), since=_str("since"),
        until=_str("until"), status=_str("status"), model=_str("model"),
        has_gap=_flag("has_gap"), scored=_flag("scored"),
        terminal=_flag("terminal"), q=_str("q"),
        limit=limit, offset=offset)
    return jsonify({
        "ok": True,
        "count": len(entries),
        "limit": limit,
        "offset": offset,
        "total_indexed": len(ledger),
        "built_at": ledger.built_at,
        "entries": [e.to_dict() for e in entries],
        # So a caller never has to guess which filter values exist.
        "surfaces": list(ledger_module.SURFACES),
        "statuses": list(ledger_module.STATUSES),
    })


@interim_bp.route(f"{BASE}/tree/<path:ref>", methods=["GET"])
@_guard
def interim_tree(ledger_module, ref: str):
    """The family tree from ANY ref — an entry_id, a job id, an asset id, an
    artifact path, a content digest. ``path:`` so a filesystem path can be the
    ref without the caller having to escape it."""
    ledger = _load(ledger_module)
    return jsonify(ledger.tree(
        ref,
        up=_int("up", 6, low=0, high=24),
        down=_int("down", 6, low=0, high=24),
        max_nodes=_int("max_nodes", 400, low=1, high=4000)))


@interim_bp.route(f"{BASE}/stats", methods=["GET"])
@_guard
def interim_stats(ledger_module):
    return jsonify(_load(ledger_module).stats())


__all__ = ["interim_bp", "BASE"]
