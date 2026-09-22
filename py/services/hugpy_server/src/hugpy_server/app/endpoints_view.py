"""hugpy adapter for the reusable ``endpoints_explorer``.

The interactive ``/endpoints`` explorer (curated listing + try-it console + /media
styling) is a portable, framework-only module — ``endpoints_explorer`` — so it can be
lifted into ``abstract_flask`` as an upgrade to the generator. This file is the thin
hugpy-specific wiring: it injects hugpy's own notion of "internal" (reusing
``operator_auth._SENSITIVE`` so the docs flag can't drift from what actually gates the
routes) and hugpy's operator check (``operator_authenticated``) for the ``?all=1`` gate.

Everything else — collection, content negotiation, curation, the try-it page — lives in
``endpoints_explorer`` and is app-agnostic. Install is wrapped by the caller in
try/except; it must never break boot.
"""

from __future__ import annotations

import re as _re
from typing import List

from hugpy_server.app.endpoints_explorer import install_endpoints_explorer


# ── hugpy sensitivity classification (reuse operator_auth's own allowlist) ────

def _sensitive_rules():
    """operator_auth's (methods, path-regex) allowlist — the same one that gates the
    real routes, so the docs "internal" flag stays in lockstep. [] if unavailable."""
    try:
        from hugpy_server.app.operator_auth import _SENSITIVE
        return _SENSITIVE
    except Exception:
        return []


def _norm_path(url: str) -> str:
    # Match operator_auth's normalization: strip a leading /api (gunicorn dual-mount),
    # collapse <converters> so its concrete-path regexes match rule strings.
    if url == "/api" or url.startswith("/api/"):
        url = url[len("/api"):] or "/"
    return _re.sub(r"<[^>]+>", "X", url)


def _classify_internal(url: str, methods: List[str]) -> bool:
    path = _norm_path(url)
    mset = set(methods)
    for smethods, rx in _sensitive_rules():
        if (mset & smethods) and rx.match(path):
            return True
    return False


def _operator_ok() -> bool:
    """Whether the caller may see internal routes (?all=1 gate). Permissive in the
    self-hosted 'open' mode; enforced once the operator auth gate is active."""
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        return bool(operator_authenticated())
    except Exception:
        return True


def install_endpoints_view(app) -> None:
    install_endpoints_explorer(
        app,
        classify_internal=lambda url, methods: _classify_internal(url, methods),
        can_view_internal=lambda: _operator_ok(),
        brand="hugpy · API endpoints",
        accent="#8ab4ff",
        # dev.hugpy.ai's front (:7001 webpack) proxies ONLY /api to Flask and
        # SPA-swallows every other bare path into index.html — so a try-it fetch
        # to a bare route returns HTML, not the endpoint. Route calls through /api
        # (hugpy dual-mounts every route there); the explorer strips a listed
        # /api first so bare- and /api-listed rows both resolve to the one path.
        call_base="/api",
    )
