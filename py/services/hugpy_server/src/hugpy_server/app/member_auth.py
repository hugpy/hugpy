"""Server-side auth gate for the STUDIO / MEDIA plane (the member surface).

WHY
---
2026-08-06 audit: ``/media/*`` (the media-intelligence bridge), ``/ml/*``,
``/uploads``, ``/session/*`` and ``/chat/stream`` were gated by NOTHING. Proven
live against gunicorn on :7002 with no cookie and no token:

    POST /media/analyze  -> 400 (route reached)      POST /uploads -> 400 (reached)
    GET  /ml             -> 200                      POST /session/ping -> 200
    GET  /keys           -> 401 (console gate)       GET /video/... -> 401 (video gate)

So an anonymous caller could upload files into the shared store, drive the media
pipelines and spend the fleet's GPU through /chat/stream — the exact holes the
console gate (operator_auth) and the video gate (video_auth) had already closed
on their own surfaces. This is the third gate, for the plane those two leave out.

DESIGN — a SEPARATE gate, mirroring video_auth.py
-------------------------------------------------
Its own ``before_request``, matching ONLY the studio/media surface. It is
structurally distinct from the console gate for the same reason the video gate
is: the credential that satisfies THIS gate (an ordinary approved member, or an
API key) must never satisfy ``operator_auth``'s console/operator rules. There is
no code path from the console gate to anything in this module.

The acceptance rule is::

    member OR operator  (operator_auth.member_authenticated())
    OR a valid hugpy API key   (the M2M product path — see _api_key_principal)

MEMBERS ARE ALLOWED HERE, DELIBERATELY. This plane is the product: uploading a
file, running a vision/extract pass, streaming a chat. What a member may NOT do
is the console's control plane — that boundary lives in operator_auth._SENSITIVE
(member -> 403), not here.

DENY SHAPE mirrors video_auth: a browser SHELL navigation is redirected to the
console root (where the ONE login lives — same session, no second login); a
data/XHR/media call gets a clean JSON 401.

Mode-aware and safe to ship: in ``open`` mode (self-hosted ``pip install hugpy``
with no operator token) ``principal_role()`` reports operator, so this plane
stays open with no login exactly as it is today — only ``external`` mode
(dev.hugpy.ai) enforces.
"""
from __future__ import annotations

import logging
import re

from flask import request, redirect, jsonify

from hugpy_server.app.operator_auth import member_authenticated

logger = logging.getLogger(__name__)

# The studio/media surface: the media-intelligence bridge (/media/analyze and
# friends), the per-task ML routes (/ml, /ml/gate), the upload endpoint
# (/uploads), the upload SESSION lifecycle (/session/ping|end|file) and the chat
# stream (/chat/stream), and the generic dispatch passthrough (/prompt and
# /prompt/tasks — the SAME execute_prompt surface as /ml, added 2026-09-14 after
# the review found it reachable UNAUTHENTICATED unlike its /chat + /ml siblings).
# Matched AFTER stripping a leading /api (nginx strips it; the ApiPrefixMiddleware
# does too — we strip defensively so a direct-to-gunicorn /api/uploads is covered
# identically).
#
# NOT here: /video and /movie (video_auth owns those), /v1/* (gated by the
# API-key system), and every console route (operator_auth owns those).
_MEMBER_SURFACE = re.compile(r"^/(media|ml|uploads|session|chat|prompt)(/|$)")


from hugpy_server.app.auth_common import normalized_path as _normalized_path


def _is_member_surface() -> bool:
    return bool(_MEMBER_SURFACE.match(_normalized_path()))


# --------------------------------------------------------------------------- #
# M2M SEAM. The only non-session credential this gate accepts: a valid hugpy API
# key (the same store /v1 validates against). It exists because this plane has
# programmatic consumers that are NOT browsers with a console session — the
# Discord bot arm and the CLI stream /chat/stream, and a key is the credential
# the product already mints for exactly that. Like video_auth's share seam, it is
# consulted ONLY from this gate, so an API key can never authorize a console
# route. The store import is lazy so this module stays importable if the key
# layer is unhappy, and nothing is consulted for a request carrying no key.
# --------------------------------------------------------------------------- #
def _presented_api_key() -> str:
    key = (request.headers.get("X-API-Key") or "").strip()
    if key:
        return key
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
        # An hpv_ bearer is a VIDEO-SHARE token (video_auth's seam) — never a
        # product key; leave it to the gate that owns it.
        if bearer and not bearer.startswith("hpv_"):
            return bearer
    return ""


def _api_key_principal(request) -> bool:  # noqa: A002 (mirror caller's name)
    token = _presented_api_key()
    if not token:
        return False
    try:
        from hugpy_server.app.functions.imports.utils.api_keys import verify_api_key
        return bool(verify_api_key(token))
    except Exception:  # noqa: BLE001 — a store hiccup must not open OR crash the gate
        logger.warning("member gate: api-key verification failed", exc_info=True)
        return False


def _member_authorized() -> bool:
    """A request may use the studio/media plane iff it carries a member or
    operator identity (or open-mode/operator-token access), or a valid API key."""
    if member_authenticated():
        return True
    if _api_key_principal(request):
        return True
    return False


from hugpy_server.app.auth_common import is_shell_request as _is_shell_request


def install_member_gate(app) -> None:
    """Register the studio/media member gate on a Flask app (idempotent)."""
    if getattr(app, "_member_gate_installed", False):
        return
    app._member_gate_installed = True

    @app.before_request
    def _member_gate():
        if request.method == "OPTIONS":
            return None  # never block CORS preflight
        if not _is_member_surface():
            return None  # not our surface — every other route is untouched
        if _member_authorized():
            return None
        # Denied. A browser navigating to the media SPA is sent to the console
        # root, where the ONE login lives. Data/upload/XHR calls get a JSON 401.
        if _is_shell_request():
            return redirect("/")
        return jsonify({
            "error": "authentication required for the studio/media surface",
        }), 401

    logger.info("member auth gate installed (surface=%s)", _MEMBER_SURFACE.pattern)
