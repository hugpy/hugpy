"""Server-side auth gate for the /video (Video Intelligence) surface.

WHY
---
The video arm (SPA at ``/video/`` plus its ``/video/*`` and ``/movie/*`` API and
media routes) went public/ungated on 2026-07-15: the video SPA ships WITHOUT any
client-side login (unlike the console SPA, which has an AuthProvider login wall),
and none of the ``/video/*`` routes were in the operator-auth allowlist. So on
``dev.hugpy.ai`` (``HUGPY_AUTH_MODE=external``) an anonymous caller could list
studio clips, read their filesystem URIs, stream media, run generation jobs, and
drive the whole arm — proven live: anon ``GET /api/video/studio/clips`` returned
200 with real job rows while ``GET /api/keys`` correctly returned 401.

The operator (2026-07-18) directed that the video surface sit behind the SAME
auth boundary as the console: a valid console session, no second login.

DESIGN — a SEPARATE gate from ``operator_auth`` (deliberately)
------------------------------------------------------------
This is its OWN ``before_request``, matching ONLY the ``/video`` and ``/movie``
surface. It is structurally distinct from the console/operator gate in
``operator_auth.py`` for one load-bearing reason: a future SHARE-LINK feature
will mint a VIDEO-SCOPED credential (an external party uses video features
without a console login). That credential must be able to satisfy the video
gate but must NEVER satisfy the console/operator gate (``/keys*``, ``/settings``,
worker verbs, ``/agent/*``, …).

The acceptance rule here is::

    member-or-operator console session  OR  valid video-share credential

(2026-08-06: the session leg is ``member_authenticated()`` — see
_video_authorized. Console MUTATIONS remain operator-only in operator_auth, and
a member sees only their OWN artifacts through the owner filters in
video_routes.)

- **Console session** — reuses ``operator_auth``'s session check: the
  same first-party session cookie validated against the upstream ``/me`` (external
  mode), the ``HUGPY_OPERATOR_TOKEN`` automation path, and the permissive
  ``open``-mode default (self-hosted ``pip install hugpy`` product: no login, so
  the video arm is open there too). This is a ONE-WAY reuse: the video gate calls
  the console-session check; the console gate never calls anything here.
- **Video-share credential** — the ``_video_share_principal(request)`` SEAM
  below. It returns ``None`` today (no share feature yet), so nothing changes for
  the console session path. When the share slice lands it plugs in HERE and ONLY
  here, so the share credential is video-scoped BY CONSTRUCTION: there is no code
  path from the console/operator gate to ``_video_share_principal``.

STRUCTURAL INVARIANT (pinned by tests/test_video_gate.py)
---------------------------------------------------------
With a video-share principal present: ``/api/video/*`` is allowed, but
``/api/keys`` (and every console/operator route) is STILL denied — because those
routes are gated by ``operator_auth`` alone, which never consults the share hook.

The gate is mode-aware and safe to ship: in ``open`` mode (self-hosted product,
no operator token) ``operator_authenticated()`` is permissive, so the video arm
stays open with no login — only ``external`` mode (dev.hugpy.ai) enforces.
"""
from __future__ import annotations

import logging
import re

from flask import request, abort, redirect

from hugpy_server.app.operator_auth import operator_authenticated, member_authenticated

logger = logging.getLogger(__name__)

# The video surface: the arm SPA shell (/video, /video/<deep-link>), every
# /video/* API + media route, and the /movie/* preset routes. Matched AFTER
# stripping a leading /api (the ApiPrefixMiddleware / nginx strips it, but strip
# defensively too so a direct-to-gunicorn /api/video/... is covered identically).
_VIDEO_SURFACE = re.compile(r"^/(video|movie)(/|$)")


from hugpy_server.app.auth_common import normalized_path as _normalized_path


def _is_video_surface() -> bool:
    return bool(_VIDEO_SURFACE.match(_normalized_path()))


# --------------------------------------------------------------------------- #
# THE SHARE SEAM. This is the ONLY place a video-scoped share credential is ever
# consulted. It returns None today (no share feature). The future share slice
# implements it here — e.g. read an operator-minted, video-scoped token from a
# request arg / header, verify it (unexpired, video scope, not revoked), and
# return a truthy principal (a dict is fine) or None. Because it is called ONLY
# from _video_authorized() (the /video gate), the credential it accepts can never
# authorize a console/operator route — that property is structural, not
# conventional. Keep it that way: do NOT wire this into operator_auth.
# --------------------------------------------------------------------------- #
def _share_token(request) -> str:  # noqa: A002 (mirror caller's name)
    """The share credential this request presents, from the three carriers the
    /video SPA uses (see k9):

      * ``?share=<key>`` query — a browser SHELL navigation (``/video/?share=``)
        AND every element-src media load (``<img>``/``<video>`` to /video/media |
        /video/studio/clip), which cannot carry a custom header;
      * ``X-Video-Share: <key>`` header — the SPA's XHR fetch layer (its single
        transport choke point stamps this on every API call);
      * ``Authorization: Bearer <hpv_…>`` — a programmatic/curl client. Only a
        ``hpv_``-prefixed bearer is treated as a share token so it can never
        shadow the operator-token bearer path (which the gate already tried and
        rejected before reaching here).
    """
    tok = (request.args.get("share") or "").strip()
    if tok:
        return tok
    hdr = (request.headers.get("X-Video-Share") or "").strip()
    if hdr:
        return hdr
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
        if bearer.startswith("hpv_"):
            return bearer
    return ""


def _video_share_principal(request):  # noqa: A002 (mirror caller's name)
    """Return a video-scoped share principal (``share:<key_id>``) for this
    request, or None.

    Validates the presented share credential against the DEDICATED video-share
    key store (unexpired, not revoked). Because this is called ONLY from the
    /video gate below, the credential it accepts can never authorize a
    console/operator route — that property is structural (see module docstring).
    The store import is lazy so ``video_auth`` stays importable even if the
    key/store layer is unhappy, and so nothing is consulted for a request that
    carries no share credential at all.
    """
    token = _share_token(request)
    if not token:
        return None
    try:
        from hugpy_server.app.functions.imports.utils.video_share_keys import share_principal
        return share_principal(token)
    except Exception:  # noqa: BLE001 — a store hiccup must not open OR crash the gate
        logger.warning("video-share principal resolution failed", exc_info=True)
        return None


def _video_authorized() -> bool:
    """A request may use the video surface iff it has a valid MEMBER-or-operator
    session (or open-mode / operator-token access) OR a valid video-share
    credential.

    2026-08-06: the session leg moved from ``operator_authenticated()`` to
    ``member_authenticated()`` — the SAME first-party session check, now
    role-aware. The video arm is the Studio plane, which an approved member is
    entitled to; before roles existed "member" and "operator" were the same
    thing, so this is the tier split, not a widening. What a member CANNOT do
    through this gate is unchanged and structural: every console route is gated
    by operator_auth alone, which never consults anything here, and the
    per-artifact OWNER filters in video_routes scope a member to their own
    clips/jobs. The share-token leg below is byte-for-byte as it was."""
    if member_authenticated():
        return True
    if _video_share_principal(request) is not None:
        return True
    return False


from hugpy_server.app.auth_common import is_shell_request as _is_shell_request


def install_video_gate(app) -> None:
    """Register the /video auth gate on a Flask app (idempotent)."""
    if getattr(app, "_video_gate_installed", False):
        return
    app._video_gate_installed = True

    @app.before_request
    def _video_gate():
        if request.method == "OPTIONS":
            return None  # never block CORS preflight
        if not _is_video_surface():
            return None  # not our surface — leave every other route untouched
        if _video_authorized():
            return None
        # Denied. A browser navigating to the video app is sent to the console
        # root, where the ONE console login lives (same session, no second
        # login). Data/media/XHR calls get a clean 401.
        if _is_shell_request():
            return redirect("/")
        abort(401, description="Authentication required for the video surface.")

    logger.info("video auth gate installed")
