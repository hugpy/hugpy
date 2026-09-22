"""Same-origin auth proxy (Backend-for-Frontend).

WHY THIS EXISTS
---------------
In ``external`` auth mode the console UI (served from e.g. https://hugpy.ai)
logs the operator in against a SEPARATE auth service (HUGPY_AUTH_BASE, e.g.
https://api.abstractendeavors.com). That service sets a *session cookie*. When
the UI then calls it directly, that cookie is **third-party** relative to the
hugpy origin — and Safari (ITP) + Firefox (Total Cookie Protection) block
third-party cookies unconditionally, so the cookie never comes back on
``GET /me`` and the user is bounced straight back to the login screen. Chrome is
the lenient outlier, which is why "it works in Chrome only".

No cookie attribute fixes this (``SameSite=None; Secure`` does NOT help — Safari
blocks third-party cookies regardless). The robust, browser-agnostic fix is to
make the cookie **first-party**: proxy the auth calls through the hugpy origin
and rewrite ``Set-Cookie`` so it binds to the hugpy host instead of the auth
service's domain.

HOW
---
The UI is told (via ``GET /api/auth/config``) to use a same-origin base
(``/api/auth-svc`` by default), so it calls e.g. ``/api/auth-svc/login`` and
``/api/auth-svc/me`` on its OWN origin. The public nginx strips the ``/api``
prefix (and the standalone ApiPrefixMiddleware does the same), so those land
here as ``/auth-svc/login`` etc. We forward them to the upstream auth service,
forwarding the request cookies up and rewriting any ``Set-Cookie`` coming back
to be first-party (drop ``Domain``, force ``Secure; SameSite=Lax``). To the
browser the session cookie is now first-party on hugpy.ai — accepted by every
browser.

The hub/auth service is unchanged; the UI's auth model (cookie-based,
``credentials: 'include'``) is unchanged. Only the *path* the cookie takes
changes. Disable with ``HUGPY_AUTH_PROXY=0`` (then auth/config advertises the
upstream directly, the legacy behaviour).
"""
import os

import requests
from flask import Blueprint, Response, request

auth_proxy_bp = Blueprint("auth_proxy_bp", __name__)

# The path the UI calls on its own origin. Kept under /api so the existing nginx
# /api proxy rule carries it to the backend with no nginx change required.
PUBLIC_BASE = "/api/auth-svc"
# The mount point AFTER the /api prefix is stripped (nginx / ApiPrefixMiddleware).
_MOUNT = "/auth-svc"

# Hop-by-hop / connection-management headers we must not forward verbatim.
_DROP_REQ = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "te", "trailer", "upgrade", "accept-encoding",
}
_DROP_RESP = {
    "content-encoding", "content-length", "transfer-encoding", "connection",
    "keep-alive", "set-cookie",
}


def upstream_base() -> str:
    """Real auth service the proxy forwards to."""
    base = (
        os.environ.get("HUGPY_AUTH_UPSTREAM")
        or os.environ.get("HUGPY_AUTH_BASE")
        or "https://api.abstractendeavors.com"
    )
    return base.rstrip("/")


def proxy_enabled() -> bool:
    val = str(os.environ.get("HUGPY_AUTH_PROXY", "1")).strip().lower()
    return val not in ("0", "false", "no", "off", "")


def public_base() -> str:
    return os.environ.get("HUGPY_AUTH_PUBLIC_BASE") or PUBLIC_BASE


def rewrite_set_cookie(raw: str) -> str:
    """Make an upstream Set-Cookie first-party to the hugpy origin.

    Drops ``Domain=...`` (so the cookie binds to the host that served the
    response — us), and forces ``Secure`` + ``SameSite=Lax`` (now that it is a
    same-site first-party cookie, Lax is both sufficient and the most
    compatible). Everything else (name=value, Path, Expires/Max-Age, HttpOnly)
    is preserved.
    """
    parts = [p.strip() for p in raw.split(";")]
    name_value = parts[0]
    kept = []
    has_secure = False
    has_samesite = False
    for attr in parts[1:]:
        low = attr.lower()
        if low.startswith("domain="):
            continue  # first-party: bind to our host, not the auth service's
        if low == "secure":
            has_secure = True
        if low.startswith("samesite="):
            has_samesite = True
            attr = "SameSite=Lax"
        kept.append(attr)
    if not has_samesite:
        kept.append("SameSite=Lax")
    if not has_secure:
        kept.append("Secure")
    return "; ".join([name_value] + kept)


def _iter_upstream_set_cookies(resp: requests.Response):
    """All Set-Cookie headers, preserving attributes (urllib3 keeps them split)."""
    try:
        return list(resp.raw.headers.getlist("Set-Cookie"))
    except Exception:
        single = resp.headers.get("Set-Cookie")
        return [single] if single else []


@auth_proxy_bp.route(
    f"{_MOUNT}/<path:subpath>",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
def auth_proxy(subpath):
    if not proxy_enabled():
        return Response('{"error":"auth proxy disabled"}', status=404,
                        mimetype="application/json")

    url = f"{upstream_base()}/{subpath}"
    fwd_headers = {k: v for k, v in request.headers if k.lower() not in _DROP_REQ}

    try:
        upstream = requests.request(
            method=request.method,
            url=url,
            params=request.args,
            data=request.get_data(),
            headers=fwd_headers,
            cookies=request.cookies,        # forward the session up to its issuer
            allow_redirects=False,
            timeout=20,
        )
    except requests.RequestException:
        return Response('{"error":"auth service unreachable"}', status=502,
                        mimetype="application/json")

    resp = Response(upstream.content, status=upstream.status_code)
    for key, val in upstream.headers.items():
        low = key.lower()
        if low in _DROP_RESP or low.startswith("access-control-"):
            continue  # same-origin now: CORS headers are noise
        resp.headers[key] = val
    for raw_cookie in _iter_upstream_set_cookies(upstream):
        if raw_cookie:
            resp.headers.add("Set-Cookie", rewrite_set_cookie(raw_cookie))
    return resp
