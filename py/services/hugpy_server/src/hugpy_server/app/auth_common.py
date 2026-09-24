"""Shared request-shape and strict member checks."""

from __future__ import annotations

from flask import abort, request


def caller_username() -> str | None:
    """Resolve the session username, or None for unavailable auth."""
    try:
        from hugpy_server.app.operator_auth import principal_username
        return principal_username()
    except Exception:  # noqa: BLE001 — an unavailable gate has no principal
        return None


def client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or ""


def normalized_path() -> str:
    path = request.path or "/"
    if path == "/api" or path.startswith("/api/"):
        path = path[len("/api"):] or "/"
    return path


def bearer_token() -> str | None:
    """Bearer header, or the query API key used by CLI clients."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.args.get("api_key")


def is_shell_request() -> bool:
    """Whether this request navigates to the SPA shell."""
    if request.endpoint == "_hugpy_ui":
        return True
    return request.headers.get("Sec-Fetch-Dest") == "document"


def require_member_strict() -> None:
    """Require a member or operator without an open-mode waiver."""
    try:
        from hugpy_server.app.operator_auth import member_authenticated
    except Exception:
        abort(401, description="Authentication required for this route.")
    if not member_authenticated():
        abort(401, description="Authentication required for this route.")
