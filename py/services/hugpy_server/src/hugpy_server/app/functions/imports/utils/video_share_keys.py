"""Video-share key store — the credential behind a `/video` SHARE LINK (k9).

WHY A SEPARATE CATEGORY (read before merging this into api_keys)
---------------------------------------------------------------
This mints a NEW category of credential: a *video-scoped* share key an operator
hands to an outside party so they can drive the video features WITHOUT a console
login. It reuses the exact storage+hash idiom of ``api_keys.py`` (a small JSON
file next to the model manifest, keys shown once and stored sha256-hashed) but
lives in its OWN store file, deliberately:

  * A video-share key must NEVER authenticate the ``/v1`` inference API or any
    console/operator route. Keeping it out of ``api_keys.json`` means
    ``api_keys.verify_api_key`` (which does not filter category) can never be
    tricked into accepting one — the isolation is structural, not conventional.
  * It is consulted ONLY by ``video_auth._video_share_principal`` (the /video
    gate's share seam). Nothing else imports this module's ``verify``/``principal``
    helpers.

So the hardening the operator asked for ("the console access for these apis
should be restricted still") is enforced by construction: a share key opens the
video surface and nothing else.

Expiry: a share key carries an ``expires_at`` epoch (default 30 days from mint;
``ttl_days <= 0`` mints a non-expiring link). ``verify`` refuses an expired or
revoked key.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from typing import Any, Optional

from hugpy_server.state import server_state_path

_LOCK = threading.Lock()
# Distinct prefix from the /v1 keys' ``hp_`` so the two are never confused on
# sight (and so an accidental cross-store paste is obviously wrong).
_KEY_PREFIX = "hpv_"
DEFAULT_TTL_DAYS = 30


def _store_path() -> str:
    return server_state_path("video_share_keys.json")


def _load() -> dict[str, Any]:
    path = _store_path()
    if not os.path.exists(path):
        return {"keys": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"keys": {}}
    data.setdefault("keys", {})
    return data


def _save(data: dict[str, Any]) -> None:
    path = _store_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Unique temp name per write (pid+token): gunicorn runs multiple processes and
    # a shared "<path>.tmp" would race between open() and os.replace() — the same
    # bug api_keys.py documents. os.replace stays the atomicity point.
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _is_expired(rec: dict[str, Any], now: Optional[float] = None) -> bool:
    exp = rec.get("expires_at")
    if not exp:
        return False  # None / 0 => non-expiring
    return float(exp) <= (now if now is not None else time.time())


def _public(rec: dict[str, Any]) -> dict[str, Any]:
    """A record minus its secret hash, plus a computed ``expired`` flag."""
    out = {k: v for k, v in rec.items() if k != "hash"}
    out["expired"] = _is_expired(rec)
    return out


def create_share_key(label: str = "", ttl_days: Optional[float] = DEFAULT_TTL_DAYS) -> dict[str, Any]:
    """Mint a video-share key. The returned dict includes the FULL key — the only
    time it is ever available; only its hash is persisted.

    ``ttl_days`` <= 0 (or None) mints a non-expiring link; otherwise the key
    expires ``ttl_days`` days after mint.
    """
    token = _KEY_PREFIX + secrets.token_hex(20)
    key_id = secrets.token_hex(8)
    now = time.time()
    try:
        ttl = float(ttl_days) if ttl_days is not None else 0.0
    except (TypeError, ValueError):
        ttl = float(DEFAULT_TTL_DAYS)
    expires_at = (now + ttl * 86400.0) if ttl > 0 else None
    record = {
        "id": key_id,
        "label": (label or "").strip() or "unnamed",
        "prefix": token[:12],
        "hash": _hash(token),
        "created_at": now,
        "expires_at": expires_at,
        "last_used": None,
        "revoked": False,
    }
    with _LOCK:
        data = _load()
        data["keys"][key_id] = record
        _save(data)
    public = _public(record)
    public["key"] = token
    return public


def list_share_keys() -> list[dict[str, Any]]:
    """Non-revoked share keys, newest first. Expired keys are KEPT (with an
    ``expired`` flag) so the operator can see and clean them up."""
    with _LOCK:
        data = _load()
    out = [_public(rec) for rec in data["keys"].values() if not rec.get("revoked")]
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out


def revoke_share_key(key_id: str) -> bool:
    with _LOCK:
        data = _load()
        rec = data["keys"].get(key_id)
        if not rec:
            return False
        rec["revoked"] = True
        _save(data)
    return True


def verify_share_key(token: Optional[str]) -> Optional[str]:
    """Return the key_id for a valid (present, not revoked, not expired) token,
    else None. Bumps ``last_used`` on a hit."""
    if not token:
        return None
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        now = time.time()
        for key_id, rec in data["keys"].items():
            if rec.get("revoked"):
                continue
            if rec.get("hash") != hashed:
                continue
            if _is_expired(rec, now):
                return None
            rec["last_used"] = now
            _save(data)
            return key_id
    return None


def share_principal(token: Optional[str]) -> Optional[str]:
    """The attribution principal for a valid share token — ``share:<key_id>`` —
    or None. This is the ONE identity a share credential ever carries."""
    key_id = verify_share_key(token)
    return f"share:{key_id}" if key_id else None
