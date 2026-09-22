"""API-key store for the public /v1 inference API.

Keys are created from the UI, shown to the user exactly once, and stored
hashed (sha256) — the store can verify a presented key but never reveal it
again. A single `require_key` flag decides whether /v1 calls must present a
key at all; with the flag off the API is open and keys are optional.

Storage is a small JSON file next to the model manifest
(PROJECTS_HOME/api_keys.json) so it lives under the storage root with
everything else and survives restarts. Guarded by a process-wide lock;
gunicorn runs a single worker here, matching how job_store is handled.
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
_KEY_PREFIX = "hp_"

# ── scopes (2026-07-23, secure install links) ──────────────────────────────
# A key may be SCOPED to a subset of the API surface. Vocabulary (small on
# purpose — grow it only when a new gated surface exists):
#   "v1"             the /v1 inference surface (chat/completions, models)
#   "ml"             the media-intelligence /ml surface
#   "agent-register" the /agent/register fleet bootstrap
#   "full"           everything (the pre-scope behavior)
# LEGACY rows have no `scopes` field: they read as ["full"] so nothing that
# exists today changes behavior (lazy/additive migration — the store file is
# never rewritten wholesale; a row gains fields only when it is next written).
SCOPES = ("v1", "ml", "agent-register", "full")
_DEFAULT_SCOPES = ["full"]


def _scopes_of(rec: dict[str, Any]) -> list[str]:
    """A record's effective scopes — legacy rows (no field / empty) = full."""
    scopes = rec.get("scopes")
    if not isinstance(scopes, list) or not scopes:
        return list(_DEFAULT_SCOPES)
    return [str(s) for s in scopes]


def _scope_ok(rec: dict[str, Any], required_scope: Optional[str]) -> bool:
    if not required_scope:
        return True
    scopes = _scopes_of(rec)
    return "full" in scopes or required_scope in scopes


def _is_expired(rec: dict[str, Any], now: Optional[float] = None) -> bool:
    """Key expiry — legacy rows have no `expires_at` and never expire."""
    exp = rec.get("expires_at")
    if not exp:
        return False
    try:
        return float(exp) <= (now if now is not None else time.time())
    except (TypeError, ValueError):
        return False


def _store_path() -> str:
    return server_state_path("api_keys.json")


def _load() -> dict[str, Any]:
    path = _store_path()
    if not os.path.exists(path):
        return {"require_key": False, "keys": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"require_key": False, "keys": {}}
    data.setdefault("require_key", False)
    data.setdefault("keys", {})
    return data


def _save(data: dict[str, Any]) -> None:
    path = _store_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Temp name must be unique PER WRITE: gunicorn runs several processes, and
    # concurrent /v1 auths all bump last_used — two writers sharing one
    # "<path>.tmp" race between open() and os.replace(), and the loser's
    # replace() dies FileNotFoundError → a raw 500 at AUTH time (bit a
    # concurrent batch 2026-07-11). pid+token keeps the write atomic AND
    # collision-free; os.replace stays the atomicity point.
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_api_key(name: str = "", pool: str = "",
                   label: str = "",
                   scopes: Optional[list[str]] = None,
                   created_by: str = "operator",
                   expires_at: Optional[float] = None) -> dict[str, Any]:
    """Mint a key. The returned dict includes the FULL key — the only time
    it is ever available; persist only its hash.

    ``pool`` binds the key to a dedicated worker pool: requests authenticated
    with it route to that pool by default (the app needs no per-request flag).

    2026-07-23 scope additions (all optional — the legacy 2-arg call mints
    exactly what it always did, a full-scope operator key):
      * ``label``      free-form operator description (distinct from ``name``,
                       which existing UI uses as the short handle).
      * ``scopes``     subset of ``SCOPES``; omitted/empty => ["full"].
                       Unknown scope strings are rejected (ValueError) — a typo
                       must never silently mint a broader or dead key.
      * ``created_by`` "operator" (console mint) | "install-link" (a key baked
                       into a one-time installer download).
      * ``expires_at`` epoch seconds; None => never expires.
    """
    if scopes is None or not scopes:
        scopes = list(_DEFAULT_SCOPES)
    scopes = [str(s).strip() for s in scopes if str(s).strip()]
    bad = [s for s in scopes if s not in SCOPES]
    if bad:
        raise ValueError(f"unknown scope(s): {bad}; valid: {list(SCOPES)}")
    token = _KEY_PREFIX + secrets.token_hex(20)
    key_id = secrets.token_hex(8)
    record = {
        "id": key_id,
        "name": (name or "").strip() or "unnamed",
        "label": (label or "").strip(),
        "pool": (pool or "").strip(),
        "scopes": scopes,
        "created_by": (created_by or "operator").strip() or "operator",
        "prefix": token[:11],
        "hash": _hash(token),
        "created_at": time.time(),
        "expires_at": float(expires_at) if expires_at else None,
        "disabled": False,
        "last_used": None,
        "revoked": False,
    }
    with _LOCK:
        data = _load()
        data["keys"][key_id] = record
        _save(data)
    public = {k: v for k, v in record.items() if k != "hash"}
    public["key"] = token
    return public


def list_api_keys() -> list[dict[str, Any]]:
    with _LOCK:
        data = _load()
    out = []
    for rec in data["keys"].values():
        if rec.get("revoked"):
            continue
        row = {k: v for k, v in rec.items() if k != "hash"}
        # Lazy-migration read view: legacy rows (pre-2026-07-23) surface the
        # defaults without the store file ever being rewritten.
        row.setdefault("label", "")
        row["scopes"] = _scopes_of(rec)
        row.setdefault("created_by", "operator")
        row.setdefault("expires_at", None)
        row.setdefault("disabled", False)
        row["expired"] = _is_expired(rec)
        out.append(row)
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out


def revoke_api_key(key_id: str) -> bool:
    with _LOCK:
        data = _load()
        rec = data["keys"].get(key_id)
        if not rec:
            return False
        rec["revoked"] = True
        _save(data)
    return True


def prune_api_keys(*, expired: bool = True,
                   older_than_days: Optional[float] = None,
                   unused_only: bool = False) -> list[dict[str, Any]]:
    """Bulk-revoke 'dated' keys and return what was pruned (id/name/prefix/why).

    A key is pruned when it matches ANY enabled criterion:
      * ``expired``          its ``expires_at`` has passed (default on).
      * ``older_than_days``  minted more than N days ago (None => off).
    ``unused_only`` narrows an ``older_than_days`` sweep to keys that have NEVER
    been used (``last_used`` is None) — so a stale-but-active integration is
    never swept out from under a caller by age alone. It does NOT gate the
    ``expired`` criterion: an expired key already refuses calls, so pruning it
    is safe regardless of prior use.

    Already-revoked rows are skipped (idempotent — pruning twice is a no-op).
    """
    now = time.time()
    cutoff = (now - older_than_days * 86400) if older_than_days is not None else None
    pruned: list[dict[str, Any]] = []
    with _LOCK:
        data = _load()
        for rec in data["keys"].values():
            if rec.get("revoked"):
                continue
            reason = None
            if expired and _is_expired(rec, now):
                reason = "expired"
            elif cutoff is not None and (rec.get("created_at") or 0) < cutoff:
                if not (unused_only and rec.get("last_used")):
                    reason = f"older than {older_than_days:g}d"
            if reason:
                rec["revoked"] = True
                pruned.append({"id": rec.get("id"), "name": rec.get("name"),
                               "prefix": rec.get("prefix"), "reason": reason})
        if pruned:
            _save(data)
    return pruned


def verify_api_key(token: Optional[str],
                   required_scope: Optional[str] = None) -> bool:
    """Validate a presented key on its own merits (hash + revocation, and —
    2026-07-23 — expiry/disabled when those fields are present on the row).

    ``required_scope`` (optional): when passed, the key must ALSO carry that
    scope or "full". When omitted, behavior is exactly the pre-scope contract —
    every existing call site is unchanged, and legacy rows (no ``scopes``
    field) read as ["full"] so they pass any scope check.
    """
    if not token:
        return False
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        for rec in data["keys"].values():
            if rec.get("revoked"):
                continue
            if rec.get("hash") == hashed:
                if rec.get("disabled"):
                    return False
                if _is_expired(rec):
                    return False
                if not _scope_ok(rec, required_scope):
                    return False
                rec["last_used"] = time.time()
                _save(data)
                return True
    return False


def pool_for_key(token: Optional[str]) -> Optional[str]:
    """The dedicated pool bound to this key (or None if no key / no pool / revoked).
    Used to route an app's traffic to its reserved workers from its key alone."""
    if not token:
        return None
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        for rec in data["keys"].values():
            if rec.get("revoked"):
                continue
            if rec.get("hash") == hashed:
                p = (rec.get("pool") or "").strip()
                return p or None
    return None


def key_id_for_token(token: Optional[str]) -> Optional[str]:
    """The key record id behind a plaintext token (or None). Lets the
    principal layer (comms.principals) attribute a request to a stable key
    identity without ever storing the plaintext."""
    if not token:
        return None
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        for key_id, rec in data["keys"].items():
            if rec.get("revoked"):
                continue
            if rec.get("hash") == hashed:
                return key_id
    return None


def key_name_for_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    hashed = _hash(token.strip())
    with _LOCK:
        data = _load()
        for rec in data["keys"].values():
            if rec.get("revoked"):
                continue
            if rec.get("hash") == hashed:
                return rec.get("name") or None
    return None


def api_key_required() -> bool:
    with _LOCK:
        return bool(_load().get("require_key"))


def set_api_key_required(required: bool) -> bool:
    with _LOCK:
        data = _load()
        data["require_key"] = bool(required)
        _save(data)
    return bool(required)


# Media-intelligence access gate — a SEPARATE flag from `require_key` (the /v1
# gate) and from the console's own login. When on, the media-intelligence /ml/*
# inference endpoints require a valid Bearer key (the same keys minted above),
# letting the operator flip media between "open within the console" and
# "key-gated" without touching the console auth.
def media_key_required() -> bool:
    with _LOCK:
        return bool(_load().get("media_require_key"))


def set_media_key_required(required: bool) -> bool:
    with _LOCK:
        data = _load()
        data["media_require_key"] = bool(required)
        _save(data)
    return bool(required)
