"""Shared mutations for API and video-share key stores."""

from __future__ import annotations


def revoke_key(key_id: str, lock, load, save) -> bool:
    """Mark a stored key revoked while holding its store lock."""
    with lock:
        data = load()
        rec = data["keys"].get(key_id)
        if not rec:
            return False
        rec["revoked"] = True
        save(data)
    return True


def is_expired(rec: dict, now=None, *, clock) -> bool:
    """Treat absent or malformed expiry as non-expiring."""
    exp = rec.get("expires_at")
    if not exp:
        return False
    try:
        return float(exp) <= (now if now is not None else clock())
    except (TypeError, ValueError):
        return False
