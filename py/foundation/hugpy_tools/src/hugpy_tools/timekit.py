"""Time/date helpers (stdlib ``datetime`` / ``time`` only).

A tiny, unopinionated slice of abstract_utilities.time_utils: the handful of
calls an agent reaches for — a UTC ISO stamp, epoch seconds, and human/ISO
conversions — without the sleep loops, format-guessing, or globals.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def now_epoch() -> float:
    """Current Unix timestamp (seconds, float)."""
    return time.time()


def epoch_to_iso(epoch: float) -> str:
    """Convert Unix seconds to a UTC ISO-8601 string."""
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


def iso_to_epoch(iso: str) -> float:
    """Parse an ISO-8601 string to Unix seconds. A trailing ``Z`` is accepted."""
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()
