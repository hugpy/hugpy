"""Shared input checks for production and audio master plans."""

from __future__ import annotations

from typing import Any


def require_text(value: Any, what: str) -> str:
    text = str(value or "")
    if not text.strip():
        raise ValueError(f"{what} must be non-empty")
    return text


def require_non_negative(value: Any, what: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{what} must be non-negative, got {number}")
    return number
