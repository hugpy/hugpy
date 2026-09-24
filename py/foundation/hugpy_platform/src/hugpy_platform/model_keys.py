"""Canonical spellings used to match a model key across the fleet."""

from __future__ import annotations

from typing import Any


def model_key_forms(model_key: Any) -> set[str]:
    """Return the raw key, case variants, and its ``/`` and ``~`` tails."""
    if not model_key:
        return set()
    raw = str(model_key).strip()
    forms = {raw, raw.lower()}
    tail = raw.split("/")[-1]
    forms.add(tail)
    forms.add(tail.lower())
    if "~" in raw:
        base = raw.split("~", 1)[1]
        if base:
            forms.add(base)
            forms.add(base.lower())
    return forms
