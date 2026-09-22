"""Tone — ONE conversion point (TODO-14).

Two scales exist on purpose and must never be confused:

* **unit scale** ``[0.0, 1.0]`` — the INTERNAL contract on the locked,
  content-addressed artifacts (``LockedContext.tone``, ``SegmentSpec.tone``,
  ``PerformanceGoal.tone``). Changing it would change every lock digest and
  persisted artifact, so it stays.
* **operator scale** ``[0, 10]`` — what the directive, the UI, the prompt
  compiler and the spatial/render profile speak (``spatial.StyleSpec.tone``,
  ``spatial.tone_profile``, ``prompt_compiler`` signals).

Every crossing goes through :func:`to_operator` / :func:`to_unit`. Nothing
else multiplies or divides by ten. Both refuse out-of-range input instead of
clamping, because a silently clamped 20 (a double-scaled 2.0) is precisely the
bug this module exists to prevent.
"""
from __future__ import annotations

__all__ = ["OPERATOR_MAX", "to_operator", "to_unit", "describe"]

OPERATOR_MAX = 10.0


def to_operator(tone_unit: float) -> float:
    """unit [0,1] → operator [0,10]."""
    t = float(tone_unit)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"unit-scale tone must be within [0, 1], got {t} "
                         f"(if this is already operator-scale, do not convert it)")
    return round(t * OPERATOR_MAX, 6)


def to_unit(tone_operator: float) -> float:
    """operator [0,10] → unit [0,1]."""
    t = float(tone_operator)
    if not 0.0 <= t <= OPERATOR_MAX:
        raise ValueError(f"operator-scale tone must be within [0, 10], got {t}")
    return round(t / OPERATOR_MAX, 6)


def describe(tone_operator: float) -> str:
    t = float(tone_operator)
    if t <= 1.0:
        return "photorealistic / physically based"
    if t < 4.0:
        return "cinematic, lightly stylized"
    if t <= 6.0:
        return "balanced stylization"
    if t < 9.0:
        return "graphic, stylized shading"
    return "vector / cartoon presentation"
