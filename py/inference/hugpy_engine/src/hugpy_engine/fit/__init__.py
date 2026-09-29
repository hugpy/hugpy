"""``hugpy_engine.fit`` — the pure, deterministic evict-to-fit core.

Standalone: imports only ``hugpy_engine`` siblings (``eviction``, ``spill``)
and the standard library — no torch, no nvml, no fleet. See ``plan.py``.
"""
from __future__ import annotations

from hugpy_engine.fit.flex import (  # noqa: F401
    FlexPlan, PartialPlan, band_bounds, band_ceiling, band_floor,
    ctx_band_bounds, flex_priority_key, kv_at_ctx_pct, leniency_floor_pct,
    plan_explicit_offload, plan_flex, plan_partial_offload,
)
from hugpy_engine.fit.plan import evict_order, need_split, plan_fit  # noqa: F401
from hugpy_engine.fit.types import (  # noqa: F401
    FAILURE_KINDS, DeviceVram, Eviction, FitFailure, FitPlan, FitPolicy,
    FitRequest, MoeSplit, Resident, ResourceSnapshot, VramSnapshot,
)

__all__ = [
    "plan_fit", "evict_order", "need_split",
    "ResourceSnapshot", "VramSnapshot", "DeviceVram", "FitPolicy", "Resident",
    "FitRequest", "Eviction", "MoeSplit", "FitFailure", "FAILURE_KINDS", "FitPlan",
    "FlexPlan", "PartialPlan", "band_bounds", "band_ceiling", "band_floor",
    "ctx_band_bounds", "flex_priority_key", "kv_at_ctx_pct", "leniency_floor_pct",
    "plan_explicit_offload", "plan_flex", "plan_partial_offload",
]
