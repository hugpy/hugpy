"""Compatibility shim — the tolerance-band flex engine now lives in
``hugpy_engine.fit.flex`` (core isolation, step 1: the pure evict-to-fit
decision is an engine module so it can be imported and tested standalone,
without the worker agent or the fleet package).

Every public name is re-exported here so existing fleet / server / test
callers keep working unchanged. New code should import from
``hugpy_engine.fit`` directly.
"""
from __future__ import annotations

from hugpy_engine.fit.flex import (  # noqa: F401 — re-exports
    FlexPlan,
    PartialPlan,
    band_bounds,
    band_ceiling,
    band_floor,
    ctx_band_bounds,
    flex_priority_key,
    kv_at_ctx_pct,
    leniency_floor_pct,
    plan_explicit_offload,
    plan_flex,
    plan_partial_offload,
    _neighbour_sort_key,
)

__all__ = [
    "FlexPlan", "PartialPlan", "band_bounds", "band_ceiling", "band_floor",
    "ctx_band_bounds", "flex_priority_key", "kv_at_ctx_pct",
    "leniency_floor_pct", "plan_explicit_offload", "plan_flex",
    "plan_partial_offload",
]
