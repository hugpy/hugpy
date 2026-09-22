"""Resource-fit and allocation decisions."""

from __future__ import annotations

from typing import Any, Optional

from .backend import get_backend


def feasible_allocations(
    engine: Any,
    model_bytes: Optional[int],
    gpu_total_bytes: Optional[int],
    ram_total_bytes: Optional[int],
    *,
    moe_split_gpu_bytes: Optional[int] = None,
    bnb: bool = False,
) -> tuple[str, ...]:
    return get_backend().feasible_allocations(
        engine,
        model_bytes,
        gpu_total_bytes,
        ram_total_bytes,
        moe_split_gpu_bytes=moe_split_gpu_bytes,
        bnb=bnb,
    )


def choose_allocation(
    engine: Any,
    model_bytes: Optional[int],
    gpu_total_bytes: Optional[int],
    ram_total_bytes: Optional[int],
    *,
    moe: Optional[dict] = None,
    bnb: bool = False,
    moe_force: Optional[bool] = None,
) -> dict:
    return get_backend().choose_allocation(
        engine,
        model_bytes,
        gpu_total_bytes,
        ram_total_bytes,
        moe=moe,
        bnb=bnb,
        moe_force=moe_force,
    )


def allocation_spill(mode: Any, **options: Any) -> dict:
    return get_backend().allocation_spill(mode, **options)


def resource_status() -> dict:
    return get_backend().resource_status()


def plan_admission(*args: Any, **kwargs: Any):
    return get_backend().plan_admission(*args, **kwargs)


__all__ = [
    "allocation_spill",
    "choose_allocation",
    "feasible_allocations",
    "plan_admission",
    "resource_status",
]
