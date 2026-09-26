"""Per-GPU-DEVICE placement within ONE box (a multi-GPU worker).

The fleet allocator (``allocator.py``) places a model onto a NODE (a box) using
that box's SUMMED free VRAM. That is right for cross-node RPC sharding and for a
single-GPU box, but on a multi-GPU box the box-sum hides a hard physical fact:

  * a diffusers / comfy / transformers PIPELINE lives on ONE device, so a 30 GiB
    model does NOT fit a 4x24 GiB box even though the box reports 96 GiB free;
  * a llama.cpp GGUF genuinely CAN tensor-split a single model across the box's
    cards, so the same 30 GiB model DOES fit — split across the four 24 GiB
    devices.

This module makes that per-device decision PURELY (stdlib only), so central, the
worker, and the tests all share ONE vocabulary without import weight. It answers
two questions for one box:

  1. ``whole_pipeline_gpu_capacity(devices, splittable)`` — the GPU-side capacity
     a WHOLE model may actually claim on this box: the SUM across devices for a
     splittable engine, the LARGEST SINGLE device for a non-splittable one. This
     is the number central feeds the engine-aware fit / feasibility / admission
     predicates instead of the naive box-sum.

  2. ``plan_device_placement(need_bytes, devices, *, splittable)`` — WHICH device
     (or devices) the box should load it on:

        single  — the SMALLEST single device whose free VRAM fits (best-fit
                  packing: leave the big cards free for jobs that need them; the
                  device INDEX is the stable tiebreak). This is the allocator's
                  own WHOLE rule, applied per device instead of per box.
        split   — SPLITTABLE engines ONLY: the FEWEST devices (largest-first)
                  whose summed free VRAM fits, with a VRAM-proportional
                  ``tensor_split`` and ``main_gpu`` = the first (largest) device.
        none    — nothing on this box can hold it on the GPU (the caller falls to
                  RAM / CPU per the existing derivation, or to another box).

Design rules mirror ``allocator.py``: pure & deterministic (same inputs => same
plan, every sort has an index tiebreak), and BYTE-IDENTICAL to the old box-sum
behavior on a single-GPU box (one device => single-device fit == box fit, and
capacity == that device's total, which == the box sum).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class GpuDevice:
    """One physical GPU on a box, as reported in a worker's ``gpus[]`` heartbeat
    entry (``{index, memory_free, memory_total}`` -> here). ``free``/``total`` are
    bytes; ``None`` means the box did not report that figure (a fresh worker
    before its first probe) and the device sorts as unmeasured."""
    index: int
    free: Optional[int] = None
    total: Optional[int] = None

    @property
    def free_b(self) -> int:
        try:
            return int(self.free) if self.free is not None else 0
        except (TypeError, ValueError):
            return 0

    @property
    def total_b(self) -> int:
        try:
            return int(self.total) if self.total is not None else 0
        except (TypeError, ValueError):
            return 0


@dataclass(frozen=True)
class DevicePlan:
    """The per-device decision for one box. ``kind`` in {single, split, none}.

    ``devices`` are the chosen device INDICES (single: one; split: largest-first).
    ``main_gpu`` is the primary device index (the first chosen), the value the
    worker sets as ``HUGPY_MAIN_GPU`` / a slot child's ``CUDA_VISIBLE_DEVICES``.
    ``tensor_split`` is the VRAM-proportional split ordered to MATCH ``devices``
    (only for a split plan; empty otherwise). ``reason`` is one honest line naming
    the numbers the decision saw (for routing diagnostics)."""
    kind: str
    devices: tuple = ()
    main_gpu: Optional[int] = None
    tensor_split: tuple = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.kind != "none"


def _measured(devices: Sequence[GpuDevice]) -> List[GpuDevice]:
    """The devices with a positive TOTAL — the ones central actually measured.
    An unmeasured device (no total reported) contributes nothing to a fit
    decision, exactly like ``allocator`` ignores a node with no free VRAM."""
    return [d for d in devices if d.total_b > 0]


def whole_pipeline_gpu_capacity(devices: Sequence[GpuDevice],
                                splittable: bool,
                                *, use: str = "total") -> Optional[int]:
    """The GPU-side capacity a WHOLE model may claim on this box, in bytes.

    ``use='total'`` (physical capacity, for the fit/feasibility gates) or
    ``use='free'`` (currently-free VRAM, for admission).

      * splittable engine (llama.cpp GGUF) -> the SUM across devices (the model
        tensor-splits across the box's cards, so the box sum is the real ceiling
        — unchanged from today).
      * non-splittable engine (diffusers / transformers / comfy) -> the LARGEST
        SINGLE device (a pipeline lives on ONE card, so the biggest card is the
        real ceiling; the box sum would over-promise a multi-GPU box).

    Returns ``None`` only when NO device was measured (true first contact) so the
    caller degrades exactly as it did on a missing box total — never invents a 0.

    On a single-GPU box max-single == sum == that device's figure, so this is
    byte-identical to the old box-sum path."""
    meas = _measured(devices)
    if not meas:
        return None
    pick = (lambda d: d.total_b) if use == "total" else (lambda d: d.free_b)
    if splittable:
        return int(sum(pick(d) for d in meas))
    return int(max(pick(d) for d in meas))


def _target_bytes(need_bytes: int, headroom: float) -> int:
    try:
        return int(int(need_bytes) * max(1.0, float(headroom)))
    except (TypeError, ValueError):
        return int(need_bytes or 0)


def plan_device_placement(need_bytes: int,
                          devices: Sequence[GpuDevice],
                          *, splittable: bool,
                          headroom: float = 1.15) -> DevicePlan:
    """Decide which local GPU device(s) hold a model needing ``need_bytes``.

    Deterministic. Best-fit single device first (smallest that fits, index
    tiebreak); a multi-device tensor-split ONLY for a splittable engine when no
    single device fits; else ``none`` (the box cannot hold it on the GPU).

    On a single-GPU box this returns a ``single`` plan on device ``index`` exactly
    when the box would have fit under the old box-sum test — behavior preserved."""
    target = _target_bytes(need_bytes, headroom)
    meas = _measured(devices)
    if not meas:
        return DevicePlan(kind="none",
                          reason="no GPU device measured on this box")

    # 1) SINGLE — smallest device whose FREE VRAM fits (best-fit; index tiebreak).
    for d in sorted(meas, key=lambda d: (d.free_b, d.index)):
        if d.free_b >= target:
            return DevicePlan(
                kind="single", devices=(d.index,), main_gpu=d.index,
                reason=(f"single: device {d.index} free {d.free_b} "
                        f">= need {target}"))

    # 2) SPLIT — splittable engines only: fewest devices (largest-first) whose
    #    summed free fits, VRAM-proportional tensor_split, main_gpu = the first.
    if splittable:
        ordered = sorted(meas, key=lambda d: (-d.free_b, d.index))
        chosen: List[GpuDevice] = []
        total = 0
        for d in ordered:
            if total >= target:
                break
            chosen.append(d)
            total += d.free_b
        if total >= target and len(chosen) >= 2:
            split = tuple(round(d.free_b / total, 4) for d in chosen)
            idx = tuple(d.index for d in chosen)
            return DevicePlan(
                kind="split", devices=idx, main_gpu=idx[0],
                tensor_split=split,
                reason=(f"split: devices {list(idx)} summed free {total} "
                        f">= need {target}"))

    # 3) NONE — nothing on the GPU. Caller derives RAM/CPU or another box.
    biggest = max(d.free_b for d in meas)
    box_free = sum(d.free_b for d in meas)
    why = (f"no single device fits (need {target}, largest free {biggest}"
           + ("" if splittable else "; non-splittable engine — box sum "
              f"{box_free} cannot be pooled for one pipeline") + ")")
    return DevicePlan(kind="none", reason=why)


def devices_from_gpus(gpus) -> List[GpuDevice]:
    """Build ``GpuDevice`` list from a worker's ``gpus[]`` heartbeat entries
    (``[{index, memory_free, memory_total}, ...]``). Tolerant of missing keys and
    non-dict junk — an entry with no index falls back to its position so the plan
    still has a stable device number to hand the worker."""
    out: List[GpuDevice] = []
    for pos, g in enumerate(gpus or []):
        if not isinstance(g, dict):
            continue
        idx = g.get("index")
        try:
            idx = int(idx) if idx is not None else pos
        except (TypeError, ValueError):
            idx = pos
        out.append(GpuDevice(index=idx,
                             free=g.get("memory_free"),
                             total=g.get("memory_total")))
    return out


__all__ = ["GpuDevice", "DevicePlan", "whole_pipeline_gpu_capacity",
           "plan_device_placement", "devices_from_gpus"]
