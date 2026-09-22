"""Deterministic fleet allocator.

Given a task's resource NEED and a SNAPSHOT of what the fleet has free right now,
compute a PLACEMENT — whole-GPU, sharded across several GPUs, or CPU — choosing
the cheapest placement that satisfies the need.

Design rules:
  * Pure & deterministic. ``allocate(need, nodes)`` depends ONLY on its inputs —
    no clock, no randomness. Same snapshot + same need ⇒ same placement, every
    time. Every sort has a stable tiebreaker (node id) so ordering never wobbles.
  * Cheapest-sufficient first:
        1. WHOLE   — the *smallest* single GPU that fits (best packing; leaves big
                     GPUs free for jobs that actually need them).
        2. SHARD   — the *fewest* GPUs (largest-first) whose summed free VRAM fits,
                     with a VRAM-proportional tensor_split. GPU-only: layers never
                     spill to the (weak) CPUs. Used only when nothing single fits.
        3. CPU     — only if the task permits it (``need.cpu_ok``) and no GPU path
                     works. This is where CPU-viable / tiny tasks land.
        4. NONE    — nothing currently free can hold it.

This module is web-layer-agnostic: the caller builds the node snapshot from the
live worker registry and acts on the returned placement (route to a worker, or
ship rpc_servers+tensor_split as a spill override to the lead, or run on CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Node:
    """A fleet member as seen at one moment (built from the registry snapshot)."""
    id: str
    free_vram: int = 0                    # bytes free on its GPU (0 = no usable GPU)
    free_ram: int = 0                     # bytes free RAM (for the CPU tier)
    rpc_endpoint: Optional[str] = None    # "host:port" if it can be a shard BACKEND
    can_lead: bool = True                 # serves /infer (role=worker) → can be lead/whole/cpu
    online: bool = True
    # Runtime-env tier this worker's venv serves ("stable", "edge", ...). The
    # allocator itself never reads it — callers filter the snapshot to the
    # need's tier BEFORE allocate(), keeping this module env-agnostic and pure.
    env_tier: str = "stable"


@dataclass(frozen=True)
class Need:
    """What a task needs, on the target device."""
    bytes_needed: int                     # weights + KV cache, in bytes
    cpu_ok: bool = False                  # may it fall back to CPU?
    headroom: float = 1.15               # multiply need for fragmentation/KV slack


@dataclass(frozen=True)
class Placement:
    """The allocator's decision. ``kind`` ∈ {whole, shard, cpu, none}."""
    kind: str
    lead_id: Optional[str] = None         # node that loads/serves the model
    node_ids: tuple = ()                  # all nodes involved (lead first for shard)
    rpc_servers: tuple = ()               # endpoints to offload to (shard, excl. lead)
    tensor_split: tuple = ()              # VRAM-proportional, ordered [lead, *rpc]
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.kind != "none"


def _target_bytes(need: Need) -> int:
    return int(need.bytes_needed * max(1.0, need.headroom))


def allocate(need: Need, nodes: Sequence[Node]) -> Placement:
    """Deterministically place ``need`` on the currently-available ``nodes``."""
    target = _target_bytes(need)
    gpus = [n for n in nodes if n.online and n.free_vram > 0]

    # 1) WHOLE — smallest /infer-capable GPU that fits (ascending VRAM, id tiebreak).
    for n in sorted(gpus, key=lambda n: (n.free_vram, n.id)):
        if n.can_lead and n.free_vram >= target:
            return Placement(
                kind="whole", lead_id=n.id, node_ids=(n.id,),
                reason=f"whole: {n.id} has {n.free_vram} ≥ {target}",
            )

    # 2) SHARD — a /infer-capable LEAD (largest such GPU) + the fewest rpc
    #    BACKENDS (largest-first) needed to reach target. Lead serves the request
    #    and offloads to the backends' rpc-servers; tensor_split is [lead, *rpc].
    leads = sorted([n for n in gpus if n.can_lead], key=lambda n: (-n.free_vram, n.id))
    if leads:
        lead = leads[0]
        backends = sorted([n for n in gpus if n.rpc_endpoint and n.id != lead.id],
                          key=lambda n: (-n.free_vram, n.id))
        chosen: list[Node] = [lead]
        total = lead.free_vram
        for b in backends:
            if total >= target:
                break
            chosen.append(b)
            total += b.free_vram
        if total >= target and len(chosen) >= 2:
            split = tuple(round(c.free_vram / total, 4) for c in chosen)
            return Placement(
                kind="shard",
                lead_id=lead.id,
                node_ids=tuple(c.id for c in chosen),
                rpc_servers=tuple(c.rpc_endpoint for c in chosen[1:]),
                tensor_split=split,
                reason=f"shard: lead {lead.id} + {len(chosen)-1} rpc, sum {total} ≥ {target}",
            )

    # 3) CPU — only if permitted and an /infer-capable node has the RAM.
    if need.cpu_ok:
        for n in sorted([n for n in nodes if n.online and n.can_lead and n.free_ram >= target],
                        key=lambda n: (n.free_ram, n.id)):
            return Placement(
                kind="cpu", lead_id=n.id, node_ids=(n.id,),
                reason=f"cpu: {n.id} has {n.free_ram} RAM ≥ {target}",
            )

    gpu_total = sum(n.free_vram for n in gpus)
    return Placement(
        kind="none",
        reason=(f"unsatisfiable: need {target}, gpu_total {gpu_total}"
                f"{' (shard needs rpc_endpoints)' if gpu_total >= target else ''}"),
    )
