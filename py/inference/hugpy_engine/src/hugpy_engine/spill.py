"""GPU/CPU spill configuration — how much of a model lives on the GPU.

Two backends, two knobs:

  * llama.cpp (GGUF): ``n_gpu_layers`` passed to ``Llama(...)``.
        -1 = every layer on GPU, 0 = pure CPU, N = first N transformer layers
        on GPU (the rest spill to CPU/RAM). This is the ONLY thing that lights
        up the GPU for GGUF models — without it llama.cpp runs CPU-only.
  * transformers: ``max_memory`` passed to ``from_pretrained(device_map="auto")``,
        e.g. ``{0: "7GiB", "cpu": "32GiB"}`` — accelerate then shards layers to
        fit the per-device budget, spilling the overflow to CPU.

Config comes from environment variables so the worker agent (which owns the
process) can set it per model load without threading new fields through the
resolver/dispatch chain:

    HUGPY_N_GPU_LAYERS   "auto" | "off" | int   llama.cpp layers on GPU
    HUGPY_TENSOR_SPLIT   csv floats             multi-GPU split e.g. "0.7,0.3"
    HUGPY_MAIN_GPU       int                    primary GPU index (llama.cpp)
    HUGPY_GPU_MEM_GIB    float                  per-GPU budget (transformers)
    HUGPY_CPU_MEM_GIB    float                  CPU/RAM budget (transformers)
    HUGPY_N_GPU          int                    #GPUs to spread across

Everything is optional. Default mode is "auto": detect free VRAM, estimate the
model's size, and fit as many layers as will hold — falling back to CPU (0
layers) when no GPU is visible, so a CPU-only host behaves exactly as before.
"""
from __future__ import annotations

import os
import sys
import logging
from typing import Any, Optional

logger = logging.getLogger("abstract_hugpy_dev.spill")

# VRAM headroom multiplier, applied on top of the EXPLICIT ctx reserve below.
#
# LIFTED 0.85 -> 1.0 (operator, 2026-07-25). It was double-counting: its own
# comment said it existed for "the KV-cache + activations", which is precisely
# what ``HUGPY_VRAM_CTX_RESERVE_GIB`` reserves — and that reserve is ACCURATE.
# Measured on computron (flux2-klein-9b q4_k_m, all 36 layers, ctx 16384):
# 7441 MiB on the card, 4.68 GiB of weights, so 2.59 GiB of KV+compute+ctx
# against a 2.5 GiB reserve. The safety factor is the older, cruder margin for
# the same risk; the reserve superseded it and nobody removed the multiplier.
#
# What it cost, and why the operator called it: quant sizes are chosen against
# REAL card capacities, so a 23.5 GiB release IS the "fits a 24 GiB card" build.
# Stacking 1 GiB reserve + 15% + 2.5 GiB ctx left a 24 GiB 3090 with a 17.05 GiB
# budget — 29% of the card gone, and the model refused on the hardware it was
# published for. Operator: "i've accepted lack of safety simply due to the
# 23.5 GB transformers that obviously are for a 3090".
#
# STILL IN FORCE, so this is not "no safety": the ctx reserve (COMPUTED from the
# model's real geometry at the real n_ctx since 2026-07-27 — see
# vram_ctx_reserve_bytes; it was a flat 2.5 GiB when this note was written),
# HUGPY_VRAM_RESERVE_GIB (1.0 GiB, for consumers central cannot see), and the
# admission ceiling (agent._vram_ceiling_reserve_bytes — a bounded compute
# cushion since 2026-07-27, HUGPY_VRAM_CEILING_FRAC to override) which is the
# real OOM backstop. This removes a redundant fourth margin, not the floor.
# Tunable if it bites: set HUGPY_VRAM_SAFETY below 1.0 (read at CALL time by
# _vram_safety(), not here — _env_float is defined further down this module and
# a module-level call would NameError on import, taking every worker with it).
_VRAM_SAFETY = 1.0
# When we can't read a GGUF's real layer count, assume a 7B-ish 32-layer model.
_ASSUMED_LAYERS = 32


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _env_float(name: str) -> Optional[float]:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return None


def _env_int(name: str) -> Optional[int]:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", name, raw)
        return None


def _vram_safety() -> float:
    """The VRAM headroom multiplier, read at CALL time (see _VRAM_SAFETY).

    Default 1.0 — the explicit ctx reserve carries this job now. Env override
    HUGPY_VRAM_SAFETY for a box that wants the old cushion back; clamped to
    (0, 1] so a typo can only ever be MORE conservative, never budget past the
    card.
    """
    v = _env_float("HUGPY_VRAM_SAFETY")
    if v is None:
        return _VRAM_SAFETY
    if not (0 < v <= 1):
        logger.warning("ignoring HUGPY_VRAM_SAFETY=%r (want 0 < x <= 1); "
                       "using %s", v, _VRAM_SAFETY)
        return _VRAM_SAFETY
    return v


# ---------------------------------------------------------------------------
# hardware probes (best-effort, never raise)
# ---------------------------------------------------------------------------
def vram_reserve_bytes() -> int:
    """VRAM kept out of every budget (HUGPY_VRAM_RESERVE_GIB, default 1.0).

    The box may have GPU consumers central knows nothing about (a desktop
    session, another app). Reserving a slice at the probe layer means autofit,
    preflights, and heartbeat-fed budgets all leave it alone, while the raw
    per-GPU numbers shown in the console stay truthful."""
    gib = _env_float("HUGPY_VRAM_RESERVE_GIB")
    return int((1.0 if gib is None else gib) * 2**30)


def ram_reserve_bytes() -> int:
    """RAM kept out of every budget (HUGPY_RAM_RESERVE_GIB, default 4.0).

    Same idea as the VRAM reserve: local processes central can't see need
    room, and a load that consumes MemAvailable to the floor gets the whole
    agent OOM-killed mid-request."""
    gib = _env_float("HUGPY_RAM_RESERVE_GIB")
    return int((4.0 if gib is None else gib) * 2**30)


# ---------------------------------------------------------------------------
# Honest budget-bar semantics (t13/t14, operator spec 2026-07-17, REFINED)
# ---------------------------------------------------------------------------
# The console resource bars used to draw a numerator and denominator from
# different universes (physical-derived "used" vs central-limit "total"), so on
# any under-budget box the bar collapsed to physical_total − central_limit — an
# ARTIFACT, not usage. The operator specified the honest model, adopted as
# doctrine. It is IDENTICAL for RAM and VRAM, so it lives here once and is used
# by BOTH the central summary (the bar) and the worker allocator (budgetable
# free) — bar and admission can then never disagree.
#
# Final formula (both clamps mandatory — operator refinement 2026-07-17,
# "the limit can never lead into negative, but the limit should not be
#  encroached by ram unless it exceeds the difference in worker process"):
#
#   external_headroom = physical_total − central_limit    # never the worker's
#   encroachment      = max(0, external_usage − external_headroom)
#   bar_used          = min(central_limit, worker_usage + encroachment)  # ≤ limit
#   remaining         = max(0, central_limit − worker_usage − encroachment)  # ≥ 0
#
# The central_limit is the WORKER'S budget. External consumers (a desktop
# session, ComfyUI, another app) first spend their OWN headroom (physical above
# the limit); only what they use BEYOND that headroom encroaches on the worker's
# budget. Worked example (operator): 128 physical / 90 limit / 20 worker / 10
# external → headroom 38, encroachment 0, bar 20/90, 70 to go. External grows to
# 50 → encroachment 12 → bar 32/90, 58 to go.
#
# OVER-LIMIT HONESTY (operator, 2026-07-17): the CLAMPS are a RENDER/admission
# rule — the display never goes negative and the fill never overflows — but a
# genuine overrun (raw worker_usage + encroachment > central_limit) is never
# hidden behind a clean full bar. So the payload carries BOTH the clamped
# figures (bar_used/remaining, for the fill + admission) AND the RAW ones
# (raw_used, over_limit, over_by) so central/console can pin the chip at 100%
# and surface an explicit over-limit warning. The allocator floors remaining at
# 0 the same way: an over-limit box admits nothing new until it drains.
def budget_bar(physical_total: Optional[int],
               central_limit: Optional[int],
               worker_usage: Optional[int],
               external_usage: Optional[int]) -> dict:
    """Compute the honest bar (t13/t14 spec, refined) from the four measured
    inputs.

    All arguments are bytes (or None where unmeasured). Returns a dict:
      * ``semantics="central"`` when a central_limit is set: bar_used/remaining
        follow the clamped spec; ``total`` is the limit; ``raw_used`` is the
        UNCLAMPED worker+encroachment; ``over_limit`` / ``over_by`` flag a true
        overrun.
      * ``semantics="physical"`` when NO central_limit is set: headroom is
        undefined, so the bar shows plain measured usage (worker+external)
        against the physical total; no encroachment, never over-limit.
    ``bar_used``/``remaining`` are None only when the necessary inputs are
    missing (never fabricated)."""
    w = worker_usage if worker_usage is not None else None
    x = external_usage if external_usage is not None else None
    # No central limit -> physical-total semantics (plain measured usage).
    if not central_limit or central_limit <= 0:
        parts = [v for v in (w, x) if v is not None]
        bar_used = sum(parts) if parts else None
        total = physical_total
        remaining = (max(0, total - bar_used)
                     if (total is not None and bar_used is not None) else None)
        return {"semantics": "physical", "total": total,
                "bar_used": bar_used, "remaining": remaining,
                "raw_used": bar_used, "over_limit": False, "over_by": 0,
                "encroachment": 0, "worker_usage": w, "external_usage": x,
                "external_headroom": None}
    # Central-limit semantics (the spec).
    headroom = None
    if physical_total is not None:
        headroom = max(0, physical_total - central_limit)
    encroachment = 0
    if x is not None and headroom is not None:
        encroachment = max(0, x - headroom)
    elif x is not None and headroom is None:
        # No physical read to derive headroom from — the safe, non-fabricating
        # choice is to treat all external usage as encroachment (the limit is
        # the only denominator we trust). Rare: a box that reports a limit but
        # no physical total.
        encroachment = x
    # RAW (unclamped) worker+encroachment — the truth central/console must keep.
    raw_used = None
    if w is not None:
        raw_used = w + encroachment
    elif encroachment:
        raw_used = encroachment
    # CLAMPED fill: never overflows the limit (the ≤ clamp).
    bar_used = min(central_limit, raw_used) if raw_used is not None else None
    # CLAMPED remaining: never negative (the ≥0 clamp). Derived from the raw
    # worker+encroachment, floored at 0 — the figure admission also uses.
    remaining = (max(0, central_limit - raw_used)
                 if raw_used is not None else None)
    over_by = (max(0, raw_used - central_limit) if raw_used is not None else 0)
    over_limit = over_by > 0
    return {"semantics": "central", "total": central_limit,
            "bar_used": bar_used, "remaining": remaining,
            "raw_used": raw_used, "over_limit": over_limit, "over_by": over_by,
            "encroachment": encroachment, "worker_usage": w,
            "external_usage": x, "external_headroom": headroom}


def free_vram_bytes() -> Optional[int]:
    """Budgetable free VRAM on the primary GPU (raw minus the operator
    reserve), or None if no GPU / can't tell."""
    from hugpy_platform.hardware import free_vram_bytes as _free_vram

    raw = _free_vram(_env_int("HUGPY_MAIN_GPU") or 0)
    if raw is None:
        return None
    return max(0, raw - vram_reserve_bytes())


def total_vram_bytes() -> Optional[int]:
    """Total INSTALLED VRAM on the primary GPU in bytes, or None if no GPU /
    can't tell.

    Unlike ``free_vram_bytes`` this is RAW — the operator reserve is NOT
    subtracted, because total is a fixed physical property of the card (its
    capacity), while the reserve is a slice held out of the FREE budget. The
    VRAM-ceiling gate (Fix A) uses it as the denominator for the ~90% ceiling —
    "keep the physical card at/under N% full" is a statement about the whole
    card, so it must be the whole card. Same probe family as ``free_vram_bytes``
    (torch.cuda.mem_get_info total, then nvidia-smi), so ceiling and free reads
    come from the same ComfyUI-visible device truth. Degrades to None (never 0)
    so the ceiling gate can tell "unmeasurable" from "no capacity" and fail
    OPEN."""
    from hugpy_platform.hardware import total_vram_bytes as _total_vram

    return _total_vram(_env_int("HUGPY_MAIN_GPU") or 0)


def _rss_anon_bytes(pid: int) -> Optional[int]:
    """RssAnon (anonymous, non-reclaimable resident) for ``pid`` from
    /proc/<pid>/status, in bytes. None if unreadable (non-Linux, permission, or
    the process exited between enumeration and read)."""
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def ram_worker_bytes() -> Optional[int]:
    """NON-RECLAIMABLE RAM this worker's process tree holds (the agent + every
    slot child), in bytes. ``worker_usage`` for the budget-bar spec: the RAM
    hugpy itself actually consumes, distinct from external processes.

    Sums **RssAnon**, NOT VmRSS (operator ruling 2026-07-31, option (a)). VmRSS
    counts memory-mapped GGUF weight pages, which are clean, file-backed page
    cache the kernel reports as AVAILABLE (it is total−MemAvailable that defines
    box-used). Counting them as "used" made the bar read ~74/88 GiB when the box
    truly used ~20; and because ``ram_external_bytes`` is
    ``max(0, box_used − worker_usage)``, an inflated worker figure OVERSHOT the
    whole box's real usage and clamped external to 0 — a doubly-wrong bar. The
    RSS-counts-mmap'd-GGUF landmine, surfacing on the RAM bar.

    ⚠ NOT ``memory_full_info().uss``: USS still counts these pages, because a
    llama.cpp weight mmap is a PRIVATE file mapping (clean but not shared), so
    USS ≈ VmRSS here (measured). RssAnon is the anonymous-only footprint that
    matches the kernel's used/available split.

    RssAnon comes from /proc (cheap); psutil only enumerates the tree. Per-proc
    fallback to VmRSS where /proc is unreadable (never fabricate). None if
    nothing was measurable at all."""
    try:
        import psutil
        me = psutil.Process()
        procs = [me] + me.children(recursive=True)
    except Exception:  # noqa: BLE001 — no psutil / permission: don't fabricate
        return None
    total = 0
    measured = False
    for p in procs:
        try:
            anon = _rss_anon_bytes(p.pid)
            if anon is None:
                anon = p.memory_info().rss   # non-Linux / unreadable: RSS fallback
            total += anon
            measured = True
        except Exception:  # noqa: BLE001 — a child may exit mid-walk
            continue
    return int(total) if measured else None


def ram_external_bytes() -> Optional[int]:
    """RAM used by everything OUTSIDE this worker's process tree, in bytes:
    (box used) − (worker own RSS). ``external_usage`` for the budget-bar spec.

    Box-used = physical total − MemAvailable, read against the SAME psutil
    snapshot as the total so the two can't skew. Clamped ≥0. None when either
    side is unmeasurable (never fabricated)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        box_used = int(vm.total) - int(vm.available)
    except Exception:  # noqa: BLE001
        return None
    own = ram_worker_bytes()
    if own is None:
        return None
    return max(0, box_used - own)


def ram_max_bytes() -> Optional[int]:
    """The central/local RAM CEILING in bytes (HUGPY_RAM_MAX_GIB), or None if
    unset. Set by _apply_central_limits from central's limits.ram_max_gib; the
    budget-bar spec's ``central_limit`` for RAM."""
    cap = _env_float("HUGPY_RAM_MAX_GIB")
    return int(cap * 2**30) if cap is not None else None


def free_ram_raw_bytes() -> Optional[int]:
    """Reserve-adjusted budgetable free RAM, UNCLAMPED by the RAM ceiling:
    ``max(0, MemAvailable − reserve)``. This is the honest "free after reserve"
    the console needs to show physical-semantics bars and that the ceiling-aware
    budget below is derived from. None if the raw read fails."""
    from hugpy_platform.hardware import free_ram_bytes as _free_ram

    raw = _free_ram()
    if raw is None:
        return None
    return max(0, raw - ram_reserve_bytes())


def free_ram_bytes() -> Optional[int]:
    """Budgetable free RAM the allocator's fit decisions consume.

    Reworked for the t13/t14 budget-bar spec so the bar and admission can NEVER
    disagree: the budgetable free is the SPEC's ``remaining`` for RAM, floored
    by the reserve-only free —

        budgetable = min(free_after_reserve, limit − worker_usage − encroachment)

    Interaction with HUGPY_RAM_RESERVE_GIB: the reserve is applied FIRST (in
    free_ram_raw_bytes) so a load can never consume MemAvailable to the OOM
    floor — that floor is independent of the central ceiling and always binds.
    The ceiling term then further constrains it to the WORKER'S budget: the
    limit minus what the worker already uses minus any external ENCROACHMENT
    (external usage that has spilled past the physical headroom into the
    worker's budget — spill.budget_bar). Where no ceiling is set, this is the
    old reserve-only behavior verbatim (limit term absent → min() is a no-op)."""
    raw = free_ram_raw_bytes()
    if raw is None:
        return None
    limit = ram_max_bytes()
    if limit is None:
        # No ceiling -> reserve-only behavior, exactly as before.
        return raw
    from hugpy_platform.hardware import free_ram_bytes as _free_ram
    physical = None
    try:
        import psutil
        physical = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        physical = None
    bar = budget_bar(physical_total=physical, central_limit=limit,
                     worker_usage=ram_worker_bytes(),
                     external_usage=ram_external_bytes())
    remaining = bar.get("remaining")
    if remaining is None:
        # Couldn't compute the spec remaining (missing worker/external reads) —
        # degrade to the historical hard cap so behavior never gets LOOSER than
        # before: min(free_after_reserve, limit).
        return min(raw, limit)
    # The allocator's free is the tighter of the reserve floor and the spec
    # remaining — the bar's number, so admission and the console agree.
    return min(raw, remaining)


def cpu_resident_bytes(model_path: str, n_gpu_layers: int) -> Optional[int]:
    """Rough RAM footprint of a GGUF load: the weight share NOT offloaded."""
    try:
        file_bytes = os.path.getsize(model_path)
    except OSError:
        return None
    if n_gpu_layers == -1:
        return 0
    if n_gpu_layers <= 0:
        return file_bytes
    total = _gguf_layer_count(model_path) or _ASSUMED_LAYERS
    frac = min(1.0, max(0.0, 1.0 - (n_gpu_layers / max(total, 1))))
    return int(file_bytes * frac)


# GGUF header reading moved DOWN to hugpy_storage.gguf_inspect (2026-09-22):
# file-format inspection is storage-level. Re-exported here so every existing
# caller (draft_models, placement, the fleet/platform tests) keeps its import.
from hugpy_storage.gguf_inspect import (  # noqa: E402
    _MOE_DETAIL_CACHE,
    _expert_tensor_re,
    _gguf_metadata,
    _gguf_scan_moe,
    _gguf_shard_paths,
    _layer_index,
    gguf_moe_detail,
)


def _gguf_layer_count(model_path: str) -> Optional[int]:
    """Read ``*.block_count`` from a GGUF header. Best-effort; None on any issue."""
    bc = _gguf_metadata(model_path, (".block_count",)).get(".block_count")
    try:
        return int(bc) if bc is not None else None
    except (TypeError, ValueError):
        return None


# n_cpu_moe value meaning "ALL expert layers on CPU" (llama-server caps it to
# the model's layer count, so any large sentinel works; 999 matches the ae
# deployment's proven LLAMA_ARG_N_CPU_MOE=999).
MOE_ALL_LAYERS = 999


def moe_split_need(detail: dict, n_cpu_moe: Optional[int] = None) -> "Optional[dict]":
    """Per-layer-aware pricing of a MoE split: what a load with ``--n-cpu-moe N``
    puts where. ``{"cpu_bytes", "gpu_bytes", "layers_on_cpu"}`` or None for a
    dense/unreadable detail (caller keeps opaque-size pricing).

    llama-server moves the expert tensors of the FIRST N block indices to CPU
    (everything else — attention, router, shared experts, embeddings, output,
    and the expert tensors of layers >= N — stays GPU-side). ``n_cpu_moe`` of
    None or >= the attributed layer count means ALL experts on CPU (the 999
    sentinel). The per-layer map keeps a future partial split precisely
    priceable instead of re-flattening the typed tensor list to one number."""
    if not isinstance(detail, dict) or not detail.get("is_moe"):
        return None
    expert = int(detail.get("expert_bytes") or 0)
    nexpert = int(detail.get("non_expert_bytes") or 0)
    by_layer = detail.get("expert_bytes_by_layer") or {}
    layers = sorted(by_layer)
    if n_cpu_moe is None or not layers or int(n_cpu_moe) >= len(layers):
        return {"cpu_bytes": expert, "gpu_bytes": nexpert,
                "layers_on_cpu": len(layers)}
    n = max(0, int(n_cpu_moe))
    cpu = sum(int(by_layer[i]) for i in layers[:n])
    return {"cpu_bytes": int(cpu), "gpu_bytes": int(nexpert + (expert - cpu)),
            "layers_on_cpu": n}


def moe_dense_first_plan(detail: dict,
                         gpu_budget_bytes: Optional[int],
                         *, extra_reserve_bytes: int = 0) -> "Optional[dict]":
    """DENSE BACKBONE FIRST: how to spend a GPU budget on a MoE model.

    Operator ruling 2026-07-31 (k53): for ANY MoE model the dense backbone —
    attention, router, shared experts, embeddings/output, and the KV cache that
    rides with them — is FIRST in line for whatever GPU budget the allocation
    mode grants, REGARDLESS of mode. The expert FFN tensors get only what is
    left. The inversion this retires is the one that stranded ae: a stale
    ``{"n_gpu_layers": -1}`` row read as an explicit demand, disabled the split,
    and llama.cpp then answered with a 17/48 LAYER split — 31 layers of DENSE
    attention on the CPU while expert weights sat on the card. Dense bytes are
    touched by EVERY token; expert bytes by ~expert_used/expert_count of one.
    Dense-on-CPU is therefore always the wrong trade.

    ``gpu_budget_bytes`` is the budgetable VRAM this load may claim (the mode's
    number: the free card for gpu-only/max-gpu, ``gpu_mem_gib`` for explicit,
    the RAM overflow for max-ram). ``extra_reserve_bytes`` is VRAM that lands on
    the card beside the weights (mmproj projector + the KV/context reserve) and
    is charged BEFORE anything is placed.

    Returns ``None`` for a dense/unreadable detail (caller keeps its existing
    pricing), else::

        {"n_cpu_moe": N,            # the --n-cpu-moe llama-server must launch with
         "expert_layers_on_gpu": k, # experts of the k HIGHEST block indices
         "cpu_bytes": …,            # expert bytes that land in host RAM
         "gpu_bytes": …,            # dense + the experts that fit
         "dense_fits": bool,        # the backbone itself fits the budget
         "budget_bytes": …}         # what was actually spent against

    llama-server's ``--n-cpu-moe N`` keeps the experts of the FIRST N BLOCK
    INDICES on the CPU, so the layers that ride the GPU are a SUFFIX of the
    block range — the plan fills that suffix from the top down and reports the
    threshold as the block index of the lowest expert layer kept on the card
    (not a positional count: a model whose first blocks are dense would be
    mispriced by position). ``MOE_ALL_LAYERS`` when nothing is left for the
    experts — the measured coder-next default (+59% tok/s at 5x less VRAM);
    ``0`` when the whole expert set fits, which is the honest, env-hack-proof
    way to say "everything on the card" (an explicit ``--n-cpu-moe 0`` also
    beats any inherited ``LLAMA_ARG_N_CPU_MOE``).
    """
    if not isinstance(detail, dict) or not detail.get("is_moe"):
        return None
    expert = int(detail.get("expert_bytes") or 0)
    nexpert = int(detail.get("non_expert_bytes") or 0)
    by_layer = {int(k): int(v) for k, v in
                (detail.get("expert_bytes_by_layer") or {}).items()}
    layers = sorted(by_layer)
    budget = int(gpu_budget_bytes or 0) - int(extra_reserve_bytes or 0)
    dense_fits = budget >= nexpert
    # The backbone is charged first, always. Whatever is left buys expert
    # layers from the TOP block index down (the suffix --n-cpu-moe expresses).
    remaining = budget - nexpert if dense_fits else 0
    kept = 0
    for i in reversed(layers):
        if by_layer[i] <= remaining:
            remaining -= by_layer[i]
            kept += 1
        else:
            break
    if kept <= 0:
        n_cpu_moe = MOE_ALL_LAYERS
    elif kept >= len(layers):
        n_cpu_moe = 0
    else:
        n_cpu_moe = layers[len(layers) - kept]
    cpu_bytes = (expert if n_cpu_moe == MOE_ALL_LAYERS
                 else sum(b for i, b in by_layer.items() if i < n_cpu_moe))
    return {"n_cpu_moe": int(n_cpu_moe),
            "expert_layers_on_gpu": int(kept),
            "cpu_bytes": int(cpu_bytes),
            "gpu_bytes": int(nexpert + expert - cpu_bytes),
            "dense_fits": bool(dense_fits),
            "budget_bytes": int(budget)}


def n_cpu_moe_env() -> Optional[int]:
    """Explicit per-request/per-model n_cpu_moe from HUGPY_N_CPU_MOE (the spill
    wire, set by the worker's _apply_spill), or None when unset. The number of
    MoE layers whose EXPERT tensors stay on CPU (MOE_ALL_LAYERS/999 = all);
    explicit always wins over the auto policy."""
    return _env_int("HUGPY_N_CPU_MOE")


# ── KV-cache quantification (slice 11 / t27) ────────────────────────────────
# Operator (2026-07-17): "the context can necessarily be quantified into ram
# needed correct? ... this should be a variable as well based on percentage max."
#
# The KV cache is the attention key/value tensors held for every token in the
# context window — the RAM/VRAM tax that fit/admission ignored (weights-only).
# The exact cache size is a function of the model's real geometry and the ctx:
#
#   kv_bytes = 2 (K and V) × n_layers × ctx_tokens × n_kv_heads × head_dim
#              × dtype_bytes
#
# n_kv_heads (NOT n_attention_heads) is what modern GQA/MQA models actually
# cache — Qwen2.5-Coder-3B has 16 attention heads but only 2 KV heads, an 8×
# reduction, so using attention-heads would over-count KV by 8×. head_dim =
# embedding_length / attention_head_count when not stated explicitly.
#
# dtype: llama.cpp caches fp16 by default (2 bytes); a quantized-KV config
# (-ctk/-ctv q8_0 / q4_0) lowers it. transformers caches in the model's compute
# dtype (torch_dtype: bf16/fp16 = 2, fp32 = 4) unless a cache override says else.
_KV_DTYPE_BYTES = {
    "f32": 4.0, "float32": 4.0, "fp32": 4.0,
    "f16": 2.0, "float16": 2.0, "fp16": 2.0, "bf16": 2.0, "bfloat16": 2.0,
    "q8_0": 1.0, "q8": 1.0, "int8": 1.0,
    "q5_0": 0.65, "q5_1": 0.69,
    "q4_0": 0.5, "q4_1": 0.56, "q4": 0.5, "int4": 0.5,
}
# When geometry is unavailable we NEVER silently return zero (that reintroduces
# the unplanned tax). A stated conservative heuristic: bytes per token per layer
# for a typical mid-size GQA model, cross-checked against the exact formula for
# Qwen2.5-Coder-3B (36L × 2kv × 128hd × 2B × 2 = ~256 KiB/token total ≈
# 7.3 KiB/token/layer → round UP to be conservative). Used only as a floor when
# real geometry can't be read; a WARN says so at the call site.
_KV_HEURISTIC_BYTES_PER_TOKEN_PER_LAYER = 8 * 1024  # 8 KiB, deliberately generous


def _kv_dtype_bytes(name: Optional[str], default: float = 2.0) -> float:
    if not name:
        return default
    return _KV_DTYPE_BYTES.get(str(name).strip().lower(), default)


def _gguf_kv_geometry(model_path: str) -> dict:
    """Layers / kv-heads / head-dim / trained ctx from a GGUF header, or {}.
    head_dim falls back to embedding_length / attention.head_count (llama.cpp's
    own derivation) when a *.attention.key_length is absent."""
    md = _gguf_metadata(model_path, (
        ".block_count", ".attention.head_count", ".attention.head_count_kv",
        ".embedding_length", ".attention.key_length", ".context_length"))
    if not md:
        return {}
    n_layers = md.get(".block_count")
    n_heads = md.get(".attention.head_count")
    n_kv = md.get(".attention.head_count_kv") or n_heads      # MHA: kv == heads
    emb = md.get(".embedding_length")
    head_dim = md.get(".attention.key_length")
    if not head_dim and emb and n_heads:
        try:
            head_dim = int(emb) // int(n_heads)
        except (TypeError, ValueError, ZeroDivisionError):
            head_dim = None
    out = {}
    for k, v in (("n_layers", n_layers), ("n_kv_heads", n_kv),
                 ("head_dim", head_dim), ("ctx_train", md.get(".context_length"))):
        try:
            if v is not None:
                out[k] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def _transformers_kv_geometry(config: dict) -> dict:
    """Layers / kv-heads / head-dim / dtype from a transformers config.json dict.
    Mirrors HF conventions: num_key_value_heads defaults to num_attention_heads
    (MHA) when absent; head_dim defaults to hidden_size / num_attention_heads."""
    if not isinstance(config, dict):
        return {}
    n_layers = config.get("num_hidden_layers")
    n_heads = config.get("num_attention_heads")
    n_kv = config.get("num_key_value_heads") or n_heads      # MHA fallback
    head_dim = config.get("head_dim")
    if not head_dim and config.get("hidden_size") and n_heads:
        try:
            head_dim = int(config["hidden_size"]) // int(n_heads)
        except (TypeError, ValueError, ZeroDivisionError):
            head_dim = None
    out: dict = {"dtype": config.get("torch_dtype")}
    for k, v in (("n_layers", n_layers), ("n_kv_heads", n_kv),
                 ("head_dim", head_dim),
                 ("ctx_train", config.get("max_position_embeddings"))):
        try:
            if v is not None:
                out[k] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def kv_bytes(*, ctx_tokens: int, n_layers: Optional[int] = None,
             n_kv_heads: Optional[int] = None, head_dim: Optional[int] = None,
             dtype_bytes: float = 2.0) -> Optional[int]:
    """KV-cache bytes for ``ctx_tokens`` given the model geometry, or a stated
    conservative HEURISTIC when geometry is missing (never silently zero).

    kv = 2 × n_layers × ctx × n_kv_heads × head_dim × dtype_bytes. Returns None
    only when ctx is non-positive (no cache). When n_layers is known but the
    per-head geometry is not, falls back to the bytes-per-token-per-layer
    heuristic (× n_layers × ctx); when even n_layers is unknown, uses an assumed
    layer count so the caller still gets a non-zero, conservative reserve."""
    try:
        ctx = int(ctx_tokens)
    except (TypeError, ValueError):
        return None
    if ctx <= 0:
        return None
    if n_layers and n_kv_heads and head_dim:
        return int(2 * int(n_layers) * ctx * int(n_kv_heads) * int(head_dim)
                   * float(dtype_bytes))
    # Geometry incomplete — conservative heuristic, never zero.
    layers = int(n_layers) if n_layers else _ASSUMED_LAYERS
    return int(layers * ctx * _KV_HEURISTIC_BYTES_PER_TOKEN_PER_LAYER)


# ---------------------------------------------------------------------------
# llama.cpp (GGUF)
# ---------------------------------------------------------------------------
def vision_projector_bytes(model_path: str) -> int:
    """Bytes of the mmproj / CLIP projector sidecar beside a vision GGUF (0 if none).

    A vision GGUF is a PAIR: the language-model quant + a separate ``mmproj-*.gguf``
    (the image encoder / projector). llama.cpp loads the projector ONTO THE GPU
    alongside the offloaded layers, so its VRAM must be reserved BEFORE we fit
    language-model layers — otherwise an 8 GB card computes "offload all layers"
    against the model file alone, then OOMs when the ~1.3 GB projector lands on
    top (Qwen2.5-VL-7B ships a 1.35 GB mmproj). Text models have no projector, so
    this returns 0 and the fit math is byte-identical to before.

    Self-contained (no import of the imports package) so the fit math stays
    offline-testable and can never be broken by a heavy import chain. Best-effort:
    any filesystem error yields 0 (fail open — never inflate the reserve).

    Sized as THE projector the child loads (2026-09-23): the slot passes
    ``--mmproj find_mmproj(path)``, so reserve exactly that file when the
    resolver is importable — a repo shipping mmproj-F32 + BF16 must not reserve
    one and load the other. Falls back to the filename scan below."""
    _hints = ("mmproj", "mm-proj", "mm_proj", "projector")
    try:
        from hugpy_platform.utils import find_mmproj as _find_mmproj
        _pick = _find_mmproj(model_path)
        if _pick and os.path.isfile(_pick):
            return int(os.path.getsize(_pick))
    except Exception:  # noqa: BLE001 — resolver unavailable: filename scan below
        pass
    try:
        directory = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        if not directory or not os.path.isdir(directory):
            return 0
        main_abs = os.path.abspath(model_path) if os.path.isfile(model_path) else None
        for fn in os.listdir(directory):
            low = fn.lower()
            if not low.endswith(".gguf"):
                continue
            if not any(h in low for h in _hints):
                continue
            cand = os.path.join(directory, fn)
            if main_abs is not None and os.path.abspath(cand) == main_abs:
                continue                        # never count the main file itself
            try:
                return int(os.path.getsize(cand))
            except OSError:
                return 0
    except OSError:
        return 0
    return 0


# ── The llama_context reserve: COMPUTED, not a flat constant (2026-07-27) ───
# THE DEFECT this replaces. ``autofit_gpu_layers`` held back a FLAT
# HUGPY_VRAM_CTX_RESERVE_GIB (2.5 GiB) on every card, and ``free_vram_bytes``
# had already held back a FLAT HUGPY_VRAM_RESERVE_GIB (1.0 GiB) before autofit
# ever saw the number. 3.5 GiB flat is 15% of a 24 GiB 3090 (tolerable) and 44%
# of computron's 8 GiB 4060 (crippling). The 2.5 was hand-tuned so a 70B could
# still build its llama_context on a 24 GB card; it never scaled DOWN, so a
# constant sized for the biggest card starved the smallest.
#
# What it cost, MEASURED on computron (RTX 4060, 7807 MiB card) with
# flux2-klein-9b-uncensored q4_k_m (5_027_783_648 B = 4.68 GiB, 36 layers):
#     autofit given 7.5 GiB (the whole empty card)  ->  29/36 layers
#     autofit given 6.5 GiB (after the 1.0 reserve) ->  23/36 layers
# yet a manual re-seat at n_gpu_layers=-1 put ALL 36 layers on that card —
# 6739 MiB used, 1068 MiB STILL FREE. The model demonstrably fit and the fit
# math could not see it. By the cliff measured 2026-07-25 that is a ~4x loss: a
# dense GGUF runs ~135 tok/s fully resident and ~36 the moment ONE layer spills.
#
# THE REPLACEMENT. The reserve is what a llama_context actually costs, and that
# is COMPUTABLE from geometry the GGUF header already carries and this module
# already parses (``_gguf_kv_geometry`` -> block_count / head_count_kv /
# key_length / context_length, verified against real files):
#
#     ctx_reserve = kv_bytes(n_ctx, geometry, fp16) + _CTX_COMPUTE_RESERVE_BYTES
#
# KV dominates and is EXACT (llama.cpp caches fp16 by default; we never assume a
# quantized cache). Verified against the real flux2 header (36 layers, 8 kv
# heads, head_dim 128): 2 x 36 x 8 x 128 x 2 B = 147_456 B/token = 144 KiB/token,
# so at n_ctx 16384 the KV cache is 2.25 GiB. The module's own measurement at
# that ctx was "7441 MiB on the card, 4.68 GiB of weights" -> 2.59 GiB of
# KV+compute+context, i.e. a 348 MiB residual on top of the computed KV. That
# residual is the compute-graph buffers + logits buffer + the process's own CUDA
# runtime/kernels, which do NOT scale with n_ctx; _CTX_COMPUTE_RESERVE_BYTES
# (512 MiB) covers it with ~47% headroom.
#
# DEGRADE-NOT-GUESS. Geometry that cannot be read (a non-GGUF path, a truncated
# header, a model with no head_count_kv) returns TODAY'S FLAT 2.5 GiB, so every
# unparseable case behaves exactly as before. An EXPLICIT
# HUGPY_VRAM_CTX_RESERVE_GIB is an operator override and is honoured verbatim,
# flat, with no computation and no un-stacking (see autofit_gpu_layers).
_DEFAULT_CTX_RESERVE_GIB = 2.5
# Mirrors serve.DEFAULT_LLAMA_CTX — the cap the llama.cpp loader applies to
# every served ctx. Duplicated (not imported) so the fit math stays self-
# contained and offline-testable; llama_ctx_cap() prefers the REAL value
# whenever the serve layer is already loaded in this process.
_CTX_CAP_FALLBACK = 16384
# Compute-graph + logits + CUDA-runtime VRAM that lands beside the KV cache.
# Measured 348 MiB on the reference model (see above); 512 MiB is that with
# headroom. Deliberately ctx-INDEPENDENT: llama.cpp sizes these off n_ubatch
# (512 by default here) and n_vocab, not off n_ctx.
_CTX_COMPUTE_RESERVE_BYTES = 512 * 2**20


def llama_ctx_cap() -> int:
    """The ctx cap the llama.cpp loader would apply (``serve.DEFAULT_LLAMA_CTX``).

    Read WITHOUT importing the serve layer: if serve is already in
    ``sys.modules`` (every real serving path) its authoritative value is used;
    otherwise the same ``DEFAULT_LLAMA_CTX`` env var serve itself reads; else the
    module fallback. Never raises, always returns a positive int."""
    cap = None
    try:
        mod = sys.modules.get("hugpy_engine.serve.serve")
        if mod is not None:
            cap = getattr(mod, "DEFAULT_LLAMA_CTX", None)
    except Exception:  # noqa: BLE001 — a cap read must never break a load
        cap = None
    if not cap:
        cap = _env_int("DEFAULT_LLAMA_CTX")
    try:
        cap = int(cap or 0)
    except (TypeError, ValueError):
        cap = 0
    return cap if cap > 0 else _CTX_CAP_FALLBACK


def ctx_for_fit(model_path: str, n_ctx: Optional[int] = None,
                geometry: Optional[dict] = None) -> int:
    """The n_ctx the fit math should price the KV cache against.

    An explicit ``n_ctx`` (the caller KNOWS what the child will be launched with
    — slot_agent resolves ``-c`` before it fits layers) always wins. Otherwise
    mirror what the loader would pick: ``min(the model's trained context, the
    engine cap)`` — the same shape as ``serve._ctx_for``'s fallback, derived from
    the GGUF's own ``*.context_length`` rather than guessed."""
    try:
        n = int(n_ctx) if n_ctx else 0
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return n
    cap = llama_ctx_cap()
    geo = geometry if isinstance(geometry, dict) else _gguf_kv_geometry(model_path)
    try:
        trained = int((geo or {}).get("ctx_train") or 0)
    except (TypeError, ValueError):
        trained = 0
    return min(trained, cap) if trained > 0 else cap


def vram_ctx_reserve_bytes(model_path: str,
                           n_ctx: Optional[int] = None) -> "tuple[int, str, dict]":
    """VRAM a llama_context needs BESIDES the weights: ``(bytes, source, detail)``.

    ``source`` is one of:
      * ``"env"``       — HUGPY_VRAM_CTX_RESERVE_GIB was explicitly set. The
        operator's number, flat, verbatim; no geometry is consulted.
      * ``"computed"``  — KV(n_ctx, real geometry, fp16) + the compute-graph
        allowance. The honest per-model figure.
      * ``"default"``   — geometry unreadable (non-GGUF, truncated header, no
        head_count_kv): TODAY'S FLAT 2.5 GiB, byte-identical to the old
        behaviour. Degrade-not-guess.

    ``detail`` carries the inputs (ctx, kv_bytes, geometry) for logging; it is
    never load-bearing."""
    env_gib = _env_float("HUGPY_VRAM_CTX_RESERVE_GIB")
    if env_gib is not None:
        return int(env_gib * 2**30), "env", {"ctx": None}
    flat = int(_DEFAULT_CTX_RESERVE_GIB * 2**30)
    try:
        geo = _gguf_kv_geometry(model_path) or {}
    except Exception:  # noqa: BLE001 — an unreadable header is the flat path
        geo = {}
    n_layers = geo.get("n_layers")
    n_kv_heads = geo.get("n_kv_heads")
    head_dim = geo.get("head_dim")
    if not (n_layers and n_kv_heads and head_dim):
        # Incomplete geometry: do NOT fall to kv_bytes' bytes-per-token heuristic
        # here — a fit decision must not be made on a guessed cache size. Today's
        # flat reserve is the stated degrade.
        return flat, "default", {"ctx": None}
    ctx = ctx_for_fit(model_path, n_ctx=n_ctx, geometry=geo)
    kv = kv_bytes(ctx_tokens=ctx, n_layers=n_layers, n_kv_heads=n_kv_heads,
                  head_dim=head_dim, dtype_bytes=2.0)   # llama.cpp caches fp16
    if not kv:
        return flat, "default", {"ctx": ctx}
    return (int(kv) + _CTX_COMPUTE_RESERVE_BYTES, "computed",
            {"ctx": ctx, "kv_bytes": int(kv),
             "compute_bytes": _CTX_COMPUTE_RESERVE_BYTES,
             "n_layers": int(n_layers), "n_kv_heads": int(n_kv_heads),
             "head_dim": int(head_dim)})


def autofit_gpu_layers(model_path: str,
                       free_vram: Optional[int] = None,
                       extra_reserve_bytes: int = 0,
                       n_ctx: Optional[int] = None) -> int:
    """How many GGUF layers fit in free VRAM. -1 (all) when the whole file fits.

    ``free_vram`` is a BUDGETABLE figure — the operator VRAM reserve
    (``vram_reserve_bytes``) already removed, i.e. what ``free_vram_bytes()``
    returns. Every caller in the tree passes one from that family; the reserve
    arithmetic below relies on it (see THE STACKING DECISION).

    ``extra_reserve_bytes`` is VRAM that must be held OUT of the layer budget
    because something else lands on the GPU next to the offloaded layers — for a
    vision GGUF this is the mmproj/CLIP projector (see ``vision_projector_bytes``).
    It is subtracted from the budget alongside the KV/context allowance, so a
    partial split leaves honest room for the projector and we only return -1 (all
    layers) when the whole model AND the projector AND the context all fit. 0
    (the default) reproduces the historical text-model behaviour exactly.

    ``n_ctx`` is the context the child will ACTUALLY be launched with. The KV
    cache is linear in it, so this is the single biggest input to the reserve;
    slot_agent resolves ``-c`` before calling. Omitted -> ``ctx_for_fit``
    reproduces the loader's own choice from the GGUF header.

    THE STACKING DECISION (2026-07-27). Two reserves used to be charged against
    the same card for the same load: ``vram_reserve_bytes`` (1.0 GiB, removed
    upstream inside ``free_vram_bytes``, for "GPU consumers central knows nothing
    about") and this context reserve. They are different CONCERNS and they still
    stack — for a SPLIT. They no longer stack for the WHOLE-FIT test:

        whole fits?  free - max(0, context need - external floor) >= file
                     (equivalently: raw free - max(need, floor) >= file)
        how many?    free - (context need + projector)
                     (fully stacked: a split never plans into the floor)

    Why the whole-fit test is un-stacked, in one measured line: the manual
    re-seat proved flux2-klein-9b costs 6739 MiB all-in on a 7807 MiB card, so
    the honest non-weight overhead is ~1.9 GiB. Against the post-reserve 6.5 GiB
    even a PERFECT context reserve leaves 4.60 GiB for a 4.68 GiB file — it
    still refuses, by 84 MiB. No achievable accuracy in this reserve fixes the
    measured case while the two stack; the stack ITSELF is the refusal. And the
    external reserve is a GROWTH cushion, not an occupancy measure: the free read
    already excludes every byte a stranger holds right now.

    Why the SPLIT stays stacked: the credit exists to stop the CLIFF — a model
    that measurably fits being spilled anyway costs ~4x throughput (135 -> 36
    tok/s the moment one layer leaves the card). A model that cannot fit whole
    has no cliff to avoid, so there is nothing to buy by planning into the floor
    — and everything to lose (that is how a 70B lands 20 GB of weights and then
    dies with "Failed to create llama_context"). So the credit can only ever
    convert a spill into a whole seat; it never buys extra layers in a split.

    The real OOM backstop lives elsewhere — the device ceiling in agent
    ``_vram_ceiling_reserve_bytes`` / ``_worker_slot_fit_check``, which demands
    real device headroom after the weights land and reads device truth, so it
    sees out-of-band growth this reserve never could. Since 2026-07-27 that
    ceiling defaults to a BOUNDED cushion — ``_CTX_COMPUTE_RESERVE_BYTES``, the
    same 512 MiB compute term this function adds, un-stacked against
    ``vram_reserve_bytes`` — rather than (1-frac) of the WHOLE card, which
    re-charged for the KV the incoming need already carries. An explicit
    HUGPY_VRAM_CEILING_FRAC still means exactly what it always did.
    ``free_vram_bytes`` itself is UNCHANGED: every other consumer (heartbeat
    budgets, contention fit, transformers max_memory, preflights) still gets the
    reserve-adjusted figure.

    An EXPLICIT HUGPY_VRAM_CTX_RESERVE_GIB opts out of all of the above: the
    operator's number is used flat and stacks exactly as it did before, for both
    tests. So does the geometry-unreadable fallback (today's flat 2.5 GiB,
    stacked) — degrade-not-guess means the degraded path is the OLD path,
    unchanged."""
    if free_vram is None:
        free_vram = free_vram_bytes()
        # Operator VRAM budget (the console's "VRAM budget…" mode / spill
        # gpu_mem_gib) caps the autofit. GGUF loads ignored this before — the
        # knob only reached the transformers path, so a per-model budget on a
        # GGUF worker silently did nothing.
        gpu_gib = _env_float("HUGPY_GPU_MEM_GIB")
        if gpu_gib is not None:
            cap = int(gpu_gib * 2**30)
            free_vram = min(free_vram, cap) if free_vram else cap
    if not free_vram:
        # Fail OPEN, not closed. free_vram is unknown here, but if the box HAS a
        # GPU (detect_gpus finds a card even when the free-VRAM probe on the
        # primary index came back None — e.g. the slot supervisor whose
        # torch/nvidia-smi view differs from the agent's) an offload-capable
        # llama.cpp must put every layer on the GPU, not drop the model to CPU.
        # Only a genuinely GPU-less host stays on CPU (0). Gate on detect_gpus()
        # (hardware truth) rather than importing llama_cpp here — a CUDA
        # llama_cpp import would poison a later torch import in the in-process
        # (agent/central) caller of autofit.
        try:
            from hugpy_platform.hardware import detect_gpus
            if detect_gpus():
                return -1
        except Exception:
            pass
        return 0

    try:
        file_bytes = os.path.getsize(model_path)
    except OSError:
        return 0
    if file_bytes <= 0:
        return 0

    # Weights are not the whole story: llama_context still needs VRAM AFTER the
    # weights land (KV cache, linear in n_ctx, plus compute-graph buffers). That
    # is the OOM this guard was built for — a 70B uploads ~20 GB of weights and
    # then dies with "Failed to create llama_context". It is now COMPUTED from
    # the model's real geometry at the real n_ctx instead of a flat constant
    # (vram_ctx_reserve_bytes), and it no longer stacks on the external floor
    # (see THE STACKING DECISION in the docstring).
    ctx_reserve, ctx_source, ctx_detail = vram_ctx_reserve_bytes(
        model_path, n_ctx=n_ctx)
    extra = max(0, int(extra_reserve_bytes))
    # SPLIT budget — fully stacked, the conservative figure. The external floor
    # is already off ``free_vram``, so a partial plan leaves that floor intact:
    # predicted card use (layers + context) <= raw free - floor.
    split_reserve = ctx_reserve + extra
    # WHOLE-FIT budget — the un-stacked one (see THE STACKING DECISION). Only a
    # "computed" reserve earns the credit; an operator override and the
    # geometry-unreadable degrade keep today's stacked arithmetic exactly.
    whole_reserve = split_reserve
    if ctx_source == "computed":
        whole_reserve = max(0, ctx_reserve - vram_reserve_bytes()) + extra
    scaled = int(free_vram * _vram_safety())
    total_layers = _gguf_layer_count(model_path) or _ASSUMED_LAYERS
    if scaled - whole_reserve >= file_bytes:
        logger.info(
            "autofit gguf: file=%.1fGiB free=%.1fGiB reserve=%.2fGiB (%s, "
            "ctx=%s) -> ALL %d layers on GPU",
            file_bytes / 2**30, free_vram / 2**30, whole_reserve / 2**30,
            ctx_source, ctx_detail.get("ctx"), total_layers)
        return -1                           # everything fits on the GPU
                                            # (model + projector + context)
    budget = scaled - split_reserve
    if budget <= 0:
        logger.info(
            "autofit gguf: file=%.1fGiB free=%.1fGiB reserve=%.2fGiB (%s, "
            "ctx=%s) -> 0/%d layers on GPU (no budget)",
            file_bytes / 2**30, free_vram / 2**30, split_reserve / 2**30,
            ctx_source, ctx_detail.get("ctx"), total_layers)
        return 0

    # Weights dominate the file; approximate per-layer cost as an even split.
    per_layer = file_bytes / max(total_layers, 1)
    fit = int(budget // per_layer)
    fit = max(0, min(fit, total_layers))
    logger.info(
        "autofit gguf: file=%.1fGiB free=%.1fGiB reserve=%.2fGiB (%s, ctx=%s) "
        "-> %d/%d layers on GPU",
        file_bytes / 2**30, free_vram / 2**30, split_reserve / 2**30,
        ctx_source, ctx_detail.get("ctx"), fit, total_layers,
    )
    return fit


# The worker's honest partial-offload admission (agent._vram_evict_to_fit) pins an
# EXPLICIT n_gpu_layers per SERVED-quant path here, so the in-process llama_cpp
# load offloads exactly the admitted layer count. Without it the in-process path
# would fall to autofit_gpu_layers, which sizes off the on-disk file (shard-1
# only) and UNDER-counts a sharded model -> over-offloads -> the very OOM this
# fixes. Path-keyed (not model_key) because gguf_gpu_layers only sees a path;
# absent/None -> historical env+autofit behaviour, byte-identical.
_NGL_OVERRIDE: dict[str, int] = {}


def set_ngl_override(model_path, n_gpu_layers) -> None:
    """Pin an explicit n_gpu_layers for a resolved GGUF path (the worker's honest
    partial-offload plan). Consulted by gguf_gpu_layers BEFORE env/autofit."""
    try:
        _NGL_OVERRIDE[os.path.abspath(str(model_path))] = int(n_gpu_layers)
    except (TypeError, ValueError, OSError):
        pass


def clear_ngl_override(model_path) -> None:
    """Drop the partial-offload pin for a path (re-admission / full-fit re-decide)."""
    try:
        _NGL_OVERRIDE.pop(os.path.abspath(str(model_path)), None)
    except (TypeError, ValueError, OSError):
        pass


# ── k37 allocation modes (worker-side wire: HUGPY_ALLOC_MODE etc.) ──────────
# The five-mode selector's NEW spill keys land here as env (set per request by
# agent._apply_spill): alloc_mode -> HUGPY_ALLOC_MODE, leniency_pct ->
# HUGPY_LENIENCY_PCT, priority_device -> HUGPY_PRIORITY_DEVICE. Only max-ram /
# explicit ever ride this wire — gpu-only/ram-only/max-gpu keep the unchanged
# legacy n_gpu_layers encoding. Central version-gates emission so an old
# worker never receives these; on a mode-aware worker they must never be a
# dead knob, so both engine paths below consult them.
def alloc_mode_env() -> Optional[str]:
    """The request's allocation mode from HUGPY_ALLOC_MODE ("max-ram" |
    "explicit"), or None (legacy encoding / no mode — behave exactly as
    before)."""
    raw = _env("HUGPY_ALLOC_MODE")
    if raw is None:
        return None
    low = raw.strip().lower()
    return low or None


def leniency_pct_env() -> Optional[float]:
    """explicit mode's leniency (percent OF THE MODEL that may land off its
    ideal device before bust), 0..100, or None when unset."""
    v = _env_float("HUGPY_LENIENCY_PCT")
    if v is None:
        return None
    return max(0.0, min(100.0, v))


def priority_device_env() -> str:
    """explicit mode's priority device ("gpu" default | "ram")."""
    raw = (_env("HUGPY_PRIORITY_DEVICE") or "gpu").strip().lower()
    return "ram" if raw == "ram" else "gpu"


def no_evict_env() -> bool:
    """k56 POLITE LOAD: True when this request's spill asked for a load that may
    spend only genuinely free headroom and must NEVER evict a resident
    (HUGPY_NO_EVICT, set per request by agent._apply_spill and cleared when
    absent). Unset -> False, i.e. the declare-need-then-evict doctrine that
    remains the rule for every unflagged load."""
    raw = (_env("HUGPY_NO_EVICT") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# RAM safety factor for the max-ram fill (mirror of _VRAM_SAFETY: never budget
# every last byte of MemAvailable for weights).
_RAM_FILL_SAFETY = 0.95


def maxram_gpu_layers(model_path: str, free_ram: Optional[int] = None) -> int:
    """n_gpu_layers for the **max-ram** mode: fill the RAM budget FIRST, and
    only the OVERFLOW layers go to the GPU — autofit's per-layer pricing,
    inverted. 0 when the whole model fits RAM (pure RAM residency; the GPU is
    only touched when RAM genuinely can't hold everything).

    The RAM budget = budgetable free RAM (reserve- and ceiling-aware) capped by
    an explicit HUGPY_CPU_MEM_GIB when set, with a safety factor so the fill
    never rides MemAvailable to the OOM floor. Whether the GPU can actually
    hold the overflow is the ADMISSION engine's question
    (flex.plan_explicit_offload, ram priority) — this is the loader-level
    intent, exactly like autofit_gpu_layers is for max-gpu."""
    try:
        file_bytes = os.path.getsize(model_path)
    except OSError:
        return 0
    if file_bytes <= 0:
        return 0
    if free_ram is None:
        free_ram = free_ram_bytes()
        cpu_gib = _env_float("HUGPY_CPU_MEM_GIB")
        if cpu_gib is not None:
            cap = int(cpu_gib * 2**30)
            free_ram = min(free_ram, cap) if free_ram else cap
    total_layers = _gguf_layer_count(model_path) or _ASSUMED_LAYERS
    per_layer = file_bytes / max(total_layers, 1)
    ram_budget = int((free_ram or 0) * _RAM_FILL_SAFETY)
    in_ram = min(total_layers, int(ram_budget // per_layer)) if per_layer > 0 else total_layers
    overflow = max(0, total_layers - in_ram)
    logger.info(
        "max-ram gguf: file=%.1fGiB ram_budget=%.1fGiB -> %d/%d layers in RAM, "
        "%d overflow to GPU",
        file_bytes / 2**30, ram_budget / 2**30, in_ram, total_layers, overflow)
    return overflow


def gguf_gpu_layers(model_path: str) -> int:
    """Resolve n_gpu_layers for a GGUF model: the worker's partial-offload pin
    first (honest layers-that-fit), else the k37 allocation mode (max-ram's
    inverted fill), else env (+autofit)."""
    try:
        pinned = _NGL_OVERRIDE.get(os.path.abspath(model_path))
    except (TypeError, OSError):
        pinned = None
    if pinned is not None:
        return int(pinned)
    # k37: max-ram inverts the fill (RAM first, overflow to GPU). explicit
    # falls through to autofit — its VRAM target rides HUGPY_GPU_MEM_GIB which
    # already caps autofit; the leniency FLOOR is enforced at admission
    # (flex.plan_explicit_offload), not here.
    if alloc_mode_env() == "max-ram":
        return maxram_gpu_layers(model_path)
    raw = _env("HUGPY_N_GPU_LAYERS")
    if raw is None or raw.lower() == "auto":
        return autofit_gpu_layers(model_path)
    if raw.lower() in ("off", "cpu", "none"):
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning("bad HUGPY_N_GPU_LAYERS=%r; using autofit", raw)
        return autofit_gpu_layers(model_path)


def tensor_split() -> Optional[list[float]]:
    raw = _env("HUGPY_TENSOR_SPLIT")
    if not raw:
        return None
    try:
        parts = [float(x) for x in raw.split(",") if x.strip()]
        return parts or None
    except ValueError:
        logger.warning("bad HUGPY_TENSOR_SPLIT=%r; ignoring", raw)
        return None


def main_gpu() -> Optional[int]:
    return _env_int("HUGPY_MAIN_GPU")


def rpc_servers() -> Optional[str]:
    """Comma-separated "host:port" of llama.cpp rpc-servers to shard onto, or None.

    Set (as a per-request spill override) by central's allocator when it decides
    to shard a model across multiple GPUs on different machines.
    """
    raw = _env("HUGPY_RPC_SERVERS")
    return raw.strip() if raw and raw.strip() else None


def _binding_supports_rpc() -> bool:
    """Whether this llama-cpp-python build accepts ``Llama(rpc_servers=…)``.

    The param existed in 0.2.78–0.2.90 and was dropped in the 0.3.x rewrite —
    passing it there raises TypeError. Shard leads on a >=0.3 binding must be
    served via ``llama-server --rpc …`` (the serve layer's extra_args) instead
    of the in-process runner.
    """
    try:
        import inspect
        from llama_cpp import Llama
        return "rpc_servers" in inspect.signature(Llama.__init__).parameters
    except Exception:
        return False


def llama_kwargs(model_path: str) -> dict[str, Any]:
    """Spill kwargs for ``Llama(...)``. Always includes n_gpu_layers.

    When ``HUGPY_RPC_SERVERS`` is set we're the LEAD of a cross-machine shard:
    pass ``rpc_servers`` and force ``n_gpu_layers=-1`` (offload ALL layers across
    the pooled GPUs — the whole point is to never touch CPU). ``tensor_split``
    (also supplied by the allocator) then weights layers across [local, *rpc].
    """
    rpc = rpc_servers()
    if rpc and not _binding_supports_rpc():
        logger.warning(
            "HUGPY_RPC_SERVERS=%s set but this llama-cpp-python has no "
            "rpc_servers param (dropped in 0.3.x) — ignoring the shard plan "
            "and loading locally. Serve shard leads via llama-server --rpc "
            "(cfg.extra['llama_extra_args']) instead.", rpc,
        )
        rpc = None
    if rpc:
        kwargs: dict[str, Any] = {"n_gpu_layers": -1, "rpc_servers": rpc}
    else:
        kwargs = {"n_gpu_layers": gguf_gpu_layers(model_path)}
    ts = tensor_split()
    if ts is not None:
        kwargs["tensor_split"] = ts
    mg = main_gpu()
    if mg is not None:
        kwargs["main_gpu"] = mg
    return kwargs


# ---------------------------------------------------------------------------
# placement intent — one wire field (n_gpu_layers), interpreted PER ENGINE
# ---------------------------------------------------------------------------
# The console's Autofit / Max GPU / CPU only controls are ENGINE-AGNOSTIC
# PLACEMENT INTENT (operator ruling 2026-07-17: "the autofit, maxgpu and cpu only
# should still be on the table as its easy to infer what those would indicate for
# a transformer"). They ride the SAME wire field the GGUF path uses —
# HUGPY_N_GPU_LAYERS (-1 / "off" / "auto") — but that field NAMES llama.cpp layer
# counts. For a GGUF model it IS a layer count (gguf_gpu_layers). For a
# transformers model there are no "gpu layers" to count, so the field is read as
# PLACEMENT INTENT and mapped onto the transformers device_map/max_memory world:
#   * -1  ("Max GPU")  -> put the WHOLE model on the GPU (no CPU budget)
#   * 0/"off" ("CPU only") -> keep it entirely on CPU (gpu budget 0)
#   * unset/"auto"     -> today's autofit (shard to fit VRAM, spill to CPU)
# Same wire, per-engine interpretation — placement intent, NOT layer counts.
def n_gpu_layers_intent() -> str:
    """Decode HUGPY_N_GPU_LAYERS into an engine-agnostic PLACEMENT class:
    ``"gpu"`` (all-on-GPU, from -1), ``"cpu"`` (CPU-only, from 0/"off"/"cpu"/
    "none"), or ``"auto"`` (fit-and-spill, from unset/"auto"/a positive int).

    A positive int is a llama.cpp partial-offload count with no transformers
    analogue, so for the transformers path it reads as ``"auto"`` (fit as much as
    fits) — the honest 'some on GPU' behavior — rather than inventing a split."""
    raw = _env("HUGPY_N_GPU_LAYERS")
    if raw is None:
        return "auto"
    low = raw.strip().lower()
    if low in ("auto", ""):
        return "auto"
    if low in ("off", "cpu", "none"):
        return "cpu"
    if low == "-1":
        return "gpu"
    try:
        return "cpu" if int(low) == 0 else "auto"
    except ValueError:
        return "auto"


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------
def _gib(n: float) -> str:
    return f"{n:.2f}GiB"


def bnb_4bit_env() -> bool:
    """Is the operator's bitsandbytes 4-bit lever ON for the model being loaded?

    Rides the same spill wire as n_cpu_moe (HUGPY_BNB_4BIT, set per-load by the
    agent's _apply_spill and cleared when absent). Central decides; this is the
    worker-side read."""
    return str(os.environ.get("HUGPY_BNB_4BIT", "")).strip().lower() in (
        "1", "true", "yes", "on")


def transformers_quantization_config():
    """``BitsAndBytesConfig`` for a 4-bit load, or None when the lever is off /
    bitsandbytes is unavailable.

    nf4 + double-quant + fp16 compute — the same recipe generate/coder.py has
    used since before this lever existed, so a model quantized through the
    console loads identically to one quantized by the coder path.

    Returns None (never raises) when bitsandbytes is missing: the caller then
    loads full precision, which is the pre-lever behaviour. A worker without
    the package must degrade honestly rather than fail the load — central's
    eligibility gate already refuses CPU-only boxes, so this is the belt to
    that braces."""
    if not bnb_4bit_env():
        return None
    try:
        import bitsandbytes  # noqa: F401 — availability probe
        from hugpy_platform.module_imports import get_transformers
        BitsAndBytesConfig = get_transformers("BitsAndBytesConfig")
        import torch
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    except Exception as exc:  # noqa: BLE001 — never break a load over the lever
        logger.warning("4-bit requested but bitsandbytes is unusable (%s) — "
                       "loading full precision", exc)
        return None


def planned_gpu_need_bytes(need_bytes: Optional[int]) -> Optional[int]:
    """What the CURRENT placement intent will actually put on the GPU, given a
    total need of ``need_bytes``. THE admission companion to
    ``transformers_max_memory`` / the GGUF ngl mapping — one derivation, so the
    gate and the loader cannot disagree (incident 2026-07-29: a model designated
    RAM-only was refused "won't fit on GPU: needs 70.2 GB" and an idle resident
    was evicted on the way — for a load that was about to place 0 B on the card).

      * intent "cpu" (n_gpu_layers off/0)  -> 0 (nothing lands on the GPU)
      * alloc_mode "max-ram"               -> the REMAINDER after the CPU budget
        (same math as transformers_max_memory: cpu = explicit or free_ram*0.8;
        gpu = max(0, need - cpu); unknown need -> 0, matching the loader's
        never-silently-fill-the-GPU rule)
      * everything else                    -> ``need_bytes`` unchanged

    Returns None only when ``need_bytes`` is None and the intent would not force
    it to 0 — i.e. "unknown", which admission already fails open on."""
    if n_gpu_layers_intent() == "cpu":
        return 0
    if alloc_mode_env() == "max-ram":
        if not need_bytes:
            return 0                       # loader: unknown need -> GPU budget 0
        cpu_gib = _env_float("HUGPY_CPU_MEM_GIB")
        if cpu_gib is None:
            fr = free_ram_bytes()
            cpu_gib = (fr * 0.8) / 2**30 if fr else 16.0
        return max(0, int(need_bytes) - int(cpu_gib * 2**30))
    return need_bytes


def transformers_max_memory(model_need_bytes: Optional[int] = None) -> Optional[dict]:
    """Build a ``max_memory`` map for device_map='auto', or None to skip.

    Explicit env budgets win; otherwise autofit from detected free VRAM/RAM.
    Returns None when no GPU is visible (let transformers stay on CPU).

    k37 **max-ram** (HUGPY_ALLOC_MODE=max-ram): RAM-priority placement —
    generous CPU budget + only the REMAINDER on the GPU. accelerate fills GPUs
    first, so RAM priority is expressed by capping the GPU budget at what the
    CPU budget cannot hold: gpu = max(0, need − cpu). ``model_need_bytes``
    (optional, from a loader that knows its size — Slice C wires the gap
    loaders) makes that remainder honest; without it the GPU budget is 0 (pure
    RAM — safe, never a silent GPU fill against an explicit RAM priority).
    NOTE central engine-gates max-ram/explicit to GGUF models today, so this
    branch is defense-in-depth + the Slice C seam, not a live central path.

    PLACEMENT INTENT (t26): HUGPY_N_GPU_LAYERS is honored here as engine-agnostic
    placement, NOT a layer count (see n_gpu_layers_intent):
      * "cpu"  (n_gpu_layers 0/"off") -> gpu budget 0 GiB so device_map='auto'
        places the whole model on CPU. Returned as a map (not None) so the
        intent BINDS even when a GPU is present — the operator asked for CPU.
      * "gpu"  (n_gpu_layers -1)      -> no CPU budget: the model is forced onto
        the card (accelerate raises honestly if it truly can't fit, rather than
        us silently spilling against an explicit 'all on GPU').
      * "auto"                        -> today's fit-and-spill behavior, unchanged.
    An EXPLICIT gpu_mem_gib/cpu_mem_gib budget still wins over the intent-derived
    default for that axis (the GGUF-only explicit class; when present here it is
    simply honored)."""
    intent = n_gpu_layers_intent()
    gpu_gib = _env_float("HUGPY_GPU_MEM_GIB")
    cpu_gib = _env_float("HUGPY_CPU_MEM_GIB")
    n_gpu = _env_int("HUGPY_N_GPU") or 1

    # k37 max-ram: RAM-priority — generous CPU, remainder (if known) on GPU.
    if alloc_mode_env() == "max-ram":
        if cpu_gib is None:
            fr = free_ram_bytes()
            cpu_gib = (fr * 0.8) / 2**30 if fr else 16.0
        if gpu_gib is None:
            if model_need_bytes:
                gpu_gib = max(0.0, (int(model_need_bytes) / 2**30) - cpu_gib)
            else:
                gpu_gib = 0.0          # unknown need: never silently fill the GPU
        mm_ram: dict[Any, str] = {i: _gib(gpu_gib) for i in range(max(n_gpu, 1))}
        mm_ram["cpu"] = _gib(cpu_gib)
        logger.info("transformers placement=max-ram max_memory=%s", mm_ram)
        return mm_ram

    # CPU-only intent: force everything off the GPU. Bind even with a GPU present
    # (an explicit CPU-only placement, not autofit) — gpu budget 0, generous CPU.
    if intent == "cpu":
        if cpu_gib is None:
            fr = free_ram_bytes()
            cpu_gib = (fr * 0.8) / 2**30 if fr else 16.0
        mm_cpu: dict[Any, str] = {i: _gib(0.0) for i in range(max(n_gpu, 1))}
        mm_cpu["cpu"] = _gib(cpu_gib)
        logger.info("transformers placement=cpu-only max_memory=%s", mm_cpu)
        return mm_cpu

    if gpu_gib is None:
        fv = free_vram_bytes()
        if not fv:
            return None                     # no GPU -> no spill map
        gpu_gib = (fv * _vram_safety()) / 2**30

    # All-on-GPU intent: no CPU budget so accelerate keeps the whole model on the
    # card. Skip the CPU-spill fallback below (a bare gpu-only max_memory). An
    # explicit cpu_mem_gib still wins if the operator set one alongside.
    if intent == "gpu":
        mm_gpu: dict[Any, str] = {i: _gib(gpu_gib) for i in range(max(n_gpu, 1))}
        if cpu_gib is not None:
            mm_gpu["cpu"] = _gib(cpu_gib)
        logger.info("transformers placement=all-gpu max_memory=%s", mm_gpu)
        return mm_gpu

    if cpu_gib is None:
        fr = free_ram_bytes()
        cpu_gib = (fr * 0.8) / 2**30 if fr else 16.0

    mm: dict[Any, str] = {i: _gib(gpu_gib) for i in range(max(n_gpu, 1))}
    mm["cpu"] = _gib(cpu_gib)
    logger.info("transformers placement=auto max_memory=%s", mm)
    return mm


def describe() -> dict[str, Any]:
    """Human-readable snapshot of the current spill config (for heartbeats/UI)."""
    return {
        "mode": (_env("HUGPY_N_GPU_LAYERS") or "auto"),
        "alloc_mode": alloc_mode_env(),          # k37: max-ram|explicit|None
        "leniency_pct": leniency_pct_env(),
        "priority_device": (priority_device_env()
                            if alloc_mode_env() == "explicit" else None),
        "n_gpu_layers_env": _env("HUGPY_N_GPU_LAYERS"),
        "n_cpu_moe": n_cpu_moe_env(),            # MoE expert-split knob (env wire)
        "gpu_mem_gib": _env_float("HUGPY_GPU_MEM_GIB"),
        "cpu_mem_gib": _env_float("HUGPY_CPU_MEM_GIB"),
        "tensor_split": tensor_split(),
        "free_vram_bytes": free_vram_bytes(),
    }
