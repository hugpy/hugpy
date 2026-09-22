"""The FIVE allocation modes — the operator-facing placement vocabulary (k37).

Operator spec 2026-07-23 (allocation-modes-spec): every model gets ONE of five
FLAT modes. Each mode = placement preference + bust condition:

    gpu-only   all layers on the GPU, no spill; won't fit GPU (after evict) -> bust
    ram-only   all in RAM, never the GPU (binds CPU even with a GPU present);
               won't fit RAM -> bust
    max-gpu    as much GPU as available+needed, spill the rest to RAM;
               can't satisfy -> bust                      (THE DEFAULT — a blank
               model serves-and-spills, never OOMs: defaults-are-promises)
    max-ram    as much RAM as available+needed, spill the rest to GPU;
               can't satisfy -> bust
    explicit   target VRAM/RAM budgets + a leniency%% + a device priority
               (gpu default); can't fit even at the loosened floor -> bust

Internally max-gpu / max-ram are explicit(priority-device, ~100%% target,
generous leniency) — but they stay FIVE FLAT NAMES on every surface. The split
is a COGNITIVE-LOAD boundary, not an implementation one (operator ruling):
"use my GPU, spill the rest" must stay a zero-knob pick; the moment you tune
(%%/leniency/priority) you reach for explicit. Never collapse max-* into
explicit in any UI/API surface.

LENIENCY MATH (operator-confirmed): N%% leniency = up to N%% OF THE MODEL may
land off its ideal device before bust. Ideal 100%% GPU + 30%% leniency ->
degrade step-by-step down to the FLOOR 70%% GPU / 30%% RAM; only bust when even
the floor won't fit. The conversion onto the tolerance-band engine
(worker_agent/flex.py) is: whole = the MODEL's bytes, deviation = leniency%%,
so ``band_floor(target_bytes, leniency_pct, model_bytes)`` IS the floor.

THE HONEST RENAME (keeper owns nomenclature): today's console "Max GPU"
(n_gpu_layers=-1, all-or-OOM) is really **gpu-only**; today's "autofit"
(as-much-GPU-as-fits, spill rest) is really **max-gpu**. Legacy names are
accepted on INPUT (resolved + logged), never emitted back. NOTE the collision:
the STRING "max-gpu" exists in both vocabularies with different meanings —
resolution is canonical-first, so "max-gpu" always reads as the NEW max-gpu
(fit-and-spill); the historical -1 meaning is only reachable as "gpu-only"
(nothing persisted ever stored the old string — the old wire encoding was the
n_gpu_layers value itself, which read-time DERIVATION maps correctly).

WIRE ENCODING (unchanged for the three legacy-expressible modes — keeper
amendment 3: n_gpu_layers semantics NEVER change on the wire):
    gpu-only -> {"n_gpu_layers": -1}
    ram-only -> {"n_gpu_layers": "off"}
    max-gpu  -> {}                        (autofit, zero knobs)
    max-ram  -> {"alloc_mode": "max-ram"}                       (NEW keys)

⚠ max-gpu HAS TWO ENCODINGS, and the distinction is the whole point of the
2026-07-25 bugfix — they differ in PROVENANCE, not in behavior:

    {}                          a DERIVED / blank max-gpu. Nothing is persisted;
                                the model re-derives against current conditions
                                on every read. Also THE CLEAR SIGNAL: assign_model
                                treats an empty spill as "remove this row".
    {"alloc_mode": "max-gpu"}   an EXPLICIT operator pick. Persisted, so it is
                                immune to the derivation changing under it.

Both mean max-gpu TO THE WORKER, and neither reaches it as a mode key: the
explicit form is stripped at the emission seam (WorkerStore.spill_for), so the
wire is byte-identical to today's {} in both cases. The stripping stays because
the wire contract is frozen, but it is no longer LOAD-BEARING for placement: a
mode key on the worker used to suppress the auto MoE split entirely (making an
explicit max-gpu WORSE than a blank one), and since the dense-backbone-first
ruling (2026-07-31) no mode can suppress it — every mode that grants a GPU
budget spends it on the dense backbone first (managers.serve.slot_agent.
_moe_gpu_budget + spill.moe_dense_first_plan). What a mode still decides is HOW
MUCH card the model may claim.

The old code collapsed the explicit form to {} here, which fed it straight into
the clear path: selecting max-gpu in the console DELETED the row instead of
writing it. See normalize_spill for the full incident note.
    explicit -> {"alloc_mode": "explicit", gpu_mem_gib?, cpu_mem_gib?,
                 "leniency_pct"?, "priority_device"?, priority?} (NEW keys)

VERSION GATE (no dead knobs): the NEW spill keys are only emitted to workers
whose pkg_version honors them (>= MODE_MIN_PKG_VERSION). Released workers
IGNORE unknown spill keys (verified safe), but a selected mode must never be a
silent no-op — an old worker's request is downgraded to max-gpu ({} autofit)
and the downgrade is logged/surfaced (gate_spill_for_worker).

This module is PURE (stdlib only) so chaos, the routes, overrides, and the
worker can all share ONE vocabulary without import weight.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The five flat operator-facing modes. Order matters only for display.
ALLOC_MODES = ("gpu-only", "ram-only", "max-gpu", "max-ram", "explicit")

# Legacy name -> canonical mode. Accepted on input, resolved + logged, never
# emitted back. "max-gpu" is listed for the HISTORICAL record (old "Max GPU"
# = -1 = new gpu-only) but is UNREACHABLE at runtime: resolution is
# canonical-first and "max-gpu" is a canonical name of the new vocabulary.
LEGACY_ALLOC_ALIASES = {
    "autofit": "max-gpu",     # old fit-and-spill default -> the new honest name
    "cpu-only": "ram-only",   # old CPU-only -> RAM is what it actually binds
    "cpu_only": "ram-only",
    "max-gpu": "gpu-only",    # HISTORICAL ONLY (all-or-OOM); see module note
    "max_gpu": "gpu-only",    # (underscore form has no canonical collision)
    "gpu_only": "gpu-only",
    "ram_only": "ram-only",
    "max_ram": "max-ram",
    "budget": "explicit",     # old chaos budget draw = an explicit gpu budget
    "bands": "explicit",      # old chaos band draw = explicit with a band
}

# Spill keys that only a mode-aware worker understands. Presence of ANY of
# these on a spill makes it version-gated (gate_spill_for_worker). n_cpu_moe
# (the MoE expert-split knob, 2026-07-24) ships in the SAME cut as the mode
# keys, so the one MODE_MIN_PKG_VERSION gate covers it — an older worker would
# silently drop the knob (unknown spill keys are ignored), and a selected knob
# must never be a silent no-op.
NEW_SPILL_KEYS = frozenset({"alloc_mode", "leniency_pct", "priority_device",
                            "n_cpu_moe"})

# First worker package version whose spill/env plumbing honors the new keys
# (Slice B2 ships in this cut). Anything older gets the max-gpu fallback.
MODE_MIN_PKG_VERSION = "0.1.203"

# ── k56: POLITE LOAD (``no_evict``) ─────────────────────────────────────────
# The per-model inverse of declare-need-then-evict (which stays the rule for
# unflagged loads): admission may spend only GENUINELY FREE headroom — the free
# VRAM left after the tolerance-band flex — and never triggers an eviction.
#
# It rides the SAME spill wire as the mode keys but carries its OWN version
# gate, because it ships a whole cut later (0.1.226) and the two must not be
# conflated: a 0.1.210 worker honors alloc_mode perfectly and would still evict
# a resident for a polite load. Deliberately NOT in NEW_SPILL_KEYS — that set
# means "downgrade the whole spill to max-gpu ({} autofit)", which for a polite
# load would silently DROP the politeness and keep the placement, i.e. exactly
# the silent-no-op failure the gate exists to prevent. The gate below strips the
# key and says so LOUDLY instead; central's resolution refuses to route a polite
# model to a worker that predates it (see workers._polite_admits).
NO_EVICT_SPILL_KEY = "no_evict"
NO_EVICT_MIN_PKG_VERSION = "0.1.226"

# ── t74: HARD NO-THINK — per-request engine chat keys ───────────────────────
# ``chat_template_kwargs`` (the Qwen3 ``enable_thinking:false`` idiom — the
# chat template pre-closes ``<think>\n\n</think>`` so generation starts with
# thinking already shut) and ``logit_bias`` (the <think>-token-ban fallback)
# now ride ChatRequest and are forwarded verbatim onto the slot child's
# llama-server /v1/chat/completions body. They exist because /no_think is a
# SOFT switch some models ignore outright (wazimondo~Qwen3.6-35B-A3B-
# Uncensored-Wasserstein-GGUF burns its whole budget thinking anyway); the
# template kwarg is enforced by the TEMPLATE, not by model obedience.
#
# These are NOT spill keys (they steer the request body, never placement), so
# they are gated separately from NEW_SPILL_KEYS — but by the same rule: a
# worker whose builder predates them silently DROPS the keys (its chat-request
# whitelist doesn't forward them), and a selected suppression must never be a
# silent no-op. Central strips them for an old worker and logs the downgrade;
# the soft directive + caller-side strip (utils/no_think.py) still ride the
# messages, so the request degrades to today's behavior, never breaks.
CHAT_EXTRAS_WIRE_KEYS = frozenset({"chat_template_kwargs", "logit_bias"})
CHAT_EXTRAS_MIN_PKG_VERSION = "0.1.229"

# Modes a non-GGUF (transformers/comfy) model may select. Slice C wired the gap
# loaders to the spill seam, and this slice (2026-07-24, operator-approved)
# OPENS the central gate for ``max-ram``: transformers honor it via
# transformers_max_memory's RAM-priority branch, diffusers via
# enable_model_cpu_offload — so it is a real, honest mode for non-GGUF now.
# ``explicit`` STAYS GGUF-only: its banded leniency floor is a llama.cpp concept
# with no transformers analogue (opening it would break the mode's promise). So
# the non-GGUF set is the four modes that have a working non-GGUF meaning; only
# explicit is dropped.
NONGGUF_ALLOWED_MODES = ("gpu-only", "ram-only", "max-gpu", "max-ram")

# GGUF family: the HF-canonical 'gguf' plus the llama_cpp synonym. A GGUF model
# is ALWAYS max-gpu by default (partial offload makes any size feasible — the
# runner spills whatever won't fit to RAM), so its feasible default never
# depends on the box's totals. Everything else (transformers / comfy) loads
# whole-tensor via accelerate, so an oversized model genuinely cannot land on
# the GPU and its feasible default is worker-dependent.
GGUF_ENGINES = frozenset({"gguf", "llama_cpp"})

# Headroom factor for the transformers GPU-fit test (feasible_default_mode).
# A transformers model whose effective footprint exceeds this fraction of the
# box's TOTAL GPU capacity "clearly cannot fit the GPU" and defaults to
# ram-only on that worker. 0.9 leaves ~10% for the CUDA context, activation
# working set, and KV/attention scratch that ride alongside the weights — a
# model at 0.95× total VRAM would OOM the moment it allocated its first
# forward-pass buffer, so 0.9 is the defensible "clearly cannot fit" line
# (not a fit PREDICTION — the real autofit/accelerate placement still runs; this
# only picks the read-time DEFAULT so a doomed max-gpu is never the blank
# promise). Above 0.9× GPU but within RAM -> ram-only; above RAM too -> leave
# max-gpu and let the worker refuse honestly (no invented fourth state).
#
# LIFTED 0.9 -> 0.98 (operator, 2026-07-25). Quant sizes are chosen against REAL
# card capacities, so a 23.5 GiB release IS the "fits a 24 GiB card" build. At
# 0.9 the cutoff was 21.6 GiB, so a 23.5 GiB model on a 24 GiB 3090 derived
# **ram-only** — never touching the card it was published for, at the ~7 tok/s
# floor the cliff sweep measured, when the alternative was ~135. Rejecting the
# top tier of every card in the fleet is a worse failure than an occasional
# honest load-time refusal.
#
# Why this is the safe one of the two margins lifted today: the derived mode is
# only a PREFERENCE. The real placement still runs autofit/accelerate, and the
# ~90% admission ceiling still gates the load — so a wrong guess here degrades
# to a clear refusal, never a crash. 0.98 rather than 1.0 keeps a token margin
# for the CUDA context itself.
_GPU_FIT_HEADROOM = 0.98

# Bytes per GiB, as a float, for the human-readable ``why`` lines and the
# gpu_mem_gib/cpu_mem_gib budget conversion in default_allocation.
_GIB_F = float(1 << 30)

# n_cpu_moe value meaning "ALL expert layers on CPU". MIRRORS
# ``managers.spill.MOE_ALL_LAYERS`` (999 — llama-server caps it to the model's
# layer count, so any large sentinel works). Duplicated as a literal ON PURPOSE:
# this module is PURE stdlib so chaos, the routes, overrides and the worker can
# share ONE vocabulary without import weight, and importing spill here would
# drag the GGUF reader (struct/glob/os.stat) into every one of those callers.
# The two are asserted equal in tests so the mirror can never drift silently.
MOE_ALL_LAYERS = 999

# ── bitsandbytes "specialization" (operator, 2026-07-26) ────────────────────
# A SEPARATE AXIS FROM THE MoE SPLIT, and the distinction matters:
#   * the MoE split is PLACEMENT — the weights are untouched, we just decide
#     which tensors sit on the card and which sit in RAM (coder-next is 45 GiB
#     either way; only 1.49 GiB of it is non-expert);
#   * bitsandbytes is COMPRESSION — it changes what the weights ARE, loading
#     fp16 at nf4 so a 67 GiB model becomes ~17 GiB and genuinely fits a 24 GiB
#     card. It needs no structural seam, so it applies to dense models too.
# They compose (a transformers MoE could be both) and they live at different
# layers: --n-cpu-moe is a llama.cpp flag, bitsandbytes is a transformers
# load-time quantization_config.
#
# 4-bit nf4 + double quant ~= 1/4 of fp16, but the RESIDUAL is not free: the
# quantization constants, the un-quantized layers (embeddings/norms/lm_head are
# commonly skipped) and the compute buffers all stay. 0.30 is the honest
# working figure rather than a flattering 0.25 — an OPTIMISTIC ratio here would
# derive "fits the GPU" for a model that then OOMs at load, which is exactly the
# failure defaults-are-promises forbids.
BNB_4BIT_SIZE_RATIO = 0.30

# Repos that are ALREADY quantized must never be offered bitsandbytes: loading a
# pre-quantized checkpoint under a second quantization config fails, and the
# size win is already banked. Matched on the key/hub id because that is what the
# registry reliably carries for a not-yet-downloaded model.
_PREQUANT_MARKERS = ("4bit", "8bit", "nvfp4", "fp4", "int4", "int8",
                     "gptq", "awq", "-nf4", "bnb")


def is_prequantized(model_key: Any) -> bool:
    """True if the model name declares an existing quantization."""
    k = str(model_key or "").lower()
    return any(m in k for m in _PREQUANT_MARKERS)


def bnb_eligible(engine: Any, model_key: Any, *, has_cuda: bool = True) -> bool:
    """Can this (model, worker) take a bitsandbytes specialization?

    Three gates, all necessary:
      * NOT a GGUF — llama.cpp carries its own quantization; bitsandbytes is a
        transformers/diffusers loader concept and has no meaning here;
      * the worker HAS CUDA — bitsandbytes' 4-bit kernels are CUDA-only, so
        offering the lever on a CPU-only box would be a promise that fails at
        load (op has no GPU);
      * NOT already quantized — see is_prequantized.
    """
    if is_gguf_engine(engine):
        return False
    if not has_cuda:
        return False
    return not is_prequantized(model_key)


def bnb_effective_bytes(model_bytes: Any) -> "Optional[int]":
    """The size the allocator should PRICE when bitsandbytes is enabled."""
    n = _as_int(model_bytes)
    return int(n * BNB_4BIT_SIZE_RATIO) if n else None


def is_gguf_engine(engine: Any) -> bool:
    """True for the GGUF family (gguf / llama_cpp), case-insensitive. Anything
    else — including None/unknown — is treated as non-GGUF by the caller, but
    the feasible-default logic degrades unknown engines to max-gpu regardless."""
    return str(engine or "").strip().lower() in GGUF_ENGINES


def _as_int(v: Any) -> "Optional[int]":
    try:
        return int(v) if v else None
    except (TypeError, ValueError):
        return None


def feasible_modes(engine: Any,
                   model_bytes: "Optional[int]",
                   gpu_total_bytes: "Optional[int]",
                   ram_total_bytes: "Optional[int]",
                   moe_split_gpu_bytes: "Optional[int]" = None,
                   bnb: bool = False) -> tuple:
    """The allocation modes that are FEASIBLE for one (model x worker), in
    ALLOC_MODES display order (operator ruling 2026-07-24 scope-extension: "the
    user shouldn't be able to select an option that implies something it cannot
    do"). The blank default is the best member of this set (feasible_default_mode
    returns exactly that). PURE — the enforcement/surface glue lives at the
    routes.

    Feasibility matrix (each rule is a hard can-it-physically-land test, not a
    fit prediction — the real placement still runs; this only bounds what may be
    SELECTED so an impossible mode is never offered):

      * ``gpu-only`` — the model fits the GPU total within the headroom factor
        (``model <= _GPU_FIT_HEADROOM * gpu_total``). All-or-bust on the GPU, so
        it must plausibly fit the GPU alone. STRICT for a MoE too (operator
        ruling 2026-07-31): the FULL footprint is priced, experts included —
        see ``moe_split_gpu_bytes`` below.
      * ``ram-only`` — the model fits RAM total (``model <= ram_total``). Binds
        the CPU; never touches the GPU.
      * ``max-gpu`` — GGUF: ALWAYS (partial offload spills whatever won't fit to
        RAM, so it is universally feasible). Transformers/comfy: the same
        numbers rule as max-ram — fits GPU+RAM COMBINED (118ac64, operator
        ruling 2026-07-29: max-gpu is a spill PREFERENCE; the non-GGUF loaders
        honor the spill via max_memory / cpu_offload, so a fits-the-card-whole
        gate here made spill-capable models unassignable).
      * ``max-ram`` — a split exists: the model fits RAM+GPU COMBINED (its
        overflow rides the GPU). Engine-agnostic as of 2026-07-24: GGUF and
        non-GGUF both honor it (transformers RAM-priority max_memory, diffusers
        enable_model_cpu_offload — Slice C wired the loaders, this slice opened
        the gate). The numbers rule (fits GPU+RAM combined) applies to BOTH.
      * ``explicit`` — some split exists (``model <= gpu_total + ram_total``);
        ENGINE-GATED: non-GGUF stays dropped regardless of the numbers — its
        banded leniency floor is a llama.cpp concept with no transformers
        analogue, so opening it would break the mode's promise.

    UNKNOWN size or unknown totals -> ALL modes feasible: never eliminate on
    missing data (degrade to today's permissiveness). Specifically, if
    ``model_bytes`` is unknown, or the total a rule needs is unknown, that rule
    does NOT eliminate its mode. The engine gate on ``explicit`` is the ONLY
    elimination that fires without size/totals (it is a capability fact, not a
    measurement); ``max-ram`` is now engine-agnostic and eliminated only by the
    numbers.

    ``moe_split_gpu_bytes`` (MoE, 2026-07-24): the GPU-side need of a detected
    MoE's expert split (non-expert bytes — surfaced by gguf_variants_detail's
    ``moe`` at enrichment); None for dense models. It USED to relax the gpu-only
    gate: a 45 GiB MoE on a 24 GiB card was offered gpu-only because the split
    would only put ~3 GiB on the card.

    ⚠ IT NO LONGER DOES (operator ruling 2026-07-31: "gpu-only must be honest to
    its name"). ALL tensors on the card, experts included — a MoE whose full
    footprint doesn't fit VRAM makes gpu-only INFEASIBLE, not "feasible with the
    experts quietly in RAM", which silently redefined *only*. The
    GPU-maximal-then-spill behavior the split expresses belongs to **max-gpu**,
    whose contract already says spill and which is UNCONDITIONALLY feasible for
    a GGUF for exactly that reason (the partial-offload/expert split is what
    makes any size serveable) — so the split's feasibility home is that branch,
    not this one, and nothing is lost by pricing gpu-only honestly.

    The argument is still accepted: callers pass it, and central reports it in
    ``feasibility_context`` so a 409 can name the numbers the decision saw. It
    is deliberately inert as a GATE — the split relaxes no mode's elimination
    now that the mode it relaxed is the one that must not be relaxed."""
    size = _as_int(model_bytes)
    gpu_total = _as_int(gpu_total_bytes)
    ram_total = _as_int(ram_total_bytes)
    # t159: a priced expert share can only come from a GGUF header read, so an
    # unlabelled engine (registry framework None) with one IS a GGUF — keep the
    # selectable set in step with default_allocation's structure routing.
    gguf = is_gguf_engine(engine) or _as_int(moe_split_gpu_bytes) is not None
    # BITSANDBYTES RE-PRICING (operator ruling 2026-07-29: "its auto should
    # change upon 4-bit delegation"). default_allocation has priced the 4-bit
    # footprint since 2026-07-26 while THIS function — the set the console
    # offers and /assign enforces — still priced the full fp16 size. The two
    # disagreed: the allocator planned ~15 GiB for a 51.7 GiB model and would
    # have placed it on the card, but the gate refused the mode outright. Same
    # re-pricing, same place in the order (before any rule reads `size`), so
    # 4-bit and MoE still compose.
    #
    # ⚠ ESTIMATE, NOT A MEASUREMENT: BNB_4BIT_SIZE_RATIO is a flat 0.30 of the
    # whole directory, deliberately pessimistic vs the theoretical 0.25. It is
    # NOT per-component, so for a diffusers PIPELINE (VAE/tokenizer/scheduler
    # never quantize) it can still flatter. Feasibility is a gate, not a
    # promise the load will fit — the worker's own admission preflight remains
    # the measured authority.
    if bool(bnb) and not gguf and size:
        size = bnb_effective_bytes(size) or size
    unknown_size = size is None

    out = []
    for mode in ALLOC_MODES:
        if mode == "gpu-only":
            # Fits GPU (headroom), priced on the FULL footprint — a MoE gets no
            # discount here (see moe_split_gpu_bytes above: "only" means the
            # experts are on the card too, and a footprint that needs RAM is
            # max-gpu's job). Unknown size/gpu_total -> don't eliminate.
            feasible = (unknown_size or gpu_total is None
                        or size <= _GPU_FIT_HEADROOM * gpu_total)
        elif mode == "ram-only":
            feasible = (unknown_size or ram_total is None
                        or size <= ram_total)
        elif mode == "max-gpu":
            if gguf:
                # Partial offload: universal — AND the home of the MoE expert
                # split since the 2026-07-31 gpu-only ruling. "As much GPU as
                # fits, spill the rest" is precisely what a dense-first expert
                # split does, so an oversized MoE is feasible HERE and only
                # here.
                feasible = True
            else:
                # SAME numbers rule as max-ram (operator ruling 2026-07-29:
                # "feasible modes should be auto-derived from auto setting; if
                # they derive non-feasible modes that's a blanket function
                # problem"). max-gpu means "as much GPU as fits, SPILL THE REST
                # TO RAM" — so the test is fits-GPU+RAM-combined, never
                # fits-the-card-whole. That is the same spill the non-GGUF
                # loaders already honor for max-ram (transformers RAM-priority
                # max_memory, diffusers enable_model_cpu_offload — wired by
                # Slice C), so gating the two differently was an inconsistency,
                # not a capability fact.
                #
                # WHAT IT COST: every non-GGUF model larger than VRAM lost THE
                # DEFAULT mode, so a stored/derived max-gpu became unassignable
                # — /assign 409'd with "not feasible" on 8 of ae's designations
                # after the 2026-07-29 in-VM migration (24 GiB card), making
                # models that CAN spill simply uncallable. defaults-are-promises
                # says the default must be a success path, not a refusal.
                combined = None
                if gpu_total is not None or ram_total is not None:
                    combined = (gpu_total or 0) + (ram_total or 0)
                feasible = (unknown_size or combined is None
                            or size <= combined)
        elif mode == "max-ram":
            # Engine-agnostic (2026-07-24): both GGUF and non-GGUF honor max-ram
            # now, so no engine gate — only the numbers rule (fits GPU+RAM
            # combined). Unknown size/totals -> don't eliminate.
            combined = None
            if gpu_total is not None or ram_total is not None:
                combined = (gpu_total or 0) + (ram_total or 0)
            feasible = (unknown_size or combined is None
                        or size <= combined)
        elif mode == "explicit":
            # Engine gate (a capability fact, fires even on missing data): the
            # banded leniency floor has no transformers analogue, so explicit
            # stays GGUF-only. GGUF then also honors the numbers rule.
            if not gguf:
                feasible = False
            else:
                combined = None
                if gpu_total is not None or ram_total is not None:
                    combined = (gpu_total or 0) + (ram_total or 0)
                feasible = (unknown_size or combined is None
                            or size <= combined)
        else:  # pragma: no cover — ALLOC_MODES is closed
            feasible = True
        if feasible:
            out.append(mode)
    # Never return an empty set — a model must always have SOMETHING selectable
    # (defaults-are-promises). If the numbers eliminated everything (e.g. a
    # transformers model bigger than RAM and GPU), fall back to max-gpu so the
    # worker can refuse HONESTLY downstream rather than the UI offering nothing.
    return tuple(out) if out else ("max-gpu",)


def feasible_default_mode(engine: Any,
                          model_bytes: "Optional[int]",
                          gpu_total_bytes: "Optional[int]",
                          ram_total_bytes: "Optional[int]",
                          moe: "Optional[dict]" = None,
                          bnb: bool = False,
                          moe_force: "Optional[bool]" = None) -> str:
    """The BLANK default alloc mode derived by FEASIBILITY for one (model x
    worker), engine-aware (operator ruling 2026-07-24). This ONLY supplies the
    default when NOTHING is persisted — an explicit alloc_mode always wins
    upstream; this is never consulted for a model that has one.

    NAME-ONLY VIEW: this returns the mode NAME, for the surfaces that reason in
    names (console dropdown, the /assign feasibility gate). The MoE leaf's
    content is a SPLIT — a pair of numbers plus a knob — which a ``str`` cannot
    carry, so callers that PERSIST an allocation must use
    :func:`default_allocation` instead. The two agree by construction: the MoE
    branch below delegates to it.

      * GGUF DENSE -> ``gpu-only`` when it fits the card whole (operator
        default order 2026-07-31: prefer full-card residency), else the
        max-ram/ram-only tail; unknown size/totals degrade to ``max-gpu``
        (partial offload keeps every size servable — never guess).
      * GGUF MoE (``moe`` supplied and is_moe, 2026-07-25) -> the operator's MoE
        branch via ``default_allocation``: ``explicit`` when the non-expert
        share fits the GPU and the experts fit RAM, else that function's
        fall-through. Absent a ``moe`` detail this is byte-identical to the
        dense path — the caller simply didn't price the structure, and
        degrade-not-guess says don't invent it.
      * transformers/comfy:
          - if the footprint CLEARLY cannot fit the GPU
            (``model_bytes > _GPU_FIT_HEADROOM * gpu_total_bytes``) but DOES fit
            RAM (``model_bytes <= ram_total_bytes``) -> ``ram-only`` (emits the
            legacy ``{"n_gpu_layers": "off"}`` — works on ANY worker version, no
            gate). This is the "68 GB model, 24 GB GPU, 124 GB RAM -> RAM-only"
            case: the only feasible option, so it IS the default.
          - if it plausibly fits the GPU -> ``max-gpu`` (today's default;
            autofit/accelerate handles it).
          - if it fits NEITHER (bigger than RAM too) -> ``max-gpu`` and let the
            worker refuse honestly (no invented fourth state).

    DEGRADE-NOT-GUESS: any missing input (unknown size, unknown GPU total, or —
    for the ram-only decision — unknown RAM total) falls back to ``max-gpu``,
    today's behavior. A default is never derived from a guessed number."""
    # ── ONE TREE, TWO VIEWS (2026-07-25, eviction-flow slice) ────────────────
    # This function is now a PURE PROJECTION of ``default_allocation``: same
    # inputs, same tree, ``["mode"]`` taken off the end. It used to walk its own
    # copy of the tree for the non-MoE cases, and the copies had DRIFTED — the
    # spec's third open item ("feasible_default_mode and default_allocation
    # still disagree for a large dense GGUF").
    #
    # THE DISAGREEMENT, precisely: for a DENSE GGUF this returned "max-gpu"
    # unconditionally ("a GGUF is ALWAYS max-gpu by default — partial offload
    # makes any size feasible"), while ``default_allocation``'s dense-GGUF leaf
    # returns **max-ram** when the model exceeds the GPU's usable share but
    # fits RAM (the PREFERENCE-vs-PROHIBITION ruling: prefer RAM, don't FORBID
    # the card). So a 40 GiB dense GGUF on a 24 GiB card showed "max-gpu" in the
    # console dropdown and persisted "max-ram" when chosen.
    #
    # WHY IT NOW MATTERS ENOUGH TO FIX. Under the eviction flow the mode is no
    # longer just a placement label — it is the eviction PREFERENCE (key ①), so
    # the label decides WHICH MODEL DIES. A dropdown reading "max-gpu" for a
    # model the system treats as max-ram tells the operator its residents sort
    # one way while they sort the other. That is not a cosmetic mismatch.
    #
    # WHICH SIDE WON, and why ``default_allocation``: it is the side that gets
    # PERSISTED (it returns the wire encoding, so what the operator picks is
    # what lands), it is the side with the measured rationale attached (the
    # ~5x cliff), and it is the only one that can express the MoE leaf. The
    # name-only view had no argument for its answer beyond "GGUFs spill" —
    # true, and exactly why the other side chose max-ram rather than ram-only.
    return default_allocation(engine, model_bytes, gpu_total_bytes,
                              ram_total_bytes, moe=moe, bnb=bnb,
                              moe_force=moe_force)["mode"]


def default_allocation(engine: Any,
                       model_bytes: "Optional[int]",
                       gpu_total_bytes: "Optional[int]",
                       ram_total_bytes: "Optional[int]",
                       *,
                       moe: "Optional[dict]" = None,
                       bnb: bool = False,
                       moe_force: "Optional[bool]" = None) -> dict:
    """THE full operator decision tree (2026-07-25) for a model's INITIAL
    DEFAULT allocation, DERIVED from its own structure instead of a blanket
    stamp. Returns ``{"mode": <one of ALLOC_MODES>, "spill": {...}, "why": str}``
    — the mode, the wire encoding to persist, and one honest line naming the
    numbers that chose it.

    This is a SIBLING of :func:`feasible_default_mode`, not a replacement. That
    function answers "which of the five NAMES is the blank default" and is
    consumed by surfaces (the console dropdown, the /assign gate) that reason in
    mode names only. The MoE leaf of this tree cannot be expressed as a name:
    its whole content is a SPLIT — a pair of numbers plus a knob — so a function
    returning ``str`` structurally cannot carry it. Rather than widen the return
    type of a function with several call sites that index it as a string, the
    tree lives here and ``feasible_default_mode`` stays the name-only view
    (it now delegates, so the two can never disagree).

    THE TREE (operator, verbatim structure):

        transformers ── gpu large enough? ─ yes ──────────────► max-gpu*
                                          └ no ─ ram large? ─ yes ► ram-only
                                                            └ no ► break
        gguf ─ is_moe? ─ yes ─ gpu large enough for the NON-EXPERT share?
                        │        ├ yes ─ ram large enough for the EXPERTS?
                        │        │        ├ yes ────────────► explicit (SPLIT)
                        │        │        └ no ─────────────► (fall through)
                        │        └ no ── ram large enough? ─ yes ► ram-only
                        │                                   └ no ► break
                        └ no (dense) ─ gpu large enough? ─ yes ─► max-gpu*
                                                          └ no ─ ram? ─► ram-only
                                                                       └► break

    (*) the tree's "-- gpu only" leaf is implemented LITERALLY as ``gpu-only``
    since 2026-07-31 (operator default order; between 07-25 and 07-31 it read
    max-gpu as a bust-avoidance departure — history and the reversal argument
    are on ``_gpu_else_ram`` below). Only a fits-the-card-whole model reaches
    it; every other leaf keeps a spill-capable or split spelling.

    THE MoE LEAF IS THE POINT. coder-next Q4_K_M is 45 GiB of file but only
    1.49 GiB of NON-EXPERT tensors; the 43.59 GiB of experts are meant for RAM.
    Priced whole it "doesn't fit" a 24 GiB card and would be stamped max-gpu;
    priced by STRUCTURE it fits with 22 GiB to spare, and the split is the
    measured success path (+59%% tok/s at 5x less VRAM on ae). The default must
    therefore be the split, DERIVED — never a blanket stamp.

    WIRE ENCODING FOR THE MoE LEAF — ``{"alloc_mode": "explicit",
    "n_cpu_moe": MOE_ALL_LAYERS, "n_gpu_layers": -1, "gpu_mem_gib":
    <non-expert>, "cpu_mem_gib": <experts>}``. Both the knob AND the -1 are
    load-bearing, and omitting either produces a plausible-looking dud:

      * ``n_cpu_moe`` alone is NOT enough. Setting ``alloc_mode: "explicit"``
        makes ``spill.alloc_mode_env()`` non-None on the worker, which is
        exactly the gate that DISABLES ``slot_agent._build_cmd``'s auto MoE
        policy ("a k37 alloc_mode -> no auto split; the operator is driving the
        numbers themselves"). A derived default that silences the very policy
        that would have produced the right answer is the incident in miniature.
        Carrying the knob explicitly re-supplies what the gate suppressed —
        ``_build_cmd`` reads ``n_cpu_moe`` BEFORE the auto branch, so an
        explicit value wins regardless of the mode.
      * ``n_gpu_layers: -1`` is NOT redundant with it. On the SLOT path the
        agent ships only budgets / n_cpu_moe / an explicit n_gpu_layers to the
        child (``llama.runners.get``) — ``HUGPY_ALLOC_MODE`` is deliberately not
        forwarded. So without the -1 the slot child autofits a LAYER split and
        then adds an expert split on top: a hybrid, not the measured
        configuration. With both, the child takes the ``moe_mode="explicit"``
        branch at ngl=-1 — byte-identical argv to what the auto policy emits
        (``--n-gpu-layers -1 --n-cpu-moe 999``), which is the point.
      * the -1 is SAFE here precisely because the knob rides with it: the
        inverse VRAM preflight that catches "-1 onto a card that can't hold it"
        (the 5.5-hour ae stall) is explicitly skipped when an expert split is
        configured, because total-bytes-vs-VRAM is the wrong comparison for a
        split model. A bare -1 is the bug; -1 WITH the split is the fix.
      * the budgets (``gpu_mem_gib`` / ``cpu_mem_gib``) are the honest
        DECLARATION of the split's two sides — what the console shows and what
        the worker's RAM-budget preflight prices the CPU share against. They
        make the allocation self-describing rather than an opaque pair of flags.

    VERSION GATE: ``alloc_mode`` and ``n_cpu_moe`` are both in NEW_SPILL_KEYS,
    so this spill is gated by ``gate_spill_for_worker`` at the emission seam
    like any other mode spill — a pre-0.1.203 worker gets {} (max-gpu autofit),
    whose own auto MoE policy then does the right thing anyway. The fleet runs
    0.1.208, so the gate passes today. NOTHING new needed on the wire: every key
    used here is already in ``_SPILL_ENV`` on the released worker (verified:
    ``n_cpu_moe -> HUGPY_N_CPU_MOE``, cleared-when-absent so it cannot leak to
    the next model).

    DEGRADE-NOT-GUESS: every leaf requires MEASURED inputs. A missing size, a
    missing GPU/RAM total, or an unreadable MoE detail falls back to today's
    behavior (``max-gpu`` / ``{}``) — a default is NEVER derived from a guessed
    number. Note the asymmetry that makes this safe: falling back COSTS nothing
    (max-gpu on a GGUF still gets the worker's own auto split), whereas guessing
    could persist a stamp that suppresses it.

    THE "break" LEAVES (fits neither GPU nor RAM) return ``max-gpu`` / ``{}``
    — deliberately NOT a fourth state. There is no "infeasible" allocation to
    persist: the contract is that the worker refuses HONESTLY at load time with
    a message naming the real numbers, and a blank max-gpu is what lets it get
    that far. Inventing a refusal here would move the refusal away from the
    place that can explain it.
    """
    size = _as_int(model_bytes)
    gpu_total = _as_int(gpu_total_bytes)
    ram_total = _as_int(ram_total_bytes)

    # BITSANDBYTES SPECIALIZATION (operator, 2026-07-26). The lever is a
    # COMPRESSION choice, so it belongs here — at the PRICING step, before the
    # placement tree walks. Re-pricing rather than adding a branch is what makes
    # the two axes compose: a 4-bit model still goes through the SAME tree and
    # can still take a MoE split if it has one. The visible effect is the point —
    # a 67 GiB transformers model prices at ~20 GiB and flips ram-only ->
    # max-gpu, i.e. it stops being banished to RAM for a size it no longer has.
    # t159 (operator, 2026-09-14: "gpu-only when it fits VRAM, explicit for
    # MoE"): ROUTE BY STRUCTURE, not by the registry label alone. A priced
    # expert table only ever comes from a GGUF header read, so a model whose
    # engine is unknown (framework None — every owner-qualified `Org~X-GGUF`
    # key on ae) but carries ``is_moe`` IS a GGUF. Before this it fell into the
    # transformers branch: bnb re-priced a 45 GiB coder-next to "13.5 GiB", the
    # fits-test passed, and it derived gpu-only (all experts on the card) while
    # its split was fully priceable — the blanket default the todo names.
    detail = moe if isinstance(moe, dict) else None
    gguf = is_gguf_engine(engine) or bool(detail and detail.get("is_moe"))
    if bool(bnb) and not gguf and size:
        size = bnb_effective_bytes(size) or size

    def _plain(mode: str, why: str) -> dict:
        return {"mode": mode, "spill": mode_to_spill(mode), "why": why}

    # DEGRADE-NOT-GUESS gate, shared by every branch: without a measured size
    # and a measured GPU total there is no tree to walk.
    if size is None or gpu_total is None:
        return _plain("max-gpu",
                      "size or GPU total unknown — kept the max-gpu default "
                      "(degrade-not-guess: a default is never derived from a "
                      "guessed number)")

    gpu_fits = size <= _GPU_FIT_HEADROOM * gpu_total
    ram_fits = ram_total is not None and size <= ram_total

    def _gpu_else_ram(label: str) -> dict:
        """The shared 'gpu large enough? else ram large enough? else break'
        tail — identical for transformers and for dense GGUF.

        THE "gpu large enough -> gpu only" LEAF READS AS ``gpu-only``, taking
        the operator's tree at its literal word (operator order 2026-07-31:
        "change the default to gpu only rather than max-gpu"). HISTORY: from
        2026-07-25 to 2026-07-31 this leaf deliberately read max-gpu — the
        argument was that the fit test is a headroom heuristic against TOTAL
        capacity, so a contended card could turn a derived gpu-only into a
        hard OOM nobody chose. Two things changed under that argument:

          * gpu-only is STRICT and HONEST now (k55): the fits-whole test here
            is the same full-footprint pricing admission re-checks at load
            time, and a miss is an explicit refusal naming max-gpu as the
            remedy — not a llama.cpp OOM mid-load. The bust this leaf used to
            fear became a diagnosable, recoverable verdict.
          * defaults-are-promises still holds BECAUSE the leaf is guarded:
            only a model that fits the card WHOLE derives gpu-only. Oversized
            dense stays max-ram/ram-only below, MoE keeps its explicit-split
            branch, and every degrade-not-guess path still lands on max-gpu.
            The default remains a success path per (model x worker) — it just
            prefers full-card residency wherever that is genuinely available.
        """
        if gpu_fits:
            return _plain("gpu-only",
                          f"{label}: {size / _GIB_F:.2f} GiB fits the "
                          f"{gpu_total / _GIB_F:.2f} GiB GPU "
                          f"(<= {_GPU_FIT_HEADROOM:.0%} headroom) -> gpu-only "
                          "(full-card residency; operator default order "
                          "2026-07-31 — oversized models still derive a "
                          "spill-capable mode below)")
        if ram_total is None:
            return _plain("max-gpu",
                          f"{label}: too big for the GPU but RAM total is "
                          "unknown — kept max-gpu (never derive ram-only from "
                          "a guess)")
        if ram_fits:
            # PREFERENCE vs PROHIBITION (operator, 2026-07-25): "the 'max'
            # settings were intended to be indicative of a PREFERENCE for
            # spill... the 'only' designations are the truly stringent ones."
            #
            # A large DENSE GGUF that overflows the card is the case where that
            # distinction pays. It CAN use the GPU — it just cannot fit whole —
            # so llama.cpp keeps the layers that fit and spills the rest.
            # Deriving `ram-only` would FORBID the card entirely, and today's
            # cliff sweep prices that mistake: a spilled dense model runs ~36
            # tok/s with partial GPU vs ~7.5 tok/s at ngl=0. Defaulting to
            # ram-only would be a ~5x self-inflicted loss, i.e. a default that
            # promises dross.
            #
            # TRANSFORMERS ARE DIFFERENT and keep ram-only: the three
            # all-or-fail loaders cannot partially offload, so "doesn't fit the
            # GPU" really does mean RAM or nothing. Same leaf, two engines, two
            # honest answers.
            if gguf:
                return _plain("max-ram",
                              f"{label}: {size / _GIB_F:.2f} GiB exceeds the "
                              f"GPU's usable "
                              f"{_GPU_FIT_HEADROOM * gpu_total / _GIB_F:.2f} "
                              f"GiB but fits {ram_total / _GIB_F:.2f} GiB RAM "
                              "-> max-ram (RAM-first PREFERENCE; a GGUF still "
                              "keeps the layers that fit — ram-only would "
                              "forbid the card and cost ~5x throughput)")
            return _plain("ram-only",
                          f"{label}: {size / _GIB_F:.2f} GiB exceeds the GPU's "
                          f"usable {_GPU_FIT_HEADROOM * gpu_total / _GIB_F:.2f} "
                          f"GiB but fits {ram_total / _GIB_F:.2f} GiB RAM "
                          "-> ram-only (this loader is all-or-fail; it cannot "
                          "partially offload, so RAM is the only landing)")
        return _plain("max-gpu",
                      f"{label}: {size / _GIB_F:.2f} GiB fits NEITHER the GPU "
                      f"({gpu_total / _GIB_F:.2f} GiB) nor RAM "
                      f"({ram_total / _GIB_F:.2f} GiB) — no feasible mode; kept "
                      "max-gpu so the worker refuses honestly at load with the "
                      "real numbers (no invented fourth state)")

    if not gguf:
        # ── transformers / comfy: whole-tensor placement, no split exists ────
        return _gpu_else_ram("transformers")

    # ── GGUF (by engine label, or by a priced expert structure — see above) ──
    # OPERATOR OVERRIDE (2026-07-26). moe_force=False means "never split this
    # one" — the escape hatch when a derived split misbehaves; it drops to the
    # dense tail so the model still SERVES, just without the expert placement.
    # moe_force=True cannot manufacture a split for a file with no expert
    # structure (there would be nothing to place), so it never fabricates one —
    # the priced branch below still has to agree the numbers work.
    if moe_force is False:
        return _gpu_else_ram("gguf (operator forced the MoE split OFF)")
    if not (detail and detail.get("is_moe")):
        return _gpu_else_ram("dense gguf")

    non_expert = _as_int(detail.get("non_expert_bytes"))
    experts = _as_int(detail.get("expert_bytes"))
    if not non_expert or not experts:
        # Detected MoE but the split is unpriceable — degrade to the dense tail
        # rather than encode a split from numbers we don't have.
        return _gpu_else_ram("gguf (MoE detected but split unpriceable)")

    # "gpu large enough" for a MoE prices the NON-EXPERT share — the only part
    # the card ever holds under the split. This is the whole derivation.
    if non_expert <= _GPU_FIT_HEADROOM * gpu_total:
        if ram_total is not None and experts <= ram_total:
            spill = mode_to_spill(
                "explicit",
                gpu_mem_gib=round(non_expert / _GIB_F, 3),
                cpu_mem_gib=round(experts / _GIB_F, 3))
            # The two keys that make this a SPLIT rather than a label. See the
            # docstring: n_cpu_moe re-supplies the auto policy that alloc_mode
            # suppresses; the -1 is what the slot path actually forwards, and
            # is safe only because the knob rides with it.
            spill["n_cpu_moe"] = MOE_ALL_LAYERS
            spill["n_gpu_layers"] = -1
            return {
                "mode": "explicit", "spill": spill,
                "why": (f"MoE split: {non_expert / _GIB_F:.2f} GiB of "
                        f"non-expert tensors on the "
                        f"{gpu_total / _GIB_F:.2f} GiB GPU, "
                        f"{experts / _GIB_F:.2f} GiB of experts in "
                        f"{ram_total / _GIB_F:.2f} GiB RAM "
                        f"(--n-cpu-moe {MOE_ALL_LAYERS}, n_gpu_layers=-1) "
                        "-> explicit; the split IS the allocation"),
            }
        # Experts don't fit RAM (or RAM unknown) — the operator's tree falls
        # THROUGH here rather than breaking, so the dense tail decides on the
        # whole file. It will not choose gpu-only (the full model is far bigger
        # than the non-expert share that just fit), so this lands on ram-only
        # or the honest-refusal max-gpu.
        return _gpu_else_ram(
            "gguf MoE (non-expert fits GPU but experts exceed RAM)")

    # Non-expert share alone won't fit the card: no split is worth encoding.
    if ram_total is None:
        return _plain("max-gpu",
                      "gguf MoE: non-expert share exceeds the GPU and RAM total "
                      "is unknown — kept max-gpu (degrade-not-guess)")
    if ram_fits:
        # max-ram, not ram-only: this is still a GGUF, so even when the
        # non-expert share alone overflows the card the loader keeps whatever
        # layers fit. Prefer RAM; do not FORBID the GPU. (See the dense leaf:
        # ram-only on a spillable GGUF costs ~5x throughput.)
        return _plain("max-ram",
                      f"gguf MoE: even the {non_expert / _GIB_F:.2f} GiB "
                      f"non-expert share exceeds the GPU's usable "
                      f"{_GPU_FIT_HEADROOM * gpu_total / _GIB_F:.2f} GiB, but "
                      f"the whole {size / _GIB_F:.2f} GiB fits "
                      f"{ram_total / _GIB_F:.2f} GiB RAM -> max-ram "
                      "(RAM-first preference; the loader still keeps the "
                      "layers that fit)")
    return _plain("max-gpu",
                  f"gguf MoE: neither the non-expert share nor the whole "
                  f"{size / _GIB_F:.2f} GiB fits this box — no feasible mode; "
                  "kept max-gpu so the worker refuses honestly at load")


# ─────────────────────────────────────────────────────────────────────────────
# CASE A — "will not fit ANY worker, in ANY mode" (operator ruling 2026-07-25)
# ─────────────────────────────────────────────────────────────────────────────
# The operator: "models that … simply will not fit on a worker no matter what,
# if allocated, should be blocked. the user will be forced to acknowledge it and
# act or not". This is ARITHMETIC, not judgement, so it is automatic — with the
# existing /unblock as the escape hatch.
#
# WHY IT IS WORTH DOING AT ALL: today the "fits neither" leaf of
# ``default_allocation`` returns a soft max-gpu "so the worker refuses honestly
# at load". That refusal is real, but it arrives MINUTES LATER as a failed
# request rather than as visible state. Blocking makes the same fact loud, at
# the moment of allocation, with the numbers attached.
#
# THREE-VALUED BY CONSTRUCTION. ``worker_fit_verdict`` returns:
#     True   — a mode exists on this worker that can physically hold the model
#     False  — CONFIDENTLY cannot: measured size > measured GPU AND > measured RAM
#     None   — UNKNOWN: some input was missing, so this worker has NO OPINION
# DEGRADE-NOT-GUESS is the whole safety property here: only a False from at
# least one worker, and NO True from any worker, can block. A fleet of Nones
# blocks nothing — "defaults are promises", and a wrong auto-block takes a
# WORKING model out of the pool, which is strictly worse than a late refusal.


def worker_fit_verdict(engine: Any,
                       model_bytes: "Optional[int]",
                       gpu_total_bytes: "Optional[int]",
                       ram_total_bytes: "Optional[int]",
                       moe_split_gpu_bytes: "Optional[int]" = None) -> "Optional[bool]":
    """Can ``model_bytes`` land on THIS one worker in SOME allocation mode?

    ``True`` = yes (some mode fits), ``False`` = confidently no, ``None`` =
    unknown (missing data — this worker gets no vote).

    THE TEST IS DELIBERATELY THE LOOSEST ONE THAT IS STILL TRUE. A model "fits
    somewhere" if it fits the GPU **or** RAM **or** the two COMBINED — because
    max-gpu/max-ram spill across both, so combined capacity is the real
    physical ceiling, and refusing anything below it would be central inventing
    a limit the loader does not have. No headroom factor is applied here (unlike
    ``_GPU_FIT_HEADROOM`` in the DEFAULT-picking path): a headroom fudge is the
    right conservatism when choosing a default that must SUCCEED, and exactly
    the wrong one when deciding to take a model out of the pool. Erring loose
    means the worst case is a late honest load-time refusal — today's behavior —
    instead of a working model being blocked.

    NOTE the engine is accepted (and ignored) for signature parity with the rest
    of this module: combined GPU+RAM is the ceiling for GGUF (partial offload)
    and for transformers (accelerate's cpu/disk offload) alike, so no engine
    distinction is defensible at THIS question's granularity.

    ``moe_split_gpu_bytes`` (mirroring ``feasible_modes``): a detected MoE's
    GPU-resident share under the expert split (non-expert + mmproj). When
    present it is the GPU-side figure tested — the expert bytes are file-backed
    page cache (mmap-streamable), so charging the FULL file against the GPU
    would refuse exactly the model the split makes serveable (a 45 GiB MoE with
    a 1.5 GiB backbone on a 24 GiB box). The COMBINED GPU+RAM ceiling still
    applies to the WHOLE file, though (coder-next/computron 2026-08-28): the
    worker's own expert-RAM guard refuses a split whose expert tensors overflow
    budgetable host RAM, so a box whose VRAM+RAM combined cannot hold the model
    is a PERMANENT no — offering it per-request only harvests that refusal
    (computron, 8 GiB card: passed the backbone-only test, refused every
    dispatch). Loosening on the GPU side only — never past the box's physical
    ceiling."""
    size = _as_int(model_bytes)
    moe_gpu = _as_int(moe_split_gpu_bytes)
    gpu_total = _as_int(gpu_total_bytes)
    ram_total = _as_int(ram_total_bytes)
    if size is None and moe_gpu is None:
        return None                     # unsizable model: nobody may vote
    if gpu_total is None and ram_total is None:
        return None                     # unmeasured box: it has no opinion
    capacity = (gpu_total or 0) + (ram_total or 0)
    if moe_gpu is not None:
        if size is None:
            return moe_gpu <= capacity  # unsizable file: the split share votes
        return moe_gpu <= capacity and size <= capacity
    return size <= capacity


def fleet_fit_verdict(boxes: Any) -> dict:
    """Roll per-worker verdicts up into the FLEET answer.

    ``boxes`` is an iterable of dicts, each ``{"name", "engine", "model_bytes",
    "gpu_total_bytes", "ram_total_bytes"}`` (the caller resolves them; this
    stays pure). Returns::

        {"fits_somewhere": bool|None,   # True / False / None(=no confident data)
         "blockable": bool,             # True IFF it is safe to auto-block
         "fits_on": [name, ...],        # workers that CAN hold it
         "refused_by": [name, ...],     # workers that confidently CANNOT
         "unknown": [name, ...],        # workers with no opinion
         "why": str}                    # the operator-facing reasoning line

    ``blockable`` is the ONLY field a caller should act on, and it is True only
    when **at least one** worker returned a confident False and **no** worker
    returned True. An empty fleet, an all-unknown fleet, or a single confident
    "fits" anywhere all yield ``blockable=False``. That asymmetry is the
    operator's "no matter what": a model that fits SOME worker must never be
    blocked, and missing data is never evidence of anything."""
    fits_on, refused_by, unknown = [], [], []
    worst_gpu = worst_ram = 0
    size_seen = None
    for box in (boxes or []):
        if not isinstance(box, dict):
            continue
        name = str(box.get("name") or box.get("id") or "?")
        verdict = worker_fit_verdict(box.get("engine"), box.get("model_bytes"),
                                     box.get("gpu_total_bytes"),
                                     box.get("ram_total_bytes"))
        if verdict is True:
            fits_on.append(name)
        elif verdict is False:
            refused_by.append(name)
            worst_gpu = max(worst_gpu, _as_int(box.get("gpu_total_bytes")) or 0)
            worst_ram = max(worst_ram, _as_int(box.get("ram_total_bytes")) or 0)
            size_seen = size_seen or _as_int(box.get("model_bytes"))
        else:
            unknown.append(name)

    if fits_on:
        return {"fits_somewhere": True, "blockable": False, "fits_on": fits_on,
                "refused_by": refused_by, "unknown": unknown,
                "why": (f"fits {fits_on[0]}" + (f" (+{len(fits_on) - 1} more)"
                                                if len(fits_on) > 1 else "")
                        + " — never blocked while ANY worker can hold it")}
    if not refused_by:
        return {"fits_somewhere": None, "blockable": False, "fits_on": [],
                "refused_by": [], "unknown": unknown,
                "why": ("no worker could return a confident verdict "
                        f"({len(unknown)} unknown) — degrade-not-guess: "
                        "never block on missing data")}
    return {
        "fits_somewhere": False, "blockable": True, "fits_on": [],
        "refused_by": refused_by, "unknown": unknown,
        "why": (f"auto: {(size_seen or 0) / _GIB_F:.1f} GiB exceeds every "
                f"worker's GPU ({worst_gpu / _GIB_F:.1f}) and RAM "
                f"({worst_ram / _GIB_F:.1f}) — refused by "
                f"{', '.join(refused_by)}"
                + (f"; {len(unknown)} worker(s) had no data" if unknown else "")
                + " — unblock to override"),
    }


def resolve_alloc_mode(name: Any) -> "tuple[Optional[str], bool]":
    """``(canonical_mode, was_alias)`` for a mode name, canonical-first.

    Canonical names pass through untouched. A legacy alias resolves to its
    canonical mode with ``was_alias=True`` (callers log; never emit the alias
    back). Unknown/empty -> ``(None, False)`` — the caller degrades (keeps its
    default) rather than raising."""
    if name is None:
        return None, False
    s = str(name).strip().lower()
    if not s:
        return None, False
    if s in ALLOC_MODES:
        return s, False
    alias = LEGACY_ALLOC_ALIASES.get(s)
    if alias:
        return alias, True
    return None, False


def derive_alloc_mode(override: "Optional[dict]") -> str:
    """The model's EFFECTIVE mode from a persisted override/spill dict —
    read-time derivation, the migration (no file rewrite ever needed).

      * an explicit ``alloc_mode`` wins (aliases resolved);
      * else the legacy wire value: n_gpu_layers -1 -> gpu-only,
        0/"off"/"cpu"/"none" -> ram-only;
      * else an explicit-budget/band/leniency contract -> explicit
        (today's "budget"/"bands" spills ARE explicit allocations);
      * else -> max-gpu (THE DEFAULT: a blank model fits-and-spills, never
        OOMs — defaults-are-promises).
    """
    ov = override or {}
    got, was_alias = resolve_alloc_mode(ov.get("alloc_mode"))
    if got:
        if was_alias:
            logger.info("alloc_mode legacy name %r resolved to %r",
                        ov.get("alloc_mode"), got)
        return got
    ngl = ov.get("n_gpu_layers")
    if ngl is not None:
        s = str(ngl).strip().lower()
        if s == "-1":
            return "gpu-only"
        if s in ("0", "off", "cpu", "none"):
            return "ram-only"
        # positive layer count / "auto": a fit-and-spill flavor -> max-gpu
        # (the explicit layer count still rides the wire untouched).
    for k in ("leniency_pct", "gpu_mem_gib", "cpu_mem_gib",
              "gpu_mem_gib_deviation_pct", "cpu_mem_gib_deviation_pct"):
        if ov.get(k) is not None:
            return "explicit"
    return "max-gpu"


def mode_to_spill(mode: Any, *, ctx_pct: "Optional[int]" = None,
                  gpu_mem_gib: "Optional[float]" = None,
                  cpu_mem_gib: "Optional[float]" = None,
                  leniency_pct: "Optional[float]" = None,
                  priority: "Optional[int]" = None,
                  priority_device: "Optional[str]" = None,
                  explicit_pick: bool = False) -> dict:
    """Materialize a mode (+ optional explicit knobs) into the /assign spill
    contract. Legacy aliases are resolved first. Unknown mode -> {} (max-gpu),
    logged — degrade, never raise.

    ``explicit_pick`` (2026-07-25) is THE DERIVED-vs-CHOSEN switch, and it only
    affects max-gpu — the one mode whose natural encoding ({}) collides with the
    "clear this override" signal:

      * False (default) — max-gpu yields ``{}``. This is the DERIVED default's
        encoding: nothing is persisted, so the model keeps re-deriving against
        current conditions. ``default_allocation`` uses this path exclusively,
        which is why deriving max-gpu still writes nothing.
      * True — max-gpu yields ``{"alloc_mode": "max-gpu"}``: a real, persisted
        row recording that a human chose this. Non-empty, so assign_model
        WRITES it instead of reading it as a clear.

    The other four modes already have non-empty encodings and ignore the flag.
    Defaulting to False keeps every existing caller byte-identical — only the
    operator-facing /assign path opts in."""
    canonical, was_alias = resolve_alloc_mode(mode)
    if canonical is None:
        if mode not in (None, ""):
            logger.warning("unknown alloc mode %r -> defaulting to max-gpu", mode)
        canonical = "max-gpu"
    elif was_alias:
        logger.info("alloc_mode legacy name %r resolved to %r", mode, canonical)
    spill: dict = {}
    if canonical == "gpu-only":
        spill = {"n_gpu_layers": -1}
    elif canonical == "ram-only":
        return {"n_gpu_layers": "off"}          # ctx irrelevant off-GPU
    elif canonical == "max-gpu":
        # {} when DERIVED (stays unpersisted, re-derives); the explicit key when
        # a human picked it (persists, immune to the derivation moving).
        spill = {"alloc_mode": "max-gpu"} if explicit_pick else {}
    elif canonical == "max-ram":
        spill = {"alloc_mode": "max-ram"}
    elif canonical == "explicit":
        spill = {"alloc_mode": "explicit"}
        if gpu_mem_gib is not None:
            spill["gpu_mem_gib"] = float(gpu_mem_gib)
        if cpu_mem_gib is not None:
            spill["cpu_mem_gib"] = float(cpu_mem_gib)
        if leniency_pct is not None:
            spill["leniency_pct"] = float(leniency_pct)
        if priority is not None:
            spill["priority"] = int(priority)
        if priority_device is not None:
            spill["priority_device"] = str(priority_device)
    if ctx_pct is not None and canonical != "max-gpu":
        spill["ctx_pct"] = int(ctx_pct)
    return spill


def normalize_spill(spill: "Optional[dict]") -> "tuple[dict, Optional[str]]":
    """Normalize a client-supplied spill's ``alloc_mode`` value IN PLACE of the
    wire encoding: legacy aliases resolve to canonical; the two legacy-
    expressible modes (gpu-only / ram-only) are REWRITTEN onto the unchanged
    legacy ``n_gpu_layers`` wire, so ``alloc_mode`` survives here only for
    max-ram / explicit (the version-gated pair) and max-gpu.

    max-gpu KEEPS its key as of 2026-07-25 — it used to be rewritten to ``{}``,
    which is also the CLEAR signal, so an explicit max-gpu deleted its own row
    instead of persisting (see the branch comment below). The key is stripped
    again at the emission seam, so the WIRE is unchanged; this function's output
    is what gets PERSISTED, and persistence is exactly what max-gpu was missing.

    Returns ``(normalized_spill, note)`` — note is a human line when something
    was resolved/rewritten (for logs / say-why), None when untouched. Unknown
    mode values are DROPPED with a note (degrade-not-500; the rest of the
    spill still applies)."""
    if not isinstance(spill, dict) or "alloc_mode" not in spill:
        return (dict(spill) if isinstance(spill, dict) else {}), None
    out = dict(spill)
    raw = out.pop("alloc_mode")
    canonical, was_alias = resolve_alloc_mode(raw)
    if canonical is None:
        note = (f"unknown alloc_mode {raw!r} dropped (recognized: "
                f"{', '.join(ALLOC_MODES)}); rest of the spill still applies")
        logger.warning("normalize_spill: %s", note)
        return out, note
    note = (f"alloc_mode {raw!r} resolved to {canonical!r}" if was_alias else None)
    if canonical == "gpu-only":
        out["n_gpu_layers"] = -1
    elif canonical == "ram-only":
        out["n_gpu_layers"] = "off"
    elif canonical == "max-gpu":
        # EXPLICIT max-gpu KEEPS ITS KEY (bugfix 2026-07-25). It used to be
        # rewritten to {} here — but {} is ALSO the "clear this override" signal
        # (assign_model), so an operator who deliberately picked max-gpu had
        # their row DELETED instead of written, and then silently inherited
        # whatever the read-time derivation produced. That was harmless only
        # while the derivation was max-gpu for everything; once derived defaults
        # landed, a 12.29 GiB GGUF on a 7.6 GiB card derived max-ram and the
        # operator's explicit max-gpu became max-ram (the live repro).
        #
        # So the PERSISTED encoding is {"alloc_mode": "max-gpu"} — non-empty,
        # therefore structurally distinguishable from a clear. The WIRE stays
        # byte-identical to before: WorkerStore.spill_for strips this key at the
        # emission seam (see _WIRE_INERT_MODES there), so no worker ever sees it
        # and no version gate is engaged. A DERIVED max-gpu still yields {} via
        # mode_to_spill — only an operator's explicit pick persists.
        out.pop("n_gpu_layers", None)       # no layer knob: max-gpu IS autofit
        out["alloc_mode"] = "max-gpu"
    else:                                   # max-ram / explicit keep the key
        out["alloc_mode"] = canonical
    if note:
        logger.info("normalize_spill: %s", note)
    return out, note


def _ver_tuple(v: Any) -> "Optional[tuple]":
    try:
        parts = str(v).strip().split(".")
        return tuple(int(p) for p in parts) if parts else None
    except (TypeError, ValueError):
        return None


def worker_honors_mode_keys(pkg_version: Any) -> bool:
    """True when a worker's reported package version understands the NEW spill
    keys (>= MODE_MIN_PKG_VERSION). Unknown/unparseable -> False (fail SAFE:
    never ship a knob we can't prove the worker reads)."""
    have = _ver_tuple(pkg_version)
    need = _ver_tuple(MODE_MIN_PKG_VERSION)
    return have is not None and need is not None and have >= need


def worker_honors_no_evict(pkg_version: Any) -> bool:
    """True when a worker's reported package version honors the polite-load
    flag (>= NO_EVICT_MIN_PKG_VERSION). Unknown/unparseable -> False (fail
    SAFE, same rule as worker_honors_mode_keys: never promise politeness we
    can't prove the worker keeps — a dropped no_evict evicts residents)."""
    have = _ver_tuple(pkg_version)
    need = _ver_tuple(NO_EVICT_MIN_PKG_VERSION)
    return have is not None and need is not None and have >= need


def no_evict_downgrade_note(pkg_version: Any, worker_name: str = "") -> str:
    """The LOUD note for a polite load aimed at a worker that predates it."""
    return (f"worker {worker_name or '?'} (pkg {pkg_version or 'unknown'}) "
            f"predates the polite-load flag (needs >= "
            f"{NO_EVICT_MIN_PKG_VERSION}); no_evict STRIPPED — this worker "
            f"would evict residents to make room, so central does not route a "
            f"polite model here (update the worker to honor it)")


def worker_honors_chat_extras(pkg_version: Any) -> bool:
    """True when a worker's reported package version forwards the per-request
    engine chat keys (>= CHAT_EXTRAS_MIN_PKG_VERSION). Unknown/unparseable ->
    False (fail SAFE, same rule as worker_honors_mode_keys: never ship a knob
    we can't prove the worker reads)."""
    have = _ver_tuple(pkg_version)
    need = _ver_tuple(CHAT_EXTRAS_MIN_PKG_VERSION)
    return have is not None and need is not None and have >= need


def gate_chat_extras_for_worker(payload: "Optional[dict]", pkg_version: Any,
                                worker_name: str = "") -> "tuple[dict, Optional[str]]":
    """t74 version gate at the relay: a chat payload carrying
    ``chat_template_kwargs``/``logit_bias`` only ships them to a worker that
    honors them; an older worker gets the keys STRIPPED, with a note the caller
    logs/surfaces. The stripped request still carries the /no_think directive
    in its messages (utils/no_think.py appends it caller-side), so hard
    suppression degrades to the soft switch — never to a rejected request and
    never to a silent no-op.

    Returns ``(payload_to_emit, downgrade_note)``. A payload with neither key
    passes through untouched (None note) regardless of version."""
    p = dict(payload or {})
    present = sorted(set(p) & CHAT_EXTRAS_WIRE_KEYS)
    if not present or worker_honors_chat_extras(pkg_version):
        return p, None
    for k in present:
        p.pop(k, None)
    note = (f"worker {worker_name or '?'} (pkg {pkg_version or 'unknown'}) "
            f"predates per-request engine chat keys (needs >= "
            f"{CHAT_EXTRAS_MIN_PKG_VERSION}); {', '.join(present)} STRIPPED — "
            f"hard no-think degrades to the /no_think directive for this "
            f"request (update the worker to honor it)")
    return p, note


def gate_spill_for_worker(spill: "Optional[dict]", pkg_version: Any,
                          worker_name: str = "") -> "tuple[dict, Optional[str]]":
    """THE version gate at emission: a spill carrying NEW mode keys is only
    sent to a worker that honors them; an older worker gets max-gpu ({}
    autofit) for that request, with a note the caller logs/surfaces.

    Returns ``(spill_to_emit, downgrade_note)``. A spill with no new keys
    passes through untouched (None note) regardless of version.

    k56: ``no_evict`` is gated SEPARATELY (its own, later min version) and is
    stripped rather than collapsing the spill — see NO_EVICT_SPILL_KEY. Checked
    first so a polite spill carrying no mode keys at all is still gated."""
    s = dict(spill or {})
    polite_note = None
    if s.get(NO_EVICT_SPILL_KEY) and not worker_honors_no_evict(pkg_version):
        s.pop(NO_EVICT_SPILL_KEY, None)
        polite_note = no_evict_downgrade_note(pkg_version, worker_name)
    if not (set(s) & NEW_SPILL_KEYS):
        return s, polite_note
    if worker_honors_mode_keys(pkg_version):
        return s, polite_note
    mode = s.get("alloc_mode") or "explicit"
    note = (f"worker {worker_name or '?'} (pkg {pkg_version or 'unknown'}) "
            f"predates allocation-mode spill keys (needs >= "
            f"{MODE_MIN_PKG_VERSION}); '{mode}' downgraded to max-gpu "
            f"(autofit) for this request — update the worker to honor it")
    # A worker old enough to miss the mode keys misses politeness too; report
    # BOTH downgrades or the operator fixes one and re-hits the other.
    return {}, "; ".join(n for n in (polite_note, note) if n)
