"""Observation schema for the chaos-and-learn exerciser (p1, EPOCH CLOSER).

THIS IS THE CONTRACT between the chaos *runner* (which drives real trials on the
live fleet) and the t28 *learner* (a sibling module that calibrates placement
templates from measured reality). The learner reads the JSONL this schema
describes; keep it stable and versioned. Coordinate schema changes via
``SCHEMA_VERSION`` and ``chaos/SCHEMA.md`` — never silently reshape a field a
consumer already parses.

Every trial emits exactly ONE observation object. Each observation pairs a
PREDICTED side (what central/the operator intended, priced cheaply BEFORE the
load) against a MEASURED side (what actually happened once the model loaded and
served). The learner prices ``predicted.need_bytes`` against
``measured.allocation.vram_bytes`` / ``rss_bytes`` and learns the correction.

Top-level shape
---------------
{
  "schema_version": "chaos-obs/1",
  "run_id":   str,           # one chaos run
  "trial_id": str,           # unique per observation
  "seed":     int,           # the run's RNG seed (reproducible draws)
  "round":    int,           # 0-based round index
  "ts_start": float, "ts_end": float, "duration_s": float,
  "kind":     "trial" | "skip",
  "skip_reason": null | "predicted-infeasible" | "back-off-foreign-jobs"
                 | "health-degraded" | "alloc-refused" | "no-servable-worker"
                 | "assortment-empty" | "stopped",
  "back_off": bool,          # true when the round yielded to real traffic

  "combo": {                 # the drawn point in the assortment cube
    "model_key":       str,
    "framework":       str,           # gguf | transformers | ...
    "effective_bytes": int,           # weights on disk (from /models)
    "alloc_mode":      str,           # gpu-only|ram-only|max-gpu|max-ram|explicit
    "spill":           dict,          # the EXACT /assign spill applied
    "ctx_pct":         int | null,    # context target (% of model max)
    "target_workers":  [str, ...],    # workers the spill was written to
    "was_warm":        bool,          # model already resident before the fire
    "warm_on":         [str, ...]
  },

  "predicted": {             # priced BEFORE firing, from /models/<key>/meta
    "need_bytes":          int | null,   # weights + KV @ ctx_pct
    "needs_weights_bytes": int | null,   # weights (effective GGUF / dir)
    "needs_kv_bytes":      int | null,   # need_bytes - weights (KV estimate)
    "ctx_pct":             int | null,
    "ctx_resolved":        int | null,
    "ctx_max":             int | null,
    "params_b":            float | null,
    "quant":               str | null,
    "placement_mode":      str,          # == combo.alloc_mode
    "band": {                            # t21 tolerance-band intent, or null
      "gpu_mem_gib": float, "gpu_mem_gib_deviation_pct": float,
      "ctx_deviation_pct": float, "priority": int } | null,
    "per_worker": {                      # candidate workers + fit advice
      "<worker>": {
        "vram_total": int|null, "ram_total": int|null,
        "hybrid_total": int|null,        # vram_total + ram_total
        "feasible_hybrid": bool,         # need_bytes <= margin*hybrid_total
        "advice": {                      # /meta?vram_gib= offload advice
          "fits_vram": bool|null, "n_gpu_layers": int|"auto"|null,
          "reason": str|null } | null } },
    "feasible":          bool,           # ANY candidate feasible-hybrid
    "infeasible_reason": str | null
  },

  "measured": {              # observed AFTER firing /chat/stream
    "served_worker":    str | null,      # from the stream status / job row
    "outcome":          str,             # done|error|refused|held|load-timeout|
                                         # closed_no_token|client_exception
    "error":            str | null,      # verbatim error text
    "finish_reason":    str | null,
    "ttft_s":           float | null,    # time to first token
    "load_duration_s":  float | null,    # awaiting-load -> first token/done
    "wall_s":           float | null,
    "tokens":           int | null,
    "stages":           [ [stage, worker_name, progress], ... ],
    "allocation": {                      # the served worker's row for the model
      "kind": str,                       # slot (GPU) | ram (CPU)
      "device": str|null, "endpoint": str|null, "slot_id": str|null,
      "vram_bytes": int|null, "rss_bytes": int|null,
      "n_gpu_layers": int|null, "total_layers": int|null,
      "ctx": int|null, "serving": bool|null, "busy": bool|null } | null,
    "loaded_detail":    dict | null,     # loaded_detail[model_key] verbatim
    "admission": {
      "verdict": "proceed"|"cpu"|"flex"|"evicted"|"partial"|"refuse"|"unknown",
      "partial_offload_considered": dict | null,   # verbatim from refusal
      "refusal_reason":             dict | null,    # verbatim _vram_evict_to_fit
                                                    # reason (needs_bytes,
                                                    # needs_weights_bytes,
                                                    # needs_kv_bytes, ctx_pct,
                                                    # protected, evicted, ...)
      "vram_evictions_delta":       int | null
    },
    "worker_state": {                    # served worker, nvidia-smi-derived
      "vram_total": int|null, "vram_free": int|null, "vram_used": int|null,
      "ram_total": int|null, "free_ram": int|null,
      "gpu_memory_free": int|null, "last_load_error": (str|dict)|null }
  },

  "restore": {               # snapshot-before / restore-after proof
    "ok": bool,
    "per_worker": {
      "<worker>": { "before": dict|None, "after": dict|None, "matches": bool } }
  }
}
"""
from __future__ import annotations

SCHEMA_VERSION = "chaos-obs/1"

# The alloc modes the runner draws from — the FIVE flat operator modes (k37),
# single-sourced from managers.alloc_modes. GGUF models get the full set; a
# non-GGUF (transformers) model is engine-gated to the three coarse placement
# intents (gpu-only / ram-only / max-gpu) — max-ram/explicit on a non-GGUF is
# refused at /assign (fine-grained placement is GGUF-only until Slice C),
# which the runner honours. Legacy names (autofit/cpu-only/budget/bands) are
# still accepted by build_spill via the alias table, never emitted.
from hugpy_engine.alloc_modes import ALLOC_MODES

# The ctx% dimension of the assortment cube (percent of a model's max context).
CTX_PCTS = (25, 50, 75, 100)

# ── sweep mode (k7 offload speed-cliff) ──────────────────────────────────────
# The DETERMINISTIC VRAM-share sweep grid: each point is a % of a model's TOTAL
# resident requirement (weights + KV @ the sweep ctx) that the gpu_mem_gib budget
# caps to VRAM — the rest spills to host RAM. 100% first (== "as full as the card
# allows"), stepping down to find where tok/s collapses.
SWEEP_VRAM_PCTS = (100, 85, 70, 55, 40, 25)
# Coarser grid for big models (each point is a cold reload of many GB) — the
# runner picks this when a model's requirement exceeds SWEEP_BIG_MODEL_GIB.
SWEEP_VRAM_PCTS_COARSE = (100, 70, 40)
SWEEP_BIG_MODEL_GIB = 20.0

# The sweep block appended to each point's chaos-obs/1 observation (additive; a
# non-sweep chaos trial carries sweep=None). vram_share_pct is the REQUESTED %;
# actual_gpu_pct / actual_n_gpu_layers are what the serving contract MEASURED.
REQUIRED_SWEEP_KEYS = (
    "vram_share_pct", "requested_gib", "actual_n_gpu_layers", "actual_gpu_pct",
    "ttft_s", "tokens_per_s", "load_s",
)

# Recognised /assign spill keys (mirror worker_routes._ALLOC_SPILL_KEYS). The
# runner never sends a key outside this set, so a typo can't write a no-op
# contract that the route would 400.
SPILL_KEYS = frozenset({
    "n_gpu_layers", "gpu_mem_gib", "cpu_mem_gib", "threads", "tensor_split",
    "gpu_mem_gib_deviation_pct", "cpu_mem_gib_deviation_pct",
    "ctx_pct", "ctx_deviation_pct", "priority",
    # k37 allocation-mode keys (max-ram / explicit ride these; version-gated
    # at central's emission seam so they are never a dead knob on an old worker).
    "alloc_mode", "leniency_pct", "priority_device",
})

# Skip reasons — the runner records a skip observation rather than firing.
SKIP_REASONS = frozenset({
    "predicted-infeasible", "back-off-foreign-jobs", "health-degraded",
    "alloc-refused", "no-servable-worker", "assortment-empty", "stopped",
    # sweep-mode (k7) skips — the point was NOT fired; recorded for honesty.
    "back-off-reservation", "back-off-headroom", "serving-busy",
    "budget-exhausted", "meta-unavailable", "unload-failed",
})

# Every observation MUST carry these top-level keys (completeness invariant the
# schema test enforces).
REQUIRED_TOP_KEYS = (
    "schema_version", "run_id", "trial_id", "seed", "round",
    "ts_start", "ts_end", "duration_s", "kind", "skip_reason", "back_off",
    "combo", "predicted", "measured", "restore",
)
REQUIRED_COMBO_KEYS = (
    "model_key", "framework", "effective_bytes", "alloc_mode", "spill",
    "ctx_pct", "target_workers", "was_warm", "warm_on",
)
REQUIRED_PREDICTED_KEYS = (
    "need_bytes", "needs_weights_bytes", "needs_kv_bytes", "ctx_pct",
    "ctx_resolved", "ctx_max", "params_b", "quant", "placement_mode",
    "band", "per_worker", "feasible", "infeasible_reason",
)
REQUIRED_MEASURED_KEYS = (
    "served_worker", "outcome", "error", "finish_reason", "ttft_s",
    "load_duration_s", "wall_s", "tokens", "stages", "allocation",
    "loaded_detail", "admission", "worker_state",
)
REQUIRED_ADMISSION_KEYS = (
    "verdict", "partial_offload_considered", "refusal_reason",
    "vram_evictions_delta",
)


def blank_observation() -> dict:
    """A fully-formed observation with every required key present and null-ish.

    The runner fills this in; using a single constructor guarantees schema
    completeness even on the earliest skip/error path (so the learner never
    trips over a missing key)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": None, "trial_id": None, "seed": None, "round": None,
        "ts_start": None, "ts_end": None, "duration_s": None,
        "kind": "trial", "skip_reason": None, "back_off": False,
        "combo": {
            "model_key": None, "framework": None, "effective_bytes": None,
            "alloc_mode": None, "spill": {}, "ctx_pct": None,
            "target_workers": [], "was_warm": False, "warm_on": [],
        },
        "predicted": {
            "need_bytes": None, "needs_weights_bytes": None,
            "needs_kv_bytes": None, "ctx_pct": None, "ctx_resolved": None,
            "ctx_max": None, "params_b": None, "quant": None,
            "placement_mode": None, "band": None, "per_worker": {},
            "feasible": None, "infeasible_reason": None,
        },
        "measured": {
            "served_worker": None, "outcome": None, "error": None,
            "finish_reason": None, "ttft_s": None, "load_duration_s": None,
            "wall_s": None, "tokens": None, "stages": [],
            "allocation": None, "loaded_detail": None,
            "admission": {
                "verdict": "unknown", "partial_offload_considered": None,
                "refusal_reason": None, "vram_evictions_delta": None,
            },
            "worker_state": {},
        },
        "restore": {"ok": None, "per_worker": {}},
        # Optional sweep block (k7). None for a chaos trial; blank_sweep() for a
        # sweep point. Additive to chaos-obs/1 — not a REQUIRED_TOP_KEY.
        "sweep": None,
    }


def blank_sweep() -> dict:
    """A fully-formed sweep block (every REQUIRED_SWEEP_KEY present, null-ish).

    ``vram_share_pct`` is the REQUESTED share of the model's total requirement;
    ``requested_gib`` is the gpu_mem_gib budget that encodes it (clamped to a safe
    fraction of the card's live free VRAM); the ``actual_*`` fields are read back
    from the serving contract; ``tokens_per_s`` is the median of the timed runs."""
    return {
        "vram_share_pct": None,     # requested % of total requirement (the grid)
        "requested_gib": None,      # gpu_mem_gib budget applied (clamped)
        "clamped": False,           # true if requested_gib was capped to safe VRAM
        "denom_need_bytes": None,   # weights+KV @ sweep ctx (the 100% denominator)
        "ctx_pct": None,            # fixed sweep context %
        "card_vram_bytes": None,    # the worker's vram_total
        "actual_n_gpu_layers": None,
        "actual_total_layers": None,
        "actual_gpu_pct": None,     # n_gpu_layers / total_layers * 100 (measured)
        "actual_vram_bytes": None,  # measured serving-contract VRAM footprint
        "actual_vram_share_pct": None,  # actual_vram_bytes / denom_need_bytes * 100
        "load_s": None,             # cold-load duration (warm-up fire)
        "ttft_s": None,             # median time-to-first-token over timed runs
        "tokens_per_s": None,       # median decode rate over timed runs
        "runs": [],                 # per-timed-run [{ttft_s, tokens_per_s, tokens}]
        "warmup_outcome": None,     # outcome of the excluded warm-up fire
        "floor_rejected": None,     # true if partial plan hit min_offload_frac floor
    }


def validate_observation(obs: dict) -> list[str]:
    """Return a list of schema-completeness problems (empty == valid).

    Used by the schema test and by the runner as a cheap self-check before it
    appends a line — a malformed observation is a bug that must not silently
    poison the learner's training set."""
    problems: list[str] = []
    if obs.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version != {SCHEMA_VERSION}")
    for k in REQUIRED_TOP_KEYS:
        if k not in obs:
            problems.append(f"missing top key: {k}")
    for k in REQUIRED_COMBO_KEYS:
        if k not in (obs.get("combo") or {}):
            problems.append(f"missing combo key: {k}")
    for k in REQUIRED_PREDICTED_KEYS:
        if k not in (obs.get("predicted") or {}):
            problems.append(f"missing predicted key: {k}")
    measured = obs.get("measured") or {}
    for k in REQUIRED_MEASURED_KEYS:
        if k not in measured:
            problems.append(f"missing measured key: {k}")
    for k in REQUIRED_ADMISSION_KEYS:
        if k not in (measured.get("admission") or {}):
            problems.append(f"missing admission key: {k}")
    if obs.get("kind") == "skip" and not obs.get("skip_reason"):
        problems.append("kind=skip but skip_reason is empty")
    if obs.get("skip_reason") and obs["skip_reason"] not in SKIP_REASONS:
        problems.append(f"unknown skip_reason: {obs['skip_reason']}")
    # The sweep block is optional (None on a chaos trial); when present it must be
    # complete so the cliff analyser never trips over a missing key.
    sweep = obs.get("sweep")
    if sweep is not None:
        if not isinstance(sweep, dict):
            problems.append("sweep present but not a dict")
        else:
            for k in REQUIRED_SWEEP_KEYS:
                if k not in sweep:
                    problems.append(f"missing sweep key: {k}")
    return problems
