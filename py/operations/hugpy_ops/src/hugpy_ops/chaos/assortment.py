"""Enumerate the live assortment and draw combos (seeded, reproducible).

The assortment cube is: servable models x cards (workers) x alloc modes x ctx%.
A combo is one point in that cube, constrained to what is actually EXERCISABLE
and REVERSIBLE:

  * a model is servable only if it is a chat model (text-generation /
    image-text-to-text) and NOT operator-blocked (``blocked: true`` in /models);
  * a combo targets workers the model is ALREADY assigned to — the runner only
    changes an existing spill and restores it, never creates/removes a
    designation (pins are routing-only, but staying on already-assigned pairs
    keeps every trial trivially reversible);
  * alloc modes are framework-gated: a GGUF model gets the full FIVE-mode set
    (k37: gpu-only / ram-only / max-gpu / max-ram / explicit); a non-GGUF
    (transformers) model gets the FOUR non-explicit modes (gpu-only / ram-only
    / max-gpu / max-ram) — max-ram was opened for non-GGUF 2026-07-24
    (transformers RAM-priority max_memory, diffusers cpu-offload); only
    ``explicit`` stays refused at /assign by the engine gate (its banded
    leniency floor has no transformers analogue), which the runner honours.

Nothing here talks to the network — it consumes /models + /llm/workers payloads
so it is unit-testable with fixtures."""
from __future__ import annotations

import random

from hugpy_ops.chaos.schema import ALLOC_MODES, CTX_PCTS
from hugpy_engine.alloc_modes import NONGGUF_ALLOWED_MODES, resolve_alloc_mode

CHAT_TASKS = frozenset({"text-generation", "image-text-to-text",
                        "text2text-generation"})

# Framework gating (k37): GGUF gets all five modes; a transformers model is
# engine-gated at /assign to the four non-explicit modes (gpu-only / ram-only /
# max-gpu / max-ram), so only offer those (explicit on a non-GGUF would be a
# guaranteed 409 — its banded leniency floor has no transformers analogue).
GGUF_MODES = ALLOC_MODES
NONGGUF_MODES = NONGGUF_ALLOWED_MODES

# Hybrid feasibility margin: predicted need must fit under margin*(vram+ram) of
# at least one candidate box, else the combo is predicted-infeasible (skip,
# never fire). Coarse and safe — matches the operator's "exceeds a box's total
# even hybrid" rule with headroom.
FEASIBLE_MARGIN = 0.90

GIB = 1 << 30


def servable_models(models: list[dict],
                    max_model_bytes: int | None = None) -> list[dict]:
    """Chat models that are not operator-blocked, with a normalized view.

    ``max_model_bytes`` (optional) caps the exercised assortment to models whose
    weights are at or below the cap — the polite-guest knob for a live proof on a
    fleet whose big models share a busy card."""
    out = []
    for m in models:
        if m.get("blocked"):
            continue
        tasks = set(m.get("tasks") or [])
        if m.get("primary_task"):
            tasks.add(m["primary_task"])
        if not (tasks & CHAT_TASKS):
            continue
        eb = int(m.get("effective_bytes") or m.get("size_bytes") or 0)
        if max_model_bytes is not None and eb > max_model_bytes:
            continue
        out.append({
            "model_key": m.get("model_key"),
            "framework": m.get("framework"),
            "effective_bytes": int(m.get("effective_bytes")
                                    or m.get("size_bytes") or 0),
            "ctx_max": m.get("model_max_length"),
            "tasks": sorted(tasks & CHAT_TASKS),
        })
    return out


def worker_index(workers: list[dict]) -> dict:
    """Map name -> {id, vram_total, ram_total, assigned:set, warm:set}."""
    idx = {}
    for w in workers:
        name = w.get("name") or w.get("hostname") or (w.get("id") or "")[:8]
        idx[name] = {
            "id": w.get("id"),
            "name": name,
            "status": w.get("status"),
            "vram_total": w.get("vram_total"),
            "ram_total": w.get("ram_total"),
            "assigned": set(w.get("models") or []),
            "warm": set(w.get("loaded_models") or []),
        }
    return idx


def candidate_workers(model_key: str, widx: dict) -> list[str]:
    """Online workers the model is already assigned to (spill targets)."""
    return sorted(
        name for name, w in widx.items()
        if model_key in w["assigned"] and w.get("status") == "online")


def modes_for(framework: str) -> tuple:
    return GGUF_MODES if (framework or "").lower() == "gguf" else NONGGUF_MODES


def enumerate_assortment(models: list[dict], workers: list[dict],
                         max_model_bytes: int | None = None) -> dict:
    """A structured, JSON-able summary of the live cube (for --dry-run and the
    run manifest). Reports every servable model with its candidate workers and
    allowed modes, plus the blocked models it excluded."""
    widx = worker_index(workers)
    sm = servable_models(models, max_model_bytes)
    blocked = sorted(m.get("model_key") for m in models if m.get("blocked"))
    rows = []
    for m in sm:
        mk = m["model_key"]
        cands = candidate_workers(mk, widx)
        rows.append({
            "model_key": mk, "framework": m["framework"],
            "effective_bytes": m["effective_bytes"],
            "candidate_workers": cands,
            "alloc_modes": list(modes_for(m["framework"])),
            "warm_on": sorted(w for w in cands if mk in widx[w]["warm"]),
            "exercisable": bool(cands),
        })
    exercisable = [r for r in rows if r["exercisable"]]
    combos = 0
    for r in exercisable:
        combos += len(r["candidate_workers"]) * len(r["alloc_modes"]) * len(CTX_PCTS)
    return {
        "workers": [{"name": n, "id": w["id"], "status": w["status"],
                     "vram_total": w["vram_total"], "ram_total": w["ram_total"],
                     "n_assigned": len(w["assigned"])}
                    for n, w in sorted(widx.items())],
        "n_models_total": len(models),
        "n_servable": len(sm),
        "n_exercisable": len(exercisable),
        "blocked_excluded": blocked,
        "ctx_pcts": list(CTX_PCTS),
        "alloc_modes": list(ALLOC_MODES),
        "approx_combo_space": combos,
        "models": rows,
    }


def _budget_gib_for(need_bytes: int | None, card_gib: float | None,
                    rng: random.Random) -> float:
    """Draw a plausible explicit gpu_mem_gib budget: a fraction of the card,
    biased near the model's predicted need so the trial actually exercises a
    fit/partial boundary rather than a trivially-huge budget."""
    ceil = card_gib or 8.0
    choices = [round(x, 1) for x in (ceil * 0.25, ceil * 0.5, ceil * 0.75)]
    if need_bytes:
        choices.append(round(need_bytes / GIB, 1))
    choices = sorted({max(0.5, min(ceil, c)) for c in choices})
    return rng.choice(choices)


def build_spill(mode: str, ctx_pct: int, budget_gib: float | None,
                rng: random.Random) -> dict:
    """Construct the /assign spill for a drawn (mode, ctx_pct) — the k37
    five-mode wire encoding. ctx_pct rides the spill (validated 1..100) on the
    GPU-bearing tunable modes so the ctx dimension is real; ram-only omits it
    (irrelevant off-GPU) and max-gpu stays ZERO-knob (the operator's
    cut-and-dry autofit — {}).

    Legacy mode names (autofit / cpu-only / old max-gpu semantics via
    "budget"/"bands") are accepted and resolved through the alias table —
    accepted on input, never emitted back."""
    canonical, _alias = resolve_alloc_mode(mode)
    mode = canonical or "max-gpu"
    if mode == "max-gpu":
        spill: dict = {}                    # autofit, zero knobs — by design
    elif mode == "gpu-only":
        spill = {"n_gpu_layers": -1, "ctx_pct": int(ctx_pct)}
    elif mode == "ram-only":
        spill = {"n_gpu_layers": "off"}
    elif mode == "max-ram":
        spill = {"alloc_mode": "max-ram", "ctx_pct": int(ctx_pct)}
    elif mode == "explicit":
        spill = {
            "alloc_mode": "explicit",
            "gpu_mem_gib": budget_gib,
            "leniency_pct": rng.choice([10.0, 25.0, 50.0]),
            "ctx_pct": int(ctx_pct),
            "ctx_deviation_pct": rng.choice([10.0, 25.0, 50.0]),
            "priority": rng.choice([0, 0, 1]),  # bias to normal
        }
    else:  # pragma: no cover — modes come from ALLOC_MODES
        spill = {}
    return spill


def draw_combo(rng: random.Random, models: list[dict], workers: list[dict],
               max_model_bytes: int | None = None) -> dict | None:
    """Draw ONE chaotic combo from the exercisable cube, or None if empty.

    Deterministic for a given (rng state, assortment): the sequence of draws is
    reproducible from the seed. Returns a combo dict the runner consumes."""
    widx = worker_index(workers)
    sm = servable_models(models, max_model_bytes)
    pool = [(m, candidate_workers(m["model_key"], widx)) for m in sm]
    pool = [(m, c) for m, c in pool if c]
    if not pool:
        return None
    m, cands = rng.choice(pool)
    mk = m["model_key"]
    mode = rng.choice(modes_for(m["framework"]))
    ctx_pct = rng.choice(CTX_PCTS)
    # smallest candidate card governs the budget draw
    card_gibs = [((widx[c]["vram_total"] or 0) / GIB) for c in cands
                 if widx[c].get("vram_total")]
    card_gib = min(card_gibs) if card_gibs else None
    budget_gib = _budget_gib_for(m["effective_bytes"], card_gib, rng)
    spill = build_spill(mode, ctx_pct, budget_gib, rng)
    warm_on = sorted(c for c in cands if mk in widx[c]["warm"])
    # combo.ctx_pct == what the spill actually encodes (None for max-gpu/
    # ram-only where ctx is not a controlled dimension), so the record never
    # claims a ctx target it didn't apply.
    applied_ctx = spill.get("ctx_pct")
    return {
        "model_key": mk, "framework": m["framework"],
        "effective_bytes": m["effective_bytes"],
        "alloc_mode": mode, "spill": spill,
        "ctx_pct": (int(applied_ctx) if applied_ctx is not None else None),
        "target_workers": cands,
        "was_warm": bool(warm_on), "warm_on": warm_on,
        "_card_gib": card_gib,
    }


def feasibility(need_bytes: int | None, cands: list[str], workers: list[dict]
                ) -> dict:
    """Per-candidate hybrid feasibility. Returns {feasible, infeasible_reason,
    per_worker}. A box is feasible-hybrid when need_bytes <= margin*(vram+ram)
    — for a CPU-only box that is just RAM. Unknown need fails OPEN (feasible)
    so an unmeasurable model is still exercised (the admission gate itself fails
    open the same way)."""
    widx = worker_index(workers)
    per = {}
    any_feasible = False
    for c in cands:
        w = widx.get(c, {})
        vram = int(w.get("vram_total") or 0)
        ram = int(w.get("ram_total") or 0)
        hybrid = vram + ram
        if not need_bytes or not hybrid:
            feasible = True  # fail open
        else:
            feasible = need_bytes <= FEASIBLE_MARGIN * hybrid
        any_feasible = any_feasible or feasible
        per[c] = {"vram_total": vram or None, "ram_total": ram or None,
                  "hybrid_total": hybrid or None, "feasible_hybrid": feasible}
    reason = None
    if cands and not any_feasible:
        reason = (f"predicted need {need_bytes} bytes exceeds "
                  f"{int(FEASIBLE_MARGIN*100)}% of every candidate box's "
                  f"vram+ram hybrid total")
    return {"feasible": (any_feasible if cands else False),
            "infeasible_reason": reason, "per_worker": per}
