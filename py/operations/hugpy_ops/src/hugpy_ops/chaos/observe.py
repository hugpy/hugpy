"""Build the predicted and measured sides of an observation.

PREDICTED is priced cheaply BEFORE firing from /models/<key>/meta (central's own
estimator) — never by reimplementing worker need-pricing (that is the learner's
/ worker's turf; do not fork it here).

MEASURED is read AFTER firing from /llm/workers: the served worker's allocation
row (the serving contract's vram_bytes/rss/n_gpu_layers), nvidia-smi-derived
worker fields, and — when a load is refused — the verbatim ``_vram_evict_to_fit``
refusal reason (needs_bytes / needs_weights_bytes / needs_kv_bytes / ctx_pct /
protected / evicted / partial_offload_considered), captured wherever it surfaces
(the stream error or the worker's last_load_error)."""
from __future__ import annotations

import json

GIB = 1 << 30


def build_predicted(client, combo: dict, workers: list[dict]) -> dict:
    """Price the predicted side from /models/<key>/meta. Uses a per-candidate
    ?vram_gib= query to capture central's offload advice per card. Fails soft:
    a missing meta yields nulls, never an exception."""
    from hugpy_ops.chaos.assortment import feasibility, worker_index

    mk = combo["model_key"]
    ctx_pct = combo.get("ctx_pct")
    widx = worker_index(workers)
    cands = combo.get("target_workers") or []

    meta = client.model_meta(mk, ctx_pct=ctx_pct)
    rec = (meta.get("recommended") or {}) if isinstance(meta, dict) else {}
    need = rec.get("need_bytes")
    weights = combo.get("effective_bytes") or meta.get("size_bytes")
    kv = None
    if isinstance(need, int) and isinstance(weights, int) and need >= weights:
        kv = need - weights

    per_worker = {}
    for c in cands:
        w = widx.get(c, {})
        vram_gib = ((w.get("vram_total") or 0) / GIB) or None
        advice = None
        if vram_gib:
            am = client.model_meta(mk, vram_gib=round(vram_gib, 2), ctx_pct=ctx_pct)
            arec = (am.get("recommended") or {}) if isinstance(am, dict) else {}
            advice = {"fits_vram": arec.get("fits_vram"),
                      "n_gpu_layers": arec.get("n_gpu_layers"),
                      "reason": arec.get("reason")}
        per_worker[c] = {"advice": advice}

    feas = feasibility(need if isinstance(need, int) else None, cands, workers)
    for c, fw in feas["per_worker"].items():
        per_worker.setdefault(c, {}).update(fw)

    band = None
    spill = combo.get("spill") or {}
    if combo.get("alloc_mode") in ("bands", "explicit"):   # k37 rename (bands->explicit)
        band = {k: spill.get(k) for k in (
            "gpu_mem_gib", "gpu_mem_gib_deviation_pct",
            "ctx_deviation_pct", "priority", "leniency_pct")}

    return {
        "need_bytes": need if isinstance(need, int) else None,
        "needs_weights_bytes": weights if isinstance(weights, int) else None,
        "needs_kv_bytes": kv,
        "ctx_pct": ctx_pct,
        "ctx_resolved": rec.get("ctx"),
        "ctx_max": meta.get("ctx_max"),
        "params_b": meta.get("params_b"),
        "quant": meta.get("quant"),
        "placement_mode": combo.get("alloc_mode"),
        "band": band,
        "per_worker": per_worker,
        "feasible": feas["feasible"],
        "infeasible_reason": feas["infeasible_reason"],
    }


def _find_allocation(worker_row: dict, model_key: str) -> dict | None:
    for a in (worker_row.get("allocations") or []):
        if a.get("model_key") == model_key:
            return a
    return None


def _parse_refusal(*sources) -> dict | None:
    """Pull the structured refusal reason out of whatever surfaced it: a dict
    (worker last_load_error), or a JSON string embedded in an error message."""
    for s in sources:
        if isinstance(s, dict):
            if s.get("state") == "refused" or "needs_bytes" in s or "reason" in s:
                return s
        if isinstance(s, str) and s:
            # try to locate a JSON object inside the string
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    d = json.loads(s[start:end + 1])
                    if isinstance(d, dict) and (
                            d.get("state") == "refused" or "needs_bytes" in d):
                        return d
                except Exception:
                    pass
    return None


def _infer_verdict(outcome: str, alloc: dict | None, refusal: dict | None,
                   evictions_delta: int | None) -> str:
    if refusal or outcome == "refused":
        return "refuse"
    if alloc is None:
        return "unknown"
    kind = alloc.get("kind")
    if kind == "ram":
        return "cpu"
    if kind == "slot":
        ngl = alloc.get("n_gpu_layers")
        total = alloc.get("total_layers")
        if isinstance(ngl, int) and isinstance(total, int) and 0 <= ngl < total:
            return "partial"
        if evictions_delta and evictions_delta > 0:
            return "evicted"
        return "proceed"
    return "unknown"


def build_measured(terminal: dict, workers_after: list[dict],
                   model_key: str, evictions_before: dict,
                   jobs_after: dict | None = None) -> dict:
    """Assemble the measured side from the stream terminal record + a fresh
    /llm/workers read (+ optional /llm/jobs for the job's final row)."""
    served = terminal.get("served_worker")
    # fall back to the job row's worker if the stream didn't name one
    if not served and jobs_after:
        for j in (jobs_after.get("jobs") or []):
            if j.get("worker"):
                served = j.get("worker")
                break

    by_name = {w.get("name"): w for w in workers_after}
    # last-resort attribution (esp. refusals that error before naming a worker):
    # find a worker whose allocation / last_load_error references this model.
    if not served:
        for w in workers_after:
            if _find_allocation(w, model_key):
                served = w.get("name"); break
            lle = w.get("last_load_error")
            if isinstance(lle, dict) and lle.get("model_key") == model_key:
                served = w.get("name"); break
            if isinstance(lle, str) and model_key in lle:
                served = w.get("name"); break


    wrow = by_name.get(served) or {}
    alloc = _find_allocation(wrow, model_key) if wrow else None
    loaded_detail = (wrow.get("loaded_detail") or {}).get(model_key) if wrow else None
    last_load_error = wrow.get("last_load_error") if wrow else None

    refusal = _parse_refusal(last_load_error, terminal.get("error"), loaded_detail)
    partial_considered = refusal.get("partial_offload_considered") if refusal else None

    ev_delta = None
    if served is not None:
        before = evictions_before.get(served)
        after = wrow.get("vram_evictions")
        if isinstance(before, int) and isinstance(after, int):
            ev_delta = after - before

    verdict = _infer_verdict(terminal.get("outcome"), alloc, refusal, ev_delta)

    allocation = None
    if alloc:
        allocation = {
            "kind": alloc.get("kind"), "device": alloc.get("device"),
            "endpoint": alloc.get("endpoint"), "slot_id": alloc.get("slot_id"),
            "vram_bytes": alloc.get("vram_bytes"),
            "rss_bytes": alloc.get("rss_bytes"),
            "n_gpu_layers": alloc.get("n_gpu_layers"),
            "total_layers": alloc.get("total_layers"),
            "ctx": alloc.get("ctx"), "serving": alloc.get("serving"),
            "busy": alloc.get("busy"),
        }

    worker_state = {}
    if wrow:
        gpus = wrow.get("gpus") or []
        worker_state = {
            "vram_total": wrow.get("vram_total"),
            "vram_free": wrow.get("vram_free"),
            "vram_used": wrow.get("vram_used"),
            "ram_total": wrow.get("ram_total"),
            "free_ram": wrow.get("free_ram"),
            "gpu_memory_free": (gpus[0].get("memory_free") if gpus else None),
            "last_load_error": last_load_error,
        }

    return {
        "served_worker": served,
        "outcome": terminal.get("outcome"),
        "error": terminal.get("error"),
        "finish_reason": terminal.get("finish_reason"),
        "ttft_s": terminal.get("ttft_s"),
        "load_duration_s": terminal.get("load_duration_s"),
        "wall_s": terminal.get("wall_s"),
        "tokens": terminal.get("tokens"),
        "stages": terminal.get("stages") or [],
        "allocation": allocation,
        "loaded_detail": loaded_detail,
        "admission": {
            "verdict": verdict,
            "partial_offload_considered": partial_considered,
            "refusal_reason": refusal,
            "vram_evictions_delta": ev_delta,
        },
        "worker_state": worker_state,
    }


def evictions_snapshot(workers: list[dict]) -> dict:
    """{name: vram_evictions} for computing a per-trial eviction delta."""
    return {w.get("name"): w.get("vram_evictions") for w in workers}
