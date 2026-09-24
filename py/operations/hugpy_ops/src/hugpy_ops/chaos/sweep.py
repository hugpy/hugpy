"""k7 — the offload speed-cliff sweep (a DETERMINISTIC mode of the chaos exerciser).

Where the chaos ``runner`` draws random points to teach the learner, this sweep
walks a fixed grid: for each of the top-N most-used GGUF chat models, on its
designated GPU worker, it measures generation speed at full GPU offload ("as full
as the card allows") and then steps the VRAM share DOWN — shifting layers to host
RAM via the per-model ``gpu_mem_gib`` budget — to find each model's true
performance cliff (the VRAM% below which tok/s collapses).

The lever is the same operator-gated ``/assign`` spill the chaos runner uses; the
new bit is that the sweep FORCES THE MODEL COLD (``/unload``) between points so
each load re-reads the fresh budget (agent: a spill change on an already-loaded
model has no effect until reload). Every point:

  1. safety-gate the card (health, no active GPU reservation, no foreign active
     job, enough live VRAM headroom — else record a skip and move on);
  2. force the model cold (unload) — NEVER a model actively serving someone;
  3. apply the point's ``gpu_mem_gib`` budget (snapshot-first, so the model's
     original spill is restored byte-identical after its whole sweep);
  4. fire ONE warm-up generation (the cold load; EXCLUDED from timing);
  5. fire ``timed_runs`` timed generations (fixed prompt, fixed max_new_tokens),
     record ttft + tok/s (median);
  6. read the serving contract's ACTUAL n_gpu_layers / gpu_pct / vram_bytes;
  7. emit ONE chaos-obs/1 observation extended with a ``sweep`` block.

After a model's grid completes (or on early stop / budget exhaustion) the runner
RESTORES its original spill and verifies byte-identical (the proven zero-drift
discipline), forcing cold once more so the restored contract is what loads next.

Safety rails (mirrors runner.py; see the module tests):
  * only already-designated (worker, model) pairs — restore is a clean write-back;
  * one model in flight per box (the run is strictly sequential);
  * back off a card with an active reservation claim or a foreign active job;
  * never force-cold a model that is actively serving a foreign request;
  * clamp every GPU budget to a safe fraction of the card's LIVE free VRAM so a
    load can never admit-then-OOM (the t21/vram-admission landmine);
  * bounded by --budget-minutes; clean SIGTERM restores the in-flight model.

CLI:  hugpy-chaos sweep [--plan] [--top-n N]
                                               [--budget-minutes M] [--ctx-pct P]
      (or:  hugpy-chaos sweep ...)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

from hugpy_ops.chaos import alloc, observe
from hugpy_ops.chaos.assortment import CHAT_TASKS, worker_index
from hugpy_ops.chaos.client import CentralClient, DEFAULT_BASE, default_base_url
from hugpy_ops.chaos.runner import (
    DEFAULT_ENV_FILE,
    DEFAULT_OUT_DIR,
    default_env_file,
    default_out_dir,
    load_operator_token,
)
from hugpy_ops.chaos.schema import (
    SWEEP_BIG_MODEL_GIB,
    SWEEP_VRAM_PCTS,
    SWEEP_VRAM_PCTS_COARSE,
    blank_observation,
    blank_sweep,
)

GIB = 1 << 30
# A deliberately VERBOSE, "don't stop" prompt: with unbounded=False the engine
# hard-caps at max_new_tokens, so a prompt that keeps generating fills the cap
# and yields a stable, comparable steady-state decode rate at every point. A
# terse prompt would stop naturally after a few tokens (too few to time).
PROMPT = (
    "Write a long, richly detailed description of a bustling seaside market at "
    "dawn. Describe the sights, the sounds, the smells, the vendors, the fishing "
    "boats, and the changing weather in vivid, continuous prose. Do not stop "
    "early — keep describing in specific detail.")


# ── pure helpers (unit-tested; no network) ───────────────────────────────────
def median(values: list[float]) -> float | None:
    """Median of the timed runs. For the default 2 runs this is their mean —
    a fair central estimate that is not thrown by a single stray reading."""
    xs = sorted(v for v in values if isinstance(v, (int, float)))
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return (xs[mid - 1] + xs[mid]) / 2.0


def ceiling_pct(need_gib: float, safe_cap_gib: float) -> int:
    """The highest VRAM share (% of TOTAL requirement) the card can actually
    hold: 100 when the model fully fits, else floor(100·safe_cap/need). This is
    the "as full as the card allows" baseline for an oversize model."""
    if not need_gib or need_gib <= 0:
        return 100
    if safe_cap_gib >= need_gib:
        return 100
    return max(1, int(100.0 * safe_cap_gib / need_gib))


def sweep_points(need_gib: float, safe_cap_gib: float,
                 big_model_gib: float = SWEEP_BIG_MODEL_GIB) -> list[int]:
    """The DETERMINISTIC descending VRAM-share grid for one model, in %-of-total-
    requirement (weights+KV @ the sweep ctx). Coarse grid for big models (each
    point is a cold reload of many GB).

        fits card (ceiling 100%):  fine -> 100 85 70 55 40 25   (big -> 100 70 40)

    When the model does NOT fully fit the card, the top achievable point is the
    card CEILING (e.g. 35% for a 45 GiB model on a 24 GiB card = "as full as the
    card allows"). The grid is then SCALED to that ceiling so the sweep still
    steps meaningfully down from the achievable max (35 → 25 → 14) instead of
    collapsing to a single clamped point. Every returned point is achievable.
    """
    base = (SWEEP_VRAM_PCTS_COARSE if need_gib > big_model_gib
            else SWEEP_VRAM_PCTS)
    cap = ceiling_pct(need_gib, safe_cap_gib)
    if cap >= 100:
        pts = [p for p in base if p <= 100]
    else:
        # scale the grid RATIOS to the achievable ceiling (base[0]==100 -> cap)
        pts = [max(5, int(round(cap * b / 100.0))) for b in base]
    # dedupe preserving descending order
    seen: set[int] = set()
    out = []
    for p in sorted(pts, reverse=True):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def point_budget_gib(pct: int, need_gib: float, safe_cap_gib: float
                     ) -> tuple[float, bool]:
    """gpu_mem_gib budget for a grid point = pct% of the model's total
    requirement, clamped to a safe fraction of the card's live free VRAM. Returns
    (requested_gib, clamped). The clamp is the OOM guard — a budget can never
    exceed what the card can currently hold."""
    want = round(pct / 100.0 * need_gib, 2)
    if safe_cap_gib is not None and want > safe_cap_gib:
        return round(max(0.1, safe_cap_gib), 2), True
    return want, False


def detect_cliff(curve: list[dict]) -> dict:
    """Find the performance cliff in one model's speed-vs-VRAM% curve.

    ``curve`` = points (any order) with ``vram_share_pct`` and ``tokens_per_s``.
    A point with tok/s None (refused / fell to CPU / hit the min-offload floor)
    is treated as a full collapse (0 tok/s) — that transition IS a cliff. The
    cliff is the adjacent (descending) pair with the largest RELATIVE tok/s drop.

    Returns {cliff: bool, from_pct, to_pct, from_tps, to_tps, rel_drop, series}.
    ``from_pct`` is the last healthy share (the recommended floor sits here);
    ``to_pct`` is where speed collapsed.

    The x-axis is the MEASURED VRAM share (actual_vram_share_pct) when present —
    on a volatile card the worker caps the budget to live free VRAM, so requested
    and actual diverge; the cliff lives on what was actually resident. Falls back
    to the requested ``vram_share_pct``.
    """
    def _share(p):
        a = p.get("actual_vram_share_pct")
        return a if a is not None else p.get("vram_share_pct")

    pts = sorted((p for p in curve if _share(p) is not None),
                 key=_share, reverse=True)
    series = [(_share(p),
               p.get("tokens_per_s") if p.get("tokens_per_s") is not None else 0.0)
              for p in pts]
    best = {"cliff": False, "from_pct": None, "to_pct": None,
            "from_tps": None, "to_tps": None, "rel_drop": 0.0, "series": series}
    for (hp, ht), (lp, lt) in zip(series, series[1:]):
        if ht and ht > 0:
            drop = (ht - lt) / ht
            if drop > best["rel_drop"]:
                best.update({"cliff": drop > 0.0, "from_pct": hp, "to_pct": lp,
                             "from_tps": ht, "to_tps": lt, "rel_drop": round(drop, 4)})
    return best


def rank_targets(models: list[dict], workers: list[dict], *, top_n: int,
                 worker_filter: list[str] | None = None,
                 model_filter: list[str] | None = None,
                 big_model_gib: float = SWEEP_BIG_MODEL_GIB) -> dict:
    """Select the top-N GGUF chat models to sweep, ranked by recent usage.

    A target is one (model, GPU worker) pair where the model is DESIGNATED on
    that worker (in ``models``), is a GGUF chat model, and is NOT operator-blocked.
    A model designated on several GPU cards is pinned to the card it was most
    recently PICKED on (model_last_picked). Ranking is by that last-pick time
    (most-recent first); never-picked models sort last, tie-broken by weight size
    (bigger = a more informative cliff) then key.

    Excludes (with a recorded reason, never silently): blocked models, non-GGUF /
    non-chat models, models with unknown weight size, and models whose weights
    alone exceed the box's vram+ram hybrid (can't even hybrid-fit — pure proxy;
    the precise weights+KV need is priced per-model at sweep time).

    Returns {chosen: [...], excluded: [...], considered: int}. PURE — consumes
    /models + /llm/workers payloads so it is unit-testable with fixtures.
    """
    widx = worker_index(workers)
    # GPU workers only (a card with a VRAM total); op/CPU boxes carry no offload
    # cliff to find.
    gpu_workers = {n: w for n, w in widx.items()
                   if w.get("vram_total") and (worker_filter is None
                                               or n in worker_filter)}
    mby = {m.get("model_key"): m for m in models}

    def is_chat(m: dict) -> bool:
        t = set(m.get("tasks") or [])
        if m.get("primary_task"):
            t.add(m["primary_task"])
        return bool(t & CHAT_TASKS)

    mfilter = set(model_filter) if model_filter else None
    # one candidate row per (model, GPU worker it is designated on)
    rows: dict[str, dict] = {}
    excluded: list[dict] = []
    for name, w in gpu_workers.items():
        mlp = (workers_by_name(workers).get(name) or {}).get("model_last_picked") or {}
        for mk in (w.get("assigned") or set()):
            if mfilter is not None and mk not in mfilter:
                continue
            m = mby.get(mk) or {}
            eb = int(m.get("effective_bytes") or m.get("size_bytes") or 0)
            lp = mlp.get(mk)
            keep = rows.get(mk)
            cand = {"model_key": mk, "framework": m.get("framework"),
                    "effective_bytes": eb, "worker": name,
                    "worker_id": w.get("id"), "vram_total": w.get("vram_total"),
                    "ram_total": w.get("ram_total"), "last_picked": lp}
            # pin to the card it was most recently picked on
            if keep is None or (lp or 0) > (keep.get("last_picked") or 0):
                rows[mk] = cand

    chosen_pool = []
    for mk, cand in rows.items():
        m = mby.get(mk) or {}
        if m.get("blocked"):
            excluded.append({**cand, "reason": "blocked"})
            continue
        if (cand.get("framework") or "").lower() != "gguf":
            excluded.append({**cand, "reason": "not-gguf"})
            continue
        if not is_chat(m):
            excluded.append({**cand, "reason": "not-chat"})
            continue
        eb = cand["effective_bytes"]
        if eb <= 0:
            excluded.append({**cand, "reason": "unknown-weight-size"})
            continue
        hybrid = (cand.get("vram_total") or 0) + (cand.get("ram_total") or 0)
        if hybrid and eb > hybrid:
            excluded.append({**cand, "reason": "weights-exceed-hybrid"})
            continue
        chosen_pool.append(cand)

    chosen_pool.sort(key=lambda c: (-(c.get("last_picked") or 0),
                                    -c["effective_bytes"], c["model_key"]))
    for i, c in enumerate(chosen_pool):
        c["rank"] = i + 1
        c["need_gib_proxy"] = round(c["effective_bytes"] / GIB, 2)
    return {"chosen": chosen_pool[:top_n],
            "overflow": chosen_pool[top_n:],
            "excluded": excluded,
            "considered": len(chosen_pool)}


def workers_by_name(workers: list[dict]) -> dict:
    out = {}
    for w in workers:
        out[w.get("name") or w.get("hostname") or (w.get("id") or "")[:8]] = w
    return out


# ── the live sweep runner ────────────────────────────────────────────────────
class SweepRunner:
    def __init__(self, client: CentralClient, *, top_n: int, ctx_pct: int,
                 budget_minutes: float, out_dir: str, max_new_tokens: int,
                 warmup_tokens: int, timed_runs: int, chat_ceiling_s: float,
                 headroom_gib: float, vram_safety_frac: float,
                 assign_settle_s: float, settle_s: float,
                 big_model_gib: float, worker_filter: list[str] | None,
                 run_id: str | None = None):
        self.client = client
        self.top_n = top_n
        self.ctx_pct = ctx_pct
        self.budget_s = budget_minutes * 60.0
        self.out_dir = Path(out_dir)
        self.max_new_tokens = max_new_tokens
        self.warmup_tokens = warmup_tokens
        self.timed_runs = timed_runs
        self.chat_ceiling_s = chat_ceiling_s
        self.headroom_gib = headroom_gib
        self.vram_safety_frac = vram_safety_frac
        self.assign_settle_s = assign_settle_s
        self.settle_s = settle_s
        self.big_model_gib = big_model_gib
        self.worker_filter = worker_filter
        self.run_id = run_id or f"sweep-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self.obs_path = self.out_dir / f"{self.run_id}.jsonl"
        self.manifest_path = self.out_dir / "runs" / f"{self.run_id}.json"
        self._stop = False
        self._health_fails = 0
        self.counts: dict[str, int] = {}
        self.restore_ledger: list[dict] = []
        self.point_records: list[dict] = []   # in-memory mirror for the summary

    # ── lifecycle ────────────────────────────────────────────────────────────
    def request_stop(self, *_):
        self._stop = True

    def _tick(self, key: str):
        self.counts[key] = self.counts.get(key, 0) + 1

    def _append(self, obs: dict):
        from hugpy_ops.chaos.observations import append_observation
        append_observation(self.out_dir, self.obs_path, obs)

    def _is_sweep_job(self, job: dict) -> bool:
        rid = str(job.get("id") or job.get("request_id") or "")
        return self.run_id in rid or rid.startswith("sweep-")

    # ── per-card safety gate ─────────────────────────────────────────────────
    def _reservation_on(self, worker_name: str, worker_id: str) -> bool:
        """True if an active GPU reservation claims this card — respect it."""
        for r in self.client.reservations():
            state = str(r.get("state") or r.get("status") or "active").lower()
            if state in ("released", "expired", "done", "cancelled", "canceled"):
                continue
            vals = {str(v) for v in r.values() if isinstance(v, (str, int))}
            if worker_name in vals or (worker_id and worker_id in vals):
                return True
            for key in ("worker", "worker_name", "worker_id", "target",
                        "card", "host"):
                if str(r.get(key) or "") in (worker_name, worker_id):
                    return True
        return False

    def _foreign_active_on(self, worker_name: str) -> bool:
        """True if a NON-sweep job is active/queued on this worker (be a guest)."""
        for j in (self.client.jobs().get("jobs") or []):
            status = str(j.get("status") or "").lower()
            if status not in ("active", "running", "processing", "waiting",
                              "queued", "pending"):
                continue
            if self._is_sweep_job(j):
                continue
            jw = j.get("worker") or j.get("worker_name")
            # a job pinned to this worker, OR an unattributed active job (be safe)
            if jw in (worker_name, None, ""):
                return True
        return False

    def _live_worker(self, worker_name: str) -> dict:
        for w in self.client.workers():
            if w.get("name") == worker_name:
                return w
        return {}

    def _model_serving_foreign(self, worker_row: dict, model_key: str) -> bool:
        """True if the model is actively DECODING a request right now — never
        force-cold a model mid-reply. Keys on ``busy`` (actively processing), NOT
        ``serving`` (which is the normal RESIDENT-and-ready state — a resident
        model is always fine to cold-cycle between our own points). Foreign
        traffic is separately caught by the /llm/jobs back-off."""
        for a in (worker_row.get("allocations") or []):
            if a.get("model_key") == model_key and a.get("busy"):
                return True
        return False

    def _safe_cap_gib(self, worker_row: dict) -> float:
        vram_total = worker_row.get("vram_total") or 0
        vram_free = worker_row.get("vram_free")
        # prefer live free VRAM; fall back to total when the row omits it
        base = vram_free if isinstance(vram_free, int) else vram_total
        cap = (base / GIB) * self.vram_safety_frac - self.headroom_gib
        return max(0.0, round(cap, 2))

    # ── one point ────────────────────────────────────────────────────────────
    def _emit_skip(self, target: dict, pct: int, reason: str, t0: float,
                   need_bytes=None, extra: dict | None = None) -> dict:
        obs = self._blank_point(target, pct, need_bytes)
        obs.update({"kind": "skip", "skip_reason": reason, "back_off": True})
        obs["ts_end"] = time.time()
        obs["duration_s"] = round(obs["ts_end"] - t0, 3)
        if extra:
            obs["measured"]["error"] = json.dumps(extra)[:400]
        self._tick(reason)
        self._append(obs)
        self.point_records.append(obs)
        return obs

    def _blank_point(self, target: dict, pct: int, need_bytes) -> dict:
        obs = blank_observation()
        obs.update({"run_id": self.run_id,
                    "trial_id": f"{self.run_id}-{target['model_key']}-p{pct}",
                    "seed": None, "round": None, "ts_start": time.time()})
        obs["combo"].update({
            "model_key": target["model_key"], "framework": target["framework"],
            "effective_bytes": target["effective_bytes"], "alloc_mode": "explicit",
            "ctx_pct": self.ctx_pct, "target_workers": [target["worker"]]})
        sw = blank_sweep()
        sw.update({"vram_share_pct": pct, "ctx_pct": self.ctx_pct,
                   "denom_need_bytes": need_bytes,
                   "card_vram_bytes": target.get("vram_total")})
        obs["sweep"] = sw
        return obs

    def _fire(self, model_key: str, tag: str, max_tok: int) -> dict:
        rid = f"{self.run_id}-{model_key}-{tag}"
        # max_collect_tokens bounds the fire CLIENT-SIDE (the server won't self-cap
        # — see chat_stream); tok/s comes back as 1/median(inter-token interval).
        return self.client.chat_stream(model_key, PROMPT, rid, max_tok,
                                       self.chat_ceiling_s, unbounded=False,
                                       max_collect_tokens=max_tok)

    def run_point(self, target: dict, pct: int, need_bytes: int, snap: dict,
                  is_last: bool) -> dict:
        wname, wid, mk = target["worker"], target["worker_id"], target["model_key"]
        t0 = time.time()
        need_gib = need_bytes / GIB

        # 1. health + per-card safety gate
        if self.client.health() != 200:
            self._health_fails += 1
            return self._emit_skip(target, pct, "health-degraded", t0, need_bytes)
        self._health_fails = 0
        if self._reservation_on(wname, wid):
            return self._emit_skip(target, pct, "back-off-reservation", t0, need_bytes)
        if self._foreign_active_on(wname):
            return self._emit_skip(target, pct, "back-off-foreign-jobs", t0, need_bytes)

        wrow = self._live_worker(wname)
        safe_cap = self._safe_cap_gib(wrow)
        if safe_cap < 0.2:
            return self._emit_skip(target, pct, "back-off-headroom", t0, need_bytes)
        if self._model_serving_foreign(wrow, mk):
            return self._emit_skip(target, pct, "serving-busy", t0, need_bytes)

        requested_gib, clamped = point_budget_gib(pct, need_gib, safe_cap)
        # The FULL/baseline point of a model that fits must be TRUE full offload
        # (all layers on GPU via n_gpu_layers=-1) — a gpu_mem_gib budget set to
        # exactly `need` can leave a few layers on CPU to rounding/fragmentation,
        # and even a handful of CPU layers tanks tok/s (the cliff is steep near
        # the top), which would corrupt the baseline. Every lower point (and the
        # ceiling of an oversize model) uses the gpu_mem_gib budget to spill.
        fits_full = need_gib <= safe_cap
        if pct >= 100 and fits_full:
            spill = {"n_gpu_layers": -1}
            requested_gib = round(need_gib, 2)
        else:
            spill = {"gpu_mem_gib": requested_gib}
        # ctx override is opt-in (--ctx-pct). Default = the worker's natural ctx,
        # held CONSTANT across the sweep so we vary ONLY the VRAM offload; forcing
        # a large ctx% inflates KV and confounds the baseline.
        if self.ctx_pct is not None:
            spill["ctx_pct"] = int(self.ctx_pct)

        # 2. force the model COLD so the next load re-reads the new budget
        try:
            self.client.unload(wid, mk)
        except Exception:  # noqa: BLE001 — an unload hiccup is data, not a crash
            pass
        time.sleep(self.settle_s)

        # 3. apply the point's budget (snapshot already taken at model start)
        applied = alloc.apply(self.client, mk, spill, snap)
        if not applied["ok"]:
            return self._emit_skip(target, pct, "alloc-refused", t0, need_bytes,
                                   extra=applied["results"])
        time.sleep(self.assign_settle_s)  # let central register the fresh spill

        # 4. warm-up fire (the cold load) — EXCLUDED from timing
        warm = self._fire(mk, f"p{pct}-warm", self.warmup_tokens)
        load_s = warm.get("load_duration_s") or warm.get("ttft_s")

        # 5. timed fires
        runs = []
        ev_before = observe.evictions_snapshot(self.client.workers())
        last_term = warm
        for i in range(self.timed_runs):
            if self._stop:
                break
            term = self._fire(mk, f"p{pct}-t{i}", self.max_new_tokens)
            last_term = term
            runs.append({"ttft_s": term.get("ttft_s"),
                         "tokens_per_s": term.get("tokens_per_s"),
                         "tokens": term.get("tokens"),
                         "outcome": term.get("outcome")})
            time.sleep(self.settle_s)

        # 6. measure the serving contract
        workers_after = self.client.workers()
        jobs_after = self.client.jobs()
        measured = observe.build_measured(last_term, workers_after, mk,
                                          ev_before, jobs_after)

        # 7. assemble + emit the observation
        obs = self._blank_point(target, pct, need_bytes)
        obs["combo"]["spill"] = spill
        obs["combo"]["was_warm"] = False
        obs["predicted"].update({
            "need_bytes": need_bytes, "needs_weights_bytes": target["effective_bytes"],
            "needs_kv_bytes": (need_bytes - target["effective_bytes"]
                               if need_bytes >= target["effective_bytes"] else None),
            "ctx_pct": self.ctx_pct, "placement_mode": "explicit"})
        obs["measured"] = measured
        allocation = measured.get("allocation") or {}
        ngl = allocation.get("n_gpu_layers")
        total = allocation.get("total_layers")
        vram_bytes = allocation.get("vram_bytes")
        if ngl == -1 and total:
            gpu_pct = 100
        elif isinstance(ngl, int) and isinstance(total, int) and total > 0:
            gpu_pct = round(100.0 * ngl / total, 1)
        else:
            gpu_pct = None
        ttft_med = median([r["ttft_s"] for r in runs])
        tps_med = median([r["tokens_per_s"] for r in runs])
        refused = measured["admission"].get("verdict") == "refuse"
        sw = obs["sweep"]
        sw.update({
            "vram_share_pct": pct, "requested_gib": requested_gib, "clamped": clamped,
            "denom_need_bytes": need_bytes, "ctx_pct": self.ctx_pct,
            "card_vram_bytes": target.get("vram_total"),
            "actual_n_gpu_layers": ngl, "actual_total_layers": total,
            "actual_gpu_pct": gpu_pct, "actual_vram_bytes": vram_bytes,
            "actual_vram_share_pct": (round(100.0 * vram_bytes / need_bytes, 1)
                                      if isinstance(vram_bytes, int) and need_bytes
                                      else None),
            "load_s": round(load_s, 3) if isinstance(load_s, (int, float)) else None,
            "ttft_s": round(ttft_med, 3) if ttft_med is not None else None,
            "tokens_per_s": round(tps_med, 2) if tps_med is not None else None,
            "runs": runs, "warmup_outcome": warm.get("outcome"),
            "floor_rejected": bool(refused),
        })
        if is_last:  # attach the model's restore proof to its final point
            obs["restore"] = {"ok": None, "per_worker": {},
                              "note": "restored after final point"}
        obs["ts_end"] = time.time()
        obs["duration_s"] = round(obs["ts_end"] - t0, 3)
        self._tick(measured.get("outcome") or "point")
        self._append(obs)
        self.point_records.append(obs)
        return obs

    # ── sweep one model ──────────────────────────────────────────────────────
    def sweep_model(self, target: dict) -> dict:
        wname, wid, mk = target["worker"], target["worker_id"], target["model_key"]
        widx = worker_index(self.client.workers())
        snap = alloc.snapshot(self.client, mk, [wname], widx)

        # price the denominator ONCE: weights + KV @ the sweep ctx
        meta = self.client.model_meta(mk, ctx_pct=self.ctx_pct)
        rec = (meta.get("recommended") or {}) if isinstance(meta, dict) else {}
        need_bytes = rec.get("need_bytes")
        if not isinstance(need_bytes, int) or need_bytes <= 0:
            need_bytes = int(target["effective_bytes"] * 1.15) or None  # rough
        wrow = self._live_worker(wname)
        safe_cap = self._safe_cap_gib(wrow)
        pts = sweep_points(need_bytes / GIB, safe_cap, self.big_model_gib)

        print(f"\n[{target.get('rank','?')}] {mk} on {wname} "
              f"need={need_bytes/GIB:.2f}GiB card_free_cap={safe_cap:.1f}GiB "
              f"points={pts}", flush=True)

        fired = 0
        try:
            for i, pct in enumerate(pts):
                if self._stop:
                    self._emit_skip(target, pct, "stopped", time.time(), need_bytes)
                    break
                if time.time() - self._started >= self.budget_s:
                    self._emit_skip(target, pct, "budget-exhausted", time.time(),
                                    need_bytes)
                    self._budget_hit = True
                    break
                obs = self.run_point(target, pct, need_bytes, snap,
                                     is_last=(i == len(pts) - 1))
                sw = obs.get("sweep") or {}
                print(f"    p{pct:>3}% req={sw.get('requested_gib')}G "
                      f"-> gpu {sw.get('actual_n_gpu_layers')}/{sw.get('actual_total_layers')} "
                      f"({sw.get('actual_gpu_pct')}%) ttft={sw.get('ttft_s')}s "
                      f"tps={sw.get('tokens_per_s')} "
                      f"[{obs.get('skip_reason') or obs['measured'].get('outcome')}]",
                      flush=True)
                if obs.get("kind") != "skip":
                    fired += 1
                    # Budget guard: if the FULLEST offload (the first fired point)
                    # can't even produce a token — a non-generative model (e.g. a
                    # text encoder) or one that won't load — lower shares won't
                    # either. Stop the model rather than burn the budget on it.
                    generated = (obs["measured"].get("tokens") or 0) > 0
                    if fired == 1 and not generated:
                        print(f"    -> fullest point produced no tokens "
                              f"({obs['measured'].get('outcome')}); skipping the "
                              f"rest of this model (budget guard).", flush=True)
                        break
                if self._health_fails >= 2:
                    break
        finally:
            # 8. RESTORE the model's original spill + verify byte-identical; force
            # cold so the restored contract is what loads next.
            rest = alloc.restore(self.client, mk, snap)
            try:
                self.client.unload(wid, mk)
            except Exception:  # noqa: BLE001
                pass
            self.restore_ledger.append({"model_key": mk, "worker": wname,
                                        "ok": rest.get("ok"),
                                        "per_worker": rest.get("per_worker")})
            print(f"    restore ok={rest.get('ok')}", flush=True)
        return {"model_key": mk, "points_fired": fired}

    # ── the run ──────────────────────────────────────────────────────────────
    def run(self, chosen: list[dict], selection: dict) -> dict:
        self._started = time.time()
        self._budget_hit = False
        # Run order: computron before ae, then SMALLEST-first within a card. Small
        # models give clean full->partial cliffs fast (cheap cold reloads); the
        # big oversize models (long partial loads, compressed range) run last so
        # that if the budget runs low THEY truncate, not the informative curves.
        order = sorted(chosen, key=lambda c: (0 if c["worker"] == "computron"
                                              else 1, c.get("effective_bytes", 0),
                                              c.get("rank", 999)))
        self._write_manifest("running", None, selection, order)
        status = "completed"
        try:
            for target in order:
                if self._stop:
                    status = "stopped-sigterm"
                    break
                if time.time() - self._started >= self.budget_s:
                    status = "budget-exhausted"
                    break
                self.sweep_model(target)
                if self._health_fails >= 2:
                    status = "health-degraded"
                    break
            if self._budget_hit and status == "completed":
                status = "budget-exhausted"
        finally:
            ended = time.time()
            self._write_manifest(status, ended, selection, order)
        all_restored = all(r.get("ok") for r in self.restore_ledger)
        return {"run_id": self.run_id, "status": status, "counts": self.counts,
                "observations_path": str(self.obs_path),
                "manifest_path": str(self.manifest_path),
                "all_restored": all_restored,
                "restore_ledger": self.restore_ledger}

    def _write_manifest(self, status: str, ended: float | None,
                        selection: dict, order: list[dict]):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": self.run_id, "mode": "sweep", "status": status,
            "ctx_pct": self.ctx_pct, "top_n": self.top_n,
            "budget_minutes": round(self.budget_s / 60.0, 2),
            "max_new_tokens": self.max_new_tokens, "timed_runs": self.timed_runs,
            "started": getattr(self, "_started", None), "ended": ended,
            "observations_path": str(self.obs_path),
            "base_url": self.client.base, "counts": dict(self.counts),
            "order": [{"model_key": c["model_key"], "worker": c["worker"],
                       "rank": c.get("rank")} for c in order],
            "selection": {"chosen": selection.get("chosen"),
                          "excluded": selection.get("excluded"),
                          "considered": selection.get("considered")},
            "restore_ledger": self.restore_ledger,
        }
        with open(self.manifest_path, "w") as f:
            f.write(json.dumps(manifest, indent=1))


# ── markdown summary (per-model curves + cliffs + band-floor recommendations) ─
def _fmt(v, suffix=""):
    return "—" if v is None else f"{v}{suffix}"


def build_markdown(run_id: str, records: list[dict], restore_ledger: list[dict],
                   selection: dict, meta: dict) -> str:
    """Render OFFLOAD-CLIFF-<date>.md: per-model speed-vs-VRAM% curve, the
    detected cliff, and a closing band-floor recommendation section."""
    by_model: dict[str, list[dict]] = {}
    order: list[str] = []
    for obs in records:
        mk = (obs.get("combo") or {}).get("model_key")
        if obs.get("kind") == "skip" and not (obs.get("sweep") or {}).get("tokens_per_s"):
            # keep skips associated so the table shows what didn't fire
            pass
        if mk not in by_model:
            by_model[mk] = []
            order.append(mk)
        by_model[mk].append(obs)

    L: list[str] = []
    L.append(f"# Offload speed-cliff sweep — {meta.get('date','2026-07-18')}")
    L.append("")
    L.append(f"Run `{run_id}` · central `{meta.get('base_url')}` · "
             f"ctx {meta.get('ctx_pct')}% · {meta.get('timed_runs')} timed "
             f"gen/point · `max_new_tokens={meta.get('max_new_tokens')}` · "
             f"status **{meta.get('status')}**.")
    L.append("")
    L.append("**What each point is:** the model is forced cold, its per-model "
             "`gpu_mem_gib` budget is set to *N%* of its total requirement "
             "(weights + KV @ the sweep ctx — the t49 denominator), then 2 timed "
             "generations measure ttft + decode tok/s. `vram%` = requested share; "
             "`gpu%`/`layers` = the serving contract's MEASURED offload; `act.vram%` "
             "= measured VRAM bytes ÷ requirement. tok/s is the median.")
    L.append("")

    # selection evidence
    L.append("## Chosen models (ranking evidence)")
    L.append("")
    L.append("| # | model | card | weights (GiB) | last picked | last-pick age |")
    L.append("|---|-------|------|--------------:|-------------|---------------|")
    now = time.time()
    for c in selection.get("chosen", []):
        lp = c.get("last_picked")
        age = (f"{(now - lp)/3600:.1f} h ago" if lp else "never")
        lp_s = (time.strftime("%m-%d %H:%M", time.localtime(lp)) if lp else "—")
        L.append(f"| {c.get('rank')} | `{c['model_key']}` | {c['worker']} | "
                 f"{c['effective_bytes']/GIB:.2f} | {lp_s} | {age} |")
    L.append("")
    if selection.get("excluded"):
        L.append("Excluded (with reason): "
                 + ", ".join(f"`{e['model_key']}` ({e['reason']})"
                             for e in selection["excluded"][:20]) + ".")
        L.append("")

    # per-model curves + cliffs
    cliffs: list[dict] = []
    L.append("## Per-model curves")
    for mk in order:
        if mk is None:
            continue
        obss = by_model[mk]
        fired = [o for o in obss if o.get("kind") != "skip"]
        wname = (fired[0] if fired else obss[0])["combo"].get("target_workers", ["?"])[0]
        L.append("")
        L.append(f"### `{mk}`  ({wname})")
        L.append("")
        L.append("| vram% | budget GiB | gpu layers | gpu% | act.vram% | ttft s | tok/s | outcome |")
        L.append("|------:|-----------:|-----------|-----:|----------:|-------:|------:|---------|")
        curve = []
        for o in sorted(obss, key=lambda x: -((x.get("sweep") or {}).get("vram_share_pct") or 0)):
            sw = o.get("sweep") or {}
            outc = o.get("skip_reason") or (o.get("measured") or {}).get("outcome")
            layers = (f"{sw.get('actual_n_gpu_layers')}/{sw.get('actual_total_layers')}"
                      if sw.get("actual_total_layers") else _fmt(sw.get("actual_n_gpu_layers")))
            L.append(f"| {_fmt(sw.get('vram_share_pct'),'%')} | {_fmt(sw.get('requested_gib'))} | "
                     f"{layers} | {_fmt(sw.get('actual_gpu_pct'),'%')} | "
                     f"{_fmt(sw.get('actual_vram_share_pct'),'%')} | {_fmt(sw.get('ttft_s'))} | "
                     f"{_fmt(sw.get('tokens_per_s'))} | {outc} |")
            if o.get("kind") != "skip":
                curve.append(sw)
        cliff = detect_cliff(curve)
        cliff["model_key"] = mk
        cliff["worker"] = wname
        cliffs.append(cliff)
        if cliff["cliff"] and cliff["from_pct"] is not None:
            L.append("")
            L.append(f"**Cliff:** tok/s drops **{cliff['rel_drop']*100:.0f}%** crossing "
                     f"from **{cliff['from_pct']}%** ({_fmt(cliff['from_tps'])} tok/s) "
                     f"to **{cliff['to_pct']}%** ({_fmt(cliff['to_tps'])} tok/s).")
        else:
            L.append("")
            L.append("**Cliff:** none detected in the swept range (speed held).")

    # band-floor recommendations
    L.append("")
    L.append("## Recommended band floors (t21)")
    L.append("")
    L.append("Floor = the last healthy VRAM share above the cliff (cliff + one grid "
             "step of margin). A band floored below the cliff is a slow-serving "
             "promise. Compared against the earlier universal ±10% VRAM guess "
             "(i.e. a band whose floor sits 10% below a 100% target = 90%).")
    L.append("")
    L.append("| model | cliff (from→to) | recommended floor | vs ±10% (90%) guess |")
    L.append("|-------|-----------------|-------------------|---------------------|")
    for c in cliffs:
        if c["cliff"] and c["from_pct"] is not None:
            floor = c["from_pct"]
            transition = f"{c['from_pct']}%→{c['to_pct']}%"
            verdict = ("90% floor CLEARS the cliff" if 90 >= (c["to_pct"] or 0)
                       and floor <= 90 else
                       f"90% floor is BELOW the cliff — needs ≥{floor}%")
            if floor > 90:
                verdict = f"needs a HIGHER floor ({floor}%) than the ±10% guess"
        else:
            floor = "n/a (no cliff)"
            transition = "—"
            verdict = "±10% band is safe (no cliff in range)"
        L.append(f"| `{c['model_key']}` | {transition} | {floor}"
                 f"{'%' if isinstance(floor,int) else ''} | {verdict} |")
    L.append("")

    # restore proof
    L.append("## Restore proof (zero drift)")
    L.append("")
    all_ok = all(r.get("ok") for r in restore_ledger) if restore_ledger else None
    L.append(f"All {len(restore_ledger)} swept models restored to their original "
             f"spill and verified byte-identical: **{all_ok}**.")
    L.append("")
    for r in restore_ledger:
        L.append(f"- `{r['model_key']}` on {r['worker']}: restore ok=**{r['ok']}**")
    L.append("")
    return "\n".join(L)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="hugpy-chaos sweep",
        description="Offload speed-cliff sweep (k7): measure tok/s vs VRAM share "
                    "for the top-N GGUF chat models on their GPU worker, and find "
                    "each model's performance cliff.")
    ap.add_argument("--base-url", default=default_base_url(),
                    help="central base URL (env HUGPY_BASE_URL; legacy aliases "
                         "honoured; default loopback central)")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--workers", default=None,
                    help="comma-list of GPU workers to sweep (default: all)")
    ap.add_argument("--models", default=None,
                    help="comma-list of exact model_keys to restrict the "
                         "selection to (still ranked/validated; default: top-N "
                         "by usage)")
    ap.add_argument("--ctx-pct", type=int, default=None,
                    help="OPTIONAL fixed context %% override for every point. "
                         "Default: the worker's natural ctx, held constant so the "
                         "sweep varies only VRAM offload (a large ctx%% inflates "
                         "KV and confounds the full-GPU baseline).")
    ap.add_argument("--budget-minutes", type=float, default=90.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--timed-runs", type=int, default=2)
    ap.add_argument("--chat-ceiling-s", type=float, default=180.0)
    ap.add_argument("--headroom-gib", type=float, default=1.0,
                    help="VRAM kept free under every budget (OOM guard)")
    ap.add_argument("--vram-safety-frac", type=float, default=1.0)
    ap.add_argument("--assign-settle-s", type=float, default=1.5)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--big-model-gib", type=float, default=SWEEP_BIG_MODEL_GIB)
    ap.add_argument("--out-dir", default=None,
                    help="observations + run manifests (default: "
                         "<store root>/comms/chaos, HUGPY_CHAOS_OUT_DIR)")
    ap.add_argument("--markdown", default=None,
                    help="report path (default <out-dir>/OFFLOAD-CLIFF-<date>.md)")
    ap.add_argument("--operator-token", default=None)
    ap.add_argument("--env-file", default=None,
                    help="KEY=VALUE file holding HUGPY_OPERATOR_TOKEN "
                         "(default HUGPY_ENV_FILE or %s)" % DEFAULT_ENV_FILE)
    ap.add_argument("--plan", action="store_true",
                    help="select + plan points + time estimate; NO mutation")
    args = ap.parse_args(argv)
    args.out_dir = args.out_dir or default_out_dir()
    args.env_file = args.env_file or default_env_file()
    run_date = time.strftime("%Y-%m-%d")
    args.markdown = args.markdown or os.path.join(
        args.out_dir, "OFFLOAD-CLIFF-%s.md" % run_date)

    worker_filter = ([w.strip() for w in args.workers.split(",")]
                     if args.workers else None)
    model_filter = ([m.strip() for m in args.models.split(",")]
                    if args.models else None)
    client = CentralClient(args.base_url, operator_token=None)
    if client.health() != 200:
        print("ERROR: central health != 200; refusing to start.", file=sys.stderr)
        return 3

    models = client.models()
    workers = client.workers()
    selection = rank_targets(models, workers, top_n=args.top_n,
                             worker_filter=worker_filter,
                             model_filter=model_filter,
                             big_model_gib=args.big_model_gib)

    print(f"\n=== selection: {len(selection['chosen'])} chosen "
          f"(of {selection['considered']} eligible), "
          f"{len(selection['excluded'])} excluded ===")
    for c in selection["chosen"]:
        lp = c.get("last_picked")
        age = f"{(time.time()-lp)/3600:.1f}h" if lp else "never"
        print(f"  #{c['rank']:>2} {c['model_key']:52} {c['worker']:9} "
              f"{c['effective_bytes']/GIB:6.2f}GiB last_picked={age}")

    # plan / estimate points (a meta read per model — no mutation)
    total_points = 0
    for c in selection["chosen"]:
        meta = client.model_meta(c["model_key"], ctx_pct=args.ctx_pct)
        rec = (meta.get("recommended") or {}) if isinstance(meta, dict) else {}
        need = rec.get("need_bytes") or int(c["effective_bytes"] * 1.15)
        wrow = next((w for w in workers if w.get("name") == c["worker"]), {})
        vram_free = wrow.get("vram_free") or wrow.get("vram_total") or 0
        cap = (vram_free / GIB) * args.vram_safety_frac - args.headroom_gib
        pts = sweep_points(need / GIB, max(0.0, cap), args.big_model_gib)
        total_points += len(pts)
        print(f"       -> need={need/GIB:.1f}GiB points={pts}")
    per_point_s = 12 + (1 + args.timed_runs) * 8   # rough: load + fires
    est_min = total_points * per_point_s / 60.0
    print(f"\n  ~{total_points} points; rough estimate ~{est_min:.0f} min "
          f"(budget {args.budget_minutes:.0f} min).")
    if est_min > args.budget_minutes:
        print("  ⚠ estimate EXCEEDS budget — big models will be truncated "
              "(budget-exhausted) rather than dropping points silently.")

    if args.plan:
        return 0

    token = load_operator_token(args.operator_token, args.env_file)
    if not token:
        print("ERROR: no operator token (HUGPY_OPERATOR_TOKEN / --operator-token "
              "/ d-env/env) — /assign + /unload are operator-gated.",
              file=sys.stderr)
        return 2
    client = CentralClient(args.base_url, operator_token=token)

    runner = SweepRunner(
        client, top_n=args.top_n, ctx_pct=args.ctx_pct,
        budget_minutes=args.budget_minutes, out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens, warmup_tokens=args.warmup_tokens,
        timed_runs=args.timed_runs, chat_ceiling_s=args.chat_ceiling_s,
        headroom_gib=args.headroom_gib, vram_safety_frac=args.vram_safety_frac,
        assign_settle_s=args.assign_settle_s, settle_s=args.settle_s,
        big_model_gib=args.big_model_gib, worker_filter=worker_filter)
    signal.signal(signal.SIGTERM, runner.request_stop)
    signal.signal(signal.SIGINT, runner.request_stop)
    print(f"\nsweep {runner.run_id} -> {runner.obs_path}", flush=True)
    result = runner.run(selection["chosen"], selection)

    md = build_markdown(runner.run_id, runner.point_records,
                        runner.restore_ledger, selection,
                        {"date": run_date, "base_url": client.base,
                         "ctx_pct": args.ctx_pct, "timed_runs": args.timed_runs,
                         "max_new_tokens": args.max_new_tokens,
                         "status": result["status"]})
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(md)
    result["markdown_path"] = args.markdown
    print(json.dumps(result, indent=1))
    print(f"\nmarkdown -> {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
