"""Computed-once facts for the hugpy PLANNER (moved from react/testshell/compute.py
2026-10-02, operator: "move the planner out of the testshell into the hugpy
engine package"; the testshell now imports this module) (operator ruling 2026-10-01: the 4-bit
size, the auto allocation and the MoE split are computed ONCE per
(model, worker, inputs) and stored in the DB; nothing here rides a heartbeat).

Three layers (operator, 2026-10-01):
  1. models.weights      immutable facts of the weights: engine, size, tasks,
                         max_ctx, MoE structure, 4-bit size. Once per file.
  2. model_workers.plan  one row per worker: feasible? then the auto
                         allocation for standard and for 4-bit, and the explicit
                         MoE split. Once per (weights, worker resources).
  3. (central, at load)  only max-gpu / max-ram layer resolution happens when a
                         call arrives and the model is not loaded. Not here.

A plan is recomputed only when its inputs differ from the stored ones: the
weights file (size/mtime) and the worker's GPU/RAM totals + package version.
Operator knobs are NOT inputs: the plan holds every variant, the knobs pick.

Uses the shared engine math (hugpy_engine.alloc_modes.default_allocation,
bnb_effective_bytes, hugpy_engine.spill.gguf_moe_detail) so the stored answer
is the same one central derives on every read today. gpu_reserve_bytes (the KV
reserve central adds for a MoE card contract) is not applied here yet; the
`why` string says what was priced.
"""
from __future__ import annotations

import os
import sys
import time

from hugpy_engine.alloc_modes import (ALLOC_MODES, MOE_ALL_LAYERS, bnb_effective_bytes, bnb_eligible,  # noqa: E402
                                      default_allocation, feasible_modes, is_gguf_engine)
from hugpy_engine import spill as _spill  # noqa: E402
from hugpy_engine.spill import gguf_moe_detail  # noqa: E402
from hugpy_storage import hugpy_marker as _marker  # noqa: E402

try:
    from hugpy_engine.spill import _gguf_shard_paths
except ImportError:  # pragma: no cover
    _gguf_shard_paths = None

PLAN_VERSION = 12  # v12: MoE GGUF verdicts carry memory.explicit.band — per-class (attention / experts on GPU) feasible band primitives + e_max table. v11: engine-specific compute reserve — transformers/bnb loads price HUGPY_TRANSFORMERS_COMPUTE_GIB (2 GiB) for activations + dequant workspace, not llama.cpp's 512 MiB. v10: transformers dirs carry config/safetensors facts (KV geo, hybrid state, experts); verdicts price bnb_4bit (memory.bnb_4bit / moe.bnb_4bit). v9: every mode carries ctx_max per KV cache type (f16/q8_0/q4_0) — the slider range, by the same budgets as the quant verdict. v8: facts carry kv_cost (ctx -> KV precalculated per quant). v7: worker budgets are an EARMARK (worker_budgets.rev); reduction keeps unfeasible verdicts, expansion re-evaluates all
HUB_FETCH = (os.environ.get("HUGPY_PLANNER_HUB_FETCH") or os.environ.get("HUGPY_TESTSHELL_HUB_FETCH", "")) == "1"   # the ONE permitted backfill call for an uncached Hub row
_GIB = 1024 ** 3
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".ckpt")


def _sig(path: str) -> dict:
    st = os.stat(path)
    return {"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _sig_current(sig: dict | None) -> bool:
    if not sig or not sig.get("path"):
        return False
    try:
        st = os.stat(sig["path"])
    except OSError:
        return False
    return st.st_size == sig.get("size") and st.st_mtime_ns == sig.get("mtime_ns")


def model_engine(model: dict) -> str | None:
    a = model.get("attributes") or {}
    return (model.get("framework") or a.get("framework") or None)


def resolve_gguf(model: dict, gguf_file: str | None = None) -> str | None:
    """Path of the GGUF that serves: the worker's gguf_file knob if given, else
    the model's own filename, else its first quant. Searches the model dir for
    a bare filename so a knob that names just the file still resolves."""
    a = model.get("attributes") or {}
    d = a.get("dir")
    if not d or not os.path.isdir(d):
        return None
    cands = [c for c in (gguf_file, a.get("filename")) if c]
    for q in model.get("quants") or []:
        if q.get("file"):
            cands.append(q["file"])
    for c in cands:
        p = c if os.path.isabs(c) else os.path.join(d, c)
        if os.path.isfile(p):
            return p
        base = os.path.basename(c)
        for root, _dirs, files in os.walk(d):
            if base in files:
                return os.path.join(root, base)
    return None


def gguf_bytes(path: str) -> int:
    if _gguf_shard_paths:
        try:
            return sum(os.path.getsize(s) for s in _gguf_shard_paths(path))
        except Exception:  # noqa: BLE001
            pass
    return os.path.getsize(path)


def dir_bytes(d: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.endswith(WEIGHT_EXT):
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


# ── layer 1, per model: weights ─────────────────────────────────────────────
ERROR_RETRY_S = 3600


def _read_marker(model: dict):
    d = (model.get("attributes") or {}).get("dir")
    if not d or not os.path.isdir(d):
        return None, None
    return d, _marker.read_hugpy_marker(d)


def weights_stale(model: dict) -> bool:
    """Stale when the stored row no longer mirrors the marker: version, the
    marker's weights_facts.computed_at, or the hub capture changed."""
    w = model.get("weights")
    if not w or w.get("version") != PLAN_VERSION:
        return True
    if w.get("error"):   # not on disk / no marker: retry hourly, not every pass
        return (time.time() - float(w.get("computed_at") or 0)) > ERROR_RETRY_S
    _d, m = _read_marker(model)
    if not isinstance(m, dict):
        return True
    wf = m.get(_marker.WEIGHTS_FACTS_KEY) or {}
    if wf.get("computed_at") != w.get("facts_computed_at"):
        return True
    if ((m.get(_marker.HUB_KEY) or {}).get("captured_at")) != w.get("hub_captured_at"):
        return True
    if w.get("init_gaps") and (time.time() - float(w.get("computed_at") or 0)) > ERROR_RETRY_S:
        return True   # a gap (uncached hub, …) is retried hourly
    return False


def _max_ctx(a: dict):
    for k in ("model_max_length", "max_position_embeddings", "tokenizer_model_max_length"):
        v = a.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return None


def compute_weights(model: dict) -> dict:
    """Layer 1 = INGEST hugpy.json (operator ruling 2026-10-02: the marker is the
    model's init record for the DB). Backfills the marker's ``weights_facts``
    and ``hub`` blocks when missing (idempotent, local reads; a Hub fetch only
    with HUGPY_TESTSHELL_HUB_FETCH=1), then mirrors them: ``variants`` one per
    weights file, ``hub`` the full Hub record. Nothing here derives a fact the
    marker does not state."""
    a = model.get("attributes") or {}
    engine = model_engine(model)
    now = time.time()
    out = {"version": PLAN_VERSION, "engine": engine, "tasks": a.get("tasks") or ([a["primary_task"]] if a.get("primary_task") else None),
           "max_ctx": _max_ctx(a), "computed_at": now, "moe": {"is_moe": False}, "size_bytes": None, "variants": {},
           "init_gaps": [], "hub": None, "facts_computed_at": None, "hub_captured_at": None, "marker_stamped_at": None}
    d, m = _read_marker(model)
    if not d:
        out["error"] = "model dir not on disk"
        return out
    if not isinstance(m, dict):
        out["error"] = "no hugpy.json — model is not initialised for the DB"
        return out
    try:
        _marker.stamp_marker_weights_facts(d, marker=m, write=True)
    except Exception as exc:  # noqa: BLE001 — recorded on the row
        out["stamp_error"] = f"weights_facts: {exc}"
    try:
        _marker.stamp_marker_hub(d, marker=m, write=True, fetch=HUB_FETCH)
    except Exception as exc:  # noqa: BLE001
        out["hub_error"] = f"hub: {exc}"
    wf = m.get(_marker.WEIGHTS_FACTS_KEY) or {}
    files = wf.get("files") or {}
    for key, f in files.items():
        moe = {k: f.get(k) for k in ("is_moe", "expert_bytes", "non_expert_bytes", "expert_count", "expert_used_count", "sparsity",
                                     "expert_bytes_by_layer")
               if f.get(k) is not None} or {"is_moe": False}
        size = f.get("size_bytes")
        eligible = bool(f.get("bnb_eligible"))
        out["variants"][key] = {"file": f.get("path"), "dir": f.get("dir"), "quant": f.get("quant"), "sig": f.get("sig"),
                                "size_bytes": size, "sha256": f.get("sha256"), "shards": f.get("shards"), "moe": moe,
                                "kv_geo": f.get("kv_geo") or {}, "kv_cost": f.get("kv_cost"), "error": f.get("error"),
                                "bnb_4bit": {"eligible": eligible, "bytes": bnb_effective_bytes(size) if (eligible and size) else None}}
    out["facts_computed_at"] = wf.get("computed_at")
    out["hub"] = m.get(_marker.HUB_KEY) if isinstance(m.get(_marker.HUB_KEY), dict) else None
    out["hub_captured_at"] = (out["hub"] or {}).get("captured_at")
    out["marker_stamped_at"] = m.get("stamped_at")
    out["init_gaps"] = _marker.marker_init_gaps(m)
    if not out["variants"]:
        out["error"] = "weights_facts empty — no weights file found in the model dir"
        return out
    fn = os.path.basename(m.get("filename") or "")
    default = fn if fn in out["variants"] else next(iter(out["variants"]))
    dv = out["variants"][default]
    out.update(default_variant=default, file=dv.get("file"), sig=dv.get("sig"), size_bytes=dv.get("size_bytes"),
               moe=dv.get("moe"), bnb_4bit=dv.get("bnb_4bit"))
    return out


# ── per (model, worker): plan ────────────────────────────────────────────────
def worker_totals(payload: dict | None) -> dict:
    payload = payload or {}
    # A card's total is what processes can hold: nvidia-smi memory.total minus the
    # driver's reserved share (firmware/GSP — immutable, in no process; 450 MiB on
    # a 3090). Pricing against the raw total over-promised that much on every card.
    gpu = 0
    for g in payload.get("gpus") or []:
        try:
            gpu += max(0, int((g or {}).get("memory_total") or 0) - int((g or {}).get("memory_reserved") or 0))
        except (TypeError, ValueError):
            pass
    try:
        ram = int(payload.get("ram_total") or 0)
    except (TypeError, ValueError):
        ram = 0
    lim = payload.get("limits") if isinstance(payload.get("limits"), dict) else {}
    encroach = external_vram_bytes(payload)
    def _gib(v):
        try:
            return int(float(v) * _GIB) if v is not None else None
        except (TypeError, ValueError):
            return None
    return {"gpu_total": gpu or None, "ram_total": ram or None,
            "gpu_limit": _gib(lim.get("gpu_mem_gib")), "ram_limit": _gib(lim.get("cpu_mem_gib")),
            "vram_encroach": encroach,
            "pkg_version": payload.get("pkg_version")}


ENCROACH_QUANTUM = 128 * 2 ** 20


def external_vram_bytes(payload: dict | None) -> int:
    """ENCROACHMENT the planner budgets around (operator 2026-10-02: "encroachment
    is an immutable"): VRAM held on the worker's cards by processes that are not
    the worker — ComfyUI's process rows and the genuinely foreign/unattributed
    pids, measured per pid by the worker's registry. The driver's reserved share
    is NOT here (gpu_total already excludes it), nor the worker's own CUDA
    context (worker usage). Rounded UP to 128 MiB so a few MiB of drift does not
    bump the earmark rev and re-walk every ladder. 0 when nothing is reported."""
    reg = (payload or {}).get("pid_registry")
    if not isinstance(reg, dict):
        return 0
    total = 0
    for row in reg.get("models") or []:
        if isinstance(row, dict) and row.get("host_mode") == "comfy":
            try:
                total += int(row.get("vram_bytes") or 0)
            except (TypeError, ValueError):
                pass
    for row in reg.get("unattributed") or []:
        if isinstance(row, dict):
            try:
                total += int(row.get("mib") or 0) * 2 ** 20
            except (TypeError, ValueError):
                pass
    if total <= 0:
        return 0
    return -(-total // ENCROACH_QUANTUM) * ENCROACH_QUANTUM


def worker_earmark(totals: dict) -> dict:
    """The EARMARK (operator 2026-10-02): what this worker's models may be priced
    against. budget = min(total - named reserve, operator limit). Offline (no
    reading) -> None totals; the caller keeps the stored earmark then."""
    vr, rr = _spill.vram_reserve_bytes(), _spill.ram_reserve_bytes()
    gpu, ram = totals.get("gpu_total"), totals.get("ram_total")
    # Budget = what the worker's models may hold: the card less the driver
    # (gpu_total), the operator reserve and the measured ENCROACHMENT, capped by
    # the operator limit — the same figure the console bar's "remaining + worker
    # usage" draws (spill.budget_bar: min(limit, physical - external)).
    gb = None if gpu is None else max(0, int(gpu) - vr - int(totals.get("vram_encroach") or 0))
    rb = None if ram is None else max(0, int(ram) - rr)
    if gb is not None and totals.get("gpu_limit") is not None:
        gb = min(gb, int(totals["gpu_limit"]))
    if rb is not None and totals.get("ram_limit") is not None:
        rb = min(rb, int(totals["ram_limit"]))
    return {**totals, "vram_reserve": vr, "ram_reserve": rr, "gpu_budget": gb, "ram_budget": rb}


def budgets_from_earmark(e: dict) -> dict:
    return {"gpu_total": e.get("gpu_total"), "ram_total": e.get("ram_total"),
            "vram_reserve": e.get("vram_reserve"), "ram_reserve": e.get("ram_reserve"),
            "gpu_limit": e.get("gpu_limit"), "ram_limit": e.get("ram_limit"),
            "gpu_budget": e.get("gpu_budget"), "ram_budget": e.get("ram_budget")}


def plan_inputs(model: dict, earmark: dict | None) -> dict:
    """What a pair's verdicts are keyed on: the worker's earmark (budgets + rev)
    and every weights file's signature. Knobs are NOT inputs."""
    w = model.get("weights") or {}
    e = earmark or {}
    return {"budget_rev": e.get("rev"), "gpu_budget": e.get("gpu_budget"), "ram_budget": e.get("ram_budget"),
            "gpu_total": e.get("gpu_total"), "ram_total": e.get("ram_total"), "pkg_version": e.get("pkg_version"),
            "engine": w.get("engine"), "file_sig": w.get("sig"), "model_bytes": w.get("size_bytes"),
            "variant_sigs": {k: v.get("sig") for k, v in (w.get("variants") or {}).items()}}


def plan_stale(plan: dict | None, inputs: dict) -> bool:
    if not plan or plan.get("version") != PLAN_VERSION:
        return True
    return (plan.get("inputs") or {}) != inputs

def budgets(gpu, ram) -> dict:
    """What the weights may claim on each device, from central's NAMED reserves
    (no percentages — operator, 2026-10-01): VRAM minus HUGPY_VRAM_RESERVE_GIB
    (default 0: foreign usage is measured, not assumed), RAM minus
    HUGPY_RAM_RESERVE_GIB (default 4 GiB: OS + agent, OOM floor). Context is NOT
    in here — it is priced per model from its KV geometry (see moe_split)."""
    vr, rr = _spill.vram_reserve_bytes(), _spill.ram_reserve_bytes()
    return {"gpu_total": gpu, "ram_total": ram, "vram_reserve": vr, "ram_reserve": rr,
            "gpu_budget": None if gpu is None else max(0, int(gpu) - vr),
            "ram_budget": None if ram is None else max(0, int(ram) - rr)}


def fits_bound(size, b: dict):
    """Layer 0: the hard bound, stated ONCE. weights <= gpu_budget + ram_budget.
    True/False when size and both totals are known; None otherwise. A False is
    the whole answer for the pair: nothing else is priced."""
    if not size or b.get("gpu_budget") is None or b.get("ram_budget") is None:
        return None
    return int(size) <= b["gpu_budget"] + b["ram_budget"]


def plan_fits(plan: dict | None):
    f = (plan or {}).get("feasible") or {}
    return f.get("fits") if isinstance(f, dict) else None


def moe_split(det: dict, geo: dict, size, b: dict, engine=None) -> dict:
    """THE definition of the MoE tick for one (model, worker) (operator, 2026-10-01:
    "if it's not explicit it's not MoE … since it is an option it needs to be
    more defined"). RAM-priority: expert layers fill the RAM budget first
    (llama.cpp --n-cpu-moe N = the first N layers' experts on CPU); the
    overflowing layers ride the GPU with the non-expert share; the GPU must then
    still hold KV for a context >= the ctx floor plus the compute allowance.
    ctx is derived by the calc: the largest (multiple of 1024, <= trained) that
    fits in what the GPU budget has left after weights. Returns
    {offered, why, spill?, kv?, gpu_bytes?, ram_bytes?, leniency_pct?}."""
    layers = det.get("expert_bytes_by_layer") or {}
    per_layer = [int(v) for v in (layers.values() if isinstance(layers, dict) else layers) if v]
    non_expert = int(det.get("non_expert_bytes") or 0)
    gpu_b, ram_b = b.get("gpu_budget"), b.get("ram_budget")
    if not per_layer or non_expert <= 0 or not size or gpu_b is None or ram_b is None:
        return {"offered": False, "why": "MoE structure or worker totals unknown — no split can be stated"}
    cpu_layers = cpu_bytes = 0
    for lb in per_layer:
        if cpu_bytes + lb > ram_b:
            break
        cpu_bytes += lb
        cpu_layers += 1
    gpu_experts = sum(per_layer[cpu_layers:])
    gpu_weights = non_expert + gpu_experts
    compute_b = compute_reserve(engine)
    state = _state_bytes(geo)
    left = gpu_b - gpu_weights - compute_b - state
    numbers = (f"experts {sum(per_layer) / _GIB:.2f} GiB in {len(per_layer)} layers, non-expert {non_expert / _GIB:.2f} GiB; "
               f"RAM budget {ram_b / _GIB:.2f} GiB (= {b['ram_total'] / _GIB:.2f} - {b['ram_reserve'] / _GIB:.1f} GiB reserve) takes "
               f"{cpu_layers} layers ({cpu_bytes / _GIB:.2f} GiB); GPU budget {gpu_b / _GIB:.2f} GiB (= {b['gpu_total'] / _GIB:.2f} - "
               f"{b['vram_reserve'] / _GIB:.1f} GiB reserve) must hold the other {len(per_layer) - cpu_layers} layers "
               f"({gpu_experts / _GIB:.2f} GiB) + non-expert = {gpu_weights / _GIB:.2f} GiB + {compute_b / _GIB:.2f} GiB compute")
    if left <= 0:
        return {"offered": False, "why": f"not offered — weights alone overrun the GPU budget by {-left / _GIB:.2f} GiB: {numbers}"}
    bpt = _spill.kv_bytes_for_geo(geo, 4096, 2.0) / 4096.0 if geo else 0
    if not bpt:
        return {"offered": False, "why": f"not offered — KV geometry unreadable, context cannot be priced: {numbers}"}
    floor = _spill._ctx_floor()
    trained = int(geo.get("ctx_train") or 0) or None
    ctx = _spill._round_down_multiple(left / bpt, 1024)
    if trained:
        ctx = min(ctx, trained)
    kv = (int(_spill.kv_bytes_for_geo(geo, ctx, 2.0)) + state) if ctx > 0 else 0
    if ctx < floor:
        return {"offered": False, "why": (f"not offered — only {ctx} ctx fits after weights (floor {floor}; "
                                          f"{left / _GIB:.2f} GiB left for KV at {bpt / 2**20:.2f} MiB/token): {numbers}")}
    leniency = int(-(-100 * gpu_experts // size))
    spill = {"alloc_mode": "explicit", "n_gpu_layers": -1,
             "n_cpu_moe": MOE_ALL_LAYERS if cpu_layers >= len(per_layer) else cpu_layers,
             "gpu_mem_gib": round((gpu_weights + kv + compute_b) / _GIB, 2), "cpu_mem_gib": round(cpu_bytes / _GIB, 2),
             "llama_ctx": ctx, "leniency_pct": leniency, "priority_device": "ram"}
    return {"offered": True, "spill": spill, "gpu_bytes": gpu_weights, "ram_bytes": cpu_bytes, "leniency_pct": leniency,
            "kv": {"ctx": ctx, "ctx_train": trained, "kv_bytes": kv, "bytes_per_token": int(bpt), "compute_bytes": compute_b},
            "why": (f"explicit, RAM priority: {cpu_layers}/{len(per_layer)} expert layers in RAM, "
                    f"{len(per_layer) - cpu_layers} on GPU (leniency {leniency}% of the model off its ideal device); "
                    f"GPU = {gpu_weights / _GIB:.2f} GiB weights + {kv / _GIB:.2f} GiB KV @ ctx {ctx} + {compute_b / _GIB:.2f} GiB compute "
                    f"= {(gpu_weights + kv + compute_b) / _GIB:.2f} of {gpu_b / _GIB:.2f} GiB budget. {numbers}")}


def _ctx_fit(left_bytes, bpt, floor, trained):
    """Largest ctx (multiple of 1024, <= trained) whose KV fits in left_bytes; 0 when < floor."""
    if bpt <= 0 or left_bytes <= 0:
        return 0
    ctx = _spill._round_down_multiple(left_bytes / bpt, 1024)
    if trained:
        ctx = min(ctx, trained)
    return ctx if ctx >= floor else 0


def compute_reserve(engine) -> int:
    """The NAMED compute allowance on the KV device, per engine (operator
    rulings: no flat % headroom, budgets are named reserves). llama.cpp: the
    engine's 512 MiB scratch. transformers (incl. bitsandbytes 4-bit): the
    activation + dequantization workspace a HF generate needs on top of weights
    and KV — HUGPY_TRANSFORMERS_COMPUTE_GIB, default 2 GiB (operator
    2026-10-02: a 4-bit Qwen3.6-35B-A3B priced to exactly the 24 GiB budget
    "isn't right")."""
    if str(engine or "").strip().lower() in ("gguf", "llama_cpp"):
        return int(_spill._CTX_COMPUTE_RESERVE_BYTES)
    try:
        gib = float(os.environ.get("HUGPY_TRANSFORMERS_COMPUTE_GIB", "2"))
    except ValueError:
        gib = 2.0
    return int(max(0.0, gib) * _GIB)


def ctx_range(left_bytes, geo: dict | None, floor: int, trained: int | None) -> dict:
    """The ctx CEILING per KV cache type for one device budget remainder
    (operator 2026-10-02: "a range of ctx is necessary per cache quantization,
    determined by the same feasibility standards as the quant to worker
    assessment"). For each llama.cpp cache type (KV_CACHE_TYPES bytes/elem) the
    largest ctx (multiple of 1024, <= trained) whose KV at that type fits in
    ``left_bytes``; 0 when even the floor does not fit. Linear in ctx (exact for
    full attention; an upper bound for sliding-window blocks, so never optimistic)."""
    out = {}
    left_bytes = (left_bytes or 0) - _state_bytes(geo)
    for name, per in KV_CACHE_TYPES.items():
        if name == "bf16":
            continue                          # same cost as f16; one entry is enough
        bpt = (_spill.kv_bytes_for_geo(geo, 4096, float(per)) / 4096.0) if geo else 0.0
        out[name] = _ctx_fit(left_bytes, bpt, floor, trained) if bpt else 0
    return out


def _state_bytes(geo) -> int:
    """Hybrid (linear-attention / mamba) layers hold a FIXED recurrent state
    per sequence that is not in the ctx-linear KV: priced once, on the KV
    device. GGUF geometries price it inside kv_bytes_for_geo already."""
    g = geo or {}
    return 0 if g.get("gguf_path") else int(g.get("state_bytes") or 0)


def explicit_band(det: dict, geo: dict | None, size, b: dict, engine=None, ctx: int | None = None) -> dict | None:
    """THE per-class feasible band for explicit mode (operator 2026-10-02: "two
    per-class percentages — attention and experts on GPU — with the feasible
    band from the verdict; attention and expert layers as the floor for either
    end"). Per layer: attention bytes (non-expert / L), expert bytes (the real
    per-layer table), KV bytes at ``ctx`` (KV follows its layer's attention).
    ``a`` attention layers on the GPU = the LAST a layers (n_gpu_layers = a);
    ``e`` expert layers on the GPU = the last e of those (n_cpu_moe = L - e),
    so e <= a. ``e_max_by_a[a]`` = the most expert layers the GPU budget holds
    with a attention layers (+ their KV) on it; ``a_min`` = the fewest attention
    layers the RAM budget forces onto the GPU. Everything else (the sliders'
    grey zones at any ctx / cache type) is linear in these primitives, so the
    UI recomputes from them. None when the structure is unknown."""
    layers = det.get("expert_bytes_by_layer") or {}
    per_layer = [int(x) for x in (layers.values() if isinstance(layers, dict) else layers) if x]
    non_expert = int(det.get("non_expert_bytes") or 0)
    gpu_b, ram_b = b.get("gpu_budget"), b.get("ram_budget")
    if not per_layer or non_expert <= 0 or gpu_b is None or ram_b is None:
        return None
    L = len(per_layer)
    attn = non_expert / L
    compute_b = compute_reserve(engine)
    trained = int((geo or {}).get("ctx_train") or 0) or None
    ctx = int(ctx or trained or 0)
    kv_total = (int(_spill.kv_bytes_for_geo(geo, ctx, 2.0)) + _state_bytes(geo)) if (geo and ctx) else 0
    kv_layer = kv_total / L
    suffix = [0] * (L + 1)                      # suffix[e] = bytes of the LAST e expert layers
    for e in range(1, L + 1):
        suffix[e] = suffix[e - 1] + per_layer[L - e]
    e_max_by_a, a_min = [], None
    for a in range(L + 1):
        room = gpu_b - compute_b - a * (attn + kv_layer)
        e_max = 0
        for e in range(min(a, L), -1, -1):
            if suffix[e] <= room:
                e_max = e
                break
        if room < 0:
            e_max = -1                           # attention alone overruns the GPU
        e_max_by_a.append(e_max)
        if a_min is None and e_max >= 0:
            ram_need = (L - a) * (attn + kv_layer) + (sum(per_layer) - suffix[e_max])
            if ram_need <= ram_b:
                a_min = a
    return {"layers": L, "attn_layer_bytes": int(attn), "kv_layer_bytes": int(kv_layer), "kv_ctx": ctx,
            "kv_total_bytes": int(kv_total), "expert_layer_bytes": per_layer, "expert_bytes": int(sum(per_layer)),
            "non_expert_bytes": non_expert, "compute_bytes": compute_b, "gpu_budget": int(gpu_b), "ram_budget": int(ram_b),
            "e_max_by_a": e_max_by_a, "a_min": a_min,
            "note": ("a = attention layers on the GPU (n_gpu_layers = a; KV of those layers on the GPU), "
                     "e = expert layers on the GPU (n_cpu_moe = L - e), e <= a; e_max_by_a[a] < 0 means a does not fit")}


def memory_plan(engine, v: dict, geo: dict, det: dict | None, b: dict, moe_verdict: dict | None) -> dict:
    """The EXPLICIT memory plan per allocation mode for one quant on one worker:
    gpu_bytes / ram_bytes / kv_bytes / ctx / compute_bytes / fits / why — numbers,
    stated once (operator 2026-10-02: "I don't explicitly see the key-values for
    the precalculated memory plans"). Dense layers are priced as size/n_layers
    (uniform) when a layer count is known; KV from the file's geometry at the
    ctx the calc derives (largest that fits, >= floor). Policies:
      gpu-only  all weights + KV + compute on the GPU
      ram-only  all weights + KV in RAM (CPU inference)
      max-gpu   weights-first on the GPU: as many layers as leave room for KV at
                the ctx FLOOR, the rest in RAM; ctx then grows into what is left
      max-ram   weights-first in RAM: as many layers as fit, the rest on the GPU;
                ctx = what the GPU has left after the overflow
      explicit  the MoE RAM-priority split (moe_split) when offered"""
    size = int(v.get("size_bytes") or 0)
    gpu_b, ram_b = b.get("gpu_budget"), b.get("ram_budget")
    out = {}
    if not size or gpu_b is None or ram_b is None:
        return {"why": "size or worker totals unknown — no memory plan"}
    compute_b = compute_reserve(engine)
    floor = _spill._ctx_floor()
    trained = int((geo or {}).get("ctx_train") or 0) or None
    bpt = (_spill.kv_bytes_for_geo(geo, 4096, 2.0) / 4096.0) if geo else 0.0
    n_layers = int((geo or {}).get("n_layers") or 0)
    layer = size / n_layers if n_layers else None
    state = _state_bytes(geo)
    def kv(ctx): return (int(_spill.kv_bytes_for_geo(geo, ctx, 2.0)) + state) if (geo and ctx) else 0
    # gpu-only
    ctx = _ctx_fit(gpu_b - size - compute_b, bpt, floor, trained) if bpt else 0
    fits = size + compute_b <= gpu_b and (ctx > 0 or not bpt)
    out["gpu-only"] = {"gpu_bytes": size + kv(ctx) + compute_b, "ram_bytes": 0, "kv_bytes": kv(ctx), "ctx": ctx, "compute_bytes": compute_b,
                       "weights_gpu": size, "weights_ram": 0, "fits": fits, "kv_device": "gpu",
                       "ctx_max": ctx_range(gpu_b - size - compute_b, geo, floor, trained),
                       "why": f"all {size / _GIB:.2f} GiB on the GPU" + (f", KV {kv(ctx) / _GIB:.2f} GiB @ ctx {ctx}" if ctx else (" — no room for KV at the ctx floor" if bpt else ", KV geometry unknown"))}
    # ram-only
    ctx = _ctx_fit(ram_b - size, bpt, floor, trained) if bpt else 0
    fits = size <= ram_b and (ctx > 0 or not bpt)
    out["ram-only"] = {"gpu_bytes": 0, "ram_bytes": size + kv(ctx), "kv_bytes": kv(ctx), "ctx": ctx, "compute_bytes": 0,
                       "weights_gpu": 0, "weights_ram": size, "fits": fits, "kv_device": "ram",
                       "ctx_max": ctx_range(ram_b - size, geo, floor, trained),
                       "why": f"all {size / _GIB:.2f} GiB in RAM (CPU inference)" + (f", KV {kv(ctx) / _GIB:.2f} GiB @ ctx {ctx}" if ctx else (" — no room for KV at the ctx floor" if bpt else ""))}
    if layer and bpt:
        # max-gpu: weights first on the card, keep KV at the floor
        room = gpu_b - compute_b - kv(floor)
        n_gpu = max(0, min(n_layers, int(room // layer))) if room > 0 else 0
        w_gpu, w_ram = int(n_gpu * layer), int(size - n_gpu * layer)
        ctx = _ctx_fit(gpu_b - w_gpu - compute_b, bpt, floor, trained)
        fits = w_ram <= ram_b and ctx > 0 and n_gpu > 0
        out["max-gpu"] = {"gpu_bytes": w_gpu + kv(ctx) + compute_b, "ram_bytes": w_ram, "kv_bytes": kv(ctx), "ctx": ctx, "compute_bytes": compute_b,
                          "weights_gpu": w_gpu, "weights_ram": w_ram, "n_gpu_layers": n_gpu, "n_layers": n_layers, "fits": fits, "kv_device": "gpu",
                          "ctx_max": ctx_range(gpu_b - w_gpu - compute_b, geo, floor, trained),
                          "why": f"{n_gpu}/{n_layers} layers ({w_gpu / _GIB:.2f} GiB) on the GPU, {w_ram / _GIB:.2f} GiB in RAM, KV {kv(ctx) / _GIB:.2f} GiB @ ctx {ctx}"}
        # max-ram: weights first in RAM, overflow to the card
        n_ram = max(0, min(n_layers, int(ram_b // layer)))
        w_ram, w_gpu = int(n_ram * layer), int(size - n_ram * layer)
        ctx = _ctx_fit(gpu_b - w_gpu - compute_b, bpt, floor, trained)
        fits = w_gpu + compute_b <= gpu_b and ctx > 0
        out["max-ram"] = {"gpu_bytes": w_gpu + kv(ctx) + compute_b, "ram_bytes": w_ram, "kv_bytes": kv(ctx), "ctx": ctx, "compute_bytes": compute_b,
                          "weights_gpu": w_gpu, "weights_ram": w_ram, "n_gpu_layers": n_layers - n_ram, "n_layers": n_layers, "fits": fits, "kv_device": "gpu",
                          "ctx_max": ctx_range(gpu_b - w_gpu - compute_b, geo, floor, trained),
                          "why": f"{n_ram}/{n_layers} layers ({w_ram / _GIB:.2f} GiB) in RAM, {n_layers - n_ram} layers ({w_gpu / _GIB:.2f} GiB) on the GPU, KV {kv(ctx) / _GIB:.2f} GiB @ ctx {ctx}"}
    if moe_verdict and moe_verdict.get("offered"):
        sp = moe_verdict["spill"]; k = moe_verdict.get("kv") or {}
        out["explicit"] = {"gpu_bytes": int(sp["gpu_mem_gib"] * _GIB), "ram_bytes": moe_verdict["ram_bytes"], "kv_bytes": k.get("kv_bytes"),
                           "ctx": k.get("ctx"), "compute_bytes": k.get("compute_bytes"), "weights_gpu": moe_verdict["gpu_bytes"],
                           "weights_ram": moe_verdict["ram_bytes"], "n_cpu_moe": sp["n_cpu_moe"], "leniency_pct": sp["leniency_pct"],
                           "priority_device": "ram", "fits": True, "why": moe_verdict["why"], "kv_device": "gpu",
                           "ctx_max": ctx_range(gpu_b - int(moe_verdict["gpu_bytes"]) - int(k.get("compute_bytes") or compute_b), geo, floor, trained),
                           "band": explicit_band(det, geo, size, b, engine=engine, ctx=k.get("ctx") or trained) if det else None}
    return out


def _plan_variant(engine: str | None, v: dict, b: dict, name: str) -> dict:
    """Layer 2 for ONE quant on ONE worker: fits bound, then modes/auto/MoE."""
    size = v.get("size_bytes")
    out = {"quant": v.get("quant"), "file": os.path.basename(v["file"]) if v.get("file") else name, "size_bytes": size,
           "feasible": {"fits": None, "ok": False, "modes": [], "why": ""}, "auto": None,
           "moe": {"offered": False, "why": "dense — no expert structure"}}
    fits = fits_bound(size, b)
    out["feasible"]["fits"] = fits
    if fits is False:
        out["feasible"]["why"] = (f"{size / _GIB:.2f} GiB of weights exceed this worker's budgets "
                                  f"{b['gpu_budget'] / _GIB:.2f} GiB GPU + {b['ram_budget'] / _GIB:.2f} GiB RAM "
                                  f"= {(b['gpu_budget'] + b['ram_budget']) / _GIB:.2f} GiB (after reserves VRAM "
                                  f"{b['vram_reserve'] / _GIB:.1f} / RAM {b['ram_reserve'] / _GIB:.1f} GiB) — can never fit; nothing priced")
        out["moe"] = {"offered": False, "why": "not offered — this quant does not fit this worker"}
        return out
    if fits is None:
        out["feasible"]["why"] = "weights size or worker totals unknown — bound not decidable; modes degrade to permissive"
    moe = v.get("moe") or {}
    det = dict(moe) if moe.get("is_moe") else None
    geo = dict(v.get("kv_geo") or {})
    is_gguf = bool(v.get("file")) and str(v["file"]).lower().endswith(".gguf")
    if det and is_gguf:
        det = dict(gguf_moe_detail(v["file"]) or {})   # per-layer table (engine-cached)
    if is_gguf:
        geo["gguf_path"] = v["file"]
    # transformers MoE: the per-layer expert table is a marker FACT
    # (safetensors headers); without it the split cannot be stated
    if det and not det.get("expert_bytes_by_layer"):
        det = None
    modes = list(feasible_modes(engine, size, b["gpu_budget"], b["ram_budget"],
                                moe_split_gpu_bytes=(det or {}).get("non_expert_bytes"), bnb=False))
    out["feasible"].update(ok=bool(modes), modes=modes)
    if fits:
        out["feasible"]["why"] = (f"{size / _GIB:.2f} GiB <= {(b['gpu_budget'] + b['ram_budget']) / _GIB:.2f} GiB GPU+RAM budgets; "
                                  f"modes per central feasible_modes")
    if modes:
        auto = {"standard": default_allocation(engine, size, b["gpu_budget"], b["ram_budget"], moe=None, bnb=False)}
        if (v.get("bnb_4bit") or {}).get("eligible"):
            auto["bnb_4bit"] = default_allocation(engine, size, b["gpu_budget"], b["ram_budget"], moe=None, bnb=True)
        out["auto"] = auto
    if det and not is_gguf:
        # The MoE EXPERT split (experts in RAM, attention + KV on the GPU) is a
        # llama.cpp mechanism (--n-cpu-moe). The transformers loader has no
        # expert-wise placement — device_map offload is whole-layer and all-or-
        # fail — so for a safetensors MoE the split is a FACT about the model,
        # not an offer (operator 2026-10-02: "it's a transformers and an MoE?").
        out["moe"] = {"offered": False, "engine": "transformers",
                      "expert_count": det.get("expert_count"), "expert_used_count": det.get("expert_used_count"),
                      "expert_bytes": det.get("expert_bytes"), "non_expert_bytes": det.get("non_expert_bytes"),
                      "why": (f"not offered — MoE structure ({det.get('expert_count')} experts, {det.get('expert_used_count')} active, "
                              f"{int(det.get('expert_bytes') or 0) / _GIB:.2f} GiB experts) but the transformers loader cannot split "
                              f"experts to RAM; serve the GGUF form for an expert split")}
        det = None
    if det:
        out["moe"] = moe_split(det, geo, size, b, engine=engine)
        shared = default_allocation(engine, size, b["gpu_budget"], b["ram_budget"], moe=det, bnb=False, moe_force=True)
        out["moe"]["shared_answer"] = (shared or {}).get("mode")
        if out["auto"] is not None and out["moe"].get("offered"):
            out["auto"]["moe_explicit"] = {"mode": "explicit", "spill": out["moe"]["spill"], "why": out["moe"]["why"],
                                           "gpu_bytes": out["moe"]["gpu_bytes"], "ram_bytes": out["moe"]["ram_bytes"], "split": True}
    out["memory"] = memory_plan(engine, v, geo, det, b, out.get("moe"))
    # bitsandbytes 4-bit (transformers only): the SAME verdict shape priced at
    # the 4-bit size — weights, experts and non-expert scaled alike — so the
    # bnb knob selects memory.bnb_4bit[mode] / moe.bnb_4bit exactly as the
    # MoE knob selects explicit (operator 2026-10-02: Qwen3.6-35B-A3B
    # "exceeds feasibility under certain circumstances ... moe and 4-bit").
    bnb = v.get("bnb_4bit") or {}
    if bnb.get("eligible") and bnb.get("bytes") and size:
        ratio = float(bnb["bytes"]) / float(size)
        v4 = dict(v, size_bytes=int(bnb["bytes"]))
        det4 = None
        if det:
            det4 = dict(det, expert_bytes=int(det.get("expert_bytes") or 0) * ratio,
                        non_expert_bytes=int(int(det.get("non_expert_bytes") or 0) * ratio),
                        expert_bytes_by_layer={k: int(int(x) * ratio) for k, x in (det.get("expert_bytes_by_layer") or {}).items()})
        moe4 = moe_split(det4, geo, int(bnb["bytes"]), b, engine=engine) if det4 else {"offered": False, "why": "dense — no expert structure"}
        out["moe"]["bnb_4bit"] = moe4
        if out["auto"] is not None and moe4.get("offered"):
            out["auto"]["bnb_moe_explicit"] = {"mode": "explicit", "spill": dict(moe4["spill"], bnb_4bit=True), "why": moe4["why"],
                                               "gpu_bytes": moe4["gpu_bytes"], "ram_bytes": moe4["ram_bytes"], "split": True}
        out["memory"]["bnb_4bit"] = memory_plan(engine, v4, geo, det4, b, moe4)
    return out


def compute_plan(model: dict, inputs: dict, earmark: dict | None = None, keep: dict | None = None) -> dict:
    """Layer 2 for one worker: one verdict PER QUANT (``by_quant``), plus the
    pair-level summary (``fits`` = any quant fits; ``moe.offered`` = any quant
    offers the split). The knob ``gguf_file`` picks which quant serves.
    ``keep`` = {file: prior verdict} carried over unchanged (operator rule: on a
    budget REDUCTION an unfeasible verdict stays unfeasible — nothing is re-priced
    for it until a resource expands)."""
    w = model.get("weights") or {}
    engine = inputs["engine"]
    b = budgets_from_earmark(earmark or inputs)
    variants = w.get("variants") or {}
    out = {"version": PLAN_VERSION, "inputs": inputs, "computed_at": time.time(), "budgets": b,
           "default_variant": w.get("default_variant"), "by_quant": {},
           "feasible": {"fits": None, "ok": False, "modes": [], "why": ""}, "auto": None,
           "moe": {"offered": False, "why": "dense — no expert structure"}}
    for name, v in variants.items():
        if keep and name in keep:
            kept = dict(keep[name]); kept["kept"] = {"since_rev": kept.get("budget_rev"), "at_rev": inputs.get("budget_rev"),
                                                     "why": "budget reduced — an unfeasible verdict stays until a resource expands"}
            out["by_quant"][name] = kept
            continue
        out["by_quant"][name] = _plan_variant(engine, v, b, name)
    if not out["by_quant"]:
        out["feasible"]["why"] = w.get("error") or "no weights facts"
        return out
    vals = list(out["by_quant"].values())
    fits_list = [x["feasible"]["fits"] for x in vals]
    out["feasible"]["fits"] = True if any(f is True for f in fits_list) else (None if any(f is None for f in fits_list) else False)
    out["feasible"]["ok"] = any(x["feasible"]["ok"] for x in vals)
    out["feasible"]["modes"] = sorted({m for x in vals for m in x["feasible"]["modes"]})
    n_fit = sum(1 for f in fits_list if f is True)
    out["feasible"]["why"] = f"{n_fit}/{len(vals)} quants fit this worker's budgets"
    offered = [n for n, x in out["by_quant"].items() if x["moe"].get("offered")]
    if any(x["moe"].get("why", "").startswith("dense") is False for x in vals):
        out["moe"] = {"offered": bool(offered), "quants": offered,
                      "why": (f"offered for {len(offered)}/{len(vals)} quants: {', '.join(offered)}" if offered
                              else "not offered — no quant lands an explicit split on this worker")}
    d = out["by_quant"].get(w.get("default_variant") or "") or vals[0]
    out["auto"] = d.get("auto")
    return out


def selected_variant(plan: dict | None, knobs: dict | None) -> dict | None:
    """The per-quant verdict the knobs actually pick: knob gguf_file (basename
    match) else the default variant."""
    bq = (plan or {}).get("by_quant") or {}
    if not bq:
        return None
    want = (knobs or {}).get("gguf_file")
    if want:
        base = os.path.basename(str(want))
        if base in bq:
            return bq[base]
    return bq.get((plan or {}).get("default_variant") or "") or next(iter(bq.values()))


from hugpy_engine.model_index.query_registry import PAIR_KNOB_KEYS as KNOB_KEYS, KV_CACHE_TYPES  # noqa: E402  one truth with central
