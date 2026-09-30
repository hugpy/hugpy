"""THE GGUF need function (2026-09-30): what a llama-server load of a GGUF puts
on the GPU and in host RAM, priced from the file's typed structure
(``hugpy_storage.gguf_inspect.gguf_structure``) — never a flat file size for a
MoE, never a ``layers x dense`` KV for a hybrid.

One source of truth: the worker's admission (``_incoming_need_detail``), the
allocation rows, central's placement preflight (``_worker_fit``), the planned
split on the worker row and the Ctx slider preview all call :func:`gguf_need`.

Placement follows llama.cpp exactly:
  * ``--n-gpu-layers L`` puts blocks ``i >= n_block - L`` on the GPU; the
    output tensor goes to the GPU only when ``L > n_block`` (``-1``/all does);
    ``token_embd`` is ALWAYS on the CPU (llama.cpp's input device); a tied
    output (no ``output.weight``) duplicates ``token_embd`` onto the output
    device.
  * ``--n-cpu-moe N`` keeps the expert tensors (``ffn_{up,down,gate,gate_up}_exps``)
    of blocks ``i < N`` on the CPU; every always-on tensor of a GPU block stays
    on the GPU.
  * KV / recurrent state lives with its block (offload_kqv default): full
    attention = ``ctx x n_head_kv x (k_len + v_len) x dtype``; sliding-window
    blocks cap the cells at ``window + ubatch``; linear-attention (Gated
    DeltaNet / Mamba) blocks hold a fixed f32 state per sequence, independent
    of ctx.

The weights margin applies to the WEIGHTS of each side separately (measured
per side when a record exists, else the prior); KV, state, the compute
cushion and an mmproj projector are added unmultiplied. A dense model keeps
the historical pricing (file bytes x margin on the GPU when fully offloaded),
so dense admission is byte-for-byte unchanged.
"""
from __future__ import annotations

from typing import Any, Optional

COMPUTE_CUSHION_BYTES = 512 * 2 ** 20            # spill._CTX_COMPUTE_RESERVE_BYTES
SWA_UBATCH = 512                                  # llama.cpp default n_ubatch


def structure_for(path: Optional[str]) -> dict:
    """``gguf_structure`` of the served file, ``{}`` when unreadable."""
    if not path:
        return {}
    try:
        from hugpy_storage.gguf_inspect import gguf_structure
        return gguf_structure(path) or {}
    except Exception:  # noqa: BLE001 — a structure gap degrades to opaque pricing
        return {}


def _layers(struct: dict) -> dict:
    return {int(k): v for k, v in (struct.get("layers") or {}).items()}


def kv_breakdown(struct: dict, ctx: Optional[int], *, dtype_bytes: float = 2.0,
                 n_seq: int = 1, gpu_layers: Optional[set] = None) -> dict:
    """Context cost by attention kind: ``{kv_bytes, kv_full_bytes, kv_swa_bytes,
    state_bytes, layers: {full, swa, linear}, per_token_bytes}``; with
    ``gpu_layers`` also the GPU-side share (``gpu_bytes``)."""
    ctx = max(0, int(ctx or 0))
    win = struct.get("swa_window")
    swa_cells = min(ctx, int(win) + SWA_UBATCH) if win else ctx
    full = swa = state = gpu = 0
    per_tok = 0
    counts = {"full": 0, "swa": 0, "linear": 0}
    for i, L in _layers(struct).items():
        kind = L.get("attn")
        e = int(L.get("kv_elems_per_token") or 0)
        b = 0
        if kind == "full":
            b = int(ctx * e * dtype_bytes)
            full += b
            per_tok += int(e * dtype_bytes)
        elif kind == "swa":
            b = int(swa_cells * e * dtype_bytes)
            swa += b
        elif kind == "linear":
            b = int(L.get("state_bytes") or 0) * max(1, int(n_seq))
            state += b
        if kind in counts:
            counts[kind] += 1
        if gpu_layers is not None and i in gpu_layers:
            gpu += b
    out = {"ctx": ctx, "kv_bytes": full + swa + state, "kv_full_bytes": full,
           "kv_swa_bytes": swa, "state_bytes": state, "layers": counts,
           "per_token_bytes": per_tok, "swa_window": win}
    if gpu_layers is not None:
        out["gpu_bytes"] = gpu
    return out


def gguf_need(struct: dict, *, ctx: Optional[int], n_cpu_moe: Optional[int] = 0,
              n_gpu_layers: int = -1, gpu_margin: float = 1.15,
              ram_margin: float = 1.15, dtype_bytes: float = 2.0,
              cushion_bytes: int = COMPUTE_CUSHION_BYTES, mmproj_bytes: int = 0,
              n_seq: int = 1) -> dict:
    """GPU / RAM bytes of a llama-server load at ``ctx`` (see module doc)."""
    layers = _layers(struct)
    n_block = int(struct.get("block_count") or len(layers))
    ngl = int(n_gpu_layers if n_gpu_layers is not None else -1)
    if ngl < 0 or ngl > n_block:
        start, out_on_gpu = 0, True
    else:
        start, out_on_gpu = n_block - ngl, False
    gpu_layers = {i for i in layers if i >= start} if ngl != 0 else set()
    is_moe = bool(struct.get("is_moe"))
    ncm = max(0, int(n_cpu_moe or 0)) if is_moe else 0
    gw = rw = 0
    experts_gpu = experts_cpu = 0
    for i, L in layers.items():
        a, e = int(L.get("always_bytes") or 0), int(L.get("expert_bytes") or 0)
        if i in gpu_layers:
            gw += a
            if i < ncm:
                rw += e
                experts_cpu += e
            else:
                gw += e
                experts_gpu += e
        else:
            rw += a + e
            experts_cpu += e
    embd = int(struct.get("token_embd_bytes") or 0)
    outb = int(struct.get("output_bytes") or 0)
    glob = int(struct.get("global_bytes") or 0)
    rw += embd                                     # input layer: always CPU
    if out_on_gpu:
        gw += outb + glob + (embd if struct.get("output_tied") else 0)
    else:
        rw += outb + glob
    file_bytes = int(struct.get("file_bytes") or 0)
    if not is_moe and out_on_gpu and file_bytes:
        # Dense, fully offloaded: the historical pricing (file x margin on the
        # GPU) — dense admission stays byte-identical.
        gw, rw = file_bytes, 0
    kv = kv_breakdown(struct, ctx, dtype_bytes=dtype_bytes, n_seq=n_seq,
                      gpu_layers=gpu_layers)
    kv_gpu = int(kv.get("gpu_bytes") or 0)
    kv_cpu = int(kv["kv_bytes"]) - kv_gpu
    # Compute cushion on the GPU for a MoE split (dense pricing never carried one).
    cushion = int(cushion_bytes) if (gpu_layers and is_moe) else 0
    gpu_total = int(gw * float(gpu_margin)) + kv_gpu + cushion + int(mmproj_bytes or 0)
    ram_total = int(rw * float(ram_margin)) + kv_cpu
    return {"n_cpu_moe": (min(ncm, n_block) if is_moe else None),
            "n_gpu_layers": ngl, "block_count": n_block, "is_moe": is_moe,
            "ctx": kv["ctx"],
            "gpu_bytes": gpu_total, "ram_bytes": ram_total,
            "gpu_weights_bytes": int(gw), "ram_weights_bytes": int(rw),
            "gpu_margin": float(gpu_margin), "ram_margin": float(ram_margin),
            "experts_gpu_bytes": int(experts_gpu), "experts_cpu_bytes": int(experts_cpu),
            "kv_bytes": int(kv["kv_bytes"]), "kv_gpu_bytes": kv_gpu, "kv_cpu_bytes": kv_cpu,
            "kv_full_bytes": kv["kv_full_bytes"], "kv_swa_bytes": kv["kv_swa_bytes"],
            "state_bytes": kv["state_bytes"], "kv_layers": kv["layers"],
            "cushion_bytes": cushion, "mmproj_bytes": int(mmproj_bytes or 0),
            # mmap: the whole file is mapped; pages of GPU-uploaded tensors stay
            # in page cache (RSS) until reclaimed — reclaimable, not a need.
            "mmap_bytes": file_bytes}


def choose_n_cpu_moe(struct: dict, room_bytes: Optional[int], **kw) -> dict:
    """The SMALLEST ``n_cpu_moe`` whose GPU side fits ``room_bytes`` whole
    (GPU bytes fall monotonically as N grows). ``fits`` False -> every expert
    on the CPU and the backbone still does not fit (N = block_count)."""
    n_block = int(struct.get("block_count") or 0)
    if room_bytes is None:
        need = gguf_need(struct, n_cpu_moe=n_block, **kw)
        return dict(need, fits=None, choice="unmeasurable card — all experts on CPU")
    for n in range(0, n_block + 1):
        need = gguf_need(struct, n_cpu_moe=n, **kw)
        if need["gpu_bytes"] <= int(room_bytes):
            return dict(need, fits=True, room_bytes=int(room_bytes),
                        choice=f"smallest n_cpu_moe fitting {int(room_bytes)} B")
    need = gguf_need(struct, n_cpu_moe=n_block, **kw)
    return dict(need, fits=False, room_bytes=int(room_bytes),
                choice="backbone does not fit even with every expert on CPU")


def whole_seat_ctx(need_at, room: int, upper: int, floor: int = 4096) -> "tuple[int, bool]":
    """Largest ctx (1024 multiple in [floor, upper]) with ``need_at(ctx) <= room``."""
    upper = max(int(floor), int(upper))
    top = upper if upper < 1024 else (upper // 1024) * 1024
    if need_at(top) <= room:
        return top, True
    lo, hi, best = -(-int(floor) // 1024), top // 1024, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if need_at(mid * 1024) <= room:
            best, lo = mid * 1024, mid + 1
        else:
            hi = mid - 1
    if best is None:
        return int(floor), need_at(int(floor)) <= room
    return max(int(floor), best), True


def verdict_text(need: dict) -> str:
    """"MoE: 12/48 layers' experts on CPU; GPU 17.2 GiB, RAM 21.0 GiB" (+ KV)."""
    g = 2 ** 30
    if not need.get("is_moe"):
        return (f"GPU {need['gpu_bytes'] / g:.1f} GiB, RAM {need['ram_bytes'] / g:.1f} GiB "
                f"(KV {need['kv_bytes'] / g:.2f} GiB @ {need['ctx']})")
    return (f"MoE: {need['n_cpu_moe']}/{need['block_count']} layers' experts on CPU; "
            f"GPU {need['gpu_bytes'] / g:.1f} GiB, RAM {need['ram_bytes'] / g:.1f} GiB "
            f"(KV {need['kv_bytes'] / g:.2f} GiB @ {need['ctx']} ctx)")
