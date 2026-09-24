"""F3.1 — model metadata + recommended settings, from ONE place (CON-03).

Every model-selection surface (HugPy Console ModelTable, media-intelligence
ComposerControls, Discord embeds) reads this — no UI hardcodes a model fact.
The registry (ModelConfig) stays the identity source; this module derives
the operational metadata the roadmap wants surfaced everywhere:

    size_bytes    real on-disk footprint (all shards; cached by dir mtime)
    quant         parsed from the GGUF filename (q4_k_m, iq3_xs, f16, ...)
    params_b      billions of parameters, parsed from name/hub_id (7B, 0.5b)
    ctx_max       the model's context window (model_max_length)
    recommended   settings annotated against a target's VRAM — not static
                  constants: ctx, n_gpu_layers, threads, and the reason.

Honesty rule: when we don't know, fields are None and the reason says why —
a picker that shows "?" beats one that shows a confident wrong number.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Optional

# quant tokens as they appear in GGUF filenames, longest-match first.
_QUANT_RE = re.compile(
    r"(?i)[-._](i?q[0-9]+(?:_[a-z0-9]+)*|f16|f32|bf16|fp16|fp32)(?=[-._]|$)")
# parameter counts: 7B, 0.5b, 14B, 1.5B — bounded so '2024' never matches.
_PARAMS_RE = re.compile(r"(?i)(?<![0-9.])([0-9]{1,3}(?:\.[0-9]+)?)\s?b(?![a-z0-9])")

# Rough default VRAM headroom multiplier (weights + KV cache + overhead).
_VRAM_HEADROOM = 1.15

_SIZE_CACHE: dict[str, tuple[float, int]] = {}
_SIZE_LOCK = threading.Lock()


def parse_quant(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    m = _QUANT_RE.search(filename)
    return m.group(1).lower() if m else None


def parse_params_b(*names: Optional[str]) -> Optional[float]:
    for name in names:
        if not name:
            continue
        m = _PARAMS_RE.search(name)
        if m:
            try:
                v = float(m.group(1))
                if 0.05 <= v <= 700:
                    return v
            except ValueError:
                continue
    return None


from hugpy_storage.model_presence import dir_size_bytes  # noqa: E402  (storage owns the footprint cache)


def _model_dir(cfg: Any) -> Optional[str]:
    """Absolute model directory from a ModelConfig(-ish) object or dict.

    Prefer the ROUTED install destination (route_destination) — the same path the
    manifest records and the weights-presence check uses — because a discovered
    model's ``cfg.folder`` can be stale and resolve to a near-empty dir (which made
    the GGUF size read as a tiny stub instead of the real quant). Fall back to
    MODELS_HOME/folder only when the routed dir doesn't exist on disk."""
    d = cfg if isinstance(cfg, dict) else (cfg.to_dict() if hasattr(cfg, "to_dict") else {})
    try:
        from hugpy_storage.model_paths import route_destination
        dest = route_destination(d)
        if dest and os.path.isdir(dest):
            return dest
    except Exception:
        pass
    folder = d.get("folder") if isinstance(d, dict) else getattr(cfg, "folder", None)
    if not folder:
        return None
    if os.path.isabs(folder):
        return folder
    try:
        from hugpy_platform.constants import MODELS_HOME
        return os.path.join(str(MODELS_HOME), folder)
    except Exception:
        return None


def recommended_settings(*, size_bytes: Optional[int], ctx_max: Optional[int],
                         framework: str = "",
                         vram_bytes: Optional[int] = None) -> dict:
    """Recommendations annotated against a target's VRAM (CON-03: not static
    constants). vram_bytes None = no target known; recommendations degrade
    honestly and the reason explains what's missing."""
    rec: dict[str, Any] = {
        "ctx": min(int(ctx_max), 32768) if ctx_max else None,
        "n_gpu_layers": None,
        "threads": max(2, min(8, (os.cpu_count() or 4) // 2)),
        "fits_vram": None,
        "reason": "",
    }
    if size_bytes is None:
        rec["reason"] = "model size unknown (not downloaded?) — no offload advice"
        return rec
    need = int(size_bytes * _VRAM_HEADROOM)
    rec["need_bytes"] = need
    if vram_bytes is None:
        rec["reason"] = "no target VRAM given — pass ?vram_gib= for offload advice"
        return rec
    if need <= vram_bytes:
        rec["n_gpu_layers"] = -1
        rec["fits_vram"] = True
        rec["reason"] = "fits fully in VRAM (size×%.2f ≤ free)" % _VRAM_HEADROOM
    else:
        frac = max(0.0, min(1.0, vram_bytes / need))
        rec["fits_vram"] = False
        rec["gpu_fraction"] = round(frac, 2)
        rec["reason"] = ("~%d%% of weights fit — partial offload; exact "
                         "n_gpu_layers depends on layer count (use probe)"
                         % int(frac * 100))
    return rec


def model_meta(cfg: Any, *, vram_bytes: Optional[int] = None,
               select_gguf: Optional[str] = None) -> dict:
    """The one metadata dict every picker renders. cfg = ModelConfig or its
    to_dict().

    ``select_gguf`` (2026-09-23, the per-worker quant dropdown): size + recommend for
    THAT variant (basename or quant token) instead of the effective one, so the
    console can show the fit of a quant on a worker before/after pinning it
    (``select_gguf``).
    Only a COMPLETE on-disk variant is accepted; otherwise ``selected_gguf_error``
    says why and the effective sizing stands."""
    d = cfg if isinstance(cfg, dict) else cfg.to_dict()
    name = d.get("name") or ""
    hub_id = d.get("hub_id") or ""
    filename = d.get("filename")
    ctx_max = d.get("model_max_length")
    framework = d.get("framework") or ""
    model_dir = _model_dir(d)
    dir_bytes = dir_size_bytes(model_dir)          # whole dir (every file/variant)

    # A GGUF repo often holds several quantizations but only ONE serves, so the
    # dir sum badly overstates "the model" — and it flowed straight into the VRAM
    # recommendation. Resolve the single effective quant (+ its mmproj) exactly as
    # the runner will, honoring the operator's gguf_file choice, and size by that.
    size = dir_bytes
    gguf = {}
    if str(framework).lower() in ("gguf", "llama_cpp") and model_dir:
        try:
            from hugpy_engine.serve.overrides import gguf_variants_detail
            gguf = gguf_variants_detail(d.get("model_key") or "", model_dir, cfg) or {}
        except Exception:  # noqa: BLE001 — best-effort; keep the dir size
            gguf = {}
        if gguf.get("effective_bytes"):
            size = gguf["effective_bytes"]

    selected = selected_err = None
    gguf_sel = str(select_gguf or "").strip()
    if gguf_sel:
        want = os.path.basename(gguf_sel).lower()
        variants = [v for v in (gguf.get("variants") or [])
                    if isinstance(v, dict) and v.get("complete") is not False]
        hits = [v for v in variants if str(v.get("filename") or "").lower() == want] or \
               [v for v in variants if want in str(v.get("filename") or "").lower()]
        if len(hits) == 1:
            selected = hits[0]
            size = selected.get("bytes") or size
        else:
            selected_err = (f"{len(hits)} complete GGUF variants on central match {gguf_sel!r}"
                            if hits else f"no complete GGUF variant on central matches {gguf_sel!r}"
                            f" ({len(variants)} complete variant(s) listed)")

    out = {
        "model_key": d.get("model_key"),
        "size_bytes": size,                        # effective quant for GGUF
        "dir_bytes": dir_bytes,                     # whole-dir footprint (all variants)
        "quant": parse_quant(gguf.get("effective_gguf") or filename),
        "params_b": parse_params_b(name, hub_id, filename),
        "ctx_max": ctx_max,
        "framework": framework,
        "recommended": recommended_settings(
            size_bytes=size, ctx_max=ctx_max, framework=framework,
            vram_bytes=vram_bytes),
    }
    if selected is not None:
        out["selected_gguf"] = selected.get("filename")
        out["selected_bytes"] = selected.get("bytes")
    if selected_err:
        out["selected_gguf_error"] = selected_err
    if gguf:
        out["effective_gguf"] = gguf.get("effective_gguf")
        out["gguf_variants"] = gguf.get("variants") or []
        out["mmproj_bytes"] = gguf.get("mmproj_bytes")
    return out
