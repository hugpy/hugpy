"""k120 — the weights, quant by quant, with a VRAM number for each.

``review/screen.py`` already sizes ONE quant: the biggest acceptable variant
that fits the card, because that is all a pass/fail screen needs. A dossier
needs the whole ladder — "Q4_K_M fits with 6 GiB spare, Q6_K fits with 0.4,
Q8_0 does not" is the trade an operator actually makes, and hiding it behind a
single ``best_quant`` is how a 24 GB card ends up running a model at Q4 that
would have run happily at Q6.

WHAT IS REUSED AND WHAT IS NEW
    The KV-cache maths and the compute-buffer constant are ``screen``'s and
    are IMPORTED, never re-derived — two VRAM estimators that disagree by a
    gigabyte is a bug that only shows up at 3am on the timer. New here is
    the per-quant expansion, the bits-per-weight table, and the param-count
    fallback that lets a GGUF-only repo (which ships no safetensors index and
    therefore no parameter total) still report a size.

HONESTY
    ``est_kv_bytes`` is None when the geometry is unknown, and the resulting
    ``est_vram_bytes`` then carries a ``note`` saying it EXCLUDES the KV cache
    and reads low. A VRAM figure that quietly omits a term is worse than no
    figure: it passes the fit check and then OOMs on load.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from hugpy_curation.dossier.cards import architecture_family
from hugpy_curation.dossier.dossier import QuantFact, WeightsFacts

#: Effective bits per weight, INCLUDING the per-block scale/zero overhead that
#: makes a "4-bit" GGUF actually ~4.8 bpw. Numbers are llama.cpp's own
#: published figures for the K-quants; anything unlisted falls back to the
#: leading digit of the name, which is the right order of magnitude and is
#: marked as an estimate.
BITS_PER_WEIGHT: dict[str, float] = {
    "IQ1_S": 1.56, "IQ1_M": 1.75,
    "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7,
    "Q2_K": 2.63, "Q2_K_S": 2.16,
    "IQ3_XXS": 3.06, "IQ3_XS": 3.3, "IQ3_S": 3.44, "IQ3_M": 3.66,
    "Q3_K_S": 3.44, "Q3_K_M": 3.74, "Q3_K_L": 4.03, "Q3_K": 3.74,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q4_0": 4.55, "Q4_1": 5.0, "Q4_K_S": 4.58, "Q4_K_M": 4.85, "Q4_K": 4.85,
    "Q5_0": 5.54, "Q5_1": 6.0, "Q5_K_S": 5.52, "Q5_K_M": 5.69, "Q5_K": 5.69,
    "Q6_K": 6.56,
    "Q8_0": 8.5,
    "F16": 16.0, "BF16": 16.0, "F32": 32.0,
}

#: Size hints a publisher puts in the repo name when the metadata has none.
_PARAM_IN_NAME = re.compile(r"(?<![\d.])(\d{1,4}(?:\.\d)?)\s*([bBmM])(?![a-z])")

#: A MoE repo names its ACTIVE params too (``35B-A3B``). Both matter: total
#: params drive the file size, active params drive the speed.
_MOE_ACTIVE = re.compile(r"[-_](?:A|a)(\d{1,3}(?:\.\d)?)[bB]\b")


def bits_per_weight(quant: str | None) -> tuple[float | None, bool]:
    """``(bpw, exact)``. ``exact`` is False for the leading-digit fallback, so
    a caller can label the estimate rather than present it as measured."""
    if not quant:
        return None, False
    key = quant.upper()
    if key in BITS_PER_WEIGHT:
        return BITS_PER_WEIGHT[key], True
    m = re.match(r"^I?Q(\d+)", key)
    if m:
        return float(m.group(1)) + 0.5, False
    return None, False


def params_from_name(hub_id: str) -> tuple[int | None, int | None]:
    """``(total_params, active_params)`` read off the repo name, or (None,
    None). Last-resort only — a name is a claim, and the caller records
    ``params_source='name'`` so the UI can say where the number came from."""
    tail = hub_id.split("/")[-1]
    active = None
    m = _MOE_ACTIVE.search(tail)
    if m:
        active = int(float(m.group(1)) * 1e9)
    best = None
    for match in _PARAM_IN_NAME.finditer(tail):
        value = float(match.group(1))
        scale = 1e9 if match.group(2).lower() == "b" else 1e6
        # Skip the A3B half of a MoE name; it is the ACTIVE count, not total.
        if active is not None and abs(value * scale - active) < 1:
            continue
        candidate = int(value * scale)
        if candidate < 1e7 or candidate > 2e12:
            continue
        best = candidate if best is None else max(best, candidate)
    return best, active


def params_from_bytes(total_bytes: int | None, quant: str | None
                      ) -> int | None:
    """Invert the quant maths: params ~= bytes * 8 / bpw. Coarse — shards
    carry metadata and a token embedding is often kept at higher precision —
    but it is within ~10% and it is the ONLY parameter signal a GGUF-only repo
    gives us."""
    bpw, _exact = bits_per_weight(quant)
    if not total_bytes or not bpw:
        return None
    return int(total_bytes * 8 / bpw)


def geometry_of(config: Mapping[str, Any]) -> dict[str, Any]:
    """The layer/head numbers a KV cache is sized from, named consistently."""
    cfg = config or {}
    return {k: v for k, v in {
        "num_hidden_layers": cfg.get("num_hidden_layers") or cfg.get("n_layer"),
        "hidden_size": cfg.get("hidden_size") or cfg.get("n_embd"),
        "num_attention_heads": cfg.get("num_attention_heads") or cfg.get("n_head"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "head_dim": cfg.get("head_dim"),
        "vocab_size": cfg.get("vocab_size"),
        "rope_theta": cfg.get("rope_theta"),
        "num_experts": cfg.get("num_experts") or cfg.get("num_local_experts"),
    }.items() if v is not None}


def _kv_bytes(config: Mapping[str, Any], context: int) -> int | None:
    """screen.kv_cache_bytes, imported. See the module docstring for why this
    is not reimplemented."""
    try:
        from hugpy_curation.review.screen import kv_cache_bytes
    except Exception:                              # noqa: BLE001
        return None
    try:
        return kv_cache_bytes(dict(config or {}), int(context))
    except Exception:                              # noqa: BLE001
        return None


def _overhead_bytes() -> int:
    try:
        from hugpy_curation.review.screen import COMPUTE_OVERHEAD_BYTES
        return int(COMPUTE_OVERHEAD_BYTES)
    except Exception:                              # noqa: BLE001
        return 1_200 * 1024 ** 2


def quant_facts(quants: Sequence[Mapping[str, Any]],
                config: Mapping[str, Any] | None = None,
                context: int = 16384,
                vram_budget_bytes: int | None = None,
                params: int | None = None) -> tuple[QuantFact, ...]:
    """Expand the screen's quant list into a full ladder with a VRAM estimate
    each.

    ``quants`` is ``ScreenResult.quants`` (dicts with ``quant``/``bytes``/
    ``files``). Every entry gets its own KV term at the SAME target context,
    which is what makes two quants of two different repos comparable at all.
    """
    kv = _kv_bytes(config or {}, context)
    overhead = _overhead_bytes()
    out: list[QuantFact] = []
    for row in quants or ():
        name = str(row.get("quant") or "")
        size = row.get("bytes")
        size = int(size) if isinstance(size, (int, float)) and size else None
        bpw, exact = bits_per_weight(name)
        notes: list[str] = []
        if size is None and params and bpw:
            size = int(params * bpw / 8)
            notes.append("file size not reported by the hub — weights bytes "
                         "estimated from the parameter count")
        est_vram = None
        if size is not None:
            est_vram = size + (kv or 0) + overhead
        if kv is None and est_vram is not None:
            notes.append("KV cache size unknown (no config geometry) — this "
                         "estimate EXCLUDES it and reads low")
        if bpw is not None and not exact:
            notes.append(f"{name} is not in the bits-per-weight table; "
                         f"{bpw} bpw assumed from the name")
        fits = None
        if est_vram is not None and vram_budget_bytes:
            fits = est_vram <= vram_budget_bytes
        out.append(QuantFact(
            quant=name, bytes=size,
            files=tuple(str(f) for f in (row.get("files") or ())),
            bits_per_weight=bpw, est_weights_bytes=size, est_kv_bytes=kv,
            est_vram_bytes=est_vram, fits_vram=fits, note="; ".join(notes)))
    out.sort(key=lambda q: (q.bytes if q.bytes is not None else math.inf,
                            q.quant))
    return tuple(out)


def build_weights(hub_id: str, screen_row: Mapping[str, Any],
                  config: Mapping[str, Any] | None = None,
                  vram_budget_bytes: int | None = None,
                  target_context: int = 16384,
                  tokenizer: str | None = None) -> WeightsFacts:
    """The physical facts, with every number carrying where it came from.

    ``screen_row`` is ``ScreenResult.to_dict()``. ``config`` is the repo's (or
    its base model's) config.json when one was fetched."""
    cfg = dict(config or {})
    notes: list[str] = []

    params = screen_row.get("params")
    source = "safetensors" if params else None
    if not params:
        by_name, active = params_from_name(hub_id)
        if by_name:
            params, source = by_name, "name"
            if active:
                notes.append(f"repo name declares a mixture-of-experts with "
                             f"~{active/1e9:.1f}B active parameters")
    if not params:
        by_bytes = params_from_bytes(screen_row.get("total_bytes"),
                                     screen_row.get("best_quant"))
        if by_bytes:
            params, source = by_bytes, "file-bytes"
    if not params:
        notes.append("parameter count is unknown — the hub reports no "
                     "safetensors index and the name declares no size")

    context = screen_row.get("context_length")
    context_source = "config.json" if context else None
    if not context and isinstance(cfg.get("max_position_embeddings"), int):
        context, context_source = cfg["max_position_embeddings"], "config.json"

    architecture = screen_row.get("architecture")
    quants = quant_facts(screen_row.get("quants") or (), cfg, target_context,
                         vram_budget_bytes, params)
    if not quants:
        notes.append("no GGUF quants published for this repo")

    return WeightsFacts(
        params=params, params_source=source,
        total_bytes=screen_row.get("total_bytes"),
        context_length=context, context_source=context_source,
        architecture=architecture,
        architecture_family=architecture_family(architecture, hub_id),
        tokenizer=tokenizer or cfg.get("tokenizer_class"),
        torch_dtype=cfg.get("torch_dtype"),
        geometry=geometry_of(cfg), quants=quants,
        best_quant=screen_row.get("best_quant"),
        vram_budget_bytes=vram_budget_bytes, target_context=target_context,
        notes=tuple(notes))


__all__ = ["BITS_PER_WEIGHT", "bits_per_weight", "build_weights",
           "geometry_of", "params_from_bytes", "params_from_name",
           "quant_facts"]
