"""review/screen.py — stage 1: judge a repo from metadata alone.

No weights are fetched here beyond config.json (a few KB), so a hundred
candidates can be screened for the cost of one download. Everything that would
disqualify a model on paper — won't fit the card at the target context, no
usable quant, too small to be interesting, stale, unvetted — is caught before
disk is committed.

Two attribute families, per the operator's brief:

  fit / runtime cost — params, quant, file bytes, context length, and the VRAM
      the thing would actually occupy at `target_context` (weights + KV cache +
      compute buffers), checked against the card.
  capability / quality — task, tags, downloads momentum, recency, base-model
      lineage against what the fleet already runs, and publisher trust.

Reuses the Flask app's existing HF helpers where they exist (trust tiers, the
permanent metadata cache) and degrades to direct hub calls when this is run
outside the app — the CLI and the timer must work with no Flask context.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field, asdict
from typing import Any

from hugpy_curation.review.criteria import ReviewCriteria

GGUF_QUANT_RE = re.compile(
    r"(IQ\d+_[A-Z]+|Q\d+_K_[MSL]|Q\d+_K|Q\d+_\d+|Q\d+_[01]|F16|BF16|F32)",
    re.IGNORECASE)

# Bytes of GPU memory llama.cpp needs beyond weights + KV for compute buffers,
# CUDA context and the graph. Measured empirically on the 3090; conservative.
COMPUTE_OVERHEAD_BYTES = 1_200 * 1024**2


# ── HF access (works with or without the Flask app) ────────────────────────
def _hf_api():
    from huggingface_hub import HfApi
    try:
        from hugpy_storage.hf_token import get_hf_token
        return HfApi(token=get_hf_token() or False)
    except Exception:
        # standalone (CLI/timer): env token, else anonymous — public metadata
        # needs no auth, a token only lifts the rate limit
        return HfApi(token=os.environ.get("HF_TOKEN") or False)


def _trust_tier(hub_id: str, author) -> int:
    """Publisher trust from the fleet's one curated allowlist, so the reviewer
    and the search UI can never disagree about who is first-party. Imported
    from the Flask-free constants module — going through search_routes needs an
    app context and silently returned 0 for everyone outside it."""
    from hugpy_platform.trust import trust_tier
    return trust_tier(hub_id, author)


def _repo_info(api, hub_id: str) -> dict[str, Any] | None:
    """Repo metadata, through the permanent central cache when available."""
    try:
        from hugpy_storage.model_metadata import fetch_repo_info
        payload = fetch_repo_info(hub_id, files_metadata=True, api=api)
        if payload:
            return payload
    except Exception:
        pass
    try:
        info = api.model_info(hub_id, files_metadata=True)
    except Exception:
        return None
    card = {}
    try:
        card = info.card_data.to_dict() if info.card_data else {}
    except Exception:
        card = {}
    return {
        "siblings": [{"rfilename": s.rfilename, "size": getattr(s, "size", None)}
                     for s in (info.siblings or [])],
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(getattr(info, "tags", None) or []),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "author": getattr(info, "author", None),
        "gated": getattr(info, "gated", None),
        "license": card.get("license"),
        "base_model": card.get("base_model"),
        "last_modified": str(getattr(info, "last_modified", "") or ""),
        "safetensors_params": _params_of(info),
    }


# Moved DOWN to hugpy_engine.model_classifier (2026-09-22): the engine's
# registry needs the same lineage parse and must not import curation.
from hugpy_engine.model_classifier import base_model_of  # noqa: E402,F401


def _params_of(info) -> int | None:
    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    total = getattr(st, "total", None)
    if isinstance(total, int):
        return total
    try:
        return sum(v for v in st.parameters.values() if isinstance(v, int))
    except Exception:
        return None


def _fetch_config(api, hub_id: str) -> dict[str, Any]:
    try:
        path = api.hf_hub_download(hub_id, "config.json")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def _config(api, hub_id: str, files: list[dict],
            base_model: str | None = None) -> tuple[dict[str, Any], str | None]:
    """config.json (a few KB) — the only place the real layer/head geometry
    lives, and therefore the only way to size a KV cache honestly.

    A GGUF-only repo usually ships NO config.json: the geometry is baked into
    the .gguf header instead. Those are exactly the repos this reviewer cares
    about, and without a fallback every one of them reported a 0-byte KV cache
    and an under-estimated VRAM figure. So fall back to the config of the
    base model the quantizer names in its card. Returns (config, source).
    """
    if any(f.get("rfilename") == "config.json" for f in files):
        cfg = _fetch_config(api, hub_id)
        if cfg:
            return cfg, hub_id
    if base_model and base_model != hub_id:
        cfg = _fetch_config(api, base_model)
        if cfg:
            return cfg, base_model
    return {}, None


# ── fit maths ──────────────────────────────────────────────────────────────
def kv_cache_bytes(cfg: dict, context: int) -> int | None:
    """fp16 KV cache for `context` tokens.

    2 tensors (K and V) x 2 bytes x layers x kv_dim x context, where kv_dim
    accounts for grouped-query attention — a GQA model's cache is a fraction of
    what the head count alone would suggest, and ignoring that overestimates a
    70B's footprint badly enough to reject models that do fit.
    """
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    hidden = cfg.get("hidden_size") or cfg.get("n_embd")
    heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    if not (layers and hidden and heads):
        return None
    kv_heads = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim") or (hidden // heads)
    return int(2 * 2 * layers * kv_heads * head_dim * context)


def context_length(cfg: dict) -> int | None:
    for k in ("max_position_embeddings", "n_positions", "max_sequence_length",
              "seq_length"):
        v = cfg.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None


def quant_of(path: str) -> str | None:
    m = GGUF_QUANT_RE.search(os.path.basename(path))
    return m.group(0).upper() if m else None


@dataclass
class QuantOption:
    """One installable GGUF variant of a repo."""
    quant: str
    bytes: int
    files: list[str]
    est_vram_bytes: int | None = None
    fits_vram: bool | None = None

    def to_dict(self):
        return asdict(self)


def gguf_options(files: list[dict]) -> list[QuantOption]:
    groups: dict[str, list[dict]] = {}
    for f in files:
        p = f.get("rfilename") or ""
        if not p.lower().endswith(".gguf"):
            continue
        # a vision projector is a sidecar, not a weight variant
        if os.path.basename(p).lower().startswith("mmproj"):
            continue
        groups.setdefault(quant_of(p) or p, []).append(f)
    out = []
    for quant, group in groups.items():
        total = sum(g.get("size") or 0 for g in group)
        out.append(QuantOption(quant=quant, bytes=total,
                               files=sorted(g["rfilename"] for g in group)))
    return sorted(out, key=lambda o: o.bytes)


@dataclass
class ScreenResult:
    hub_id: str
    passed: bool = False
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)   # why it failed
    notes: list[str] = field(default_factory=list)     # why it's interesting
    # fit
    params: int | None = None
    context_length: int | None = None
    architecture: str | None = None
    total_bytes: int | None = None
    quants: list[dict] = field(default_factory=list)
    best_quant: str | None = None
    est_vram_bytes: int | None = None
    kv_bytes: int | None = None
    # capability
    task: str | None = None
    tags: list[str] = field(default_factory=list)
    downloads: int | None = None
    likes: int | None = None
    trust_tier: int = 0
    base_model: str | None = None
    lineage_of: str | None = None
    license: str | None = None
    gated: Any = None
    last_modified: str | None = None
    age_days: int | None = None

    def to_dict(self):
        return asdict(self)


def _age_days(last_modified: str | None) -> int | None:
    if not last_modified:
        return None
    try:
        s = last_modified.replace("Z", "+00:00")
        dt = _dt.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return max(0, (_dt.datetime.now(_dt.timezone.utc) - dt).days)
    except Exception:
        return None


def screen(hub_id: str, crit: ReviewCriteria, api=None) -> ScreenResult:
    """Score one repo against the criteria without downloading weights."""
    api = api or _hf_api()
    r = ScreenResult(hub_id=hub_id)

    payload = _repo_info(api, hub_id)
    if not payload:
        r.reasons.append("no metadata from the hub")
        return r

    files = [s for s in (payload.get("siblings") or []) if s.get("rfilename")]

    bm = base_model_of(payload)
    cfg, cfg_source = _config(api, hub_id, files, base_model=bm)

    r.task = payload.get("pipeline_tag")
    r.tags = list(payload.get("tags") or [])
    r.downloads = payload.get("downloads")
    r.likes = payload.get("likes")
    r.license = payload.get("license")
    r.gated = payload.get("gated")
    r.last_modified = payload.get("last_modified")
    r.age_days = _age_days(r.last_modified)
    r.trust_tier = _trust_tier(hub_id, payload.get("author"))
    r.params = payload.get("safetensors_params")
    r.architecture = (cfg.get("architectures") or [None])[0] \
        if isinstance(cfg.get("architectures"), list) else cfg.get("model_type")
    r.context_length = context_length(cfg)
    r.total_bytes = sum(f.get("size") or 0 for f in files) or None

    r.base_model = bm
    if cfg_source and cfg_source != hub_id:
        r.notes.append(f"geometry read from base model {cfg_source} "
                       f"(this repo ships no config.json)")

    # ── fit ───────────────────────────────────────────────────────────────
    opts = gguf_options(files)
    kv = kv_cache_bytes(cfg, crit.target_context)
    r.kv_bytes = kv
    if kv is None and opts:
        # Unknown geometry: the VRAM figure below is weights + overhead only.
        # Say so rather than quietly presenting an under-estimate as a fit.
        r.notes.append("KV cache size unknown (no config.json anywhere) — "
                       "VRAM estimate excludes it and reads low")
    budget = crit.usable_vram_bytes
    for o in opts:
        if o.bytes:
            o.est_vram_bytes = o.bytes + (kv or 0) + COMPUTE_OVERHEAD_BYTES
            o.fits_vram = o.est_vram_bytes <= budget
    r.quants = [o.to_dict() for o in opts]

    allowed = {q.upper() for q in crit.allowed_quants}
    # biggest quant that still fits = best quality within budget
    viable = [o for o in opts
              if o.fits_vram and (not allowed or o.quant.upper() in allowed)]
    if viable:
        best = max(viable, key=lambda o: o.bytes)
        r.best_quant = best.quant
        r.est_vram_bytes = best.est_vram_bytes

    if crit.require_gguf and not opts:
        r.reasons.append("no GGUF weights (fleet runtime is llama.cpp)")
    elif crit.require_gguf and not viable:
        # Three distinct failures, and the operator needs to know WHICH: the
        # repo has no quant we accept, or it has one but it won't fit, or the
        # hub reported no file sizes so we can't tell.
        acceptable = [o for o in opts if not allowed or o.quant.upper() in allowed]
        if not acceptable:
            r.reasons.append(
                f"no acceptable quant (have {sorted({o.quant for o in opts})}, "
                f"want {sorted(allowed)})")
        elif all(o.est_vram_bytes is None for o in acceptable):
            r.reasons.append(
                f"cannot size {sorted({o.quant for o in acceptable})} — the hub "
                f"reported no file sizes for this repo")
        else:
            sized = [o for o in acceptable if o.est_vram_bytes is not None]
            small = min(sized, key=lambda o: o.est_vram_bytes)
            r.reasons.append(
                f"nothing fits {budget/1024**3:.1f} GiB at {crit.target_context} ctx "
                f"(smallest acceptable {small.quant} needs "
                f"~{small.est_vram_bytes/1024**3:.1f} GiB)")

    if crit.max_total_bytes and r.total_bytes and r.total_bytes > crit.max_total_bytes:
        r.reasons.append(f"repo is {r.total_bytes/1024**3:.1f} GiB, cap is "
                         f"{crit.max_total_bytes/1024**3:.1f} GiB")
    if r.context_length is not None and r.context_length < crit.min_context:
        r.reasons.append(f"context {r.context_length} < required {crit.min_context}")
    if crit.min_params and r.params and r.params < crit.min_params:
        r.reasons.append(f"{r.params/1e9:.1f}B params < min {crit.min_params/1e9:.1f}B")
    if crit.max_params and r.params and r.params > crit.max_params:
        r.reasons.append(f"{r.params/1e9:.1f}B params > max {crit.max_params/1e9:.1f}B")

    # free disk is a fit attribute too — a pass we can't act on is not a pass
    if r.best_quant:
        need = next((o.bytes for o in opts if o.quant == r.best_quant), 0)
        free = _free_bytes()
        if free is not None and need and need > free:
            r.reasons.append(f"needs {need/1024**3:.1f} GiB, "
                             f"{free/1024**3:.1f} GiB free on the model store")

    # ── capability ────────────────────────────────────────────────────────
    if crit.task and r.task and r.task != crit.task:
        r.reasons.append(f"task {r.task} != {crit.task}")
    if crit.min_downloads and (r.downloads or 0) < crit.min_downloads:
        r.reasons.append(f"{r.downloads or 0} downloads < {crit.min_downloads}")
    if r.trust_tier < crit.min_trust_tier:
        r.reasons.append(f"publisher trust {r.trust_tier} < {crit.min_trust_tier}")
    if crit.max_age_days is not None and r.age_days is not None \
            and r.age_days > crit.max_age_days:
        r.reasons.append(f"last updated {r.age_days}d ago > {crit.max_age_days}d")
    tagset = {t.lower() for t in r.tags}
    missing = [t for t in crit.required_tags if t.lower() not in tagset]
    if missing:
        r.reasons.append(f"missing required tags {missing}")
    banned = [t for t in crit.excluded_tags if t.lower() in tagset]
    if banned:
        r.reasons.append(f"has excluded tags {banned}")
    if r.gated:
        r.notes.append("gated repo — needs an accepted licence on your HF account")

    # k120's card knobs (required_specializations / licenses_allowed), answered
    # from the tags and the repo name only so the screen stays network-free.
    # Both default to "no rule", so a pre-k120 criteria file screens exactly as
    # it did before. See discovery_dossier/screening.py.
    try:
        from hugpy_curation.dossier.screening import extra_reasons
        r.reasons.extend(extra_reasons(hub_id, crit, r.license, r.tags, r.task))
    except Exception as exc:                        # noqa: BLE001 — a missing
        # or broken dossier package must never stop the reviewer screening.
        r.notes.append(f"k120 screen knobs unavailable ({type(exc).__name__})")

    # lineage: a fine-tune of something already in the fleet is the candidate
    # most likely to be a drop-in improvement, so surface it explicitly
    for inc in crit.incumbents:
        needle = inc.split("/")[-1].lower()
        if needle and (needle in (r.base_model or "").lower()
                       or needle in hub_id.lower()):
            r.lineage_of = inc
            r.notes.append(f"lineage: derived from / related to {inc}")
            break

    r.passed = not r.reasons
    r.score = _score(r, crit)
    return r


def _score(r: ScreenResult, crit: ReviewCriteria) -> float:
    """Rank order among passing candidates. Capability first (trust, lineage,
    momentum), then how comfortably it fits — a model that leaves headroom can
    be run at a longer context or alongside something else."""
    s = 0.0
    s += 1.5 * r.trust_tier
    s += 1.0 if r.lineage_of else 0.0
    s += 0.35 * math.log10((r.downloads or 0) + 10)
    s += 0.15 * math.log10((r.likes or 0) + 10)
    if r.context_length:
        s += min(1.0, r.context_length / max(1, crit.target_context)) * 0.5
    if r.est_vram_bytes and crit.usable_vram_bytes:
        headroom = 1 - (r.est_vram_bytes / crit.usable_vram_bytes)
        s += max(0.0, min(1.0, headroom)) * 0.75
    if r.age_days is not None:
        s += 0.5 if r.age_days <= 90 else (0.25 if r.age_days <= 365 else 0.0)
    if r.params:
        s += min(1.0, math.log10(r.params / 1e9 + 1)) * 0.5   # bigger ~ better
    return round(s, 4)


def _free_bytes() -> int | None:
    try:
        from hugpy_platform.constants import DEFAULT_ROOT as root
    except Exception:
        root = os.environ.get("DEFAULT_ROOT") or "/"
    try:
        return shutil.disk_usage(root if os.path.exists(root) else "/").free
    except OSError:
        return None


def search_candidates(crit: ReviewCriteria, api=None) -> list[str]:
    """Candidate repo ids from HF search, most-downloaded first.

    When the criteria requires GGUF (the fleet's llama.cpp runtime), filter the
    search on the `gguf` tag. Without that, a download-ranked search returns
    the first-party safetensors repos — Qwen/Qwen3-8B and friends — and every
    single candidate is rejected for having no GGUF weights, because the
    quantizers who publish GGUF never outrank the originals on downloads.
    """
    api = api or _hf_api()
    filt = crit.library or ("gguf" if crit.require_gguf else None)
    try:
        models = list(api.list_models(
            search=crit.query or None, author=crit.author,
            pipeline_tag=crit.task or None, filter=filt,
            sort="downloads", limit=crit.pool_limit, full=False))
    except Exception:
        return []
    return [m.modelId for m in models if getattr(m, "modelId", None)]
