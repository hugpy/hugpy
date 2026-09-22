"""DRAFT-MODEL-GATE-20260910: recognise GGUFs that are speculative-decoding DRAFT heads.

A draft model (architecture `dflash`, ...) drafts tokens for a BASE model and is
verified by it; it has no standalone forward pass. Loaded on its own it crashes
the engine, and the console must not offer it for chat. The check reads only the
GGUF header (cheap) and is cached; any failure to decide is "not a draft" so a
broken header can never block a real model.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

DRAFT_ARCHS = {"dflash"}
_CACHE: dict = {}          # model_key -> (reason | None, ts)
_TTL_S = 600.0


def gguf_architecture(path: str) -> str | None:
    try:
        from hugpy_engine.spill import _gguf_metadata
        arch = _gguf_metadata(path, (".architecture",)).get(".architecture")
        return str(arch).strip().lower() if arch else None
    except Exception:  # noqa: BLE001
        return None


def _base_model_name(path: str) -> str | None:
    try:
        from hugpy_engine.spill import _gguf_metadata
        v = _gguf_metadata(path, (".base_model.0.name",)).get(".base_model.0.name")
        return str(v).strip() if v else None
    except Exception:  # noqa: BLE001
        return None


def _model_file(model_key: str) -> str | None:
    from hugpy_engine.config.main import get_model_config
    from hugpy_engine.serve.serve import _model_file_for
    cfg = get_model_config(model_key)
    return _model_file_for(model_key, cfg) or None


def draft_model_reason(model_key: str | None) -> str | None:
    """Why `model_key` cannot be served standalone, or None when it can."""
    if not model_key:
        return None
    now = time.time()
    hit = _CACHE.get(model_key)
    if hit and now - hit[1] < _TTL_S:
        return hit[0]
    reason = None
    try:
        path = _model_file(model_key)
        if path and os.path.isfile(path):
            arch = gguf_architecture(path)
            if arch in DRAFT_ARCHS:
                base = _base_model_name(path)
                reason = (
                    f"{model_key} is a speculative-decoding DRAFT model (GGUF architecture "
                    f"'{arch}'{', drafting for ' + base if base else ''}). It has no answers of its own "
                    f"and cannot be loaded standalone (the engine crashes on it) — chat with the base "
                    f"model instead, or attach this file as a draft (--model-draft) to that model."
                )
    except Exception as exc:  # noqa: BLE001 — never block a real model on a probe error
        logger.debug("draft check for %s failed: %s", model_key, exc)
        reason = None
    _CACHE[model_key] = (reason, now)
    return reason
