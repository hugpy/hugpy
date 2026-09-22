"""PEFT adapter dirs: detect one, name its base, resolve the pair — or refuse honestly.

WHY THIS EXISTS (fleet bench, 2026-07-29)
    Four registry rows answered every single serve attempt with

        ValueError: Unrecognized model in <dir>.
                    Should have a `model_type` key in its config.json.

    Three of them (``qwen3.5-test-stage1-lora``,
    ``Qwen2.5-1.5B-LFGRPO-300S``, ``veeraragavan410~Llama-3.2-3B-sentiment``)
    are bare LoRA ADAPTER directories: an ``adapter_config.json`` and an
    ``adapter_model.safetensors``, and NO ``config.json``. An adapter has no
    ``model_type`` because it is not a standalone model — it is a delta that
    patches a BASE model named in ``adapter_config.json``:
    ``base_model_name_or_path``.

    Two independent defects put those rows in that state:

      1. DISCOVERY read only ``config.json``, so ``base_model`` / ``peft_type``
         came back None for every adapter on the fleet. The registry's
         "adapter needs its base on disk" gate keys on ``base_model``, so with
         the field empty the gate never fired and the row was minted as an
         ordinary transformers text-generation model.
      2. The LOAD path had adapter support wired at the far end
         (``DeepCoder._load_model`` reads ``cfg.adapter_dir`` and applies
         ``PeftModel.from_pretrained``) but ``DeepCoderConfig`` had no such
         field and nothing ever set it — dead code behind a config shape that
         could not express it. The adapter dir went in as ``model_dir`` and
         transformers detonated on it.

    Operator doctrine (2026-07-29): every model must be AVAILABLE and CALLABLE;
    silent unavailability is the defect class, and inefficient-but-working beats
    hidden. So there are exactly two honest outcomes for an adapter row, and
    "Unrecognized model" is neither of them:

      SERVE   — base is on disk: load the base, apply the adapter on top.
      REFUSE  — base is NOT on disk: say so, NAME the base to pull. The row
                stays visible, flagged unserveable with that reason.

    Never a silent multi-GB download: the base is resolved through hugpy's own
    store only (``route_destination``). Acquiring an absent base is an operator
    act, and the refusal text is what tells them which id to acquire.

THE FOURTH ROW is a different shape entirely and deliberately not handled here:
    ``Viral2AI~chatterbox`` has neither ``config.json`` NOR
    ``adapter_config.json``. It is a bespoke TTS repo (loose ``t3_*``/``s3gen``
    ``.pt``/``.safetensors`` checkpoints) mis-registered as
    transformers/text-generation. ``is_adapter_dir`` returns False for it, and
    ``standalone_load_refusal`` is what gives it an honest reason instead of the
    same opaque transformers crash.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

from hugpy_storage.model_paths import DEFAULT_ROOT, route_destination

__all__ = [
    "ADAPTER_CONFIG_NAME", "MODEL_CONFIG_NAME",
    "AdapterBaseUnavailable",
    "read_adapter_config", "read_model_config", "has_standalone_config",
    "is_adapter_dir", "adapter_base_id", "adapter_peft_type",
    "find_base_model_dir", "base_model_present",
    "adapter_unserveable_reason", "standalone_load_refusal",
    "resolve_adapter_pair", "adapter_metadata_fields",
]

ADAPTER_CONFIG_NAME = "adapter_config.json"
MODEL_CONFIG_NAME = "config.json"

# What "the base model is actually on disk here" means. A dir holding only a
# tokenizer + a README is not a base model, and pointing a load at it just moves
# the crash one directory over.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


class AdapterBaseUnavailable(RuntimeError):
    """A dir IS a PEFT adapter, but its base model can't be resolved locally.

    Carries the pieces so a caller can re-word without re-parsing: ``.adapter_dir``,
    ``.base_model`` (the id named by adapter_config.json, or None if it named none).
    """

    def __init__(self, message: str, *, adapter_dir: str,
                 base_model: Optional[str] = None):
        super().__init__(message)
        self.adapter_dir = adapter_dir
        self.base_model = base_model


# ---------------------------------------------------------------------------
# Reading the two config shapes
# ---------------------------------------------------------------------------
def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 — absent/unreadable/malformed are all "no"
        return None
    return data if isinstance(data, dict) else None


def read_adapter_config(model_dir: Optional[str]) -> Optional[dict]:
    """``adapter_config.json`` parsed, or None when absent/unreadable."""
    if not model_dir:
        return None
    return _read_json(os.path.join(model_dir, ADAPTER_CONFIG_NAME))


def read_model_config(model_dir: Optional[str]) -> Optional[dict]:
    """``config.json`` parsed, or None when absent/unreadable."""
    if not model_dir:
        return None
    return _read_json(os.path.join(model_dir, MODEL_CONFIG_NAME))


def has_standalone_config(model_dir: Optional[str]) -> bool:
    """True iff ``config.json`` names a ``model_type`` — i.e. transformers can
    load this dir on its own. This is precisely the condition whose absence
    produces "Should have a `model_type` key in its config.json"."""
    cfg = read_model_config(model_dir) or {}
    mt = cfg.get("model_type")
    return isinstance(mt, str) and bool(mt.strip())


def is_adapter_dir(model_dir: Optional[str]) -> bool:
    """True iff this dir is a PEFT adapter that CANNOT stand alone.

    A merged checkpoint that ships both files (some trainers leave the adapter
    config next to fully-merged weights) is NOT an adapter for load purposes —
    ``config.json`` names a model_type, so transformers loads it directly and
    the base/adapter dance would be wrong."""
    return read_adapter_config(model_dir) is not None and not has_standalone_config(model_dir)


def adapter_base_id(model_dir: Optional[str], *,
                    adapter_cfg: Optional[dict] = None) -> Optional[str]:
    """The base model id an adapter declares, or None if it declares none."""
    cfg = adapter_cfg if adapter_cfg is not None else read_adapter_config(model_dir)
    base = (cfg or {}).get("base_model_name_or_path")
    base = base.strip() if isinstance(base, str) else ""
    return base or None


def adapter_peft_type(model_dir: Optional[str], *,
                      adapter_cfg: Optional[dict] = None) -> Optional[str]:
    """``peft_type`` ("LORA", ...) as declared, or None."""
    cfg = adapter_cfg if adapter_cfg is not None else read_adapter_config(model_dir)
    pt = (cfg or {}).get("peft_type")
    pt = pt.strip() if isinstance(pt, str) else ""
    return pt or None


def adapter_metadata_fields(model_dir: Optional[str]) -> dict:
    """The metadata a bare adapter dir CAN state about itself — {} if not one.

    Shaped to slot straight into the enrichment resolvers' return dict
    (``peft_type`` / ``base_model``, plus the base's architecture class when the
    adapter records one under ``auto_mapping``, which unsloth exports do). This
    is the only thing that makes ``base_model`` non-None for an adapter, and
    every downstream adapter decision keys on that field.
    """
    cfg = read_adapter_config(model_dir)
    if cfg is None or has_standalone_config(model_dir):
        return {}
    auto_map = cfg.get("auto_mapping") if isinstance(cfg.get("auto_mapping"), dict) else {}
    base_cls = auto_map.get("base_model_class")
    out = {
        "peft_type": adapter_peft_type(model_dir, adapter_cfg=cfg),
        "base_model": adapter_base_id(model_dir, adapter_cfg=cfg),
    }
    if isinstance(base_cls, str) and base_cls.strip():
        out["architectures"] = [base_cls.strip()]
    return out


# ---------------------------------------------------------------------------
# Locating the base — LOCAL STORE ONLY
# ---------------------------------------------------------------------------
def find_base_model_dir(base_model: Optional[str],
                        root: str = DEFAULT_ROOT) -> Optional[str]:
    """Where the base model lives in THIS store, or None if it isn't here.

    Goes through ``route_destination`` so every historical layout resolves the
    same way the rest of hugpy resolves paths, then insists on real weight files
    — a tokenizer-only shell is not a base model. Never downloads, never
    contacts the hub: an absent base is a REFUSAL, not an acquisition trigger.
    """
    if not base_model:
        return None
    try:
        base_dir = route_destination(
            {"hub_id": base_model, "framework": "transformers",
             "primary_task": "text-generation"},
            root,
        )
    except Exception:  # noqa: BLE001 — an unroutable id is simply not present
        return None
    if not base_dir or not os.path.isdir(base_dir):
        return None
    try:
        names = os.listdir(base_dir)
    except OSError:
        # EMFILE / permission: "can't tell" must not be reported as "absent"
        # (see store-presence-scan-emfile-false-negative). Raise, don't lie.
        raise
    if not any(n.endswith(_WEIGHT_SUFFIXES) for n in names):
        return None
    return base_dir


def base_model_present(base_model: Optional[str],
                       root: str = DEFAULT_ROOT) -> bool:
    """True when ``base_model`` is empty (nothing required) or on disk here."""
    if not base_model:
        return True   # not an adapter; nothing to require
    return find_base_model_dir(base_model, root) is not None


# ---------------------------------------------------------------------------
# The two honest outcomes
# ---------------------------------------------------------------------------
def _fix_hint(base_model: str) -> str:
    return (f"acquire the base model {base_model!r} into the store "
            f"(POST /llm/models/download with hub_id={base_model!r}), "
            f"or serve a merged checkpoint instead")


def adapter_unserveable_reason(model_dir: Optional[str], *,
                               base_model: Optional[str] = None,
                               root: str = DEFAULT_ROOT) -> Optional[str]:
    """The refusal text for an adapter that cannot be served, else None.

    None means "nothing to refuse" — either it isn't an adapter, or it is one
    and its base is right here. The string always names the CAUSE and the FIX.
    """
    if not is_adapter_dir(model_dir):
        return None
    base = base_model or adapter_base_id(model_dir)
    if not base:
        return ("PEFT adapter with no base model declared: "
                f"{os.path.join(str(model_dir), ADAPTER_CONFIG_NAME)} has no "
                "`base_model_name_or_path`, and an adapter is a delta — it "
                "cannot be loaded standalone. FIX: re-export the adapter with "
                "its base recorded, or merge it into its base and register the "
                "merged checkpoint.")
    if find_base_model_dir(base, root) is None:
        return (f"PEFT adapter (base {base!r}) — the adapter is on disk but its "
                f"base model is NOT in this store, and an adapter cannot be "
                f"loaded without it. FIX: {_fix_hint(base)}.")
    return None


def standalone_load_refusal(model_dir: Optional[str]) -> Optional[str]:
    """Refusal text for a dir transformers cannot load AT ALL, else None.

    Catches the non-adapter sibling of the same bench failure: no
    ``config.json`` with a ``model_type`` and no ``adapter_config.json`` either
    (``Viral2AI~chatterbox``). Turns the opaque "Unrecognized model" into a
    statement of what's missing.
    """
    if not model_dir or has_standalone_config(model_dir) or is_adapter_dir(model_dir):
        return None
    if read_model_config(model_dir) is not None:
        return (f"{model_dir} has a config.json with no `model_type`, so "
                "transformers cannot identify the architecture. FIX: this is "
                "not a transformers model — register it under the framework "
                "that owns its weights, or add the upstream config.json.")
    return (f"{model_dir} is not a loadable transformers model: no config.json "
            "(so no `model_type`) and no adapter_config.json either. FIX: the "
            "dir holds loose checkpoints for a bespoke runtime — re-register it "
            "under the framework/task that can serve it, or acquire the real "
            "transformers repo.")


def resolve_adapter_pair(model_dir: str, *,
                         base_model: Optional[str] = None,
                         root: str = DEFAULT_ROOT) -> Tuple[str, Optional[str]]:
    """``(dir_to_load, adapter_dir_or_None)`` for any model dir.

    Ordinary model  -> ``(model_dir, None)``            — path unchanged.
    Adapter + base  -> ``(base_dir, model_dir)``        — load base, apply delta.
    Adapter, no base-> raises ``AdapterBaseUnavailable`` naming base and fix.
    """
    if not is_adapter_dir(model_dir):
        return model_dir, None
    base = base_model or adapter_base_id(model_dir)
    reason = adapter_unserveable_reason(model_dir, base_model=base, root=root)
    if reason:
        raise AdapterBaseUnavailable(reason, adapter_dir=model_dir, base_model=base)
    return find_base_model_dir(base, root), model_dir
