"""resolve() must key the whole request off the RESOLVED REGISTRY KEY.

Follow-up to the comfy-sd-turbo x computron mis-route (2026-09-24): a registry
row can carry a ``cfg.model_key`` that differs from its registry key — a comfy
checkpoint row learned/synthesized as ``comfy-<stem>`` may hold the bare
checkpoint stem (``sd-turbo``) in ``model_key``. The DelegatingRunner routes by
``cfg.model_key`` (``_base_model_key``), so after the OUTER placement picked the
worker for ``comfy-sd-turbo`` the delegation re-selected under the bare
``sd-turbo`` — a DIFFERENT, non-comfy row on another worker. resolve() now
reconciles the cfg so its model_key IS the resolved key, and the fix must never
mutate the shared registry object.

Runs standalone and under pytest.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

mr = importlib.import_module("hugpy_engine.resolvers.model_resolver")
mc = importlib.import_module("hugpy_engine.config.models.models_config")
from hugpy_engine.schemas.model_schemas import ModelConfig

REG_KEY = "comfy-sd-turbo"     # the registry key / what placement + the caller use
STEM = "sd-turbo"              # the bare checkpoint stem a comfy row may carry


def _comfy_row(model_key_field: str) -> ModelConfig:
    return ModelConfig(
        name=REG_KEY, hub_id="comfy/sd-turbo", folder="misc/comfy/sd-turbo",
        model_key=model_key_field,               # deliberately the BARE stem
        framework="comfy", tasks=["text-to-image", "image-to-image"],
        primary_task="text-to-image", filename="sd_turbo.safetensors",
    )


def _with_registry_row(key: str, cfg: ModelConfig):
    """Inject a row into MODEL_REGISTRY for the test, restoring after."""
    reg = mc.MODEL_REGISTRY
    had = key in reg
    prev = reg.get(key)
    reg[key] = cfg
    def _restore():
        if had:
            reg[key] = prev
        else:
            reg.pop(key, None)
    return _restore


def test_resolve_reconciles_a_bare_stem_cfg_model_key_to_the_registry_key():
    restore = _with_registry_row(REG_KEY, _comfy_row(STEM))
    try:
        res = mr.resolve({"model_key": REG_KEY, "task": "text-to-image"})
        # The Resolution keys off the registry key both ways.
        assert res.model_key == REG_KEY, res.model_key
        assert res.cfg.model_key == REG_KEY, res.cfg.model_key
        assert res.cfg.framework == "comfy"
        # The DelegatingRunner routes by cfg.model_key — prove it now sees the
        # registry key, not the bare stem that would collide with a non-comfy row.
        runner = res.runner_cls(res.cfg)
        assert runner.model_key == REG_KEY, runner.model_key
        # The shared registry object is NEVER mutated by the reconcile.
        assert mc.MODEL_REGISTRY[REG_KEY].model_key == STEM
    finally:
        restore()


def test_resolve_leaves_an_aligned_row_untouched():
    # A well-formed row (cfg.model_key == registry key) is not rebuilt — the
    # identity is preserved and routing is byte-identical.
    restore = _with_registry_row(REG_KEY, _comfy_row(REG_KEY))
    try:
        res = mr.resolve({"model_key": REG_KEY, "task": "text-to-image"})
        assert res.model_key == REG_KEY
        assert res.cfg.model_key == REG_KEY
        # same object as the registry (no needless copy)
        assert res.cfg is mc.MODEL_REGISTRY[REG_KEY]
    finally:
        restore()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok - {fn.__name__}")
    print(f"\nall {len(fns)} checks passed")
