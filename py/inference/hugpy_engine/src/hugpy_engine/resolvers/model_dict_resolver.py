"""Compatibility exports for the canonical model-discovery implementation.

The public legacy module path remains importable; discovery and resolver behavior
lives in ``hugpy_engine.apis.get_module``.
"""

from hugpy_engine.apis.get_module import (
    build_resolver_chain,
    discover_model,
    discover_models,
    enrich,
    resolve_local_config,
    resolve_local_tokenizer,
)

__all__ = [
    "build_resolver_chain",
    "discover_model",
    "discover_models",
    "enrich",
    "resolve_local_config",
    "resolve_local_tokenizer",
]
