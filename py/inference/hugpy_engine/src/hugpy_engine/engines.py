"""Inference-engine availability and GGUF runner lifecycle."""

from __future__ import annotations

from .backend import get_backend


def native_engine_status(*, probe: bool = True) -> dict:
    return get_backend().native_engine_status(probe=probe)


def native_engine_binaries() -> dict:
    return get_backend().native_engine_binaries()


def install_native_engine(**options) -> dict:
    return get_backend().install_native_engine(**options)


def build_native_engine(**options) -> dict:
    return get_backend().build_native_engine(**options)


def llama_runner(model_key: str):
    return get_backend().llama_runner(model_key)


def loaded_llama_runners() -> dict:
    return get_backend().loaded_llama_runners()


def evict_llama_runner(model_key: str) -> bool:
    return get_backend().evict_llama_runner(model_key)


__all__ = [
    "build_native_engine",
    "evict_llama_runner",
    "install_native_engine",
    "llama_runner",
    "loaded_llama_runners",
    "native_engine_binaries",
    "native_engine_status",
]
