"""Adapt a routing or manifest mapping for model completeness checks."""

from __future__ import annotations

from types import SimpleNamespace


def model_config_shim(model: dict) -> SimpleNamespace:
    """Expose the fields read by ``model_looks_downloaded``."""
    return SimpleNamespace(
        framework=model.get("framework"),
        filename=model.get("filename"),
        include=model.get("include"),
        primary_task=model.get("primary_task") or model.get("task"),
        tasks=model.get("tasks"),
    )
