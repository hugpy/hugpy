from __future__ import annotations

import json
import os
import re
from typing import Any


def load_manifest(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a JSON object.")

    return data


def load_manifest_or_empty(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    return load_manifest(path)


def save_manifest(path: str, manifest: dict[str, dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def key_for_hub_id(hub_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", hub_id.strip())
    return slug.strip("_") or hub_id


def upsert_model(
    path: str,
    model: dict[str, Any],
    *,
    key: str | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Insert or update a manifest entry. Returns (key, full_manifest)."""
    manifest = load_manifest_or_empty(path)
    chosen_key = key or key_for_hub_id(model.get("hub_id") or model.get("name", ""))
    existing = manifest.get(chosen_key, {})
    manifest[chosen_key] = {**existing, **model}
    save_manifest(path, manifest)
    return chosen_key, manifest
