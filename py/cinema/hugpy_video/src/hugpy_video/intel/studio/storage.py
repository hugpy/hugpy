"""Atomic, os.path-only persistence (INV-6). A killed process never leaves a
half-written file that a downstream stage mistakes for done: write to a temp
sibling, fsync, then os.replace onto the final path (atomic on POSIX)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from hugpy_video.intel.studio.schemas import RenderManifest


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".tmp-{os.path.basename(path)}-{os.getpid()}")
    with open(tmp, "w", encoding=encoding) as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)  # atomic; never a partial final file


def persist_manifest(manifest: RenderManifest, out_dir: str) -> str:
    """Write a manifest to <out_dir>/<content_hash>.json atomically and return
    the path. Content-addressed: identical intents collapse to one file (INV-6)."""
    payload = {
        "render_id": manifest.render_id,
        "content_hash": manifest.content_hash(),
        "manifest": asdict(manifest),
    }
    path = os.path.join(out_dir, f"{manifest.content_hash()}.json")
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))
    return path


def load_manifest_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
