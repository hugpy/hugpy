"""Re-stamp wrong image tasks from the models' own directories — one shot.

WHY (k61, 2026-07-31). The classifier fix stops NEW wrong stamps; it does not
undo the ones already written. The flux2 incident had the wrong task in all three
task stores — the central discovery report, the worker's discovery report, and
the per-model ``hugpy.json`` sidecar (the sovereign one) — and the keeper
corrected them BY HAND. Hand-corrected data is not a fix: the next walk, the next
box, the next model regress it. This walks the existing rows and re-derives the
task from each dir's own declaration, so the correction is made by CODE and is
repeatable on every box.

    hugpy reclassify-images                 # dry run — prints what WOULD change
    hugpy reclassify-images --apply         # rewrite sidecars + discovery rows
    POST /llm/models/reclassify-images {"apply": true}

SCOPE, deliberately narrow: only rows whose dir carries a diffusers
``model_index.json`` (a pipeline that declares its own class) or that hold ONLY
adapter weights. Everything else is left exactly as it is — this is a corrector
for the two shapes k61 names, not a re-derive of the whole catalog.

IDEMPOTENT: a second run finds nothing to change, because it compares the derived
tasks with what is already stamped and only writes a difference.
"""
from __future__ import annotations

import os
from typing import Optional

from hugpy_platform.constants import MODELS_DISCOVERY_PATH
from hugpy_storage.hugpy_marker import read_hugpy_marker, write_hugpy_marker
from hugpy_engine.model_classifier import classify_model_dir
from abstract_utilities import safe_dump_to_file, safe_load_from_json

__all__ = ["reclassify_images", "reclassify_dir"]


def reclassify_dir(directory: str, *, apply: bool = False,
                   name: Optional[str] = None) -> Optional[dict]:
    """Re-derive one dir's tasks from its contents; None when nothing changes.

    Writes the SIDECAR (hugpy.json) when applying — that marker is the sovereign
    task store, so a discovery row corrected without it is corrected until the
    next walk reads the marker back.
    """
    verdict = classify_model_dir(directory)
    if not verdict:
        return None
    marker = read_hugpy_marker(directory) or {}
    before = marker.get("tasks")
    after = verdict["tasks"]
    if list(before or []) == list(after):
        return None
    change = {
        "name": name or marker.get("name") or os.path.basename(directory.rstrip("/")),
        "dir": directory,
        "from": before,
        "to": after,
        "source": verdict["source"],
        "pipeline_class": verdict.get("pipeline_class"),
        "adapter": bool(verdict.get("adapter")),
        "applied": False,
    }
    if apply:
        extra = {k: v for k, v in marker.items()
                 if k not in ("hub_id", "name", "framework", "tasks",
                              "primary_task", "filename", "include", "source",
                              "stamped_at")}
        if verdict.get("adapter"):
            extra["adapter"] = True
        write_hugpy_marker(
            directory,
            hub_id=marker.get("hub_id"),
            name=marker.get("name"),
            framework=marker.get("framework"),
            tasks=after,
            primary_task=verdict["primary_task"],
            filename=marker.get("filename"),
            include=marker.get("include"),
            source=marker.get("source") or "reclassify",
            **extra,
        )
        change["applied"] = True
    return change


def reclassify_images(*, apply: bool = False,
                      discovery_path: Optional[str] = None) -> dict:
    """Walk the discovery report and re-derive image tasks from disk.

    Returns a report: ``{"applied", "scanned", "changed": [...], "skipped": n}``.
    Dry by default — nothing on disk is touched unless ``apply=True``.
    """
    path = discovery_path or str(MODELS_DISCOVERY_PATH)
    rows = safe_load_from_json(path) if os.path.isfile(path) else None
    rows = rows if isinstance(rows, dict) else {}

    changed, scanned, skipped = [], 0, 0
    for key, row in rows.items():
        directory = (row or {}).get("dir")
        if not directory or not os.path.isdir(directory):
            skipped += 1
            continue
        scanned += 1
        change = reclassify_dir(directory, apply=apply,
                                name=(row or {}).get("name") or key)
        if change is None:
            continue
        change["model_key"] = key
        changed.append(change)
        if apply:
            # The discovery row is the CACHE of the sidecar; correcting one and
            # not the other is how the fleet ended up with three stores that
            # disagreed. Both move together.
            row["tasks"] = change["to"]
            row["primary_task"] = change["to"][0]
            row.pop("needs_classification", None)
            if change["adapter"]:
                row["adapter"] = True

    if apply and changed:
        safe_dump_to_file(data=rows, file_path=path)
    return {"applied": bool(apply), "scanned": scanned, "skipped": skipped,
            "changed": changed, "report_path": path}
