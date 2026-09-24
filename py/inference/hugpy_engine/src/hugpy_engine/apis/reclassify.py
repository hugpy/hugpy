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

__all__ = ["reclassify_images", "reclassify_dir", "reclassify_vl_gguf", "reclassify_vl_dir"]


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


# ── VISION GGUF (2026-09-23) ────────────────────────────────────────────────
# A GGUF dir holding an mmproj projector is image-text-to-text; the marker
# writer now stamps that at install (hugpy_marker.vl_gguf_tasks — the ONE
# rule). This one-shot re-stamps the rows written before the rule existed so
# the suite registry stops text-grading VL models. Dry by default; idempotent.

def reclassify_vl_dir(directory: str, *, apply: bool = False,
                      name: Optional[str] = None,
                      framework: Optional[str] = None,
                      row: Optional[dict] = None) -> Optional[dict]:
    """``row`` (the catalog/discovery row) supplies the task the catalog
    actually serves when the marker never recorded one — a row the catalog
    already calls text-to-video (an LTX text encoder dir) is not re-labelled."""
    from hugpy_storage.hugpy_marker import vl_gguf_tasks
    marker = read_hugpy_marker(directory) or {}
    if not marker:
        return None
    fw = marker.get("framework") or framework
    before_t = marker.get("tasks") or (row or {}).get("tasks")
    before_p = marker.get("primary_task") or (row or {}).get("primary_task")
    after_t, after_p = vl_gguf_tasks(directory, fw, before_t, before_p)
    if (after_p == before_p) and list(after_t or []) == list(before_t or []):
        return None
    change = {"name": name or marker.get("name") or os.path.basename(directory.rstrip("/")),
              "dir": directory, "from": {"tasks": before_t, "primary_task": before_p},
              "to": {"tasks": after_t, "primary_task": after_p}, "applied": False}
    if apply:
        extra = {k: v for k, v in marker.items()
                 if k not in ("hub_id", "name", "framework", "tasks", "primary_task",
                              "filename", "include", "source", "stamped_at")}
        write_hugpy_marker(directory, hub_id=marker.get("hub_id"), name=marker.get("name"),
                           framework=fw, tasks=after_t, primary_task=after_p,
                           filename=marker.get("filename"), include=marker.get("include"),
                           source=marker.get("source") or "reclassify", **extra)
        change["applied"] = True
        change["updated"] = [f"{os.path.join(directory, 'hugpy.json')}: tasks, primary_task"]
    return change


def reclassify_vl_gguf(*, apply: bool = False,
                       discovery_path: Optional[str] = None,
                       served: Optional[dict] = None) -> dict:
    """Walk the discovery report; re-stamp every GGUF row whose dir holds an
    mmproj projector as image-text-to-text (sidecar AND report row move
    together, like reclassify_images). Dry by default; idempotent.

    ``served`` (``{model_key: {"primary_task", "tasks"}}`` — the live catalog)
    is the task the registry actually serves when neither the marker nor the
    report recorded one (e.g. a classifier-derived ``pipeline-component``)."""
    path = discovery_path or str(MODELS_DISCOVERY_PATH)
    rows = safe_load_from_json(path) if os.path.isfile(path) else None
    rows = rows if isinstance(rows, dict) else {}
    changed, scanned, skipped = [], 0, 0
    for key, row in rows.items():
        directory = (row or {}).get("dir")
        if (row or {}).get("framework") not in ("gguf", "llama_cpp") or not directory \
                or not os.path.isdir(directory):
            skipped += 1
            continue
        scanned += 1
        basis = dict(row or {})
        for fld in ("primary_task", "tasks"):
            if not basis.get(fld) and (served or {}).get(key, {}).get(fld):
                basis[fld] = served[key][fld]
        change = reclassify_vl_dir(directory, apply=apply, name=(row or {}).get("name") or key,
                                   framework=(row or {}).get("framework"), row=basis)
        if change is None:
            continue
        change["model_key"] = key
        changed.append(change)
        if apply:
            row["tasks"] = change["to"]["tasks"]
            row["primary_task"] = change["to"]["primary_task"]
            row.pop("needs_classification", None)
            change.setdefault("updated", []).append(
                f"{path} [{key}]: tasks, primary_task")
    db = None
    if apply and changed:
        safe_dump_to_file(data=rows, file_path=path)
        db = _sync_registry_db(rows, path)
        for change in changed:
            change.setdefault("updated", [])
            if db.get("saved"):
                change["updated"].append(f"registry DB discovery_models [{change['model_key']}]: "
                                         f"row.tasks, row.primary_task")
            else:
                change["not_updated"] = f"registry DB discovery_models: {db.get('why')}"
    return {"applied": bool(apply), "scanned": scanned, "skipped": skipped,
            "changed": changed, "report_path": path, "registry_db": db}


def _sync_registry_db(rows: dict, path: str) -> dict:
    """Write the corrected report into the registry DB (``discovery_models``).

    The engine's registry reads the DB FIRST (``models_config._load_discovery_
    report``) when HUGPY_REGISTRY_DB=pg, so a correction written only to the JSON
    report is invisible to every registry rebuild until the next full discovery
    walk re-saves the DB. Same full-report save the walk performs
    (``model_index.save_discovery``) — and only for the default report, since a
    ``--report`` pointing elsewhere is not the set the DB mirrors."""
    if os.path.abspath(path) != os.path.abspath(str(MODELS_DISCOVERY_PATH)):
        return {"saved": False, "why": f"report {path} is not the registry's report "
                                      f"{MODELS_DISCOVERY_PATH}; DB left as is"}
    try:
        from hugpy_engine.model_index import enabled, last_db_error, save_discovery
    except Exception as exc:  # noqa: BLE001
        return {"saved": False, "why": f"model_index unavailable ({exc})"}
    if not enabled():
        return {"saved": False, "why": "HUGPY_REGISTRY_DB is not pg in this process (JSON report is the registry source)"}
    if save_discovery(rows):
        return {"saved": True, "rows": len(rows)}
    err = last_db_error() or {}
    return {"saved": False, "why": f"save_discovery failed: {err.get('error') or 'no error recorded'}"}
