"""Console-side presence helpers — the live ``model_status`` read and the
install-marker writer. Flask-free: these are storage facts the HTTP layer
merely shapes into responses."""
import os
from datetime import datetime, timezone
from typing import Any

from hugpy_platform.constants import HUGPY_MARKER
from hugpy_storage.model_paths import route_destination
from hugpy_storage.model_presence import model_looks_downloaded, pinned_filename_present
def model_status(model: dict) -> dict:
    # route_destination now RESOLVES through every historical layout, so
    # `destination` is the real on-disk dir when one exists (flat or a legacy
    # task path), not a task reconstruction that pointed at an empty twin.
    destination = route_destination(model)              # was model_destination(...)
    marker = os.path.join(destination, HUGPY_MARKER)    # was install_marker(...)

    # PRESENCE HONESTY (operator-locked 2026-07-11): a dir with COMPLETE weights
    # is installed even if it has no hugpy.json marker yet (the AEON case: three
    # good quants on disk, no marker -> used to read "partial"). GGUF counts as
    # installed when ANY complete quant is present (model_looks_downloaded's
    # any-quant semantics); the config.json + mmproj/vision gates are unchanged.
    complete = False
    try:
        complete = bool(model_looks_downloaded(destination, _status_cfg(model)))
    except Exception:  # noqa: BLE001 — never break the feed over a presence probe
        complete = False

    if complete or os.path.exists(marker):
        status = "installed"
    elif os.path.exists(destination) and os.listdir(destination):
        status = "partial"
    else:
        status = "not_installed"

    out = {"status": status, "destination": destination, "installed_marker": marker}

    # A pinned `filename` that isn't among the installed quants is a WARNING
    # surfaced on the /models feed — NOT a "partial" status (the model still
    # serves off another complete quant). None when there's nothing to warn about.
    try:
        if status == "installed" and pinned_filename_present(
                destination, _status_cfg(model)) is False:
            out["filename_warning"] = (
                f"pinned filename {model.get('filename')!r} is not among the "
                f"installed quants; serving off another quant")
    except Exception:  # noqa: BLE001
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────
# model_status is EXPENSIVE: route_destination globs four runtime families'
# legacy task dirs and stats every candidate, then model_looks_downloaded globs
# the winner — ~10^2 filesystem calls per model, every one a virtiofs round-trip
# to the host on central. A listing route loops ~107 models, so re-deriving it
# per request cost ~10^4 of them and collapsed under concurrency (18.5s → 55.3s
# and degrading, 2026-07-27).
#
# It is no longer re-derived per request. Installation status changes on
# download / delete / prune / reconcile / discovery, so it is DERIVED AT THOSE
# MOMENTS and PERSISTED beside the registry (comms/model_physical.py); the
# listings read a dict. ``model_status`` here stays THE live read — the single
# implementation both the write points and the derive-on-miss fallback call.
#
# There is deliberately no ``cached_model_status``/``refresh_model_status``
# wrapper any more: a memo in front of this call would be a SECOND mechanism
# answering the same question as the persisted record, and two answers to one
# question is exactly how "preview vs auto-evict propose different victims"
# happened here. See flask_app/.../downloads/model_physical.py for the read path.
# ──────────────────────────────────────────────────────────────────────────


from hugpy_storage.model_config_shim import model_config_shim as _status_cfg

def write_install_marker(destination: str, model_key: str, model: dict[str, Any]) -> None:
    # The marker IS the authoritative hugpy.json declared-identity read back by
    # discovery (resolve_hugpy_marker / get_module) to reconstruct capability.
    # Discovery keys on the FULL `tasks` list + `primary_task`, so stamp both —
    # a singular-only marker silently drops secondary tasks (e.g. image-to-image)
    # on re-discovery. Mirror the live path (write_hugpy_marker / _stamp).
    primary = model.get("primary_task") or model.get("task")
    tasks = model.get("tasks") or ([primary] if primary else None)
    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]
    payload = {
        "model_key": model_key,
        "hub_id": model.get("hub_id"),
        "framework": model.get("framework"),
        "task": primary,          # singular kept for back-compat; = primary_task
        "tasks": tasks,           # full capability list — what discovery reads
        "primary_task": primary or (tasks[0] if tasks else None),
        "filename": model.get("filename"),
        "include": model.get("include"),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    # INSTALL MANIFEST (2026-09-23): an install record already captured is
    # kept; otherwise capture it now from the files on disk (HF download
    # metadata supplies sha256/revision when present — no Hub call).
    from hugpy_storage.hugpy_marker import (MANIFEST_KEY, _save_marker,
                                            build_install_manifest, read_hugpy_marker)
    prior = (read_hugpy_marker(destination) or {}).get(MANIFEST_KEY)
    try:
        payload[MANIFEST_KEY] = (prior if isinstance(prior, dict)
                                 else build_install_manifest(destination, source="huggingface"))
    except Exception:  # noqa: BLE001 — the marker still lands without it
        pass

    os.makedirs(destination, exist_ok=True)
    _save_marker(destination, payload)          # atomic (temp + os.replace)



