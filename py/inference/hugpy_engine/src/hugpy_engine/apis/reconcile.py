#!/usr/bin/env python3
"""Store reconcile — the store-flattening MIGRATION (operator-locked 2026-07-11).

The storage layout flattened to ``models/<runtime>/<owner>/<repo>``: the
``primary_task`` path segment DIED (it baked derived, mutable, PLURAL metadata
into an immutable path -> task-twin dirs, sticky wrong-task discovery, empty
re-routed dirs with weights stranded in legacy ``misc/``, "redownload ready
models" complaints). This module is the one-shot, RESUMABLE, IDEMPOTENT
migration that walks every catalog entry, finds where its files actually sit
across EVERY historical layout, picks the COMPLETE copy, and renames it into the
flat path — merging complements (a mmproj sidecar in a twin), ARCHIVING losers
and ``.part``-only orphans (NEVER deleting), and updating the registry entry +
its hugpy.json marker (metadata's new home).

MONITOR-FIRST: ``reconcile_store(apply=False)`` (the default) reports the full
plan and touches NOTHING. The operator/keeper flips ``apply=True`` to execute.
Re-running converges (no ping-pong): once a repo is a single flat COMPLETE dir,
its plan is a no-op.

Discipline: errors as data (a per-entry ``error`` field, never a raise that
aborts the whole run); never delete (archive under ``models/_archive/<ts>/``);
same-filesystem rename is instant, cross-device falls back to copy+verify+
archive-source; heavy imports are lazy.
"""
from __future__ import annotations

import os
import glob
import json
import time
import shutil
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy accessors — keep this module import-cheap and free of import cycles.
# ---------------------------------------------------------------------------
def _paths():
    from hugpy_storage import model_paths as p
    return p


def _looks_downloaded(directory, cfg):
    from hugpy_engine.config.main import model_looks_downloaded
    return bool(model_looks_downloaded(directory, cfg))


def _write_marker(directory, **kw):
    from hugpy_storage.hugpy_marker import write_hugpy_marker
    return write_hugpy_marker(directory, **kw)


def _constants():
    from hugpy_platform import constants as c
    return c


# ---------------------------------------------------------------------------
# Classification of one on-disk dir.
# ---------------------------------------------------------------------------
_GGUF_EXTS = (".gguf", ".GGUF")
_WEIGHT_EXTS = (".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def _iter_files(directory):
    for r, _d, files in os.walk(directory):
        # skip HF bookkeeping so a bare .cache/ doesn't read as content
        base = os.path.basename(r)
        if base in (".cache", ".locks", "blobs", "refs"):
            continue
        for f in files:
            yield os.path.join(r, f)


def _dir_weight_bytes(directory):
    """Sum of REAL weight bytes (ignores .part staging + bookkeeping)."""
    total = 0
    for f in _iter_files(directory):
        low = f.lower()
        if low.endswith(".part") or low.endswith(".state.json"):
            continue
        if low.endswith(_WEIGHT_EXTS):
            try:
                total += os.path.getsize(f)
            except OSError:
                pass
    return total


def _has_only_part_weights(directory):
    """True when the dir holds .part staging files but NO finished weight."""
    saw_part = saw_weight = False
    for f in _iter_files(directory):
        low = f.lower()
        if low.endswith(".part"):
            saw_part = True
        elif low.endswith(_WEIGHT_EXTS):
            saw_weight = True
    return saw_part and not saw_weight


def _is_effectively_empty(directory):
    """No real files at all (bookkeeping-only or truly empty)."""
    for f in _iter_files(directory):
        low = f.lower()
        if low.endswith(".part"):
            continue
        # any real payload file (weights, config, tokenizer, marker) counts
        return False
    return True


def _read_marker(directory):
    try:
        from hugpy_storage.hugpy_marker import read_hugpy_marker
        return read_hugpy_marker(directory) or {}
    except Exception:  # noqa: BLE001
        return {}


def _effective_cfg(directory, base_cfg):
    """The cfg to judge THIS dir by — task derived from the dir's OWN hugpy.json
    marker (the authoritative, de-tasked home) rather than the catalog entry's
    task, which for an un-migrated dir is often a LAYOUT-PATH GUESS. This is what
    stops a text gguf mis-filed under image-text-to-text (marker tasks=null) from
    tripping the vision/mmproj gate and reading 'incomplete'. No marker -> fall
    back to the entry cfg unchanged."""
    m = _read_marker(directory)
    if not m:
        return base_cfg
    from types import SimpleNamespace

    def _base(attr):
        return base_cfg.get(attr) if isinstance(base_cfg, dict) else getattr(base_cfg, attr, None)

    return SimpleNamespace(
        framework=_base("framework") or m.get("framework"),
        filename=_base("filename") or m.get("filename"),
        include=_base("include") or m.get("include"),
        primary_task=m.get("primary_task"),   # marker is authoritative (null ok)
        tasks=m.get("tasks"),
    )


def classify_dir(directory, cfg):
    """One dir's disposition (judged by the dir's OWN marker task, not the
    entry's possibly path-guessed one):
        complete   — model_looks_downloaded passes (usable copy)
        part-only  — .part staging present, no finished weight (crashed pull)
        empty      — bookkeeping-only / nothing real
        incomplete — real files present but not a usable copy
    """
    try:
        if _looks_downloaded(directory, _effective_cfg(directory, cfg)):
            return "complete"
    except Exception as exc:  # noqa: BLE001 — classification is data, never fatal
        logger.debug("classify %s: model_looks_downloaded raised %s", directory, exc)
    if _has_only_part_weights(directory):
        return "part-only"
    if _is_effectively_empty(directory):
        return "empty"
    return "incomplete"


# ---------------------------------------------------------------------------
# Move / merge / archive primitives.
# ---------------------------------------------------------------------------
def _same_device(a, b):
    """Do path *a* and the (existing) parent of *b* live on one filesystem?
    Same device => os.rename is atomic + instant; else we copy+verify."""
    try:
        pa = a
        while pa and not os.path.exists(pa):
            pa = os.path.dirname(pa)
        pb = b
        while pb and not os.path.exists(pb):
            pb = os.path.dirname(pb)
        return os.stat(pa).st_dev == os.stat(pb).st_dev
    except OSError:
        return False


def _archive_dest(root, ts, src):
    """Where a loser/orphan is ARCHIVED (never deleted): mirrors its path under
    models/_archive/<ts>/ so the original location is recoverable."""
    models = os.path.join(root, "models")
    try:
        rel = os.path.relpath(src, models)
    except ValueError:
        rel = os.path.basename(src.rstrip("/"))
    if rel.startswith(".."):
        rel = os.path.basename(src.rstrip("/"))
    return os.path.join(models, "_archive", ts, rel)


def _rename_or_copy(src, dst, apply, verify=True):
    """Move src->dst. Same device: os.rename (instant). Cross device: copytree,
    verify weight bytes match, then the CALLER archives the source (we never
    delete). Returns (op, detail). Dry-run returns the planned op only."""
    same = _same_device(src, dst)
    op = "rename" if same else "copy"
    if not apply:
        return op, {"same_device": same}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if same:
        os.rename(src, dst)
        return op, {"same_device": True}
    # cross-device: copy then verify (source archived by caller, not deleted)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    if verify and _dir_weight_bytes(src) != _dir_weight_bytes(dst):
        raise RuntimeError(f"cross-device copy verify failed: {src} -> {dst}")
    return op, {"same_device": False, "copied": True}


def _move_entry_into(src_dir, dst_dir, name, apply):
    """Move ONE top-level entry (file or subdir) src_dir/name -> dst_dir/name.
    Used by merge to pull a complement (e.g. an mmproj sidecar) into the winner.
    NEVER overwrites: skips if the target basename already exists."""
    src = os.path.join(src_dir, name)
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        return False
    if not apply:
        return True
    os.makedirs(dst_dir, exist_ok=True)
    if _same_device(src, dst):
        os.rename(src, dst)
    elif os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True


def _complement_names(winner_dir, other_dir):
    """Top-level entries present in other_dir but ABSENT (by basename) in
    winner_dir — the files the winner LACKS, candidates for a merge. Skips
    bookkeeping. Weight ordering keeps big sidecars deterministic."""
    try:
        have = set(os.listdir(winner_dir))
    except OSError:
        have = set()
    out = []
    try:
        for name in sorted(os.listdir(other_dir)):
            if name in have or name in (".cache", ".locks", "blobs", "refs"):
                continue
            low = name.lower()
            # never pull STAGING artifacts into a winner — a .part/.state/
            # .chunksums is a crashed-pull remnant, not a complement.
            if low.endswith((".part", ".state.json")) or ".chunksums" in low:
                continue
            out.append(name)
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# Catalog loading.
# ---------------------------------------------------------------------------
def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _catalog_entries(root):
    """Every catalog entry across BOTH persisted artifacts — the discovery
    report (the /models registry source) and the download manifest — keyed by
    registry key. Each carries hub_id/framework/filename/include/tasks/
    primary_task and (discovery) dir/folder."""
    c = _constants()
    entries = {}
    for src in (c.MODELS_DISCOVERY_PATH, c.MODELS_DICT_PATH):
        for key, row in _load_json(src).items():
            if not isinstance(row, dict):
                continue
            merged = dict(entries.get(key, {}))
            for k, v in row.items():
                if v is not None and merged.get(k) is None:
                    merged[k] = v
            merged.setdefault("_key", key)
            entries[key] = merged
    return entries


def _group_key(entry, root):
    """Repo identity used to dedupe entries that name the same physical repo
    (a task-twin can mint two registry keys): (runtime, owner/repo)."""
    p = _paths()
    hub_path = p._hub_path_of(entry)
    runtime = p.runtime_folder(entry.get("framework") or "", hub_path,
                               include=entry.get("include"),
                               filename=entry.get("filename"))
    return (runtime, hub_path)


# ---------------------------------------------------------------------------
# Presence-honesty helper (shared with the /models feed).
# ---------------------------------------------------------------------------
from hugpy_storage.model_presence import pinned_filename_present  # noqa: E402  (storage owns presence)


# ---------------------------------------------------------------------------
# The plan for ONE repo group.
# ---------------------------------------------------------------------------
def _plan_group(gkey, members, root):
    """Survey every dir holding this repo, choose the winner, and build the
    ordered action list. Returns a plan dict (no side effects)."""
    p = _paths()
    runtime, hub_path = gkey
    # richest member for cfg fields
    rep = max(members, key=lambda m: sum(1 for v in m.values() if v))
    cfg = p._routing_as_cfg(rep)
    flat_dest = os.path.join(root, "models", runtime, hub_path)

    # Survey: union of candidate dirs from every member, existing only.
    seen, cands = set(), []
    for m in members:
        for d in p.candidate_model_dirs(m, root):
            if d not in seen and os.path.isdir(d):
                seen.add(d)
                cands.append(d)

    plan = {
        "hub_id": rep.get("hub_id"),
        "runtime": runtime,
        "keys": sorted({m.get("_key") for m in members if m.get("_key")}),
        "flat_dest": flat_dest,
        "flat_folder": os.path.relpath(flat_dest, os.path.join(root, "models")),
        "dirs": [{"path": d, "class": classify_dir(d, cfg)} for d in cands],
        "actions": [],
        "warnings": [],
        "status": "noop",
        "winner": None,
    }

    if not cands:
        plan["status"] = "absent"          # cataloged but nothing on disk
        return plan

    classes = {d["path"]: d["class"] for d in plan["dirs"]}
    complete = [d for d in cands if classes[d] == "complete"]

    # ---- winner selection -------------------------------------------------
    # 1) a complete copy wins (candidate order already prefers flat, then the
    #    model's own task). 2) else the dir with the most real weight bytes,
    #    which a merge may COMPLETE (quant here + mmproj in a twin).
    if complete:
        winner = flat_dest if flat_dest in complete else complete[0]
    else:
        winner = max(cands, key=_dir_weight_bytes) if any(
            _dir_weight_bytes(d) for d in cands) else None
    plan["winner"] = winner

    # ---- merge complements INTO the winner -------------------------------
    # Files the winner lacks that a loser/twin holds (e.g. the mmproj sidecar):
    # move them in (never overwrite). Done BEFORE the move so the winner is whole.
    merged_from = []
    if winner is not None:
        for d in cands:
            if d == winner:
                continue
            names = _complement_names(winner, d)
            for name in names:
                plan["actions"].append(
                    {"op": "merge", "file": name, "from": d, "into": winner})
            if names:
                merged_from.append(d)

    # ---- move winner into the flat path ----------------------------------
    if winner is not None and os.path.normpath(winner) != os.path.normpath(flat_dest):
        same = _same_device(winner, flat_dest)
        plan["actions"].append({
            "op": "move", "src": winner, "dst": flat_dest,
            "mechanism": "rename" if same else "copy+verify+archive-source",
        })
        plan["status"] = "move"
    elif plan["actions"]:
        plan["status"] = "merge-only"

    effective_winner = flat_dest if winner is not None else None

    # ---- archive losers + orphans (NEVER delete) -------------------------
    # With a winner, every OTHER copy is a loser -> archive (its unique files
    # were already merged in). WITHOUT a winner, only clear TRUE orphans
    # (.part-only crashed pulls, empties); an `incomplete` dir with real
    # content is LEFT IN PLACE for the operator to resume/reclassify — never
    # archive the only copies of a model out from under it.
    ts = plan.get("_ts", "PENDING")
    for d in cands:
        if d == winner:
            continue
        cls = classes[d]
        if winner is None and cls == "incomplete":
            plan["warnings"].append(f"incomplete copy left in place (no winner): {d}")
            continue
        reason = {"complete": "loser-duplicate", "part-only": "part-orphan",
                  "empty": "empty-twin", "incomplete": "incomplete-twin"}[cls]
        plan["actions"].append({
            "op": "archive", "src": d,
            "dst": _archive_dest(root, ts, d), "reason": reason,
            "class": cls, "merged_complements": d in merged_from,
        })

    # ---- registry + marker update ----------------------------------------
    if effective_winner is not None:
        for key in plan["keys"]:
            plan["actions"].append({
                "op": "registry", "key": key,
                "destination": effective_winner, "folder": plan["flat_folder"]})
        # Marker (task's new home): PRESERVE the winner's own marker task —
        # never bake the entry's path-GUESSED task. When the winner's marker has
        # a real task, keep it (a genuine vision model keeps its mmproj gate);
        # when its marker task is null OR there's no marker, flag
        # needs_classification instead of committing a guess (the registry's
        # default still serves text models; a text gguf mis-filed under
        # image-text-to-text stays null, not vision).
        wm = _read_marker(winner)
        if wm:
            marker_primary = wm.get("primary_task")
            marker_tasks = wm.get("tasks")
            needs_class = marker_primary is None and marker_tasks is None
        else:
            marker_primary = rep.get("primary_task") or rep.get("task")
            marker_tasks = rep.get("tasks") or (
                [marker_primary] if marker_primary else None)
            needs_class = True
        plan["actions"].append({
            "op": "marker", "path": effective_winner,
            "hub_id": rep.get("hub_id"),
            "framework": rep.get("framework") or wm.get("framework"),
            "primary_task": marker_primary,
            "tasks": marker_tasks,
            "needs_classification": needs_class,
        })
    else:
        plan["status"] = "incomplete-no-winner"
        plan["warnings"].append(
            "no complete copy on disk; left in place (nothing moved/deleted)")

    # ---- presence-honesty warning ----------------------------------------
    if effective_winner is not None:
        pin = pinned_filename_present(winner, cfg)
        if pin is False:
            plan["warnings"].append(
                f"pinned filename {rep.get('filename')!r} not present among the "
                f"installed quants — serves off another quant (WARNING, not partial)")

    if plan["status"] == "noop" and not plan["actions"]:
        plan["status"] = "already-flat"
    return plan


# ---------------------------------------------------------------------------
# Apply one plan's actions.
# ---------------------------------------------------------------------------
def _apply_plan(plan, root, ts, apply, registry_updates):
    """Execute (or, when apply=False, just resolve) a plan's actions in a
    resumable order: merge -> move -> archive -> registry/marker. Records
    registry field changes into ``registry_updates`` for a single JSON write."""
    p = _paths()
    winner = plan.get("winner")
    flat_dest = plan["flat_dest"]

    for act in plan["actions"]:
        op = act["op"]
        try:
            if op == "merge":
                _move_entry_into(act["from"], act["into"], act["file"], apply)
            elif op == "move":
                # winner may already have received merged files; move the dir.
                if apply and os.path.normpath(act["src"]) != os.path.normpath(act["dst"]):
                    o, _detail = _rename_or_copy(act["src"], act["dst"], apply)
                    if o == "copy":
                        # cross-device: source archived (never deleted)
                        os.makedirs(os.path.dirname(
                            _archive_dest(root, ts, act["src"])), exist_ok=True)
                        os.rename(act["src"], _archive_dest(root, ts, act["src"]))
            elif op == "archive":
                dst = _archive_dest(root, ts, act["src"])
                act["dst"] = dst
                if apply and os.path.isdir(act["src"]):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    # winner already moved to flat; a leftover source is archived
                    if os.path.exists(act["src"]):
                        os.rename(act["src"], dst)
            elif op == "registry":
                registry_updates.setdefault(act["key"], {})
                registry_updates[act["key"]].update(
                    {"destination": act["destination"], "dir": act["destination"],
                     "folder": act["folder"]})
            elif op == "marker":
                if apply:
                    _write_marker(
                        act["path"], hub_id=act.get("hub_id"),
                        framework=act.get("framework"),
                        primary_task=act.get("primary_task"),
                        tasks=act.get("tasks"), source="reconcile",
                        needs_classification=act.get("needs_classification", False))
        except Exception as exc:  # noqa: BLE001 — one action failing is data
            act["error"] = f"{type(exc).__name__}: {exc}"
            plan["warnings"].append(f"action {op} failed: {act['error']}")


def _persist_registry(registry_updates, apply):
    """Write dir/folder/destination back into BOTH catalog artifacts (discovery
    report is the registry source; manifest is the download intent). Only keys
    that already exist in a file are touched. Then refresh the live registry."""
    if not registry_updates or not apply:
        return {"written": 0, "files": []}
    c = _constants()
    written, files = 0, []
    for path in (c.MODELS_DISCOVERY_PATH, c.MODELS_DICT_PATH):
        data = _load_json(path)
        if not data:
            continue
        touched = False
        for key, fields in registry_updates.items():
            if key in data and isinstance(data[key], dict):
                data[key].update(fields)
                touched = True
                written += 1
        if touched:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, path)
            files.append(path)
    # An APPLIED reconcile MOVED weights between layouts, so every persisted
    # destination and size is suspect — for models we rewrote AND for their
    # neighbours (a merge/archive action retargets dirs the registry update
    # never names). refresh_registry(run_discovery=False) deliberately only
    # drops re-keyed rows, so say the wider thing here, where we know weights
    # moved. Whole-table: over-dropping costs a re-derive, under-dropping leaves
    # a record pointing at a dir that is no longer there.
    try:
        from hugpy_storage.model_physical import forget_all_physical
        forget_all_physical("reconcile applied")
    except Exception as exc:  # noqa: BLE001 — never fail a reconcile over a cache
        logger.debug("physical-state drop after reconcile skipped: %s", exc)
    try:
        from hugpy_engine.config.models.models_config import refresh_registry
        refresh_registry(run_discovery=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry refresh after reconcile failed: %s", exc)
    return {"written": written, "files": files}


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def reconcile_store(root=None, apply=False, report_path=None):
    """Plan (and, when apply=True, execute) the store-flattening migration.

    Returns a JSON-serializable report. Dry-run (default) touches NOTHING.
    Idempotent: re-running on an already-flat store yields an all-noop plan."""
    p = _paths()
    root = root or p.DEFAULT_ROOT
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    entries = _catalog_entries(root)
    groups = {}
    for entry in entries.values():
        gkey = _group_key(entry, root)
        groups.setdefault(gkey, []).append(entry)

    plans, registry_updates = [], {}
    for gkey, members in sorted(groups.items()):
        try:
            plan = _plan_group(gkey, members, root)
            plan["_ts"] = ts
            # re-resolve archive dsts now that ts is known
            for act in plan["actions"]:
                if act.get("op") == "archive":
                    act["dst"] = _archive_dest(root, ts, act["src"])
            _apply_plan(plan, root, ts, apply, registry_updates)
        except Exception as exc:  # noqa: BLE001 — never abort the whole run
            plan = {"hub_id": members[0].get("hub_id"), "runtime": gkey[0],
                    "status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "actions": [], "warnings": []}
        plan.pop("_ts", None)
        plans.append(plan)

    persisted = _persist_registry(registry_updates, apply)

    def _count(status):
        return sum(1 for pl in plans if pl["status"] == status)

    def _acts(op):
        return sum(1 for pl in plans for a in pl["actions"] if a.get("op") == op)

    report = {
        "root": root,
        "apply": bool(apply),
        "timestamp": ts,
        "archive_dir": os.path.join(root, "models", "_archive", ts),
        "summary": {
            "entries": len(entries),
            "groups": len(groups),
            "moves": _acts("move"),
            "merges": _acts("merge"),
            "archives": _acts("archive"),
            "already_flat": _count("already-flat") + _count("noop"),
            "incomplete_no_winner": _count("incomplete-no-winner"),
            "absent": _count("absent"),
            "errors": _count("error"),
            "warnings": sum(len(pl["warnings"]) for pl in plans),
            "registry_written": persisted["written"],
        },
        "plans": plans,
    }
    if report_path:
        try:
            tmp = report_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
            os.replace(tmp, report_path)
        except OSError as exc:
            logger.warning("could not write reconcile report to %s: %s",
                           report_path, exc)
    return report
