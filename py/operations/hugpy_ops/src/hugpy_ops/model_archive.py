"""hugpy-model-archive — turn the console's ARCHIVE MARKS into the archive state.

Two states only: LIVE or ARCHIVE (operator ruling 2026-09-23). The console's
🗄 Archive button only MARKS a model (``hugpy.json["archive"] = {marked, at,
by, reason}``, see hugpy_storage.archive_mark) and central then refuses to
place or route it. This sweep is the operator's own act that finishes the job,
per marked model:

  1. SELECT every catalog row (``/api/models?verbose=1``) whose ``archived``
     projection is marked AND whose on-disk hugpy.json still carries the mark
     (the marker is the authority; a catalog row that disagrees is reported,
     never moved).
  2. REFUSE when the model is loaded / serving / seated on any worker (the
     same ``workers[]`` check as hugpy-model-resweep), unless ``--force``.
  3. MOVE central's model directory with one ``rename`` (same filesystem;
     nothing copied, nothing deleted) to
     ``ARCHIVE/MODELS_ARCHIVED-<date>/<framework>/<owner>/<name>/``; a comfy
     model's checkpoint drop-file (+ ``.civitai.json``) moves beside it as
     ``<name>.checkpoint/`` so the checkpoint sweep cannot re-register it.
  4. MANIFEST: append one row to that dir's ``MANIFEST.md`` — model, original,
     archived, who marked it / when / why (from the mark), restore command.
  5. CATALOG: the detail read (``GET /api/models/<key>``) re-derives the row as
     not-installed, then ``POST /models/<key>/prune`` removes it from the
     catalog (prune refuses while files exist, so it only lands after the move).
  6. OUTCOME: one JSON line per model in ``OUTCOMES.jsonl`` beside the manifest
     (archived / refused / skipped / error, with the reason) and a summary.

Dry-run by default; ``--apply`` performs. Worker copies are left to hugpy's
own storage reconcile — this tool never touches a worker, never downloads.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import urllib.parse
from typing import Any, Optional

from hugpy_ops import model_audit as ma
from hugpy_ops.resweep import (ARCHIVE_ROOT, Central, archive_target, checkpoint_filename,
                               loaded_on, move, storage_root)

MANIFEST_HEADER = (
    "# MODELS_ARCHIVED — central model dirs the operator marked for archive\n\n"
    "Marked from the console (🗄 Archive: hugpy.json `archive` = {marked, at, by, reason}); "
    "moved here with one `rename` each (same filesystem; nothing copied, nothing deleted) by "
    "`hugpy-model-archive --apply`, then pruned from the catalog. Restore = the row's command "
    "(moves the files back, un-prunes the catalog row, clears the mark).\n\n"
    "| model | original | archived | marked by | marked at | reason | restore |\n"
    "|---|---|---|---|---|---|---|\n")


def _q(key: str) -> str:
    return urllib.parse.quote(key, safe="~")


def _cell(v: Any) -> str:
    return str(v if v not in (None, "") else "(none recorded)").replace("|", "\\|").replace("\n", " ")


def restore_command(model_key: str, moves: list) -> str:
    """mv every moved path back, un-prune the catalog row, clear the mark."""
    parts = [f'mv "{dst}" "{src}"' for src, dst in moves]
    parts.append("python3 -c 'from hugpy_engine.config.models.models_config import unprune_model; "
                 f"print(unprune_model(\"{model_key}\"))'")
    parts.append(f'curl -s -X DELETE -H "X-Operator-Token: $HUGPY_OPERATOR_TOKEN" '
                 f'"$CENTRAL/api/llm/models/{_q(model_key)}/archive"')
    return " && ".join(parts)


def manifest_row(model_key: str, moves: list, mark: dict) -> str:
    src, dst = moves[0]
    return (f"| `{model_key}` | `{src}` | `{dst}` | {_cell(mark.get('by'))} | {_cell(mark.get('at'))} | "
            f"{_cell(mark.get('reason'))} | `{restore_command(model_key, moves)}` |")


def append_manifest(archive_base: str, row: str) -> str:
    os.makedirs(archive_base, exist_ok=True)
    path = os.path.join(archive_base, "MANIFEST.md")
    fresh = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as fh:
        if fresh:
            fh.write(MANIFEST_HEADER)
        fh.write(row + "\n")
    return path


def record_outcome(archive_base: str, outcome: dict) -> str:
    os.makedirs(archive_base, exist_ok=True)
    path = os.path.join(archive_base, "OUTCOMES.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(outcome, default=str) + "\n")
    return path


def marked_rows(catalog: list) -> list:
    return [r for r in catalog or []
            if isinstance(r, dict) and isinstance(r.get("archived"), dict) and r["archived"].get("marked")]


def plan_one(row: dict, *, archive_base: str, checkpoints_dir: str, force: bool = False) -> dict:
    from hugpy_storage.archive_mark import read_archive_mark, is_marked
    key = row.get("model_key") or row.get("key")
    dest = row.get("destination")
    entry = {"model_key": key, "hub_id": row.get("hub_id"), "framework": row.get("framework")}
    p: dict = {"model_key": key, "framework": row.get("framework"), "original": dest,
               "archived": archive_target(archive_base, entry), "moves": [],
               "mark": dict(row.get("archived") or {})}
    disk = read_archive_mark(dest) if dest else None
    if not is_marked(disk):
        p["skipped"] = (f"catalog says marked ({p['mark'].get('by')} at {p['mark'].get('at')}) but "
                        f"{dest}/hugpy.json carries no archive mark — not moved")
        return p
    p["mark"] = {k: disk.get(k) for k in ("marked", "at", "by", "reason")}
    busy = loaded_on(row)
    p["loaded_on"] = busy
    if busy and not force:
        p["refused"] = f"loaded/serving/seated on {', '.join(busy)} — unload it first or pass --force"
    if dest and os.path.lexists(dest):
        p["moves"].append((dest, p["archived"]))
    else:
        p["skipped"] = f"no directory at {dest!r} on central — nothing to move"
        return p
    if row.get("framework") == "comfy":
        fn = checkpoint_filename(entry | {"destination": dest}, row)
        for f in ((fn, fn + ".civitai.json") if fn else ()):
            cp = os.path.join(checkpoints_dir, f)
            if os.path.lexists(cp):
                p["moves"].append((cp, os.path.join(p["archived"] + ".checkpoint", f)))
    for _src, dst in p["moves"]:
        if os.path.lexists(dst):
            p["refused"] = f"archive target already exists: {dst}"
    return p


def apply_one(central: Central, p: dict, archive_base: str) -> dict:
    key = p["model_key"]
    out = {"model_key": key, "at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "mark": p.get("mark"), "moves": p.get("moves")}
    if p.get("refused") or p.get("skipped"):
        return {**out, "status": "refused" if p.get("refused") else "skipped",
                "reason": p.get("refused") or p.get("skipped")}
    done = []
    try:
        for src, dst in p["moves"]:
            move(src, dst)
            done.append((src, dst))
        out["manifest"] = append_manifest(archive_base, manifest_row(key, done, p["mark"]))
    except Exception as exc:  # noqa: BLE001 — report exactly what moved
        return {**out, "moves": done, "status": "error",
                "reason": f"move failed after {len(done)}/{len(p['moves'])}: {type(exc).__name__}: {exc}"}
    st, _ = central.get("/api/models/" + _q(key))
    out["refresh_http"] = st
    pst, presp = central.post("/api/models/" + _q(key) + "/prune", {})
    out["prune_http"] = pst
    if pst not in (200, 201):
        return {**out, "status": "archived_not_pruned",
                "reason": f"files moved + manifest row written; POST /models/{key}/prune -> "
                          f"HTTP {pst}: {str(presp)[:300]}"}
    return {**out, "status": "archived", "reason": f"moved to {done[0][1]}; catalog row pruned"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hugpy-model-archive", description=__doc__.split("\n\n")[0])
    p.add_argument("--apply", action="store_true", help="move + manifest + prune (default: dry-run)")
    p.add_argument("--only", nargs="+", metavar="KEY", help="only these (marked) model keys")
    p.add_argument("--force", action="store_true", help="proceed even when a worker has it loaded")
    p.add_argument("--central", help="central base URL")
    p.add_argument("--archive-root", default=ARCHIVE_ROOT)
    p.add_argument("--date", default=_dt.date.today().isoformat())
    p.add_argument("--json", action="store_true", help="print the plan as JSON")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    central = Central(args.central)
    try:
        catalog = ma.fetch_json(central.base + "/api/models?verbose=1")
    except Exception as exc:  # noqa: BLE001
        print(f"hugpy-model-archive: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    catalog = catalog if isinstance(catalog, list) else (catalog or {}).get("models") or []
    rows = marked_rows(catalog)
    if args.only:
        want = set(args.only)
        missing = want - {r.get("model_key") for r in rows}
        rows = [r for r in rows if r.get("model_key") in want]
        for k in sorted(missing):
            print(f"[not marked] {k}: no marked catalog row with this key")
    base = os.path.join(args.archive_root, f"MODELS_ARCHIVED-{args.date}")
    checkpoints = os.path.join(storage_root(), "checkpoints")
    plans = [plan_one(r, archive_base=base, checkpoints_dir=checkpoints, force=args.force) for r in rows]
    unread = sum(1 for r in catalog if isinstance(r, dict) and r.get("archived") is None)
    print(f"catalog: {len(catalog)} rows, {len(rows)} marked for archive"
          + (f", {unread} with an unreadable marker (archived=null)" if unread else ""))
    print(f"archive: {base}   mode: {'APPLY' if args.apply else 'dry-run'}")
    if args.json:
        print(json.dumps(plans, indent=1, default=str))
    for p in plans:
        m = p.get("mark") or {}
        state = "REFUSED" if p.get("refused") else "SKIP" if p.get("skipped") else "plan"
        print(f"\n[{state}] {p['model_key']} ({p['framework']}) — marked by {m.get('by')} at "
              f"{m.get('at')}: {m.get('reason') or '(no reason recorded)'}")
        for src, dst in p["moves"]:
            print(f"  move  {src}\n     -> {dst}")
        if p.get("refused") or p.get("skipped"):
            print(f"  why   {p.get('refused') or p.get('skipped')}")
    if not args.apply:
        return 0
    results = []
    for p in plans:
        outcome = apply_one(central, p, base)
        try:
            outcome["outcomes_file"] = record_outcome(base, outcome)
        except OSError as exc:
            outcome["outcomes_file_error"] = f"{type(exc).__name__}: {exc}"
        results.append(outcome)
        print(f"{outcome['model_key']}: {outcome['status']} — {outcome['reason']}")
    print("\nARCHIVE RESULTS")
    counts: dict = {}
    for o in results:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
        print(f"  {o['status']:<20} {o['model_key']}: {o['reason']}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "  nothing marked")
    return 0 if all(o["status"] == "archived" for o in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
