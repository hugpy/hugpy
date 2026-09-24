"""hugpy-model-manifest-backfill — the ONE-TIME capture of install manifests
for models downloaded before hugpy.json carried one.

Since 2026-09-23 every download writes ``manifest`` into the model's hugpy.json
(``{revision, captured_at, source, files: [{path, bytes, sha256}]}``) and every
later verification reads that record — nothing asks Hugging Face what a static
file should be after install. Models installed before then have no record;
this tool writes one, once:

  * ``hub_id`` is a real Hub repo -> the Hub listing (sizes + LFS sha256 +
    revision), for the files ACTUALLY on disk (the install-time capture that
    never happened). The revision the local HF download metadata records is
    pinned when it is uniform, so the sizes are those of what was downloaded.
    One listing per repo, sequential, bounded retry on 429.
  * otherwise (placeholder ids like ``comfy/<name>``, comfy checkpoints, a Hub
    404/401, no hub_id) -> the local files: bytes from disk, sha256 from the
    local HF download metadata when present else null, ``source: "local"``.

SIZE (2026-09-23, "size must never be blank"): every model whose hugpy.json
carries a manifest but no ``size_bytes`` gets its size fields stamped from that
manifest (hugpy_storage.hugpy_marker.stamp_marker_sizes — a sum over the
record, no walk, no Hub); a manifest this run writes stamps them in the same
write. The summary reports ``sized`` / ``would_size`` and the reasons a model
could not be sized.

DRY-RUN BY DEFAULT: it lists what it would write and from which source, and
makes no network call to the Hub. ``--apply`` writes (atomically, only into
hugpy.json files that lack a manifest or lack a size). Model files are never
touched.

Exit codes: 0 ok, 2 tool error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Optional

HUB_API = "https://huggingface.co/api/models/{repo}{rev}?blobs=true"
HUB_TIMEOUT = 20.0
HUB_RETRIES = 3
# Owners that are hugpy-side placeholders, never Hub namespaces.
PLACEHOLDER_OWNERS = {"comfy", "local", "custom", "hugpy", "localhost"}
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def is_hub_repo_id(hub_id: Optional[str], framework: Optional[str] = None,
                   marker_source: Optional[str] = None) -> bool:
    """Is ``hub_id`` shaped like a real Hub repo (not a hugpy placeholder)?
    Decided locally — no network."""
    if not hub_id or not _REPO_RE.match(str(hub_id)):
        return False
    if str(framework or "").lower() == "comfy" or str(marker_source or "").lower() == "custom":
        return False
    return str(hub_id).split("/", 1)[0].lower() not in PLACEHOLDER_OWNERS


def _cache_revision(dest: str, rels) -> Optional[str]:
    from hugpy_storage.hugpy_marker import read_hf_download_metadata
    commits = set()
    for rel in rels:
        m = read_hf_download_metadata(dest, rel)
        if m and m.get("commit"):
            commits.add(m["commit"])
    return next(iter(commits)) if len(commits) == 1 else None


def hub_listing(hub_id: str, revision: Optional[str] = None, token: Optional[str] = None,
                timeout: float = HUB_TIMEOUT) -> dict:
    """``{status, revision, files: {rfilename: {bytes, sha256}}}``. Only the
    backfill calls this — once per repo, never at audit/verify time."""
    from hugpy_ops.model_audit import _request
    rev = f"/revision/{revision}" if revision else ""
    st, body = None, None
    for attempt in range(HUB_RETRIES + 1):
        try:
            st, body = _request(HUB_API.format(repo=hub_id, rev=rev), token=token, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "files": {}, "error": f"{type(exc).__name__}: {exc}"}
        if st != 429 or attempt == HUB_RETRIES:
            break
        time.sleep(min(30.0, 5.0 * (attempt + 1)))
    if st == 404 and revision:
        return hub_listing(hub_id, None, token, timeout)   # pinned revision gone: current listing
    if st != 200 or not isinstance(body, dict):
        return {"status": st, "files": {}}
    files = {}
    for s in body.get("siblings") or []:
        name = s.get("rfilename")
        if not name:
            continue
        lfs = s.get("lfs") or {}
        size = s.get("size") if s.get("size") is not None else lfs.get("size")
        files[name] = {"bytes": size, "sha256": lfs.get("sha256")}
    return {"status": 200, "revision": body.get("sha"), "files": files}


def _hf_token() -> Optional[str]:
    try:
        from hugpy_storage.hf_token import get_hf_token
        tok = get_hf_token()
        if tok:
            return tok
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("HF_TOKEN") or None


def plan_one(dest: str, row: dict) -> dict:
    """What the backfill would do for one model dir. No network."""
    from hugpy_storage.hugpy_marker import (MANIFEST_KEY, manifest_candidate_files,
                                            read_hf_download_metadata, read_hugpy_marker)
    out = {"model_key": row.get("model_key"), "hub_id": row.get("hub_id"),
           "framework": row.get("framework"), "destination": dest}
    if not dest or not os.path.isdir(dest):
        return {**out, "action": "skip", "reason": "destination absent"}
    marker = read_hugpy_marker(dest)
    if not isinstance(marker, dict):
        return {**out, "action": "skip", "reason": "no hugpy.json"}
    if isinstance(marker.get(MANIFEST_KEY), dict):
        if marker.get("size_bytes") is None:
            from hugpy_storage.hugpy_marker import manifest_size_fields
            fields = manifest_size_fields(marker)
            if fields is None:
                return {**out, "action": "skip",
                        "reason": "manifest present but lists no file bytes (cannot size)"}
            return {**out, "action": "size", "size_bytes": fields.get("size_bytes"),
                    "size_note": fields.get("size_note")}
        return {**out, "action": "skip", "reason": "manifest + size present"}
    files = manifest_candidate_files(dest)
    if not files:
        return {**out, "action": "skip", "reason": "no model files"}
    hub_id = marker.get("hub_id") or row.get("hub_id")
    fw = marker.get("framework") or row.get("framework")
    source = "hub" if is_hub_repo_id(hub_id, fw, marker.get("source")) else "local"
    cached = sum(1 for r in files if read_hf_download_metadata(dest, r))
    return {**out, "hub_id": hub_id, "action": "backfill", "source": source,
            "n_files": len(files), "n_hf_cache_metadata": cached}


def backfill_one(plan: dict, token: Optional[str] = None, listings: Optional[dict] = None) -> dict:
    """Write the manifest planned by :func:`plan_one`. Returns the plan + result."""
    from hugpy_storage.hugpy_marker import (build_install_manifest, manifest_candidate_files,
                                            write_marker_manifest)
    dest = plan["destination"]
    files = manifest_candidate_files(dest)
    manifest = None
    hub_status = None
    if plan["source"] == "hub":
        rev = _cache_revision(dest, files)
        key = (plan["hub_id"], rev)
        listing = (listings or {}).get(key)
        if listing is None:
            listing = hub_listing(plan["hub_id"], rev, token)
            if listings is not None:
                listings[key] = listing
        hub_status = listing.get("status")
        if hub_status == 200:
            expected = {r: m for r, m in listing["files"].items() if m.get("bytes") is not None}
            manifest = build_install_manifest(dest, files=files, source="huggingface",
                                              revision=listing.get("revision"), expected=expected)
            listed = sum(1 for r in files if r in expected)
            manifest["backfill"] = {"hub_status": 200, "files_listed_on_hub": listed,
                                    "files_from_disk": len(files) - listed}
    if manifest is None:
        manifest = build_install_manifest(dest, files=files, source="local")
        manifest["backfill"] = {"hub_status": hub_status} if hub_status is not None else {}
    manifest["backfill"]["backfilled_at"] = manifest["captured_at"]
    path = write_marker_manifest(dest, manifest, merge=False)   # stamps the size fields too
    from hugpy_storage.hugpy_marker import read_hugpy_marker
    sized = (read_hugpy_marker(dest) or {}).get("size_bytes") if path else None
    return {**plan, "written": bool(path), "wrote_source": manifest["source"],
            "hub_status": hub_status, "n_files": len(manifest["files"]), "size_bytes": sized}


def size_one(plan: dict) -> dict:
    """Stamp the size fields planned by :func:`plan_one` (``action: size``)."""
    from hugpy_storage.hugpy_marker import stamp_marker_sizes
    fields, why = stamp_marker_sizes(plan["destination"])
    return {**plan, "written": fields is not None,
            "size_bytes": (fields or {}).get("size_bytes"), "reason": why or None}


def catalog_rows(central: Optional[str]) -> list:
    from hugpy_ops.model_audit import central_url, fetch_json
    base = central_url(central)
    cat = fetch_json(f"{base}/api/models?verbose=1")
    if isinstance(cat, dict):
        cat = cat.get("models") or cat.get("rows") or []
    return list(cat or [])


def run(rows: list, *, apply: bool = False, only=None) -> dict:
    if only:
        want = set(only)
        rows = [r for r in rows if r.get("model_key") in want]
    seen: set = set()
    plans = []
    for r in rows:
        dest = r.get("destination") or ""
        real = os.path.realpath(dest) if dest else dest
        if real and real in seen:
            continue
        seen.add(real)
        plans.append(plan_one(dest, r))
    results = plans
    if apply:
        token = _hf_token()
        listings: dict = {}
        results = [backfill_one(p, token, listings) if p["action"] == "backfill"
                   else size_one(p) if p["action"] == "size" else p for p in plans]
    todo = [p for p in plans if p["action"] == "backfill"]
    to_size = [p for p in plans if p["action"] == "size"]
    summary = {
        "mode": "apply" if apply else "dry-run",
        "models": len(plans),
        "would_backfill" if not apply else "backfilled": len(todo),
        "from_hub": sum(1 for p in todo if p["source"] == "hub"),
        "from_local": sum(1 for p in todo if p["source"] == "local"),
        "with_hf_cache_metadata": sum(1 for p in todo if p.get("n_hf_cache_metadata")),
        "would_size" if not apply else "size_planned": len(to_size),
        "skipped": {},
    }
    for p in plans:
        if p["action"] == "skip":
            summary["skipped"][p["reason"]] = summary["skipped"].get(p["reason"], 0) + 1
    if apply:
        summary["sized"] = sum(1 for r in results if r.get("action") in ("size", "backfill")
                               and r.get("written") and r.get("size_bytes") is not None)
        summary["size_failed"] = [{"model_key": r.get("model_key"), "reason": r.get("reason")}
                                  for r in results if r.get("action") == "size" and not r.get("written")]
        summary["written_source"] = {}
        for r in results:
            if r.get("wrote_source"):
                summary["written_source"][r["wrote_source"]] = summary["written_source"].get(r["wrote_source"], 0) + 1
    return {"summary": summary, "models": results}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hugpy-model-manifest-backfill",
        description="One-time: write the install manifest (hugpy.json 'manifest') for models that lack one, "
                    "and stamp size_bytes from the manifest for models that lack a size. "
                    "Dry-run by default (no Hub calls); --apply writes.")
    p.add_argument("--central", help="central base URL (default: HUGPY_BASE_URL or http://127.0.0.1:7002)")
    p.add_argument("--dirs", nargs="+", metavar="DIR", help="model dirs to consider instead of central's catalog")
    p.add_argument("--only", nargs="+", metavar="KEY", help="only these model keys")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="report the plan only (default)")
    g.add_argument("--apply", action="store_true", help="write the manifests")
    p.add_argument("--json", metavar="PATH", help="write the full per-model result as JSON")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.dirs:
            rows = [{"model_key": os.path.basename(d.rstrip("/")), "destination": d} for d in args.dirs]
        else:
            rows = catalog_rows(args.central)
        res = run(rows, apply=bool(args.apply), only=args.only)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"hugpy-model-manifest-backfill: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for m in res["models"]:
        if m["action"] == "backfill":
            print(f"{'wrote' if m.get('written') else 'would':6} {m.get('wrote_source') or m['source']:12} "
                  f"{m['n_files']:4d} files  {m['model_key']}")
        elif m["action"] == "size":
            print(f"{'sized' if m.get('written') else 'would':6} {'size':12} "
                  f"{m.get('size_bytes') if m.get('size_bytes') is not None else '?':>14} B  {m['model_key']}"
                  + (f"  ({m['size_note']})" if m.get("size_note") else "")
                  + (f"  FAILED: {m['reason']}" if m.get("reason") else ""))
    print(json.dumps(res["summary"], indent=1))
    if args.json:
        with open(args.json + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
        os.replace(args.json + ".tmp", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
