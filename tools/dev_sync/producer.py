#!/usr/bin/env python3
"""dev_sync producer: turn file saves in the canonical dev tree into dev_changes rows.

Scan-based (robust: a missed inotify event can't cause drift — a scan reconciles).
For each package under <repo>/py/*/*/, walk the package root, hash every source
file, and append a full-content row to dev_changes for anything that differs from
the latest recorded row. Emits an install_signal when pyproject.toml or a native
source changes (metadata / rebuild), never for plain .py edits.

    python producer.py            # scan + write to the toolserver DB
    python producer.py --dry-run  # report what it WOULD write, touch nothing

Reuses abstract_toolserver.db._connect() for the same DB the toolserver uses.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

from abstract_toolserver import db

REPO = Path(__file__).resolve().parents[2]          # ~/hugpy-ws
PKG_GLOB = "py/*/*/pyproject.toml"
AUTHOR = os.environ.get("USER", "producer") + "@" + os.uname().nodename

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache",
             "node_modules", "build", "dist"}
SKIP_SUFFIX = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".egg-link"}
NATIVE_SUFFIX = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".pyx", ".pyi",
                 ".pxd", ".cu", ".cuh"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def discover():
    """Yield (package, rel_root, src_dir, import_name, has_native) per pyproject."""
    for pp in sorted(REPO.glob(PKG_GLOB)):
        root = pp.parent
        text = pp.read_text(encoding="utf-8", errors="replace")
        name = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("name") and "=" in s:
                name = s.split("=", 1)[1].strip().strip('"').strip("'")
                break
        if not name:
            continue
        src = root / "src"
        import_name = None
        if src.is_dir():
            for d in sorted(src.iterdir()):
                if d.is_dir() and (d / "__init__.py").exists() and not d.name.endswith(".egg-info"):
                    import_name = d.name
                    break
        has_native = any(
            p.suffix in NATIVE_SUFFIX for p in root.rglob("*")
            if not _skip(p)
        )
        yield (name, str(root.relative_to(REPO)),
               str((src if src.is_dir() else root).relative_to(REPO)),
               import_name or name.replace("-", "_"), has_native)


def _skip(p: Path) -> bool:
    if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in p.parts):
        return True
    return p.suffix in SKIP_SUFFIX


def walk(root: Path):
    """Yield (rel_path_to_root, bytes) for every tracked file under a package root."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and not _skip(p):
            yield str(p.relative_to(root)), p.read_bytes()


def latest_map(cur, package):
    """rel_path -> (op, result_hash) for the newest row of each path."""
    cur.execute(
        "SELECT DISTINCT ON (rel_path) rel_path, op, result_hash "
        "FROM dev_changes WHERE package = %s ORDER BY rel_path, id DESC",
        (package,),
    )
    return {r["rel_path"]: (r["op"], r["result_hash"]) for r in cur.fetchall()}


def is_meta(rel_path: str) -> bool:
    name = Path(rel_path).name
    return name in ("pyproject.toml", "setup.py", "setup.cfg") \
        or Path(rel_path).suffix in NATIVE_SUFFIX


def scan_package(cur, meta, dry):
    name, rel_root, src_dir, import_name, has_native = meta
    root = REPO / rel_root
    known = latest_map(cur, name)
    on_disk = dict(walk(root))
    changes = []   # (op, rel_path, content_or_None, base_hash, result_hash)

    for rel, content in on_disk.items():
        h = sha(content)
        prev = known.get(rel)
        if prev is None or prev[0] == "delete":
            changes.append(("create", rel, content, None, h))
        elif prev[1] != h:
            changes.append(("modify", rel, content, prev[1], h))
    for rel, (op, rh) in known.items():
        if op != "delete" and rel not in on_disk:
            changes.append(("delete", rel, None, rh, None))

    if not changes:
        return 0, False
    meta_touched = any(is_meta(c[1]) for c in changes)

    if dry:
        for op, rel, _c, _b, _r in changes:
            print(f"  [{name}] {op:6} {rel}")
        if meta_touched:
            print(f"  [{name}] install_signal (metadata/native changed)")
        return len(changes), meta_touched

    last_id = None
    for op, rel, content, base, res in changes:
        cur.execute(
            "INSERT INTO dev_changes (package, rel_path, op, content, base_hash, "
            "result_hash, kind, author) VALUES (%s,%s,%s,%s,%s,%s,'file',%s) RETURNING id",
            (name, rel, op, content, base, res, AUTHOR),
        )
        last_id = cur.fetchone()["id"]
    if meta_touched:
        cur.execute(
            "INSERT INTO dev_changes (package, rel_path, op, kind, author) "
            "VALUES (%s,%s,'modify','install_signal',%s) RETURNING id",
            (name, "pyproject.toml", AUTHOR),
        )
        last_id = cur.fetchone()["id"]

    tree = sha("\n".join(f"{r}:{sha(c)}" for r, c in sorted(on_disk.items())).encode())
    cur.execute(
        "INSERT INTO dev_modules (package, rel_root, src_dir, import_name, "
        "internal_version, content_hash, dirty, has_native, updated_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,true,%s,now()) "
        "ON CONFLICT (package) DO UPDATE SET rel_root=EXCLUDED.rel_root, "
        "src_dir=EXCLUDED.src_dir, import_name=EXCLUDED.import_name, "
        "internal_version=EXCLUDED.internal_version, content_hash=EXCLUDED.content_hash, "
        "dirty=true, has_native=EXCLUDED.has_native, updated_at=now()",
        (name, rel_root, src_dir, import_name, last_id, tree, has_native),
    )
    return len(changes), meta_touched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pkgs = list(discover())
    if not pkgs:
        print(f"no packages found under {REPO}/{PKG_GLOB}", file=sys.stderr)
        return 1

    total = 0
    conn = db._connect()
    try:
        with conn.cursor() as cur:
            for meta in pkgs:
                n, _ = scan_package(cur, meta, args.dry_run)
                total += n
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    finally:
        conn.close()

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {total} change row(s) across {len(pkgs)} package(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
