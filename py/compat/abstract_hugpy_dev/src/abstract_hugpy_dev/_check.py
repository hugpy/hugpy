"""``abstract-hugpy-dev-check``: prove the alias layer resolves.

    abstract-hugpy-dev-check              import every aliased old module
    abstract-hugpy-dev-check --surface    also check the re-exported names
    abstract-hugpy-dev-check scan PATH... list old imports found in PATH with
                                          their new targets (migration aid)

Exit status 0 when every check passes, 1 otherwise. ``--json`` prints the
report as JSON instead of text.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path

from . import _relocations as R

ROOT = R._ROOT


def _import_report(surface: bool) -> dict:
    ok: list[str] = []
    failed: dict[str, str] = {}
    mismatched: dict[str, str] = {}
    optional_missing: dict[str, str] = {}
    for old, new, kind in R.iter_old_modules():
        if kind == "relocated" and not R._target_exists(new):
            failed[old] = f"target {new} is not installed"
            continue
        try:
            mod = importlib.import_module(old)
        except ModuleNotFoundError as exc:
            # A hugpy_* target that needs an extra (discord.py, torch, ...)
            # is aliased correctly; the extra just is not installed here.
            missing = exc.name or ""
            if missing and not missing.startswith(("hugpy", ROOT)):
                optional_missing[old] = missing
            else:
                failed[old] = f"{type(exc).__name__}: {exc}"
            continue
        except Exception as exc:  # noqa: BLE001 - report, never raise
            failed[old] = f"{type(exc).__name__}: {exc}"
            continue
        if kind == "relocated":
            target = sys.modules.get(new)
            if target is None or target is not mod:
                mismatched[old] = new
                continue
        ok.append(old)
    report = {
        "ok": len(ok),
        "failed": failed,
        "mismatched": mismatched,
        "optional_missing": optional_missing,
        "retired": dict(R._RETIRED),
    }
    if surface:
        pkg = importlib.import_module(ROOT)
        missing = [n for n in getattr(pkg, "_SURFACE", ()) if not hasattr(pkg, n)]
        report["surface_total"] = len(getattr(pkg, "_SURFACE", ()))
        report["surface_missing"] = missing
    return report


def _old_imports(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        yield path, 0, f"<syntax error: {exc}>", None
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == ROOT or a.name.startswith(ROOT + "."):
                    yield path, node.lineno, a.name, R.relocated(a.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == ROOT or node.module.startswith(ROOT + "."):
                yield path, node.lineno, node.module, R.relocated(node.module)


def _scan(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for raw in paths:
        p = Path(raw)
        files = [p] if p.is_file() else sorted(p.rglob("*.py"))
        for f in files:
            for path, line, old, new in _old_imports(f):
                rows.append({"file": str(path), "line": line, "old": old,
                             "new": new, "kind": ("aggregator" if R.is_namespace(old) and new is None
                                                  else "retired" if new is None else "relocated")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="abstract-hugpy-dev-check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--surface", action="store_true",
                    help="also verify every re-exported top-level name resolves")
    sub = ap.add_subparsers(dest="cmd")
    scan = sub.add_parser("scan", help="list old imports under PATH with their new targets")
    scan.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "scan":
        rows = _scan(args.paths)
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                print(f"{r['file']}:{r['line']}: {r['old']} -> {r['new'] or '(' + r['kind'] + ')'}")
            print(f"{len(rows)} old import(s) found", file=sys.stderr)
        return 0

    report = _import_report(args.surface)
    bad = bool(report["failed"] or report["mismatched"] or report.get("surface_missing"))
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(f"{ROOT} alias layer: {report['ok']} old module(s) resolve")
        for old, why in report["failed"].items():
            print(f"  FAIL {old}: {why}")
        for old, new in report["mismatched"].items():
            print(f"  MISMATCH {old}: not the same object as {new}")
        for old, dep in report["optional_missing"].items():
            print(f"  SKIP {old}: optional dependency {dep!r} not installed")
        if args.surface:
            print(f"top-level surface: {report['surface_total'] - len(report['surface_missing'])}"
                  f"/{report['surface_total']} names present")
            for n in report["surface_missing"]:
                print(f"  MISSING {n}")
        print(f"retired without alias: {len(report['retired'])}")
        print("result:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
