#!/usr/bin/env python3
"""Install the partitioned Hugpy workspace editable into a virtualenv.

Stdlib only. Reads ``py/partition.toml`` for the thirteen ``hugpy-*``
distributions, installs them (plus the ``abstract_hugpy_dev`` compatibility
shell) with one ``pip install -e`` so every inter-package dependency resolves
from this checkout instead of PyPI, and optionally proves the install.

    py/local_install.py                       # into ./.venv (created if missing)
    py/local_install.py --venv /tmp/v --check # fresh venv + verification
    py/local_install.py --extras server       # hugpy[server]: the central box
    py/local_install.py --extras worker,gpu   # any extras of the hugpy meta package
    py/local_install.py --no-compat           # skip the abstract_hugpy_dev alias
    py/local_install.py --check               # only verify an existing venv

The system interpreter is refused on purpose: the miniconda on the dev box has
the retired monolith 0.1.266 in site-packages, which silently satisfies old
imports and hides partition mistakes. ``--allow-system`` overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parent
WORKSPACE = PY_ROOT.parent
MANIFEST = PY_ROOT / "partition.toml"
COMPAT = PY_ROOT / "compat" / "abstract_hugpy_dev"
COMPAT_VERSION_PREFIX = "0.1.267"
META_ID = "meta"

_BLOCK = re.compile(r"^\[\[package\]\]\s*$", re.M)
_KEY = re.compile(r'^(\w+)\s*=\s*(.+?)\s*$', re.M)


def read_packages(manifest: Path = MANIFEST) -> list[dict]:
    """``[[package]]`` entries in manifest (cut) order, without tomllib."""
    text = manifest.read_text(encoding="utf-8")
    starts = [m.end() for m in _BLOCK.finditer(text)]
    out: list[dict] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end]
        # a package block ends at its own [package.moves] table
        head = block.split("[package.moves]", 1)[0]
        head = re.sub(r"\[\[external_package\]\].*", "", head, flags=re.S)
        entry: dict = {}
        for key, raw in _KEY.findall(head):
            if key in ("id", "distribution", "import_name", "destination", "status"):
                entry[key] = raw.strip().strip('"')
        entry["depends"] = re.findall(r'"([^"]+)"', _list_field(head, "depends"))
        if "id" in entry:
            out.append(entry)
    if not out:
        raise SystemExit(f"no [[package]] entries found in {manifest}")
    return out


def _list_field(head: str, name: str) -> str:
    m = re.search(rf"^{name}\s*=\s*\[(.*?)\]", head, re.S | re.M)
    return m.group(1) if m else ""


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def ensure_venv(venv: Path, base_python: str, dry_run: bool) -> Path:
    py = venv_python(venv)
    if py.exists():
        return py
    cmd = [base_python, "-m", "venv", "--upgrade-deps", str(venv)]
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)
    return py


def is_system_interpreter(py: Path) -> bool:
    code = "import sys; print(int(sys.prefix == sys.base_prefix))"
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return out.strip() == "1"


def install_specs(packages: list[dict], extras: list[str], compat: bool, only: set[str] | None) -> list[str]:
    specs: list[str] = []
    for p in packages:
        if only and p["id"] not in only:
            continue
        root = PY_ROOT / p["destination"]
        if not (root / "pyproject.toml").exists():
            raise SystemExit(f"{p['distribution']}: no pyproject.toml at {root}")
        spec = str(root)
        if p["id"] == META_ID and extras:
            spec += "[" + ",".join(extras) + "]"
        specs.append(spec)
    if compat and (not only or "compat" in only):
        if not (COMPAT / "pyproject.toml").exists():
            raise SystemExit(f"compat shell missing at {COMPAT}")
        specs.append(str(COMPAT))
    return specs


def pip_install(py: Path, specs: list[str], pip_args: list[str], dry_run: bool) -> None:
    cmd = [str(py), "-m", "pip", "install", *pip_args]
    for s in specs:
        cmd += ["-e", s]
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


_CHECK_SNIPPET = r"""
import importlib, importlib.metadata as md, json, sys
from pathlib import Path
spec = json.loads(sys.argv[1])
report = {"imports": {}, "compat": None, "server": None, "stale_monolith": None}
for name in spec["imports"]:
    try:
        m = importlib.import_module(name)
        report["imports"][name] = str(Path(m.__file__).resolve().parent)
    except Exception as exc:
        report["imports"][name] = f"ERROR {type(exc).__name__}: {exc}"
if spec["compat"]:
    try:
        import abstract_hugpy_dev
        where = str(Path(abstract_hugpy_dev.__file__).resolve().parent)
        ver = md.version("abstract_hugpy_dev")
        report["compat"] = {"where": where, "version": ver,
                            "ok": where.startswith(spec["compat_src"]) and ver.startswith(spec["compat_ver"])}
    except Exception as exc:
        report["compat"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
else:
    try:
        ver = md.version("abstract_hugpy_dev")
        report["stale_monolith"] = ver
    except md.PackageNotFoundError:
        report["stale_monolith"] = None
if spec["server"]:
    try:
        import hugpy_server.wsgi_app, hugpy_server.wiring  # noqa: F401
        report["server"] = "ok"
    except Exception as exc:
        report["server"] = f"ERROR {type(exc).__name__}: {exc}"
print(json.dumps(report))
"""


def run_check(py: Path, packages: list[dict], compat: bool, only: set[str] | None) -> int:
    imports = [p["import_name"] for p in packages if not only or p["id"] in only]
    spec = {
        "imports": imports,
        "compat": compat and (not only or "compat" in only),
        "compat_src": str(COMPAT / "src"),
        "compat_ver": COMPAT_VERSION_PREFIX,
        "server": "hugpy_server" in imports,
    }
    proc = subprocess.run([str(py), "-c", _CHECK_SNIPPET, json.dumps(spec)],
                          capture_output=True, text=True, cwd=str(WORKSPACE))
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        print("check: interpreter run failed")
        return 1
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    failures = 0
    for name, where in report["imports"].items():
        editable = where.startswith(str(PY_ROOT))
        bad = where.startswith("ERROR") or not editable
        failures += bad
        print(f"  {'FAIL' if bad else 'ok  '} {name:16s} {where}")
    if report["compat"] is not None:
        c = report["compat"]
        failures += not c.get("ok")
        print(f"  {'ok  ' if c.get('ok') else 'FAIL'} abstract_hugpy_dev {c.get('version') or ''} "
              f"{c.get('where') or c.get('error')}")
    elif report["stale_monolith"]:
        failures += 1
        print(f"  FAIL abstract_hugpy_dev {report['stale_monolith']} is installed but --no-compat was "
              f"given: that is the retired monolith, uninstall it")
    if report["server"] is not None:
        bad = report["server"] != "ok"
        failures += bad
        print(f"  {'FAIL' if bad else 'ok  '} hugpy_server.wsgi_app + wiring import: {report['server']}")
    if spec["compat"] and not failures:
        exe = py.parent / ("abstract-hugpy-dev-check.exe" if os.name == "nt" else "abstract-hugpy-dev-check")
        cmd = [str(exe)] if exe.exists() else [str(py), "-m", "abstract_hugpy_dev._check"]
        print("+", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE))
        tail = [ln for ln in proc.stdout.splitlines()
                if not ln.startswith(("  SKIP", "[", "20"))][-4:]
        print("  " + "\n  ".join(tail))
        failures += proc.returncode != 0
    print("check:", "FAIL" if failures else "OK")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", type=Path, default=WORKSPACE / ".venv",
                    help="virtualenv to install into, created if missing (default: <workspace>/.venv)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used to create the venv when it does not exist")
    ap.add_argument("--extras", default="",
                    help="comma-separated extras of the hugpy meta package (server, worker, gpu-worker, all, ...)")
    ap.add_argument("--only", default="",
                    help="comma-separated package ids to install (platform, control, ..., meta, compat)")
    ap.add_argument("--no-compat", action="store_true", help="do not install the abstract_hugpy_dev alias shell")
    ap.add_argument("--check", action="store_true", help="verify the install afterwards (or alone with --no-install)")
    ap.add_argument("--no-install", action="store_true", help="skip pip; useful with --check")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--allow-system", action="store_true", help="permit a non-venv interpreter")
    ap.add_argument("--pip-arg", action="append", default=[],
                    help="extra argument for pip install (repeatable; e.g. --pip-arg=--no-index)")
    args = ap.parse_args(argv)

    packages = read_packages()
    extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    only = {o.strip() for o in args.only.split(",") if o.strip()} or None
    known = {p["id"] for p in packages} | {"compat"}
    if only and not only <= known:
        raise SystemExit(f"unknown package id(s): {sorted(only - known)}; known: {sorted(known)}")
    compat = not args.no_compat

    py = ensure_venv(args.venv, args.python, args.dry_run)
    if not args.dry_run and not args.allow_system and is_system_interpreter(py):
        raise SystemExit(f"{py} is not a virtualenv interpreter; refusing (see --allow-system)")

    if not args.no_install:
        specs = install_specs(packages, extras, compat, only)
        pip_install(py, specs, args.pip_arg, args.dry_run)
        print(f"installed {len(specs)} editable distribution(s) into {args.venv}")
    if args.check and not args.dry_run:
        return run_check(py, packages, compat, only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
