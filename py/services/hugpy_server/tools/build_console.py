#!/usr/bin/env python3
"""Build or fetch the React consoles into ``hugpy_server/console_dist/``.

Driven by ``console_manifest.json`` (next to ``pyproject.toml``), which maps each
mount to its React package directory, npm name, pinned version and copy target
under ``console_dist/``. Stdlib only (Python >= 3.10).

    python tools/build_console.py --from-source            # npm ci + build in react/<pkg>
    python tools/build_console.py --from-npm               # npm pack <name>@<pinned version>
    python tools/build_console.py --from-source --only /fleet
    python tools/build_console.py --check                  # what console_dist holds

Only the selected mounts are replaced; every other mount under console_dist/ is
left untouched. The root mount ("/") never deletes the sub-mount directories
(fleet/, media/, video/), and the sub-mount directories that ui's own postbuild
copies into ui/dist/ are not copied over them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
DEFAULT_MANIFEST = PKG_ROOT / "console_manifest.json"
REQUIRED_KEYS = ("package_dir", "npm", "version", "target")


class BuildError(RuntimeError):
    pass


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(data)
    return data


def validate_manifest(data: dict) -> None:
    """Raise ValueError unless ``data`` has the documented shape."""
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    for key in ("react_root", "console_dist", "mounts"):
        if key not in data:
            raise ValueError(f"manifest is missing {key!r}")
    mounts = data["mounts"]
    if not isinstance(mounts, dict) or not mounts:
        raise ValueError("manifest 'mounts' must be a non-empty object")
    targets = set()
    for mount, entry in mounts.items():
        if not mount.startswith("/"):
            raise ValueError(f"mount {mount!r} must start with '/'")
        missing = [k for k in REQUIRED_KEYS if k not in entry]
        if missing:
            raise ValueError(f"mount {mount!r} is missing {missing}")
        target = entry["target"]
        expected = mount.strip("/")
        if target != expected:
            raise ValueError(f"mount {mount!r} must copy to {expected!r}, not {target!r}")
        if not entry["npm"].startswith("@hugpy/"):
            raise ValueError(f"mount {mount!r}: npm name must be in the @hugpy scope")
        if target in targets:
            raise ValueError(f"duplicate target {target!r}")
        targets.add(target)
    if "" not in targets:
        raise ValueError("manifest must define the root mount '/'")


def normalize_mount(name: str) -> str:
    return "/" + name.strip().strip("/")


def select_mounts(manifest: dict, only: list[str] | None) -> list[str]:
    mounts = list(manifest["mounts"])
    if not only:
        return mounts
    chosen = []
    for raw in only:
        m = normalize_mount(raw)
        if m not in manifest["mounts"]:
            raise BuildError(f"unknown mount {raw!r}; known: {', '.join(mounts)}")
        if m not in chosen:
            chosen.append(m)
    return chosen


def sub_targets(manifest: dict) -> set[str]:
    """Top-level console_dist entries owned by the non-root mounts."""
    return {e["target"] for e in manifest["mounts"].values() if e["target"]}


# ---------------------------------------------------------------- check

def check(manifest: dict, console_dist: Path) -> list[dict]:
    owned = sub_targets(manifest)
    rows = []
    for mount, entry in manifest["mounts"].items():
        target = entry["target"]
        root = console_dist / target if target else console_dist
        files = 0
        if root.is_dir():
            for dirpath, dirnames, filenames in os.walk(root):
                if not target and Path(dirpath) == root:
                    dirnames[:] = [d for d in dirnames if d not in owned]
                files += len(filenames)
        rows.append({
            "mount": mount,
            "target": (target + "/") if target else "./",
            "npm": entry["npm"],
            "version": entry["version"],
            "index_html": (root / "index.html").is_file(),
            "files": files,
        })
    return rows


def format_check(rows: list[dict], console_dist: Path) -> str:
    lines = [f"console_dist: {console_dist}"]
    for r in rows:
        state = "present" if r["index_html"] else "MISSING"
        lines.append(f"{r['mount']:<8} {r['target']:<8} index.html: {state:<8} "
                     f"files: {r['files']:<5} {r['npm']}@{r['version']}")
    ok = sum(r["index_html"] for r in rows)
    lines.append(f"{ok}/{len(rows)} mounts have index.html")
    return "\n".join(lines)


# ---------------------------------------------------------------- copy

def install_dist(manifest: dict, mount: str, src: Path, console_dist: Path) -> Path:
    """Replace ONE mount's files under console_dist with the contents of ``src``."""
    if not (src / "index.html").is_file():
        raise BuildError(f"{mount}: {src} has no index.html; refusing to install it")
    target = manifest["mounts"][mount]["target"]
    console_dist.mkdir(parents=True, exist_ok=True)
    if target:
        dest = console_dist / target
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return dest
    # Root mount: clear and copy everything except the sub-mount directories.
    owned = sub_targets(manifest)
    for child in console_dist.iterdir():
        if child.name in owned:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in src.iterdir():
        if child.name in owned:
            continue
        if child.is_dir():
            shutil.copytree(child, console_dist / child.name)
        else:
            shutil.copy2(child, console_dist / child.name)
    return console_dist


# ---------------------------------------------------------------- modes

def _npm() -> str:
    npm = shutil.which("npm")
    if not npm:
        raise BuildError("npm not found on PATH")
    return npm


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    print(f"+ (cd {cwd} && {' '.join(cmd)})", flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True)


def build_from_source(manifest: dict, mount: str, react_root: Path, console_dist: Path,
                      skip_install: bool = False) -> Path:
    npm = _npm()
    pkg_dir = react_root / manifest["mounts"][mount]["package_dir"]
    if not (pkg_dir / "package.json").is_file():
        raise BuildError(f"{mount}: no package.json in {pkg_dir}")
    # react/ is a workspace root without a root lockfile: install per package.
    if not skip_install:
        verb = "ci" if (pkg_dir / "package-lock.json").is_file() else "install"
        _run([npm, verb, "--workspaces=false"], pkg_dir)
    _run([npm, "run", "build"], pkg_dir)
    return install_dist(manifest, mount, pkg_dir / "dist", console_dist)


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise BuildError(f"refusing link in tarball: {member.name}")
        if not (dest / member.name).resolve().is_relative_to(dest):
            raise BuildError(f"refusing path outside extract dir: {member.name}")
    if sys.version_info >= (3, 12):
        tar.extractall(dest, filter="data")
    else:
        tar.extractall(dest)


def build_from_npm(manifest: dict, mount: str, console_dist: Path) -> Path:
    npm = _npm()
    entry = manifest["mounts"][mount]
    spec = f"{entry['npm']}@{entry['version']}"
    with tempfile.TemporaryDirectory(prefix="hugpy-console-") as tmp:
        tmpdir = Path(tmp)
        out = subprocess.run([npm, "pack", spec, "--json", "--pack-destination", str(tmpdir)],
                             cwd=tmpdir, check=True, capture_output=True, text=True).stdout
        filename = json.loads(out)[0]["filename"]
        with tarfile.open(tmpdir / filename) as tar:
            _safe_extract(tar, tmpdir / "x")
        dist = tmpdir / "x" / "package" / "dist"
        if not (dist / "index.html").is_file():
            raise BuildError(f"{mount}: {spec} ships no built dist/ (no dist/index.html in the "
                             "tarball); use --from-source or pin a release that includes dist")
        return install_dist(manifest, mount, dist, console_dist)


# ---------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-source", action="store_true",
                      help="npm ci/install + npm run build in react/<pkg>, copy dist/")
    mode.add_argument("--from-npm", action="store_true",
                      help="npm pack the pinned published version and copy its dist/")
    mode.add_argument("--check", action="store_true",
                      help="report which mounts console_dist holds")
    ap.add_argument("--only", action="append", metavar="MOUNT",
                    help="limit to a mount (/, /fleet, /media, /video); repeatable")
    ap.add_argument("--skip-install", action="store_true",
                    help="--from-source: reuse existing node_modules")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--console-dist", type=Path,
                    help="override the manifest's console_dist directory")
    ap.add_argument("--react-root", type=Path,
                    help="override the manifest's react_root directory")
    args = ap.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        base = args.manifest.resolve().parent
        console_dist = (args.console_dist or base / manifest["console_dist"]).resolve()
        react_root = (args.react_root or base / manifest["react_root"]).resolve()
        mounts = select_mounts(manifest, args.only)

        if args.check:
            rows = [r for r in check(manifest, console_dist) if r["mount"] in mounts]
            print(format_check(rows, console_dist))
            return 0 if all(r["index_html"] for r in rows) else 1

        for mount in mounts:
            if args.from_source:
                dest = build_from_source(manifest, mount, react_root, console_dist,
                                         skip_install=args.skip_install)
            else:
                dest = build_from_npm(manifest, mount, console_dist)
            print(f"[build_console] {mount} -> {dest}", flush=True)
    except (BuildError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[build_console] error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
