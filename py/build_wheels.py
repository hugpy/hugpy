#!/usr/bin/env python3
"""Build every workspace distribution and (optionally) publish the wheels to a
central pip index directory.

    python py/build_wheels.py [--out DIR] [--expect X.Y.Z | --expect-tag vX.Y.Z]
                              [--publish INDEX_DIR] [--no-sdist] [--only a,b,c]

One build path for every consumer (CONSISTENCY.md):

  * CI ``pypi-publish.yml`` runs it with ``--expect-tag "$GITHUB_REF_NAME"`` so
    every artifact provably carries the tag's version before PyPI sees it.
  * A central box runs it from a checkout of the release tag with
    ``--publish "$(hugpy-pkg-index-dir)"`` (or the path central prints in
    ``/api/health``) to serve the release from its own PEP 503 index at
    ``/api/llm/pip/simple/`` — the same channel workers already pull model
    files from. Workers then converge from central with no PyPI involvement.
  * A dev box runs it bare to get a wheelhouse for an offline
    ``local_install.sh --pip-arg=--find-links=...``.

Every ``[[package]]`` in ``py/partition.toml`` is built in cut order with
``python -m build`` (sdist + wheel). The versions setuptools-scm derived are read
back from the artifact filenames and must all be identical; with ``--expect``
or ``--expect-tag`` they must also equal that version (the ``v`` prefix is
stripped; ``vX.Y.Z`` with an optional aN/bN/rcN/.postN suffix). Any mismatch
exits 1 and nothing is published.

``--publish DIR`` copies the verified WHEELS flat into DIR — the layout
``hugpy_fleet.central.workers.pkg_index_dir`` and the
``/llm/pip/simple/<project>/`` route expect; workers install wheels only and
sdists stay in ``--out`` for PyPI. Builds are reproducible (SOURCE_DATE_EPOCH
= HEAD's commit time), so re-running the publish is a no-op; a same-named
wheel with different bytes is refused (an artifact for a version is
immutable; delete it by hand first). Stdlib only apart from the ``build``
package the subprocess needs.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
MANIFEST = WORKSPACE / "py" / "partition.toml"
_TAG_RE = re.compile(r"^v?(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?)$")


CONSOLE_TOOL = WORKSPACE / "py" / "services" / "hugpy_server" / "tools" / "build_console.py"
CONSOLE_DIST = WORKSPACE / "py" / "services" / "hugpy_server" / "src" / "hugpy_server" / "console_dist"
REACT_UI_SRC = WORKSPACE / "react" / "ui" / "src"


def _console_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hugpy_build_console", CONSOLE_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def console_stale_reason(src: Path = REACT_UI_SRC, console_dist: Path = CONSOLE_DIST,
                         *, mtime_fallback: bool = True) -> str | None:
    """Why the tracked console bundle does not match react/ui/src (None = it
    does, or there is no React tree to compare). The pipeline never runs npm, so
    a UI edit without a rebuild must fail the wheel build, never ship silently."""
    if not CONSOLE_TOOL.is_file():
        return None
    return _console_tool().console_staleness(src, console_dist, mtime_fallback=mtime_fallback)


def packages(only: set[str] | None = None) -> list[dict]:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for p in manifest["package"]:
        if only and p["id"] not in only and p["distribution"] not in only:
            continue
        out.append(p)
    return out


def file_stem(distribution: str) -> str:
    """PEP 427/625 filename stem for ``distribution`` (``hugpy-fleet`` -> ``hugpy_fleet-``)."""
    return re.sub(r"[-_.]+", "_", distribution).lower() + "-"


def artifact_version(distribution: str, path: Path) -> str | None:
    """The version encoded in a built artifact's filename, or None if the file
    is not an artifact of ``distribution``."""
    stem = file_stem(distribution)
    name = path.name
    if not name.lower().startswith(stem):
        return None
    rest = name[len(stem):]
    if name.endswith(".whl"):
        return rest.split("-", 1)[0]
    if name.endswith(".tar.gz"):
        return rest[: -len(".tar.gz")]
    return None


def source_date_epoch() -> str | None:
    """The committer timestamp of HEAD, so two builds of one commit produce
    byte-identical artifacts (wheel and setuptools honor SOURCE_DATE_EPOCH):
    that is what lets ``--publish`` be re-run safely and lets a central-served
    wheel be compared with CI's. An explicit SOURCE_DATE_EPOCH wins."""
    if os.environ.get("SOURCE_DATE_EPOCH"):
        return os.environ["SOURCE_DATE_EPOCH"]
    try:
        return subprocess.run(["git", "-C", str(WORKSPACE), "log", "-1", "--format=%ct"],
                              check=True, capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_one(pkg_dir: Path, out: Path, sdist: bool, python: str, epoch: str | None) -> None:
    cmd = [python, "-m", "build", "--wheel", "--outdir", str(out), str(pkg_dir)]
    if sdist:
        cmd.insert(3, "--sdist")
    env = dict(os.environ)
    if epoch:
        env["SOURCE_DATE_EPOCH"] = epoch
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   text=True, env=env)


def verify(out: Path, pkgs: list[dict], expect: str | None, sdist: bool,
           strict: bool = True) -> tuple[str, list[Path]]:
    """Every package has exactly one wheel (and one sdist when ``sdist``), all at
    one version, equal to ``expect`` when given. With ``strict`` a local
    version (``+gSHA`` between tags, ``+unknown`` without git metadata) is
    refused: central never advertises one to the fleet and PyPI rejects it.
    Returns (version, files)."""
    files = sorted(p for p in out.iterdir() if p.is_file())
    problems: list[str] = []
    versions: set[str] = set()
    picked: list[Path] = []
    for p in pkgs:
        dist = p["distribution"]
        wheels = [f for f in files if f.suffix == ".whl" and artifact_version(dist, f)]
        sdists = [f for f in files if f.name.endswith(".tar.gz") and artifact_version(dist, f)]
        want = wheels + (sdists if sdist else [])
        if len(wheels) != 1 or (sdist and len(sdists) != 1):
            problems.append(f"{dist}: expected one wheel{' and one sdist' if sdist else ''}, "
                            f"found {[f.name for f in wheels + sdists]}")
            continue
        for f in want:
            v = artifact_version(dist, f)
            versions.add(v)
            picked.append(f)
            if expect and v != expect:
                problems.append(f"{f.name}: version {v} != expected {expect}")
    if len(versions) > 1:
        problems.append(f"lockstep broken: artifacts carry {sorted(versions)}")
    local = sorted(v for v in versions if "+" in v)
    if local and strict:
        problems.append(f"local version {local[0]}: not a release tag checkout"
                        + (" and no git metadata at build time (clone with history)"
                           if "+unknown" in local[0] else "")
                        + " — a wheelhouse only; central will not advertise it and PyPI rejects it")
    if problems:
        raise SystemExit("build_wheels: artifacts do not verify:\n  " + "\n  ".join(problems))
    return next(iter(versions)), picked


def publish(files: list[Path], index_dir: Path) -> list[Path]:
    """Copy the wheels among ``files`` into ``index_dir``; skip byte-identical
    ones, refuse differing ones. Returns the newly copied paths."""
    index_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for f in (f for f in files if f.suffix == ".whl"):
        dst = index_dir / f.name
        if dst.exists():
            if filecmp.cmp(f, dst, shallow=False):
                continue  # identical bytes already served
            raise SystemExit(f"build_wheels: {dst} exists with different bytes; an artifact "
                             f"for a version is immutable — remove it by hand to replace it")
        shutil.copy2(f, dst)
        copied.append(dst)
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(WORKSPACE / "dist"), help="artifact dir (default: ./dist, cleared first)")
    ap.add_argument("--expect", help="every artifact must be exactly this version")
    ap.add_argument("--expect-tag", help="like --expect but from a vX.Y.Z tag name")
    ap.add_argument("--publish", metavar="INDEX_DIR", help="copy the verified artifacts flat into central's pip index dir")
    ap.add_argument("--no-sdist", action="store_true", help="wheels only")
    ap.add_argument("--only", help="comma-separated package ids or distribution names")
    ap.add_argument("--python", default=sys.executable, help="interpreter with the `build` package")
    args = ap.parse_args(argv)

    expect = args.expect
    if args.expect_tag:
        m = _TAG_RE.match(args.expect_tag.strip())
        if not m:
            ap.error(f"--expect-tag {args.expect_tag!r} is not vX.Y.Z (optional aN/bN/rcN/.postN)")
        expect = m.group(1)
    if expect:
        expect = expect.lstrip("v")

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    pkgs = packages(only)
    if not pkgs:
        ap.error("no packages selected")

    if any(p["distribution"] == "hugpy-server" for p in pkgs):
        stale = console_stale_reason()
        if stale:
            raise SystemExit(f"build_wheels: refusing to build hugpy-server: {stale}")

    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sdist = not args.no_sdist

    epoch = source_date_epoch()
    print(f"build_wheels: {len(pkgs)} distribution(s) -> {out}" + (f" (expect {expect})" if expect else "")
          + (f" (SOURCE_DATE_EPOCH={epoch})" if epoch else ""))
    for p in pkgs:
        pkg_dir = WORKSPACE / "py" / p["destination"]
        try:
            build_one(pkg_dir, out, sdist, args.python, epoch)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(exc.stderr or "")
            raise SystemExit(f"build_wheels: building {p['distribution']} ({pkg_dir}) failed (rc {exc.returncode})")
        print(f"  built {p['distribution']}")

    version, files = verify(out, pkgs, expect, sdist, strict=bool(expect or args.publish))
    for f in files:
        print(f"  ok {f.name}")
    print(f"build_wheels: {len(pkgs)} distribution(s) at {version}" + (f", matching {expect}" if expect else ""))
    if "+" in version:
        print("build_wheels: note: local (untagged) version — usable as a --find-links wheelhouse, "
              "never published (central advertises release tags only)")

    if args.publish:
        index_dir = Path(args.publish).resolve()
        wheels = [f for f in files if f.suffix == ".whl"]
        copied = publish(files, index_dir)
        print(f"build_wheels: published {len(copied)} new wheel(s) ({len(wheels) - len(copied)} already present) -> {index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
