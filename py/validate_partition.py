#!/usr/bin/env python3
"""Validate the monolith-to-distributions partition manifest.

Default mode checks the manifest itself: unique ids, known dependencies, every
declared source path exists (in the monolith or already at its destination),
every monolith file has exactly one owner, destinations never collide, and the
dependency graph is acyclic.

``--edges`` additionally scans the monolith and every extracted package for
actual Python imports and reports every edge the manifest forbids, grouped
by source package. That report is the extraction to-do list.

``--versions`` enforces the lockstep versioning contract (CONSISTENCY.md):
every ``[[package]]`` pyproject takes its version from the workspace git tag
via setuptools-scm (``dynamic = ["version"]``, no static ``version =``,
``[tool.setuptools_scm]`` rooted at the workspace with the ``0.0.0+unknown``
fallback, setuptools-scm in the build requirements) and no ``__init__.py``
under a package carries a version literal other than that fallback.
``--self-test`` proves the ``--versions`` checker against synthetic fixtures.

``--json PATH`` writes the edge report for tooling.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tooling"))

from partition_lib import (  # noqa: E402
    RETIRE_ID,
    dump_json,
    forbidden_edges,
    load_manifest,
    summarize_edges,
    workspace_edges,
)


def _cycle(graph: dict[str, set[str]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    trail: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            start = trail.index(node)
            return trail[start:] + [node]
        if node in visited:
            return None
        visiting.add(node)
        trail.append(node)
        for dependency in graph[node]:
            found = visit(dependency)
            if found:
                return found
        trail.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in graph:
        found = visit(node)
        if found:
            return found
    return None


def validate_manifest(manifest) -> list[str]:
    import json

    errors: list[str] = []
    # the monolith's generated map; after the tree was retired the compat shell
    # keeps the last copy under its generator inputs
    reloc_path = manifest.source_root / "_relocations.json"
    if not reloc_path.exists():
        reloc_path = (Path(__file__).resolve().parent / "compat" / "abstract_hugpy_dev"
                      / "tools" / "inputs" / "monolith_relocations.json")
    relocations = json.loads(reloc_path.read_text()) if reloc_path.exists() else {}
    ids = manifest.order
    if len(ids) != len(set(ids)):
        errors.append("package ids are not unique")
    known = set(ids)
    graph: dict[str, set[str]] = {}
    for pkg in manifest.packages.values():
        unknown = pkg.allowed - known
        if unknown:
            errors.append(f"{pkg.id}: unknown dependencies {sorted(unknown)}")
        graph[pkg.id] = pkg.allowed
        dests: dict[Path, str] = {}
        for src, dest in pkg.moves.items():
            exists_here = (manifest.source_root / src).exists()
            landing = manifest.dest_path(src)
            exists_there = landing is not None and landing[1].exists()
            already_moved = manifest.old_module(src) in relocations
            if not exists_here and not exists_there and not already_moved:
                errors.append(f"{pkg.id}: missing source path {src}")
            if landing is not None:
                prior = dests.get(landing[1])
                if prior is not None:
                    errors.append(
                        f"{pkg.id}: destination collision {landing[1]} <- {prior} and {src}"
                    )
                dests[landing[1]] = src
    for src in manifest.retired:
        if not (manifest.source_root / src).exists():
            # Retired paths may already have been removed; only warn.
            continue

    unowned = [
        rel
        for rel in manifest.iter_source_files()
        if manifest.owner_of(rel)[0] == "unowned"
        and not rel.endswith((".pyc",))
        and rel not in ("_relocations.json", "_relocations.py")
    ]
    if unowned:
        errors.append(f"unowned monolith files ({len(unowned)}): " + ", ".join(unowned[:25]))

    found_cycle = _cycle(graph)
    if found_cycle:
        errors.append("dependency cycle: " + " -> ".join(found_cycle))
    return errors


# ---------------------------------------------------------------------------
# --versions: the lockstep versioning contract (CONSISTENCY.md)
#
# One git-derived version for every in-tree hugpy-* distribution. A release is
# a workspace tag vX.Y.Z; setuptools-scm turns it into X.Y.Z (X.Y.Z.devN+gSHA
# between tags) at build time. Nothing below may carry a hand-written version.

PY_ROOT = Path(__file__).resolve().parent
WORKSPACE = PY_ROOT.parent
SCM_ROOT = "../../.."           # pyproject dir -> py/<layer>/<pkg> -> workspace
SCM_FALLBACK = "0.0.0+unknown"  # what a checkout without git metadata builds as
_VERSION_LITERAL = re.compile(r"""^\s*__version__\s*=\s*(['"])(?P<value>[^'"]*)\1""", re.M)
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _req_name(spec: str) -> str:
    """PEP 503-normalised project name of a requirement string."""
    m = _REQ_NAME.match(spec)
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else ""


def check_versions(
    packages: list[tuple[str, Path, str]],
    workspace: Path = WORKSPACE,
    scm_root: str = SCM_ROOT,
    fallback: str = SCM_FALLBACK,
) -> list[str]:
    """Return one message per violation of the lockstep contract.

    ``packages`` is ``[(distribution, package_root, import_name), ...]``: the
    ``[[package]]`` entries of the manifest (never ``[[external_package]]`` and
    never the compat shell, which is static-versioned and never published).
    Messages are prefixed with the offending file, workspace-relative.
    """
    errors: list[str] = []

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(workspace))
        except ValueError:
            return str(p)

    for distribution, root, import_name in packages:
        pyproject = root / "pyproject.toml"
        where = rel(pyproject)
        if not pyproject.exists():
            errors.append(f"{where}: missing (declared for {distribution})")
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"{where}: not valid TOML ({exc})")
            continue

        project = data.get("project") or {}
        name = project.get("name")
        if name != distribution:
            errors.append(f"{where}: [project] name = {name!r}, manifest says {distribution!r}")
        if "version" in project:
            errors.append(
                f"{where}: [project] has static version = {project['version']!r}; "
                f"the version is git-derived, use dynamic = [\"version\"]"
            )
        if "version" not in (project.get("dynamic") or []):
            errors.append(f"{where}: [project] dynamic must contain \"version\"")

        requires = ((data.get("build-system") or {}).get("requires")) or []
        if "setuptools-scm" not in {_req_name(r) for r in requires}:
            errors.append(f"{where}: [build-system] requires must include setuptools-scm")

        scm = (data.get("tool") or {}).get("setuptools_scm")
        if scm is None:
            errors.append(f"{where}: missing [tool.setuptools_scm]")
        else:
            got_root = scm.get("root")
            if got_root != scm_root:
                errors.append(f"{where}: [tool.setuptools_scm] root = {got_root!r}, expected {scm_root!r}")
            elif (root / got_root).resolve() != workspace.resolve():
                errors.append(
                    f"{where}: [tool.setuptools_scm] root resolves to "
                    f"{(root / got_root).resolve()}, not the workspace {workspace}"
                )
            got_fallback = scm.get("fallback_version")
            if got_fallback != fallback:
                errors.append(
                    f"{where}: [tool.setuptools_scm] fallback_version = {got_fallback!r}, "
                    f"expected {fallback!r}"
                )

        src_dir = root / "src" / import_name
        if not src_dir.is_dir():
            errors.append(f"{rel(src_dir)}: import package directory missing")
            continue
        inits = sorted(p for p in src_dir.rglob("__init__.py") if "__pycache__" not in p.parts)
        if not (src_dir / "__init__.py").exists():
            errors.append(f"{rel(src_dir / '__init__.py')}: missing")
        for init in inits:
            text = init.read_text(encoding="utf-8", errors="replace")
            for m in _VERSION_LITERAL.finditer(text):
                value = m.group("value")
                if value != fallback:
                    line = text.count("\n", 0, m.start()) + 1
                    errors.append(
                        f"{rel(init)}:{line}: __version__ = {value!r} literal; only the "
                        f"{fallback!r} fallback may be spelled out (read importlib.metadata)"
                    )
    return errors


def manifest_version_targets(manifest) -> list[tuple[str, Path, str]]:
    """``check_versions`` input for every ``[[package]]`` in cut order."""
    return [(pkg.distribution, pkg.root, pkg.import_name) for pkg in manifest.packages.values()]


def self_test() -> int:
    """Prove ``check_versions`` on synthetic fixtures: one clean, several broken."""
    import tempfile

    good_pyproject = """
[build-system]
requires = ["setuptools>=77", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"
[project]
name = "hugpy-demo"
dynamic = ["version"]
[tool.setuptools_scm]
root = "../../.."
fallback_version = "0.0.0+unknown"
"""
    good_init = (
        "try:\n    from importlib.metadata import version as _v\n"
        "    __version__ = _v('hugpy-demo')\nexcept Exception:\n"
        "    __version__ = \"0.0.0+unknown\"\n"
    )

    def make(ws: Path, layer: str, name: str, pyproject: str, init: str, sub_init: str = "") -> Path:
        root = ws / "py" / layer / name
        (root / "src" / name).mkdir(parents=True)
        (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        (root / "src" / name / "__init__.py").write_text(init, encoding="utf-8")
        if sub_init:
            (root / "src" / name / "sub").mkdir()
            (root / "src" / name / "sub" / "__init__.py").write_text(sub_init, encoding="utf-8")
        return root

    failures = 0
    with tempfile.TemporaryDirectory(prefix="validate-versions-") as tmp:
        ws = Path(tmp).resolve()
        clean = make(ws, "foundation", "hugpy_demo", good_pyproject, good_init)
        errs = check_versions([("hugpy-demo", clean, "hugpy_demo")], ws)
        if errs:
            failures += 1
            print("self-test FAIL: clean fixture reported", errs)
        else:
            print("self-test ok: clean fixture passes")

        cases = {
            "static version": (
                good_pyproject.replace('dynamic = ["version"]', 'version = "0.1.0"'), good_init, "",
                ["static version = '0.1.0'", 'dynamic must contain "version"'],
            ),
            "no setuptools-scm requirement": (
                good_pyproject.replace(', "setuptools-scm>=8"', ""), good_init, "",
                ["requires must include setuptools-scm"],
            ),
            "wrong scm root": (
                good_pyproject.replace('root = "../../.."', 'root = ".."'), good_init, "",
                ["root = '..'"],
            ),
            "missing fallback": (
                good_pyproject.replace('fallback_version = "0.0.0+unknown"\n', ""), good_init, "",
                ["fallback_version = None"],
            ),
            "missing scm table": (
                good_pyproject.split("[tool.setuptools_scm]")[0], good_init, "",
                ["missing [tool.setuptools_scm]"],
            ),
            "version literal in __init__": (
                good_pyproject, good_init.replace('"0.0.0+unknown"', '"0.1.0"'), "",
                ["__version__ = '0.1.0' literal"],
            ),
            "version literal in a subpackage": (
                good_pyproject, good_init, "__version__ = '1.2.3'\n",
                ["sub/__init__.py:1: __version__ = '1.2.3' literal"],
            ),
        }
        for i, (label, (pyproject, init, sub_init, expect)) in enumerate(cases.items()):
            root = make(ws, "foundation", f"hugpy_bad{i}", pyproject, init, sub_init)
            errs = check_versions([("hugpy-demo", root, f"hugpy_bad{i}")], ws)
            missing = [e for e in expect if not any(e in got for got in errs)]
            if missing:
                failures += 1
                print(f"self-test FAIL: {label}: expected {missing}, got {errs}")
            else:
                print(f"self-test ok: {label} -> {len(errs)} violation(s)")
    print("self-test:", "FAIL" if failures else "OK")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", action="store_true", help="report forbidden import edges")
    parser.add_argument("--package", help="restrict the edge report to one package id")
    parser.add_argument("--json", type=Path, help="write the full edge list as JSON")
    parser.add_argument("--all-edges", action="store_true", help="print allowed edges too")
    parser.add_argument("--versions", action="store_true",
                        help="enforce the lockstep git-derived versioning contract (CONSISTENCY.md)")
    parser.add_argument("--self-test", action="store_true",
                        help="prove the --versions checker against synthetic fixtures and exit")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    manifest = load_manifest()
    errors = validate_manifest(manifest)
    if errors:
        print("partition invalid:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    moves = sum(len(p.moves) for p in manifest.packages.values())
    print(
        f"partition valid: {len(manifest.packages)} packages, {moves} move entries, "
        f"{len(manifest.retired)} retirements, acyclic dependencies"
    )

    if args.versions:
        targets = manifest_version_targets(manifest)
        errors = check_versions(targets)
        if errors:
            print("versioning contract violated:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print(
            f"versions valid: {len(targets)} distributions git-derived via setuptools-scm "
            f"(root {SCM_ROOT!r}, fallback {SCM_FALLBACK!r}), no version literals"
        )

    if not args.edges and not args.json:
        return 0

    edges = workspace_edges(manifest)
    bad = forbidden_edges(manifest, edges)
    if args.package:
        edges = [e for e in edges if e.src_pkg == args.package]
        bad = [e for e in bad if e.src_pkg == args.package]
    if args.json:
        dump_json(
            {
                "summary": summarize_edges(edges),
                "forbidden": [e.__dict__ for e in bad],
            },
            args.json,
        )
    if args.all_edges:
        print("\nall cross-package edges (src -> dst : count)")
        for src, row in sorted(summarize_edges(edges).items()):
            for dst, count in sorted(row.items(), key=lambda kv: -kv[1]):
                print(f"  {src:10} -> {dst:10} {count}")
    print(f"\nforbidden edges: {len(bad)}")
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in bad:
        grouped[e.src_pkg][e.dst_pkg].append(e)
    for src in manifest.order:
        if src not in grouped:
            continue
        total = sum(len(v) for v in grouped[src].values())
        print(f"\n[{src}] {total} forbidden")
        for dst, rows in sorted(grouped[src].items(), key=lambda kv: -len(kv[1])):
            files = sorted({r.src_file for r in rows})
            print(f"  -> {dst:10} {len(rows):4}  in {len(files)} files")
            if args.package:
                for r in rows:
                    print(f"       {r.src_file}:{r.lineno}  {r.dst_module}")
    return 0 if not (args.package and bad) else 2


if __name__ == "__main__":
    raise SystemExit(main())
