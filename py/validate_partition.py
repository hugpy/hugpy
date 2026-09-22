#!/usr/bin/env python3
"""Validate the monolith-to-distributions partition manifest.

Default mode checks the manifest itself: unique ids, known dependencies, every
declared source path exists (in the monolith or already at its destination),
every monolith file has exactly one owner, destinations never collide, and the
dependency graph is acyclic.

``--edges`` additionally scans the monolith and every extracted package for
actual Python imports and reports every edge the manifest forbids, grouped
by source package. That report is the extraction to-do list.

``--json PATH`` writes the edge report for tooling.
"""

from __future__ import annotations

import argparse
import sys
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
    reloc_path = manifest.source_root / "_relocations.json"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", action="store_true", help="report forbidden import edges")
    parser.add_argument("--package", help="restrict the edge report to one package id")
    parser.add_argument("--json", type=Path, help="write the full edge list as JSON")
    parser.add_argument("--all-edges", action="store_true", help="print allowed edges too")
    args = parser.parse_args()

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
