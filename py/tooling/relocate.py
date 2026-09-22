#!/usr/bin/env python3
"""Re-home an already-extracted module after a manifest correction.

    python py/tooling/relocate.py <monolith-relative-source-path> [--dry-run]

Use when partition.toml changes the owner or destination of a file that has
already been moved out of the monolith. The tool finds the file's current
location through ``_relocations.json``, moves it to the destination the
manifest now declares, updates the relocation map, and rewrites every
reference to the old new-name across the workspace (imports and dotted
strings). Relative imports inside the moved file are absolutized.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_package import RELOCATIONS_NAME, Rewriter, ensure_inits, known_monolith_modules, prune_empty_dirs  # noqa: E402
from partition_lib import PY_ROOT, load_manifest  # noqa: E402


def module_path(manifest, dotted: str) -> Path | None:
    top, *rest = dotted.split(".")
    for pkg in manifest.packages.values():
        if pkg.import_name == top:
            base = pkg.src_dir.joinpath(*rest)
            if base.is_dir():
                return base / "__init__.py"
            if base.with_suffix(".py").exists():
                return base.with_suffix(".py")
    return None


def workspace_py_files(manifest):
    yield from (p for p in manifest.source_root.rglob("*.py") if "__pycache__" not in p.parts)
    tests = manifest.source_root.parents[1] / "tests"
    if tests.exists():
        yield from tests.rglob("*.py")
    for pkg in manifest.packages.values():
        for base in (pkg.src_dir, pkg.root / "tests"):
            if base.exists():
                yield from (p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="monolith-relative path, e.g. imports/src/constants/paths.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    reloc_path = manifest.source_root / RELOCATIONS_NAME
    relocations = json.loads(reloc_path.read_text()) if reloc_path.exists() else {}
    old_module = manifest.old_module(args.source)
    if old_module is None:
        raise SystemExit("only .py sources can be relocated with this tool")
    current_new = relocations.get(old_module)
    target = manifest.new_module(args.source)
    landing = manifest.dest_path(args.source)
    if target is None or landing is None:
        raise SystemExit(f"{args.source} has no destination in the manifest")
    _, dest = landing

    if current_new is None:
        src = manifest.source_root / args.source
        if not src.exists():
            raise SystemExit(f"{args.source} is neither in the monolith nor in _relocations.json")
        print(f"{args.source} is still in the monolith; use extract_package.py instead")
        return 1
    if current_new == target:
        print(f"{args.source} already lives at {target}")
        return 0
    src = module_path(manifest, current_new)
    if src is None:
        raise SystemExit(f"cannot find current file for {current_new}")
    print(f"move {src.relative_to(PY_ROOT)} -> {dest.relative_to(PY_ROOT)}  ({current_new} -> {target})")
    if args.dry_run:
        return 0
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    for pkg in manifest.packages.values():
        if pkg.src_dir.exists():
            ensure_inits(pkg.src_dir)
            prune_empty_dirs(pkg.src_dir)

    relocations[old_module] = target
    reloc_path.write_text(json.dumps(relocations, indent=1, sort_keys=True), encoding="utf-8")

    # Rewrite references: old new-name -> target, as imports and dotted strings.
    pattern = re.compile(r"\b" + re.escape(current_new) + r"\b(?=[\s.,;:'\")\]]|$)")
    changed = 0
    for path in workspace_py_files(manifest):
        text = path.read_text(encoding="utf-8", errors="replace")
        if current_new not in text:
            continue
        new_text = pattern.sub(target, text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
    # Absolutize relative imports inside the moved file (relative to its ORIGINAL name).
    rewriter = Rewriter(manifest, relocations, known_monolith_modules(manifest, relocations))
    rewriter.rewrite_file(dest, old_module, args.source.endswith("__init__.py"), moved=True)
    print(f"rewrote {changed} files; relocation map updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
