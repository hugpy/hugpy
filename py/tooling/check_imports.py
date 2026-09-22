#!/usr/bin/env python3
"""Report dangling imports: ecosystem/monolith modules that cannot be found.

    python py/tooling/check_imports.py [package_id ...]

Statically lists every ``import``/``from ... import`` in the given packages
(default: all extracted packages plus the monolith) whose target is a Hugpy
ecosystem module or a monolith module, and checks that the module resolves
with ``importlib.util.find_spec``. ``from mod import name`` is verified by
confirming ``name`` is either a submodule or a top-level binding in ``mod``'s
source. Run inside the workspace venv with the monolith on PYTHONPATH when
transitional imports are still expected.
"""

from __future__ import annotations

import ast
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from partition_lib import PY_ROOT, imported_modules, load_manifest  # noqa: E402


MANIFEST = load_manifest()


def _static_path(module: str) -> Path | None:
    """Locate a module file without importing anything."""
    parts = module.split(".")
    top, rest = parts[0], parts[1:]
    if top == MANIFEST.monolith_import:
        base = MANIFEST.source_root
    else:
        pkg = next((p for p in MANIFEST.packages.values() if p.import_name == top), None)
        if pkg is None:
            return None
        base = pkg.src_dir
    target = base.joinpath(*rest) if rest else base
    if target.is_dir() and (target / "__init__.py").exists():
        return target / "__init__.py"
    if target.with_suffix(".py").exists() and rest:
        return target.with_suffix(".py")
    return None


@lru_cache(maxsize=None)
def spec_exists(module: str) -> bool:
    return _static_path(module) is not None


@lru_cache(maxsize=None)
def bindings(module: str) -> frozenset[str]:
    origin = _static_path(module)
    if origin is None:
        return frozenset()
    try:
        tree = ast.parse(origin.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return frozenset()
    names: set[str] = set()
    has_star = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    has_star = True
                else:
                    names.add(a.asname or a.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)
        elif isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            has_star = True
    if has_star:
        names.add("*")
    # A module-level __getattr__ (PEP 562 lazy exports) can serve any name.
    if any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body):
        names.add("*")
    return frozenset(names)


def check(base: Path, top: str, manifest) -> list[str]:
    problems = []
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(base).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            problems.append(f"{path}: syntax error {exc}")
            continue
        modparts = [top, *rel[:-3].split("/")]
        if modparts[-1] == "__init__":
            modparts = modparts[:-1]
        is_pkg = rel.endswith("__init__.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if manifest.package_for_module(a.name) and not spec_exists(a.name):
                        problems.append(f"{path}:{node.lineno} import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    cut = len(modparts) - node.level + (1 if is_pkg else 0)
                    mod = ".".join(modparts[:cut] + ([node.module] if node.module else []))
                else:
                    mod = node.module or ""
                if not manifest.package_for_module(mod):
                    continue
                if not spec_exists(mod):
                    problems.append(f"{path}:{node.lineno} from {mod} import ...  (module missing)")
                    continue
                names = bindings(mod)
                for a in node.names:
                    if a.name == "*":
                        continue
                    if a.name in names or "*" in names or spec_exists(f"{mod}.{a.name}") or a.name in ("__file__", "__path__", "__name__", "__doc__", "__version__"):
                        continue
                    problems.append(f"{path}:{node.lineno} from {mod} import {a.name}  (name missing)")
    return problems


def main() -> int:
    manifest = MANIFEST
    wanted = sys.argv[1:] or [*manifest.order, "monolith"]
    total = 0
    for pid in wanted:
        if pid == "monolith":
            base, top = manifest.source_root, manifest.monolith_import
            tests = manifest.source_root.parents[1] / "tests"
        else:
            pkg = manifest.packages[pid]
            base, top = pkg.src_dir, pkg.import_name
            tests = pkg.root / "tests"
        problems = check(base, top, manifest)
        if tests.exists():
            problems += check(tests, "tests", manifest)
        total += len(problems)
        print(f"[{pid}] {len(problems)} dangling")
        for p in problems:
            print("   ", p)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
