#!/usr/bin/env python3
"""Replace ``from X import *`` in a module with explicit imports.

    python py/tooling/explicit_imports.py FILE [FILE ...] [--write] [--report]

For each file: every name that is *used* but not *bound* locally (and is not a
builtin) must come from one of the wildcard imports. The tool resolves where
each such name is actually defined by walking the wildcard chain statically
(assignments, defs, classes, explicit imports, nested star-imports) and, when
the module can be imported, cross-checking against the runtime namespace so
``obj.__module__`` can pinpoint the defining module of functions and classes.

Resolved names are grouped per origin module and emitted as explicit import
statements at the position of the first wildcard import. Origins that live in
the monolith are rewritten to their relocated package module when
``_relocations.json`` knows about them. Names that cannot be resolved are
listed in a ``# TODO(partition): unresolved`` comment and the wildcard line is
kept, so the file keeps working while a human finishes the job.

Environment: PYTHONPATH must let the interpreter import the monolith and the
extracted packages (the workspace ``.venv`` with everything installed -e).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from partition_lib import PY_ROOT, load_manifest  # noqa: E402

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__loader__", "__package__", "__path__", "__builtins__", "__all__", "__version__", "__annotations__", "__dict__", "__class__"}
STDLIB = set(sys.stdlib_module_names)


# ---------------------------------------------------------------------------
# Static analysis helpers
# ---------------------------------------------------------------------------

class _Binder(ast.NodeVisitor):
    """Collect names bound anywhere in the module (approximate scope-agnostic)."""

    def __init__(self):
        self.bound: set[str] = set()
        self.used: set[str] = set()
        self.star_imports: list[ast.ImportFrom] = []

    def visit_Import(self, node):
        for a in node.names:
            self.bound.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            if a.name == "*":
                self.star_imports.append(node)
            else:
                self.bound.add(a.asname or a.name)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)
        for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.bound.add(arg.arg)
        if node.args.vararg:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.bound.add(node.args.kwarg.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.bound.add(arg.arg)
        if node.args.vararg:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.bound.add(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)
        else:
            self.used.add(node.id)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.bound.update(node.names)

    def visit_Nonlocal(self, node):
        self.bound.update(node.names)

    def visit_MatchAs(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node):
        if node.name:
            self.bound.add(node.name)


@dataclass
class Origin:
    module: str  # dotted module to import from ("" means `import name`)
    name: str  # the name inside that module
    kind: str  # "from" | "import"


@dataclass
class ModuleInfo:
    dotted: str
    path: Path | None
    tree: ast.Module | None
    all_names: set[str] | None = None
    defined: dict[str, Origin] = field(default_factory=dict)
    stars: list[str] = field(default_factory=list)


class Resolver:
    def __init__(self, manifest, relocations: dict[str, str]):
        self.manifest = manifest
        self.relocations = relocations
        self.cache: dict[str, ModuleInfo] = {}

    # -- module location ---------------------------------------------------
    def relocate(self, dotted: str) -> str:
        if dotted in self.relocations:
            return self.relocations[dotted]
        parts = dotted.split(".")
        for cut in range(len(parts) - 1, 1, -1):
            head = ".".join(parts[:cut])
            if head in self.relocations:
                return self.relocations[head] + "." + ".".join(parts[cut:])
        return dotted

    def locate(self, dotted: str) -> Path | None:
        try:
            spec = importlib.util.find_spec(dotted)
        except (ModuleNotFoundError, ValueError, ImportError):
            return None
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return None
        return Path(spec.origin)

    def info(self, dotted: str) -> ModuleInfo:
        dotted = self.relocate(dotted)
        if dotted in self.cache:
            return self.cache[dotted]
        path = self.locate(dotted)
        tree = None
        if path is not None:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                tree = None
        mi = ModuleInfo(dotted, path, tree)
        self.cache[dotted] = mi
        if tree is None:
            return mi
        is_pkg = path.name == "__init__.py"
        for node in tree.body:
            self._collect(node, mi, dotted, is_pkg)
        return mi

    def _abs(self, node: ast.ImportFrom, dotted: str, is_pkg: bool) -> str:
        if not node.level:
            return node.module or ""
        parts = dotted.split(".")
        cut = len(parts) - node.level + (1 if is_pkg else 0)
        base = parts[:max(cut, 0)]
        return ".".join(base + ([node.module] if node.module else []))

    def _collect(self, node, mi: ModuleInfo, dotted: str, is_pkg: bool):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        mi.defined.setdefault(n.id, Origin(dotted, n.id, "from"))
                        if n.id == "__all__":
                            mi.all_names = self._literal_all(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            mi.defined.setdefault(node.target.id, Origin(dotted, node.target.id, "from"))
        elif isinstance(node, (ast.AnnAssign,)) and isinstance(node.target, ast.Name):
            mi.defined.setdefault(node.target.id, Origin(dotted, node.target.id, "from"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mi.defined.setdefault(node.name, Origin(dotted, node.name, "from"))
        elif isinstance(node, ast.Import):
            for a in node.names:
                bound = (a.asname or a.name).split(".")[0]
                if a.asname:
                    mi.defined.setdefault(bound, Origin(a.name, "", "import_as"))
                else:
                    mi.defined.setdefault(bound, Origin(bound, "", "import"))
        elif isinstance(node, ast.ImportFrom):
            target = self._abs(node, dotted, is_pkg)
            for a in node.names:
                if a.name == "*":
                    mi.stars.append(target)
                else:
                    bound = a.asname or a.name
                    mi.defined.setdefault(bound, Origin(target, a.name, "from"))
        elif isinstance(node, (ast.Try,)):
            for sub in node.body + [h for hh in node.handlers for h in hh.body] + node.orelse + node.finalbody:
                self._collect(sub, mi, dotted, is_pkg)
        elif isinstance(node, ast.If):
            for sub in node.body + node.orelse:
                self._collect(sub, mi, dotted, is_pkg)

    @staticmethod
    def _literal_all(value) -> set[str] | None:
        try:
            names = ast.literal_eval(value)
            return {str(n) for n in names}
        except Exception:
            return None

    # -- resolution --------------------------------------------------------
    def resolve(self, name: str, star_module: str, seen: set[str] | None = None) -> Origin | None:
        """Where does ``name`` come from when doing ``from star_module import *``?"""
        seen = seen or set()
        dotted = self.relocate(star_module)
        if dotted in seen:
            return None
        seen.add(dotted)
        top = dotted.split(".")[0]
        if top in STDLIB or self.manifest.package_for_module(dotted) is None:
            # Foreign module: trust only the runtime namespace (and its __all__).
            return self._runtime_only(name, dotted, respect_all=True)
        mi = self.info(dotted)
        if mi.tree is None:
            return self._runtime_only(name, dotted)
        exported = mi.all_names
        if exported is not None and name not in exported:
            return None
        if name in mi.defined:
            origin = mi.defined[name]
            return self._canonicalize(name, origin, dotted)
        for star in mi.stars:
            found = self.resolve(name, star, seen)
            if found:
                return found
        return None

    def _canonicalize(self, name: str, origin: Origin, via: str) -> Origin:
        """Prefer the module that *defines* the object over re-exporters."""
        if origin.kind in ("import", "import_as"):
            return origin
        if origin.module != via:
            # explicit re-export: chase to the true home when it is ours
            deeper = self._chase(origin.name, origin.module, {via})
            if deeper:
                return deeper
            return Origin(self.relocate(origin.module), origin.name, "from")
        return Origin(self.relocate(via), name, "from")

    def _chase(self, name: str, module: str, seen: set[str]) -> Origin | None:
        dotted = self.relocate(module)
        top = dotted.split(".")[0]
        if top in STDLIB or self.manifest.package_for_module(dotted) is None:
            return Origin(dotted, name, "from")
        if dotted in seen:
            return None
        seen.add(dotted)
        mi = self.info(dotted)
        if mi.tree is None:
            return Origin(dotted, name, "from")
        if name in mi.defined:
            origin = mi.defined[name]
            if origin.kind in ("import", "import_as"):
                return origin
            if origin.module != dotted:
                return self._chase(origin.name, origin.module, seen) or Origin(self.relocate(origin.module), origin.name, "from")
            return Origin(dotted, name, "from")
        for star in mi.stars:
            found = self._chase(name, star, seen)
            if found:
                return found
        return None

    def _runtime_only(self, name: str, dotted: str, respect_all: bool = False) -> Origin | None:
        try:
            mod = importlib.import_module(dotted)
        except Exception:
            return None
        exported = getattr(mod, "__all__", None)
        if respect_all and exported is not None and name not in exported:
            return None
        if respect_all and exported is None and name.startswith("_"):
            return None
        if not hasattr(mod, name):
            return None
        obj = getattr(mod, name)
        if respect_all:
            return Origin(dotted, name, "from")
        home = getattr(obj, "__module__", None)
        if isinstance(home, str) and home != dotted and self.manifest.package_for_module(home) is not None:
            return Origin(self.relocate(home), name, "from")
        return Origin(dotted, name, "from")

    def runtime_check(self, name: str, origin: Origin) -> bool:
        """Confirm the origin really exposes ``name`` (best effort)."""
        if origin.kind in ("import", "import_as"):
            return importlib.util.find_spec(origin.module.split(".")[0]) is not None
        try:
            mod = importlib.import_module(origin.module)
        except Exception:
            return False
        return hasattr(mod, origin.name)


# ---------------------------------------------------------------------------
# Rewriting one file
# ---------------------------------------------------------------------------

def module_name_for(path: Path, manifest) -> tuple[str | None, bool]:
    path = path.resolve()
    for pkg in manifest.packages.values():
        src = pkg.src_dir.resolve()
        if src in path.parents:
            rel = path.relative_to(src).with_suffix("")
            parts = [pkg.import_name, *rel.parts]
            is_pkg = parts[-1] == "__init__"
            return ".".join(parts[:-1] if is_pkg else parts), is_pkg
    mono = manifest.source_root.resolve()
    if mono in path.parents:
        rel = path.relative_to(mono).with_suffix("")
        parts = [manifest.monolith_import, *rel.parts]
        is_pkg = parts[-1] == "__init__"
        return ".".join(parts[:-1] if is_pkg else parts), is_pkg
    return None, False


def _is_retired_or_aggregator(manifest, dotted: str) -> bool:
    """True for monolith modules that only re-export (retired aggregators)."""
    if manifest.package_for_module(dotted) != "monolith":
        return False
    rel = "/".join(dotted.split(".")[1:])
    for cand in (rel + ".py", rel + "/__init__.py"):
        if (manifest.source_root / cand).exists():
            return manifest.owner_of(cand)[0] == "retire"
    return False


def resource_explicit(path: Path, resolver: Resolver, manifest, write: bool, verbose: bool) -> int:
    """Rewrite ``from <retired aggregator> import a, b`` to the true origins."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    current, is_pkg = module_name_for(path, manifest)
    lines = text.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or any(a.name == "*" for a in node.names):
            continue
        if node.level:
            if not current:
                continue
            parts = current.split(".")
            cut = len(parts) - node.level + (1 if is_pkg else 0)
            target = ".".join(parts[:cut] + ([node.module] if node.module else []))
        else:
            target = node.module or ""
        if not _is_retired_or_aggregator(manifest, target):
            continue
        groups: dict[str, list[str]] = {}
        plain: list[str] = []
        ok = True
        for alias in node.names:
            origin = resolver.resolve(alias.name, target)
            if origin is None or not resolver.runtime_check(alias.name, origin):
                ok = False
                break
            bound = alias.asname or alias.name
            if origin.kind == "import":
                plain.append(origin.module if bound == origin.module else f"{origin.module} as {bound}")
            elif origin.kind == "import_as":
                plain.append(f"{origin.module} as {bound}")
            else:
                item = origin.name if origin.name == bound else f"{origin.name} as {bound}"
                groups.setdefault(origin.module, []).append(item)
        if not ok:
            if verbose:
                print(f"    keep {target} import at line {node.lineno} (unresolved {alias.name})")
            continue
        out = [f"import {p}" for p in plain]
        for mod, items in groups.items():
            out.append(f"from {mod} import {', '.join(items)}")
        edits.append((node.lineno, node.end_lineno or node.lineno, "\n".join(out)))
    for start, end, stmt in sorted(edits, key=lambda e: -e[0]):
        indent = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
        lines[start - 1 : end] = ["".join(indent + l + "\n" for l in stmt.splitlines())]
    if edits and write:
        path.write_text("".join(lines), encoding="utf-8")
    return len(edits)


def process(path: Path, resolver: Resolver, manifest, write: bool, verbose: bool) -> dict:
    resourced = resource_explicit(path, resolver, manifest, write, verbose)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    binder = _Binder()
    binder.visit(tree)
    if not binder.star_imports:
        return {"file": str(path), "stars": 0, "resourced": resourced}
    current, is_pkg = module_name_for(path, manifest)
    free = sorted(n for n in binder.used - binder.bound if n not in BUILTINS)
    # Names annotated in `__all__` of this file need to be resolvable too.
    all_node = next((n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)), None)
    exported = Resolver._literal_all(all_node.value) if all_node else None
    if exported:
        free = sorted(set(free) | (exported - binder.bound))

    stars_abs: list[tuple[ast.ImportFrom, str]] = []
    for node in binder.star_imports:
        if node.level and current:
            parts = current.split(".")
            cut = len(parts) - node.level + (1 if is_pkg else 0)
            target = ".".join(parts[:cut] + ([node.module] if node.module else []))
        else:
            target = node.module or ""
        stars_abs.append((node, target))

    resolved: dict[str, Origin] = {}
    unresolved: list[str] = []
    for name in free:
        origin = None
        for _, target in stars_abs:
            origin = resolver.resolve(name, target)
            if origin:
                break
        if origin and resolver.runtime_check(name, origin):
            resolved[name] = origin
        elif origin:
            resolved[name] = origin  # keep static answer; flag for review
            if verbose:
                print(f"    ? {name} -> {origin.module}.{origin.name} (unverified)")
        else:
            unresolved.append(name)

    # Group imports
    plain: list[str] = []
    froms: dict[str, list[str]] = {}
    for name, origin in sorted(resolved.items()):
        if origin.kind == "import":
            plain.append(origin.module)
        elif origin.kind == "import_as":
            plain.append(f"{origin.module} as {name}")
        else:
            mod = origin.module
            if current and mod == current:
                continue  # defined in this very module via re-export loop
            froms.setdefault(mod, []).append(name if origin.name == name else f"{origin.name} as {name}")

    def order_key(mod: str) -> tuple[int, str]:
        top = mod.split(".")[0]
        if top in STDLIB:
            return (0, mod)
        if manifest.package_for_module(mod) is None:
            return (1, mod)
        return (2, mod)

    lines: list[str] = []
    for mod in sorted(set(plain), key=lambda m: order_key(m.split(" as ")[0])):
        lines.append(f"import {mod}")
    for mod in sorted(froms, key=order_key):
        names = sorted(set(froms[mod]))
        stmt = f"from {mod} import {', '.join(names)}"
        if len(stmt) > 100:
            stmt = f"from {mod} import (\n    " + ",\n    ".join(names) + ",\n)"
        lines.append(stmt)

    src_lines = text.splitlines(keepends=True)
    first = min(n.lineno for n, _ in stars_abs)
    keep_star = bool(unresolved)
    replacement = "\n".join(lines) + ("\n" if lines else "")
    if keep_star:
        replacement += "# TODO(partition): unresolved names from wildcard imports: " + ", ".join(unresolved) + "\n"
    # Remove star import statements (bottom-up), insert replacement at first.
    for node, _ in sorted(stars_abs, key=lambda x: -x[0].lineno):
        start, end = node.lineno, node.end_lineno or node.lineno
        if keep_star:
            continue
        del src_lines[start - 1 : end]
    insert_at = first - 1
    src_lines.insert(insert_at, replacement)
    new_text = "".join(src_lines)
    if write and new_text != text:
        path.write_text(new_text, encoding="utf-8")
    return {
        "file": str(path),
        "stars": len(stars_abs),
        "resolved": len(resolved),
        "unresolved": unresolved,
        "origins": sorted({o.module for o in resolved.values()}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    reloc_path = manifest.source_root / "_relocations.json"
    relocations = {}
    if reloc_path.exists():
        import json

        relocations = json.loads(reloc_path.read_text())
    resolver = Resolver(manifest, relocations)
    status = 0
    files: list[Path] = []
    for f in args.files:
        if f.is_dir():
            files.extend(sorted(p for p in f.rglob("*.py") if "__pycache__" not in p.parts))
        else:
            files.append(f)
    for path in files:
        try:
            result = process(path, resolver, manifest, args.write, args.verbose)
        except Exception:
            print(f"!! {path}")
            traceback.print_exc()
            status = 1
            continue
        if result.get("resourced"):
            print(f"RES {path}: {result['resourced']} explicit import(s) re-sourced from retired modules")
        if result["stars"] == 0:
            continue
        flag = "OK " if not result["unresolved"] else "TODO"
        print(f"{flag} {path}: {result['stars']} wildcard(s), {result['resolved']} names resolved"
              + (f", unresolved: {', '.join(result['unresolved'])}" if result["unresolved"] else ""))
        if args.verbose:
            for o in result["origins"]:
                print(f"     from {o}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
