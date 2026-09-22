#!/usr/bin/env python3
"""Physically extract one package from the monolith according to partition.toml.

    python py/tooling/extract_package.py <package_id> [--dry-run] [--tests]

What it does, in order:

1. Scaffold the destination distribution (pyproject, src layout, README,
   generated import-policy test) when it does not exist yet.
2. Move every monolith file owned by the package to its destination. Package
   directories created on the way get an ``__init__.py``.
3. Record the moved modules in ``<monolith>/_relocations.json`` and make sure
   the monolith installs the relocation finder first thing in ``__init__``.
   Anything that still imports the old dotted path (dynamic imports, patch
   strings, code this tool could not rewrite) resolves to the *same* module
   object living in the new package, so nothing is imported twice.
4. Rewrite imports everywhere in the workspace (monolith src + tests, every
   extracted package): relative imports inside moved files become absolute;
   ``abstract_hugpy_dev.<old>`` becomes ``<new_package>.<path>``; dotted-string
   references to moved modules are updated.
5. With ``--tests``, move monolith tests whose imports are covered by this
   package and its allowed dependencies into ``<dest>/tests``.

The tool never deletes anything and never edits files outside the workspace.
Run ``python py/validate_partition.py --edges --package <id>`` afterwards; the
forbidden edges it prints are the semantic work left for a human or agent.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from partition_lib import (  # noqa: E402
    PY_ROOT,
    Manifest,
    Package,
    load_manifest,
    referenced_modules,
)

WORKSPACE = PY_ROOT.parent
RELOCATIONS_NAME = "_relocations.json"
FINDER_NAME = "_relocations.py"


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "{distribution}"
version = "0.1.0"
description = "{description}"
readme = "README.md"
requires-python = ">=3.10"
license = "LicenseRef-Proprietary"
authors = [{{ name = "putkoff" }}]
dependencies = [
{dependencies}]

[project.optional-dependencies]
test = ["pytest>=8"]
{optional}
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
{import_name} = ["py.typed", "**/*.json", "**/*.sh", "**/*.md", "**/*.txt"]
"""

README_TEMPLATE = """\
# {distribution}

`{import_name}` — extracted from `abstract_hugpy_dev` as part of the Hugpy
partition. Ownership and allowed dependencies are declared in
`py/partition.toml`; see `PARTITION.md` at the workspace root.

Allowed Python dependencies inside the ecosystem: {allowed}.
"""

CONFTEST_TEMPLATE = '''"""Test session guard: this package's tests must pass without the monolith."""

from __future__ import annotations

import os
import sys

# The retired monolith must never satisfy an import from these tests. Blocking
# the name makes any leftover `abstract_hugpy_dev` import fail loudly. During
# the transition, HUGPY_ALLOW_MONOLITH=1 lifts the block for integration runs.
if not os.environ.get("HUGPY_ALLOW_MONOLITH"):
    sys.modules.setdefault("abstract_hugpy_dev", None)
'''

IMPORT_POLICY_TEMPLATE = '''"""Structural import policy for {import_name} (generated from partition.toml).

The package may import only itself, the standard library, third-party
distributions, and the ecosystem packages listed in ALLOWED. Optional
ecosystem dependencies (OPTIONAL) may only be imported lazily, never at module
import time. Wildcard imports are forbidden. The monolith is forbidden.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = "{import_name}"
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE
ALLOWED = {allowed!r}
OPTIONAL = {optional!r}
ECOSYSTEM = {ecosystem!r}
MONOLITH = "abstract_hugpy_dev"


def _py_files():
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _top(name):
    return name.split(".")[0]


def _module_level_imports(tree):
    """Imports executed unconditionally at module import time."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If):
            for sub in node.body + node.orelse:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    yield sub


def _names(node):
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if node.level:
        return []
    return [node.module or ""]


def test_no_wildcard_imports():
    bad = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                bad.append(f"{{path.relative_to(PACKAGE_ROOT)}}:{{node.lineno}}")
    assert bad == [], "wildcard imports: " + ", ".join(bad)


def test_no_monolith_imports():
    bad = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in _names(node):
                    if _top(name) == MONOLITH:
                        bad.append(f"{{path.relative_to(PACKAGE_ROOT)}}:{{node.lineno}} {{name}}")
    assert bad == [], "monolith imports: " + ", ".join(bad)


def test_only_allowed_ecosystem_imports():
    bad = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in _names(node):
                    top = _top(name)
                    if top in ECOSYSTEM and top != PACKAGE and top not in ALLOWED and top not in OPTIONAL:
                        bad.append(f"{{path.relative_to(PACKAGE_ROOT)}}:{{node.lineno}} {{name}}")
    assert bad == [], "imports outside the allowed dependency set: " + ", ".join(bad)


def test_optional_dependencies_are_lazy():
    bad = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _module_level_imports(tree):
            for name in _names(node):
                if _top(name) in OPTIONAL:
                    bad.append(f"{{path.relative_to(PACKAGE_ROOT)}}:{{node.lineno}} {{name}}")
    assert bad == [], "optional dependencies imported at module level: " + ", ".join(bad)


def test_package_imports_without_monolith():
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None; "
        + "".join(f"sys.modules[{{name!r}}] = None; " for name in OPTIONAL)
        + f"import {{PACKAGE}}; print({{PACKAGE}}.__name__)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip() == PACKAGE
'''


def scaffold(manifest: Manifest, pkg: Package, dry_run: bool) -> None:
    root = pkg.root
    ecosystem = [p.import_name for p in manifest.packages.values()]
    allowed = [manifest.packages[d].import_name for d in pkg.depends]
    optional = [manifest.packages[d].import_name for d in pkg.optional_depends]
    deps = "".join(f'    "{manifest.packages[d].distribution}",\n' for d in pkg.depends)
    opt_lines = ""
    for d in pkg.optional_depends:
        dist = manifest.packages[d].distribution
        opt_lines += f'{d} = ["{dist}"]\n'
    files = {
        root / "pyproject.toml": PYPROJECT_TEMPLATE.format(
            distribution=pkg.distribution,
            description=f"Hugpy {pkg.id} package ({pkg.import_name})",
            dependencies=deps,
            optional=opt_lines,
            import_name=pkg.import_name,
        ),
        root / "README.md": README_TEMPLATE.format(
            distribution=pkg.distribution,
            import_name=pkg.import_name,
            allowed=", ".join(allowed) or "none",
        ),
        pkg.src_dir / "__init__.py": f'"""{pkg.distribution}: {pkg.id} package of the Hugpy ecosystem."""\n\n__version__ = "0.1.0"\n',
        pkg.src_dir / "py.typed": "",
        root / "tests" / "conftest.py": CONFTEST_TEMPLATE,
    }
    # The import policy test is always regenerated so manifest edits propagate.
    policy = IMPORT_POLICY_TEMPLATE.format(
        import_name=pkg.import_name,
        allowed=allowed,
        optional=optional,
        ecosystem=ecosystem,
    )
    always = {root / "tests" / "test_import_policy.py": policy}
    # A move that lands a package __init__ on the root must not be pre-empted.
    if any(dest in (".", "", "__init__.py") for dest in pkg.moves.values()):
        files.pop(pkg.src_dir / "__init__.py", None)
    for path, content in files.items():
        if path.exists():
            continue
        print(f"  scaffold {path.relative_to(WORKSPACE)}")
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    for path, content in always.items():
        print(f"  generate {path.relative_to(WORKSPACE)}")
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Moving
# ---------------------------------------------------------------------------

def move_sources(manifest: Manifest, pkg: Package, dry_run: bool) -> dict[str, str]:
    """Move owned files; return {old_module: new_module} for moved .py files."""
    moved: dict[str, str] = {}
    for rel in list(manifest.iter_source_files()):
        pid, _root, _dest = manifest.owner_of(rel)
        if pid != pkg.id:
            continue
        landing = manifest.dest_path(rel)
        assert landing is not None
        _, dest = landing
        src = manifest.source_root / rel
        print(f"  move {rel} -> {dest.relative_to(PY_ROOT)}")
        if rel.endswith(".py"):
            old = manifest.old_module(rel)
            new = manifest.new_module(rel)
            if old and new:
                moved[old] = new
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise SystemExit(f"refusing to overwrite existing destination {dest}")
        shutil.move(str(src), str(dest))
    if not dry_run:
        ensure_inits(pkg.src_dir)
        prune_empty_dirs(manifest.source_root)
    return moved


def ensure_inits(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir() and "__pycache__" not in path.parts:
            init = path / "__init__.py"
            if not init.exists() and any(p.suffix == ".py" for p in path.rglob("*.py")):
                init.write_text("", encoding="utf-8")
    if not (root / "__init__.py").exists():
        (root / "__init__.py").write_text("", encoding="utf-8")


def prune_empty_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if path.is_dir():
            entries = [e for e in path.iterdir() if e.name != "__pycache__"]
            if not entries:
                shutil.rmtree(path)


# ---------------------------------------------------------------------------
# Relocation map + finder in the monolith
# ---------------------------------------------------------------------------

FINDER_SOURCE = '''"""Generated by py/tooling/extract_package.py — do not edit.

Modules extracted from this monolith into standalone packages are recorded in
``_relocations.json``. This finder keeps the old dotted import path working by
aliasing it to the *same* module object from the new package, so code that has
not been rewritten yet (dynamic imports, patch strings) keeps functioning
without importing anything twice. It is deleted together with the monolith.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path

_MAP: dict[str, str] = json.loads(
    (Path(__file__).with_name("_relocations.json")).read_text(encoding="utf-8")
)
_PREFIX = __name__.rsplit(".", 1)[0] + "."


def relocated(fullname: str) -> str | None:
    if fullname in _MAP:
        return _MAP[fullname]
    parts = fullname.split(".")
    for cut in range(len(parts) - 1, 1, -1):
        head = ".".join(parts[:cut])
        if head in _MAP:
            return _MAP[head] + "." + ".".join(parts[cut:])
    return None


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        # An old *package* aliased to a plain module still needs submodule
        # lookups (``from old.pkg import sub``) to reach this finder.
        if not hasattr(module, "__path__") and any(k.startswith(spec.name + ".") for k in _MAP):
            module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        return None


class RelocationFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        new = relocated(fullname)
        if new is None or new == fullname:
            return None
        if importlib.util.find_spec(new) is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(new))


def install() -> None:
    if not any(isinstance(f, RelocationFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, RelocationFinder())
'''


def record_relocations(manifest: Manifest, moved: dict[str, str], dry_run: bool) -> dict[str, str]:
    root = manifest.source_root
    path = root / RELOCATIONS_NAME
    existing: dict[str, str] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(moved)
    if dry_run:
        return existing
    path.write_text(json.dumps(existing, indent=1, sort_keys=True), encoding="utf-8")
    (root / FINDER_NAME).write_text(FINDER_SOURCE, encoding="utf-8")
    init = root / "__init__.py"
    text = init.read_text(encoding="utf-8") if init.exists() else ""
    hook = "from ._relocations import install as _install_relocations\n_install_relocations()\n"
    if "_install_relocations" not in text:
        init.write_text(hook + text, encoding="utf-8")
    return existing


# ---------------------------------------------------------------------------
# Import rewriting
# ---------------------------------------------------------------------------

class Rewriter:
    def __init__(self, manifest: Manifest, relocations: dict[str, str], known_modules: set[str]):
        self.manifest = manifest
        self.map = relocations
        self.known = known_modules  # every monolith module that ever existed
        self.top = manifest.monolith_import

    # -- resolution ------------------------------------------------------
    def map_module(self, module: str) -> str | None:
        """New dotted name for a monolith module, or None if unmoved."""
        if module in self.map:
            return self.map[module]
        # A file owned by a different package than its parent directory must
        # map to the owner's destination, never to the parent's prefix.
        rel = self._rel_for(module)
        if rel is not None and rel not in ("__init__.py",):
            owner = self.manifest.owner_of(rel)[0]
            if owner not in ("retire", "unowned") and not (self.manifest.source_root / rel).exists():
                return self.manifest.new_module(rel)
        parts = module.split(".")
        for cut in range(len(parts) - 1, 1, -1):
            head = ".".join(parts[:cut])
            if head in self.map:
                return self.map[head] + "." + ".".join(parts[cut:])
        return None

    def _rel_for(self, module: str) -> str | None:
        parts = module.split(".")
        if parts[0] != self.top or len(parts) == 1:
            return None
        tail = parts[1:]
        for cand in ("/".join(tail) + ".py", "/".join(tail) + "/__init__.py"):
            if cand in self._rel_cache():
                return cand
        return None

    _rels: set[str] | None = None

    def _rel_cache(self) -> set[str]:
        if self._rels is None:
            rels = set()
            for pkg in self.manifest.packages.values():
                for src in pkg.moves:
                    rels.add(src)
            # every known module maps back to a path candidate
            for mod in self.known:
                tail = mod.split(".")[1:]
                if tail:
                    rels.add("/".join(tail) + ".py")
                    rels.add("/".join(tail) + "/__init__.py")
            self._rels = rels
        return self._rels

    def is_module(self, dotted: str) -> bool:
        if dotted in self.known:
            return True
        new = self.map_module(dotted)
        if new is None:
            return False
        return self._exists(new)

    def _exists(self, new: str) -> bool:
        top = new.split(".")[0]
        for pkg in self.manifest.packages.values():
            if pkg.import_name == top:
                rel = Path(*new.split(".")[1:]) if "." in new else Path()
                return (pkg.src_dir / rel).is_dir() or (pkg.src_dir / rel).with_suffix(".py").exists()
        return False

    # -- statement rewriting ----------------------------------------------
    def rewrite_file(self, path: Path, current_module: str | None, is_pkg: bool, moved: bool = False) -> bool:
        """``current_module`` is the module's ORIGINAL (monolith) dotted name when
        ``moved`` is true, so relative imports resolve against where the file
        came from; otherwise it is the file's current dotted name."""
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return False
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return False
        lines = text.splitlines(keepends=True)
        edits: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                new_stmt = self._rewrite_from(node, current_module, is_pkg, moved)
            elif isinstance(node, ast.Import):
                new_stmt = self._rewrite_import(node, lines)
            else:
                continue
            if new_stmt is not None:
                edits.append((node.lineno, node.end_lineno or node.lineno, new_stmt))
        changed = False
        for start, end, new_stmt in sorted(edits, key=lambda e: -e[0]):
            indent = re.match(r"[ \t]*", lines[start - 1]).group(0)
            replacement = "".join(indent + line + "\n" for line in new_stmt.splitlines())
            lines[start - 1 : end] = [replacement]
            changed = True
        new_text = "".join(lines)
        new_text2 = self._rewrite_strings(new_text)
        if new_text2 != new_text:
            changed = True
        if changed:
            path.write_text(new_text2, encoding="utf-8")
        return changed

    def _absolute(self, node: ast.ImportFrom, current_module: str | None, is_pkg: bool) -> str | None:
        if not node.level:
            return node.module
        if current_module is None:
            return None
        parts = current_module.split(".")
        cut = len(parts) - node.level + (1 if is_pkg else 0)
        if cut < 0:
            return None
        base = parts[:cut]
        return ".".join(base + ([node.module] if node.module else []))

    def _rewrite_from(self, node, current_module, is_pkg, moved) -> str | None:
        absolute = self._absolute(node, current_module, is_pkg)
        if absolute is None:
            return None
        if not absolute.startswith(self.top + ".") and absolute != self.top:
            return None
        # Only touch statements whose target (or a submodule target) moved.
        groups: dict[str, list[str]] = {}
        order: list[str] = []
        any_moved = False
        for alias in node.names:
            name = alias.name
            asname = alias.asname
            candidate = f"{absolute}.{name}"
            if name != "*" and self.is_module(candidate) and self.map_module(candidate):
                new = self.map_module(candidate)
                assert new is not None
                any_moved = True
                parent, _, last = new.rpartition(".")
                if parent == "":
                    key = f"import {new}"
                    item = f"{new} as {asname}" if asname else new
                else:
                    key = parent
                    item = last if (asname is None and last == name) else f"{last} as {asname or name}"
            else:
                new_mod = self.map_module(absolute)
                if new_mod is None:
                    key = absolute
                    item = f"{name} as {asname}" if asname else name
                else:
                    any_moved = True
                    key = new_mod
                    item = f"{name} as {asname}" if asname else name
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(item)
        if not any_moved:
            # A relative import inside a moved file must become absolute even
            # when its target is still in the monolith.
            if node.level and moved:
                items = ", ".join(f"{a.name} as {a.asname}" if a.asname else a.name for a in node.names)
                return self._fmt(absolute, items)
            return None
        parts = []
        for key in order:
            items = ", ".join(groups[key])
            if key.startswith("import "):
                parts.append(f"import {items}")
            else:
                parts.append(self._fmt(key, items))
        return "\n".join(parts)

    def _fmt(self, module: str, items: str) -> str:
        stmt = f"from {module} import {items}"
        if len(stmt) > 100 and "," in items:
            return f"from {module} import (\n    " + ",\n    ".join(i.strip() for i in items.split(",")) + ",\n)"
        return stmt

    def _rewrite_import(self, node: ast.Import, lines) -> str | None:
        out = []
        changed = False
        for alias in node.names:
            new = self.map_module(alias.name) if alias.name.startswith(self.top + ".") else None
            if new is None:
                out.append(f"{alias.name} as {alias.asname}" if alias.asname else alias.name)
                continue
            changed = True
            if alias.asname:
                out.append(f"{new} as {alias.asname}")
            else:
                # `import a.b.c` binds `a`; keep attribute chains working via an alias.
                out.append(f"{new} as {alias.name.split('.')[-1]}  # TODO(partition): was `import {alias.name}`; fix attribute chains")
        if not changed:
            return None
        return "import " + ", ".join(out) if len(out) == 1 else "\n".join(f"import {o}" for o in out)

    _STRING_RE = None

    def _rewrite_strings(self, text: str) -> str:
        if self.top not in text:
            return text
        pattern = re.compile(r"(?P<q>['\"])" + re.escape(self.top) + r"(?P<rest>(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?P<tail>(?::[A-Za-z_][A-Za-z0-9_.]*)?)(?P=q)")

        def sub(match: re.Match) -> str:
            dotted = self.top + match.group("rest")
            new = self.map_module(dotted)
            if new is None:
                # attribute on a moved module: "pkg.mod.attr"
                head, _, attr = dotted.rpartition(".")
                new_head = self.map_module(head)
                if new_head is None:
                    return match.group(0)
                new = f"{new_head}.{attr}"
            return f"{match.group('q')}{new}{match.group('tail')}{match.group('q')}"

        return pattern.sub(sub, text)


def known_monolith_modules(manifest: Manifest, relocations: dict[str, str]) -> set[str]:
    known = set(relocations)
    for rel in manifest.iter_source_files():
        old = manifest.old_module(rel)
        if old:
            known.add(old)
    known.add(manifest.monolith_import)
    return known


def rewrite_workspace(manifest: Manifest, relocations: dict[str, str], dry_run: bool) -> int:
    rewriter = Rewriter(manifest, relocations, known_monolith_modules(manifest, relocations))
    reverse = {new: old for old, new in relocations.items()}
    count = 0
    targets: list[tuple[Path, str | None, bool, bool]] = []
    # monolith source
    for path in manifest.source_root.rglob("*.py"):
        if "__pycache__" in path.parts or path.name == FINDER_NAME:
            continue
        rel = path.relative_to(manifest.source_root).as_posix()
        targets.append((path, manifest.old_module(rel), rel.endswith("__init__.py"), False))
    # monolith tests
    tests = manifest.source_root.parents[1] / "tests"
    if tests.exists():
        for path in tests.rglob("*.py"):
            targets.append((path, None, False, False))
    # extracted packages
    for pkg in manifest.packages.values():
        for base in (pkg.src_dir, pkg.root / "tests"):
            if not base.exists():
                continue
            for path in base.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                if base == pkg.src_dir:
                    rel = path.relative_to(pkg.src_dir).with_suffix("")
                    parts = [pkg.import_name, *rel.parts]
                    is_pkg = parts[-1] == "__init__"
                    if is_pkg:
                        parts = parts[:-1]
                    new_name = ".".join(parts)
                    old_name = reverse.get(new_name)
                    if old_name is not None:
                        # The old path was a package __init__ if any relocated
                        # module lives beneath it (e.g. flask_app/__init__.py
                        # that now lives as a plain module).
                        old_is_pkg = is_pkg or any(k.startswith(old_name + ".") for k in relocations)
                        targets.append((path, old_name, old_is_pkg, True))
                    else:
                        targets.append((path, None, is_pkg, False))
                else:
                    targets.append((path, None, False, False))
    for path, module, is_pkg, moved in targets:
        if dry_run:
            continue
        if rewriter.rewrite_file(path, module, is_pkg, moved):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def move_tests(manifest: Manifest, pkg: Package, dry_run: bool) -> list[str]:
    tests = manifest.source_root.parents[1] / "tests"
    if not tests.exists():
        return []
    allowed = {pkg.id, *pkg.depends}
    moved = []
    for path in sorted(tests.rglob("test_*.py")):
        rel = path.relative_to(tests).as_posix()
        pkgs = set()
        for mod, _ in referenced_modules(path, rel, "tests"):
            dst = manifest.package_for_module(mod)
            if dst is None:
                continue
            pkgs.add(dst)
        if pkg.id not in pkgs or not pkgs <= allowed:
            continue
        dest = pkg.root / "tests" / path.name
        print(f"  test {rel} -> {dest.relative_to(PY_ROOT)}")
        moved.append(rel)
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest = dest.with_name(dest.stem + "_" + path.parent.name + ".py")
            shutil.move(str(path), str(dest))
    return moved


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tests", action="store_true", help="also move covered monolith tests")
    parser.add_argument("--rewrite-only", action="store_true", help="only rerun the import rewrite")
    parser.add_argument("--scaffold-only", action="store_true", help="only (re)generate scaffold files and the import-policy test")
    parser.add_argument("--tests-only", action="store_true", help="only move covered monolith tests")
    args = parser.parse_args()

    manifest = load_manifest()
    pkg = manifest.packages.get(args.package)
    if pkg is None:
        raise SystemExit(f"unknown package {args.package}; known: {manifest.order}")

    reloc_path = manifest.source_root / RELOCATIONS_NAME
    relocations = json.loads(reloc_path.read_text()) if reloc_path.exists() else {}

    if args.scaffold_only:
        print(f"== scaffold {pkg.distribution}")
        scaffold(manifest, pkg, args.dry_run)
        return 0
    if args.tests_only:
        moved_tests = move_tests(manifest, pkg, args.dry_run)
        print(f"   {len(moved_tests)} tests moved")
        return 0
    if not args.rewrite_only:
        print(f"== scaffold {pkg.distribution}")
        scaffold(manifest, pkg, args.dry_run)
        print(f"== move sources for {pkg.id}")
        moved = move_sources(manifest, pkg, args.dry_run)
        print(f"   {len(moved)} modules moved")
        relocations = record_relocations(manifest, moved, args.dry_run)
    print("== rewrite imports across the workspace")
    changed = rewrite_workspace(manifest, relocations, args.dry_run)
    print(f"   {changed} files rewritten")
    if args.tests and not args.rewrite_only:
        print(f"== move covered tests")
        moved_tests = move_tests(manifest, pkg, args.dry_run)
        print(f"   {len(moved_tests)} tests moved")
    print("== next: python py/validate_partition.py --edges --package", pkg.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
