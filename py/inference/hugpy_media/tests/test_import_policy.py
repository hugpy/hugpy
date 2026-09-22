"""Structural import policy for hugpy_media (generated from partition.toml).

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

PACKAGE = "hugpy_media"
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE
ALLOWED = ['hugpy_platform', 'hugpy_storage', 'hugpy_engine']
OPTIONAL = []
ECOSYSTEM = ['hugpy_platform', 'hugpy_control', 'hugpy_storage', 'hugpy_engine', 'hugpy_media', 'hugpy_video', 'hugpy_oracle', 'hugpy_fleet', 'hugpy_curation', 'hugpy_ops', 'hugpy_discord', 'hugpy_server', 'hugpy']
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
                bad.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
    assert bad == [], "wildcard imports: " + ", ".join(bad)


def test_no_monolith_imports():
    bad = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in _names(node):
                    if _top(name) == MONOLITH:
                        bad.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} {name}")
    assert bad == [], "monolith imports: " + ", ".join(bad)


def test_only_allowed_ecosystem_imports():
    bad = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in _names(node):
                    top = _top(name)
                    if top in ECOSYSTEM and top != PACKAGE and top not in ALLOWED:
                        bad.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} {name}")
    assert bad == [], "imports outside the allowed dependency set: " + ", ".join(bad)


def test_optional_dependencies_are_lazy():
    bad = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _module_level_imports(tree):
            for name in _names(node):
                if _top(name) in OPTIONAL:
                    bad.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} {name}")
    assert bad == [], "optional dependencies imported at module level: " + ", ".join(bad)


def test_package_imports_without_monolith():
    statements = ["import sys", "sys.modules['abstract_hugpy_dev'] = None"]
    statements += [f"sys.modules[{name!r}] = None" for name in OPTIONAL]
    statements += [f"import {PACKAGE}", f"print({PACKAGE}.__name__)"]
    code = "; ".join(statements)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip() == PACKAGE
