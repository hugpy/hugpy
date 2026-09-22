"""``hpy`` is the stdlib-only fleet browser; it must run on a bare install."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from hugpy import hpy

HPY = Path(hpy.__file__)


def test_hpy_help_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "hugpy.hpy", "--help"],
                          capture_output=True, text=True, timeout=110)
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ("workers", "models", "where", "call", "ask"):
        assert name in proc.stdout


def test_hpy_runs_straight_from_the_source_file():
    proc = subprocess.run([sys.executable, str(HPY), "--help"],
                          capture_output=True, text=True, timeout=110)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.startswith("usage: hpy")


def test_hpy_imports_only_the_stdlib():
    tree = ast.parse(HPY.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)


def test_plan_prefers_warm_then_local_then_catalog():
    cat = {
        "warm-m": {"task": "text-generation", "tasks": [], "serveable": True, "blocked": False},
        "disk-m": {"task": "text-generation", "tasks": [], "serveable": True, "blocked": False},
        "cold-m": {"task": "text-generation", "tasks": [], "serveable": True, "blocked": False},
        "img-m": {"task": "text-to-image", "tasks": [], "serveable": True, "blocked": False},
    }
    ws = [{"name": "a", "loaded": ["warm-m"], "local": ["disk-m"]}]
    plan = hpy._plan(cat, ws)
    pairs = [(m, w) for m, w, _ in plan]
    assert pairs[:2] == [("warm-m", "a"), ("disk-m", "a")]
    assert ("cold-m", None) in pairs
    assert pairs.index(("cold-m", None)) > pairs.index(("disk-m", "a"))
    assert all(m != "img-m" for m, _, _ in plan)


def test_plan_forced_model_falls_back_to_central():
    plan = hpy._plan({}, [{"name": "a", "loaded": ["x"], "local": []}], force_model="x", force_worker="b")
    assert plan[0] == ("x", "b", "forced model+worker")
    assert plan[-1][1] is None
