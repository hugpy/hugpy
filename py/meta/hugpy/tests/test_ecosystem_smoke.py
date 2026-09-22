"""Ecosystem smoke: every distribution in ``py/partition.toml`` imports cleanly
with the monolith blocked.

This is the meta package's view of the split: if any ``hugpy_*`` package still
reaches for ``abstract_hugpy_dev`` at import time, ``pip install hugpy[...]``
would ship a broken command. Packages whose directory has no ``src`` yet are
skipped, not failed.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = PY_ROOT / "partition.toml"


def _packages() -> list[tuple[str, str, Path]]:
    if not MANIFEST.exists():
        return []
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    monolith = data.get("monolith_import", "abstract_hugpy_dev")
    out = []
    for pkg in data.get("package", []):
        out.append((pkg["distribution"], pkg["import_name"], PY_ROOT / pkg["destination"]))
    return [(d, i, p, monolith) for d, i, p in out]


PACKAGES = _packages()


@pytest.mark.skipif(not PACKAGES, reason="partition.toml not found (installed wheel, not the workspace)")
@pytest.mark.parametrize("distribution,import_name,root,monolith", PACKAGES,
                         ids=[p[0] for p in PACKAGES])
def test_distribution_imports_without_monolith(distribution, import_name, root, monolith):
    if not (root / "src").is_dir():
        pytest.skip(f"{distribution}: {root} has no src/ yet")
    code = (
        f"import sys; sys.modules[{monolith!r}] = None; "
        f"import {import_name}; "
        f"assert {monolith!r} not in sys.modules or sys.modules[{monolith!r}] is None; "
        f"print({import_name}.__name__)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=110)
    assert proc.returncode == 0, f"{distribution}:\n{proc.stderr[-4000:]}"
    assert proc.stdout.strip() == import_name
