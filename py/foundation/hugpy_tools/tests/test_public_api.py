"""The package surface resolves and importing it stays stdlib-only."""
from __future__ import annotations

import subprocess
import sys

import hugpy_tools


def test_all_names_resolve():
    for name in hugpy_tools.__all__:
        assert getattr(hugpy_tools, name) is not None, name


def test_import_is_stdlib_only():
    code = (
        "import sys; import hugpy_tools; "
        "heavy = [m for m in ('requests', 'bs4', 'yaml', 'flask', 'torch', "
        "'abstract_utilities', 'abstract_webtools', 'tiktoken', 'playwright', "
        "'selenium') if m in sys.modules]; print(','.join(heavy))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "", "non-stdlib import: %s" % proc.stdout


def test_no_hugpy_sibling_imports():
    # Foundation layer: hugpy_tools must not import any other hugpy_* package.
    code = (
        "import sys; import hugpy_tools; "
        "sib = [m for m in sys.modules if m.split('.')[0].startswith('hugpy_') "
        "and m.split('.')[0] != 'hugpy_tools']; print(','.join(sorted(sib)))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "", "sibling import: %s" % proc.stdout
