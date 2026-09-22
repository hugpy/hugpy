"""The package surface: ``__all__`` resolves, and importing the package is light."""
from __future__ import annotations

import subprocess
import sys

import hugpy_platform


def test_all_names_resolve():
    for name in hugpy_platform.__all__:
        assert getattr(hugpy_platform, name) is not None, name


def test_import_is_side_effect_free_and_stdlib_first():
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None; "
        "import hugpy_platform; "
        "heavy = [m for m in ('hugpy_platform.constants', 'huggingface_hub', 'psutil', "
        "'requests', 'flask', 'torch') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "", f"heavy modules imported by `import hugpy_platform`: {proc.stdout}"
