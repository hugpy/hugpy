"""``import hugpy_control`` is light (no singleton is built) and lazy submodule
access works."""
from __future__ import annotations

import subprocess
import sys

import hugpy_control


def test_all_submodules_resolve_lazily():
    for name in hugpy_control.__all__:
        assert getattr(hugpy_control, name) is not None, name
    assert hugpy_control.bus.TOPIC_CATALOG_CHANGED == "catalog.changed"


def test_import_creates_no_store_singletons():
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None; "
        "import hugpy_control; "
        "print(','.join(m for m in ('hugpy_control.jobs', 'hugpy_control.settings', "
        "'hugpy_control.principals', 'hugpy_control.shared', 'flask', 'hugpy_engine', "
        "'hugpy_fleet', 'hugpy_server') if m in sys.modules))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "", proc.stdout
