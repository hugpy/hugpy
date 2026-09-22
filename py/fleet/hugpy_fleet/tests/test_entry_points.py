"""Every console-script entry point answers ``--help`` in strict mode
(monolith and optional packages blocked)."""

from __future__ import annotations

import subprocess
import sys

import pytest

ENTRY_POINTS = {
    "hugpy-worker": "hugpy_fleet.worker.agent:main",
    "hugpy-gguf-worker": "hugpy_fleet.gguf_worker.agent:main",
    "hugpy-phone-brick": "hugpy_fleet.phone_brick.__main__:main",
}


@pytest.mark.parametrize("name,target", sorted(ENTRY_POINTS.items()))
def test_entry_point_help(name, target):
    mod, fn = target.split(":")
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None; "
        "sys.modules['hugpy_media'] = None; sys.modules['hugpy_video'] = None; "
        f"import importlib; m = importlib.import_module({mod!r}); "
        f"sys.exit(getattr(m, {fn!r})(['--help']))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "usage" in proc.stdout.lower()


def test_pyproject_declares_entry_points():
    import pathlib
    text = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    for name, target in ENTRY_POINTS.items():
        assert f'{name} = "{target}"' in text
