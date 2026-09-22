"""Every console script answers ``--help`` in a STRICT subprocess (monolith
blocked, no PYTHONPATH help) and exits 0. Also: ``import hugpy_ops`` is light."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

ENTRY_POINTS = {
    "hugpy-sentinel": "hugpy_ops.sentinel.__main__",
    "hugpy-chaos": "hugpy_ops.chaos.runner",
    "hugpy-keeper": "hugpy_ops.keeper",
    "hugpy-todo-keeper": "hugpy_ops.todo_keeper_daemon",
    "hugpy-provisioner": "hugpy_ops.provisioner",
}


def _strict_env():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.pop("HUGPY_ALLOW_MONOLITH", None)
    return env


def _run(code: str):
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120, env=_strict_env())


@pytest.mark.parametrize("script, module", sorted(ENTRY_POINTS.items()))
def test_console_script_help_in_strict_subprocess(script, module):
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None\n"
        f"from {module} import main\n"
        "try:\n"
        "    rc = main(['--help'])\n"
        "except SystemExit as e:\n"
        "    rc = e.code\n"
        "raise SystemExit(rc or 0)\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "usage:" in proc.stdout and script in proc.stdout


def test_pyproject_declares_every_console_script():
    import tomllib
    from pathlib import Path
    data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert set(scripts) == set(ENTRY_POINTS)
    for name, module in ENTRY_POINTS.items():
        assert scripts[name] == f"{module}:main"


def test_import_hugpy_ops_is_light():
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None\n"
        "import hugpy_ops\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "('hugpy_engine', 'hugpy_video', 'hugpy_storage', 'hugpy_fleet', "
        "'hugpy_curation', 'hugpy_control', 'torch', 'flask')]\n"
        "assert not heavy, heavy\n"
        "assert set(hugpy_ops.__all__) >= {'sentinel', 'chaos', 'keeper', "
        "'provisioner', 'todo_keeper', 'todo_keeper_daemon', 'versions'}\n"
        "print(hugpy_ops.__version__)\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "0.1.0"


def test_lazy_submodule_access_and_versions_helper():
    import hugpy_ops
    assert hugpy_ops.versions.distribution_version("definitely-not-a-dist") is None
    versions = hugpy_ops.versions.distribution_versions()
    assert set(versions) == {"hugpy-fleet", "hugpy-server"}
    assert "sentinel" in dir(hugpy_ops)
    with pytest.raises(AttributeError):
        hugpy_ops.no_such_module
