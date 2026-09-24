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
    "hugpy-drift-check": "hugpy_ops.drift",
    "hugpy-model-audit": "hugpy_ops.model_audit",
    "hugpy-model-manifest-backfill": "hugpy_ops.model_manifest_backfill",
    "hugpy-model-archive": "hugpy_ops.model_archive",
    "hugpy-admission-seed": "hugpy_ops.admission:seed_main",
    "hugpy-model-resweep": "hugpy_ops.resweep",
    "hugpy-vl-reclassify": "hugpy_ops.admission:vl_reclassify_main",
}


def _target(module):
    """``(module, function)`` — an entry is ``module`` (function ``main``)
    or ``module:function``."""
    mod, _, fn = module.partition(":")
    return mod, fn or "main"


def _strict_env():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.pop("HUGPY_ALLOW_MONOLITH", None)
    return env


def _run(code: str):
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120, env=_strict_env())


@pytest.mark.parametrize("script, module", sorted(ENTRY_POINTS.items()))
def test_console_script_help_in_strict_subprocess(script, module):
    mod, fn = _target(module)
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None\n"
        f"from {mod} import {fn} as main\n"
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
        mod, fn = _target(module)
        assert scripts[name] == f"{mod}:{fn}"


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
    # The version is the installed distribution's (git-derived, lockstep),
    # never a literal; the source-tree fallback is 0.0.0+unknown.
    from importlib.metadata import PackageNotFoundError, version
    try:
        expected = version("hugpy-ops")
    except PackageNotFoundError:
        expected = "0.0.0+unknown"
    assert proc.stdout.strip() == expected


def test_lazy_submodule_access_and_versions_helper():
    import hugpy_ops
    assert hugpy_ops.versions.distribution_version("definitely-not-a-dist") is None
    versions = hugpy_ops.versions.distribution_versions()
    assert set(versions) == {"hugpy-fleet", "hugpy-server"}
    assert "sentinel" in dir(hugpy_ops)
    with pytest.raises(AttributeError):
        hugpy_ops.no_such_module
