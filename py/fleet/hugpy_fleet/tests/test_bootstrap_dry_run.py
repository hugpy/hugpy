"""The shell bootstrap must have a real --dry-run: change NOTHING, print the plan.

bootstrap.sh is the one shell entry point (bare box -> enrolled worker). --dry-run
lets an operator SEE the venv/pip/install steps without running them.

Run: `PYTHONPATH=$(ls -d py/*/*/src|tr '\n' :) pytest tests/test_bootstrap_dry_run.py -q`
"""
import subprocess
import sys
from pathlib import Path

BOOTSTRAP = (Path(__file__).resolve().parents[1]
             / "src" / "hugpy_fleet" / "worker" / "bootstrap.sh")


def test_bootstrap_is_valid_bash():
    cp = subprocess.run(["bash", "-n", str(BOOTSTRAP)], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr


def test_dry_run_makes_no_venv_and_prints_plan(tmp_path):
    venv = tmp_path / "hugpy-worker" / "venv"
    cp = subprocess.run(
        ["bash", str(BOOTSTRAP), "--dry-run",
         "--central", "http://bogus.invalid:7002",
         "--name", "testbox", "--venv", str(venv)],
        capture_output=True, text=True, timeout=120,
    )
    # Dry-run must succeed and must not create the venv or its python.
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert not venv.exists(), "dry-run must not create the venv"
    out = cp.stdout + cp.stderr
    assert "DRY-RUN" in out
    # It must SHOW the pip install + venv steps it would run.
    assert "would run:" in out
    assert "python3 -m venv" in out
    assert "pip install" in out


def test_dry_run_help_lists_flag():
    cp = subprocess.run(["bash", str(BOOTSTRAP), "--help"],
                        capture_output=True, text=True)
    assert cp.returncode == 0
    assert "--dry-run" in (cp.stdout + cp.stderr)
