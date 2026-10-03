"""Env profile base "worker" (2026-10-02): an overlay venv that sees the worker
venv's packages, records only its OWN packages as the lock. Real venv, no pip
installs (fast, offline)."""
import subprocess

from hugpy_engine.serve import profiles as P


def test_worker_base_overlay_sees_worker_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_ROOT", str(tmp_path))
    res = P.materialize("overlay", [], base="worker")
    assert res["state"] == "ready", res
    py = P.profile_python("overlay")
    out = subprocess.run([py, "-c", "import pytest, sys; print(pytest.__file__)"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr          # pytest lives only in the worker venv
    st = P.read_state("overlay")
    assert st["base"] == "worker" and st["lock"] == []
    assert P.state_for("overlay", [], "worker") == "ready"
    assert P.state_for("overlay", [], "isolated") == "materializing"   # base is part of the hash


def test_isolated_hash_unchanged():
    assert P.manifest_hash(["a"]) == P.manifest_hash(["a"], "isolated")
    assert P.manifest_hash(["a"]) != P.manifest_hash(["a"], "worker")
