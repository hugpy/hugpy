"""Test session guard for the compatibility shell.

Unlike the hugpy_* package conftests this one must NOT block the name
``abstract_hugpy_dev``: the shell *is* that name. It instead asserts that the
name resolves to this package and not to a leftover monolith checkout or a
stale site-packages install.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_SANDBOX = tempfile.mkdtemp(prefix="hugpy-compat-tests-")
os.environ.setdefault("DEFAULT_ROOT", os.path.join(_SANDBOX, "storage"))
os.environ.setdefault("HUGPY_HOME", os.path.join(_SANDBOX, "hugpy_home"))
os.environ.setdefault("HUGPY_DATA_DIR", os.path.join(_SANDBOX, "data"))
os.environ.setdefault("HUGPY_CONFIG_DIR", os.path.join(_SANDBOX, "config"))
os.environ.setdefault("HUGPY_CACHE_DIR", os.path.join(_SANDBOX, "cache"))

SHELL_SRC = Path(__file__).resolve().parents[1] / "src" / "abstract_hugpy_dev"


@pytest.fixture(scope="session", autouse=True)
def _shell_is_this_package():
    import abstract_hugpy_dev

    got = Path(abstract_hugpy_dev.__file__).resolve().parent
    assert got == SHELL_SRC, (
        f"abstract_hugpy_dev resolved to {got}, not the compatibility shell "
        f"{SHELL_SRC}; a monolith checkout or old site-packages install is shadowing it")


@pytest.fixture(autouse=True)
def _cwd_outside_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
