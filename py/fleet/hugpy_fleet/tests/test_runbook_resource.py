"""The fleet runbook ships inside the package and resolves via importlib.resources."""

from __future__ import annotations

import os

from hugpy_fleet import doctrine
from hugpy_fleet.doctrine import runbook


def test_runbook_resolves_and_parses():
    p = runbook.runbook_path()
    assert os.path.isfile(p) and p.endswith(runbook.RUNBOOK_NAME)
    assert "abstract_hugpy_dev" not in p
    data = runbook.load_runbook()
    assert isinstance(data, dict) and data


def test_runbook_exported_from_doctrine_package():
    assert doctrine.load_runbook is runbook.load_runbook
    assert "load_runbook" in doctrine.__all__
