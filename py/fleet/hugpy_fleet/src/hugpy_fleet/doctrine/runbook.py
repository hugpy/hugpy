"""Locate and load the packaged fleet runbook (``doctrine/fleet_runbook.json``).

Resolved through ``importlib.resources`` so it works from a wheel, an
editable install or a zipapp — never through the monolith's ``__file__``.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict

__all__ = ["RUNBOOK_NAME", "runbook_path", "runbook_text", "load_runbook"]

RUNBOOK_NAME = "fleet_runbook.json"


def _resource():
    return resources.files("hugpy_fleet.doctrine").joinpath(RUNBOOK_NAME)


def runbook_path() -> str:
    """Filesystem path of the runbook (materialised if the package is zipped)."""
    with resources.as_file(_resource()) as p:
        return str(p)


def runbook_text() -> str:
    return _resource().read_text(encoding="utf-8")


def load_runbook() -> Dict[str, Any]:
    return json.loads(runbook_text())
