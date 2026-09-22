"""k118 — the environment DOCTRINE: what a hugpy worker is supposed to HAVE.

Four real failures in one week, all the same shape — a worker accepted work it
could not do, and found out at runtime:

  * a-brain had no ``ffmpeg``          -> round-trip ASR dead
  * computron had no ``bitsandbytes``  -> every 4-bit job failed
  * a-brain t2i: "diffusers + torch not installed" after the TTS seating
  * chatterbox silently needs ``setuptools<81``

None of those is a bug in the code that failed. Each is a box whose environment
did not match what the fleet assumed, discovered at the worst possible moment.
So: the box SELF-REPORTS (``worker_agent.environment_report``), one reference
box's report is frozen as a versioned DOCTRINE (``doctrine.py``), and every
install and every heartbeat is DIFFED against it (``doctor.py``). k101's probes
consume the diff, so a missing dep becomes honest ineligibility BEFORE dispatch
instead of a stack trace after it.

The rule that keeps this honest, inherited from ``oracle.probes``: a check that
could not be performed answers UNKNOWN, never ``ok``. An unreadable venv, an
unreachable worker and a box with no report yet are all "we do not know" — and
"we do not know" never clears a worker for work.
"""
from __future__ import annotations

from hugpy_fleet.doctrine.doctrine import (
    DEFAULT_SEVERITY,
    KNOWN_REQUIREMENTS,
    SEVERITIES,
    Doctrine,
    DoctrineEntry,
    Requirement,
    classification_for,
    doctrine_dir,
    doctrine_path,
    latest,
    list_versions,
    load,
    pin_satisfied,
    save,
    snapshot,
)
from hugpy_fleet.doctrine.doctor import DoctrineReport, Finding, assess

__all__ = [
    "DEFAULT_SEVERITY",
    "Doctrine",
    "DoctrineEntry",
    "DoctrineReport",
    "Finding",
    "KNOWN_REQUIREMENTS",
    "Requirement",
    "SEVERITIES",
    "assess",
    "classification_for",
    "doctrine_dir",
    "doctrine_path",
    "latest",
    "list_versions",
    "load",
    "pin_satisfied",
    "save",
    "snapshot",
    "RUNBOOK_NAME",
    "runbook_path",
    "runbook_text",
    "load_runbook",
]
from hugpy_fleet.doctrine.runbook import RUNBOOK_NAME, load_runbook, runbook_path, runbook_text  # noqa: E402
