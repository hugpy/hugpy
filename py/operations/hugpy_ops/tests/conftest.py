"""Test session guard + isolation for hugpy_ops.

* The retired monolith must never satisfy an import from these tests.
  Blocking the name makes any leftover ``abstract_hugpy_dev`` import fail
  loudly. During the transition, HUGPY_ALLOW_MONOLITH=1 lifts the block for
  integration runs.
* Every state root the ecosystem could touch is pointed at a throwaway
  directory BEFORE any package import: the control job mirror is off, the
  platform's HUGPY_HOME / DEFAULT_ROOT land in a temp dir, and central
  resolves to loopback (no HUGPY_BASE_URL leaking in from the shell).
"""

from __future__ import annotations

import os
import sys
import tempfile

if not os.environ.get("HUGPY_ALLOW_MONOLITH"):
    sys.modules.setdefault("abstract_hugpy_dev", None)

_SCRATCH = tempfile.mkdtemp(prefix="hugpy-ops-tests-")
os.environ.setdefault("PROJECTS_HOME", os.path.join(_SCRATCH, "projects"))
os.environ.setdefault("HUGPY_COMMS_DB", "off")
os.environ.setdefault("HUGPY_HOME", os.path.join(_SCRATCH, "hugpy-home"))
os.environ.setdefault("DEFAULT_ROOT", os.path.join(_SCRATCH, "llm_storage"))
for _name in ("HUGPY_BASE_URL", "HUGPY_CENTRAL", "HUGPY_URL",
              "WORKER_CENTRAL_URL", "HUGPY_SENTINEL_CENTRAL",
              "HUGPY_SENTINEL_DIR", "HUGPY_CHAOS_OUT_DIR"):
    os.environ.pop(_name, None)
