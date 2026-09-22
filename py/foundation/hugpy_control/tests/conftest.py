"""Test session guard: this package's tests must pass without the monolith."""

from __future__ import annotations

import os
import sys
import tempfile

# The retired monolith must never satisfy an import from these tests. Blocking
# the name makes any leftover `abstract_hugpy_dev` import fail loudly. During
# the transition, HUGPY_ALLOW_MONOLITH=1 lifts the block for integration runs.
if not os.environ.get("HUGPY_ALLOW_MONOLITH"):
    sys.modules.setdefault("abstract_hugpy_dev", None)

# Every store in this package resolves its file lazily from env; pin them all to
# a throwaway tree so the module singletons (job_store, settings_store,
# principal_store) never touch a live comms db, settings.json or ~/.hugpy.
_SANDBOX = tempfile.mkdtemp(prefix="hugpy-control-tests-")
os.environ.setdefault("DEFAULT_ROOT", os.path.join(_SANDBOX, "storage"))
os.environ.setdefault("HUGPY_HOME", os.path.join(_SANDBOX, "hugpy_home"))
os.environ.setdefault("HUGPY_COMMS_DB", os.path.join(_SANDBOX, "comms.db"))
os.environ.setdefault("HUGPY_SETTINGS_PATH", os.path.join(_SANDBOX, "settings.json"))
os.environ.setdefault("HUGPY_PRINCIPALS_PATH", os.path.join(_SANDBOX, "principals.json"))
os.environ.setdefault("HUGPY_CALL_LOG", os.path.join(_SANDBOX, "calls.jsonl"))
