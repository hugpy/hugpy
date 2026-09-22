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

# hugpy_platform.constants creates the storage tree on import and app_dirs
# creates ~/.hugpy on first use. Point both at a throwaway root for the whole
# session so no test can touch a real /mnt/llm_storage or the home directory.
_SANDBOX = tempfile.mkdtemp(prefix="hugpy-platform-tests-")
os.environ.setdefault("DEFAULT_ROOT", os.path.join(_SANDBOX, "storage"))
os.environ.setdefault("HUGPY_HOME", os.path.join(_SANDBOX, "hugpy_home"))
os.environ.setdefault("HUGPY_DATA_DIR", os.path.join(_SANDBOX, "data"))
os.environ.setdefault("HUGPY_CONFIG_DIR", os.path.join(_SANDBOX, "config"))
os.environ.setdefault("HUGPY_CACHE_DIR", os.path.join(_SANDBOX, "cache"))
