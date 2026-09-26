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
#
# FORCE (not setdefault) via test_isolation: an inherited DEFAULT_ROOT=<live>
# used to WIN over the setdefault below and route constants' import-time
# directory creation + write-probe into the operator's live storage (the same
# setdefault hole behind incident 2026-09-24). The DEFAULT_ROOT setdefault is
# kept as the fallback for the HUGPY_TEST_LIVE_ROOT=1 opt-out path.
from hugpy_platform.test_isolation import (  # noqa: E402
    install_live_storage_guard, isolate_storage_to_tmp,
)

_SANDBOX = tempfile.mkdtemp(prefix="hugpy-platform-tests-")
isolate_storage_to_tmp(base=os.path.join(_SANDBOX, "storage"))
install_live_storage_guard()
os.environ.setdefault("DEFAULT_ROOT", os.path.join(_SANDBOX, "storage"))
os.environ.setdefault("HUGPY_HOME", os.path.join(_SANDBOX, "hugpy_home"))
os.environ.setdefault("HUGPY_DATA_DIR", os.path.join(_SANDBOX, "data"))
os.environ.setdefault("HUGPY_CONFIG_DIR", os.path.join(_SANDBOX, "config"))
os.environ.setdefault("HUGPY_CACHE_DIR", os.path.join(_SANDBOX, "cache"))
