"""Test session guard: this package's tests must pass without the monolith."""

from __future__ import annotations

import os
import sys

# The retired monolith must never satisfy an import from these tests. Blocking
# the name makes any leftover `abstract_hugpy_dev` import fail loudly. During
# the transition, HUGPY_ALLOW_MONOLITH=1 lifts the block for integration runs.
if not os.environ.get("HUGPY_ALLOW_MONOLITH"):
    sys.modules.setdefault("abstract_hugpy_dev", None)

# Force every storage root to a private temp dir before hugpy_platform.constants
# is imported, and arm the live-storage audit guard (incident 2026-09-24). Fleet
# storage-budget / reap tests classify + size paths under DEFAULT_ROOT; keep the
# default resolution off any inherited live root. Per-test monkeypatched roots
# (tmp_path) still win — the guard allows system temp dirs.
from hugpy_platform.test_isolation import (  # noqa: E402
    install_live_storage_guard, isolate_storage_to_tmp,
)

isolate_storage_to_tmp()
install_live_storage_guard()
