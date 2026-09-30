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


import pytest  # noqa: E402 — the overlay reset below


@pytest.fixture(autouse=True)
def _reset_request_spill_overlay():
    """Each test is a fresh request: clear the per-request spill overlay
    (hugpy_engine.spill ContextVar, 2026-09-30) before and after, so one
    test's _apply_spill never shadows another test's os.environ setup."""
    try:
        from hugpy_engine.spill import set_request_env
    except Exception:  # noqa: BLE001 — an engine without the overlay
        yield
        return
    set_request_env(None)
    yield
    set_request_env(None)
