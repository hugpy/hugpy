"""Test session guard: this package's tests must pass without the monolith."""

from __future__ import annotations

import os
import sys

# The retired monolith must never satisfy an import from these tests. Blocking
# the name makes any leftover `abstract_hugpy_dev` import fail loudly. During
# the transition, HUGPY_ALLOW_MONOLITH=1 lifts the block for integration runs.
if not os.environ.get("HUGPY_ALLOW_MONOLITH"):
    sys.modules.setdefault("abstract_hugpy_dev", None)

# Script-style studio tests create their work dirs with tempfile.mkdtemp(dir=...)
# under the storage root's video_intel/_scratch (the route jails source paths to
# it). A fresh machine (CI runner) has no storage root yet, so make it here,
# before collection imports the modules that mkdtemp at import time.
from hugpy_platform.constants import DEFAULT_ROOT  # noqa: E402

os.makedirs(os.path.join(DEFAULT_ROOT, "video_intel", "_scratch"), exist_ok=True)


# --------------------------------------------------------------------------- #
# Script-style tests moved from the monolith patch ``media_bus`` module globals
# directly (``media_bus.set_progress = ...``, ``DB_PATH``, ``_initialized``,
# ``is_cancelling``) and do not always restore them; a later test then reads a
# stale hook. Snapshot the mutable bus surface around EVERY test so order can
# never leak state between files.
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402

_BUS_ATTRS = ("DB_PATH", "_initialized", "set_progress", "is_cancelling",
              "_bridge", "_admission_enabled", "_probe_admission",
              "_force_admit_safe", "work_once", "get", "enqueue")


@pytest.fixture(autouse=True)
def _restore_media_bus_globals():
    mod = sys.modules.get("hugpy_video.intel.media_bus")
    saved = {a: getattr(mod, a) for a in _BUS_ATTRS if mod is not None and hasattr(mod, a)}
    yield
    mod = sys.modules.get("hugpy_video.intel.media_bus")
    if mod is None:
        return
    for attr, value in saved.items():
        setattr(mod, attr, value)
