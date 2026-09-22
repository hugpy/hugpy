"""k37 Slice B, the ops half: chaos draws from the engine's five-mode
allocation vocabulary and ``build_spill`` materialises every mode (and every
legacy alias) onto recognised ``/assign`` keys.

The engine/fleet/server halves of the original script (override persistence,
flex band floors, spill planner, WorkerStore version gate, worker_routes
validation) live in ``py/services/hugpy_server/tests/integration/
test_alloc_modes_stack.py`` — they need packages ops may not import.
"""
from __future__ import annotations

import random

import pytest

from hugpy_engine.alloc_modes import (
    ALLOC_MODES,
    LEGACY_ALLOC_ALIASES,
    NONGGUF_ALLOWED_MODES,
    resolve_alloc_mode,
)
from hugpy_ops.chaos import assortment as A
from hugpy_ops.chaos.schema import ALLOC_MODES as CHAOS_MODES, SPILL_KEYS


def test_vocabulary_and_aliases():
    assert ALLOC_MODES == ("gpu-only", "ram-only", "max-gpu", "max-ram", "explicit")
    assert CHAOS_MODES == ALLOC_MODES
    assert resolve_alloc_mode("autofit") == ("max-gpu", True)
    assert resolve_alloc_mode("cpu-only") == ("ram-only", True)
    assert resolve_alloc_mode("budget") == ("explicit", True)
    assert resolve_alloc_mode("bands") == ("explicit", True)
    assert resolve_alloc_mode("max-gpu") == ("max-gpu", False)
    assert LEGACY_ALLOC_ALIASES["max-gpu"] == "gpu-only"
    assert resolve_alloc_mode("warp-drive") == (None, False)


def test_build_spill_materialises_all_five_modes():
    rng = random.Random(7)
    assert A.build_spill("gpu-only", 50, 4.0, rng)["n_gpu_layers"] == -1
    assert A.build_spill("ram-only", 50, 4.0, rng) == {"n_gpu_layers": "off"}
    assert A.build_spill("max-gpu", 50, 4.0, rng) == {}
    mr = A.build_spill("max-ram", 50, 4.0, rng)
    assert mr["alloc_mode"] == "max-ram" and mr["ctx_pct"] == 50
    ex = A.build_spill("explicit", 75, 6.0, rng)
    assert ex["alloc_mode"] == "explicit" and ex["gpu_mem_gib"] == 6.0
    assert 0 < ex["leniency_pct"] <= 100 and "priority" in ex and ex["ctx_pct"] == 75


@pytest.mark.parametrize("mode", ALLOC_MODES)
def test_every_built_spill_uses_recognised_assign_keys(mode):
    assert set(A.build_spill(mode, 50, 4.0, random.Random(1))) <= SPILL_KEYS


def test_build_spill_accepts_legacy_aliases():
    rng = random.Random(7)
    assert A.build_spill("autofit", 50, 4.0, rng) == {}
    assert A.build_spill("cpu-only", 50, 4.0, rng) == {"n_gpu_layers": "off"}
    assert A.build_spill("budget", 50, 4.0, random.Random(3))["alloc_mode"] == "explicit"


def test_framework_gating_matches_engine():
    assert A.NONGGUF_MODES == ("gpu-only", "ram-only", "max-gpu", "max-ram") == NONGGUF_ALLOWED_MODES
    assert A.modes_for("gguf") == ALLOC_MODES
