"""spill.budget_bar — the honest resource-bar spec (t13/t14, operator 2026-07-17).

The console bars used to draw numerator and denominator from different universes
(physical-derived used vs central-limit total), collapsing on any under-budget
box to physical_total − central_limit — an artifact, not usage. This locks the
operator's REFINED spec (both clamps mandatory):

    external_headroom = physical_total − central_limit
    encroachment      = max(0, external_usage − external_headroom)
    bar_used          = min(central_limit, worker_usage + encroachment)   # ≤ limit
    remaining         = max(0, central_limit − worker_usage − encroachment)  # ≥ 0

Includes the operator's exact worked examples (in GiB units, scaled to bytes),
the encroachment clamp at 0, the over-limit clamp cases (at the limit → remaining
0 no warning; just over → remaining 0, over_limit true, raw fields exceed the
limit), no-limit physical fallback, and the allocator/bar agreement (free_ram_bytes
== the spec remaining, floored by reserve).

Runs like the other tests here: venv/bin/python tests/test_budget_bar_spec.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib
sp = importlib.import_module("hugpy_engine.spill")

G = 2 ** 30

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_budget_bar_spec():
    """Converted from the script-style suite: every `check` is an assert."""


    # ── operator's exact worked example: 128 physical / 90 limit / 20 worker / 10 ext
    b = sp.budget_bar(128 * G, 90 * G, 20 * G, 10 * G)
    check("example: semantics=central", b["semantics"] == "central")
    check("example: headroom = 128−90 = 38", b["external_headroom"] == 38 * G)
    check("example: encroachment 0 (10 ext fits 38 headroom)", b["encroachment"] == 0)
    check("example: bar_used = 20 (worker only)", b["bar_used"] == 20 * G)
    check("example: total = limit = 90", b["total"] == 90 * G)
    check("example: remaining = 70 to go", b["remaining"] == 70 * G)
    check("example: not over limit", b["over_limit"] is False and b["over_by"] == 0)
    check("example: raw_used == bar_used when under", b["raw_used"] == 20 * G)

    # ── external grows to 50 → encroachment 12 → bar 32/90, 58 to go
    b = sp.budget_bar(128 * G, 90 * G, 20 * G, 50 * G)
    check("ext=50: encroachment = 50−38 = 12", b["encroachment"] == 12 * G)
    check("ext=50: bar_used = 20+12 = 32", b["bar_used"] == 32 * G)
    check("ext=50: remaining = 90−32 = 58", b["remaining"] == 58 * G)
    check("ext=50: still not over limit", b["over_limit"] is False)

    # ── encroachment clamps at 0 (external well within its headroom)
    b = sp.budget_bar(128 * G, 90 * G, 20 * G, 0)
    check("no external: encroachment 0", b["encroachment"] == 0)
    check("no external: bar_used = worker", b["bar_used"] == 20 * G)
    check("no external: remaining = 70", b["remaining"] == 70 * G)

    # ── CLAMP: worker+encroachment EXACTLY at limit → remaining 0, NO warning
    #    90 limit, worker 80, ext = headroom(38) + 10 = 48 → encroach 10 → 80+10 = 90
    b = sp.budget_bar(128 * G, 90 * G, 80 * G, 48 * G)
    check("at-limit: encroachment 10", b["encroachment"] == 10 * G)
    check("at-limit: raw_used = 90 (== limit)", b["raw_used"] == 90 * G)
    check("at-limit: bar_used = 90 (fill full, not over)", b["bar_used"] == 90 * G)
    check("at-limit: remaining 0", b["remaining"] == 0)
    check("at-limit: NOT over_limit (== is not >)", b["over_limit"] is False and b["over_by"] == 0)

    # ── CLAMP: JUST over → remaining 0, over_limit true, raw exceeds limit
    #    worker 85, ext 48 → encroach 10 → raw 95 > 90
    b = sp.budget_bar(128 * G, 90 * G, 85 * G, 48 * G)
    check("over: raw_used = 95 (exceeds limit — the truth is kept)", b["raw_used"] == 95 * G)
    check("over: bar_used CLAMPED to 90 (fill never overflows)", b["bar_used"] == 90 * G)
    check("over: remaining CLAMPED to 0 (never negative)", b["remaining"] == 0)
    check("over: over_limit flag true", b["over_limit"] is True)
    check("over: over_by = 95−90 = 5", b["over_by"] == 5 * G)

    # ── CLAMP: worker ALONE over the limit (no external) still flags + clamps
    b = sp.budget_bar(128 * G, 90 * G, 100 * G, 0)
    check("worker-over: raw 100 kept", b["raw_used"] == 100 * G)
    check("worker-over: bar clamped 90", b["bar_used"] == 90 * G)
    check("worker-over: remaining 0", b["remaining"] == 0)
    check("worker-over: over_by 10", b["over_by"] == 10 * G and b["over_limit"] is True)

    # ── NO central limit → physical-total semantics (plain measured usage)
    b = sp.budget_bar(128 * G, None, 20 * G, 10 * G)
    check("no-limit: semantics=physical", b["semantics"] == "physical")
    check("no-limit: bar_used = worker+external (plain)", b["bar_used"] == 30 * G)
    check("no-limit: total = physical", b["total"] == 128 * G)
    check("no-limit: remaining = 128−30 = 98", b["remaining"] == 98 * G)
    check("no-limit: never over_limit", b["over_limit"] is False)
    check("no-limit: no encroachment concept", b["encroachment"] == 0)
    # 0 limit is treated the same as unset (defensive)
    check("zero-limit == no-limit semantics",
          sp.budget_bar(128 * G, 0, 20 * G, 10 * G)["semantics"] == "physical")

    # ── missing inputs never fabricate
    b = sp.budget_bar(128 * G, 90 * G, None, None)
    check("no worker/external: bar_used None", b["bar_used"] is None)
    check("no worker/external: remaining None", b["remaining"] is None)
    b = sp.budget_bar(None, 90 * G, 20 * G, 30 * G)
    check("no physical: all external treated as encroachment (limit is only denom)",
          b["encroachment"] == 30 * G and b["bar_used"] == 50 * G)

    # ── ALLOCATOR agreement: free_ram_bytes == the spec remaining, floored by reserve.
    #    Drive spill's own helpers via monkeypatched inputs so the allocator and the
    #    bar demonstrably read the SAME number.
    _orig = (sp.free_ram_raw_bytes, sp.ram_max_bytes, sp.ram_worker_bytes,
             sp.ram_external_bytes)
    try:
        import psutil  # noqa: F401 — free_ram_bytes reads psutil.virtual_memory().total
    except Exception:
        psutil = None

    try:
        # physical 128, limit 90, worker 20, external 10 → spec remaining 70.
        # reserve floor (raw) set generous (100) so the SPEC remaining binds.
        sp.free_ram_raw_bytes = lambda: 100 * G
        sp.ram_max_bytes = lambda: 90 * G
        sp.ram_worker_bytes = lambda: 20 * G
        sp.ram_external_bytes = lambda: 10 * G
        import types
        # Patch psutil.virtual_memory().total to 128G for the physical read inside
        # free_ram_bytes (only if psutil is importable; else the fn's physical=None
        # branch treats all external as encroachment — a different, still-valid path).
        if psutil is not None:
            _vm = psutil.virtual_memory
            psutil.virtual_memory = lambda: types.SimpleNamespace(total=128 * G, available=0)
        try:
            got = sp.free_ram_bytes()
        finally:
            if psutil is not None:
                psutil.virtual_memory = _vm
        check("allocator: budgetable free == spec remaining (70), reserve floor not binding",
              got == 70 * G)

        # Now make the reserve floor the binding constraint (raw 50 < remaining 70).
        sp.free_ram_raw_bytes = lambda: 50 * G
        if psutil is not None:
            psutil.virtual_memory = lambda: types.SimpleNamespace(total=128 * G, available=0)
        try:
            got = sp.free_ram_bytes()
        finally:
            if psutil is not None:
                psutil.virtual_memory = _vm
        check("allocator: reserve floor (50) wins when tighter than spec remaining (70)",
              got == 50 * G)

        # Over-limit box: worker 100 > limit 90 → spec remaining 0 → admits nothing.
        sp.free_ram_raw_bytes = lambda: 100 * G
        sp.ram_worker_bytes = lambda: 100 * G
        sp.ram_external_bytes = lambda: 0
        if psutil is not None:
            psutil.virtual_memory = lambda: types.SimpleNamespace(total=128 * G, available=0)
        try:
            got = sp.free_ram_bytes()
        finally:
            if psutil is not None:
                psutil.virtual_memory = _vm
        check("allocator: over-limit box admits 0 (remaining floored at 0)", got == 0)

        # No ceiling → reserve-only behavior verbatim (limit term absent).
        sp.ram_max_bytes = lambda: None
        sp.free_ram_raw_bytes = lambda: 42 * G
        check("allocator: no ceiling → reserve-only free (unchanged behavior)",
              sp.free_ram_bytes() == 42 * G)
    finally:
        (sp.free_ram_raw_bytes, sp.ram_max_bytes, sp.ram_worker_bytes,
         sp.ram_external_bytes) = _orig

    print(f"\nall {ok} checks passed")
