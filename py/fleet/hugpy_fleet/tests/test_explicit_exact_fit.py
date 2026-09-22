"""Explicit-mode EXACT-FIT admission (coder-next/ae, 2026-08-28).

The live refusal this pins: Qwen3-Coder-Next (48 layers, MoE) under mode
"explicit" with gpu_mem_gib ~= the 1.49 GiB non-expert share and 0% leniency.
Central's MoE-aware admission makes the worker's vram_budget == gpu_target ==
leniency floor EXACTLY — and plan_explicit_offload busted against its own
target: n_gpu rounded the achieved share DOWN to whole layers while
floor_layers rounded the floor UP, so any non-layer-aligned target lost the
layer comparison by construction ("bust past the leniency floor" at a fit the
budget's BYTES fully covered).

The invariant (the fix): the floor is a BYTE share — if the budget's bytes
cover the floor's bytes, layer quantization must never turn that into a bust;
and an admitted plan never prices more layer-bytes than the budget holds.

Run: venv/bin/python tests/test_explicit_exact_fit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")

GIB = 2 ** 30

from hugpy_fleet.worker.flex import plan_explicit_offload

# Deliberately NON-layer-aligned geometry: whole is not a multiple of the
# layer count, and the target is a small byte share (the MoE backbone shape).
TOTAL = 48
WEIGHTS = 51 * GIB + 12345                    # ~the honest 1.15x'd coder-next
TARGET = int(0.03 * WEIGHTS)                  # ~3% — 1.44/48ths: not aligned
assert (TARGET * TOTAL) % WEIGHTS != 0, "test geometry must not be layer-aligned"

# 1) EXACT fit: budget == target == floor (0% leniency) -> ADMIT. This is the
#    live ae shape (band_ceiling collapses the budget onto the gpu_mem_gib
#    target, so equality is the COMMON case, not a fluke).
p = plan_explicit_offload(weights_bytes=WEIGHTS, kv_bytes=0, total_layers=TOTAL,
                          vram_budget_bytes=TARGET, ram_free_bytes=1024 * GIB,
                          mode="explicit", priority_device="gpu",
                          gpu_target_bytes=TARGET, leniency_pct=0.0)
check("exact-fit target at 0% leniency ADMITS (no bust against its own target)",
      p is not None and p.admit)
check("exact fit still offloads something", p.n_gpu_layers >= 1)
check("exact fit never prices past the budget (bytes-to-bytes)",
      p.n_gpu_layers * WEIGHTS <= TARGET * TOTAL)
check("admitted plan's vram_need stays within the budget",
      p.vram_need_bytes <= p.vram_budget_bytes + TOTAL)  # +TOTAL: rounding slop < 1 B/layer

# 2) One byte UNDER the floor -> still an honest refusal (the bust path is
#    preserved for genuinely insufficient budgets).
p2 = plan_explicit_offload(weights_bytes=WEIGHTS, kv_bytes=0, total_layers=TOTAL,
                           vram_budget_bytes=TARGET - 1, ram_free_bytes=1024 * GIB,
                           mode="explicit", priority_device="gpu",
                           gpu_target_bytes=TARGET, leniency_pct=0.0)
check("budget one byte under the floor still REFUSES",
      p2 is not None and not p2.admit)
check("the refusal still names mode + floor",
      "explicit" in p2.reject_reason and "floor" in p2.reject_reason)

# 3) Non-aligned target, budget comfortably >= floor -> admit, reach the
#    ceil-side layer count (>= floor_layers), and never exceed the budget's
#    layer-bytes.
BUDGET = 20 * GIB
p3 = plan_explicit_offload(weights_bytes=WEIGHTS, kv_bytes=0, total_layers=TOTAL,
                           vram_budget_bytes=BUDGET, ram_free_bytes=1024 * GIB,
                           mode="explicit", priority_device="gpu",
                           gpu_target_bytes=TARGET, leniency_pct=0.0)
check("non-aligned target with roomy budget admits", p3 is not None and p3.admit)
check("roomy budget reaches the floor's layer count (ceil-consistent)",
      p3.n_gpu_layers >= p3.floor_layers)
check("admitted layers' bytes fit the budget: n_gpu*layer_bytes <= budget",
      p3.n_gpu_layers * WEIGHTS <= BUDGET * TOTAL)

# 4) The literal live shape, in-band: 48 layers, ~51.8 GiB whole, target ==
#    budget == ~1.49 GiB backbone, 0% leniency (was: "2/48 layers, ~1.5 GiB
#    but only 1.5 GiB budgetable — bust past the leniency floor").
BACKBONE = int(1.49 * GIB)
p4 = plan_explicit_offload(weights_bytes=WEIGHTS, kv_bytes=0, total_layers=TOTAL,
                           vram_budget_bytes=BACKBONE, ram_free_bytes=1024 * GIB,
                           mode="explicit", priority_device="gpu",
                           gpu_target_bytes=BACKBONE, leniency_pct=0.0)
check("the live coder-next/ae shape admits instead of busting",
      p4 is not None and p4.admit and p4.n_gpu_layers >= 1)

print(f"all {ok} checks passed")
