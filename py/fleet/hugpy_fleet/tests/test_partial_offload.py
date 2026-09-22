"""t21 stage (2.5) — honest GGUF PARTIAL offload (autofit's hybrid contract).

The PURE layers-that-fit math in worker_agent/flex.plan_partial_offload: when the
FULL GGUF doesn't fit the GPU even after flex+evict, degrade to a hybrid (offload
the layers that fit, stream the rest to CPU RAM) instead of hard-refusing.

Asserts, with NO GPU and NO I/O:
  * layers-that-fit math (weights+KV per layer, floor division);
  * KV priced proportionally to the offloaded fraction (KV rides its GPU layer);
  * a smaller (band-capped) VRAM budget yields fewer layers;
  * RAM-remainder guard refuses a hybrid whose CPU share would OOM host RAM;
  * degenerate-offload floor refuses a hybrid that offloads almost nothing;
  * intent modes: cpu -> 0 layers; Max GPU == autofit for an oversize model;
    an explicit requested count is honored and capped to fit;
  * not-computable geometry -> None (caller keeps its refusal);
  * the coder-next-on-ae worked example (23.6 GB card, 2.4 GB reserve).

Run: venv/bin/python -m pytest tests/test_partial_offload.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import flex as F

GIB = 1 << 30


def _plan(**kw):
    base = dict(weights_bytes=40 * GIB, kv_bytes=0, total_layers=40,
                vram_budget_bytes=20 * GIB, ram_free_bytes=200 * GIB, intent="auto")
    base.update(kw)
    return F.plan_partial_offload(**base)


# ── layers-that-fit math ────────────────────────────────────────────────────
def test_half_the_budget_offloads_half_the_layers():
    # 40 GiB weights / 40 layers = 1 GiB/layer; 20 GiB budget -> 20 layers.
    p = _plan(weights_bytes=40 * GIB, kv_bytes=0, total_layers=40,
              vram_budget_bytes=20 * GIB)
    assert p.admit is True
    assert p.n_gpu_layers == 20
    assert p.gpu_pct == 50
    assert p.vram_need_bytes == 20 * GIB          # 20 layers * 1 GiB
    assert p.weights_cpu_bytes == 20 * GIB        # the remainder streams to RAM
    assert p.ram_need_bytes == 20 * GIB


def test_floor_division_never_over_admits_the_budget():
    # 1 GiB/layer, 20.9 GiB budget -> 20 layers (floor), never 21.
    p = _plan(vram_budget_bytes=int(20.9 * GIB))
    assert p.n_gpu_layers == 20
    assert p.vram_need_bytes <= int(20.9 * GIB)


def test_smaller_band_capped_budget_yields_fewer_layers():
    big = _plan(vram_budget_bytes=20 * GIB).n_gpu_layers
    small = _plan(vram_budget_bytes=10 * GIB).n_gpu_layers
    assert small < big
    assert small == 10                            # 10 GiB / 1 GiB-per-layer


# ── KV priced proportionally to the offloaded fraction ──────────────────────
def test_kv_rides_the_offloaded_fraction():
    # 32 GiB weights + 8 GiB KV over 32 layers. Budget 20 GiB. Per-layer VRAM =
    # (32+8)/32 = 1.25 GiB -> 16 layers fit (16*1.25 = 20). Half the layers ->
    # half the KV on GPU (4 GiB), half streamed to RAM (4 GiB).
    p = _plan(weights_bytes=32 * GIB, kv_bytes=8 * GIB, total_layers=32,
              vram_budget_bytes=20 * GIB)
    assert p.n_gpu_layers == 16
    assert p.gpu_pct == 50
    assert p.kv_gpu_bytes == 4 * GIB
    assert p.kv_cpu_bytes == 4 * GIB
    assert p.vram_need_bytes == 20 * GIB          # weights_gpu(16) + kv_gpu(4)
    assert p.ram_need_bytes == 20 * GIB           # weights_cpu(16) + kv_cpu(4)


# ── RAM-remainder guard: never admit-then-OOM ───────────────────────────────
def test_ram_guard_refuses_when_cpu_remainder_would_oom():
    # 40 layers, 1 GiB/layer, budget 10 GiB -> 10 on GPU, 30 GiB streamed to RAM.
    # Only 5 GiB RAM budgetable -> refuse (would OOM), with a RAM reject reason.
    p = _plan(vram_budget_bytes=10 * GIB, ram_free_bytes=5 * GIB)
    assert p.admit is False
    assert "host RAM" in p.reject_reason
    assert p.n_gpu_layers == 10                    # the plan is still reported


def test_ram_guard_admits_when_remainder_fits():
    p = _plan(vram_budget_bytes=10 * GIB, ram_free_bytes=40 * GIB)
    assert p.admit is True


# ── degenerate-offload floor: don't spend VRAM for ~nothing ─────────────────
def test_degenerate_offload_below_floor_refuses():
    # 40 layers -> floor = ceil(0.05*40) = 2. A 1 GiB budget offloads 1 layer
    # (below the 2-layer floor) -> refuse the hybrid.
    p = _plan(vram_budget_bytes=1 * GIB)
    assert p.n_gpu_layers == 1
    assert p.admit is False
    assert "degenerate" in p.reject_reason


def test_just_above_floor_admits():
    # 2 layers == the floor -> admit.
    p = _plan(vram_budget_bytes=2 * GIB)
    assert p.n_gpu_layers == 2
    assert p.admit is True


# ── intent modes ────────────────────────────────────────────────────────────
def test_cpu_intent_offloads_nothing_and_is_not_degenerate():
    p = _plan(intent="cpu", vram_budget_bytes=20 * GIB, ram_free_bytes=200 * GIB)
    assert p.n_gpu_layers == 0
    assert p.gpu_pct == 0
    assert p.admit is True                         # explicit CPU-only, not dross
    assert p.vram_need_bytes == 0


def test_cpu_intent_still_ram_guarded():
    p = _plan(intent="cpu", weights_bytes=40 * GIB, total_layers=40,
              ram_free_bytes=5 * GIB)
    assert p.admit is False                        # can't hold the whole model
    assert "host RAM" in p.reject_reason


def test_max_gpu_equals_autofit_for_an_oversize_model():
    auto = _plan(intent="auto", vram_budget_bytes=13 * GIB)
    maxg = _plan(intent="gpu", vram_budget_bytes=13 * GIB)
    assert auto.n_gpu_layers == maxg.n_gpu_layers  # converge — no reserve squeeze
    assert maxg.n_gpu_layers == 13


def test_explicit_requested_count_is_honored_and_capped():
    # Budget fits 20 layers; an explicit request of 8 is honored (<= fit).
    p = _plan(vram_budget_bytes=20 * GIB, requested_layers=8)
    assert p.n_gpu_layers == 8
    # An explicit request ABOVE what fits is capped to the budget.
    p2 = _plan(vram_budget_bytes=10 * GIB, requested_layers=30)
    assert p2.n_gpu_layers == 10


# ── not computable -> None (caller keeps its honest refusal) ────────────────
def test_missing_layer_count_returns_none():
    assert _plan(total_layers=None) is None
    assert _plan(total_layers=0) is None


def test_missing_weight_size_returns_none():
    assert _plan(weights_bytes=None) is None
    assert _plan(weights_bytes=0) is None


# ── the coder-next-on-ae worked example ─────────────────────────────────────
def test_coder_next_on_ae_worked_example():
    # The live refusal: needs 51.8 GB, 21.2 GB free of 23.6 GB (2.4 GB reserve).
    # Honest split ~= 48.4 GiB weights + 3.4 GiB KV @ ctx16k; the offload budget
    # is free - ceiling_reserve = 21.2 - 2.4 = 18.8 GiB. total_layers read from
    # .block_count at runtime; 48 is representative for Qwen3-Coder-Next.
    p = F.plan_partial_offload(
        weights_bytes=int(48.4 * GIB), kv_bytes=int(3.4 * GIB), total_layers=48,
        vram_budget_bytes=int(18.8 * GIB), ram_free_bytes=100 * GIB, intent="auto")
    assert p.admit is True
    assert p.n_gpu_layers == 17                    # 17/48 layers fit the 18.8 GiB
    assert p.gpu_pct == 35
    assert p.vram_need_bytes <= int(18.8 * GIB)    # never over the budget
    # ~33 GiB streams to CPU RAM (fits ae's ample host RAM).
    assert 32 * GIB < p.ram_need_bytes < 35 * GIB
