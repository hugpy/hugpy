"""Context RANGE (operator 2026-10-02): an explicit [min, target] window —
the admission shrinks the subject's own ctx only as far as the deficit needs,
never below min; a ± deviation band keeps its floor jump."""
from hugpy_engine.fit import flex as F

G = 1 << 30


def test_range_bounds():
    assert F.ctx_band_bounds(50, None, 20) == (20, 50)
    assert F.ctx_band_bounds(50, 30, 20) == (20, 50)          # explicit floor wins over ±dev
    assert F.ctx_band_bounds(50, None, 0) == (1, 50)          # "0 point" = down to the smallest
    assert F.ctx_band_bounds(50, None, 80) == (50, 50)        # min above target collapses


def test_range_shrinks_only_as_far_as_needed():
    subj = {"weights_bytes": 4 * G, "kv_bytes": 4 * G, "ctx_pct": 50, "ctx_floor_pct": 10}
    p = F.plan_flex(subj, [], 1 * G)                          # 1 GiB short: 50% -> 37%
    assert p.action == "flex" and p.self_ctx_pct == 37
    assert F.kv_at_ctx_pct(4 * G, 50, p.self_ctx_pct) <= 3 * G


def test_range_floor_when_not_enough():
    subj = {"weights_bytes": 4 * G, "kv_bytes": 4 * G, "ctx_pct": 50, "ctx_floor_pct": 40}
    p = F.plan_flex(subj, [], 2 * G)                          # floor saves only 0.8 GiB
    assert p.self_ctx_pct == 40 and p.action == "evict"


def test_deviation_band_keeps_its_floor_jump():
    subj = {"weights_bytes": 4 * G, "kv_bytes": 4 * G, "ctx_pct": 50, "ctx_deviation_pct": 20}
    assert F.plan_flex(subj, [], 1 * G).self_ctx_pct == 30
