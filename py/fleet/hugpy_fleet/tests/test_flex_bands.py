"""t21 — tolerance-band secondary allocation layer: the PURE band math + the
flex-before-evict decision (worker_agent/flex.py).

Asserts, with NO GPU and NO I/O (that's the point of keeping flex.py pure):
  * band floors/ceilings are the hard percent-of-total interval, clamped;
  * ctx% band math + the linear KV re-pricing;
  * uncontended == target (bands untouched when it already fits);
  * self-flex compresses the subject's OWN ctx first (ctx-first ordering);
  * a higher-priority subject compresses lower-priority neighbours within THEIR
    bands, never below the floor;
  * protection outranks priority (a protected neighbour never flex-compresses);
  * equal/greater-priority neighbours are immune (only STRICTLY lower yield);
  * insufficient flex -> evict with a lowest-priority-first order.

Run: venv/bin/python -m pytest tests/test_flex_bands.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import flex as F

GIB = 1 << 30


# ── the priority comparator seam ────────────────────────────────────────────
def test_priority_key_unset_is_zero():
    assert F.flex_priority_key(None) == 0
    assert F.flex_priority_key({}) == 0
    assert F.flex_priority_key({"priority": None}) == 0
    assert F.flex_priority_key({"priority": 0}) == 0


def test_priority_key_parses_int_and_stringy():
    assert F.flex_priority_key({"priority": 5}) == 5
    assert F.flex_priority_key({"priority": "3"}) == 3
    assert F.flex_priority_key({"priority": "garbage"}) == 0
    assert F.flex_priority_key("not-a-dict") == 0


# ── bytes-domain band math (VRAM / RAM) ─────────────────────────────────────
def test_band_no_deviation_collapses_to_point():
    lo, hi = F.band_bounds(10 * GIB, None, 24 * GIB)
    assert lo == hi == 10 * GIB
    lo, hi = F.band_bounds(10 * GIB, 0, 24 * GIB)
    assert lo == hi == 10 * GIB


def test_band_deviation_is_percent_of_whole():
    # 20% of a 24 GiB whole == 4.8 GiB span each side of a 10 GiB target.
    lo, hi = F.band_bounds(10 * GIB, 20, 24 * GIB)
    assert abs(lo - (10 * GIB - 4.8 * GIB)) < 1024
    assert abs(hi - (10 * GIB + 4.8 * GIB)) < 1024
    assert F.band_floor(10 * GIB, 20, 24 * GIB) == lo
    assert F.band_ceiling(10 * GIB, 20, 24 * GIB) == hi


def test_band_clamps_to_domain():
    # floor never below 0, ceiling never above the whole.
    lo, hi = F.band_bounds(2 * GIB, 50, 24 * GIB)   # span 12 GiB > target
    assert lo == 0.0
    lo, hi = F.band_bounds(23 * GIB, 50, 24 * GIB)
    assert hi == 24 * GIB


def test_band_unknown_whole_collapses_to_point():
    # no honest denominator -> can't size a percent band -> point (no flex).
    assert F.band_bounds(10 * GIB, 20, None) == (10 * GIB, 10 * GIB)
    assert F.band_bounds(10 * GIB, 20, 0) == (10 * GIB, 10 * GIB)


# ── ctx% band math + KV re-pricing ──────────────────────────────────────────
def test_ctx_band_none_target_is_none():
    assert F.ctx_band_bounds(None, 20) is None


def test_ctx_band_points_and_clamps():
    assert F.ctx_band_bounds(50, 0) == (50, 50)
    assert F.ctx_band_bounds(50, 20) == (30, 70)
    assert F.ctx_band_bounds(10, 25) == (1, 35)      # floor clamps to 1
    assert F.ctx_band_bounds(90, 25) == (65, 100)    # ceil clamps to 100


def test_kv_scales_linearly_with_ctx():
    assert F.kv_at_ctx_pct(1000, 50, 30) == 600
    assert F.kv_at_ctx_pct(1000, 50, 50) == 1000
    assert F.kv_at_ctx_pct(0, 50, 30) == 0           # degenerate -> unchanged
    assert F.kv_at_ctx_pct(1000, 0, 30) == 1000      # missing target pct -> unchanged


# ── plan_flex: uncontended == target ────────────────────────────────────────
def test_uncontended_proceeds_untouched():
    plan = F.plan_flex({"kv_bytes": 1000, "ctx_pct": 50,
                        "ctx_deviation_pct": 20, "priority": 9}, [], 0)
    assert plan.action == "proceed"
    assert plan.self_ctx_pct is None and plan.compress == []


# ── plan_flex: self-flex (ctx-first, compress the subject's OWN ctx) ─────────
def test_self_flex_ctx_covers_the_deficit():
    # subject KV 1000 at ctx 50%, band ±20 -> floor 30% -> KV 600 -> frees 400.
    subject = {"weights_bytes": 8 * GIB, "kv_bytes": 1000, "ctx_pct": 50,
               "ctx_deviation_pct": 20, "priority": 0}
    plan = F.plan_flex(subject, [], deficit_bytes=300)
    assert plan.action == "flex"
    assert plan.self_ctx_pct == 30           # compressed to its band floor
    assert plan.compress == []               # no neighbour disturbed
    assert plan.freed_bytes >= 300


def test_self_flex_tried_before_neighbours():
    # If the subject's OWN ctx flex is enough, no neighbour is compressed even
    # when the subject outranks a compressible neighbour.
    subject = {"kv_bytes": 1000, "ctx_pct": 50, "ctx_deviation_pct": 20,
               "priority": 9}
    neigh = {"model_key": "n", "kv_bytes": 5000, "ctx_pct": 80,
             "ctx_deviation_pct": 40, "protected": False, "pinned": False,
             "alloc": {"priority": 0}}
    plan = F.plan_flex(subject, [neigh], deficit_bytes=200)
    assert plan.action == "flex"
    assert plan.self_ctx_pct == 30
    assert plan.compress == []               # neighbour untouched — self was enough


# ── plan_flex: neighbour compression by priority ────────────────────────────
def _neigh(mk, kv, ctx, dev, prio, protected=False, pinned=False):
    return {"model_key": mk, "kv_bytes": kv, "ctx_pct": ctx,
            "ctx_deviation_pct": dev, "protected": protected, "pinned": pinned,
            "alloc": {"priority": prio}}


def test_higher_priority_compresses_lower_neighbour():
    subject = {"kv_bytes": 0, "ctx_pct": None, "priority": 5}   # no self ctx flex
    # neighbour KV 2000 at ctx 80, band ±25 -> floor 55 -> KV 1375 -> frees 625.
    lo_prio = _neigh("lo", 2000, 80, 25, prio=1)
    plan = F.plan_flex(subject, [lo_prio], deficit_bytes=500)
    assert plan.action == "flex"
    assert plan.self_ctx_pct is None
    assert [c["model_key"] for c in plan.compress] == ["lo"]
    assert plan.compress[0]["to_ctx_pct"] == 55


def test_equal_or_higher_priority_neighbour_is_immune():
    subject = {"kv_bytes": 0, "ctx_pct": None, "priority": 5}
    equal = _neigh("eq", 2000, 80, 25, prio=5)     # not strictly lower
    higher = _neigh("hi", 2000, 80, 25, prio=9)
    plan = F.plan_flex(subject, [equal, higher], deficit_bytes=500)
    assert plan.action == "evict"                  # nobody flex-eligible
    assert plan.compress == []


def test_protection_outranks_priority():
    subject = {"kv_bytes": 0, "ctx_pct": None, "priority": 9}
    # A protected neighbour is LOWER priority AND has compressible ctx, but
    # protection is absolute — it is never offered for flex compression.
    prot = _neigh("prot", 5000, 90, 40, prio=0, protected=True)
    plan = F.plan_flex(subject, [prot], deficit_bytes=500)
    assert plan.action == "evict"
    assert plan.compress == []


def test_neighbour_order_lowest_priority_first_then_pin_tiebreak():
    subject = {"kv_bytes": 0, "ctx_pct": None, "priority": 9}
    # two equally-low-priority neighbours; the UNPINNED yields before the pinned
    # (pin-as-tiebreak). Small headroom each so BOTH are needed.
    a = _neigh("pinned_low", 1000, 60, 20, prio=1, pinned=True)
    b = _neigh("unpinned_low", 1000, 60, 20, prio=1, pinned=False)
    # each frees KV 1000 -> floor 40 -> 667 -> ~333 each; need both.
    plan = F.plan_flex(subject, [a, b], deficit_bytes=600)
    assert plan.action == "flex"
    order = [c["model_key"] for c in plan.compress]
    assert order[0] == "unpinned_low"              # unpinned yields first


def test_insufficient_flex_falls_to_evict_priority_ordered():
    subject = {"kv_bytes": 0, "ctx_pct": None, "priority": 9}
    # tiny compressible headroom, big deficit -> flex can't cover -> evict, with
    # the priority order (lowest first) handed to the evictor.
    lo = _neigh("lo", 200, 60, 10, prio=1)
    mid = _neigh("mid", 200, 60, 10, prio=4)
    plan = F.plan_flex(subject, [lo, mid], deficit_bytes=50 * GIB)
    assert plan.action == "evict"
    assert plan.priority_order[0] == "lo"          # lowest priority yields first
    assert plan.deficit_bytes > 0
