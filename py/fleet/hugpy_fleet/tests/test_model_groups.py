"""MODEL GROUPS — derivation, tick filtering, the ladder walk, and the
priority -> declare-need handoff.

Spec: dev/MODEL-GROUPS-SPEC.md. The off-path guarantee lives in its own file
(test_model_groups_offpath.py) and is the release gate; this file tests what the
feature DOES once an operator turns it on.

FIXTURES ARE REAL. The catalog rows are lifted verbatim from the live dev
catalog (/mnt/llm_storage/projects/model_discovery.json, 2026-07-28) and the
quant ladder is the real on-disk contents of
/mnt/llm_storage/models/gguf/Qwen/Qwen2.5-7B-Instruct-GGUF, litter included —
the lone ``q4_0-00002-of-00002`` with no shard 1 is in here on purpose, because
"an incomplete shard set is never a rung" is the kind of rule that only stays
true if something keeps proving it. Nothing here reads or mutates that file.

    ./venv/bin/pytest tests/test_model_groups.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine.resolvers import groups as G


# ---------------------------------------------------------------------------
# Fixtures — verbatim from the live catalog
# ---------------------------------------------------------------------------
CATALOG = {
    # The operator's motivating pair. NOTE: the transformers half is NOT in the
    # live dev catalog (only the GGUF is) — it is included here because the
    # ruling is about the PAIR, and a test that only ever sees one member could
    # not prove the cross-framework tick filter works.
    "Qwen2.5-7B-Instruct-GGUF": {
        "framework": "gguf", "hub_id": "Qwen/Qwen2.5-7B-Instruct-GGUF"},
    "Qwen~Qwen2.5-7B-Instruct": {
        "framework": "transformers", "hub_id": "Qwen/Qwen2.5-7B-Instruct"},
    # A REAL transformers+gguf pair from the live catalog.
    "Qwen2.5-VL-7B-Instruct-GGUF": {
        "framework": "gguf", "hub_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"},
    "Qwen~Qwen2.5-VL-7B-Instruct": {
        "framework": "transformers", "hub_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    # Two PUBLISHERS of the same base — one group (publisher is ignored).
    "Qwen~Qwen3-Coder-Next-GGUF": {
        "framework": "gguf", "hub_id": "Qwen/Qwen3-Coder-Next-GGUF"},
    "unsloth~Qwen3-Coder-Next-GGUF": {
        "framework": "gguf", "hub_id": "unsloth/Qwen3-Coder-Next-GGUF"},
    # imatrix packaging on one half only — still one group.
    "DavidAU~MN-GRAND-23.5B-Gutenberg-UNCENSORED-V2-GLM4.7-Thinking": {
        "framework": "transformers",
        "hub_id": "DavidAU/MN-GRAND-23.5B-Gutenberg-UNCENSORED-V2-GLM4.7-Thinking"},
    "DavidAU~MN-GRAND-23.5B-Gutenberg-UNCENSORED-V2-GLM4.7-Thinking-NEO-Imatrix-GGUF": {
        "framework": "gguf",
        "hub_id": "DavidAU/MN-GRAND-23.5B-Gutenberg-UNCENSORED-V2-GLM4.7-Thinking-NEO-Imatrix-GGUF"},
    # DIFFERENT bases that must NOT merge (false merges are worse than splits).
    "Qwen2.5-Coder-32B-4bit": {
        "framework": "transformers", "hub_id": "mlx-community/Qwen2.5-Coder-32B-4bit"},
    "Qwen2.5-Coder-32B-Instruct-GGUF": {
        "framework": "gguf", "hub_id": "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF"},
    "Qwen2.5-Coder-3B-Instruct-GGUF": {
        "framework": "gguf", "hub_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"},
}

# The real directory listing, as gguf_variants_detail reports it.
LADDER_7B = [
    {"filename": "qwen2.5-7b-instruct-fp16-00001-of-00004.gguf",
     "bytes": 15237853536, "is_effective": False},
    {"filename": "qwen2.5-7b-instruct-q2_k.gguf",
     "bytes": 3015940000, "is_effective": False},
    {"filename": "qwen2.5-7b-instruct-q3_k_m.gguf",
     "bytes": 3808391072, "is_effective": False},
    {"filename": "qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf",
     "bytes": 448162496, "is_effective": False, "complete": False,
     "incomplete_reason": "1 of 2 shards present — missing #1",
     "shards": 1, "shard_total": 2, "missing_shards": [1]},
    {"filename": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
     "bytes": 4683073632, "is_effective": True},
    {"filename": "qwen2.5-7b-instruct-q5_0-00001-of-00002.gguf",
     "bytes": 5315176576, "is_effective": False},
    {"filename": "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
     "bytes": 5444831360, "is_effective": False},
    {"filename": "qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf",
     "bytes": 6254198880, "is_effective": False},
    {"filename": "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf",
     "bytes": 8098525408, "is_effective": False},
]

GIB = 1 << 30
# The real fleet, measured 2026-07-28.
COMPUTRON = {"id": "computron", "name": "computron",
             "vram_total": 8585740288, "vram_free": 5119148032,
             "ram_total": 16362602496, "ram_free": 8327557120,
             "has_gpu": True, "alloc_mode": "max-gpu", "alloc_explicit": False}
AE = {"id": "ae", "name": "ae",
      "vram_total": 25769803776, "vram_free": 10629414912,
      "ram_total": 134112841728, "ram_free": 96820330496,
      "has_gpu": True, "alloc_mode": "max-gpu", "alloc_explicit": False}
OP = {"id": "op", "name": "op", "vram_total": None, "vram_free": 0,
      "ram_total": 50304335872, "ram_free": 31646453760,
      "has_gpu": False, "alloc_mode": "max-gpu", "alloc_explicit": False}


def gguf_cand(worker, model_key="Qwen2.5-7B-Instruct-GGUF",
              variants=None, **kw):
    return {"model_key": model_key, "framework": "gguf",
            "hub_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "variants": LADDER_7B if variants is None else variants,
            "bytes": 4683073632, "worker": dict(worker), **kw}


def tf_cand(worker, model_key="Qwen~Qwen2.5-7B-Instruct", bytes_=15_400_000_000,
            **kw):
    return {"model_key": model_key, "framework": "transformers",
            "hub_id": "Qwen/Qwen2.5-7B-Instruct", "variants": [],
            "bytes": bytes_, "worker": dict(worker), **kw}


def group(key="qwen2.5-7b-instruct", **ticks):
    return {"group_key": key, "derived": True,
            "ticks": G.normalize_ticks(ticks), "members": []}


# ---------------------------------------------------------------------------
# 1. DERIVATION
# ---------------------------------------------------------------------------
def test_base_name_strips_packaging_and_publisher():
    assert G.base_name("unsloth/Qwen2.5-VL-7B-Instruct-GGUF") == "qwen2.5-vl-7b-instruct"
    assert G.base_name("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen2.5-vl-7b-instruct"
    # Two passes: "-NEO-Imatrix" then "-GGUF" (order is right-to-left).
    assert G.base_name("DavidAU/Foo-Thinking-NEO-Imatrix-GGUF") == "foo-thinking"
    assert G.base_name("org/Bar-i1-GGUF") == "bar"
    assert G.base_name("org/Bar-Q4_K_M.gguf".replace(".gguf", "")) == "bar"
    # The tree's publisher~Repo key form is tolerated.
    assert G.base_name("Qwen~Qwen3-Coder-Next-GGUF") == "qwen3-coder-next"


def test_the_operator_pair_is_one_group():
    groups = G.derive_groups(CATALOG)
    g = groups["qwen2.5-7b-instruct"]
    assert [m["model_key"] for m in g["members"]] == [
        "Qwen2.5-7B-Instruct-GGUF", "Qwen~Qwen2.5-7B-Instruct"]
    assert {m["framework"] for m in g["members"]} == {"gguf", "transformers"}
    # Auto-derived groups get ticks all-false. Non-negotiable default.
    assert g["ticks"] == {"quality": False, "speed": False, "priority": False}
    assert g["derived"] is True


def test_two_publishers_of_one_base_are_one_group():
    groups = G.derive_groups(CATALOG)
    assert len(groups["qwen3-coder-next"]["members"]) == 2


def test_derivation_is_conservative_about_different_bases():
    """False merges are worse than false splits — these must stay apart."""
    groups = G.derive_groups(CATALOG)
    # -4bit strips, but the BASE still differs from "...-instruct".
    assert G.base_name("mlx-community/Qwen2.5-Coder-32B-4bit") == "qwen2.5-coder-32b"
    assert "qwen2.5-coder-32b" in groups
    assert "qwen2.5-coder-32b-instruct" in groups
    assert "qwen2.5-coder-3b-instruct" in groups
    for k in ("qwen2.5-coder-32b", "qwen2.5-coder-32b-instruct",
              "qwen2.5-coder-3b-instruct"):
        assert len(groups[k]["members"]) == 1, f"{k} wrongly merged"


def test_derivation_is_deterministic():
    a = G.derive_groups(CATALOG)
    b = G.derive_groups(dict(reversed(list(CATALOG.items()))))
    assert a == b


# ---------------------------------------------------------------------------
# 2. THE OVERRIDE MAP
# ---------------------------------------------------------------------------
def test_override_sets_ticks_without_touching_membership():
    ov = {"qwen2.5-7b-instruct": {"ticks": {"quality": True}}}
    g = G.derive_groups(CATALOG, ov)["qwen2.5-7b-instruct"]
    assert g["ticks"] == {"quality": True, "speed": False, "priority": False}
    assert len(g["members"]) == 2          # membership still derived


def test_override_members_corrects_a_bad_split_and_claims_exclusively():
    """An explicit membership REPLACES the derived one, and the claimed members
    leave whatever group the deriver put them in — or a model would be in two
    groups at once and two groups' ticks would both apply to it."""
    ov = {"qwen-coder-all": {
        "members": ["Qwen2.5-Coder-32B-4bit", "Qwen2.5-Coder-32B-Instruct-GGUF"],
        "ticks": {"speed": True}}}
    groups = G.derive_groups(CATALOG, ov)
    g = groups["qwen-coder-all"]
    assert [m["model_key"] for m in g["members"]] == [
        "Qwen2.5-Coder-32B-4bit", "Qwen2.5-Coder-32B-Instruct-GGUF"]
    assert g["derived"] is False
    assert g["ticks"]["speed"] is True
    assert "qwen2.5-coder-32b" not in groups          # emptied and dropped
    assert "qwen2.5-coder-32b-instruct" not in groups
    assert "qwen2.5-coder-3b-instruct" in groups      # untouched


def test_the_enabled_flag_is_never_a_group():
    groups = G.derive_groups(CATALOG, {"enabled": True})
    assert "enabled" not in groups


def test_unknown_tick_names_are_dropped():
    """The vocabulary is closed — keeper owns nomenclature."""
    assert G.normalize_ticks({"quality": 1, "fast": True}) == {
        "quality": True, "speed": False, "priority": False}


# ---------------------------------------------------------------------------
# 3. THE LADDER
# ---------------------------------------------------------------------------
def test_ladder_is_best_quality_first_and_excludes_incomplete_shard_sets():
    rungs = G.ladder(LADDER_7B)
    quants = [r["quant"] for r in rungs]
    # The lone q4_0 shard is LITTER, never a rung.
    assert "q4_0" not in quants
    assert len(rungs) == 8
    # FULL PRECISION LAST — unconditionally. This is the shipped doctrine from
    # the 2026-07-28 election incident ("electing fp16 is a failure promise"),
    # and a model group must not re-open it. A first cut of ladder() ranked
    # purely by bit width, which put fp16 at the head and had the live fleet
    # electing the 15.2 GB split the moment a box had room for it.
    assert quants[-1] == "fp16"
    assert quants[0] == "q8_0"          # the best QUANT is the top rung
    # Everything above fp16 is descending bit width; q5_k_m before q5_0 on the
    # election tiebreak within the 5-bit class.
    assert quants[:-1] == ["q8_0", "q6_k", "q5_k_m", "q5_0", "q4_k_m",
                           "q3_k_m", "q2_k"]


def test_a_roomy_box_never_climbs_to_full_precision():
    """THE REGRESSION GUARD. ae has 10.6 GB free — enough for q8_0 (8.1 GB) but
    not fp16 (15.2 GB). Give it room for BOTH and it must still not take fp16."""
    rungs = G.ladder(LADDER_7B)
    rung, _ = G.walk_ladder(rungs, lambda n: n <= 40 * GIB)
    assert rung["quant"] == "q8_0"
    sel = G.select_member(group(speed=True),
                          [gguf_cand(dict(AE, vram_free=40 * GIB))])
    assert sel["as"] == "q8_0"


def test_walk_picks_the_best_rung_that_fits_now():
    rungs = G.ladder(LADDER_7B)
    # computron: 5.1 GB free VRAM+... use a VRAM-only budget (speed semantics).
    fits = lambda n: n <= COMPUTRON["vram_free"]  # noqa: E731
    rung, why = G.walk_ladder(rungs, fits)
    assert rung["quant"] == "q4_k_m"        # 4.68 GB — the best that fits 5.1
    assert "fits now" in why
    # ae has 10.6 GB free: q8_0 (8.1 GB) is the best that fits.
    rung, _ = G.walk_ladder(rungs, lambda n: n <= AE["vram_free"])
    assert rung["quant"] == "q8_0"


def test_quality_floor_stops_the_walk_above_the_four_bit_class():
    rungs = G.ladder(LADDER_7B)
    # A budget that only q3_k_m/q2_k would satisfy...
    tiny = lambda n: n <= 4_000_000_000  # noqa: E731
    rung, why = G.walk_ladder(rungs, tiny)
    assert rung["quant"] == "q3_k_m"        # unticked: steps all the way down
    # ...with the quality floor, the walk refuses to cross it.
    rung, why = G.walk_ladder(rungs, tiny, floor_bits=G.QUALITY_FLOOR_BITS)
    assert rung is None
    assert "quality floor" in why and "quality" in why


def test_quant_bits_and_the_degraded_class():
    assert G.quant_bits("q4_k_m") == 4
    assert G.quant_bits("iq4_xs") == 4
    assert G.quant_bits("q5_k_m") == 5
    assert G.quant_bits("q8_0") == 8
    assert G.quant_bits("fp16") == 16
    for tok in ("q4_k_m", "iq4_xs", "q4_0", "q3_k_m", "q2_k", "iq2_s", "iq1_m"):
        assert G.is_degraded_quant(tok), tok
    for tok in ("q5_k_s", "q5_k_m", "q6_k", "q8_0", "fp16", "bf16"):
        assert not G.is_degraded_quant(tok), tok
    # An unheard-of quant is still a quant — never guessed degraded.
    assert not G.is_degraded_quant("q9_z_weird")


# ---------------------------------------------------------------------------
# 4. TICK FILTERING — the motivating case
# ---------------------------------------------------------------------------
def test_quality_excludes_a_transformers_member_that_only_fits_as_4bit():
    """THE OPERATOR'S CASE. computron (4060, 8 GB / 15 GB RAM): the 7B
    transformers repo fits only as a bitsandbytes 4-bit load, so a
    quality-ticked group excludes it and the GGUF is the member."""
    cands = [gguf_cand(COMPUTRON), tf_cand(COMPUTRON, bnb=True)]
    sel = G.select_member(group(quality=True), cands)
    assert sel["model_key"] == "Qwen2.5-7B-Instruct-GGUF"
    reasons = " | ".join(s["reason"] for s in sel["skipped"])
    assert "transformers member excluded" in reasons
    assert "4-bit" in reasons and "(quality)" in reasons


def test_quality_excludes_a_prequantized_repo_by_name():
    cands = [gguf_cand(AE),
             tf_cand(AE, model_key="org~Thing-AWQ", bytes_=5_000_000_000)]
    sel = G.select_member(group(quality=True), cands)
    assert sel["model_key"] == "Qwen2.5-7B-Instruct-GGUF"
    assert any("org~Thing-AWQ" == s["model_key"] for s in sel["skipped"])


def test_untricked_group_keeps_the_4bit_member_as_a_candidate():
    """Without quality, nothing is excluded for being degraded."""
    cands = [gguf_cand(COMPUTRON), tf_cand(COMPUTRON, bnb=True)]
    sel = G.select_member(group(), cands)
    skipped_for_quality = [s for s in sel["skipped"]
                           if "(quality)" in (s["reason"] or "")]
    assert skipped_for_quality == []


def test_speed_excludes_spill_and_ram_only_and_gpu_less_boxes():
    """speed = fully GPU-resident. op has no GPU; a member too big for free
    VRAM would have to spill. Both are out."""
    big = [{"filename": "x-q8_0.gguf", "bytes": 20 * GIB, "is_effective": True}]
    cands = [gguf_cand(OP), gguf_cand(COMPUTRON, variants=big)]
    sel = G.select_member(group(speed=True), cands)
    reasons = " | ".join(s["reason"] or "" for s in sel["skipped"])
    assert "has no GPU" in reasons and "(speed)" in reasons
    # computron survives the filter but cannot fit the 20 GB rung in 5.1 GB of
    # free VRAM, so the verdict is not a clean select.
    assert sel["verdict"] != "select"


def test_speed_forbids_the_ram_spill_that_an_unticked_group_would_take():
    """The same candidate, the same box: unticked it fits (VRAM+RAM), ticked it
    does not (VRAM alone). That difference IS the speed tick."""
    mid = [{"filename": "x-q6_k.gguf", "bytes": 7 * GIB, "is_effective": True}]
    cands = [gguf_cand(COMPUTRON, variants=mid)]
    assert G.select_member(group(), cands)["verdict"] == "select"
    assert G.select_member(group(speed=True), cands)["verdict"] == "bestfit"


def test_speed_yields_to_an_explicit_operator_allocation_mode():
    """OPERATOR LEVERS OUTRANK GROUP TICKS. An EXPLICITLY-set ram-only wins;
    the tick records that it yielded instead of routing around it."""
    box = dict(COMPUTRON, alloc_mode="ram-only", alloc_explicit=True)
    sel = G.select_member(group(speed=True), [gguf_cand(box)])
    reasons = " | ".join(s["reason"] or "" for s in sel["skipped"])
    assert "speed tick yielded" in reasons and "ram-only" in reasons
    # The member is still routed — the operator's mode decided placement.
    assert sel["model_key"] == "Qwen2.5-7B-Instruct-GGUF"


def test_speed_overrides_a_DERIVED_ram_only():
    """A derived mode is not an operator statement, so the tick wins."""
    box = dict(COMPUTRON, alloc_mode="ram-only", alloc_explicit=False)
    sel = G.select_member(group(speed=True), [gguf_cand(box)])
    reasons = " | ".join(s["reason"] or "" for s in sel["skipped"])
    assert "ram-only avenue excluded (speed)" in reasons
    assert sel["verdict"] == "none"


# ---------------------------------------------------------------------------
# 5. CONTENTION AND THE PRIORITY -> DECLARE-NEED HANDOFF
# ---------------------------------------------------------------------------
def test_under_contention_an_unticked_group_steps_down_the_ladder():
    """The operator's addendum, as a test: the same box, less free VRAM, a
    lower rung. Nothing about the box's CAPACITY changed."""
    roomy = G.select_member(group(speed=True), [gguf_cand(AE)])
    assert roomy["as"] == "q8_0"
    squeezed = G.select_member(
        group(speed=True), [gguf_cand(dict(AE, vram_free=5 * GIB))])
    # Stepped DOWN to the best rung that still fits — q5_k_m (5.44 GB) is now
    # too big, q5_0 (5.32 GB) is not. No eviction was considered.
    assert squeezed["as"] == "q5_0"
    # ...and further down still, as the box gets tighter.
    tighter = G.select_member(
        group(speed=True), [gguf_cand(dict(AE, vram_free=5_000_000_000))])
    assert tighter["as"] == "q4_k_m"


def test_quality_keeps_the_contention_walk_above_the_four_bit_class():
    tight = dict(AE, vram_free=4_000_000_000)
    assert G.select_member(group(speed=True),
                           [gguf_cand(tight)])["as"] == "q3_k_m"
    sel = G.select_member(group(speed=True, quality=True), [gguf_cand(tight)])
    assert sel["as"] is None                # refused to cross the floor
    assert sel["verdict"] == "bestfit"      # no priority: soften, never evict


def test_priority_declares_a_need_instead_of_stepping_down():
    """priority forbids the step-down: declare the need and let the EXISTING
    declare-need -> evict-to-fit admission run."""
    tight = dict(COMPUTRON, vram_free=1 * GIB)
    sel = G.select_member(group(speed=True, quality=True, priority=True),
                          [gguf_cand(tight)])
    assert sel["verdict"] == "need"
    assert sel["demanded_by"] == "speed"
    # The declared need is the SMALLEST rung at or above the quality floor —
    # q5_0 (5.32 GB), not the biggest thing we would have liked. Declaring more
    # than you need is how an eviction pass evicts more than it had to.
    assert sel["need_bytes"] == 5315176576
    assert "declaring need" in sel["why"]


def test_without_priority_the_ticks_soften_to_preferences():
    tight = dict(COMPUTRON, vram_free=1 * GIB)
    sel = G.select_member(group(speed=True, quality=True), [gguf_cand(tight)])
    assert sel["verdict"] == "bestfit"
    assert "softening to best fit" in sel["why"]
    assert sel["need_bytes"] is not None     # honest about what it WOULD need
    # ...and it still routes something. Softening means serving, not refusing.
    assert sel["model_key"] == "Qwen2.5-7B-Instruct-GGUF"


def test_priority_confers_no_protection_anywhere_in_the_selection():
    """DOCTRINE: eviction protection is exactly two classes (static +
    actively-answering). A priority group's own residents gain NO shield.

    The structural proof is that a Selection has no field that could carry one:
    nothing this module produces marks a model protected, pinned, or exempt.
    If a future change adds such a field, this test fails and the reviewer is
    forced to read the doctrine before shipping it."""
    sel = G.select_member(group(priority=True), [gguf_cand(COMPUTRON)])
    forbidden = {"protect", "protected", "shield", "exempt", "pin", "pinned",
                 "static", "immune", "reserve", "reserved"}
    assert forbidden.isdisjoint(sel.keys())
    for s in sel.get("skipped") or ():
        assert forbidden.isdisjoint(s.keys())


def test_selection_is_deterministic_and_reason_bearing():
    cands = [gguf_cand(COMPUTRON), gguf_cand(AE), tf_cand(AE)]
    a = G.select_member(group(quality=True), cands)
    b = G.select_member(group(quality=True), list(reversed(cands)))
    assert (a["model_key"], a["worker_id"], a["as"]) == \
           (b["model_key"], b["worker_id"], b["as"])
    # EVERY loser carries a sentence. A skip with no reason is worse than none.
    assert a["skipped"]
    for s in a["skipped"]:
        assert (s.get("reason") or "").strip()


def test_empty_candidates_is_a_verdict_not_a_crash():
    sel = G.select_member(group(), [])
    assert sel["verdict"] == "none" and sel["model_key"] is None
    assert G.select_member(None, None)["verdict"] == "none"


# ---------------------------------------------------------------------------
# 6. TELEMETRY VOCABULARY
# ---------------------------------------------------------------------------
def test_member_stages_are_in_the_wire_vocabulary():
    from hugpy_fleet.central import evictions as ev
    assert ev.STAGE_MEMBER_SELECT == "member.select"
    assert ev.STAGE_MEMBER_SKIP == "member.skip"
    assert set(ev.GROUP_STAGES) <= set(ev.STAGES)


def test_member_events_carry_the_group_and_the_reason():
    from hugpy_fleet.central import evictions as ev
    ev.reset_for_tests()
    seen = []
    ev.register_sink(seen.append)
    try:
        ev.emit_member_select(group_key="g", model_key="m", reason="because",
                              as_="q4_k_m", ticks={"quality": True})
        ev.emit_member_skip(group_key="g", model_key="other",
                            reason="excluded: fits only as 4-bit (quality)")
    finally:
        ev.clear_sinks()
        ev.reset_for_tests()
    assert [e["stage"] for e in seen] == ["member.select", "member.skip"]
    assert seen[0]["as"] == "q4_k_m" and seen[0]["group_key"] == "g"
    assert "(quality)" in seen[1]["reason"]


def test_group_scope_tags_headroom_start_and_is_absent_by_default():
    from hugpy_fleet.central import evictions as ev
    ev.reset_for_tests()
    assert ev.current_group() is None
    with ev.group_scope("qwen2.5-7b-instruct", tick="speed"):
        g = ev.current_group()
        assert g == {"group_key": "qwen2.5-7b-instruct", "tick": "speed"}
        # Nesting restores the parent rather than clearing.
        with ev.group_scope("other", tick="quality"):
            assert ev.current_group()["group_key"] == "other"
        assert ev.current_group() == g
    assert ev.current_group() is None
    # An event built with group=None must not GROW the key — absent beats null.
    assert "group" not in ev.build_event("headroom.start", group=None)
    ev.reset_for_tests()
