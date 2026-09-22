"""THE eviction flow — the operator's spec, asserted (assets/evictionflow.html).

Every test here names the box or invariant of the spec it holds to. The spec
is authoritative; if one of these fails, the CODE is wrong, not the test.

Run: venv/bin/python -m pytest tests/test_eviction_flow.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine import eviction as ev

GIB = 1 << 30
NOW = 1_000_000.0


def R(key, gib, *, pref=ev.VRAM, last=None, calls=0, static=False,
      mid_generation=False, since=None):
    """One resident. `last`/`since` are SECONDS AGO, for readable fixtures."""
    return ev.EvictUnit(
        model_key=key, bytes=int(gib * GIB), pref=pref,
        last_call=(NOW - last) if last is not None else None,
        calls=calls, static=static, mid_generation=mid_generation,
        resident_since=(NOW - since) if since is not None else None)


def plan(device, need_gib, residents, **kw):
    return ev.evict_plan(device, int(need_gib * GIB), residents, now=NOW, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# BOX 2 — the sort. ① pref mismatch, ② idle longest, ③ fewest calls, ④ key.
# ─────────────────────────────────────────────────────────────────────────────
def test_key1_mismatched_preference_sorts_first_the_cliff_order():
    """① A resident whose preference names the OTHER device is already off the
    cliff by design and yields FIRST — even though it is the HOTTEST and most
    called model here. A matched victim losing residency is the measured
    135->36 tok/s drop, so it sorts LAST. This is the whole rationale for ①."""
    p = plan(ev.VRAM, 5, [
        R("matched-and-freezing", 10, pref=ev.VRAM, last=99_999, calls=0),
        R("mismatched-and-hot", 10, pref=ev.RAM, last=1, calls=500),
    ])
    assert p.victims == ["mismatched-and-hot"], (
        "cliff order: the RAM-preferring resident goes first despite being "
        "hotter and more-called — it is only on this device opportunistically")


def test_key2_idle_longest_first_and_never_called_anchors_at_load():
    """② Longest since last call goes first. A NEVER-called model anchors its
    clock at LOAD time ('never-called = since load'), so a model loaded long
    ago and never called is colder than one called recently."""
    p = plan(ev.VRAM, 5, [
        R("called-recently", 10, last=10),
        R("never-called-loaded-long-ago", 10, since=50_000),
    ])
    assert p.victims == ["never-called-loaded-long-ago"]

    # And the anchor is a real clock, not a sentinel: a model loaded SECONDS
    # ago (never called) is HOTTER than one called an hour ago.
    p2 = plan(ev.VRAM, 5, [
        R("called-an-hour-ago", 10, last=3600),
        R("just-loaded-never-called", 10, since=5),
    ])
    assert p2.victims == ["called-an-hour-ago"], (
        "never-called must anchor at LOAD time, not at epoch 0 — otherwise "
        "every fresh load is instantly the coldest thing on the box")


def test_key3_fewest_calls_breaks_an_idle_tie():
    """③ Same idle time -> the model that has served FEWER calls goes first."""
    p = plan(ev.VRAM, 5, [
        R("worked-hard", 10, last=100, calls=900),
        R("barely-used", 10, last=100, calls=2),
    ])
    assert p.victims == ["barely-used"]


def test_key4_model_key_is_the_stable_final_tiebreak():
    """④ Otherwise identical residents order deterministically by model_key —
    so central and the worker cannot disagree on a coin-flip."""
    rows = [R("bbb", 10, last=100, calls=5), R("aaa", 10, last=100, calls=5)]
    assert plan(ev.VRAM, 5, rows).victims == ["aaa"]
    assert plan(ev.VRAM, 5, list(reversed(rows))).victims == ["aaa"], (
        "the key must be total — input order must not change the answer")


def test_the_key_is_lexicographic_pref_outranks_idle():
    """The four keys are LEXICOGRAPHIC, not weighted: ① is absolute. A
    mismatched resident called one second ago still precedes a matched one
    idle for a day."""
    p = plan(ev.VRAM, 5, [
        R("matched-idle-a-day", 10, pref=ev.VRAM, last=86_400),
        R("mismatched-just-called", 10, pref=ev.RAM, last=1),
    ])
    assert p.victims == ["mismatched-just-called"]


# ─────────────────────────────────────────────────────────────────────────────
# LEAST REAPING — the spec's own worked example, and the frontier rule.
# ─────────────────────────────────────────────────────────────────────────────
def test_least_reaping_y1_is_spared_when_y2_alone_covers_the_need():
    """THE SPEC'S EXAMPLE, verbatim: 'if y1 is first by order but y2 must go
    anyway and y2 alone covers the need, y1 is spared.'

    need 12. y1 (5) is first by order but insufficient, so the walk continues
    to y2 (20). The drop pass then removes y1 — the remaining set covers 12 on
    its own. One model unloaded instead of two."""
    p = plan(ev.VRAM, 12, [
        R("y1", 5, last=900), R("y2", 20, last=800), R("y3", 40, last=1),
    ])
    assert p.victims == ["y2"]
    assert p.spared == ["y1"]
    assert p.enough and p.freed == 20 * GIB


def test_the_drop_pass_never_reaches_past_the_walk_frontier():
    """THE FRONTIER RULE: 'a hot resident past the frontier is never taken just
    for being the right size.'

    need 12. The walk takes y1(5)+y2(10)=15 and STOPS. y3 is a perfect 12 GiB
    fit and would be the 'optimal' single victim — but it is HOT and past the
    frontier, so the drop pass (which only REMOVES) can never pull it in."""
    p = plan(ev.VRAM, 12, [
        R("y1", 5, last=900), R("y2", 10, last=800),
        R("perfect-fit-but-hot", 12, last=1),
    ])
    assert "perfect-fit-but-hot" not in p.victims, (
        "the drop pass only removes; it must never reach past the walk")
    assert set(p.victims) == {"y1", "y2"}


def test_the_drop_pass_can_spare_several():
    """Walk-then-drop removes EVERY redundant victim, not just the first."""
    p = plan(ev.VRAM, 30, [
        R("a", 2, last=900), R("b", 3, last=800), R("c", 4, last=700),
        R("d", 40, last=600),
    ])
    assert p.victims == ["d"]
    assert p.spared == ["a", "b", "c"]


def test_nothing_is_evicted_when_the_need_is_already_zero():
    p = plan(ev.VRAM, 0, [R("a", 10, last=900)])
    assert p.victims == [] and p.enough


# ─────────────────────────────────────────────────────────────────────────────
# FULL UNLOAD — a victim's cost is its OWN size, never a recursive function.
# ─────────────────────────────────────────────────────────────────────────────
def test_freed_is_the_sum_of_the_victims_own_sizes():
    """'A victim is unloaded entirely, not spill-chained. Its cost is its own
    size, never a function of recursive state on the other device. This is what
    keeps the choice externally derivable.' So `freed` is recomputable by hand
    from the inputs — no hidden term."""
    p = plan(ev.VRAM, 25, [R("a", 10, last=900), R("b", 20, last=800)])
    assert p.freed == 30 * GIB == sum([10 * GIB, 20 * GIB])


# ─────────────────────────────────────────────────────────────────────────────
# THE POOL — 🔒static is the only lock; 📌pin is NOT an eviction input.
# ─────────────────────────────────────────────────────────────────────────────
def test_static_is_never_a_victim_and_is_reported_as_blocking():
    p = plan(ev.VRAM, 50, [R("locked", 90, static=True, last=99_999),
                           R("free", 5, last=1)])
    assert "locked" not in p.victims
    assert not p.enough
    assert any(b["model_key"] == "locked" and "static" in b["why"]
               for b in p.blocking), "a refusal must REPORT the blocking resident"


def test_pin_is_not_an_eviction_input_at_all():
    """Operator ruling: 📌pin = routing persistence. There is no `pinned` field
    on EvictUnit, so pin CANNOT influence this function — the strongest possible
    form of the ruling (unrepresentable, not merely unused)."""
    assert not hasattr(ev.EvictUnit("x"), "pinned")
    assert "pinned" not in ev.EvictUnit.__dataclass_fields__


# ─────────────────────────────────────────────────────────────────────────────
# OPEN ITEM 1 (ENACTED PROPOSAL) — in-flight guard.
# ─────────────────────────────────────────────────────────────────────────────
def test_in_flight_is_unevictable_regardless_of_rank():
    """ENACTED PROPOSAL: 'in-flight is unevictable regardless of rank.' Here
    the in-flight model is FIRST by every key (mismatched, never called) and
    still is not touched."""
    p = plan(ev.VRAM, 5, [
        R("streaming", 10, pref=ev.RAM, mid_generation=True),
        R("idle", 10, pref=ev.VRAM, last=99_999),
    ])
    assert p.victims == ["idle"]
    assert any(b["model_key"] == "streaming" and "in flight" in b["why"]
               for b in p.blocking)


def test_in_flight_holds_even_when_it_is_the_only_candidate():
    """Unevictable means UNEVICTABLE — this is why the proposal removes it from
    the pool rather than penalising its rank. A rank penalty still evicts it
    when nothing else is available, which is the failure it exists to prevent."""
    p = plan(ev.VRAM, 5, [R("only", 10, mid_generation=True)])
    assert p.victims == [] and not p.enough


def test_last_activity_is_max_of_request_start_and_last_token():
    """ENACTED PROPOSAL: a long stream must not read as idle. The caller passes
    max(request start, last token emitted) as `last_call`; this asserts the
    CONSEQUENCE — a model whose stream started an hour ago but emitted a token
    a second ago ranks as HOT, not as an hour idle."""
    started, last_token = 3600, 1
    p = plan(ev.VRAM, 5, [
        R("long-stream", 10, last=min(started, last_token)),   # max() of ages-ago
        R("genuinely-idle", 10, last=600),
    ])
    assert p.victims == ["genuinely-idle"]


# ─────────────────────────────────────────────────────────────────────────────
# OPEN ITEM 2 — the thrash floor, RETIRED 2026-07-27 (operator: "is there still
# some timeblock on a model being evicted? if so eliminate it").
#
# These tests are inverted from what they asserted before: age must NOT veto.
# The full removal regression suite lives in test_evict_policy_knobs.py §2.
# ─────────────────────────────────────────────────────────────────────────────
def test_a_fresh_load_is_an_ordinary_candidate():
    """THE OLD THRASH SETUP, now asserting the ruling instead of the floor.

    `just-loaded` is 30s old and never called, so its anchor is 30s ago.
    `veteran` answered 10s ago. The fresh load therefore has the OLDER anchor
    and sorts FIRST, so it is the victim.

    That IS load -> evict -> reload, and it is accepted deliberately: the
    alternative — vetoing it — refuses the admission outright, which is worse
    than a reload. Freshness is expressed as RANK, never as a veto, and the two
    classes that DO block eviction are 🔒static and actively-answering.
    """
    fresh = R("just-loaded", 10, since=30)          # 30s old, never called
    busy = R("veteran", 10, last=10, calls=1000)    # answered 10s ago
    p = plan(ev.VRAM, 5, [fresh, busy])
    assert p.victims == ["just-loaded"]
    assert not any("residency" in b["why"] for b in p.blocking)


def test_a_fresh_model_can_satisfy_a_need_nothing_else_could():
    """The concrete cost the floor used to impose: with one fresh resident and a
    big need, the floor refused the admission entirely. Now it is taken."""
    p = plan(ev.VRAM, 50, [R("just-loaded", 90, since=10)])
    assert p.victims == ["just-loaded"] and p.enough
    assert p.blocking == []


def test_a_settled_model_is_also_an_ordinary_candidate():
    """Age cuts neither way — old models were never protected and still aren't."""
    p = plan(ev.VRAM, 5, [R("settled", 10, since=9_000)])
    assert p.victims == ["settled"]


# ─────────────────────────────────────────────────────────────────────────────
# DEGRADE-NOT-GUESS.
# ─────────────────────────────────────────────────────────────────────────────
def test_an_unmeasurable_resident_is_never_a_planned_victim():
    """'unmeasurable size/free/idle -> today's behaviour, never a guessed
    eviction.' Evicting an occupant of unknown size frees an unknown amount, so
    the plan could not be verified against `need`. It is reported, never walked."""
    unknown = ev.EvictUnit(model_key="mystery", bytes=None, last_call=0.0)
    p = plan(ev.VRAM, 5, [unknown, R("known", 10, last=100)])
    assert p.victims == ["known"]
    assert any(b["model_key"] == "mystery" and "unmeasurable" in b["why"]
               for b in p.blocking)


def test_a_pool_of_only_unmeasurable_residents_refuses_rather_than_guesses():
    p = plan(ev.VRAM, 5, [ev.EvictUnit(model_key="mystery", bytes=None)])
    assert p.victims == [] and not p.enough


# ─────────────────────────────────────────────────────────────────────────────
# PARITY — the function is pure, so identical inputs give identical victims.
# ─────────────────────────────────────────────────────────────────────────────
def test_the_clock_is_an_input_never_read_from_the_wall():
    """'idle times come from ONE ledger ... never each side's own clock.'
    `now` is a required keyword with no default, so neither side can silently
    substitute time.time() and drift. Asserted structurally."""
    import inspect
    sig = inspect.signature(ev.evict_plan)
    assert sig.parameters["now"].default is inspect.Parameter.empty, (
        "`now` must have NO default — a default clock read is exactly how "
        "central and the worker drift into different victim sets")


def test_identical_inputs_give_identical_victims_across_repeated_calls():
    rows = [R("a", 7, pref=ev.RAM, last=50, calls=3),
            R("b", 9, pref=ev.VRAM, last=50, calls=3),
            R("c", 4, pref=ev.VRAM, last=10, calls=99)]
    first = plan(ev.VRAM, 8, rows).victims
    for _ in range(5):
        assert plan(ev.VRAM, 8, list(reversed(rows))).victims == first


# ─────────────────────────────────────────────────────────────────────────────
# BOX 1 — admission & placement, branch by branch.
# ─────────────────────────────────────────────────────────────────────────────
def A(size_gib, mode, *, vfree, rfree, vtotal=None, rtotal=None,
      vres=(), rres=()):
    return ev.plan_admission_split(
        int(size_gib * GIB), mode,
        vram_free=int(vfree * GIB), ram_free=int(rfree * GIB),
        vram_total=(int(vtotal * GIB) if vtotal is not None else None),
        ram_total=(int(rtotal * GIB) if rtotal is not None else None),
        vram_residents=list(vres), ram_residents=list(rres),
        now=NOW)


def test_preference_names_the_device_max_gpu_vram_max_ram_ram():
    """'max-* is a DEVICE PREFERENCE.' D := P's device."""
    assert ev.preferred_device("max-gpu") == ev.VRAM
    assert ev.preferred_device("max-ram") == ev.RAM
    assert ev.other_device(ev.VRAM) == ev.RAM


def test_both_spellings_of_max_gpu_mean_the_same_preference():
    """`{}` (derived) and {"alloc_mode": "max-gpu"} (explicit, b0e02ff) differ
    in PROVENANCE, not preference. Both resolve to the mode name "max-gpu",
    and an unset/blank mode degrades to the same VRAM default."""
    assert ev.preferred_device("max-gpu") == ev.preferred_device(None) == ev.VRAM


def test_reject_when_the_model_exceeds_both_devices_combined():
    """Z > X + Y -> reject, infeasible on this card."""
    p = A(100, "max-gpu", vfree=20, rfree=60, vtotal=24, rtotal=64)
    assert p.action == "reject" and p.victims == []


def test_a_rejection_is_never_derived_from_an_unmeasured_total():
    """A rejection from a guess takes a WORKING model out of the pool, which is
    strictly worse than a late honest refusal. Both totals or no rejection."""
    p = A(100, "max-gpu", vfree=20, rfree=60, vtotal=24, rtotal=None)
    assert p.action != "reject"


def test_fits_the_preferred_device_places_all_of_z_and_evicts_nothing():
    """D_free >= Z -> place all of Z on D, resident · honored."""
    p = A(10, "max-gpu", vfree=24, rfree=64, vtotal=24, rtotal=64,
          vres=[R("neighbour", 5, last=99_999)])
    assert p.action == "place" and p.on_device == 10 * GIB
    assert p.victims == [], "nothing is evicted when the model already fits"


def test_evicting_the_preferred_device_is_enough_so_all_of_z_lands_there():
    """EVICT(D, Z - D_free); freed enough -> place all of Z on D."""
    p = A(20, "max-gpu", vfree=5, rfree=64, vtotal=24, rtotal=64,
          vres=[R("cold", 18, last=99_999), R("hot", 4, last=1)])
    assert p.action == "place" and p.on_device == 20 * GIB
    assert p.victims == ["cold"], "and only the cold one — least reaping"


def test_split_residency_when_the_preferred_device_cannot_hold_it_all():
    """Not enough freed -> place what fits on D; R := Z - placed; O_free >= R
    -> place R on O (split residency)."""
    p = A(30, "max-gpu", vfree=10, rfree=64, vtotal=24, rtotal=64)
    assert p.action == "split"
    assert p.on_device == 10 * GIB and p.on_other == 20 * GIB


def test_the_second_evict_call_is_the_same_function_on_the_other_device():
    """'Blue subroutine boxes are the one shared EVICT function — both call
    sites run the identical sort over identical inputs.' Here D cannot be
    cleared enough AND O is full, so BOTH call sites fire."""
    p = A(30, "max-gpu", vfree=10, rfree=2, vtotal=24, rtotal=64,
          rres=[R("ram-cold", 25, pref=ev.RAM, last=99_999)])
    assert p.action == "split"
    assert [e.device for e in p.evict] == [ev.VRAM, ev.RAM]
    assert p.evict[1].victims == ["ram-cold"]


def test_refuse_reports_the_blocking_residents_from_both_devices():
    """'else refuse, REPORTING THE BLOCKING RESIDENTS.'"""
    p = A(60, "max-gpu", vfree=2, rfree=2, vtotal=24, rtotal=64,
          vres=[R("vlock", 20, static=True)],
          rres=[R("rlock", 40, static=True)])
    assert p.action == "refuse"
    keys = {b["model_key"] for b in p.blocking}
    assert keys == {"vlock", "rlock"}, (
        "the operator's question is 'what is holding my box', not 'which "
        "sub-step failed' — both devices' blockers must be named")


def test_max_ram_mirrors_the_flow_with_the_devices_swapped():
    """The SAME preference that decides placement decides eviction: under
    max-ram, D is RAM, so the RAM pool is evicted first and the VRAM-preferring
    resident there is the mismatched one that yields."""
    p = A(30, "max-ram", vfree=24, rfree=5, vtotal=24, rtotal=64,
          rres=[R("gpu-pref-squatting-in-ram", 28, pref=ev.VRAM, last=1),
                R("ram-native-freezing", 28, pref=ev.RAM, last=99_999)])
    assert p.device == ev.RAM
    assert p.action == "place" and p.on_device == 30 * GIB
    assert p.victims == ["gpu-pref-squatting-in-ram"], (
        "cliff order on the RAM device: the VRAM-preferring squatter yields "
        "first even though it is far hotter")


def test_unknown_size_degrades_and_never_evicts():
    p = ev.plan_admission_split(None, "max-gpu", vram_free=0, ram_free=0,
                               now=NOW)
    assert p.action == "degrade" and p.victims == []


def test_unknown_free_on_the_target_device_degrades_and_never_evicts():
    p = ev.plan_admission_split(10 * GIB, "max-gpu", vram_free=None,
                               ram_free=64 * GIB, now=NOW)
    assert p.action == "degrade" and p.victims == []
