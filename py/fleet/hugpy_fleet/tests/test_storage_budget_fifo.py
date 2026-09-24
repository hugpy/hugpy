"""Storage budget: evict-to-fit (FIFO), or REFUSE the pull before it starts.

The incident (2026-07-16): the operator's workstation "op" filled to 0 bytes
free. provision.py had NO disk checks — it downloaded until [Errno 28].

The operator's design, which these tests hold to:
  1. over the central-allocated budget -> FIFO the models
  2. "remove an existing model and install the one that is being called"
     (the CALLER always wins)
  3. "refuse it if it wont, show it as missing, hover info why"

⚠ This codebase has TWICE shipped tests that asserted the bug. So these tests
assert BEHAVIOR, not implementation: eviction ORDER is asserted explicitly, the
refusal asserts the download NEVER STARTED (by spying on the actual transfer
function), and the no-thrash case asserts NOTHING is evicted.

Run: venv/bin/python -m pytest tests/test_storage_budget_fifo.py -v
"""
import sys
from pathlib import Path

import json
import os
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import budget


# The reserve is carved OUT of disk_cache_gib since 2026-08-22 (effective =
# allocation - reserve). These tests reason in pure-allocation numbers, so pin
# the reserve to 0 here; the carve-out itself is covered by
# tests/test_budget_reserve_carveout.py.
@pytest.fixture(autouse=True)
def _fleet_storage_providers():
    """ensure_model_present consults hugpy_storage.providers; the worker installs
    the fleet gate at boot — do the same here so refusals reach the pull."""
    from hugpy_fleet.worker.storage_hooks import install_storage_providers, uninstall_storage_providers
    install_storage_providers()
    yield
    uninstall_storage_providers()


@pytest.fixture(autouse=True)
def _zero_disk_reserve(monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_DISK_RESERVE_GIB", "0")


GIB = 1 << 30


def _model(key, gib, last_picked=None, **flags):
    """One storage-view row, as the worker's heartbeat reports it."""
    row = {"model_key": key, "bytes": int(gib * GIB), "protected": False,
           "why": "", "pinned": False, "loaded": False, "loading": False,
           "provisioning": False, "assigned": False}
    row.update(flags)
    return row


def _static(key, gib, last_picked=None, **flags):
    """A 🔒static row — the one tier that blocks disk eviction. Used where a test
    needs an UNRECLAIMABLE model to force a refusal (📌pin no longer protects
    files as of 2026-07-17, so pinned rows are eviction candidates)."""
    flags.setdefault("protected", True)
    flags.setdefault("why", "static")
    return _model(key, gib, last_picked=last_picked, **flags)


def _storage(models):
    # disk_free is generous: fit_plan now also keeps a free-space floor as a
    # second safety check (2026-08-22); these tests are about the CAP.
    return {"cache_used_bytes": sum(m["bytes"] for m in models),
            "models": models, "disk_free": 1000 * GIB}


# ── 1. fits under budget -> no eviction, proceeds ───────────────────────────
def test_pull_that_fits_evicts_nothing_and_proceeds():
    storage = _storage([_model("old", 10), _model("warm", 10)])
    plan = budget.fit_plan("newbie", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {"old": 1, "warm": 100})
    assert plan["action"] == "proceed"
    assert plan["evict"] == []


def test_exactly_filling_the_budget_still_proceeds():
    """Boundary: used + need == cap must FIT (<=), not trip an off-by-one."""
    storage = _storage([_model("a", 30)])
    plan = budget.fit_plan("newbie", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {"a": 1})
    assert plan["action"] == "proceed"


# ── 2. doesn't fit, enough reclaimable -> FIFO oldest-first, then install ───
def test_fifo_evicts_oldest_first_and_only_as_many_as_needed():
    # 45 GiB used of a 50 GiB budget; a 20 GiB pull needs 15 GiB freed.
    storage = _storage([
        _model("oldest", 10),   # last_picked 100 -> goes 1st
        _model("middle", 10),   # last_picked 200 -> goes 2nd
        _model("newest", 25),   # last_picked 300 -> must SURVIVE
    ])
    last_picked = {"oldest": 100, "middle": 200, "newest": 300}
    plan = budget.fit_plan("caller", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, last_picked)

    assert plan["action"] == "evict"
    # ORDER is the assertion: oldest-first, and it STOPS once it fits.
    assert plan["evict"] == ["oldest", "middle"]
    assert "newest" not in plan["evict"]
    assert plan["freed_bytes"] >= plan["must_free_bytes"]


def test_never_served_models_are_coldest_and_go_first():
    """A model central never served has no last_picked -> 0 -> evicted first.
    Exactly right for never-called test-churn leftovers."""
    storage = _storage([_model("served", 20), _model("never_served", 20)])
    plan = budget.fit_plan("caller", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {"served": 500})
    assert plan["action"] == "evict"
    assert plan["evict"][0] == "never_served"


def test_equally_cold_models_break_the_tie_on_model_key_then_least_reap():
    """All-zero last_picked and zero calls -> the spec's key ④ (model_key)
    decides, and walk-then-drop keeps the set minimal.

    ⚠ RENAMED from test_equally_cold_models_evict_largest_first. SIZE IS NO
    LONGER IN THE KEY (spec 2026-07-25, assets/evictionflow.html): the old
    ``-bytes`` largest-first term cleared a budget in the fewest deletes but had
    no relationship to what the admission needed, and walk-then-drop achieves
    "fewest deletes" honestly instead. Here "big" still goes first — but now
    because "big" < "mid" < "small" alphabetically, not because it is biggest.
    Asserting a REASON, not just an outcome, is the point: this file has twice
    shipped tests that asserted the bug."""
    storage = _storage([_model("small", 5), _model("big", 30),
                        _model("mid", 15)])
    plan = budget.fit_plan("caller", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {})
    assert plan["action"] == "evict"
    # need = 50 used + 20 - 50 cap = 20 GiB. Walk in key order: big(30) covers
    # it alone, so the walk stops there and nothing is dropped.
    assert plan["evict"] == ["big"]

    # PROOF the order is by KEY, not by size: rename so the largest sorts LAST
    # alphabetically. If size were still in the key, "zbig" would lead anyway.
    storage2 = _storage([_model("aaa", 5), _model("bbb", 15), _model("zbig", 30)])
    plan2 = budget.fit_plan("caller", 20 * GIB, storage2,
                            {"disk_cache_gib": 50}, {})
    assert plan2["evict"][0] == "aaa", (
        "size must not be in the sort key — the walk starts at the "
        "lexicographically first equally-cold model")


# ── 3. never evicts protected models or the keep-target ────────────────────
# NOTE (operator ruling 2026-07-17): 📌pin is DELIBERATELY absent from this
# list — pin designates only that the allocation/routing survives restarts and
# has NO bearing on eviction. A pinned model's files are a normal candidate (see
# test_pinned_model_is_a_candidate_and_evicts below). Only 🔒static and the
# live-use guards (loaded/loading/provisioning) protect files here.
@pytest.mark.parametrize("flag", ["loaded", "loading", "provisioning"])
def test_protected_models_are_never_evicted(flag):
    storage = _storage([_model("protected_one", 40, **{flag: True}),
                        _model("cold", 20)])
    plan = budget.fit_plan("caller", 10 * GIB, storage,
                           {"disk_cache_gib": 50}, {"protected_one": 1,
                                                    "cold": 999})
    # protected_one is the OLDEST, so a policy ignoring guards would take it.
    assert "protected_one" not in plan["evict"]
    assert plan["evict"] == ["cold"]


def test_static_residency_is_never_evicted():
    """🔒static is the one tier that blocks eviction (pin-vs-static semantics)."""
    storage = _storage([_model("stat", 40, protected=True, why="static"),
                        _model("cold", 20)])
    plan = budget.fit_plan("caller", 10 * GIB, storage,
                           {"disk_cache_gib": 50}, {"stat": 1, "cold": 999})
    assert "stat" not in plan["evict"]


def test_pinned_model_is_a_candidate_and_evicts_when_fifo_reaches_it():
    """📌pin has NO bearing on eviction (operator, 2026-07-17): a pinned model's
    FILES are a normal FIFO candidate. Here the pinned model is the OLDEST, so it
    is evicted first — its pin/allocation are runtime state the fit_plan never
    touches, so nothing in the returned plan disturbs them."""
    storage = _storage([_model("pinned_old", 40, pinned=True),
                        _model("warm", 20)])
    plan = budget.fit_plan("caller", 10 * GIB, storage,
                           {"disk_cache_gib": 50},
                           {"pinned_old": 1, "warm": 999})
    assert plan["action"] == "evict"
    assert plan["evict"] == ["pinned_old"]   # pinned + oldest -> goes first
    # fit_plan is PURE — it names files to delete, never mutates pin/allocation.
    assert plan["reason"] is None


def test_pin_is_not_an_eviction_input_at_all():
    """Pin has NO eviction role whatsoever (operator ruling, 2026-07-25).

    This previously asserted the opposite — that at an exact last_picked tie the
    unpinned candidate evicts first — a tiebreak the operator had already called
    "trivial and likely unnecessary" on 2026-07-17. It is now GONE, not softened:
    _"pin routing has nothing to do with eviction... the only eviction protection
    is 'static' residency"_. Pin means exactly two things — this model is
    ALLOCATED to this worker, and that allocation SURVIVES a restart.

    A vestigial pin term is worse than none: it keeps teaching readers that pin
    is an eviction input, which is how the hot tier ended up freezing ~321 GiB of
    ae's 600 GiB NVMe budget behind pinned models that had never been called.

    With size and last_picked identical, the ONLY remaining tiebreak is the
    stable model-key sort — so "pinned" (alphabetically first) is chosen, i.e.
    being pinned demonstrably does not shield it.
    """
    storage = _storage([_model("pinned", 20, pinned=True),
                        _model("plain", 20, pinned=False)])
    plan = budget.fit_plan("caller", 15 * GIB, storage,
                           {"disk_cache_gib": 50},
                           {"pinned": 100, "plain": 100})
    assert plan["action"] == "evict"
    # Pin confers nothing: the pinned model is evicted here purely because the
    # key tiebreak lands on it. The assertion that matters is that pin did NOT
    # move it to the back of the queue.
    assert plan["evict"] == ["pinned"]


def test_the_model_being_provisioned_is_never_evicted():
    """model_cache.evict_for's keep_dir exclusion, by model_key: never evict
    the very thing we are making room for."""
    storage = _storage([_model("caller", 30, last_picked=1),
                        _model("other", 20)])
    plan = budget.fit_plan("caller", 30 * GIB, storage,
                           {"disk_cache_gib": 50}, {"caller": 1, "other": 999})
    assert "caller" not in plan["evict"]


def test_partial_bytes_of_the_keep_target_count_as_headroom():
    """A resumed pull already has bytes on disk; only the DELTA is new. Without
    this the check double-counts and refuses a pull that actually fits."""
    storage = _storage([_model("caller", 18), _model("other", 30)])
    # cap 50, used 48. A 20 GiB model with 18 already there needs only 2 more.
    plan = budget.fit_plan("caller", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {})
    assert plan["action"] == "proceed"


# ── 4. can't fit even after a full FIFO -> REFUSE, never download ──────────
def test_refuses_when_even_a_full_fifo_cannot_free_enough():
    # 🔒static (not pin) is the blocker now: pin no longer protects files, so a
    # pinned big model would simply be evicted and the pull would fit.
    storage = _storage([_static("static_big", 40),
                        _model("cold", 5)])
    plan = budget.fit_plan("huge", 30 * GIB, storage,
                           {"disk_cache_gib": 50}, {})
    assert plan["action"] == "refuse"
    assert plan["evict"] == []          # refuse -> delete NOTHING

    reason = plan["reason"]
    assert reason["state"] == "refused"
    # Honest + specific: needed, budget, reclaimable, and what blocked.
    assert "won't fit" in reason["reason"]
    assert reason["needs_bytes"] == 30 * GIB
    assert reason["budget_bytes"] == 50 * GIB
    assert reason["reclaimable_bytes"] == 5 * GIB
    assert reason["blocked"] == {"static": 1}
    assert "1 static" in reason["reason"]
    assert reason["shortfall_bytes"] > 0


def test_refusal_reason_is_human_readable_with_real_numbers():
    storage = _storage([_static("p1", 30),
                        _model("p2", 20, loaded=True)])
    plan = budget.fit_plan("big", 25 * GIB, storage, {"disk_cache_gib": 50}, {})
    assert plan["action"] == "refuse"
    text = plan["reason"]["reason"]
    assert "25.0 GB" in text and "50.0 GB" in text   # needs + budget
    assert "0 B reclaimable" in text
    assert "1 static" in text and "1 loaded" in text


def test_refused_pull_never_starts_the_download():
    """THE POINT OF THE FEATURE: no more 7%-wedged pulls that fill a disk.
    Spies on the real transfer functions — they must NEVER be called."""
    from hugpy_storage import provision

    calls = []

    class _FakeState:
        limits = {"disk_cache_gib": 50}
        model_last_picked = {}
        central_url = "http://central"
        refused: dict = {}

    # 🔒static blocker (pin no longer protects files) so the pull is REFUSED.
    storage = _storage([_static("static_big", 48)])

    orig_fetch = provision.fetch_from_central
    orig_arch = provision.fetch_archive_from_central
    orig_local = provision.model_is_local
    orig_reg = provision.ensure_model_registered
    orig_size = provision.central_total_bytes
    orig_evict = budget.evict_to_fit
    try:
        provision.fetch_from_central = lambda *a, **k: calls.append("central")
        provision.fetch_archive_from_central = lambda *a, **k: calls.append("archive")
        provision.model_is_local = lambda mk: False
        provision.ensure_model_registered = lambda mk, url: mk
        provision.central_total_bytes = lambda url, mk: 30 * GIB
        # Drive the REAL fit_plan through evict_to_fit's decision, without a disk.
        def _fake_evict(state, mk, need):
            plan = budget.fit_plan(mk, need, storage, state.limits,
                                   state.model_last_picked)
            if plan["action"] == "refuse":
                raise budget.BudgetRefusal(plan["reason"])
        budget.evict_to_fit = _fake_evict

        with pytest.raises(budget.BudgetRefusal) as err:
            provision.ensure_model_present("huge", "http://central",
                                           state=_FakeState())
        assert err.value.reason["state"] == "refused"
    finally:
        provision.fetch_from_central = orig_fetch
        provision.fetch_archive_from_central = orig_arch
        provision.model_is_local = orig_local
        provision.ensure_model_registered = orig_reg
        provision.central_total_bytes = orig_size
        budget.evict_to_fit = orig_evict

    # The assertion that matters: NOT ONE BYTE was transferred.
    assert calls == []


def test_no_state_means_no_budget_check_pure_superset():
    """Standalone/CLI provisions (state=None) behave byte-for-byte as before."""
    from hugpy_storage import provision

    seen = []
    orig_local = provision.model_is_local
    orig_reg = provision.ensure_model_registered
    orig_now = provision._provision_now
    try:
        provision.model_is_local = lambda mk: False
        provision.ensure_model_registered = lambda mk, url: mk
        provision._provision_now = lambda *a, **k: (seen.append("pulled"), True)[1]
        assert provision.ensure_model_present("x", "http://central") is True
    finally:
        provision.model_is_local = orig_local
        provision.ensure_model_registered = orig_reg
        provision._provision_now = orig_now
    assert seen == ["pulled"]


# ── 5. disk_cache_gib unset -> sane fallback, NO thrash ────────────────────
def test_unset_disk_cache_gib_does_not_evict_anything():
    """D: op has limits {"ram_max_gib":30,"threads":3} — no disk_cache_gib. On a
    100%-full drive a free-disk-reserve basis makes EVERYTHING over-budget, and
    an auto path on that basis would evict model after model on every call.
    Unset must mean 'unmanaged', never 'evict everything'."""
    storage = {"cache_used_bytes": 900 * GIB, "disk_free": 0,   # drive FULL
               "models": [_model("a", 300), _model("b", 300), _model("c", 300)]}
    for limits in ({"ram_max_gib": 30.0, "threads": 3}, {}, None,
                   {"disk_cache_gib": None}, {"disk_cache_gib": ""},
                   {"disk_cache_gib": "not-a-number"}, {"disk_cache_gib": 0}):
        plan = budget.fit_plan("caller", 50 * GIB, storage, limits, {})
        assert plan["action"] == "proceed", f"thrashed on limits={limits!r}"
        assert plan["evict"] == []


def test_cap_bytes_parses_real_values():
    assert budget.cap_bytes({"disk_cache_gib": 50}) == 50 * GIB
    assert budget.cap_bytes({"disk_cache_gib": "50"}) == 50 * GIB
    assert budget.cap_bytes({"disk_cache_gib": 0.5}) == GIB // 2
    assert budget.cap_bytes({"disk_cache_gib": -5}) is None


# ── the EXECUTOR: evict_to_fit must drive the guarded delete path ──────────
def test_evict_to_fit_executes_evictions_through_the_guarded_reaper():
    """fit_plan only PLANS. This asserts the executor actually calls the single
    guarded delete path (_reap_reclaim, which re-proves every guard per key),
    with the SHARED eviction function's victim set — not some second, divergent
    delete of its own.

    ⚠ THE VICTIM HERE IS "newest", AND THAT IS THE SPEC (2026-07-25,
    assets/evictionflow.html), not a regression. This test previously asserted
    ``reaped[0] == "oldest"`` under the old ``(last_picked, -bytes, key)`` walk.
    The spec replaces that with WALK-THEN-DROP: the walk takes "oldest" (10 GiB,
    coldest) but 10 < the 15 GiB needed, so it continues to "newest" (35 GiB);
    the drop pass then removes "oldest" because the REMAINING set covers the
    need on its own. That is least-reaping working exactly as specified — the
    fewest models unloaded, and each one only if it is genuinely required.

    Note the deliberate consequence: least reaping counts MODELS, not bytes. It
    can free more bytes than needed (35 for 15) to avoid unloading two things.
    """
    from hugpy_fleet.worker import agent

    reaped = []
    state = type("S", (), {})()
    state.limits = {"disk_cache_gib": 50}
    state.model_last_picked = {"oldest": 1, "newest": 900}
    state.refused = {}

    orig_storage, orig_reap = agent._worker_storage, agent._reap_reclaim
    orig_shared = budget._store_is_shared
    try:
        # This is the WORKER-BOX case (cap + FIFO). Pin the non-shared verdict so
        # the test does not depend on the runner's own model-root mount (the dev
        # VM's /mnt/llm_storage carries the central-catalog sentinel).
        budget._store_is_shared = lambda: False
        agent._worker_storage = lambda s: {
            "cache_used_bytes": 45 * GIB, "disk_free": 0,
            "models": [_model("oldest", 10), _model("newest", 35)]}
        agent._reap_reclaim = lambda s, keys: (
            reaped.extend(keys), {"ok": True, "freed_bytes": 10 * GIB})[1]
        budget.evict_to_fit(state, "caller", 20 * GIB)
    finally:
        agent._worker_storage, agent._reap_reclaim = orig_storage, orig_reap
        budget._store_is_shared = orig_shared

    # The shared plan's victim set reaches the real deleter — and ONLY it.
    assert reaped == ["newest"], (
        "walk-then-drop: 'oldest' was walked first but dropped once 'newest' "
        "alone covered the need (least reaping)")


def test_evict_to_fit_raises_and_deletes_nothing_when_it_cannot_fit():
    from hugpy_fleet.worker import agent

    reaped = []
    state = type("S", (), {})()
    state.limits = {"disk_cache_gib": 50}
    state.model_last_picked = {}
    state.refused = {}

    orig_storage, orig_reap = agent._worker_storage, agent._reap_reclaim
    orig_shared = budget._store_is_shared
    try:
        # Worker-box case (cap gate). Pin non-shared so the cap actually applies
        # regardless of the runner's model-root mount (see the FIFO test above).
        budget._store_is_shared = lambda: False
        agent._worker_storage = lambda s: {
            "cache_used_bytes": 48 * GIB, "disk_free": 0,
            "models": [_static("big", 48)]}   # static blocks; pin would not
        agent._reap_reclaim = lambda s, keys: (reaped.extend(keys), {})[1]
        with pytest.raises(budget.BudgetRefusal):
            budget.evict_to_fit(state, "huge", 30 * GIB)
    finally:
        agent._worker_storage, agent._reap_reclaim = orig_storage, orig_reap
        budget._store_is_shared = orig_shared

    assert reaped == []                 # a refusal deletes NOTHING


def test_a_bookkeeping_failure_never_breaks_a_working_pull():
    """If the budget check itself blows up, the pull proceeds as it did before
    this feature — a storage-accounting bug must not ground the fleet."""
    from hugpy_fleet.worker import agent

    state = type("S", (), {})()
    state.limits = {"disk_cache_gib": 50}
    state.model_last_picked = {}
    state.refused = {}
    orig = agent._worker_storage
    try:
        def _boom(s):
            raise RuntimeError("storage survey exploded")
        agent._worker_storage = _boom
        budget.evict_to_fit(state, "caller", 20 * GIB)   # must NOT raise
    finally:
        agent._worker_storage = orig


# ── C: the refusal must actually REACH the console ─────────────────────────
def test_refused_survives_centrals_storage_proposal():
    """storage_proposal REBUILDS the storage dict field-by-field, so a new
    worker-reported key is dropped unless it is explicitly carried. The console
    reads THIS dict — without the passthrough the reason never renders and the
    model is invisible rather than missing-with-a-reason."""
    from hugpy_fleet.central.workers import storage_proposal

    reason = {"state": "refused", "reason": "won't fit: needs 30.0 GB…",
              "needs_bytes": 30 * GIB}
    out = storage_proposal({
        "storage": {"cache_used_bytes": 48 * GIB, "disk_free": 2 * GIB,
                    "models": [_model("big", 48, protected=True,
                                      why="static")],
                    "refused": {"huge": reason}},
        "disk": {"free_bytes": 2 * GIB, "total_bytes": 100 * GIB},
        "limits": {"disk_cache_gib": 50},
    })
    assert out["refused"] == {"huge": reason}
    # A refused model has no files, so it must never be in models/proposals.
    assert out["proposed_evictions"] == []


def test_storage_proposal_degrades_for_a_pre_feature_worker():
    """An older agent reports no `refused` key; central must yield {} — never
    a KeyError and never a phantom refusal."""
    from hugpy_fleet.central.workers import storage_proposal

    assert storage_proposal({
        "storage": {"cache_used_bytes": 0, "disk_free": GIB, "models": []},
        "disk": {}, "limits": {},
    })["refused"] == {}
    assert storage_proposal({"disk": {}})["refused"] == {}     # no survey at all


# ── the caller always wins (the operator's rule 2), end-to-end on the plan ──
def test_caller_wins_over_many_cold_models():
    """A big pull sweeps as many cold models as it takes — the caller installs."""
    models = [_model(f"cold{i}", 10) for i in range(6)]      # 60 GiB used
    storage = _storage(models)
    lp = {f"cold{i}": i + 1 for i in range(6)}               # cold0 oldest
    plan = budget.fit_plan("caller", 30 * GIB, storage,
                           {"disk_cache_gib": 60}, lp)
    assert plan["action"] == "evict"
    assert plan["evict"] == ["cold0", "cold1", "cold2"]      # strict FIFO order


# ══ ALLOCATION-LEVEL view (operator, 2026-07-16) ═══════════════════════════
# "it should also show how much is needed based on the total size of all models
# allocated". The per-pull numbers answer "can THIS pull fit". These assert the
# STRUCTURAL question: can the ASSIGNED SET fit at all?

def _alloc(total_gib, count, unknown=0):
    """Central's allocation totals as the heartbeat reply ships them."""
    return {"allocated_total_bytes": int(total_gib * GIB),
            "allocated_count": count, "allocated_unknown_count": unknown}


def test_refusal_reports_the_allocated_set_total_and_overage():
    """1+3: the refusal carries the ASSIGNMENT-set total and how far over the
    budget it is — the operator's "how much is needed" for the WHOLE set."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    plan = budget.fit_plan("huge", 30 * GIB, storage, {"disk_cache_gib": 50},
                           {}, _alloc(180, 12))
    assert plan["action"] == "refuse"
    r = plan["reason"]
    assert r["allocated_total_bytes"] == 180 * GIB
    assert r["allocated_count"] == 12
    assert r["allocated_unknown_count"] == 0
    # 180 assigned - 50 budget = 130 over. The STRUCTURAL deficit.
    assert r["allocated_over_budget_bytes"] == 130 * GIB


def test_allocated_total_is_the_assignment_set_not_what_is_on_disk():
    """1: THE point. Lazy download means an assigned model often has NO files.
    Sizing only on-disk models would UNDER-report and make an over-subscribed
    set look fine — hiding exactly what this feature exists to show."""
    from hugpy_fleet.central import workers

    # 5 models ASSIGNED; only ONE has landed on disk.
    sizes = {"a": 40 * GIB, "b": 40 * GIB, "c": 40 * GIB, "d": 30 * GIB,
             "e": 30 * GIB}
    orig = workers._model_size_bytes
    try:
        workers._model_size_bytes = lambda mk: sizes.get(mk)
        out = workers.allocated_totals({"models": ["a", "b", "c", "d", "e"]})
    finally:
        workers._model_size_bytes = orig

    assert out["allocated_count"] == 5           # the ASSIGNMENT set, not 1
    assert out["allocated_total_bytes"] == 180 * GIB
    assert out["allocated_unknown_count"] == 0


def test_model_size_bytes_really_resolves_against_the_real_manifest():
    """NO MOCK. Every other allocation test stubs _model_size_bytes, so all of
    them passed while the REAL function returned None for every model in the
    manifest (a wrong relative-import depth, swallowed by a broad except). That
    is this repo's recurring failure: a green test asserting a dead function.
    This drives the real registry and demands a real number."""
    from hugpy_fleet.central import workers
    from hugpy_engine.config.models.models_config import get_models_dict
    from hugpy_platform.app_dirs import model_physical_json
    from hugpy_platform.constants import MODELS_DIR

    # Sizes come from the persisted physical record, or are derived cold from
    # the model files on disk. A host with neither (a CI runner; other tests
    # leave only empty family dirs behind) has nothing real to size: skip, do
    # not fake it.
    store = Path(MODELS_DIR)
    has_files = any(f.is_file() for f in store.rglob("*") if "cache" not in f.parts) if store.is_dir() else False
    try:
        with open(model_physical_json(), encoding="utf-8") as fh:
            records = (json.load(fh).get("models") or {}).values()
        has_sizes = any((r.get("fields") or {}).get("size_bytes") for r in records)
    except (OSError, ValueError, AttributeError):
        has_sizes = False
    if not has_files and not has_sizes:
        pytest.skip(f"no model files under {store} and no sized record in {model_physical_json()}: "
                    "nothing for the real sizing path to size (CI runner)")
    manifest = get_models_dict(dict_return=True) or {}
    on_disk = [k for k in manifest if workers._model_size_bytes(k)]
    assert on_disk, ("_model_size_bytes returned None for EVERY manifest model "
                     "— the sizing path is dead (import depth?), not merely cold")
    # A real, sane size — not a 0 dressed up as an answer.
    assert workers._model_size_bytes(on_disk[0]) > 0
    # An unknowable model is None (-> counted as unknown), never 0.
    assert workers._model_size_bytes("definitely/not-a-real-model") is None


def test_unknown_sizes_are_counted_never_silently_zeroed():
    """2: a model central can't size must be COUNTED and REPORTED. Silently
    treating it as 0 makes an over-subscribed set read as comfortable."""
    from hugpy_fleet.central import workers

    orig = workers._model_size_bytes
    try:
        # 2 sizable, 2 unknowable (not in the manifest / not on disk).
        workers._model_size_bytes = lambda mk: (40 * GIB if mk in ("a", "b")
                                                else None)
        out = workers.allocated_totals({"models": ["a", "b", "ghost1", "ghost2"]})
    finally:
        workers._model_size_bytes = orig

    assert out["allocated_count"] == 4
    assert out["allocated_unknown_count"] == 2   # surfaced, not swallowed
    assert out["allocated_total_bytes"] == 80 * GIB   # a FLOOR, honestly


def test_unknown_sizes_are_named_in_the_human_reason_as_a_floor():
    """2: the hover must SAY the total is incomplete — "≥" + an unknown count —
    rather than present a floor as a precise number."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    plan = budget.fit_plan("huge", 30 * GIB, storage, {"disk_cache_gib": 50},
                           {}, _alloc(180, 12, unknown=3))
    text = plan["reason"]["reason"]
    assert "≥" in text and "3 unknown" in text


def test_allocated_over_budget_is_zero_when_the_assigned_set_fits():
    """3: a set that fits reports 0 overage — a refusal is then honestly about
    THIS pull, not a structural over-subscription."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    plan = budget.fit_plan("huge", 30 * GIB, storage, {"disk_cache_gib": 50},
                           {}, _alloc(30, 2))
    r = plan["reason"]
    assert r["allocated_over_budget_bytes"] == 0
    assert "over budget" not in r["reason"].split("assigned set")[1]


def test_human_reason_states_the_allocation_total_plainly():
    """5: the operator reads the HOVER, not the JSON."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    plan = budget.fit_plan("huge", 30 * GIB, storage, {"disk_cache_gib": 50},
                           {}, _alloc(180, 12))
    text = plan["reason"]["reason"]
    assert "won't fit" in text                    # the per-pull half survives
    assert "180.0 GB" in text                     # the allocation total
    assert "130.0 GB over budget" in text         # the structural deficit
    assert "assigned set (12)" in text


def test_allocation_clause_is_omitted_when_central_has_not_said():
    """Honesty: before the first heartbeat (or against an older central) there
    are NO totals. Say nothing — never claim a 0 GiB allocation, which would
    read as "nothing is assigned" and is a lie."""
    storage = _storage([_static("p", 48)])   # static blocks the FIFO -> refusal
    for allocated in (None, {}, {"allocated_count": None}):
        plan = budget.fit_plan("huge", 30 * GIB, storage,
                               {"disk_cache_gib": 50}, {}, allocated)
        text = plan["reason"]["reason"]
        assert "assigned set" not in text
        assert "0 B" not in text.split("reclaimable")[1]
        assert "won't fit" in text               # the pull verdict still lands


def test_allocation_totals_never_change_the_fit_verdict():
    """Additive by construction: the decision is a function of REAL bytes on
    REAL disk. A wildly over-subscribed assignment must not refuse a pull that
    genuinely fits — that would ground a working fleet on bookkeeping."""
    storage = _storage([_model("a", 10)])
    plan = budget.fit_plan("newbie", 20 * GIB, storage,
                           {"disk_cache_gib": 50}, {}, _alloc(900, 30))
    assert plan["action"] == "proceed"


# ── the new fields must REACH the console (the drop-on-rebuild trap) ────────
def test_allocation_fields_survive_centrals_storage_proposal():
    """4: storage_proposal REBUILDS the dict field-by-field — the previous slice
    found it silently DROPPED `refused`. Assert the new fields make it out, or
    the console renders nothing and the operator is told nothing."""
    from hugpy_fleet.central import workers

    orig = workers._model_size_bytes
    try:
        workers._model_size_bytes = lambda mk: 60 * GIB      # 3 x 60 = 180
        out = workers.storage_proposal({
            "models": ["a", "b", "c"],           # the ASSIGNMENT set
            "storage": {"cache_used_bytes": 10 * GIB, "disk_free": 40 * GIB,
                        "models": [_model("a", 10)]},        # only ONE landed
            "disk": {"free_bytes": 40 * GIB, "total_bytes": 500 * GIB},
            "limits": {"disk_cache_gib": 50},
        })
    finally:
        workers._model_size_bytes = orig

    assert out["allocated_total_bytes"] == 180 * GIB
    assert out["allocated_count"] == 3
    assert out["allocated_unknown_count"] == 0
    assert out["allocated_over_budget_bytes"] == 130 * GIB
    # Structural truth with NO pull happening and the disk comfortably under.
    assert out["over_budget"] is False


def test_storage_proposal_allocation_degrades_for_an_unassigned_worker():
    """A worker with no assignments reports a real, honest zero — and never a
    phantom overage."""
    from hugpy_fleet.central.workers import storage_proposal

    out = storage_proposal({"storage": {"cache_used_bytes": 0, "disk_free": GIB,
                                        "models": []},
                            "disk": {}, "limits": {"disk_cache_gib": 50}})
    assert out["allocated_count"] == 0
    assert out["allocated_total_bytes"] == 0
    assert out["allocated_over_budget_bytes"] == 0


def test_worker_adopts_allocation_totals_from_the_heartbeat_reply():
    """The wire: sizing needs the manifest (central-only), so central ships the
    totals and the worker adopts them — no HTTP under the provision lock."""
    from hugpy_fleet.worker import agent

    state = type("S", (), {})()
    state.limits = {}
    state.model_last_picked = {}
    state.allocated = {}
    agent._adopt_storage_inputs(state, {
        "limits": {"disk_cache_gib": 50},
        "storage": {"allocated_total_bytes": 180 * GIB, "allocated_count": 12,
                    "allocated_unknown_count": 1},
    })
    assert state.allocated["allocated_total_bytes"] == 180 * GIB
    assert state.allocated["allocated_count"] == 12
    assert state.allocated["allocated_unknown_count"] == 1


def test_adopting_a_pre_feature_reply_leaves_allocation_untouched():
    """An older central sends no allocation totals: keep what we had, never
    clobber it with a phantom zero."""
    from hugpy_fleet.worker import agent

    state = type("S", (), {})()
    state.limits = {}
    state.model_last_picked = {}
    state.allocated = {"allocated_total_bytes": 180 * GIB,
                       "allocated_count": 12, "allocated_unknown_count": 0}
    agent._adopt_storage_inputs(state, {"limits": {"disk_cache_gib": 50},
                                        "storage": {"cache_used_bytes": 0}})
    assert state.allocated["allocated_count"] == 12       # preserved


# ── 9. the store-reapable gate is a REAL protection reason ──────────────────
# THE ae REGRESSION (2026-07-17). ae refused a 91.1 GB pull with:
#   "0 B reclaimable (1 loaded) — assigned set (65) totals 1.2 TB"
# which read as a BROKEN FIFO and sent the operator hunting the eviction policy.
# The policy was correct. The REPORT lied: _reap_scan correctly classified 101
# on-disk models as protected by the store gate (_model_store_reapable false),
# but _storage_model_row ignored its why_hint when setting `protected`, so all
# 101 shipped to central as protected=False/why="" — unprotected rows with NO
# reason. These tests hold the seam that lied.

def _row(mk, **kw):
    """Call the real _storage_model_row with the store gate's why_hint, holding
    _pinned/_residency to plain values so the row reflects the ARGUMENTS."""
    from hugpy_fleet.worker import agent

    orig_pinned, orig_res = agent._pinned, agent._residency
    try:
        agent._pinned = lambda k: kw.pop("_is_pinned", False)
        agent._residency = lambda k: kw.pop("_res", "on-demand")
        return agent._storage_model_row(
            mk, kw.get("size", 10 * GIB),
            kw.get("loaded", set()), kw.get("loading", set()),
            kw.get("provisioning", set()), kw.get("assigned", set()),
            why_hint=kw.get("why_hint", ""))
    finally:
        agent._pinned, agent._residency = orig_pinned, orig_res


def test_store_gated_row_is_reported_protected_with_its_reason():
    """A model held ONLY by the store gate (none of static/loaded/loading/
    provisioning/assigned) must ship protected=True and carry the reason."""
    row = _row("m", why_hint="model store not marked reapable")
    assert row["protected"] is True
    assert row["why"] == "model store not marked reapable"


def test_store_gated_row_is_blocked_not_silently_reclaimable():
    """End-to-end: the row the worker actually ships must land in `blocked`, so
    the refusal names the true cause instead of implying a tiny candidate set."""
    rows = [_row(f"m{i}", why_hint="model store not marked reapable",
                 size=10 * GIB) for i in range(3)]
    plan = budget.fit_plan("huge", 90 * GIB, _storage(rows),
                           {"disk_cache_gib": 100}, {})
    assert plan["action"] == "refuse"
    # The gated models are NOT counted as free bytes...
    assert plan["reason"]["reclaimable_bytes"] == 0
    assert plan["reason"]["reclaimable_count"] == 0
    # ...they are named as the blocker, with the true COUNT.
    assert plan["reason"]["blocked"] == {"model store not marked reapable": 3}
    assert "3 model store not marked reapable" in plan["reason"]["reason"]


def test_shared_central_storage_reason_also_protects():
    """The gate's other verdict — the shared/central store — is equally real."""
    row = _row("m", why_hint="shared/central storage — never reaped")
    assert row["protected"] is True
    assert budget._is_protected(row) == "shared/central storage — never reaped"


def test_store_gate_outranks_assigned_so_the_carve_out_cannot_erase_it():
    """ae's ACTUAL shape: store-gated AND assigned. `assigned` is a deliberate
    carve-out in _is_protected (routing is never a disk shield), so if it won the
    `why`, the carve-out would call this row a candidate and the refusal would
    lie again. The enforced filesystem fact must outrank the routing label."""
    row = _row("m", why_hint="model store not marked reapable",
               assigned={"m"})
    assert row["assigned"] is True          # attribution still reported
    assert row["protected"] is True
    assert row["why"] == "model store not marked reapable"
    assert budget._is_protected(row) == "model store not marked reapable"


def test_bare_assigned_row_remains_a_candidate():
    """The carve-out still holds: assignment alone is routing, not protection.
    An assigned-but-not-store-gated row must still be evictable."""
    row = _row("m", assigned={"m"})         # no why_hint -> store is reapable
    assert row["why"] == "assigned"
    assert budget._is_protected(row) == ""  # CANDIDATE
    # ...and the FIFO really does take it: "m" (10 GB, assigned) is the only
    # thing on disk, and a 15 GB caller under a 20 GB cap needs it gone.
    plan = budget.fit_plan("caller", 15 * GIB, _storage([row]),
                           {"disk_cache_gib": 20}, {})
    assert plan["action"] == "evict"
    assert plan["evict"] == ["m"]


def test_pin_still_never_protects_files_even_next_to_the_store_gate():
    """Operator ruling 2026-07-17 preserved: a pinned row on a REAPABLE store is
    still a candidate — the fix must not smuggle pin back in as a disk shield."""
    row = _row("m", _is_pinned=True)
    assert row["pinned"] is True
    assert row["protected"] is False
    assert row["why"] == "pinned"           # attribution only
    assert budget._is_protected(row) == ""  # CANDIDATE


def test_ungated_unflagged_row_is_still_a_plain_candidate():
    """Guard against over-protecting: an empty why_hint must change nothing."""
    row = _row("m", why_hint="")
    assert row["protected"] is False
    assert row["why"] == ""
    assert budget._is_protected(row) == ""


# ── 10. shared/central store: the per-worker CAP is not applicable ──────────
# THE ae DEFECT (2026-07-17). ae's DEFAULT_ROOT IS the shared catalog (13T
# volume). Its measured cache_used ≈ 693GB is the FLEET's resident catalog, not
# ae's own consumption; cap = 400GB. So used > cap permanently, every candidate
# is (correctly, post-sentinel) protected "shared/central storage — never
# reaped", and every pull refuses forever. The cap protects a WORKER'S OWN drive
# from fill-up; a shared/central volume has no such exposure (central governs
# it). Fix: on a shared store, skip the cap gate entirely — proceed, never evict
# — keeping only a disk-free floor so a genuinely full volume still refuses.
# _store_is_shared() is exercised end-to-end in evict_to_fit's own tests below;
# here we drive fit_plan directly with the flag the caller passes.

def _shared_storage(models, disk_free):
    """A storage view whose cache_used ≫ cap on purpose (the ae shape), with an
    explicit disk_free so the volume's real headroom is under test."""
    return {"cache_used_bytes": sum(m["bytes"] for m in models),
            "models": models, "disk_free": int(disk_free)}


def test_shared_store_over_cap_pull_proceeds_with_no_evictions_and_a_note():
    """ae's exact shape: 693 GB resident, 400 GB cap, all rows store-protected.
    Under the OLD code this refused forever. Now: proceed, evict NOTHING, and
    carry an honest note saying WHY the configured cap is not enforced."""
    catalog = [_static("resident_catalog", 693)]        # ≫ the 400 cap
    catalog[0]["why"] = "shared/central storage — never reaped"
    storage = _shared_storage(catalog, disk_free=4800 * GIB)   # ae: 4.8T free
    plan = budget.fit_plan("newmodel", 91 * GIB, storage,
                           {"disk_cache_gib": 400}, {}, shared_store=True)
    assert plan["action"] == "proceed"
    assert plan["evict"] == []
    assert plan["reason"] is None
    assert "shared/central" in plan["note"]
    assert "not applicable" in plan["note"]


def test_shared_store_still_refuses_when_the_real_volume_is_too_full():
    """The op incident guard survives: a shared volume that is genuinely full
    must NOT be written to [Errno 28]. A pull larger than (disk_free - reserve)
    refuses with an honest disk-free reason — and STILL evicts nothing (those
    files are the fleet's source of truth, never deletable from here)."""
    catalog = [_static("resident_catalog", 693)]
    catalog[0]["why"] = "shared/central storage — never reaped"
    # Only 10 GB free, default 50 GB reserve — a 91 GB pull cannot land.
    storage = _shared_storage(catalog, disk_free=10 * GIB)
    plan = budget.fit_plan("newmodel", 91 * GIB, storage,
                           {"disk_cache_gib": 400}, {}, shared_store=True)
    assert plan["action"] == "refuse"
    assert plan["evict"] == []                       # never evict on shared store
    reason = plan["reason"]
    assert reason["state"] == "refused"
    assert reason["shared_store"] is True
    assert reason["needs_bytes"] == 91 * GIB
    assert reason["disk_free_bytes"] == 10 * GIB
    assert "shared/central volume" in reason["reason"]
    assert "91.0 GB" in reason["reason"]             # the honest need
    # And it is a DISK-FREE refusal, NOT a cap/FIFO one:
    assert "reclaimable" not in reason["reason"]
    assert reason["note"].startswith("store is shared/central")


def test_shared_store_small_pull_within_free_space_proceeds():
    """A pull that comfortably fits the free volume proceeds even on a shared
    store — the reserve floor only bites when the volume is actually tight."""
    storage = _shared_storage([_static("resident_catalog", 693)],
                              disk_free=4800 * GIB)
    plan = budget.fit_plan("small", 5 * GIB, storage,
                           {"disk_cache_gib": 400}, {}, shared_store=True)
    assert plan["action"] == "proceed"
    assert plan["evict"] == []


def test_non_shared_box_behaviour_is_completely_unchanged():
    """Regression fence: with shared_store=False (the default and every real
    worker), the cap gate + FIFO behave EXACTLY as before this slice. A cold
    candidate under an over-budget cap is still FIFO-evicted."""
    storage = _storage([_model("cold", 40), _model("warm", 10)])
    # Default call (no shared_store kwarg) must equal the explicit-False call.
    plan_default = budget.fit_plan("caller", 30 * GIB, storage,
                                   {"disk_cache_gib": 50},
                                   {"cold": 1, "warm": 999})
    plan_explicit = budget.fit_plan("caller", 30 * GIB, storage,
                                    {"disk_cache_gib": 50},
                                    {"cold": 1, "warm": 999},
                                    shared_store=False)
    assert plan_default == plan_explicit
    assert plan_default["action"] == "evict"
    assert plan_default["evict"] == ["cold"]         # oldest, over-budget -> FIFO
    assert "note" not in plan_default or "shared" not in (plan_default.get("note") or "")


def test_disk_reserve_bytes_honours_the_existing_env_and_default():
    """The disk-free floor reuses the tree's ONE reserve knob — same env var and
    default (50 GiB) as central-side _disk_reserve_bytes — never a new constant."""
    import os
    orig = os.environ.get("HUGPY_WORKER_DISK_RESERVE_GIB")
    try:
        os.environ.pop("HUGPY_WORKER_DISK_RESERVE_GIB", None)
        assert budget.disk_reserve_bytes() == 50 * GIB       # default
        os.environ["HUGPY_WORKER_DISK_RESERVE_GIB"] = "80"
        assert budget.disk_reserve_bytes() == 80 * GIB       # override honoured
        os.environ["HUGPY_WORKER_DISK_RESERVE_GIB"] = "-5"
        assert budget.disk_reserve_bytes() == 0              # clamped >= 0
    finally:
        if orig is None:
            os.environ.pop("HUGPY_WORKER_DISK_RESERVE_GIB", None)
        else:
            os.environ["HUGPY_WORKER_DISK_RESERVE_GIB"] = orig


def test_evict_to_fit_detects_shared_store_and_never_evicts(monkeypatch):
    """End-to-end at the impure seam: when _store_is_shared() is True,
    evict_to_fit must proceed (no BudgetRefusal, no reaper call) even though the
    cache is ≫ cap and the reaper would otherwise be consulted."""
    from hugpy_fleet.worker import agent, budget as b
    from hugpy_storage import provision

    reap_calls = []
    monkeypatch.setattr(agent, "_worker_storage", lambda s: _shared_storage(
        [_static("resident_catalog", 693)], disk_free=4800 * GIB))
    monkeypatch.setattr(agent, "_reap_reclaim",
                        lambda s, keys: reap_calls.append(keys) or {"freed_bytes": 0})
    # Force the shared-store verdict through the REAL resolution helpers.
    monkeypatch.setattr(agent, "_models_store_root", lambda: "/mnt/llm_storage")
    monkeypatch.setattr(provision, "_on_shared_model_store", lambda rp: True)

    state = type("S", (), {})()
    state.limits = {"disk_cache_gib": 400}
    state.model_last_picked = {}
    state.allocated = {}
    # Must NOT raise, must NOT evict.
    b.evict_to_fit(state, "newmodel", 91 * GIB)
    assert reap_calls == []


def test_evict_to_fit_on_a_real_worker_still_uses_the_cap(monkeypatch):
    """The complement: a NON-shared box (probe False) keeps the cap + FIFO, so an
    over-budget pull still drives the guarded reaper. Proves the shared-store
    skip is gated on the probe, not applied blanket."""
    from hugpy_fleet.worker import agent, budget as b
    from hugpy_storage import provision

    reap_calls = []
    monkeypatch.setattr(agent, "_worker_storage", lambda s: _storage(
        [_model("cold", 40), _model("warm", 10)]))
    monkeypatch.setattr(agent, "_reap_reclaim",
                        lambda s, keys: reap_calls.append(keys) or {"freed_bytes": 40 * GIB})
    monkeypatch.setattr(agent, "_models_store_root", lambda: "/home/op/models")
    monkeypatch.setattr(provision, "_on_shared_model_store", lambda rp: False)

    state = type("S", (), {})()
    state.limits = {"disk_cache_gib": 50}
    state.model_last_picked = {"cold": 1, "warm": 999}
    state.allocated = {}
    b.evict_to_fit(state, "caller", 30 * GIB)
    assert reap_calls == [["cold"]]          # cap gate + FIFO, unchanged


# ── 11. central passes the scan diagnostics through verbatim (slice 3, B) ────
# The ae 2026-07-17 defect: a broken reap scan surfaced to central as rows:0,
# indistinguishable from a clean empty store, because storage_proposal rebuilds
# the storage dict field-by-field and dropped the diagnostics. These lock the
# passthrough so a broken scan can never masquerade as "nothing on disk".

def test_scan_diagnostics_survive_centrals_storage_proposal():
    from hugpy_fleet.central.workers import storage_proposal

    out = storage_proposal({
        "storage": {"cache_used_bytes": 784 * GIB, "disk_free": 900 * GIB,
                    "models": [],
                    "scan_error": "get_models_dict: discovery report unreadable",
                    "scan_keys_considered": 65, "scan_rows": 0,
                    "scan_row_errors": 0},
        "disk": {"free_bytes": 900 * GIB, "total_bytes": 1700 * GIB},
        "limits": {"disk_cache_gib": 400},
    })
    # The fingerprint of the ae failure reaches central intact: a scan that
    # considered 65 keys but produced 0 rows, WITH an error string.
    assert out["scan_error"] == "get_models_dict: discovery report unreadable"
    assert out["scan_keys_considered"] == 65
    assert out["scan_rows"] == 0


def test_scan_diagnostics_degrade_for_a_pre_slice3_worker():
    """An older agent reports none of these keys; central must yield falsy
    defaults — never a KeyError, never a phantom error string."""
    from hugpy_fleet.central.workers import storage_proposal

    out = storage_proposal({
        "storage": {"cache_used_bytes": 0, "disk_free": GIB, "models": []},
        "disk": {}, "limits": {},
    })
    assert out["scan_error"] == ""
    assert out["scan_keys_considered"] == 0
    assert out["scan_rows"] == 0
    # No survey at all -> still falsy, no crash.
    assert storage_proposal({"disk": {}})["scan_error"] == ""
