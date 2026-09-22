"""k56 — ORDERED WORKER PREFERENCE + POLITE (no-evict) LOAD.

Operator ruling 2026-07-31, motivated by flux2: load it on ae iff the 3090 has
genuinely free room, else on computron, else refuse — never evicting anyone to
make space. Two general per-model placement options fall out of that:

  1. Designation generalizes from ONE hard worker binding to an ORDERED
     candidate list. Resolution tries candidates in the stated order and takes
     the FIRST whose admission accepts; a model carrying a list NEVER lands off
     it (hardness is preserved per candidate); a single designation is the
     one-element degenerate case and must behave exactly as it always did.
  2. ``no_evict`` — a polite load may spend only genuinely free headroom (free
     VRAM after the tolerance-band flex) and never triggers an eviction. The
     deliberate inverse of declare-need-then-evict, which stays the rule for
     every unflagged load.

What is asserted here: the resolution order, the off-list refusal, the polite
admission on both sides of the wire (central declines to route, the worker
declines to evict), the LOUD version gate, and — the compatibility argument for
the whole slice — that a model with neither flag routes byte-identically.

Mock workers throughout; no real load ever happens.

Run: venv/bin/python -m pytest tests/test_worker_preference_polite.py -q
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-k56-test-")
os.environ.setdefault("HUGPY_COMMS_DB", "off")

from hugpy_fleet.central import workers as W
from hugpy_engine import alloc_modes as AM
from hugpy_engine.serve import overrides as OV
from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import gen_gate
# managers/__init__ star-imports shadow the subpackage attrs — bind the REAL
# module the agent uses via import_module (the dispatch module-shadowing
# landmine; a mispatched dispatch silently runs the real thing).
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30
MK = "FLUX.2-klein-9B"
NEW = AM.NO_EVICT_MIN_PKG_VERSION          # first worker that honors politeness
OLD = "0.1.225"                            # the cut before it


# ---------------------------------------------------------------------------
# Persistence — the overrides layer is the SoT for BOTH flags
# ---------------------------------------------------------------------------

@pytest.fixture()
def ov(monkeypatch, tmp_path):
    """A private serve_overrides.json for one test."""
    monkeypatch.setattr(OV, "_OVERRIDES_PATH", str(tmp_path / "serve_overrides.json"))
    return OV


def test_worker_prefs_persist_in_order(ov):
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"]})
    assert ov.get_override(MK)["worker_prefs"] == ["ae", "computron"]
    assert ov.placement_prefs(MK) == (["ae", "computron"], False)


def test_worker_prefs_dedupe_preserving_first_position(ov):
    ov.set_override(MK, {"worker_prefs": ["ae", "computron", " AE ", ""]})
    assert ov.get_override(MK)["worker_prefs"] == ["ae", "computron"]


def test_worker_prefs_accept_a_comma_string(ov):
    """curl/scripts post a string; the console posts a list. Same order."""
    ov.set_override(MK, {"worker_prefs": "ae, computron"})
    assert ov.get_override(MK)["worker_prefs"] == ["ae", "computron"]


def test_an_empty_list_clears_the_preference(ov):
    ov.set_override(MK, {"worker_prefs": ["ae"]})
    ov.set_override(MK, {"worker_prefs": []})
    assert "worker_prefs" not in ov.get_override(MK)


def test_no_evict_off_removes_the_key(ov):
    """Absent must unambiguously mean the ordinary declare-then-evict rule."""
    ov.set_override(MK, {"no_evict": True})
    assert ov.placement_prefs(MK)[1] is True
    ov.set_override(MK, {"no_evict": False})
    assert "no_evict" not in ov.get_override(MK)
    assert ov.placement_prefs(MK)[1] is False


def test_placement_prefs_is_tilde_tolerant(ov):
    """A placement set under the registry key must apply to the bare spelling —
    the k30 invisible-mismatch class, in the one place that decides where a
    call goes."""
    ov.set_override("black-forest-labs~" + MK, {"worker_prefs": ["ae"],
                                                "no_evict": True})
    assert ov.placement_prefs(MK) == (["ae"], True)


def test_an_unset_model_reads_as_pre_k56(ov):
    assert ov.placement_prefs("something-nobody-configured") == ([], False)
    assert ov.placement_policy("something-nobody-configured") == ([], False, {})


# ---------------------------------------------------------------------------
# k62 — politeness individualized per (model × worker)
# ---------------------------------------------------------------------------

def test_the_per_worker_map_persists_both_verdicts(ov):
    """Unlike the model-wide boolean, an explicit ``false`` is STORED: "polite
    on ae, ordinary eviction rights on computron" is the whole point."""
    ov.set_override(MK, {"no_evict_by_worker": {"ae": True, "computron": False}})
    assert ov.get_override(MK)["no_evict_by_worker"] == {"ae": True,
                                                         "computron": False}
    assert ov.placement_policy(MK)[2] == {"ae": True, "computron": False}


def test_the_map_accepts_the_curl_spelling(ov):
    ov.set_override(MK, {"no_evict_by_worker": "ae=yes,computron=no"})
    assert ov.get_override(MK)["no_evict_by_worker"] == {"ae": True,
                                                         "computron": False}


def test_the_map_dedupes_names_like_worker_prefs_does(ov):
    """Two spellings of one box would be two different verdicts for it."""
    ov.set_override(MK, {"no_evict_by_worker": {"ae": True, " AE ": False,
                                                "": True}})
    assert ov.get_override(MK)["no_evict_by_worker"] == {"ae": True}


def test_an_empty_map_clears_the_key(ov):
    ov.set_override(MK, {"no_evict_by_worker": {"ae": True}})
    ov.set_override(MK, {"no_evict_by_worker": {}})
    assert "no_evict_by_worker" not in ov.get_override(MK)


def test_a_junk_map_is_ignored_not_stored(ov):
    ov.set_override(MK, {"no_evict_by_worker": 17})
    assert "no_evict_by_worker" not in ov.get_override(MK)


def test_the_map_beats_the_boolean_and_the_boolean_is_the_default(ov):
    """THE resolution rule: map[W] if present, else the model-wide boolean,
    else not polite."""
    ov.set_override(MK, {"no_evict": True,
                         "no_evict_by_worker": {"computron": False}})
    assert ov.polite_on_worker(MK, "computron") is False   # map beats the bool
    assert ov.polite_on_worker(MK, "ae") is True           # bool is the default
    ov.set_override(MK, {"no_evict": False})
    assert ov.polite_on_worker(MK, "ae") is False          # absent = not polite
    assert ov.polite_on_worker(MK, "computron") is False


def test_a_map_alone_makes_only_the_listed_worker_polite(ov):
    ov.set_override(MK, {"no_evict_by_worker": {"ae": True}})
    assert ov.polite_on_worker(MK, "ae") is True
    assert ov.polite_on_worker(MK, "computron") is False


def test_polite_resolution_matches_id_or_name(ov):
    """The console posts ids, a hand-edited file carries names — a politeness
    that failed to match would evict on a box marked polite."""
    ov.set_override(MK, {"no_evict_by_worker": {"AE": True}})
    assert ov.polite_on_worker(MK, "abc123", "ae") is True
    assert ov.polite_on_worker(MK, None, "computron") is False


def test_the_per_worker_map_is_tilde_tolerant_too(ov):
    ov.set_override("black-forest-labs~" + MK,
                    {"no_evict_by_worker": {"ae": True}})
    assert ov.placement_policy(MK)[2] == {"ae": True}
    assert ov.polite_on_worker(MK, "ae") is True


def test_resolve_polite_is_pure(ov):
    """The routing loop reads the policy ONCE and resolves per candidate."""
    assert ov.resolve_polite(False, {"ae": True}, ["ae"]) is True
    assert ov.resolve_polite(True, {"ae": False}, ["ae"]) is False
    assert ov.resolve_polite(True, {"ae": False}, ["computron"]) is True
    assert ov.resolve_polite(False, {}, ["ae"]) is False


# ---------------------------------------------------------------------------
# Resolution — the ordered list
# ---------------------------------------------------------------------------

def _w(wid, name, *, pkg=NEW, free=20 * GIB, loaded=(), last_picked=0.0):
    return {
        "id": wid, "name": name, "url": f"http://{name}:9100",
        "pkg_version": pkg,
        "loaded_models": list(loaded), "allocations": [], "grants": {},
        "_wildcard_catch": True,
        "gpus": [{"memory_total": 24 * GIB, "memory_free": free}],
        "vram_free": free, "vram_total": 24 * GIB,
        "last_picked": last_picked,
    }


@pytest.fixture()
def store(monkeypatch, tmp_path, ov):
    from hugpy_fleet.central.workers import WorkerStore
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    monkeypatch.setattr(W, "worker_store", s)
    monkeypatch.setattr(W, "_assign_memory_path", lambda: str(tmp_path / "assign.json"))
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    monkeypatch.setattr(W, "_wildcard_map", lambda: {"__all__": True})
    monkeypatch.setattr(W, "_free_room_probe", None)
    return s


def _admit(store, name, *, pkg=NEW, assign=True, **beat):
    """A live, approved worker the model is DESIGNATED to — the shape the
    ordering generalizes: designated to several boxes, now with a stated order.
    """
    w = store.register(name=name, url=f"http://{name}:9100")
    store.set_admission(w["id"], "approved")
    store.heartbeat(w["id"], pkg_version=pkg, **beat)
    if assign:
        store.assign_model(w["id"], MK)
    return w["id"]


def _gpus(free=20 * GIB):
    return [{"memory_total": 24 * GIB, "memory_free": free}]


def test_the_first_listed_worker_wins(store, ov):
    """Ranking, not luck: computron is RESIDENT and least-recently-picked, and
    the operator still said ae first."""
    _admit(store, "ae", gpus=_gpus())
    _admit(store, "computron", loaded_models=[MK], gpus=_gpus())
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"]})
    assert store.pick_for_model(MK)["name"] == "ae"


def test_reversing_the_list_reverses_the_pick(store, ov):
    _admit(store, "ae", gpus=_gpus())
    _admit(store, "computron", gpus=_gpus())
    ov.set_override(MK, {"worker_prefs": ["computron", "ae"]})
    assert store.pick_for_model(MK)["name"] == "computron"


def test_a_model_with_a_list_never_lands_off_list(store, ov):
    """Designation hardness, per candidate. ae is not registered at all, so the
    honest answer is 'nowhere' — never 'computron because it was there'."""
    _admit(store, "computron", loaded_models=[MK], gpus=_gpus())
    ov.set_override(MK, {"worker_prefs": ["ae"]})
    assert store.pick_for_model(MK) is None
    assert store.candidates_for_model(MK) == []


def test_a_single_designation_is_the_degenerate_case(store, ov):
    """The compatibility promise: a one-element list picks exactly the box the
    unflagged model would have picked."""
    _admit(store, "ae", gpus=_gpus())
    _admit(store, "computron", loaded_models=[MK], gpus=_gpus())
    unflagged = store.pick_for_model(MK)["name"]
    assert unflagged == "computron"
    ov.set_override(MK, {"worker_prefs": [unflagged]})
    assert store.pick_for_model(MK)["name"] == unflagged


def test_no_list_leaves_the_rank_key_byte_identical():
    """Term ⓪ is a constant 0 for every worker when nothing is set, so the
    ordering below it is the pre-k56 ordering exactly."""
    a, c = _w("a", "ae"), _w("c", "computron", loaded=[MK])
    wanted = W._match_keys(MK)
    for w in (a, c):
        assert (W._routing_rank(w, MK, wanted, False)
                == (0,) + W._routing_rank(w, MK, wanted, False)[1:])
    assert W._routing_rank(c, MK, wanted, False) < W._routing_rank(a, MK, wanted, False)


def test_the_preference_outranks_every_derived_signal():
    wanted = W._match_keys(MK)
    resident = _w("c", "computron", loaded=[MK])
    cold = _w("a", "ae")
    prefs = ["ae", "computron"]
    assert (W._routing_rank(cold, MK, wanted, False, W._pref_index(cold, prefs))
            < W._routing_rank(resident, MK, wanted, False,
                              W._pref_index(resident, prefs)))


def test_pref_index_matches_id_or_name():
    w = _w("abc123", "ae")
    assert W._pref_index(w, ["computron", "ae"]) == 1
    assert W._pref_index(w, ["ABC123"]) == 0
    assert W._pref_index(w, ["op"]) is None


def test_the_reroute_walk_ranks_identically_to_the_pick(store, ov):
    """A reroute that ordered differently would be a re-decision."""
    _admit(store, "ae", gpus=_gpus())
    _admit(store, "computron", loaded_models=[MK], gpus=_gpus())
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"]})
    order = [w["name"] for w in store.candidates_for_model(MK)]
    assert order == ["ae", "computron"]
    assert store.pick_for_model(MK)["name"] == order[0]


# ---------------------------------------------------------------------------
# Resolution — the polite load
# ---------------------------------------------------------------------------

def _probe(fits: dict):
    """A stand-in for worker_routes._worker_fit keyed by worker name."""
    def probe(model_key, worker):
        ok = fits.get(worker.get("name"))
        if ok is None:
            return {"vram_free": None, "need": None}
        return {"vram_free": worker.get("vram_free"), "need": 9 * GIB,
                "gpu_resident": ok,
                "reason": None if ok else "won't fit: 9.0 GiB needed, 1.0 GiB free"}
    return probe


def test_polite_takes_the_first_candidate_with_free_room(store, ov, monkeypatch):
    """THE flux2 case: ae if the card is genuinely free, else computron."""
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    _admit(store, "computron", gpus=_gpus(20 * GIB))
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"], "no_evict": True})
    monkeypatch.setattr(W, "_free_room_probe",
                        _probe({"ae": False, "computron": True}))
    assert store.pick_for_model(MK)["name"] == "computron"


def test_polite_prefers_ae_the_moment_its_card_frees_up(store, ov, monkeypatch):
    _admit(store, "ae", gpus=_gpus(20 * GIB))
    _admit(store, "computron", gpus=_gpus(20 * GIB))
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"], "no_evict": True})
    monkeypatch.setattr(W, "_free_room_probe",
                        _probe({"ae": True, "computron": True}))
    assert store.pick_for_model(MK)["name"] == "ae"


def test_polite_refuses_when_no_candidate_admits_without_eviction(store, ov,
                                                                  monkeypatch):
    from hugpy_fleet.central import evictions
    evictions.reset_for_tests()
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    _admit(store, "computron", gpus=_gpus(1 * GIB))
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"], "no_evict": True})
    monkeypatch.setattr(W, "_free_room_probe",
                        _probe({"ae": False, "computron": False}))
    assert store.pick_for_model(MK) is None
    evs = [e for e in evictions.recent(50) if e.get("stage") == "route.refuse"]
    assert len(evs) == 1
    assert evs[0]["reason"] == "no candidate admits without eviction"
    assert {a["worker"] for a in evs[0]["alternatives"]} == {"ae", "computron"}


def test_the_same_worker_still_evicts_for_an_UNFLAGGED_model(store, ov,
                                                             monkeypatch):
    """The doctrine is unchanged for everyone else: no free room is not a
    refusal for a model that never asked to be polite."""
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    monkeypatch.setattr(W, "_free_room_probe", _probe({"ae": False}))
    assert store.pick_for_model(MK)["name"] == "ae"


def test_polite_fails_open_when_free_room_is_unprovable(store, ov, monkeypatch):
    """Central can only ever prove the NEGATIVE. An unsizable model must reach
    the worker, whose measured admission makes the real call — refusing on an
    unproven guess would strand a model that would have fitted."""
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    ov.set_override(MK, {"no_evict": True})
    monkeypatch.setattr(W, "_free_room_probe", _probe({}))   # returns no numbers
    assert store.pick_for_model(MK)["name"] == "ae"
    monkeypatch.setattr(W, "_free_room_probe", None)         # nothing registered
    assert store.pick_for_model(MK)["name"] == "ae"


def test_a_polite_model_is_not_routed_to_a_worker_that_predates_the_flag(
        store, ov, monkeypatch):
    """The LOUD downgrade: an old worker would evict residents, so it is not a
    candidate at all rather than silently dropping the promise."""
    _admit(store, "ae", pkg=OLD, gpus=_gpus(20 * GIB))
    ov.set_override(MK, {"no_evict": True})
    monkeypatch.setattr(W, "_free_room_probe", _probe({"ae": True}))
    assert store.pick_for_model(MK) is None
    ok, why = W._polite_admits({"name": "ae", "pkg_version": OLD}, MK)
    assert ok is False and NEW in why


# ---------------------------------------------------------------------------
# k62 — per-worker politeness in the resolution
# ---------------------------------------------------------------------------

def test_polite_on_ae_assertive_on_computron_lands_on_computron(store, ov,
                                                                monkeypatch):
    """THE k62 case. Neither card has free room; ae is polite so it is skipped,
    computron keeps ordinary eviction rights so it takes the model."""
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    _admit(store, "computron", gpus=_gpus(1 * GIB))
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"],
                         "no_evict_by_worker": {"ae": True, "computron": False}})
    monkeypatch.setattr(W, "_free_room_probe",
                        _probe({"ae": False, "computron": False}))
    assert store.pick_for_model(MK)["name"] == "computron"


def test_the_polite_worker_still_wins_when_its_own_card_is_free(store, ov,
                                                                monkeypatch):
    _admit(store, "ae", gpus=_gpus(20 * GIB))
    _admit(store, "computron", gpus=_gpus(1 * GIB))
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"],
                         "no_evict_by_worker": {"ae": True}})
    monkeypatch.setattr(W, "_free_room_probe",
                        _probe({"ae": True, "computron": False}))
    assert store.pick_for_model(MK)["name"] == "ae"


def test_a_per_worker_exemption_overrides_the_all_workers_toggle(store, ov,
                                                                 monkeypatch):
    """The boolean says polite everywhere; the map exempts computron, so a full
    computron still takes the model rather than holding."""
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    _admit(store, "computron", gpus=_gpus(1 * GIB))
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"], "no_evict": True,
                         "no_evict_by_worker": {"computron": False}})
    monkeypatch.setattr(W, "_free_room_probe",
                        _probe({"ae": False, "computron": False}))
    assert store.pick_for_model(MK)["name"] == "computron"


def test_it_still_holds_when_every_candidate_is_polite_and_full(store, ov,
                                                                monkeypatch):
    from hugpy_fleet.central import evictions
    evictions.reset_for_tests()
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    _admit(store, "computron", gpus=_gpus(1 * GIB))
    ov.set_override(MK, {"worker_prefs": ["ae", "computron"],
                         "no_evict_by_worker": {"ae": True, "computron": True}})
    monkeypatch.setattr(W, "_free_room_probe",
                        _probe({"ae": False, "computron": False}))
    assert store.pick_for_model(MK) is None
    evs = [e for e in evictions.recent(50) if e.get("stage") == "route.refuse"]
    # The hold names the WORKER whose politeness caused each skip.
    assert all("polite on" in a["reason"] for a in evs[0]["alternatives"])
    assert {a["worker"] for a in evs[0]["alternatives"]} == {"ae", "computron"}


def test_the_reroute_walk_keeps_the_exempt_worker(store, ov, monkeypatch):
    """A reroute that dropped computron would be a re-decision: the operator
    never asked it to be polite."""
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    _admit(store, "computron", gpus=_gpus(1 * GIB))
    ov.set_override(MK, {"no_evict": True,
                         "no_evict_by_worker": {"computron": False}})
    monkeypatch.setattr(W, "_free_room_probe",
                        _probe({"ae": False, "computron": False}))
    assert [w["name"] for w in store.candidates_for_model(MK)] == ["computron"]


def test_an_old_worker_is_only_excluded_where_it_is_polite(store, ov,
                                                           monkeypatch):
    """The version gate is part of the POLITE path, so a worker the model is
    not polite on is unaffected by it — it was never promised anything."""
    _admit(store, "ae", pkg=OLD, gpus=_gpus(1 * GIB))
    ov.set_override(MK, {"no_evict": True})
    monkeypatch.setattr(W, "_free_room_probe", _probe({"ae": True}))
    assert store.pick_for_model(MK) is None
    ov.set_override(MK, {"no_evict_by_worker": {"ae": False}})
    assert store.pick_for_model(MK)["name"] == "ae"


def test_no_map_and_no_flag_routes_byte_identically(store, ov, monkeypatch):
    """The compatibility argument, re-stated for k62: an unconfigured model
    never enters the polite walk at all."""
    _admit(store, "ae", gpus=_gpus(1 * GIB))
    monkeypatch.setattr(W, "_free_room_probe", _probe({"ae": False}))
    before = store.pick_for_model(MK)["name"]
    ov.set_override(MK, {"no_evict_by_worker": {"computron": True}})
    assert store.pick_for_model(MK)["name"] == before


# ---------------------------------------------------------------------------
# The wire — version-gated emission
# ---------------------------------------------------------------------------

def test_the_flag_rides_the_spill_to_a_new_worker(store, ov):
    wid = _admit(store, "ae", gpus=_gpus())
    store.assign_model(wid, MK, spill={})
    ov.set_override(MK, {"no_evict": True})
    assert store.spill_for(wid, MK).get("no_evict") is True


def test_the_flag_never_reaches_an_old_worker(store, ov):
    wid = _admit(store, "ae", pkg=OLD, gpus=_gpus())
    store.assign_model(wid, MK, spill={})
    ov.set_override(MK, {"no_evict": True})
    assert "no_evict" not in store.spill_for(wid, MK)


def test_an_unflagged_model_emits_a_byte_identical_spill(store, ov):
    wid = _admit(store, "ae", gpus=_gpus())
    store.assign_model(wid, MK, spill={"alloc_mode": "max-ram"})
    before = store.spill_for(wid, MK)
    ov.set_override(MK, {"worker_prefs": ["ae"]})      # order alone changes nothing
    assert store.spill_for(wid, MK) == before


def test_the_spill_carries_the_flag_only_to_the_polite_worker(store, ov):
    """k62 at the wire: central includes/omits the key PER CANDIDATE; the
    worker side and the wire key itself are exactly as k56 left them."""
    ae = _admit(store, "ae", gpus=_gpus())
    comp = _admit(store, "computron", gpus=_gpus())
    store.assign_model(ae, MK, spill={})
    store.assign_model(comp, MK, spill={})
    ov.set_override(MK, {"no_evict": True,
                         "no_evict_by_worker": {"computron": False}})
    assert store.spill_for(ae, MK).get("no_evict") is True
    assert "no_evict" not in store.spill_for(comp, MK)


def test_a_map_only_model_ships_the_flag_to_its_listed_worker(store, ov):
    ae = _admit(store, "ae", gpus=_gpus())
    comp = _admit(store, "computron", gpus=_gpus())
    store.assign_model(ae, MK, spill={})
    store.assign_model(comp, MK, spill={})
    ov.set_override(MK, {"no_evict_by_worker": {"ae": True}})
    assert store.spill_for(ae, MK).get("no_evict") is True
    assert "no_evict" not in store.spill_for(comp, MK)


def test_the_gate_strips_no_evict_loudly_and_keeps_the_placement():
    """A polite spill aimed at an old worker loses the FLAG, not the mode — and
    the note says which knob died. Deliberately not in NEW_SPILL_KEYS: collapsing
    the spill to {} would keep the placement and silently drop the politeness,
    the exact silent no-op the gate exists to prevent."""
    spill = {"alloc_mode": "max-ram", "no_evict": True}
    out, note = AM.gate_spill_for_worker(spill, NEW, "ae")
    assert out == spill and note is None
    out, note = AM.gate_spill_for_worker(spill, "0.1.210", "ae")
    assert out == {"alloc_mode": "max-ram"}
    assert "no_evict STRIPPED" in note and AM.NO_EVICT_MIN_PKG_VERSION in note
    # An ancient worker misses BOTH; the note must name both downgrades.
    out, note = AM.gate_spill_for_worker(spill, "0.1.190", "ae")
    assert out == {}
    assert "no_evict STRIPPED" in note and "downgraded to max-gpu" in note


def test_a_polite_spill_with_no_mode_keys_is_still_gated():
    out, note = AM.gate_spill_for_worker({"no_evict": True}, "0.1.210", "ae")
    assert out == {} and "no_evict STRIPPED" in note


def test_no_evict_is_not_one_of_the_mode_keys():
    assert AM.NO_EVICT_SPILL_KEY not in AM.NEW_SPILL_KEYS


# ---------------------------------------------------------------------------
# Worker-side admission — politeness is a promise the worker keeps
# ---------------------------------------------------------------------------

class _State:
    pass


@pytest.fixture()
def rig(monkeypatch):
    """The same shape as test_vram_evict_to_fit's rig: a card with a mutable
    free cell, a resident set, and an evict verb that gives the bytes back."""
    card = {"total": 24 * GIB, "free": 0, "need": 0}
    residents = {}
    evicted_calls = []
    for _leak in ("HUGPY_GPU_MEM_GIB", "HUGPY_CPU_MEM_GIB", "HUGPY_ALLOC_MODE",
                  "HUGPY_LENIENCY_PCT", "HUGPY_PRIORITY_DEVICE", "HUGPY_BNB_4BIT",
                  "HUGPY_N_GPU_LAYERS", "HUGPY_VRAM_CEILING_FRAC",
                  "HUGPY_VRAM_RESERVE_GIB", "HUGPY_VRAM_CEILING_CUSHION_GIB",
                  "HUGPY_EVICT_MIN_RESIDENCY_S", "HUGPY_EVICT_LEAST_REAPING",
                  "HUGPY_NO_EVICT"):
        monkeypatch.delenv(_leak, raising=False)
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: card["need"])
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v["vram_bytes"],
                                    "host_mode": v["host_mode"], "alive": True}
                                   for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(gen_gate, "in_flight", lambda mk: 0)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(D, "last_used_snapshot", lambda: {k: 1.0 for k in residents})

    def _fake_evict(state, mk, force=False):
        evicted_calls.append(mk)
        row = residents.pop(mk, None)
        card["free"] += row["vram_bytes"] if row else 0
        return {"model_key": mk, "evicted": bool(row),
                "vram_freed": row["vram_bytes"] if row else None,
                "host_mode": row["host_mode"] if row else "none"}
    monkeypatch.setattr(A, "_evict_model", _fake_evict)
    A._VRAM_EVICTIONS.update(count=0, last=None, last_at=0.0)
    return type("Rig", (), {"card": card, "residents": residents,
                            "evicted": evicted_calls})()


def _squatted(rig):
    """One idle 20 GiB resident, 1 GiB free, a 10 GiB subject: an ordinary load
    evicts to fit here."""
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 10 * GIB
    rig.residents["idle-neighbour"] = {"vram_bytes": 20 * GIB,
                                       "host_mode": "subprocess"}


def test_an_unflagged_load_still_evicts_to_fit(rig):
    _squatted(rig)
    plan = A._vram_evict_to_fit(_State(), MK)
    assert plan["action"] == "evicted"
    assert rig.evicted == ["idle-neighbour"]


def test_a_polite_load_refuses_rather_than_evicting(rig, monkeypatch):
    monkeypatch.setenv("HUGPY_NO_EVICT", "1")
    _squatted(rig)
    plan = A._vram_evict_to_fit(_State(), MK)
    assert plan["action"] == "refuse"
    assert rig.evicted == []                      # the neighbour is untouched
    assert "idle-neighbour" in rig.residents


def test_the_polite_refusal_names_the_flag_and_what_it_spared(rig, monkeypatch):
    """"Honest refusal" means the operator can read WHY it didn't land — and
    that the room it could have taken is named, never silently reported as
    'nothing was evictable'."""
    monkeypatch.setenv("HUGPY_NO_EVICT", "1")
    _squatted(rig)
    reason = A._vram_evict_to_fit(_State(), MK)["reason"]
    assert reason["no_evict"] is True
    assert [r["model_key"] for r in reason["polite_spared"]] == ["idle-neighbour"]
    assert "POLITE LOAD (no_evict)" in reason["reason"]
    assert "SPARED" in reason["reason"]
    assert "evicted 0 idle resident(s)" in reason["reason"]


def test_a_polite_load_that_fits_free_room_just_proceeds(rig, monkeypatch):
    monkeypatch.setenv("HUGPY_NO_EVICT", "1")
    rig.card["free"] = 20 * GIB
    rig.card["need"] = 10 * GIB
    rig.residents["idle-neighbour"] = {"vram_bytes": 2 * GIB,
                                       "host_mode": "subprocess"}
    plan = A._vram_evict_to_fit(_State(), MK)
    assert plan["action"] == "proceed"
    assert rig.evicted == []
    assert "polite load" in plan["note"]


def test_the_polite_flag_is_cleared_between_requests(monkeypatch):
    """A leaked flag would make the NEXT model refuse instead of making room —
    dead-wrong in the other direction, so it clears like every mode key.

    monkeypatch.setenv first so pytest OWNS both keys and restores them: this
    test drives the real process-wide env writer, and k51 (cross-file env
    pollution) is exactly what an unrestored HUGPY_N_GPU_LAYERS causes.
    """
    monkeypatch.setenv("HUGPY_NO_EVICT", "")
    monkeypatch.setenv("HUGPY_N_GPU_LAYERS", "")
    assert "no_evict" in A._SPILL_ENV_CLEAR_WHEN_ABSENT
    A._apply_spill({"no_evict": True})
    assert os.environ.get("HUGPY_NO_EVICT") == "True"
    A._apply_spill({"n_gpu_layers": -1})
    assert "HUGPY_NO_EVICT" not in os.environ


def test_the_in_process_contention_yield_is_skipped_when_polite(monkeypatch):
    """dispatch's headroom pass exists to TAKE room from residents."""
    monkeypatch.setenv("HUGPY_NO_EVICT", "1")
    monkeypatch.setattr(D, "_FIT_CHECK", lambda mk: False)   # never fits
    monkeypatch.setattr(D, "_MAKE_ROOM", None)
    yielded = []
    monkeypatch.setattr(D, "_next_lru_evictable",
                        lambda exclude=None, run_id=None: yielded.append(1))
    assert D.ensure_headroom_for_load(MK) == []
    assert yielded == []


# ---------------------------------------------------------------------------
# Warm / reconcile — a polite model is never warmed by evicting
# ---------------------------------------------------------------------------

def _routes():
    from hugpy_server.app.routes import worker_routes
    return worker_routes


def test_a_polite_model_is_not_warmed_onto_a_full_worker(ov, monkeypatch):
    R = _routes()
    ov.set_override(MK, {"no_evict": True})
    monkeypatch.setattr(R, "_worker_fit", lambda mk, w: {
        "vram_free": 1 * GIB, "need": 9 * GIB, "gpu_resident": False})
    assert R._polite_warm_ok({"id": "a", "name": "ae"}, MK) is False


def test_a_polite_model_IS_warmed_when_the_room_is_free(ov, monkeypatch):
    R = _routes()
    ov.set_override(MK, {"no_evict": True})
    monkeypatch.setattr(R, "_worker_fit", lambda mk, w: {
        "vram_free": 20 * GIB, "need": 9 * GIB, "gpu_resident": True})
    assert R._polite_warm_ok({"id": "a", "name": "ae"}, MK) is True


def test_an_unflagged_model_is_warmed_exactly_as_before(ov, monkeypatch):
    R = _routes()
    monkeypatch.setattr(R, "_worker_fit", lambda mk, w: {
        "vram_free": 1 * GIB, "need": 9 * GIB, "gpu_resident": False})
    assert R._polite_warm_ok({"id": "a", "name": "ae"}, MK) is True


def test_the_warm_gate_resolves_politeness_per_worker(ov, monkeypatch):
    """k62: same full card, two verdicts — ae is left cold, computron is warmed
    on the ordinary rule."""
    R = _routes()
    ov.set_override(MK, {"no_evict": True,
                         "no_evict_by_worker": {"computron": False}})
    monkeypatch.setattr(R, "_worker_fit", lambda mk, w: {
        "vram_free": 1 * GIB, "need": 9 * GIB, "gpu_resident": False})
    assert R._polite_warm_ok({"id": "a", "name": "ae"}, MK) is False
    assert R._polite_warm_ok({"id": "c", "name": "computron"}, MK) is True


def test_the_warm_gate_fails_open_on_an_unsizable_model(ov, monkeypatch):
    R = _routes()
    ov.set_override(MK, {"no_evict": True})
    monkeypatch.setattr(R, "_worker_fit", lambda mk, w: {
        "vram_free": None, "need": None, "gpu_resident": False})
    assert R._polite_warm_ok({"id": "a", "name": "ae"}, MK) is True
