"""The worker adopts the pair's context range (ctx_min_pct) from central's
spill map and prices the admission's polite window from it (2026-10-02)."""
from hugpy_fleet.worker import agent as A


def test_adopts_ctx_min_from_the_spill_map(monkeypatch):
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "ctx_min_pct_db", {})
    state = A.WorkerState(name="t", url=None, worker_id="w")
    A._adopt_storage_inputs(state, {"spill_by_model": {"M": {"ctx_pct": 50, "ctx_min_pct": 10},
                                                       "N": {"ctx_pct": 20}}})
    assert A._ctx_min_pct("M") == 10
    assert A._ctx_min_pct("N") is None
    A._adopt_storage_inputs(state, {"spill_by_model": {}})
    assert A._ctx_min_pct("M") is None                       # a reply with the map is authoritative
