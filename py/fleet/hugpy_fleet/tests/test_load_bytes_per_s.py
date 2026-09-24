"""Heartbeat ``load_bytes_per_s``: the worker's measured central->worker
transfer rate reaches central's worker record, where the benchmark's cold-load
budget reads it (hugpy_curation.review.fleet_grading._cold_budget)."""
from hugpy_fleet.central.workers import WorkerStore

MB = 1 << 20


def _store(tmp_path):
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    w = s.register(name="aeb", url="http://aeb:9100")
    return s, w["id"]


def test_heartbeat_stores_rate_and_keeps_it_across_rateless_beats(tmp_path):
    s, wid = _store(tmp_path)
    assert "load_bytes_per_s" not in (s.heartbeat(wid, loaded_models=[]) or {})
    assert s.heartbeat(wid, load_bytes_per_s=135.0 * MB)["load_bytes_per_s"] == 135.0 * MB
    # a beat with no rate (fresh worker restart) keeps the last known one
    assert s.heartbeat(wid, load_bytes_per_s=None)["load_bytes_per_s"] == 135.0 * MB
    # garbage never overwrites it
    assert s.heartbeat(wid, load_bytes_per_s=-1)["load_bytes_per_s"] == 135.0 * MB


def test_worker_payload_helper_reads_the_provisioner_rate(monkeypatch):
    from hugpy_fleet.worker import agent
    from hugpy_storage import provision as prov
    monkeypatch.setattr(prov, "_RATE", {"bps": None, "samples": 0, "last_bps": None, "last_at": None})
    assert agent._load_bytes_per_s() is None
    prov.record_transfer(1000 * MB, 10)
    assert agent._load_bytes_per_s() == 100.0 * MB


def test_heartbeat_payload_carries_the_key():
    """The run loop's heartbeat dict names the key (read the source: the loop
    is not callable in isolation)."""
    import inspect
    from hugpy_fleet.worker import agent
    src = inspect.getsource(agent)
    assert '"load_bytes_per_s": _load_bytes_per_s(),' in src


def test_cold_budget_uses_the_rate():
    from hugpy_curation.review import fleet_grading as fg
    budgets = {"cold_load_min_s": 60.0, "cold_load_cap_s": 300.0, "cold_load_max_s": 3600.0}
    lane = {"worker_record": {"load_bytes_per_s": 100.0 * MB}, "size_bytes": 20000 * MB}
    assert fg._cold_budget(lane, budgets) == 2 * 200 + 30
