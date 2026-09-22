"""Fleet adapters satisfy the engine placement Protocols and are backed by a
tmpdir registry; install()/uninstall() wire and unwire the engine seam."""

from __future__ import annotations

import contextlib

import pytest

from hugpy_control.settings import settings_store
from hugpy_engine import placement as S
from hugpy_fleet.central import blocklist, evictions, model_metrics, placement as P, priority_groups
from hugpy_fleet.central import workers as W

from worker_store_isolation import swap_worker_store


@pytest.fixture
def isolated_settings(tmp_path):
    orig_path, orig_cache = settings_store._path, settings_store._cache
    settings_store._path = str(tmp_path / "settings.json")
    settings_store._cache = None
    try:
        yield settings_store
    finally:
        settings_store._path, settings_store._cache = orig_path, orig_cache


@pytest.fixture
def fleet(isolated_settings):
    with swap_worker_store(prefix="hugpy-placement-") as store:
        store.register(name="a", url="http://a:9100", worker_id="wk-a", models=["Org~Alpha"])
        store.set_admission("wk-a", "approved")
        store.register(name="b", url="http://b:9100", worker_id="wk-b", models=["Org~Beta"])
        store.set_admission("wk-b", "approved")
        store.heartbeat("wk-b", loaded_models=["Org~Beta"])
        yield store


@pytest.mark.parametrize("cls,proto", [
    (P.FleetWorkerRegistry, S.WorkerRegistry),
    (P.FleetWorkerTransport, S.WorkerTransport),
    (P.FleetEvictionLedger, S.EvictionLedger),
    (P.FleetBlocklist, S.Blocklist),
    (P.FleetModelMetrics, S.ModelMetrics),
    (P.FleetPriorityGroups, S.PriorityGroups),
])
def test_adapters_satisfy_protocols(cls, proto):
    assert isinstance(cls(), proto)


def test_registry_reads_tmp_backed_store(fleet):
    reg = P.FleetWorkerRegistry()
    ids = {w["id"] for w in reg.list_workers()}
    assert ids == {"wk-a", "wk-b"}
    assert reg.get_worker("wk-a")["name"] == "a"
    assert reg.get_worker("nope") is None
    assert [w["id"] for w in reg.workers_for_model("Org~Beta")] == ["wk-b"]
    assert {w["id"] for w in reg.workers_for_model("Alpha", ready_only=False)} == {"wk-a"}
    assert reg.match_keys("Org~Alpha", ["alpha", "Other~Alpha", "Org~Gamma", "x"]) == ("alpha", "Other~Alpha")
    assert reg.workers_for_model("") == ()
    bound = P.FleetWorkerRegistry(store=fleet)
    assert {w["id"] for w in bound.list_workers(online_only=False)} == ids


def test_transport_uses_worker_http(monkeypatch):
    from hugpy_fleet.central import worker_http
    seen = []

    class _Resp:
        text = "{bad"
        def json(self):
            return {"ok": True}

    def fake_request(method, worker, path, *, call="control", force=False, read_timeout=None, **kw):
        seen.append((method, worker["url"], path, call, read_timeout, kw.get("json")))
        return _Resp()

    @contextlib.contextmanager
    def fake_stream(method, worker, path, *, call="relay", force=False, **kw):
        seen.append(("STREAM", worker["url"], path, call, kw.get("json")))
        class _S:
            def iter_bytes(self):
                yield b"data: 1\n"
                yield b""
                yield b"data: 2\n"
        yield _S()

    monkeypatch.setattr(worker_http, "request", fake_request)
    monkeypatch.setattr(worker_http, "stream", fake_stream)
    t = P.FleetWorkerTransport()
    w = {"id": "wk", "url": "http://wk:9100/"}
    assert t.get_json(w, "/health", timeout=3.0) == {"ok": True}
    assert t.post_json(w, "/ops/evict", {"model_key": "m"}, timeout=9.0) == {"ok": True}
    assert list(t.stream(w, "/infer/stream", {"p": 1})) == [b"data: 1\n", b"data: 2\n"]
    assert seen[0] == ("GET", "http://wk:9100/", "/health", "control", 3.0, None)
    assert seen[1] == ("POST", "http://wk:9100/", "/ops/evict", "control", 9.0, {"model_key": "m"})
    assert seen[2] == ("STREAM", "http://wk:9100/", "/infer/stream", "relay", {"p": 1})


def test_transport_breaker_surface():
    from hugpy_fleet.central import worker_http
    worker_http.reset_breakers()
    t = P.FleetWorkerTransport()
    w = {"id": "wk-x", "url": "http://wk-x:9100/"}
    assert t.base_url(w) == "http://wk-x:9100"
    assert t.breaker_key(w) == "wk-x"
    assert t.transport_errors == tuple(worker_http.TRANSPORT_ERRORS)
    t.guard("wk-x", url=t.base_url(w))          # closed breaker: no raise
    t.note_failure("wk-x", RuntimeError("boom"))
    t.note_ok("wk-x")
    assert isinstance(t.breaker_snapshot(), dict)
    with t.breaker_scope(w) as key:
        assert key == "wk-x"
    client = t.async_client("relay")
    assert hasattr(client, "aclose")
    import asyncio
    asyncio.run(client.aclose())
    worker_http.reset_breakers()


def test_eviction_ledger_records_and_reads_ring():
    evictions.reset_for_tests()
    orig = evictions.get_store()
    evictions.set_store(evictions.EvictionStore(path="off"))
    try:
        led = P.FleetEvictionLedger()
        led.record({"stage": evictions.STAGE_EVICT_DONE, "model_key": "m1"})
        led.record({"model_key": "m2"})
        with led.run_scope() as rid:
            assert rid and evictions.current_run_id() == rid
            ev = led.emit(evictions.STAGE_FIT_FAIL, model_key="m3")
            assert ev and ev["run_id"] == rid
        led.emit_resolve_fail("m4", None, "no such model")
        rows = led.recent(limit=10)
        assert [r["stage"] for r in rows] == [
            evictions.STAGE_EVICT_DONE, "eviction.event", evictions.STAGE_FIT_FAIL,
            evictions.STAGE_RESOLVE_FAIL]
        assert rows[0]["model_key"] == "m1"
        assert led.new_run_id()
        assert isinstance(led.disk_stats("/"), dict)
        assert led.current_group() is None
    finally:
        evictions.set_store(orig)
        evictions.reset_for_tests()


def test_blocklist_and_priority_groups(isolated_settings):
    bl = P.FleetBlocklist()
    assert bl.blocked_keys() == ()
    blocklist.block("Org~Bad", by="test", note="broken quant")
    try:
        assert bl.blocked_keys() == ("Org~Bad",)
        assert blocklist.BLOCKED_MARKER in (bl.block_reason("Org~Bad") or "")
        assert bl.block_reason("Org~Good") is None
    finally:
        blocklist.unblock("Org~Bad")
    pg = P.FleetPriorityGroups()
    assert pg.workers_for_key("Org~Alpha") == ()
    priority_groups.put_group("g1", name="G1", members=["Org~Alpha"], enabled=True, workers=["wk-a"])
    try:
        assert pg.workers_for_key("Org~Alpha") == ("wk-a",)
    finally:
        priority_groups.delete_group("g1")


def test_model_metrics_tmp_sqlite(tmp_path):
    store = model_metrics.ModelMetricsStore(path=str(tmp_path / "metrics.db"))
    mm = P.FleetModelMetrics(store=store)
    assert mm.stats("Org~Alpha") == {} and mm.get_call("Org~Alpha") is None
    assert mm.record_call("Org~Alpha", 12.5, task="chat", compute_s=1.0, worker_id="wk-a") is True
    stats = mm.stats("Org~Alpha")
    assert stats and int(stats.get("n_calls") or 0) >= 1
    assert mm.record_call("Org~Alpha") is False          # nothing measured: no-op
    assert mm.record_load("Org~Alpha", "split", "wk-a:0", "unloaded", upload_time_s=3.0) is True
    assert mm.derive_variant(-1) in model_metrics.VARIANTS
    assert mm.derive_variant(None) is None


def test_install_and_uninstall_round_trip():
    S.reset_providers()
    try:
        out = P.install()
        assert S.get_worker_registry() is out["worker_registry"]
        assert S.get_worker_transport() is out["worker_transport"]
        assert S.get_eviction_ledger() is out["eviction_ledger"]
        assert S.get_blocklist() is out["blocklist"]
        assert S.get_model_metrics() is out["model_metrics"]
        assert S.get_priority_groups() is out["priority_groups"]
        assert P.installed()["legacy_resolver_providers"] is True
        custom = P.FleetBlocklist()
        assert P.install(blocklist=custom, legacy_resolver_providers=False)["blocklist"] is custom
        assert S.get_blocklist() is custom
    finally:
        P.uninstall()
    assert isinstance(S.get_worker_registry(), S.NullWorkerRegistry)
    assert isinstance(S.get_blocklist(), S.NullBlocklist)
    assert P.installed() == {}
