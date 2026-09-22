"""Placement seam: the Null providers make a fleet-less box behave as "no fleet".

Every ``get_*()`` accessor answers safely with nothing installed, ``set_*()``
swaps an implementation in, and ``reset_providers()`` restores the defaults.
"""
from __future__ import annotations

import pytest

from hugpy_engine import placement as P


@pytest.fixture(autouse=True)
def _clean():
    P.reset_providers()
    yield
    P.reset_providers()


def test_null_worker_registry_knows_no_workers():
    reg = P.get_worker_registry()
    assert isinstance(reg, P.WorkerRegistry)
    assert reg.list_workers() == ()
    assert reg.get_worker("w1") is None
    assert reg.workers_for_model("Qwen~Qwen2.5-7B", online_only=True) == ()
    forms = set(reg.key_forms("Owner~Qwen2.5-7B-GGUF"))
    assert {"Owner~Qwen2.5-7B-GGUF", "owner~qwen2.5-7b-gguf", "Qwen2.5-7B-GGUF", "qwen2.5-7b-gguf"} <= forms
    assert set(reg.match_keys("Owner~Qwen2.5-7B-GGUF", ["qwen2.5-7b-gguf", "other"])) == {"qwen2.5-7b-gguf"}


def test_local_key_forms_handles_hub_ids_and_empty():
    assert P.local_key_forms("") == set()
    assert "qwen2.5-7b" in P.local_key_forms("Qwen/Qwen2.5-7B")


def test_null_transport_has_no_breaker_and_builds_urls_from_the_row():
    t = P.get_worker_transport()
    assert isinstance(t, P.WorkerTransport)
    worker = {"id": "w1", "name": "boxA", "url": "http://w1:9100/"}
    assert t.base_url(worker) == "http://w1:9100"
    assert t.breaker_key(worker) == "w1"
    assert t.guard("w1", url="http://w1:9100") is None
    t.note_failure("w1", RuntimeError("x"))
    t.note_ok("w1")
    assert t.breaker_snapshot() == {}
    with t.breaker_scope(worker) as key:
        assert key == "w1"
    assert isinstance(t.transport_errors, tuple) and t.transport_errors
    client_cm = t.async_client("relay")
    assert hasattr(client_cm, "__aenter__")


def test_null_eviction_ledger_is_silent():
    led = P.get_eviction_ledger()
    assert isinstance(led, P.EvictionLedger)
    assert led.emit("evict.start", model_key="m") is None
    assert led.emit_resolve_fail("m", None, "why") is None
    assert led.disk_stats("/nonexistent") == {}
    assert led.new_run_id() == ""
    with led.run_scope("run-1") as rid:
        assert rid == "run-1"
    assert led.current_group() is None
    assert led.recent() == ()


def test_null_blocklist_metrics_and_groups():
    assert P.get_blocklist().blocked_keys() == ()
    assert P.get_blocklist().block_reason("m") is None
    m = P.get_model_metrics()
    assert m.derive_variant(-1, 48, moe_capable=False) is None
    assert m.record_call("m", 12.0, task="text-generation", compute_s=1.0) is False
    assert m.record_load("m", "split", "boxA:0", "loaded", tok_per_s=20.0) is False
    assert m.get_call("m") is None
    assert m.stats("m") == {}
    assert P.get_priority_groups().workers_for_key("m") == ()


def test_set_and_reset_swap_implementations():
    class Block:
        def blocked_keys(self):
            return ("bad",)

        def block_reason(self, model_key):
            return "operator" if model_key == "bad" else None

    P.set_blocklist(Block())
    assert P.get_blocklist().block_reason("bad") == "operator"
    P.set_blocklist(None)
    assert P.get_blocklist().block_reason("bad") is None
    P.set_priority_groups(type("G", (), {"workers_for_key": lambda self, k: ["w9"]})())
    assert list(P.get_priority_groups().workers_for_key("m")) == ["w9"]
    P.reset_providers()
    assert P.get_priority_groups().workers_for_key("m") == ()


def test_engine_call_sites_degrade_without_a_fleet():
    """The seams the engine actually consults answer the fleet-less defaults."""
    from hugpy_engine.resolvers import remote
    from hugpy_engine.resolvers.assure_model_key import _blocked_fact, _servable_facts

    assert remote._slot_match_keys("Owner~M") >= {"Owner~M", "M"}
    assert remote._blocked_reason("anything") is None
    assert _blocked_fact("anything") is False
    assert _servable_facts("zz-no-such-model")[1] is False
