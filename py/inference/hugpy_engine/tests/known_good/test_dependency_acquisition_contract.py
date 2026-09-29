"""KNOWN-GOOD CONTRACT — an adapter's absent base is a DEPENDENCY: acquired on
demand through the store's own on-demand path, or a FINAL refusal naming why
the acquisition was refused. Never a loop.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation").
Source under test: hugpy_storage/provision.py (acquire_dependency,
dependency_acquisitions, the one-attempt-per-window policy),
hugpy_engine/generate/config.py (resolve_adapter_or_acquire,
adapter_base_refusal(acquisition=...)), hugpy_engine/serve/load_failure.py
(MissingDependencyFailure.acquisition), hugpy_fleet/central/workers.py
(load_state_for_model stage ``acquiring_dependency``),
hugpy_engine/resolvers/remote.py (_cold_progress .stage).

Deterministic: a synthetic adapter dir + store root under tmp_path; the
central transfer (ensure_model_registered / ensure_model_present) is MOCKED —
nothing is ever downloaded. The live sentiment adapter on computron is not
touched.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest

pa = importlib.import_module("hugpy_engine.peft_adapters")
cfg = importlib.import_module("hugpy_engine.generate.config")
LF = importlib.import_module("hugpy_engine.serve.load_failure")
prov = importlib.import_module("hugpy_storage.provision")

BASE = "unsloth/Llama-3.2-3B-Instruct"
KEY = "veeraragavan410~Llama-3.2-3B-sentiment"


@pytest.fixture
def orphan(tmp_path, monkeypatch):
    """An adapter whose base is absent from an empty store; a central URL so
    the WORKER path (central-only acquisition) is exercised."""
    d = tmp_path / "adapter"
    d.mkdir()
    (d / pa.ADAPTER_CONFIG_NAME).write_text(json.dumps(
        {"base_model_name_or_path": BASE, "peft_type": "LORA"}))
    (d / "adapter_model.safetensors").write_bytes(b"\0" * 16)
    store = tmp_path / "store"
    store.mkdir()
    prov.reset_dependency_acquisitions()
    monkeypatch.delenv("HUGPY_DEPENDENCY_ACQUIRE", raising=False)
    monkeypatch.delenv("HUGPY_DEPENDENCY_ACQUIRE_WINDOW_S", raising=False)
    monkeypatch.setattr(prov, "worker_central_url", lambda: "http://central:7002")
    yield str(d), str(store)
    prov.reset_dependency_acquisitions()


def _materialize_base(store):
    """What a real central transfer leaves behind: the base's weights in the
    store where find_base_model_dir looks."""
    base_dir = pa.find_base_model_dir(BASE, store)
    if base_dir is None:
        from hugpy_storage.model_paths import route_destination
        base_dir = route_destination({"hub_id": BASE, "framework": "transformers",
                                      "primary_task": "text-generation"}, store)
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "model.safetensors"), "wb") as fh:
        fh.write(b"\0" * 16)
    with open(os.path.join(base_dir, "config.json"), "w") as fh:
        json.dump({"model_type": "llama"}, fh)
    return base_dir


def test_policy_allowed_acquires_the_base_and_the_load_resumes(orphan, monkeypatch):
    """INVARIANT: with the policy allowing it, the first load of an adapter
    whose base is absent starts the base acquisition through the store's
    on-demand path (ensure_model_registered -> ensure_model_present, purpose
    demand — the same central-only, budget-gated transfer every called model
    takes), reports stage=acquiring_dependency with bytes/total while it runs,
    and the adapter load proceeds when the base lands. Established: 2026-09-29."""
    adapter, store = orphan
    calls = []
    seen_live = {}

    def fake_registered(model_key, central_url):
        calls.append(("register", model_key, central_url))
        return "Llama-3.2-3B-Instruct"

    def fake_present(model_key, central_url, progress=None, state=None, purpose=None):
        calls.append(("present", model_key, purpose))
        progress(4 << 30, 6 << 30, "model-00001.safetensors")
        seen_live.update(prov.dependency_acquisitions()[KEY])   # mid-transfer snapshot
        _materialize_base(store)
        progress(6 << 30, 6 << 30, "model-00002.safetensors")
        return True

    monkeypatch.setattr(prov, "ensure_model_registered", fake_registered)
    monkeypatch.setattr(prov, "ensure_model_present", fake_present)
    monkeypatch.setattr(prov, "model_is_local", lambda k: True)

    base_dir, adapter_dir = cfg.resolve_adapter_or_acquire(KEY, adapter, allowed=True, root=store)
    assert adapter_dir == adapter and base_dir == pa.find_base_model_dir(BASE, store)
    assert calls == [("register", BASE, "http://central:7002"),
                     ("present", "Llama-3.2-3B-Instruct", "demand")]
    # the heartbeat surface while it ran
    assert seen_live["stage"] == "acquiring_dependency" and seen_live["status"] == "running"
    assert seen_live["base_id"] == BASE and seen_live["base_key"] == "Llama-3.2-3B-Instruct"
    assert seen_live["done_bytes"] == 4 << 30 and seen_live["total_bytes"] == 6 << 30
    assert seen_live["frac"] == 0.6667
    done = prov.dependency_acquisitions()[KEY]
    assert done["status"] == "done" and done["ok"] is True and done["policy"] == "allowed"
    assert done["done_bytes"] == done["total_bytes"] == 6 << 30


def test_policy_forbidden_is_the_final_refusal_with_the_reason(orphan, monkeypatch):
    """INVARIANT: when the policy forbids the acquisition (the switch is off;
    the storage budget refuses; central has no such hub id; the caller said
    auto_download=False) the existing FINAL missing_dependency refusal stands,
    now ending with the reason, and load_failure carries the acquisition
    record. No transfer is attempted. Established: 2026-09-29."""
    adapter, store = orphan
    transfers = []
    monkeypatch.setattr(prov, "ensure_model_present",
                        lambda *a, **k: transfers.append(a) or True)

    def _refusal(**kw):
        with pytest.raises(LF.MissingDependencyFailure) as ei:
            cfg.resolve_adapter_or_acquire(KEY, adapter, root=store, **kw)
        ref = ei.value
        assert ref.load_class == "missing_dependency" and ref.base_id == BASE
        assert str(ref).startswith(f"{KEY}: PEFT adapter (base {BASE!r})")
        assert "FIX:" in str(ref)
        return ref

    # 1. the switch is off
    monkeypatch.setenv("HUGPY_DEPENDENCY_ACQUIRE", "off")
    ref = _refusal(allowed=True)
    assert "base acquisition refused [disabled]" in str(ref)
    assert ref.load_failure["acquisition"]["policy"] == "disabled"
    assert "HUGPY_DEPENDENCY_ACQUIRE=off" in ref.load_failure["acquisition"]["reason"]
    monkeypatch.delenv("HUGPY_DEPENDENCY_ACQUIRE")
    prov.reset_dependency_acquisitions()

    # 2. the caller forbade downloads on this load
    ref = _refusal(allowed=False)
    assert "auto_download=False" in str(ref)
    assert ref.load_failure["acquisition"]["policy"] == "disabled"

    # 3. central has no such hub id
    monkeypatch.setattr(prov, "ensure_model_registered", lambda mk, c: None)
    ref = _refusal(allowed=True)
    assert "base acquisition refused [unknown]" in str(ref)
    assert BASE in ref.load_failure["acquisition"]["reason"]
    prov.reset_dependency_acquisitions()

    # 4. the storage budget refuses the pull
    class BudgetRefusal(Exception):
        def __init__(self):
            super().__init__("won't fit")
            self.reason = {"reason": "won't fit: needs 6.4 GB, 1.0 GB free of the disk cap"}

    def refuse(*a, **k):
        raise BudgetRefusal()
    monkeypatch.setattr(prov, "ensure_model_registered", lambda mk, c: "Llama-3.2-3B-Instruct")
    monkeypatch.setattr(prov, "ensure_model_present", refuse)
    ref = _refusal(allowed=True)
    assert "base acquisition refused [budget]" in str(ref)
    assert "1.0 GB free of the disk cap" in ref.load_failure["acquisition"]["reason"]
    assert transfers == []
    # the structured class is still FINAL for central
    remote = importlib.import_module("hugpy_engine.resolvers.remote")
    assert remote._is_final_load_class(ref.load_failure)


def test_a_failed_download_is_final_and_is_not_retried_within_the_window(orphan, monkeypatch):
    """INVARIANT: one acquisition attempt per (adapter, base) per policy
    window — a second load inside the window makes NO transfer and refuses
    naming the earlier attempt; after the window a new attempt is allowed.
    Established: 2026-09-29."""
    adapter, store = orphan
    attempts = []
    monkeypatch.setattr(prov, "ensure_model_registered", lambda mk, c: "Llama-3.2-3B-Instruct")

    def failing(model_key, central_url, progress=None, state=None, purpose=None):
        attempts.append(model_key)
        prov._record_failure(model_key, "central", "central holds no weights for it")
        return False
    monkeypatch.setattr(prov, "ensure_model_present", failing)
    monkeypatch.setattr(prov, "model_is_local", lambda k: False)

    with pytest.raises(LF.MissingDependencyFailure) as ei:
        cfg.resolve_adapter_or_acquire(KEY, adapter, root=store)
    assert "base acquisition failed [failed]" in str(ei.value)
    assert "central holds no weights" in ei.value.load_failure["acquisition"]["reason"]
    assert attempts == ["Llama-3.2-3B-Instruct"]

    with pytest.raises(LF.MissingDependencyFailure) as ei2:
        cfg.resolve_adapter_or_acquire(KEY, adapter, root=store)
    assert attempts == ["Llama-3.2-3B-Instruct"]              # no re-download
    acq = ei2.value.load_failure["acquisition"]
    assert acq["policy"] == "window" and "already attempted" in acq["reason"]
    assert "central holds no weights" in acq["reason"]         # the earlier reason is named

    monkeypatch.setenv("HUGPY_DEPENDENCY_ACQUIRE_WINDOW_S", "0.001")
    import time
    time.sleep(0.01)
    with pytest.raises(LF.MissingDependencyFailure):
        cfg.resolve_adapter_or_acquire(KEY, adapter, root=store)
    assert attempts == ["Llama-3.2-3B-Instruct"] * 2           # a new window, a new attempt


def test_central_reports_the_adapters_stage_as_acquiring_dependency():
    """INVARIANT: the worker's ``dependency_acquisitions`` on the heartbeat
    makes load_state_for_model report the ADAPTER as pulling with stage
    acquiring_dependency + the base's bytes, and the hold's status event
    carries that stage. Established: 2026-09-29."""
    W = importlib.import_module("hugpy_fleet.central.workers")
    remote = importlib.import_module("hugpy_engine.resolvers.remote")
    worker = {"id": "w1", "name": "computron", "loaded_models": [], "loading": [],
              "models_local": [KEY], "provisioning": [], "allocations": [],
              "dependency_acquisitions": {KEY: {
                  "base_id": BASE, "base_key": "Llama-3.2-3B-Instruct",
                  "stage": "acquiring_dependency", "status": "running",
                  "done_bytes": 4 << 30, "total_bytes": 6 << 30, "frac": 0.6667,
                  "finished_at": None}}}

    class _Store:
        def get(self, wid):
            return worker if wid == "w1" else None
    orig = W.worker_store
    W.worker_store = _Store()
    try:
        orig_health = getattr(W, "_live_health", None)
        W._live_health = lambda w: None
        try:
            ls = W.load_state_for_model(KEY, "w1", 0.0)
        finally:
            if orig_health is not None:
                W._live_health = orig_health
    finally:
        W.worker_store = orig
    assert ls["pulling"] is True and ls["in_progress"] is True
    assert ls["stage"] == "acquiring_dependency"
    assert ls["progress"] == 0.6667
    assert BASE in ls["message"] and KEY in ls["message"] and "of" in ls["message"]
    # the hold reads the stage off the same load state
    prev = remote._load_state_provider
    remote.set_load_state_provider(lambda mk, wid, since=0.0: ls)
    try:
        cp = remote._cold_progress(KEY, worker, 0.0)
    finally:
        remote.set_load_state_provider(prev)
    moved, prog, msg, honest, ready = cp
    assert moved is True and prog == 0.6667 and honest is None and ready is False
    assert cp.stage == "acquiring_dependency"
