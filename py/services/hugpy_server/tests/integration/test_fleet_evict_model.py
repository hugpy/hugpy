"""Targeted eviction — `evict <model_key>` (worker /ops/evict).

Central signals `evict <model_key>` (+ optional force), NEVER a raw PID (PIDs are
per-box and recycled). The worker resolves the model_key to its LIVE hosting
handle at eviction time, verifies identity, and frees it with the mechanism that
matches HOW it's hosted:

  * comfy      — framework == 'comfy'  -> comfy's OWN POST /free (no PID kill)
  * slot       — a live slot serves it -> verify identity, then slot /unload
                                          (owner does SIGTERM -> wait -> SIGKILL)
  * in-process — weights in our PID     -> drop refs + CUDA empty_cache + trim
  * not resident / foreign proc         -> idempotent no-op, never a kill

Covered here:
  (1) model_key -> handle resolution picks the right host_mode per mode;
  (2) recycled-PID guard: if the slot handle changed before we act (swapped model
      or respawned child under a new pid), we do NOT evict — no slot /unload fires;
  (3) the static / in-flight gate is honored UNLESS force=true (📌pin does NOT
      gate the evict verb — pin is designation, not a VRAM lock; 2026-07-15);
  (4) the comfy path calls comfy's /free with the documented body (mocked httpx);
  (5) a foreign/non-owned model_key resolves to "not resident" and is REFUSED,
      never killed (no in-process drop, no slot unload);
  (6) an unknown/missing model_key is an idempotent no-op at HTTP 200 (never 500);
  (7) central relay POST /llm/workers/<id>/evict forwards to /ops/evict verbatim.
  (8) HONEST accounting (2026-07-25 fix — the ae 44GB/35.6MB lie): the new
      ``freed`` breakdown is measured PER-MODEL from ground truth (nvidia-smi
      joined on child_pid + /proc rss split for slot; torch tensor bytes /
      GGUF file size for in-process), never a box-wide MemAvailable delta.
      comfy reports freed=None with a reason (no per-model attribution is
      possible); not-resident reports zeros (nothing to free); the legacy
      vram_freed/ram_freed keys stay present for wire back-compat.
  (9) central relay POST /llm/workers/<id>/reap-orphans (k32 gap) forwards to
      /ops/reap-orphans verbatim, and is operator-gated the same as evict.
  (10) the stranded-slot fix: POST /slots/<slot_id>/unload unconditionally
      tears down a slot's child regardless of its model_key claim (including
      None/stale), and its central relay + operator gate exist.

Every module-level seam the worker agent reads at call time is monkeypatched
per test (never a live GPU, slot child, or comfy).
"""
import importlib
import itertools
import sys
import types

import pytest
from flask import Flask

from worker_store_isolation import swap_worker_store

from hugpy_fleet.worker import agent

slots = importlib.import_module("hugpy_engine.serve.slots")
slot_agent = importlib.import_module("hugpy_engine.serve.slot_agent")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")


class FakeSlotPool:
    """Records every .unload(control_url) so a test can assert whether the slot
    child was actually torn down (the recycled-PID guard must PREVENT it)."""
    calls: list = []

    def __init__(self, urls=None):
        pass

    def unload(self, control_url, **kw):
        FakeSlotPool.calls.append(control_url)
        return {"ok": True}


def _new_client():
    state = agent.WorkerState(name="t", url=None, worker_id="w-e")
    return agent.build_app(state).test_client()


@pytest.fixture
def seams(monkeypatch):
    """Neutralize the agent's side-effecting globals; by default nothing is
    comfy / slot / in-process unless a test says so. Yields the monkeypatch."""
    monkeypatch.setattr(agent, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(agent, "loaded_model_keys", lambda: [])
    saved_settings = dict(agent._RUNTIME_SETTINGS)
    agent._RUNTIME_SETTINGS.clear()
    monkeypatch.setattr(agent.gen_gate, "in_flight", lambda mk: 0)
    # comfy provably idle unless a test says otherwise (the comfy branch asks
    # the watchdog's predicate before /free — 2026-09-22 ledger change).
    monkeypatch.setattr(agent, "_comfy_busy_reason", lambda state: None)
    monkeypatch.setattr(agent, "_model_framework", lambda mk: None)
    monkeypatch.setattr(agent, "_resolve_slot_handle", lambda mk: None)
    monkeypatch.setattr(agent, "_is_inprocess_resident", lambda mk: False)
    FakeSlotPool.calls.clear()
    monkeypatch.setattr(slots, "SlotPool", FakeSlotPool)
    yield monkeypatch
    agent._RUNTIME_SETTINGS.clear()
    agent._RUNTIME_SETTINGS.update(saved_settings)


@pytest.fixture
def fixed_mem(monkeypatch):
    """Patch the before/after VRAM+RAM readers to a deterministic cycle so
    vram_freed/ram_freed are computable without a GPU: each _evict_model reads
    before (1000/2000) then after (5000/9000) -> freed 4000/7000, every call."""
    vcyc, rcyc = itertools.cycle([1000, 5000]), itertools.cycle([2000, 9000])
    monkeypatch.setattr(agent, "_free_vram_bytes", lambda: next(vcyc))
    monkeypatch.setattr(agent, "_free_ram_bytes", lambda: next(rcyc))


SLOT_HANDLE = {"control_url": "http://127.0.0.1:8101", "child_pid": 4242,
               "endpoint": "http://127.0.0.1:8101"}


# --- (1a) comfy host-mode: framework==comfy -> comfy branch ------------------

def test_1a_comfy_model_goes_through_comfy_free(seams, fixed_mem):
    seams.setattr(agent, "_model_framework", lambda mk: "comfy" if mk == "cmfy" else None)
    comfy_called = []

    def _fake_comfy_free(state):
        comfy_called.append(True)
        return True, "comfy /free accepted (unload_models + free_memory)"
    seams.setattr(agent, "_comfy_free_models", _fake_comfy_free)

    r = _new_client().post("/ops/evict", json={"model_key": "cmfy"})
    b = r.get_json()
    assert r.status_code == 200
    assert b["host_mode"] == "comfy" and b["evicted"] is True
    # comfy path went through comfy's /free API (no PID kill)
    assert comfy_called == [True]
    # vram_freed reported from before/after delta
    assert b["vram_freed"] == 4000 and b["ram_freed"] == 7000


# --- (1a2) a RENDERING comfy is gated; force overrides ------------------------

def test_1a2_busy_comfy_is_gated_unless_forced(seams, fixed_mem):
    """comfy /free drops every checkpoint at once, so a render in flight against
    comfy (ours or not) vetoes the verb the same way it vetoes the watchdog."""
    seams.setattr(agent, "_model_framework", lambda mk: "comfy" if mk == "cmfy" else None)
    seams.setattr(agent, "_comfy_busy_reason", lambda state: "comfy /queue has work")
    comfy_called = []
    seams.setattr(agent, "_comfy_free_models",
                  lambda state: comfy_called.append(True) or (True, "freed"))

    b = _new_client().post("/ops/evict", json={"model_key": "cmfy"}).get_json()
    assert b["host_mode"] == "comfy" and b["evicted"] is False
    assert "comfy /queue has work" in b["reason"] and comfy_called == []

    b = _new_client().post("/ops/evict", json={"model_key": "cmfy", "force": True}).get_json()
    assert b["evicted"] is True and comfy_called == [True]


# --- (1b) slot host-mode: live slot serves it -> slot /unload ----------------

def test_1b_slot_model_unloads_resolved_control_url(seams, fixed_mem):
    seams.setattr(agent, "_resolve_slot_handle",
                  lambda mk: dict(SLOT_HANDLE) if mk == "slotmodel" else None)
    b = _new_client().post("/ops/evict", json={"model_key": "slotmodel"}).get_json()
    assert b["host_mode"] == "slot" and b["evicted"] is True
    assert b["child_pid"] == 4242
    assert FakeSlotPool.calls == ["http://127.0.0.1:8101"]


# --- (2) recycled-PID guard: handle changes before we act -> NO unload -------

def test_2_recycled_pid_guard_prevents_unload(seams, fixed_mem):
    # first call (resolve) returns pid 4242; recheck returns a DIFFERENT pid.
    seq = iter([
        {"control_url": "http://127.0.0.1:8101", "child_pid": 4242},   # resolve
        {"control_url": "http://127.0.0.1:8101", "child_pid": 9999},   # recheck (swapped)
    ])
    seams.setattr(agent, "_resolve_slot_handle", lambda mk: next(seq))
    b = _new_client().post("/ops/evict", json={"model_key": "slotmodel"}).get_json()
    assert b["host_mode"] == "slot" and b["evicted"] is False
    assert "recycled" in b["reason"].lower()
    # the guard PREVENTED the slot /unload (no kill fired)
    assert FakeSlotPool.calls == []


# --- (3) in-process host-mode + gate honored unless force --------------------

@pytest.fixture
def inprocess(seams):
    dropped = []
    seams.setattr(agent, "_is_inprocess_resident", lambda mk: mk == "ip")
    seams.setattr(agent, "_drop_inprocess_model", lambda mk: (dropped.append(mk) or True))
    return dropped


def test_3a_inprocess_on_demand_evicted(inprocess, fixed_mem):
    b = _new_client().post("/ops/evict", json={"model_key": "ip"}).get_json()
    assert b["host_mode"] == "in_process" and b["evicted"] is True
    # in-process drop actually ran (no PID kill)
    assert inprocess == ["ip"]


def test_3b_static_residency_gated_without_force(inprocess, fixed_mem):
    agent._RUNTIME_SETTINGS.update({"residency": {"ip": "static"}})
    b = _new_client().post("/ops/evict", json={"model_key": "ip"}).get_json()
    assert b["evicted"] is False and "static" in b["reason"].lower()
    assert inprocess == []


def test_3c_force_overrides_static_gate(inprocess, fixed_mem):
    agent._RUNTIME_SETTINGS.update({"residency": {"ip": "static"}})
    b = _new_client().post("/ops/evict", json={"model_key": "ip", "force": True}).get_json()
    assert b["evicted"] is True and b["forced"] is True
    assert inprocess == ["ip"]


def test_3d_in_flight_generation_gated_without_force(inprocess, fixed_mem, seams):
    seams.setattr(agent.gen_gate, "in_flight", lambda mk: 1 if mk == "ip" else 0)
    b = _new_client().post("/ops/evict", json={"model_key": "ip"}).get_json()
    assert b["evicted"] is False and "in-flight" in b["reason"].lower()
    assert inprocess == []


# --- (5) foreign / non-owned model_key -> not resident, REFUSED not killed ---

def test_5_foreign_model_not_resident_never_killed(seams, fixed_mem):
    dropped = []
    seams.setattr(agent, "_drop_inprocess_model", lambda mk: (dropped.append(mk) or True))
    r = _new_client().post("/ops/evict", json={"model_key": "who-owns-this"})
    b = r.get_json()
    assert r.status_code == 200
    assert b["host_mode"] == "none" and b["evicted"] is False
    assert "not resident" in b["reason"].lower()
    assert FakeSlotPool.calls == [] and dropped == []


# --- (6) unknown/missing model_key -> idempotent 200 no-op, never 500 --------

@pytest.mark.parametrize("body", [{}, {"model_key": "   "}])
def test_6_missing_or_blank_model_key_is_200_noop(seams, fixed_mem, body):
    r = _new_client().post("/ops/evict", json=body)
    assert r.status_code == 200
    assert r.get_json()["evicted"] is False


# --- (4) _comfy_free_models unit: calls comfy /free with the documented body -

def test_4_comfy_free_posts_documented_body(monkeypatch):
    # import httpx inside the fn -> swap sys.modules['httpx'] for a recorder.
    captured = {}
    fake_httpx = types.ModuleType("httpx")

    class _Resp:
        status_code = 200

    def _post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()
    fake_httpx.post = _post
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setenv("COMFY_URL", "http://comfy.local:8188")

    state = agent.WorkerState(name="t", url=None, worker_id="w-e")
    freed_ok, note = agent._comfy_free_models(state)
    assert freed_ok is True
    assert captured["url"] == "http://comfy.local:8188/free"
    assert captured["json"] == {"unload_models": True, "free_memory": True}


# --- (7)/(9)/(10c) central relay routes ----------------------------------------

@pytest.fixture
def relay(monkeypatch):
    """A bare app with worker_bp and a recording _relay_worker_op (the operator
    gate isn't mounted, so the route runs raw). Returns (client, relayed)."""
    relayed = {}

    def _fake_relay(worker_id, op_path, body, timeout, action, **kw):
        relayed.update(worker_id=worker_id, op_path=op_path, body=body, action=action)
        return ({"ok": True, "relayed": True}, 200)
    monkeypatch.setattr(wr, "_relay_worker_op", _fake_relay)
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client(), relayed


def test_7_relay_evict_forwards_verbatim(relay):
    c, relayed = relay
    c.post("/llm/workers/ae/evict", json={"model_key": "m", "force": True})
    assert relayed.get("op_path") == "/ops/evict" and relayed.get("action") == "evict"
    assert relayed.get("body") == {"model_key": "m", "force": True}
    assert relayed.get("worker_id") == "ae"


def test_9a_relay_reap_orphans_forwards_verbatim(relay):
    c, relayed = relay
    c.post("/llm/workers/ae/reap-orphans", json={"dry_run": False})
    assert relayed.get("op_path") == "/ops/reap-orphans"
    assert relayed.get("action") == "reap-orphans"
    assert relayed.get("body") == {"dry_run": False}
    assert relayed.get("worker_id") == "ae"


def test_10c_relay_slot_unload_forwards(relay):
    c, relayed = relay
    with swap_worker_store(prefix="hugpy-evict-workers-") as store:
        store.register(name="ae", url="http://192.0.2.9:9100", worker_id="ae")
        c.post("/llm/workers/ae/slots/1/unload", json={})
    assert relayed.get("op_path") == "/slots/1/unload"
    assert relayed.get("action") == "slot-unload"


@pytest.fixture
def gated_client(monkeypatch):
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    return app.test_client()


@pytest.mark.parametrize("path", [
    "/llm/workers/ae/reap-orphans",     # (9b)
    "/llm/workers/ae/slots/1/unload",   # (10d)
])
def test_relay_routes_are_operator_gated(gated_client, path):
    r = gated_client.post(path, json={})
    assert r.status_code == 401


# --- (8) HONEST per-model accounting (fixes the ae 44GB/35.6MB lie) ----------

def test_8a_slot_footprint_from_nvidia_smi_and_proc_rss(monkeypatch):
    monkeypatch.setattr(agent, "_gpu_process_vram",
                        lambda: {4242: {"name": "llama-server", "mib": 42000}})
    monkeypatch.setattr(slot_agent, "_proc_rss_detail", lambda pid: (
        {"rss_anon_bytes": 1_500_000_000, "rss_file_bytes": 43_600_000_000}
        if pid == 4242 else {}))
    fp = agent._model_footprint_before_evict(
        "slotmodel", "slot", {"child_pid": 4242, "control_url": "x"})
    assert fp["vram_bytes"] == 42000 * agent._MIB
    # the HONEST pinned figure (rss_anon); file bytes carried separately
    assert fp["ram_anon_bytes"] == 1_500_000_000
    assert fp["ram_file_bytes"] == 43_600_000_000
    assert "nvidia-smi" in fp["measured_from"]


def test_8b_unmeasurable_slot_is_none_never_fabricated(monkeypatch):
    monkeypatch.setattr(agent, "_gpu_process_vram", lambda: {})
    monkeypatch.setattr(slot_agent, "_proc_rss_detail", lambda pid: {})
    fp = agent._model_footprint_before_evict(
        "slotmodel", "slot", {"child_pid": 9999, "control_url": "x"})
    assert fp["vram_bytes"] is None
    assert fp["ram_anon_bytes"] is None and fp["ram_file_bytes"] is None


def test_8c_inprocess_gguf_is_file_backed_not_anon(monkeypatch):
    monkeypatch.setattr(agent, "_loaded_detail",
                        lambda: {"gguf-model": {"model_bytes": 44_000_000_000}})
    monkeypatch.setattr(agent, "_inprocess_gpu_bytes", lambda: {})
    fp = agent._model_footprint_before_evict("gguf-model", "in_process")
    assert fp["ram_file_bytes"] == 44_000_000_000
    assert fp["vram_bytes"] is None
    assert "NOT pinned anon RAM" in fp["measured_from"]


def test_8d_inprocess_torch_is_real_tensor_sum(monkeypatch):
    monkeypatch.setattr(agent, "_loaded_detail", lambda: {})
    monkeypatch.setattr(agent, "_inprocess_gpu_bytes", lambda: {
        "torch-model": {"vram_bytes": 8_000_000_000, "device": "cuda"}})
    fp = agent._model_footprint_before_evict("torch-model", "in_process")
    assert fp["vram_bytes"] == 8_000_000_000
    assert "not a delta" in fp["measured_from"]


def test_8e_comfy_footprint_is_none_with_reason():
    fp = agent._model_footprint_before_evict("comfy-model", "comfy")
    assert fp["vram_bytes"] is None and fp["ram_anon_bytes"] is None
    assert fp["ram_file_bytes"] is None
    assert "no per-model_key attribution" in fp["measured_from"]


def test_8f_not_resident_is_zeros_not_null():
    fp = agent._model_footprint_before_evict("ghost", "none")
    assert fp == {"vram_bytes": 0, "ram_anon_bytes": 0, "ram_file_bytes": 0,
                  "measured_from": "not resident — nothing to free"}


def test_8g_evict_response_carries_freed_block_and_legacy_keys(seams, fixed_mem):
    """/ops/evict's slot branch carries BOTH the new 'freed' block and the legacy
    vram_freed/ram_freed (wire back-compat)."""
    seams.setattr(agent, "_gpu_process_vram",
                  lambda: {4242: {"name": "llama-server", "mib": 42000}})
    seams.setattr(slot_agent, "_proc_rss_detail",
                  lambda pid: {"rss_anon_bytes": 1_500_000_000,
                               "rss_file_bytes": 43_600_000_000})
    seams.setattr(agent, "_resolve_slot_handle",
                  lambda mk: dict(SLOT_HANDLE) if mk == "slotmodel" else None)
    b = _new_client().post("/ops/evict", json={"model_key": "slotmodel"}).get_json()
    assert b["vram_freed"] == 4000 and b["ram_freed"] == 7000
    assert b["freed"]["vram_bytes"] == 42000 * agent._MIB
    assert b["freed"]["ram_anon_bytes"] == 1_500_000_000


# --- (10) stranded-slot fix: unconditional /slots/<id>/unload -----------------

def test_10a_slot_unload_fires_even_with_none_model_key(seams):
    class _StatusSlotPool(FakeSlotPool):
        def statuses(self):
            return [{"slot_id": "1", "model_key": None, "child_pid": 7777,
                     "_control": "http://127.0.0.1:8101"}]
    seams.setattr(slots, "SlotPool", _StatusSlotPool)
    r = _new_client().post("/slots/1/unload", json={})
    b = r.get_json()
    assert r.status_code == 200 and b["ok"] is True
    assert b["model_key_before"] is None and b["child_pid_before"] == 7777
    # the slot's control url actually got .unload()'d
    assert FakeSlotPool.calls == ["http://127.0.0.1:8101"]


def test_10b_unknown_slot_id_is_404(seams):
    class _StatusSlotPool(FakeSlotPool):
        def statuses(self):
            return [{"slot_id": "1", "model_key": None, "child_pid": 7777,
                     "_control": "http://127.0.0.1:8101"}]
    seams.setattr(slots, "SlotPool", _StatusSlotPool)
    r = _new_client().post("/slots/99/unload", json={})
    assert r.status_code == 404
