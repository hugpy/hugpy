"""Integration: boot the real flask app, exercise queue/jobs/cancel surfaces."""
import os
import time

import pytest

from hugpy_control.bus import bus
from hugpy_control import principals as _pr, settings as _se
from hugpy_control.jobs import job_store
from hugpy_engine.dispatch import activity


def test_store_to_bus_adapter_wired(server_app):
    # wire_cancel/wire_job_events ran in the factory
    assert job_store.on_change is not None


# --- queue view (activity shim) ---
def test_queue_view_activity_shim(client):
    r = client.get("/llm/queue")
    assert r.status_code == 200
    q = r.get_json()
    assert "active" in q and "counts" in q
    assert set(q["counts"]) == {"waiting", "active", "total"}

    # simulate a chat stream lifecycle through the shim (what /chat/stream does)
    activity.begin("it-1", "some-model", "Some Model")
    q = client.get("/llm/queue").get_json()
    mine = [e for e in q["active"] if e["request_id"] == "it-1"]
    assert len(mine) == 1 and mine[0]["state"] == "waiting", "begun request visible as waiting"
    assert set(mine[0]) == {"request_id", "model_key", "model", "kind", "state",
                            "elapsed", "wait", "tokens"}, "old snapshot keys intact"
    activity.on_token("it-1"); activity.on_token("it-1")
    q = client.get("/llm/queue").get_json()
    mine = [e for e in q["active"] if e["request_id"] == "it-1"][0]
    assert mine["state"] == "active" and mine["tokens"] == 2, "active after tokens"
    activity.end("it-1")
    q = client.get("/llm/queue").get_json()
    assert not [e for e in q["active"] if e["request_id"] == "it-1"], "gone from queue after end"


# --- downloads /jobs surface: legacy wire, download-only ---
def test_downloads_jobs_surface_legacy_wire(client):
    job_store.create("some-model", id="dl-1", kind="download", transport="web")
    job_store.update("dl-1", status="running", progress=0.4)
    job_store.create("m", id="chat-x", kind="chat")   # must NOT appear on /jobs
    try:
        r = client.get("/jobs")
        jobs = {j["id"]: j for j in r.get_json()}
        assert jobs.get("dl-1", {}).get("status") == "running", \
            "download job on /jobs with legacy status"
        assert "chat-x" not in jobs, "chat job not on /jobs"
        r = client.get("/jobs/dl-1")
        assert r.get_json()["status"] == "running", "GET /jobs/<id> legacy shape"
    finally:
        job_store.finish("chat-x")
        job_store.finish("dl-1")


# --- chat cancel route: local job gets cancel_requested via the bus ---
def test_chat_cancel_route_via_bus(client):
    fired = []
    job_store.create("m", id="cx-1", kind="chat", transport="web")
    job_store.attach_cancel("cx-1", lambda: fired.append(1))
    r = client.post("/llm/chat/cancel/cx-1")
    assert r.status_code == 200 and r.get_json()["cancelled"] is True
    deadline = time.time() + 2
    while not fired and time.time() < deadline:
        time.sleep(0.02)
    assert fired == [1], "cancel handle fired via bus"
    assert job_store.get("cx-1").cancel_requested, "job flagged cancel_requested"
    job_store.finish("cx-1")
    assert job_store.get("cx-1").to_dict()["status"] == "cancelled", "teardown resolves cancelled"

    # unknown request: no local job, no workers -> cancelled false
    r = client.post("/llm/chat/cancel/nope")
    assert r.get_json()["cancelled"] is False


# --- bus saw the lifecycle events ---
def test_bus_sees_lifecycle_events(server_app):
    sub = bus.subscribe("job.*")
    try:
        job_store.create("m", id="ev-1", kind="chat")
        m = sub.get(timeout=1)
        assert m is not None and m.job_id == "ev-1", "job.created published on app bus"
        job_store.finish("ev-1")
        m2 = sub.get(timeout=1)
        assert m2 is not None and m2.topic == "job.done", "job.done published"
    finally:
        sub.close()


# ---------------- Phase-0 F2/F3/F4 route surfaces ----------------
@pytest.fixture
def foundations(client, tmp_path, monkeypatch):
    """Throwaway principal/settings stores + the documented self-hosted "open"
    auth mode.

    Settings WRITES are operator-gated (operator_auth._SENSITIVE ^/settings/.+$),
    and the bare test env resolves to the fail-closed external mode. This section
    tests the settings API, not the gate — run it in the "open" mode (permissive
    while no operator token is set)."""
    monkeypatch.setattr(_pr.principal_store, "_path", str(tmp_path / "principals.json"))
    monkeypatch.setattr(_se.settings_store, "_path", str(tmp_path / "settings.json"))
    monkeypatch.setattr(_se.settings_store, "_cache", None)
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    return client


def test_f4_settings_control_api(foundations):
    c = foundations
    r = c.post("/settings/discord.channels/42", json={"value": {"respond": "all"}})
    assert r.status_code == 200, "settings write 200"
    r = c.get("/settings/discord.channels")
    assert r.get_json()["values"].get("42") == {"respond": "all"}, "settings ns read"
    r = c.post("/settings/discord.channels/42", json={"merge": {"personality": "pirate"}, "value": None})
    assert r.get_json()["value"]["personality"] == "pirate", "settings merge"
    r = c.get("/settings")
    assert "discord.channels" in r.get_json()["namespaces"], "namespaces listed"


def test_discord_prefs_m2m_write_lands_in_settings(foundations):
    c = foundations
    r = c.post("/discord/prefs", json={"user_id": "777", "model_key": "qwen"})
    assert r.status_code == 200, "discord prefs M2M write"
    r = c.get("/settings/discord.users/777")
    assert r.get_json()["value"] == {"model": "qwen"}, "pref visible in settings"


def test_f2_principals_lifecycle_over_http(foundations):
    c = foundations
    r = c.post("/auth/principals", json={"kind": "user", "name": "bob", "groups": ["media"]})
    assert r.status_code == 200, "principal create 200"
    body = r.get_json()
    assert (body.get("token") or "").startswith("hpp_"), "token returned once"
    tok = body["token"]
    r = c.get("/auth/whoami", headers={"Authorization": f"Bearer {tok}"})
    assert (r.get_json()["principal"] or {}).get("name") == "bob", "whoami resolves principal"
    r = c.post("/auth/discord-link", json={"token": tok, "discord_user_id": "555"})
    assert r.status_code == 200 and r.get_json()["linked"], "discord-link 200"
    r = c.post("/auth/discord-link", json={"token": "hpp_bad", "discord_user_id": "556"})
    assert r.status_code == 401, "discord-link bad token 401"


def test_f5_unified_jobs_endpoint_filters_by_transport(foundations):
    c = foundations
    job_store.create("m", id="uj-1", kind="chat", transport="cli", principal="pr_x")
    try:
        r = c.get("/llm/jobs?transport=cli")
        rows = r.get_json()["jobs"]
        assert any(j["id"] == "uj-1" and j["principal"] == "pr_x" for j in rows)
    finally:
        job_store.finish("uj-1")


def test_f3_1_model_meta_endpoint(foundations):
    c = foundations
    r = c.get("/models/Qwen2.5-3B-Instruct-GGUF/meta")
    if r.status_code == 200:
        m = r.get_json()
        assert "recommended" in m and m.get("params_b") == 3.0, "meta has quant + recommended"
    else:
        assert r.status_code == 404, "meta 404 for unknown key only"


def test_f3_2_worker_registry_surfacing(foundations):
    c = foundations
    r = c.get("/llm/workers")
    assert all("version_ok" in w for w in r.get_json()), "workers rows carry version_ok"
