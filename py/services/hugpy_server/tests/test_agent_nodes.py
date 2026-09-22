"""P3.1 — offline tests for the /agent/* node registry + dispatch.

Three layers, NO network:
  * pure store (comms.agent_nodes.AgentNodeStore): enroll, token hashing +
    fail-closed auth, heartbeat, the dispatch queue's monotonic pull cursor,
    revocation, and the cross-process (shared-db) property that makes it
    gunicorn 3-worker safe.
  * the Flask blueprint via test_client: the M2M contract (register/heartbeat/
    tasks) and its node-token gate (401/403/410), plus the operator gate on
    /agent/nodes and /agent/<id>/dispatch — and the full register -> heartbeat
    -> dispatch -> pull acceptance flow.
  * the central operator allowlist (operator_auth._SENSITIVE) covers exactly
    the two operator routes and none of the M2M ones.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_agent_nodes.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Isolate any incidental writes; point the comms db at a scratch file so the
# imported module singletons never touch a real one.
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-agent-test-"))
os.environ["HUGPY_COMMS_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="hugpy-agent-comms-"), "comms.db")

import pytest
from flask import Flask

from hugpy_fleet.central.agent_nodes import AgentNodeStore
from hugpy_server.app.routes import agent_routes
from hugpy_server.app import operator_auth


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def store(tmp_path):
    return AgentNodeStore(path=str(tmp_path / "agent.db"))


_GOOD_KEY = "hp_test_valid_key"


@pytest.fixture
def client(store, monkeypatch):
    """A bare app with ONLY the agent blueprint — so the inline gates are what's
    under test (no central before_request gate installed), and the store is a
    fresh scratch db per test.

    2026-07-16 permanent-gate ruling: ``/agent/register`` now requires a valid
    console API key ALWAYS, independent of the sitewide ``api_key_required``
    toggle (see ``agent_routes._require_api_key``). ``verify_api_key`` is
    stubbed here to accept exactly ``_GOOD_KEY`` so the M2M-contract tests
    (heartbeat/tasks/dispatch/etc.) can enroll via ``_enroll()`` without a real
    key store; the gate itself — including its independence from the sitewide
    toggle — is exercised explicitly by the ``test_register_gate_*`` tests
    below using ``unkeyed_client``."""
    monkeypatch.setattr(agent_routes, "agent_node_store", store)
    from hugpy_server.app.functions.imports.utils import api_keys as _ak
    monkeypatch.setattr(_ak, "verify_api_key", lambda tok, required_scope=None: tok == _GOOD_KEY)
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    return app.test_client()


@pytest.fixture
def unkeyed_client(store, monkeypatch):
    """Like ``client``, but ``verify_api_key`` always refuses — for asserting
    the gate's negative space (no key, bad key) without a real key store."""
    monkeypatch.setattr(agent_routes, "agent_node_store", store)
    from hugpy_server.app.functions.imports.utils import api_keys as _ak
    monkeypatch.setattr(_ak, "verify_api_key", lambda tok, required_scope=None: False)
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    return app.test_client()


def _enroll(client, name="luigi", host="10.0.0.9", caps=("chat",),
            key=_GOOD_KEY):
    r = client.post("/agent/register",
                    json={"name": name, "host": host,
                          "capabilities": list(caps)},
                    headers=_auth(key) if key else {})
    assert r.status_code == 201, r.get_json()
    return r.get_json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ── pure store ──────────────────────────────────────────────────────────────
def test_register_mints_prefixed_id_and_token(store):
    node = store.register(name="luigi", host="h", capabilities=["chat", "tools"])
    assert node["id"].startswith("agn_")
    assert node["token"].startswith("agt_")
    assert node["name"] == "luigi"
    assert node["capabilities"] == ["chat", "tools"]
    assert node["status"] == "enrolled"


def test_register_public_view_never_leaks_secret(store):
    node = store.register(name="n")
    # get()/all() are the API-facing views: no plaintext, no hash, ever.
    got = store.get(node["id"])
    assert "token" not in got and "token_hash" not in got
    assert all("token" not in n and "token_hash" not in n for n in store.all())


def test_authenticate_is_fail_closed(store):
    node = store.register(name="n")
    nid, tok = node["id"], node["token"]
    assert store.authenticate(nid, tok) is True
    assert store.authenticate(nid, "agt_wrong") is False
    assert store.authenticate(nid, "no-prefix") is False
    assert store.authenticate(nid, None) is False
    assert store.authenticate(nid, "") is False
    assert store.authenticate("agn_unknown", tok) is False


def test_token_is_hashed_at_rest(store):
    node = store.register(name="n")
    import sqlite3
    with sqlite3.connect(store.path) as conn:
        row = conn.execute("SELECT token_hash FROM agent_nodes WHERE id=?",
                           (node["id"],)).fetchone()
    stored = row[0]
    assert stored != node["token"]           # never the plaintext
    assert len(stored) == 64                 # sha256 hex
    import hashlib
    assert stored == hashlib.sha256(node["token"].encode()).hexdigest()


def test_heartbeat_updates_reported_fields(store):
    node = store.register(name="n")
    assert store.get(node["id"])["last_seen"] is not None  # set at enroll
    hb = store.heartbeat(node["id"], status="busy",
                         current_task="atsk_1", version="0.1.0")
    assert hb["status"] == "busy"
    assert hb["current_task"] == "atsk_1"
    assert hb["version"] == "0.1.0"
    assert hb["last_seen"] is not None


def test_heartbeat_only_writes_provided_fields(store):
    node = store.register(name="n")
    store.heartbeat(node["id"], status="idle", version="0.1.0")
    store.heartbeat(node["id"], current_task="atsk_9")  # status/version untouched
    got = store.get(node["id"])
    assert got["status"] == "idle"
    assert got["version"] == "0.1.0"
    assert got["current_task"] == "atsk_9"


def test_heartbeat_unknown_node_returns_none(store):
    assert store.heartbeat("agn_nope", status="busy") is None


def test_dispatch_and_pull_cursor_is_monotonic(store):
    node = store.register(name="n")
    nid = node["id"]
    t1 = store.dispatch(nid, {"kind": "chat", "prompt": "a"})
    t2 = store.dispatch(nid, {"kind": "chat", "prompt": "b"})
    assert t1["id"].startswith("atsk_") and t2["seq"] > t1["seq"]
    assert t1["task"] == {"kind": "chat", "prompt": "a"}
    all_ = store.tasks_since(nid, 0)
    assert [t["seq"] for t in all_] == [t1["seq"], t2["seq"]]
    # advancing the cursor past t1 yields only t2 (idempotent pull)
    assert [t["seq"] for t in store.tasks_since(nid, t1["seq"])] == [t2["seq"]]
    # cursor at the tail yields nothing
    assert store.tasks_since(nid, t2["seq"]) == []


def test_dispatch_is_scoped_per_node(store):
    a = store.register(name="a")["id"]
    b = store.register(name="b")["id"]
    ta = store.dispatch(a, {"x": 1})
    store.dispatch(b, {"y": 2})
    # a only ever sees its own task
    got = store.tasks_since(a, 0)
    assert [t["seq"] for t in got] == [ta["seq"]]
    assert got[0]["node_id"] == a


def test_dispatch_unknown_node_returns_none(store):
    assert store.dispatch("agn_nope", {"x": 1}) is None


def test_revoke_blocks_auth_and_dispatch(store):
    node = store.register(name="n")
    nid, tok = node["id"], node["token"]
    assert store.revoke(nid) is True
    assert store.authenticate(nid, tok) is False       # revoked -> no auth
    assert store.dispatch(nid, {"x": 1}) is None        # revoked -> no queue
    assert store.get(nid)["revoked"] is True


def test_cross_process_shared_db(tmp_path):
    """Gunicorn 3-worker safety: a second store on the SAME db file sees a
    node registered by the first, and can pull tasks dispatched through it."""
    path = str(tmp_path / "shared.db")
    s1 = AgentNodeStore(path=path)
    s2 = AgentNodeStore(path=path)
    node = s1.register(name="n", capabilities=["chat"])
    nid, tok = node["id"], node["token"]
    assert s2.get(nid) is not None
    assert s2.authenticate(nid, tok) is True
    s1.dispatch(nid, {"kind": "chat"})
    assert len(s2.tasks_since(nid, 0)) == 1


# ── Flask blueprint: M2M contract ───────────────────────────────────────────
def test_route_register_returns_201_with_token(client):
    node = _enroll(client)
    assert node["id"].startswith("agn_")
    assert node["token"].startswith("agt_")


def test_route_register_requires_name(client):
    assert client.post("/agent/register", json={},
                       headers=_auth(_GOOD_KEY)).status_code == 400
    assert client.post("/agent/register", json={"name": "  "},
                       headers=_auth(_GOOD_KEY)).status_code == 400


def test_route_register_rejects_non_list_capabilities(client):
    r = client.post("/agent/register",
                    json={"name": "n", "capabilities": "chat"},
                    headers=_auth(_GOOD_KEY))
    assert r.status_code == 400


# ── register API-key gate (keeper 2026-07-16; PERMANENT — decoupled from the
#    sitewide api_key_required() toggle, see agent_routes._require_api_key) ──
def test_register_gate_refuses_without_key(unkeyed_client):
    """No key at all -> 401."""
    assert unkeyed_client.post(
        "/agent/register", json={"name": "n"}).status_code == 401


def test_register_gate_refuses_without_key_even_when_site_policy_off(
        unkeyed_client, monkeypatch):
    """THE regression this gate exists to close: a bare, keyless register must
    401 even when the SITEWIDE api_key_required() toggle is OFF — the exact
    live hole (2026-07-16, a public POST returned 201 with the toggle off).
    The gate must never consult that flag."""
    from hugpy_server.app.functions.imports.utils import api_keys as _ak
    monkeypatch.setattr(_ak, "api_key_required", lambda: False)
    r = unkeyed_client.post("/agent/register", json={"name": "n"})
    assert r.status_code == 401


def test_register_gate_refuses_without_key_even_in_agent_open_mode(
        unkeyed_client, monkeypatch):
    """The SECOND silent-reopen path, closed 2026-07-16 (operator: "if the
    hugpy_agent_open is bypassing gating then rework the method so that it abides
    to the gate").

    ``HUGPY_AGENT_OPEN`` used to waive this key gate as well as the operator gate.
    That made open mode a rename of the very bug we just fixed: flip it for a test
    and the PUBLIC credential-minting bootstrap door reopens (these routes are
    internet-reachable via the /api→:7001→:7002 chain). Open mode may still waive
    the OPERATOR gate on nodes/dispatch — that's its testing purpose — but it must
    NEVER waive the register key.
    """
    monkeypatch.setenv("HUGPY_AGENT_OPEN", "true")
    r = unkeyed_client.post("/agent/register", json={"name": "n"})
    assert r.status_code == 401, (
        "open mode must not waive the register key gate — the bootstrap endpoint "
        "mints credentials and is publicly reachable")


def test_agent_open_still_waives_the_operator_gate(unkeyed_client, monkeypatch):
    """The flip side: open mode KEEPS its documented testing purpose. It waives the
    OPERATOR gate on nodes/dispatch (a human gate) — only the register key and the
    node-token identity checks are non-waivable. Asserted so a future tightening
    doesn't silently delete the escape hatch the operator asked to keep."""
    monkeypatch.setenv("HUGPY_AGENT_OPEN", "true")
    r = unkeyed_client.get("/agent/nodes")
    assert r.status_code == 200


def test_register_gate_rejects_bad_key(unkeyed_client):
    r = unkeyed_client.post("/agent/register", json={"name": "n"},
                            headers={"Authorization": "Bearer hp_wrong"})
    assert r.status_code == 401


def test_register_gate_rejects_revoked_key(store, monkeypatch):
    """A key that verify_api_key refuses because it was revoked -> 401 (the
    gate defers entirely to verify_api_key's own revocation check)."""
    monkeypatch.setattr(agent_routes, "agent_node_store", store)
    from hugpy_server.app.functions.imports.utils import api_keys as _ak
    revoked = {"hp_was_good"}

    def _verify(tok, required_scope=None):
        return tok is not None and tok not in revoked

    monkeypatch.setattr(_ak, "verify_api_key", _verify)
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    c = app.test_client()
    r = c.post("/agent/register", json={"name": "n"},
               headers=_auth("hp_was_good"))
    assert r.status_code == 401


def test_register_gate_allows_valid_key(client):
    """A legit node presenting a valid console key enrolls (bootstrap works)."""
    r = client.post("/agent/register", json={"name": "n"},
                    headers=_auth(_GOOD_KEY))
    assert r.status_code == 201, r.get_json()
    assert r.get_json()["token"].startswith("agt_")


def test_register_gate_accepts_key_via_query_param(client):
    """?api_key= is accepted too (same curl affordance /v1 and /ml give)."""
    r = client.post(f"/agent/register?api_key={_GOOD_KEY}", json={"name": "n"})
    assert r.status_code == 201, r.get_json()


def test_register_gate_fails_closed_when_key_module_unimportable(
        unkeyed_client, monkeypatch):
    """If the key module can't even load, the gate must refuse rather than
    admit — fail-closed on an unloadable dependency, not just a bad key."""
    import builtins
    real_import = builtins.__import__

    def _boom_import(name, *a, **kw):
        if name.endswith("utils.api_keys") or name == "api_keys":
            raise ImportError("simulated: key module unavailable")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _boom_import)
    try:
        r = unkeyed_client.post("/agent/register", json={"name": "n"},
                                headers=_auth(_GOOD_KEY))
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    assert r.status_code == 401


def test_route_heartbeat_requires_valid_token(client):
    node = _enroll(client)
    nid = node["id"]
    # no token
    assert client.post(f"/agent/{nid}/heartbeat", json={}).status_code == 401
    # wrong token
    assert client.post(f"/agent/{nid}/heartbeat", json={},
                       headers=_auth("agt_wrong")).status_code == 401


def test_route_heartbeat_ok_with_token(client):
    node = _enroll(client)
    r = client.post(f"/agent/{node['id']}/heartbeat",
                    json={"status": "busy", "current_task": "atsk_1",
                          "version": "0.1.0"},
                    headers=_auth(node["token"]))
    assert r.status_code == 200
    assert r.get_json()["status"] == "busy"


def test_route_heartbeat_accepts_x_agent_token_header(client):
    node = _enroll(client)
    r = client.post(f"/agent/{node['id']}/heartbeat", json={"status": "idle"},
                    headers={"X-Agent-Token": node["token"]})
    assert r.status_code == 200


def test_route_heartbeat_unknown_node_410(client):
    r = client.post("/agent/agn_nope/heartbeat", json={},
                    headers=_auth("agt_whatever"))
    assert r.status_code == 410


def test_route_heartbeat_revoked_node_403(client, store):
    node = _enroll(client)
    store.revoke(node["id"])
    r = client.post(f"/agent/{node['id']}/heartbeat", json={},
                    headers=_auth(node["token"]))
    assert r.status_code == 403


def test_route_tasks_requires_valid_token(client):
    node = _enroll(client)
    assert client.get(f"/agent/{node['id']}/tasks").status_code == 401
    assert client.get(f"/agent/{node['id']}/tasks",
                      headers=_auth("agt_wrong")).status_code == 401


def test_route_tasks_pull_with_cursor(client, store):
    node = _enroll(client)
    nid = node["id"]
    store.dispatch(nid, {"n": 1})
    store.dispatch(nid, {"n": 2})
    r = client.get(f"/agent/{nid}/tasks", headers=_auth(node["token"]))
    body = r.get_json()
    assert r.status_code == 200
    assert len(body["tasks"]) == 2
    cursor = body["cursor"]
    # re-pull from the cursor: nothing new
    r2 = client.get(f"/agent/{nid}/tasks?since={cursor}",
                    headers=_auth(node["token"]))
    assert r2.get_json()["tasks"] == []


# ── Flask blueprint: operator gate ──────────────────────────────────────────
def _operator_env(monkeypatch, token="op-secret"):
    """Force the operator gate to actually bite, no network: 'open' mode with a
    token set means the token is REQUIRED (external-session path never taken)."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", token)
    return {"X-Operator-Token": token}


def test_route_nodes_operator_gated(client, monkeypatch):
    hdr = _operator_env(monkeypatch)
    assert client.get("/agent/nodes").status_code == 401           # no operator
    r = client.get("/agent/nodes", headers=hdr)
    assert r.status_code == 200 and isinstance(r.get_json(), list)


def test_route_nodes_open_when_no_token(client, monkeypatch):
    # Self-hosted default: open mode, no operator token -> permissive.
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    assert client.get("/agent/nodes").status_code == 200


def test_route_dispatch_operator_gated(client, monkeypatch):
    node = _enroll(client)
    _operator_env(monkeypatch)
    # gate runs BEFORE body validation -> 401 even with a valid task
    r = client.post(f"/agent/{node['id']}/dispatch", json={"task": {"k": 1}})
    assert r.status_code == 401


def test_route_dispatch_ok_and_queues(client, monkeypatch):
    node = _enroll(client)
    hdr = _operator_env(monkeypatch)
    r = client.post(f"/agent/{node['id']}/dispatch",
                    json={"task": {"kind": "chat", "prompt": "hi"}},
                    headers=hdr)
    assert r.status_code == 201
    queued = r.get_json()
    assert queued["task"] == {"kind": "chat", "prompt": "hi"}
    # and the node can pull it
    pull = client.get(f"/agent/{node['id']}/tasks",
                      headers=_auth(node["token"]))
    assert len(pull.get_json()["tasks"]) == 1


def test_route_dispatch_requires_task(client, monkeypatch):
    node = _enroll(client)
    hdr = _operator_env(monkeypatch)
    assert client.post(f"/agent/{node['id']}/dispatch", json={},
                       headers=hdr).status_code == 400


def test_route_dispatch_unknown_node_404(client, monkeypatch):
    hdr = _operator_env(monkeypatch)
    r = client.post("/agent/agn_nope/dispatch", json={"task": {"k": 1}},
                    headers=hdr)
    assert r.status_code == 404


def test_end_to_end_register_heartbeat_dispatch_pull(client, monkeypatch):
    """The P3.1 acceptance flow, offline."""
    node = _enroll(client, name="acceptance")
    nid, tok = node["id"], node["token"]
    # heartbeat
    assert client.post(f"/agent/{nid}/heartbeat", json={"status": "idle"},
                       headers=_auth(tok)).status_code == 200
    # operator sees it live
    hdr = _operator_env(monkeypatch)
    roster = client.get("/agent/nodes", headers=hdr).get_json()
    assert any(n["id"] == nid and n["status"] == "idle" for n in roster)
    # operator dispatches; node pulls exactly that task
    client.post(f"/agent/{nid}/dispatch", json={"task": {"do": "thing"}},
                headers=hdr)
    tasks = client.get(f"/agent/{nid}/tasks", headers=_auth(tok)).get_json()
    assert [t["task"] for t in tasks["tasks"]] == [{"do": "thing"}]


# ── central operator allowlist (operator_auth._SENSITIVE) ───────────────────
def _is_sensitive(method, path):
    app = Flask(__name__)
    with app.test_request_context(path=path, method=method):
        return operator_auth._path_is_sensitive()


def test_sensitive_covers_operator_agent_routes():
    assert _is_sensitive("GET", "/agent/nodes")
    assert _is_sensitive("POST", "/agent/agn_abc/dispatch")
    # and via the /api-prefixed form (gate strips /api first)
    assert _is_sensitive("GET", "/api/agent/nodes")
    assert _is_sensitive("POST", "/api/agent/agn_abc/dispatch")


def test_sensitive_excludes_m2m_agent_routes():
    assert not _is_sensitive("POST", "/agent/register")
    assert not _is_sensitive("POST", "/agent/agn_abc/heartbeat")
    assert not _is_sensitive("GET", "/agent/agn_abc/tasks")


# ══ P3.1b — task result reporting ══════════════════════════════════════════
# ── pure store: complete_task + get_task + size cap + legacy migration ──────
def test_complete_task_transitions_and_stores_result(store):
    nid = store.register(name="n")["id"]
    t = store.dispatch(nid, {"kind": "chat"})
    # a freshly dispatched task carries the new columns, empty
    assert t["status"] == "queued"
    assert t["result"] is None and t["finished_at"] is None
    out = store.complete_task(nid, t["seq"], status="done", result="the answer")
    assert out["ok"] is True
    view = out["task"]
    assert view["status"] == "done"
    assert view["result"] == "the answer"
    assert view["finished_at"] is not None
    # and it is persisted / re-readable
    assert store.get_task(nid, t["seq"])["result"] == "the answer"


def test_complete_task_error_status(store):
    nid = store.register(name="n")["id"]
    t = store.dispatch(nid, {"k": 1})
    out = store.complete_task(nid, t["seq"], status="error", result="boom")
    assert out["ok"] and out["task"]["status"] == "error"


def test_complete_task_unknown_seq_is_not_found(store):
    nid = store.register(name="n")["id"]
    out = store.complete_task(nid, 9999, status="done", result="x")
    assert out == {"ok": False, "reason": "not_found"}


def test_complete_task_wrong_node_is_not_found(store):
    a = store.register(name="a")["id"]
    b = store.register(name="b")["id"]
    t = store.dispatch(a, {"k": 1})              # task belongs to a
    out = store.complete_task(b, t["seq"], status="done", result="x")
    assert out == {"ok": False, "reason": "not_found"}   # b can't finalize a's
    assert store.get_task(a, t["seq"])["status"] == "queued"  # untouched


def test_complete_task_conflict_does_not_overwrite(store):
    nid = store.register(name="n")["id"]
    t = store.dispatch(nid, {"k": 1})
    store.complete_task(nid, t["seq"], status="done", result="first")
    out = store.complete_task(nid, t["seq"], status="error", result="second")
    assert out["ok"] is False and out["reason"] == "conflict"
    # FIRST report wins — the stored result/status is not clobbered
    assert out["task"]["result"] == "first"
    assert store.get_task(nid, t["seq"])["status"] == "done"


def test_complete_task_caps_oversized_result(store):
    from hugpy_fleet.central.agent_nodes import _MAX_RESULT_BYTES
    nid = store.register(name="n")["id"]
    t = store.dispatch(nid, {"k": 1})
    big = "x" * (_MAX_RESULT_BYTES + 5000)
    out = store.complete_task(nid, t["seq"], status="done", result=big)
    stored = out["task"]["result"]
    assert len(stored.encode("utf-8")) <= _MAX_RESULT_BYTES
    assert "truncated" in stored                 # marker present
    assert out["task"]["status"] == "done"       # still finalizes


def test_complete_task_non_string_result_round_trips_as_json(store):
    nid = store.register(name="n")["id"]
    t = store.dispatch(nid, {"k": 1})
    out = store.complete_task(nid, t["seq"], status="done",
                              result={"answer": 42})
    assert out["task"]["result"] == '{"answer": 42}'


def test_get_task_scoped_to_node(store):
    a = store.register(name="a")["id"]
    b = store.register(name="b")["id"]
    t = store.dispatch(a, {"k": 1})
    assert store.get_task(a, t["seq"])["seq"] == t["seq"]
    assert store.get_task(b, t["seq"]) is None   # wrong node
    assert store.get_task(a, 9999) is None       # unknown seq


def test_task_view_exposes_status(store):
    """The listing the operator/node sees carries status (+result/finished_at)."""
    nid = store.register(name="n")["id"]
    store.dispatch(nid, {"k": 1})
    row = store.tasks_since(nid, 0)[0]
    assert "status" in row and row["status"] == "queued"
    assert "result" in row and "finished_at" in row


def test_migration_backfills_result_columns_on_legacy_db(tmp_path):
    """A db whose agent_tasks predates P3.1b (no result/finished_at) is migrated
    online by _ensure — the store then completes tasks against it."""
    import sqlite3
    path = str(tmp_path / "legacy.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE agent_tasks (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
            "id TEXT NOT NULL, node_id TEXT NOT NULL, task TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'queued', created_at REAL NOT NULL)")
    s = AgentNodeStore(path=path)
    nid = s.register(name="n")["id"]             # _ensure runs the migration
    with sqlite3.connect(path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(agent_tasks)")}
    assert {"result", "finished_at"} <= cols
    t = s.dispatch(nid, {"k": 1})
    assert s.complete_task(nid, t["seq"], status="done", result="ok")["ok"]


# ── Flask blueprint: the node-token result route ────────────────────────────
def _dispatch_one(client, store, nid, task=None):
    return store.dispatch(nid, task or {"kind": "chat", "prompt": "hi"})


def test_route_result_requires_valid_token(client, store):
    node = _enroll(client)
    t = _dispatch_one(client, store, node["id"])
    # no token / wrong token → 401 (same node gate as heartbeat/tasks)
    assert client.post(f"/agent/{node['id']}/tasks/{t['seq']}/result",
                       json={"status": "done"}).status_code == 401
    assert client.post(f"/agent/{node['id']}/tasks/{t['seq']}/result",
                       json={"status": "done"},
                       headers=_auth("agt_wrong")).status_code == 401


def test_route_result_done_finalizes(client, store):
    node = _enroll(client)
    t = _dispatch_one(client, store, node["id"])
    r = client.post(f"/agent/{node['id']}/tasks/{t['seq']}/result",
                    json={"status": "done", "result": "answered"},
                    headers=_auth(node["token"]))
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "done" and body["result"] == "answered"
    assert body["finished_at"] is not None


def test_route_result_invalid_status_400(client, store):
    node = _enroll(client)
    t = _dispatch_one(client, store, node["id"])
    r = client.post(f"/agent/{node['id']}/tasks/{t['seq']}/result",
                    json={"status": "weird", "result": "x"},
                    headers=_auth(node["token"]))
    assert r.status_code == 400


def test_route_result_unknown_task_404(client):
    node = _enroll(client)
    r = client.post(f"/agent/{node['id']}/tasks/9999/result",
                    json={"status": "done"}, headers=_auth(node["token"]))
    assert r.status_code == 404


def test_route_result_wrong_node_task_404(client, store):
    a = _enroll(client, name="a")
    b = _enroll(client, name="b")
    ta = _dispatch_one(client, store, a["id"])
    # b authenticates fine, but ta is not b's task → 404 (fail-closed scope)
    r = client.post(f"/agent/{b['id']}/tasks/{ta['seq']}/result",
                    json={"status": "done"}, headers=_auth(b["token"]))
    assert r.status_code == 404


def test_route_result_idempotent_conflict_409(client, store):
    node = _enroll(client)
    t = _dispatch_one(client, store, node["id"])
    first = client.post(f"/agent/{node['id']}/tasks/{t['seq']}/result",
                        json={"status": "done", "result": "first"},
                        headers=_auth(node["token"]))
    assert first.status_code == 200
    again = client.post(f"/agent/{node['id']}/tasks/{t['seq']}/result",
                        json={"status": "error", "result": "second"},
                        headers=_auth(node["token"]))
    assert again.status_code == 409
    body = again.get_json()
    assert body["already_finalized"] is True
    assert body["result"] == "first"            # first report wins
    assert body["status"] == "done"


def test_route_result_unknown_node_410(client):
    r = client.post("/agent/agn_nope/tasks/1/result",
                    json={"status": "done"}, headers=_auth("agt_whatever"))
    assert r.status_code == 410


def test_route_result_revoked_node_403(client, store):
    node = _enroll(client)
    t = _dispatch_one(client, store, node["id"])
    store.revoke(node["id"])
    r = client.post(f"/agent/{node['id']}/tasks/{t['seq']}/result",
                    json={"status": "done"}, headers=_auth(node["token"]))
    assert r.status_code == 403


# ── Flask blueprint: the operator task-detail route ─────────────────────────
def test_route_task_detail_operator_gated(client, store, monkeypatch):
    node = _enroll(client)
    t = _dispatch_one(client, store, node["id"])
    hdr = _operator_env(monkeypatch)
    # no operator creds → 401 (before any lookup)
    assert client.get(
        f"/agent/{node['id']}/tasks/{t['seq']}").status_code == 401
    r = client.get(f"/agent/{node['id']}/tasks/{t['seq']}", headers=hdr)
    assert r.status_code == 200 and r.get_json()["seq"] == t["seq"]


def test_route_task_detail_unknown_404(client, monkeypatch):
    node = _enroll(client)
    hdr = _operator_env(monkeypatch)
    assert client.get(f"/agent/{node['id']}/tasks/9999",
                      headers=hdr).status_code == 404


def test_route_task_detail_shows_result_after_completion(client, store,
                                                         monkeypatch):
    """P3.3 read path: dispatch → node posts result → operator GET shows it."""
    node = _enroll(client)
    t = _dispatch_one(client, store, node["id"])
    client.post(f"/agent/{node['id']}/tasks/{t['seq']}/result",
                json={"status": "done", "result": "the final answer"},
                headers=_auth(node["token"]))
    hdr = _operator_env(monkeypatch)
    got = client.get(f"/agent/{node['id']}/tasks/{t['seq']}",
                     headers=hdr).get_json()
    assert got["status"] == "done"
    assert got["result"] == "the final answer"


# ── central operator allowlist covers the new operator route only ───────────
def test_sensitive_covers_operator_task_detail():
    assert _is_sensitive("GET", "/agent/agn_abc/tasks/7")
    assert _is_sensitive("GET", "/api/agent/agn_abc/tasks/7")


def test_sensitive_excludes_node_result_and_pull_routes():
    # the node-token result POST and the node-token pull stay M2M-open
    assert not _is_sensitive("POST", "/agent/agn_abc/tasks/7/result")
    assert not _is_sensitive("GET", "/agent/agn_abc/tasks")
    assert not _is_sensitive("POST", "/api/agent/agn_abc/tasks/7/result")
