"""Regression (2026-10-01): the console-trace middleware wrote to the fleet's
main comms SQLite twice per traced request (up to 12000 span rows), and agent
heartbeats on that DB failed with "database is locked".

The trace now has its own WAL file beside the comms DB, and its writes are
best-effort. Concurrent heartbeat-style writers on the main DB see no lock
errors, and a locked trace DB never fails or blocks a request."""
from __future__ import annotations

import os
import sqlite3
import threading
import time

import pytest
from flask import Flask, jsonify

ct = pytest.importorskip("hugpy_server.app.routes.console_trace_routes")


@pytest.fixture
def env(monkeypatch, tmp_path):
    main = tmp_path / "comms" / "hugpy-comms.db"
    main.parent.mkdir()
    monkeypatch.setenv("HUGPY_COMMS_DB", str(main))
    monkeypatch.delenv("HUGPY_CONSOLE_TRACE_DB", raising=False)
    con = sqlite3.connect(main)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE hb (id INTEGER PRIMARY KEY, at REAL)")
    con.commit(); con.close()
    app = Flask(__name__)
    ct.install_console_trace(app)
    app.register_blueprint(ct.trace_bp)

    @app.route("/work")
    def work():
        return jsonify({"n": sum(range(2000))})

    return app, str(main)


def test_trace_db_is_not_the_main_db(env):
    _, main = env
    p = ct.trace_db_path()
    assert os.path.abspath(p) != os.path.abspath(main)
    assert os.path.dirname(p) == os.path.dirname(main)


def test_concurrent_heartbeats_and_traced_requests(env):
    app, main = env
    errors, statuses = [], []
    stop = time.time() + 3.0

    def heartbeats():
        while time.time() < stop:
            try:
                c = sqlite3.connect(main, timeout=5.0)
                c.execute("INSERT INTO hb(at) VALUES (?)", (time.time(),))
                c.commit(); c.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

    def traced():
        client = app.test_client()
        for _ in range(40):
            r = client.get("/work", headers={"X-Hugpy-Trace": "1"})
            statuses.append(r.status_code)

    ts = [threading.Thread(target=heartbeats) for _ in range(3)]
    ts += [threading.Thread(target=traced) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []
    assert statuses and set(statuses) == {200}
    tables = {r[0] for r in sqlite3.connect(main).execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert not any(t.startswith("console_trace") for t in tables)
    n = sqlite3.connect(ct.trace_db_path()).execute(
        "SELECT COUNT(*) FROM console_trace_requests WHERE status=200").fetchone()[0]
    assert n > 0


def test_locked_trace_db_drops_row_and_never_blocks(env):
    app, _ = env
    client = app.test_client()
    client.get("/work", headers={"X-Hugpy-Trace": "1"})   # creates the schema
    holder = sqlite3.connect(ct.trace_db_path(), timeout=0)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        t0 = time.time()
        r = client.get("/work", headers={"X-Hugpy-Trace": "1"})
        assert r.status_code == 200
        assert time.time() - t0 < 2.0
        r = client.post("/console/trace/events", json={"kind": "x", "payload": {}})
        assert r.status_code == 200 and r.get_json()["stored"] is False
    finally:
        holder.rollback(); holder.close()


def test_untraced_request_touches_no_db(env, monkeypatch):
    app, _ = env
    called = []
    monkeypatch.setattr(ct, "_db", lambda *a, **k: called.append(1))
    assert app.test_client().get("/work").status_code == 200
    assert called == []
