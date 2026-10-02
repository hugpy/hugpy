"""hugpy-link presence: merge() folds the connection into liveness rows."""
from hugpy_fleet.central import presence

NOW = 1_000_000.0


def _rows(monkeypatch, pres):
    monkeypatch.setattr(presence, "rows", lambda: pres)


def _live(status="offline", wid="w1"):
    return [{"id": wid, "name": wid, "status": status}]


def test_no_daemon_row_untouched(monkeypatch):
    _rows(monkeypatch, {})
    live = presence.merge(_live("offline"), now=NOW)
    assert live[0]["status"] == "offline"
    assert live[0]["link"] is None
    assert "state_stale" not in live[0]


def test_link_up_worker_up_reads_online_with_stale_state(monkeypatch):
    _rows(monkeypatch, {"w1": {"ts": NOW - 3, "worker_state": "up", "version": "x", "payload": {}}})
    row = presence.merge(_live("offline"), now=NOW)[0]
    assert row["status"] == "online"
    assert row["link"] == "up" and row["worker_state"] == "up"
    assert row["state_stale"] is True


def test_link_up_worker_busy_online(monkeypatch):
    _rows(monkeypatch, {"w1": {"ts": NOW - 1, "worker_state": "busy", "version": None, "payload": {}}})
    row = presence.merge(_live("offline"), now=NOW)[0]
    assert row["status"] == "online" and row["worker_state"] == "busy"


def test_link_up_worker_restarting_or_down_keeps_heartbeat_status(monkeypatch):
    for state in ("restarting", "down"):
        _rows(monkeypatch, {"w1": {"ts": NOW - 1, "worker_state": state, "version": None, "payload": {}}})
        row = presence.merge(_live("offline"), now=NOW)[0]
        assert row["status"] == "offline" and row["worker_state"] == state and row["link"] == "up"


def test_link_down_drops_worker_state(monkeypatch):
    monkeypatch.delenv("HUGPY_LINK_STALE_S", raising=False)
    _rows(monkeypatch, {"w1": {"ts": NOW - 60, "worker_state": "up", "version": None, "payload": {}}})
    row = presence.merge(_live("offline"), now=NOW)[0]
    assert row["link"] == "down" and row["worker_state"] is None and row["status"] == "offline"


def test_fresh_heartbeat_stays_online_even_with_link_down(monkeypatch):
    _rows(monkeypatch, {"w1": {"ts": NOW - 600, "worker_state": "down", "version": None, "payload": {}}})
    row = presence.merge(_live("online"), now=NOW)[0]
    assert row["status"] == "online" and row["state_stale"] is False and row["link"] == "down"


def test_stale_threshold_env(monkeypatch):
    monkeypatch.setenv("HUGPY_LINK_STALE_S", "60")
    _rows(monkeypatch, {"w1": {"ts": NOW - 30, "worker_state": "up", "version": None, "payload": {}}})
    assert presence.merge(_live(), now=NOW)[0]["link"] == "up"
    monkeypatch.setenv("HUGPY_LINK_STALE_S", "1")  # floored at 5 s
    assert presence.link_stale_s() == 5.0


def test_fail_open_without_dsn(monkeypatch):
    monkeypatch.setattr(presence, "_dsn", lambda: None)
    assert presence.rows() == {}
    assert presence.record("w1", {"worker_state": "up"}) is False
    live = presence.merge(_live("online"), now=NOW)
    assert live[0]["status"] == "online" and live[0]["link"] is None


def test_record_normalizes_unknown_state(monkeypatch):
    seen = {}

    class Cur:
        def execute(self, sql, params=None):
            seen["params"] = params

    monkeypatch.setattr(presence, "_run", lambda fn: fn(Cur()))
    assert presence.record("w1", {"worker_state": "Exploded", "version": "v"}) is True
    assert seen["params"][2] is None and seen["params"][3] == "v"


def test_empty_live_passthrough():
    assert presence.merge(None) is None
    assert presence.merge([]) == []


def test_dsn_falls_back_to_registry_dsn(monkeypatch):
    monkeypatch.delenv("HUGPY_REGISTRY_PG_DSN", raising=False)
    monkeypatch.setenv("HUGPY_REGISTRY_DB", "pg")
    monkeypatch.setenv("SOLCATCHER_POSTGRESQL_HOST", "dbhost")
    assert "host=dbhost" in presence._dsn()
    monkeypatch.setenv("HUGPY_REGISTRY_DB", "json")
    assert presence._dsn() is None
