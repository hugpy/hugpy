"""Stores accept configured paths and otherwise route through env / hugpy_platform
— never a hardcoded monolith path."""
from __future__ import annotations

import json
import os

from hugpy_control import calllog
from hugpy_control.principals import PrincipalStore, allowed
from hugpy_control.settings import SettingsStore


def test_settings_store_explicit_path_roundtrip(tmp_path):
    store = SettingsStore(path=str(tmp_path / "s.json"))
    assert store.path() == str(tmp_path / "s.json")
    assert store.get("ns", "k", "d") == "d"
    store.set("ns", "k", {"a": 1})
    assert store.merge("ns", "k", {"b": 2}) == {"a": 1, "b": 2}
    assert store.increment("ns", "n") == 1 and store.increment("ns", "n", 2) == 3
    assert store.namespaces() == ["ns"] and store.all("ns") == {"k": {"a": 1, "b": 2}, "n": 3}
    assert store.delete("ns", "k") and not store.delete("ns", "k")
    assert json.loads((tmp_path / "s.json").read_text())["ns"]["n"] == 3


def test_settings_and_principals_paths_route_through_env_then_platform(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_SETTINGS_PATH", str(tmp_path / "env-settings.json"))
    monkeypatch.setenv("HUGPY_PRINCIPALS_PATH", str(tmp_path / "env-principals.json"))
    assert SettingsStore().path() == str(tmp_path / "env-settings.json")
    assert PrincipalStore().path() == str(tmp_path / "env-principals.json")

    monkeypatch.delenv("HUGPY_SETTINGS_PATH")
    monkeypatch.delenv("HUGPY_PRINCIPALS_PATH")
    monkeypatch.setenv("PROJECTS_HOME", str(tmp_path / "projects"))
    assert SettingsStore().path() == str(tmp_path / "projects" / "settings.json")
    assert PrincipalStore().path() == str(tmp_path / "projects" / "principals.json")

    monkeypatch.delenv("PROJECTS_HOME")
    from hugpy_platform.constants import PROJECTS_HOME   # sandboxed by conftest
    assert SettingsStore().path() == os.path.join(PROJECTS_HOME, "settings.json")


def test_principal_store_tokens_and_capabilities(tmp_path):
    store = PrincipalStore(path=str(tmp_path / "p.json"))
    p = store.create(kind="user", name="alice")
    tok = store.issue_token(p.id)
    assert store.resolve_token(tok).id == p.id
    assert store.resolve_token("nope") is None
    assert allowed(store.resolve_operator(), "settings.write")
    assert store.revoke(p.id) and store.resolve_token(tok) is None


def test_calllog_path_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_CALL_LOG", str(tmp_path / "calls.jsonl"))
    assert calllog.path() == str(tmp_path / "calls.jsonl")
    monkeypatch.delenv("HUGPY_CALL_LOG")
    monkeypatch.setenv("HUGPY_COMMS_DB", str(tmp_path / "db" / "comms.db"))
    assert calllog.path() == str(tmp_path / "db" / "calls.jsonl")

    monkeypatch.setenv("HUGPY_CALL_LOG", str(tmp_path / "calls.jsonl"))
    job = type("J", (), {"id": "j-1", "kind": "chat", "model_key": "m", "transport": "web",
                         "principal": None, "channel": None, "worker": None,
                         "created_ts": 1.0, "ended_ts": None, "error": None})()
    calllog.record("start", job)
    calllog.record("end", job, status="done", duration_ms=5)
    rows = calllog.read()
    assert len(rows) == 1 and rows[0]["id"] == "j-1" and rows[0]["status"] == "done"
