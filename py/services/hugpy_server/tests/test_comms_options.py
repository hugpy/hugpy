"""Interactive-escalation `options`: store passthrough + session-send validation.

Covers the additive, backward-compatible path that lets a keeper escalation offer
clickable choices in Discord:
  * enqueue_outbound(options=[...]) stamps `options` on the outbox msg; with no
    options it writes NO `options` key (preserves today's shape + old tests).
  * POST /discord/session/<token>/send validates `options` (list of 1..5 non-empty
    <=80-char strings) and rides them onto the outbox.

Written as real pytest functions (unlike the legacy script-style test_comms*.py
modules) so `pytest tests/test_comms*.py -q` reports an actual pass count.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_server.app.functions.imports.utils import discord_bindings as db


def _fresh_store():
    """Repoint the process-wide store singleton at a throwaway file so each test
    is isolated (the flask routes share this same singleton)."""
    tmp = tempfile.mkdtemp(prefix="comms-options-")
    db.discord_store._path = os.path.join(tmp, "discord_bindings.json")
    db.discord_store._cache = None
    db.discord_store._cache_at = 0.0
    return tmp


def _client():
    from hugpy_server.wsgi_app import get_hugpy_flask
    return get_hugpy_flask().test_client()


def _mint(channel_id="55501"):
    """A keeper-brained bridge + a scoped session token on one channel."""
    db.discord_store.add_bridge(channel_id=channel_id, brain="keeper")
    token, _ = db.discord_store.add_session(channel_id=channel_id)
    return channel_id, token


# ── store unit: enqueue_outbound options passthrough ──────────────────────
def test_enqueue_outbound_carries_options():
    _fresh_store()
    m = db.enqueue_outbound(content="Ship?", channel_id="123", options=["A", "B"])
    assert m["options"] == ["A", "B"]
    drained = db.drain_outbound()
    mine = [d for d in drained if d["id"] == m["id"]]
    assert mine and mine[0].get("options") == ["A", "B"]


def test_enqueue_outbound_no_options_writes_no_key():
    _fresh_store()
    m = db.enqueue_outbound(content="hello", channel_id="123")
    assert "options" not in m
    drained = db.drain_outbound()
    mine = [d for d in drained if d["id"] == m["id"]]
    assert mine and "options" not in mine[0]


def test_enqueue_outbound_empty_options_writes_no_key():
    _fresh_store()
    for empty in ([], None):
        m = db.enqueue_outbound(content="hello", channel_id="123", options=empty)
        assert "options" not in m


# ── route: POST /discord/session/<token>/send ─────────────────────────────
def test_session_send_valid_options_201_and_rides_onto_outbox():
    _fresh_store()
    channel_id, token = _mint()
    r = _client().post(
        f"/discord/session/{token}/send",
        json={"content": "Ship now or hold?",
              "options": ["Ship", "Hold", "Investigate"]},
    )
    assert r.status_code == 201, r.get_data(as_text=True)
    doc = db.discord_store._read_unlocked()   # fresh disk read, bypasses cache
    outs = [m for m in doc["outbox"] if m.get("channel_id") == channel_id]
    assert outs, "no outbound queued for the channel"
    assert outs[-1].get("options") == ["Ship", "Hold", "Investigate"]
    # transcript record mirrors the options too (nicety)
    bridge = db.bridge_for_channel(channel_id)
    msgs = db.get_bridge_messages(bridge["id"])
    assert msgs and msgs[-1].get("options") == ["Ship", "Hold", "Investigate"]


def test_session_send_options_are_stripped():
    _fresh_store()
    channel_id, token = _mint()
    r = _client().post(
        f"/discord/session/{token}/send",
        json={"content": "Q?", "options": ["  Ship  ", "Hold"]},
    )
    assert r.status_code == 201, r.get_data(as_text=True)
    doc = db.discord_store._read_unlocked()
    outs = [m for m in doc["outbox"] if m.get("channel_id") == channel_id]
    assert outs[-1].get("options") == ["Ship", "Hold"]


def test_session_send_too_many_options_400():
    _fresh_store()
    _channel_id, token = _mint()
    r = _client().post(
        f"/discord/session/{token}/send",
        json={"content": "Q?", "options": ["a", "b", "c", "d", "e", "f"]},
    )
    assert r.status_code == 400
    assert "at most 5" in r.get_data(as_text=True)


def test_session_send_empty_string_option_400():
    _fresh_store()
    _channel_id, token = _mint()
    r = _client().post(
        f"/discord/session/{token}/send",
        json={"content": "Q?", "options": ["Ship", "   "]},
    )
    assert r.status_code == 400


def test_session_send_non_list_options_400():
    _fresh_store()
    _channel_id, token = _mint()
    r = _client().post(
        f"/discord/session/{token}/send",
        json={"content": "Q?", "options": "Ship"},
    )
    assert r.status_code == 400


def test_session_send_overlong_option_400():
    _fresh_store()
    _channel_id, token = _mint()
    r = _client().post(
        f"/discord/session/{token}/send",
        json={"content": "Q?", "options": ["x" * 81]},
    )
    assert r.status_code == 400


def test_session_send_no_options_unchanged_201():
    """Backward compatibility: omitting `options` behaves exactly as before —
    201 and no `options` key on the outbox message."""
    _fresh_store()
    channel_id, token = _mint()
    r = _client().post(
        f"/discord/session/{token}/send",
        json={"content": "plain text"},
    )
    assert r.status_code == 201, r.get_data(as_text=True)
    doc = db.discord_store._read_unlocked()
    outs = [m for m in doc["outbox"] if m.get("channel_id") == channel_id]
    assert outs and "options" not in outs[-1]
