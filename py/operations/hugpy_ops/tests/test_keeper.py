"""Keeper: the action loop is BOUNDED (``--max-steps``), every command runs
through the executor with a wall-clock cap and truncated observation, and
the central URL comes from the platform resolver."""
from __future__ import annotations

import argparse
import subprocess

import pytest

from hugpy_ops import keeper


def _args(max_steps=3):
    return argparse.Namespace(central="http://central:1", model="m",
                              max_tokens=8, max_steps=max_steps)


class ScriptedExecutor:
    target = "faux"
    where = "the fake box"

    def __init__(self):
        self.commands = []

    def argv(self, command):
        self.commands.append(command)
        return ["true"]


def test_action_loop_stops_after_max_steps(monkeypatch):
    """A model that ALWAYS asks for another action is cut off at max_steps —
    the keeper never loops forever, and every step went through run_action."""
    turns = []

    def always_act(central, model, messages, api_key, max_tokens):
        turns.append(len(messages))
        yield "```action\necho hi\n```"

    ran = []
    monkeypatch.setattr(keeper, "stream_chat", always_act)
    monkeypatch.setattr(keeper, "run_action",
                        lambda ex, cmd, lim: ran.append(cmd) or (0, "hi"))
    messages = [{"role": "system", "content": "x"}]
    reply = keeper.run_agent_turn(_args(max_steps=3), ScriptedExecutor(),
                                  messages, "", 100, echo=False)
    assert len(turns) == 3 and ran == ["echo hi"] * 3
    assert "```action" in reply
    # every action fed an observation back; nothing ran outside the loop
    assert sum(1 for m in messages if m["role"] == "user") == 3


def test_action_loop_returns_on_first_plain_reply(monkeypatch):
    scripted = iter(["```action\nls\n```", "all good"])
    monkeypatch.setattr(keeper, "stream_chat",
                        lambda *a, **k: iter([next(scripted)]))
    ran = []
    monkeypatch.setattr(keeper, "run_action",
                        lambda ex, cmd, lim: ran.append(cmd) or (0, "out"))
    reply = keeper.run_agent_turn(_args(max_steps=8), ScriptedExecutor(),
                                  [{"role": "system", "content": "x"}], "", 100,
                                  echo=False)
    assert reply == "all good" and ran == ["ls"]


def test_run_action_is_bounded_and_truncated(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise subprocess.TimeoutExpired(cmd="x", timeout=300)
    monkeypatch.setattr(keeper.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(
                            subprocess.TimeoutExpired(cmd="x", timeout=300)))
    rc, out = keeper.run_action(ScriptedExecutor(), "sleep 999", 50)
    assert rc == 124 and "timed out" in out

    class P:
        returncode = 0
        stdout = "y" * 500
        stderr = ""
    monkeypatch.setattr(keeper.subprocess, "run", lambda *a, **k: P())
    rc, out = keeper.run_action(ScriptedExecutor(), "yes", 50)
    assert rc == 0 and out.startswith("y" * 50) and "truncated" in out


def test_parse_action_prefers_last_fenced_block():
    text = "thinking\n```action\nfirst\n```\nmore\n```bash\nsecond\n```"
    assert keeper.parse_action(text) == "second"
    assert keeper.parse_action("Action: uptime") == "uptime"
    assert keeper.parse_action("no command here") is None


def test_executor_factory_and_lxc_argv():
    assert isinstance(keeper.make_executor("local", "1000", "1000", "/home/u"), keeper.LocalExec)
    lxc = keeper.make_executor("lxc:box", "1000", "1000", "/home/u")
    assert lxc.argv("id")[:4] == ["lxc", "exec", "box", "--user"]
    with pytest.raises(ValueError):
        keeper.make_executor("lxc:", "1", "1", "/h")
    with pytest.raises(ValueError):
        keeper.make_executor("ssh:box", "1", "1", "/h")


def test_central_default_comes_from_platform(monkeypatch):
    monkeypatch.setenv("HUGPY_BASE_URL", "http://platform-central:7002/")
    assert keeper.central_base_url() == "http://platform-central:7002"
    from hugpy_platform.central import central_base_url as platform_resolver
    assert keeper.central_base_url is platform_resolver
