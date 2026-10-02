"""The chatshare adapter against a fake chat-service websocket (no real network)."""
from __future__ import annotations

import asyncio

from hugpy_discord.chatshare import ChatshareAdapter
from hugpy_discord.core.specs import command_specs
from hugpy_discord.hugpy_client import HugpyError
from _fakes import FakeBot, FakeService


def run(coro):
    return asyncio.run(coro)


def make(bot=None):
    bot = bot or FakeBot()
    adapter = ChatshareAdapter(bot, url="ws://test/chat/bot", token="csb_test")
    fake = FakeService()
    fake.adapter = adapter
    adapter._ws = fake
    adapter.bot_name = "hugpy"
    return bot, adapter, fake


def invoke_command(name, args=None, *, author_id=5, room_id=9):
    return {"invocation_id": "i1", "kind": "command", "author": {"id": author_id},
            "room": {"id": room_id},
            "command": {"name": name, "args": args or {}}}


def test_bot_ready_registers_all_specs():
    async def body():
        bot, adapter, fake = make()
        await adapter._set_commands()
        cs = fake.frames("commands.set")
        assert cs and len(cs[0]["commands"]) == len(command_specs()) == 22
        names = {c["name"] for c in cs[0]["commands"]}
        assert "model set" in names and "chat" in names and "describe" in names
        assert "link" not in names                    # Discord-only, dropped
    run(body())


def test_specs_gate_admin_only():
    specs = {c["name"]: c for c in command_specs()}
    assert specs["download"].get("admin_only") is True
    assert specs["canceljob"].get("admin_only") is True
    assert specs["chat"].get("admin_only") is not True
    # model sub-command + autocomplete + attachment + ranges mirrored
    model_set = specs["model set"]
    assert model_set["options"][0]["autocomplete"] is True
    chat_opts = {o["name"]: o for o in specs["chat"]["options"]}
    assert chat_opts["attachment"]["type"] == "attachment"
    assert chat_opts["temperature"]["min"] == 0.0 and chat_opts["temperature"]["max"] == 2.0


def test_bot_ready_frame_sets_name_and_body_max():
    async def body():
        bot, adapter, fake = make()
        adapter.bot_name = None
        adapter._on_frame({"t": "ev", "ev": "bot.ready",
                           "bot": {"name": "hugpy-ai"}, "limits": {"body_max": 4096}})
        assert adapter.bot_name == "hugpy-ai"
        assert adapter.body_max == 4096 and adapter.engine._char_limit == 4096
    run(body())


def test_command_status_renders_text():
    async def body():
        bot, adapter, fake = make()
        await adapter._handle_invoke(invoke_command("status"))
        assert fake.frames("invocation.defer")          # deferred like Discord
        assert any("hugpy central" in b for b in fake.bodies())
    run(body())


def test_command_imagine_uploads_file():
    async def body():
        import base64
        bot, adapter, fake = make()
        bot.hugpy.prompt_result = {"images": [{"b64": base64.b64encode(b"PNG").decode()}]}
        await adapter._handle_invoke(invoke_command("imagine", {"prompt": "cat"}))
        assert fake.frames("file.put")                  # image shipped via file.put
        send = fake.frames("msg.send")[-1]
        assert send["attachments"] and "bot_file_id" in send["attachments"][0]
    run(body())


def test_command_reset_is_ephemeral_and_not_deferred():
    async def body():
        bot, adapter, fake = make()
        await adapter._handle_invoke(invoke_command("reset"))
        assert not fake.frames("invocation.defer")
        send = fake.frames("msg.send")[-1]
        assert send.get("ephemeral") is True and "history cleared" in send["body"]
    run(body())


def test_command_model_set_namespaces_identity():
    async def body():
        bot, adapter, fake = make()
        await adapter._handle_invoke(
            invoke_command("model set", {"model": "alpha"}, author_id=42))
        # stored under the cs: namespace, never colliding with a Discord id
        assert bot._models == {"cs:42": "alpha"}
        assert "42" not in bot._models
    run(body())


def test_mention_streams_reply_and_strips_name():
    async def body():
        bot, adapter, fake = make()
        frame = {"invocation_id": "i2", "kind": "mention",
                 "author": {"id": 3}, "room": {"id": 8},
                 "message": {"body": "@hugpy tell me", "attachments": []}}
        await adapter._handle_invoke(frame)
        all_bodies = fake.bodies("msg.send") + fake.bodies("msg.edit")
        assert "Hello world" in all_bodies
        # prompt reached central with the @name stripped
        assert bot.hugpy.last_stream["messages"][-1]["content"] == "tell me"
        assert bot.hugpy.last_stream["channel"] == "cs:8"
    run(body())


def test_dm_streams_reply():
    async def body():
        bot, adapter, fake = make()
        frame = {"invocation_id": "i3", "kind": "dm", "author": {"id": 1},
                 "room": {"id": 2}, "message": {"body": "hey", "attachments": []}}
        await adapter._handle_invoke(frame)
        all_bodies = fake.bodies("msg.send") + fake.bodies("msg.edit")
        assert "Hello world" in all_bodies
    run(body())


def test_chat_private_is_ephemeral_stream():
    async def body():
        bot, adapter, fake = make()
        await adapter._handle_invoke(
            invoke_command("chat", {"prompt": "hi", "private": True}))
        sends = fake.frames("msg.send")
        assert sends and all(s.get("ephemeral") is True for s in sends)
    run(body())


def test_autocomplete_returns_model_choices():
    async def body():
        bot, adapter, fake = make()
        await adapter._handle_autocomplete(
            {"aid": "a1", "option": "model", "value": "al"})
        res = fake.frames("autocomplete.result")[-1]
        assert {"name": "alpha", "value": "alpha"} in res["choices"]
    run(body())


def test_autocomplete_non_model_is_empty():
    async def body():
        bot, adapter, fake = make()
        await adapter._handle_autocomplete({"aid": "a2", "option": "task", "value": "t"})
        assert fake.frames("autocomplete.result")[-1]["choices"] == []
    run(body())


def test_forward_attachment_rejects_too_large():
    async def body():
        bot, adapter, fake = make()
        from hugpy_discord import config
        big = config.MAX_ATTACHMENT_BYTES + 1
        try:
            await adapter._forward_attachment(
                {"share_url": "http://x/f", "filename": "f.bin", "size": big})
            assert False, "expected HugpyError"
        except HugpyError as exc:
            assert "too large" in str(exc)
    run(body())


def test_download_command_edits_progress():
    async def body():
        bot, adapter, fake = make()
        from hugpy_discord.core import commands as core
        orig = core.JOB_POLL_SECONDS
        core.JOB_POLL_SECONDS = 0.001
        try:
            await adapter._handle_invoke(invoke_command("download", {"model": "alpha"}))
        finally:
            core.JOB_POLL_SECONDS = orig
        sends = fake.frames("msg.send")
        edits = fake.frames("msg.edit")
        assert sends and sends[0]["body"].startswith("⬇️ downloading")
        assert edits and edits[-1]["body"].startswith("✅")
    run(body())
