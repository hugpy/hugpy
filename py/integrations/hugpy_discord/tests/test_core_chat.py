"""The transport-neutral ChatEngine: streaming, history, /running, /stop."""
from __future__ import annotations

import asyncio

from hugpy_discord.core.chat import ChatEngine, stream_prompt
from _fakes import FakeBot, collecting_sender


def run(coro):
    return asyncio.run(coro)


def test_run_turn_streams_remembers_and_tags_transport():
    async def body():
        bot = FakeBot()
        eng = ChatEngine(bot, transport="chatshare", char_limit=100)
        send, messages = collecting_sender()
        await eng.run_turn(send, conv_key="cs:7", user_id="cs:42", prompt="hi there")
        # the final content is the whole streamed reply (no cursor on finish)
        assert messages[-1][0] == "Hello world"
        # history remembered as a user+assistant pair for this conversation
        hist = list(eng._history["cs:7"])
        assert hist[-2] == {"role": "user", "content": "hi there"}
        assert hist[-1] == {"role": "assistant", "content": "Hello world"}
        # transport + channel tag reached central; model resolved via the bot
        assert bot.hugpy.last_stream["transport"] == "chatshare"
        assert bot.hugpy.last_stream["channel"] == "cs:7"
        assert bot.hugpy.last_stream["model_key"] == "default-model"
    run(body())


def test_private_turn_is_not_remembered():
    async def body():
        bot = FakeBot()
        eng = ChatEngine(bot, transport="chatshare")
        send, _ = collecting_sender()
        await eng.run_turn(send, conv_key="cs:1", user_id="cs:1",
                           prompt="secret", remember=False)
        assert "cs:1" not in eng._history or not eng._history["cs:1"]
    run(body())


def test_running_and_stop_cancel_server_side():
    async def body():
        bot = FakeBot()
        bot.hugpy.gate = asyncio.Event()           # block the turn after 1 chunk
        eng = ChatEngine(bot, transport="chatshare")
        send, _ = collecting_sender()
        task = asyncio.ensure_future(
            eng.run_turn(send, conv_key="cs:9", user_id="cs:3", prompt="go"))
        for _ in range(50):                        # let the turn register as active
            await asyncio.sleep(0)
            if eng.active_snapshot():
                break
        snap = eng.active_snapshot()
        assert len(snap) == 1 and snap[0]["channel_id"] == "cs:9"
        stopped, unknown = await eng.stop(conv_key="cs:9")
        assert stopped and not unknown
        assert bot.hugpy.cancelled                 # central-side cancel happened
        await task                                 # cancellation handled, no raise
        assert not eng.active_snapshot()
    run(body())


def test_stop_unknown_turn_id():
    async def body():
        eng = ChatEngine(FakeBot(), transport="chatshare")
        stopped, unknown = await eng.stop(conv_key="cs:1", turn_id="t999")
        assert stopped == [] and unknown is True
    run(body())


def test_stream_prompt_fallback():
    async def body():
        bot = FakeBot()
        send, messages = collecting_sender()
        await stream_prompt(bot, send, prompt="summarize this", model="m",
                            file=None, char_limit=100)
        assert messages[-1][0] == "Hello world"
        assert bot.hugpy.last_stream["prompt"] == "summarize this"
        assert bot.hugpy.last_stream["model_key"] == "m"
    run(body())
