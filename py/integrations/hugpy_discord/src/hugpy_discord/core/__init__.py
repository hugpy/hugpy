"""Transport-neutral core shared by the Discord cogs and the chatshare adapter.

No transport imports here (no ``discord``, no ``aiohttp``): the core only talks to
hugpy central through :class:`~hugpy_discord.hugpy_client.HugpyClient` and returns
:class:`~hugpy_discord.core.results.CommandResult` objects (or, for chat, streams
through :class:`~hugpy_discord.core.chat.ChatEngine`). Both transports render those.
"""
from __future__ import annotations

from hugpy_discord.core.chat import ChatEngine, stream_prompt
from hugpy_discord.core.results import CommandResult, OutFile

__all__ = ["ChatEngine", "stream_prompt", "CommandResult", "OutFile"]
