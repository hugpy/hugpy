"""Progressively edit Discord messages as tokens stream in.

Discord rate-limits edits and caps messages at 2000 chars, so the streamer
throttles edits and rolls over to a fresh message when a chunk fills up.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable

import discord

from hugpy_discord.config import MESSAGE_CHAR_LIMIT

EDIT_INTERVAL_SECONDS = 1.2
CURSOR = " ▌"

# Anything we can call with text that returns a message we can later edit:
# channel.send, interaction.followup.send, message.reply, ...
Sender = Callable[[str], Awaitable[discord.Message]]


class MessageStreamer:
    def __init__(self, send: Sender):
        self._send = send
        self._message: discord.Message | None = None
        self._chunk = ""        # text belonging to the message currently being edited
        self._pending = ""      # text not yet flushed to discord
        self._last_edit = 0.0
        self.full_text = ""

    async def feed(self, text: str) -> None:
        self.full_text += text
        self._pending += text
        if time.monotonic() - self._last_edit >= EDIT_INTERVAL_SECONDS:
            await self._flush(final=False)

    async def finish(self) -> None:
        await self._flush(final=True)
        if self._message is None and not self.full_text:
            self._message = await self._send("*(empty response)*")

    async def fail(self, reason: str) -> None:
        note = f"\n\n⚠️ {reason}" if self.full_text else f"⚠️ {reason}"
        self._pending += note
        self.full_text += note
        await self._flush(final=True)

    async def _flush(self, *, final: bool) -> None:
        while self._pending:
            room = MESSAGE_CHAR_LIMIT - len(self._chunk)
            take, self._pending = self._pending[:room], self._pending[room:]
            self._chunk += take

            rollover = bool(self._pending)
            cursor = "" if (final and not rollover) else CURSOR
            content = self._chunk + ("" if rollover else cursor)

            if self._message is None:
                self._message = await self._send(content)
            else:
                await self._message.edit(content=content)

            if rollover:
                # Current message is full: seal it and continue in a new one.
                await self._message.edit(content=self._chunk)
                self._message = None
                self._chunk = ""
            elif not final:
                break  # flushed what fits; wait for the next throttle window

        self._last_edit = time.monotonic()
        if final and self._message is not None and self._chunk:
            await self._message.edit(content=self._chunk)
