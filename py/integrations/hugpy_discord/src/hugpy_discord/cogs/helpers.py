"""Shared cog helpers: model autocomplete, attachment forwarding, and
non-streaming result delivery."""
from __future__ import annotations

import io
import re
import time

import discord
from discord import app_commands

from hugpy_discord.config import MAX_ATTACHMENT_BYTES, MESSAGE_CHAR_LIMIT
from hugpy_discord.hugpy_client import HugpyError

_model_cache: tuple[float, list[dict]] = (0.0, [])
_MODEL_CACHE_TTL = 60.0


async def cached_models(bot) -> list[dict]:
    global _model_cache
    stamp, models = _model_cache
    if time.monotonic() - stamp > _MODEL_CACHE_TTL:
        try:
            models = await bot.hugpy.list_models()
            _model_cache = (time.monotonic(), models)
        except HugpyError:
            pass  # serve stale (or empty) rather than fail autocomplete
    return models


def model_label(model: dict) -> str:
    key = model.get("key") or model.get("name") or "?"
    status = (model.get("status") or "").strip()
    # Surface a status suffix ONLY when it's something other than the ordinary
    # "installed". An "(installed)" on every model is noise, and — because this
    # label is what users read in the dropdown / `/models` list — it collides
    # with the plain model key when the name is typed or copied as input.
    if status and status.lower() != "installed":
        return f"{key} ({status})"
    return key


_STATUS_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def clean_model_key(s: str | None) -> str | None:
    """Strip a trailing ' (status)' a user may have copied from a model label
    (valid model keys never end in a parenthetical), so 'Foo (installed)' still
    resolves to 'Foo'."""
    if not s:
        return s
    return _STATUS_SUFFIX.sub("", s).strip() or s


async def model_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    models = await cached_models(interaction.client)
    current = current.lower()
    choices = []
    for model in models:
        key = model.get("key") or model.get("name") or ""
        if current in key.lower():
            choices.append(app_commands.Choice(name=model_label(model)[:100], value=key))
        if len(choices) == 25:
            break
    return choices


async def forward_attachment(bot, attachment: discord.Attachment) -> str:
    """Pull an attachment from Discord and upload it to central.

    Returns the server-side path to pass as the chat body's ``file``.
    """
    if attachment.size > MAX_ATTACHMENT_BYTES:
        raise HugpyError(
            f"attachment too large ({attachment.size // (1024 * 1024)} MB; "
            f"limit {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB)"
        )
    data = await attachment.read()
    uploaded = await bot.hugpy.upload(attachment.filename, data)
    return uploaded["path"]


def split_items(raw: str) -> list[str]:
    """Split user input into items: '||' wins, else newlines, else one item."""
    sep = "||" if "||" in raw else "\n"
    return [part.strip() for part in raw.split(sep) if part.strip()]


async def send_long(send, text: str, *, filename: str = "result.txt") -> None:
    """Deliver a one-shot (non-streamed) result.

    Short replies go inline; anything past two messages becomes a file
    attachment so a long transcript/summary doesn't flood the channel.
    """
    text = text.strip() or "*<empty result>*"
    if len(text) <= MESSAGE_CHAR_LIMIT:
        await send(content=text)
        return
    if len(text) <= MESSAGE_CHAR_LIMIT * 2:
        for start in range(0, len(text), MESSAGE_CHAR_LIMIT):
            await send(content=text[start:start + MESSAGE_CHAR_LIMIT])
        return
    buffer = io.BytesIO(text.encode("utf-8"))
    await send(
        content=text[: MESSAGE_CHAR_LIMIT - 100] + "\n… *(full result attached)*",
        file=discord.File(buffer, filename=filename),
    )
