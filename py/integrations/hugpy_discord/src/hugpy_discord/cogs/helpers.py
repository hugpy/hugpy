"""Discord-side cog helpers: model autocomplete, attachment forwarding, and
non-streaming result delivery.

The transport-neutral helpers (``cached_models``, ``model_label``,
``clean_model_key``, ``split_items``, ``model_choice_pairs``) live in
``hugpy_discord.core.models`` and are re-exported here for the cogs. This module
keeps only the Discord-specific pieces (``discord.Attachment`` / ``discord.File``
/ ``app_commands.Choice``).
"""
from __future__ import annotations

import io

import discord
from discord import app_commands

from hugpy_discord.config import MAX_ATTACHMENT_BYTES, MESSAGE_CHAR_LIMIT
from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.core.models import (  # re-exported for the cogs
    cached_models, clean_model_key, model_choice_pairs, model_label, split_items,
)
from hugpy_discord.core.results import CommandResult

__all__ = [
    "cached_models", "clean_model_key", "model_choice_pairs", "model_label",
    "split_items", "model_autocomplete", "forward_attachment", "send_long",
    "render_result",
]

_COLOR = {"blurple": discord.Colour.blurple, "red": discord.Colour.red}


async def model_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    pairs = await model_choice_pairs(interaction.client, current)
    return [app_commands.Choice(name=p["name"], value=p["value"]) for p in pairs]


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


async def render_result(interaction: discord.Interaction,
                        result: CommandResult) -> None:
    """Render a core :class:`CommandResult` onto a (already-deferred) Discord
    interaction, preserving the historical idioms: attachments via
    ``discord.File``, embed-style replies via ``discord.Embed`` (fields for
    sections, description otherwise), long text via :func:`send_long`."""
    eph = result.ephemeral
    if result.files:
        files = [discord.File(io.BytesIO(f.data), filename=f.filename)
                 for f in result.files]
        await interaction.followup.send(
            (result.text or "")[:MESSAGE_CHAR_LIMIT], files=files, ephemeral=eph)
        return
    if result.sections or result.title:
        embed = discord.Embed(colour=_COLOR.get(result.color, _COLOR["blurple"])())
        if result.title:
            embed.title = result.title
        if result.text:
            embed.description = result.text[:4000]
        for name, value in result.sections:
            embed.add_field(name=name, value=value, inline=False)
        if result.footer:
            embed.set_footer(text=result.footer)
        await interaction.followup.send(embed=embed, ephemeral=eph)
        return
    if result.long:
        async def _send(**kw):
            return await interaction.followup.send(ephemeral=eph, **kw)
        await send_long(_send, result.text, filename=result.filename)
        return
    await interaction.followup.send(result.text, ephemeral=eph)
