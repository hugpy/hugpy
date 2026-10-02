"""Task commands over hugpy's dispatch categories: summarize, keywords,
transcribe, describe.

The behavior (central calls, result shaping, and the /chat/stream compatibility
fallback for a central that predates POST /prompt) lives in
``hugpy_discord.core.commands`` / ``hugpy_discord.core.chat``; this cog is the
Discord face (interactions, attachments, embeds/files)."""
from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from hugpy_discord.config import MESSAGE_CHAR_LIMIT
from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.core import commands as core
from hugpy_discord.core.chat import stream_prompt
from hugpy_discord.cogs.helpers import (
    forward_attachment, model_autocomplete, render_result,
)

# Re-exported for compatibility.
KEYWORD_PRESETS = core.KEYWORD_PRESETS
SUMMARY_MODES = core.SUMMARY_MODES
WHISPER_SIZES = core.WHISPER_SIZES


class ToolsCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # ── shared plumbing ────────────────────────────────────────────────────
    async def _upload(self, interaction, attachment) -> str | None:
        """Forward an attachment to central; reports failure itself."""
        try:
            return await forward_attachment(self.bot, attachment)
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}", ephemeral=True)
            return None

    async def _deliver(self, interaction, result, *, model) -> None:
        """Render a core result, or run the /chat/stream fallback it asked for."""
        if result.fallback_prompt is not None:
            await stream_prompt(
                self.bot,
                lambda content: interaction.followup.send(content, wait=True),
                prompt=result.fallback_prompt,
                model=model or self.bot.model_for(interaction.user.id),
                file=result.fallback_file,
                char_limit=MESSAGE_CHAR_LIMIT,
            )
            return
        await render_result(interaction, result)

    # ── /summarize ─────────────────────────────────────────────────────────
    @app_commands.command(name="summarize", description="Summarize text or a file (dedicated summarizer)")
    @app_commands.describe(
        text="Text to summarize",
        attachment="Or attach a document",
        mode="Summary length (default: central decides)",
        preset="Named parameter bundle: short/medium/long",
        model="Model override (default: central's summarizer)",
    )
    @app_commands.choices(
        mode=[app_commands.Choice(name=m, value=m) for m in core.SUMMARY_MODES],
        preset=[app_commands.Choice(name=p, value=p) for p in ("short", "medium", "long")],
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def summarize(
        self,
        interaction: discord.Interaction,
        text: str | None = None,
        attachment: discord.Attachment | None = None,
        mode: str | None = None,
        preset: str | None = None,
        model: str | None = None,
    ) -> None:
        if not text and not attachment:
            await interaction.response.send_message(
                "⚠️ give me text or an attachment", ephemeral=True
            )
            return
        await interaction.response.defer(thinking=True)
        file_path = None
        if attachment:
            file_path = await self._upload(interaction, attachment)
            if file_path is None:
                return
        result = await core.summarize_core(
            self.bot, text=text, file=file_path, mode=mode, preset=preset, model=model)
        await self._deliver(interaction, result, model=model)

    # ── /keywords ──────────────────────────────────────────────────────────
    @app_commands.command(name="keywords", description="Extract keywords (KeyBERT + spaCy)")
    @app_commands.describe(
        text="Text to extract keywords from",
        preset="Keyword preset (default: seo)",
        top_n="How many keywords to consider (default: preset's)",
        diversity="Result diversity, 0-1 (default: preset's)",
        attachment="Or attach a document",
        model="Embedding model override (default: central's)",
    )
    @app_commands.choices(
        preset=[app_commands.Choice(name=p, value=p) for p in core.KEYWORD_PRESETS]
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def keywords(
        self,
        interaction: discord.Interaction,
        text: str | None = None,
        preset: str | None = None,
        top_n: app_commands.Range[int, 1, 100] | None = None,
        diversity: app_commands.Range[float, 0.0, 1.0] | None = None,
        attachment: discord.Attachment | None = None,
        model: str | None = None,
    ) -> None:
        if not text and not attachment:
            await interaction.response.send_message(
                "⚠️ give me text or an attachment", ephemeral=True
            )
            return
        await interaction.response.defer(thinking=True)
        file_path = None
        if attachment:
            file_path = await self._upload(interaction, attachment)
            if file_path is None:
                return
        result = await core.keywords_core(
            self.bot, text=text, file=file_path, preset=preset, top_n=top_n,
            diversity=diversity, model=model)
        await self._deliver(interaction, result, model=model)

    # ── /transcribe ────────────────────────────────────────────────────────
    @app_commands.command(name="transcribe", description="Transcribe attached audio/video (whisper)")
    @app_commands.describe(
        attachment="Audio or video file",
        language="Source language hint (default: english)",
        size="Whisper model size (default: central decides)",
        translate="Translate to English instead of transcribing",
        timestamps="Include per-segment timestamps",
        model="Model override (default: central's whisper)",
    )
    @app_commands.choices(
        size=[app_commands.Choice(name=s, value=s) for s in core.WHISPER_SIZES]
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def transcribe(
        self,
        interaction: discord.Interaction,
        attachment: discord.Attachment,
        language: str | None = None,
        size: str | None = None,
        translate: bool = False,
        timestamps: bool = False,
        model: str | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        file_path = await self._upload(interaction, attachment)
        if file_path is None:
            return
        result = await core.transcribe_core(
            self.bot, file=file_path, language=language, size=size,
            translate=translate, timestamps=timestamps, model=model)
        await self._deliver(interaction, result, model=model)

    # ── /describe ──────────────────────────────────────────────────────────
    @app_commands.command(name="describe", description="Analyze an attached image (vision model)")
    @app_commands.describe(
        attachment="Image to analyze",
        prompt="What to ask about it (default: describe in detail)",
        max_tokens="Cap the response length in tokens",
        model="Model override (default: central's vision model)",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def describe(
        self,
        interaction: discord.Interaction,
        attachment: discord.Attachment,
        prompt: str | None = None,
        max_tokens: app_commands.Range[int, 1, 32768] | None = None,
        model: str | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        file_path = await self._upload(interaction, attachment)
        if file_path is None:
            return
        result = await core.describe_core(
            self.bot, file=file_path, prompt=prompt, max_tokens=max_tokens, model=model)
        await self._deliver(interaction, result, model=model)


async def setup(bot):
    await bot.add_cog(ToolsCog(bot))
