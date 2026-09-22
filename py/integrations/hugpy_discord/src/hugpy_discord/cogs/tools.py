"""Task commands over hugpy's dispatch categories: summarize, keywords,
transcribe, describe.

All four call central's POST /prompt (the execute_prompt passthrough) with
their dispatch task key, so every task-specific knob (summary_mode, language,
translate, keyword presets, …) reaches the real runner. When central predates
/prompt they fall back to the original /chat/stream prompt-template path —
the command surface stays the same.
"""
from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from hugpy_discord.no_think import strip_think, with_no_think
from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.streamer import MessageStreamer
from hugpy_discord.cogs.helpers import forward_attachment, model_autocomplete, send_long

# Mirrors keybert_model's registered presets.
KEYWORD_PRESETS = ("default", "seo", "metadata", "social", "long_tail", "article")
SUMMARY_MODES = ("auto", "short", "medium", "long")
WHISPER_SIZES = ("tiny", "small", "medium", "large")

SUMMARIZE_PROMPT = (
    "Summarize the following content concisely. Lead with a 1-2 sentence "
    "overview, then key points as a short bullet list.\n\n{content}"
)
KEYWORDS_PROMPT = (
    "Extract keywords from the following content for the '{preset}' use case. "
    "Return: primary keywords, secondary keywords, hashtags, and a url slug.\n\n{content}"
)
TRANSCRIBE_PROMPT = "Transcribe this audio/video file verbatim."
DESCRIBE_PROMPT = "Describe this file in detail."


def _no_prompt_route(exc: HugpyError) -> bool:
    return "does not expose /prompt" in str(exc)


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

    async def _stream_fallback(
        self,
        interaction: discord.Interaction,
        prompt: str,
        *,
        model: str | None = None,
        file_path: str | None = None,
    ) -> None:
        """Original /chat/stream path — used by /keywords and as the
        compatibility fallback when central has no /prompt route yet."""
        streamer = MessageStreamer(
            lambda content: interaction.followup.send(content, wait=True)
        )
        try:
            async for chunk in self.bot.hugpy.chat_stream(
                prompt=prompt,
                model_key=model or self.bot.model_for(interaction.user.id),
                file=file_path,
            ):
                await streamer.feed(chunk)
            await streamer.finish()
        except HugpyError as exc:
            await streamer.fail(str(exc))

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
        mode=[app_commands.Choice(name=m, value=m) for m in SUMMARY_MODES],
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
        try:
            result = await self.bot.hugpy.execute_prompt(
                task="text-summarization",
                text=text,
                file=file_path,
                summary_mode=mode,
                preset=preset,
                model_key=model,
            )
        except HugpyError as exc:
            if _no_prompt_route(exc):
                prompt = SUMMARIZE_PROMPT.format(content=text or "the attached file")
                await self._stream_fallback(interaction, prompt, model=model, file_path=file_path)
                return
            await interaction.followup.send(f"⚠️ {exc}")
            return

        summary = result.get("text") or ""
        stats = []
        if result.get("input_word_count"):
            stats.append(f"{result['input_word_count']}→{result.get('output_word_count', '?')} words")
        if result.get("preset_used"):
            stats.append(f"preset: {result['preset_used']}")
        if result.get("input_warning"):
            stats.append(f"⚠️ {result['input_warning']}")
        if stats:
            summary += f"\n-# {' · '.join(stats)}"
        await send_long(interaction.followup.send, summary, filename="summary.txt")

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
        preset=[app_commands.Choice(name=p, value=p) for p in KEYWORD_PRESETS]
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
        try:
            result = await self.bot.hugpy.execute_prompt(
                task="keyword-extraction",
                text=text,
                file=file_path,
                preset=preset,
                top_n=top_n,
                diversity=diversity,
                model_key=model,
            )
        except HugpyError as exc:
            if _no_prompt_route(exc):
                prompt = KEYWORDS_PROMPT.format(
                    preset=preset or "seo", content=text or "the attached file"
                )
                await self._stream_fallback(interaction, prompt, model=model, file_path=file_path)
                return
            await interaction.followup.send(f"⚠️ {exc}")
            return

        lines = []
        if result.get("primary"):
            lines.append("**primary:** " + ", ".join(f"`{kw}`" for kw in result["primary"]))
        if result.get("secondary"):
            lines.append("**secondary:** " + ", ".join(f"`{kw}`" for kw in result["secondary"]))
        if result.get("hashtags"):
            lines.append("**hashtags:** " + " ".join(result["hashtags"]))
        if result.get("slug_candidates"):
            lines.append("**slugs:** " + ", ".join(f"`{slug}`" for slug in result["slug_candidates"]))
        if not lines:
            lines.append(result.get("text") or "*no keywords extracted*")
        footer = []
        if result.get("preset_used"):
            footer.append(f"preset: {result['preset_used']}")
        if result.get("backends_used"):
            footer.append(f"backends: {'+'.join(result['backends_used'])}")
        if footer:
            lines.append(f"-# {' · '.join(footer)}")
        await send_long(interaction.followup.send, "\n".join(lines), filename="keywords.txt")

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
        size=[app_commands.Choice(name=s, value=s) for s in WHISPER_SIZES]
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
        try:
            result = await self.bot.hugpy.execute_prompt(
                task="automatic-speech-recognition",
                file=file_path,
                language=language,
                model_size=size,
                translate=translate or None,
                model_key=model,
            )
        except HugpyError as exc:
            if _no_prompt_route(exc):
                await self._stream_fallback(
                    interaction, TRANSCRIBE_PROMPT, model=model, file_path=file_path
                )
                return
            await interaction.followup.send(f"⚠️ {exc}")
            return

        if timestamps and result.get("segments"):
            lines = [
                f"[{seg.get('start', 0):>7.2f} → {seg.get('end', 0):>7.2f}] {seg.get('text', '').strip()}"
                for seg in result["segments"]
            ]
            text = "\n".join(lines)
        else:
            text = result.get("text") or ""
        footer = []
        if result.get("language"):
            footer.append(f"language: {result['language']}")
        if result.get("duration"):
            footer.append(f"duration: {result['duration']:.0f}s")
        if footer:
            text += f"\n-# {' · '.join(footer)}"
        await send_long(interaction.followup.send, text, filename="transcript.txt")

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
        try:
            # NO-THINK (hugpy_discord.no_think). The reply is posted to Discord as the
            # description — display-as-prose, nobody watching tokens arrive. The
            # prompt is only rewritten when the caller gave one; leaving it None
            # keeps central's own default, and the strip below covers that case.
            result = await self.bot.hugpy.execute_prompt(
                task="image-text-to-text",
                file=file_path,
                prompt=with_no_think(prompt) if prompt else prompt,
                max_new_tokens=max_tokens,
                model_key=model,
            )
        except HugpyError as exc:
            if _no_prompt_route(exc):
                await self._stream_fallback(
                    interaction, prompt or DESCRIBE_PROMPT, model=model, file_path=file_path
                )
                return
            await interaction.followup.send(f"⚠️ {exc}")
            return
        description, _reasoning = strip_think(result.get("text") or "")
        await send_long(interaction.followup.send, description,
                        filename="description.txt")


async def setup(bot):
    await bot.add_cog(ToolsCog(bot))
