"""Direct surface for the rest of hugpy's dispatch categories.

/embed, /similarity and /imagine are the dedicated commands; /task is the
fully-explicit escape hatch that mirrors execute_prompt one-to-one; /tasks shows
what central serves. The behavior lives in ``hugpy_discord.core.commands``; this
cog is the Discord face (interactions, attachments, embeds/files)."""
from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.core import commands as core
from hugpy_discord.cogs.helpers import (
    forward_attachment, model_autocomplete, render_result, split_items,
)

# Re-exported for compatibility with anything importing the cog's registry.
TASKS = core.TASKS


class MLCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    async def _upload(self, interaction, attachment) -> str | None:
        try:
            return await forward_attachment(self.bot, attachment)
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}", ephemeral=True)
            return None

    # ── /embed ─────────────────────────────────────────────────────────────
    @app_commands.command(name="embed", description="Embed text into vectors (feature-extraction)")
    @app_commands.describe(
        text="Text to embed — separate multiple texts with '||' or newlines",
        normalize="L2-normalize the vectors (central default: on)",
        batch_size="Encoder batch size (central default: 32)",
        model="Model override (default: central's embedder)",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def embed(
        self,
        interaction: discord.Interaction,
        text: str,
        normalize: bool | None = None,
        batch_size: app_commands.Range[int, 1, 256] | None = None,
        model: str | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        result = await core.embed_core(
            self.bot, texts=split_items(text), normalize=normalize,
            batch_size=batch_size, model=model)
        await render_result(interaction, result)

    # ── /similarity ────────────────────────────────────────────────────────
    @app_commands.command(name="similarity", description="Rank candidates by semantic similarity")
    @app_commands.describe(
        text="Query text",
        compare_to="Candidates — separate with '||' or newlines",
        normalize="L2-normalize before comparing (central default: on)",
        model="Model override (default: central's embedder)",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def similarity(
        self,
        interaction: discord.Interaction,
        text: str,
        compare_to: str,
        normalize: bool | None = None,
        model: str | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        result = await core.similarity_core(
            self.bot, text=text, candidates=split_items(compare_to),
            normalize=normalize, model=model)
        await render_result(interaction, result)

    # ── /imagine ───────────────────────────────────────────────────────────
    @app_commands.command(name="imagine", description="Generate an image from text")
    @app_commands.describe(
        prompt="What to generate",
        negative="What to avoid in the image",
        width="Image width in px (multiple of 8; default: model's native)",
        height="Image height in px (multiple of 8; default: model's native)",
        steps="Inference steps (default: model's native)",
        guidance="Guidance scale, 0-50 (default: model's native)",
        seed="Seed for reproducible output",
        count="How many images, 1-4 (default: 1)",
        model="Model override (default: central's image model)",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def imagine(
        self,
        interaction: discord.Interaction,
        prompt: str,
        negative: str | None = None,
        width: app_commands.Range[int, 64, 4096] | None = None,
        height: app_commands.Range[int, 64, 4096] | None = None,
        steps: app_commands.Range[int, 1, 200] | None = None,
        guidance: app_commands.Range[float, 0.0, 50.0] | None = None,
        seed: int | None = None,
        count: app_commands.Range[int, 1, 4] = 1,
        model: str | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        result = await core.imagine_core(
            self.bot, prompt=prompt, negative=negative, width=width, height=height,
            steps=steps, guidance=guidance, seed=seed, count=count, model=model)
        await render_result(interaction, result)

    # ── /task — fully explicit execute_prompt mirror ──────────────────────
    @app_commands.command(name="task", description="Run any hugpy task with explicit parameters")
    @app_commands.describe(
        task="Dispatch task key",
        input="Primary input (prompt/text; '||'-separated for embed tasks)",
        attachment="File input (image/audio/document)",
        model="Model override",
        params='Extra execute_prompt kwargs as JSON, e.g. {"summary_mode": "short", "seed": 7}',
        temperature="Sampling temperature",
        top_p="Nucleus sampling",
        max_tokens="Token cap",
        do_sample="Enable stochastic sampling",
    )
    @app_commands.choices(
        task=[app_commands.Choice(name=t, value=t) for t in core.TASKS]
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def task(
        self,
        interaction: discord.Interaction,
        task: str,
        input: str | None = None,
        attachment: discord.Attachment | None = None,
        model: str | None = None,
        params: str | None = None,
        temperature: app_commands.Range[float, 0.0, 2.0] | None = None,
        top_p: app_commands.Range[float, 0.0, 1.0] | None = None,
        max_tokens: app_commands.Range[int, 1, 32768] | None = None,
        do_sample: bool | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        file_path = None
        if attachment:
            file_path = await self._upload(interaction, attachment)
            if file_path is None:
                return
        result = await core.task_core(
            self.bot, task=task, input=input, file=file_path, model=model,
            params=params, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, do_sample=do_sample)
        await render_result(interaction, result)

    # ── /tasks ─────────────────────────────────────────────────────────────
    @app_commands.command(name="tasks", description="List hugpy task categories and their default models")
    async def tasks(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await core.tasks_core(self.bot)
        await render_result(interaction, result)


async def setup(bot):
    await bot.add_cog(MLCog(bot))
