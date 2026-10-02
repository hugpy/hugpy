"""Operational view of hugpy central: status, model registry, download
jobs, GPU workers, and Hugging Face hub search.

The behavior lives in ``hugpy_discord.core.commands``; this cog is the Discord
face (interactions, embeds, the live-edited download progress message)."""
from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from hugpy_discord.core import commands as core
from hugpy_discord.cogs.helpers import model_autocomplete, render_result


class OpsCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="status", description="hugpy central health, serving models, workers")
    async def status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        result = await core.status_core(self.bot)
        await render_result(interaction, result)

    @app_commands.command(name="models", description="List models in the hugpy registry")
    @app_commands.describe(installed_only="Only show installed models")
    async def models(self, interaction: discord.Interaction, installed_only: bool = False) -> None:
        await interaction.response.defer(thinking=True)
        result = await core.models_core(self.bot, installed_only=installed_only)
        await render_result(interaction, result)

    @app_commands.command(name="download", description="Download a model (registry key or HF hub id)")
    @app_commands.describe(model="Registry model key, or a hub id like org/name")
    @app_commands.autocomplete(model=model_autocomplete)
    async def download(self, interaction: discord.Interaction, model: str) -> None:
        await interaction.response.defer(thinking=True)
        message = None
        async for line in core.download_core(self.bot, model=model):
            if message is None:
                message = await interaction.followup.send(line, wait=True)
            else:
                await message.edit(content=line)

    @app_commands.command(name="jobs", description="List download jobs")
    async def jobs(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        result = await core.jobs_core(self.bot)
        await render_result(interaction, result)

    @app_commands.command(name="canceljob", description="Cancel a download job")
    async def canceljob(self, interaction: discord.Interaction, job_id: str) -> None:
        await interaction.response.defer(ephemeral=True)
        result = await core.canceljob_core(self.bot, job_id=job_id)
        await render_result(interaction, result)

    @app_commands.command(name="hf", description="Search the Hugging Face hub")
    @app_commands.describe(query="Search terms", task="Pipeline tag filter (e.g. text-generation)")
    async def hf(self, interaction: discord.Interaction, query: str, task: str | None = None) -> None:
        await interaction.response.defer(thinking=True)
        result = await core.hf_core(self.bot, query=query, task=task)
        await render_result(interaction, result)


async def setup(bot):
    await bot.add_cog(OpsCog(bot))
