"""Operational view of hugpy central: status, model registry, download
jobs, GPU workers, and Hugging Face hub search.
"""
from __future__ import annotations

import asyncio

import discord
from discord import app_commands
from discord.ext import commands

from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.cogs.helpers import cached_models, model_autocomplete, model_label

JOB_POLL_SECONDS = 5
JOB_POLL_MAX_MINUTES = 30
TERMINAL_JOB_STATES = {"done", "complete", "completed", "error", "failed", "cancelled", "canceled"}


def _trim(text: str, limit: int = 1024) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class OpsCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="status", description="hugpy central health, serving models, workers")
    async def status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        embed = discord.Embed(title="hugpy central", colour=discord.Colour.blurple())
        embed.set_footer(text=self.bot.hugpy.base_url)
        try:
            health = await self.bot.hugpy.health()
            embed.add_field(
                name="health",
                value=f"✅ ok — storage `{health.get('storage_root', '?')}`",
                inline=False,
            )
        except HugpyError as exc:
            embed.colour = discord.Colour.red()
            embed.add_field(name="health", value=f"❌ {_trim(str(exc), 200)}", inline=False)
            await interaction.followup.send(embed=embed)
            return

        try:
            serving = await self.bot.hugpy.serving()
            keys = list(serving) if isinstance(serving, (list, dict)) else []
            embed.add_field(
                name=f"serving ({len(keys)})",
                value=_trim(", ".join(f"`{k}`" for k in keys) or "nothing"),
                inline=False,
            )
        except HugpyError:
            pass

        try:
            workers = await self.bot.hugpy.list_workers()
            lines = [
                f"`{w.get('name') or w.get('id')}` — {w.get('status', '?')}"
                + (f" ({', '.join(w.get('models') or [])})" if w.get("models") else "")
                for w in workers
            ]
            embed.add_field(
                name=f"workers ({len(workers)})",
                value=_trim("\n".join(lines) or "none registered"),
                inline=False,
            )
        except HugpyError:
            pass

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="models", description="List models in the hugpy registry")
    @app_commands.describe(installed_only="Only show installed models")
    async def models(self, interaction: discord.Interaction, installed_only: bool = False) -> None:
        await interaction.response.defer(thinking=True)
        try:
            models = await self.bot.hugpy.list_models()
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return
        if installed_only:
            models = [
                m for m in models
                if m.get("installed") or str(m.get("status", "")).lower() == "installed"
            ]
        if not models:
            await interaction.followup.send("no models found")
            return

        lines = [f"• {model_label(m)}" for m in models]
        embed = discord.Embed(
            title=f"models ({len(models)})",
            description=_trim("\n".join(lines), 4000),
            colour=discord.Colour.blurple(),
        )
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="download", description="Download a model (registry key or HF hub id)")
    @app_commands.describe(model="Registry model key, or a hub id like org/name")
    @app_commands.autocomplete(model=model_autocomplete)
    async def download(self, interaction: discord.Interaction, model: str) -> None:
        await interaction.response.defer(thinking=True)
        try:
            if "/" in model:
                job = await self.bot.hugpy.download_repo(model)
            else:
                job = await self.bot.hugpy.download_model(model)
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return

        job_id = job.get("id") or job.get("job_id")
        message = await interaction.followup.send(
            f"⬇️ downloading `{model}` — job `{job_id}`", wait=True
        )
        if not job_id:
            return
        # Poll the job and keep one message updated until it settles.
        for _ in range(JOB_POLL_MAX_MINUTES * 60 // JOB_POLL_SECONDS):
            await asyncio.sleep(JOB_POLL_SECONDS)
            try:
                job = await self.bot.hugpy.get_job(job_id)
            except HugpyError:
                continue
            status = str(job.get("status", "?")).lower()
            progress = job.get("progress") or job.get("percent")
            line = f"⬇️ `{model}` — {status}" + (f" ({progress}%)" if progress is not None else "")
            if status in TERMINAL_JOB_STATES:
                icon = "✅" if status in ("done", "complete", "completed") else "❌"
                await message.edit(content=f"{icon} `{model}` — {status}")
                return
            await message.edit(content=line)
        await message.edit(content=f"⏱️ `{model}` — still running; check /jobs (job `{job_id}`)")

    @app_commands.command(name="jobs", description="List download jobs")
    async def jobs(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        try:
            jobs = await self.bot.hugpy.list_jobs()
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return
        if not jobs:
            await interaction.followup.send(
                "no download jobs — chat generations are listed by /running"
            )
            return
        lines = [
            f"`{j.get('id') or j.get('job_id')}` — {j.get('model_key', '?')} — {j.get('status', '?')}"
            for j in jobs[-20:]
        ]
        await interaction.followup.send(embed=discord.Embed(
            title="download jobs", description="\n".join(lines),
            colour=discord.Colour.blurple(),
        ))

    @app_commands.command(name="canceljob", description="Cancel a download job")
    async def canceljob(self, interaction: discord.Interaction, job_id: str) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            await self.bot.hugpy.cancel_job(job_id)
            await interaction.followup.send(f"🛑 cancelled `{job_id}`", ephemeral=True)
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}", ephemeral=True)

    @app_commands.command(name="hf", description="Search the Hugging Face hub")
    @app_commands.describe(query="Search terms", task="Pipeline tag filter (e.g. text-generation)")
    async def hf(self, interaction: discord.Interaction, query: str, task: str | None = None) -> None:
        await interaction.response.defer(thinking=True)
        try:
            results = await self.bot.hugpy.hf_search(query, limit=10, task=task)
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return
        items = results if isinstance(results, list) else results.get("results") or results.get("models") or []
        if not items:
            await interaction.followup.send("no results")
            return
        lines = []
        for item in items[:10]:
            hub_id = item.get("hub_id") or item.get("id") or item.get("modelId") or "?"
            extras = []
            if item.get("downloads") is not None:
                extras.append(f"{item['downloads']:,} dl")
            if item.get("likes") is not None:
                extras.append(f"{item['likes']} ♥")
            if item.get("size_human") or item.get("size"):
                extras.append(str(item.get("size_human") or item.get("size")))
            lines.append(f"• `{hub_id}`" + (f" — {', '.join(extras)}" if extras else ""))
        embed = discord.Embed(
            title=f"hf search: {query}",
            description=_trim("\n".join(lines), 4000),
            colour=discord.Colour.blurple(),
        )
        embed.set_footer(text="install with /download <hub_id>")
        await interaction.followup.send(embed=embed)


async def setup(bot):
    await bot.add_cog(OpsCog(bot))
