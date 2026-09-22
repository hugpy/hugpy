"""Direct surface for the rest of hugpy's dispatch categories.

/embed, /similarity and /imagine are the dedicated commands; /task is the
fully-explicit escape hatch that mirrors execute_prompt one-to-one — pick any
task key, set any builder kwarg via params JSON, and whatever you leave out
falls to central's default-resolution chain. /tasks shows what central serves.
"""
from __future__ import annotations

import base64
import io
import json

import discord
from discord import app_commands
from discord.ext import commands

from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.cogs.helpers import (
    model_autocomplete,
    forward_attachment,
    send_long,
    split_items,
)

# Static mirror of hugpy's KNOWN_TASKS_REGISTRY — used for slash-command
# choices (those must be static) and as the /tasks fallback when central
# predates GET /prompt/tasks.
TASKS = (
    "text-generation",
    "image-text-to-text",
    "automatic-speech-recognition",
    "text-summarization",
    "text2text-generation",
    "feature-extraction",
    "sentence-similarity",
    "text-to-image",
    "keyword-extraction",
)

# How the generic /task input string maps onto each task's primary field.
_INPUT_FIELD = {
    "text-generation": "prompt",
    "image-text-to-text": "prompt",
    "automatic-speech-recognition": None,   # input is the attachment
    "text-summarization": "text",
    "text2text-generation": "text",
    "feature-extraction": "texts",
    "sentence-similarity": "texts",
    "text-to-image": "prompt",
    "keyword-extraction": "text",
}


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
        texts = split_items(text)
        try:
            result = await self.bot.hugpy.execute_prompt(
                task="feature-extraction",
                texts=texts,
                normalize=normalize,
                batch_size=batch_size,
                model_key=model,
            )
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return
        vectors = result.get("embeddings") or []
        dim = len(vectors[0]) if vectors else 0
        payload = json.dumps(
            {"model": result.get("model_key"), "texts": texts, "embeddings": vectors},
            indent=2,
        )
        await interaction.followup.send(
            f"🧮 embedded **{len(vectors)}** text(s) → **{dim}**-dim vectors "
            f"(`{result.get('model_key')}`) — vectors attached",
            file=discord.File(io.BytesIO(payload.encode("utf-8")), filename="embeddings.json"),
        )

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
        candidates = split_items(compare_to)
        try:
            result = await self.bot.hugpy.execute_prompt(
                task="sentence-similarity",
                texts=[text],
                other_texts=candidates,
                normalize=normalize,
                model_key=model,
            )
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return
        scores = (result.get("similarities") or [[]])[0]
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        lines = [f"**similarity to:** {text}"]
        lines += [f"`{score:+.4f}` {candidate}" for candidate, score in ranked]
        await send_long(interaction.followup.send, "\n".join(lines), filename="similarity.txt")

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
        try:
            result = await self.bot.hugpy.execute_prompt(
                task="text-to-image",
                prompt=prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance,
                seed=seed,
                num_images=count if count != 1 else None,
                model_key=model,
            )
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return

        files = []
        for index, image in enumerate(result.get("images") or []):
            if image.get("b64"):
                files.append(discord.File(
                    io.BytesIO(base64.b64decode(image["b64"])),
                    filename=f"imagine_{index}.png",
                ))
        if not files:
            await interaction.followup.send(
                f"⚠️ no image bytes returned: {result.get('text') or result.get('error') or '?'}"
            )
            return
        caption = f"🎨 **{prompt}**" + (f"\n-# seed: {seed}" if seed is not None else "")
        await interaction.followup.send(caption[:1900], files=files)

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
        task=[app_commands.Choice(name=t, value=t) for t in TASKS]
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
        extra: dict = {}
        if params:
            try:
                extra = json.loads(params)
                if not isinstance(extra, dict):
                    raise ValueError("params must be a JSON object")
            except ValueError as exc:
                await interaction.response.send_message(f"⚠️ bad params JSON: {exc}", ephemeral=True)
                return

        await interaction.response.defer(thinking=True)
        kwargs: dict = {
            "task": task,
            "model_key": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }
        if attachment:
            file_path = await self._upload(interaction, attachment)
            if file_path is None:
                return
            kwargs["file"] = file_path
        if input:
            field = _INPUT_FIELD.get(task)
            if field == "texts":
                kwargs["texts"] = split_items(input)
            elif field:
                kwargs[field] = input
        kwargs.update(extra)  # explicit params JSON wins over the mapping

        try:
            result = await self.bot.hugpy.execute_prompt(**kwargs)
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ {exc}")
            return
        await self._render_result(interaction, task, result)

    async def _render_result(self, interaction, task: str, result: dict) -> None:
        """Shape-aware rendering: images attach, vectors summarize, text sends."""
        files = []
        for index, image in enumerate(result.get("images") or []):
            if isinstance(image, dict) and image.get("b64"):
                files.append(discord.File(
                    io.BytesIO(base64.b64decode(image["b64"])),
                    filename=f"{task}_{index}.png",
                ))
        if files:
            await interaction.followup.send((result.get("text") or "🎨 done")[:1900], files=files)
            return

        if result.get("embeddings") is not None and not result.get("similarities"):
            vectors = result["embeddings"]
            dim = len(vectors[0]) if vectors else 0
            payload = json.dumps(result, indent=2)
            await interaction.followup.send(
                f"🧮 {len(vectors)} vector(s) × {dim} dims — attached",
                file=discord.File(io.BytesIO(payload.encode("utf-8")), filename="result.json"),
            )
            return

        if result.get("similarities") is not None:
            rows = result["similarities"]
            lines = [
                f"row {row_index}: " + ", ".join(f"{score:+.4f}" for score in row)
                for row_index, row in enumerate(rows)
            ]
            await send_long(interaction.followup.send, "\n".join(lines), filename="similarities.txt")
            return

        await send_long(
            interaction.followup.send,
            result.get("text") or json.dumps(result, indent=2)[:4000],
            filename=f"{task}.txt",
        )

    # ── /tasks ─────────────────────────────────────────────────────────────
    @app_commands.command(name="tasks", description="List hugpy task categories and their default models")
    async def tasks(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            info = await self.bot.hugpy.supported_tasks()
            tasks, defaults = info.get("tasks") or [], info.get("defaults") or {}
            note = ""
        except HugpyError:
            tasks, defaults = list(TASKS), {}
            note = "\n-# central predates /prompt/tasks — showing the bot's static list"
        lines = ["**hugpy task categories**"]
        for key in tasks:
            default = defaults.get(key)
            lines.append(f"• `{key}`" + (f" → `{default}`" if default else ""))
        await interaction.followup.send("\n".join(lines) + note, ephemeral=True)


async def setup(bot):
    await bot.add_cog(MLCog(bot))
