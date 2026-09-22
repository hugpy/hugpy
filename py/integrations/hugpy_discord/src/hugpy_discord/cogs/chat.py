"""Conversational interface: mention/DM chat plus /chat, /model, /reset,
/running, /stop."""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
import uuid
from collections import defaultdict, deque

import discord
from discord import app_commands
from discord.ext import commands

from hugpy_discord.config import HISTORY_MAX_TURNS
from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.streamer import MessageStreamer
from hugpy_discord.cogs.helpers import (
    forward_attachment,
    model_autocomplete,
    model_label,
    clean_model_key,
)

log = logging.getLogger(__name__)


class ChatCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=HISTORY_MAX_TURNS * 2)
        )
        # In-flight generations: turn id -> info dict (task, channel, prompt…).
        # Cancelling the task tears down the SSE stream, which makes central
        # drop the worker-side generation.
        self._active: dict[str, dict] = {}
        self._turn_ids = itertools.count(1)

    # ── core streaming turn ────────────────────────────────────────────────
    async def run_turn(
        self,
        send,
        *,
        channel_id: int,
        user_id: int,
        prompt: str,
        model_key: str | None = None,
        file: str | None = None,
        remember: bool = True,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
    ) -> None:
        # A console-managed Discord binding for this channel/user wins; otherwise
        # the user's saved pref / configured default (resolved inside the bot).
        model_key = clean_model_key(model_key) or await self.bot.resolve_model_for(user_id, channel_id)
        history = list(self._history[channel_id]) if remember else []
        messages = history + [{"role": "user", "content": prompt}]

        # DISC-06: a channel personality bundles system prompt + params.
        # Its system prompt heads the message list every turn; its params are
        # DEFAULTS only — explicit per-turn values keep winning. (Its model
        # was already applied in resolve_model_for; explicit model wins.)
        persona = await self.bot.channel_personality(channel_id)
        if persona:
            if persona.get("system"):
                messages = ([{"role": "system", "content": persona["system"]}]
                            + messages)
            params = persona.get("params") or {}
            if temperature is None:
                temperature = params.get("temperature")
            if top_p is None:
                top_p = params.get("top_p")
            if do_sample is None:
                do_sample = params.get("do_sample")
            if max_new_tokens is None:
                max_new_tokens = params.get("max_new_tokens")

        streamer = MessageStreamer(send)
        turn_id = f"t{next(self._turn_ids)}"
        # The request_id central's job store tracks this turn under — /stop
        # cancels through it, so the generation actually stops server-side
        # instead of us just dropping our end of the SSE pipe.
        request_id = uuid.uuid4().hex
        self._active[turn_id] = {
            "task": asyncio.current_task(),
            "request_id": request_id,
            "channel_id": channel_id,
            "user_id": user_id,
            "model": model_key,
            "prompt": prompt,
            "started": time.monotonic(),
        }
        try:
            async for chunk in self.bot.hugpy.chat_stream(
                messages=messages, model_key=model_key, file=file,
                temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
                request_id=request_id,
                transport="discord", channel=str(channel_id),
            ):
                await streamer.feed(chunk)
            await streamer.finish()
        except asyncio.CancelledError:
            log.info("chat turn %s stopped via /stop", turn_id)
            try:
                await streamer.fail("stopped via /stop")
            except Exception:
                pass
            return
        except HugpyError as exc:
            log.warning("chat turn failed: %s", exc)
            await streamer.fail(str(exc))
            return
        except discord.HTTPException:
            log.exception("discord edit failed mid-stream")
            return
        finally:
            self._active.pop(turn_id, None)

        if remember and streamer.full_text:
            self._history[channel_id].append({"role": "user", "content": prompt})
            self._history[channel_id].append(
                {"role": "assistant", "content": streamer.full_text}
            )

    # ── mention / DM chat ─────────────────────────────────────────────────
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or self.bot.user is None:
            return
        # Bridged channel: relay every message up to the console and let the
        # bridge (console/directive) own the response — don't auto-reply here.
        if self.bot.is_bridged(message.channel.id):
            await self.bot.relay_inbound(message)
            return
        is_dm = message.guild is None
        mentioned = self.bot.user in message.mentions
        if not (is_dm or mentioned):
            # DISC-04: a channel flipped to respond-to-all (console-set, F4
            # settings) answers every message, not just mentions. Default
            # stays mention-only — respond-to-all in a busy channel is a
            # queue flood, which is why CON-01's live view exists.
            mode = (await self.bot.channel_settings(message.channel.id)
                    ).get("respond")
            if mode != "all":
                return

        prompt = message.content
        for mention in (f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"):
            prompt = prompt.replace(mention, "")
        prompt = prompt.strip()

        file_path = None
        if message.attachments:
            try:
                file_path = await forward_attachment(self.bot, message.attachments[0])
            except HugpyError as exc:
                await message.reply(f"⚠️ couldn't forward attachment: {exc}")
                return
            if not prompt:
                prompt = "Describe this file."
        if not prompt:
            return

        async with message.channel.typing():
            await self.run_turn(
                message.reply,
                channel_id=message.channel.id,
                user_id=message.author.id,
                prompt=prompt,
                file=file_path,
            )

    # ── slash commands ────────────────────────────────────────────────────
    @app_commands.command(name="chat", description="Chat with the hugpy model")
    @app_commands.describe(
        prompt="What to say",
        model="Model to use for this turn (defaults to your /model choice)",
        attachment="Optional file (image/audio/document) to include",
        private="Only you see the reply",
        temperature="Sampling temperature, 0-2 (central default: 0.1)",
        top_p="Nucleus sampling, 0-1 (central default: 1.0)",
        max_tokens="Cap the response length in tokens (default: unbounded)",
        do_sample="Enable stochastic sampling (central default: off)",
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def chat(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str | None = None,
        attachment: discord.Attachment | None = None,
        private: bool = False,
        temperature: app_commands.Range[float, 0.0, 2.0] | None = None,
        top_p: app_commands.Range[float, 0.0, 1.0] | None = None,
        max_tokens: app_commands.Range[int, 1, 32768] | None = None,
        do_sample: bool | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=private, thinking=True)
        file_path = None
        if attachment:
            try:
                file_path = await forward_attachment(self.bot, attachment)
            except HugpyError as exc:
                await interaction.followup.send(f"⚠️ {exc}", ephemeral=True)
                return
        await self.run_turn(
            lambda content: interaction.followup.send(content, ephemeral=private, wait=True),
            channel_id=interaction.channel_id or interaction.user.id,
            user_id=interaction.user.id,
            prompt=prompt,
            model_key=model,
            file=file_path,
            remember=not private,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
        )

    @app_commands.command(name="reset", description="Forget this channel's conversation history")
    async def reset(self, interaction: discord.Interaction) -> None:
        self._history.pop(interaction.channel_id or interaction.user.id, None)
        await interaction.response.send_message("🧹 history cleared", ephemeral=True)

    @app_commands.command(name="running", description="List in-progress generations")
    async def running(self, interaction: discord.Interaction) -> None:
        if not self._active:
            await interaction.response.send_message(
                "nothing is generating right now", ephemeral=True
            )
            return
        now = time.monotonic()
        lines = [
            f"`{turn_id}` — <#{info['channel_id']}> — `{info['model'] or 'default'}`"
            f" — {int(now - info['started'])}s — “{info['prompt'][:60]}”"
            for turn_id, info in self._active.items()
        ]
        await interaction.response.send_message(
            embed=discord.Embed(
                title=f"running generations ({len(lines)})",
                description="\n".join(lines)[:4000],
                colour=discord.Colour.blurple(),
            ).set_footer(text="stop one with /stop <id>, or /stop in its channel"),
            ephemeral=True,
        )

    @app_commands.command(name="stop", description="Stop in-progress generation(s)")
    @app_commands.describe(
        turn_id="A specific generation from /running (default: everything in this channel)"
    )
    async def stop(self, interaction: discord.Interaction, turn_id: str | None = None) -> None:
        if turn_id:
            info = self._active.get(turn_id)
            if not info:
                await interaction.response.send_message(
                    f"no running generation `{turn_id}` — see /running", ephemeral=True
                )
                return
            targets = {turn_id: info}
        else:
            channel_id = interaction.channel_id or interaction.user.id
            targets = {
                tid: info for tid, info in self._active.items()
                if info["channel_id"] == channel_id
            }
            if not targets:
                await interaction.response.send_message(
                    "nothing is generating in this channel — see /running", ephemeral=True
                )
                return
        for info in targets.values():
            # Server-side first: central's control plane stops the generation
            # and frees the slot (locally-served or worker-relayed). Then the
            # task cancel tears down our SSE read. Best-effort — an old
            # central without the route still gets the task cancel.
            rid = info.get("request_id")
            if rid:
                try:
                    await self.bot.hugpy.cancel_chat(rid)
                except Exception as exc:
                    log.debug("central-side cancel of %s failed: %s", rid, exc)
            info["task"].cancel()
        stopped = ", ".join(f"`{tid}`" for tid in targets)
        await interaction.response.send_message(f"⏹️ stopped {stopped}", ephemeral=True)

    @app_commands.command(
        name="link",
        description="Link your Discord account to a hugpy principal token")
    @app_commands.describe(token="The principal token an operator issued you (hpp_…)")
    async def link(self, interaction: discord.Interaction, token: str) -> None:
        """DISC-05: proves possession of a principal token and binds this
        Discord account to that principal. Ephemeral — the token never
        appears in the channel."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            out = await self.bot.hugpy.discord_link(token.strip(), interaction.user.id)
        except HugpyError as exc:
            await interaction.followup.send(f"⚠️ link failed: {exc}", ephemeral=True)
            return
        p = out.get("principal") or {}
        await interaction.followup.send(
            f"🔗 linked to principal `{p.get('id')}`"
            f" ({p.get('name') or 'unnamed'}, groups: "
            f"{', '.join(p.get('groups') or []) or 'none'})",
            ephemeral=True,
        )

    model_group = app_commands.Group(name="model", description="Your default hugpy model")

    @model_group.command(name="set", description="Set your default model")
    @app_commands.autocomplete(model=model_autocomplete)
    async def model_set(self, interaction: discord.Interaction, model: str) -> None:
        # Write-through: central settings (console-visible) + local fallback.
        await self.bot.set_user_model(interaction.user.id, model)
        await interaction.response.send_message(f"✅ default model: `{model}`", ephemeral=True)

    @model_group.command(name="show", description="Show your current default model")
    async def model_show(self, interaction: discord.Interaction) -> None:
        model = self.bot.model_for(interaction.user.id)
        text = f"current model: `{model}`" if model else "no default model set (central decides)"
        await interaction.response.send_message(text, ephemeral=True)

    @model_group.command(name="clear", description="Clear your default model")
    async def model_clear(self, interaction: discord.Interaction) -> None:
        await self.bot.set_user_model(interaction.user.id, None)
        await interaction.response.send_message("✅ default model cleared", ephemeral=True)


async def setup(bot):
    await bot.add_cog(ChatCog(bot))
