"""Conversational interface: mention/DM chat plus /chat, /model, /reset,
/running, /stop, /link.

The streaming turn, per-channel history and the in-flight registry live in the
transport-neutral :class:`~hugpy_discord.core.chat.ChatEngine`; this cog is the
Discord face of it (interactions, embeds, typing, attachments)."""
from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from hugpy_discord.core.chat import ChatEngine
from hugpy_discord.config import MESSAGE_CHAR_LIMIT
from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.cogs.helpers import forward_attachment, model_autocomplete

log = logging.getLogger(__name__)


class ChatCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.engine = ChatEngine(
            bot, transport="discord", char_limit=MESSAGE_CHAR_LIMIT,
            transport_errors=(discord.HTTPException,))

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
            await self.engine.run_turn(
                message.reply,
                conv_key=message.channel.id,
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
        await self.engine.run_turn(
            lambda content: interaction.followup.send(content, ephemeral=private, wait=True),
            conv_key=interaction.channel_id or interaction.user.id,
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
        self.engine.reset(interaction.channel_id or interaction.user.id)
        await interaction.response.send_message("🧹 history cleared", ephemeral=True)

    @app_commands.command(name="running", description="List in-progress generations")
    async def running(self, interaction: discord.Interaction) -> None:
        snapshot = self.engine.active_snapshot()
        if not snapshot:
            await interaction.response.send_message(
                "nothing is generating right now", ephemeral=True
            )
            return
        lines = [
            f"`{s['turn_id']}` — <#{s['channel_id']}> — `{s['model'] or 'default'}`"
            f" — {s['elapsed']}s — “{s['prompt'][:60]}”"
            for s in snapshot
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
        conv = interaction.channel_id or interaction.user.id
        stopped, unknown = await self.engine.stop(conv_key=conv, turn_id=turn_id)
        if unknown:
            await interaction.response.send_message(
                f"no running generation `{turn_id}` — see /running", ephemeral=True
            )
            return
        if not stopped:
            await interaction.response.send_message(
                "nothing is generating in this channel — see /running", ephemeral=True
            )
            return
        stopped_str = ", ".join(f"`{tid}`" for tid in stopped)
        await interaction.response.send_message(f"⏹️ stopped {stopped_str}", ephemeral=True)

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
