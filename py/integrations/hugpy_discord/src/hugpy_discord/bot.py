"""HugpyBot — discord UI arm of a Hugpy central service (HTTP client only)."""
from __future__ import annotations

import argparse
import logging
import os
import sys

import discord
from discord.ext import commands, tasks

from hugpy_discord import config
from hugpy_discord.hugpy_client import HugpyClient
from hugpy_discord.prefs import PrefsStore

log = logging.getLogger(__name__)

COGS = (".cogs.chat", ".cogs.tools", ".cogs.ml", ".cogs.ops")


# ── keeper escalations: clickable operator choices ────────────────────────
# An escalation outbound may carry `options` (labels). The bot renders them as
# buttons (≤5) or a select menu (>5) whose custom_id encodes the routing:
#     esc:<target>:<msg>:<idx>   (button, idx = option index)
#     esc:<target>:<msg>:sel     (select, chosen value = option index)
# `target` is the outbound's channel_id (or user_id for a DM); `msg` is the
# outbox message id (uuid4 hex). Because the custom_id carries a *dynamic* msg
# id, we use discord.ui.DynamicItem so a click still resolves after a restart —
# the item type is re-registered in setup_hook and re-parses the id from the
# message. The chosen label is recovered from the message's own component
# payload (not the custom_id), which likewise survives a restart. The click is
# posted back INBOUND via central's /discord/inbox so it appears to
# `hugpy-escalate replies` exactly like a typed reply.

def _iter_leaf_components(message):
    """Yield each leaf component (Button/SelectMenu) of a message, flattening
    the action-row wrappers."""
    for row in (getattr(message, "components", None) or []):
        children = getattr(row, "children", None)
        if children is None:
            yield row
        else:
            for child in children:
                yield child


def _label_from_message(message, custom_id, chosen_value=None):
    """Recover the chosen label from the message's own component payload — this
    survives a bot restart (the label lives in the message, not the custom_id).
    Button: match custom_id, read `.label`. Select: find the option whose
    `.value == chosen_value` and read that option's `.label`."""
    for comp in _iter_leaf_components(message):
        if getattr(comp, "custom_id", None) != custom_id:
            continue
        options = getattr(comp, "options", None)
        if options is not None and chosen_value is not None:
            for opt in options:
                if str(getattr(opt, "value", "")) == str(chosen_value):
                    return getattr(opt, "label", None)
        label = getattr(comp, "label", None)
        if label:
            return label
    return None


def _all_components_disabled(message) -> bool:
    comps = list(_iter_leaf_components(message))
    if not comps:
        return False
    return all(getattr(c, "disabled", False) for c in comps)


def _disabled_view_from_message(message):
    """A View mirroring the message's components with every item disabled, so the
    ack edit leaves the choice visible but un-clickable. None on failure (the
    caller then edits with no view — the ✓ text still lands)."""
    try:
        view = discord.ui.View.from_message(message)
        for item in view.children:
            try:
                item.disabled = True
            except Exception:
                pass
        return view
    except Exception:
        return None


async def _record_escalation_choice(interaction, *, target, msg_id, custom_id,
                                     chosen_value=None, fallback_label=""):
    """Shared click handler for escalation buttons/selects. Operator-gated,
    idempotent, acks by disabling the controls, and relays the chosen label back
    inbound. Guarded so a malformed click never crashes the gateway."""
    bot = interaction.client
    try:
        # 1. Operator gate — fail closed.
        op_id = getattr(config, "OPERATOR_DISCORD_ID", None)
        if op_id is None:
            await interaction.response.send_message(
                "Escalation controls aren't configured (operator gate not set).",
                ephemeral=True)
            return
        if interaction.user.id != op_id:
            await interaction.response.send_message(
                "This escalation isn't yours to answer.", ephemeral=True)
            return

        # 2. Chosen label, recovered from the message payload (restart-safe).
        label = _label_from_message(interaction.message, custom_id,
                                    chosen_value=chosen_value) or fallback_label

        # 3. Idempotency (best-effort in-memory + the message's own disabled state).
        answered = getattr(bot, "_answered_escalations", None)
        if answered is None:
            answered = bot._answered_escalations = set()
        if msg_id in answered or _all_components_disabled(interaction.message):
            await interaction.response.send_message(
                "That escalation was already recorded.", ephemeral=True)
            return
        answered.add(msg_id)

        # 4. Ack: edit the message, disabling all controls.
        view = _disabled_view_from_message(interaction.message)
        base = interaction.message.content or ""
        await interaction.response.edit_message(
            content=f"{base}\n\n✓ you chose **{label}**", view=view)

        # 5. Relay the choice back INBOUND via the existing path (channel-based).
        #    For a DM/user target this is a harmless no-op on central
        #    (bridge_for_channel returns None -> bridged:false); escalations use a
        #    channel, so the channel path is the one that lands as direction="in"
        #    and trips `hugpy-escalate replies` + keeper-notify.
        try:
            author = getattr(interaction.user, "display_name", None) or str(interaction.user)
            await bot.hugpy.relay_inbound(channel_id=int(target), author=author,
                                          content=label)
        except Exception:
            log.warning("escalation inbound relay failed for msg %s", msg_id, exc_info=True)
    except Exception:
        log.warning("escalation click handling failed (custom_id=%s)", custom_id, exc_info=True)


class EscalationButton(discord.ui.DynamicItem[discord.ui.Button],
                       template=r"esc:(?P<target>\d+):(?P<msg>[0-9a-f]+):(?P<idx>\d+)"):
    def __init__(self, target: int, msg: str, idx: int, label: str = ""):
        self.target = target
        self.msg = msg
        self.idx = idx
        super().__init__(
            discord.ui.Button(
                label=(label or f"Option {idx + 1}")[:80],
                style=discord.ButtonStyle.primary,
                custom_id=f"esc:{target}:{msg}:{idx}",
            )
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(int(match["target"]), match["msg"], int(match["idx"]),
                   label=getattr(item, "label", "") or "")

    async def callback(self, interaction):
        await _record_escalation_choice(
            interaction, target=self.target, msg_id=self.msg,
            custom_id=self.item.custom_id,
            fallback_label=getattr(self.item, "label", "") or "")


class EscalationSelect(discord.ui.DynamicItem[discord.ui.Select],
                       template=r"esc:(?P<target>\d+):(?P<msg>[0-9a-f]+):sel"):
    def __init__(self, target: int, msg: str, options=None):
        self.target = target
        self.msg = msg
        super().__init__(
            discord.ui.Select(
                custom_id=f"esc:{target}:{msg}:sel",
                placeholder="Choose one…",
                min_values=1, max_values=1,
                options=options or [discord.SelectOption(label="…", value="0")],
            )
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        opts = list(getattr(item, "options", None) or [])
        return cls(int(match["target"]), match["msg"], options=opts or None)

    async def callback(self, interaction):
        try:
            chosen_value = interaction.data["values"][0]
        except (KeyError, IndexError, TypeError):
            chosen_value = None
        await _record_escalation_choice(
            interaction, target=self.target, msg_id=self.msg,
            custom_id=self.item.custom_id, chosen_value=chosen_value,
            fallback_label=str(chosen_value or ""))


def _build_escalation_view(options, target, msg_id) -> "discord.ui.View":
    """Render option labels as buttons (≤5) or a single select (>5, cap 25)."""
    view = discord.ui.View(timeout=None)
    labels = [str(o) for o in options]
    if len(labels) <= 5:
        for idx, label in enumerate(labels):
            view.add_item(EscalationButton(int(target), msg_id, idx, label=label))
    else:
        if len(labels) > 25:
            log.warning("escalation msg %s has %d options; truncating to 25 for "
                        "the select menu", msg_id, len(labels))
            labels = labels[:25]
        opts = [discord.SelectOption(label=lbl[:100], value=str(i))
                for i, lbl in enumerate(labels)]
        view.add_item(EscalationSelect(int(target), msg_id, options=opts))
    return view


class HugpyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        if config.MEMBERS_INTENT:
            intents.members = True   # privileged — also needs the portal toggle
        super().__init__(command_prefix="!", intents=intents)
        self.hugpy = HugpyClient(config.HUGPY_BASE_URL)
        self.prefs = PrefsStore()
        self._bridged: set[str] = set()   # channel ids wired to a console session
        # outbox message ids whose escalation choice we've already recorded
        # (best-effort in-memory idempotency; the message's own disabled state is
        # the durable guard across a restart).
        self._answered_escalations: set[str] = set()

    async def setup_hook(self) -> None:
        # Register the escalation click handlers so button/select clicks resolve
        # even after a restart (their custom_ids carry a dynamic outbox msg id).
        self.add_dynamic_items(EscalationButton, EscalationSelect)
        for cog in COGS:
            await self.load_extension(cog, package=__package__)
        if config.GUILD_ID:
            guild = discord.Object(id=config.GUILD_ID)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()
        # Deliver console-queued messages from a model into its Discord target.
        self._outbox_poller.start()
        # Report the channels the bot can see so the console can offer them as a
        # dropdown when binding a model to a channel.
        self._channel_reporter.start()
        # Members too, but only when the privileged intent is enabled.
        if config.MEMBERS_INTENT:
            self._member_reporter.start()

    async def on_ready(self) -> None:
        log.info("logged in as %s (%s); hugpy central: %s",
                 self.user, self.user.id, config.HUGPY_BASE_URL)

    async def close(self) -> None:
        self._outbox_poller.cancel()
        self._channel_reporter.cancel()
        if config.MEMBERS_INTENT:
            self._member_reporter.cancel()
        await self.hugpy.close()
        await super().close()

    def model_for(self, user_id: int) -> str | None:
        """Local fallback: the user's saved pref, else the configured default."""
        return self.prefs.get_model(user_id) or config.DEFAULT_MODEL_KEY

    # ── F4 runtime settings (central is the SoT; the bot only reads) ──────
    _SETTINGS_TTL = 10.0

    async def settings_ns(self, ns: str) -> dict:
        """Cached view of a central settings namespace. Central unreachable →
        the last known values (or {}), so a blip never mutes the bot."""
        cache = getattr(self, "_settings_cache", None)
        if cache is None:
            cache = self._settings_cache = {}
        hit = cache.get(ns)
        import time as _time
        now = _time.monotonic()
        if hit and now - hit[0] < self._SETTINGS_TTL:
            return hit[1]
        try:
            values = await self.hugpy.settings_ns(ns)
        except Exception:
            log.debug("settings fetch failed for %s", ns, exc_info=True)
            values = hit[1] if hit else {}
        cache[ns] = (now, values)
        return values

    async def channel_settings(self, channel_id) -> dict:
        return (await self.settings_ns("discord.channels")).get(
            str(channel_id)) or {}

    async def channel_personality(self, channel_id) -> dict | None:
        """DISC-06: the personality assigned to this channel (from the
        personality registry), or None."""
        name = (await self.channel_settings(channel_id)).get("personality")
        if not name:
            return None
        p = (await self.settings_ns("personalities")).get(str(name))
        return {**p, "name": name} if isinstance(p, dict) else None

    async def set_user_model(self, user_id: int, model_key: str | None) -> None:
        """Write-through: central settings (authoritative, console-visible)
        plus the local prefs file (offline fallback)."""
        try:
            await self.hugpy.set_user_model(user_id, model_key)
        except Exception:
            log.debug("central pref write failed; local only", exc_info=True)
        await self.prefs.set_model(user_id, model_key)

    async def resolve_model_for(self, user_id: int, channel_id: int | None = None) -> str | None:
        """Pick the model for a turn. Precedence (defined once, here):
        explicit per-turn choice is handled by the caller before this runs;
        then console-managed Discord binding > channel personality's model
        (the personality owns its model) > channel delegation (CON-07,
        fallback when the personality doesn't pin one) > user pref (central
        settings, then local file) > configured default."""
        try:
            bound = await self.hugpy.resolve_discord_model(channel_id=channel_id, user_id=user_id)
            if bound:
                return bound
        except Exception:
            log.debug("discord resolve failed; trying settings", exc_info=True)
        if channel_id is not None:
            persona = await self.channel_personality(channel_id)
            if persona and persona.get("model_key"):
                return persona["model_key"]
            delegated = (await self.settings_ns("delegation")).get(
                f"channel:{channel_id}")
            if delegated:
                return delegated
        central_pref = (await self.settings_ns("discord.users")).get(
            str(user_id)) or {}
        if central_pref.get("model"):
            return central_pref["model"]
        return self.model_for(user_id)

    # ── outbound: model -> Discord (the "mobile arm") ─────────────────────
    @tasks.loop(seconds=8.0)
    async def _outbox_poller(self) -> None:
        try:
            self._bridged = await self.hugpy.list_bridged_channels()
        except Exception:
            pass  # keep the last known set if central blips
        try:
            messages = await self.hugpy.drain_discord_outbox()
        except Exception:
            return  # central unreachable this tick; try again next loop
        for m in messages:
            await self._deliver_outbound(m)

    def is_bridged(self, channel_id) -> bool:
        return str(channel_id) in self._bridged

    async def relay_inbound(self, message) -> None:
        """Forward a bridged-channel message (text + any attachments) up to
        central's bridge inbox. Each attachment is pulled from Discord and
        uploaded to central so the console / keeper gets a stable local file;
        the original Discord CDN url is kept as a fallback ref."""
        try:
            author = getattr(message.author, "display_name", None) or str(message.author)
            attachments = []
            for att in (getattr(message, "attachments", None) or []):
                ref = {
                    "filename": getattr(att, "filename", None),
                    "content_type": getattr(att, "content_type", None),
                    "size": getattr(att, "size", None),
                    "url": getattr(att, "url", None),
                }
                try:
                    if (getattr(att, "size", 0) or 0) <= config.MAX_ATTACHMENT_BYTES:
                        up = await self.hugpy.upload(att.filename, await att.read())
                        ref["path"] = up.get("path")
                    else:
                        ref["skipped"] = "too large"
                except Exception:
                    log.debug("attachment upload failed for %s",
                              getattr(att, "filename", "?"), exc_info=True)
                attachments.append(ref)
            await self.hugpy.relay_inbound(
                channel_id=message.channel.id, author=author,
                content=message.content or "",
                attachments=attachments or None,
            )
        except Exception:
            log.debug("relay_inbound failed", exc_info=True)

    @_outbox_poller.before_loop
    async def _before_outbox(self) -> None:
        await self.wait_until_ready()

    # ── report visible channels -> central (for the console dropdown) ─────
    @tasks.loop(seconds=60.0)
    async def _channel_reporter(self) -> None:
        channels: list[dict] = []
        for guild in self.guilds:
            me = guild.me
            for ch in guild.text_channels:
                try:
                    if me is not None and not ch.permissions_for(me).send_messages:
                        continue  # only offer channels the bot can actually post in
                except Exception:
                    pass
                channels.append({
                    "id": str(ch.id),
                    "name": ch.name,
                    "guild": guild.name,
                    "guild_id": str(guild.id),
                })
        try:
            await self.hugpy.report_discord_channels(channels)
        except Exception:
            log.debug("channel report failed; will retry next loop", exc_info=True)

    @_channel_reporter.before_loop
    async def _before_channel_report(self) -> None:
        await self.wait_until_ready()

    # ── report guild members -> central (user dropdown; needs members intent) ─
    @tasks.loop(seconds=300.0)
    async def _member_reporter(self) -> None:
        users: dict[str, dict] = {}
        for guild in self.guilds:
            for m in guild.members:
                if m.bot:
                    continue  # don't offer bots as binding targets
                users[str(m.id)] = {
                    "id": str(m.id),
                    "name": m.display_name or m.name,
                    "guild": guild.name,
                }
        try:
            await self.hugpy.report_discord_users(list(users.values()))
        except Exception:
            log.debug("member report failed; will retry next loop", exc_info=True)

    @_member_reporter.before_loop
    async def _before_member_report(self) -> None:
        await self.wait_until_ready()

    async def _deliver_outbound(self, m: dict) -> None:
        content = (m.get("content") or "").strip()
        if not content:
            return
        channel_id, user_id = m.get("channel_id"), m.get("user_id")
        # Optional clickable choices (keeper escalation). Absent/empty -> the send
        # is byte-for-byte the plain-text path below (view stays None).
        options = m.get("options")
        view = None
        if isinstance(options, list) and options:
            try:
                view = _build_escalation_view(options, channel_id or user_id, m["id"])
            except Exception:
                log.warning("could not build escalation view for %s", m.get("id"), exc_info=True)
                view = None
        try:
            if channel_id:
                channel = self.get_channel(int(channel_id)) or await self.fetch_channel(int(channel_id))
                if view is not None:
                    await channel.send(content, view=view)
                else:
                    await channel.send(content)
            elif user_id:
                user = self.get_user(int(user_id)) or await self.fetch_user(int(user_id))
                if view is not None:
                    await user.send(content, view=view)
                else:
                    await user.send(content)
        except Exception:
            log.warning("could not deliver outbound discord message %s", m.get("id"), exc_info=True)



# ── entry point (``hugpy-bot``) ───────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hugpy-bot",
        description="Run the Hugpy Discord bot against a Hugpy central "
                    "(HUGPY_BASE_URL; default http://127.0.0.1:7002).")
    parser.add_argument("--env", metavar="FILE",
                        help="path to a .env holding DISCORD_TOKEN etc. "
                             "(sets HUGPY_BOT_ENV before config loads)")
    parser.add_argument("--base-url", metavar="URL",
                        help="Hugpy central base URL (overrides HUGPY_BASE_URL)")
    parser.add_argument("--guild", metavar="ID", type=int,
                        help="sync slash commands to one guild only (GUILD_ID)")
    parser.add_argument("--log-level", default="INFO",
                        help="root log level (default INFO)")
    return parser


def main(argv=None) -> int:
    """Console entry point: parse flags, resolve the token, run the bot.

    Returns a process exit code; only exits via SystemExit for ``--help``.
    """
    args = _build_parser().parse_args(argv)
    # The parse must happen before ``config`` is (re)loaded: the env file and the
    # base-url flag both feed module-level values there.
    if args.env:
        os.environ["HUGPY_BOT_ENV"] = args.env
    if args.base_url:
        os.environ["HUGPY_BASE_URL"] = args.base_url
    if args.guild:
        os.environ["GUILD_ID"] = str(args.guild)
    if args.env or args.base_url or args.guild:
        import importlib
        importlib.reload(config)

    try:
        token = config.get_discord_token()
    except RuntimeError as exc:
        print(f"hugpy-bot: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    for noisy in ("httpx", "httpcore", "discord.http"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    HugpyBot().run(token)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
