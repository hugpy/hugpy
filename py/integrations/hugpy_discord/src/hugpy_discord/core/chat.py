"""Transport-neutral conversational engine: the streaming chat turn, per-conversation
history, and the in-flight-generation registry behind /running and /stop.

Both the Discord ``ChatCog`` and the chatshare adapter own a :class:`ChatEngine`
bound to the one shared :class:`~hugpy_discord.bot.HugpyBot` (same HugpyClient,
model resolution, prefs and personalities). Each transport keeps its OWN engine so
Discord channel histories and chatshare room histories never mix, while the brain
(central) is shared. Histories are keyed by a transport conversation key (a Discord
channel id, or a ``cs:<room id>`` string) that never collides across transports.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
import uuid
from collections import defaultdict, deque

from hugpy_discord.config import HISTORY_MAX_TURNS, MESSAGE_CHAR_LIMIT
from hugpy_discord.hugpy_client import HugpyError
from hugpy_discord.streamer import MessageStreamer

log = logging.getLogger(__name__)


class ChatEngine:
    def __init__(self, bot, *, transport: str, char_limit: int = MESSAGE_CHAR_LIMIT,
                 transport_errors: tuple = ()):
        self.bot = bot
        self._transport = transport
        self._char_limit = char_limit
        # Exceptions a transport raises when a mid-stream edit fails (e.g.
        # discord.HTTPException). Caught like the Discord cog always did: log and
        # stop the turn rather than crash.
        self._transport_errors = transport_errors
        self._history: dict = defaultdict(lambda: deque(maxlen=HISTORY_MAX_TURNS * 2))
        # In-flight generations: turn id -> info dict (task, conv, prompt…).
        # Cancelling the task tears down the SSE stream, which makes central drop
        # the worker-side generation.
        self._active: dict[str, dict] = {}
        self._turn_ids = itertools.count(1)

    def reset(self, conv_key) -> None:
        self._history.pop(conv_key, None)

    # ── core streaming turn ────────────────────────────────────────────────
    async def run_turn(
        self,
        send,
        *,
        conv_key,
        user_id,
        prompt: str,
        model_key: str | None = None,
        file: str | None = None,
        remember: bool = True,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
    ) -> "MessageStreamer":
        from hugpy_discord.core.models import clean_model_key

        # A console-managed binding for this conversation/user wins; otherwise the
        # user's saved pref / configured default (resolved inside the bot).
        model_key = clean_model_key(model_key) or await self.bot.resolve_model_for(
            user_id, conv_key)
        history = list(self._history[conv_key]) if remember else []
        messages = history + [{"role": "user", "content": prompt}]

        # A channel personality bundles system prompt + params. Its system prompt
        # heads the message list every turn; its params are DEFAULTS only — explicit
        # per-turn values keep winning. (Its model was already applied in
        # resolve_model_for; explicit model wins.)
        persona = await self.bot.channel_personality(conv_key)
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

        streamer = MessageStreamer(send, char_limit=self._char_limit)
        turn_id = f"t{next(self._turn_ids)}"
        # The request_id central's job store tracks this turn under — /stop cancels
        # through it, so the generation actually stops server-side instead of us
        # just dropping our end of the SSE pipe.
        request_id = uuid.uuid4().hex
        self._active[turn_id] = {
            "task": asyncio.current_task(),
            "request_id": request_id,
            "channel_id": conv_key,
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
                transport=self._transport, channel=str(conv_key),
            ):
                await streamer.feed(chunk)
            await streamer.finish()
        except asyncio.CancelledError:
            log.info("chat turn %s stopped via /stop", turn_id)
            try:
                await streamer.fail("stopped via /stop")
            except Exception:
                pass
            return streamer
        except HugpyError as exc:
            log.warning("chat turn failed: %s", exc)
            await streamer.fail(str(exc))
            return streamer
        except self._transport_errors:
            log.exception("transport edit failed mid-stream")
            return streamer
        finally:
            self._active.pop(turn_id, None)

        if remember and streamer.full_text:
            self._history[conv_key].append({"role": "user", "content": prompt})
            self._history[conv_key].append(
                {"role": "assistant", "content": streamer.full_text})
        return streamer

    # ── /running + /stop ───────────────────────────────────────────────────
    def active_snapshot(self) -> list[dict]:
        """A point-in-time view of running turns for /running (with elapsed secs)."""
        now = time.monotonic()
        return [
            {"turn_id": tid, "channel_id": info["channel_id"],
             "model": info["model"], "prompt": info["prompt"],
             "elapsed": int(now - info["started"])}
            for tid, info in self._active.items()
        ]

    @property
    def active(self) -> dict[str, dict]:
        return self._active

    async def stop(self, *, conv_key=None, turn_id: str | None = None):
        """Stop a specific turn (``turn_id``) or every turn in ``conv_key``.

        Returns ``(stopped_ids, unknown_turn)``: ``unknown_turn`` is True only when
        a ``turn_id`` was given that is not running. Mirrors the Discord /stop:
        central-side cancel first (frees the slot), then the task cancel tears down
        our SSE read.
        """
        if turn_id is not None:
            info = self._active.get(turn_id)
            if not info:
                return [], True
            targets = {turn_id: info}
        else:
            targets = {tid: info for tid, info in self._active.items()
                       if info["channel_id"] == conv_key}
        for info in targets.values():
            rid = info.get("request_id")
            if rid:
                try:
                    await self.bot.hugpy.cancel_chat(rid)
                except Exception as exc:
                    log.debug("central-side cancel of %s failed: %s", rid, exc)
            info["task"].cancel()
        return list(targets), False


async def stream_prompt(bot, send, *, prompt: str, model: str | None,
                        file: str | None, char_limit: int) -> None:
    """Original /chat/stream path — the compatibility fallback used by the
    task commands when central has no POST /prompt route yet. ``model`` must be
    resolved by the caller (the cog/adapter), mirroring the Discord fallback."""
    streamer = MessageStreamer(send, char_limit=char_limit)
    try:
        async for chunk in bot.hugpy.chat_stream(prompt=prompt, model_key=model,
                                                 file=file):
            await streamer.feed(chunk)
        await streamer.finish()
    except HugpyError as exc:
        await streamer.fail(str(exc))
