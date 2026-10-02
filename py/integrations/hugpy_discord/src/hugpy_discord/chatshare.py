"""Chatshare front end for the hugpy-bot process.

A second transport for the SAME :class:`~hugpy_discord.bot.HugpyBot` brain: it
connects to the abstractendeavors.com chat service's bot websocket
(``/chat/bot``) and exposes the Discord commands there, @mention/DM chat with
streamed replies, and per-room conversation memory — all through the shared
:class:`~hugpy_discord.core.chat.ChatEngine` and
:mod:`hugpy_discord.core.commands`. The chat service stays generic; all hugpy
knowledge lives here.

The adapter is DORMANT unless ``CHATSHARE_BOT_TOKEN`` is set, and it is isolated
from the Discord arm: every handler is guarded and the connection loop reconnects
with exponential backoff forever, so nothing here can crash or block Discord.

Transport note: the websocket client is aiohttp's ``ws_connect`` (aiohttp already
ships in the bot's venv via discord.py), with NO Origin header — bot tokens must
never be used from a browser, and the service refuses a bot connection that
carries one.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import itertools
import json
import logging

import aiohttp

from hugpy_discord import config
from hugpy_discord.core import commands as core
from hugpy_discord.core.chat import ChatEngine, stream_prompt
from hugpy_discord.core.identity import cs_conv_key, cs_user_key
from hugpy_discord.core.models import model_choice_pairs
from hugpy_discord.core.results import CommandResult
from hugpy_discord.core.specs import command_specs
from hugpy_discord.hugpy_client import HugpyError

log = logging.getLogger(__name__)

MAX_FRAME = 24 * 1024 * 1024          # service cap for file.put frames
REQUEST_TIMEOUT = 620.0               # generation + download polling are slow
# Commands that defer on the Discord side (show a "thinking" state) — mirror it.
# `tasks` and `canceljob` reply ephemerally; chatshare's defer broadcasts a
# room-wide bot.thinking, which would leak an otherwise-private invocation, so
# they are NOT deferred here (Discord defers them ephemerally — closest match is
# to skip the public thinking state and just send the ephemeral reply).
_DEFERRED = {
    "chat", "embed", "similarity", "imagine", "task", "status",
    "models", "download", "jobs", "hf", "summarize", "keywords",
    "transcribe", "describe",
}


class _Handle:
    """A streamed chatshare message the :class:`MessageStreamer` edits in place."""
    def __init__(self, adapter, invocation_id, *, message_id=None, ephemeral_id=None):
        self._adapter = adapter
        self._iid = invocation_id
        self._message_id = message_id
        self._ephemeral_id = ephemeral_id

    async def edit(self, *, content: str) -> None:
        body = content[: self._adapter.body_max]
        if self._ephemeral_id is not None:
            await self._adapter.request("msg.edit", ephemeral_id=self._ephemeral_id,
                                        body=body)
        else:
            await self._adapter.request("msg.edit", message_id=self._message_id,
                                        body=body)


class ChatshareAdapter:
    def __init__(self, bot, *, url: str, token: str):
        self.bot = bot
        self._url = url
        self._token = token
        self.engine = ChatEngine(
            bot, transport="chatshare", char_limit=config.CHATSHARE_CHAR_LIMIT,
            transport_errors=(aiohttp.ClientError, ConnectionError,
                              asyncio.TimeoutError))
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._rids = itertools.count(1)
        self._send_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.bot_name: str | None = None
        self.body_max = config.CHATSHARE_CHAR_LIMIT

    # ── connection lifecycle ────────────────────────────────────────────────
    async def run(self) -> None:
        """Connect-serve-reconnect forever. Never raises (except on cancel)."""
        backoff = 1.0
        while True:
            try:
                await self._serve()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — isolate from the Discord arm
                log.warning("chatshare adapter disconnected: %s", exc)
            log.info("chatshare adapter reconnecting in %.0fs", backoff)
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, 60.0)

    async def _serve(self) -> None:
        headers = {"Authorization": f"Bot {self._token}"}
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # NO Origin header: ws_connect(origin=None) sends none, which is what
            # the service requires for a bot token.
            async with session.ws_connect(
                self._url, headers=headers, origin=None, max_msg_size=MAX_FRAME,
                heartbeat=30.0,
            ) as ws:
                self._ws = ws
                log.info("chatshare adapter connected to %s", self._url)
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._on_frame(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                          aiohttp.WSMsgType.ERROR):
                            break
                finally:
                    self._ws = None
                    self._fail_pending(ConnectionError("chatshare connection closed"))

    def _fail_pending(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    # ── frame routing ───────────────────────────────────────────────────────
    def _on_frame(self, frame: dict) -> None:
        kind = frame.get("t")
        if kind == "res":
            fut = self._pending.pop(frame.get("rid"), None)
            if fut is not None and not fut.done():
                fut.set_result(frame)
            return
        if kind != "ev":
            return
        ev = frame.get("ev")
        if ev == "bot.ready":
            self.bot_name = (frame.get("bot") or {}).get("name")
            limits = frame.get("limits") or {}
            self.body_max = int(limits.get("body_max") or config.CHATSHARE_CHAR_LIMIT)
            self.engine._char_limit = self.body_max
            self._spawn(self._set_commands())
        elif ev == "invoke":
            self._spawn(self._handle_invoke(frame))
        elif ev == "autocomplete":
            self._spawn(self._handle_autocomplete(frame))

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ── request / notify plumbing ───────────────────────────────────────────
    async def request(self, t: str, **params):
        if self._ws is None:
            raise ConnectionError("chatshare not connected")
        rid = f"r{next(self._rids)}"
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        async with self._send_lock:
            await self._ws.send_str(json.dumps({"t": t, "rid": rid, **params}))
        try:
            frame = await asyncio.wait_for(fut, REQUEST_TIMEOUT)
        finally:
            self._pending.pop(rid, None)
        if not frame.get("ok", False):
            err = frame.get("error") or {}
            raise HugpyError(err.get("message") or err.get("code") or "request failed")
        return frame.get("data") or {}

    async def notify(self, t: str, **params) -> None:
        if self._ws is None:
            return
        async with self._send_lock:
            await self._ws.send_str(json.dumps({"t": t, **params}))

    async def _set_commands(self) -> None:
        try:
            await self.request("commands.set", commands=command_specs())
            log.info("chatshare adapter registered %d commands", len(command_specs()))
        except Exception:
            log.warning("chatshare commands.set failed", exc_info=True)

    # ── invoke handling ─────────────────────────────────────────────────────
    async def _handle_invoke(self, frame: dict) -> None:
        try:
            kind = frame.get("kind")
            if kind in ("mention", "dm"):
                await self._handle_chat_message(frame)
            elif kind == "command":
                await self._handle_command(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("chatshare invoke handling failed", exc_info=True)

    async def _handle_chat_message(self, frame: dict) -> None:
        iid = frame["invocation_id"]
        room = frame.get("room") or {}
        author = frame.get("author") or {}
        message = frame.get("message") or {}
        conv = cs_conv_key(room.get("id"))
        uid = cs_user_key(author.get("id"))

        prompt = self._strip_mention(message.get("body") or "")
        file_path = None
        attachments = message.get("attachments") or []
        if attachments:
            try:
                file_path = await self._forward_attachment(attachments[0])
            except HugpyError as exc:
                await self._reply(iid, f"⚠️ couldn't forward attachment: {exc}")
                return
            if not prompt:
                prompt = "Describe this file."
        if not prompt:
            return

        with contextlib.suppress(Exception):
            await self.notify("typing", invocation_id=iid)
        await self.engine.run_turn(
            self._stream_sender(iid, ephemeral=False),
            conv_key=conv, user_id=uid, prompt=prompt, file=file_path)

    def _strip_mention(self, body: str) -> str:
        if self.bot_name:
            import re
            body = re.sub(rf"(?i)@{re.escape(self.bot_name)}\b", "", body)
        return body.strip()

    # ── command dispatch ────────────────────────────────────────────────────
    async def _handle_command(self, frame: dict) -> None:
        iid = frame["invocation_id"]
        command = frame.get("command") or {}
        name = command.get("name")
        args = command.get("args") or {}
        author = frame.get("author") or {}
        room = frame.get("room") or {}
        uid = cs_user_key(author.get("id"))
        conv = cs_conv_key(room.get("id"))

        if name in _DEFERRED:
            with contextlib.suppress(Exception):
                await self.request("invocation.defer", invocation_id=iid)

        handler = getattr(self, f"_cmd_{name.replace(' ', '_').replace('-', '_')}", None)
        if handler is None:
            await self._reply(iid, f"⚠️ unknown command `{name}`", ephemeral=True)
            return
        await handler(iid, args, uid, conv)

    # chat-family commands ----------------------------------------------------
    async def _cmd_chat(self, iid, args, uid, conv) -> None:
        private = bool(args.get("private"))
        file_path = None
        att = args.get("attachment")
        if att:
            try:
                file_path = await self._forward_attachment(att)
            except HugpyError as exc:
                await self._reply(iid, f"⚠️ {exc}", ephemeral=True)
                return
        await self.engine.run_turn(
            self._stream_sender(iid, ephemeral=private),
            conv_key=conv, user_id=uid, prompt=args.get("prompt") or "",
            model_key=args.get("model"), file=file_path, remember=not private,
            temperature=args.get("temperature"), top_p=args.get("top_p"),
            max_new_tokens=args.get("max_tokens"), do_sample=args.get("do_sample"))

    async def _cmd_reset(self, iid, args, uid, conv) -> None:
        self.engine.reset(conv)
        await self._reply(iid, "🧹 history cleared", ephemeral=True)

    async def _cmd_running(self, iid, args, uid, conv) -> None:
        snapshot = self.engine.active_snapshot()
        if not snapshot:
            await self._reply(iid, "nothing is generating right now", ephemeral=True)
            return
        lines = [f"running generations ({len(snapshot)})"]
        lines += [
            f"`{s['turn_id']}` — `{s['channel_id']}` — `{s['model'] or 'default'}`"
            f" — {s['elapsed']}s — “{s['prompt'][:60]}”"
            for s in snapshot]
        lines.append("-# stop one with /stop <id>, or /stop in this room")
        await self._reply(iid, "\n".join(lines), ephemeral=True)

    async def _cmd_stop(self, iid, args, uid, conv) -> None:
        turn_id = args.get("turn_id")
        stopped, unknown = await self.engine.stop(conv_key=conv, turn_id=turn_id)
        if unknown:
            await self._reply(iid, f"no running generation `{turn_id}` — see /running",
                              ephemeral=True)
        elif not stopped:
            await self._reply(iid, "nothing is generating in this room — see /running",
                              ephemeral=True)
        else:
            await self._reply(iid, "⏹️ stopped " + ", ".join(f"`{t}`" for t in stopped),
                              ephemeral=True)

    async def _cmd_model_set(self, iid, args, uid, conv) -> None:
        model = args.get("model")
        await self.bot.set_user_model(uid, model)
        await self._reply(iid, f"✅ default model: `{model}`", ephemeral=True)

    async def _cmd_model_show(self, iid, args, uid, conv) -> None:
        model = self.bot.model_for(uid)
        text = (f"current model: `{model}`" if model
                else "no default model set (central decides)")
        await self._reply(iid, text, ephemeral=True)

    async def _cmd_model_clear(self, iid, args, uid, conv) -> None:
        await self.bot.set_user_model(uid, None)
        await self._reply(iid, "✅ default model cleared", ephemeral=True)

    # ML / tools / ops commands ----------------------------------------------
    async def _cmd_embed(self, iid, args, uid, conv) -> None:
        from hugpy_discord.core.models import split_items
        res = await core.embed_core(
            self.bot, texts=split_items(args.get("text") or ""),
            normalize=args.get("normalize"), batch_size=args.get("batch_size"),
            model=args.get("model"))
        await self._render(iid, res)

    async def _cmd_similarity(self, iid, args, uid, conv) -> None:
        from hugpy_discord.core.models import split_items
        res = await core.similarity_core(
            self.bot, text=args.get("text") or "",
            candidates=split_items(args.get("compare_to") or ""),
            normalize=args.get("normalize"), model=args.get("model"))
        await self._render(iid, res)

    async def _cmd_imagine(self, iid, args, uid, conv) -> None:
        res = await core.imagine_core(
            self.bot, prompt=args.get("prompt") or "", negative=args.get("negative"),
            width=args.get("width"), height=args.get("height"), steps=args.get("steps"),
            guidance=args.get("guidance"), seed=args.get("seed"),
            count=args.get("count") or 1, model=args.get("model"))
        await self._render(iid, res)

    async def _cmd_task(self, iid, args, uid, conv) -> None:
        file_path = None
        att = args.get("attachment")
        if att:
            try:
                file_path = await self._forward_attachment(att)
            except HugpyError as exc:
                await self._reply(iid, f"⚠️ {exc}", ephemeral=True)
                return
        res = await core.task_core(
            self.bot, task=args.get("task"), input=args.get("input"), file=file_path,
            model=args.get("model"), params=args.get("params"),
            temperature=args.get("temperature"), top_p=args.get("top_p"),
            max_tokens=args.get("max_tokens"), do_sample=args.get("do_sample"))
        await self._render(iid, res)

    async def _cmd_tasks(self, iid, args, uid, conv) -> None:
        await self._render(iid, await core.tasks_core(self.bot))

    async def _cmd_status(self, iid, args, uid, conv) -> None:
        await self._render(iid, await core.status_core(self.bot))

    async def _cmd_models(self, iid, args, uid, conv) -> None:
        await self._render(iid, await core.models_core(
            self.bot, installed_only=bool(args.get("installed_only"))))

    async def _cmd_jobs(self, iid, args, uid, conv) -> None:
        await self._render(iid, await core.jobs_core(self.bot))

    async def _cmd_canceljob(self, iid, args, uid, conv) -> None:
        await self._render(iid, await core.canceljob_core(
            self.bot, job_id=args.get("job_id")))

    async def _cmd_hf(self, iid, args, uid, conv) -> None:
        await self._render(iid, await core.hf_core(
            self.bot, query=args.get("query") or "", task=args.get("task")))

    async def _cmd_download(self, iid, args, uid, conv) -> None:
        handle = None
        async for line in core.download_core(self.bot, model=args.get("model")):
            if handle is None:
                data = await self.request("msg.send", invocation_id=iid,
                                          body=line[: self.body_max])
                handle = _Handle(self, iid, message_id=data["message"]["id"])
            else:
                await handle.edit(content=line)

    async def _cmd_summarize(self, iid, args, uid, conv) -> None:
        file_path = await self._maybe_forward(iid, args)
        if file_path is False:
            return
        res = await core.summarize_core(
            self.bot, text=args.get("text"), file=file_path, mode=args.get("mode"),
            preset=args.get("preset"), model=args.get("model"))
        await self._deliver(iid, res, uid=uid, model=args.get("model"))

    async def _cmd_keywords(self, iid, args, uid, conv) -> None:
        file_path = await self._maybe_forward(iid, args)
        if file_path is False:
            return
        res = await core.keywords_core(
            self.bot, text=args.get("text"), file=file_path, preset=args.get("preset"),
            top_n=args.get("top_n"), diversity=args.get("diversity"),
            model=args.get("model"))
        await self._deliver(iid, res, uid=uid, model=args.get("model"))

    async def _cmd_transcribe(self, iid, args, uid, conv) -> None:
        file_path = await self._maybe_forward(iid, args, required=True)
        if file_path in (False, None):
            return
        res = await core.transcribe_core(
            self.bot, file=file_path, language=args.get("language"),
            size=args.get("size"), translate=bool(args.get("translate")),
            timestamps=bool(args.get("timestamps")), model=args.get("model"))
        await self._deliver(iid, res, uid=uid, model=args.get("model"))

    async def _cmd_describe(self, iid, args, uid, conv) -> None:
        file_path = await self._maybe_forward(iid, args, required=True)
        if file_path in (False, None):
            return
        res = await core.describe_core(
            self.bot, file=file_path, prompt=args.get("prompt"),
            max_tokens=args.get("max_tokens"), model=args.get("model"))
        await self._deliver(iid, res, uid=uid, model=args.get("model"))

    # ── autocomplete ────────────────────────────────────────────────────────
    async def _handle_autocomplete(self, frame: dict) -> None:
        aid = frame.get("aid")
        try:
            if frame.get("option") == "model":
                choices = await model_choice_pairs(self.bot, frame.get("value") or "")
            else:
                choices = []
            await self.request("autocomplete.result", aid=aid, choices=choices)
        except Exception:
            log.debug("chatshare autocomplete failed", exc_info=True)
            with contextlib.suppress(Exception):
                await self.request("autocomplete.result", aid=aid, choices=[])

    # ── rendering helpers ────────────────────────────────────────────────────
    def _stream_sender(self, invocation_id, *, ephemeral: bool):
        async def send(content: str):
            body = content[: self.body_max] or "…"
            if ephemeral:
                data = await self.request("msg.send", invocation_id=invocation_id,
                                          body=body, ephemeral=True)
                return _Handle(self, invocation_id, ephemeral_id=data["ephemeral_id"])
            data = await self.request("msg.send", invocation_id=invocation_id, body=body)
            return _Handle(self, invocation_id, message_id=data["message"]["id"])
        return send

    async def _reply(self, iid, text: str, *, ephemeral: bool = False) -> None:
        body = (text or "")[: self.body_max] or "*(empty)*"
        kwargs = {"ephemeral": True} if ephemeral else {}
        with contextlib.suppress(Exception):
            await self.request("msg.send", invocation_id=iid, body=body, **kwargs)

    async def _deliver(self, iid, result: CommandResult, *, uid, model) -> None:
        if result.fallback_prompt is not None:
            await stream_prompt(
                self.bot, self._stream_sender(iid, ephemeral=False),
                prompt=result.fallback_prompt,
                model=model or self.bot.model_for(uid),
                file=result.fallback_file, char_limit=self.body_max)
            return
        await self._render(iid, result)

    async def _render(self, iid, result: CommandResult) -> None:
        eph = result.ephemeral
        if result.files:
            attachments = []
            for f in result.files:
                fp = await self.request(
                    "file.put", invocation_id=iid, filename=f.filename, mime=f.mime,
                    data_b64=base64.b64encode(f.data).decode())
                attachments.append({"bot_file_id": fp["bot_file_id"]})
            await self.request("msg.send", invocation_id=iid,
                               body=(result.text or "")[: self.body_max] or "📎",
                               attachments=attachments)
            return
        if result.sections or result.title:
            parts = []
            if result.title:
                parts.append(f"**{result.title}**")
            if result.text:
                parts.append(result.text)
            for name, value in result.sections:
                parts.append(f"**{name}**\n{value}")
            if result.footer:
                parts.append(f"-# {result.footer}")
            await self._reply(iid, "\n\n".join(parts), ephemeral=eph)
            return
        if result.long:
            await self._send_long(iid, result.text, filename=result.filename,
                                  ephemeral=eph)
            return
        await self._reply(iid, result.text, ephemeral=eph)

    async def _send_long(self, iid, text: str, *, filename: str,
                         ephemeral: bool) -> None:
        text = (text or "").strip() or "*<empty result>*"
        if len(text) <= self.body_max or ephemeral:
            await self._reply(iid, text[: self.body_max], ephemeral=ephemeral)
            return
        fp = await self.request(
            "file.put", invocation_id=iid, filename=filename, mime="text/plain",
            data_b64=base64.b64encode(text.encode("utf-8")).decode())
        note = text[: self.body_max - 100] + "\n… *(full result attached)*"
        await self.request("msg.send", invocation_id=iid, body=note,
                           attachments=[{"bot_file_id": fp["bot_file_id"]}])

    # ── attachment forwarding (share_url → central path) ─────────────────────
    async def _maybe_forward(self, iid, args, *, required: bool = False):
        """Resolve an ``attachment`` option to a central path. Returns the path, or
        ``None`` when absent (and not required), or ``False`` when it errored
        (already reported). ``required`` with no attachment is a reported error."""
        att = args.get("attachment")
        if not att:
            if required:
                await self._reply(iid, "⚠️ an attachment is required", ephemeral=True)
                return False
            return None
        try:
            return await self._forward_attachment(att)
        except HugpyError as exc:
            await self._reply(iid, f"⚠️ {exc}", ephemeral=True)
            return False

    async def _forward_attachment(self, att: dict) -> str:
        """Download a chatshare attachment (``share_url``) and upload it to central,
        exactly as ``forward_attachment`` does for Discord — honoring
        MAX_ATTACHMENT_BYTES."""
        share_url = att.get("share_url")
        filename = att.get("filename") or "attachment"
        size = att.get("size")
        limit = config.MAX_ATTACHMENT_BYTES
        if not share_url:
            raise HugpyError("attachment is not shareable (no share_url)")
        if size and size > limit:
            raise HugpyError(
                f"attachment too large ({size // (1024 * 1024)} MB; "
                f"limit {limit // (1024 * 1024)} MB)")
        data = await self._download(share_url, limit)
        uploaded = await self.bot.hugpy.upload(filename, data)
        return uploaded["path"]

    async def _download(self, url: str, limit: int) -> bytes:
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status >= 400:
                    raise HugpyError(f"could not fetch attachment (HTTP {resp.status})")
                chunks, total = [], 0
                async for chunk in resp.content.iter_chunked(1 << 16):
                    total += len(chunk)
                    if total > limit:
                        raise HugpyError(
                            f"attachment too large (limit {limit // (1024 * 1024)} MB)")
                    chunks.append(chunk)
                return b"".join(chunks)


def start(bot) -> "asyncio.Task | None":
    """Start the chatshare adapter as a background task on the bot's loop IFF a
    token is configured. Returns the task (or None when dormant). Never raises."""
    token = config.CHATSHARE_BOT_TOKEN
    if not token:
        log.info("chatshare adapter dormant (CHATSHARE_BOT_TOKEN not set)")
        return None
    adapter = ChatshareAdapter(bot, url=config.CHATSHARE_BOT_URL, token=token)
    bot.chatshare = adapter
    task = asyncio.ensure_future(adapter.run())
    log.info("chatshare adapter started against %s", config.CHATSHARE_BOT_URL)
    return task
