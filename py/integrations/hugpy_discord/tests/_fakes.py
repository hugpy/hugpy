"""Fakes for the transport-neutral core + the chatshare adapter tests.

No Discord, no real network: a fake HugpyClient stands in for central, a fake bot
exposes the surface the core calls, and a fake chat-service websocket records what
the adapter sends and auto-answers its requests.
"""
from __future__ import annotations

import asyncio
import itertools
import json

from hugpy_discord.hugpy_client import HugpyError


class FakeHugpy:
    def __init__(self):
        self.base_url = "http://central.test"
        self.cancelled: list[str] = []
        self.uploads: list[tuple[str, int]] = []
        self.last_stream: dict | None = None
        self.last_prompt: dict | None = None
        self.user_prefs: dict = {}
        # test knobs
        self.stream_chunks = ["Hello", " world"]
        self.prompt_result: dict = {}
        self.prompt_error: HugpyError | None = None
        self.gate: asyncio.Event | None = None   # set to block mid-stream

    async def chat_stream(self, **kw):
        self.last_stream = kw
        for i, c in enumerate(self.stream_chunks):
            if self.gate is not None and i == 1:
                await self.gate.wait()
            yield c

    async def execute_prompt(self, **kw):
        self.last_prompt = kw
        if self.prompt_error is not None:
            raise self.prompt_error
        return self.prompt_result

    async def list_models(self):
        return [{"key": "alpha", "status": "installed"},
                {"key": "beta", "status": "downloading"}]

    async def upload(self, name, data):
        self.uploads.append((name, len(data)))
        return {"path": f"/srv/uploads/{name}"}

    async def cancel_chat(self, rid):
        self.cancelled.append(rid)
        return {}

    async def health(self):
        return {"storage_root": "/srv/storage"}

    async def serving(self):
        return ["alpha"]

    async def list_workers(self):
        return [{"name": "w1", "status": "idle", "models": ["alpha"]}]

    async def list_jobs(self):
        return [{"id": "j1", "model_key": "alpha", "status": "done"}]

    async def cancel_job(self, jid):
        return {}

    async def download_model(self, m):
        return {"id": "job-local"}

    async def download_repo(self, m):
        return {"id": "job-repo"}

    async def get_job(self, jid):
        return {"status": "done"}

    async def hf_search(self, q, limit=10, task=None):
        return [{"id": "org/model", "downloads": 10, "likes": 3}]

    async def supported_tasks(self):
        return {"tasks": ["text-generation"], "defaults": {"text-generation": "alpha"}}

    async def resolve_discord_model(self, **kw):
        return None

    async def settings_ns(self, ns):
        return {}

    async def set_user_model(self, uid, model):
        self.user_prefs[str(uid)] = model
        return {}


class FakeBot:
    def __init__(self):
        self.hugpy = FakeHugpy()
        self._models: dict = {}

    async def resolve_model_for(self, user_id, channel_id=None):
        return self._models.get(str(user_id)) or "default-model"

    async def channel_personality(self, channel_id):
        return None

    def model_for(self, user_id):
        return self._models.get(str(user_id))

    async def set_user_model(self, user_id, model):
        if model is None:
            self._models.pop(str(user_id), None)
        else:
            self._models[str(user_id)] = model
        await self.hugpy.set_user_model(user_id, model)


def collecting_sender():
    """A streamer `send` that records every message and its final edited content."""
    messages: list[list[str]] = []    # each entry: [current content]

    async def send(content):
        slot = [content]
        messages.append(slot)

        class Handle:
            async def edit(self, *, content):
                slot[0] = content
        return Handle()

    return send, messages


class FakeService:
    """A fake /chat/bot peer: records frames the adapter sends and auto-answers
    its requests so adapter handlers run to completion in-process."""
    def __init__(self):
        self.sent: list[dict] = []
        self.adapter = None
        self._mid = itertools.count(1)
        self._eid = itertools.count(1)
        self._fid = itertools.count(1)

    async def send_str(self, s):
        frame = json.loads(s)
        self.sent.append(frame)
        rid = frame.get("rid")
        if rid is None:
            return
        self.adapter._on_frame(
            {"t": "res", "rid": rid, "ok": True, "data": self._respond(frame)})

    def _respond(self, frame):
        t = frame.get("t")
        if t == "msg.send":
            if frame.get("ephemeral"):
                return {"ephemeral_id": f"e{next(self._eid)}"}
            return {"message": {"id": next(self._mid), "bot": {"name": "hugpy"}}}
        if t == "file.put":
            n = next(self._fid)
            return {"bot_file_id": n, "url": f"http://files.test/{n}"}
        return {}

    def frames(self, t):
        return [f for f in self.sent if f.get("t") == t]

    def bodies(self, t="msg.send"):
        return [f.get("body", "") for f in self.frames(t)]
