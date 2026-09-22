"""Async HTTP client for a Hugpy central service.

Endpoint map (see hugpy_server.app.routes):
    GET  /health
    GET  /models                     -> list of manifest entries w/ status
    GET  /models/<key>
    POST /models/<key>/download      -> download job dict
    DELETE /models/<key>
    POST /llm/repos/download         -> acquire arbitrary HF repo
    GET  /jobs, /jobs/<id>; POST /jobs/<id>/cancel
    GET  /llm/workers, /llm/serving
    GET  /search?q=...               -> HF hub search
    POST /uploads (multipart "file") -> {"path","name","size"}
    POST /chat/stream                -> SSE: {"type":"token"|"done"|"error",...}
    POST /prompt                     -> execute_prompt passthrough (any task)
    GET  /prompt/tasks               -> {"tasks": [...], "defaults": {...}}
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import httpx


class HugpyError(RuntimeError):
    """Raised when central returns an error response or event."""


def _default_api_key() -> str:
    """The bot's M2M credential for central's MEMBER-gated plane.

    2026-08-06: ``member_auth`` closed ``/chat/stream``, ``/ml``, ``/uploads``
    and ``/session`` to anonymous callers, which broke this arm — the bot is not
    a browser with a console session, so it had no credential at all and every
    chat turn came back 401. The gate's documented M2M seam is a hugpy API key
    (``member_auth._presented_api_key`` -> ``X-API-Key`` header, falling back to
    a non-``hpv_`` bearer), so that is what the bot presents. The key is read
    from the environment (``HUGPY_API_KEY``, with ``HUGPY_BOT_API_KEY`` as an
    arm-specific override) — never hard-coded — and lives in the bot's .env
    alongside DISCORD_TOKEN. Empty when unset: the client then behaves exactly as
    it did before, so a self-hosted ``open``-mode deployment (where the gate does
    not enforce) needs no key."""
    return (os.getenv("HUGPY_BOT_API_KEY") or os.getenv("HUGPY_API_KEY") or "").strip()


class HugpyClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url
        self.api_key = (api_key if api_key is not None else _default_api_key()).strip()
        # Set on the CLIENT (not per call) so every request — the JSON verbs,
        # the multipart upload, and the streaming POST /chat/stream — carries it.
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            # Generation and model downloads are slow; only connect fails fast.
            timeout=httpx.Timeout(600.0, connect=5.0),
        )

    async def close(self) -> None:
        await self._http.aclose()

    # ── plumbing ──────────────────────────────────────────────────────────
    async def _json(self, method: str, path: str, **kwargs) -> Any:
        resp = await self._http.request(method, path, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error") or resp.json().get("description")
            except Exception:
                detail = resp.text[:300]
            raise HugpyError(f"{method} {path} -> {resp.status_code}: {detail}")
        try:
            return resp.json()
        except Exception:
            # Central's SPA catch-all answers unknown GETs with HTML 200.
            raise HugpyError(f"{method} {path} -> non-JSON response (HTTP {resp.status_code})")

    # ── service / models / jobs / workers ─────────────────────────────────
    async def health(self) -> dict:
        return await self._json("GET", "/health")

    async def list_models(self) -> list[dict]:
        return await self._json("GET", "/models")

    async def get_model(self, model_key: str) -> dict:
        return await self._json("GET", f"/models/{model_key}")

    async def download_model(self, model_key: str) -> dict:
        return await self._json("POST", f"/models/{model_key}/download", json={})

    # ── discord bindings (console-managed model <-> channel/user) ─────────
    async def resolve_discord_model(self, *, channel_id=None, user_id=None) -> str | None:
        """Ask central which model a (channel, user) is bound to, or None."""
        params: dict[str, str] = {}
        if channel_id is not None:
            params["channel_id"] = str(channel_id)
        if user_id is not None:
            params["user_id"] = str(user_id)
        data = await self._json("GET", "/discord/resolve", params=params)
        return data.get("model_key")

    async def drain_discord_outbox(self) -> list[dict]:
        """Claim undelivered model-originated messages to push into Discord."""
        data = await self._json("POST", "/discord/outbox/drain", json={})
        return data.get("messages", [])

    async def report_discord_channels(self, channels: list[dict]) -> dict:
        """Tell central which text channels the bot can currently see, so the
        console can offer them as a dropdown when creating a binding."""
        return await self._json("POST", "/discord/channels", json={"channels": channels})

    async def report_discord_users(self, users: list[dict]) -> dict:
        """Tell central the guild members the bot can see (needs the privileged
        members intent), so the console can offer a user dropdown."""
        return await self._json("POST", "/discord/users", json={"users": users})

    async def list_bridged_channels(self) -> set[str]:
        """Channel ids that are bridged to a console session. The bot relays
        inbound messages for these and lets the bridge own the response."""
        data = await self._json("GET", "/discord/bridges")
        return {str(b["channel_id"]) for b in data.get("bridges", []) if b.get("channel_id")}

    async def relay_inbound(self, *, channel_id, author=None, content="",
                            attachments=None) -> dict:
        """Forward an inbound channel message (text + attachments) to central's
        bridge inbox."""
        body = {"channel_id": str(channel_id), "author": author, "content": content}
        if attachments:
            body["attachments"] = attachments
        return await self._json("POST", "/discord/inbox", json=body)

    async def delete_model(self, model_key: str) -> dict:
        return await self._json("DELETE", f"/models/{model_key}")

    async def download_repo(self, hub_id: str, register: bool = True) -> dict:
        return await self._json(
            "POST", "/llm/repos/download", json={"hub_id": hub_id, "register": register}
        )

    async def list_jobs(self) -> list[dict]:
        return await self._json("GET", "/jobs")

    async def get_job(self, job_id: str) -> dict:
        return await self._json("GET", f"/jobs/{job_id}")

    async def cancel_job(self, job_id: str) -> dict:
        return await self._json("POST", f"/jobs/{job_id}/cancel")

    async def list_workers(self) -> list[dict]:
        return await self._json("GET", "/llm/workers")

    async def serving(self) -> Any:
        return await self._json("GET", "/llm/serving")

    async def hf_search(self, query: str, limit: int = 10, task: str | None = None) -> Any:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if task:
            params["task"] = task
        return await self._json("GET", "/search", params=params)

    # ── uploads ───────────────────────────────────────────────────────────
    async def upload(self, filename: str, data: bytes) -> dict:
        """Ship a file to central; returns its server-side path for file-chat."""
        return await self._json(
            "POST", "/uploads", files={"file": (filename, data)}
        )

    # ── execute_prompt passthrough ────────────────────────────────────────
    async def execute_prompt(self, **kwargs: Any) -> dict:
        """Mirror of hugpy's dispatch.execute_prompt over HTTP.

        Pass exactly what you'd pass in-process: ``task`` plus whatever that
        task's builder reads (prompt / text / texts / other_texts / file /
        image_b64 / summary_mode / language / width / seed / …). Explicit
        values win; omitted ones fall to central's default-resolution chain.
        Returns the TaskResult as a dict.
        """
        body = {k: v for k, v in kwargs.items() if v is not None}
        try:
            result = await self._json("POST", "/prompt", json=body)
        except HugpyError as exc:
            # A central without the route answers 404 (bare API), 405 (the
            # SPA catch-all only serves GET), or HTML (catch-all GET).
            message = str(exc)
            if "-> 404" in message or "-> 405" in message or "non-JSON" in message:
                raise HugpyError(
                    "central does not expose /prompt yet — deploy the latest "
                    "hugpy package to enable this command"
                ) from exc
            raise
        if not result.get("ok", True) or result.get("error"):
            raise HugpyError(result.get("error") or "execute_prompt failed")
        return result

    async def supported_tasks(self) -> dict:
        """Task categories central can serve: {"tasks": [...], "defaults": {...}}."""
        return await self._json("GET", "/prompt/tasks")

    # ── F4 settings + F2 principal link ────────────────────────────────────
    async def settings_ns(self, ns: str) -> dict:
        """All values in a settings namespace (runtime SoT on central) —
        {key: value}. The bot READS these (channel modes, personalities,
        delegation); writes happen from the console."""
        out = await self._json("GET", f"/settings/{ns}")
        return out.get("values") or {}

    async def set_user_model(self, user_id: int, model_key: str | None) -> dict:
        """Persist a user's default model in central settings (M2M route,
        same trust tier as the discord outbox/inbox endpoints)."""
        return await self._json("POST", "/discord/prefs",
                                json={"user_id": str(user_id),
                                      "model_key": model_key})

    async def discord_link(self, token: str, discord_user_id: int) -> dict:
        """DISC-05: bind a Discord user to the principal behind this token."""
        return await self._json("POST", "/auth/discord-link",
                                json={"token": token,
                                      "discord_user_id": str(discord_user_id)})

    async def cancel_chat(self, request_id: str) -> dict:
        """Ask central to stop an in-flight generation (F1 control plane).

        Central cancels a locally-served stream through its job store and
        fans out to GPU workers for relayed ones — either way the model stops
        producing and the slot frees, instead of us merely closing our end of
        the SSE pipe."""
        return await self._json("POST", f"/llm/chat/cancel/{request_id}")

    # ── streaming chat ────────────────────────────────────────────────────
    async def chat_stream(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        model_key: str | None = None,
        task: str | None = None,
        file: str | None = None,
        images: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
        max_new_tokens: int | None = None,
        unbounded: bool | None = None,
        request_id: str | None = None,
        transport: str | None = None,
        channel: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield generated text chunks; raises HugpyError on an error event."""
        body: dict[str, Any] = {}
        if messages:
            body["messages"] = messages
        else:
            body["prompt"] = prompt or ""
        for key, value in (
            ("model_key", model_key),
            ("task", task),
            ("file", file),
            ("images", images),
            ("temperature", temperature),
            ("top_p", top_p),
            ("do_sample", do_sample),
            ("max_new_tokens", max_new_tokens),
            ("unbounded", unbounded),
            ("request_id", request_id),
            ("transport", transport),
            ("channel", channel),
        ):
            if value is not None:
                body[key] = value

        async with self._http.stream("POST", "/chat/stream", json=body) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")
                raise HugpyError(f"chat/stream -> {resp.status_code}: {text[:300]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "token":
                    text = event.get("text")
                    if text:
                        yield text
                elif kind == "error":
                    raise HugpyError(event.get("message") or "generation failed")
                elif kind == "done":
                    return
