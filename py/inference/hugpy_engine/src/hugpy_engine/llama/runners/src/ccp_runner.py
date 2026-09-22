from hugpy_engine.llama.runners.src.base_runner import LlamaCppBaseRunner
import json
import httpx
from typing import Optional
from hugpy_engine.llama.runners.src.imports.constants import DEFAULT_HTTP_TIMEOUT
from hugpy_engine.llama.runners.src.imports.init_imports import logger
from hugpy_engine.llama.runners.src.imports.utils import messages_to_prompt_from_dicts
from hugpy_engine.llama.serve import serve_endpoint, serve_model_name

def __init__(self, model_key, *, env_path=None):
    base = serve_endpoint(model_key)          # None for mode=off
    if not base:
        raise RuntimeError(f"{model_key}: no HTTP endpoint (mode=off) — use in-process")
    self.model_key = model_key
    self.base_url = base
    self.served_model = serve_model_name(model_key)
# ===========================================================================
# HTTP runner — talks to a running llama-server process
# ===========================================================================


def _model_is_vision(model_key: str) -> bool:
    """True for an image-text-to-text GGUF — so the base runner folds the image
    into the request (_attach_image) before forwarding to the native
    llama-server --mmproj, which DOES understand image_url content. Without this
    the HTTP runner sends text-only and the server is blind to the image."""
    try:
        from hugpy_engine.config.main import get_model_config
        cfg = get_model_config(model_key)
    except Exception:
        return False
    if getattr(cfg, "primary_task", None) == "image-text-to-text":
        return True
    return "image-text-to-text" in (getattr(cfg, "tasks", None) or [])


def _reinline_reasoning(content, reasoning):
    """Recent llama-server (``--jinja`` on + ``--reasoning auto``, the default in
    builds since ~mid-2026) EXTRACTS a reasoning model's ``<think>…</think>`` into
    ``message.reasoning_content`` and STRIPS it from ``content`` — so a client that
    reads only ``content`` (as this runner did) silently loses the whole thought,
    and ``content`` is even empty when a reply is cut off mid-reasoning. Re-inline
    the reasoning as ``<think>…</think>`` so it surfaces downstream exactly as it
    did before the engine started splitting it, without double-wrapping when a
    server (older, or ``--reasoning-format none``) already inlined it."""
    content = content or ""
    reasoning = reasoning or ""
    if reasoning and "<think>" not in content:
        return f"<think>{reasoning}</think>{content}"
    return content


class LlamaCppRunner(LlamaCppBaseRunner):
    def __init__(self, model_key: str, *, env_path: Optional[str] = None,
                 base_url: Optional[str] = None):
        # Vision GGUFs: fold the image into the request (gated on is_vision in
        # _attach_image). The native llama-server --mmproj sees image_url content.
        self.is_vision = _model_is_vision(model_key)
        # Explicit base_url: a managed server we were handed (e.g. a shard
        # lead from ensure_shard_server) rather than the serve layer's.
        if base_url:
            self.model_key = model_key
            self.base_url = base_url.rstrip("/")
            self.served_model = model_key
            self._slot_backed = True   # STALE-SLOT-FIX-20260910
            return
        self._slot_backed = False      # STALE-SLOT-FIX-20260910

        base = serve_endpoint(model_key)          # None for mode=off
        if not base:
            raise RuntimeError(f"{model_key}: no HTTP endpoint (mode=off) — use in-process")
        self.model_key = model_key
        self.base_url = base
        self.served_model = serve_model_name(model_key)

##        cfg = load_llama_config(env_path=env_path)
##        self.model_key = model_key
##        self.llama_host: str = cfg["LLAMA_HOST"]
##        self.port: int = cfg[model_key]
##        self.base_url = f"{self.llama_host}:{self.port}"

    def _refresh_endpoint(self) -> bool:
        """Re-resolve the serving endpoint after a stale-slot failure.

        This runner instance is CACHED by dispatch, so its base_url can
        outlive the slot it points at (the slot self-heals/unloads/TTLs the
        model, the agent restarts, …) — the proxy then 503s "no model
        loaded" forever while the cached instance keeps knocking. serve
        resolution is authoritative and side-effecting (it loads the model
        into a free slot and waits healthy), so one refresh + retry turns a
        permanently-broken cached runner into a slow first request."""
        base = None
        if getattr(self, "_slot_backed", False):
            # STALE-SLOT-FIX-20260910: this runner was handed a slot endpoint; re-resolve through
            # the slot pool (loads into a free seat / evicts on demand), not the
            # serve layer, which knows nothing about slots.
            try:
                from hugpy_engine.serve.slots import SlotPool
                base = SlotPool().endpoint_for(self.model_key)
            except Exception:  # noqa: BLE001
                base = None
        if not base:
            try:
                base = serve_endpoint(self.model_key)
            except Exception:
                return False
        if not base:
            return False
        # SAME URL is still a successful refresh: serve resolution is
        # side-effecting — it (re)loads the model into the slot and WAITS for
        # it to go healthy — so retrying the identical endpoint after it
        # returns is exactly right (post-reexec cold window, mid-reload).
        self.base_url = base.rstrip("/")
        self.served_model = serve_model_name(self.model_key)
        return True

    async def _iter_stream(self, messages, max_tokens, temp, top_p, extras=None):
        # Mild anti-repetition on the streaming path (llama.cpp sampling
        # extension accepted by llama-server's OpenAI-compatible endpoint;
        # harmless if a given build ignores it). Complements the loop-guard in
        # base_runner.stream_chat_unbounded — this discourages the repetition,
        # the guard stops it if it starts anyway.
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": True,
                   "repeat_penalty": 1.1,
                   "model": self.model_key}   # STALE-SLOT-FIX-20260910: lets the slot refuse a mismatch
        if extras:
            # t74: chat_template_kwargs / logit_bias, verbatim body keys —
            # llama-server applies chat_template_kwargs at template render
            # (--jinja builds; e.g. enable_thinking:false pre-closes <think>)
            # and logit_bias at sampling. Older builds ignore unknown keys.
            payload.update(extras)
        for attempt in (1, 2):
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                             json=payload) as response:
                        response.raise_for_status()
                        in_reasoning = False
                        async for line in response.aiter_lines():
                            if not line or line.strip() == "[DONE]":
                                continue
                            line = line.removeprefix("data: ")
                            try:
                                data = json.loads(line)
                                # Recent llama-server sends a final chunk with
                                # real token counts; stash it for the DoneEvent
                                # (base_runner._take_stream_usage). Absent on
                                # older builds -> usage stays None downstream.
                                if data.get("usage"):
                                    self._stream_usage = data["usage"]
                                # STREAMING DOES carry the engine's own decode
                                # rate — verified against llama.cpp
                                # tools/server/server-task.cpp
                                # (to_json_oaicompat_chat_stream): the FINAL
                                # chunk gets `deltas.back().push_back({"timings",
                                # ...})` unconditionally, independent of
                                # `include_usage`. So the same take-once slot
                                # works for both transports; absent on an older
                                # build -> stays None and nothing is recorded.
                                if isinstance(data.get("timings"), dict):
                                    self._stream_timings = data["timings"]
                                choice = data["choices"][0]
                                delta = choice.get("delta") or {}
                                fr    = choice.get("finish_reason")
                                # Re-inline the engine-extracted reasoning stream as
                                # <think>…</think>: open on the first reasoning delta,
                                # close when content begins (or at finish) — so a
                                # reasoning model streams its thoughts as it used to.
                                rc = delta.get("reasoning_content") or ""
                                ct = delta.get("content") or ""
                                text = ""
                                if rc:
                                    if not in_reasoning:
                                        text += "<think>"
                                        in_reasoning = True
                                    text += rc
                                if ct:
                                    if in_reasoning:
                                        text += "</think>"
                                        in_reasoning = False
                                    text += ct
                                if fr and in_reasoning:
                                    text += "</think>"
                                    in_reasoning = False
                            except Exception:
                                text, fr = "", None
                            yield text, fr
                return
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 1 and (status in (409, 500, 502, 503) or isinstance(exc, httpx.ConnectError)):  # STALE-SLOT-FIX-20260910: 409 = slot holds another model
                    # Stale cached endpoint (slot unloaded/agent restarted):
                    # re-resolve — serve_endpoint reloads the model — and retry once.
                    # 500 belongs here too (2026-07-29): a single-slot box that
                    # SWAPPED to a different model keeps answering 500 (e.g.
                    # "image input is not supported … mmproj" when a text model
                    # replaced a vision one) — the child is up, so only a
                    # re-resolve (which reloads OUR model into a slot) recovers.
                    if self._refresh_endpoint():
                        continue
                raise
    def _chat_complete(self, messages, max_tokens, temp, top_p, stop, extras=None):
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": False,
                   "model": self.model_key}   # STALE-SLOT-FIX-20260910
        if stop:
            payload["stop"] = stop
        if extras:
            payload.update(extras)      # t74 body keys — see _iter_stream
        for attempt in (1, 2):
            try:
                with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                    r = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                    r.raise_for_status()
                    data = r.json()
                break
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 1 and (status in (409, 500, 502, 503) or isinstance(exc, httpx.ConnectError)):  # STALE-SLOT-FIX-20260910
                    if self._refresh_endpoint():   # stale/swapped cached endpoint — see _iter_stream
                        continue
                raise
        # Non-streaming twin of the capture in _iter_stream: llama-server puts
        # `timings` on the response body of EVERY /v1/chat/completions call.
        # Stashed in the same take-once slot so both transports feed one reader.
        if isinstance(data.get("timings"), dict):
            self._stream_timings = data["timings"]
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        return (_reinline_reasoning(msg.get("content"), msg.get("reasoning_content")),
                choice.get("finish_reason") or "stop")

    def _raw_complete(self, prompt, max_tokens, temp, top_p, stop, return_full_text):
        payload = {"prompt": prompt, "n_predict": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": False}
        if stop:
            payload["stop"] = stop
        with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
            r = client.post(f"{self.base_url}/completion", json=payload)
            r.raise_for_status()
            data = r.json()
        text = data.get("content") or data.get("text") or ""
        if return_full_text:
            text = prompt + text
        finish = data.get("stop_type") or data.get("finish_reason") or "stop"
        return text, finish
    def _blocking_complete(
        self,
        messages: list[dict] | str,
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        use_chat_template: bool,
        return_full_text: bool,
        extras: Optional[dict] = None,
    ) -> tuple[str, str]:
        timeout = DEFAULT_HTTP_TIMEOUT

        if use_chat_template and isinstance(messages, list):
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": top_p,
                "stream": False,
            }

            if stop:
                payload["stop"] = stop
            if extras:
                payload.update(extras)  # t74 body keys — see _iter_stream

            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            if isinstance(data.get("timings"), dict):
                self._stream_timings = data["timings"]   # see _chat_complete
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            text = _reinline_reasoning(msg.get("content"), msg.get("reasoning_content"))
            finish = choice.get("finish_reason") or "stop"

            return text, finish

        # Raw-prompt fallback: llama-server /completion endpoint
        if extras:
            # /completion renders no chat template — the t74 keys cannot apply.
            # Say so rather than silently dropping a selected suppression; the
            # /no_think directive in the prompt text still rides.
            logger.info("raw-prompt path cannot apply engine chat extras %s "
                        "for %s — relying on the /no_think directive",
                        sorted(extras), self.model_key)
        prompt = (
            messages
            if isinstance(messages, str)
            else messages_to_prompt_from_dicts(messages)
        )

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url}/completion",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = data.get("content") or data.get("text") or ""

        if return_full_text:
            text = prompt + text

        finish = data.get("stop_type") or data.get("finish_reason") or "stop"

        return text, finish
