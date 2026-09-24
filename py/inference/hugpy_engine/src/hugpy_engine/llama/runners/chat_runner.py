"""Llama.cpp chat runner.

Adapter over the existing LlamaCppPythonRunner / LlamaCppRunner (HTTP).
Both are kept in their per-process singleton cache (get_llama_runner),
so the heavy GGUF load happens once per model_key regardless of how
many adapter wrappers exist.

Constructor signature is uniform with the other runners in _RUNNERS:
    __init__(self, cfg: ModelConfig, **runtime_kwargs)

This matches what dispatch._get_or_build_runner expects, so the
dispatch table can stay a flat (framework, task) -> class mapping
without any per-runner adapter logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from hugpy_engine.schemas.chat_schemas import ChatRequest, ChatResult
from hugpy_engine.schemas.event_schemas import ErrorEvent
from hugpy_engine.schemas.runner_schemas import StreamEvent
from hugpy_platform.constants import DEFAULT_MAX_TOKENS
from hugpy_engine.llama.runners.get import get_llama_runner, evict_llama_runner

logger = logging.getLogger(__name__)


# ── Stale-slot self-heal (operator, 2026-09-10) ─────────────────────────────
# A slot-backed runner whose seat was LRU-evicted under it keeps its old slot
# endpoint and fails EVERY call with the child's 503 ("slot N has no model
# loaded") — the worker never re-seats because the cached runner short-circuits
# get_llama_runner. That inverted the whole point of the fleet ("have a model
# called, and have it find a worker and load" — the computron/flux2 wedge):
# the model sat on disk while calls died forever until an agent restart.
# The heal: on that signature, evict the cached runner and re-acquire ONCE —
# re-acquisition runs SlotPool.endpoint_for, which seats AND LOADS, so the
# SAME call completes. Anything else re-raises untouched.
_STALE_SLOT_MARKERS = ("503", "no model loaded")


def _stale_slot_error(exc_or_msg) -> bool:
    msg = str(exc_or_msg)
    return any(m in msg for m in _STALE_SLOT_MARKERS)


class LlamaCppChatRunner:
    """Runner for GGUF models loaded in-process via llama_cpp or via HTTP.

    Uses the get_llama_runner() singleton cache so multiple adapter
    wrappers for the same model_key share a single underlying runner
    (which itself holds the loaded GGUF + KV cache + generate_lock).
    """

    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        # model_key is whatever the registry uses as its key; ModelConfig
        # exposes it as model_key (set by get_models_dict in models_config).
        self.model_key = cfg.model_key
        # **runtime_kwargs is accepted to keep the uniform _RUNNERS constructor
        # signature, but deliberately NOT stored or applied: the underlying GGUF
        # runner is a per-model_key singleton (get_llama_runner), so per-call
        # n_ctx/n_threads overrides can't be honored without forcing a second
        # load. GPU/context placement is resolved once, from env, in spill.py.
        if runtime_kwargs:
            logger.debug("LlamaCppChatRunner ignoring runtime_kwargs %s for %s "
                         "(singleton runner; placement comes from spill.py)",
                         sorted(runtime_kwargs), self.model_key)

    @property
    def runner(self):
        # Lazy resolution. First access triggers the GGUF load (which can
        # take seconds for a 14B model); subsequent accesses are dict lookups.
        return get_llama_runner(self.model_key)

    def ensure_loaded(self):
        """Force the underlying GGUF runner to MATERIALIZE now.

        __init__ and runner_for() build only this lazy wrapper — the heavy
        runner (which seats a llama-server slot via get_llama_runner ->
        _build_runner -> SlotPool.endpoint_for, or loads in-process) is
        resolved on first .runner access. Warm / slot-fill / probe paths call
        this so the model actually becomes resident + slot-seated instead of a
        hollow shell that still registers as "loaded". Idempotent — the heavy
        runner is a per-model_key singleton (get_llama_runner cache)."""
        return self.runner

    # --- non-streaming -----------------------------------------------------

    async def run(self, req) -> ChatResult:
        req = ChatRequest.coerce(req, model_key=self.model_key)
        runner = self.runner
        messages = [
            m.model_dump() if hasattr(m, "model_dump") else m
            for m in req.messages
        ]

        # Vision GGUFs: fold the image into the latest user turn as an image_url
        # part. _attach_image was only wired into the STREAMING path, so the
        # non-streaming run() that /ml/vision uses silently dropped the image —
        # the model saw "describe this image" with no image and asked for one.
        has_image = bool(getattr(req, "file", None) or getattr(req, "images", None)) \
            and getattr(runner, "is_vision", False)
        if has_image:
            messages = runner._attach_image(messages, req)

        # t74 hard no-think: per-request engine chat keys ride the request to
        # the chat-completion body (the streaming path reads them off req in
        # base_runner._engine_extras; this one-shot path threads them by kwarg).
        _extras_kw = {k: v for k, v in
                      (("chat_template_kwargs", req.chat_template_kwargs),
                       ("logit_bias", req.logit_bias))
                      if isinstance(v, dict) and v}

        async def _generate(r):
            if req.unbounded and not has_image:
                return await r.generate_text_async(
                    messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    do_sample=req.do_sample,
                    **_extras_kw,
                )
            # Image turns MUST go through the chat-template/chat-completion path
            # so the multimodal handler sees the image_url parts; the raw-prompt
            # path (used by unbounded text) flattens messages and drops them.
            return await r.generate_text_async(
                messages,
                max_new_tokens=req.max_new_tokens or 512,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                use_chat_template=True,
                return_full_text=False,
                **_extras_kw,
            )

        try:
            text = await _generate(runner)
        except Exception as exc:
            if not _stale_slot_error(exc):
                raise
            # Stale slot seat (see module note): drop the cached runner and
            # re-acquire — the rebuild re-seats AND loads, this call completes.
            logger.warning("stale slot seat for %s (%s) — evicting cached "
                           "runner and re-seating", self.model_key,
                           str(exc))
            evict_llama_runner(self.model_key)
            runner = self.runner
            text = await _generate(runner)

        # The engine's OWN measured decode rate for this one-shot, when the
        # engine reported one (operator 2026-07-25, "maximizing tok/s").
        # llama-server puts a `timings` block on every completion body; the
        # runner stashed it in its take-once slot. TaskResult is extra="allow",
        # so this rides back to central as an extra field with no wire version
        # bump and no schema change — an older central simply ignores it.
        # Absent (non-llama.cpp runner, older server build) -> omitted entirely
        # rather than sent as None, so "no measurement" stays distinguishable.
        # ⚠ RECORDING ONLY — nothing ranks on it yet.
        _timings = None
        try:
            _take = getattr(runner, "_take_stream_timings", None)
            if callable(_take):
                _timings = _take()
        except Exception:  # noqa: BLE001 — never fail a served reply over metrics
            _timings = None
        _extra = {"timings": _timings} if isinstance(_timings, dict) and _timings else {}
        return ChatResult(
            request_id=req.request_id,
            model_key=req.model_key,
            ok=True,
            text=text,
            finish_reason="stop",
            **_extra,
        )

    # --- streaming ---------------------------------------------------------

    async def stream(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Pick stream_chat or stream_chat_unbounded based on req.unbounded.

        Both methods already conform to the StreamEvent contract
        (TokenEvent stream + one terminal DoneEvent/ErrorEvent), so the
        adapter is a straight passthrough.
        """
        # Per-request loop bounds (bug B): the unbounded continue-loop used to
        # drop the caller's caps entirely. Mirror the DeepCoder path in
        # generate_runner._run/stream unbounded, which threads max_chunks and uses
        # `req.max_new_tokens or 1024` as the per-pass chunk budget. ChatRequest
        # here differs from that mental model in one way that matters: its
        # max_new_tokens is NEVER None — it defaults to DEFAULT_MAX_TOKENS — so a
        # literal `req.max_new_tokens or 1024` would silently change the historical
        # default per-pass budget from 1024 to 32768. To honor the "preserve the
        # unbounded default; only make the caps honorable when a caller SETS them"
        # rule, treat the schema default as "unset" (keep the runner's own 1024)
        # and honor any explicit, non-default value. req.max_chunks is genuinely
        # Optional[None] -> pass through; None lets base_runner apply its
        # HUGPY_MAX_CHUNKS ceiling (256), so the default is unchanged.
        if req.max_new_tokens and req.max_new_tokens != DEFAULT_MAX_TOKENS:
            chunk_tokens = req.max_new_tokens
        else:
            chunk_tokens = 1024

        def _make(runner):
            return (
                runner.stream_chat_unbounded(
                    req,
                    cancel_event=cancel_event,
                    chunk_tokens=chunk_tokens,
                    max_chunks=req.max_chunks,
                )
                if req.unbounded
                else runner.stream_chat(req, cancel_event=cancel_event)
            )

        # Stale-slot self-heal (module note): base_runner swallows the failure
        # into a terminal ErrorEvent, so detect it on the FIRST event — nothing
        # has streamed yet — evict the cached runner and retry the stream once
        # against a freshly re-seated (and therefore loaded) runner.
        first = True
        async for event in _make(self.runner):
            if (first and isinstance(event, ErrorEvent)
                    and _stale_slot_error(getattr(event, "message", ""))):
                logger.warning("stale slot seat for %s (%s) — evicting cached "
                               "runner and re-seating the stream",
                               self.model_key,
                               str(getattr(event, "message", "")))
                evict_llama_runner(self.model_key)
                async for retry_event in _make(self.runner):
                    yield retry_event
                return
            first = False
            yield event
