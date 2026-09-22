"""DeepCoder chat runner.

Thin adapter that wraps the existing DeepCoder + REGISTRY + build_deepcoder_runtime
machinery behind the Runner protocol.

Construction is cheap — just stores the model_key. The model load happens
lazily on first .run()/.stream(), when REGISTRY.get() fires, matching
DeepCoder's 'one instance per cfg.cache_key()' caching so multiple runners
for the same model share weights.

Sync->async: DeepCoder.generate_text is sync (PyTorch inference on the
calling thread). We wrap it in asyncio.to_thread so the event loop stays
free during long generations.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

from hugpy_engine.schemas.chat_schemas import ChatRequest, ChatResult
from hugpy_engine.schemas.runner_schemas import StreamEvent
from hugpy_engine.schemas.task_schemas import TaskRequest
from hugpy_platform.except_utils import attempt
from hugpy_engine.generate.coder import REGISTRY, DeepCoder
from hugpy_engine.generate.config import (
    DeepCoderConfig,
    build_deepcoder_runtime,
    DoneEvent,
    TokenEvent,
)

logger = logging.getLogger(__name__)


class DeepCoderChatRunner:
    """Runner for transformers-based causal LMs (DeepCoder, DAN-Qwen3, etc)."""

    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, cfg, **runtime_kwargs):
        self.model_key = cfg.model_key
        self._cfg = build_deepcoder_runtime(model_key=cfg.model_key, **runtime_kwargs)
        # Not calling REGISTRY.get(self._cfg) here — that would force the
        # model load at construction. Defer to first use.

    @property
    def coder(self) -> DeepCoder:
        """Resolve the underlying DeepCoder. Loads on first access."""
        return REGISTRY.get(self._cfg)

    # --- result helpers ----------------------------------------------------

    def _error_result(self, req: ChatRequest, error: str) -> ChatResult:
        """Single construction site for ok=False results — so the failure
        shape can't drift between the two error paths."""
        return ChatResult(
            request_id=req.request_id,
            model_key=req.model_key,
            ok=False,
            error=error,
            text="",
            finish_reason="error",
        )

    # --- non-streaming -----------------------------------------------------

    async def run(self, req: ChatRequest) -> ChatResult:
        messages = [
            m.model_dump() if hasattr(m, "model_dump") else m
            for m in req.messages
        ]

        # Unbounded, non-streaming requests chain capped passes through the
        # shared run_unbounded driver (the streaming path has its own loop in
        # stream()). Without this an unbounded non-streaming request silently
        # truncates at one cap's worth of tokens.
        if getattr(req, "unbounded", False):
            return await self._run_unbounded(req, messages)

        # generate_text is sync; offload so the event loop keeps running.
        # attempt() logs the full traceback and hands back the exception so
        # we can branch on its type below.
        ok, analysis, exc = await asyncio.to_thread(
            attempt,
            self.coder.generate_analysis,
            messages,
            label=f"DeepCoderChatRunner.run model={self.model_key} req={req.request_id}",
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.do_sample,
            use_chat_template=True,
            return_full_text=False,
        )

        if ok:
            completion = analysis["completion_tokens"]
            elapsed = analysis["elapsed_s"]
            return ChatResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                text=analysis["text"],
                finish_reason=analysis["finish_reason"],
                output_chunks=completion,
                usage={"prompt_tokens": analysis["input_tokens"],
                       "completion_tokens": completion,
                       "total_tokens": analysis["input_tokens"] + completion},
                timings={"predicted_n": completion,
                         "predicted_ms": elapsed * 1000.0,
                         "predicted_per_second": analysis["tok_s"],
                         "measurement_source": "transformers_generation_wall"},
            )

        # _resolve_max_new_tokens raises ValueError on cap violation — a
        # request-side problem, not a model failure. attempt() already
        # logged it at exception level; downgrade the *meaning* here.
        if isinstance(exc, ValueError):
            logger.warning("DeepCoderChatRunner.run rejected: %s", exc)
            return self._error_result(req, str(exc))

        return self._error_result(req, f"{type(exc).__name__}: {exc}")

    async def _run_unbounded(self, req: ChatRequest, messages: list) -> ChatResult:
        """Non-streaming auto-continuation via run_unbounded.

        Drives coder.generate_once one pass at a time; run_unbounded appends a
        'continue' nudge whenever a pass ends with finish_reason 'length' and
        stops on natural completion or after max_chunks. The non-streaming twin
        of stream()'s unbounded loop, so both transports continue past the cap.
        """
        # Lazy import keeps this cross-manager dependency off module-load time
        # (no import cycle with chat_context) and the helper is cheap to import.
        from hugpy_engine.chat_context.unbounded import run_unbounded, GenerationOutcome, map_finish_reason

        max_chunks = req.max_chunks
        if not max_chunks:
            # Same high-but-bounded ceiling as the streaming path.
            try:
                max_chunks = int(os.environ.get("HUGPY_MAX_CHUNKS", "256"))
            except ValueError:
                max_chunks = 256

        # Per-pass budget; the coder clamps it to its own max_new_tokens_cap.
        chunk_tokens = req.max_new_tokens or 1024

        def _generate_once(convo: list, cap: int) -> "GenerationOutcome":
            text, finish = self.coder.generate_once(
                convo,
                max_new_tokens=cap,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
            )
            return GenerationOutcome(text=text, finish_reason=finish)

        # run_unbounded is sync (it drives the sync generate_once); offload the
        # whole multi-pass run so the event loop stays free. attempt() logs the
        # traceback and hands back the exception for type-branching below.
        ok, outcome, exc = await asyncio.to_thread(
            attempt,
            run_unbounded,
            _generate_once,
            messages,
            label=f"DeepCoderChatRunner._run_unbounded model={self.model_key} req={req.request_id}",
            chunk_tokens=chunk_tokens,
            max_chunks=max_chunks,
        )

        if ok:
            return ChatResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                text=outcome.text,
                # 'length' -> 'max_tokens' so a continuation that still hit the
                # chunk ceiling reports truncation in the OpenAI vocabulary.
                finish_reason=map_finish_reason(outcome.finish_reason),
                usage=outcome.usage,
            )

        if isinstance(exc, ValueError):
            logger.warning("DeepCoderChatRunner._run_unbounded rejected: %s", exc)
            return self._error_result(req, str(exc))

        return self._error_result(req, f"{type(exc).__name__}: {exc}")

    # --- streaming ---------------------------------------------------------

    async def stream(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Delegate to DeepCoder.stream_chat — it already implements the
        StreamEvent protocol. Pass events through unwrapped.

        The route layer SSE-encodes (`data: {...}\\n\\n`); runners just
        yield typed events.

        Bounded requests are a straight passthrough. Unbounded requests chain
        capped passes (mirroring the llama runner's stream_chat_unbounded):
        when a pass ends with finish_reason='max_tokens', the partial answer
        plus a 'continue' nudge is appended to the conversation and another
        pass streams. Intermediate DoneEvents are suppressed so the SSE
        consumer sees one logical stream with a single terminal event.

        Not wrapped in attempt(): this is a generator, and attempt() runs a
        callable to completion — it can't drive a stream. Errors mid-stream
        belong in DeepCoder.stream_chat as ErrorEvents, not swallowed here.
        """
        if not getattr(req, "unbounded", False):
            async for event in self.coder.stream_chat(req, cancel_event=cancel_event):
                yield event
            return

        max_chunks = req.max_chunks
        if not max_chunks:
            # High ceiling so "unbounded" really means until-the-model-stops,
            # still bounded so a looping model can't run forever.
            try:
                max_chunks = int(os.environ.get("HUGPY_MAX_CHUNKS", "256"))
            except ValueError:
                max_chunks = 256

        convo = [
            m.model_dump() if hasattr(m, "model_dump") else dict(m)
            for m in req.messages
        ]
        total_chunks = 0
        last_done: Optional[DoneEvent] = None

        for _ in range(max_chunks):
            piece_text = ""
            pass_done: Optional[DoneEvent] = None
            pass_req = req.model_copy(
                update={"messages": convo, "unbounded": False},
            )

            async for event in self.coder.stream_chat(pass_req, cancel_event=cancel_event):
                if isinstance(event, TokenEvent):
                    piece_text += event.text or ""
                    total_chunks += 1
                    yield event
                elif isinstance(event, DoneEvent):
                    pass_done = event
                else:
                    # ErrorEvent: terminal — surface it and end the stream.
                    yield event
                    return

            last_done = pass_done
            if (
                pass_done is None
                or pass_done.finish_reason != "max_tokens"
                or not piece_text
            ):
                break

            convo = convo + [
                {"role": "assistant", "content": piece_text},
                {"role": "user", "content": "continue"},
            ]

        yield DoneEvent(
            request_id=req.request_id,
            input_tokens=getattr(last_done, "input_tokens", 0) or 0,
            output_chunks=total_chunks,
            finish_reason=getattr(last_done, "finish_reason", None) or "stop",
        )
