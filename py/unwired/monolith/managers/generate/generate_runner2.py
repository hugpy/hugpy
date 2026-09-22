"""DeepCoder chat runner.

Thin adapter that wraps the existing DeepCoder + REGISTRY + build_deepcoder_runtime
machinery (in deepcoder/coder.py and deepcoder/config.py) and exposes them
behind the Runner protocol.

Construction is cheap — just stores the model_key. The actual model load
happens lazily on first .run() / .stream() call, which is when REGISTRY.get()
fires. That matches DeepCoder's existing 'one instance per cfg.cache_key()'
caching, so multiple runners for the same model still share weights.

Sync->async: DeepCoder.generate_text is sync (it does PyTorch inference on
the calling thread). We wrap it in asyncio.to_thread so the FastAPI event
loop stays free during long generations. Without this, one /generate call
blocks every other request the worker is serving — including SSE heartbeats
on /chat for other clients.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from hugpy_engine.generate.imports import TaskRequest, ChatRequest, ChatResult, StreamEvent

# These imports go through the existing module layout. Adjust the dotted
# paths to match wherever you wire this file in — the runner doesn't care
# about the path, only that these names resolve.
from hugpy_engine.generate.coder import REGISTRY, DeepCoder
from hugpy_engine.generate.config import DeepCoderConfig, build_deepcoder_runtime

logger = logging.getLogger(__name__)


class DeepCoderChatRunner:
    """Runner for transformers-based causal LMs (DeepCoder, DAN-Qwen3, etc).

    The model_key -> DeepCoderConfig translation happens once in __init__
    so the cache_key is stable and REGISTRY.get() returns the same DeepCoder
    instance across requests.
    """

    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, cfg, **runtime_kwargs):
        self.model_key = cfg.model_key
        self._cfg = build_deepcoder_runtime(model_key=cfg.model_key, **runtime_kwargs)
        # Note: not calling REGISTRY.get(self._cfg) here — that would force
        # the model load at runner construction time. Defer to first use.

    @property
    def coder(self) -> DeepCoder:
        """Resolve the underlying DeepCoder instance. Loads on first access."""
        return REGISTRY.get(self._cfg)

    def ensure_loaded(self) -> None:
        """Force the weights RESIDENT now — the load a first request would trigger.
        Building this runner is lazy (see __init__), so a static/eager warm that
        only calls runner_for() gets a hollow shell that occupies no VRAM/RAM and
        reads as loaded-but-not-loaded. Static means 'live in the resources', so
        the warm path calls this to actually materialize the model."""
        _ = self.coder            # REGISTRY.get -> DeepCoder(...) -> _load_model

    # --- non-streaming -----------------------------------------------------

    async def run(self, req: ChatRequest) -> ChatResult:
        try:
            # Convert pydantic ChatMessage -> dict for apply_chat_template
            messages = [
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in req.messages
            ]
            # generate_text is sync; offload so the event loop keeps running.
            text = await asyncio.to_thread(
                self.coder.generate_text,
                messages,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                use_chat_template=True,
                return_full_text=False,
            )

            return ChatResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                text=text,
                # generate_text doesn't currently surface a finish_reason.
                # 'stop' is the honest default; if we hit max_new_tokens
                # the caller can detect it from output length.
                finish_reason="stop",
            )

        except ValueError as exc:
            # _resolve_max_new_tokens raises ValueError on cap violation.
            # That's a request-side problem, not a model failure.
            logger.warning("DeepCoderChatRunner.run rejected: %s", exc)
            return ChatResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=False, error=str(exc),
                text="", finish_reason="error",
            )

        except Exception as exc:
            logger.exception(
                "DeepCoderChatRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            return ChatResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=False, error=f"{type(exc).__name__}: {exc}",
                text="", finish_reason="error",
            )

    # --- streaming ---------------------------------------------------------

    async def stream(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Delegate to DeepCoder.stream_chat — it already implements the
        StreamEvent protocol. We don't re-wrap the events; pass them through.

        The route layer is responsible for SSE-encoding (`data: {...}\\n\\n`).
        Runners just yield typed events.
        """
        # DeepCoder.stream_chat already accepts the project's ChatRequest
        # type (from .imports), which has the same fields as ours. If the
        # two ChatRequest classes ever diverge, this is the line that
        # breaks first — and that's a feature, not a bug.
        async for event in self.coder.stream_chat(req, cancel_event=cancel_event):
            yield event
