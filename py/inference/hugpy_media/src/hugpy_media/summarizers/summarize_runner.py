import asyncio
import logging

from hugpy_engine.schemas.event_schemas import DoneEvent, ErrorEvent, TokenEvent

from hugpy_media.schemas.summarizer_schemas import InputPolicy, SummarizeRequest, SummarizeResult
from hugpy_media.summarizers.summarizers import summarize

logger = logging.getLogger(__name__)


# runners/summarize_runner.py
class SummarizeRunner:
    """Drives summarization for a single (model, task).

    Instantiated by dispatch as `runner_cls(res.cfg)` — same as the embed and
    vision runners — so it reads model_key off the cfg. The cfg's task selects
    the backend *strategy*: the HF "summarization" pipeline for summarization
    models, generate()-based chunk+consolidate for text2text models. The two
    are different axes — which weights (model_key) vs which strategy (backend) —
    and conflating them was the old TypeError-on-first-call bug.
    """

    request_type = SummarizeRequest
    result_type = SummarizeResult

    # cfg.primary_task -> summarizers backend strategy key
    _STRATEGY_BY_TASK = {
        "summarization": "pipeline_chunked",
        "text2text-generation": "seq2seq_chunked",
    }

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_key = cfg.model_key
        self.backend = self._STRATEGY_BY_TASK.get(
            getattr(cfg, "primary_task", None), "seq2seq_chunked"
        )

    async def run(self, req):
        try:
            input_policy = (
                InputPolicy(req.input_policy) if req.input_policy else None
            )
            summary = await asyncio.to_thread(
                summarize,
                req.text,
                backend=self.backend,
                model_key=self.model_key,
                preset=req.preset,
                summary_mode=req.summary_mode,
                input_policy=input_policy,
                max_chunk_tokens=req.max_chunk_tokens,
                min_length=req.min_length,
                max_length=req.max_length,
                do_sample=req.do_sample,
                min_input_words=req.min_input_words,
                consolidation_min_length=req.consolidation_min_length,
                consolidation_max_length=req.consolidation_max_length,
                max_output_words=req.max_output_words,
            )
            return SummarizeResult(
                request_id=req.request_id,
                model_key=req.model_key,
                backend=self.backend,
                preset_used=req.preset,
                text=summary,
                chunks_processed=1,
            )
        except Exception as exc:
            logger.exception(
                "SummarizeRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            return SummarizeResult(
                request_id=req.request_id,
                model_key=req.model_key,
                backend=self.backend,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                finish_reason="error",
            )

    async def stream(self, req, cancel_event=None):
        result = await self.run(req)
        if getattr(result, "ok", True):
            yield TokenEvent(request_id=req.request_id, text=result.text or "")
            yield DoneEvent(
                request_id=req.request_id,
                input_tokens=0,
                output_chunks=1,
                finish_reason=getattr(result, "finish_reason", None) or "stop",
            )
        else:
            yield ErrorEvent(
                request_id=req.request_id,
                message=result.error or "summarization failed",
            )
