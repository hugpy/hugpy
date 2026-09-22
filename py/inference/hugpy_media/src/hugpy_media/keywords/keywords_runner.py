"""Keyword-extraction runner.

Serves ("transformers", "keyword-extraction"). KeyBERT rides any
sentence-transformers model, so this task registers on the embedding
models and shares their weights conceptually — but KeyBERT wraps the
sentence-BERT in its own object, so the runner keeps a per-model_key
KeyBERT cache (class-level singleton, same pattern as
FeatureExtractionRunner._MODELS).

keybert/spacy are imported lazily inside the extraction call (via
keybert_model's require()), so importing this module costs nothing.
The spaCy backend is best-effort: if only one backend is installed the
result still comes back, with backend_errors saying what was skipped.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict

from hugpy_engine.schemas.event_schemas import DoneEvent, ErrorEvent, TokenEvent
from hugpy_storage.download_models import ensure_model

from hugpy_media.keywords.keybert_model import _build_sentence_bert, refine_keywords, require
from hugpy_media.keywords.schemas import KeywordTaskRequest, KeywordTaskResult

logger = logging.getLogger(__name__)


def _render_text(result: KeywordTaskResult) -> str:
    """One-line-per-section summary for chat-stream degradation."""
    parts = []
    if result.primary:
        parts.append("primary: " + ", ".join(result.primary))
    if result.secondary:
        parts.append("secondary: " + ", ".join(result.secondary))
    if result.hashtags:
        parts.append("hashtags: " + " ".join(result.hashtags))
    if result.slug_candidates:
        parts.append("slugs: " + ", ".join(result.slug_candidates))
    return "\n".join(parts) or "no keywords extracted"


class KeywordRunner:
    request_type = KeywordTaskRequest
    result_type = KeywordTaskResult

    # Class-level so all instances of this runner share one cache.
    _MODELS: Dict[str, Any] = {}
    _LOCK = threading.Lock()

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        self._runtime_kwargs = runtime_kwargs

    # --- model loading (lazy, singleton) -----------------------------------

    @property
    def keybert(self):
        cached = self._MODELS.get(self.model_key)
        if cached is not None:
            return cached

        with self._LOCK:
            cached = self._MODELS.get(self.model_key)
            if cached is not None:
                return cached

            KeyBERT = require("keybert", reason="needed by keyword-extraction").KeyBERT
            model_dir = ensure_model(self.model_key)
            instance = KeyBERT(_build_sentence_bert(model_path=model_dir))

            logger.info(
                "KeywordRunner: loaded model=%s dir=%s", self.model_key, model_dir,
            )
            self._MODELS[self.model_key] = instance
            return instance

    # --- extraction ---------------------------------------------------------

    def _extract(self, req: KeywordTaskRequest):
        """Blocking extract. Called from a worker thread by .run()."""
        kwargs: Dict[str, Any] = {"model": self.keybert}
        for field in ("top_n", "diversity", "use_mmr", "stop_words",
                      "keyphrase_ngram_range"):
            value = getattr(req, field)
            if value is not None:
                kwargs[field] = value
        if req.refine:
            for field in ("min_density", "max_density", "min_score",
                          "max_words_per_phrase"):
                value = getattr(req, field)
                if value is not None:
                    kwargs[field] = value
            return refine_keywords(req.text, preset=req.preset or "seo", **kwargs)

        from hugpy_media.keywords.keybert_model import extract_keywords
        return extract_keywords(req.text, preset=req.preset, **kwargs)

    # --- public API ---------------------------------------------------------

    async def run(self, req: KeywordTaskRequest) -> KeywordTaskResult:
        try:
            extracted = await asyncio.to_thread(self._extract, req)

            if req.refine:
                # RefinedResult — raw KeywordResult nested under .raw
                result = KeywordTaskResult(
                    request_id=req.request_id,
                    model_key=req.model_key,
                    ok=True,
                    preset_used=extracted.preset_used,
                    primary=extracted.primary,
                    secondary=extracted.secondary,
                    dropped=extracted.dropped,
                    combined=extracted.raw.combined,
                    density=extracted.density,
                    density_flags=extracted.density_flags,
                    meta_keywords=extracted.meta_keywords,
                    hashtags=extracted.hashtags,
                    slug_candidates=extracted.slug_candidates,
                    backends_used=extracted.raw.backends_used,
                    backend_errors=extracted.raw.backend_errors,
                )
            else:
                # raw KeywordResult — merged extraction, no post-processing
                result = KeywordTaskResult(
                    request_id=req.request_id,
                    model_key=req.model_key,
                    ok=True,
                    preset_used=req.preset,
                    primary=extracted.combined,
                    combined=extracted.combined,
                    density=extracted.density,
                    backends_used=extracted.backends_used,
                    backend_errors=extracted.backend_errors,
                )
            return result.model_copy(update={"text": _render_text(result)})

        except Exception as exc:
            logger.exception(
                "KeywordRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            return KeywordTaskResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(self, req: KeywordTaskRequest, cancel_event=None):
        """One-shot wrapped as a stream, mirroring VisionRunner."""
        result = await self.run(req)
        if result.ok:
            yield TokenEvent(request_id=req.request_id, text=result.text)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                            output_chunks=1, finish_reason="stop")
        else:
            yield ErrorEvent(request_id=req.request_id,
                             message=result.error or "keyword extraction failed")
