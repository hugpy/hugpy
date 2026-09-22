"""Feature-extraction runner.

Serves both ("transformers", "feature-extraction") and
("transformers", "sentence-similarity"). One model instance per
model_key (class-level singleton cache — same pattern as the llama
runners), behavior switches on req.other_texts.

sentence-transformers is imported lazily inside the .model property,
so importing this module doesn't require the library to be installed.
Only callers that actually instantiate an embedding model pay the
import cost — and if it fails, the error fires at first use, not at
dispatch import time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict

from hugpy_storage.download_models import ensure_model

from hugpy_media.schemas.embeded_schemas import EmbedRequest, EmbedResult


logger = logging.getLogger(__name__)


class FeatureExtractionRunner:
    """Runner for sentence-transformer-style embedding models.

    Per-process singleton cache (_MODELS) means many runner instances
    for the same model_key share one loaded SentenceTransformer. The
    Runner wrapper itself is cheap; the model isn't.
    """

    request_type = EmbedRequest
    result_type = EmbedResult

    # Class-level so all instances of this runner share one cache.
    # Same shape as get_llama_runner's _LLAMA_INSTANCES.
    _MODELS: Dict[str, Any] = {}
    _LOCK = threading.Lock()

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        self._runtime_kwargs = runtime_kwargs

    # --- model loading (lazy, singleton) -----------------------------------

    @property
    def model(self):
        cached = self._MODELS.get(self.model_key)
        if cached is not None:
            return cached

        with self._LOCK:
            cached = self._MODELS.get(self.model_key)
            if cached is not None:
                return cached

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for feature-extraction "
                    "tasks but is not installed. "
                    "`pip install sentence-transformers`."
                ) from exc

            model_dir = ensure_model(self.model_key)
            # trust_remote_code is needed for gte-large-en-v1.5 and similar
            # models that ship custom modeling code. all-minilm-l6-v2 ignores it.
            instance = SentenceTransformer(
                model_dir,
                trust_remote_code=True,
            )

            logger.info(
                "FeatureExtractionRunner: loaded model=%s dir=%s",
                self.model_key, model_dir,
            )
            self._MODELS[self.model_key] = instance
            return instance

    # --- encoding ---------------------------------------------------------

    def _encode(self, texts, normalize: bool, batch_size: int):
        """Blocking encode. Called from a worker thread by .run()."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # --- public API -------------------------------------------------------

    async def run(self, req: EmbedRequest) -> EmbedResult:
        try:
            embeddings = await asyncio.to_thread(
                self._encode, req.texts, req.normalize, req.batch_size,
            )

            if req.other_texts is None:
                # feature-extraction mode
                return EmbedResult(
                    request_id=req.request_id,
                    model_key=req.model_key,
                    ok=True,
                    embeddings=embeddings.tolist(),
                )

            # sentence-similarity mode
            other = await asyncio.to_thread(
                self._encode, req.other_texts, req.normalize, req.batch_size,
            )

            if req.normalize:
                # normalized vectors -> dot product is cosine similarity
                sims = embeddings @ other.T
            else:
                # explicit cosine similarity
                import numpy as np
                eps = 1e-12
                a_norm = embeddings / (
                    np.linalg.norm(embeddings, axis=1, keepdims=True) + eps
                )
                b_norm = other / (
                    np.linalg.norm(other, axis=1, keepdims=True) + eps
                )
                sims = a_norm @ b_norm.T

            return EmbedResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                embeddings=embeddings.tolist(),
                similarities=sims.tolist(),
            )

        except Exception as exc:
            logger.exception(
                "FeatureExtractionRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            return EmbedResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
