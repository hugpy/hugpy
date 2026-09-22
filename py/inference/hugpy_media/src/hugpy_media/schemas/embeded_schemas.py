"""Schemas for feature-extraction / sentence-similarity.

One request type covers both tasks:
    - texts only                 -> embeddings only
    - texts + other_texts        -> embeddings + similarity matrix

Keeps the runner free of branching on kwargs. The presence of
`other_texts` is the entire dispatch signal inside the runner.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmbedRequest(BaseModel):
    """Input for feature-extraction and sentence-similarity tasks."""
    model_config = ConfigDict(frozen=True)

    model_key: str
    request_id: str
    pool: Optional[str] = None   # dedicated worker pool (routing); None = general

    # Always a list, even for single-text input. Builders normalize.
    texts: List[str] = Field(min_length=1)

    # Sentence-similarity mode: similarities are texts x other_texts.
    # None means feature-extraction mode (embeddings only).
    other_texts: Optional[List[str]] = None

    # L2-normalize embeddings. For cosine similarity, leaving this True
    # means the similarity = dot product (faster, same result).
    normalize: bool = True

    # Encoder batch size. Sentence-transformers handles padding per batch.
    batch_size: int = 32


class EmbedResult(BaseModel):
    """Result for feature-extraction and sentence-similarity tasks.

    `embeddings` is always present on success — a list of vectors aligned
    with req.texts. `similarities` is present only when req.other_texts
    was set; shape is len(texts) x len(other_texts).
    """
    model_config = ConfigDict(frozen=True)

    request_id: str
    model_key: str
    ok: bool

    embeddings: Optional[List[List[float]]] = None
    similarities: Optional[List[List[float]]] = None

    error: Optional[str] = None
