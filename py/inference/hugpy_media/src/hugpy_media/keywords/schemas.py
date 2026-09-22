"""Dispatch schemas for the keyword-extraction task.

These are the request/result types the dispatch layer trades in (pydantic,
request_id/model_key threading, ok/error contract) — distinct from the
internal KeywordRequest/KeywordResult dataclasses keybert_model.py uses as
its parameter bags. The builder maps prompt_kwargs onto this request; the
runner maps a RefinedResult (or raw KeywordResult) back onto the result.
"""
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class KeywordTaskRequest(BaseModel):
    """One unit of keyword-extraction work. Built per call."""
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    pool: Optional[str] = None   # dedicated worker pool (routing); None = general
    text: str = Field(min_length=1)

    # Named parameter bundle (default/seo/metadata/social/long_tail/article).
    # Explicit kwargs below override the preset, same resolution as summarizers.
    preset: Optional[str] = None
    # refine=True runs the full extract -> filter -> classify -> format
    # pipeline (refine_keywords); False returns the raw merged extraction.
    refine: bool = True

    top_n: Optional[int] = Field(default=None, ge=1, le=100)
    diversity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    use_mmr: Optional[bool] = None
    stop_words: Optional[str] = None
    keyphrase_ngram_range: Optional[Tuple[int, int]] = None

    # refine-only gates
    min_density: Optional[float] = None
    max_density: Optional[float] = None
    min_score: Optional[float] = None
    max_words_per_phrase: Optional[int] = None


class KeywordTaskResult(BaseModel):
    request_id: str
    model_key: str
    ok: bool = True

    preset_used: Optional[str] = None
    primary: List[str] = Field(default_factory=list)
    secondary: List[str] = Field(default_factory=list)
    dropped: List[str] = Field(default_factory=list)
    combined: List[str] = Field(default_factory=list)

    density: Dict[str, float] = Field(default_factory=dict)
    density_flags: Dict[str, str] = Field(default_factory=dict)

    meta_keywords: str = ""
    hashtags: List[str] = Field(default_factory=list)
    slug_candidates: List[str] = Field(default_factory=list)

    backends_used: List[str] = Field(default_factory=list)
    backend_errors: Dict[str, str] = Field(default_factory=dict)

    # Human-readable rendering so chat-stream wrapping (stream_runner's
    # one-shot path reads result.text) degrades to something sensible.
    text: str = ""
    error: Optional[str] = None
