"""Pydantic contracts for chat-side video analysis (frame-by-frame vision runs)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class VideoAnalysisConfig(BaseModel):
    """How to analyze the frames produced by execute_prompt. Built once per run."""
    model_config = ConfigDict(frozen=True)

    prompt: str = "please provide analysis of this video frame"
    max_new_tokens: int = Field(default=128, gt=0)
    max_tokens: int = Field(default=384, gt=0)
    min_tokens: int = Field(default=32, gt=0)
    resume: bool = True
    raise_on_frame_error: bool = False
    save_every: int = Field(default=1, ge=1)


class FrameAnalysis(BaseModel):
    """One frame's persisted record. Validates on write so drift is caught early."""
    # extra="allow" so upstream frame_context fields (timestamp, etc.) round-trip
    model_config = ConfigDict(extra="allow")

    frame_index: int
    frame_path: str
    total_frames: int
    total_video_length: Optional[float] = None
    analysis_prompt: str
    analysis: Optional[str] = None
    model_key: str
    analysis_duration: float
    error: Optional[str] = None


class VideoAnalysisSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_path: str
    analysis_json_path: str
    frames_total: int
    frames_succeeded: int
    frames_failed: int
