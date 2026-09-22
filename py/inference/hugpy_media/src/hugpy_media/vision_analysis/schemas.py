"""Schemas for the vision-analysis pipeline tasks.

One request/result pair serves the whole family (depth-estimation,
object-detection, image-classification, image-segmentation): the input is
always ONE image, the output is always items (labels/scores/boxes) and/or
derived images (a depth map, segmentation masks). Task-specific knobs are
all optional — anything unset defers to the transformers pipeline default.

The derived-image shape is imagegen's GeneratedImage on purpose: the console
already knows how to render that (path + optional b64), so depth maps and
masks display exactly like generated images.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hugpy_media.imagegen.schemas import GeneratedImage


class VisionAnalysisRequest(BaseModel):
    """One unit of image-analysis work. Built per call."""
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    pool: Optional[str] = None   # dedicated worker pool (routing); None = general

    # Exactly one image source. image_path rides the worker-offload inliner
    # (remote._PATH_KEYS) so offloaded requests carry the bytes.
    image_path: Optional[str] = None
    image_b64: Optional[str] = None

    # Optional knobs; None defers to the pipeline's per-model defaults.
    top_k: Optional[int] = Field(default=None, ge=1, le=100)          # classification
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # detection score cut
    candidate_labels: Optional[List[str]] = None                      # zero-shot variants
    # b64 in derived images lets HTTP callers fetch bytes without a second
    # round-trip; False for in-process callers that only want saved paths.
    return_b64: bool = True

    @model_validator(mode="after")
    def _exactly_one_image(self):
        if bool(self.image_path) == bool(self.image_b64):
            raise ValueError(
                "vision-analysis request needs exactly one of "
                "'image_path' or 'image_b64'")
        return self


class VisionAnalysisResult(BaseModel):
    request_id: str
    model_key: str
    ok: bool = True
    task: str = ""
    # Structured findings: [{label, score, box?}, ...] for detection /
    # classification / segmentation; summary stats for depth.
    items: List[Dict[str, Any]] = Field(default_factory=list)
    # Derived images (depth map, segmentation masks) — console-renderable.
    images: List[GeneratedImage] = Field(default_factory=list)
    # Human-readable summary so chat-stream wrapping (stream_runner's one-shot
    # path reads result.text) degrades to something sensible.
    text: str = ""
    error: Optional[str] = None
