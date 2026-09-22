"""Schemas for the ``text-to-video`` / ``image-to-video`` tasks (studio seat).

Shaped like the tts pair on purpose: the registry-facing request is the
transport, the studio spine's ``StudioI2VSpec`` is the contract, and the
result carries the SHARED content-addressed clip path rather than bytes —
studio renders land on /mnt/llm_storage where every caller can read them, so
there is no b64 seam here (a video is not a wav; materializing it into the
response would dwarf the envelope).

``frames``/``width``/``height``/``duration_s`` are MEASURED off the produced
clip by the studio spine (ClipOutcome), never taken from the request's ask —
the ask is resolved against the bound model's ceiling and cadence at render
time (81-frame Wan reference, 4k+1 snap).
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class VideoGenRequest(BaseModel):
    """One clip to render. Field names line up 1:1 with ``make_studio_i2v``
    kwargs (this schema is the transport, that factory is the validator) —
    plus the transport-only ``request_id``/``model_key``/``pool`` every other
    request in this tree carries."""
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    pool: Optional[str] = None

    prompt: Optional[str] = None        # required for t2v; optional conditioning for i2v
    negative: Optional[str] = None
    start_image: Optional[str] = None   # abs path to a still (i2v conditioning)
    source_video: Optional[str] = None  # abs path to a prior clip (extend-from-last-frame)

    # Wan reference geometry as defaults — the measured-known-good shape on
    # this fleet (832x480@16, ~352s on ae's 3090 for the 81-frame reference).
    width: int = Field(default=832, ge=64)
    height: int = Field(default=480, ge=64)
    fps: int = Field(default=16, ge=1)
    requested_frames: Optional[int] = Field(default=None, ge=1)

    seed: int = Field(default=0, ge=0)
    steps: Optional[int] = Field(default=None, ge=1, le=100)
    cfg: Optional[float] = Field(default=None, ge=0, le=20)
    # None = AUTOFIT: sized to the serving worker's measured free VRAM at
    # render time (render_clip's ladder), never a guaranteed-fail low guess.
    vram_budget_gb: Optional[float] = Field(default=None, gt=0)
    project: Optional[str] = None


class VideoGenResult(BaseModel):
    request_id: str
    model_key: str
    ok: bool = True
    #: SHARED content-addressed clip path (readable by every box on the
    #: /mnt/llm_storage spine) — the caller ingests this, it never re-renders.
    path: Optional[str] = None
    content_hash: Optional[str] = None
    frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None
    resumed: bool = False
    #: AUTOFIT provenance, verbatim from ClipOutcome (honesty in the artifact).
    effective_budget_gb: Optional[float] = None
    budget_source: Optional[str] = None
    error: Optional[str] = None
    #: The bus JobError's code when the studio spine refused/failed, in its own
    #: vocabulary — never re-worded here.
    error_code: Optional[str] = None
