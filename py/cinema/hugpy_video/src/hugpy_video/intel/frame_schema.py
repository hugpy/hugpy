"""Frame-extraction schema — map §4.3.

Frozen spec + validating factory, mirroring crop_schema.py exactly. A frame
extraction samples a video at a fixed `fps`, optionally within a temporal
`window`, into many still frames of one `fmt`. Per-fmt quality validity is
enforced in the factory with LOCAL raises (construction-time, never across a
boundary). `max_frames` is a RUNTIME cap enforced by the runner (map §4.3:
refuse loudly, do not truncate) — NOT validated here because it depends on the
source duration which the runner is the authority on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

from hugpy_video.intel.media_schema import MediaRef
from hugpy_video.intel.crop_schema import TemporalRegion

FrameFmt = Literal["jpg", "png", "webp"]

# Construction-time guard set for `fmt` (registry-over-globals discipline).
FRAME_FMTS = frozenset({"jpg", "png", "webp"})


@dataclass(frozen=True)
class FrameExtractSpec:
    """Sample `source` (a video) at `fps` into still frames of `fmt`.

        window     optional [start_s, end_s) to restrict extraction
        quality    per-fmt: jpg/webp 1..100 (higher=better), png 0..9 (higher=smaller)
        max_frames RUNTIME cap (runner refuses if the expected count exceeds it)
    """
    source: MediaRef
    fps: float
    quality: int
    fmt: FrameFmt
    window: Optional[TemporalRegion] = None
    max_frames: Optional[int] = None


def make_frame_extract(
    source: MediaRef,
    fps: float,
    quality: int,
    fmt: str,
    window: Optional[TemporalRegion] = None,
    max_frames: Optional[int] = None,
) -> FrameExtractSpec:
    """Validate + build a FrameExtractSpec. A raise here is fine: local to
    construction, never crosses a boundary (map §4 / §6).

    Also the reconstruction path used by the bus when it deserializes the spec
    from JSON — kwargs line up 1:1 with the fields so it round-trips.
    """
    if source.kind != "video":
        raise ValueError(f"frame_extract source must be a video; got kind={source.kind!r}")
    if not (isinstance(fps, (int, float)) and fps > 0):
        raise ValueError(f"fps must be > 0; got {fps!r}")
    if fmt not in FRAME_FMTS:
        raise ValueError(f"fmt must be one of {sorted(FRAME_FMTS)}; got {fmt!r}")
    # per-fmt quality range: jpg/webp use ffmpeg -qscale:v 1..100 (higher=better),
    # png uses -compression_level 0..9 (higher=smaller/slower).
    if fmt == "png":
        if not (isinstance(quality, int) and 0 <= quality <= 9):
            raise ValueError(f"png quality (compression_level) must be 0..9; got {quality!r}")
    else:
        if not (isinstance(quality, int) and 1 <= quality <= 100):
            raise ValueError(f"{fmt} quality must be 1..100; got {quality!r}")
    if window is not None:
        if not (0 <= window.start_s < window.end_s):
            raise ValueError(
                f"window must satisfy 0 <= start_s < end_s; got {window}"
            )
    return FrameExtractSpec(
        source=source,
        fps=fps,
        quality=quality,
        fmt=fmt,
        window=window,
        max_frames=max_frames,
    )
