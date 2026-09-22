"""Audio-extraction schema — map §4.4.

Frozen spec + validating factory, mirroring frame_schema.py exactly. An audio
extraction pulls a video's audio track out to a standalone audio file of one
`fmt`. Per-field validity (`source` is a video, `fmt` is a known container) is
enforced in the factory with LOCAL raises (construction-time, never across a
boundary). Temporal trimming of the resulting audio is NOT this job's concern —
that already works through the existing `crop` job's temporal branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hugpy_video.intel.media_schema import MediaRef

AudioFmt = Literal["wav", "mp3", "m4a"]

# Construction-time guard set for `fmt` (registry-over-globals discipline).
AUDIO_FMTS = frozenset({"wav", "mp3", "m4a"})


@dataclass(frozen=True)
class AudioExtractSpec:
    """Extract `source`'s (a video) audio track into a standalone audio file.

        fmt   output container/codec family: wav|mp3|m4a (default wav)
    """
    source: MediaRef
    fmt: str = "wav"


def make_audio_extract(source: MediaRef, fmt: str = "wav") -> AudioExtractSpec:
    """Validate + build an AudioExtractSpec. A raise here is fine: local to
    construction, never crosses a boundary (map §4 / §6).

    Also the reconstruction path used by the bus when it deserializes the spec
    from JSON — kwargs line up 1:1 with the fields so it round-trips.
    """
    if source.kind != "video":
        raise ValueError(f"audio_extract source must be a video; got kind={source.kind!r}")
    if fmt not in AUDIO_FMTS:
        raise ValueError(f"fmt must be one of {sorted(AUDIO_FMTS)}; got {fmt!r}")
    return AudioExtractSpec(source=source, fmt=fmt)
