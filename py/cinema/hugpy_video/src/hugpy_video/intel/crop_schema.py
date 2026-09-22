"""Unified region / crop schema — map §4.2 exactly.

The load-bearing decision (map §9.1): one CropSpec, spatial/temporal as
orthogonal optional axes. Image crop = spatial bbox. Audio crop = temporal
interval. Video = either or both. Per-kind axis validity is enforced in the
factory with LOCAL raises (construction-time, never across a boundary).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hugpy_video.intel.media_schema import MediaRef


@dataclass(frozen=True)
class SpatialRegion:
    """Pixel-space bbox, origin top-left, half-open on the far edges."""
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class TemporalRegion:
    """Half-open interval [start_s, end_s) in seconds."""
    start_s: float
    end_s: float


@dataclass(frozen=True)
class CropSpec:
    """A crop is a region over one or both axes.
        image -> spatial only
        audio -> temporal only
        video -> spatial and/or temporal
    Exactly-one-axis validity is enforced per source.kind in the factory."""
    source: MediaRef
    spatial: Optional[SpatialRegion] = None
    temporal: Optional[TemporalRegion] = None


def make_crop(source: MediaRef,
              spatial: Optional[SpatialRegion] = None,
              temporal: Optional[TemporalRegion] = None) -> CropSpec:
    if source.kind == "image" and temporal is not None:
        raise ValueError("image crop cannot carry a temporal region")
    if source.kind == "audio" and spatial is not None:
        raise ValueError("audio crop cannot carry a spatial region")
    if spatial is None and temporal is None:
        raise ValueError("crop must specify at least one axis")
    return CropSpec(source=source, spatial=spatial, temporal=temporal)
