"""Media substrate — the immutable handle to an asset in the store.

Mirrors hugpy_video_intelligence_map.md §4.1 exactly. Metadata is resolved ONCE
at ingest (see media_store.py); a MediaRef is never mutated afterward. It is the
single source of truth for "what is this asset".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Literal

MediaKind = Literal["image", "audio", "video"]

# The construction-time guard set for `kind`. A frozenset so it is a cheap,
# immutable membership check (registry-over-globals discipline).
MEDIA_KINDS = frozenset({"image", "audio", "video"})


@dataclass(frozen=True)
class MediaRef:
    """Immutable handle to an asset in the MediaStore. Metadata resolved ONCE at
    ingest via the layered resolver; never mutated afterward. Single source of
    truth for 'what is this asset'."""
    asset_id: str
    kind: MediaKind
    uri: str                       # store key / local path (os.path-built)
    mime: str
    # descriptive, resolved at ingest:
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None
    fps_native: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    # ATTRIBUTION (2026-08-06): the central-account username this asset was
    # ingested for, when it was ingested inside a request that had one. NOT a
    # descriptive property of the pixels — it is carried so an ingested source
    # can be traced to the member who supplied it, alongside the authoritative
    # per-job `owner` column in media_bus. None for every asset produced by a
    # runner (no request context) and for every pre-2026-08-06 spec rehydrated
    # from JSON — the field is optional and defaults to None, so
    # ``make_media_ref(**asdict(old_ref))`` still round-trips.
    #
    # DELIBERATELY NOT a content-hash input: studio's canonical_inputs() builds
    # its reproducibility key from explicit fields (never asdict of a MediaRef),
    # so adding this column re-addresses nothing.
    owner: Optional[str] = None
    # VISIBILITY (t172): the public/private hint carried alongside ``owner`` — the
    # SIBLING of the owner attribution, mirroring identity profiles' owner+private
    # pair. DEFAULT False == PUBLIC. Like ``owner`` this is not a property of the
    # pixels and is NOT the authoritative visibility record — the /video routes
    # gate on the per-job ``private`` column in media_bus. It is threaded through
    # ingest() so a runner-minted OUTPUT ref can carry the initiating request's
    # private choice to the record it stamps. Optional + default False, so
    # ``make_media_ref(**asdict(old_ref))`` still round-trips a pre-t172 spec.
    private: bool = False


def make_media_ref(
    asset_id: str,
    kind: str,
    uri: str,
    mime: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    duration_s: Optional[float] = None,
    fps_native: Optional[float] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    owner: Optional[str] = None,
    private: bool = False,
) -> MediaRef:
    """Validate + build a MediaRef. A raise here is fine: it is local to
    construction and never crosses a module boundary (see map §4 / §6).

    Also the reconstruction path used by the bus when it deserializes a spec
    from JSON — the kwargs line up 1:1 with MediaRef's fields so
    ``make_media_ref(**dataclasses.asdict(ref))`` round-trips.
    """
    if kind not in MEDIA_KINDS:
        raise ValueError(f"kind must be one of {sorted(MEDIA_KINDS)}; got {kind!r}")
    if not uri or not os.path.isabs(uri):
        raise ValueError(f"uri must be an absolute path; got {uri!r}")
    return MediaRef(
        asset_id=asset_id,
        kind=kind,
        uri=uri,
        mime=mime,
        width=width,
        height=height,
        duration_s=duration_s,
        fps_native=fps_native,
        sample_rate=sample_rate,
        channels=channels,
        owner=owner,
        private=bool(private),
    )
