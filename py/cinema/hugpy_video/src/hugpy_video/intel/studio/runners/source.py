"""Common source-video selection for enhancement runners."""

from __future__ import annotations

from hugpy_video.intel.studio.schemas import RenderManifest


def resolve_source(manifest: RenderManifest) -> str | None:
    src = getattr(manifest, "source_video", "") or ""
    return src or None
