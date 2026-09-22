"""Request builders for the engine tasks video serves.

Copied verbatim (imports aside) from
``hugpy_engine.resolvers.categories.builders._build_videogen_request`` — the
engine no longer knows the video request shape; this module is what
``hugpy_video.plugin.register`` hands to ``hugpy_engine.tasks.register_task``.
Import-light: pydantic + the request schema only.
"""

from __future__ import annotations

from typing import Any, Dict

from hugpy_platform.utils import make_request_id

from hugpy_video.video_gen.schemas import VideoGenRequest

__all__ = ["build_videogen_request"]


def build_videogen_request(kwargs: Dict[str, Any], model_key: str) -> VideoGenRequest:
    """text-to-video / image-to-video: one clip to render on the studio spine.

    A t2v ask needs a prompt; an i2v ask needs a conditioning input
    (``start_image`` or ``source_video``) — the runner derives the capability
    from which of those is present, so the guard here is "at least one of the
    three", refused with the keys we got (never a silent empty render).
    Geometry/length/sampler fields pass through; their real validation lives
    in ``make_studio_i2v`` (the contract), range-guards in the schema.
    """
    prompt = kwargs.get("prompt") or kwargs.get("text")
    start_image = kwargs.get("start_image") or kwargs.get("image")
    source_video = kwargs.get("source_video")
    if not (prompt or start_image or source_video):
        raise ValueError(
            "video request needs 'prompt' (t2v) or 'start_image'/"
            f"'source_video' (i2v); got keys: {sorted(kwargs)}")

    out: Dict[str, Any] = {
        "request_id": kwargs.get("request_id", make_request_id()),
        "model_key": model_key,
        "prompt": prompt,
        "start_image": start_image,
        "source_video": source_video,
    }
    for k in ("negative", "width", "height", "fps", "requested_frames",
              "seed", "steps", "cfg", "vram_budget_gb", "project", "pool"):
        if kwargs.get(k) is not None:
            out[k] = kwargs[k]
    return VideoGenRequest(**out)


# Engine-registry name kept for callers that grep for the original.
_build_videogen_request = build_videogen_request
