"""Runner for the studio video spine (Wan / VACE / LTX zoo).

The fleet's Runner protocol is about SERVING, not about which library holds
the weights (the chatterbox precedent): a registry row's Wan checkpoint is
rendered by the studio spine — ``render_clip`` delegates to a studio GPU
worker when ``HUGPY_STUDIO_WORKER`` resolves and the spec binds a real model,
else renders in-process — and this runner just lifts the registry request
into that spine and reads the normalized ``ClipOutcome`` back. Before this
class, ``text-to-video`` rows could only refuse with "no runner registered"
while the studio path rendered the same models fine one floor below.

Kept import-light on purpose: ``studio_i2v`` (bus, delegation loop) is paid
for inside ``run()``, never at registry-build time.
"""
from __future__ import annotations

import asyncio
import logging

from hugpy_video.video_gen.schemas import VideoGenRequest, VideoGenResult

logger = logging.getLogger(__name__)

# Registry model keys and studio zoo ids are two vocabularies for one zoo:
# `Wan-AI~Wan2.1-VACE-1.3B` (hub-derived) vs `wan2.1-vace-1.3b` (studio).
# Normalize the mechanical differences (case, org~ prefix, packaging suffix)
# and let the studio router refuse anything still unknown as err-as-data —
# never guess a different model.
_PACKAGING_SUFFIXES = ("-diffusers", "-gguf")


def studio_model_id(model_key: str) -> str:
    mid = (model_key or "").strip().lower()
    if "~" in mid:
        mid = mid.split("~", 1)[1]
    for suffix in _PACKAGING_SUFFIXES:
        if mid.endswith(suffix):
            mid = mid[: -len(suffix)]
    return mid


from hugpy_platform.runner_config import RunnerConfig


class StudioVideoRunner(RunnerConfig):
    """Serves ``text-to-video`` (capability t2v) and ``image-to-video``
    (capability i2v — a request carrying ``start_image``/``source_video``)."""

    request_type = VideoGenRequest
    result_type = VideoGenResult


    def _render(self, req: VideoGenRequest) -> VideoGenResult:
        from hugpy_video.intel.runners.studio_i2v import render_clip
        from hugpy_video.intel.studio.job import make_studio_i2v

        model_key = req.model_key or self.model_key
        capability = "i2v" if (req.start_image or req.source_video) else "t2v"
        try:
            spec = make_studio_i2v(
                capability=capability,
                width=req.width, height=req.height, fps=req.fps,
                vram_budget_gb=req.vram_budget_gb,
                seed=req.seed,
                prompt=req.prompt, negative=req.negative,
                start_image=req.start_image, source_video=req.source_video,
                steps=req.steps, cfg=req.cfg,
                requested_frames=req.requested_frames,
                project=req.project,
                model_id=studio_model_id(model_key),
            )
        except (TypeError, ValueError) as exc:
            return VideoGenResult(
                request_id=req.request_id, model_key=model_key, ok=False,
                error=str(exc), error_code="bad_spec")

        outcome = render_clip(spec, render_id=req.request_id)
        if not outcome.ok:
            err = outcome.error
            return VideoGenResult(
                request_id=req.request_id, model_key=model_key, ok=False,
                error=getattr(err, "message", None) or str(err),
                error_code=getattr(err, "code", None) or "render_failed",
                effective_budget_gb=outcome.effective_budget_gb,
                budget_source=outcome.budget_source)
        return VideoGenResult(
            request_id=req.request_id, model_key=model_key, ok=True,
            path=outcome.path, content_hash=outcome.content_hash,
            frames=outcome.frames, width=outcome.width, height=outcome.height,
            duration_s=outcome.duration_s, resumed=outcome.resumed,
            effective_budget_gb=outcome.effective_budget_gb,
            budget_source=outcome.budget_source)

    async def run(self, req: VideoGenRequest) -> VideoGenResult:
        # render_clip is synchronous (it polls the delegated worker); a render
        # is minutes-long, so it must not sit on the event loop.
        return await asyncio.to_thread(self._render, req)

    async def stream(self, req, cancel_event):
        raise NotImplementedError("video generation does not stream")
