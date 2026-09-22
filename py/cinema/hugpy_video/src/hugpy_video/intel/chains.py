"""Composition chains — map §7 (video -> frames -> generate).

`resolve_video_parts(spec)` turns a GenerateImageSpec that may contain VIDEO
parts into an equivalent spec containing only text + image parts, by running
frame extraction on each video part and substituting the resulting frame
MediaRefs as image parts (in order). This is PLAIN CODE in a small module — NOT
a special case buried inside the runner — so `run_generate_image` never sees a
video part.

It is invoked in the enqueue path (the generate_image route) BEFORE the spec is
handed to the bus.

Manual frame-pick is the DEFAULT UX: the UI adds picked frames as image parts
directly, so in practice a raw video part is the FALLBACK path. This uniform-N
sampling is exactly that fallback.

Error policy (documented choice): frame extraction failures SURFACE as a raised
ValueError here. The route catches it and returns 400 — i.e. we only ever return
a resolved spec when extraction actually succeeded; we never silently pass an
unresolved (or partly-resolved) spec through to the bus.
"""
from __future__ import annotations

import math
from uuid import uuid4

from hugpy_video.intel.frame_schema import make_frame_extract
from hugpy_video.intel.gen_schema import (
    GenerateImageSpec,
    GenPromptPart,
    image_part,
    make_generate_image,
)
from hugpy_video.intel.movie_schema import GoalInterval, MovieSpec, make_movie
from hugpy_video.intel.scene_schema import GenerateSceneSpec, make_generate_scene
from hugpy_video.intel.runners.ffmpeg_frames import run_frame_extract


def _extract_uniform_frames(media, uniform_n: int):
    """Run frame extraction on a single video MediaRef, returning ~uniform_n
    frame MediaRefs sampled uniformly across it. Raises ValueError on failure
    (surfaced to the route as a 400)."""
    duration = media.duration_s
    if duration is None or duration <= 0:
        raise ValueError(
            f"cannot uniformly sample video {media.asset_id}: unknown/zero duration"
        )
    # fps chosen so fps*duration == uniform_n -> ~uniform_n frames across the clip.
    fps = uniform_n / duration
    # cap sits comfortably above the expected count (ffmpeg's fps filter may emit
    # one extra at a boundary); the cap is a safety valve, not the target.
    cap = uniform_n + 4
    fe_spec = make_frame_extract(
        source=media,
        fps=fps,
        quality=90,
        fmt="jpg",
        window=None,
        max_frames=cap,
    )
    result = run_frame_extract(fe_spec, job_id=uuid4().hex)
    if not result.ok:
        err = result.error
        raise ValueError(
            f"frame extraction failed for video {media.asset_id}: "
            f"{err.code}: {err.message}" if err else
            f"frame extraction failed for video {media.asset_id}"
        )
    if not result.outputs:
        raise ValueError(
            f"frame extraction produced no frames for video {media.asset_id}"
        )
    return result.outputs


def resolve_video_parts(spec: GenerateImageSpec, *, uniform_n: int = 4) -> GenerateImageSpec:
    """Return a NEW GenerateImageSpec whose parts contain only text + image
    (no video). Text/image parts pass through unchanged; each video part is
    replaced (in order) by the image parts of its uniformly-sampled frames."""
    new_parts = []
    for p in spec.parts:
        if p.kind == "video":
            frames = _extract_uniform_frames(p.media, uniform_n)
            new_parts.extend(image_part(ref) for ref in frames)
        else:
            new_parts.append(p)
    # Re-validate through the factory so the resolved spec is provably video-free.
    return make_generate_image(
        parts=tuple(new_parts),
        model_id=spec.model_id,
        width=spec.width,
        height=spec.height,
        steps=spec.steps,
        guidance=spec.guidance,
        seed=spec.seed,
        negative=spec.negative,
        strength=spec.strength,   # carry the img2img knob through re-validation
        project=spec.project,     # carry the auto-archive NAME through re-validation
    )


def resolve_video_parts_scene(spec: GenerateSceneSpec, *, uniform_n: int = 4) -> GenerateSceneSpec:
    """Scene twin of resolve_video_parts: return a NEW GenerateSceneSpec whose
    parts contain only text + image (no video). Text/image parts pass through
    unchanged; each video part is replaced (in order) by the image parts of its
    uniformly-sampled frames. Reuses the same _extract_uniform_frames helper +
    image_part, and re-validates through make_generate_scene (carrying all scene
    fields). Extraction failure SURFACES as a raised ValueError (route -> 400)."""
    new_parts = []
    for p in spec.parts:
        if p.kind == "video":
            frames = _extract_uniform_frames(p.media, uniform_n)
            new_parts.extend(image_part(ref) for ref in frames)
        else:
            new_parts.append(p)
    # Re-validate through the factory so the resolved spec is provably video-free.
    return make_generate_scene(
        parts=tuple(new_parts),
        model_id=spec.model_id,
        width=spec.width,
        height=spec.height,
        steps=spec.steps,
        guidance=spec.guidance,
        n_frames=spec.n_frames,
        fps=spec.fps,
        assemble=spec.assemble,
        seed=spec.seed,
        motion=spec.motion,
        negative=spec.negative,
        strength=spec.strength,   # carry img2img knobs through re-validation
        chain=spec.chain,
        project=spec.project,     # carry the auto-archive NAME through re-validation
    )


def resolve_video_parts_movie(spec: MovieSpec) -> MovieSpec:
    """Movie twin of resolve_video_parts_scene: return a NEW MovieSpec whose goal
    refs are provably video-free. A goal.ref that is a VIDEO is replaced by the
    FIRST frame of a uniform 1-frame extraction (its representative still), so the
    runner's per-segment start image is always an image. Text-only goals + goals
    with an image ref (or no ref) pass through unchanged. Reuses the same
    _extract_uniform_frames helper and re-validates through make_movie (carrying
    every scene-template field + director knob). Extraction failure SURFACES as a
    raised ValueError (route -> 400)."""
    new_goals = []
    for g in spec.goals:
        ref = g.ref
        if ref is not None and ref.kind == "video":
            frames = _extract_uniform_frames(ref, uniform_n=1)
            ref = frames[0]  # the representative still frame
        new_goals.append(GoalInterval(
            start_frame=g.start_frame, end_frame=g.end_frame,
            prompt=g.prompt, ref=ref,
        ))
    return make_movie(
        goals=tuple(new_goals),
        model_id=spec.model_id,
        width=spec.width,
        height=spec.height,
        steps=spec.steps,
        guidance=spec.guidance,
        fps=spec.fps,
        assemble=spec.assemble,
        seed=spec.seed,
        negative=spec.negative,
        strength=spec.strength,
        chain=spec.chain,
        project=spec.project,
        vision_enabled=spec.vision_enabled,
        score_threshold=spec.score_threshold,
        max_attempts_per_segment=spec.max_attempts_per_segment,
        judge_model_id=spec.judge_model_id,
        time_budget_s=spec.time_budget_s,
    )
