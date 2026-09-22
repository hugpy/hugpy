"""Frame-by-frame video analysis over a vision runner (chat-side helper).

Import-light: the vision runner/schemas come from ``hugpy_media`` (allowed),
the no-think helpers from ``hugpy_engine`` (lazy, inside the loop).
"""
from __future__ import annotations

import copy
import os.path as osp
import time
import uuid
from typing import Any, Union

from abstract_essentials import safe_dump_to_file, safe_load_from_json
from hugpy_media.vision.schemas import VisionRequest
from hugpy_media.vision.vision_runner import VisionRunner
from hugpy_platform.utils import get_base_64_image, require_file

from hugpy_video.schemas.video_schemas import (
    FrameAnalysis,
    VideoAnalysisConfig,
    VideoAnalysisSummary,
)

def _resolve_manifest_path(source: Union[str, Any]) -> str:
    if isinstance(source, str):
        return source
    p = getattr(source, "manifest_path", None)
    if isinstance(p, str):
        return p
    raise TypeError(
        f"expected manifest path str or object with .manifest_path, "
        f"got {type(source).__name__}"
    )

def _build_prompt(user_prompt: str, frame_context: dict) -> str:
    return (
        "-----------prompt---------\n"
        f"{user_prompt}\n"
        "-----------frame context---------\n"
        f"{frame_context}\n"
    )


async def analyze_video(
    source: Union[str, Any],
    runner: VisionRunner,
    config: VideoAnalysisConfig,
) -> VideoAnalysisSummary:
    manifest_path = require_file(_resolve_manifest_path(source), "manifest_path")
    manifest_data = safe_load_from_json(manifest_path)
    if not isinstance(manifest_data, dict):
        raise ValueError(f"manifest is not a dict: {manifest_path}")

    workspace_dir = manifest_data.get("workspace_dir")
    if not workspace_dir or not osp.isdir(workspace_dir):
        raise FileNotFoundError(f"workspace_dir invalid: {workspace_dir!r}")

    files = manifest_data.get("files") or {}
    frame_context_path = require_file(files.get("frame_context"), "files.frame_context")

    frame_contexts = safe_load_from_json(frame_context_path)
    if not isinstance(frame_contexts, list) or not frame_contexts:
        raise ValueError(
            f"frame_context must be a non-empty list, "
            f"got {type(frame_contexts).__name__}"
        )

    total_frames = len(frame_contexts)
    last = frame_contexts[-1]
    total_video_length = last.get("timestamp") if isinstance(last, dict) else None

    analysis_json_path = osp.join(workspace_dir, "analysis.json")
    manifest_data.setdefault("files", {})["analysis_json"] = analysis_json_path
    safe_dump_to_file(data=manifest_data, file_path=manifest_path)

    # Resume keyed by frame_path so reruns are idempotent.
    done_by_frame: dict[str, dict] = {}
    if config.resume and osp.isfile(analysis_json_path):
        prior = safe_load_from_json(analysis_json_path) or []
        for entry in prior:
            fp = entry.get("frame_path")
            if fp and entry.get("error") is None:
                done_by_frame[fp] = entry

    records: list[dict] = list(done_by_frame.values())
    model_key = runner.cfg.model_key  # runner is the source of truth

    for i, raw_ctx in enumerate(frame_contexts):
        if not isinstance(raw_ctx, dict):
            records.append({
                "frame_index": i,
                "error": f"frame_context[{i}] is not a dict",
            })
            continue

        ctx = copy.deepcopy(raw_ctx)  # never mutate caller's data
        frame_path = ctx.get("frame_path")

        if not frame_path or not osp.isfile(frame_path):
            err = f"frame_path missing or not found: {frame_path!r}"
            if config.raise_on_frame_error:
                raise FileNotFoundError(err)
            ctx.update({"frame_index": i, "error": err})
            records.append(ctx)
            continue

        if frame_path in done_by_frame:
            continue

        ctx["frame_index"] = i
        ctx["total_frames"] = total_frames
        ctx["total_video_length"] = total_video_length
        # NO-THINK (utils/no_think.py). Every frame's `analysis` is VALIDATED and
        # PERSISTED to analysis.json — a monologue here poisons the whole dataset,
        # and nothing downstream re-reads the frame to notice. The reasoning is
        # kept alongside (FrameAnalysis is extra="allow") rather than discarded.
        from hugpy_engine.utils.no_think import with_no_think, strip_think
        rendered_prompt = with_no_think(_build_prompt(config.prompt, ctx))
        image_b64 = get_base_64_image(frame_path)
        req = VisionRequest(
            request_id=f"frame-{i}-{uuid.uuid4().hex[:8]}",
            model_key=model_key,
            prompt=rendered_prompt,
            max_new_tokens=config.max_new_tokens,
            max_tokens=config.max_tokens,
            image_b64=image_b64,
        )

        t0 = time.time()
        try:
            vresult = await runner.run(req)
            text, err = vresult.text, vresult.error
        except Exception as e:
            if config.raise_on_frame_error:
                raise
            text, err = None, f"{type(e).__name__}: {e}"
        duration = time.time() - t0

        reasoning = ""
        if text:
            text, reasoning = strip_think(text)
            if not text:
                # Nothing but thinking — record it as an error rather than
                # persisting an empty analysis that reads like a clean run.
                err = err or ("model returned only reasoning and no analysis "
                              "(ignored the no-think directive)")

        ctx.update({
            "analysis_prompt": rendered_prompt,
            "analysis": text,
            "analysis_reasoning": reasoning,
            "model_key": model_key,
            "analysis_duration": duration,
            "error": err,
        })

        # Validate-on-write: schema drift fails fast, not three days from now
        FrameAnalysis.model_validate(ctx)
        records.append(ctx)

        if (i + 1) % config.save_every == 0 or (i + 1) == total_frames:
            safe_dump_to_file(data=records, file_path=analysis_json_path)

    succeeded = sum(1 for r in records if r.get("error") is None)
    failed = sum(1 for r in records if r.get("error") is not None)

    return VideoAnalysisSummary(
        manifest_path=manifest_path,
        analysis_json_path=analysis_json_path,
        frames_total=total_frames,
        frames_succeeded=succeeded,
        frames_failed=failed,
    )
