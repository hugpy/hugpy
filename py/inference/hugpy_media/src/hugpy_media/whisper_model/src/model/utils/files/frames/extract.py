import json
import os
from typing import Any

from hugpy_media.whisper_model.src.model.utils.files.frames.utils import get_segment_frame_times
import subprocess
def extract_frame_ffmpeg(
    video_path: str,
    timestamp: float,
    output_path: str,
    quality: int = 2,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    from hugpy_platform.binaries import resolve_bin
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"   # .exe on Windows; bare name as last resort
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        str(quality),
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to extract frame.\n\n"
            f"Command:\n{' '.join(command)}\n\n"
            f"stderr:\n{result.stderr}"
        )

    return output_path


def extract_context_frames_from_whisper(
    video_path: str,
    whisper_result: dict[str, Any],
    output_dir: str,
    min_gap_seconds: float = 1.5,
    long_segment_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)

    candidates = get_segment_frame_times(
        whisper_result=whisper_result,
        min_gap_seconds=min_gap_seconds,
        long_segment_seconds=long_segment_seconds,
    )

    extracted: list[dict[str, Any]] = []

    for item_index, candidate in enumerate(candidates):
        frame_name = (
            f"frame_{item_index:04d}_"
            f"seg_{candidate.segment_index:04d}_"
            f"{candidate.timestamp:.3f}s.jpg"
        )
        frame_path = os.path.join(output_dir, frame_name)

        extract_frame_ffmpeg(
            video_path=video_path,
            timestamp=candidate.timestamp,
            output_path=frame_path,
        )

        extracted.append(
            {
                "frame_path": frame_path,
                "timestamp": candidate.timestamp,
                "reason": candidate.reason,
                "segment_index": candidate.segment_index,
                "text": candidate.text,
            }
        )

    frame_context_path = os.path.join(os.path.dirname(output_dir), "frame_context.json")
    with open(frame_context_path, "w", encoding="utf-8") as file:
        json.dump(extracted, file, indent=2, ensure_ascii=False)
    return extracted
