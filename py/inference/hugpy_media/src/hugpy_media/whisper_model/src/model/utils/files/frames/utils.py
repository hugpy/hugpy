import math
from typing import Any

from hugpy_media.schemas.whisper_schemas import FrameCandidate
def get_segment_frame_times(
    whisper_result: dict[str, Any],
    min_gap_seconds: float = 1.5,
    long_segment_seconds: float = 8.0,
) -> list[FrameCandidate]:
    candidates: list[FrameCandidate] = []
    last_timestamp = -math.inf

    for index, segment in enumerate(whisper_result.get("segments", [])):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        duration = max(0.0, end - start)

        if duration <= 0.0:
            continue

        if duration >= long_segment_seconds:
            timestamps = [
                (start + min(1.0, duration * 0.15), "segment_start_context"),
                ((start + end) / 2.0, "segment_midpoint"),
                (end - min(1.0, duration * 0.15), "segment_end_context"),
            ]
        else:
            timestamps = [((start + end) / 2.0, "segment_midpoint")]

        for timestamp, reason in timestamps:
            if timestamp - last_timestamp < min_gap_seconds:
                continue

            candidates.append(
                FrameCandidate(
                    timestamp=timestamp,
                    reason=reason,
                    segment_index=index,
                    text=text,
                )
            )
            last_timestamp = timestamp

    return candidates
