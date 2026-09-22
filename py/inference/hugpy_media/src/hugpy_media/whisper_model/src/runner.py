import asyncio
import os
from typing import Any

from abstract_essentials import derive_media_type

from hugpy_media.schemas.whisper_schemas import (
    FrameContext,
    TranscribeRequest,
    TranscribeResult,
    TranscribeSegment,
    TranscribeWord,
)
from hugpy_media.whisper_model.src.model.execute import transcribe_file_with_workspace


class WhisperRunner:
    request_type = TranscribeRequest
    result_type = TranscribeResult


    def __init__(self, cfg):
        self.model_key = cfg.model_key

    async def run(self, req: TranscribeRequest) -> TranscribeResult:
        if not req.file_path:
            raise ValueError("must provide file_path")

        if not os.path.isfile(req.file_path):
            raise FileNotFoundError(f"Media file does not exist: {req.file_path}")

        # Transcription failures (codec issues, OOM, model load) come back as a
        # structured ok=False result so the route/dispatch layer can report them
        # cleanly instead of letting a raw exception escape the runner.
        try:
            result = await asyncio.to_thread(
                transcribe_file_with_workspace,
                file_path=req.file_path,
                model_size=req.model_size,
                language=req.language,
                task=req.task,
                whisper_model_path=req.whisper_model_path,
                audio_path=req.output_audio_path,
                output_root=req.output_root,
                copy_source=req.copy_source,
                capture_frames=req.capture_frames,
                min_gap_seconds=req.min_gap_seconds,
                long_segment_seconds=req.long_segment_seconds,
                word_timestamps=req.word_timestamps,
            )

            whisper_result = result.get("whisper_result", {})
            media_type = derive_media_type(req.file_path)

            return TranscribeResult(
                ok=True,
                request_id=req.request_id,
                file_path=req.file_path,
                media_type=media_type,
                model_key=req.model_key or self.model_key,
                model_size=req.model_size,
                task=req.task,
                text=whisper_result.get("text", ""),
                language=whisper_result.get("language", req.language),
                duration=whisper_result.get("duration"),
                audio_path=result.get("audio_path"),
                workspace_dir=result.get("workspace_dir"),
                transcript_json_path=result.get("transcript_json_path"),
                transcript_text_path=result.get("transcript_text_path"),
                manifest_path=result.get("manifest_path"),
                segments=[
                    self._segment_from_dict(segment)
                    for segment in whisper_result.get("segments", [])
                ],
                frames=[
                    FrameContext(**frame)
                    for frame in result.get("frames", [])
                ],
                raw=result,
            )
        except Exception as exc:
            return TranscribeResult(
                ok=False,
                request_id=req.request_id,
                file_path=req.file_path,
                media_type=derive_media_type(req.file_path),
                model_key=req.model_key or self.model_key,
                model_size=req.model_size,
                task=req.task,
                language=req.language,
                error=str(exc),
                raw={"error_type": type(exc).__name__},
            )

    @staticmethod
    def _segment_from_dict(segment: dict[str, Any]) -> TranscribeSegment:
        words = [
            TranscribeWord(**word)
            for word in segment.get("words", [])
            if isinstance(word, dict)
        ]

        return TranscribeSegment(
            id=int(segment.get("id", 0)),
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=str(segment.get("text", "")),
            seek=segment.get("seek"),
            tokens=segment.get("tokens") or [],
            temperature=segment.get("temperature"),
            avg_logprob=segment.get("avg_logprob"),
            compression_ratio=segment.get("compression_ratio"),
            no_speech_prob=segment.get("no_speech_prob"),
            words=words,
        )
