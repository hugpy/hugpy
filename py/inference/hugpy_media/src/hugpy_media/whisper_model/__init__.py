"""Speech recognition (openai-whisper) — runner, schemas and URL streaming."""

from hugpy_media.schemas.whisper_schemas import (
    FrameContext,
    TranscribeRequest,
    TranscribeResult,
    TranscribeSegment,
    TranscribeWord,
)
from hugpy_media.whisper_model.src.runner import WhisperRunner
from hugpy_media.whisper_model.src.stream import (
    extension_from_content_type,
    stream_url_to_temp_file,
    whisper_transcribe_url_stream,
)

__all__ = [
    "FrameContext",
    "TranscribeRequest",
    "TranscribeResult",
    "TranscribeSegment",
    "TranscribeWord",
    "WhisperRunner",
    "extension_from_content_type",
    "stream_url_to_temp_file",
    "whisper_transcribe_url_stream",
]
