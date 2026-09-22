import logging
import os
from typing import Any

from abstract_essentials import derive_media_type


from hugpy_media.whisper_model.src.model.utils.audio import (
    extract_audio_from_video,
    extract_audio_from_video_ffmpeg,
    get_audio_path,
)
from hugpy_media.whisper_model.src.model.utils.files.artifacts.workspace import create_media_workspace
from hugpy_media.whisper_model.src.model.utils.files.frames.extract import extract_context_frames_from_whisper
from hugpy_media.whisper_model.src.model.utils.files.save import save_transcript_outputs

logger = logging.getLogger(__name__)
from hugpy_media.whisper_model.src.model.model import get_whisper_model, WHISPER_TRANSCRIBE_LOCK
def whisper_transcribe(
    audio_path: str,
    model_size: str = "base",
    language: str | None = "english",
    task: str = "transcribe",
    whisper_model_path: str | None = None,
    word_timestamps: bool = False,
) -> dict[str, Any]:
    if not os.path.isfile(audio_path):
        raise ValueError(f"Audio file does not exist: {audio_path}")

    if task not in {"transcribe", "translate"}:
        raise ValueError(f"Unsupported Whisper task: {task}")

    model = get_whisper_model(
        module_size=model_size,
        whisper_model_path=whisper_model_path,
    )

    options: dict[str, Any] = {"task": task}

    if language:
        options["language"] = language

    # oracle k98 (audio.transcribe.word_timestamps): the last hop of the
    # passthrough chain — openai-whisper's own transcribe() accepts this flag
    # and, when set, returns per-word timing on each segment.
    if word_timestamps:
        options["word_timestamps"] = True

    # Single-flight: only one CPU inference runs at a time. With no swap on this
    # box, stacking concurrent transcribes would multiply anon RAM and saturate
    # the CPU; serializing keeps each request honest.
    with WHISPER_TRANSCRIBE_LOCK:
        return model.transcribe(audio_path, **options)
def transcribe_from_video(
    video_path: str,
    audio_path: str | None = None,
    model_size: str = "base",
    language: str | None = "english",
    task: str = "transcribe",
    whisper_model_path: str | None = None,
) -> dict[str, Any]:
    extracted_audio_path = extract_audio_from_video(
        video_path=video_path,
        audio_path=audio_path,
    )

    return whisper_transcribe(
        audio_path=extracted_audio_path,
        model_size=model_size,
        language=language,
        task=task,
        whisper_model_path=whisper_model_path,
    )


def transcribe_file(
    file_path: str,
    model_size: str = "base",
    language: str | None = "english",
    task: str = "transcribe",
    whisper_model_path: str | None = None,
    audio_path: str | None = None,
) -> dict[str, Any]:
    if not os.path.isfile(file_path):
        raise ValueError(f"Media file does not exist: {file_path}")

    media_type = derive_media_type(file_path)

    if media_type == "audio":
        return whisper_transcribe(
            audio_path=file_path,
            model_size=model_size,
            language=language,
            task=task,
            whisper_model_path=whisper_model_path,
        )

    if media_type == "video":
        audio_path = get_audio_path(file_path, audio_path)
        return transcribe_from_video(
            video_path=file_path,
            audio_path=audio_path,
            model_size=model_size,
            language=language,
            task=task,
            whisper_model_path=whisper_model_path,
        )

    raise ValueError(f"Unsupported media type for transcription: {media_type}")


def transcribe_file_with_workspace(
    file_path: str,
    model_size: str = "base",
    language: str | None = "english",
    task: str = "transcribe",
    whisper_model_path: str | None = None,
    audio_path: str | None = None,
    output_root: str | None = None,
    copy_source: bool = False,
    capture_frames: bool = False,
    min_gap_seconds: float = 2.0,
    long_segment_seconds: float = 8.0,
    word_timestamps: bool = False,
) -> dict[str, Any]:
    """
    Main media transcription pipeline.

    Creates a workspace directory, extracts audio if needed, transcribes it,
    saves transcript artifacts, optionally extracts context frames, and writes
    a manifest.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"Media file does not exist: {file_path}")

    workspace = create_media_workspace(
        source_path=file_path,
        output_root=output_root,
        copy_source=copy_source,
    )

    media_type = derive_media_type(file_path)
    source_video_path: str | None = None

    if media_type == "audio":
        resolved_audio_path = file_path
        workspace.manifest.set_file("audio", resolved_audio_path)

    elif media_type == "video":
        source_video_path = file_path
        resolved_audio_path = audio_path or workspace.audio_path

        extract_audio_from_video_ffmpeg(
            video_path=file_path,
            audio_path=resolved_audio_path,
        )

        workspace.manifest.set_file("audio", resolved_audio_path)

    else:
        raise ValueError(f"Unsupported media type for transcription: {media_type}")

    whisper_result = whisper_transcribe(
        audio_path=resolved_audio_path,
        model_size=model_size,
        language=language,
        task=task,
        whisper_model_path=whisper_model_path,
        word_timestamps=word_timestamps,
    )

    save_transcript_outputs(
        whisper_result=whisper_result,
        transcript_json_path=workspace.transcript_json_path,
        transcript_text_path=workspace.transcript_text_path,
    )

    frames: list[dict[str, Any]] = []

    if capture_frames:
        if media_type != "video" or not source_video_path:
            logger.warning("capture_frames=True ignored because source media is not video")
        elif extract_context_frames_from_whisper is None:
            raise ImportError(
                "capture_frames=True requires .video_context.extract_context_frames_from_whisper"
            )
        else:
            frames_dir = workspace.frames_dir

            frames = extract_context_frames_from_whisper(
                video_path=source_video_path,
                whisper_result=whisper_result,
                output_dir=frames_dir,
                min_gap_seconds=min_gap_seconds,
                long_segment_seconds=long_segment_seconds,
            )

            workspace.manifest.set_file("frames_dir", frames_dir)
            workspace.manifest.set_file("frame_context", workspace.frame_context_path)
            workspace.manifest.set_metadata("frame_count", len(frames))

    workspace.manifest.set_file("transcript_json", workspace.transcript_json_path)
    workspace.manifest.set_file("transcript_text", workspace.transcript_text_path)
    workspace.manifest.set_metadata("media_type", media_type)
    workspace.manifest.set_metadata("model_size", model_size)
    workspace.manifest.set_metadata("language", language)
    workspace.manifest.set_metadata("task", task)
    workspace.manifest.save()

    return {
        "workspace_dir": workspace.root_dir,
        "audio_path": resolved_audio_path,
        "transcript_json_path": workspace.transcript_json_path,
        "transcript_text_path": workspace.transcript_text_path,
        "manifest_path": os.path.join(workspace.root_dir,"manifest.json"),
        "frames": frames,
        "whisper_result": whisper_result
    }
