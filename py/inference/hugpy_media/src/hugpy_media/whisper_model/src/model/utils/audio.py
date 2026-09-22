import logging
import os


logger = logging.getLogger(__name__)
import subprocess
def get_audio_path(video_path: str | None = None, audio_path: str | None = None) -> str:
    if audio_path is None:
        if not video_path:
            raise ValueError("video_path is required when audio_path is not provided")

        video_directory = os.path.dirname(video_path) or "."
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(video_directory, f"{base_name}.wav")

    if os.path.isdir(audio_path):
        if not video_path:
            raise ValueError("video_path is required when audio_path is a directory")

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(audio_path, f"{base_name}.wav")

    return audio_path

def extract_audio_from_video_ffmpeg(
    video_path: str,
    audio_path: str,
) -> str:
    if not os.path.isfile(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")

    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)

    from hugpy_platform.binaries import resolve_bin
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"   # .exe on Windows; bare name as last resort
    command = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]

    logger.info(f"Extracting audio from {video_path} to {audio_path}")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to extract audio.\n\n"
            f"Command:\n{' '.join(command)}\n\n"
            f"stderr:\n{result.stderr}"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file was not created: {audio_path}")

    return audio_path

def extract_audio_from_video(
    video_path: str,
    audio_path: str | None = None,
) -> str:
    audio_path = get_audio_path(video_path, audio_path)

    return extract_audio_from_video_ffmpeg(
        video_path=video_path,
        audio_path=audio_path,
    )
