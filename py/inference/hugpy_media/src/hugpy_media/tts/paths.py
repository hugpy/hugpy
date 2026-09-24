"""Shared output home for text-to-speech runners."""

import os

from hugpy_platform.constants import DEFAULT_ROOT


def output_dir() -> str:
    return os.path.join(DEFAULT_ROOT, "video_intel", "tts")
