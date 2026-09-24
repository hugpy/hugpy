"""One ffprobe invocation for media ingest and editor handoff."""

from __future__ import annotations

import json
import subprocess

from hugpy_platform.binaries import resolve_bin


def ffprobe(path: str) -> dict:
    ffprobe_bin = resolve_bin("ffprobe") or "ffprobe"
    command = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe failed.\n\n"
            f"Command:\n{' '.join(command)}\n\n"
            f"stderr:\n{result.stderr}")
    return json.loads(result.stdout or "{}")
