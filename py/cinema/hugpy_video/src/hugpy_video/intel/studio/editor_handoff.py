"""Studio -> Editor handoff — GUARANTEE a Filmora-native MP4 for the operator's
LAN editing folder (k12).

The Studio "Send to Editor" flow hands a produced clip into a stable inbox
(``EDITOR_INBOX_ROOT``, a sibling of clips/) that the operator's Windows
workstation (Filmora desktop) picks up over the LAN. Filmora opens H.264 /
yuv420p 8-bit MP4 cleanly; some studio renders (and future runners) can land
other codecs / pixel formats. This module is the FORMAT GUARANTEE:

  * COPY branch — the source is ALREADY H.264 / yuv420p / mp4: a lossless,
    near-instant stream-copy REMUX with ``+faststart`` (front-loads the moov
    atom so the editor opens the file without scanning to the end);
  * TRANSCODE branch — anything else: re-encode into the safe H.264 yuv420p +
    AAC envelope, also ``+faststart``.

House discipline (mirrors ``runners/ffmpeg_crop.py`` + ``media_store._ffprobe``):
the same ``resolve_bin`` subprocess idiom; EXPECTED failures (a bad probe, a
nonzero ffmpeg, a missing/empty output) are returned as DATA (a ``HandoffResult``
with ``ok=False``), never raised — the caller (route) maps a False result to a
clean 500. os.path only.

This module is PURELY ADDITIVE. It is invoked ONLY at handoff time by the
``POST /video/studio/clip/<id>/to-editor`` route; it never touches the render
pipeline (``runners/studio_i2v.py``, ``produce.py``, ``manifest.py``, …).
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from hugpy_platform.binaries import resolve_bin
from hugpy_platform.utils import slugify

# The Filmora-native envelope: an H.264 video stream in 8-bit yuv420p, inside an
# MP4 container. A source matching all three needs only a REMUX (stream copy);
# anything else is TRANSCODED into this envelope.
_TARGET_VCODEC = "h264"
_TARGET_PIX_FMT = "yuv420p"

# Bound the filename slug so a long prompt can't yield an absurd filename. The
# short job-id suffix (see editor_filename) keeps two clips with the same prompt
# from colliding on the stem.
_SLUG_MAX_CHARS = 60


@dataclass(frozen=True)
class HandoffResult:
    """Outcome of an editor handoff — errors-as-data. Mirrors the JobError-ish
    shape the runners use (ok + code/message) WITHOUT being an actual
    JobResult/media-bus job, since a handoff is a synchronous route action, not a
    bus-dispatched render."""
    ok: bool
    dest: Optional[str] = None
    # "copy" (remux, stream-copied) or "transcode" (re-encoded) — which branch ran.
    mode: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None


def editor_filename(title: Optional[str], job_id: str) -> str:
    """Build a Windows-safe inbox filename from a render's title + job id.

    ``slugify`` (imports/src/utils) already maps every Windows-illegal character
    (``: < > " / \\ | ? *`` and control chars — none of them are in ``[\\w.\\- ]``)
    to ``_`` and strips edge punctuation, so the result is a clean, legal stem.
    The ``_<job_id[:8]>`` suffix (mirrors the frontend ``shortId`` 8-char
    convention) makes the name collision-resistant across DIFFERENT clips that
    share a prompt, and incidentally keeps a reserved-device title (``con``,
    ``nul`` …) from ever being the exact stem. Same-clip RE-SENDS are handled by
    the route's ``unique_path`` (numeric suffix), not here."""
    base = title if (isinstance(title, str) and title.strip()) else "clip"
    slug = slugify(base[:_SLUG_MAX_CHARS], fallback="clip")
    short = (job_id if isinstance(job_id, str) else "")[:8] or "job"
    return f"{slug}_{short}.mp4"


def _ffprobe(path: str) -> dict:
    """Probe a media file to JSON (format + streams). Same resolve_bin + flags as
    ``media_store._ffprobe``; raises RuntimeError on a nonzero probe so the caller
    turns it into a clean ``HandoffResult(ok=False)``."""
    ffprobe = resolve_bin("ffprobe") or "ffprobe"
    command = [
        ffprobe,
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


def _is_filmora_native(probe: dict) -> bool:
    """True iff the probed media is ALREADY Filmora-safe: an H.264 video stream in
    yuv420p, inside an mp4 container. These are exactly the three properties the
    COPY branch depends on — any miss routes to the TRANSCODE branch."""
    fmt = probe.get("format") or {}
    # ffmpeg reports the mp4 container as the shared muxer family
    # "mov,mp4,m4a,3gp,3g2,mj2"; mp4 is native iff "mp4" is one of those tokens.
    format_name = fmt.get("format_name") or ""
    if "mp4" not in [t.strip() for t in format_name.split(",")]:
        return False
    streams = probe.get("streams") or []
    video = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"),
        None)
    if not isinstance(video, dict):
        return False
    if (video.get("codec_name") or "").lower() != _TARGET_VCODEC:
        return False
    if (video.get("pix_fmt") or "").lower() != _TARGET_PIX_FMT:
        return False
    return True


def plan_handoff(probe: dict) -> str:
    """Decide the branch from a probe: ``"copy"`` (already Filmora-native → remux)
    or ``"transcode"`` (re-encode into the safe envelope). Split out from
    ``send_to_editor`` so it is unit-testable against fabricated ffprobe JSON
    without spawning ffmpeg."""
    return "copy" if _is_filmora_native(probe) else "transcode"


def _tail(text: str, lines: int = 20) -> str:
    """Last ``lines`` lines of an ffmpeg stderr, for a compact error message."""
    if not text:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def send_to_editor(src: str, dest: str) -> HandoffResult:
    """Produce a Filmora-native MP4 at ``dest`` from the studio clip at ``src``.

    COPY branch (source already H.264/yuv420p/mp4): stream-copy remux with
    ``+faststart`` — lossless, near-instant, moov atom front-loaded.
    TRANSCODE branch (anything else): re-encode to H.264 yuv420p + AAC,
    ``+faststart``.

    Every EXPECTED failure — missing input, a bad probe, a nonzero ffmpeg, or a
    missing/empty output — is returned as ``HandoffResult(ok=False, code=…,
    message=…)``, never raised. A partial output from a failed encode is removed
    so a failed handoff never leaves a truncated file in the operator's inbox."""
    if not (isinstance(src, str) and os.path.isfile(src)):
        return HandoffResult(ok=False, code="missing_input",
                             message=f"source clip does not exist: {src}")

    try:
        probe = _ffprobe(src)
    except (RuntimeError, ValueError) as exc:
        return HandoffResult(ok=False, code="probe_failed", message=str(exc))

    mode = plan_handoff(probe)

    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    if mode == "copy":
        command = [ffmpeg, "-y", "-i", src,
                   "-c", "copy", "-movflags", "+faststart", dest]
    else:
        command = [ffmpeg, "-y", "-i", src,
                   "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                   "-pix_fmt", "yuv420p", "-c:a", "aac",
                   "-movflags", "+faststart", dest]

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        # Never leave a truncated file behind in the inbox.
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except OSError:
            pass
        return HandoffResult(ok=False, mode=mode, code="ffmpeg_failed", message=(
            f"ffmpeg exited {result.returncode}.\n"
            f"cmd: {' '.join(command)}\n"
            f"stderr:\n{_tail(result.stderr)}"))

    if not os.path.isfile(dest) or os.path.getsize(dest) == 0:
        return HandoffResult(ok=False, mode=mode, code="missing_output",
                             message=f"ffmpeg reported success but produced no output at {dest}")

    return HandoffResult(ok=True, dest=dest, mode=mode)
