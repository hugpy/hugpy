"""MediaStore ingest — resolve an asset's metadata ONCE and mint a MediaRef.

`ingest(path)` probes the file exactly once with ffprobe, determines the kind
authoritatively from the streams (probe wins; kind_hint only breaks ties), fills
width/height/duration/fps/sample_rate/channels, and returns an immutable
MediaRef. The resolved realpath is jailed under UPLOADS_HOME or DEFAULT_ROOT.

os.path only. ffprobe is invoked with the same subprocess idiom as
managers/whisper_model/.../audio.py (resolve_bin, PIPE, text, returncode check).
"""
from __future__ import annotations

import json
import mimetypes
import os
import subprocess
from typing import Optional
from uuid import uuid4

from hugpy_platform.binaries import resolve_bin
from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

from hugpy_video.intel.media_schema import MediaRef, make_media_ref

# Video streams whose codec is really a still-image codec. A "video" stream with
# one of these + no real duration / single frame is an IMAGE, not a video.
IMAGE_CODECS = frozenset({
    "png", "mjpeg", "bmp", "webp", "gif", "tiff", "jpeg", "jpg", "mjpg",
})


# --------------------------------------------------------------------------- #
# small typed coercions (ffprobe returns everything as strings)
# --------------------------------------------------------------------------- #
def _to_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # ffprobe uses "N/A" -> already excluded; guard NaN just in case
    if f != f:  # NaN
        return None
    return f


def _parse_fps(rate: Optional[str]) -> Optional[float]:
    """Parse ffprobe's avg_frame_rate "num/den" into a float. "0/0" -> None."""
    if not rate:
        return None
    if "/" in rate:
        num, _, den = rate.partition("/")
        n = _to_float(num)
        d = _to_float(den)
        if n is None or d is None or d == 0:
            return None
        return n / d
    return _to_float(rate)


# --------------------------------------------------------------------------- #
# storage jail
# --------------------------------------------------------------------------- #
def _is_within(path: str, root: str) -> bool:
    rp = os.path.realpath(path)
    rr = os.path.realpath(root)
    try:
        return os.path.commonpath([rp, rr]) == rr
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# ffprobe (resolved metadata, once)
# --------------------------------------------------------------------------- #
def _ffprobe(path: str) -> dict:
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
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe failed.\n\n"
            f"Command:\n{' '.join(command)}\n\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(result.stdout or "{}")


def _classify(streams, fmt: dict, kind_hint: Optional[str]) -> str:
    """Authoritative kind from the probe. Probe wins; kind_hint biases ties only."""
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if video_streams:
        vs = video_streams[0]
        codec = (vs.get("codec_name") or "").lower()
        nb_frames = _to_int(vs.get("nb_frames"))
        dur = _to_float(vs.get("duration"))
        if dur is None:
            dur = _to_float(fmt.get("duration"))
        is_image_codec = codec in IMAGE_CODECS

        # A real video: more than one frame, or a real duration on a non-image codec.
        if nb_frames is not None and nb_frames > 1:
            return "video"
        if dur is not None and dur > 0 and not is_image_codec:
            return "video"
        # A single-frame / image-codec stream with no real duration is an image.
        if is_image_codec:
            return "image"
        # Ambiguous (non-image codec, single/unknown frame, no duration).
        if kind_hint in ("image", "video"):
            return kind_hint
        return "video"

    if audio_streams:
        return "audio"

    # No recognizable A/V stream: fall back to the hint if plausible.
    if kind_hint in MEDIA_HINT_FALLBACK:
        return kind_hint
    raise ValueError("ffprobe found no image/audio/video stream to classify")


MEDIA_HINT_FALLBACK = frozenset({"image", "audio", "video"})


def _guess_mime(path: str, kind: str, streams) -> str:
    """mime from extension first, then a probe-derived fallback per kind."""
    guessed, _ = mimetypes.guess_type(path)
    if guessed:
        return guessed
    if kind == "image":
        vs = next((s for s in streams if s.get("codec_type") == "video"), {})
        codec = (vs.get("codec_name") or "").lower()
        sub = "jpeg" if codec in ("mjpeg", "jpg", "mjpg", "jpeg") else (codec or "octet-stream")
        return f"image/{sub}"
    if kind == "audio":
        return "audio/octet-stream"
    return "video/octet-stream"


def ingest(path: str, kind_hint: Optional[str] = None, sid: Optional[str] = None,
           owner: Optional[str] = None, private: bool = False) -> MediaRef:
    """Ingest an absolute file path already under a storage root -> MediaRef.

    Metadata is resolved exactly once via ffprobe. `kind_hint` may break a tie
    but the probe is authoritative.

    `sid` (the per-browser-TAB session id) is accepted for Phase 3 call-site
    compatibility and is still deliberately discarded: it identifies a tab, not
    a person, so it can never answer "whose asset is this".

    `owner` (2026-08-06) is the answer to that question — the central-account
    USERNAME, resolved by the caller in the request context (this module has no
    Flask coupling) and recorded on the MediaRef. None for every runner-produced
    ref (no request, no user) and for open-mode/self-hosted calls; the
    authoritative ownership record for a produced artifact is the media_bus
    row's `owner` column, which the /video routes filter on.

    `private` (t172) is the visibility SIBLING of `owner`, mirroring identity
    profiles' owner+private pair — DEFAULT False == PUBLIC. Carried onto the
    MediaRef so a runner-minted OUTPUT ref can hold the initiating request's
    private choice; like `owner` it is not the authoritative visibility record
    (the media_bus row's `private` column is, and the /video routes gate on it).

    Raises locally (FileNotFoundError / ValueError / RuntimeError) — ingest is
    a helper, not a boundary runner, so a raise here never crosses the job
    boundary as a raise. (The one place it is called from inside a runner, the
    runner has already verified the output exists.)
    """
    abspath = os.path.abspath(path)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(f"ingest: no such file: {abspath}")

    realpath = os.path.realpath(abspath)
    if not (_is_within(realpath, UPLOADS_HOME) or _is_within(realpath, DEFAULT_ROOT)):
        raise ValueError(
            f"ingest: path escapes storage jail: {realpath} "
            f"(allowed roots: {UPLOADS_HOME!r}, {DEFAULT_ROOT!r})"
        )

    probe = _ffprobe(abspath)
    streams = probe.get("streams") or []
    fmt = probe.get("format") or {}

    kind = _classify(streams, fmt, kind_hint)

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = height = None
    fps_native = None
    if video_stream is not None:
        width = _to_int(video_stream.get("width"))
        height = _to_int(video_stream.get("height"))
        if kind == "video":
            fps_native = _parse_fps(video_stream.get("avg_frame_rate"))

    sample_rate = channels = None
    if audio_stream is not None:
        sample_rate = _to_int(audio_stream.get("sample_rate"))
        channels = _to_int(audio_stream.get("channels"))

    # duration: format is authoritative; fall back to the primary stream.
    duration_s = _to_float(fmt.get("duration"))
    if duration_s is None:
        primary = (audio_stream if kind == "audio" else video_stream) or {}
        duration_s = _to_float(primary.get("duration"))
    # an image has no meaningful duration even if a still-image container reports one
    if kind == "image":
        duration_s = None

    mime = _guess_mime(abspath, kind, streams)

    return make_media_ref(
        asset_id=uuid4().hex,
        kind=kind,
        uri=abspath,
        mime=mime,
        width=width,
        height=height,
        duration_s=duration_s,
        fps_native=fps_native,
        sample_rate=sample_rate,
        channels=channels,
        owner=owner,
        private=bool(private),
    )
