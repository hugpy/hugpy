"""Assist MEDIA PRETEXT — KEEPER-TASK k93 §C.

``POST /video/prompt/assist`` (detail / generate / spread) accepts an optional

    "context": {"media": {"uri": "<library uri>", "mime": "video/mp4|image/*",
                          "label": "optional"}}

The media is DESCRIBED server-side by a vision model and the description is
prepended to the assist prompt as pretext ("Reference video: …"), so a group
generate/enhance can be steered by a clip the operator attached.

Three rules this module keeps:

* **Reuse, never re-implement.** Video frames come out of the SAME frame-extract
  runner ``/video/jobs/frame_extract`` enqueues (``ffmpeg_frames.run_frame_extract``
  — one ffmpeg ``fps=`` pass, frames re-ingested into the library). Descriptions
  go through the SAME chat plane every other assist call uses (``execute_prompt``
  with ``task="image-text-to-text"`` + ``file=``, exactly the movie vision judge's
  call shape) — never a worker URL, never an in-process model load on central.
* **Absent → byte-identical.** ``validate_media`` returns ``None`` for a context
  without ``media`` and the route does nothing else; no prompt text, no log field,
  no response key changes.
* **Cached per uri.** generate → enhance on the same clip must not describe it
  twice. Module-level LRU keyed on (realpath, mtime, size, model) so a replaced
  file under the same name is re-described.

Flask-free on purpose (the route passes the jail resolver + the executor in), so
the sampling and the prepend are unit-testable with a fake vision call.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "MediaError",
    "MAX_FRAMES",
    "validate_media",
    "describe_media",
    "render_pretext",
    "prepend_pretext",
    "cache_clear",
]

# ≤ 4 evenly spaced frames per video (task §C).
MAX_FRAMES = 4
# Per-frame description budget. Small on purpose: four frames x 120 tokens is
# pretext, not the prompt — and a no-think vision reply of this size is a
# sentence or three, which is what "Reference video:" wants.
DESCRIBE_MAX_TOKENS = 120
# Frame format for the describe pass — a modest jpg is all a VLM needs.
_FRAME_FMT = "jpg"
_FRAME_QUALITY = 80

_CACHE_MAX = 64
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


class MediaError(ValueError):
    """A caller-fixable (``status`` 400/404) or fleet (502) media problem. The
    route maps it to ``jsonify({"error": str(exc)}), exc.status`` — the same
    envelope every other assist validation error uses."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_media(context: Any) -> Optional[Dict[str, str]]:
    """Return the normalized ``{"uri", "mime", "label"}`` or ``None`` when
    ``context.media`` is absent. Raises ``MediaError`` (400) on a malformed
    block, in the same ``context.<field> must be …`` style ``context.kind`` uses.
    """
    if not isinstance(context, dict):
        return None
    media = context.get("media")
    if media is None:
        return None
    if not isinstance(media, dict):
        raise MediaError("context.media must be an object with string uri and mime")
    uri = media.get("uri")
    mime = media.get("mime")
    if not isinstance(uri, str) or not uri.strip():
        raise MediaError("context.media.uri must be a non-empty string")
    if not isinstance(mime, str) or not mime.strip():
        raise MediaError("context.media.mime must be a non-empty string")
    mime = mime.strip().lower()
    if not (mime.startswith("video/") or mime.startswith("image/")):
        raise MediaError("context.media.mime must be video/* or image/*")
    label = media.get("label")
    if label is not None and not isinstance(label, str):
        raise MediaError("context.media.label must be a string")
    return {"uri": uri.strip(), "mime": mime, "label": (label or "").strip()}


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def _cache_key(path: str, model_key: Optional[str]) -> tuple:
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size, model_key or "")
    except OSError:
        return (path, 0, 0, model_key or "")


def _cache_get(key: tuple) -> Optional[dict]:
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
        return hit


def _cache_put(key: tuple, value: dict) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


def cache_clear() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# --------------------------------------------------------------------------- #
# sampling + describing
# --------------------------------------------------------------------------- #
def _even_subsample(items: List[Any], n: int) -> List[Any]:
    """Pick ``n`` evenly spaced items (first always included)."""
    if len(items) <= n:
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def _sample_frames(ref, job_tag: str) -> List[Tuple[float, str]]:
    """≤ MAX_FRAMES evenly spaced frames of ``ref`` via the frame-extract runner.
    Returns ``[(t_seconds, frame_uri), …]``. Raises MediaError on failure."""
    from hugpy_video.intel.crop_schema import TemporalRegion
    from hugpy_video.intel.frame_schema import make_frame_extract
    from hugpy_video.intel.runners.ffmpeg_frames import run_frame_extract

    duration = ref.duration_s or 0.0
    if duration > 0:
        n = MAX_FRAMES if duration >= MAX_FRAMES * 0.25 else max(1, int(duration / 0.25))
        fps = n / duration                     # frames at 0, d/n, 2d/n, …
        window = None
        # ffmpeg's fps filter can round to one extra frame; cap generously and
        # subsample back down to n below rather than refusing the job.
        max_frames = n + 2
    else:
        # Unknown duration (odd container): a single frame from the first second.
        n = 1
        fps = 1.0
        window = TemporalRegion(start_s=0.0, end_s=1.0)
        max_frames = 2
    try:
        spec = make_frame_extract(source=ref, fps=fps, quality=_FRAME_QUALITY,
                                  fmt=_FRAME_FMT, window=window, max_frames=max_frames)
    except (ValueError, TypeError) as exc:
        raise MediaError(f"context.media: cannot sample frames: {exc}", 400)
    res = run_frame_extract(spec, f"assist-{job_tag}")
    if not res.ok or not res.outputs:
        err = getattr(res, "error", None)
        msg = getattr(err, "message", None) or "frame extraction produced nothing"
        raise MediaError(f"context.media: frame extraction failed: {msg}", 502)
    frames = _even_subsample(list(res.outputs), n)
    spacing = duration / n if duration > 0 else 0.0
    return [(round(i * spacing, 2), fr.uri) for i, fr in enumerate(frames)]


def _result_text(res) -> str:
    if isinstance(res, dict):
        return res.get("text") or ""
    txt = getattr(res, "text", None)
    if txt:
        return txt
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(res, attr, None)
        if callable(fn):
            try:
                d = fn()
            except TypeError:
                continue
            if isinstance(d, dict) and d.get("text"):
                return d["text"]
    return ""


def _describe_one(execute: Callable[..., Any], image_uri: str, model_key: str,
                  what: str) -> str:
    """One vision call through the shared chat plane. Returns the think-stripped
    prose; raises MediaError(502) when the plane fails or answers empty."""
    from hugpy_engine.utils.no_think import with_no_think, strip_think

    prompt = (
        f"Describe this {what} for a prompt writer in two or three sentences: "
        "subject, setting, lighting, colour palette, camera framing, and any "
        "motion or action visible. Plain prose, no preamble, no lists."
    )
    try:
        res = execute(
            model_key=model_key,
            task="image-text-to-text",
            file=image_uri,
            prompt=with_no_think(prompt),
            max_new_tokens=DESCRIBE_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 — mapped to an honest 502 by the route
        raise MediaError(f"context.media: vision model {model_key!r} unavailable: "
                         f"{type(exc).__name__}: {exc}", 502)
    ok = res.get("ok", True) if isinstance(res, dict) else getattr(res, "ok", True)
    if not ok:
        err = res.get("error") if isinstance(res, dict) else getattr(res, "error", None)
        raise MediaError(f"context.media: vision model {model_key!r} failed: "
                         f"{err or 'not ok'}", 502)
    raw = (_result_text(res) or "").strip()
    prose, reasoning = strip_think(raw)
    text = (prose or reasoning or "").strip()
    if not text:
        raise MediaError(f"context.media: vision model {model_key!r} returned no "
                         "description", 502)
    return " ".join(text.split())


def describe_media(media: Dict[str, str], *, jail_resolve: Callable[[str], Optional[str]],
                   execute: Callable[..., Any], model_key: Optional[str] = None,
                   owner: Optional[str] = None) -> dict:
    """Resolve, sample, describe (or serve from cache).

    Returns ``{"uri", "kind", "label", "model", "cached", "frames":
    [{"t", "uri", "description"}], "pretext"}``. Raises ``MediaError`` with the
    HTTP status the route should return.
    """
    from hugpy_video.intel import media_store

    path = jail_resolve(media["uri"])
    if path is None:
        raise MediaError("context.media.uri outside storage jail", 400)
    if not os.path.isfile(path):
        raise MediaError("context.media.uri not found", 404)

    if not model_key:
        from hugpy_engine.config.models.models_default import DEFAULT_VISION_MODEL
        model_key = DEFAULT_VISION_MODEL

    key = _cache_key(path, model_key)
    hit = _cache_get(key)
    if hit is not None:
        out = dict(hit)
        out["cached"] = True
        out["label"] = media.get("label") or hit.get("label") or ""
        out["pretext"] = render_pretext(out)
        return out

    try:
        ref = media_store.ingest(path, owner=owner)
    except Exception as exc:  # noqa: BLE001 — unreadable / not media = caller error
        raise MediaError(f"context.media.uri is not a readable media file: {exc}", 400)
    want_video = media["mime"].startswith("video/")
    if want_video and ref.kind != "video":
        raise MediaError(f"context.media.mime says video but the file is {ref.kind}", 400)
    if not want_video and ref.kind != "image":
        raise MediaError(f"context.media.mime says image but the file is {ref.kind}", 400)

    tag = hashlib.sha1(f"{path}|{key[1]}|{key[2]}".encode("utf-8")).hexdigest()[:12]
    frames: List[dict] = []
    if ref.kind == "video":
        for t, furi in _sample_frames(ref, tag):
            frames.append({"t": t, "uri": furi,
                           "description": _describe_one(execute, furi, model_key,
                                                        "video frame")})
    else:
        frames.append({"t": 0.0, "uri": ref.uri,
                       "description": _describe_one(execute, ref.uri, model_key, "image")})

    out = {"uri": media["uri"], "kind": ref.kind, "label": media.get("label") or "",
           "model": model_key, "cached": False, "frames": frames}
    out["pretext"] = render_pretext(out)
    _cache_put(key, out)
    return dict(out)


# --------------------------------------------------------------------------- #
# rendering + prepend
# --------------------------------------------------------------------------- #
def render_pretext(info: dict) -> str:
    """The "Reference video: …" paragraph. One line per sampled frame for a
    video; a single sentence for an image."""
    label = info.get("label") or ""
    head = "Reference video" if info.get("kind") == "video" else "Reference image"
    if label:
        head += f' ("{label}")'
    frames = info.get("frames") or []
    if info.get("kind") == "video":
        n = len(frames)
        bits = [f"{head}: {n} frame{'s' if n != 1 else ''} sampled evenly across the clip."]
        for i, fr in enumerate(frames, 1):
            bits.append(f"Frame {i} (t={fr['t']:g}s): {fr['description']}")
        body = " ".join(bits)
    else:
        body = f"{head}: {frames[0]['description']}" if frames else f"{head}: (no description)"
    return (body + " Use this as the visual pretext: match its subject, setting, "
            "palette and mood unless the instructions below say otherwise.")


def prepend_pretext(user: str, pretext: str) -> str:
    """Pretext FIRST, then the instruction — the model reads the reference
    before it is told what to do with it. Empty pretext returns ``user`` as-is."""
    if not pretext:
        return user
    return f"{pretext}\n\n{user}"
