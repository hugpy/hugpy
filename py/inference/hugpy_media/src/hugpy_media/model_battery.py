"""model_battery — the shared per-session render/generation battery recorder.

One RUN-DIR per studio session under ``/srv/share/projects/hugpy/model-battery/
<YYYYMMDD-HHMM>/`` (fallback ``/mnt/llm_storage/model-battery/<ts>/`` when the
primary root is unwritable for the service user — the root actually used is the
first line of ``run.log``). Each run-dir holds:

  * ``results.json`` — a JSON ARRAY of rows ``{"model","axis","ok","secs","uri",
    "thumb_b64"}`` (+ additive ``error`` / ``ts``), the exact shape the 2026-07-04
    17-model battery established (``model-battery/20260704-1029/results.json``);
  * ``gallery.html`` — self-contained (thumbnails inlined as data: URIs);
  * ``run.log`` — one line per record + per-iteration aptitude lines.

Every append ATOMICALLY REWRITES ``results.json`` and ``gallery.html`` (tmp +
``os.replace``) so a reader never sees a torn file, and a crash mid-run leaves
every prior row intact.

ROBUSTNESS IS THE CONTRACT. This module instruments the render path; it must
never become the render path's failure mode. Every public entry point catches
``Exception`` and degrades to a no-op (``record`` returns False, ``run_for_session``
returns None, ``thumb_b64_for`` returns ""). Callers still wrap their own hook in
try/except (belt and braces) — but nothing here raises by design.

Shared by the studio spine (``video_intel/studio/produce.py`` — one row per
render iteration) and the diffusers imagegen runners
(``managers/imagegen/imagegen_runner.py`` — one row per image generation).
Deliberately dependency-free at import time: stdlib only; PIL/ffmpeg are lazy
and optional (no thumbnail is an empty ``thumb_b64``, never an error).

Levers:
  * ``HUGPY_MODEL_BATTERY=off|0|false|no`` — disable recording entirely.
  * ``HUGPY_MODEL_BATTERY_ROOT=<dir>`` — override the root (tests point this at
    a tmpdir; it REPLACES both the primary and the fallback).
"""
from __future__ import annotations

import base64
import html
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PRIMARY_ROOT = "/srv/share/projects/hugpy/model-battery"
FALLBACK_ROOT = "/mnt/llm_storage/model-battery"

_ENV_KILL = "HUGPY_MODEL_BATTERY"
_ENV_ROOT = "HUGPY_MODEL_BATTERY_ROOT"

# The ratified row schema (20260704-1029 battery). ``error`` and ``ts`` are the
# only additive keys — do not grow this without updating every reader.
ROW_KEYS = ("model", "axis", "ok", "secs", "uri", "thumb_b64")

_THUMB_MAX_EDGE = 192
_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")

# movie segments render under <movie_root>/segment_NN — one SESSION per movie,
# not one per segment (see session_key_for_out_root).
_SEGMENT_TAIL = re.compile(r"[/\\]segment_\d+$")


def enabled() -> bool:
    return (os.environ.get(_ENV_KILL) or "").strip().lower() not in (
        "off", "0", "false", "no")


def _roots() -> tuple[str, ...]:
    override = (os.environ.get(_ENV_ROOT) or "").strip()
    if override:
        return (override,)
    return (PRIMARY_ROOT, FALLBACK_ROOT)


def session_key_for_out_root(out_root: str) -> str:
    """Stable session key for a studio out_root: movie segments (…/segment_NN)
    collapse to their movie root so a whole movie is ONE battery session."""
    try:
        key = os.path.normpath(str(out_root or "default"))
        return _SEGMENT_TAIL.sub("", key) or "default"
    except Exception:
        return "default"


class BatteryRun:
    """One run-dir: an in-memory row list + the three files, atomically rewritten
    on every append. All methods are no-raise (False/None on failure)."""

    def __init__(self, run_dir: str, root_used: str) -> None:
        self.run_dir = run_dir
        self.root_used = root_used
        self.rows: list[dict] = []
        self._lock = threading.Lock()

    # -- files ---------------------------------------------------------------
    @property
    def results_path(self) -> str:
        return os.path.join(self.run_dir, "results.json")

    @property
    def gallery_path(self) -> str:
        return os.path.join(self.run_dir, "gallery.html")

    @property
    def log_path(self) -> str:
        return os.path.join(self.run_dir, "run.log")

    def log(self, message: str) -> bool:
        """Append one timestamped line to run.log. Never raises."""
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {message}\n")
            return True
        except Exception:
            logger.debug("battery run.log append failed (non-fatal)", exc_info=True)
            return False

    def record(
        self,
        *,
        model: str,
        axis: str,
        ok: bool,
        secs: float,
        uri: str = "",
        thumb_b64: str = "",
        error: str | None = None,
        ts: str | None = None,
    ) -> bool:
        """Append one row and atomically rewrite results.json + gallery.html.
        Returns False (never raises) on any failure — the render path must not
        care whether its telemetry landed."""
        try:
            row: dict = {
                "model": str(model),
                "axis": str(axis),
                "ok": bool(ok),
                "secs": round(float(secs), 2),
                "uri": str(uri or ""),
                "thumb_b64": str(thumb_b64 or ""),
                "ts": ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if error:
                row["error"] = str(error)[:2000]
            with self._lock:
                self.rows.append(row)
                self._rewrite_locked()
            status = "ok" if row["ok"] else f"FAIL ({row.get('error', 'unknown')})"
            self.log(f"{row['model']} · {row['axis']}: {status} {row['secs']}s {row['uri']}")
            return True
        except Exception:
            logger.debug("battery record failed (non-fatal)", exc_info=True)
            return False

    # -- internals -------------------------------------------------------------
    def _rewrite_locked(self) -> None:
        _atomic_write(self.results_path, json.dumps(self.rows, indent=1))
        _atomic_write(self.gallery_path, _render_gallery(self.rows, self.run_dir))


def _atomic_write(path: str, text: str) -> None:
    """tmp-in-same-dir + os.replace: a reader sees the old file or the new one,
    never a torn one."""
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _render_gallery(rows: list[dict], run_dir: str) -> str:
    """Self-contained gallery: thumbnails ride inline as data: URIs, so the file
    works copied anywhere with zero siblings."""
    cells = []
    n_ok = sum(1 for r in rows if r.get("ok"))
    for r in rows:
        thumb = r.get("thumb_b64") or ""
        img = (
            f'<img src="data:image/jpeg;base64,{thumb}" alt="">'
            if thumb else '<div class="noimg">no thumbnail</div>'
        )
        err = r.get("error") or ""
        cells.append(
            '<figure class="{cls}">{img}<figcaption><b>{model}</b> · {axis} · '
            "{verdict} · {secs}s<br><small>{uri}</small>{err}</figcaption></figure>".format(
                cls="ok" if r.get("ok") else "fail",
                img=img,
                model=html.escape(str(r.get("model", ""))),
                axis=html.escape(str(r.get("axis", ""))),
                verdict="ok" if r.get("ok") else "FAIL",
                secs=html.escape(str(r.get("secs", ""))),
                uri=html.escape(str(r.get("uri", ""))),
                err=(f"<br><small class=err>{html.escape(err)}</small>" if err else ""),
            )
        )
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>model battery — {html.escape(os.path.basename(run_dir))}</title>"
        "<style>"
        "body{font:14px system-ui;background:#111;color:#ddd;margin:16px}"
        "figure{display:inline-block;vertical-align:top;width:200px;margin:6px;"
        "padding:6px;background:#1c1c1c;border-radius:6px;border:1px solid #333}"
        "figure.fail{border-color:#a33}"
        "img{width:100%;border-radius:4px}"
        ".noimg{width:100%;height:120px;display:flex;align-items:center;"
        "justify-content:center;background:#222;color:#666;border-radius:4px}"
        "figcaption{margin-top:4px;word-break:break-all}"
        ".err{color:#e88}"
        "</style>"
        f"<h1>model battery</h1><p>{n_ok}/{len(rows)} ok · {html.escape(run_dir)}</p>"
        + "".join(cells)
    )


# --------------------------------------------------------------------------- #
# Session registry: one BatteryRun per (process, session key), created lazily on
# first record. A root that cannot be written falls through to the next; when
# every root refuses, the session maps to None (recording disabled, logged once).
# --------------------------------------------------------------------------- #
_runs: dict[str, "BatteryRun | None"] = {}
_runs_lock = threading.Lock()


def run_for_session(session_key: str = "default") -> "BatteryRun | None":
    """The session's BatteryRun, creating its run-dir on first use. Returns None
    (never raises) when disabled or when no root is writable."""
    try:
        if not enabled():
            return None
        key = str(session_key or "default")
        with _runs_lock:
            if key in _runs:
                return _runs[key]
            run = _create_run()
            _runs[key] = run
            return run
    except Exception:
        logger.debug("battery run_for_session failed (non-fatal)", exc_info=True)
        return None


def _create_run() -> "BatteryRun | None":
    stamp = time.strftime("%Y%m%d-%H%M")
    for root in _roots():
        try:
            os.makedirs(root, exist_ok=True)
            run_dir = _claim_run_dir(root, stamp)
            run = BatteryRun(run_dir, root)
            # record the root used FIRST, per the fallback contract.
            run.log(f"battery root: {root} (run dir {run_dir})")
            return run
        except Exception:
            logger.debug("battery root %s unusable, trying next", root, exc_info=True)
    logger.warning("model battery: no writable root among %s — recording disabled "
                   "for this session", _roots())
    return None


def _claim_run_dir(root: str, stamp: str) -> str:
    """<root>/<YYYYMMDD-HHMM>, suffixed -2, -3… when two sessions land in the
    same minute (makedirs(exist_ok=False) is the claim)."""
    for n in range(1, 100):
        candidate = os.path.join(root, stamp if n == 1 else f"{stamp}-{n}")
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not claim a run dir under {root}")


def reset_for_tests() -> None:
    """Forget every session→run mapping (tests only)."""
    with _runs_lock:
        _runs.clear()


# --------------------------------------------------------------------------- #
# Thumbnails — best-effort, empty string on ANY failure. Images via PIL; videos
# via one ffmpeg frame-grab piped through PIL. No thumbnail is a cosmetic loss,
# never an error.
# --------------------------------------------------------------------------- #
def thumb_b64_for(path: str, max_edge: int = _THUMB_MAX_EDGE) -> str:
    try:
        if not path or not os.path.isfile(path):
            return ""
        if path.lower().endswith(_VIDEO_EXTS):
            return _video_thumb(path, max_edge)
        return _image_thumb(path, max_edge)
    except Exception:
        logger.debug("battery thumbnail failed for %s (non-fatal)", path, exc_info=True)
        return ""


def _image_thumb(path: str, max_edge: int) -> str:
    from PIL import Image  # lazy: PIL missing => no thumbnail, not an error

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge))
        import io

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _video_thumb(path: str, max_edge: int) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return ""
    fd, tmp = tempfile.mkstemp(prefix="battery-thumb-", suffix=".jpg")
    os.close(fd)
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", path, "-frames:v", "1",
             "-vf", f"scale={max_edge}:-2", tmp],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if proc.returncode != 0 or not os.path.getsize(tmp):
            return ""
        with open(tmp, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


__all__ = [
    "PRIMARY_ROOT", "FALLBACK_ROOT", "ROW_KEYS",
    "BatteryRun", "enabled", "run_for_session", "session_key_for_out_root",
    "thumb_b64_for", "reset_for_tests",
]
