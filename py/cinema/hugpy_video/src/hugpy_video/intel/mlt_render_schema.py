"""Headless MLT/Kdenlive RENDER job schema (k22) — the durable, JSON-safe intent for
"render an operator-authored .kdenlive/.mlt project server-side with ``melt``".

WORKFLOW (k22): the operator edits in Kdenlive on Windows against the Samba ``studio``
share (a share of ``/mnt/llm_storage/video_intel/studio`` on this VM), saves the
``.kdenlive`` project into the WRITABLE ``edits/`` subtree, and hugpy renders it here as an
``mlt_render`` media_bus job, writing the output back under ``edits/renders/``.

House style mirrors ``frame_schema`` / ``identity_video_extract_schema``: a frozen,
JSON-safe, validate-at-construction spec built ONLY via ``make_mlt_render``; the bus
rehydrates it through ``mlt_render_from_dict`` (reconstruct + RE-VALIDATE). Every field is
a primitive so ``asdict`` -> ``json`` round-trips cleanly.

A raise inside the factory / rehydrator is FINE — it is local to construction and never
crosses a module boundary (house discipline: a structurally-invalid spec is caller error
caught at the boundary). A raise inside the RUNNER is NOT — every expected failure there is
error-as-data (``JobResult(ok=False, JobError(...))``).

STORAGE-JAIL note: this schema only validates STRUCTURE (project_path absolute + a
project extension; output_rel a non-escaping relative path). The actual jail — project_path
must live under the studio tree, output must land under ``edits/renders/`` — is enforced by
the ROUTE (fast 400) AND re-checked by the RUNNER (the single authority that computes the
real paths), mirroring the house pattern (routes jail; the backbone re-checks).

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# The project extensions melt understands as a top-level project (a .kdenlive file IS an MLT
# XML document; .mlt is the same format). A tuple, not a set, so the error message is stable.
PROJECT_EXTS = (".kdenlive", ".mlt")

# Format defaults that WORK (defaults-are-promises doctrine): a widely-playable mp4 with
# H.264 video + AAC audio. Overridable per-spec; these are the success-path defaults.
DEFAULT_VCODEC = "libx264"
DEFAULT_ACODEC = "aac"
DEFAULT_CONTAINER = "mp4"


@dataclass(frozen=True)
class MltRenderSpec:
    """Frozen, JSON-safe currency of an ``mlt_render`` bus job.

        project_path   ABSOLUTE path to the ``.kdenlive`` / ``.mlt`` project to render. The
                       ROUTE + RUNNER jail it under the studio tree; here it is only checked
                       to be absolute with a project extension.
        output_rel     RELATIVE output path (basename or ``sub/dir/name.mp4``) resolved under
                       the studio ``edits/renders/`` dir by the runner. Must not escape (no
                       leading ``/``, no ``..`` component). Empty -> the runner derives
                       ``<project-stem>_<job8>.<container>``.
        width/height   OPTIONAL profile override (both or neither). None -> the runner
                       RESPECTS the project's embedded MLT ``<profile>`` (the correct default
                       — a Kdenlive project already carries its authored resolution). When
                       both are set the runner synthesizes a custom melt profile.
        fps            OPTIONAL frame-rate override (used with width/height's custom profile,
                       or alone atop the project profile's resolution). None -> project fps.
        profile        OPTIONAL named melt profile (e.g. ``atsc_1080p_25``) — passed to
                       ``melt -profile``. Takes precedence over width/height/fps when set.
        vcodec/acodec/container  format knobs for the avformat consumer. Defaults are the
                       H.264+AAC mp4 success path.
        vb             OPTIONAL video bitrate string (e.g. ``"8M"``) for the consumer. None
                       -> the encoder's own default (libx264 CRF).
        drive_letter   OPTIONAL Windows mapped-drive prefix (e.g. ``"Z:"``) the operator's
                       machine used for the share; the runner rewrites that drive form to the
                       studio ``edits/`` root in addition to the always-handled UNC forms.
    """
    project_path: str
    output_rel: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    profile: Optional[str] = None
    vcodec: str = DEFAULT_VCODEC
    acodec: str = DEFAULT_ACODEC
    container: str = DEFAULT_CONTAINER
    vb: Optional[str] = None
    drive_letter: Optional[str] = None


def _norm_rel(output_rel: Optional[str]) -> str:
    """Structural check for output_rel: empty is OK (runner derives a name); otherwise it
    must be a RELATIVE path with no escaping component. Returns the normalized value."""
    if output_rel is None:
        return ""
    if not isinstance(output_rel, str):
        raise ValueError(f"output_rel must be a string or None; got {type(output_rel).__name__}")
    rel = output_rel.strip().replace("\\", "/").lstrip("/")
    if not rel:
        return ""
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"output_rel must not contain '..' (jail escape); got {output_rel!r}")
    if not parts:
        return ""
    return "/".join(parts)


def make_mlt_render(
    *,
    project_path: str,
    output_rel: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    profile: Optional[str] = None,
    vcodec: str = DEFAULT_VCODEC,
    acodec: str = DEFAULT_ACODEC,
    container: str = DEFAULT_CONTAINER,
    vb: Optional[str] = None,
    drive_letter: Optional[str] = None,
) -> MltRenderSpec:
    """Validate every field and build the frozen ``MltRenderSpec``. Raises
    ``ValueError``/``TypeError`` LOCALLY on any structural violation (never across the bus)."""
    if not (isinstance(project_path, str) and project_path.strip()):
        raise ValueError(f"project_path must be a non-empty string; got {project_path!r}")
    project_path = project_path.strip()
    if not os.path.isabs(project_path):
        raise ValueError(f"project_path must be an absolute path; got {project_path!r}")
    if os.path.splitext(project_path)[1].lower() not in PROJECT_EXTS:
        raise ValueError(
            f"project_path must be a Kdenlive/MLT project ending in one of "
            f"{list(PROJECT_EXTS)}; got {project_path!r}")

    output_rel = _norm_rel(output_rel)

    # width/height: both-or-neither, positive ints when present.
    def _posint(name, v):
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} must be an int or None; got {v!r}")
        if v <= 0:
            raise ValueError(f"{name} must be positive; got {v!r}")
        return v
    width = _posint("width", width)
    height = _posint("height", height)
    if (width is None) != (height is None):
        raise ValueError("width and height must be given together (both or neither)")

    if fps is not None:
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError(f"fps must be a number or None; got {fps!r}")
        if fps <= 0:
            raise ValueError(f"fps must be positive; got {fps!r}")
        fps = float(fps)

    def _nonempty_str(name, v, default):
        if v is None:
            return default
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{name} must be a non-empty string; got {v!r}")
        return v.strip()
    vcodec = _nonempty_str("vcodec", vcodec, DEFAULT_VCODEC)
    acodec = _nonempty_str("acodec", acodec, DEFAULT_ACODEC)
    container = _nonempty_str("container", container, DEFAULT_CONTAINER)

    if profile is not None:
        if not isinstance(profile, str):
            raise ValueError(f"profile must be a string or None; got {type(profile).__name__}")
        profile = profile.strip() or None

    if vb is not None:
        if not isinstance(vb, str):
            raise ValueError(f"vb must be a string or None; got {type(vb).__name__}")
        vb = vb.strip() or None

    if drive_letter is not None:
        if not isinstance(drive_letter, str):
            raise ValueError(
                f"drive_letter must be a string or None; got {type(drive_letter).__name__}")
        dl = drive_letter.strip()
        if dl:
            if len(dl.rstrip(":")) != 1 or not dl[0].isalpha():
                raise ValueError(
                    f"drive_letter must be a single drive letter like 'Z:' ; got {drive_letter!r}")
            drive_letter = dl.rstrip(":") + ":"
        else:
            drive_letter = None

    return MltRenderSpec(
        project_path=project_path,
        output_rel=output_rel,
        width=width,
        height=height,
        fps=fps,
        profile=profile,
        vcodec=vcodec,
        acodec=acodec,
        container=container,
        vb=vb,
        drive_letter=drive_letter,
    )


def mlt_render_from_dict(d: dict) -> MltRenderSpec:
    """Rebuild an ``MltRenderSpec`` from its ``asdict`` form THROUGH the validating factory
    (deserialize-then-revalidate, like every other bus spec). Registered in
    ``media_bus.SPEC_DESERIALIZERS`` under ``"mlt_render"``."""
    return make_mlt_render(
        project_path=d["project_path"],
        output_rel=d.get("output_rel"),
        width=d.get("width"),
        height=d.get("height"),
        fps=d.get("fps"),
        profile=d.get("profile"),
        vcodec=d.get("vcodec", DEFAULT_VCODEC),
        acodec=d.get("acodec", DEFAULT_ACODEC),
        container=d.get("container", DEFAULT_CONTAINER),
        vb=d.get("vb"),
        drive_letter=d.get("drive_letter"),
    )
