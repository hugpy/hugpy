"""Video-owned state roots, resolved once and injectable (PARTITION.md §state).

``hugpy-video`` is the sole owner of the media library jail, the media job bus,
the reservation ledger, identity profiles and studio sessions. Every path is
resolved here — from ``hugpy_platform`` config (process env, then ``.env`` via
``hugpy_platform.platform_facade.env_value``) with defaults under the
platform's ``DEFAULT_ROOT`` — so a deployment, a test or a CLI can repoint the
whole package without touching any other package's state.

Environment keys (all optional):

    HUGPY_VIDEO_STATE_DIR   base for the job bus + studio sessions
                            (default ``<DEFAULT_ROOT>/video_intel``)
    HUGPY_MEDIA_JOBS_DB     sqlite media job bus (default ``<state>/media_jobs.db``)
    HUGPY_RESERVATIONS_DB   sqlite GPU reservation ledger
                            (default ``<PROJECTS_HOME>/reservations.db``)
    STUDIO_OUTPUT_ROOT      studio sessions/renders (the studio's own strict
                            env — see ``intel.studio.env``; defaulted here only
                            for tooling such as ``hugpy-video jobs list``)
    UPLOADS_HOME / DEFAULT_ROOT / IDENTITIES_HOME / PROJECTS_HOME
                            platform storage roots (``hugpy_platform.constants``)

Programmatic override: :func:`configure_state` (tests, embedding) — pass the
fields you want, the rest keep resolving from config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple

__all__ = [
    "VideoStateRoots",
    "get_state_roots",
    "configure_state",
    "reset_state",
    "media_jobs_db_path",
    "reservations_db_path",
]


def _env(name: str) -> Optional[str]:
    try:
        from hugpy_platform.platform_facade import env_value
        val = env_value(name)
    except Exception:  # noqa: BLE001 — platform facade unavailable: raw env
        val = os.environ.get(name)
    val = (val or "").strip() if isinstance(val, str) else None
    return val or None


@dataclass(frozen=True)
class VideoStateRoots:
    state_dir: str            # HUGPY_VIDEO_STATE_DIR
    media_jobs_db: str        # HUGPY_MEDIA_JOBS_DB
    reservations_db: str      # HUGPY_RESERVATIONS_DB
    media_roots: Tuple[str, ...]   # jail for MediaRef paths (UPLOADS_HOME, DEFAULT_ROOT)
    identities_home: str      # IDENTITIES_HOME
    studio_sessions_root: str  # STUDIO_OUTPUT_ROOT


def _resolve() -> VideoStateRoots:
    from hugpy_platform.constants import (
        DEFAULT_ROOT, IDENTITIES_HOME, PROJECTS_HOME, UPLOADS_HOME,
    )

    state_dir = _env("HUGPY_VIDEO_STATE_DIR") or os.path.join(DEFAULT_ROOT, "video_intel")
    return VideoStateRoots(
        state_dir=state_dir,
        media_jobs_db=_env("HUGPY_MEDIA_JOBS_DB") or os.path.join(state_dir, "media_jobs.db"),
        reservations_db=_env("HUGPY_RESERVATIONS_DB") or os.path.join(str(PROJECTS_HOME), "reservations.db"),
        media_roots=(str(UPLOADS_HOME), str(DEFAULT_ROOT)),
        identities_home=str(IDENTITIES_HOME),
        studio_sessions_root=_env("STUDIO_OUTPUT_ROOT") or os.path.join(state_dir, "studio"),
    )


_override: Optional[VideoStateRoots] = None


def get_state_roots() -> VideoStateRoots:
    """The effective roots: a :func:`configure_state` override, else config."""
    return _override if _override is not None else _resolve()


def configure_state(**fields) -> VideoStateRoots:
    """Override some roots for this process (tests / embedding). Unknown
    fields raise; ``None`` values are ignored. Returns the effective roots."""
    global _override
    base = get_state_roots()
    clean = {k: v for k, v in fields.items() if v is not None}
    unknown = set(clean) - set(VideoStateRoots.__dataclass_fields__)
    if unknown:
        raise TypeError(f"unknown state fields: {sorted(unknown)}")
    _override = replace(base, **clean)
    return _override


def reset_state() -> None:
    """Drop the override; roots resolve from config again."""
    global _override
    _override = None


def media_jobs_db_path() -> str:
    return get_state_roots().media_jobs_db


def reservations_db_path() -> str:
    return get_state_roots().reservations_db
