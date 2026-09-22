"""Explicit environment wiring, loud on missing (INV-5). One source of truth,
no smart defaults. Missing STUDIO_MASTER_COLORSPACE fails at boot with a named
error listing EVERY missing var - not at frame 900 with a green tint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from hugpy_video.intel.studio.errors import ConfigError

# (attribute, env var, caster). Required - no defaults.
_REQUIRED: tuple[tuple[str, str, type], ...] = (
    ("output_root", "STUDIO_OUTPUT_ROOT", str),
    ("weights_root", "STUDIO_WEIGHTS_ROOT", str),
    ("manifest_root", "STUDIO_MANIFEST_ROOT", str),
    ("master_colorspace", "STUDIO_MASTER_COLORSPACE", str),
    ("master_fps", "STUDIO_MASTER_FPS", int),
    ("max_vram_gb", "STUDIO_MAX_VRAM_GB", float),
    ("loudness_target_lufs", "STUDIO_LOUDNESS_LUFS", float),
)


@dataclass(frozen=True, slots=True)
class StudioEnv:
    output_root: str
    weights_root: str
    manifest_root: str
    master_colorspace: str
    master_fps: int
    max_vram_gb: float
    loudness_target_lufs: float
    allow_unpinned: bool

    def to_snapshot(self) -> tuple[tuple[str, str], ...]:
        """FIX-5: the resolved env, as sorted (env_var, value) string pairs, for
        RenderManifest.env_snapshot. Manifest builders MUST source env_snapshot
        from here (single source of truth) rather than hand-filling it, so the
        recorded env matches what actually resolved at boot (INV-1/INV-5)."""
        pairs = tuple(
            (var, str(getattr(self, attr))) for attr, var, _caster in _REQUIRED
        ) + (("STUDIO_ALLOW_UNPINNED", "1" if self.allow_unpinned else "0"),)
        return tuple(sorted(pairs))


def load_env() -> StudioEnv:
    missing: list[str] = []
    bad: list[str] = []
    values: dict[str, object] = {}

    for attr, var, caster in _REQUIRED:
        raw = os.environ.get(var)
        if raw is None or raw == "":
            missing.append(var)
            continue
        try:
            values[attr] = caster(raw)
        except (TypeError, ValueError):
            bad.append(f"{var}={raw!r} (expected {caster.__name__})")

    if missing or bad:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if bad:
            parts.append("uncastable: " + "; ".join(bad))
        raise ConfigError("environment wiring incomplete -> " + " | ".join(parts))

    values["allow_unpinned"] = os.environ.get("STUDIO_ALLOW_UNPINNED") == "1"
    return StudioEnv(**values)  # type: ignore[arg-type]


def require_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise ConfigError(f"required directory does not exist: {path}")
    return path
