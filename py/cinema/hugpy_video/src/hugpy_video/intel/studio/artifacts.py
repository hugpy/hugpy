"""Render artifacts — the frozen result a runner returns (INV-6).

An ``Artifact`` is the content-addressed handle to a produced clip. It is a value
(frozen dataclass), never a live file handle: ``path`` points at the on-disk mp4,
``content_hash`` is the manifest hash that addresses its directory, and
``resumed`` records whether this call regenerated the pixels or found an existing
non-empty clip and skipped straight to returning it (INV-6 resume).

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str                # absolute path to the produced clip.mp4
    content_hash: str        # manifest.content_hash() — addresses the output dir
    frames: int
    width: int
    height: int
    duration_s: float
    resumed: bool = False     # True => existing non-empty clip found, not regenerated
