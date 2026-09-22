"""Boundary results — map §6 exactly.

A runner's outcome crosses a module boundary (bus <-> runner), so it is DATA,
never a raise. Expected failures become a JobError carried inside a JobResult.
Only the worker loop (media_bus.run_claimed) is allowed to catch an UNEXPECTED
raise and convert it to a JobResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from hugpy_video.intel.media_schema import MediaRef

# --- Task 2 (JobError class-collapse) --------------------------------------
# JobError is UNIFIED with comms.jobs.JobError: there is now exactly ONE JobError
# class in the tree, and this name re-binds to it, so every existing
# ``from ..result_schema import JobError`` site (the runners + media_bus)
# transparently constructs the canonical class. Direction is legal — video_intel
# already depends on comms (job_bridge) — and comms imports nothing from
# video_intel, so no cycle. The canonical class is a superset of the old shape
# (it adds a nullable ``detail`` between ``message`` and ``retryable``, plus
# ``to_dict``/``coerce``); every construction here passes ``code=/message=/
# retryable=`` as KEYWORDS, so the extra middle field stays inert and defaults to
# None. On /api/video/jobs (media_bus serializes JobResult via ``asdict``) the
# nested error therefore gains ``"detail": null`` — additive, backward-compatible.
from hugpy_control.jobs import JobError


# ARCHIVED — do NOT delete (operator rule: archive preexisting code, never delete).
# The original result_schema-local JobError, SUPERSEDED by comms.jobs.JobError in
# the Task 2 collapse. Retained for history only; nothing references this name —
# the live ``JobError`` above is the canonical comms class. Kept frozen exactly as
# it was.
@dataclass(frozen=True)
class _ArchivedResultSchemaJobError:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True)
class JobResult:
    job_id: str
    ok: bool
    outputs: Tuple[MediaRef, ...] = ()    # frame_extract -> many; crop/gen -> one
    error: Optional[JobError] = None      # data across the boundary, never a raise
    # auto-archive descriptor: {"name": str|None, "uuid": str, "dir": "assets/<meta>"}.
    # A plain dict (not a dataclass) so it serializes transparently via asdict.
    project: Optional[Dict[str, Any]] = None
    # generate_movie manifest: {"goals":[...], "segments":[...], ...} — the goal
    # timeline + per-segment director record. Plain dict; absent for non-movie jobs.
    movie: Optional[Dict[str, Any]] = None
    # char360 REVIEW-mode grouped manifest (CHARACTER-GROUPS-PLAN S1): the per-character
    # grouped views an ``identity_video_extract`` job with target="review" returns to the
    # UI to curate BEFORE any identity profile is written. Shape:
    #   {"n_characters": int, "groups": [{"char": str, "face_centroid": [float]|null,
    #                                     "views": [{"url","yaw","bin","score"}]}]}.
    # A plain dict (mirrors project/movie) so it serializes transparently via asdict and
    # reaches the client through GET /video/jobs/<id> -> result.groups. Absent (None) for
    # every non-review job — additive, backward-compatible.
    groups: Optional[Dict[str, Any]] = None
    # identity FROM-VIDEO manifest (k94): what ONE chained ``identity_from_video`` job
    # created — {"name": str, "n_characters": int, "profiles": [{"char","slug","name",
    # "glb": bool, "n_views": int, "canonical": int, "error"?}]} — so the UI can jump to
    # the minted profile(s). Plain dict (mirrors groups); None for every other job.
    identities: Optional[Dict[str, Any]] = None
