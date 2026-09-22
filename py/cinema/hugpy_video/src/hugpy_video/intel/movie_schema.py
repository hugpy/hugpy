"""Movie-generation schema — a GOAL TIMELINE rendered as contiguous SEGMENTS.

A movie is a SEQUENCE OF SEGMENTS. Each segment is one scene render (via the
extracted `runners.scene.render_scene_frames` core) that covers a half-open
``[start_frame, end_frame)`` slice of the movie's frame timeline and is driven by
its own goal prompt. The segments TILE the timeline exactly — contiguous and
non-overlapping — so ``total n_frames == max(end_frame)``.

A `GoalInterval` is one entry on that timeline: a frame range + the prompt the
segment should achieve, plus an optional `ref` MediaRef (an explicit start image
for that segment). The `MovieSpec` bundles the scene-template generation fields
(model/size/steps/…) shared by every segment, the ordered `goals`, and the
DIRECTOR knobs that turn on optional vision scoring + retry.

Mirrors scene_schema.py exactly: a frozen spec + a validating factory whose
raises are LOCAL to construction (never across a boundary), and asdict-friendly
serialization (nested MediaRef/GoalInterval round-trip through the bus).

Per-segment frame count is capped by the SAME FRAME_CAP as a scene (each segment
IS a scene render); the movie TOTAL is unbounded by design (movies are long).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from hugpy_video.intel.media_schema import MediaRef
from hugpy_video.intel.scene_schema import FRAME_CAP

# --------------------------------------------------------------------------- #
# legacy pixel chain gate (board or-k2 / proposal or-p1)
# --------------------------------------------------------------------------- #
# Chaining (frame i+1 conditioning on frame i; segment N+1 starting from segment
# N's last frame) is the LEGACY behaviour the oracle directive prohibits. It is
# OFF by default and only available when the operator opts in fleet-wide with
# HUGPY_LEGACY_CHAIN=1. Every runner reads the switch through ONE helper so the
# movie and scene paths can never disagree.
LEGACY_CHAIN_ENV = "HUGPY_LEGACY_CHAIN"
LEGACY_CHAIN_LABEL = "legacy (pixel-chained)"
_TRUTHY = ("1", "true", "yes", "on")


def legacy_chain_enabled() -> bool:
    """True only when HUGPY_LEGACY_CHAIN is set to 1/true/yes/on. Unset -> False."""
    return os.environ.get(LEGACY_CHAIN_ENV, "0").strip().lower() in _TRUTHY


def effective_chain(requested: bool) -> bool:
    """The chain value a runner may actually USE: the requested flag, forced to
    False unless the legacy switch is opted in."""
    return bool(requested) and legacy_chain_enabled()


@dataclass(frozen=True)
class GoalInterval:
    """One entry on the movie's goal timeline.

        start_frame  inclusive frame index (>= 0)
        end_frame    EXCLUSIVE frame index (> start_frame) — half-open range
        prompt       the goal this segment should achieve (non-empty)
        ref          optional explicit start image (MediaRef, kind='image') for
                     this segment; when absent the orchestrator carries the
                     previous segment's LAST frame for cross-segment drift.
    """
    start_frame: int
    end_frame: int
    prompt: str
    ref: Optional[MediaRef] = None
    # --- per-goal PROMPT-COMPONENT overrides (k92) ---
    # Each goal IS a prompt component (mirrors Scene's per-part settings): it may
    # carry its OWN generation knobs. ``None`` on any field means INHERIT the
    # movie-level (MovieSpec) value — so a movie with no overrides renders exactly
    # as before. Resolved per segment by ``goal_effective``. ``motion`` is
    # goal-only (MovieSpec has no shared motion), ``None`` = no schedule.
    model_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    negative: Optional[str] = None
    strength: Optional[float] = None
    chain: Optional[bool] = None
    motion: Optional[str] = None


@dataclass(frozen=True)
class MovieSpec:
    """Render a GOAL TIMELINE as a sequence of contiguous scene segments.

    Scene-template fields (shared by every segment) mirror GenerateSceneSpec:
        model_id, width, height, steps, guidance, fps, assemble, seed, negative,
        strength, chain, project.

    goals   the ORDERED, contiguous, non-overlapping tuple of GoalInterval that
            tiles ``[0, total)`` (total == max(end_frame)).

    Director knobs (optional vision scoring + retry):
        vision_enabled            score each segment's KEY frame before proceeding
        score_threshold           0..100; a take below it is "weak"
        max_attempts_per_segment  retry budget per segment (>=1)
        judge_model_id            optional model_key for the vision judge (else the
                                  plane's default image-text-to-text model)
        time_budget_s             optional wall-clock budget the runner owns itself
                                  (the single-daemon bus has no timeout/reaper).
    """
    # --- scene-template fields (shared by every segment) ---
    model_id: str
    width: int
    height: int
    steps: int
    guidance: float
    fps: int
    assemble: bool
    goals: Tuple[GoalInterval, ...]
    seed: Optional[int] = None
    negative: Optional[str] = None
    strength: Optional[float] = None
    chain: bool = False
    project: Optional[str] = None
    # --- director knobs ---
    vision_enabled: bool = False
    score_threshold: int = 60
    max_attempts_per_segment: int = 1
    judge_model_id: Optional[str] = None
    time_budget_s: Optional[int] = None


def make_movie(
    goals: Tuple[GoalInterval, ...],
    model_id: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    fps: int,
    assemble: bool,
    seed: Optional[int] = None,
    negative: Optional[str] = None,
    strength: Optional[float] = None,
    chain: bool = False,
    project: Optional[str] = None,
    vision_enabled: bool = False,
    score_threshold: int = 60,
    max_attempts_per_segment: int = 1,
    judge_model_id: Optional[str] = None,
    time_budget_s: Optional[int] = None,
) -> MovieSpec:
    """Validate + build a MovieSpec. Raises are LOCAL to construction.

    Goal-timeline invariants (the load-bearing ones):
      * goals non-empty;
      * each goal.start_frame is an int >= 0 and goal.end_frame > start_frame
        (half-open, non-empty range);
      * each goal.prompt is a non-empty string;
      * each goal.ref (when present) is a MediaRef of kind 'image';
      * each segment's frame count (end-start) is 1..FRAME_CAP (a segment IS a
        scene render, so it shares the scene cap);
      * the goals — IN THE GIVEN ORDER — are CONTIGUOUS and non-overlapping and
        tile ``[0, total)``: the first starts at 0, each next starts exactly where
        the previous ended, so total == max(end_frame).

    Scene-template invariants mirror make_generate_scene: truthy model_id;
    positive width/height/steps; fps >= 1; assemble bool; strength in [0,1] or
    None; chain bool. Director-knob invariants: score_threshold int in 0..100;
    max_attempts_per_segment int >= 1; vision_enabled bool; time_budget_s None or
    a positive int; judge_model_id None or a non-empty str.

    Also the reconstruction path used by the bus deserializer — goals are rebuilt
    into GoalInterval (their ref through make_media_ref) before this is called.
    """
    goals = tuple(goals)
    if not goals:
        raise ValueError("make_movie requires at least one GoalInterval")

    # ---- scene-template fields ----
    if not model_id:
        raise ValueError(f"model_id must be a non-empty model key; got {model_id!r}")
    if not (isinstance(width, int) and width > 0):
        raise ValueError(f"width must be a positive int; got {width!r}")
    if not (isinstance(height, int) and height > 0):
        raise ValueError(f"height must be a positive int; got {height!r}")
    if not (isinstance(steps, int) and steps > 0):
        raise ValueError(f"steps must be a positive int; got {steps!r}")
    if not (isinstance(fps, int) and fps >= 1):
        raise ValueError(f"fps must be an int >= 1; got {fps!r}")
    if not isinstance(assemble, bool):
        raise ValueError(f"assemble must be a bool; got {assemble!r}")
    if strength is not None and not (isinstance(strength, (int, float))
                                     and 0.0 <= float(strength) <= 1.0):
        raise ValueError(f"strength must be a float in [0, 1] or None; got {strength!r}")
    if not isinstance(chain, bool):
        raise ValueError(f"chain must be a bool; got {chain!r}")

    # ---- director knobs ----
    if not isinstance(vision_enabled, bool):
        raise ValueError(f"vision_enabled must be a bool; got {vision_enabled!r}")
    if not (isinstance(score_threshold, int) and 0 <= score_threshold <= 100):
        raise ValueError(
            f"score_threshold must be an int in 0..100; got {score_threshold!r}")
    if not (isinstance(max_attempts_per_segment, int) and max_attempts_per_segment >= 1):
        raise ValueError(
            f"max_attempts_per_segment must be an int >= 1; got {max_attempts_per_segment!r}")
    if judge_model_id is not None and not (isinstance(judge_model_id, str) and judge_model_id):
        raise ValueError(
            f"judge_model_id must be a non-empty str or None; got {judge_model_id!r}")
    if time_budget_s is not None and not (isinstance(time_budget_s, int)
                                          and not isinstance(time_budget_s, bool)
                                          and time_budget_s > 0):
        raise ValueError(
            f"time_budget_s must be a positive int or None; got {time_budget_s!r}")

    # ---- goal-timeline invariants (contiguity in the GIVEN order) ----
    cursor = 0
    for gi, g in enumerate(goals):
        if not isinstance(g, GoalInterval):
            raise ValueError(f"goals[{gi}] must be a GoalInterval; got {type(g).__name__}")
        if not (isinstance(g.start_frame, int) and not isinstance(g.start_frame, bool)
                and g.start_frame >= 0):
            raise ValueError(
                f"goals[{gi}].start_frame must be an int >= 0; got {g.start_frame!r}")
        if not (isinstance(g.end_frame, int) and not isinstance(g.end_frame, bool)
                and g.end_frame > g.start_frame):
            raise ValueError(
                f"goals[{gi}].end_frame must be an int > start_frame "
                f"({g.start_frame}); got {g.end_frame!r}")
        if not (isinstance(g.prompt, str) and g.prompt.strip()):
            raise ValueError(f"goals[{gi}].prompt must be a non-empty string")
        if g.ref is not None:
            if not isinstance(g.ref, MediaRef):
                raise ValueError(
                    f"goals[{gi}].ref must be a MediaRef or None; got {type(g.ref).__name__}")
            if g.ref.kind != "image":
                raise ValueError(
                    f"goals[{gi}].ref must be an image MediaRef; got kind={g.ref.kind!r}")
        # ---- per-goal PROMPT-COMPONENT override invariants (k92) ----
        # Each is OPTIONAL (None = inherit the movie-level value); when present it
        # must satisfy the SAME rule as the movie-level field it overrides.
        if g.model_id is not None and not (isinstance(g.model_id, str) and g.model_id.strip()):
            raise ValueError(
                f"goals[{gi}].model_id must be a non-empty str or None; got {g.model_id!r}")
        for _f in ("width", "height", "steps"):
            _v = getattr(g, _f)
            if _v is not None and not (isinstance(_v, int) and not isinstance(_v, bool) and _v > 0):
                raise ValueError(f"goals[{gi}].{_f} must be a positive int or None; got {_v!r}")
        if g.guidance is not None and not (isinstance(g.guidance, (int, float))
                                           and not isinstance(g.guidance, bool)):
            raise ValueError(
                f"goals[{gi}].guidance must be a number or None; got {g.guidance!r}")
        if g.seed is not None and not (isinstance(g.seed, int) and not isinstance(g.seed, bool)):
            raise ValueError(f"goals[{gi}].seed must be an int or None; got {g.seed!r}")
        if g.negative is not None and not isinstance(g.negative, str):
            raise ValueError(f"goals[{gi}].negative must be a str or None; got {g.negative!r}")
        if g.strength is not None and not (isinstance(g.strength, (int, float))
                                           and not isinstance(g.strength, bool)
                                           and 0.0 <= float(g.strength) <= 1.0):
            raise ValueError(
                f"goals[{gi}].strength must be a float in [0, 1] or None; got {g.strength!r}")
        if g.chain is not None and not isinstance(g.chain, bool):
            raise ValueError(f"goals[{gi}].chain must be a bool or None; got {g.chain!r}")
        if g.motion is not None and not isinstance(g.motion, str):
            raise ValueError(f"goals[{gi}].motion must be a str or None; got {g.motion!r}")
        seg_frames = g.end_frame - g.start_frame
        if seg_frames > FRAME_CAP:
            raise ValueError(
                f"frame_cap_exceeded: goals[{gi}] spans {seg_frames} frames "
                f"({g.start_frame}..{g.end_frame}) which exceeds the per-segment "
                f"cap {FRAME_CAP}")
        # contiguity / non-overlap in the GIVEN order (must tile [0, total))
        if g.start_frame != cursor:
            raise ValueError(
                f"goals must be CONTIGUOUS + non-overlapping starting at 0: "
                f"goals[{gi}].start_frame={g.start_frame} but the previous goal "
                f"ended at {cursor} (gap or overlap)")
        cursor = g.end_frame

    return MovieSpec(
        model_id=model_id,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        fps=fps,
        assemble=assemble,
        goals=goals,
        seed=seed,
        negative=negative,
        strength=(float(strength) if strength is not None else None),
        chain=chain,
        project=(project or None),
        vision_enabled=vision_enabled,
        score_threshold=score_threshold,
        max_attempts_per_segment=max_attempts_per_segment,
        judge_model_id=(judge_model_id or None),
        time_budget_s=time_budget_s,
    )


def total_frames(spec: MovieSpec) -> int:
    """The movie's total frame count == max(end_frame) (goals tile [0, total))."""
    return max(g.end_frame for g in spec.goals)


def goal_effective(goal: GoalInterval, spec: MovieSpec) -> dict:
    """Resolve ONE goal's effective generation knobs (k92): the goal's own override
    when set (not None), else the movie-level (MovieSpec) value. A movie with no
    per-goal overrides therefore renders EXACTLY as before. Frame count is NOT
    resolved here — it comes from the goal's half-open window (end-start).

    ``model_id`` uses ``or`` so an empty override also inherits; every other field
    keys off ``is not None`` so a legitimate 0 (guidance/seed/strength) is honored.
    ``motion`` is goal-only (MovieSpec has no shared motion)."""
    return {
        "model_id": goal.model_id or spec.model_id,
        "width": goal.width if goal.width is not None else spec.width,
        "height": goal.height if goal.height is not None else spec.height,
        "steps": goal.steps if goal.steps is not None else spec.steps,
        "guidance": goal.guidance if goal.guidance is not None else spec.guidance,
        "seed": goal.seed if goal.seed is not None else spec.seed,
        "negative": goal.negative if goal.negative is not None else spec.negative,
        "strength": (float(goal.strength) if goal.strength is not None else spec.strength),
        "chain": goal.chain if goal.chain is not None else spec.chain,
        "motion": goal.motion,
    }
