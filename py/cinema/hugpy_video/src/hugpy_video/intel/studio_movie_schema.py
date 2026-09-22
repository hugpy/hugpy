"""Studio-movie schema — an ordered strip of REAL studio clips conjoined at splice
points, like a single NLE timeline ROW: ``[segment 0 | segment 1 | …]``.

A studio movie is a SEQUENCE OF STUDIO CLIPS (produce_clip outputs), not a
frame-timeline like ``movie_schema`` (which tiles ``[0, total)`` with scene
segments). Each segment is one studio render:

  * Segment 0 renders t2v (or i2v when a movie-level ``start_image`` is given).
  * Each LATER segment renders i2v, conditioned on ONE still — the **branch frame**
    of the PREVIOUS segment's clip (``branch_frame: int | null``; null ⇒ the
    parent's LAST frame). Motion is not carried across the splice (still
    conditioning only); VACE-extend is the planned upgrade.

NON-DESTRUCTIVE. Clip files stay WHOLE. A mid-frame branch means the assembled
movie uses the parent clip only UP TO the branch frame — the trim is METADATA
(recorded per joint, honored at concat time), never a re-render of the parent.

TAKE-TREE DATA MODEL (grows without rework). Segments are stored as a list of
NODES ``{segment_id, parent_segment_id, branch_frame, prompt, …}``. TODAY the
chain is LINEAR — each node's ``parent_segment_id`` is the previous node's
``segment_id`` and assembly walks that single path — and ``make_studio_movie``
ENFORCES that linearity (see below). Sibling divergences from the same parent
(a real take-tree) become possible LATER with no schema change: the node shape
already carries the parent pointer, so only the validator's linear-chain rule and
the assembly walk need to relax.

House style mirrors ``movie_schema`` / ``studio.job``: a frozen spec + frozen
node, a validating factory whose raises are LOCAL to construction (a structurally
invalid spec is caller error caught at the boundary, never carried across the
bus), and asdict-friendly, JSON-safe fields so the bus round-trips it through
``studio_movie_from_dict`` (reconstruct + RE-VALIDATE).

V0 SIMPLIFICATIONS (stated here + in the runner header + the report):
  * TIER IS MOVIE-LEVEL. ``vram_budget_gb`` (the synthetic-vs-real tier selector)
    is a single movie-level value applied to every segment. Per-segment tier
    selection is planned growth.
  * GEOMETRY IS UNIFORM. ``width`` / ``height`` / ``fps`` are movie-level because
    the segments concat into one row (the assembler needs matching geometry).
  * ``model_id`` / ``steps`` / ``cfg`` are movie-level defaults, but a node MAY
    OVERRIDE them per segment (honored when set — not dead fields). ``negative``
    and ``seed`` are likewise per-segment-overridable over the movie default.
  * PER-SEGMENT + MOVIE-LEVEL ``frames`` EXIST NOW (2026-08-13, operator: "the
    length of the video generations has been ambiguous") — added TOGETHER with
    the runner thread (studio_movie.py sets ``requested_frames`` per segment:
    goal.frames else spec.frames else None -> the bound model's default), per
    the rule below. The spine clamps to the model ceiling and snaps to the 4k+1
    temporal cadence, so an out-of-envelope ask degrades honestly, never lies.
  * (HISTORICAL NOTE — the reasoning that kept this field OUT until the thread
    existed:)
    Clip length is a pure function of the manifest, derived by the studio spine
    (``RenderManifest.requested_frames`` when the caller asks for a length, else the
    bound model's default — 81 frames for a real model, ``fps * 2`` FRAMES for the
    synthetic prover — clamped to that model's ceiling and snapped to the 4k+1
    temporal cadence; see ``studio/runners/synthetic.resolve_frames``).

    ⚠ CORRECTION (2026-07-27): the previous wording here — "``StudioI2VSpec`` itself
    carries none" — is no longer true. ``StudioI2VSpec.requested_frames`` EXISTS and
    is threaded spec -> ``produce_clip`` -> manifest -> runner. What is still missing
    is the MOVIE side of that thread: ``video_intel/runners/studio_movie.py`` builds
    one ``StudioI2VSpec`` per segment and does not set ``requested_frames``, so every
    segment renders at the bound model's default. A movie-level + per-goal length knob
    is therefore ONE runner change away, not a schema redesign — and it is
    deliberately NOT a field here until that runner threads it, because a knob the
    runner ignores is a dead field that lies to the caller (exactly the defect class
    the single-clip lever was just fixed for: a documented, hashed, unreachable
    ``requested_frames``). Add the field and the runner thread TOGETHER or not at all.

    ``branch_frame`` is consequently bounded by the parent clip's ACTUAL frame count,
    checkable only at RUN time (the schema validates ``branch_frame >= 0``;
    ``branch_frame < parent_frames`` is a runtime error-as-data from the runner).

No pathlib anywhere. os.path only (there is none here — pure data).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from hugpy_video.intel.media_schema import MediaRef, make_media_ref

# Sampler-override sanity ranges — mirror ``studio.job``'s bounds so a movie spec
# rejects the same out-of-range steps/cfg the single-clip route does (one denoise
# vocabulary across the studio surface). steps<1 is a no-op denoise; a huge count
# or cfg is almost always a typo. cfg 0 is valid (unguided).
_MIN_STEPS, _MAX_STEPS = 1, 100
_MIN_CFG, _MAX_CFG = 0.0, 20.0

# Movie-level tier default: below the smallest real i2v footprint so the router
# binds the SYNTHETIC spine (a GPU-less box produces a clip), mirroring
# ``studio.job._DEFAULT_VRAM_BUDGET_GB``.
_DEFAULT_VRAM_BUDGET_GB = 0.5

# JOINT MODE — how a segment is spliced onto its parent at the branch point:
#   * "still"       (default, backward-compatible): condition the segment's render on ONE
#                   frame (the branch still). Motion is NOT carried across the splice —
#                   the historical behavior.
#   * "vace_extend": carry MOTION across the splice by conditioning on the parent's
#                   TRAILING ``context_frames`` frames through the VACE video+mask
#                   extend idiom (the runner routes the segment through the VACE path).
#   * "cut"         (SCENE CUT — no frame carry at all): the child is a FRESH render of its
#                   own prompt, spliced with a HARD cut. The parent is NOT trimmed (it plays
#                   in FULL) and NO branch still / context window is extracted — a cut
#                   carries no frame. In an IDENTITY MOVIE (movie-level ``reference_images``
#                   set) the child is a fresh id_lock render so the SUBJECT carries across
#                   the scene change even though no pixels do; in a plain movie it is a hard
#                   cut between two independent scenes. Because a cut carries no frame it is
#                   INCOMPATIBLE with ``branch_frame`` / ``context_frames`` (both must be
#                   None) — enforced in the factory.
# goal 0 (the root) has no parent, so it MUST be "still" (there is nothing to splice onto)
# — enforced in the factory (so "cut" and "vace_extend" are valid only on goals >= 1).
_VALID_JOINT_MODES = frozenset({"still", "vace_extend", "cut"})
_DEFAULT_JOINT_MODE = "still"

# IDENTITY LOCK (id_lock) — movie-level subject reference images. When set, EVERY segment
# renders capability ``id_lock`` (Wan-VACE reference-to-video), so the locked SUBJECT
# carries across scene changes (the operator's "take that id and use it for a video — her
# on the beach, then playing volleyball"). Mirrors ``studio.job``'s ``_MAX_REFERENCE_IMAGES``
# (diffusers 0.39 consumes each as a prepended VACE reference latent). () = a plain movie.
_MAX_REFERENCE_IMAGES = 4

# CONTEXT FRAMES — how many trailing parent frames condition a ``vace_extend`` splice
# (movie-level default; a node MAY override). Bounded: >=1 (a splice needs at least one
# context frame) and <= the ceiling below (a generous cap kept well under the VACE
# model's 81-frame clip so an extension always has room to GENERATE new frames after
# the kept context). Only consulted for ``vace_extend`` joints; ignored for "still".
_DEFAULT_CONTEXT_FRAMES = 8
_MIN_CONTEXT_FRAMES, _MAX_CONTEXT_FRAMES = 1, 32


@dataclass(frozen=True)
class StudioMovieGoal:
    """One NODE on the studio movie's take-tree.

        segment_id         stable id of this segment's node (non-empty, unique in
                           the movie). The parent pointer + assembly key.
        prompt             the text this segment renders (non-empty).
        parent_segment_id  the node this segment branches FROM. None ⇒ this is the
                           root (segment 0). TODAY it must be the PREVIOUS node's
                           segment_id (linear chain); the field already models a
                           tree for future sibling divergence.
        branch_frame       the frame INDEX of the parent's clip this segment is
                           conditioned on (>= 0). None ⇒ the parent's LAST frame.
                           Bounds against the parent's real frame count are a
                           RUN-time check (the schema only enforces >= 0).
        negative           optional per-segment negative prompt; when None the
                           movie-level ``negative`` is used.
        seed               optional per-segment seed; when None the runner derives
                           a deterministic one (movie seed + segment index).
        model_id           optional per-segment model pin; when None the movie-level
                           ``model_id`` is used (v0: usually None -> auto-pick).
        steps / cfg        optional per-segment sampler overrides; when None the
                           movie-level value (or the bound model's family default)
                           is used.
        joint_mode         how this segment is spliced onto its parent: "still"
                           (default — condition on ONE branch frame, motion NOT carried),
                           "vace_extend" (condition on the parent's trailing
                           ``context_frames`` frames via the VACE video+mask extend
                           idiom, carrying motion), or "cut" (a HARD scene cut — no frame
                           carry: the parent plays in FULL and the child is a fresh render;
                           INCOMPATIBLE with ``branch_frame`` / ``context_frames``). goal 0
                           (the root) MUST be "still".
        frames             optional per-segment CLIP LENGTH in frames; None ⇒ the
                           movie-level ``frames`` (else the bound model's default,
                           81 for real models). Clamped to the model ceiling and
                           snapped to the 4k+1 cadence by the spine at render.
        context_frames     optional per-segment override of how many trailing parent
                           frames condition a ``vace_extend`` splice; None ⇒ the
                           movie-level ``context_frames``. Only consulted when
                           ``joint_mode == "vace_extend"``.
        reference_images   PER-GOAL id_lock DNA override (IDENTITY-3D-CONTINUITY-PLAN.md
                           S2-movie): the ORDERED tuple of subject reference paths/uris
                           THIS segment conditions on, when it should differ from the
                           movie-level set. None ⇒ INHERIT the movie-level
                           ``reference_images`` (byte-identical to today's every-segment
                           behavior). The point is CONTINUITY ACROSS A CUT: an identity
                           movie can hold the SAME person while pointing each shot at a
                           different turntable VIEW of them (segment 1 front, segment 2
                           back), so a ``cut`` into a new scene ("beach" -> "volleyball")
                           keeps the character but re-frames the camera. The route resolves
                           a goal's ``view`` to the K angle-nearest ring frames and sets
                           this; the runner just prefers it over the movie-level set. Same
                           shape/bounds as the movie-level field (tuple of non-empty
                           path/uri strings, at most ``_MAX_REFERENCE_IMAGES``); jail /
                           existence / image-classification is the ROUTE's job.
    """
    segment_id: str
    prompt: str
    parent_segment_id: Optional[str] = None
    branch_frame: Optional[int] = None
    negative: Optional[str] = None
    seed: Optional[int] = None
    model_id: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    joint_mode: str = _DEFAULT_JOINT_MODE
    context_frames: Optional[int] = None
    frames: Optional[int] = None
    reference_images: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class StudioMovieSpec:
    """Frozen currency of a ``generate_studio_movie`` bus job. Built ONLY via
    ``make_studio_movie`` (validate-at-construction); the bus rehydrates it via
    ``studio_movie_from_dict``, which re-validates through the same factory. Every
    field is JSON-safe (primitives + a nested ``MediaRef`` / ``StudioMovieGoal``
    tuple) so ``asdict`` -> ``json.dumps`` round-trips cleanly.

        goals            the ORDERED tuple of take-tree nodes (>=1). goals[0] is the
                         root; today goals[i>0].parent_segment_id == goals[i-1].segment_id.
        width/height/fps movie geometry (uniform across segments — they concat).
        vram_budget_gb   the movie-level tier selector (synthetic vs. real).
        seed             base seed; a node with no seed uses ``seed + index``.
        negative         movie-level default negative (a node may override).
        model_id         movie-level default model pin (a node may override; None
                         = router auto-pick).
        steps/cfg        movie-level default sampler overrides (a node may override;
                         None = the bound model's family default).
        title            optional human NAME for this movie SESSION (k91) — free text,
                         non-canonical, purely for the operator ("the lighthouse
                         cut"). OPTIONAL BY CONTRACT: every pre-k91 caller omits it
                         and gets None, and nothing downstream branches on it, so an
                         unnamed movie behaves exactly as it always has. It is
                         DELIBERATELY not ``project``: ``project`` is a SLUG that
                         becomes the on-disk movie dir (``movie_root_for``), so
                         renaming it would move the session's directory and orphan
                         every rendered segment. ``title`` is display-only and can
                         therefore be changed freely — which is the whole point,
                         since the UI warns about an unnamed movie AFTER the render
                         is already under way.
        project          optional human auto-archive NAME (non-canonical metadata).
                         ⚠ Also the movie DIR's leaf name (slugified) — see
                         ``runners.studio_movie.movie_root_for``.
        session_id       optional PIN for the movie dir's leaf name (k91). None — the
                         only value a SUBMIT ever sets — means "derive the leaf as
                         always": the slugified ``project``, else the bus job id.
                         RESUME sets it, and must.

                         WHY IT HAS TO EXIST. A movie DIR is the resumable session, and
                         resume works by re-enqueueing this spec so ``produce_clip``
                         finds each completed segment's clip already sitting at
                         ``<movie_dir>/segment_NN/<content_hash>/clip.mp4``. That lookup
                         is BY PATH — so if the dir moves, nothing is found and every
                         segment re-renders from scratch. For an UNNAMED movie the leaf
                         is the job id, and a resume mints a NEW job id, which would
                         move the dir on every single resume: the feature would appear
                         to work while silently re-rendering the whole movie. Pinning
                         the leaf is what makes "the clips are the checkpoint" true.

                         Deliberately NOT folded into ``project``: ``project`` is
                         auto-archive metadata that threads down into every per-segment
                         studio spec, and overloading it would put a hex session id in
                         front of the operator as their project name.

                         Constrained to a bare path LEAF (no separators, not "."/"..")
                         because it becomes a directory name.
        out_root         output location (a dir under the media-store root); None
                         resolves to the studio-movie default in the runner.
        start_image      optional conditioning still for SEGMENT 0 (a MediaRef of
                         kind 'image'); when present segment 0 renders i2v from it,
                         else t2v.
        time_budget_s    optional wall-clock budget the runner owns itself (the
                         single-daemon bus has no timeout/reaper), honored BETWEEN
                         segments — mirrors ``MovieSpec.time_budget_s``.
        frames           movie-level default CLIP LENGTH per segment (a goal's own
                         ``frames`` overrides); None ⇒ each bound model's default.
        context_frames   movie-level default number of trailing parent frames that
                         condition a ``vace_extend`` splice (a node may override via
                         its own ``context_frames``). Only consulted for vace_extend
                         joints; ignored by "still" joints.
        reference_images IDENTITY LOCK: the ORDERED tuple of jailed abs paths of the
                         subject reference image(s). When NON-EMPTY the movie is an
                         IDENTITY MOVIE — EVERY segment renders capability ``id_lock``
                         (Wan-VACE reference-to-video) so the locked subject carries
                         across every scene change. () = a plain movie (t2v/i2v/v2v as
                         before). CANONICAL (the references define the identity). At most
                         ``_MAX_REFERENCE_IMAGES``. Jail/existence/image-classification is
                         the ROUTE's job (a runtime input check), like ``StudioI2VSpec``.
    """
    goals: Tuple[StudioMovieGoal, ...]
    width: int
    height: int
    fps: int
    # AUTOFIT tier. A number PINS the movie-level tier (applied to every segment). ``None``
    # means AUTOFIT: a BLANK budget is sized per segment to the serving worker's MEASURED
    # free VRAM at render time (see runners.studio_movie / render_clip) — never a
    # guaranteed-fail low guess. The VACE floor bump (id_lock / vace_extend) applies only
    # to an EXPLICIT budget; an autofit None flows through untouched (the resolved free VRAM
    # already clears the floor on a real box, and a too-small box honestly Errs).
    vram_budget_gb: Optional[float] = _DEFAULT_VRAM_BUDGET_GB
    seed: int = 0
    negative: Optional[str] = None
    model_id: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    # SESSION NAME (k91). Optional, display-only, never load-bearing — see the class
    # docstring for why this is separate from ``project`` (which names the DIR).
    title: Optional[str] = None
    project: Optional[str] = None
    # SESSION DIR PIN (k91). None on every submit; set by RESUME so the re-enqueued run
    # lands on the SAME dir and finds the already-rendered clips. See the docstring.
    session_id: Optional[str] = None
    out_root: Optional[str] = None
    start_image: Optional[MediaRef] = None
    time_budget_s: Optional[int] = None
    context_frames: int = _DEFAULT_CONTEXT_FRAMES
    frames: Optional[int] = None
    reference_images: Tuple[str, ...] = ()
    # k120 slice 2 — PRODUCER CONTINUITY REFRESH (opt-in, advisory). When True the
    # runner rewrites each non-root segment's prompt from a vision description of
    # the frame the previous segment ACTUALLY ended on, before rendering it.
    # continuity_model pins the TEXT model for the rewrite (None = the runner's
    # explicit default, never the silent task-default).
    continuity_refresh: bool = False
    continuity_model: Optional[str] = None


def _check_steps_cfg(where: str, steps, cfg) -> None:
    """Shared steps/cfg range guard (movie-level + per-node). bool is an int
    subclass — reject it explicitly. Raises LOCALLY (caller error at construction)."""
    if steps is not None:
        if not isinstance(steps, int) or isinstance(steps, bool) \
                or not (_MIN_STEPS <= steps <= _MAX_STEPS):
            raise ValueError(
                f"{where} steps must be an int in [{_MIN_STEPS}, {_MAX_STEPS}] or None; "
                f"got {steps!r}")
    if cfg is not None:
        if not isinstance(cfg, (int, float)) or isinstance(cfg, bool) \
                or not (_MIN_CFG <= cfg <= _MAX_CFG):
            raise ValueError(
                f"{where} cfg must be a number in [{_MIN_CFG}, {_MAX_CFG}] or None; "
                f"got {cfg!r}")


def _check_frames(where: str, value) -> None:
    """Clip-length guard (movie-level + per-goal): a positive int or None.
    The UPPER bound is deliberately not here — the spine clamps to the bound
    model's ceiling at render (per-model, unknowable at construction)."""
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{where}: frames must be a positive int or None; got {value!r}")


def _check_context_frames(where: str, value) -> None:
    """Shared context_frames range guard (movie-level + per-node). bool is an int
    subclass — reject it explicitly. Raises LOCALLY (caller error at construction)."""
    if not isinstance(value, int) or isinstance(value, bool) \
            or not (_MIN_CONTEXT_FRAMES <= value <= _MAX_CONTEXT_FRAMES):
        raise ValueError(
            f"{where} context_frames must be an int in "
            f"[{_MIN_CONTEXT_FRAMES}, {_MAX_CONTEXT_FRAMES}]; got {value!r}")


def make_studio_movie(
    goals: Tuple[StudioMovieGoal, ...],
    width: int,
    height: int,
    fps: int,
    vram_budget_gb: float = _DEFAULT_VRAM_BUDGET_GB,
    seed: int = 0,
    negative: Optional[str] = None,
    model_id: Optional[str] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    title: Optional[str] = None,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    out_root: Optional[str] = None,
    start_image: Optional[MediaRef] = None,
    time_budget_s: Optional[int] = None,
    context_frames: int = _DEFAULT_CONTEXT_FRAMES,
    frames: Optional[int] = None,
    reference_images: Optional[Tuple[str, ...]] = None,
    continuity_refresh: bool = False,
    continuity_model: Optional[str] = None,
) -> StudioMovieSpec:
    """Validate every field and build the frozen ``StudioMovieSpec``. Raises
    ``ValueError``/``TypeError`` LOCALLY on any structural violation — a
    structurally-invalid spec is caller error caught at the boundary, never carried
    across the bus. Runtime policy failures (an unroutable request, a branch_frame
    past the parent's real length) are NOT validated here; they surface as
    errors-as-data from the runner.

    Take-tree invariants (the load-bearing ones):
      * ``goals`` is non-empty;
      * every ``segment_id`` is a non-empty string, UNIQUE across the movie;
      * every ``prompt`` is a non-empty string;
      * every ``branch_frame`` is None or an int >= 0 (a negative index is caller
        error; the < parent_frames upper bound is a run-time check — see header);
      * LINEAR CHAIN (v0): goals[0].parent_segment_id is None (the root), and for
        i>0 goals[i].parent_segment_id == goals[i-1].segment_id. This is the ONLY
        rule that must relax for a real take-tree; the node shape already carries
        the parent pointer, so sibling divergence needs no schema change.
      * JOINT MODE: every node's ``joint_mode`` is "still", "vace_extend", or "cut"; goal
        0 (the root, no parent) MUST be "still" — there is nothing to splice onto. A "cut"
        node carries NO frame, so it is INCOMPATIBLE with ``branch_frame`` /
        ``context_frames`` (both must be None) — rejected LOCALLY.
      * CONTEXT FRAMES: the movie-level + any per-node ``context_frames`` is an int in
        [1, 32] (a splice needs >=1 context frame; the cap keeps room to generate).
      * TITLE (k91): None or a string; whitespace-only normalizes to None (an unnamed
        session), mirroring ``project``'s normalization. Never required — a movie
        submitted without one is valid and renders identically.
      * REFERENCE IMAGES: movie-level ``reference_images`` is None/() or a tuple of
        non-empty path strings, at most ``_MAX_REFERENCE_IMAGES`` (an identity movie).
        Each node's OPTIONAL ``reference_images`` (the per-goal id_lock override) is
        validated the SAME way — None (inherit) or a list/tuple of non-empty path/uri
        strings, at most ``_MAX_REFERENCE_IMAGES`` — and coerced to a tuple (so an
        asdict->json->from_dict round-trip lands a tuple, mirroring the movie-level field).

    Also the reconstruction path used by the bus deserializer — goals are rebuilt
    into ``StudioMovieGoal`` (and ``start_image`` through ``make_media_ref``) before
    this is called.
    """
    goals = tuple(goals)
    if not goals:
        raise ValueError("make_studio_movie requires at least one StudioMovieGoal")

    # ---- movie-level geometry / tier / sampler ----
    for name, val in (("width", width), ("height", height), ("fps", fps)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive int; got {val!r}")
    # AUTOFIT: None is a LEGAL value (blank -> fit to the serving worker's free VRAM at
    # render time). A NUMBER is the manual override and must be positive.
    if vram_budget_gb is not None:
        if not isinstance(vram_budget_gb, (int, float)) or isinstance(vram_budget_gb, bool) \
                or vram_budget_gb <= 0:
            raise ValueError(
                f"vram_budget_gb must be a positive number or None (autofit); "
                f"got {vram_budget_gb!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int; got {seed!r}")
    if negative is not None and not isinstance(negative, str):
        raise ValueError(f"negative must be a string or None; got {negative!r}")
    if model_id is not None and not (isinstance(model_id, str) and model_id.strip()):
        raise ValueError(f"model_id must be a non-empty string or None; got {model_id!r}")
    _check_steps_cfg("movie-level", steps, cfg)
    # SESSION NAME (k91): free text, normalized the SAME way ``project`` is — a
    # whitespace-only name is not a name, so it collapses to None rather than being
    # stored as "   ". That matters because the UI's "warn if unnamed" test is a plain
    # truthiness check on this field, and a blank-but-present title would defeat it
    # while looking named in the session list.
    if title is not None and not isinstance(title, str):
        raise ValueError(f"title must be a string or None; got {title!r}")
    title = (title.strip() or None) if isinstance(title, str) else None
    if project is not None and not isinstance(project, str):
        raise ValueError(f"project must be a string or None; got {project!r}")
    project = (project.strip() or None) if isinstance(project, str) else None
    # SESSION DIR PIN (k91): None (derive as always) or a bare path LEAF. It becomes a
    # DIRECTORY NAME, so a separator or a parent ref is rejected LOCALLY here rather
    # than turned into a path the runner would then write through — the same
    # never-build-a-path-from-unvalidated-input rule the routes' jail enforces, applied
    # at the one place this value can enter the system.
    if session_id is not None:
        if not (isinstance(session_id, str) and session_id.strip()):
            raise ValueError(
                f"session_id must be a non-empty string or None; got {session_id!r}")
        session_id = session_id.strip()
        # Separators spelled out rather than reached for via ``os.sep`` — this module
        # is pure data and imports no os (see the header), and both spellings are
        # rejected regardless of platform.
        if ("/" in session_id or "\\" in session_id or "\0" in session_id
                or session_id in (".", "..")):
            raise ValueError(
                f"session_id must be a bare directory leaf (no path separators, "
                f"not '.'/'..'); got {session_id!r}")
    if out_root is not None and not (isinstance(out_root, str) and out_root.strip()):
        raise ValueError(f"out_root must be a non-empty string or None; got {out_root!r}")
    if start_image is not None:
        if not isinstance(start_image, MediaRef):
            raise ValueError(
                f"start_image must be a MediaRef or None; got {type(start_image).__name__}")
        if start_image.kind != "image":
            raise ValueError(
                f"start_image must be an image MediaRef; got kind={start_image.kind!r}")
    if time_budget_s is not None and not (isinstance(time_budget_s, int)
                                          and not isinstance(time_budget_s, bool)
                                          and time_budget_s > 0):
        raise ValueError(
            f"time_budget_s must be a positive int or None; got {time_budget_s!r}")
    _check_context_frames("movie-level", context_frames)

    # ---- IDENTITY LOCK: movie-level reference images (structural check only) ----
    # None -> (); coerce a list/tuple to a tuple (so an asdict->json->from_dict round-trip
    # lands a tuple). Each must be a non-empty string; at most _MAX_REFERENCE_IMAGES. Jail /
    # existence / image-classification is the ROUTE's job (mirrors StudioI2VSpec).
    if reference_images is None:
        reference_images = ()
    if isinstance(reference_images, (list, tuple)):
        reference_images = tuple(reference_images)
    else:
        raise ValueError(
            f"reference_images must be a list/tuple of paths or None; got {reference_images!r}")
    for ri, r in enumerate(reference_images):
        if not (isinstance(r, str) and r.strip()):
            raise ValueError(f"reference_images[{ri}] must be a non-empty string; got {r!r}")
    if len(reference_images) > _MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"at most {_MAX_REFERENCE_IMAGES} reference_images are accepted; "
            f"got {len(reference_images)}")

    # ---- per-node validation + linear-chain enforcement (v0) ----
    seen_ids: set = set()
    prev_id: Optional[str] = None
    norm_goals: list = []   # goals with per-node reference_images coerced to tuple
    for gi, g in enumerate(goals):
        if not isinstance(g, StudioMovieGoal):
            raise ValueError(f"goals[{gi}] must be a StudioMovieGoal; got {type(g).__name__}")
        if not (isinstance(g.segment_id, str) and g.segment_id.strip()):
            raise ValueError(f"goals[{gi}].segment_id must be a non-empty string")
        if g.segment_id in seen_ids:
            raise ValueError(f"goals[{gi}].segment_id={g.segment_id!r} is not unique")
        seen_ids.add(g.segment_id)
        if not (isinstance(g.prompt, str) and g.prompt.strip()):
            raise ValueError(f"goals[{gi}].prompt must be a non-empty string")
        if g.branch_frame is not None:
            if not (isinstance(g.branch_frame, int) and not isinstance(g.branch_frame, bool)
                    and g.branch_frame >= 0):
                raise ValueError(
                    f"goals[{gi}].branch_frame must be an int >= 0 or None; "
                    f"got {g.branch_frame!r}")
        if g.negative is not None and not isinstance(g.negative, str):
            raise ValueError(f"goals[{gi}].negative must be a string or None; got {g.negative!r}")
        if g.seed is not None and (not isinstance(g.seed, int) or isinstance(g.seed, bool)):
            raise ValueError(f"goals[{gi}].seed must be an int or None; got {g.seed!r}")
        if g.model_id is not None and not (isinstance(g.model_id, str) and g.model_id.strip()):
            raise ValueError(
                f"goals[{gi}].model_id must be a non-empty string or None; got {g.model_id!r}")
        _check_steps_cfg(f"goals[{gi}]", g.steps, g.cfg)

        # JOINT MODE: "still" (default) or "vace_extend". goal 0 (the root) has no
        # parent to extend from, so it MUST be "still" (enforced below).
        if not (isinstance(g.joint_mode, str) and g.joint_mode in _VALID_JOINT_MODES):
            raise ValueError(
                f"goals[{gi}].joint_mode must be one of {sorted(_VALID_JOINT_MODES)}; "
                f"got {g.joint_mode!r}")
        if g.context_frames is not None:
            _check_context_frames(f"goals[{gi}]", g.context_frames)
        # CUT (scene cut) carries NO frame — a branch still / context window is meaningless
        # for it, so branch_frame + context_frames MUST both be None (reject a body that
        # sets them together with an explicit, clear message).
        if g.joint_mode == "cut":
            if g.branch_frame is not None:
                raise ValueError(
                    f"goals[{gi}].joint_mode='cut' carries no frame, so branch_frame must "
                    f"be None (a scene cut does not condition on a parent frame); "
                    f"got branch_frame={g.branch_frame!r}")
            if g.context_frames is not None:
                raise ValueError(
                    f"goals[{gi}].joint_mode='cut' carries no frame, so context_frames must "
                    f"be None (a scene cut extends no context); "
                    f"got context_frames={g.context_frames!r}")

        # PER-GOAL REFERENCE IMAGES (S2-movie id_lock override): None -> inherit the
        # movie-level set (unchanged behavior). Otherwise SAME shape/bounds as the
        # movie-level field — a list/tuple of non-empty path/uri strings, at most
        # _MAX_REFERENCE_IMAGES — coerced to a tuple so a round-trip lands a tuple. A bad
        # value is caller error caught here (a wrong type, an empty path, too many refs).
        node_refs = g.reference_images
        if node_refs is not None:
            if not isinstance(node_refs, (list, tuple)):
                raise ValueError(
                    f"goals[{gi}].reference_images must be a list/tuple of paths or None; "
                    f"got {node_refs!r}")
            node_refs = tuple(node_refs)
            for ri, r in enumerate(node_refs):
                if not (isinstance(r, str) and r.strip()):
                    raise ValueError(
                        f"goals[{gi}].reference_images[{ri}] must be a non-empty string; "
                        f"got {r!r}")
            if len(node_refs) > _MAX_REFERENCE_IMAGES:
                raise ValueError(
                    f"goals[{gi}]: at most {_MAX_REFERENCE_IMAGES} reference_images are "
                    f"accepted; got {len(node_refs)}")
            if node_refs != g.reference_images:   # coerced a list -> tuple: rebuild the node
                g = replace(g, reference_images=node_refs)
        norm_goals.append(g)

        # LINEAR CHAIN (v0): the root has no parent; every later node's parent is the
        # node right before it. The one rule a real take-tree relaxes.
        if gi == 0:
            if g.parent_segment_id is not None:
                raise ValueError(
                    f"goals[0].parent_segment_id must be None (the root has no parent); "
                    f"got {g.parent_segment_id!r}")
            if g.joint_mode != _DEFAULT_JOINT_MODE:
                raise ValueError(
                    f"goals[0].joint_mode must be {_DEFAULT_JOINT_MODE!r} (the root has no "
                    f"parent to extend from); got {g.joint_mode!r}")
        else:
            if g.parent_segment_id != prev_id:
                raise ValueError(
                    f"goals[{gi}].parent_segment_id must be the previous node's segment_id "
                    f"{prev_id!r} (linear chain for now; tree branching is planned growth); "
                    f"got {g.parent_segment_id!r}")
        prev_id = g.segment_id

        _check_frames("movie", frames)
    for _g in goals:
        _check_frames(f"goal {_g.segment_id!r}", _g.frames)
    return StudioMovieSpec(
        goals=tuple(norm_goals),
        width=width,
        height=height,
        fps=fps,
        # AUTOFIT: keep None verbatim (resolved per segment at render time); else float.
        vram_budget_gb=(float(vram_budget_gb) if vram_budget_gb is not None else None),
        seed=seed,
        negative=negative,
        model_id=model_id,
        steps=steps,
        cfg=(float(cfg) if cfg is not None else None),
        title=title,
        project=project,
        session_id=session_id,
        out_root=out_root,
        start_image=start_image,
        time_budget_s=time_budget_s,
        context_frames=context_frames,
        frames=frames,
        reference_images=reference_images,
        # k120 slice 2 — bool() collapses truthy dict values; a non-string model
        # pin is a caller error worth raising on.
        continuity_refresh=bool(continuity_refresh),
        continuity_model=_checked_continuity_model(continuity_model),
    )


def _checked_continuity_model(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"continuity_model must be a non-empty string or None; got {value!r}")
    return value.strip()


def _goal_from_dict(d: dict) -> StudioMovieGoal:
    """Rebuild ONE take-tree node from its ``asdict`` form (shape only; the full
    invariant check runs in ``make_studio_movie``)."""
    return StudioMovieGoal(
        segment_id=d.get("segment_id"),
        prompt=d.get("prompt"),
        parent_segment_id=d.get("parent_segment_id"),
        branch_frame=d.get("branch_frame"),
        negative=d.get("negative"),
        seed=d.get("seed"),
        model_id=d.get("model_id"),
        steps=d.get("steps"),
        cfg=d.get("cfg"),
        joint_mode=d.get("joint_mode", _DEFAULT_JOINT_MODE),
        context_frames=d.get("context_frames"),
        # PER-GOAL id_lock override (S2-movie): None -> inherit the movie-level set.
        # make_studio_movie coerces a JSON list back to a tuple + re-validates it.
        reference_images=d.get("reference_images"),
    )


def studio_movie_from_dict(d: dict) -> StudioMovieSpec:
    """Rebuild a ``StudioMovieSpec`` from its ``asdict`` form, THROUGH the validating
    factory (mirrors ``studio.job.studio_i2v_from_dict`` / ``movie_schema``'s
    deserialize-then-revalidate) so a rehydrated spec is re-checked, never trusted
    blind. Registered in ``media_bus.SPEC_DESERIALIZERS`` under the name
    ``"generate_studio_movie"``."""
    raw_goals = d.get("goals") or ()
    goals = tuple(_goal_from_dict(g) for g in raw_goals)
    start_image = d.get("start_image")
    ref = make_media_ref(**start_image) if isinstance(start_image, dict) else None
    return make_studio_movie(
        goals=goals,
        width=d["width"],
        height=d["height"],
        fps=d["fps"],
        vram_budget_gb=d.get("vram_budget_gb", _DEFAULT_VRAM_BUDGET_GB),
        seed=d.get("seed", 0),
        negative=d.get("negative"),
        model_id=d.get("model_id"),
        steps=d.get("steps"),
        cfg=d.get("cfg"),
        # SESSION NAME (k91) — absent in every spec persisted before this field
        # existed, so ``.get`` (not ``[]``) is what keeps a pre-k91 spec.json / bus
        # row rehydratable. None is the correct value for an unnamed movie.
        title=d.get("title"),
        project=d.get("project"),
        # SESSION DIR PIN (k91) — absent on every pre-k91 spec and on every submit;
        # present on a spec.json that has been resumed at least once.
        session_id=d.get("session_id"),
        out_root=d.get("out_root"),
        start_image=ref,
        time_budget_s=d.get("time_budget_s"),
        context_frames=d.get("context_frames", _DEFAULT_CONTEXT_FRAMES),
        # MOVIE-LEVEL FRAMES: this factory used to drop the field on every bus
        # rehydrate/resume (present on the dataclass + set by the route since
        # 2026-08-13, never round-tripped here) — k120 recon caught it.
        frames=d.get("frames"),
        reference_images=d.get("reference_images"),
        # k120 slice 2 — absent on every pre-k120 spec; .get keeps them rehydratable.
        continuity_refresh=d.get("continuity_refresh", False),
        continuity_model=d.get("continuity_model"),
    )
