"""The sibling SegmentSpec compiler (k104) — doc Stage 8 timing, Stage 14 shape.

Stage 14 is a diagram and a prohibition:

    prohibited:  SegmentSpec[1] -> SegmentSpec[2] -> SegmentSpec[3]
    required:    Locked production artifacts -> SegmentSpec[1], [2], [3]

and invariant 9 is the sentence behind it: *"No segment prompt generated during
a run may become the source of another segment prompt in that run."* This module
is that shape, enforced STRUCTURALLY rather than checked afterwards.

HOW THE PROHIBITION IS MADE UNREACHABLE, not merely tested:

  1. ``compile_segments`` builds ONE :class:`LockedContext` from the locked
     artifacts and passes THAT SAME OBJECT to the prompt writer for every
     index. The writer's signature is ``(locked_context, index) -> str``. There
     is no parameter through which a previous spec could arrive.
  2. ``LockedContext`` has no field that can hold a ``SegmentSpec`` or a
     generated prompt — only locked material: continuity states, audio windows,
     camera/blocking/lighting, rubrics, identity refs, exclusions. A malicious
     writer that goes looking for the shot before it finds the LOCKED plan for
     that shot, which is the required relationship, not the prohibited one.
  3. Compilation is TWO PHASES. Every prompt is written first, then the specs
     are assembled. While any prompt is being written, no ``SegmentSpec``
     object exists anywhere in the call — so even a writer that reached into
     the compiler's frame would find nothing to chain to.
  4. ``SegmentSpec.parents`` is filled from ``ProductionLock.parent_digests``
     and from nowhere else, and ``assert_siblings`` re-checks that no spec
     names another spec's digest before ``compile_segments`` returns.
  5. ``to_plan_graph`` emits a locked-parent node fanning out to one ``task``
     node per segment and runs k103's ``sibling_check`` on its own output
     before handing it back.

This is the typed cousin of k93's ``video_intel/prompt_spread.py`` — "one
generator call that writes the prompts for an ENTIRE movie at once, holding the
rows the user did not select as locked context". Same idiom, different
substrate: the locked context here is a production lock rather than a UI's
unselected rows, and the joint modes are still rendered as SENTENCES rather than
raw tokens, because a model shown ``vace_extend`` guesses. What is deliberately
NOT reused is that module's LLM call, its HTTP shape and its registry-heavy
import: ``prompt_writer`` is an injectable seam so this compiler stays pure,
offline and deterministic, and k106/k110 pass the real generator in.

AUDIO-FIRST TIMING (Stage 8): *"The definitive audio timeline precedes final
shot timing. Do not generate arbitrary clip durations and force dialogue onto
them afterward."* ``shot_windows_from_audio`` therefore derives windows FROM the
line timings: boundaries land on line edges or on MEASURED word-gap pauses, and
nowhere else. It pads only into silence that already exists, merges a too-short
window into its neighbour rather than stretching it, and when a long line has no
measured pause to split at, it emits the long window honestly instead of
inventing a cut point. A window shorter than ``min_shot_s`` that cannot merge is
emitted short — k107's ``SHOT_TOO_SHORT`` is the right place to say so, and a
silently padded shot would hide it.

JOINT MODES AND INVARIANT 9. ``SegmentSpec.joint_mode`` records how a segment is
spliced onto the previous one at RENDER time (``cut`` / ``still`` /
``vace_extend``, mirroring ``studio_movie_schema``). That is not a spec
dependency and does not make two specs non-siblings: every field of every spec
is derived from the lock, and the conditioning frame a ``still`` or
``vace_extend`` join needs is an ACCEPTED RENDER — an artifact the orchestrator
supplies at execution, not a prompt. ``to_plan_graph`` accordingly never draws
a segment -> segment edge; :func:`render_dependencies` reports the frame
handoffs separately so k106 can sequence them without smuggling them into
prompt lineage.

No pathlib anywhere. os.path only (not that this module touches the disk).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from hugpy_oracle.audio_master import AudioMaster, LineTiming
from hugpy_oracle.contracts import ArtifactKind, CheckKind, GoalSpec, RepairCode
from hugpy_oracle.plan import (
    AcceptanceTest,
    Edge,
    FrozenParams,
    NodeKind,
    PlanGraph,
    PlanNode,
    Port,
    SEGMENT_PARAM,
    goal_digest,
    sibling_check,
    sibling_violations,
)
from hugpy_oracle.production import (
    ContentAddressed,
    ContinuityBible,
    ContinuityState,
    GenerationSnapshot,
    ProductionError,
    ProductionLock,
    ShotPlan,
    ShotPlanEntry,
    _q,
    _require_text,
    _str_tuple,
)

_EPS = 1e-6

#: Mirrors ``video_intel.prompt_spread.VALID_JOINT_MODES`` /
#: ``studio_movie_schema._VALID_JOINT_MODES`` — keep in sync (there is a test).
#: Ordered ``cut`` first because ``cut`` is this compiler's default and the only
#: legal mode for segment 0, which has no previous shot to carry anything from.
JOINT_MODES: tuple[str, ...] = ("cut", "still", "vace_extend")

#: Joint mode -> the SENTENCE a writer sees. Mirrors
#: ``prompt_spread.JOINT_MODE_PLAIN`` in substance (k93's rule: never show the
#: model a raw token), reworded for a locked-production context where the
#: neighbour is a planned shot rather than an already-written row.
JOINT_MODE_PLAIN: dict[str, str] = {
    "cut": ("this shot is a hard cut — nothing is carried over from the shot "
            "before it, so it may change location or framing freely, but the "
            "characters, wardrobe and world must stay the same"),
    "still": ("this shot begins from a single still frame of the shot before "
              "it; no motion is carried across the join, so it must start from "
              "rest rather than mid-movement"),
    "vace_extend": ("this shot continues the shot before it, carrying its "
                    "motion across the join; it must continue that movement "
                    "rather than restart or reverse it"),
}

#: The capability a segment node names in an emitted ``PlanGraph``. The real
#: catalog name (``catalog.STUDIO_CAPABILITY_NAME[Capability.I2V]``), mirrored so this
#: module does not import the catalog: Stage 15 makes every segment
#: keyframe-first i2v, and k106 may override it per call.
SEGMENT_CAPABILITY = "video.generate.i2v"

#: The id of the locked-parent node in an emitted ``PlanGraph``.
LOCK_NODE_ID = "production_lock"

#: Prefix for segment node ids, so ``graph.node("segment:s1")`` is guessable and
#: a segment node can never collide with the lock node.
SEGMENT_NODE_PREFIX = "segment:"


class SiblingViolation(ProductionError):
    """A ``SegmentSpec`` names another ``SegmentSpec`` as a parent, or a plan
    graph makes one segment depend on another. Invariant 9 / Stage 14.

    ``pairs`` carries ``((child, parent), …)`` so a caller can report every
    place the chain formed, not just the first."""

    def __init__(self, message: str,
                 pairs: Sequence[tuple[str, str]] = ()) -> None:
        super().__init__(message)
        self.pairs = tuple(pairs)


class CompileRefused(ProductionError):
    """``compile_segments`` refused its inputs, and said which one and why."""


def segment_node_id(segment_id: str) -> str:
    return f"{SEGMENT_NODE_PREFIX}{segment_id}"


# ---------------------------------------------------------------------------
# Stage 8 — audio-first shot windows
# ---------------------------------------------------------------------------


def _word_pauses(timing: LineTiming) -> tuple[tuple[float, float], ...]:
    """``((midpoint, width), …)`` for every MEASURED silence inside a line, in
    time order.

    Derived from the round-trip word timings only. A line whose words were
    never measured (this fleet's whisper path returns none today — see the k102
    record) yields an empty tuple, and the caller must then not split it: an
    invented cut point in the middle of a spoken word is exactly the "arbitrary
    clip duration" Stage 8 prohibits. Zero-width word boundaries are not
    pauses and are excluded."""
    out: list[tuple[float, float]] = []
    for previous, following in zip(timing.words, timing.words[1:]):
        width = _q(following.start_s - previous.end_s)
        if width > _EPS:
            out.append((_q(previous.end_s + width / 2.0), width))
    return tuple(out)


def _split_at_pauses(start: float, end: float,
                     pauses: Sequence[tuple[float, float]],
                     min_shot_s: float, max_shot_s: float
                     ) -> tuple[tuple[float, float], ...]:
    """Cut ``[start, end)`` at measured pauses so no piece exceeds
    ``max_shot_s``, or return a single piece when it cannot be done honestly.

    The rule is "cut at the LATEST admissible pause": walk forward, and for each
    piece take the last pause that is at least ``min_shot_s`` in and at most
    ``max_shot_s`` in. That produces the fewest cuts, keeps every piece as close
    to the ceiling as the audio allows, and is deterministic. When no pause sits
    in that window the loop stops and the remainder is emitted whole — a long
    shot that k107 can flag beats a cut through a word."""
    pieces: list[tuple[float, float]] = []
    piece_start = start
    while end - piece_start > max_shot_s + _EPS:
        admissible = [mid for mid, _w in pauses
                      if piece_start + min_shot_s - _EPS <= mid
                      <= piece_start + max_shot_s + _EPS]
        if not admissible:
            break
        cut = admissible[-1]
        pieces.append((piece_start, cut))
        piece_start = cut
    pieces.append((piece_start, end))
    # A tail shorter than the floor folds back into the piece before it: better
    # one slightly long shot than a two-frame orphan.
    if len(pieces) > 1 and pieces[-1][1] - pieces[-1][0] + _EPS < min_shot_s:
        tail = pieces.pop()
        pieces[-1] = (pieces[-1][0], tail[1])
    return tuple(pieces)


def shot_windows_from_audio(audio_master: AudioMaster, *,
                            min_shot_s: float = 1.0,
                            max_shot_s: float = 8.0,
                            pad_s: float = 0.0,
                            ) -> tuple[tuple[float, float, tuple[str, ...]], ...]:
    """Doc Stage 8 — derive shot windows FROM the locked audio timeline.

    Returns ``((start_s, end_s, line_ids), …)`` in timeline order, covering
    every line exactly once, never overlapping. Every boundary lands on a line
    edge, on existing silence, or on a MEASURED word-gap pause; none is
    computed from a target clip length.

    * ``pad_s`` extends a window into silence that ALREADY EXISTS — at most the
      gap before the line and at most the pause the line holds after itself.
      Padding never eats a neighbour's audio, so windows stay disjoint.
    * A line longer than ``max_shot_s`` is split at its measured pauses; with
      no measured pause in range it is emitted whole (see ``_split_at_pauses``).
    * A window shorter than ``min_shot_s`` merges FORWARD into the next one
      while the union still fits under ``max_shot_s``; a short final window
      merges backward. A short window that cannot merge without breaking the
      ceiling is emitted SHORT — the honest answer, and the one k107 turns into
      ``SHOT_TOO_SHORT``.

    Raises ``ValueError`` on nonsense bounds; returns ``()`` for a master with
    no line timings."""
    if not isinstance(audio_master, AudioMaster):
        raise TypeError(f"shot_windows_from_audio takes an AudioMaster, got "
                        f"{type(audio_master).__name__}")
    if float(min_shot_s) <= 0:
        raise ValueError(f"min_shot_s must be positive, got {min_shot_s}")
    if float(max_shot_s) < float(min_shot_s):
        raise ValueError(f"max_shot_s {max_shot_s} must be >= min_shot_s "
                         f"{min_shot_s}")
    if float(pad_s) < 0:
        raise ValueError(f"pad_s must be non-negative, got {pad_s}")
    min_shot_s, max_shot_s, pad_s = (float(min_shot_s), float(max_shot_s),
                                     float(pad_s))
    timings = audio_master.line_timings
    if not timings:
        return ()

    # 1. one padded window per line; padding only consumes existing silence.
    pieces: list[tuple[float, float, tuple[str, ...]]] = []
    previous_end = 0.0
    for timing in timings:
        lead = max(0.0, timing.start_s - previous_end)
        start = _q(timing.start_s - min(pad_s, lead))
        end = _q(timing.end_s + min(pad_s, timing.pause_after_s))
        # 2. split a long line at its measured pauses; never anywhere else.
        cuts = _split_at_pauses(start, end, _word_pauses(timing),
                                min_shot_s, max_shot_s)
        for piece_start, piece_end in cuts:
            pieces.append((_q(piece_start), _q(piece_end), (timing.line_id,)))
        previous_end = timing.next_start_s

    # 3. merge windows under the floor, forward first.
    merged: list[list[Any]] = []
    for start, end, line_ids in pieces:
        if merged and (merged[-1][1] - merged[-1][0]) + _EPS < min_shot_s \
                and end - merged[-1][0] <= max_shot_s + _EPS:
            merged[-1][1] = end
            merged[-1][2] = tuple(dict.fromkeys(merged[-1][2] + line_ids))
            continue
        merged.append([start, end, line_ids])
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) + _EPS < min_shot_s \
            and merged[-1][1] - merged[-2][0] <= max_shot_s + _EPS:
        tail = merged.pop()
        merged[-1][1] = tail[1]
        merged[-1][2] = tuple(dict.fromkeys(merged[-1][2] + tail[2]))

    windows = tuple((_q(s), _q(e), tuple(ids)) for s, e, ids in merged)

    # Internal invariant, asserted rather than assumed: the windows partition
    # the timeline in order. If padding or merging ever broke that, a locked
    # plan built on it would silently double-cover a line.
    for first, second in zip(windows, windows[1:]):
        if first[1] > second[0] + _EPS:
            raise ValueError(
                f"shot_windows_from_audio produced overlapping windows "
                f"{first} and {second} — this is a bug in the window walk, "
                f"not in the audio")
    covered = [l for _s, _e, ids in windows for l in ids]
    if sorted(set(covered)) != sorted(set(audio_master.line_ids)):
        raise ValueError(
            f"shot_windows_from_audio dropped or invented lines: covered "
            f"{sorted(set(covered))} vs master {sorted(set(audio_master.line_ids))}")
    return windows


def shot_plan_from_windows(windows: Iterable[tuple[float, float, Sequence[str]]],
                           *, rubric: Sequence[str],
                           camera: Mapping[str, Any] | None = None,
                           blocking: str | None = None,
                           lighting: str | None = None,
                           prefix: str = "s") -> ShotPlan:
    """A minimal ``ShotPlan`` over audio-derived windows — the seam k106 uses
    before k110's real shot plan exists.

    Every entry gets the SAME camera/blocking/lighting and the same rubric,
    because this helper knows nothing about direction and must not pretend to.
    Segment ids are ``s1, s2, …`` in timeline order."""
    entries = []
    for index, (start, end, line_ids) in enumerate(windows, start=1):
        entries.append(ShotPlanEntry(segment_id=f"{prefix}{index}",
                                     line_ids=tuple(line_ids),
                                     start_s=start, end_s=end,
                                     camera=FrozenParams(camera or {}),
                                     blocking=blocking, lighting=lighting,
                                     rubric=tuple(rubric)))
    return ShotPlan(entries=tuple(entries))


# ---------------------------------------------------------------------------
# The locked context — everything a prompt writer is allowed to see
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LockedSegmentBrief:
    """One segment as the prompt writer sees it: LOCKED material only.

    Note what is absent and cannot be added without changing this class: a
    prompt, a negative prompt, a spec digest, a render, a candidate. Those are
    outputs of the run, and Stage 4 forbids a run output from becoming a
    sibling input. The writer sees the planned shot, not the written shot."""

    segment_id: str
    index: int
    scene_ref: str | None = None
    start_s: float = 0.0
    end_s: float = 0.0
    line_ids: tuple[str, ...] = ()
    lines: tuple[tuple[str, str], ...] = ()   # (line_id, locked dialogue text)
    camera: Mapping[str, Any] = field(default_factory=FrozenParams)
    blocking: str | None = None
    lighting: str | None = None
    joint_mode: str = "cut"
    beat: str | None = None
    rubric: tuple[str, ...] = ()
    state_before: Mapping[str, Any] = field(default_factory=FrozenParams)
    state_after: Mapping[str, Any] = field(default_factory=FrozenParams)
    spatial_ref: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.segment_id, "LockedSegmentBrief.segment_id")
        if isinstance(self.index, bool) or not isinstance(self.index, int) \
                or self.index < 0:
            raise ValueError(f"LockedSegmentBrief.index must be a non-negative "
                             f"int, got {self.index!r}")
        object.__setattr__(self, "camera", FrozenParams(self.camera))
        object.__setattr__(self, "state_before", FrozenParams(self.state_before))
        object.__setattr__(self, "state_after", FrozenParams(self.state_after))
        object.__setattr__(self, "line_ids", tuple(self.line_ids))
        object.__setattr__(self, "lines",
                           tuple((str(a), str(b)) for a, b in self.lines))
        object.__setattr__(self, "rubric", tuple(self.rubric))
        if self.joint_mode not in JOINT_MODES:
            raise ValueError(f"LockedSegmentBrief.joint_mode must be one of "
                             f"{list(JOINT_MODES)}, got {self.joint_mode!r}")

    @property
    def duration_s(self) -> float:
        return _q(self.end_s - self.start_s)

    @property
    def dialogue(self) -> str:
        """The locked lines of this shot, one per line, as plain text."""
        return "\n".join(text for _line_id, text in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "index": self.index,
                "scene_ref": self.scene_ref, "start_s": self.start_s,
                "end_s": self.end_s, "line_ids": list(self.line_ids),
                "lines": [list(l) for l in self.lines],
                "camera": FrozenParams(self.camera).to_dict(),
                "blocking": self.blocking, "lighting": self.lighting,
                "joint_mode": self.joint_mode, "beat": self.beat,
                "rubric": list(self.rubric),
                "state_before": FrozenParams(self.state_before).to_dict(),
                "state_after": FrozenParams(self.state_after).to_dict(),
                "spatial_ref": self.spatial_ref}


@dataclass(frozen=True, slots=True)
class LockedContext(ContentAddressed):
    """The whole locked production, as one immutable object handed to the
    prompt writer — the SAME object for every index.

    This is k93's spread idiom made typed: every segment sees the full
    timeline, one shared world, and the only thing that varies per call is the
    index. The difference is where the context comes from. In the studio spread
    it is the rows the operator did not select; here it is the production lock,
    which is what makes the result a Stage 14 sibling set rather than a chain.

    ``segments`` holds :class:`LockedSegmentBrief`, which structurally cannot
    carry a generated prompt. That is the enforcement: there is no field for
    the thing invariant 9 prohibits."""

    lock_digest: str
    deliverable: str = ""
    exclusions: tuple[str, ...] = ()
    identity_refs: tuple[str, ...] = ()
    characters: tuple[str, ...] = ()
    wardrobe: tuple[str, ...] = ()
    props: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    notes: str = ""
    tone: float = 0.5
    registry_version: str | None = None
    segments: tuple[LockedSegmentBrief, ...] = ()
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.lock_digest, "LockedContext.lock_digest")
        object.__setattr__(self, "segments", tuple(self.segments))
        for brief in self.segments:
            if not isinstance(brief, LockedSegmentBrief):
                raise TypeError(f"LockedContext.segments takes "
                                f"LockedSegmentBrief, got {type(brief).__name__}")
        for name in ("exclusions", "identity_refs", "characters", "wardrobe",
                     "props", "locations", "parents"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not 0.0 <= float(self.tone) <= 1.0:
            raise ValueError(f"LockedContext.tone must be in [0, 1], got "
                             f"{self.tone}")
        object.__setattr__(self, "tone", float(self.tone))

    def __len__(self) -> int:
        return len(self.segments)

    def brief(self, index: int) -> LockedSegmentBrief:
        """The brief for ``index``. Negative indices are REFUSED: ``ctx.brief(i
        - 1)`` is precisely the chaining reflex this module exists to prevent,
        and Python would silently hand back the last segment for ``-1``."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"LockedContext.brief takes an int index, got "
                            f"{type(index).__name__}")
        if index < 0:
            raise IndexError(
                f"LockedContext.brief({index}): negative indices are refused. "
                f"Reaching backwards from a segment is the Stage 14 chain "
                f"({index} would silently return another segment); every "
                f"segment is compiled from the LOCK, not from its neighbour")
        if index >= len(self.segments):
            raise IndexError(f"LockedContext has {len(self.segments)} "
                             f"segment(s); no index {index}")
        return self.segments[index]

    def preface(self, index: int) -> str:
        """The locked production rendered for a writer, with this shot marked.

        Joint modes become SENTENCES (k93's rule: a model shown the raw token
        ``vace_extend`` guesses; a model shown the sentence does not). The whole
        timeline is shown because that is what makes the segments read as one
        film, and it is safe here precisely because every line of it comes from
        the lock."""
        target = self.brief(index)
        blocks: list[str] = []
        head = [f"DELIVERABLE: {self.deliverable}" if self.deliverable else "",
                f"TONE: {self.tone:.2f}",
                ("CHARACTERS: " + ", ".join(self.characters)) if self.characters else "",
                ("WARDROBE: " + ", ".join(self.wardrobe)) if self.wardrobe else "",
                ("PROPS: " + ", ".join(self.props)) if self.props else "",
                ("LOCATIONS: " + ", ".join(self.locations)) if self.locations else "",
                ("IDENTITIES: " + ", ".join(self.identity_refs)) if self.identity_refs else "",
                ("DO NOT INCLUDE: " + ", ".join(self.exclusions)) if self.exclusions else "",
                ("CONTINUITY NOTES: " + self.notes) if self.notes else ""]
        blocks.append("\n".join(line for line in head if line))
        rows: list[str] = ["THE WHOLE FILM (locked):"]
        for brief in self.segments:
            marker = ">>" if brief.index == index else "  "
            rows.append(f"{marker} [{brief.index}] {brief.segment_id} "
                        f"{brief.start_s:.3f}s-{brief.end_s:.3f}s"
                        + (f" scene {brief.scene_ref}" if brief.scene_ref else ""))
            if brief.beat:
                rows.append(f"      beat: {brief.beat}")
            if brief.lines:
                for _line_id, text in brief.lines:
                    rows.append(f"      line: {text}")
            if brief.camera:
                rows.append("      camera: " + ", ".join(
                    f"{k}={brief.camera[k]}" for k in sorted(brief.camera)))
            if brief.blocking:
                rows.append(f"      blocking: {brief.blocking}")
            if brief.lighting:
                rows.append(f"      lighting: {brief.lighting}")
            rows.append(f"      join: {JOINT_MODE_PLAIN[brief.joint_mode]}")
            if brief.state_before:
                rows.append("      state before: " + ", ".join(
                    f"{k}={brief.state_before[k]}" for k in sorted(brief.state_before)))
            if brief.state_after:
                rows.append("      state after: " + ", ".join(
                    f"{k}={brief.state_after[k]}" for k in sorted(brief.state_after)))
        blocks.append("\n".join(rows))
        blocks.append(f"WRITE THE SHOT MARKED >> ({target.segment_id}). "
                      f"It must satisfy: " + "; ".join(target.rubric)
                      if target.rubric else
                      f"WRITE THE SHOT MARKED >> ({target.segment_id}).")
        return "\n\n".join(b for b in blocks if b)

    def to_dict(self) -> dict[str, Any]:
        return {"lock_digest": self.lock_digest,
                "deliverable": self.deliverable,
                "exclusions": list(self.exclusions),
                "identity_refs": list(self.identity_refs),
                "characters": list(self.characters),
                "wardrobe": list(self.wardrobe),
                "props": list(self.props),
                "locations": list(self.locations),
                "notes": self.notes, "tone": self.tone,
                "registry_version": self.registry_version,
                "segments": [s.to_dict() for s in self.segments],
                "parents": list(self.parents)}


PromptWriter = Callable[[LockedContext, int], str]


def default_prompt_writer(context: LockedContext, index: int) -> str:
    """A deterministic template, and honestly nothing more.

    It writes a legible sentence out of the locked shot — subject, action,
    camera, light — so the whole pipeline can be exercised, digested and tested
    with no model in the loop. It is not a screenwriter: k110's LLM pass is,
    and it arrives through the ``prompt_writer`` seam with the same signature.
    The default is pure so two runs of the same lock produce the same specs."""
    brief = context.brief(index)
    parts: list[str] = []
    if brief.scene_ref:
        parts.append(brief.scene_ref)
    if context.locations:
        parts.append(context.locations[0])
    if context.characters:
        parts.append(", ".join(context.characters))
    if brief.beat:
        parts.append(brief.beat)
    camera_bits = [f"{k} {brief.camera[k]}" for k in sorted(brief.camera)]
    if camera_bits:
        parts.append("camera: " + ", ".join(camera_bits))
    if brief.blocking:
        parts.append(brief.blocking)
    if brief.lighting:
        parts.append(brief.lighting)
    if brief.lines:
        parts.append("speaking: " + " / ".join(t for _i, t in brief.lines))
    parts.append(f"held for {brief.duration_s:.2f}s")
    parts.append(JOINT_MODE_PLAIN[brief.joint_mode])
    return ". ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# SegmentSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentSpec(ContentAddressed):
    """Doc Stage 14's composition, as one immutable artifact::

        SegmentSpec[i] = ScreenplayScene[i] + ContinuityState[i] + AudioWindow[i]
                       + ShotPlan[i] + SpatialSceneManifest[i] + ToneProfile[i]
                       + AcceptanceRubric[i]

    Every field is derived from the production lock's inputs. ``parents`` names
    the lock-side digests it was derived FROM, and only those — never another
    ``SegmentSpec``'s digest, which is what makes the set siblings rather than
    a chain.

    ``spatial_ref`` is ``<tool>:<path>#sha256:<source hash>`` when the goal or
    recipe supplied a scene source for the shot (glTF / GLB / USDA / pose
    track, or a ready ``SpatialSceneManifest``; see
    :func:`resolve_spatial_ref`), validated before any prompt was written. It
    is ``None`` when nothing was supplied, and ``None`` reads as "no spatial
    authority for this shot", never as "unconstrained and therefore fine".

    ``rubric`` must be non-empty: Stage 9 gives every shot an acceptance rubric,
    and a shot with no rubric cannot be judged, which under invariant 11 means
    it cannot be accepted."""

    segment_id: str
    index: int
    scene_ref: str | None
    continuity: ContinuityState
    audio_window: tuple[float, float, tuple[str, ...]]
    shot: ShotPlanEntry
    spatial_ref: str | None
    tone: float
    rubric: tuple[str, ...]
    prompt: str
    negative_prompt: str | None
    identity_refs: tuple[str, ...]
    joint_mode: str
    seed_base: int
    lock_digest: str
    parents: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.segment_id, "SegmentSpec.segment_id")
        if isinstance(self.index, bool) or not isinstance(self.index, int) \
                or self.index < 0:
            raise ValueError(f"SegmentSpec.index must be a non-negative int, "
                             f"got {self.index!r}")
        if not isinstance(self.continuity, ContinuityState):
            raise TypeError(f"SegmentSpec.continuity takes a ContinuityState, "
                            f"got {type(self.continuity).__name__}")
        if self.continuity.segment_id != self.segment_id:
            raise ValueError(
                f"SegmentSpec({self.segment_id}) carries the continuity state "
                f"of {self.continuity.segment_id!r} — a segment reading another "
                f"segment's continuity is how a chain starts")
        if not isinstance(self.shot, ShotPlanEntry):
            raise TypeError(f"SegmentSpec.shot takes a ShotPlanEntry, got "
                            f"{type(self.shot).__name__}")
        if self.shot.segment_id != self.segment_id:
            raise ValueError(f"SegmentSpec({self.segment_id}) carries the shot "
                             f"plan entry of {self.shot.segment_id!r}")
        start, end, line_ids = tuple(self.audio_window)
        start, end = _q(start), _q(end)
        if start < 0:
            raise ValueError(f"SegmentSpec({self.segment_id}).audio_window "
                             f"starts before zero: {start}")
        if end < start:
            raise ValueError(f"SegmentSpec({self.segment_id}).audio_window ends "
                             f"before it starts: {end} < {start}")
        object.__setattr__(self, "audio_window",
                           (start, end, _str_tuple(line_ids,
                                                   "SegmentSpec.audio_window "
                                                   "line ids")
                            if line_ids else ()))
        if self.spatial_ref is not None:
            object.__setattr__(self, "spatial_ref",
                               _require_text(self.spatial_ref,
                                             "SegmentSpec.spatial_ref"))
        if not 0.0 <= float(self.tone) <= 1.0:
            raise ValueError(f"SegmentSpec({self.segment_id}).tone must be in "
                             f"[0, 1], got {self.tone}")
        object.__setattr__(self, "tone", float(self.tone))
        object.__setattr__(self, "rubric",
                           _str_tuple(self.rubric, "SegmentSpec.rubric"))
        if not self.rubric:
            raise ValueError(
                f"SegmentSpec({self.segment_id}) has no acceptance rubric; "
                f"Stage 9 gives every shot one, and a shot that cannot be "
                f"judged cannot be accepted (invariant 11)")
        _require_text(self.prompt, f"SegmentSpec({self.segment_id}).prompt")
        if self.negative_prompt is not None:
            object.__setattr__(self, "negative_prompt",
                               _require_text(self.negative_prompt,
                                             "SegmentSpec.negative_prompt"))
        object.__setattr__(self, "identity_refs",
                           _str_tuple(self.identity_refs,
                                      "SegmentSpec.identity_refs"))
        if self.joint_mode not in JOINT_MODES:
            raise ValueError(f"SegmentSpec({self.segment_id}).joint_mode must "
                             f"be one of {list(JOINT_MODES)}, got "
                             f"{self.joint_mode!r}")
        if self.index == 0 and self.joint_mode != "cut":
            raise ValueError(
                f"SegmentSpec({self.segment_id}) is index 0 and cannot join "
                f"with {self.joint_mode!r}: there is no previous shot to carry "
                f"a frame or motion from, so the only honest mode is 'cut'")
        if isinstance(self.seed_base, bool) or \
                not isinstance(self.seed_base, int) or \
                not 0 <= self.seed_base < 2 ** 32:
            raise ValueError(f"SegmentSpec({self.segment_id}).seed_base must be "
                             f"an int in [0, 2**32), got {self.seed_base!r}")
        _require_text(self.lock_digest, "SegmentSpec.lock_digest")
        object.__setattr__(self, "parents",
                           _str_tuple(self.parents, "SegmentSpec.parents"))
        if not self.parents:
            raise ValueError(
                f"SegmentSpec({self.segment_id}) has no parents; a spec that "
                f"names nothing it was derived from has no lineage (invariant "
                f"4) and cannot be checked against the lock")

    # -- reading -----------------------------------------------------------

    @property
    def start_s(self) -> float:
        return self.audio_window[0]

    @property
    def end_s(self) -> float:
        return self.audio_window[1]

    @property
    def line_ids(self) -> tuple[str, ...]:
        return self.audio_window[2]

    @property
    def duration_s(self) -> float:
        return _q(self.end_s - self.start_s)

    @property
    def needs_previous_frames(self) -> bool:
        """True when this shot's RENDER conditions on the previous shot's
        accepted frames. Not a spec dependency: the spec was compiled from the
        lock either way, and the frames are an artifact the orchestrator
        supplies at execution time. See :func:`render_dependencies`."""
        return self.joint_mode != "cut"

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "index": self.index,
            "scene_ref": self.scene_ref,
            "continuity": self.continuity.to_dict(),
            "audio_window": [self.audio_window[0], self.audio_window[1],
                             list(self.audio_window[2])],
            "shot": self.shot.to_dict(),
            "spatial_ref": self.spatial_ref,
            "tone": self.tone,
            "rubric": list(self.rubric),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "identity_refs": list(self.identity_refs),
            "joint_mode": self.joint_mode,
            "seed_base": self.seed_base,
            "lock_digest": self.lock_digest,
            "parents": list(self.parents),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SegmentSpec":
        window = d.get("audio_window") or (0.0, 0.0, ())
        return cls(
            segment_id=d["segment_id"],
            index=int(d["index"]),
            scene_ref=d.get("scene_ref"),
            continuity=ContinuityState.from_dict(d["continuity"]),
            audio_window=(window[0], window[1], tuple(window[2])),
            shot=ShotPlanEntry.from_dict(d["shot"]),
            spatial_ref=d.get("spatial_ref"),
            tone=d.get("tone", 0.5),
            rubric=tuple(d.get("rubric", ())),
            prompt=d["prompt"],
            negative_prompt=d.get("negative_prompt"),
            identity_refs=tuple(d.get("identity_refs", ())),
            joint_mode=d.get("joint_mode", "cut"),
            seed_base=int(d.get("seed_base", 0)),
            lock_digest=d["lock_digest"],
            parents=tuple(d.get("parents", ())),
        )


def segment_seed(lock_digest: str, segment_id: str, seed_salt: int = 0) -> int:
    """A deterministic ``seed_base`` for one segment of one lock.

    Derived from the lock digest, so the same locked production always draws
    the same seeds and a repair re-renders the same shot rather than a new one;
    ``seed_salt`` is the dial that asks for a genuinely different take while
    staying reproducible (the same idiom as k102's ``candidate_seed``). Range is
    ``[0, 2**32)`` — what numpy and torch accept."""
    payload = f"{lock_digest}:{segment_id}:{int(seed_salt)}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Invariant 9 — the sibling check over compiled specs
# ---------------------------------------------------------------------------


def sibling_violations_in(specs: Sequence[SegmentSpec]
                          ) -> tuple[tuple[str, str], ...]:
    """``((child, parent), …)`` for every spec that names ANOTHER spec's digest
    in its ``parents``. Empty means the set is a Stage 14 sibling set."""
    by_digest = {spec.digest: spec.segment_id for spec in specs}
    out: list[tuple[str, str]] = []
    for spec in specs:
        for parent in spec.parents:
            owner = by_digest.get(parent)
            if owner is not None and owner != spec.segment_id:
                out.append((spec.segment_id, owner))
    return tuple(out)


def assert_siblings(specs: Sequence[SegmentSpec], *,
                    lock: ProductionLock | None = None) -> None:
    """Invariant 9 over a compiled set. Raises :class:`SiblingViolation`.

    Three checks, in the order they can go wrong:

    1. No spec names another spec's digest as a parent (the prohibition).
    2. Every spec shares one ``lock_digest`` — a set compiled against two locks
       is not a sibling set, it is two productions spliced together.
    3. With ``lock`` supplied, every parent is in ``lock.parent_digests``: not
       merely "not a sibling" but positively one of the locked artifacts, so a
       parent from anywhere else is caught too."""
    specs = tuple(specs)
    if not specs:
        return
    pairs = sibling_violations_in(specs)
    if pairs:
        rendered = ", ".join(f"{child} <- {parent}" for child, parent in pairs)
        raise SiblingViolation(
            f"segment spec(s) name a sibling as a parent ({rendered}); Stage 14 "
            f"requires locked artifacts -> S1, S2, S3 and prohibits "
            f"S1 -> S2 -> S3 (invariant 9)", pairs=pairs)
    digests = {spec.lock_digest for spec in specs}
    if len(digests) > 1:
        raise SiblingViolation(
            f"segment specs were compiled against {len(digests)} different "
            f"production locks {sorted(digests)}; siblings share one lock")
    if lock is not None:
        allowed = set(lock.parent_digests)
        for spec in specs:
            stray = [p for p in spec.parents if p not in allowed]
            if stray:
                raise SiblingViolation(
                    f"segment {spec.segment_id} names parent(s) "
                    f"{[p[:12] + '…' for p in stray]} that are not locked "
                    f"artifacts of this production; a spec's parents are the "
                    f"lock and what it locked, and nothing else")


def render_dependencies(specs: Sequence[SegmentSpec]
                        ) -> tuple[tuple[str, str], ...]:
    """``((segment, needs_frames_from), …)`` — the RENDER-time frame handoffs a
    ``still`` / ``vace_extend`` join implies, in timeline order.

    Reported separately and deliberately kept OUT of the plan graph: the frame
    a join conditions on is an accepted render (an artifact), not a prompt, so
    it is not the lineage invariant 9 governs. k106 sequences on this; the
    sibling check never sees it."""
    ordered = sorted(specs, key=lambda s: s.index)
    out: list[tuple[str, str]] = []
    for previous, spec in zip(ordered, ordered[1:]):
        if spec.needs_previous_frames:
            out.append((spec.segment_id, previous.segment_id))
    return tuple(out)


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# spatial refs (or-k8): a goal/recipe may supply, per segment, a scene SOURCE
# (glTF / GLB / USDA / pose-track JSON path), a ``SpatialSceneManifest``, a
# ``spatial_sources.SpatialSource`` or a ready ref string. Sources are parsed
# and validated HERE, before any prompt is written, and ``SegmentSpec
# .spatial_ref`` carries ``<tool>:<path>#sha256:<source hash>`` — a content
# address of the SOURCE, so re-reading an unchanged file yields the same ref
# and a changed file changes it. The resolved ``SpatialSource`` is kept
# in an in-process registry keyed by ref (``spatial_source_for``) so the stages
# downstream (segment_context, the recipe's ``spatial:`` node) can read the
# camera track and entity trajectories without a second parse.
# ---------------------------------------------------------------------------

_SPATIAL_EXTS = (".gltf", ".glb", ".usd", ".usda", ".usdc", ".usdz", ".json")
#: ref -> segment_id -> SpatialSource. Two segments may share one scene file
#: (same ref) and still own distinct manifests (segment_id is in the digest).
_SPATIAL_REGISTRY: dict[str, dict[str, Any]] = {}


def _register(src: Any) -> str:
    _SPATIAL_REGISTRY.setdefault(src.ref, {})[src.manifest.segment_id] = src
    return src.ref


def spatial_source_for(ref: str | None, segment_id: str | None = None, *,
                       run_id: str = "rederive") -> Any:
    """The ``SpatialSource`` behind a ``spatial_ref``, or ``None``.

    A ref that names a path is re-derived on a registry miss (e.g. after a
    resume); the re-derived file must carry the sha256 the ref names, or the
    scene changed after the lock and the call is refused (invariant 4)."""
    if not ref:
        return None
    bucket = _SPATIAL_REGISTRY.get(ref) or {}
    if segment_id is not None and segment_id in bucket:
        return bucket[segment_id]
    if bucket and segment_id is None:
        return next(iter(bucket.values()))
    _tool, _, rest = ref.partition(":")
    path, _, digest = rest.rpartition("#")
    if not path or path == "-" or not digest.startswith("sha256:"):
        return None
    import os
    if not os.path.isfile(path):
        return None
    from hugpy_oracle.spatial_sources import load_source
    src = load_source(path, run_id=run_id, segment_id=segment_id or "rederive")
    if src.source_sha256 != digest:
        raise CompileRefused(
            f"spatial source {path} now hashes to {src.source_sha256[:19]}…, "
            f"not the {digest[:19]}… its spatial_ref names; the scene file "
            f"changed after the lock (invariant 4)")
    _register(src)
    return src


def resolve_spatial_ref(value: Any, *, run_id: str, segment_id: str,
                        fps: float | None = None,
                        known_entity_ids: Iterable[str] | None = None) -> str | None:
    """Turn whatever a goal supplied for one segment into a ``spatial_ref``.

    * ``None`` / ``""`` -> ``None`` (honest: no spatial authority).
    * a ``SpatialSource`` -> its ``ref`` (registered).
    * a ``SpatialSceneManifest`` -> ``manifest:-#sha256:<manifest digest>``; it is
      validated first and registered as a track-less source.
    * a mapping with ``path`` (+ optional producer kwargs) or a bare path
      string ending in a known scene extension -> parsed via
      ``spatial_sources.load_source``, validated, registered.
    * any other string is taken as an already-minted ref and passed through.

    A source that fails to parse or validate raises ``CompileRefused`` naming
    the segment and the faults: a spec with a broken spatial ref would claim
    a geometric authority it does not have."""
    if value is None or value == "":
        return None
    from hugpy_oracle import spatial as sp
    from hugpy_oracle import spatial_sources as ss
    if isinstance(value, ss.SpatialSource):
        return _register(value)
    if isinstance(value, sp.SpatialSceneManifest):
        val = sp.validate_manifest(value, known_entity_ids=known_entity_ids)
        if not val.ok:
            raise CompileRefused(
                f"spatial manifest for {segment_id!r} is not admissible: "
                + "; ".join(f"{f.code.value}@{f.where}: {f.message}" for f in val.faults))
        return _register(ss.SpatialSource(value, (), (), "manifest", None, "sha256:" + value.digest, val))
    kwargs: dict[str, Any] = {}
    if isinstance(value, Mapping):
        kwargs = dict(value)
        path = kwargs.pop("path", None) or kwargs.pop("source", None)
        if not path:
            raise CompileRefused(f"spatial source mapping for {segment_id!r} has no 'path'")
        value = str(path)
    if not isinstance(value, str):
        raise CompileRefused(f"spatial_refs[{segment_id!r}] must be a path, manifest, "
                             f"SpatialSource or ref string, got {type(value).__name__}")
    if not value.lower().endswith(_SPATIAL_EXTS):
        return value            # an already-minted ref (or foreign handle); passed through
    if fps is not None and "fps" not in kwargs:
        kwargs["fps"] = fps
    if known_entity_ids is not None and "known_entity_ids" not in kwargs:
        kwargs["known_entity_ids"] = tuple(known_entity_ids)
    try:
        src = ss.load_source(value, run_id=run_id, segment_id=segment_id, **kwargs)
    except NotImplementedError as exc:
        raise CompileRefused(f"spatial source for {segment_id!r} unsupported: {exc}") from exc
    except (ss.SpatialSourceError, OSError, ValueError, KeyError, TypeError) as exc:
        raise CompileRefused(f"spatial source for {segment_id!r} refused: {exc}") from exc
    return _register(src)


def _per_segment(value: Any, segment_ids: Sequence[str], what: str,
                 default: Any = None) -> tuple[Any, ...]:
    """Normalize a per-segment argument given as a mapping, a sequence, a
    single value, or None into a tuple aligned with ``segment_ids``."""
    if value is None:
        return tuple(default for _ in segment_ids)
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(segment_ids))
        if unknown:
            raise CompileRefused(f"{what} names unknown segment(s) {unknown}; "
                                 f"the shot plan has {list(segment_ids)}")
        return tuple(value.get(sid, default) for sid in segment_ids)
    if isinstance(value, (str, int, float)) or not isinstance(value, Sequence):
        return tuple(value for _ in segment_ids)
    items = tuple(value)
    if len(items) != len(segment_ids):
        raise CompileRefused(f"{what} has {len(items)} entries for "
                             f"{len(segment_ids)} segment(s)")
    return items


def build_locked_context(lock: ProductionLock, *,
                         snapshot: GenerationSnapshot,
                         audio_master: AudioMaster,
                         continuity: ContinuityBible,
                         shot_plan: ShotPlan,
                         tone: float,
                         identity_refs: Sequence[str] | None = None,
                         dialogue: Mapping[str, str] | None = None,
                         scene_refs: Any = None,
                         joint_modes: Any = None,
                         spatial_refs: Any = None,
                         beats: Any = None) -> LockedContext:
    """Assemble the one context every prompt in this run is written from.

    Split out of :func:`compile_segments` so a caller (k106, a route, a test)
    can inspect exactly what a writer will see before spending a model call on
    it — and so the "same object for every index" property is a fact about a
    value, not a claim about a loop."""
    segment_ids = shot_plan.segment_ids
    scenes = _per_segment(scene_refs, segment_ids, "scene_refs")
    modes = _per_segment(joint_modes, segment_ids, "joint_modes", default="cut")
    spatials = tuple(
        resolve_spatial_ref(raw, run_id=lock.digest, segment_id=sid)
        for raw, sid in zip(_per_segment(spatial_refs, segment_ids, "spatial_refs"),
                            segment_ids))
    beat_texts = _per_segment(beats, segment_ids, "beats")
    dialogue = dict(dialogue or {})

    briefs: list[LockedSegmentBrief] = []
    for index, entry in enumerate(shot_plan.entries):
        state = continuity.state(entry.segment_id)
        mode = "cut" if index == 0 else str(modes[index] or "cut")
        briefs.append(LockedSegmentBrief(
            segment_id=entry.segment_id, index=index,
            scene_ref=scenes[index], start_s=entry.start_s, end_s=entry.end_s,
            line_ids=entry.line_ids,
            lines=tuple((line_id, dialogue[line_id]) for line_id in entry.line_ids
                        if line_id in dialogue),
            camera=entry.camera, blocking=entry.blocking,
            lighting=entry.lighting, joint_mode=mode, beat=beat_texts[index],
            rubric=entry.rubric, state_before=state.state_before,
            state_after=state.state_after, spatial_ref=spatials[index]))

    refs = (_str_tuple(identity_refs, "identity_refs")
            if identity_refs is not None else lock.identity_refs)
    return LockedContext(
        lock_digest=lock.digest, deliverable=snapshot.deliverable,
        exclusions=snapshot.exclusions, identity_refs=refs,
        characters=continuity.characters, wardrobe=continuity.wardrobe,
        props=continuity.props, locations=continuity.locations,
        notes=continuity.notes, tone=float(tone),
        registry_version=lock.registry_version, segments=tuple(briefs),
        parents=lock.parent_digests)


def compile_segments(lock: ProductionLock, *,
                     snapshot: GenerationSnapshot,
                     audio_master: AudioMaster,
                     continuity: ContinuityBible,
                     shot_plan: ShotPlan,
                     tone: float,
                     identity_refs: Sequence[str] | None = None,
                     prompt_writer: PromptWriter | None = None,
                     negative_prompt: str | None = None,
                     dialogue: Mapping[str, str] | None = None,
                     scene_refs: Any = None,
                     joint_modes: Any = None,
                     spatial_refs: Any = None,
                     beats: Any = None,
                     seed_salt: int = 0) -> tuple[SegmentSpec, ...]:
    """Doc Stage 14 — compile every ``SegmentSpec`` independently.

    ``prompt_writer`` is ``(locked_context, index) -> str`` and is called with
    the SAME context object for every index; when omitted,
    :func:`default_prompt_writer` writes a deterministic template. k110's LLM
    pass and k93's ``prompt_spread`` both fit that signature, which is the
    point: the compiler cannot hand a writer a previous spec, whatever the
    writer would like to do with one.

    Before anything is written, the artifacts are checked against the lock BY
    DIGEST. That is the payoff of content addressing: a caller that locked one
    audio master and then passed a re-synthesized one gets a refusal naming the
    artifact, instead of a production quietly compiled against the wrong take.

    Returns the specs in shot-plan order, after ``assert_siblings`` has passed
    on them."""
    if not isinstance(lock, ProductionLock):
        raise TypeError(f"compile_segments takes a ProductionLock, got "
                        f"{type(lock).__name__}")
    for label, artifact, expected in (
            ("snapshot", snapshot, lock.snapshot_digest),
            ("continuity", continuity, lock.continuity_digest),
            ("audio_master", audio_master, lock.audio_master_digest),
            ("shot_plan", shot_plan, lock.shot_plan_digest)):
        actual = getattr(artifact, "digest", None)
        if actual != expected:
            raise CompileRefused(
                f"{label} does not match the production lock: locked "
                f"{str(expected)[:12]}… but was given {str(actual)[:12]}… — "
                f"compiling against an artifact the lock never saw would make "
                f"the lock a lie (invariant 4)")
    if not shot_plan.entries:
        raise CompileRefused("the shot plan is empty — nothing to compile")
    if not 0.0 <= float(tone) <= 1.0:
        raise CompileRefused(f"tone must be in [0, 1], got {tone}")
    refs = (_str_tuple(identity_refs, "identity_refs")
            if identity_refs is not None else lock.identity_refs)
    stray = [r for r in refs if r not in lock.identity_refs]
    if stray:
        raise CompileRefused(
            f"identity ref(s) {stray} are not in the production lock (has: "
            f"{list(lock.identity_refs)}); a segment cannot condition on an "
            f"identity the lock never authorized")

    context = build_locked_context(
        lock, snapshot=snapshot, audio_master=audio_master,
        continuity=continuity, shot_plan=shot_plan, tone=tone,
        identity_refs=refs, dialogue=dialogue, scene_refs=scene_refs,
        joint_modes=joint_modes, spatial_refs=spatial_refs, beats=beats)

    writer: PromptWriter = prompt_writer or default_prompt_writer

    # PHASE 1 — write every prompt from the locked context. No SegmentSpec
    # exists yet, anywhere: this is invariant 9 made structural rather than
    # merely intended. The same `context` object goes into every call.
    prompts: list[str] = []
    for index in range(len(context.segments)):
        written = writer(context, index)
        if not isinstance(written, str) or not written.strip():
            raise CompileRefused(
                f"prompt_writer returned {written!r} for index {index}; a "
                f"segment prompt must be a non-empty string (an invented "
                f"placeholder here would be a fabricated shot)")
        prompts.append(written)

    # PHASE 2 — assemble. Nothing below calls the writer again.
    parents = lock.parent_digests
    specs: list[SegmentSpec] = []
    for index, entry in enumerate(shot_plan.entries):
        brief = context.segments[index]
        specs.append(SegmentSpec(
            segment_id=entry.segment_id, index=index,
            scene_ref=brief.scene_ref,
            continuity=continuity.state(entry.segment_id),
            audio_window=(entry.start_s, entry.end_s, entry.line_ids),
            shot=entry, spatial_ref=brief.spatial_ref, tone=float(tone),
            rubric=entry.rubric, prompt=prompts[index],
            negative_prompt=negative_prompt, identity_refs=refs,
            joint_mode=brief.joint_mode,
            seed_base=segment_seed(lock.digest, entry.segment_id, seed_salt),
            lock_digest=lock.digest, parents=parents))

    out = tuple(specs)
    assert_siblings(out, lock=lock)
    return out


# ---------------------------------------------------------------------------
# PlanGraph emission + execution order
# ---------------------------------------------------------------------------


def to_plan_graph(specs: Sequence[SegmentSpec],
                  lock: ProductionLock | None = None, *,
                  goal: GoalSpec | None = None,
                  graph_id: str = "production",
                  capability: str = SEGMENT_CAPABILITY,
                  candidates: int = 1,
                  revision: int = 0) -> PlanGraph:
    """Emit the Stage 14 shape as a k103 ``PlanGraph``.

    One structural ``gate`` node (``production_lock``) fans out to one ``task``
    node per segment, each tagged ``params["segment"] = True`` so
    ``PlanGraph.segment_node_ids()`` and the validator's ``SIBLING_VIOLATION``
    check find them without being told. There is exactly one edge per segment
    and it comes from the lock: no segment node is ever an ancestor of another,
    which is the required relationship drawn as a graph.

    ``lock`` is optional because every spec already carries ``lock_digest`` and
    ``parents``; supply it when you have it (the node then records the
    individual artifact digests too) and omit it when reconstructing a graph
    from specs alone. ``goal`` pins the plan to a ``GoalSpec``; with none, the
    graph pins to the LOCK digest instead — still immutable, still a real
    provenance tie, and k106 passes the true goal.

    The graph is sibling-checked before it is returned. A function that emits an
    invariant violation and leaves the checking to its caller is how the
    violation ships."""
    specs = tuple(specs)
    if not specs:
        raise CompileRefused("to_plan_graph needs at least one SegmentSpec")
    assert_siblings(specs, lock=lock)
    lock_digest = specs[0].lock_digest
    parents = specs[0].parents

    pinned = goal_digest(goal) if goal is not None else lock_digest
    lock_params: dict[str, Any] = {"lock_digest": lock_digest,
                                   "parents": list(parents)}
    if lock is not None:
        lock_params.update({
            "snapshot_digest": lock.snapshot_digest,
            "screenplay_digest": lock.screenplay_digest,
            "continuity_digest": lock.continuity_digest,
            "audio_master_digest": lock.audio_master_digest,
            "shot_plan_digest": lock.shot_plan_digest,
            "registry_version": lock.registry_version,
            "revision": lock.revision,
        })
    nodes: list[PlanNode] = [PlanNode(
        node_id=LOCK_NODE_ID, kind=NodeKind.GATE, approval_gate=True,
        outputs=(Port("locked", ArtifactKind.JSON),),
        params=FrozenParams(lock_params))]
    edges: list[Edge] = []
    for spec in sorted(specs, key=lambda s: s.index):
        node_id = segment_node_id(spec.segment_id)
        nodes.append(PlanNode(
            node_id=node_id, kind=NodeKind.TASK, capability=capability,
            inputs=(Port("lock", ArtifactKind.JSON),),
            outputs=(Port("clip", ArtifactKind.VIDEO),),
            candidates=candidates,
            # Stage 9's per-shot rubric, carried into the plan so the repair
            # controller (k107) reads the code->node mapping off the graph
            # instead of rediscovering it from a stack trace.
            acceptance=tuple(AcceptanceTest(kind=CheckKind.INTENT,
                                            threshold=criterion,
                                            repair_code=RepairCode.INTENT_MISMATCH)
                             for criterion in spec.rubric),
            params=FrozenParams({
                SEGMENT_PARAM: True,
                "segment_id": spec.segment_id,
                "index": spec.index,
                "spec_digest": spec.digest,
                "lock_digest": spec.lock_digest,
                "parents": list(spec.parents),
                "prompt": spec.prompt,
                "negative_prompt": spec.negative_prompt,
                "identity_refs": list(spec.identity_refs),
                "joint_mode": spec.joint_mode,
                "seed_base": spec.seed_base,
                "start_s": spec.start_s,
                "end_s": spec.end_s,
                "line_ids": list(spec.line_ids),
                "tone": spec.tone,
            })))
        edges.append(Edge(LOCK_NODE_ID, "locked", node_id, "lock"))

    graph = PlanGraph(graph_id=graph_id, goal_digest=pinned, revision=revision,
                      nodes=tuple(nodes), edges=tuple(edges))
    segment_ids = graph.segment_node_ids()
    if not sibling_check(graph, segment_ids):
        raise SiblingViolation(
            f"the emitted plan graph violates Stage 14: "
            f"{list(sibling_violations(graph, segment_ids))}",
            pairs=sibling_violations(graph, segment_ids))
    return graph


def execution_order(specs: Sequence[SegmentSpec],
                    mode: str = "parallel", *,
                    lock: ProductionLock | None = None,
                    graph: PlanGraph | None = None
                    ) -> tuple[tuple[str, ...], ...]:
    """Batches of node ids to run, derived from the ``PlanGraph``.

    Doc Stage 14's closing line: *"Segment execution may be sequential or
    parallel without changing this dependency structure."* Both modes therefore
    read the SAME graph and differ only in how they group its topological
    order — ``sequential`` emits one node per batch, ``parallel`` emits one
    batch per dependency level. Neither adds, removes or reorders an edge, so a
    fleet with one GPU and a fleet with eight execute the same plan."""
    if mode not in ("sequential", "parallel"):
        raise ValueError(f"execution_order mode must be 'sequential' or "
                         f"'parallel', got {mode!r}")
    plan = graph if graph is not None else to_plan_graph(specs, lock)
    order = plan.topological_order()
    if mode == "sequential":
        return tuple((node_id,) for node_id in order)
    predecessors = plan.predecessors()
    depth: dict[str, int] = {}
    for node_id in order:
        parents = predecessors.get(node_id) or set()
        depth[node_id] = 0 if not parents else 1 + max(depth[p] for p in parents)
    batches: list[list[str]] = []
    for node_id in order:                      # topological order = stable
        level = depth[node_id]
        while len(batches) <= level:
            batches.append([])
        batches[level].append(node_id)
    return tuple(tuple(batch) for batch in batches)


__all__ = [
    "JOINT_MODES",
    "JOINT_MODE_PLAIN",
    "LOCK_NODE_ID",
    "SEGMENT_CAPABILITY",
    "SEGMENT_NODE_PREFIX",
    "CompileRefused",
    "LockedContext",
    "LockedSegmentBrief",
    "PromptWriter",
    "SegmentSpec",
    "SiblingViolation",
    "assert_siblings",
    "build_locked_context",
    "compile_segments",
    "default_prompt_writer",
    "execution_order",
    "render_dependencies",
    "resolve_spatial_ref",
    "segment_node_id",
    "segment_seed",
    "shot_plan_from_windows",
    "shot_windows_from_audio",
    "sibling_violations_in",
    "spatial_source_for",
    "to_plan_graph",
]
