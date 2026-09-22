"""COORDINATION REVIEW (k121) — the knobs must match the words.

THE INCIDENT THIS DELETES (operator, cinema session 2026-08-20). A multi-segment
cinema prompt set was generated/enhanced through the studio spread. Every prompt
carried explicit in-context expectations: scenes declared CONTINUOUS, a scene
that "picks up where the last one left off", a character recurring across four
shots — *"though not a single prompt was created with the proper knobs turned"*:

  * ``joint_mode`` was never set (every segment stayed ``cut``);
  * no start-image / last-frame carry from the previous segment's render;
  * *"no lengthening of the # of frames"* — every segment took the model default;
  * *"no character identity building at any point in between… from the previous
    generation"* — the recurring character was a NAME in prose and nothing else.

k93's :mod:`prompt_spread` already SPEAKS the joint modes: it renders them to the
writer as sentences ("this shot continues the previous shot, carrying its motion
across the join") so the prose comes back coherent. That is exactly half the job.
The prose came back coherent and the RENDER did not, because nothing on the way
back turned the knob the sentence described. **The spread says the joint modes;
this module SETS them.**

Operator directive: *"every aspect of the above needs to be explicitly reviewed
upon the creation of any generated or enhanced prompting in video."* Hence
:func:`review`, which is not optional and is never silent: EVERY generated or
enhanced prompt set gets a :class:`CoordinationReport` with a row per segment —
including the segments that were already right (``ok``).

────────────────────────────────────────────────────────────────────────────────
INVARIANT 9 IS NOT TOUCHED HERE, AND THAT IS THE WHOLE ARCHITECTURE
────────────────────────────────────────────────────────────────────────────────
The sibling rule (``oracle/segments.py``, doc Stage 14) stands: *no segment
prompt generated during a run may become the source of another segment prompt in
that run.* This module reads prompt TEXT and writes only MECHANICS. It has no
code path that can put one row's prose into another row — :func:`apply_decisions`
writes ONLY the knob keys in :data:`APPLIABLE_KNOBS`, and ``prompt`` is not one
of them (there is a test).

The sanctioned mechanical channels, all of which already exist:

  ``joint_mode``        ``cut`` / ``still`` / ``vace_extend`` + ``parent_segment_id``
                        (``studio_movie_schema``; splice logic in
                        ``runners/studio_movie.py``)
  last-frame carry      ``branch_frame=None`` on a carrying join == the parent's
                        LAST frame (``runners/movie.py``'s ``prev_last_frame`` /
                        ``drift_mode`` is the frame-timeline equivalent)
  identity              ``reference_images`` + capability ``id_lock``
                        (``identity_profiles``), consent-gated per k97
  length                per-segment ``frames`` -> ``requested_frames`` -> the
                        spine's ``resolve_frames`` clamp + 4k+1 snap

Vocabulary is deliberately k104's (``oracle/segments.py``: ``joint_mode``,
``seed_base``, ``identity_refs``, ``segment_id``, ``index``) so the script-first
compiler and this prompt-first reviewer name the same knobs.

────────────────────────────────────────────────────────────────────────────────
THE RATCHET RULE (why ``set`` is safe and ``proposed`` exists)
────────────────────────────────────────────────────────────────────────────────
A review that silently rewrote knobs in both directions would be a new way to
lose work. So:

    COORDINATION IS ADDED AUTOMATICALLY. IT IS NEVER REMOVED AUTOMATICALLY.

``set`` is only ever emitted for a change that INCREASES coordination and
destroys nothing: ``cut`` -> ``still``/``vace_extend``, linking a parent on the
linear chain, sharing a seed across a concurrency pair, LENGTHENING frames,
attaching an identity profile that is already in the locked context and already
authorized. Every change that REMOVES carry (-> ``cut``, dropping a branch
frame, shortening a clip, detaching an identity) is ``proposed`` and waits for a
human. Anything the knobs cannot express at all — a continuation on segment 0, a
parent pointer that breaks the linear-chain rule, a duration past the
checkpoint's ceiling, a recurrence whose profile exists but carries no k97
consent, a LOCKED row whose words fight its knobs — is a ``mismatch``, reported
WITH THE EVIDENCE QUOTE and never auto-resolved.

────────────────────────────────────────────────────────────────────────────────
DETERMINISTIC FIRST; THE LLM IS NEVER LOAD-BEARING
────────────────────────────────────────────────────────────────────────────────
:func:`extract_expectations` is a pure phrase/regex/fuzzy-name pass — no network,
no model, no clock. An optional ``llm`` seam is consulted ONLY for rows the rules
found nothing in, its output is validated (known kind, real segment ids, and an
``evidence_quote`` that must be a LITERAL substring of that row's text — a
fabricated quote is dropped, not shown), it is marked ``source="llm"``, its
confidence is capped below :data:`BLOCK_CONFIDENCE`, and it can only ever produce
``proposed`` decisions. Delete the seam and every deterministic finding is
byte-identical.

No pathlib anywhere. os.path only (there is none here — pure data).
"""
from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CoordinationError",
    "Expectation",
    "KnobDecision",
    "SegmentReview",
    "CoordinationReport",
    "KIND_CONTINUATION",
    "KIND_CONCURRENCY",
    "KIND_DISCONTINUITY",
    "KIND_RECURRING_CHARACTER",
    "KIND_DURATION",
    "EXPECTATION_KINDS",
    "STATUS_SET",
    "STATUS_PROPOSED",
    "STATUS_MISMATCH",
    "STATUS_OK",
    "STATUS_ORDER",
    "KNOB_JOINT_MODE",
    "KNOB_PARENT",
    "KNOB_BRANCH_FRAME",
    "KNOB_SEED",
    "KNOB_FRAMES",
    "KNOB_REFERENCE_IMAGES",
    "KNOB_IDENTITY",
    "APPLIABLE_KNOBS",
    "STEP_IDENTITY_CAPTURE",
    "BLOCK_CONFIDENCE",
    "LLM_MAX_CONFIDENCE",
    "REPORT_VERSION",
    "normalize_rows",
    "extract_expectations",
    "derive_knobs",
    "review",
    "review_goals",
    "apply_decisions",
    "blocking_mismatches",
    "bind_coordination_llm",
    "build_llm_messages",
    "parse_llm_expectations",
]


class CoordinationError(ValueError):
    """A structurally invalid review request — the caller's to fix (maps to 400)."""


# --------------------------------------------------------------------------- #
# VOCABULARY
# --------------------------------------------------------------------------- #
#: What the prose CLAIMED. One kind per claim class the operator named.
KIND_CONTINUATION = "continuation"          # "continuous", "picks up where"
KIND_CONCURRENCY = "concurrency"            # "meanwhile", "simultaneously"
KIND_DISCONTINUITY = "discontinuity"        # "hard cut to", "three days later"
KIND_RECURRING_CHARACTER = "recurring_character"
KIND_DURATION = "duration"                  # "long take", dialogue length, "12 seconds"
EXPECTATION_KINDS: Tuple[str, ...] = (
    KIND_CONTINUATION, KIND_CONCURRENCY, KIND_DISCONTINUITY,
    KIND_RECURRING_CHARACTER, KIND_DURATION,
)

#: What the review DID about it.
STATUS_SET = "set"              # the knob was turned (ratchet-safe, applied)
STATUS_PROPOSED = "proposed"    # needs operator confirm (destructive or ambiguous)
STATUS_MISMATCH = "mismatch"    # words demand X, knobs say Y, cannot auto-set
STATUS_OK = "ok"                # the knob already matched the words
#: Worst-first, for folding a segment's decisions into ONE badge.
STATUS_ORDER: Tuple[str, ...] = (STATUS_MISMATCH, STATUS_PROPOSED, STATUS_SET, STATUS_OK)

#: The knob names — ``StudioMovieGoal`` field names EXACTLY (so a decision can be
#: applied to a row without a translation layer to get wrong).
KNOB_JOINT_MODE = "joint_mode"
KNOB_PARENT = "parent_segment_id"
KNOB_BRANCH_FRAME = "branch_frame"
KNOB_SEED = "seed"
KNOB_FRAMES = "frames"
KNOB_REFERENCE_IMAGES = "reference_images"
KNOB_IDENTITY = "identity"      # an id_lock PLAN (profile slug + refs), not a goal field

#: The ONLY keys :func:`apply_decisions` may write. ``prompt`` is deliberately
#: absent and must stay absent: that absence IS invariant 9 in this module.
APPLIABLE_KNOBS: Tuple[str, ...] = (
    KNOB_JOINT_MODE, KNOB_PARENT, KNOB_BRANCH_FRAME, KNOB_SEED, KNOB_FRAMES,
    KNOB_REFERENCE_IMAGES,
)

#: A proposed WORK STEP rather than a knob: build an identity profile from a
#: prior segment's accepted frames, through the existing identity pipeline.
STEP_IDENTITY_CAPTURE = "IDENTITY_CAPTURE"

#: A ``mismatch`` at or above this confidence BLOCKS a movie submit (overridable).
#: Below it, a mismatch is loud but advisory — the duration-past-the-ceiling case,
#: where there is nothing the operator can do at this layer.
BLOCK_CONFIDENCE = 0.8
#: Hard cap on any LLM-sourced confidence. Deliberately below BLOCK_CONFIDENCE:
#: a model's guess can never block a render.
LLM_MAX_CONFIDENCE = 0.6

#: Bumped when the wire shape of :meth:`CoordinationReport.as_dict` changes.
REPORT_VERSION = "k121.1"

# Mirrors ``studio_movie_schema._VALID_JOINT_MODES`` / ``prompt_spread``.
_VALID_JOINT_MODES: Tuple[str, ...] = ("still", "vace_extend", "cut")
_CARRYING_MODES: Tuple[str, ...] = ("still", "vace_extend")

# Clip-length facts. IMPORTED at use (``_frame_limits``) from ``studio.schemas``,
# which is where ``runners/synthetic.resolve_frames`` reads them — but that import
# builds the studio MODEL REGISTRY (a ~50s cold import), so this module must not
# pay it to be importable from a route or a test. These are the FALLBACK literals
# used when the studio package cannot be imported; ``tests/test_prompt_coordination``
# asserts they still equal the studio's own, so a drift fails a test rather than
# silently quoting a caller the wrong clip length.
_FALLBACK_WAN_MAX_FRAMES = 81
_FALLBACK_FRAME_CADENCE = 4
_FALLBACK_DEFAULT_FRAMES = 81
_DEFAULT_FPS = 16

#: Words-per-second used to turn quoted dialogue into a duration. Deliberately
#: conservative (an unhurried delivery), plus a second of air for the shot to
#: breathe: under-asking for length is recoverable, over-asking burns GPU.
_DIALOGUE_WPS = 2.5
_DIALOGUE_AIR_S = 1.0
#: What "a long take" means when the prose gives no number.
_LONG_TAKE_S = 6.0


# --------------------------------------------------------------------------- #
# PHRASE TABLES — whole-token matched (the ``shot_intent`` idiom: "rear" must
# never fire on "rearrange"). Ordered most-specific-first inside each bucket.
# --------------------------------------------------------------------------- #
#: Continuation cues that carry MOTION across the join -> ``vace_extend``.
_MOTION_CONTINUATION_CUES: Tuple[str, ...] = (
    "continuous", "continuously", "one continuous", "unbroken",
    "without cutting", "without a cut", "no cut", "same shot continues",
    "the shot continues", "continues the", "continuing the", "continues from",
    "continuing from", "carries on", "carrying on", "still running",
    "still walking", "still moving", "mid-motion", "mid motion", "mid-stride",
    "mid stride", "mid-sprint", "mid sprint", "keeps running", "keeps walking",
    "keeps moving", "same motion", "uninterrupted",
)
#: Continuation cues that resume from a FRAME but not from motion -> ``still``.
_FRAME_CONTINUATION_CUES: Tuple[str, ...] = (
    "picks up where", "picking up where", "picks up from", "resumes",
    "resuming", "a beat later", "moments later", "seconds later",
    "immediately after", "directly after", "same moment as the previous",
    "from the same position", "holding the same frame",
)
#: Screenplay slug-line continuity: "SCENE 4 - CONTINUOUS", "INT. BAR - CONTINUOUS".
_CONTINUOUS_HEADING_RE = re.compile(
    r"^\s*(?:scene\s*\d+|int\.|ext\.|int/ext\.)[^\n]{0,120}?[-–—(,]\s*continuous\b",
    re.IGNORECASE | re.MULTILINE)

#: Same time, DIFFERENT place: a cut whose world/lighting/seed must still match.
_CONCURRENCY_CUES: Tuple[str, ...] = (
    "meanwhile", "simultaneously", "concurrent", "concurrently",
    "at the same time", "at the same moment", "same moment", "elsewhere",
    "across town", "in parallel", "while this happens",
    "while that happens", "at that very moment",
)

#: Explicit DISCONTINUITY: the prose has left the previous shot behind.
_DISCONTINUITY_CUES: Tuple[str, ...] = (
    "hard cut", "smash cut", "cut to", "cuts to", "we cut to", "jump cut",
    "match cut", "new scene", "a new location", "days later", "a day later",
    "the next day", "the next morning", "weeks later", "years later",
    "months later", "hours later", "much later", "a week later",
    "somewhere else entirely",
)

#: Duration cues with no number attached -> ``_LONG_TAKE_S``.
_LONG_TAKE_CUES: Tuple[str, ...] = (
    "long take", "long, slow", "slow pan", "slowly pans", "slow push",
    "slowly pushes", "slow dolly", "slowly dollies", "slow zoom",
    "slowly zooms", "lingering", "lingers", "extended take", "unbroken take",
    "holds on", "holds for", "drifts slowly", "slow tracking",
)
#: Explicit durations: "12 seconds", "a 10-second take", "half a minute".
_DURATION_RE = re.compile(
    r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*[-–]?\s*(seconds?|secs?|minutes?|mins?)(?![a-z])",
    re.IGNORECASE)
#: Dialogue: straight or curly quotes, non-greedy, bounded so a stray quote
#: cannot swallow a whole prompt.
_DIALOGUE_RE = re.compile(r"[\"“]([^\"”\n]{4,400})[\"”]")

#: Tokens that look like a name but never are. Sentence starters, camera
#: vocabulary, screenplay slugs, days/months, and the words a director writes in
#: title case. Extend freely — a false positive here costs an operator a badge.
_NAME_STOPWORDS = frozenset("""
a an and the this that these those there their they them then than but or for
nor yet so as at by in into on onto of off out over under up down from to with
without within across through between behind before after during while when
where why how what which who whom whose it its his her hers him he she we us our
you your i me my mine is are was were be been being do does did done has have had
can could should would may might must shall will not no yes if else once again
int ext continuous cut cuts scene shot take fade dissolve match smash jump close
closeup wide medium long angle pan tilt zoom dolly track crane handheld steadicam
camera lens frame frames light lighting shadow shadows color colour grade
morning afternoon evening night day days dawn dusk midnight noon later moment
moments monday tuesday wednesday thursday friday saturday sunday january february
march april may june july august september october november december
he's she's they're it's we're don't doesn't didn't won't can't
meanwhile elsewhere simultaneously concurrently suddenly finally eventually
afterwards afterward beforehand meanwhile's inside outside above below beyond
nearby somewhere anywhere everywhere everything nothing something anything
another each every both all one two three four five six seven eight nine ten
first second third fourth fifth last next previous still already almost nearly
slowly quickly quietly loudly softly sharply briefly finally instead rather
here now soon today tonight yesterday tomorrow always never often sometimes
continuous continuing continues picks resumes unbroken
""".split())

#: Descriptor-style recurring characters ("the woman in the red coat"). The
#: SUBJECT nouns a film prompt actually uses; the optional trailing "in the …"
#: clause is what makes two mentions the SAME person rather than any woman.
_DESCRIPTOR_RE = re.compile(
    r"\bthe\s+((?:young|old|elderly|tall|short|small|large|thin|heavy|grey|gray|"
    r"bearded|masked|hooded|uniformed|older|younger)\s+)?"
    r"(man|woman|girl|boy|child|kid|figure|stranger|detective|driver|soldier|"
    r"pilot|doctor|nurse|guard|dancer|singer|writer|painter|sailor|hunter|"
    r"climber|runner|swimmer|barkeep|bartender|waitress|waiter|courier|"
    r"mechanic|engineer|scientist|teacher|student|priest|nun|monk|thief)"
    r"(\s+in\s+(?:the|a|an)\s+[a-z]+(?:\s+[a-z]+)?)?\b",
    re.IGNORECASE)

#: Capitalized proper-noun runs — the ordinary way a prompt names a character.
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")

#: Fuzzy-name agreement floor. 0.86 catches "Marek"/"Marec" and a dropped letter
#: without merging "Mara" into "Marcus" (0.67).
_NAME_RATIO = 0.86


# --------------------------------------------------------------------------- #
# TYPES
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Expectation:
    """ONE claim the prose made, and the words that made it.

        kind            an :data:`EXPECTATION_KINDS` member.
        segments        the segment ids the claim binds, in TIMELINE order. For a
                        continuation/discontinuity that is one id (the claiming
                        segment); for a concurrency or a recurring character it is
                        every id involved, EARLIEST FIRST.
        evidence_quote  the literal words, verbatim from the row's own text. The
                        operator must be able to read WHY a knob moved; a
                        paraphrase would make the badge unfalsifiable.
        confidence      0..1. >= :data:`BLOCK_CONFIDENCE` may block a submit.
        source          "rule" (deterministic) or "llm" (advisory, capped).
        detail          kind-specific extras: ``{"character": …}``,
                        ``{"seconds": …}``, ``{"mode": "vace_extend"}``.
    """
    kind: str
    segments: Tuple[str, ...]
    evidence_quote: str
    confidence: float
    source: str = "rule"
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "segments": list(self.segments),
                "evidence_quote": self.evidence_quote,
                "confidence": round(float(self.confidence), 3),
                "source": self.source, "detail": dict(self.detail)}


@dataclass(frozen=True)
class KnobDecision:
    """What the review DID (or refused to do) about one expectation, on one knob.

    ``current`` / ``proposed_value`` are the knob DIFF the badge renders. ``step``
    is set only for a work step that is not a knob at all
    (:data:`STEP_IDENTITY_CAPTURE`)."""
    segment_id: str
    index: int
    knob: str
    current: Any
    proposed_value: Any
    status: str
    reason: str
    confidence: float = 0.5
    expectation: Optional[Expectation] = None
    step: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        """A mismatch confident enough to stop a submit. The ONE threshold."""
        return self.status == STATUS_MISMATCH and self.confidence >= BLOCK_CONFIDENCE

    @property
    def evidence_quote(self) -> str:
        return self.expectation.evidence_quote if self.expectation else ""

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "segment_id": self.segment_id, "index": self.index, "knob": self.knob,
            "current": self.current, "proposed": self.proposed_value,
            "status": self.status, "reason": self.reason,
            "confidence": round(float(self.confidence), 3),
            "blocking": self.blocking,
            "evidence_quote": self.evidence_quote,
        }
        if self.step:
            out["step"] = self.step
        if self.expectation is not None:
            out["expectation"] = self.expectation.as_dict()
        if self.detail:
            out["detail"] = dict(self.detail)
        return out


@dataclass(frozen=True)
class SegmentReview:
    """Every segment gets one of these — including the ones that were already
    right. NOTHING SILENT is the operator's directive, so an ``ok`` row is a
    POSITIVE statement ("reviewed, knobs match"), not an absence."""
    segment_id: str
    index: int
    status: str
    decisions: Tuple[KnobDecision, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"segment_id": self.segment_id, "index": self.index,
                "status": self.status,
                "decisions": [d.as_dict() for d in self.decisions]}


@dataclass(frozen=True)
class CoordinationReport:
    """The whole review. ``as_dict`` is the wire shape the route and the UI read."""
    segments: Tuple[SegmentReview, ...]
    expectations: Tuple[Expectation, ...]
    decisions: Tuple[KnobDecision, ...]
    version: str = REPORT_VERSION
    llm_used: bool = False
    notes: Tuple[str, ...] = ()

    @property
    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in STATUS_ORDER}
        for d in self.decisions:
            out[d.status] = out.get(d.status, 0) + 1
        out["segments"] = len(self.segments)
        return out

    @property
    def status(self) -> str:
        """The worst segment status — the one badge for the whole set."""
        for s in STATUS_ORDER:
            if any(seg.status == s for seg in self.segments):
                return s
        return STATUS_OK

    def applied(self) -> Tuple[KnobDecision, ...]:
        return tuple(d for d in self.decisions if d.status == STATUS_SET)

    def blocking(self) -> Tuple[KnobDecision, ...]:
        return tuple(d for d in self.decisions if d.blocking)

    def for_segment(self, segment_id: str) -> Optional[SegmentReview]:
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "counts": self.counts,
            "llm_used": self.llm_used,
            "notes": list(self.notes),
            "segments": [s.as_dict() for s in self.segments],
            "expectations": [e.as_dict() for e in self.expectations],
            "decisions": [d.as_dict() for d in self.decisions],
        }


# --------------------------------------------------------------------------- #
# ROW NORMALIZATION
# --------------------------------------------------------------------------- #
_ROW_TEXT_KEYS = ("prompt", "direction", "continuity_note", "scene", "text")


def _clean(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _int_or_none(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool) or not isinstance(v, int):
        return None
    return v


def normalize_rows(rows: Any) -> Tuple[Dict[str, Any], ...]:
    """Validate + order the prompt rows a review runs over.

    Accepts the shape every caller already holds: a ``prompt_spread`` segment ref,
    a parsed spread result row, a ``StudioMovieGoal`` asdict, or the UI's prompt
    card. ``index`` orders the timeline when present; list order backs it up
    (rows arrive out of order the moment a UI mixes selected + unselected).

    The ``locked`` flag is load-bearing: a locked row is the operator's own work,
    so it is REVIEWED but never ``set``."""
    if rows is None:
        return ()
    if isinstance(rows, Mapping):
        raise CoordinationError("rows must be a list of segment objects")
    try:
        seq = list(rows)
    except TypeError as exc:  # pragma: no cover - defensive
        raise CoordinationError("rows must be a list of segment objects") from exc

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for i, raw in enumerate(seq):
        if not isinstance(raw, Mapping):
            raise CoordinationError(f"rows[{i}] must be an object")
        sid = _clean(raw.get("segment_id")) or _clean(raw.get("id"))
        if not sid:
            raise CoordinationError(f"rows[{i}].segment_id is required")
        if sid in seen:
            raise CoordinationError(f"duplicate segment_id {sid!r}")
        seen.add(sid)
        jm = raw.get("joint_mode")
        if jm is not None and jm not in _VALID_JOINT_MODES:
            raise CoordinationError(
                f"rows[{i}].joint_mode must be one of " + "|".join(_VALID_JOINT_MODES))
        refs = raw.get("reference_images")
        if refs is None:
            refs = raw.get("identity_refs")
        row = {
            "segment_id": sid,
            "index": _int_or_none(raw.get("index")),
            "order": i,
            "prompt": _clean(raw.get("prompt")),
            "direction": _clean(raw.get("direction")),
            "continuity_note": _clean(raw.get("continuity_note")),
            "negative": _clean(raw.get("negative")),
            "joint_mode": jm,
            "parent_segment_id": _clean(raw.get("parent_segment_id")) or None,
            "branch_frame": _int_or_none(raw.get("branch_frame")),
            "frames": _int_or_none(raw.get("frames")),
            "seed": _int_or_none(raw.get("seed")),
            "model_id": _clean(raw.get("model_id")) or None,
            "reference_images": tuple(str(r) for r in refs) if isinstance(refs, (list, tuple)) else (),
            "locked": bool(raw.get("locked")),
        }
        out.append(row)

    out.sort(key=lambda r: (r["index"] if r["index"] is not None else r["order"],
                            r["order"]))
    for position, row in enumerate(out):
        row["index"] = position
    return tuple(out)


def _row_text(row: Mapping[str, Any]) -> str:
    """Everything the operator/model WROTE for this row, joined for scanning.

    The direction and the continuity note count: an expectation stated as a
    direction ("keep this continuous with shot 2") is exactly as binding as one
    stated in the prose, and the operator's failing session had both."""
    return "\n".join(_clean(row.get(k)) for k in _ROW_TEXT_KEYS if _clean(row.get(k)))


# --------------------------------------------------------------------------- #
# PHRASE MATCHING
# --------------------------------------------------------------------------- #
def _cue_span(text_lower: str, phrase: str) -> Optional[Tuple[int, int]]:
    """Whole-token span of ``phrase`` in ``text_lower``, or None.

    Alphanumerics are the token chars (``shot_intent._has_cue``'s rule), so
    "cut to" matches "we cut to black" but "no cut" never fires inside
    "no cutlery"."""
    m = re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text_lower)
    return (m.start(), m.end()) if m else None


def _quote_around(text: str, start: int, end: int, width: int = 90) -> str:
    """The evidence quote: the sentence around a hit, bounded and squeezed.

    Verbatim from the row's own text — never rewritten. A badge that quotes the
    operator's own words is falsifiable; a paraphrase is not."""
    left = max(0, start - width)
    right = min(len(text), end + width)
    # Prefer sentence boundaries when they are inside the window.
    dot = text.rfind(". ", left, start)
    if dot != -1:
        left = dot + 2
    nl = text.rfind("\n", left, start)
    if nl != -1:
        left = nl + 1
    stop = text.find(". ", end, right)
    if stop != -1:
        right = stop + 1
    else:
        nl2 = text.find("\n", end, right)
        if nl2 != -1:
            right = nl2
    quote = re.sub(r"\s+", " ", text[left:right]).strip()
    prefix = "…" if left > 0 and not text[:left].strip().endswith((".", "\n")) else ""
    suffix = "…" if right < len(text) and not text[right:].lstrip().startswith(".") else ""
    return (prefix + quote + suffix).strip()


def _first_cue(text: str, cues: Sequence[str]) -> Optional[Tuple[str, str]]:
    """(matched phrase, evidence quote) for the FIRST cue that fires, else None.

    Ordered scan: the tables are written most-specific-first so a prose that
    mixes cues resolves deterministically (the ``shot_intent`` rule)."""
    low = text.lower()
    for phrase in cues:
        span = _cue_span(low, phrase)
        if span:
            return phrase, _quote_around(text, span[0], span[1])
    return None


# --------------------------------------------------------------------------- #
# CHARACTER MENTIONS
# --------------------------------------------------------------------------- #
def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _names_agree(a: str, b: str) -> bool:
    """Fuzzy character-name agreement.

    Three ways two mentions are the same person, in order of how sure we are:
      1. identical once folded ("Mara," == "mara");
      2. one is a leading token of the other ("Mara" vs "Mara Vex" — a film
         introduces a full name once and uses the first name after);
      3. a close edit distance (>= :data:`_NAME_RATIO`), which catches the
         generator's own typos without merging two different characters.
    """
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    first_a = _norm_name(a.split()[0]) if a.split() else ""
    first_b = _norm_name(b.split()[0]) if b.split() else ""
    if first_a and first_b and (first_a == nb or first_b == na):
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= _NAME_RATIO


def _mentions_in(text: str) -> List[Tuple[str, int, int]]:
    """``[(mention, start, end)]`` — proper-noun names and subject descriptors.

    Two extractors because a film prompt names people two ways: "Mara" and "the
    woman in the red coat". Both are CHARACTERS for continuity purposes, and the
    operator's failing set used the second form for its new character."""
    found: List[Tuple[str, int, int]] = []
    for m in _PROPER_NOUN_RE.finditer(text):
        raw = m.group(1)
        parts = [p for p in raw.split() if _norm_name(p) not in _NAME_STOPWORDS]
        if not parts:
            continue
        if len(parts) != len(raw.split()):
            # A stopword rode along ("The Mara") — keep only the real tokens and
            # re-locate them so the evidence quote still points at the name.
            sub = " ".join(parts)
            at = text.find(sub, m.start())
            if at == -1:
                continue
            found.append((sub, at, at + len(sub)))
            continue
        found.append((raw, m.start(), m.end()))
    for m in _DESCRIPTOR_RE.finditer(text):
        found.append((re.sub(r"\s+", " ", m.group(0)).strip().lower(),
                      m.start(), m.end()))
    return found


def _character_clusters(rows: Sequence[Mapping[str, Any]]
                        ) -> List[Dict[str, Any]]:
    """Cluster character mentions across rows.

    A cluster is ``{"label", "aliases", "hits": [(segment_id, index, quote)]}``.
    Clustering is order-stable (rows are already in timeline order and mentions
    are scanned left-to-right), so the same prompt set always produces the same
    clusters — the property the whole deterministic core rests on."""
    clusters: List[Dict[str, Any]] = []
    for row in rows:
        text = _row_text(row)
        if not text:
            continue
        seen_here: set = set()
        for mention, start, end in _mentions_in(text):
            key = _norm_name(mention)
            if not key or key in seen_here:
                continue
            seen_here.add(key)
            quote = _quote_around(text, start, end)
            for cl in clusters:
                if any(_names_agree(mention, a) for a in cl["aliases"]):
                    if mention not in cl["aliases"]:
                        cl["aliases"].append(mention)
                    cl["hits"].append((row["segment_id"], row["index"], quote))
                    break
            else:
                clusters.append({"label": mention, "aliases": [mention],
                                 "hits": [(row["segment_id"], row["index"], quote)]})
    return clusters


# --------------------------------------------------------------------------- #
# DURATION
# --------------------------------------------------------------------------- #
def _explicit_seconds(text: str) -> Optional[Tuple[float, str]]:
    """(seconds, evidence quote) from an explicit "N seconds"/"N minutes"."""
    m = _DURATION_RE.search(text)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    seconds = value * 60.0 if unit.startswith(("minute", "min")) else value
    if seconds <= 0:
        return None
    return seconds, _quote_around(text, m.start(), m.end())


def _dialogue_seconds(text: str) -> Optional[Tuple[float, str]]:
    """(seconds, evidence quote) implied by the LONGEST quoted line.

    Audio-first in miniature (doc Stage 8): a line takes as long as it takes, so
    the clip is sized to the line rather than the line crammed into the clip."""
    best: Optional[Tuple[float, str]] = None
    for m in _DIALOGUE_RE.finditer(text):
        words = len(m.group(1).split())
        if words < 4:
            continue
        seconds = words / _DIALOGUE_WPS + _DIALOGUE_AIR_S
        if best is None or seconds > best[0]:
            best = (seconds, _quote_around(text, m.start(), m.end()))
    return best


# --------------------------------------------------------------------------- #
# EXTRACTION
# --------------------------------------------------------------------------- #
def extract_expectations(prompt_rows: Any, *, notes: str = "",
                         llm: Optional[Callable[[str], str]] = None,
                         ) -> Tuple[Expectation, ...]:
    """Every claim the PROSE made, as typed :class:`Expectation` values.

    Pure and deterministic: same rows in, same tuple out, no clock, no network,
    no model. ``notes`` is the operator's free-form context ("scenes 2 and 3 are
    concurrent") and is scanned with the SAME tables against the row ids it
    names, because a claim stated in the notes box is exactly as binding as one
    stated in a prompt.

    ``llm`` (optional, ``(prompt) -> str``) is consulted ONLY for rows the rules
    found nothing in. Its findings are validated, capped at
    :data:`LLM_MAX_CONFIDENCE`, marked ``source="llm"``, and can never produce a
    ``set`` or a blocking mismatch — see :func:`parse_llm_expectations`.
    """
    rows = normalize_rows(prompt_rows)
    found: List[Expectation] = []
    by_id = {r["segment_id"]: r for r in rows}

    for row in rows:
        text = _row_text(row)
        if not text:
            continue
        sid = row["segment_id"]
        first = row["index"] == 0

        # ── CONTINUATION ────────────────────────────────────────────────────
        hit = _first_cue(text, _MOTION_CONTINUATION_CUES)
        mode_wanted = "vace_extend"
        if hit is None:
            hit = _first_cue(text, _FRAME_CONTINUATION_CUES)
            mode_wanted = "still"
        heading = None if hit else _CONTINUOUS_HEADING_RE.search(text)
        if heading is not None:
            hit = ("continuous (slug line)",
                   _quote_around(text, heading.start(), heading.end()))
            mode_wanted = "vace_extend"
        if hit is not None:
            phrase, quote = hit
            found.append(Expectation(
                kind=KIND_CONTINUATION, segments=(sid,), evidence_quote=quote,
                # A continuation claimed by the FIRST row is a real finding but a
                # confused one (there is nothing before it) — it lands as a
                # mismatch, so it is confident enough to block.
                confidence=0.9,
                detail={"cue": phrase, "mode": mode_wanted,
                        "first_segment": first}))

        # ── CONCURRENCY ─────────────────────────────────────────────────────
        hit = _first_cue(text, _CONCURRENCY_CUES)
        if hit is not None:
            phrase, quote = hit
            previous = rows[row["index"] - 1]["segment_id"] if row["index"] > 0 else None
            segs = (previous, sid) if previous else (sid,)
            found.append(Expectation(
                kind=KIND_CONCURRENCY, segments=tuple(s for s in segs if s),
                evidence_quote=quote, confidence=0.85,
                detail={"cue": phrase}))

        # ── DISCONTINUITY ───────────────────────────────────────────────────
        hit = _first_cue(text, _DISCONTINUITY_CUES)
        if hit is not None:
            phrase, quote = hit
            found.append(Expectation(
                kind=KIND_DISCONTINUITY, segments=(sid,), evidence_quote=quote,
                confidence=0.85, detail={"cue": phrase}))

        # ── DURATION ────────────────────────────────────────────────────────
        seconds_hit = _explicit_seconds(text)
        why = "explicit"
        if seconds_hit is None:
            seconds_hit = _dialogue_seconds(text)
            why = "dialogue"
        long_take = _first_cue(text, _LONG_TAKE_CUES)
        if seconds_hit is None and long_take is not None:
            seconds_hit = (_LONG_TAKE_S, long_take[1])
            why = "long_take"
        elif seconds_hit is not None and long_take is not None:
            # Both — take the longer ask, keep the numeric evidence.
            seconds_hit = (max(seconds_hit[0], _LONG_TAKE_S), seconds_hit[1])
        if seconds_hit is not None:
            seconds, quote = seconds_hit
            found.append(Expectation(
                kind=KIND_DURATION, segments=(sid,), evidence_quote=quote,
                confidence=0.9 if why == "explicit" else 0.7,
                detail={"seconds": round(float(seconds), 3), "basis": why}))

    # ── RECURRING CHARACTERS (cross-row) ────────────────────────────────────
    for cl in _character_clusters(rows):
        segs = []
        for sid, _idx, _q in cl["hits"]:
            if sid not in segs:
                segs.append(sid)
        if len(segs) < 2:
            continue
        # Evidence is the SECOND mention — the one that makes it recurring.
        second_quote = ""
        seen_first = False
        for sid, _idx, quote in cl["hits"]:
            if sid == segs[0]:
                seen_first = True
                continue
            if seen_first:
                second_quote = quote
                break
        found.append(Expectation(
            kind=KIND_RECURRING_CHARACTER, segments=tuple(segs),
            evidence_quote=second_quote or cl["hits"][-1][2],
            confidence=0.85,
            detail={"character": cl["label"], "aliases": list(cl["aliases"]),
                    "first_segment": segs[0], "mentions": len(cl["hits"])}))

    # ── OPERATOR NOTES ──────────────────────────────────────────────────────
    found.extend(_notes_expectations(notes, by_id, rows))

    # ── OPTIONAL LLM ASSIST — never load-bearing ────────────────────────────
    if llm is not None:
        covered = {s for e in found for s in e.segments}
        blind = [r for r in rows if _row_text(r) and r["segment_id"] not in covered]
        if blind:
            found.extend(_llm_expectations(llm, blind))

    return tuple(_dedupe_expectations(found))


def _notes_expectations(notes: str, by_id: Mapping[str, Any],
                        rows: Sequence[Mapping[str, Any]]) -> List[Expectation]:
    """Claims stated in the operator's free-form context notes.

    Only fires when the note NAMES segments (by id, or as "scene/segment/shot N"
    resolved positionally) — an unattached "make it continuous" has no row to
    bind to, and inventing one would be worse than saying nothing."""
    text = _clean(notes)
    if not text:
        return []
    named: List[str] = []
    low = text.lower()
    for sid in by_id:
        if _cue_span(low, sid.lower()):
            named.append(sid)
    for m in re.finditer(r"(?:scene|segment|shot)\s*#?\s*(\d+)", low):
        n = int(m.group(1))
        # Operators count from 1; index 0 is the first row.
        if 1 <= n <= len(rows) and rows[n - 1]["segment_id"] not in named:
            named.append(rows[n - 1]["segment_id"])
    if not named:
        return []
    named.sort(key=lambda s: by_id[s]["index"])
    out: List[Expectation] = []
    for kind, cues, conf in (
        (KIND_CONTINUATION, _MOTION_CONTINUATION_CUES, 0.85),
        (KIND_CONCURRENCY, _CONCURRENCY_CUES, 0.85),
        (KIND_DISCONTINUITY, _DISCONTINUITY_CUES, 0.8),
    ):
        hit = _first_cue(text, cues)
        if hit is None:
            continue
        phrase, quote = hit
        detail: Dict[str, Any] = {"cue": phrase, "from_notes": True}
        if kind == KIND_CONTINUATION:
            detail["mode"] = "vace_extend"
            # A note binds the LATER named rows (the first has nothing before it).
            segs = tuple(s for s in named if by_id[s]["index"] > 0) or tuple(named)
            detail["first_segment"] = by_id[segs[0]]["index"] == 0
            for s in segs:
                out.append(Expectation(kind, (s,), quote, conf, "rule", dict(detail)))
            continue
        out.append(Expectation(kind, tuple(named), quote, conf, "rule", detail))
    return out


def _dedupe_expectations(items: Iterable[Expectation]) -> List[Expectation]:
    """Keep the HIGHEST-confidence expectation per (kind, segments, character).

    A claim repeated in the prose and again in the notes is one claim; showing it
    twice would double a badge and (worse) double a knob decision."""
    best: Dict[Tuple[Any, ...], Expectation] = {}
    order: List[Tuple[Any, ...]] = []
    for e in items:
        key = (e.kind, e.segments, _norm_name(str(e.detail.get("character", ""))))
        if key not in best:
            best[key] = e
            order.append(key)
        elif e.confidence > best[key].confidence:
            best[key] = e
    return [best[k] for k in order]


# --------------------------------------------------------------------------- #
# THE OPTIONAL LLM SEAM
# --------------------------------------------------------------------------- #
_LLM_SYSTEM = (
    "You read film shot prompts and report, in JSON only, what CONTINUITY the "
    "prose CLAIMS — never what it should be.\n"
    "\n"
    "For each shot you are shown, report any of:\n"
    '  "continuation"   — it continues the shot before it\n'
    '  "concurrency"    — it happens at the same time as another shot\n'
    '  "discontinuity"  — it is a hard cut / a different time or place\n'
    '  "duration"       — it implies a specific clip length\n'
    "\n"
    "Return ONLY a JSON array of objects:\n"
    '[{"segment_id": "...", "kind": "...", '
    '"evidence_quote": "<words copied EXACTLY from that shot>", '
    '"confidence": 0.0-1.0}]\n'
    "\n"
    "The evidence_quote MUST be copied verbatim from the shot's own text. If a "
    "shot claims nothing, omit it. Return [] if none claim anything. No prose, "
    "no markdown, no explanation."
)


def build_llm_messages(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """The ONE advisory call's messages (one call for all ambiguous rows)."""
    body = "\n\n".join(
        f'[{r["segment_id"]}]\n{_row_text(r)}' for r in rows)
    return [{"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": "THE SHOTS:\n\n" + body}]


def parse_llm_expectations(text: str, rows: Sequence[Mapping[str, Any]]
                           ) -> Tuple[Expectation, ...]:
    """Validate a model's advisory reply into capped, marked expectations.

    EVERY row is dropped unless it survives all four checks: a known ``kind``, a
    ``segment_id`` that is one of the rows we ASKED about, an ``evidence_quote``
    that is a LITERAL substring of that row's own text (case-insensitively, with
    whitespace squeezed — the anti-fabrication check: a model that invents a
    quote is inventing the finding), and a numeric confidence. What survives is
    marked ``source="llm"`` and capped at :data:`LLM_MAX_CONFIDENCE`, so it can
    inform an operator but never turn a knob or block a submit."""
    from hugpy_engine.utils.json_scavenge import extract_json_array

    parsed = extract_json_array(text or "", accept_lone_object=True)
    if not isinstance(parsed, list):
        return ()
    texts = {r["segment_id"]: re.sub(r"\s+", " ", _row_text(r)).lower() for r in rows}
    out: List[Expectation] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        sid = _clean(item.get("segment_id"))
        kind = _clean(item.get("kind")).lower()
        quote = _clean(item.get("evidence_quote"))
        if sid not in texts or kind not in EXPECTATION_KINDS or not quote:
            continue
        if re.sub(r"\s+", " ", quote).lower() not in texts[sid]:
            continue          # fabricated quote -> fabricated finding. Dropped.
        conf = item.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            conf = 0.4
        conf = max(0.0, min(LLM_MAX_CONFIDENCE, float(conf)))
        detail: Dict[str, Any] = {"advisory": True}
        if kind == KIND_CONTINUATION:
            detail["mode"] = "still"      # the conservative join for a guess
            detail["first_segment"] = False
        out.append(Expectation(kind, (sid,), quote, conf, "llm", detail))
    return tuple(out)


def _llm_expectations(llm: Callable[[str], str],
                      rows: Sequence[Mapping[str, Any]]) -> List[Expectation]:
    """One advisory call, fully swallowed on failure.

    A coordination review must never fail because a text model is down — the
    deterministic findings are the product, and this is a bonus."""
    try:
        messages = build_llm_messages(rows)
        reply = llm(messages[-1]["content"])
        return list(parse_llm_expectations(reply if isinstance(reply, str) else "", rows))
    except Exception:  # noqa: BLE001 — advisory only, never load-bearing
        return []


def bind_coordination_llm(*, requested_model: Optional[str] = None
                          ) -> Optional[Callable[[str], str]]:
    """A live ``(prompt) -> str`` through the CATALOG's own text route, or None.

    Reuses ``oracle.screenplay.bind_llm`` (capability ``text.chat``, the k109
    routing matrix, the k97 authority gate) rather than minting new inference
    machinery. Imported INSIDE the function: building the model registry is a
    ~50s cold import and this module must stay importable from a route or a
    test without paying it. Returns ``None`` on any gap — the caller then runs
    deterministic-only, which is the supported configuration, not a degraded
    one."""
    try:
        from hugpy_oracle.screenplay import bind_llm, AuthoringGap
    except Exception:  # noqa: BLE001
        return None
    try:
        bound = bind_llm(objective="review prompt-set coordination",
                         requested_model=requested_model)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(bound, AuthoringGap):
        return None
    return bound


# --------------------------------------------------------------------------- #
# FRAME ARITHMETIC
# --------------------------------------------------------------------------- #
#: Import path of the studio module that owns the clip-length literals.
_STUDIO_SCHEMAS = "hugpy_video.intel.studio.schemas"
_STUDIO_REGISTRY = "hugpy_video.intel.studio.registry"


def _frame_limits(model_id: Optional[str]) -> Tuple[int, int, int]:
    """``(ceiling, cadence, default)`` for a model row.

    ``model_id`` narrows the ceiling to that checkpoint's ``max_frames``: the
    CODE_GAPS finding (``wan2.1-i2v-14b-720p``: "tensor a (36) vs b (16) dim 1")
    is a frame count the checkpoint does not accept, and a coordination review
    that lengthened a clip past its row's ceiling would MANUFACTURE that crash
    out of prose.

    ⚠ THE STUDIO PACKAGE IS READ ONLY IF IT IS ALREADY LOADED. Importing
    ``video_intel.studio.*`` executes ``studio/__init__`` and BUILDS THE MODEL
    REGISTRY — a measured ~50s cold import. A prompt-assist call must not pay
    that to size a clip, and a coordination review that hung for a minute would
    simply be turned off. So: when the studio is already in ``sys.modules`` (the
    movie SUBMIT path, which has just run ``preflight_movie``, and every studio
    test) the precise per-row ceiling is used; otherwise the Wan literals below
    are, which is the correct answer for every Wan row on this fleet and an
    honest conservative one for anything else (81 is the highest ceiling any row
    declares, so this can only ever under-ask, never over-ask). The fallbacks are
    asserted equal to the studio's own in ``tests/test_prompt_coordination.py``,
    so a drift fails a test rather than quoting a caller the wrong length.
    """
    import sys as _sys

    ceiling = _FALLBACK_WAN_MAX_FRAMES
    cadence = _FALLBACK_FRAME_CADENCE
    default = _FALLBACK_DEFAULT_FRAMES
    mod = _sys.modules.get(_STUDIO_SCHEMAS)
    if mod is not None:
        ceiling = getattr(mod, "WAN_MAX_FRAMES", ceiling)
        cadence = getattr(mod, "WAN_FRAME_CADENCE", cadence)
        default = getattr(mod, "DEFAULT_FRAMES_REAL", default)
    if model_id:
        reg_mod = _sys.modules.get(_STUDIO_REGISTRY)
        registry = getattr(reg_mod, "MODEL_REGISTRY", None) if reg_mod else None
        if registry is not None:
            try:
                cfg = registry.get(model_id)
                row_max = getattr(cfg, "max_frames", None) if cfg is not None else None
                if isinstance(row_max, int) and row_max > 0:
                    ceiling = min(ceiling, row_max)
            except Exception:  # noqa: BLE001 — a registry hiccup must not fail a review
                pass
    return ceiling, cadence, default


def _snap_down(n: int, cadence: int) -> int:
    """Nearest legal frame count AT OR BELOW ``n`` (``4k+1``). Snap DOWN, never
    up — up could push past the ceiling, which is the one direction that turns a
    clamp into an OOM (``studio.schemas.snap_wan_frames``'s own reasoning)."""
    n = max(1, int(n))
    return ((n - 1) // cadence) * cadence + 1


def _frames_for_seconds(seconds: float, fps: int, model_id: Optional[str]
                        ) -> Tuple[int, int, bool]:
    """``(frames, wanted, clamped)`` for a duration ask.

    ``wanted`` is the honest un-clamped number so the mismatch can say what the
    prose asked for; ``frames`` is what the checkpoint will actually accept."""
    ceiling, cadence, _default = _frame_limits(model_id)
    wanted = max(1, int(math.ceil(float(seconds) * max(1, int(fps)))))
    n = min(wanted, ceiling)
    return _snap_down(n, cadence), wanted, wanted > ceiling


# --------------------------------------------------------------------------- #
# IDENTITY
# --------------------------------------------------------------------------- #
def _profile_index(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The locked identity profiles, normalized.

    Accepts the wire shape of ``/video/identity-profiles`` (``slug`` /
    ``reference_images`` / ``authorization``) AND ``prompt_spread``'s validated
    ``identity_profile`` block (``identity_id`` / ``reference_asset_ids``), for
    the same reason ``prompt_spread._validate_identity`` does: the UI has the row
    in hand and should not have to rebuild it."""
    raw = context.get("identity_profiles")
    if raw is None:
        raw = context.get("identity_profile")
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: List[Dict[str, Any]] = []
    for p in items:
        if not isinstance(p, Mapping):
            continue
        name = _clean(p.get("name"))
        if not name:
            continue
        slug = _clean(p.get("slug")) or _clean(p.get("identity_id")) or _clean(p.get("id"))
        refs = p.get("reference_images")
        if refs is None:
            refs = p.get("reference_asset_ids")
        refs = [str(r) for r in refs] if isinstance(refs, (list, tuple)) else []
        canonical = p.get("canonical")
        canonical = [str(c) for c in canonical] if isinstance(canonical, (list, tuple)) else []
        out.append({"name": name, "slug": slug, "reference_images": refs,
                    "canonical": canonical,
                    "authorization": p.get("authorization")
                    if isinstance(p.get("authorization"), Mapping) else {}})
    return out


def _profile_for(name: str, profiles: Sequence[Mapping[str, Any]]
                 ) -> Optional[Mapping[str, Any]]:
    for p in profiles:
        if _names_agree(name, str(p.get("name") or "")):
            return p
    return None


def _likeness_authorized(profile: Mapping[str, Any],
                         lookup: Optional[Callable[[str], bool]]) -> Optional[bool]:
    """k97 consent for this profile's LIKENESS. None = unknowable here.

    Read from the profile's own ``authorization`` block when it carries one (the
    wire shape always does), else via ``lookup`` (default:
    ``identity_profiles.profile_authorized``, which is *deliberately incapable of
    returning True by omission*). An unknowable answer is NOT a yes."""
    block = profile.get("authorization")
    if isinstance(block, Mapping) and isinstance(block.get("likeness"), Mapping):
        row = block["likeness"]
        return bool(row.get("granted")) and bool(_clean(row.get("evidence")))
    slug = _clean(profile.get("slug"))
    if not slug:
        return None
    if lookup is not None:
        try:
            return bool(lookup(slug))
        except Exception:  # noqa: BLE001
            return None
    try:
        from hugpy_video.intel.identity_profiles import profile_authorized
        return bool(profile_authorized(slug, "likeness"))
    except Exception:  # noqa: BLE001
        return None


def _render_refs(profile: Mapping[str, Any]) -> List[str]:
    """The <= 4 references ONE id_lock render may consume. Prefers the promoted
    canonical ring (``identity_profiles.render_refs_from_canonical``) over the raw
    sources, exactly as the render path does."""
    canonical = list(profile.get("canonical") or [])
    if canonical:
        try:
            from hugpy_video.intel.identity_profiles import render_refs_from_canonical
            return list(render_refs_from_canonical(canonical))
        except Exception:  # noqa: BLE001
            return canonical[:4]
    return list(profile.get("reference_images") or [])[:4]


# --------------------------------------------------------------------------- #
# DERIVATION
# --------------------------------------------------------------------------- #
def derive_knobs(expectations: Sequence[Expectation], current_specs: Any,
                 context: Optional[Mapping[str, Any]] = None,
                 ) -> Tuple[KnobDecision, ...]:
    """For each expectation, the KNOB STATE that would honour it.

    Pure and deterministic given ``context`` (which may carry the identity
    profiles and the movie geometry). The ratchet rule decides ``set`` vs
    ``proposed`` vs ``mismatch``; see the module docstring.

    ``context`` keys, all optional:
      ``fps``                movie fps for the duration -> frames conversion (16)
      ``frames``             movie-level default clip length
      ``model_id``           movie-level pin, narrows the frame ceiling
      ``seed`` / ``seed_base``  the movie seed a shared concurrency seed derives from
      ``identity_profiles``  the LOCKED profiles (wire shape or spread shape)
      ``authorized``         optional ``(slug) -> bool`` override for the k97 gate
      ``allow_identity_capture``  False turns IDENTITY_CAPTURE proposals off
    """
    ctx = dict(context or {})
    rows = normalize_rows(current_specs)
    by_id = {r["segment_id"]: r for r in rows}
    fps = _int_or_none(ctx.get("fps")) or _DEFAULT_FPS
    movie_model = _clean(ctx.get("model_id")) or None
    movie_frames = _int_or_none(ctx.get("frames"))
    seed_base = _int_or_none(ctx.get("seed_base"))
    if seed_base is None:
        seed_base = _int_or_none(ctx.get("seed")) or 0
    profiles = _profile_index(ctx)
    lookup = ctx.get("authorized") if callable(ctx.get("authorized")) else None
    allow_capture = ctx.get("allow_identity_capture", True)

    out: List[KnobDecision] = []

    def emit(row: Mapping[str, Any], knob: str, current: Any, proposed: Any,
             status: str, reason: str, exp: Optional[Expectation],
             confidence: Optional[float] = None, step: Optional[str] = None,
             detail: Optional[Dict[str, Any]] = None) -> None:
        # THE RATCHET, enforced in ONE place: a locked row is the operator's own
        # work, so a change that would have been applied is only ever proposed.
        if status == STATUS_SET and row.get("locked"):
            status = STATUS_PROPOSED
            reason += " (this row is LOCKED — confirm before it is applied)"
        # An advisory (LLM) finding can never turn a knob or block a submit.
        if exp is not None and exp.source == "llm" and status == STATUS_SET:
            status = STATUS_PROPOSED
            reason += " (advisory finding — confirm before it is applied)"
        conf = confidence if confidence is not None else (
            exp.confidence if exp is not None else 0.5)
        if exp is not None and exp.source == "llm":
            conf = min(conf, LLM_MAX_CONFIDENCE)
        out.append(KnobDecision(
            segment_id=row["segment_id"], index=row["index"], knob=knob,
            current=current, proposed_value=proposed, status=status,
            reason=reason, confidence=conf, expectation=exp, step=step,
            detail=dict(detail or {})))

    for exp in expectations:
        if exp.kind == KIND_CONTINUATION:
            _derive_continuation(exp, rows, by_id, emit)
        elif exp.kind == KIND_CONCURRENCY:
            _derive_concurrency(exp, rows, by_id, seed_base, emit)
        elif exp.kind == KIND_DISCONTINUITY:
            _derive_discontinuity(exp, by_id, emit)
        elif exp.kind == KIND_DURATION:
            _derive_duration(exp, by_id, fps, movie_model, movie_frames, emit)
        elif exp.kind == KIND_RECURRING_CHARACTER:
            _derive_identity(exp, by_id, profiles, lookup, allow_capture, emit)

    return tuple(out)


def _derive_continuation(exp, rows, by_id, emit) -> None:
    """Continuation -> a CARRYING join + a parent pointer on the linear chain."""
    sid = exp.segments[-1]
    row = by_id.get(sid)
    if row is None:
        return
    wanted = exp.detail.get("mode") or "still"
    if wanted not in _CARRYING_MODES:
        wanted = "still"

    if row["index"] == 0:
        # goal 0 has no parent — ``make_studio_movie`` enforces "still" on the
        # root and there is nothing to splice onto. The knobs CANNOT express
        # this claim, so it is a mismatch, never a silent no-op.
        emit(row, KNOB_JOINT_MODE, row["joint_mode"], wanted, STATUS_MISMATCH,
             "this is the FIRST shot — it claims to continue a previous shot, but "
             "there is nothing before it to carry. Reorder the timeline, or give "
             "the movie a start_image and drop the continuation language.", exp)
        return

    previous = rows[row["index"] - 1]
    current = row["joint_mode"]

    # ── joint_mode ──────────────────────────────────────────────────────────
    if current == wanted:
        emit(row, KNOB_JOINT_MODE, current, wanted, STATUS_OK,
             f"the prose continues the previous shot and the join is already "
             f"{wanted!r}", exp)
    elif current is None or current == "cut":
        # THE k121 CASE: the words said continuous, the knob stayed cut.
        emit(row, KNOB_JOINT_MODE, current, wanted, STATUS_SET,
             f"the prose continues the previous shot, so the join is set to "
             f"{wanted!r} " + (
                 "(the parent's motion is carried across the splice)"
                 if wanted == "vace_extend" else
                 "(the shot begins from the parent's frame)"), exp)
    elif current == "still" and wanted == "vace_extend":
        # still -> vace_extend ADDS carry (motion as well as a frame): ratchet-safe.
        emit(row, KNOB_JOINT_MODE, current, wanted, STATUS_SET,
             "the prose continues the previous shot's MOTION; a 'still' join "
             "carries a frame but not motion, so the join is raised to "
             "'vace_extend'", exp)
    else:
        # vace_extend already carries more than a 'still' ask needs: leave it.
        emit(row, KNOB_JOINT_MODE, current, current, STATUS_OK,
             f"the join is already {current!r}, which carries at least what the "
             f"prose asks for", exp)

    # ── parent_segment_id (the linear-chain tree rule) ──────────────────────
    parent = row["parent_segment_id"]
    if parent is None:
        emit(row, KNOB_PARENT, None, previous["segment_id"], STATUS_SET,
             f"linked to {previous['segment_id']!r}, the shot immediately before "
             f"it (studio take-tree v0 is a linear chain)", exp)
    elif parent == previous["segment_id"]:
        emit(row, KNOB_PARENT, parent, parent, STATUS_OK,
             "the parent pointer already names the previous shot", exp)
    else:
        # A non-previous parent is a real TREE (sibling divergence), which
        # ``make_studio_movie`` refuses today. Re-pointing it would silently
        # discard the operator's branch, so this is reported, never rewritten.
        emit(row, KNOB_PARENT, parent, previous["segment_id"], STATUS_MISMATCH,
             f"this shot's parent is {parent!r} but the shot before it is "
             f"{previous['segment_id']!r}; the studio take-tree is a LINEAR chain "
             f"today, so a submit will refuse this. Reorder the rows or repoint "
             f"the parent.", exp, confidence=max(exp.confidence, BLOCK_CONFIDENCE))

    # ── branch_frame: last-frame carry ──────────────────────────────────────
    if row["branch_frame"] is None:
        emit(row, KNOB_BRANCH_FRAME, None, None, STATUS_OK,
             "no branch frame is pinned, so this shot conditions on the parent's "
             "LAST frame — the carry the prose describes", exp)
    else:
        emit(row, KNOB_BRANCH_FRAME, row["branch_frame"], None, STATUS_PROPOSED,
             f"this shot is pinned to frame {row['branch_frame']} of the parent, "
             f"so it does NOT pick up from where the parent ends. Clearing the pin "
             f"carries the parent's last frame instead.", exp)


def _derive_concurrency(exp, rows, by_id, seed_base, emit) -> None:
    """Same moment, different place -> a HARD CUT whose world still matches.

    The knobs that make two shots read as the same moment are a shared seed (one
    lighting/grade lottery for both) and shared setting references — NOT a frame
    carry, which would make them the same shot. Text is never copied.

    ONLY THE CLAIMING SEGMENT'S JOIN IS DECIDED. ``exp.segments`` is
    ``(anchor, claimant)``: the claimant is the row whose prose said "meanwhile",
    and the anchor is merely the shot it is concurrent WITH. Deciding the
    anchor's own join here would let a "meanwhile" in shot 3 overrule a
    "continuous" in shot 2 — two expectations fighting over one knob on a row
    that never claimed anything. The anchor contributes its seed and its
    references, and nothing else."""
    segs = [s for s in exp.segments if s in by_id]
    if len(segs) < 2:
        return
    anchor = by_id[segs[0]]
    claimant = by_id[segs[-1]]
    shared_seed = anchor["seed"] if anchor["seed"] is not None else (
        (int(seed_base) + anchor["index"]) & 0x7FFFFFFF)
    anchor_refs = anchor["reference_images"]

    current = claimant["joint_mode"]
    if claimant["index"] == 0:
        pass  # the root's join is fixed at "still" by the factory; nothing to say
    elif current == "cut":
        emit(claimant, KNOB_JOINT_MODE, current, "cut", STATUS_OK,
             "concurrent shots are a hard cut, and this join is already 'cut'", exp)
    elif current in _CARRYING_MODES:
        # -> cut REMOVES carry. Never automatic (the ratchet).
        emit(claimant, KNOB_JOINT_MODE, current, "cut", STATUS_PROPOSED,
             f"the prose says this happens at the same time somewhere else, which "
             f"is a hard cut — but the join is {current!r}, which conditions this "
             f"shot on the previous one's frame. Dropping a carry is destructive, "
             f"so confirm it.", exp)
    else:
        emit(claimant, KNOB_JOINT_MODE, current, "cut", STATUS_SET,
             "concurrent shots are spliced as a hard cut (no frame carry)", exp)

    for sid in segs:
        row = by_id[sid]
        if row["seed"] is None:
            emit(row, KNOB_SEED, None, shared_seed, STATUS_SET,
                 f"this shot and {(claimant if sid == segs[0] else anchor)['segment_id']!r} "
                 f"are the same moment, so they share a seed base ({shared_seed}) "
                 f"and render in the same light", exp)
        elif row["seed"] == shared_seed:
            emit(row, KNOB_SEED, row["seed"], shared_seed, STATUS_OK,
                 "this shot already shares the concurrency seed base", exp)
        else:
            emit(row, KNOB_SEED, row["seed"], shared_seed, STATUS_PROPOSED,
                 f"this shot pins seed {row['seed']}, which differs from its "
                 f"concurrent sibling's {shared_seed}; the same moment will render "
                 f"in two different lotteries. Re-seeding overwrites a pinned "
                 f"choice, so confirm it.", exp)

    if anchor_refs and not claimant["reference_images"]:
        emit(claimant, KNOB_REFERENCE_IMAGES, [], list(anchor_refs), STATUS_SET,
             f"the concurrent shot {anchor['segment_id']!r} conditions on "
             f"{len(anchor_refs)} setting/subject reference(s); sharing them holds "
             f"the same world across the cut", exp)


def _derive_discontinuity(exp, by_id, emit) -> None:
    """Explicit cut language against a CARRYING join — the contradiction case.

    Flipping to ``cut`` would drop the frame carry (and a ``cut`` node must have
    ``branch_frame``/``context_frames`` cleared — the factory rejects otherwise),
    so this is exactly the class the review REPORTS and refuses to auto-set."""
    sid = exp.segments[0]
    row = by_id.get(sid)
    if row is None or row["index"] == 0:
        return
    current = row["joint_mode"]
    if current in _CARRYING_MODES:
        emit(row, KNOB_JOINT_MODE, current, "cut", STATUS_MISMATCH,
             f"the prose calls for a hard cut / a new time and place, but the join "
             f"is {current!r}, which conditions this shot on the previous one's "
             f"frame" + (" and its motion" if current == "vace_extend" else "") +
             ". The words and the knob disagree; dropping a carry is never "
             "automatic — set the join to 'cut' or drop the cut language.",
             exp, confidence=max(exp.confidence, BLOCK_CONFIDENCE))
    elif current == "cut":
        emit(row, KNOB_JOINT_MODE, current, "cut", STATUS_OK,
             "the prose calls for a hard cut and the join is already 'cut'", exp)
    else:
        emit(row, KNOB_JOINT_MODE, current, "cut", STATUS_SET,
             "the prose calls for a hard cut, so the join is set to 'cut'", exp)


def _derive_duration(exp, by_id, fps, movie_model, movie_frames, emit) -> None:
    """A stated / implied duration -> a per-segment ``frames`` recomputation.

    CLAMPED to the checkpoint's ceiling and snapped to the 4k+1 cadence, because
    the render path clamps anyway (``resolve_frames``) and because an unclamped
    ask is the CODE_GAPS ``wan2.1-i2v-14b-720p`` crash. When the ask exceeds the
    ceiling the clamped value is still set AND a mismatch says so — the caller is
    never told "no" without being told what they actually get."""
    sid = exp.segments[0]
    row = by_id.get(sid)
    if row is None:
        return
    seconds = float(exp.detail.get("seconds") or 0.0)
    if seconds <= 0:
        return
    model_id = row["model_id"] or movie_model
    frames, wanted, clamped = _frames_for_seconds(seconds, fps, model_id)
    ceiling, _cadence, default = _frame_limits(model_id)
    current = row["frames"] if row["frames"] is not None else movie_frames
    effective = current if current is not None else default

    if frames > effective:
        emit(row, KNOB_FRAMES, current, frames, STATUS_SET,
             f"the shot implies ~{seconds:.1f}s; at {fps}fps that is {frames} "
             f"frames, up from {effective}. Lengthening a clip adds nothing "
             f"destructive, so it is applied.", exp,
             detail={"seconds": round(seconds, 2), "fps": fps,
                     "wanted_frames": wanted, "ceiling": ceiling})
    elif clamped:
        # The clip is already AT the ceiling and the prose wants more. Saying
        # "ok" here and "mismatch" below would be two contradictory badges on one
        # knob; the mismatch is the whole truth, so it is the only thing emitted.
        pass
    elif frames == effective:
        emit(row, KNOB_FRAMES, current, frames, STATUS_OK,
             f"the shot implies ~{seconds:.1f}s and the clip is already {frames} "
             f"frames at {fps}fps", exp,
             detail={"seconds": round(seconds, 2), "fps": fps})
    else:
        emit(row, KNOB_FRAMES, current, frames, STATUS_PROPOSED,
             f"the shot implies ~{seconds:.1f}s ({frames} frames at {fps}fps) but "
             f"the clip is {effective} frames. Shortening throws rendered time "
             f"away, so confirm it.", exp,
             detail={"seconds": round(seconds, 2), "fps": fps})

    if clamped:
        emit(row, KNOB_FRAMES, current, frames, STATUS_MISMATCH,
             f"the prose asks for ~{seconds:.1f}s ({wanted} frames at {fps}fps), "
             f"which is past what this checkpoint accepts: "
             f"{model_id or 'the bound model'} tops out at {ceiling} frames "
             f"({ceiling / max(1, fps):.2f}s). The clip is set to {frames} frames "
             f"and the rest of the beat has to move to another shot.", exp,
             # Advisory: there is no knob that fixes this, so it must not block.
             confidence=min(exp.confidence, BLOCK_CONFIDENCE - 0.1),
             detail={"seconds": round(seconds, 2), "wanted_frames": wanted,
                     "ceiling": ceiling, "fps": fps})


def _derive_identity(exp, by_id, profiles, lookup, allow_capture, emit) -> None:
    """A recurring character -> an identity PLAN, consent-gated (k97).

    Three worlds:
      * a LOCKED, AUTHORIZED profile exists -> attach its render refs to every
        segment that names the character (``set``: capability ``id_lock`` then
        carries the subject across every cut — the whole point of the field);
      * a profile exists but carries NO k97 likeness consent -> ``mismatch``.
        Not a proposal: consent is a human rights decision, never a checkbox the
        review offers to tick for you (``profile_authorized`` is deliberately
        incapable of returning True by omission);
      * NO profile — the character FIRST APPEARS in an earlier segment's own
        output -> propose :data:`STEP_IDENTITY_CAPTURE`: build a profile from
        that segment's ACCEPTED frames through the existing identity pipeline,
        then attach it to the later segments. Proposed, never automatic: capture
        spends real work and, for a real-person likeness, needs consent first.
    """
    name = _clean(exp.detail.get("character"))
    segs = [s for s in exp.segments if s in by_id]
    if not name or len(segs) < 2:
        return
    profile = _profile_for(name, profiles)

    if profile is not None:
        authorized = _likeness_authorized(profile, lookup)
        refs = _render_refs(profile)
        slug = _clean(profile.get("slug"))
        if authorized is not True:
            for sid in segs:
                row = by_id[sid]
                emit(row, KNOB_IDENTITY, list(row["reference_images"]),
                     {"profile": slug or name, "reference_images": refs},
                     STATUS_MISMATCH,
                     f"{name!r} recurs across {len(segs)} shots and a profile "
                     f"exists, but it carries no recorded likeness authorization "
                     f"(k97). Identity references are NOT attached. Record consent "
                     f"with its evidence on the profile, then re-review.", exp,
                     confidence=max(exp.confidence, BLOCK_CONFIDENCE),
                     detail={"character": name, "profile": slug,
                             "consent": "missing"})
            return
        if not refs:
            for sid in segs:
                row = by_id[sid]
                emit(row, KNOB_IDENTITY, list(row["reference_images"]),
                     {"profile": slug or name, "reference_images": []},
                     STATUS_MISMATCH,
                     f"{name!r} recurs across {len(segs)} shots and its profile is "
                     f"authorized, but it holds no reference images to condition "
                     f"on — nothing can be attached.", exp,
                     confidence=min(exp.confidence, BLOCK_CONFIDENCE - 0.1),
                     detail={"character": name, "profile": slug})
            return
        for sid in segs:
            row = by_id[sid]
            if list(row["reference_images"]) == refs:
                emit(row, KNOB_REFERENCE_IMAGES, list(row["reference_images"]), refs,
                     STATUS_OK,
                     f"{name!r} is already locked to profile {slug or name!r} on "
                     f"this shot", exp, detail={"character": name, "profile": slug})
            elif row["reference_images"]:
                emit(row, KNOB_REFERENCE_IMAGES, list(row["reference_images"]), refs,
                     STATUS_PROPOSED,
                     f"this shot already conditions on {len(row['reference_images'])} "
                     f"reference(s); swapping them for {name!r}'s profile "
                     f"({slug or name}) would replace an explicit choice.", exp,
                     detail={"character": name, "profile": slug})
            else:
                emit(row, KNOB_REFERENCE_IMAGES, [], refs, STATUS_SET,
                     f"{name!r} recurs across {len(segs)} shots and profile "
                     f"{slug or name!r} is authorized, so its {len(refs)} reference "
                     f"image(s) are attached — every shot renders id_lock and the "
                     f"subject carries across the cuts", exp,
                     detail={"character": name, "profile": slug,
                             "capability": "id_lock"})
        return

    # ── no profile: the character is BORN in an earlier segment's render ────
    if not allow_capture:
        return
    origin = segs[0]
    for sid in segs[1:]:
        row = by_id[sid]
        emit(row, KNOB_IDENTITY, list(row["reference_images"]),
             {"from_segment": origin, "character": name},
             STATUS_PROPOSED,
             f"{name!r} first appears in {origin!r} and returns here, but no "
             f"identity profile exists — so nothing carries the face across the "
             f"cut. Proposed: capture a profile from {origin!r}'s ACCEPTED frames "
             f"and lock this shot to it. Capture is real work and, for a "
             f"real-person likeness, needs recorded consent (k97) before it runs.",
             exp, step=STEP_IDENTITY_CAPTURE,
             detail={"character": name, "from_segment": origin,
                     "target_segments": segs[1:], "consent": "required",
                     "pipeline": "identity_profiles.create_profile"})


# --------------------------------------------------------------------------- #
# REVIEW
# --------------------------------------------------------------------------- #
def review(specs: Any, *, notes: str = "",
           context: Optional[Mapping[str, Any]] = None,
           llm: Optional[Callable[[str], str]] = None,
           ) -> CoordinationReport:
    """THE entry point: review a generated/enhanced prompt set's knobs.

    Never silent — EVERY segment appears in the report, including the ones whose
    knobs already matched their words (``ok``). That is the operator's directive:
    *"every aspect of the above needs to be explicitly reviewed upon the creation
    of any generated or enhanced prompting in video."* A review that only spoke
    up on failure would leave "nothing was reviewed" and "everything was fine"
    looking identical, which is how the original incident stayed invisible.
    """
    rows = normalize_rows(specs)
    expectations = extract_expectations(rows, notes=notes, llm=llm)
    decisions = derive_knobs(expectations, rows, context)

    by_segment: Dict[str, List[KnobDecision]] = {r["segment_id"]: [] for r in rows}
    for d in decisions:
        by_segment.setdefault(d.segment_id, []).append(d)

    segments: List[SegmentReview] = []
    for row in rows:
        mine = tuple(by_segment.get(row["segment_id"], ()))
        status = STATUS_OK
        for s in STATUS_ORDER:
            if any(d.status == s for d in mine):
                status = s
                break
        segments.append(SegmentReview(row["segment_id"], row["index"], status, mine))

    return CoordinationReport(
        segments=tuple(segments), expectations=tuple(expectations),
        decisions=tuple(decisions),
        llm_used=any(e.source == "llm" for e in expectations))


def review_goals(spec: Any, *, context: Optional[Mapping[str, Any]] = None,
                 ) -> CoordinationReport:
    """:func:`review` for a built ``StudioMovieSpec`` (the submit preflight).

    Reads the goals' own knobs off the frozen spec and folds the movie-level
    geometry into the context, so what is reviewed at submit is exactly what
    would be rendered."""
    goals = list(getattr(spec, "goals", ()) or ())
    rows = []
    for i, g in enumerate(goals):
        rows.append({
            "segment_id": getattr(g, "segment_id", "") or f"segment_{i}",
            "index": i,
            "prompt": getattr(g, "prompt", "") or "",
            "joint_mode": getattr(g, "joint_mode", None),
            "parent_segment_id": getattr(g, "parent_segment_id", None),
            "branch_frame": getattr(g, "branch_frame", None),
            "frames": getattr(g, "frames", None),
            "seed": getattr(g, "seed", None),
            "model_id": getattr(g, "model_id", None),
            "reference_images": list(getattr(g, "reference_images", None)
                                     or getattr(spec, "reference_images", ()) or ()),
        })
    ctx = dict(context or {})
    ctx.setdefault("fps", getattr(spec, "fps", None) or _DEFAULT_FPS)
    ctx.setdefault("frames", getattr(spec, "frames", None))
    ctx.setdefault("model_id", getattr(spec, "model_id", None))
    ctx.setdefault("seed_base", getattr(spec, "seed", 0))
    # A movie-level identity is already attached to every segment; proposing a
    # capture on top of it would be noise.
    if getattr(spec, "reference_images", ()):
        ctx.setdefault("allow_identity_capture", False)
    return review(rows, context=ctx)


def apply_decisions(rows: Any, report: CoordinationReport
                    ) -> Tuple[List[Dict[str, Any]], List[KnobDecision]]:
    """Apply ONLY the ``set`` decisions to ``rows``; return ``(rows, applied)``.

    INVARIANT 9 LIVES HERE. The only keys this may write are
    :data:`APPLIABLE_KNOBS` — mechanics. ``prompt`` is not among them, so there
    is no code path through which one row's generated text can reach another
    row. (``tests/test_prompt_coordination.py`` asserts the prose is
    byte-identical after a full apply.)

    Input rows are not mutated: a shallow copy per row is returned, so a caller
    can diff what changed.
    """
    out: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}
    for raw in (rows or []):
        row = dict(raw)
        out.append(row)
        sid = _clean(row.get("segment_id")) or _clean(row.get("id"))
        if sid:
            index[sid] = row
    applied: List[KnobDecision] = []
    for d in report.decisions:
        if d.status != STATUS_SET or d.knob not in APPLIABLE_KNOBS:
            continue
        row = index.get(d.segment_id)
        if row is None:
            continue
        value = d.proposed_value
        if d.knob == KNOB_REFERENCE_IMAGES and isinstance(value, (list, tuple)):
            value = list(value)
        row[d.knob] = value
        applied.append(d)
    return out, applied


def blocking_mismatches(report: CoordinationReport,
                        threshold: float = BLOCK_CONFIDENCE) -> List[Dict[str, Any]]:
    """The submit-blocking rows, shaped like ``movie_plan.preflight_movie``'s.

    Same dict vocabulary (``index`` / ``segment_id`` / ``reason`` / ``detail``)
    so the movie route can refuse in ONE style, and a console that already reads
    capability refusals reads these with no new code."""
    out: List[Dict[str, Any]] = []
    for d in report.decisions:
        if d.status != STATUS_MISMATCH or d.confidence < threshold:
            continue
        out.append({
            "index": d.index, "segment_id": d.segment_id,
            "reason": "coordination_mismatch", "knob": d.knob,
            "current": d.current, "proposed": d.proposed_value,
            "detail": d.reason, "evidence_quote": d.evidence_quote,
            "confidence": round(float(d.confidence), 3),
            "expectation": d.expectation.as_dict() if d.expectation else None,
        })
    return out
