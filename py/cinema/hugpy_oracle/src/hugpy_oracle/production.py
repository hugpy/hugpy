"""GenerationSnapshot + the production lock (k104) — doc Stages 4, 7, 9, 10, 11.

This module holds the artifacts that are LOCKED before a single frame is
generated, and the transition that locks them. It is the upstream half of
Stage 14: ``oracle/segments.py`` compiles sibling ``SegmentSpec``s out of what
is locked here, and out of nothing else.

    Stage  4  GenerationSnapshot   the immutable pre-run source snapshot
    Stage  7  ContinuityBible      explicit state_before / state_after per segment
    Stage  9  ShotPlan             camera, blocking, lighting, per-shot rubric
    Stage 11  ProductionLock       one versioned digest over all of the above

WHY A SNAPSHOT IS NOT JUST "THE INPUTS". Stage 4 is one sentence with teeth:
*"Prompts created during the active generation run may not become sibling
prompt inputs."* A snapshot therefore carries ``prompts_before_run`` and only
those. The enforcement is ``RunPromptLedger``: the orchestrator records the
digest of every prompt it MINTS during the run, and any attempt to feed one of
those back into a snapshot is refused by digest, not by convention. Without the
ledger the rule is a comment; with it, the rule is a raise.

    ledger = RunPromptLedger()
    written = writer(locked_context, i)      # minted DURING the run
    ledger.record(written)
    snapshot.with_prompt(written, ledger=ledger)   # -> RunPromptRefused

WHY THE LOCK VALIDATES ITS INPUTS INSTEAD OF TRUSTING THEM. A lock is a
promise that everything downstream can stop asking questions. ``lock()``
therefore refuses an ``AudioMaster`` that is not itself ``locked`` (Stage 8: the
definitive audio timeline precedes final shot timing — locking shots against a
draft timeline is exactly the mistake the doc names), a shot plan that drops a
spoken line, a continuity bible missing a segment's before/after, shot windows
that run off the end of the audio, overlapping windows, and identity refs the
immutable snapshot never carried. Each refusal is a typed ``LockRefused`` that
names the artifact and the reason.

WHY ``revise()`` AND NOT MUTATION. Stage 10: spatial feasibility may feed
revisions back into the screenplay or shot plan *only before the lock*; after
locking, "a material narrative or spatial change requires a new artifact
revision and graph validation". So the lock is frozen and ``revise(reason)``
returns revision N+1 with ``parent_revision`` set — the same shape k103 gave
``PlanGraph.revise``, for the same reason. An unexplained revision is refused:
a production change nobody wrote down is an unaudited production change.

NO CLOCK, NO REGISTRY, NO DISK. ``created_at`` / ``locked_at`` are strings the
caller supplies (ISO-8601 by convention, like ``Authorization.granted_at``);
nothing here calls ``time``. A digest that changed because you ran it twice
would not be an id. Imports are stdlib + ``.contracts`` (via ``.plan``) +
``.audio_master``; in particular this module does NOT import ``video_intel``,
because that package's ``__init__`` builds the model registry and a contract
that costs two seconds to import stops being importable from anywhere. The one
vocabulary borrowed from over there (``CAMERA_VIEWS``) is MIRRORED with a
keep-in-sync test, the same way ``prompt_spread`` mirrors
``studio_movie_schema._VALID_JOINT_MODES``.

No pathlib anywhere. os.path only (not that this module touches the disk).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Mapping, Sequence

from hugpy_oracle.audio_master import AudioMaster
from hugpy_oracle.plan import FrozenParams

# Seconds are quantized before they reach a digest: 1 microsecond, the same
# grid k102 puts its timings on, so a window and the audio it plays cannot
# disagree in the last float bit and fork two digests over nothing.
TIME_PRECISION = 6
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProductionError(ValueError):
    """Base for every refusal in the production-lock family."""


class RunPromptRefused(ProductionError):
    """Stage 4: a prompt minted DURING the run was offered as a sibling input.

    ``prompt_digest`` names the offending prompt by digest — the caller can
    match it against its own ledger without the text travelling any further."""

    def __init__(self, message: str, prompt_digest: str = "") -> None:
        super().__init__(message)
        self.prompt_digest = prompt_digest


class LockRefused(ProductionError):
    """``ProductionLock.lock()`` / ``revise()`` said no, and said why."""


# ---------------------------------------------------------------------------
# Canonical JSON + content addressing
# ---------------------------------------------------------------------------


from hugpy_oracle.audio_master import canonical_json


def digest_payload(payload: Any) -> str:
    """sha256 over :func:`canonical_json` of ``payload``."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def prompt_digest(prompt: str) -> str:
    """The stable id of a PROMPT.

    Namespaced under a ``{"prompt": …}`` envelope on purpose: a prompt digest
    and an artifact digest must never collide in a ledger lookup, and wrapping
    is cheaper than carrying a type tag alongside every digest."""
    return digest_payload({"prompt": str(prompt)})


def _q(seconds: Any) -> float:
    """Quantize a second-value. ``+ 0.0`` normalizes -0.0, which JSON-encodes
    differently and would fork a digest for nothing."""
    return round(float(seconds), TIME_PRECISION) + 0.0


class ContentAddressed:
    """Canonical-JSON identity for every artifact in the k104 family.

    ``__slots__ = ()`` so slotted dataclasses can inherit without growing a
    ``__dict__``; ``digest`` is a PROPERTY because an artifact's id is a
    function of its values and must never be a stored field that can drift from
    them (invariant 4: artifacts are immutable and content-addressed)."""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:      # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


from hugpy_oracle.validation import require_text as _require_text


from hugpy_oracle.validation import require_non_negative as _require_non_negative


def _str_tuple(values: Any, what: str) -> tuple[str, ...]:
    """A deduplicated, order-preserving tuple of non-empty strings.

    Dedup is first-seen so the field is stable under a caller that appends the
    same reference twice; order is preserved because a reference list is read
    by humans and re-sorting it would lose the operator's grouping."""
    if values is None:
        return ()
    if isinstance(values, str):
        raise TypeError(f"{what} takes a sequence of strings, not a bare "
                        f"string (a lone str is the missing-brackets bug)")
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            raise ValueError(f"{what} must not contain an empty entry")
        if text not in out:
            out.append(text)
    return tuple(out)


# ---------------------------------------------------------------------------
# Stage 4 — the immutable generation snapshot
# ---------------------------------------------------------------------------


class RunPromptLedger:
    """Digests of every prompt MINTED during the active generation run.

    This is the enforcement side of invariant 9 / Stage 4. The orchestrator
    records each prompt it generates; the snapshot consults the ledger before
    accepting a prompt as a sibling input. Because the ledger stores digests
    and not text, it can be handed across a process boundary, journalled, or
    put in a receipt without disclosing anything.

    Deliberately MUTABLE — it is a run-scoped log, not an artifact. Nothing
    downstream digests it, so it has no ``to_dict``/``digest``; ``digests``
    hands back a sorted tuple when a caller wants a stable snapshot of it.

    ``in`` accepts either a prompt or a digest: a 64-character lowercase hex
    string is read as a digest, anything else is hashed first. That rule is
    stated rather than guessed because a prompt that happens to be 64 hex
    characters would otherwise be silently misread."""

    __slots__ = ("_digests",)

    def __init__(self, prompts: Iterable[str] = (),
                 digests: Iterable[str] = ()) -> None:
        self._digests: set[str] = set()
        for prompt in prompts:
            self.record(prompt)
        for digest in digests:
            self.record_digest(digest)

    # -- writing -----------------------------------------------------------

    def record(self, prompt: str) -> str:
        """Record a prompt minted during the run; returns its digest."""
        digest = prompt_digest(_require_text(prompt, "RunPromptLedger.record"))
        self._digests.add(digest)
        return digest

    def record_all(self, prompts: Iterable[str]) -> tuple[str, ...]:
        return tuple(self.record(p) for p in prompts)

    def record_digest(self, digest: str) -> str:
        text = _require_text(digest, "RunPromptLedger.record_digest").strip()
        self._digests.add(text)
        return text

    # -- reading -----------------------------------------------------------

    @staticmethod
    def _as_digest(item: str) -> str:
        text = str(item or "")
        stripped = text.strip()
        if len(stripped) == 64 and all(c in "0123456789abcdef" for c in stripped):
            return stripped
        return prompt_digest(text)

    def __contains__(self, item: Any) -> bool:
        return self._as_digest(item) in self._digests

    def __len__(self) -> int:
        return len(self._digests)

    def __iter__(self) -> Iterator[str]:
        return iter(self.digests)

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return f"RunPromptLedger({len(self._digests)} minted prompt(s))"

    @property
    def digests(self) -> tuple[str, ...]:
        """Every recorded digest, sorted — a stable view of a mutable log."""
        return tuple(sorted(self._digests))

    def refuses(self, prompt_or_digest: str) -> bool:
        """True when this prompt was minted during the run and may therefore
        not become a sibling input."""
        return prompt_or_digest in self

    def assert_admissible(self, prompt: str) -> str:
        """Return ``prompt``'s digest, or raise ``RunPromptRefused``."""
        digest = prompt_digest(prompt)
        GenerationSnapshot.assert_not_run_prompt(digest, self._digests)
        return digest


@dataclass(frozen=True, slots=True)
class GenerationSnapshot(ContentAddressed):
    """Doc Stage 4 — the immutable source snapshot, created BEFORE generation.

    Every field is a pre-run fact:

    ``raw_request_ref``      invariant 1: the operator's raw request, by
                             reference, never re-normalized in place.
    ``prompts_before_run``   prompts that EXISTED before execution. A prompt
                             minted during the run belongs in a
                             ``RunPromptLedger``, not here (invariant 9).
    ``operator_refs``        operator-supplied references.
    ``acquisition_refs``     authorized acquisition artifacts (k122's web
                             adapters, when they land — a ref, not a payload).
    ``identity_refs`` /
    ``voice_refs``           the identity and voice analyses, as k97-shaped
                             subject refs (``identity_profile:ana``), so the
                             authority gate reads them with the same scanner.
    ``deliverable``          what was requested. Required: a snapshot with no
                             requested deliverable cannot gate anything.
    ``exclusions``           what was ruled out — carried so a later stage
                             cannot quietly reintroduce it.
    ``registry_version``     the ACCEPTED model-routing registry version
                             (k105). ``None`` while the registry is unversioned
                             — never a guess.

    ``created_at`` is caller-supplied ISO-8601 (or ``""``); this module owns no
    clock, so two snapshots of the same facts share a digest."""

    raw_request_ref: str
    prompts_before_run: tuple[str, ...] = ()
    operator_refs: tuple[str, ...] = ()
    acquisition_refs: tuple[str, ...] = ()
    identity_refs: tuple[str, ...] = ()
    voice_refs: tuple[str, ...] = ()
    deliverable: str = ""
    exclusions: tuple[str, ...] = ()
    registry_version: str | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        _require_text(self.raw_request_ref, "GenerationSnapshot.raw_request_ref")
        _require_text(self.deliverable,
                      "GenerationSnapshot.deliverable (Stage 4 lists the "
                      "requested deliverable as snapshot content; a snapshot "
                      "that does not say what was asked for gates nothing)")
        for name in ("prompts_before_run", "operator_refs", "acquisition_refs",
                     "identity_refs", "voice_refs", "exclusions"):
            object.__setattr__(self, name,
                               _str_tuple(getattr(self, name),
                                          f"GenerationSnapshot.{name}"))
        if self.registry_version is not None:
            object.__setattr__(self, "registry_version",
                               _require_text(self.registry_version,
                                             "GenerationSnapshot.registry_version"))

    # -- Stage 4's rule ----------------------------------------------------

    @property
    def prompt_digests(self) -> tuple[str, ...]:
        """One digest per pre-run prompt, in declaration order."""
        return tuple(prompt_digest(p) for p in self.prompts_before_run)

    @staticmethod
    def assert_not_run_prompt(prompt_digest_: str,
                              run_prompt_digests: Iterable[str]) -> str:
        """Invariant 9, by digest. Raise ``RunPromptRefused`` when
        ``prompt_digest_`` names a prompt that was minted during the run.

        A static method because the check is about the PROMPT, not about any
        one snapshot: the orchestrator, the compiler and the routes all make
        the same call and must not each grow their own version of it."""
        digest = _require_text(prompt_digest_, "prompt digest").strip()
        minted = {str(d).strip() for d in run_prompt_digests}
        if digest in minted:
            raise RunPromptRefused(
                f"prompt {digest[:12]}… was generated during this run and may "
                f"not become a sibling prompt input (invariant 9 / Stage 4: "
                f"the snapshot carries only prompts that existed BEFORE "
                f"generation execution)", prompt_digest=digest)
        return digest

    def assert_pre_run(self, ledger: "RunPromptLedger | Iterable[str]"
                       ) -> "GenerationSnapshot":
        """Every prompt in this snapshot predates the run, or raise.

        Cheap enough to call at the lock, which is where it earns its keep: a
        snapshot assembled early and appended to later is exactly how a
        run-minted prompt sneaks in."""
        minted = (ledger.digests if isinstance(ledger, RunPromptLedger)
                  else tuple(ledger))
        for digest in self.prompt_digests:
            self.assert_not_run_prompt(digest, minted)
        return self

    def with_prompt(self, prompt: str, *,
                    ledger: "RunPromptLedger | Iterable[str] | None" = None
                    ) -> "GenerationSnapshot":
        """A NEW snapshot carrying one more pre-run prompt (invariant 4: never
        mutate, always re-version). Refuses when ``ledger`` says the prompt was
        minted during the run."""
        text = _require_text(prompt, "GenerationSnapshot.with_prompt(prompt)")
        if ledger is not None:
            minted = (ledger.digests if isinstance(ledger, RunPromptLedger)
                      else tuple(ledger))
            self.assert_not_run_prompt(prompt_digest(text), minted)
        return replace(self,
                       prompts_before_run=self.prompts_before_run + (text,))

    # -- reading -----------------------------------------------------------

    @property
    def refs(self) -> tuple[str, ...]:
        """Every reference the snapshot carries, deduplicated in field order —
        what a disclosure or authority pass iterates over."""
        return _str_tuple(list(self.operator_refs) + list(self.acquisition_refs)
                          + list(self.identity_refs) + list(self.voice_refs),
                          "GenerationSnapshot.refs")

    # -- wire shape --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_request_ref": self.raw_request_ref,
            "prompts_before_run": list(self.prompts_before_run),
            "operator_refs": list(self.operator_refs),
            "acquisition_refs": list(self.acquisition_refs),
            "identity_refs": list(self.identity_refs),
            "voice_refs": list(self.voice_refs),
            "deliverable": self.deliverable,
            "exclusions": list(self.exclusions),
            "registry_version": self.registry_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GenerationSnapshot":
        return cls(
            raw_request_ref=d["raw_request_ref"],
            prompts_before_run=tuple(d.get("prompts_before_run", ())),
            operator_refs=tuple(d.get("operator_refs", ())),
            acquisition_refs=tuple(d.get("acquisition_refs", ())),
            identity_refs=tuple(d.get("identity_refs", ())),
            voice_refs=tuple(d.get("voice_refs", ())),
            deliverable=d.get("deliverable", ""),
            exclusions=tuple(d.get("exclusions", ())),
            registry_version=d.get("registry_version"),
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Stage 7 — continuity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContinuityState(ContentAddressed):
    """Doc Stage 7's last bullet, verbatim: *"Explicit ``state_before`` and
    ``state_after`` for every segment."*

    Explicit and SEPARATE is the whole point. A single "state" field forces
    every consumer to guess whether it describes the shot's opening or its
    close, and continuity errors live exactly in that gap — the coat that is on
    at the start of one shot and off at the start of the next. Two mappings
    make the delta computable (:attr:`changed_keys`) instead of inferred.

    Both mappings are ``FrozenParams`` (k103's hashable, JSON-safe mapping), so
    a state is immutable, digestible and refuses a value that could not survive
    a round trip through the artifact store.

    The CONTENT — what the keys mean, which are load-bearing — is authored by
    k110's LLM pass. This is the typed shell it fills."""

    segment_id: str
    state_before: Mapping[str, Any] = field(default_factory=FrozenParams)
    state_after: Mapping[str, Any] = field(default_factory=FrozenParams)

    def __post_init__(self) -> None:
        _require_text(self.segment_id, "ContinuityState.segment_id")
        object.__setattr__(self, "state_before", FrozenParams(self.state_before))
        object.__setattr__(self, "state_after", FrozenParams(self.state_after))

    @property
    def changed_keys(self) -> tuple[str, ...]:
        """Keys whose value differs between before and after, sorted. What
        actually happens in this segment, as continuity sees it."""
        keys = set(self.state_before) | set(self.state_after)
        _missing = object()
        return tuple(sorted(
            k for k in keys
            if self.state_before.get(k, _missing) != self.state_after.get(k, _missing)))

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id,
                "state_before": FrozenParams(self.state_before).to_dict(),
                "state_after": FrozenParams(self.state_after).to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContinuityState":
        return cls(segment_id=d["segment_id"],
                   state_before=FrozenParams(d.get("state_before") or {}),
                   state_after=FrozenParams(d.get("state_after") or {}))


@dataclass(frozen=True, slots=True)
class ContinuityBible(ContentAddressed):
    """Doc Stage 7 — the script breakdown, as a minimal typed shell.

    ``entries`` is the per-segment before/after log; the four reference tuples
    are the breakdown's standing inventories (characters, wardrobe, props,
    locations) that every segment draws from. ``notes`` carries the free text
    Stage 7 also asks for (time, weather, lighting direction, screen direction)
    until k110 gives each of those a field it has earned.

    Kept deliberately thin: this task owns the LOCK, not the writing. Every
    field here is something ``ProductionLock.lock()`` or the segment compiler
    actually reads. Nothing is present "for later"."""

    entries: tuple[ContinuityState, ...] = ()
    characters: tuple[str, ...] = ()
    wardrobe: tuple[str, ...] = ()
    props: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        for entry in self.entries:
            if not isinstance(entry, ContinuityState):
                raise TypeError(f"ContinuityBible.entries takes ContinuityState, "
                                f"got {type(entry).__name__}")
        ids = [e.segment_id for e in self.entries]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(
                f"ContinuityBible has two continuity states for segment(s) "
                f"{duplicates} — a segment has exactly one before and one after")
        for name in ("characters", "wardrobe", "props", "locations"):
            object.__setattr__(self, name,
                               _str_tuple(getattr(self, name),
                                          f"ContinuityBible.{name}"))
        object.__setattr__(self, "notes", str(self.notes or ""))

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return tuple(e.segment_id for e in self.entries)

    def state(self, segment_id: str) -> ContinuityState:
        for entry in self.entries:
            if entry.segment_id == segment_id:
                return entry
        raise KeyError(f"no continuity state for segment {segment_id!r}")

    def missing(self, segment_ids: Iterable[str]) -> tuple[str, ...]:
        """The requested segments this bible has no state for, in the order
        asked. Empty means Stage 7's "for every segment" holds."""
        known = set(self.segment_ids)
        return tuple(s for s in dict.fromkeys(segment_ids) if s not in known)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries],
                "characters": list(self.characters),
                "wardrobe": list(self.wardrobe),
                "props": list(self.props),
                "locations": list(self.locations),
                "notes": self.notes}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContinuityBible":
        return cls(entries=tuple(ContinuityState.from_dict(e)
                                 for e in d.get("entries", ())),
                   characters=tuple(d.get("characters", ())),
                   wardrobe=tuple(d.get("wardrobe", ())),
                   props=tuple(d.get("props", ())),
                   locations=tuple(d.get("locations", ())),
                   notes=d.get("notes", ""))


# ---------------------------------------------------------------------------
# Stage 9 — the shot plan
# ---------------------------------------------------------------------------

# MIRRORS ``video_intel.identity_profiles.SEMANTIC_VIEWS`` — keep in sync
# (``tests/test_oracle_production.py`` asserts the two agree, and that test is
# the only thing in the oracle test suite that imports the studio side).
# Mirrored rather than imported because importing ``video_intel`` builds the
# whole model registry at import time (~2 s + log chatter); a typed contract
# has to stay importable from a route, a test, or the agent without that.
CAMERA_VIEWS: frozenset[str] = frozenset({
    "front", "back", "back-left", "back-right", "left-profile", "right-profile",
    "three-quarter-left", "three-quarter-right",
})

#: Doc Stage 9's "camera placement, lens, framing, and movement" as a closed
#: set of KEYS. An unknown key is refused rather than ignored: a shot plan whose
#: ``"lense_mm"`` typo silently vanished is worse than one that would not build.
CAMERA_KEYS: frozenset[str] = frozenset({
    "view",        # a CAMERA_VIEWS name — the identity-bank turntable view
    "shot_size",   # a SHOT_SIZES name
    "movement",    # a CAMERA_MOVES name
    "lens_mm",     # focal length, a number
    "angle",       # eye/high/low/dutch/overhead — free text, Stage 9 "placement"
    "height",      # camera height note — free text
    "framing",     # free text: what is in frame and where
    "eyeline",     # Stage 9's "blocking and eyelines" — free text
    "focus",       # focus intent / pulls — free text
})

SHOT_SIZES: frozenset[str] = frozenset({
    "extreme_wide", "wide", "full", "medium_wide", "medium", "medium_close",
    "close", "extreme_close", "insert", "two_shot", "over_shoulder",
})

CAMERA_MOVES: frozenset[str] = frozenset({
    "static", "pan", "tilt", "dolly_in", "dolly_out", "track", "crane",
    "handheld", "steadicam", "zoom_in", "zoom_out", "orbit", "push_in",
    "pull_out",
})


def camera_view_from_prompt(prompt: str) -> str | None:
    """Derive a ``CAMERA_VIEWS`` name from prompt text, or ``None``.

    A thin, LAZY wrapper over ``video_intel.shot_intent.derive_view_from_prompt``
    (the deterministic keyword pass — no LLM). The import happens inside the
    call so that merely importing this contract never drags the studio registry
    in; a caller that has no studio side installed gets ``None`` rather than an
    ImportError, because "no derivable view" is already this function's honest
    answer for an unrecognizable prompt."""
    try:
        from hugpy_video.intel.shot_intent import derive_view_from_prompt
    except Exception:       # pragma: no cover - depends on install layout
        return None
    view = derive_view_from_prompt(prompt)
    return view if view in CAMERA_VIEWS else None


@dataclass(frozen=True, slots=True)
class ShotPlanEntry(ContentAddressed):
    """One planned shot: doc Stage 9's block / light / frame decisions for one
    segment, plus the acceptance rubric that shot will be judged against.

    ``line_ids`` names the dialogue this shot covers — possibly none (an
    insert, a reaction, a landscape) and possibly several (one shot holding two
    lines). It is the join back to the ``AudioMaster``, which is why the lock
    checks it against the master rather than trusting it.

    ``start_s``/``end_s`` are positions on the AUDIO MASTER's timeline, not
    clip-local times. Stage 8 is the reason: the audio is authoritative and the
    shot is cut to it, so a shot's coordinates are the audio's coordinates.

    ``rubric`` is Stage 9's "per-shot acceptance rubrics" — free text, one
    criterion per entry. It rides all the way into the ``PlanGraph`` as
    ``AcceptanceTest``s, so a shot nobody wrote a rubric for cannot be judged;
    ``SegmentSpec`` therefore requires at least one."""

    segment_id: str
    line_ids: tuple[str, ...] = ()
    start_s: float = 0.0
    end_s: float = 0.0
    camera: Mapping[str, Any] = field(default_factory=FrozenParams)
    blocking: str | None = None
    lighting: str | None = None
    rubric: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.segment_id, "ShotPlanEntry.segment_id")
        object.__setattr__(self, "line_ids",
                           _str_tuple(self.line_ids, "ShotPlanEntry.line_ids"))
        object.__setattr__(self, "start_s",
                           _q(_require_non_negative(self.start_s,
                                                    "ShotPlanEntry.start_s")))
        object.__setattr__(self, "end_s", _q(self.end_s))
        if self.end_s < self.start_s:
            raise ValueError(
                f"ShotPlanEntry({self.segment_id}) ends before it starts: "
                f"{self.end_s} < {self.start_s}")
        object.__setattr__(self, "camera", FrozenParams(self.camera))
        unknown = sorted(set(self.camera) - CAMERA_KEYS)
        if unknown:
            raise ValueError(
                f"ShotPlanEntry({self.segment_id}).camera has unknown key(s) "
                f"{unknown}; the Stage 9 camera vocabulary is "
                f"{sorted(CAMERA_KEYS)} — a key nobody reads is a silently "
                f"dropped direction")
        _closed = (("view", CAMERA_VIEWS), ("shot_size", SHOT_SIZES),
                   ("movement", CAMERA_MOVES))
        for key, vocabulary in _closed:
            value = self.camera.get(key)
            if value is None:
                continue
            if value not in vocabulary:
                raise ValueError(
                    f"ShotPlanEntry({self.segment_id}).camera[{key!r}]="
                    f"{value!r} is not in the vocabulary "
                    f"{sorted(vocabulary)}")
        lens = self.camera.get("lens_mm")
        if lens is not None and (isinstance(lens, bool)
                                 or not isinstance(lens, (int, float))
                                 or lens <= 0):
            raise ValueError(f"ShotPlanEntry({self.segment_id}).camera['lens_mm'] "
                             f"must be a positive number, got {lens!r}")
        for name in ("blocking", "lighting"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name,
                                   _require_text(value,
                                                 f"ShotPlanEntry.{name}"))
        object.__setattr__(self, "rubric",
                           _str_tuple(self.rubric, "ShotPlanEntry.rubric"))

    @property
    def duration_s(self) -> float:
        return _q(self.end_s - self.start_s)

    @property
    def window(self) -> tuple[float, float, tuple[str, ...]]:
        """This entry as an audio window — the exact shape
        ``shot_windows_from_audio`` emits and ``SegmentSpec`` carries."""
        return (self.start_s, self.end_s, self.line_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "line_ids": list(self.line_ids),
                "start_s": self.start_s, "end_s": self.end_s,
                "camera": FrozenParams(self.camera).to_dict(),
                "blocking": self.blocking, "lighting": self.lighting,
                "rubric": list(self.rubric)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ShotPlanEntry":
        return cls(segment_id=d["segment_id"],
                   line_ids=tuple(d.get("line_ids", ())),
                   start_s=d.get("start_s", 0.0), end_s=d.get("end_s", 0.0),
                   camera=FrozenParams(d.get("camera") or {}),
                   blocking=d.get("blocking"), lighting=d.get("lighting"),
                   rubric=tuple(d.get("rubric", ())))


@dataclass(frozen=True, slots=True)
class ShotPlan(ContentAddressed):
    """The ordered shot list (Stage 9), one entry per segment.

    Entries must be unique by ``segment_id`` and non-decreasing in ``start_s``:
    a shot list out of timeline order is a bug in whatever produced it, not a
    sorting problem for whoever reads it. OVERLAP is deliberately NOT refused
    here — two entries covering the same seconds is a legitimate artifact (two
    angles on one line) — but ``ProductionLock.lock()`` refuses it, because a
    locked production timeline has to partition the audio it is cut to."""

    entries: tuple[ShotPlanEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        for entry in self.entries:
            if not isinstance(entry, ShotPlanEntry):
                raise TypeError(f"ShotPlan.entries takes ShotPlanEntry, got "
                                f"{type(entry).__name__}")
        ids = [e.segment_id for e in self.entries]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"ShotPlan has two entries for segment(s) "
                             f"{duplicates}")
        for previous, entry in zip(self.entries, self.entries[1:]):
            if entry.start_s + _EPS < previous.start_s:
                raise ValueError(
                    f"ShotPlan is out of timeline order: {entry.segment_id} "
                    f"starts at {entry.start_s} after {previous.segment_id} "
                    f"started at {previous.start_s}")

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return tuple(e.segment_id for e in self.entries)

    @property
    def line_ids(self) -> tuple[str, ...]:
        """Every dialogue line this plan covers, in timeline order, deduped."""
        out: list[str] = []
        for entry in self.entries:
            for line_id in entry.line_ids:
                if line_id not in out:
                    out.append(line_id)
        return tuple(out)

    @property
    def total_seconds(self) -> float:
        return _q(max((e.end_s for e in self.entries), default=0.0))

    def entry(self, segment_id: str) -> ShotPlanEntry:
        for entry in self.entries:
            if entry.segment_id == segment_id:
                return entry
        raise KeyError(f"no shot plan entry for segment {segment_id!r}")

    def overlaps(self) -> tuple[tuple[str, str], ...]:
        """``((earlier, later), …)`` for every pair of entries whose windows
        intersect. Empty means the plan partitions the timeline."""
        out: list[tuple[str, str]] = []
        for i, first in enumerate(self.entries):
            for second in self.entries[i + 1:]:
                if second.start_s + _EPS < first.end_s and \
                        first.start_s + _EPS < second.end_s:
                    out.append((first.segment_id, second.segment_id))
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ShotPlan":
        return cls(entries=tuple(ShotPlanEntry.from_dict(e)
                                 for e in d.get("entries", ())))


# ---------------------------------------------------------------------------
# Stage 11 — the production lock
# ---------------------------------------------------------------------------

#: The keys ``revise()`` will accept as a material change. Anything else is a
#: different lock, not a revision of this one.
REVISABLE_FIELDS: frozenset[str] = frozenset({
    "snapshot_digest", "screenplay_digest", "continuity_digest",
    "audio_master_digest", "shot_plan_digest", "storyboard_digest",
    "identity_refs", "registry_version", "locked_at",
})


@dataclass(frozen=True, slots=True)
class ProductionLock(ContentAddressed):
    """Doc Stage 11 — one versioned digest over everything a segment may read.

    The lock stores DIGESTS, not the artifacts. Three reasons, all load-bearing:
    it stays small enough to put in every receipt; it can be compared against a
    re-derived digest to catch a swapped artifact (which is exactly what
    ``compile_segments`` does before it writes a single prompt); and it makes
    ``SegmentSpec.parents`` a list of content addresses rather than a graph of
    object references that could accidentally include a sibling.

    ``screenplay_digest`` is ``str | None`` because a screenplay is k110's
    artifact and is not required to exist for the first audio-first slice — a
    three-line performance piece is locked against dialogue, continuity, audio
    and shots. ``None`` says "there is no screenplay", never "there is one and
    we did not record it": :meth:`parent_digests` simply omits it.

    ``revision`` / ``parent_revision`` / ``revision_reason`` are Stage 10's
    post-lock rule. ``revision_reason`` is an addition to the doc's field list,
    matching ``PlanGraph.revision_reason`` for the same reason k103 added it:
    ``revise()`` refuses an unexplained change.

    ``storyboard_digest`` (k124) is ``str | None`` for exactly k110's reason
    about ``screenplay_digest``: a production may lock without a storyboard,
    and ``None`` says "there is none", never "there is one and we did not
    record it". It is the LAST field on purpose (positional construction is
    unchanged) and — uniquely here — it is OMITTED FROM :meth:`to_dict` when it
    is None, so every lock digest ever computed before k124 is byte-identical
    afterwards. That matters because lock digests are journalled: k106's resume
    and k114's run state compare a rehydrated artifact against the lock's own
    recorded digest, and a lock whose id moved because a field was added would
    invalidate every run on disk for a field none of them uses."""

    snapshot_digest: str
    screenplay_digest: str | None
    continuity_digest: str
    audio_master_digest: str
    shot_plan_digest: str
    identity_refs: tuple[str, ...] = ()
    registry_version: str | None = None
    locked_at: str = ""
    revision: int = 0
    revision_reason: str = ""
    parent_revision: int | None = None
    storyboard_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("snapshot_digest", "continuity_digest",
                     "audio_master_digest", "shot_plan_digest"):
            _require_text(getattr(self, name), f"ProductionLock.{name}")
        if self.screenplay_digest is not None:
            object.__setattr__(self, "screenplay_digest",
                               _require_text(self.screenplay_digest,
                                             "ProductionLock.screenplay_digest"))
        if self.storyboard_digest is not None:
            object.__setattr__(self, "storyboard_digest",
                               _require_text(self.storyboard_digest,
                                             "ProductionLock.storyboard_digest"))
        object.__setattr__(self, "identity_refs",
                           _str_tuple(self.identity_refs,
                                      "ProductionLock.identity_refs"))
        if self.registry_version is not None:
            object.__setattr__(self, "registry_version",
                               _require_text(self.registry_version,
                                             "ProductionLock.registry_version"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) \
                or self.revision < 0:
            raise ValueError(f"ProductionLock.revision must be a non-negative "
                             f"int, got {self.revision!r}")
        if self.parent_revision is not None:
            if self.parent_revision < 0:
                raise ValueError(f"ProductionLock.parent_revision must be >= 0, "
                                 f"got {self.parent_revision}")
            if self.parent_revision >= self.revision:
                raise ValueError(
                    f"ProductionLock.parent_revision {self.parent_revision} "
                    f"must precede revision {self.revision}")
        if self.revision > 0 and not str(self.revision_reason).strip():
            raise ValueError(
                f"ProductionLock revision {self.revision} carries no reason — "
                f"a post-lock material change with nothing written down is an "
                f"unaudited production change (Stage 10)")
        object.__setattr__(self, "revision_reason", str(self.revision_reason or ""))

    # -- the transition ----------------------------------------------------

    @classmethod
    def lock(cls, snapshot: GenerationSnapshot, *,
             audio_master: AudioMaster,
             continuity: ContinuityBible,
             shot_plan: ShotPlan,
             screenplay_digest: str | None = None,
             storyboard_digest: str | None = None,
             identity_refs: Sequence[str] | None = None,
             registry_version: str | None = None,
             locked_at: str = "",
             run_prompts: "RunPromptLedger | Iterable[str] | None" = None,
             ) -> "ProductionLock":
        """Lock a production, or refuse with a reason.

        The refusals, each of them a real failure mode of this pipeline:

        1. Wrong types — caller error, raised at the boundary.
        2. ``audio_master.locked`` is False. Stage 8 puts the definitive audio
           BEFORE final shot timing; locking shots against a draft timeline is
           the exact mistake the doc names, and the fix is one call: ``.lock()``.
        3. An empty shot plan — nothing to compile.
        4. A shot naming a line the audio master does not have (a stale plan),
           or a spoken line no shot covers (a line that would silently vanish
           from the picture while still playing on the track).
        5. Overlapping shot windows — a locked timeline must partition its audio.
        6. A shot window running past the end of the audio master.
        7. A segment with no continuity state (Stage 7: "for EVERY segment").
        8. An identity ref the immutable snapshot never carried — invariant 1
           says the snapshot is the source of truth about what was supplied.
        9. Two disagreeing registry versions in one lock (k105).
        10. With ``run_prompts`` supplied: a snapshot prompt that was minted
            during the run (invariant 9 / Stage 4).
        """
        if not isinstance(snapshot, GenerationSnapshot):
            raise TypeError(f"lock() takes a GenerationSnapshot, got "
                            f"{type(snapshot).__name__}")
        if not isinstance(audio_master, AudioMaster):
            raise TypeError(f"lock(audio_master=) takes an AudioMaster, got "
                            f"{type(audio_master).__name__}")
        if not isinstance(continuity, ContinuityBible):
            raise TypeError(f"lock(continuity=) takes a ContinuityBible, got "
                            f"{type(continuity).__name__}")
        if not isinstance(shot_plan, ShotPlan):
            raise TypeError(f"lock(shot_plan=) takes a ShotPlan, got "
                            f"{type(shot_plan).__name__}")

        if not audio_master.locked:
            raise LockRefused(
                "the audio master is not locked: doc Stage 8 puts the "
                "definitive audio timeline BEFORE final shot timing, so a "
                "production cannot lock against a draft one. Call "
                "AudioMaster.lock() on the accepted master first")

        if not shot_plan.entries:
            raise LockRefused("the shot plan is empty — there is nothing to lock")

        master_lines = set(audio_master.line_ids)
        planned_lines: list[str] = []
        for entry in shot_plan.entries:
            unknown = [l for l in entry.line_ids if l not in master_lines]
            if unknown:
                raise LockRefused(
                    f"shot {entry.segment_id} names line(s) {unknown} that the "
                    f"audio master does not carry (has: "
                    f"{sorted(master_lines)}) — the shot plan is stale")
            planned_lines.extend(entry.line_ids)
        uncovered = [l for l in audio_master.line_ids if l not in set(planned_lines)]
        if uncovered:
            raise LockRefused(
                f"line(s) {uncovered} are spoken in the audio master but no "
                f"shot covers them — locking here would put dialogue on the "
                f"track with no picture behind it")

        overlaps = shot_plan.overlaps()
        if overlaps:
            raise LockRefused(
                f"shot windows overlap: {list(overlaps)} — a locked production "
                f"timeline must partition the audio it is cut to")

        overrun = [e.segment_id for e in shot_plan.entries
                   if e.end_s > audio_master.total_seconds + _EPS]
        if overrun:
            raise LockRefused(
                f"shot(s) {overrun} end after the audio master does "
                f"({audio_master.total_seconds}s) — shots are cut TO the audio, "
                f"never past it")

        missing = continuity.missing(shot_plan.segment_ids)
        if missing:
            raise LockRefused(
                f"the continuity bible has no state for segment(s) "
                f"{list(missing)}; Stage 7 requires an explicit state_before "
                f"and state_after for EVERY segment")

        refs = (_str_tuple(identity_refs, "lock(identity_refs=)")
                if identity_refs is not None else snapshot.identity_refs)
        unsnapshotted = [r for r in refs if r not in snapshot.identity_refs]
        if unsnapshotted:
            raise LockRefused(
                f"identity ref(s) {unsnapshotted} are not in the generation "
                f"snapshot (has: {list(snapshot.identity_refs)}) — the "
                f"immutable snapshot is what was supplied, and a lock cannot "
                f"introduce an identity it never saw")

        version = registry_version if registry_version is not None \
            else snapshot.registry_version
        if audio_master.registry_version is not None:
            if version is None:
                version = audio_master.registry_version
            elif version != audio_master.registry_version:
                raise LockRefused(
                    f"two routing-registry versions in one lock: "
                    f"{version!r} vs the audio master's "
                    f"{audio_master.registry_version!r} — Stage 11 locks ONE "
                    f"accepted registry version")

        if run_prompts is not None:
            snapshot.assert_pre_run(run_prompts)

        return cls(snapshot_digest=snapshot.digest,
                   screenplay_digest=screenplay_digest,
                   continuity_digest=continuity.digest,
                   audio_master_digest=audio_master.digest,
                   shot_plan_digest=shot_plan.digest,
                   identity_refs=refs,
                   registry_version=version,
                   locked_at=locked_at,
                   storyboard_digest=storyboard_digest)

    def revise(self, reason: str, **changes: Any) -> "ProductionLock":
        """Stage 10: a post-lock material change is a NEW revision, never an
        edit. Returns revision N+1 with ``parent_revision`` set to N.

        ``changes`` may name any of :data:`REVISABLE_FIELDS`; anything else is
        a different lock, not a revision of this one, and is refused by name."""
        text = str(reason or "").strip()
        if not text:
            raise LockRefused(
                "revise() needs a reason: a post-lock material change with "
                "nothing written down is an unaudited production change "
                "(Stage 10)")
        unknown = sorted(set(changes) - REVISABLE_FIELDS)
        if unknown:
            raise LockRefused(
                f"revise() cannot change {unknown}; revisable fields are "
                f"{sorted(REVISABLE_FIELDS)} (revision, parent_revision and "
                f"the reason are set by revise() itself)")
        if "identity_refs" in changes:
            changes["identity_refs"] = _str_tuple(changes["identity_refs"],
                                                  "revise(identity_refs=)")
        return replace(self, revision=self.revision + 1,
                       parent_revision=self.revision, revision_reason=text,
                       **changes)

    # -- reading -----------------------------------------------------------

    @property
    def parent_digests(self) -> tuple[str, ...]:
        """Every digest a ``SegmentSpec`` may legally name as a parent: this
        lock and the artifacts it locked — and nothing else.

        This is the whitelist that makes invariant 9 checkable rather than
        merely intended. ``assert_siblings`` compares each spec's ``parents``
        against it, so a compiler that ever put a sibling's digest in there
        fails a test instead of shipping a chain. The lock's OWN digest leads
        the tuple because it is the single artifact that names all the others:
        a receipt carrying it can re-derive the rest."""
        out = [self.digest, self.snapshot_digest, self.continuity_digest,
               self.audio_master_digest, self.shot_plan_digest]
        if self.screenplay_digest:
            out.append(self.screenplay_digest)
        if self.storyboard_digest:
            out.append(self.storyboard_digest)
        return tuple(dict.fromkeys(out))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "snapshot_digest": self.snapshot_digest,
            "screenplay_digest": self.screenplay_digest,
            "continuity_digest": self.continuity_digest,
            "audio_master_digest": self.audio_master_digest,
            "shot_plan_digest": self.shot_plan_digest,
            "identity_refs": list(self.identity_refs),
            "registry_version": self.registry_version,
            "locked_at": self.locked_at,
            "revision": self.revision,
            "revision_reason": self.revision_reason,
            "parent_revision": self.parent_revision,
        }
        # k124: present only when there IS a storyboard. See the class
        # docstring — this is what keeps every pre-k124 lock digest, including
        # the ones already journalled on disk, byte-identical.
        if self.storyboard_digest is not None:
            payload["storyboard_digest"] = self.storyboard_digest
        return payload

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProductionLock":
        return cls(
            snapshot_digest=d["snapshot_digest"],
            screenplay_digest=d.get("screenplay_digest"),
            continuity_digest=d["continuity_digest"],
            audio_master_digest=d["audio_master_digest"],
            shot_plan_digest=d["shot_plan_digest"],
            identity_refs=tuple(d.get("identity_refs", ())),
            registry_version=d.get("registry_version"),
            locked_at=d.get("locked_at", ""),
            revision=int(d.get("revision", 0)),
            revision_reason=d.get("revision_reason", ""),
            parent_revision=d.get("parent_revision"),
            storyboard_digest=d.get("storyboard_digest"),
        )


__all__ = [
    "CAMERA_KEYS",
    "CAMERA_MOVES",
    "CAMERA_VIEWS",
    "SHOT_SIZES",
    "REVISABLE_FIELDS",
    "ContentAddressed",
    "ContinuityBible",
    "ContinuityState",
    "GenerationSnapshot",
    "LockRefused",
    "ProductionError",
    "ProductionLock",
    "RunPromptLedger",
    "RunPromptRefused",
    "ShotPlan",
    "ShotPlanEntry",
    "camera_view_from_prompt",
    "canonical_json",
    "digest_payload",
    "prompt_digest",
]
