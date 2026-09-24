"""Audio-first timing artifacts + the TTS candidate fan-out (k102).

doc Stage 8, stated verbatim: *"Lock the dialogue, generate several authorized
speech candidates where required, evaluate them, and assemble ``AudioMaster``
with exact line, word, pause, and speaker timing. The definitive audio timeline
precedes final shot timing. Do not generate arbitrary clip durations and force
dialogue onto them afterward."*

This module is that sentence as code, and nothing else:

    DialogueTimeline  the LOCKED dialogue (Stage 8's "dialogue lock")
    VoiceProfile      who speaks it, and by what authority
    SpeechCandidate   one take of one line (N per line, never one)
    WordTiming /
    LineTiming        exact word, line and pause timing
    AudioMaster       the definitive audio timeline that shots are cut TO

    build_audio_master(...)  fan out N candidates per line, judge each one with
                             k98's speech checks, pick the best ACCEPTED take,
                             assemble the timeline — or return a typed gap.

WHY THIS IS A PURE ORCHESTRATOR. ``build_audio_master`` never imports a
backend, never touches a GPU, a worker, a socket or the disk. It takes three
injected callables (``synth`` / ``transcribe`` / ``similarity``) and composes
them. Two payoffs: (1) the whole audio-first slice is testable on this GPU-less
box against fakes, which is the only way the RULES get pinned down before the
hardware exists; (2) the k106 FAT orchestrator passes the REAL seams
(``video_intel.runners.tts_chatterbox.synthesize`` behind the media bus,
``audio.transcribe.word_timestamps`` through ``oracle.runtime``) without this
module changing a line. Nothing here knows which is which — the whole point.

INVARIANT 11 — THE GENERATOR IS NOT ITS OWN JUDGE. Word timings and line
presence come from the ROUND-TRIP transcript of the produced audio, never from
the synthesizer's own claim about what it said. A TTS backend that swallows a
word reports success; the ASR pass over its output does not. That is why
``transcribe`` is a required argument and not an optional refinement, and why
``SpeechCandidate.duration_s`` (measured on the artifact) is used for timing
while the LINE evidence is taken from the transcript.

AUTHORITY, THIRD LINE OF DEFENCE. k97 refuses an unauthorized voice route
before a model is picked; k98's runner refuses a reference-conditioned spec
before the backend is imported; here ``VoiceProfile(kind=REFERENCE)`` cannot be
CONSTRUCTED without ``authorized=True``, so no fan-out can even name an
unauthorized reference voice. The doc's fallback for the unauthorized case is
``VoiceKind.STYLE``: non-identifying delivery traits (cadence, pitch range,
energy) instead of a specific person's voice — describing a performance is not
cloning a human being, and the two are different TYPES here so they can never
be confused by a caller in a hurry.

NEVER SILENTLY ACCEPT. If no candidate for a line passes the judge, the builder
does not return the least-bad take. It returns an ``AudioGap`` carrying every
repair code seen (``LINE_OMITTED`` / ``VOICE_SIMILARITY_LOW`` /
``SHOT_TOO_SHORT`` / ``CAPABILITY_GAP``) plus the scorecard of every rejected
candidate, and ``AudioBuildResult.master`` stays None. Every line is judged even
after one fails, so the repair controller gets the COMPLETE picture in one pass
rather than discovering the next gap after each repair.

DETERMINISM. Every artifact is a frozen, slotted dataclass with canonical JSON
(sorted keys, no whitespace) and a ``digest`` property; candidate seeds are
derived from the line + voice digests, so the same inputs produce the same
seeds, the same takes and the same digests. No timestamps, no uuids, no
iteration-order dependence anywhere in this module.

No pathlib. No I/O at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from hugpy_oracle import speech
from hugpy_oracle.contracts import Check, CheckKind, RepairCode, Scorecard
# Same-package reuse on purpose: "what is this transcript word's text" must have
# exactly ONE definition, and it lives with the line check that depends on it.
from hugpy_oracle.speech import _word_text

# ---------------------------------------------------------------------------
# Tunables — named, not magic.
# ---------------------------------------------------------------------------

#: Seconds of silence held after a line unless the policy overrides it. 0.35 s
#: is the conversational beat between turns in scripted dialogue (long enough to
#: read as a breath, short enough not to read as a pause for effect). It is a
#: DEFAULT for pacing, never a measured fact about a performance.
DEFAULT_PAUSE_AFTER_S: float = 0.35

#: Candidates synthesized per line. doc Stage 16: "generate several short
#: candidate clips per shot rather than committing immediately to one long
#: result" — the same discipline one stage earlier, for speech.
DEFAULT_CANDIDATES: int = 3

#: Decimal places every second-valued field is quantized to before it enters an
#: artifact. Microseconds are far below any audible or measurable difference and
#: they make digests stable against float accumulation across platforms — a
#: timeline whose digest changes with the last bit of a float is not an id.
TIME_PRECISION: int = 6

#: Seeds are reduced mod 2**32 because every backend seeding API downstream
#: (numpy, torch) takes a uint32; a 64-bit seed would be silently truncated
#: somewhere out of sight, and a silently truncated seed is not determinism.
SEED_MODULUS: int = 2 ** 32

#: Float slack for "did this word run past the end of its own audio" tests.
_EPS: float = 1e-9

#: The check a policy adds when it REQUIRES similarity evidence. Not a k98
#: check: k98 asks "is the voice right", this asks "did anyone measure".
SIMILARITY_EVIDENCE_CHECK: str = "speech.similarity_evidence"

#: Check name -> repair code, extending k98's table with the one code this
#: module can produce that the pure checks cannot: a policy that demands
#: speaker-similarity evidence on a fleet with no embedding backend is a
#: CAPABILITY_GAP (the fleet's problem), never VOICE_SIMILARITY_LOW (which
#: would blame the artifact for a measurement nobody took).
AUDIO_REPAIR: dict[str, RepairCode] = {
    **speech.SPEECH_REPAIR,
    SIMILARITY_EVIDENCE_CHECK: RepairCode.CAPABILITY_GAP,
}

#: Repair-first order: a missing LINE invalidates the take; a wrong VOICE
#: invalidates the performance; an unmeasurable voice is a fleet gap; a take
#: that overruns its budget is a timing decision on otherwise good audio.
_REPAIR_PRIORITY: tuple[str, ...] = (
    "speech.lines_present",
    "speech.speaker_similarity",
    SIMILARITY_EVIDENCE_CHECK,
    "sync.duration_fit",
)


# ---------------------------------------------------------------------------
# Canonical JSON + digests — one definition, used by every artifact here.
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, ASCII-escaped.
    Same rule as ``mct.manifest._canonical_json`` — two stores, one encoding."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def digest_payload(payload: Any) -> str:
    """sha256 over :func:`canonical_json` of ``payload``."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def scorecard_digest(card: Scorecard) -> str:
    """The stable id of a ``Scorecard`` — what ``SpeechCandidate`` points at
    instead of embedding the whole card (the card is evidence and belongs in
    the gap/receipt; the candidate only needs to name it)."""
    return digest_payload(card.to_dict())


def _q(seconds: Any) -> float:
    """Quantize a second-value to TIME_PRECISION. ``+ 0.0`` normalizes -0.0,
    which JSON-encodes differently and would fork a digest for nothing."""
    return round(float(seconds), TIME_PRECISION) + 0.0


class _Artifact:
    """Canonical-JSON identity for every artifact in this module.

    ``__slots__ = ()`` so slotted dataclasses can inherit without growing a
    ``__dict__``; ``digest`` is a PROPERTY because an artifact's id is a
    function of its values, never a stored field that could drift from them."""

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


# ---------------------------------------------------------------------------
# Line + DialogueTimeline — the dialogue lock
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Line(_Artifact):
    """One locked line of dialogue.

    ``line_id`` is the stable handle every later artifact refers to (timings,
    tracks, segment specs, repair targets) — it survives re-synthesis, so a
    repaired take lands on the same line rather than creating a new one.
    ``emotion`` is a delivery note for the synthesizer and a semantic-evidence
    hint for the judge; it is descriptive, never a gate. ``max_seconds`` is the
    operator's declared budget for this line and feeds
    ``speech.check_duration_fit`` — a take that overruns it is reported as
    SHOT_TOO_SHORT because doc Stage 8 fixes the direction of that test: the
    audio is authoritative and the window gives way, never "speak faster"."""

    line_id: str
    speaker: str
    text: str
    emotion: str | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.line_id, "Line.line_id")
        _require_text(self.speaker, "Line.speaker")
        _require_text(self.text, "Line.text")
        if self.max_seconds is not None and float(self.max_seconds) <= 0:
            raise ValueError(
                f"Line({self.line_id}).max_seconds must be positive when set, "
                f"got {self.max_seconds}")

    def to_dict(self) -> dict[str, Any]:
        return {"line_id": self.line_id, "speaker": self.speaker,
                "text": self.text, "emotion": self.emotion,
                "max_seconds": self.max_seconds}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Line":
        return cls(line_id=d["line_id"], speaker=d["speaker"], text=d["text"],
                   emotion=d.get("emotion"), max_seconds=d.get("max_seconds"))


@dataclass(frozen=True, slots=True)
class DialogueTimeline(_Artifact):
    """The locked dialogue: an ORDERED sequence of lines with unique ids.

    Order is the contract — ``build_audio_master`` lays the audio out in it and
    k98's line check matches lines against the transcript with an advancing
    cursor, so a take that reorders lines fails even with every word present.

    ``locked`` is the Stage 8 gate. ``build_audio_master`` REFUSES an unlocked
    timeline: generating definitive audio for dialogue that may still change is
    how a pipeline ends up retiming shots to a script revision nobody recorded.
    Lock it (``.lock()``) when the dialogue is final; the digest of the locked
    timeline is what ``AudioMaster.timeline_digest`` points back at."""

    lines: tuple[Line, ...]
    locked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))
        if not self.lines:
            raise ValueError("DialogueTimeline needs at least one line — an "
                             "empty dialogue lock is not a lock")
        for line in self.lines:
            if not isinstance(line, Line):
                raise TypeError(f"DialogueTimeline.lines takes Line, got "
                                f"{type(line).__name__}")
        ids = [ln.line_id for ln in self.lines]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate line_id(s): {', '.join(dupes)}")

    # --- reading -----------------------------------------------------------
    @property
    def line_ids(self) -> tuple[str, ...]:
        return tuple(ln.line_id for ln in self.lines)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(ln.text for ln in self.lines)

    @property
    def speakers(self) -> tuple[str, ...]:
        """Distinct speakers in first-appearance order (the casting table's
        key set — deterministic, never a set's iteration order)."""
        seen: list[str] = []
        for line in self.lines:
            if line.speaker not in seen:
                seen.append(line.speaker)
        return tuple(seen)

    def line(self, line_id: str) -> Line:
        for candidate in self.lines:
            if candidate.line_id == line_id:
                return candidate
        raise KeyError(f"unknown line_id: {line_id!r}")

    def lock(self) -> "DialogueTimeline":
        """The locked twin of this timeline (self if already locked)."""
        return self if self.locked else replace(self, locked=True)

    def to_dict(self) -> dict[str, Any]:
        return {"lines": [ln.to_dict() for ln in self.lines],
                "locked": self.locked}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DialogueTimeline":
        return cls(lines=tuple(Line.from_dict(x) for x in d.get("lines", ())),
                   locked=bool(d.get("locked", False)))


# ---------------------------------------------------------------------------
# VoiceProfile — who speaks, and by what authority
# ---------------------------------------------------------------------------


class VoiceKind(str, Enum):
    """The three honest ways to answer "whose voice is this?".

    REFERENCE  a specific person's voice, cloned from an authorized reference.
               Requires ``authorized=True`` AT CONSTRUCTION — see below.
    SYNTHETIC  the model's own default voice. Belongs to nobody, needs no
               release, carries no reference.
    STYLE      the doc's UNAUTHORIZED FALLBACK: non-identifying delivery traits
               (cadence, pitch range, energy) instead of an identity. "A gravelly
               low voice, slow cadence" is a direction; it is not a person."""

    REFERENCE = "reference"
    SYNTHETIC = "synthetic"
    STYLE = "style"


#: The trait vocabulary a STYLE voice is expected to speak in. Not a closed set
#: (a backend may accept more), but a STYLE profile must use at least one of
#: these or it is not describing a delivery — it is probably naming a person,
#: which is exactly what this kind exists to avoid.
STYLE_TRAITS: tuple[str, ...] = (
    "cadence", "pitch_range", "energy", "accent", "tempo", "timbre", "age_range")

_JSON_SCALARS = (str, int, float, bool)


@dataclass(frozen=True, slots=True)
class VoiceProfile(_Artifact):
    """The voice a line is cast to.

    THE GATE: ``kind=REFERENCE`` REQUIRES ``authorized=True`` and a
    ``reference_ref``. Cloning a specific person's voice is a rights decision,
    not a rendering parameter (k97 types it, k98's runner re-checks it, and here
    it is unrepresentable) — an unauthorized reference voice cannot be
    CONSTRUCTED, so no fan-out, test, or future orchestrator can pass one down.

    THE MIRROR RULE: a non-REFERENCE profile may NOT carry a ``reference_ref``.
    Otherwise "kind=synthetic, reference_ref=/voices/mira.wav" would launder a
    clone past the gate by relabelling it, and the type system would help.

    ``style`` is stored as canonical sorted ``(key, json-text)`` pairs so the
    dataclass stays frozen, hashable and digest-stable; pass a plain mapping and
    read it back with :meth:`style_dict`. Values must be JSON scalars — a style
    note is a description, not a payload."""

    voice_id: str
    kind: VoiceKind = VoiceKind.SYNTHETIC
    reference_ref: str | None = None
    authorized: bool = False
    style: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.voice_id, "VoiceProfile.voice_id")
        object.__setattr__(self, "kind", VoiceKind(self.kind))
        object.__setattr__(self, "style", self.normalize_style(self.style))
        ref = (self.reference_ref or "").strip() or None
        object.__setattr__(self, "reference_ref", ref)
        object.__setattr__(self, "authorized", bool(self.authorized))

        if self.kind is VoiceKind.REFERENCE:
            if not self.authorized:
                raise ValueError(
                    f"VoiceProfile({self.voice_id!r}, kind=reference) requires "
                    f"authorized=True: reproducing a specific person's voice is "
                    f"a rights decision with evidence behind it (k97's "
                    f"AuthorityKind.VOICE), never a default. Use "
                    f"kind='style' with non-identifying traits when no "
                    f"authorization exists — a silent lookalike is worse than "
                    f"a refusal")
            if not ref:
                raise ValueError(
                    f"VoiceProfile({self.voice_id!r}, kind=reference) needs a "
                    f"reference_ref — a reference voice with no reference is "
                    f"not a reference voice")
        elif ref:
            raise ValueError(
                f"VoiceProfile({self.voice_id!r}, kind={self.kind.value}) must "
                f"not carry reference_ref={ref!r}: relabelling a clone as "
                f"'{self.kind.value}' would route it around the authorization "
                f"gate. Declare kind='reference' with authorization, or drop "
                f"the reference")

        if self.kind is VoiceKind.STYLE:
            keys = {k for k, _ in self.style}
            if not keys & set(STYLE_TRAITS):
                raise ValueError(
                    f"VoiceProfile({self.voice_id!r}, kind=style) must describe "
                    f"a delivery with at least one of {', '.join(STYLE_TRAITS)}; "
                    f"got {sorted(keys) or 'nothing'}")

    # --- style normalization ----------------------------------------------
    @staticmethod
    def normalize_style(style: Any) -> tuple[tuple[str, str], ...]:
        """Mapping -> sorted ``(key, json-text)`` pairs (the same trick as
        ``ExecutionReceipt.normalize_request`` / ``mct.normalize_parameters``).
        An already-normalized pair sequence passes through, but its values must
        BE valid JSON text — pass a mapping if you have raw Python values."""
        if not style:
            return ()
        if isinstance(style, Mapping):
            return tuple(sorted(
                (str(k), json.dumps(style[k], sort_keys=True,
                                    separators=(",", ":")))
                for k in style
                if _is_json_scalar(style[k], str(k))))
        out: list[tuple[str, str]] = []
        for key, value in style:
            try:
                json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"VoiceProfile.style pair {key!r} carries non-JSON text "
                    f"{value!r} — pass a mapping to encode raw values") from exc
            out.append((str(key), str(value)))
        return tuple(sorted(out))

    def style_dict(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.style}

    # --- constructors for the two safe kinds ------------------------------
    @classmethod
    def synthetic(cls, voice_id: str, **style: Any) -> "VoiceProfile":
        """The model's own voice. Nobody's likeness, no release needed."""
        return cls(voice_id=voice_id, kind=VoiceKind.SYNTHETIC, style=style)

    @classmethod
    def style_fallback(cls, voice_id: str, **traits: Any) -> "VoiceProfile":
        """The doc's fallback when a reference voice is not authorized: cast the
        DELIVERY, not the person. ``VoiceProfile.style_fallback('narrator',
        cadence='slow', energy='low')``."""
        return cls(voice_id=voice_id, kind=VoiceKind.STYLE, style=traits)

    def to_dict(self) -> dict[str, Any]:
        return {"voice_id": self.voice_id, "kind": self.kind.value,
                "reference_ref": self.reference_ref,
                "authorized": self.authorized, "style": self.style_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VoiceProfile":
        return cls(voice_id=d["voice_id"],
                   kind=VoiceKind(d.get("kind", VoiceKind.SYNTHETIC.value)),
                   reference_ref=d.get("reference_ref"),
                   authorized=bool(d.get("authorized", False)),
                   style=d.get("style") or {})


def _is_json_scalar(value: Any, key: str) -> bool:
    if value is None or isinstance(value, _JSON_SCALARS):
        return True
    raise ValueError(
        f"VoiceProfile.style[{key!r}] must be a JSON scalar (a delivery note), "
        f"got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Timings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WordTiming(_Artifact):
    """One word of the ROUND-TRIP transcript, placed on the master timeline.

    These come from an ASR pass over the produced audio — never from the
    synthesizer (invariant 11). ``prob`` is the recognizer's confidence when it
    reports one; None means unreported, never 1.0."""

    word: str
    start_s: float
    end_s: float
    prob: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.word, "WordTiming.word")
        object.__setattr__(self, "start_s",
                           _require_non_negative(self.start_s,
                                                 "WordTiming.start_s"))
        object.__setattr__(self, "end_s", float(self.end_s))
        if self.end_s < self.start_s:
            raise ValueError(
                f"WordTiming({self.word!r}) ends before it starts: "
                f"{self.end_s} < {self.start_s}")
        if self.prob is not None:
            prob = float(self.prob)
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"WordTiming.prob must be in [0, 1], got {prob}")
            object.__setattr__(self, "prob", prob)

    @property
    def duration_s(self) -> float:
        return _q(self.end_s - self.start_s)

    def shifted(self, offset: float) -> "WordTiming":
        """This word moved onto the master timeline by ``offset`` seconds."""
        return WordTiming(word=self.word, start_s=_q(self.start_s + offset),
                          end_s=_q(self.end_s + offset), prob=self.prob)

    def to_dict(self) -> dict[str, Any]:
        return {"word": self.word, "start_s": self.start_s,
                "end_s": self.end_s, "prob": self.prob}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WordTiming":
        return cls(word=d["word"], start_s=d["start_s"], end_s=d["end_s"],
                   prob=d.get("prob"))


@dataclass(frozen=True, slots=True)
class LineTiming(_Artifact):
    """Where one line sits on the master timeline, with its words and the pause
    held after it.

    ``words`` may be EMPTY and that is not a failure: this fleet's whisper path
    currently returns no per-word times (see the k98 dispatch record), so a
    timeline built today has exact LINE timing from measured durations and no
    word timing. Empty says "not measured"; it never fabricates evenly-spaced
    words, which would look like evidence and be a lie.

    ``pause_after_s`` belongs to the LINE, not to the gap between lines, so a
    single-line repair can change its own trailing beat without renegotiating
    the neighbour's start."""

    line_id: str
    start_s: float
    end_s: float
    words: tuple[WordTiming, ...] = ()
    pause_after_s: float = 0.0

    def __post_init__(self) -> None:
        _require_text(self.line_id, "LineTiming.line_id")
        object.__setattr__(self, "start_s",
                           _require_non_negative(self.start_s,
                                                 "LineTiming.start_s"))
        object.__setattr__(self, "end_s", float(self.end_s))
        if self.end_s < self.start_s:
            raise ValueError(
                f"LineTiming({self.line_id}) ends before it starts: "
                f"{self.end_s} < {self.start_s}")
        object.__setattr__(self, "pause_after_s",
                           _require_non_negative(self.pause_after_s,
                                                 "LineTiming.pause_after_s"))
        object.__setattr__(self, "words", tuple(self.words))
        for word in self.words:
            if not isinstance(word, WordTiming):
                raise TypeError(f"LineTiming.words takes WordTiming, got "
                                f"{type(word).__name__}")

    @property
    def duration_s(self) -> float:
        return _q(self.end_s - self.start_s)

    @property
    def next_start_s(self) -> float:
        """Where the following line begins: this line's end plus its pause."""
        return _q(self.end_s + self.pause_after_s)

    def to_dict(self) -> dict[str, Any]:
        return {"line_id": self.line_id, "start_s": self.start_s,
                "end_s": self.end_s,
                "words": [w.to_dict() for w in self.words],
                "pause_after_s": self.pause_after_s}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LineTiming":
        return cls(line_id=d["line_id"], start_s=d["start_s"],
                   end_s=d["end_s"],
                   words=tuple(WordTiming.from_dict(w)
                               for w in d.get("words", ())),
                   pause_after_s=d.get("pause_after_s", 0.0))


# ---------------------------------------------------------------------------
# SpeechCandidate — one take
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpeechCandidate(_Artifact):
    """One synthesized take of one line.

    ``duration_s`` is MEASURED on the artifact (the runner reads it off the wav)
    — it is the only number here the fan-out trusts for timing. ``seed`` is the
    deterministic seed this take was rolled with (:func:`candidate_seed`), so a
    take can be reproduced exactly. ``scorecard_digest`` points at the judge's
    verdict rather than embedding it: the card is evidence and rides in the
    receipt/gap, the candidate only names which card judged it. ``accepted`` is
    the judge's word, never the generator's."""

    candidate_id: str
    line_id: str
    voice_id: str
    audio_ref: str
    duration_s: float
    seed: int
    scorecard_digest: str | None = None
    accepted: bool = False

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "SpeechCandidate.candidate_id")
        _require_text(self.line_id, "SpeechCandidate.line_id")
        _require_text(self.voice_id, "SpeechCandidate.voice_id")
        _require_text(self.audio_ref, "SpeechCandidate.audio_ref")
        object.__setattr__(self, "duration_s",
                           _require_non_negative(self.duration_s,
                                                 "SpeechCandidate.duration_s"))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"SpeechCandidate.seed must be an int, got "
                            f"{type(self.seed).__name__}")
        if self.seed < 0:
            raise ValueError(f"SpeechCandidate.seed must be non-negative, got "
                             f"{self.seed}")
        object.__setattr__(self, "accepted", bool(self.accepted))

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "line_id": self.line_id,
                "voice_id": self.voice_id, "audio_ref": self.audio_ref,
                "duration_s": self.duration_s, "seed": self.seed,
                "scorecard_digest": self.scorecard_digest,
                "accepted": self.accepted}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpeechCandidate":
        return cls(candidate_id=d["candidate_id"], line_id=d["line_id"],
                   voice_id=d["voice_id"], audio_ref=d["audio_ref"],
                   duration_s=d["duration_s"], seed=int(d["seed"]),
                   scorecard_digest=d.get("scorecard_digest"),
                   accepted=bool(d.get("accepted", False)))


# ---------------------------------------------------------------------------
# AudioMaster — the definitive audio timeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioMaster(_Artifact):
    """The definitive audio timeline. Shot timing is cut TO this (doc Stage 8);
    this is never retimed to fit shots that were generated first.

    ``timeline_digest`` binds the master to the exact locked dialogue it
    realizes — a script edit changes that digest, and a master pointing at a
    digest nobody recognizes is caught instead of being quietly reused.
    ``tracks`` are ``(line_id, audio_ref)`` in timeline order, one per line
    timing; the two tuples are validated to agree, so the audio a window plays
    can never drift from the window itself. ``total_seconds`` includes the
    trailing pause of the last line (the timeline's full length, not the last
    word's end). ``candidates_considered`` is provenance the operator can read:
    how many takes were synthesized and judged to produce this master.
    ``registry_version`` records which routing-registry snapshot produced the
    audio (k105) — None while the registry is unversioned, never a guess."""

    timeline_digest: str
    line_timings: tuple[LineTiming, ...]
    tracks: tuple[tuple[str, str], ...]
    total_seconds: float
    candidates_considered: int = 0
    registry_version: str | None = None
    locked: bool = False

    def __post_init__(self) -> None:
        _require_text(self.timeline_digest, "AudioMaster.timeline_digest")
        object.__setattr__(self, "line_timings", tuple(self.line_timings))
        object.__setattr__(self, "tracks",
                           tuple((str(a), str(b)) for a, b in self.tracks))
        object.__setattr__(self, "total_seconds",
                           _require_non_negative(self.total_seconds,
                                                 "AudioMaster.total_seconds"))
        for timing in self.line_timings:
            if not isinstance(timing, LineTiming):
                raise TypeError(f"AudioMaster.line_timings takes LineTiming, "
                                f"got {type(timing).__name__}")
        for line_id, audio_ref in self.tracks:
            if not line_id.strip() or not audio_ref.strip():
                raise ValueError(f"AudioMaster track ({line_id!r}, "
                                 f"{audio_ref!r}) needs both halves")
        timed = [t.line_id for t in self.line_timings]
        tracked = [t[0] for t in self.tracks]
        if timed != tracked:
            raise ValueError(
                f"AudioMaster timings and tracks disagree: timings {timed} vs "
                f"tracks {tracked} — every line window must name the audio it "
                f"plays, in the same order")
        if not isinstance(self.candidates_considered, int) or \
                isinstance(self.candidates_considered, bool) or \
                self.candidates_considered < 0:
            raise ValueError(f"AudioMaster.candidates_considered must be a "
                             f"non-negative int, got "
                             f"{self.candidates_considered!r}")
        if self.line_timings:
            last_end = max(t.end_s for t in self.line_timings)
            if self.total_seconds + _EPS < last_end:
                raise ValueError(
                    f"AudioMaster.total_seconds {self.total_seconds} is shorter "
                    f"than its own last line end {last_end}")

    # --- reading -----------------------------------------------------------
    @property
    def line_ids(self) -> tuple[str, ...]:
        return tuple(t.line_id for t in self.line_timings)

    def timing(self, line_id: str) -> LineTiming:
        for timing in self.line_timings:
            if timing.line_id == line_id:
                return timing
        raise KeyError(f"unknown line_id: {line_id!r}")

    def audio_ref(self, line_id: str) -> str:
        for candidate_id, ref in self.tracks:
            if candidate_id == line_id:
                return ref
        raise KeyError(f"unknown line_id: {line_id!r}")

    def windows(self, include_pause: bool = False) -> tuple[
            tuple[str, float, float], ...]:
        """``(line_id, start_s, end_s)`` per line — the shot windows k106 cuts
        to. ``include_pause`` extends each window over its trailing pause, for a
        shot that must hold the beat after the line rather than cut on it."""
        return tuple(
            (t.line_id, t.start_s, t.next_start_s if include_pause else t.end_s)
            for t in self.line_timings)

    def lock(self) -> "AudioMaster":
        return self if self.locked else replace(self, locked=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline_digest": self.timeline_digest,
            "line_timings": [t.to_dict() for t in self.line_timings],
            "tracks": [list(t) for t in self.tracks],
            "total_seconds": self.total_seconds,
            "candidates_considered": self.candidates_considered,
            "registry_version": self.registry_version,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AudioMaster":
        return cls(
            timeline_digest=d["timeline_digest"],
            line_timings=tuple(LineTiming.from_dict(t)
                               for t in d.get("line_timings", ())),
            tracks=tuple((t[0], t[1]) for t in d.get("tracks", ())),
            total_seconds=d.get("total_seconds", 0.0),
            candidates_considered=int(d.get("candidates_considered", 0)),
            registry_version=d.get("registry_version"),
            locked=bool(d.get("locked", False)),
        )


# ---------------------------------------------------------------------------
# Policy + typed results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpeechPolicy:
    """Pacing + acceptance rules for one fan-out. Everything that would
    otherwise be a magic number inside ``build_audio_master`` lives here, so the
    operator's pacing decisions are data the receipt can show.

    ``require_similarity`` is the honesty switch: with it False (the default on
    this fleet, which has NO speaker-embedding backend — see
    ``catalog.audio.speaker_similarity``) an unmeasured similarity is UNSCORED
    and a take stays selectable on its other evidence. With it True, "nobody
    measured" is not good enough and the line fails with CAPABILITY_GAP — the
    fleet's gap, named as such, never VOICE_SIMILARITY_LOW blaming the take.

    ``seed_salt`` re-rolls the candidate seeds deterministically: a repair pass
    sets ``seed_salt='repair:1'`` and gets DIFFERENT takes that are still exactly
    reproducible. Re-running with the same salt reproduces the same takes, which
    is what makes a repair auditable."""

    pause_after_s: float = DEFAULT_PAUSE_AFTER_S
    pause_overrides: tuple[tuple[str, float], ...] = ()
    lead_in_s: float = 0.0
    require_similarity: bool = False
    similarity_threshold: float = speech.DEFAULT_SIMILARITY_THRESHOLD
    duration_tolerance: float = speech.DEFAULT_DURATION_TOLERANCE
    seed_salt: str = ""
    registry_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pause_after_s",
                           _require_non_negative(self.pause_after_s,
                                                 "SpeechPolicy.pause_after_s"))
        object.__setattr__(self, "lead_in_s",
                           _require_non_negative(self.lead_in_s,
                                                 "SpeechPolicy.lead_in_s"))
        object.__setattr__(
            self, "duration_tolerance",
            _require_non_negative(self.duration_tolerance,
                                  "SpeechPolicy.duration_tolerance"))
        threshold = float(self.similarity_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"SpeechPolicy.similarity_threshold must be in "
                             f"[0, 1], got {threshold}")
        object.__setattr__(self, "similarity_threshold", threshold)
        overrides = self.pause_overrides
        items = (overrides.items() if isinstance(overrides, Mapping)
                 else tuple(overrides))
        normalized = tuple(sorted(
            (str(k), _require_non_negative(v, f"pause override {k!r}"))
            for k, v in items))
        object.__setattr__(self, "pause_overrides", normalized)

    def pause_for(self, line_id: str) -> float:
        """The pause held after ``line_id``: its override, else the default."""
        for key, value in self.pause_overrides:
            if key == line_id:
                return _q(value)
        return _q(self.pause_after_s)

    def to_dict(self) -> dict[str, Any]:
        return {"pause_after_s": self.pause_after_s,
                "pause_overrides": {k: v for k, v in self.pause_overrides},
                "lead_in_s": self.lead_in_s,
                "require_similarity": self.require_similarity,
                "similarity_threshold": self.similarity_threshold,
                "duration_tolerance": self.duration_tolerance,
                "seed_salt": self.seed_salt,
                "registry_version": self.registry_version}


@dataclass(frozen=True, slots=True)
class AudioGap:
    """No take of one line passed the judge — a typed refusal, not a downgrade.

    Carries every repair code seen across the rejected candidates (not just the
    first), because the repair controller decides what to regenerate from the
    WHOLE picture: a line that failed one take on LINE_OMITTED and another on
    SHOT_TOO_SHORT is a different problem from one that failed both the same
    way. ``scorecards`` are the evidence behind those codes, one per rejected
    candidate, in candidate order."""

    line_id: str
    repair_codes: tuple[RepairCode, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    scorecards: tuple[Scorecard, ...] = ()
    candidates_considered: int = 0
    diagnosis: str = ""

    def __post_init__(self) -> None:
        _require_text(self.line_id, "AudioGap.line_id")
        object.__setattr__(self, "repair_codes", tuple(self.repair_codes))
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        object.__setattr__(self, "scorecards", tuple(self.scorecards))

    @property
    def primary_code(self) -> RepairCode | None:
        """The code to repair FIRST — ``_REPAIR_PRIORITY`` order, largest thing
        first."""
        seen = set(self.repair_codes)
        for name in _REPAIR_PRIORITY:
            code = AUDIO_REPAIR[name]
            if code in seen:
                return code
        return self.repair_codes[0] if self.repair_codes else None

    def to_dict(self) -> dict[str, Any]:
        return {"line_id": self.line_id,
                "repair_codes": [c.value for c in self.repair_codes],
                "primary_code": (self.primary_code.value
                                 if self.primary_code else None),
                "candidate_ids": list(self.candidate_ids),
                "scorecards": [s.to_dict() for s in self.scorecards],
                "candidates_considered": self.candidates_considered,
                "diagnosis": self.diagnosis}


@dataclass(frozen=True, slots=True)
class AudioBuildResult:
    """What one fan-out produced: a locked ``AudioMaster``, or gaps.

    Never both — a master is emitted only when EVERY line was accepted. The
    candidates (accepted and rejected alike) ride along as provenance, and
    ``warnings`` records what was honest-but-incomplete (a transcript with no
    word times, words clamped into their own line window)."""

    master: AudioMaster | None = None
    gaps: tuple[AudioGap, ...] = ()
    candidates: tuple[SpeechCandidate, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gaps", tuple(self.gaps))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.master is not None and self.gaps:
            raise ValueError(
                "AudioBuildResult cannot carry a master AND gaps — a timeline "
                "with an unresolved line is not a definitive timeline")

    @property
    def ok(self) -> bool:
        return self.master is not None and not self.gaps

    @property
    def candidates_considered(self) -> int:
        return len(self.candidates)

    @property
    def repair_codes(self) -> tuple[RepairCode, ...]:
        """Every distinct code across every gap, in repair-first order."""
        seen = {c for g in self.gaps for c in g.repair_codes}
        ordered = [AUDIO_REPAIR[n] for n in _REPAIR_PRIORITY
                   if AUDIO_REPAIR[n] in seen]
        ordered += [c for c in sorted(seen, key=lambda c: c.value)
                    if c not in ordered]
        return tuple(ordered)

    def accepted(self) -> tuple[SpeechCandidate, ...]:
        return tuple(c for c in self.candidates if c.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok,
                "master": self.master.to_dict() if self.master else None,
                "gaps": [g.to_dict() for g in self.gaps],
                "candidates": [c.to_dict() for c in self.candidates],
                "candidates_considered": self.candidates_considered,
                "repair_codes": [c.value for c in self.repair_codes],
                "warnings": list(self.warnings)}


# ---------------------------------------------------------------------------
# Seeds + transcript coercion
# ---------------------------------------------------------------------------


def candidate_seed(line: Line, voice: VoiceProfile, index: int,
                   salt: str = "") -> int:
    """The deterministic seed for take ``index`` of ``line`` in ``voice``.

    Derived from the two artifact DIGESTS, so: editing the line's text re-rolls
    its takes (the old ones judged different words); re-casting the line to
    another voice re-rolls them (the old ones were somebody else); everything
    else reproduces bit-for-bit. ``salt`` is the repair dial — a second pass
    over the same line with ``salt='repair:1'`` gets different but reproducible
    takes instead of re-rolling the identical failure.

    Reduced mod 2**32 because that is what numpy/torch seeding accepts."""
    if index < 0:
        raise ValueError(f"candidate index must be non-negative, got {index}")
    payload = f"{line.digest}:{voice.digest}:{int(index)}:{salt}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) \
        % SEED_MODULUS


def _candidate_id(line_id: str, index: int, seed: int) -> str:
    """A stable, readable take id: ``line-2#0@1a2b3c4d``. Deterministic (no
    uuid) so an identical fan-out produces identical candidate digests."""
    return f"{line_id}#{index}@{seed:08x}"


def _field(item: Any, *names: str) -> Any:
    for name in names:
        value = (item.get(name) if isinstance(item, Mapping)
                 else getattr(item, name, None))
        if value is not None:
            return value
    return None


def coerce_word_timings(items: Iterable[Any]) -> tuple[
        tuple[WordTiming, ...], int]:
    """Whatever the ASR seam returned -> ``(WordTiming tuple, untimed count)``.

    Accepts ``WordTiming``, the pydantic ``TranscribeWord``
    (``word/start/end/probability``), the raw whisper mapping, or this module's
    own key names — one coercion so callers never convert between the schema and
    the dict the runner also returns.

    A word with TEXT but no usable start/end is NOT invented onto the timeline;
    it is counted and reported as a warning. Its text still reaches the line
    check, because that check runs on the raw transcript — losing a word's time
    must never look like losing the word."""
    out: list[WordTiming] = []
    untimed = 0
    for item in items or ():
        if isinstance(item, WordTiming):
            out.append(item)
            continue
        text = _word_text(item).strip()
        if not text:
            continue
        start = _field(item, "start_s", "start")
        end = _field(item, "end_s", "end")
        if start is None or end is None:
            untimed += 1
            continue
        start_f = max(0.0, float(start))
        end_f = max(start_f, float(end))
        prob = _field(item, "prob", "probability")
        out.append(WordTiming(word=text, start_s=_q(start_f), end_s=_q(end_f),
                              prob=None if prob is None else float(prob)))
    return tuple(out), untimed


# ---------------------------------------------------------------------------
# The judge (k98's checks, folded per candidate)
# ---------------------------------------------------------------------------


def _similarity_evidence_check(similarity: float | None,
                               required: bool) -> Check | None:
    """The policy check: was similarity MEASURED at all?

    Distinct from k98's ``speech.speaker_similarity`` (which asks whether the
    voice matches) because the two have different owners: a low score is the
    take's problem, no score at all is the fleet's. Returns None when the policy
    does not require the evidence — an absent check beats a passing one nobody
    asked for."""
    if not required:
        return None
    measured = similarity is not None
    return Check(
        name=SIMILARITY_EVIDENCE_CHECK, kind=CheckKind.IDENTITY,
        value=similarity, threshold=None, passed=measured,
        detail=("speaker similarity was measured" if measured else
                "policy requires speaker-similarity evidence and none was "
                "measured (no speaker-embedding backend on this fleet) — this "
                "is a fleet capability gap, not a fault of the take"))


def judge_candidate(line: Line, duration_s: float, transcript: Sequence[Any],
                    similarity: float | None,
                    policy: SpeechPolicy) -> Scorecard:
    """k98's three speech checks over one take, plus the policy's evidence gate.

    The expected line is the LOCKED text and the transcript is the round trip
    over the produced audio: the generator's own claim about what it said is
    never an input here (invariant 11)."""
    extra = _similarity_evidence_check(similarity, policy.require_similarity)
    return speech.speech_scorecard(
        expected_lines=(line.text,),
        transcript_words=transcript,
        similarity=similarity,
        similarity_threshold=policy.similarity_threshold,
        audio_seconds=duration_s,
        shot_seconds=line.max_seconds,
        duration_tolerance=policy.duration_tolerance,
        extra_checks=() if extra is None else (extra,))


def candidate_repair_codes(card: Scorecard) -> tuple[RepairCode, ...]:
    """Every repair code the failing checks on ``card`` diagnose, in
    repair-first order. Empty for a passing card (a repair diagnoses failure)."""
    failing = {c.name for c in card.checks if not c.passed}
    return tuple(AUDIO_REPAIR[n] for n in _REPAIR_PRIORITY if n in failing)


def _rank_key(index: int, card: Scorecard, duration_s: float,
              similarity: float | None, line: Line) -> tuple:
    """Selection order, worst-to-best on every axis, ties broken determinis-
    tically:

      1. ACCEPTED first — a rejected take never outranks an accepted one, no
         matter how good its other numbers look.
      2. Fewest failed checks (only ever compares rejected takes).
      3. Closest to the line's declared budget; with no budget there is no
         duration preference and this term is constant (0.0) rather than
         inventing one.
      4. Highest MEASURED similarity; unmeasured (-1.0) ranks below any measured
         score instead of being treated as perfect.
      5. Candidate index — so the same fan-out always picks the same take.
    """
    failed = sum(1 for c in card.checks if not c.passed)
    target = line.max_seconds
    distance = abs(float(duration_s) - float(target)) if target else 0.0
    sim = -1.0 if similarity is None else float(similarity)
    return (0 if card.hard_pass else 1, failed, distance, -sim, index)


# ---------------------------------------------------------------------------
# The fan-out
# ---------------------------------------------------------------------------


SynthFn = Callable[[Line, VoiceProfile, int], Any]
TranscribeFn = Callable[[str], Any]
SimilarityFn = Callable[[str, VoiceProfile], Any]


def _casting_table(voices: Any) -> dict[str, VoiceProfile]:
    """``voices`` -> ``{speaker: VoiceProfile}``. Accepts the casting mapping
    directly, or an iterable of profiles keyed by their own ``voice_id``."""
    if isinstance(voices, Mapping):
        table = {str(k): v for k, v in voices.items()}
    else:
        table = {v.voice_id: v for v in voices or ()}
    for speaker, profile in table.items():
        if not isinstance(profile, VoiceProfile):
            raise TypeError(f"casting for {speaker!r} must be a VoiceProfile, "
                            f"got {type(profile).__name__}")
    return table


def _synth_result(raw: Any, line: Line) -> tuple[str, float]:
    """Unpack ``synth`` -> ``(audio_ref, duration_s)``, strictly. A seam that
    returns something else is a wiring bug and says so immediately, rather than
    producing a timeline built on a misread tuple."""
    try:
        audio_ref, duration_s = raw
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"synth(line={line.line_id!r}) must return (audio_ref, "
            f"duration_s), got {raw!r}") from exc
    return str(audio_ref), float(duration_s)


def build_audio_master(
        timeline: DialogueTimeline,
        voices: Mapping[str, VoiceProfile] | Iterable[VoiceProfile],
        *,
        synth: SynthFn,
        transcribe: TranscribeFn,
        similarity: SimilarityFn | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        policy: SpeechPolicy) -> AudioBuildResult:
    """Fan out N takes per line, judge each on its ROUND TRIP, assemble the
    definitive audio timeline — or return typed gaps.

    Seams (all injected; this module imports no backend):

        ``synth(line, voice, seed) -> (audio_ref, duration_s)``
        ``transcribe(audio_ref)    -> [WordTiming | whisper word | ...]``
        ``similarity(audio_ref, voice) -> float | None``   (optional)

    Per line: ``candidates`` takes at deterministic seeds
    (:func:`candidate_seed`), each judged by :func:`judge_candidate`; the best
    ACCEPTED take wins on ``(hard_pass, fewest failed checks, closest to
    ``max_seconds``, highest measured similarity, index)``. If no take is
    accepted, the line yields an :class:`AudioGap` with every repair code seen
    and no master is emitted — the least-bad take is never quietly promoted.
    EVERY line is judged even after one gaps, so one pass reports every problem.

    Then the accepted takes are laid out sequentially from ``policy.lead_in_s``,
    each followed by ``policy.pause_for(line_id)``, with word timings taken from
    the round-trip transcript shifted onto the master timeline (and clamped into
    their own line's window — an ASR that reports past the end of its input is
    reporting about nothing). The result is ``AudioMaster(locked=True)``: the
    thing shot timing is cut TO.

    Raises (programmer error, not runtime data): an unlocked timeline, a line
    with no voice cast for its speaker, ``candidates < 1``, a seam that returns
    the wrong shape."""
    if not isinstance(timeline, DialogueTimeline):
        raise TypeError(f"timeline must be a DialogueTimeline, got "
                        f"{type(timeline).__name__}")
    if not timeline.locked:
        raise ValueError(
            "build_audio_master needs a LOCKED DialogueTimeline (doc Stage 8: "
            "lock the dialogue, THEN generate definitive audio). Call "
            "timeline.lock() when the dialogue is final")
    if not isinstance(policy, SpeechPolicy):
        raise TypeError(f"policy must be a SpeechPolicy, got "
                        f"{type(policy).__name__}")
    if int(candidates) < 1:
        raise ValueError(f"candidates must be >= 1, got {candidates}")
    if synth is None or transcribe is None:
        raise ValueError("build_audio_master needs both synth and transcribe: "
                         "the round-trip transcript IS the judge's evidence "
                         "(the generator does not grade itself)")

    table = _casting_table(voices)
    missing = [ln.line_id for ln in timeline.lines if ln.speaker not in table]
    if missing:
        raise ValueError(
            f"no voice cast for speaker(s) of line(s) {', '.join(missing)}; "
            f"cast table has {sorted(table) or 'nothing'}")

    all_candidates: list[SpeechCandidate] = []
    warnings: list[str] = []
    gaps: list[AudioGap] = []
    chosen: dict[str, tuple[SpeechCandidate, tuple[WordTiming, ...]]] = {}

    for line in timeline.lines:
        voice = table[line.speaker]
        takes: list[tuple[tuple, SpeechCandidate, Scorecard,
                          tuple[WordTiming, ...]]] = []

        for index in range(int(candidates)):
            seed = candidate_seed(line, voice, index, policy.seed_salt)
            audio_ref, duration_s = _synth_result(
                synth(line, voice, seed), line)
            transcript = tuple(transcribe(audio_ref) or ())
            words, untimed = coerce_word_timings(transcript)
            score = None if similarity is None else similarity(audio_ref, voice)
            score = None if score is None else float(score)

            card = judge_candidate(line, duration_s, transcript, score, policy)
            try:  # k113a: the card is evidence about the TTS model that produced the take
                from hugpy_oracle import selection as _selection
                _selection.note_verdict_for_ref(audio_ref, hard_pass=bool(card.hard_pass),
                                                repair_code=card.repair_code)
            except Exception:  # noqa: BLE001
                pass
            candidate = SpeechCandidate(
                candidate_id=_candidate_id(line.line_id, index, seed),
                line_id=line.line_id, voice_id=voice.voice_id,
                audio_ref=audio_ref, duration_s=_q(duration_s), seed=seed,
                scorecard_digest=scorecard_digest(card),
                accepted=card.hard_pass)
            all_candidates.append(candidate)
            if untimed and card.hard_pass:
                warnings.append(
                    f"{candidate.candidate_id}: {untimed} transcript word(s) "
                    f"carried no timing and were left off the timeline")
            takes.append((_rank_key(index, card, duration_s, score, line),
                          candidate, card, words))

        takes.sort(key=lambda t: t[0])
        _, best, best_card, best_words = takes[0]
        if not best.accepted:
            codes: list[RepairCode] = []
            for _, _, card, _ in takes:
                for code in candidate_repair_codes(card):
                    if code not in codes:
                        codes.append(code)
            ordered = tuple(AUDIO_REPAIR[n] for n in _REPAIR_PRIORITY
                            if AUDIO_REPAIR[n] in set(codes))
            gaps.append(AudioGap(
                line_id=line.line_id,
                repair_codes=ordered or tuple(codes),
                candidate_ids=tuple(c.candidate_id for _, c, _, _ in takes),
                scorecards=tuple(card for _, _, card, _ in takes),
                candidates_considered=len(takes),
                diagnosis=(
                    f"line {line.line_id!r} ({line.speaker}): none of "
                    f"{len(takes)} candidate(s) passed — " +
                    "; ".join(card.diagnosis or "no diagnosis"
                              for _, _, card, _ in takes))))
            continue
        chosen[line.line_id] = (best, best_words)

    if gaps:
        return AudioBuildResult(master=None, gaps=tuple(gaps),
                                candidates=tuple(all_candidates),
                                warnings=tuple(warnings))

    timings: list[LineTiming] = []
    tracks: list[tuple[str, str]] = []
    cursor = _q(policy.lead_in_s)
    for line in timeline.lines:
        best, words = chosen[line.line_id]
        start = _q(cursor)
        end = _q(start + best.duration_s)
        placed: list[WordTiming] = []
        clamped = False
        for word in words:
            if word.end_s > best.duration_s + _EPS:
                clamped = True
            w_start = min(max(_q(start + word.start_s), start), end)
            w_end = min(max(_q(start + word.end_s), w_start), end)
            placed.append(WordTiming(word=word.word, start_s=w_start,
                                     end_s=w_end, prob=word.prob))
        if clamped:
            warnings.append(
                f"{best.candidate_id}: transcript word(s) ran past the "
                f"{best.duration_s:.3f}s of audio they describe and were "
                f"clamped into the line window")
        pause = policy.pause_for(line.line_id)
        timings.append(LineTiming(line_id=line.line_id, start_s=start,
                                  end_s=end, words=tuple(placed),
                                  pause_after_s=pause))
        tracks.append((line.line_id, best.audio_ref))
        cursor = _q(end + pause)

    master = AudioMaster(
        timeline_digest=timeline.digest,
        line_timings=tuple(timings),
        tracks=tuple(tracks),
        total_seconds=_q(cursor),
        candidates_considered=len(all_candidates),
        registry_version=policy.registry_version,
        locked=True)
    return AudioBuildResult(master=master, gaps=(),
                            candidates=tuple(all_candidates),
                            warnings=tuple(warnings))


__all__ = [
    # tunables
    "AUDIO_REPAIR", "DEFAULT_CANDIDATES", "DEFAULT_PAUSE_AFTER_S",
    "SEED_MODULUS", "SIMILARITY_EVIDENCE_CHECK", "STYLE_TRAITS",
    "TIME_PRECISION",
    # artifacts
    "AudioMaster", "DialogueTimeline", "Line", "LineTiming", "SpeechCandidate",
    "VoiceKind", "VoiceProfile", "WordTiming",
    # results
    "AudioBuildResult", "AudioGap", "SpeechPolicy",
    # functions
    "build_audio_master", "candidate_repair_codes", "candidate_seed",
    "canonical_json", "coerce_word_timings", "digest_payload",
    "judge_candidate", "scorecard_digest",
]
