"""Speech evidence producers (k98) — the doc §9 "Speech" row, as pure functions.

The heterogeneous-evaluation table names one evidence class this module owns::

    Speech | Round-trip transcription, omitted words, prosody ranges, cadence,
            and speaker similarity.

Everything here is DETERMINISTIC and OFFLINE: no network, no model, no disk.
Each producer takes data structures that already exist elsewhere in the tree
(``TranscribeWord``-shaped words from ``imports/src/schemas/whisper_schemas``,
a similarity float from a speaker-embedding backend, two durations) and returns
a ``Check`` — the same currency ``scorecard.build_technical_scorecard`` folds.
Producing the INPUTS is somebody else's job (round-trip ASR is a dispatch;
speaker embeddings need a model this fleet does not have yet — see
``catalog.audio.speaker_similarity``, which is honestly ineligible). That split
is deliberate: an evaluator that cannot be run without a GPU cannot be tested,
and these rules are exactly the part worth pinning down now.

THE THREE CHECKS

``check_lines_present``  — LINE_OMITTED. Round-trip ASR said the words; did the
    locked dialogue actually survive synthesis? Token-level, in order, with a
    documented miss budget (see ``TOKEN_MISS_DIVISOR``).
``check_speaker_similarity`` — VOICE_SIMILARITY_LOW. Embedding cosine (or any
    [0,1] similarity) against a threshold.
``check_duration_fit`` — SHOT_TOO_SHORT. doc Stage 8: "the definitive audio
    timeline precedes final shot timing"; audio longer than its shot window is
    the shot's fault, never the line's.

UNSCORED IS NOT PASSED. A check whose input is missing (no similarity score
because no embedding model is seated; no expected lines; no durations) is
recorded with ``detail`` starting ``"unscored: "`` and does not manufacture
evidence. ``speech_scorecard`` counts it OUT of ``confidence`` rather than
letting it silently prop up ``hard_pass`` — the same honesty rule the catalog
applies to eligibility.

REPAIR CODES. ``Check`` has no repair field (contracts.py), so the check-name ->
``RepairCode`` mapping lives in ``SPEECH_REPAIR`` and ``speech_scorecard``
applies it in ``_REPAIR_PRIORITY`` order — mirroring ``scorecard._FAILURE_REPAIR``
so the two builders can never tell different stories. Every code used here
already exists in ``contracts.RepairCode``; k98 defines NO new enum member.

For the keeper: this module wants no ``RepairCode`` additions. If a later slice
needs "prosody out of range" / "cadence drift" as first-class codes, they belong
in contracts.py next to VOICE_SIMILARITY_LOW; until then those signals ride as
``JudgeResult`` evidence, not as gates.

No pathlib anywhere. No I/O at all.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from hugpy_oracle.contracts import Check, CheckKind, RepairCode, Scorecard

# ---------------------------------------------------------------------------
# Tunables — named, not magic.
# ---------------------------------------------------------------------------

#: Below this the voice is not the requested voice (doc §9 Identity/Speech).
#: 0.75 is the ECAPA/x-vector cosine convention for "same speaker" on clean
#: speech; it is a DEFAULT, always overridable per call, never a measured fact
#: of this fleet (no speaker-embedding model is registered here yet).
DEFAULT_SIMILARITY_THRESHOLD: float = 0.75

#: Seconds of slack a shot may borrow before ``SHOT_TOO_SHORT``. This is the
#: "bounded retiming" budget of doc Stage 18 expressed as a number: assembly may
#: stretch a shot by this much, so audio inside the slack is not yet a failure.
DEFAULT_DURATION_TOLERANCE: float = 0.15

#: The line-match miss budget: ONE missing token per this many expected tokens
#: (``allowed = len(tokens) // TOKEN_MISS_DIVISOR``). A 7-token line must match
#: every token; 8-15 tokens may lose one; 16-23 may lose two. Rationale: ASR
#: round-trip drops articles/contractions on short function words at roughly
#: this rate, and a zero-budget rule turns every "and"->"" into a regeneration.
#: Integer floor division on purpose — the budget must never round UP into
#: tolerating a dropped content word on a short line.
TOKEN_MISS_DIVISOR: int = 8

#: Prefix that marks a Check as carrying NO evidence (input absent).
UNSCORED_PREFIX: str = "unscored: "

# Check name -> the repair code a FAILING instance diagnoses. Same shape as
# scorecard._FAILURE_REPAIR: one table, two builders, one story.
SPEECH_REPAIR: dict[str, RepairCode] = {
    "speech.lines_present":     RepairCode.LINE_OMITTED,
    "speech.speaker_similarity": RepairCode.VOICE_SIMILARITY_LOW,
    "sync.duration_fit":        RepairCode.SHOT_TOO_SHORT,
}

# Which failure wins when several fail at once: a missing LINE invalidates the
# take outright; a wrong VOICE invalidates the performance; a too-short SHOT is
# a timing fix on otherwise good audio. Repair the largest thing first.
_REPAIR_PRIORITY: tuple[str, ...] = (
    "speech.lines_present", "speech.speaker_similarity", "sync.duration_fit")


# ---------------------------------------------------------------------------
# Normalization — the rule every token comparison in this module uses.
# ---------------------------------------------------------------------------

# Everything that is not a letter, a digit or an intra-word apostrophe becomes a
# separator. Apostrophes are KEPT inside a word ("don't" stays one token) because
# a TTS that says "do not" for "don't" changed the line, and the operator should
# see that; they are stripped at the edges ('quoted' -> quoted).
_SEPARATORS = re.compile(r"[^\w'’]+", re.UNICODE)
_EDGE_APOSTROPHES = re.compile(r"^['’]+|['’]+$")


def normalize_tokens(text: str) -> tuple[str, ...]:
    """Text -> comparable tokens. Deterministic and documented:

      1. Unicode NFKC (typographic quotes/ligatures fold to their ASCII kin);
      2. curly apostrophe -> straight;
      3. casefold (case-insensitive, and correct for non-ASCII unlike ``lower``);
      4. split on every non-word run (punctuation, dashes, whitespace);
      5. strip edge apostrophes; drop empties.

    Punctuation and case are therefore invisible to the line check, exactly as
    required: ASR emits "Hello there!" for a script's "Hello, there." and that
    is not an omission.
    """
    if not text:
        return ()
    folded = unicodedata.normalize("NFKC", str(text)).replace("’", "'")
    out: list[str] = []
    for raw in _SEPARATORS.split(folded.casefold()):
        tok = _EDGE_APOSTROPHES.sub("", raw)
        if tok:
            out.append(tok)
    return tuple(out)


def _word_text(word: Any) -> str:
    """One transcript word -> its text, whatever shape it arrived in.

    Accepts a ``TranscribeWord`` (``.word``), a plain mapping from the raw
    whisper JSON (``{"word": " hello"}`` / ``{"text": "hello"}``) or a bare
    string, so callers never have to convert between the pydantic schema and the
    ``raw`` dict the runner also returns. Unknown shapes read as empty (and are
    then simply absent from the stream) rather than raising: an evaluator must
    not crash on a transcript."""
    if isinstance(word, str):
        return word
    if isinstance(word, Mapping):
        return str(word.get("word") or word.get("text") or "")
    return str(getattr(word, "word", "") or getattr(word, "text", "") or "")


def transcript_token_stream(
        transcript_words: Iterable[Any]) -> tuple[str, ...]:
    """The flat, normalized token stream of a transcript. A single ASR "word"
    may normalize into several tokens (whisper emits "don't." and, with some
    backends, "New York" as one item), so this flattens rather than assuming
    one word == one token."""
    stream: list[str] = []
    for word in transcript_words or ():
        stream.extend(normalize_tokens(_word_text(word)))
    return tuple(stream)


def allowed_misses(n_tokens: int) -> int:
    """The miss budget for a line of ``n_tokens`` — see TOKEN_MISS_DIVISOR."""
    return max(0, int(n_tokens) // TOKEN_MISS_DIVISOR)


# ---------------------------------------------------------------------------
# Check 1 — LINE_OMITTED
# ---------------------------------------------------------------------------


def _match_line(line_tokens: Sequence[str], stream: Sequence[str],
                cursor: int) -> tuple[bool, int, tuple[str, ...]]:
    """Greedy in-order (subsequence) match of ``line_tokens`` in ``stream``
    starting at ``cursor``.

    Returns (ok, new_cursor, missed_tokens). IN ORDER is the whole point: the
    tokens must appear in the transcript in the sequence the script wrote them,
    and each line resumes where the previous line ended, so a take that says
    line 2 before line 1 fails even though every word is present.

    Misses do NOT consume the cursor (a token that was never said cannot have
    advanced the tape), and a line passes when ``len(missed) <=
    allowed_misses(len(line_tokens))``.
    """
    missed: list[str] = []
    pos = cursor
    for token in line_tokens:
        try:
            found = stream.index(token, pos)
        except ValueError:
            missed.append(token)
            continue
        pos = found + 1
    ok = len(missed) <= allowed_misses(len(line_tokens))
    return ok, pos, tuple(missed)


def check_lines_present(expected_lines: Sequence[str],
                        transcript_words: Iterable[Any]) -> Check:
    """Did every locked line survive into the round-trip transcript?

    ``expected_lines`` is the locked dialogue (doc Stage 8); ``transcript_words``
    is the word list of an ASR pass over the PRODUCED audio (the
    ``audio.transcribe.word_timestamps`` capability). Both are normalized with
    ``normalize_tokens``, so punctuation and case never cause a false omission.

    THE RULE, stated once: a line passes when its normalized tokens appear in
    the transcript stream IN ORDER, allowing ``len(tokens) // 8`` misses
    (TOKEN_MISS_DIVISOR); lines are matched with a shared, advancing cursor, so
    ordering holds ACROSS lines too. The check fails (``RepairCode.LINE_OMITTED``
    via ``SPEECH_REPAIR``) as soon as one line does not.

    ``value`` is the number of lines matched, ``threshold`` the number expected.
    """
    lines = [ln for ln in (expected_lines or ())]
    if not lines:
        return Check(
            name="speech.lines_present", kind=CheckKind.SPEECH,
            value=None, threshold=None, passed=True,
            detail=(UNSCORED_PREFIX + "no expected lines supplied — nothing to "
                    "verify (a non-dialogue artifact has no line evidence)"))

    stream = transcript_token_stream(transcript_words)
    if not stream:
        return Check(
            name="speech.lines_present", kind=CheckKind.SPEECH,
            value=0, threshold=len(lines), passed=False,
            detail=(f"transcript carries no words: all {len(lines)} expected "
                    f"line(s) are unaccounted for"))

    cursor = 0
    matched = 0
    failures: list[str] = []
    for index, line in enumerate(lines):
        tokens = normalize_tokens(line)
        if not tokens:
            matched += 1        # an empty line is vacuously present
            continue
        ok, cursor, missed = _match_line(tokens, stream, cursor)
        if ok:
            matched += 1
            continue
        failures.append(
            f"line {index}: {len(missed)}/{len(tokens)} token(s) missing or "
            f"out of order (budget {allowed_misses(len(tokens))}) — "
            f"{', '.join(repr(m) for m in missed[:6])}"
            f"{' …' if len(missed) > 6 else ''}")

    passed = not failures
    return Check(
        name="speech.lines_present", kind=CheckKind.SPEECH,
        value=matched, threshold=len(lines), passed=passed,
        detail=("; ".join(failures) if failures else
                f"all {matched} line(s) present in order "
                f"({len(stream)} transcript token(s))"))


# ---------------------------------------------------------------------------
# Check 2 — VOICE_SIMILARITY_LOW
# ---------------------------------------------------------------------------


def check_speaker_similarity(
        score: float | None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Check:
    """Is the produced voice the requested voice?

    ``score`` is an embedding similarity in [0, 1] produced OUTSIDE this module
    (doc §9 Identity: "face and speaker embeddings"). The comparison is
    ``score >= threshold`` — the threshold value itself PASSES, so a fleet that
    calibrates its threshold to a measured equal-error rate does not have to
    reason about an off-by-epsilon at the boundary.

    ``score is None`` means NOTHING MEASURED (no speaker-embedding model is
    registered on this fleet — ``catalog`` says so out loud). That is recorded
    as unscored, never as a pass with a fabricated number and never as a
    failure blaming the artifact for the fleet's gap.
    """
    if score is None:
        return Check(
            name="speech.speaker_similarity", kind=CheckKind.IDENTITY,
            value=None, threshold=threshold, passed=True,
            detail=(UNSCORED_PREFIX + "no speaker-embedding similarity was "
                    "measured (audio.speaker_similarity has no registered "
                    "backend on this fleet) — this check contributes no "
                    "evidence and is excluded from confidence"))

    value = float(score)
    if math.isnan(value):
        return Check(
            name="speech.speaker_similarity", kind=CheckKind.IDENTITY,
            value=None, threshold=threshold, passed=True,
            detail=(UNSCORED_PREFIX + "similarity is NaN (the embedding "
                    "backend produced no usable comparison)"))

    passed = value >= float(threshold)
    return Check(
        name="speech.speaker_similarity", kind=CheckKind.IDENTITY,
        value=value, threshold=float(threshold), passed=passed,
        detail=(f"speaker embedding similarity {value:.4f} "
                f"{'>=' if passed else '<'} threshold {float(threshold):.4f}"))


# ---------------------------------------------------------------------------
# Check 3 — SHOT_TOO_SHORT
# ---------------------------------------------------------------------------


def check_duration_fit(audio_seconds: float | None,
                       shot_seconds: float | None,
                       tolerance: float = DEFAULT_DURATION_TOLERANCE) -> Check:
    """Does the definitive audio fit the shot window it was cut for?

    doc Stage 8 fixes the direction of this test: the audio timeline is LOCKED
    FIRST and shot timing follows it, so audio longer than its window is a
    SHOT_TOO_SHORT diagnosis (regenerate/extend the shot), never a "speak
    faster" instruction to the voice. ``tolerance`` is the bounded-retiming
    slack in SECONDS (doc Stage 18) — audio inside the slack still passes.

    Passes when ``audio_seconds <= shot_seconds + tolerance``. Unscored when
    either duration is unknown; a negative duration is programmer error and
    raises (it cannot be produced by any honest measurement).
    """
    if audio_seconds is None or shot_seconds is None:
        missing = ", ".join(
            n for n, v in (("audio_seconds", audio_seconds),
                           ("shot_seconds", shot_seconds)) if v is None)
        return Check(
            name="sync.duration_fit", kind=CheckKind.SYNC,
            value=audio_seconds, threshold=shot_seconds, passed=True,
            detail=(UNSCORED_PREFIX + f"unknown duration(s): {missing}"))

    audio = float(audio_seconds)
    shot = float(shot_seconds)
    slack = float(tolerance)
    if audio < 0 or shot < 0 or slack < 0:
        raise ValueError(
            f"durations/tolerance must be non-negative, got "
            f"audio={audio}, shot={shot}, tolerance={slack}")

    budget = shot + slack
    passed = audio <= budget
    overrun = audio - budget
    return Check(
        name="sync.duration_fit", kind=CheckKind.SYNC,
        value=audio, threshold=budget, passed=passed,
        detail=(f"audio {audio:.3f}s fits shot {shot:.3f}s "
                f"(+{slack:.3f}s retime slack)" if passed else
                f"audio {audio:.3f}s exceeds shot {shot:.3f}s "
                f"(+{slack:.3f}s retime slack) by {overrun:.3f}s — extend the "
                f"shot; the locked audio timeline is authoritative"))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def is_unscored(check: Check) -> bool:
    """Whether ``check`` carries no evidence (its input was absent)."""
    return check.detail.startswith(UNSCORED_PREFIX)


def speech_repair_code(checks: Sequence[Check]) -> RepairCode | None:
    """The repair code for the highest-priority FAILING speech check, or None
    when nothing speech-related failed."""
    failing = {c.name for c in checks if not c.passed}
    for name in _REPAIR_PRIORITY:
        if name in failing:
            return SPEECH_REPAIR[name]
    return None


def speech_scorecard(
        *,
        expected_lines: Sequence[str] = (),
        transcript_words: Iterable[Any] = (),
        similarity: float | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        audio_seconds: float | None = None,
        shot_seconds: float | None = None,
        duration_tolerance: float = DEFAULT_DURATION_TOLERANCE,
        judge_results: tuple = (),
        extra_checks: Sequence[Check] = ()) -> Scorecard:
    """The three speech checks folded into the existing ``Scorecard`` shape.

    Same discipline as ``scorecard.build_technical_scorecard``: ``hard_pass`` is
    the conjunction of every check, ``repair_code`` is set ONLY on a failing
    card (contracts refuses it otherwise) and comes from ``SPEECH_REPAIR`` in
    ``_REPAIR_PRIORITY`` order, and ``judge_results`` stays a caller-supplied
    seam — a deterministic evaluator never invents a judge opinion.

    ``confidence`` is the SCORED fraction of the checks (1.0 when everything was
    measurable, 0.667 when one of three was unscored, and so on). That is the
    honest reading: a card built with no similarity score is not as sure as one
    built with it, and the number says so instead of the card pretending.

    ``extra_checks`` lets a caller fold in evidence produced elsewhere (a
    technical decode check on the wav, a lip-sync check from a later slice)
    without this function growing knowledge of it.
    """
    checks: list[Check] = [
        check_lines_present(expected_lines, transcript_words),
        check_speaker_similarity(similarity, similarity_threshold),
        check_duration_fit(audio_seconds, shot_seconds, duration_tolerance),
        *extra_checks,
    ]
    hard_pass = all(c.passed for c in checks)
    scored = [c for c in checks if not is_unscored(c)]
    confidence = round(len(scored) / len(checks), 3) if checks else 1.0

    diagnoses = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    unscored = [c.name for c in checks if is_unscored(c)]
    if unscored:
        diagnoses.append("unscored (no evidence): " + ", ".join(unscored))

    code = None if hard_pass else speech_repair_code(checks)
    return Scorecard(
        hard_pass=hard_pass,
        checks=tuple(checks),
        judge_results=tuple(judge_results),
        confidence=confidence,
        diagnosis="; ".join(diagnoses) or None,
        recommended_repair=None if hard_pass else _RECOMMENDED.get(
            code, "regenerate the failing speech artifact"),
        repair_code=code)


_RECOMMENDED: dict[RepairCode | None, str] = {
    RepairCode.LINE_OMITTED: (
        "re-synthesize the omitted line(s) only — the locked dialogue is "
        "authoritative; do not rewrite the script to match the take"),
    RepairCode.VOICE_SIMILARITY_LOW: (
        "re-synthesize with the authorized reference voice (or raise candidate "
        "count); never substitute a lookalike voice silently"),
    RepairCode.SHOT_TOO_SHORT: (
        "extend the shot window to the locked audio duration (doc Stage 8: the "
        "audio timeline precedes shot timing), then re-render that shot only"),
}


__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD", "DEFAULT_DURATION_TOLERANCE",
    "TOKEN_MISS_DIVISOR", "UNSCORED_PREFIX", "SPEECH_REPAIR",
    "allowed_misses", "check_duration_fit", "check_lines_present",
    "check_speaker_similarity", "is_unscored", "normalize_tokens",
    "speech_repair_code", "speech_scorecard", "transcript_token_stream",
]
