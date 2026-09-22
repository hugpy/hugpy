"""Randomized STEERING for the LLM prompt-assist "generate" mode.

WHY THIS EXISTS
---------------
``POST /video/prompt/assist`` with ``mode="generate"`` and no draft sent the
model one fixed instruction — *"Write one compelling, original image-generation
prompt of your choosing."* — with the same system message every time. A small
instruct model handed the same instruction twice returns near-identical text,
so clicking **Generate** repeatedly produced the same handful of prompts
(operator, 2026-07-27: _"i need generate in the /video (the llm generate) to
randomize the prompt"_).

Raising temperature alone does not fix that: it jitters wording while the model
still walks to the same attractor (a lone figure, a sunset, a mountain). What
actually diversifies output is changing the QUESTION, so this picks one value
per creative axis and hands the model a different brief on every call.

WHAT IT IS NOT
--------------
Not a prompt generator. The LLM still writes the prompt — these are only
CONSTRAINTS handed to it, so the result stays a coherent piece of prose rather
than a slot-filled template. The demo path (``ui/src/demo/promptAssist.ts``)
does the opposite: it assembles prompts from fragment banks entirely
client-side, because the showroom has no backend. Same vocabulary world,
deliberately different mechanism.

A DRAFT ALWAYS WINS
-------------------
When the caller supplied a draft, that draft is the theme and the SUBJECT axis
is dropped — steering may colour the shot, never replace what the operator
asked for. ``mode="detail"`` (Enhance) is not steered at all: its whole
contract is to preserve the draft's subject and intent.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

# ── axes ────────────────────────────────────────────────────────────────────
# One coherent slot each. Kept deliberately BROAD (not one house style) — the
# old failure was every result converging on the same cinematic sunset, so the
# banks span genre, era, palette and scale on purpose.
_SUBJECTS: Sequence[str] = (
    "a solitary traveller", "two figures meeting", "a working craftsperson",
    "an animal moving through its habitat", "a crowd caught mid-motion",
    "an abandoned machine", "a child at play", "an elderly face in close-up",
    "a vessel under way", "a structure dwarfing its surroundings",
    "hands completing a delicate task", "a performer mid-gesture",
)
_SETTINGS: Sequence[str] = (
    "a dense city street", "an alpine ridge above the cloud line",
    "a flooded forest", "a cramped workshop", "an empty stadium",
    "a desert salt flat", "a rain-lashed harbour", "a subterranean cavern",
    "a greenhouse gone wild", "a rooftop at night", "a frozen lake",
    "a market at full tilt", "an orbital platform", "a chalk cliff path",
)
_LIGHTS: Sequence[str] = (
    "hard midday sun with short black shadows", "overcast, flat and silver",
    "single-source lamplight in deep dark", "backlit haze, subject in silhouette",
    "cold blue hour just before dawn", "firelight, warm and unsteady",
    "fluorescent strip light, unflattering and green",
    "dappled light through moving leaves", "neon spill on wet surfaces",
    "moonlight, desaturated and sharp",
)
_MOODS: Sequence[str] = (
    "tense", "serene", "melancholy", "triumphant", "unsettling", "intimate",
    "playful", "reverent", "exhausted", "curious", "defiant", "wistful",
)
_PALETTES: Sequence[str] = (
    "muted earth tones", "high-contrast monochrome", "saturated primaries",
    "cool teal and slate", "warm ochre and rust", "pastel and washed out",
    "deep jewel tones", "bleached near-white", "sodium-vapour amber",
)
# Stills get framing; video gets framing PLUS motion — a t2v/i2v prompt that
# names no movement wastes the medium.
_SHOTS: Sequence[str] = (
    "extreme close-up", "tight portrait framing", "waist-up medium shot",
    "full-body wide shot", "sweeping establishing shot", "low angle looking up",
    "high angle looking down", "over-the-shoulder framing", "dutch-tilted frame",
)
_MOTIONS: Sequence[str] = (
    "a slow push in", "a lateral tracking move", "a static locked-off frame",
    "a handheld drift", "a crane rise", "a slow orbit around the subject",
    "a whip pan settling on the subject", "a pull-back reveal",
    "subject moving toward camera", "a rack focus between two planes",
)

_VIDEO_KINDS = frozenset({"movie", "clip"})

# ANTI-REPEAT. Uniform sampling over 12 subjects repeats often — observed twice
# in the first three live calls, which reads to the operator as "it didn't
# randomize" even though it did. So the two axes a reader actually NOTICES
# (subject, setting) never repeat back-to-back within a process. Deliberately
# in-memory and process-local: this is a cosmetic anti-clustering nicety, not
# state worth persisting or synchronising across gunicorn workers.
_last_pick: Dict[str, str] = {}


def _choose(r: random.Random, axis: str, bank: Sequence[str],
            avoid_repeat: bool = False) -> str:
    """One value from ``bank``; when ``avoid_repeat``, never the same value this
    axis produced last time (unless the bank has only one entry).

    ⚠ ANTI-REPEAT IS SKIPPED WHEN AN ``rng`` WAS INJECTED. The retry loop draws a
    VARIABLE number of times depending on process-global ``_last_pick``, so with
    it enabled the same seeded Random yields different results on successive
    calls — i.e. an injected seed would stop being reproducible, which is the one
    property it exists to provide. Caught by
    ``test_injected_rng_is_deterministic``. Production (rng=None) keeps the
    anti-repeat; a seeded caller gets pure, replayable sampling."""
    value = r.choice(bank)
    if avoid_repeat and len(bank) > 1:
        previous = _last_pick.get(axis)
        attempts = 0
        while value == previous and attempts < 8:
            value = r.choice(bank)
            attempts += 1
        _last_pick[axis] = value
    return value


def steering_axes(kind: Optional[str] = None,
                  has_draft: bool = False,
                  rng: Optional[random.Random] = None) -> Dict[str, str]:
    """Pick one value per axis. ``rng`` is injectable so tests are deterministic.

    ``has_draft`` drops the SUBJECT axis: the caller's draft is the subject, and
    steering must never overwrite what they asked for.
    """
    r = rng or random
    # A seeded caller wants reproducibility; production wants anti-clustering.
    no_repeat = rng is None
    axes: Dict[str, str] = {}
    if not has_draft:
        axes["subject"] = _choose(r, "subject", _SUBJECTS, avoid_repeat=no_repeat)
    axes["setting"] = _choose(r, "setting", _SETTINGS, avoid_repeat=no_repeat)
    axes["light"] = r.choice(_LIGHTS)
    axes["mood"] = r.choice(_MOODS)
    axes["palette"] = r.choice(_PALETTES)
    axes["shot"] = r.choice(_SHOTS)
    if kind in _VIDEO_KINDS:
        axes["motion"] = r.choice(_MOTIONS)
    return axes


def steering_clause(kind: Optional[str] = None,
                    has_draft: bool = False,
                    rng: Optional[random.Random] = None) -> str:
    """The text appended to the user message. Empty string is never returned —
    callers that don't want steering simply don't call this."""
    axes = steering_axes(kind, has_draft=has_draft, rng=rng)
    lines: List[str] = [
        f"- {label}: {value}" for label, value in axes.items()
    ]
    lead = ("Interpret these creative constraints — honour every one, but write "
            "flowing prose, not a list of them:")
    return lead + "\n" + "\n".join(lines)


# ── SPREAD steering (STUDIO-SPREAD-SPEC §1a) ────────────────────────────────
# THE INVERSE OF THE ABOVE. Everything above this line randomizes PER CALL, which
# is exactly right for one Generate click and exactly WRONG for a movie: six rows
# generated one-at-a-time drew six independent steering sets and produced six
# unrelated worlds (the spec's "per-row Generate is actively incoherent").
#
# So a spread draws ONE steering set for the WHOLE movie — shared world, subject,
# light, mood, palette — and varies only the BEAT per segment. Coherence is not
# asked of the model as a politeness; it is a property of the brief it receives.
#
# The set is seeded (``steering_seed``) so a spread is REPLAYABLE: the same seed
# rebuilds the same world, which is what makes "regenerate just rows 2 and 5"
# land in the movie that already exists rather than a new one. No seed -> a fresh
# random world, and the seed actually used is returned so the caller can pin it.
_BEATS: Sequence[str] = (
    "establish the place and who is in it",
    "introduce the first change or arrival",
    "press the situation — raise the stakes",
    "the turn: something gives way",
    "consequence — the aftermath lands",
    "a quiet held moment",
    "close in on the smallest telling detail",
    "pull back and let the world answer",
)

# Beat density (spec §3): one segment carries ONE beat. A dense scene is spread
# across segments rather than crammed into one — too much action in a single
# clip is what makes motion unreliable.
_BEAT_DENSITY_RULE = (
    "Each segment carries ONE beat only. Do not cram several actions into one "
    "segment — a shot with too much action renders unreliable motion."
)


def beat_for_index(index: int, total: int) -> str:
    """The dramatic beat for segment ``index`` of ``total``.

    Walks the bank across the movie's length so a 3-segment spread gets a
    beginning/middle/end and an 8-segment spread gets eight distinct beats
    instead of the same three repeated. Pure function of (index, total) — no
    randomness, because the beat is the ONLY axis allowed to vary within a
    spread and it must vary in a legible, ordered way.
    """
    if total <= 1:
        return _BEATS[0]
    n = len(_BEATS)
    if total >= n:
        return _BEATS[min(index, n - 1)]
    # Spread the chosen beats evenly across the bank, always ending on a closer.
    step = (n - 1) / float(total - 1)
    return _BEATS[min(int(round(index * step)), n - 1)]


def spread_axes(kind: Optional[str] = "movie",
                seed: Optional[int] = None) -> Dict[str, str]:
    """ONE steering set for a whole spread. Deterministic for a given ``seed``.

    Unlike :func:`steering_axes` this NEVER drops the subject axis: a movie's
    subject is the thing that has to survive across segments, so it is part of
    the shared world even when individual rows carry their own prompts. The
    per-call anti-repeat is deliberately not used — it would break replay.
    """
    r = random.Random(seed) if seed is not None else random.Random()
    axes: Dict[str, str] = {
        "subject": r.choice(_SUBJECTS),
        "setting": r.choice(_SETTINGS),
        "light": r.choice(_LIGHTS),
        "mood": r.choice(_MOODS),
        "palette": r.choice(_PALETTES),
    }
    if kind in _VIDEO_KINDS or kind is None:
        axes["motion"] = r.choice(_MOTIONS)
    return axes


def spread_steering_clause(axes: Dict[str, str]) -> str:
    """Render a spread's shared steering set for the model preface.

    Takes the axes rather than drawing them so the caller can echo the EXACT set
    back to the UI (a spread has to be reproducible, and a clause that redrew
    internally could not be).
    """
    lines = [f"- {label}: {value}" for label, value in axes.items()]
    lead = ("SHARED WORLD — every segment below belongs to the SAME film and "
            "must honour all of these. They do not change between segments:")
    return lead + "\n" + "\n".join(lines) + "\n" + _BEAT_DENSITY_RULE


def combinations(kind: Optional[str] = None, has_draft: bool = False) -> int:
    """How many distinct briefs the banks can produce — used by the test that
    guards against the banks being thinned to the point of repetition."""
    n = len(_SETTINGS) * len(_LIGHTS) * len(_MOODS) * len(_PALETTES) * len(_SHOTS)
    if not has_draft:
        n *= len(_SUBJECTS)
    if kind in _VIDEO_KINDS:
        n *= len(_MOTIONS)
    return n
