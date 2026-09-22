"""Shot-intent derivation (IDENTITY-3D-CONTINUITY-PLAN.md S3) — turn a segment's
PROMPT text into a view hint WITHOUT asking the operator.

WHY this exists (defaults-are-promises): S2 lets a movie segment condition on a
specific turntable VIEW of the identity ("back", "left-profile", …) so a ``cut``
into a new scene can hold the character while turning the camera. But making the
operator hand-annotate every segment with a ``view`` defeats the promise that a
BLANK request produces the smart result. So at enqueue we read the segment's own
prompt for a shot cue — "she walks away", "left profile shot", "over the shoulder"
— and DERIVE the view. An explicit per-goal ``view`` always wins; a prompt with no
recognizable cue derives NOTHING (the segment inherits the movie-level DNA, exactly
today's behavior). Never an LLM call here — a pure, deterministic keyword pass so
the enqueue path stays cheap and unit-testable (the LLM assist is a later, flagged
option per the plan, not this leg).

CONSERVATIVE BY DESIGN. An ambiguous prompt returns ``None`` (inherit). We only
fire on distinctive, orientation-bearing phrases, matched WHOLE-WORD (so "rear"
never triggers on "rearrange"). Left/right is disambiguated when the prompt names a
side ("left profile" -> left-profile); an unsided cue resolves to a documented
default side. Every value returned is a real ``SEMANTIC_VIEWS`` key (guarded), so
the derived hint always resolves cleanly through ``azimuth_for_view``.

THE MAPPING (phrase -> view; trivially extendable — add a phrase to the right
bucket, or a bucket, and it flows through the same whole-word matcher):

    bucket          example cues                              -> view
    ------          ------------                              -------
    back            "from behind", "walks away", "back of",   -> back            (180°)
                    "rear", "facing away", "turns away"
    over-the-       "over the shoulder",                      -> back-<side>     (135°/225°)
      shoulder      "over her/his shoulder"                      (a back-quarter)
    profile         "profile", "side view", "sideways",       -> <side>-profile (90°/270°)
                    "side-on", "in profile"
    three-quarter   "three-quarter", "three quarter", "3/4",  -> three-quarter-<side> (45°/315°)
                    "45 degree"

Buckets are tested in the order above (most specific orientation first), so a
prompt that mixes cues resolves deterministically. Unsided defaults: profile ->
right-profile, three-quarter -> three-quarter-right, over-the-shoulder ->
back-right (documented, arbitrary — degrees are canonical, the label side is a
convenience per identity_profiles' azimuth convention).

No pathlib; pure text + a tiny regex. Depends only on ``SEMANTIC_VIEWS`` (the
canonical view vocabulary) for its output guard.
"""
from __future__ import annotations

import re
from typing import Optional

# The output guard: every derived name must be a real view in the canonical vocabulary
# so it resolves through ``azimuth_for_view``. Importing the map (rather than hardcoding
# the names) keeps this helper in lockstep with the bank's nomenclature — if a view is
# renamed there, a stale derivation here is dropped to None rather than emitting a name
# the resolver would reject.
from hugpy_video.intel.identity_profiles import SEMANTIC_VIEWS


def _has_cue(text: str, phrase: str) -> bool:
    """Whole-word (well, whole-token) containment of *phrase* in *text*. Alphanumerics
    are the token chars, so "rear" matches "the rear" / "rear." but NOT "rearrange", and
    "3/4" matches "a 3/4 shot" (the internal '/' is literal, only the ends are bounded).
    Multi-word / hyphenated phrases ("over the shoulder", "side-on") are matched verbatim
    with the same end-boundaries. Case is normalized by the caller."""
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


# Phrase buckets. ORDERED most-specific-orientation-first (see module docstring): a prompt
# that mixes cues (rare) resolves by the first bucket it hits. Extend by adding a phrase to
# a bucket or a new bucket + its resolver below.
_BACK_CUES = (
    "from behind", "from the back", "seen from behind", "back of the head",
    "back of her head", "back of his head", "back of",
    "walks away", "walking away", "walked away", "walks off", "walking off",
    "facing away", "turns away", "turned away", "turning away",
    "rear view", "rear shot", "from the rear", "rear",
)
_OVER_SHOULDER_CUES = (
    "over the shoulder", "over-the-shoulder", "over her shoulder", "over his shoulder",
)
_PROFILE_CUES = (
    "profile", "in profile", "side view", "side-view", "side profile",
    "sideways", "side-on", "side on",
)
_THREE_QUARTER_CUES = (
    "three-quarter", "three quarter", "3/4", "3-quarter", "45 degree", "45-degree",
)


def _side(text: str) -> Optional[str]:
    """"left" / "right" from the prompt, or None when unsided/ambiguous (BOTH named)."""
    has_left = _has_cue(text, "left")
    has_right = _has_cue(text, "right")
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def derive_view_from_prompt(prompt: str) -> Optional[str]:
    """Derive a ``SEMANTIC_VIEWS`` view name from a segment prompt, or ``None`` when the
    prompt carries no recognizable shot cue (the segment then INHERITS the movie-level
    DNA — exactly today's behavior). Pure + deterministic (no LLM). See the module
    docstring for the phrase table and the left/right defaulting rules.

    Conservative: only distinctive, orientation-bearing phrases fire; anything else is
    ``None``. The returned name is always a valid view (guarded against ``SEMANTIC_VIEWS``)
    so it resolves cleanly through ``azimuth_for_view`` downstream."""
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    text = prompt.lower()
    side = _side(text)

    view: Optional[str] = None
    if any(_has_cue(text, c) for c in _BACK_CUES):
        view = "back"
    elif any(_has_cue(text, c) for c in _OVER_SHOULDER_CUES):
        # a back-quarter framing; default back-right when the prompt names no side.
        view = "back-left" if side == "left" else "back-right"
    elif any(_has_cue(text, c) for c in _PROFILE_CUES):
        view = "left-profile" if side == "left" else "right-profile"
    elif any(_has_cue(text, c) for c in _THREE_QUARTER_CUES):
        view = "three-quarter-left" if side == "left" else "three-quarter-right"

    # Output guard: never emit a name the resolver would reject.
    return view if view in SEMANTIC_VIEWS else None
