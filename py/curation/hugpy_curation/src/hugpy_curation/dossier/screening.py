"""k120 — the card's new screen knobs, computed with NO extra network.

``review/screen.py`` screens sixty candidates for the cost of one download, and
that property is worth protecting: the k120 card knobs that gate the screen
(``required_specializations``, ``licenses_allowed``) are therefore answered from
what the screen ALREADY has — the repo name, the tag list and the licence field
— and never by fetching a model card.

The card IS read later, once, for the two or three candidates that survive. A
specialization judged from tags and the repo name is coarser than one judged
from the README, and that is the correct trade at screen time: this is a
cheap filter deciding what deserves the expensive look, not the final word.
When it rejects, it says which evidence it had, so a false negative is
diagnosable from the row rather than from a re-run.

This module is the whole of k120's footprint inside the screen. ``screen.py``
calls :func:`extra_reasons` once and appends what it returns.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from hugpy_curation.dossier.cards import build_emphasis

#: A specialization must clear this weight to count as REQUIRED-satisfied at
#: screen time. 0.3 is exactly one tag hit or one strong card phrase — low
#: enough that a correctly-tagged repo passes, high enough that a coincidental
#: substring in a long repo name does not.
REQUIRED_WEIGHT_FLOOR: float = 0.3


def _knob(crit: Any, name: str, default: Any) -> Any:
    value = (crit.get(name, default) if isinstance(crit, Mapping)
             else getattr(crit, name, default))
    return default if value is None else value


def licence_allowed(license_id: str | None,
                    allowed: Sequence[str]) -> tuple[bool, str]:
    """``(ok, why)``. An EMPTY allow-list allows everything (the default, which
    is what keeps existing cards behaving exactly as before).

    Matching is substring, case-insensitive, both ways: a card asking for
    ``apache`` accepts ``apache-2.0``, and one asking for ``apache-2.0``
    accepts a repo that just says ``apache``. Licence strings on the hub are
    not a controlled vocabulary and an exact-match rule rejects half the fleet's
    own models."""
    if not allowed:
        return True, ""
    if not license_id:
        return False, (f"the repo declares no licence and this card only "
                       f"accepts {sorted(allowed)}")
    low = str(license_id).lower()
    for candidate in allowed:
        needle = str(candidate).lower()
        if needle in low or low in needle:
            return True, ""
    return False, (f"licence {license_id!r} is not in this card's "
                   f"licenses_allowed {sorted(allowed)}")


def specializations_met(hub_id: str, tags: Sequence[str],
                        pipeline_tag: str | None,
                        required: Sequence[str]) -> tuple[bool, str]:
    """``(ok, why)`` for ``required_specializations``, from tags + name only."""
    if not required:
        return True, ""
    weights = {e.domain: e.weight
               for e in build_emphasis(hub_id, tags, "", pipeline_tag)}
    missing = [d for d in required
               if weights.get(str(d).lower(), 0.0) < REQUIRED_WEIGHT_FLOOR]
    if not missing:
        return True, ""
    have = sorted(d for d, w in weights.items() if w >= REQUIRED_WEIGHT_FLOOR)
    return False, (f"specialization {missing} not evident from this repo's "
                   f"tags or name (found {have or 'none'}) — this card "
                   f"requires {list(required)}")


def extra_reasons(hub_id: str, crit: Any, license_id: str | None,
                  tags: Sequence[str],
                  pipeline_tag: str | None = None) -> list[str]:
    """Every k120 screen rejection for this candidate, or an empty list.

    Empty is the DEFAULT outcome: a card that sets neither knob gets no extra
    rules, which is the additivity guarantee this whole extension rests on."""
    reasons: list[str] = []
    ok, why = licence_allowed(license_id,
                              _knob(crit, "licenses_allowed", ()) or ())
    if not ok:
        reasons.append(why)
    ok, why = specializations_met(
        hub_id, tags or (), pipeline_tag,
        _knob(crit, "required_specializations", ()) or ())
    if not ok:
        reasons.append(why)
    return reasons


__all__ = ["REQUIRED_WEIGHT_FLOOR", "extra_reasons", "licence_allowed",
           "specializations_met"]
