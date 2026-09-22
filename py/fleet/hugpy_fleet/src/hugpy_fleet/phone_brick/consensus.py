"""Plurality consensus across a chain of phones.

Each phone reports a top class for an image. The plurality (most common
non-"nodet" class) is the chain's verdict; every phone is then flagged relative
to it:

    AGR  agrees with the plurality
    DIS  detected something, but disagrees
    NOD  no detection ("nodet")
"""
from __future__ import annotations

from collections import Counter

AGREE = "AGR"
DISAGREE = "DIS"
NO_DETECTION = "NOD"
NODET = "nodet"


def plurality_consensus(top_by_phone: dict[str, str]) -> dict[str, str]:
    """Map ``{phone_name: top_cls}`` to ``{phone_name: AGR|DIS|NOD}``."""
    detected = [cls for cls in top_by_phone.values()
                if cls and cls.lower() != NODET]
    plurality = Counter(detected).most_common(1)[0][0] if detected else None

    flags: dict[str, str] = {}
    for phone, cls in top_by_phone.items():
        if not cls or cls.lower() == NODET:
            flags[phone] = NO_DETECTION
        elif plurality is not None and cls == plurality:
            flags[phone] = AGREE
        else:
            flags[phone] = DISAGREE
    return flags
