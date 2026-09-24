"""NO-THINK helpers shared with the inference engine.

The bot uses the platform's pure text helpers without depending on the engine.
"""
from __future__ import annotations

from hugpy_platform.no_think import (
    NO_THINK_DIRECTIVE, THINK_BLOCK_RE, strip_think, with_no_think,
)

__all__ = ["NO_THINK_DIRECTIVE", "THINK_BLOCK_RE", "strip_think", "with_no_think"]
