"""hugpy-discord: the Discord bot arm of the Hugpy ecosystem.

The bot is a pure HTTP client of a Hugpy central (``hugpy_platform.central``
resolves the base URL). ``discord.py`` and ``python-dotenv`` are the ``bot``
extra and are imported lazily: ``import hugpy_discord`` works without them, and
:func:`main` / :class:`HugpyBot` are resolved on first access.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "HugpyBot", "main"]


def __getattr__(name: str):
    if name in ("HugpyBot", "main"):
        from hugpy_discord import bot as _bot  # requires the ``bot`` extra
        return getattr(_bot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
