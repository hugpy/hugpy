"""Transport-neutral model helpers: the model registry cache, label formatting,
key cleaning, and autocomplete choice pairs.

These are shared by the Discord cogs (which wrap the pairs in
``app_commands.Choice``) and the chatshare adapter (which returns them as
``{name, value}`` in ``autocomplete.result``). No transport imports here.
"""
from __future__ import annotations

import re
import time

from hugpy_discord.hugpy_client import HugpyError

_model_cache: tuple[float, list[dict]] = (0.0, [])
_MODEL_CACHE_TTL = 60.0


async def cached_models(bot) -> list[dict]:
    global _model_cache
    stamp, models = _model_cache
    if time.monotonic() - stamp > _MODEL_CACHE_TTL:
        try:
            models = await bot.hugpy.list_models()
            _model_cache = (time.monotonic(), models)
        except HugpyError:
            pass  # serve stale (or empty) rather than fail autocomplete
    return models


def model_label(model: dict) -> str:
    key = model.get("key") or model.get("name") or "?"
    status = (model.get("status") or "").strip()
    # Surface a status suffix ONLY when it's something other than the ordinary
    # "installed". An "(installed)" on every model is noise, and — because this
    # label is what users read in the dropdown / `/models` list — it collides
    # with the plain model key when the name is typed or copied as input.
    if status and status.lower() != "installed":
        return f"{key} ({status})"
    return key


_STATUS_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def clean_model_key(s: str | None) -> str | None:
    """Strip a trailing ' (status)' a user may have copied from a model label
    (valid model keys never end in a parenthetical), so 'Foo (installed)' still
    resolves to 'Foo'."""
    if not s:
        return s
    return _STATUS_SUFFIX.sub("", s).strip() or s


async def model_choice_pairs(bot, current: str) -> list[dict]:
    """Up to 25 ``{name, value}`` autocomplete pairs matching ``current``."""
    models = await cached_models(bot)
    current = (current or "").lower()
    choices: list[dict] = []
    for model in models:
        key = model.get("key") or model.get("name") or ""
        if current in key.lower():
            choices.append({"name": model_label(model)[:100], "value": key})
        if len(choices) == 25:
            break
    return choices


def split_items(raw: str) -> list[str]:
    """Split user input into items: '||' wins, else newlines, else one item."""
    sep = "||" if "||" in raw else "\n"
    return [part.strip() for part in raw.split(sep) if part.strip()]
