"""Tiny JSON-backed preference store (per-user default model, etc.)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hugpy_discord.config import DATA_DIR


class PrefsStore:
    def __init__(self, path: Path | None = None):
        self._path = path or DATA_DIR / "prefs.json"
        self._lock = asyncio.Lock()
        try:
            self._data: dict = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def get_model(self, user_id: int) -> str | None:
        return self._data.get("models", {}).get(str(user_id))

    async def set_model(self, user_id: int, model_key: str | None) -> None:
        async with self._lock:
            models = self._data.setdefault("models", {})
            if model_key is None:
                models.pop(str(user_id), None)
            else:
                models[str(user_id)] = model_key
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))
