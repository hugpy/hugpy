"""F4 — one runtime-settings store, one control API surface.

The boundary this codifies: **env files are the deploy-time source of truth;
this store is the runtime source of truth.** Anything an operator flips
while the system runs — per-channel Discord modes, personalities, model
delegation, user gates — lives HERE, keyed by namespace, and every surface
reads it instead of keeping a private config file.

Existing single-purpose stores (serve_overrides.json, pruned_models.json,
media_models.json, workers.json) remain authoritative for their domains —
they already have the locking and routes they need. This store is for the
cross-surface settings that had no home (which is why bot/prefs.py grew a
private prefs.json — that migrates here) and it is the ONE place the console
writes runtime config through (CON-08).

Namespaces in use (create more freely; dots for hierarchy):

    discord.channels    {channel_id: {"respond": "mention"|"all",
                                      "personality": <name>|None}}
    discord.users       {user_id: {"model": <model_key>}}
    personalities       {name: {"system": str, "model_key": str|None,
                                "params": {...}}}       (DISC-06)
    delegation          {"channel:<id>"|"surface:<name>": <model_key>}
                                                          (CON-07)
    gates               {"media:user:<principal_id>": bool, ...}  (UTIL-03)

Storage: ``$PROJECTS_HOME/settings.json``, fcntl-locked read-modify-write
with a short read cache — the exact pattern workers.py proved out. Every
mutation publishes ``settings.changed`` on the comms bus (payload: ns, key)
so running services react without a restart; remote consumers (the bot)
poll the control API with a small TTL instead.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Optional

try:
    import fcntl
except ImportError:
    fcntl = None

_READ_TTL = 3.0


class SettingsStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._cache: Optional[dict] = None
        self._cache_at = 0.0
        # on_change(ns, key, value) — wired to the bus by wire_settings_events.
        self.on_change: Optional[Callable[[str, str, Any], None]] = None

    def path(self) -> str:
        if self._path:
            return self._path
        env = (os.environ.get("HUGPY_SETTINGS_PATH") or "").strip()
        if env:
            return env
        base = (os.environ.get("PROJECTS_HOME") or "").strip()
        if not base:
            try:
                from hugpy_platform.constants import PROJECTS_HOME as _PH
                base = str(_PH)
            except Exception:
                base = os.path.expanduser("~/.hugpy")
        return os.path.join(base, "settings.json")

    # -- io -------------------------------------------------------------------
    def _read_disk(self, fh=None) -> dict:
        try:
            if fh is not None:
                fh.seek(0)
                raw = fh.read()
            else:
                with open(self.path(), "r", encoding="utf-8") as f:
                    raw = f.read()
        except FileNotFoundError:
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Corrupt settings must not take the service down; loud + empty.
            import logging
            logging.getLogger(__name__).error(
                "settings.json unparseable at %s — treating as empty",
                self.path())
            return {}
        return data if isinstance(data, dict) else {}

    def _snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            if self._cache is not None and now - self._cache_at < _READ_TTL:
                return self._cache
            data = self._read_disk()
            self._cache, self._cache_at = data, now
            return data

    def _transaction(self, mutate) -> Any:
        with self._lock:
            path = self.path()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a+", encoding="utf-8") as fh:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    data = self._read_disk(fh)
                    result = mutate(data)
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as out:
                        json.dump(data, out, indent=2)
                    os.replace(tmp, path)
                    self._cache, self._cache_at = data, time.time()
                    return result
                finally:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # -- api --------------------------------------------------------------
    def get(self, ns: str, key: str, default: Any = None) -> Any:
        return self._snapshot().get(ns, {}).get(str(key), default)

    def all(self, ns: str) -> dict:
        return dict(self._snapshot().get(ns, {}))

    def namespaces(self) -> list[str]:
        return sorted(self._snapshot().keys())

    def set(self, ns: str, key: str, value: Any) -> Any:
        def _mut(data):
            data.setdefault(ns, {})[str(key)] = value
            return value
        result = self._transaction(_mut)
        self._emit(ns, str(key), value)
        return result

    def merge(self, ns: str, key: str, patch: dict) -> dict:
        """Shallow-merge a dict value (None values delete their field)."""
        def _mut(data):
            cur = data.setdefault(ns, {}).setdefault(str(key), {})
            if not isinstance(cur, dict):
                cur = {}
            for k, v in patch.items():
                if v is None:
                    cur.pop(k, None)
                else:
                    cur[k] = v
            data[ns][str(key)] = cur
            return cur
        result = self._transaction(_mut)
        self._emit(ns, str(key), result)
        return result

    def delete(self, ns: str, key: str) -> bool:
        def _mut(data):
            existed = str(key) in data.get(ns, {})
            data.get(ns, {}).pop(str(key), None)
            return existed
        result = self._transaction(_mut)
        if result:
            self._emit(ns, str(key), None)
        return result

    def increment(self, ns: str, key: str, by: int = 1) -> int:
        """Atomically add ``by`` to an integer counter and return the new value.

        Read-modify-write happens inside the same flock the other mutators use,
        so concurrent bumps from the request path never lose an increment (a
        plain get()+set() would race). Non-int / missing values start at 0.
        """
        def _mut(data):
            cur = data.setdefault(ns, {}).get(str(key), 0)
            if not isinstance(cur, int):
                cur = 0
            new = cur + int(by)
            data[ns][str(key)] = new
            return new
        result = self._transaction(_mut)
        self._emit(ns, str(key), result)
        return result

    def _emit(self, ns: str, key: str, value: Any) -> None:
        cb = self.on_change
        if cb is None:
            return
        try:
            cb(ns, key, value)
        except Exception:
            pass


settings_store = SettingsStore()


def wire_settings_events(the_bus=None, store: Optional[SettingsStore] = None,
                         source: Optional[str] = None) -> None:
    """Publish settings.changed on the comms bus for live in-process
    propagation (CON-08: changes take effect without a restart)."""
    from hugpy_control.bus import bus as _default_bus
    the_bus = the_bus or _default_bus
    store = store or settings_store

    def _on_change(ns: str, key: str, value: Any) -> None:
        the_bus.publish("settings.changed", source=source,
                        payload={"ns": ns, "key": key, "value": value})

    store.on_change = _on_change
