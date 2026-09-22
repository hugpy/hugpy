"""Disk-authoritative store binding hugpy models to Discord channels/users.

Mirrors WorkerStore (functions.imports.utils.workers): a single JSON file beside
the model manifest is the source of truth; every mutation takes an exclusive
``fcntl`` lock, reloads, mutates, and writes back atomically, with a short read
cache so console polls don't hammer the mount.

Two consumers:
  * the console writes/lists bindings (model_key <-> a Discord channel and/or
    user) and can enqueue outbound messages;
  * the hugpy bot reads bindings to route an inbound message to the right model,
    and drains the outbox to push model-originated messages into a channel
    ("the model's mobile arm").
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

try:
    import fcntl  # POSIX advisory file locks — cross-process coordination.
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from hugpy_server.state import server_state_path


def _default_path() -> str:
    """Sit the binding registry next to the model manifest (…/projects/)."""
    return server_state_path("discord_bindings.json")


def _norm_id(value: Any) -> Optional[str]:
    """Discord snowflake ids are 64-bit; carry them as strings so neither JSON
    nor JS rounds them. Accept int/str, return a non-empty string or None."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _public_binding(b: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": b.get("id"),
        "model_key": b.get("model_key"),
        "channel_id": b.get("channel_id"),
        "user_id": b.get("user_id"),
        "label": b.get("label") or "",
        "enabled": b.get("enabled", True),
        "created_at": b.get("created_at"),
    }


class DiscordBindingStore:
    """Multi-process-safe registry of model<->Discord bindings + an outbox.

    Same rationale as WorkerStore: under gunicorn the API is several processes,
    so an in-memory dict would split-brain. ``discord_bindings.json`` is the one
    source of truth; reads serve from a short cache, writes take an exclusive
    ``fcntl`` lock and refresh the cache.
    """

    _READ_TTL = 3.0
    _OUTBOX_KEEP = 200  # trim delivered tail so the file can't grow unbounded

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or _default_path()
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_at = 0.0
        self._ensure_parent()

    # -- persistence (disk-authoritative) ----------------------------------
    def _ensure_parent(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"bindings": [], "outbox": [], "channels": [], "channels_at": 0,
                "users": [], "users_at": 0, "bridges": [], "bridge_msgs": {},
                "sessions": []}

    def _read_unlocked(self, fh=None) -> Dict[str, Any]:
        """Parse the bindings doc. A non-empty but unparseable file is CORRUPTION:
        log and re-raise rather than return an empty doc — otherwise a torn write
        would be silently healed into empty bindings and persisted, dropping every
        channel/user/bridge. Absent/empty files still return empty (normal start).
        """
        try:
            if fh is not None:
                fh.seek(0)
                raw = fh.read()
            elif os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = f.read()
            else:
                return self._empty()
        except OSError:
            return self._empty()
        if not raw.strip():
            return self._empty()
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("discord-bindings root is not a JSON object")
            data.setdefault("bindings", [])
            data.setdefault("outbox", [])
            data.setdefault("channels", [])
            data.setdefault("channels_at", 0)
            data.setdefault("users", [])
            data.setdefault("users_at", 0)
            data.setdefault("bridges", [])
            data.setdefault("bridge_msgs", {})
            data.setdefault("sessions", [])
            return data
        except ValueError as exc:
            import logging as _logging
            _logging.getLogger(__name__).error(
                "discord-bindings store %s is unparseable (%d bytes) — refusing to "
                "treat as empty; leaving the file intact for recovery (%s)",
                self._path, len(raw), exc,
            )
            raise

    def _write_unlocked(self, fh, doc: Dict[str, Any]) -> None:
        payload = json.dumps(doc, indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    def _load(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            if self._cache is not None and (now - self._cache_at) < self._READ_TTL:
                return self._cache
            try:
                data = self._read_unlocked()
            except ValueError:
                # Corrupt on-disk file: serve the last good snapshot rather than
                # crash reads (the error is already logged).
                if self._cache is not None:
                    return self._cache
                raise
            self._cache = data
            self._cache_at = now
            return data

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._ensure_parent()
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
            fh = os.fdopen(fd, "r+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                doc = self._read_unlocked(fh)
                yield doc
                self._write_unlocked(fh, doc)
                self._cache = doc
                self._cache_at = time.time()
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()

    # -- bindings ----------------------------------------------------------
    def list_bindings(self) -> List[Dict[str, Any]]:
        return [_public_binding(b) for b in self._load().get("bindings", [])]

    def add_binding(self, *, model_key: str, channel_id=None, user_id=None,
                    label: Optional[str] = None) -> Dict[str, Any]:
        channel_id = _norm_id(channel_id)
        user_id = _norm_id(user_id)
        if channel_id is None and user_id is None:
            raise ValueError("a binding needs a channel_id and/or a user_id")
        binding = {
            "id": uuid.uuid4().hex,
            "model_key": model_key,
            "channel_id": channel_id,
            "user_id": user_id,
            "label": (label or "").strip(),
            "enabled": True,
            "created_at": time.time(),
        }
        with self._transaction() as doc:
            doc["bindings"].append(binding)
        return _public_binding(binding)

    def remove_binding(self, binding_id: str) -> bool:
        with self._transaction() as doc:
            before = len(doc["bindings"])
            doc["bindings"] = [b for b in doc["bindings"] if b.get("id") != binding_id]
            return len(doc["bindings"]) < before

    def resolve(self, channel_id=None, user_id=None) -> Optional[str]:
        """Most specific enabled binding wins: (channel AND user) > channel > user."""
        channel_id = _norm_id(channel_id)
        user_id = _norm_id(user_id)
        bindings = [b for b in self._load().get("bindings", []) if b.get("enabled", True)]

        if channel_id and user_id:
            for b in bindings:
                if b.get("channel_id") == channel_id and b.get("user_id") == user_id:
                    return b.get("model_key")
        if channel_id:
            for b in bindings:
                if b.get("channel_id") == channel_id and not b.get("user_id"):
                    return b.get("model_key")
        if user_id:
            for b in bindings:
                if b.get("user_id") == user_id and not b.get("channel_id"):
                    return b.get("model_key")
        return None

    # -- outbox (model -> channel push) ------------------------------------
    def enqueue_outbound(self, *, content: str, channel_id=None, user_id=None,
                         binding_id: Optional[str] = None,
                         options: Optional[list] = None) -> Dict[str, Any]:
        channel_id = _norm_id(channel_id)
        user_id = _norm_id(user_id)
        with self._transaction() as doc:
            if binding_id and channel_id is None and user_id is None:
                for b in doc["bindings"]:
                    if b.get("id") == binding_id:
                        channel_id = b.get("channel_id")
                        user_id = b.get("user_id")
                        break
            if channel_id is None and user_id is None:
                raise ValueError("outbound needs a channel_id, user_id, or a known binding_id")
            msg = {
                "id": uuid.uuid4().hex,
                "binding_id": binding_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "content": content,
                "created_at": time.time(),
                "delivered": False,
            }
            # Only stamp `options` when there's a non-empty list, so an ordinary
            # outbound keeps its exact historical shape (no `options` key) and the
            # bot's plain-text delivery path is unaffected. When present, the bot
            # renders these labels as clickable choices (buttons ≤5 / select >5).
            if isinstance(options, list) and options:
                msg["options"] = options
            doc["outbox"].append(msg)
        return dict(msg)

    def drain_outbound(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Hand out undelivered messages once (at-most-once) and mark them
        delivered. The bot calls this on a poll loop; a crash between drain and
        Discord-send drops that message — acceptable for a notify channel."""
        out: List[Dict[str, Any]] = []
        with self._transaction() as doc:
            for m in doc["outbox"]:
                if not m.get("delivered"):
                    m["delivered"] = True
                    m["delivered_at"] = time.time()
                    out.append(dict(m))
                    if len(out) >= limit:
                        break
            if len(doc["outbox"]) > self._OUTBOX_KEEP:
                doc["outbox"] = doc["outbox"][-self._OUTBOX_KEEP:]
        return out

    # -- channel snapshot (bot reports the channels it can see) -------------
    def set_channels(self, channels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Replace the cached list of channels the bot can see. Each entry is
        normalised to {id, name, guild, guild_id}; ids carried as strings."""
        clean: List[Dict[str, Any]] = []
        for c in channels or []:
            cid = _norm_id(c.get("id"))
            if not cid:
                continue
            clean.append({
                "id": cid,
                "name": str(c.get("name") or "").strip() or cid,
                "guild": str(c.get("guild") or "").strip(),
                "guild_id": _norm_id(c.get("guild_id")),
            })
        clean.sort(key=lambda c: (c["guild"].lower(), c["name"].lower()))
        with self._transaction() as doc:
            doc["channels"] = clean
            doc["channels_at"] = time.time()
        return {"count": len(clean), "channels_at": doc["channels_at"]}

    def get_channels(self) -> Dict[str, Any]:
        data = self._load()
        return {"channels": data.get("channels", []),
                "channels_at": data.get("channels_at", 0)}

    def set_users(self, users: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Replace the cached list of guild members the bot can see. Each entry
        is normalised to {id, name, guild}; ids carried as strings."""
        clean: List[Dict[str, Any]] = []
        seen = set()
        for u in users or []:
            uid = _norm_id(u.get("id"))
            if not uid or uid in seen:
                continue
            seen.add(uid)
            clean.append({
                "id": uid,
                "name": str(u.get("name") or "").strip() or uid,
                "guild": str(u.get("guild") or "").strip(),
            })
        clean.sort(key=lambda u: u["name"].lower())
        with self._transaction() as doc:
            doc["users"] = clean
            doc["users_at"] = time.time()
        return {"count": len(clean), "users_at": doc["users_at"]}

    def get_users(self) -> Dict[str, Any]:
        data = self._load()
        return {"users": data.get("users", []),
                "users_at": data.get("users_at", 0)}

    # -- bridges (a console session <-> a Discord channel) ------------------
    DEFER_MODES = ("auto", "defer", "directive")
    _MSGS_KEEP = 200  # cap transcript length per bridge

    BRAINS = ("model", "keeper")
    # log_mode: "open" -> retain the transcript (default; required by keeper,
    #   defer/approval, and session-token bridges, which all READ it to work);
    #   "none" -> retain nothing (ephemeral) — only valid for an auto model
    #   bridge, which generates its reply from the current inbound alone.
    LOG_MODES = ("open", "none")

    def add_bridge(self, *, channel_id, model_key=None, user_id=None,
                   directive: str = "", defer_mode: str = "auto",
                   brain: str = "model",
                   keeper_target: Optional[str] = None,
                   log_mode: str = "open") -> Dict[str, Any]:
        channel_id = _norm_id(channel_id)
        if channel_id is None:
            raise ValueError("a bridge needs a channel_id")
        if defer_mode not in self.DEFER_MODES:
            raise ValueError(f"defer_mode must be one of {self.DEFER_MODES}")
        if brain not in self.BRAINS:
            raise ValueError(f"brain must be one of {self.BRAINS}")
        if log_mode not in self.LOG_MODES:
            raise ValueError(f"log_mode must be one of {self.LOG_MODES}")
        if log_mode == "none" and (brain != "model" or defer_mode != "auto"):
            # A no-logs bridge stores nothing, so any consumer that reads the
            # transcript would break: keeper brains poll it, defer/approval holds
            # pending candidates in it, and session tokens read it. Guard here so
            # the incompatible combination can't be created.
            raise ValueError("no-logs is only allowed for a model bridge in auto "
                             "mode — keeper/defer/session bridges need a retained "
                             "transcript to function")
        bridge = {
            "id": uuid.uuid4().hex,
            "channel_id": channel_id,
            "user_id": _norm_id(user_id),
            "model_key": model_key,
            "directive": (directive or "").strip(),
            "defer_mode": defer_mode,
            # brain: "model" -> central auto-generates a reply on inbound;
            #        "keeper" -> an attached keeper process drives replies and
            #        central only records inbound (no auto-candidate).
            "brain": brain,
            "keeper_target": (keeper_target or "").strip() or None,
            "log_mode": log_mode,
            "created_at": time.time(),
        }
        with self._transaction() as doc:
            # one bridge per channel — replace any existing for this channel
            doc["bridges"] = [b for b in doc["bridges"] if b.get("channel_id") != channel_id]
            doc["bridges"].append(bridge)
        return dict(bridge)

    def list_bridges(self) -> List[Dict[str, Any]]:
        return [dict(b) for b in self._load().get("bridges", [])]

    def get_bridge(self, bridge_id: str) -> Optional[Dict[str, Any]]:
        for b in self._load().get("bridges", []):
            if b.get("id") == bridge_id:
                return dict(b)
        return None

    def remove_bridge(self, bridge_id: str) -> bool:
        with self._transaction() as doc:
            before = len(doc["bridges"])
            doc["bridges"] = [b for b in doc["bridges"] if b.get("id") != bridge_id]
            doc.get("bridge_msgs", {}).pop(bridge_id, None)
            return len(doc["bridges"]) < before

    def bridge_for_channel(self, channel_id) -> Optional[Dict[str, Any]]:
        channel_id = _norm_id(channel_id)
        for b in self._load().get("bridges", []):
            if b.get("channel_id") == channel_id:
                return dict(b)
        return None

    def bridged_channel_ids(self) -> List[str]:
        return [b["channel_id"] for b in self._load().get("bridges", []) if b.get("channel_id")]

    def append_message(self, bridge_id: str, *, direction: str, source: str,
                       content: str, author: Optional[str] = None,
                       status: str = "sent",
                       attachments: Optional[List[Dict[str, Any]]] = None,
                       options: Optional[list] = None) -> Optional[Dict[str, Any]]:
        msg = {
            "id": uuid.uuid4().hex,
            "bridge_id": bridge_id,
            "direction": direction,   # "in" (toward console) | "out" (toward Discord)
            "source": source,         # "discord" | "console" | "terminal" | "model"
            "author": author or "",
            "content": content,
            # each: {filename, content_type, size, url, path?} — path is the
            # central-uploaded copy (keeper-readable); url is the Discord fallback.
            "attachments": attachments or [],
            "status": status,         # "sent" | "pending" (awaiting operator) | "rejected"
            "ts": time.time(),
        }
        # Interactive-escalation nicety: when the outbound offered clickable
        # choices, mirror the labels into the transcript so the console shows
        # what was asked. Only stamped when non-empty, so an ordinary message
        # keeps its exact historical shape (no `options` key).
        if isinstance(options, list) and options:
            msg["options"] = options
        with self._transaction() as doc:
            bridge = next((b for b in doc["bridges"] if b.get("id") == bridge_id), None)
            if bridge is None:
                return None
            if bridge.get("log_mode", "open") == "none":
                # Ephemeral bridge: the message transits (caller still gets it for
                # the immediate reply flow) but nothing is retained.
                return dict(msg)
            log = doc.setdefault("bridge_msgs", {}).setdefault(bridge_id, [])
            log.append(msg)
            if len(log) > self._MSGS_KEEP:
                doc["bridge_msgs"][bridge_id] = log[-self._MSGS_KEEP:]
        return dict(msg)

    def get_messages(self, bridge_id: str, since: float = 0.0) -> List[Dict[str, Any]]:
        log = self._load().get("bridge_msgs", {}).get(bridge_id, [])
        return [dict(m) for m in log if m.get("ts", 0) > since]

    def update_message(self, bridge_id: str, msg_id: str, *, status: Optional[str] = None,
                       content: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._transaction() as doc:
            for m in doc.get("bridge_msgs", {}).get(bridge_id, []):
                if m.get("id") == msg_id:
                    if status is not None:
                        m["status"] = status
                    if content is not None:
                        m["content"] = content
                    return dict(m)
        return None

    def clear_messages(self, bridge_id: str) -> Optional[int]:
        """Wipe a bridge's stored transcript. Returns how many messages were
        removed, or None if the bridge is unknown. The bridge and its Discord
        binding are untouched — only the console-side history is cleared (which
        also resets the model reply context for model-brained bridges, since
        _generate_candidate rebuilds from this transcript). Does NOT delete
        anything from the Discord channel itself."""
        with self._transaction() as doc:
            if not any(b.get("id") == bridge_id for b in doc["bridges"]):
                return None
            removed = len(doc.get("bridge_msgs", {}).get(bridge_id, []))
            doc.setdefault("bridge_msgs", {})[bridge_id] = []
            return removed

    # -- comms sessions (scoped bearer tokens for terminal agents) -----------
    # A session is the "drop this into a terminal/agent chat" credential: its
    # holder can read one channel's bridge transcript and send into that
    # channel — nothing else on the API. Only the sha256 of the token is
    # stored, so the registry file never contains a usable secret.
    _SESSION_TOKEN_BYTES = 32

    def add_session(self, *, channel_id, label: str = "",
                    ttl_hours: Optional[float] = None,
                    author: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Mint a session; returns (raw_token, public_view). The raw token is
        shown exactly once — it is not recoverable from the store."""
        channel_id = _norm_id(channel_id)
        if channel_id is None:
            raise ValueError("a session needs a channel_id")
        token = secrets.token_urlsafe(self._SESSION_TOKEN_BYTES)
        session = {
            "id": uuid.uuid4().hex,
            "channel_id": channel_id,
            "label": (label or "").strip(),
            "author": (author or "").strip() or None,
            "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "created_at": time.time(),
            "expires_at": (time.time() + float(ttl_hours) * 3600.0) if ttl_hours else None,
            "revoked": False,
            "last_used": None,
        }
        with self._transaction() as doc:
            doc.setdefault("sessions", []).append(session)
        return token, _public_session(session)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [_public_session(s) for s in self._load().get("sessions", [])]

    def revoke_session(self, session_id: str) -> bool:
        with self._transaction() as doc:
            for s in doc.get("sessions", []):
                if s.get("id") == session_id and not s.get("revoked"):
                    s["revoked"] = True
                    return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """REMOVE a session row from the ledger (2026-08-13 dead-weight
        cleanup). A still-live session dies with its row — the token's hash
        goes, so the bearer stops verifying — which is revoke-and-erase in one
        step; dead rows just leave. False only for unknown ids."""
        with self._transaction() as doc:
            sessions = doc.get("sessions", [])
            for i, s in enumerate(sessions):
                if s.get("id") == session_id:
                    del sessions[i]
                    return True
        return False

    def prune_sessions(self) -> int:
        """Sweep every DEAD session row (revoked, or past expires_at) and
        return how many went. Live sessions are never touched — this is the
        ledger janitor, not a bulk revoke."""
        now = time.time()

        def _dead(s: Dict[str, Any]) -> bool:
            if s.get("revoked"):
                return True
            exp = s.get("expires_at")
            return bool(exp) and float(exp) < now

        with self._transaction() as doc:
            sessions = doc.get("sessions", [])
            keep = [s for s in sessions if not _dead(s)]
            n = len(sessions) - len(keep)
            if n:
                doc["sessions"] = keep
            return n

    def session_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Presented bearer token -> live session dict, or None (bad, revoked
        or expired — indistinguishable to the caller, no oracle). last_used is
        stamped at most once a minute so polling stays read-mostly."""
        if not token:
            return None
        digest = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
        now = time.time()
        for s in self._load().get("sessions", []):
            if not hmac.compare_digest(s.get("token_sha256") or "", digest):
                continue
            if s.get("revoked") or (s.get("expires_at") and now > s["expires_at"]):
                return None
            if (s.get("last_used") or 0) < now - 60.0:
                with self._transaction() as doc:
                    for t in doc.get("sessions", []):
                        if t.get("id") == s.get("id"):
                            t["last_used"] = now
            return dict(s)
        return None


def _public_session(s: Dict[str, Any]) -> Dict[str, Any]:
    """Everything about a session except the token digest."""
    return {
        "id": s.get("id"),
        "channel_id": s.get("channel_id"),
        "label": s.get("label") or "",
        "author": s.get("author"),
        "created_at": s.get("created_at"),
        "expires_at": s.get("expires_at"),
        "revoked": s.get("revoked", False),
        "last_used": s.get("last_used"),
    }


discord_store = DiscordBindingStore()


def list_bindings() -> List[Dict[str, Any]]:
    return discord_store.list_bindings()


def add_binding(**kwargs) -> Dict[str, Any]:
    return discord_store.add_binding(**kwargs)


def remove_binding(binding_id: str) -> bool:
    return discord_store.remove_binding(binding_id)


def resolve_model(channel_id=None, user_id=None) -> Optional[str]:
    return discord_store.resolve(channel_id, user_id)


def enqueue_outbound(**kwargs) -> Dict[str, Any]:
    return discord_store.enqueue_outbound(**kwargs)


def drain_outbound(limit: int = 50) -> List[Dict[str, Any]]:
    return discord_store.drain_outbound(limit)


def set_channels(channels: List[Dict[str, Any]]) -> Dict[str, Any]:
    return discord_store.set_channels(channels)


def get_channels() -> Dict[str, Any]:
    return discord_store.get_channels()


def set_users(users: List[Dict[str, Any]]) -> Dict[str, Any]:
    return discord_store.set_users(users)


def get_users() -> Dict[str, Any]:
    return discord_store.get_users()


def add_bridge(**kwargs) -> Dict[str, Any]:
    return discord_store.add_bridge(**kwargs)


def list_bridges() -> List[Dict[str, Any]]:
    return discord_store.list_bridges()


def get_bridge(bridge_id: str) -> Optional[Dict[str, Any]]:
    return discord_store.get_bridge(bridge_id)


def remove_bridge(bridge_id: str) -> bool:
    return discord_store.remove_bridge(bridge_id)


def bridge_for_channel(channel_id) -> Optional[Dict[str, Any]]:
    return discord_store.bridge_for_channel(channel_id)


def bridged_channel_ids() -> List[str]:
    return discord_store.bridged_channel_ids()


def append_bridge_message(bridge_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    return discord_store.append_message(bridge_id, **kwargs)


def get_bridge_messages(bridge_id: str, since: float = 0.0) -> List[Dict[str, Any]]:
    return discord_store.get_messages(bridge_id, since)


def update_bridge_message(bridge_id: str, msg_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    return discord_store.update_message(bridge_id, msg_id, **kwargs)


def clear_bridge_messages(bridge_id: str) -> Optional[int]:
    return discord_store.clear_messages(bridge_id)


def add_session(**kwargs) -> Tuple[str, Dict[str, Any]]:
    return discord_store.add_session(**kwargs)


def list_sessions() -> List[Dict[str, Any]]:
    return discord_store.list_sessions()


def revoke_session(session_id: str) -> bool:
    return discord_store.revoke_session(session_id)


def delete_session(session_id: str) -> bool:
    return discord_store.delete_session(session_id)


def prune_sessions() -> int:
    return discord_store.prune_sessions()


def session_by_token(token: str) -> Optional[Dict[str, Any]]:
    return discord_store.session_by_token(token)
