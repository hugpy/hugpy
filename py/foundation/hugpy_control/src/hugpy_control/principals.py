"""F2 — one Principal model over the four identity systems.

Before this module the tree had four separate identity notions, none aware
of the others: the operator gate (env token / external cookie), API keys
(api_keys.json), worker enrollment tokens (enrollment_tokens.json), and
Discord snowflakes riding through bindings. Every gate answered "may this
request do X" with its own bespoke check.

This store gives them one vocabulary:

    Principal — id, kind (operator | user | worker | discord | ephemeral |
    service), display name, groups, expiry/revocation, and LINKS to the
    legacy identities (api_key_id, worker_id, discord_user_id, session_id).

The legacy stores stay authoritative for their own verification (an API key
still verifies in api_keys.json) — a Principal WRAPS them, it does not
replace them. resolve_* helpers map a verified legacy credential to its
Principal (creating lightweight implicit ones where none was issued), so
job attribution (Job.principal) and group checks work everywhere today
while surfaces migrate at their own pace.

Principal tokens (``hpp_`` + hex, sha256-hashed at rest, like the other
token stores) are the NEW credential: issued from the console, they carry
the principal directly — this is what DISC-05's in-Discord /link command
consumes, and what future surfaces (CLI, clients) authenticate with.

Groups, not per-surface flags: authorization asks ``allowed(principal,
action)`` against the ACTIONS policy below. Operators pass everything;
everyone else needs a group the action lists. Keep actions coarse — this is
a gate vocabulary, not an ACL engine.

Storage: ``$PROJECTS_HOME/principals.json`` (next to api_keys.json et al.),
fcntl-locked read-modify-write like enrollment_tokens.py — the established
multi-process-safe pattern in this tree. Stdlib-only, like all of comms.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

try:
    import fcntl
except ImportError:          # non-POSIX dev box; single-process fallback
    fcntl = None

KINDS = ("operator", "user", "worker", "discord", "ephemeral", "service")

# action -> groups allowed (operators always pass). "*" = any authenticated
# principal. Coarse on purpose; refine when a real case demands it.
ACTIONS = {
    "chat": ("*",),
    "media": ("media", "api"),
    "media.admin": (),
    "models.manage": (),
    "workers.manage": (),
    "settings.write": (),
    "pip.install": (),
    "principals.manage": (),
    "upload": ("*",),
}

OPERATORS_GROUP = "operators"


@dataclass
class Principal:
    id: str
    kind: str = "user"
    name: str = ""
    groups: list = field(default_factory=list)
    # Links to the legacy identity that this principal wraps (any subset).
    api_key_id: Optional[str] = None
    worker_id: Optional[str] = None
    discord_user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    revoked: bool = False

    @property
    def active(self) -> bool:
        if self.revoked:
            return False
        if self.expires_at is not None and time.time() > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Principal":
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        known.setdefault("id", d.get("id") or "")
        p = cls(**known)
        p.groups = list(p.groups or [])
        return p


def allowed(principal: Optional[Principal], action: str) -> bool:
    """The one authorization question. Unknown actions fail closed for
    everyone but operators."""
    if principal is None or not principal.active:
        return False
    groups = set(principal.groups or [])
    if OPERATORS_GROUP in groups:
        return True
    permitted = ACTIONS.get(action)
    if permitted is None:
        return False
    if "*" in permitted:
        return True
    return bool(groups.intersection(permitted))


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PrincipalStore:
    """Disk-authoritative, fcntl-locked, same shape as the sibling stores.

    File layout::

        {"principals": {id: {...Principal...}},
         "tokens":     {sha256: {"principal_id", "created_at", "last_used",
                                 "revoked"}}}
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._lock = threading.RLock()

    # -- location (resolved lazily so imports stay side-effect free) --------
    def path(self) -> str:
        if self._path:
            return self._path
        env = (os.environ.get("HUGPY_PRINCIPALS_PATH") or "").strip()
        if env:
            return env
        base = (os.environ.get("PROJECTS_HOME") or "").strip()
        if not base:
            try:
                from hugpy_platform.constants import PROJECTS_HOME as _PH
                base = str(_PH)
            except Exception:
                base = os.path.expanduser("~/.hugpy")
        return os.path.join(base, "principals.json")

    # -- locked read-modify-write -------------------------------------------
    def _empty(self) -> dict:
        return {"principals": {}, "tokens": {}}

    def _read(self, fh=None) -> dict:
        try:
            if fh is not None:
                fh.seek(0)
                raw = fh.read()
            else:
                with open(self.path(), "r", encoding="utf-8") as f:
                    raw = f.read()
        except FileNotFoundError:
            return self._empty()
        if not raw.strip():
            return self._empty()
        data = json.loads(raw)
        data.setdefault("principals", {})
        data.setdefault("tokens", {})
        return data

    def _transaction(self, mutate) -> Any:
        """mutate(data) -> result; data written back atomically under an
        exclusive lock. Same discipline as EnrollmentTokenStore."""
        with self._lock:
            path = self.path()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a+", encoding="utf-8") as fh:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    data = self._read(fh)
                    result = mutate(data)
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as out:
                        json.dump(data, out, indent=2)
                    os.replace(tmp, path)
                    return result
                finally:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # -- lifecycle -----------------------------------------------------------
    def create(self, *, kind: str = "user", name: str = "",
               groups: Optional[list] = None,
               expires_in: Optional[float] = None,
               **links: Any) -> Principal:
        if kind not in KINDS:
            raise ValueError(f"unknown principal kind: {kind!r}")
        p = Principal(id="pr_" + uuid.uuid4().hex[:12], kind=kind, name=name,
                      groups=list(groups or []))
        if expires_in:
            p.expires_at = time.time() + float(expires_in)
        for k, v in links.items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)

        def _mut(data):
            data["principals"][p.id] = p.to_dict()
            return p
        return self._transaction(_mut)

    def get(self, principal_id: str) -> Optional[Principal]:
        data = self._read()
        d = data["principals"].get(principal_id)
        return Principal.from_dict(d) if d else None

    def all(self) -> list[Principal]:
        data = self._read()
        return [Principal.from_dict(d) for d in data["principals"].values()]

    def update(self, principal_id: str, **changes: Any) -> Optional[Principal]:
        def _mut(data):
            d = data["principals"].get(principal_id)
            if d is None:
                return None
            for k, v in changes.items():
                if k in Principal.__dataclass_fields__ and k != "id":
                    d[k] = v
            return Principal.from_dict(d)
        return self._transaction(_mut)

    def revoke(self, principal_id: str) -> bool:
        return self.update(principal_id, revoked=True) is not None

    # -- principal tokens (the new credential) --------------------------------
    def issue_token(self, principal_id: str,
                    expires_in: Optional[float] = None) -> Optional[str]:
        """Mint a bearer token for a principal. Plaintext returned ONCE."""
        token = "hpp_" + secrets.token_hex(20)

        def _mut(data):
            if principal_id not in data["principals"]:
                return None
            data["tokens"][_hash(token)] = {
                "principal_id": principal_id,
                "created_at": time.time(),
                "expires_at": (time.time() + float(expires_in))
                              if expires_in else None,
                "last_used": None,
                "revoked": False,
            }
            return token
        return self._transaction(_mut)

    def resolve_token(self, token: Optional[str]) -> Optional[Principal]:
        """token -> active Principal, or None. Touches last_used."""
        if not token or not token.startswith("hpp_"):
            return None
        h = _hash(token)

        def _mut(data):
            rec = data["tokens"].get(h)
            if rec is None or rec.get("revoked"):
                return None
            exp = rec.get("expires_at")
            if exp is not None and time.time() > exp:
                return None
            rec["last_used"] = time.time()
            d = data["principals"].get(rec["principal_id"])
            if d is None:
                return None
            p = Principal.from_dict(d)
            return p if p.active else None
        return self._transaction(_mut)

    def revoke_token(self, token_hash_prefix: str) -> int:
        """Revoke stored tokens whose sha256 starts with the given prefix
        (console shows hash prefixes, never plaintext)."""
        def _mut(data):
            n = 0
            for h, rec in data["tokens"].items():
                if h.startswith(token_hash_prefix) and not rec.get("revoked"):
                    rec["revoked"] = True
                    n += 1
            return n
        return self._transaction(_mut)

    # -- discord linkage (DISC-05) --------------------------------------------
    def link_discord(self, token: str,
                     discord_user_id: str) -> Optional[Principal]:
        """The /link handshake: a user proves they hold a principal token by
        presenting it in Discord; we bind their snowflake to that principal.
        Never trust a raw snowflake — this is the only way one gets attached."""
        p = self.resolve_token(token)
        if p is None:
            return None
        return self.update(p.id, discord_user_id=str(discord_user_id))

    def for_discord_user(self, discord_user_id: str) -> Optional[Principal]:
        want = str(discord_user_id)
        for p in self.all():
            if p.discord_user_id == want and p.active:
                return p
        return None

    # -- legacy-credential resolution (implicit principals) -------------------
    def resolve_api_key(self, key_id: str, name: str = "") -> Principal:
        """A verified API key acts as a lightweight principal even if none
        was issued for it — attribution must never require a migration."""
        for p in self.all():
            if p.api_key_id == key_id and p.active:
                return p
        return Principal(id=f"apikey:{key_id}", kind="user",
                         name=name or f"api key {key_id}", groups=["api"])

    def resolve_operator(self) -> Principal:
        return Principal(id="operator", kind="operator", name="operator",
                         groups=[OPERATORS_GROUP])

    def resolve_worker(self, worker_id: str, name: str = "") -> Principal:
        for p in self.all():
            if p.worker_id == worker_id and p.active:
                return p
        return Principal(id=f"worker:{worker_id}", kind="worker",
                         name=name or worker_id, groups=["workers"])

    def resolve_session(self, session_id: str) -> Principal:
        return Principal(id=f"session:{session_id[:16]}", kind="ephemeral",
                         name="anonymous", groups=[], session_id=session_id)


principal_store = PrincipalStore()
