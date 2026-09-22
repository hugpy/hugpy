"""Worker enrollment tokens.

A *single bottleneck* for the worker fleet: a machine may only register with this
central if it presents a valid, un-revoked enrollment token (when
``HUGPY_WORKER_ENROLL_REQUIRED`` is on — see ``workers.enroll_required``). The
console issues a token, the install script bakes ``{central, token}`` into the
worker's unit, and the agent sends it as ``Authorization: Bearer`` on every
register/heartbeat. Revoking a token is the hard eviction: the worker's next
contact is refused and its agent stops instead of respawning.

Tokens are ``hpw_<hex>`` and are shown **once** at creation; only their SHA-256
hash is persisted, beside the worker registry and model manifest. The on-disk
shape mirrors ``workers.py`` (disk-authoritative, fcntl-locked) so multiple
gunicorn workers stay consistent.
"""
from __future__ import annotations

import os
import json
import time
import uuid
import hashlib
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

try:
    import fcntl  # POSIX advisory file locks — cross-process coordination.
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from hugpy_fleet.central.config import settings

# EMFILE burst hardening (incident 2026-07-23): this store lives on the virtiofs
# mount (beside workers.json / the manifest). On a restart all gunicorn workers
# open it at once and the mount momentarily throws EMFILE — which surfaced here
# as a 500 on POST /llm/workers/register (worker 'ae' logged "initial
# registration failed: HTTP 500"). Retrying the file-open past the transient
# lets the burst settle so the wrapped reader succeeds. Best-effort import so a
# packaging skew can never break the token store.
try:
    from hugpy_control.shared import retry_on_emfile
except Exception:  # pragma: no cover - degrade to un-retried open on import skew
    def retry_on_emfile(fn, **_kw):  # type: ignore[misc]
        return fn()

_TOKEN_PREFIX = "hpw_"


def _default_tokens_path() -> str:
    """Sit the token store next to the model manifest / workers.json."""
    return os.path.join(settings.state_dir, "enrollment_tokens.json")


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class EnrollmentTokenStore:
    """Disk-authoritative, multi-process-safe store of enrollment tokens."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or _default_tokens_path()
        self._lock = threading.RLock()
        self._ensure_parent()

    def _ensure_parent(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass

    def _read_unlocked(self, fh=None) -> Dict[str, Dict[str, Any]]:
        """Parse the tokens map. A non-empty but unparseable file is CORRUPTION:
        log and re-raise rather than return {} — otherwise a torn write would be
        silently healed into an empty token set and persisted, revoking every
        enrollment token. Absent/empty files still return {} (normal cold start).
        """
        try:
            if fh is not None:
                fh.seek(0)
                raw = fh.read()
            elif os.path.exists(self._path):
                # Retry the open past a restart-burst EMFILE (the read itself is
                # cheap once the handle exists).
                with retry_on_emfile(
                    lambda: open(self._path, "r", encoding="utf-8")
                ) as f:
                    raw = f.read()
            else:
                return {}
        except OSError:
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("enrollment-tokens registry root is not a JSON object")
            return {t["id"]: t for t in data.get("tokens", []) if t.get("id")}
        except (ValueError, KeyError) as exc:
            import logging as _logging
            _logging.getLogger(__name__).error(
                "enrollment-tokens registry %s is unparseable (%d bytes) — refusing to "
                "treat as empty; leaving the file intact for recovery (%s)",
                self._path, len(raw), exc,
            )
            raise

    def _write_unlocked(self, fh, tokens: Dict[str, Dict[str, Any]]) -> None:
        payload = json.dumps({"tokens": list(tokens.values())}, indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._ensure_parent()
            # verify_enrollment_token() (the register/heartbeat auth gate) goes
            # through here to stamp last_used — so a restart-burst EMFILE on this
            # open is exactly what 500'd registration. Retry the open past the
            # transient before building the buffered handle.
            fd = retry_on_emfile(
                lambda: os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
            )
            fh = os.fdopen(fd, "r+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                tokens = self._read_unlocked(fh)
                yield tokens
                self._write_unlocked(fh, tokens)
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()

    @staticmethod
    def _public_view(tok: Dict[str, Any]) -> Dict[str, Any]:
        """Safe shape for API callers — never includes the hash or plaintext."""
        return {
            "id": tok.get("id"),
            "label": tok.get("label"),
            "created_at": tok.get("created_at"),
            "revoked": bool(tok.get("revoked")),
            "last_used": tok.get("last_used"),
        }

    # -- lifecycle ----------------------------------------------------------
    def create(self, label: str = "") -> Dict[str, Any]:
        """Mint a token. Returns the public view PLUS the one-time ``token``."""
        plaintext = _TOKEN_PREFIX + uuid.uuid4().hex + uuid.uuid4().hex
        tid = uuid.uuid4().hex[:12]
        rec = {
            "id": tid,
            "label": label or "",
            "hash": _hash(plaintext),
            "created_at": time.time(),
            "revoked": False,
            "last_used": None,
        }
        with self._transaction() as tokens:
            tokens[tid] = rec
        view = self._public_view(rec)
        view["token"] = plaintext  # shown once; never stored or returned again
        return view

    def verify(self, plaintext: Optional[str]) -> bool:
        """True iff ``plaintext`` matches a stored, un-revoked token.

        Records ``last_used`` on a hit (best-effort). Constant-ish time over the
        token set is acceptable here — the set is tiny (operator-issued).
        """
        if not plaintext or not plaintext.startswith(_TOKEN_PREFIX):
            return False
        h = _hash(plaintext)
        with self._transaction() as tokens:
            for tok in tokens.values():
                if tok.get("hash") == h and not tok.get("revoked"):
                    tok["last_used"] = time.time()
                    return True
        return False

    def revoke(self, token_id: str) -> bool:
        with self._transaction() as tokens:
            tok = tokens.get(token_id)
            if tok is None:
                return False
            tok["revoked"] = True
            return True

    def all(self) -> List[Dict[str, Any]]:
        with self._transaction() as tokens:
            return [self._public_view(t) for t in tokens.values()]


token_store = EnrollmentTokenStore()


# Module-level convenience wrappers (mirrors workers.py style).
def create_enrollment_token(label: str = "") -> Dict[str, Any]:
    return token_store.create(label)


def verify_enrollment_token(plaintext: Optional[str]) -> bool:
    return token_store.verify(plaintext)


def revoke_enrollment_token(token_id: str) -> bool:
    return token_store.revoke(token_id)


def list_enrollment_tokens() -> List[Dict[str, Any]]:
    return token_store.all()
