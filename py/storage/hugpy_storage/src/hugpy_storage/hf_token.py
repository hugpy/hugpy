"""Console-managed Hugging Face token store.

WHY
---
Central makes a lot of Hugging Face calls (model search, per-repo metadata /
size lookups, weight downloads). Made anonymously they get rate-limited, which
shows up as flaky search and stalled downloads. This lets the operator save an
HF token from the console once; every HF call then goes out authenticated.

STORAGE
-------
The token is persisted as a single 0600 file (``HF_TOKEN_PATH``) next to the
model manifest under ``PROJECTS_HOME`` — the same runtime state root that
``api_keys.json`` uses, and deliberately OUTSIDE any git tree. The token is
never logged. The canonical path + the low-level reader live in
``imports.src.constants.constants`` (so the constants layer can seed the process
env at import with no flask->constants dependency); THIS module owns writing,
validation against the HF API, and the apply-to-live-clients seam.

PRECEDENCE
----------
A STORED token wins. An env ``HF_TOKEN`` (operator/CI intent) is the fallback
and is reported as ``source: "env"``. DELETE removes only the stored file — an
env token, if present, remains in effect.
"""
from __future__ import annotations

import os
import threading

from hugpy_platform.constants import HF_TOKEN_PATH, read_stored_hf_token

_LOCK = threading.Lock()
_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"


# ── token resolution ────────────────────────────────────────────────────────
def _env_token() -> str | None:
    """The GENUINE operator-supplied env token captured at constants import —
    NOT live os.environ, which apply_hf_token_to_env() overwrites with the
    stored token (so reading it live would report a stored token as "env")."""
    from hugpy_platform import constants as _c
    tok = _c.HF_TOKEN_ENV
    return tok if tok else None


def _stored_token() -> str | None:
    tok = read_stored_hf_token()
    return tok or None


def get_hf_token() -> str | None:
    """The effective HF token: a stored token wins, env HF_TOKEN is the fallback.
    Returns None when neither is set. This is the single seam call sites use."""
    return _stored_token() or _env_token()


def token_source() -> str | None:
    """"stored" | "env" | None — where the effective token came from."""
    if _stored_token():
        return "stored"
    if _env_token():
        return "env"
    return None


def _last4(tok: str | None) -> str | None:
    return tok[-4:] if tok and len(tok) >= 4 else None


# ── HF validation (whoami-v2) ───────────────────────────────────────────────
def validate_hf_token(token: str, timeout: float = 5.0):
    """Call HF whoami-v2 with *token*.

    Returns ``(status, username, error)`` where status is:
      * "ok"      — valid token, ``username`` is the account name;
      * "invalid" — HF rejected the token (401/403), ``error`` is HF's message;
      * "network" — HF unreachable / timed out, ``error`` is the reason. The
        caller decides whether that is fatal (POST) or just unknown (GET).
    """
    import requests
    try:
        resp = requests.get(
            _WHOAMI_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return "network", None, f"could not reach Hugging Face: {exc}"
    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            data = {}
        return "ok", (data.get("name") or data.get("fullname")), None
    # HF returns a JSON {"error": "..."} on a bad token.
    msg = None
    try:
        msg = (resp.json() or {}).get("error")
    except ValueError:
        msg = None
    return "invalid", None, msg or f"Hugging Face rejected the token (HTTP {resp.status_code})"


# ── live-client apply seam ──────────────────────────────────────────────────
def apply_hf_token_to_env() -> None:
    """Push the effective token into the process env and rebuild the long-lived
    HfApi client built at import in constants, so a token saved/cleared at
    runtime takes effect without a process restart.

    Env is the broad seam: every huggingface_hub call that does not force
    ``token=False`` (bare ``hf_hub_download`` / ``snapshot_download`` /
    ``HfApi()``) resolves ``HF_TOKEN`` at call time. ``search_routes`` rebuilds
    its own module-level ``api`` in the route handler (it forces token=False for
    anonymous safety, so env alone would not reach it)."""
    tok = get_hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    else:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    try:
        from huggingface_hub import HfApi
        from hugpy_platform import constants as _c
        _c.HF_TOKEN = tok or False
        _c.hfApi = HfApi(token=tok or False)
    except Exception:
        # Best-effort: the env seam above already covers most call sites.
        pass
    # Upper layers that built their OWN long-lived HfApi at import (the
    # server's utils/constants.py) register a listener and rebuild there.
    for fn in list(_TOKEN_LISTENERS):
        try:
            fn(tok or None)
        except Exception:  # noqa: BLE001 — a listener must never block a token save
            pass


# ── token-change listeners ──────────────────────────────────────────────────
_TOKEN_LISTENERS: list = []


def add_token_listener(fn) -> None:
    """Register ``fn(token_or_None)`` to run after every apply (store/delete).
    Idempotent per callable."""
    if fn not in _TOKEN_LISTENERS:
        _TOKEN_LISTENERS.append(fn)


def remove_token_listener(fn) -> None:
    try:
        _TOKEN_LISTENERS.remove(fn)
    except ValueError:
        pass


# ── store mutations ─────────────────────────────────────────────────────────
def store_hf_token(token: str) -> None:
    """Persist *token* to a 0600 file, atomically. Caller validates first."""
    token = (token or "").strip()
    with _LOCK:
        parent = os.path.dirname(HF_TOKEN_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{HF_TOKEN_PATH}.{os.getpid()}.tmp"
        # Create with 0600 from the start so the secret is never briefly world-
        # readable between write and chmod.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(token)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, HF_TOKEN_PATH)
        os.chmod(HF_TOKEN_PATH, 0o600)
    apply_hf_token_to_env()


def delete_hf_token() -> bool:
    """Remove the stored token file. Returns True if a file was removed. An env
    token, if set, remains in effect afterwards."""
    with _LOCK:
        try:
            os.remove(HF_TOKEN_PATH)
            removed = True
        except OSError:
            removed = False
    apply_hf_token_to_env()
    return removed


# ── status shape (shared by GET / POST responses) ───────────────────────────
def hf_auth_status(validate: bool = True) -> dict:
    """The GET /llm/hf/auth response shape.

    With ``validate`` the effective token is checked against HF whoami-v2 so the
    console can show the real account name. On a network failure ``authenticated``
    is ``None`` (unknown) with a ``note`` — never a 500.
    """
    tok = get_hf_token()
    src = token_source()
    out: dict = {
        "authenticated": False,
        "username": None,
        "token_last4": _last4(tok),
        "source": src,
    }
    if not tok:
        return out
    if not validate:
        out["authenticated"] = None
        return out
    status, username, error = validate_hf_token(tok)
    if status == "ok":
        out["authenticated"] = True
        out["username"] = username
    elif status == "network":
        out["authenticated"] = None
        out["note"] = error
    else:  # invalid
        out["authenticated"] = False
        out["note"] = error
    return out
