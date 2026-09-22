"""One-time secure INSTALL LINKS for the hugpy-agent installer (2026-07-23).

WHAT THIS IS
------------
The operator mints a labeled, scoped, revocable install link from the console's
API tab. Fetching the link serves ``install_hugpy_agent.py`` with a freshly
minted API key baked into its ``EMBEDDED_API_KEY`` slot — so a new box installs
and enrolls with ONE line and the operator never handles (or sees) the raw key.

THE KEY IS NEVER A STANDING OPERATOR KEY. Each link mints its OWN key via
``api_keys.create_api_key`` (created_by="install-link", the operator's label,
the operator's chosen scopes — default ["v1"]) so it is individually
scoped/labeled/revocable like any other console key.

WHERE THE RAW KEY LIVES (read before touching)
----------------------------------------------
``api_keys`` stores only the sha256 of a key — by design it can never re-reveal
one. But an install link must template the RAW key into a download that happens
LATER than the mint. So the raw key is held here, in the link record, for the
link's lifetime only, and is SCRUBBED (overwritten with "") the moment the link
can no longer serve it:
  * on the download that exhausts the last use,
  * on revoke,
  * lazily on first touch after expiry.
This store file therefore holds live secrets while links are active — the same
exposure class as api_keys.json holding hashes plus the manifest dir generally
(server-side, storage-root, never served). The mint response NEVER contains the
raw key; the ONLY place it ever leaves the server is inside the templated
installer download itself.

USE COUNTING (wrapper vs payload)
---------------------------------
``/agent/install/<link_id>.sh`` and ``.ps1`` are convenience wrappers that
locate a python and curl the ``.py`` from the same link path. Only the ``.py``
fetch decrements ``uses_left`` — the wrapper fetch does NOT — so the canonical
one-liner (wrapper fetch + the wrapper's own .py fetch) counts as ONE use, not
two. Documented contract; the route layer enforces it by calling
``consume_download`` only from the ``.py`` branch.

Storage idiom mirrors ``video_share_keys.py``: a small JSON file next to the
model manifest, process-wide lock, unique-per-write temp + os.replace.

LINK KINDS (2026-08-13)
-----------------------
``kind`` distinguishes what a link installs: ``"agent"`` (the historical
hugpy-agent installer — records minted before the field simply lack it, which
reads as "agent") and ``"console"`` (the fleet-console .deb). Both kinds share
this store, the ledger, and revocation; the ROUTE layer serves different
payloads per kind and passes ``expect_kind`` so an agent link can never fetch
the console payload or vice versa. For a console link the ``.sh`` IS the
payload (the raw key is templated straight into it — there is no .py), so the
.sh fetch is the consuming download; the follow-up .deb fetch is gated by
``peek_artifact_serveable`` (link known, unrevoked, unexpired — uses_left
deliberately NOT checked, the .sh just spent it; the deb itself carries no
secret, the member-gated artifact is merely being handed to the capability
holder who a moment ago consumed the key download).
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from typing import Any, Optional

from hugpy_server.state import server_state_path
from hugpy_server.app.functions.imports.utils import api_keys

_LOCK = threading.Lock()

DEFAULT_LINK_TTL_S = 86400          # 24h
DEFAULT_MAX_USES = 1
DEFAULT_SCOPES = ["v1"]


def _store_path() -> str:
    return server_state_path("install_links.json")


def _load() -> dict[str, Any]:
    path = _store_path()
    if not os.path.exists(path):
        return {"links": {}, "audit": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"links": {}, "audit": []}
    data.setdefault("links", {})
    data.setdefault("audit", [])
    return data


def _save(data: dict[str, Any]) -> None:
    path = _store_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Unique temp name per write (pid+token) — same multi-process atomicity
    # rationale api_keys.py documents; os.replace stays the atomicity point.
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _now() -> float:
    return time.time()


def _is_expired(rec: dict[str, Any], now: Optional[float] = None) -> bool:
    exp = rec.get("expires_at")
    if not exp:
        return False
    try:
        return float(exp) <= (now if now is not None else _now())
    except (TypeError, ValueError):
        return False


def _status(rec: dict[str, Any]) -> str:
    if rec.get("revoked"):
        return "revoked"
    if _is_expired(rec):
        return "expired"
    if int(rec.get("uses_left") or 0) <= 0:
        return "exhausted"
    return "active"


def _scrub(rec: dict[str, Any]) -> None:
    """Drop the raw key from a record that can no longer serve a download."""
    rec["raw_key"] = ""


def _public(rec: dict[str, Any]) -> dict[str, Any]:
    """A link record WITHOUT the raw key, plus its computed status."""
    out = {k: v for k, v in rec.items() if k != "raw_key"}
    out["status"] = _status(rec)
    return out


def _kind_of(rec: dict[str, Any]) -> str:
    """A record's kind; pre-2026-08-13 records lack the field = "agent"."""
    return rec.get("kind") or "agent"


def create_install_link(label: str,
                        scopes: Optional[list[str]] = None,
                        key_expires_at: Optional[float] = None,
                        link_ttl_s: Optional[float] = None,
                        max_uses: Optional[int] = None,
                        owner: Optional[str] = None,
                        kind: str = "agent") -> dict[str, Any]:
    """Mint a fresh scoped key + its one-time link. Returns the PUBLIC view
    (link_id, key_id, label, scopes, expires_at, uses…) — NEVER the raw key.

    ``owner`` (2026-08-06) is the central-account USERNAME that minted this
    link, recorded so a MEMBER may mint one for their own machine and still be
    scoped to it: ``list_install_links(owner=…)`` shows a member only their own
    ledger, and the route's DELETE accepts the operator OR that creator. None =
    an operator-token / open-mode mint (the historical shape) — which the
    listing treats as operator-owned, i.e. invisible to members.

    The minted key is also labeled with the owner (``created_by`` carries it),
    so the /keys ledger says WHO a key was minted for without a join.

    Raises ValueError on a blank label or unknown scope (api_keys validates
    the vocabulary)."""
    label = (label or "").strip()
    if not label:
        raise ValueError("an install link requires a label")
    scopes = list(scopes) if scopes else list(DEFAULT_SCOPES)
    try:
        ttl = float(link_ttl_s) if link_ttl_s is not None else float(DEFAULT_LINK_TTL_S)
    except (TypeError, ValueError):
        ttl = float(DEFAULT_LINK_TTL_S)
    try:
        uses = int(max_uses) if max_uses is not None else DEFAULT_MAX_USES
    except (TypeError, ValueError):
        uses = DEFAULT_MAX_USES
    uses = max(1, uses)

    # Mint the key FIRST (api_keys raises on a bad scope before anything is
    # persisted here). name = the label so the key list reads sensibly.
    kind = (kind or "agent").strip() or "agent"
    owner = (owner or "").strip() or None
    minted = api_keys.create_api_key(
        name=label, label=(f"{label} (for {owner})" if owner else label),
        scopes=scopes,
        created_by=(f"install-link:{owner}" if owner else "install-link"),
        expires_at=key_expires_at)
    raw_key = minted["key"]

    link_id = secrets.token_urlsafe(24)
    now = _now()
    rec = {
        "link_id": link_id,
        "kind": kind,
        "key_id": minted["id"],
        "label": label,
        # WHO minted it (central username) — None for an operator-token/open-mode
        # mint. Legacy records simply lack the field, which reads as None.
        "owner": owner,
        "scopes": scopes,
        "created_at": now,
        "expires_at": (now + ttl) if ttl > 0 else None,
        "key_expires_at": float(key_expires_at) if key_expires_at else None,
        "max_uses": uses,
        "uses_left": uses,
        "downloads": [],       # audit rows: {ts, remote_addr, kind}
        "revoked": False,
        "raw_key": raw_key,    # scrubbed on exhaustion/revoke/expiry
    }
    with _LOCK:
        data = _load()
        data["links"][link_id] = rec
        _save(data)
    return _public(rec)


def get_link(link_id: str) -> Optional[dict[str, Any]]:
    """Public view of one link (no raw key), or None."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        return _public(rec) if rec else None


def list_install_links(owner: Optional[str] = None) -> list[dict[str, Any]]:
    """Every link (incl. exhausted/expired/revoked — the operator sees the full
    ledger), newest first, never the raw key.

    ``owner`` (2026-08-06) scopes the ledger to ONE creator — what a member's
    GET passes, so they see their own links and nobody else's. Records with no
    owner (operator-token / open-mode / pre-2026-08-06 mints) never match an
    owner filter, so they stay operator-only."""
    with _LOCK:
        data = _load()
    recs = data["links"].values()
    if owner:
        recs = [r for r in recs if (r.get("owner") or None) == owner]
    out = [_public(rec) for rec in recs]
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out


def owner_of(link_id: str) -> tuple[bool, Optional[str]]:
    """``(found, owner)`` for a link — the ownership probe the DELETE route uses
    to allow "operator OR the member who created it"."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
    if rec is None:
        return (False, None)
    return (True, (rec.get("owner") or None))


def consume_download(link_id: str, remote_addr: str = "",
                     expect_kind: str = "agent",
                     audit_kind: str = "py") -> Optional[str]:
    """The consuming download path (.py for agent links, .sh for console
    links): if the link is active, of the EXPECTED kind, AND its key still
    verifies as un-revoked, return the RAW KEY, decrement uses_left, audit the
    download, and scrub the raw key if that was the last use. Returns None
    (and scrubs where appropriate) for exhausted/expired/revoked links, a
    revoked key, or a kind mismatch (an agent link must never fetch the
    console payload and vice versa — same 410 as a dead link, no oracle).

    This is the ONLY function that ever returns the raw key."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return None
        if _kind_of(rec) != (expect_kind or "agent"):
            return None
        if rec.get("revoked") or _is_expired(rec) or int(rec.get("uses_left") or 0) <= 0:
            if rec.get("raw_key"):
                _scrub(rec)
                _save(data)
            return None
        raw = rec.get("raw_key") or ""
        if not raw:
            return None  # already scrubbed (shouldn't happen while active)
        rec["uses_left"] = int(rec["uses_left"]) - 1
        rec["downloads"].append({"ts": _now(),
                                 "remote_addr": remote_addr or "",
                                 "kind": audit_kind or "py"})
        if rec["uses_left"] <= 0:
            _scrub(rec)
        _save(data)
    # Key revocation check OUTSIDE our lock (api_keys has its own): a link whose
    # key the operator already revoked must not hand out a dead — or worse,
    # resurrected-looking — credential.
    if not api_keys.verify_api_key(raw):
        return None
    return raw


def peek_active(link_id: str, expect_kind: str = "agent") -> bool:
    """True iff the link could currently serve a download — used by the .sh/.ps1
    wrapper routes so a dead link 410s at the wrapper too, WITHOUT consuming a
    use (only the .py fetch decrements). Kind-checked like consume_download."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return False
        if _kind_of(rec) != (expect_kind or "agent"):
            return False
        return _status(rec) == "active" and bool(rec.get("raw_key"))


def peek_artifact_serveable(link_id: str, expect_kind: str = "console") -> bool:
    """True iff the link may still fetch its (non-secret) ARTIFACT — the
    console .deb gate. Checks known + kind + unrevoked + unexpired but NOT
    uses_left/raw_key: the consuming .sh fetch has just spent the use by the
    time the script turns around and downloads the deb. Revoke or expiry still
    kills this immediately (the capability dies with the link)."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return False
        if _kind_of(rec) != (expect_kind or "console"):
            return False
        return not rec.get("revoked") and not _is_expired(rec)


def note_wrapper_fetch(link_id: str, remote_addr: str = "", kind: str = "sh") -> None:
    """Audit a wrapper (.sh/.ps1) fetch. Does NOT decrement uses_left."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return
        rec.setdefault("downloads", []).append(
            {"ts": _now(), "remote_addr": remote_addr or "", "kind": kind})
        _save(data)


def revoke_install_link(link_id: str) -> bool:
    """Revoke a link AND the key behind it. Idempotent-ish: revoking an already
    revoked link still ensures the key is revoked. False only for unknown ids."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return False
        rec["revoked"] = True
        _scrub(rec)
        _save(data)
        key_id = rec.get("key_id")
    if key_id:
        api_keys.revoke_api_key(key_id)
    return True


def delete_install_link(link_id: str) -> bool:
    """REMOVE a link's row from the ledger entirely (2026-08-13 — the console
    "dead weight" cleanup). The audit trail for the row goes with it —
    deliberate: this IS the "clear it out" affordance; operators who want
    history keep the row and use revoke. False only for unknown ids.

    KEY LIFECYCLE (the part that must not be gotten wrong): a link's key is
    revoked with the row ONLY while the link is still ACTIVE — an undelivered
    (or partially delivered, same revoke-semantics as revoke_install_link)
    credential must die rather than orphan. An EXHAUSTED link's key is the
    opposite case: it was DELIVERED — it is now some machine's live agent
    credential — and clearing the dead row out of the ledger must never kill
    a deployed agent. Expired/revoked rows' keys likewise keep their own
    lifecycle (already dead, or independently expiring)."""
    with _LOCK:
        data = _load()
        rec = data["links"].get(link_id)
        if rec is None:
            return False
        was_active = _status(rec) == "active"
        del data["links"][link_id]
        _save(data)
        key_id = rec.get("key_id")
    if was_active and key_id:
        api_keys.revoke_api_key(key_id)
    return True


def prune_install_links(owner: Optional[str] = None) -> int:
    """Sweep every DEAD row (status != active — exhausted, expired, revoked)
    from the ledger and return how many went. Active links are never touched:
    prune is the bulk janitor, not a bulk revoke. ``owner`` scopes the sweep
    to one creator's rows (the member call); None sweeps the whole ledger
    (operator) INCLUDING owner-less operator mints.

    NO key is ever revoked here: an exhausted row's key is a deployed
    machine's live credential (see delete_install_link), a revoked row's key
    is already dead, and an expired row's key has its own key_expires_at
    lifecycle. Prune removes ledger noise, nothing else."""
    with _LOCK:
        data = _load()
        doomed = []
        for lid, rec in data["links"].items():
            if owner is not None and (rec.get("owner") or None) != owner:
                continue
            if _status(rec) == "active":
                continue
            doomed.append(lid)
        for lid in doomed:
            del data["links"][lid]
        if doomed:
            _save(data)
    return len(doomed)
