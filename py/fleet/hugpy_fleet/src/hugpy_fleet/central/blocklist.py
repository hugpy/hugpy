"""Model BLOCK list — the operator's model-level serving-pool primitive.

WHAT
----
An operator can **block** a ``model_key`` (and unblock it). A blocked model is
removed from the SERVING POOL: central never routes to it, never
designates/assigns it, never warms/provisions/transfers it on a sweep, and never
resolves it through a fallback ladder (incl. the agent-brain default). Its files
stay on disk untouched and its existing designation rows stay VISIBLE (inert +
labeled) — block is a ROUTING/pool override, engine-agnostic (a sibling of the
WORKER-level block verb), reversible with one click.

Block outranks pin at routing time: pin is routing *persistence* (the allocation
survives restarts), block is an operator *override*. Blocking does NOT
auto-unassign and does NOT fight the pin's unassign-409 — the designation stays
recorded, routing simply refuses to use it.

SCOPE (today)
-------------
GLOBAL only: a blocked model is blocked EVERYWHERE. The per-worker case is
already covered by *unassign*; global is the missing primitive. The stored value
is an extensible dict (``{blocked, by, ts, note?}``) so a per-worker ``scope``
field can be added later without a migration.

PERSISTENCE
-----------
The F4 runtime settings store (``comms.settings.settings_store``), namespace
``models.blocked`` keyed by ``model_key``. That is the runtime source-of-truth
idiom the console's control plane already uses (fcntl-locked read-modify-write,
atomic replace, a short read cache) — it survives a central restart and rides the
existing operator-gated ``/settings`` surface. No new storage mechanism is
introduced here.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from hugpy_control.settings import settings_store

logger = logging.getLogger(__name__)

# Settings namespace for the block registry (model_key -> {blocked, by, ts, note?}).
NS = "models.blocked"

# The canonical honest-refusal phrase. Kept here so the routing gate, the
# no-worker diagnostic, and remote.py's _PERMANENT_LOAD_MARKERS agree on ONE
# distinct string (never softened / reused for an unrelated reason).
BLOCKED_MARKER = "blocked from the serving pool"


def _record(model_key: str) -> dict:
    """The raw stored record for ``model_key`` (``{}`` when never blocked).

    Normalizes legacy/degenerate shapes into a dict so ``is_blocked`` has one
    truth: a bare truthy scalar is treated as ``{"blocked": True}``.
    """
    try:
        v = settings_store.get(NS, str(model_key))
    except Exception as exc:  # noqa: BLE001 — a read must never break a caller
        logger.warning("blocklist read failed for %s: %s", model_key, exc)
        return {}
    if isinstance(v, dict):
        return v
    return {"blocked": True} if v else {}


def is_blocked(model_key: Optional[str]) -> bool:
    """True iff ``model_key`` is currently blocked from the serving pool."""
    if not model_key:
        return False
    return bool(_record(model_key).get("blocked"))


def block_info(model_key: Optional[str]) -> Optional[dict]:
    """The block record (``{blocked, by, ts, note?}``) when blocked, else None."""
    if not model_key:
        return None
    rec = _record(model_key)
    return dict(rec) if rec.get("blocked") else None


def blocked_keys() -> set:
    """The set of all currently-blocked model_keys."""
    try:
        allv = settings_store.all(NS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("blocklist enumerate failed: %s", exc)
        return set()
    out = set()
    for k, v in (allv or {}).items():
        if (isinstance(v, dict) and v.get("blocked")) or (not isinstance(v, dict) and v):
            out.add(k)
    return out


def block_reason(model_key: Optional[str]) -> Optional[str]:
    """The honest refusal reason when ``model_key`` is blocked, else None.

    Distinct, operator-authored, and NEVER reused for an unrelated cause — the
    string carries ``BLOCKED_MARKER`` so the cold-hold classifier in
    resolvers.remote treats it as a PERMANENT refusal (fail fast, no retry)."""
    if not is_blocked(model_key):
        return None
    return (f"'{model_key}' is {BLOCKED_MARKER} by the operator "
            f"— unblock it to route to it again")


def block(model_key: str, *, by: Optional[str] = None,
          note: Optional[str] = None) -> dict:
    """Block ``model_key`` from the serving pool (idempotent). Returns the record."""
    rec: dict[str, Any] = {"blocked": True, "by": (by or "operator"),
                           "ts": time.time()}
    if note:
        rec["note"] = str(note)
    settings_store.set(NS, str(model_key), rec)
    logger.info("model BLOCKED from serving pool: %s (by=%s)", model_key, rec["by"])
    return rec


def unblock(model_key: str, *, by: Optional[str] = None) -> bool:
    """Unblock ``model_key``. Returns whether it was blocked.

    OPERATOR-UNBLOCK STICKINESS (operator ruling 2026-07-25). An OPERATOR unblock
    does not merely delete the record — it leaves a tombstone
    ``{"blocked": False, "operator_unblocked": True, "by", "ts"}``. That marker
    is what makes ``auto_block`` a no-op for this key forever after: a human
    looked at the machine's arithmetic and overruled it, and a re-block on the
    next sweep would silently undo a human decision every few seconds.

    WHY A TOMBSTONE RATHER THAN A SECOND NAMESPACE: this namespace ALREADY
    stores an extensible dict per key and ``is_blocked`` already reads only
    ``rec["blocked"]``, so a falsey record is inert everywhere by construction —
    no migration, no second store to keep consistent, and the audit trail
    ("who released it, when") lands in the same place an auditor already looks.
    The cost is that a key can hold an inert record; ``blocked_keys`` filters on
    the flag, so nothing downstream sees it.

    ``by="auto"`` (the machine retracting its OWN block — e.g. the fleet grew a
    box that now fits) DELETES instead, leaving no marker: only a human's
    override earns stickiness, and an auto-retraction must not immunize a key
    against a future, genuinely-correct auto-block."""
    rec = _record(model_key)
    was = bool(rec.get("blocked"))
    who = (by or "operator")
    if who == "auto":
        settings_store.delete(NS, str(model_key))
        if was:
            logger.info("model auto-unblocked (fits again): %s", model_key)
        return was
    settings_store.set(NS, str(model_key), {
        "blocked": False, "operator_unblocked": True, "by": who,
        "ts": time.time()})
    logger.info("model UNBLOCKED by %s (returned to serving pool, auto-block "
                "suppressed for this key): %s", who, model_key)
    return was


def operator_unblocked(model_key: Optional[str]) -> bool:
    """True iff an OPERATOR explicitly released ``model_key`` — the sticky
    marker the auto-blocker must honor (never re-block a human's override)."""
    if not model_key:
        return False
    return bool(_record(model_key).get("operator_unblocked"))


def auto_block(model_key: str, note: str) -> Optional[dict]:
    """Machine-authored block (``by="auto"``), the CASE-A entry point.

    Returns the record, or ``None`` when the block was DECLINED. It declines in
    exactly two situations, both deliberate:

      * an operator already released this key (``operator_unblocked``) — the
        stickiness rule above; a human's decision outranks the arithmetic;
      * the key is already blocked BY THE OPERATOR — an auto re-stamp would
        rewrite a human's note/authorship with a machine's.

    Re-stamping an existing ``by="auto"`` record IS allowed: it refreshes the
    numbers in the note when the fleet's capacity changes."""
    rec = _record(model_key)
    if rec.get("operator_unblocked"):
        logger.debug("auto-block declined for %s: operator released it", model_key)
        return None
    if rec.get("blocked") and rec.get("by") not in (None, "auto"):
        logger.debug("auto-block declined for %s: already blocked by %s",
                     model_key, rec.get("by"))
        return None
    return block(model_key, by="auto", note=note)


# ── per-(model × worker) PAIR blocks (operator ruling 2026-08-28) ────────────
# "wouldn't it be best if these groups instead 'blocked' from the individual
# workers those that do not fit?" — a POSITIVE permanent-no fit verdict
# (worker_fit_verdict False: the box's combined VRAM+RAM can never hold the
# model, e.g. coder-next × computron) is recorded HERE as a durable, visible
# pair block, so the dispatch candidate filter excludes the pair statically
# instead of re-pricing (and re-refusing) it on every request. Doctrine
# mirrors auto_block: only a positive verdict ever records (missing data is
# never evidence), and the block CLEARS the moment the verdict flips to True
# (e.g. a quant pin shrinks the model, or the box gains RAM). Same F4
# settings store; keys are supplied by the caller (workers.py normalizes the
# model form so bare/qualified spellings land on one record).
NS_PAIR = "models.blocked_workers"


def pair_block_map(model_key: Optional[str]) -> dict:
    """``{worker_id: {blocked, why, ts, by}}`` for ``model_key`` (``{}`` when
    none). Console-readable — the visibility half of the ruling."""
    if not model_key:
        return {}
    try:
        v = settings_store.get(NS_PAIR, str(model_key))
    except Exception as exc:  # noqa: BLE001 — a read must never break routing
        logger.warning("pair-block read failed for %s: %s", model_key, exc)
        return {}
    return v if isinstance(v, dict) else {}


def pair_blocked(model_key: Optional[str], worker_id: Optional[str]) -> bool:
    """True iff (model, worker) carries a durable pair block."""
    if not model_key or not worker_id:
        return False
    rec = pair_block_map(model_key).get(str(worker_id))
    return bool(rec and rec.get("blocked"))


def auto_block_pair(model_key: str, worker_id: str, *, why: str = "",
                    worker_name: Optional[str] = None) -> bool:
    """Record a machine-derived pair block (idempotent; refreshes the why).
    Returns True when a NEW block was recorded (for one honest log line)."""
    try:
        cur = pair_block_map(model_key)
        fresh = not (cur.get(str(worker_id)) or {}).get("blocked")
        cur = dict(cur)
        cur[str(worker_id)] = {"blocked": True, "by": "auto", "ts": time.time(),
                               "why": why, "worker_name": worker_name}
        settings_store.set(NS_PAIR, str(model_key), cur)
        if fresh:
            logger.warning("pair-BLOCKED %s on %s: %s", model_key,
                           worker_name or worker_id, why or "permanent no-fit")
        return fresh
    except Exception as exc:  # noqa: BLE001 — recording must never break routing
        logger.warning("pair-block write failed for %s/%s: %s",
                       model_key, worker_id, exc)
        return False


def clear_pair_block(model_key: str, worker_id: str) -> bool:
    """Drop the pair block (the verdict changed). Returns True when one was
    actually cleared."""
    try:
        cur = pair_block_map(model_key)
        if str(worker_id) not in cur:
            return False
        cur = dict(cur)
        rec = cur.pop(str(worker_id))
        settings_store.set(NS_PAIR, str(model_key), cur)
        if rec.get("blocked"):
            logger.info("pair block CLEARED for %s on %s (fit verdict changed)",
                        model_key, rec.get("worker_name") or worker_id)
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("pair-block clear failed for %s/%s: %s",
                       model_key, worker_id, exc)
        return False
