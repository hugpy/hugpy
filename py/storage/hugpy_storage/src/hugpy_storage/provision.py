"""Model provisioning for the worker — central-first, Hugging Face fallback.

The worker ships with only a small curated model registry; CENTRAL is the
source of truth for what models exist. So before any files are fetched, the
worker first makes sure it KNOWS the model — pulling the model's config row
from central and registering it into the worker's own in-memory registry
(:func:`ensure_model_registered`). Without that step a model central assigns
but the worker wasn't built with fails to even resolve ("Unknown model_key=
None") and to provision ("Unknown model").

Once the model is known, its files are fetched in this order:

    1. From the CENTRAL node, over WireGuard, using the read-only endpoints
       /api/llm/models/<key>/manifest and /api/llm/models/<key>/file. This needs
       no Hugging Face token on the worker and reuses whatever central already
       downloaded.
    2. If central doesn't have it (409) or is unreachable, fall back to the
       normal Hugging Face download via ``hugpy_storage.download_models.ensure_model``
       — which the inference path would call anyway.

Files are placed under the worker's OWN storage root using the same
route_destination() layout central uses, so the existing loader/`ensure_model`
finds them with no further config.

LAYERING (2026-09-22). This module is STORAGE: it never imports the fleet or
the engine. What it used to pull from ``worker_agent`` siblings now arrives
through two seams:

  * the model REGISTRY (which keys exist, their routing facts, the dir the
    engine would load from, registering a central-provided row) — via
    :mod:`hugpy_storage.catalog_source` (null default = "no models known");
  * the fleet's storage-budget gate, transfer telemetry and thread-pool
    registrar — via :mod:`hugpy_storage.providers` (defaults: admit / silent /
    no-op). The worker agent installs the real ones at boot AND in every slot
    child it spawns, or its transfers run ungated and unobserved.
"""
from __future__ import annotations

import os
import json
import time
import logging
import threading
import urllib.parse
import urllib.request
import urllib.error

from hugpy_storage import providers as _providers
from hugpy_storage.catalog_source import (
    catalog_canonical_key,
    catalog_get,
    catalog_register,
    catalog_resolve_dir,
    routing_of,
)
from hugpy_storage.events import publish_catalog_changed
from hugpy_storage.model_paths import resolve_model_dir, route_destination
from hugpy_storage.model_presence import (
    describe_disk_error,
    errno_name,
    model_looks_downloaded,
)

logger = logging.getLogger("hugpy_storage.provision")

_CHUNK = 8 * 1024 * 1024  # 8 MiB streaming chunks

# ---------------------------------------------------------------------------
# BUDGET-STATE PROVIDER (slice 8, Part A — gate EVERY transfer entry).
# ---------------------------------------------------------------------------
# The storage-budget gate (evict_to_fit) needs a WorkerState (limits, assigned,
# last_picked). Historically only callers that HELD the state passed it, so
# entry points that didn't — the /redownload route and the slot child's
# _ensure_present — pulled bytes onto an over-budget store WITHOUT the gate (the
# 2026-07-17 ae incident: chunks downloading while a different pid was actively
# REFUSING the same model). This closes those: ensure_model_present now resolves
# a budget state itself when the caller passes state=None.
#
# The worker agent registers its live WorkerState here at boot (in-process
# callers — /redownload, reconcile — then gate against the REAL state). A
# separate process (the slot child) has no such state; for it we synthesize a
# MINIMAL state from env + on-disk truth (resolve_effective_cap reads the cap
# from env — slice 4's projection; _worker_storage tolerates an empty assigned
# set: the reap scan still enumerates on-disk models via get_models_dict). Either
# way the gate runs before the first byte. NEVER gate a pull the gate already
# admitted (see _ensure_model_present_inner's resume note).
_BUDGET_STATE = None


def set_budget_state(state) -> None:
    """Register the live WorkerState so state-less entry points (the /redownload
    route, and — in-process — anything that calls ensure_model_present without a
    state) still run the storage-budget gate. Idempotent; called once at boot."""
    global _BUDGET_STATE
    _BUDGET_STATE = state


class _MinimalBudgetState:
    """A stand-in WorkerState for a process that has none (the slot child).

    Carries only what evict_to_fit reads: limits (the cap — env-resolved via
    resolve_effective_cap, so a slot honours the same disk_cache_gib the agent
    does), an empty assigned set (the reap scan still finds on-disk models), and
    empty last_picked/allocated (the FIFO falls back to on-disk mtime, and the
    refusal simply omits the allocation clause). It is READ-only by the gate."""
    def __init__(self):
        self.assigned_models = []
        self._provisioning = []
        self.model_last_picked = {}
        self.allocated = {}
        self.refused = {}
        # limits carry the central disk_cache_gib the agent projected into env
        # (slice 4); evict_to_fit → resolve_effective_cap folds worker knobs in.
        self.limits = {}
        try:
            raw = os.environ.get("_HUGPY_CENTRAL_DISK_CACHE_GIB")
            if raw not in (None, ""):
                self.limits = {"disk_cache_gib": float(raw)}
            rr = os.environ.get("_HUGPY_CENTRAL_DISK_RESERVE_GIB")
            if rr not in (None, ""):
                self.limits["disk_reserve_gib"] = float(rr)
        except (TypeError, ValueError):
            self.limits = {}


def _resolve_budget_state(state):
    """The state the gate should run against for THIS pull. Explicit caller state
    wins; else the agent-registered live state; else a minimal env/disk state so
    a state-less process (slot child) still gates. Never None — the gate always
    runs (Part A: gate-everywhere is unconditional)."""
    if state is not None:
        return state
    if _BUDGET_STATE is not None:
        return _BUDGET_STATE
    return _MinimalBudgetState()


# ---------------------------------------------------------------------------
# Central chain-of-command: refusal is AUTHORITATIVE (operator ruling, ae 1.2TB
# incident 2026-07-17).
# ---------------------------------------------------------------------------
# "unless it is and the worker is resolving to hf download. then that needs to
# be something that is fixed on the py module level via chain of command."
#
# Central OWNS the distribution decision. If central answered AT ALL — any HTTP
# status, including a deliberate refusal (404/409/"archive refused") or a 5xx —
# it is ALIVE and its verdict STANDS. The worker must NOT slip out the HF side
# door and pull weights central just declined to serve (that is how a refusal
# turned into a silent 55GB/700GB HF pull).
#
# HF fallback is a SURVIVAL path, permitted ONLY when central gave NO verdict at
# all: the box can't reach central (connection refused / timeout / DNS) or no
# central URL is configured. We distinguish the two by EXCEPTION TYPE at the
# transfer call sites, never by string-matching a reason (urllib's taxonomy):
#
#   * urllib.error.HTTPError  -> central RESPONDED with a status. A VERDICT.
#                                (HTTPError is a subclass of URLError, so it is
#                                 caught FIRST everywhere below.)
#   * urllib.error.URLError (non-HTTPError), socket.timeout, TimeoutError,
#     ConnectionError, OSError -> no HTTP response reached us. UNREACHABLE.
#
# The fetchers RAISE CentralUnreachable for the unreachable class (instead of
# the old swallow-to-False, which made "central refused" and "central down"
# indistinguishable at _provision_now). A verdict-shaped failure returns False /
# raises a non-CentralUnreachable error; _provision_now then refuses HF.
#
# Escape hatch: env HUGPY_HF_FALLBACK=always restores the pre-ruling behavior
# (any central failure falls through to HF) for emergencies.
import socket as _socket


class CentralUnreachable(Exception):
    """Central gave NO HTTP response — connection refused, timeout, or DNS
    failure. The ONLY condition (besides no central URL) under which the HF
    survival fallback is permitted. Carries the originating exception."""

    def __init__(self, cause: BaseException):
        self.cause = cause
        super().__init__(f"{type(cause).__name__}: {cause}")


def _is_unreachable(exc: BaseException) -> bool:
    """True when ``exc`` means central never answered (no HTTP verdict).

    HTTPError is a URLError subclass but IS a verdict (central responded), so it
    is explicitly excluded. Everything else in urllib's connection-failure
    taxonomy — a bare URLError (its ``.reason`` is the socket error), a raw
    socket timeout, ConnectionError, OSError — is unreachable."""
    if isinstance(exc, urllib.error.HTTPError):
        return False
    return isinstance(exc, (urllib.error.URLError, _socket.timeout,
                            TimeoutError, ConnectionError, OSError))


def _hf_fallback_always() -> bool:
    """Emergency escape hatch: HUGPY_HF_FALLBACK=always restores the pre-2026-07-17
    behavior (ANY central failure — verdict or not — falls through to HF)."""
    return (os.environ.get("HUGPY_HF_FALLBACK", "").strip().lower() == "always")


# ---------------------------------------------------------------------------
# Central-transfer authentication.
# ---------------------------------------------------------------------------
# Central's worker file-transfer endpoints (/manifest, /file, /chunksums,
# /archive) gate on a credential: the operator token OR a valid worker
# enrollment bearer token (_transfer_authorized in worker_routes.py). The puller
# therefore presents the SAME enrollment token the agent's CentralClient already
# sends on register/heartbeat — ``Authorization: Bearer <token>`` (see
# CentralClient._post in agent.py). When no token is available (the gradual-
# rollout default, HUGPY_WORKER_ENROLL_REQUIRED off) NO header is added, so
# behavior is byte-for-byte what it was before this change — a pure superset.
_ENROLL_TOKEN: str | None = None


def set_enroll_token(token: str | None) -> None:
    """Remember the worker's enrollment token so central-transfer requests carry
    it. Called once by the agent at startup with the very value it hands
    CentralClient (``args.token`` = --token / WORKER_ENROLL_TOKEN). A blank/None
    token clears it, restoring the tokenless (pre-auth) behavior."""
    global _ENROLL_TOKEN
    _ENROLL_TOKEN = (token or "").strip() or None


def _enroll_token() -> str | None:
    """The worker enrollment token, if any. Prefers a value the agent explicitly
    set (:func:`set_enroll_token`, i.e. exactly CentralClient's token, so a
    CLI-only ``--token`` is honored); falls back to the same
    ``WORKER_ENROLL_TOKEN`` env var CentralClient's ``--token`` defaults to, so a
    provision that runs without the agent bootstrap still authenticates. Returns
    None when neither is set."""
    if _ENROLL_TOKEN:
        return _ENROLL_TOKEN
    env = (os.environ.get("WORKER_ENROLL_TOKEN") or "").strip()
    return env or None


# ---------------------------------------------------------------------------
# Transfer self-identification (central budget handshake, ae 1.2TB 2026-07-17).
# ---------------------------------------------------------------------------
# Every central-transfer request now says WHO is asking (worker id) and WHY
# (purpose), so central "abides by the limits set within its own backend": it
# can 409 a BACKGROUND pull that would push the worker over budget, while never
# refusing a DEMAND pull (a called model — the worker's own fit_plan evicts to
# fit it). worker_id is a stable module value (set once at startup). purpose is
# per-CALL, so it rides a thread-local set by ensure_model_present's ``purpose``
# arg. Both headers are ADDITIVE: absent -> byte-for-byte the pre-feature
# requests, and central treats a missing purpose as demand (permissive during
# fleet convergence).
_WORKER_ID: str | None = None
_PURPOSE = threading.local()

# The purposes a transfer can declare. "demand" = a real call (never budget-
# refused). "probe" = a fit check (non-downloading as of Deliverable A, but
# labeled for completeness). "reconcile"/"assign" = background pre-fetch (the
# budget-refusable class).
_BACKGROUND_PURPOSES = ("reconcile", "assign")


def set_worker_id(worker_id: str | None) -> None:
    """Remember this worker's id so central-transfer requests identify their
    origin. Called once at agent startup alongside set_enroll_token."""
    global _WORKER_ID
    _WORKER_ID = (worker_id or "").strip() or None


def _current_purpose() -> str | None:
    """The purpose of the transfer running on THIS thread, or None. Set for the
    duration of an ensure_model_present call via _purpose_scope."""
    return getattr(_PURPOSE, "value", None)


class _purpose_scope:
    """Context manager: stamp the transfer purpose on this thread for the life of
    a provision call, restoring the prior value on exit (nested calls are safe)."""

    def __init__(self, purpose: str | None):
        self.purpose = (purpose or "").strip() or None
        self._prev = None

    def __enter__(self):
        self._prev = getattr(_PURPOSE, "value", None)
        _PURPOSE.value = self.purpose
        return self

    def __exit__(self, *exc):
        _PURPOSE.value = self._prev
        return False


def _identity_headers() -> dict:
    """``X-Worker-Id`` + ``X-Transfer-Purpose`` when known, else omitted. Additive
    on top of the auth header; central reads these to apply its budget gate."""
    h = {}
    if _WORKER_ID:
        h["X-Worker-Id"] = _WORKER_ID
    purpose = _current_purpose()
    if purpose:
        h["X-Transfer-Purpose"] = purpose
    return h


def _auth_headers() -> dict:
    """``{"Authorization": "Bearer <token>"}`` when an enrollment token is
    available, else ``{}``, PLUS the worker-identity/purpose headers (also
    additive). The exact header CentralClient._post attaches; an empty result
    means no headers added (today's tokenless behavior)."""
    tok = _enroll_token()
    h = {"Authorization": f"Bearer {tok}"} if tok else {}
    h.update(_identity_headers())
    return h


def _auth_request(url: str, headers: dict | None = None) -> "urllib.request.Request":
    """Build a GET Request for a central-transfer URL, adding the enrollment
    bearer token (when available) on top of any caller headers (e.g. Range).
    Token absent -> no Authorization header, i.e. an ordinary ``Request(url,
    headers)`` identical to the pre-auth call sites."""
    merged = dict(headers or {})
    merged.update(_auth_headers())
    return urllib.request.Request(url, headers=merged)


# Single-flight provisioning: one download per model_key at a time. Without this
# every concurrent /infer/stream (plus the pre-provision) kicks off its own full
# multi-GB transfer into the SAME directory, and the parallel writers stomp each
# other (the symptom: a transfer "stuck" partway). Waiters block, then find the
# model already present and return immediately.
_PROVISION_LOCKS: dict[str, threading.Lock] = {}
_PROVISION_LOCKS_GUARD = threading.Lock()


def _provision_lock(model_key: str) -> threading.Lock:
    with _PROVISION_LOCKS_GUARD:
        lock = _PROVISION_LOCKS.get(model_key)
        if lock is None:
            lock = _PROVISION_LOCKS[model_key] = threading.Lock()
        return lock


# ---------------------------------------------------------------------------
# Registry sync — teach the worker about a model it wasn't built with.
# ---------------------------------------------------------------------------
def _clean_hub(value) -> str:
    """Normalise a hub_id for comparison (strip storage-path leakage)."""
    try:
        from hugpy_storage.download_models import _clean_repo_id
        return _clean_repo_id(value)
    except Exception:
        return str(value or "").strip("/")


def _assure_local_key(model_key: str):
    """Canonical local registry key for model_key (key/hub_id/suffix), or None.
    Answered by the installed catalog source (the engine's assure_model_key)."""
    return catalog_canonical_key(model_key)


def _model_dir(model_key: str, cfg=None) -> str | None:
    """Where THIS box holds ``model_key``: the catalog's answer (the engine's
    get_model_path — env override, recorded folder, read-through) when a
    registry is installed, else the read-through resolver over the routing
    facts we have. None when nothing is known."""
    d = catalog_resolve_dir(model_key)
    if d:
        return d
    if cfg is None:
        cfg = catalog_get(model_key)
    if cfg is None:
        return None
    try:
        return resolve_model_dir(routing_of(cfg), cfg=cfg, require_complete=False)
    except Exception:  # noqa: BLE001
        return None


def _central_manifest(central_url: str, model_key: str) -> dict:
    """GET central's file manifest + routing meta for a model key."""
    base = central_url.rstrip("/") + "/api/llm/models/" + urllib.parse.quote(model_key)
    return _get_json(base + "/manifest")


# Central's model list/config lives under a different prefix than the worker
# file-share routes, and that prefix has moved between builds. Try the known
# candidates rather than hard-coding one (which 404'd in the field).
_MODEL_LIST_PATHS = ("/api/models", "/api/llm/models", "/models")


def list_central_models(central_url: str) -> list[dict]:
    """Return central's model rows, trying each known list endpoint in turn.

    Each row is a model config dict (carrying at least ``model_key``/``key`` and
    ``hub_id``). Returns [] if none of the endpoints answer with a usable list.
    """
    for path in _MODEL_LIST_PATHS:
        url = central_url.rstrip("/") + path
        try:
            listing = _get_json(url)
        except Exception:
            continue
        if isinstance(listing, dict):
            return [r for r in listing.values() if isinstance(r, dict)]
        if isinstance(listing, list):
            return [r for r in listing if isinstance(r, dict)]
    return []


def _fetch_central_model_row(central_url: str, model_key: str) -> dict | None:
    """Pull one model's config row from central, however possible.

    Order:
      1. the worker file-share **manifest** endpoint
         (/api/llm/models/<key>/manifest) — proven to work and it also carries
         the routing meta we need to register. Used when model_key is central's
         own single-segment key (e.g. the assignment key ``DAN-Qwen3-1.7B``).
      2. central's model list/config, trying each known prefix, matching on key
         OR cleaned hub_id (handles a request that carries the hub_id while
         central keys the model by a short name).
    """
    # 1) manifest endpoint — reliable, and confirms central actually has files.
    if "/" not in model_key:
        try:
            meta = _central_manifest(central_url, model_key)
            if isinstance(meta, dict) and meta.get("hub_id"):
                row = dict(meta)
                row.setdefault("key", model_key)
                return row
        except urllib.error.HTTPError as exc:
            if exc.code not in (404, 409):
                logger.warning("central manifest for %s: HTTP %s", model_key, exc.code)
        except Exception as exc:
            logger.warning("central manifest for %s failed: %s", model_key, exc)

    # 2) resolve via the model list (handles hub_id + unknown prefix).
    want = _clean_hub(model_key)
    for path in _MODEL_LIST_PATHS:
        url = central_url.rstrip("/") + path
        try:
            listing = _get_json(url)
        except Exception:
            continue
        if isinstance(listing, dict):
            rows = list(listing.values())
        elif isinstance(listing, list):
            rows = listing
        else:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if (row.get("key") or row.get("model_key")) == model_key:
                return row
            if want and _clean_hub(row.get("hub_id")) == want:
                return row
    logger.warning("central has no resolvable config for %s", model_key)
    return None


def _register_local_model(model_key: str, row: dict) -> bool:
    """Insert a central-provided model row into the worker's live registry.

    The registry is engine state; the installed catalog source performs the
    mutation (the engine's derive_model_config_row + update_model_config_dict)
    so every holder of the registry dicts sees the model at once. The null
    source reports False — a box with no engine simply provisions against the
    central row's routing facts.
    """
    row = dict(row)
    row.setdefault("model_key", model_key)
    if catalog_register(model_key, row):
        logger.info("registered model %s from central into local registry", model_key)
        return True
    logger.warning("registration of %s did not stick (no catalog source, or it "
                   "refused the row)", model_key)
    return False


def ensure_model_registered(model_key: str, central_url: str | None) -> str | None:
    """Make sure ``model_key`` exists in the worker's LOCAL registry.

    Accepts a registry key OR a hub_id. If the worker already knows it,
    returns the canonical local key. Otherwise pulls the config row from
    central and registers it. Returns the canonical local key, or None if the
    model can't be learned (no central / central doesn't have it).
    """
    local = _assure_local_key(model_key)
    if local:
        return local
    if not central_url:
        return None

    row = _fetch_central_model_row(central_url, model_key)
    if not row:
        logger.warning("central has no config for %s; cannot register", model_key)
        return None

    key = row.get("key") or row.get("model_key") or model_key
    if _register_local_model(key, row):
        return _assure_local_key(key) or key
    return None


def _comfy_checkpoints_dir() -> str:
    """Where THIS box's ComfyUI loads checkpoints from. Override with
    COMFY_CHECKPOINTS_DIR; default matches the standard install."""
    return os.environ.get("COMFY_CHECKPOINTS_DIR") or os.path.expanduser(
        "~/ComfyUI/models/checkpoints")


def ensure_comfy_checkpoint(model_key: str, central_url: str | None) -> bool:
    """Comfy provisioning: the checkpoint must end up INSIDE ComfyUI's
    models/checkpoints — everything is symlinks, never copies.

      1. already in ComfyUI's dir            -> done (operator hand-placed)
      2. present in the hugpy model layout   -> symlink it in
      3. neither -> pull from central via the NORMAL file machinery (central
         symlinks its /checkpoints store into the manifest layout), then
         symlink into ComfyUI's dir.
    """
    try:
        model_key = ensure_model_registered(model_key, central_url) or model_key
        cfg = catalog_get(model_key)
        filename = getattr(cfg, "filename", None)
        if not filename:
            return False
        dest = os.path.join(_comfy_checkpoints_dir(), filename)
        if os.path.exists(dest):
            return True
        # in the hugpy layout already?
        src_dir = route_destination({
            "hub_id": getattr(cfg, "hub_id", None), "framework": "comfy",
            "primary_task": getattr(cfg, "primary_task", None),
            "name": getattr(cfg, "name", None),
            "folder": getattr(cfg, "folder", None), "filename": filename})
        src = os.path.join(src_dir, filename)
        if not os.path.isfile(src) and central_url:
            logger.info("comfy checkpoint %s missing locally — pulling from "
                        "central", filename)
            if not ensure_model_present(model_key, central_url):
                return False
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.symlink(os.path.realpath(src), dest)
            logger.info("comfy checkpoint linked: %s -> %s", dest, src)
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_comfy_checkpoint(%s) failed: %s", model_key, exc)
        return False


def model_is_local(model_key: str) -> bool:
    """True if the model already looks downloaded under the worker's storage."""
    try:
        cfg = catalog_get(model_key)
        # Comfy rows: "local" means the checkpoint is loadable by THIS box's
        # ComfyUI (its own dir counts — operator hand-placed files included).
        if getattr(cfg, "framework", None) == "comfy":
            filename = getattr(cfg, "filename", "") or ""
            if filename and os.path.exists(
                    os.path.join(_comfy_checkpoints_dir(), filename)):
                return True
        path = _model_dir(model_key, cfg)
        if not path:
            return False
        return bool(model_looks_downloaded(path, cfg))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ONE dir<->key normalization, shared by _orphan_scan (agent.py) and every
# locality predicate (model_is_local above, models_local heartbeat). This is
# the fix for the 2026-07-17 over-report: _orphan_scan used to compare a
# path-sliced "hub id" against known keys with only a lowercase/strip
# normalization — a designated key like ``Qwen~Qwen3-Coder-Next-GGUF`` (the
# ``~``-qualifier discover_models() mints on an owner collision, see
# imports/apis/get_module.py) was compared VERBATIM against directory-derived
# names that only ever contain ``/``, never ``~``. The two never collided on
# their own; the scan only survived by an accidental bare-repo-name fallback
# match, which breaks the moment two owners share a repo basename (the exact
# case ``~`` exists to disambiguate). Route every "is this on-disk dir a known
# model" decision through the SAME expansion so the two can't diverge again.
# ---------------------------------------------------------------------------

def _dir_slug(value: str) -> str:
    """Collapse a key / hub_id / relative-path into one comparable slug.

    ``~``, ``/``, ``_``, ``-``, whitespace all collapse to a single ``_`` and
    case is folded — so ``Qwen~Qwen3-Coder-Next-GGUF`` (assignment key),
    ``Qwen/Qwen3-Coder-Next-GGUF`` (hub_id / on-disk path), and
    ``qwen_qwen3_coder_next_gguf`` all land on the same slug. This mirrors
    ``managers/resolvers/assure_model_key._slugify`` (the resolver already
    proven to bridge key/hub_id/folder forms) so the orphan scan's notion of
    "known" can never quietly diverge from the resolver's.
    """
    import re
    return re.sub(r"[^A-Za-z0-9.]+", "_", str(value or "").strip()).strip("_").lower()


def known_model_dir_forms(known_keys) -> set:
    """Expand ``known_keys`` (assignment/catalog/loaded/etc. model_keys) into
    every SLUG a legitimately-present on-disk directory might produce.

    For each key this folds in:
      * the key itself (``owner~repo``, a bare staple name, a hub_id, …);
      * the bare trailing component (``repo`` alone) — handles a dir whose
        path-derived name lost its owner segment;
      * the resolved model's ``hub_id`` (``owner/repo``) when the key is
        known to this worker's registry;
      * the RELATIVE PATH (root-relative, slash form) of every dir
        ``candidate_model_dirs`` would accept for that model — the flat
        target AND every legacy task-dir / other-runtime-family shape the
        read-through resolver (``resolve_model_dir``) honors. This is what
        makes a legacy nested-path copy of a KNOWN model read as legacy-path,
        never orphan, without this helper having to reimplement layout
        knowledge that already lives in imports/src/constants/paths.py.

    Best-effort per key: a key this worker can't resolve to a config just
    contributes its own slug forms and is skipped for the candidate-dir
    expansion — it still narrows nothing incorrectly, it just can't widen.
    """
    try:
        from hugpy_storage.model_paths import candidate_model_dirs
        from hugpy_platform.constants import DEFAULT_ROOT, MODELS_HOME
    except Exception:  # noqa: BLE001 — never let import shape break the scan
        candidate_model_dirs = None
        DEFAULT_ROOT = MODELS_HOME = None

    root = None
    if MODELS_HOME:
        root = str(MODELS_HOME)
    elif DEFAULT_ROOT:
        import os as _os
        root = _os.path.join(str(DEFAULT_ROOT), "models")

    forms: set = set()
    for k in known_keys:
        if not k:
            continue
        forms.add(_dir_slug(k))
        tail = str(k).replace("~", "/").rsplit("/", 1)[-1]
        forms.add(_dir_slug(tail))

        cfg = catalog_get(k)
        if cfg is None:
            continue

        hub_id = getattr(cfg, "hub_id", None)
        if hub_id:
            forms.add(_dir_slug(hub_id))
            forms.add(_dir_slug(str(hub_id).rsplit("/", 1)[-1]))

        if candidate_model_dirs is None or not root:
            continue
        try:
            routing = {
                "hub_id": hub_id, "framework": getattr(cfg, "framework", None),
                "filename": getattr(cfg, "filename", None),
                "include": getattr(cfg, "include", None),
                "primary_task": getattr(cfg, "primary_task", None),
                "tasks": getattr(cfg, "tasks", None),
                "folder": getattr(cfg, "folder", None),
                "dir": getattr(cfg, "dir", None),
            }
            for d in candidate_model_dirs(routing, root):
                try:
                    import os as _os
                    rel = _os.path.relpath(d, root)
                except Exception:
                    continue
                if rel.startswith(".."):
                    continue                       # outside the store root
                forms.add(_dir_slug(rel))
        except Exception:  # noqa: BLE001 — best-effort widening only
            continue
    return forms


def dir_is_known_model(rel_path: str, known_forms: set) -> bool:
    """True when ``rel_path`` (a model dir, root-relative) matches something in
    ``known_forms`` — either the exact relative-path slug (covers a legacy
    nested-path copy of a known model verbatim) or the bare trailing-component
    slug (covers a hub-id-derived / path-sliced fallback). Single choke point
    so _orphan_scan and any future locality check compare dirs the same way.
    """
    slug = _dir_slug(rel_path)
    if slug in known_forms:
        return True
    tail = str(rel_path).replace("~", "/").rsplit("/", 1)[-1]
    return _dir_slug(tail) in known_forms


# Doctrinally-unattributed on-disk classes: never orphaned regardless of
# catalog membership. comfy checkpoints sit under models/misc/comfy/** as
# symlinks into <root>/checkpoints (see models_config._sweep_comfy_checkpoints)
# and are explicitly excluded from the storage/allocation accounting the
# reaper and heartbeat use (agent.py's _reap_scan / _storage_model_row skip
# framework=="comfy" rows outright) — the operator doctrine is "comfy is
# excluded from allocations; models can sit on the drive unattributed". The
# orphan scan must honor the same exclusion so misc/comfy/* is never reported
# as unattributed-on-disk residue.
DOCTRINE_EXCLUDED_PREFIXES = ("misc/comfy/", "misc\\comfy\\")


def is_doctrine_excluded(rel_path: str) -> bool:
    """True when ``rel_path`` (root-relative) belongs to a class the operator
    has declared never-orphaned regardless of catalog membership (comfy)."""
    norm = str(rel_path or "").strip("/\\").replace("\\", "/")
    return any(norm == p.rstrip("/") or norm.startswith(p.replace("\\", "/"))
               for p in DOCTRINE_EXCLUDED_PREFIXES)


def _on_shared_model_store(rp: str) -> bool:
    """True when `rp` lives on SHARED/central model storage — the canonical
    catalog other fleet nodes read — so it must NEVER be deleted from here.

    Operator invariant: "nothing should delete from the central drive." A box
    that mounts the shared catalog (e.g. ae on the USB-C NAS) serves the fleet's
    source-of-truth copies; a reap there would remove them for everyone. Two
    independent signals, either one trips the guard (fail-safe):
      1. ``HUGPY_SHARED_MODEL_STORE`` truthy — explicit per-box opt-out of ALL
         deletes; set on every box whose model root is the shared/central volume.
      2. A ``.hugpy-central-catalog`` sentinel at or above ``rp`` — the central
         node drops it at MODELS_HOME root, so a shared mount auto-detects even
         if the env flag was forgotten.
    """
    if os.environ.get("HUGPY_SHARED_MODEL_STORE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    d = rp
    for _ in range(64):                       # walk up to the mount root
        try:
            if os.path.exists(os.path.join(d, ".hugpy-central-catalog")):
                return True
        except OSError:
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return False


def _model_store_reapable(rp: str) -> bool:
    """SAFE-BY-DEFAULT gate for deleting a model file. Returns True ONLY when the
    box has explicitly declared its model store LOCAL & disposable
    (``HUGPY_MODEL_STORE_REAPABLE`` truthy) AND ``rp`` is not on shared/central
    storage. Every other state — the flag unset, a shared-store flag, a central
    sentinel — returns False, so an UNCONFIGURED or shared box never deletes a
    model file. This makes "nothing deletes from the central drive" hold even
    with zero per-box setup: reaping is opt-in, not opt-out."""
    if _on_shared_model_store(rp):
        return False
    return os.environ.get("HUGPY_MODEL_STORE_REAPABLE", "").strip().lower() in ("1", "true", "yes", "on")


def wipe_model(model_key: str, path: str = "") -> bool:
    """Delete the model's local files so the next provision re-downloads it.

    Used by the `redownload` path: a plain provision only fetches when the model
    is MISSING (see model_is_local), so refreshing a corrupt/stale copy requires
    removing it first. Returns True if the path is gone afterwards. Jailed against
    obviously-wrong targets (root/home/short paths) AND against shared/central
    model storage — the operator invariant that no reap may touch the central
    drive.

    ``path`` (slice 3, C) — an EXPLICIT delete target: the STORE-ROOT copy the
    reaper classified, used instead of get_model_path's read-through. On a box
    like ae (hot store root + a mounted shared/central NAS) get_model_path can
    resolve to the NAS copy, which the shared gate below correctly refuses — so
    without this the hot copy the scan wants gone would never be deleted. The
    SAME jail + shared-gate re-proof runs on the resolved realpath regardless of
    where it came from: a caller-supplied NAS path is still refused, so this only
    lets the reaper act on the hot copy it already vetted, never widens what may
    be deleted."""
    import os
    import shutil
    if path:
        # Caller supplied the exact copy to delete (the reaper's store-root row).
        # Re-prove every guard on THIS realpath below — never trust the caller.
        pass
    else:
        path = _model_dir(model_key)
    if not path:
        return False
    rp = os.path.realpath(path)
    if len(rp) < 6 or rp in ("/", os.path.expanduser("~")):
        return False  # refuse a dangerous target
    # HARD INVARIANT (single choke point for the reaper AND the redownload path):
    # delete ONLY when the box declared its model store reapable AND the target is
    # not shared/central; never follow a symlink to delete its (shared/operator-
    # managed) target. Safe-by-default: unconfigured or shared boxes refuse.
    if os.path.islink(path) or not _model_store_reapable(rp):
        logger.warning(
            "wipe_model REFUSED for %s (%s): model store not reapable / shared / "
            "symlink — never deleted (set HUGPY_MODEL_STORE_REAPABLE=1 only on "
            "boxes with local disposable model storage).",
            model_key, rp)
        return False
    try:
        if os.path.isdir(rp):
            shutil.rmtree(rp, ignore_errors=True)
        elif os.path.exists(rp):
            os.unlink(rp)
    except OSError:
        pass
    gone = not os.path.exists(rp)
    if gone:
        # The inventory moved: tell the catalog (engine) so its rows stop
        # reporting the model as installed.
        publish_catalog_changed("wipe_model", model_key=model_key,
                                destination=rp, change="wipe")
    return gone


def _local_destination(meta: dict) -> str:
    """Where this file-set should live on the worker (same layout as central)."""
    return route_destination({
        "hub_id": meta.get("hub_id"),
        "name": meta.get("name"),
        "framework": meta.get("framework"),
        "task": meta.get("task"),
        "primary_task": meta.get("task"),
        "filename": meta.get("filename"),
        "include": meta.get("include"),
    })


def _get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(_auth_request(url), timeout=timeout) as resp:
        import json
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, dest_path: str, expected_size: int | None,
                   on_bytes=None) -> None:
    """Stream one file to dest_path, resuming if a partial is already present.

    ``on_bytes(n)`` is called with the number of newly-written bytes per chunk
    so the caller can report download progress.
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    have = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    if expected_size is not None and have == expected_size:
        if on_bytes:
            on_bytes(have)   # count the already-present bytes toward progress
        return  # already complete

    req = _auth_request(url)
    if have and expected_size and have < expected_size:
        req.add_header("Range", f"bytes={have}-")
        mode = "ab"
        if on_bytes:
            on_bytes(have)   # resumed: pre-existing bytes already on disk
    else:
        have = 0
        mode = "wb"

    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, mode) as fh:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            if on_bytes:
                on_bytes(len(chunk))


def _download_with_retry(url: str, dest_path: str, expected_size: int | None,
                         on_bytes=None, attempts: int = 4) -> None:
    """Download one file, retrying transient failures with backoff.

    Verifies the on-disk size against ``expected_size`` (when known) and retries
    until it matches, so a truncated/short file never passes as complete. Raises
    the last error if every attempt fails. Progress (``on_bytes``) is reported
    only on the first attempt to avoid double-counting on retry.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            _download_file(url, dest_path, expected_size,
                           on_bytes=on_bytes if i == 0 else None)
            size_ok = (expected_size is None
                       or (os.path.exists(dest_path)
                           and os.path.getsize(dest_path) == expected_size))
            if size_ok and _gguf_header_ok(dest_path):
                return
            if size_ok:
                # Right size, garbage content (preallocated-then-crashed pull).
                # Size alone lies here — and the resumer treats full-size files
                # as complete, so remove the shell before retrying.
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
                last_exc = RuntimeError(
                    f"corrupt GGUF header for {os.path.basename(dest_path)} — "
                    "removed, re-downloading")
            else:
                last_exc = RuntimeError(
                    f"size mismatch for {os.path.basename(dest_path)}: "
                    f"{os.path.getsize(dest_path) if os.path.exists(dest_path) else 0}"
                    f"/{expected_size}")
        except Exception as exc:  # noqa: BLE001 — retry transient network errors
            last_exc = exc
        time.sleep(min(2 ** i, 8))
    raise last_exc or RuntimeError(f"failed to download {url}")


def _gguf_header_ok(path: str) -> bool:
    """Cheap validity check: GGUF files must start with the b'GGUF' magic.

    A crashed multi-connection pull leaves a PREALLOCATED full-size file of
    zeros — it passes the size check forever, and llama.cpp only reveals the
    truth at load time ('invalid magic characters'). Non-.gguf files pass
    unchecked (no cheap magic for them)."""
    if not path.lower().endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False


def _missing_or_short(dest: str, files: list[dict]) -> list[tuple]:
    """Return [(rel, expected_size, reason)] for files not fully present.

    Side effect: a full-size file with a corrupt GGUF header is UNLINKED here
    (and reported as "corrupt") — the resume logic downstream treats
    right-sized files as complete, so the only way to force a clean re-pull
    is to remove the shell before the download pass sees it."""
    out = []
    for entry in files:
        rel = entry.get("path")
        if not rel:
            continue
        size = entry.get("size")
        target = os.path.join(dest, rel)
        if not os.path.exists(target):
            out.append((rel, size, "absent"))
        elif size is not None and os.path.getsize(target) != size:
            out.append((rel, size, "short"))
        elif not _gguf_header_ok(target):
            try:
                os.remove(target)
            except OSError:
                pass
            out.append((rel, size, "corrupt"))
    return out


def _pull_concurrency() -> int:
    """Max simultaneous connections for a transfer (env HUGPY_PULL_CONCURRENCY)."""
    try:
        return max(1, int(os.environ.get("HUGPY_PULL_CONCURRENCY", "8")))
    except ValueError:
        return 8


# Files bigger than this are split into byte-range segments fetched in parallel,
# so a single multi-GB weights file isn't stuck on one connection.
_SEGMENT_MIN_BYTES = 64 * 1024 * 1024
_SEGMENT_BYTES = 64 * 1024 * 1024


def _supports_range(url: str) -> bool:
    """True if central honours HTTP Range (returns 206) for this file URL."""
    try:
        req = _auth_request(url, {"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.getcode() == 206
    except Exception:
        return False


def _download_segment(url: str, dest_path: str, start: int, end: int,
                      on_bytes=None) -> None:
    """Fetch one inclusive byte range [start, end] into dest_path at its offset."""
    req = _auth_request(url, {"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.getcode() != 206:
            raise RuntimeError("server ignored Range request")
        remaining = end - start + 1
        with open(dest_path, "r+b") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = resp.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
                if on_bytes:
                    on_bytes(len(chunk))
    if remaining > 0:
        raise RuntimeError(f"short segment {start}-{end} of {dest_path}")


def _download_segment_with_retry(url, dest_path, start, end, on_bytes=None,
                                 attempts: int = 4) -> None:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            _download_segment(url, dest_path, start, end,
                              on_bytes=on_bytes if i == 0 else None)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(2 ** i, 8))
    raise last_exc or RuntimeError(f"failed segment {start}-{end} of {url}")


def _segment_ranges(size: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) ranges covering a file, ~_SEGMENT_BYTES each."""
    nseg = max(1, min(_pull_concurrency() * 4, -(-size // _SEGMENT_BYTES)))
    step = -(-size // nseg)  # ceil
    ranges = []
    start = 0
    while start < size:
        end = min(start + step, size) - 1
        ranges.append((start, end))
        start += step
    return ranges


# ── verified chunked transfer ────────────────────────────────────────────────
# Content-verified, crash-safe pulls: central serves per-chunk SHA-256 sums
# (GET .../chunksums), the worker fetches chunk-aligned ranges into a .part
# staging file, hashes each chunk AS IT LANDS, records verified chunks in a
# .state sidecar (so a crash resumes from proven content, not a byte offset),
# and only os.replace()s onto the final name once every chunk verified.
# "Exists under its final name" therefore MEANS complete — the invariant the
# old preallocate-then-size-check scheme couldn't give (a crashed pull left a
# full-size zero shell that passed as present forever).

def _chunk_bytes() -> int:
    try:
        return max(4 * 2**20, min(int(os.environ.get("HUGPY_CHUNK_BYTES", 32 * 2**20)),
                                  256 * 2**20))
    except ValueError:
        return 32 * 2**20


def _fetch_chunksums(base: str, rel: str, chunk_bytes: int) -> list[str] | None:
    """Per-chunk sums from central, or None (older central / any failure —
    the caller falls back to the size+magic scheme)."""
    try:
        d = _get_json(base + "/chunksums?path=" + urllib.parse.quote(rel)
                      + f"&chunk={chunk_bytes}", timeout=590)
        sums = d.get("sums")
        return list(sums) if sums else None
    except Exception as exc:  # noqa: BLE001
        logger.info("no chunksums for %s (%s) — size+magic mode", rel, exc)
        return None


def _load_chunk_state(state_path: str, nchunks: int) -> set[int]:
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            got = json.load(fh).get("verified") or []
        return {i for i in got if isinstance(i, int) and 0 <= i < nchunks}
    except Exception:
        return set()


def _save_chunk_state(state_path: str, verified: set[int]) -> None:
    try:
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"verified": sorted(verified)}, fh)
        os.replace(tmp, state_path)
    except OSError:
        pass  # state is an optimization; losing it only costs re-verification


def _fetch_chunk_verified(url: str, part_path: str, index: int, size: int,
                          chunk_bytes: int, want_sha: str, on_bytes=None,
                          attempts: int = 4) -> None:
    """Fetch chunk ``index``, verify its SHA-256, write it at its offset."""
    import hashlib

    start = index * chunk_bytes
    end = min(start + chunk_bytes, size) - 1
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            req = _auth_request(url, {"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.getcode() != 206:
                    raise RuntimeError("server ignored Range request")
                buf = resp.read(end - start + 1)
            if len(buf) != end - start + 1:
                raise RuntimeError(f"short chunk {index}: {len(buf)}/{end - start + 1}")
            got = hashlib.sha256(buf).hexdigest()
            if got != want_sha:
                raise RuntimeError(f"chunk {index} hash mismatch")
            with open(part_path, "r+b") as fh:
                fh.seek(start)
                fh.write(buf)
            if on_bytes and i == 0:
                on_bytes(len(buf))
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(2 ** i, 8))
    raise last_exc or RuntimeError(f"failed chunk {index} of {url}")


def _ensure_part(part_path: str, size: int) -> None:
    """Preallocate the staging file — only if absent or wrong-sized, so a
    resumed pull never clobbers already-verified chunks."""
    os.makedirs(os.path.dirname(part_path) or ".", exist_ok=True)
    if not os.path.exists(part_path) or os.path.getsize(part_path) != size:
        with open(part_path, "wb") as fh:
            fh.truncate(size)


def fetch_from_central(central_url: str, model_key: str, progress=None) -> bool:
    """Pull a model's ENTIRE directory from central — parallel and segmented.

    Speed comes from two kinds of parallelism over central's existing ``/file``
    endpoint (which supports HTTP Range): small files download concurrently, and
    each large file is split into byte-range segments fetched concurrently, so a
    single multi-GB weights file isn't bottlenecked on one connection. Total
    simultaneous connections are capped at ``HUGPY_PULL_CONCURRENCY`` (default 8).

    Only files not already complete on disk are fetched (file-level resume).
    ``progress(done_bytes, total_bytes, name)`` reports aggregate bytes. Returns
    True only once every manifest file is present at its expected size (verified,
    with re-fetch of any gap); False if central lacks the model (404/409); raises
    if central has it but the transfer can't be completed.
    """
    from concurrent.futures import ThreadPoolExecutor

    base = central_url.rstrip("/") + "/api/llm/models/" + urllib.parse.quote(model_key)
    try:
        manifest = _get_json(base + "/manifest")
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 409):
            logger.info("central has no copy of %s (HTTP %s)", model_key, exc.code)
            return False
        raise                       # other HTTP status = a VERDICT — propagate it
    except urllib.error.URLError as exc:
        # No HTTP response reached us: central is UNREACHABLE. RAISE (don't
        # swallow to False) so _provision_now can tell "central down" (HF ok)
        # from "central refused" (HF forbidden). See CentralUnreachable.
        logger.warning("central unreachable for %s (%s)", model_key, exc)
        raise CentralUnreachable(exc) from exc

    dest = _local_destination(manifest)
    files = manifest.get("files") or []
    total = manifest.get("total_bytes") or sum((e.get("size") or 0) for e in files)
    concurrency = _pull_concurrency()

    # File-level resume: only fetch what isn't already complete. If nothing is
    # pending, this is a pure no-op (not even a Range probe).
    pending = [{"path": r, "size": s} for r, s, _w in _missing_or_short(dest, files)]
    if not pending:
        logger.info("%s already complete on disk (%d files)", model_key, len(files))
        return True

    # Range support lets us segment big files; probe once on the largest pending.
    ranged_ok = False
    biggest = max(pending, key=lambda e: e.get("size") or 0)
    if (biggest.get("size") or 0) >= _SEGMENT_MIN_BYTES:
        ranged_ok = _supports_range(
            base + "/file?path=" + urllib.parse.quote(biggest["path"]))

    logger.info("provisioning %s from central: %d files (%s), %d-way parallel"
                "%s -> %s", model_key, len(files), _human(total), concurrency,
                " (segmented)" if ranged_ok else "", dest)

    done_lock = threading.Lock()
    pstate = {"done": 0, "last": 0.0}

    def _on_bytes(n):
        if not progress:
            return
        with done_lock:
            pstate["done"] += n
            now = time.time()
            if now - pstate["last"] < 0.3 and pstate["done"] < total:
                return
            pstate["last"] = now
            done = pstate["done"]
        progress(min(done, total) if total else done, total, "files")

    # Per-file transfer context for the staged/verified scheme. Everything
    # downloads into <target>.part and is promoted by _finalize only after
    # verification — a crash mid-pull leaves a .part (+ chunk state), never a
    # plausible-looking final file.
    cb = _chunk_bytes()
    ctx: dict = {}   # rel -> {size, part, state, sums, verified, lock}

    def _build_units(entries):
        """Flatten entries into download units, capping total connections.

        Preferred unit is a VERIFIED CHUNK (central supplied per-chunk sums:
        fetch chunk-aligned range → hash → write → record). Fallbacks: plain
        byte-range segments (Range but no sums), then whole-file streaming —
        both still staged to .part and sealed by size+magic at finalize.
        """
        units = []
        for entry in entries:
            rel, size = entry["path"], entry.get("size")
            part = os.path.join(dest, rel) + ".part"
            c = ctx.setdefault(rel, {"lock": threading.Lock()})
            c.update(size=size, part=part, state=None, sums=None, verified=None)
            sums = None
            if ranged_ok and size and size >= _SEGMENT_MIN_BYTES:
                sums = _fetch_chunksums(base, rel, cb)
                if sums is not None and len(sums) != -(-size // cb):
                    logger.warning("chunksums length mismatch for %s (%d vs %d) — "
                                   "falling back to size+magic", rel, len(sums),
                                   -(-size // cb))
                    sums = None
            if sums is not None:
                _ensure_part(part, size)
                state_path = part + ".state.json"
                verified = _load_chunk_state(state_path, len(sums))
                c.update(state=state_path, sums=sums, verified=verified)
                units += [(rel, size, "chunk", i)
                          for i in range(len(sums)) if i not in verified]
            elif ranged_ok and size and size >= _SEGMENT_MIN_BYTES:
                _ensure_part(part, size)
                units += [(rel, size, "segment", se) for se in _segment_ranges(size)]
            else:
                units.append((rel, size, "whole", None))
        return units

    def _run_unit(unit):
        rel, size, kind, arg = unit
        url = base + "/file?path=" + urllib.parse.quote(rel)
        c = ctx[rel]
        try:
            if kind == "whole":
                _download_with_retry(url, c["part"], size, on_bytes=_on_bytes)
            elif kind == "segment":
                _download_segment_with_retry(url, c["part"], arg[0], arg[1],
                                             on_bytes=_on_bytes)
            else:  # verified chunk
                _fetch_chunk_verified(url, c["part"], arg, size, cb,
                                      c["sums"][arg], on_bytes=_on_bytes)
                with c["lock"]:
                    c["verified"].add(arg)
                    _save_chunk_state(c["state"], c["verified"])
        except Exception as exc:  # noqa: BLE001 — gate below re-fetches/decides
            logger.warning("download of %s (%s %s) failed: %s; will re-try in "
                           "verify pass", rel, kind, arg, exc)

    def _finalize(entries):
        """Promote fully-landed .part files onto their final names (atomic)."""
        for entry in entries:
            rel, size = entry["path"], entry.get("size")
            c = ctx.get(rel)
            if not c or not os.path.exists(c["part"]):
                continue
            part = c["part"]
            if size is not None and os.path.getsize(part) != size:
                continue                     # still incomplete — next pass resumes
            if c["sums"] is not None:
                with c["lock"]:
                    if len(c["verified"]) < len(c["sums"]):
                        continue             # unverified chunks remain
            if not _gguf_header_ok(part):
                # Chunk-verified content with a bad header means CENTRAL's copy
                # is corrupt — re-pulling would reproduce it. Fail loud.
                logger.error("%s: staged file fails the GGUF magic check — "
                             "central's copy may itself be corrupt", rel)
                for p in (part, c.get("state")):
                    if p:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                continue
            os.replace(part, os.path.join(dest, rel))
            if c.get("state"):
                try:
                    os.remove(c["state"])
                except OSError:
                    pass

    def _parallel(entries):
        units = _build_units(entries)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # Register this transfer pool so a restart shuts it down first (avoids
            # racing a pending pull into 'cannot schedule new futures'). The
            # agent installs the registrar (providers.set_executor_registrar);
            # a standalone provision has the no-op default.
            try:
                _providers.get_executor_registrar()(pool)
            except Exception:  # noqa: BLE001 — registration is best-effort
                pass
            list(pool.map(_run_unit, units))
        _finalize(entries)

    # Initial pass over the pending files (computed above).
    _parallel(pending)

    # Completeness gate: re-verify and re-fetch anything that didn't fully land.
    for _ in range(3):
        missing = _missing_or_short(dest, files)
        if not missing:
            break
        logger.warning("central transfer of %s incomplete: %d/%d files "
                       "missing/short; re-fetching", model_key, len(missing), len(files))
        _parallel([{"path": rel, "size": size} for rel, size, _why in missing])

    missing = _missing_or_short(dest, files)
    if missing:
        raise RuntimeError(
            f"central transfer of {model_key} incomplete: "
            f"{len(missing)}/{len(files)} files still missing/short "
            f"(e.g. {missing[0][0]} [{missing[0][2]}]) under {dest}")

    logger.info("provisioned %s from central in full (%d files, %s)",
                model_key, len(files), _human(total))
    return True


class _CountingReader:
    """Wrap a byte stream, reporting bytes read as download progress.

    tarfile in streaming mode pulls from this via ``read``; every pull advances
    a byte counter that reflects the actual network transfer (not file
    boundaries). Emits a throttled progress callback and a throttled journal log
    so a slow-but-moving transfer is visibly distinct from a real stall.
    """

    def __init__(self, fileobj, total, model_key, on_progress=None):
        self._f = fileobj
        self._total = int(total or 0)
        self._model_key = model_key
        self._on_progress = on_progress
        self._done = 0
        self._last_emit = 0.0
        self._last_log = 0.0

    def read(self, size=-1):
        chunk = self._f.read(size)
        if chunk:
            self._done += len(chunk)
            now = time.time()
            done = min(self._done, self._total) if self._total else self._done
            if self._on_progress and now - self._last_emit > 0.5:
                self._last_emit = now
                try:
                    self._on_progress(done, self._total, "archive")
                except Exception:  # progress is best-effort; never break the read
                    pass
            if now - self._last_log > 5.0:
                self._last_log = now
                pct = (100.0 * done / self._total) if self._total else 0.0
                logger.info("downloading %s archive: %s / %s (%.0f%%)",
                            self._model_key, _human(self._done),
                            _human(self._total), pct)
        return chunk


def fetch_archive_from_central(central_url: str, model_key: str, progress=None) -> bool:
    """Pull the model's ENTIRE directory from central as one streamed tar.

    Downloads central's ``/archive`` endpoint and extracts it on the fly (no
    temp tar on disk, bounded memory), confining every member to the model's
    destination, then verifies the result against central's manifest. This is
    the primary transport: one sequential stream can't "drop" files the way N
    independent GETs can.

    Returns True once the directory is present in full. Returns False if central
    can't serve an archive — the endpoint is missing on an older central
    (404/405) or central doesn't have the model (404/409) — so the caller can
    fall back to the per-file transfer. Raises if the archive arrived but
    couldn't be completed (so a partial directory is never reported as success).
    """
    import tarfile

    base = central_url.rstrip("/") + "/api/llm/models/" + urllib.parse.quote(model_key)

    # Manifest gives us the destination + the file set to verify against.
    try:
        manifest = _get_json(base + "/manifest")
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 409):
            logger.info("central has no copy of %s (HTTP %s)", model_key, exc.code)
            return False
        raise                       # other HTTP status = a VERDICT — propagate it
    except urllib.error.URLError as exc:
        logger.warning("central unreachable for %s (%s)", model_key, exc)
        raise CentralUnreachable(exc) from exc

    dest = _local_destination(manifest)
    files = manifest.get("files") or []
    total = manifest.get("total_bytes") or sum((e.get("size") or 0) for e in files)
    dest_real = os.path.realpath(dest)
    os.makedirs(dest, exist_ok=True)

    logger.info("provisioning %s from central archive: %d files (%s) -> %s",
                model_key, len(files), _human(total), dest)

    try:
        resp = urllib.request.urlopen(_auth_request(base + "/archive"), timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 405, 409):
            logger.info("central has no archive endpoint for %s (HTTP %s); "
                        "will use per-file transfer", model_key, exc.code)
            return False
        raise                       # other HTTP status = a VERDICT — propagate it
    except urllib.error.URLError as exc:
        logger.warning("central archive unreachable for %s (%s)", model_key, exc)
        raise CentralUnreachable(exc) from exc

    # Wrap the stream so progress reflects BYTES read off the socket — the true
    # download rate — rather than ticking only when a whole file finishes
    # extracting (which sat at 0% while a multi-GB file streamed). Also logs
    # throughput to the journal so a real stall is visible vs. a slow file.
    reader = _CountingReader(resp, total, model_key, on_progress=progress)

    # mode "r|" = sequential streaming read, matching central's "w|" writer.
    with resp, tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            # LEXICAL containment check (normpath, NOT realpath). realpath
            # follows an EXISTING symlink already sitting at the target — and
            # comfy checkpoints ARE symlinks — so a prior link made realpath
            # resolve outside dest and false-flagged safe bare filenames like
            # 'model.safetensors' as "unsafe path", blocking every comfy pull
            # and driving the reconcile storm. normpath catches real traversal
            # ('..'/absolute) without chasing a link at the destination.
            target = os.path.normpath(os.path.join(dest_real, member.name))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise RuntimeError(f"unsafe path in archive member: {member.name!r}")
            # A symlink/hardlink/device MEMBER *inside* the tar is the genuine
            # traversal threat (central packs only regular files + dirs).
            if member.issym() or member.islnk():
                raise RuntimeError(f"refusing non-regular archive member: {member.name!r}")
            # A symlink already at the target (a prior comfy link) must be
            # removed first, or tar.extract writes THROUGH it to its
            # outside-dest target. Replace it with the real bytes.
            if os.path.islink(target) or os.path.isfile(target):
                try:
                    os.unlink(target)
                except OSError:
                    pass
            tar.extract(member, dest)

    if progress:
        progress(total, total, "archive")  # final 100%

    # Completeness gate against the manifest.
    missing = _missing_or_short(dest, files)
    if missing:
        raise RuntimeError(
            f"central archive of {model_key} incomplete after extract: "
            f"{len(missing)}/{len(files)} files missing/short "
            f"(e.g. {missing[0][0]} [{missing[0][2]}]) under {dest}")

    logger.info("provisioned %s from central archive in full (%d files, %s)",
                model_key, len(files), _human(total))
    return True


def _human(n) -> str:
    if not n:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.1f} {units[i]}"


def fetch_from_hf(model_key: str) -> str:
    """Last-resort: pull from Hugging Face via the normal code path."""
    from hugpy_storage.download_models import ensure_model

    logger.info("provisioning %s from Hugging Face", model_key)
    return ensure_model(model_key)


def central_total_bytes(central_url: str | None, model_key: str) -> int | None:
    """The model's total on-disk size per central's manifest, or None.

    The SIZE OF THE PULL, known BEFORE a byte is transferred — the input the
    budget check needs to answer "will this fit?" without discovering the answer
    from [Errno 28] halfway through. Cheap: the manifest is a small JSON that
    every central pull already fetches first. Returns None when central can't
    say (no URL / unreachable / 404 / no size in the manifest), which the caller
    treats as "unknown size — pull as before" rather than as zero.
    """
    if not central_url:
        return None
    base = central_url.rstrip("/") + "/api/llm/models/" + urllib.parse.quote(model_key)
    try:
        manifest = _get_json(base + "/manifest")
    except Exception:  # noqa: BLE001 — unknown size is a valid answer here
        return None
    if not isinstance(manifest, dict):
        return None
    total = manifest.get("total_bytes")
    if not total:
        total = sum((e.get("size") or 0) for e in (manifest.get("files") or []))
    try:
        total = int(total)
    except (TypeError, ValueError):
        return None
    return total or None


def ensure_model_present(model_key: str, central_url: str | None, progress=None,
                         state=None, purpose: str | None = None) -> bool:
    """Make sure model_key is on local disk. Central-first, then HF fallback.

    ``progress(done_bytes, total_bytes, filename)`` is forwarded to the central
    download so callers can stream provisioning status. Returns True if the
    model is present (or already was), False if it could not be provisioned.

    ``state`` (the WorkerState) opts this pull into the STORAGE BUDGET: before
    any bytes move, the worker FIFO-evicts cold models to make room for this
    one, and REFUSES the pull outright (raising ``budget.BudgetRefusal``) if
    even a full eviction can't seat it. Omitted/None -> no budget check, i.e.
    byte-for-byte the pre-feature behavior (standalone/CLI provisions).

    ``purpose`` labels WHY this pull is happening for central's budget handshake
    (2026-07-17): "demand" (a real call — never budget-refused centrally),
    "reconcile"/"assign" (background pre-fetch — budget-refusable at 409), or
    None (old behavior; central treats it as demand during rollout). It rides a
    thread-local onto the X-Transfer-Purpose header of every transfer request.
    """
    # Teach the worker about the model first (central is the source of truth),
    # then provision against the canonical local key. This is what lets the
    # worker serve a model it wasn't built with.
    with _purpose_scope(purpose):
        return _ensure_model_present_inner(model_key, central_url,
                                           progress=progress, state=state)


def _ensure_model_present_inner(model_key: str, central_url: str | None,
                                progress=None, state=None) -> bool:
    canonical = ensure_model_registered(model_key, central_url) or model_key

    if model_is_local(canonical):
        return True

    # Single-flight: serialize provisioning of this model so concurrent callers
    # (multiple infer requests + the pre-provision) don't each download it in
    # parallel into the same directory.
    lock = _provision_lock(canonical)
    if not lock.acquire(blocking=False):
        logger.info("provisioning of %s already in progress; waiting for it",
                    canonical)
        lock.acquire()
    try:
        # Another thread may have finished the download while we waited.
        if model_is_local(canonical):
            logger.info("%s became available while waiting; using it", canonical)
            return True
        # STORAGE BUDGET (incident 2026-07-16 — op filled to 0 bytes free; and the
        # 2026-07-17 ae incident — an UNGATED entry pulled chunks onto an over-cap
        # store while a different pid was actively REFUSING the same model).
        #
        # Part A (slice 8): the gate is now UNCONDITIONAL — every transfer entry
        # runs it before the first byte, not only callers that threaded a state.
        # A state-less caller (/redownload; the slot child, a separate process)
        # resolves a budget state via _resolve_budget_state (the agent's live
        # state in-process, else a minimal env/disk state) so no path can download
        # atop the cap. Evict-to-fit / refuse runs INSIDE the single-flight lock:
        # the check must see (and its evictions must land) without racing another
        # pull of the same key. Raises BudgetRefusal when even a full FIFO can't
        # seat the model — the caller surfaces MISSING-with-a-reason.
        #
        # RESUME SEMANTICS (do not re-refuse an ALREADY-ADMITTED pull into a
        # wedge): fit_plan counts the bytes THIS model already has on disk as
        # headroom it doesn't need to re-take (its `have`/`delta` split), so a
        # resumed/partial pull asks the gate only for its REMAINING delta, not the
        # whole size again. A model that was admitted, started, and is resuming
        # therefore re-passes the gate for free (delta shrinks as .part grows) and
        # is never refused into a half-downloaded wedge. The verify-pass re-fetch
        # (fetch_from_central's _run_unit) lives BELOW this gate, inside an
        # already-admitted _provision_now, so it never re-enters here.
        gate_state = _resolve_budget_state(state)
        evict_to_fit = _providers.get_budget_gate()
        need = central_total_bytes(central_url, canonical)
        if need:
            evict_to_fit(gate_state, canonical, need)
        else:
            logger.info("budget: no manifest size for %s — pulling without a "
                        "fit check (size unknown)", canonical)
        return _provision_now(canonical, central_url, progress=progress)
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Provisioning telemetry + the preserved failure CAUSE.
#
# THE 2026-07-28 INCIDENT, in one sentence: computron's drive was 100% full,
# ``fetch_from_central`` and the archive fallback both died with
# ``OSError: [Errno 28] No space left on device``, ``_provision_now`` knew that
# exactly — it is right there in ``central_reason`` — and then returned a bare
# ``False``, at which point every word of the diagnosis was gone. The operator's
# chat said "could not fetch model X from central or HF" and finding the truth
# cost an ssh session and a journalctl read.
#
# So: the reason survives the boolean boundary now. ``_provision_now`` records a
# structured cause here before returning False, and the agent's streaming error
# path reads it to compose an honest one-line message. Signatures are unchanged
# — a stashed cause is additive, where widening the return type would touch six
# call sites across two processes for no gain.
#
# Bounded and self-trimming: a cause is only interesting until the next attempt
# at the same model, and a worker that fails to provision thousands of distinct
# keys must not grow a dict forever.
# ---------------------------------------------------------------------------

_FAILURES: dict = {}
_FAILURES_LOCK = threading.Lock()
_FAILURES_MAX = 256


class _SafeTelemetry:
    """Wraps the installed telemetry sink so a missing method or a raising
    emitter is a no-op: telemetry is never a reason for a provision to fail."""

    def __init__(self, sink):
        self._sink = sink

    def __getattr__(self, name):
        fn = getattr(self._sink, name, None)
        if fn is None:
            return lambda *a, **k: None
        def _call(*a, **k):
            try:
                return fn(*a, **k)
            except Exception:  # noqa: BLE001
                return None
        return _call


def _evt():
    """The telemetry emitter (the fleet's ``central.evictions`` when the agent
    installed it via providers.set_transfer_telemetry), or None. This module
    runs inside the slot child and in standalone CLI provisions where no sink
    exists, and telemetry is never a reason for a provision to fail."""
    sink = _providers.get_transfer_telemetry()
    return _SafeTelemetry(sink) if sink is not None else None


def _dest_hint(model_key: str) -> str | None:
    """Best-effort local destination for ``model_key`` — the path whose
    FILESYSTEM we report free/total for. Only a hint: ``disk_stats`` walks up to
    the nearest existing ancestor, so the models root answers the question even
    when the model's own directory was never created."""
    try:
        return _model_dir(model_key)
    except Exception:  # noqa: BLE001
        return None


def _record_failure(model_key: str, source: str, reason: str,
                    exc: BaseException | None = None,
                    dest_path: str | None = None) -> dict:
    """Remember WHY provisioning failed, in a form a human can read.

    ``human`` is the operator-grade one-liner ("disk full (ENOSPC) on
    /mnt/storage — 0 B free of 938 GB") when the failure is one we can say
    something sharp about; otherwise it is the raw reason, which is still far
    better than the flattened "from central or HF"."""
    human = ""
    errno_nm = ""
    if exc is not None:
        try:
            errno_nm = errno_name(exc) or ""
            human = describe_disk_error(exc, dest_path) or ""
        except Exception:  # noqa: BLE001
            human, errno_nm = "", ""
    rec = {
        "model_key": model_key, "source": source, "reason": reason,
        "errno_name": errno_nm or None,
        "error_class": type(exc).__name__ if exc is not None else None,
        "human": human or reason,
        "dest_path": dest_path,
        "ts": time.time(),
    }
    try:
        with _FAILURES_LOCK:
            if len(_FAILURES) >= _FAILURES_MAX:
                oldest = sorted(_FAILURES.items(),
                                key=lambda kv: kv[1].get("ts") or 0)[:_FAILURES_MAX // 2]
                for k, _v in oldest:
                    _FAILURES.pop(k, None)
            _FAILURES[str(model_key)] = rec
    except Exception:  # noqa: BLE001
        pass
    return rec


def last_failure(model_key: str) -> dict | None:
    """The most recent recorded provisioning failure for ``model_key``, or None.

    Read by the agent's streaming error path so the chat message can name the
    actual cause instead of the generic "could not fetch"."""
    with _FAILURES_LOCK:
        rec = _FAILURES.get(str(model_key))
    return dict(rec) if rec else None


def clear_failure(model_key: str) -> None:
    """Forget a recorded cause — called on a successful provision so a stale
    "disk full" can never be reported against a model that has since landed."""
    with _FAILURES_LOCK:
        _FAILURES.pop(str(model_key), None)


def _provision_now(canonical: str, central_url: str | None, progress=None) -> bool:
    """Do the actual fetch (central archive -> per-file -> HF). Caller holds the
    per-model provisioning lock.

    CHAIN OF COMMAND (operator ruling, ae 1.2TB incident 2026-07-17): central's
    verdict is AUTHORITATIVE. If central answered at all — a refusal (404/409/
    archive-refused) OR any other HTTP status — it is ALIVE and its "no" STANDS:
    this returns False WITHOUT trying HF. HF is a SURVIVAL fallback, reached ONLY
    when central was UNREACHABLE (no HTTP response: CentralUnreachable) or no
    central URL is configured. The distinction is by EXCEPTION TYPE, never by
    string-matching the reason. Escape hatch: HUGPY_HF_FALLBACK=always restores
    the old any-failure-falls-through behavior for emergencies.
    """
    # Daylight item 4: the chosen SOURCE must be loud — logged AND streamed
    # through the progress callback (a "source=…" marker the agent stores on
    # provision_progress), so a silent 55GB HF pull while central held the
    # files can never happen unnoticed again.
    hf_ok = _hf_fallback_always()            # escape hatch pre-decides HF
    central_reason = "no central URL configured"
    # Telemetry (observation only — every call below is total; see comms/
    # evictions.py). The run scope JOINS whatever pass is already open so the
    # provisioning rows land in the SAME console card as the eviction rows that
    # follow, which is the whole point: one card = one request = where it died.
    ev = _evt()
    dest = _dest_hint(canonical)
    scope = ev.serve_scope() if ev is not None else None
    if scope is not None and not hasattr(scope, "__enter__"):
        scope = None                     # sink without a run scope
    if scope is not None:
        scope.__enter__()
    try:
        return _provision_sources(canonical, central_url, progress, ev, dest)
    finally:
        if scope is not None:
            try:
                scope.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass


def _provision_sources(canonical: str, central_url: str | None, progress,
                       ev, dest: str | None) -> bool:
    """The source chain itself. Split out of ``_provision_now`` only so the
    telemetry run-scope wraps it without indenting the whole decision body."""
    hf_ok = _hf_fallback_always()            # escape hatch pre-decides HF
    central_reason = "no central URL configured"

    def _t_start(source: str):
        if ev is not None:
            ev.emit_provision_start(canonical, source, dest_path=dest)

    def _t_done(source: str, t0: float):
        if ev is not None:
            ev.emit_provision_done(canonical, source,
                                   duration_ms=int((time.time() - t0) * 1000))
        clear_failure(canonical)
        # Weights landed: the catalog must re-read this row.
        publish_catalog_changed(f"provisioned from {source}", model_key=canonical,
                                destination=dest, change="download", source=source)

    # Has THIS attempt already captured an errno-bearing (i.e. OS-level) cause?
    # A full disk fails identically from every source, so the first source to
    # hit ENOSPC has already diagnosed the box; a later "HF returned 404" is a
    # symptom, not the cause, and must not overwrite it.
    sharp = {"seen": False}

    def _t_fail(source: str, reason: str, exc: BaseException | None = None):
        """Emit provision.fail AND remember the cause for the chat message.

        These two always happen together — the operator watching the feed and
        the operator reading the chat must never be told different stories."""
        if ev is not None:
            ev.emit_provision_fail(canonical, source, exc=exc, dest_path=dest,
                                   detail=reason)
        has_errno = bool(exc is not None and errno_name(exc))
        if sharp["seen"] and not has_errno:
            return                      # don't downgrade an OS-level diagnosis
        _record_failure(canonical, source, reason, exc=exc, dest_path=dest)
        if has_errno:
            sharp["seen"] = True

    if central_url is None:
        # No central configured at all: HF is the only source (a worker cut off
        # from the fleet). This is the survival path, not a side door.
        hf_ok = True
    else:
        central_alive = False                # did central give us an HTTP verdict?
        # 1) parallel + segmented per-file transfer — fastest (saturates the
        #    link; a big weights file is split across many connections).
        if progress:
            progress(0, 0, "source=central")
        _t0 = time.time()
        _t_start("central-transfer")
        try:
            if fetch_from_central(central_url, canonical, progress=progress):
                logger.info("PROVENANCE: %s provisioned from CENTRAL (parallel)",
                            canonical)
                _t_done("central-transfer", _t0)
                return True
            central_alive = True             # returned False = a 409/empty VERDICT
            central_reason = "central does not have the files (per-file 409/empty)"
            _t_fail("central-transfer", central_reason)
        except CentralUnreachable as exc:
            central_reason = f"central unreachable (parallel): {exc}"
            _t_fail("central-transfer", central_reason, exc)
            logger.warning("central parallel transfer of %s: central UNREACHABLE "
                           "(%s); trying archive", canonical, exc)
        except Exception as exc:
            central_alive = True             # any other error = central RESPONDED
            central_reason = f"parallel transfer failed: {type(exc).__name__}: {exc}"
            _t_fail("central-transfer", central_reason, exc)
            logger.warning("central parallel transfer of %s failed: %s; "
                           "trying archive", canonical, exc)
        # 2) whole-directory tar stream — single-connection fallback.
        _t0 = time.time()
        _t_start("archive")
        try:
            if fetch_archive_from_central(central_url, canonical, progress=progress):
                logger.info("PROVENANCE: %s provisioned from CENTRAL (archive)",
                            canonical)
                _t_done("archive", _t0)
                return True
            central_alive = True
            central_reason = "central cannot provide the files (archive refused)"
            _t_fail("archive", central_reason)
        except CentralUnreachable as exc:
            central_reason = f"central unreachable (archive): {exc}"
            _t_fail("archive", central_reason, exc)
            logger.warning("central archive of %s: central UNREACHABLE (%s)",
                           canonical, exc)
        except Exception as exc:
            central_alive = True
            central_reason = f"archive transfer failed: {type(exc).__name__}: {exc}"
            _t_fail("archive", central_reason, exc)

        # THE GATE. Central ALIVE (any verdict) => its "no" is authoritative;
        # NO HF, unless the emergency escape hatch is set. Central UNREACHABLE
        # on BOTH transports (central_alive stayed False) => HF survival path ok.
        if not central_alive:
            hf_ok = True
        if central_alive and not hf_ok:
            logger.error("PROVENANCE: %s NOT provisioned — central gave a verdict "
                         "and HF is forbidden by chain of command (2026-07-17): "
                         "%s. Set HUGPY_HF_FALLBACK=always to override.",
                         canonical, central_reason)
            if progress:
                progress(0, 0, f"source=refused ({central_reason[:120]})")
            # KEEP the sharper cause already recorded by the parallel/archive
            # attempt. Those recorded an exception and therefore an errno; this
            # gate has only a prose reason, and overwriting an "ENOSPC / disk
            # full" record with "central gave a verdict" would re-flatten the
            # exact diagnosis this whole change exists to preserve. Only record
            # here when nothing sharper was captured.
            if not (last_failure(canonical) or {}).get("errno_name"):
                _record_failure(canonical, "central-transfer",
                                f"{central_reason} (HF fallback forbidden by "
                                f"chain of command)", dest_path=dest)
            return False

    logger.warning("PROVENANCE: %s falling back to HUGGING FACE — %s",
                   canonical, central_reason)
    _t0 = time.time()
    _t_start("hf")
    try:
        if progress:
            progress(0, 0, f"source=hf ({central_reason[:120]})")
        fetch_from_hf(canonical)
        logger.warning("PROVENANCE: %s provisioned from HUGGING FACE (%s)",
                       canonical, central_reason)
        _t_done("hf", _t0)
        return True
    except Exception as exc:
        # LAST source exhausted. This failure — not the central one that
        # preceded it — is the one the operator's chat should name, unless HF
        # merely repeated a local problem (a full disk fails identically from
        # every source, and "disk full" beats "HF said 404" every time).
        _t_fail("hf", f"HF fetch failed: {type(exc).__name__}: {exc}", exc)
        logger.error("could not provision %s from central or HF: %s "
                     "(central: %s)", canonical, exc, central_reason)
        return False

