"""Central EXTERNAL MODEL METADATA store (SQLite, stdlib-only, fetch-once).

Two providers ride the same store and the same operator policy:

    HuggingFace — PER-REPO metadata (repo_info, repo_files),
    Civitai     — PER-MODEL metadata (civitai_meta, keyed by checkpoint stem).

RENAME NOTE (2026-07-22, keeper nomenclature ruling): this module was born as
``comms/hf_metadata.py`` with class ``HfMetadataStore`` / singleton
``hf_metadata_store`` / env ``HUGPY_HF_CACHE_DB`` / db ``hf_metadata.db``.
Widening it to Civitai made the HF-specific names wrong, so it is now
``comms/model_metadata.py`` / ``ModelMetadataStore`` / ``model_metadata_store``
/ ``HUGPY_MODEL_METADATA_DB`` / ``model_metadata.db``. Backward-compat shims,
all three layers: (a) ``comms/hf_metadata.py`` survives as a thin re-export so
old imports keep working, (b) the old env var is honored when the new one is
unset, (c) on first store init an existing ``hf_metadata.db`` beside an absent
``model_metadata.db`` is renamed in place (one-shot migration, best-effort).

Operator policy, ratified verbatim: **a cache miss queries the provider once;
the response is logged in a central SQLite DB; that model/repo is
theoretically never pinged again.** Two reasons:

    privacy    — repeated PER-REPO metadata lookups enumerate the fleet's
                 held model inventory to huggingface.co; one fetch per repo,
                 ever, minimizes that signal,
    reliability — anonymous rate limits and API flakiness turn every extra
                 metadata call into a 502 waiting to happen.

So there is NO TTL, NO background refresh, NO expiry sweep. The ONLY re-fetch
paths are explicit operator affordances: ``forget(hub_id)`` (DELETE
/hf/cache/<hub_id>) which drops the rows so the next access re-fetches, and a
``force=True`` / ``?refresh=1`` read-through that overwrites in place.

Deliberately OUT of scope (operator ruling 2026-07-22, "search is fine to
ping"): /search's ``list_models`` query stays LIVE on every call. Search is
discovery of NEW things — staleness there is user-visible — and a search
query names an interest, not the held inventory. Only per-repo facts
(model_info, list_repo_files) are cached here. /search's per-row SIZE
enrichment does ride this cache: that's per-repo metadata, not search.

Design (mirrors ``comms/shared.py`` SqliteMirror — the proven cross-process
idiom): one short-lived connection per op, WAL, idempotent CREATE TABLE IF NOT
EXISTS, additive ``_migrate()`` guarded by a PRAGMA table_info probe,
self-disabling after MAX_FAILURES. Like ``comms/calibration.py`` this DB is
DURABLE by default (a fetched fact should survive restarts — that's the whole
point) — ``HUGPY_MODEL_METADATA_DB`` if set (legacy ``HUGPY_HF_CACHE_DB``
honored when the new var is unset), else
``$PROJECTS_HOME/model_metadata.db``.

CRITICAL degradation semantics (this differs from the gate-like stores): this
is a CACHE, never a gate. Any store failure — disabled, locked, corrupt,
unwritable path — must fall through to the live HF call and, where possible,
still attempt the write-back. A broken cache DB must never break /search,
/hf/spec, or downloads; the worst case is exactly today's world (live calls).

``fetch_repo_info()`` at module level is the single read-through funnel every
repo-info consumer goes through (sizes, spec, resolve, download estimates) —
one rich fetch (files_metadata=True) serves all future callers.

``fetch_civitai_meta()`` is the Civitai analogue with one deliberate contract
difference: it NEVER propagates a live-call error (enrichment is best-effort
decoration; no caller has a failure contract on it) and it NEVER searches by
filename guess — a miss without a held civitai/version id returns None,
because metadata rides ids we actually hold, never speculative egress.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Optional

from hugpy_control.shared import retry_on_emfile

logger = logging.getLogger(__name__)

# Telemetry off at the earliest shared import point. setdefault ONLY — an
# operator-set value (either way) is never clobbered. This keeps hf_hub from
# phoning home usage headers on the calls we do still make.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MAX_FAILURES = 5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repo_info (
    hub_id     TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    revision   TEXT,
    fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS repo_files (
    hub_id     TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS civitai_meta (
    stem       TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
"""

_LEGACY_DB_BASENAME = "hf_metadata.db"


def default_db_path() -> str:
    # New env wins; the legacy var is honored when the new one is unset
    # (rename shim layer b).
    env = (os.environ.get("HUGPY_MODEL_METADATA_DB") or "").strip()
    if not env:
        env = (os.environ.get("HUGPY_HF_CACHE_DB") or "").strip()
    if env:
        return env
    # Consolidated under HUGPY_HOME (~/.hugpy/state); migrates a legacy
    # ~/model_metadata.db in place on first resolve.
    from hugpy_platform import app_dirs as _hp
    return _hp.model_metadata_db()


def serialize_model_info(info: Any) -> dict:
    """Defensively flatten a huggingface_hub ModelInfo into a plain JSON-native
    dict carrying every field the codebase consumes — id/sha, pipeline_tag,
    library_name, tags, license (from cardData), gated, downloads/likes,
    last_modified iso, safetensors param total, transformers auto_model, and
    siblings as [{rfilename, size}]. Store EVERYTHING so one fetch serves all
    future callers (sizes, spec, resolve). Every getattr is guarded — HF has
    changed these shapes before and a missing field must degrade to None, not
    poison the cache write."""
    def _iso(v: Any) -> "str | None":
        if v is None:
            return None
        try:
            return v.isoformat()
        except AttributeError:
            s = str(v)
            return s or None

    card = getattr(info, "card_data", None)

    def _card(key: str) -> Any:
        if card is None:
            return None
        try:
            if isinstance(card, dict):
                return card.get(key)
            d = card.to_dict() if hasattr(card, "to_dict") else None
            if isinstance(d, dict) and key in d:
                return d.get(key)
            return getattr(card, key, None)
        except Exception:  # noqa: BLE001
            return getattr(card, key, None)

    siblings = []
    for s in (getattr(info, "siblings", None) or []):
        try:
            siblings.append({"rfilename": getattr(s, "rfilename", None),
                             "size": getattr(s, "size", None)})
        except Exception:  # noqa: BLE001
            continue

    st = getattr(info, "safetensors", None)
    ti = getattr(info, "transformers_info", None)
    languages = _card("language")
    if isinstance(languages, str):
        languages = [languages]

    out = {
        "id": getattr(info, "id", None) or getattr(info, "modelId", None),
        "sha": getattr(info, "sha", None),
        "author": getattr(info, "author", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library_name": getattr(info, "library_name", None),
        "tags": list(getattr(info, "tags", None) or []),
        "license": _card("license"),
        "languages": languages,
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "last_modified": _iso(getattr(info, "last_modified", None)),
        "created_at": _iso(getattr(info, "created_at", None)),
        "safetensors_params": getattr(st, "total", None) if st else None,
        "auto_model_class": getattr(ti, "auto_model", None) if ti else None,
        "siblings": siblings,
    }
    # Round-trip through json to guarantee the dict is storable — any exotic
    # value degrades to its str() rather than failing the whole serialization.
    try:
        json.dumps(out)
        return out
    except (TypeError, ValueError):
        return json.loads(json.dumps(out, default=str))


def sum_sibling_sizes(payload: "dict | None") -> "int | None":
    """Total repo bytes from a cached repo_info payload (None when unknown)."""
    if not isinstance(payload, dict):
        return None
    total = sum((s.get("size") or 0) for s in (payload.get("siblings") or [])
                if isinstance(s, dict))
    return total or None


class ModelMetadataStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self._explicit_path = path
        self._path: Optional[str] = path
        self._failures = 0
        self._disabled = False
        self._init_lock = threading.Lock()
        self._initialized = False

    @property
    def path(self) -> str:
        # Resolved lazily so tests can point HUGPY_MODEL_METADATA_DB (or the
        # legacy HUGPY_HF_CACHE_DB) at a temp file after import; frozen at
        # first use.
        if self._path is None:
            self._path = default_db_path()
        return self._path

    def _maybe_migrate_legacy_file(self) -> None:
        """Rename shim layer c: if the resolved path is the NEW default name,
        it does not exist yet, and the OLD ``hf_metadata.db`` sits beside it
        (live rows from the pre-rename dev cut), rename it in place — one
        shot, best-effort, degradation rules apply (any failure just means we
        start a fresh DB; the old file stays for manual recovery)."""
        try:
            p = self.path
            if os.path.exists(p):
                return
            if os.path.basename(p) != "model_metadata.db":
                return  # explicit/env path — never guess a sibling migration
            legacy = os.path.join(os.path.dirname(p) or ".",
                                  _LEGACY_DB_BASENAME)
            if os.path.exists(legacy):
                os.rename(legacy, p)
                # WAL sidecars carry not-yet-checkpointed rows — they must
                # travel with the main file or recent writes vanish.
                for suffix in ("-wal", "-shm"):
                    if os.path.exists(legacy + suffix):
                        os.rename(legacy + suffix, p + suffix)
                logger.info("model metadata store: migrated legacy %s -> %s",
                            legacy, p)
        except Exception as exc:  # noqa: BLE001 — never a gate
            logger.warning("model metadata store: legacy db migration "
                           "skipped: %s", exc)

    # -- plumbing ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # Retry the store-open past the restart-burst EMFILE (see
        # comms.shared.retry_on_emfile) before running the handle-local PRAGMAs.
        from hugpy_control.shared import connect_wal
        return connect_wal(self.path, retry=retry_on_emfile)

    def _ensure(self) -> bool:
        if self._disabled:
            return False
        if self._initialized:
            return True
        with self._init_lock:
            if self._initialized:
                return True
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                self._maybe_migrate_legacy_file()
                with self._connect() as conn:
                    conn.executescript(_SCHEMA)
                    self._migrate(conn)
                self._initialized = True
                return True
            except Exception as exc:  # noqa: BLE001
                self._note_failure("init", exc)
                return False

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotent, safe on an existing populated DB. Additive columns only,
        guarded by a table_info probe so a second run is a no-op and a DB
        already carrying a column never raises. (No migrations yet — the probe
        pattern is here so future additive columns follow the proven idiom.)"""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(repo_info)")}
        if "revision" not in cols:
            conn.execute("ALTER TABLE repo_info ADD COLUMN revision TEXT")

    def _note_failure(self, op: str, exc: Exception) -> None:
        self._failures += 1
        if self._failures >= MAX_FAILURES and not self._disabled:
            self._disabled = True
            logger.error("model metadata cache DISABLED after %d failures "
                         "(last: %s during %s) — metadata lookups degrade to "
                         "live provider calls until restart",
                         self._failures, exc, op)
        else:
            logger.warning("model metadata cache %s failed: %s", op, exc)

    def _ok(self) -> None:
        self._failures = 0

    # -- repo_info -----------------------------------------------------------
    def get_repo_info(self, hub_id: str) -> "dict | None":
        """Cached serialized model_info dict, or None (miss OR store failure —
        indistinguishable by design: both mean 'go live')."""
        if not hub_id or not self._ensure():
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload FROM repo_info WHERE hub_id=?",
                    (hub_id,)).fetchone()
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("get_repo_info", exc)
            return None
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:  # noqa: BLE001 — a corrupt row is a miss, not a break
            return None

    def put_repo_info(self, hub_id: str, payload: dict,
                      revision: "str | None" = None) -> None:
        if not hub_id or not isinstance(payload, dict) or not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO repo_info (hub_id, payload, revision, fetched_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(hub_id) DO UPDATE SET "
                    "  payload=excluded.payload, revision=excluded.revision, "
                    "  fetched_at=excluded.fetched_at",
                    (hub_id, json.dumps(payload), revision, time.time()))
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("put_repo_info", exc)

    # -- repo_files ----------------------------------------------------------
    def get_repo_files(self, hub_id: str) -> "list | None":
        if not hub_id or not self._ensure():
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload FROM repo_files WHERE hub_id=?",
                    (hub_id,)).fetchone()
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("get_repo_files", exc)
            return None
        if not row:
            return None
        try:
            files = json.loads(row[0])
        except Exception:  # noqa: BLE001
            return None
        return files if isinstance(files, list) else None

    def put_repo_files(self, hub_id: str, files: list) -> None:
        if not hub_id or not isinstance(files, list) or not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO repo_files (hub_id, payload, fetched_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(hub_id) DO UPDATE SET "
                    "  payload=excluded.payload, fetched_at=excluded.fetched_at",
                    (hub_id, json.dumps(files), time.time()))
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("put_repo_files", exc)

    # -- civitai_meta (per-model, keyed by checkpoint filename stem) ---------
    def get_civitai_meta(self, stem: str) -> "dict | None":
        """Cached Civitai metadata dict for one checkpoint stem, or None
        (miss OR store failure — indistinguishable by design)."""
        if not stem or not self._ensure():
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload FROM civitai_meta WHERE stem=?",
                    (stem,)).fetchone()
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("get_civitai_meta", exc)
            return None
        if not row:
            return None
        try:
            payload = json.loads(row[0])
        except Exception:  # noqa: BLE001 — corrupt row is a miss, not a break
            return None
        return payload if isinstance(payload, dict) else None

    def put_civitai_meta(self, stem: str, payload: dict) -> None:
        if not stem or not isinstance(payload, dict) or not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO civitai_meta (stem, payload, fetched_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(stem) DO UPDATE SET "
                    "  payload=excluded.payload, fetched_at=excluded.fetched_at",
                    (stem, json.dumps(payload), time.time()))
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("put_civitai_meta", exc)

    # -- operator affordances ------------------------------------------------
    def forget(self, hub_id: str) -> int:
        """The explicit-refresh hatch: drop every cached row for one repo so
        the next access re-fetches live. Returns the number of rows deleted."""
        if not hub_id or not self._ensure():
            return 0
        try:
            with self._connect() as conn:
                n = 0
                n += conn.execute(
                    "DELETE FROM repo_info WHERE hub_id=?", (hub_id,)).rowcount
                n += conn.execute(
                    "DELETE FROM repo_files WHERE hub_id=?", (hub_id,)).rowcount
            self._ok()
            return n
        except Exception as exc:  # noqa: BLE001
            self._note_failure("forget", exc)
            return 0

    def stats(self) -> dict:
        """Observability: row counts + where the DB lives (GET /hf/cache)."""
        out = {"db_path": self.path, "repos": 0, "file_lists": 0,
               "civitai_models": 0, "disabled": self._disabled}
        if not self._ensure():
            return out
        try:
            with self._connect() as conn:
                out["repos"] = conn.execute(
                    "SELECT COUNT(*) FROM repo_info").fetchone()[0]
                out["file_lists"] = conn.execute(
                    "SELECT COUNT(*) FROM repo_files").fetchone()[0]
                out["civitai_models"] = conn.execute(
                    "SELECT COUNT(*) FROM civitai_meta").fetchone()[0]
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("stats", exc)
        return out


# One process-wide store (central). Best-effort throughout; safe to import from
# any surface (stdlib-only, no cycles).
model_metadata_store = ModelMetadataStore()


#: The code paths that may read THROUGH to the Hub: finding a model
#: (search / spec / screen) and fetching it (the download's size estimate).
#: Everything about an INSTALLED model reads hugpy.json or this store only.
FETCH_PURPOSES = ("discovery", "download")


def fetch_repo_info(hub_id: str, files_metadata: bool = True,
                    force: bool = False, api: Any = None,
                    purpose: "str | None" = None) -> "dict | None":
    """The single read-through funnel for repo metadata.

    ``purpose`` GATES THE NETWORK (operator ruling 2026-09-23: "the only
    Hugging Face API calls should be ones that obviously are needed"). Only a
    caller that declares ``purpose`` in :data:`FETCH_PURPOSES` — discovery
    (search/spec/screen) or download — may go live on a miss. Any other caller
    gets the cached row or ``None`` with a logged reason; it never fetches.

    Cache hit -> the cached dict, zero network. Miss (or ``force=True``) ->
    ONE live ``model_info`` call, serialize, write-back, return. Per operator
    policy the cached row never expires — ``force`` / ``forget()`` are the
    only re-fetch paths.

    ``api`` lets a caller pass its own authenticated HfApi (search_routes'
    module-level client); default builds from the process token env.

    Error contract: a broken CACHE never surfaces here — any store failure
    degrades to the live path. The LIVE call's exception, however, PROPAGATES
    (each caller already has its own fallback contract: /hf/spec 502s,
    model_size logs-and-returns-None, resolve returns {}). Swallowing it here
    would silently hide gated/404/network truths the callers report today."""
    if not hub_id:
        return None
    if purpose not in FETCH_PURPOSES:
        cached = model_metadata_store.get_repo_info(hub_id)
        if cached is None:
            logger.info("repo info for %r not cached and caller purpose=%r may not "
                        "fetch (only %s may reach the Hub) — returning None",
                        hub_id, purpose, "/".join(FETCH_PURPOSES))
        return cached
    if not force:
        cached = model_metadata_store.get_repo_info(hub_id)
        if cached is not None:
            return cached
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or False)
    info = api.model_info(hub_id, files_metadata=files_metadata)
    payload = serialize_model_info(info)
    model_metadata_store.put_repo_info(
        hub_id, payload, revision=payload.get("sha"))
    return payload


def checkpoint_stem(filename: str) -> str:
    """Canonical civitai_meta key for a local checkpoint file — EXACTLY the
    slug the comfy sweep synthesizes (comfy-<stem>), so the stamp written at
    download time and the lookup made at sweep time agree by construction."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9]+", "-",
                   os.path.basename(filename).rsplit(".", 1)[0]
                   ).strip("-").lower()


def serialize_civitai_model(data: dict,
                            version_id: "int | str | None" = None) -> dict:
    """Defensively flatten a Civitai /models/<id> (or /model-versions/<vid>)
    response to a plain JSON-native dict. Handles BOTH shapes: a model payload
    (has ``modelVersions``) and a bare version payload (has ``model`` +
    ``files``). Every access is guarded — a missing field degrades to None,
    never poisons the write-back."""
    if not isinstance(data, dict):
        return {}
    versions = data.get("modelVersions")
    if isinstance(versions, list):           # /models/<id> shape
        model = data
        ver = None
        if version_id is not None:
            ver = next((v for v in versions
                        if str(v.get("id")) == str(version_id)), None)
        if ver is None:
            ver = versions[0] if versions else {}
    else:                                    # /model-versions/<vid> shape
        ver = data
        model = data.get("model") or {}
    ver = ver if isinstance(ver, dict) else {}
    model = model if isinstance(model, dict) else {}
    files = []
    for f in (ver.get("files") or []):
        if not isinstance(f, dict):
            continue
        try:
            files.append({"name": f.get("name"),
                          "size_bytes": int((f.get("sizeKB") or 0) * 1024)
                          or None,
                          "type": f.get("type")})
        except Exception:  # noqa: BLE001
            continue
    model_id = model.get("id") or ver.get("modelId")
    out = {
        "civitai_id": model_id,
        "version_id": ver.get("id"),
        "name": model.get("name"),
        "version_name": ver.get("name"),
        "base_model": ver.get("baseModel"),
        "type": model.get("type"),
        "nsfw": model.get("nsfw"),
        "tags": list(model.get("tags") or []),
        "trained_words": list(ver.get("trainedWords") or []),
        "page_url": (f"https://civitai.com/models/{model_id}"
                     if model_id else None),
        "files": files,
    }
    try:
        json.dumps(out)
        return out
    except (TypeError, ValueError):
        return json.loads(json.dumps(out, default=str))


def fetch_civitai_meta(stem: str, civitai_id: "int | str | None" = None,
                       version_id: "int | str | None" = None,
                       force: bool = False,
                       fetcher: Any = None,
                       timeout: float = 15.0) -> "dict | None":
    """Fetch-once Civitai metadata for one local checkpoint (keyed by stem).

    Cache hit -> cached dict, zero network. Miss WITH an id -> ONE live fetch
    (``/models/<civitai_id>`` preferred, else ``/model-versions/<version_id>``),
    serialize, write-back, return. Miss WITHOUT any id -> None, NO network —
    NEVER search Civitai by filename guess (operator policy: metadata rides
    ids we actually hold, never speculative egress).

    Error contract (differs from ``fetch_repo_info``): live-call errors LOG
    and return None — enrichment is best-effort decoration, no caller has a
    failure contract on it. Failures are NOT cached, so a later call retries.

    ``fetcher`` is a test seam: fetcher(url, headers, timeout) -> dict."""
    if not stem:
        return None
    if not force:
        cached = model_metadata_store.get_civitai_meta(stem)
        if cached is not None:
            # A download-time provenance stamp (marked provenance_only) is not
            # an API response — upgrade it with the one live fetch IF we hold
            # ids (from the stamp itself or the caller); otherwise it IS the
            # best we're allowed to have (no speculative egress).
            if not cached.get("provenance_only"):
                return cached
            civitai_id = civitai_id or cached.get("civitai_id")
            version_id = version_id or cached.get("version_id")
            if civitai_id is None and version_id is None:
                return cached
    if civitai_id is None and version_id is None:
        return None
    if civitai_id is not None:
        url = f"https://civitai.com/api/v1/models/{civitai_id}"
    else:
        url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {}
    token = (os.environ.get("CIVITAI_API_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        if fetcher is not None:
            data = fetcher(url, headers=headers, timeout=timeout)
        else:
            import httpx
            r = httpx.get(url, headers=headers, timeout=timeout,
                          follow_redirects=True)
            r.raise_for_status()
            data = r.json()
        payload = serialize_civitai_model(data, version_id=version_id)
    except Exception as exc:  # noqa: BLE001 — best-effort, never propagates
        logger.warning("civitai metadata fetch for %s (%s) failed: %s",
                       stem, url, exc)
        return None
    if not payload:
        return None
    model_metadata_store.put_civitai_meta(stem, payload)
    return payload
