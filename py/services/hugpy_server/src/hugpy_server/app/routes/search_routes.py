### routes/search_routes.py

import json
import os
import shutil

from abstract_flask import get_bp
from flask import abort, jsonify, request
from huggingface_hub import HfApi
from hugpy_engine.wire.model_schemas import FileSpec, ModelSearchResult, ModelSpec
from hugpy_platform.constants import MODELS_DIR
from hugpy_server.app.functions.imports.options.install import resolve_options
from hugpy_server.app.functions.imports.options.search import model_size
search_bp, logger = get_bp("search_bp", __name__)

# Single declared token source for ALL Hugging Face calls in this module.
# The token comes from the console-managed store (get_hf_token: a saved token
# wins, env HF_TOKEN is the fallback). `or False` FORCES anonymous when there is
# no token — HF public search/metadata needs no auth, and a stale
# ~/.cache/huggingface/token cannot then poison the call. A saved token lifts
# the anonymous rate limits that make search/metadata flaky. Everything below
# goes through this `api` object, and _rebuild_hf_api() re-points it when the
# operator saves/clears the token at runtime (no process restart).
from hugpy_storage.hf_token import get_hf_token


def _rebuild_hf_api():
    global api
    api = HfApi(token=get_hf_token() or False)
    return api


api = HfApi(token=get_hf_token() or False)


# ── helpers ───────────────────────────────────────────────────────────────
def _free_bytes() -> int | None:
    """Headroom on the filesystem where downloads actually land (MODELS_DIR)."""
    try:
        probe = MODELS_DIR if os.path.exists(MODELS_DIR) else "/"
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _context_length(hub_id: str, files) -> int | None:
    if not any(f.path == "config.json" for f in files):
        return None
    try:
        cfg_path = api.hf_hub_download(hub_id, "config.json")   # tiny, cached
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg.get("max_position_embeddings") or cfg.get("n_positions")
    except Exception:
        return None


def _license_of(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    try:
        return card.to_dict().get("license")
    except Exception:
        return getattr(card, "license", None)


# ── uploader trust (curated, extensible — NOT an official HF signal) ─────────
# The tables and the tier function now live in imports/src/constants/trust.py,
# which imports no Flask, so the review CLI/timer and any worker resolve trust
# identically to this UI. Names re-bound here so existing references stand.
from hugpy_platform.trust import (
    TRUST_TIER1 as _TRUST_TIER1,
    TRUST_TIER2 as _TRUST_TIER2,
    trust_tier as _trust_tier,
    trust_label as _trust_label,
)


def _relevance_score(q: str, hub_id: str, downloads: int, author=None) -> float:
    """How well a repo matches the query, name-first, then WHO published it.

    The model NAME (last path segment) carries the intent, so name matches
    dominate. Among similarly-named repos — the "odd iterations" problem, where a
    fork/requant shares the canonical name — UPLOADER TRUST breaks the tie (a
    first-party org outranks a community repackager outranks an unknown), which
    matters more than raw popularity. Downloads are only a faint last-resort
    tiebreak (log-scaled, tiny weight); likes are ignored entirely. So trust and
    name both push the canonical repo up, and a well-liked fork never wins on
    likes alone. Case-insensitive. All local — no network."""
    import difflib
    import math
    import re as _re
    q = (q or "").lower().strip()
    if not q:
        return 0.0
    hid = (hub_id or "").lower()
    name = hid.rsplit("/", 1)[-1]
    score = difflib.SequenceMatcher(None, q, name).ratio()   # 0..1 base
    if name == q:
        score += 3.0
    elif name.startswith(q):
        score += 2.0
    elif q in name:
        score += 1.0
    elif q in hid:
        score += 0.5
    q_tok = {t for t in _re.split(r"[-_/\s.]+", q) if t}
    n_tok = {t for t in _re.split(r"[-_/\s.]+", name) if t}
    if q_tok and n_tok:
        score += 0.5 * len(q_tok & n_tok) / len(q_tok)
    # Trust: strong enough to reorder repos with equally-close names, but below an
    # exact-name match — typing the exact repo name still lands it. tier1 +1.5,
    # tier2 +0.75. This is the "trust over arbitrary likes" the operator asked for.
    score += (0.75 * _trust_tier(hub_id, author))
    score += 0.02 * math.log10((downloads or 0) + 10)        # faint popularity tiebreak
    return score


# ── search: list available HF repos ────────────────────────────────────────
@search_bp.route("/search", methods=["GET"])
def search_models():
    q = request.args.get("q", "").strip()
    limit = request.args.get("limit", default=20, type=int)
    # `offset` is how the console walks the WHOLE Hub rather than just the first
    # page: with no query and no task the listing is "every model on HF", and the
    # client pages through it by bumping offset. HF's own `limit` param is the
    # page size it serves in one request, so a window of offset+limit under ~1000
    # is still a single upstream call; deeper windows make huggingface_hub follow
    # its cursor. Both are clamped so a hand-crafted URL can't ask for a
    # thousand-page scan (and, with with_size=1, a thousand metadata lookups).
    limit = max(1, min(limit, 200))
    offset = max(0, request.args.get("offset", default=0, type=int))
    offset = min(offset, 100_000)
    author = request.args.get("author")
    task = request.args.get("task")          # -> pipeline_tag
    library = request.args.get("library")    # -> filter
    sort = request.args.get("sort")
    direction = request.args.get("direction", default=-1, type=int)  # client-side only now
    with_size = request.args.get("with_size", default="1") != "0"

    # Default to name-RELEVANCE when the user typed a query — closest-name-first is
    # what "search" should mean, and it eliminates sifting past odd fine-tuned
    # iterations to find the canonical repo. Fall back to recency when just browsing.
    if not sort:
        sort = "relevance" if q else "last_modified"
    relevance = (sort == "relevance")
    if not relevance and sort not in ("last_modified", "downloads", "likes", "created_at"):
        sort = "last_modified"

    # The window the client actually wants is [offset, offset+limit). Relevance
    # re-ranks locally, so it pulls a LARGER candidate pool (HF's own search order)
    # and sorts it by name closeness — but only the trimmed window is enriched
    # (per-repo size lookups), so network cost stays proportional to one page.
    window = offset + limit
    pool_limit = min(max(window * 5, 50), 1000) if (relevance and q) else window
    try:
        models = list(api.list_models(
            search=q or None,
            author=author,
            pipeline_tag=task or None,        # task is its own param
            filter=library or None,           # filter = library/tag
            sort=(None if relevance else sort),
            limit=pool_limit,
            full=False,
        ))
    except Exception as exc:
        abort(502, description=f"Hugging Face request failed: {exc}")

    if relevance and q:
        models.sort(
            key=lambda m: _relevance_score(
                q, m.modelId, getattr(m, "downloads", 0) or 0,
                getattr(m, "author", None)),
            reverse=True)
    models = models[offset:offset + limit]

    # Repo sizes are one metadata lookup each (cached forever after the first —
    # see model_size). Serially that made a 40-row page 40 round-trips deep on a
    # cold cache; fanned out, a page costs about one. Bounded so browsing the Hub
    # can't open an unreasonable number of sockets at once.
    sizes: dict[str, int | None] = {}
    if with_size and models:
        from concurrent.futures import ThreadPoolExecutor
        hub_ids = [m.modelId for m in models]
        with ThreadPoolExecutor(max_workers=min(8, len(hub_ids))) as pool:
            sizes = dict(zip(hub_ids, pool.map(model_size, hub_ids)))

    results = []
    for model in models:
        hub_id = model.modelId
        results.append(
            ModelSearchResult(
                hub_id=hub_id,
                author=getattr(model, "author", None),
                downloads=getattr(model, "downloads", None),
                likes=getattr(model, "likes", None),
                tags=getattr(model, "tags", []) or [],
                pipeline_tag=getattr(model, "pipeline_tag", None),
                library_name=getattr(model, "library_name", None),
                private=getattr(model, "private", None),
                total_bytes=sizes.get(hub_id) if with_size else None,   # #2
                last_modified=str(getattr(model, "last_modified", "")) or None,
                created_at=str(getattr(model, "created_at", "")) or None,
                trust=_trust_label(
                    _trust_tier(hub_id, getattr(model, "author", None))),
            ).model_dump()
        )

    return jsonify(results)



# ── spec: per-repo detail + install options (lazy, on row expand) ───────────
@search_bp.route("/hf/spec", methods=["GET"])
def hf_spec():
    """Per-repo metadata rides the PERMANENT central HF cache (fetch-once,
    no TTL — operator policy, see comms/model_metadata.py): the first spec of a
    repo ever hits HF, every later one is served from SQLite. ``?refresh=1``
    is the explicit operator re-fetch affordance (forces one live call and
    overwrites the cached row)."""
    from hugpy_storage.model_metadata import fetch_repo_info
    hub_id = request.args.get("hub_id")
    if not hub_id:
        abort(400, description="hub_id is required.")
    refresh = request.args.get("refresh", default="0") not in ("0", "", None)

    try:
        payload = fetch_repo_info(hub_id, files_metadata=True,
                                  force=refresh, api=api)
    except Exception as exc:
        abort(502, description=f"Hugging Face request failed: {exc}")
    if payload is None:
        abort(502, description="Hugging Face metadata unavailable.")

    files = [FileSpec(path=s.get("rfilename"), size=s.get("size"))
             for s in (payload.get("siblings") or []) if s.get("rfilename")]
    total = sum(f.size for f in files if f.size) or None
    free = _free_bytes()

    task = payload.get("pipeline_tag") or "text-generation"
    options = resolve_options(hub_id, task, files, free)

    spec = ModelSpec(
        hub_id=hub_id,
        license=payload.get("license"),
        gated=payload.get("gated"),
        last_modified=payload.get("last_modified"),
        total_bytes=total,
        num_params=payload.get("safetensors_params"),
        context_length=_context_length(hub_id, files),
        gguf_quants=[o.id.split(":", 1)[1] for o in options.options
                     if o.id.startswith("gguf:")],
        files=files,
    )

    # free_bytes: the same MODELS_DIR headroom the per-option fits_disk flags
    # were computed against — the client's Auto-fit needs the budget itself,
    # not just the per-file verdicts.
    return jsonify({"spec": spec.model_dump(), "options": options.model_dump(),
                    "free_bytes": free})


# ── HF metadata cache observability (operator) ──────────────────────────────
# The permanent per-repo cache (comms/model_metadata.py): GET shows what's held;
# DELETE <hub_id> is how the operator forces a repo re-fetch — forget the rows,
# the next access re-fetches live. The DELETE is operator-gated (see
# operator_auth ^/hf/cache/).
@search_bp.route("/hf/cache", methods=["GET"])
def hf_cache_stats():
    from hugpy_storage.model_metadata import model_metadata_store
    return jsonify(model_metadata_store.stats())


@search_bp.route("/hf/cache/<path:hub_id>", methods=["DELETE"])
def hf_cache_forget(hub_id):
    from hugpy_storage.model_metadata import model_metadata_store
    removed = model_metadata_store.forget(hub_id)
    return jsonify({"hub_id": hub_id, "rows_removed": removed,
                    "note": "next access re-fetches live"})


# ── Civitai — the checkpoint habitat, wired to the comfy drop-a-file flow ────
# Search is anonymous (their public API); DOWNLOAD streams the single-file
# checkpoint straight into <root>/checkpoints, where the registry sweep
# self-registers it as a comfy-<slug> model on the next read. Some files
# require a Civitai account token: set CIVITAI_API_TOKEN in the env (d-env).

_CIVITAI = "https://civitai.com/api/v1"
_CIVITAI_DL: dict = {}   # filename -> {done_bytes, total_bytes, status, error?}


def _checkpoints_dir() -> str:
    from hugpy_platform.constants import DEFAULT_ROOT
    d = os.path.join(DEFAULT_ROOT, "checkpoints")
    os.makedirs(d, exist_ok=True)
    return d


def _stamp_civitai_provenance(dest: str, url: str, filename: str,
                              provenance: dict) -> None:
    """Best-effort provenance stamp after a checkpoint lands: a
    ``<dest>.civitai.json`` sidecar (atomic tmp+replace) plus the same dict
    into the central model-metadata store keyed by the sweep's stem. Failure
    must NEVER fail the download — the checkpoint is already good; provenance
    is decoration (the pre-stamp world is exactly today's world)."""
    import time as _time
    stamp = {
        "civitai_id": provenance.get("civitai_id"),
        "version_id": provenance.get("version_id"),
        "name": provenance.get("name"),
        "base_model": provenance.get("base_model"),
        "download_url": url,
        "filename": filename,
        "fetched_at": _time.time(),
        # This row is the download-time stamp, NOT a Civitai API response —
        # fetch_civitai_meta upgrades it (one live call) when asked to enrich.
        "provenance_only": True,
    }
    sidecar = dest + ".civitai.json"
    try:
        tmp = sidecar + ".part"
        with open(tmp, "w") as fh:
            json.dump(stamp, fh, indent=2)
        os.replace(tmp, sidecar)
    except Exception as exc:  # noqa: BLE001
        logger.warning("civitai: provenance sidecar for %s failed: %s",
                       filename, exc)
    try:
        from hugpy_storage.model_metadata import checkpoint_stem, model_metadata_store
        model_metadata_store.put_civitai_meta(checkpoint_stem(filename), stamp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("civitai: central provenance row for %s failed: %s",
                       filename, exc)


@search_bp.route("/civitai/search", methods=["GET"])
def civitai_search():
    """Proxy Civitai model search into rows the console table can render."""
    import httpx
    params = {
        "types": "Checkpoint",
        "limit": max(1, min(int(request.args.get("limit", 20)), 50)),
        "sort": request.args.get("sort") or "Highest Rated",
        "nsfw": "false",
    }
    q = (request.args.get("query") or "").strip()
    if q:
        params["query"] = q
    base = (request.args.get("base") or "").strip()
    if base:
        params["baseModels"] = base
    try:
        r = httpx.get(_CIVITAI + "/models", params=params, timeout=20.0)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"civitai search failed: {exc}"}), 502

    rows = []
    for it in payload.get("items", []):
        v = (it.get("modelVersions") or [{}])[0]
        f = next((x for x in (v.get("files") or [])
                  if str(x.get("name", "")).endswith((".safetensors", ".ckpt"))
                  and (x.get("type") == "Model" or x.get("type") is None)), None)
        if not f or not f.get("downloadUrl"):
            continue
        stats = it.get("stats") or {}
        rows.append({
            "civitai_id": it.get("id"),
            "version_id": v.get("id"),
            "name": it.get("name"),
            "base_model": v.get("baseModel"),
            "version": v.get("name"),
            "filename": f.get("name"),
            "total_bytes": int((f.get("sizeKB") or 0) * 1024),
            "downloads": stats.get("downloadCount"),
            "likes": stats.get("thumbsUpCount"),
            "published": (v.get("publishedAt") or it.get("createdAt") or "")[:10],
            "download_url": f.get("downloadUrl"),
            "page_url": f"https://civitai.com/models/{it.get('id')}",
            # comfy-compat hint: our vanilla template speaks SD1.x/2.x/SDXL.
            "comfy_ready": str(v.get("baseModel") or "").startswith(("SD 1", "SD 2", "SDXL")),
        })
    return jsonify(rows)


@search_bp.route("/civitai/download", methods=["POST"])
def civitai_download():
    """Stream ONE checkpoint into <root>/checkpoints (background). The comfy
    sweep registers it automatically once the file lands — the whole install
    is this download. Operator-gated (writes to central storage).

    Optional provenance fields (the UI has them from /civitai/search rows;
    hand-fed URLs may omit any/all): civitai_id, version_id, name, base_model.
    When present they're stamped into a ``<dest>.civitai.json`` sidecar next
    to the checkpoint AND into the central model-metadata store, so the comfy
    sweep can decorate its synthesized row and fetch-once enrichment has an id
    to ride (never a filename guess)."""
    import threading
    body = request.get_json(silent=True) or {}
    url = str(body.get("download_url") or "").strip()
    filename = os.path.basename(str(body.get("filename") or "").strip())
    provenance = {k: body.get(k) for k in
                  ("civitai_id", "version_id", "name", "base_model")}
    if not url.startswith("https://civitai.com/"):
        return jsonify({"error": "download_url must be a civitai.com URL"}), 400
    if not filename.endswith((".safetensors", ".ckpt")):
        return jsonify({"error": "filename must end in .safetensors/.ckpt"}), 400
    dest = os.path.join(_checkpoints_dir(), filename)
    if os.path.exists(dest):
        return jsonify({"ok": True, "already": True, "filename": filename})
    if _CIVITAI_DL.get(filename, {}).get("status") == "downloading":
        return jsonify({"ok": True, "already": False, "filename": filename,
                        "note": "download already in progress"})

    free = _free_bytes()
    want = int(body.get("total_bytes") or 0)
    if free is not None and want and free < want * 1.1:
        return jsonify({"error": f"not enough space: needs ~{want/1e9:.1f}GB, "
                        f"{free/1e9:.1f}GB free on central"}), 409

    def _pull():
        import httpx
        tmp = dest + ".part"
        headers = {}
        token = os.getenv("CIVITAI_API_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        st = _CIVITAI_DL[filename] = {"done_bytes": 0, "total_bytes": want,
                                      "status": "downloading"}
        try:
            with httpx.stream("GET", url, headers=headers, timeout=60.0,
                              follow_redirects=True) as r:
                if r.status_code in (401, 403):
                    raise PermissionError(
                        "civitai requires an account token for this file — set "
                        "CIVITAI_API_TOKEN in d-env/env and restart")
                r.raise_for_status()
                st["total_bytes"] = int(r.headers.get("content-length") or want or 0)
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
                        st["done_bytes"] += len(chunk)
            os.replace(tmp, dest)
            st["status"] = "done"
            _stamp_civitai_provenance(dest, url, filename, provenance)
            logger.info("civitai: %s landed in /checkpoints — the sweep "
                        "registers it on the next registry read", filename)
        except Exception as exc:  # noqa: BLE001
            st["status"] = "failed"
            st["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("civitai download of %s failed: %s", filename, exc)
            try:
                os.remove(tmp)
            except OSError:
                pass

    threading.Thread(target=_pull, daemon=True).start()
    return jsonify({"ok": True, "started": True, "filename": filename,
                    "note": "lands in /checkpoints; registers automatically"})


@search_bp.route("/civitai/downloads", methods=["GET"])
def civitai_downloads():
    """Live progress for in-flight civitai pulls (the UI polls this)."""
    return jsonify(_CIVITAI_DL)


# ── Hugging Face credentials (console-managed) ──────────────────────────────
# Save an HF token so central's HF calls (this module's search/metadata, plus
# downloads) go out authenticated instead of anonymously rate-limited. The token
# is stored 0600 outside any git tree and is never returned (only its last4).
# Operator-gated (see operator_auth ^/llm/hf/auth$).
@search_bp.route("/llm/hf/auth", methods=["GET"])
def hf_auth_get():
    from hugpy_storage.hf_token import hf_auth_status
    return jsonify(hf_auth_status(validate=True))


@search_bp.route("/llm/hf/auth", methods=["POST"])
def hf_auth_set():
    from hugpy_storage.hf_token import hf_auth_status, store_hf_token, validate_hf_token
    body = request.get_json(silent=True) or {}
    token = str(body.get("token") or "").strip()
    if not token:
        return jsonify({"error": "token is required"}), 400
    # Validate BEFORE storing so we never persist a dud (and can echo HF's own
    # message). A network failure here is fatal for a WRITE — we won't store an
    # unverifiable token.
    status, _username, error = validate_hf_token(token)
    if status == "network":
        return jsonify({"error": error}), 502
    if status != "ok":
        return jsonify({"error": error or "invalid Hugging Face token"}), 400
    store_hf_token(token)
    _rebuild_hf_api()
    return jsonify(hf_auth_status(validate=True))


@search_bp.route("/llm/hf/auth", methods=["DELETE"])
def hf_auth_delete():
    from hugpy_storage.hf_token import delete_hf_token, hf_auth_status
    removed = delete_hf_token()
    _rebuild_hf_api()
    out = hf_auth_status(validate=True)
    out["removed"] = removed
    return jsonify(out)


# ── the FINDER (operator ask 2026-07-29): the console's 🔍 for the API ──────
# abstract-search (the operator's own package) worked into hugpy: grep/files
# search over the VM's trees, returning ONLY file paths + the specific lines
# that matched — the keeper/agent shape ("find my target data fast, cheap").
# Read-gated like the eviction telemetry: operator session OR a valid API key.
# Roots are a WHITELIST — a search surface over an HTTP port must not be a
# filesystem browser; anything outside these answers 400 naming the allowed.
_FINDER_ROOTS = {
    "dev":     "/srv/share/projects/hugpy/dev",
    "station": "/srv/share/projects/hugpy",
    "comms":   "/mnt/llm_storage/comms",
    "spool":   os.path.expanduser("~/.hugpy-lean/spool"),
}


def _finder_authorized() -> bool:
    try:
        from hugpy_server.app.operator_auth import operator_authenticated
        if operator_authenticated():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from hugpy_server.app.routes.worker_routes import _bearer_token
        from hugpy_server.app.functions.imports.utils.api_keys import verify_api_key
        tok = _bearer_token()
        # Scope parity with the other key-gated surfaces (ml -> required_scope="ml",
        # v1/messages -> "v1"): the finder reads source / comms-db / spool trees, so
        # the fallback key must carry "full" (legacy keys read as ["full"], so they
        # still pass). A narrowly-scoped v1/ml key can no longer read arbitrary tree
        # content. Operator sessions/tokens still pass via the branch above.
        return bool(tok and verify_api_key(tok, required_scope="full"))
    except Exception:  # noqa: BLE001
        return False


@search_bp.route("/finder/search", methods=["GET", "POST"])
def finder_search():
    """GET/POST /finder/search?q=<phrase>[&q=...]&root=dev&mode=grep|files&limit=N

    grep  -> [{file_path, lines: [{line, content}]}]  (the console's shape)
    files -> bare file paths whose CONTENT matches (no line detail, faster)
    """
    if not _finder_authorized():
        return jsonify({"error": "not authorized"}), 401
    body = request.get_json(silent=True) or {}
    qs = request.args.getlist("q") or body.get("q") or body.get("strings") or []
    if isinstance(qs, str):
        qs = [qs]
    qs = [str(s) for s in qs if str(s).strip()]
    if not qs:
        return jsonify({"error": "q required (one or more phrases)"}), 400
    root_key = (request.args.get("root") or body.get("root") or "dev").strip()
    if root_key not in _FINDER_ROOTS:
        return jsonify({"error": f"unknown root {root_key!r}",
                        "allowed_roots": sorted(_FINDER_ROOTS)}), 400
    mode = (request.args.get("mode") or body.get("mode") or "grep").strip()
    try:
        limit = int(request.args.get("limit") or body.get("limit") or 200)
    except (TypeError, ValueError):
        limit = 200
    limit = max(1, min(limit, 1000))
    try:
        from abstract_search.find_content import findContent
    except ImportError:
        return jsonify({"error": "abstract-search not installed on this central"}), 501
    # Walk per TOP-LEVEL subtree, not the root in one call: abstract-search
    # RAISES on a dangling symlink ("Not a valid path"), and one poisoned
    # subtree must cost its own results, never the whole search. Skipped
    # subtrees are NAMED in the reply — silence would read as "no matches".
    root = _FINDER_ROOTS[root_key]
    subtrees = [root]
    try:
        entries = sorted(os.listdir(root))
        subs = [os.path.join(root, e) for e in entries
                if os.path.isdir(os.path.join(root, e)) and e not in (".git", "venv",
                    "node_modules", "__pycache__")]
        files_at_root = [os.path.join(root, e) for e in entries
                         if os.path.isfile(os.path.join(root, e))]
        if subs:
            subtrees = subs
    except OSError:
        files_at_root = []
    hits, skipped = [], []

    def _walk(sub, depth):
        # RECURSIVE containment: a subtree that raises (e.g. the deliberate
        # host-side tests/imports/src symlink, dangling in the VM) splits into
        # its children and each retries — the poison costs its own directory,
        # never its siblings. Bounded depth keeps a pathological tree cheap.
        if len(hits) >= limit:
            return
        try:
            hits.extend(findContent(directory=sub, strings=qs,
                                    parse_lines=True,
                                    get_lines=(mode == "grep")) or [])
            return
        except Exception as exc:  # noqa: BLE001
            if depth <= 0:
                skipped.append({"subtree": sub, "error": str(exc)[:200]})
                return
        try:
            kids = sorted(os.listdir(sub))
        except OSError:
            skipped.append({"subtree": sub, "error": "unlistable"})
            return
        for k in kids:
            kp = os.path.join(sub, k)
            if os.path.islink(kp) and not os.path.exists(kp):
                skipped.append({"subtree": kp, "error": "dangling symlink"})
                continue
            if os.path.isdir(kp) and k not in (".git", "venv", "node_modules",
                                               "__pycache__"):
                _walk(kp, depth - 1)

    for sub in subtrees:
        _walk(sub, 3)
        if len(hits) >= limit:
            break
    hits = hits[:limit]
    out = {"root": root, "mode": mode, "strings": qs,
           "count": len(hits), "hits": hits}
    if skipped:
        out["skipped_subtrees"] = skipped
    return jsonify(out)
