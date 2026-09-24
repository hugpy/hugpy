"""Dedicated ML "amenity" endpoints (`/ml/*`).

These are named, *preordained* routes for the FIXED set of media-intelligence
tasks (ASR, summarize, keywords, embeddings, similarity, vision, text-to-image).
They're a static part of hugpy's default amenities — necessarily distinct from
the open-ended LLM chat surface — so they get stable named endpoints AND a
reserved worker pool, instead of riding the generic `/prompt {task}` verb (which
stays for general/back-compat use).

DESIGN — deliberately THIN. Each route forces its `task` and a default ML worker
pool, then hands off to `execute_prompt`. No heavy/contrived ML code is imported
here: the dispatch lazy-imports each task's implementation only when it actually
runs — ideally on the reserved `ml`-pool worker (and on a CLEAN seam, NOT the
contrived `utils` helpers, which drag unhospitable imports). If no `ml`-pool
worker is registered, the pool reservation falls back to LOCAL (central
in-process) — never the general LLM pool.

Pool: `HUGPY_ML_POOL` (default ``ml``; empty disables the default). An API-key
bound pool or an explicit per-request `pool` still wins over the default.
"""
from __future__ import annotations

import os

from abstract_flask import get_bp
from flask import jsonify, request
from hugpy_server.app.functions.imports.utils.api_keys import (
    verify_api_key,
    media_key_required,
    set_media_key_required,
)

ml_bp, logger = get_bp("ml_bp", __name__)


from hugpy_server.app.auth_common import bearer_token as _bearer_token


# Endpoints exempt from the media gate: the gate-management routes themselves
# (you must be able to read/flip the gate from the auth-walled console even when
# it's on). Matched by function-name SUFFIX so it holds regardless of the
# blueprint-name prefix the /api alias / get_bp produce (e.g. "ml_bp_bp.").
_GATE_EXEMPT_SUFFIXES = (".ml_gate_get", ".ml_gate_set")


@ml_bp.before_request
def _enforce_media_gate():
    """When the media gate is on, the /ml/* inference amenities require a valid
    Bearer key. GET discovery (GET /ml, GET /ml/gate) stays open, and the
    gate-management routes are always exempt. Separate from the /v1 key gate and
    the console login."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    endpoint = request.endpoint or ""
    if any(endpoint.endswith(sfx) for sfx in _GATE_EXEMPT_SUFFIXES):
        return
    # required_scope="ml" (2026-07-23): the key must carry "ml" or "full".
    # Legacy keys read as ["full"] and still pass; a "v1"-only key does not.
    if media_key_required() and not verify_api_key(_bearer_token(),
                                                   required_scope="ml"):
        return jsonify({
            "ok": False,
            "error": ("media-intelligence requires an API key. Pass "
                      "'Authorization: Bearer <key>' (create keys in the console "
                      "under API access)."),
        }), 401

# The fixed media-intelligence amenity set: route name -> dispatch task key.
ML_TASKS = {
    "transcribe": "automatic-speech-recognition",
    "summarize":  "text-summarization",
    "keywords":   "keyword-extraction",
    "embed":      "feature-extraction",
    "similarity": "sentence-similarity",
    "vision":     "image-text-to-text",
    "imagine":    "text-to-image",
    "extract":    "document-extraction",
    "fetch":      "url-extraction",
    # Vision-analysis family (generic pipeline runner).
    "depth":      "depth-estimation",
    "detect":     "object-detection",
    "classify":   "image-classification",
    "segment":    "image-segmentation",
}

# Deterministic ingest amenities that are NOT model inference: they read text from
# an uploaded file or a URL (no model to load, nothing to delegate). _run_ml
# branches these to a thin LOCAL handler instead of execute_prompt. Keeping them in
# ML_TASKS means the route auto-registers and GET /ml lists them like every other
# amenity. Maps the task → the handler name resolved in _run_ml.
_DETERMINISTIC_ML = {"document-extraction", "url-extraction"}


# Per-amenity dependency probe -> the extra that provides it. Reported by GET /ml
# so the UI shows enabled tools vs a clean "enable with abstract_hugpy_dev[X]"
# affordance — WITHOUT importing the heavy dep (find_spec only; phone-clean).
#
# DERIVED (not hand-copied) from the canonical task->dependency map in
# managers.task_deps, joined through ML_TASKS so this route-keyed view and the
# WORKER's task-keyed ``task_capabilities`` advertisement can never drift on which
# module powers which task. (2026-07-11: the drift is exactly what let requests
# reach workers that couldn't run the task and fail at request time.)
from hugpy_engine.task_deps import TASK_DEPS
ML_DEP = {name: TASK_DEPS[task] for name, task in ML_TASKS.items()
          if task in TASK_DEPS}


def _have(mod: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _ml_pool() -> str:
    """The reserved worker pool these endpoints route to (env-overridable).
    Empty string disables the default (normal pool resolution then applies)."""
    # Default UN-pooled (opt-in reservation): with a pool default, any install
    # lacking a pool-tagged worker had its ENTIRE media arm silently falling
    # back to central — models assigned to general workers were never used.
    # Set HUGPY_ML_POOL=ml (+ tag a worker) to opt into a dedicated pool.
    return (os.environ.get("HUGPY_ML_POOL", "") or "").strip()


def _general_route_tasks() -> set:
    """Tasks that follow the GENERAL worker route instead of the reserved ML pool.

    Such a task resolves to its default model (e.g. the default VL model for
    vision) and routes to whatever live *un-pooled* worker already serves that
    model — the "established worker route" — falling back to LOCAL when none
    exists. Image analysis lives here so a real vision worker, when present, is
    actually used instead of the work being trapped on an empty reserved pool.
    Operator-overridable via ``HUGPY_ML_GENERAL_ROUTE_TASKS`` (comma list)."""
    raw = os.environ.get("HUGPY_ML_GENERAL_ROUTE_TASKS", "image-text-to-text")
    return {t.strip() for t in raw.split(",") if t.strip()}


from hugpy_platform.async_runtime import await_sync as _await_sync


from hugpy_platform.results import result_payload as _result_payload


def _run_extract(body: dict):
    """`/ml/extract` — read text from an uploaded document (PDF/DOCX/text). A thin
    LOCAL handler (no model), path-jailed to the storage root. Returns the same
    {ok, text, ...} shape as the model amenities so the chat narrates it the same."""
    path = (body.get("file") or body.get("file_path")
            or body.get("image_path") or body.get("path"))
    if not path:
        return jsonify({"ok": False, "error": "no file path provided"}), 400
    from hugpy_media.extract import extract_document
    try:
        result = extract_document(path)
    except PermissionError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 403
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404
    except Exception as exc:
        logger.exception("ml document-extraction failed")
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify(result), (200 if result.get("ok") else 400)


def _run_fetch(body: dict):
    """`/ml/fetch` — structured assessment of a public URL. Thin LOCAL handler
    (no model), SSRF-guarded at the front door (http/https only, no private/
    loopback addresses). Returns the {ok, text, ...} shape plus an assessment
    (title/description/links/render) when abstract_webtools is available, and
    degrades to a plain readable-text fetch otherwise."""
    url = body.get("url") or body.get("link") or body.get("text")
    from hugpy_media.extract import assess_url
    try:
        result = assess_url(url)
    except PermissionError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 403
    except Exception as exc:
        logger.exception("ml url-extraction failed")
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502
    return jsonify(result), (200 if result.get("ok") else 400)


def _save_multipart_upload():
    """If this request is a multipart POST carrying a file part, save it under
    UPLOADS_HOME (exactly like POST /uploads does) and return the server path.
    Returns None when there's no usable file part.

    The console UI pre-uploads to /uploads and then POSTs JSON {file:<path>};
    this fallback lets /ml/* also accept a direct multipart file in one shot."""
    if not request.files:
        return None
    f = request.files.get("file") or next(iter(request.files.values()), None)
    if not f or not f.filename:
        return None
    import uuid
    from werkzeug.utils import secure_filename
    try:
        from hugpy_platform.constants import UPLOADS_HOME
    except Exception:
        UPLOADS_HOME = globals().get("UPLOADS_HOME") or os.path.join(
            os.environ.get("DEFAULT_ROOT", "/tmp"), "uploads")
    os.makedirs(UPLOADS_HOME, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}"
    dest = os.path.join(UPLOADS_HOME, name)
    f.save(dest)
    return dest


def normalize_ml_kwargs(task: str, body: dict) -> dict:
    """Body -> execute_prompt kwargs, exactly as /ml/<amenity> normalizes them.

    Extracted from _run_ml (k91) so POST /oracle/route dispatches through the
    SAME normalization (model fold, vision b64 fold, pool resolution) instead of
    growing a rival one. Behavior-preserving: _run_ml calls this verbatim."""
    # Underscore-prefixed keys are internal routing controls; never client-set.
    kwargs = {k: v for k, v in body.items()
              if not k.startswith("_") and v is not None}
    kwargs["task"] = task  # the route fixes the task (any client `task` ignored)
    # Model normalization: the resolver reads ``model_key`` only, but this
    # route's documented surface (and the identity front-select relay) sends
    # ``model``. Un-mapped it was SILENTLY IGNORED — the request ran on the
    # task-default model while claiming nothing (e.g. an explicit VL-7B
    # front-select judgment quietly answered by the 3B). Fold it in here;
    # callers already sending model_key win untouched.
    if "model_key" not in kwargs:
        m = kwargs.pop("model", None)
        if m:
            kwargs["model_key"] = m
    # Vision payload normalization: the prompt pipeline (and the GGUF chat
    # runner's _attach_image) reads ``images`` (list of base64 / data URIs) or
    # ``file`` (server path) — but this route's documented surface also takes
    # ``image_b64``. Un-mapped it was silently DROPPED and the VL model
    # answered text-blind ("I do not have access to visual information"), so
    # fold it into ``images`` here. Callers already sending images/file are
    # untouched.
    if task == "image-text-to-text" and "images" not in kwargs:
        b64 = kwargs.pop("image_b64", None)
        if b64:
            kwargs["images"] = [b64]

    # Effective pool: API-key-bound pool / explicit override always wins. With
    # neither, most ML tasks default to the reserved ML pool (so they never
    # compete on the general LLM pool); GENERAL-route tasks (e.g. vision) instead
    # default to the general route (empty), letting image analysis follow the
    # established worker route — any live worker serving the default VL model,
    # else LOCAL — rather than being pinned to a possibly-empty reserved pool.
    default_pool = "" if task in _general_route_tasks() else _ml_pool()
    try:
        from hugpy_server.app.functions.chat.streaming import _resolve_request_pool
        eff = _resolve_request_pool(kwargs.get("pool")) or default_pool
    except Exception:
        eff = (kwargs.get("pool") or default_pool)
    if eff:
        kwargs["pool"] = eff
    else:
        kwargs.pop("pool", None)
    return kwargs


def _run_ml(task: str):
    body = request.get_json(silent=True) or {}
    # Multipart fallback: accept a direct multipart/form-data POST carrying a file
    # part (in addition to the JSON {file:<server path>} the console sends after
    # pre-uploading). Save it like /uploads and inject its server path so the
    # downstream pipeline is identical to the JSON path.
    if not body and request.files:
        saved = _save_multipart_upload()
        if saved is not None:
            body = {**request.form.to_dict(), "file": saved}
    # Deterministic ingest amenities (read a file / URL) bypass the model resolver.
    if task == "document-extraction":
        return _run_extract(body)
    if task == "url-extraction":
        return _run_fetch(body)
    kwargs = normalize_ml_kwargs(task, body)

    from hugpy_engine.dispatch.dispatch import execute_prompt
    try:
        result = _await_sync(execute_prompt(**kwargs))
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        # builder/validation errors — caller's to fix.
        return jsonify({"ok": False, "error": str(exc).strip("'\"")}), 400
    except Exception as exc:
        # Same two non-failures /prompt handles: central declining to start
        # another concurrent model load (503 + Retry-After) and a caller who
        # hung up (499). Neither is a fault, so neither gets a 500 traceback.
        from hugpy_server.app.routes.prompt_routes import _capacity_refusal, _client_gone
        cap = _capacity_refusal(exc)
        if cap is not None:
            return cap
        if _client_gone(exc):
            logger.info("ml task %s abandoned: client disconnected", task)
            return jsonify({"ok": False, "error": "client disconnected"}), 499
        logger.exception("ml task %s failed", task)
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    payload = _result_payload(result)
    payload.setdefault("ok", not payload.get("error"))
    return jsonify(payload)


def _make_view(task: str):
    def _view():
        return _run_ml(task)
    return _view


# Register one POST route per named amenity (closure-factory avoids late-binding).
for _name, _task in ML_TASKS.items():
    ml_bp.add_url_rule(
        f"/ml/{_name}",
        endpoint=f"ml_{_name}",
        view_func=_make_view(_task),
        methods=["POST"],
    )


@ml_bp.route("/ml", methods=["GET"])
def ml_index():
    """Discovery + capability: each amenity's task, whether its dependency is
    installed, and the extra that enables it. Uses find_spec only — it never
    imports the heavy dep, so this stays cheap and phone-clean."""
    endpoints = {}
    for name, task in ML_TASKS.items():
        mod, extra = ML_DEP.get(name, (None, None))
        endpoints[f"/ml/{name}"] = {
            "task": task,
            "ready": _have(mod) if mod else True,
            "extra": extra,
        }
    return jsonify({"endpoints": endpoints, "pool": _ml_pool() or None})


@ml_bp.route("/ml/gate", methods=["GET"])
def ml_gate_get():
    """Read the media-intelligence access gate (key-required flag)."""
    return jsonify({"require_key": media_key_required()})


@ml_bp.route("/ml/gate", methods=["PUT"])
def ml_gate_set():
    """Flip the media-intelligence access gate. Body: {"require": bool}. When on,
    /ml/* inference requires a Bearer key — separate from the /v1 gate and the
    console login."""
    body = request.get_json(silent=True) or {}
    val = body.get("require", body.get("require_key", body.get("enabled")))
    return jsonify({"require_key": set_media_key_required(bool(val))})
