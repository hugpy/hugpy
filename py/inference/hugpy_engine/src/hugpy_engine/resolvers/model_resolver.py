"""Model resolution — single source of truth.

Everything in dispatch reads from `Resolution`, which is built exactly
once per request by `resolve()`. No downstream layer is allowed to
re-derive task, framework, builder, or runner_cls from kwargs — if it
needs any of those, it reads them off the Resolution object.

Adding a new (framework, task) pair:
    1. Implement a runner class conforming to the Runner protocol.
    2. Add a row to FRAMEWORK_RUNNERS.
    3. Add a row to MODEL_REQUEST_BUILDERS.
    4. (Optional) Add a row to TASK_DEFAULTS if there's a sensible
       default model for "task only" callers.

Adding a new model:
    Add a row to MODEL_REGISTRY (in models_dict.py). validate_registry()
    will fail at import time if (framework, primary_task) or any
    (framework, task) in cfg.tasks isn't registered.
"""

import threading

import os
from typing import Any, Dict, Optional
from abstract_essentials import derive_media_type, safe_load_from_json
from pydantic import BaseModel
import logging
logger = logging.getLogger(__name__)
from hugpy_engine.categories import MEDIA_DEFAULTS, TASK_DEFAULTS
from hugpy_engine.config.models.models_config import MODELS, MODEL_REGISTRY
from hugpy_engine.config.models.models_default import DEFAULT_CHAT_MODEL
from hugpy_engine.schemas.task_schemas import Resolution
from hugpy_platform.constants import HUGPY_AUTO_DOWNLOAD, PROJECTS_HOME
# Serve path: weights are local-or-central on a worker, never Hugging Face
# (hugpy_storage.provision.ensure_serving_weights; computron 2026-09-23).
from hugpy_storage.provision import ensure_serving_weights
from hugpy_engine.resolvers.categories.builders import MODEL_REQUEST_BUILDERS
from hugpy_engine.resolvers.categories.frameworks import FRAMEWORK_RUNNERS, KNOWN_TASKS_REGISTRY
from hugpy_engine.resolvers.assure_model_key import assure_model_key
from hugpy_engine.resolvers.remote import (
    make_peer_runner,
    make_delegating_runner,
    set_worker_provider,
    get_worker_provider,
)
# ---------------------------------------------------------------------------
# Peer placement (System A) — placement.json delegation.
#
# resolve() calls peer_for() on EVERY request, so these must always be defined,
# even when no placement file exists. A missing/empty placement.json means
# "everything runs locally" — peer_for() returns None and resolve() falls
# through to the local runner. (This block was historically edited only on the
# deployed server and never committed, so the repo's resolve() raised NameError
# on peer_for for every request — which looked like 'no compute allocated'.)
# ---------------------------------------------------------------------------
try:
    PLACEMENT_PATH
except NameError:
    PLACEMENT_PATH = os.path.join(PROJECTS_HOME, "placement.json")


class Peer(BaseModel):
    name: str
    base_url: str              # http://192.168.1.x:PORT — the peer's flask app
    role: str = "compute"
    status: str = "unknown"    # filled by a health ping


# Placement registry: "model_key::task" -> worker name | "local" | absent.
# Empty default = everything runs locally. _load_placement() populates these.
_placement: Dict[str, str] = {}
_peers: Dict[str, Peer] = {}


def _load_placement(path: Optional[str] = None) -> None:
    """Populate _placement/_peers from placement.json. Explicit call, not
    import-time magic — so a missing/empty file means 'all local', never a
    crash."""
    global _placement, _peers
    data = safe_load_from_json(path or PLACEMENT_PATH) or {}
    _placement = data.get("placement", {}) or {}
    _peers = {name: Peer(**cfg) for name, cfg in (data.get("peers", {}) or {}).items()}


def peer_for(model_key: str, task: str) -> Optional[Peer]:
    """Return the Peer that should serve (model_key, task), or None for local.

    Looks up "model_key::task" in the placement map; "local"/absent -> None.
    """
    name = _placement.get(f"{model_key}::{task}")
    if name in (None, "local"):
        return None
    return _peers.get(name)


# Load once at import; safe no-op when placement.json is absent.
try:
    _load_placement()
except Exception as exc:  # never let placement config break resolution
    logger.warning("placement.json load failed (%s); all models run local", exc)


# ---------------------------------------------------------------------------
# Unclassified / adapter rows — refuse by NAME, never by assumption (k61).
#
# A null task used to default to text-generation, so an image LoRA was routed as
# a chat model and refused with "supported: ['text-generation']" — a message that
# named neither the cause nor a fix, and sent the operator hunting for seven
# attempts. Null now means UNCLASSIFIED, and unclassified refuses here, at the one
# resolution authority every request passes through, quoting the remedy.
# ---------------------------------------------------------------------------
def _refuse_if_unclassified(model_key, cfg) -> None:
    from hugpy_engine.model_classifier import (
        ADAPTER_TASK,
        NEEDS_CLASSIFICATION_TASK,
        adapter_refusal,
        needs_classification_refusal,
    )
    tasks = list(getattr(cfg, "tasks", None) or [])
    if not tasks or tasks == [NEEDS_CLASSIFICATION_TASK]:
        raise ValueError(needs_classification_refusal(model_key))
    if tasks == [ADAPTER_TASK]:
        raise ValueError(adapter_refusal(
            model_key, base_model=getattr(cfg, "base_model", None)))


# ---------------------------------------------------------------------------
# resolve_model_key — picks the model. Default-resolution chain only.
# Does NOT pick task; that's resolve()'s job.
# ---------------------------------------------------------------------------
def _adopt_vl_tasks_from_disk(model_key, cfg, task):
    """A stale registry row refusing ``image-text-to-text`` for a GGUF: re-apply
    the vision-GGUF rule (hugpy_marker.vl_gguf_tasks — the rule the marker
    writer and ``hugpy-vl-reclassify`` use) to the model's own dir on THIS box.

    The in-memory registry is derived once (boot / refresh) from the discovery
    report + the local marker copy; a reclassify done on central afterwards
    never reaches it. The mmproj on disk is the fact, so when the dir holds one
    the row is re-derived in place (MODEL_REGISTRY + MODEL_REGISTRY_DICT +
    task registries) and the updated cfg returned. None when the dir says the
    model is not vision (the caller refuses exactly as before)."""
    if task != "image-text-to-text":
        return None
    fw = cfg.framework[0] if isinstance(cfg.framework, (list, tuple)) else cfg.framework
    if fw not in ("gguf", "llama_cpp"):
        return None
    dirs = []
    try:
        from hugpy_engine.config.main import get_model_path
        dirs.append(get_model_path(model_key))
    except Exception:  # noqa: BLE001 — unresolvable local path: try the row's dir
        pass
    dirs.append(getattr(cfg, "dir", None))
    try:
        from hugpy_storage.hugpy_marker import vl_gguf_tasks
    except Exception:  # noqa: BLE001
        return None
    for directory in dirs:
        if not directory or not os.path.isdir(directory):
            continue
        new_t, new_p = vl_gguf_tasks(directory, fw, list(cfg.tasks or []), cfg.primary_task)
        if task not in (new_t or []):
            continue
        import dataclasses
        from hugpy_engine.config.models.models_config import MODEL_REGISTRY_DICT
        fresh = dataclasses.replace(cfg, tasks=list(new_t), primary_task=new_p)
        MODEL_REGISTRY[model_key] = fresh
        row = MODEL_REGISTRY_DICT.get(model_key)
        if isinstance(row, dict):
            row["tasks"], row["primary_task"] = list(new_t), new_p
        try:
            from hugpy_engine.config.models.models_default import refresh_task_registries
            refresh_task_registries()
        except Exception:  # noqa: BLE001
            logger.debug("task registry refresh after VL adopt failed", exc_info=True)
        logger.warning("registry row %s re-derived from %s: tasks %s -> %s "
                       "(mmproj projector on disk)", model_key, directory,
                       list(cfg.tasks or []), list(new_t))
        return fresh
    return None


def resolve_model_key(
    *,
    model_key: Optional[str] = None,
    file: Optional[str] = None,
    media_type: Optional[str] = None,
    task: Optional[str] = None,
) -> str:
    """Pick a model_key via explicit resolution chain.

    Order: explicit model_key > explicit task > explicit media_type
           > file -> media_type > chat default.

    `task`, when given alongside `model_key`, is validated against
    cfg.tasks. When given alone, it picks TASK_DEFAULTS[task].
    """
    if task is not None and task not in KNOWN_TASKS_REGISTRY:
        raise KeyError(
            f"Unknown task={task!r}; known: {sorted(KNOWN_TASKS_REGISTRY)}"
        )
    
    if model_key is not None:
        requested = model_key
        model_key = assure_model_key(model_key)
        if not model_key:
            # Fail with NEAR MATCHES, not the whole registry. Two cases produce
            # a None here: nothing matched, or a fuzzy tie the pipeline could
            # not break (both land as an ambiguous/unknown 400 for the caller).
            from hugpy_engine.resolvers.assure_model_key import fuzzy_model_candidates
            near = fuzzy_model_candidates(requested)
            if near:
                raise ValueError(
                    f"Ambiguous or unresolved model_key={requested!r}; "
                    f"did you mean one of: {sorted(near)}"
                )
            raise ValueError(
                f"Unknown model_key={requested!r}; "
                f"no near matches (see /models for the catalog)"
            )
        _refuse_if_unclassified(model_key, MODEL_REGISTRY[model_key])
        if task is not None and task not in MODEL_REGISTRY[model_key].tasks \
                and _adopt_vl_tasks_from_disk(model_key, MODEL_REGISTRY[model_key], task) is None:
            raise ValueError(
                f"Model {model_key!r} does not support task={task!r}; "
                f"supported: {sorted(MODEL_REGISTRY[model_key].tasks)}"
            )
        logger.debug("resolve_model_key: explicit key=%s task=%s", model_key, task)
        return model_key

    if task is not None:
        inferred = TASK_DEFAULTS.get(task)
        if inferred is None:
            raise KeyError(
                f"No default model for task={task!r}; "
                f"tasks with defaults: {sorted(TASK_DEFAULTS)}"
            )
        if inferred not in MODEL_REGISTRY:
            raise KeyError(
                f"Task default {inferred!r} for {task!r} not in MODEL_REGISTRY:{MODEL_REGISTRY}"
            )
        if task not in MODEL_REGISTRY[inferred].tasks:
            raise ValueError(
                f"Task default {inferred!r} for {task!r} does not list "
                f"{task!r} in cfg.tasks={sorted(MODEL_REGISTRY[inferred].tasks)!r}"
            )
        logger.debug("resolve_model_key: task=%s -> key=%s", task, inferred)
        return inferred

    if media_type is None and file is not None:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"resolve_model_key: file does not exist: {file!r}"
            )
        media_type = derive_media_type(file)
        logger.debug("resolve_model_key: file=%s -> media=%s", file, media_type)

    if media_type is not None:
        inferred = MEDIA_DEFAULTS.get(media_type)
        if inferred is None:
            raise KeyError(
                f"No default model for media_type={media_type!r}; "
                f"known: {sorted(MEDIA_DEFAULTS)}"
            )
        if inferred not in MODEL_REGISTRY:
            raise KeyError(
                f"Media default {inferred!r} for {media_type!r} "
                f"not in MODEL_REGISTRY"
            )
        logger.debug("resolve_model_key: media=%s -> key=%s", media_type, inferred)
        return inferred

    if DEFAULT_CHAT_MODEL not in MODEL_REGISTRY:
        raise KeyError(
            f"DEFAULT_CHAT_MODEL={DEFAULT_CHAT_MODEL!r} not in MODEL_REGISTRY"
        )
    logger.debug("resolve_model_key: fallback to chat default=%s", DEFAULT_CHAT_MODEL)
    return DEFAULT_CHAT_MODEL


# ---------------------------------------------------------------------------
# Staple auto-download — resolve() pulls package-default weights on demand.
#
# MODELS (models_config.py) is the hardcoded default fleet shipped with the
# package: a fresh pip install has those registry rows but no weights on disk.
# When resolve() lands on a staple that may run locally, ensure_serving_weights() pulls
# its weights right here — same download path the runners use — so first use
# works out of the box. Non-staple rows come from disk discovery (already on
# disk) or runtime registration (provisioned via worker sync), so they are
# never surprise-downloaded by resolution.
# ---------------------------------------------------------------------------
_ensured_staples: set = set()
_ensure_locks: Dict[str, threading.Lock] = {}
_ensure_locks_guard = threading.Lock()


def ensure_staple_weights(model_key: str) -> Optional[str]:
    """Make sure a package-staple model's weights are on disk; download if not.

    No-op (returns None) for non-staples and when HUGPY_AUTO_DOWNLOAD is off.
    Per-key locked so concurrent requests for the same model trigger one
    download, while different models download independently. A failed download
    logs a warning instead of raising — the request may still be served by a
    worker, and the local runner raises its own clear error if it actually
    needs the weights.
    """
    if not HUGPY_AUTO_DOWNLOAD or model_key not in MODELS:
        return None
    if model_key in _ensured_staples:
        return None
    with _ensure_locks_guard:
        lock = _ensure_locks.setdefault(model_key, threading.Lock())
    with lock:
        if model_key in _ensured_staples:
            return None
        try:
            # Fast no-op when already on disk. On a WORKER this is
            # local-or-central only and RAISES when central holds no weights —
            # a staple is best-effort here, so that becomes the warning below.
            path = ensure_serving_weights(model_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("resolve: staple %s weights unavailable: %s",
                           model_key, exc)
            path = None
        if path:
            _ensured_staples.add(model_key)
            return path
        logger.warning(
            "resolve: staple %s is not on disk and its download failed; "
            "local runs will fail until the weights are acquired", model_key
        )
        return None


# ---------------------------------------------------------------------------
# resolve — the only function that maps kwargs -> Resolution.
# ---------------------------------------------------------------------------

def resolve(prompt_kwargs: Dict[str, Any]) -> Resolution:
    """Build a Resolution from request kwargs. One call site for all routing.

    `task`, if given by the caller, wins over cfg.primary_task. This is the
    single rule that the old dispatch broke in three different places.
    """
    requested_task = prompt_kwargs.get("task")

    model_key = resolve_model_key(
        model_key=prompt_kwargs.get("model_key"),
        file=prompt_kwargs.get("file"),
        media_type=prompt_kwargs.get("media_type"),
        task=requested_task,
    )

    cfg = MODEL_REGISTRY[model_key]

    # Reload the task-filtered registries (vision/whisper/embed/chat) from the
    # live MODEL_REGISTRY before any backend reads them. Those derived registries
    # are import-time snapshots, so a model registered at runtime (learned from
    # central after boot) is in MODEL_REGISTRY but missing from them — the vision
    # backend would then raise "Unknown vision model key ... Available: []".
    # Doing it here, inside the single resolve() authority, means every backend
    # sees the current model set before the runner is built/loaded.
    from hugpy_engine.config.models.models_default import refresh_task_registries
    refresh_task_registries()

    # An unclassified/adapter row never becomes a task by falling through to
    # cfg.primary_task — it refuses here with the remedy named (k61).
    _refuse_if_unclassified(model_key, cfg)

    task = requested_task or cfg.primary_task
    if isinstance(task, (list, tuple)):       # primary_task must be scalar
        task = task[0]

    framework = cfg.framework
    if isinstance(framework, (list, tuple)):  # framework must be scalar too
        framework = framework[0]

    # TEMP diagnostic — shows which field is malformed in the registry row.
    logger.info("resolve types: model=%s framework=%r task=%r primary=%r tasks=%r",
                model_key, cfg.framework, task, cfg.primary_task, cfg.tasks)

    if task not in cfg.tasks:
        cfg = _adopt_vl_tasks_from_disk(model_key, cfg, task) or cfg
    if task not in cfg.tasks:
        raise ValueError(
            f"Model {model_key!r} does not support task={task!r}; "
            f"supported: {sorted(cfg.tasks)}"
        )

    key = (framework, task)

    builder = MODEL_REQUEST_BUILDERS.get(key)
    if builder is None:
        raise KeyError(
            f"No request builder for {key!r}; model={model_key!r}, "
            f"known: {sorted(MODEL_REQUEST_BUILDERS)}"
        )
    # The local runner must exist for this (framework, task) no matter where the
    # request ultimately runs — validate it here so a bad registry still fails
    # loudly at resolution, and so remote runners can borrow its request/result
    # types.
    local_runner_cls = FRAMEWORK_RUNNERS.get(key)
    if local_runner_cls is None:
        raise KeyError(
            f"No runner for {key!r}; model={model_key!r}, "
            f"known: {sorted(FRAMEWORK_RUNNERS)}"
        )

    # ── Routing: the single decision point for local vs remote ───────────────
    # Precedence:
    #   1. _force_local  — loop guard (a delegated request already arrived here
    #      from a peer/worker) → run in-process, never re-delegate.
    #   2. placement.json peer — operator pinned this (model, task) to another
    #      central node → PeerRunner (one-shot /api/llm/execute). "System A".
    #   3. default — try a live GPU worker for this model, fall back to local.
    #      DelegatingRunner re-selects per request. "System B".
    force_local = prompt_kwargs.pop("_force_local", False)
    peer = None if force_local else peer_for(model_key, task)

    if force_local:
        runner_cls = local_runner_cls
    elif peer:
        runner_cls = make_peer_runner(peer, framework, task)
    else:
        runner_cls = make_delegating_runner(framework, task)

    # Local or worker-with-local-fallback: the weights may be loaded in this
    # process, so pull package staples now. Peer-pinned requests run elsewhere
    # and never trigger a local download.
    if peer is None:
        ensure_staple_weights(model_key)

    logger.debug(
        "resolve: model=%s framework=%s task=%s (requested=%s primary=%s)",
        model_key, framework, task, requested_task, cfg.primary_task,
    )

    return Resolution(
        model_key=model_key,
        framework=framework,          # scalar, not cfg.framework
        task=task,
        cfg=cfg,
        builder=builder,
        runner_cls=runner_cls,
        cache_key=(model_key, task),
    )

# ---------------------------------------------------------------------------
# EXTERNAL runners — declared, not registered (k98).
#
# FRAMEWORK_RUNNERS answers "which in-process Runner class serves (framework,
# task)". Some tasks have no such answer and never will: 'text-to-speech' is
# served by Chatterbox, which is not a transformers pipeline and does not fit
# the Runner protocol — it is a media-bus runner that needs the `chatterbox`
# package and a GPU, i.e. a WORKER. Before this table, the only thing the tree
# could say about such a row was "no runner registered" (see the 2026-07-03
# incident in validate_registry's docstring below, which is exactly this row).
#
# So: task -> (runner module, probe attribute). The module exposes a zero-arg
# probe returning {"importable": bool, "reason": str}. This is a DECLARATION
# (where the runner lives + how to ask whether this box can run it), never a
# registration — nothing here changes resolve()'s behavior, and a task in this
# table still refuses locally with "No runner for (framework, task)". The
# oracle catalog reads it (oracle.catalog._tts_runner_module_name) so
# GET /oracle/capabilities can say WHERE a capability's runner is and WHY this
# box cannot seat it, instead of only that something is missing.
# ---------------------------------------------------------------------------

EXTERNAL_TASK_RUNNERS: dict[str, tuple[str, str]] = {
    "text-to-speech": (
        "hugpy_media.tts.chatterbox_runner", "probe"),
}


def external_runner_for(task: str) -> tuple[str, str] | None:
    """(module, probe attribute) for ``task``, or None when no external runner
    is declared for it. Pure dict read — no import, no probe call; the caller
    decides whether paying for the import is worth it."""
    return EXTERNAL_TASK_RUNNERS.get(task)


# ---------------------------------------------------------------------------
# validate_registry — fail at import time, not on first request.
# ---------------------------------------------------------------------------

def validate_registry() -> None:
    """Walk MODEL_REGISTRY and assert every entry can actually be served.

    Two checks per model:
      1. (framework, primary_task) has a runner registered.
      2. Every task in cfg.tasks has a runner AND a builder registered.

    CURATED staples (declared in code, models_config.MODELS) fail HARD —
    that's a code bug, and import is the right place to catch it, listing
    ALL broken entries at once. DISCOVERED entries (data: a row someone put
    in model_discovery.json) must never brick package import — a data file
    is not allowed to take the whole service down (2026-07-03: a discovered
    'chatterbox' row declaring text-to-speech with no runner made every
    import — and thus any service restart — fail; k98: that row's runner is
    now DECLARED in EXTERNAL_TASK_RUNNERS above — it lives on a worker, not
    in this process, so "no runner registered" here is correct AND now says
    where it is). Unservable discovered
    entries are logged loudly and KEPT (2026-07-29 — they used to be popped
    from MODEL_REGISTRY only, which desynced it from MODEL_REGISTRY_DICT and
    made the model listable but undesignatable; see the comment at the log
    call). This function no longer mutates the registry: it reports.
    """
    errors: list[str] = []
    missing_plugins: dict[str, list[str]] = {}

    try:
        from hugpy_engine.config.models.models_config import MODELS as _STAPLES
    except Exception:
        _STAPLES = {}

    from hugpy_engine.categories import RUNNER_PAIRS

    def _plugin_missing(framework, task) -> bool:
        # A pair the ecosystem serves (categories.RUNNER_PAIRS) that no package
        # has registered here is an uninstalled plugin (hugpy-media /
        # hugpy-video not wired on this box), not a broken registry row.
        # Report it once per task, never raise. A pair OUTSIDE RUNNER_PAIRS on
        # a staple is still a code bug and fails hard below.
        return (framework, task) in RUNNER_PAIRS and (framework, task) not in FRAMEWORK_RUNNERS

    for model_key, cfg in list(MODEL_REGISTRY.items()):
        entry_errors: list[str] = []
        primary_key = (cfg.framework, cfg.primary_task)
        if primary_key not in FRAMEWORK_RUNNERS:
            entry_errors.append(
                f"  {model_key}: primary_task={cfg.primary_task!r} on "
                f"framework={cfg.framework!r} has no runner registered"
            )

        for task in cfg.tasks:
            task_key = (cfg.framework, task)
            if task_key not in FRAMEWORK_RUNNERS:
                entry_errors.append(
                    f"  {model_key}: task={task!r} in cfg.tasks on "
                    f"framework={cfg.framework!r} has no runner registered"
                )
            if task_key not in MODEL_REQUEST_BUILDERS:
                entry_errors.append(
                    f"  {model_key}: task={task!r} in cfg.tasks on "
                    f"framework={cfg.framework!r} has no request builder registered"
                )

        if not entry_errors:
            continue
        pairs = {(cfg.framework, t) for t in [cfg.primary_task, *cfg.tasks]}
        unplugged = [t for (fw, t) in pairs if _plugin_missing(fw, t)]
        if unplugged and all((fw, t) in FRAMEWORK_RUNNERS or _plugin_missing(fw, t) for fw, t in pairs):
            for t in unplugged:
                missing_plugins.setdefault(str(t), []).append(model_key)
            continue
        if model_key in _STAPLES:
            errors.extend(entry_errors)     # code bug — fail import
        else:
            # KEPT, not dropped (2026-07-29). Popping only MODEL_REGISTRY left
            # it ASYMMETRIC with MODEL_REGISTRY_DICT (never popped, and
            # refresh_registry re-adds the row to BOTH anyway) — so the model
            # listed in /models and counted on disk, but get_model_config(k)
            # in OBJECT form raised "Unknown model" and every designation
            # refusal blamed the DISK instead of the missing runner. Operator
            # standing order: a model may be inefficient, never silently
            # unavailable. Refusal belongs at the point of use — resolve()
            # already refuses this exact pair loudly and precisely with
            # "No runner for (framework, task)".
            logger.error(
                "registry: discovered model %s cannot be SERVED here (%s). "
                "The row is KEPT and remains listable/designatable; any "
                "attempt to resolve it refuses with the missing runner/builder "
                "pair. Register that pair and it becomes servable.",
                model_key, "; ".join(e.strip() for e in entry_errors))

    for task, keys in sorted(missing_plugins.items()):
        logger.warning(
            "registry: no runner plugin installed for task %r (%d model(s): %s); "
            "install/register hugpy-media or hugpy-video to serve it here",
            task, len(keys), ", ".join(sorted(keys)[:8]))

    if errors:
        raise RuntimeError(
            f"MODEL_REGISTRY validation failed ({len(errors)} issues):\n"
            + "\n".join(errors)
            + f"\n\nRegistered runners:  {sorted(FRAMEWORK_RUNNERS)}"
            + f"\nRegistered builders: {sorted(MODEL_REQUEST_BUILDERS)}"
        )

    logger.info(
        "validate_registry: ok — %d models, %d runner pairs, %d builder pairs",
        len(MODEL_REGISTRY), len(FRAMEWORK_RUNNERS), len(MODEL_REQUEST_BUILDERS),
    )


# Run at import time. If the registry is bad, fail loudly here — not
# halfway through a user's request.
validate_registry()
