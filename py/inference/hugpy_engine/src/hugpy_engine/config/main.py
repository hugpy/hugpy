import os
from typing import Optional
from abstract_essentials import get_env_value, is_dir, join_path
from hugpy_engine.config.models.models_config import MODEL_REGISTRY, get_model_registry
from hugpy_engine.schemas.model_schemas import ModelConfig
from hugpy_platform.constants import MODELS_HOME
from hugpy_platform.utils import config_exists, exists, get_glob, itter_dir, st_mtime, st_size
import logging
logger = logging.getLogger(__name__)



def resolve_hf_model_dir(base_dir: str) -> str:

    if config_exists(base_dir):
        return base_dir

    snapshots = join_path(base_dir,"snapshots")
    if is_dir(snapshots):
        candidates = [
            p for p in itter_dir(snapshots)
            if is_dir(p) and config_exists(p)
        ]

        if candidates:
            return max(candidates, key=lambda p: st_mtime(p))

    raise FileNotFoundError(f"No usable Hugging Face model dir found under: {base_dir}")

# ---------------------------------------------------------------------
# Registry utilities
# ---------------------------------------------------------------------

def list_models():
    return list(MODEL_REGISTRY.keys())


def _resolve_model_key(model_key, registry, prefer=None):
    """Map a possibly-bare model_key to a concrete registry key.

    Keys are bare basenames unless an owner collision forced an
    ``<owner>~<name>`` qualifier (see discover_models). Resolution order:
      1. exact key — covers unique bare keys AND fully-qualified keys;
      2. a single qualified key whose bare suffix matches;
      3. ambiguous bare key -> prefer a variant already allocated to a
         slot/worker (``prefer``), else the first qualified key in sorted order.
    Returns the resolved key, or None when nothing matches."""
    if model_key in registry:
        return model_key
    def _bare(k):  # name without the "owner~" qualifier
        return k.split("~", 1)[1] if "~" in k else k
    candidates = sorted(k for k in registry if _bare(k) == model_key)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for p in (prefer or []):                       # slot/worker allocation wins
        if p in candidates:
            return p
        for c in candidates:                       # prefer may itself be bare
            if _bare(c) == _bare(p):
                return c
    logger.info("ambiguous model_key %r -> %s (candidates: %s)",
                model_key, candidates[0], candidates)
    return candidates[0]                            # stable first-in-list


def get_model_config(model_key: str=None,dict_return=False,return_dict=False,key: str=None,prefer=None) -> ModelConfig or dict:
    model_key = model_key or key
    model_registry = get_model_registry(dict_return=dict_return,return_dict=return_dict)

    resolved = _resolve_model_key(model_key, model_registry, prefer=prefer)
    if resolved is None:
        raise KeyError(f"Unknown model: {model_key}")
    return model_registry[resolved]


def list_model_options():
    return {
        key: {
            "name": cfg.name,
            "hub_id": cfg.hub_id,
            "folder": cfg.folder,
            "tasks": cfg.tasks,
            "framework": cfg.framework,
            "filename": cfg.filename,
            "max_new_tokens": cfg.max_new_tokens,
            "port": cfg.port,
        
        }
        for key, cfg in MODEL_REGISTRY.items()
    }

# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------

def get_model_path(key: str):
    env_override = get_env_value(f"MODEL_{key.upper()}")
    if env_override:
        return env_override
    cfg = get_model_config(key)
    path = os.path.join(MODELS_HOME,cfg.folder)
    # Read-through (2026-07-12 hotfix): cfg.folder is CENTRAL's layout — after
    # the store flattening it names the FLAT dir, but a worker's files may
    # still sit under a legacy task path. Without the resolver here, every
    # presence check (provision.model_is_local) and loader that builds its
    # path from this function declared migrated models "missing" fleet-wide,
    # and the reconcile loops re-pull-stormed central (503s / broken pipes /
    # failed chats). _resolved_local returns the folder path when complete,
    # else the first complete dir under any historical layout.
    return _resolved_local(key, cfg, path)


def get_gguf_file(path: str, cfg: ModelConfig, prefer: Optional[str] = None) -> Optional[str]:
    # The multimodal projector (mmproj-*.gguf) is NOT the model — exclude it so a
    # vision GGUF dir resolves to the language model, not the CLIP projector.
    from hugpy_platform.utils import is_mmproj_file
    # Recursive: split-gguf models nest their shards in a subdir; the shallow
    # glob missed them and resolved to None (the "filename fiasco").
    ggufs = [g for g in (get_glob(path, "*.gguf", recursive=True)
                         + get_glob(path, "*.GGUF", recursive=True))
             if not is_mmproj_file(g)]
    if not ggufs:
        return None

    # 1) Designation wins, and it auto-grabs the file that exists. `prefer` is a
    #    per-request user choice; cfg.filename is the config-level default. Each
    #    matches by exact basename OR by substring (so "Q4_K_M" selects the right
    #    quant without naming the whole file).
    for want in (prefer, getattr(cfg, "filename", None)):
        if not want or is_mmproj_file(want):
            continue
        base = os.path.basename(want).lower()
        for g in ggufs:                                   # exact basename
            if os.path.basename(g).lower() == base:
                return g
        hits = sorted(g for g in ggufs if base in os.path.basename(g).lower())
        if hits:                                          # substring / quant tag
            return hits[0]

    # 2) Exactly one model gguf: unambiguous, grab it.
    if len(ggufs) == 1:
        return ggufs[0]

    # 3) Multiple, no designation: ELECT one. See imports/src/gguf_election.py
    #    for the rule and the incident that produced it — in short: fold shard
    #    sets into variants, refuse to elect an INCOMPLETE shard set, and rank
    #    quantized ahead of full precision.
    #
    #    The rule this replaces tried "first shard of a split gguf" BEFORE the
    #    quant rank and broke ties lexically. On a directory holding a complete
    #    fp16 4-shard split beside a complete q4_k_m 2-shard split, ``f`` sorts
    #    before ``q`` — so it elected 15.2 GB of fp16 over a 4.7 GB q4_k_m on a
    #    fleet whose smallest card is 8 GB (2026-07-28). Sharding is packaging;
    #    it was never a reason to prefer a quantization.
    from hugpy_engine.gguf_election import elect_path
    return elect_path(ggufs) or sorted(ggufs)[0]


# Completeness of a model dir is storage's rule (2026-09-22): one implementation
# in hugpy_storage.model_presence, re-exported here for every existing caller.
from hugpy_storage.model_presence import model_looks_downloaded  # noqa: E402

# ---------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------




def _resolved_local(key: str, cfg, local: str) -> str:
    """The folder-based path, unless it isn't complete AND a copy exists under
    another (legacy/flat) layout — then the read-through resolver's dir. This
    routes the imagegen/embed/keywords/summarizer LOAD path (via DEFAULT_PATHS)
    through the same resolver ensure_model uses, so a model on disk under an OLD
    task path is never 404'd into a re-download during the store flattening."""
    try:
        if model_looks_downloaded(local, cfg):
            return local
        from hugpy_storage.model_paths import resolve_model_dir
        routing = {
            "hub_id": getattr(cfg, "hub_id", None),
            "framework": getattr(cfg, "framework", None),
            "filename": getattr(cfg, "filename", None),
            "include": getattr(cfg, "include", None),
            "primary_task": getattr(cfg, "primary_task", None),
            "tasks": getattr(cfg, "tasks", None),
            "folder": getattr(cfg, "folder", None),
        }
        return resolve_model_dir(routing, cfg=cfg) or local
    except Exception:  # noqa: BLE001 — never raise into resolution
        return local


def resolve_model_source(key: str) -> str:
    cfg = get_model_config(key)
    local = get_model_path(key)
    env_override = get_env_value(f"MODEL_{key.upper()}")

    if env_override and not exists(local):
        raise FileNotFoundError(
            f"MODEL_{key.upper()}={env_override} was set but path does not exist"
        )

    # Read through every historical layout before concluding "not downloaded".
    local = _resolved_local(key, cfg, local)

    if cfg.framework == "gguf":
        if not model_looks_downloaded(local, cfg):
            return cfg.hub_id

        gguf = get_gguf_file(local, cfg)
        if not gguf:
            raise FileNotFoundError(f"No GGUF file found in {local}")

        return str(gguf)

    if model_looks_downloaded(local, cfg):
        # Read-through the box-local NVMe HOT-CACHE tier: a transformers/diffusers
        # model dir is served from the hot copy when complete, else the shared dir
        # unchanged (a background promotion is scheduled). Env-gated
        # (HUGPY_HOT_CACHE_ROOT) so central and un-configured boxes are unaffected
        # — byte-identical when unset. GGUF stays UN-hooked here: its hot-caching
        # happens at the slot load site (slot_agent) so a model is never promoted
        # twice. Never raises into resolution.
        return _hot_cache_dir(str(local))

    return cfg.hub_id


def _hot_cache_dir(shared_dir: str) -> str:
    """Env-gated hot-cache read-through for a transformers/diffusers model dir.
    Returns ``shared_dir`` unchanged on any failure or when the tier is disabled.
    Lazy import keeps imports.config free of a managers dependency at import
    time (both sides import lazily -> no cycle)."""
    try:
        from hugpy_engine.serve import hot_cache
        return hot_cache.use(shared_dir)
    except Exception:  # noqa: BLE001
        return shared_dir


# ---------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------

class _LazyModelPaths:
    """
    Dict-like that resolves on access, not import.

    Managers that cache DEFAULT_PATHS["foo"] in __init__ get the
    correct value at construction time — even if the model was
    downloaded or deleted after the module was first imported.
    """

    def __getitem__(self, key: str) -> str:
        return resolve_model_source(key)

    def get(self, key: str, default=None) -> str:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in MODEL_REGISTRY


DEFAULT_PATHS: _LazyModelPaths = _LazyModelPaths()
