"""Storage roots, Hugging Face cache layout and fleet-wide defaults.

Every value is env-overridable. Env reads go through
:func:`hugpy_platform.platform_facade.env_value` (process environment first,
then a ``.env`` file via ``abstract_essentials``), which also strips inline
``# comments`` and stray quotes so a path can never be poisoned by a note in a
``.env`` file.

Importing this module has deliberate side effects: the storage directories are
created (best-effort) and ``HF_HOME``/``HF_HUB_CACHE``/``TORCH_HOME``/
``PIP_CACHE_DIR`` are seeded into the process environment. ``hfApi`` is
resolved lazily (PEP 562) so ``huggingface_hub`` is only imported by callers
that actually use it.
"""
import os
import logging
import re
from typing import Literal, Optional

from abstract_essentials.json_utils import safe_dump_to_file
from abstract_essentials.types import make_list

from hugpy_platform.platform_facade import env_value as get_env_value

# Tokenizers set this as a sentinel for "no enforced limit". It's never a real window.

# ---------------------------------------------------------------------
# Model storage root
# ---------------------------------------------------------------------
HUGGINGFACE_DOMAIN = "https://huggingface.co"

# Don't phone home usage telemetry on the HF calls we do still make (the
# per-repo metadata cache in hugpy_storage.model_metadata minimizes those calls; this
# strips the tracking headers from the rest). setdefault ONLY — an
# operator-set value is never clobbered.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HF_TOKEN = get_env_value("HF_TOKEN") or False


def env_bool(key: str, default: bool = False) -> bool:
    """Env flag -> bool. `get_env_value(...) or default` can never yield False
    (the string \"false\" is truthy, unset -> default), so flags parsed that way
    are stuck at their default forever. This coerces properly.

    Process environment wins over the .env file: get_env_value only reads the
    .env file, but operational flags must also respond to a systemd
    `Environment=` line or a `FLAG=false cmd` shell prefix."""
    value = os.environ.get(key)
    if value is None:
        value = get_env_value(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")

def _resolve_default_root():
    # Single source of truth for the storage root, per-OS. Honours DEFAULT_ROOT,
    # keeps the historical /mnt/llm_storage mount when present and writable, and
    # otherwise lands in a per-user data dir (XDG / AppData / Library) so the API
    # works out of the box on a fresh box, Windows, or macOS.
    from hugpy_platform.app_dirs import models_root

    return models_root()

DEFAULT_ROOT = _resolve_default_root()


def _root_is_usable(path: str) -> bool:
    """True only if *path* can be created AND actually written to.

    ``os.makedirs`` catches the PermissionError case (an unwritable parent like
    ``/opt/flaskapps``); the create+unlink probe catches filesystems where
    ``os.access`` lies (root-squash NFS, euid=0). Cheap, local, no network — this
    runs once at import. Mirrors ``app_dirs._usable`` but with a real
    write probe because an env-supplied root that can't be written is the exact
    failure we're guarding against."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return False
    probe = os.path.join(path, ".hugpy_write_test")
    try:
        with open(probe, "w"):
            pass
        os.unlink(probe)
        return True
    except OSError:
        try:
            os.unlink(probe)
        except OSError:
            pass
        return False


def _env_root_or_default(env_key: str, default_path: str) -> str:
    """Resolve a storage root from *env_key*, falling back to *default_path* when
    the env value points somewhere we cannot actually write.

    Root cause (worker boxes, 2026-07-05): a stale ``~/.env`` carried
    ``MODELS_HOME`` / ``DEFAULT_ROOT`` etc. pointing at an unwritable legacy path
    (``/opt/flaskapps``, a dead ``/mnt/llm_storage`` mount). Provisioning then
    died with PermissionError forever and the models_local scan looked in the
    wrong place. An env-supplied root that can't be created + written is a
    misconfiguration, not a destination: warn LOUDLY naming the offending var and
    value, and use the platform default (derived from the already-validated
    DEFAULT_ROOT) instead of carrying the poisoned path forward.

    Unset env -> default, no probe: the default is under DEFAULT_ROOT, which
    ``_platform.paths.models_root()`` already validated. Env set + usable -> the
    env value UNCHANGED, so central and every correctly-configured box behave
    exactly as before (the reported /mnt/llm_storage on central IS writable and
    passes)."""
    value = get_env_value(env_key)
    if not value:
        return default_path
    if _root_is_usable(value):
        return value
    logging.getLogger("abstract_hugpy").warning(
        "%s=%r is not writable (cannot create/write there); IGNORING it and "
        "falling back to %s. Fix or remove this env var — it is usually a stale "
        "~/.env carrying a legacy/dead storage root onto this box.",
        env_key, value, default_path,
    )
    return default_path


MODELS_HOME = MODELS_DIR =  _env_root_or_default("MODELS_HOME", os.path.join(DEFAULT_ROOT,"models"))

UPLOADS_HOME = CHAT_UPLOAD_DIR =  _env_root_or_default("UPLOADS_HOME", os.path.join(DEFAULT_ROOT,"uploads"))

PROJECTS_HOME = PROJECTS_DIR =  _env_root_or_default("PROJECTS_HOME", os.path.join(DEFAULT_ROOT,"projects"))

# Identity profiles are PERSISTENT library items that OWN their reference images.
# IDENTITIES_HOME is a first-class storage root and a SIBLING of UPLOADS_HOME (both
# under DEFAULT_ROOT) — never a child of UPLOADS_HOME — so the session-scoped
# upload reaper (upload_routes._wipe_session, jailed to UPLOADS_HOME) structurally
# cannot reach an identity's copied reference images. Env-overridable exactly like
# its siblings so tests can repoint it before import.
IDENTITIES_HOME = IDENTITIES_DIR =  _env_root_or_default("IDENTITIES_HOME", os.path.join(DEFAULT_ROOT,"identities"))

PROJECTS_PLACEMENT_PATH = get_env_value("PROJECTS_PLACEMENT_PATH") or os.path.join(PROJECTS_HOME,"placement.json")

DATASETS_HOME = DATASETS_DIR =  _env_root_or_default("DATASETS_HOME", os.path.join(DEFAULT_ROOT,"datasets"))

MODELS_DISCOVERY_PATH = get_env_value("MODELS_DISCOVERY_PATH") or os.path.join(PROJECTS_HOME,"model_discovery.json")

MODELS_DICT_PATH = get_env_value("MODELS_DICT_PATH") or os.path.join(PROJECTS_HOME,"model_manifest.json")

# ── Console-managed Hugging Face token ──────────────────────────────────────
# The operator can save an HF token from the console so HF calls (search /
# metadata / downloads) are authenticated instead of anonymously rate-limited.
# It is persisted as a 0600 file OUTSIDE any git tree, next to the model
# manifest under PROJECTS_HOME — the same runtime state root api_keys.json uses.
# The storage-side store (hugpy_storage.hf_token) OWNS writing,
# validation, and the routes; it reads THIS path so there is one source of truth
# for where the file lives, with no flask->constants layering violation.
#
# Precedence: a STORED token wins; an env HF_TOKEN is the fallback (source:"env").
HF_TOKEN_PATH = get_env_value("HF_TOKEN_PATH") or os.path.join(PROJECTS_HOME, "hf_token")


def read_stored_hf_token():
    """The console-saved HF token from HF_TOKEN_PATH, or False if none/unreadable.
    Never raises; never logs the token."""
    try:
        with open(HF_TOKEN_PATH, "r", encoding="utf-8") as fh:
            tok = fh.read().strip()
        return tok or False
    except OSError:
        return False


# The GENUINE operator-supplied env token, captured ONCE before we pollute the
# process env with a stored token below. Source detection / fallback keys off
# THIS (never live os.environ, which apply_hf_token_to_env overwrites) so a
# stored token can never masquerade as an "env" token after it is cleared.
# Both the process env and the .env-backed get_env_value are consulted.
HF_TOKEN_ENV = (os.environ.get("HF_TOKEN") or get_env_value("HF_TOKEN") or "").strip() or False

# Seed the process token: a stored token overrides the env one. Setting
# os.environ["HF_TOKEN"] means every huggingface_hub call that does NOT force
# token=False (bare hf_hub_download / snapshot_download / HfApi()) picks it up
# automatically at call time; the lazy ``hfApi`` (module __getattr__ below) and
# the server's search routes honour this resolved value.
_stored_hf_token = read_stored_hf_token()
if _stored_hf_token:
    HF_TOKEN = _stored_hf_token
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

HF_CACHE = _env_root_or_default("HF_CACHE", os.path.join(MODELS_HOME,"cache"))

HF_HOME = get_env_value("HF_HOME") or os.path.join(HF_CACHE,"huggingface")

HF_HUB_CACHE = get_env_value("HF_HUB_CACHE") or os.path.join(HF_HOME,"hub")

TORCH_HOME = get_env_value("TORCH_HOME") or os.path.join(HF_CACHE,"torch")

PIP_CACHE_DIR = get_env_value("PIP_CACHE_DIR") or os.path.join(HF_CACHE,"pip")

PATHS = [
    MODELS_HOME,
    UPLOADS_HOME,
    PROJECTS_HOME,
    IDENTITIES_HOME,
    DATASETS_HOME,
    HF_HUB_CACHE,
    HF_CACHE,
    HF_HUB_CACHE,
    TORCH_HOME,
    PIP_CACHE_DIR
]



def _ensure_dirs(paths):
    """Best-effort create the storage dirs.

    Importing abstract_hugpy must never hard-crash just because a storage path
    can't be made — e.g. on a worker box where DEFAULT_ROOT (/mnt/llm_storage)
    is a broken/stale mount (OSError errno 5) or simply not present. Each dir is
    created independently; failures are warned about, not fatal. Set
    DEFAULT_ROOT to a local, writable path on such boxes.
    """
    import logging
    failed = []
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            failed.append((path, exc))
    if failed:
        logging.getLogger("abstract_hugpy").warning(
            "could not create %d storage dir(s); continuing. "
            "Set DEFAULT_ROOT to a writable path to silence this. Details: %s",
            len(failed),
            "; ".join(f"{p} ({e.__class__.__name__}: {e})" for p, e in failed),
        )


_ensure_dirs(PATHS)
if not os.path.isfile(PROJECTS_PLACEMENT_PATH):
    safe_dump_to_file(file_path=PROJECTS_PLACEMENT_PATH,data={})
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
os.environ.setdefault("TORCH_HOME", TORCH_HOME)
os.environ.setdefault("PIP_CACHE_DIR", PIP_CACHE_DIR)

HUGPY_MARKER= get_env_value("HUGPY_MARKER") or "hugpy.json"

LLAMA_HOST= get_env_value("LLAMA_HOST") or "http://127.0.0.1"
VISION_HOST= get_env_value("VISION_HOST") or "http://127.0.0.1"

# `legacy` is the operator's data-hoarding store — kept on disk but never
# walked, indexed, or counted by any metric/appendage in these systems.
EXCLUDE_DIR_NAMES = make_list(get_env_value("EXCLUDE_DIR_NAMES") or ".cache,.git,.locks,snapshots,blobs,refs,1_Pooling,2_Normalize,onnx,legacy,_archive,incoming,dedupes")  # _archive/incoming/dedupes (2026-09-10): cold storage + staging are not live catalog rows
EXCLUDE_DIR_NAMES = frozenset(EXCLUDE_DIR_NAMES)

EXCLUDE_DIR_PREFIXES = make_list(get_env_value("EXCLUDE_DIR_PREFIXES") or "models--")
EXCLUDE_DIR_PREFIXES = tuple(EXCLUDE_DIR_PREFIXES)  # HF cache root naming

TOKENIZER_SENTINEL_THRESHOLD = float(get_env_value("TOKENIZER_SENTINEL_THRESHOLD") or 10**9)
DEFAULT_TIMEOUT= float(get_env_value("DEFAULT_TIMEOUT") or 3600.0)
DEFAULT_MAX_TOKENS= int(get_env_value("DEFAULT_MAX_TOKENS") or 32768)
MIN_INPUT_WORDS_DEFAULT = get_env_value("MIN_INPUT_WORDS_DEFAULT") or 10
SOURCEKIND = make_list(get_env_value("SOURCEKIND") or "text,url,file,image")
SOURCEKIND =Literal[*SOURCEKIND]

JOBSTATUS = make_list(get_env_value("JOBSTATUS") or "queued,running,completed,failed,cancelled")
JOBSTATUS =Literal[*JOBSTATUS]

DEFAULT_TEMPERATURE = float(get_env_value("DEFAULT_TEMPERATURE") or 0.1)
DEFAULT_TOP_P = float(get_env_value("DEFAULT_TOP_P") or 1)

FINISH_REASONS = make_list(get_env_value("FINISH_REASONS") or "stop,max_tokens,cancelled,error")
FINISH_REASONS =Literal[*FINISH_REASONS]

ROLES = make_list(get_env_value("ROLES") or "system,user,assistant,tool")
ROLES = Literal[*ROLES]

# Stock defaults: one model per task, every one a curated staple in
# models_config.MODELS — efficiency-first picks (whole fleet ~17GB).
DEFAULT_CHAT_MODEL = get_env_value("DEFAULT_CHAT_MODEL") or "Qwen2.5-7B-Instruct-GGUF"
# The AGENT BRAIN — the model agent nodes/loops default to. A dedicated
# variable (operator ask, 2026-07-17) so the agents' default is never
# conflated with the generic chat default or the Discord bot's
# DEFAULT_MODEL_KEY. Brain switch 2026-07-17 (P3.4 scorecard, reliability
# over speed: coder 4/4 vs flux2-klein 3/4): Qwen3-Coder-Next. The
# hugpy_agent CLIENT package carries the same default with its own
# HUGPY_AGENT_BRAIN env knob; this is central's copy of the truth.
DEFAULT_AGENT_BRAIN = get_env_value("DEFAULT_AGENT_BRAIN") or "Qwen~Qwen3-Coder-Next-GGUF"
DEFAULT_VISION_MODEL = get_env_value("DEFAULT_VISION_MODEL") or "Qwen2.5-VL-3B-Instruct-GGUF"
DEFAULT_WHISPER_MODEL = get_env_value("DEFAULT_WHISPER_MODEL") or "whisper-large-v3-turbo"
DEFAULT_SUMMARIZE_MODEL = get_env_value("DEFAULT_SUMMARIZE_MODEL") or "flan-t5-large"
DEFAULT_EMBED_MODEL = get_env_value("DEFAULT_EMBED_MODEL") or "all-minilm-l6-v2"
DEFAULT_IMAGEGEN_MODEL = get_env_value("DEFAULT_IMAGEGEN_MODEL") or "sd-turbo"
DEFAULT_KEYWORDS_MODEL = get_env_value("DEFAULT_KEYWORDS_MODEL") or "all-minilm-l6-v2"
# Vision-analysis family (generic transformers-pipeline runner).
DEFAULT_DEPTH_MODEL = get_env_value("DEFAULT_DEPTH_MODEL") or "depth-anything-v2-small"
DEFAULT_DETECT_MODEL = get_env_value("DEFAULT_DETECT_MODEL") or "detr-resnet-50"
DEFAULT_IMG_CLASSIFY_MODEL = get_env_value("DEFAULT_IMG_CLASSIFY_MODEL") or "vit-base-patch16-224"
DEFAULT_SEGMENT_MODEL = get_env_value("DEFAULT_SEGMENT_MODEL") or "segformer-b0-ade"

DISK_AUTHORITATIVE = make_list(get_env_value("DISK_AUTHORITATIVE") or "name,folder,framework,filename")
OVERLAY_ALLOWED = set(make_list(get_env_value("OVERLAY_ALLOWED") or "port, host, timeout_s, include"))

GGUF_QUANT = re.compile(r"(Q\d+_[A-Z0-9_]+|F16|BF16|F32)", re.I)

DEFAULT_LOCAL_FILES_ONLY = env_bool("DEFAULT_LOCAL_FILES_ONLY", True)

# Kill switch for resolve()-time staple downloads. Default on: a fresh install
# pulls the curated MODELS fleet on first use. Set HUGPY_AUTO_DOWNLOAD=false on
# air-gapped boxes / workers that must never touch the network.
HUGPY_AUTO_DOWNLOAD = env_bool("HUGPY_AUTO_DOWNLOAD", True)


def __getattr__(name: str):
    """Lazy ``hfApi`` (PEP 562): an authenticated ``huggingface_hub.HfApi`` bound
    to the resolved ``HF_TOKEN``. Built on first access and cached in the module
    namespace; ``hugpy_storage.hf_token`` rebinds it when the operator changes the
    stored token. Keeps ``huggingface_hub`` out of the platform import path."""
    if name == "hfApi":
        from huggingface_hub import HfApi  # optional extra: hugpy-platform[hf]

        api = HfApi(token=HF_TOKEN)
        globals()["hfApi"] = api
        return api
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
