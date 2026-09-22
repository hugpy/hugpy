"""Small filesystem / message / naming helpers shared across the ecosystem.

Stdlib plus ``abstract_essentials``. The few helpers that need a storage root or
a fleet default import :mod:`hugpy_platform.constants` lazily, so importing this
module never creates directories.
"""
import base64
import glob
import os
import os.path as osp
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from abstract_essentials import (
    eatAll,
    get_any_value,
    get_dirlist,
    get_env_value,
    get_logFile,
    is_number,
    join_path,
    safe_load_from_json,
)
logger = get_logFile(__name__)

def get_glob(path,ext,recursive=False):
    if recursive:
        return sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))
    return sorted(glob.glob(os.path.join(path, ext)))
def exists(obj):
    try:
        if obj and os.path.exists(str(obj)):
            return True
    except Exception as e:
        print(f"exists: {e}")
        return False
    return False
def is_file(obj):
    try:
        if obj and os.path.isfile(str(obj)):
            return True
    except Exception as e:
        print(f"is_file: {e}")
        return False
    return False
def is_dir(obj):
    try:
        if obj and os.path.isdir(str(obj)):
            return True
    except Exception as e:
        print(f"is_dir: {e}")
        return False
    return False
def get_stat(obj):
    try:
        if obj and isinstance(obj,str):
            obj = Path(obj)
        stat = obj.stat()
        return stat
    except Exception as e:
        print(f"stat: {e}")
        return False
    return False
def st_size(obj):
    try:
        stat = get_stat(obj)
        if stat:
            return stat.st_size
    except Exception as e:
        print(f"st_size: {e}")
        return False
    return False
def st_mtime(obj):
    try:
        stat = get_stat(obj)
        if stat:
            return stat.st_mtime
    except Exception as e:
        print(f"st_size: {e}")
        return False
    return False
def itter_dir(directory):
    itter = []
    if directory and os.path.isdir(directory):
        itter = os.listdir(directory)
    return itter

def get_message(prompt:str=None,role:str=None,content:str=None):
    content = content or prompt
    role = role or "user"
    return {"role": role, "content": content}

def get_messages(prompt:str=None,role:str=None,content:str=None) -> List[dict]:
    message = get_message(prompt=prompt,role=role,content=content)
    return [message]

def config_exists(directory):
    if directory and is_dir(directory):
        json_path = os.path.join(directory,'config.json')
        return is_file(json_path)
    return False
def get_request_id() -> str:
    return str(uuid.uuid1())

def get_parts(obj: str) -> List[str]:
    return [item for item in obj.split("/") if item]

def find_keys_by_type(obj, target_type, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, target_type):
                yield path + (k,)
            yield from find_keys_by_type(v, target_type, path + (k,))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from find_keys_by_type(v, target_type, path + (i,))

def get_unique_part(parts: List[str], comp_parts: List[str]) -> List[str]:
    return [part for part in parts if part not in comp_parts]

def safe_dtype_name(value: Any) -> str:
    """
    Converts torch dtypes or dtype-like objects into stable string values.

    Examples:
        torch.float16  -> "torch.float16"
        torch.bfloat16 -> "torch.bfloat16"
        "auto"         -> "auto"
    """
    if value is None:
        return "None"

    return str(value)


def message_to_dict(message: Any) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()

    if isinstance(message, dict):
        return {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }

    return {
        "role": str(getattr(message, "role", "user")),
        "content": str(getattr(message, "content", "")),
    }


def messages_to_dicts(messages: list[Any]) -> list[dict]:
    return [message_to_dict(message) for message in messages]


def slugify(value: str, fallback: str = "media") -> str:
    value = value.strip()
    value = re.sub(r"[^\w.\- ]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._-")
    return value or fallback


def unique_path(path) -> str:
    if not os.path.exists(path):
        return path
    parent = os.path.dirname(path)
    basename = os.path.basename(path)
    stem,suffix = os.path.splitext(basename)

    for index in range(1, 10_000):
        basename = f"{stem}_{index}{suffix}"
        candidate = os.path.join(parent,basename)
        if not os.path.exists(candidate):
            return candidate

    raise RuntimeError(f"Could not create unique path for: {path}")


def require_file(path, label: str=None) -> str:
    if not path or not isinstance(path, str):
        label = label or os.path.basename(path)
        raise ValueError(f"{label}: missing or not a string ({path!r})")
    if not osp.isfile(path):
        label = label or os.path.basename(path)
        raise FileNotFoundError(f"{label}: not found at {path}")
    return path

def get_base_64_image(path):
    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")
        return image_b64


# ---------------------------------------------------------------------------
# Path / name helpers
# ---------------------------------------------------------------------------
def normalize_folder(folder: str) -> str:
    """Canonical form for folder comparison: stripped, no leading/trailing slashes."""
    return eatAll(folder or "", "/").strip()

def get_name(targetname: str, shortnames: List[str]) -> List[str]:
    shortnames = [s for s in shortnames if s != targetname]
    targetparts = get_parts(targetname)
    for shortname in shortnames:
        shortparts = get_parts(shortname)
        targetparts = get_unique_part(targetparts, shortparts)
    return targetparts

def get_target_name(shortname: str, shortnames: List[str]) -> str:
    targetname = get_name(shortname, shortnames)
    targetnames = targetname or get_parts(shortname)
    return targetnames[-1]

def get_max_model_length(folder):
    from hugpy_platform.constants import DEFAULT_MAX_TOKENS, MODELS_HOME
    all_max_values = []
    module_dir = os.path.join(MODELS_HOME,folder)
    dirlist = os.listdir(module_dir)
    files = [os.path.join(module_dir,file) for file in dirlist if file.endswith('.json')]
    for file in files:
        data = safe_load_from_json(file)
        values = list(find_keys_by_type(data, int, path=()))
        max_values = [value for value in values if 'max_' in value[-1]] or []
        for max_value in max_values:
            if max_value[-1] == "model_max_length":
                return get_any_value(data,"model_max_length")
        all_max_values+=max_values
    return DEFAULT_MAX_TOKENS


def get_port(name):
    key = f"{name}_PORT"
    port = get_env_value(key)
    if port and is_number(port):
        port = int(port)
    return port
def get_host(name):
    key = f"{name}_HOST"
    host = get_env_value(key)
    return host

# ---------------------------------------------------------------------------
# Filesystem inspection
# ---------------------------------------------------------------------------
def get_guffs_in_dir(directory: str) -> List[str]:
    """Every .gguf under a model dir, INCLUDING a split-gguf shard subdir
    (e.g. ``<model>/Qwen3-Coder-Next-Q4_K_M/*-00001-of-00004.gguf``). Recursive
    because shard downloads nest one level down; a model dir is a leaf
    (discovery never descends past one) so this never strays into another model."""
    return sorted(set(
        get_glob(directory, "*.gguf", recursive=True)
        + get_glob(directory, "*.GGUF", recursive=True)
    ))


# Multimodal projector (mmproj) detection. A vision GGUF (e.g.
# Qwen2.5-VL-*-GGUF) ships TWO ggufs: the language model and a separate
# `mmproj-*.gguf` CLIP/vision projector. llama-server loads it via --mmproj;
# llama-cpp-python via a chat handler's clip_model_path. These helpers keep the
# projector out of model-file selection and locate it for the vision wiring.
_MMPROJ_HINTS = ("mmproj", "mm-proj", "mm_proj", "projector")

def is_mmproj_file(name: str) -> bool:
    """True if a filename looks like a multimodal projector gguf, not a model."""
    n = os.path.basename(str(name or "")).lower()
    return n.endswith(".gguf") and any(h in n for h in _MMPROJ_HINTS)

def find_mmproj(model_file_or_dir: str) -> Optional[str]:
    """Return the mmproj projector gguf living beside a model, else None.

    Accepts either the model's gguf path or its directory."""
    p = str(model_file_or_dir or "")
    directory = p if os.path.isdir(p) else os.path.dirname(p)
    if not directory or not os.path.isdir(directory):
        return None
    for g in get_guffs_in_dir(directory):
        if is_mmproj_file(g):
            return g
    return None


def get_config_in_dir(directory: str) -> List[str]:
    return [
        join_path(directory, item)
        for item in get_dirlist(directory)
        if item.endswith("config.json")
    ]

def infer_framework(directory: str) -> Optional[str]:
    """Decide a framework from what's actually on disk.

    Returns None when ambiguous so the registry can fill in. The previous
    rule was 'llama_cpp if filename else None', which mislabels anything
    that isn't a GGUF — including diffusion models like FLUX that happened
    to be tagged llama_cpp by hand.
    """
    try:
        files = get_dirlist(directory)
    except OSError:
        return None

    has_ext = lambda ext: any(f.endswith(ext) for f in files)

    if has_ext(".gguf") or has_ext(".GGUF"):
        return "gguf"
    if has_ext(".safetensors") or has_ext(".bin"):
        return "transformers"
    if has_ext(".onnx"):
        return "onnx"
    return None


def extract_gguf_filename(guffs: List[str], directory: str) -> Optional[str]:
    """Return the GGUF path relative to its folder.

    The previous version did `directory.replace(guff, "")` — backwards,
    since guff is the full joined path and directory doesn't contain it.
    That returned the directory unchanged and then ate slashes off it.

    For multi-shard models (Qwen3-Coder-Next splits into 4 files), this
    returns the path including the shard subdirectory, matching what's
    already in your registry: 'Qwen3-Coder-Next-Q4_K_M/...-00001-of-00004.gguf'.
    """
    if not guffs:
        return None
    # Pick the first shard if there are multiples; sort so it's stable.
    chosen = sorted(guffs)[0]
    return os.path.relpath(chosen, start=directory)

# ---------------------------------------------------------------------------
# Merge — disk wins, registry fills gaps (FIX #2)
# ---------------------------------------------------------------------------

# Fields where a discovered (non-None) value should override the registry.
# Keeps task/hub_id/include in registry territory because those can't be
# inferred from a directory listing.

def merge_disk_over_registry(
    discovered: dict,
    registry: Optional[dict],
) -> Tuple[dict, Dict[str, str]]:
    """Merge with disk taking precedence over registry on _DISK_AUTHORITATIVE.

    Returns (merged_dict, provenance_map) where provenance_map[field] is
    'disk' | 'registry' | 'default'. Provenance is for logging only — not
    fed into ModelConfig.
    """
    from hugpy_platform.constants import DISK_AUTHORITATIVE
    registry = registry or {}
    merged: dict = {}
    prov: Dict[str, str] = {}

    all_fields = set(registry) | set(discovered)
    for field in all_fields:
        disk_val = discovered.get(field)
        reg_val = registry.get(field)

        if field in DISK_AUTHORITATIVE and disk_val is not None:
            merged[field] = disk_val
            prov[field] = "disk"
        elif reg_val is not None:
            merged[field] = reg_val
            prov[field] = "registry"
        elif disk_val is not None:
            merged[field] = disk_val
            prov[field] = "disk"
        else:
            merged[field] = None
            prov[field] = "default"

    return merged, prov


def make_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"



