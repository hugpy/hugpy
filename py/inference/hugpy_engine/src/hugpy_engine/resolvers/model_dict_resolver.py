import json
import os
from abstract_essentials import get_file_parts, safe_dump_to_file
from hugpy_engine.apis.get_module import build_resolver_chain, enrich
from hugpy_engine.peft_adapters import adapter_metadata_fields
from hugpy_platform.constants import (
    DEFAULT_MAX_TOKENS,
    MODELS_DICT_PATH,
    MODELS_DISCOVERY_PATH,
    MODELS_HOME,
    TOKENIZER_SENTINEL_THRESHOLD,
)
from hugpy_storage.model_paths import get_model_dirs
# ---------------------------------------------------------------------------
# Resolvers — each is independent; failure returns {} not an exception
# ---------------------------------------------------------------------------

def resolve_local_config(directory: str, hub_id: str) -> dict:
    path = os.path.join(directory, "config.json")
    if not os.path.isfile(path):
        # A bare PEFT adapter dir has only adapter_config.json. Returning {}
        # here is half of the 2026-07-29 bench defect — see the twin resolver in
        # imports/apis/get_module.py and imports/src/peft_adapters.py.
        return adapter_metadata_fields(directory)
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "architectures":           cfg.get("architectures"),
        "model_type":              cfg.get("model_type"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "sliding_window":          cfg.get("sliding_window"),
        "rope_scaling":            cfg.get("rope_scaling"),
        "torch_dtype":             cfg.get("torch_dtype"),
        "vocab_size":              cfg.get("vocab_size"),
        # peft markers — present iff this is an adapter
        "peft_type":               cfg.get("peft_type"),
        "base_model":              cfg.get("base_model_name_or_path"),
    }

def resolve_local_tokenizer(directory: str, hub_id: str) -> dict:
    path = os.path.join(directory, "tokenizer_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            tk = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    raw = tk.get("model_max_length")
    if isinstance(raw, (int, float)) and 0 < raw < TOKENIZER_SENTINEL_THRESHOLD:
        return {"tokenizer_model_max_length": int(raw)}
    return {}




# ---------------------------------------------------------------------------
# Registry of resolvers, ordered by trust
# ---------------------------------------------------------------------------








def discover_model(save_json=True,verbose=True,use_hub=True):
    discovered={}
    model_dirs = get_model_dirs()
    chain = build_resolver_chain(use_hub=use_hub)
    for directory in model_dirs:
        file_parts = get_file_parts(directory)
        folder = directory[len(MODELS_HOME):]
        
        name = file_parts.get("basename")
        parent_dirname = file_parts.get("parent_dirname")
        hub_id = directory[len(parent_dirname)+1:]
        meta, meta_sources = enrich(directory, hub_id, chain)
        discovered[name] = {"name":name,"hub_id":hub_id,"folder":folder}
        discovered[name].update({
            "pipeline_tag":            meta.pipeline_tag,
            "library_name":            meta.library_name,
            "auto_model_class":        meta.auto_model_class,
            "architectures":           meta.architectures,
            "model_type":              meta.model_type,
            "max_position_embeddings": meta.max_position_embeddings,
            "model_max_length":        meta.tokenizer_model_max_length or meta.max_position_embeddings or DEFAULT_MAX_TOKENS,
            "parameter_count":         meta.parameter_count,
            "license":                 meta.license,
            "gated":                   meta.gated,
            "languages":               meta.languages,
        })

        if verbose:
            print(f"[enrich] {hub_id}: {meta_sources}")
    if save_json:
        safe_dump_to_file(data=discovered,file_path=MODELS_DICT_PATH)
        
    return discovered
def discover_models(save_json=True, verbose=True, use_hub=True):
    """Walk the model tree and enrich each dir. Descriptive only.
    Writes a REPORT, never the overlay — acquire() owns MODELS_DICT_PATH."""
    discovered = {}
    chain = build_resolver_chain(use_hub=use_hub)

    for directory in get_model_dirs():
        file_parts = get_file_parts(directory)
        name = file_parts.get("basename")
        hub_id = directory[len(file_parts.get("parent_dirname")) + 1:]

        meta, meta_sources = enrich(directory, hub_id, chain)
        discovered[name] = meta.to_dict()        # the typed record, whole
        if verbose:
            print(f"[enrich] {hub_id}: {meta_sources}")

    if save_json:
        safe_dump_to_file(data=discovered, file_path=MODELS_DISCOVERY_PATH)  # NOT MODELS_DICT_PATH
    return discovered
