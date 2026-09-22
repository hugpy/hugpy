"""Request dispatch: runner cache, execution, streaming, eviction."""
from hugpy_engine.dispatch.acquire import acquire
from hugpy_engine.dispatch.dispatch import (
    LoadRefusal,
    clear,
    ensure_headroom_for_load,
    evict,
    execute_chat_stream,
    execute_prompt,
    execute_prompt_stream,
    infer_arg_name,
    last_used_snapshot,
    loaded_disk_detail,
    loaded_model_keys,
    loading_model_keys,
    no_makeroom_active,
    normalize_prompt_kwargs,
    runner_for,
    set_evict_reason,
    set_evictable,
    set_fit_check,
    set_make_room,
    set_post_evict_hook,
    stream_runner,
    supported_task_keys,
    touch_model,
)

__all__ = [
    "LoadRefusal", "acquire", "clear", "ensure_headroom_for_load", "evict",
    "execute_chat_stream", "execute_prompt", "execute_prompt_stream",
    "infer_arg_name", "last_used_snapshot", "loaded_disk_detail",
    "loaded_model_keys", "loading_model_keys", "no_makeroom_active",
    "normalize_prompt_kwargs", "runner_for", "set_evict_reason", "set_evictable",
    "set_fit_check", "set_make_room", "set_post_evict_hook", "stream_runner",
    "supported_task_keys", "touch_model",
]
