"""Request builders: ``(kwargs, model_key) -> TaskRequest`` per (framework, task).

The engine owns only the chat builders (``_build_chat_request``,
``_build_vision_chat_request``); the media and video builders live with their
runners in ``hugpy_media.plugin_builders`` / ``hugpy_video.plugin_builders``
and arrive through :mod:`hugpy_engine.tasks` registration.
:data:`MODEL_REQUEST_BUILDERS` is the live ``(framework, task) -> builder``
view every caller reads; a plugin's own ``build_request`` wins, else the
engine's ``LEGACY_BUILDERS`` entry for that pair (chat only today).
"""
import os
from typing import Any, Callable, Dict, Tuple
from abstract_essentials import derive_media_type, read_from_file
from pydantic import BaseModel
from hugpy_engine.schemas.chat_schemas import ChatRequest
from hugpy_engine.tasks import ANY_FRAMEWORK, BuilderTable
from hugpy_platform.constants import DEFAULT_ROOT, UPLOADS_HOME
from hugpy_platform.utils import make_request_id

def _file_as_chat_text(file: str) -> str:
    """A file attachment rendered as chat text, or a clear refusal.

    text-generation models only consume text, so anything else (image, audio,
    video) is rejected here with the routing hint — the same answer whether the
    caller sent 'prompt' or 'messages'.
    """
    media = derive_media_type(file)
    if media not in ("text", "document", "code"):
        raise ValueError(
            f"text-generation can't consume a {media!r} file "
            f"({os.path.basename(file)}); route it to the matching model"
        )
    content = read_from_file(file)
    return f"------ {os.path.basename(file)} ------\n{content}"


def _build_chat_request(kwargs: Dict[str, Any], model_key: str) -> ChatRequest:
    out: Dict[str, Any] = {"model_key": model_key}
    file = kwargs.get("file")

    if "messages" in kwargs:
        messages = [dict(m) if isinstance(m, dict) else m for m in kwargs["messages"]]
        if file:
            # The attachment belongs to the latest user turn (the chat UI sends
            # the whole history plus the file each round).
            blob = _file_as_chat_text(file)
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role", "user") == "user":
                    m["content"] = f"{m.get('content') or ''}\n{blob}"
                    break
            else:
                messages.append({"role": "user", "content": blob})
        out["messages"] = messages
    elif "prompt" in kwargs:
        prompt = kwargs["prompt"] or ""
        if file:
            prompt = f"{prompt}\n{_file_as_chat_text(file)}"
        out["messages"] = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(
            "chat request needs either 'messages' or 'prompt'; "
            f"got keys: {sorted(kwargs)}"
        )

    # max_chunks: the caller's continuation budget (ChatRequest already defines
    # it). It was silently dropped here, so a /v1 client could never bound the
    # unbounded continue-loop — part of the 2026-07-14 /v1 stall fix.
    # alloc: per-request placement triggers (2026-07-29) — same silent-drop trap
    # as max_chunks above; without forwarding it here the relay never sees it.
    # chat_template_kwargs / logit_bias: t74 hard no-think — this whitelist IS
    # the seam that made the template kwarg unreachable (utils/no_think.py's
    # "zero hits across this package"); forwarding them here is what makes the
    # llama-server body keys reachable end-to-end, on central AND on the
    # worker's second builder pass.
    # no_makeroom: k96 no-evict guarantee — same silent-drop trap as the keys
    # above; without forwarding it here the relay never learns the request is
    # polite and a cold brain load could evict a fleet resident.
    for k in ("max_new_tokens", "temperature", "top_p", "do_sample", "request_id",
              "unbounded", "max_chunks", "pool", "images", "alloc",
              "chat_template_kwargs", "logit_bias", "no_makeroom"):
        if k in kwargs:
            out[k] = kwargs[k]
    out.setdefault("request_id", make_request_id())
    # Default chat to unbounded so the runner keeps generating until the model
    # naturally stops, instead of truncating at a single token cap. Callers can
    # still force a bounded response with unbounded=False / a max_new_tokens cap.
    if "unbounded" not in out and not kwargs.get("max_new_tokens"):
        out["unbounded"] = True
    req = ChatRequest(**out)
    # Ctx-fit guard: a session whose history outgrew the model's window drops
    # its oldest non-system turns here instead of dying on an over-ctx refusal.
    # This builder is the one funnel every chat pass (console, /v1, and each
    # continuation pass) goes through, so a growing history is re-fitted per
    # pass. Fail-open twice over: the guard itself returns the request
    # untouched on any internal error, and an import failure changes nothing.
    try:
        from hugpy_engine.chat_context.chat_context import ctx_fit_chat_request
        req = ctx_fit_chat_request(req)
    except Exception:  # noqa: BLE001
        pass
    return req


def _build_vision_chat_request(kwargs: Dict[str, Any], model_key: str) -> ChatRequest:
    """llama.cpp vision (GGUF + mmproj) rides the chat path.

    Unlike ``_build_chat_request``, an *image* attachment is NOT flattened to
    text (which would raise "text-generation can't consume an image"). Instead
    the image stays on ``ChatRequest.file`` and the llama.cpp runner folds it
    into the latest user turn as an OpenAI ``image_url`` part for the multimodal
    chat handler. Non-image files (docs/text) and imageless turns fall back to
    the normal chat builder unchanged, so a VL model still answers text turns.
    """
    # /ml/vision delivers the image under "image_path"; chat attachments use
    # "file". Accept either (mirrors _build_vision_request) — reading only "file"
    # silently dropped every /ml/vision image to a text-only turn.
    file = kwargs.get("file") or kwargs.get("image_path")
    is_image = bool(file) and derive_media_type(file) == "image"
    if not is_image:
        return _build_chat_request(kwargs, model_key)
    # Build the chat request WITHOUT the image (so it isn't text-flattened),
    # then re-attach it on .file for the runner to pick up (ChatRequest is
    # frozen, so copy with the update).
    kw = {k: v for k, v in kwargs.items() if k not in ("file", "image_path")}
    return _build_chat_request(kw, model_key).model_copy(update={"file": file})


# ---------------------------------------------------------------------------
# Registries — single source of truth.
# ---------------------------------------------------------------------------

# Engine-owned defaults by (framework, task). Media/video pairs are registered
# by their packages with their own builders (hugpy_media.plugin,
# hugpy_video.plugin); nothing here imports them.
LEGACY_BUILDERS: Dict[Tuple[str, str], Callable[[Dict[str, Any], str], BaseModel]] = {
    ("transformers", "text-generation"):              _build_chat_request,
    ("gguf",         "text-generation"):              _build_chat_request,
    # GGUF vision rides the chat path: the image stays on ChatRequest.file and
    # the runner attaches it as an image_url part for the multimodal handler.
    ("gguf",         "image-text-to-text"):           _build_vision_chat_request,
}

# Wildcard fallbacks: a plugin that registers text-generation for a framework
# the table does not spell (or under "*") still gets the chat builder.
for _fw, _task in list(LEGACY_BUILDERS):
    LEGACY_BUILDERS.setdefault((ANY_FRAMEWORK, _task), LEGACY_BUILDERS[(_fw, _task)])
del _fw, _task

# Live view over hugpy_engine.tasks: a plugin's own ``build_request`` wins,
# else the legacy builder for that (framework, task) — the shape every caller
# has always read (``MODEL_REQUEST_BUILDERS.get(key)``, ``key in ...``).
MODEL_REQUEST_BUILDERS = BuilderTable(LEGACY_BUILDERS)

__all__ = [
    "LEGACY_BUILDERS",
    "MODEL_REQUEST_BUILDERS",
    "_build_chat_request",
    "_build_vision_chat_request",
]
