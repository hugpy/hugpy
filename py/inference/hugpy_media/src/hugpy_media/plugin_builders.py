"""Request builders for the media tasks: ``(prompt_kwargs, model_key) -> request``.

One builder per task, registered with the engine's task table by
``hugpy_media.plugin.register``. They translate the loose ``prompt_kwargs`` a
route or a worker receives into the typed request each media runner consumes,
refusing early (``ValueError``) with the keys they saw when the input cannot
name a task.

Import discipline: this module imports only the media request schemas
(pydantic) and stdlib/platform helpers — never a model stack — so building a
request is always cheap and never needs torch.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from abstract_essentials import derive_media_type, read_from_file

from hugpy_platform.constants import DEFAULT_ROOT, UPLOADS_HOME
from hugpy_platform.utils import make_request_id

from hugpy_media.imagegen.schemas import ImageGenRequest
from hugpy_media.keywords.schemas import KeywordTaskRequest
from hugpy_media.schemas.embeded_schemas import EmbedRequest
from hugpy_media.schemas.summarizer_schemas import SummarizeRequest
from hugpy_media.schemas.whisper_schemas import TranscribeRequest
from hugpy_media.tts.schemas import TtsRequest
from hugpy_media.vision.schemas import VisionRequest
from hugpy_media.vision_analysis.schemas import VisionAnalysisRequest

__all__ = [
    "build_vision_request",
    "build_vl_text_or_vision_request",
    "build_whisper_request",
    "build_tts_request",
    "build_summarize_request",
    "build_embed_request",
    "build_similarity_request",
    "build_imagegen_request",
    "build_img2img_request",
    "build_keywords_request",
    "build_vision_analysis_request",
    "build_document_extraction_request",
    "build_url_extraction_request",
]


def _chat_builder():
    """The engine's chat request builder, looked up through the one task table
    (the engine registers ``text-generation`` there). Falls back to the
    engine's category builders while that registration is still landing."""
    from hugpy_engine.tasks import request_builder_for_task

    builder = request_builder_for_task("text-generation")
    if builder is not None:
        return builder
    from hugpy_engine.resolvers.categories.builders import _build_chat_request

    return _build_chat_request


def _build_vl_text_or_vision_request(kwargs: Dict[str, Any], model_key: str):
    """transformers VL: an IMAGELESS turn is a chat turn, not an error.

    A VL model is text-capable too — its tasks[] carries text-generation
    alongside image-text-to-text — and capability is the FULL list, never the
    primary label. Routing every request for it to the strict vision builder
    made a plain chat call fail with:

        ValueError: vision request needs 'image_path', 'file', or 'image_b64';
        got keys: ['max_chunks','max_new_tokens','messages','model_key',
                   'request_id']

    i.e. a chat request (it has `messages`) rejected for not being an image
    request. Observed on Surogate-3.5-2B during a whole-fleet probe, where it
    fails EVERY text prompt.

    The GGUF half of the engine's table already did the right thing —
    _build_vision_chat_request falls back to chat when there is no image,
    documented there as "so a VL model still answers text turns". The two
    engines disagreed: the identical request succeeded on a GGUF VL model and
    500'd on a transformers one. This closes that gap on the transformers side
    while keeping its distinct shape — WITH an image it still builds a real
    VisionRequest (transformers vision is a different runner path from
    llama.cpp's image_url part, so it cannot simply reuse the GGUF builder).
    """
    if (kwargs.get("image_path") or kwargs.get("file")
            or kwargs.get("image_b64")):
        return _build_vision_request(kwargs, model_key)
    return _chat_builder()(kwargs, model_key)


def _build_vision_request(kwargs: Dict[str, Any], model_key: str) -> VisionRequest:
    image_path = kwargs.get("image_path") or kwargs.get("file")
    image_b64 = kwargs.get("image_b64")
    if image_path is None and image_b64 is None:
        raise ValueError(
            "vision request needs 'image_path', 'file', or 'image_b64'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    # VisionRequest enforces exactly one image source.
    if image_path is not None:
        out["image_path"] = image_path
    else:
        out["image_b64"] = image_b64
    for k in ("prompt", "max_new_tokens", "max_tokens", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    return VisionRequest(**out)


def _build_whisper_request(kwargs: Dict[str, Any], model_key: str) -> TranscribeRequest:
    file_path = kwargs.get("audio_path") or kwargs.get("file")
    if file_path is None:
        raise ValueError(
            "whisper request needs 'audio_path' or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    out: Dict[str, Any] = {
        "model_key": model_key,
        "file_path": file_path,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("model_size", "language", "capture_frames",
              "min_gap_seconds", "long_segment_seconds", "pool",
              "word_timestamps"):
        if k in kwargs:
            out[k] = kwargs[k]
    # 'task' in prompt_kwargs is the dispatch task key, so whisper's own
    # transcribe/translate switch rides separate names.
    whisper_task = kwargs.get("whisper_task")
    if whisper_task is None and kwargs.get("translate"):
        whisper_task = "translate"
    if whisper_task is not None:
        out["task"] = whisper_task
    return TranscribeRequest(**out)


def _build_tts_request(kwargs: Dict[str, Any], model_key: str) -> TtsRequest:
    """text-to-speech: the line to speak, plus an optional AUTHORIZED reference
    voice.

    ``authorized`` is forwarded but never DEFAULTED to True here: it means "k97's
    VOICE gate demanded a grant for this request and got one", a fact only the
    caller upstream of this builder can know. A reference voice that arrives
    without it is refused by the adapter (``ReferenceVoiceUnauthorized``) rather
    than silently downgraded to the default voice — doc invariant 12.
    """
    text = kwargs.get("text") or kwargs.get("prompt")
    if text is None and kwargs.get("file"):
        text = read_from_file(kwargs["file"])
    if text is None or not str(text).strip():
        raise ValueError(
            "text-to-speech request needs 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "text": str(text),
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    # ``reference_audio`` is the voice to clone; a worker rematerializes an
    # inlined file under the key ``file``, so an audio ``file`` counts as the
    # reference ONLY when the caller did not also send the line as text —
    # otherwise the same key would mean two different things. Explicit key wins.
    for k in ("reference_audio", "authorized", "voice_style", "seed",
              "language", "device", "return_b64", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    return TtsRequest(**out)


def _build_summarize_request(kwargs: Dict[str, Any], model_key: str) -> SummarizeRequest:
    text = kwargs.get("text") or kwargs.get("prompt")
    if text is None and kwargs.get("file"):
        text = read_from_file(kwargs["file"])
    if text is None:
        raise ValueError(
            "summarize request needs 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "text": text,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in (
        "preset", "summary_mode", "input_policy",
        "max_chunk_tokens", "min_length", "max_length",
        "do_sample", "min_input_words",
        "consolidation_min_length", "consolidation_max_length",
        "max_output_words", "pool",
    ):
        if k in kwargs:
            out[k] = kwargs[k]
    return SummarizeRequest(**out)


def _texts_from_kwargs(kwargs: Dict[str, Any]) -> list[str]:
    """Shared text extraction: texts | text | prompt | file -> list[str].

    Used by both embed builders. Returns a list even for single-string
    input so the runner doesn't have to branch.
    """
    raw = kwargs.get("texts") or kwargs.get("text") or kwargs.get("prompt")
    if raw is None and kwargs.get("file"):
        raw = read_from_file(kwargs["file"])
    if raw is None:
        raise ValueError(
            "embed request needs 'texts', 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(t, str) for t in raw):
        return list(raw)
    raise TypeError(
        f"embed input must be str or list[str], got {type(raw).__name__}"
    )


def _build_embed_request(kwargs: Dict[str, Any], model_key: str) -> EmbedRequest:
    return EmbedRequest(
        model_key=model_key,
        request_id=kwargs.get("request_id", make_request_id()),
        pool=kwargs.get("pool"),
        texts=_texts_from_kwargs(kwargs),
        normalize=kwargs.get("normalize", True),
        batch_size=kwargs.get("batch_size", 32),
    )


def _build_similarity_request(kwargs: Dict[str, Any], model_key: str) -> EmbedRequest:
    """sentence-similarity needs a second set of texts to compare against."""
    other_raw = (
        kwargs.get("other_texts")
        or kwargs.get("other_text")
        or kwargs.get("compare_to")
    )
    if other_raw is None:
        raise ValueError(
            "sentence-similarity needs 'other_texts', 'other_text', or 'compare_to' "
            f"in addition to 'texts'/'text'/'prompt'/'file'; got keys: {sorted(kwargs)}"
        )
    if isinstance(other_raw, str):
        other_texts = [other_raw]
    elif isinstance(other_raw, list) and all(isinstance(t, str) for t in other_raw):
        other_texts = list(other_raw)
    else:
        raise TypeError(
            f"other_texts must be str or list[str], got {type(other_raw).__name__}"
        )

    return EmbedRequest(
        model_key=model_key,
        request_id=kwargs.get("request_id", make_request_id()),
        pool=kwargs.get("pool"),
        texts=_texts_from_kwargs(kwargs),
        other_texts=other_texts,
        normalize=kwargs.get("normalize", True),
        batch_size=kwargs.get("batch_size", 32),
    )


# ID-LOCK reference stills: at most this many subject references per request.
# Mirrors video_intel.studio.job._MAX_REFERENCE_IMAGES so the STILL arm and the
# VIDEO arm agree on the ceiling (one vernacular — [[keeper-owns-nomenclature]]).
_MAX_REFERENCE_IMAGES = 4


def _is_within(path: str, root: str) -> bool:
    """Whether ``path`` (realpath-resolved) sits under ``root`` — the SAME
    realpath+commonpath jail check as ``video_intel.media_store._is_within``, so
    the imagegen id_lock path enforces the storage jail identically to the studio
    (video) id_lock path. Symlink-safe: it compares realpaths, so a symlink that
    points OUT of the jail is rejected."""
    try:
        return os.path.commonpath([os.path.realpath(path),
                                   os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:                    # different drives / relative garbage
        return False


def _jailed_reference_images(kwargs: Dict[str, Any]):
    """Resolve + jail + image-classify id_lock reference paths — CENTRAL-side.

    ``None`` when absent (the plain, non-locked path). Otherwise a list of jailed
    realpaths. Raises ``ValueError`` AS DATA on a jail escape / missing file /
    non-image / over-count — the same structural discipline the studio id_lock
    route uses.

    Runs only where the paths are REAL: on the worker's second builder pass the
    unreachable paths have already been dropped (remote._worker_payload replaced
    them with reference_images_b64), so ``reference_images`` is absent and this
    returns ``None`` — the jail check never fires on a box that can't see them."""
    raw = kwargs.get("reference_images")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not all(
            isinstance(r, str) and r.strip() for r in raw):
        raise ValueError(
            "reference_images must be a list of non-empty path strings; "
            f"got {raw!r}")
    if len(raw) > _MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"at most {_MAX_REFERENCE_IMAGES} reference_images are accepted; "
            f"got {len(raw)}")
    resolved = []
    for i, p in enumerate(raw):
        rp = os.path.realpath(p)
        if not (_is_within(rp, UPLOADS_HOME) or _is_within(rp, DEFAULT_ROOT)):
            raise ValueError(
                f"reference_images[{i}] escapes the storage jail: {p!r}")
        if not os.path.isfile(rp):
            raise ValueError(f"reference_images[{i}] not found: {p!r}")
        if derive_media_type(rp) != "image":
            raise ValueError(
                f"reference_images[{i}] is not an image "
                f"({derive_media_type(rp)!r}): {os.path.basename(p)}")
        resolved.append(rp)
    return resolved


def _apply_id_lock(out: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
    """Fold the id_lock (identity-locked STILL) fields onto a built imagegen
    ``out`` dict — shared by the text2img and img2img builders so both reach the
    comfy IPAdapter graph the same way. Absent reference_images -> nothing added,
    so a plain request is byte-for-byte what it was before this slice."""
    refs = _jailed_reference_images(kwargs)
    if refs is not None:
        out["reference_images"] = refs
    # reference_images_b64 is the OFFLOAD transport (remote._worker_payload fills
    # it from the paths); pass it through verbatim on the worker's builder pass.
    if kwargs.get("reference_images_b64") is not None:
        out["reference_images_b64"] = kwargs["reference_images_b64"]
    if "id_strength" in kwargs:
        out["id_strength"] = kwargs["id_strength"]


def _build_imagegen_request(kwargs: Dict[str, Any], model_key: str) -> ImageGenRequest:
    prompt = kwargs.get("prompt") or kwargs.get("text")
    if prompt is None and kwargs.get("file"):
        prompt = read_from_file(kwargs["file"])
    if prompt is None:
        raise ValueError(
            "text-to-image request needs 'prompt', 'text', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "prompt": prompt,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("negative_prompt", "width", "height", "num_inference_steps",
              "guidance_scale", "sampler_name", "scheduler", "seed",
              "num_images", "return_b64", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    # 'steps' is the colloquial alias clients reach for first.
    if "steps" in kwargs and "num_inference_steps" not in out:
        out["num_inference_steps"] = kwargs["steps"]
    # 'sampler' is the colloquial alias (matches presets.py's field name).
    if "sampler" in kwargs and "sampler_name" not in out:
        out["sampler_name"] = kwargs["sampler"]
    # ID-LOCK: reference stills + id_strength (identity-locked STILL generation via
    # the comfy IPAdapter graph). Absent -> unchanged text2img.
    _apply_id_lock(out, kwargs)
    return ImageGenRequest(**out)


def _build_img2img_request(kwargs: Dict[str, Any], model_key: str) -> ImageGenRequest:
    """image-to-image (img2img): text2img PLUS a mandatory init image.

    Mirrors _build_imagegen_request, but ALSO resolves the init image. CRITICAL:
    a worker rematerializes the inlined image under key ``file`` (not
    ``image_path``) — so accept EITHER, exactly like _build_vision_request. Raises
    ValueError with a clear message when no init image is present (img2img has
    nothing to condition on)."""
    prompt = kwargs.get("prompt") or kwargs.get("text")
    if prompt is None and kwargs.get("file"):
        # a text file may carry the prompt; an image file is the init image, not
        # the prompt — only read text/document files as prompt text.
        f = kwargs["file"]
        if derive_media_type(f) in ("text", "document", "code"):
            prompt = read_from_file(f)
    if prompt is None:
        raise ValueError(
            "image-to-image request needs 'prompt' or 'text'; "
            f"got keys: {sorted(kwargs)}"
        )

    image_path = kwargs.get("image_path") or kwargs.get("file")
    if image_path is None:
        raise ValueError(
            "image-to-image request needs an init image via 'image_path' or "
            f"'file'; got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "prompt": prompt,
        "image_path": image_path,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("negative_prompt", "width", "height", "num_inference_steps",
              "guidance_scale", "sampler_name", "scheduler", "seed",
              "num_images", "return_b64", "pool", "strength"):
        if k in kwargs:
            out[k] = kwargs[k]
    # 'steps' is the colloquial alias clients reach for first.
    if "steps" in kwargs and "num_inference_steps" not in out:
        out["num_inference_steps"] = kwargs["steps"]
    # 'sampler' is the colloquial alias (matches presets.py's field name).
    if "sampler" in kwargs and "sampler_name" not in out:
        out["sampler_name"] = kwargs["sampler"]
    # ID-LOCK: an init image AND reference stills can co-exist (identity-locked
    # img2img). Absent reference_images -> unchanged img2img.
    _apply_id_lock(out, kwargs)
    return ImageGenRequest(**out)


def _build_keywords_request(kwargs: Dict[str, Any], model_key: str) -> KeywordTaskRequest:
    text = kwargs.get("text") or kwargs.get("prompt")
    if text is None and kwargs.get("file"):
        text = read_from_file(kwargs["file"])
    if text is None:
        raise ValueError(
            "keyword-extraction request needs 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "text": text,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("preset", "refine", "top_n", "diversity", "use_mmr",
              "stop_words", "keyphrase_ngram_range",
              "min_density", "max_density", "min_score", "max_words_per_phrase", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    return KeywordTaskRequest(**out)


def _build_vision_analysis_request(kwargs: Dict[str, Any], model_key: str) -> VisionAnalysisRequest:
    """One builder for the whole vision-analysis family (depth, detection,
    classification, segmentation): input is always one image."""
    image_path = kwargs.get("image_path") or kwargs.get("file")
    image_b64 = kwargs.get("image_b64")
    if image_path is None and image_b64 is None:
        raise ValueError(
            "vision-analysis request needs 'image_path', 'file', or "
            f"'image_b64'; got keys: {sorted(kwargs)}"
        )
    if image_path is not None and derive_media_type(image_path) != "image":
        raise ValueError(
            f"vision-analysis needs an image file; got "
            f"{derive_media_type(image_path)!r} ({os.path.basename(image_path)})"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    if image_path is not None:
        out["image_path"] = image_path
    else:
        out["image_b64"] = image_b64
    for k in ("top_k", "threshold", "candidate_labels", "return_b64", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    return VisionAnalysisRequest(**out)


def _build_document_extraction_request(kwargs: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    """document-extraction (no model): the file to read. Plain dict request —
    ``hugpy_media.extract.extract_document`` takes a path."""
    path = kwargs.get("path") or kwargs.get("file") or kwargs.get("file_path")
    if not path:
        raise ValueError(
            "document-extraction request needs 'path', 'file', or 'file_path'; "
            f"got keys: {sorted(kwargs)}"
        )
    return {
        "model_key": model_key,
        "request_id": kwargs.get("request_id", make_request_id()),
        "path": str(path),
    }


def _build_url_extraction_request(kwargs: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    """url-extraction (no model): the public URL to read. Plain dict request —
    ``hugpy_media.extract.fetch_url_text`` / ``assess_url`` take a URL."""
    url = kwargs.get("url") or kwargs.get("prompt") or kwargs.get("text")
    if not url or not str(url).strip():
        raise ValueError(
            "url-extraction request needs 'url' (or 'prompt'/'text' holding one); "
            f"got keys: {sorted(kwargs)}"
        )
    return {
        "model_key": model_key,
        "request_id": kwargs.get("request_id", make_request_id()),
        "url": str(url).strip(),
        "assess": bool(kwargs.get("assess", False)),
    }


# Public aliases (the engine table historically addressed these as private
# names inside its own builders module; the media package exports them).
build_vision_request = _build_vision_request
build_vl_text_or_vision_request = _build_vl_text_or_vision_request
build_whisper_request = _build_whisper_request
build_tts_request = _build_tts_request
build_summarize_request = _build_summarize_request
build_embed_request = _build_embed_request
build_similarity_request = _build_similarity_request
build_imagegen_request = _build_imagegen_request
build_img2img_request = _build_img2img_request
build_keywords_request = _build_keywords_request
build_vision_analysis_request = _build_vision_analysis_request
build_document_extraction_request = _build_document_extraction_request
build_url_extraction_request = _build_url_extraction_request
