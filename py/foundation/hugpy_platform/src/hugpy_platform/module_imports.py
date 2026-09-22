"""
Lazy import accessors for heavy third-party libraries.

Rules:
    1. Every accessor is named `get_<package>`.  Downstream modules that
       re-export a *different* thing must choose a different name
       (e.g. `get_keybert_instance`) so the import accessor is never shadowed.
    2. Accessors that support sub-module lookup (`module` param) build a
       dispatch dict once per call.  The dict is cheap; the import behind it
       is cached by Python's import system.
    3. Torch is guarded against the _refs/_prims recursion bug in PyTorch 2.x
       by disabling dynamo eagerly.
    4. `require()` is the gateway for any code that *cannot* function without
       a dependency.  It turns a silent nullProxy into an immediate, readable
       ImportError — right at init, not four calls later as a TypeError.
"""

from __future__ import annotations

from abstract_essentials.lazy_utils import lazy_import, nullProxy


# ---------------------------------------------------------------------------
# Dependency gate
# ---------------------------------------------------------------------------

def is_available(name: str) -> bool:
    """Check whether *name* can be imported without side-effects."""
    obj = lazy_import(name)
    return not isinstance(obj, nullProxy)


# Map an importable package name to the hugpy *extra* that provides it, so the
# "not installed" message points at the one-line opt-in. The base install stays
# lean / phone-clean; these heavy ML stacks are deliberately opt-in extras of
# the `hugpy` meta distribution.
_PKG_EXTRA = {
    "keybert": "keywords",
    "spacy": "keywords",
    "whisper": "audio",
    "openai_whisper": "audio",
    "faster_whisper": "audio",
    "diffusers": "imagegen",
    "transformers": "transformers",
    "torch": "transformers",
    "accelerate": "transformers",
    "sentencepiece": "transformers",
    "sentence_transformers": "embed",
    "peft": "finetune",
    "llama_cpp": "engine",
}


def require(name: str, reason: str = ""):
    """
    Import *name* or raise immediately with a human-readable message.

    Use this in any manager __init__ where the dependency is non-optional.
    nullProxy is designed for graceful degradation — but a model manager
    that silently stores None and explodes later is not graceful, it's a trap.
    """
    obj = lazy_import(name)
    if isinstance(obj, nullProxy):
        hint = f" ({reason})" if reason else ""
        extra = _PKG_EXTRA.get(name)
        how = (
            f"pip install 'hugpy[{extra}]'   (or: pip install {name})"
            if extra else f"pip install {name}"
        )
        raise ImportError(
            f"Required package {name!r} is not installed{hint}. "
            f"Install it with:  {how}"
        )
    return obj


# ---------------------------------------------------------------------------
# Torch  (with dynamo guard)
# ---------------------------------------------------------------------------

def get_torch():
    torch = lazy_import("torch")

    # Guard against PyTorch 2.x _refs dispatch recursion during
    # nn.Embedding init and similar paths.  This is a no-op if dynamo
    # is already disabled or on torch < 2.0.
    _dynamo = getattr(torch, "_dynamo", None)
    if _dynamo is not None and callable(getattr(_dynamo, "disable", None)):
        _dynamo.config.suppress_errors = True

    return torch


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------

def get_spacy():
    return lazy_import("spacy")


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

def get_whisper():
    return lazy_import("whisper")


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

def get_transformers(module=None):
    tf = lazy_import("transformers")
    if module is None:
        return tf

    allowed = {
        "AutoProcessor",
        "AutoTokenizer",
        "AutoModelForSeq2SeqLM",
        "AutoModelForCausalLM",
        "GenerationConfig",
        "BitsAndBytesConfig",
        "pipeline",
        "LEDTokenizer",
        "LEDForConditionalGeneration",
        "T5TokenizerFast",
        "T5ForConditionalGeneration",
        "AutoModelForVision2Seq",
        "Qwen2_5_VLForConditionalGeneration",
        "TextIteratorStreamer",
        "StoppingCriteriaList",
        "BitsAndBytesConfig",
    }
    if module not in allowed:
        raise KeyError(
            f"Unknown transformers sub-module {module!r}. "
            f"Available: {sorted(allowed)}"
        )

    try:
        return getattr(tf, module)
    except AttributeError as exc:
        raise AttributeError(
            f"transformers module is missing {module!r}. "
            f"This can happen if transformers is only partially initialized "
            f"during a circular import or concurrent first-load."
        ) from exc


# ---------------------------------------------------------------------------
# Sentence Transformers
# ---------------------------------------------------------------------------

def get_sentence_transformers(module=None):
    st = lazy_import("sentence_transformers")
    if module is None:
        return st

    _dispatch = {
        "SentenceTransformer": st.SentenceTransformer,
        "models": st.models,
        "cos_sim": lazy_import("sentence_transformers.util").cos_sim,
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown sentence_transformers sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]


# ---------------------------------------------------------------------------
# KeyBERT
# ---------------------------------------------------------------------------

def get_keybert():
    """Return the KeyBERT *class*, not an instance.
    Downstream code that returns an initialised KeyBERT model must use
    a different name (e.g. get_keybert_instance) to avoid shadowing."""
    return lazy_import("keybert").KeyBERT


# ---------------------------------------------------------------------------
# MoviePy
# ---------------------------------------------------------------------------

def get_moviepy(module=None):
    mp = lazy_import("moviepy.editor")
    if module is None:
        return mp

    _dispatch = {
        "mp": mp,
        "VideoFileClip": mp.VideoFileClip,
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown moviepy sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]

def get_tiktoken():
    return lazy_import("tiktoken")


# ---------------------------------------------------------------------------
# PyTesseract
# ---------------------------------------------------------------------------

def get_pytesseract():
    return lazy_import("pytesseract")


# ---------------------------------------------------------------------------
# PyPDF2
# ---------------------------------------------------------------------------

def get_pypdf2():
    return lazy_import("PyPDF2")


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

def get_easyocr():
    return lazy_import("easyocr")


# ---------------------------------------------------------------------------
# SpeechRecognition
# ---------------------------------------------------------------------------

def get_speech_recognition():
    return lazy_import("speech_recognition")


# ---------------------------------------------------------------------------
# PydubAudio
# ---------------------------------------------------------------------------

def get_pydub(module=None):
    pydub = lazy_import("pydub")
    if module is None:
        return pydub
    
    _dispatch = {
        "AudioSegment": lazy_import("pydub").AudioSegment,
        "silence": lazy_import("pydub.silence"),
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown pydub sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

def get_paddleocr():
    return lazy_import("paddleocr").PaddleOCR


# ---------------------------------------------------------------------------
# pdf2image
# ---------------------------------------------------------------------------

def get_pdf2image(module=None):
    pdf2img = lazy_import("pdf2image")
    if module is None:
        return pdf2img
    
    _dispatch = {
        "convert_from_path": pdf2img.convert_from_path,
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown pdf2image sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]

def require_peft():
    try:
        from peft import PeftModel
    except ImportError as exc:
        # `peft` is an EXTRA (pyproject: finetune = ["peft"]), deliberately not a
        # base dependency — this package keeps base installs lean, and most
        # workers never touch an adapter. So an adapter row on a worker without
        # the extra must DEGRADE with the fix in the message, never crash
        # opaquely: the operator pushes it over the existing /ops/pip relay.
        raise RuntimeError(
            "this model is a PEFT/LoRA adapter, but `peft` is not installed on "
            "this worker, so the adapter cannot be applied to its base model. "
            "FIX: install it here — `pip install peft` (or POST /ops/pip with "
            "packages=['peft'] then /ops/restart)."
        ) from exc
    return PeftModel
