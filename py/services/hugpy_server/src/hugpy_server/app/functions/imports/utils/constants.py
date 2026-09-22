"""Server-side constants and the long-lived Hugging Face client.

``hfApi`` is built once at import from the platform's ``HF_TOKEN``; the
storage package's ``hf_token`` listener (installed by ``hugpy_server.wiring``)
calls ``rebuild_hf_api`` after every token store/delete so a token saved
through the console takes effect without a restart. Read it through
``get_hf_api()`` (never bind the name at import) so the rebuild is seen.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from huggingface_hub import HfApi

from hugpy_platform.constants import GGUF_QUANT, HF_TOKEN  # noqa: F401 — GGUF_QUANT re-exported

JOBSTATUS = Literal["queued", "running", "completed", "failed", "cancelled"]
hfApi = HfApi(token=HF_TOKEN)


def rebuild_hf_api(token: Optional[str] = None) -> HfApi:
    """Replace the module-level client with one bound to ``token`` (``None``
    -> anonymous). Returns the new client."""
    global hfApi
    hfApi = HfApi(token=token or False)
    return hfApi


def get_hf_api() -> HfApi:
    return hfApi


__all__ = ["JOBSTATUS", "hfApi", "GGUF_QUANT", "rebuild_hf_api", "get_hf_api"]
