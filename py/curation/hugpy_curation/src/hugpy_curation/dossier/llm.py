"""k120 — the one text model this package is allowed to ask for prose.

Two places in a dossier are written by a model rather than measured: the
``research_notes`` summary of the card + papers, and the community CLAIMS
extracted from public posts. Both are labelled model-generated everywhere they
surface, and both go through this seam so there is exactly one answer to "which
model wrote this?" and it is recorded on the dossier.

RESOLUTION ORDER, and why
    1. ``DOSSIER_LLM_MODEL`` — an operator pin beats everything.
    2. The k109 ROUTING MATRIX's primary for ``plot.construct``. If the fleet
       has measured which model is best at authoring prose from a brief, that
       is the model that should be writing prose from a brief. The matrix is
       registry-version-verified by ``load_latest_matrix``, so a stale one is
       never honoured.
    3. The unified catalog's ``text.chat`` capability, preferring an
       instruct/chat variant. Discovery is the CATALOG's job, not this
       module's — the same rule ``benchmark.discover_models`` states.

    No model at any step is a complete, honest answer: the caller writes
    ``research_notes=None`` and records the reason. A dossier with no notes is
    fine. A dossier with invented notes is not.

Dispatch goes through ``oracle.runtime`` — the SAME front door the runtime and
the k90c judge use — so operator blocks, eligibility and the authority gate all
still apply to a summary request.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

#: Operator pin. Set it and nothing else is consulted.
MODEL_ENV: str = "DOSSIER_LLM_MODEL"

#: The operation whose measured winner we borrow (see the docstring).
PROSE_OPERATION: str = "plot.construct"

#: Per-call ceiling. A summary that has not arrived in two minutes is not going
#: to improve the dossier enough to justify holding the nightly run.
DEADLINE_S: float = float(os.environ.get("DOSSIER_LLM_DEADLINE") or 120.0)

DEFAULT_MAX_TOKENS: int = 700

#: Name fragments that mark a chat/instruct variant, best first. Used only for
#: the catalog fallback, where we must pick one of many.
_PREFERRED = ("instruct", "chat", "-it", "distill")


def _matrix_model() -> tuple[str | None, str]:
    try:
        from hugpy_oracle.routing_matrix import best_route, load_latest_matrix
    except Exception as exc:                        # noqa: BLE001
        return None, f"routing matrix unavailable ({type(exc).__name__})"
    try:
        matrix, reason = load_latest_matrix()
    except Exception as exc:                        # noqa: BLE001
        return None, f"routing matrix load failed ({type(exc).__name__}: {exc})"
    if matrix is None:
        return None, reason
    choice = best_route(PROSE_OPERATION, matrix)
    if choice is None or not choice.primary:
        return None, (f"the routing matrix carries no primary for "
                      f"{PROSE_OPERATION!r}")
    return choice.primary, (f"routing matrix primary for {PROSE_OPERATION} "
                            f"({reason})")


def _catalog_model() -> tuple[str | None, str]:
    try:
        from hugpy_oracle import catalog
        view = catalog.get_capability("text.chat")
    except Exception as exc:                        # noqa: BLE001
        return None, f"catalog unreadable ({type(exc).__name__}: {exc})"
    if view is None or not getattr(view.eligibility, "eligible", False):
        reasons = "; ".join(getattr(getattr(view, "eligibility", None),
                                    "reasons", ()) or ()) or "not eligible"
        return None, f"text.chat is not eligible on this fleet: {reasons}"
    ids = list(view.model_ids or ())
    if not ids:
        return None, "the catalog lists no text.chat model on this fleet"
    for needle in _PREFERRED:
        for model_id in ids:
            if needle in model_id.lower():
                return model_id, f"catalog text.chat pick (matched {needle!r})"
    return ids[0], "catalog text.chat, first eligible"


def resolve_model() -> tuple[str | None, str]:
    """``(model_id, why)``. ``why`` is filled on BOTH branches — a caller
    showing an operator model-generated text needs to say which model, and a
    caller showing nothing needs to say why nothing."""
    pinned = os.environ.get(MODEL_ENV)
    if pinned:
        return pinned, f"pinned by {MODEL_ENV}"
    model, why = _matrix_model()
    if model:
        return model, why
    fallback, why2 = _catalog_model()
    if fallback:
        return fallback, f"{why2} (matrix unavailable: {why})"
    return None, f"{why2}; matrix: {why}"


def _no_think(prompt: str) -> str:
    """NO-THINK, the package seam. A <think> block can eat the whole token
    budget before the answer starts — the same failure ``review/judge.py``
    documents."""
    try:
        from hugpy_engine.utils.no_think import with_no_think
        return with_no_think(prompt)
    except Exception:                               # noqa: BLE001
        return prompt


def _dispatch(model: str, prompt: str, max_tokens: int) -> str:
    from hugpy_oracle import benchmark, runtime
    body: dict[str, Any] = {
        "prompt": _no_think(prompt), "model_key": model, "temperature": 0.0,
        "max_new_tokens": max_tokens,
    }
    result = runtime.run_bounded(
        lambda: runtime._dispatch(runtime._normalized_kwargs(
            "text-generation", body)),
        DEADLINE_S, f"dossier-llm:{model}")
    payload = benchmark._payload(result)
    if not isinstance(payload, Mapping):
        return ""
    if payload.get("ok") is False or payload.get("error"):
        raise RuntimeError(str(payload.get("error") or "dispatch not-ok"))
    return str(payload.get("text") or "")


def ask(prompt: str, *, model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        dispatch: Callable[[str, str, int], str] | None = None
        ) -> tuple[str | None, str | None, str]:
    """``(text, model_id, detail)``. Never raises.

    ``dispatch`` is the injection seam the tests use: a callable
    ``(model, prompt, max_tokens) -> text``, so every caller of this module is
    exercisable with no fleet, no GPU and no network."""
    chosen, why = (model, f"caller-supplied {model!r}") if model \
        else resolve_model()
    if not chosen:
        return None, None, why
    send = dispatch or _dispatch
    try:
        text = send(chosen, prompt, max_tokens)
    except Exception as exc:                        # noqa: BLE001
        logger.info("dossier llm: %s failed (%s: %s)", chosen,
                    type(exc).__name__, exc)
        return None, chosen, f"{chosen} did not answer ({type(exc).__name__}: {exc})"
    text = strip_think(text or "")
    if not text.strip():
        return None, chosen, f"{chosen} returned no text"
    return text.strip(), chosen, why


def strip_think(text: str) -> str:
    """Drop a reasoning block. Delegates to the package seam when it is
    importable (one implementation of this in the tree, not four) and falls
    back to the same regex so this module stays usable standalone."""
    try:
        from hugpy_engine.utils.no_think import strip_think as _strip
        cleaned, _reasoning = _strip(text or "")
        return cleaned
    except Exception:                               # noqa: BLE001
        return re.sub(r"<think>.*?</think>", "", text or "",
                      flags=re.S | re.I).strip()


def extract_json(text: str) -> Any:
    """The package's JSON scavenger, reused. Small local models wrap JSON in
    prose however firmly you ask them not to."""
    try:
        from hugpy_engine.utils.json_scavenge import extract_json_object
        return extract_json_object(text or "")
    except Exception:                               # noqa: BLE001
        return None


__all__ = ["DEADLINE_S", "MODEL_ENV", "PROSE_OPERATION", "ask", "extract_json",
           "resolve_model", "strip_think"]
