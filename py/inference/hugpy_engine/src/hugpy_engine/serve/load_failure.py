"""Structured model-load failures (2026-09-23).

A GGUF the native loader rejects (``check_tensor_dims: tensor 'token_embd.weight'
has wrong shape``, ``unknown model architecture: 'clip'``) used to reach central,
the benchmark and the grader as the in-process fallback's generic
``ValueError: Failed to load model from file: <path>`` — the slot's HARD
verdict was logged and discarded, the fallback's cause chain severed
(``from None``). The why survived only in the worker journal.

This module is the one vocabulary every layer shares:

* ``ModelLoadFailure`` — a RuntimeError carrying ``load_class`` (one of
  ``LOAD_FAILURE_CLASSES``), ``loader_stderr`` (verbatim and WHOLE — every
  line the loader wrote; never truncated), ``log_ref`` (the per-load log file
  the full stream was persisted to, when the slot agent wrote one) and ``path``
  (the file the loader opened). ``.hard`` is True for ``hard_load_failure`` — the loader rejected
  the file itself, deterministically; no retry or fallback can fix it.
* ``load_failure_of(exc)`` — the additive ``load_failure`` dict for an error
  payload: walks the cause chain for a structured marker, else classifies by
  type/text (``classify=True``), so consumers never need a regex.
* ``from_slot_reply(error, reply)`` — rebuilds the marker on the client side
  of the slot control plane (the slot agent is a separate process), from the
  reply's structured ``load_failure`` or — for a slot process still running
  older code — from the "hard load failure ... Loader stderr: ..." wording.
"""
from __future__ import annotations

import re

# 2026-09-23: no cap. The operator's rule — "logs, if its not logs, its not
# helpful" — the whole loader stream travels; kept as None for importers.
LOADER_STDERR_MAX = None
HARD = "hard_load_failure"
# vision_needs_slot (2026-09-23): a GGUF that ships a multimodal projector
# (mmproj) can only see images through the native llama-server ``--mmproj``
# (a slot child). When no slot could seat it, the load is REFUSED with this
# class instead of silently loading the language weights text-only in-process.
VISION_NEEDS_SLOT = "vision_needs_slot"
# missing_dependency (2026-09-29): the model is on disk but something it
# cannot load without is NOT — a PEFT adapter whose base model is absent from
# the worker's store (veeraragavan410~Llama-3.2-3B-sentiment, base
# unsloth/Llama-3.2-3B-Instruct). FINAL for this attempt and spends nothing:
# no retry can conjure the base; central must mark the job terminal instead of
# "retrying ... worker load state is not confirmed" forever. Repaired by a
# download, so it is state-dependent (fast refusal, never a cached verdict).
MISSING_DEPENDENCY = "missing_dependency"
LOAD_FAILURE_CLASSES = (HARD, "vram_fit", "engine_unavailable", "unreachable",
                        VISION_NEEDS_SLOT, MISSING_DEPENDENCY, "other")
# Classes that are FINAL for the attempt by construction — central's hold loop
# must never classify one as transient, whatever the prose says.
FINAL_LOAD_CLASSES = frozenset((HARD, VISION_NEEDS_SLOT, MISSING_DEPENDENCY))
# The slot agent's HARD wording (slot_agent.Slot.load); central's
# remote._PERMANENT_LOAD_MARKERS family keys on the same phrase.
HARD_MARKER = "hard load failure"


def bound_stderr(text) -> "str | None":
    """Verbatim loader stderr, WHOLE (no cap — 2026-09-23). None when empty.
    The name is kept for importers; it no longer bounds anything."""
    if text is None:
        return None
    s = str(text).strip("\n")
    if not s.strip():
        return None
    return s


class ModelLoadFailure(RuntimeError):
    """A model load failed, with a machine-readable classification.

    Constructible from a single message (``exc_type(msg)``) so the get.py
    refuse-backoff re-raise keeps the type."""

    def __init__(self, message: str = "", *, load_class: str = "other",
                 loader_stderr: "str | None" = None, path: "str | None" = None,
                 model_key: "str | None" = None, log_ref: "str | None" = None):
        super().__init__(message)
        self.load_class = load_class if load_class in LOAD_FAILURE_CLASSES else "other"
        self.loader_stderr = bound_stderr(loader_stderr)
        self.path = str(path) if path else None
        self.model_key = model_key
        # The per-load log file holding the loader's FULL stream (slot agent).
        self.log_ref = str(log_ref) if log_ref else None

    @property
    def hard(self) -> bool:
        return self.load_class == HARD

    @property
    def load_failure(self) -> dict:
        return {"class": self.load_class, "loader_stderr": self.loader_stderr,
                "path": self.path, "model_key": self.model_key,
                "log_ref": self.log_ref, "message": str(self) or None}


class MissingDependencyFailure(ModelLoadFailure):
    """``missing_dependency``: the model cannot load because a dependency it
    names is not in this store. ``base_id`` is the thing to acquire; ``fix``
    is the operator instruction, verbatim. Both ride ``load_failure``."""

    def __init__(self, message: str = "", *, base_id: "str | None" = None,
                 fix: "str | None" = None, **kw):
        kw.setdefault("load_class", MISSING_DEPENDENCY)
        super().__init__(message, **kw)
        self.base_id = base_id
        self.fix = fix

    @property
    def load_failure(self) -> dict:
        out = super().load_failure
        out.update({"base_id": self.base_id, "fix": self.fix})
        return out


class HardLoadFailure(ModelLoadFailure):
    """The native loader rejected the file (``hard_load_failure``). Never
    retried in-process: the same bytes fail the same way, less informatively."""

    def __init__(self, message: str = "", **kw):
        kw.setdefault("load_class", HARD)
        super().__init__(message, **kw)


_STDERR_IN_TEXT = re.compile(r"Loader stderr:\s*(.*?)(?:\s+Attempt \d+, backing off|\s*$)",
                             re.S | re.I)


def _stderr_from_text(text: str) -> "str | None":
    m = _STDERR_IN_TEXT.search(text or "")
    return bound_stderr(m.group(1)) if m else None


def classify_text(text: str) -> str:
    t = (text or "").lower()
    if HARD_MARKER in t or "hard_load_failure" in t:
        return HARD
    if VISION_NEEDS_SLOT in t:
        return VISION_NEEDS_SLOT
    if ("vram" in t and ("needs ~" in t or "won't fit" in t or "ceiling" in t)) \
            or "won't fit" in t or "loadrefusal" in t or "out of memory" in t:
        return "vram_fit"
    if "no local inference engine" in t or "localengineunavailable" in t \
            or "local serving disabled" in t or "needs dependency profile" in t:
        return "engine_unavailable"
    if "connecterror" in t or "connection refused" in t or "timed out" in t \
            or "unreachable" in t or "readtimeout" in t:
        return "unreachable"
    return "other"


def from_slot_reply(error: str, reply: "dict | None" = None, *,
                    model_key: "str | None" = None,
                    path: "str | None" = None) -> ModelLoadFailure:
    """The client-side marker for a slot ``/load`` error reply."""
    lf = (reply or {}).get("load_failure") if isinstance(reply, dict) else None
    msg = f"slot load failed: {error}"
    if isinstance(lf, dict) and lf.get("class"):
        cls = lf.get("class")
        stderr = lf.get("loader_stderr")
        p = lf.get("path") or path
        ref = lf.get("log_ref")
    else:                                   # older slot process: read the wording
        cls = classify_text(error)
        stderr = _stderr_from_text(error)
        p = path
        ref = None
    kind = HardLoadFailure if cls == HARD else ModelLoadFailure
    return kind(msg, load_class=cls, loader_stderr=stderr, path=p, model_key=model_key,
                log_ref=ref)


def _chain(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def load_failure_of(exc: BaseException, *, classify: bool = False,
                    message: bool = False) -> "dict | None":
    """The additive ``load_failure`` payload dict for ``exc``.

    Returns the first structured marker found on the cause chain. Without one:
    ``classify=True`` derives the class from type/text (for a path where every
    failure IS a load failure, e.g. /probe); otherwise None (a generation error
    is not a load failure). ``message=True`` adds the exception text; it is
    also added whenever there is no loader stderr, so a failure never reaches a
    consumer as a bare class name without the real error text."""
    if exc is None:
        return None
    out = None
    for e in _chain(exc):
        lf = getattr(e, "load_failure", None)
        if isinstance(lf, dict) and lf.get("class"):
            out = dict(lf)
            break
    if out is None:
        names = {type(e).__name__ for e in _chain(exc)}
        if "LocalEngineUnavailable" in names and not classify:
            out = {"class": "engine_unavailable", "loader_stderr": None, "path": None}
        elif "LoadRefusal" in names and not classify:
            out = {"class": "vram_fit", "loader_stderr": None, "path": None}
        elif "AdapterBaseUnavailable" in names:
            # A bare adapter refusal that escaped without the structured
            # wrapper (an older call site): still a missing dependency.
            _ab = next(e for e in _chain(exc) if type(e).__name__ == "AdapterBaseUnavailable")
            out = {"class": MISSING_DEPENDENCY, "loader_stderr": None, "path": None,
                   "base_id": getattr(_ab, "base_model", None),
                   "fix": getattr(_ab, "fix", None)}
        elif classify:
            text = " ".join(f"{type(e).__name__}: {e}" for e in _chain(exc))
            cls = classify_text(text)
            if cls == "other":
                if "LocalEngineUnavailable" in names:
                    cls = "engine_unavailable"
                elif "LoadRefusal" in names or "BudgetRefusal" in names:
                    cls = "vram_fit"
                elif any(n in names for n in ("ConnectError", "ConnectTimeout",
                                                "ReadTimeout", "ConnectionError")):
                    cls = "unreachable"
            out = {"class": cls, "loader_stderr": _stderr_from_text(text), "path": None}
    # STRUCTURED REFUSAL (fit-hotfix 2026-09-29): a LoadRefusal carries the
    # worker's typed verdict on ``.reason`` — the plan_fit ``fit_failure``
    # (kind/code + the two figures in ONE basis) and the refusal numbers
    # (budgetable free, device free, external floor, reserve, protected
    # count). Until now only the prose sentence reached load_reports / the
    # inference error, so a 'vram_fit' on a card the heartbeat still showed
    # ~1 GiB free could not be told apart from a mis-priced need. Additive:
    # every existing key keeps its meaning; ``fit_failure`` / ``refusal`` are
    # present only when the chain carries a typed reason.
    for e in _chain(exc):
        r = getattr(e, "reason", None)
        if not isinstance(r, dict):
            continue
        ff = r.get("fit_failure")
        if not (isinstance(ff, dict) or r.get("state") == "refused"):
            continue
        if out is None:
            out = {"class": "vram_fit", "loader_stderr": None, "path": None}
        if isinstance(ff, dict) and ff.get("kind"):
            out["fit_failure"] = {k: ff.get(k) for k in (
                "kind", "code", "need_bytes", "budget_bytes", "plan_n_cpu_moe",
                "contract_n_cpu_moe", "permanent", "state_dependent")}
        numbers = {k: r.get(k) for k in (
            "needs_bytes", "free_vram_bytes", "free_vram_device_bytes",
            "external_floor_bytes", "ceiling_reserve_bytes", "fit_budget_bytes",
            "total_vram_bytes", "subject_resident_bytes", "free_vram_effective_bytes",
            "evicted_freed_bytes", "no_evict") if r.get(k) is not None}
        prot = r.get("protected")
        if isinstance(prot, list):
            numbers["protected_count"] = len(prot)
            numbers["protected"] = [
                {"model_key": p.get("model_key"), "why": p.get("why"),
                 "vram_bytes": p.get("vram_bytes")}
                for p in prot if isinstance(p, dict)]
        if r.get("evicted") is not None:
            numbers["evicted"] = list(r.get("evicted") or [])
        if numbers:
            out["refusal"] = numbers
        break
    if out is not None and (message or not out.get("loader_stderr")):
        out["message"] = f"{type(exc).__name__}: {exc}"
    return out
