"""Studio-model DISK PRESENCE — the single source of truth for "which real studio
models have their weights on THIS box's disk right now".

The worker agent advertises this set on its heartbeat (``studio.models``) so central's
placement can route a REAL-model studio render to a worker that ACTUALLY holds the
weights — never to a box that would have to transfer them first (operator ruling
2026-09-24: *feasible routing requires the model's files ON DISK on that worker; never
trigger transfers from a route*). It is a pure disk read: it NEVER downloads and NEVER
resolves anything over the network.

The presence gate mirrors what the diffusers studio runners use to decide a model is
loadable off disk: ``<weights_root>/<org>/<name>/model_index.json`` present (the same
completeness gate ``runners/wan_i2v.py::_resolve_model_dir`` applies for both the box-
local NVMe hot root and the shared cold root). A model whose ``weight_uri`` is not an
HF-style ``org/name`` layout (no ``/``) is not reported here — the diffusers presence
gate cannot speak to it, and reporting it would be a guess. SYNTHETIC / ffmpeg last-
resort rows are excluded: they need no weights and always render in-process on central.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# The env vars that name where studio weights live. HOT is a per-box NVMe copy; SHARED
# (STUDIO_WEIGHTS_ROOT) is the fleet cold tier. Both are honoured because a render loads
# from EITHER (wan_i2v._resolve_model_dir prefers hot, falls back to shared).
_SHARED_ROOT_ENV = "STUDIO_WEIGHTS_ROOT"
_HOT_ROOT_ENV = "STUDIO_WEIGHTS_HOT_ROOT"

# The completeness marker a diffusers pipeline dir must carry to be loadable — the SAME
# gate the runner uses, so "advertised present" means "the runner will find it", never a
# half-copied dir.
_PRESENCE_MARKER = "model_index.json"


def weights_roots() -> "tuple[str, ...]":
    """The configured weights roots (hot first, then shared), skipping unset/empty.
    Empty tuple when neither env var is set — the worker then advertises no studio
    models, and central refuses to route a real model there (honest absence)."""
    out: "list[str]" = []
    for env in (_HOT_ROOT_ENV, _SHARED_ROOT_ENV):
        root = (os.environ.get(env) or "").strip()
        if root and root not in out:
            out.append(root)
    return tuple(out)


def _model_dir(root: str, weight_uri: str) -> str:
    """``<root>/<org>/<name>`` for an HF-style ``org/name`` weight_uri (identical to
    ``wan_i2v._local_model_dir``)."""
    parts = [p for p in str(weight_uri).split("/") if p]
    return os.path.join(root, *parts)


def _present_on_disk(weight_uri: str, roots: "tuple[str, ...]") -> bool:
    """True iff the diffusers completeness marker for ``weight_uri`` exists under ANY
    configured root. Pure ``os.path.isfile`` — no download, no network."""
    for root in roots:
        if os.path.isfile(os.path.join(_model_dir(root, weight_uri), _PRESENCE_MARKER)):
            return True
    return False


def present_model_ids(roots: "tuple[str, ...] | None" = None) -> "tuple[str, ...]":
    """The sorted ``model_id`` tuple of every REAL (non-synthetic) studio model whose
    weights are on disk under a configured root.

    Pure and side-effect-free. Any failure (registry unimportable in a stripped context,
    an unreadable root) degrades to ``()`` — the worker then advertises no studio models,
    which central reads as an honest "cannot serve real studio renders here", never a
    false capability."""
    if roots is None:
        roots = weights_roots()
    if not roots:
        return ()
    try:
        from hugpy_video.intel.studio.registry import MODEL_REGISTRY
    except Exception as exc:  # noqa: BLE001 — a stripped worker degrades to "none present"
        logger.debug("studio presence: registry import failed: %s", exc)
        return ()
    out: "list[str]" = []
    for model_id, cfg in MODEL_REGISTRY.items():
        if getattr(cfg, "synthetic", False):
            continue                       # synthetic/ffmpeg need no weights
        weight_uri = getattr(cfg, "weight_uri", "") or ""
        if "/" not in weight_uri:
            continue                       # not an HF org/name layout the gate can read
        try:
            if _present_on_disk(weight_uri, roots):
                out.append(str(model_id))
        except OSError:
            continue                       # one unreadable root must not hide the rest
    return tuple(sorted(out))
