"""Physical model state — DERIVED here, persisted once, read as a lookup.

This is the derivation half of the persisted-physical-state design; the storage
half (schema, provenance, atomic multi-process writes, staleness policy) is
``comms/model_physical.py`` — read that docstring first.

WHY IT LIVES HERE
-----------------
"Physical state" is four questions that were each answered by walking the store
on EVERY request that touched a model:

    status / destination / installed_marker / filename_warning
        ``downloader.model_status`` — route_destination globs four runtime
        families' legacy task dirs and stats every candidate, then
        model_looks_downloaded globs the winner. ~10^2 filesystem calls.
    effective_bytes / effective_gguf / gguf_variants / mmproj_bytes / moe
        ``annotate_gguf_size`` (was ``llm_storage_routes._annotate_gguf_size``)
        — gguf_variants_detail recursively lists every servable .gguf.
    dir_bytes / size_bytes
        ``annotate_size`` (was ``llm_storage_routes._annotate_size``) — a
        recursive os.walk of the model dir plus format_select.walk_listing.
    hugpy_marker
        the model's declared-identity blob; its ``moe_capable`` /
        ``bnb_capable`` bools are what ``/llm/workers`` asks TWICE per
        designated model, each a dir resolution plus a JSON read.

Central DOWNLOADED these models; it knows the answers. The two size annotators
moved here from the route so the WRITE path (download completion, the discovery
repair sweep) and the READ path derive through the ONE implementation — a route
private cannot be a write point, and two copies would drift. The workers view
(``functions/imports/utils/workers.py``) reads the same records through the same
functions for the same reason.

THE READ PATH
-------------
:func:`status_fields` / :func:`size_fields` are the listing entry points. Each is
a dict lookup against the persisted table; a miss (never derived / re-keyed /
expired) derives LIVE and writes through, so the cost is paid once per change
rather than once per request. Absent NEVER means zero: a model with no record
is derived, exactly as it was before this change.

:func:`marker_fields` is the same contract for the marker blob.

The three are separate ASPECTS because they cost orders of magnitude apart and
are read by different surfaces: ``/v1/models`` needs only the status half (it
filters on ``status``), ``/models`` needs status + size, ``/llm/workers`` needs
size + marker. A cold ``/v1/models`` therefore still costs one ``model_status``
per unknown model — not the full size walk it never looks at — and the marker
blob can never leak a key into a listing that does not ask for it.

THE WRITE PATH
--------------
:func:`refresh_fields` is the explicit force-refresh (``GET /models/<key>`` —
opening a row IS the re-read) and :func:`rebuild_physical` is the bulk repair
sweep (``/models/discover``, in its background thread). Both always derive live
and rewrite what they found.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from hugpy_storage.catalog_source import catalog_get, catalog_resolve_dir, routing_of
from hugpy_storage.model_paths import resolve_model_dir
from hugpy_storage.model_physical import (
    ASPECT_MARKER,
    ASPECT_SIZE,
    ASPECT_STATUS,
    ASPECT_FIELDS,
    MARKER_FIELDS,
    SIZE_FIELDS,
    STATUS_FIELDS,
    lookup_physical,
    physical_store,
    identity_of,
)

logger = logging.getLogger(__name__)

__all__ = [
    "annotate_gguf_size", "annotate_size",
    "derive_status", "derive_sizes", "derive_marker", "derive_physical",
    "status_fields", "size_fields", "marker_fields", "refresh_fields",
    "stamp_fields", "rebuild_physical",
    "ASPECT_STATUS", "ASPECT_SIZE", "ASPECT_MARKER",
    "STATUS_FIELDS", "SIZE_FIELDS", "MARKER_FIELDS",
]


def _live_model_status(model: dict) -> dict:
    """``downloader.model_status`` resolved at CALL time.

    Late-bound on purpose: it keeps this module out of the package's import
    order (downloader pulls the whole flask functions tree) and it keeps the
    established test idiom — monkeypatching ``downloader.model_status`` — working
    through every caller."""
    from hugpy_storage.console import downloader
    return downloader.model_status(model)


def _model_dir(mk: str, model: Optional[dict] = None, cfg=None) -> Optional[str]:
    """The dir the engine would load ``mk`` from (catalog source: env
    override / recorded folder / read-through), else the resolver over the
    row's own routing facts. None when nothing resolves."""
    if not mk:
        return None
    d = catalog_resolve_dir(mk)
    if d:
        return d
    routing = routing_of(cfg) if cfg is not None else (dict(model) if model else None)
    if not routing:
        return None
    try:
        return resolve_model_dir(routing, require_complete=False)
    except Exception:  # noqa: BLE001
        return None


def _gguf_variants_detail(mk: str, model_dir: str, cfg=None) -> dict:
    """Storage's variant manifest: every on-disk quant (shard sets folded)
    with its bytes, the mmproj size, and the EFFECTIVE variant when it can be
    told from storage facts alone — the pinned ``filename`` or a single
    variant. The engine's election (quant ranking, operator overrides) is a
    serving decision it applies on its own catalog rows; storage never picks
    a quant for it."""
    from hugpy_storage.hugpy_marker import list_gguf_quants
    from hugpy_storage.model_presence import pinned_filename_present, _find_mmproj
    if not model_dir or not os.path.isdir(model_dir):
        return {}
    quants = list_gguf_quants(model_dir)
    if not quants:
        return {}
    pin = getattr(cfg, "filename", None) if cfg is not None else None
    eff = None
    if pin and pinned_filename_present(model_dir, cfg):
        base = os.path.basename(str(pin)).lower()
        for q in quants:
            if q["file"].lower() == base or base in q["file"].lower():
                eff = q
                break
    if eff is None and len(quants) == 1:
        eff = quants[0]
    mm = _find_mmproj(model_dir)
    try:
        mmproj_bytes = os.path.getsize(mm) if mm else None
    except OSError:
        mmproj_bytes = None
    out = {
        "variants": [{"filename": q["file"], "bytes": q["bytes"],
                      "is_effective": q is eff} for q in quants],
        "effective_gguf": eff["file"] if eff else None,
        "effective_bytes": eff["bytes"] if eff else None,
        "mmproj_bytes": mmproj_bytes,
    }
    if eff:
        try:
            from hugpy_storage.gguf_inspect import gguf_moe_detail
            # The first on-disk file of the effective variant (shards sum inside).
            for root, _dirs, files in os.walk(model_dir):
                hit = [f for f in files if f.lower().endswith(".gguf")
                       and f.lower().startswith(os.path.splitext(eff["file"])[0].lower())]
                if hit:
                    d = gguf_moe_detail(os.path.join(root, sorted(hit)[0]))
                    if d and d.get("is_moe"):
                        out["moe"] = d
                    break
        except Exception:  # noqa: BLE001
            pass
    return out


# ──────────────────────────────────────────────────────────────────────────
# derivation — the expensive half, unchanged in behaviour
# ──────────────────────────────────────────────────────────────────────────
def annotate_gguf_size(model: dict, mk: str) -> None:
    """For a GGUF model, attach the EFFECTIVE-quant size — the single quant that
    actually serves (operator ``gguf_file`` override → ``cfg.filename`` → auto),
    plus its mmproj projector — so the console shows the model's real size instead
    of the whole-directory / whole-repo sum (a GGUF repo holds many quants; only
    one is served). No-op for transformers models and for GGUF dirs not downloaded
    here. Model-level + worker-agnostic, so the Models tab AND the worker-card
    strip both read this one number (same /models feed)."""
    fw = (model.get("framework") or "").lower()
    if fw not in ("gguf", "llama_cpp"):
        return
    try:
        cfg = catalog_get(mk)
        model_dir = model.get("destination") or _model_dir(mk, model, cfg)
        d = _gguf_variants_detail(mk, model_dir, cfg)
    except Exception:  # noqa: BLE001 — never break the models list over sizing
        d = {}
    if not d:
        return
    if d.get("effective_bytes"):
        model["effective_bytes"] = d["effective_bytes"]
    model["effective_gguf"] = d.get("effective_gguf")
    model["gguf_variants"] = d.get("variants") or []
    model["mmproj_bytes"] = d.get("mmproj_bytes")
    # MoE (2026-07-24): the effective quant's expert/non-expert split, riding
    # the registry the same way effective_bytes does (computed once per file —
    # spill.gguf_moe_detail caches by path+size+mtime). Feasibility prices the
    # GPU side of a MoE by non_expert_bytes; absent for dense models.
    if d.get("moe"):
        model["moe"] = d["moe"]


def annotate_size(model: dict, mk: str) -> None:
    """Give EVERY model a ``size_bytes`` the picker can show in ANY disposition
    (cold / idle / serving) — so choosing a model for static or on-demand
    residency shows what it costs BEFORE you commit it. GGUF: the effective quant
    that actually serves (already resolved into ``effective_bytes`` by
    annotate_gguf_size), never the all-quants dir sum. Everything else
    (transformers / comfy): the SINGLE-FORMAT effective footprint — the one usable
    weight format + sidecars a worker actually holds, NOT the whole-snapshot sum
    (a mirrored HF repo carries the same weights in 3-5 formats + an fp32 dupe;
    ledgering the dir sum made an ~11GB model read as 45GB — the 2026-07-16 scare).
    ``dir_bytes`` keeps the whole-dir footprint for diagnostics. ``None`` when the
    model isn't on disk."""
    eff = model.get("effective_bytes")
    if eff:
        model["size_bytes"] = eff
        model.setdefault("dir_bytes", eff)
        return
    try:
        from hugpy_storage.model_presence import dir_size_bytes, walk_listing
        from hugpy_storage.providers import get_footprint_selector
        model_dir = model.get("destination") or _model_dir(mk, model)
        dir_bytes = dir_size_bytes(model_dir)          # whole snapshot (all formats)
        model["dir_bytes"] = dir_bytes
        if model_dir:
            listing = walk_listing(model_dir)
            if listing:
                # Same single-format selection the transfer manifest applies
                # (the server installs format_select.effective_bytes via
                # providers.set_footprint_selector), so the ledger equals what
                # a worker would actually hold post-pull. Default: the sum.
                model["size_bytes"] = get_footprint_selector()(
                    listing, model.get("framework"))
            else:
                model["size_bytes"] = dir_bytes
        else:
            model["size_bytes"] = dir_bytes
    except Exception:  # noqa: BLE001 — never break the models list over sizing
        model["size_bytes"] = None


def _dir_mtime(destination: Optional[str]) -> Optional[float]:
    """Provenance only: the model dir's mtime as observed at derive time.

    Recorded so a repair pass (and a human) can see WHAT the record was derived
    from. Deliberately not consulted on the read path — see the staleness note
    in comms/model_physical.py."""
    if not destination:
        return None
    try:
        return os.path.getmtime(destination)
    except OSError:
        return None


def derive_status(model: dict) -> dict:
    """LIVE status half. Exactly ``model_status``'s dict — no more, no less."""
    out = _live_model_status(model)
    return dict(out) if isinstance(out, dict) else {}


def derive_sizes(model: dict, mk: str, status: Optional[dict] = None) -> dict:
    """LIVE size half — precisely the keys the two annotators produce.

    Runs them against a scratch row (identity + status) and returns every SIZE
    field they left behind, so the persisted record carries the same keys, with
    the same values, a caller would have got by running the annotators on the
    row itself. That is what keeps the response shape byte-identical.

    The scratch row is stripped of any size field FIRST. Registry rows are
    cached and stamped IN PLACE by the listings, so a re-derive is routinely
    handed a row that already carries last time's ``size_bytes``. Diffing
    against that would drop every key whose value did not change — the record
    would lose ``size_bytes`` for a model whose size is (correctly) identical,
    and the listing would then report it as unsized. Strip, derive, take what
    is there: no diff, no way to lose a field by being right twice.

    ``annotate_size`` deliberately still sees ``effective_bytes`` — it is set by
    ``annotate_gguf_size`` in this same pass, which is the ordering it expects."""
    scratch: Dict[str, Any] = {k: v for k, v in model.items()
                               if k not in SIZE_FIELDS}
    if status:
        scratch.update(status)
    annotate_gguf_size(scratch, mk)
    annotate_size(scratch, mk)
    return {k: scratch[k] for k in SIZE_FIELDS if k in scratch}


def derive_marker(model: dict, mk: str) -> dict:
    """LIVE read of the model's ``hugpy.json`` declared-identity blob.

    ``{"hugpy_marker": {...}}`` when the dir resolved (``{}`` inside when the
    model has no marker yet — a real answer), ``{"hugpy_marker": None}`` when
    there is no resolvable dir at all. Raises if the resolution machinery
    itself is unavailable, so the caller can degrade to "never determined"
    rather than persist a fabricated absence."""
    from hugpy_storage.hugpy_marker import read_hugpy_marker
    cfg = catalog_get(mk)
    d = _model_dir(mk, model, cfg)
    d = d or getattr(cfg, "dir", None) or getattr(cfg, "directory", None) \
        or (model or {}).get("destination")
    if not d:
        return {"hugpy_marker": None}
    return {"hugpy_marker": read_hugpy_marker(d) or {}}


def derive_physical(model: dict, mk: Optional[str] = None
                    ) -> Tuple[dict, dict, Optional[float]]:
    """Everything physical about one model, live. ``(status, sizes, dir_mtime)``.

    The MARKER aspect is deliberately NOT in the tuple — it is a different
    shape and only the workers view asks for it; :func:`rebuild_physical` and
    :func:`refresh_fields` add it separately."""
    mk = mk or model.get("model_key") or ""
    status = derive_status(model)
    sizes = derive_sizes(model, mk, status)
    return status, sizes, _dir_mtime(status.get("destination"))


# ──────────────────────────────────────────────────────────────────────────
# read path — lookup, derive on miss, write through
# ──────────────────────────────────────────────────────────────────────────
def _persist(mk: str, model: dict, fields: dict, aspects, source: str,
             dir_mtime: Optional[float]) -> None:
    """Best-effort write-through. A store we cannot write must never stop a
    listing from answering — it just keeps deriving live."""
    try:
        physical_store.put(mk, identity_of(model), fields, aspects,
                           source=source, dir_mtime=dir_mtime)
    except Exception:  # noqa: BLE001
        logger.debug("model-physical write-through failed for %s", mk,
                     exc_info=True)


def status_fields(model: dict, mk: Optional[str] = None, *,
                  source: str = "listing") -> dict:
    """The status half for ``model`` — persisted lookup, live derive on a miss.

    THE listing hot path (``/v1/models``, and the first half of ``/models``).
    A warm row costs zero filesystem calls."""
    mk = mk or model.get("model_key") or ""
    if not mk:
        # No key = nothing to persist against. Derive live: correct, uncached,
        # exactly today's behaviour.
        return derive_status(model)
    try:
        fields, state = lookup_physical(mk, model, ASPECT_STATUS)
    except Exception:  # noqa: BLE001
        fields, state = None, "absent"
    if state == "fresh":
        return fields or {}
    status = derive_status(model)
    _persist(mk, model, status, [ASPECT_STATUS], source,
             _dir_mtime(status.get("destination")))
    return status


def size_fields(model: dict, mk: Optional[str] = None, *,
                source: str = "listing") -> dict:
    """The size half for ``model`` — persisted lookup, live derive on a miss.

    Deriving sizes needs ``destination``, so a miss resolves the status half
    first (itself a lookup) and writes both halves back."""
    mk = mk or model.get("model_key") or ""
    if not mk:
        return derive_sizes(model, "", derive_status(model))
    try:
        fields, state = lookup_physical(mk, model, ASPECT_SIZE)
    except Exception:  # noqa: BLE001
        fields, state = None, "absent"
    if state == "fresh":
        return fields or {}
    status = status_fields(model, mk, source=source)
    sizes = derive_sizes(model, mk, status)
    _persist(mk, model, {**status, **sizes}, [ASPECT_STATUS, ASPECT_SIZE],
             source, _dir_mtime(status.get("destination")))
    return sizes


def marker_fields(model: dict, mk: Optional[str] = None, *,
                  source: str = "workers") -> dict:
    """The model's ``hugpy.json`` blob — persisted lookup, live read on a miss.

    THE WORKERS-VIEW hot path: ``/llm/workers`` asks two capability bools
    (moe_capable, bnb_capable) per DESIGNATED model, ~111 across the fleet, and
    each was a dir resolution plus a JSON read from the store on every poll.

    Returns ``{}`` — no ``hugpy_marker`` key at all — when the marker could not
    be read. That is "never determined", which every caller must treat as
    unknown; it is NOT persisted, so the next read tries again."""
    mk = mk or model.get("model_key") or ""
    if not mk:
        return {}
    try:
        fields, state = lookup_physical(mk, model, ASPECT_MARKER)
    except Exception:  # noqa: BLE001
        fields, state = None, "absent"
    if state == "fresh":
        return fields or {}
    try:
        marker = derive_marker(model, mk)
    except Exception:  # noqa: BLE001 — unresolvable: unknown, never a fake absence
        logger.debug("hugpy marker unresolvable for %s", mk, exc_info=True)
        return {}
    if "hugpy_marker" not in marker:
        return {}
    _persist(mk, model, marker, [ASPECT_MARKER], source, None)
    return marker


def refresh_fields(model: dict, mk: Optional[str] = None, *,
                   source: str = "detail") -> dict:
    """Force-refresh: always derive LIVE, rewrite the record, return both halves.

    ``GET /models/<key>`` is this — opening a row IS how an operator forces a
    re-read of a shared, mutable store.

    Returns the STATUS + SIZE halves only. The marker aspect is refreshed and
    rewritten too (it is physical state on the same dir, and this is the force-
    refresh) but is NOT returned: ``/models/<key>`` spreads this dict into its
    response, and the response shape must not change."""
    mk = mk or model.get("model_key") or ""
    status, sizes, mtime = derive_physical(model, mk)
    if mk:
        _persist(mk, model, {**status, **sizes},
                 [ASPECT_STATUS, ASPECT_SIZE], source, mtime)
        try:
            marker = derive_marker(model, mk)
            if "hugpy_marker" in marker:
                _persist(mk, model, marker, [ASPECT_MARKER], source, mtime)
        except Exception:  # noqa: BLE001 — the marker is additive, never a gate
            logger.debug("hugpy marker refresh skipped for %s", mk,
                         exc_info=True)
    return {**status, **sizes}


def stamp_fields(model: dict, fields: dict, aspect: str) -> dict:
    """Stamp ``fields`` onto a registry row IN PLACE and clear the aspect's
    stale leftovers.

    Registry rows are cached and mutated in place, so a key the deriver stopped
    producing (``filename_warning`` after the pinned quant was fixed) would
    otherwise linger forever on the row — the same trap the ``block`` record hit.
    Only keys belonging to THIS aspect are cleared, never the other half's."""
    for field in ASPECT_FIELDS.get(aspect, ()):  # type: ignore[arg-type]
        if field not in fields:
            model.pop(field, None)
    model.update(fields)
    return model


# ──────────────────────────────────────────────────────────────────────────
# repair sweep
# ──────────────────────────────────────────────────────────────────────────
def rebuild_physical(manifest: dict, *, source: str = "discover") -> dict:
    """Re-derive and rewrite the physical state of EVERY model in ``manifest``.

    The repair path for a SHARED, MUTABLE store: another box wrote weights, an
    operator moved a directory, the reaper deleted — none of which fires one of
    our events. ``/models/discover`` runs this in its background thread right
    after the registry re-walk, so the console's next listing is both correct
    and warm.

    Slow BY DESIGN (it is the full walk we removed from the request path) and
    off the request path. Never raises: a model that cannot be derived is
    counted and skipped, leaving its row absent, which means "derive live"."""
    entries, failed = [], 0
    for key, model in (manifest or {}).items():
        if not isinstance(model, dict):
            continue
        try:
            status, sizes, mtime = derive_physical(model, key)
        except Exception:  # noqa: BLE001 — one bad model must not stop the sweep
            logger.warning("physical rebuild failed for %s", key, exc_info=True)
            failed += 1
            continue
        aspects = [ASPECT_STATUS, ASPECT_SIZE]
        fields = {**status, **sizes}
        try:
            marker = derive_marker(model, key)
            if "hugpy_marker" in marker:
                fields.update(marker)
                aspects.append(ASPECT_MARKER)
        except Exception:  # noqa: BLE001 — additive; an unreadable marker just
            logger.debug("hugpy marker unreadable for %s", key, exc_info=True)
        entries.append((key, identity_of(model), fields, aspects, mtime))
    written = 0
    if entries:
        try:
            written = physical_store.put_many(entries, source=source)
        except Exception:  # noqa: BLE001
            logger.warning("physical rebuild could not be persisted",
                           exc_info=True)
    logger.info("physical rebuild: %d model(s) written, %d failed (%s)",
                written, failed, source)
    return {"written": written, "failed": failed}
