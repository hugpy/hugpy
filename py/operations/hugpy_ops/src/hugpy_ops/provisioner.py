"""PROVISIONER (k97, operator ruling 2026-08-06: autonomous downloads are
greenlit — especially for ComfyUI).

Detect DECLARED-BUT-MISSING model weights across the fleet's three registries
and enqueue their downloads on the EXISTING transfer plane. This module is
detection + enqueue + provenance ONLY — there is no download engine here. An
enqueued want becomes a queued job of kind "download" via
``downloader.queue.enqueue_download`` — the exact function the console's
add-models routes (``POST /models/<key>/download`` and
``POST /llm/repos/download`` in llm_storage_routes.py) call — so dedupe,
progress, cancel and retry all work through the same daemon
(hugpy-downloader-dev) and the same /jobs view as a human-clicked download.

The three registries scanned (in fleet-priority order — ComfyUI first-class):

  comfy  — curated ``framework == "comfy"`` rows in models_config.MODELS
           (checkpoint file expected in <root>/checkpoints; the sweep
           ``_sweep_comfy_checkpoints`` only ever REGISTERS what exists, so a
           curated row whose file is absent is exactly the starvation the
           sweep can never repair) + the shared id_lock assets WORKER-SETUP
           §5b places at <root>/ipadapter and <root>/clip_vision.
  studio — video_intel.studio MODEL_REGISTRY rows whose weights dir under the
           studio weights root holds no bytes (the ZERO_BYTE_MODELS /
           "declared intent" starvation presets.py documents — codeformer et
           al — detected from the filesystem, not from the frozen list).
  tasks  — the remaining curated models_config.MODELS rows, checked through
           ``resolve_model_dir`` (the same read-through resolver the download
           engine uses, so a model landed under a legacy layout is never
           re-wanted).

SOURCES ARE NEVER GUESSED. A want is enqueueable only when its hub id +
filename are derivable from the ComfyUI-Manager catalog / the registry row /
an explicit manifest entry in this file; anything else is surfaced as
UNRESOLVED in the dry-run output — visible, not silent — and ``enqueue``
refuses it with a logged reason.

k97b (operator ruling 2026-08-06): "comfy offers downloads to most models —
if it offers that, and the model isn't in llm_storage, take that route."
For comfy-registry wants the resolution order is now
  (a) the running ComfyUI-Manager's model catalog (matched on filename,
      kind-preferred) — its url resolved to HF hub-id form, its save_path
      mapped onto the dirs comfy actually scans;
  (b) the k97 HF derivation from the registry row / §5b manifest;
  (c) UNRESOLVED (a matched non-HF source keeps its url IN THE NOTE — the
      transfer plane speaks HF hub ids only, so civitai/github sources are
      surfaced, never fetched by a side channel).

CLI:  hugpy-provisioner [--apply] [--floor-gb N]
      Default is DRY RUN (prints the want-list). --apply enqueues every
      resolved want, guarded by a free-space floor on the destination volume
      (default: keep >= 500 GB free, HUGPY_PROVISION_FLOOR_GB overrides).
      hugpy-provisioner catalog [--type X] [--missing]
      Browse the ComfyUI-Manager catalog vs local presence.

Registries are read through the owning packages' PUBLIC surfaces —
``hugpy_engine.config.models.models_config.MODELS`` (the curated engine
registry, comfy rows included), ``hugpy_video.intel.studio.MODEL_REGISTRY``
(the studio zoo; importing the package seeds it) and
``hugpy_storage.model_paths.resolve_model_dir`` — and every scan accepts a
:class:`Registries` of fakes so the whole pass (``wants`` / ``provision``) is
testable without a store, a comfy or a queue.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Free space the destination volume must KEEP after a provisioning pass.
DEFAULT_FLOOR_GB = 500.0

REASON_ZERO_BYTE = "0-byte"
REASON_ABSENT = "absent"
REASON_WORKFLOW = "workflow-requires"


def floor_bytes() -> int:
    gb = float(os.environ.get("HUGPY_PROVISION_FLOOR_GB", DEFAULT_FLOOR_GB)
               or DEFAULT_FLOOR_GB)
    return int(gb * 1e9)


# --------------------------------------------------------------------------
# Want — one declared-but-missing weight
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Want:
    registry: str                 # "comfy" | "studio" | "tasks"
    name: str                     # model_key / asset name
    reason: str                   # REASON_* above
    dest: str                     # where the CONSUMER expects the weight
    hub_id: Optional[str] = None  # None => UNRESOLVED, never enqueued
    filename: Optional[str] = None
    include: Optional[list] = None
    framework: str = "transformers"   # drives the transfer plane's layout
    est_bytes: Optional[int] = None
    note: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.hub_id)

    @property
    def fingerprint(self) -> str:
        return "weight_missing:%s:%s" % (self.registry, self.name)

    def to_evidence(self) -> dict:
        """Flat dict form — the sentinel anomaly evidence AND the JSON the
        CLI prints, so every consumer sees the same facts."""
        return {"registry": self.registry, "name": self.name,
                "reason": self.reason, "dest": self.dest,
                "hub_id": self.hub_id, "filename": self.filename,
                "include": self.include, "framework": self.framework,
                "est_bytes": self.est_bytes, "note": self.note,
                "fingerprint": self.fingerprint, "resolved": self.resolved}


def _tree_bytes(path: str) -> int:
    """Total file bytes under path (0 for an absent path). Symlinked files
    count via their target so a linked checkpoint reads as present."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _missing_reason(path: str) -> Optional[str]:
    """REASON_ABSENT / REASON_ZERO_BYTE / None (present with bytes)."""
    if not os.path.exists(path):
        return REASON_ABSENT
    size = _tree_bytes(path) if os.path.isdir(path) else _file_size(path)
    return REASON_ZERO_BYTE if size == 0 else None


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _default_root() -> str:
    from hugpy_platform.constants import DEFAULT_ROOT
    return DEFAULT_ROOT


# --------------------------------------------------------------------------
# ComfyUI — workflow requirements manifest (EXPLICIT, not guessed)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ComfyAsset:
    """One asset a comfy workflow requires, with its EXPLICIT source.

    ``hub_id``/``hub_filename`` are pinned from WORKER-SETUP §5b (the operator
    install recipe the runner's defaults in comfy_runner._IPADAPTER_FILES
    already assume) — cited provenance, not a guess. ``hub_id=None`` marks an
    asset the transfer plane CANNOT provision (e.g. a custom-node pack, which
    is code cloned per box): it appears in the manifest for completeness and
    in dry-run output as UNRESOLVED when relevant, never as an enqueue."""
    kind: str                     # checkpoint | ipadapter | clip_vision | node_pack
    dest_dir: str                 # relative to the store root ("" = live probe only)
    dest_name: Optional[str]      # None => per-registry-row (checkpoint)
    hub_id: Optional[str]
    hub_filename: Optional[str]
    provenance: str
    est_bytes: Optional[int] = None
    note: str = ""


# The shared id_lock assets of WORKER-SETUP §5b. dest names MUST stay in sync
# with comfy_runner._IPADAPTER_FILES (the builder wires these exact filenames
# into the IPAdapter graph).
COMFY_SHARED_ASSETS: dict[str, ComfyAsset] = {
    "ipadapter:sd15": ComfyAsset(
        kind="ipadapter", dest_dir="ipadapter",
        dest_name="ip-adapter_sd15.safetensors",
        hub_id="h94/IP-Adapter",
        hub_filename="models/ip-adapter_sd15.safetensors",
        provenance="WORKER-SETUP §5b"),
    "ipadapter:sdxl": ComfyAsset(
        kind="ipadapter", dest_dir="ipadapter",
        dest_name="ip-adapter_sdxl_vit-h.safetensors",
        hub_id="h94/IP-Adapter",
        hub_filename="sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
        provenance="WORKER-SETUP §5b"),
    "clip_vision:vit-h": ComfyAsset(
        kind="clip_vision", dest_dir="clip_vision",
        dest_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        hub_id="h94/IP-Adapter",
        hub_filename="models/image_encoder/model.safetensors",
        provenance="WORKER-SETUP §5b",
        note="hub file is models/image_encoder/model.safetensors; §5b renames "
             "it on placement to the name comfy_runner defaults to"),
    "node_pack:ipadapter_plus": ComfyAsset(
        kind="node_pack", dest_dir="", dest_name=None,
        hub_id=None, hub_filename=None,
        provenance="WORKER-SETUP §5b",
        note="ComfyUI_IPAdapter_plus is CODE cloned per box, not a weight — "
             "presence is probed live per-comfy (comfy_has_ipadapter), never "
             "detectable from the store and never enqueued"),
}

# workflow -> required asset kinds/names, mirroring exactly what
# comfy_runner builds: _t2i_workflow / _i2i_workflow need the registry row's
# checkpoint; _ipadapter_workflow (id_lock) additionally wires the §5b
# weights (family-matched at request time) and needs the node pack.
COMFY_WORKFLOW_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "t2i": ("checkpoint",),
    "i2i": ("checkpoint",),
    "id_lock": ("checkpoint", "ipadapter:sd15", "ipadapter:sdxl",
                "clip_vision:vit-h", "node_pack:ipadapter_plus"),
}


# --------------------------------------------------------------------------
# ComfyUI-Manager catalog (k97b). The Manager on the running comfy exposes
# its curated catalog at /api/externalmodel/getlist — entries carry name /
# type / base / save_path / filename / url / size and a STRING "installed"
# flag — and /api/experiment/models maps comfy's folder names onto the real
# dirs it scans (our store-root dirs included). Both fetches are memoised
# per-run, success AND failure: a down comfy costs one timeout, not one per
# want, and every consumer falls back to the k97 HF derivation.
# --------------------------------------------------------------------------
_CATALOG_TIMEOUT_S = float(os.environ.get("HUGPY_COMFY_CATALOG_TIMEOUT_S",
                                          "10") or 10)
_UNFETCHED = object()        # per-run memo state: not asked yet
_LIVE = object()             # arg marker: use the (memoised) live fetch
_catalog_cache: Any = _UNFETCHED
_folders_cache: Any = _UNFETCHED


def comfy_base_url() -> str:
    # Same convention as managers.comfy.comfy_runner._comfy_url.
    return (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")


def _fetch_json(url: str, timeout: float = _CATALOG_TIMEOUT_S) -> Any:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def reset_catalog_caches() -> None:
    global _catalog_cache, _folders_cache
    _catalog_cache = _UNFETCHED
    _folders_cache = _UNFETCHED


def manager_catalog() -> Optional[list]:
    """The Manager catalog entries, or None when comfy/Manager is down —
    callers degrade to the k97 HF derivation, never fail the scan."""
    global _catalog_cache
    if _catalog_cache is _UNFETCHED:
        try:
            payload = _fetch_json(comfy_base_url()
                                  + "/api/externalmodel/getlist?mode=cache")
            models = payload.get("models") if isinstance(payload, dict) else None
            _catalog_cache = models if isinstance(models, list) else None
        except Exception:  # noqa: BLE001 — comfy down is an expected state
            logger.info("ComfyUI-Manager catalog unavailable at %s — "
                        "falling back to HF derivation", comfy_base_url())
            _catalog_cache = None
    return _catalog_cache


def comfy_folder_map() -> Optional[dict]:
    """comfy folder name -> the dirs that comfy instance actually scans
    (/api/experiment/models), or None when comfy is down."""
    global _folders_cache
    if _folders_cache is _UNFETCHED:
        try:
            payload = _fetch_json(comfy_base_url() + "/api/experiment/models")
            _folders_cache = ({str(f.get("name")): list(f.get("folders") or [])
                               for f in payload if isinstance(f, dict)}
                              if isinstance(payload, list) else None)
        except Exception:  # noqa: BLE001
            _folders_cache = None
    return _folders_cache


_SIZE_UNITS = {"B": 1.0, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def _parse_size(size: Any) -> Optional[int]:
    """Manager sizes are strings like '4.71MB' / '6.94GB'; None when absent
    or unparseable — an unknown size must not block a resolution."""
    m = re.match(r"^\s*([0-9.]+)\s*([KMGT]?B)\s*$", str(size or ""), re.I)
    if not m:
        return None
    try:
        return int(float(m.group(1)) * _SIZE_UNITS[m.group(2).upper()])
    except (ValueError, KeyError):
        return None


def hub_from_url(url: str) -> Optional[tuple[str, str]]:
    """(hub_id, path-in-repo) for a huggingface.co file URL, else None.

    Accepts the /resolve/<rev>/, /raw/<rev>/ and /blob/<rev>/ forms (the live
    catalog is 489x resolve + one raw). None for every non-HF host — the
    transfer plane speaks HF hub ids only (engine -> snapshot/hf_hub_download),
    so civitai/github sources stay UNRESOLVED with the url noted rather than
    growing a new fetcher here."""
    try:
        parts = urllib.parse.urlsplit(url or "")
    except ValueError:
        return None
    if parts.netloc.lower() not in ("huggingface.co", "www.huggingface.co"):
        return None
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) >= 5 and segs[2] in ("resolve", "raw", "blob"):
        return "/".join(segs[:2]), "/".join(segs[4:])
    return None


def catalog_match(catalog: Optional[list], filename: Optional[str],
                  kind: Optional[str] = None) -> Optional[dict]:
    """Best Manager entry for one asset. Matched on the entry ``filename`` —
    the DESIRED local name (for ~37 entries it differs from the url basename,
    e.g. the §5b clip_vision rename) — and on the url basename as a fallback.
    Entries whose save_path top segment or type equals ``kind`` win over
    cross-kind homonyms (diffusion_pytorch_model.safetensors appears 20x)."""
    if not catalog or not filename:
        return None
    fn = filename.lower()
    hits = []
    for e in catalog:
        if not isinstance(e, dict):
            continue
        efn = str(e.get("filename") or "").lower()
        if efn.startswith("<"):        # "<huggingface>" placeholder rows
            continue
        url_base = os.path.basename(
            urllib.parse.urlsplit(str(e.get("url") or "")).path).lower()
        if efn == fn or url_base == fn:
            hits.append(e)
    if not hits:
        return None
    if kind:
        k = kind.lower()
        exact = [e for e in hits
                 if str(e.get("save_path") or "").split("/")[0].lower() == k
                 or str(e.get("type") or "").lower() == k]
        if exact:
            hits = exact
    return hits[0]


# save_path "default" routes by entry type; comfy folder names it maps to.
_TYPE_FOLDERS = {
    "checkpoint": "checkpoints", "checkpoints": "checkpoints",
    "unclip": "checkpoints", "lora": "loras", "vae": "vae",
    "controlnet": "controlnet", "upscale": "upscale_models",
    "clip": "text_encoders", "clip_vision": "clip_vision",
    "diffusion_model": "diffusion_models", "embedding": "embeddings",
    "embeddings": "embeddings", "taesd": "vae_approx", "gligen": "gligen",
    "ip-adapter": "ipadapter",
}


def _static_comfy_dir(top: str, root: str) -> str:
    """Under-root fallback when the live folder map is silent: mirrors the
    layout the running comfy already scans — checkpoints at the root, the §5b
    ipadapter/clip_vision dirs beside it, everything else comfy-kinds/."""
    if top == "checkpoints":
        return os.path.join(root, "checkpoints")
    if top in ("ipadapter", "clip_vision"):
        return os.path.join(root, top)
    return os.path.join(root, "comfy-kinds", top)


def catalog_dest(entry: dict, root: str,
                 folders: Optional[dict] = None) -> Optional[str]:
    """Where WE place this Manager entry so the running comfy finds it: its
    save_path mapped onto the dirs comfy scans, preferring the one under our
    store root, sub-dirs preserved (comfy's folder scan is recursive).
    None = unmappable -> the caller keeps its own dest."""
    save_path = str(entry.get("save_path") or "").strip().strip("/")
    fname = entry.get("filename") or ""
    if not fname:
        return None
    if not save_path or save_path == "default":
        save_path = _TYPE_FOLDERS.get(str(entry.get("type") or "").lower(), "")
        if not save_path:
            return None
    top, _, sub = save_path.partition("/")
    prefix = root.rstrip("/") + os.sep
    base = next((d for d in (folders or {}).get(top, [])
                 if d.startswith(prefix)), None)
    if base is None:
        base = _static_comfy_dir(top, root)
    return os.path.join(base, sub, fname) if sub else os.path.join(base, fname)


def _catalog_resolution(catalog: Optional[list], folders: Optional[dict],
                        root: str, filename: Optional[str],
                        kind: Optional[str]) -> Optional[dict]:
    """(a) of the k97b resolution order: the Manager catalog's answer for one
    comfy asset. None = no entry (or comfy down) -> (b) the k97 derivation.
    A matched entry whose url is non-HF comes back with hub_id=None + the
    url — the caller surfaces it, never guesses around it."""
    entry = catalog_match(catalog, filename, kind=kind)
    if entry is None:
        return None
    url = str(entry.get("url") or "")
    hub = hub_from_url(url)
    return {"entry": entry, "url": url,
            "hub_id": hub[0] if hub else None,
            "hub_filename": hub[1] if hub else None,
            "dest": catalog_dest(entry, root, folders=folders),
            "est_bytes": _parse_size(entry.get("size")),
            "label": "%s (type %s, save_path %s)"
                     % (entry.get("name"), entry.get("type"),
                        entry.get("save_path"))}


def _manager_note(res: dict, local_name: str) -> str:
    """Provenance note for a Manager-resolved want, including the rename
    caveat when the hub file lands under a different basename."""
    note = ("ComfyUI-Manager catalog: %s — transfer plane lands at "
            "models/misc/%s; link/place the file at dest"
            % (res["label"], res["hub_id"]))
    hub_base = os.path.basename(res["hub_filename"] or "")
    if hub_base and hub_base != local_name:
        note += (" (hub file is %s — rename to %s on placement)"
                 % (res["hub_filename"], local_name))
    return note


def curated_rows() -> dict[str, dict]:
    """The engine's curated registry (``models_config.MODELS``): the
    authoritative base rows, comfy rows included. Read through the engine's
    public module — never a private helper — and only when a scan runs."""
    from hugpy_engine.config.models.models_config import MODELS
    return MODELS


def studio_models() -> dict:
    """The studio zoo (``hugpy_video.intel.studio.MODEL_REGISTRY``). Importing
    the studio package registers ``models_seed``, so the public export is
    populated by construction."""
    from hugpy_video.intel.studio import MODEL_REGISTRY
    return MODEL_REGISTRY


def default_resolver() -> Callable:
    """Storage's read-through model-dir resolver (flat + every legacy layout)
    — the same one the download engine uses."""
    from hugpy_storage.model_paths import resolve_model_dir
    return resolve_model_dir


@dataclass(frozen=True)
class Registries:
    """The three registry reads + the resolver, injectable as a unit.

    ``None`` for any field means "the live one" (the public readers above);
    tests pass plain dicts / callables. ``catalog``/``folders`` default to the
    per-run-memoised ComfyUI-Manager fetch; pass ``None`` for "comfy down".
    """
    curated: Optional[dict] = None          # engine registry rows (comfy + tasks)
    studio: Optional[dict] = None           # studio MODEL_REGISTRY
    resolver: Optional[Callable] = None     # storage resolve_model_dir-shaped
    catalog: Any = _LIVE                    # ComfyUI-Manager catalog entries
    folders: Any = _LIVE                    # comfy folder map
    studio_weights_root: Optional[str] = None

    def curated_rows(self) -> dict:
        return curated_rows() if self.curated is None else self.curated

    def studio_models(self) -> dict:
        return studio_models() if self.studio is None else self.studio

    def resolve(self) -> Callable:
        return default_resolver() if self.resolver is None else self.resolver


LIVE_REGISTRIES = Registries()


def _resolved_comfy_want(name: str, reason: str, dest: str, local_name: str,
                         kind: str, fallback: dict, base_note: str,
                         catalog: Optional[list], folders: Optional[dict],
                         root: str) -> Optional[Want]:
    """One comfy want through the k97b resolution ladder:

      (a) Manager catalog entry with an HF url -> ITS hub id/filename,
          save_path-mapped dest, catalog size estimate;
      (a') Manager entry with a NON-HF url -> the k97 fallback when the row
          proves a hub id (url noted), else UNRESOLVED with the url noted;
      (b) the k97 derivation (registry row / §5b manifest) as before;
      (c) UNRESOLVED as before.

    None = not a want after all: the file already sits at the CATALOG-mapped
    dest (a prior Manager-route placement) — comfy finds it there, so
    re-wanting it forever against the k97 path would be a detection loop.
    """
    res = _catalog_resolution(catalog, folders, root, local_name, kind)
    if (res is not None and res["dest"] and res["dest"] != dest
            and _missing_reason(res["dest"]) is None):
        return None
    if res is not None and res["hub_id"]:
        return Want(
            registry="comfy", name=name, reason=reason,
            dest=res["dest"] or dest, hub_id=res["hub_id"],
            filename=res["hub_filename"], include=[res["hub_filename"]],
            framework="comfy", est_bytes=res["est_bytes"],
            note=_manager_note(res, local_name))
    note = base_note
    if res is not None:            # matched, but a source our plane can't speak
        offer = ("ComfyUI-Manager offers %s via non-HF url %s — the transfer "
                 "plane speaks HF hub ids only" % (res["label"], res["url"]))
        note = (offer + ("; using the registry hub id"
                         if fallback.get("hub_id") else
                         " (url noted for a future fetcher)")
                + ((" — " + base_note) if base_note else ""))
    return Want(
        registry="comfy", name=name, reason=reason, dest=dest,
        hub_id=fallback.get("hub_id") or None,
        filename=fallback.get("filename"), include=fallback.get("include"),
        framework="comfy", est_bytes=fallback.get("est_bytes"), note=note)


def comfy_wants(rows: Optional[dict] = None, root: Optional[str] = None,
                catalog: Any = _LIVE, folders: Any = _LIVE) -> list[Want]:
    """Comfy starvation: curated checkpoint rows whose file is not in
    <root>/checkpoints, plus the §5b shared id_lock assets. The sweep only
    registers files that EXIST, so only curated rows can starve — a swept row
    is present by construction. Sources resolve catalog-first (k97b) —
    ``catalog``/``folders`` default to the per-run-cached live Manager fetch
    and are injectable (None = comfy down) for tests."""
    rows = curated_rows() if rows is None else rows
    root = root or _default_root()
    if catalog is _LIVE:
        catalog = manager_catalog()
    if folders is _LIVE:
        folders = comfy_folder_map()
    ckpt_root = os.path.join(root, "checkpoints")
    out: list[Want] = []
    for key, row in sorted(rows.items()):
        if not isinstance(row, dict) or row.get("framework") != "comfy":
            continue
        fn = row.get("filename")
        if not fn:
            continue
        dest = os.path.join(ckpt_root, fn)
        reason = _missing_reason(dest)
        if reason is None:
            continue
        w = _resolved_comfy_want(
            key, reason, dest, fn, "checkpoints",
            {"hub_id": row.get("hub_id"), "filename": fn,
             "include": row.get("include")},
            "checkpoint for comfy workflows t2i/i2i (registry row)",
            catalog, folders, root)
        if w is not None:
            out.append(w)
    # Shared id_lock assets (workflow-requires): wanted whenever missing —
    # every comfy checkpoint row is an id_lock candidate.
    for asset_key, asset in COMFY_SHARED_ASSETS.items():
        if asset.kind == "node_pack":
            continue        # code, not a weight — see the manifest entry
        dest = os.path.join(root, asset.dest_dir, asset.dest_name)
        reason = _missing_reason(dest)
        if reason is None:
            continue
        w = _resolved_comfy_want(
            "comfy-" + asset_key.replace(":", "-"),
            REASON_WORKFLOW if reason == REASON_ABSENT else reason,
            dest, asset.dest_name, asset.dest_dir,
            {"hub_id": asset.hub_id, "filename": asset.hub_filename,
             "est_bytes": asset.est_bytes},
            "id_lock workflow requirement (%s)%s"
            % (asset.provenance, (" — " + asset.note) if asset.note else ""),
            catalog, folders, root)
        if w is not None:
            out.append(w)
    return out


# --------------------------------------------------------------------------
# studio registry
# --------------------------------------------------------------------------
def studio_weights_root() -> str:
    return (os.environ.get("STUDIO_WEIGHTS_ROOT")
            or os.path.join(_default_root(), "video_intel", "studio",
                            "weights"))


def _studio_hub_source(cfg) -> Optional[str]:
    """The HF hub id for a studio row, ONLY when the row itself proves it:
    weight_uri must be org/name-shaped AND source_url must point at
    huggingface.co. GitHub-sourced rows (codeformer, rife-practical — both
    also flagged verify_uri) come back None => UNRESOLVED, never guessed."""
    uri = (getattr(cfg, "weight_uri", "") or "").strip()
    src = (getattr(cfg, "source_url", "") or "")
    if "://" in uri or uri.count("/") != 1:
        return None
    if "huggingface.co" not in src:
        return None
    return uri


def studio_wants(models: Optional[dict] = None,
                 weights_root: Optional[str] = None) -> list[Want]:
    models = studio_models() if models is None else models
    weights_root = weights_root or studio_weights_root()
    out: list[Want] = []
    for model_id, cfg in sorted(models.items()):
        uri = (getattr(cfg, "weight_uri", "") or "")
        if getattr(cfg, "synthetic", False) or "://" in uri:
            continue        # ffmpeg:// / synthetic:// rows hold no weights
        dest = os.path.join(weights_root, *[p for p in uri.split("/") if p])
        reason = _missing_reason(dest)
        if reason is None:
            continue
        hub = _studio_hub_source(cfg)
        if hub is None:
            note = ("source not derivable: weight_uri=%r source_url=%r "
                    "(not a confirmed HF repo — see the seed row's "
                    "verify_uri/notes)" % (uri, getattr(cfg, "source_url", "")))
        else:
            # The transfer plane lands in ITS layout (models/<runtime>/
            # <org>/<name>), not the studio weights root — the studio
            # runners read only dest, so the landed copy needs one symlink.
            note = ("transfer plane lands at models/transformers/%s; after "
                    "landing, link the studio dest at it: ln -s" % uri)
        out.append(Want(
            registry="studio", name=model_id, reason=reason, dest=dest,
            hub_id=hub, framework="transformers", note=note))
    return out


# --------------------------------------------------------------------------
# tasks registry (curated models_config.MODELS, non-comfy)
# --------------------------------------------------------------------------
def tasks_wants(rows: Optional[dict] = None, root: Optional[str] = None,
                resolver: Optional[Callable] = None) -> list[Want]:
    """Curated non-comfy rows, resolved through the SAME read-through
    resolver the download engine uses (flat + every legacy layout), so a
    model living under an old task path is never re-wanted."""
    rows = curated_rows() if rows is None else rows
    root = root or _default_root()
    if resolver is None:
        resolver = default_resolver()
    out: list[Want] = []
    for key, row in sorted(rows.items()):
        if not isinstance(row, dict) or row.get("framework") == "comfy":
            continue
        if not row.get("hub_id"):
            continue
        complete = None
        try:
            complete = resolver(row, root, require_complete=True)
        except Exception:  # noqa: BLE001 — a resolver hiccup must not hide the row
            logger.warning("tasks scan: resolver failed for %s", key,
                           exc_info=True)
        if complete:
            continue
        # Not complete anywhere: partial vs nothing, via the write target.
        try:
            partial = resolver(row, root, require_complete=False)
        except Exception:  # noqa: BLE001
            partial = None
        dest = partial or os.path.join(root, "models")
        reason = _missing_reason(dest) or REASON_ABSENT
        note = ""
        if reason == REASON_ABSENT and partial and os.path.exists(partial):
            note = ("partial: %d bytes on disk — enqueue resumes, never "
                    "refetches" % _tree_bytes(partial))
        out.append(Want(
            registry="tasks", name=key, reason=reason, dest=dest,
            hub_id=row.get("hub_id"), filename=row.get("filename"),
            include=row.get("include"),
            framework=row.get("framework") or "transformers", note=note))
    return out


def wants(root: Optional[str] = None,
          registries: Optional[Registries] = None) -> list[Want]:
    """Every declared-but-missing weight, ComfyUI first-class. Each registry
    scan is independent — one failing (e.g. studio env absent on a worker
    box) is logged and skipped, never hides the others. ``registries``
    injects the three registry reads (+ resolver, catalog, folders) for
    tests; None = the live public readers."""
    reg = LIVE_REGISTRIES if registries is None else registries
    out: list[Want] = []
    scans = (
        lambda: comfy_wants(rows=reg.curated, root=root,
                            catalog=reg.catalog, folders=reg.folders),
        lambda: studio_wants(models=reg.studio,
                             weights_root=reg.studio_weights_root),
        lambda: tasks_wants(rows=reg.curated, root=root,
                            resolver=reg.resolver),
    )
    for scan in scans:
        try:
            out.extend(scan())
        except Exception:  # noqa: BLE001 — one registry down must not blind the rest
            logger.warning("provisioner scan failed", exc_info=True)
    return out


# --------------------------------------------------------------------------
# enqueue — rides the existing transfer plane
# --------------------------------------------------------------------------
def _live_download_jobs() -> list[dict]:
    from hugpy_control.jobs import job_store, normalize_status, TERMINAL_STATUSES
    from hugpy_storage.downloader.engine import DOWNLOAD_KIND
    rows = job_store.snapshot(kinds={DOWNLOAD_KIND}, live_only=False,
                              terminal_kinds=(DOWNLOAD_KIND,))
    return [d for d in rows
            if normalize_status(d.get("status")) not in TERMINAL_STATUSES]


def _is_duplicate(want: Want, jobs: list[dict]) -> bool:
    for d in jobs:
        if d.get("model_key") == want.name:
            return True
        model = (d.get("payload") or {}).get("model") or {}
        if (model.get("hub_id") == want.hub_id
                and (model.get("filename") or None) == (want.filename or None)):
            return True
    return False


def enqueue(want: Want, *, existing_jobs: Optional[list] = None,
            free_bytes: Optional[int] = None,
            floor: Optional[int] = None,
            enqueue_fn: Optional[Callable] = None) -> dict:
    """Enqueue one want on the existing download queue — the same
    ``enqueue_download`` the console's add-models routes call, so the daemon,
    dedupe, /jobs progress, cancel and retry all apply unchanged.

    Refusals are RETURNED (and logged), never silent:
      unresolved-source — the registry could not prove a hub id (never guess)
      duplicate         — a live download job already covers this model
      disk-floor        — enqueueing would eat into the free-space floor
    """
    if not want.resolved:
        logger.warning("provisioner: REFUSING %s — source cannot be derived "
                       "(%s)", want.name, want.note or "no hub id")
        return {"enqueued": False, "reason": "unresolved-source",
                "want": want.name}
    jobs = _live_download_jobs() if existing_jobs is None else existing_jobs
    if _is_duplicate(want, jobs):
        logger.info("provisioner: %s already queued/running — not re-enqueued",
                    want.name)
        return {"enqueued": False, "reason": "duplicate", "want": want.name}
    if free_bytes is None:
        free_bytes = shutil.disk_usage(_default_root()).free
    floor = floor_bytes() if floor is None else floor
    if free_bytes - (want.est_bytes or 0) < floor:
        logger.warning("provisioner: REFUSING %s — destination volume would "
                       "drop under the %.0f GB free floor (%.0f GB free now)",
                       want.name, floor / 1e9, free_bytes / 1e9)
        return {"enqueued": False, "reason": "disk-floor", "want": want.name}
    if enqueue_fn is None:
        from hugpy_storage.downloader.queue import enqueue_download
        enqueue_fn = enqueue_download
    model = {"name": want.name, "hub_id": want.hub_id,
             "framework": want.framework, "filename": want.filename,
             "include": want.include}
    job = enqueue_fn(want.name, model, total_bytes=want.est_bytes,
                     transport="provisioner")
    logger.info("provisioner: enqueued download %s for %s (%s)",
                getattr(job, "id", None), want.name, want.hub_id)
    return {"enqueued": True, "job_id": getattr(job, "id", None),
            "want": want.name}


# --------------------------------------------------------------------------
# catalog gems surface (k97b): the Manager's 538-entry curated catalog vs
# what this box actually holds, so the operator can browse and pick.
# --------------------------------------------------------------------------
def _local_comfy_filenames(root: str, folders: Optional[dict] = None) -> set:
    """Every weight filename under our comfy dirs — the store staples plus
    whatever under-root dirs the running comfy says it scans. Basename set:
    comfy resolves models by filename across its folders, so a file present
    under ANY scanned dir counts as present."""
    prefix = root.rstrip("/") + os.sep
    dirs = {os.path.join(root, "checkpoints"), os.path.join(root, "ipadapter"),
            os.path.join(root, "clip_vision"), os.path.join(root, "comfy-kinds")}
    for folder_list in (folders or {}).values():
        dirs.update(d for d in folder_list if d.startswith(prefix))
    names: set = set()
    for d in dirs:
        for _r, _dd, files in os.walk(d):
            names.update(files)
    return names


def catalog_report(entries: Any = _LIVE, folders: Any = _LIVE,
                   root: Optional[str] = None,
                   type_filter: Optional[str] = None,
                   missing_only: bool = False, out=sys.stdout) -> dict:
    """List Manager catalog entries against local presence. Present = the
    Manager's own installed flag (a STRING "True"/"False" on the wire), OR
    the save_path-mapped dest exists, OR the filename is anywhere under our
    comfy dirs."""
    root = root or _default_root()
    if entries is _LIVE:
        entries = manager_catalog()
    if folders is _LIVE:
        folders = comfy_folder_map()
    if entries is None:
        print("ComfyUI-Manager catalog unavailable at %s" % comfy_base_url(),
              file=out)
        return {"available": False, "total": 0, "present": 0, "absent": 0,
                "rows": []}
    local = _local_comfy_filenames(root, folders=folders)
    rows = []
    total = present_n = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        etype = str(e.get("type") or "")
        if type_filter and etype.lower() != type_filter.lower():
            continue
        total += 1
        fname = str(e.get("filename") or "")
        dest = catalog_dest(e, root, folders=folders)
        present = (str(e.get("installed")).lower() == "true"
                   or bool(dest and os.path.exists(dest))
                   or (fname in local if fname else False))
        present_n += 1 if present else 0
        if missing_only and present:
            continue
        host = urllib.parse.urlsplit(str(e.get("url") or "")).netloc
        rows.append({"name": e.get("name"), "type": etype,
                     "base": e.get("base"), "size": e.get("size"),
                     "host": host, "filename": fname, "present": present})
    for r in rows:
        print("  [%s] %s | %s | %s | %s | %s"
              % ("present" if r["present"] else "absent ", r["name"],
                 r["type"], r["base"] or "-", r["size"] or "-",
                 r["host"] or "-"), file=out)
    print("catalog: %d entr%s%s — %d present, %d absent"
          % (total, "y" if total == 1 else "ies",
             (" (type=%s)" % type_filter) if type_filter else "",
             present_n, total - present_n), file=out)
    return {"available": True, "total": total, "present": present_n,
            "absent": total - present_n, "rows": rows}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def provision(apply: bool = False, root: Optional[str] = None,
              out=sys.stdout, *, registries: Optional[Registries] = None,
              existing_jobs: Optional[list] = None,
              free_bytes: Optional[int] = None,
              enqueue_fn: Optional[Callable] = None) -> dict:
    """One pass: print the want-list; enqueue resolved wants when apply.

    ``registries`` injects the registry reads; ``existing_jobs`` /
    ``free_bytes`` / ``enqueue_fn`` inject the queue mirror, the disk probe
    and the storage enqueue function (see :func:`enqueue`). All default to
    the live public APIs."""
    found = wants(root=root, registries=registries)
    resolved = [w for w in found if w.resolved]
    unresolved = [w for w in found if not w.resolved]
    print("provisioner: %d want(s) — %d resolved, %d UNRESOLVED%s"
          % (len(found), len(resolved), len(unresolved),
             "" if apply else "  [DRY RUN — nothing enqueued]"), file=out)
    for w in found:
        src = ("%s%s" % (w.hub_id, (" :: " + w.filename) if w.filename else "")
               if w.resolved else "UNRESOLVED")
        print("  [%s/%s] %s <- %s\n      dest: %s%s"
              % (w.registry, w.reason, w.name, src, w.dest,
                 ("\n      note: " + w.note) if w.note else ""), file=out)
    results = []
    if apply:
        jobs = existing_jobs
        if jobs is None:
            try:
                jobs = _live_download_jobs()
            except Exception:  # noqa: BLE001 — no mirror = no dedupe source; enqueue decides
                jobs = []
        jobs = list(jobs)
        for w in resolved:
            res = enqueue(w, existing_jobs=jobs, free_bytes=free_bytes,
                          enqueue_fn=enqueue_fn)
            results.append(res)
            if res.get("enqueued"):
                # Two registries naming the same hub file must not
                # double-enqueue within one pass.
                jobs.append({"model_key": w.name, "status": "pending",
                             "payload": {"model": {"hub_id": w.hub_id,
                                                   "filename": w.filename}}})
            print("  -> %s: %s" % (w.name,
                  ("job %s" % res["job_id"]) if res["enqueued"]
                  else "refused (%s)" % res["reason"]), file=out)
    return {"wants": [w.to_evidence() for w in found],
            "applied": apply, "results": results}


def _catalog_main(argv: list) -> int:
    parser = argparse.ArgumentParser(
        prog="hugpy-provisioner catalog",
        description="Browse the ComfyUI-Manager model catalog vs local "
                    "presence (installed flag + our own dir scan).")
    parser.add_argument("--type", dest="type_filter", default=None,
                        help="only entries of this catalog type "
                             "(checkpoint, lora, controlnet, VAE, ...)")
    parser.add_argument("--missing", action="store_true",
                        help="only entries absent on this box")
    parser.add_argument("--root", default=None,
                        help="store root to scan (default: the store root)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="emit the result as JSON")
    args = parser.parse_args(argv)
    result = catalog_report(root=args.root, type_filter=args.type_filter,
                            missing_only=args.missing,
                            out=(open(os.devnull, "w") if args.as_json
                                 else sys.stdout))
    if args.as_json:
        print(json.dumps(result, indent=2))
    return 0


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["catalog"]:
        return _catalog_main(argv[1:])
    parser = argparse.ArgumentParser(
        epilog="subcommand: `hugpy-provisioner catalog [--type X] [--missing]` "
               "browses the ComfyUI-Manager catalog vs local presence.",
        prog="hugpy-provisioner",
        description="Detect declared-but-missing model weights and enqueue "
                    "their downloads on the existing transfer plane.")
    parser.add_argument("--apply", action="store_true",
                        help="enqueue resolved wants (default: dry run)")
    parser.add_argument("--floor-gb", type=float, default=None,
                        help="free-space floor on the destination volume "
                             "(default %s GB / HUGPY_PROVISION_FLOOR_GB)"
                             % DEFAULT_FLOOR_GB)
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="emit the result as JSON")
    args = parser.parse_args(argv)
    if args.floor_gb is not None:
        os.environ["HUGPY_PROVISION_FLOOR_GB"] = str(args.floor_gb)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stderr)
    result = provision(apply=args.apply,
                       out=(open(os.devnull, "w") if args.as_json
                            else sys.stdout))
    if args.as_json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
