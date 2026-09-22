import os
import re
from typing import Any
from abstract_essentials import join_path
from hugpy_storage.hugpy_marker import get_model_home, read_hugpy_marker
from hugpy_platform.constants import (
    DEFAULT_ROOT,
    EXCLUDE_DIR_NAMES,
    EXCLUDE_DIR_PREFIXES,
    HUGPY_MARKER,
)
def safe_path_part(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"/+", "/", value)
    return value.strip("/")
def safe_name(value: str) -> str:
    value = value.strip()
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_/")
def split_hub_id(hub_id: str) -> tuple[str, str | None]:
    parts = hub_id.strip("/").split("/")
    if len(parts) <= 2:
        return hub_id, None
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    return repo_id, subfolder
def runtime_folder(framework: str, hub_id: str, include: Any = None, filename: str | None = None) -> str:
    framework = framework.lower().strip()

    if framework == "gguf":
        return "gguf"

    if filename and filename.lower().endswith(".gguf"):
        return "gguf"

    if include:
        patterns = include if isinstance(include, list) else [include]
        if any("gguf" in pattern.lower() for pattern in patterns):
            return "gguf"

    if framework == "transformers":
        return "transformers"

    return "misc"

# ---------------------------------------------------------------------------
# Model Paths — One row of everything we know. All Optional — partial fills are valid.
# ---------------------------------------------------------------------------
def is_model_dir(directory: str) -> bool:
    """A model dir declares itself with a hugpy.json. Fall back to weight
    markers for dirs not yet stamped (legacy / first-run before backfill).

    ``model_index.json`` marks a diffusers PIPELINE root — the model itself,
    whose weights live in per-component subdirs (text_encoder/, transformer/,
    vae/, …). It MUST count as a model dir so the walk stops here (a leaf) and
    never descends to register those components as phantom standalone models
    (the bare `text_encoder`/`transformer` rows). Without this, an unstamped
    pipeline (freshly downloaded, before its marker is backfilled) has no
    top-level config.json/weights, so the walk fell through into its parts.

    The same phantom-leaf rule applies to GGUF repos whose shards sit in a
    quant SUBFOLDER (e.g. Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M/
    *-0000N-of-0000M.gguf, mirroring the hub layout): the repo root has no
    top-level weights, so without the immediate-subdir check the walk
    descended and registered the quant dir as a standalone model — an
    unloaded, unprotected alias of the live model that the reaper then
    proposed evicting (the ae coder-next reap loop, 2026-09-02)."""
    if os.path.isfile(os.path.join(directory, HUGPY_MARKER)):
        return True
    try:
        entries = os.listdir(directory)
    except (OSError, NotADirectoryError):
        return False
    if any(e == "config.json" or e == "model_index.json"
           or e.lower().endswith((".gguf", ".safetensors"))
           for e in entries):
        return True
    # No top-level weights: a dir whose immediate subdir holds .gguf shards is
    # the model root (quant-subfolder layout) — a leaf, same as model_index.json.
    #
    # NEVER at ORG level. In the flat layout (models/<runtime>/<owner>/<repo>)
    # an OWNER dir whose child repo keeps its ggufs at repo top level matches
    # this exact shape too, and stopping there swallowed the whole org into one
    # phantom row named after the org — gguf/ponpoke ate the flux2-klein text-
    # encoder's catalog row (2026-09-10), and every 'gguf/<org>' hub-404 spam
    # row is the same swallow. At exactly two segments below the store root the
    # dir is an owner, not a model: keep walking.
    #
    # The same rule one level deeper for the LEGACY task layout
    # (models/<runtime>/<task>/<owner>/<repo>, the shape legacy_task_dirs()
    # globs): there the OWNER sits at depth 3 (gguf/text-generation/Qwen), and
    # a repo beneath it keeping its ggufs at top level matched the subdir
    # probe too — the orphan scan then reported the owner dir as a stale-dir
    # phantom. Depth 3 is an owner ONLY when the middle segment is a task
    # name; a flat-layout REPO at depth 3 (gguf/<owner>/<repo> holding quant
    # subfolders) must still be the leaf (the 2026-09-02 quant-subdir case).
    try:
        _root = os.path.realpath(get_model_home())
        _rel = os.path.realpath(directory)[len(_root):].strip(os.sep)
        _parts = [p for p in _rel.split(os.sep) if p]
        if _parts and _parts[0] in RUNTIME_FAMILIES:
            if len(_parts) == 2:
                return False
            if len(_parts) == 3 and _looks_like_task_dir(_parts[1]):
                return False
    except Exception:  # noqa: BLE001 — a depth probe must never break the walk
        pass
    for e in entries:
        sub = os.path.join(directory, e)
        try:
            if os.path.isdir(sub) and any(
                    f.lower().endswith(".gguf") for f in os.listdir(sub)):
                return True
        except OSError:
            continue
    return False

# The task segment of the LEGACY layout. A fixed vocabulary (the HF pipeline
# tags this store has routed under, plus the single-word buckets) and, as a
# backstop for a tag not listed here, a hyphenated lowercase name carrying a
# task word. An OWNER like ``black-forest-labs`` is hyphenated too, so the
# hyphen alone is never enough: ``labs`` is not a task word.
_TASK_DIR_NAMES = frozenset((
    "automatic-speech-recognition", "depth-estimation", "feature-extraction",
    "image-classification", "image-segmentation", "image-text-to-text",
    "image-to-image", "image-to-video", "keyword-extraction",
    "object-detection", "sentence-similarity", "text-generation",
    "text-summarization", "text-to-image", "text-to-speech", "text-to-video",
    "text-classification", "question-answering", "fill-mask",
    "token-classification", "zero-shot-classification", "image-to-text",
    "text-to-audio", "audio-classification", "video-classification",
    "misc", "dataset", "summarization", "translation", "conversational",
    "embeddings", "embedding", "keywords", "vision", "tts", "comfy",
))
_TASK_WORDS = frozenset((
    "generation", "to", "recognition", "classification", "extraction",
    "estimation", "detection", "segmentation", "similarity", "summarization",
    "embedding", "answering", "translation", "mask",
))


def _looks_like_task_dir(name: str) -> bool:
    """Is ``name`` a task segment of the legacy ``<runtime>/<task>/<owner>``
    layout (never an owner/repo name)?"""
    n = str(name or "").strip().lower()
    if not n or n != str(name):
        return False                     # tasks are lowercase; owners often aren't
    if n in _TASK_DIR_NAMES:
        return True
    if "-" not in n or not re.fullmatch(r"[a-z]+(-[a-z]+)+", n):
        return False
    return any(w in _TASK_WORDS for w in n.split("-"))


def hub_id_for(directory: str, fallback: str | None = None) -> str | None:
    """Repo id from the declared marker; path slice only if unstamped."""
    marker = read_hugpy_marker(directory)
    if marker and marker.get("hub_id"):
        return marker["hub_id"]
    return fallback


def get_model_dirs(models_home=None):
    """Walk MODELS_HOME and return every directory that actually holds a model,
    regardless of how deep it sits. Skips excluded dirs and never lists files."""
    root = get_model_home(models_home=models_home)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded subtrees in-place so os.walk doesn't descend them
        dirnames[:] = [d for d in dirnames
                       if not is_directory_excluded(join_path(dirpath, d))]
        if is_model_dir(dirpath):
            found.append(dirpath)
            dirnames[:] = []          # don't descend into a model's own files
    return found

def get_hub_id_from_directory(directory, models_home=None):
    """Best-effort repo id (``owner/repo[/subfolder]``) for a dir under the
    model store, marker-first.

    ``hugpy.json`` (when present) names the repo authoritatively — read it
    first via :func:`hub_id_for`, exactly like discovery does. Only a dir with
    no marker falls back to a PATH GUESS, and that guess must match the
    CURRENT layout: the store is flat (operator-locked 2026-07-11) —
    ``models/<runtime>/<owner>/<repo>[/<subfolder>]`` — a 3+-segment relative
    path with exactly ONE leading runtime segment, not the old
    ``family/task/owner/repo`` shape this used to assume. Unconditionally
    dropping 2 leading segments on a 3-segment flat path stripped the OWNER
    too, collapsing every hub id to a bare repo name (the 2026-07-17 orphan
    over-report root cause — see worker_agent/provision.py's
    known_model_dir_forms, which no longer trusts this alone for matching but
    this must still return the truest guess it can for display/lookup).
    Legacy ``runtime/task/owner/repo`` dirs (task an unknown-length single
    segment) still parse wrong here by path alone — that's inherent to a
    positional guess with no task vocabulary; callers needing correctness
    across legacy layouts must go through the read-through resolver
    (``candidate_model_dirs``/``resolve_model_dir``), not this function.
    """
    marker = hub_id_for(directory)
    if marker:
        return marker
    root = get_model_home(models_home=models_home)
    rel = directory[len(root):].strip("/")
    parts = rel.split("/")
    if len(parts) <= 1:
        return rel
    # Flat layout: exactly 1 leading runtime segment (gguf/transformers/misc/
    # safetensors) -> drop only that one, keep owner/repo[/subfolder] whole.
    if parts[0] in RUNTIME_FAMILIES and len(parts) >= 3:
        return "/".join(parts[1:])
    # Unknown/legacy shape: fall back to the old best-effort guess (drop 2)
    # rather than assume flat, so a genuine 4-segment legacy dir isn't made
    # worse; still layout-guessing, so callers should prefer the resolver.
    return "/".join(parts[2:]) if len(parts) > 2 else rel

def is_directory_excluded(directory):
    base = os.path.basename(directory.rstrip("/"))
    if base in EXCLUDE_DIR_NAMES:
        return True
    # Downloader staging (<Repo>.tmp-<pid>) is scaffolding mid-transfer, not a
    # model: counting it made the console's model total wobble with every
    # download start/finish and even minted download jobs for '<repo>.tmp-<pid>'
    # hub ids (2026-09-10). The rename into place is what makes it visible.
    if ".tmp-" in base:
        return True
    return any(base.startswith(p) for p in EXCLUDE_DIR_PREFIXES)

def exclude_dirs(directories):
    return [d for d in directories if not is_directory_excluded(d)]
def resolve_task(model: dict) -> str:
    """Effective task across all dict shapes:
       ModelConfig.to_dict() -> primary_task (+ tasks list)
       /repos/download dict   -> primary_task
       legacy manifest        -> task (singular)
    First non-empty wins; misc is the floor."""
    pt = model.get("primary_task")
    if pt:
        return pt
    tasks = model.get("tasks")
    if tasks:
        return tasks[0] if isinstance(tasks, list) else tasks
    return model.get("task") or "misc"
def _existing_sibling_task_dir(root: str, runtime: str, hub_path: str) -> str | None:
    """Same runtime + hub_id under a DIFFERENT task folder — where a model's files
    physically landed at download time, which can diverge from a later
    content-corrected task (e.g. a text gguf mis-routed under image-text-to-text,
    then re-derived to text-generation once we saw it has no mmproj). Returns the
    first existing such dir, else None."""
    base = os.path.join(root, "models", runtime)
    try:
        entries = os.listdir(base)
    except OSError:
        return None
    for task_dir in entries:
        cand = os.path.join(base, task_dir, hub_path)
        if os.path.isdir(cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# Storage layout — operator-locked 2026-07-11: FLAT.
#
#   models/<runtime>/<owner>/<repo>        (datasets keep datasets/<owner>/<repo>)
#
# The task segment DIED. It baked derived, mutable, PLURAL metadata (a model
# advertises many tasks) into an IMMUTABLE path — which produced task-twin dirs
# (the same repo under text-generation AND image-text-to-text), sticky wrong-task
# discovery, empty re-routed dirs with the weights stranded in legacy misc/, and
# the "redownload models that are ready" complaints. Task/framework now live
# ONLY in the registry + the per-dir hugpy.json marker. route_destination() emits
# the flat path for ALL new work; resolve_model_dir() reads THROUGH every
# historical layout so a model already on disk under an old task path is never
# re-downloaded, mis-flagged, or 404'd — during OR after the migration.
# ---------------------------------------------------------------------------

# Runtime families = the top storage segment. "misc" is the catch-all runtime
# (comfy + any odd loader); "safetensors" is a historical family kept for reads.
RUNTIME_FAMILIES = ("gguf", "transformers", "misc", "safetensors")


def _hub_path_of(model: dict) -> str:
    return safe_path_part(
        model.get("hub_id") or model.get("name") or model.get("folder") or "")


def flat_destination(model: dict, root: str = DEFAULT_ROOT) -> str:
    """The FLAT write target — models/<runtime>/<owner>/<repo> — where ALL new
    downloads land. No task segment. Single source of truth for the new layout;
    datasets keep their own top-level home (unchanged)."""
    hub_path = _hub_path_of(model)
    if resolve_task(model) == "dataset":
        return os.path.join(root, "datasets", hub_path)
    runtime = runtime_folder(model.get("framework") or "", hub_path,
                             include=model.get("include"),
                             filename=model.get("filename"))
    return os.path.join(root, "models", runtime, hub_path)


def _routing_as_cfg(model: dict):
    """A minimal cfg shim for model_looks_downloaded from a bare routing dict
    (which carries no ModelConfig). Only the fields the completeness gate reads
    matter: framework (the gguf branch), filename/include (pin + vision), and
    primary_task/tasks (the vision-needs-mmproj gate)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        framework=model.get("framework"),
        filename=model.get("filename"),
        include=model.get("include"),
        primary_task=model.get("primary_task") or model.get("task"),
        tasks=model.get("tasks"),
    )


def legacy_task_dirs(hub_path: str, runtime: str, root: str = DEFAULT_ROOT) -> list:
    """Every task-based legacy dir on disk holding this repo under ``runtime``:
    models/<runtime>/<task>/<owner>/<repo>. The task segment is globbed, so this
    finds task-twins (the same repo under several task folders) without knowing
    the task set in advance. Sorted for a deterministic order."""
    import glob as _glob
    pattern = os.path.join(root, "models", runtime, "*", hub_path)
    return sorted(d for d in _glob.glob(pattern) if os.path.isdir(d))


def candidate_model_dirs(model: dict, root: str = DEFAULT_ROOT) -> list:
    """ORDERED list of every dir this model's files might occupy, best-layout
    first — the read-through search order AND the reconcile survey set:

      1. the entry's recorded on-disk dir (discovery ground truth): ``dir``,
         then MODELS_HOME/``folder``;
      2. the FLAT path models/<runtime>/<owner>/<repo>;
      3. legacy task dirs models/<runtime>/<task>/<owner>/<repo> — the model's
         advertised primary_task first, then the rest, deterministically;
      4. the same across the OTHER runtime families (a repo mis-filed under a
         different family, or a text gguf vs. its vision twin).

    De-duplicated, order-preserving. Dirs need NOT exist — callers test."""
    hub_path = _hub_path_of(model)
    task = resolve_task(model)
    out: list = []
    seen: set = set()

    def _add(d):
        if d and d not in seen:
            seen.add(d)
            out.append(d)

    if task == "dataset":
        _add(os.path.join(root, "datasets", hub_path))
        return out

    rec = model.get("dir")
    if rec:
        _add(rec)
    folder = model.get("folder")
    if folder:
        _add(folder if os.path.isabs(folder)
             else os.path.join(root, "models", folder))

    primary_runtime = runtime_folder(model.get("framework") or "", hub_path,
                                     include=model.get("include"),
                                     filename=model.get("filename"))
    families = [primary_runtime] + [f for f in RUNTIME_FAMILIES
                                    if f != primary_runtime]
    for fam in families:
        _add(os.path.join(root, "models", fam, hub_path))        # flat
        legacy = legacy_task_dirs(hub_path, fam, root)           # task-based
        legacy.sort(key=lambda d: (0 if os.sep + task + os.sep in d else 1, d))
        for d in legacy:
            _add(d)
    return out


def _is_archived_path(path: str) -> bool:
    """True when ``path`` sits under an ``_archive`` component — reconcile's
    archive/de-dupe area (``<root>/models/_archive/...`` and
    ``<root>/models/_archive/dedupes/...``), where the weight file is often a
    SYMLINK back into /checkpoints and NOT a live serve/transfer source.

    Component-matched (os.sep-aware), never a substring test — a repo
    legitimately named e.g. ``foo_archive_bar`` must NOT false-positive. We
    split on os.sep and require a component that is EXACTLY ``_archive``."""
    if not path:
        return False
    return "_archive" in path.replace("\\", "/").split("/")


def _safe_complete(directory: str, cfg) -> bool:
    """``model_looks_downloaded`` (storage's completeness rule), guarded so a
    probe never raises into a resolve. Falls back to "exists and non-empty"
    only if the presence module itself cannot be imported."""
    try:
        from hugpy_storage.model_presence import model_looks_downloaded
        return bool(model_looks_downloaded(directory, cfg))
    except Exception:
        pass
    try:
        return os.path.isdir(directory) and bool(os.listdir(directory))
    except OSError:
        return False


def resolve_model_dir(model: dict, root: str = DEFAULT_ROOT, cfg=None,
                      require_complete: bool = True):
    """Read-through resolver — the ONE place that turns a routing/config into the
    real on-disk dir, checking the FLAT layout first, then EVERY legacy layout
    (see candidate_model_dirs). Returns the first dir that passes
    ``model_looks_downloaded``. This is the guarantee that a model downloaded
    under an OLD task path is never re-downloaded or 404'd during (or after) the
    migration. Loaders/provisioners route through it.

      require_complete=True  -> first COMPLETE dir, else None.
      require_complete=False -> first COMPLETE dir, else the first EXISTING dir
                                (so resume/delete/status act on the real partial
                                files, never orphaning them), else the flat write
                                target for a genuinely-new download.

    ARCHIVE EXCLUSION: candidate_model_dirs() is also the reconcile survey set,
    so it legitimately SURFACES dirs under <root>/models/_archive/... (reconcile's
    archive/de-dupe area — weight files there are often SYMLINKs back into
    /checkpoints, not a live copy). A dir under _archive is NEVER a valid live
    serve/transfer source, so it is excluded from selection here entirely —
    both the "complete" and "existing" passes only ever consider non-archive
    candidates. If literally the only thing on disk is an archive copy, we do
    NOT hand it out just because "something" exists: we fall all the way
    through to flat_destination() (the real, safe write target) instead —
    exactly like a genuinely-new download with nothing on disk yet. reconcile
    is expected to notice the stranded archive-only case and re-materialize the
    live copy; this resolver must never paper over that by pointing a
    load/transfer at reconcile's own archive/de-dupe area."""
    _cfg = cfg if cfg is not None else _routing_as_cfg(model)
    cands = candidate_model_dirs(model, root)
    live_cands = [d for d in cands if not _is_archived_path(d)]

    for d in live_cands:
        if os.path.isdir(d) and _safe_complete(d, _cfg):
            return d
    if require_complete:
        return None
    for d in live_cands:
        if os.path.isdir(d):
            return d
    return flat_destination(model, root)


def route_destination(model: dict, root: str = DEFAULT_ROOT) -> str:
    """THE single path chokepoint. Reads resolve through every historical layout
    (flat + legacy task dirs + misc/other families); a genuinely-new download
    with nothing on disk gets the FLAT target models/<runtime>/<owner>/<repo>.
    Kept single-positional-arg compatible (root optional) — every call site and
    the worker-side re-export depend on that shape."""
    return resolve_model_dir(model, root, require_complete=False)
