#!/usr/bin/env python3
import argparse
import json
import logging
import os
from typing import Any
from abstract_essentials import safe_dump_to_file, safe_load_from_json
from hugpy_platform.constants import DEFAULT_ROOT, MODELS_DISCOVERY_PATH, MODELS_HOME
from hugpy_storage.catalog_source import catalog_get, catalog_rows
from hugpy_storage.events import publish_catalog_changed
from hugpy_storage.providers import serve_path
from hugpy_storage.hugpy_marker import write_hugpy_marker
from hugpy_storage.model_paths import (
    flat_destination,
    resolve_model_dir,
    route_destination,
    split_hub_id,
)
import threading
import shutil as _shutil
import time as _time
import re as _re

logger = logging.getLogger("hugpy_storage.download_models")
_REPORT_LOCK = threading.Lock()


def _hf():
    """``huggingface_hub`` on first use — the transport is a heavy optional
    stack and importing this module (the daemon's queue helpers, the marker
    stamper, the discovery-report writer) must not require it."""
    import huggingface_hub
    return huggingface_hub


# ---------------------------------------------------------------------------
# Atomic provisioning (operator-locked 2026-07-11): a download lands in a
# per-pid STAGING sibling and is only renamed onto its final path once complete,
# so a partially-transferred dir can NEVER sit at a resolvable model path (the
# "redownload / phantom-partial" class of bug). Temp cleanup is EXEMPT from the
# never-delete rule — a staging temp is our own in-flight scratch, never catalog
# data.
# ---------------------------------------------------------------------------
def _staging_dir(dest: str) -> str:
    """The atomic staging path for a download: a per-pid sibling of the final
    dest. Same-pid retries reuse it (HF resume); a crashed pid leaves a
    ``.tmp-<pid>`` the next run discards."""
    return f"{dest}.tmp-{os.getpid()}"


def _merge_tree(src: str, dst: str) -> None:
    """Recursively merge directory ``src`` INTO existing directory ``dst`` on
    the SAME filesystem, at every depth.

    The old single-level merge did ``os.replace(src/child, dst/child)`` for a
    dir-onto-dir collision — but ``os.replace`` onto an EXISTING NON-EMPTY
    directory raises ``OSError [Errno 39] Directory not empty`` (ENOTEMPTY).
    HF Hub leaves a populated ``.cache/huggingface/download/`` inside BOTH the
    staged copy and any prior-partial dest, so that collision tripped ENOTEMPTY
    and wedged the finalize forever (Hunyuan3D-2mv/-2mini). This walks instead:

      * both sides are directories  -> recurse (never replace a dir whole);
      * anything else (src is a file, or dst is absent) -> ``os.replace`` is an
        atomic same-fs rename that overwrites a file / lands into an absent
        slot — never touches ENOTEMPTY;

    then removes the now-emptied ``src`` dir. os.replace stays instant because
    src and dst are the same store filesystem.

    NEVER-DELETE / overwrite judgment: on a leaf collision the STAGED (just
    downloaded, complete) copy wins via os.replace — that IS the resume/re-pull
    intent (finish an interrupted pull with the fresh, complete bytes). The
    only files that realistically collide here are HF-scratch metadata under
    ``.cache/huggingface`` (regenerable, not catalog data) and, in a re-pull,
    a model's own weight files being replaced by their freshly-downloaded
    identical selves. We do NOT archive the pre-existing leaf: it is either
    regenerable HF scratch or the same weight re-fetched. Real catalog data is
    never reached by this path — a COMPLETE model short-circuits download long
    before promote (resolve_model_dir read-through), so ``dest`` here is only
    ever a PARTIAL prior attempt, not a finished model.
    """
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        s_is_dir = os.path.isdir(s) and not os.path.islink(s)
        d_is_dir = os.path.isdir(d) and not os.path.islink(d)
        if s_is_dir and d_is_dir:
            _merge_tree(s, d)          # both dirs -> recurse (avoid ENOTEMPTY)
            os.rmdir(s)                # src subtree drained -> drop the shell
        else:
            # File-over-file / into-absent is a plain atomic os.replace. The
            # only mismatch that os.replace itself can't do is a FILE landing
            # where a DIR exists (EISDIR) or a DIR landing where a file exists.
            # HF never produces such a type-flip for the same path, but be
            # robust: the STAGED (complete) side wins, so drop the stale dest
            # node first. This only ever removes a PARTIAL prior attempt's
            # regenerable node (dest here is never a finished model — a
            # complete copy short-circuits download before promote), so it is
            # EXEMPT from never-delete, same class as the staging temp itself.
            if d_is_dir and not s_is_dir:
                _shutil.rmtree(d, ignore_errors=True)   # stale dir <- staged file
            elif os.path.exists(d) and s_is_dir and not d_is_dir:
                os.remove(d)                            # stale file <- staged dir
            os.replace(s, d)           # now an atomic, unobstructed rename


def _promote_staged(staged: str, dest: str) -> str:
    """Atomically move a COMPLETED staging dir onto its final path. Same
    filesystem -> os.rename (instant). If dest already exists (a prior partial
    attempt), RECURSIVELY merge staged's tree into it (never os.replace a dir
    whole — that raises ENOTEMPTY on a non-empty pre-existing nested dir such
    as HF's ``.cache/huggingface/download``) and drop the emptied staging dir.

    Pure function ``(staged, dest) -> dest``."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        os.rename(staged, dest)          # fast path: clean, whole-dir rename
        return dest
    _merge_tree(staged, dest)            # resume/merge: recurse, depth-safe
    _shutil.rmtree(staged, ignore_errors=True)   # temp cleanup: EXEMPT from never-delete
    return dest


def _discard_staged(staged: str) -> None:
    """Remove a failed/partial staging dir. Temp cleanup — EXEMPT from the
    never-delete rule (in-flight scratch, never catalog data)."""
    _shutil.rmtree(staged, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orphan staging reaper (operator-locked 2026-07-12): a central restart
# SIGKILLs whatever download subprocesses were mid-pull, leaving their
# `.tmp-<pid>` staging dirs behind with no owner ever coming back to promote
# or discard them — dead weight on the shared store (several GB observed).
# A pid is DEFINITELY dead only on ProcessLookupError (os.kill(pid, 0));
# anything else (alive, EPERM, or an unexpected errno) is treated as alive so
# the reaper never removes a staging dir it isn't certain is orphaned. Temp
# staging is EXEMPT from the never-delete rule (module docstring above) — this
# is our own in-flight scratch, never catalog data.
# ---------------------------------------------------------------------------
_ORPHAN_GRACE_SECONDS = int(os.environ.get("HUGPY_STAGING_ORPHAN_GRACE_SECONDS", "600"))
_STAGING_SUFFIX_RE = _re.compile(r"\.tmp-(\d+)$")


def _staging_pid_from_name(name: str) -> "int | None":
    """The pid suffix of a `<dest>.tmp-<pid>` staging dir's basename, else None
    (not a staging dir at all)."""
    m = _STAGING_SUFFIX_RE.search(name)
    return int(m.group(1)) if m else None


def _pid_alive(pid: int) -> bool:
    """Conservative liveness check: only a definite ProcessLookupError (ESRCH)
    says dead. EPERM means it exists under another user; any other unexpected
    OSError is treated as alive too — an ambiguous read must never lead to
    deleting a live download's scratch."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _staging_siblings(dest: str) -> "list[tuple[str, int | None, float]]":
    """Every `<dest>.tmp-*` staging dir currently on disk, as
    ``(path, pid_or_None, mtime)``."""
    parent = os.path.dirname(dest) or "."
    prefix = f"{os.path.basename(dest)}.tmp-"
    try:
        entries = os.listdir(parent)
    except OSError:
        return []
    out = []
    for name in entries:
        if not name.startswith(prefix):
            continue
        path = os.path.join(parent, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        out.append((path, _staging_pid_from_name(name), mtime))
    return out


def staged_bytes(dest: str) -> int:
    """Bytes currently on disk under every live ``<dest>.tmp-*`` staging
    sibling of `dest` — dead or alive, this just answers "what's out there".

    PROGRESS HONESTY (operator-locked 2026-07-12): atomic provisioning lands
    an in-flight download in a per-pid staging sibling and only renames it
    onto `dest` on completion (see `_staging_dir`/`_promote_staged` above).
    The download-progress reader (flask_app/.../downloads/cancelable_downloads.py)
    used to measure bytes at `dest` ONLY, so every in-flight pull showed 0%
    until the finishing rename. It now sums `_dir_bytes(dest) + staged_bytes(dest)`
    — cheap to reason about because `_promote_staged` is a rename (whole-dir
    in the common case, per-file os.replace in the resume/merge case), so a
    byte only ever lives under ONE of {dest, a staging sibling} at any given
    instant; summing both here can never double-count.
    """
    total = 0
    for staged, _pid, _mtime in _staging_siblings(dest):
        for root, _dirs, files in os.walk(staged):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


def reap_orphaned_staging(root: str = None, grace_seconds: int = None) -> "list[str]":
    """Store-wide sweep: remove every ``<dest>.tmp-<pid>`` staging dir whose
    pid is DEFINITELY dead (see `_pid_alive`) AND whose mtime clears
    `grace_seconds` (default 10 min, env HUGPY_STAGING_ORPHAN_GRACE_SECONDS —
    the grace protects a pid that JUST died from being reaped mid-race with
    something still settling). A live pid's staging dir is another process's
    in-flight scratch and is never touched.

    Hooked into the discovery walk (imports/apis/get_module.py discover_model*)
    — it already visits the whole tree, so this needs no new daemon.
    """
    root = root or MODELS_HOME
    grace_seconds = _ORPHAN_GRACE_SECONDS if grace_seconds is None else grace_seconds
    now = _time.time()
    removed = []
    for dirpath, dirnames, _filenames in os.walk(root):
        staging = [d for d in dirnames if _staging_pid_from_name(d) is not None]
        for name in staging:
            dirnames.remove(name)          # scratch, not a model — never descend
            path = os.path.join(dirpath, name)
            pid = _staging_pid_from_name(name)
            if pid is not None and _pid_alive(pid):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if (now - mtime) < grace_seconds:
                continue
            _shutil.rmtree(path, ignore_errors=True)   # temp cleanup: EXEMPT from never-delete
            removed.append(path)
    return removed


def _adopt_or_reap_staging(dest: str, staged: str, grace_seconds: int = None) -> str:
    """Called right before a fresh download starts staging into ``staged``
    (this pid's own ``_staging_dir(dest)``). Looks at ``<dest>.tmp-*``
    siblings left by earlier pids first:

      * a LIVE pid's staging dir is another in-flight download of the same
        model — left alone entirely (no adopt, no reap);
      * the NEWEST dead-pid staging dir is ADOPTED: renamed onto ``staged`` so
        snapshot_download resumes from whatever bytes already landed instead
        of re-fetching from zero. A fresh run always gets a brand new
        ``.tmp-<pid>`` name (that's the point of embedding the pid), so reuse
        requires this rename — HF resume does not work across two different
        ``local_dir`` paths, only within the same one;
      * any OTHER dead-pid orphans (an older crashed attempt — e.g. a second
        central restart mid-pull) are reaped once they clear `grace_seconds`,
        same rule as `reap_orphaned_staging`'s store-wide sweep.

    Returns the path to actually stage into: ``staged`` unchanged, or the
    adopted dir now renamed to live at ``staged``.
    """
    grace_seconds = _ORPHAN_GRACE_SECONDS if grace_seconds is None else grace_seconds
    dead = [(p, pid, mtime) for (p, pid, mtime) in _staging_siblings(dest)
            if not (pid is not None and _pid_alive(pid))]
    if not dead:
        return staged
    dead.sort(key=lambda t: t[2], reverse=True)   # newest mtime first
    newest_path, _newest_pid, _newest_mtime = dead[0]
    rest = dead[1:]
    try:
        os.rename(newest_path, staged)
        print(f"  resuming from orphaned staging: {os.path.basename(newest_path)}")
    except OSError as exc:
        print(f"  [warn] could not adopt orphaned staging {newest_path!r} ({exc}); starting fresh")
        rest = dead   # adoption failed — newest is just another orphan to age out
    now = _time.time()
    for path, _pid, mtime in rest:
        if (now - mtime) >= grace_seconds:
            _shutil.rmtree(path, ignore_errors=True)   # temp cleanup: EXEMPT from never-delete
    return staged


def record_downloaded_model(model: dict, dest: str = None, root: str = DEFAULT_ROOT,
                            report_path: str = None) -> dict:
    """Write one downloaded model into the discovery report so the registry —
    which derives from that report — shows it immediately, no full re-walk.

    Records the real on-disk dir (dest) so folder resolution is exact instead
    of a routed guess. Locked: monitor threads for concurrent downloads all
    read-modify-write the same file."""
    report_path = report_path or MODELS_DISCOVERY_PATH
    dest = dest or route_destination(model, root)
    hub_id = model.get("hub_id")
    name = model.get("name") or (hub_id.split("/")[-1] if hub_id else None)
    if not name:
        return {}

    primary = model.get("primary_task") or model.get("task")
    tasks = model.get("tasks") or ([primary] if primary else None)
    record = {
        "hub_id": hub_id,
        "framework": model.get("framework"),
        "primary_task": primary,
        "tasks": tasks,
        "filename": model.get("filename"),
        "include": model.get("include"),
        "model_max_length": model.get("model_max_length"),
        "dir": dest,
        "folder": os.path.relpath(dest, MODELS_HOME) if dest.startswith(MODELS_HOME) else dest,
    }
    with _REPORT_LOCK:
        report = safe_load_from_json(report_path) or {}
        report[name] = {**report.get(name, {}),
                        **{k: v for k, v in record.items() if v is not None}}
        safe_dump_to_file(data=report, file_path=report_path)
    return report[name]






def _clean_repo_id(hub_id: str) -> str:
    """Strip storage-path routing (family/task/...) back to owner/repo.

    cfg.hub_id sometimes carries the full storage path
    (gguf/text-generation/owner/repo) because discovery recorded the path as
    the id. HF wants just owner/repo. Drop leading family/task segments.
    """
    parts = hub_id.strip("/").split("/")
    FAMILIES = {"gguf", "transformers", "misc", "datasets", "models"}
    while len(parts) > 2 and parts[0] in FAMILIES:
        parts = parts[1:]                       # drop family
        if parts and parts[0] not in FAMILIES:
            parts = parts[1:]                   # drop the task that followed
    return "/".join(parts)

def _stamp(destination: str, key: str, model: dict[str, Any]) -> None:
    """Write hugpy.json into the destination so the model self-describes for
    discovery — identity no longer has to be inferred from the path later."""
    try:
        write_hugpy_marker(
            destination,
            hub_id=model.get("hub_id"),
            name=model.get("name") or key,
            framework=model.get("framework"),
            tasks=model.get("tasks"),
            primary_task=model.get("primary_task"),
            filename=model.get("filename"),
            include=model.get("include"),
            source="download",
        )
    except OSError as exc:
        print(f"  [warn] could not write hugpy.json: {exc}")


def _fetch_mmproj_sidecars(repo_id: str, destination: str) -> None:
    """Best-effort pull of any mmproj/projector GGUF sidecars from ``repo_id``.

    Vision GGUFs are useless for images without their CLIP projector, and the
    sidecar is routinely overlooked when a model is registered by single
    ``filename`` — the model then loads fine but silently answers text-blind.
    So every GGUF download tries these patterns; repos without sidecars (the
    common, text-only case) match nothing and this is a no-op. Never raises:
    a missing projector should surface at serve time (probe/vision honesty),
    not break the weights download that just succeeded."""
    from hugpy_platform.utils import find_mmproj
    try:
        if find_mmproj(destination):
            return  # already have one beside the weights
        _hf().snapshot_download(
            repo_id=repo_id,
            allow_patterns=["*mmproj*.gguf", "*mm-proj*.gguf",
                            "*mm_proj*.gguf", "*projector*.gguf"],
            local_dir=destination,
            local_dir_use_symlinks=False,
        )
        got = find_mmproj(destination)
        if got:
            print(f"  auto-downloaded vision projector sidecar: {os.path.basename(got)}")
    except Exception as exc:
        print(f"  mmproj sidecar check skipped ({type(exc).__name__}: {exc})")


def requested_file_missing(model: dict[str, Any], path: str) -> bool:
    """True when the caller named a SPECIFIC gguf file and THAT file is not in
    *path*. (k119)

    ``model_looks_downloaded`` asks "does this dir hold a usable gguf", and for
    SERVING that is the right question — ``get_gguf_file`` deliberately ELECTS a
    quant when the designated one is absent, so a load still works. For a
    DOWNLOAD it is the wrong question, and it silently ate the operator's
    requests: adding 26 quants of unsloth/Qwen3.8-27B-GGUF from the Add-models
    tab produced 3 files on disk and 26 jobs reading `completed`, because once
    any quant landed, every later request resolved the dir as "already present
    (complete) — skipping".

    Deliberately narrow: only an EXPLICIT ``filename`` on a gguf model. A
    pattern/snapshot download has no single file to check for, and every
    non-gguf framework keeps the existing behaviour exactly."""
    if (model.get("framework") or "") != "gguf":
        return False
    fn = (model.get("filename") or "").strip()
    if not fn:
        return False
    base = os.path.basename(fn).lower()
    try:
        for _root, _dirs, files in os.walk(path):
            if any(f.lower() == base for f in files):
                return False
    except OSError:
        return False        # unreadable: fall back to the old (permissive) rule
    return True


def download_one(model: dict[str, Any],root: str=None,model_key=None, dry_run: bool = False) -> None:
    
    hub_id = model.get("hub_id")
    framework = model.get("framework")
    filename = model.get("filename")
    include = model.get("include")
    if hub_id and not model_key:
        model_key = hub_id.split('/')[-1]
    root = root or DEFAULT_ROOT
    if not hub_id:
        print(f"[SKIP] {model_key}: missing hub_id")
        return

    # New work lands FLAT (task segment retired). If a COMPLETE copy already
    # exists anywhere (flat OR a legacy task path), don't re-download it — the
    # read-through resolver is what stops "redownload models that are ready".
    destination = flat_destination(model, root)
    existing = resolve_model_dir(model, root)
    repo_id, subfolder_from_hub = split_hub_id(hub_id)

    print()
    print(f"[MODEL] {model_key}")
    print(f"  framework: {framework}")
    print(f"  task:      {model.get('primary_task')}")
    print(f"  repo_id:   {repo_id}")
    if subfolder_from_hub:
        print(f"  subfolder: {subfolder_from_hub}")
    print(f"  dest:      {destination}")

    if dry_run:
        return
    if existing:
        if not requested_file_missing(model, existing):
            print(f"  already present (complete) at {existing} — skipping")
            return
        # The dir is "complete" by the resolver's serving rule, but the exact
        # quant the operator asked for is NOT in it. Fetch just that file, INTO
        # the dir that already holds this repo — _promote_staged merges, so the
        # repo stays in one place instead of being split across two layouts.
        destination = existing
        print(f"  present, but {filename} is missing — fetching it into {existing}")

    # ATOMIC: download into a per-pid staging sibling, promote on success only.
    # Before creating a fresh one, adopt the newest dead-pid orphan for this
    # SAME dest (resume its bytes instead of re-fetching) and reap any others.
    staged = _staging_dir(destination)
    staged = _adopt_or_reap_staging(destination, staged)
    os.makedirs(staged, exist_ok=True)
    try:
        # GGUF / single-file
        if framework == "gguf":
            if filename:
                _hf().hf_hub_download(
                    repo_id=repo_id, filename=filename, subfolder=subfolder_from_hub,
                    local_dir=staged, local_dir_use_symlinks=False)
                print(f"  downloaded file: {filename}")
                _fetch_mmproj_sidecars(repo_id, staged)
            elif include:
                _hf().snapshot_download(
                    repo_id=repo_id, allow_patterns=include,
                    local_dir=staged, local_dir_use_symlinks=False)
                print(f"  downloaded pattern: {include}")
                _fetch_mmproj_sidecars(repo_id, staged)
            else:
                _hf().snapshot_download(
                    repo_id=repo_id, local_dir=staged, local_dir_use_symlinks=False)
                print("  downloaded full snapshot")
        elif model.get("task") == "dataset" or model.get("primary_task") == "dataset":
            _hf().snapshot_download(
                repo_id=repo_id, repo_type="dataset",
                local_dir=staged, local_dir_use_symlinks=False)
            print("  downloaded dataset snapshot")
        else:
            # Transformers / full repo
            allow_patterns = include if include else None
            _hf().snapshot_download(
                repo_id=repo_id, allow_patterns=allow_patterns,
                local_dir=staged, local_dir_use_symlinks=False)
            print(f"  downloaded snapshot with pattern: {allow_patterns}"
                  if allow_patterns else "  downloaded full snapshot")
        _stamp(staged, model_key, model)
        try:
            _promote_staged(staged, destination)
        except OSError as exc:
            # Finalize failed AFTER a complete download (bytes are all in
            # `staged`, progress is ~1.0). A bare OSError here reads to the
            # download-job monitor as a transient worker exit and burns all
            # MAX_ATTEMPTS retrying the SAME unfixable collision silently
            # (the 0.999 "does & doesn't gen" wedge). Part A's recursive merge
            # should prevent this, but if a finalize STILL can't reconcile,
            # surface an EXPLICIT, human-readable terminal reason instead of a
            # raw traceback (operator doctrine: failures must be explicit).
            raise RuntimeError(
                f"download finalize failed for {destination}: {exc}") from exc
        publish_catalog_changed("download_one promoted", model_key=model_key,
                                hub_id=hub_id, destination=destination,
                                change="promote")
    except BaseException:
        # A partial/aborted pull must never sit at a resolvable path — discard
        # the staging temp (exempt from never-delete) and re-raise.
        _discard_staged(staged)
        raise


def download_dict_models() -> None:
    parser = argparse.ArgumentParser(description="Auto-distribute LLM downloads by framework/task.")
    parser.add_argument("manifest", type=str, help="Path to model manifest JSON.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="LLM storage root.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional model keys to download.")
    parser.add_argument("--dry-run", action="store_true", help="Print destinations without downloading.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(args.root, "cache", "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(args.root, "cache", "huggingface", "hub"))
    os.environ.setdefault("TORCH_HOME", os.path.join(args.root, "cache", "torch"))
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(args.root, "cache", "pip"))

    # args.manifest is a str -> use open(), not args.manifest.open()
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key, model in manifest.items():
        if args.only and key not in args.only:
            continue
        try:
            download_one(key, model, args.root, dry_run=args.dry_run)
        except Exception as exc:
            print(f"[ERROR] {key}: {exc}")




def _fp16_ignore_patterns(repo_id: str) -> "list[str] | None":
    """Item 4 variant filter: for a DIFFUSERS pipeline repo whose weights ship
    fp16 twins (``x.fp16.safetensors`` beside ``x.safetensors``), return
    ignore_patterns that skip the full-precision twins and the alt runtimes
    (onnx/openvino/flax) — sdxl-turbo drops from 55GB to ~7GB. Returns None
    (no filtering) for non-diffusers repos or repos without fp16 variants, so
    a model that only ships full precision still downloads completely."""
    # File list rides the permanent central HF cache (fetch-once —
    # comms/model_metadata.py repo_files): only the first pull of a repo ever
    # asks HF; a broken cache degrades to the live call as today.
    from hugpy_storage.model_metadata import model_metadata_store
    cached = model_metadata_store.get_repo_files(repo_id)
    if cached is not None:
        files = set(cached)
    else:
        from huggingface_hub import HfApi
        files = set(HfApi().list_repo_files(repo_id))
        model_metadata_store.put_repo_files(repo_id, sorted(files))
    if "model_index.json" not in files:
        return None                      # not a diffusers pipeline — untouched
    twins = [f for f in files
             if f.endswith(".safetensors") and ".fp16." not in f
             and f.replace(".safetensors", ".fp16.safetensors") in files]
    if not twins:
        return None                      # no fp16 variant — pull everything
    ignore = sorted(twins)
    # Alt runtimes ride along in these repos and are never used by our loader.
    ignore += ["*.onnx", "*.onnx_data", "*.msgpack", "*openvino*", "*.ckpt"]
    # .bin twins of safetensors weights (legacy format duplicates).
    ignore += [f for f in files
               if f.endswith(".bin")
               and f.replace(".bin", ".safetensors") in files]
    return ignore


def ensure_model(key: str, root: str = DEFAULT_ROOT) -> str:
    """Make sure registry model ``key`` is on disk and return its dir.

    The row comes from the installed catalog source (the engine's registry);
    with no source installed this raises KeyError — a caller that only has a
    routing dict should use :func:`download_one`."""
    cfg = catalog_get(key)
    if cfg is None:
        raise KeyError(f"Unknown model {key!r} (no catalog source knows it)")

    # Clean identity FIRST — both the repo HF pulls from and the path we
    # route to must use owner/repo, not the storage path.
    repo_id = _clean_repo_id(cfg.hub_id)

    # Route to framework/primary_task/hub_id using route_destination, the same
    # function download_one uses — single source of truth for layout. Build a
    # routing dict whose hub_id is the CLEAN id so we don't re-nest the prefix.
    routing = {
        "hub_id":       repo_id,
        "name":         getattr(cfg, "name", None) or key,
        "framework":    getattr(cfg, "framework", None),
        "primary_task": getattr(cfg, "primary_task", None) or "misc",
    }
    # Read-through: a COMPLETE copy anywhere (flat OR any legacy task path)
    # short-circuits — this is what stops re-downloading ready models.
    existing = resolve_model_dir(routing, root, cfg=cfg)
    if existing:
        # Read-through the box-local NVMe HOT-CACHE tier for the steady-state
        # load: a complete hot copy is served, else the shared path unchanged
        # (a background promotion is scheduled -> the NEXT call is NVMe-hot).
        # Env-gated (HUGPY_HOT_CACHE_ROOT); byte-identical when unset. This is
        # the transformers/diffusers dir chokepoint that resolve_model_source
        # does NOT see (imagegen/embed/keywords/summarizers load via
        # ensure_model). A just-downloaded model (below) is intentionally NOT
        # promoted inline — it promotes on its next call, so a fresh multi-GB
        # pull is never immediately re-copied. Never raises into a load.
        return serve_path(existing)

    # New work lands FLAT. ATOMIC: download into a per-pid staging sibling and
    # promote onto the final path only once complete — a partial pull can never
    # sit at a resolvable model dir.
    path = flat_destination(routing, root)
    staged = _staging_dir(path)
    # Adopt a dead-pid orphan's bytes for this SAME dest before starting
    # fresh (resume instead of re-fetch); reap any other stale orphans.
    staged = _adopt_or_reap_staging(path, staged)
    os.makedirs(staged, exist_ok=True)

    repo_id, subfolder = split_hub_id(repo_id)  # handle owner/repo/subfolder
    download_kwargs = {
        "repo_id": repo_id,
        "local_dir": staged,
        "local_dir_use_symlinks": False,
    }
    if subfolder:
        download_kwargs["subfolder"] = subfolder
    if getattr(cfg, "include", None):
        download_kwargs["allow_patterns"] = cfg.include
    elif getattr(cfg, "filename", None):
        # Single-file model (GGUF quants live N-per-repo): pull just that file,
        # mirroring download_one's llama_cpp branch — never the whole repo.
        download_kwargs["allow_patterns"] = [cfg.filename]
    else:
        # Daylight item 4: diffusers pipelines often ship EVERY precision +
        # runtime (sdxl-turbo: a 55GB repo whose fp16 pipeline is ~7GB). When
        # the repo IS a diffusers pipeline and fp16 twins exist, skip the
        # full-precision twins and the alt runtimes. Introspection failure ->
        # full snapshot (correct, just big) — never a broken model.
        try:
            ignore = _fp16_ignore_patterns(repo_id)
            if ignore:
                download_kwargs["ignore_patterns"] = ignore
                print(f"  fp16 variant filter: skipping {len(ignore)} "
                      f"full-precision/alt-runtime pattern(s)")
        except Exception:
            pass
    try:
        _hf().snapshot_download(**download_kwargs)

        # Vision GGUFs need their mmproj projector beside the weights; a
        # filename/include entry that omits it produces a text-blind model.
        # No-op for repos without sidecars (the common text-only case).
        if getattr(cfg, "framework", None) == "gguf":
            _fetch_mmproj_sidecars(repo_id, staged)

        write_hugpy_marker(
            staged,
            hub_id=repo_id if not subfolder else f"{repo_id}/{subfolder}",
            name=getattr(cfg, "name", None) or key,
            framework=getattr(cfg, "framework", None),
            tasks=getattr(cfg, "tasks", None),
            primary_task=getattr(cfg, "primary_task", None),
            filename=getattr(cfg, "filename", None),
            include=getattr(cfg, "include", None),
            source="download",
        )
        try:
            _promote_staged(staged, path)
        except OSError as exc:
            # See download_one: a finalize collision must surface as an
            # EXPLICIT terminal reason, not a raw OSError that reads as a
            # transient worker exit and loops MAX_ATTEMPTS silently.
            raise RuntimeError(
                f"download finalize failed for {path}: {exc}") from exc
        publish_catalog_changed("ensure_model promoted", model_key=key,
                                hub_id=repo_id, destination=path, change="promote")
        return path
    except BaseException as e:
        # A partial/aborted pull must never sit at a resolvable path — discard
        # the staging temp (exempt from never-delete) and report the failure.
        _discard_staged(staged)
        logger.info(f"Download Failed: {e}")
def ensure_models(models_dict_path=None):
    """Pull every model the catalog lists (``models_dict_path`` is accepted
    for call-site compatibility; the installed catalog source decides where
    its rows come from)."""
    for model in catalog_rows():
        ensure_model(model)
