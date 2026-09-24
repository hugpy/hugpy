import json
import os
from datetime import datetime, timezone
from abstract_essentials import safe_dump_to_json
from hugpy_platform.constants import HUGPY_MARKER, MODELS_HOME
# ---------------------------------------------------------------------------
# Model Paths — One row of everything we know. All Optional — partial fills are valid.
# ---------------------------------------------------------------------------


def join_path(*paths):
    return os.path.join(*paths)

def get_model_home(models_home=None):
    return models_home or MODELS_HOME



# ---------------------------------------------------------------------------
# CAPABILITY FLAGS on the marker (operator, 2026-07-26)
#
# "the 4-bit capable designation should be made a bool in the hugpy.json that
# accompanies the models ... this as well should be the same for an moe capable
# model."
#
# WHY THE MARKER AND NOT A HEURISTIC. Both facts were being INFERRED at read
# time — bnb-eligibility from the model NAME (does it contain "4bit"/"awq"/…)
# and MoE-ness by parsing GGUF headers on demand. Name-matching is a guess that
# a differently-named repo defeats, and the header parse only works for GGUF on
# a box that holds the file. The marker is the model's declared identity and is
# already what discovery keys on, so a capability recorded here is durable,
# survives re-discovery, needs no re-parse, and answers for models the local box
# has never opened.
#
# BOTH ARE CAPABILITY, NOT PREFERENCE. `moe_capable` says the file HAS an expert
# structure; it does not say a split is in use (that is the derived allocation).
# `bnb_capable` says the weights COULD be loaded 4-bit; whether they are is the
# operator's per-worker lever (bnb_by_model). Keeping capability on the model
# and preference on the worker is what lets the same model be 4-bit on the 3090
# and full precision elsewhere.
#
# NULL IS MEANINGFUL: absent/None = "never determined" (an older marker), which
# readers must treat as unknown and fall back to their existing inference —
# never as False. A stamped False is a real measured negative.
# ---------------------------------------------------------------------------

# Repos whose NAME declares an existing quantization: re-quantizing a
# pre-quantized checkpoint fails, and the size win is already banked.
_PREQUANT_NAME_MARKERS = ("4bit", "8bit", "nvfp4", "fp4", "int4", "int8",
                          "gptq", "awq", "-nf4", "bnb")


def detect_bnb_capable(directory, *, framework=None, hub_id=None, name=None):
    """Can this model be loaded with bitsandbytes 4-bit?

    Structural, in preference order:
      * GGUF/comfy -> False. llama.cpp carries its own quantization and comfy
        checkpoints are not transformers loads; bitsandbytes has no meaning.
      * an existing quantization_config in config.json -> False (already
        quantized; a second config fails at load).
      * a name that declares a quantization -> False (the pre-download case,
        where no config.json is on disk yet).
      * otherwise a transformers/diffusers model -> True.
    Returns None only when the framework is unknown — "not determined" rather
    than a guess."""
    fw = str(framework or "").strip().lower()
    if fw in ("gguf", "llama_cpp", "comfy"):
        return False
    blob = f"{hub_id or ''} {name or ''}".lower()
    if any(m in blob for m in _PREQUANT_NAME_MARKERS):
        return False
    cfg_path = os.path.join(directory or "", "config.json")
    try:
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                if cfg.get("quantization_config"):
                    return False
                # A config.json proves it is a transformers-style load.
                return True
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    if fw in ("transformers", "diffusers", "sentence_transformers"):
        return True
    return None


def detect_moe_capable(directory, *, framework=None):
    """Does this model have an EXPERT structure (MoE)?

    GGUF: asks the real reader (gguf_inspect.gguf_moe_detail), which gates on
    the header's expert_count and confirms by tensor name OR shape — the same
    ground truth the allocator prices a split from, so the marker can never
    disagree with the split it enables.
    transformers: reads config.json for the standard expert keys
    (num_experts / num_local_experts / n_routed_experts / moe_layer_freq).
    Returns None when nothing could be read — unknown, never a guessed False."""
    fw = str(framework or "").strip().lower()
    if fw in ("gguf", "llama_cpp"):
        try:
            # The reader is storage's own (gguf_inspect); the engine's spill
            # re-exports it. Guarded anyway: a marker stamp must never fail.
            from hugpy_storage.gguf_inspect import gguf_moe_detail
        except Exception:  # noqa: BLE001
            gguf_moe_detail = None
        if gguf_moe_detail is not None:
            # Quants live in a PER-VARIANT SUBDIR (…/Coder-Next-GGUF/
            # Qwen3-Coder-Next-Q4_K_M/*.gguf), so a top-level listdir finds
            # nothing — walk one level down. Shard-aware by construction: the
            # reader sums split files itself, so the FIRST .gguf answers for the
            # whole set and we stop there rather than parsing every shard.
            try:
                for root, _dirs, files in os.walk(directory or ""):
                    for fn in sorted(files):
                        if fn.lower().endswith(".gguf"):
                            d = gguf_moe_detail(os.path.join(root, fn))
                            return bool(d and d.get("is_moe"))
            except OSError:
                return None
        return None
    cfg_path = os.path.join(directory or "", "config.json")
    try:
        if not os.path.isfile(cfg_path):
            return None
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    blobs = [cfg]
    for k in ("text_config", "llm_config", "language_config"):
        sub = cfg.get(k)
        if isinstance(sub, dict):
            blobs.append(sub)
    for b in blobs:
        for k in ("num_experts", "num_local_experts", "n_routed_experts",
                  "num_experts_per_tok", "moe_layer_freq", "n_expert"):
            v = b.get(k)
            if isinstance(v, (int, float)) and v and int(v) > 1:
                return True
            if isinstance(v, list) and any(v):
                return True
    return False


# ---------------------------------------------------------------------------
# QUANT MANIFEST on the marker (operator, 2026-09-10)
#
# "the hugpy.json should reflect the quants of these models also, and update as
# they have new ones added" — the marker carries a `quants` list, one entry per
# on-disk .gguf VARIANT (shard sets collapse to one entry; mmproj excluded):
#   {"file": <entrypoint basename>, "quant": <token|null>, "bytes": N, "shards": n}
# Stamped at write time and re-synced on every discovery walk
# (sync_marker_quants), so central can answer "which quants exist, at what
# size" straight off the marker instead of re-walking model dirs.
# ---------------------------------------------------------------------------

_QUANT_TOKEN_RE = None  # compiled lazily; hugpy_marker's star imports may lack re


def list_gguf_quants(directory):
    """The on-disk .gguf variant manifest for a model dir. [] when none."""
    import re as _re
    global _QUANT_TOKEN_RE
    if _QUANT_TOKEN_RE is None:
        _QUANT_TOKEN_RE = _re.compile(
            r"(?i)(iq[0-9][a-z0-9_]*|q[0-9][a-z0-9_]*|bf16|fp16|f16|fp32|f32)")
    shard = _re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", _re.I)
    variants = {}
    for root, _dirs, files in os.walk(directory or ""):
        for fn in files:
            low = fn.lower()
            if not low.endswith(".gguf") or "mmproj" in low:
                continue
            try:
                sz = os.path.getsize(os.path.join(root, fn))
            except OSError:
                continue
            key = shard.sub("", fn)
            v = variants.setdefault(key, {"file": key, "quant": None,
                                          "bytes": 0, "shards": 0})
            v["bytes"] += sz
            v["shards"] += 1
    out = []
    for key in sorted(variants):
        v = variants[key]
        hits = _QUANT_TOKEN_RE.findall(os.path.splitext(key)[0])
        if hits:
            v["quant"] = hits[-1].lower()
        out.append(v)
    return out


def sync_marker_quants(directory, marker=None, write=True):
    """Keep an EXISTING marker's `quants` manifest true to disk. Returns the
    updated marker dict when something changed (and was written), else None.
    Creates no marker — stamping identity stays the download/backfill path's
    job. Best-effort by design: any failure leaves the marker untouched."""
    marker = marker if marker is not None else read_hugpy_marker(directory)
    if not isinstance(marker, dict):
        return None
    try:
        quants = list_gguf_quants(directory)
    except Exception:  # noqa: BLE001
        return None
    if not quants and "quants" not in marker:
        return None                       # nothing on disk, nothing declared
    if marker.get("quants") == quants:
        return None                       # already true
    marker["quants"] = quants
    marker["quants_synced_at"] = datetime.now(timezone.utc).isoformat()
    if write:
        try:
            _save_marker(directory, marker)   # atomic; keeps the manifest block
        except Exception:  # noqa: BLE001 — an unwritable marker must not block discovery
            return None
    return marker


# ---------------------------------------------------------------------------
# INSTALL MANIFEST on the marker (operator, 2026-09-23)
#
# "grab the config on install. no reason at all to ever call hugging face for
# anything after a model is downloaded."  The files a download chose, with
# their sizes (and sha256 when the Hub's LFS metadata gave it for free), are
# recorded ONCE — at install — in the model's own hugpy.json:
#
#   "manifest": {"revision": <hub commit sha | null>,
#                "captured_at": <iso>,
#                "source": "huggingface" | "central" | "local",
#                "files": [{"path": <rel>, "bytes": N, "sha256": <hex | null>}]}
#
# Every later verification (worker pull, model audit) reads THIS record; no
# code path asks the Hub what a static file "should" be after install.
#
# sha256 source: huggingface_hub leaves `.cache/huggingface/download/<rel>.metadata`
# beside a local-dir download (commit_hash / etag / timestamp). For an LFS file
# the etag IS the sha256 of the content (a 64-hex string); for a small git file
# it is a git blob sha1 (40-hex) and is not recorded. Nothing is hashed here.
# ---------------------------------------------------------------------------

MANIFEST_KEY = "manifest"
ADMISSION_KEY = "admission"          # see hugpy_storage.admission
ARCHIVE_KEY = "archive"              # see hugpy_storage.archive_mark
# OUTPUT REPAIR (2026-09-23) — a data-derived, per-model output filter spec
# written by hugpy-model-audit (e.g. ``kind: dead_think_delimiters`` — the
# fine-tune's <think>/</think> rows were never trained and the model emits
# ordinary glitch tokens in their place). Central's output path
# (hugpy_engine.output_repair) reads it; nothing else interprets it.
OUTPUT_REPAIR_KEY = "output_repair"
HUB_META_KEY = "hub_meta"
# The Hub-described fields discovery consumes (get_module.resolve_hub_meta).
HUB_META_FIELDS = ("pipeline_tag", "library_name", "auto_model_class",
                   "parameter_count", "license", "gated", "languages", "tags")
# hub_id owner prefixes that are local conventions, never HF namespaces.
LOCAL_HUB_NAMESPACES = {"comfy"}


def hub_meta_from_repo_info(payload):
    """The HUB_META_FIELDS subset of a serialized repo-info row (or None)."""
    if not isinstance(payload, dict):
        return None
    gated = payload.get("gated")
    out = {
        "pipeline_tag": payload.get("pipeline_tag"),
        "library_name": payload.get("library_name"),
        "auto_model_class": payload.get("auto_model_class"),
        "parameter_count": payload.get("safetensors_params"),
        "license": payload.get("license"),
        "gated": bool(gated) if gated is not None else None,
        "languages": payload.get("languages"),
        "tags": payload.get("tags"),
    }
    return out if any(v is not None for v in out.values()) else None


VL_TASK = "image-text-to-text"
_TEXT_TASK = "text-generation"


def vl_gguf_tasks(directory, framework, tasks, primary_task):
    """``(tasks, primary_task)`` with the vision-GGUF rule applied: a
    gguf/llama_cpp dir that holds an mmproj projector (header arch ``clip``,
    or a projector-named file/dir) whose task is text-generation (or unset)
    becomes ``primary_task="image-text-to-text"`` with ``text-generation`` kept
    in ``tasks``. Anything else is returned unchanged."""
    if (framework or "") not in ("gguf", "llama_cpp"):
        return tasks, primary_task
    if primary_task not in (None, _TEXT_TASK, VL_TASK):
        return tasks, primary_task
    if tasks and any(t not in (_TEXT_TASK, VL_TASK) for t in tasks):
        return tasks, primary_task
    if primary_task == VL_TASK and tasks and _TEXT_TASK in tasks:
        return tasks, primary_task                      # already right
    from hugpy_platform.utils import find_mmproj
    if not directory or not find_mmproj(directory):
        return tasks, primary_task
    return [VL_TASK, _TEXT_TASK], VL_TASK


def cached_hub_meta(hub_id):
    """Hub facts for ``hub_id`` from the LOCAL metadata store (the row the
    download/discovery path cached) — never a network call. None on a miss,
    a local namespace (comfy/...) or a malformed id."""
    hub_id = (hub_id or "").strip("/")
    if "/" not in hub_id or hub_id.split("/", 1)[0].lower() in LOCAL_HUB_NAMESPACES:
        return None
    from hugpy_storage.model_metadata import model_metadata_store
    repo = "/".join(hub_id.split("/")[:2])
    return hub_meta_from_repo_info(model_metadata_store.get_repo_info(repo))
_SHA256_RE = None
# Transfer / download machinery that is never part of a model's file set.
_MANIFEST_SKIP_SUFFIXES = (".incomplete", ".part", ".partial", ".downloading",
                           ".aria2", ".lock", ".metadata", ".part.state.json", ".tmp")


def _is_sha256(value):
    import re as _re
    global _SHA256_RE
    if _SHA256_RE is None:
        _SHA256_RE = _re.compile(r"^[0-9a-f]{64}$")
    return isinstance(value, str) and bool(_SHA256_RE.match(value.strip().lower()))


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def manifest_candidate_files(directory):
    """Repo-relative paths of the model files in ``directory`` — what a
    download left there, minus markers and transfer machinery. Never descends
    into dot-dirs (``.cache``/``.git``) or ``*.tmp-<pid>`` staging dirs."""
    out = []
    if not directory or not os.path.isdir(directory):
        return out
    for root, dirs, names in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and ".tmp-" not in d)
        for fn in sorted(names):
            if fn == HUGPY_MARKER or fn.startswith(".llm_storage"):
                continue
            if ".chunksums-" in fn or fn.endswith(_MANIFEST_SKIP_SUFFIXES):
                continue
            full = os.path.join(root, fn)
            if not os.path.isfile(full):
                continue
            out.append(os.path.relpath(full, directory).replace(os.sep, "/"))
    return out


def read_hf_download_metadata(directory, rel):
    """``{"commit": sha, "etag": etag, "timestamp": ts}`` from the metadata
    huggingface_hub wrote beside a local-dir download, or None. Pure file read
    (no huggingface_hub import, no network)."""
    path = os.path.join(directory, ".cache", "huggingface", "download",
                        rel.replace("/", os.sep) + ".metadata")
    try:
        with open(path, "r", encoding="utf-8") as f:
            commit = f.readline().strip()
            etag = f.readline().strip().strip('"')
            ts = float(f.readline().strip())
    except (OSError, ValueError):
        return None
    return {"commit": commit or None, "etag": etag or None, "timestamp": ts}


def build_install_manifest(directory, *, files=None, source="huggingface",
                           revision=None, expected=None):
    """The install manifest for ``directory``.

    ``files``: repo-relative paths to cover (default: every model file on
    disk, :func:`manifest_candidate_files`). ``expected``: optional
    ``{rel: {"bytes": N, "sha256": hex}}`` from a listing already in hand
    (the one-time backfill's Hub listing); otherwise bytes come from disk —
    the download just completed, so disk IS what was fetched — and sha256 from
    the HF download metadata when it is an LFS sha256 still current for the
    file (metadata timestamp not older than the file's mtime). ``revision``
    defaults to the commit the HF metadata agrees on."""
    expected = expected or {}
    rels = list(files) if files is not None else manifest_candidate_files(directory)
    commits = set()
    entries = []
    for rel in sorted(dict.fromkeys(str(r).replace(os.sep, "/") for r in rels)):
        full = os.path.join(directory, rel.replace("/", os.sep))
        try:
            st = os.stat(full)
        except OSError:
            continue                         # not on disk -> not part of this install
        exp = expected.get(rel) or {}
        nbytes = exp.get("bytes")
        sha = exp.get("sha256") if _is_sha256(exp.get("sha256")) else None
        meta = read_hf_download_metadata(directory, rel)
        if meta is not None and meta["timestamp"] + 1 >= st.st_mtime:
            if meta.get("commit"):
                commits.add(meta["commit"])
            if sha is None and _is_sha256(meta.get("etag")):
                sha = meta["etag"].lower()
        entries.append({"path": rel,
                        "bytes": int(nbytes) if nbytes is not None else int(st.st_size),
                        "sha256": sha})
    if revision is None and len(commits) == 1:
        revision = next(iter(commits))
    return {"revision": revision, "captured_at": _utc_now_iso(),
            "source": source, "files": entries}


def merge_manifests(prior, new):
    """``new`` wins per path; ``prior`` entries for paths ``new`` does not
    cover are kept (a quant fetched INTO an existing model dir must not erase
    the record of the quants already there). A carried-over entry from a
    different revision keeps that revision on the entry."""
    if not isinstance(prior, dict) or not prior.get("files"):
        return new
    if not isinstance(new, dict):
        return prior
    have = {e.get("path") for e in new.get("files") or []}
    carried = []
    for e in prior.get("files") or []:
        if not isinstance(e, dict) or e.get("path") in have:
            continue
        e = dict(e)
        if prior.get("revision") and prior.get("revision") != new.get("revision"):
            e.setdefault("revision", prior.get("revision"))
        carried.append(e)
    out = dict(new)
    out["files"] = sorted((new.get("files") or []) + carried, key=lambda e: e.get("path") or "")
    return out


def _save_marker(directory, payload):
    """Atomic write of hugpy.json (unique temp + os.replace)."""
    from hugpy_platform.atomic_json import save_json
    path = os.path.join(directory, HUGPY_MARKER)
    save_json(path, payload)
    return path


def write_marker_manifest(directory, manifest, *, merge=True):
    """Set the ``manifest`` block on an EXISTING hugpy.json (atomic). Returns
    the path, or None when there is no marker to extend. The size fields are
    re-derived from the new manifest in the same write."""
    marker = read_hugpy_marker(directory)
    if not isinstance(marker, dict):
        return None
    if merge:
        manifest = merge_manifests(marker.get(MANIFEST_KEY), manifest)
    marker[MANIFEST_KEY] = manifest
    apply_size_fields(marker, manifest_size_fields(marker))
    return _save_marker(directory, marker)


# ---------------------------------------------------------------------------
# SIZE ON THE MARKER (2026-09-23): "size must never be blank". The install
# manifest already lists every file with its bytes, so the model's size is a
# SUM over a record central wrote at install — no directory walk, no Hub:
#
#   size_bytes      the single-format footprint a worker holds (the same
#                   format_select rule the transfer manifest applies — a
#                   mirrored HF repo's 3-5 formats are not summed); for GGUF the
#                   effective quant when one is recorded (pin or single variant)
#   effective_bytes GGUF only: that effective quant's bytes (shards summed)
#   manifest_bytes  every file the manifest lists
#   size_source     "manifest" | "dir_walk" (the fallback for a marker with no
#                   manifest, stamped back so it is measured once)
#   size_at         when it was derived
#   size_note       set when the number needs a qualifier (GGUF with several
#                   quants and none recorded as effective: the variants' sum)
# ---------------------------------------------------------------------------
SIZE_MARKER_KEYS = ("size_bytes", "effective_bytes", "manifest_bytes",
                    "size_source", "size_at", "size_note")
_GGUF_SHARD_RE = None


def _gguf_group(rel):
    """``(logical variant file, is_projector)`` for a .gguf path, else None."""
    import re as _re
    global _GGUF_SHARD_RE
    if _GGUF_SHARD_RE is None:
        _GGUF_SHARD_RE = _re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", _re.I)
    base = os.path.basename(str(rel))
    if not base.lower().endswith(".gguf"):
        return None
    return _GGUF_SHARD_RE.sub("", base), "mmproj" in base.lower()


def size_fields_from_listing(listing, *, framework=None, filename=None, source="manifest"):
    """The SIZE_MARKER_KEYS dict for ``[(relpath, bytes)]``, or None when the
    listing is empty. Pure: no disk access."""
    listing = [(str(r), int(b)) for r, b in (listing or []) if b is not None]
    if not listing:
        return None
    total = sum(b for _r, b in listing)
    fw = str(framework or "").lower()
    out = {"manifest_bytes": total, "size_source": source, "size_at": _utc_now_iso()}
    if fw in ("gguf", "llama_cpp"):
        groups = {}
        for rel, b in listing:
            g = _gguf_group(rel)
            if g is None or g[1]:
                continue
            groups[g[0]] = groups.get(g[0], 0) + b
        eff = None
        if filename:
            pin = _gguf_group(filename)
            key = pin[0].lower() if pin else os.path.basename(str(filename)).lower()
            eff = next((v for k, v in groups.items() if k.lower() == key), None)
        if eff is None and len(groups) == 1:
            eff = next(iter(groups.values()))
        if eff is not None:
            out["effective_bytes"] = eff
            out["size_bytes"] = eff
        elif groups:
            out["size_bytes"] = sum(groups.values())
            out["size_note"] = (f"no effective quant recorded (no filename pin, "
                                f"{len(groups)} GGUF variants): sum of all variants")
        else:
            out["size_bytes"] = total
            out["size_note"] = "GGUF model with no .gguf weight file in the listing"
        return out
    try:
        from hugpy_storage.format_select import effective_bytes
        out["size_bytes"] = int(effective_bytes(listing, framework=framework))
    except Exception:  # noqa: BLE001 — the plain sum is still a measured fact
        out["size_bytes"] = total
    return out


def manifest_size_fields(marker, *, framework=None, filename=None):
    """Size fields from the marker's install manifest alone, or None when it
    carries no manifest with sized files."""
    if not isinstance(marker, dict):
        return None
    man = marker.get(MANIFEST_KEY)
    files = man.get("files") if isinstance(man, dict) else None
    if not files:
        return None
    listing = [(e.get("path"), e.get("bytes")) for e in files
               if isinstance(e, dict) and e.get("path") and e.get("bytes") is not None]
    return size_fields_from_listing(listing, framework=framework or marker.get("framework"),
                                    filename=filename or marker.get("filename"))


def apply_size_fields(marker, fields):
    """Replace the marker's size keys with ``fields`` (in place). No-op on None."""
    if not isinstance(marker, dict) or not fields:
        return marker
    for k in SIZE_MARKER_KEYS:
        marker.pop(k, None)
    marker.update({k: v for k, v in fields.items() if k in SIZE_MARKER_KEYS and v is not None})
    return marker


def stamp_marker_sizes(directory, *, marker=None, allow_walk=False, write=True):
    """Derive the size fields for an EXISTING marker (manifest first; the dir
    walk only when ``allow_walk`` and the marker has no manifest) and write
    them back. Returns ``(fields | None, reason)`` — ``reason`` names why no
    size could be derived."""
    marker = marker if marker is not None else read_hugpy_marker(directory)
    if not isinstance(marker, dict):
        return None, f"no hugpy.json in {directory}"
    fields = manifest_size_fields(marker)
    if fields is None and allow_walk:
        listing = []
        for rel in manifest_candidate_files(directory):
            try:
                listing.append((rel, os.path.getsize(os.path.join(directory, rel))))
            except OSError:
                continue
        fields = size_fields_from_listing(listing, framework=marker.get("framework"),
                                          filename=marker.get("filename"), source="dir_walk")
        if fields is None:
            return None, f"no manifest in {directory}/hugpy.json and no model files under {directory}"
    if fields is None:
        return None, f"no manifest (with file bytes) in {directory}/hugpy.json"
    apply_size_fields(marker, fields)
    if write:
        _save_marker(directory, marker)
    return fields, ""


def read_output_repair(directory):
    """The ``output_repair`` block of the model's hugpy.json, or None."""
    marker = read_hugpy_marker(directory) if directory else None
    block = (marker or {}).get(OUTPUT_REPAIR_KEY)
    return block if isinstance(block, dict) else None


def write_output_repair(directory, block):
    """Set (``block`` dict) or clear (``None``) the ``output_repair`` block on an
    EXISTING hugpy.json (atomic). Returns the path, or None when there is no
    marker to extend."""
    marker = read_hugpy_marker(directory)
    if not isinstance(marker, dict):
        return None
    if block is None:
        if OUTPUT_REPAIR_KEY not in marker:
            return None
        marker.pop(OUTPUT_REPAIR_KEY, None)
    else:
        marker[OUTPUT_REPAIR_KEY] = dict(block)
    return _save_marker(directory, marker)


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt",
                    ".onnx", ".onnx_data", ".msgpack", ".h5", ".npz", ".sft")


def manifest_file_status(directory, manifest, *, paths=None, weights_only=False):
    """Compare disk to an install manifest. ``{"checked": n, "missing":
    [rel], "mismatch": [(rel, disk_bytes, expected_bytes)]}``. ``paths``
    restricts the check to that scope (a worker holds one quant of a
    multi-quant repo); ``weights_only`` skips small non-weight files that
    hugpy legitimately edits after install (config shims)."""
    scope = None if paths is None else {str(p).replace(os.sep, "/") for p in paths}
    out = {"checked": 0, "missing": [], "mismatch": []}
    for e in (manifest or {}).get("files") or []:
        rel = e.get("path") if isinstance(e, dict) else None
        if not rel or (scope is not None and rel not in scope):
            continue
        if weights_only and not rel.lower().endswith(_WEIGHT_SUFFIXES):
            continue
        out["checked"] += 1
        try:
            size = os.path.getsize(os.path.join(directory, rel.replace("/", os.sep)))
        except OSError:
            out["missing"].append(rel)
            continue
        want = e.get("bytes")
        if want is not None and int(want) != size:
            out["mismatch"].append((rel, size, int(want)))
    return out


def write_hugpy_marker(directory, *, hub_id, name=None, framework=None,
                       tasks=None, primary_task=None, filename=None,
                       include=None, source="download", manifest=None, **extra):
    """Stamp a model dir with its identity. Single source of truth for what
    this model IS — discovery keys on it instead of guessing from the path.

    ``manifest`` (the install manifest, see build_install_manifest) is written
    when given; when omitted, an install manifest ALREADY on the marker is
    carried over untouched — re-stamping identity (reclassify, reconcile) must
    never drop the once-captured record."""
    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]
    # VISION GGUF (2026-09-23): a GGUF dir holding an mmproj projector is an
    # image-text-to-text model, not text-generation — stamped here, once, so
    # the catalog, the suite registry and routing all read the same task.
    try:
        tasks, primary_task = vl_gguf_tasks(directory, framework, tasks, primary_task)
    except Exception:  # noqa: BLE001 — a header probe must never block the stamp
        pass
    payload = {
        "hub_id": hub_id,
        "name": name or (hub_id.split("/")[-1] if hub_id else None),
        "framework": framework,
        "tasks": tasks,
        "primary_task": primary_task or (tasks[0] if tasks else None),
        "filename": filename,
        "include": include,
        "source": source,                       # "download" | "custom"
        "stamped_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    # CAPABILITY FLAGS (operator, 2026-07-26) — stamped at write time so the
    # facts are durable rather than re-inferred from the model's NAME (bnb) or a
    # GGUF header re-parse (moe) on every read. An explicit value passed by the
    # caller via **extra always wins; these only fill what wasn't supplied, and a
    # detector that cannot tell leaves the key ABSENT (unknown), never False.
    for field, detect in (("bnb_capable",
                           lambda: detect_bnb_capable(
                               directory, framework=framework,
                               hub_id=hub_id, name=payload.get("name"))),
                          ("moe_capable",
                           lambda: detect_moe_capable(
                               directory, framework=framework))):
        if payload.get(field) is None:
            try:
                val = detect()
            except Exception:  # noqa: BLE001 — a probe must never block the stamp
                val = None
            if val is not None:
                payload[field] = bool(val)
    # QUANT MANIFEST (operator, 2026-09-10) — stamp what variants are on disk
    # right now; discovery's sync_marker_quants keeps it true as quants are
    # added/removed later. Caller-supplied `quants` via **extra wins.
    if payload.get("quants") is None:
        try:
            q = list_gguf_quants(directory)
            if q:
                payload["quants"] = q
        except Exception:  # noqa: BLE001
            pass
    prior = read_hugpy_marker(directory)
    prior = prior if isinstance(prior, dict) else {}
    if manifest is None and isinstance(prior.get(MANIFEST_KEY), dict):
        manifest = prior[MANIFEST_KEY]
    if manifest is not None:
        payload[MANIFEST_KEY] = manifest
        # SIZE from the install manifest (see SIZE_MARKER_KEYS) — stamped with
        # every identity write so the size is never re-walked.
        try:
            apply_size_fields(payload, manifest_size_fields(payload))
        except Exception:  # noqa: BLE001 — a size must never block the stamp
            pass
    # ADMISSION (2026-09-23) — the post-download gate's verdict lives on this
    # record (hugpy_storage.admission). A re-stamp (reclassify, reconcile,
    # comfy sweep) carries it over; a download sets it afresh via the install
    # hook after the dir is promoted.
    if payload.get(ADMISSION_KEY) is None and isinstance(prior.get(ADMISSION_KEY), dict):
        payload[ADMISSION_KEY] = prior[ADMISSION_KEY]
    # ARCHIVE MARK (2026-09-23, hugpy_storage.archive_mark) — the operator's
    # recorded intent to archive this model. Only the operator's unmark (or the
    # archive sweep) ends it; a re-stamp (reclassify, reconcile, re-download)
    # carries it over.
    if payload.get(ARCHIVE_KEY) is None and isinstance(prior.get(ARCHIVE_KEY), dict):
        payload[ARCHIVE_KEY] = prior[ARCHIVE_KEY]
    # OUTPUT REPAIR — derived from the weights, so any re-stamp carries it over.
    if payload.get(OUTPUT_REPAIR_KEY) is None and isinstance(prior.get(OUTPUT_REPAIR_KEY), dict):
        payload[OUTPUT_REPAIR_KEY] = prior[OUTPUT_REPAIR_KEY]
    # HUB FACTS, CAPTURED ONCE (2026-09-23: "no more verifying static values
    # with api calls"). The descriptive fields discovery used to ask the Hub
    # for on every walk (pipeline_tag, license, ...) are stamped here from the
    # metadata store row the download path already filled — a local read,
    # never a fetch — and read back by get_module's resolver chain.
    if payload.get(HUB_META_KEY) is None:
        hub_meta = prior.get(HUB_META_KEY) if isinstance(prior.get(HUB_META_KEY), dict) else None
        if hub_meta is None and source == "download":
            try:
                hub_meta = cached_hub_meta(hub_id)
            except Exception:  # noqa: BLE001 — a cache read must never block the stamp
                hub_meta = None
        if hub_meta:
            payload[HUB_META_KEY] = hub_meta
    os.makedirs(directory, exist_ok=True)
    return _save_marker(directory, payload)


def read_hugpy_marker(directory):
    """Return the declared identity dict, or None if unstamped/unreadable."""
    path = os.path.join(directory, HUGPY_MARKER)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def has_hugpy_marker(directory):
    return os.path.isfile(os.path.join(directory, HUGPY_MARKER))


def hub_id_for(directory, fallback=None):
    """Repo id from the declared marker; explicit sources, path slice last.

    1. hugpy.json (authoritative — declared at download/custom time)
    2. legacy .llm_storage_installed.json marker
    3. config.json _name_or_path
    4. fallback (path slice) — only if nothing self-describes
    """
    marker = read_hugpy_marker(directory)
    if marker and marker.get("hub_id"):
        return marker["hub_id"]

    legacy = os.path.join(directory, ".llm_storage_installed.json")
    if os.path.isfile(legacy):
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                hid = json.load(f).get("hub_id")
            if hid:
                return hid
        except (OSError, json.JSONDecodeError):
            pass

    cfg = os.path.join(directory, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                nop = json.load(f).get("_name_or_path")
            if nop and "/" in nop and not os.path.isabs(nop):
                return nop
        except (OSError, json.JSONDecodeError):
            pass

    return fallback


def backfill_markers(get_model_dirs, hub_id_fallback=lambda d: None, verbose=True):
    """One-time: stamp a hugpy.json into any model dir that lacks one, using
    whatever identity can be salvaged (legacy marker, config.json, fallback).
    After this, every dir is self-describing."""
    stamped, skipped = [], []
    for directory in get_model_dirs():
        if has_hugpy_marker(directory):
            skipped.append(directory)
            continue
        hub_id = hub_id_for(directory, hub_id_fallback(directory))
        if not hub_id:
            if verbose:
                print(f"[backfill] no hub_id resolvable, skipping: {directory}")
            continue
        framework = None
        cfg = read_hugpy_marker(directory)  # None here, but keep shape
        write_hugpy_marker(directory, hub_id=hub_id, source="backfill")
        stamped.append(directory)
        if verbose:
            print(f"[backfill] stamped {hub_id} -> {directory}")
    return {"stamped": stamped, "skipped": skipped}
