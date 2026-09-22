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
            safe_dump_to_json(file_path=os.path.join(directory, HUGPY_MARKER),
                              data=marker, indent=2)
        except Exception:  # noqa: BLE001 — an unwritable marker must not block discovery
            return None
    return marker


def write_hugpy_marker(directory, *, hub_id, name=None, framework=None,
                       tasks=None, primary_task=None, filename=None,
                       include=None, source="download", **extra):
    """Stamp a model dir with its identity. Single source of truth for what
    this model IS — discovery keys on it instead of guessing from the path."""
    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]
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
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, HUGPY_MARKER)
    safe_dump_to_json(file_path=path,data=payload, indent=2)
    return path


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
