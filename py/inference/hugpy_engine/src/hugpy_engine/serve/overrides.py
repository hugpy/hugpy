"""Persisted, per-model serving overrides — the UI-writable layer.

The registry (MODELS + discovery) gives each model its baseline serving config
in ``cfg.extra``; this overlay lets the console change it per model at runtime
without rebuilding the registry or editing code. Stored as one JSON file keyed
by model_key:

    {"DAN-L3-R1-8B-i1-GGUF": {"serve_mode": "systemd", "n_gpu_layers": -1,
                              "threads": 8, "llama_ctx": 8192}}

:func:`serve_spec_for` merges this over ``cfg.extra`` (override wins), so the
systemd unit, the swap config, and the HTTP runner endpoint all reflect it.
"""
from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

try:
    from hugpy_platform.constants import PROJECTS_HOME
except Exception:  # pragma: no cover - fall back if layout differs
    from hugpy_platform.app_dirs import models_root
    PROJECTS_HOME = os.environ.get("PROJECTS_HOME") or os.path.join(
        os.environ.get("DEFAULT_ROOT") or models_root(), "projects")

_OVERRIDES_PATH = os.environ.get("SERVE_OVERRIDES_PATH") or os.path.join(
    PROJECTS_HOME, "serve_overrides.json")
_LOCK = threading.Lock()

# Fields the console may set per model. Anything else is ignored.
ALLOWED_FIELDS = {
    "serve_mode",     # off | systemd | swap
    "n_gpu_layers",   # GPU offload (-1 all, 0 cpu, N layers)
    "n_cpu_moe",      # MoE expert split: N MoE layers whose EXPERT tensors stay
                      # on CPU (999 = all — spill.MOE_ALL_LAYERS). Rides
                      # llama-server --n-cpu-moe; measured on ae 2026-07-24:
                      # -1 layers + 999 beat the 17/48 layer split by +59% tok/s
                      # at 5x less VRAM on an 80B-A3B MoE. Absent -> auto policy
                      # (MoE hybrid auto-splits; dense/fits-whole unchanged).
    "threads",        # CPU threads
    "llama_ctx",      # context window
    "gpu_mem_gib",    # transformers per-GPU budget / explicit-mode VRAM target
    "cpu_mem_gib",    # transformers CPU/RAM budget / explicit-mode RAM target
    "always_on",      # systemd always-on vs swap on-demand
    "ttl_seconds",    # swap idle-unload TTL
    "gguf_file",      # which downloaded .gguf to serve (basename; "" = auto/default)
    # k37 — the five-mode allocation selector (gpu-only|ram-only|max-gpu|
    # max-ram|explicit; legacy names resolved on write, never stored back).
    "alloc_mode",
    "leniency_pct",   # explicit mode: N% OF THE MODEL may land off its ideal
                      # device before bust (100% GPU + 30% -> floor 70/30)
    "priority",       # explicit mode: flex priority (0 = normal; higher
                      # compresses lower-priority neighbours within bands)
    "priority_device",  # explicit mode: which device the target favors
                        # ("gpu" default | "ram")
    # ── k56 (operator ruling 2026-07-31) — the two GENERAL placement options ──
    "worker_prefs",   # ORDERED worker preference: ["ae", "computron"]. The
                      # generalization of designation from ONE hard binding to a
                      # ranked candidate list — resolution tries them in order
                      # and takes the FIRST whose admission accepts, and a model
                      # carrying a list NEVER lands off it (designation hardness
                      # is preserved per candidate). A single-entry list is the
                      # degenerate case and behaves exactly as the one hard
                      # designation always did. Model-scoped by necessity: the
                      # designation SoT (worker["models"] membership, one set per
                      # WORKER) can express "designated to both" but has nowhere
                      # to put a cross-worker ORDER.
    "no_evict",       # POLITE LOAD: admission may spend only genuinely free
                      # headroom (free VRAM after the tolerance-band flex) and
                      # never evicts a resident to land. The deliberate inverse
                      # of declare-need-then-evict, which stays the rule for
                      # every unflagged load. Composes with worker_prefs: try
                      # each candidate politely, refuse honestly if none admits.
                      # k62: this boolean is now the ALL-WORKERS DEFAULT — the
                      # per-worker map below overrides it per candidate.
    "no_evict_by_worker",  # k62 — politeness INDIVIDUALIZED per (model ×
                      # worker): {"ae": true, "computron": false}. Politeness is
                      # a statement about ONE box's contention, not about the
                      # model (flux2 is polite on ae's contended 3090 and holds
                      # ordinary eviction rights on computron), so it belongs at
                      # the same grain as the decision it changes. Central
                      # resolves map[W] if W is in it, else ``no_evict``, and
                      # simply includes/omits the spill key for THAT worker —
                      # the wire and the worker's admission are untouched.
    "strict",         # k-dist (operator ruling 2026-09-24): keep the HARD
                      # designation fence for THIS model even under the fleet
                      # "feasible" distribution default. True -> worker_prefs /
                      # designations are a sealed scope and an unmet preference
                      # REFUSES (pre-2026-09-24 behaviour); absent/False -> the
                      # preference is an ORDER with a feasible-set fallback.
    "gguf_file_by_worker",  # per-(model × worker) quant pin, modeled on
                      # no_evict_by_worker: {worker_name_or_id: basename_or_
                      # quant_token}. WHICH .gguf serves is a statement about
                      # ONE box's VRAM, not about the model (q8_0 on a 24 GiB
                      # card, q4_k_m on the 8 GiB 4060), so it belongs at the
                      # same grain. Resolution: map[W] wins over the model-wide
                      # ``gguf_file`` for THAT worker; every other worker keeps
                      # the model-wide pin / election exactly as before.
}
_INT_FIELDS = {"n_gpu_layers", "n_cpu_moe", "threads", "llama_ctx", "ttl_seconds",
               "priority"}
_FLOAT_FIELDS = {"gpu_mem_gib", "cpu_mem_gib", "leniency_pct"}
# NOTE: "strict" is a boolean but is handled in _coerce with the OFF-clears
# discipline (like "no_evict"), so it is deliberately NOT in _BOOL_FIELDS.
_BOOL_FIELDS = {"always_on"}


def _load() -> dict:
    try:
        with open(_OVERRIDES_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def all_overrides() -> dict:
    return _load()


def get_override(model_key: str) -> dict:
    return _load().get(model_key, {}) or {}


def _gguf_basenames(model_dir: str) -> list:
    """Sorted basenames of the model's downloaded .gguf files, excluding the
    mmproj projector (which is not a servable language model)."""
    try:
        from hugpy_platform.utils import is_mmproj_file
    except Exception:
        is_mmproj_file = lambda p: "mmproj" in os.path.basename(str(p)).lower()
    out = []
    try:
        for fn in os.listdir(model_dir):
            if fn.lower().endswith(".gguf") and not is_mmproj_file(os.path.join(model_dir, fn)):
                out.append(fn)
    except OSError:
        return []
    return sorted(out)


def available_gguf_files(model_dir: str) -> list:
    """Public: the .gguf variants the operator may pick for this model."""
    return _gguf_basenames(model_dir)


def resolve_gguf_for_worker(by_worker, forms) -> "str | None":
    """Effective per-worker quant pin for ONE worker: ``map[W]`` when W is in the
    ``gguf_file_by_worker`` map, else None (the model-wide ``gguf_file`` /
    election governs). PURE (no disk read), shaped like :func:`resolve_polite`
    and matching id OR name case-insensitively for the same reason: the console
    posts ids, an operator editing the file writes names, and a pin that
    silently failed to match would serve the wrong quant on the box the
    operator sized it for."""
    want = {str(f).strip().lower() for f in (forms or []) if str(f).strip()}
    for name, val in (by_worker or {}).items():
        if str(name).strip().lower() in want and str(val or "").strip():
            return str(val).strip()
    return None


_LOCAL_WORKER_ID = None      # lazily-read id from the worker id-file, cached


def _local_worker_forms() -> list:
    """This process's own worker identity spellings — env WORKER_NAME, the
    hostname, and the enrolled worker id from the id-file when readable. Used
    when a caller resolves an override with no explicit worker: the load paths
    all run ON the box that serves, so 'this box' is the right default, and a
    process with no identity (central sizing a model in the abstract) simply
    matches nothing in the per-worker map."""
    global _LOCAL_WORKER_ID
    forms = []
    name = os.environ.get("WORKER_NAME")
    if name:
        forms.append(name)
    try:
        import socket
        forms.append(socket.gethostname())
    except Exception:  # noqa: BLE001
        pass
    if _LOCAL_WORKER_ID is None:
        try:
            from hugpy_platform import app_dirs as _hp
            idf = _hp.worker_id_file()
            with open(idf, "r", encoding="utf-8") as fh:
                _LOCAL_WORKER_ID = str((json.load(fh) or {}).get("worker_id") or "")
        except Exception:  # noqa: BLE001
            _LOCAL_WORKER_ID = ""
    if _LOCAL_WORKER_ID:
        forms.append(_LOCAL_WORKER_ID)
    return forms


def resolve_override_gguf(model_key: str, model_dir: str, worker=None):
    """Absolute path of the operator-selected .gguf for this model, IF a pin is
    set and resolves to a file under ``model_dir``; else None (caller falls back
    to the registry/auto resolution). Honored by both the in-process runner and
    the systemd/swap serve spec, so the choice is global.

    PRECEDENCE (mirrors gguf_election rule 0 — a designation always wins):
    per-worker ``gguf_file_by_worker`` → model-wide ``gguf_file``. ``worker``
    is an optional id/name (or iterable of spellings) naming WHICH worker is
    asking; None means this process's own identity (:func:`_local_worker_forms`)
    — the load paths run on the box that serves, and a process with no identity
    matches nothing per-worker, i.e. exactly the old model-wide behavior.

    The stored value may be a full basename OR a quant token ("q8_0"): an exact
    basename that exists wins, else a case-insensitive substring match over the
    servable files (the same tolerance ``get_gguf_file``'s ``prefer`` applies,
    so the two resolvers cannot disagree about what a token selects)."""
    ov = get_override(model_key) or {}
    forms = ([worker] if isinstance(worker, str) else list(worker or ())) \
        or _local_worker_forms()
    fn = resolve_gguf_for_worker(ov.get("gguf_file_by_worker"), forms) \
        or ov.get("gguf_file")
    if not fn:
        return None
    cand = os.path.join(model_dir, os.path.basename(str(fn)))
    if os.path.isfile(cand):
        return cand
    want = os.path.basename(str(fn)).lower()
    hits = sorted(rel for rel, _sz in _servable_gguf_files(model_dir)
                  if want in os.path.basename(rel).lower())
    return os.path.join(model_dir, hits[0]) if hits else None


def _file_bytes(model_dir: str, fn: str) -> int:
    try:
        return int(os.path.getsize(os.path.join(model_dir, os.path.basename(str(fn)))))
    except OSError:
        return 0


def _mmproj_bytes(model_dir: str) -> int:
    """Bytes of THE ONE projector the runner actually loads, not the sum of
    every precision variant the repo ships. A vision GGUF is a PAIR (quant +
    mmproj-*.gguf), and llama.cpp loads exactly ONE projector via ``--mmproj`` /
    ``clip_model_path`` — the one ``find_mmproj`` resolves (the same resolver the
    runner uses). A repo commonly ships the SAME projector at several precisions
    (e.g. unsloth's Qwen2.5-VL: mmproj-F32 2.5GB + F16 1.26GB + BF16 1.26GB);
    summing all three (the old behavior) over-counted effective_bytes by ~4GB
    and wrongly rejected the model on cards it fits — the exact mirror of the
    GGUF quant-ladder over-count ``_incoming_need_bytes`` already fixes.

    Falls back to the single LARGEST projector under the dir when the resolver
    misses (e.g. a projector nested in a subdir that ``find_mmproj``'s top-level
    scan doesn't reach) — never the sum, and never 0 when a projector is present.
    """
    try:
        from hugpy_platform.utils import is_mmproj_file, find_mmproj
    except Exception:  # noqa: BLE001
        is_mmproj_file = lambda p: "mmproj" in os.path.basename(str(p)).lower()
        find_mmproj = None
    # The runner's pick, sized as-is.
    if find_mmproj is not None:
        try:
            pick = find_mmproj(model_dir)
            if pick:
                p = pick if os.path.isabs(pick) else os.path.join(model_dir, pick)
                if os.path.isfile(p):
                    return int(os.path.getsize(p))
        except Exception:  # noqa: BLE001 — resolution is best-effort
            pass
    # Fallback: the largest single projector anywhere under the dir (one loads,
    # not the sum). Covers a nested projector the top-level resolver misses.
    largest = 0
    try:
        for root, _dirs, files in os.walk(model_dir):
            for fn in files:
                if fn.lower().endswith(".gguf") and is_mmproj_file(os.path.join(root, fn)):
                    try:
                        largest = max(largest,
                                      int(os.path.getsize(os.path.join(root, fn))))
                    except OSError:
                        pass
    except OSError:
        pass
    return largest


# A split/sharded GGUF ships as N files ``<stem>-<NNNNN>-of-<MMMMM>.gguf``; they
# are ONE logical model that llama.cpp loads from the first shard. Re-exported
# from the shared elector so this module and ``get_gguf_file`` cannot drift.
from hugpy_engine.gguf_election import SHARD_RE as _SHARD_RE


def _servable_gguf_files(model_dir: str) -> list:
    """Recursively list servable .gguf files (relative paths + sizes), excluding
    the mmproj projector.

    RECURSIVE, unlike :func:`_gguf_basenames`: a split/sharded GGUF nests its
    shards in a subdir (e.g. ``<quant>/<quant>-00001-of-00004.gguf``), and a
    shallow ``os.listdir`` misses them entirely — the model then resolved to NO
    servable variant and no ``effective_bytes`` at all (the sharded-GGUF
    effective-size blind spot, t33). Mirrors ``get_gguf_file``'s recursive glob
    so the two agree on what is servable.

    Returns ``[(relpath, bytes), …]`` sorted by relpath.
    """
    try:
        from hugpy_platform.utils import is_mmproj_file
    except Exception:  # noqa: BLE001
        is_mmproj_file = lambda p: "mmproj" in os.path.basename(str(p)).lower()
    out = []
    try:
        for root, _dirs, files in os.walk(model_dir):
            for fn in files:
                if not fn.lower().endswith(".gguf") or is_mmproj_file(os.path.join(root, fn)):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, model_dir)
                try:
                    sz = int(os.path.getsize(full))
                except OSError:
                    sz = 0
                out.append((rel, sz))
    except OSError:
        return []
    return sorted(out)


def _gguf_variant_groups(files: list) -> list:
    """Collapse a recursive servable-gguf listing into pickable VARIANTS,
    shard-aware.

    A split GGUF presents as N shard files (``-00001-of-0000N.gguf`` …) that are
    ONE logical model; they fold into a single variant whose ``bytes`` = SUM of
    all shards and whose entrypoint/canonical filename is the ``-00001`` shard
    (what ``get_gguf_file``/llama-server is actually pointed at). Every non-shard
    .gguf is its own variant. Multiple quant families that are each sharded stay
    separate (grouped by containing dir + shard stem), so they rank as distinct
    variants exactly like single-file quants do.

    ``files``: ``[(relpath, bytes), …]``. Returns the variant dicts described in
    ``imports.src.gguf_election.group_variants`` — ``{filename, bytes, members,
    complete, shards, shard_total, missing_shards, incomplete_reason, quant,
    full_precision}``.

    DELEGATES to the shared elector. This function used to fold shards with its
    own private copy of the shard regex (the tree had six), and like every other
    copy it captured the ``-of-MMMMM`` total and never read it — so a directory
    holding ``q8_0-00001-of-00003`` and nothing else of that quant reported a
    third of a model as a whole, electable variant. Completeness now comes from
    the same code that decides the election, which is the only way the two can
    agree.
    """
    from hugpy_engine.gguf_election import group_variants
    return group_variants(files)


def gguf_variants_detail(model_key: str, model_dir: str, cfg=None) -> dict:
    """Per-variant sizes + the EFFECTIVE (resolved) quant for a GGUF model.

    A GGUF repo commonly holds several quantizations (Q4_K_M, Q5_K_M, Q8_0…) but
    only ONE is served, so summing the whole directory badly overstates "the
    model". This resolves the single quant the runner will actually load — exactly
    as ``get_gguf_file`` does (operator ``gguf_file`` override → ``cfg.filename``
    → deterministic auto-rank) — and reports its size plus each pickable variant's
    size. Model-level and worker-agnostic (a ``.gguf`` is identical bytes on every
    box), so the console reuses this one number wherever a GGUF's size is shown.

    Returns ``{}`` for a dir with no servable .gguf (e.g. a transformers model or
    a not-yet-downloaded repo), so callers fall back to their existing size.

    Shard-aware: a split GGUF is N shard files that are ONE model — they collapse
    into a single variant whose bytes SUM the shards (see ``_gguf_variant_groups``),
    so a sharded model resolves a real ``effective_bytes`` instead of ``None``
    (t33: a ~48GB sharded coder model that read as no size at all).
    """
    files = _servable_gguf_files(model_dir)         # recursive; catches nested shards
    if not files:
        return {}
    mmproj = _mmproj_bytes(model_dir)
    variants = _gguf_variant_groups(files)          # shard sets folded into one variant
    # Resolve the effective entrypoint exactly as the runner does (operator
    # gguf_file override -> cfg.filename -> deterministic auto-rank), then find
    # which variant that resolved file belongs to (a shard maps to its group).
    eff_full = None
    try:
        from hugpy_engine.config.main import get_gguf_file
        prefer = (get_override(model_key) or {}).get("gguf_file") or None
        p = get_gguf_file(model_dir, cfg, prefer=prefer)
        eff_full = os.path.abspath(p) if p else None
    except Exception:  # noqa: BLE001 — resolution is best-effort
        eff_full = None
    eff_variant = None
    if eff_full:
        eff_base = os.path.basename(eff_full).lower()
        for v in variants:
            member_fulls = {os.path.abspath(os.path.join(model_dir, m))
                            for m in v["members"]}
            if eff_full in member_fulls or any(
                    os.path.basename(m).lower() == eff_base for m in v["members"]):
                eff_variant = v
                break
    if eff_variant is None and len(variants) == 1:
        eff_variant = variants[0]                    # single variant is unambiguously it
    eff = eff_variant["filename"] if eff_variant else None
    eff_quant = eff_variant["bytes"] if eff_variant else 0
    # INCOMPLETE variants stay LISTED — the operator needs to see the litter (a
    # lone ``-00002-of-00002`` shard, a q8_0 missing two thirds of itself) to
    # know it is there and reclaim the space. They are simply never electable,
    # and they say why. Reporting a third of a model as a variant with no
    # qualifier is how a 3-of-3-shard set read as installed for months.
    out_variants = []
    for v in variants:
        row = {"filename": v["filename"], "bytes": v["bytes"],
               "is_effective": (v is eff_variant)}
        if not v.get("complete"):
            row["complete"] = False
            row["incomplete_reason"] = v.get("incomplete_reason")
            row["shards"] = v.get("shards")
            row["shard_total"] = v.get("shard_total")
            row["missing_shards"] = v.get("missing_shards")
        out_variants.append(row)
    # MoE detection + expert/non-expert byte split of the EFFECTIVE quant
    # (spill.gguf_moe_detail — cached per file, so this rides the same
    # discovery/enrichment reads effective_bytes does at no recurring cost).
    # Central feasibility uses non_expert_bytes as the GPU-side need under the
    # MoE-split plan; a dense model or any read failure simply omits the key.
    moe = None
    try:
        moe_path = eff_full
        if not moe_path and eff_variant:
            moe_path = os.path.join(model_dir, eff_variant["members"][0])
        if moe_path and os.path.isfile(moe_path):
            from hugpy_engine.spill import gguf_moe_detail
            d_moe = gguf_moe_detail(moe_path)
            if d_moe.get("is_moe"):
                # Compact wire view (the per-layer map stays worker-side in the
                # spill cache; central sizing needs only the byte totals).
                moe = {"is_moe": True,
                       "expert_count": d_moe.get("expert_count"),
                       "expert_used_count": d_moe.get("expert_used_count"),
                       "sparsity": d_moe.get("sparsity"),
                       "expert_bytes": d_moe.get("expert_bytes"),
                       "non_expert_bytes": d_moe.get("non_expert_bytes")}
    except Exception:  # noqa: BLE001 — MoE detail is additive; never break sizing
        moe = None
    return {
        **({"moe": moe} if moe else {}),
        "variants": out_variants,                   # [{filename, bytes, is_effective}]
        "mmproj_bytes": mmproj,
        "effective_gguf": eff,
        "effective_quant_bytes": eff_quant,
        # What the model actually is on disk when served: the one quant (SUMMED
        # across shards for a split GGUF) + its projector (0 for text-only). None
        # only when no effective variant can be resolved among many — caller
        # keeps its own size.
        "effective_bytes": (eff_quant + mmproj) if eff_quant else None,
    }


# ── fit-aware quant pick + the ONE shared load-requirement computation ───────
_GIB = float(2 ** 30)


def select_fitting_gguf(model_key: str, model_dir: str, cfg=None,
                        budget_bytes=None, headroom: float = 1.15):
    """The "most amicable" quant for a byte budget: the LARGEST COMPLETE variant
    whose ``(bytes + mmproj) × headroom`` fits ``budget_bytes``. Returns the
    winner's entrypoint basename, or None when nothing fits (caller falls back
    to the plain election — the worker still loads-and-spills as today).

    Selection only — it never overrides a designation; callers gate it behind
    :func:`autofit_gguf_prefer`, which returns None whenever any pin exists.
    ``budget_bytes`` is the caller's already-derated figure (free VRAM minus the
    admission reserve minus the KV estimate)."""
    try:
        budget = int(budget_bytes)
    except (TypeError, ValueError):
        return None
    if budget <= 0:
        return None
    detail = gguf_variants_detail(model_key, model_dir, cfg) or {}
    mmproj = int(detail.get("mmproj_bytes") or 0)
    best = None
    for v in detail.get("variants") or []:
        if v.get("complete") is False:
            continue                      # incomplete shard sets are unelectable
        b = int(v.get("bytes") or 0)
        if b <= 0 or int((b + mmproj) * headroom) > budget:
            continue
        if best is None or b > int(best.get("bytes") or 0):
            best = v
    return best["filename"] if best else None


# Worker-side fit-aware auto-pick, registered by the worker agent (which owns
# the free-VRAM / KV / reserve numbers). Unregistered — central, tests, a bare
# import — means no auto-pick, i.e. exactly the plain election.
_GGUF_AUTOFIT_HOOK = None


def set_gguf_autofit_hook(fn) -> None:
    """Register ``fn(model_key, model_dir, cfg) -> basename | None`` as the
    fit-aware quant picker (the worker agent's selector). None un-registers."""
    global _GGUF_AUTOFIT_HOOK
    _GGUF_AUTOFIT_HOOK = fn


def autofit_gguf_prefer(model_key: str, model_dir: str, cfg=None):
    """The fit-aware auto-pick basename for a load, or None. GATED so it can
    NEVER silently upgrade an operator pin: any designation — per-worker
    ``gguf_file_by_worker`` (this box), model-wide ``gguf_file``, or the
    registry ``cfg.filename`` — returns None, and the designation resolves via
    its own path. Safe to pass straight to ``get_gguf_file(prefer=...)``."""
    try:
        ov = get_override(model_key) or {}
        if ov.get("gguf_file"):
            return None
        if resolve_gguf_for_worker(ov.get("gguf_file_by_worker"),
                                   _local_worker_forms()):
            return None
        fname = (cfg or {}).get("filename") if isinstance(cfg, dict) \
            else getattr(cfg, "filename", None)
        if fname:
            return None
        if _GGUF_AUTOFIT_HOOK is None:
            return None
        return _GGUF_AUTOFIT_HOOK(model_key, model_dir, cfg)
    except Exception:  # noqa: BLE001 — an auto-pick miss must never break a load
        return None


def effective_load_requirement(model_key: str, model_dir: str, cfg=None, *,
                               free_vram=None, worker=None) -> dict:
    """THE shared answer to "what does loading this model actually require?" —
    ``{weights_bytes, gpu_bytes, cpu_bytes, chosen_gguf, moe, basis,
    mmap_eligible}``. One computation for the dispatch disk detail, the worker
    fit-guard and central admission, so no path re-invents (and mis-invents)
    the model's size.

      * ``weights_bytes`` — the artifact that LOADS: for GGUF the designated/
        elected quant SUMMED across shards + its mmproj (via
        :func:`gguf_variants_detail`, one cached read), never the multi-quant
        dir sum. Non-GGUF: the weight-file sum, minus duplicate torch
        serializations when a safetensors set shadows them.
      * ``gpu_bytes``  — the GPU-resident share: MoE non_expert + mmproj under
        the expert split, else ``weights_bytes``.
      * ``cpu_bytes``  — the MoE expert share (0 for dense). ``mmap_eligible``
        is True: file-backed weights stream through the page cache, so this is
        NOT a hard RAM reservation and admission must not price it as one.
      * ``basis``      — short human string naming what was priced, for honest
        refusal text.

    ``free_vram`` (optional, already net of reserve/KV): with NO designation,
    the chosen quant becomes the largest complete variant fitting that budget
    (:func:`select_fitting_gguf`), falling back to the election. ``worker``:
    id/name spellings for per-worker pin resolution (None = this process's own
    identity). Pure and cheap — no I/O beyond gguf_variants_detail / one dir
    walk for non-GGUF."""
    out = {"weights_bytes": None, "gpu_bytes": None, "cpu_bytes": 0,
           "chosen_gguf": None, "moe": None, "basis": None,
           "mmap_eligible": True}
    detail = {}
    try:
        detail = gguf_variants_detail(model_key, model_dir, cfg) or {}
    except Exception:  # noqa: BLE001 — sizing degrades, never raises
        detail = {}
    if detail:
        mmproj = int(detail.get("mmproj_bytes") or 0)
        chosen = detail.get("effective_gguf")
        chosen_bytes = int(detail.get("effective_quant_bytes") or 0)
        how = "elected"
        try:
            ov = get_override(model_key) or {}
            forms = ([worker] if isinstance(worker, str)
                     else list(worker or ())) or _local_worker_forms()
            pin = resolve_gguf_for_worker(ov.get("gguf_file_by_worker"), forms)
            if pin:
                how = "worker pin"
            elif ov.get("gguf_file"):
                pin, how = ov.get("gguf_file"), "pinned"
            else:
                fname = (cfg or {}).get("filename") if isinstance(cfg, dict) \
                    else getattr(cfg, "filename", None)
                if fname and chosen:
                    how = "designated"
            if pin:
                want = os.path.basename(str(pin)).lower()
                for v in detail.get("variants") or []:
                    if want == v["filename"].lower() or want in v["filename"].lower():
                        chosen, chosen_bytes = v["filename"], int(v.get("bytes") or 0)
                        break
            elif how == "elected" and free_vram is not None:
                fitted = select_fitting_gguf(model_key, model_dir, cfg,
                                             budget_bytes=free_vram)
                if fitted:
                    for v in detail.get("variants") or []:
                        if v["filename"] == fitted:
                            chosen, chosen_bytes = fitted, int(v.get("bytes") or 0)
                            how = "auto-fit"
                            break
        except Exception:  # noqa: BLE001 — pin resolution is best-effort
            pass
        if not chosen or not chosen_bytes:
            return out
        weights = chosen_bytes + mmproj
        out["weights_bytes"] = weights
        out["chosen_gguf"] = chosen
        out["gpu_bytes"] = weights
        out["basis"] = f"{chosen} ({how}) {weights / _GIB:.1f} GiB"
        # The MoE detail rides the EFFECTIVE variant's header read; it only
        # describes the chosen quant when the two are the same file.
        moe = detail.get("moe")
        if (isinstance(moe, dict) and moe.get("is_moe")
                and chosen == detail.get("effective_gguf")):
            nexp = int(moe.get("non_expert_bytes") or 0)
            exp = int(moe.get("expert_bytes") or 0)
            if nexp:
                out["moe"] = moe
                out["gpu_bytes"] = nexp + mmproj
                out["cpu_bytes"] = exp
                out["basis"] = (f"MoE split: {(nexp + mmproj) / _GIB:.1f} GiB "
                                f"GPU-resident + {exp / _GIB:.1f} GiB experts "
                                f"(mmap)")
        return out
    # Non-GGUF: the existing per-framework logic — sum the weight files, but a
    # repo shipping BOTH a safetensors set and its torch twin (.bin/.pt/.pth)
    # loads only one of them, so the duplicate serialization is excluded.
    st = torchy = other = 0
    try:
        for root, _dirs, files in os.walk(model_dir):
            for fn in files:
                low = fn.lower()
                try:
                    sz = int(os.path.getsize(os.path.join(root, fn)))
                except OSError:
                    continue
                if low.endswith(".safetensors"):
                    st += sz
                elif low.endswith((".bin", ".pt", ".pth")):
                    torchy += sz
                elif low.endswith((".ckpt", ".onnx")):
                    other += sz
    except OSError:
        return out
    if st and torchy:
        weights = st + other
        basis_note = " (duplicate torch weights excluded)"
    else:
        weights = st + torchy + other
        basis_note = ""
    if weights:
        out["weights_bytes"] = weights
        out["gpu_bytes"] = weights
        out["basis"] = f"weights {weights / _GIB:.1f} GiB{basis_note}"
    return out


def _truthy(value) -> bool:
    """The one on/off reading for the polite levers — a JSON bool from the
    console and the "yes"/"on"/"1" spellings a curl or a hand-edited file uses
    must mean the same thing on both."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _coerce(field: str, value):
    if value is None or value == "":
        return None  # signals "clear this field"
    if field == "alloc_mode":
        # Legacy names (autofit/cpu-only/...) resolve to canonical on WRITE, so
        # the stored value is always one of the five flat modes — accepted on
        # input, never stored/emitted back. Unknown -> clear + log (degrade,
        # never 500 a serving-settings POST over a typo).
        from hugpy_engine.alloc_modes import resolve_alloc_mode, ALLOC_MODES
        canonical, was_alias = resolve_alloc_mode(value)
        if canonical is None:
            logger.warning("ignoring unknown alloc_mode %r (recognized: %s)",
                           value, ", ".join(ALLOC_MODES))
            return None
        if was_alias:
            logger.info("alloc_mode legacy name %r stored as %r", value, canonical)
        return canonical
    if field == "priority_device":
        v = str(value).strip().lower()
        if v not in ("gpu", "ram"):
            logger.warning("ignoring unknown priority_device %r (gpu|ram)", value)
            return None
        return v
    if field == "worker_prefs":
        # Accept a list (the console) or a comma string (curl/scripts). Order is
        # the whole point, so dedupe PRESERVING first position; an empty result
        # clears the key, which is what "no preference" must look like on disk.
        raw = value.split(",") if isinstance(value, str) else list(value or [])
        out, seen = [], set()
        for item in raw:
            name = str(item).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            out.append(name)
        return out or None
    if field == "no_evict":
        # OFF removes the key rather than storing false (same discipline as the
        # bnb lever): the file then holds only real operator opt-ins, and an
        # absent entry unambiguously means the normal declare-need-then-evict.
        return True if _truthy(value) else None
    if field == "strict":
        # k-dist (operator ruling 2026-09-24): same OFF-clears discipline as
        # no_evict above — the file holds only real fences, and an absent entry
        # unambiguously means "follow the fleet distribution default". The
        # PlacementControl posts strict on every save, so a cleared toggle drops
        # the key rather than littering the store with strict:false rows.
        return True if _truthy(value) else None
    if field == "no_evict_by_worker":
        # k62. Accept the console's {"ae": true} or a curl/script string
        # ("ae=yes,computron=no"). Name handling mirrors worker_prefs — strip,
        # drop blanks, dedupe case-insensitively keeping the FIRST spelling —
        # because both keys are matched against the same worker id/name forms
        # and two spellings of one box would be two different verdicts.
        #
        # Unlike the model-wide boolean, ``false`` is STORED here: an explicit
        # no is how a worker opts OUT of a polite default, which is a different
        # statement from "unset, follow the default". Removing the entry is how
        # you go back to the default; an empty map clears the key entirely.
        if isinstance(value, str):
            items = []
            for part in value.split(","):
                if not part.strip():
                    continue
                name, sep, val = part.partition("=")
                if not sep:
                    name, sep, val = part.partition(":")
                items.append((name, val if sep else "yes"))
        elif isinstance(value, dict):
            items = list(value.items())
        else:
            logger.warning("ignoring no_evict_by_worker of type %s (want a map)",
                           type(value).__name__)
            return None
        out, seen = {}, set()
        for name, val in items:
            name = str(name).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            out[name] = _truthy(val)
        return out or None
    if field == "gguf_file_by_worker":
        # Per-worker quant pin — same accepted shapes and name discipline as
        # no_evict_by_worker (the console's {"ae": "q8_0"} or a curl string
        # "ae=q8_0,computron=m.q4_k_m.gguf"), except the VALUE is a basename or
        # quant token rather than a boolean. An empty value drops that worker's
        # entry (back to the model-wide pin / election); an empty map clears
        # the key entirely.
        if isinstance(value, str):
            items = []
            for part in value.split(","):
                if not part.strip():
                    continue
                name, sep, val = part.partition("=")
                if not sep:
                    name, sep, val = part.partition(":")
                items.append((name, val if sep else ""))
        elif isinstance(value, dict):
            items = list(value.items())
        else:
            logger.warning("ignoring gguf_file_by_worker of type %s (want a map)",
                           type(value).__name__)
            return None
        out, seen = {}, set()
        for name, val in items:
            name = str(name).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            val = os.path.basename(str(val or "").strip())
            if val:
                out[name] = val
        return out or None
    if field in _INT_FIELDS:
        return int(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    if field in _BOOL_FIELDS:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return str(value)


def set_override(model_key: str, fields: dict) -> dict:
    """Merge ``fields`` into the model's override; a None/"" value clears a key.

    Returns the model's full override after the update.
    """
    with _LOCK:
        data = _load()
        current = dict(data.get(model_key, {}) or {})
        for key, raw in (fields or {}).items():
            if key not in ALLOWED_FIELDS:
                continue
            coerced = _coerce(key, raw)
            if coerced is None:
                current.pop(key, None)
            else:
                current[key] = coerced
        if current:
            data[model_key] = current
        else:
            data.pop(model_key, None)
        _save(data)
    # ``gguf_file`` picks WHICH quant serves, so it changes effective_bytes /
    # effective_gguf / mmproj_bytes / moe — the persisted size half. It is an
    # operator choice, NOT part of the model's routing identity, so nothing else
    # would notice: drop this model's physical record so the next listing
    # re-derives the size of the quant the operator just chose. Targeted and
    # guarded — a settings write must never fail over a cache.
    if "gguf_file" in (fields or {}) or "gguf_file_by_worker" in (fields or {}):
        try:
            from hugpy_storage.model_physical import forget_physical
            forget_physical(model_key, "gguf_file override changed")
        except Exception as exc:  # noqa: BLE001
            logger.debug("physical-state drop after override skipped: %s", exc)
    return current


def placement_prefs(model_key: str) -> tuple:
    """k56: ``(ordered worker preference list, polite)`` for a model.

    THE one reader for both flags, so routing, the warm gate, the emission seam
    and the console can never disagree about what was set. Totally guarded — an
    unreadable overrides file degrades to ``([], False)``, which is the
    pre-k56 behaviour exactly (no list, ordinary declare-need-then-evict).

    ~-TOLERANT: routing reaches this with whatever spelling the caller used,
    while the console writes the registry key. A placement the operator set
    must not go silently un-applied because one side said "Qwen~X" and the
    other "X" (the k30 class of invisible mismatch), so an exact miss falls
    back to a bare-key, case-insensitive match.

    k62: the polite half of this tuple is the ALL-WORKERS DEFAULT. A caller that
    is deciding about ONE worker must use :func:`placement_policy` /
    :func:`polite_on_worker` instead — see them for why."""
    prefs, polite, _by_worker = placement_policy(model_key)
    return prefs, polite


def placement_policy(model_key: str) -> tuple:
    """k62: ``(ordered worker preference, model-wide polite, per-worker polite)``.

    The full placement statement, and the superset :func:`placement_prefs`
    returns the first two of. The third element is the ``no_evict_by_worker``
    map: politeness individualized per (model × worker), because contention is
    a property of a BOX, not of a model — flux2 is polite on ae's contended
    3090 and keeps ordinary eviction rights on computron.

    Same total guarding as placement_prefs: an unreadable overrides file
    degrades to ``([], False, {})``, which is pre-k56 behaviour exactly.

    GROUP FALLBACK (2026-08-25): when the model carries NO per-model
    ``worker_prefs``, the ordered ``workers`` allocation of the enabled
    priority group claiming it (``comms.priority_groups.workers_for_key``)
    supplies the preference instead. Per-model prefs OUTRANK the group's — an
    order written on the model itself is the more specific statement — and a
    model in no group (or a group with no ``workers``) resolves ``[]`` exactly
    as before. Politeness stays per-model in both cases: the group states
    WHERE, never eviction manners."""
    try:
        ov = get_override(model_key) or {}
        if not ov:
            want = _bare_key(str(model_key)).lower()
            for k, row in (_load() or {}).items():
                if _bare_key(str(k)).lower() == want and isinstance(row, dict):
                    ov = row
                    break
    except Exception:  # noqa: BLE001 — placement must never break over a read
        return [], False, {}
    prefs = ov.get("worker_prefs")
    prefs = [str(w) for w in prefs if str(w).strip()] if isinstance(prefs, list) else []
    if not prefs:
        try:
            from hugpy_engine.placement import get_priority_groups
            prefs = list(get_priority_groups().workers_for_key(model_key))
        except Exception:  # noqa: BLE001 — the group half must never break placement
            prefs = []
    by_worker = ov.get("no_evict_by_worker")
    by_worker = ({str(k): bool(v) for k, v in by_worker.items() if str(k).strip()}
                 if isinstance(by_worker, dict) else {})
    return prefs, bool(ov.get("no_evict")), by_worker


def model_strict(model_key: str) -> bool:
    """k-dist: does ``model_key`` keep the HARD designation fence even under the
    fleet "feasible" distribution default?

    True when the per-model override sets ``strict: true`` OR the enabled
    priority group claiming it declares the group strict. Absent -> False (the
    preference is an ORDER with a feasible-set fallback). Total guarding: an
    unreadable override / group layer degrades to False, which is the new-default
    behaviour, never a surprise refusal."""
    try:
        ov = get_override(model_key) or {}
        if not ov:
            want = _bare_key(str(model_key)).lower()
            for k, row in (_load() or {}).items():
                if _bare_key(str(k)).lower() == want and isinstance(row, dict):
                    ov = row
                    break
        if bool(ov.get("strict")):
            return True
    except Exception:  # noqa: BLE001 — placement must never break over a read
        return False
    try:
        from hugpy_engine.placement import get_priority_groups
        gs = getattr(get_priority_groups(), "strict_for_key", None)
        if callable(gs):
            return bool(gs(model_key))
    except Exception:  # noqa: BLE001 — the group half must never break placement
        pass
    return False


def resolve_polite(polite: bool, by_worker: dict, forms) -> bool:
    """Effective politeness for ONE worker: ``map[W]`` when W is in the map,
    else the model-wide boolean. PURE (no disk read) so the routing loop resolves
    every candidate off a single policy read.

    ``forms``: the worker's id/name spellings, matched case-insensitively — the
    same tolerance ``_pref_index`` applies, and for the same reason: the console
    posts ids, an operator editing the file writes names, and a politeness that
    silently failed to match would evict on a box the operator marked polite."""
    want = {str(f).strip().lower() for f in (forms or []) if str(f).strip()}
    for name, val in (by_worker or {}).items():
        if str(name).strip().lower() in want:
            return bool(val)
    return bool(polite)


def polite_on_worker(model_key: str, *forms) -> bool:
    """Convenience: is ``model_key`` polite on the worker named by ``forms``
    (any mix of id/name spellings)? One-shot readers (the warm gate, the spill
    emission) use this; the routing loop reads the policy once and calls
    :func:`resolve_polite` per candidate."""
    _prefs, polite, by_worker = placement_policy(model_key)
    return resolve_polite(polite, by_worker, forms)


def effective_alloc_mode(model_key: str) -> str:
    """The model's EFFECTIVE allocation mode (k37) — the persisted
    ``alloc_mode`` when set, else READ-TIME DERIVATION from the legacy knobs
    (n_gpu_layers -1 -> gpu-only, 0/"off" -> ram-only, explicit budgets/bands
    -> explicit, unset -> max-gpu). Derivation IS the migration: no override
    file is ever rewritten for the rename, and a blank model reads max-gpu
    (fit-and-spill, never OOM — defaults-are-promises)."""
    from hugpy_engine.alloc_modes import derive_alloc_mode
    return derive_alloc_mode(get_override(model_key))


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(_OVERRIDES_PATH) or ".", exist_ok=True)
    tmp = _OVERRIDES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, _OVERRIDES_PATH)


def _bare_key(k: str) -> str:
    """Model name without the 'owner~' collision qualifier."""
    return k.split("~", 1)[1] if "~" in k else k


def migrate_overrides(registry) -> dict:
    """Heal overrides orphaned when a model key got collision-qualified
    (``name`` -> ``owner~name``). For each override key absent from the registry,
    re-key it to the qualified key sharing its bare suffix. If several owners
    match (a real collision), disambiguate by the override's ``gguf_file`` vs each
    variant's ``filename``; skip + log when still ambiguous so a human picks the
    owner. Idempotent; never clobbers an existing override on the target.

    ``registry``: ``{model_key: cfg-dict-or-ModelConfig}``. Returns ``{old: new}``.
    """
    moved: dict = {}
    with _LOCK:
        ov = _load()
        if not ov:
            return moved
        keys = list(registry)

        def _fname(k):
            cfg = registry.get(k)
            fn = cfg.get("filename") if isinstance(cfg, dict) else getattr(cfg, "filename", None)
            return os.path.basename(str(fn or ""))

        for okey in list(ov):
            if okey in registry:
                continue                                   # still a valid key
            cands = [k for k in keys if _bare_key(k) == okey]
            if not cands:
                continue                                   # model gone — leave override
            target = cands[0] if len(cands) == 1 else None
            if target is None:                             # multi-owner: use gguf_file hint
                gf = os.path.basename(str((ov[okey] or {}).get("gguf_file") or ""))
                matched = [k for k in cands if gf and _fname(k) == gf]
                target = matched[0] if len(matched) == 1 else None
            if target and target not in ov:
                ov[target] = ov.pop(okey)
                moved[okey] = target
                logger.info("serve override migrated: %r -> %r", okey, target)
            elif not target:
                logger.warning("serve override %r orphaned + ambiguous across %s; "
                               "re-key manually", okey, cands)
        if moved:
            _save(ov)
    return moved


def migrate_worker_tokens(resolve) -> dict:
    """Rewrite STALE worker-name references in the placement overrides to a stable
    worker id (operator incident 2026-09-25).

    Placement is stored by NAME in three per-model fields — ``worker_prefs`` (a
    list), and the ``no_evict_by_worker`` / ``gguf_file_by_worker`` maps (keyed by
    worker) — so a worker rename ("aeb" -> "ae-worker", same id) strands every
    token written under the old name and central logs "ordered worker preference
    ['aeb'] but NONE of them is an eligible candidate" every few minutes.

    ``resolve(token) -> worker_id | None`` returns the id ONLY for a token that
    does NOT already resolve to a live worker but DOES match a worker's recorded
    former name (the caller builds it from the registry). A None leaves the token
    untouched (it either already resolves or is genuinely orphaned — the honest
    route-time warning still fires). Returns ``{model_key: [(field, old, new)]}``.
    Idempotent: a rewritten token is an id, which never matches a former name."""
    changed: dict = {}
    with _LOCK:
        data = _load()
        if not data:
            return changed
        for mk, ov in list(data.items()):
            if not isinstance(ov, dict):
                continue
            rows = []
            prefs = ov.get("worker_prefs")
            if isinstance(prefs, list):
                new_prefs, seen = [], set()
                for tok in prefs:
                    new = resolve(tok)
                    use = new or str(tok)
                    if new:
                        rows.append(("worker_prefs", str(tok), new))
                    if use.lower() not in seen:
                        seen.add(use.lower())
                        new_prefs.append(use)
                if new_prefs != prefs:
                    ov["worker_prefs"] = new_prefs
            for field in ("no_evict_by_worker", "gguf_file_by_worker"):
                m = ov.get(field)
                if not isinstance(m, dict):
                    continue
                new_map = {}
                for tok, val in m.items():
                    new = resolve(tok)
                    key = new or str(tok)
                    if new:
                        rows.append((field, str(tok), new))
                    new_map[key] = val
                if new_map != m:
                    ov[field] = new_map
            if rows:
                changed[mk] = rows
                for field, old, new in rows:
                    logger.info("placement override %s: %s %r -> worker id %r",
                                mk, field, old, new)
        if changed:
            _save(data)
    return changed
