"""Single-format weight selection for transformers/diffusers model dirs.

Central mirrors WHOLE HuggingFace snapshots: a repo routinely ships the same
weights in five formats (safetensors + pytorch bin + TF h5 + Flax msgpack +
ONNX/OpenVINO/CoreML/rust), often BOTH fp16 and an fp32 duplicate. Shipping all
of them to a worker wastes disk (op held flan-t5-xl in FOUR formats on a headless
Linux box), and *ledgering* the whole-dir sum makes an ~11GB model read as 45GB —
the dishonest byte accounting that caused the 2026-07-16 operator scare.

This module answers, for one on-disk model directory, "which files would a worker
ACTUALLY hold to serve this model?" — ONE usable weight format plus every sidecar
(config / tokenizer / processor / pooling …). It is the transformers analogue of
``gguf_variants_detail``'s effective-quant selection (which already solved this for
GGUF, 0.1.151): compute the effective file set ONCE, centrally.

Design doctrine — DEGRADE TO CORRECT, NEVER TO BROKEN:
  * A worker missing a needed sidecar is a broken model; an extra format is only
    wasted disk. So every ambiguity resolves toward INCLUDING the file.
  * We only ever exclude a redundant weight format when we can POSITIVELY see a
    complete usable format remains. If we cannot positively identify a complete
    keep-format, ``select_files`` returns the whole listing unchanged (the
    pre-feature behavior).
  * GGUF/llama_cpp dirs are NOT touched here — their effective size is resolved by
    the existing gguf_variants_detail path. This module is a no-op for them.

The functions are PURE over a file listing (list of ``(relpath, size)``), so the
central ``/manifest`` and ``/archive`` routes, the storage annotators, and the
unit tests all share one implementation with no filesystem coupling.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── weight-format classification ────────────────────────────────────────────
# Redundant framework formats a Linux GPU worker never loads when a torch-usable
# format is present. Matched on the BASENAME (case-insensitive). These are the
# always-safe-to-drop-IF-a-keep-format-exists set. rust_model.ot / tf / flax /
# h5 / msgpack are alternate-framework serializations of the SAME weights.
_ALT_FRAMEWORK_SUFFIXES = (".h5", ".msgpack", ".ot")
_ALT_FRAMEWORK_PREFIXES = ("tf_model", "flax_model")

# Subdirectories that hold an ENTIRELY separate (non-torch) export of the model.
# A relpath whose first path component (case-insensitive) is one of these is an
# alternate runtime export, never used by the transformers/torch loader.
_ALT_RUNTIME_DIRS = frozenset({"onnx", "openvino", "coreml", "tflite", "tensorrt"})

_SAFETENSORS_RE = re.compile(r"(?i)(?:^|[/\\])(?:.*?)\.safetensors$")
# fp32 shard markers embedded in the filename, e.g.
#   model.fp32-00001-of-00002.safetensors
#   pytorch_model.fp32-00001-of-00002.bin
#   model.safetensors.index.fp32.json
_FP32_TAG_RE = re.compile(r"(?i)(?:[.\-_]fp32|\.fp32)(?=[.\-_]|$)")


def _basename(rel: str) -> str:
    return rel.replace("\\", "/").rsplit("/", 1)[-1]


def _first_component(rel: str) -> str:
    return rel.replace("\\", "/").split("/", 1)[0]


def _is_safetensors(rel: str) -> bool:
    return _basename(rel).lower().endswith(".safetensors")


def _is_pytorch_bin(rel: str) -> bool:
    b = _basename(rel).lower()
    # pytorch_model.bin, pytorch_model-00001-of-00002.bin, and the generic .bin
    # weight files diffusers/transformers write. NOT tokenizer .model (sentencepiece).
    return b.endswith(".bin")


def _is_alt_framework(rel: str) -> bool:
    b = _basename(rel).lower()
    if b.endswith(_ALT_FRAMEWORK_SUFFIXES):
        return True
    if b.startswith(_ALT_FRAMEWORK_PREFIXES):
        return True
    return False


def _in_alt_runtime_dir(rel: str) -> bool:
    return _first_component(rel).lower() in _ALT_RUNTIME_DIRS


def _is_fp32_duplicate(rel: str) -> bool:
    """A weight/index file tagged fp32 — a full-precision DUPLICATE of the fp16
    weights. Only ever dropped when the non-fp32 counterpart is complete."""
    b = _basename(rel)
    return bool(_FP32_TAG_RE.search(b)) and (
        _is_safetensors(rel) or _is_pytorch_bin(rel) or b.lower().endswith(".json"))


def _index_json_for(weight_kind: str, rel: str) -> bool:
    """True if rel is the shard-index JSON belonging to weight_kind
    ('safetensors' or 'bin')."""
    b = _basename(rel).lower()
    if weight_kind == "safetensors":
        return b == "model.safetensors.index.json"
    if weight_kind == "bin":
        return b == "pytorch_model.bin.index.json"
    return False


# ── precision-variant selection (diffusers) ─────────────────────────────────
# A diffusers pipeline is NOT a flat repo — it is per-component subdirs
# (unet/ vae/ text_encoder/ …), and each component routinely ships the SAME
# weights at two precisions:
#     unet/diffusion_pytorch_model.safetensors        (default — full precision)
#     unet/diffusion_pytorch_model.fp16.safetensors   (the fp16 variant)
# A GPU worker loads ONE variant (torch_dtype float16 / variant="fp16"), so
# summing both is the same double-count that made sd-turbo read as 12.07 GiB
# (operator incident 2026-09-25) when its fp16 footprint is ~2.5 GiB — and it
# CUDA-OOM'd after being mis-sized to ram-only. Unlike the transformers layout
# above (where the UNTAGGED file IS the fp16 that serves and the redundant copy
# carries an explicit ``.fp32`` tag), diffusers tags the SMALL copy (``.fp16``)
# and leaves the full-precision copy untagged — so the fp32-tag rule cannot see
# it. This selects the loader's preferred precision per component instead.
#
# The precision the loader prefers, low rank = kept first: fp16 < bf16 < fp8 <
# untagged (native/default) < fp32 (explicit full-precision duplicate).
_VARIANT_RANK = {
    "fp16": 0, "float16": 0, "f16": 0,
    "bf16": 1, "bfloat16": 1,
    "fp8": 2, "float8": 2, "f8": 2,
    "fp32": 4, "float32": 4, "f32": 4,
}
_UNTAGGED_RANK = 3
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$")


def _weight_meta(rel: str):
    """``(dir, base, fmt, rank, shard)`` for a torch weight file, else None.

    ``base`` has the format extension, the shard suffix and any precision-variant
    segment stripped, so the two precisions of one component group together.
    ``rank`` is the load preference (see _VARIANT_RANK). ``shard`` is ``(part,
    total)`` for a sharded file, else None."""
    b = _basename(rel)
    low = b.lower()
    if low.endswith(".safetensors"):
        fmt, stem = "safetensors", b[: -len(".safetensors")]
    elif low.endswith(".bin"):
        fmt, stem = "bin", b[: -len(".bin")]
    else:
        return None
    shard = None
    m = _SHARD_RE.search(stem)
    if m:
        shard = (int(m.group(1)), int(m.group(2)))
        stem = stem[: m.start()]
    # A precision variant is a dot-delimited segment of the stem (diffusers'
    # ``diffusion_pytorch_model.fp16`` convention). Split on '.' so we never
    # mistake a substring inside a component name for a precision tag.
    segs = stem.split(".")
    rank = _UNTAGGED_RANK
    kept_segs = []
    for seg in segs:
        r = _VARIANT_RANK.get(seg.lower())
        if r is not None and rank == _UNTAGGED_RANK:
            rank = r          # first recognised precision segment wins
            continue          # drop it from the base name
        kept_segs.append(seg)
    base = ".".join(kept_segs).lower()
    d = rel.replace("\\", "/").rsplit("/", 1)
    dirpart = d[0] if len(d) > 1 else ""
    return dirpart, base, fmt, rank, shard


def _best_rank_set_complete(idxs, metas) -> bool:
    """A sharded best-rank set is only safe to keep-alone when it names all of
    its own shards (part 1..total present). An unsharded winner is complete by
    itself. Degrade-to-correct: an incomplete winner means we do NOT drop the
    other precision — we keep everything rather than risk an unloadable model."""
    shards = [metas[i][4] for i in idxs if metas[i][4] is not None]
    if not shards:
        return True
    totals = {t for (_p, t) in shards}
    if len(totals) != 1:
        return False
    total = next(iter(totals))
    return {p for (p, _t) in shards} == set(range(1, total + 1))


def _dedup_precision_variants(
    items: List[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """Drop the redundant PRECISION copy of a weight when a preferred one is
    present in the same component (see the variant block comment). Grouped by
    (dir, base, format) so unet's two precisions dedup while unet vs vae — and
    safetensors vs bin — never collide. Conservative: only drops within a group
    that holds two precisions AND whose winning precision is a complete set."""
    metas = {i: _weight_meta(r) for i, (r, _s) in enumerate(items)}
    groups: dict = {}
    for i, m in metas.items():
        if m is None:
            continue
        groups.setdefault((m[0], m[1], m[2]), []).append(i)
    drop = set()
    for idxs in groups.values():
        ranks = [metas[i][3] for i in idxs]
        best = min(ranks)
        if best == max(ranks):
            continue                      # one precision present — nothing to do
        winners = [i for i in idxs if metas[i][3] == best]
        if not _best_rank_set_complete(winners, metas):
            continue                      # winner incomplete — keep everything
        for i in idxs:
            if metas[i][3] != best:
                drop.add(i)
    if not drop:
        return items
    return [it for i, it in enumerate(items) if i not in drop]


def _has_complete_safetensors(rels: Iterable[str]) -> bool:
    """A COMPLETE non-fp32 safetensors weight set is present.

    Complete means either a single ``*.safetensors`` weight file, or a sharded
    set accompanied by its ``model.safetensors.index.json`` (the map the loader
    needs to assemble the shards). fp32-tagged safetensors do NOT count — they
    are the redundant duplicate we want to be able to drop."""
    st = [r for r in rels if _is_safetensors(r) and not _is_fp32_duplicate(r)
          and not _in_alt_runtime_dir(r)]
    if not st:
        return False
    # A sharded set names files like model-00001-of-00002.safetensors and MUST
    # have its index; a single unsharded model.safetensors is complete alone.
    sharded = any(re.search(r"-\d{5}-of-\d{5}\.safetensors$", _basename(r), re.I)
                  for r in st)
    if not sharded:
        return True
    has_index = any(_basename(r).lower() == "model.safetensors.index.json"
                    for r in rels)
    return has_index


def select_files(
    files: Iterable[Tuple[str, int]],
    *,
    framework: Optional[str] = None,
) -> List[Tuple[str, int]]:
    """Return the single-format effective file set for a model directory listing.

    ``files`` is an iterable of ``(relpath, size_bytes)``. Returns the same shape,
    filtered to ONE usable weight format + all sidecars. Order is preserved.

    Rules (conservative — an unrecognized file is always KEPT):
      1. GGUF/llama_cpp framework  -> return the listing unchanged (handled by the
         effective-quant path elsewhere; never second-guess it here).
      2. Drop alternate-framework serializations (tf_model*, flax_model*, *.h5,
         *.msgpack, rust_model.ot) and alternate-runtime export subdirs
         (onnx/ openvino/ coreml/ tflite/ tensorrt/) — but ONLY when a complete
         torch-usable weight format (safetensors OR pytorch bin) survives.
      3. If a COMPLETE non-fp32 safetensors set exists, also drop the pytorch
         ``*.bin`` weights + their index and the fp32 duplicates.
      4. If NO complete safetensors exists, keep the pytorch bins (they are then
         the serving format) and still drop the alt-framework/alt-runtime copies.
      5. Anything not positively classified as a redundant weight — configs,
         tokenizers, processors, pooling dirs, .pt, unknown files — is KEPT.

    If step 1 doesn't apply and NO torch-usable weight format can be positively
    identified as complete, the WHOLE listing is returned (degrade to correct):
    we will not risk shipping a folder that can't load to save disk.
    """
    items = [(r, s) for (r, s) in files]
    if str(framework or "").lower() in ("gguf", "llama_cpp"):
        return items

    rels = [r for (r, _s) in items]

    have_safetensors = _has_complete_safetensors(rels)
    have_bin = any(_is_pytorch_bin(r) and not _is_fp32_duplicate(r)
                   and not _in_alt_runtime_dir(r) for r in rels)

    # No positively-complete torch format we can stand on -> ship everything.
    # (have_bin is a weaker signal than a verified-complete safetensors set, but
    # a present pytorch_model.bin is the historical always-loadable case; if even
    # that is absent we've identified no keep-format and must not prune.)
    if not have_safetensors and not have_bin:
        return items

    keep: List[Tuple[str, int]] = []
    for rel, size in items:
        # Alt-runtime export subdirs: redundant whenever we have a torch format.
        if _in_alt_runtime_dir(rel):
            continue
        # Alt-framework serializations of the same weights: redundant likewise.
        if _is_alt_framework(rel):
            continue
        if have_safetensors:
            # safetensors is the serving format -> drop bin weights + bin index
            # + every fp32 duplicate (fp32 safetensors, fp32 bin, fp32 index).
            if _is_fp32_duplicate(rel):
                continue
            if _is_pytorch_bin(rel):
                continue
            if _index_json_for("bin", rel):
                continue
        else:
            # bin is the serving format -> only drop fp32 duplicates of it.
            if _is_fp32_duplicate(rel):
                continue
        keep.append((rel, size))
    # DIFFUSERS PRECISION DEDUP (operator incident 2026-09-25): after the flat
    # transformers pruning, drop the per-component full-precision duplicate when
    # a preferred (fp16/bf16) variant is present — the fp32-tag rule above cannot
    # see it because diffusers tags the SMALL copy, not the large one. No-op on a
    # repo carrying a single precision (every transformers layout in the suite).
    return _dedup_precision_variants(keep)


def effective_bytes(
    files: Iterable[Tuple[str, int]],
    framework: Optional[str] = None,
) -> int:
    """Sum of the single-format effective file set — the honest ledger size.

    ``framework`` is positional-or-keyword ON PURPOSE: this is the function the
    server installs as the storage footprint selector, whose contract
    (providers.Footprint) is ``(listing, framework)`` positional. It was
    keyword-only until 2026-09-23, so every selector call raised TypeError and
    console/model_physical.annotate_size recorded ``size_bytes: None`` for every
    non-GGUF model."""
    return sum(s for (_r, s) in select_files(files, framework=framework))


def walk_listing(root: str) -> List[Tuple[str, int]]:
    """Directory -> ``[(relpath, size)]``, skipping transfer-machinery sidecars.

    This is THE shared walk for /manifest and /archive (both call it directly;
    no more hand-copied mirrors) so select_files always sees the same file
    universe those routes filter.

    Never descends into a dot-directory (``.cache/``, ``.git/``, …). Those are
    HF/git bookkeeping, never servable weights — and critically, HF's own local
    cache scheme drops metadata files (``.cache/huggingface/trees/*.json``) that
    can be mode 0600 owned by whatever uid ran the download, which a
    differently-provisioned central process can enumerate but not read (live
    2026-07-18: PermissionError serving a comfy checkpoint's manifest-offered
    ``.cache`` file — see worker_routes.py's ``model_file``). Pruning the
    directory here means the file is never offered, not just tolerated.
    """
    out: List[Tuple[str, int]] = []
    if not root or not os.path.isdir(root):
        return out
    for r, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            if ".chunksums-" in name or name.endswith((".part", ".part.state.json")):
                continue
            full = os.path.join(r, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                # Unreadable entry (permission-restricted, vanished mid-walk, …):
                # degrade by skipping it, never let a single bad entry 500 the
                # whole listing.
                logger.warning("walk_listing: skipping unreadable entry %s", full)
                continue
            out.append((os.path.relpath(full, root), size))
    return out
