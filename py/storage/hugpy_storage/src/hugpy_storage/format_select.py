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
    return keep


def effective_bytes(
    files: Iterable[Tuple[str, int]],
    *,
    framework: Optional[str] = None,
) -> int:
    """Sum of the single-format effective file set — the honest ledger size."""
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
