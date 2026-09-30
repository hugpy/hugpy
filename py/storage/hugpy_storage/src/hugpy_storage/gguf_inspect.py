"""GGUF header inspection — file-format reading with no engine semantics.

Storage owns "what is in this file": the GGUF KV table (architecture-namespaced
geometry such as ``*.block_count`` / ``*.expert_count``), the tensor-info table
and the expert/non-expert byte split a Mixture-of-Experts model carries. The
ENGINE decides what to do with those facts (how many layers to spill, whether
to run a MoE split); it imports the readers from here (``hugpy_engine.spill``
re-exports them so every existing caller keeps working).

Moved verbatim from ``hugpy_engine.spill`` (formerly ``managers/spill.py``) on
2026-09-22: ``_gguf_metadata``, ``_gguf_scan_moe``, ``_gguf_shard_paths``,
``gguf_moe_detail`` and their private helpers. Behaviour is unchanged; only the
home moved down to the storage layer so the marker writer
(``hugpy_storage.hugpy_marker.detect_moe_capable``) never reaches up into the
engine.

Public surface: :func:`gguf_metadata` (suffix-matched KV read) and
:func:`gguf_moe_detail` (the cached, shard-aware MoE reader).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("hugpy_storage.gguf_inspect")

def _gguf_metadata(model_path: str, want_suffixes: tuple) -> dict:
    """Scan a GGUF header's KV table and return the values whose keys END WITH any
    of ``want_suffixes`` (e.g. ``.block_count``, ``.attention.head_count_kv``,
    ``.embedding_length``, ``.context_length``). Best-effort; {} on any issue.

    The GGUF geometry keys are namespaced by architecture (``qwen2.block_count``,
    ``llama.attention.head_count_kv``, …), so suffix-matching is arch-agnostic —
    verified 2026-07-17 against a real Qwen2.5-Coder-3B q4 gguf:
      qwen2.block_count=36, qwen2.attention.head_count=16,
      qwen2.attention.head_count_kv=2, qwen2.embedding_length=2048,
      qwen2.context_length=32768.
    (GGUF spec: github.com/ggml-org/ggml/blob/master/docs/gguf.md — the
    general/architecture KVs are the canonical model geometry.)"""
    out: dict = {}
    try:
        import struct

        with open(model_path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return out
            version = struct.unpack("<I", fh.read(4))[0]
            if version < 2:
                return out
            struct.unpack("<Q", fh.read(8))[0]              # tensor count
            n_kv = struct.unpack("<Q", fh.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(n).decode("utf-8", "replace")

            # Minimal GGUF value-type reader — enough to scan the KV table.
            def read_val(t: int):
                simple = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i",
                          5: "<I", 6: "<f", 7: "<?", 10: "<q", 11: "<Q", 12: "<d"}
                if t in simple:
                    fmt = simple[t]
                    return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
                if t == 8:                                   # string
                    return read_str()
                if t == 9:                                   # array
                    et = struct.unpack("<I", fh.read(4))[0]
                    n = struct.unpack("<Q", fh.read(8))[0]
                    return [read_val(et) for _ in range(n)]
                raise ValueError(f"unknown gguf type {t}")

            for _ in range(n_kv):
                key = read_str()
                vtype = struct.unpack("<I", fh.read(4))[0]
                val = read_val(vtype)
                for suf in want_suffixes:
                    if key.endswith(suf):
                        out[suf] = val
                        break
    except Exception:
        return out
    return out


# ── MoE detection + expert/non-expert byte split (2026-07-24 measured win) ──
# Measured on ae/3090 (Qwen3-Coder-Next, an 80B-A3B-class MoE): the naive layer
# split (autofit 17/48) gave ~15.2 tok/s @ 16.6 GiB VRAM; the MoE-aware split
# (--n-cpu-moe 999 + n_gpu_layers=-1: ALL attention/shared/KV on GPU, expert FFN
# tensors on CPU) gave ~24.1 tok/s @ 3.2 GiB VRAM — +59% AND 5x less VRAM.
# Mechanism: MoE bytes are mostly expert FFN tensors and each token touches only
# a few experts, so keeping the always-hot non-expert tensors on the GPU beats
# splitting whole layers. Dense models have no expert tensors -> the flag is a
# no-op and everything below reads "not MoE" (byte-identical behavior).
#
# Detection (operator-grounded 2026-07-24; verified against the real coder-next
# GGUF header — qwen3next.expert_count = 512, expert_used_count = 10):
#   * KV ``{arch}.expert_count`` — the arch-agnostic suffix ``.expert_count``.
#     ``expert_count == 0`` (or absent) IS the definition of dense: detection is
#     this ONE key, no heuristics. ``.expert_used_count`` rides for reporting.
#   * Tensor names literally mark the experts: ``blk.<i>.ffn_(gate|up|down)_exps.*``
#     — the ``_exps`` suffix is the per-tensor is_expert bit, with the layer
#     attribution ``<i>`` built in. The router (``ffn_gate_inp``) and the
#     shared experts (``ffn_*_shexp``) do NOT carry the suffix and stay on the
#     GPU, exactly as llama-server's --n-cpu-moe keeps them.
#   * The name is a CONVENTION, so it is only the fast/primary path — a
#     name-INDEPENDENT SHAPE backstop guards against a converter that names
#     experts differently: an expert tensor is the STACKED one, ``n_dims >= 3``
#     with the header's expert_count in its last dims slot (``dims[-1]``).
#     Verified 2026-07-24 against the real coder-next shards (expert_count=512):
#     all 144 ``_exps`` tensors are 3-D with dims[-1]==512, and name & shape
#     select the IDENTICAL set. The ``nd>=3`` guard matters — 168 *2-D*
#     non-expert tensors (router/shexp/attn_k/v) also carry 512 in dims[-1], so
#     only the stacked-ness tells a real expert weight from a coincidental dim.
#     is_expert = name-match OR shape-match; metadata (expert_count) is the gate;
#     when the two methods disagree that is drift worth a log line, never silent.
# Tensor bytes come from the header's tensor-info table (name + data offset),
# sized by offset-difference within the data section — exact (padding included)
# without a GGML type-size table. Parsed ONCE per file (cache keyed by
# path+size+mtime, mirroring how block_count is only read at plan time — never
# re-parsed per beat).
#
# THE PRINCIPLE (why this exists as a decision input, not a nicety): autofit's
# defect was reducing a TYPED tensor list to an opaque byte-bag at one decision
# step — asking "how many whole layers fit" instead of "what KIND of bytes are
# these". The GGUF header already says which bytes are cold expert weights and
# which are always-hot attention/shared/KV; the fix is the decision function
# CONSUMING metadata it already has. Any future placement decision should start
# from this typed view, never re-flatten it to a single size.
_MOE_EXPERT_TENSOR_RE = None                     # compiled lazily (re import below)
_MOE_DETAIL_CACHE: dict = {}                     # abspath -> {"sig": (sz, mt), "detail": {...}}


def _expert_tensor_re():
    """The is_expert bit: a name segment ending in ``_exps`` (with the layer
    index captured for per-layer attribution). ``ffn_gate_inp`` (router) and
    ``ffn_*_shexp`` (shared experts) never match — GPU-resident by design."""
    global _MOE_EXPERT_TENSOR_RE
    if _MOE_EXPERT_TENSOR_RE is None:
        import re
        _MOE_EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\..*_exps(\.|$)")
    return _MOE_EXPERT_TENSOR_RE


def _layer_index(name: str) -> Optional[int]:
    """The ``<i>`` of a ``blk.<i>.…`` tensor name, for per-layer attribution, or
    None. Used to attribute a SHAPE-matched expert tensor to its block even when
    the name doesn't carry the ``_exps`` suffix (a nonstandard converter)."""
    import re
    m = re.match(r"^blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _gguf_scan_moe(model_path: str, expert_count_hint: Optional[int] = None) -> dict:
    """One-file GGUF header scan for the MoE split: KV expert counts + the
    expert/non-expert tensor byte split, computed BOTH ways —

      * NAME match: the ``_exps`` suffix (the fast path / primary bit), and
      * SHAPE match: a STACKED tensor (``n_dims >= 3``) carrying the header's
        ``expert_count`` in its last dims slot (``dims[-1]``) — name-independent.

    The dims rule is empirical, verified 2026-07-24 against the real coder-next
    Q4_K_M shards (expert_count=512): every one of the 144 ``ffn_*_exps`` tensors
    is 3-D with ``dims[-1] == 512``, and the two methods select the IDENTICAL set.
    The ``n_dims >= 3`` guard is load-bearing: on those shards 168 *2-D* tensors
    (router ``ffn_gate_inp``, shared experts ``ffn_*_shexp``, ``attn_k/v``) also
    happen to hold 512 in their last slot — only the STACKED expert weights are
    3-D, so the stacked-ness is what distinguishes a real expert tensor from a
    coincidental dimension. (GGUF stores dims reversed vs the logical shape, so
    the stacked-expert axis lands in the last stored slot, ``dims[-1]``.)

    Returns per-shard byte splits for BOTH methods plus the hit counts the
    caller (gguf_moe_detail) needs for the cross-method consistency check.
    {} on any parse issue (dense path)."""
    out: dict = {}
    try:
        import struct

        with open(model_path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return out
            version = struct.unpack("<I", fh.read(4))[0]
            if version < 2:
                return out
            n_tensors = struct.unpack("<Q", fh.read(8))[0]
            n_kv = struct.unpack("<Q", fh.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(n).decode("utf-8", "replace")

            def read_val(t: int):
                simple = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i",
                          5: "<I", 6: "<f", 7: "<?", 10: "<q", 11: "<Q", 12: "<d"}
                if t in simple:
                    fmt = simple[t]
                    return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
                if t == 8:
                    return read_str()
                if t == 9:
                    et = struct.unpack("<I", fh.read(4))[0]
                    n = struct.unpack("<Q", fh.read(8))[0]
                    return [read_val(et) for _ in range(n)]
                raise ValueError(f"unknown gguf type {t}")

            alignment = 32
            # Split GGUFs carry the full KV metadata only in shard 1; later
            # shards have no expert_count of their own, so the hint (the count
            # discovered from shard 1) lets the SHAPE backstop still fire on
            # them. A shard's OWN header always wins if it has one.
            expert_count = expert_count_hint
            for _ in range(n_kv):
                key = read_str()
                vtype = struct.unpack("<I", fh.read(4))[0]
                val = read_val(vtype)
                if key.endswith(".expert_count"):
                    out["expert_count"] = val
                    try:
                        expert_count = int(val)
                    except (TypeError, ValueError):
                        expert_count = None
                elif key.endswith(".expert_used_count"):
                    out["expert_used_count"] = val
                elif key == "general.alignment":
                    try:
                        alignment = int(val) or 32
                    except (TypeError, ValueError):
                        pass

            infos = []                                   # (name, offset, dims)
            for _ in range(n_tensors):
                name = read_str()
                nd = struct.unpack("<I", fh.read(4))[0]
                dims = struct.unpack(f"<{nd}Q", fh.read(8 * nd)) if nd else ()
                fh.read(4)                               # ggml type
                off = struct.unpack("<Q", fh.read(8))[0]
                infos.append((name, off, dims))
            header_end = fh.tell()

        data_start = (header_end + alignment - 1) // alignment * alignment
        data_bytes = os.path.getsize(model_path) - data_start
        if data_bytes < 0 or not infos:
            # A header with no tensor table (or a truncated file) can't be split.
            out["expert_bytes"] = 0
            out["non_expert_bytes"] = 0
            out["expert_bytes_shape"] = 0
            out["non_expert_bytes_shape"] = 0
            out["name_expert_hits"] = 0
            out["shape_expert_hits"] = 0
            return out
        infos.sort(key=lambda t: t[1])
        rx = _expert_tensor_re()
        # NAME method (primary): the _exps suffix.
        exp = nexp = 0
        by_layer: dict = {}
        # SHAPE method (backstop): stacked (nd>=3) tensor with expert_count in the
        # last dims slot. Off when the header carries no expert_count.
        exp_s = nexp_s = 0
        by_layer_s: dict = {}
        name_hits = shape_hits = 0
        for i, (name, off, dims) in enumerate(infos):
            end = infos[i + 1][1] if i + 1 < len(infos) else data_bytes
            size = max(0, end - off)
            m = rx.match(name)
            if m:
                name_hits += 1
                exp += size
                layer = int(m.group(1))
                by_layer[layer] = by_layer.get(layer, 0) + size
            else:
                nexp += size
            is_shape_expert = bool(
                expert_count and expert_count > 0
                and len(dims) >= 3 and dims[-1] == expert_count)
            if is_shape_expert:
                shape_hits += 1
                exp_s += size
                sl = _layer_index(name)
                if sl is not None:
                    by_layer_s[sl] = by_layer_s.get(sl, 0) + size
            else:
                nexp_s += size
        out["expert_bytes"] = int(exp)
        out["non_expert_bytes"] = int(nexp)
        out["expert_bytes_by_layer"] = by_layer
        out["expert_bytes_shape"] = int(exp_s)
        out["non_expert_bytes_shape"] = int(nexp_s)
        out["expert_bytes_by_layer_shape"] = by_layer_s
        out["name_expert_hits"] = int(name_hits)
        out["shape_expert_hits"] = int(shape_hits)
    except Exception:  # noqa: BLE001 — unreadable header == dense path, never raise
        return {}
    return out


def _gguf_shard_paths(model_path: str) -> list:
    """All shard files of a split GGUF (``…-00001-of-0000N.gguf``), or just
    ``[model_path]`` for a single-file model. Mirrors the slot supervisor's
    shard-summing (_total_gguf_bytes) so the MoE split is shard-aware too."""
    try:
        import glob
        import re
        base = os.path.basename(model_path)
        m = re.search(r"-\d{5}-of-(\d{5})\.gguf$", base)
        if m:
            patt = f"{base[:m.start()]}-*-of-{m.group(1)}.gguf"
            shards = sorted(
                s for s in glob.glob(os.path.join(os.path.dirname(model_path), patt))
                if os.path.isfile(s))
            if shards:
                return shards
    except Exception:  # noqa: BLE001
        pass
    return [model_path]


def gguf_moe_detail(model_path) -> dict:
    """THE MoE reader: ``{is_moe, expert_count, expert_used_count, expert_bytes,
    non_expert_bytes, files}`` for a GGUF (shard-aware: byte splits summed across
    all shards of a split model). ``{"is_moe": False}`` for a dense model, a
    non-GGUF path, or ANY read failure — missing metadata degrades to the dense
    path, never raises. Cached per path by (size, mtime): the header is parsed
    once per file version, never per beat/request.

    DOCTRINE — the name is convention; the shape is ground truth; metadata is the
    gate. Which tensors are experts is decided by NAME (the ``_exps`` suffix, the
    fast primary bit) OR by SHAPE (a stacked ``nd>=3`` tensor carrying the
    header's ``expert_count`` in its last dims slot — the name-independent
    backstop, so a converter that renames experts can't silently zero the split).
    The header's ``expert_count`` KV is the GATE: no positive count, no MoE, no
    matter how the tensors are named — names alone never activate the split. When
    the two methods DISAGREE that is real drift in the file, and drift is worth a
    log line, never a silent misprice: a nonstandard-naming file logs a WARNING
    and is served by shape; a file whose header claims MoE but shows no expert
    tensors at all (by either method) logs a WARNING and falls back to the dense
    split (a safe plain layer split, never a mispriced one); a file whose names
    look like experts but whose header has no count is treated as dense with a
    WARNING (no false MoE). The (path,size,mtime) cache makes every such warning
    fire once per file version."""
    try:
        path = os.path.abspath(str(model_path))
        st = os.stat(path)
        sig = (int(st.st_size), int(st.st_mtime))
    except (TypeError, ValueError, OSError):
        return {"is_moe": False}
    cached = _MOE_DETAIL_CACHE.get(path)
    if cached is not None and cached.get("sig") == sig:
        return cached["detail"]
    expert = nexpert = 0
    expert_s = nexpert_s = 0
    name_hits = shape_hits = 0
    expert_count = expert_used = None
    by_layer: dict = {}
    by_layer_s: dict = {}
    shards = _gguf_shard_paths(path)
    for shard in shards:
        # Thread the expert_count discovered so far (shard 1 carries it; later
        # shards don't) so the SHAPE backstop can fire on every shard.
        scan = _gguf_scan_moe(shard, expert_count_hint=expert_count)
        if not scan:
            continue
        expert += int(scan.get("expert_bytes") or 0)
        nexpert += int(scan.get("non_expert_bytes") or 0)
        expert_s += int(scan.get("expert_bytes_shape") or 0)
        nexpert_s += int(scan.get("non_expert_bytes_shape") or 0)
        name_hits += int(scan.get("name_expert_hits") or 0)
        shape_hits += int(scan.get("shape_expert_hits") or 0)
        for layer, b in (scan.get("expert_bytes_by_layer") or {}).items():
            by_layer[layer] = by_layer.get(layer, 0) + int(b)
        for layer, b in (scan.get("expert_bytes_by_layer_shape") or {}).items():
            by_layer_s[layer] = by_layer_s.get(layer, 0) + int(b)
        if expert_count is None and scan.get("expert_count") is not None:
            try:
                expert_count = int(scan["expert_count"])
            except (TypeError, ValueError):
                pass
        if expert_used is None and scan.get("expert_used_count") is not None:
            try:
                expert_used = int(scan["expert_used_count"])
            except (TypeError, ValueError):
                pass

    # ── Cross-method consistency: name (primary) vs shape (backstop) ──────────
    # The header's positive expert_count is the GATE for MoE. Given the gate,
    # reconcile the two expert-selection methods and log any disagreement ONCE
    # (the cache below makes this per-(path,size,mtime)).
    has_count = bool(expert_count and expert_count > 0)
    if not has_count and name_hits > 0:
        # REVERSE inconsistency: tensors named like experts but no metadata gate.
        # Names alone never activate the split — treat as dense, no false MoE.
        logger.warning(
            "gguf MoE: %s has %d _exps-named tensor(s) but no positive "
            "expert_count in the header — metadata is the gate, treating as "
            "dense (no split).", path, name_hits)
        # Fold the name-matched bytes back into non-expert so no downstream
        # consumer that reads expert_bytes directly can misprice a dense file.
        nexpert += expert
        expert = 0
        by_layer = {}
    elif has_count and name_hits == 0:
        if shape_hits > 0:
            # Nonstandard converter: experts present by SHAPE, not by name.
            # Use the shape results so the split is still priced correctly.
            logger.warning(
                "gguf MoE: %s — expert tensors present by shape (%d stacked "
                "tensors with dims[-1]==expert_count=%d) but not by _exps "
                "naming — nonstandard converter? Using shape-derived split.",
                path, shape_hits, expert_count)
            expert, nexpert = expert_s, nexpert_s
            by_layer = by_layer_s
        else:
            # Header claims MoE but NEITHER method finds expert tensors — the
            # safe fallback is the dense split (plain layer split), never a
            # mispriced one. expert stays 0 -> is_moe False below.
            logger.warning(
                "gguf MoE: %s header claims MoE (expert_count=%d) but no expert "
                "tensors identifiable by name or shape — treating as dense.",
                path, expert_count)

    # expert_count == 0 or absent IS the definition of dense (operator
    # grounding); expert bytes must also exist for a split to mean anything.
    is_moe = bool(expert_count and expert_count > 0 and expert > 0)
    # Sparsity (expert_used_count / expert_count): the fraction of expert bytes
    # a token actually touches — it predicts the per-token CPU traffic of a
    # split (coder-next: 43.59 GiB x 10/512 ~= 0.85 GiB/token, which against
    # RAM bandwidth reproduces the measured ~24 tok/s). Carried for the
    # placement evaluator; None when either count is unreadable.
    sparsity = None
    if expert_count and expert_used:
        try:
            sparsity = float(expert_used) / float(expert_count)
        except (TypeError, ValueError, ZeroDivisionError):
            sparsity = None
    detail = {"is_moe": is_moe, "expert_count": expert_count,
              "expert_used_count": expert_used, "sparsity": sparsity,
              "expert_bytes": int(expert), "non_expert_bytes": int(nexpert),
              "expert_bytes_by_layer": by_layer,
              "files": len(shards)}
    _MOE_DETAIL_CACHE[path] = {"sig": sig, "detail": detail}
    return detail


# ── Full header read + integrity facts (2026-09-23, hugpy-model-audit) ───────
# The KV/tensor readers above stop at what placement needs. The integrity audit
# needs the WHOLE header: every tensor's type + dims (so each tensor's byte
# extent is computable) and the data-section start, to answer two questions
# with no engine semantics — is the file TRUNCATED (a tensor's bytes run past
# EOF) and is it SELF-CONSISTENT (the tensor shapes agree with the file's own
# metadata, the llama.cpp ``check_tensor_dims`` rule). Array values are never
# materialised: only their element count is kept (a 150k-token vocab is a
# length, not a list), so reading a header costs one pass and O(1) memory.

# ggml_type id -> (block elements, bytes per block). Mirrors ggml.h / ggml.c
# ``type_traits``; a type missing here makes that tensor's size unknown (the
# truncation check then skips it — never a false "broken").
GGML_TYPE_SIZES = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),   # MXFP4
}

_GGUF_SCALAR = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i", 5: "<I",
                6: "<f", 7: "<?", 10: "<q", 11: "<Q", 12: "<d"}


class GGUFHeaderError(ValueError):
    """The header itself is unreadable (bad magic/version, EOF mid-header)."""


def gguf_tensor_nbytes(ggml_type: int, dims) -> Optional[int]:
    """Bytes a tensor of ``ggml_type`` with ``dims`` occupies, or None when the
    type is not in :data:`GGML_TYPE_SIZES` (or the row is not block-aligned)."""
    ts = GGML_TYPE_SIZES.get(int(ggml_type))
    if ts is None:
        return None
    blk, size = ts
    n = 1
    for d in dims:
        n *= int(d)
    if n % blk:
        return None
    return n // blk * size


def gguf_read_header(model_path: str) -> dict:
    """Parse a GGUF header completely. Returns::

        {version, n_tensors, n_kv, alignment, kv: {key: scalar | str |
         {"array_type": t, "len": n}}, tensors: [{name, dims, type, offset,
         nbytes}], data_offset, file_size}

    Raises :class:`GGUFHeaderError` on bad magic / unsupported version / a
    header that runs past EOF (a file cut inside its own header)."""
    import struct

    file_size = os.path.getsize(model_path)
    with open(model_path, "rb") as fh:
        def need(n: int) -> bytes:
            b = fh.read(n)
            if len(b) != n:
                raise GGUFHeaderError(f"EOF inside header at byte {fh.tell()}")
            return b

        def u32() -> int:
            return struct.unpack("<I", need(4))[0]

        def u64() -> int:
            return struct.unpack("<Q", need(8))[0]

        def read_str() -> str:
            n = u64()
            if n > file_size:
                raise GGUFHeaderError(f"string length {n} exceeds file size")
            return need(n).decode("utf-8", "replace")

        def skip_val(t: int) -> None:
            if t in _GGUF_SCALAR:
                need(struct.calcsize(_GGUF_SCALAR[t]))
            elif t == 8:
                n = u64()
                if n > file_size:
                    raise GGUFHeaderError("string length exceeds file size")
                fh.seek(n, 1)
            elif t == 9:
                et = u32()
                n = u64()
                if et in _GGUF_SCALAR:
                    fh.seek(n * struct.calcsize(_GGUF_SCALAR[et]), 1)
                else:
                    for _ in range(n):
                        skip_val(et)
            else:
                raise GGUFHeaderError(f"unknown gguf value type {t}")

        def read_val(t: int):
            if t in _GGUF_SCALAR:
                fmt = _GGUF_SCALAR[t]
                return struct.unpack(fmt, need(struct.calcsize(fmt)))[0]
            if t == 8:
                return read_str()
            if t == 9:
                et = u32()
                n = u64()
                start = fh.tell()
                if et in _GGUF_SCALAR:
                    fh.seek(n * struct.calcsize(_GGUF_SCALAR[et]), 1)
                else:
                    for _ in range(n):
                        skip_val(et)
                if fh.tell() > file_size:
                    raise GGUFHeaderError("array runs past EOF")
                del start
                return {"array_type": et, "len": n}
            raise GGUFHeaderError(f"unknown gguf value type {t}")

        magic = fh.read(4)
        if magic != b"GGUF":
            raise GGUFHeaderError(f"bad magic {magic!r}")
        version = u32()
        if version not in (2, 3):
            raise GGUFHeaderError(f"unsupported gguf version {version}")
        n_tensors = u64()
        n_kv = u64()
        if n_tensors > 10_000_000 or n_kv > 10_000_000:
            raise GGUFHeaderError(f"implausible counts n_tensors={n_tensors} n_kv={n_kv}")
        kv: dict = {}
        for _ in range(n_kv):
            key = read_str()
            kv[key] = read_val(u32())
        tensors = []
        for _ in range(n_tensors):
            name = read_str()
            nd = u32()
            if nd > 8:
                raise GGUFHeaderError(f"tensor {name!r} has {nd} dims")
            dims = list(struct.unpack(f"<{nd}Q", need(8 * nd))) if nd else []
            gt = u32()
            off = u64()
            tensors.append({"name": name, "dims": dims, "type": gt, "offset": off,
                            "nbytes": gguf_tensor_nbytes(gt, dims)})
        header_end = fh.tell()
    try:
        alignment = int(kv.get("general.alignment") or 32) or 32
    except (TypeError, ValueError):
        alignment = 32
    data_offset = (header_end + alignment - 1) // alignment * alignment
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "alignment": alignment, "kv": kv, "tensors": tensors,
            "data_offset": data_offset, "file_size": file_size}


def gguf_integrity(model_path: str, header: Optional[dict] = None,
                   meta_from: Optional[dict] = None) -> dict:
    """Structural facts about one GGUF file, no engine semantics::

        {ok_header, error, arch, truncated, data_end, file_size,
         unknown_type_tensors, inconsistencies: [str], expert_count,
         embedding_length, vocab_size}

    * ``truncated`` — the furthest tensor byte (data_offset + offset + nbytes,
      over tensors with a known type) lies past EOF, or the header itself does.
    * ``inconsistencies`` — the llama.cpp ``check_tensor_dims`` rule for the
      embedding/output matrices: ``token_embd.weight`` / ``output.weight`` must
      be ``[embedding_length, vocab_size]`` where vocab_size is
      ``len(tokenizer.ggml.tokens)`` (as llama.cpp sizes it), else the
      ``<arch>.vocab_size`` KV.
      ``meta_from`` lets a later shard of a split model be checked against
      shard 1's header (only shard 1 carries the metadata)."""
    out = {"ok_header": False, "error": None, "arch": None, "truncated": False,
           "data_end": None, "file_size": None, "unknown_type_tensors": 0,
           "inconsistencies": [], "expert_count": None,
           "embedding_length": None, "vocab_size": None, "n_tensors": None}
    try:
        h = header if header is not None else gguf_read_header(model_path)
    except GGUFHeaderError as exc:
        out["error"] = str(exc)
        out["truncated"] = "EOF" in str(exc) or "past EOF" in str(exc)
        return out
    except OSError as exc:
        out["error"] = f"unreadable: {exc}"
        return out
    out["ok_header"] = True
    out["file_size"] = h["file_size"]
    out["n_tensors"] = h["n_tensors"]
    kv = h["kv"]
    mkv = (meta_from or {}).get("kv") if meta_from else None
    src = kv if kv.get("general.architecture") else (mkv or kv)
    arch = src.get("general.architecture")
    out["arch"] = arch if isinstance(arch, str) else None
    end = h["data_offset"]
    unknown = 0
    for t in h["tensors"]:
        if t["nbytes"] is None:
            unknown += 1
            continue
        end = max(end, h["data_offset"] + t["offset"] + t["nbytes"])
    out["data_end"] = end
    out["unknown_type_tensors"] = unknown
    out["truncated"] = end > h["file_size"]
    if out["arch"]:
        a = out["arch"]

        def _int(v):
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        n_embd = _int(src.get(f"{a}.embedding_length"))
        # llama.cpp sizes the embedding by the TOKENIZER (n_tokens), not by
        # the ``<arch>.vocab_size`` KV (Echo-Mini: KV 32000, tokens 32005 ->
        # "expected 4096, 32005"); the KV is only the fallback.
        toks = src.get("tokenizer.ggml.tokens")
        vocab = _int(toks.get("len")) if isinstance(toks, dict) else None
        if vocab is None:
            vocab = _int(src.get(f"{a}.vocab_size"))
        ec = _int(src.get(f"{a}.expert_count"))
        out.update(embedding_length=n_embd, vocab_size=vocab, expert_count=ec)
        if n_embd and vocab:
            for t in h["tensors"]:
                if t["name"] in ("token_embd.weight", "output.weight"):
                    d = [int(x) for x in t["dims"]]
                    got = d + [1] * (4 - len(d))
                    if got[:2] != [n_embd, vocab] or any(x != 1 for x in got[2:]):
                        out["inconsistencies"].append(
                            f"tensor '{t['name']}' has wrong shape; expected "
                            f"{n_embd}, {vocab}, got {', '.join(str(x) for x in got)}")
    return out


def gguf_metadata(model_path: str, want_suffixes: tuple) -> dict:
    """Public name for :func:`_gguf_metadata` (suffix-matched GGUF KV read)."""
    return _gguf_metadata(model_path, want_suffixes)


# ── KV-bearing (full-attention) layer count — hybrid-arch honest KV pricing ──
# ``<arch>.block_count`` counts EVERY transformer block, but on a HYBRID model
# (Qwen3-Next and other attention/recurrent stacks) only a fraction of the
# blocks run FULL ATTENTION and therefore carry a context-growing KV cache; the
# rest are linear/SSM/recurrent layers whose fixed-size state does NOT grow with
# the context window. Pricing the KV cache against block_count overcounts —
# verified on the real Qwen3-Coder-Next Q4_K_M header: block_count=48 but only
# the 12 blocks with an ``blk.<i>.attn_k`` tensor (indices i where (i+1)%4==0,
# the full_attention_interval=4 pattern) actually cache K/V; the other 36 carry
# ``ssm_*`` tensors (gated-deltanet linear attention). A 4x KV overcount would
# needlessly shrink the served context.
#
# The KV-bearing count is read from the ONE ground truth the file always carries
# — its tensor names: a block runs standard attention iff it has an ``attn_k``
# projection. Shard-aware (a split GGUF spreads blocks across shards). None when
# NO ``attn_k`` tensor is found in any shard (an arch that names its KV tensors
# differently), so the caller degrades to ``block_count`` — never a guess.
_KV_LAYERS_CACHE: dict = {}


def gguf_kv_bearing_layers(model_path) -> Optional[int]:
    """Distinct transformer blocks that carry a standard attention KV cache
    (a ``blk.<i>.attn_k`` tensor), summed across all shards of a split GGUF.

    This is the honest ``n_layers`` for KV-cache byte math on HYBRID models
    (attention + linear/SSM/recurrent), where ``block_count`` overcounts. None
    when no ``attn_k`` tensor is present in any shard (unknown naming) so the
    caller falls back to ``block_count``. Cached per (path, size, mtime)."""
    try:
        path = os.path.abspath(str(model_path))
        st = os.stat(path)
        sig = (int(st.st_size), int(st.st_mtime))
    except (TypeError, ValueError, OSError):
        return None
    cached = _KV_LAYERS_CACHE.get(path)
    if cached is not None and cached.get("sig") == sig:
        return cached["val"]
    import re
    rx = re.compile(r"^blk\.(\d+)\.attn_k(\.|$)")
    blocks: set = set()
    saw_blocks = False
    for shard in _gguf_shard_paths(path):
        try:
            h = gguf_read_header(shard)
        except (GGUFHeaderError, OSError):
            continue
        for t in h.get("tensors") or ():
            name = t.get("name") or ""
            if name.startswith("blk."):
                saw_blocks = True
            m = rx.match(name)
            if m:
                blocks.add(int(m.group(1)))
    # A file with block tensors but none named attn_k -> unknown naming (None);
    # a file we couldn't read at all -> also None. Never 0 (0 would zero KV).
    val = len(blocks) if blocks else None
    if saw_blocks or val is not None:
        _KV_LAYERS_CACHE[path] = {"sig": sig, "val": val}
    return val


__all__ = ["gguf_metadata", "gguf_moe_detail", "gguf_read_header",
           "gguf_integrity", "gguf_tensor_nbytes", "gguf_kv_bearing_layers",
           "GGUFHeaderError", "GGML_TYPE_SIZES"]
