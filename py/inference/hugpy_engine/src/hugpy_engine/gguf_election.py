"""Which .gguf in a directory is THE one — shard-aware, completeness-aware.

WHY THIS EXISTS (incident, 2026-07-28)
    ``/mnt/llm_storage/models/gguf/Qwen/Qwen2.5-7B-Instruct-GGUF`` holds a soup:
    a complete fp16 4-shard split, a complete q4_k_m 2-shard split, complete
    q5_k_m / q6_k / q2_k / q3_k_m singles — and litter: a lone
    ``q4_0-00002-of-00002`` with no shard 1, a ``q8_0-00001-of-00003`` missing
    two thirds of itself, a stale q5_0 chunksums json. With no manifest
    ``filename`` pin, the elector picked THE FP16 SPLIT: 15.2 GB of
    full-precision weights elected over a 4.7 GB q4_k_m, on a fleet whose
    smallest card is an 8 GB 4060.

    It did that because the old rule tried "first shard of a split GGUF" BEFORE
    it tried the quant rank, and resolved ties lexically — and ``f`` sorts
    before ``q``. Sharding is a packaging detail; it was never a reason to
    prefer one quantization over another. The old rule also had no notion of a
    shard set being INCOMPLETE, so ``q8_0-00001-of-00003`` (one file of three)
    was as electable as a model that is actually all there.

    Operator doctrine, "defaults are promises": every default must be a SUCCESS
    PATH on the real fleet. Electing fp16 is a failure promise.

THE RULE (deterministic; this is the whole policy)
    0. A DESIGNATION always wins — the operator's per-request ``gguf_file``
       override, then the registry/manifest ``cfg.filename`` pin. Election is
       only what happens when nobody said. (Enforced by the caller,
       ``get_gguf_file``; documented here because it is part of the order.)
    1. Fold shards: ``<stem>-NNNNN-of-MMMMM.gguf`` files sharing a (dir, stem)
       are ONE variant, entrypoint = lowest shard index, bytes = SUM.
    2. A variant whose shard set is INCOMPLETE — any of 1..MMMMM missing on
       disk — is NOT ELECTABLE. It stays listed, and listed as incomplete, so
       the operator can see the litter; it just cannot be chosen.
    3. Among complete variants, rank by:
           (is_full_precision, quant_rank, filename)
       lowest wins. ``is_full_precision`` (f16/fp16/bf16/f32/fp32) sorts LAST,
       so a quantized variant always beats full precision. ``quant_rank`` is
       QUANT_ORDER below — q4_k_m first, matching the fleet's established
       default (and ``options/install.py``'s recommended install). Filename is
       the final tiebreak so the answer never depends on glob order.
    4. If NOTHING is complete, elect the best INCOMPLETE variant by the same
       key rather than returning None. Deliberate: returning None here would
       flip ``model_looks_downloaded`` to False for the whole dir, which reads
       as "absent" and provokes a re-download storm (see the 2026-07-12
       read-through hotfix and the presence-scan false-negative incident). A
       half-present model should fail LOUDLY at load with a real error, not
       silently restage the fleet. The variant listing still says incomplete.

WHY NOT "the largest complete quant": q6_k (~6.3 GB) would beat q4_k_m
(~4.7 GB) on that exact directory, and q6_k does not fit computron's 8 GB card
with any usable context. Largest-wins optimizes fidelity; the fleet's default
has to optimize *fitting*. Fidelity is what the designation is for.

PURE AND CHEAP. Every function here is a pure transform over a list the caller
already collected — no directory walks, no stat calls. This runs on the model
listing hot path.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, Optional

# A split/sharded GGUF ships as N files ``<stem>-<NNNNN>-of-<MMMMM>.gguf``,
# loaded by llama.cpp from the first shard. THE canonical pattern — the tree
# had six independent copies of this regex and not one of them read the
# ``total`` group it captured.
SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)

# Quantization token, matched on DELIMITERS rather than as a bare substring.
# Substring matching is why the old rank table double-counted: "q4_0" is inside
# "q4_0_4_8", and "f16" is inside "bf16". Mirrors model_meta._QUANT_RE.
QUANT_RE = re.compile(
    r"[-._](i?q\d+(?:_[a-z0-9]+)*|fp16|fp32|f16|f32|bf16)(?=[-._]|$)", re.I)

# Preference among quantizations, best first. The head of this list is the
# fleet's long-standing default and the operator's own hand-pin for the model
# in the incident. Full-precision tokens are NOT here — they are ranked last
# unconditionally by FULL_PRECISION below, so no reordering of this table can
# ever promote fp16 above a quant.
QUANT_ORDER = (
    "q4_k_m", "q4_k_s", "q4_k",
    "iq4_xs", "iq4_nl",
    "q5_k_m", "q5_k_s", "q5_k",
    "q6_k",
    "q8_0",
    "q4_0", "q4_1", "q5_0", "q5_1",
    "q3_k_l", "q3_k_m", "q3_k_s", "q3_k",
    "iq3_m", "iq3_s", "iq3_xs", "iq3_xxs",
    "q2_k",
    "iq2_m", "iq2_s", "iq2_xs", "iq2_xxs",
    "iq1_m", "iq1_s",
)
_QUANT_RANK = {q: i for i, q in enumerate(QUANT_ORDER)}

# Full precision — never quantized, always last. A model dir that offers these
# alongside a quant is offering a 3x-larger file for no fleet benefit.
FULL_PRECISION = frozenset(("f16", "fp16", "bf16", "f32", "fp32"))

# An unrecognized quant token sorts after every known one but ahead of full
# precision: a new quant we've never heard of is still a quant.
UNKNOWN_QUANT_RANK = len(QUANT_ORDER) + 10
# No token at all (e.g. "model.gguf") — could be anything; rank it with the
# unknowns rather than assuming the worst.
NO_QUANT_RANK = UNKNOWN_QUANT_RANK + 1


def parse_shard(name: str) -> Optional[tuple]:
    """``("stem", idx, total)`` for a shard filename, else None."""
    m = SHARD_RE.match(os.path.basename(str(name or "")))
    if not m:
        return None
    try:
        return (m.group("stem"), int(m.group("idx")), int(m.group("total")))
    except (TypeError, ValueError):
        return None


def quant_token(name: str) -> str:
    """The quantization token in a filename, lowercased ("q4_k_m", "fp16"), or
    "" when the name carries none. Last match wins: a repo directory can repeat
    the quant in the path, and the token nearest the extension is the file's."""
    hits = QUANT_RE.findall(os.path.basename(str(name or "")))
    return hits[-1].lower() if hits else ""


def is_full_precision(name: str) -> bool:
    return quant_token(name) in FULL_PRECISION


def quant_rank(name: str) -> int:
    tok = quant_token(name)
    if not tok:
        return NO_QUANT_RANK
    if tok in FULL_PRECISION:
        return NO_QUANT_RANK        # unused for ordering; the flag dominates
    return _QUANT_RANK.get(tok, UNKNOWN_QUANT_RANK)


def group_variants(entries: Iterable) -> list:
    """Fold ``[(relpath, bytes), …]`` into pickable, completeness-checked variants.

    Each variant::

        {filename, bytes, members[], entry,
         shards, shard_total, missing_shards[], complete, incomplete_reason,
         quant, full_precision}

    ``filename`` is the entrypoint basename (the lowest shard, i.e. what
    llama-server is pointed at); ``entry`` is its relpath. A non-shard file is
    its own variant with ``shard_total=None`` and ``complete=True``.

    COMPLETENESS is the presence of shards 1..MMMMM. Extra/duplicate indices are
    tolerated (a stale copy is not a missing file); a shard set claiming a
    ``total`` it cannot reach is the only thing that fails."""
    groups: dict = {}
    variants: list = []
    for rel, sz in entries or ():
        rel = str(rel)
        base = os.path.basename(rel)
        try:
            sz = int(sz)
        except (TypeError, ValueError):
            sz = 0
        parsed = parse_shard(base)
        if not parsed:
            variants.append({
                "filename": base, "entry": rel, "bytes": sz, "members": [rel],
                "shards": 1, "shard_total": None, "missing_shards": [],
                "complete": True, "incomplete_reason": None,
                "quant": quant_token(base),
                "full_precision": is_full_precision(base),
            })
            continue
        stem, idx, total = parsed
        key = (os.path.dirname(rel), stem.lower())
        g = groups.setdefault(key, {"bytes": 0, "members": [], "idxs": set(),
                                    "total": total, "entry": None,
                                    "entry_idx": None})
        g["bytes"] += sz
        g["members"].append(rel)
        g["idxs"].add(idx)
        # Disagreeing totals within one stem is itself corruption; keep the
        # largest claim so completeness is judged against the biggest promise.
        g["total"] = max(int(g["total"] or 0), int(total or 0))
        if g["entry_idx"] is None or idx < g["entry_idx"]:
            g["entry_idx"] = idx
            g["entry"] = rel
    for g in groups.values():
        total = int(g["total"] or 0)
        missing = sorted(i for i in range(1, total + 1) if i not in g["idxs"])
        base = os.path.basename(g["entry"])
        reason = None
        if missing:
            shown = ", ".join(str(i) for i in missing[:6])
            if len(missing) > 6:
                shown += f", +{len(missing) - 6} more"
            reason = (f"{len(g['idxs'])} of {total} shards present — "
                      f"missing #{shown}")
        variants.append({
            "filename": base, "entry": g["entry"], "bytes": int(g["bytes"]),
            "members": sorted(g["members"]),
            "shards": len(g["idxs"]), "shard_total": total,
            "missing_shards": missing, "complete": not missing,
            "incomplete_reason": reason,
            "quant": quant_token(base),
            "full_precision": is_full_precision(base),
        })
    variants.sort(key=lambda v: v["filename"].lower())
    return variants


def election_key(variant: dict) -> tuple:
    """The sort key. Full precision last, then quant preference, then name."""
    name = variant.get("filename") or ""
    return (1 if variant.get("full_precision") else 0,
            quant_rank(name),
            name.lower())


def elect(variants: Iterable) -> Optional[dict]:
    """The elected variant, or None when there are none at all.

    Complete variants are considered first; only if NO variant is complete does
    this fall back to the best incomplete one (see rule 4 in the module
    docstring — a re-download storm is worse than an honest load failure)."""
    vs = [v for v in (variants or ()) if isinstance(v, dict)]
    if not vs:
        return None
    complete = [v for v in vs if v.get("complete")]
    return sorted(complete or vs, key=election_key)[0]


def elect_path(paths: Iterable, sizes: Optional[dict] = None) -> Optional[str]:
    """Convenience for callers holding a flat list of gguf PATHS (not (rel,size)
    pairs): returns the elected ENTRYPOINT path.

    Sizes only affect reported bytes, never the election, so a caller with no
    cheap size source may omit them — this is what keeps the listing hot path
    free of extra stat() calls."""
    paths = [str(p) for p in (paths or ())]
    if not paths:
        return None
    sizes = sizes or {}
    root = os.path.commonpath([os.path.dirname(p) for p in paths]) \
        if len(paths) > 1 else os.path.dirname(paths[0])
    by_rel = {}
    entries = []
    for p in paths:
        try:
            rel = os.path.relpath(p, root)
        except ValueError:                       # different drives (never on posix)
            rel = os.path.basename(p)
        by_rel[rel] = p
        entries.append((rel, sizes.get(p, 0)))
    winner = elect(group_variants(entries))
    if not winner:
        return None
    return by_rel.get(winner["entry"]) or by_rel.get(winner["filename"])
