"""MODEL GROUPS — group derivation and the member-selection STAGE PIPELINE.

Spec: ``dev/MODEL-GROUPS-SPEC.md``. Glossary: *model group*, *tick*, *ladder walk*.

WHAT A GROUP IS (operator ruling 2026-07-28)
    Multiple ITERATIONS OF THE SAME BASE MODEL — a transformers repo, a GGUF repo
    with a quant ladder, a second publisher's GGUF of the same weights — form a
    **model group**. A group determines exactly two things:
      (a) PRIORITY among like members: which iteration serves a request on a
          given worker;
      (b) the placement AVENUES permitted BEFORE eviction is considered.
    It is NOT a residency class, NOT a protection class, NOT a designation. It
    sits strictly upstream of all three, choosing WHICH KEY to route; the
    unchanged worker-selection and admission machinery does everything after.

THREE TICKS, named exactly (keeper owns nomenclature — do not rename):
    quality   rules out degraded variants: the 4-bit class and below. Sets the
              FLOOR of any ladder walk.
    speed     rules out ram-only placement and spill/partial offload — the
              chosen member must be fully GPU-resident.
    priority  the group may EVICT other residents to meet the ticked standards.
              WITHOUT priority, quality/speed SOFTEN to best-fit preferences.

    ⚠ ``priority`` is an eviction INITIATOR, never a protection class. A
    priority group's own residents gain NO shield: eviction protection stays
    exactly two classes (🔒static + actively-answering). Nothing in this module
    marks anything protected, and nothing here evicts — the priority verdict is
    a DECLARED NEED handed to the existing declare-need -> evict-to-fit
    admission path, which is the only thing allowed to displace a resident.

THIS MODULE IS PURE. No flask, no network, no filesystem, no settings reads, no
clock. Every input is passed in; every output is a plain dict. That is what lets
the whole pipeline be unit-tested against real catalog fixtures without a fleet,
and it is why the kill switch lives at the CALLER (the central-side provider),
not in here — a pure function has nothing to gate.

THE PIPELINE IS A STAGE LIST, not an if-tree. ``select_member`` walks ``STAGES``
in order over a candidate list; each stage either drops candidates WITH A
ONE-LINE REASON or re-ranks them. Adding a rule means adding a stage, and every
drop is reportable — the same say-why discipline as ``explain_no_worker``, but
per-candidate rather than only-when-empty.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional

from hugpy_engine.gguf_election import FULL_PRECISION, election_key, quant_token

# ---------------------------------------------------------------------------
# DERIVATION
# ---------------------------------------------------------------------------
#
# A group key is a BASE NAME: the repo name with publisher and every
# packaging/quantization suffix stripped. PUBLISHER IS DELIBERATELY IGNORED —
# two publishers' GGUFs of the same base ARE the same group, which is what makes
# Qwen/Qwen3-Coder-Next-GGUF and unsloth/Qwen3-Coder-Next-GGUF one group. That is
# the operator's "multiple iterations of the same base model".
#
# CONSERVATIVE BY CONSTRUCTION: false merges are worse than false splits. A bad
# merge silently routes a request to different weights; a bad split just means
# the operator never got the choice. Only the suffixes listed here are stripped,
# and a differing BASE (Qwen2.5-Coder-32B vs Qwen2.5-Coder-32B-Instruct) always
# stays split. Correct a mistake with the ``members`` override map — NEVER by
# loosening this list.
_SUFFIXES = (
    # packaging / quantization families
    r"i1-gguf", r"gguf", r"awq", r"gptq", r"exl2", r"exl3", r"mlx",
    r"nvfp4", r"fp8", r"bnb-4bit", r"4bit", r"8bit", r"int4", r"int8", r"nf4",
    # imatrix / "unsloth dynamic" packaging markers
    r"neo-imatrix", r"imatrix", r"i1", r"ud",
    # a bare quant token, same grammar the elector uses
    r"i?q\d+(?:_[a-z0-9]+)*", r"fp16", r"fp32", r"f16", r"f32", r"bf16",
)
_SUFFIX_RE = re.compile(r"[-_.](" + "|".join(_SUFFIXES) + r")$", re.I)

# How many times to re-strip. "-Thinking-NEO-Imatrix-GGUF" needs two passes;
# eight is slack, and the loop exits early on a fixed point anyway.
_MAX_STRIP_PASSES = 8

TICKS = ("quality", "speed", "priority")
NO_TICKS = {"quality": False, "speed": False, "priority": False}


def base_name(name: Any) -> str:
    """The group key for a hub id or model key.

    ``unsloth/Qwen2.5-VL-7B-Instruct-GGUF`` -> ``qwen2.5-vl-7b-instruct``.
    Also tolerates the tree's ``publisher~Repo`` key form."""
    s = str(name or "").strip()
    if not s:
        return ""
    s = s.split("/")[-1]
    s = s.split("~")[-1]
    s = s.lower()
    for _ in range(_MAX_STRIP_PASSES):
        stripped = _SUFFIX_RE.sub("", s)
        if stripped == s:
            break
        s = stripped
    return s


def normalize_ticks(raw: Any) -> dict:
    """A tick dict with exactly the three keys, all booleans. Unknown keys are
    dropped (the vocabulary is closed); absent keys default False."""
    out = dict(NO_TICKS)
    if isinstance(raw, dict):
        for t in TICKS:
            if t in raw:
                out[t] = bool(raw[t])
    return out


def derive_groups(catalog: Any, overrides: Any = None) -> dict:
    """Derive the group registry from the model catalog.

    ``catalog``   {model_key: {hub_id, framework, ...}} — model_discovery.json's
                  shape, or anything dict-of-dicts that carries ``hub_id``.
    ``overrides`` settings ns ``model_groups``:
                  {group_key: {"members": [model_key...]?, "ticks": {...}?}}
                  ``members`` present  => membership is EXACTLY that list
                                          (an explicit split or merge);
                  ``members`` absent   => membership stays derived, only the
                                          ticks are applied.
                  The reserved key ``enabled`` is the kill switch and is never a
                  group.

    Returns {group_key: {"group_key", "derived", "ticks", "members": [
             {"model_key", "framework", "hub_id"}]}}. Deterministic ordering
    throughout — group keys and members are sorted, so two calls on the same
    catalog produce byte-identical output.

    Groups of size 1 are STILL GROUPS. They have nothing to choose between, but
    an operator can tick one in anticipation of a second member arriving, and
    the ticks then apply the moment it does.
    """
    catalog = catalog if isinstance(catalog, dict) else {}
    overrides = overrides if isinstance(overrides, dict) else {}

    by_key: dict = {}
    for model_key, row in catalog.items():
        row = row if isinstance(row, dict) else {}
        by_key[str(model_key)] = {
            "model_key": str(model_key),
            "framework": str(row.get("framework") or "") or None,
            "hub_id": row.get("hub_id") or None,
        }

    # 1. auto-derive
    groups: dict = {}
    for model_key, member in by_key.items():
        gk = base_name(member.get("hub_id") or model_key)
        if not gk:
            continue
        groups.setdefault(gk, {"group_key": gk, "derived": True,
                               "ticks": dict(NO_TICKS), "members": []})
        groups[gk]["members"].append(member)

    # 2. apply the override map
    for gk, ov in overrides.items():
        gk = str(gk)
        if gk == "enabled" or not isinstance(ov, dict):
            continue                      # the kill switch is not a group
        g = groups.setdefault(gk, {"group_key": gk, "derived": False,
                                   "ticks": dict(NO_TICKS), "members": []})
        if "ticks" in ov:
            g["ticks"] = normalize_ticks(ov.get("ticks"))
        members = ov.get("members")
        if isinstance(members, list):
            # An explicit membership REPLACES the derived one — that is the
            # whole point of a correction. Unknown keys are kept as bare
            # members so an operator can pre-declare a model not yet in the
            # catalog without the entry silently vanishing.
            g["derived"] = False
            g["members"] = [
                by_key.get(str(mk), {"model_key": str(mk), "framework": None,
                                     "hub_id": None})
                for mk in members
            ]
            # Members claimed by an explicit group leave whatever group the
            # deriver put them in, or the same model would be in two groups.
            claimed = {str(mk) for mk in members}
            for other_gk, other in groups.items():
                if other_gk == gk:
                    continue
                other["members"] = [m for m in other["members"]
                                    if m["model_key"] not in claimed]

    for g in groups.values():
        g["members"].sort(key=lambda m: m["model_key"])
    return {k: groups[k] for k in sorted(groups) if groups[k]["members"]}


def group_for_model(groups: Any, model_key: Any) -> Optional[dict]:
    """The group containing ``model_key`` (or keyed BY it, when the caller named
    a group key directly). None when the key is in no group."""
    groups = groups if isinstance(groups, dict) else {}
    mk = str(model_key or "")
    if mk in groups:                      # the caller named a group key
        return groups[mk]
    for g in groups.values():
        for m in g.get("members") or ():
            if m.get("model_key") == mk:
                return g
    return None


# ---------------------------------------------------------------------------
# QUALITY — the 4-bit class and below
# ---------------------------------------------------------------------------
#
# NOTE this is a DIFFERENT axis from gguf_election.QUANT_ORDER. That table is a
# FIT preference (q4_k_m first — the rung most likely to fit the fleet's
# smallest card); quality is BIT WIDTH. Conflating them is how you get "the
# highest-ranked quant" meaning the smallest one. Both are used here, for their
# own questions: bit width decides the quality floor and the ladder's direction,
# election_key is only the deterministic tiebreak.
_BITS_RE = re.compile(r"^i?q(\d+)", re.I)

# "the 4-bit class and below" — the operator's phrase, as a number.
FOUR_BIT = 4
# The floor a ``quality`` ladder walk may never cross. Named rather than
# computed so the spec and the code say the same thing.
QUALITY_FLOOR_BITS = 5
QUALITY_FLOOR_QUANT = "q5_k_s"

# Name markers that declare a transformers checkpoint is ALREADY 4-bit. Mirrors
# alloc_modes._PREQUANT_MARKERS, minus the 8-bit ones — an int8 repo is not the
# degraded class ``quality`` is about.
_FOUR_BIT_MARKERS = ("4bit", "-4-bit", "int4", "nf4", "fp4", "nvfp4",
                     "gptq", "awq", "bnb")


def quant_bits(name_or_token: Any) -> Optional[int]:
    """Bit width declared by a quant token or filename. None when unknown.

    Full precision reports its own width (16/32) so it is never mistaken for a
    degraded variant — ``quality`` excludes 4-bit-and-below, and fp16 is neither.
    """
    tok = str(name_or_token or "").strip().lower()
    if not tok:
        return None
    if tok not in FULL_PRECISION and not _BITS_RE.match(tok):
        tok = quant_token(tok)            # a filename was passed
    if not tok:
        return None
    if tok in FULL_PRECISION:
        return 32 if "32" in tok else 16
    m = _BITS_RE.match(tok)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def is_degraded_quant(name_or_token: Any) -> bool:
    """True for the 4-bit class and below. Unknown tokens are NOT degraded —
    a quant we have never heard of is still a quant (same posture as
    gguf_election.UNKNOWN_QUANT_RANK), and guessing "degraded" would take a
    working rung out of the ladder."""
    bits = quant_bits(name_or_token)
    return bits is not None and bits <= FOUR_BIT


def member_is_degraded(member: Any) -> bool:
    """True when a NON-GGUF member can only be loaded as a 4-bit checkpoint.

    Two ways that happens: the repo is already 4-bit (name markers), or the
    fleet would have to apply a bitsandbytes 4-bit specialization to fit it
    (``bnb`` on the candidate, set by the caller from ``bnb_by_model``)."""
    member = member if isinstance(member, dict) else {}
    if member.get("bnb"):
        return True
    hay = f"{member.get('model_key') or ''} {member.get('hub_id') or ''}".lower()
    return any(m in hay for m in _FOUR_BIT_MARKERS)


# ---------------------------------------------------------------------------
# THE LADDER
# ---------------------------------------------------------------------------
def ladder(variants: Any) -> list:
    """A member's rungs: COMPLETE variants only, best-QUALITY first.

    Incomplete shard sets are never rungs (a half-present model is not an
    option; ``gguf_election`` already marks them and says why).

    ORDERING, and the one thing here that is easy to get catastrophically wrong:

      1. FULL PRECISION SORTS LAST. Unconditionally, exactly as
         ``gguf_election.election_key`` does. It is electable — it is on the
         ladder — but it is never a rung the walk climbs TO. This is the shipped
         doctrine from the 2026-07-28 incident ("electing fp16 is a failure
         promise": 15.2 GB of full-precision weights beat a 4.7 GB q4_k_m on a
         fleet whose smallest card is 8 GB), and a group must not re-open it.
         A first cut of this function ranked purely by bit width, which put fp16
         at the head and had ae serving the 15 GB split the moment it had room.
         Full precision is what a DESIGNATION is for, never a default.
      2. Then descending bit width — the quality axis. Note this is NOT
         ``QUANT_ORDER``, which is a FIT preference (q4_k_m first, the rung most
         likely to fit the smallest card). Conflating the two makes "the
         best-ranked quant" mean the smallest one.
      3. Then ``election_key`` as the deterministic tiebreak within a bit width
         (q5_k_m before q5_0).

    So the walk starts at the best QUANT the repo offers and steps DOWN toward
    fit — the operator's "options for a best fit when under workloads" — and
    only ever reaches full precision if nothing else is there.
    """
    rungs = []
    for v in (variants or ()):
        if not isinstance(v, dict):
            continue
        # ``complete`` absent means complete — gguf_variants_detail only adds
        # the key on the incomplete rows (it "does not grow four keys for
        # nothing"), so absence is the common, complete case.
        if v.get("complete") is False:
            continue
        name = v.get("filename") or v.get("quant") or ""
        tok = v.get("quant") or quant_token(name) or ""
        rungs.append({
            "quant": tok,
            "filename": v.get("filename"),
            "bytes": int(v.get("bytes") or 0),
            "bits": quant_bits(tok or name),
            "full_precision": tok in FULL_PRECISION,
        })
    rungs.sort(key=lambda r: (
        1 if r["full_precision"] else 0,          # full precision LAST, always
        -(r["bits"] or 0),                        # then best quant first
        election_key({"filename": r["filename"] or r["quant"],
                      "full_precision": r["full_precision"]}),
    ))
    return rungs


def walk_ladder(rungs: Any, fits: Callable[[int], bool],
                floor_bits: Optional[int] = None) -> tuple:
    """THE LADDER WALK. ``(chosen_rung|None, reason)``.

    Descend the rungs best-first and return the first that ``fits`` RIGHT NOW.
    ``fits(bytes) -> bool`` is supplied by the caller and is where contention
    lives: it reads measured FREE resources, never total capacity. When
    ``floor_bits`` is set (the ``quality`` tick) the walk stops there — a rung
    below the floor is never returned, even if it is the only thing that fits.
    """
    below_floor = 0
    too_big = 0
    for r in (rungs or ()):
        if floor_bits is not None and (r.get("bits") or 0) < floor_bits:
            below_floor += 1
            continue
        if fits(int(r.get("bytes") or 0)):
            return r, f"{r['quant'] or 'the only rung'} is the best complete rung that fits now"
        too_big += 1
    if below_floor and not too_big:
        return None, (f"every complete rung is below the {QUALITY_FLOOR_QUANT} "
                      f"quality floor (quality)")
    if below_floor:
        return None, (f"no complete rung at or above the {QUALITY_FLOOR_QUANT} "
                      f"quality floor fits right now (quality)")
    if too_big:
        return None, "no complete rung fits right now"
    return None, "no complete rung on the ladder"


# ---------------------------------------------------------------------------
# THE STAGE PIPELINE
# ---------------------------------------------------------------------------
#
# A CANDIDATE is one (member x worker) pair:
#
#   {"model_key", "framework", "hub_id", "bnb": bool,
#    "variants": [...],            # gguf_variants_detail rows, model-level
#    "bytes": int|None,            # non-gguf size
#    "worker": {"id", "name",
#               "vram_total", "vram_free", "ram_total", "ram_free",
#               "has_gpu": bool,
#               "alloc_mode": str, "alloc_explicit": bool}}
#
# Stages mutate only two fields they own — ``rung`` and ``rank`` — and drop by
# appending to ``skipped``. Nothing here reads global state.

VERDICT_SELECT = "select"     # a member satisfies the ticked standards now
VERDICT_BESTFIT = "bestfit"   # ticks unmet but not ticked priority — soften
VERDICT_NEED = "need"         # priority: declare a need, let admission evict
VERDICT_NONE = "none"         # nothing to route


def _fits_fn(worker: dict, speed: bool):
    """The contention-aware fit predicate for one worker.

    ``speed`` ticked => must be FULLY GPU-RESIDENT, so the budget is measured
    FREE VRAM alone — no spill, no partial offload, no ram-only. Otherwise the
    budget is free VRAM + free RAM, the same combined ceiling
    ``alloc_modes.worker_fit_verdict`` uses (max-gpu/max-ram spill across both).

    FREE, never TOTAL. Sizing against capacity would be exactly the
    "guess the budget" the declare-need doctrine forbids; this predicate only
    ever answers "does it fit RIGHT NOW", and the priority path escalates to a
    declared need when the answer is no.
    """
    vram_free = int(worker.get("vram_free") or 0)
    ram_free = int(worker.get("ram_free") or 0)
    budget = vram_free if speed else vram_free + ram_free

    def _fits(n: int) -> bool:
        return bool(n) and int(n) <= budget
    return _fits


def _stage_alloc_mode(cands: list, ticks: dict, skipped: list) -> list:
    """Read each candidate's allocation mode and mark operator-explicit ones.

    OPERATOR LEVERS OUTRANK GROUP TICKS (doctrine
    allocation-modes-are-operator-levers). An EXPLICITLY-set mode is immune:
    the tick yields, records why, and the ticked standard is reported unmet.
    A DERIVED mode is not an operator statement, so a tick may override it.
    """
    for c in cands:
        w = c.get("worker") or {}
        mode = str(w.get("alloc_mode") or "max-gpu")
        explicit = bool(w.get("alloc_explicit"))
        c["alloc_mode"] = mode
        c["alloc_explicit"] = explicit
        c["tick_yielded"] = None
        if ticks.get("speed") and mode in ("ram-only", "max-ram", "explicit"):
            if explicit:
                c["tick_yielded"] = "speed"
                skipped.append({
                    "model_key": c["model_key"],
                    "reason": (f"speed tick yielded: operator set {mode} on "
                               f"{w.get('name') or w.get('id')}"),
                })
    return cands


def _stage_quality(cands: list, ticks: dict, skipped: list) -> list:
    """Drop degraded members: the 4-bit class and below."""
    if not ticks.get("quality"):
        return cands
    kept = []
    for c in cands:
        if c.get("variants"):
            kept.append(c)               # GGUF: the ladder floor handles it
            continue
        if member_is_degraded(c):
            why = ("fits only as 4-bit" if c.get("bnb")
                   else "is a 4-bit checkpoint")
            skipped.append({
                "model_key": c["model_key"],
                "reason": (f"{c.get('framework') or 'non-gguf'} member "
                           f"excluded: {why} (quality)"),
            })
            continue
        kept.append(c)
    return kept


def _stage_speed(cands: list, ticks: dict, skipped: list) -> list:
    """Drop members that cannot be fully GPU-resident on their worker."""
    if not ticks.get("speed"):
        return cands
    kept = []
    for c in cands:
        w = c.get("worker") or {}
        if c.get("tick_yielded") == "speed":
            kept.append(c)               # an operator lever already won here
            continue
        if not w.get("has_gpu"):
            skipped.append({
                "model_key": c["model_key"],
                "reason": (f"{w.get('name') or w.get('id')} has no GPU — "
                           f"ram-only is excluded (speed)"),
            })
            continue
        if str(c.get("alloc_mode")) == "ram-only":
            skipped.append({
                "model_key": c["model_key"],
                "reason": "ram-only avenue excluded (speed)",
            })
            continue
        kept.append(c)
    return kept


def _stage_ladder(cands: list, ticks: dict, skipped: list) -> list:
    """Pick each GGUF candidate's rung; size each non-GGUF candidate."""
    floor = QUALITY_FLOOR_BITS if ticks.get("quality") else None
    speed = bool(ticks.get("speed")) and True
    kept = []
    for c in cands:
        w = c.get("worker") or {}
        # A tick that YIELDED to an operator lever must not also constrain the
        # fit budget — the operator's mode is now in charge of placement.
        effective_speed = speed and c.get("tick_yielded") != "speed"
        fits = _fits_fn(w, effective_speed)
        rungs = ladder(c.get("variants"))
        if rungs:
            rung, why = walk_ladder(rungs, fits, floor)
            c["rungs"] = rungs
            c["rung"] = rung
            c["why"] = why
            if rung is None:
                # NOT dropped: a candidate that cannot fit now is exactly what
                # the priority path escalates on. It is ranked last instead.
                c["unfit"] = True
                # The smallest rung at or above the floor is the honest need.
                allowed = [r for r in rungs
                           if floor is None or (r.get("bits") or 0) >= floor]
                c["need_bytes"] = min((r["bytes"] for r in allowed), default=None)
            else:
                c["unfit"] = False
                c["need_bytes"] = int(rung["bytes"])
        else:
            size = int(c.get("bytes") or 0)
            c["rungs"] = []
            c["rung"] = None
            c["need_bytes"] = size or None
            c["unfit"] = not fits(size)
            c["why"] = ("fits now" if not c["unfit"]
                        else "does not fit right now")
        kept.append(c)
    return kept


def _rank_key(c: dict) -> tuple:
    """Ticked-standards-satisfying first, then quality, then measured fit.

    Deterministic to the last field: two candidates never tie, so the same
    inputs always produce the same member. (Non-determinism in a routing
    decision is a debugging nightmare the tree has paid for before.)
    """
    rung = c.get("rung") or {}
    return (
        1 if c.get("unfit") else 0,             # fits now wins, always
        0 if c.get("tick_yielded") is None else 1,   # unyielded ticks first
        1 if rung.get("full_precision") else 0,      # full precision LAST (see ladder)
        -(rung.get("bits") or quant_bits(c.get("hub_id")) or 0),  # quality desc
        0 if c.get("variants") else 1,          # a GGUF ladder beats a fixed repo
        int(c.get("need_bytes") or 0),          # smaller footprint breaks ties
        str(c.get("model_key") or ""),
        str((c.get("worker") or {}).get("id") or ""),
    )


def _stage_rank(cands: list, ticks: dict, skipped: list) -> list:
    return sorted(cands, key=_rank_key)


# The pipeline, in order. Adding a rule means adding a stage HERE — never an
# if-branch inside another one.
STAGES: tuple = (
    ("alloc-mode", _stage_alloc_mode),
    ("tick:quality", _stage_quality),
    ("tick:speed", _stage_speed),
    ("ladder", _stage_ladder),
    ("rank", _stage_rank),
)


def select_member(group: Any, candidates: Any) -> dict:
    """Run the stage pipeline. Returns a SELECTION — never raises, never evicts.

    ``group``      a derive_groups() entry (its ``ticks`` drive everything).
    ``candidates`` the expanded (member x worker) list, shape documented above.

    Returns::

        {"group_key", "ticks",
         "verdict": "select"|"bestfit"|"need"|"none",
         "model_key": str|None,      # THE member to route
         "worker_id": str|None,
         "as": str|None,             # the chosen rung's quant, when GGUF
         "why": str,
         "need_bytes": int|None,     # set on the "need" verdict
         "demanded_by": str|None,    # the tick that demanded a headroom pass
         "skipped": [{"model_key", "reason"}]}

    THE VERDICTS
      select   a member satisfies the ticked standards and fits right now.
      bestfit  nothing satisfies them, and ``priority`` is NOT ticked — so
               quality/speed SOFTEN to preferences and the best available
               member is routed anyway. Never evicts.
      need     nothing satisfies them and ``priority`` IS ticked. The selection
               carries ``need_bytes`` for the existing declare-need ->
               evict-to-fit admission. This module does not evict; it declares.
      none     there was nothing to choose from.
    """
    group = group if isinstance(group, dict) else {}
    ticks = normalize_ticks(group.get("ticks"))
    gk = str(group.get("group_key") or "")
    skipped: list = []
    cands = [dict(c) for c in (candidates or ()) if isinstance(c, dict)]

    for _name, stage in STAGES:
        cands = stage(cands, ticks, skipped)

    base = {"group_key": gk, "ticks": ticks, "skipped": skipped,
            "model_key": None, "worker_id": None, "as": None,
            "need_bytes": None, "demanded_by": None}

    if not cands:
        return {**base, "verdict": VERDICT_NONE,
                "why": "no member survived the group's ticks"}

    winner = cands[0]
    rung = winner.get("rung") or {}
    chosen = {
        **base,
        "model_key": winner.get("model_key"),
        "worker_id": (winner.get("worker") or {}).get("id"),
        "as": rung.get("quant") or None,
        "need_bytes": winner.get("need_bytes"),
        "why": winner.get("why") or "",
    }
    # Everything we did not pick is a skip WITH A REASON — that is the whole
    # point of the feed line the operator reads.
    for c in cands[1:]:
        skipped.append({
            "model_key": c.get("model_key"),
            "reason": _not_chosen_reason(c, winner),
        })

    if not winner.get("unfit"):
        return {**chosen, "verdict": VERDICT_SELECT}

    # Nothing fits right now.
    demanded = ("speed" if ticks.get("speed")
                else "quality" if ticks.get("quality") else None)
    if ticks.get("priority"):
        return {**chosen, "verdict": VERDICT_NEED, "demanded_by": demanded,
                "why": (winner.get("why") or "nothing fits right now")
                       + " — priority: declaring need for the eviction pass"}
    return {**chosen, "verdict": VERDICT_BESTFIT,
            "why": (winner.get("why") or "nothing fits right now")
                   + " — no priority tick: softening to best fit"}


def _not_chosen_reason(c: dict, winner: dict) -> str:
    if c.get("unfit") and not winner.get("unfit"):
        return c.get("why") or "does not fit right now"
    if c.get("tick_yielded"):
        return f"{c['tick_yielded']} tick yielded to an explicit allocation mode"
    wr = (winner.get("rung") or {}).get("quant")
    cr = (c.get("rung") or {}).get("quant")
    if wr and cr and wr != cr:
        return f"{cr} passed over: {wr} is the better rung that still fits"
    return f"passed over for {winner.get('model_key')}"
