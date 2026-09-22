import logging
import re
from pathlib import PurePosixPath
from typing import Optional

from hugpy_engine.config.models.models_config import MODEL_REGISTRY
from hugpy_engine.name_match import Candidate, _distance, resolve_name
from hugpy_engine.name_match import _tokens as _nm_tokens
from hugpy_engine.placement import get_blocklist, get_model_metrics, get_worker_registry

_log = logging.getLogger(__name__)


def _normalize_model_path(value: str) -> str:
    """
    Normalize a registry folder/model path for suffix comparison.
    Works for Unix-style paths and Hugging Face-style repo ids.
    """
    return str(PurePosixPath(str(value).strip().rstrip("/")))


def _path_suffix_matches(folder: str, model_key: str) -> bool:
    """
    Return True when `model_key` matches the trailing path parts of `folder`.

    Examples:
        folder:    /mnt/llm_storage/models/Qwen/Qwen2.5-7B
        model_key: Qwen/Qwen2.5-7B        -> True
        model_key: Qwen2.5-7B             -> True
        model_key: other/Qwen2.5-7B       -> False
    """
    folder_parts = _normalize_model_path(folder).split("/")
    model_parts = _normalize_model_path(model_key).split("/")

    if len(model_parts) > len(folder_parts):
        return False

    return folder_parts[-len(model_parts):] == model_parts


def _slugify(value: str) -> str:
    """Collapse a key/hub_id to the manifest slug form for comparison.

    The manifest keys models as key_for_hub_id("C10X/Qwen2.5-1.5B-Instruct")
    -> "C10X_Qwen2.5-1.5B-Instruct", while the registry keys by folder tail
    ("Qwen2.5-1.5B-Instruct"). Comparing slugs (case-insensitive, separators
    collapsed to "_") lets either form resolve to the canonical registry key.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_").lower()


def _bare_tail(value: str) -> str:
    """The model name without its ``publisher~`` collision qualifier.

    Central keys a name-collision family as ``Owner~Repo``; the bare ``Repo`` is
    the SAME model. Routing already treats the two as one (``_match_keys`` adds
    the "~"-tail), but the canonical registry resolver did not, so a bare key
    slug-compared against an ``Owner~Repo`` key never matched and central kept
    offering the bare twin as if it were a distinct (routable) model. k67.
    """
    s = str(value)
    return s.split("~", 1)[1] if "~" in s else s


def _tokens(value: str) -> list[str]:
    """Lower-case word tokens (hugpy_engine.name_match._tokens — single source)."""
    return _nm_tokens(value)


def fuzzy_model_candidates(model_key) -> list[str]:
    """Every registry key whose tokens are a superset of the request's tokens.

    Order-independent token-subset match — the deliberately-loose layer used
    ONLY when :func:`assure_model_key`'s exact / slug / hub_id / folder /
    bare-tail cascade has already failed (so a determinate name never reaches
    here).  Returns the raw candidate list; callers decide how to break a tie
    (assure_model_key ranks by adjusted usage; resolve_model_key lists them in
    the error).  Empty request tokens -> [] (never "matches everything").
    """
    want = set(_tokens(model_key))
    if not want:
        return []
    return [k for k in MODEL_REGISTRY if want <= set(_tokens(k))]


def _adjusted_use(key: str) -> int:
    """Lifetime call count (metrics ``n_calls``) minus tie-driven picks.

    Subtracting the tie-driven picks stops a model that only got popular *by
    winning tiebreaks* from compounding that lead.  Both reads are fail-open:
    a metrics DB round-trip or settings read that errors contributes 0, never
    raises into the resolve path.  Read only for the >1-candidate tie case.
    """
    total = 0
    try:
        row = get_model_metrics().get_call(key) or {}
        total = int(row.get("n_calls") or 0)
    except Exception:                       # noqa: BLE001 — usage is advisory
        total = 0
    tie = 0
    try:
        from hugpy_control.settings import settings_store
        tie = int(settings_store.get("model_tie_picks", key, 0) or 0)
    except Exception:                       # noqa: BLE001
        tie = 0
    return total - tie


def _is_classified(key: str) -> bool:
    """True unless ``key`` is unclassified / an adapter — i.e. unless
    resolution would REFUSE it via _refuse_if_unclassified.  Mirrors that
    check so the servable stage never lets an unresolvable model win a tie.
    Fail-open to True (let a later refusal speak) only on import error.
    """
    try:
        from hugpy_engine.model_classifier import ADAPTER_TASK, NEEDS_CLASSIFICATION_TASK
        cfg = MODEL_REGISTRY.get(key)
        tasks = list(getattr(cfg, "tasks", None) or [])
        if not tasks or tasks == [NEEDS_CLASSIFICATION_TASK] or tasks == [ADAPTER_TASK]:
            return False
        return True
    except Exception:                       # noqa: BLE001
        return True


def _servable_facts(key: str) -> tuple[bool, bool]:
    """(servable, serving) for ``key``, fail-open to (False, False).

    servable = a concrete loadable artifact is installed on disk AND the model
    is classified (not adapter / needs-classification, which resolution
    refuses); serving = a live worker holds it right now.  Both feed the name
    pipeline's servable/serving stages so a real GGUF beats a bare stub — and
    an unresolvable model never wins — at cold start.
    """
    servable = serving = False
    try:
        from hugpy_storage.console.downloader import model_status
        st = (model_status(key) or {}).get("status")
        servable = (st == "installed") and _is_classified(key)
    except Exception:                       # noqa: BLE001
        servable = False
    try:
        serving = bool(get_worker_registry().workers_for_model(key, online_only=True))
    except Exception:                       # noqa: BLE001
        serving = False
    return servable, serving


def _blocked_fact(key: str) -> bool:
    try:
        return get_blocklist().block_reason(key) is not None
    except Exception:                       # noqa: BLE001
        return False


def _starred_fact(key: str) -> bool:
    try:
        cfg = MODEL_REGISTRY.get(key)
        return bool(getattr(cfg, "meta", None) and cfg.meta.get("starred"))
    except Exception:                       # noqa: BLE001
        return False


def _build_candidates(keys: list[str]):
    """Adapt registry keys into name_match.Candidate records.

    The DB/serving seam reads happen HERE, once, and only on the >1-candidate
    tie path (the caller guards it), so an exact or single-candidate resolve
    never pays for them.
    """
    out = []
    for k in keys:
        servable, serving = _servable_facts(k)
        out.append(Candidate(
            namespace=k,
            name=_bare_tail(k),
            blocked=_blocked_fact(k),
            starred=_starred_fact(k),
            servable=servable,
            serving=serving,
            fits=True,                      # VRAM predesignation is out of band here
            adjusted_use=_adjusted_use(k),
        ))
    return out


def _pipeline_pick(requested: str, keys: list[str]) -> Optional[str]:
    """Resolve a multi-candidate fuzzy tie through the name pipeline.

    Returns the winner's key, or None when the pipeline eliminates everything
    or leaves an unbroken tie (head not unique at rank distance 0) — in which
    case resolve_model_key fails with the candidate list.  On the winner, the
    tie is recorded so its usage-lead cannot compound next time.
    """
    ranked, _steps = resolve_name(requested, _build_candidates(keys))
    if not ranked:
        return None
    if len(ranked) > 1:
        # Ambiguous only if the top two are truly indistinguishable: same
        # closest-string distance AND neither starred/serving edge broke it.
        top, nxt = ranked[0], ranked[1]
        if (_distance(top.name, requested) == _distance(nxt.name, requested)
                and top.starred == nxt.starred and top.serving == nxt.serving
                and top.adjusted_use == nxt.adjusted_use):
            return None
    winner = ranked[0].model_id
    try:
        from hugpy_control.settings import settings_store
        settings_store.increment("model_tie_picks", winner)
    except Exception:                       # noqa: BLE001 — never block a pick
        pass
    return winner


# Identity-match tiers for the deterministic exact-loop, strongest first. A key
# is scored at the STRONGEST tier it matches; the highest non-empty tier wins,
# and a same-tier multi-match is broken by a total order (see _pick_by_total_order)
# so resolution never depends on registry iteration order — the source of the
# old "always one wrinkle" first-match bug.
_TIER_SLUG = 4
_TIER_HUBID = 3
_TIER_FOLDER = 2
_TIER_BARE = 1

# Coarse quant-quality rank, used ONLY to break a tie between variants of the
# same logical model (Q8 > Q6 > … > Q2; f16/bf16 high). Advisory, fail-open 0.
_QUANT_BITS = {
    "f32": 32, "fp32": 32, "bf16": 16, "f16": 16, "fp16": 16,
    "q8": 8, "q6": 6, "q5": 5, "q4": 4, "q3": 3, "q2": 2,
}


def _quant_rank(key: str) -> int:
    """Highest quant-bits token found in ``key``; 0 when none. Self-contained
    (no imports), so it can never fail the resolve path."""
    toks = set(re.split(r"[^a-z0-9]+", str(key).lower()))
    return max((_QUANT_BITS[t] for t in toks if t in _QUANT_BITS), default=0)


def _match_tier(key, values, slug, bare_slug, model_key) -> int:
    """The strongest identity tier at which ``key`` matches the request, or 0."""
    if _slugify(key) == slug:
        return _TIER_SLUG
    hub_id = getattr(values, "hub_id", None)
    if hub_id and _slugify(hub_id) == slug:
        return _TIER_HUBID
    folder = getattr(values, "folder", None)
    if folder and _path_suffix_matches(folder, model_key):
        return _TIER_FOLDER
    if _slugify(_bare_tail(key)) == bare_slug:
        return _TIER_BARE
    return 0


def _pick_by_total_order(keys, *, strict):
    """Break a same-tier multi-match with a TOTAL order:
        adjusted_use desc, quant_rank desc, canonical_key asc.
    The final canonical-key sort makes the winner a pure function of
    (input, registry) — never registry iteration order, so there is provably
    exactly one winner. ``strict`` returns None on a genuinely degenerate tie
    (top two equal on both discriminators) so the caller surfaces the candidate
    list instead of the resolver silently inventing a choice."""
    ranked = sorted(keys, key=lambda k: (-_adjusted_use(k), -_quant_rank(k), k))
    if strict and len(ranked) > 1:
        a, b = ranked[0], ranked[1]
        if _adjusted_use(a) == _adjusted_use(b) and _quant_rank(a) == _quant_rank(b):
            return None
    return ranked[0]


def assure_model_key(model_key, *, strict: bool = False):
    """
    Resolve a user-provided model key, repo id, manifest slug, folder name,
    or folder suffix into the canonical key from MODEL_REGISTRY.

    Collect-and-rank, not return-on-first: every registry key is scored at the
    strongest identity tier it matches (exact > slug > hub_id > folder >
    bare-tail); the highest non-empty tier wins, and a same-tier multi-match is
    broken by a total-order comparator (adjusted usage, then quant quality, then
    canonical key). That makes resolution a pure function of (input, registry) —
    the old return-on-first loop leaked registry iteration order into the result,
    which is the "always one wrinkle" class of ambiguity bugs. When no tier
    matches, it falls through to the loose fuzzy token-subset pipeline.

    ``strict=True`` returns None on a genuinely ambiguous tie (so the caller can
    surface the candidate list — resolve_model_key already turns a None into an
    "ambiguous/unknown, did you mean …" error) instead of picking deterministically.
    """
    if not model_key:
        return None

    model_key = str(model_key).strip().rstrip("/")

    if model_key in MODEL_REGISTRY:
        return model_key

    slug = _slugify(model_key)
    bare_slug = _slugify(_bare_tail(model_key))

    # Score every key at its strongest matching tier, then let the highest tier
    # win. Iteration order is collected, never returned on — the comparator below
    # is what decides, deterministically.
    tiers = {}
    for key, values in MODEL_REGISTRY.items():
        tier = _match_tier(key, values, slug, bare_slug, model_key)
        if tier:
            tiers[key] = tier

    if tiers:
        top_tier = max(tiers.values())
        top = [k for k, t in tiers.items() if t == top_tier]
        if len(top) == 1:
            return top[0]
        winner = _pick_by_total_order(top, strict=strict)
        if winner is not None:
            _log.debug("assure_model_key: broke tier-%d tie for %r -> %s (of %s)",
                       top_tier, model_key, winner, sorted(top))
        return winner

    # Nothing determinate matched. Last resort: loose token-subset fuzzy match
    # (this is where "coder_next" -> "Qwen~Qwen3-Coder-Next-GGUF" happens).
    # One candidate -> use it; several -> break the tie by adjusted usage;
    # a true tie or no candidates -> None (resolve_model_key fails with the
    # list). The usage/tie-pick reads are guarded to the >1 case so an exact
    # or single-candidate resolve never pays a metrics DB round-trip.
    candidates = fuzzy_model_candidates(model_key)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return _pipeline_pick(model_key, candidates)
