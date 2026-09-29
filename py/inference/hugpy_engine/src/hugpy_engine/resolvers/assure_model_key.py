import json
import logging
import os
import re
import threading
from pathlib import PurePosixPath
from typing import Optional

from hugpy_engine.config.models.models_config import MODEL_REGISTRY
from hugpy_engine.name_match import Candidate, _distance, resolve_name
from hugpy_engine.name_match import _tokens as _nm_tokens
from hugpy_engine.placement import get_blocklist, get_model_metrics, get_worker_registry

_log = logging.getLogger(__name__)


# ── OPERATOR MODEL ALIASES (2026-09-24) ──────────────────────────────────────
# An OPT-IN, operator-editable map of harness/vendor model names -> a hugpy model
# name, so a harness that hardcodes its own default ("anthropic/claude-opus-4.6",
# "gpt-6-luna", ...) resolves through hugpy without editing the harness. It is
# EXPLICIT config, never a guess: the file is empty/absent by default (behaviour
# byte-identical to before), an alias TARGET must itself resolve to a real
# registry key (a broken alias is IGNORED so the honest "unknown model" refusal
# for the original name still fires — an alias can never silently substitute a
# nonexistent model), and the alias tier runs BEFORE fuzzy matching but the exact
# registry membership of the literal name still wins first. JSON shape:
#   {"anthropic/claude-opus-4.6": "Qwen~Qwen3-Coder-Next-GGUF", ...}
# Path: $HUGPY_MODEL_ALIASES_PATH, else <PROJECTS_HOME>/model_aliases.json (the
# same runtime-state root api_keys.json / the manifests use).
_ALIAS_LOCK = threading.Lock()
_ALIAS_CACHE: "dict[str, str]" = {}
_ALIAS_MTIME: float = -1.0
_ALIAS_PATH_CACHED: Optional[str] = None


def _aliases_path() -> str:
    global _ALIAS_PATH_CACHED
    if _ALIAS_PATH_CACHED is None:
        p = os.environ.get("HUGPY_MODEL_ALIASES_PATH")
        if not p:
            try:
                from hugpy_platform.constants import PROJECTS_HOME
                p = os.path.join(PROJECTS_HOME, "model_aliases.json")
            except Exception:  # noqa: BLE001 — no platform root -> a stable dud path
                p = os.path.join(os.path.expanduser("~"), ".hugpy", "model_aliases.json")
        _ALIAS_PATH_CACHED = p
    return _ALIAS_PATH_CACHED


def _load_aliases() -> "dict[str, str]":
    """The alias map {lookup_form: target_name}, reloaded on file mtime change.

    Lookup forms are pre-normalized (raw-stripped-lower AND slugified) so a call
    can match either the literal alias key or its slug. Fail-open to {} on any
    fault (missing file, bad JSON, non-string values) — a broken alias file must
    never break resolution."""
    global _ALIAS_CACHE, _ALIAS_MTIME
    path = _aliases_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        if _ALIAS_MTIME != -1.0 or _ALIAS_CACHE:
            with _ALIAS_LOCK:
                _ALIAS_CACHE, _ALIAS_MTIME = {}, -1.0
        return {}
    if mtime == _ALIAS_MTIME:
        return _ALIAS_CACHE
    built: "dict[str, str]" = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if not isinstance(k, str) or not isinstance(v, str) or not v.strip():
                    continue
                target = v.strip()
                built[k.strip().lower()] = target
                built.setdefault(_slugify(k), target)
    except Exception:  # noqa: BLE001 — a bad alias file is no aliases, never a crash
        _log.warning("model aliases at %s ignored (unreadable/invalid)", path,
                     exc_info=True)
        built = {}
    with _ALIAS_LOCK:
        _ALIAS_CACHE, _ALIAS_MTIME = built, mtime
    return built


def _alias_target(model_key: str) -> Optional[str]:
    """The operator-registered target for ``model_key``, or None. Matches the
    literal name (case-insensitive) or its slug."""
    aliases = _load_aliases()
    if not aliases:
        return None
    return aliases.get(model_key.strip().lower()) or aliases.get(_slugify(model_key))


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


# Packaging / quant / framework markers — a name carrying any of these named a
# representation, not merely a base-name family. Single source for both the
# serving-override gate (_package_specific) and the logical-identity strip
# (_logical_id) so the two never drift.
_FORMAT_TOKENS = frozenset({
    "gguf", "transformers", "transformer", "safetensors", "pytorch",
    "awq", "gptq", "exl2", "exl3", "mlx", "bnb", "4bit", "8bit",
    "fp8", "fp16", "bf16", "f16", "q2", "q3", "q4", "q5", "q6", "q8",
})

# The GGUF engine family (mirrors alloc_modes.GGUF_ENGINES; inlined to keep this
# resolver import-cheap and cycle-free).
_GGUF_ENGINES = frozenset({"gguf", "llama_cpp"})


def _package_specific(model_key: str) -> bool:
    """Whether the caller explicitly selected a model packaging/framework.

    A bare family request (``Qwen3-Coder-Next``) must follow the live serving
    representation, whether that is GGUF or Transformers.  ``GGUF`` and the
    common Transformers/package markers are deliberate specificity and must
    remain exact rather than being rewritten to whichever sibling is hot.
    """
    return bool(set(_tokens(model_key)) & _FORMAT_TOKENS)


def _framework_of(key: str):
    """The registry framework for ``key`` (scalar), or None when absent."""
    cfg = MODEL_REGISTRY.get(key)
    fw = getattr(cfg, "framework", None) if cfg is not None else None
    if isinstance(fw, (list, tuple)):
        fw = fw[0] if fw else None
    return fw


def _representation_of(key: str):
    """The gguf-vs-transformers REPRESENTATION of a registry key, read from its
    framework (never a name substring). GGUF family -> "gguf"; any other
    concrete framework -> "transformers"; None when the key is not in the
    registry."""
    fw = _framework_of(key)
    if fw is None:
        return None
    return "gguf" if str(fw).strip().lower() in _GGUF_ENGINES else "transformers"


def _logical_id(key: str) -> str:
    """Base-name identity of a key with packaging/quant tokens dropped, so the
    GGUF and Transformers rows of one model share it (``…-Coder-Next-GGUF`` and
    ``…-Coder-Next`` -> ``coder next``). Used ONLY to pair representation
    siblings under an explicit model_format."""
    return " ".join(t for t in _tokens(_bare_tail(key)) if t not in _FORMAT_TOKENS)


def _owner_of(key: str) -> str:
    """The publisher qualifier of ``key`` (``Owner~Repo`` / ``owner/repo``), or
    its hub_id owner, or "" — the same owner rule _match_tier enforces, so a
    representation redirect never crosses publishers."""
    s = str(key)
    if "~" in s:
        return _slugify(s.split("~", 1)[0])
    hub_id = getattr(MODEL_REGISTRY.get(key), "hub_id", None)
    if hub_id and "/" in str(hub_id):
        return _slugify(str(hub_id).split("/", 1)[0])
    if "/" in s:
        return _slugify(s.split("/", 1)[0])
    return ""


def _redirect_to_format(resolved: str, model_key: str, fmt: str, *, strict: bool):
    """Return the key of the requested REPRESENTATION for the model ``resolved``
    names. When ``resolved`` is already that representation it is returned
    unchanged; otherwise the same-logical-identity, same-owner sibling of
    representation ``fmt`` is chosen (deterministically). When the registry has
    no such representation the resolved key is kept and a warning logged — a
    resolver cannot invent a row that is not on record."""
    if not resolved or _representation_of(resolved) == fmt:
        return resolved
    want_logical = _logical_id(resolved)
    want_owner = _owner_of(resolved)
    siblings = [
        k for k in MODEL_REGISTRY
        if _representation_of(k) == fmt
        and _logical_id(k) == want_logical
        and (not want_owner or _owner_of(k) == want_owner)
    ]
    if not siblings:
        _log.warning("assure_model_key: model_format=%r requested for %r but no "
                     "%s representation of %r is on record; kept %s",
                     fmt, model_key, fmt, resolved, resolved)
        return resolved
    if len(siblings) == 1:
        return siblings[0]
    return _pick_by_total_order(siblings, strict=strict) or siblings[0]


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
    # A qualified request names a publisher, not merely a base-name family.
    # Never let the bare-tail tier silently turn an explicit
    # ``unsloth~Qwen3-Coder-Next-GGUF`` request into the Qwen variant (or vice
    # versa) when one qualified registry row is absent.  The qualified owner
    # may be represented by the registry key or by the model's Hub ID while
    # the registry key itself remains bare.
    if _slugify(_bare_tail(key)) == bare_slug:
        requested_owner = ""
        if "~" in str(model_key):
            requested_owner = str(model_key).split("~", 1)[0].strip()
        elif "/" in str(model_key):
            requested_owner = str(model_key).split("/", 1)[0].strip()
        if requested_owner:
            candidate_owner = ""
            if "~" in str(key):
                candidate_owner = str(key).split("~", 1)[0].strip()
            elif hub_id and "/" in str(hub_id):
                candidate_owner = str(hub_id).split("/", 1)[0].strip()
            if _slugify(candidate_owner) != _slugify(requested_owner):
                return 0
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


def assure_model_key(model_key, *, strict: bool = False, fmt: str = "auto",
                     _expand_aliases: bool = True):
    """Resolve a model reference to a canonical registry key.

    ``fmt`` (model_format: "auto" | "gguf" | "transformers") is the EXPLICIT
    representation override (operator 2026-09-29). "auto" (default) is
    byte-identical to legacy resolution. "gguf"/"transformers" is authoritative:
    the base resolution runs first, then :func:`_redirect_to_format` swaps in the
    same-model, same-owner sibling of the requested representation (ignoring what
    is currently serving and any contradicting name suffix). The GGUF-vs-
    Transformers special-casing downstream keys off the RESOLVED key's framework,
    so nothing beyond the returned key needs to know about ``fmt``.
    """
    resolved = _assure_model_key_base(model_key, strict=strict,
                                      _expand_aliases=_expand_aliases)
    if resolved and fmt and str(fmt).strip().lower() not in ("", "auto"):
        return _redirect_to_format(resolved, model_key, str(fmt).strip().lower(),
                                   strict=strict)
    return resolved


def _assure_model_key_base(model_key, *, strict: bool = False,
                           _expand_aliases: bool = True):
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

    # Literal registry membership always wins first — an alias can only ever
    # redirect a name hugpy does NOT already serve under that spelling.
    if model_key in MODEL_REGISTRY:
        return model_key

    # OPERATOR ALIAS TIER (opt-in): a registered vendor/harness name resolves to
    # its hugpy target. Resolve the TARGET through the normal cascade (with alias
    # expansion OFF, so a chained/cyclic alias can't loop); a broken alias whose
    # target does not resolve is IGNORED — we fall through to the honest refusal
    # for the original name rather than silently substituting a missing model.
    if _expand_aliases:
        target = _alias_target(model_key)
        if target and target != model_key:
            resolved = _assure_model_key_base(target, strict=strict, _expand_aliases=False)
            if resolved is not None:
                _log.debug("assure_model_key: alias %r -> %r -> %s",
                           model_key, target, resolved)
                return resolved

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
        # Same-family variants (notably GGUF vs Transformers) are not a real
        # ambiguity when the caller did not name a packaging.  Let the normal
        # serving-aware pipeline choose the live representation first.  An
        # explicitly qualified request still takes the deterministic exact
        # path below and cannot be redirected across frameworks.
        if not _package_specific(model_key):
            chosen = _pipeline_pick(model_key, top)
            if chosen is not None:
                return chosen
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
