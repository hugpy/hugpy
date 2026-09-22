"""k120 — assembling one dossier, section by section, without ever failing.

This is the orchestrator. It calls the eight section builders in the order
their dependencies allow (the card is fetched once and feeds BOTH the research
digest and the specialization; the screen result feeds the weights; the trial
feeds the verdict) and it wraps each one so that a section which cannot be
built leaves an honest note and the other seven still arrive.

    A dossier is never all-or-nothing. The operator is far better served by
    "everything except the community scan, which Reddit 403'd" than by nothing
    at all — and the nightly timer that produced it must not die because an
    external service was rude.

COST DISCIPLINE
    The expensive parts are gated by the CARD, not by this module:
    ``external_research`` gates the network, ``trial_depth`` gates the GPU,
    ``community`` gates the mention scan, and ``sample_count`` bounds the
    battery. Defaults preserve the pre-k120 behaviour of every existing card —
    a card that has never heard of these knobs screens exactly as it did
    before, then gets the sections that cost nothing.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence

from hugpy_curation.dossier import cards, community as community_mod, research as research_mod
from hugpy_curation.dossier import trial as trial_mod, verdicts, weights as weights_mod
from hugpy_curation.dossier.dossier import Identity, ModelDossier, Source, TrustSignals, utc_now

logger = logging.getLogger(__name__)


def _knob(crit: Any, name: str, default: Any) -> Any:
    """One card knob, defaulted. ``crit`` may be a dataclass, a dict, or an
    older ReviewCriteria that predates the knob entirely — all three read the
    same way, which is what makes the extension additive."""
    if crit is None:
        return default
    if isinstance(crit, Mapping):
        value = crit.get(name, default)
    else:
        value = getattr(crit, name, default)
    return default if value is None else value


def build_identity(hub_id: str, payload: Mapping[str, Any],
                   screen_row: Mapping[str, Any]) -> Identity:
    """Lineage from config + card, nearest first.

    The chain is walked ONE hop (the base model the card names) plus whatever
    the screen already resolved. Walking further would mean a request per hop
    on a nightly job for a fact that is rarely load-bearing; when a deeper
    chain matters the operator can see the base model and click through."""
    org, _, name = hub_id.partition("/")
    if not name:
        org, name = "", hub_id
    base = screen_row.get("base_model")
    lineage: list[str] = [b for b in (base,) if b]
    relation = None
    tags = [str(t).lower() for t in (payload.get("tags") or ())]
    tail = hub_id.split("/")[-1].lower()
    if any(t.startswith("base_model:quantized") for t in tags) or \
            any(k in tail for k in ("gguf", "awq", "gptq", "exl2", "mlx",
                                    "imatrix", "nvfp4")):
        relation = "quantization"
    elif any(t.startswith("base_model:merge") for t in tags) or "merge" in tail:
        relation = "merge"
    elif any(t.startswith("base_model:adapter") for t in tags) or \
            "lora" in tail or "adapter" in tail:
        relation = "adapter"
    elif base:
        relation = "finetune"
    note = ""
    for n in (screen_row.get("notes") or ()):
        if "geometry read from base model" in str(n):
            note = str(n)
    return Identity(hub_id=hub_id, org=org or None, name=name,
                    base_model=base, lineage=tuple(lineage), relation=relation,
                    is_derivative=bool(base) or relation is not None,
                    created_from_note=note)


def build_trust(payload: Mapping[str, Any], screen_row: Mapping[str, Any],
                discussions: Sequence[Any] = ()) -> TrustSignals:
    open_count = sum(1 for m in discussions
                     if str(getattr(m, "snippet", "") or "").lower() == "open")
    license_id = screen_row.get("license") or payload.get("license")
    return TrustSignals(
        downloads=screen_row.get("downloads"), likes=screen_row.get("likes"),
        last_modified=screen_row.get("last_modified"),
        age_days=screen_row.get("age_days"),
        trust_tier=screen_row.get("trust_tier"),
        org_verified=(screen_row.get("trust_tier") or 0) >= 2,
        license=license_id,
        license_url=(f"https://huggingface.co/{screen_row.get('hub_id')}"
                     f"#license" if license_id else None),
        gated=screen_row.get("gated"), private=payload.get("private"),
        discussions_open=open_count if discussions else None,
        discussions_total=len(discussions) or None)


def _screen(hub_id: str, crit: Any, api: Any) -> tuple[dict, dict, dict]:
    """``(screen_row, repo_payload, config)`` — one pass over the reviewer's
    own metadata layer, reused rather than duplicated."""
    from hugpy_curation.review.screen import _config, _repo_info, screen
    payload = _repo_info(api, hub_id) or {}
    row = screen(hub_id, crit, api=api).to_dict()
    files = [s for s in (payload.get("siblings") or []) if s.get("rfilename")]
    config, _source = _config(api, hub_id, files,
                              base_model=row.get("base_model"))
    return row, payload, (config or {})


def build_dossier(hub_id: str, crit: Any = None, *,
                  screen_row: Mapping[str, Any] | None = None,
                  payload: Mapping[str, Any] | None = None,
                  config: Mapping[str, Any] | None = None,
                  api: Any = None, local_path: str | None = None,
                  load: Mapping[str, Any] | None = None,
                  run_dir: str | None = None,
                  dispatch: Callable[[str, str, int], str] | None = None,
                  ) -> tuple[ModelDossier, tuple[str, ...]]:
    """One candidate, comprehensively. Returns ``(dossier, community_urls)``.

    ``community_urls`` are the cached pulls the radar re-reads — see
    ``radar.scan``. Everything else lives on the dossier."""
    dossier = ModelDossier(hub_id=hub_id,
                           criteria=str(_knob(crit, "name", "") or ""))

    # ── the metadata layer (the reviewer's own screen) ───────────────────
    if screen_row is None:
        try:
            screen_row, payload, config = _screen(hub_id, crit, api)
        except Exception as exc:                    # noqa: BLE001
            dossier.add_note(f"metadata screen failed: "
                             f"{type(exc).__name__}: {exc}")
            screen_row, payload, config = {"hub_id": hub_id}, {}, {}
    screen_row = dict(screen_row or {})
    payload = dict(payload or {})
    config = dict(config or {})
    dossier.screen = screen_row
    dossier.add_source(Source(kind="hf-api",
                              url=f"https://huggingface.co/{hub_id}",
                              fetched_at=utc_now(), ok=bool(payload),
                              detail="" if payload else
                              "the hub returned no metadata for this repo"))

    # ── external research (gated by the card) ────────────────────────────
    want_research = bool(_knob(crit, "external_research", True))
    readme = None
    try:
        research, sources, readme = research_mod.build_research(
            hub_id, payload, api=api, enabled=want_research,
            want_notes=want_research and bool(_knob(crit, "judge", True)),
            dispatch=dispatch)
        dossier.research = research
        for source in sources:
            dossier.add_source(source)
    except Exception as exc:                        # noqa: BLE001
        dossier.add_note(f"external research failed: "
                         f"{type(exc).__name__}: {exc}")

    # ── identity, specialization, weights, trust (all local, all cheap) ──
    try:
        dossier.identity = build_identity(hub_id, payload, screen_row)
    except Exception as exc:                        # noqa: BLE001
        dossier.add_note(f"identity failed: {type(exc).__name__}: {exc}")
    try:
        card = dossier.research.card if dossier.research else None
        dossier.specialization = cards.build_specialization(
            hub_id, payload, card, card_text=readme or "")
    except Exception as exc:                        # noqa: BLE001
        dossier.add_note(f"specialization failed: {type(exc).__name__}: {exc}")
    try:
        budget = _knob(crit, "usable_vram_bytes", None)
        if budget is None:
            budget = _knob(crit, "vram_bytes", None)
        dossier.weights = weights_mod.build_weights(
            hub_id, screen_row, config, vram_budget_bytes=budget,
            target_context=int(_knob(crit, "target_context", 16384)))
    except Exception as exc:                        # noqa: BLE001
        dossier.add_note(f"weights failed: {type(exc).__name__}: {exc}")

    # ── community intelligence (gated by the card) ───────────────────────
    urls: tuple[str, ...] = ()
    if _knob(crit, "community", True):
        try:
            found, urls = community_mod.gather(
                hub_id, sources=tuple(_knob(crit, "community_sources",
                                            community_mod.DEFAULT_SOURCES)),
                subreddits=tuple(_knob(crit, "subreddits",
                                       community_mod.DEFAULT_SUBREDDITS)),
                api=api, want_claims=bool(_knob(crit, "judge", True)),
                dispatch=dispatch)
            dossier.community = found
        except Exception as exc:                    # noqa: BLE001
            dossier.add_note(f"community scan failed: "
                             f"{type(exc).__name__}: {exc}")
    else:
        dossier.add_note("community scan is off for this card (community: false)")

    try:
        dossier.trust = build_trust(
            payload, screen_row,
            discussions=[m for m in (dossier.community.mentions
                                     if dossier.community else ())
                         if m.source == "hf-discussions"])
    except Exception as exc:                        # noqa: BLE001
        dossier.add_note(f"trust failed: {type(exc).__name__}: {exc}")

    # ── the trial ────────────────────────────────────────────────────────
    depth = str(_knob(crit, "trial_depth", "load-test"))
    try:
        modality = (dossier.specialization.modality
                    if dossier.specialization else None)
        dossier.trial = trial_mod.run_trial(
            hub_id, modality=modality, depth=depth,
            sample_count=int(_knob(crit, "sample_count", 2)),
            operations=tuple(_knob(crit, "compare_against", ())),
            local_path=local_path, gated=screen_row.get("gated"), load=load,
            compare_against=tuple(_knob(crit, "compare_against", ())),
            run_dir=run_dir)
    except Exception as exc:                        # noqa: BLE001
        dossier.trial = trial_mod.blocked(
            depth, f"the trial could not run: {type(exc).__name__}: {exc}")

    # ── the verdict, which must cite the above ───────────────────────────
    try:
        dossier.verdict = verdicts.decide(
            dossier, judge=bool(_knob(crit, "judge", True)), dispatch=dispatch)
    except Exception as exc:                        # noqa: BLE001
        dossier.add_note(f"verdict failed: {type(exc).__name__}: {exc}")

    return dossier, urls


__all__ = ["build_dossier", "build_identity", "build_trust"]
