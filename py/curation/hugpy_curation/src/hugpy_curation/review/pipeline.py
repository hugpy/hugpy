"""review/pipeline.py — search → screen → download survivors → smoke → judge.

The two-stage shape is the whole point: screening is metadata-only and costs
nothing, so it runs over the entire candidate pool; downloading and loading
cost disk, bandwidth and GPU time, so they only ever run on what survived. A
run that screens sixty repos might download two.

`max_downloads_per_run` is a hard cap, not a suggestion — this runs unattended
on a timer and the model store is finite.

A finished run is handed to central (best-effort) so the operator's console
reads one source of truth instead of whichever box happened to run the timer.
Configured by REVIEW_CENTRAL_URL + REVIEW_CENTRAL_TOKEN (falling back to
WORKER_ENROLL_TOKEN) — see push.py. Unset = off, and a push that fails costs
one log line, never the run.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict

from hugpy_curation.review import store
from hugpy_curation.review.criteria import ReviewCriteria
from hugpy_curation.review.screen import ScreenResult, screen, search_candidates, _hf_api

# Don't re-screen a repo the timer already looked at this recently.
RESCREEN_AFTER_SECONDS = float(os.environ.get("REVIEW_RESCREEN_AFTER") or 7 * 86400)


@dataclass
class Review:
    """Everything known about one candidate after a run."""
    hub_id: str
    criteria: str
    stage: str = "screened"                 # screened | downloaded | smoked
    screen: dict = field(default_factory=dict)
    smoke: dict | None = None
    judgement: dict | None = None
    local_path: str | None = None
    error: str | None = None
    reviewed_at: float = field(default_factory=time.time)
    # k120: the compact dossier summary (the FULL dossier is a file — see
    # discovery_dossier/store.py). Kept small on purpose: this dict rides in
    # the review row that gets pushed to central in batches and read on every
    # console poll, and a 40 KB model card in it would make both worse.
    dossier: dict | None = None

    def to_dict(self):
        return asdict(self)

    @property
    def passed(self) -> bool:
        return bool(self.screen.get("passed"))

    @property
    def score(self) -> float:
        return float(self.screen.get("score") or 0.0)

    @property
    def verdict(self) -> str | None:
        if self.judgement:
            return self.judgement.get("verdict")
        return None


# ── download ───────────────────────────────────────────────────────────────
def _download(hub_id: str, quant: str, files: list[str]) -> str:
    """Fetch one quant through ``hugpy_storage`` and return the directory it
    landed in. See :mod:`hugpy_curation.review.download`: the storage daemon's
    queue when one is alive, storage's synchronous ``download_one`` in this
    process otherwise. Raises on any non-completed outcome so ``review_one``
    records it as a 'download failed' row."""
    from hugpy_curation.review.download import download_quant
    return download_quant(hub_id, quant, files)


def _dossier(rv: "Review", sc, crit: ReviewCriteria, api=None,
             blocked: str | None = None, log=print) -> None:
    """k120: build the candidate's dossier and hang its SUMMARY on the review.

    Called for every candidate that passed the screen, including the ones whose
    download or load failed — a dossier that says "trial blocked: download
    failed: <cause>" is the honest row the operator asked for, and it is the
    only way a broken download path shows up as a finding instead of as a
    missing model.

    Never raises and never changes the review's own verdict path: if the
    dossier package is absent or throws, the review completes exactly as it did
    before k120."""
    if not getattr(crit, "dossier", True):
        return
    try:
        from hugpy_curation.dossier import store as dstore
        from hugpy_curation.dossier.build import build_dossier
        dossier, urls = build_dossier(
            rv.hub_id, crit, screen_row=sc.to_dict() if hasattr(sc, "to_dict")
            else dict(sc or {}), api=api, local_path=rv.local_path,
            load=rv.smoke)
        # The caller's `blocked` string describes what the REVIEW could not do
        # (download, load). It is only the truth about the trial when the trial
        # itself produced nothing: a card asking for full-samples can run the
        # stationary battery through the fleet's own dispatch path WITHOUT a
        # download, and stamping "no download" over real evidence would be a
        # lie in the one field that exists to stop lies.
        if blocked and dossier.trial is not None \
                and not dossier.trial.has_evidence and not dossier.trial.blocked:
            dossier.trial.blocked = blocked
        path = dstore.save(dossier)
        rv.dossier = dstore.summary(dossier, path)
        rv.dossier["community_urls"] = list(urls)
        verdict = dossier.verdict
        if verdict is not None:
            log(f"    dossier: {verdict.verdict} — "
                f"{'; '.join(verdict.reasons[:2])[:160]}")
    except Exception as exc:                        # noqa: BLE001
        log(f"    (dossier unavailable: {type(exc).__name__}: {exc})")


def _find_gguf(directory: str, quant: str) -> str | None:
    """First shard of `quant` under `directory` — what llama.cpp is handed."""
    if not directory or not os.path.isdir(directory):
        return None
    hits = []
    for root, _dirs, names in os.walk(directory):
        for n in names:
            if n.lower().endswith(".gguf") and not n.lower().startswith("mmproj"):
                if quant.upper() in n.upper():
                    hits.append(os.path.join(root, n))
    if not hits:
        for root, _dirs, names in os.walk(directory):
            for n in names:
                if n.lower().endswith(".gguf") and not n.lower().startswith("mmproj"):
                    hits.append(os.path.join(root, n))
    return sorted(hits)[0] if hits else None


# ── one candidate, end to end ──────────────────────────────────────────────
def review_one(hub_id: str, crit: ReviewCriteria, api=None,
               download: bool = True, log=print,
               run_id: int | None = None) -> Review:
    """Screen one repo and, if it passes and `download` is set, fetch and load
    it. Never raises: a failure at any stage is recorded on the Review.

    ``run_id`` stamps every row this call writes with the run that owns it, so
    ``push`` can ship a run and exactly its results as one batch. None for an
    ad-hoc CLI review, which belongs to no run."""
    rv = Review(hub_id=hub_id, criteria=crit.name)
    try:
        sc: ScreenResult = screen(hub_id, crit, api=api)
    except Exception as exc:
        rv.error = f"screen failed: {type(exc).__name__}: {exc}"
        store.record(crit.name, hub_id, "screened", rv.to_dict(), passed=False,
                     run_id=run_id)
        return rv
    rv.screen = sc.to_dict()

    if not sc.passed:
        log(f"  ✗ {hub_id}: {'; '.join(sc.reasons)}")
        store.record(crit.name, hub_id, "screened", rv.to_dict(),
                     passed=False, score=sc.score, run_id=run_id)
        return rv

    log(f"  ✓ {hub_id} score={sc.score} quant={sc.best_quant} "
        f"vram≈{(sc.est_vram_bytes or 0)/1024**3:.1f}GiB")
    if not download or not crit.smoke_test or not sc.best_quant:
        # Only the EXPENSIVE pass (download=True) gets a dossier. Stage 1 of a
        # run screens the whole pool with download=False, and building sixty
        # dossiers — sixty card fetches, sixty mention scans, sixty LLM
        # summaries — would destroy the property that makes the two-stage shape
        # worth having. The survivors come back through here with download=True.
        if download:
            _dossier(rv, sc, crit, api=api, log=log, blocked=(
                "trial blocked: no acceptable quant to download"
                if not sc.best_quant else
                "screened only — smoke_test is off for this card"))
        store.record(crit.name, hub_id, "screened", rv.to_dict(),
                     passed=True, score=sc.score, run_id=run_id)
        return rv

    quant_files = next((q["files"] for q in sc.quants
                        if q["quant"] == sc.best_quant), [])
    try:
        log(f"    downloading {sc.best_quant} …")
        directory = _download(hub_id, sc.best_quant, quant_files)
        rv.local_path = _find_gguf(directory, sc.best_quant)
        rv.stage = "downloaded"
    except Exception as exc:
        rv.error = f"download failed: {type(exc).__name__}: {exc}"
        log(f"    ! {rv.error}")
        _dossier(rv, sc, crit, api=api, log=log,
                 blocked=f"trial blocked: {rv.error}")
        store.record(crit.name, hub_id, "screened", rv.to_dict(),
                     passed=True, score=sc.score, run_id=run_id)
        return rv

    if not rv.local_path:
        rv.error = "download completed but no .gguf found on disk"
        _dossier(rv, sc, crit, api=api, log=log,
                 blocked=f"trial blocked: {rv.error}")
        store.record(crit.name, hub_id, "downloaded", rv.to_dict(),
                     passed=True, score=sc.score, run_id=run_id)
        return rv

    from hugpy_curation.review.smoke import smoke_test
    log(f"    loading {os.path.basename(rv.local_path)} …")
    sm = smoke_test(rv.local_path,
                    n_ctx=min(crit.target_context, sc.context_length or crit.target_context))
    rv.smoke = sm.to_dict()
    rv.stage = "smoked"

    if not sm.ok:
        rv.error = sm.error
        log(f"    ! smoke test failed: {sm.error}")
    else:
        log(f"    {sm.gen_tokens_per_sec} tok/s · load {sm.load_seconds}s · "
            f"coherence {sm.coherence}")
        if sm.gen_tokens_per_sec is not None and \
                sm.gen_tokens_per_sec < crit.min_tokens_per_sec:
            rv.screen["passed"] = False
            rv.screen.setdefault("reasons", []).append(
                f"{sm.gen_tokens_per_sec} tok/s < required {crit.min_tokens_per_sec}")
            log(f"    ✗ too slow — {sm.gen_tokens_per_sec} tok/s")

    if crit.judge:
        try:
            from hugpy_curation.review.judge import judge as _judge
            rv.judgement = _judge(sc, sm, crit)
            if rv.judgement:
                log(f"    agent: {rv.judgement.get('verdict')} — "
                    f"{rv.judgement.get('summary', '')[:120]}")
        except Exception as exc:
            log(f"    (judge unavailable: {type(exc).__name__}: {exc})")

    _dossier(rv, sc, crit, api=api, log=log)
    store.record(crit.name, hub_id, rv.stage, rv.to_dict(),
                 passed=rv.passed, score=rv.score, verdict=rv.verdict,
                 run_id=run_id)
    return rv


# ── a full run ─────────────────────────────────────────────────────────────
def run(crit: ReviewCriteria, hub_ids: list[str] | None = None,
        force: bool = False, log=print) -> dict:
    """Screen a pool, then download+load the best survivors up to the run cap.

    A criteria the console switched off (``enabled: false``, 2026-08-13)
    no-ops here — BEFORE start_run, so a skipped night writes no run row.
    The timer keeps firing; this is the cheap, root-less off switch. An
    explicit ask still runs: ``force=True`` (the console's "Run now") or a
    caller-supplied ``hub_ids`` list (screening specific repos was requested
    by a human, not the schedule)."""
    if not getattr(crit, "enabled", True) and not force and not hub_ids:
        log(f"criteria '{crit.name}' is disabled (enabled: false) — skipping "
            f"run; flip it on in the console settings or pass force")
        return {"run_id": None, "criteria": crit.name, "skipped": "disabled"}
    run_id = store.start_run(crit.name)
    api = _hf_api()
    counts = {"screened": 0, "passed": 0, "downloaded": 0, "smoked": 0}
    reviews: list[Review] = []
    try:
        candidates = hub_ids or search_candidates(crit, api=api)
        if not candidates:
            log("no candidates from the hub search")
        else:
            log(f"screening {len(candidates)} candidates for '{crit.name}'")

        # stage 1 — cheap, over everything
        screened: list[Review] = []
        for hub_id in candidates:
            if not force and not hub_ids and \
                    store.seen_since(crit.name, hub_id, RESCREEN_AFTER_SECONDS):
                continue
            rv = review_one(hub_id, crit, api=api, download=False, log=log,
                            run_id=run_id)
            counts["screened"] += 1
            screened.append(rv)
            if rv.passed:
                counts["passed"] += 1

        # stage 2 — expensive, only on survivors, best first, capped
        survivors = sorted([r for r in screened if r.passed],
                           key=lambda r: r.score, reverse=True)
        budget = crit.max_downloads_per_run if not hub_ids else len(survivors)
        if len(survivors) > budget:
            log(f"{len(survivors)} passed; downloading the top {budget} "
                f"(cap: max_downloads_per_run) — the rest keep their screen result")
        for rv in survivors[:budget]:
            full = review_one(rv.hub_id, crit, api=api, download=True, log=log,
                              run_id=run_id)
            reviews.append(full)
            if full.local_path:
                counts["downloaded"] += 1
            if full.smoke:
                counts["smoked"] += 1
        reviews.extend(survivors[budget:])
        reviews.extend([r for r in screened if not r.passed])
    except Exception as exc:
        store.finish_run(run_id, error=f"{type(exc).__name__}: {exc}", **counts)
        # A crashed run still produced rows worth seeing on the console — push
        # what there is, THEN re-raise so the caller's failure handling is
        # unchanged.
        _push(run_id, log)
        raise
    store.finish_run(run_id, **counts)
    _push(run_id, log)

    return {"run_id": run_id, "criteria": crit.name, **counts,
            "radar": _radar(crit, screened + reviews, log),
            "reviews": [r.to_dict() for r in reviews]}


def _radar(crit: ReviewCriteria, reviews: list["Review"], log) -> list[dict]:
    """k120 GEM RADAR — a second pass over the SAME cached community pulls.

    Costs no requests: ``community.gather`` already fetched and cached those
    listings for the candidates, and this re-reads them looking for model names
    NO card is tracking. Off unless the card asks (``radar: true``), because a
    tip nobody asked for is noise.

    ``known`` is every hub id this run touched plus every hub id the store has
    ever recorded for this criteria — a radar that kept re-reporting the
    incumbent would be worse than none."""
    if not getattr(crit, "radar", False):
        return []
    try:
        from hugpy_curation.dossier import store as dstore
        from hugpy_curation.dossier.radar import scan
        known = {r.hub_id for r in reviews}
        known.update(row.get("hub_id") for row in
                     store.recent(crit.name, limit=400) if row.get("hub_id"))
        known.update(crit.incumbents or [])
        hits, provenance = scan(known)
        dstore.save_radar(crit.name, [h.to_dict() for h in hits],
                          detail=provenance.detail)
        if hits:
            log(f"radar: {len(hits)} model(s) the cards are not tracking — "
                + ", ".join(h.hub_id or h.name for h in hits[:5]))
        else:
            log(f"radar: {provenance.detail}")
        return [h.to_dict() for h in hits]
    except Exception as exc:                        # noqa: BLE001
        log(f"radar unavailable: {type(exc).__name__}: {exc}")
        return []


def _push(run_id: int, log) -> None:
    """Best-effort hand-off of a finished run to central.

    Guarded exactly like the eviction relay: central's DB is where the operator
    looks, but the LOCAL db write already happened and is the on-box record, so
    a push that cannot happen costs one log line and nothing else. Off entirely
    unless REVIEW_CENTRAL_URL is set — see push.py."""
    try:
        from hugpy_curation.review.push import push_run
        push_run(run_id, log=log)
    except Exception as exc:                     # noqa: BLE001
        # push_run does not raise; this catches an import-time failure only
        # (e.g. a partially installed release). Still never fails the run.
        log(f"review push unavailable: {type(exc).__name__}: {exc}")


# ── reporting ──────────────────────────────────────────────────────────────
def report_markdown(result: dict) -> str:
    """Human-readable run summary — what the timer leaves behind to be read."""
    lines = [f"# model review — {result.get('criteria')}", ""]
    lines.append(f"screened {result.get('screened', 0)} · "
                 f"passed {result.get('passed', 0)} · "
                 f"downloaded {result.get('downloaded', 0)} · "
                 f"load-tested {result.get('smoked', 0)}")
    lines.append("")

    smoked = [r for r in result.get("reviews", []) if r.get("smoke")]
    passed = [r for r in result.get("reviews", [])
              if r.get("screen", {}).get("passed") and not r.get("smoke")]
    failed = [r for r in result.get("reviews", [])
              if not r.get("screen", {}).get("passed")]

    if smoked:
        lines += ["## load-tested", ""]
        for r in sorted(smoked, key=lambda x: -(x.get("screen", {}).get("score") or 0)):
            s, k = r.get("screen", {}), r.get("smoke", {})
            j = r.get("judgement") or {}
            lines.append(f"### {r['hub_id']}")
            lines.append(
                f"- {s.get('best_quant')} · {(s.get('est_vram_bytes') or 0)/1024**3:.1f} GiB est VRAM"
                f" · ctx {s.get('context_length')} · trust tier {s.get('trust_tier')}")
            if k.get("ok"):
                lines.append(
                    f"- **{k.get('gen_tokens_per_sec')} tok/s**, load {k.get('load_seconds')}s,"
                    f" measured VRAM {(k.get('vram_used_bytes') or 0)/1024**3:.1f} GiB,"
                    f" coherence {k.get('coherence')}")
            else:
                lines.append(f"- **failed to load**: {k.get('error')}")
            if j.get("verdict"):
                lines.append(f"- agent verdict: **{j['verdict']}** — {j.get('summary', '')}")
                for w in (j.get("weaknesses") or [])[:3]:
                    lines.append(f"  - weakness: {w}")
            lines.append("")

    if passed:
        lines += ["## passed the screen (not downloaded this run)", ""]
        for r in sorted(passed, key=lambda x: -(x.get("screen", {}).get("score") or 0)):
            s = r.get("screen", {})
            lines.append(f"- `{r['hub_id']}` score {s.get('score')} · "
                         f"{s.get('best_quant')} · "
                         f"{(s.get('est_vram_bytes') or 0)/1024**3:.1f} GiB")
        lines.append("")

    if failed:
        lines += ["## rejected", ""]
        for r in failed[:40]:
            reasons = "; ".join((r.get("screen", {}).get("reasons") or [])[:2]) \
                or r.get("error") or "unknown"
            lines.append(f"- `{r['hub_id']}` — {reasons}")
        if len(failed) > 40:
            lines.append(f"- …and {len(failed) - 40} more")
        lines.append("")
    return "\n".join(lines)
