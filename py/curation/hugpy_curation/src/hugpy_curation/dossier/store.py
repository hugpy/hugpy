"""k120 — where dossiers live.

A dossier is 20-60 KB of JSON: a card digest, every quant, every mention, every
sample. The review DB's ``payload`` column is not the place for that — it is
pushed worker->central in batches of 250 and it is read on every console poll.

So: the FULL dossier is a file, and the review row carries a compact
:func:`summary` plus the path. The console lists from the summary and expands
one dossier at a time, which is also how an operator actually reads them.

    <DEFAULT_ROOT>/review/dossiers/<criteria>/<org__repo>.json

One file per (criteria, repo), overwritten by the newest review — the same
"newest wins" rule the review store already uses, and the reason a re-review
does not accumulate twelve copies of a model card. ``DOSSIER_DIR`` overrides
the location for a worker that keeps its own.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable, Mapping

from hugpy_curation.dossier.dossier import ModelDossier

logger = logging.getLogger(__name__)

ENV_DIR: str = "DOSSIER_DIR"

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def root_dir() -> str:
    """``DOSSIER_DIR``, else ``<review_root>/dossiers`` — see
    :mod:`hugpy_curation.config` for the whole resolution order."""
    from hugpy_curation.config import dossiers_root
    return dossiers_root()


def _safe(name: str) -> str:
    return _SAFE.sub("-", (name or "").replace("/", "__")).strip("-") or "unnamed"


def path_for(criteria: str, hub_id: str) -> str:
    return os.path.join(root_dir(), _safe(criteria), f"{_safe(hub_id)}.json")


def save(dossier: ModelDossier) -> str | None:
    """Write one dossier. Returns its path, or None when it could not be
    written — a dossier that cannot be filed must not fail the review that
    produced it, so this logs and returns None instead of raising."""
    path = path_for(dossier.criteria or "adhoc", dossier.hub_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(dossier.to_json())
        os.replace(tmp, path)                       # atomic: the API may read
        return path
    except OSError as exc:
        logger.warning("dossier store: could not write %s (%s)", path, exc)
        return None


def load(criteria: str, hub_id: str) -> ModelDossier | None:
    path = path_for(criteria, hub_id)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return ModelDossier.from_dict(json.load(fh))
    except (OSError, ValueError):
        return None


def load_path(path: str) -> ModelDossier | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return ModelDossier.from_dict(json.load(fh))
    except (OSError, ValueError):
        return None


def list_for(criteria: str) -> list[str]:
    """Every dossier file for one card, newest first."""
    directory = os.path.join(root_dir(), _safe(criteria))
    try:
        names = [n for n in os.listdir(directory) if n.endswith(".json")]
    except OSError:
        return []
    rows = []
    for name in names:
        full = os.path.join(directory, name)
        try:
            rows.append((os.path.getmtime(full), full))
        except OSError:
            continue
    rows.sort(reverse=True)
    return [p for _mtime, p in rows]


def summary(dossier: ModelDossier, path: str | None = None) -> dict[str, Any]:
    """The compact row that rides in the review payload.

    Everything the console's LIST view shows, and nothing it does not: the
    headline, the fit, the verdict with its reason count, the margin against
    the incumbent, and the path to the rest."""
    spec = dossier.specialization
    weights = dossier.weights
    trial = dossier.trial
    verdict = dossier.verdict
    best = weights.quant(weights.best_quant) if weights else None
    wins = [c for c in (trial.comparisons if trial else ())
            if c.beats_incumbent == "yes"]
    best_margin = max((c.margin for c in (trial.comparisons if trial else ())
                       if c.margin is not None), default=None)
    return {
        "schema": dossier.schema, "hub_id": dossier.hub_id,
        "criteria": dossier.criteria, "generated_at": dossier.generated_at,
        "path": path,
        "specialization": spec.headline if spec else None,
        "domains": list(spec.domains) if spec else [],
        "params": weights.params if weights else None,
        "best_quant": weights.best_quant if weights else None,
        "est_vram_bytes": best.est_vram_bytes if best else None,
        "context_length": weights.context_length if weights else None,
        "license": dossier.trust.license if dossier.trust else None,
        "gated": dossier.trust.gated if dossier.trust else None,
        "community_heat": dossier.community.heat if dossier.community else None,
        "community_claims": (len(dossier.community.claims)
                             if dossier.community else 0),
        "papers": len(dossier.research.papers) if dossier.research else 0,
        "has_research_notes": bool(dossier.research
                                   and dossier.research.research_notes),
        "trial_depth": trial.depth if trial else None,
        "trial_backend": trial.backend if trial else None,
        "trial_blocked": trial.blocked if trial else None,
        "samples": len(trial.samples) if trial else 0,
        "mean_quality": trial.mean_quality if trial else None,
        "beats_incumbent": ("yes" if wins else
                            ("no" if trial and trial.comparisons else
                             "untested")),
        "margin": best_margin,
        "verdict": verdict.verdict if verdict else None,
        "verdict_reasons": len(verdict.reasons) if verdict else 0,
        "verdict_confidence": verdict.confidence if verdict else None,
        "unavailable": list(dossier.unavailable),
    }


def save_radar(criteria: str, hits: Iterable[Mapping[str, Any]],
               detail: str = "") -> str | None:
    """The gem radar's output for one card. One file, replaced each run."""
    path = os.path.join(root_dir(), _safe(criteria), "_radar.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"criteria": criteria, "detail": detail,
                       "hits": list(hits)}, fh, indent=2, default=str)
        os.replace(tmp, path)
        return path
    except OSError as exc:
        logger.warning("dossier store: could not write radar %s (%s)", path, exc)
        return None


def load_radar(criteria: str) -> dict[str, Any]:
    path = os.path.join(root_dir(), _safe(criteria), "_radar.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


__all__ = ["ENV_DIR", "list_for", "load", "load_path", "load_radar",
           "path_for", "root_dir", "save", "save_radar", "summary"]
