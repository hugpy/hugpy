"""Shared grouping/percentile logic for the per-query tok/s ledger.

ONE implementation, imported by BOTH the ``GET /llm/toks/report`` route (via
``worker_routes``) and the ``tools/toks_report.py`` CLI, so the console table
and the terminal table can never drift. Pure and dependency-free on purpose:
the CLI must read ``toks_log.jsonl`` without importing the Flask app.

Each JSONL line is one completed relay. Entries are grouped by the triple
``(worker, model, config_key)`` — the unit the allocation study compares — and
each group reports ``n``, ``mean``, ``p50`` and ``p95`` of ``tok_s``. Groups are
returned best-mean first, which is the whole point: "which configuration gave
the best tok/s".
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def percentile(values: List[float], q: float) -> Optional[float]:
    """The ``q`` (0..100) percentile via linear interpolation. None if empty.

    Nearest-rank with interpolation between the two straddling samples — the
    same method ``numpy.percentile`` uses by default, reimplemented here so the
    ledger tools carry no numpy dependency."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _config_key_of(entry: Dict[str, Any]) -> str:
    """The config_key stamped into the entry's state, or ``"-"`` when a
    pre-snapshot row has none (still groupable by worker+model)."""
    state = entry.get("state")
    if isinstance(state, dict):
        ck = state.get("config_key")
        if ck:
            return str(ck)
    return "-"


def group_entries(entries: Iterable[Dict[str, Any]],
                  worker: Optional[str] = None,
                  model: Optional[str] = None) -> List[Dict[str, Any]]:
    """Group ledger entries by ``(worker, model, config_key)``.

    ``worker``/``model`` optionally filter first. Returns a list of
    ``{worker, model, config_key, n, mean, p50, p95}`` dicts sorted best-mean
    first (ties broken by higher n, then worker/model). A row with no usable
    ``tok_s`` is skipped rather than counted as zero."""
    buckets: Dict[tuple, List[float]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        wid = e.get("worker_id")
        mk = e.get("model_key")
        if worker and wid != worker:
            continue
        if model and mk != model:
            continue
        try:
            v = float(e.get("tok_s"))
        except (TypeError, ValueError):
            continue
        if not (v > 0.0) or v != v:
            continue
        buckets.setdefault((wid, mk, _config_key_of(e)), []).append(v)

    rows: List[Dict[str, Any]] = []
    for (wid, mk, ck), vals in buckets.items():
        n = len(vals)
        mean = sum(vals) / n
        rows.append({
            "worker": wid, "model": mk, "config_key": ck, "n": n,
            "mean": round(mean, 3),
            "p50": round(percentile(vals, 50) or 0.0, 3),
            "p95": round(percentile(vals, 95) or 0.0, 3),
        })
    rows.sort(key=lambda r: (-r["mean"], -r["n"],
                             str(r["worker"]), str(r["model"])))
    return rows
