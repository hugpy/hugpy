"""Per-call model selection (k113a): every inference point in a build asks,
at the moment it executes, "which model, and why" — and the answer is
evidence, not a default.

Today's router (``router.resolve_route``) picks ``requested → deterministic
→ only-eligible → TASK_DEFAULTS`` and never reads the benchmark matrix, VRAM
hints, the quality profile, or any record of outcomes. This module is the
ordered resolution the directive (§4) mandates, producing a ``requested_model``
the router then executes with. It sits ABOVE the router; it does not replace
the authority gate or the catalog (those stay authoritative).

Order, each step recorded per candidate as selected/rejected + reason:

 1. compatibility   — capability known and eligible on the fleet (catalog view)
 2. authority       — delegated to the router's typed gate (recorded, not re-done)
 3. registration    — the view's eligible ``model_ids`` (probe-failed views never reach here)
 4. health          — ``model_health(model_id)`` seam (None = unknown, passes)
 5. resources       — per-model VRAM (seam) vs node ResourceRequest / goal budget
 6. quality profile — PREVIEW favours speed, BEST favours quality, BALANCED blends
 7. latency budget  — matrix latency vs ``goal.budget.max_seconds``
 8. reliability     — the ReliabilityLedger: measured outcomes for THIS capability
                      (scorecard hard-pass, failures, repair codes, latency) — a model
                      that keeps failing a step loses it during the run, not after
                      the next benchmark sweep
 9. recommendation  — routing-matrix primary/fallback + candidate evidence

Candidate spread: a FANOUT with N candidates sends candidate *i* to the i-th
ranked distinct model whose score is within ``spread_margin`` of the best,
so the judge compares models, not seeds. Only one eligible model → seeds vary.

Everything is stdlib; the catalog/matrix/health seams are injectable so the
selector is unit-testable without a fleet.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from hugpy_oracle.contracts import GoalSpec, QualityProfile, RepairCode
from hugpy_oracle.plan import PlanNode

__all__ = [
    "CandidateVerdict",
    "ModelStats",
    "ReliabilityLedger",
    "SelectionDecision",
    "SelectionPolicy",
    "Selector",
    "note_execution",
    "note_verdict",
    "note_verdict_for_ref",
    "pinned",
    "producer_of",
    "producer_stamp",
    "remember_producer",
    "operation_for",
    "process_selector",
    "requested_model_for",
    "select",
]


_log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# reliability ledger
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ModelStats:
    capability: str
    model_id: str
    n: int
    ok_rate: float            # executed without error
    pass_rate: float          # scorecard hard_pass among executed
    mean_latency_s: float | None
    repair_codes: tuple[tuple[str, int], ...] = ()   # (code, count) most frequent first

    def to_dict(self) -> dict[str, Any]:
        return {"capability": self.capability, "model_id": self.model_id, "n": self.n,
                "ok_rate": self.ok_rate, "pass_rate": self.pass_rate,
                "mean_latency_s": self.mean_latency_s,
                "repair_codes": [list(x) for x in self.repair_codes]}


class ReliabilityLedger:
    """Append-only record of what each model actually did at each capability.
    SQLite WAL, one table. Window-limited reads so the record is RECENT."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            "CREATE TABLE IF NOT EXISTS outcomes ("
            " seq INTEGER PRIMARY KEY AUTOINCREMENT, capability TEXT NOT NULL,"
            " model_id TEXT NOT NULL, operation TEXT, ok INTEGER NOT NULL,"
            " hard_pass INTEGER, repair_code TEXT, latency_s REAL,"
            " run_id TEXT, node_id TEXT, attempt INTEGER, at TEXT NOT NULL,"
            " score REAL, explored INTEGER NOT NULL DEFAULT 0);"
            "CREATE INDEX IF NOT EXISTS outcomes_cap_model ON outcomes(capability, model_id, seq);"
            "CREATE TABLE IF NOT EXISTS producers ("
            " ref TEXT PRIMARY KEY, capability TEXT NOT NULL, model_id TEXT NOT NULL,"
            " ts TEXT NOT NULL, worker TEXT);"
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(self, capability: str, model_id: str, *, ok: bool, hard_pass: bool | None = None,
               repair_code: RepairCode | str | None = None, latency_s: float | None = None,
               operation: str | None = None, run_id: str | None = None,
               node_id: str | None = None, attempt: int | None = None,
               score: float | None = None, explored: bool = False) -> None:
        code = getattr(repair_code, "value", repair_code)
        with self._lock:
            self._conn.execute(
                "INSERT INTO outcomes(capability, model_id, operation, ok, hard_pass, repair_code,"
                " latency_s, run_id, node_id, attempt, at, score, explored)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (capability, model_id, operation, 1 if ok else 0,
                 None if hard_pass is None else (1 if hard_pass else 0), code, latency_s,
                 run_id, node_id, attempt, _utc_now(), score, 1 if explored else 0),
            )

    def recent(self, capability: str | None = None, *, limit: int = 500) -> list[dict[str, Any]]:
        """Newest-first raw rows (for the steward's calibration / streak checks)."""
        if capability is None:
            rows = self._conn.execute("SELECT * FROM outcomes ORDER BY seq DESC LIMIT ?", (int(limit),)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM outcomes WHERE capability=? ORDER BY seq DESC LIMIT ?",
                                      (capability, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def capabilities(self) -> tuple[str, ...]:
        rows = self._conn.execute("SELECT DISTINCT capability FROM outcomes ORDER BY capability").fetchall()
        return tuple(r["capability"] for r in rows)

    def calls_since_model_tried(self, capability: str, model_id: str) -> int | None:
        """How many outcomes at ``capability`` since ``model_id`` was last tried
        (None = never tried)."""
        last = self._conn.execute("SELECT seq FROM outcomes WHERE capability=? AND model_id=?"
                                  " ORDER BY seq DESC LIMIT 1", (capability, model_id)).fetchone()
        if last is None:
            return None
        n = self._conn.execute("SELECT COUNT(*) FROM outcomes WHERE capability=? AND seq>?",
                               (capability, last["seq"])).fetchone()[0]
        return int(n)

    def stats(self, capability: str, model_id: str, *, window: int = 50) -> ModelStats:
        rows = self._conn.execute(
            "SELECT ok, hard_pass, repair_code, latency_s FROM outcomes WHERE capability=? AND"
            " model_id=? ORDER BY seq DESC LIMIT ?", (capability, model_id, int(window)),
        ).fetchall()
        n = len(rows)
        if n == 0:
            return ModelStats(capability, model_id, 0, 0.0, 0.0, None)
        ok = [r for r in rows if r["ok"]]
        judged = [r for r in ok if r["hard_pass"] is not None]
        passed = [r for r in judged if r["hard_pass"]]
        lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
        codes: dict[str, int] = {}
        for r in rows:
            if r["repair_code"]:
                codes[r["repair_code"]] = codes.get(r["repair_code"], 0) + 1
        ranked = tuple(sorted(codes.items(), key=lambda kv: (-kv[1], kv[0])))
        return ModelStats(
            capability, model_id, n, len(ok) / n,
            (len(passed) / len(judged)) if judged else (1.0 if ok else 0.0),
            (sum(lat) / len(lat)) if lat else None, ranked,
        )

    def stats_for(self, capability: str, model_ids: Iterable[str], *, window: int = 50
                  ) -> dict[str, ModelStats]:
        return {m: self.stats(capability, m, window=window) for m in model_ids}

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])

    # -- producer attribution (or-k16): artifact ref -> (capability, model_id) --
    # Persisted in the SAME sqlite file so every process that opens the ledger
    # (worker, judge, central) sees the same truth. Upsert: the first producer
    # of a ref is kept unless an explicit re-attribution arrives.

    def remember_producer(self, ref: str, capability: str, model_id: str, *,
                          worker: str | None = None, ts: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO producers(ref, capability, model_id, ts, worker) VALUES(?,?,?,?,?)"
                " ON CONFLICT(ref) DO UPDATE SET capability=excluded.capability,"
                " model_id=excluded.model_id, ts=excluded.ts, worker=excluded.worker",
                (str(ref), capability, str(model_id), ts or _utc_now(), worker),
            )

    def producer(self, ref: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM producers WHERE ref=?", (str(ref),)).fetchone()
        return dict(row) if row is not None else None

    def producers(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM producers ORDER BY ts DESC, ref LIMIT ?",
                                      (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def producer_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM producers").fetchone()[0])

    def summary(self) -> dict[str, Any]:
        """Operator-facing snapshot: row counts, per-(capability, model)
        outcome tallies, producer counts per (capability, model), workers seen."""
        with self._lock:
            outcomes = self._conn.execute(
                "SELECT capability, model_id, COUNT(*) AS n, SUM(ok) AS ok,"
                " SUM(CASE WHEN hard_pass=1 THEN 1 ELSE 0 END) AS passed,"
                " SUM(CASE WHEN hard_pass=0 THEN 1 ELSE 0 END) AS failed_judged,"
                " MAX(at) AS last_at FROM outcomes GROUP BY capability, model_id"
                " ORDER BY capability, model_id").fetchall()
            producers = self._conn.execute(
                "SELECT capability, model_id, COUNT(*) AS n, MAX(ts) AS last_ts FROM producers"
                " GROUP BY capability, model_id ORDER BY capability, model_id").fetchall()
            workers = self._conn.execute(
                "SELECT worker, COUNT(*) AS n FROM producers GROUP BY worker ORDER BY n DESC").fetchall()
            n_out = int(self._conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])
            n_prod = int(self._conn.execute("SELECT COUNT(*) FROM producers").fetchone()[0])
        return {
            "ledger_path": self.db_path,
            "outcomes": n_out,
            "producers": n_prod,
            "by_model": [dict(r) for r in outcomes],
            "producers_by_model": [dict(r) for r in producers],
            "workers": [{"worker": r["worker"], "n": r["n"]} for r in workers],
        }


# --------------------------------------------------------------------------- #
# policy + decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    min_samples: int = 3            # ledger evidence below this is advisory only
    reliability_floor: float = 0.34  # ok_rate/pass_rate below this (with samples) -> rejected
    spread_margin: float = 0.15     # candidates may go to models within this score of the best
    spread_candidates: bool = True
    explore_every: int = 12         # every Nth call at a capability, try the runner-up (0 = never)
    explore_margin: float = 0.25    # ...only if it is within this score of the best
    # score weights (sum to 1.0 before the profile tilt)
    w_matrix_quality: float = 0.35
    w_matrix_ok: float = 0.15
    w_ledger_pass: float = 0.30
    w_ledger_ok: float = 0.10
    w_speed: float = 0.10
    primary_bonus: float = 0.05


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    model_id: str
    selected: bool
    score: float
    reasons: tuple[str, ...]              # why it scored what it scored
    rejected_at: str | None = None        # step name when rejected
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "selected": self.selected, "score": round(self.score, 4),
                "reasons": list(self.reasons), "rejected_at": self.rejected_at,
                "evidence": dict(self.evidence)}


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    capability: str
    operation: str
    model_id: str | None
    rationale: str
    ranked: tuple[CandidateVerdict, ...]     # eligible, best first
    rejected: tuple[CandidateVerdict, ...]
    steps: tuple[str, ...]                   # the ordered log
    candidate_index: int = 0
    spread: bool = False
    gap: bool = False
    explored: bool = False

    @property
    def score(self) -> float | None:
        for v in self.ranked:
            if v.selected:
                return v.score
        return None

    @property
    def fallback(self) -> str | None:
        for v in self.ranked:
            if v.model_id != self.model_id:
                return v.model_id
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"capability": self.capability, "operation": self.operation,
                "model_id": self.model_id, "fallback": self.fallback, "rationale": self.rationale,
                "ranked": [v.to_dict() for v in self.ranked],
                "rejected": [v.to_dict() for v in self.rejected],
                "steps": list(self.steps), "candidate_index": self.candidate_index,
                "spread": self.spread, "gap": self.gap, "explored": self.explored,
                "score": self.score}


# capability -> routing-matrix operation when the node does not name one
_OPERATION_BY_CAPABILITY = {
    "text.chat": "text.chat",
    "image.generate": "image.generate",
    "image.understand": "image.judge",
    "audio.tts": "audio.tts",
    "audio.transcribe.word_timestamps": "audio.transcribe",
    "video.generate.i2v": "video.generate",
}


def operation_for(capability: str, node: PlanNode | None = None, goal: GoalSpec | None = None) -> str:
    if node is not None:
        op = node.params.get("operation") if hasattr(node.params, "get") else None
        if op:
            return str(op)
    return _OPERATION_BY_CAPABILITY.get(capability, capability)


# --------------------------------------------------------------------------- #
# the selector
# --------------------------------------------------------------------------- #


def _norm_latency(latency_s: float | None, scale_s: float = 60.0) -> float:
    """0 (instant) .. 1 (>= scale)."""
    if latency_s is None:
        return 0.5
    return max(0.0, min(1.0, float(latency_s) / scale_s))


def select(capability: str, *,
           view: Any = None,
           goal: GoalSpec | None = None,
           node: PlanNode | None = None,
           matrix: Any = None,
           ledger: ReliabilityLedger | None = None,
           policy: SelectionPolicy | None = None,
           model_health: Callable[[str], bool | None] | None = None,
           model_vram_gib: Callable[[str], float | None] | None = None,
           exclude: Iterable[str] = (),
           candidate_index: int = 0,
           candidates: int = 1) -> SelectionDecision:
    pol = policy or SelectionPolicy()
    operation = operation_for(capability, node, goal)
    steps: list[str] = []
    rejected: list[CandidateVerdict] = []

    # 1. compatibility
    if view is None:
        steps.append("1.compatibility: capability unknown to the catalog -> CAPABILITY_GAP")
        return SelectionDecision(capability, operation, None, "capability_gap", (), (), tuple(steps), gap=True)
    elig = getattr(view, "eligibility", None)
    if elig is not None and not elig.eligible:
        why = "; ".join(getattr(elig, "reasons", ()) or ()) or "ineligible"
        steps.append(f"1.compatibility: ineligible on this fleet ({why}) -> CAPABILITY_GAP")
        return SelectionDecision(capability, operation, None, "capability_gap", (), (), tuple(steps), gap=True)
    steps.append("1.compatibility: eligible")
    # 2. authority
    steps.append("2.authority: delegated to router.resolve_route typed gate")
    # 3. registration
    pool = [m for m in (getattr(view, "model_ids", ()) or ())]
    excluded = set(exclude)
    for m in list(pool):
        if m in excluded:
            pool.remove(m)
            rejected.append(CandidateVerdict(m, False, 0.0, ("excluded by caller (failed earlier attempt)",), "3.registration"))
    steps.append(f"3.registration: {len(pool)} eligible model(s)" + (f", {len(excluded)} excluded" if excluded else ""))
    if not pool:
        steps.append("-> CAPABILITY_GAP: no eligible model left")
        return SelectionDecision(capability, operation, None, "capability_gap", (), tuple(rejected), tuple(steps), gap=True)

    # 4. health
    if model_health is not None:
        for m in list(pool):
            h = model_health(m)
            if h is False:
                pool.remove(m)
                rejected.append(CandidateVerdict(m, False, 0.0, ("health probe failed",), "4.health"))
    steps.append(f"4.health: {len(pool)} healthy")

    # 5. resources
    cap_vram = None
    if node is not None and node.resources.vram_gib is not None:
        cap_vram = float(node.resources.vram_gib)
    if goal is not None and goal.budget is not None:
        b = getattr(goal.budget, "max_vram_gib", None) or getattr(goal.budget, "max_vram_gb", None)
        if b is not None:
            cap_vram = float(b) if cap_vram is None else min(cap_vram, float(b))
    if model_vram_gib is not None and cap_vram is not None:
        for m in list(pool):
            need = model_vram_gib(m)
            if need is not None and need > cap_vram:
                pool.remove(m)
                rejected.append(CandidateVerdict(m, False, 0.0, (f"needs {need} GiB > cap {cap_vram} GiB",),
                                                 "5.resources", {"vram_gib": need, "cap_gib": cap_vram}))
    steps.append(f"5.resources: cap {cap_vram} GiB; {len(pool)} fit")
    if not pool:
        return SelectionDecision(capability, operation, None, "capability_gap", (), tuple(rejected), tuple(steps), gap=True)

    # 6. quality profile tilt
    profile = goal.quality if goal is not None else QualityProfile.BALANCED
    wq, wmo, wlp, wlo, ws = (pol.w_matrix_quality, pol.w_matrix_ok, pol.w_ledger_pass,
                             pol.w_ledger_ok, pol.w_speed)
    if profile is QualityProfile.PREVIEW:
        wq, ws = wq * 0.25, ws * 5.0
    elif profile is QualityProfile.BEST:
        wq, wlp, ws = wq * 1.4, wlp * 1.2, ws * 0.3
    steps.append(f"6.quality: profile={profile.value}")

    # 7. latency budget + 9. matrix evidence (read once)
    entry = None
    if matrix is not None:
        try:
            entry = matrix.entry(operation)
        except Exception:  # noqa: BLE001
            entry = None
    mcand: dict[str, Any] = {}
    if entry is not None:
        for c in getattr(entry, "candidates", ()) or ():
            mcand[c.model] = c
    max_s = goal.budget.max_seconds if (goal is not None and goal.budget is not None) else None
    if max_s is not None:
        for m in list(pool):
            c = mcand.get(m)
            lat = getattr(c, "latency_s", None) if c is not None else None
            if lat is not None and lat > float(max_s):
                pool.remove(m)
                rejected.append(CandidateVerdict(m, False, 0.0, (f"measured latency {lat:.1f}s > budget {max_s}s",),
                                                 "7.latency", {"latency_s": lat}))
    steps.append(f"7.latency: budget {max_s}s; {len(pool)} within")
    if not pool:
        return SelectionDecision(capability, operation, None, "capability_gap", (), tuple(rejected), tuple(steps), gap=True)

    # 8. reliability
    lstats: dict[str, ModelStats] = ledger.stats_for(capability, pool) if ledger is not None else {}
    for m in list(pool):
        s = lstats.get(m)
        if s is not None and s.n >= pol.min_samples and (s.ok_rate < pol.reliability_floor
                                                          or s.pass_rate < pol.reliability_floor):
            pool.remove(m)
            rejected.append(CandidateVerdict(
                m, False, 0.0,
                (f"measured reliability below floor: ok={s.ok_rate:.2f} pass={s.pass_rate:.2f} over {s.n}",),
                "8.reliability", s.to_dict()))
    steps.append(f"8.reliability: {len(pool)} above floor ({sum(1 for s in lstats.values() if s.n)} with evidence)")
    if not pool:
        # every model measured bad: do not pretend — surface the gap with evidence
        return SelectionDecision(capability, operation, None, "all_models_below_reliability_floor",
                                 (), tuple(rejected), tuple(steps), gap=True)

    # score
    verdicts: list[CandidateVerdict] = []
    primary = getattr(entry, "primary", None) if entry is not None else None
    for m in pool:
        reasons: list[str] = []
        ev: dict[str, Any] = {}
        c = mcand.get(m)
        mq = float(getattr(c, "quality", 0.0) or 0.0) if c is not None else 0.5
        mok = float(getattr(c, "ok_rate", 0.0) or 0.0) if c is not None else 0.5
        lat = getattr(c, "latency_s", None) if c is not None else None
        if c is not None:
            reasons.append(f"matrix: quality={mq:.2f} ok={mok:.2f} latency={lat}")
            ev["matrix"] = {"quality": mq, "ok_rate": mok, "latency_s": lat}
        else:
            reasons.append("matrix: not benchmarked for this operation (neutral prior)")
        s = lstats.get(m)
        if s is not None and s.n:
            lp, lo = s.pass_rate, s.ok_rate
            reasons.append(f"ledger: pass={lp:.2f} ok={lo:.2f} n={s.n}"
                           + (f" top_repair={s.repair_codes[0][0]}" if s.repair_codes else ""))
            ev["ledger"] = s.to_dict()
            if s.n < pol.min_samples:
                # thin evidence: shrink toward neutral
                k = s.n / pol.min_samples
                lp, lo = 0.5 + (lp - 0.5) * k, 0.5 + (lo - 0.5) * k
            if s.mean_latency_s is not None:
                lat = s.mean_latency_s if lat is None else (lat + s.mean_latency_s) / 2
        else:
            lp = lo = 0.5
            reasons.append("ledger: no outcomes yet (neutral prior)")
        score = wq * mq + wmo * mok + wlp * lp + wlo * lo + ws * (1.0 - _norm_latency(lat))
        if primary == m:
            score += pol.primary_bonus
            reasons.append("matrix primary (+bonus)")
        verdicts.append(CandidateVerdict(m, False, score, tuple(reasons), None, ev))
    verdicts.sort(key=lambda v: (-v.score, v.model_id))
    steps.append("9.recommendation: " + (f"matrix primary={primary}" if primary else "no matrix entry")
                 + f"; ranked {[v.model_id for v in verdicts]}")

    # candidate spread
    chosen = verdicts[0]
    spread = False
    if pol.spread_candidates and candidates > 1 and len(verdicts) > 1:
        best = verdicts[0].score
        near = [v for v in verdicts if best - v.score <= pol.spread_margin]
        if len(near) > 1:
            chosen = near[candidate_index % len(near)]
            spread = chosen.model_id != verdicts[0].model_id
    explored = False
    if (not spread and pol.explore_every > 0 and ledger is not None and len(verdicts) > 1
            and candidate_index == 0):
        total = sum(s.n for s in lstats.values())
        if total and total % pol.explore_every == 0:
            runner = verdicts[1]
            stale = ledger.calls_since_model_tried(capability, runner.model_id)
            if verdicts[0].score - runner.score <= pol.explore_margin and (stale is None or stale >= pol.explore_every):
                chosen, explored = runner, True
    ranked = tuple(CandidateVerdict(v.model_id, v.model_id == chosen.model_id, v.score, v.reasons,
                                    None, v.evidence) for v in verdicts)
    rationale = ("spread:candidate %d -> %s" % (candidate_index, chosen.model_id)) if spread \
        else ("explore:runner-up %s (keeps evidence fresh)" % chosen.model_id if explored
              else ("evidence-ranked" if (entry is not None or lstats) else "no evidence: first eligible"))
    steps.append(f"-> {chosen.model_id} ({rationale}); fallback={ranked[1].model_id if len(ranked) > 1 else None}")
    return SelectionDecision(capability, operation, chosen.model_id, rationale, ranked,
                             tuple(rejected), tuple(steps), candidate_index, spread, explored=explored)


class Selector:
    """Bound selector: live catalog + latest matrix + ledger, callable per node.
    Every seam is overridable so tests and offline tools can drive it."""

    def __init__(self, *, ledger: ReliabilityLedger | None = None,
                 policy: SelectionPolicy | None = None,
                 get_view: Callable[[str], Any] | None = None,
                 get_matrix: Callable[[], Any] | None = None,
                 model_health: Callable[[str], bool | None] | None = None,
                 model_vram_gib: Callable[[str], float | None] | None = None) -> None:
        self.ledger = ledger
        self.policy = policy or SelectionPolicy()
        self._get_view = get_view
        self._get_matrix = get_matrix
        self.model_health = model_health
        self.model_vram_gib = model_vram_gib
        self._matrix_cache: tuple[bool, Any] = (False, None)

    def view(self, capability: str) -> Any:
        if self._get_view is not None:
            return self._get_view(capability)
        from hugpy_oracle import catalog
        return catalog.get_capability(capability)

    def matrix(self) -> Any:
        if self._matrix_cache[0]:
            return self._matrix_cache[1]
        m = None
        try:
            if self._get_matrix is not None:
                m = self._get_matrix()
            else:
                from hugpy_oracle import routing_matrix
                m, _reason = routing_matrix.load_latest_matrix()
        except Exception:  # noqa: BLE001 — a missing matrix is a normal state
            m = None
        self._matrix_cache = (True, m)
        return m

    def invalidate(self) -> None:
        self._matrix_cache = (False, None)

    def rebalance(self, policy: SelectionPolicy, reason: str) -> None:
        """Adopt a new policy (bounded changes come from the steward) and log why."""
        self.policy = policy
        self.rebalance_log = getattr(self, "rebalance_log", []) + [(_utc_now(), reason, policy)]

    def decide(self, capability: str, *, goal: GoalSpec | None = None, node: PlanNode | None = None,
               exclude: Iterable[str] = (), candidate_index: int = 0, candidates: int = 1
               ) -> SelectionDecision:
        return select(capability, view=self.view(capability), goal=goal, node=node,
                      matrix=self.matrix(), ledger=self.ledger, policy=self.policy,
                      model_health=self.model_health, model_vram_gib=self.model_vram_gib,
                      exclude=exclude, candidate_index=candidate_index, candidates=candidates)

    # -- DagRuntime seams ---------------------------------------------------- #

    def for_node(self, node: PlanNode, ctx: Any, inputs: Mapping[str, Any]) -> SelectionDecision | None:
        if not node.capability:
            return None
        exclude = tuple(getattr(ctx, "exclude_models", ()) or ())
        return self.decide(node.capability, node=node, exclude=exclude,
                           candidate_index=getattr(ctx, "candidate", 0),
                           candidates=getattr(ctx, "candidates", 1))

    def record_outcome(self, node: PlanNode, ctx: Any, *, ok: bool, hard_pass: bool | None,
                       repair_code: Any, latency_s: float | None) -> None:
        if self.ledger is None or not node.capability:
            return
        model_id = getattr(ctx, "model_id", None)
        if not model_id:
            return
        sel = getattr(ctx, "selection", None) or {}
        self.ledger.record(node.capability, model_id, ok=ok, hard_pass=hard_pass,
                           repair_code=repair_code, latency_s=latency_s,
                           operation=operation_for(node.capability, node),
                           run_id=getattr(ctx, "run_id", None), node_id=getattr(ctx, "node_id", None),
                           attempt=getattr(ctx, "attempt", None),
                           score=sel.get("score"), explored=bool(sel.get("explored")))


# --------------------------------------------------------------------------- #
# process-level binding for the live seams (performance / runtime)
# --------------------------------------------------------------------------- #

LEDGER_PATH_ENV = "ORACLE_LEDGER_PATH"
LEDGER_REMOTE_ENV = "ORACLE_LEDGER_REMOTE"   # central oracle base url; write-through, best-effort
LEDGER_REMOTE_TIMEOUT_S = 2.0
SELECTION_DISABLE_ENV = "ORACLE_SELECTION_DISABLE"
_PROCESS_SELECTOR: Selector | None = None
_PROCESS_LOCK = threading.Lock()


def default_ledger_path() -> str:
    """``ORACLE_LEDGER_PATH`` else ``<oracle home>/reliability.sqlite``
    (see ``hugpy_oracle.config`` for the root resolution)."""
    from hugpy_oracle.config import ledger_path
    return ledger_path()


def process_selector() -> Selector | None:
    """One selector per process, bound to the live catalog, the newest
    registry-verified matrix, and the reliability ledger. None when disabled
    (``ORACLE_SELECTION_DISABLE=1``) or the ledger cannot be opened."""
    global _PROCESS_SELECTOR
    if os.environ.get(SELECTION_DISABLE_ENV, "").strip() in ("1", "true", "yes"):
        return None
    with _PROCESS_LOCK:
        if _PROCESS_SELECTOR is None:
            try:
                _PROCESS_SELECTOR = Selector(ledger=ReliabilityLedger(default_ledger_path()))
            except Exception:  # noqa: BLE001 — selection must never take the build down
                return None
        return _PROCESS_SELECTOR


def requested_model_for(goal: GoalSpec | None, capability: str, *,
                        candidate_index: int = 0, candidates: int = 1,
                        exclude: Iterable[str] = ()) -> tuple[str | None, dict[str, Any] | None]:
    """(model_id, decision_dict) for a live seam call; (None, None) means
    "no opinion — let the router default", never an exception. A model
    PINNED by an outer decision (``pinned``) wins: the DAG runtime decided
    per candidate already, and the seam must execute THAT decision so the
    artifact is attributed to the model that actually produced it."""
    pin = _pinned_for(capability)
    if pin is not None:
        return pin[0], pin[1]
    sel = process_selector()
    if sel is None:
        return None, None
    try:
        d = sel.decide(capability, goal=goal, candidate_index=candidate_index, candidates=candidates,
                       exclude=exclude)
    except Exception:  # noqa: BLE001
        return None, None
    if d.gap or not d.model_id:
        return None, d.to_dict()
    return d.model_id, d.to_dict()


# --------------------------------------------------------------------------- #
# pinning: an outer decision (DagRuntime per-candidate selection) binds the
# model a seam must use for the duration of one call, per thread
# --------------------------------------------------------------------------- #

import contextlib as _contextlib

_PIN = threading.local()


def _pinned_for(capability: str) -> tuple[str, dict[str, Any] | None] | None:
    stack = getattr(_PIN, "stack", None)
    if not stack:
        return None
    for cap, model, decision in reversed(stack):
        if cap == capability:
            return model, decision
    return None


@_contextlib.contextmanager
def pinned(capability: str, model_id: str | None, decision: Mapping[str, Any] | None = None):
    """Within the block, ``requested_model_for(capability)`` returns
    ``model_id`` (no-op when ``model_id`` is None)."""
    if not model_id:
        yield
        return
    stack = getattr(_PIN, "stack", None)
    if stack is None:
        stack = _PIN.stack = []
    stack.append((capability, model_id, dict(decision) if decision else None))
    try:
        yield
    finally:
        stack.pop()


def note_execution(capability: str, model_id: str | None, *, ok: bool, latency_s: float | None,
                   failure: str | None = None) -> None:
    """Ledger hook for ``runtime.execute_route``: executed-or-not + latency."""
    sel = process_selector()
    if sel is None or sel.ledger is None or not model_id:
        return
    try:
        sel.ledger.record(capability, model_id, ok=ok, latency_s=latency_s,
                          repair_code=None if ok else (failure or None))
    except Exception:  # noqa: BLE001
        pass


def note_verdict(capability: str, model_id: str | None, *, hard_pass: bool,
                 repair_code: Any = None) -> None:
    """Ledger hook for judges (scorecard / evaluation): the verdict on what a
    model produced at a capability."""
    sel = process_selector()
    if sel is None or sel.ledger is None or not model_id:
        return
    try:
        sel.ledger.record(capability, model_id, ok=True, hard_pass=hard_pass, repair_code=repair_code)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# producer attribution: artifact ref -> (capability, model_id)
# --------------------------------------------------------------------------- #
# Seams return bare artifact refs; judges see refs. To credit or blame the
# model that PRODUCED an artifact, the producing call registers the ref here
# and the judge's verdict is written back through it.
#
# or-k16: the record of truth is the ``producers`` table in the reliability
# ledger (same sqlite file as outcomes), so a judge in ANOTHER process sees
# what a worker produced. ``_PRODUCERS`` is a bounded read-through cache in
# front of it. With ``ORACLE_LEDGER_REMOTE=<central url>`` set, a worker
# ALSO posts the attribution to central (POST /api/oracle/producers) and a
# local miss asks central (GET) — best-effort, never blocking a build on a
# network failure: log and keep local.

_PRODUCERS: dict[str, tuple[str, str]] = {}
_PRODUCERS_MAX = 4096


def _cache_put(ref: str, capability: str, model_id: str) -> None:
    with _PROCESS_LOCK:
        if len(_PRODUCERS) >= _PRODUCERS_MAX:
            for k in list(_PRODUCERS)[: _PRODUCERS_MAX // 4]:
                _PRODUCERS.pop(k, None)
        _PRODUCERS[ref] = (capability, model_id)


def _ledger() -> ReliabilityLedger | None:
    sel = process_selector()
    return None if sel is None else sel.ledger


def _worker_name() -> str:
    node = os.uname().nodename if hasattr(os, "uname") else "host"
    return f"{os.getpid()}@{node}"


def ledger_remote_url() -> str | None:
    url = os.environ.get(LEDGER_REMOTE_ENV, "").strip().rstrip("/")
    return url or None


def _remote_request(method: str, path: str, body: Mapping[str, Any] | None = None,
                    *, base: str | None = None) -> dict[str, Any] | None:
    base = base or ledger_remote_url()
    if not base:
        return None
    url = f"{base}{path}"
    data = json.dumps(dict(body)).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=LEDGER_REMOTE_TIMEOUT_S) as resp:  # noqa: S310
            raw = resp.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        _log.warning("oracle ledger remote %s %s failed (kept local): %s", method, url, exc)
        return None
    try:
        out = json.loads(raw.decode("utf-8") or "{}")
    except ValueError:
        _log.warning("oracle ledger remote %s %s returned non-JSON (kept local)", method, url)
        return None
    return out if isinstance(out, dict) else None


def remember_producer(ref: str | None, capability: str, model_id: str | None, *,
                      worker: str | None = None) -> None:
    """Attribute ``ref`` to (capability, model_id): cache, local ledger,
    and (when ``ORACLE_LEDGER_REMOTE`` is set) central — in that order, each
    best-effort. Never raises."""
    if not ref or not model_id:
        return
    ref, model_id = str(ref), str(model_id)
    _cache_put(ref, capability, model_id)
    worker = worker or _worker_name()
    ledger = _ledger()
    if ledger is not None:
        try:
            ledger.remember_producer(ref, capability, model_id, worker=worker)
        except Exception as exc:  # noqa: BLE001 — attribution must never take the build down
            _log.warning("oracle ledger: producer write failed for %s: %s", ref, exc)
    if ledger_remote_url():
        _remote_request("POST", "/api/oracle/producers",
                        {"ref": ref, "capability": capability, "model_id": model_id, "worker": worker})


def producer_of(ref: str | None) -> tuple[str, str] | None:
    """(capability, model_id) that produced ``ref``: cache, then the local
    ledger, then central (remote configured only). Misses fill the cache."""
    if not ref:
        return None
    ref = str(ref)
    hit = _PRODUCERS.get(ref)
    if hit is not None:
        return hit
    ledger = _ledger()
    if ledger is not None:
        try:
            row = ledger.producer(ref)
        except Exception as exc:  # noqa: BLE001
            _log.warning("oracle ledger: producer read failed for %s: %s", ref, exc)
            row = None
        if row is not None:
            _cache_put(ref, row["capability"], row["model_id"])
            return row["capability"], row["model_id"]
    if ledger_remote_url():
        out = _remote_request("GET", f"/api/oracle/producers?ref={urllib.request.quote(ref, safe='')}")
        prod = (out or {}).get("producer") if out else None
        if isinstance(prod, dict) and prod.get("capability") and prod.get("model_id"):
            cap, mid = str(prod["capability"]), str(prod["model_id"])
            _cache_put(ref, cap, mid)
            if ledger is not None:
                try:
                    ledger.remember_producer(ref, cap, mid, worker=prod.get("worker"), ts=prod.get("ts"))
                except Exception:  # noqa: BLE001
                    pass
            return cap, mid
    return None


def producer_stamp(ref: str | None) -> dict[str, str] | None:
    """``{"capability", "model_id"}`` for stamping onto a receipt / manifest
    at production time; None when unattributed."""
    prod = producer_of(ref)
    return None if prod is None else {"capability": prod[0], "model_id": prod[1]}


def note_verdict_for_ref(ref: str | None, *, hard_pass: bool, repair_code: Any = None) -> bool:
    """Write a judge verdict against the model that produced ``ref``.
    Returns True when attribution existed and the ledger took it."""
    prod = producer_of(ref)
    if prod is None:
        return False
    note_verdict(prod[0], prod[1], hard_pass=hard_pass, repair_code=repair_code)
    return True
