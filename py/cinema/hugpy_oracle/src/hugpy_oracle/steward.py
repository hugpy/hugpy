"""Steward (k113c): the system monitors itself, checks itself, balances itself.

A selector that is never audited becomes a default with extra steps. The
steward reads the evidence the runtime produces — the reliability ledger,
the run journal, the routing matrix — and answers, on a schedule or on
demand:

* **Is the evidence predictive?** Calibration per capability: do higher
  selection scores actually pass the judge more often? (Spearman-style rank
  agreement over the recent window.) If the matrix is predictive and the
  ledger is not (or vice versa), weights shift toward the better predictor —
  bounded, logged, reversible.
* **Is anything failing in a streak?** ``k`` consecutive failures for one
  (capability, model) is an alarm with the dominant repair code.
* **Is anything starving?** An eligible model never or rarely tried at a
  capability is an evidence deficit; exploration is the remedy (the selector
  already explores every Nth call; the steward raises the rate when the
  deficit grows and lowers it when evidence is fresh).
* **Are gaps rising?** Fraction of node failures that are CAPABILITY_GAP /
  selection gaps in the journal — a fleet problem, not a model problem.
* **Is the matrix stale?** Older than ``matrix_max_age_days`` or from a
  different registry version.
* **Is the cache honest?** Cache-hit rate per capability (a 100% hit rate on
  a generative step means nothing new is being produced).

The output is a ``HealthReport``: findings with severity, evidence, and the
action taken or recommended. **It is never silent** — a clean check is a
report that says so, with the numbers.

Stdlib only. Every threshold is a ``StewardPolicy`` field.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from hugpy_oracle.selection import ReliabilityLedger, SelectionPolicy, Selector

__all__ = ["Finding", "HealthReport", "Steward", "StewardPolicy", "rank_agreement"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class StewardPolicy:
    window: int = 200                  # ledger rows per capability considered
    streak_k: int = 4                  # consecutive failures that raise an alarm
    min_calibration_n: int = 12        # judged rows needed before calibration is trusted
    calibration_floor: float = 0.15    # rank agreement below this = "not predictive"
    starvation_calls: int = 40         # calls at a capability without trying a model = starving
    gap_rate_warn: float = 0.2
    matrix_max_age_days: float = 14.0
    cache_hit_warn: float = 0.95
    weight_step: float = 0.05          # max weight shift per check
    weight_floor: float = 0.05
    weight_ceiling: float = 0.5
    explore_every_min: int = 6
    explore_every_max: int = 30


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str               # calibration | streak | starvation | gap_rate | matrix_stale | cache | ok
    severity: str           # info | warn | alarm
    capability: str | None
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    action: str = ""        # what the steward DID (or recommends)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "severity": self.severity, "capability": self.capability,
                "message": self.message, "evidence": dict(self.evidence), "action": self.action}


@dataclass(frozen=True, slots=True)
class HealthReport:
    at: str
    findings: tuple[Finding, ...]
    policy_before: SelectionPolicy | None
    policy_after: SelectionPolicy | None
    summary: str

    @property
    def ok(self) -> bool:
        return not any(f.severity == "alarm" for f in self.findings)

    @property
    def alarms(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "alarm")

    def to_dict(self) -> dict[str, Any]:
        return {"at": self.at, "ok": self.ok, "summary": self.summary,
                "findings": [f.to_dict() for f in self.findings],
                "policy_changed": self.policy_before != self.policy_after,
                "policy_after": (self.policy_after.__dict__ if hasattr(self.policy_after, "__dict__")
                                 else {f: getattr(self.policy_after, f) for f in self.policy_after.__slots__})
                if self.policy_after is not None else None}


def rank_agreement(pairs: list[tuple[float, float]],
                   weights: list[float] | None = None) -> float | None:
    """Spearman rank correlation between predicted score and outcome (0/1 or a
    metric). None when there is no variance to rank. ``weights`` (k115): one
    per pair, in [0, 1] — the judge panel's agreement rate for that row. A
    verdict two independent judges disagreed on counts half as much as a
    unanimous one; zero-weight rows are dropped. None/omitted = unweighted."""
    n = len(pairs)
    if weights is not None:
        if len(weights) != n:
            raise ValueError("weights must align with pairs")
        kept = [(p, w) for p, w in zip(pairs, weights) if w and w > 0]
        pairs = [p for p, _w in kept]
        weights = [float(w) for _p, w in kept]
        n = len(pairs)
    if n < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    ws = weights if weights is not None else [1.0] * n
    tw = sum(ws)
    mx, my = sum(w * a for w, a in zip(ws, rx)) / tw, sum(w * b for w, b in zip(ws, ry)) / tw
    cov = sum(w * (a - mx) * (b - my) for w, a, b in zip(ws, rx, ry))
    vx = math.sqrt(sum(w * (a - mx) ** 2 for w, a in zip(ws, rx)))
    vy = math.sqrt(sum(w * (b - my) ** 2 for w, b in zip(ws, ry)))
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


class Steward:
    def __init__(self, ledger: ReliabilityLedger, *, selector: Selector | None = None,
                 journal: Any = None, matrix: Any = None,
                 eligible_models: Mapping[str, tuple[str, ...]] | None = None,
                 policy: StewardPolicy | None = None) -> None:
        self.ledger = ledger
        self.selector = selector
        self.journal = journal
        self.matrix = matrix
        self.eligible_models = dict(eligible_models or {})
        self.policy = policy or StewardPolicy()
        self.history: list[HealthReport] = []

    # -- checks ------------------------------------------------------------- #

    def check(self) -> HealthReport:
        pol = self.policy
        findings: list[Finding] = []
        before = self.selector.policy if self.selector is not None else None
        after = before
        caps = self.ledger.capabilities()

        predictive: dict[str, float | None] = {}
        for cap in caps:
            rows = self.ledger.recent(cap, limit=pol.window)
            findings.extend(self._streaks(cap, rows))
            findings.extend(self._starvation(cap, rows))
            r, f = self._calibration(cap, rows)
            predictive[cap] = r
            if f is not None:
                findings.append(f)

        findings.extend(self._gap_rate())
        findings.extend(self._matrix_staleness())
        findings.extend(self._cache_honesty())

        if self.selector is not None:
            after = self._rebalance(before, predictive, findings)

        if not findings:
            total = self.ledger.count()
            findings.append(Finding("ok", "info", None,
                                    f"no anomalies over {total} ledger rows across {len(caps)} capabilities",
                                    {"rows": total, "capabilities": list(caps)}))
        alarms = sum(1 for f in findings if f.severity == "alarm")
        warns = sum(1 for f in findings if f.severity == "warn")
        report = HealthReport(_utc_now(), tuple(findings), before, after,
                              f"{alarms} alarm(s), {warns} warning(s), {len(findings) - alarms - warns} info"
                              + ("; selection policy rebalanced" if before != after else ""))
        self.history.append(report)
        return report

    def _streaks(self, cap: str, rows: list[dict[str, Any]]) -> list[Finding]:
        out: list[Finding] = []
        by_model: dict[str, list[dict[str, Any]]] = {}
        for r in rows:  # newest first
            by_model.setdefault(r["model_id"], []).append(r)
        for model, rs in by_model.items():
            streak = 0
            codes: dict[str, int] = {}
            for r in rs:
                failed = (not r["ok"]) or (r["hard_pass"] == 0)
                if not failed:
                    break
                streak += 1
                if r["repair_code"]:
                    codes[r["repair_code"]] = codes.get(r["repair_code"], 0) + 1
            if streak >= self.policy.streak_k:
                top = max(codes.items(), key=lambda kv: kv[1])[0] if codes else None
                out.append(Finding("streak", "alarm", cap,
                                   f"{model} has failed {streak} consecutive calls at {cap}"
                                   + (f" (dominant: {top})" if top else ""),
                                   {"model_id": model, "streak": streak, "repair_codes": codes},
                                   action="selector reliability floor will reject it; operator should check seating/probe"))
        return out

    def _starvation(self, cap: str, rows: list[dict[str, Any]]) -> list[Finding]:
        out: list[Finding] = []
        eligible = self.eligible_models.get(cap)
        if not eligible or len(rows) < self.policy.starvation_calls:
            return out
        for model in eligible:
            stale = self.ledger.calls_since_model_tried(cap, model)
            if stale is None or stale >= self.policy.starvation_calls:
                out.append(Finding("starvation", "warn", cap,
                                   f"{model} is eligible at {cap} but "
                                   + ("has never been tried" if stale is None else f"untried for {stale} calls"),
                                   {"model_id": model, "calls_since": stale},
                                   action="raise exploration rate"))
        return out

    @staticmethod
    def _row_confidence(row: Mapping[str, Any]) -> float:
        """k115: the judge panel's agreement rate for a ledger row, when the
        ledger carries one (``confidence`` / ``judge_confidence`` column or
        key); 1.0 for rows written before the panel existed — they are not
        retroactively doubted, they simply predate the second opinion."""
        for key in ("confidence", "judge_confidence"):
            v = row.get(key)
            if v is not None:
                try:
                    return max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    continue
        return 1.0

    def _calibration(self, cap: str, rows: list[dict[str, Any]]) -> tuple[float | None, Finding | None]:
        judged = [(float(r["score"]), 1.0 if r["hard_pass"] else 0.0, self._row_confidence(r))
                  for r in rows if r["score"] is not None and r["hard_pass"] is not None and r["ok"]]
        if len(judged) < self.policy.min_calibration_n:
            return None, None
        pairs = [(s, o) for s, o, _c in judged]
        weights = [c for _s, _o, c in judged]
        effective_n = round(sum(weights), 2)
        rho = rank_agreement(pairs, weights)
        if rho is None:
            return None, Finding("calibration", "info", cap,
                                 f"no variance to calibrate at {cap} over {len(judged)} judged calls",
                                 {"n": len(judged), "effective_n": effective_n})
        sev = "warn" if rho < self.policy.calibration_floor else "info"
        return rho, Finding("calibration", sev, cap,
                            f"selection score vs judge pass rank-agreement {rho:+.2f} over {len(judged)} calls at {cap}"
                            f" (confidence-weighted n {effective_n:g})"
                            + ("" if sev == "info" else " — scores are NOT predictive here"),
                            {"rho": rho, "n": len(judged), "effective_n": effective_n},
                            action="" if sev == "info" else "shift weight toward the better predictor; raise exploration")

    def _gap_rate(self) -> list[Finding]:
        if self.journal is None:
            return []
        try:
            runs = self.journal.runs()
        except Exception:  # noqa: BLE001
            return []
        failed = gaps = 0
        for run in runs:
            for rec in self.journal.nodes(run.run_id).values():
                if rec.failure is not None:
                    failed += 1
                    if (rec.failure_class or "") == "capability_gap":
                        gaps += 1
        if failed == 0:
            return []
        rate = gaps / failed
        if rate >= self.policy.gap_rate_warn:
            return [Finding("gap_rate", "alarm", None,
                            f"{gaps}/{failed} node failures are capability/selection gaps ({rate:.0%})",
                            {"gaps": gaps, "failed": failed},
                            action="fleet problem: seat models / fix probes; no amount of routing fixes this")]
        return [Finding("gap_rate", "info", None, f"gap share of failures {rate:.0%}", {"gaps": gaps, "failed": failed})]

    def _matrix_staleness(self) -> list[Finding]:
        m = self.matrix
        if m is None:
            return [Finding("matrix_stale", "warn", None, "no routing matrix loaded: selection runs on ledger + catalog only",
                            action="run the benchmark battery to publish a matrix")]
        gen = getattr(m, "generated_at", None)
        if gen:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(str(gen).replace("Z", "+00:00"))).days
            except Exception:  # noqa: BLE001
                age = None
            if age is not None and age > self.policy.matrix_max_age_days:
                return [Finding("matrix_stale", "warn", None, f"routing matrix is {age} days old",
                                {"generated_at": gen}, action="re-run the battery")]
        return []

    def _cache_honesty(self) -> list[Finding]:
        if self.journal is None:
            return []
        out: list[Finding] = []
        try:
            runs = self.journal.runs()
        except Exception:  # noqa: BLE001
            return out
        per_cap: dict[str, list[int]] = {}
        for run in runs:
            graph = self.journal.graph(run.run_id)
            for nid, rec in self.journal.nodes(run.run_id).items():
                node = graph.node(nid)
                if node is None or not node.capability or rec.state.value != "succeeded":
                    continue
                per_cap.setdefault(node.capability, []).append(1 if rec.cached else 0)
        for cap, hits in per_cap.items():
            if len(hits) >= 10 and sum(hits) / len(hits) >= self.policy.cache_hit_warn and not cap.startswith(("media.", "audio.transcribe")):
                out.append(Finding("cache", "warn", cap,
                                   f"{sum(hits)}/{len(hits)} recent {cap} nodes were cache hits — nothing new is being generated",
                                   {"hits": sum(hits), "n": len(hits)},
                                   action="expected for deterministic steps; for generative steps check cache keys include seed/params"))
        return out

    # -- balance ------------------------------------------------------------ #

    def _rebalance(self, before: SelectionPolicy, predictive: Mapping[str, float | None],
                   findings: list[Finding]) -> SelectionPolicy:
        pol = self.policy
        after = before
        rhos = [r for r in predictive.values() if r is not None]
        starving = any(f.kind == "starvation" for f in findings)
        uncalibrated = any(f.kind == "calibration" and f.severity == "warn" for f in findings)

        def clamp(v: float) -> float:
            return max(pol.weight_floor, min(pol.weight_ceiling, round(v, 4)))

        if rhos:
            mean_rho = sum(rhos) / len(rhos)
            if mean_rho < pol.calibration_floor:
                # scores not predicting outcomes: lean on direct ledger pass-rate, less on matrix quality
                after = replace(after,
                                w_matrix_quality=clamp(after.w_matrix_quality - pol.weight_step),
                                w_ledger_pass=clamp(after.w_ledger_pass + pol.weight_step))
                findings.append(Finding("rebalance", "info", None,
                                        f"mean rank-agreement {mean_rho:+.2f} < floor: shifted {pol.weight_step} from matrix quality to ledger pass-rate",
                                        {"mean_rho": mean_rho}, action="policy updated (bounded, reversible)"))
            elif mean_rho > 0.5 and after.w_matrix_quality < SelectionPolicy().w_matrix_quality:
                after = replace(after,
                                w_matrix_quality=clamp(after.w_matrix_quality + pol.weight_step),
                                w_ledger_pass=clamp(after.w_ledger_pass - pol.weight_step))
                findings.append(Finding("rebalance", "info", None,
                                        f"mean rank-agreement {mean_rho:+.2f}: restoring matrix weight toward default",
                                        {"mean_rho": mean_rho}, action="policy updated"))
        if starving or uncalibrated:
            new_every = max(pol.explore_every_min, after.explore_every - 3)
            if new_every != after.explore_every:
                after = replace(after, explore_every=new_every)
                findings.append(Finding("rebalance", "info", None,
                                        f"exploration raised: every {new_every} calls", action="policy updated"))
        elif after.explore_every < pol.explore_every_max and not rhos:
            pass  # no evidence either way; leave exploration alone
        elif rhos and after.explore_every < SelectionPolicy().explore_every:
            new_every = min(pol.explore_every_max, after.explore_every + 2)
            after = replace(after, explore_every=new_every)
            findings.append(Finding("rebalance", "info", None,
                                    f"evidence fresh and predictive: exploration relaxed to every {new_every} calls",
                                    action="policy updated"))
        if after != before and self.selector is not None:
            self.selector.rebalance(after, "steward check " + _utc_now())
        return after
