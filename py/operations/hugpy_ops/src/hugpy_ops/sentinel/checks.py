"""Detection: pure functions over central's four read surfaces.

The sentinel is an HTTP CLIENT of central — the fetch layer is three GETs
(injectable for tests) and every rule is a pure function over the parsed
JSON, so central's own state machines (comms/jobs.py, the worker store,
the oracle catalog) remain the single authority on what "stalled",
"offline" and "ineligible" mean. The sentinel only decides which of those
conditions has persisted past its bound and deserves a case.

Shapes (verified against the routes, 2026-08-06):
  GET /llm/jobs?live=0   -> {"jobs": [row...], "counts": {...}}
                            row: id, status (canonical, incl. terminal
                            "expired"), stalled (central-computed, 90s),
                            progressed_at, worker, slot, model_key, stage,
                            message, ...
  GET /llm/workers       -> BARE ARRAY of rows: name, status
                            ("online"|"offline"), version_ok (tri-state:
                            None = central pins nothing), unreachable, ...
  GET /oracle/capabilities -> {"ok", "count", "capabilities": [{name,
                            eligibility: {eligible, reasons}, ...}]}
Scorecards ride POST /oracle/route responses; the sentinel never calls
route (route EXECUTES). It reads a small local history it appends —
capability snapshots each pass, scorecard observations fed via
append_scorecard() / the record-scorecard CLI.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from hugpy_ops.sentinel.cases import Anomaly
from hugpy_ops.sentinel.settings import SentinelSettings

# history.jsonl stays small: prune to the newest half past this size.
_HISTORY_MAX_BYTES = 256 * 1024


# --------------------------------------------------------------------------
# fetch layer (injectable)

def _default_get(url: str, timeout: float = 20.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_surfaces(settings: SentinelSettings, http_get=None) -> dict:
    """The three GETs. live=0 so terminal (expired) rows are visible."""
    get = http_get or _default_get
    return {
        "jobs": get(settings.central + "/llm/jobs?live=0"),
        "workers": get(settings.central + "/llm/workers"),
        "capabilities": get(settings.central + "/oracle/capabilities"),
    }


# --------------------------------------------------------------------------
# local history (capability snapshots + scorecard observations)

def load_history(settings: SentinelSettings) -> list[dict]:
    try:
        with open(settings.history_path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []


def _append_history(settings: SentinelSettings, record: dict) -> None:
    os.makedirs(settings.state_dir, exist_ok=True)
    with open(settings.history_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    try:
        if os.path.getsize(settings.history_path) > _HISTORY_MAX_BYTES:
            with open(settings.history_path, encoding="utf-8") as fh:
                lines = fh.readlines()
            with open(settings.history_path, "w", encoding="utf-8") as fh:
                fh.writelines(lines[len(lines) // 2:])
    except OSError:
        pass


def append_capability_snapshot(settings: SentinelSettings, caps_body: dict,
                               now: float | None = None) -> None:
    snap = {c.get("name"): bool((c.get("eligibility") or {}).get("eligible"))
            for c in (caps_body or {}).get("capabilities", [])}
    _append_history(settings, {"ts": now or time.time(),
                               "kind": "capabilities", "snapshot": snap})


def append_scorecard(settings: SentinelSettings, capability: str, model: str,
                     hard_pass: bool, detail: dict | None = None,
                     now: float | None = None) -> None:
    """Feed one observed /oracle/route scorecard into the streak history."""
    _append_history(settings, {"ts": now or time.time(), "kind": "scorecard",
                               "capability": capability, "model": model,
                               "hard_pass": bool(hard_pass),
                               "detail": detail or {}})


def last_capability_snapshot(history: list[dict]) -> dict | None:
    for rec in reversed(history):
        if rec.get("kind") == "capabilities":
            return rec.get("snapshot") or {}
    return None


# --------------------------------------------------------------------------
# rules — each pure over parsed data

def check_jobs(jobs_body: dict, now: float,
               settings: SentinelSettings) -> list[Anomaly]:
    out = []
    for row in (jobs_body or {}).get("jobs", []):
        jid = row.get("id")
        evidence = {k: row.get(k) for k in
                    ("id", "status", "worker", "slot", "model_key", "kind",
                     "stage", "message", "progressed_at", "stalled",
                     "attempt", "max_attempts")}
        evidence["request_id"] = jid
        # Stalled beyond bound: central labels stalled at 90s of silence
        # (HUGPY_JOB_STALL_SECONDS); the sentinel cases it only once the
        # silence has ALSO outlived its own grace — a transient hiccup that
        # recovers inside the grace never becomes a case.
        if row.get("stalled") and row.get("status") in ("processing",
                                                        "streaming"):
            try:
                idle = now - float(row.get("progressed_at") or now)
            except (TypeError, ValueError):
                idle = 0.0
            if idle > settings.stalled_grace_s:
                evidence["idle_s"] = round(idle, 1)
                out.append(Anomaly(
                    fingerprint="job_stalled:%s" % jid,
                    kind="job_stalled", severity="critical",
                    evidence=evidence))
        # Expired: central retired the job WITHOUT it ever reaching a real
        # terminal outcome — either never dispatched or wedged mid-run.
        # The retirement is central's; the case is about WHY it wedged.
        elif row.get("status") == "expired":
            out.append(Anomaly(
                fingerprint="job_expired:%s" % jid,
                kind="job_expired", severity="warn", evidence=evidence))
    return out


def check_workers(workers_body: list, local_versions=None) -> list[Anomaly]:
    """Offline/unreachable workers (critical) and version skew (warn).

    A skew case carries the ecosystem distribution versions installed where
    the sentinel runs (``hugpy-fleet`` = worker code, ``hugpy-server`` =
    central code) as ``local_distributions`` evidence; ``local_versions`` is
    injectable, else :func:`hugpy_ops.versions.distribution_versions` is
    consulted lazily and only when a skew is actually found."""
    out = []
    for row in (workers_body or []):
        name = row.get("name") or row.get("id")
        evidence = {k: row.get(k) for k in
                    ("name", "id", "status", "url", "pkg_version",
                     "required_pkg_version", "version_ok", "unreachable",
                     "unreachable_reason")}
        evidence["worker"] = name
        if row.get("status") != "online" or row.get("unreachable"):
            out.append(Anomaly(
                fingerprint="worker_offline:%s" % name,
                kind="worker_offline", severity="critical",
                evidence=evidence))
        # version_ok is TRI-STATE: None means central pins nothing, which
        # is not an anomaly. Only a literal False is skew.
        elif row.get("version_ok") is False:
            if local_versions is None:
                from hugpy_ops.versions import distribution_versions
                local_versions = distribution_versions()
            evidence["local_distributions"] = dict(local_versions)
            out.append(Anomaly(
                fingerprint="worker_version:%s" % name,
                kind="worker_version", severity="warn", evidence=evidence))
    return out


def _warm_keys(workers_body) -> list:
    """model_keys WARM on any worker row: allocations[] entries carrying a
    model_key whose ``healthy`` is not False ('ram' residents carry no healthy
    key and count — they answer without a load). Defensive over the same
    /llm/workers surface check_workers reads; malformed rows are skipped."""
    keys = []
    rows = workers_body if isinstance(workers_body, list) else \
        workers_body.get("workers") if isinstance(workers_body, dict) else []
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        for alloc in (row.get("allocations") or []):
            if not isinstance(alloc, dict) or not alloc.get("model_key"):
                continue
            if alloc.get("healthy") is False:
                continue
            keys.append(str(alloc["model_key"]))
    return keys


def _brain_key_match(a: str, b: str) -> bool:
    """Catalog 'Org~Name' and bare-name forms name the same model — compare
    the bare tails (mirrors hugpy_agent.gateway.brain_matches_key)."""
    a, b = (a or "").strip(), (b or "").strip()
    return bool(a and b) and (a == b or a.split("~")[-1] == b.split("~")[-1])


def check_pilot_light(workers_body, settings: SentinelSettings) -> list[Anomaly]:
    """k96: the agent brain ladder's PILOT LIGHT (its last entry — the brain a
    case run falls back to when nothing else is warm) should always be warm on
    some worker. Not warm = every future case run pays a cold load, and a
    fleet-wide wedge could leave the sentinel brainless exactly when it is
    needed — worth a warn-severity case while the fleet is otherwise fine.
    Off (no anomaly ever) when no pilot light is configured."""
    pilot = (settings.pilot_light or "").strip()
    if not pilot:
        return []
    warm = _warm_keys(workers_body)
    if any(_brain_key_match(pilot, k) for k in warm):
        return []
    return [Anomaly(
        fingerprint="pilot_light_not_resident:%s" % pilot,
        kind="pilot_light_not_resident", severity="warn",
        evidence={"model_key": pilot, "warm_model_keys": sorted(set(warm)),
                  "hint": ("pin it: POST %s/llm/workers/<worker_id>/"
                           "boot-prewarm {\"model_key\": %r} — the keep-warm "
                           "star reseats it every reconcile beat"
                           % (settings.central, pilot))})]


def check_capabilities(caps_body: dict,
                       previous_snapshot: dict | None) -> list[Anomaly]:
    """Eligible -> ineligible TRANSITION only: a capability that was never
    eligible (no model registered yet) is a catalog fact, not an incident."""
    out = []
    if not previous_snapshot:
        return out
    for cap in (caps_body or {}).get("capabilities", []):
        name = cap.get("name")
        elig = (cap.get("eligibility") or {})
        if previous_snapshot.get(name) is True and not elig.get("eligible"):
            out.append(Anomaly(
                fingerprint="capability_lost:%s" % name,
                kind="capability_lost", severity="critical",
                evidence={"capability": name,
                          "reasons": list(elig.get("reasons") or []),
                          "model_ids": list(cap.get("model_ids") or [])}))
    return out


def check_missing_weights(wants: list[dict]) -> list[Anomaly]:
    """k97: one info-severity anomaly per declared-but-missing weight
    (provisioner.wants() fingerprints, evidence = the Want itself). Severity
    info — nothing is failing, a declared capability is starved; the remedy
    (enqueue_download, downloads gate) is deterministic, so the runner
    handles these cases WITHOUT an agent spawn."""
    out = []
    for w in (wants or []):
        fp = w.get("fingerprint") or "weight_missing:%s:%s" % (
            w.get("registry"), w.get("name"))
        out.append(Anomaly(fingerprint=fp, kind="weight_missing",
                           severity="info", evidence=dict(w)))
    return out


def default_weight_wants() -> list[dict]:
    """The live fleet scan (provisioner.wants), as evidence dicts. Local
    import + broad guard: the sentinel must stay runnable on a box where the
    registries can't load — a failed scan means no weight anomalies this
    pass, never a dead sentinel."""
    try:
        from hugpy_ops import provisioner
        return [w.to_evidence() for w in provisioner.wants()]
    except Exception:  # noqa: BLE001 — scan failure must not blind the HTTP checks
        return []


def check_scorecards(history: list[dict],
                     settings: SentinelSettings) -> list[Anomaly]:
    """N CONSECUTIVE hard_pass=False for one (capability, model)."""
    streaks: dict[tuple, list[dict]] = {}
    for rec in history:
        if rec.get("kind") != "scorecard":
            continue
        key = (rec.get("capability"), rec.get("model"))
        if rec.get("hard_pass"):
            streaks[key] = []
        else:
            streaks.setdefault(key, []).append(rec)
    out = []
    for (capability, model), fails in streaks.items():
        if len(fails) >= settings.hard_fail_streak:
            last = fails[-1]
            evidence = {"capability": capability, "model": model,
                        "consecutive_hard_fails": len(fails),
                        "last_detail": last.get("detail") or {}}
            worker = (last.get("detail") or {}).get("worker")
            if worker:
                evidence["worker"] = worker
            out.append(Anomaly(
                fingerprint="scorecard_hard_fail:%s:%s" % (capability, model),
                kind="scorecard_hard_fail", severity="critical",
                evidence=evidence))
    return out


# --------------------------------------------------------------------------
# one detection pass

def detect(settings: SentinelSettings, http_get=None,
           now: float | None = None, wants_fn=None) -> list[Anomaly]:
    """`wants_fn` (injectable like http_get) supplies the k97 weight scan;
    None means the real provisioner scan (default_weight_wants)."""
    now = time.time() if now is None else now
    surfaces = fetch_surfaces(settings, http_get=http_get)
    history = load_history(settings)
    anomalies = []
    anomalies += check_jobs(surfaces["jobs"], now, settings)
    anomalies += check_workers(surfaces["workers"])
    anomalies += check_pilot_light(surfaces["workers"], settings)
    anomalies += check_capabilities(surfaces["capabilities"],
                                    last_capability_snapshot(history))
    anomalies += check_scorecards(history, settings)
    anomalies += check_missing_weights(
        (default_weight_wants if wants_fn is None else wants_fn)())
    # Snapshot AFTER computing the transition, so this pass's view becomes
    # the next pass's "previous".
    append_capability_snapshot(settings, surfaces["capabilities"], now=now)
    return anomalies
