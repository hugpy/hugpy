"""Case runner: one bounded `hugpy-agent case` subprocess per NEW case.

The agent run exists for one case and dies with it: brief in, markdown
report out, timeout-bounded, output captured under the case dir. Dedupe
lives in the store (open_or_touch), so a re-detected condition can never
re-spawn. The brief carries the DOCUMENT-ONLY contract; the agent-side
`case` subcommand pins the matching policy profile independently, so the
contract is enforced twice (prompt + policy), not once.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess

from hugpy_ops.sentinel import checks, remedies
from hugpy_ops.sentinel.cases import Anomaly, Case, CaseStore
from hugpy_ops.sentinel.settings import SentinelSettings

CASE_BRIEF_TEMPLATE = """\
# CASE BRIEF — {kind} ({severity}) — case {case_id}

You are a one-shot diagnosis agent spawned by the hugpy sentinel for exactly
this case. When you have finished, deliver the case report via final_answer
and stop. You run in DOCUMENT-ONLY MODE.

## Anomaly
- fingerprint: {fingerprint}
- kind: {kind}
- severity: {severity}
- opened: {opened_iso}

## Evidence (as detected)
```json
{evidence_json}
```

## Diagnosis surfaces you may READ
- http_fetch against central at {central} — read-only GETs only:
  GET {central}/llm/jobs (job states incl. stalled/expired),
  GET {central}/llm/workers (online / version_ok),
  GET {central}/oracle/capabilities (eligibility with reasons).
- The read-only fleet tools: models_list, oracle_capabilities.
- fs_read/fs_glob, and fs_write ONLY inside your case directory (it is your
  workspace) for scratch notes.

## DOCUMENT-ONLY RULE
You must NOT mutate anything: no model unloads, no slot unload/relaunch, no
request cancels, no POSTs that change state, no shell. Your policy profile
denies these tools — a denial is expected, not an obstacle to work around.
If a remedy is warranted you RECOMMEND it in the report; the sentinel and
its operator decide whether it runs. The worker "ae" is the prod host and
is NEVER remediated automatically — for ae, recommend escalation only.

## Required output — your final_answer must be exactly this markdown report
# Case report: {fingerprint}
## Symptom
(what is failing, in operator terms)
## Evidence
(what you observed on the surfaces above, with numbers/states quoted)
## Root-cause hypothesis
(most likely cause; alternatives if genuinely uncertain)
## Recommended remedy
(one concrete action, or "none — escalate")
## Remedy whitelist status
(state whether the recommended remedy is one of the sentinel whitelist:
{whitelist_names} — and whether the target is remediable at all, noting
that "ae" is document+escalate only)
"""


def build_brief(case: Case, settings: SentinelSettings) -> str:
    return CASE_BRIEF_TEMPLATE.format(
        case_id=case.id,
        kind=case.kind,
        severity=case.severity,
        fingerprint=case.fingerprint,
        opened_iso=datetime.datetime.fromtimestamp(
            case.opened_at, datetime.timezone.utc).isoformat(),
        evidence_json=json.dumps(case.evidence, indent=2, sort_keys=True),
        central=settings.central,
        whitelist_names=", ".join(r.name for r in remedies.WHITELIST))


def case_dir_for(case: Case, settings: SentinelSettings) -> str:
    return os.path.join(settings.cases_dir, "%04d-%s" % (case.id, case.kind))


def _extract_report(stdout: str) -> dict | None:
    """The agent prints the report JSON last on stdout (progress is stderr);
    tolerate stray lines by scanning from the last '{'."""
    idx = stdout.rfind("\n{")
    blob = stdout[idx:] if idx >= 0 else stdout
    try:
        return json.loads(blob)
    except (ValueError, TypeError):
        return None


def spawn_agent_for_case(case: Case, store: CaseStore,
                         settings: SentinelSettings,
                         run=subprocess.run) -> Case:
    """Run the one-shot agent for a NEWLY OPENED case; returns the updated
    case. `run` is subprocess.run-shaped, injectable for tests."""
    case_dir = case_dir_for(case, settings)
    os.makedirs(case_dir, exist_ok=True)
    brief_path = os.path.join(case_dir, "brief.md")
    with open(brief_path, "w", encoding="utf-8") as fh:
        fh.write(build_brief(case, settings))

    cmd = [settings.agent_cmd, "case", brief_path, "--case-dir", case_dir,
           "-q", "--max-steps", str(settings.agent_max_steps)]
    cmd += list(settings.agent_extra_args)
    # No env= on purpose: the case agent INHERITS the sentinel's environment,
    # so the unit's HUGPY_BASE — and the k96 brain ladder HUGPY_AGENT_BRAINS
    # (whose last entry is the pilot light the sentinel watches via
    # check_pilot_light) — reach every spawned run without extra plumbing.
    store.set_state(case.id, "agent_running")
    try:
        proc = run(cmd, capture_output=True, text=True,
                   timeout=settings.agent_timeout_s)
    except subprocess.TimeoutExpired:
        store.set_state(case.id, "escalated",
                        note="agent run exceeded %.0fs timeout"
                             % settings.agent_timeout_s)
        _ledger(settings, store.get(case.id))
        return store.get(case.id)
    except OSError as e:              # agent binary missing/unrunnable
        store.set_state(case.id, "escalated",
                        note="agent spawn failed: %s" % e)
        _ledger(settings, store.get(case.id))
        return store.get(case.id)

    with open(os.path.join(case_dir, "agent.stdout"), "w",
              encoding="utf-8") as fh:
        fh.write(proc.stdout or "")
    with open(os.path.join(case_dir, "agent.stderr"), "w",
              encoding="utf-8") as fh:
        fh.write(proc.stderr or "")

    report = _extract_report(proc.stdout or "")
    if proc.returncode == 0 and report and report.get("outcome") == "done":
        report_path = os.path.join(case_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report.get("answer") or "(agent returned no answer)")
        store.attach_agent(case.id, report.get("run_id"), report_path)
        store.set_state(case.id, "documented")
    else:
        outcome = (report or {}).get("outcome", "no-report")
        store.attach_agent(case.id, (report or {}).get("run_id"), None)
        store.set_state(case.id, "escalated",
                        note="agent rc=%s outcome=%s"
                             % (proc.returncode, outcome))
    updated = store.get(case.id)
    _ledger(settings, updated)
    return updated


WEIGHT_MISSING_REPORT = """\
# Case report: {fingerprint}
## Symptom
Declared weight is missing from the store: {registry} registry entry
`{name}` ({reason}) — expected at `{dest}`.
## Evidence
```json
{evidence_json}
```
## Action taken (deterministic remedy — no agent spawned)
{action}
"""


def handle_weight_missing_case(case: Case, store: CaseStore,
                               settings: SentinelSettings,
                               http_post=None) -> Case:
    """k97 pre-agent FAST PATH for kind=weight_missing: the remedy is
    DETERMINISTIC (enqueue the download the evidence already names), so no
    diagnosis agent is spawned. Enqueue (downloads gate, default ON — its
    own gate, separate from HUGPY_SENTINEL_REMEDIES) then document. An
    unresolved source is documented only: the provisioner never guesses a
    hub id, and neither does the sentinel."""
    case_dir = case_dir_for(case, settings)
    os.makedirs(case_dir, exist_ok=True)
    evidence = case.evidence or {}
    anomaly = Anomaly(fingerprint=case.fingerprint, kind=case.kind,
                      severity=case.severity, evidence=evidence)
    remedy = next((r for r in remedies.eligible(anomaly)
                   if r.name == "enqueue_download"), None)
    if remedy is None:
        state = "documented"
        action = ("nothing enqueued — source is UNRESOLVED (no proven hub "
                  "id; the provisioner never guesses one). Resolve the "
                  "source in the registry row, or fetch by hand.")
    else:
        params = {"central": settings.central,
                  "hub_id": evidence.get("hub_id"),
                  # do NOT (re-)register: provisioning restores bytes for an
                  # already-declared row; it must not upsert manifest rows.
                  "register": False}
        for k in ("filename", "include", "name", "framework"):
            if evidence.get(k):
                params[k] = evidence[k]
        try:
            res = remedies.execute(remedy, params, settings,
                                   http_post=http_post)
            state = "remedied"
            action = ("download enqueued on the transfer plane via POST "
                      "%s/llm/repos/download -> job %s (kind=download; the "
                      "hugpy-downloader-dev daemon claims it)."
                      % (settings.central,
                         res.get("id") or res.get("job_id") or "?"))
        except remedies.DownloadsDisabled:
            state = "documented"
            action = ("nothing enqueued — downloads gate is OFF "
                      "(HUGPY_SENTINEL_DOWNLOADS=0). Enqueue by hand or run "
                      "hugpy-provisioner --apply.")
        except Exception as e:  # noqa: BLE001 — a failed POST is a case fact, not a crash
            state = "escalated"
            action = "enqueue attempt FAILED: %s" % e
    report_path = os.path.join(case_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(WEIGHT_MISSING_REPORT.format(
            fingerprint=case.fingerprint,
            registry=evidence.get("registry"), name=evidence.get("name"),
            reason=evidence.get("reason"), dest=evidence.get("dest"),
            evidence_json=json.dumps(evidence, indent=2, sort_keys=True),
            action=action))
    store.attach_agent(case.id, None, report_path)
    store.set_state(case.id, state, note=action.splitlines()[0])
    updated = store.get(case.id)
    _ledger(settings, updated)
    return updated


def _ledger(settings: SentinelSettings, case: Case) -> None:
    """One line per case outcome in cases.md — the human-scan surface."""
    os.makedirs(settings.state_dir, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%MZ")
    rel = (os.path.relpath(case.report_path, settings.state_dir)
           if case.report_path else "—")
    line = ("- %s case %d [%s/%s] `%s` -> %s (report: %s)%s\n"
            % (stamp, case.id, case.kind, case.severity, case.fingerprint,
               case.state, rel, " — %s" % case.note if case.note else ""))
    with open(settings.ledger_path, "a", encoding="utf-8") as fh:
        fh.write(line)


def run_once(settings: SentinelSettings, store: CaseStore | None = None,
             http_get=None, run=subprocess.run, http_post=None,
             wants_fn=None) -> dict:
    """One sentinel pass: detect -> open/touch cases -> handle NEW ones.

    weight_missing cases take the deterministic fast path (enqueue +
    document — no agent); everything else spawns the one-shot diagnosis
    agent. `http_get`, `run`, `http_post` and `wants_fn` are injectable for
    tests.
    """
    own_store = store is None
    store = store or CaseStore(settings.db_path)
    try:
        anomalies = checks.detect(settings, http_get=http_get,
                                  wants_fn=wants_fn)
        opened, touched = [], []
        for anomaly in anomalies:
            case, created = store.open_or_touch(anomaly)
            (opened if created else touched).append(case)
        for case in opened:
            if case.kind == "weight_missing":
                handle_weight_missing_case(case, store, settings,
                                           http_post=http_post)
            else:
                spawn_agent_for_case(case, store, settings, run=run)
        return {
            "anomalies": len(anomalies),
            "opened": [c.id for c in opened],
            "touched": [c.id for c in touched],
        }
    finally:
        if own_store:
            store.close()
