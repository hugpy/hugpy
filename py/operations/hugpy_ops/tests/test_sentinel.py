"""Sentinel (k95): detection over the four surfaces, case dedupe, the
document-only agent spawn, and the remedy whitelist gate + ae exclusion.

Run:  python -m pytest tests/test_sentinel.py -q   (in py/operations/hugpy_ops)
"""
import json
import os
import subprocess

import pytest

from hugpy_ops.sentinel import checks, remedies, runner
from hugpy_ops.sentinel.cases import Anomaly, CaseStore
from hugpy_ops.sentinel.settings import SentinelSettings, load_settings

NOW = 1_754_000_000.0


def _settings(tmp_path, **kw) -> SentinelSettings:
    s = SentinelSettings(state_dir=str(tmp_path / "state"))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _http_get_for(payloads):
    """Map url-suffix -> body; asserts only known surfaces are fetched."""
    def get(url, timeout=20.0):
        for suffix, body in payloads.items():
            if url.endswith(suffix):
                return body
        raise AssertionError("unexpected GET %s" % url)
    return get


def _surfaces(jobs=(), workers=(), capabilities=()):
    return {
        "/llm/jobs?live=0": {"jobs": list(jobs),
                             "counts": {"waiting": 0, "active": 0,
                                        "total": len(jobs)}},
        "/llm/workers": list(workers),
        "/oracle/capabilities": {"ok": True, "count": len(capabilities),
                                 "capabilities": list(capabilities)},
    }


# --------------------------------------------------------------------------
# checks


def test_stalled_job_beyond_grace_is_an_anomaly(tmp_path):
    s = _settings(tmp_path, stalled_grace_s=120.0)
    jobs = [{"id": "j1", "status": "processing", "stalled": True,
             "progressed_at": NOW - 300, "worker": "computron", "slot": 2,
             "model_key": "m", "stage": "prefill", "message": "…"}]
    out = checks.check_jobs({"jobs": jobs}, NOW, s)
    assert [a.kind for a in out] == ["job_stalled"]
    a = out[0]
    assert a.fingerprint == "job_stalled:j1"
    assert a.severity == "critical"
    assert a.evidence["worker"] == "computron"
    assert a.evidence["idle_s"] == 300.0
    assert a.evidence["request_id"] == "j1"          # for central_chat_cancel


def test_stalled_inside_grace_is_not_cased(tmp_path):
    s = _settings(tmp_path, stalled_grace_s=120.0)
    jobs = [{"id": "j1", "status": "streaming", "stalled": True,
             "progressed_at": NOW - 100}]
    assert checks.check_jobs({"jobs": jobs}, NOW, s) == []


def test_expired_job_is_an_anomaly(tmp_path):
    s = _settings(tmp_path)
    jobs = [{"id": "j2", "status": "expired", "stalled": False,
             "message": "no forward progress in stage 'load' for 950s"},
            {"id": "j3", "status": "done"}]
    out = checks.check_jobs({"jobs": jobs}, NOW, s)
    assert [(a.kind, a.fingerprint) for a in out] == \
        [("job_expired", "job_expired:j2")]


def test_worker_offline_and_version_skew():
    workers = [
        {"name": "computron", "status": "offline", "version_ok": True},
        {"name": "op", "status": "online", "version_ok": False,
         "pkg_version": "0.1.228", "required_pkg_version": "0.1.229"},
        # tri-state: None = central pins nothing — NOT an anomaly.
        {"name": "localhost", "status": "online", "version_ok": None},
        {"name": "ae", "status": "online", "version_ok": True},
    ]
    out = checks.check_workers(workers)
    assert {(a.kind, a.evidence["worker"]) for a in out} == \
        {("worker_offline", "computron"), ("worker_version", "op")}


def test_unreachable_online_worker_is_offline_anomaly():
    out = checks.check_workers([{"name": "op", "status": "online",
                                 "unreachable": True,
                                 "unreachable_reason": "breaker open"}])
    assert [a.kind for a in out] == ["worker_offline"]


def test_capability_transition_needs_previous_snapshot():
    caps = {"capabilities": [
        {"name": "vision.describe",
         "eligibility": {"eligible": False,
                         "reasons": ["no online worker registered"]}}]}
    # No history yet: a never-eligible capability is a catalog fact.
    assert checks.check_capabilities(caps, None) == []
    assert checks.check_capabilities(caps, {"vision.describe": False}) == []
    out = checks.check_capabilities(caps, {"vision.describe": True})
    assert [a.fingerprint for a in out] == ["capability_lost:vision.describe"]
    assert out[0].evidence["reasons"] == ["no online worker registered"]


def test_scorecard_streak_thresholds(tmp_path):
    s = _settings(tmp_path, hard_fail_streak=3)
    rec = lambda hp: {"kind": "scorecard", "capability": "vision.describe",
                      "model": "m1", "hard_pass": hp,
                      "detail": {"worker": "computron"}}
    assert checks.check_scorecards([rec(False)] * 2, s) == []
    # A pass in between resets the streak.
    assert checks.check_scorecards(
        [rec(False), rec(False), rec(True), rec(False)], s) == []
    out = checks.check_scorecards([rec(False)] * 3, s)
    assert [a.fingerprint for a in out] == \
        ["scorecard_hard_fail:vision.describe:m1"]
    assert out[0].evidence["consecutive_hard_fails"] == 3
    assert out[0].evidence["worker"] == "computron"


def test_detect_appends_capability_snapshot_for_next_pass(tmp_path):
    s = _settings(tmp_path)
    cap = {"name": "text.chat", "eligibility": {"eligible": True,
                                                "reasons": []}}
    get = _http_get_for(_surfaces(capabilities=[cap]))
    assert checks.detect(s, http_get=get, now=NOW, wants_fn=lambda: []) == []
    # Pass 2: same capability now ineligible -> transition fires.
    cap2 = {"name": "text.chat",
            "eligibility": {"eligible": False, "reasons": ["all blocked"]}}
    get2 = _http_get_for(_surfaces(capabilities=[cap2]))
    out = checks.detect(s, http_get=get2, now=NOW + 600,
                        wants_fn=lambda: [])
    assert [a.kind for a in out] == ["capability_lost"]


# --------------------------------------------------------------------------
# case store


def test_open_case_dedupe_touches_never_respawns(tmp_path):
    store = CaseStore(str(tmp_path / "cases.db"))
    a = Anomaly("worker_offline:computron", "worker_offline", "critical",
                {"worker": "computron"})
    c1, created1 = store.open_or_touch(a, now=NOW)
    c2, created2 = store.open_or_touch(a, now=NOW + 60)
    assert created1 is True and created2 is False
    assert c2.id == c1.id
    assert c2.last_seen == NOW + 60 and c2.opened_at == NOW
    # Still deduped after the agent ran (state moved past open).
    store.set_state(c1.id, "documented")
    c3, created3 = store.open_or_touch(a, now=NOW + 120)
    assert created3 is False and c3.id == c1.id
    # A CLOSED case releases the fingerprint: a recurrence is a new case.
    store.set_state(c1.id, "closed")
    c4, created4 = store.open_or_touch(a, now=NOW + 180)
    assert created4 is True and c4.id != c1.id
    store.close()


def test_case_lifecycle_and_listing(tmp_path):
    store = CaseStore(str(tmp_path / "cases.db"))
    c, _ = store.open_or_touch(Anomaly("f1", "job_stalled", "critical", {}))
    store.set_state(c.id, "agent_running")
    store.attach_agent(c.id, "run-123", "/x/report.md")
    store.set_state(c.id, "documented")
    got = store.get(c.id)
    assert (got.state, got.agent_run_id, got.report_path) == \
        ("documented", "run-123", "/x/report.md")
    assert [x.id for x in store.list(state="documented")] == [c.id]
    with pytest.raises(ValueError):
        store.set_state(c.id, "bogus")
    store.close()


# --------------------------------------------------------------------------
# remedies: whitelist gating + structural ae exclusion


def _wedge_anomaly(worker="computron"):
    return Anomaly("scorecard_hard_fail:vision.describe:m1",
                   "scorecard_hard_fail", "critical",
                   {"worker": worker, "model_key": "m1"})


def test_whitelist_is_all_reversible_and_typed():
    assert {r.name for r in remedies.WHITELIST} == {
        "worker_model_unload", "worker_slot_unload",
        "worker_slot_relaunch", "central_chat_cancel",
        "enqueue_download"}
    assert all(r.reversible for r in remedies.WHITELIST)
    assert all(r.method == "POST" for r in remedies.WHITELIST)
    # k97: downloads are the ONLY remedy on the downloads gate; every other
    # remedy stays behind the default-OFF remedies gate.
    assert {r.name for r in remedies.WHITELIST if r.gate == "downloads"} \
        == {"enqueue_download"}


def test_eligible_maps_kinds_to_remedies():
    names = {r.name for r in remedies.eligible(_wedge_anomaly())}
    assert names == {"worker_model_unload"}
    stalled = Anomaly("job_stalled:j1", "job_stalled", "critical",
                      {"worker": "computron", "slot_id": 2,
                       "request_id": "j1"})
    names = {r.name for r in remedies.eligible(stalled)}
    assert names == {"worker_slot_unload", "worker_slot_relaunch",
                     "central_chat_cancel"}


def test_ae_is_structurally_excluded_from_eligibility():
    assert remedies.eligible(_wedge_anomaly(worker="ae")) == []


def test_execute_is_gated_off_by_default(tmp_path):
    settings = SentinelSettings(state_dir=str(tmp_path))
    assert settings.remedies_enabled is False          # THE default
    remedy = next(r for r in remedies.WHITELIST
                  if r.name == "worker_model_unload")
    with pytest.raises(remedies.RemediesDisabled):
        remedies.execute(remedy, {"worker": "computron",
                                  "worker_base": "http://w:9100",
                                  "model_key": "m1"}, settings,
                         http_post=lambda url, body: pytest.fail(
                             "must not POST while disabled"))


def test_execute_refuses_ae_even_when_enabled(tmp_path):
    settings = SentinelSettings(state_dir=str(tmp_path),
                                remedies_enabled=True)
    remedy = next(r for r in remedies.WHITELIST
                  if r.name == "worker_model_unload")
    with pytest.raises(remedies.ProdWorkerExcluded):
        remedies.execute(remedy, {"worker": "ae",
                                  "worker_base": "http://ae:9100",
                                  "model_key": "m1"}, settings,
                         http_post=lambda url, body: pytest.fail(
                             "must not POST at ae"))


def test_execute_happy_path_when_enabled(tmp_path):
    settings = SentinelSettings(state_dir=str(tmp_path),
                                remedies_enabled=True)
    posts = []
    remedy = next(r for r in remedies.WHITELIST
                  if r.name == "worker_slot_relaunch")
    out = remedies.execute(remedy, {"worker": "computron",
                                    "worker_base": "http://w:9100",
                                    "slot_id": 2}, settings,
                           http_post=lambda url, body:
                               posts.append((url, body)) or {"ok": True})
    assert out == {"ok": True}
    assert posts == [("http://w:9100/slots/2/relaunch", {"slot_id": 2})]
    with pytest.raises(ValueError):
        remedies.execute(remedy, {"worker": "computron",
                                  "worker_base": "http://w:9100"}, settings,
                         http_post=lambda u, b: {"ok": True})


def test_remedies_env_gate_default_off():
    s = load_settings(environ={})
    assert s.remedies_enabled is False
    assert load_settings(environ={"HUGPY_SENTINEL_REMEDIES": "1"}
                         ).remedies_enabled is True


# --------------------------------------------------------------------------
# runner: spawn with subprocess monkeypatched


class _FakeProc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = rc, stdout, stderr


def _fake_run(calls, report=None, rc=0, raise_timeout=False):
    def run(cmd, capture_output=None, text=None, timeout=None):
        calls.append({"cmd": cmd, "timeout": timeout})
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd, timeout)
        stdout = "" if report is None else "\n" + json.dumps(report)
        return _FakeProc(rc=rc, stdout=stdout, stderr="[tool] …")
    return run


def test_spawn_writes_brief_and_documents_case(tmp_path):
    s = _settings(tmp_path, agent_cmd="hugpy-agent", agent_max_steps=20,
                  agent_timeout_s=900.0)
    store = CaseStore(s.db_path)
    case, _ = store.open_or_touch(_wedge_anomaly(), now=NOW)
    calls = []
    report = {"run_id": "r-1", "outcome": "done",
              "answer": "# Case report: scorecard_hard_fail:vision.describe:m1"}
    updated = runner.spawn_agent_for_case(
        case, store, s, run=_fake_run(calls, report=report))

    # The invocation: the `case` subcommand, brief file, case dir, bounded.
    (call,) = calls
    assert call["cmd"][0] == "hugpy-agent"
    assert call["cmd"][1] == "case"
    assert call["timeout"] == 900.0
    assert "--case-dir" in call["cmd"] and "--max-steps" in call["cmd"]
    brief_path = call["cmd"][2]
    case_dir = call["cmd"][call["cmd"].index("--case-dir") + 1]
    assert brief_path.startswith(case_dir)

    # The brief: evidence + document-only rule + output contract + ae rule.
    brief = open(brief_path).read()
    assert '"worker": "computron"' in brief
    assert '"model_key": "m1"' in brief
    assert "DOCUMENT-ONLY" in brief
    assert "must NOT mutate anything" in brief
    assert "Root-cause hypothesis" in brief
    assert "worker_model_unload" in brief          # whitelist names listed
    assert '"ae" is document+escalate only' in brief

    # Case documented, report stored under the case dir, ledger appended.
    assert updated.state == "documented"
    assert updated.agent_run_id == "r-1"
    assert open(updated.report_path).read().startswith("# Case report:")
    ledger = open(s.ledger_path).read()
    assert "scorecard_hard_fail" in ledger and "documented" in ledger
    store.close()


def test_spawn_timeout_escalates(tmp_path):
    s = _settings(tmp_path, agent_timeout_s=5.0)
    store = CaseStore(s.db_path)
    case, _ = store.open_or_touch(_wedge_anomaly(), now=NOW)
    updated = runner.spawn_agent_for_case(
        case, store, s, run=_fake_run([], raise_timeout=True))
    assert updated.state == "escalated"
    assert "timeout" in (updated.note or "")
    store.close()


def test_spawn_bad_outcome_escalates(tmp_path):
    s = _settings(tmp_path)
    store = CaseStore(s.db_path)
    case, _ = store.open_or_touch(_wedge_anomaly(), now=NOW)
    updated = runner.spawn_agent_for_case(
        case, store, s,
        run=_fake_run([], report={"run_id": "r-2", "outcome": "max_steps"},
                      rc=1))
    assert updated.state == "escalated"
    assert updated.report_path is None
    assert "max_steps" in (updated.note or "")
    store.close()


def test_run_once_spawns_only_for_new_cases(tmp_path):
    s = _settings(tmp_path)
    jobs = [{"id": "j1", "status": "processing", "stalled": True,
             "progressed_at": NOW - 400, "worker": "computron"}]
    get = _http_get_for(_surfaces(jobs=jobs))
    calls = []
    report = {"run_id": "r-9", "outcome": "done", "answer": "# Case report"}
    store = CaseStore(s.db_path)
    summary1 = runner.run_once(s, store=store, http_get=get,
                               run=_fake_run(calls, report=report),
                               wants_fn=lambda: [])
    assert len(summary1["opened"]) == 1 and len(calls) == 1
    # Re-detection: same fingerprint -> touch, NO second spawn.
    summary2 = runner.run_once(s, store=store, http_get=get,
                               run=_fake_run(calls, report=report),
                               wants_fn=lambda: [])
    assert summary2["opened"] == [] and summary2["touched"] == summary1["opened"]
    assert len(calls) == 1
    store.close()


# --------------------------------------------------------------------------
# k96 pilot light — the agent brain ladder's always-on floor


def _workers_with(*allocs):
    return [{"name": "computron", "id": "w-1", "status": "online",
             "allocations": list(allocs)}]


PILOT = "Qwen~Qwen2.5-3B-Instruct-GGUF"


def test_pilot_light_unset_check_is_off(tmp_path):
    s = _settings(tmp_path, pilot_light="")
    assert checks.check_pilot_light(_workers_with(), s) == []


def test_pilot_light_warm_slot_is_quiet(tmp_path):
    s = _settings(tmp_path, pilot_light=PILOT)
    body = _workers_with({"kind": "slot", "model_key": PILOT, "healthy": True})
    assert checks.check_pilot_light(body, s) == []


def test_pilot_light_ram_resident_is_quiet_and_bare_tail_matches(tmp_path):
    s = _settings(tmp_path, pilot_light=PILOT)
    body = _workers_with({"kind": "ram",
                          "model_key": "Qwen2.5-3B-Instruct-GGUF"})
    assert checks.check_pilot_light(body, s) == []


def test_pilot_light_absent_or_unhealthy_is_warn_anomaly(tmp_path):
    s = _settings(tmp_path, pilot_light=PILOT)
    # absent entirely
    out = checks.check_pilot_light(
        _workers_with({"kind": "slot", "model_key": "other", "healthy": True}), s)
    assert [a.kind for a in out] == ["pilot_light_not_resident"]
    assert out[0].severity == "warn"
    assert out[0].evidence["model_key"] == PILOT
    assert "boot-prewarm" in out[0].evidence["hint"]
    # present but wedged (healthy: False) is NOT warm
    out2 = checks.check_pilot_light(
        _workers_with({"kind": "slot", "model_key": PILOT, "healthy": False}), s)
    assert [a.kind for a in out2] == ["pilot_light_not_resident"]
    # malformed rows never crash the check
    assert checks.check_pilot_light(["junk", {"allocations": ["x"]}], s) != []


def test_pilot_light_rides_detect(tmp_path):
    s = _settings(tmp_path, pilot_light=PILOT)
    get = _http_get_for(_surfaces(workers=_workers_with(
        {"kind": "slot", "model_key": "other", "healthy": True})))
    kinds = [a.kind for a in checks.detect(s, http_get=get, now=NOW,
                                           wants_fn=lambda: [])]
    assert "pilot_light_not_resident" in kinds


def test_pilot_light_settings_resolution():
    # explicit override wins; else derived from the ladder's LAST entry;
    # neither -> off.
    s = load_settings({"HUGPY_SENTINEL_PILOT_LIGHT": "X~Y-GGUF",
                       "HUGPY_AGENT_BRAINS": "A~B,C~D"})
    assert s.pilot_light == "X~Y-GGUF"
    s = load_settings({"HUGPY_AGENT_BRAINS": " A~B , %s " % PILOT})
    assert s.pilot_light == PILOT
    assert load_settings({}).pilot_light == ""
