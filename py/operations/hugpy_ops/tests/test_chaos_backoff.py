"""chaos runner: end-to-end trial behaviour with an in-memory fleet —
back-off on foreign jobs, health-degraded stop, predicted-infeasible skip, the
per-worker sequential gate, a full happy-path trial with verified restore, and
observation append + schema validity."""
from __future__ import annotations

import json
from pathlib import Path

from chaos_fakes import GIB, FakeClient

from hugpy_ops.chaos.runner import ChaosRunner, WorkerGate, load_operator_token
from hugpy_ops.chaos.schema import validate_observation


def mk_runner(client, tmp_path, seed=7):
    return ChaosRunner(client, seed=seed, rounds=1, budget_minutes=45,
                       out_dir=str(tmp_path), max_new_tokens=8,
                       chat_ceiling_s=5, settle_s=0)


def test_worker_gate_is_sequential_per_worker():
    g = WorkerGate()
    assert g.claim(["ae", "computron"]) is True
    assert g.claim(["ae"]) is False
    g.release(["ae", "computron"])
    assert g.claim(["ae"]) is True


def test_foreign_active_jobs_back_off(tmp_path):
    foreign = {"jobs": [{"id": "v1-realtraffic", "status": "active"}],
               "counts": {"active": 1}}
    c = FakeClient(jobs=foreign)
    r = mk_runner(c, tmp_path)
    obs = r.run_trial(0)
    assert obs["kind"] == "skip" and obs["skip_reason"] == "back-off-foreign-jobs"
    assert obs["back_off"] is True
    assert c.chat_calls == [] and c.assign_calls == []
    own = {"jobs": [{"id": f"{r.run_id}-r0", "status": "active"}]}
    assert r._foreign_active(own) is False


def test_health_degraded_skips(tmp_path):
    c = FakeClient(health_code=503)
    obs = mk_runner(c, tmp_path).run_trial(0)
    assert obs["kind"] == "skip" and obs["skip_reason"] == "health-degraded"
    assert c.chat_calls == []


def test_predicted_infeasible_is_recorded_not_fired(tmp_path):
    class OnlyHuge(FakeClient):
        def models(self):
            return [m for m in super().models() if m["model_key"] == "huge-gguf"]
    c = OnlyHuge()
    obs = mk_runner(c, tmp_path).run_trial(0)
    assert obs["kind"] == "skip" and obs["skip_reason"] == "predicted-infeasible"
    assert c.chat_calls == [] and c.assign_calls == []
    assert obs["predicted"]["need_bytes"] and obs["predicted"]["infeasible_reason"]


def test_happy_path_trial_restores_and_appends_observation(tmp_path):
    class OnlySmall(FakeClient):
        def models(self):
            return [m for m in super().models() if m["model_key"] == "small-gguf"]
    alloc_row = {"kind": "slot", "vram_bytes": 2 * GIB, "rss_bytes": 3 * GIB,
                 "n_gpu_layers": -1, "total_layers": 29, "ctx": 16384,
                 "serving": True}
    c = OnlySmall(chat_terminal={"outcome": "done", "served_worker": "computron",
                                 "error": None, "finish_reason": "stop",
                                 "ttft_s": 0.3, "load_duration_s": 1.1,
                                 "wall_s": 1.8, "tokens": 2, "stages": []},
                  materialize_alloc={"computron": alloc_row})
    r = mk_runner(c, tmp_path, seed=1)
    pre = {w["name"]: dict(w.get("spill_by_model", {})) for w in c.workers()}
    obs = r.run_trial(0)
    post = {w["name"]: dict(w.get("spill_by_model", {})) for w in c.workers()}
    assert obs["kind"] == "trial"
    assert len(c.chat_calls) == 1
    assert obs["measured"]["allocation"]["vram_bytes"] == 2 * GIB
    assert obs["measured"]["admission"]["verdict"] == "proceed"
    assert obs["restore"]["ok"] is True
    assert pre == post
    assert validate_observation(obs) == []

    obs_file = Path(tmp_path) / "observations.jsonl"
    lines = obs_file.read_text().strip().splitlines()
    assert len(lines) == 1
    assert validate_observation(json.loads(lines[0])) == []
    r._write_manifest("test", 1.0, 2.0, {"n_models_total": 1, "workers": []})
    assert (Path(tmp_path) / "runs" / f"{r.run_id}.json").is_file()


def test_operator_token_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / "env"
    env_file.write_text('HUGPY_OPERATOR_TOKEN="tok-from-file"\n')
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    assert load_operator_token("explicit-tok", str(env_file)) == "explicit-tok"
    assert load_operator_token(None, str(env_file)) == "tok-from-file"
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "tok-from-env")
    assert load_operator_token(None, str(env_file)) == "tok-from-env"
    assert load_operator_token(None, str(tmp_path / "missing")) == "tok-from-env"
