"""Chaos-and-learn exerciser (p1, EPOCH CLOSER).

A randomization mechanism that exercises the entire assortment — models x cards
x alloc modes x ctx% — chaotically over a bounded run, recording ONE structured
observation per trial so the t28 learner can learn placement templates from
measured reality.

Each trial:
  1. draw a combo (seeded, reproducible);
  2. price the PREDICTED side (/models/<key>/meta) and check feasibility —
     predicted-infeasible combos are recorded and NOT fired;
  3. snapshot the target workers' current spill, apply the chaos spill
     (operator-gated /assign);
  4. fire a small keyless /chat/stream generation;
  5. read the MEASURED side (worker allocation row + refusal detail);
  6. RESTORE the prior spill and verify;
  7. append the observation.

Safety rails (non-negotiable, see the module tests):
  * only already-assigned (worker, model) pairs are exercised → restore is a
    clean write-back, never an unassign;
  * honours protections implicitly — it never evicts anything itself; the
    worker's own admission gate (protection-aware) decides;
  * at most ONE heavy trial in flight per worker (the run is sequential; a
    WorkerGate enforces it as an invariant);
  * predicted-infeasible combos are skipped, not fired;
  * backs off (skips the round) when /llm/jobs shows non-chaos active work;
  * stops + restores if central health != 200 twice;
  * clean SIGTERM: finish/restore the current trial, then stop;
  * bounded by --rounds and --budget-minutes.

CLI:  hugpy-chaos [--dry-run] [--rounds N] [--seed S] [--budget-minutes M]
      hugpy-chaos sweep [--plan] [--top-n N] ...   (chaos/sweep.py)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

from hugpy_ops.chaos import alloc, observe
from hugpy_ops.chaos.assortment import draw_combo, enumerate_assortment, worker_index
from hugpy_ops.chaos.client import CentralClient, DEFAULT_BASE, default_base_url
from hugpy_ops.chaos.schema import blank_observation, validate_observation

# Legacy fixed locations, kept as the documented names; the CLI defaults go
# through default_out_dir() / default_env_file() so the platform's store root
# (DEFAULT_ROOT / models_root()) and HUGPY_ENV_FILE win when set.
DEFAULT_OUT_DIR = "/mnt/llm_storage/comms/chaos"
DEFAULT_ENV_FILE = "/srv/share/projects/hugpy/d-env/env"


def default_out_dir() -> str:
    """``<models_root()>/comms/chaos`` — observations + run manifests live
    beside the store the platform resolves (``/mnt/llm_storage`` where that
    legacy mount exists, else the per-user data dir)."""
    if os.environ.get("HUGPY_CHAOS_OUT_DIR"):
        return os.environ["HUGPY_CHAOS_OUT_DIR"]
    from hugpy_platform.app_dirs import models_root
    return os.path.join(models_root(), "comms", "chaos")


def default_env_file() -> str:
    return os.environ.get("HUGPY_ENV_FILE") or DEFAULT_ENV_FILE
PROMPT = "Reply with exactly one word: OK"


class WorkerGate:
    """At most one in-flight heavy trial per worker. The run is sequential so
    this is normally a no-op, but it is enforced (and tested) so the invariant
    holds if the runner is ever parallelised."""

    def __init__(self):
        self._busy: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, workers: list[str]) -> bool:
        with self._lock:
            if any(w in self._busy for w in workers):
                return False
            self._busy.update(workers)
            return True

    def release(self, workers: list[str]) -> None:
        with self._lock:
            self._busy.difference_update(workers)


def load_operator_token(explicit: str | None, env_file: str) -> str | None:
    """Resolve the operator token: explicit arg > env var > d-env/env file."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("HUGPY_OPERATOR_TOKEN")
    if env:
        return env.strip()
    p = Path(env_file)
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("HUGPY_OPERATOR_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


class ChaosRunner:
    def __init__(self, client: CentralClient, *, seed: int, rounds: int,
                 budget_minutes: float, out_dir: str, max_new_tokens: int,
                 chat_ceiling_s: float, settle_s: float = 1.0,
                 max_model_bytes: int | None = None,
                 run_id: str | None = None):
        self.client = client
        self.seed = seed
        self.rng = random.Random(seed)
        self.rounds = rounds
        self.budget_s = budget_minutes * 60.0
        self.out_dir = Path(out_dir)
        self.max_new_tokens = max_new_tokens
        self.chat_ceiling_s = chat_ceiling_s
        self.settle_s = settle_s
        self.max_model_bytes = max_model_bytes
        self.run_id = run_id or f"chaos-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self.gate = WorkerGate()
        self._stop = False
        self._health_fails = 0
        self.counts: dict[str, int] = {}
        self.obs_path = self.out_dir / "observations.jsonl"
        self.manifest_path = self.out_dir / "runs" / f"{self.run_id}.json"

    # ── lifecycle ────────────────────────────────────────────────────────────
    def request_stop(self, *_):
        self._stop = True

    def _tick(self, key: str):
        self.counts[key] = self.counts.get(key, 0) + 1

    def _is_chaos_job(self, job: dict) -> bool:
        rid = str(job.get("id") or job.get("request_id") or "")
        return self.run_id in rid or rid.startswith("chaos-")

    def _foreign_active(self, jobs: dict) -> bool:
        """True if any non-chaos job is active/waiting (be a guest — back off)."""
        for j in (jobs.get("jobs") or []):
            status = str(j.get("status") or "").lower()
            if status in ("active", "running", "processing", "waiting",
                          "queued", "pending") and not self._is_chaos_job(j):
                return True
        return False

    def _append(self, obs: dict):
        problems = validate_observation(obs)
        if problems:  # a malformed obs is a bug — record it, don't poison silently
            obs.setdefault("_schema_problems", problems)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.obs_path, "a") as f:
            f.write(json.dumps(obs) + "\n")

    def _write_manifest(self, status: str, started: float, ended: float | None,
                        assortment: dict):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": self.run_id, "seed": self.seed, "status": status,
            "rounds_requested": self.rounds,
            "budget_minutes": round(self.budget_s / 60.0, 2),
            "max_new_tokens": self.max_new_tokens,
            "started": started, "ended": ended,
            "observations_path": str(self.obs_path),
            "counts": dict(self.counts),
            "assortment_summary": {
                k: assortment.get(k) for k in (
                    "n_models_total", "n_servable", "n_exercisable",
                    "blocked_excluded", "approx_combo_space", "alloc_modes",
                    "ctx_pcts")},
            "workers": assortment.get("workers"),
            "base_url": self.client.base,
        }
        with open(self.manifest_path, "w") as f:
            f.write(json.dumps(manifest, indent=1))

    # ── one trial ────────────────────────────────────────────────────────────
    def run_trial(self, rnd: int) -> dict:
        obs = blank_observation()
        t0 = time.time()
        obs.update({"run_id": self.run_id, "trial_id": f"{self.run_id}-r{rnd}",
                    "seed": self.seed, "round": rnd, "ts_start": t0})

        # health gate
        if self.client.health() != 200:
            self._health_fails += 1
            obs.update({"kind": "skip", "skip_reason": "health-degraded"})
            return self._finish(obs, t0)
        self._health_fails = 0

        workers = self.client.workers()
        models = self.client.models()

        # back off if real traffic is active
        jobs = self.client.jobs()
        if self._foreign_active(jobs):
            obs.update({"kind": "skip", "skip_reason": "back-off-foreign-jobs",
                        "back_off": True})
            return self._finish(obs, t0)

        combo = draw_combo(self.rng, models, workers, self.max_model_bytes)
        if combo is None:
            obs.update({"kind": "skip", "skip_reason": "assortment-empty"})
            return self._finish(obs, t0)
        card_gib = combo.pop("_card_gib", None)  # internal draw hint, not schema
        obs["combo"] = {k: combo[k] for k in (
            "model_key", "framework", "effective_bytes", "alloc_mode", "spill",
            "ctx_pct", "target_workers", "was_warm", "warm_on")}

        cands = combo["target_workers"]
        if not cands:
            obs.update({"kind": "skip", "skip_reason": "no-servable-worker"})
            return self._finish(obs, t0)

        # predicted side + feasibility
        obs["predicted"] = observe.build_predicted(self.client, combo, workers)
        if obs["predicted"].get("feasible") is False:
            obs.update({"kind": "skip", "skip_reason": "predicted-infeasible"})
            return self._finish(obs, t0)

        # per-worker sequential invariant
        if not self.gate.claim(cands):
            obs.update({"kind": "skip", "skip_reason": "stopped",
                        "back_off": True})
            obs["measured"]["error"] = "worker busy with another chaos trial"
            return self._finish(obs, t0)

        widx = worker_index(workers)
        snap = alloc.snapshot(self.client, combo["model_key"], cands, widx)
        applied = alloc.apply(self.client, combo["model_key"], combo["spill"], snap)
        try:
            if not applied["ok"]:
                # engine gate / bad shape refused the alloc — restore whatever
                # DID change, record, do NOT fire.
                obs.update({"kind": "skip", "skip_reason": "alloc-refused"})
                obs["measured"]["error"] = json.dumps(applied["results"])[:500]
                obs["restore"] = alloc.restore(self.client, combo["model_key"], snap)
                return self._finish(obs, t0)

            ev_before = observe.evictions_snapshot(workers)
            terminal = self.client.chat_stream(
                combo["model_key"], PROMPT, obs["trial_id"],
                self.max_new_tokens, self.chat_ceiling_s)
            time.sleep(self.settle_s)  # let the heartbeat settle
            workers_after = self.client.workers()
            jobs_after = self.client.jobs()
            obs["measured"] = observe.build_measured(
                terminal, workers_after, combo["model_key"], ev_before, jobs_after)
        finally:
            obs["restore"] = alloc.restore(self.client, combo["model_key"], snap)
            self.gate.release(cands)
        return self._finish(obs, t0)

    def _finish(self, obs: dict, t0: float) -> dict:
        t1 = time.time()
        obs["ts_end"] = t1
        obs["duration_s"] = round(t1 - t0, 3)
        self._tick(obs.get("skip_reason") or obs["measured"].get("outcome")
                   or "trial")
        self._append(obs)
        return obs

    # ── run loop ─────────────────────────────────────────────────────────────
    def run(self) -> dict:
        started = time.time()
        workers = self.client.workers()
        models = self.client.models()
        assortment = enumerate_assortment(models, workers, self.max_model_bytes)
        self._write_manifest("running", started, None, assortment)
        status = "completed"
        try:
            for rnd in range(self.rounds):
                if self._stop:
                    status = "stopped-sigterm"
                    break
                if time.time() - started >= self.budget_s:
                    status = "budget-exhausted"
                    break
                obs = self.run_trial(rnd)
                if self._health_fails >= 2:
                    status = "health-degraded"
                    break
                sr = obs.get("skip_reason")
                oc = obs["measured"].get("outcome")
                print(f"[r{rnd}] {obs['combo'].get('model_key')} "
                      f"mode={obs['combo'].get('alloc_mode')} "
                      f"ctx={obs['combo'].get('ctx_pct')} -> "
                      f"{sr or oc} verdict={obs['measured']['admission']['verdict']} "
                      f"served={obs['measured'].get('served_worker')} "
                      f"restore_ok={obs['restore'].get('ok')}", flush=True)
        finally:
            ended = time.time()
            self._write_manifest(status, started, ended, assortment)
        return {"run_id": self.run_id, "status": status, "counts": self.counts,
                "observations_path": str(self.obs_path),
                "manifest_path": str(self.manifest_path)}


def main(argv=None) -> int:
    # Subcommand dispatch: `hugpy-chaos sweep ...` -> the deterministic k7 offload
    # speed-cliff sweep (chaos/sweep.py); anything else -> the random exerciser.
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] == "sweep":
        from hugpy_ops.chaos import sweep as _sweep
        return _sweep.main(args_in[1:])

    ap = argparse.ArgumentParser(
        prog="hugpy-chaos",
        description="Chaos-and-learn exerciser: randomly exercise the model x "
                    "card x alloc-mode x ctx%% assortment, recording predicted-"
                    "vs-measured observations for the learner. Subcommand "
                    "`sweep` runs the deterministic k7 offload speed-cliff sweep.")
    ap.add_argument("--base-url", default=default_base_url(),
                    help="central base URL (env HUGPY_BASE_URL; legacy aliases "
                         "honoured; default loopback central)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed (default: random; printed for reproduction)")
    ap.add_argument("--budget-minutes", type=float, default=45.0)
    ap.add_argument("--max-new-tokens", type=int, default=16,
                    help="cap per-trial generation length (small = polite guest)")
    ap.add_argument("--chat-ceiling-s", type=float, default=120.0)
    ap.add_argument("--max-model-gib", type=float, default=None,
                    help="cap exercised models to <= this weight (polite guest "
                         "on a busy fleet; e.g. 3.5 for the live proof)")
    ap.add_argument("--out-dir", default=None,
                    help="observations + run manifests (default: "
                         "<store root>/comms/chaos, HUGPY_CHAOS_OUT_DIR)")
    ap.add_argument("--operator-token", default=None,
                    help="operator token (else env HUGPY_OPERATOR_TOKEN or "
                         "the env file)")
    ap.add_argument("--env-file", default=None,
                    help="KEY=VALUE file holding HUGPY_OPERATOR_TOKEN "
                         "(default HUGPY_ENV_FILE or %s)" % DEFAULT_ENV_FILE)
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate + plan only; no /assign, no generation")
    args = ap.parse_args(argv)
    args.out_dir = args.out_dir or default_out_dir()
    args.env_file = args.env_file or default_env_file()

    seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
    max_model_bytes = (int(args.max_model_gib * (1 << 30))
                       if args.max_model_gib else None)

    if args.dry_run:
        client = CentralClient(args.base_url, operator_token=None)
        models = client.models()          # fetched once; the cube is static here
        workers = client.workers()
        assortment = enumerate_assortment(models, workers, max_model_bytes)
        rng = random.Random(seed)
        plan = []
        for i in range(args.rounds):
            c = draw_combo(rng, models, workers, max_model_bytes)
            if not c:
                continue
            c.pop("_card_gib", None)
            plan.append({"round": i, "model_key": c["model_key"],
                         "framework": c["framework"],
                         "alloc_mode": c["alloc_mode"],
                         "ctx_pct": c["ctx_pct"], "spill": c["spill"],
                         "target_workers": c["target_workers"],
                         "was_warm": c["was_warm"]})
        out = {"seed": seed, "assortment": assortment, "planned_trials": plan}
        print(json.dumps(out, indent=1))
        return 0

    token = load_operator_token(args.operator_token, args.env_file)
    if not token:
        print("ERROR: no operator token (need HUGPY_OPERATOR_TOKEN / "
              "--operator-token / d-env/env) — /assign is operator-gated.",
              file=sys.stderr)
        return 2

    client = CentralClient(args.base_url, operator_token=token)
    if client.health() != 200:
        print("ERROR: central health != 200; refusing to start.", file=sys.stderr)
        return 3

    runner = ChaosRunner(
        client, seed=seed, rounds=args.rounds,
        budget_minutes=args.budget_minutes, out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens, chat_ceiling_s=args.chat_ceiling_s,
        max_model_bytes=max_model_bytes)
    signal.signal(signal.SIGTERM, runner.request_stop)
    signal.signal(signal.SIGINT, runner.request_stop)
    print(f"chaos run {runner.run_id} seed={seed} rounds={args.rounds} "
          f"budget={args.budget_minutes}min -> {runner.obs_path}", flush=True)
    result = runner.run()
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
