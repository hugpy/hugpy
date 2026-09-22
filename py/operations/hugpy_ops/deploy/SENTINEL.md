# hugpy-sentinel — bound-exceeded watcher (k95)

Operator directive (2026-08-06): *"the hugpy-agent needs to be deployed under
suspected hangs and edge cases where expected bounds are exceeded — to
document, remedy, fix or other."* The sentinel formalizes the loop the keeper
ran by hand on the 2026-08-06 vision wedge (repeated worker 500s → diagnose
stale projector-less seat → evict → verify → document).

## What one pass does (`hugpy-sentinel run-once`)

1. **Detect** — four read surfaces of central, over HTTP (never imports
   central's internals): `GET /llm/jobs?live=0` (stalled beyond grace,
   expired-without-real-terminal), `GET /llm/workers` (offline/unreachable,
   `version_ok: false` — the tri-state `null` is not an anomaly),
   `GET /oracle/capabilities` (eligible→ineligible transition vs. the last
   snapshot), plus a local scorecard history (`record-scorecard` feeds it
   from observed `POST /oracle/route` responses; N consecutive
   `hard_pass=false` for one (capability, model) is an anomaly).
2. **Case** — one OPEN case per fingerprint in a SQLite store under
   `$HUGPY_SENTINEL_DIR` (partial unique index; re-detection touches
   `last_seen`, never re-spawns).
3. **Diagnose** — for each NEWLY opened case, spawn ONE bounded
   `hugpy-agent case <brief> --case-dir <dir>` subprocess (default 900 s
   timeout). The agent runs under a pinned DOCUMENT-ONLY policy profile
   (readonly + case-dir-jailed `fs_write` + `http_fetch`; shell/spawn
   hard-denied) and must return a markdown case report, stored as
   `report.md` in the case dir; a one-liner lands in `cases.md`.
4. States: `open → agent_running → documented | escalated`; `remedied` and
   `closed` are operator moves (`status` lists them all).

## Remedies are OFF by default

`hugpy_ops/sentinel/remedies.py` is the complete whitelist (worker
`POST /models/unload`, `POST /slots/<id>/unload|relaunch`, central
`POST /llm/chat/cancel/<id>` — all reversible). Execution raises
`RemediesDisabled` unless `HUGPY_SENTINEL_REMEDIES=1` is set, and the prod
worker **ae** is excluded structurally (`PROD_EXCLUDED_WORKERS`, enforced in
both eligibility and execute) — ae cases are document+escalate only, always.

**Exception — downloads are ON by default (k97, operator ruling 2026-08-06:
"autonomous downloads should be greenlit — especially for ComfyUI").** The
`enqueue_download` remedy rides its OWN gate, `HUGPY_SENTINEL_DOWNLOADS`
(default ON; set `0` to turn it off), separate from the remedies gate above.
Each pass runs the k97 provisioner scan (`hugpy_ops/provisioner.py`
— declared-but-missing weights across the comfy/studio/tasks registries,
read through `hugpy_engine`, `hugpy_video.intel.studio` and `hugpy_storage`) and
opens one info-severity `weight_missing` case per missing weight. Those cases
take a deterministic FAST PATH (no agent spawn): the sentinel POSTs
`{central}/llm/repos/download` — the console's own add-models enqueue, so the
job lands on the hugpy-downloader-dev queue with dedupe/progress/cancel —
then documents the case as `remedied`. A want whose source can't be proven
from its registry row (no HF hub id) is documented UNRESOLVED, never guessed.
The ae exclusion is irrelevant here by construction: downloads land on
central's shared storage, never on a worker. Dry-run the same scan by hand:
`hugpy-provisioner` (add `--apply` to enqueue).

## Install (operator; the keeper only writes these files)

```sh
pip install hugpy-ops            # provides the hugpy-sentinel console script
sudo cp <hugpy_ops>/deploy/hugpy-sentinel.{service,timer} /etc/systemd/system/
# edit ExecStart's venv path per box, then
sudo systemctl daemon-reload
sudo systemctl enable --now hugpy-sentinel.timer
systemctl list-timers hugpy-sentinel*; journalctl -u hugpy-sentinel -f
```

Later, to arm remedies (deliberate, after reviewing documented cases): add
`Environment=HUGPY_SENTINEL_REMEDIES=1` to the service, `daemon-reload`,
done — nothing else changes.

## Brain ladder + pilot light (k96, operator ruling 2026-08-06)

The 2026-08-06 shakedown aborted its case runs because the agent's single
configured brain (Qwen3-Coder-Next) was mid-reload. The fix is an ordered
**brain ladder**: `HUGPY_AGENT_BRAINS` (csv, best-first) on the sentinel unit
— spawned case agents inherit it. A run starts on the first entry that is
**warm** on a fleet worker (`GET /llm/workers` allocations) and walks DOWN one
entry per capacity-class / permanent-verdict refusal (forward-only). Every
brain chat carries `"no_makeroom": true`, so a cold brain load can NEVER evict
a fleet model — it takes genuinely free room or is refused fast (which the
ladder walks). A run answered below ladder position 1 stamps
`[answered by <model> (ladder position N of M)]` onto its report, so a
pilot-light case report is visibly reduced-depth.

**The pilot light** is the LAST ladder entry by convention: a model small
enough that a cold load is cheap, so the sentinel always has *some* brain even
when the fleet is wedged. Keep it warm with the existing keep-warm star (no
new pinning machinery — reconcile reseats a star every beat, even after a
pressure eviction):

```sh
# pin the pilot light on a worker (operator-gated route); example: computron
curl -X POST http://127.0.0.1:7002/api/llm/workers/<worker_id>/boot-prewarm \
     -H 'Content-Type: application/json' \
     -d '{"model_key": "Qwen~Qwen2.5-3B-Instruct-GGUF"}'
# verify: the star map, and the model's allocation on /llm/workers
curl http://127.0.0.1:7002/api/llm/workers/boot-prewarm
```

Recommended ladder for THIS fleet (ae 3090 + computron 4060-8GB + op,
2026-08-06 — warm today: Qwen3-Coder-Next on the ae slot, Qwen2.5-VL-3B,
small LFM/3B models):

```
HUGPY_AGENT_BRAINS=Qwen~Qwen3-Coder-Next-GGUF,Qwen~Qwen2.5-3B-Instruct-GGUF
```

Coder stays the quality brain; the 3B instruct is the pilot light — star it on
computron (8 GB holds a 3B GGUF comfortably beside the VL model) so ae's slot
churn can never leave case runs brainless. (VL-3B is deliberately NOT on the
ladder: it is a vision model holding the vision seat.)

The sentinel watches the floor itself: `pilot_light_not_resident` (severity
warn) fires when the pilot light (from `HUGPY_SENTINEL_PILOT_LIGHT`, else the
last `HUGPY_AGENT_BRAINS` entry) has no healthy allocation on any worker, and
the case evidence carries the exact boot-prewarm POST to fix it.

## Knobs (env; all read by `hugpy_ops/sentinel/settings.py`)

| var | default | meaning |
|---|---|---|
| `HUGPY_SENTINEL_CENTRAL` | `central_base_url()` → `http://127.0.0.1:7002` | central base |
| `HUGPY_SENTINEL_DIR` | `<HUGPY_HOME>/state/sentinel` (`hugpy_platform.hugpy_state_dir()`) | case db, case dirs, `cases.md`, history |
| `HUGPY_SENTINEL_REMEDIES` | unset (OFF) | THE remedy gate |
| `HUGPY_SENTINEL_DOWNLOADS` | unset (ON) | k97 downloads gate — `enqueue_download` only; `0` disables |
| `HUGPY_PROVISION_FLOOR_GB` | 500 | provisioner free-space floor on the destination volume |
| `HUGPY_SENTINEL_STALLED_GRACE_S` | 120 | stall must persist this long past central's 90 s label |
| `HUGPY_SENTINEL_HARD_FAIL_STREAK` | 3 | consecutive `hard_pass=false` to case |
| `HUGPY_SENTINEL_AGENT` | `hugpy-agent` | absolute path in the unit |
| `HUGPY_SENTINEL_AGENT_TIMEOUT_S` | 900 | one case run's wall clock |
| `HUGPY_SENTINEL_AGENT_MAX_STEPS` | 20 | agent loop bound |
| `HUGPY_AGENT_BRAINS` | unset | k96 brain ladder, inherited by case agents (csv, best-first; last = pilot light) |
| `HUGPY_SENTINEL_PILOT_LIGHT` | last `HUGPY_AGENT_BRAINS` entry | override for the pilot-light-not-resident check; empty ladder + unset = check off |

A worker `version_ok: false` case additionally carries the `hugpy-fleet` /
`hugpy-server` distribution versions installed beside the sentinel
(`hugpy_ops.versions`, resolved lazily through `importlib.metadata`).

Tests: `python -m pytest tests/test_sentinel.py tests/test_sentinel_http.py -q`
in `py/operations/hugpy_ops` and `python -m unittest test_case_cli` in
`py/inference/hugpy_agent/tests` (the pinned agent-side profile).
