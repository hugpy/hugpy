# Model curation & reliability daemons — mechanics

**Scope:** `discovery_dossier/`, `review/`, `sentinel/`, `chaos/`,
`fleet_doctrine/`, `downloader/` (under `src/abstract_hugpy_dev/src/abstract_hugpy_dev/`
unless noted); also `provisioner.py`/`model_sync.py` (package root) and, noted
only, `src/evaluations/studio-aptitude/`.
**One-liner:** discovers and grades candidate HF models into the fleet (dossier
+ review + judge), runs bounded reliability daemons that probe and exercise the
live fleet (sentinel, chaos, fleet_doctrine), and fetches model weights off the
request-serving path (downloader, + the adjacent provisioner/model_sync).
**Owner process(es):** no single owner. `review/`+`discovery_dossier/` run as a
worker-box systemd timer (pip wheel) *and* as a central-HTTP-triggered
background thread (live `src`) — §9. `sentinel/` is its own systemd timer, a
pure HTTP client of central. `chaos/` is a manual operator CLI, never a daemon.
`fleet_doctrine/` is a library, self-assessed by `worker_agent`, served by a
central Flask blueprint. `downloader/` is its own long-lived daemon
(`hugpy-downloader-dev.service`); central's API only enqueues into it.

## 1. Purpose & responsibilities

- **Curate** — `discovery_dossier/`+`review/` turn a candidate HF repo into a
  graded, evidence-backed dossier and an adopt/trial/reject verdict, cheap
  (metadata screen, whole pool) before expensive (download+GPU load+LLM judge,
  capped per run).
- **Probe** — `sentinel/` watches central's read surfaces for bound-exceeded
  conditions (stalled jobs, offline workers, lost capabilities, hard-fail
  streaks, missing weights), opens deduplicated cases, spawns one bounded
  diagnosis agent per new case.
- **Exercise** — `chaos/` randomly (or, via `sweep`, deterministically) fires
  real generations across model × worker × alloc-mode × ctx% to compare
  predicted vs. measured placement, for a sibling learner (out of scope).
- **Diagnose environment** — `fleet_doctrine/` diffs a worker's self-reported
  environment against a versioned "doctrine"; an unproven case is always
  `unknown`, never coerced into `ok`/`blocked`.
- **Fetch** — `downloader/` is the one process that spends network+disk time on
  weights; everything else here only enqueues into its queue or reads its
  heartbeat.
- All six mutate fleet state only through explicit, reversible, gated paths
  (sentinel's remedy whitelist, chaos's snapshot/apply/restore) — one blur in §8 (#5).

## 2. Key modules (file → responsibility)

| subsystem | file | responsibility |
|---|---|---|
| discovery_dossier | `dossier.py` | typed shapes — `ModelDossier` + 8 section dataclasses |
| discovery_dossier | `build.py` | orchestrator: research→identity→specialization→weights→community→trust→trial→verdict |
| discovery_dossier | `fetch.py` | the one outbound HTTP helper — never raises, 20h disk-cached |
| discovery_dossier | `cards.py` | README → `CardDigest`, `Specialization` scoring |
| discovery_dossier | `research.py` | README + arXiv papers + LLM `research_notes` |
| discovery_dossier | `community.py` | Reddit/HN/HF-discussions/YouTube mentions, `heat`, LLM claim extraction |
| discovery_dossier | `radar.py` | re-reads the fetch cache for models no card is tracking ("gem radar") |
| discovery_dossier | `weights.py` | per-quant VRAM ladder from metadata only (no tensor bytes touched) |
| discovery_dossier | `screening.py` | k120 screen-time knobs (specializations/licenses), network-free |
| discovery_dossier | `trial.py` | runs the candidate through the stationary battery, scores vs. incumbent |
| discovery_dossier | `verdicts.py` | evidence-gated adopt/trial/reject (+ `llm.py`, the one text-model seam) |
| discovery_dossier | `store.py` | file-per-(criteria, hub_id) persistence + compact `summary()` |
| review | `pipeline.py` | orchestrator: screen pool → download+smoke+judge+dossier survivors, capped |
| review | `screen.py` | stage 1 — metadata-only fit/capability score, no weights fetched |
| review | `criteria.py` | `ReviewCriteria` — the saved, re-runnable question ("a card") |
| review | `smoke.py` | stage 2 — subprocess-loads the GGUF with llama_cpp, 4 fixed probes |
| review | `judge.py` | asks a live fleet model (`/v1/chat/completions`) for a verdict |
| review | `store.py`+`push.py` | worker-local SQLite (on-box record) + best-effort hand-off to central's DB |
| sentinel | `checks.py` | detection — pure functions over central's 3 HTTP surfaces + local history |
| sentinel | `cases.py` | SQLite case store, one OPEN case per fingerprint (structural dedupe) |
| sentinel | `runner.py` | one pass: detect → open/touch cases → spawn diagnosis agent, or fast-path |
| sentinel | `remedies.py` | the whitelist — typed, reversible, gated, prod-worker-excluded |
| chaos | `runner.py` | random exerciser: draw → price → apply → fire → measure → restore |
| chaos | `sweep.py` | k7 deterministic offload speed-cliff sweep (separate subcommand) |
| chaos | `assortment.py` | enumerate the live cube, draw seeded combos, feasibility |
| chaos | `alloc.py` | snapshot/apply/restore a model's per-worker spill, verified |
| chaos | `client.py` | the one HTTP seam onto central (stdlib-only, injectable) |
| chaos | `observe.py` | build the predicted (pre-fire) / measured (post-fire) observation halves |
| chaos | `schema.py` | `chaos-obs/1` — the observation contract shared with the (out-of-scope) learner |
| fleet_doctrine | `doctrine.py` | the versioned reference document — observed + declared requirements |
| fleet_doctrine | `doctor.py` | pure diff: worker report vs. doctrine → severity-classified findings |
| downloader | `daemon.py` | long-lived process — claims queued jobs, adopts stale ones, runs them |
| downloader | `engine.py` | transfer lifecycle — spawn (never fork), stall-killer, resume, terminal classification |
| downloader | `queue.py` | everything the API may do — enqueue/cancel/retry/list, never starts a transfer |
| downloader | `presence.py` | the daemon's heartbeat file so a queued-but-orphaned job says so |
| (root) | `provisioner.py` | detects declared-but-missing weights (comfy/studio/tasks registries), enqueues via `downloader.queue` — no engine of its own |
| (root) | `model_sync.py` | pulls whole model **directories** box→box from central's archive/file routes via `worker_agent.provision` — unrelated transport to `downloader/` |

`sentinel/settings.py` (env thresholds) and each subsystem's `__main__.py`
(CLI dispatch) are omitted above as thin — see §3 for their entry points.

## 3. Entry points

- **`discovery_dossier/`** — no `__main__.py`; library-only, invoked from
  `review/pipeline.py:_dossier()` (pipeline.py:90-129) and `_radar()`
  (pipeline.py:325-356, gated on `crit.radar`).
- **`review/`** — three triggers converge on `pipeline.py`: nightly systemd
  timer `hugpy-review@<criteria>.timer` (03:20 local, ±45m jitter) → oneshot
  `.service` (worker box, pip wheel) → `python -m abstract_hugpy_dev.review run
  <criteria>`; ad-hoc CLI `hugpy-review screen|review|run|criteria|reports|push`
  (`__main__.py:54-121`); `POST /llm/review/run` on central
  (`review_routes.py:83-115`, background thread, live `src`).
- **`sentinel/`** — `hugpy-sentinel.timer` (`OnBootSec=2m`,
  `OnUnitActiveSec=10m`) → oneshot `.service` → `python -m
  abstract_hugpy_dev.sentinel run-once`; plus `status`/`record-scorecard` CLI
  (`sentinel/__main__.py:61-89`).
- **`chaos/`** — manual only: `python -m abstract_hugpy_dev.chaos [--dry-run]
  [--rounds N] ...` or `... chaos sweep [--plan] [--top-n N] ...`
  (`chaos/runner.py:291-297` dispatches the subcommand). No systemd unit
  references `chaos` anywhere under `deploy/`.
- **`fleet_doctrine/`** — library only. Central: `GET
  /fleet/doctrine/status[/<worker>]`, `GET /fleet/doctrine/versions`
  (`fleet_doctrine_routes.py`). Worker self-assessment:
  `worker_agent/agent.py:11276-11280` (heartbeat),
  `worker_agent/install.py:236-237` (post-install check).
- **`downloader/`** — its own daemon, `hugpy-downloader-dev.service`
  (`Type=simple`, `Restart=always`) → `daemon.py:main`. The API never starts a
  transfer: `queue.py:enqueue_download()` (65-92) is the only write surface,
  called from `POST /llm/repos/download`/`POST /models/<key>/download`, from
  `provisioner.enqueue()` (provisioner.py:683-725), and from sentinel's
  `weight_missing` fast path.

## 4. Data flow (the spine)

**A. Discovery → admission** (`review/` + `discovery_dossier/`)
1. `pipeline.run(crit)` pulls candidates via `screen.search_candidates` (HF hub
   search, gguf-filtered — screen.py:455-473).
2. Stage 1, whole pool: `screen()` (screen.py:276-420) scores fit
   (quant/VRAM/context) and capability (trust/downloads/tags) from metadata
   only — no weights fetched.
3. Survivors sorted by score, capped at `crit.max_downloads_per_run` (default
   2 — pipeline.py:293-300).
4. Per survivor: `_download()` (pipeline.py:68-87) fetches the quant via
   `imports.apis.download_models.download_one` **directly** (§8 #1);
   `smoke.smoke_test()` (smoke.py:176-231) loads it in a llama_cpp subprocess,
   4 fixed probes; `judge()` (judge.py:102-164) asks a live fleet model.
5. `_dossier()` (pipeline.py:90-129) → `discovery_dossier.build.build_dossier`:
   reused screen → research (README+arXiv+LLM notes) → identity →
   specialization → weights ladder → community → trust → trial (stationary
   battery, depth-gated) → `verdicts.decide()`.
6. `review.store.record()` persists locally (worker SQLite);
   `discovery_dossier.store.save()` persists the dossier file.
7. `_push()` (pipeline.py:359-372) best-effort POSTs to
   `{REVIEW_CENTRAL_URL}/llm/review/ingest` (push.py:48); central's
   `review_ingest()` (review_routes.py:189-256) upserts idempotently, keyed on
   `(host, run_id[, hub_id, stage])` — the console reads that copy back via
   `GET /llm/review/results|dossiers|dossier|radar`.

**B. Sentinel: detect → case → diagnose/remedy**
1. `checks.detect()` (checks.py:299-318): 3 GETs (`/llm/jobs?live=0`,
   `/llm/workers`, `/oracle/capabilities`) + local scorecard history → 6 rule
   functions → `Anomaly` list.
2. `runner.run_once()` (runner.py:250-283): each anomaly through
   `CaseStore.open_or_touch()` (cases.py:92-119) — a partial UNIQUE index on
   `fingerprint WHERE state != 'closed'` makes dedupe a DB guarantee.
3. New non-`weight_missing` cases: `spawn_agent_for_case()` (runner.py:101-157)
   runs `hugpy-agent case <brief> --case-dir <dir>` bounded (900s default,
   document-only policy), parses the JSON report → `documented`/`escalated`.
4. `weight_missing` cases fast-path (runner.py:174-233):
   `remedies.execute(enqueue_download)` POSTs `{central}/llm/repos/download`
   directly — no agent spawned.
5. Other remedies (unload/relaunch, chat cancel) stay behind
   `HUGPY_SENTINEL_REMEDIES` (default OFF); downloads have their own gate
   (`HUGPY_SENTINEL_DOWNLOADS`, default ON); worker `ae` is structurally
   excluded from remedies (remedies.py:21,114,137-140).

**C. Chaos: random exercise / deterministic sweep**
1. `draw_combo()` (assortment.py:195-230) picks a servable model, an
   already-assigned worker ("card"), a framework-gated alloc mode, a ctx% —
   seeded, reproducible.
2. `observe.build_predicted()` (observe.py:20-77) prices the combo via
   `/models/<key>/meta` before firing; infeasible combos are skipped, never
   fired (runner.py:207-211).
3. `alloc.snapshot()`→`alloc.apply()` (alloc.py:28-62, the **only** mutating
   call, operator-gated `/assign`) → `client.chat_stream()` fires a small real
   generation over the console's own public path → `observe.build_measured()`
   reads the served worker's row back.
4. `alloc.restore()` (65-89) writes the prior spill back, verifies
   byte-identical — every trial only touches an already-assigned pair, so
   restore is a write-back, never an unassign.
5. `sweep` (sweep.py): ranks top-N used GGUF models (`rank_targets`,
   180-266), forces each COLD between grid points (`/unload`) so a fresh
   `gpu_mem_gib` budget takes effect, times generations per VRAM share
   (`run_point`, 422-550), detects the tok/s cliff (`detect_cliff`, 143-177).

**D. fleet_doctrine: assess**
1. `doctrine.snapshot(report)` (422-518) freezes a reference worker's report
   into a versioned `Doctrine` (observed + declared `KNOWN_REQUIREMENTS`, 179-291).
2. `doctor.assess(report, doctrine)` (386-447) diffs any worker's report
   against it — pure, no I/O — `ok/drift/missing/pin_violation/unknown/
   profile_absent`; only a **proven** missing/pin-violated blocker is
   `blocking` (78-86).
3. Consumed two ways: `/fleet/doctrine/status` reads **heartbeats only**
   (cheap); `?live=1` pulls `/ops/environment` from one worker and assesses on
   the spot (fleet_doctrine_routes.py:145-189).

**E. Downloader: claim → transfer → terminal**
1. The API (or provisioner/sentinel) calls `queue.enqueue_download()` → one
   `pending` job row in the comms mirror ([[comms-index]]) — no transfer starts here.
2. `daemon.py`'s poll loop `claim_next()`s it (compare-and-set), becomes owner;
   `engine.run_download_job()` (409-547) spawns (never forks) a child calling
   `download_one()`.
3. Stall killer: no new bytes for `STALL_SECONDS` (180s, past a
   `STARTUP_GRACE_SECONDS` of 600s — 58-66) kills the process group; up to
   `MAX_ATTEMPTS` (4) with backoff; HF partials + staging dir survive, so a
   resume continues rather than refetches.
4. Terminal failures (gated repo, bad auth, 404, no space) classify once and
   fail **fast** rather than retry 4× (`classify_download_error`, 132-159).
5. Daemon restart mid-transfer: `adopt_stale()` re-queues rows claimed by a
   dead owner; the next daemon adopts the staging dir by name and resumes
   (daemon.py:104-111).

## 5. State, persistence & invariants

- **discovery_dossier/**: one JSON file per `(criteria, hub_id)` at
  `<DEFAULT_ROOT>/review/dossiers/<criteria>/<org__repo>.json` (atomic
  tmp+`os.replace`); separate `_radar.json`; 20h disk cache at
  `~/.cache/hugpy/discovery-dossier/` keyed by `sha256(url)`. `SCHEMA_VERSION="dossier/1"`.
- **review/**: worker-local SQLite (`REVIEW_DB`) is the on-box record;
  central's ingested copy is the **source of truth** the console reads —
  push is one-way, documented at store.py:1-28.
- **sentinel/**: SQLite case store (`$HUGPY_SENTINEL_DIR/cases.db`, partial-
  unique dedupe index); `cases.md` ledger (append-only); `history.jsonl`
  (capability/scorecard history, pruned past 256KB).
- **chaos/**: append-only `observations.jsonl` (`chaos-obs/1`) + per-run
  manifest JSON under `/mnt/llm_storage/comms/chaos/`. No DB; never touches
  central's DB directly — HTTP only, and only `/assign`+`/unload` mutate
  anything, always restored.
- **fleet_doctrine/**: versioned JSON, `hugpy-worker-doctrine-<version>.json`,
  resolved env-override → repo-relative `deploy/doctrine/` →
  `~/hugpy-worker/doctrine` → `/etc/hugpy/doctrine` (526-556).
- **downloader/**: no DB of its own — rides the existing comms job
  store/mirror (`HUGPY_COMMS_DB`); a heartbeat **file** (deliberately not a job
  row) next to it.
- Cross-cutting invariant: see §1's last bullet — one blur noted in §8 #5.

## 6. Cross-subsystem edges

- `review/`→`discovery_dossier/`: `_dossier()`→`build_dossier()`;
  `_radar()`→`radar.scan()`; `screen.py:402-403`→`screening.extra_reasons()` —
  all wrapped so a broken dossier package never stops a review.
- `discovery_dossier/`→`review/`: reaches into `review/screen.py`'s private
  helpers (`_config`, `_repo_info`, `_hf_api`, `kv_cache_bytes`) rather than
  duplicating them — a rename there breaks this silently at runtime.
- `discovery_dossier/`→`fleet_doctrine/`: only `verdicts.py:162-163`
  (`doctrine_note()`), reading `fleet_doctrine.latest()`.
- `discovery_dossier/`→`oracle/`: heaviest external dependency — routing
  matrix, capability catalog, benchmark/runtime, `stationary_scenario` — see [[video-oracle]].
- `sentinel/`→`provisioner.py`→`downloader/`: `checks.default_weight_wants()`
  (255-264) calls `provisioner.wants()`; the weight_missing fast path POSTs
  `{central}/llm/repos/download` — the **same** route a human "add model" click
  hits — landing on `downloader.queue.enqueue_download()`. `provisioner.enqueue()`
  (683-725) calls that function directly when run by hand (`--apply`).
- `chaos/`→`managers/alloc_modes.py`: `schema.py:120`/`assortment.py:28` import
  `ALLOC_MODES`/`NONGGUF_ALLOWED_MODES`/`resolve_alloc_mode` — one shared
  vocabulary with [[serving-core]], never redefined locally.
- `fleet_doctrine/` ← [[worker-fleet]] (`agent.py`, `install.py`),
  `oracle/probes.py`, `discovery_dossier/verdicts.py`, central's
  `fleet_doctrine_routes.py` — small, library-only, consumed widely (§3).
- `downloader/` ← flask routes (`llm_storage_routes.py`), `provisioner.py`,
  `sentinel/remedies.py` — always through `queue.enqueue_download`, never a
  direct `engine` call from outside `daemon.py`.
- `sentinel/`, `chaos/`, and two of `review/`'s three entry points only ever
  talk to central over plain HTTP — none import central's Flask internals;
  central is addressed the way an external operator would (checks.py:1-8,
  client.py:1-11 both say this explicitly).

## 7. Key contracts / types

- `discovery_dossier.dossier.ModelDossier` (534-550): `hub_id, criteria,
  schema, generated_at, identity, specialization, weights, trust, research,
  community, trial, verdict, screen, sources, notes`. `Verdict.verdict ∈
  {adopt, trial, reject, screened-only}` (507-526).
- `review.screen.ScreenResult` / `review.smoke.SmokeResult` /
  `review.criteria.ReviewCriteria` (`vram_bytes`, `target_context`,
  `allowed_quants`, `max_downloads_per_run`, `trial_depth`, … — criteria.py:37-153)
  / `review.pipeline.Review` (`hub_id, criteria, stage, screen, smoke,
  judgement, dossier`).
- `sentinel.cases.Anomaly{fingerprint, kind, severity, evidence}` / `Case`
  (states `open→agent_running→documented|escalated→remedied|closed`) /
  `sentinel.remedies.Remedy{name, method, url_template, applies_to,
  reversible, gate}` (remedies.py:43-57).
- `chaos.schema` — `chaos-obs/1`: top-level `{combo, predicted, measured,
  restore}` + optional `sweep`; `blank_observation()`/`validate_observation()`
  (192-300) enforce completeness on the earliest skip path.
- `fleet_doctrine.doctrine.{Requirement, DoctrineEntry, Doctrine}` (severity ∈
  `blocker/warn/info`) / `fleet_doctrine.doctor.{Finding, DoctrineReport}`
  (`Finding.blocking` true only for a proven missing/pin-violated blocker).
- `downloader` job kind `"download"` (`DOWNLOAD_KIND`), riding
  `comms.jobs.Job` ([[comms-index]]); `provisioner.Want{registry, name, reason,
  dest, hub_id, filename, ...}` — `resolved` iff `hub_id` is provable, never
  guessed (87-116).

## 8. Gotchas, tech-debt & review findings

1. ⚠ `review/pipeline.py:68-87` (`_download`) calls `download_one()`
   **synchronously in-process** — none of `downloader/engine.py`'s
   stall-killer/resume/backoff (engine.py:58-66,409-547, `MAX_ATTEMPTS=4`)
   applies. The nightly timer runs unattended up to 4h
   (`TimeoutStartSec=4h`); one wedged HF connection can stall the whole run
   with zero self-healing — the exact class `downloader/` was built to fix.
2. ⚠ `discovery_dossier/dossier.py:476-481` — `TrialEvidence.has_evidence` is
   true merely from `load.ok`. Under the **default** `trial_depth="load-test"`
   (criteria.py:94), `run_trial()` returns with empty `scores`/`samples`
   (trial.py:477-487), yet `rule_verdict()` (verdicts.py:196-202,265) still
   emits a confident adopt/trial/reject — contradicting the module's own "NO
   EVIDENCE, NO VERDICT" rule (verdicts.py:1-22).
3. ⚠ `oracle/interim_ledger.py:1544,1561` reads a dossier's `verdict`
   expecting a bare string, but a persisted dossier nests `Verdict.to_dict()`
   (dossier.py:587) there — likely stores repr noise instead of `"adopt"`.
   The same lookup's `gap`/`"screening"` keys don't exist in
   `ModelDossier.to_dict()`, so a blocked dossier never reports `STATUS_GAP`
   to the fleet ledger.
4. △ `review/judge.py:25` hardcodes the **retired** hugpy VM's IP
   (`192.168.1.250` — see top-level `README.md`'s "Retired" section) as the
   first fallback probe, tried with a 30s timeout (judge.py:75) before
   loopback — up to 30s wasted per `judge()` call if unreachable-not-refused.
5. △ `POST /llm/review/run` (review_routes.py:83-115) runs the same
   synchronous download/subprocess-smoke path as finding 1, in a background
   thread **inside central's own gunicorn process** (live `src`) — the
   console's "Run now" button reintroduces the exact problem shape
   `downloader/` eliminated; the nightly-timer path (a worker box) does not.
6. △ `chaos/client.py:20-24` hardcodes `DEFAULT_BASE` and reads only
   `HUGPY_BASE_URL` (runner.py:305, sweep.py:822), bypassing the shared
   `central.py:central_base_url()` alias resolver `sentinel/settings.py:73-78`
   uses — `HUGPY_CENTRAL`/`HUGPY_URL`/`WORKER_CENTRAL_URL` are silently
   ignored by chaos though every other arm honours them.
7. △ `chaos/sweep.py:848` — the `--markdown` default path is hardcoded to the
   date of the original k7 investigation (`OFFLOAD-CLIFF-2026-07-18.md`);
   every sweep run without an explicit `--markdown` overwrites that same
   fixed-date file regardless of when it runs.
8. ℹ "card" means three unrelated things across this scope's own files: a
   saved `ReviewCriteria` (criteria.py:81-111), a GPU-bearing worker
   (chaos/schema.py, assortment.py), and README-derived `CardDigest` scoring
   (`discovery_dossier/cards.py`) — track the sense per file. Relatedly,
   `rv.judgement` (pre-k120 `review/judge.py`) and `dossier.verdict`
   (pipeline.py:239-247) are two independent verdict fields on one `Review`
   row that nothing reconciles.
9. ℹ `src/evaluations/studio-aptitude/` (noted, not in scope) writes grades
   **only** to local dated files (report.py:1-6, run.py:125-149,511-632) — no
   push/POST/sqlite/central write path found anywhere in that directory.
   Relevant to future metrics-grades work: `review/`'s working
   push→`/llm/review/ingest` pattern is the template this harness lacks.

## 9. Deploy/run boundary

None of `review/`'s nightly-timer path, `sentinel/`, or `downloader/` run on
central's "runs `src` now" checkout — each is a separately deployed
**pip-installed wheel** on its own path/box/venv:

- `review/` (timer path): worker box, `User=solcatcher`,
  `.../hugpy-worker/venv` — live only after that venv's package is upgraded;
  the next oneshot `hugpy-review@<criteria>.service` picks it up.
- `sentinel/`: `.../station/dev/abstract_hugpy_dev/venv` — same boundary;
  nothing to bounce until its next 10-min tick.
- `downloader/`: same venv root, `Restart=always` — **does** need an explicit
  `systemctl restart hugpy-downloader-dev`; central's `7002` restart does not
  touch it, despite sharing `HUGPY_COMMS_DB`.
- `chaos/`: whatever interpreter an operator invokes it from — no fixed boundary.

Exception: `POST /llm/review/run` and every `fleet_doctrine` read through
central's blueprint (`fleet_doctrine_routes.py`) **do** run inside central's
live-`src` process — same code, live-on-restart via this path but
pip-wheel-boundary via the worker/dev-box timers (§3).

`discovery_dossier/` ships inside whichever wheel/checkout imports it — no
independent deploy story. `fleet_doctrine/`'s doctrine JSON files are
**data**, not code — a new version is placed by hand or a separate step,
never by `pip install -U` alone.
