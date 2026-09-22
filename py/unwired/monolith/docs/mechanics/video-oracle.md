<!-- MECHANICS DOC — see _TEMPLATE.md for the shape this follows. -->

# Video-Oracle — mechanics

**Scope (paths this doc covers):** `video_intel/` (schemas, `media_bus.py`, `media_store.py`,
`runners/`, `studio/`, `reservation/`), `oracle/` (`pipeline.py`/`pipelines.py`, `dag_runtime.py`,
`router.py`/`routing_matrix.py`, `selection.py`, `validator.py`, `steward.py`, `prompt_compiler.py`,
`screenplay.py`, `storyboard.py`, `production.py`, `postproduction.py`, `spatial*.py`, `scorecard.py`,
`repair.py`/`repair_controller.py`). Paths relative to
`src/abstract_hugpy_dev/src/abstract_hugpy_dev/` unless noted.
**One-liner:** the headless video-intelligence backbone (durable job bus + frozen schemas + pure
runners + GPU-worker studio rendering) plus the oracle eliminate-then-rank planning system that
turns a creative request into a locked plan → screenplay → storyboard → production → postproduction.
**Owner process(es):** central `7002_hugpy_api` (Flask blueprints `oracle_bp`, `script_first_bp`,
`video_bp`, plus thin adapters in `video_coordination.py`/`video_assist_media.py`, and reads from
`comms_routes.py`/`worker_routes.py`); the GPU worker process (`worker_agent/studio_render.py`'s own
Flask mount); `hugpy-steward.timer` (systemd, 15-min self-check against central, **not** installed by
default). No `hugpy` CLI subcommand (`serve|worker|bot|keeper|chat|install-engine|install-deps|
reclassify-images`) reaches this subsystem directly.

## 1. Purpose & responsibilities
- `video_intel/`: validate creative-job specs, run them durably through a sqlite job bus
  (`media_bus.py`) with cross-process single-writer semantics, dispatch to pure `spec -> JobResult`
  runners (ffmpeg/diffusers/studio/identity/tts/mlt), and reserve/release GPU VRAM around heavy renders.
- `oracle/`: decide *which model* serves a capability (eliminate-then-rank: `router.py`/`selection.py`),
  author a locked, content-addressed creative plan (`production.py`/`screenplay.py`/`storyboard.py`/
  `postproduction.py`), durably execute/repair a DAG of that work (`dag_runtime.py`/
  `repair_controller.py`), and audit its own selection quality on a schedule (`steward.py`).
- Deliberately NOT: `video_intel` never decides which model renders a shot (oracle's job); oracle's
  creative-authoring modules never touch ffmpeg/GPU/disk — they hand a locked spec to `video_intel`'s
  bus/runners.
- Deliberately NOT (yet): a docstring-promised `ORACLE_PIPELINE` (a `pipeline.py`-framework mirror of
  `selection.select`) does not exist in code — only the model-**name** resolution pipeline is built.

## 2. Key modules

**`video_intel/`**

| file | responsibility |
|---|---|
| `media_schema.py:21 MediaRef` / `:51 make_media_ref` | immutable asset handle; metadata resolved once at ingest |
| `media_store.py:161 ingest()` | ffprobe a file once, classify kind, mint a `MediaRef`, jailed under `UPLOADS_HOME`/`DEFAULT_ROOT` |
| `crop_schema.py`, `frame_schema.py`, `audio_schema.py`, `scene_schema.py`, `movie_schema.py` | frozen specs + validating `make_*` factories for crop/frame-extract/audio-extract/scene/movie jobs |
| `identity_video_extract_schema.py`, `identity_from_video_schema.py` | frozen specs for the two char360 RELAY jobs (central has no GPU; forwarded to `IDENTITY_RENDER_URL`) |
| `job_schema.py:41 JOB_REGISTRY` | job `name` → `JobSpec(spec_type, runner_key, queue, timeout_s)` |
| `media_bus.py` | durable sqlite(WAL) job bus: enqueue/claim/run/cancel/archive/list, migrations, orphan reap |
| `runners/__init__.py:67 DISPATCH` | `(framework,task)` → pure runner callable |
| `runners/*.py` | pure `spec -> JobResult` runners: ffmpeg primitives, imagegen/scene/movie orchestrators, `studio_i2v.py` (bus↔worker adapter), identity relays, tts, mlt, oracle-performance relay |
| `studio/registry.py` | 3 global registries — `MODEL_REGISTRY`/`RUNNER_REGISTRY`/`CAPABILITY_TASKS` — validated-total at import |
| `studio/router.py:251 CapabilityRouter` | picks a studio GPU backend (`studio/runners/*.py`) for a capability |
| `studio/produce.py:158 produce_clip()` | the actual GPU render entry the worker calls |
| `studio/errors.py` | `Result`/`Ok`/`Err`/`ErrorCode` errors-as-data vocabulary; SPEC-before-SOURCE-before-DEPS/GPU/weights check ordering |
| `reservation/templates.py` | per-task VRAM budget templates (built-in estimate + measured overlay) |
| `reservation/engine.py` `acquire`/`release` | pre-claims/releases GPU VRAM around a heavy run via worker eviction verbs |
| `reservation/registry.py` | sqlite-backed claim ledger, leased/TTL, self-expiring |

**`oracle/`**

| file | responsibility |
|---|---|
| `pipeline.py` | generic stdlib eliminate-then-rank framework (`Candidate`/`Stage`/`StageRegistry`/`run_pipeline`) |
| `pipelines.py:100 NAME_PIPELINE` | the one concrete pipeline built on it: fuzzy model-NAME → canonical key |
| `router.py:269 resolve_route()` | live per-request goal→capability→model dispatcher behind `POST /oracle/route` |
| `routing_matrix.py:345 derive_matrix()` | pure, offline, benchmark-derived leaderboard per operation (not itself a dispatcher) |
| `selection.py:365 select()` | the real 9-step eliminate-then-rank model selector |
| `validator.py:495 validate()` | pure static `PlanGraph` checker (11 checks); returns a report, never raises |
| `dag_runtime.py:858 DagRuntime` | durable sqlite(WAL)-journaled `PlanGraph` executor — the repair/benchmark path, not the creative-authoring path |
| `repair.py:79 attempt_repair()` | single-route bounded-retry policy |
| `repair_controller.py:243 RepairController` | DAG-scale repair/replan built on `DagRuntime` |
| `steward.py:153 Steward` | self-audits selection calibration/failure-streaks/starvation/gap-rate/matrix-staleness; bounded rebalance |
| `production.py` | `GenerationSnapshot`/`ContinuityBible`/`ShotPlan`/`ProductionLock` — the locked artifact stack |
| `screenplay.py` | `PlotSpec`/`Screenplay`/`build_continuity`/`build_shot_plan`/`lock_production` |
| `prompt_compiler.py:551 compile_context()` | per-segment prompt context plan + length + multiplicity |
| `storyboard.py:594 render_storyboards()` | cheap pre-production stills, judged, before a real shoot |
| `postproduction.py:1686 build_postproduction_plan()`, `:2163 evaluate_take()` | EDL/sound/music cues + take-vs-spec judging |
| `spatial.py` / `spatial_sources.py` / `spatial_eval.py` | contract / producers / evaluators for spatially-aware shots |
| `scorecard.py:157 build_technical_scorecard()` | deterministic pass/fail card every route response carries |

## 3. Entry points
- HTTP, central (blueprint-relative paths; final mount prefix is [[api-routes]]'s concern —
  `video_bp` is confirmed mounted at `/api` in `flask_app/wsgi_app.py:227`):
  - `oracle_bp` (`flask_app/app/routes/oracle_routes.py:78`): `GET /oracle/capabilities` (:81),
    `POST /oracle/route` (:274, single-capability ad hoc resolve), `GET|POST /oracle/steward` (:553),
    `POST /oracle/selection` (:586, explain-only), `GET|POST /oracle/producers` (:625),
    `GET /oracle/ledger/summary` (:665).
  - `script_first_bp` (`flask_app/app/routes/script_first_routes.py:29`, `BASE="/video/script"`) —
    the real creative-pipeline API: `POST .../runs` create, `POST|PUT .../plot`,
    `POST|PUT .../screenplay`, `PUT .../audio_master`, `POST .../preproduction`, `POST .../lock`,
    `POST .../revise`, `POST|GET .../segments`, `POST .../segments/<id>/generate`, `POST .../promote`.
  - `video_bp` (`video_routes.py`, mounted `/api`): ~15 `POST /video/jobs/*` + `/video/studio/*`
    routes, each building a spec then calling `media_bus.enqueue` via `_video_enqueue()` (`:240-247`).
  - `video_coordination.py`/`video_assist_media.py`: thin adapters onto `video_intel/prompt_coordination.py`
    (outside this doc's reviewed file set).
  - `comms_routes.py:179` reads `video_intel.media_bus` to mirror media jobs into `GET /llm/jobs`.
  - `worker_routes.py:880,888,1521` reads `video_intel.reservation.{registry,templates}` for
    GPU-reservation introspection.
- Worker HTTP (`worker_agent/studio_render.py`, `register_studio_routes()`): `POST /studio/render`,
  `GET /studio/render/<job_id>`, `POST /studio/cancel/<job_id>` — where `produce_clip` actually runs.
- systemd: `deploy/hugpy-steward.timer` (15-min cadence) → `curl -X POST $HUGPY_API/api/oracle/steward`.

## 4. Data flow (the spine)

**Flow A — video_intel job lifecycle** (every job rides this):
1. A route builds a validated spec via a `make_*` factory (e.g. `make_crop`, `crop_schema.py:44`) —
   a bad request 400s here, never reaches the bus.
2. `media_bus.enqueue(name, spec, ...)` (`media_bus.py:1001`) looks up `JOB_REGISTRY[name]`,
   serializes the spec, `INSERT ... status='queued'` (:1023-1032), returns `job_id`.
3. `claim(worker_token)` (`:1039`): `BEGIN IMMEDIATE`, `SELECT ... WHERE status='queued' ORDER BY
   created LIMIT 1`, then `UPDATE ... SET status='claimed', claim_token=? WHERE job_id=? AND
   status='queued'` (:1055-1059) — the `WHERE status='queued'` on the UPDATE is what lets exactly one
   claimant win across processes.
4. `run_claimed(job_id, worker_token)` (`:1069`) re-checks `claim_token` ownership, flips to
   `running` (gated `WHERE claim_token=? AND status='claimed'`, :1092-1096), optionally pre-claims GPU
   VRAM via `_acquire_reservation` (:1127 → `reservation/engine.py acquire`), dispatches
   `DISPATCH[job_spec.runner_key](spec, job_id)` (:1132) *outside* the DB connection, converts any
   raise into `JobResult(ok=False, JobError('internal',...))` at the one sanctioned catch point
   (:1138), releases the reservation in `finally` (:1153), then writes the terminal row once
   (`UPDATE ... WHERE job_id=? AND claim_token=?`, :1168-1172).
5. For a **studio** job the runner (`runners/studio_i2v.py`) does not render locally: it POSTs the
   spec to the GPU worker's `/studio/render`, polls `/studio/render/<job_id>`, and on completion
   ingests the clip from the shared content-addressed path both boxes mount — central's
   `media_jobs.db` stays the single writer throughout.

**Flow B — oracle creative pipeline** (a `script_first` run, `oracle/script_first.py`, out of file
scope but the confirmed orchestration shell):
1. `POST .../runs` creates a run.
2. `.../plot` → `author_plot`/`PlotSpec` (Stage 5, `screenplay.py:321`).
3. `.../screenplay` → `author_screenplay`/`Screenplay` (Stage 6, `:779`).
4. `.../preproduction` → `build_continuity`/`ContinuityBible` (Stage 7, `:1119`) + `build_shot_plan`/
   `ShotPlan` (Stage 9, `:1546`) + `storyboard.render_storyboards()` (`storyboard.py:594`) renders
   cheap stills per shot for operator approval before a real render is spent.
5. `.../lock` → `lock_production()` (`screenplay.py:1704`) seals everything into one digest,
   `ProductionLock` (`production.py:808`).
6. `.../segments` compiles sibling `SegmentSpec`s from the lock (`segments.py`, out of scope);
   `script_first.py`'s header names `compile_segments`/`to_plan_graph` as the seam that can turn a
   locked plan into a `PlanGraph` for `dag_runtime`/`validator` — but **none of `production.py`/
   `screenplay.py`/`storyboard.py`/`postproduction.py` import `dag_runtime` or `validator` directly**
   (grep-confirmed); those two surface only inside `repair_controller.py` and the segments/recipe path.
7. `.../segments/<id>/generate` → `prompt_compiler.compile_context()`/`render_prompt()`
   (`prompt_compiler.py:551,686`) builds the prompt from locked artifacts; `selection.py:365 select()`
   / `router.py:269 resolve_route()` picks the model; the render itself goes through **Flow A**.
8. `postproduction.evaluate_take()` (`postproduction.py:2163`) judges footage against spec/continuity;
   `build_postproduction_plan()` (`:1686`) assembles the EDL/sound/music cues/regeneration notes.

**Steward** is a separate, parallel loop, not part of either flow: `GET|POST /oracle/steward` →
`Steward.check()` (`steward.py:153`) reads the reliability ledger + newest `routing_matrix` + live
catalog eligibility, and on POST applies a bounded (±0.05 weight step) rebalance to the live selector
(`oracle_routes.py:553-580`).

## 5. State, persistence & invariants
- `media_jobs.db` (sqlite, WAL, `media_bus.py:54,282-285`): one table `media_jobs`
  (`job_id,name,status,spec_json,result_json,claim_token,created,updated,progress_json,stage_log_json`
  + additively migrated `archived_at`,`principal`,`owner`, `:306-395`, each `ALTER TABLE ADD COLUMN`
  wrapped in a swallow-if-`OperationalError` idempotent guard). Invariant: **one writer per job_id**,
  enforced entirely by `WHERE claim_token=?` / `WHERE status='queued'` guards on every UPDATE — never
  by an application-level lock.
- `reservation/registry.py`: a second sqlite store, claims keyed by `(worker/gpu, run_id)` with a
  heartbeat-refreshed lease; an orphaned claim self-expires (crash-safe by construction).
- `routing_matrix.py:405,423 save_matrix/load_matrix`: a persisted JSON leaderboard file, reloaded by
  `selection.select()` as `matrix` evidence.
- Production artifacts (`GenerationSnapshot`/`Screenplay`/`ProductionLock`/etc.) are `ContentAddressed`
  (`production.py:135`) — digest-identified, immutable once built; `script_first.py` persists a run to
  `runs/<kind>/<run_id>/state.json` (out of scope) — the artifact classes themselves do no I/O.
- Invariant (`production.py`'s "non-negotiable generation semantics"): a prompt minted mid-run may
  never feed a sibling segment of the same run — enforced structurally by `RunPromptLedger`
  (`production.py:197`) + `GenerationSnapshot.assert_pre_run`, not by convention.

## 6. Cross-subsystem edges
→ `managers/dispatch` (`runners/imagegen.py` calls `execute_prompt` for `generate_image`/
`generate_scene`/`generate_movie`) — see [[media-gen]].
→ `managers/resolvers/assure_model_key.py` uses `oracle/pipelines.py`'s `NAME_PIPELINE` for fuzzy
model-name resolution — the only live consumer of `oracle/pipeline.py`'s generic framework today.
→ `worker_agent` executes every studio job's `produce_clip` and hosts the `/studio/render` HTTP
surface — see [[worker-fleet]].
→ `comms`: a one-directional `_bridge()` mirror (`media_bus.py:408`) surfaces media jobs into
`GET /llm/jobs`; best-effort and exception-swallowing by design so a comms hiccup never breaks a
media job — see [[comms-index]].
← Reached by `flask_app/app/routes/{oracle_routes,script_first_routes,video_routes,
video_coordination,video_assist_media,comms_routes,worker_routes}.py` — see [[api-routes]] — and
`worker_agent/{studio_render,_studio_subproc}.py`.
`oracle/selection.py`'s `resources`/`health` steps read live model-placement evidence — see
[[serving-core]].

## 7. Key contracts / types
- `MediaRef` (`media_schema.py:21`), `CropSpec`/`SpatialRegion`/`TemporalRegion` (`crop_schema.py`),
  `FrameExtractSpec`, `AudioExtractSpec`, `GenerateSceneSpec` (`scene_schema.py:34`), `MovieSpec`/
  `GoalInterval` (`movie_schema.py:88,56`) — all `@dataclass(frozen=True)` with a validating `make_*`
  factory; every factory doubles as the bus's JSON-deserialization path (`asdict` → JSON →
  `make_*(**d)` round-trips).
- `JobSpec`/`JOB_REGISTRY` (`job_schema.py:29,41`), `JobResult`/`JobError` (`result_schema.py`).
- `CandidateVerdict`/`SelectionDecision` (`selection.py:282,297`), `RouteDecision`/`RouteRefusal`
  (`router.py:218,212`), `ValidationReport`/`ErrorCode` (`validator.py`), `Scorecard` (`scorecard.py`,
  consumed by nearly every in-scope oracle module).
- `GenerationSnapshot`/`ContinuityBible`/`ShotPlan`/`ProductionLock` (`production.py`), `PlotSpec`/
  `Screenplay`/`Scene` (`screenplay.py`), `StoryboardFrame`/`Storyboard` (`storyboard.py`), `Take`/
  `EditDecisionList`/`PostProductionPlan` (`postproduction.py`) — all `ContentAddressed` (digest-keyed).
- `ReservationTemplate`/`Stage` (`reservation/templates.py:83,60`) — `stage_name -> vram bytes` with a
  measured-overlay override.

## 8. Gotchas, tech-debt & review findings

⚠ **`media_bus.py` has no recovery path for sqlite corruption.** The only exception type ever caught
around a `sqlite3` call is `OperationalError`, and only for two narrow cases: the RO-connect fallback
(`_connect_ro`, `:302`) and the idempotent `ADD COLUMN` migration swallow (`:319-394`). A
`sqlite3.DatabaseError` — the class `"database disk image is malformed"` raises as — is never caught
anywhere in the file. `enqueue` (`:1023-1032`), `claim`'s select/update (`:1045-1061`), and
`run_claimed`'s status flips (`:1076-1098`, `:1166-1174`) all wrap the DB call in a bare
`try/finally` (or `except Exception: ROLLBACK; raise` at `:1062-1064`, which still re-raises) — so a
corrupted `media_jobs.db` propagates uncaught. Most `POST /video/jobs/*` handlers in `video_routes.py`
only catch `(ValueError, TypeError)` around the *spec-construction* call, not the subsequent
`media_bus.enqueue()`, and `flask_app/` has no `app.errorhandler(Exception)`. Net effect: a corrupted
job DB 500s the entire video-job surface at once with no automatic backup/`integrity_check`/recovery
anywhere in scope. (Two narrow exceptions: `video_routes.py:303`'s `_resolve_asset_uri` catches the
base `sqlite3.Error`, and `:1420`'s `_movie_live_status` catches `Exception` — both single-row read
helpers, not the job lifecycle.)

ℹ `oracle/pipeline.py` vs `pipelines.py` — **not a dupe**, framework vs. instance, both live but only
partially built out: `pipelines.py`'s own docstring (`:17`) promises a second pipeline,
`ORACLE_PIPELINE`/`oracle_registry`, mirroring `selection.select`'s 9 steps — no such symbol exists
anywhere in the tree (repo-wide grep). Only `NAME_PIPELINE` (fuzzy model-name resolution, consumed by
`managers/resolvers/assure_model_key.py`) is real; `selection.py`'s actual 9-step selector is its own
hand-written implementation and does not import `.pipeline`/`.pipelines`.

ℹ `oracle/repair.py` vs `repair_controller.py` — **not a dupe**. `repair.py` (`:79 attempt_repair`) is
a single-route bounded-retry policy behind `/oracle/route`; `repair_controller.py`
(`:243 RepairController.diagnose/apply`) is DAG-scale repair built on `DagRuntime`/`PlanGraph`. Zero
cross-import between them; each has independent callers.

ℹ `oracle/spatial.py`/`spatial_eval.py`/`spatial_sources.py` — **not a dupe**, a 3-stage pipeline:
`spatial.py` is the frozen contract (`SpatialSceneManifest`), `spatial_sources.py` produces a manifest
from gltf/usd/pose-track, `spatial_eval.py` (the only one of the three that imports `spatial.py`)
scores rendered output against it and maps failures to `repair_controller`'s spatial `RepairCode`s.

ℹ **The creative-authoring spine does not run through `DagRuntime`.** `production.py`/`screenplay.py`/
`storyboard.py`/`postproduction.py` never import `dag_runtime` or `validator` (grep-confirmed) — both
are scoped to `repair_controller.py` and (per `script_first.py`'s header) `segments.py`'s
`to_plan_graph`, outside this doc's file scope. A reader expecting one unified "plan graph" executing
the whole creative pipeline will not find it in these files.

ℹ `media_bus.py:1991 start_worker_daemon()` — its own docstring calls it "DEFINED but never called at
import; Phase 3 wires it at app init," which reads as unresolved but isn't: `flask_app/wsgi_app.py:
503-517` does call it, once per process at app-creation time, guarded by a module-level
`_VIDEO_DAEMON_STARTED` flag and wrapped in try/except so a start failure only logs. Central runs
multiple gunicorn workers, so **every worker process starts its own claim-loop daemon**; the bus's
atomic `claim()` (`media_bus.py:1039`, §4 Flow A step 3) is what keeps exactly one of them from
double-running a given job — an intentional multi-daemon race resolved entirely by the DB, worth
knowing before "why are there N daemons" looks like a bug.

△ `steward.py`'s alerting channel is a systemd exit code: `deploy/hugpy-steward.service`'s own comment
says alarms "exit non-zero so the unit shows failed in systemctl — that IS the alert until the UI feed
renders it (TODO-18)." An operator not watching `systemctl`/`journalctl` for this unit gets no signal.

ℹ `video_intel/studio/errors.py:18-20` carries its own `TODO(P0-1)`: three parallel error vocabularies
exist today (studio's `Ok`/`Err`/`StageError`/`ErrorCode`, `result_schema.JobError`,
`comms.jobs.JobError`) and are not yet reconciled.

ℹ `scorecard.py:97-102` documents a real, already-fixed incident: a 2026-08-21 TTS-silence fault
produced 2.32s of PCM16 at peak amplitude 1 that every existing technical check `hard_pass`ed;
`SILENT_AUDIO_PEAK_FLOOR = 500` (`:66`) is the fix — a good example of this codebase's
incident-driven hardening, not an open issue.

## 9. Deploy/run boundary
Central now runs the **src** checkout directly (`PYTHONPATH`), so any edit under `video_intel/`/
`oracle/` is live on the next `7002_hugpy_api` restart. The GPU worker runs the installed pip wheel,
so a `video_intel/studio/*` or `runners/*` change needs a wheel publish + `pip install -U` + worker
restart before a worker's `/studio/render` picks it up (`README.md §Deploy loop`) — central and worker
can transiently run different `studio/` code across a rollout. `hugpy-steward.timer`/`.service` are
checked into `deploy/` but **not installed/enabled by default** (`deploy/hugpy-steward.service`'s own
header: operator installs after review).
