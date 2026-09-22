# Worker fleet — mechanics

**Scope (paths this doc covers):** `worker_agent/` (full GPU worker), `gguf_worker/`
(slim ARM/Termux worker), `phone_brick/` (phone-as-worker) — plus the central
`/api/llm/workers/*` and `/api/phone-brick/*` HTTP contracts these processes speak
against (read from `flask_app/app/routes/worker_routes.py` / `phone_brick_routes.py`
for wire-format accuracy; central's own implementation is [[api-routes]], not here).
**One-liner:** turns a GPU box (or an ARM/Termux/phone box) into fleet capacity —
register with central, advertise a reachable address, accept model assignments,
serve inference, manage local VRAM/RAM/storage budget, self-update.
**Owner process(es):** each is a **standalone daemon on a worker box**, never
loaded by central. `hugpy worker` / `python -m abstract_hugpy_dev.worker_agent`
(full worker); `python -m abstract_hugpy_dev.gguf_worker` (slim); `python -m
abstract_hugpy_dev.phone_brick worker|rpc-backend|orchestrate|analyze` (phone arm).

## 1. Purpose & responsibilities
- Register/enroll with a central, advertise a callback address, heartbeat, adopt
  whatever model assignment central hands back, and serve `/infer`.
- Own **all local resource management**: VRAM/RAM budget and eviction, a storage
  budget with FIFO evict-or-refuse, a local slot pool for crash-isolated native
  serving, self-update to central's required package version.
- Explicitly does **not** decide placement/routing (central owns the assignment
  list and ships it in every register/heartbeat reply) and does not implement its
  own auth beyond presenting an enrollment token — admission is an operator action
  on central, not something the worker negotiates.
- `gguf_worker` and `phone_brick` are not central's serving fleet at all in the
  vision case — `phone_brick/registration.py` talks to a **separate** `/phone-brick/*`
  pool; only `phone_brick/rpc_backend.py` (a phone lending RAM as a shard backend)
  joins the real `/llm/workers` pool.

## 2. Key modules (file → responsibility)
| file | responsibility |
|---|---|
| `worker_agent/agent.py` | Core daemon (~12k lines): identity/advertise, register+heartbeat loop, slot-pool supervision, hooks into `managers/serve`+`managers/dispatch`, self-update, the worker's own HTTP surface (`/infer`, `/ops/*`, …). |
| `worker_agent/__main__.py` | `python -m abstract_hugpy_dev.worker_agent` → `agent.main()`. |
| `worker_agent/install.py` | **Canonical** cross-platform installer: k118 doctrine preflight, then a systemd `--user` unit (Linux) / launchd agent (macOS) / scheduled task (Windows). |
| `worker_agent/bootstrap.sh` | Bare-box→enrolled-worker script: venv + `pip install abstract_hugpy_dev[engine]`, then runs `install.py`. |
| `worker_agent/deploy/install.sh` | Legacy **SYSTEM**-unit installer for `abstract-hugpy-worker.service` — stale, superseded (§8, §9). |
| `worker_agent/provision.py` | Model file provisioning: central-first (WireGuard manifest/file endpoints) → Hugging Face fallback; chunked/resumable transfer. |
| `worker_agent/budget.py` | Storage budget: FIFO evict-to-fit or refuse-before-download — "the model being called always wins." |
| `worker_agent/flex.py` | Tolerance-band VRAM%/RAM%/ctx% "flex before evict" allocation math. |
| `worker_agent/environment_report.py` | k118 self-report (venvs, binaries, GPU/driver, OS) behind `/ops/environment` + the heartbeat's `environment_digest`. |
| `worker_agent/studio_render.py` | Worker-side GPU render endpoints (`/studio/render`) for the Oracle/video-studio pipeline. |
| `worker_agent/comfy_watchdog.py` | Reclaims idle VRAM from an adopted, out-of-band ComfyUI process via ComfyUI's own `/free` API. |
| `worker_agent/gen_gate.py` | Per-model in-process generation lock — a llama.cpp `Llama` context is not concurrency-safe. |
| `worker_agent/pid_registry.py` | Model → PID/VRAM attribution registry consumed by every heartbeat. |
| `worker_agent/aggregate.py` | Worker-side rolling aggregate of its own served-request stats; central pulls it on read, never polls. |
| `worker_agent/aptitude/` | Worker-local mechanical self-test scorer, dark by default (`HUGPY_WORKER_SELFTEST=on`). |
| `gguf_worker/agent.py` | Slim GGUF-only worker for constrained/Termux/ARM boxes — same register/heartbeat wire contract, ~1/14th the size, single-slot in-process only. |
| `phone_brick/worker.py` | On-phone Flask job queue wrapping the ONNX detector. |
| `phone_brick/orchestrator.py` | Control-box conductor: fans one image across a phone chain, collates consensus. |
| `phone_brick/rpc_backend.py` | Lets a phone join the **same** `/llm/workers` shard pool as `role=rpc`, launching llama.cpp's real `rpc-server`. |
| `phone_brick/registration.py` | A phone's own register/heartbeat against the **separate** `/phone-brick/*` vision pool. |
| `phone_brick/detector.py`, `consensus.py` | ONNX YOLOv8 inference core; plurality-vote aggregation across phones. |
| `phone_brick/analyze.py` | Detections → text → GGUF chat-model reasoning step. |

## 3. Entry points
- `hugpy worker …` → `cli.py:77-79` (`_worker()`) → `worker_agent/agent.py:main()`
  (`agent.py:11738`); all trailing CLI args pass straight through to the agent's
  own parser (`cli.py:11-13`).
- `python -m abstract_hugpy_dev.worker_agent` → `worker_agent/__main__.py:1-4`.
- `python -m abstract_hugpy_dev.worker_agent.install …` (or `bootstrap.sh` step 4)
  → `install.py:main()` (`install.py:286-393`) — the canonical installer.
- `python -m abstract_hugpy_dev.gguf_worker` → `gguf_worker/__main__.py` → same
  `agent.main()` pattern, slim module.
- `python -m abstract_hugpy_dev.phone_brick {worker|rpc-backend|orchestrate|analyze}`
  → `phone_brick/__main__.py` subcommand dispatch.
- **Inbound HTTP the worker itself serves** (central/ops call in): `build_app()`
  in `agent.py:3615-4779` — `/health`, `/infer`, `/infer/stream`,
  `/infer/cancel/<id>`, `/identity`, `/probe/<model_key>`, `/models/unload`,
  `/models/redownload`, `/reap`, `/slots/<id>/{relaunch,unload}`, and the `/ops/*`
  family (`restart`, `free-ram`, `update`, `pip`, `config`, `evict`,
  `cache-evict`, `reap-orphans`, `vram-holders`, `aggregate`, `environment`,
  `heartbeat-nudge`). No auth on this surface — trusted-LAN by design.
- **Outbound HTTP the worker makes** to central: `CentralClient` (`agent.py:607-661`)
  — `register()`, `heartbeat()`, `evictions_ingest()` (§7).

## 4. Data flow (the spine)
1. **Startup** (`agent.py:11738` `main()`): parse CLI/env (`_build_parser`,
   `agent.py:11649-11710`), strip a leaked CUDA env that survives re-exec ("boot
   detox"), prime torch before any stray `llama_cpp` import can poison it.
2. **Advertise resolution** (`agent.py:11772-11785`): only advertise a URL the
   operator set explicitly (`--advertise`/`WORKER_URL`); otherwise call
   `_local_ip_toward(central_url)` (`agent.py:538-562`), which opens a UDP socket
   *toward* central and reads the kernel's chosen source address — the worker's
   real outbound LAN IP, never `127.0.0.1`/`127.0.1.1`. This exists because a
   loopback advertise is only correct for a worker truly co-located with central;
   a remote box that ends up advertising loopback would hand central an address
   it can never dial back on. If detection also fails, the worker advertises
   nothing and central falls back to the request's source IP (defeated by NAT
   hairpinning for a box reached via a public domain — the reason this helper
   exists at all).
3. **Register** (`agent.py:11596-11647` `_register()`): `CentralClient.register()`
   POSTs identity + capabilities (§7) to central. Central's reply carries (at
   least) `id`, `models` (the assignment), `required_pkg_version`. The worker
   saves `worker_id` to its id-file, adopts assignment + limits + boot-prewarm
   star, then checks self-update.
4. **Admission** is entirely **operator/console-driven**, not a worker-initiated
   step: `POST /api/llm/workers/<id>/admit` and `/admission`
   (`flask_app/app/routes/worker_routes.py:1621,1660`) are console actions. The
   worker only ever *discovers* its admission state indirectly — a blocked or
   unenrolled worker gets HTTP 401/403 on register/heartbeat, which
   `CentralClient._post` turns into `WorkerRejected` (`agent.py:568-576,624-631`),
   handled by `_terminal_exit()` (`agent.py:11000-11015`): log why, `os._exit(0)`
   — exit **0** so `Restart=on-failure` does not respawn a deliberately-evicted
   worker. There is no retry; the operator must re-admit/re-issue a token.
5. **Heartbeat loop** (`agent.py:11378-11594`, thread started at `agent.py:12036`):
   every `--heartbeat`/`WORKER_HEARTBEAT` seconds (default 15, `agent.py:11667`),
   POST a large capability+telemetry payload (§7) to
   `/api/llm/workers/<id>/heartbeat`, and **also** mirror it straight into
   Postgres via `comms/heartbeat_db.py` (`agent.py:579-604`) so central/console can
   read worker state from the DB without an HTTP round trip. On the reply: adopt
   assignment (`_sync_assignment`), central resource limits, calibration
   corrections, blocked-model set, least-reaping policy, the boot-prewarm star,
   storage-budget inputs, and the self-update target — all one-way, worker never
   originates these. HTTP 410 → central forgot this worker, re-register
   (`agent.py:11586-11589`); `WorkerRejected` → terminal exit; anything else is
   logged and the loop continues (a missed beat must never crash the agent).
6. **Assign → provision → serve** (`_sync_assignment`, `agent.py:5051-5143`):
   central's `models` list in the register/heartbeat reply is the **sole**
   assignment source. Adoption is tier-lazy: only 🔒static models are eagerly
   pre-pulled (`_kick_provision`); on-demand and 📌pinned models download on first
   call (`_ensure_present`, `agent.py:667+`). `_fill_empty_slots` seats
   already-local models immediately rather than waiting for the next maintenance
   tick. Two distinct central verbs feed this: `POST …/<id>/assign`
   (`worker_routes.py:1827`) only *designates* a model (the lazy path above);
   `POST …/<id>/load` (`worker_routes.py:4061-4089`) is stronger — a VRAM
   preflight, then assign, then a background warm/probe that actually seats the
   model now, refusing 409 up front when it provably won't fit.
7. **Serve**: a `framework=="gguf"` model prefers a native, crash-isolated
   `llama-server` slot, gated by `_slot_capability()` — the **engine-usability
   gate** (`agent.py:11119-11154`): it resolves a real `llama-server` binary via
   `engine.resolve.server_bin()`, and only if that succeeds does it advertise
   `slot_capable: true`. Absent a usable binary it falls back to the in-process
   `llama_cpp.server` child (text only, vision GGUF refused) or, with
   `SLOT_COUNT=0`, the fully in-process runner — and says so honestly in
   `slot_incapable_reason` on **every** beat, closing a real 2026-07-11 incident
   where a box silently served non-native with central none the wiser.
8. **Self-update** (`agent.py:10946-10998`): each register/heartbeat reply's
   `required_pkg_version` drives `pip install -U --no-deps <pkg>==<version>`
   (PyPI by default, or `--pkg-index`/`WORKER_PKG_INDEX` — e.g. central's own
   PEP-503 simple index for an egress-less/WireGuard-only box), then a guarded
   restart (§5) reloads the new code.

## 5. State, persistence & invariants
- **On disk**: a worker-id file (`_platform.paths.worker_id_file()`); a sibling
  `<id-file>.settings.json` holding console-set runtime overrides, projected onto
  `os.environ` **once** at boot by `_apply_settings_env` (`agent.py:10129-10283`)
  — **settings always win over env/systemd drop-in, loudly logged** when they
  disagree (e.g. `SLOT_COUNT`, `COMFY_URL`, `HUGPY_HOT_CACHE_ROOT`,
  eviction policy); an update-state json throttling repeated self-update attempts;
  `aggregate.py`'s rolling stats file (debounced flush, pulled via
  `GET /ops/aggregate` / central's `GET /llm/workers/<id>/aggregate`, never
  polled at beat cadence); provision.py's resumable chunk-state files.
- **In memory**: `WorkerState` (`agent.py:4936-5002`) — `assigned_models`,
  `_provisioning`, and central-owned `limits`/`model_last_picked`/`allocated`/
  `refused` (all **adopted**, never computed locally); a provision concurrency
  semaphore (default 1 — serial — after a 2026-07-15 thundering-herd incident
  where one big assignment spawned N concurrent multi-threaded pulls); the live
  werkzeug `http_server` handle, closed cleanly before a restart to release the
  bind port.
- **Invariants**:
  - `off` is the fleet-install default for auto-serving (`install.py:195-198`) —
    a freshly-enrolled worker registers but an operator turns models on from the
    console.
  - Central owns assignment; `_sync_assignment` only ever *adopts* it
    (`agent.py:5051-5143`).
  - A ⭐ boot-prewarm star loads **once** per process lifetime and never
    re-warms after eviction (`deploy/WORKER-BOOT-PREWARM.md`; latched via
    `_BOOT_PREWARM_DONE` in `_adopt_boot_prewarm`, `agent.py:5991-6055`) — a
    reconcile-kept-warm version of this caused a live stall incident and was
    reverted.
  - A `wildcard: false` (default) worker only ever serves its own designated
    assignment; opting in lets it take overflow for undesignated models, never
    changes eviction (`deploy/WORKER-WILDCARD.md`).
  - Under systemd the agent **never** `os.execv`s itself — it exits with a
    distinct non-zero code (`_RESTART_EXIT_CODE=42`, `agent.py:2167-2199`) so
    `Restart=on-failure` spawns exactly **one** tracked process. `execv` remains
    correct standalone (no systemd). This is itself a hardening after two real
    incidents (§8).
  - A 401/403 from central is terminal (exit 0, never retried).

## 6. Cross-subsystem edges
- → `managers/serve/{slots,slot_agent,serve,profiles,overrides}` — slot pool
  (spawn/port, `ServeMode` selection, ctx resolver, quant-autofit hook).
- → `managers/dispatch/dispatch` — fit/evictable/make-room/evict-reason hooks the
  worker registers at boot (`agent.py:11936-11953`).
- → `managers/comfy/comfy_runner` — headroom-before-render hook.
- → `engine/resolve` — native `llama-server` binary resolution (the
  engine-usability gate, §4 step 7).
- → `comms/heartbeat_db` (direct DB heartbeat mirror), `comms` bus
  (`wire_cancel`/`wire_job_events`).
- → `fleet_doctrine/{doctrine,doctor}` — k118 preflight (`install.py:220-268`)
  and the heartbeat's `doctrine_status`.
- ← central `flask_app/app/routes/worker_routes.py` (`/api/llm/workers/*`) and
  `phone_brick_routes.py` (`/api/phone-brick/*`), both mounted `url_prefix="/api"`
  in `flask_app/wsgi_app.py:200,208` — see [[api-routes]].
- Sibling docs: [[serving-core]] (`managers/serve` internals this subsystem
  drives), [[engine-generation]] (`engine/`, `provisioner.py`), [[media-gen]]
  (ComfyUI/comfy_runner), [[comms-index]] (`heartbeat_db`, model registry).

## 7. Key contracts / types
- `CentralClient` (`agent.py:607-661`): base = `{central}/api/llm/workers`;
  `register(payload)` → `POST /register`; `heartbeat(worker_id, payload)` →
  `POST /{worker_id}/heartbeat`; `evictions_ingest(events)` →
  `POST {central}/api/llm/evictions/ingest` (its own endpoint, outside the
  `/workers` prefix — telemetry must never ride the beat, "the 2026-07-27 stat
  storm starved heartbeats" this exact way). Confirmed central-side at
  `worker_routes.py:981` (`/llm/workers/register`) and `:1395`
  (`/llm/workers/<worker_id>/heartbeat`).
- `WorkerRejected(code, message)` — HTTP 401 (bad/revoked/required enroll token)
  or 403 (blocked) → terminal, never retried.
- **register payload** (`agent.py:11598-11626`): `name, url, port, gpus, role,
  rpc_endpoint, free_ram, ram_total, models, worker_id, pkg_version,
  engine_build, engine, pool, caps, env, serving_limits, slot_capable,
  slot_incapable_reason, task_capabilities`.
- **heartbeat payload** adds (`agent.py:11449-11553`): `loaded_models, loading,
  models_local, provisioning, provision_progress, spill, disk, comfy,
  loaded_detail, slots, allocations, calibration_samples, pid_registry, storage,
  vram_evictions, vram_holders, install` (= `{unit, via_systemd, venv, python,
  canonical}` — the install-shape drift detector, `agent.py:11043-11106`),
  `environment_digest, doctrine_status, aggregate`.
- **register/heartbeat response fields the worker reads**: `models` (assignment),
  `required_pkg_version`, plus central-owned limits/calibration/blocked-models/
  least-reaping/boot-prewarm-star/storage-budget inputs — all via `_adopt_*`
  helpers, all strictly read-only from the worker's side.
- `ServeMode` (`managers/serve/serve.py:131-135`): `OFF | SYSTEMD | SUPERVISED |
  SWAP`, selected from `DEFAULT_SERVE_MODE` (env) or a per-model
  `cfg.extra["serve_mode"]` override (`serve.py:258`) — see §8 for a real
  vocabulary mismatch with `install.py`'s own flag of the same name.
- `gguf_worker` and `phone_brick/rpc_backend.py` speak the **identical**
  register/heartbeat contract at the same `/api/llm/workers/*` path (own
  `CentralClient`s built the same way). `phone_brick/registration.py`'s vision-pool
  arm speaks a **different** contract: `POST {central}/api/phone-brick/register`
  and `/{id}/heartbeat`.

## 8. Gotchas, tech-debt & review findings
- **⚠ `WORKER_PORT` and slot-child ports are two independent, uncoordinated
  allocations.** The agent binds its own HTTP API at `args.host:args.port`
  (`WORKER_PORT`, default 9100 — `agent.py:11661,12056`). Slot children bind at
  `SLOT_PORT_BASE + (SLOT_ID-1)` (default base `8101`,
  `managers/serve/slots.py:126-130,141`; `managers/serve/slot_agent.py:38,47`).
  Nothing anywhere cross-validates that these ranges are disjoint — an operator
  setting `WORKER_PORT=8101`, or bumping `SLOT_PORT_BASE`/`SLOT_COUNT` into the
  9100s, gets a raw "Address already in use" with no pre-flight warning.
- **⚠ `SLOT_PORT_BASE` has no per-worker namespace, and two workers verifiably
  share a host today.** `services/aeb-worker/hugpy-worker.service` (`WORKER_PORT=
  9200`) and `signals/hugpy_systems/hugpy-worker.service` (`WORKER_PORT=9100`,
  worker `ae`) both run on the **same physical host** — confirmed via
  `scripts/bin/aeb`'s own health check (`curl http://127.0.0.1:9200/health`,
  `scripts/bin/aeb:15`) and live-verified (`curl 127.0.0.1:9200/health` answers
  from the aeb worker on this box). Neither unit sets `SLOT_PORT_BASE`, so both
  default their slot children to `:8101`/`:8102`. Worse than a bind failure: the
  supervisor's orphan-adoption probe (`_slot_answering`, `agent.py:2127-2142`)
  would see the **sibling** worker's slot answering `/health` on that port and
  conclude it already owns a recovered orphan — silently leaving its own slot
  unseated rather than erroring loudly. Live check on 2026-09-14: only `:9200` is
  currently listening and its one loaded model has no active slot bind, so this
  is not observed firing right now, but nothing in the code prevents it.
  (Related, already fixed: the agent's **own** port colliding with an orphaned
  copy of *itself* after a bad `os.execv`-under-systemd re-exec was a real,
  separate incident — computron's 160→219 restart-loop — closed by the
  `_RESTART_EXIT_CODE`/never-execv-under-systemd rule at `agent.py:2167-2199`.
  That fix does not cover either case above.)
- **⚠ `install.py --serve-mode` and the real `ServeMode` it configures disagree.**
  `install.py`'s own flag is `choices=("off","on")` (`install.py:304-306`) and
  writes the chosen string verbatim into env `DEFAULT_SERVE_MODE`
  (`install.py:198`). But that env var is actually consumed by
  `managers/serve/serve.py`'s `ServeMode(str, Enum)` — `OFF="off"`,
  `SYSTEMD="systemd"`, `SUPERVISED="supervised"`, `SWAP="swap"`
  (`serve.py:131-135`), via `ServeMode(DEFAULT_SERVE_MODE)` (`serve.py:258`).
  `"on"` is not a member — `--serve-mode on` produces an env value that raises
  `ValueError` the first time a GGUF model's serve mode resolves on that box.
  The live `aeb` unit sidesteps the installer's own vocabulary entirely by
  hand-setting `DEFAULT_SERVE_MODE=swap` — a value `install.py`'s flag can never
  produce. `agent.py` itself never reads `DEFAULT_SERVE_MODE`/`SERVE_MODE` at
  all; it is purely a `managers/serve` concept the worker inherits via env.
- **△ `worker_agent/deploy/install.sh` is a stale installer, and the code still
  half-trusts it.** It provisions a SYSTEM unit `abstract-hugpy-worker.service`
  for module `abstract_hugpy` (the old package name). `SERVICES.md:83` confirms
  this exact unit is masked/disabled fleet-wide — the live worker is always the
  `install.py`-produced `--user` unit `hugpy-worker.service` — yet
  `_compute_install_shape` (`agent.py:11043-11106`) still treats **both** unit
  names as `canonical: true`. Code and ops doc disagree on what "canonical"
  means; `deploy/install.sh` is a candidate for removal or a stale-noise banner,
  and `_CANONICAL_UNITS` (`agent.py:11051`) should drop the legacy alias.
- **△ `gguf_worker` has no enrollment/auth path.** Its `CentralClient._post`
  sends no `Authorization` header and `_build_parser` defines no `--token`
  (unlike the full worker's `WorkerRejected`-on-401/403 handling,
  `agent.py:608,624-631`). If central has `HUGPY_WORKER_ENROLL_REQUIRED` on, a
  slim worker can't supply a token and fails generically instead of surfacing a
  clear rejection.
- **△ `gguf_worker --pkg-name` is a dead flag.** It's defined in
  `_build_parser` but self-update actually targets a module-level `PKG_NAME`
  captured once at import from `WORKER_PKG_NAME` — passing `--pkg-name` on the
  command line silently does nothing.
- **△ `gguf_worker` has no disk-budget/eviction analog to `budget.py`.**
  `ModelStore.provision()` (`gguf_worker/agent.py:341-370`) downloads a newly
  assigned model's GGUF file(s) into `GGUF_WORKER_MODELS_DIR` with no size cap
  and no cleanup of a superseded model's directory; the only eviction in this
  file is `LlamaSlot` swapping the one **in-memory** loaded model
  (`gguf_worker/agent.py:376-424`, "small-RAM devices are the target"). On a
  small-storage ARM/phone box, cycling through several assigned models only
  grows disk usage — nothing ever reclaims it.
- **ℹ `gguf_worker`'s self-update index defaults to central, not PyPI.**
  `index = args.pkg_index or (args.central.rstrip("/") + "/api/llm/pip/simple")`
  (`gguf_worker/agent.py:119`) always resolves to *some* index; the full
  `worker_agent` instead defaults to plain PyPI and only points at central's
  PEP-503 index when `--pkg-index`/`WORKER_PKG_INDEX` is set explicitly
  (`agent.py:10972-10977`, docstring `10949-10953`) — asymmetric defaults for
  an otherwise-identical mechanism, presumably because `gguf_worker`'s
  Termux/ARM targets are more often egress-constrained by default.
- **△ `phone_brick/rpc_backend.py` keeps advertising `role=rpc` after its
  `rpc-server` subprocess fails to spawn.** A missing binary only logs a
  warning; registration/heartbeat continue reporting a live `rpc_endpoint` and
  non-zero free VRAM, so a shard load routed there fails at runtime against
  nothing listening. The spawned process is also never health-checked or
  restarted if it dies after a successful start.
- **ℹ `phone_brick/analyze.py`'s NO-THINK gap is self-documented as latent.**
  Its own comment (`analyze.py:103-107`) names the exact failure mode: the
  reasoning step assumes `DEFAULT_CHAT_MODEL` is non-reasoning and hands the
  raw completion straight to the operator as "the analysis," which is "latent
  — but it fires the moment the chat default moves to a reasoning model,
  which is exactly what happened to `/video/prompt/assist`" — i.e. this
  exact bug class has already shipped once elsewhere in the codebase.
- **△ `GET /llm/workers/install.sh` is registered twice in `worker_routes.py`,
  with two different bodies.** Once at `:897-921` (serves the packaged
  `worker_agent/bootstrap.sh` resource, sed-patched to the live origin — the
  one the top-level README documents) and again at `:5565-5567+`
  (`worker_install_script`, a second, inline literal script built from
  `_central_base_url()`). Flask/Werkzeug resolves same-rule ties by
  registration order, so the second is very likely unreachable dead code — at
  minimum it is two sources of truth for one install script that can silently
  diverge.
- **△ `bootstrap.sh` hand-maintains a second dependency list.**
  `bootstrap.sh:125-126` pip-installs `sentence-transformers openai-whisper
  keybert "numpy<2.5"` as a shell literal — the exact contents of
  `pyproject.toml`'s declared `embed`, `audio`, and `keywords` extras
  (`pyproject.toml:109,114-115`) plus one ad hoc pin, instead of `pip install
  abstract_hugpy_dev[embed,audio,keywords]`. The two lists can drift silently.
- **△ The NAT-hairpin LAN-advertise helper is duplicated, buggy in one copy,
  and missing where it matters most.** `_local_ip_toward()` (UDP-connect to
  learn the real outbound IP rather than trust a possibly-loopback default)
  is implemented nearly identically in `agent.py:538-562` and
  `phone_brick/rpc_backend.py:83-101`, with no shared helper — and only the
  `worker_agent` copy rejects a `127.`-prefixed result (`agent.py:558`, `if
  ip and not ip.startswith("127."):`); the phone_brick copy returns
  `getsockname()` unconditionally and would advertise a loopback
  `rpc_endpoint` verbatim if the route to central ever resolved that way. The
  helper is **absent** entirely from `gguf_worker/agent.py`, whose `main()`
  only uses `args.advertise` verbatim or falls back to central's
  request-source-IP guess — so a `gguf_worker` reached through a public
  domain (NAT hairpinning) has no defense against the exact misadvertisement
  bug the other two exist to solve.
- **ℹ No auth on any of these processes' inbound HTTP surfaces** (`agent.py`'s
  `/ops/*`, `gguf_worker`'s `/infer`, `phone_brick.worker`'s `/queue`) —
  trusted-LAN-only by design; worth stating plainly rather than assuming.
  `phone_brick`'s `sh` verb (`PHONE_BRICK_ENABLE_SHELL=1`) is an explicit,
  self-documented opt-in RCE footgun on top of that.
- **ℹ Possible duplicate**, flagged but not chased: `managers/
  phone_brick_orchestrator/runner.py` (141 lines) alongside top-level
  `phone_brick/orchestrator.py` (142 lines) — near-identical size, undocumented
  which is canonical/live. ARCHITECTURE.md's own duplicate-impl list doesn't
  mention this pair; worth adding once disambiguated.

## 9. Deploy/run boundary
- **Workers run the pip-installed wheel, not `src/`.** Every observed unit
  `ExecStart`s a venv's `python -m abstract_hugpy_dev.worker_agent`; self-update
  is literally `pip install -U --no-deps abstract_hugpy_dev==<version>`
  (`agent.py:10972-10975`) from PyPI or central's own simple index. Editing
  `src/` does **nothing** to a running worker until a version is published and
  either the worker's own self-update or a manual `pip install -U` + restart
  picks it up — unlike central, which now runs `src/` directly (see
  `docs/mechanics/README.md`'s cross-cutting facts).
- **Canonical path for a new worker**: `bootstrap.sh` (venv +
  `pip install "abstract_hugpy_dev[engine]==<version>"`) → `worker_agent/install.py`
  (k118 preflight → refuses on doctrine BLOCKERS unless `--force` → writes +
  enables `hugpy-worker.service`). Linger is enabled so the unit survives logout
  and reboot (`install.py:118-127`); `Restart=on-failure`, deliberately **not**
  `always` — a 401/403-evicted worker must stay stopped (§5).
- **Two live examples**, both same-host: `services/aeb-worker/hugpy-worker.service`
  (worker `aeb`, port 9200, `DEFAULT_SERVE_MODE=swap`) and
  `signals/hugpy_systems/hugpy-worker.service` (worker `ae`, port 9100).
- **Stale path**: `worker_agent/deploy/install.sh`'s SYSTEM unit
  `abstract-hugpy-worker.service` — masked, do not use for a new install (§8).
- `gguf_worker`/`phone_brick` have no systemd-unit installer of their own in this
  scope; `phone_brick/bootstrap.sh` provisions a Termux box directly by role
  (`worker`, `rpc-backend`, or both).
