# Central API & console serving — mechanics

**Scope (paths this doc covers):** `flask_app/` — `wsgi_app.py`, `app/routes/*`
(28 blueprints), `app/member_auth.py`, `app/operator_auth.py`, `app/video_auth.py`,
`app/endpoints_explorer.py`, `app/endpoints_view.py`
**One-liner:** the central Flask app factory and its 28 route blueprints — HTTP
dispatch, three independent `before_request` auth gates, and the console/SPA
serving boundary for the whole hugpy fleet (console UI, `/v1` OpenAI-compatible
API, `/v1/messages` Anthropic shim, GPU workers, phone-bricks, P3.1 agent nodes,
the Discord bot, `pip`).
**Owner process(es):** central `7002_hugpy_api` (systemd unit; `hugpy serve` →
`get_hugpy_flask()` under gunicorn, single worker process). Nothing else imports
this package for its side effects — it is the one process that mounts these
blueprints.

## 1. Purpose & responsibilities
- Builds the one Flask app central serves on :7002: discovers + registers 28
  blueprints, installs 3 independent `before_request` auth gates, mounts the
  prebuilt console SPA, and wires the comms bus + video-intel daemon at boot.
- Translates HTTP into calls on the real subsystems — dispatch/serving, comms
  substrate, oracle, video_intel, downloader — via THIN routes; most route files
  say so in their own docstrings (e.g. `comms_routes.py:1-8`, `fleet_routes.py:14`,
  `oracle_routes.py:3`: "if a route here grows logic, it's in the wrong file").
- Does NOT serve models itself ([[serving-core]] / [[engine-generation]]), does
  NOT own the stores it reads/writes ([[comms-index]]), and does NOT build the
  console UI it serves (a prebuilt `console_dist/` bundle, [[frontends]]).

## 2. Key modules (file → responsibility)
| file | responsibility |
|---|---|
| `flask_app/wsgi_app.py` | app factory `get_hugpy_flask()` (:167) — middleware, dual-mounts, SPA mount, gates, bus wiring, video daemon |
| `flask_app/app/routes/__init__.py` | blueprint discovery source — 28 `from .X import Y_bp` lines |
| `flask_app/app/operator_auth.py` | console/control-plane gate; `_SENSITIVE` allowlist (~45 rules); `principal_role()` |
| `flask_app/app/member_auth.py` | studio/media-plane gate (`/media,/ml,/uploads,/session,/chat`) |
| `flask_app/app/video_auth.py` | `/video,/movie` gate + the video-share credential seam |
| `flask_app/app/endpoints_explorer.py` + `endpoints_view.py` | interactive `/endpoints` API browser; reuses `_SENSITIVE` for curation |
| `routes/v1_routes.py` + `v1_helpers.py` | OpenAI-compatible `/v1/*`; API-key mgmt; tools prompt-shim (49 routes incl. keys) |
| `routes/chat_routes.py` + `functions/chat/streaming.py` | console-native `/chat/stream` (reflection-registered, not `@bp.route`) |
| `routes/messages_routes.py` + `messages_helpers.py` | Anthropic `/v1/messages` shim — reuses the /v1 pipeline verbatim |
| `routes/worker_routes.py` | GPU worker fleet control-plane — register/heartbeat/admit/assign/ops (82 routes, 296KB) |
| `routes/video_routes.py` | video-intelligence/studio surface (60 routes, 313KB — largest file in the app) |
| `routes/llm_storage_routes.py` | model registry reads, download enqueue, discovery/reconcile sweeps |
| `routes/comms_routes.py` | F2 principals, F4 settings, F5 unified jobs — thin adapter over `comms/*` |
| `routes/oracle_routes.py` | thin adapter over `abstract_hugpy_dev.oracle` (route-to-best-result) |
| `routes/agent_routes.py` | P3.1 agent-node fleet + one-time install-link distribution (22 routes) |
| `routes/discord_routes.py` | Discord bot's HTTP arm — bindings/bridges/sessions/bot-links (30 routes) |
| `routes/model_group_routes.py` | explicit model priority groups **+** fleet templates (2 surfaces, 1 file, 14 routes) |
| `routes/ml_routes.py` | fixed media-intelligence amenities — 13 of 16 routes registered via a loop, not `@bp.route` |
| `routes/phone_brick_routes.py` | phone-as-worker pool — enroll/heartbeat/run/provision (14 routes) |
| `routes/search_routes.py` | HF Hub search + Civitai + an unrelated host-filesystem "finder" grep (11 routes) |
| `routes/review_routes.py`, `eviction_routes.py`, `metrics_routes.py`, `fleet_routes.py`, `fleet_doctrine_routes.py` | model-curation review pipeline; eviction telemetry SSE; compute-metrics reads; fleet-config templates; doctrine reads — each a thin adapter over one package |
| `routes/upload_routes.py` | `/uploads` + per-tab session lifecycle; per-account namespace jail |
| `routes/interim_routes.py`, `script_first_routes.py` | mounted under `/video/...` on purpose, to inherit `video_auth`'s gate instead of growing a second one |
| `routes/keeper_help_routes.py`, `prompt_routes.py`, `pypi_routes.py`, `auth_proxy_routes.py`, `welcome_routes.py`, `group_routes.py` | keeper ticket filing; generic dispatch passthrough; private PyPI index; same-origin auth BFF; onboarding readiness; read-only model-groups feed |

## 3. Entry points
- CLI: `cli.py:28 _serve()` (subcommand `hugpy serve`) — sets `HUGPY_AUTH_MODE`
  (default `open`), builds the app via `get_hugpy_flask()` (`wsgi_app.py:167`),
  then runs it under gunicorn (`cli.py:61-73`; **`workers` hard-coded to `1`**,
  `cli.py:64`; `threads` default 8, `cli.py:342` — there is no `--workers` flag
  at all) or falls back to waitress / the Flask dev server if gunicorn is
  missing (`cli.py:44-59`).
- systemd: `/srv/hugpy/etc/systemd/7002_hugpy_api.service` runs
  `venv/bin/hugpy serve --host 0.0.0.0 --port 7002 --auth external`, with
  `Environment=PYTHONPATH=/srv/hugpy/src/abstract_hugpy_dev/src` — confirmed live
  against the running process (2026-09-14): central now imports the **src
  checkout**, not the installed wheel (§9).
- HTTP: nginx `dev.hugpy.ai` / `api.hugpy.ai` → `127.0.0.1:7002`
  (`SERVICES.md:14-15`); every blueprint's rules, plus the SPA catch-all
  (`wsgi_app.py:136-164`).
- App-factory boot order inside `get_hugpy_flask()` (`wsgi_app.py:167-520`):
  discover/register the 28 blueprints via `abstract_flask.get_Flask_app(
  routes=routes, ...)` (:177-182, third-party pip pkg — `_discover_blueprints`
  scans `vars(routes_module)` for `*_bp` attrs, `abstract_flask.py:36-60`) →
  CORS + upload-size cap (:170-189) → 10 blueprints DUAL-MOUNTED under `/api`
  (:199-312) → optional `media_intelligence` bridge (:347-403) →
  `ApiPrefixMiddleware` wrap (:408) → SPA mount if built (:409-411) → optional
  `/demo-media/*` (:412-424) → client-disconnect probe (:425-437) → the 3 auth
  gates (:439-474) → `/endpoints` override (:476-486) → comms-bus wiring
  (:488-501) → video-intel job-worker daemon start, once per process (:503-519).

## 4. Data flow (the spine)
Three independent HTTP surfaces converge on ONE dispatch core —
`managers/dispatch/dispatch.py:775 execute_chat_stream()` ([[serving-core]],
[[engine-generation]]) — but differ in registration mechanism, auth, and
bookkeeping. That divergence is the thing to understand first.

**A. `/chat/stream` — the console-native path**
1. `chat_routes.py:9` does **not** use `@bp.route`. It calls
   `register_categories(chat_bp, {"chat": {"stream": chat_stream}})` —
   reflection from the third-party `abstract_flask` package
   (`generator/generateFlaskRoute.py:183→177→115`) that mints TWO Werkzeug rules
   (`/chat/stream`, `/chat/stream/`, GET+POST) whose view function parses the
   request into a dict (`get_request_datas`, same file :50-53), maps it onto
   `chat_stream`'s parameters via `prune_inputs` (:130/155), calls it, and
   returns the result AS-IS when it is already a Flask `Response` (:138-139/
   163-164) — which it always is here.
2. Auth: `member_auth.py`'s blanket gate (`/chat` ∈ `_MEMBER_SURFACE`, :65) —
   member/operator session OR any valid API key.
3. `chat_stream()` (`functions/chat/streaming.py:362-392`) builds a pydantic
   `ChatBody`, resolves the dedicated pool + principal from Flask's request
   context BEFORE entering the async generator (:374-381 — the SSE generator
   later runs on the shared async runtime loop, which has no request context),
   returns a `text/event-stream` `Response`.
4. `stream_events()` (:125-284) builds `prompt_kwargs`, mints/reuses
   `request_id`, creates a row in `comms.jobs.job_store` (:227-235 — full F5
   lifecycle: pending→processing→streaming→terminal, visible on `/llm/jobs`),
   attaches a REAL cancel handle (`job_store.attach_cancel`, :241-246), then
   drives `execute_chat_stream(cancel_event=..., **prompt_kwargs)` (:135,249),
   translating each `StreamEvent` to browser SSE (`event_to_sse`, :60-74).

**B. `/v1/chat/completions` — the OpenAI-compatible path**
1. Plain `@v1_bp.route(...)` (`v1_routes.py:215`) decorated with `v1_auth`
   (:95-111): an API key is required ONLY IF the site's `require_key` flag is on
   (`api_key_required()`, defaults **False** — `api_keys.py:68-73`); when
   required, scope `"v1"` or `"full"`. **This is the one chat surface with no
   auth by default.**
2. `_completion_kwargs(payload)` (`v1_helpers.py:19-89`, stdlib-only, unit-
   testable standalone) translates the OpenAI payload. A ROUTE-LOCAL tools shim
   (`_build_tools_preamble`/`_parse_tool_calls`, invoked `v1_routes.py:225-230,
   355-366`) prompt-injects a Qwen/Hermes `<tools>` preamble — "the frozen engine
   schema can't carry tools" (:220-224) — and parses `<tool_call>` blocks back
   out of the reply.
3. `_v1_events()` (:169-193) calls the SAME `execute_chat_stream()` (:188) and
   records it via `managers.dispatch.activity.begin/on_token/end` (:176,186-193)
   — which IS a thin shim over the same `comms.jobs.job_store`
   (`managers/dispatch/activity.py:1-13,27-37`: `begin()` calls
   `job_store.create(..., transport="v1")`), so a `/v1` call **does** appear on
   `/llm/jobs`. It does **not**, however, create a `cancel_event` or call
   `job_store.attach_cancel()` anywhere in `v1_routes.py` — only
   `streaming.py:241-249` does that (grep-confirmed across all three chat
   route files). See §8 finding.
4. Emits OpenAI-shaped SSE (`chat.completion.chunk`) or one JSON body, with its
   own error-shape mapping (503+Retry-After for capacity, 400 for request-shape
   faults — :60-92,409-448).

**C. `/v1/messages` — the Anthropic shim** (Claude Code / Claude Agent SDK)
1. Plain `@messages_bp.route("/v1/messages", ...)` (`messages_routes.py:143`).
   Auth is its OWN in-file check (not in `_SENSITIVE` or either blanket gate):
   open unless `require_key`, else `verify_api_key(token, required_scope="v1")`
   — also reads Anthropic's `x-api-key` header, not just `Authorization: Bearer`.
2. `anthropic_to_openai_payload()` (`messages_helpers.py:264`) builds the SAME
   OpenAI-style payload `v1_chat_completions` builds, then REUSES
   `_build_tools_preamble`/`_completion_kwargs`/`_inject_tools_preamble` from
   `v1_helpers.py` and `_v1_events`/`_finish_reason` from `v1_routes.py`
   verbatim (`messages_routes.py:22-39`) — "no new model plumbing", only
   protocol translation both ways (`AnthropicStreamEncoder`,
   `messages_helpers.py:401`). Inherits the same no-`attach_cancel` gap as B.3.

**D. A plain read — `GET /models`** (`llm_storage_routes.py:54-141`): reads the
manifest, stamps operator-block state, media-chat flags, physical size, and a
live HOT/COLD residency join against `list_workers()` (:109-126); `?verbose=1`
adds a full per-worker serving join (`_verbose_worker_join`, :144-204). No
dispatch, no job_store; open (not in `_SENSITIVE`, not in either blanket gate).

**E. Worker M2M — register/heartbeat, and a push-not-pull dispatch model.**
`worker_routes.py` (82 static `@worker_bp.route` rules, no dynamic
registration) is central's half of the GPU-worker contract; `worker_agent/`
is the other half ([[worker-fleet]]). `POST /llm/workers/register` (:981,
body = `RegisterRequest` :447-499) and `POST /llm/workers/<id>/heartbeat`
(:1395, body = `HeartbeatRequest` :560-711, ~35 fields incl. `gpus`,
`loaded_models`, `slots`, `allocations`, `storage`) are gated by
`_enrollment_ok()` (:394-407 — a Bearer enrollment token, or none at all when
`HUGPY_WORKER_ENROLL_REQUIRED` is off), never by `operator_auth._SENSITIVE`
— the fleet keeps breathing unauthenticated-by-default at this layer.
**Live dispatch is CENTRAL-PUSH, not a worker pull from this blueprint**:
central calls OUT via `worker_http.post(worker, op_path, ...)` for every ops
verb (`_relay_worker_op`, :2085-2125) and to warm a model (`_kick_warm`,
:154-211). The one true PULL here is model-file PROVISIONING: `GET
/llm/models/<key>/manifest|file|chunksums|archive` (:4493,4752,4910,4942,
gated by `_transfer_authorized()` :410-437 — operator session OR enrollment
token) — "workers pull missing model files from central" (:4263).

## 5. State, persistence & invariants
- **Model registry / JSON sidecars** (single-host JSON + file locks, distinct
  from the comms sqlite substrate): manifest + discovery report at
  `MODELS_DISCOVERY_PATH` (`imports/src/constants/constants.py:131`, default
  `<PROJECTS_HOME>/model_discovery.json`); siblings in the same directory —
  `pruned_models.json`, `media_models.json`, `media_default.json`
  (`imports/config/models/models_config.py:865,923,1003`).
- **API keys**: `api_keys.json` next to the manifest
  (`functions/imports/utils/api_keys.py:68-69`); the site-wide `require_key`
  flag lives in the SAME file, default `False`. Legacy rows with no `scopes`
  field read as `["full"]` (`_scopes_of`) so every pre-scope key still passes a
  scoped check.
- **Video-share keys**: a SEPARATE store, `functions/imports/utils/
  video_share_keys.py` — its own `hpv_` token prefix, never the API-key store.
- **Uploads**: `UPLOADS_HOME/<namespace>/…`, namespace = `operator_auth.
  upload_namespace(username)` (`operator_auth.py:607-621`); accountless callers
  (operator-token M2M, open mode) keep the historical flat layout. Audit log:
  `PROJECTS_HOME/audit.log` (JSONL) + comms bus topic `"audit"`
  (`comms_routes.py:27-45`).
- **comms substrate** (shared, cross-process — [[comms-index]]): `job_store`/
  `jobs`, `feeds`, `calllog`, `settings`, `principals`, `model_metrics`,
  `blocklist`, `priority_groups`, `evictions`, `agent_nodes` — all reached
  LAZILY from routes, never imported at module load (so a store outage can
  never break app boot).
- **Auth caches**: `operator_auth._SESSION_CACHE`/`_PRINCIPAL_CACHE`
  (`operator_auth.py:334-341`), 30s TTL, in-process dict keyed by a SHA-256 of
  the cookie header.
- **Invariant — exactly one gunicorn worker process.** `cli.py:64` hardcodes
  `workers=1` ("singleton registries/job store"); no `--workers` flag exists.
  The in-process auth caches above are safe under this; several OTHER files'
  design-rationale comments assume multiple worker processes regardless (§8).
- **Invariant — `open` mode is permissive by default.** Only
  `HUGPY_AUTH_MODE=external` (or a configured `HUGPY_OPERATOR_TOKEN`) turns on
  real enforcement in `operator_auth`/`member_auth`/`video_auth`; the
  self-hosted product ships with no login wall.
- **Invariant — the operator-console gate fails CLOSED** on an unreachable
  upstream auth service (`operator_auth.py:445-447,467-468`) — never silently
  opens.

## 6. Cross-subsystem edges
**→ calls out to:** `managers/dispatch/dispatch.py` `execute_chat_stream()` /
`execute_prompt_stream()` (:775,697) — the ONE convergence point for
`/chat/stream`, `/v1/chat/completions`, `/v1/messages`, and `/prompt`
([[serving-core]], [[engine-generation]]); `managers/dispatch/activity` (/v1
bookkeeping, now a `job_store` shim); `managers/resolvers/remote.py` (capacity/
timeout classification, cold-hold retry-after) and `managers/resolvers/
model_resolver.py` (`resolve_model_key`, reject-at-intake); `managers/
draft_models.py` (draft-head rejection); `managers/fleet/templates.py`
(`fleet_routes.py`); `comms/*` — `job_store`/`jobs`, `feeds`, `calllog`,
`settings`, `principals`, `model_metrics`, `blocklist`, `priority_groups`,
`evictions`, `agent_nodes` ([[comms-index]]); `downloader/queue.py` +
`presence.py` — the SEPARATE download daemon `llm_storage_routes.py` enqueues
into, never runs in-process (`llm_storage_routes.py:377-389`);
`abstract_hugpy_dev.oracle` — `catalog`/`router`/`runtime`/`scorecard`/
`authority`/`selection`/`steward`/`repair`/`evaluation`/`script_first`/
`interim_ledger` ([[video-oracle]]); `abstract_hugpy_dev.video_intel` —
`media_bus` (started FROM `wsgi_app.py` itself, :503-519) and
`identity_profiles` ([[video-oracle]]); the GPU `worker_agent` contract
([[worker-fleet]]); `phone_brick_store` + `managers/phone_brick_orchestrator`
(complementary, not duplicate: the orchestrator executes centrally, the
`phone_brick/` device payload is only tarred and shipped —
`phone_brick_routes.py:120-121`); `review/pipeline.py` +
`discovery_dossier/store.py` ([[curation-reliability]]); `fleet_doctrine/
doctrine.py` + `doctor.py` ([[curation-reliability]]); an optional external
`media_intelligence` package bridge (`wsgi_app.py:347-403`).

**← called by:** the console SPA (`src/react/ui`, [[frontends]]) via
same-origin `/api/*` fetches; nginx (`dev.hugpy.ai`, `api.hugpy.ai`); the GPU
`worker_agent` process ([[worker-fleet]]); the Discord bot (`bot/`,
[[cli-agents-platform]]) polling `discord_routes.py`; phone-brick devices;
P3.1 agent-node daemons; `hugpy chat` CLI / `hugpy_agent`
([[cli-agents-platform]]); external OpenAI-SDK / Anthropic-SDK / curl / `pip`
clients hitting `/v1/*`, `/v1/messages`, `/pypi/*`.

Sibling docs: [[serving-core]], [[engine-generation]] (the dispatch core every
chat surface converges on), [[comms-index]] (every state store these routes
read/write), [[worker-fleet]] (the other half of `worker_routes.py`'s M2M
contract), [[video-oracle]] (`oracle_routes.py`, `video_routes.py`'s execution
backends), [[curation-reliability]] (review / discovery-dossier /
fleet-doctrine / downloader), [[cli-agents-platform]] (bot, agent nodes, CLI
clients), [[frontends]] (the console SPA this app serves).

## 7. Key contracts / types
- **Two DIFFERENT "chat" pydantic schemas — do not conflate.** `ChatBody`
  (`functions/imports/utils/schemas/chat_schemas.py:3`, the route-facing shape
  `/chat/stream` accepts: `model_key, prompt, messages, file, images,
  max_new_tokens, temperature, top_p, do_sample, unbounded, task, request_id,
  pool, alloc, transport, channel, principal`) vs `ChatRequest`
  (`imports/src/schemas/chat_schemas.py:129`, **frozen**, `extra="forbid"` —
  the ENGINE-facing shape `_completion_kwargs` translates an OpenAI payload
  into).
- **`StreamEvent` family** (yielded by `execute_chat_stream`/
  `execute_prompt_stream`): token / status / done / error — THREE different
  serializers exist for the same events: `event_to_sse()` (`streaming.py:60-74`,
  browser SSE), `_v1_events()` + the inline `chunk()` builder (`v1_routes.py`,
  OpenAI `chat.completion.chunk` SSE), `AnthropicStreamEncoder`
  (`messages_helpers.py:401`, Anthropic SSE).
- **`operator_auth._SENSITIVE`** (`operator_auth.py:78-332`):
  `list[tuple[set[str] methods, re.Pattern path]]`, matched AFTER stripping a
  leading `/api` — the single source of truth for "is this route
  operator-only", ALSO reused verbatim by the `/endpoints` explorer to mark
  routes "internal" (`endpoints_view.py:25-32`).
- **API key record** (`api_keys.json` rows): `{id, name, hash, pool, label,
  scopes, revoked, disabled, expires_at, last_used}`.
- **Video-share token**: `hpv_`-prefixed; carried as `?share=`,
  `X-Video-Share:`, or `Authorization: Bearer hpv_…` (`video_auth.py:97-122`)
  — a structurally separate credential space from API keys and
  operator/member sessions (the share check is consulted ONLY from
  `video_auth.py`, by construction).
- **Job record** (`comms.jobs.JobStore`, `comms/jobs.py:362`):
  `create`/`cancel_authoritative`/`snapshot`/`expire_pending_orphans`
  (:391,636,807,873); `to_legacy_dict()` backs the older `/jobs`
  (`llm_storage_routes.py`) while `snapshot()` backs the newer unified
  `/llm/jobs` (`comms_routes.py:124-143`).

## 8. Gotchas, tech-debt & review findings

**⚠ Auth gaps (highest priority — each lets a caller past the blanket gates do
more than its siblings allow):**
- **`worker_routes.py`: `moe`/`bnb` registry writes have NO auth at all.**
  `POST /llm/workers/<id>/moe` (:1224-1274) and `.../bnb` (:1277-1342) are
  absent from `operator_auth._SENSITIVE`, have no in-file check, and are not
  even `audit()`-logged — yet `bnb`'s own docstring (:1297) claims "Operator-
  gated like the other registry writes." It is not.
- **`worker_routes.py`: `cache-evict`/`auto-reap` are NOT matched by the
  `_SENSITIVE` regex that claims to gate them.** `POST /llm/workers/<id>/
  cache-evict` (:2305-2316) and `.../auto-reap` (:3499-3511) both cite
  operator-gating in their own docstrings (:2090, :3501-3505), but
  `operator_auth.py:227`'s alternation lists `evict`/`reap`/`reap-orphans`/…
  as literal strings — "cache-evict" and "auto-reap" don't match any of them,
  and neither route has its own `operator_authenticated()` check (the only
  hits in the whole file are inside `_transfer_authorized`, :426-431).
- **`video_routes.py`: the movie-session trio has no per-account ownership
  check.** `GET /video/studio/movies` (:1618), `POST .../pause` (:1683,
  `media_bus.cancel` at :1713) and `.../resume` (:1744, re-enqueues at :1829)
  never call the file's own `owner_of`/`_may_view_job`/`_viewer` helpers,
  unlike the near-identical owner-gated `POST /video/jobs/<id>/cancel`
  (:3818, gate at :3828-3831). Any principal past the blanket `/video` gate —
  including an unscoped `hpv_` share guest (§4.C note) — can enumerate,
  pause, or resume another account's movie render.
- **`video_routes.py`: the entire `/video/identity-profiles*` cluster
  (:4787-6017, 21 routes) has zero owner check**, including the destructive
  `DELETE /video/identity-profiles/<slug>` (:4834) and `PATCH .../<slug>`
  (:4868) — unlike every job/clip/media route in the same file, which IS
  owner-scoped via a hand-copied triad (`owner_of`+`_may_view_job`+
  `_forbidden_artifact`, repeated at 6 call sites e.g. :3802-3804,4142-4144 —
  never centralized, plausibly how both video gaps above happened).
- **`POST /prompt` (`prompt_routes.py:78`) appears to have no auth at all** —
  not in `_SENSITIVE`, not matched by `member_auth._MEMBER_SURFACE`
  (`^/(media|ml|uploads|session|chat)`, :65), no in-file check found. Drives
  the same `execute_prompt` surface as the gated `/ml/*` amenities for free,
  unlike `/chat/stream` (member-gated) and `/v1` (opt-in key-gated).
- **`GET/POST /finder/search`** (`search_routes.py:509`, gate
  `_finder_authorized()` :493-506) **accepts any valid API key with no
  `required_scope` check** — unlike `ml`'s `required_scope="ml"` or
  `messages`'s `required_scope="v1"`. A narrowly-scoped key can read
  arbitrary source/comms-db/spool content via the `_FINDER_ROOTS` whitelist
  (:485-490).
- **`/v1/chat/completions` and `/v1/messages` create a job_store row (via
  `activity.begin()`, `managers/dispatch/activity.py:27-37` — now a thin
  `job_store.create(..., transport="v1")` shim, so `/v1` IS visible on
  `/llm/jobs`) but never attach a live cancel handle.** Only
  `functions/chat/streaming.py:241-246` calls `job_store.attach_cancel(...)`;
  grepping `v1_routes.py` and `messages_routes.py` finds none. A
  `POST /llm/jobs/<id>/cancel` on one of these likely hits
  `cancel_authoritative()`'s "no live owner → force-terminal" branch
  (`comms_routes.py:159-165`): the row flips `cancelled` while generation
  keeps running server-side.

**△ Debt:**
- **`ApiPrefixMiddleware` (`wsgi_app.py:20-34`, installed unconditionally at
  :408) likely makes the ~10 explicit `/api`-prefixed dual-mounts
  (`wsgi_app.py:199-312`) unreachable dead weight** — the middleware strips a
  leading `/api` from `PATH_INFO` before Flask routing runs on every request,
  so PATH_INFO can never still carry `/api` by the time a dual-mounted rule
  could match. Confirm live (remove one dual-mount, curl `/api/llm/workers`)
  before deleting anything.
- **`worker_routes.py` (5735 lines) and `video_routes.py` (6017 lines) dwarf
  every other route file** (next is `agent_routes.py` at ~1679 lines).
  Symptoms of the scale: a route registered TWICE in worker_routes.py — `GET
  /llm/workers/install.sh` at both :897 (`importlib.resources`-served
  `bootstrap.sh`) and :5565 (a distinct ~150-line inline heredoc script) —
  Werkzeug's stable sort means :897 wins today, making :5565 likely-dead-or-
  silently-diverging; and inconsistent error shapes within one function
  (`workers_probe`, :3660-3682, mixes `abort(code, description=...)` and
  `jsonify({"ok": False, ...}), code` in the same function). `video_routes.py`
  has a confirmed-dead `_retired_video_identity_profile_reconstructions`
  (:4932-4996, kept whole under a never-delete policy after its `@route` was
  removed for a Werkzeug-shadowing bug, :4926-4931) and a header (:1-31) still
  describing itself as the original 6-route "Phase 3a crop feature" — never
  updated as studio/identity-profiles/prompt-assist/mlt-render accreted.

**ℹ Notes:**
- **`cli.py:64` hard-codes gunicorn `workers=1`** (no `--workers` flag exists
  at all) — yet `eviction_routes.py:12-21`, `comms_routes.py:49-51`, and
  `wsgi_app.py:503-507` all justify their design by MULTIPLE gunicorn worker
  processes. The defensive multi-process-safe code is harmless under
  `workers=1`, but the stated rationale is stale relative to the deployed
  topology.
- **Three route-registration mechanisms coexist**, so a plain `@bp.route(`
  grep can undercount: static decorators (most of the app, incl. all 82+60
  routes of `worker_routes.py`/`video_routes.py`); reflection-based
  `register_categories` (`chat_routes.py:9`, `abstract_flask/generator/
  generateFlaskRoute.py:183`), used only for `/chat/stream`; a dynamic
  `add_url_rule` loop in `ml_routes.py:326-332`, minting 13 of 16 routes with
  no matching `@ml_bp.route(` line anywhere in the file.
- **Doc drift, not code:** `/srv/hugpy/README.md:13` states central runs "the
  pip-installed wheel, not `src/`" — empirically false as of this review
  (§9); the live systemd unit and `docs/mechanics/README.md`'s 2026-09-14 note
  are authoritative. Exactly the drift `/srv/hugpy/CLAUDE.md §5` warns readers
  to check code against, not just trust.
- **The auth surface is per-route, not one rule for the whole app.** Beyond
  the 3 blanket gates and `v1_auth`, more DISTINCT in-file idioms exist:
  `ml_routes.py:49-69` stacks an extra `@ml_bp.before_request` media-key gate
  on top of `member_auth`'s blanket gate; `pypi_routes.py` has a three-door
  check; `worker_routes.py` alone has two (`_enrollment_ok()` :394-407 for
  register/heartbeat, `_transfer_authorized()` :410-437 for file transfer).
  "Not in `_SENSITIVE`" does not mean "unauthenticated" — check each route.
- `model_group_routes.py`'s docstring (:1-28) documents only the "priority
  groups" half; 7 of 14 routes (the `/llm/templates` cluster, :271-491) go
  unmentioned.

## 9. Deploy/run boundary
- Editing anything under `flask_app/` is a **src edit**: the running unit
  (`/srv/hugpy/etc/systemd/7002_hugpy_api.service`) sets
  `PYTHONPATH=/srv/hugpy/src/abstract_hugpy_dev/src`, confirmed against the
  live process (`ps aux` shows `/srv/hugpy/venv/bin/hugpy serve` importing from
  that path, 2026-09-14) — so `systemctl restart 7002_hugpy_api` is sufficient;
  no wheel rebuild needed. `/srv/hugpy/README.md:13`'s "central runs the
  pip-installed wheel, not `src/`" is stale (§8) — trust the unit file.
- The console UI this app serves is a SEPARATE build artifact (`console_dist/`,
  [[frontends]]) — a `flask_app/` edit never touches it; only `_ui_dist_dir()`'s
  search path (`wsgi_app.py:37-63`) changes which prebuilt bundle gets mounted.
- Both nginx hosts (`dev.hugpy.ai` strips `/api`, `api.hugpy.ai` does not —
  `SERVICES.md:14-15`) front the SAME app; route code does not need to know
  which one a request arrived through — `ApiPrefixMiddleware`
  (`wsgi_app.py:20-34`) plus the `/api` dual-mounts (§8) exist to cover both.
- Both auth enforcement and its emergency override are env-flippable with no
  code change or restart-time risk: `install_operator_gate`/
  `install_member_gate`/`install_video_gate` (`wsgi_app.py:443-474`) are
  installed unconditionally but only actively enforce once
  `HUGPY_AUTH_MODE=external` or `HUGPY_OPERATOR_TOKEN` is set (safe to deploy
  ahead of flipping the mode); `HUGPY_TESTING_LOCKDOWN`
  (`operator_auth.py:676-697`) 503s everyone but operators (plus a small
  worker/health exempt list) — worth knowing before debugging a mysterious
  fleet-wide 503 during a model battery / fleet test.
