# Engine + generation lanes — mechanics

**Scope (paths this doc covers):** `engine/`, `managers/generate/`, `managers/llama/`,
`provisioner.py`, `model_sync.py`, `model_battery.py`
**One-liner:** resolves/provisions the native llama.cpp engine, picks and drives one of three
generation lanes (transformers-in-process, llama-cpp-python-in-process, or a native/python
`llama-server` child reached over HTTP) for a GGUF or transformers model, and gets model weight
bytes onto a box (detect-missing→enqueue, or pull-whole-dir).
**Owner process(es):** library code, imported in-process by central (`7002_hugpy_api`) and the
worker agents (`worker_agent`, `gguf_worker`) — none of these files run as their own daemon. Two
exceptions: `engine/` is also driven by the one-shot `hugpy install-engine` CLI, and
`provisioner.py` is also invoked every 10 minutes by `hugpy-sentinel.timer` → `sentinel run-once`
(`Type=oneshot`, not a long-lived loop). `model_sync.py` has no automatic caller — CLI/library
only. `model_battery.py` is library-only, called synchronously from the studio/imagegen render
path.

## 1. Purpose & responsibilities

- Resolve, fetch (prebuilt GitHub release) or build (cmake-from-source) the native `llama-server`
  / `rpc-server` binaries llama.cpp ships outside PyPI (`engine/`).
- Select and drive ONE of three generation lanes per request: transformers-in-process
  (`DeepCoder`), llama-cpp-python-in-process (`LlamaCppPythonRunner`), or an HTTP proxy to a
  `llama-server` child (slot, cross-machine RPC shard lead, or native vision `--mmproj` server) —
  `managers/generate/` + `managers/llama/`.
- Thread MoE expert-offload (`--n-cpu-moe`) placement from the allocator down to the native slot
  child, and REFUSE (never silently mis-serve) an MoE GGUF that can only be expressed by a native
  slot.
- Detect declared-but-missing weights across 3 registries and enqueue them on the existing
  download queue (`provisioner.py`); pull a whole model directory from a central node
  (`model_sync.py`).
- Record self-contained, best-effort per-session render/generation telemetry (`model_battery.py`)
  — off the request path, never allowed to fail it.
- Deliberately does **not**: implement the download transfer engine (`downloader/`), build the
  native `llama-server` argv or `--n-cpu-moe` value itself (`managers/serve/slot_agent.py`,
  [[serving-core]]), or do model registry/config resolution (`imports/config/models_config.py`,
  `managers/resolvers/`).

## 2. Key modules (file → responsibility)

| file | responsibility |
|---|---|
| `engine/resolve.py` | locate `llama-server`/`rpc-server`/`llama-cli`; persisted-install record; `LD_LIBRARY_PATH` derivation; spawn-probe health |
| `engine/fetch.py` | download+unpack the matching prebuilt GitHub release asset (OS/arch/CUDA scored) |
| `engine/build.py` | cmake-from-source fallback (git clone + cmake --build, `-DGGML_RPC=ON` always) |
| `managers/generate/coder.py` | `DeepCoder` — transformers/PyTorch in-process causal-LM runtime + `REGISTRY` singleton cache |
| `managers/generate/config.py` | `DeepCoderConfig`, device/dtype pick, 4-bit/LoRA wiring, `build_deepcoder_runtime()` |
| `managers/generate/coder_guff.py` | `DeepCoderGGUF` — **dead**, unreferenced llama-cpp-python wrapper, CPU-only |
| `managers/generate/generate_runner.py` | `DeepCoderChatRunner` — Runner-protocol adapter over `DeepCoder` (the live transformers lane) |
| `managers/generate/generate_runner2.py` | **dead** — near-duplicate of `generate_runner.py`, zero importers anywhere |
| `managers/llama/serve.py` | re-export shim onto `managers/serve/serve.py` (kills a drift-prone duplicate) |
| `managers/llama/runners/get.py` | `get_llama_runner()` singleton cache + `_build_runner()` — the lane-selection cascade, the central decision point of this doc |
| `managers/llama/runners/chat_runner.py` | `LlamaCppChatRunner` — Runner-protocol adapter + stale-slot self-heal |
| `managers/llama/runners/src/base_runner.py` | `LlamaCppBaseRunner` ABC — shared streaming/unbounded-continuation/loop-guard/usage logic |
| `managers/llama/runners/src/ccp_runner.py` | `LlamaCppRunner` — HTTP client to any `llama-server` (slot child, shard lead, vision server, plain `serve_endpoint`) |
| `managers/llama/runners/src/python_runner.py` | `LlamaCppPythonRunner` — the real in-process llama-cpp-python GGUF load |
| `managers/llama/runners/src/shard_server.py` | `ensure_shard_server` / `ensure_vision_server` — spawn+cache managed `llama-server` child processes |
| `provisioner.py` | `Want` dataclass; `comfy_wants`/`studio_wants`/`tasks_wants` scanners; `enqueue()` onto the download queue |
| `model_sync.py` | CLI/library: pull a whole model dir from a central node via `worker_agent.provision.ensure_model_present` |
| `model_battery.py` | `BatteryRun` — atomic JSON+HTML render/generation telemetry recorder (studio + imagegen only) |

## 3. Entry points

- CLI: `hugpy install-engine [--cuda] [--build-from-source] [--tag] [--jobs] [--force]` →
  `cli.py:198 _install_engine()` → `engine/fetch.py:117 install()`, falling back to
  `engine/build.py:34 build_from_source()` on any fetch failure (`cli.py:210-215`).
- `python -m abstract_hugpy_dev.model_sync --central URL {--model KEY|--all|--list}`
  (`model_sync.py:121 main()`) — standalone; nothing else in the package imports this module.
- `python -m abstract_hugpy_dev.provisioner [--apply] [--floor-gb N] [--json]` or `... catalog
  [--type] [--missing]` (`provisioner.py:867 main()`).
- Automatic: `sentinel/checks.py:261 default_weight_wants()` feeds `provisioner.wants()` into
  every sentinel pass; `sentinel/runner.py` applies the deterministic `enqueue_download` remedy,
  gated by `HUGPY_SENTINEL_DOWNLOADS` (default on) — see [[curation-reliability]].
- The request path: `managers/resolvers/categories/frameworks.py:2 FRAMEWORK_RUNNERS` maps
  `(framework,task)`→runner class (`("transformers","text-generation")→DeepCoderChatRunner`;
  `("gguf","text-generation"|"image-text-to-text")→LlamaCppChatRunner`, lines 3/4/9) — read by
  dispatch ([[serving-core]]) to route a chat/completion request into this subsystem.
- `managers/llama/runners/get.py:160 get_llama_runner(model_key)` — every GGUF-lane caller (the
  chat runner, `shard_server.py`) goes through this one function.
- `model_battery.run_for_session(session_key)` / `.record(...)` — called from
  `video_intel/studio/produce.py`, `managers/imagegen/imagegen_runner.py`,
  `oracle/interim_ledger.py`, `video_intel/studio/tester.py`.

## 4. Data flow (the spine)

**A — `hugpy install-engine`**
1. `cli.py:198 _install_engine()` → default `engine/fetch.py:117 install()`, short-circuited if
   `resolve.server_bin()` already resolves one and `force` is False (`fetch.py:124-125`).
2. `_release()`(93) + `_pick_asset`/`_score_asset`(61-90) score GitHub release assets by OS/arch
   token and CUDA-wanted-vs-present; download, unpack into `paths.engine_dir()`, flatten a single
   wrapper dir (156), `chmod +x` (104); verify `resolve.server_bin()` finds it (148-152) or raise.
3. On ANY fetch failure `cli.py:210-215` falls back to `engine/build.py:34 build_from_source()`
   (git clone + cmake, `-DGGML_RPC=ON` always, `-DGGML_CUDA=on` opt-in) — same verify-or-raise.
4. Either path calls `resolve.py:64 persist_install(dir, server_bin)`, writing
   `{"engine":{"dir":...,"server_bin":...}}` into the **worker agent's own settings JSON** so a
   different-user/HOME process (a systemd worker unit) still finds it (k94).
5. Later callers resolve via `resolve.py:171 server_bin()`'s 5-step order (env override →
   env-pinned dir → persisted record → per-user default dir → `PATH`); a spawner also derives
   `LD_LIBRARY_PATH` via `resolve.py:265 ld_library_path_with_engine()` so the child finds sibling
   `.so`s.

**B — a GGUF request picks a lane (`managers/llama/runners/get.py`)**
1. Dispatch builds a `LlamaCppChatRunner` (cheap, no load — `chat_runner.py:58`). First
   `.run()`/`.stream()` touches `.runner` (73-77) → `get.py:160 get_llama_runner(model_key)` →
   cache hit, or `_build_runner()` (244).
2. `_build_runner()` cascade: refuse if `no_local_serving()` is set (251-254) → resolve an
   env-profile venv gate if attributed (213-241) → if `HUGPY_RPC_SERVERS` is set, spawn/reuse an
   RPC shard lead via `shard_server.py:118 ensure_shard_server()` → HTTP `LlamaCppRunner`
   (263-277) → probe an already-serving HTTP endpoint (`serve_endpoint()`, 279-285) → else
   `SlotPool().endpoint_for(model_key, opts=...)` (296-380, hand-off into [[serving-core]]),
   carrying `opts["n_cpu_moe"]` from `HUGPY_N_CPU_MOE` (344) plus path/ctx/gpu-layers/alloc-mode.
3. No slot seated it → an MoE GGUF is **refused** (`LocalEngineUnavailable`, 410-459), never
   loaded in-process, because llama-cpp-python has no `n_cpu_moe` equivalent. A vision GGUF
   instead gets a native `llama-server --mmproj` via `shard_server.py:264 ensure_vision_server()`
   (460-475) — the in-process multimodal handler is known to fail loading the projector.
4. Last resort: true in-process `python_runner.py:61 LlamaCppPythonRunner(model_key)` —
   `llama_cpp.Llama(model_path=..., n_gpu_layers=spill.llama_kwargs(...)['n_gpu_layers'], ...)`
   (166-173) in this process; raises `LocalEngineUnavailable` with an actionable message if
   `llama_cpp` isn't installed (480-492).
5. Whichever runner comes back, `base_runner.py:276 stream_chat` / `:325 stream_chat_unbounded`
   drive it uniformly (loop-guard, usage/timings, unbounded continue-loop):
   `LlamaCppRunner._iter_stream` (HTTP SSE, `ccp_runner.py:112`) and
   `LlamaCppPythonRunner._iter_stream` (local `create_chat_completion(stream=True)`,
   `python_runner.py:312`) both implement the same `(text_chunk, finish_reason)` hook.
6. The native `--n-cpu-moe` argv is built OUTSIDE this doc's scope, in
   `managers/serve/slot_agent.py:597 _build_cmd()` — named here because step 2/3 only make sense
   with it: a `--help`-probe version check (`_server_supports_flag`, 374-387, cached per binary)
   gates `["--n-cpu-moe", str(eff_n_cpu_moe)]` (1101), degrading to plain layer-split (one warning
   log) if the binary predates the flag (1061-1071), or if the box has no native binary at all and
   falls back to `python -m llama_cpp.server` (1167-1173 — also can't express `--n-cpu-moe`,
   1142-1156, or vision, 1129-1141). See [[serving-core]].

**C — getting model bytes onto a box**
1. `provisioner.py:646 wants()` runs 3 independent try/excepted scanners: `comfy_wants()` (483,
   `<root>/checkpoints/<file>` + WORKER-SETUP §5b shared id_lock assets, catalog-first against a
   live ComfyUI-Manager, k97b), `studio_wants()` (570, `video_intel.studio.MODEL_REGISTRY` weight
   dirs), `tasks_wants()` (603, curated `models_config.MODELS` rows via the same
   `resolve_model_dir` resolver the download engine itself uses).
2. Presence is coarse: `_missing_reason()` (132) distinguishes ABSENT vs ZERO_BYTE only — any
   nonzero total under a dir reads as present.
3. `enqueue(want)` (683) refuses (returns, never raises) on: unresolved source (line 31, "sources
   are never guessed"), a duplicate live job (`_is_duplicate`, 672), or breaching the free-space
   floor (`floor_bytes()`, 78 — default 500 GB / `HUGPY_PROVISION_FLOOR_GB`). Else it calls
   `downloader/queue.py enqueue_download()` (715) — the SAME function the console's `POST
   /models/<key>/download` route calls, so dedupe/progress/cancel/retry all run through one
   `hugpy-downloader-dev` daemon.
4. Separately, `model_sync.py` pulls an ENTIRE model directory from a central node's read-only
   surface (`GET /api/llm/models{,/<key>/manifest,/file,/archive}`) by delegating wholesale to
   `worker_agent/provision.py ensure_model_present()` (`model_sync.py:36,64`) — archive-first with
   a verified/resumable per-file fallback that IS exact-size-checked and complete-or-raise ("a
   truncated/short file never passes as complete", `worker_agent/provision.py:852-863`; raises on
   incomplete at `~1285-1292`/`~1432`).
5. Neither module enforces PROVISIONS.md's hot/cold rule ("workers serve from their own hot dir,
   never cold; cold→hot is a local copy, never a re-download") — both just land bytes at whatever
   `DEFAULT_ROOT`/resolved dest already points to (§8).

## 5. State, persistence & invariants

- `engine/resolve.py` reads env (`HUGPY_ENGINE_DIR`/`LLAMA_CPP_DIR`, `LLAMA_SERVER_BIN`,
  `WORKER_RPC_BIN`/`LLAMA_RPC_BIN`, `LLAMA_CLI_BIN`, `HUGPY_ENGINE_REPO`, `HUGPY_ENGINE_TAG`,
  `HUGPY_ENGINE_GIT_URL`, `GITHUB_TOKEN`); writes/reads the persisted-install record under
  settings-JSON key `"engine"` (45-89, the worker agent's own settings file, not a new format).
  `native_engine_status()` TTL-caches a `--version` spawn-probe per binary path for 300s
  in-process (306-307).
- `managers/llama/runners/get.py`: process-local singletons — `_LLAMA_INSTANCES`
  (model_key→runner, 15), `_REFUSED` backoff cache (120s default, `HUGPY_LOAD_REFUSE_BACKOFF_S`,
  21) so a doomed model isn't retried every request. Two locks: `_LLAMA_LOCK` (micro-held,
  registry read/write only) and `_LLAMA_BUILD_LOCK` (held for a whole slow build) — invariant:
  never hold `_LLAMA_LOCK` across a build, or heartbeat/status readers queue behind a minutes-long
  load.
- `managers/generate/coder.py`: a SEPARATE process-local singleton, `REGISTRY` (`_Registry`, 561)
  keyed by `DeepCoderConfig.cache_key()` (`config.py:67`) — the transformers lane and the GGUF
  lane never share a cache entry.
- `managers/llama/runners/src/shard_server.py`: `_SERVERS` dict
  (`(model_key,rpc,tensor_split)`→`{proc,base_url}`) — one managed `llama-server` child per
  shard-plan or per vision model, alive for the process's life, health-polled (`/health`, 180s
  default via `HUGPY_SHARD_HEALTH_TIMEOUT`), spawned via `popen_detached` so it survives a reload.
- `provisioner.py` holds no state of its own beyond a per-process ComfyUI-Manager catalog memo
  (`_catalog_cache`/`_folders_cache`, success AND failure cached, only cleared by
  `reset_catalog_caches()`) — safe because `provisioner.wants()`'s only automatic caller
  (`sentinel`) is a fresh `Type=oneshot` process every 10 minutes, not a long-lived loop.
- `model_battery.py`: a writable run-dir under
  `/srv/share/projects/hugpy/model-battery/<YYYYMMDD-HHMM>[-N]/` (fallback
  `/mnt/llm_storage/model-battery/...`, see §8) holding `results.json`+`gallery.html` (always
  atomically rewritten whole, tmp+`os.replace`) and an append-only `run.log`.
  `HUGPY_MODEL_BATTERY=off` disables; `HUGPY_MODEL_BATTERY_ROOT` overrides both roots at once.
- **Invariant** (`get.py`): an MoE GGUF is never served in-process — it gets a slot (native
  `--n-cpu-moe`) or the request is refused.
- **Invariant** (`base_runner.py`): every `stream_chat*` call yields zero-or-more `TokenEvent`
  then exactly one terminal `DoneEvent`/`ErrorEvent`.

## 6. Cross-subsystem edges

**→ calls out to** (all out of this doc's file scope): `managers/serve/slots.py
SlotPool.endpoint_for()` and `managers/serve/slot_agent.py _build_cmd()`/`_server_supports_flag()`
(native child + `--n-cpu-moe` argv, [[serving-core]]); `managers/serve/serve.py
serve_endpoint()`/`serve_model_name()`/`_ctx_for()` (`managers/llama/serve.py` is a deliberate
re-export of this, not a copy); `managers/serve/policy.py no_local_serving()`;
`managers/serve/overrides.py resolve_override_gguf()`/`autofit_gguf_prefer()`; `managers/spill.py`
— `llama_kwargs()`, `rpc_servers()`/`tensor_split()`/`main_gpu()`, `gguf_moe_detail()`,
`cpu_resident_bytes()`/`free_ram_bytes()`, `_binding_supports_rpc()`, `transformers_max_memory()`,
`bnb_4bit_env()` (ALL GPU/CPU placement math for both lanes lives here, not in this subsystem);
`managers/chat_context/unbounded.py run_unbounded()` (the non-streaming continuation driver
`generate_runner.py` uses); `downloader/queue.py enqueue_download()`, `downloader/engine.py
DOWNLOAD_KIND`, `comms/jobs.py job_store` (the transfer daemon `provisioner.py`/`model_sync.py`
ride on); `worker_agent/provision.py ensure_model_present()`/`list_central_models()` (what
`model_sync.py` wraps); `_platform/paths.py`, `_platform/binaries.py`, `_platform/procutil.py
popen_detached()`.

**← called by:** `managers/resolvers/categories/frameworks.py FRAMEWORK_RUNNERS` (the dispatch
table routing into this subsystem); `worker_agent/agent.py`, `gguf_worker/agent.py` (report
`llama_supports_gpu_offload()`/engine health upward, construct runners locally);
`sentinel/checks.py`/`sentinel/runner.py` ([[curation-reliability]] — automatic
`provisioner.wants()` scan + deterministic download remedy); `cli.py` (`install-engine`);
`video_intel/studio/produce.py`, `managers/imagegen/imagegen_runner.py`,
`oracle/interim_ledger.py`, `video_intel/studio/tester.py` ([[media-gen]] — `model_battery`
recording); `flask_app/app/functions/imports/utils/workers.py` (central's worker selection reads a
worker's reported `engine.supports_gpu_offload`).

Sibling docs: [[serving-core]] (slots/slot_agent/dispatch — the other half of every GGUF request),
[[worker-fleet]] (the process that reports `supports_gpu_offload` and calls the equivalent of
`model_sync` provisioning), [[curation-reliability]] (sentinel — `provisioner.py`'s automatic
trigger), [[media-gen]] (`imagegen_runner` — a `model_battery` consumer).

## 7. Key contracts / types

- `Want` (`provisioner.py:87`, frozen dataclass) —
  `registry|name|reason|dest|hub_id|filename|include|framework|est_bytes|note`; `.resolved` ⟺
  `hub_id` truthy; `.fingerprint` = `"weight_missing:<registry>:<name>"` (also the sentinel
  anomaly fingerprint).
- `DeepCoderConfig` (`config.py:28`, frozen dataclass) —
  `model_dir|device|torch_dtype|use_quantization|...|adapter_dir|max_new_tokens_cap|...`;
  `.cache_key()` is the `REGISTRY` dict key.
- `LlamaCppBaseRunner` (`base_runner.py:71`, ABC) — subclasses implement exactly 3 hooks:
  `_iter_stream(messages,max_tokens,temp,top_p,extras)→AsyncIterator[(text,finish_reason)]`,
  `_chat_complete(...)→(text,finish_reason)`, `_raw_complete(...)→(text,finish_reason)`; the
  streaming loop, unbounded continuation, loop-guard, and usage/timings capture are inherited and
  final.
- `StreamEvent` family (`TokenEvent`/`DoneEvent`/`ErrorEvent`) — the wire contract every lane must
  emit identically (see invariant in §5).
- `FRAMEWORK_RUNNERS: Dict[(framework,task), Type[Runner]]`
  (`managers/resolvers/categories/frameworks.py:2`) — the closed map from a model's
  `(framework,task)` to a runner class; `KNOWN_TASKS_REGISTRY` is derived from it so it can't
  drift.
- `LocalEngineUnavailable(RuntimeError)` (`get.py:4`) — the one exception every "this box can't
  serve it locally" refusal raises (missing llama-cpp-python, MoE-needs-a-slot, profile-not-ready,
  vision-needs-native); a worker propagates its message verbatim into central's `load_reports`.
- Engine resolution order (`resolve.py` docstring, 5 steps, §4-A) and the persisted record shape
  `{"dir":str,"server_bin":str}` under settings-JSON key `"engine"`.
- `ROW_KEYS = ("model","axis","ok","secs","uri","thumb_b64")` (`model_battery.py:60`) — the frozen
  battery row schema; `error`/`ts` are the only additive keys.

## 8. Gotchas, tech-debt & review findings

- △ `managers/generate/coder_guff.py` (`DeepCoderGGUF`) — **dead**: zero importers anywhere, not
  wired into `FRAMEWORK_RUNNERS`, superseded by `python_runner.py:LlamaCppPythonRunner`. Also
  hardcodes `n_gpu_layers=0` (line 23). Easy to mistake for "the" in-process GGUF lane; it isn't.
- △ `managers/generate/generate_runner2.py` — **dead**, confirmed (ARCHITECTURE.md's claim
  verified: zero importers, including tests). A near-duplicate of `generate_runner.py`'s
  `DeepCoderChatRunner` missing its unbounded-continuation / `attempt()` error-typing. A
  maintenance trap until deleted.
- △ `managers/llama/runners/src/ccp_runner.py:5-11` — a stray **module-level** `def __init__(self,
  model_key, *, env_path=None)` outside any class, directly above the real `LlamaCppRunner` (line
  48) whose own `__init__` (49-69) duplicates the same 4 lines. Never called (would `NameError` on
  `self` if it were) — refactor leftover, no runtime effect, purely confusing.
- ⚠ **No proactive engine-version-vs-model-architecture check in `engine/`.** `resolve.py:
  _spawn_probe()` (310-332) only proves the binary *executes* (`--version` succeeds); nothing
  compares its build against a GGUF's `general.architecture` (e.g. `qwen3next`) before spawning
  it — a too-old build just fails at llama-server startup/load with a native error, surfaced only
  as a slot-refusal reason. PROVISIONS.md tracks the fact that matters (binary hash `039e20a`
  "supports qwen3next") as **operator knowledge in a doc**, not a value this code checks. Flag
  support IS probed (`--n-cpu-moe`/`_server_supports_flag`, `managers/serve/slot_agent.py:374-387`,
  cached `--help` grep, graceful degrade), but that's a narrow flag probe in a sibling subsystem,
  not an arch-compat check here.
- ⚠ **`supports_gpu_offload` is checked for fleet placement, not at this subsystem's own load call
  site.** `llama_cpp.llama_supports_gpu_offload()` is queried/reported upward in
  `worker_agent/agent.py:194` / `gguf_worker/agent.py:204`, and central's worker-selection skips a
  worker that affirmatively can't offload (`flask_app/app/functions/imports/utils/workers.py:
  1020-1025`) — but neither `python_runner.py:LlamaCppPythonRunner.__init__` nor the (dead)
  `coder_guff.py:DeepCoderGGUF` calls it before handing `n_gpu_layers` to `Llama(...)`, so a
  CPU-only llama-cpp-python wheel serving in-process outside central's capability-aware scheduling
  (e.g. a plain single-box `hugpy serve`) gets no local warning its GPU-offload request was a no-op.
- ℹ The in-process llama-cpp-python lane **cannot express MoE expert offload at all** —
  `get.py:413-416` ("llama-cpp-python exposes no such parameter"), confirmed by
  `python_runner.py:166-173`'s `Llama(...)` call (no `n_cpu_moe` kwarg on the binding).
  `_build_runner()` detects this (`spill.gguf_moe_detail`) and **refuses** the in-process fallback
  for a real MoE GGUF (410-459) rather than silently loading all experts onto the GPU — deliberate
  (operator ruling 2026-07-26), not a bug. The "native slot lane" isn't monolithic either: with no
  native binary resolved on a box (`managers/serve/slot_agent.py:1121-1180`, "typical for
  WORKERS"), the slot spawns `python -m llama_cpp.server` instead, inheriting the SAME
  no-`--n-cpu-moe` limit (1142-1156) and the SAME vision/`--mmproj` limit (1129-1141, refuses
  rather than seating a model text-blind) — so an MoE model needs a *binary*-backed native slot
  specifically, not merely "a slot." A slot's `child_kind` field (`"binary"` vs `"python"`)
  records which.
- ℹ `managers/spill.py:1602-1634 _binding_supports_rpc()` — llama-cpp-python **dropped the
  `rpc_servers` constructor param in its 0.3.x rewrite** (present 0.2.78–0.2.90 only);
  `llama_kwargs()` detects this via `inspect.signature` and silently drops a configured shard plan
  back to local-only loading (one warning, never raises) — exactly why `shard_server.py` spawns a
  managed `llama-server --rpc` subprocess instead of the in-process binding. Re-check on any
  `llama-cpp-python` version bump.
- △ `model_battery.py:53 FALLBACK_ROOT = "/mnt/llm_storage/model-battery"` — **this path does not
  exist on this host** (verified; the real cold archive is `/mnt/16T_toshiba/llm_storage` per
  PROVISIONS.md). Inert today because `PRIMARY_ROOT` (`/srv/share/projects/hugpy/model-battery`)
  is normally writable, so the "try next root, else silently disable" design (257-271) masks the
  stale path — if the primary share is ever down, recording goes dark instead of actually falling
  back.
- ℹ **PROVISIONS.md's "coder-next" incident (2026-09-14, truncated hot-cache split-GGUF shard) did
  not go through this subsystem's verified-transfer code.** `model_sync.py`'s HTTP pull delegates
  entirely to `worker_agent/provision.py:ensure_model_present()`, which IS exact-size-checked and
  complete-or-raise; the incident's fix was a manual "re-copy cold→hot" outside any code in this
  doc's scope, with no completeness check here. `provisioner.py`'s own presence check
  (`_missing_reason()`, 132-137) is coarser still — ABSENT vs totally-ZERO-byte only — so a
  nonzero-but-truncated shard reads as present and is never re-wanted.
- △ `model_sync.py:64 pull_model()` calls `ensure_model_present(model_key, central_url,
  progress=progress)` with **no `state=`**. Per `worker_agent/provision.py:1498-1502` that omission
  is documented as intentional "pre-feature behavior (standalone/CLI provisions)", but it means
  every `model_sync --all` pull skips the worker's storage-budget FIFO-eviction/refusal check —
  no guardrail against filling a disk, unlike a real demand-triggered pull on a worker.
- △ `managers/llama/runners/get.py` carries two differently-scoped stale-cache recovery paths:
  `_slot_still_holds()` (145-157, a cheap `/status` GET before reusing a cached runner) vs
  `chat_runner.py:39-44 _stale_slot_error()` (substring-matches `"503"`/`"no model loaded"` in a
  free-text exception message). The second is fragile by construction — a llama-server wording
  change would silently regress the self-heal, reviving the "model sat on disk while calls died
  forever" symptom it exists to fix.

## 9. Deploy/run boundary

Central (`7002_hugpy_api`) now runs the **src** checkout via `PYTHONPATH` — every file in this
doc's scope is live on a `7002` restart after an edit, no wheel rebuild needed. Workers
(`worker_agent`, `gguf_worker`) run the **installed pip wheel** — an edit here reaches a worker
only after `signals/*.trigger` → wheel rebuild/publish → `pip install -U` + service restart on
that worker (`README.md §Deploy loop`). `hugpy install-engine` is a one-time-per-box provisioning
step independent of the package version — it fetches/builds a native binary that outlives any
single wheel upgrade; re-run it only when llama.cpp itself needs to move (e.g. a new GGUF
architecture). `sentinel`'s `hugpy-sentinel.timer` (10-minute cadence, `Type=oneshot`) picks up a
`provisioner.py` change on its very next fire — no restart needed, since every pass is a fresh
process. `HUGPY_MODEL_BATTERY`/`HUGPY_MODEL_BATTERY_ROOT`/`HUGPY_PROVISION_FLOOR_GB` and the other
env levers named above are read per-call, not cached at import.

