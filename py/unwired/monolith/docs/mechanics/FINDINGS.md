# hugpy code-review — consolidated findings (2026-09-14)

Output of the 10-agent mechanics review. Each finding is `file:line`-cited in its subsystem
doc (`docs/mechanics/<name>.md §8`). Paths under `src/abstract_hugpy_dev/src/abstract_hugpy_dev/`
unless noted. Ranked: **security → correctness → tech-debt**. This is an audit snapshot; verify
before acting (central runs the **src** checkout now, so a src fix is live on `7002` restart).

## ⚠⚠ Security (unauth / missing ownership) — highest priority
| # | finding | where | why it matters |
|---|---|---|---|
| S1 | Worker `moe`/`bnb` writes are **zero-auth** (docstrings falsely claim operator-gated) | `flask_app/app/routes/worker_routes.py:1224,1277` | anyone on the network can flip a worker's MoE/bnb config |
| S2 | `cache-evict` / `auto-reap` **zero-auth** — missed by `operator_auth._SENSITIVE` regex | `worker_routes.py:2305,3499` | unauth eviction/reap of a worker's models |
| S3 | Movie-session pause/resume/list have **no per-account owner check** | `video_routes.py:1618-1829` | any member can control another's movie sessions |
| S4 | Entire identity-profiles cluster (21 routes incl. **DELETE**) has **no owner check** | `video_routes.py:4787-6017` | cross-account read/delete of identity profiles |
| S5 | `POST /prompt` and `/finder/search` reachable with weaker/any-scope auth than siblings | `flask_app/app/routes/` (prompt/search) | privilege inconsistency |

## ⚠ Correctness / robustness
| # | finding | where | why it matters |
|---|---|---|---|
| C1 | **Alloc auto-defaults to `max-gpu`** for everything; a persisted `{"alloc_mode":"max-gpu"}` *suppresses* the auto MoE split. Fits-VRAM should be `gpu-only`, MoE `explicit`. | `managers/alloc_modes.py`, `workers.py` `_placement_spill_for`/`_effective_alloc_mode` | models that fit get spilled to CPU instead of evicting+using GPU (the grading-CPU-starvation cause) |
| C2 | **MoE checkbox bypass**: `moe_by_model` only consulted when `spill_by_model` is blank | `workers.py:3849-3941` (gate at :3881) | UI shows "MoE: ON" while the wire never sends `--n-cpu-moe` for any pinned model |
| C3 | `DelegatingRunner` has **no `ensure_loaded`**; `/load`-warm silently no-ops | `managers/resolvers/remote.py:2275-2311`, `worker_agent/agent.py:1248` | image/video/vision "warm" is a lie — only a real inference seats them |
| C4 | Metrics store fails **silently** — bare `except` with zero logging in a 492-line module; both PG + SQLite arms dead for days | `comms/model_metrics.py:637-642`, `abstract_toolserver/metrics.py:81-145` | `#metrics` empties with no diagnosable cause (role grant is NOT the trigger) |
| C5 | **`WORKER_PORT` vs slot-child ports never cross-validated** (`SLOT_PORT_BASE`+1000) | `worker_agent/` + `managers/serve/slot_agent.py:53` | the exact collision that stopped `aeb` seating (WORKER_PORT 9101 == slot1 child 9101) |
| C6 | No check of native `llama-server` build vs GGUF `general.architecture` before spawn | `engine/` / `managers/serve/slot_agent.py` `_build_cmd` | a too-old engine silently fails to load a newer arch (the coder-next dead-end class) |
| C7 | `supports_gpu_offload` gates fleet placement but is **unchecked at in-process load** | `managers/llama/runners/src/python_runner.py` | a CPU-only llama-cpp-python silently no-ops a GPU request on a standalone `hugpy serve` |
| C8 | `media_bus` sqlite corruption is **uncaught** (only `OperationalError` handled), no global error handler | `video_intel/media_bus.py` | a malformed `media_jobs.db` 500s the whole video-job surface, no auto-recovery |
| C9 | Nightly `review/` downloads weights **in-process, bypassing** `downloader/` stall-killer/resume | `review/pipeline.py:68-87` | a wedged HF connection hangs the 4-hour unattended run with no self-heal |
| C10 | Dossier default `trial_depth="load-test"` adopts/rejects with **zero scored samples** | `discovery_dossier/dossier.py:476-481` | verdicts contradict the module's own "NO EVIDENCE, NO VERDICT" rule |
| C11 | `hugpy keeper` is **broken** — relative-import crash on both documented invocations | `keeper.py:38` (via `cli.py:403-409`) | the keeper REPL doesn't start; only `python -m …keeper` works (undocumented) |
| C12 | `_compat_pydantic.BaseModel` lacks `model_copy` → `AttributeError` where the shim is active | `utils/text/combined.py:131,172,186` (+5 managers, 2 flask_app) | the "shim silently used" failure class (bit the ae worker) |
| C13 | `install.py --serve-mode` writes `"on"`, not in the `ServeMode` enum (`off\|systemd\|supervised\|swap`) | `worker_agent/install.py` | `ValueError` risk; live `aeb` unit already hand-patches around it |
| C14 | `model_sync --all` calls `ensure_model_present()` without `state=` → bypasses the storage-budget FIFO eviction guard | `model_sync.py:64` | a bulk sync can blow the worker's disk budget |
| C15 | `/v1/chat/completions` + `/v1/messages` create a `job_store` row with **no cancel handle** | `v1_routes.py` | a cancel force-marks the row terminal without stopping the actual generation |

## △ Tech-debt / cleanliness
- **Dead code** (zero importers): `managers/falconsai/falconsai_module.py` (394 lines), `managers/summarizers/generation.py:GeneratorManager`, `managers/vision/utils.py:fit_to_token_budget`, `managers/generate/coder_guff.py:DeepCoderGGUF`, `managers/generate/generate_runner2.py`.
- `imports/` is a flat star-import waterfall with almost no `__all__`; `abstract_identity` is a hard pyproject dep with **zero import sites**.
- `hugpy-todo-keeper.service` runs the pip **wheel 0.1.251**, not src — a `todo_keeper*.py` src edit isn't live on central restart (unlike other comms consumers).
- `ApiPrefixMiddleware` (installed unconditionally) likely makes ~10 explicitly `/api`-dual-mounted blueprints **dead code**.
- Console→prod deploy is a **manual, unautomated rsync** ("step 0c, NO JOB DOES THIS"); media/video UI arms fail silently on perm-locked `dist/`; 3 docs disagree on `deploy-ui.trigger`'s target.
- `gguf_worker` has no enroll-token auth; `phone_brick` keeps advertising `role=rpc` after its RPC subprocess fails to spawn, and its NAT-hairpin LAN-advertise helper is a buggy unshared copy (missing the loopback-reject guard).
- Four independent eviction layers (SlotPool LRU, dispatch contention-LRU, `eviction.py` planner, `hot_cache.py` disk-tier) wired only at worker boot — high cognitive load.
- `oracle/interim_ledger.py:1544,1561` likely stores a Python-repr'd dict; its `gap` keys don't exist on a dossier → blocked dossiers never surface as gaps.
- **Doc drift:** top-level `README.md:13` still says central runs the pip wheel (it runs src now); `serve.py:794` `serve_endpoint()` seats without threading `opts`.

## ℹ Confirmed NOT bugs (disambiguated during review)
- `oracle/pipeline.py` (framework) vs `pipelines.py` (concrete) — distinct-live. `repair.py` (single-route) vs `repair_controller.py` (DAG) — distinct-live. `spatial.py`/`spatial_sources.py`/`spatial_eval.py` — a 3-stage pipeline, not dupes.
- The real in-process GGUF lane is `managers/llama/runners/src/python_runner.py:LlamaCppPythonRunner` (the two dead ones above are not it).
- `worker_prefs` is a genuine **hard scope** (refuses off-list), not a soft ranking — `workers.py:2814-2827`.
