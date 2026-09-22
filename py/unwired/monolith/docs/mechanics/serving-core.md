<!-- Written by a per-subsystem code-review pass (2026-09-14). Follows _TEMPLATE.md. -->

# Serving core (load → slot → serve → evict + placement) — mechanics

**Scope (paths this doc covers):** `managers/serve/*` (`serve.py`, `supervisor.py`,
`slot_agent.py`, `slots.py`, `model_cache.py`, `hot_cache.py`, `policy.py`,
`profiles.py`, plus `overrides.py` for `worker_prefs`/polite-load),
`managers/eviction.py`, `managers/spill.py`, `managers/alloc_modes.py`,
`managers/draft_models.py`, `managers/dispatch/`, `managers/resolvers/`. Paths
relative to `src/abstract_hugpy_dev/src/abstract_hugpy_dev/` unless noted.
**One-liner:** resolves a request to a runner, seats a model (spawns/reuses a
native `llama-server` slot child or an in-process runner), decides WHERE its
bytes land under a 5-mode placement vocabulary, and reclaims GPU/RAM/disk room
under contention.
**Owner process(es):** central `7002_hugpy_api` (in-process `dispatch`/`resolvers`
+ the declarative `serve.py` drivers) and any worker's slot pool (`abstract-hugpy-
slot@<N>.service`, `python -m abstract_hugpy_dev.managers.serve.slot_agent`);
`eviction.py`/`alloc_modes.py`/`spill.py`/`allocator.py` are pure libraries with
no process of their own.

## 1. Purpose & responsibilities
- One routing decision point (`resolvers.resolve`) turns a request into a
  `runner_cls`: force-local (loop guard), a pinned peer, or a worker-first/
  local-fallback `DelegatingRunner`.
- Seat a GGUF model: `SlotPool` schedules a fixed pool of slot children over
  plain HTTP (no root/systemctl at request time); each slot spawns/proxies ONE
  native `llama-server` (or a `llama_cpp.server` python fallback), autofitting
  GPU layers and the MoE expert split to the VRAM free right now.
- Own the placement VOCABULARY and wire CONTRACT (five alloc modes, MoE
  dense-backbone-first policy, `worker_prefs` hard-scope, polite/`no_evict`
  loads) that central's worker registry persists and relays — this doc's
  modules define the contract; the registry itself is comms-index.md's.
- Reclaim room under contention — but this is FOUR independent mechanisms
  (slot-pool LRU, in-process runner-cache LRU, a pure VRAM/RAM admission
  planner, and an NVMe hot-cache LRU), not one; see §5.
- Does **not** own the worker registry, heartbeat DB, or worker ranking, and
  does not decide model→task→framework routing beyond `resolve()` itself —
  both are documented as deliberate centralization, not a gap.

## 2. Key modules (file → responsibility)
| file | responsibility |
|---|---|
| `serve/serve.py` | Declarative `ServeSpec`/`ServeMode` (off\|systemd\|supervised\|swap) planner + driver registry; `serve_endpoint()` is the generic slot-first/swap-fallback resolver |
| `serve/slots.py` | `SlotPool` — the stateless HTTP scheduler over N slot children: reuse → ceiling-gate/evict → idle-load → promote → static-refuse |
| `serve/slot_agent.py` | The slot supervisor process itself: `Slot` owns one `llama-server` child; `_build_cmd` is the sole argv choke point (path/hot-cache, autofit, MoE split); Flask app exposes `/load /unload /relaunch /v1/<sub>` |
| `serve/supervisor.py` | Portable (non-systemd) always-on driver: detached `llama-server` + JSON pidfile, for OSes without systemd |
| `serve/model_cache.py` | Legacy shared→local model-file cache; superseded by hot_cache, kept as fallback |
| `serve/hot_cache.py` | NVMe hot-tier LRU cache of the shared model catalog: read-through, async promote-on-use, its own disk eviction |
| `serve/policy.py` | `HUGPY_NO_LOCAL_SERVING` per-box kill switch, read at call time (never cached) |
| `serve/profiles.py` | Per-model dependency-isolation venvs; hands the slot child's python/PATH |
| `serve/overrides.py` | Persisted per-model serve overrides (`serve_overrides.json`): `alloc_mode`, `n_cpu_moe`, `worker_prefs` (hard scope), `no_evict`/`no_evict_by_worker` |
| `eviction.py` | Pure Box1/Box2 admission+eviction planner (`plan_admission_split`/`evict_plan`) — shared algorithm, mostly consumed outside this scope |
| `spill.py` | VRAM/RAM measurement + autofit: `autofit_gpu_layers`, `gguf_moe_detail`/`moe_dense_first_plan`, env readers (`alloc_mode_env`, `n_cpu_moe_env`, `no_evict_env`) |
| `alloc_modes.py` | The five-mode vocabulary: `default_allocation` (structure-derived tree), `mode_to_spill`/`normalize_spill`, `gate_spill_for_worker` (version gate) |
| `draft_models.py` | Flags speculative-decoding DRAFT ggufs so the console/loader refuse to serve them standalone |
| `dispatch/dispatch.py` | Per-process runner cache (`_INSTANCES`); `execute_prompt`/`runner_for`; in-process contention-LRU eviction (`ensure_headroom_for_load`) |
| `dispatch/acquire.py`, `dispatch/activity.py` | Peripheral: registers a downloaded model into the JSON overlay; a thin shim over `comms.jobs` for the console's live queue view — neither is on the seat/evict path |
| `resolvers/model_resolver.py` | `resolve()` — the one routing decision point: force-local / peer / delegating-runner |
| `resolvers/remote.py` | `make_delegating_runner`/`DelegatingRunner` (worker-first, local-fallback), `_select`/worker-provider hooks, model-group member key |
| `resolvers/allocator.py` | Pure multi-GPU fleet placement (whole/shard/cpu/none) for RPC-sharded loads |
| `resolvers/assure_model_key.py`, `resolvers/model_dict_resolver.py` | Peripheral: fuzzy model-key resolution against the registry; HF/local metadata enrichment for the models-dict overlay — lookup/registration, not the seat/evict path |
| `resolvers/groups.py` | Model groups: quality/speed/priority ticks, ladder walk, member selection |

## 3. Entry points
- Python API: `dispatch.execute_prompt(...)` / `dispatch.runner_for(model_key=, task=)`
  (`dispatch/dispatch.py:602-638`); `resolvers.resolve(prompt_kwargs)`
  (`resolvers/model_resolver.py:262`); `serve.serve_endpoint(model_key)`
  (`serve/serve.py:757`); `slots.SlotPool().endpoint_for(model_key, opts=…)`
  (`serve/slots.py:244`).
- HTTP, per slot child (its own Flask app, `build_app()`, `serve/slot_agent.py:1764`):
  `GET /health /status /identity`, `POST /load /unload /relaunch`,
  `POST|GET /v1/<sub>` (OpenAI-shaped proxy). Central admin routes also hit a
  slot directly, e.g. `worker_routes.py:5442`.
- Process entry: `python -m abstract_hugpy_dev.managers.serve.slot_agent`
  (`serve/slot_agent.py:1938 main()`), one per `abstract-hugpy-slot@<N>.service`
  instance (systemd template rendered by `serve/slots.py:508 render_slot_unit`).
- CLI: none of its own; reached transitively through `hugpy serve` / `hugpy worker`.

## 4. Data flow (the spine)

**A — request → resolve → seat → serve (the GGUF/llama.cpp path)**
1. `dispatch.execute_prompt`/`runner_for` calls `resolvers.resolve(prompt_kwargs)`
   (`resolvers/model_resolver.py:262`), which picks `(model_key, framework, task)`
   and a `runner_cls` in this precedence (`model_resolver.py:330-346`):
   `_force_local` (loop guard — this box IS the assigned worker executing a
   relayed request) → a `placement.json`-pinned peer → else
   `make_delegating_runner(framework, task)` (`resolvers/remote.py:2249`).
2. `DelegatingRunner.run`/`.stream` (`remote.py:2311` / `2528`) asks the
   externally-registered worker provider (`_select`, `remote.py:437`, wired to
   hooks set outside this scope — comms-index.md) for a live worker and
   relays; with none eligible, or when force-local already applies, it falls
   back to `self._local_runner()` (`remote.py:2306`) — for llama.cpp/chat that
   class is `chat_runner.py`'s `LlamaCppChatRunner` (engine-generation.md),
   whose `.runner` property lazily calls `get_llama_runner(model_key)` → … →
   `SlotPool.endpoint_for` (lands back in this scope at `serve/slots.py:244`).
3. `SlotPool.endpoint_for` (`serve/slots.py:244-493`): (1) a slot already
   serving/loading the model is reused, waiting out an in-flight load rather
   than proxying to a not-yet-healthy child (`slots.py:260-283`); (2) the
   real-VRAM ceiling gate (`slots.py:285-414`, driven by the `_FIT_CHECK`/
   `_MAKE_ROOM` hooks at `slots.py:40-54,76-82`) may evict the coldest
   on-demand occupant or admit a partial/MoE-split plan; (3) an idle slot
   loads it via `POST <slot>/load` (`slots.py:416-425`); (4) all slots busy →
   tiers-v2 promotion bumps the LRU idle on-demand occupant (`slots.py:427-473`);
   (5) every occupant 🔒static → fails loudly rather than a misleading `None`
   (`slots.py:481-492`).
4. The slot's own `/load` handler (`serve/slot_agent.py:1789-1828`) calls
   `Slot.load` (`slot_agent.py:1454`) → `_build_cmd` (`slot_agent.py:597-1180`):
   resolves the GGUF path (hot-cache promotion via `hot_cache.use`,
   `slot_agent.py:700-706`), autofits `n_gpu_layers` from free VRAM + the ctx
   KV-cache reserve (`slot_agent.py:708-745`), decides the MoE `--n-cpu-moe`
   split dense-backbone-first (`slot_agent.py:757-1049` — see Flow B), and
   `subprocess.Popen`s the native `llama-server` (or the `llama_cpp.server`
   python fallback) on `SLOT_CHILD_PORT = SLOT_PORT + 1000`
   (`slot_agent.py:53`, argv built at `1090-1096`/`1167-1173`, spawned at `1534`).
5. `Slot._wait_healthy` blocks the `/load` HTTP call until the child answers,
   or fails it on STALL/hard-cap with exponential backoff per model
   (`slot_agent.py:1543-1599`); once healthy the slot proxies `/v1/<sub>` to the
   child for the life of the seat (`slot_agent.py:1855-1933`) — this proxy call
   is the actual "seat," not the `/load` HTTP response alone (see §8).

**B — the placement contract, operator pick → argv**
1. An operator picks one of five modes — `gpu-only` / `ram-only` / `max-gpu`
   (default) / `max-ram` / `explicit` (`alloc_modes.py:1-83` names the wire
   encoding for each) — via the console; `mode_to_spill`/`normalize_spill`
   (`alloc_modes.py:1024-1139`) persist it into `spill_by_model[model_key]`
   (central's worker registry, out of scope — comms-index.md).
2. Per request, central emits the wire spill through `WorkerStore.spill_for` →
   `_placement_spill_for` (`workers.py:3801`, `:3849` — cross-edge; the
   contract keys `_PLACEMENT_SPILL_KEYS` at `workers.py:1115-1124` mirror
   `alloc_modes`' key families). A **blank** spill (no key intersects
   `_PLACEMENT_SPILL_KEYS`, checked at `workers.py:3881`) is re-derived by
   `alloc_modes.default_allocation` (`alloc_modes.py:534-640`, structure-aware:
   transformers vs GGUF vs MoE non-expert/expert split). A **pinned** spill
   (any placement key already persisted) returns close to verbatim
   (`workers.py:3919-3941`, after `gate_spill_for_worker` version-gates the
   newer keys, `alloc_modes.py:1214-1242`).
3. On the worker the spill becomes env (`spill.alloc_mode_env`/`n_cpu_moe_env`)
   and/or per-load opts threaded into `Slot.load(..., alloc_mode=, n_cpu_moe=)`
   (`slot_agent.py:1457-1471` projects `alloc_mode` into
   `os.environ["HUGPY_ALLOC_MODE"]` for the process, single-flight under the
   slot's lock). `_build_cmd` (`slot_agent.py:757-798`) honors an EXPLICIT
   `n_cpu_moe` absolutely; absent that, a detected MoE gets the AUTO
   dense-backbone-first split (`spill.moe_dense_first_plan`, `spill.py:885`) —
   every mode that grants any GPU budget spends it on the always-hot backbone
   FIRST, experts get the remainder (`MOE_ALL_LAYERS = 999`, `spill.py:547`,
   for "experts entirely on CPU").
4. `worker_prefs` (`serve/overrides.py:61-71`, resolved by `placement_policy`
   at `overrides.py:833-874`) is a **hard scope**, not a soft rank: central's
   `workers_for_model` (`workers.py:3984`, `:4361-4367` — cross-edge) filters
   candidates down to the listed workers, tried in order, and refuses outright
   if none is eligible right now — it never falls back to an unlisted worker.

**C — eviction.** Four independent mechanisms trigger under different
conditions and are gated by different hooks; see §5 (this is intentionally not
folded into A/B — conflating them is the single easiest way to misread this
subsystem).

## 5. State, persistence & invariants
- `SlotPool` holds no persisted state of its own — a stateless HTTP scheduler
  over N slot children. Each `Slot` (`slot_agent.py:1206`) is one live
  process's in-memory state (`model_key`, `ngl`, `ctx`, `n_cpu_moe`,
  per-model load-failure/backoff dict, `threading.Lock`).
- `dispatch._INSTANCES`/`_BUILDING` (`dispatch/dispatch.py:81-90`) is the
  process-wide in-memory runner cache keyed by `(model_key, task)`; residency
  policy is CONTENTION-based LRU (doctrine 2026-07-11, `dispatch.py:100-117`),
  explicitly **not** a TTL clock — an on-demand model stays resident until
  another load needs the room.
- Disk: `serve_overrides.json` under `PROJECTS_HOME` (`serve/overrides.py:30`)
  is the UI-writable per-model knob overlay; `hot_cache.py` keeps its own JSON
  index (`hot_cache.py:428-486`) plus a `.part`-then-rename copy protocol for
  atomicity (`hot_cache.py:28-32`).
- **Four independent eviction layers** — easy to conflate, gated by different
  hooks, wired together only by the worker agent's hook registration at boot
  (worker-fleet.md, out of scope); none of the four calls another directly:
  1. `SlotPool` LRU among SLOT occupants only (`_evict_coldest_on_demand` /
     promotion, `slots.py:195-224,427-473`) — gated by `slots.py`'s own
     `_EVICTION_POLICY`/`_RESIDENCY_LOOKUP`/`_FIT_CHECK`/`_MAKE_ROOM` (`slots.py:27-89`).
  2. `dispatch.py`'s in-process `_INSTANCES` contention yield
     (`ensure_headroom_for_load`/`_headroom_pass`, `dispatch.py:338-513`) —
     gated by `dispatch.py`'s own, differently-scoped `_FIT_CHECK`/`_EVICTABLE`/
     `_MAKE_ROOM`/`_POST_EVICT` (`dispatch.py:153-168`).
  3. `eviction.py`'s pure Box1/Box2 VRAM+RAM admission planner
     (`plan_admission_split`/`evict_plan`, `eviction.py:399-469,547-578`) — a
     shared algorithm, consumed mainly outside this scope.
  4. `hot_cache.py`'s disk-tier LRU (`_make_room`/`_evict_locked`,
     `hot_cache.py:614-659,553-567`) — evicts NVMe hot-copies only, never
     touches VRAM/RAM or the shared array.
- Invariant: one slot child per control port; the child always lives at
  `SLOT_CHILD_PORT = SLOT_PORT + 1000` (`slot_agent.py:53`).
- Invariant: `policy.no_local_serving()` is a per-box kill switch checked at
  THREE independent layers, defense-in-depth — `slots.py:110-119,133-139`,
  `slot_agent.py:1799-1803`, `serve.py:769-771`.
- Invariant: a polite load (`no_evict` spill key / `no_makeroom` request flag)
  never evicts — it lands only in genuinely free room or fails fast with
  `LoadRefusal` (`dispatch.py:134-141`; `slots.py:226-330` implements the same
  contract for the slot pool).

## 6. Cross-subsystem edges
**→ calls (out of this doc's scope):** `managers/llama/runners/get.py`
(`get_llama_runner`/`_build_runner`, engine-generation.md) is the primary
opts-aware caller into `SlotPool.endpoint_for`; `worker_agent/agent.py`
(worker-fleet.md) registers every hook this doc's modules expose at boot and
owns `_materialize`/`_probe_model`, the function that actually forces a lazy
runner resident (see §8); `flask_app/.../workers.py` (comms-index.md) owns
`spill_by_model`/`moe_by_model`/`_placement_spill_for`/`workers_for_model` —
the registry this doc's contract keys are written into and read from.
**← called by:** `worker_routes.py` (admin routes hit `SlotPool`/`Slot`
directly); `v1_routes.py`/`chat_routes.py` (api-routes.md) via `execute_prompt`.
**Siblings:** [[engine-generation]] (the llama-runner singleton this doc's
`SlotPool` ultimately serves), [[worker-fleet]] (hook registration,
`_materialize`/probe warm path, the HTTP relay `DelegatingRunner` targets),
[[comms-index]] (the worker registry, `spill_by_model`/`moe_by_model`
persistence, `workers_for_model` ranking — §4 Flow B's central half),
[[media-gen]] (`DelegatingRunner` also carries image/video/vision tasks
through this same worker-selection machinery).

## 7. Key contracts / types
- `Resolution` / `Runner` protocol (imported from `resolvers`, cache key
  `(model_key, task)`) — every runner this doc builds implements it.
- `ServeMode` enum (`serve.py:131`): `off | systemd | supervised | swap`.
- `ServeSpec` dataclass (`serve.py:275`) — resolved per-model serve config for
  the non-slot drivers.
- Slot `/load` POST body: `{model_key, n_gpu_layers?, ctx?, threads?, cpus?,
  gpu?, path?, gpu_mem_gib?, cpu_mem_gib?, profile_bin?, n_cpu_moe?,
  alloc_mode?, ngl_defaulted?}` (`slot_agent.py:1790-1826`); `/status` and
  `/identity` response shapes at `slot_agent.py:1394` and `:1779`.
- `_PLACEMENT_SPILL_KEYS` (cross-edge, `workers.py:1115-1124`): the frozenset
  deciding "persisted placement" vs "blank" — `alloc_mode`, `n_gpu_layers`,
  `leniency_pct`, `priority`, `priority_device`, `gpu_mem_gib`, `cpu_mem_gib`,
  `*_deviation_pct`, `tensor_split`, `threads`, `n_cpu_moe`.
- Five alloc modes + frozen wire encoding (`alloc_modes.py:41-73`), e.g.
  `gpu-only → {"n_gpu_layers": -1}`, `max-gpu → {}` (derived) or
  `{"alloc_mode": "max-gpu"}` (explicit pick — provenance, not behavior,
  differs).
- `EvictUnit` / `EvictPlan` / `Placement` dataclasses (`eviction.py:173-256`,
  `476-510`) — the pure admission-planner's vocabulary.
- `Node` / `Need` / `Placement` dataclasses (`allocator.py:31-66`) — the fleet
  multi-GPU sharding vocabulary (name-collides with `eviction.Placement`; see §8).

## 8. Gotchas, tech-debt & review findings
- ⚠ **MoE `moe_by_model` operator override vs a pinned `spill_by_model`
  bypass.** `_placement_spill_for` (`workers.py:3849-3941`) only threads
  `moe_override(worker, model_key)` (`workers.py:1765`) into the emitted spill
  via `derived_default_allocation` (`workers.py:1865-1887`, which passes
  `moe_force=moe_override(...)`) on the **blank**-spill branch
  (`workers.py:3881`). The instant ANY placement key is already persisted in
  `spill_by_model[model_key]` — e.g. an operator pinned `alloc_mode`,
  `priority`, or `threads`, none of them MoE-related — `_placement_spill_for`
  takes the "PERSISTENCE-ONLY MODES" branch (`workers.py:3919-3941`) and
  returns the raw persisted spill essentially verbatim; `moe_override`/
  `moe_by_model` is **never consulted there**. Meanwhile `moe_effective`
  (`workers.py:1785-1793`), which feeds the console's `moe_effective` map
  (`workers.py:792`, inside `_public_view`), DOES report the operator's forced
  checkbox state. Net effect: the console can render "MoE split: ON" for a
  (worker, model) whose emitted `--n-cpu-moe` never reflects that override,
  because some unrelated placement key already happens to be pinned.
  `n_cpu_moe` itself IS in `_PLACEMENT_SPILL_KEYS` (the k67 fix,
  `workers.py:1119-1124`) — but that only helps once the persisted spill
  *already* carries an explicit `n_cpu_moe`; a `moe_by_model`-only toggle on a
  model with one other pinned key is silently dropped on the wire.
- ⚠ **`DelegatingRunner` has no `ensure_loaded` — "warm" can be a no-op for
  non-GGUF tasks.** `DelegatingRunner` (`resolvers/remote.py:2275-2311`)
  defines `__init__`, `_local_runner`, `run`, `stream` — no `ensure_loaded`.
  Contrast `managers/llama/runners/chat_runner.py:79-90`'s
  `LlamaCppChatRunner.ensure_loaded`, whose docstring names the exact contract
  this is missing: *"`__init__` and `runner_for()` build only this lazy
  wrapper... Warm / slot-fill / probe paths call this so the model actually
  becomes resident + slot-seated instead of a hollow shell that still
  registers as 'loaded'."* `worker_agent/agent.py:1248-1260 _materialize()`
  duck-types `getattr(runner, "ensure_loaded", None)` and silently no-ops when
  absent (documented as intentional there). So `_probe_model`
  (`worker_agent/agent.py:4791`, the implementation behind central's warm
  sweep / the worker's `/probe` route) forces real residency for GGUF/chat
  models via that method (`agent.py:4842-4847`, its own comment: *"runner_for
  only BUILDS the (lazy) wrapper... Force the underlying runner resident"*)
  but is a confirmed **silent no-op** for any task/framework resolved to a
  bare `DelegatingRunner` — image-gen, video-gen, vision-via-worker-pool,
  anything through `make_delegating_runner` (`remote.py:2249`) that isn't the
  llama-chat local fallback. Such a model's first REAL request is what
  actually seats it (`DelegatingRunner.run`, `remote.py:2311`); a "warm" call
  against it returns success having loaded nothing.
- △ **Two identically-named but independent hook registries.**
  `slots.py`'s module globals `_FIT_CHECK`/`_MAKE_ROOM`/`_EVICTION_POLICY`/
  `_RESIDENCY_LOOKUP` (`slots.py:27-89`) and `dispatch.py`'s own
  `_FIT_CHECK`/`_MAKE_ROOM`/`_EVICTABLE`/`_POST_EVICT` (`dispatch.py:153-168`)
  gate two *different* residency layers (the GPU slot pool vs. the in-process
  `_INSTANCES` cache), both populated by the worker agent at boot — reading
  only one file's hooks gives an incomplete picture of "what can evict what."
- △ **`serve_endpoint()` seats without opts.** The generic/declarative
  `serve_endpoint()` (`serve.py:757-803`) calls `pool.endpoint_for(model_key)`
  with **no** `opts` (`serve.py:794`), unlike `managers/llama/runners/get.py:376`
  (out of scope), which passes `opts` derived from spill/env. A model reached
  only through `serve_endpoint()` seats via bare autofit, silently bypassing
  any `alloc_mode`/`n_cpu_moe` override — worth confirming which live callers
  actually reach `serve_endpoint()` before relying on it for a non-default model.
- △ **`Placement` name collision.** `eviction.Placement` (`eviction.py:476`)
  and `resolvers.allocator.Placement` (`allocator.py:54`) are unrelated
  dataclasses sharing a name in the same subsystem doc — importing both
  unqualified in one module would silently shadow.
- ℹ **The `-ngl`/`--fit` spelling migration is load-bearing, not cosmetic.**
  A `--fit`-capable `llama-server` binary gets `-1` translated to the literal
  string `"all"` plus an explicit `--fit off` (`slot_agent.py:1073-1096`),
  because llama-server's own `--fit` (default ON) would otherwise silently
  hand placement back to llama.cpp's internal autofit and discard this
  module's dense-backbone-first plan — a bare engine binary upgrade alone
  could have silently regressed MoE placement before this translation existed.
- ℹ **Load-failure classification matters for retries.** `Slot.load`
  (`slot_agent.py:1554-1597`) distinguishes a signaled crash (OOM/SIGSEGV —
  state-dependent, never permanently cached) from a clean nonzero exit
  (permanent — matched by central's `_PERMANENT_LOAD_MARKERS`, out of scope)
  from a stall/hard-cap timeout; debugging "why did this model fail to load"
  should read `last_load_error`'s wording, not the raw exit code.

## 9. Deploy/run boundary
Central (`7002_hugpy_api`) now runs the **src** checkout via `PYTHONPATH`, so a
source edit under `managers/serve|dispatch|resolvers` and
`managers/{eviction,spill,alloc_modes,draft_models}.py` is live on the next
`7002_hugpy_api` restart (`SERVICES.md`). Slot children are **separate,
long-running processes** (`abstract-hugpy-slot@<N>.service`) that only pick up
a `slot_agent.py`/`serve.py`/`spill.py`/`alloc_modes.py` change on their own
restart — editing `_build_cmd`'s MoE policy does nothing to an already-spawned
slot until `systemctl restart abstract-hugpy-slot@N`. Workers run the
pip-installed wheel, not this src tree, so an edit here reaches a worker box
only after `signals/*.trigger` → wheel rebuild/publish → `pip install -U` +
worker-agent restart (`README.md §Deploy loop`). `serve_overrides.json`, the
hot-cache JSON index, and slot-supervisor pidfiles are runtime data, not code —
untouched by any of the above and safe to inspect live.
