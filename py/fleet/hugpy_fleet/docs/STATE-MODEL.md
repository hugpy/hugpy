# Model State Nomenclature — Canonical Reference

**Status:** canonical vocabulary (target state). Some producers/consumers do not yet
conform; every known divergence is listed in the **Divergence Register** below with a
`file:line` and a resolution. New code MUST use these terms; drift against this doc is a
bug in the code, not in the doc.

**Why this exists:** the same words (`cold`, `hot`, `warm`, `serving`, `resident`,
`loaded`, `in_flight`) were used for different things in different subsystems, which
produced real incidents — e.g. an already-on-disk model refused as if it needed a
multi‑GB download (the cold‑hold `cold_load_capacity` refusal). This doc fixes the
meanings.

---

## The one rule

**A term names exactly one thing on exactly one layer.**

- **Layer 1 — Residency state**: *where the weights physically are* (a thermal ladder).
- **Layer 2 — Cost model (ETG)**: *how long until this candidate produces output* (times).
- **Layer 3 — Policy / eligibility**: *may we, and is it even resolvable* (boolean gates).

State words are nouns for a location. Cost words always carry a time unit (`t_*`, seconds).
Policy words are adjectives that filter candidates and never enter a cost sum.

---

## Layer 1 — Residency state (the thermal ladder)

"Temperature" = **proximity to GPU compute**. The colder a model is, the more work stands
between it and producing a token. Per `(worker, model, variant)` the model occupies exactly
one state:

| State | Meaning | Physical location | Authoritative signal |
|---|---|---|---|
| `UNAVAILABLE` | Not resolvable anywhere — not on the fleet source-of-truth nor a remote origin (HF). Cannot be provisioned. **Off the ladder: no worker delegation.** | nowhere fetchable | `serveable=False` + `unserveable_reason` |
| `COLD` | Resolvable on the **source-of-truth** array / origin, but **not on this worker's hot drive**. Must be pulled before it can load. | `MODELS_HOME` (`/mnt/llm_storage/models`) / HF only | resolvable **and** NOT in worker `models_local` |
| `HOT` | Present on this worker's **hot drive** — the box-local NVMe it loads into VRAM from. Downloaded, ready to load. | `HUGPY_HOT_CACHE_ROOT` (box-local NVMe hot-cache) | in worker `models_local` (disk-truth) |
| `LOADED` | Weights **measured resident** in VRAM, idle (not answering a request right now). | GPU VRAM | heartbeat allocation row `healthy && materialized`, not `serving`/`busy` |
| `SERVING` | Resident in VRAM **and actively answering** a request (within the active window). | GPU VRAM | allocation row `serving` (busy or last_used within `_SERVING_WINDOW_S`) |

### Transitions (each is the unit of work Layer 2 prices)

| Transition | From → To | Name | In-flight state | Signal |
|---|---|---|---|---|
| Download | COLD → HOT | **pull** (fetch) | `PULLING` | `provisioning` / `provision_progress` |
| Load | HOT → LOADED | **load** | `LOADING` (heating) | `loading` |
| Serve | LOADED → SERVING | — (a request arrives) | — | `serving` flips true |
| VRAM evict | LOADED/SERVING → HOT | **evict** | `EVICTING` | eviction plan; weights remain HOT on the drive |
| Hot-drive reap | HOT → COLD | **reap** | — | hot-cache FIFO by time-since-called (`hot_cache.py`) |

Notes:
- **evict ≠ reap.** Eviction frees **VRAM** (the model falls back to `HOT`, still on the
  drive — cheap to re-load). Reaping frees **hot-drive space** (the model falls back to
  `COLD`, must be re-pulled). Conflating these is how "cold" came to mean two things.
- `RESERVED` (a grant/allocation held but **not** measured-resident, `materialized=false`)
  is not a residency state — it is a Layer-3 attribution. See Layer 3.

---

## Layer 2 — Cost model (Estimated Time to Generation)

Placement and eviction are **decisions**, and a residency state cannot make a decision — a
cost can. For each candidate `(worker, model, variant)`:

```
ETG = t_pull      # >0 only if COLD          bytes / download_bw(worker)
    + t_evict     # >0 only if it won't fit  cost to evict victim(s) for `need` bytes
    + t_load      # >0 unless LOADED/SERVING  upload_time_s EMA  (learned per worker-card)
    + t_queue     # wait behind requests already in flight on that worker/slot   [MISSING]
    + t_generate  # avg_output_tokens / tok_per_s EMA            (learned per worker-card)
```

Choose the candidate with **min ETG**. Eviction is the dual: a victim's cost is not its
bytes, it is its **regret** — `P(needed soon) × (its future t_pull + t_load)`.

**Wiring status (as of this writing): NOT WIRED.**
- `estimate_total_time` computes `t_pull + t_evict + t_load + t_generate`
  (`hugpy_fleet/central/model_metrics.py:265-292`) but has **zero callers** — the estimator is orphaned.
- `t_queue` does not exist yet.
- Placement (`_pick_worker` → injected provider, `hugpy_engine/resolvers/remote.py:394`) ranks
  by **residency** (`_resident_on`, `hugpy_fleet/central/workers.py:2646`), not ETG.
- Eviction (`hugpy_engine/eviction.py`) ranks victims by **recency + bytes**, not regret.

This doc defines the vocabulary the wiring will use; the wiring itself is a separate change.

---

## Layer 3 — Policy / eligibility (boolean gates)

These **filter** the candidate set. They never enter the ETG sum.

| Flag | Meaning | Signal |
|---|---|---|
| `serveable` / `unserveable_reason` | Is the model resolvable & offerable at all (the `UNAVAILABLE` gate; "no delegation") | `v1_routes.py:154-156`; `hugpy_engine/resolvers/model_resolver.py` |
| `assigned` | **Attribution only** — central designated this box. NOT disk presence, NOT eviction protection. | `assigned_models` (`hugpy_fleet/worker/agent.py:3594`); ruling `hugpy_fleet/worker/agent.py:3263-3272` |
| `materialized` | Honesty bit: were weights actually **measured** in VRAM (vs merely claimed). `false` ⇒ `RESERVED`, not `LOADED`. | `hugpy_fleet/worker/agent.py:1201-1205` |
| `pinned` | Never evict from VRAM. | residency/pin config |
| `protected` (+ `why`) | Transient VRAM-evict shield: `static / loaded / loading / provisioning / store-gate`. | `hugpy_fleet/worker/agent.py:3229-3298` |
| `blocked` | Operator block — refused from the serving pool. | `hugpy_fleet/central/blocklist.py` |
| `serves_locally` | Worker **policy**: does this box serve at all (`HUGPY_NO_LOCAL_SERVING`). Unrelated to any model's state. | `remote.py:2449` |
| `evictable` | Derived: `resident && !pinned && !protected && !mid_generation`. | eviction planner |

---

## Canonical term dictionary (quick lookup)

| Term | The one meaning | Layer |
|---|---|---|
| **unavailable** | not resolvable anywhere; cannot be delegated | 1 (floor) |
| **cold** | on source-of-truth/origin, **not** on this worker's hot drive | 1 |
| **hot** | on this worker's hot drive (box-local NVMe), not in VRAM | 1 |
| **loaded** | measured-resident in VRAM, idle | 1 |
| **serving** | resident in VRAM and actively answering | 1 |
| **pull / fetch** | download source-of-truth → hot drive | 1 (transition) |
| **load** | hot drive → VRAM | 1 (transition) |
| **evict** | drop from VRAM (→ hot) | 1 (transition) |
| **reap** | delete from hot drive (→ cold), hot-cache FIFO | 1 (transition) |
| **hot drive** | the box-local NVMe a worker loads from (`HUGPY_HOT_CACHE_ROOT`) | storage tier |
| **source of truth** | the shared model array (`MODELS_HOME`), never reaped | storage tier |
| **t_pull / t_load / t_evict / t_queue / t_generate** | ETG cost terms (seconds) | 2 |
| **ETG** | estimated time to generation = sum of `t_*` | 2 |
| **materialized** | measured-in-VRAM honesty bit | 3 |
| **assigned** | central attribution only | 3 |
| **serves_locally** | worker serving policy | 3 |

---

## Divergence Register (shore up here; do not repeat)

Every known place code contradicts this doc. Resolution = what the code should become.

1. **`cold` = "not in VRAM" (should be "not on hot drive").** ✅ RESOLVED (2026-08-29)
   `hugpy_engine/resolvers/remote.py` (`_admit_cold_hold`, cold-hold gate) treated any
   not-`healthy` model as cold, refusing it under the **download** cap even when it was
   already `HOT` (on the worker's drive). The load-state contract had no on-disk field.
   **Fix applied:** `load_state_for_model` (`hugpy_fleet/central/workers.py`) now emits
   `on_disk` from `models_local` (disk-truth), and `_admit_cold_hold` admits a HOT model
   UNCOUNTED — a `HOT` model's admission is a `t_load`, never a `t_pull`, so it no longer
   consumes a download permit. Landed 2026-08-29; patch archived (fully applied, verified
   2026-09-14) to `archive/deploy-patches-20260914/div1-cold-hold-on_disk.patch`.

2. **Console labels invert `cold`.** ✅ RESOLVED-by-verification (2026-08-30)
   The LIVE console SPA is already canon-correct: `WorkerRow.jsx deriveModelState()` maps
   `onWorkerDisk?'hot':centralHas?'cold':'missing'` with a `🌡 hot` pill and a correct
   sort-rank ladder. The inversion survives ONLY in a dead backup monolith
   (`backs/WorkersPanel/WorkersPanel.jsx`, not imported/bundled). The backend no longer
   encodes the triple (it ships raw signals; the stale `dispatch.py:86` ref is gone). No
   change needed in live code.

3. **`hot`/`cold` = VRAM-at-pick latency bucket.** ✅ RESOLVED (2026-08-30)
   `TEMPERATURES=("hot","cold")` used `hot` for in-VRAM-at-pick. **Fix applied:** buckets
   are now `("loaded","unloaded")` keyed on LOADED-at-pick; `estimate_total_time` default
   and the record path updated; an idempotent SQLite migration rebuckets legacy hot→loaded
   / cold→unloaded so history is preserved. Landed 2026-08-30; patch archived (fully
   applied, verified 2026-09-14) to `archive/deploy-patches-20260914/div3-6-loaded_at_pick.patch`.

4. **Video picker `· cold` = "not answered in 180s".** ✅ RESOLVED (2026-08-30)
   **Fix applied:** `video_prompt_assist_models` now computes the hottest per-key state
   across online workers (serving>loaded>hot>cold, folding in `models_local`+`allocations`)
   and emits a canonical `state` field; the picker (`PromptAssistButtons.tsx`,
   `GenerateStation.tsx`) renders serving/loaded=ready, `· hot (loads)`, `· cold (downloads)`
   — `cold` reserved for not-on-hot-drive. Landed 2026-08-30; patch archived (fully applied
   backend+frontend, verified 2026-09-14) to
   `archive/deploy-patches-20260914/div4-video-picker-canonical-state.patch`.

5. **`serving` is three things.** ✅ RESOLVED-by-verification (2026-08-30)
   Audit shows the three concepts are ALREADY distinct at the field level: the alloc-row
   `serving` is computed solely from *answering within `_SERVING_WINDOW_S` (180s)*
   (`_slot_serving` / `last_used<window`, `hugpy_fleet/worker/agent.py`) = canonical SERVING; readiness is the
   separate `healthy` field (load-state seam); policy is `serves_locally` /
   `HUGPY_NO_LOCAL_SERVING`. No rename needed. Residual `"serving"` appeared only as (a) loose
   log prose and (b) a LEGACY INPUT alias for the on-demand residency default (originally
   retained for backward-compat, like `warm` in #6). **Update 2026-08-30 (same day, later):**
   the compat-shim cleanup removed the `"serving"`/`"warm"` legacy input aliases entirely
   (operator ruling: no back-compat) — neither string is accepted by `hugpy_fleet/worker/agent.py`'s residency
   handler or `worker_routes.py:_normalize_residency` any more (verified 2026-09-14). Patch
   archived to `archive/deploy-patches-20260914/div-cleanup-strip-compat-shims.patch`. New
   code must not use "serving" to mean "ready".

6. **`warm` is overloaded.** ✅ RESOLVED (2026-08-30)
   **Fix applied:** the `warm_at_pick` metric → `loaded_at_pick` (producer + both readers;
   the telemetry `model.warm` field → `model.loaded_at_pick`). The `"warm"` *input* alias
   (originally retained to map an old client string to the on-demand default) was itself
   removed by the later same-day compat-shim cleanup (operator ruling: no back-compat) —
   `"warm"` no longer appears anywhere in `hugpy_fleet/worker/agent.py` (verified 2026-09-14; cleanup patch
   archived to `archive/deploy-patches-20260914/div-cleanup-strip-compat-shims.patch`).
   Landed 2026-08-30; patch archived to
   `archive/deploy-patches-20260914/div3-6-loaded_at_pick.patch`.

7. **`resident` = VRAM in two places, disk in a third.** ✅ RESOLVED (2026-08-30)
   VRAM predicate (canonical, unchanged), eviction unit, and disk `resident_bytes`.
   **#7A:** eviction unit class `Resident` → `EvictUnit` (4 constructors + hints; compat
   alias `Resident = EvictUnit` originally kept). The alias was itself removed by the later
   same-day compat-shim cleanup (no back-compat) — `Resident` no longer appears anywhere in
   `hugpy_engine/eviction.py` (verified 2026-09-14). **#7B:** on-disk footprint
   `resident_bytes`/`resident_model_bytes`/`resident_source` → `hot_bytes`/`hot_model_bytes`/
   `hot_source`, `gauge_basis:"resident"`→`"hot"` (workers.py, worker_routes.py) — this was a
   WIRE key read by the console SPA (`FleetResidency.jsx`, `WorkerStorageBar.jsx`,
   `ResourceStrip.jsx`, `WorkerRow.jsx`); console consumer `WorkerStorageBar.jsx` reads
   `hot_bytes`. No back-compat. Both landed 2026-08-30; patches archived to
   `archive/deploy-patches-20260914/div7a-Resident-to-EvictUnit.patch`,
   `archive/deploy-patches-20260914/div7b-resident_bytes-to-hot_bytes.patch`, and
   (the alias-removal) `archive/deploy-patches-20260914/div-cleanup-strip-compat-shims.patch`.

8. **`loaded_models` has three scopes under one name.** ✅ RESOLVED-by-verification (2026-08-30)
   The authoritative shape ALREADY EXISTS: `_allocations()` (heartbeat allocation rows,
   `hugpy_fleet/worker/agent.py`) reports one row per slot-seated model ∪ per in-process resident — explicitly
   "a NEW field parallel to loaded_models/slots so old central/UI keep working."
   `loaded_model_keys()` deliberately subtracts `slot_backed_model_keys()` (stops the
   loaded/serving flap). `loaded_models` is INTENTIONALLY retained as backward-compat; the
   canonical residency truth is `allocations`. No change made — the operator's compat design
   is deliberate.

9. **`/health` is not residency-truthful.** ✅ RESOLVED-by-verification (2026-08-30)
   Verified the DECISION path already conforms: the central residency predicate
   `_resident_on()` reads the heartbeat `allocations` rows (honoring a slot row only when
   `healthy`/`serving`/`busy`) plus `loaded_models` — never `/health`. The only `/health`
   residency read is an ADDITIVE overlay in `load_state_for_model` that can turn "no
   movement" into movement but never the reverse, so it cannot produce a false
   "nothing loaded". No decision is misled; no change needed.

10. **`in_flight` overloaded.** ✅ RESOLVED (2026-08-30)
    Request-concurrency cap vs eviction mid-generation lock shared the word. **Fix
    applied:** the eviction lock `Resident.in_flight` → `mid_generation` (internal, the
    misleading one — removes the collision); the central `WorkerBusyError` concurrency
    field → `requests_in_flight` (the `"in_flight"` error-payload wire key originally kept
    for client compat). That wire-key alias was itself dropped by the later same-day
    compat-shim cleanup (no back-compat) — `remote.py`'s payload now emits only
    `requests_in_flight` (verified 2026-09-14). gen_gate concurrency counters keep
    `in_flight` (unambiguous — different subsystem). Landed 2026-08-30; patch archived to
    `archive/deploy-patches-20260914/div10-in_flight-disambiguation.patch`.

11. **`in_progress` fuses pull + load.** ✅ RESOLVED (2026-08-30)
    Load-state field documented as "weights loading OR still downloading now" only fed
    error wording. **Fix applied:** `load_state_for_model` now emits `pulling` (COLD→HOT
    provisioning) and `loading` (HOT→VRAM) as distinct flags, keeping `in_progress` as
    their union for forward-progress callers; the `cold_load_capacity` refusal now names
    the actual transition ("still downloading onto" / "still loading into VRAM on" / "not
    loaded yet"). Landed 2026-08-30; patch archived to
    `archive/deploy-patches-20260914/div11-in_progress-split.patch`.

12. **`assigned` was conflated with disk presence.** ✅ RESOLVED (by prior ruling) — stated here:
    `assigned_models` is attribution only (`hugpy_fleet/worker/agent.py:3263-3272`, `hugpy_fleet/central/workers.py:2245-2253`);
    the on-disk signal is `models_local` (`hugpy_fleet/worker/agent.py:5129`). Do not reintroduce the conflation.

---

## Retired words (do not use in new code)

| Retired | Use instead |
|---|---|
| `warm` (any sense) | `LOADED` / `SERVING` (state) or `residency_mode="on_demand"` (config) |
| `hot` for "in VRAM" | `LOADED` / `SERVING`; `hot` means **on the hot drive** only |
| `cold` for "not in VRAM" | `HOT` (on drive, not loaded) — `cold` means **not on the hot drive** |
| `serving` for "ready/healthy" | `LOADED` (resident-idle) |
| `serving` for the policy | `serves_locally` |
| `resident` for on-disk bytes | `hot_bytes` |
| `in_flight` for evict lock | `mid_generation` |

---

## Glossary anchors (source of truth for the reconciliation)

- Hot-drive doctrine (verbatim operator ruling): `hugpy_engine/serve/hot_cache.py` header.
- Two-stage lifecycle (`provision.*` vs `load.*`) and residency tiers: `hugpy_fleet/central/evictions.py:110-183`.
- ETG estimator (orphaned): `hugpy_fleet/central/model_metrics.py:8-9,265-292`.
- Disk-truth signal: `hugpy_fleet/worker/agent.py:5129-5171` (`models_local`).
- Residency truth (heartbeat allocation rows): `hugpy_fleet/worker/agent.py:1830-2010`.
