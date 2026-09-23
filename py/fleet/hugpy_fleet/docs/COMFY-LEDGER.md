# The ComfyUI resident ledger

**Since:** 2026-09-22. **Code:** `hugpy_fleet/worker/comfy_ledger.py` (pure),
bound in `hugpy_fleet/worker/agent.py` (`_comfy_ledger`, `_comfy_need_detail`,
`_comfy_headroom_target`, `_comfy_busy_reason`). **Tests:**
`tests/test_comfy_ledger.py`, `tests/test_vram_evict_to_fit.py`,
`tests/test_comfy_headroom.py`.

## The problem it closes

ComfyUI is an external process. nvidia-smi shows one VRAM lump under its PID
and nothing says which checkpoints are inside it, so the worker's VRAM budget
never had a per-occupant figure for comfy — which meant three loops with three
different targets and no shared notion of *need*:

| Loop | What it did | Blind spot |
|---|---|---|
| eviction planner (`_partition_residents`) | ranks measured residents LRU and evicts the minimum set for a load | comfy rows were **always protected** (0.1.137 "out of allocations") |
| Fix B headroom (`_worker_ensure_comfy_headroom`) | evicts managed models before a comfy gen | cleared a **constant** 7 GiB for every gen — 2 GiB SD1.5 and 12 GiB checkpoints alike |
| idle watchdog (`comfy_watchdog`) | frees an idle comfy after a TTL, or under contention | a side channel the planner could not see or rank |

The one thing a budget needs — how much *this* checkpoint takes — was never
computed, although it is knowable before the load: `ComfyRunner` names the
checkpoint file in the graph and, for the fp16 safetensors comfy serves,
weights on disk ≈ weights in VRAM.

## What the ledger is

`ComfyLedger` records what hugpy has asked comfy to load since comfy's last
known `/free`: `{model_key, filename, bytes, loaded_at, last_used}`, oldest
first. Writes happen at the two places the truth changes:

* `note_dispatch` — inside the headroom hook, right before the runner POSTs
  the graph; the file is sized by `checkpoint_size_bytes`.
* `note_freed` — inside `_comfy_free_models` on a 200, which every `/free`
  goes through (watchdog, evict verb, planner).

Reads:

* `_vram_residents` unions the ledger's rows in as `host_mode="comfy"` with
  per-row bytes **capped at comfy's measured process VRAM**, and only when a
  comfy process actually holds VRAM (a ledger with no process behind it names
  nothing). The registry still contributes the active call's attribution;
  the ledger adds what comfy still caches.
* `_comfy_status()["resident"]` carries the same rows in the heartbeat, so the
  console's comfy card can show checkpoints by name instead of one lump.

## The three behaviour changes

1. **Per-checkpoint need.** `_comfy_need_detail` prices a gen as
   `checkpoint file size + gen cushion`, or the cushion alone when comfy
   already holds the checkpoint *and* its measured VRAM covers the weights (a
   ledger claim the device does not back is dropped, never trusted).
   `_comfy_headroom_target` uses that need; when the file cannot be sized the
   legacy constant applies unchanged, and an **explicitly set**
   `HUGPY_COMFY_TARGET_FREE_GIB` is a floor the need never undercuts.
2. **Comfy is an evictable resident under contention.** `_partition_residents`
   asks `_comfy_busy_reason` once per partition — the watchdog's own predicate
   (a registered comfy call, a non-empty `/queue`, or anything unreadable = busy)
   — and comfy rows are candidates when idle, protected with
   `why="comfy busy: …"` otherwise. `_evict_model`'s comfy branch honours the
   same predicate (unless `force`), because `/free` drops every checkpoint at
   once. The **idle sweep** still never touches comfy; TTL-based reclaim stays
   the watchdog's job.
3. **Named residents in telemetry** (above).

## Knobs

| Env | Default | Meaning |
|---|---|---|
| `HUGPY_COMFY_GEN_CUSHION_GIB` | `2.0` | working room on top of the weights (activations, VAE, CLIP, fragmentation); ae recon: SD1.5 at 512² grew the comfy process ~1 GiB during a gen |
| `HUGPY_COMFY_TARGET_FREE_GIB` | `7.0` (fallback only) | the legacy constant: used when the checkpoint cannot be sized; a floor when set explicitly |
| `COMFY_CHECKPOINTS_DIR` | `~/ComfyUI/models/checkpoints` | where hugpy symlinks checkpoints for comfy (`hugpy_storage.provision`); first place a name is resolved |
| `COMFY_CHECKPOINT_DIRS` | unset | `os.pathsep`-separated extra roots — set it to the roots comfy's own `extra_model_paths.yaml` scans on this box so a name resolves to the file comfy will really load |

Resolution order for a `ckpt_name`: as given under each root, then by basename
anywhere below each root (comfy's scan is recursive). Symlinks are followed;
a dangling link is "not found" (unknown need → legacy target), never 0.

## Doctrine kept

* Degrade-not-guess: unmeasurable free VRAM, an unsizable file, a failing
  catalog read — each falls back to exactly the previous behaviour.
* A render is never killed for an LLM load: the busy predicate is the
  watchdog's, and unprovable idleness reads as busy.
* Comfy's weights are never counted as headroom for another model's admission
  (`_subject_resident_vram_bytes` is unchanged).
