# Chaos-and-learn observation schema (`chaos-obs/1`)

**Contract between the chaos _runner_ and the t28 _learner_.** The runner
(`hugpy_ops.chaos`) exercises the live assortment (models × cards ×
alloc modes × ctx%) chaotically and appends one observation per trial to:

```
/mnt/llm_storage/comms/chaos/observations.jsonl      # append-only, one JSON obj / line
/mnt/llm_storage/comms/chaos/runs/<run_id>.json      # per-run manifest
```

The learner reads the JSONL and calibrates placement templates by pricing the
**predicted** side against the **measured** side. Keep this stable; bump
`SCHEMA_VERSION` (in `schema.py`) on any breaking change and coordinate here.

## Why "predicted vs measured"

Every trial pairs what central/the operator *intended* (priced cheaply BEFORE
the load, from `GET /models/<key>/meta`) against what *actually happened* (read
AFTER the load from `GET /llm/workers` — the serving contract's real
`vram_bytes`/`rss_bytes`/`n_gpu_layers`, plus the verbatim admission verdict when
a refusal/partial surfaces). The learner's job is to learn the correction from
`predicted.need_bytes` → `measured.allocation.vram_bytes`.

## The predicted → measured pairing (what the learner joins on)

| predicted field | measured counterpart | meaning |
|---|---|---|
| `predicted.need_bytes` | `measured.allocation.vram_bytes` + `rss_bytes` | total priced need vs real footprint (slot=GPU, ram=CPU RSS) |
| `predicted.needs_weights_bytes` | `measured.allocation.rss_bytes` (weights part) | weights estimate vs resident |
| `predicted.needs_kv_bytes` | `measured.allocation.ctx` / `n_gpu_layers` | KV @ ctx_pct vs served ctx & offload split |
| `predicted.ctx_pct` / `ctx_resolved` | `measured.allocation.ctx` | requested ctx vs served ctx |
| `combo.alloc_mode` / `combo.spill` | `measured.admission.verdict` | intended placement vs actual admission outcome |
| `predicted.per_worker[w].advice.n_gpu_layers` | `measured.allocation.n_gpu_layers` | predicted offload vs measured offload |
| `predicted.per_worker[w].feasible_hybrid` | `measured.outcome` / `admission.refusal_reason` | fit prediction vs reality |

When the load is **refused**, `measured.admission.refusal_reason` carries the
verbatim `_vram_evict_to_fit` reason dict (`needs_bytes`, `needs_weights_bytes`,
`needs_kv_bytes`, `ctx_pct`, `ctx_resolved`, `free_vram_bytes`,
`total_vram_bytes`, `ceiling_reserve_bytes`, `evicted`, `protected`, and
`partial_offload_considered` when a hybrid was declined). These are the
worker's own numbers — the learner should prefer them over central's estimate
when present.

## Admission verdicts (`measured.admission.verdict`)

- `proceed` — loaded fully on GPU, no eviction needed.
- `cpu` — loaded on CPU RAM (`allocation.kind == "ram"`).
- `partial` — hybrid GPU/CPU offload (`0 <= n_gpu_layers < total_layers`).
- `evicted` — loaded after evicting ≥1 idle resident (`vram_evictions_delta > 0`).
- `flex` — fit within t21 tolerance bands (best-effort inference; raw signals
  always retained so the learner can re-derive).
- `refuse` — honest refusal before any CUDA alloc; see `refusal_reason`.
- `unknown` — couldn't classify (retain raw fields).

## Full shape

See the module docstring in `schema.py` for the field-by-field layout, and
`schema.blank_observation()` for a fully-formed template with every required key.
`schema.validate_observation(obs)` returns a list of completeness problems
(empty == valid); the runner self-checks every line before appending.

## What the runner does NOT record / touch

- It never edits model files, `worker_assignments.json`, `flex.py`, or
  worker-agent need-pricing.
- It only mutates `spill_by_model[model_key]` via the operator-gated `/assign`,
  and restores the prior value (verified) after every trial.
- It exercises only **already-assigned** (worker, model) pairs, so restore is a
  clean write-back — never an unassign.
