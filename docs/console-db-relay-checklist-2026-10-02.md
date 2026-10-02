# Console → model-DB relay — checklist (2026-10-02)

Ruling: everything the workers panel shows and writes is a DATABASE relay (model_workers / model_quant_facts /
model_worker_quants / worker_budgets), never central's JSON. Central converges to the DB (designation_relay) and
reads DB pair knobs at load time (`_placement_spill_for` overlay).

## Workers-panel columns

| Column / control | Reads | Writes | Status |
|---|---|---|---|
| Alloc (label + menu) | DB `user_settings.alloc_mode`, verdict `plan.auto.*`, ctx-aware fallback (gpu-only → max-ram → explicit → ram-only) | `/api/models/database/<m>/workers/<w>/knobs` (`set` placement knobs; Auto = `unset`) | DB ✅ |
| Alloc → Explicit (MoE GGUF) | verdict `memory.explicit.band` (per-class primitives, experts cap table, attention floor) | knobs `attention_gpu_layers` / `experts_cpu_layers` (optional; else `explicit_spill` prefer-gpu/prefer-ram derives it) → server derives `alloc_mode=explicit`, `n_gpu_layers`, `n_cpu_moe` | DB ✅ (live check pending deploy) |
| Alloc → Explicit (dense GGUF) | — | knobs `gpu_mem_gib` / `cpu_mem_gib` (leniency / priority_device dropped) | DB ✅ |
| Actions ▶ assign | — | `/assigned {true}` + knobs for a spill (filtered to PAIR_KNOB_KEYS) | DB ✅ |
| Actions × unassign | — | `/assigned {false}`; row kept (assigned=false) until central drops the pair | DB ✅ |
| Actions ⏏ free / ⛔ block / 💬 chat | central live | central | central (by design) |
| Memory | verdict `memory[mode]` (or `memory.bnb_4bit[mode]`), `quants.kv_cost` × cache type, `plan.budgets.gpu_budget`, per-class split from the band | — | DB ✅ (compute reserve shown separately) |
| 4-bit | DB `user_settings.bnb_4bit` via pairDb (fixed: was central `bnb_by_model`) | knobs `bnb_4bit` (bulk tick still central `/bnb`) | DB ✅ / bulk ⏳ |
| MoE | DB `user_settings.moe`; offer = verdict `moe_offered` (transformers MoE: never offered) | knobs `moe` (bulk tick still central `/moe`) | DB ✅ / bulk ⏳ |
| Ctx value / slider / KV cache / flash | DB `ctx_pct` (auto = largest that fits), `ctx_max` per cache type, `kv_cache_type`, `flash_attn` | knobs `ctx_pct` / `kv_cache_type` / `flash_attn` | DB ✅ |
| Size | catalog size; 4-bit size from facts when the knob is on | — | DB ✅ |
| Residency (static / on-demand) | central `worker.config.residency` | central | ⏳ queued: DB relay |
| Seat | central `allocations` | — | central live (measured) |
| 📌 pin | central designations; relay pins (`pinned_by=db-designation`) do NOT count | central `set_pin` | central (operator pin = permanent attribution) |
| State | central live (loaded/loading/slots) | — | central live (measured) |

## Review findings (opus48 audit) and disposition

1. HIGH Explicit panel sent `priority_device`/`leniency_pct` → knobs 400 — **fixed**: per-class panel replaces them; assign/applyAllocMode filter to PAIR_KNOB_KEYS.
2. HIGH Relay pins lit 📌 and disabled × — **fixed**: `effectivePin` ignores `pinned_by === 'db-designation'`.
3. MED Revert-to-derived never cleared the knob — **fixed**: `{}` → `unset` of the placement knobs.
4. MED 4-bit tick read central, wrote DB — **fixed**: reads `pairDb().bnbOn`.
5. HIGH No DDL for facts/verdicts/budgets tables + `plan` column — **fixed**: CREATE TABLE IF NOT EXISTS in `ModelQueries.MIGRATIONS` (central owns the schema).
6. LOW `pairDb()` recomputed per cell, stale `dbPairCtx` locals — **open** (memoize once per row).
7. LOW `ContextPreview.jsx` dead — **fixed**: deleted.
8. Write rule vs UI rule match — verified; `bf16` not selectable in the KV dropdown (minor).
9. MED Convergence lag: relay 3 changes / 10 s; row shows central's list meanwhile — by design; console should reload after a bulk ×.
10. LOW KV knobs → slot wiring — **done after the audit**: central overlay → agent `_SPILL_ENV` → slot load opts → `--cache-type-k/-v`, `--flash-attn`.
11. No tests for `/knobs` and `/assigned` routes — **open** (live checks done with the `hugpy_testoperator` identity; add route tests).

## Still queued
- Residency (static / on-demand) as a DB relay.
- Bulk MoE / bulk 4-bit ticks → knobs route.
- Quant selection per row: checkbox per quant + sortable priority + evict-to-fitting.
- Central reading verdicts (modes/memory) from the DB instead of its own derivation; residency DB relay.
