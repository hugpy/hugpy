# hugpy-fleet docs

Canonical copies moved from the monolith's `deploy/` folder (partition 2026-09-22).

| Document | Subject | Owner |
|---|---|---|
| `STATE-MODEL.md` | Model state nomenclature: residency ladder, ETG cost model, policy gates, divergence register | fleet (Layers 2–3, registry/heartbeat truth); Layer-1 residency vocabulary is shared with `hugpy-engine`'s eviction planner |
| `WORKER-BOOT-PREWARM.md` | The per-worker boot-load star (`boot_prewarm`) | fleet |
| `WORKER-WILDCARD.md` | The per-worker wildcard routing opt-in | fleet |

Code paths in these documents refer to the extracted packages
(`hugpy_fleet/...`, `hugpy_engine/...`).
