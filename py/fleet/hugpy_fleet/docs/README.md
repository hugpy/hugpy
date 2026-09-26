# hugpy-fleet docs

Canonical copies moved from the monolith's `deploy/` folder (partition 2026-09-22).

| Document | Subject | Owner |
|---|---|---|
| `STATE-MODEL.md` | Model state nomenclature: residency ladder, ETG cost model, policy gates, divergence register | fleet (Layers 2–3, registry/heartbeat truth); Layer-1 residency vocabulary is shared with `hugpy-engine`'s eviction planner |
| `WORKER-WIREGUARD.md` | One-step WireGuard join for a remote worker: the wg-peer helper + sudoers, `hugpy-fleet join-code`, the worker join script, and revocation | fleet |
| `WORKER-BOOT-PREWARM.md` | The per-worker boot-load star (`boot_prewarm`) | fleet |
| `WORKER-WILDCARD.md` | The per-worker wildcard routing opt-in | fleet |
| `COMFY-LEDGER.md` | What ComfyUI holds, by name and size: per-checkpoint headroom need, comfy as an evictable resident under contention, the knobs | fleet |

Code paths in these documents refer to the extracted packages
(`hugpy_fleet/...`, `hugpy_engine/...`).
