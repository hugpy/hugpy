# hugpy mechanics — the partitioned innerworkings map

Purpose: a **read-the-doc-not-the-code** layer. hugpy is large (~500k files, a ~20-subsystem
package). Instead of ingesting the tree, open the one or few mechanics docs for the subsystem
you're touching. Each doc is self-contained and follows `_TEMPLATE.md` (purpose, key modules,
entry points, data flow, state/invariants, cross-edges, contracts, **review findings**, deploy
boundary). Written by a per-subsystem code-review pass (2026-09-14).

Orientation stack: `/srv/hugpy/CLAUDE.md` (session router) → `../ARCHITECTURE.md` (package map) →
**these mechanics docs** (per-subsystem deep dives) → the code.

## Partition (one doc per row; `[[link]]` = sibling doc)

| doc | subsystem | scope (under `src/abstract_hugpy_dev/src/abstract_hugpy_dev/` unless noted) |
|---|---|---|
| `api-routes.md` | Central API & console serving | `flask_app/` (`wsgi_app.py`, `app/routes/*` ~35 blueprints, `member_auth`/`operator_auth`/`video_auth`, `endpoints_*`) |
| `serving-core.md` | Model serving / slots / placement | `managers/serve/*`, `managers/{eviction,spill,alloc_modes,draft_models}.py`, `managers/dispatch/`, `managers/resolvers/` |
| `engine-generation.md` | Engine + generation lanes | `engine/`, `managers/generate/`, `managers/llama/`, `provisioner.py`, `model_sync.py`, `model_battery.py` |
| `media-gen.md` | Image/video/audio/vision generation | `managers/{comfy,imagegen,video_gen,vision,whisper_model,tts,summarizers,embed}/` |
| `worker-fleet.md` | Worker agents + enroll/admit/assign | `worker_agent/`, `gguf_worker/`, `phone_brick/` (+ the central `/llm/workers/*` contract they use) |
| `comms-index.md` | Comms substrate + model registry DB | `comms/*`, `imports/src/model_index/*` (the `model_metrics`/`model_calls` registry), `flask_app/app/functions/imports/utils/workers.py` |
| `video-oracle.md` | Video-intelligence + oracle planning | `video_intel/`, `oracle/` |
| `curation-reliability.md` | Model curation + reliability daemons | `discovery_dossier/`, `review/`, `sentinel/`, `chaos/`, `fleet_doctrine/`, `downloader/` |
| `cli-agents-platform.md` | CLI/bot/keeper + platform/utils + py side-pkgs | `cli.py`, `bot/`, `keeper.py`, `_platform/`, `utils/`, `imports/`, and `src/py/*` (`abstract_apply`, `abstract_identity`, `hugpy_agent`, `hugpy_video`) |
| `frontends.md` | Web/desktop UIs | `src/react/` (`ui/` console, `agents_ui/`, `media_intelligence_ui/`, `video_intelligence_ui/`, `ui_shared/`) and `src/station-app/` |

## Status (filled as agents complete)
_(each agent appends its row: doc ✅, and a one-line headline finding)_

## Cross-cutting facts every doc assumes
- Central `7002_hugpy_api` now runs the **src** checkout (`PYTHONPATH=…/src`), version 0.1.257 — src edits are live on restart. Workers run the pip wheel. Console serves `console_dist/`.
- Durable operating rules: `/srv/hugpy/PROVISIONS.md`. Fixed nomenclature: the station `NOMENCLATURE.md`.
