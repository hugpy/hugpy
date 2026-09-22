---
name: ledger-2026-06-25-vision-media-verify
description: Ownership ledger for the 2026-06-25 vision-fix + media-arm verification/consolidation pass (fault-hunter + keeper-briefer)
metadata:
  type: project
---

Verification+consolidation pass over the landed image-analysis fix (dev only). See [[composition-patterns]].

## Ledger
| id | type | purpose | created | state | deactivated |
|----|------|---------|---------|-------|-------------|
| fault-hunter-vision-media | fault-hunter | regression/async/security/contrast sweep over the changed surface ONLY | 2026-06-25 | INACTIVE (objective met) | 2026-06-25 (agentId ac0f1e1de758826b6) |
| keeper-briefer-vision-recon | keeper-briefer | confirm already-written memories accurate + flag stale xrefs, no duplication | 2026-06-25 | INACTIVE (objective met) | 2026-06-25 (agentId a7ac4e6b21e71c525) |

## Outcome (both stayed on task, no drift)
- fault-hunter: NO critical/high; 1 MEDIUM (/uploads anonymous + unbounded disk → slow-DoS), 2 LOW (stuck ToolTray pill while disabled; benign runFileOption ordering). Confirmed: two chat_schemas are DIFFERENT classes (not a divergence); remote.py has no dead vision mandate/gate + local fallback intact; get_gguf_file deterministic; workers.py cannot leak +local; ChatPanel/AttachmentBar clean; NO cached asyncio primitives. Corrected brief: supports_vision lives in gguf_worker/agent.py, not central workers.py.
- keeper-briefer: 4 minimal edits. Key finding — the _worker_serves_vision capability GATE was itself removed (not just the _vision_local mandate), so vision is no longer special-cased anywhere; corrected hugpy-vision-image-analysis-fix point 4 + hugpy-vision-pool-routing (now fully historical) + 2 MEMORY.md hooks. No stale xrefs left in other notes. Flagged (not written): stale comment ~agent.py L180.

## Canonical changed-file surface (live, disambiguated)
Backend dev tree (real, not archive): `dev/abstract_hugpy_dev/src/abstract_hugpy_dev/`
- `imports/src/schemas/chat_schemas.py` (ChatRequest._normalize_multimodal) — NOTE a SECOND copy exists at `flask_app/app/functions/imports/utils/schemas/chat_schemas.py`; live-changed one is the imports/src copy
- `managers/resolvers/remote.py` (removed _vision_local mandate + later capability gate)
- `imports/config/main.py` (get_gguf_file deterministic selection)
- `flask_app/app/functions/imports/utils/workers.py` (supports_vision, mmproj-aware model_looks_downloaded, required_pkg_version no +local) — single canonical copy
- `flask_app/app/operator_auth.py` (POST /uploads removed from operator gate)
UI: `dev/ui/src/components/ChatPanel/ChatPanel.jsx` (payload.images now set)
Media arm chat (`dev/media_intelligence_ui/src/chat/src/`): `ui/ToolTray.tsx`, `main.tsx` (sticky pre-selects), `ui/AttachmentBar.tsx` (accent contrast fix)
Pip hygiene: `keeper/stock_pip_index.sh` (no +local minting)

Live services confirmed: UI :7001 (webpack pid 661070), API :7002 (gunicorn pid 696992/696989).

## Boundary
DEV ONLY. No prod, no host, no other VMs. Agents must stay on the changed surface; stop if drifting.
