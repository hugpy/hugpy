---
name: composition-patterns
description: Reusable agent-composition patterns, scoping rules, and boundary notes for the hugpy dev VM
metadata:
  type: project
---

Patterns for orchestrating verification/consolidation work on the hugpy dev VM. See [[ledger-2026-06-25-vision-media-verify]].

## Effective decomposition
- A "landed change" verification pass cleanly splits into TWO non-overlapping agents:
  - **fault-hunter** — code-correctness/security sweep over the *changed surface only*. Always hand it an explicit allowlist of absolute paths + the live service ports so it audits the live copy, not a backup/archive twin.
  - **keeper-briefer** — knowledge-base reconciliation (confirm memories accurate, flag stale xrefs, NO duplication). Hand it the exact memory slugs already written so it reconciles rather than re-writes.
- These two never overlap (code vs. knowledge), so they run concurrently with no conflict risk.

## Scoping rules that prevented drift
- This repo has duplicated files (e.g. two chat_schemas.py, multiple main.tsx). ALWAYS disambiguate which copy is live before scoping, and tell the agent which copy is canonical so it doesn't audit a dead twin.
- Real backend dev tree is `dev/abstract_hugpy_dev/src/abstract_hugpy_dev/` (NOT dev/archive, NOT flask_app/app/functions for the schema copy).

## Boundary notes (hard)
- DEV ONLY mandate: no agent may reach prod, the host (ae), or other VMs. Bake "dev only, do not touch prod/host" into every spawn prompt.
- Only manage agents this manager spawned; treat any pre-existing/external agent as read-only.
