<!-- MECHANICS DOC TEMPLATE — copy this shape exactly. Goal: a future agent reads
THIS FILE instead of the subsystem's code and can reason correctly about it.
Be concrete: cite `file.py:line` (paths relative to the importable package
src/abstract_hugpy_dev/src/abstract_hugpy_dev/ unless noted). Keep ~150–320 lines. -->

# <Subsystem name> — mechanics

**Scope (paths this doc covers):** `managers/serve/`, `…`
**One-liner:** what this subsystem is responsible for, in one sentence.
**Owner process(es):** which running unit(s) load this (central `7002_hugpy_api`,
the worker agent, the toolserver, the bot, a daemon) — and whether it's library-only.

## 1. Purpose & responsibilities
2–5 bullets: what it does and, importantly, what it deliberately does NOT do.

## 2. Key modules (file → responsibility)
A tight table. One line each, only the load-bearing files (skip trivial).
| file | responsibility |
|---|---|

## 3. Entry points
How execution reaches this subsystem: HTTP routes (`/llm/…` → handler),
CLI subcommands, `python -m …`, or the functions other subsystems import. Name them.

## 4. Data flow (the spine)
The main path(s) step by step — the thing a reader most needs. e.g. "request →
resolver → SlotPool.acquire → slot_agent spawns llama-server → stream back".
Number the steps; cite the file:line at each hop. Cover the 1–3 dominant flows.

## 5. State, persistence & invariants
What it reads/writes (DB tables, JSON files, in-memory caches, env), and the
invariants that MUST hold (e.g. "one slot child per control port", "off is the
serve-mode default"). Note where state lives on disk.

## 6. Cross-subsystem edges
What this calls (→) and what calls it (←). Keep to real, load-bearing edges so
the partition graph is navigable. Link sibling mechanics docs: `[[serving-core]]`.

## 7. Key contracts / types
The frozen schemas, dataclasses, dict shapes, or wire formats other code depends on.

## 8. Gotchas, tech-debt & review findings
The payoff of the *review*: sharp edges, footguns, duplicate/dead code, silent
failure modes, version/schema drift, and any correctness/robustness concerns you
found — each with `file:line` and a one-line "why it matters". Mark severity
(⚠ bug / △ debt / ℹ note). This is not just docs — it's the audit trail.

## 9. Deploy/run boundary
How a change here reaches the running system (central now runs the **src** checkout
via `PYTHONPATH`, so a src edit is live on `7002` restart; workers run the pip wheel;
console builds to `console_dist/`). Note anything env/unit-gated.
