# Per-worker BOOT-LOAD STAR (`boot_prewarm`)

Operator RULING **2026-07-23 (post-incident)** — verbatim:

- "the star is only supposed to indicate **load that model on boot**."
- "it **shouldn't effect anything but priority for ambiguous model calls**."

A worker carries a single ⭐ **star** model. The star does **exactly two things
and nothing else**:

1. **Load on boot** — once per worker process lifetime.
2. **Ranking priority** — a tie-break for ambiguous / no-warm model calls:
   central prefers the worker whose star == the requested model.

It is **NOT keep-warm**, it does **NOT re-warm after eviction**, and it has **no
eviction interaction**. A starred model that gets evicted under pressure **stays
cold until the worker process restarts**. The identifier stays
`boot_prewarm`/`boot-prewarm` (rename churn isn't worth it) and once again means
exactly **boot-once**.

> **Why the revert (incident 2026-07-23).** The prior release (0.1.201) made the
> star **reconcile-kept-warm** — central re-warmed it every beat and the worker
> reloaded it whenever it was absent. On **ae** that re-warm fired against
> **coder-next while active inference was in flight** → the star's slot child
> stalled → a zombie seat → the agent froze. Re-warm-after-eviction is only safe
> once a **co-fit gate** (reload an evicted model only when it co-fits its
> evictor) exists — that is **future work (Slice D), not yet built** — so the
> star is strictly boot-once until then.

## The three levers (locked semantics)

| lever | keeps warm? | boot-loads? | eviction | notes |
|---|---|---|---|---|
| ⭐ **star** (`boot_prewarm`) | **No** | **Yes — once, on boot** | **Evictable; once evicted STAYS cold until restart** | Also a **ranking tie-break** for ambiguous calls. NOT eviction-protected, NOT re-warmed. |
| 🔒 **static** | **Yes** — the keep-warm tier | yes (eager) | **Protected** — never evicted by the LLM plane | "start here AND stay here". |
| 📌 **pin** | **No** | no | n/a | Routing persistence only — never warms. |

🔒 **static is the keep-warm tier** — the one lever that keeps a model resident
across evictions. If you want "start here **and stay here**", promote the model
to static; the ⭐ star will not hold it.

And for contrast, the /media star:

| | media_default (the /media star) | ⭐ boot_prewarm (this per-worker star) |
|---|---|---|
| Scope | one global model | one model **per worker** |
| Effect | "first in the list + default-selected" — a **UI/routing preference** | "**boot-load** this model + rank it first for ambiguous calls" |
| Loads anything? | **No** | **Yes — once, on the worker's boot** |

**Nothing warms until starred (boot-load) or static.** The fleet `TASK_DEFAULTS`
(sd-turbo et al.) are a **routing fallback** — a request naming only a task still
resolves to its default model (`model_resolver.TASK_DEFAULTS`) — but they are
**not kept warm** (the task-defaults floor is dead and is not coming back). 📌
pins never warm. Everything but 🔒static lazy-loads on first real request (the
star additionally boot-loads once).

### How the star works

- **Worker side — boot-load once.** `_adopt_boot_prewarm` runs on every
  register/heartbeat reply, but a **process-lifetime done-latch**
  (`_BOOT_PREWARM_DONE`) makes it fire the load **exactly once**: on the **first**
  reply carrying a star. It loads through the normal on-demand path (no residency
  write — the model stays FIFO-evictable). Every later beat is a **no-op**,
  **including after an eviction** — the star is **not** reloaded. A genuine retry
  only comes with a worker **restart**. Missing model → logged, never crashes.
- **Central side — ranking priority only.** `_reconcile_warm_set(worker)` keeps
  warm **`🔒static ∩ models_local − blocked`** — the **star is not in it**, so
  central never re-probes the star warm. The star's only central effect is a
  **ranking tie-break**: in `pick_for_model` / `candidates_for_model` the sort key
  is `(home, warm, star, gpu, last_picked, id)` — home beats everything, a warm
  box beats a starred box, and the star breaks the tie when **nothing is warm**
  (prefer the box that boot-loads the model anyway). Alias-tolerant: a star
  recorded under a `~`-qualified key matches a bare-key request and vice versa.

**Future work (Slice D, not built): co-fit-gated re-entry** — an evicted star
would reload only when it **co-fits** with whatever evicted it (no thrash by
construction). Until that exists, an evicted star stays cold until restart.

## API

All paths are relative to central (e.g. `https://dev.hugpy.ai/api`). Worker ids
come from `GET /llm/workers` (the `id` field).

- **Set / replace a worker's star** (operator-gated):
  `POST /llm/workers/<worker_id>/boot-prewarm`
  Body: `{"model_key": "<key>", "enabled": true}` — makes `model_key` the star,
  replacing any previous one. `{"enabled": false}` (optionally with a matching
  `model_key`, or none) clears it.

- **Read the full star map** (open, read tier): `GET /llm/workers/boot-prewarm`
  → `{"<worker_id>": "<model_key>", ...}`

- **Per-worker surfacing**: every row of `GET /llm/workers` (and
  `GET /llm/workers/<id>`) carries `"boot_prewarm": "<model_key>"|null`, so the
  console can render the star.

The star rides the register/heartbeat reply to the worker as
`boot_prewarm: "<model_key>"` — additive and **omit-when-unset**, so a released
worker that predates the feature simply ignores it (the `extra=forbid` relay
schema is never broken).

## Seed the operator's two defaults

These are **data, not code** — run them once against central with the operator
token. Replace `<computron-id>` / `<ae-id>` with the ids from `GET /llm/workers`.

```bash
# computron (RTX-4060) -> a present 7B. Use whichever 7B key the fleet actually
# holds (check `GET /llm/workers/<computron-id>` models / `GET /models`):
#   Qwen~Qwen2-7B-Instruct-GGUF   (or Qwen~Qwen2.5-7B-Instruct-GGUF)
curl -fsS -X POST https://dev.hugpy.ai/api/llm/workers/<computron-id>/boot-prewarm \
  -H "X-Operator-Token: $HUGPY_OPERATOR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_key": "Qwen~Qwen2-7B-Instruct-GGUF", "enabled": true}'

# ae (RTX-3090) -> the agent brain, coder-next (DEFAULT_AGENT_BRAIN):
curl -fsS -X POST https://dev.hugpy.ai/api/llm/workers/<ae-id>/boot-prewarm \
  -H "X-Operator-Token: $HUGPY_OPERATOR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_key": "Qwen~Qwen3-Coder-Next-GGUF", "enabled": true}'
```

Verify: `GET /llm/workers/boot-prewarm` returns both, and each worker
**boot-loads** its star once on start (`boot star … loading once at boot …` in
the agent log; then every later beat is a no-op — the star is **not** re-warmed).
An evicted star stays cold until the worker restarts.

### Note on the 7B key for computron

The brief named `Qwen2-7B-Instruct-GGUF` / `Qwen2.5 7B`. Neither is a curated
staple in `models_config.MODELS` (they arrive via discovery), so the exact
`model_key` is whatever the fleet discovered — confirm against `GET /models`
before seeding. The star store does **not** require the model to be present or
allocated when you set it; if the model can't be loaded, the worker logs and
continues (never crashes).
