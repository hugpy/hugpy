# Per-worker WILDCARD flag (`wildcard`) — "take all comers"

Operator doctrine 2026-07-23 (WILDCARD PLACEMENT + DESIGNATION SCOPE):

- **Worker designations are a HARD routing scope** — they seal where designated
  models CAN route.
- An **undesignated model "gets in where it fits in"** — but ONLY on workers
  that opted in as **wildcard**: "a worker can be designated to take all comers
  while adhering to its allocated model list as priority, or it can not be
  selected as a wildcard and adhere only to its own allocated models."
- A designated model whose designated workers are ALL unavailable **overflows**
  onto wildcard workers; it **busts only when neither** designated nor wildcard
  workers can serve.
- Once resident, **normal eviction rules apply** — the flag affects routing,
  never eviction. "You don't want random evictions simply because you have a
  verbose model registry" — which is exactly why all-comers is an **explicit
  per-worker opt-in**.

## The default-false promise

**Every worker defaults to `wildcard: false`.** With no flags set, routing is
**identical to the pre-feature fleet**: a model routes only to workers
designated for it — or already holding it (**resident = de facto designation**:
a box that reports the model in `loaded_models`, or holds a placement grant, is
always eligible for it; routing never refuses a box that already has the
weights up). The flag is purely additive.

What wildcard is **not**:

| lever | what it does |
|---|---|
| `wildcard` (this flag) | **Routing eligibility only** — the worker may catch undesignated models + designated overflow |
| ⭐ star (`boot_prewarm`) | keep-warm designation — reconcile keeps it warm |
| 🔒 static | warm AND eviction-protected |
| 📌 pin | routing persistence — never warms |

A wildcard catch never bypasses the hard gates: model **block**, worker
**admission**, dedicated-**pool** reservation, **engine** usability, env
**tier**, **task** capability and id-lock all still apply.

## The two API calls

```bash
# Read the opt-in map ({worker_id: true}; absent = false). Open, read-only.
curl $CENTRAL/llm/workers/wildcard

# Opt a worker in / out. Operator-gated (X-Operator-Token / session).
curl -X POST $CENTRAL/llm/workers/<worker_id>/wildcard \
     -H "X-Operator-Token: $TOK" -H "Content-Type: application/json" \
     -d '{"enabled": true}'      # or {"enabled": false} to restore the seal
```

Every worker row on `GET /llm/workers` and `GET /llm/workers/<id>` carries
`wildcard: true|false` (stamped on the response copy — never persisted onto the
stored worker record). State lives server-side in `worker_wildcard.json`
beside the other model-config stores.

## Ranking / overflow behavior

Overflow is **pure ordering**, not separate machinery. Candidate ranking in
`pick_for_model` / `candidates_for_model` is:

1. **home** match (designated / resident / granted) before **wildcard catch**,
2. warm (model already loaded) before cold,
3. usable GPU before CPU-only,
4. least-recently-picked, stable id tiebreak.

So a designated model always tries its home workers first; when every home
worker is refused (offline, engine-broken, at cap, wrong tier...) the same
ranked walk lands on a wildcard worker; when neither can serve, the request
fails exactly as before. A wildcard worker's **own designated models keep
priority** on that box by the same rule — home outranks catch.

Related (ships with this slice): key matching now unifies `~`-qualified and
bare model keys (`Qwen~X` ↔ `X`) at every routing match site, with a
blocked-sibling guard so an alias match can never serve a BLOCKED sibling
(`B~X` blocked ≠ served via a request for `A~X`).
