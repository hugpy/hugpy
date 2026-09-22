# hugpy-engine

`hugpy-engine` is the standalone boundary for Hugpy's complete inference path:

```text
prompt/messages
    -> model discovery and resolution
    -> RAM/VRAM allocation and admission
    -> engine and runner selection/loading
    -> generation and continuation
    -> streamed events or a completed reply
```

The package is split by responsibility:

- `catalog` — discovered models, task/media/default model resolution;
- `allocation` — feasible modes, default placement, spill and admission plans;
- `engines` — native llama.cpp discovery and GGUF runner lifecycle;
- `runtime` — request resolution, execution, runner caching and eviction;
- `query` — prompt/messages to streamed events, text, or `QueryResult`;
- `backend` — the injection contract joining those layers to an implementation.

The package imports CPU-only: `import hugpy_engine` never loads llama_cpp,
torch, transformers or peft. Those arrive through extras and are imported by
the runner that needs them:

| extra | provides |
|---|---|
| `hugpy-engine[gguf]` | in-process GGUF inference via `llama-cpp-python` |
| `hugpy-engine[transformers]` | the transformers text-generation runner |
| `hugpy-engine[finetune]` | PEFT adapter loading |
| `hugpy-engine[index]` | the optional Postgres model index |

The default backend is the in-process `LocalBackend` (`hugpy_engine.backends`),
built from the engine's own catalog, allocation, native-engine and dispatch
modules; `configure_backend(...)` / `backend_scope(...)` swap in another
`InferenceBackend` (a fake in tests, a remote proxy in a thin client).

```python
from hugpy_engine import query_sync

reply = query_sync("Explain tensor parallelism", model_key="my-model")
```

Async callers use `await query(...)`. `stream_query(...)` yields engine events;
`query_result(...)` retains request ID, finish reason, usage, and timings.

## Seams (what the engine asks of the packages above it)

- `hugpy_engine.placement` — Protocols + `get_*()/set_*()` providers for
  everything the engine asks about the fleet (worker registry and key forms,
  worker HTTP transport/breaker, eviction telemetry ledger, blocklist, model
  metrics, priority groups). Null defaults mean "no fleet"; `hugpy_fleet`
  implements them and the server wires them.
- `hugpy_engine.tasks` — the task-runner registry. Media and video runners
  plug in with `register_task(task, runner=..., build_request=...,
  frameworks=(...), extra=..., source=...)`; `FRAMEWORK_RUNNERS`,
  `MODEL_REQUEST_BUILDERS` and `KNOWN_TASKS_REGISTRY` in
  `hugpy_engine.resolvers.categories` are live views over it. Packages may
  also expose a `hugpy_engine.tasks` entry point (a zero-arg register
  function); the engine loads those on first use.
- `hugpy_engine.catalog_bridge` — installs the engine's registry as
  `hugpy_storage`'s catalog source, the hot cache as its serve-path hook, and
  refreshes discovery on `hugpy_control.bus` `catalog.changed` events.
  `LocalBackend` installs it lazily.
- `hugpy_engine.name_match` — the pure eliminate-then-rank name pipeline
  (`resolve_name`, `Candidate`) that `assure_model_key` uses; the oracle
  re-exports it.
