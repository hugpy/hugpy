# hugpy-oracle

`hugpy_oracle` — extracted from `abstract_hugpy_dev` as part of the Hugpy
partition. Ownership and allowed dependencies are declared in
`py/partition.toml`; see `PARTITION.md` at the workspace root.

Allowed Python dependencies inside the ecosystem: hugpy_platform, hugpy_control, hugpy_engine, hugpy_media, hugpy_video.

## Seams

- `hugpy_oracle.providers` — Protocols for what the oracle needs from packages
  it must not import, each with a safe default: `DossierSource` (curation's
  dossier store), `DoctrineSource` (fleet doctrine), `TaskCapabilityGate`
  (fleet's capability-honesty rule), `LoadStateSource` (model residency per
  worker). The composition root installs implementations with `set_*()`.
  Worker rosters and the blocklist come from `hugpy_engine.placement`.
- `hugpy_oracle.relay.hooks` / `hugpy_oracle.install_hooks()` — the oracle's
  side of `hugpy_video.hooks`: the `PromptCoordinator` implementation and the
  `(oracle, performance)` bus runner.
- `hugpy_oracle.config` — every state root (reliability ledger, routing
  matrix, benchmark battery, run journals) resolved in one place; per-store env
  vars (`ORACLE_LEDGER_PATH`, `ORACLE_BENCHMARK_ROOT`, ...) win, then
  `HUGPY_ORACLE_HOME` / `HUGPY_ORACLE_RUNS_ROOT`, then `hugpy_platform.app_dirs`.

## Command line

`hugpy-oracle steward [--apply] [--api URL|--local] [--ledger PATH]` runs one
steward pass (remote against a running central when `HUGPY_API` is set,
otherwise in-process over the shared ledger); `deploy/hugpy-steward.{service,timer}`
schedule it. `hugpy-oracle install-hooks` reports what the video hooks wired.

## Performance in the video console

The Oracle station submits `POST /video/jobs/performance` and reads
`GET /video/performance/probe`. The submit route validates the nested goal,
locked dialogue and casting before a job enters the media bus. It uses the same
owner and visibility stamping as other `/video/jobs/*` routes. The station
offers `stop_after` so an operator can inspect authority, audio, production
lock and segment planning before attempting visual stages.

The probe lists bound and unbound seams. A full performance is ready only when
image generation and judging, clip generation and judging, audio, and assembly
are all bound. An unbound clip seam is a capability gap, not a successful render.
