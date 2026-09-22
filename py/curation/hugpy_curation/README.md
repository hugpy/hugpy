# hugpy-curation

`hugpy_curation` — discovery dossiers and the search / screen / download /
smoke / judge review pipeline, extracted from `abstract_hugpy_dev` as part of
the Hugpy partition. Ownership and allowed dependencies are declared in
`py/partition.toml`; see `PARTITION.md` at the workspace root.

Allowed Python dependencies inside the ecosystem: hugpy_platform,
hugpy_control, hugpy_storage, hugpy_engine, hugpy_oracle.

## Layout

| Module | What |
|---|---|
| `review/` | `criteria`, `screen` (HF metadata), `download` (through `hugpy_storage`), `smoke` (llama_cpp subprocess), `judge`, `pipeline`, `store` (sqlite record), `push` (worker -> central) |
| `dossier/` | the per-model dossier: `cards`, `weights`, `research`, `community`, `radar`, `trial`, `verdicts`, `build`, `store`; `oracle_source` is the store as the oracle's `DossierSource` |
| `config.py` | every state root (`REVIEW_DB`, `DOSSIER_DIR`, `DOSSIER_TRIAL_ROOT`, `REVIEW_CRITERIA_DIR`, `DOSSIER_CACHE_DIR`, then `HUGPY_CURATION_ROOT`, then `hugpy_platform.app_dirs`) |
| `providers.py` | `DoctrineSource` seam (fleet doctrine read by the dossier judge; null default) |
| `cli.py` | `hugpy-curation review ...` / `dossier list` / `install-providers` |
| `deploy/` | `hugpy-review@.service` + `.timer` (system and `user/` variants) |

## Wiring

`hugpy_curation.install_providers(doctrine_source=None)` registers the dossier
store with `hugpy_oracle.providers.set_dossier_source` and, when given, a
fleet doctrine adapter with `hugpy_curation.providers.set_doctrine_source`.
`hugpy_server` calls it at startup; `hugpy-curation install-providers` does
the same in a standalone process.

Downloads never run from curation's own code: with a live `hugpy-downloader`
daemon the review enqueues on `hugpy_storage.downloader.queue`; without one
it calls `hugpy_storage.download_models.download_one` in-process.
