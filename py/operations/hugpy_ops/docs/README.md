# hugpy-ops docs

| Document | Subject |
|---|---|
| `ops-reliability.md` | Mechanics of the operational daemons/CLIs this package owns: sentinel (probe), chaos (exercise), provisioner (fetch-declared), keeper and todo-keeper. The ops half of the monolith's `docs/mechanics/curation-reliability.md`; the curation half (discovery dossier, review pipeline) belongs to `hugpy-curation`, the downloader to `hugpy-storage`, fleet doctrine to `hugpy-fleet`. |
| `../deploy/SENTINEL.md` | Operator notes for the sentinel: install, remedy gates, brain ladder / pilot light, knobs. |

Code paths in these documents refer to the extracted packages
(`hugpy_ops/...`, `hugpy_storage/...`, `hugpy_engine/...`).
