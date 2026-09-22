# Hugpy Python distributions

This directory is the target home for independently buildable Python
distributions extracted from the retired `abstract_hugpy_dev` monolith (history: `unwired/archive/`).

The architecture and migration order are documented in
[`../PARTITION.md`](../PARTITION.md). Exact source ownership and allowed
dependencies are recorded in [`partition.toml`](partition.toml).
The working extraction checklist is [`../MOVE_MAP.md`](../MOVE_MAP.md).

Validate the map with:

```bash
python validate_partition.py
```

## Routes

- `foundation/` — dependency-light platform and control libraries;
- `inference/` — prompt, model and media inference runtimes;
- `storage/` — model artifact acquisition and physical inventory;
- `fleet/` — central/worker coordination and worker daemons;
- `cinema/` — video, creative planning and identity rendering;
- `curation/` — model discovery, trials and review;
- `operations/` — diagnostics, chaos and remediation;
- `integrations/` — external-channel adapters such as Discord;
- `services/` — deployable composition services;
- `meta/` — thin user-facing aggregate distributions;
- `tools/` — applications built on the ecosystem;
- `unwired/` — quarantined, retired or not-yet-owned code; never published.

Existing packages are not absorbed simply because they share a feature. Their
service or library boundaries remain intact and are composed through documented
public contracts.
