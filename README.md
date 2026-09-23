# hugpy — inference you own

Self-hosted LLM console, OpenAI-compatible API and GPU worker fleet. Pull
models from the Hugging Face Hub, chat with streaming continuation, mint your
own API keys, and pool the GPUs you already have (workstations, laptops, boxes
on another network, even phones) into one fleet.

```bash
pip install "hugpy[server]"
hugpy serve            # console at http://localhost:7002/ , API at /api/v1
```

From a checkout, `./local_install.sh` installs every package editable into
`.venv` (see [`LOCAL_INSTALL.md`](LOCAL_INSTALL.md)).

## What is in this repository

The product is thirteen independently buildable Python distributions plus the
React consoles they serve. Every package has one owner, one import namespace,
and an enforced set of allowed dependencies; the graph is acyclic.

| Directory | Distribution | Role |
|---|---|---|
| `py/foundation/hugpy_platform` | `hugpy-platform` | Configuration, app dirs, hardware/process probes, compatibility shims |
| `py/foundation/hugpy_control` | `hugpy-control` | Event bus, jobs, settings, principals, call log |
| `py/storage/hugpy_storage` | `hugpy-storage` | Downloads, transfer, physical model inventory, on-disk layout |
| `py/inference/hugpy_engine` | `hugpy-engine` | Model registry, allocation, runners, native engines, dispatch |
| `py/inference/hugpy_media` | `hugpy-media` | Speech, TTS, vision, embeddings, keywords, summaries, image generation |
| `py/cinema/hugpy_video` | `hugpy-video` | Video job bus, runners, studio, reservations |
| `py/cinema/hugpy_oracle` | `hugpy-oracle` | Creative planning, model selection, DAG runtime, evaluation |
| `py/fleet/hugpy_fleet` | `hugpy-fleet` | Central worker registry and the full, GGUF and phone workers |
| `py/curation/hugpy_curation` | `hugpy-curation` | Model discovery dossiers and the review pipeline |
| `py/operations/hugpy_ops` | `hugpy-ops` | Sentinel, chaos, keeper, provisioner |
| `py/integrations/hugpy_discord` | `hugpy-discord` | Discord bot over the HTTP API |
| `py/services/hugpy_server` | `hugpy-server` | Flask composition root, routes, auth, console mounting |
| `py/meta/hugpy` | `hugpy` | The `hugpy` and `hpy` commands and install profiles |
| `react/ui`, `react/agents_ui`, `react/media_intelligence_ui`, `react/video_intelligence_ui`, `react/ui_shared` | `@hugpy/*` | React consoles mounted by `hugpy-server` |

Related repositories: [hugpy-agent](https://github.com/hugpy/hugpy-agent)
(dependency-free agent runtime), [hugpy-station](https://github.com/hugpy/hugpy-station)
(desktop cockpit), [abstract-identity](https://github.com/hugpy/abstract-identity)
(identity pipeline for video).

## Architecture and contribution

- [`PARTITION.md`](PARTITION.md): package boundaries, dependency direction, state ownership.
- [`py/partition.toml`](py/partition.toml): the machine-readable ownership manifest; `python py/validate_partition.py --edges` is the structural gate.
- [`py/WIRING.md`](py/WIRING.md): what the composition roots install at startup.
- [`py/EXTRACTION_GUIDE.md`](py/EXTRACTION_GUIDE.md): the working rules every package follows.
- [`ECOSYSTEM.md`](ECOSYSTEM.md): registries, naming, versioning and release plan.
- [`CONSISTENCY.md`](CONSISTENCY.md): a release is a git tag; how the one version reaches PyPI, checkouts and the fleet, and the drift check that proves it.
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md).

Each package builds and tests on its own:

```bash
cd py/inference/hugpy_engine
python -m build --wheel
pytest -q --timeout=120
```

## License

Source-available; see [`LICENSE`](LICENSE). Commercial licensing at
https://hugpy.ai. Support: support@hugpy.ai.
