# hugpy-server

`hugpy_server` — the Flask composition root of the Hugpy ecosystem, extracted
from `abstract_hugpy_dev`. Ownership and allowed dependencies are declared in
`py/partition.toml`; see `PARTITION.md` at the workspace root.

* `hugpy_server.wsgi_app.create_app()` / `get_hugpy_flask()` build the app:
  every route blueprint, the `/api` dual mounts, the operator/member/video
  gates and the packaged React console (`console_dist/`, see
  `BUILD_CONSOLE.md`).
* `hugpy_server.wiring.install_all()` installs every cross-package seam listed
  in `py/WIRING.md` (fleet placement into the engine, oracle providers, task
  plugins, catalog bridge, oracle video hooks, curation providers, storage
  footprint selector, HF-token listener, control bus, eviction sink). The
  report lands at `app.extensions["hugpy_wiring"]`.
* `hugpy-serve` (`hugpy_server.wsgi_app:main`) serves with gunicorn on POSIX,
  waitress on Windows, and the Flask dev server as the last resort.
* Server-owned state (API keys, video share keys, Discord bindings, install
  links) lives under `hugpy_server.state.server_state_dir()`
  (`HUGPY_SERVER_STATE_DIR`, else the platform's `PROJECTS_HOME`).

Allowed Python dependencies inside the ecosystem: hugpy_platform, hugpy_control,
hugpy_storage, hugpy_engine, hugpy_media, hugpy_video, hugpy_oracle, hugpy_fleet,
hugpy_curation (optional, lazy: hugpy_discord, hugpy_ops).

Tests: `cd py/services/hugpy_server && ../../.venv/bin/python -m pytest -q --timeout=120`
(`tests/integration/` holds the multi-package tests moved from the monolith).
