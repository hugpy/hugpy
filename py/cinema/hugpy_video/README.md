# hugpy-video

Execution and artifacts for Hugpy video work, extracted from the monolith's
`abstract_hugpy_dev.video_intel` (+ `managers/video`, `managers/video_gen`,
`comms/studio_assist_log`). Import name `hugpy_video`.

**Status: extracted.** Runtime code lives here; the package imports with the
monolith blocked and with no optional stack (torch/diffusers/ffmpeg bindings)
installed. Allowed ecosystem imports: `hugpy_platform`, `hugpy_control`,
`hugpy_engine`, `hugpy_media`. Never `hugpy_oracle`, `hugpy_fleet`,
`hugpy_server` (see `py/partition.toml`, `PARTITION.md`).

## Layout
- `intel/` — media references + jail (`media_store`), sqlite media job bus
  (`media_bus`) and its control-plane bridge (`job_bridge`), frozen job specs
  and registry (`job_schema`), presets, GPU reservation engine
  (`reservation/`), runners (ffmpeg, diffusers plane, studio, identity relays,
  MLT), the studio spine (`studio/`).
- `chat_video/` — chat-side frame analysis over a `hugpy_media` vision runner.
- `video_gen/` — `StudioVideoRunner` / `VideoGenRequest`: the engine task
  runner for text-to-video / image-to-video.
- `plugin.py` — `hugpy_engine.tasks` entry point (`hugpy_video.plugin:register`);
  import-light, runner resolved lazily.
- `hooks.py` — `PromptCoordinator` protocol (oracle installs its implementation;
  no-op default). `jobs.py` — public bus API incl. `register_job` for
  upper-layer jobs (the oracle's `video_performance`).
- `state.py` — injectable state roots (`HUGPY_VIDEO_STATE_DIR`,
  `HUGPY_MEDIA_JOBS_DB`, `HUGPY_RESERVATIONS_DB`, platform storage roots).
- `intel/plane.py` — the one seam onto the engine's `execute_prompt`.
- `cli.py` — `hugpy-video --help | jobs list | jobs registry | selftest | state`.
- `hugpy-video models audit [--json]` inventories every declared model's
  runner gaps, weight pin and minimum VRAM without loading weights.
- `config.py` — canonical `HUGPY_API_KEY` / `HUGPY_BASE` loader.

Extras: `render` (numpy/Pillow/requests), `studio` (torch/diffusers zoo),
`identity` (abstract-identity client), `test`.

The `identity` extra installs the separate `abstract-identity` distribution.
The central video runner relays mesh and char360 jobs to its service using
`IDENTITY_RENDER_URL` and `IDENTITY_RENDER_TOKEN`; installing the extra alone
does not start a GPU service. Studio model sweeps use
`POST /video/studio/tester`; each attempt records its model, input image,
sampler settings, frame count and negative prompt in the battery log.

## Rules (inherited from char360 / hugpy-agent)
- Heavy imports lazy — must import on a CPU-only box.
- Version single-sourced from pyproject (importlib.metadata; no hardcoded `__version__`).
- Credentials only via `hugpy_video.config.load_config` — canonical names
  `HUGPY_API_KEY` / `HUGPY_BASE`; legacy aliases accepted but recorded in
  `Config.deprecations`.
- Build: `python -m build`; publish = copy dists (console dir needs the sha256
  sidecar; the pypi index computes hashes itself, dir name must be the
  normalized dashed name `hugpy-video`).
