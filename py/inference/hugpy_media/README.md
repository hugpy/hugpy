# hugpy-media

`hugpy_media` — extracted from `abstract_hugpy_dev` as part of the Hugpy
partition. Ownership and allowed dependencies are declared in
`py/partition.toml`; see `PARTITION.md` at the workspace root.

Allowed Python dependencies inside the ecosystem: hugpy_platform, hugpy_storage, hugpy_engine.

Register the media tasks with the engine via `hugpy_media.plugin.register()` (also the `hugpy_engine.tasks` entry point `media`). Heavy stacks are extras: `keywords`, `audio`, `imagegen`, `embed`, `transformers`, `vision`, `extract`, `tts`, `comfy`, umbrella `all`.
