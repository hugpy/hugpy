# abstract_hugpy_dev relocations

Generated from `abstract_hugpy_dev-0.1.266-py3-none-any.whl` (the last monolith release) and the installed
`hugpy-*` packages by `tools/generate.py`. `import abstract_hugpy_dev` installs
a finder that makes every **old module** below resolve to the **new module**
object (same object, not a copy). Retired aggregators are synthesised as lazy
namespace modules; retired code has no alias and must be replaced.

| Kind | Count |
|---|---|
| Relocated modules (aliased) | 503 |
| Retired aggregators (lazy namespaces) | 55 |
| Retired modules (no alias) | 10 |
| Top-level names re-exported | 657 |
| Top-level names not relocated | 11 |

## Relocated modules per package

| Package | Modules |
|---|---|
| `hugpy_platform` | 14 |
| `hugpy_control` | 7 |
| `hugpy_storage` | 23 |
| `hugpy_engine` | 95 |
| `hugpy_media` | 61 |
| `hugpy_video` | 83 |
| `hugpy_oracle` | 43 |
| `hugpy_fleet` | 63 |
| `hugpy_curation` | 24 |
| `hugpy_ops` | 20 |
| `hugpy_discord` | 12 |
| `hugpy_server` | 56 |
| `hugpy` | 2 |

## Old module -> new module

| Old module | New module |
|---|---|
| `abstract_hugpy_dev._compat_pydantic` | `hugpy_platform.compat_pydantic` |
| `abstract_hugpy_dev._platform` | `hugpy_platform.platform_facade` |
| `abstract_hugpy_dev._platform.async_runtime` | `hugpy_platform.async_runtime` |
| `abstract_hugpy_dev._platform.binaries` | `hugpy_platform.binaries` |
| `abstract_hugpy_dev._platform.client_liveness` | `hugpy_platform.client_liveness` |
| `abstract_hugpy_dev._platform.hardware` | `hugpy_platform.hardware` |
| `abstract_hugpy_dev._platform.paths` | `hugpy_platform.app_dirs` |
| `abstract_hugpy_dev._platform.procutil` | `hugpy_platform.procutil` |
| `abstract_hugpy_dev.bot` | `hugpy_discord` |
| `abstract_hugpy_dev.bot.bot` | `hugpy_discord.bot` |
| `abstract_hugpy_dev.bot.cogs` | `hugpy_discord.cogs` |
| `abstract_hugpy_dev.bot.cogs.chat` | `hugpy_discord.cogs.chat` |
| `abstract_hugpy_dev.bot.cogs.helpers` | `hugpy_discord.cogs.helpers` |
| `abstract_hugpy_dev.bot.cogs.ml` | `hugpy_discord.cogs.ml` |
| `abstract_hugpy_dev.bot.cogs.ops` | `hugpy_discord.cogs.ops` |
| `abstract_hugpy_dev.bot.cogs.tools` | `hugpy_discord.cogs.tools` |
| `abstract_hugpy_dev.bot.config` | `hugpy_discord.config` |
| `abstract_hugpy_dev.bot.hugpy_client` | `hugpy_discord.hugpy_client` |
| `abstract_hugpy_dev.bot.prefs` | `hugpy_discord.prefs` |
| `abstract_hugpy_dev.bot.streamer` | `hugpy_discord.streamer` |
| `abstract_hugpy_dev.central` | `hugpy_platform.central` |
| `abstract_hugpy_dev.chaos` | `hugpy_ops.chaos` |
| `abstract_hugpy_dev.chaos.__main__` | `hugpy_ops.chaos.__main__` |
| `abstract_hugpy_dev.chaos.alloc` | `hugpy_ops.chaos.alloc` |
| `abstract_hugpy_dev.chaos.assortment` | `hugpy_ops.chaos.assortment` |
| `abstract_hugpy_dev.chaos.client` | `hugpy_ops.chaos.client` |
| `abstract_hugpy_dev.chaos.observe` | `hugpy_ops.chaos.observe` |
| `abstract_hugpy_dev.chaos.runner` | `hugpy_ops.chaos.runner` |
| `abstract_hugpy_dev.chaos.schema` | `hugpy_ops.chaos.schema` |
| `abstract_hugpy_dev.chaos.sweep` | `hugpy_ops.chaos.sweep` |
| `abstract_hugpy_dev.cli` | `hugpy.cli` |
| `abstract_hugpy_dev.comms.agent_nodes` | `hugpy_fleet.central.agent_nodes` |
| `abstract_hugpy_dev.comms.blocklist` | `hugpy_fleet.central.blocklist` |
| `abstract_hugpy_dev.comms.bus` | `hugpy_control.bus` |
| `abstract_hugpy_dev.comms.calibration` | `hugpy_fleet.central.calibration` |
| `abstract_hugpy_dev.comms.calllog` | `hugpy_control.calllog` |
| `abstract_hugpy_dev.comms.evict_policy` | `hugpy_fleet.central.evict_policy` |
| `abstract_hugpy_dev.comms.evictions` | `hugpy_fleet.central.evictions` |
| `abstract_hugpy_dev.comms.feeds` | `hugpy_fleet.central.feeds` |
| `abstract_hugpy_dev.comms.heartbeat_db` | `hugpy_fleet.central.heartbeat_db` |
| `abstract_hugpy_dev.comms.hf_metadata` | `hugpy_storage.hf_metadata` |
| `abstract_hugpy_dev.comms.jobs` | `hugpy_control.jobs` |
| `abstract_hugpy_dev.comms.model_metadata` | `hugpy_storage.model_metadata` |
| `abstract_hugpy_dev.comms.model_metrics` | `hugpy_fleet.central.model_metrics` |
| `abstract_hugpy_dev.comms.model_physical` | `hugpy_storage.model_physical` |
| `abstract_hugpy_dev.comms.model_status_cache` | `hugpy_storage.model_status_cache` |
| `abstract_hugpy_dev.comms.principals` | `hugpy_control.principals` |
| `abstract_hugpy_dev.comms.priority_groups` | `hugpy_fleet.central.priority_groups` |
| `abstract_hugpy_dev.comms.settings` | `hugpy_control.settings` |
| `abstract_hugpy_dev.comms.shared` | `hugpy_control.shared` |
| `abstract_hugpy_dev.comms.studio_assist_log` | `hugpy_video.studio_assist_log` |
| `abstract_hugpy_dev.comms.task_templates` | `hugpy_fleet.central.task_templates` |
| `abstract_hugpy_dev.comms.todo_keeper` | `hugpy_ops.todo_keeper` |
| `abstract_hugpy_dev.comms.todo_keeper_daemon` | `hugpy_ops.todo_keeper_daemon` |
| `abstract_hugpy_dev.discovery_dossier` | `hugpy_curation.dossier` |
| `abstract_hugpy_dev.discovery_dossier.build` | `hugpy_curation.dossier.build` |
| `abstract_hugpy_dev.discovery_dossier.cards` | `hugpy_curation.dossier.cards` |
| `abstract_hugpy_dev.discovery_dossier.community` | `hugpy_curation.dossier.community` |
| `abstract_hugpy_dev.discovery_dossier.dossier` | `hugpy_curation.dossier.dossier` |
| `abstract_hugpy_dev.discovery_dossier.fetch` | `hugpy_curation.dossier.fetch` |
| `abstract_hugpy_dev.discovery_dossier.llm` | `hugpy_curation.dossier.llm` |
| `abstract_hugpy_dev.discovery_dossier.radar` | `hugpy_curation.dossier.radar` |
| `abstract_hugpy_dev.discovery_dossier.research` | `hugpy_curation.dossier.research` |
| `abstract_hugpy_dev.discovery_dossier.screening` | `hugpy_curation.dossier.screening` |
| `abstract_hugpy_dev.discovery_dossier.store` | `hugpy_curation.dossier.store` |
| `abstract_hugpy_dev.discovery_dossier.trial` | `hugpy_curation.dossier.trial` |
| `abstract_hugpy_dev.discovery_dossier.verdicts` | `hugpy_curation.dossier.verdicts` |
| `abstract_hugpy_dev.discovery_dossier.weights` | `hugpy_curation.dossier.weights` |
| `abstract_hugpy_dev.downloader` | `hugpy_storage.downloader` |
| `abstract_hugpy_dev.downloader.__main__` | `hugpy_storage.downloader.__main__` |
| `abstract_hugpy_dev.downloader.daemon` | `hugpy_storage.downloader.daemon` |
| `abstract_hugpy_dev.downloader.engine` | `hugpy_storage.downloader.engine` |
| `abstract_hugpy_dev.downloader.presence` | `hugpy_storage.downloader.presence` |
| `abstract_hugpy_dev.downloader.queue` | `hugpy_storage.downloader.queue` |
| `abstract_hugpy_dev.engine` | `hugpy_engine.native` |
| `abstract_hugpy_dev.engine.build` | `hugpy_engine.native.build` |
| `abstract_hugpy_dev.engine.fetch` | `hugpy_engine.native.fetch` |
| `abstract_hugpy_dev.engine.resolve` | `hugpy_engine.native.resolve` |
| `abstract_hugpy_dev.flask_app.app` | `hugpy_server.app` |
| `abstract_hugpy_dev.flask_app.app.endpoints_explorer` | `hugpy_server.app.endpoints_explorer` |
| `abstract_hugpy_dev.flask_app.app.endpoints_view` | `hugpy_server.app.endpoints_view` |
| `abstract_hugpy_dev.flask_app.app.functions` | `hugpy_server.app.functions` |
| `abstract_hugpy_dev.flask_app.app.functions.chat` | `hugpy_server.app.functions.chat` |
| `abstract_hugpy_dev.flask_app.app.functions.chat.streaming` | `hugpy_server.app.functions.chat.streaming` |
| `abstract_hugpy_dev.flask_app.app.functions.chat.task_selection` | `hugpy_server.app.functions.chat.task_selection` |
| `abstract_hugpy_dev.flask_app.app.functions.downloads.cancelable_downloads` | `hugpy_storage.console.cancelable_downloads` |
| `abstract_hugpy_dev.flask_app.app.functions.downloads.downloader` | `hugpy_storage.console.downloader` |
| `abstract_hugpy_dev.flask_app.app.functions.downloads.downloads` | `hugpy_storage.console.downloads` |
| `abstract_hugpy_dev.flask_app.app.functions.downloads.model_physical` | `hugpy_storage.console.model_physical` |
| `abstract_hugpy_dev.flask_app.app.functions.imports` | `hugpy_server.app.functions.imports` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.options` | `hugpy_server.app.functions.imports.options` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.options.install` | `hugpy_server.app.functions.imports.options.install` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.options.search` | `hugpy_server.app.functions.imports.options.search` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils` | `hugpy_server.app.functions.imports.utils` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.api_keys` | `hugpy_server.app.functions.imports.utils.api_keys` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.constants` | `hugpy_server.app.functions.imports.utils.constants` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.discord_bindings` | `hugpy_server.app.functions.imports.utils.discord_bindings` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.enrollment_tokens` | `hugpy_fleet.central.enrollment_tokens` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.format_select` | `hugpy_storage.format_select` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.hf_token` | `hugpy_storage.hf_token` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.install_links` | `hugpy_server.app.functions.imports.utils.install_links` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.manifest` | `hugpy_engine.manifest` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.model_groups` | `hugpy_fleet.central.model_groups` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.peers` | `hugpy_fleet.central.peers` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.phone_brick_store` | `hugpy_fleet.central.phone_brick_store` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.pid_attribution` | `hugpy_fleet.central.pid_attribution` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.pool_guard` | `hugpy_fleet.central.pool_guard` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.priority_groups` | `hugpy_fleet.central.priority_group_settings` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.chat_schemas` | `hugpy_engine.wire.chat_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.config_schemas` | `hugpy_engine.wire.config_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.download_schemas` | `hugpy_storage.schemas.download_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.install_schemas` | `hugpy_engine.wire.install_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.job_schemas` | `hugpy_control.job_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.model_schemas` | `hugpy_engine.wire.model_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.request_schemas` | `hugpy_engine.wire.request_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.specs_schemas` | `hugpy_engine.wire.specs_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.video_share_keys` | `hugpy_server.app.functions.imports.utils.video_share_keys` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.worker_http` | `hugpy_fleet.central.worker_http` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.workers` | `hugpy_fleet.central.workers` |
| `abstract_hugpy_dev.flask_app.app.functions.media_extract` | `hugpy_media.extract` |
| `abstract_hugpy_dev.flask_app.app.keeper_line` | `hugpy_server.app.keeper_line` |
| `abstract_hugpy_dev.flask_app.app.member_auth` | `hugpy_server.app.member_auth` |
| `abstract_hugpy_dev.flask_app.app.operator_auth` | `hugpy_server.app.operator_auth` |
| `abstract_hugpy_dev.flask_app.app.routes` | `hugpy_server.app.routes` |
| `abstract_hugpy_dev.flask_app.app.routes.agent_routes` | `hugpy_server.app.routes.agent_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.auth_proxy_routes` | `hugpy_server.app.routes.auth_proxy_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.chat_routes` | `hugpy_server.app.routes.chat_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.comms_routes` | `hugpy_server.app.routes.comms_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.discord_routes` | `hugpy_server.app.routes.discord_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.eviction_routes` | `hugpy_server.app.routes.eviction_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.fleet_doctrine_routes` | `hugpy_server.app.routes.fleet_doctrine_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.fleet_routes` | `hugpy_server.app.routes.fleet_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.group_routes` | `hugpy_server.app.routes.group_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.installer_assets.generate_icons` | `hugpy_server.app.routes.installer_assets.generate_icons` |
| `abstract_hugpy_dev.flask_app.app.routes.interim_routes` | `hugpy_server.app.routes.interim_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.keeper_help_routes` | `hugpy_server.app.routes.keeper_help_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.llm_storage_routes` | `hugpy_server.app.routes.llm_storage_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.messages_helpers` | `hugpy_server.app.routes.messages_helpers` |
| `abstract_hugpy_dev.flask_app.app.routes.messages_routes` | `hugpy_server.app.routes.messages_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.metrics_routes` | `hugpy_server.app.routes.metrics_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.ml_routes` | `hugpy_server.app.routes.ml_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.model_group_routes` | `hugpy_server.app.routes.model_group_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.oracle_routes` | `hugpy_server.app.routes.oracle_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.phone_brick_routes` | `hugpy_server.app.routes.phone_brick_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.prompt_routes` | `hugpy_server.app.routes.prompt_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.pypi_routes` | `hugpy_server.app.routes.pypi_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.review_routes` | `hugpy_server.app.routes.review_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.script_first_routes` | `hugpy_server.app.routes.script_first_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.search_routes` | `hugpy_server.app.routes.search_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.upload_routes` | `hugpy_server.app.routes.upload_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.v1_helpers` | `hugpy_server.app.routes.v1_helpers` |
| `abstract_hugpy_dev.flask_app.app.routes.v1_routes` | `hugpy_server.app.routes.v1_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.video_assist_media` | `hugpy_server.app.routes.video_assist_media` |
| `abstract_hugpy_dev.flask_app.app.routes.video_coordination` | `hugpy_server.app.routes.video_coordination` |
| `abstract_hugpy_dev.flask_app.app.routes.video_routes` | `hugpy_server.app.routes.video_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.welcome_routes` | `hugpy_server.app.routes.welcome_routes` |
| `abstract_hugpy_dev.flask_app.app.routes.worker_routes` | `hugpy_server.app.routes.worker_routes` |
| `abstract_hugpy_dev.flask_app.app.video_auth` | `hugpy_server.app.video_auth` |
| `abstract_hugpy_dev.flask_app.wsgi_app` | `hugpy_server.wsgi_app` |
| `abstract_hugpy_dev.fleet_doctrine` | `hugpy_fleet.doctrine` |
| `abstract_hugpy_dev.fleet_doctrine.doctor` | `hugpy_fleet.doctrine.doctor` |
| `abstract_hugpy_dev.fleet_doctrine.doctrine` | `hugpy_fleet.doctrine.doctrine` |
| `abstract_hugpy_dev.gguf_worker` | `hugpy_fleet.gguf_worker` |
| `abstract_hugpy_dev.gguf_worker.__main__` | `hugpy_fleet.gguf_worker.__main__` |
| `abstract_hugpy_dev.gguf_worker.agent` | `hugpy_fleet.gguf_worker.agent` |
| `abstract_hugpy_dev.hpy` | `hugpy.hpy` |
| `abstract_hugpy_dev.imports.apis.call_api` | `hugpy_engine.apis.call_api` |
| `abstract_hugpy_dev.imports.apis.download_models` | `hugpy_storage.download_models` |
| `abstract_hugpy_dev.imports.apis.get_module` | `hugpy_engine.apis.get_module` |
| `abstract_hugpy_dev.imports.apis.huggingface_api` | `hugpy_storage.huggingface_api` |
| `abstract_hugpy_dev.imports.apis.reclassify` | `hugpy_engine.apis.reclassify` |
| `abstract_hugpy_dev.imports.apis.reconcile` | `hugpy_engine.apis.reconcile` |
| `abstract_hugpy_dev.imports.apis.serve` | `hugpy_engine.apis.serve` |
| `abstract_hugpy_dev.imports.apis.serve.serve` | `hugpy_engine.apis.serve.serve` |
| `abstract_hugpy_dev.imports.apis.serve.serve_cli` | `hugpy_engine.apis.serve.serve_cli` |
| `abstract_hugpy_dev.imports.apis.systemd_units` | `hugpy_engine.apis.systemd_units` |
| `abstract_hugpy_dev.imports.config.main` | `hugpy_engine.config.main` |
| `abstract_hugpy_dev.imports.config.models.model_meta` | `hugpy_engine.config.models.model_meta` |
| `abstract_hugpy_dev.imports.config.models.models_config` | `hugpy_engine.config.models.models_config` |
| `abstract_hugpy_dev.imports.config.models.models_default` | `hugpy_engine.config.models.models_default` |
| `abstract_hugpy_dev.imports.config.models.models_dict` | `hugpy_engine.config.models.models_dict` |
| `abstract_hugpy_dev.imports.src.chunking` | `hugpy_media.chunking` |
| `abstract_hugpy_dev.imports.src.constants.categories` | `hugpy_engine.categories` |
| `abstract_hugpy_dev.imports.src.constants.constants` | `hugpy_platform.constants` |
| `abstract_hugpy_dev.imports.src.constants.hugpy_marker` | `hugpy_storage.hugpy_marker` |
| `abstract_hugpy_dev.imports.src.constants.paths` | `hugpy_storage.model_paths` |
| `abstract_hugpy_dev.imports.src.constants.trust` | `hugpy_platform.trust` |
| `abstract_hugpy_dev.imports.src.except_utils` | `hugpy_platform.except_utils` |
| `abstract_hugpy_dev.imports.src.gguf_election` | `hugpy_engine.gguf_election` |
| `abstract_hugpy_dev.imports.src.model_classifier` | `hugpy_engine.model_classifier` |
| `abstract_hugpy_dev.imports.src.model_index` | `hugpy_engine.model_index` |
| `abstract_hugpy_dev.imports.src.model_index.client` | `hugpy_engine.model_index.client` |
| `abstract_hugpy_dev.imports.src.model_index.query_registry` | `hugpy_engine.model_index.query_registry` |
| `abstract_hugpy_dev.imports.src.model_index.repositories` | `hugpy_engine.model_index.repositories` |
| `abstract_hugpy_dev.imports.src.model_index.service` | `hugpy_engine.model_index.service` |
| `abstract_hugpy_dev.imports.src.module_imports` | `hugpy_platform.module_imports` |
| `abstract_hugpy_dev.imports.src.peft_adapters` | `hugpy_engine.peft_adapters` |
| `abstract_hugpy_dev.imports.src.schemas.chat_schemas` | `hugpy_engine.schemas.chat_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.embeded_schemas` | `hugpy_media.schemas.embeded_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.event_schemas` | `hugpy_engine.schemas.event_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.metadata_schemas` | `hugpy_engine.schemas.metadata_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.model_schemas` | `hugpy_engine.schemas.model_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.runner_schemas` | `hugpy_engine.schemas.runner_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.summarizer_schemas` | `hugpy_media.schemas.summarizer_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.task_schemas` | `hugpy_engine.schemas.task_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.video_schemas` | `hugpy_video.schemas.video_schemas` |
| `abstract_hugpy_dev.imports.src.schemas.whisper_schemas` | `hugpy_media.schemas.whisper_schemas` |
| `abstract_hugpy_dev.imports.src.utils` | `hugpy_platform.utils` |
| `abstract_hugpy_dev.keeper` | `hugpy_ops.keeper` |
| `abstract_hugpy_dev.managers.alloc_modes` | `hugpy_engine.alloc_modes` |
| `abstract_hugpy_dev.managers.chat_context` | `hugpy_engine.chat_context` |
| `abstract_hugpy_dev.managers.chat_context.chat_context` | `hugpy_engine.chat_context.chat_context` |
| `abstract_hugpy_dev.managers.chat_context.context_budget` | `hugpy_engine.chat_context.context_budget` |
| `abstract_hugpy_dev.managers.chat_context.unbounded` | `hugpy_engine.chat_context.unbounded` |
| `abstract_hugpy_dev.managers.comfy` | `hugpy_media.comfy` |
| `abstract_hugpy_dev.managers.comfy.comfy_runner` | `hugpy_media.comfy.comfy_runner` |
| `abstract_hugpy_dev.managers.dispatch` | `hugpy_engine.dispatch` |
| `abstract_hugpy_dev.managers.dispatch.acquire` | `hugpy_engine.dispatch.acquire` |
| `abstract_hugpy_dev.managers.dispatch.activity` | `hugpy_engine.dispatch.activity` |
| `abstract_hugpy_dev.managers.dispatch.dispatch` | `hugpy_engine.dispatch.dispatch` |
| `abstract_hugpy_dev.managers.draft_models` | `hugpy_engine.draft_models` |
| `abstract_hugpy_dev.managers.embed` | `hugpy_media.embed` |
| `abstract_hugpy_dev.managers.embed.embed_runner` | `hugpy_media.embed.embed_runner` |
| `abstract_hugpy_dev.managers.eviction` | `hugpy_engine.eviction` |
| `abstract_hugpy_dev.managers.fleet` | `hugpy_fleet.fleet_manager` |
| `abstract_hugpy_dev.managers.fleet.templates` | `hugpy_fleet.fleet_manager.templates` |
| `abstract_hugpy_dev.managers.generate` | `hugpy_engine.generate` |
| `abstract_hugpy_dev.managers.generate.coder` | `hugpy_engine.generate.coder` |
| `abstract_hugpy_dev.managers.generate.config` | `hugpy_engine.generate.config` |
| `abstract_hugpy_dev.managers.generate.generate_runner` | `hugpy_engine.generate.generate_runner` |
| `abstract_hugpy_dev.managers.imagegen` | `hugpy_media.imagegen` |
| `abstract_hugpy_dev.managers.imagegen.imagegen_runner` | `hugpy_media.imagegen.imagegen_runner` |
| `abstract_hugpy_dev.managers.imagegen.schemas` | `hugpy_media.imagegen.schemas` |
| `abstract_hugpy_dev.managers.imagegen.vram_retry` | `hugpy_media.imagegen.vram_retry` |
| `abstract_hugpy_dev.managers.keywords` | `hugpy_media.keywords` |
| `abstract_hugpy_dev.managers.keywords.keybert_model` | `hugpy_media.keywords.keybert_model` |
| `abstract_hugpy_dev.managers.keywords.keywords_runner` | `hugpy_media.keywords.keywords_runner` |
| `abstract_hugpy_dev.managers.keywords.schemas` | `hugpy_media.keywords.schemas` |
| `abstract_hugpy_dev.managers.llama` | `hugpy_engine.llama` |
| `abstract_hugpy_dev.managers.llama.runners` | `hugpy_engine.llama.runners` |
| `abstract_hugpy_dev.managers.llama.runners.chat_runner` | `hugpy_engine.llama.runners.chat_runner` |
| `abstract_hugpy_dev.managers.llama.runners.get` | `hugpy_engine.llama.runners.get` |
| `abstract_hugpy_dev.managers.llama.runners.src` | `hugpy_engine.llama.runners.src` |
| `abstract_hugpy_dev.managers.llama.runners.src.base_runner` | `hugpy_engine.llama.runners.src.base_runner` |
| `abstract_hugpy_dev.managers.llama.runners.src.ccp_runner` | `hugpy_engine.llama.runners.src.ccp_runner` |
| `abstract_hugpy_dev.managers.llama.runners.src.imports` | `hugpy_engine.llama.runners.src.imports` |
| `abstract_hugpy_dev.managers.llama.runners.src.imports.config` | `hugpy_engine.llama.runners.src.imports.config` |
| `abstract_hugpy_dev.managers.llama.runners.src.imports.constants` | `hugpy_engine.llama.runners.src.imports.constants` |
| `abstract_hugpy_dev.managers.llama.runners.src.imports.init_imports` | `hugpy_engine.llama.runners.src.imports.init_imports` |
| `abstract_hugpy_dev.managers.llama.runners.src.imports.utils` | `hugpy_engine.llama.runners.src.imports.utils` |
| `abstract_hugpy_dev.managers.llama.runners.src.python_runner` | `hugpy_engine.llama.runners.src.python_runner` |
| `abstract_hugpy_dev.managers.llama.runners.src.shard_server` | `hugpy_engine.llama.runners.src.shard_server` |
| `abstract_hugpy_dev.managers.llama.serve` | `hugpy_engine.llama.serve` |
| `abstract_hugpy_dev.managers.phone_brick_orchestrator` | `hugpy_fleet.phone_brick_orchestrator` |
| `abstract_hugpy_dev.managers.phone_brick_orchestrator.runner` | `hugpy_fleet.phone_brick_orchestrator.runner` |
| `abstract_hugpy_dev.managers.resolvers` | `hugpy_engine.resolvers` |
| `abstract_hugpy_dev.managers.resolvers.allocator` | `hugpy_engine.resolvers.allocator` |
| `abstract_hugpy_dev.managers.resolvers.assure_model_key` | `hugpy_engine.resolvers.assure_model_key` |
| `abstract_hugpy_dev.managers.resolvers.categories` | `hugpy_engine.resolvers.categories` |
| `abstract_hugpy_dev.managers.resolvers.categories.builders` | `hugpy_engine.resolvers.categories.builders` |
| `abstract_hugpy_dev.managers.resolvers.categories.frameworks` | `hugpy_engine.resolvers.categories.frameworks` |
| `abstract_hugpy_dev.managers.resolvers.groups` | `hugpy_engine.resolvers.groups` |
| `abstract_hugpy_dev.managers.resolvers.model_dict_resolver` | `hugpy_engine.resolvers.model_dict_resolver` |
| `abstract_hugpy_dev.managers.resolvers.model_resolver` | `hugpy_engine.resolvers.model_resolver` |
| `abstract_hugpy_dev.managers.resolvers.remote` | `hugpy_engine.resolvers.remote` |
| `abstract_hugpy_dev.managers.serve` | `hugpy_engine.serve` |
| `abstract_hugpy_dev.managers.serve.chunksum_verify` | `hugpy_engine.serve.chunksum_verify` |
| `abstract_hugpy_dev.managers.serve.hot_cache` | `hugpy_engine.serve.hot_cache` |
| `abstract_hugpy_dev.managers.serve.model_cache` | `hugpy_engine.serve.model_cache` |
| `abstract_hugpy_dev.managers.serve.overrides` | `hugpy_engine.serve.overrides` |
| `abstract_hugpy_dev.managers.serve.policy` | `hugpy_engine.serve.policy` |
| `abstract_hugpy_dev.managers.serve.profiles` | `hugpy_engine.serve.profiles` |
| `abstract_hugpy_dev.managers.serve.serve` | `hugpy_engine.serve.serve` |
| `abstract_hugpy_dev.managers.serve.serve_cli` | `hugpy_engine.serve.serve_cli` |
| `abstract_hugpy_dev.managers.serve.slot_agent` | `hugpy_engine.serve.slot_agent` |
| `abstract_hugpy_dev.managers.serve.slots` | `hugpy_engine.serve.slots` |
| `abstract_hugpy_dev.managers.serve.supervisor` | `hugpy_engine.serve.supervisor` |
| `abstract_hugpy_dev.managers.spill` | `hugpy_engine.spill` |
| `abstract_hugpy_dev.managers.summarizers` | `hugpy_media.summarizers` |
| `abstract_hugpy_dev.managers.summarizers.generation` | `hugpy_media.summarizers.generation` |
| `abstract_hugpy_dev.managers.summarizers.media` | `hugpy_media.summarizers.media` |
| `abstract_hugpy_dev.managers.summarizers.summarize_runner` | `hugpy_media.summarizers.summarize_runner` |
| `abstract_hugpy_dev.managers.summarizers.summarizers` | `hugpy_media.summarizers.summarizers` |
| `abstract_hugpy_dev.managers.task_deps` | `hugpy_engine.task_deps` |
| `abstract_hugpy_dev.managers.toks_report` | `hugpy_fleet.toks_report` |
| `abstract_hugpy_dev.managers.tts` | `hugpy_media.tts` |
| `abstract_hugpy_dev.managers.tts._backend_main` | `hugpy_media.tts._backend_main` |
| `abstract_hugpy_dev.managers.tts.schemas` | `hugpy_media.tts.schemas` |
| `abstract_hugpy_dev.managers.tts.seat` | `hugpy_media.tts.seat` |
| `abstract_hugpy_dev.managers.tts.tts_runner` | `hugpy_media.tts.tts_runner` |
| `abstract_hugpy_dev.managers.video` | `hugpy_video.chat_video` |
| `abstract_hugpy_dev.managers.video.video_analyzer` | `hugpy_video.chat_video.video_analyzer` |
| `abstract_hugpy_dev.managers.video_gen` | `hugpy_video.video_gen` |
| `abstract_hugpy_dev.managers.video_gen.schemas` | `hugpy_video.video_gen.schemas` |
| `abstract_hugpy_dev.managers.video_gen.video_gen_runner` | `hugpy_video.video_gen.video_gen_runner` |
| `abstract_hugpy_dev.managers.vision` | `hugpy_media.vision` |
| `abstract_hugpy_dev.managers.vision.schemas` | `hugpy_media.vision.schemas` |
| `abstract_hugpy_dev.managers.vision.utils` | `hugpy_media.vision.utils` |
| `abstract_hugpy_dev.managers.vision.vision_backends` | `hugpy_media.vision.vision_backends` |
| `abstract_hugpy_dev.managers.vision.vision_coder` | `hugpy_media.vision.vision_coder` |
| `abstract_hugpy_dev.managers.vision.vision_runner` | `hugpy_media.vision.vision_runner` |
| `abstract_hugpy_dev.managers.vision_analysis` | `hugpy_media.vision_analysis` |
| `abstract_hugpy_dev.managers.vision_analysis.runner` | `hugpy_media.vision_analysis.runner` |
| `abstract_hugpy_dev.managers.vision_analysis.schemas` | `hugpy_media.vision_analysis.schemas` |
| `abstract_hugpy_dev.managers.whisper_model` | `hugpy_media.whisper_model` |
| `abstract_hugpy_dev.managers.whisper_model.constants` | `hugpy_media.whisper_model.constants` |
| `abstract_hugpy_dev.managers.whisper_model.src` | `hugpy_media.whisper_model.src` |
| `abstract_hugpy_dev.managers.whisper_model.src.model` | `hugpy_media.whisper_model.src.model` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.execute` | `hugpy_media.whisper_model.src.model.execute` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.model` | `hugpy_media.whisper_model.src.model.model` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils` | `hugpy_media.whisper_model.src.model.utils` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.audio` | `hugpy_media.whisper_model.src.model.utils.audio` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files` | `hugpy_media.whisper_model.src.model.utils.files` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.artifacts` | `hugpy_media.whisper_model.src.model.utils.files.artifacts` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.artifacts.workspace` | `hugpy_media.whisper_model.src.model.utils.files.artifacts.workspace` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.frames` | `hugpy_media.whisper_model.src.model.utils.files.frames` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.frames.extract` | `hugpy_media.whisper_model.src.model.utils.files.frames.extract` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.frames.utils` | `hugpy_media.whisper_model.src.model.utils.files.frames.utils` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.save` | `hugpy_media.whisper_model.src.model.utils.files.save` |
| `abstract_hugpy_dev.managers.whisper_model.src.runner` | `hugpy_media.whisper_model.src.runner` |
| `abstract_hugpy_dev.managers.whisper_model.src.stream` | `hugpy_media.whisper_model.src.stream` |
| `abstract_hugpy_dev.model_battery` | `hugpy_media.model_battery` |
| `abstract_hugpy_dev.model_sync` | `hugpy_storage.model_sync` |
| `abstract_hugpy_dev.oracle` | `hugpy_oracle` |
| `abstract_hugpy_dev.oracle.audio_master` | `hugpy_oracle.audio_master` |
| `abstract_hugpy_dev.oracle.authority` | `hugpy_oracle.authority` |
| `abstract_hugpy_dev.oracle.benchmark` | `hugpy_oracle.benchmark` |
| `abstract_hugpy_dev.oracle.benchmark_cases` | `hugpy_oracle.benchmark_cases` |
| `abstract_hugpy_dev.oracle.catalog` | `hugpy_oracle.catalog` |
| `abstract_hugpy_dev.oracle.contracts` | `hugpy_oracle.contracts` |
| `abstract_hugpy_dev.oracle.dag_runtime` | `hugpy_oracle.dag_runtime` |
| `abstract_hugpy_dev.oracle.evaluation` | `hugpy_oracle.evaluation` |
| `abstract_hugpy_dev.oracle.interim_ledger` | `hugpy_oracle.interim_ledger` |
| `abstract_hugpy_dev.oracle.performance` | `hugpy_oracle.performance` |
| `abstract_hugpy_dev.oracle.pipeline` | `hugpy_oracle.pipeline` |
| `abstract_hugpy_dev.oracle.pipelines` | `hugpy_oracle.pipelines` |
| `abstract_hugpy_dev.oracle.plan` | `hugpy_oracle.plan` |
| `abstract_hugpy_dev.oracle.postproduction` | `hugpy_oracle.postproduction` |
| `abstract_hugpy_dev.oracle.probes` | `hugpy_oracle.probes` |
| `abstract_hugpy_dev.oracle.production` | `hugpy_oracle.production` |
| `abstract_hugpy_dev.oracle.prompt_compiler` | `hugpy_oracle.prompt_compiler` |
| `abstract_hugpy_dev.oracle.recipes` | `hugpy_oracle.recipes` |
| `abstract_hugpy_dev.oracle.recipes.video_performance` | `hugpy_oracle.recipes.video_performance` |
| `abstract_hugpy_dev.oracle.repair` | `hugpy_oracle.repair` |
| `abstract_hugpy_dev.oracle.repair_controller` | `hugpy_oracle.repair_controller` |
| `abstract_hugpy_dev.oracle.router` | `hugpy_oracle.router` |
| `abstract_hugpy_dev.oracle.routing_matrix` | `hugpy_oracle.routing_matrix` |
| `abstract_hugpy_dev.oracle.runtime` | `hugpy_oracle.runtime` |
| `abstract_hugpy_dev.oracle.schema_export` | `hugpy_oracle.schema_export` |
| `abstract_hugpy_dev.oracle.scorecard` | `hugpy_oracle.scorecard` |
| `abstract_hugpy_dev.oracle.screenplay` | `hugpy_oracle.screenplay` |
| `abstract_hugpy_dev.oracle.script_first` | `hugpy_oracle.script_first` |
| `abstract_hugpy_dev.oracle.segments` | `hugpy_oracle.segments` |
| `abstract_hugpy_dev.oracle.selection` | `hugpy_oracle.selection` |
| `abstract_hugpy_dev.oracle.spatial` | `hugpy_oracle.spatial` |
| `abstract_hugpy_dev.oracle.spatial_eval` | `hugpy_oracle.spatial_eval` |
| `abstract_hugpy_dev.oracle.spatial_sources` | `hugpy_oracle.spatial_sources` |
| `abstract_hugpy_dev.oracle.speech` | `hugpy_oracle.speech` |
| `abstract_hugpy_dev.oracle.stationary_scenario` | `hugpy_oracle.stationary_scenario` |
| `abstract_hugpy_dev.oracle.steward` | `hugpy_oracle.steward` |
| `abstract_hugpy_dev.oracle.storyboard` | `hugpy_oracle.storyboard` |
| `abstract_hugpy_dev.oracle.tone_scale` | `hugpy_oracle.tone_scale` |
| `abstract_hugpy_dev.oracle.validator` | `hugpy_oracle.validator` |
| `abstract_hugpy_dev.oracle.worldbuild` | `hugpy_oracle.worldbuild` |
| `abstract_hugpy_dev.phone_brick` | `hugpy_fleet.phone_brick` |
| `abstract_hugpy_dev.phone_brick.__main__` | `hugpy_fleet.phone_brick.__main__` |
| `abstract_hugpy_dev.phone_brick.analyze` | `hugpy_fleet.phone_brick.analyze` |
| `abstract_hugpy_dev.phone_brick.client` | `hugpy_fleet.phone_brick.client` |
| `abstract_hugpy_dev.phone_brick.consensus` | `hugpy_fleet.phone_brick.consensus` |
| `abstract_hugpy_dev.phone_brick.detector` | `hugpy_fleet.phone_brick.detector` |
| `abstract_hugpy_dev.phone_brick.orchestrator` | `hugpy_fleet.phone_brick.orchestrator` |
| `abstract_hugpy_dev.phone_brick.protocol` | `hugpy_fleet.phone_brick.protocol` |
| `abstract_hugpy_dev.phone_brick.registration` | `hugpy_fleet.phone_brick.registration` |
| `abstract_hugpy_dev.phone_brick.rendering` | `hugpy_fleet.phone_brick.rendering` |
| `abstract_hugpy_dev.phone_brick.rpc_backend` | `hugpy_fleet.phone_brick.rpc_backend` |
| `abstract_hugpy_dev.phone_brick.schemas` | `hugpy_fleet.phone_brick.schemas` |
| `abstract_hugpy_dev.phone_brick.worker` | `hugpy_fleet.phone_brick.worker` |
| `abstract_hugpy_dev.provisioner` | `hugpy_ops.provisioner` |
| `abstract_hugpy_dev.review` | `hugpy_curation.review` |
| `abstract_hugpy_dev.review.__main__` | `hugpy_curation.review.__main__` |
| `abstract_hugpy_dev.review.criteria` | `hugpy_curation.review.criteria` |
| `abstract_hugpy_dev.review.fleet_grading` | `hugpy_curation.review.fleet_grading` |
| `abstract_hugpy_dev.review.judge` | `hugpy_curation.review.judge` |
| `abstract_hugpy_dev.review.pipeline` | `hugpy_curation.review.pipeline` |
| `abstract_hugpy_dev.review.push` | `hugpy_curation.review.push` |
| `abstract_hugpy_dev.review.screen` | `hugpy_curation.review.screen` |
| `abstract_hugpy_dev.review.smoke` | `hugpy_curation.review.smoke` |
| `abstract_hugpy_dev.review.store` | `hugpy_curation.review.store` |
| `abstract_hugpy_dev.sentinel` | `hugpy_ops.sentinel` |
| `abstract_hugpy_dev.sentinel.__main__` | `hugpy_ops.sentinel.__main__` |
| `abstract_hugpy_dev.sentinel.cases` | `hugpy_ops.sentinel.cases` |
| `abstract_hugpy_dev.sentinel.checks` | `hugpy_ops.sentinel.checks` |
| `abstract_hugpy_dev.sentinel.remedies` | `hugpy_ops.sentinel.remedies` |
| `abstract_hugpy_dev.sentinel.runner` | `hugpy_ops.sentinel.runner` |
| `abstract_hugpy_dev.sentinel.settings` | `hugpy_ops.sentinel.settings` |
| `abstract_hugpy_dev.utils.json_scavenge` | `hugpy_engine.utils.json_scavenge` |
| `abstract_hugpy_dev.utils.no_think` | `hugpy_engine.utils.no_think` |
| `abstract_hugpy_dev.utils.pdfs` | `hugpy_media.pdfs` |
| `abstract_hugpy_dev.utils.pdfs.utils` | `hugpy_media.pdfs.utils` |
| `abstract_hugpy_dev.utils.seo` | `hugpy_media.seo` |
| `abstract_hugpy_dev.utils.seo.pdf_utils` | `hugpy_media.seo.pdf_utils` |
| `abstract_hugpy_dev.utils.text` | `hugpy_media.text` |
| `abstract_hugpy_dev.utils.text.combined` | `hugpy_media.text.combined` |
| `abstract_hugpy_dev.video_intel` | `hugpy_video.intel` |
| `abstract_hugpy_dev.video_intel.audio_schema` | `hugpy_video.intel.audio_schema` |
| `abstract_hugpy_dev.video_intel.chains` | `hugpy_video.intel.chains` |
| `abstract_hugpy_dev.video_intel.crop_schema` | `hugpy_video.intel.crop_schema` |
| `abstract_hugpy_dev.video_intel.frame_schema` | `hugpy_video.intel.frame_schema` |
| `abstract_hugpy_dev.video_intel.gen_schema` | `hugpy_video.intel.gen_schema` |
| `abstract_hugpy_dev.video_intel.identity_from_video_schema` | `hugpy_video.intel.identity_from_video_schema` |
| `abstract_hugpy_dev.video_intel.identity_profiles` | `hugpy_video.intel.identity_profiles` |
| `abstract_hugpy_dev.video_intel.identity_reconstruction_schema` | `hugpy_video.intel.identity_reconstruction_schema` |
| `abstract_hugpy_dev.video_intel.identity_video_extract_schema` | `hugpy_video.intel.identity_video_extract_schema` |
| `abstract_hugpy_dev.video_intel.job_bridge` | `hugpy_video.intel.job_bridge` |
| `abstract_hugpy_dev.video_intel.job_lifecycle` | `hugpy_video.intel.job_lifecycle` |
| `abstract_hugpy_dev.video_intel.job_schema` | `hugpy_video.intel.job_schema` |
| `abstract_hugpy_dev.video_intel.media_bus` | `hugpy_video.intel.media_bus` |
| `abstract_hugpy_dev.video_intel.media_schema` | `hugpy_video.intel.media_schema` |
| `abstract_hugpy_dev.video_intel.media_store` | `hugpy_video.intel.media_store` |
| `abstract_hugpy_dev.video_intel.mlt_render_schema` | `hugpy_video.intel.mlt_render_schema` |
| `abstract_hugpy_dev.video_intel.movie_schema` | `hugpy_video.intel.movie_schema` |
| `abstract_hugpy_dev.video_intel.placement` | `hugpy_video.intel.placement` |
| `abstract_hugpy_dev.video_intel.presets` | `hugpy_video.intel.presets` |
| `abstract_hugpy_dev.video_intel.prompt_coordination` | `hugpy_oracle.relay.prompt_coordination` |
| `abstract_hugpy_dev.video_intel.prompt_intent` | `hugpy_video.intel.prompt_intent` |
| `abstract_hugpy_dev.video_intel.prompt_seeds` | `hugpy_video.intel.prompt_seeds` |
| `abstract_hugpy_dev.video_intel.prompt_spread` | `hugpy_video.intel.prompt_spread` |
| `abstract_hugpy_dev.video_intel.reservation` | `hugpy_video.intel.reservation` |
| `abstract_hugpy_dev.video_intel.reservation.engine` | `hugpy_video.intel.reservation.engine` |
| `abstract_hugpy_dev.video_intel.reservation.registry` | `hugpy_video.intel.reservation.registry` |
| `abstract_hugpy_dev.video_intel.reservation.templates` | `hugpy_video.intel.reservation.templates` |
| `abstract_hugpy_dev.video_intel.result_schema` | `hugpy_video.intel.result_schema` |
| `abstract_hugpy_dev.video_intel.runners` | `hugpy_video.intel.runners` |
| `abstract_hugpy_dev.video_intel.runners._gpu_guard` | `hugpy_video.intel.runners._gpu_guard` |
| `abstract_hugpy_dev.video_intel.runners._img2img` | `hugpy_video.intel.runners._img2img` |
| `abstract_hugpy_dev.video_intel.runners.ffmpeg_audio` | `hugpy_video.intel.runners.ffmpeg_audio` |
| `abstract_hugpy_dev.video_intel.runners.ffmpeg_crop` | `hugpy_video.intel.runners.ffmpeg_crop` |
| `abstract_hugpy_dev.video_intel.runners.ffmpeg_frames` | `hugpy_video.intel.runners.ffmpeg_frames` |
| `abstract_hugpy_dev.video_intel.runners.identity_from_video` | `hugpy_video.intel.runners.identity_from_video` |
| `abstract_hugpy_dev.video_intel.runners.identity_mesh` | `hugpy_video.intel.runners.identity_mesh` |
| `abstract_hugpy_dev.video_intel.runners.identity_reconstruction` | `hugpy_video.intel.runners.identity_reconstruction` |
| `abstract_hugpy_dev.video_intel.runners.identity_render_client` | `hugpy_video.intel.runners.identity_render_client` |
| `abstract_hugpy_dev.video_intel.runners.identity_render_relay` | `hugpy_video.intel.runners.identity_render_relay` |
| `abstract_hugpy_dev.video_intel.runners.identity_video_extract_relay` | `hugpy_video.intel.runners.identity_video_extract_relay` |
| `abstract_hugpy_dev.video_intel.runners.imagegen` | `hugpy_video.intel.runners.imagegen` |
| `abstract_hugpy_dev.video_intel.runners.mlt_render` | `hugpy_video.intel.runners.mlt_render` |
| `abstract_hugpy_dev.video_intel.runners.movie` | `hugpy_video.intel.runners.movie` |
| `abstract_hugpy_dev.video_intel.runners.performance_relay` | `hugpy_oracle.relay.performance_relay` |
| `abstract_hugpy_dev.video_intel.runners.scene` | `hugpy_video.intel.runners.scene` |
| `abstract_hugpy_dev.video_intel.runners.studio_i2v` | `hugpy_video.intel.runners.studio_i2v` |
| `abstract_hugpy_dev.video_intel.runners.studio_movie` | `hugpy_video.intel.runners.studio_movie` |
| `abstract_hugpy_dev.video_intel.runners.studio_tester` | `hugpy_video.intel.runners.studio_tester` |
| `abstract_hugpy_dev.video_intel.runners.tts_chatterbox` | `hugpy_media.tts.chatterbox_runner` |
| `abstract_hugpy_dev.video_intel.scene_schema` | `hugpy_video.intel.scene_schema` |
| `abstract_hugpy_dev.video_intel.shot_intent` | `hugpy_video.intel.shot_intent` |
| `abstract_hugpy_dev.video_intel.studio` | `hugpy_video.intel.studio` |
| `abstract_hugpy_dev.video_intel.studio.artifacts` | `hugpy_video.intel.studio.artifacts` |
| `abstract_hugpy_dev.video_intel.studio.editor_handoff` | `hugpy_video.intel.studio.editor_handoff` |
| `abstract_hugpy_dev.video_intel.studio.enums` | `hugpy_video.intel.studio.enums` |
| `abstract_hugpy_dev.video_intel.studio.env` | `hugpy_video.intel.studio.env` |
| `abstract_hugpy_dev.video_intel.studio.errors` | `hugpy_video.intel.studio.errors` |
| `abstract_hugpy_dev.video_intel.studio.job` | `hugpy_video.intel.studio.job` |
| `abstract_hugpy_dev.video_intel.studio.manifest` | `hugpy_video.intel.studio.manifest` |
| `abstract_hugpy_dev.video_intel.studio.models_seed` | `hugpy_video.intel.studio.models_seed` |
| `abstract_hugpy_dev.video_intel.studio.movie_plan` | `hugpy_video.intel.studio.movie_plan` |
| `abstract_hugpy_dev.video_intel.studio.presets` | `hugpy_video.intel.studio.presets` |
| `abstract_hugpy_dev.video_intel.studio.produce` | `hugpy_video.intel.studio.produce` |
| `abstract_hugpy_dev.video_intel.studio.registry` | `hugpy_video.intel.studio.registry` |
| `abstract_hugpy_dev.video_intel.studio.router` | `hugpy_video.intel.studio.router` |
| `abstract_hugpy_dev.video_intel.studio.runners` | `hugpy_video.intel.studio.runners` |
| `abstract_hugpy_dev.video_intel.studio.runners.ffmpeg_enhance` | `hugpy_video.intel.studio.runners.ffmpeg_enhance` |
| `abstract_hugpy_dev.video_intel.studio.runners.ltx_upscale` | `hugpy_video.intel.studio.runners.ltx_upscale` |
| `abstract_hugpy_dev.video_intel.studio.runners.rife_interpolate` | `hugpy_video.intel.studio.runners.rife_interpolate` |
| `abstract_hugpy_dev.video_intel.studio.runners.synthetic` | `hugpy_video.intel.studio.runners.synthetic` |
| `abstract_hugpy_dev.video_intel.studio.runners.wan_i2v` | `hugpy_video.intel.studio.runners.wan_i2v` |
| `abstract_hugpy_dev.video_intel.studio.runners.wan_t2v` | `hugpy_video.intel.studio.runners.wan_t2v` |
| `abstract_hugpy_dev.video_intel.studio.runners.wan_vace` | `hugpy_video.intel.studio.runners.wan_vace` |
| `abstract_hugpy_dev.video_intel.studio.schemas` | `hugpy_video.intel.studio.schemas` |
| `abstract_hugpy_dev.video_intel.studio.storage` | `hugpy_video.intel.studio.storage` |
| `abstract_hugpy_dev.video_intel.studio.tester` | `hugpy_video.intel.studio.tester` |
| `abstract_hugpy_dev.video_intel.studio_movie_schema` | `hugpy_video.intel.studio_movie_schema` |
| `abstract_hugpy_dev.video_intel.studio_presets` | `hugpy_video.intel.studio_presets` |
| `abstract_hugpy_dev.worker_agent` | `hugpy_fleet.worker` |
| `abstract_hugpy_dev.worker_agent.__main__` | `hugpy_fleet.worker.__main__` |
| `abstract_hugpy_dev.worker_agent._studio_subproc` | `hugpy_fleet.worker._studio_subproc` |
| `abstract_hugpy_dev.worker_agent.agent` | `hugpy_fleet.worker.agent` |
| `abstract_hugpy_dev.worker_agent.aggregate` | `hugpy_fleet.worker.aggregate` |
| `abstract_hugpy_dev.worker_agent.aptitude` | `hugpy_fleet.worker.aptitude` |
| `abstract_hugpy_dev.worker_agent.aptitude.cases` | `hugpy_fleet.worker.aptitude.cases` |
| `abstract_hugpy_dev.worker_agent.aptitude.parse` | `hugpy_fleet.worker.aptitude.parse` |
| `abstract_hugpy_dev.worker_agent.aptitude.score` | `hugpy_fleet.worker.aptitude.score` |
| `abstract_hugpy_dev.worker_agent.aptitude.selftest` | `hugpy_fleet.worker.aptitude.selftest` |
| `abstract_hugpy_dev.worker_agent.budget` | `hugpy_fleet.worker.budget` |
| `abstract_hugpy_dev.worker_agent.comfy_watchdog` | `hugpy_fleet.worker.comfy_watchdog` |
| `abstract_hugpy_dev.worker_agent.environment_report` | `hugpy_fleet.worker.environment_report` |
| `abstract_hugpy_dev.worker_agent.flex` | `hugpy_fleet.worker.flex` |
| `abstract_hugpy_dev.worker_agent.gen_gate` | `hugpy_fleet.worker.gen_gate` |
| `abstract_hugpy_dev.worker_agent.imports` | `hugpy_fleet.worker.imports` |
| `abstract_hugpy_dev.worker_agent.install` | `hugpy_fleet.worker.install` |
| `abstract_hugpy_dev.worker_agent.pid_registry` | `hugpy_fleet.worker.pid_registry` |
| `abstract_hugpy_dev.worker_agent.provision` | `hugpy_storage.provision` |
| `abstract_hugpy_dev.worker_agent.studio_render` | `hugpy_fleet.worker.studio_render` |
| `abstract_hugpy_dev.worker_agent.studio_reserve` | `hugpy_fleet.worker.studio_reserve` |

## Retired aggregators (lazy namespaces)

Attribute lookups replay the aggregator's old import statements against the
new packages, so `from abstract_hugpy_dev.comms import job_store` still works.

| Old module | Re-exported from |
|---|---|
| `abstract_hugpy_dev` | `abstract_hugpy_dev._compat_pydantic`, `importlib.metadata`, `abstract_hugpy_dev.imports`, `abstract_hugpy_dev.managers`, `abstract_hugpy_dev.utils` |
| `abstract_hugpy_dev.comms` | `abstract_hugpy_dev.comms.jobs`, `abstract_hugpy_dev.comms.bus`, `abstract_hugpy_dev.comms.principals`, `abstract_hugpy_dev.comms.settings`, `abstract_hugpy_dev.comms.blocklist`, `abstract_hugpy_dev.comms.calibration`, `abstract_hugpy_dev.comms.model_metrics` |
| `abstract_hugpy_dev.flask_app` | `abstract_hugpy_dev.flask_app.app`, `abstract_hugpy_dev.flask_app.wsgi_app` |
| `abstract_hugpy_dev.flask_app.app.functions.chat.imports` | `abstract_hugpy_dev.flask_app.app.functions.imports` |
| `abstract_hugpy_dev.flask_app.app.functions.downloads` | `abstract_hugpy_dev.flask_app.app.functions.downloads.downloads`, `abstract_hugpy_dev.flask_app.app.functions.downloads.downloader`, `abstract_hugpy_dev.flask_app.app.functions.downloads.model_physical`, `abstract_hugpy_dev.flask_app.app.functions.downloads.cancelable_downloads` |
| `abstract_hugpy_dev.flask_app.app.functions.downloads.imports` | `abstract_hugpy_dev.flask_app.app.functions.imports` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.init_imports` | `abstract_flask`, `abstract_hugpy_dev.imports`, `abstract_hugpy_dev.managers`, `abstract_hugpy_dev.utils` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.options.imports` | `abstract_hugpy_dev.flask_app.app.functions.imports.init_imports`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.imports` | `abstract_hugpy_dev.flask_app.app.functions.imports.init_imports` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas` | `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.download_schemas`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.install_schemas`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.job_schemas`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.model_schemas`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.specs_schemas`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.request_schemas`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.config_schemas`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.chat_schemas` |
| `abstract_hugpy_dev.flask_app.app.functions.imports.utils.schemas.imports` | `abstract_hugpy_dev.flask_app.app.functions.imports.utils.imports`, `abstract_hugpy_dev.flask_app.app.functions.imports.utils.constants` |
| `abstract_hugpy_dev.flask_app.app.routes.imports` | `abstract_hugpy_dev.flask_app.app.functions` |
| `abstract_hugpy_dev.imports` | `abstract_hugpy_dev.imports.src`, `abstract_hugpy_dev.imports.apis`, `abstract_hugpy_dev.imports.config` |
| `abstract_hugpy_dev.imports.apis` | `abstract_hugpy_dev.imports.apis.call_api`, `abstract_hugpy_dev.imports.apis.huggingface_api`, `abstract_hugpy_dev.imports.apis.get_module`, `abstract_hugpy_dev.imports.apis.download_models` |
| `abstract_hugpy_dev.imports.apis.imports` | `abstract_hugpy_dev.imports.src`, `abstract_hugpy_dev.imports.config` |
| `abstract_hugpy_dev.imports.config` | `abstract_hugpy_dev.imports.config.models`, `abstract_hugpy_dev.imports.config.main` |
| `abstract_hugpy_dev.imports.config.imports` | `abstract_hugpy_dev.imports.src` |
| `abstract_hugpy_dev.imports.config.models` | `abstract_hugpy_dev.imports.config.models.models_config`, `abstract_hugpy_dev.imports.config.models.models_default` |
| `abstract_hugpy_dev.imports.config.models.imports` | `abstract_hugpy_dev.imports.config.imports` |
| `abstract_hugpy_dev.imports.src` | `abstract_hugpy_dev.imports.src.constants`, `abstract_hugpy_dev.imports.src.init_imports`, `abstract_hugpy_dev.imports.src.module_imports`, `abstract_hugpy_dev.imports.src.chunking`, `abstract_hugpy_dev.imports.src.schemas`, `abstract_hugpy_dev.imports.src.utils`, `abstract_hugpy_dev.imports.src.except_utils`, `abstract_hugpy_dev.imports.src.peft_adapters` |
| `abstract_hugpy_dev.imports.src.constants` | `abstract_hugpy_dev.imports.src.constants.constants`, `abstract_hugpy_dev.imports.src.constants.paths`, `abstract_hugpy_dev.imports.src.constants.categories`, `abstract_hugpy_dev.imports.src.constants.hugpy_marker` |
| `abstract_hugpy_dev.imports.src.constants.imports` | `abstract_hugpy_dev.imports.src.init_imports` |
| `abstract_hugpy_dev.imports.src.init_imports` | `pydantic`, `dataclasses`, `typing`, `urllib.parse`, `collections`, `PyPDF2`, `uuid`, `pathlib` ... |
| `abstract_hugpy_dev.imports.src.schemas` | `abstract_hugpy_dev.imports.src.schemas.chat_schemas`, `abstract_hugpy_dev.imports.src.schemas.event_schemas`, `abstract_hugpy_dev.imports.src.schemas.model_schemas`, `abstract_hugpy_dev.imports.src.schemas.runner_schemas`, `abstract_hugpy_dev.imports.src.schemas.task_schemas`, `abstract_hugpy_dev.imports.src.schemas.video_schemas`, `abstract_hugpy_dev.imports.src.schemas.whisper_schemas`, `abstract_hugpy_dev.imports.src.schemas.summarizer_schemas` ... |
| `abstract_hugpy_dev.imports.src.schemas.imports` | `typing`, `pydantic`, `abstract_hugpy_dev.imports.src.constants`, `abstract_hugpy_dev.imports.src.utils`, `abstract_hugpy_dev.imports.src.init_imports` |
| `abstract_hugpy_dev.imports.src.standalone_utils` | `abstract_essentials` |
| `abstract_hugpy_dev.managers` | `abstract_hugpy_dev.managers.embed`, `abstract_hugpy_dev.managers.imagegen`, `abstract_hugpy_dev.managers.keywords`, `abstract_hugpy_dev.managers.whisper_model`, `abstract_hugpy_dev.managers.generate`, `abstract_hugpy_dev.managers.vision`, `abstract_hugpy_dev.managers.video`, `abstract_hugpy_dev.managers.summarizers` ... |
| `abstract_hugpy_dev.managers.chat_context.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.dispatch.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.embed.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.falconsai` | `abstract_hugpy_dev.managers.falconsai.falconsai_module` |
| `abstract_hugpy_dev.managers.falconsai.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.generate.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.imagegen.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.imports` | `abstract_hugpy_dev.imports` |
| `abstract_hugpy_dev.managers.keywords.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.llama.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.llama.runners.imports` | `abstract_hugpy_dev.managers.llama.imports` |
| `abstract_hugpy_dev.managers.resolvers.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.serve.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.summarizers.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.video.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.vision.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.vision_analysis.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.whisper_model.imports` | `abstract_hugpy_dev.managers.imports` |
| `abstract_hugpy_dev.managers.whisper_model.src.imports` | `abstract_hugpy_dev.managers.whisper_model.imports`, `abstract_hugpy_dev.managers.whisper_model.constants` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.imports` | `abstract_hugpy_dev.managers.whisper_model.src.imports` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.artifacts.imports` | `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.imports` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.frames.imports` | `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.imports` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.files.imports` | `abstract_hugpy_dev.managers.whisper_model.src.model.utils.imports` |
| `abstract_hugpy_dev.managers.whisper_model.src.model.utils.imports` | `abstract_hugpy_dev.managers.whisper_model.src.model.imports` |
| `abstract_hugpy_dev.utils` | `abstract_hugpy_dev.utils.seo`, `abstract_hugpy_dev.utils.text` |
| `abstract_hugpy_dev.utils.imports` | `abstract_hugpy_dev.imports`, `abstract_hugpy_dev.managers` |
| `abstract_hugpy_dev.utils.seo.imports` | `abstract_hugpy_dev.utils.imports` |
| `abstract_hugpy_dev.utils.text.imports` | `abstract_hugpy_dev.utils.imports`, `abstract_hugpy_dev.utils.pdfs` |

## Retired modules (no alias)

| Old module | Note |
|---|---|
| `abstract_hugpy_dev.get_vids` | retired by partition.toml (dead or superseded code); copy at py/unwired/monolith/get_vids.py |
| `abstract_hugpy_dev.imports.src._compat` | target `hugpy_platform.compat` no longer exists in its package |
| `abstract_hugpy_dev.managers.falconsai.falconsai_module` | retired by partition.toml (dead or superseded code); copy at py/unwired/monolith/managers/falconsai/falconsai_module.py |
| `abstract_hugpy_dev.managers.generate.coder_guff` | retired by partition.toml (dead or superseded code); copy at py/unwired/monolith/managers/generate/coder_guff.py |
| `abstract_hugpy_dev.managers.generate.generate_runner2` | retired by partition.toml (dead or superseded code); copy at py/unwired/monolith/managers/generate/generate_runner2.py |
| `abstract_hugpy_dev.managers.get_pids` | target `hugpy_fleet.get_pids` no longer exists in its package; copy at py/unwired/fleet/get_pids.py |
| `abstract_hugpy_dev.managers.resolvers.categories.imports` | target `hugpy_engine.resolvers.categories.imports` no longer exists in its package; copy at py/unwired/monolith/managers/imports.py |
| `abstract_hugpy_dev.video_intel._selftest_movie_presets` | target `hugpy_video.intel._selftest_movie_presets` no longer exists in its package |
| `abstract_hugpy_dev.video_intel._selftest_scene` | target `hugpy_video.intel._selftest_scene` no longer exists in its package; copy at py/unwired/video/_selftest_scene.py |
| `abstract_hugpy_dev.worker_agent.get_size` | target `hugpy_fleet.worker.get_size` no longer exists in its package; copy at py/unwired/fleet/get_size.py |
