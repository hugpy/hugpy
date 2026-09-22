# Hugpy public ecosystem map

Inventory date: 2026-09-22. Scope: the four public registries (GitHub, PyPI,
npm, Hugging Face) against the partitioned build described in
[`PARTITION.md`](PARTITION.md) and [`py/partition.toml`](py/partition.toml).
Everything in section D is a plan: **no command in this document has been
executed**. Only the local metadata edits listed in D.0 were applied.

---

## A. Current inventory

### A.1 GitHub

| Owner / repo | Status | Default | Last push | License | Topics | Notes |
|---|---|---|---|---|---|---|
| `hugpy` (org) | live, **0 public repos** | – | – | – | – | Created 2026-06-04; bio "A self-hosted LLM console…"; blog `hugpy.ai`; email `support@hugpy.ai`. No `.github` profile repo. |
| `hugpy/hugpy` | **404** | – | – | – | – | Already referenced by PyPI `hugpy` 0.1.181 (`Source = …/hugpy/hugpy/tree/main/api`). |
| `hugpy/hugpy-agent` | **404** | – | – | – | – | Already referenced by PyPI `hugpy-agent` 0.1.70 (`Repository`, `Issues`). |
| `AbstractEndeavors/hugpy` | live | main | 2026-07-01 | NOASSERTION (custom) | abstract, ai, gpu, huggingface, hugpy, llm, offloading-framework, python | Snapshot of the old `hugpy` monolith (`hugpy/` package, `PKG-INFO`); README badges point at PyPI `hugpy`. Stale vs. current code. |
| `AbstractEndeavors/abstract_hugpy_dev` | live | main | 2026-09-18 | NOASSERTION (custom) | none | The current monolith source (`src/`, `tests/`, `deploy/`). README is the `hugpy` README with `abstract_hugpy_dev` badges. |
| `AbstractEndeavors/abstract_hugpy` | live | main | 2026-06-27 | none | none | The 2025 predecessor (`abstract-hugpy` on PyPI). Superseded. |
| `AbstractEndeavors/abstract_essentials` | live | main | 2026-06-27 | none | none | Runtime dependency of `hugpy-platform` and others. |
| `AbstractEndeavors/media_intelligence` | live | main | 2026-06-27 | none | none | Nearly empty (size 28 KB); PyPI `media-intelligence` 0.1.2 is stale. |
| `AbstractEndeavors/abstract_toolserver` | **404** | – | – | – | – | Referenced by PyPI `abstract-toolserver` `url`. |
| `AbstractEndeavors/abstract_apply` | **404** | – | – | – | – | Referenced by local `abstract_apply` `Homepage`. |
| `AbstractEndeavors/abstract-claude`, `abstract_gpt`, `abstract_identity`, `abstract_search` | **404** | – | – | – | – | `abstract_claude/map.toml` names `git@github.com:AbstractEndeavors/abstract-claude.git`. |
| `putkoff/*` | 41 public repos | – | 2019-2024 | – | – | None hugpy-related. |

Every GitHub URL currently published in PyPI/npm/local metadata for the Hugpy
family resolves to a 404. Nothing hugpy-branded is public on GitHub except two
stale monolith snapshots under `AbstractEndeavors`.

### A.2 PyPI

| Distribution | Latest | Uploaded | Releases | Author | License field | Repo URL | Status |
|---|---|---|---|---|---|---|---|
| `abstract_hugpy_dev` | 0.1.266 | 2026-09-22 | 256 | putkoff | full custom text; classifier `Other/Proprietary` | none (Homepage only) | **live monolith** (daily uploads) |
| `hugpy` | 0.1.181 | 2026-09-11 | 135 | putkoff | full custom text; `Other/Proprietary` | `github.com/hugpy/hugpy/tree/main/api` (404) | **live, same monolith under the product name**; last release 11 days ago; 67 requires-dist |
| `abstract-hugpy` | 0.1.401 | 2026-08-05 | 601 | putkoff (partners@abstractendeavors.com) | MIT classifier | `AbstractEndeavors/abstract_hugpy` | **stale predecessor**; summary already says "current project: hugpy" |
| `hugpy-agent` | 0.1.70 | 2026-09-20 | 33 | putkoff <support@hugpy.ai> | custom text | `hugpy/hugpy-agent` (404) | live, matches local 0.1.70 |
| `abstract-identity` | 0.1.0 | 2026-08-27 | 1 | putkoff | MIT | none | live, matches local 0.1.0; no `project.urls`, no classifiers |
| `abstract-toolserver` | 0.0.27 | 2026-09-02 | 27 | putkoff (abstractendeavors) | MIT classifier | `AbstractEndeavors/abstract_toolserver` (404) | live, matches local 0.0.27 |
| `abstract-claude` | 0.1.51 | 2026-09-20 | 35 | jrputkey | custom text; `Other/Proprietary` | none | live, matches `map.toml` 0.1.51 |
| `abstract-gpt` | 0.1.2 | 2026-09-15 | 3 | jrputkey | none | none | live; local `map.toml` says 0.1.3 unreleased, "NO wheel; only stale 0.1.2 sdist" |
| `abstract-apply` | – | – | – | – | – | **name free** (local 0.2.0 never published) |
| `hugpy-platform`, `-control`, `-storage`, `-engine`, `-media`, `-video`, `-oracle`, `-fleet`, `-curation`, `-ops`, `-discord`, `-server` | – | – | – | – | – | **all 12 names free** |
| `hugpy-station`, `hugpy-ui`, `hugpy-client` | – | – | – | – | – | name free (not needed) |
| `abstract-essentials` | 0.0.0.19 | 2026-08-23 | 18 | putkoff | MIT | none | dependency, live |
| `abstract-utilities` | 0.2.2.788 | 2026-09-22 | 908 | putkoff | – | `AbstractEndeavors/abstract_utilities` | dependency of toolserver, live |
| `abstract-search` | 0.0.0.42 | 2026-09-14 | 6 | putkoff | MIT | none | dependency of `hugpy-agent[mct]`, live |
| `abstract-apis` / `abstract-flask` / `abstract-queries` | 0.0.1.153 / 0.0.0.1049 / 0.0.0.116 | Jun-Jul 2026 | – | putkoff | – | mixed, some wrong repo (`abstract_ide`, `abstract_logins`) | dependencies, stale-ish |
| `media-intelligence` | 0.1.2 | 2026-06-25 | 3 | AbstractEndeavors | – | none | stale; functionality now `hugpy-media` |

No name in the Hugpy family is squatted by another party. `putkoff` and
`jrputkey` are both the same maintainer but appear as two PyPI identities.

### A.3 npm scope `@hugpy`

The search index (`text=scope:hugpy`) returns 0 results and `/org/hugpy`
returns 403 to unauthenticated fetches; the registry documents are public.

| Package | Latest | Published | Versions | repository / homepage / bugs | Local | Status |
|---|---|---|---|---|---|---|
| `@hugpy/ui` | 0.3.0 | 2026-08-27 | 0.2.1, 0.2.2, 0.3.0 | none / none / none | 0.3.0 | live, in sync |
| `@hugpy/agents-ui` | 0.3.0 | 2026-08-27 | 0.1.0, 0.3.0 | none | 0.3.0 | live; README on npm is 28 bytes (no README locally) |
| `@hugpy/media-intelligence-ui` | 0.3.0 | 2026-08-27 | 0.1.0, 0.3.0 | none | 0.3.0 | live; README on npm is 28 bytes (no README locally) |
| `@hugpy/video-intelligence-ui` | 0.3.2 | 2026-08-27 | 0.1.0 … 0.3.2 | none | 0.3.2 | live, in sync |
| `@hugpy/ui-shared` | 0.3.0 | 2026-08-27 | 0.1.0, 0.1.1, 0.3.0 | none | 0.3.0 | live, in sync |
| `@hugpy/station` | 1.0.44 | 2026-08-27 | 1.0.36-1.0.44 | none / `hugpy.ai` / none | .deb 1.0.109 (`map.toml`) | **drift**: npm carries the Electron *source* at 1.0.44 while the deb channel is at 1.0.109 |
| `@hugpy/console` | 0.3.1 | 2026-07-22 | 0.1.0 … 0.3.1 | none | not in this workspace | live; no local source of truth here |
| `@hugpy/vm-mgr` | 0.2.0 | 2026-07-22 | 0.2.0 | none | not in this workspace | live; no local source of truth here |

All eight packages are published by `putkoff` with `"license": "SEE LICENSE IN LICENSE"`.
None declares `repository`, `bugs` or (except station) `homepage`.

### A.4 Hugging Face org `hugpy-ai`

Org card: "Self-hosted, GPU-pooling LLM inference — run your own model fleet
on your own hardware. One streaming, OpenAI-compatible API over transformers
& llama.cpp. Inference you own." 1 member, 1 follower, 0 models, 0 datasets,
6 spaces (5 visible through the public API).

| Space | SDK | Title / short description | Last modified | Runtime |
|---|---|---|---|---|
| `hugpy-ai/hugpy` | docker | "hugpy — Start Here": the hugpy.ai platform | 2026-08-13 | sleeping |
| `hugpy-ai/hugpy-console` | docker | "hugpy Console": explore the operator console | 2026-08-13 | sleeping |
| `hugpy-ai/hugpy-chat` | docker | "Media Intelligence": your models, your data | 2026-08-13 | sleeping |
| `hugpy-ai/hugpy-agent` | static | "hugpy-agent": the fleet's agent CLI | 2026-08-13 | running |
| `hugpy-ai/console-embed` | static | "@hugpy/console": embeddable web console | 2026-08-13 | running |

`grep -rn "hugpy-ai" py/` finds no code reference to the org; no package
downloads anything from `hugpy-ai/*`. The engine's curated defaults
(`hugpy_engine/config/models/models_default.py`) and curation's dossiers refer
to third-party Hub repos only.

### A.5 Mismatches to resolve

| # | Mismatch | Where |
|---|---|---|
| 1 | The product is published twice on PyPI (`hugpy` 0.1.181 and `abstract_hugpy_dev` 0.1.266) with the same summary, keywords and README; the newer code goes to the *non*-product name. | PyPI |
| 2 | `hugpy` on PyPI is the monolith; locally `hugpy` is a thin meta package at 0.1.0. Publishing 0.1.0 would sort *below* 0.1.181. | PyPI vs `py/meta/hugpy` |
| 3 | Every GitHub URL in published metadata 404s (`hugpy/hugpy`, `hugpy/hugpy-agent`, `AbstractEndeavors/abstract_toolserver`). | PyPI, local |
| 4 | Only `AbstractEndeavors` has public hugpy source, and it is a July snapshot plus an un-described `abstract_hugpy_dev` mirror with no topics. | GitHub |
| 5 | The 13 new distributions declare `license = "LicenseRef-Proprietary"` but ship **no LICENSE file** (`License-File` absent from the built wheel). PyPI/npm consumers cannot see terms. | 13 `pyproject.toml` |
| 6 | `hugpy-agent` and `abstract_hugpy_dev` use `license = { file = "LICENSE" }`; `abstract-identity`/`abstract-apply` are MIT; `abstract-claude` is proprietary text; two styles inside one family. | mixed |
| 7 | Seven of the 13 package READMEs are the 7-line extraction stub ("extracted from abstract_hugpy_dev…"); that text would become the PyPI long description. | platform, control, storage, media, fleet, discord, server |
| 8 | `@hugpy/agents-ui` and `@hugpy/media-intelligence-ui` have no README locally or on npm. | react |
| 9 | `@hugpy/station` on npm (1.0.44) lags the deb channel (1.0.109); the npm artefact is the Electron source, not an installable. | station |
| 10 | `@hugpy/console` and `@hugpy/vm-mgr` are published but have no source in this workspace; the HF space `console-embed` advertises `@hugpy/console`. | npm, HF |
| 11 | PyPI author identity is split: `putkoff` (most), `jrputkey` (`abstract-claude`, `abstract-gpt`), `AbstractEndeavors` (`media-intelligence`); emails split between `support@hugpy.ai` and `partners@abstractendeavors.com`. | PyPI |
| 12 | `abstract-gpt` local is 0.1.3 with no built wheel; PyPI has only a 0.1.2 sdist. | `py/inference/abstract_gpt` |
| 13 | HF org has spaces only; the org card says "GPU-pooling LLM inference" but the spaces are Docker demos last touched 2026-08-13 and sleeping. | HF |
| 14 | `media-intelligence` on PyPI and `AbstractEndeavors/media_intelligence` are stale ancestors of `hugpy-media` / `@hugpy/media-intelligence-ui`. | PyPI, GitHub |

---

## B. Target map

### B.1 Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Repo layout | **One monorepo `hugpy/hugpy`** for the 13 distributions, the five React packages and the workspace docs (`PARTITION.md`, `MOVE_MAP.md`, `WIRING.md`); **per-package repos** only for things that already release on their own cadence and have no in-tree import edges: `hugpy/hugpy-agent`, `hugpy/hugpy-station`, `hugpy/abstract-identity`. | The 13 packages form one acyclic DAG (13 nodes, ~45 edges) validated by `py/validate_partition.py`; seams (`hugpy_engine.placement`, `hugpy_engine.tasks`, `hugpy_video.hooks`, `WIRING.md`) change on both sides at once and need atomic commits and one CI. Independent *releases* are still possible from a monorepo (per-directory tags `hugpy-engine/v0.2.1`). `hugpy-agent` is dependency-free and consumed by third parties (`abstract-apply`), so it keeps its own repo, which PyPI already points at. |
| Non-Hugpy independents | `abstract-toolserver`, `abstract-apply`, `abstract-claude`, `abstract-gpt` stay under **`AbstractEndeavors`** (their published names, emails and licenses are Abstract Endeavors'). Their 404 repo URLs are fixed there, not moved. | They are consumers/tools of the `abstract_*` ecosystem that happen to use hugpy; moving them would break the `abstract-*` naming story. |
| Naming | PyPI `hugpy-<component>`; import `hugpy_<component>`; npm `@hugpy/<kebab>`; GitHub `hugpy/hugpy` (monorepo) or `hugpy/hugpy-<component>`; HF `hugpy-ai/<kebab>`. Local directory = import name (`py/<layer>/hugpy_<component>`, `react/<snake>`). | Matches what already exists on PyPI (`hugpy-agent`) and npm. |
| Version policy | Semantic versioning. **One initial `0.2.0` line for all 13** (bump locally from 0.1.0 at release time), then independent per-package bumps. `hugpy` pins siblings as `>=0.2,<0.3` in its extras. React packages continue their own lines (0.3.x); station keeps 1.0.x. | `hugpy` on PyPI is already at 0.1.181, so 0.2.0 is the first version that sorts above the monolith; using the same number across the family makes the first partitioned release legible ("the 0.2 line"). |
| License | Keep the **hugpy Source-Available License** (already the text behind `hugpy-agent`, `hugpy` 0.1.x, `abstract_hugpy_dev`, and the five npm packages). Express it as PEP 639 `license = "LicenseRef-Proprietary"` **plus** `license-files = ["LICENSE"]` with the text copied into every package. On npm keep `"license": "SEE LICENSE IN LICENSE"` with the same file. `abstract-identity` stays MIT (it already shipped MIT and bundles AGPL/non-commercial third-party notices). | PEP 639 requires the `LicenseRef-` text to be shipped as a license file; today the wheels ship none. One text, one identifier, every registry. Switching to an OSI license is a business decision; if taken later, change `license =` to the SPDX id and drop the file in one commit. |
| Retiring `abstract_hugpy_dev` | Final release `0.1.267` whose `dependencies = ["hugpy[all]>=0.2,<0.3"]` and whose `abstract_hugpy_dev/__init__.py` emits a `DeprecationWarning` and re-exports via `_relocations.py`. 30 days later **yank** every `abstract_hugpy_dev` release except 0.1.267 (yank, do not delete: existing pins keep resolving). `abstract-hugpy` (2025 line) gets the same treatment with a single 0.1.402 bridge. | Nobody's `pip install abstract_hugpy_dev` breaks; they get the meta package and a warning pointing to `hugpy`. |
| Archived repos with reused names | The org has no public repos, so there is nothing to un-archive. If a private archived `hugpy/hugpy`, `hugpy/hugpy-agent` or `hugpy/hugpy-station` exists: **delete it** (the user has authorised nuking archived names) after `gh repo clone` into `py/unwired/archive/<name>-<date>` for the record, then create fresh. `AbstractEndeavors/hugpy` and `AbstractEndeavors/abstract_hugpy` are **archived** (not deleted) with a README pointer to `hugpy/hugpy`; `AbstractEndeavors/abstract_hugpy_dev` is archived once the monolith is deleted from this workspace. | Deleting archived clones avoids two `hugpy` repos with divergent history; archiving the AE repos keeps PyPI "Source" links for old releases resolvable. |

### B.2 Component map

Local versions are what is checked in today; "first public" is the version to
publish after review. Descriptions are the ones now set in each manifest.

| Component | GitHub | Registry name | Import / mount | Local | First public | License | Description | Source of truth |
|---|---|---|---|---|---|---|---|---|
| hugpy (meta) | `hugpy/hugpy` `py/meta/hugpy` | PyPI `hugpy` | `hugpy`; CLI `hugpy`, `hpy` | 0.1.0 | 0.2.0 | Source-Available | Self-hosted LLM console and OpenAI-compatible API; meta distribution with the `hugpy`/`hpy` commands and install profiles for the `hugpy-*` packages. | `py/meta/hugpy` |
| platform | `py/foundation/hugpy_platform` | `hugpy-platform` | `hugpy_platform` | 0.1.0 | 0.2.0 | Source-Available | Stdlib-first foundation shared by every Hugpy package: env-backed configuration, app dirs, hardware and process probes, central URL, pydantic shim. | `py/foundation/hugpy_platform` |
| control | `py/foundation/hugpy_control` | `hugpy-control` | `hugpy_control` | 0.1.0 | 0.2.0 | Source-Available | Control plane: typed event bus, generic jobs, settings, principals, call log and shared control records. | `py/foundation/hugpy_control` |
| storage | `py/storage/hugpy_storage` | `hugpy-storage` | `hugpy_storage`; CLI `hugpy-storage`, `hugpy-downloader` | 0.1.0 | 0.2.0 | Source-Available | Model download queue and daemon, Hugging Face transport and token, resumable transfers, model sync, physical inventory and on-disk layout. | `py/storage/hugpy_storage` |
| engine | `py/inference/hugpy_engine` | `hugpy-engine` | `hugpy_engine`; CLI `hugpy-serve` | 0.1.0 | 0.2.0 | Source-Available | Model discovery and classification, allocation and eviction, native llama.cpp and GGUF runners, and the prompt-to-reply path with task and placement seams. | `py/inference/hugpy_engine` |
| media | `py/inference/hugpy_media` | `hugpy-media` | `hugpy_media`; entry point `hugpy_engine.tasks:media` | 0.1.0 | 0.2.0 | Source-Available | Engine task plugins for speech recognition, TTS, embeddings, keywords, summaries, vision, image generation, ComfyUI and document/URL extraction. | `py/inference/hugpy_media` |
| video | `py/cinema/hugpy_video` | `hugpy-video` | `hugpy_video`; CLI `hugpy-video`; entry point `hugpy_engine.tasks:video` | 0.1.0 | 0.2.0 | Source-Available | Media library and job bus, ffmpeg, synthetic and studio runners, reservations and studio sessions for video intelligence. | `py/cinema/hugpy_video` |
| oracle | `py/cinema/hugpy_oracle` | `hugpy-oracle` | `hugpy_oracle`; CLI `hugpy-oracle` | 0.1.0 | 0.2.0 | Source-Available | Creative planning, model selection, screenplay and storyboard generation, DAG runtime, repair, evaluation and ledgers over the video pipeline. | `py/cinema/hugpy_oracle` |
| fleet | `py/fleet/hugpy_fleet` | `hugpy-fleet` | `hugpy_fleet`; CLI `hugpy-worker`, `hugpy-gguf-worker`, `hugpy-phone-brick` | 0.1.0 | 0.2.0 | Source-Available | Central worker registry, enrollment, heartbeats, evictions, doctrine and placement, plus the full, GGUF and phone worker agents. | `py/fleet/hugpy_fleet` |
| curation | `py/curation/hugpy_curation` | `hugpy-curation` | `hugpy_curation`; CLI `hugpy-curation` | 0.1.0 | 0.2.0 | Source-Available | Discovery dossiers and the search, screen, download, smoke and judge review pipeline for candidate models. | `py/curation/hugpy_curation` |
| ops | `py/operations/hugpy_ops` | `hugpy-ops` | `hugpy_ops` | 0.1.0 | 0.2.0 | Source-Available | Sentinel, chaos runner, keeper REPL, todo keeper and provisioner for declared-but-missing model weights. | `py/operations/hugpy_ops` |
| discord | `py/integrations/hugpy_discord` | `hugpy-discord` | `hugpy_discord`; CLI `hugpy-bot` | 0.1.0 | 0.2.0 | Source-Available | The Discord bot arm: an HTTP client of a Hugpy central with streaming replies and per-user preferences. | `py/integrations/hugpy_discord` |
| server | `py/services/hugpy_server` | `hugpy-server` | `hugpy_server` (`wsgi_app`) | 0.1.0 | 0.2.0 | Source-Available | Flask composition root: routes, auth, SSE, API keys and the built web console; wires fleet placement, task plugins and oracle hooks. | `py/services/hugpy_server` |
| ui | `hugpy/hugpy` `react/ui` | npm `@hugpy/ui` | mount `/` | 0.3.0 | 0.3.1 | Source-Available | Embeddable React UI for Hugpy: configurable panels and a drop-in `<HugpyConsole/>`; served at `/` by hugpy-server. | `react/ui` |
| agents-ui | `react/agents_ui` | `@hugpy/agents-ui` | mount `/fleet` | 0.3.0 | 0.3.1 | Source-Available | Agent fleet console for hugpy-agent nodes: portable runtime, node roster and dispatch. | `react/agents_ui` |
| media-intelligence-ui | `react/media_intelligence_ui` | `@hugpy/media-intelligence-ui` | mount `/media` | 0.3.0 | 0.3.1 | Source-Available | Media-intelligence console over hugpy-media: transcription, summaries, embeddings, vision, document analysis. | `react/media_intelligence_ui` |
| video-intelligence-ui | `react/video_intelligence_ui` | `@hugpy/video-intelligence-ui` | mount `/video` | 0.3.2 | 0.3.3 | Source-Available | Video-intelligence console over the hugpy-video job bus: image crop, audio crop, frames/models and generate stations. | `react/video_intelligence_ui` |
| ui-shared | `react/ui_shared` | `@hugpy/ui-shared` | library | 0.3.0 | 0.3.1 | Source-Available | Shared UI pieces for every Hugpy surface: Help widget and cross-app navbar links; zero dependencies. | `react/ui_shared` |
| station | `hugpy/hugpy-station` | npm `@hugpy/station` + `.deb`/`.rpm`/AppImage on `hugpy.ai/station` | Electron app | deb 1.0.109 / npm 1.0.44 | 1.0.110 (both) | Source-Available | Hugpy Station: the desktop cockpit, an Electron shell over the self-contained console backend (stations, VMs, terminals). | `/srv/hugpy/src/station-app` via `py/tools/hugpy-station/source` (symlink; **not resolvable from this workspace**) |
| agent | `hugpy/hugpy-agent` | PyPI `hugpy-agent` | `hugpy_agent`; CLI `hugpy-agent` | 0.1.70 | 0.1.71 (metadata-only) | Source-Available | Portable, dependency-free agent runtime and HTTP client for the Hugpy fleet. | `py/inference/hugpy_agent` |
| identity | `hugpy/abstract-identity` | PyPI `abstract-identity` | `abstract_identity`; CLI `char360`, `identity-render` | 0.1.0 | 0.1.1 (metadata-only) | MIT (+ third-party notices) | Identity pipeline for Hugpy video: char360 face detection/clustering and the identity-render GPU service. | `py/cinema/abstract_identity` |
| toolserver | `AbstractEndeavors/abstract_toolserver` | PyPI `abstract-toolserver` | `abstract_toolserver` | 0.0.27 | 0.0.28 (metadata-only) | MIT | The abstract_* ecosystem exposed as self-describing Flask tool endpoints. | `py/tools/abstract_toolserver` |
| apply | `AbstractEndeavors/abstract_apply` | PyPI `abstract-apply` (free) | `abstract_apply`; CLI `abstract-apply` | 0.2.0 | 0.2.0 | MIT | Tailor a cover letter and résumé to a job posting using the Hugpy fleet. | `py/tools/abstract_apply` |
| claude | `AbstractEndeavors/abstract-claude` | PyPI `abstract-claude` | `abstract_claude` | 0.1.51 | 0.1.52 | Source-Available | Fresh Claude Code session manager: settings template, durable OAuth, quota fallback. | `py/inference/abstract_claude/source` (station convention) |
| gpt | `AbstractEndeavors/abstract_gpt` | PyPI `abstract-gpt` | `abstract_gpt` | 0.1.3 (unbuilt) | 0.1.3 | Source-Available | Codex login and session management for the abstract toolserver. | `py/inference/abstract_gpt/source` |
| console / vm-mgr | `hugpy/hugpy-station` (proposed home) | npm `@hugpy/console` 0.3.1, `@hugpy/vm-mgr` 0.2.0 | – | not in workspace | unchanged | Source-Available | Embeddable web console + LXD VM fleet manager. | **unknown**: locate source before next publish |
| Hugging Face | – | org `hugpy-ai` | – | 5 spaces, 0 models, 0 datasets | – | – | See B.3 | `py/curation`, `py/inference/hugpy_engine/src/hugpy_engine/config/models` |

Retired, never to be published as packages: `abstract_hugpy_dev` (after the
bridge release), `abstract-hugpy`, `media-intelligence`, and everything under
`[[retire]]` in `py/partition.toml`.

### B.3 Hugging Face org content

What belongs under `hugpy-ai`, given that no package currently reads from it:

| Asset | Type | Content | Produced by |
|---|---|---|---|
| `hugpy-ai/hugpy` | space (keep) | "Start here" landing; card links to PyPI `hugpy`, `hugpy/hugpy`, hugpy.ai. Refresh the card to the partitioned install (`pip install "hugpy[server]"`). | manual |
| `hugpy-ai/hugpy-console`, `hugpy-ai/hugpy-chat` | spaces (keep, rebuild) | Docker demos of `@hugpy/ui` and `@hugpy/media-intelligence-ui` on top of `hugpy-server`. Rebuild from the 0.2 wheels; rename `hugpy-chat` to `hugpy-media` to match the package. | CI (optional) |
| `hugpy-ai/hugpy-agent`, `hugpy-ai/console-embed` | static spaces (keep) | Docs pages; update links to `hugpy/hugpy-agent` and `@hugpy/console`. | manual |
| `hugpy-ai/model-catalog` | **dataset (new)** | JSON export of the engine's curated defaults (`hugpy_engine.config.models.models_default`, categories, task capability map from `task_deps`) and curation's published dossier verdicts. Versioned with the `hugpy-engine` release it was exported from. | `hugpy-curation` export command (to add) |
| `hugpy-ai/<model>-GGUF` | models (only if produced) | Re-quantisations or fine-tunes that `hugpy-curation` review promotes. Standard model card with `base_model`, quant table and the hugpy marker JSON (`hugpy_storage.hugpy_marker`). | `hugpy-curation` promote step |

No third-party weights are mirrored; storage keeps downloading from the
upstream authors.

---

## C. Org standards

### C.1 Required files in every repo (monorepo root and each per-package repo)

| File | Requirement |
|---|---|
| `README.md` | Title, one-paragraph purpose, **Install** (`pip install …` / `npm i …`), **Usage** (one CLI and one Python example), **Architecture** link to `PARTITION.md`, license line. Per-package READMEs in the monorepo must replace the 7-line extraction stub (A.5 #7) before publishing; the PyPI long description is this file. |
| `LICENSE` | The hugpy Source-Available License text (identical copy in every distribution directory and every React package directory; `abstract-identity` keeps MIT). |
| `CODEOWNERS` | See C.3. |
| `.github/workflows/ci.yml` | See C.4. |
| `SECURITY.md` | "Report to support@hugpy.ai; do not open public issues for vulnerabilities." |
| `CHANGELOG.md` | Keep-a-Changelog format; monorepo has one per package directory. |
| `CONTRIBUTING.md` | Source-available: PRs accepted under a CLA-free "you license contributions under the repository LICENSE" clause; link to the release checklist. |

### C.2 Manifest metadata (now applied locally to the 13 + 5)

`pyproject.toml`: `description`, `authors = [{ name = "putkoff", email = "support@hugpy.ai" }]`,
`keywords` (first three always `hugpy`, `llm`, `self-hosted`), `classifiers`
(Alpha, Developers, OS Independent, Python 3 only, AI topic, plus one
environment classifier where relevant), and `[project.urls]` with `Homepage`,
`Documentation`, `Repository`, `Source`, `Issues`, `Changelog`, `Architecture`.
Still to add at release time: `license-files = ["LICENSE"]`.

`package.json`: `description`, `keywords`, `homepage`, `repository`
(`git+https://github.com/hugpy/hugpy.git` with `directory`), `bugs`,
`author` object, `license: "SEE LICENSE IN LICENSE"`, `LICENSE` in `files`.

### C.3 CODEOWNERS (monorepo)

```
# Default
*                                   @putkoff
# Package owners (edit when a second maintainer joins)
/py/foundation/                     @putkoff
/py/storage/                        @putkoff
/py/inference/hugpy_engine/         @putkoff
/py/inference/hugpy_media/          @putkoff
/py/cinema/                         @putkoff
/py/fleet/                          @putkoff
/py/curation/                       @putkoff
/py/operations/                     @putkoff
/py/integrations/                   @putkoff
/py/services/                       @putkoff
/py/meta/                           @putkoff
/react/                             @putkoff
/PARTITION.md /py/partition.toml /py/WIRING.md   @putkoff
```

### C.4 CI (`.github/workflows/ci.yml`, monorepo)

One matrix job per package directory; the commands are exactly the ones used in
this workspace.

```yaml
name: ci
on: [push, pull_request]
jobs:
  partition:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: python py/validate_partition.py && python py/validate_partition.py --edges
  python:
    needs: partition
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        pkg:
          - py/foundation/hugpy_platform
          - py/foundation/hugpy_control
          - py/storage/hugpy_storage
          - py/inference/hugpy_engine
          - py/inference/hugpy_media
          - py/cinema/hugpy_video
          - py/cinema/hugpy_oracle
          - py/fleet/hugpy_fleet
          - py/curation/hugpy_curation
          - py/operations/hugpy_ops
          - py/integrations/hugpy_discord
          - py/services/hugpy_server
          - py/meta/hugpy
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: python -m pip install -U pip build pytest pytest-timeout
      # sibling packages from the tree, so the matrix never pulls PyPI versions of itself
      - run: for d in py/foundation/* py/storage/* py/inference/hugpy_* py/cinema/hugpy_* py/fleet/* py/curation/* py/operations/* py/integrations/* py/services/* py/meta/*; do pip install -e "$d"; done
      - run: cd ${{ matrix.pkg }} && python -m build --wheel
      - run: cd ${{ matrix.pkg }} && pytest -q --timeout=120
      - run: pip install --force-reinstall --no-deps ${{ matrix.pkg }}/dist/*.whl && python -c "import $(basename ${{ matrix.pkg }})"
  react:
    runs-on: ubuntu-latest
    strategy:
      matrix: { pkg: [react/ui, react/agents_ui, react/media_intelligence_ui, react/video_intelligence_ui] }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22 }
      - run: cd ${{ matrix.pkg }} && npm ci && npm run build
```

Publishing is a separate, tag-triggered workflow using PyPI trusted publishing
(`pypa/gh-action-pypi-publish`) and `npm publish --provenance`; tags are
`<dist>/v<version>` (for example `hugpy-engine/v0.2.0`).

### C.5 Release checklist (per package)

1. `python py/validate_partition.py --edges` is green.
2. `CHANGELOG.md` entry; bump `version` in `pyproject.toml` / `package.json`.
3. `cd <pkg> && python -m build --wheel && pytest -q --timeout=120`.
4. Installed-wheel smoke test in a fresh venv: `pip install dist/*.whl && python -c "import <import_name>"`.
5. `twine check dist/*` (README renders; `License-File` present).
6. Tag `<dist>/v<version>`, push tag; CI publishes; verify on the registry page.
7. Update the meta `hugpy` pins if a minor version changed; release `hugpy` last.
8. Refresh the HF `model-catalog` dataset when `hugpy-engine` defaults changed.

### C.6 Org profile README (`hugpy/.github/profile/README.md`)

```markdown
# hugpy — inference you own

Self-hosted LLM console, OpenAI-compatible API and GPU worker fleet.
Pull models from the Hugging Face Hub, chat with streaming continuation,
mint your own API keys, and pool the GPUs you already have — workstations,
laptops, boxes on another network, even phones — into one fleet.

    pip install "hugpy[server]"
    hugpy serve        # console at http://localhost:7002/ , API at /api/v1

## Repositories

| Repo | What it is |
|---|---|
| [hugpy](https://github.com/hugpy/hugpy) | The product: 13 `hugpy-*` Python distributions and the React consoles. Start with `PARTITION.md`. |
| [hugpy-agent](https://github.com/hugpy/hugpy-agent) | Dependency-free agent runtime and client for the fleet. |
| [hugpy-station](https://github.com/hugpy/hugpy-station) | Desktop cockpit (Electron): stations, VMs, terminals. |
| [abstract-identity](https://github.com/hugpy/abstract-identity) | Identity pipeline for video: char360 and identity-render. |

## Packages

PyPI `hugpy`, `hugpy-platform`, `hugpy-control`, `hugpy-storage`, `hugpy-engine`,
`hugpy-media`, `hugpy-video`, `hugpy-oracle`, `hugpy-fleet`, `hugpy-curation`,
`hugpy-ops`, `hugpy-discord`, `hugpy-server`, `hugpy-agent` ·
npm `@hugpy/ui`, `@hugpy/agents-ui`, `@hugpy/media-intelligence-ui`,
`@hugpy/video-intelligence-ui`, `@hugpy/ui-shared`, `@hugpy/station` ·
Hugging Face [hugpy-ai](https://huggingface.co/hugpy-ai)

Source-available; commercial licensing at https://hugpy.ai. Support: support@hugpy.ai
```

### C.7 Topics / tags

- GitHub topics (all repos): `hugpy`, `llm`, `self-hosted`, `openai-compatible`, `llama-cpp`, `transformers`, `gpu`, `model-serving`; monorepo adds `python`, `react`, `flask`.
- PyPI keywords: as applied (C.2). npm keywords: as applied.
- HF space tags: keep the existing `llm`, `inference`, `openai-compatible`, `self-hosted`, `local-llm`, `llama-cpp`, `transformers`, `gpu`, `model-serving`.

### C.8 npm scope README (`@hugpy` org page text; also `react/README.md`)

```markdown
# @hugpy

Front-ends and desktop tooling for hugpy, the self-hosted LLM console.

| Package | Mount | Purpose |
|---|---|---|
| @hugpy/ui | / | Embeddable console: chat, models, workers, API keys, Discord |
| @hugpy/agents-ui | /fleet | Agent fleet console for hugpy-agent nodes |
| @hugpy/media-intelligence-ui | /media | Transcription, summaries, embeddings, vision, documents |
| @hugpy/video-intelligence-ui | /video | Video stations over the hugpy-video job bus |
| @hugpy/ui-shared | – | Help widget and navbar links, zero dependencies |
| @hugpy/station | – | Desktop cockpit (Electron) source; installers at https://hugpy.ai/station |
| @hugpy/console, @hugpy/vm-mgr | – | Embeddable web console and LXD VM manager |

All packages talk to a same-origin hugpy-server (`/api`). Source:
https://github.com/hugpy/hugpy (react/). License: see LICENSE in each package.
```

### C.9 Hugging Face org card (`hugpy-ai` README)

```markdown
# hugpy — inference you own

Self-hosted LLM console, OpenAI-compatible API and GPU worker fleet, built on
llama.cpp and transformers. Install with `pip install "hugpy[server]"`.

- Spaces: live demos of the console (`hugpy-console`), media intelligence
  (`hugpy-media`) and docs for `hugpy-agent` and `@hugpy/console`.
- Datasets: `model-catalog`, the curated default model list and task
  capability map that `hugpy-engine` and `hugpy-curation` ship with, exported
  per release.
- Models: quantisations promoted by the hugpy-curation review pipeline, when
  any are published. hugpy does not mirror third-party weights.

Source: https://github.com/hugpy/hugpy · PyPI: https://pypi.org/project/hugpy/ ·
npm: @hugpy · https://hugpy.ai · support@hugpy.ai
```

---

## D. Execution plan (NOT executed; for a human to run after review)

### D.0 Done locally in this pass

- 13 `pyproject.toml`: `description`, `authors` (name + `support@hugpy.ai`), `keywords`, `classifiers`, `[project.urls]` (Homepage, Documentation, Repository, Source, Issues, Changelog, Architecture). Versions and dependencies untouched. Verified: `hugpy-control` and `hugpy` build wheels with the new metadata.
- 5 React `package.json`: `description`, `keywords`, `homepage`, `repository` (with `directory`), `bugs`, `author`; `license` unchanged.

### D.1 Local prep (before any registry action)

```bash
cd /home/op/Documents/hugpy/trimming
# 1. license text into every distribution (copy the canonical file)
for d in py/foundation/hugpy_platform py/foundation/hugpy_control py/storage/hugpy_storage \
         py/inference/hugpy_engine py/inference/hugpy_media py/cinema/hugpy_video py/cinema/hugpy_oracle \
         py/fleet/hugpy_fleet py/curation/hugpy_curation py/operations/hugpy_ops \
         py/integrations/hugpy_discord py/services/hugpy_server py/meta/hugpy; do
  cp py/inference/hugpy_agent/LICENSE "$d/LICENSE"
  # then add under [project]:  license-files = ["LICENSE"]
done
# 2. replace the 7-line README stubs (platform, control, storage, media, fleet, discord, server)
# 3. add READMEs to react/agents_ui and react/media_intelligence_ui; add react/README.md (C.8)
# 4. bump the 13 versions 0.1.0 -> 0.2.0; pin meta extras to ">=0.2,<0.3"
# 5. add CODEOWNERS, SECURITY.md, CONTRIBUTING.md, .github/workflows/ci.yml (C.3, C.4)
# 6. gate
.venv/bin/python py/validate_partition.py && .venv/bin/python py/validate_partition.py --edges
for d in py/foundation/hugpy_platform py/foundation/hugpy_control py/storage/hugpy_storage \
         py/inference/hugpy_engine py/inference/hugpy_media py/cinema/hugpy_video py/cinema/hugpy_oracle \
         py/fleet/hugpy_fleet py/curation/hugpy_curation py/operations/hugpy_ops \
         py/integrations/hugpy_discord py/services/hugpy_server py/meta/hugpy; do
  (cd "$d" && ../../../.venv/bin/python -m build --wheel --outdir /tmp/wheels && ../../../.venv/bin/python -m pytest -q --timeout=120) || exit 1
done
.venv/bin/python -m twine check /tmp/wheels/*   # must show License-File
```

### D.2 GitHub

```bash
# archived-name cleanup (only if a private archived repo exists; user authorised deletion)
gh repo list hugpy --limit 100 --json name,isArchived,isPrivate
# for each archived name we will reuse:
gh repo clone hugpy/<name> py/unwired/archive/<name>-$(date +%Y%m%d) && gh repo delete hugpy/<name> --yes

# org profile
gh repo create hugpy/.github --public --description "hugpy organisation profile"
#   commit profile/README.md (C.6)

# monorepo
gh repo create hugpy/hugpy --public \
  --description "Self-hosted LLM console, OpenAI-compatible API and GPU worker fleet. 13 hugpy-* Python packages + React consoles." \
  --homepage https://hugpy.ai
#   git init in a clean export of py/ (excluding abstract_hugpy_dev, unwired, .venv, build, dist) and react/ (excluding node_modules), push main
gh repo edit hugpy/hugpy --add-topic hugpy --add-topic llm --add-topic self-hosted --add-topic openai-compatible \
  --add-topic llama-cpp --add-topic transformers --add-topic gpu --add-topic model-serving --add-topic python --add-topic react --add-topic flask
gh repo edit hugpy/hugpy --enable-issues --enable-wiki=false --delete-branch-on-merge

# per-package repos
gh repo create hugpy/hugpy-agent --public --homepage https://hugpy.ai \
  --description "Portable, dependency-free agent runtime and HTTP client for the hugpy self-hosted LLM fleet"
gh repo create hugpy/hugpy-station --public --homepage https://hugpy.ai \
  --description "hugpy Station: the desktop cockpit (Electron) over the self-contained console backend"
gh repo create hugpy/abstract-identity --public --homepage https://hugpy.ai \
  --description "Identity pipeline for hugpy video: char360 face detection/clustering and the identity-render GPU service"
for r in hugpy-agent hugpy-station abstract-identity; do
  gh repo edit hugpy/$r --add-topic hugpy --add-topic llm --add-topic self-hosted
done

# retire the AbstractEndeavors snapshots (archive, keep URLs alive)
gh repo edit AbstractEndeavors/hugpy --description "Archived: moved to https://github.com/hugpy/hugpy"
gh repo archive AbstractEndeavors/hugpy --yes
gh repo edit AbstractEndeavors/abstract_hugpy --description "Archived 2025 predecessor of hugpy: https://github.com/hugpy/hugpy"
gh repo archive AbstractEndeavors/abstract_hugpy --yes
# AbstractEndeavors/abstract_hugpy_dev: archive only after D.4 (bridge release) has shipped
# AbstractEndeavors: create the missing abstract_toolserver / abstract_apply / abstract-claude / abstract_gpt repos or fix their URLs (out of hugpy scope)

# org settings
gh api -X PATCH /orgs/hugpy -f description="Self-hosted LLM console, OpenAI-compatible API and GPU worker fleet. Inference you own." -f blog="https://hugpy.ai" -f email="support@hugpy.ai"
```

### D.3 PyPI (first partitioned release, dependency order)

Publish bottom-up so every `Requires-Dist` resolves at upload time.

```bash
cd /home/op/Documents/hugpy/trimming
rm -rf /tmp/wheels
ORDER="py/foundation/hugpy_platform py/foundation/hugpy_control py/storage/hugpy_storage \
       py/inference/hugpy_engine py/inference/hugpy_media py/cinema/hugpy_video py/cinema/hugpy_oracle \
       py/fleet/hugpy_fleet py/curation/hugpy_curation py/operations/hugpy_ops \
       py/integrations/hugpy_discord py/services/hugpy_server py/meta/hugpy"
for d in $ORDER; do (cd "$d" && ../../../.venv/bin/python -m build --outdir /tmp/wheels); done
.venv/bin/python -m twine check /tmp/wheels/*
# dry run against TestPyPI first
.venv/bin/python -m twine upload --repository testpypi /tmp/wheels/*
# then production, one distribution at a time in ORDER (fails fast on a bad upload):
for d in $ORDER; do n=$(basename "$d"); .venv/bin/python -m twine upload /tmp/wheels/${n}-0.2.0*; done
# metadata-only follow-ups for independents (add [project.urls]/classifiers first)
(cd py/cinema/abstract_identity && ../../../.venv/bin/python -m build && ../../../.venv/bin/python -m twine upload dist/*0.1.1*)
(cd py/inference/hugpy_agent   && ../../../.venv/bin/python -m build && ../../../.venv/bin/python -m twine upload dist/*0.1.71*)
(cd py/tools/abstract_apply    && ../../../.venv/bin/python -m build && ../../../.venv/bin/python -m twine upload dist/*0.2.0*)   # first publish
```

### D.4 Retire `abstract_hugpy_dev` and `abstract-hugpy`

```bash
# bridge release 0.1.267: pyproject dependencies = ["hugpy[all]>=0.2,<0.3"]; __init__ warns:
#   warnings.warn("abstract_hugpy_dev is retired; install 'hugpy' (https://pypi.org/project/hugpy/)", DeprecationWarning)
#   and keeps _relocations.py so old dotted paths alias the hugpy_* modules
(cd abstract_hugpy_dev && ../.venv/bin/python -m build && ../.venv/bin/python -m twine check dist/* && ../.venv/bin/python -m twine upload dist/*0.1.267*)
# same for abstract-hugpy 0.1.402 (dependencies = ["hugpy>=0.2,<0.3"], warning), from AbstractEndeavors/abstract_hugpy
# 30 days later: yank every earlier release (yanked releases still satisfy existing pins; nothing is deleted)
#   PyPI web UI -> project -> Releases -> Yank, reason "Superseded by hugpy 0.2 (https://pypi.org/project/hugpy/)"
#   (there is no twine/pip CLI for yank; use the web UI or the PyPI JSON-API-less form)
# Finally: gh repo archive AbstractEndeavors/abstract_hugpy_dev --yes
```

### D.5 npm

```bash
cd /home/op/Documents/hugpy/trimming/react
# scope README: paste C.8 in the npm org settings; add react/README.md with the same text
for d in ui_shared ui agents_ui media_intelligence_ui video_intelligence_ui; do
  (cd "$d" && npm version patch --no-git-tag-version && npm ci && npm run build && npm pack --dry-run)
done
# publish (ui_shared first; ui's postbuild bundles the media arm)
for d in ui_shared ui agents_ui media_intelligence_ui video_intelligence_ui; do (cd "$d" && npm publish --access public --provenance); done
# station: align npm with the deb channel; source lives at /srv/hugpy/src/station-app (symlink in py/tools/hugpy-station/source is dangling here)
#   add repository/bugs to its package.json, bump to 1.0.110 with cut-next-version.sh, then:
#   (cd /srv/hugpy/src/station-app && npm publish --access public --provenance) and ./publish-latest.sh for the .deb
# @hugpy/console and @hugpy/vm-mgr: locate their source (not in this workspace), add repository/bugs, republish as 0.3.2 / 0.2.1
```

### D.6 Hugging Face

```bash
pip install -U huggingface_hub && huggingface-cli login
# org card (C.9): Settings -> Organization card, or push README.md to the org's profile space
# rename hugpy-chat -> hugpy-media (Space settings -> Rename) and refresh all five space cards with the new package names/links
# model-catalog dataset (after adding `hugpy-curation export-catalog` that dumps models_default + task_deps + verdicts to JSON)
huggingface-cli repo create model-catalog --type dataset --organization hugpy-ai
huggingface-cli upload hugpy-ai/model-catalog ./export/model-catalog . --repo-type dataset --commit-message "hugpy-engine 0.2.0 catalog"
```

### D.7 Per-package metadata checklist

| Package | urls | description | keywords | authors | classifiers | LICENSE file + `license-files` | README (real) | version bump |
|---|---|---|---|---|---|---|---|---|
| hugpy | done | done | done | done | done | todo | ok (374 lines) | 0.1.0 -> 0.2.0 |
| hugpy-platform | done | done | done | done | done | todo | **stub** | 0.2.0 |
| hugpy-control | done | done | done | done | done | todo | **stub** | 0.2.0 |
| hugpy-storage | done | done | done | done | done | todo | **stub** | 0.2.0 |
| hugpy-engine | done | done | done | done | done | todo | ok | 0.2.0 |
| hugpy-media | done | done | done | done | done | todo | **stub** (9 lines) | 0.2.0 |
| hugpy-video | done | done | done | done | done | todo | ok | 0.2.0 |
| hugpy-oracle | done | done | done | done | done | todo | ok | 0.2.0 |
| hugpy-fleet | done | done | done | done | done | todo | **stub** | 0.2.0 |
| hugpy-curation | done | done | done | done | done | todo | ok | 0.2.0 |
| hugpy-ops | done | done | done | done | done | todo | ok | 0.2.0 |
| hugpy-discord | done | done | done | done | done | todo | **stub** | 0.2.0 |
| hugpy-server | done | done | done | done | done | todo | **stub** | 0.2.0 |
| hugpy-agent | ok | ok | ok | ok | partial (add OS/Topic) | ok | ok | 0.1.71 |
| abstract-identity | **todo** (none) | ok | todo | todo (email) | **todo** (none) | ok (MIT) | ok | 0.1.1 |
| abstract-toolserver | fix 404 repo URL | ok | todo | ok | ok | ok | ok | 0.0.28 |
| abstract-apply | fix 404 Homepage | ok | todo | ok | todo | ok | ok | 0.2.0 (first publish) |
| abstract-claude / abstract-gpt | todo (only Homepage) | ok | ok / todo | unify identity (`jrputkey` -> `putkoff`) | todo | ok / todo | ok | 0.1.52 / 0.1.3 |
| @hugpy/ui, agents-ui, media-intelligence-ui, video-intelligence-ui, ui-shared | done | done | done | done | n/a | ok | ui, video, ui-shared ok; **agents-ui, media-intelligence-ui missing** | patch |
| @hugpy/station | todo | ok | todo | ok | n/a | ok | ok | 1.0.110 |

---

## Unverified

- Private repositories in the `hugpy` org (the public API lists zero; archived
  private repos, if any, need `gh repo list hugpy` with a token).
- The npm org page (`/org/hugpy`) returns 403 to unauthenticated fetches and
  the search index does not list the scope; the registry documents were used
  instead. The sixth HF space (`numSpaces: 6`) is not returned by the public
  list and is presumably private.
- `py/tools/hugpy-station/source` symlinks to `/srv/hugpy/src/station-app`,
  which does not exist on this machine, so the station `package.json` could
  not be read; the deb version (1.0.109) comes from `map.toml`.
- Source for `@hugpy/console` and `@hugpy/vm-mgr` is not in this workspace.
- PyPI HTML search was not used (JSON per name instead), so distributions by
  `putkoff` outside the candidate list may exist.
