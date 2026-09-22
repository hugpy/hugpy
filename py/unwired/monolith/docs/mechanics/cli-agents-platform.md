# CLI, bot, keeper & platform foundation — mechanics

**Scope (paths this doc covers):** `cli.py`, `bot/`, `keeper.py`, `_platform/`,
`utils/`, `imports/` (all under `src/abstract_hugpy_dev/src/abstract_hugpy_dev/`),
plus the side packages at `/srv/hugpy/src/py/{abstract_apply,abstract_identity,
hugpy_agent,hugpy_video}` (their own separate repos/dists — NOT under the
package path above).
**One-liner:** The `hugpy` console script's subcommand dispatch (serve/worker/
bot/keeper/chat/install-*), the Discord bot arm, the terminal keeper REPL, and
the cross-platform/utility/import foundation (`_platform/`, `utils/`,
`imports/`) that most of the rest of the package is built on.
**Owner process(es):** `cli.py` is the OS-level entry for central (`hugpy serve`
→ `7002_hugpy_api`), the worker agent (`hugpy worker`), and the Discord bot
(`hugpy bot`, its own process/unit); `keeper.py`/`hugpy chat` are ad hoc terminal
tools, not daemons. `_platform/`, `utils/`, `imports/` are library-only — imported
by nearly everything, including `managers/`, `flask_app/`, `worker_agent/`.

## 1. Purpose & responsibilities
- Give the package one console-script entry (`hugpy`) that dispatches to five
  processes/tools (serve, worker, bot, keeper, chat) plus two one-shot admin
  commands (install-engine, install-deps), each importing its own heavy stack
  lazily so a bare `hugpy --help` stays cheap.
- Run a Discord front-end (`bot/`) that talks to a hugpy central purely over
  HTTP (`HugpyClient`) — never in-process — so it can point at any central,
  local or remote.
- Provide a stdlib-only terminal REPL (`keeper.py`) in which a model "keeps" a
  machine: chat + a shell action loop, runnable without installing hugpy.
- Seam off every POSIX-only assumption (systemd, `/proc`, `nvidia-smi`,
  `os.killpg`) behind `_platform/` so the same code runs on Windows/macOS/
  Termux, degrading to `None`/no-op rather than crashing; host small
  standalone helpers (`utils/`) and the centralized dependency/config
  re-export layer (`imports/`) that `__init__.py` star-imports first.
- Deliberately does NOT serve HTTP itself (`flask_app/`, `[[api-routes]]`), run
  inference (`managers/`, `[[engine-generation]]`), or manage worker enrollment
  (`worker_agent/`, `[[worker-fleet]]`) — this doc's subsystems are the entry
  seam and the foundation underneath those, not the product surface itself.

## 2. Key modules (file → responsibility)

| file | responsibility |
|---|---|
| `cli.py` | `hugpy` console-script entry; argparse subcommands → serve/worker/bot/chat/install-engine/install-deps/reclassify-images, plus `keeper` via a bypass |
| `keeper.py` | stdlib-only terminal REPL — `LocalExec`/`LxcExec` "hands", streaming-chat "brain", optional headless Discord bridge relay |
| `central.py` | single source of truth for the central base URL (`HUGPY_BASE_URL` + 3 legacy aliases) |
| `_compat_pydantic.py` | pure-Python pydantic shim registered under `sys.modules["pydantic"]` when `pydantic_core` can't import (Termux/Android) |
| `bot/bot.py` | `HugpyBot(commands.Bot)` — cog loader, escalation buttons/selects, outbox/channel/member background pollers |
| `bot/config.py` | env/.env-driven bot config; resolves `HUGPY_BASE_URL` via `central.py`, plus `DISCORD_TOKEN`/`GUILD_ID`/`OPERATOR_DISCORD_ID` |
| `bot/hugpy_client.py` | async httpx client for central's HTTP surface (models/jobs/chat/discord bindings/settings/prompt passthrough) |
| `bot/streamer.py` | `MessageStreamer` — throttled (1.2s) Discord edit-as-you-stream, rolls to a new message past 1900 chars |
| `bot/prefs.py` | JSON-backed per-user default-model store (`~/.hugpy/bot/prefs.json`); local fallback to central settings |
| `bot/cogs/chat.py` | `/chat /model /reset /running /stop` + mention/DM/bridged-channel auto-reply; owns the streaming turn + server-side cancel |
| `bot/cogs/tools.py` | `/summarize /keywords /transcribe /describe` — dedicated-task commands over `execute_prompt`, falling back to `/chat/stream` on an old central |
| `bot/cogs/ml.py` | `/embed /similarity /imagine /task /tasks` — the fully-generic `execute_prompt` mirror |
| `bot/cogs/ops.py` | `/status /models /download /jobs /canceljob /hf` — operational view of central |
| `bot/cogs/helpers.py` | shared: model autocomplete/label, attachment forwarding, long-result chunking/file-attach |
| `_platform/paths.py` | per-OS data/config/cache/engine dirs; `HUGPY_HOME` skeleton (state/config/logs/run) with one-time legacy-file migration |
| `_platform/hardware.py` | RAM/GPU probes (psutil/torch.cuda/nvidia-smi → `None` on failure, never crash) |
| `_platform/binaries.py` | `resolve_bin()` — portable executable resolution (`.exe` on Windows, PATH + extra dirs) |
| `_platform/procutil.py` | `popen_detached`/`terminate_tree`/`reexec` — portable spawn / process-tree kill / re-exec |
| `_platform/async_runtime.py` | the ONE process-wide asyncio loop; `run()`/`iter_sync()` with abandon-on-disconnect |
| `_platform/client_liveness.py` | `SocketProbe` — gunicorn-socket `MSG_PEEK` disconnect detection, bound per request via `threading.local` |
| `utils/no_think.py` | reasoning-suppression seam: soft directive + hard template kwarg + `<think>` strip/stream-split |
| `utils/json_scavenge.py` | recover a well-formed JSON object/array from a chatty model reply (fence/brace-walk); `None` on failure, never repairs |
| `utils/text/combined.py` | SEO/text-analysis surface: source→text extractor registry (image/audio/video/website/pdf) + `analyze()`/`summarize()` |
| `utils/pdfs/utils.py`, `utils/seo/pdf_utils.py` | dual-extractor (pdfplumber+PyPDF2) PDF text + page/full-document SEO report (summary+keywords) |
| `imports/__init__.py` (+ `src/`, `apis/`, `config/`) | the star-import waterfall — see §4 Flow D; `config/main.py`/`apis/*` also carry real model-path/HF-download logic, not pure re-exports |
| `imports/src/module_imports.py` | lazy-import gate: `is_available()`/`require()` over `lazy_import`/`nullProxy` |

## 3. Entry points
- Console script `hugpy` = `abstract_hugpy_dev.cli:main` (`pyproject.toml:170`).
- `hugpy serve` → `cli.py:28 _serve()` → `flask_app.get_hugpy_flask()` served via
  gunicorn (`cli.py:61` `_App`, 1 worker/N threads) or waitress/dev-server
  fallback. This is central, `:7002`.
- `hugpy worker […]` → `cli.py:77 _worker()` → `worker_agent.agent.main()`; argv
  after `worker` is split off BEFORE argparse parses it (`cli.py:399-400`) so
  the agent owns its own flags. See `[[worker-fleet]]`.
- `hugpy bot` → `cli.py:82 _bot()` — sets env vars from `--central/--env/--guild`
  BEFORE importing `bot.config` (which reads them at import time, `bot/config.py:15-16`)
  → `HugpyBot().run(token)`.
- `hugpy keeper […]` → `cli.py:403-409` bypasses the package import entirely:
  loads `keeper.py` via `importlib.util.spec_from_file_location` so it (in
  theory) works even when the heavy `abstract_hugpy_dev` `__init__` chain would
  fail. **Currently broken — see §8.**
- `hugpy chat [prompt]` → `cli.py:120 _chat()` — stdlib `urllib` SSE client
  against `/chat/stream`; Ctrl-C posts `/llm/chat/cancel/<request_id>` before
  exiting. `install-engine` / `install-deps` / `reclassify-images` are one-shot
  admin commands: `cli.py:198` / `273` / `229`.
- Discord slash commands (`/chat`, `/summarize`, `/keywords`, `/transcribe`,
  `/describe`, `/embed`, `/similarity`, `/imagine`, `/task`, `/tasks`,
  `/status`, `/models`, `/download`, `/jobs`, `/canceljob`, `/hf`, `/link`,
  `/model set|show|clear`, `/reset`, `/running`, `/stop`) — each cog's
  `setup(bot)`; `bot/bot.py:15,233` loads `COGS = (".cogs.chat", ".cogs.tools",
  ".cogs.ml", ".cogs.ops")`. Plain mention/DM chat and bridged-channel relay:
  `ChatCog.on_message` (`bot/cogs/chat.py:127`).
- `python3 keeper.py --model <id> [--exec lxc:<name>]` — advertised as
  runnable straight from source, no install (`keeper.py:1-24` docstring).
  **Currently broken — see §8.** The invocation that DOES work is
  `python -m abstract_hugpy_dev.keeper`.
- Library entry points other subsystems import: `_platform.async_runtime.run()`/
  `iter_sync()`, `_platform.client_liveness.install(app)` (Flask
  `before_request` hook), `utils.no_think.execute_prompt_no_think()`,
  `imports.src.module_imports.require()`/`is_available()`.

## 4. Data flow (the spine)

**Flow A — `hugpy` dispatch (`cli.py`).** (1) `main()` (`cli.py:333`) builds one
argparse parser with subparsers. (2) `worker`/`keeper` are special-cased BEFORE
`parser.parse_args` (`cli.py:399-409`) — worker's entire remaining argv passes
through untouched; keeper is `exec_module`'d directly off disk, bypassing the
package `__init__`. (3) Every other subcommand imports its subsystem lazily
INSIDE its handler, not at module top (e.g. `flask_app` only on `serve`,
`cli.py:37`; `discord`/`bot.bot`/`bot.config` only on `bot`, `cli.py:94-102`) —
so `hugpy --help` never pays for the heavy stack.

**Flow B — Discord message → streamed reply (`bot/cogs/chat.py`).**
(1) `on_message` (`chat.py:127`) or a slash command assembles `messages`
(history + channel-personality system prompt + new turn). (2)
`bot.resolve_model_for` (`bot/bot.py:310-335`) picks the model: console-managed
Discord binding > channel personality > channel delegation > user pref >
configured default. (3) `HugpyClient.chat_stream()` (`hugpy_client.py:232-292`)
POSTs `/chat/stream` with a `request_id`, parsing `data:` SSE lines. (4) Each
chunk feeds `MessageStreamer.feed()` (`streamer.py:32-36`), which throttles
Discord edits to 1.2s and rolls content into a new message past 1900 chars.
(5) `/stop` (`chat.py:251-284`) calls `HugpyClient.cancel_chat(request_id)`
**server-side first**, then cancels the local asyncio task — the generation
actually stops on central/worker, not just the local SSE read.

**Flow C — keeper action loop (`keeper.py`).** (1) `stream_chat()`
(`keeper.py:123-149`) POSTs `/v1/chat/completions` (OpenAI-compatible,
`stream=True`) to the configured central. (2) `parse_action()`
(`keeper.py:152-159`) looks for a fenced ` ```action ` block in the reply.
(3) `run_action()` (`keeper.py:87-101`) executes it via the chosen executor —
`LocalExec` (`bash -lc` locally) or `LxcExec` (`lxc exec … --user … -- bash
-lc`) — capped at 300s, output truncated to `--obs-limit` (default 6000
chars). (4) The observation re-enters as a `user` turn; the loop repeats up to
`--max-steps` (default 8; `run_agent_turn`, `keeper.py:162-190`). (5) Optional
`--bridge <id>` runs headless (`run_bridge_loop`, `keeper.py:221-276`), polling
`/discord/bridges/<id>/messages` and posting to `.../keeper-reply` instead of a
terminal prompt.

**Flow D — the `imports/` star-import waterfall (package import time).**
(1) `__init__.py:5-6` runs `_compat_pydantic.ensure_pydantic()` FIRST, before
anything else can import pydantic. (2) `__init__.py:25-27`: `from .imports
import *`, `from .managers import *`, `from .utils import *`. (3)
`imports/__init__.py:1-3`: `from .src import *` → `.apis` → `.config`. (4)
`imports/src/__init__.py:1-8` pulls in `constants`, `init_imports` (the raw
stdlib/3rd-party dump — `init_imports.py:4` is where `pydantic.BaseModel`/
`ConfigDict`/`Field`/`model_validator` enter the chain), `module_imports` (the
lazy-import gate), `chunking`, `schemas`, `utils`, `except_utils`,
`peft_adapters`, all via `import *` (see §8 on the lack of `__all__`). (5)
`imports/config/main.py` and `imports/apis/*` layer REAL logic on top, not
just re-exports — model-path/GGUF-quant election (`get_model_path`
`main.py:93-107`, `get_gguf_file` `:110-153`, `resolve_model_source` `:276-309`)
and HF download/audit/reconcile. `imports/src/model_index/*` (the
model-metrics/calls registry) is out of scope here — see `[[comms-index]]`.

## 5. State, persistence & invariants
- `_platform.paths.hugpy_home()` — `HUGPY_HOME` (default `~/.hugpy`), split into
  `state/`, `config/`, `logs/`, `run/`; each named accessor (`todo_file()`,
  `worker_id_file()`, `steward_state()`, `model_metadata_db()`, …) migrates a
  legacy `~/<name>` file into its new home exactly ONCE, atomically
  (`paths.py:194-219` `_relocate`, same-fs rename / cross-fs copy fallback).
- `_platform.env_value()` (`_platform/__init__.py:67-98`) is the SINGLE place
  config overrides are read — prefers `abstract_essentials.get_env_value` (via
  `imports.src.standalone_utils`), falls back to `os.environ`; both paths run
  through `_sanitize_env_str` (`:29-64`) so a trailing inline `# comment` never
  leaks into a path value. Regression-tested end-to-end:
  `tests/test_platform_env_value.py` (the "computron incident").
- `_platform.async_runtime` holds exactly ONE event loop for the whole process
  (module-level `_loop`/`_thread`, guarded by `_start_lock`, `async_runtime.py:42-69`)
  — never one loop per request, so cached asyncio sync primitives stay bound
  to a loop that's still alive.
- `_platform.client_liveness` binds a `SocketProbe` to a `threading.local` per
  Flask request (`install()`'s `before_request`/`teardown_request`,
  `client_liveness.py:186-207`); `HUGPY_CLIENT_DISCONNECT_ABANDON=off` restores
  hold-forever behavior.
- `bot/prefs.py` `PrefsStore` persists to `DATA_DIR/prefs.json` — local
  fallback only; central settings (`/discord/prefs`) is authoritative
  (`bot/bot.py:301-308` `set_user_model` writes central FIRST, then local).
- `bot/bot.py` keeps two in-memory sets: `_bridged` (channels bridged to a
  console session, refreshed every 8s by `_outbox_poller`, `:338-349`) and
  `_answered_escalations` (best-effort click idempotency — the durable guard is
  the message's own disabled-component state, since the set doesn't survive a
  restart; see §8).
- `keeper.py` is stateless across runs — no persistence beyond what
  `--bridge`/`--mirror` push to central; chat history lives only in the
  in-process `messages` list for the REPL's lifetime (`/clear` resets it).
- `imports/config/main.py`'s `DEFAULT_PATHS` (`_LazyModelPaths`, `:328-351`)
  resolves on `__getitem__`, not at import time, so a model downloaded/deleted
  after import still resolves correctly.

## 6. Cross-subsystem edges
**← calls into this doc's subsystems:** `flask_app/` (`[[api-routes]]`) calls
`_platform.async_runtime.run()`/`iter_sync()` from SSE/streaming routes and
`_platform.client_liveness.install(app)` once at app creation; systemd units
run `hugpy serve`/`worker`/`bot` via `cli.py`; `managers/llama/runners/
ccp_runner.py` re-inlines `<think>` into `content` — a wire-format contract
`utils.no_think.strip_think()` later reverses; virtually every subsystem does
`from ..imports import *` to reach the flat re-exported namespace.

**→ calls out from this doc's subsystems:** `cli.py` → `flask_app.
get_hugpy_flask`, `worker_agent.agent.main`, `bot.bot.HugpyBot`, `engine.
{build,fetch,resolve}`, `imports.apis.reclassify.reclassify_images`. `bot/` →
central's HTTP API only, never `managers`/`flask_app` in-process (deliberate —
the bot can point at ANY central). `keeper.py` → central's `/v1/chat/
completions` and `/discord/bridges/*` over stdlib `urllib` only, no package
imports (by design) — except the one broken exception, see §8. `utils.no_think`
→ `managers.dispatch.execute_prompt`, `_platform.async_runtime.run`. `utils.
text.combined` → the generation stack (`ChatRequest`/`runner_for`) via the
`imports` chain, plus optional `abstract_ocr`/`abstract_webtools` extras
(lazy-imported). `imports/config/main.py` → `managers.serve.hot_cache` (lazy
import, avoids an import-time cycle).

**Sibling docs:** `[[serving-core]]` (consumes `_platform.async_runtime`/
`paths`), `[[engine-generation]]` (dispatch is what `no_think` wraps),
`[[comms-index]]` (owns `imports/src/model_index/*`), `[[worker-fleet]]`
(`worker_agent` is what `hugpy worker` launches), `[[api-routes]]` (`flask_app`
is what `hugpy serve` launches).

## 7. Key contracts / types
- `central_base_url(default=DEFAULT_CENTRAL)` (`central.py:23-34`) — canonical
  resolver: `HUGPY_BASE_URL` first, then legacy `HUGPY_CENTRAL`/`HUGPY_URL`/
  `WORKER_CENTRAL_URL`; first non-empty wins.
- `lazy_import(name)` / `nullProxy` / `is_available(name)` / `require(name,
  reason)` (`imports/src/module_imports.py:27-75`, backed by
  `standalone_utils.py`'s `abstract_essentials` re-export) — a missing optional
  dep returns a `nullProxy` silently UNLESS the caller uses `require()`, which
  raises `ImportError` naming the exact `pip install 'abstract_hugpy_dev[extra]'`.
- `strip_think(text) -> (prose, reasoning)` / `StreamingThinkSplitter` /
  `NO_THINK_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}`
  (`utils/no_think.py:104,111-124,135-205`) — the two-half contract: soft
  directive text + hard template kwarg, always applied together, never one
  without the other.
- `extract_json_object`/`extract_json_array`/`extract_json_value`
  (`utils/json_scavenge.py:43-111`) — return `None` on failure; recovery of a
  well-formed value with garnish, never repair of malformed content.
- `SocketProbe.gone() -> bool` / `ClientGone` (`_platform/client_liveness.py:59-65,103-127`)
  — conservative-false: any uncertainty (TLS socket, unclassified errno, no
  socket published) reads as "still connected".
- pydantic shim surface (`_compat_pydantic.py:130-138`): `BaseModel.{model_dump,
  dict, model_dump_json, model_validate, parse_obj, __repr__}`, `Field`,
  `ConfigDict`, `model_validator`/`field_validator`/`validator`/`root_validator`
  (no-op passthrough decorators), `ValidationError`. No type validation/coercion
  — and, load-bearing for §8, **no `model_copy`**.
- `GenParams`/`ChatRequest`/`ChatResult` (`utils/text/combined.py:20-31`,
  `imports/src/schemas/chat_schemas.py`) — pydantic `BaseModel` wire shapes;
  `GenParams.to_kwargs()` = `model_dump()`.

## 8. Gotchas, tech-debt & review findings

⚠ **`hugpy keeper` and the docstring's own standalone invocation are both
currently broken — verified by running them.** `keeper.py:38` does `from
.central import central_base_url` at module level, but keeper.py's own
docstring (`:1-24`) advertises two invocations that cannot satisfy a relative
import: `python3 path/to/hugpy/keeper.py` (no package context at all) and,
identically, `cli.py:403-409`'s `importlib.util.spec_from_file_location(
"hugpy_keeper", …)` + `exec_module` (loads it as a top-level module named
`hugpy_keeper`, `__package__` empty). Reproduced today against BOTH copies — `python3 …/src/abstract_hugpy_dev/keeper.py
--help` and `hugpy keeper --help` (the installed wheel, `/srv/hugpy/venv/bin/hugpy`)
both raise `ImportError: attempted relative import with no known parent
package` at `keeper.py:38`. `python -m abstract_hugpy_dev.keeper --help` DOES
work (proper package context) but is not documented anywhere as the way to
invoke it. Fix is narrow: give keeper.py a fallback (`try: from .central
import central_base_url; except ImportError: <inline resolver>`), or have
`cli.py`'s bypass load it as `abstract_hugpy_dev.keeper` via `runpy`/adjusted
`sys.path` instead of a bare file spec.

⚠ **`_compat_pydantic` shim has no `model_copy` — `utils/text/combined.py:131,172,186`
will crash on a shim install.** `analyze()`/`analyze_pdf_by_page()` call
`params.model_copy(update={...})` on a `GenParams(BaseModel)` (`combined.py:20`).
The shim's `BaseModel` (`_compat_pydantic.py:56-104`) implements `model_dump`/
`dict`/`model_dump_json`/`model_validate`/`parse_obj` but not `model_copy`. On
any process where `ensure_pydantic()` had to install the shim (no
`pydantic_core` wheel — Termux/Android; `__init__.py:5-6` runs this before
anything else), this code path raises `AttributeError: 'BaseModel' object has
no attribute 'model_copy'`. `PyPDF2`/`bs4` (imported unconditionally in
`imports/src/init_imports.py:4,25`) are base deps, so `GenParams` is defined on
every install regardless of platform — only the `.model_copy()` CALL is
shim-conditional. The same call pattern appears in 5 files under `managers/`
and 2 under `flask_app/` (`grep -rl "model_copy(" src/abstract_hugpy_dev` —
out of this doc's scope, but the same fix applies package-wide: add a
`model_copy` to the shim, or avoid the call on any path reachable from a shim
install).

△ **`imports/` is a flat, unguarded star-import waterfall.** `imports/__init__.py`
→ `.src`/`.apis`/`.config`, and `imports/src/__init__.py` alone `import *`s
eight submodules, including `init_imports.py` (which dumps ~30 raw names — `os`,
`json`, `re`, `bs4`, `requests`, `pydantic.BaseModel`, `PyPDF2.PdfReader`,
`huggingface_hub.*`, a module-level `logger`, …) with no `__all__`. Only 4
files in the whole `imports/` tree define `__all__`
(`peft_adapters.py`, `model_classifier.py`, `model_index/__init__.py`,
`reclassify.py`). Any `from ..imports import *` consumer inherits all of it;
namespace collisions between two submodules that both bind, say, `logger` or
`exists` are silent (last import wins) and hard to trace to their origin.

△ **`abstract_identity` is a hard pip dependency with zero import sites.**
Declared `abstract_identity>=0.1.0` at `pyproject.toml:62`; grepping all of
`src/abstract_hugpy_dev` for the string finds only that declaration and its own
adjacent comment ("only installed on the GPU render box"). If it's meant to be
consumed as an external service/console-script (`char360`/`identity-render`)
rather than imported in-process, that should be spelled out next to the
dependency — as written it reads like unused weight on every install that
isn't the GPU render box.

ℹ **None of the four `src/py/*` side packages are imported by the main
package** (verified by grep across all of `src/abstract_hugpy_dev`):
  - `abstract_apply` — standalone CLI turning a job-posting URL into an
    application package using the hugpy fleet as its LLM backend. Pure
    consumer; zero references either direction.
  - `abstract_identity` — identity pipeline (`char360`: video → per-character
    360° view sets via YOLO+insightface+clustering; `render`: photos →
    Hunyuan3D-2mv mesh, GPU job service on :9750). Declared dependency, zero
    import sites — see the △ finding above.
  - `hugpy_agent` — portable, stdlib-only agent runtime that uses hugpy's
    OpenAI-compatible API as its inference brain; an independent consumer,
    named only in comments for default-value parity (`imports/src/constants/
    constants.py:279`, `chat_schemas.py:183`, `todo_keeper_daemon.py:66`).
  - `hugpy_video` — declared extraction target for `video_intel/` (~24k LOC);
    its own README says "Status: P0 skeleton". `video_intel/` here is still
    the real, live-imported runtime today.

ℹ **`imports/` is not a pure re-export layer despite the name.** ARCHITECTURE.md
and `docs/mechanics/README.md` both describe it as a "centralized dependency/
config re-export layer", but `imports/config/main.py` holds real model-path/
GGUF-quant-election logic and `imports/apis/{download_models,reconcile,
get_module}.py` hold real HF download/staging/audit logic (§4 Flow D). A
reader expecting inert glue will be surprised to find business logic here.

ℹ **Escalation-click idempotency is two-tier and the fast tier is provably
redundant.** `bot/bot.py`'s `_answered_escalations` in-memory set
(`:110-117`) is checked before the durable guard, `_all_components_disabled`
(the message's own disabled-component state, survives a restart). Since the
durable check alone is sufficient for correctness, the in-memory set is pure
optimization (skips one Discord-message inspection) — worth knowing before
"fixing" an apparent redundancy here.

## 9. Deploy/run boundary
- Central (`7002_hugpy_api`) runs the **src** checkout via a `PYTHONPATH=
  /srv/hugpy/src/abstract_hugpy_dev/src` unit override (confirmed via
  `systemctl show 7002_hugpy_api -p Environment`) — so editing any file in this
  doc's scope is live on the next `systemctl restart 7002_hugpy_api`, no wheel
  rebuild needed.
- Everything else — the Discord bot process, workers, a bare interactive
  `hugpy` invocation from a login shell (no `PYTHONPATH` override) — runs the
  PIP-INSTALLED wheel. Confirmed drift right now: installed `__version__` is
  `0.1.251`, `pyproject.toml`'s src version is `0.1.257` (`README.md §Deploy
  loop` closes the gap: `touch signals/publish-dev.trigger` → pyit-jobs build/
  publish → `pip install -U abstract_hugpy_dev` + restart). Don't debug via a
  bare interactive `hugpy` and assume it reflects the latest src.
- `keeper.py` is meant to run from a shared source mount with no install and no
  package import; see §8 for why that currently fails. Once fixed it still
  needs no deploy step — a raw file copy is enough.
- `_compat_pydantic.py` and the `imports/` star-import chain run at `import
  abstract_hugpy_dev` time for every consumer — a change here reaches every
  process on next restart, including `hugpy-downloader` and every
  `python -m abstract_hugpy_dev.*` daemon.
- pip extras (`bot`, `cpu-worker`, `gpu-worker`, `engine`, `native`, …;
  `pyproject.toml:75-165`) gate which `_platform` probes resolve to real
  libraries (torch/psutil/pynvml) vs. degrade to `None`/`nullProxy` — the wrong
  extra set doesn't break anything, it silently narrows what `hardware.py`/
  `module_imports.py` can report.
