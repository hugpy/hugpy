# hugpy — run your own models, your own way

**A self-hosted LLM console and OpenAI-compatible API in a single process.**
Model registry & downloads, streaming chat, an OpenAI-compatible `/v1` surface
with on-site API keys, and a GPU worker fleet with cross-machine RPC sharding —
all served by one command, with no nginx and no Node required.

```bash
pip install "hugpy[server]"
hugpy serve            # console at http://localhost:7002/ , API at /api/v1
```

`hugpy` is the **meta distribution**: it installs the `hugpy` and `hpy`
commands and picks the rest of the product through extras. The implementation
lives in a family of `hugpy-*` distributions (below); the `hugpy` command only
dispatches to them, so a bare `pip install hugpy` stays tiny and imports
nothing but the standard library and `hugpy-platform`.

---

## Table of contents

- [Why hugpy](#why-hugpy)
- [The distributions](#the-distributions)
- [Install](#install)
- [Quickstart](#quickstart)
- [OpenAI-compatible API](#openai-compatible-api)
- [Command-line interface](#command-line-interface)
- [GPU worker fleet & sharding](#gpu-worker-fleet--sharding)
- [Discord bot](#discord-bot)
- [Configuration](#configuration)
- [Storage & paths](#storage--paths)
- [Authentication](#authentication)
- [Platform notes](#platform-notes)
- [Links & license](#links--license)

---

## Why hugpy

Most "run a model" tools stop at load-and-infer. hugpy is the operational layer
around self-hosting:

- **One process, whole product.** `hugpy serve` runs the API, the web console,
  model downloads, chat, and the OpenAI `/v1` surface. No reverse proxy, no
  separate frontend build.
- **OpenAI-compatible.** Point any OpenAI SDK at your box — `base_url` + an
  `hp_…` key and you're done.
- **Bring your own GPUs.** Join any machine to a central as a worker
  (`hugpy worker`), or lend its GPU to a **cross-machine shard pool** so models
  larger than one card can run across several boxes over RPC.
- **Phone-to-server install.** The base install is small and wheels-only (it
  runs on Termux/aarch64 as a coordinator); the heavy engine, vision, OCR, and
  media stacks are opt-in extras that the code lazy-imports only when used.

---

## The distributions

hugpy is split into independently installable packages with one owner per
concern. Arrows are Python import direction; HTTP between deployed services
does not create a dependency.

| Distribution | Import name | Owns |
|---|---|---|
| `hugpy-platform` | `hugpy_platform` | stdlib-first foundation: central URL, env values, app dirs, hardware probes, pydantic shim |
| `hugpy-control` | `hugpy_control` | control-plane records: bus, jobs, principals, settings |
| `hugpy-storage` | `hugpy_storage` | download queue/daemon, HF transport and token, model sync, physical inventory, on-disk layout |
| `hugpy-engine` | `hugpy_engine` | prompt-to-reply path, model registry/classification, allocation, eviction, spill, native llama.cpp engines, task and placement seams |
| `hugpy-media` | `hugpy_media` | embeddings, summaries, keywords, speech, TTS, vision, image generation, document/URL extraction |
| `hugpy-video` | `hugpy_video` | media library and bus, job lifecycle, ffmpeg/synthetic/studio runners |
| `hugpy-oracle` | `hugpy_oracle` | creative planning, model selection, DAG runtime, repair/evaluation, ledgers |
| `hugpy-fleet` | `hugpy_fleet` | central worker registry, peers, enrollment, heartbeats, evictions, doctrine; full/GGUF/phone workers |
| `hugpy-curation` | `hugpy_curation` | discovery dossiers and the search/screen/download/smoke/judge review pipeline |
| `hugpy-ops` | `hugpy_ops` | sentinel, chaos runner, keeper REPL, todo keeper, provisioner |
| `hugpy-discord` | `hugpy_discord` | the Discord bot (an HTTP client of a central) |
| `hugpy-server` | `hugpy_server` | Flask composition root: routes, auth, SSE, API keys, the built web console |
| `hugpy` | `hugpy` | this package: the `hugpy`/`hpy` commands and the install profiles below |

---

## Install

```bash
pip install hugpy                         # base: the commands + hugpy-platform only
```

Add capabilities with extras. Each extra pulls the owning distribution(s) and
their own extras; the profiles compose, so `hugpy[all]` is a superset of
`hugpy[server]` and `hugpy[gpu-worker]`.

| Extra | Pulls | Use it for |
|-------|-------|------------|
| `server` | `hugpy-server`, `hugpy-fleet`, `hugpy-engine[gguf]`, `hugpy-media`, `hugpy-video`, `hugpy-oracle`, `hugpy-curation`, `hugpy-storage`, `hugpy-control`, `hugpy-discord[bot]`, gunicorn/waitress | A central/coordinator box (`hugpy serve`) |
| `worker` | `hugpy-fleet`, `hugpy-engine[gguf]`, `hugpy-storage`, `hugpy-control` | The minimum for `hugpy worker` / `hugpy gguf-worker` |
| `cpu-worker` | `worker` + `native` + `validate` + `hugpy-engine[transformers]` + `hugpy-media[transformers,embed,audio,keywords,extract,vision]` | CPU transformers box |
| `gpu-worker` | `cpu-worker` + `gpu` + `hugpy-engine[finetune]` + `hugpy-media[all]` + `hugpy-video[render]` | GPU box: adds image generation, fine-tuning, video runners |
| `bot` | `hugpy-discord[bot]` (discord.py, python-dotenv) | The Discord bot arm (`hugpy bot`) |
| `ops` | `hugpy-ops` | `hugpy keeper`, `sentinel`, `chaos`, `provision` |
| `media` | `hugpy-media[all]` | The full server-side media toolset |
| `video` | `hugpy-video[render]`, `hugpy-oracle` | Video intelligence and the Oracle |
| `storage` | `hugpy-storage` | `hugpy download`, `hugpy storage …` |
| `gguf` (`llama`) | `hugpy-engine[gguf]` (llama-cpp-python) | In-process GGUF inference |
| `native-engine` | `hugpy-engine` | `hugpy install-engine` (native llama-server / rpc-server) |
| `transformers` | `hugpy-engine[transformers]`, `hugpy-media[transformers]` | Transformers/PyTorch backends |
| `vision`, `embed`, `keywords`, `audio`, `imagegen`, `extract`, `tts` | the matching `hugpy-media[...]` extra | Individual media capabilities |
| `finetune` | `hugpy-engine[finetune]` (peft) | Fine-tuning helpers |
| `gpu` | `pynvml` | Richer GPU probing (else falls back to `nvidia-smi`) |
| `validate` | `pydantic>=2` | Real pydantic validation (omit on Termux) |
| `native` | psutil, pyyaml, numpy, pillow, bcrypt | Compiled deps with no Android wheel |
| `all` | `server` + `gpu-worker` + `ops` + `video` + `media` + `bot` + `native-engine` | Full desktop/server install |

```bash
pip install "hugpy[gguf]"               # local GGUF inference
pip install "hugpy[server]"             # a coordinator that hosts the console + fleet
pip install "hugpy[gpu-worker]"         # a GPU worker box (or: hugpy install-deps)
pip install "hugpy[all]"                # everything
```

**CUDA-accelerated engine** (rebuild `llama-cpp-python` against CUDA):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

Requires **Python ≥ 3.10**.

---

## Quickstart

```bash
# 1. start the console + API (single-operator, no login wall by default)
hugpy serve --host 0.0.0.0 --port 7002

# 2. open the console
#    → http://localhost:7002/

# 3. (optional) provision the native llama.cpp binaries for the serve drivers
hugpy install-engine            # add --cuda for a CUDA build
```

Download a model and chat from the console, or drive it over the API below.

---

## OpenAI-compatible API

`hugpy serve` exposes an OpenAI-compatible surface at both `/v1` and `/api/v1`.

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7002/v1",   # or https://your-hugpy/api/v1
    api_key="hp_your_key_here",            # mint keys in the console
)

resp = client.chat.completions.create(
    model="your-model-key",
    messages=[{"role": "user", "content": "Say hello in one line."}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

### curl

```bash
# list models
curl http://localhost:7002/v1/models -H "Authorization: Bearer hp_your_key_here"

# chat completion (streaming)
curl http://localhost:7002/v1/chat/completions \
  -H "Authorization: Bearer hp_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model-key","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

**API keys.** Keys (`hp_…`) are minted and revoked in the console (or via the
same-origin `/keys` endpoints). A `require_key` flag decides whether `/v1` calls
must present `Authorization: Bearer hp_…`. The `/v1` surface is public/
programmatic; the console's own routes (model management, jobs, workers) live on
the same origin and are governed by the site's auth mode, not `/v1` keys.

Every path is reachable both bare (`/health`) and `/api`-prefixed
(`/api/health`).

---

## Command-line interface

`hugpy` installs two entry points. `hugpy` is the product command; each
subcommand is implemented by the distribution named in the right-hand column
and imported only when you run it. A missing owner answers with the extra to
install, never a traceback.

```
hugpy serve              run the console + API in one process          hugpy-server
hugpy worker             join this machine to a central as a worker    hugpy-fleet
hugpy gguf-worker        GGUF-only worker (llama.cpp, no torch)        hugpy-fleet
hugpy phone-brick        the phone-brick arm                          hugpy-fleet
hugpy bot                run the Discord bot                          hugpy-discord
hugpy download KEY...    pull catalog models into the model store      hugpy-storage
hugpy storage ...        download daemon, sync from a central, status  hugpy-storage
hugpy chat "prompt"      stream a chat from a central (stdlib only)    —
hugpy keeper             terminal keeper REPL                          hugpy-ops
hugpy sentinel|chaos|provision                                         hugpy-ops
hugpy oracle ...         steward self-check, hook installation         hugpy-oracle
hugpy curation ...       dossiers and the review pipeline              hugpy-curation
hugpy video ...          media job bus, self tests                     hugpy-video
hugpy install-engine     download or build the native llama.cpp binaries  hugpy-engine
hugpy reclassify-images  re-stamp image-model tasks                    hugpy-engine
hugpy install-deps       pip install hugpy[gpu-worker|cpu-worker]      —
hugpy version            show installed hugpy distributions            —
```

`hpy` is a stdlib-only fleet browser and emergency-inference client that talks
HTTP to a running central (`HUGPY_URL`, `HUGPY_KEY`):

```
hpy                       overview: workers + warm models
hpy workers | models | where MODEL
hpy call MODEL "prompt"   chat with a model (pin a worker with -w)
hpy ask "prompt"          emergency inference: warm -> on-disk -> catalog, with fallback
```

### `hugpy serve`

```bash
hugpy serve [--host 0.0.0.0] [--port 7002] [--threads 8]
            [--auth open|external] [--origins a.com,b.com] [--debug]
```

Serves via gunicorn on POSIX, waitress on Windows, and falls back to the Flask
dev server if neither is installed.

### `hugpy install-engine`

```bash
hugpy install-engine [--cuda] [--build-from-source] [--tag <release>] [--jobs N] [--force]
```

Fetches a prebuilt `llama-server` / `rpc-server` (or builds from source with
`--build-from-source`, which needs git + cmake).

---

## GPU worker fleet & sharding

Turn any GPU box into capacity for a central instance:

```bash
pip install "hugpy[gpu-worker]"                   # or: hugpy install-deps
hugpy worker --central https://your-hugpy/        # join as a worker
```

For models larger than a single card, hugpy can split a GGUF model **across
machines** over llama.cpp RPC: the central's allocator coordinates a pool of
`rpc-server` backends. This is opt-in (`HUGPY_SHARD_MODELS`) and configured with
`HUGPY_RPC_SERVERS` / `HUGPY_TENSOR_SPLIT` (see [Configuration](#configuration)).
All flags after `worker` are passed straight to the worker agent's own parser
(`hugpy worker --help`).

### Admission, dedicated pools & self-update

- **Admission.** A newly-joined worker is `pending` and serves nothing until an
  operator **Admits** it in the console. Set `HUGPY_WORKER_ENROLL_REQUIRED=1` to
  also require a one-time enrollment token (`WORKER_ENROLL_TOKEN`).
- **Dedicated pools.** Reserve a worker for one app by starting it with
  `WORKER_POOL=<name>`: it then serves *only* requests tagged for that pool.
  Pooled workers eagerly pre-warm their assigned models (`WORKER_PRELOAD`).
- **Self-update.** Workers converge to the central's target version via
  `pip install -U` + re-exec — from PyPI by default, or from central's bundled
  PEP-503 index (`--pkg-index <central>/api/llm/pip/simple`).
- **Even a phone can help.** `hugpy phone-brick` runs ONNX object detection
  across cheap Android phones, and a phone can join the shard pool as a
  `role=rpc` llama.cpp backend (`PHONE_BRICK_RPC=1`).

---

## Discord bot

The bot arm drives a hugpy central over HTTP — it can point at this machine or a
remote central.

```bash
pip install "hugpy[bot]"
hugpy bot --central http://127.0.0.1:7002 --env /path/to/.env
```

The `.env` supplies `DISCORD_TOKEN` and bot settings (or set `HUGPY_BOT_ENV`).
Restrict slash-command sync to one guild with `--guild <id>`.

---

## Configuration

hugpy is configured by environment variables. The most useful:

| Variable | Purpose |
|----------|---------|
| `DEFAULT_ROOT` | Root directory for model weights, manifests, and data |
| `HUGPY_AUTH_MODE` | `open` (default, no login wall) or `external` (front a real auth service) |
| `HUGPY_AUTO_DOWNLOAD` | Auto-fetch missing models on demand |
| `HUGPY_BASE_URL` | Central base URL used by `hugpy bot` / `hugpy chat` / clients |
| `HUGPY_DATA_DIR` / `HUGPY_CONFIG_DIR` / `HUGPY_CACHE_DIR` | Per-OS data/config/cache overrides |
| `HUGPY_ENGINE_DIR` / `HUGPY_ENGINE_TAG` | Native engine location / llama.cpp release tag |
| `HUGPY_N_GPU` / `HUGPY_N_GPU_LAYERS` / `HUGPY_MAIN_GPU` | GPU offload controls |
| `HUGPY_TENSOR_SPLIT` | Per-GPU split for multi-GPU / sharded serving |
| `HUGPY_SHARD_MODELS` | Enable cross-machine GGUF sharding |
| `HUGPY_RPC_SERVERS` / `HUGPY_SHARD_PORT_BASE` | RPC backend pool for sharding |
| `WORKER_POOL` / `WORKER_PRELOAD` | Reserve a worker for a dedicated pool; eager-warm its models |
| `HUGPY_WORKER_ENROLL_REQUIRED` / `WORKER_PKG_INDEX` | Require enrollment tokens; worker self-update index |
| `PHONE_BRICK_RPC` | Let a phone join the shard pool as a `role=rpc` backend |
| `HUGPY_SSE_HEARTBEAT_SECS` / `HUGPY_MAX_CONTINUATIONS` | SSE keepalive; unbounded-continuation clamp |
| `HUGPY_MAX_UPLOAD_MB` | Max upload size for `/uploads` (default 100) |
| `HUGPY_UI_DIST` | Override the path to the built web console |

CLI flags take precedence over the corresponding environment variables (e.g.
`hugpy serve --auth external` sets `HUGPY_AUTH_MODE`).

---

## Storage & paths

Large artifacts (weights, snapshots, caches) live under `DEFAULT_ROOT` (or the
per-OS data dir resolved via `platformdirs`). Point `DEFAULT_ROOT` at a big,
shared volume for server installs rather than your home directory. The built
web console ships inside the `hugpy-server` wheel, so no separate asset hosting
is needed.

---

## Authentication

- **`open`** (default): single-operator instance, no login wall. The `/v1`
  API-key system still gates programmatic access.
- **`external`**: the console authenticates against a separate auth service.
  hugpy includes a same-origin auth proxy so the session cookie stays
  first-party. Toggle with `HUGPY_AUTH_PROXY`; point it at the upstream with
  `HUGPY_AUTH_BASE`.

---

## Platform notes

- **Base install is wheels-only** and runs on Linux, macOS, Windows, and
  Termux/aarch64 (as a coordinator). Heavy stacks are extras.
- **Windows**: served via `waitress` (gunicorn is POSIX-only). The CLI falls
  back to the Flask dev server if neither is present.
- **Android/Termux**: the GGUF engine, OCR and some vision wheels are not
  available; install those extras only on desktop/server.

### Mobile / Termux (Android)

`hugpy[worker]` installs and imports on Termux as a coordinator/worker. The one
historical blocker was `pydantic_core` (pydantic v2's Rust core), which has no
Termux wheel; `hugpy-platform` ships a **pure-Python pydantic fallback** that is
used automatically, so the import succeeds with no Rust toolchain. The shim
covers model construction and serialization but **does not validate types**.
For full validation install `hugpy[validate]` after `pkg install rust binutils`.

---

## Links & license

- **Homepage:** https://hugpy.ai

**License:** Source-Available (see `LICENSE`). Copyright © 2026 putkoff
(hugpy.ai). All rights reserved.
