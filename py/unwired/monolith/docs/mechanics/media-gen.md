<!-- Written by a per-subsystem code-review pass (2026-09-14). Follows _TEMPLATE.md. -->

# Media generation (image/video/audio/vision) — mechanics

**Scope (paths this doc covers):** `managers/comfy/`, `managers/imagegen/`,
`managers/video_gen/`, `managers/vision/`, `managers/whisper_model/`, `managers/tts/`,
`managers/summarizers/`, `managers/embed/`.
**One-liner:** the `Runner` implementations that serve every non-text-generation
inference task — text/image-to-image, text-to-video, image understanding (VLM),
speech-to-text, text-to-speech, summarization, and embeddings/similarity.
**Owner process(es):** loaded by **both** central (`7002_hugpy_api`, via `flask_app`
→ `managers.dispatch.execute_prompt` → `managers.resolvers.resolve`) and the worker
agent (`worker_agent`, same `resolve()`/`execute_prompt` call path — see
[[worker-fleet]]). The shared registration tables build in whichever process imports
them; `resolve()` decides **per request** which process's runner instance actually
executes (§4, §6). Not a daemon itself — no unit runs "media-gen" standalone.

## 1. Purpose & responsibilities

- Supplies one `Runner` class per `(framework, task)` pair in the registration
  tables owned by `managers/resolvers/categories/` (adjacent, not in this doc's
  scope — see [[serving-core]]) plus the pydantic request/result schemas those
  tables point at.
- Deliberately heterogeneous backends, picked per model's real needs, not a house
  style: ComfyUI over HTTP (`comfy`), in-process `diffusers` (`imagegen`), a
  delegating "studio spine" (`video_gen`), in-process `transformers` (`vision`,
  two of `summarizers`' three backends, `embed`), a non-`transformers` library
  filed under the `"transformers"` framework label (`whisper_model` runs
  `openai-whisper`), and an isolated subprocess with its own venv (`tts`'s
  chatterbox "seat").
- Every runner's `run()` catches its own exceptions and returns a typed
  `ok=False` result — a runner is not supposed to let a raw exception escape to
  dispatch (see §8 for the one place that pattern still leaks framework-shaped
  refusals).
- Does **not** decide local-vs-worker placement (`managers.resolvers.resolve`,
  `managers.dispatch`, [[serving-core]]), does **not** expose HTTP routes itself
  (`flask_app/app/routes/{ml,prompt}_routes.py`, [[api-routes]]), and — for TTS —
  does **not** own the actual synthesis code (lives in
  `video_intel/runners/tts_chatterbox.py`, [[video-oracle]]).

## 2. Key modules (file → responsibility)

| file | responsibility |
|---|---|
| `managers/comfy/comfy_runner.py` | `ComfyRunner` — drives a worker-local ComfyUI over its HTTP API; vanilla t2i/i2i graphs + an IPAdapter "id-lock" graph |
| `managers/imagegen/imagegen_runner.py` | `ImageGenRunner`/`Img2ImgRunner` — in-process `diffusers` pipelines; priced quantization + CPU-offload placement |
| `managers/imagegen/vram_retry.py` | shared one-shot VRAM/OOM-class retry classifier (used by comfy **and** both diffusers runners) |
| `managers/imagegen/schemas.py` | `ImageGenRequest`/`ImageGenResult`/`GeneratedImage` — shared by the comfy **and** diffusers image paths |
| `managers/video_gen/video_gen_runner.py` | `StudioVideoRunner` — lifts a request into the studio `render_clip` spine |
| `managers/video_gen/schemas.py` | `VideoGenRequest`/`VideoGenResult` |
| `managers/vision/vision_runner.py` | `VisionRunner` — thin shim onto a pluggable backend |
| `managers/vision/vision_backends.py` | in-process-vs-HTTP backend registry (`register_backend`) |
| `managers/vision/vision_coder.py` | `VisionCoder` — the real `transformers AutoModelForImageTextToText` loader + generate loop |
| `managers/vision/schemas.py` | `VisionRequest`/`VisionResult`/`VisionBackendConfig`/`VisionCoderConfig` |
| `managers/whisper_model/src/runner.py` | `WhisperRunner` |
| `managers/whisper_model/src/model/model.py` | `whisperManager` — **`openai-whisper`** `load_model()` singleton (not a transformers pipeline) |
| `managers/whisper_model/src/model/execute.py` | `transcribe_file_with_workspace` — extract audio, transcribe, save artifacts, optional context-frame capture |
| `managers/tts/tts_runner.py` | `ChatterboxTtsRunner` — dispatch seam + seat resolution; owns no synthesis logic |
| `managers/tts/seat.py` | `resolve()` — in-process-vs-profile-venv decision, TTL-cached |
| `managers/tts/_backend_main.py` | the profile-venv child entry point (JSON-over-stdio protocol) |
| `managers/summarizers/summarizers.py` | 3 backend strategies (`flan`, `seq2seq_chunked`, `pipeline_chunked`) + a preset system |
| `managers/summarizers/summarize_runner.py` | `SummarizeRunner` — picks a strategy from `cfg.primary_task` |
| `managers/embed/embed_runner.py` | `FeatureExtractionRunner` — `sentence-transformers` embed + cosine similarity |

Adjacent, load-bearing but **out of this doc's scope** (see [[serving-core]]):
`managers/resolvers/categories/frameworks.py` (`FRAMEWORK_RUNNERS`),
`managers/resolvers/categories/builders.py` (`MODEL_REQUEST_BUILDERS`),
`managers/resolvers/model_resolver.py` (`resolve()`, `validate_registry()`).

## 3. Entry points

- **`POST /ml/<amenity>`** — fixed named routes (`transcribe`, `summarize`,
  `keywords`, `embed`, `similarity`, `vision`, `imagine`, `depth`, `detect`,
  `classify`, `segment`) registered from the `ML_TASKS` dict
  (`flask_app/app/routes/ml_routes.py:72-87,326-332`).
- **`POST /prompt {"task": ...}`** — the generic passthrough, "the whole dispatch
  surface over HTTP" (`flask_app/app/routes/prompt_routes.py:1-16,78-129`); covers
  `text-to-speech`/`text-to-video`/`image-to-video`, which have **no** `/ml/*`
  amenity (see §8).
- **`GET /ml`** and **`GET /prompt/tasks`** — discovery: per-task dependency
  readiness and the full `KNOWN_TASKS_REGISTRY`/`TASK_DEFAULTS`
  (`ml_routes.py:335-348`, `prompt_routes.py:132-138`).
- **`POST /oracle/route`** — a second HTTP caller that lands on the same
  `execute_prompt`/`normalize_ml_kwargs` path (`ml_routes.py:226-231` comment;
  [[video-oracle]]).
- In-process: anything importing `managers.dispatch.execute_prompt` or
  `managers.resolvers.resolve` directly (e.g. a `video_intel` runner summarizing
  a transcript) — not only HTTP.

## 4. Data flow (the spine)

**Flow A — the universal dispatch spine (every task in this doc goes through this):**
1. `POST /ml/<amenity>` or `POST /prompt`, or an in-process `execute_prompt(**kwargs)`
   call — `ml_routes.py:277-296`, `prompt_routes.py:78-108`.
2. `execute_prompt` calls `managers.resolvers.resolve(prompt_kwargs)` —
   `managers/resolvers/model_resolver.py:262`.
3. `resolve()` picks `model_key` (or the task's default), reads
   `cfg = MODEL_REGISTRY[model_key]`, resolves `(framework, task)` —
   `model_resolver.py:270-311`.
4. Looks up `builder = MODEL_REQUEST_BUILDERS[(framework, task)]` and
   `local_runner_cls = FRAMEWORK_RUNNERS[(framework, task)]`; either missing raises
   `KeyError("No request builder for …")` / `KeyError("No runner for …")` — **this
   is the "no runner registered for framework=…" refusal** —
   `model_resolver.py:313-328`.
5. Routing decision (local vs remote), in order: a `_force_local` loop guard →
   local; a `placement.json` pin → `make_peer_runner` (System A, cross-central);
   else `make_delegating_runner(framework, task)` (System B — tries a live,
   capability-matched worker, falls back to local) —
   `model_resolver.py:330-346`, `managers/resolvers/remote.py:2249-2358`.
6. `dispatch._get_or_build_runner(res)` instantiates/caches `runner_cls(res.cfg)`;
   `builder(kwargs, model_key)` builds the typed request —
   `managers/dispatch/dispatch.py:515`.
7. `runner.run(req)` executes (see Flows B/C, or the per-subsystem notes in §2);
   returns a typed `*Result` (`ok`/`error`/`error_code`).
8. A `resolve()`/builder `ValueError`/`KeyError`/`TypeError`/`FileNotFoundError`
   becomes **HTTP 400** with the bare message; a runner-level failure is caught
   *inside* `run()` and returned as `ok=False` (HTTP 200) — `ml_routes.py:295-312`,
   `prompt_routes.py:104-125`.

**Flow B — ComfyUI-backed image generation (answers "how does the worker talk to
ComfyUI"):**
1. `resolve()` picks `(framework="comfy", task="text-to-image"/"image-to-image")`;
   the builder is `_build_imagegen_request` — **reused verbatim** from the
   `transformers` path, no comfy-specific request shape
   (`managers/resolvers/categories/builders.py:402,577-578`).
2. `ComfyRunner.__init__` reads `cfg.filename` as the checkpoint name **inside
   ComfyUI's own `models/checkpoints`** — hugpy holds no weights for comfy models
   (`managers/comfy/comfy_runner.py:322-331`).
3. `_generate()` calls the worker-registered VRAM-headroom eviction hook (no-op if
   unregistered), then builds a vanilla node graph (`_t2i_workflow`/`_i2i_workflow`)
   or, if `reference_images` are present, an IPAdapter "id-lock" graph — probed via
   `GET /object_info/<Class>` first, and refused as data if the node pack is
   missing (never silently degrades to a non-locked image) —
   `comfy_runner.py:103-114,190-212,362-395`.
4. `POST {COMFY_URL}/prompt` submits the workflow (`COMFY_URL` env var, default
   `http://127.0.0.1:8188`); polls `GET /history/<prompt_id>` every ~1s up to
   `COMFY_TIMEOUT_S` (default 600s); `GET /view` fetches each output image's bytes
   — `comfy_runner.py:48-49,116-117,397-448`.
5. On a VRAM/OOM-class failure (matched against `RETRYABLE_VRAM_MARKERS`), **one**
   retry: cancel/interrupt the prior prompt, re-drive the headroom hook, sleep a
   bounded settle delay, resubmit — `comfy_runner.py:469-531`,
   `managers/imagegen/vram_retry.py:46-73`.
6. Images are written under `UPLOADS_HOME/generated/`, optionally base64-encoded
   into the result — `comfy_runner.py:433-456`.

**Flow C — TTS seat resolution (the most distinctive registration/delegation
instance):**
1. `resolve()` picks `(framework="transformers", task="text-to-speech")` →
   `ChatterboxTtsRunner` (`categories/frameworks.py:19`) — the `"transformers"`
   label reflects where the registry stores the weights, not the serving library.
2. `ChatterboxTtsRunner._synthesize` calls `managers.tts.seat.resolve()`: a
   300s-TTL-cached check of whether `chatterbox` is importable **in this
   interpreter** (in-process) or only inside a per-model env-profile venv
   (`managers/serve/profiles.profile_python`, [[serving-core]]) —
   `managers/tts/seat.py:46-47,96-131`.
3. In-process: calls the k98 adapter directly. Profile-venv: `subprocess.run`
   spawns that interpreter on `_backend_main.py`, sends one JSON job on stdin,
   reads one `@@TTS_RESULT@@`-prefixed JSON line back from stdout —
   `managers/tts/tts_runner.py:127-149`, `managers/tts/_backend_main.py:69-99`.
4. Either path calls the **same** adapter functions
   (`video_intel.runners.tts_chatterbox.make_tts`/`.synthesize`) — "one
   implementation, two interpreters" — `tts_runner.py:20-22,63-66,157-158`.
5. `duration_s`/`sample_rate` are read back from the written wav header, never
   trusted from the backend's own claim — `tts_runner.py:24-25,102-109,230`.

## 5. State, persistence & invariants

- No DB tables owned here. State is in-process, class-level singleton caches
  keyed by `model_key`, each guarded by a `threading.Lock` with double-checked
  locking: `ImageGenRunner._PIPELINES`, `Img2ImgRunner._PIPELINES` (a **separate**
  cache — different pipeline class), `FeatureExtractionRunner._MODELS`,
  `summarizers.py`'s per-backend `_PIPELINES`/`_MODELS`, `vision_coder._INSTANCES`.
  `ChatterboxTtsRunner` is explicitly stateless (a fresh checkpoint load per call
  — `tts_runner.py:178-182`).
- Artifacts on disk: images under `UPLOADS_HOME/generated/`; video clips under the
  **shared** content-addressed studio store (`/mnt/llm_storage`, not
  `UPLOADS_HOME` — `managers/video_gen/schemas.py:5-8,56-58`); TTS wavs under
  `DEFAULT_ROOT/video_intel/tts` (`tts_runner.py:75-80`); whisper output is a
  per-job **workspace** directory (`transcript.json`/`.txt`, `frames/`,
  `manifest.json` — `whisper_model/src/model/execute.py:121-203`).
- Invariant: a diffusers pipeline is **not** safe under concurrent `__call__`, so
  per-model generate locks serialize same-model calls while different models still
  run in parallel — `imagegen_runner.py:30-44`.
- Invariant: idle-pipeline eviction bounds each image runner's cache to the single
  model in use (worst case one t2i + one i2i resident) — `imagegen_runner.py:76-126`.
- Invariant: the comfy VRAM retry and the diffusers VRAM retry are each **one
  attempt only**, gated by a positive-marker match, never a blind retry-on-any-error
  — `managers/imagegen/vram_retry.py:63-73`.
- Invariant (doc invariant 12): a TTS `reference_audio` without `authorized=True`
  is refused before any backend loads, never silently downgraded to the default
  voice — `tts_runner.py:27-31`, `managers/tts/schemas.py:46-52`.
- Invariant (doc invariant 11 / k102 rule 1): TTS duration/sample-rate are
  *measured*, never claimed (§4 Flow C step 5).

## 6. Cross-subsystem edges

- → `managers/resolvers/`, `managers/dispatch/`, `managers/spill.py` — the
  registration tables, `resolve()`'s local/peer/delegate routing, and the shared
  GPU/CPU placement seam (`spill.transformers_max_memory`) that `imagegen`,
  `vision_coder`, and two of the `summarizers` backends all read. [[serving-core]]
- → `flask_app/app/routes/{ml_routes,prompt_routes}.py` — the HTTP surface (§3).
  [[api-routes]]
- → `worker_agent/` — `task_capabilities` heartbeat advertisement (drives
  `make_delegating_runner`'s capability gating for vision and comfy id-lock),
  the worker-local ComfyUI install `COMFY_URL` points at, and
  `managers/serve/profiles.py`'s env-profile venvs (TTS's seat). [[worker-fleet]]
- → `video_intel/runners/tts_chatterbox.py` (the actual TTS synthesis backend —
  `managers/tts/` is a dispatch shim only) and `video_intel/runners/studio_i2v.py`
  `render_clip`/`video_intel/studio/job.make_studio_i2v` (the video render spine
  `StudioVideoRunner` delegates into). [[video-oracle]]
- ← `oracle/` — `POST /oracle/route` drives `execute_prompt` through the same
  `normalize_ml_kwargs` path as `/ml/*`; `oracle.catalog` reads
  `EXTERNAL_TASK_RUNNERS` (`model_resolver.py:390-393`) for capability
  declarations. [[video-oracle]]
- Sibling `managers/` directories registered through the **same**
  `FRAMEWORK_RUNNERS`/`MODEL_REQUEST_BUILDERS` mechanism but **not** in this
  doc's scope: `managers/keywords/` (`KeywordRunner`, keyword-extraction),
  `managers/vision_analysis/` (depth/object-detection/image-classification/
  image-segmentation — a *different* directory from `managers/vision/`).

## 7. Key contracts / types

All request/result types are frozen pydantic `BaseModel`s carrying
`request_id`/`model_key`/`pool` plus an `ok`/`error`/`error_code` result shape.
- **`ImageGenRequest`/`ImageGenResult`** (`managers/imagegen/schemas.py`) — shared
  by `comfy` **and** `transformers` image tasks. Carries both diffusers-only knobs
  (`num_inference_steps`, `guidance_scale`) and comfy-only knobs (`sampler_name`,
  `scheduler`, `seed` — "read only by the comfy runner"), plus id-lock fields
  (`reference_images`/`reference_images_b64`, `id_strength`) that every
  non-comfy runner ignores.
- **`VideoGenRequest`/`VideoGenResult`** (`managers/video_gen/schemas.py`) — no
  b64 seam (a clip is shared-storage, not inlined); `frames`/`width`/`height`/
  `duration_s` are *measured* off the produced clip, never the ask.
- **`VisionRequest`/`VisionResult`** (`managers/vision/schemas.py`) — exactly one
  of `image_path`/`image_b64` required (validated in `model_validator`).
- **`TranscribeRequest`/`TranscribeResult`** (`imports/src/schemas/whisper_schemas.py`
  — note: **outside** `managers/whisper_model/`) — `capture_frames`/
  `min_gap_seconds`/`long_segment_seconds` drive an optional video-context-frame
  extraction pass alongside the transcript.
- **`TtsRequest`/`TtsResult`** (`managers/tts/schemas.py`) — `reference_audio` +
  `authorized` is the consent gate (§5); result carries a `sidecar` dict with
  provenance (`weights_source`, `device`, `backend`, …) traveling with the bytes.
- **`SummarizeRequest`/`SummarizeResult`** (`imports/src/schemas/summarizer_schemas.py`
  — also outside `managers/summarizers/`) — `SummaryRequest` (dataclass, concrete
  defaults) is the internal contract every backend consumes; `SummarizeRequest`
  (pydantic, all-Optional) is the wire envelope `summarize()` resolves down to it.
- **`EmbedRequest`/`EmbedResult`** (`managers/embed/embed_runner.py` imports from
  `imports/src/schemas/embeded_schemas.py`) — one request type covers both
  feature-extraction and sentence-similarity; `other_texts` presence is the
  entire dispatch signal.

## 8. Gotchas, tech-debt & review findings

- **△ `managers/falconsai/falconsai_module.py`** (394 lines) is orphaned: not
  imported by `managers/__init__.py` (compare its explicit star-import list,
  `managers/__init__.py:1-12`, which omits `falconsai`), not imported by
  `managers/resolvers/categories/`, not imported by `summarizers.py`. Its
  `PipelineChunkedBackend`-equivalent class carries the same `"Backend: pipeline
  chunked (Falconsai-style, no consolidation)"` comment as
  `summarizers.py:351` (vs `falconsai_module.py:298`) — it looks superseded by
  `summarizers.py`'s generic `pipeline_chunked` backend and left in place.
- **△ `managers/summarizers/generation.py`**'s `GeneratorManager`/`get_generator()`
  (distilgpt2 singleton pipeline) has zero importers anywhere in the tree outside
  its own file — dead code sharing the `summarizers` package.
- **△ `managers/vision/utils.py:16-26`**'s `fit_to_token_budget` is an unused
  near-duplicate: nothing imports `vision/utils.py` (verified — no other module
  references it), and `vision_coder.py` defines and uses its **own**
  `fit_to_token_budget` (`vision_coder.py:76-95`) built from constants
  re-declared a *third* time in `vision/schemas.py:9-10`, rather than importing
  either existing copy.
- **ℹ `WhisperRunner` is registered under `framework="transformers"`
  (`categories/frameworks.py:10`) but actually runs `openai-whisper`**, not a
  transformers pipeline — `whisper_model/src/model/model.py:31`
  (`get_whisper().load_model(...)`) and the constants-file comment explaining the
  transformers-dir-vs-`.pt`-store mismatch (`whisper_model/constants.py:5-13`).
  Independently corroborated by `managers/task_deps.py`'s dependency probe for
  `automatic-speech-recognition`, which checks for module `"whisper"`, not
  `"transformers"` (`task_deps.py:27`). Same "framework label is a store
  convention, not a literal claim" pattern as `ChatterboxTtsRunner`.
- **ℹ `whisper_model/src/stream.py:101`**'s `whisper_transcribe_url_stream` has no
  callers anywhere else in the tree, and despite the name is not streaming ASR —
  it downloads a URL to a temp file, then transcribes synchronously. Looks
  unused/aspirational.
- **ℹ Asymmetric HTTP surface**: `ML_TASKS` (`ml_routes.py:72-87`) has no amenity
  for `text-to-speech`/`text-to-video`/`image-to-video` — those three reach
  `execute_prompt` only via the generic `POST /prompt`, or (for video) the
  cinema-specific `flask_app/app/routes/video_routes.py` blueprint
  (`/video/studio/i2v`, `/video/jobs/generate_*`), unlike every other task here
  which gets a named `/ml/*` route.
- **△ `Img2ImgRunner` is registered but currently unreachable**: fully wired into
  `FRAMEWORK_RUNNERS`/`MODEL_REQUEST_BUILDERS`, but by its own docstring "INERT
  until a model advertises (`transformers`,`image-to-image`)" — the sd-turbo
  advertisement flip is deliberately held back (`imagegen_runner.py:636,649-651`).
- **ℹ Filename typo**: `imports/src/schemas/embeded_schemas.py` ("embeded", not
  "embedded") — cosmetic, but real and citable if anyone greps for the correct
  spelling.

## 9. Deploy/run boundary

Central now runs the **src** checkout (`PYTHONPATH=…/src`) — an edit anywhere
under this doc's scope is live on the next `7002` restart, no wheel rebuild
needed. Workers still run the **installed pip wheel** (`README.md §Deploy loop`),
so a worker-side fix to e.g. `comfy_runner.py` or `tts_runner.py` needs the
publish → `pip install -U` → restart cycle before it reaches `worker_agent`
processes — a central-only edit and a worker-visible edit are on two different
clocks. Two things here are **entirely outside the package deploy loop**:
ComfyUI itself is a separately-installed, separately-started process on the
worker box (`COMFY_URL` just points at wherever it's already running — this repo
ships no ComfyUI code), and the TTS `chatterbox-tts` profile venv is materialized
via `managers/serve/profiles.materialize(...)` independent of the wheel.
