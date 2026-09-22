# hugpy · Video Intelligence UI — Architecture Map

Everything below routes through machinery you already have. Nothing here is a new
control plane — it's five new **JobSpecs**, a handful of frozen schemas, one shared
region-editor contract, and PageSpec-registered stations fed by
`MediaInputReceptacle`/`FieldSpec`. The reuse of `wmtrim`'s parallel-ffmpeg-worker
pattern and `abstract_chartkit`'s `FetchWorkerPool` is deliberate.

---

## 1. Scope — the five capabilities, mapped to subsystems

| # | Capability | Subsystem |
|---|------------|-----------|
| 1 | Route worker calls through the hugpy backend | Job bus + worker control plane (existing) |
| 2 | Comprehensive image cropping station | **spatial** region editor → `CropSpec` |
| 3 | Comprehensive audio cropping station | **temporal** region editor → `CropSpec` |
| 4 | (text ∨ image(s) ∨ video) → image | `GenerateImageSpec` + chain registry |
| 5 | Model / fps / frame-quality selection | `ModelConfig` registry + `FrameExtractSpec` |

The load-bearing observation: **#2 and #3 are the same operation on different axes.**
Image crop = spatial bbox. Audio crop = temporal interval. Video = both. One
`CropSpec`, one worker, one region-editor contract, two renderers. If you let these
diverge into three implementations, that's the code that betrays you in six months.

---

## 2. How it slots into existing hugpy primitives

| New thing | Extends / reuses |
|-----------|------------------|
| `FrameExtractJob`, `CropJob`, `AudioExtractJob`, `GenerateImageJob` | your frozen `JobSpec` registry (`hugpy_jobs.py` single-orchestrator pattern) |
| Runner keys `(ffmpeg, frame_extract)` etc. | `(framework, task)` tuple dispatch table |
| `MediaRef` metadata at ingest | layered metadata resolver (local config → tokenizer → Hub), `hugpy.json` markers |
| Model dropdowns in stations | `ModelConfig`-based unified registry |
| Multimodal prompt input | `MediaInputReceptacle` + `FieldSpec` source propagation |
| video→image, video→audio→crop | chain registry (validated pipeline composition) |
| ffmpeg worker fan-out | `wmtrim` parallel ffmpeg workers |

---

## 3. Layered view

```
┌──────────────────────────────────────────────────────────────┐
│  UI  (PageSpec-registered stations, fed by MediaInputReceptacle)│
│                                                                 │
│  [Image Crop]  [Audio Crop]  [Frame/Model Config]  [Generate]  │
│        │  spatial      │  temporal        │  fps/q/model  │     │
│        └──────RegionEditor contract───────┘               │     │
│                        │  builds frozen spec              │     │
└────────────────────────┼─────────────────────────────────┼─────┘
                         ▼                                  ▼
┌──────────────────────────────────────────────────────────────┐
│  BACKEND  — one enqueue endpoint per spec type                 │
│  spec → JobSpec lookup → enqueue(job_id) → return job_id       │
└────────────────────────┬─────────────────────────────────────┘
                         ▼   (queue, NOT callbacks)
┌──────────────────────────────────────────────────────────────┐
│  WORKER CONTROL PLANE                                          │
│  claim → dispatch on (framework, task) → run pure runner →     │
│  write JobResult (single writer)                              │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
                   MediaStore  (bytes)  +  JobStore (state)
                         ▲
                         │  UI subscribes by job_id (SSE/poll)
```

UI never touches ffmpeg or a model directly. It builds a frozen spec, enqueues, and
reacts to state transitions. That is the whole "route worker calls through the backend"
requirement.

---

## 4. Core schemas

All frozen. All construction goes through factories so validation is local (a raise at
construction is fine — it never crosses a module boundary; boundary results are data,
see §6). `os.path` only — no pathlib.

### 4.1 Media substrate

```python
# media_schema.py
from dataclasses import dataclass
from typing import Optional, Literal

MediaKind = Literal["image", "audio", "video"]

@dataclass(frozen=True)
class MediaRef:
    """Immutable handle to an asset in the MediaStore. Metadata resolved ONCE at
    ingest via the layered resolver; never mutated afterward. Single source of
    truth for 'what is this asset'."""
    asset_id: str
    kind: MediaKind
    uri: str                       # store key / local path (os.path-built)
    mime: str
    # descriptive, resolved at ingest:
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None
    fps_native: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
```

### 4.2 Unified region / crop

```python
# crop_schema.py
from dataclasses import dataclass
from typing import Optional
from media_schema import MediaRef

@dataclass(frozen=True)
class SpatialRegion:
    """Pixel-space bbox, origin top-left, half-open on the far edges."""
    x: int
    y: int
    w: int
    h: int

@dataclass(frozen=True)
class TemporalRegion:
    """Half-open interval [start_s, end_s) in seconds."""
    start_s: float
    end_s: float

@dataclass(frozen=True)
class CropSpec:
    """A crop is a region over one or both axes.
        image → spatial only
        audio → temporal only
        video → spatial and/or temporal
    Exactly-one-axis validity is enforced per source.kind in the factory."""
    source: MediaRef
    spatial: Optional[SpatialRegion] = None
    temporal: Optional[TemporalRegion] = None

def make_crop(source: MediaRef,
              spatial: Optional[SpatialRegion] = None,
              temporal: Optional[TemporalRegion] = None) -> CropSpec:
    if source.kind == "image" and temporal is not None:
        raise ValueError("image crop cannot carry a temporal region")
    if source.kind == "audio" and spatial is not None:
        raise ValueError("audio crop cannot carry a spatial region")
    if spatial is None and temporal is None:
        raise ValueError("crop must specify at least one axis")
    return CropSpec(source=source, spatial=spatial, temporal=temporal)
```

### 4.3 Frame extraction — explicit knobs, loud caps

```python
# frame_schema.py
from dataclasses import dataclass
from typing import Optional, Literal
from media_schema import MediaRef
from crop_schema import TemporalRegion

FrameFmt = Literal["jpg", "png", "webp"]

@dataclass(frozen=True)
class FrameExtractSpec:
    source: MediaRef                      # kind must be "video"
    fps: float                            # target sampling rate (not native)
    quality: int                          # interpreted per fmt by the runner*
    fmt: FrameFmt
    window: Optional[TemporalRegion] = None   # optional time slice of the video
    max_frames: Optional[int] = None          # loud cap; runner refuses to exceed
```

\*`quality` semantics differ by codec (jpg/webp 1–100, png 0–9). The runner maps it;
the station surfaces the valid range per selected `fmt`. Flagged, not hidden.

Volume note: 10 min @ 5 fps = 3000 frames. `max_frames` is a hard stop, frames land
in the store as `MediaRef`s (referenced, never inlined), and the result grid is
virtualized/paginated.

### 4.4 Generation — ordered multimodal prompt

```python
# gen_schema.py
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
from media_schema import MediaRef

PartKind = Literal["text", "image", "video"]

@dataclass(frozen=True)
class GenPromptPart:
    kind: PartKind
    text: Optional[str] = None
    media: Optional[MediaRef] = None      # exactly one of text/media populated

def text_part(s: str) -> GenPromptPart:
    return GenPromptPart(kind="text", text=s)

def image_part(ref: MediaRef) -> GenPromptPart:
    assert ref.kind == "image"
    return GenPromptPart(kind="image", media=ref)

def video_part(ref: MediaRef) -> GenPromptPart:
    assert ref.kind == "video"
    return GenPromptPart(kind="video", media=ref)

@dataclass(frozen=True)
class GenerateImageSpec:
    parts: Tuple[GenPromptPart, ...]      # ordered
    model_id: str                         # key into ModelConfig registry; loud on miss
    width: int
    height: int
    steps: int
    guidance: float
    seed: Optional[int] = None
    negative: Optional[str] = None
```

**Video as a generation input does not go straight into a diffusion runner.** Image-gen
models eat images, not video. So a `video` part is resolved by the **chain registry**:

```
video_part ─▶ FrameExtractJob (fps/quality) ─▶ frame selection ─▶ image conditioning
```

Encode that as a registered chain, not as special-casing inside the diffusion runner.
The only exception is a model whose `ModelConfig` explicitly declares native video
conditioning — then the runner takes the video part directly. Registry decides; the
runner never guesses.

---

## 5. Job envelope + registry

```python
# job_schema.py
from dataclasses import dataclass, field
from typing import Optional, Tuple, Type
from media_schema import MediaRef

@dataclass(frozen=True)
class JobSpec:
    name: str
    spec_type: Type                       # one of the frozen specs above
    runner_key: Tuple[str, str]           # (framework, task)
    queue: str
    timeout_s: int

# registry, not globals
JOB_REGISTRY = {
    "frame_extract": JobSpec("frame_extract", FrameExtractSpec,
                             ("ffmpeg", "frame_extract"), "media", 600),
    "audio_extract": JobSpec("audio_extract", AudioExtractSpec,
                             ("ffmpeg", "audio_extract"), "media", 300),
    "crop":          JobSpec("crop", CropSpec,
                             ("ffmpeg", "crop"), "media", 300),  # runner branches spatial/temporal
    "generate_image":JobSpec("generate_image", GenerateImageSpec,
                             ("diffusers", "generate_image"), "gpu", 900),
}
```

`AudioExtractSpec` is the trivial sibling of `FrameExtractSpec` (video → audio track as
a `MediaRef`); the audio station consumes its output.

---

## 6. State + routing (queues, single writer, errors as data)

State machine (your job/message state schema owns this):

```
queued ─▶ claimed ─▶ running ─┬─▶ done
                             └─▶ failed
```

```python
# result_schema.py
from dataclasses import dataclass, field
from typing import Optional, Tuple
from media_schema import MediaRef

@dataclass(frozen=True)
class JobError:
    code: str
    message: str
    retryable: bool

@dataclass(frozen=True)
class JobResult:
    job_id: str
    ok: bool
    outputs: Tuple[MediaRef, ...] = ()    # frame_extract → many; crop/gen → one
    error: Optional[JobError] = None      # data across the boundary, never a raise
```

- **Enqueue path:** `spec → JOB_REGISTRY[name] → enqueue(job_id, spec) → return job_id`.
- **Worker path:** `claim → dispatch[runner_key](spec) → JobResult → write once`.
- **Runners are pure** `spec → JobResult` (same discipline as your PyPI work-scripts).
  Failures return `JobError`; only the worker loop is allowed to catch/convert.
- **Single writer** for each `job_id`'s state (the claiming worker). UI is read-only on
  state, subscribing by `job_id` (SSE stream or poll). No callbacks.

Dispatch table:

| runner_key | does |
|------------|------|
| `(ffmpeg, frame_extract)` | video → N frames (fps/quality/fmt/window/cap) |
| `(ffmpeg, audio_extract)` | video → audio `MediaRef` |
| `(ffmpeg, crop)` | spatial (pillow/ffmpeg) or temporal (trim) branch on axes present |
| `(diffusers, generate_image)` | resolved parts → image |

---

## 7. UI stations

All four are PageSpec capabilities. Inputs arrive through `MediaInputReceptacle`;
`FieldSpec` propagates any produced `MediaRef` (a cropped frame, an extracted audio
clip) as an available input to downstream stations. That's what closes the loop:
**crop a frame → it appears as a selectable image input in the Generate station.**

### Shared: `RegionEditor` contract (one abstraction, two renderers)

```
props: value | values[]      (SpatialRegion | TemporalRegion)
       onChange
       presets[]             (1:1, 16:9, model-native 1024²  |  1s/5s/beat)
       numericInputs         (x/y/w/h  |  start/end or start+dur)
       multiRegion           (queue several crops/clips off one source)
```

- **SpatialRegionEditor** — canvas, draggable/resizable bbox, aspect lock, zoom/pan,
  snap-to-grid, numeric x/y/w/h, model-native presets, multi-crop queue, live preview.
- **TemporalRegionEditor** — waveform, draggable region handles, timeline zoom,
  scrub + region-loop playback, numeric start/end, spectrogram toggle, multi-region.
  Waveform is the temporal analog of the bbox.

### Image Crop station
`MediaRef(image)` → SpatialRegionEditor → one/many `CropSpec` → enqueue `crop`.

### Audio Crop station
Source is either an uploaded `MediaRef(audio)` **or** a `MediaRef(video)`. For video,
run `audio_extract` first (small chain), then TemporalRegionEditor over the resulting
waveform → `CropSpec(temporal)` → enqueue `crop`.

### Frame / Model Config station
Not really a separate page so much as the config surface over video: `fps`, `quality`,
`fmt`, `window`, `max_frames`, plus the **model dropdown sourced from the `ModelConfig`
registry filtered by task**. Produces `FrameExtractSpec`; the frame grid is the result.

### Generate station
`MediaInputReceptacle` collects an ordered prompt: text + image parts (including cropped
frames handed over via `FieldSpec`) + optional video part. Model dropdown filtered to
`text2image`/`image2image` (and native-video models if any). Builds `GenerateImageSpec`;
video parts resolved by the chain registry before hitting the runner.

---

## 8. Registries in play

- **`JOB_REGISTRY`** — spec type → `JobSpec` (queue, timeout, runner_key).
- **runner dispatch** — `(framework, task)` → pure runner fn.
- **`ModelConfig` registry** — model_id → config, filtered by task for dropdowns.
- **chain registry** — `video→frame→condition`, `video→audio→crop` as validated chains.
- **PageSpec registry** — the four stations as capabilities.

Registries over globals, top to bottom.

---

## 9. Key decisions (the ones that keep you un-betrayed)

1. **One `CropSpec`, spatial/temporal as orthogonal optional axes.** Kills three-way
   crop divergence. One worker, one editor contract.
2. **`MediaRef` is immutable, metadata resolved once at ingest.** Single source of truth
   per asset; no station recomputes dimensions or duration.
3. **Everything is a job through the bus; runners are pure; errors are `JobError` data.**
   UI is read-only on state. Matches your PyPI work-script discipline.
4. **Video-as-generation-input decomposes via the chain registry, not runner special-casing.**
   The diffusion runner only ever sees images (unless a `ModelConfig` opts into video).
5. **Explicit knobs, loud caps.** `fps`/`quality`/`fmt`/`max_frames` are required and
   surfaced; no smart defaults silently picking 24fps or dumping 10k frames.

---

## 10. Open questions (with my default if you don't override)

1. **What consumes cropped audio?** The requested generation path is
   (text∨image∨video)→image — audio isn't a generation input. So the audio station
   currently produces clean clips for *something* (ASR? dataset building? re-mux?).
   *Default:* treat cropped audio as a terminal artifact (`MediaRef(audio)` in the
   store) plus an optional `transcribe` chain hook, and revisit when a consumer is real.
2. **Any native-video-conditioning models in the fleet,** or is video→image always
   frame-extract→condition? *Default:* always decompose; add a `ModelConfig` flag the
   day a native model shows up.
3. **`quality` UX** — one field reinterpreted per `fmt`, or three named fields?
   *Default:* one `quality` int, range + label driven by selected `fmt`.
4. **Frame selection in the video→gen chain** — first frame, uniform-N, or manual pick
   from the grid? *Default:* manual pick from the (already-built) frame grid, with
   uniform-N as a fallback.

---

## 11. Suggested build order

1. `MediaRef` + MediaStore ingest (metadata via existing resolver).
2. `CropSpec` + `(ffmpeg, crop)` runner + `crop` JobSpec — headless, testable.
3. `RegionEditor` contract → SpatialRegionEditor → **Image Crop station** end-to-end
   (proves the enqueue → job → subscribe loop).
4. `FrameExtractSpec` + `(ffmpeg, frame_extract)` + config station + frame grid.
5. `audio_extract` + TemporalRegionEditor → **Audio Crop station**.
6. `GenerateImageSpec` + `(diffusers, generate_image)` + Generate station (image/text
   parts first).
7. Register the `video→frame→condition` chain; wire video parts into Generate.

Steps 1–3 give you the entire backbone (schema → job → worker → subscribe) on the
cheapest capability; everything after is another spec + runner + station on the same
rails.

---

## 12. As-built notes (Phases 1–3, 2026-07-02)

The backbone shipped and is live at `https://dev.hugpy.ai/video/image-crop`. Where the
build realized or refined this map:

- **Backend package** `abstract_hugpy_dev/video_intel/` (new, additive; edits no
  existing file except to register the blueprint + start the daemon). Modules:
  `media_schema`, `media_store`, `crop_schema`, `result_schema`, `job_schema`
  (`JOB_REGISTRY`), `runners/ffmpeg_crop` (+ `runners.DISPATCH`), `media_bus`.
  Frontend: `src/regions/` (contract + `SpatialRegionEditor`), `src/video/contract.ts`,
  `src/stations/ImageCropStation.tsx` + `useCropJobs.ts`.

- **Media metadata resolver = ffprobe, single-shot at ingest.** §4.1's "layered
  resolver (local config → tokenizer → Hub)" is the *model* metadata resolver; for
  media assets the faithful realization is one `ffprobe -show_format -show_streams`
  call. `kind` is decided authoritatively from the probed stream set.

- **The job bus was built fresh.** The pre-existing `comms/jobs.py` `job_store` is a
  *lifecycle tracker* (create/update/finish/get, SQLite mirror) — it has no
  enqueue/claim/dispatch loop and nowhere to store `outputs`. So `media_bus.py`
  implements §5/§6 directly: durable **stdlib `sqlite3` (WAL)** at
  `$DEFAULT_ROOT/video_intel/media_jobs.db`, state machine
  `queued → claimed → running → done|failed`, **atomic claim** (`BEGIN IMMEDIATE`)
  giving single-writer-per-job across processes, `JobResult` written once. `work_once()`
  is the headless/test driver; `start_worker_daemon()` is the production loop.

- **Multi-process reality:** gunicorn runs **3 worker processes**. Each starts one
  daemon; atomic claim ⇒ exactly one runs a given job; polls resolve from any process
  because state+results are in the shared SQLite store. This is why the store is durable
  and shared rather than in-process.

- **`run_crop(spec, job_id) -> JobResult`** (job_id passed through so the result carries
  it). Spatial = `-vf crop=w:h:x:y`; temporal = `-ss <start> -t <dur>`; out-of-bounds is
  a pre-flight check returning `JobError(region_out_of_bounds)` (error-as-data). The
  worker loop is the only place that catches *unexpected* raises → `JobError(internal)`.

- **HTTP contract (dual-mounted bare + `/api`, like `worker_bp`):**
  `POST /video/ingest {path}` → MediaRef; `POST /video/jobs/crop {source,spatial?,temporal?}`
  → `{job_id}`; `GET /video/jobs/<job_id>` → `{job_id,status,result}`;
  `GET /video/media?handle=<uri>` → bytes (jailed). **Media bytes are served by
  `handle` (the MediaRef `uri`), not by `asset_id`** — there is no asset_id→uri index
  yet; the UI already holds the uri from ingest and from job outputs.

- **`JOB_REGISTRY` currently registers only `crop`.** The §5 `frame_extract` /
  `audio_extract` / `generate_image` rows are appended as Phases 4–6 add their spec
  types + runners (registering them now would break import).

- **Storage + perms:** uploads under `UPLOADS_HOME`; crop artifacts under
  `$DEFAULT_ROOT/video_intel/crops/`; DB at `$DEFAULT_ROOT/video_intel/media_jobs.db`.
  In prod `DEFAULT_ROOT=/mnt/llm_storage` and gunicorn runs as `ubuntu`, so that tree
  must be writable by `ubuntu` (owned `ubuntu:llmstorage`). A restart of
  `hugpy-api-dev` (from the host via `lxc exec`) is required to activate backend `.py`
  changes; the frontend updates on `npm run build` (webpack `:7001` serves `dist/`
  directly) with an rsync to `console_dist/video/` for the `:7002` surface.

### §12.1 — Phase 4/6/7 additions (as-built 2026-07-03)

- **`JOB_REGISTRY` now registers `crop`, `frame_extract`, `generate_image`**
  (with matching `DISPATCH` + `SPEC_DESERIALIZERS` rows). `audio_extract` remains
  future (Phase 5, deferred).
- **`(ffmpeg, frame_extract)`** (`runners/ffmpeg_frames.py`): one ffmpeg pass with
  the `fps` filter; per-fmt quality mapping; `max_frames` refused LOUDLY up front
  (`JobError code=frame_cap_exceeded`, computed from duration×fps BEFORE spawning
  ffmpeg); each frame ingested to a `MediaRef` via `media_store`. A module-level
  `BoundedSemaphore(1)` bounds heavy fan-out per gunicorn process so a long extract
  can't starve app threads.
- **`(hugpy, generate_image)`** (`runners/imagegen.py`): THIN wrapper over the
  EXISTING inference plane — `managers.dispatch.execute_prompt(task="text-to-image",…)`
  via `_platform.async_runtime.run(...)`. The managers tree is lazy-imported inside
  the runner so the media runners' import path stays decoupled from managers-tree
  health. `pool` deliberately UN-set (the "ml"-pool phantom-central landmine).
  A parallel diffusers stack was expressly NOT built.
- **Chain (Phase 7):** `chains.py resolve_video_parts` (uniform-N frame pick) runs
  in the ENQUEUE path in `video_routes.py` — the generation runner never sees a
  video part. Native-video models would bypass via a ModelConfig flag (none yet).
- **Routes added:** `POST /video/jobs/frame_extract`, `POST /video/jobs/generate_image`
  (same dual bare+`/api` mount).
- **UI:** `FrameExtractStation` (knob rail + paginated frame grid) and
  `GenerateStation` (ordered text/image/video composer, model dropdown from
  `GET /v1/models` filtered to text-to-image, result lands in the library);
  `src/video/mediaLibrary.ts` is the sessionStorage FieldSpec-analog that carries
  produced MediaRefs between stations. Stations `frames` + `generate` now `active`.
- **Selftest isolation gotcha:** `_selftest.py`/`_selftest_gen.py` share the bus DB
  with the LIVE daemon if run with prod `DEFAULT_ROOT` — the daemon wins the claim
  and the test's `work_once()` sees nothing (looks like a hang/`running`). Run gates
  with an isolated `DEFAULT_ROOT` (e.g. `/tmp/vi_gate_$$`).
- **Live e2e evidence (public URL, 2026-07-03):** frame_extract → 6 frames servable
  by handle; cap → `failed/frame_cap_exceeded`; text→image → real 512×512 PNG
  (590 KB) servable; video-part generate → chain → image. Harness:
  `/tmp/vi_e2e.py` (+ `/tmp/vi_e2e_test.mp4`) on the host.

### §12.2 — Generation routing guard (2026-07-03, central-meltdown fix)

Symptom: generate clicks during the GPU worker's warm-up window loaded the
diffusion model onto CENTRAL's CPU inside gunicorn (13 GB RSS, "Device set to
use cpu", box-wide API timeouts — even ffmpeg frame_extract jobs starved).
Cause: `DelegatingRunner` runs LOCAL unconditionally when the worker provider
returns no live worker; the `HUGPY_LOCAL_FALLBACK` gate only covers
selected-then-failed, not assigned-but-still-warming.

Fix (arm-local, `runners/imagegen.py`): when a worker provider IS registered
(a fleet exists) but returns no live worker for `spec.model_id`, the runner
refuses with retryable `JobError code=no_live_gpu_worker` instead of melting
central. Standalone single-box deploys (no provider) keep local generation.
Override: `HUGPY_VIDEOGEN_LOCAL=always`.

Routing evidence when it works: the generated image materializes via the
remote-worker b64 branch (`<job_id>_0.png` under uploads/generated/), NOT the
local plane's `req-*_0.png` naming; central gunicorn RSS stays ~200 MB.

### §12.3 — Phase 5 additions (Audio Crop, as-built 2026-07-03)

- **`JOB_REGISTRY` now also registers `audio_extract`** (with matching
  `runner DISPATCH` `("ffmpeg","audio_extract")` + `media_bus.SPEC_DESERIALIZERS`
  rows). All four job names — crop / frame_extract / generate_image /
  audio_extract — are live.
- **`(ffmpeg, audio_extract)`** (`runners/ffmpeg_audio.py`): a pure runner that
  strips a video's audio to a standalone file — `ffmpeg -y -i <in> -vn -acodec
  <codec> <out>` where `codec` comes from `AudioExtractSpec.fmt` (wav→pcm_s16le,
  mp3→libmp3lame, m4a→aac; default `wav` — the safest for Web Audio
  `decodeAudioData` + `<audio>` playback). Output lands under
  `DEFAULT_ROOT/video_intel/audio/<uuid>.<fmt>` and is re-ingested so its
  duration/sample_rate/channels are authoritative (§9.2). EXPECTED failures are
  JobError DATA: `missing_input`, `not_a_video`, `no_audio_track` (source
  MediaRef has neither sample_rate nor channels), `ffmpeg_failed`,
  `missing_output`. `audio_schema.py` holds the frozen `AudioExtractSpec` +
  `make_audio_extract` factory (also the bus reconstruction path).
- **Temporal audio crop is NOT a new job.** It rides the existing `crop` job:
  the UI enqueues `POST /video/jobs/crop {source: MediaRef(audio), spatial:null,
  temporal:{start_s,end_s}}` and `ffmpeg_crop.py`'s temporal branch (`-ss/-t`)
  produces the clip. This is the §9.1 "one CropSpec, orthogonal axes" decision
  paying off — the audio station added zero crop code.
- **Route added:** `POST /video/jobs/audio_extract` (same dual bare+`/api` mount).
- **UI:** `TemporalRegionEditor` fulfils the RegionEditor contract's temporal
  side — canvas waveform decoded client-side via Web Audio `decodeAudioData` over
  the full `/video/media` byte fetch (no HTTP Range needed; the transport
  `request<T>()` gained an `expect:"arraybuffer"` branch to pull binary through
  the same `Result<T>` layer), a draggable region band with numeric start/end,
  region-loop playback via one `<audio>` element, and a multi-region queue.
  `AudioCropStation` accepts an uploaded audio file directly OR a video (→
  `audio_extract` chain → editor over the resulting track); cropped clips play
  in-station and push to `mediaLibrary`. **Spectrogram toggle was deferred**
  (noted future work) so it couldn't threaten the timeline. Station `audio-crop`
  flipped to `active` — all four stations are now live.
- **Phase 8 poll-cap fix (shipped alongside):** the per-station poll hooks
  previously declared `failed`/"timed out" at a ~60 s wall-clock cap even though
  jobs finish server-side. Now caps live in `stations/pollCaps.ts`
  (generate 300 s — cold GPU loads take minutes; crop/frames/audio 180 s), and on
  cap-expiry a hook flips a `capped` flag (keeping the last real status) instead
  of failing; each hook exposes `resume()`/`resume(key)` and the stations render a
  "Still running on the server — check again" + Check-again affordance. No SSE.
- **Gate tooling + live evidence (public URL, 2026-07-03):** headless
  `video_intel/_selftest_audio.py` (isolated scratch bus DB) proves
  audio_extract → audio temporal crop → `no_audio_track`. Live e2e gate 5 in
  `/tmp/vi_e2e.py` (test asset `/tmp/vi_e2e_audio.mp4`, a lavfi h264+aac clip):
  upload video-with-audio → ingest (kind=video, sample_rate=44100) →
  `audio_extract` (servable 3.02 s WAV) → temporal crop `[0.5,1.5)` (servable
  1.0 s clip). Gates 4/6/7 stayed green across the restart.

### §12.4 — Workbench restructure (Studio sidebar + tabbed studio, 2026-07-03)

Frontend-only UX pass (no backend/contract/job change). Bundle
`index-CCvxOdfx.js` / `index-nlfi_lix.css`, live at the public URL.

- **Two top tabs, not four.** `StationSpec` gained an optional `group?: string`
  field (`types.ts`); `registry.ts` tags `image-crop` / `audio-crop` / `frames`
  as `group:"studio"` and derives `NAV_SECTIONS` — stations sharing a group
  collapse into ONE top-nav section (title from `GROUP_TITLES`), ungrouped ones
  (Generate) keep their own. The shell (`StationShell.tsx`) renders SECTIONS, and
  a section's tab is active when the current route segment is one of its members
  (NavLink's exact-match can't express "active for any studio route", so it's a
  computed class off `NAV_SECTIONS`). The registry stays the single source of
  truth — no station list is hardcoded in the shell.

- **`WorkbenchStation`** (`stations/WorkbenchStation.tsx`) is a
  `[ sidebar | tabbed main ]` layout hosting the three studio stations as
  sub-tabs. It is mounted ONCE for the whole group on a dynamic `/:workbenchTab`
  route (placed AFTER the static `/generate` route, which therefore still wins by
  route-ranking); an unknown/non-studio param redirects to the default studio
  sub-tab. Because one instance stays mounted across sub-tab switches, the
  sidebar is genuinely static. The wrapped station bodies (each still a
  `.station-card`; Frames/Generate keep their `.vi-studio` rail/preview/strip/
  prompt grid) render untouched inside the main column via the same
  active-else-placeholder guard the shell uses.

- **The sidebar is the session media library** (`stations/WorkbenchSidebar.tsx`)
  — a pure VIEW over `src/video/mediaLibrary.ts`, the same session-scoped
  MediaRef store the Generate picker reads. Newest-first (`sort` by `addedAt`
  desc); per row a kind badge + preview (image → lazy `<img src=mediaBytesUrl>`,
  audio → inline `<audio preload=none>`, video/other → placeholder tile) +
  name/dims/duration/mime. Per-item "Use in Generate" is a link to the Generate
  tab (the library is shared, so the item is already selectable in that station's
  in-station picker — no cross-component prompt injection); "Remove" forgets the
  LOCAL entry only. Collapses behind a `.vi-workbench-toggle` at <62rem (pure-CSS
  responsive; a `sidebarOpen` state only drives the narrow case). All new layout
  reuses the `.vi-studio*` token system in `app.css` (`.vi-workbench*`,
  `.vi-subtab*`, `.vi-lib-*`), extended not forked.

- **mediaLibrary subscribe seam.** The store already exposed the live seam —
  `subscribeLibrary(listener)` + `useMediaLibrary()` on `useSyncExternalStore`,
  with a module-level `listeners: Set<()=>void>`, an `emit()` fired by
  `addToLibrary`/`clearLibrary`, and a `getLibrary()` snapshot whose reference
  only changes when the list changes (so React 18 doesn't loop). This pass ADDED
  one method — `removeFromLibrary(uri)` — which drops the single entry by
  `ref.uri`, persists via the existing `writeStorage`, and `emit()`s so the
  sidebar refreshes instantly. Like `clearLibrary`, it forgets the local session
  entry ONLY — it never deletes server media (house policy). The rest of the
  public surface is unchanged, so existing callers (Frames/Generate/audio hooks)
  are untouched.

- **Deploy:** `npm run build` → rsync `dist/` → `console_dist/video/` (additive,
  no `--delete`); superseded `index-*.js|css` moved to
  `console_dist/video/assets_archive/` (the `.claude/` agent-memory dir in
  `assets/` is left untouched). All five routes 200 on the public URL; served
  bundle carries the `vi-workbench` / `Session library` markers and still the
  `pollCaps` "Still running on the server — check again." invariant. SPA users
  must hard-reload to pick up the new hash.

#### §12.4 follow-up — shell-level sidebar + auto-library + composer seam (2026-07-03, bundle `index-h3yot71G.js`)

Frontend-only follow-up (no backend/contract/job change). Same two docroots; JS
hash only changed (`index-CCvxOdfx.js` → `index-h3yot71G.js`; `index-nlfi_lix.css`
unchanged — no CSS edits).

- **Sidebar hoisted to the shell.** `StationShell.tsx` now owns the
  `[ sidebar | station area ]` grid and the `sidebarOpen` toggle state, wrapping
  the `<Routes>` — so the session-library sidebar mounts ONCE above routing and is
  static across every top tab AND studio sub-tab, **including Generate**. It reuses
  the existing `.vi-workbench` / `.vi-workbench-sidebar` / `.vi-workbench-toggle`
  classes and the `@media (max-width:62rem)` collapse verbatim (no CSS change).
  `WorkbenchStation.tsx` de-nested to a Fragment of `<nav .vi-subtabs>` +
  `<div .vi-workbench-body>` rendering into the shell's `.vi-workbench-main`
  column, so there is exactly one sidebar.

- **Auto-library on job completion for ALL artifacts.** Previously only audio
  crops (`useAudioCropJobs`) and generations (`useGenerateJob`) pushed to the
  library; image crops never did and frames only via an explicit selection button.
  This pass adds pushes in the two remaining hooks: `useCropJobs.ts` (image crops,
  origin `image-crop`, label `crop W×H`) and `useFrameJobs.ts` (every produced
  frame, origin `frames`, label `frame N`) — both on the poll `done` branch,
  mirroring the audio-crop hook. `addToLibrary` de-dupes by `ref.uri`, so the
  Frames "Add N to Generate" selection button stays working as a now-redundant
  helper. `useAudioExtractJob.ts` remains intentionally un-pushed (its output is
  the intermediate audio track feeding the temporal editor, not a produced
  artifact).

- **Composer seam for sidebar → Generate.** New `src/video/composerBridge.ts`: a
  module-level single-handler registry (`registerComposer(fn) → unregister`,
  `requestAddPart(part)`; typed `PendingPart { ref, origin, label? }`, no deps).
  `GenerateStation` registers a part-adder in a mount-scoped `useEffect` (dispatch
  by `ref.kind` → image/video `PartEntry`), so exactly one handler exists while
  `/generate` is mounted. `WorkbenchSidebar` route-gates the affordance (first path
  segment `=== "generate"`): on Generate an **"Add to prompt"** accent button calls
  `requestAddPart`; off Generate the existing "Use in Generate" link stays; audio
  items get no prompt action. `requestAddPart` is a safe no-op when Generate isn't
  mounted.

- **Deploy:** rsync additive; ONLY the superseded JS archived by exact name to
  `console_dist/video/assets_archive/` (`.claude/` agent-memory dir untouched). All
  five public routes 200; served bundle carries `Add to prompt` + `Session library`
  + the `Still running on the server — check again.` invariant. SPA users
  hard-reload.

#### §12.4 follow-up — Generate folded into workbench tabs + sitewide navbar (2026-07-03, bundle `index-AyWXgElg.js` / `index-dJJ-NL00.css`)

Frontend-only (no backend/contract/job change). Both docroots; JS hash
`index-h3yot71G.js` → `index-AyWXgElg.js`, CSS `index-nlfi_lix.css` →
`index-dJJ-NL00.css` (navbar/brandmark styles are new).

- **Generate is now a workbench sub-tab.** The registry `generate` entry gains
  `group:"studio"`, so it joins `STUDIO_STATIONS` as the 4th sub-tab and the old
  top-level Studio/Generate split disappears — one tab strip
  [Image Crop | Audio Crop | Frames & Models | Generate] under the persistent shell
  sidebar. `ungrouped` is empty, so the static `/generate` route is gone;
  `/video/generate` resolves via `/:workbenchTab` → `WorkbenchStation` →
  `GenerateStation`. Deep-link contract unchanged; the `"generate"`-segment gate in
  `WorkbenchSidebar` still lights "Add to prompt"; `composerBridge` + job hooks +
  `pollCaps` untouched. The dead `NAV_SECTIONS`/`GROUP_TITLES` top-nav model was
  retired from `registry.ts`; `group` stays optional in `types.ts`.

- **Sitewide navbar ported.** The `.vi-brand`/`.vi-tabs` header is replaced by a lean
  faithful port of the shared hugpy `Navbar` from the media arm
  (`media_intelligence_ui/src/chat/src/ui/{Navbar,BrandMark}.{tsx,css}` +
  `assets/hugpy-mark.png`) → `src/nav/` + `src/assets/` (COPY, no cross-arm import,
  no new deps). Sits at the top above the workbench; links Docs/Console/Media + a
  current **Video** entry; brand/home via `hugpyConfig.siteUrl` (already exposed by
  this arm's `config.ts`). CSS self-contained (nav color custom-props moved onto
  `.hugpy-navbar`); the `demo/mode` embed helper and the two chat-scope override rules
  were dropped.

- **Deploy:** rsync additive; the superseded JS+CSS pair archived by exact name to
  `console_dist/video/assets_archive/` (the new `hugpy-mark-*.png` and the `.claude/`
  agent-memory dir untouched). All five public routes 200; served bundle carries the
  new `hugpy-navbar` + `inference you own` markers alongside `Add to prompt`,
  `Session library`, and the `Still running on the server — check again.` invariant.
  SPA users hard-reload.

#### §12.4 follow-up — always-expanded stations + viewport-height workbench + mobile hamburger drawer (2026-07-03, bundle `index-DmrNkE-o.js` / `index-CNiekhe8.css`)

Frontend-only (no backend/contract/job change). Both docroots; JS hash
`index-AyWXgElg.js` → `index-DmrNkE-o.js`, CSS `index-dJJ-NL00.css` →
`index-CNiekhe8.css`.

- **Upload no longer gates the station layout.** Each station previously showed a bare
  "Choose …" file button first and mounted its workspace only behind `{source && …}`
  (`{audioSource && …}` for AudioCrop; the whole `.vi-studio` grid for FrameExtract).
  Now the FULL station (rail / preview / strip / action bar) renders on tab entry and the
  center item-view is a **browse + drag-and-drop receptacle** while empty. New shared
  `src/video/DropReceptacle.tsx` is a PURE intake UI (no upload logic): a `.vi-dropzone`
  `role="button"` with a hidden `<input type=file>` (click/Enter/Space → picker) plus
  `dragover/dragleave/drop` handlers that take `dataTransfer.files[0]` and call the
  station's own `onFile` prop. Every station wires its EXISTING handler
  (`onPick`/`onPickVideo`) as `onFile`, so the FormData → `uploadUrl` → `videoIngestUrl`
  → `mediaRefSchema` ingest path is byte-for-byte unchanged — the receptacle only surfaces
  a `File`. One component, four call sites (no copies). Empty-state knobs render visible
  but disabled (Run/Extract inert until a source exists); ImageCrop's result grid stays
  gated on `jobs.length>0` (genuine output, not layout). Generate's main workspace was
  already ungated (text-driven), so only its video-part intake overlay adopted the
  receptacle. Marker string `Drop a file or click to browse`, class `vi-dropzone`.

- **Uniform viewport-dependent heights (flex chain, no magic calc).** The sidebar's old
  `position:sticky; max-height:calc(100dvh-3rem)` is replaced by a `min-height:0`/`flex:1`
  chain so the session-library sidebar AND all four station panels fill one box below the
  navbar+tab-strip and scroll internally, matched in height:
  `.vi-shell{height:100dvh}` → `.vi-main{flex:1;min-height:0;overflow:hidden}` →
  `.vi-workbench{height:100%;min-height:0;align-items:stretch}` →
  `.vi-workbench-main{display:flex;flex-direction:column;min-height:0;height:100%}`, with
  `.vi-subtabs{flex-shrink:0}` as the fixed top row and `.vi-workbench-body{flex:1;
  min-height:0;overflow-y:auto}` as the vertical scroll container; the sidebar itself is
  `height:100%;min-height:0;overflow:auto`. `.vi-studio` gained `min-height:0`; the
  `.vi-frames-grid` thumbnail strip keeps `overflow-x:auto` (horizontal). No tab is taller
  or shorter than the sidebar; page and navbar never scroll.

- **Mobile hamburger drawer (mirrors the media arm).** Source pattern, media arm
  `media_intelligence_ui`: `chat/src/ui/ThreadHeader.tsx` renders a 3-line `menu` glyph
  toggling a `sidebarCollapsed` boolean at the 768px `md:` breakpoint; `chat/src/ui/Sidebar.tsx`
  is an off-canvas drawer (`max-md:fixed inset-y-0 left-0 z-40`, `-translate-x-full` ↔
  `translate-x-0`, `transition-transform`) and `chat/src/main.tsx` renders a
  `fixed inset-0 z-30 bg-black/40 md:hidden` scrim button that closes on tap. Mirrored
  here in hand-written CSS (this arm authors literal `.vi-*` CSS, not Tailwind) at the
  arm's OWN 62rem layout pivot for consistency: the in-flow `.vi-workbench-toggle` button
  is retired and a mobile-only `.vi-hamburger` glyph (aria-label/title "Session library",
  `aria-expanded`) leads the tab strip (directly under the navbar), toggling the same shell
  `sidebarOpen` state. `<62rem` the `.vi-workbench-sidebar` becomes
  `position:fixed;top:0;left:0;height:100dvh;width:min(20rem,85vw);z-index:40;
  transform:translateX(-100%)`, and `.vi-workbench.is-open .vi-workbench-sidebar` →
  `translateX(0)`, with a `.vi-scrim` (`position:fixed;inset:0;z-index:35;
  background:rgba(0,0,0,.45)`). Drawer closes on scrim click, Escape, and tab navigation.
  Desktop keeps the always-visible inline grid sidebar (fixed/transform rules live only in
  the 62rem media block).

- **Deploy:** rsync additive; superseded `index-AyWXgElg.js` + `index-dJJ-NL00.css`
  archived by exact name to `console_dist/video/assets_archive/` (the `.claude/`
  agent-memory dir untouched — no wildcard used). All five public routes 200; live HTML
  references the new hash; served bundle carries the new `Drop a file or click to browse`
  / `vi-dropzone` markers alongside `Add to prompt`, `Session library`, `hugpy-navbar`,
  the four tab ids, and the `Still running on the server — check again.` invariant. SPA
  users hard-reload.

#### §12.4 follow-up — session job tracker (survives tab switches + reloads) + sidebar Processes section (2026-07-03, bundle `index-DpLofb60.js` / `index-uvz1wgvu.css`)

Frontend-only (no backend/contract/job change — the server job bus was already
durable). Both docroots; JS `index-DmrNkE-o.js` → `index-DpLofb60.js`, CSS
`index-CNiekhe8.css` → `index-uvz1wgvu.css`.

- **The problem.** `WorkbenchStation` mounts ONLY the active sub-tab; others unmount
  on switch. Each station's job hook owned a private `setInterval` + local JobCard
  state, so leaving a tab killed its polling and JobCards — the user returned to an
  apparently idle tab while the durable bus kept working unobserved, and outputs never
  reached the library because the unmounted hook's `done`-branch never fired.

- **`src/video/jobTracker.ts` — new module-level store.** Mirrors `mediaLibrary.ts`
  exactly (module `cache`, `listeners: Set`, `emit()`, one-time `bindStorageOnce`,
  `getSessionId()` sessionStorage write-through, `useSyncExternalStore`). Record:
  `{jobId, kind:"crop"|"frame_extract"|"audio_extract"|"generate_image", station:
  "image-crop"|"audio-crop"|"frames"|"generate", label, enqueuedAt, startedAt,
  status:JobStatus|null, capped, expired, error, outputs:MediaRef[], libLabel, index,
  region}`. Storage key `vi.jobTracker.v1:<sessionId>`. `trackJob(...)` is called by
  each hook right after it POSTs the enqueue and gets a `job_id`.

- **ONE polling loop.** A single `window.setInterval` (800ms, matching the old per-hook
  `POLL_MS`) armed only while ≥1 record is active (has id, not terminal/capped/expired)
  and cleared when none remain; re-armed on `trackJob`/resume/storage-event/init. Each
  tick polls every active record via `request<>(jobStatusUrl(id))` and parses with the
  same per-kind logic lifted from the hooks. Polling runs regardless of which tab (if
  any) is mounted — that is the whole fix.

- **Cap / resume — reuses `pollCaps.ts` verbatim.** `capFor()`: `generate_image` →
  `GENERATE_POLL_CAP_MS` (300000), others → `JOB_POLL_CAP_MS` (180000). On overrun:
  `capped:true`, stop polling THAT record only, keep last status (NOT failed).
  `resumeJob(jobId)` (and `resumeStation`) re-anchor `startedAt=now`, clear `capped`,
  re-arm. The `STILL_RUNNING_MESSAGE` ("Still running on the server — check again.") is
  imported, never re-typed.

- **Rehydrate + expired.** `initTracker()` runs on import: rehydrates from
  sessionStorage, re-anchors non-terminal/non-capped records' poll clock to a fresh
  post-reload window, leaves capped records capped (manual resume), then arms polling —
  fixing reload amnesia. A poll returning ok with `status:null` (server forgot the id)
  sets `expired:true` and stops polling that record (row reads "unknown/expired"); this
  graceful `{job_id,result:null,status:null}` body was confirmed against the live server
  (HTTP 200, no 500). Transport errors stay transient (retry until cap), same as the old
  hooks.

- **Seam.** `subscribeJobTracker(listener)` + `getJobs()` (returns the referentially
  stable `cache`) + `useJobTracker()` = `useSyncExternalStore(subscribeJobTracker,
  getJobs, getJobs)` returning the WHOLE list. No filtering inside getSnapshot (the
  React-18 "getSnapshot should be cached" infinite-loop footgun) — consumers filter with
  `useMemo`.

- **Hooks became thin selectors.** None owns a `setInterval` or job state now. Each keeps
  its enqueue-body construction + POST → `trackJob(...)`, and derives its returned values
  (useMemo over `useJobTracker()` filtered to its station); `resume`→`resumeJob`,
  `reset`→clear pending + dismiss terminal. Public API to every station body is unchanged
  (`useCropJobs`/`useAudioCropJobs` → `{jobs,runCrops,resume,reset}` with `jobs` still the
  same `TrackedJob`/`TrackedAudioJob` shape incl. `key/index/region/output/capped`; the
  three single-job hooks → `{jobId,status,output|frames,error,capped,running,run,resume,
  reset}`). Re-entering a tab now shows in-flight AND recent jobs.

- **Done-push centralized (single site).** `pushOutputs()` in `jobTracker.ts` runs once
  when a record hits `done`: `frame_extract`→`addToLibrary(f,"frames",`frame ${i+1}`)`
  per output; `crop`→`addToLibrary(out, rec.station, rec.libLabel)` (carried `crop W×H` /
  `clip a–bs`); `generate_image`→`addToLibrary(out,"generate","generated")`;
  `audio_extract`→none (intermediate). All five hook-level `addToLibrary` done-pushes
  were removed. A record goes terminal exactly once (then never polled again) and
  `mediaLibrary` de-dups by `ref.uri` → no double-adds. This is what makes outputs land
  in the library even when the producing tab was never re-visited.

- **Sidebar "Processes" section** (`WorkbenchSidebar.tsx`, `WorkbenchProcesses` +
  `ProcessRow`, rendered inside `.vi-lib` ABOVE the library list, visible from ANY tab):
  heading `Active processes` + count; per non-`done` record newest-first a spinner
  (`.vi-proc-spinner`, `@keyframes vi-proc-spin`) while active, station name,
  `.vi-status-*`-colored status, live mm:ss elapsed (a local 1s interval that runs only
  while something is active), label, failed/expired text, and for capped rows the
  verbatim poll-cap copy + a "Check again" button → `resumeJob(jobId)`. Each row is a
  `NavLink to={`/${station}`}`. New `.vi-proc*` classes reuse the `--vi-*` tokens and
  `.vi-status-*` color conventions.

- **Deploy:** rsync additive; superseded `index-DmrNkE-o.js` + `index-CNiekhe8.css`
  archived by exact name to `console_dist/video/assets_archive/` (the `.claude/`
  agent-memory dir untouched — no wildcard). All five public routes 200; live HTML on the
  new hash; served JS carries `vi.jobTracker.v1` / `Active processes` / `vi-proc` plus the
  preserved `Still running on the server — check again.`, `Session library`, `Add to
  prompt`, `hugpy-navbar`, four tab ids; served CSS carries `vi-proc` / `vi-proc-spin`. A
  live `frame_extract` job ran queued→done on the `/api/video/jobs/<id>` seam. SPA users
  hard-reload.

### §12.5 — Scene generation (one query → N-frame consecutive scene + assembled clip, 2026-07-03, bundle `index-mtHG1DJ6.js` / `index-DM-ij-jO.css`)

New backend job `generate_scene` on the existing rails + a Scene mode in the Generate
station. One Generate query yields N temporally-ordered frames and, optionally, an
assembled browser-playable mp4 so "scene" is tangible.

- **Coherence mode shipped — seed + prompt-schedule v1, NOT img2img (investigated
  first).** The managers plane keys runners/builders by `(framework, task)`; the ONLY
  registered image pair is `("transformers","text-to-image")` (`FRAMEWORK_RUNNERS` /
  `MODEL_REQUEST_BUILDERS`). There is no img2img runner/builder, `ImageGenRequest` has no
  init-image field, and no catalog/fleet model advertises an img2img task (`sd-turbo` →
  `["text-to-image"]`; `flux2-klein…` is a text-encoder, not an image task). Three
  independent blockers → true `frame[i+1]=img2img(frame[i])` chaining is not shippable
  now. v1 instead loops N sequential `text-to-image` generations through the SAME plane
  imagegen uses and gets coherence from (a) a fixed base `seed` with a deterministic
  per-frame schedule `seed+i` (when a base seed is given; omitted → random per frame) and
  (b) a per-frame prompt progression: `base + ", frame {i+1} of {n}"`, then the optional
  `motion` template appended with safe `{i}`/`{n}` substitution via `.replace` (never
  `.format`, so stray braces in user text can't `KeyError`). The managers tree was read
  only, never modified.
- **`scene_schema.py` — `GenerateSceneSpec` (frozen).** Fields (non-defaults first):
  `parts: Tuple[GenPromptPart,...]`, `model_id`, `width`, `height`, `steps`, `guidance`,
  `n_frames`, `fps`, `assemble`, then `seed=None`, `motion=None`, `negative=None`.
  `make_generate_scene(...)` reuses imagegen's exact part validation, then raises
  `ValueError` locally for a falsy `model_id`, non-positive `width/height/steps`,
  `n_frames < 1`, `fps < 1`, non-bool `assemble`, and — the loud cap — `n_frames >
  FRAME_CAP` (24) with a message prefixed `frame_cap_exceeded:`. Because the route catches
  `(ValueError, TypeError) → 400`, an over-cap request is refused up front as HTTP 400.
- **`runners/scene.py` — pure `run_generate_scene(spec, job_id) -> JobResult`.** Runs the
  shared GPU guard ONCE before the loop (same model each frame), refuses surviving video
  parts (`unresolved_video_part`) and empty prompt (`no_prompt`), then loops N times
  SEQUENTIALLY calling `execute_prompt` with imagegen's exact kwargs shape (task
  `text-to-image`, `model_key`, `num_inference_steps`, `guidance_scale`, `num_images=1`,
  `return_b64=True`, per-frame `seed`, optional `negative_prompt`), wrapped in
  `_platform.async_runtime.run(...)`. Each frame is materialized to a sequential
  `frame_%05d.png` under `DEFAULT_ROOT/video_intel/scenes/<job_id>/` (copy the plane's
  local path if present, else decode its inline b64) and `ingest()`-ed → N image
  `MediaRef`s. JobError codes it can emit: `no_live_gpu_worker`, `unresolved_video_part`,
  `no_prompt`, `generation_failed` (names the frame), `generation_no_output` (names the
  frame), `scene_assembly_failed`, and defensive `frame_cap_exceeded`.
- **Shared GPU guard refactor.** The local-CPU-refusal guard that was inline in
  `runners/imagegen.py` moved to `runners/_gpu_guard.py` `guard_gpu_worker(model_id,
  job_id) -> Optional[JobResult]` (returns the `no_live_gpu_worker` refusal or `None`),
  reused by both `run_generate_image` and `run_generate_scene`. imagegen's behavior is
  byte-identical (message text unchanged; only `spec.model_id` → the `model_id` param).
- **Assembly.** Module-level `_assemble_scene_mp4(frame_dir, mp4_path, fps)` runs, under a
  private module-level `threading.BoundedSemaphore(1)` (only the ffmpeg step is guarded —
  generation itself is remote-GPU-bound), the arm's FIRST frames→mp4 ffmpeg invocation:
  `ffmpeg -y -framerate <fps> -i frame_%05d.png -vf scale=trunc(iw/2)*2:trunc(ih/2)*2
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart <mp4>` (yuv420p + even-dim pad for
  browser playback; `+faststart` for progressive play). On `rc!=0`/missing file it raises,
  which the runner turns into a `scene_assembly_failed` JobError. When `assemble`, the mp4
  is `ingest()`-ed and appended LAST → `outputs` is `[image×N, video]`.
- **Registries + route.** `JOB_REGISTRY["generate_scene"] = JobSpec("generate_scene",
  GenerateSceneSpec, ("diffusers","generate_scene"), "gpu", 3600)`;
  `DISPATCH[("diffusers","generate_scene")] = run_generate_scene`;
  `SPEC_DESERIALIZERS["generate_scene"] = _generate_scene_from_dict` (mirrors the
  generate_image deserializer). `chains.py` gains an ADDITIVE `resolve_video_parts_scene`
  (mirrors `resolve_video_parts`, reusing `_extract_uniform_frames` + `image_part`) so any
  video prompt part is resolved to frames BEFORE enqueue. Route
  `POST /video/jobs/generate_scene` in `flask_app/app/routes/video_routes.py` mirrors
  `video_generate_image` (per-part `make_media_ref` + `GenPromptPart` →
  `make_generate_scene` → `resolve_video_parts_scene` → `media_bus.enqueue`), auto-mounted
  at both `/video/...` and `/api/video/...`.
- **Frontend Scene mode.** `GenerateStation.tsx` gains an Image | Scene segmented switch;
  both `useGenerateJob` (image) and the new `useGenerateSceneJob` (scene) are always
  mounted (rules-of-hooks), the shared Run row derives per-mode. Scene reuses the composer,
  model dropdown, and width/height/steps/guidance/seed/negative knobs, and adds scene-only
  knobs `n_frames` (1–24, default 6), `motion` (text), `fps` (default 6), `assemble`
  (checkbox, default true). `useGenerateSceneJob` POSTs to
  `hugpyConfig.generateSceneEnqueueUrl` and exposes `outputs: MediaRef[]` (the full array,
  vs imagegen's single `output`). The scene result view renders the frame strip via
  `.vi-frames-grid .vi-studio-strip` (`.vi-scene-frame-cell`/`.vi-scene-frame-img`) plus
  the arm's FIRST `<video controls className="vi-video-player">` when a video output is
  present. All outputs auto-land in the library through the tracker's single done-push.
- **Poll-cap scaling.** `pollCaps.ts` adds `generateScenePollCapMs(nFrames) = 300_000 +
  60_000 * nFrames` (base 300s + 60s/frame); the verbatim `STILL_RUNNING_MESSAGE` ("Still
  running on the server — check again.") is untouched. `jobTracker.ts` adds the
  `generate_scene` kind, carries `nFrames` on the `TrackedJob` record (persisted → survives
  reloads), `capFor(kind, nFrames)` returns the scaled cap for scenes, and `pushOutputs`
  gains a data-driven scene branch (image outputs → `scene frame N`, a trailing video
  output → `scene clip`).
- **Verification.** VM selftests: `_selftest.py` + `_selftest_gen.py` (regression, both
  green after the shared-file edits) + new `_selftest_scene.py` (Part A rails smoke;
  Part B assembles a real mp4 GPU-independently and confirms `ingest().kind=="video"`).
  ONE `hugpy-api-dev` restart, `is-active` active before/after; regression curls
  `/api/version` `/media/` `/video/` `/` all 200. Live e2e GATE 8 (`/tmp/vi_e2e.py`,
  n_frames=3 assemble=true model `sd-turbo`): job `done`, `outputs` kinds
  `[image,image,image,video]`, each servable 200 via `/video/media?handle=`; full e2e
  ALL PASS, no regressions. Remote-GPU proof: `HUGPY_VIDEOGEN_LOCAL` unset in the daemon +
  the guard refuses local, so `done` with real frames ⟹ remote — corroborated by the
  daemon journal (5× `POST http://192.168.1.113:9100/infer 200`, i.e. gate6 + gate7 + 3
  scene frames) and the frames materialized under `scenes/<job_id>/` after the restart.
  Deploy additive; superseded `index-DpLofb60.js` + `index-uvz1wgvu.css` archived by exact
  name to `console_dist/video/assets_archive/` (`.claude/` untouched); served `/video/`
  HTML on the new hash; served JS carries `scene frame` / `scene clip` markers alongside
  the preserved `Still running on the server — check again.`.
- **Future img2img upgrade path (what it needs).** To turn v1 into a true consecutive
  chain: (1) register `("transformers","image-to-image")` (or `("diffusers",...)`) in both
  `FRAMEWORK_RUNNERS` and `MODEL_REQUEST_BUILDERS` with an `AutoPipelineForImage2Image`
  runner; (2) add an init-image field to `ImageGenRequest` and thread it through the
  builder/runner; (3) add the img2img task to a catalog model's `tasks` and assign that
  model to a worker. Then `run_generate_scene` only needs to feed frame `i` as the init
  image + low strength for frame `i+1` — the loop/materialize/assemble scaffolding already
  fits. All three are net-new in the managers tree (off-limits for this slice).

#### §12.5 follow-up — img2img chaining v2 (as-built, STAGED pending a fleet release, 2026-07-03, bundle `index-BwCqe9va.js` / `index-BFC06I-3.css`)

Turns v1 (seed + prompt-schedule) into REAL image conditioning: a scene can start from a
user frame and chain `frame[i+1]=img2img(frame[i])`. All three "Future upgrade path"
blockers above are RESOLVED additively in the managers tree. **The user-facing go-live is
HELD** because the GPU worker `op` runs a PINNED PyPI wheel (`abstract_hugpy_dev==0.1.92`,
bumped today by a concurrent ServeMode release workstream) whose baked-in engine has no
img2img runner — pushing new task code to op is a fleet-wide PyPI release, out of scope for
this slice (the mission's declared stop boundary). So central ships the engine code + the
video wiring + the UI, but does NOT advertise the task; img2img is proven locally and dark
in production until a release.

- **Managers plane (additive, 6 edits, order-sensitive).** (1) `ImageGenRequest`
  (`managers/imagegen/schemas.py`, frozen, `extra="ignore"`) gains optional `image_path`
  (init image — rides the existing `remote._PATH_KEYS` b64 inliner) + `strength`
  (`0..1`). (2) `managers/imagegen/imagegen_runner.py` gains a sibling `Img2ImgRunner`
  mirroring `ImageGenRunner` but on `AutoPipelineForImage2Image`, its OWN
  `_PIPELINES`/`_LOCK` cache, reusing `ImageGenRequest`/`ImageGenResult` (so the remote
  delegating/peer factories work with zero worker-side change); it loads the init image
  (mirrors `VisionAnalysisRunner._load_image`), sets `image=`+`strength=`, keeps the GPU
  guard verbatim, clamps `num_inference_steps` so `int(steps*strength) >= 1` (sd-turbo runs
  1–4 steps → the product can floor to 0 and diffusers raises) with a loud log, and resizes
  the init to the requested `width/height` since diffusers 0.29.2's SD img2img `__call__`
  ignores those (keeps chained frames uniform for the mp4 mux). (3) registry rows
  `("transformers","image-to-image") → Img2ImgRunner` (`resolvers/categories/frameworks.py`)
  + `_build_img2img_request` (`resolvers/categories/builders.py`) which resolves the init
  image via `kwargs.get("image_path") or kwargs.get("file")` — CRITICAL: on an offloaded run
  the worker rematerializes the inlined image under key `file`, not `image_path`. (4)
  `imports/src/constants/categories.py`: `("transformers","image-to-image")` added to
  `RUNNER_PAIRS` — REQUIRED so `derive_model_config_row` doesn't drop the sd-turbo staple
  (and the text-to-image default with it) once the task is advertised; `TASK_DEFAULTS`
  maps `image-to-image → sd-turbo`. **`HF_TASK_TO_TASKS` deliberately does NOT gain
  `image-to-image`** — that keeps model DISCOVERY from auto-advertising unvetted,
  externally-downloaded `pipeline_tag=image-to-image` models (flux / Qwen-Image-Edit) as
  servable img2img (flux img2img is explicitly not shipped — the `…-text-encoder` model is
  not a full pipeline; sd-turbo is the ONLY supported img2img model).
- **The advertisement flip is HELD (step 6).** `imports/config/models/models_config.py`
  keeps `sd-turbo → tasks:["text-to-image"]`; a comment documents the exact one-line go-live
  edit. `validate_registry` (import-time, hard-fails on staples advertising an unregistered
  task) passes because the runner+builder+RUNNER_PAIRS exist; flipping is then safe.
- **`runners/_img2img.py` — `img2img_available(model_id)` probe.** True iff the pair is
  registered AND `resolve` succeeds for `(model_id, image-to-image)` — i.e. the model
  actually advertises it. With step 6 held this returns FALSE on live central, so the
  runner honestly refuses instead of misrouting to op's old wheel. (Local selftest flips the
  advertisement in-process, so the probe returns True there.)
- **`runners/scene.py` v2.** When a start-frame image part is present: unavailable →
  retryable `JobError{code:"image_to_image_unavailable", message:"image-to-image not
  available on the fleet"}` (honest, NOT silent fallback); available + `chain=true` →
  sequential chain (`frame1=img2img(start,strength,base)`, `frame[i+1]=img2img(frame[i],
  strength, base+motion[i])` — each saved output feeds the next); `chain=false` → every
  frame img2img off the START frame (no drift). No image part → v1 seed-schedule, untouched.
  `runners/imagegen.py` v2: a single-generate image part is used as the img2img init when
  available, else it PRESERVES the v1 "image ignored, text-to-image runs" behavior with a
  loud log (no regression). `scene_schema.py` += `strength`(default 0.45)+`chain`(default
  true); `gen_schema.py` += `strength`; `media_bus.py` deserializers + `chains.py` carry
  them (missing → defaults, old payloads tolerated); `video_routes.py` parses them
  (additive/optional).
- **Frontend.** Scene mode gains a `strength` `type="number"` knob (0–1, step 0.05, default
  0.45; reuses `.vi-knob*`, zero new CSS) + a `chain frames` checkbox (default true), both
  in the `{mode==="scene"}` block. `src/video/contract.ts` `GenerateSceneRequest` +=
  `strength:number|null`, `chain:boolean`. The start frame is the FIRST image part already
  in `parts` (the sidebar composerBridge flow) — no new field. Poll cap made chain-aware:
  `generateScenePollCapMs(nFrames, chain)` adds extra per-frame time when chaining
  (sequential img2img is slower); `chain` threaded onto the `TrackedJob` record (persisted)
  → `capFor`. The stale model hint ("image parts … not yet used as conditioning") now
  states the first image conditions Scene generation. Verbatim `STILL_RUNNING_MESSAGE`
  preserved.
- **Verification.** Managers import CLEAN (validate_registry passes, step 6 held, sd-turbo
  intact, `image-to-image ∈ RUNNER_PAIRS`, `∉ HF_TASK_TO_TASKS`). `_selftest_scene.py`
  (the ONLY selftest that exists on the share — the historical `_selftest.py` /
  `_selftest_gen.py` named in the v1 note above do NOT exist; regression is instead a clean
  combined import + v1 scene round-trip + old-payload tolerance): Part A proves the honest
  `image_to_image_unavailable` failure; **Part B RAN a REAL CPU img2img** —
  `execute_prompt(task="image-to-image", model_key="sd-turbo", image_path=…, strength=0.7,
  steps=2, 256×256)` returned a real `GeneratedImage` (≈58–76 KB b64) with sd-turbo on CPU
  (~60–70 s), proving the full path works. ONE `hugpy-api-dev` restart, regression curls
  `/api/version` `/media/` `/video/` `/` all 200. Live e2e GATE 9 (`img2img_scene`: upload a
  distinct PNG → scene n_frames=3 chain=true strength=0.45) asserts the honest
  `image_to_image_unavailable` terminal error (schema/route accept the fields; runner refuses
  when the fleet can't serve). v1 GATE 8 still `done` `[image×3,video]` with remote-GPU proof
  (journal `POST http://192.168.1.113:9100/infer 200` per frame); full e2e ALL PASS, no
  regressions. Frontend deploy additive; superseded `index-mtHG1DJ6.js` + `index-DM-ij-jO.css`
  archived by EXACT name to `console_dist/video/assets_archive/` (`.claude/` untouched);
  served `/video/` HTML on `index-BwCqe9va.js`; served JS carries `Still running on the
  server — check again.`.
- **GO-LIVE RUNBOOK (2 steps, do in order).** (1) Publish `abstract_hugpy_dev` (with these
  changes) to PyPI via the hugpy release pipeline, then converge op — `POST /ops/update
  {"version":"x.y.z"}` THROUGH central (dev.hugpy.ai; op's ops routes trust central's relay,
  no local token), or let the heartbeat auto-converge — then verify op re-advertises, still
  serves text-to-image (regression), AND serves an img2img request e2e. Rollback baseline:
  `video_intel/_op_pip_snapshot_2026-07-03.txt` (op venv freeze — `abstract_hugpy_dev==0.1.92`,
  `diffusers==0.29.2`, `torch==2.7.1+cu126`; note torchvision/torchaudio are CPU builds).
  (2) ONLY after op serves img2img: flip `sd-turbo → tasks:["text-to-image","image-to-image"]`
  (`models_config.py`, the documented one-liner) + restart central. Ordering matters — flip
  AFTER op converges, else central briefly misroutes img2img to the old wheel. `op` is reached
  by SSH pubkey (`solcatcher` `id_rsa` authorized), NOT the recorded fleet password (rotated).

#### §12.5 follow-up — img2img GO-LIVE **EXECUTED**, then displaced by a slot reassignment (2026-07-03 ~13:49 UTC)

The 2-step go-live above was **carried out** by a prior/concurrent run (git `04273d7`
"bump 0.1.95: img2img engine (held), scene chaining…" then `42db1e3` "img2img GO-LIVE:
advertise image-to-image on sd-turbo (fleet at 0.1.95)"). As-executed:
- **Published `abstract_hugpy_dev==0.1.95`** to PyPI (via the `signals/publish-dev.trigger`
  → `hugpy-jobs` → `pyit-jobs` pipeline); op **and** computron converged to 0.1.95 (wheel
  carries `Img2ImgRunner`). PyPI has since advanced to **0.1.98** (concurrent ServeMode /
  ModelPicker workstream: `0.1.96`/`0.1.97`/`0.1.98`), all published from this same tree.
- **Flipped** `sd-turbo → tasks:["text-to-image","image-to-image"]` (the held one-liner,
  now live in `models_config.py`) + restarted `hugpy-api-dev`. Live central runs **0.1.95
  code** (last restart 13:49:03 UTC; NOT restarted since, so 0.1.96–0.1.98 code is on PyPI
  but NOT yet in the running central process). Post-flip a real init-image+strength run
  returned a conditioned `ImageGenResult`; `img2img_available("sd-turbo")` = True.

**CURRENT OPERATIONAL CAVEAT (verified 2026-07-03 ~14:5x UTC) — the go-live is LIVE in
code/config but NOT exercisable, because `sd-turbo` is no longer scheduled on any worker.**
The concurrent ServeMode workstream reassigned the only diffusion-capable worker (`op`,
RTX 3090) to `[Qwen-Image-Bench, Qwen-Image-Edit-2509, Qwen3.6-35B-A3B, Qwen2.5-VL-7B,
flux-klein-q4, sdxl-turbo]` — `sd-turbo` is on NO worker (the RTX 4060 box is GGUF-only,
`supports_gpu_offload:false`; the third worker has `gpu=None`). Empirical proof against
live central: BOTH `sd-turbo` text-to-image AND `sd-turbo` img2img-scene (start-frame +
chain) now terminate `failed / no_live_gpu_worker` — crucially **NOT**
`image_to_image_unavailable`, which confirms the flip is genuinely in effect (the img2img
request passed the `img2img_available` probe and only then hit the GPU guard). The guard
held: central gunicorn RSS stayed ~235 MB, no "Device set to use cpu".

**The user-facing "image-to-image not available on the fleet" report is a DIFFERENT model.**
The daemon journal shows the real traffic is img2img attempts against **`Qwen-Image-Edit-2509`**
and **`flux-klein-q4`** (the models actually loaded on op), which deliberately do NOT
advertise `image-to-image` (only the curated `sd-turbo` does; `HF_TASK_TO_TASKS` omits the
task precisely to keep flux/Qwen-Image-Edit from auto-advertising it) → `runners/imagegen.py`
logs "image-to-image is NOT available on the fleet (model=…); ignoring the init image".

**To make img2img exercisable, ONE of these is needed (both currently out of the img2img
slice's lane):** (a) schedule `sd-turbo` back onto a GPU worker — ServeMode/slot config,
owned by the concurrent workstream, and op has no free slot (loaded 6 models incl. the 48 GB
Qwen-Image-Edit-2509); or (b) curate-advertise `image-to-image` on a model that IS loaded —
`Qwen-Image-Edit-2509` is a genuine image-editing model and the natural candidate, but it
uses its own `QwenImageEditPipeline` (the shared `Img2ImgRunner` is `AutoPipelineForImage2Image`,
built/validated for the SD family only — unvetted for Qwen, likely needs runner work) and a
central restart would also activate the concurrent workstream's un-live 0.1.96–0.1.98 code.
Both are product/scheduling decisions for the user + the ServeMode owner, not a code gate.

## 13. As-built — Active Processes & placement (observability, 2026-07-18)

**What you see.** The **Active** tab (and the studio **Active renders** tab) show
every build that is currently running or waiting, and — new — **WHERE each one
physically runs**. A build's placement line reads like `ae · cuda:0 · P-studio`
(host · gpu · process). A build marked **external** is executing **off the main
console box** — on the ae GPU worker, on ComfyUI, or on the identity-render
service — which is why it can be busy even when the console machine looks idle.

**Phases you'll see on a build:**

- **queued** — accepted, waiting its turn.
- **awaiting capacity** — a heavy render (Wan clip/movie, identity 3D) that can't
  fit the GPU *yet*. It is **not failing** — it holds until the card frees, then
  starts automatically. Hover the chip for the shortfall reason; `jumped N×` means
  N lighter builds ran ahead of it (bounded, so it can't be starved forever).
- **running** — on the card now. Reserved VRAM is shown when a GPU reservation is
  held.
- **cancelling** — you asked it to stop; it stops between frames/segments.

Everything here is **honest**: there is no fake progress bar — a percentage or
stage appears only when the render actually reports one. **Cancel** works on any
in-flight build, including a held (awaiting-capacity) one, exactly like cancelling
a queued job. The same placement badge also appears on media rows in the main
hugpy console's Activity queue. Fed by `GET /video/jobs` (see
`dev/VIDEO-TASK-SEQUENCES.md` §8.1 for the wire shape).
