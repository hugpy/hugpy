// Canned fixtures for the video-intelligence demo (?demo=1).
//
// Every MediaRef here points its `uri` at a HOSTED demo-media URL rather than a
// server store key. That matters because <img>/<video> element loads do NOT go
// through window.fetch (the shim can't intercept them) — so in canned mode
// mediaBytesUrl() passes these absolute URLs straight through and the browser
// loads the hosted bytes directly. See config.ts and demo/mediaBase.ts.
import type { MediaRef } from "../video/contract";
import { takeSet, sectionFor, type SampleSet } from "./sampleSets";
import { SAMPLES2, type Sample2 } from "./sampleData";

// Real sample media dropped by the operator — the sampels/ tree now lives at
// the HOSTED media base (default https://hugpy.ai/demo-media/sampels/; canonical
// copy /var/www/hugpy-media/ on the hugpy.ai VM). Note: the dir is intentionally
// spelled "sampels" — do NOT rename it in the hosted tree.
import { demoMedia } from "./mediaBase";
const demoClipUrl = demoMedia("sampels/scenes/woman.mp4");
const demoCropUrl = demoMedia("sampels/scenes/man.mp4");
const demoAudioUrl = demoMedia("demo-audio.mp3"); // no audio sample provided — kept
const frame1 = demoMedia("sampels/images/forrest/frame_00000.png");
const frame2 = demoMedia("sampels/images/forrest/frame_00001.png");
const frame3 = demoMedia("sampels/images/forrest/frame_00002.png");
const frame4 = demoMedia("sampels/images/forrest/frame_00003.png");
const frame5 = demoMedia("sampels/images/forrest/frame_00004.png");
const frame6 = demoMedia("sampels/images/forrest/frame_00005.png");
const gen1 = demoMedia("sampels/images/shenzo/frame_00000.png");
const gen2 = demoMedia("sampels/images/bycycle/frame_00000.png");
// The forrest scene's assembled clip (single .mp4 in that folder — hashed name).
const forrestSceneUrl = demoMedia("sampels/images/forrest/ff23090611b844a6b85c224f388bd542.mp4");
// The real 2-segment "keeper-matrix" sailboat movie (see movie.json for numbers).
const movieUrl = demoMedia("sampels/studio_movies/keeper-matrix/movie.mp4");
const movieSeg0Url = demoMedia("sampels/studio_movies/keeper-matrix/segment_00/64854c6a21d47ce10b23e884e12de448b25bfb0a375d33238dd0bec8056c076d/clip.mp4");
const movieSeg1Url = demoMedia("sampels/studio_movies/keeper-matrix/segment_01/cc926b44a42b90d7d291a4e50e3859458306c207a17c718086a3466e4049d6c6/clip.mp4");
const movieSeg1BranchUrl = demoMedia("sampels/studio_movies/keeper-matrix/segment_01/branch.png");

// ── the worked example: one ingested clip + derived assets ──────────────────

export const demoClip: MediaRef = {
  asset_id: "demo-clip",
  kind: "video",
  uri: demoClipUrl, // scenes/woman.mp4
  mime: "video/mp4",
  width: 512,
  height: 512,
  duration_s: 4.0,
  fps_native: 6,
  sample_rate: null,
  channels: null,
};

export const demoCroppedClip: MediaRef = {
  asset_id: "demo-clip-cropped",
  kind: "video",
  uri: demoCropUrl, // scenes/man.mp4
  mime: "video/mp4",
  width: 512,
  height: 512,
  duration_s: 3.333333,
  fps_native: 6,
  sample_rate: null,
  channels: null,
};

export const demoAudio: MediaRef = {
  asset_id: "demo-audio",
  kind: "audio",
  uri: demoAudioUrl,
  mime: "audio/mpeg",
  width: null,
  height: null,
  duration_s: 3.0,
  fps_native: null,
  sample_rate: 44100,
  channels: 1,
};

export const demoPoster: MediaRef = {
  asset_id: "demo-poster",
  kind: "image",
  uri: frame1, // images/forrest/frame_00000.png
  mime: "image/png",
  width: 512,
  height: 512,
};

function frameRef(url: string, i: number): MediaRef {
  return {
    asset_id: `demo-frame-${i}`,
    kind: "image",
    uri: url,
    mime: "image/png",
    width: 512,
    height: 512,
  };
}

export const demoFrames: MediaRef[] = [
  frame1,
  frame2,
  frame3,
  frame4,
  frame5,
  frame6,
].map(frameRef);

function genRef(url: string, i: number): MediaRef {
  return {
    asset_id: `demo-gen-${i}`,
    kind: "image",
    uri: url,
    mime: "image/png",
    width: 512,
    height: 512,
  };
}

// Two real generated stills: shenzo/frame_00000 + bycycle/frame_00000.
export const demoGenerated: MediaRef[] = [gen1, gen2].map(genRef);

// The forrest scene's assembled clip — the LAST output of a generate_scene run
// (N frames then the clip). Real ffprobe: 512×512, 6fps, 6 frames, 1.0s.
export const demoSceneClip: MediaRef = {
  asset_id: "demo-scene-clip",
  kind: "video",
  uri: forrestSceneUrl,
  mime: "video/mp4",
  width: 512,
  height: 512,
  duration_s: 1.0,
  fps_native: 6,
  sample_rate: null,
  channels: null,
};

// The assembled keeper-matrix movie — the terminal `movie:` MediaRef + the LAST
// output of a generate_movie run. From movie.json: 832×480, 16fps, 162 frames.
export const demoMovieClip: MediaRef = {
  asset_id: "demo-movie",
  kind: "video",
  uri: movieUrl,
  mime: "video/mp4",
  width: 832,
  height: 480,
  duration_s: 10.125,
  fps_native: 16,
  sample_rate: null,
  channels: null,
};

// The two per-segment clips (used as segment "frames" in the movie scaffolding).
// Both from movie.json: 832×480, 16fps, 81 frames, 5.0625s.
function movieSegClipRef(url: string, i: number): MediaRef {
  return {
    asset_id: `demo-movie-seg-${i}`,
    kind: "video",
    uri: url,
    mime: "video/mp4",
    width: 832,
    height: 480,
    duration_s: 5.0625,
    fps_native: 16,
    sample_rate: null,
    channels: null,
  };
}

export const demoMovieSeg0Clip: MediaRef = movieSegClipRef(movieSeg0Url, 0);
export const demoMovieSeg1Clip: MediaRef = movieSegClipRef(movieSeg1Url, 1);

// The parent branch still for segment 1 (branch.png): 832×480.
export const demoMovieSeg1Branch: MediaRef = {
  asset_id: "demo-movie-seg-1-branch",
  kind: "image",
  uri: movieSeg1BranchUrl,
  mime: "image/png",
  width: 832,
  height: 480,
};

// ── model registry / task defaults (Generate + Frames model dropdowns) ──────

export const V1_MODELS = {
  object: "list",
  data: [
    {
      id: "sd-turbo",
      object: "model",
      created: 0,
      owned_by: "hugpy",
      hub_id: "stabilityai/sd-turbo",
      task: "text-to-image",
      // Dual capability → prompt required, start image OPTIONAL (present ⇒ img2img).
      tasks: ["text-to-image", "image-to-image"],
      context_length: 77,
    },
    {
      id: "sdxl-base",
      object: "model",
      created: 0,
      owned_by: "hugpy",
      hub_id: "stabilityai/stable-diffusion-xl-base-1.0",
      task: "text-to-image",
      // Pure text-to-image → prompt required, NO start-image slot (image "none").
      tasks: ["text-to-image"],
      context_length: 77,
    },
    {
      id: "Qwen-Image-Edit-2509",
      object: "model",
      created: 0,
      owned_by: "hugpy",
      hub_id: "Qwen/Qwen-Image-Edit-2509",
      task: "image-to-image",
      // Edit-only → prompt AND start image REQUIRED (no text-only path).
      tasks: ["image-to-image"],
      context_length: 0,
    },
    {
      id: "Qwen2.5-3B-Instruct-GGUF",
      object: "model",
      created: 0,
      owned_by: "hugpy",
      hub_id: "Qwen/Qwen2.5-3B-Instruct-GGUF",
      task: "text-generation",
      context_length: 32768,
    },
    {
      id: "Qwen2.5-VL-3B-Instruct-GGUF",
      object: "model",
      created: 0,
      owned_by: "hugpy",
      hub_id: "Qwen/Qwen2.5-VL-3B-Instruct-GGUF",
      // A vision-language model — the Movie director's judge (scores each segment).
      task: "image-text-to-text",
      tasks: ["image-text-to-text"],
      context_length: 32768,
    },
  ],
};

export const PROMPT_TASKS = {
  tasks: ["text-to-image", "text-generation"],
  defaults: { "text-to-image": "sd-turbo", "text-generation": "Qwen2.5-3B-Instruct-GGUF" },
};

// ── curated presets (Generate-station "ideal default loads" dropdown) ────────
// The GET /video/presets payload + a canned /video/presets/<id>/apply response, so
// ?demo=1 exercises the picker without a backend. Shapes match the API contract.

export const PRESETS = {
  presets: [
    {
      id: "sd-turbo-square",
      name: "SD-Turbo · fast square",
      description:
        "A single 512² still in 4 steps, guidance-free — the quickest path to an image.",
      mode: "text-to-image",
      model_key: "sd-turbo",
      defaults: {
        strength: 0.45,
        steps: 4,
        guidance: 0,
        width: 512,
        height: 512,
        n_frames: 6,
        fps: 6,
        negative: "",
      },
      recommended: "gpu",
    },
    {
      id: "sd-turbo-scene",
      name: "SD-Turbo · 6-frame scene",
      description:
        "A short chained scene — 6 consecutive frames at 6fps, assembled into an mp4.",
      mode: "edit-chain",
      model_key: "sd-turbo",
      defaults: {
        strength: 0.45,
        steps: 6,
        guidance: 0,
        width: 512,
        height: 512,
        n_frames: 6,
        fps: 6,
        negative: "",
      },
      recommended: "gpu",
    },
    {
      id: "flux-klein-edit",
      name: "flux-klein · image edit",
      description:
        "img2img edit chain — needs a start image; higher strength for bolder edits.",
      mode: "img2img",
      model_key: "sd-turbo",
      defaults: {
        strength: 0.6,
        steps: 8,
        guidance: 2.5,
        width: 768,
        height: 768,
        n_frames: 4,
        fps: 4,
        negative: "blurry, low quality",
      },
      recommended: "gpu",
    },
  ],
};

export function applyPreset(id: string):
  | {
      ok: true;
      worker: { name: string; id: string };
      model_key: string;
      mode: string;
      defaults: (typeof PRESETS.presets)[number]["defaults"];
      warming: true;
    }
  | { ok: false; error: { code: string; message: string } } {
  const preset = PRESETS.presets.find((p) => p.id === id);
  if (!preset) {
    return { ok: false, error: { code: "unknown_preset", message: `No preset "${id}".` } };
  }
  return {
    ok: true,
    worker: { name: "demo-gpu", id: "demo" },
    model_key: preset.model_key,
    mode: preset.mode,
    defaults: preset.defaults,
    warming: true,
  };
}

// ── curated MOVIE TEMPLATES (Movie-tab dropdown) ─────────────────────────────
// The GET /movie/presets payload (a BARE ARRAY — the movie contract, distinct from
// the {presets:[…]} video envelope) + a canned /movie/presets/<id>/apply echo, so
// ?demo=1 exercises the template picker without a backend. Six ship. Each carries
// its WHOLE contiguous goal timeline; picking one drops it into the Movie editor.
export const MOVIE_PRESETS = [
  {
    id: "golden-hour",
    name: "Golden Hour",
    description:
      "A valley from midday to blazing sunset — four contiguous goals, warm light climbing.",
    model_key: "sd-turbo",
    width: 512,
    height: 512,
    steps: 6,
    guidance: 0,
    fps: 8,
    chain: true,
    goals: [
      { start_frame: 0, end_frame: 8, prompt: "wide valley at midday, flat neutral light" },
      { start_frame: 8, end_frame: 16, prompt: "afternoon light warming, long soft shadows" },
      { start_frame: 16, end_frame: 24, prompt: "golden hour, sun low, amber rim light" },
      { start_frame: 24, end_frame: 32, prompt: "sunset, sky ablaze orange and pink" },
    ],
    vision_enabled: false,
    score_threshold: 70,
  },
  {
    id: "four-seasons",
    name: "Four Seasons",
    description:
      "One landscape cycled through spring, summer, autumn, winter — four goals, chained.",
    model_key: "sd-turbo",
    width: 512,
    height: 512,
    steps: 6,
    guidance: 0,
    fps: 6,
    chain: true,
    goals: [
      { start_frame: 0, end_frame: 6, prompt: "green meadow in spring, blossom on the trees" },
      { start_frame: 6, end_frame: 12, prompt: "same meadow in summer, lush and bright" },
      { start_frame: 12, end_frame: 18, prompt: "same meadow in autumn, amber and red leaves" },
      { start_frame: 18, end_frame: 24, prompt: "same meadow in winter, bare trees and snow" },
    ],
    vision_enabled: false,
    score_threshold: 70,
  },
  {
    id: "rose-bloom",
    name: "Rose Bloom",
    description:
      "A single rose opening in macro — tight bud to full bloom over four goals.",
    model_key: "sd-turbo",
    width: 512,
    height: 512,
    steps: 8,
    guidance: 1.5,
    fps: 10,
    chain: true,
    goals: [
      { start_frame: 0, end_frame: 8, prompt: "macro of a tight red rose bud, dewy" },
      { start_frame: 8, end_frame: 16, prompt: "the bud loosening, outer petals parting" },
      { start_frame: 16, end_frame: 24, prompt: "the rose half open, petals unfurling" },
      { start_frame: 24, end_frame: 32, prompt: "the rose in full bloom, petals wide" },
    ],
    vision_enabled: true,
    score_threshold: 75,
  },
  {
    id: "storm-front",
    name: "Storm Front",
    description:
      "A clear sky building into a lightning storm — four goals of gathering weather.",
    model_key: "sd-turbo",
    width: 512,
    height: 512,
    steps: 6,
    guidance: 0,
    fps: 8,
    chain: true,
    goals: [
      { start_frame: 0, end_frame: 8, prompt: "wide plain under a clear blue sky" },
      { start_frame: 8, end_frame: 16, prompt: "clouds building on the horizon, wind rising" },
      { start_frame: 16, end_frame: 24, prompt: "dark storm clouds overhead, rain sheeting down" },
      { start_frame: 24, end_frame: 32, prompt: "lightning splitting the sky over the plain" },
    ],
    vision_enabled: false,
    score_threshold: 70,
  },
  {
    id: "anime-day",
    name: "Anime Day Cycle",
    description:
      "An anime city through dawn, noon, dusk and night — four goals, stylized.",
    model_key: "sd-turbo",
    width: 512,
    height: 512,
    steps: 6,
    guidance: 1,
    fps: 6,
    chain: true,
    goals: [
      { start_frame: 0, end_frame: 6, prompt: "anime city skyline at dawn, soft pink light" },
      { start_frame: 6, end_frame: 12, prompt: "anime city skyline at noon, bright blue sky" },
      { start_frame: 12, end_frame: 18, prompt: "anime city skyline at dusk, orange glow" },
      { start_frame: 18, end_frame: 24, prompt: "anime city skyline at night, neon and stars" },
    ],
    vision_enabled: false,
    score_threshold: 70,
  },
  {
    id: "cosmic-zoom",
    name: "Cosmic Zoom",
    description:
      "A pull-back from a planet's surface out to a spiral galaxy — four goals.",
    model_key: "sd-turbo",
    width: 512,
    height: 512,
    steps: 6,
    guidance: 0,
    fps: 10,
    chain: true,
    goals: [
      { start_frame: 0, end_frame: 8, prompt: "rugged alien planet surface, red rock" },
      { start_frame: 8, end_frame: 16, prompt: "the planet seen from low orbit, curved horizon" },
      { start_frame: 16, end_frame: 24, prompt: "the solar system, the planet a small disc" },
      { start_frame: 24, end_frame: 32, prompt: "a vast spiral galaxy in deep space" },
    ],
    vision_enabled: false,
    score_threshold: 70,
  },
] as const;

/**
 * Canned /movie/presets/<id>/apply — the movie contract echoes the SAME full
 * template object the list returns (or null for an unknown id → the router 404s).
 */
export function applyMoviePreset(
  id: string,
): (typeof MOVIE_PRESETS)[number] | null {
  return MOVIE_PRESETS.find((p) => p.id === id) ?? null;
}

// ── job lifecycle: enqueue → staged polls → done with per-kind outputs ───────

export type JobKind =
  | "crop"
  | "frame_extract"
  | "audio_extract"
  | "generate_image"
  | "generate_scene"
  | "generate_movie"
  | "studio_i2v"
  | "generate_studio_movie";

interface CannedJob {
  kind: JobKind;
  enqueuedAt: number;
  sourceKind: string | null; // MediaRef.kind of the request source, when sent
  // Pre-rendered SAMPLE SET picked at enqueue (generate kinds only) — stored
  // on the job so every poll answers with the SAME render, and so the ✨-armed
  // prompt⇄media pairing survives the queued→running→done lifecycle.
  sample?: SampleSet;
  // 2026-08-13b: RECORDED-ENVELOPE sample (sampleData.ts) — matched against
  // the SUBMITTED prompt at enqueue, so prompt⇄render consistency is
  // structural, not cursor-based. When set, jobStatus replays this envelope
  // verbatim on done (production's own wire shape — drift-proof).
  replay?: Sample2;
}

const jobs = new Map<string, CannedJob>();
let seq = 0;

const norm = (t: string) => t.toLowerCase().replace(/[^a-z0-9 ]+/g, "").replace(/\s+/g, " ").trim();

const SECTION2: Record<string, string> = {
  generate_image: "scene", generate_scene: "scene", generate_movie: "movie",
  studio_i2v: "clip", generate_studio_movie: "cinema",
};

const rr2: Record<string, number> = {};

/** Pick the recorded sample whose prompt matches the submitted text; else
 *  round-robin within the job's section. Matching is normalized substring in
 *  BOTH directions, so an assist-provided prompt (verbatim) and a lightly
 *  edited one both land on their render. */
function pickReplay(kind: JobKind, submitted: string): Sample2 | undefined {
  const section = SECTION2[kind];
  if (!section) return undefined;
  const pool = SAMPLES2.filter((x) => x.kind === section);
  if (!pool.length) return undefined;
  const t = norm(submitted);
  if (t.length > 8) {
    for (const x of pool) {
      for (const pr of x.prompts) {
        const n = norm(pr || "");
        if (n && (n.includes(t) || t.includes(n))) return x;
      }
    }
  }
  const i = (rr2[section] = ((rr2[section] ?? -1) + 1) % pool.length);
  return pool[i];
}

export function enqueueJob(kind: JobKind, sourceKind: string | null,
                           submittedPrompt = ""): string {
  seq += 1;
  const id = `demo-job-${kind}-${seq}`;
  const job: CannedJob = { kind, enqueuedAt: Date.now(), sourceKind };
  // The demo never touches a GPU (operator ruling): generate kinds replay a
  // RECORDED production envelope chosen by the submitted prompt, with the
  // legacy curated sets as fallback for anything unmatched.
  if (kind.startsWith("generate") || kind === "studio_i2v") {
    job.replay = pickReplay(kind, submittedPrompt);
    if (!job.replay) job.sample = takeSet(sectionFor(kind));
  }
  jobs.set(id, job);
  return id;
}

function outputsFor(job: CannedJob): MediaRef[] {
  if (job.sample) return job.sample.outputs;
  switch (job.kind) {
    case "crop":
      // image crop → a still; video crop → the cropped clip
      return job.sourceKind === "image" ? [demoFrames[0]] : [demoCroppedClip];
    case "frame_extract":
      return demoFrames;
    case "audio_extract":
      return [demoAudio];
    case "generate_image":
      return [demoGenerated[0]];
    case "generate_scene":
      // The 6 forrest frames + the assembled forrest scene clip LAST (scene contract).
      return [...demoFrames, demoSceneClip];
    case "generate_movie":
      // Per-segment clips (as segment "frames") + the assembled movie.mp4 LAST
      // (movie contract). The real keeper-matrix movie has 2 segments.
      return [demoMovieSeg0Clip, demoMovieSeg1Clip, demoMovieClip];
    case "studio_i2v":
      // A studio CLIP render: one clip out (the same bundled scene clip).
      return [demoSceneClip];
    case "generate_studio_movie":
      // CINEMA: segment clips + the assembled movie LAST — same contract as
      // generate_movie, which is exactly how the composer reads it.
      return [demoMovieSeg0Clip, demoMovieSeg1Clip, demoMovieClip];
  }
}

const cancelledJobs = new Set<string>();

export function cancelJob(id: string): { job_id: string; status: string | null; cancelled: boolean } {
  if (!jobs.has(id)) return { job_id: id, status: null, cancelled: false };
  cancelledJobs.add(id);
  return { job_id: id, status: "cancelled", cancelled: true };
}

// The image-only outputs (drops any trailing assembled clip) — the frames a live
// gallery fills in, and the `total` the running readout counts toward.
function imageOutputsFor(job: CannedJob): MediaRef[] {
  return outputsFor(job).filter((m) => m.kind === "image");
}

// The live-progress shape the pinned contract's `progress` block carries. Local to
// the demo — the real payload is validated by jobProgressSchema on the UI side.
interface DemoProgress {
  done: number;
  total: number;
  stage: string;
  label: string;
  model: string;
  frames: MediaRef[];
  started_at: number;
  eta_s: number | null;
}

interface DemoProject {
  name: string | null;
  uuid: string;
  dir: string;
}

const PROGRESS_STAGES = ["loading model", "denoising", "decoding", "encoding"];

// Age-derived progress: done climbs 0→total across the run, the completed frames
// accrue one-by-one, the stage walks the pipeline, and eta_s counts down — so
// ?demo=1 exercises the descriptive readout AND the live-fill gallery.
function progressFor(job: CannedJob, age: number, runFor: number): DemoProgress {
  const imgs = imageOutputsFor(job);
  const total = Math.max(1, imgs.length);
  const frac = Math.max(0, Math.min(1, (age - 600) / (runFor - 600)));
  const done = Math.min(total, Math.floor(frac * total));
  const isScene = job.kind === "generate_scene";
  return {
    done,
    total,
    stage: PROGRESS_STAGES[
      Math.min(PROGRESS_STAGES.length - 1, Math.floor(frac * PROGRESS_STAGES.length))
    ],
    label:
      done >= total
        ? "finalizing…"
        : isScene
          ? `generating frame ${done + 1}`
          : "generating image",
    model: "sd-turbo",
    frames: imgs.slice(0, done),
    started_at: Math.floor(job.enqueuedAt / 1000),
    eta_s: Math.max(0, Math.round((runFor - age) / 1000)),
  };
}

// The auto-archive project echoed on a terminal generation result (name set so the
// demo exercises the "Saved to <name> · <dir>" surface).
function projectFor(job: CannedJob): DemoProject {
  const slug =
    job.kind === "generate_movie"
      ? "demo-movie"
      : job.kind === "generate_scene"
        ? "demo-scene"
        : "demo-image";
  return { name: "Demo project", uuid: `demo-proj-${slug}`, dir: `assets/${slug}` };
}

// ── movie job: the NESTED progress blob + the terminal movie ledger ──────────
// The REAL 2-segment "keeper-matrix" sailboat movie (see movie.json). Each goal's
// `frames` are the real per-segment clip(s) so the live grids + per-segment strip
// paint actual media; seg0 0–81, seg1 81–162 (half-open, tiling [0,162)).
const DEMO_MOVIE_GOALS = [
  {
    goal: {
      start_frame: 0,
      end_frame: 81,
      prompt: "a small wooden sailboat on a bright blue lake, gentle waves, sunny day",
    },
    frames: [demoMovieSeg0Clip],
  },
  {
    goal: {
      start_frame: 81,
      end_frame: 162,
      prompt: "the same lake at sunset, boat sailing away",
    },
    frames: [demoMovieSeg1Branch, demoMovieSeg1Clip],
  },
];

interface DemoMovieSegment {
  index: number;
  goal: { start_frame: number; end_frame: number; prompt: string };
  prompt: string;
  attempt: number;
  score: number | null;
  status: string;
  frames: MediaRef[];
}

interface DemoMovieProgress {
  stage: string;
  segment_done: number;
  segment_total: number;
  segments: DemoMovieSegment[];
  current: DemoProgress | null;
  started_at: number;
  eta_s: number | null;
}

// Age-derived NESTED movie progress: segment_done climbs 0→total, each segment
// walks pending→generating→scoring→done with a score badge that fills in, and
// `current` mirrors the ACTIVE segment's per-frame render (fed to progressReadout).
function movieProgressFor(job: CannedJob, age: number, runFor: number): DemoMovieProgress {
  const total = DEMO_MOVIE_GOALS.length;
  const frac = Math.max(0, Math.min(1, (age - 600) / (runFor - 600)));
  const segDone = Math.min(total, Math.floor(frac * total));
  const startedAt = Math.floor(job.enqueuedAt / 1000);
  const eta = Math.max(0, Math.round((runFor - age) / 1000));
  const segments: DemoMovieSegment[] = DEMO_MOVIE_GOALS.map((gm, i) => {
    if (i < segDone)
      return {
        index: i,
        goal: gm.goal,
        prompt: gm.goal.prompt,
        attempt: i === 1 ? 2 : 1,
        score: 74 + i * 6,
        status: "done",
        frames: gm.frames,
      };
    if (i === segDone) {
      const scoring = ((frac * total) % 1) > 0.5;
      return {
        index: i,
        goal: gm.goal,
        prompt: gm.goal.prompt,
        attempt: 1,
        score: scoring ? 63 : null,
        status: scoring ? "scoring" : "generating",
        frames: gm.frames.slice(0, 1),
      };
    }
    return {
      index: i,
      goal: gm.goal,
      prompt: gm.goal.prompt,
      attempt: 0,
      score: null,
      status: "pending",
      frames: [],
    };
  });
  const active = segDone < total ? DEMO_MOVIE_GOALS[segDone] : null;
  const current: DemoProgress | null = active
    ? {
        done: 1,
        total: active.frames.length,
        stage: "denoising",
        label: `segment ${segDone + 1}`,
        model: "sd-turbo",
        frames: active.frames.slice(0, 1),
        started_at: startedAt,
        eta_s: eta,
      }
    : null;
  return {
    stage: segDone >= total ? "assembling" : "generating",
    segment_done: segDone,
    segment_total: total,
    segments,
    current,
    started_at: startedAt,
    eta_s: eta,
  };
}

// The terminal `movie` ledger echoed on a done generate_movie result (mirrors the
// contract: per-segment chosen take / attempts / scores / why + the assembled clip).
function movieResultFor(job: CannedJob): {
  ok: true;
  error: null;
  outputs: MediaRef[];
  project: DemoProject;
  movie: Record<string, unknown>;
} {
  const goals = DEMO_MOVIE_GOALS.map((g) => g.goal);
  return {
    // ok/error are REQUIRED by jobResultSchema — include them so the terminal movie
    // result actually parses (and the finished preview renders) under ?demo=1.
    ok: true,
    error: null,
    outputs: outputsFor(job),
    project: projectFor(job),
    movie: {
      goals,
      drift: { mean: 0.12, max: 0.28 },
      vision_enabled: true,
      score_threshold: 70,
      n_frames_total: 162,
      segments: goals.map((g, i) => ({
        index: i,
        goal: g,
        prompt: g.prompt,
        seed: 1000 + i,
        attempts: i === 1 ? 2 : 1,
        scores: i === 1 ? [62, 81] : [74 + i * 6],
        chosen_take: i === 1 ? 1 : 0,
        status: "done",
        why:
          i === 1
            ? "re-rolled: first take scored below threshold"
            : "passed on the first take",
        mp4: null,
      })),
      movie: demoMovieClip,
    },
  };
}

// Stage the lifecycle on wall-clock so the console's poller sees a believable
// progression: unclaimed → running (with live progress) → done. generate_* takes a
// bit longer, and carries the progress block + a terminal project.
export function jobStatus(id: string): {
  job_id: string;
  status: string | null;
  result: Record<string, unknown> | null;
  progress?: unknown;
} {
  const job = jobs.get(id);
  if (!job) {
    // Belt-and-braces: an id the shim never minted (e.g. a job-tracker record that
    // somehow leaked in from a prior live visit) must NOT resolve to status:null —
    // that drives the sidebar's "server no longer tracks this job — it may have
    // expired" row, which has no place in a hermetic brochure. Answer any unknown
    // id as a CLEAN completed image generation instead, so the row reads as finished
    // work (its output is a bundled still) and never as an error. The primary guard
    // is the demo-namespaced tracker storage key (jobTracker.ts) — this is the
    // second line of defense so no leaked id can ever surface the expired path.
    return {
      job_id: id,
      status: "done",
      result: { outputs: [demoGenerated[0]] },
    };
  }
  if (cancelledJobs.has(id)) {
    return { job_id: id, status: "cancelled",
             result: { error: { message: "cancelled" } } };
  }
  const age = Date.now() - job.enqueuedAt;
  const isGen = job.kind.startsWith("generate");
  const isMovie =
    job.kind === "generate_movie" || job.kind === "generate_studio_movie";
  // A movie is many segments — give it a longer believable run than a scene/image.
  const runFor = isMovie ? 5200 : isGen ? 3500 : 1800;
  // Unclaimed window: the job is enqueued but no worker has picked it up yet. The
  // REAL bus reports this as "queued" (a live, non-terminal status), NOT a null
  // status — and the jobTracker reads a null poll as "the server forgot this job"
  // and marks the record EXPIRED (the "server no longer tracks this job" row). So a
  // freshly-tracked job whose first (immediate) poll lands here must see "queued",
  // or it wrongly expires before it ever runs. (This also matches the contract the
  // sidebar spinner expects for a just-enqueued job.)
  if (age < 600) return { job_id: id, status: "queued", result: null };
  if (age < runFor) {
    return {
      job_id: id,
      status: "running",
      result: null,
      progress: isMovie
        ? movieProgressFor(job, age, runFor)
        : isGen
          ? progressFor(job, age, runFor)
          : null,
    };
  }
  if (job.replay) {
    const result = JSON.parse(
      JSON.stringify(job.replay.result).replace(/@@JOB@@/g, id),
    ) as Record<string, unknown>;
    return { job_id: id, status: "done", result };
  }
  return {
    job_id: id,
    status: "done",
    result: isMovie
      ? movieResultFor(job)
      : {
          outputs: outputsFor(job),
          ...(isGen ? { project: projectFor(job) } : {}),
        },
  };
}
