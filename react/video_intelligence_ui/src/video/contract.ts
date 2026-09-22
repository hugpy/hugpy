// The frozen hugpy video-intelligence backend contract, in one typed module.
// Mirrors hugpy_video_intelligence_map.md — a backend is being built to this in
// parallel, so these shapes are the source of truth the UI serialises to and
// parses from. zod schemas validate the two responses the UI actually reads
// (ingest → MediaRef, job poll → JobRecord) so contract drift surfaces as a clean
// message instead of an undefined-property crash under strictNullChecks-off.
import { z } from "zod";
import type { SpatialRegion, TemporalRegion } from "../regions/types";

// A nullable numeric MediaRef field. The contract says "numbers may be null"; we
// also tolerate the key being absent, and preserve extra unknown keys via
// passthrough so a MediaRef round-trips untouched back into a crop request.
const nullableNumber = z.number().nullable().optional();

export const mediaRefSchema = z
  .object({
    asset_id: z.string(),
    kind: z.string(),
    uri: z.string(),
    mime: z.string(),
    /** For an image, the native pixel dims — the space CropSpec regions live in. */
    width: nullableNumber,
    height: nullableNumber,
    duration_s: nullableNumber,
    fps_native: nullableNumber,
    sample_rate: nullableNumber,
    channels: nullableNumber,
  })
  .passthrough();

/** A resolved media asset. Native `width`/`height` anchor the spatial pixel space. */
export type MediaRef = z.infer<typeof mediaRefSchema>;

/** POST /uploads → the server file handle used to ingest. */
export const uploadResultSchema = z
  .object({
    path: z.string(),
    name: z.string(),
    size: z.number(),
  })
  .passthrough();
export type UploadResult = z.infer<typeof uploadResultSchema>;

/** POST /video/jobs/crop request body. Exactly one axis is set per station today. */
export interface CropJobRequest {
  source: MediaRef;
  spatial: SpatialRegion | null;
  temporal: TemporalRegion | null;
}

/** POST /video/jobs/crop → the enqueued job's id. */
export const enqueueResultSchema = z
  .object({ job_id: z.string() })
  .passthrough();
export type EnqueueResult = z.infer<typeof enqueueResultSchema>;

/** The frame image encodings the frame-extract job accepts. */
export const FRAME_FORMATS = ["jpg", "png", "webp"] as const;
export type FrameFormat = (typeof FRAME_FORMATS)[number];

/**
 * POST /video/jobs/frame_extract request body. ONE job produces MANY frame
 * MediaRefs (polled back via jobRecordSchema → result.outputs, already an array).
 * `window` null = the whole video; `max_frames` null = uncapped. `quality`
 * semantics differ by fmt (jpg/webp 1..100, png 0..9) — the station surfaces the
 * valid range per selected fmt rather than silently clamping.
 */
export interface FrameExtractRequest {
  source: MediaRef;
  fps: number;
  quality: number;
  fmt: FrameFormat;
  window: TemporalRegion | null;
  max_frames: number | null;
}

/** The audio encodings the audio-extract job can emit. Default (station) is wav. */
export const AUDIO_FORMATS = ["wav", "mp3", "m4a"] as const;
export type AudioFormat = (typeof AUDIO_FORMATS)[number];

/**
 * POST /video/jobs/audio_extract request body. Pulls the audio track out of a
 * video `source` into ONE standalone audio MediaRef (polled back via
 * jobRecordSchema → result.outputs[0]). `fmt` is the output container/codec; the
 * station defaults it to "wav" (lossless and trivially decodable for the
 * waveform). Mirrors FrameExtractRequest's explicit shape — no silent defaults on
 * the wire. The temporal crop that follows reuses the EXISTING crop endpoint
 * (CropJobRequest with spatial: null + a temporal window).
 */
export interface AudioExtractRequest {
  source: MediaRef;
  fmt: AudioFormat;
}

/**
 * POST /video/jobs/generate_image — one ordered multimodal prompt part.
 * The prompt is an ORDERED list: text segments, images from the library (picked
 * frames or prior generations), and optional raw video parts the backend chain
 * resolves into frames (uniform-N sampling). Order is preserved by insertion.
 */
export type GenPromptPart =
  | { kind: "text"; text: string }
  | { kind: "image"; media: MediaRef }
  | { kind: "video"; media: MediaRef };

/**
 * POST /video/jobs/generate_image request body. `parts` is the ordered prompt;
 * `model_id` comes from the shared image-model dropdown (task text-to-image).
 * All knobs are explicit numbers the station renders and the user can edit;
 * `seed` null = random, `negative` null = none. The station guards that at least
 * one non-empty text part exists before enabling Run (a prompt with no text part
 * fails backend-side with "no_prompt"). Polled back via jobRecordSchema →
 * result.outputs[0] = the generated image MediaRef.
 */
export interface GenerateImageRequest {
  parts: GenPromptPart[];
  model_id: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number | null;
  negative: string | null;
  /**
   * Optional auto-archive project NAME. When set, the backend files the output
   * under assets/<project> and echoes the resolved project on result.project.
   * Blank/omitted = an auto-named folder.
   */
  project?: string;
}

/**
 * ONE row of the image TESTER (Generate station, Image mode).
 *
 * The point of the tester is COMPARISON: settings belong to a prompt, not to the
 * station, so you can sit two prompts side by side with different steps/guidance/
 * seed/model and see the difference. Each row therefore carries its OWN complete
 * knob set — a row is a whole `GenerateImageRequest` waiting to happen, and "Run
 * all" fires one job per row.
 *
 * Knobs are held as STRINGS because these are live text inputs (blank seed =
 * random, blank negative = none); they are parsed into a `GenerateImageRequest`
 * only at submit, exactly as the base composer already does. A row that fails to
 * parse is skipped rather than submitted with silently coerced numbers.
 *
 * `media` is deliberately NOT per-row: rows inherit the base composer's image /
 * video / identity parts, so a tester run holds the reference material fixed and
 * varies only what you are testing. Give a row its own text and its own knobs;
 * everything else stays comparable by construction.
 */
export interface ImageTestRow {
  /** Stable client-side id (React key + jobs-by-row map key). Never sent. */
  id: string;
  /** This row's prompt text. Combined with the base composer's media parts. */
  text: string;
  model_id: string;
  width: string;
  height: string;
  steps: string;
  guidance: string;
  /** Blank = random. */
  seed: string;
  /** Blank = none. */
  negative: string;
  /**
   * PER-SETTING CARRY TICKS (operator ask 2026-08-04, k63) — which of this row's
   * settings are carried into the next component seeded FROM this one ("+ Add
   * row" / "+ Row per model" / "+ Row per preset"). An unticked setting is NOT
   * copied: the new row falls back to the mode default instead.
   *
   * CLIENT-SIDE ONLY. Like `id`, this is editor bookkeeping — it is never
   * serialized into a GenerateImageRequest and never reaches the backend (see
   * runnableTestRows in GenerateStation.tsx, which reads only the knob fields).
   */
  carry: CarryTicks;
}

/**
 * The settings a carry tick can cover. `size` is width+height as ONE tick — they
 * are edited as one "size W × H" control, so they carry together or not at all.
 */
export type CarrySettingKey =
  | "model"
  | "size"
  | "steps"
  | "guidance"
  | "seed"
  | "negative";

/** Tick state for every carryable setting. Client-side only — never sent. */
export type CarryTicks = Record<CarrySettingKey, boolean>;

/**
 * POST /video/jobs/generate_scene request body (Generate station, Scene mode).
 * ONE ordered multimodal prompt → N consecutive image frames forming a scene +
 * (when `assemble`) one assembled mp4 clip. Shares the prompt/model/knob shape of
 * GenerateImageRequest and adds the scene-only knobs: `n_frames` (REQUIRED,
 * explicit; backend caps at 24 → HTTP 400 "frame_cap_exceeded"), `fps` (REQUIRED,
 * explicit), `assemble` (build the mp4 from the frames), and `motion` (an
 * optional per-frame progression template that may contain {i}/{n}). Polled back
 * via jobRecordSchema → result.outputs = N image refs, then (if assemble) ONE
 * video ref LAST — the SAME status seam generate_image uses.
 *
 * Image-to-image conditioning: the FIRST image part in `parts` is the scene's
 * start frame. `strength` is the img2img denoising strength (0..1, null = backend
 * default of 0.45); lower keeps more of the start frame. `chain` (default true)
 * conditions each frame on the previous one (sequential img2img). Until the fleet
 * ships an img2img-capable worker a chained request may fail retryably with
 * "image-to-image not available on the fleet" — surfaced via the normal job-error
 * path.
 */
export interface GenerateSceneRequest {
  parts: GenPromptPart[];
  model_id: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  n_frames: number;
  fps: number;
  assemble: boolean;
  strength: number | null;
  chain: boolean;
  seed: number | null;
  motion: string | null;
  negative: string | null;
  /**
   * Optional auto-archive project NAME. When set, the backend files the scene's
   * frames + clip under assets/<project> and echoes it on result.project.
   * Blank/omitted = an auto-named folder.
   */
  project?: string;
}

/**
 * ONE goal row of a Movie's GOAL TIMELINE (Generate station, Movie mode). The
 * timeline is an ORDERED, CONTIGUOUS list of goals that tile the whole clip:
 * `start_frame`..`end_frame` is HALF-OPEN ([start,end)), goal[0] starts at 0, and
 * each goal begins exactly where the previous one ends (no gaps/overlaps) so the
 * union is [0,total). `prompt` describes what that interval should depict; `ref`
 * is an optional per-goal reference/start image the backend conditions on.
 */
export const movieGoalSchema = z
  .object({
    start_frame: z.number(),
    end_frame: z.number(),
    prompt: z.string(),
    ref: mediaRefSchema.optional(),
    // Per-goal PROMPT-COMPONENT overrides (k92): each goal IS a prompt component
    // and may carry its OWN generation knobs, mirroring Scene's per-part settings.
    // ALL optional — an omitted field means the backend INHERITS the movie-level
    // (GenerateMovieRequest) value, so a movie with no overrides is byte-for-byte
    // the old payload. `motion` is goal-only (the movie request has no shared one).
    model_id: z.string().optional(),
    width: z.number().optional(),
    height: z.number().optional(),
    steps: z.number().optional(),
    guidance: z.number().optional(),
    seed: z.number().nullable().optional(),
    negative: z.string().nullable().optional(),
    strength: z.number().nullable().optional(),
    chain: z.boolean().optional(),
    motion: z.string().nullable().optional(),
  })
  .passthrough();
export type MovieGoal = z.infer<typeof movieGoalSchema>;

/**
 * POST /video/jobs/generate_movie request body (Generate station, Movie mode). A
 * GOAL TIMELINE (`goals`, contiguous, tiling [0,total)) drives an N-segment movie:
 * each goal becomes one segment the backend renders into consecutive frames, then
 * (when `assemble`) an mp4 clip is built LAST. The scene-template knobs
 * (`model_id`/dims/`steps`/`guidance`/`fps`/`assemble` + optional
 * `seed`/`negative`/`strength`/`chain`/`project`) are REUSED verbatim from Scene
 * mode. The director knobs are OPT-IN: with `vision_enabled` the backend scores
 * each segment with a judge VLM (`judge_model_id`) and re-rolls a segment until it
 * clears `score_threshold` (0-100) or hits `max_attempts_per_segment` /
 * `time_budget_s`. Polled back via jobRecordSchema → a NESTED movie progress
 * (movieProgressSchema) while running, then result.outputs = segment frames then
 * the movie.mp4 LAST (+ result.movie metadata).
 */
export interface GenerateMovieRequest {
  goals: MovieGoal[];
  model_id: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  fps: number;
  assemble: boolean;
  /** Scene-template optionals (reused from GenerateSceneRequest). */
  seed?: number | null;
  negative?: string | null;
  strength?: number | null;
  chain?: boolean;
  project?: string;
  /** Director knobs — opt-in vision scoring / re-roll loop. */
  vision_enabled?: boolean;
  score_threshold?: number;
  max_attempts_per_segment?: number;
  judge_model_id?: string;
  time_budget_s?: number;
}

/**
 * ONE goal of a curated MOVIE TEMPLATE's shot list — a lighter cousin of
 * movieGoalSchema (no per-goal `ref`; a template ships prompt-only goals). The
 * timeline is still the contiguous [start,end) tiling the Movie editor loads.
 */
export const moviePresetGoalSchema = z
  .object({
    start_frame: z.number(),
    end_frame: z.number(),
    prompt: z.string(),
  })
  .passthrough();
export type MoviePresetGoal = z.infer<typeof moviePresetGoalSchema>;

/**
 * A curated MOVIE TEMPLATE (Generate station, Movie tab). GET /movie/presets → a
 * BARE ARRAY of these (NOT the {presets:[…]} envelope video presets use), and
 * POST /movie/presets/<id>/apply echoes the SAME object. Unlike a video preset
 * (knob-only "ideal default load"), a movie preset carries the WHOLE shot list —
 * picking one drops a ready `goals` timeline + the shared knobs (model/dims/steps/
 * guidance/fps/chain) + opt-in director defaults into the editor, ready to run or
 * tweak. Tolerant/passthrough: numeric knobs are nullable+optional (a template may
 * omit any — the editor keeps its current value), only id/name/model_key/goals are
 * asserted.
 */
export const moviePresetSchema = z
  .object({
    id: z.string(),
    name: z.string(),
    description: z.string().nullable().optional(),
    model_key: z.string(),
    width: z.number().nullable().optional(),
    height: z.number().nullable().optional(),
    steps: z.number().nullable().optional(),
    guidance: z.number().nullable().optional(),
    fps: z.number().nullable().optional(),
    chain: z.boolean().nullable().optional(),
    strength: z.number().nullable().optional(),
    negative: z.string().nullable().optional(),
    goals: z.array(moviePresetGoalSchema),
    vision_enabled: z.boolean().nullable().optional(),
    score_threshold: z.number().nullable().optional(),
  })
  .passthrough();
export type MoviePreset = z.infer<typeof moviePresetSchema>;

export const JOB_STATUSES = [
  "queued",
  "claimed",
  "running",
  "cancelling", // cancel requested; a running scene honors it between frames
  "cancelled",
  "done",
  "failed",
] as const;
export type JobStatus = (typeof JOB_STATUSES)[number] | null;

export const jobErrorSchema = z
  .object({
    code: z.string(),
    message: z.string(),
    retryable: z.boolean(),
  })
  .passthrough();
export type JobError = z.infer<typeof jobErrorSchema>;

/**
 * The auto-archive project a terminal generation was saved into. Present on a
 * `done` result for generate_image/generate_scene: `name` is the user-supplied
 * label (null when the backend auto-named), `dir` is the on-disk folder the
 * frames + clip were written to (the "Saved to …" surface), `uuid` a stable id.
 */
export const jobProjectSchema = z
  .object({
    name: z.string().nullable(),
    uuid: z.string(),
    dir: z.string(),
  })
  .passthrough();
export type JobProject = z.infer<typeof jobProjectSchema>;

/**
 * The terminal `movie` metadata block a generate_movie run echoes on its result —
 * the per-segment ledger (chosen take, attempts, scores, why) + the assembled
 * clip + overall stats. Tolerant/passthrough: the UI renders the movie mostly off
 * result.outputs (frames + movie.mp4 last) + result.project, so every field here
 * is optional — this schema exists so the block round-trips + is inspectable.
 */
export const movieResultSchema = z
  .object({
    goals: z.array(movieGoalSchema).optional(),
    drift: z.unknown().optional(),
    vision_enabled: z.boolean().optional(),
    score_threshold: z.number().nullable().optional(),
    n_frames_total: z.number().optional(),
    segments: z
      .array(
        z
          .object({
            index: z.number(),
            goal: z.unknown().optional(),
            prompt: z.string().optional(),
            seed: z.number().nullable().optional(),
            attempts: z.number().optional(),
            // scores carries null for UNSCORED takes (vision off) — must allow null
            scores: z.array(z.number().nullable()).optional(),
            chosen_take: z.number().nullable().optional(),
            status: z.string().optional(),
            why: z.string().nullable().optional(),
            // backend writes mp4 as a relative FILENAME string (e.g. "segment_00/
            // video.mp4"), not a MediaRef — accept either, tolerantly.
            mp4: z.union([z.string(), mediaRefSchema]).nullable().optional(),
          })
          .passthrough(),
      )
      .optional(),
    // manifest movie is the FILENAME string "movie.mp4" (the playable clip rides
    // in result.outputs as a MediaRef) — accept string or MediaRef.
    movie: z.union([z.string(), mediaRefSchema]).nullable().optional(),
  })
  .passthrough();
export type MovieResult = z.infer<typeof movieResultSchema>;

export const jobResultSchema = z
  .object({
    ok: z.boolean(),
    outputs: z.array(mediaRefSchema),
    error: jobErrorSchema.nullable(),
    /** Auto-archive destination (generation kinds); absent/null otherwise. */
    project: jobProjectSchema.nullable().optional(),
    /** Movie-mode ledger (generate_movie only); absent/null otherwise. */
    movie: movieResultSchema.nullable().optional(),
  })
  .passthrough();
export type JobResult = z.infer<typeof jobResultSchema>;

/**
 * Live per-run progress a job reports WHILE running (null/absent at terminal).
 * `done`/`total` drive the "image {done}/{total}" readout; `stage`+`label`
 * describe the current step; `frames` are completed outputs so far (SAME MediaRef
 * shape as result.outputs — rendered with the existing frame renderer so a scene
 * gallery fills in live); `started_at` (epoch seconds) anchors the elapsed clock;
 * `eta_s` is the backend's remaining-time estimate when it has one.
 */
export const jobProgressSchema = z
  .object({
    done: z.number(),
    total: z.number(),
    stage: z.string(),
    label: z.string().optional(),
    model: z.string().optional(),
    frames: z.array(mediaRefSchema).optional(),
    started_at: z.number().optional(),
    eta_s: z.number().nullable().optional(),
  })
  .passthrough();
export type JobProgress = z.infer<typeof jobProgressSchema>;

/**
 * One segment's live state inside a movie's NESTED progress. `index` is the goal's
 * position; `goal` echoes the source goal (tolerated as unknown — the UI derives a
 * label from `prompt`/`index`); `attempt` is the 1-based re-roll count; `score` is
 * the latest judge score (0-100, null before scoring / when vision is off);
 * `status` walks pending→generating→scoring→done (skipped/resumed for the vision
 * loop); `frames` are the take's frames so far (SAME MediaRef shape as outputs).
 */
export const movieSegmentProgressSchema = z
  .object({
    index: z.number(),
    goal: z.unknown().optional(),
    prompt: z.string().optional(),
    attempt: z.number().optional(),
    score: z.number().nullable().optional(),
    status: z.string().optional(),
    frames: z.array(mediaRefSchema).optional(),
  })
  .passthrough();
export type MovieSegmentProgress = z.infer<typeof movieSegmentProgressSchema>;

/**
 * The NESTED progress a generate_movie job reports WHILE running — the key shape
 * difference from a flat scene job. `segment_done`/`segment_total` drive the
 * overall "segment d/t" line; `segments[]` is the per-segment strip; `current` is
 * a flat JobProgress-shaped view of the ACTIVE segment's frame render (fed VERBATIM
 * to the shipped progressReadout + live-frame grid). `stage` names the movie-level
 * step (loading/generating/scoring/assembling/archiving).
 */
export const movieProgressSchema = z
  .object({
    stage: z.string(),
    segment_done: z.number(),
    segment_total: z.number(),
    segments: z.array(movieSegmentProgressSchema),
    current: jobProgressSchema.nullable().optional(),
    started_at: z.number().optional(),
    eta_s: z.number().nullable().optional(),
  })
  .passthrough();
export type MovieProgress = z.infer<typeof movieProgressSchema>;

/** Either progress shape a poll may carry (flat scene/image OR nested movie). */
export type AnyProgress = JobProgress | MovieProgress;

/** GET /video/jobs/<id> → the current status + terminal result of one job. */
export const jobRecordSchema = z
  .object({
    job_id: z.string(),
    status: z.enum(JOB_STATUSES).nullable(),
    result: jobResultSchema.nullable(),
    /**
     * Live progress while running; null/absent at terminal. A UNION of the flat
     * JobProgress (image/scene) and the nested MovieProgress (movie) — discriminated
     * by the presence of `segments`. A movie blob carries no top-level done/total so
     * jobProgressSchema rejects it, and a flat blob carries no segment_* so
     * movieProgressSchema rejects it — the union is unambiguous either order. Both
     * members passthrough, and `.nullable().optional()` keeps applyPoll's
     * kind-agnostic `progress: jr.progress ?? null` line working unchanged.
     */
    progress: z
      .union([jobProgressSchema, movieProgressSchema])
      .nullable()
      .optional(),
  })
  .passthrough();
export type JobRecord = z.infer<typeof jobRecordSchema>;

/** Narrow a poll's progress union to the nested movie shape (presence of segments). */
export function isMovieProgress(p: AnyProgress): p is MovieProgress {
  return (
    typeof p === "object" &&
    p != null &&
    "segments" in p &&
    Array.isArray((p as { segments?: unknown }).segments)
  );
}

/** Terminal states — polling stops here. */
export function isTerminal(status: JobStatus): boolean {
  return status === "done" || status === "failed" || status === "cancelled";
}

/**
 * Flatten an ordered multimodal prompt's TEXT parts into one human-readable
 * summary string (media parts are dropped). Used to annotate a generation's
 * library outputs with the prompt that produced them (session-library provenance)
 * and to re-stage that prompt into the composer for Replicate / Continue. Purely
 * client-side derivation — the wire contract is untouched.
 */
export function promptSummary(parts: GenPromptPart[]): string {
  return parts
    .filter(
      (p): p is Extract<GenPromptPart, { kind: "text" }> => p.kind === "text",
    )
    .map((p) => p.text.trim())
    .filter((t) => t !== "")
    .join(" ")
    .trim();
}

export const MAX_SOURCE_IMAGES = 12;
export const MAX_CANONICAL_IMAGES = 4;

/**
 * A discrete angular view in an angle-ring reconstruction.
 */
export const identityAngleViewSchema = z.object({
  view_id: z.string(),
  azimuth_deg: z.number(),
  elevation_deg: z.number().optional(),
  image_uri: z.string().nullable().optional(),
  status: z.enum(["pending", "generating", "generated", "approved", "rejected", "failed"]).optional(),
  version: z.number().optional(),
  seed: z.number().nullable().optional(),
  error: z.string().nullable().optional(),
}).passthrough();

export type IdentityAngleView = z.infer<typeof identityAngleViewSchema>;

export const identityReconstructionMeshSchema = z.object({
  status: z.enum(["none", "queued", "running", "completed", "failed"]),
  glb_uri: z.string().optional(),
  error: z.string().optional(),
}).passthrough();

export const identityReconstructionSchema = z.object({
  recon_id: z.string(),
  mode: z.enum(["sheet", "turntable", "angle-ring"]).optional(),
  views: z.array(z.any()).optional(), // Lenient: accepts both legacy strings & new view objects
  mesh: identityReconstructionMeshSchema.optional(),
  created_at: z.number().optional(),
  degrees_per_frame: z.number().optional(),
  frame_count: z.number().optional(),
}).passthrough();

export type IdentityReconstruction = z.infer<typeof identityReconstructionSchema>;