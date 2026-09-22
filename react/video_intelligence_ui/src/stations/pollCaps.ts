// Phase 8 — shared poll-cap constants + "still running" copy for the five
// job-poll hooks (useCropJobs, useFrameJobs, useGenerateJob, useAudioExtractJob,
// useAudioCropJobs). Centralised here so the caps and the human message never
// drift across the hooks.
//
// The old behaviour capped a poll loop at ~60s of wall-clock and marked the job
// `failed` with "Timed out" — even though the backend often finished the work a
// moment later (cold model loads especially). Phase 8 does two things:
//   1. Raises the ceilings substantially (generation cold-loads can take minutes).
//   2. On cap-expiry, STOPS hammering the poller but does NOT fail the job —
//      instead each hook flips a `capped` flag, keeps the last real status, and
//      exposes a `resume()` that re-anchors the poll clock (startedAt = now,
//      clears capped) so polling simply continues. No SSE, no websockets.

/** Generation cold-loads can take minutes — 5 min ceiling before we pause polling. */
export const GENERATE_POLL_CAP_MS = 300_000;

/** Crop / frame-extract / audio-extract / audio-crop — 3 min ceiling. */
export const JOB_POLL_CAP_MS = 180_000;

/**
 * Copy shown when a poll loop hits its cap but the job may still be finishing
 * server-side. Paired with a "Check again" affordance wired to the hook's resume.
 */
export const STILL_RUNNING_MESSAGE = "Still running on the server — check again.";

/**
 * Scene generation (Generate station, Scene mode) produces N frames — each a
 * full text-to-image inference — plus an optional mp4 assemble, so the ceiling
 * scales with n_frames rather than a single flat cap. Base 5 min (a cold model
 * load like generate_image) + 1 min per frame.
 *
 * Chaining (`chain: true`) runs the frames as SEQUENTIAL img2img — each frame
 * conditions on the previous, so they can't overlap and the wall-clock grows.
 * When chaining we add an extra per-frame allowance on top of the base per-frame
 * budget so the cap doesn't trip while a legitimately-slower chained scene runs.
 */
export const GENERATE_SCENE_POLL_CAP_BASE_MS = 300_000;
export const GENERATE_SCENE_POLL_CAP_PER_FRAME_MS = 60_000;
export const GENERATE_SCENE_POLL_CAP_CHAIN_PER_FRAME_MS = 60_000;

/**
 * The n_frames-scaled poll cap (ms) for a generate_scene job. `chain` adds extra
 * per-frame time because chained frames run sequentially (img2img).
 */
export function generateScenePollCapMs(nFrames: number, chain?: boolean): number {
  const frames = Math.max(1, nFrames | 0);
  const perFrame =
    GENERATE_SCENE_POLL_CAP_PER_FRAME_MS +
    (chain ? GENERATE_SCENE_POLL_CAP_CHAIN_PER_FRAME_MS : 0);
  return GENERATE_SCENE_POLL_CAP_BASE_MS + perFrame * frames;
}

/**
 * Movie generation (Generate station, Movie mode) renders MANY segments, each an
 * N-frame scene, plus an mp4 assemble — and with the vision director on, a segment
 * may be re-rolled several times. So the ceiling scales with BOTH the segment count
 * and the total frame count (segments × frames): base 5 min (a cold model load) +
 * 1 min per segment + a per-total-frame budget. Chaining adds extra per-frame time
 * (sequential img2img), matching the scene cap's chain allowance.
 */
export const GENERATE_MOVIE_POLL_CAP_BASE_MS = 300_000;
export const GENERATE_MOVIE_POLL_CAP_PER_SEGMENT_MS = 60_000;
export const GENERATE_MOVIE_POLL_CAP_PER_FRAME_MS = 20_000;
export const GENERATE_MOVIE_POLL_CAP_CHAIN_PER_FRAME_MS = 15_000;

/**
 * The segments×frames-scaled poll cap (ms) for a generate_movie job. `framesTotal`
 * is the timeline's total frame count (`total` = max end_frame); `chain` adds extra
 * per-frame time because chained frames run sequentially (img2img).
 */
export function generateMoviePollCapMs(
  segments: number,
  framesTotal?: number,
  chain?: boolean,
): number {
  const segs = Math.max(1, segments | 0);
  const frames = Math.max(segs, framesTotal ? framesTotal | 0 : segs);
  const perFrame =
    GENERATE_MOVIE_POLL_CAP_PER_FRAME_MS +
    (chain ? GENERATE_MOVIE_POLL_CAP_CHAIN_PER_FRAME_MS : 0);
  return (
    GENERATE_MOVIE_POLL_CAP_BASE_MS +
    GENERATE_MOVIE_POLL_CAP_PER_SEGMENT_MS * segs +
    perFrame * frames
  );
}
