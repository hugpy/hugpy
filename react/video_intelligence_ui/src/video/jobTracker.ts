// The session-wide job tracker — a durable, cross-station memory of every job
// this session has enqueued on the hugpy bus.
//
// The problem it solves: WorkbenchStation mounts ONLY the active sub-tab, so
// switching tabs UNMOUNTS a station and destroys its per-hook polling loop +
// JobCard state. The server keeps working (the job bus is durable: GET
// /video/jobs/<id> answers from any worker), but the UI forgot it was watching,
// so outputs never reached the library and the tab looked idle on return.
//
// This module lifts the polling out of the (unmountable) hooks into ONE
// module-level loop that lives for the whole session. Records persist to
// sessionStorage (write-through, keyed by session id) so in-flight jobs survive
// BOTH sub-tab switches AND page reloads — on reload we rehydrate and resume
// polling every non-terminal job. The five hooks now just POST their enqueue and
// register the returned job_id here; everything downstream (poll cadence, caps,
// terminal parse, the ONE centralized library push) happens in this module.
//
// Architecture mirrors mediaLibrary.ts exactly: a module-level `cache` (the
// stable snapshot source), a `listeners` Set + `emit()`, a one-time storage
// bind, sessionStorage write-through on every mutation, `getSessionId()` reuse,
// and a `useSyncExternalStore` hook. getSnapshot returns the WHOLE list (a
// referentially-stable array) — consumers filter with useMemo in their own body,
// never inside getSnapshot (a fresh array per read loops React 18).
import { useSyncExternalStore } from "react";
import { getSessionId } from "../session";
import { isCanned } from "../demo/mode";
import { installDemoTransport } from "../demo/demoFetch";
import { jobStatusUrl, jobCancelUrl } from "../config";
import { request, okValue } from "../transport/client";
import { addToLibrary } from "./mediaLibrary";
import {
  jobRecordSchema,
  isTerminal,
  isMovieProgress,
  type JobStatus,
  type MediaRef,
  type JobProject,
  type AnyProgress,
} from "./contract";
import type { SpatialRegion, TemporalRegion } from "../regions/types";
import {
  GENERATE_POLL_CAP_MS,
  JOB_POLL_CAP_MS,
  generateScenePollCapMs,
  generateMoviePollCapMs,
} from "../stations/pollCaps";

/** The job kinds this arm enqueues, one poll-parse branch per kind. */
export type JobKind =
  | "crop"
  | "frame_extract"
  | "audio_extract"
  | "generate_image"
  | "generate_scene"
  | "generate_movie";

/** The station a job belongs to — also the library origin for crop pushes.
 *  "video-crop" (Studio slice 2b) is the dedicated id for the one-pass spatial+
 *  temporal VIDEO crop (useVideoCropJobs), split out of "image-crop" so a video
 *  item's inline Video Crop records never intermix with Image Crop's tracker rows.
 *  "generate-tester" is the same split for the SAME reason: useGenerateJob picks
 *  its job as the newest record with station "generate", so the tester's N
 *  per-row jobs must not be filed there or the last row to enqueue would
 *  masquerade as the single-prompt run's result. */
export type StationId =
  | "image-crop"
  | "video-crop"
  | "audio-crop"
  | "frames"
  | "generate"
  | "generate-tester"
  | "generate-scene-parts";

/**
 * One tracked job. Persisted to sessionStorage, so — because a record may be
 * polled to completion AFTER a page reload (the enqueue closure is long gone) —
 * the terminal library push is DATA-DRIVEN off this record, never a stored
 * closure. `libLabel`/`kind`/`station` carry everything needed to reconstruct
 * the exact library origin + label per kind (see pushOutputs).
 */
export interface TrackedJob {
  /** Server job id — the durable handle GET /video/jobs/<id> answers to. */
  jobId: string;
  /**
   * Stable per-row correlation key for multi-job fan-out stations (the scene
   * composer's part key). Lets a surface rebuild its row->job map from the durable
   * tracker after an unmount, instead of a transient in-component useState. Null
   * for single-job stations.
   */
  rowKey: string | null;
  kind: JobKind;
  station: StationId;
  /** Human process-row label (shown in the sidebar Processes section). */
  label: string;
  /** When the user kicked the run off (ms epoch) — the elapsed-clock anchor. */
  enqueuedAt: number;
  /** Poll-clock anchor for the cap; resume() (and reload) re-anchor it. */
  startedAt: number;
  /** Latest polled status (null = enqueued-but-not-yet-claimed, or forgotten). */
  status: JobStatus;
  /**
   * Poll cap reached while still non-terminal (Phase 8). NOT a failure — the job
   * may still finish server-side; polling THIS record is paused until resume()
   * re-anchors it. Status is left untouched.
   */
  capped: boolean;
  /**
   * The server no longer knows this job id (a poll returned ok with status:null,
   * e.g. after a reload of a job the bus already forgot). A non-crashing,
   * terminal-ish state — polling stops; the row surfaces as "unknown / expired".
   */
  expired: boolean;
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Produced MediaRefs, recorded on terminal `done`. */
  outputs: MediaRef[];
  /**
   * The library label to push on `done` for crop kinds (carried from the enqueue
   * site — `crop W×H` / `clip a–bs`). Null for kinds with a static or per-output
   * label (generate/frames) or no push (audio_extract).
   */
  libLabel: string | null;
  /** 1-based display index for multi-job stations (crop / audio-crop). */
  index: number;
  /** The region a crop job crops — kept so cards re-render after a reload. */
  region: SpatialRegion | TemporalRegion | null;
  /**
   * Requested frame count for a generate_scene job — the poll cap scales off it
   * (see capFor). Null for every other kind. Persisted so the scaled cap survives
   * a reload of an in-flight scene job.
   */
  nFrames: number | null;
  /**
   * Whether a generate_scene job chains each frame on the previous (sequential
   * img2img). Chaining is slower, so capFor adds extra per-frame poll time when
   * true. Null for every other kind. Persisted so the scaled cap survives a reload
   * of an in-flight scene job.
   */
  chain: boolean | null;
  /**
   * Segment count (goal-timeline length) for a generate_movie job — the poll cap
   * scales off it × the total frame count (see capFor). Null for every other kind.
   * Persisted so the scaled cap survives a reload of an in-flight movie job.
   */
  segments: number | null;
  /**
   * The prompt summary + model that produced a generate_image / generate_scene /
   * generate_movie job — carried from the enqueue site so the ONE centralized
   * library push (pushOutputs) can annotate each output with its provenance
   * (session-library grouping + Replicate/Continue re-staging). Null for every
   * other kind.
   */
  prompt: string | null;
  model: string | null;
  /**
   * Live per-run progress from the latest non-terminal poll. A UNION of the flat
   * JobProgress (image/scene) and the nested MovieProgress (movie) — see
   * AnyProgress. Null when the job hasn't reported progress yet, and cleared to
   * null on any terminal transition so the running readout never lingers past the
   * result. (Persisted, so pushOutputs can read the last movie progress to derive
   * per-segment library labels — reload-safe.)
   */
  progress: AnyProgress | null;
  /**
   * The auto-archive project a terminal generation was saved into (name + on-disk
   * dir + uuid), captured from result.project on `done`. Null for non-generation
   * kinds and while running. Persisted so the "Saved to …" line survives a reload.
   */
  project: JobProject | null;
}

/** The single sessionStorage key prefix (build marker — must reach the bundle). */
const KEY_PREFIX = "vi.jobTracker.v1";

/** Module poll cadence — matches the retired per-hook POLL_MS exactly. */
const POLL_MS = 800;

// Session-scoped key so two tabs keep independent trackers, and the persisted
// list travels with the same session id the uploads/jobs are tagged by.
//
// The canned demo (?demo=1) gets a `:demo`-SUFFIXED key so the brochure is
// hermetic: it never reads or writes the SAME tab's live-session tracker. Without
// this, a job persisted by a PRIOR live visit in this tab session would rehydrate
// under the demo (the sid is shared), its GET /video/jobs/<id> poll would miss in
// the shim, and it would surface as a stale "server no longer tracks this job"
// expired row. isCanned() is URL-derived and available at module load, so the key
// is right from the first read regardless of when installVideoDemo() runs.
function storageKey(): string {
  const suffix = isCanned() ? ":demo" : "";
  try {
    return `${KEY_PREFIX}:${getSessionId()}${suffix}`;
  } catch {
    return `${KEY_PREFIX}${suffix}`;
  }
}

// In-memory cache is the source of truth for getSnapshot — its reference changes
// only on an actual mutation, which keeps useSyncExternalStore stable.
let cache: TrackedJob[] | null = null;
const listeners = new Set<() => void>();
let storageBound = false;
let pollTimer: number | null = null;

function readStorage(): TrackedJob[] {
  try {
    const raw = sessionStorage.getItem(storageKey());
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Drop malformed entries defensively — a stored record must carry a jobId.
    return parsed.filter(
      (it) => it && typeof it === "object" && typeof it.jobId === "string",
    ) as TrackedJob[];
  } catch {
    return [];
  }
}

function writeStorage(items: TrackedJob[]): void {
  try {
    sessionStorage.setItem(storageKey(), JSON.stringify(items));
  } catch {
    // storage unavailable / quota exceeded — the in-memory cache still serves
    // this session; persistence is best-effort (mirrors mediaLibrary).
  }
}

function ensureLoaded(): TrackedJob[] {
  if (cache == null) cache = readStorage();
  return cache;
}

function emit(): void {
  for (const l of listeners) l();
}

/** Terminal for polling purposes: done/failed OR the server forgot it. */
export function isJobTerminal(rec: TrackedJob): boolean {
  return isTerminal(rec.status) || rec.expired;
}

/** Still worth polling: has an id, not terminal, not capped, not expired. */
export function isJobActive(rec: TrackedJob): boolean {
  return (
    rec.jobId != null &&
    !isTerminal(rec.status) &&
    !rec.capped &&
    !rec.expired
  );
}

function anyActive(): boolean {
  return ensureLoaded().some(isJobActive);
}

function capFor(
  kind: JobKind,
  nFrames?: number | null,
  chain?: boolean | null,
  segments?: number | null,
): number {
  if (kind === "generate_movie")
    return generateMoviePollCapMs(segments ?? 1, nFrames ?? undefined, chain ?? false);
  if (kind === "generate_scene")
    return generateScenePollCapMs(nFrames ?? 1, chain ?? false);
  return kind === "generate_image" ? GENERATE_POLL_CAP_MS : JOB_POLL_CAP_MS;
}

function opFor(kind: JobKind): string {
  switch (kind) {
    case "crop":
      return "crop.poll";
    case "frame_extract":
      return "frame_extract.poll";
    case "audio_extract":
      return "audio_extract.poll";
    case "generate_image":
      return "generate_image.poll";
    case "generate_scene":
      return "generate_scene.poll";
    case "generate_movie":
      return "generate_movie.poll";
  }
}

function failMsgFor(kind: JobKind): string {
  switch (kind) {
    case "crop":
      return "Crop job failed.";
    case "frame_extract":
      return "Frame-extract job failed.";
    case "audio_extract":
      return "Audio-extract job failed.";
    case "generate_image":
      return "Generate job failed.";
    case "generate_scene":
      return "Scene generation failed.";
    case "generate_movie":
      return "Movie generation failed.";
  }
}

// Apply a partial patch to one record by jobId; persist + emit on any change.
function mutate(jobId: string, patch: Partial<TrackedJob>): void {
  const current = ensureLoaded();
  let changed = false;
  const next = current.map((rec) => {
    if (rec.jobId !== jobId) return rec;
    changed = true;
    return { ...rec, ...patch };
  });
  if (!changed) return;
  cache = next;
  writeStorage(cache);
  emit();
}

// The ONE centralized library push. Data-driven off the record (no stored
// closure) so it works even when a job completes after a page reload. mediaLibrary
// de-dups by ref.uri, so this single site can never double-add.
function pushOutputs(rec: TrackedJob, outputs: MediaRef[]): void {
  if (rec.kind === "frame_extract") {
    outputs.forEach((f, i) => addToLibrary(f, "frames", `frame ${i + 1}`));
    return;
  }
  if (rec.kind === "crop") {
    const out = outputs[0];
    if (out) addToLibrary(out, rec.station, rec.libLabel ?? undefined);
    return;
  }
  if (rec.kind === "generate_image") {
    const out = outputs[0];
    if (out)
      addToLibrary(out, "generate", "generated", {
        ...(rec.prompt != null ? { prompt: rec.prompt } : {}),
        ...(rec.model != null ? { model: rec.model } : {}),
        groupId: rec.jobId,
        genKind: "generate_image",
      });
    return;
  }
  if (rec.kind === "generate_scene") {
    const prov = {
      ...(rec.prompt != null ? { prompt: rec.prompt } : {}),
      ...(rec.model != null ? { model: rec.model } : {}),
      groupId: rec.jobId,
      genKind: "generate_scene" as const,
    };
    outputs.forEach((m, i) => {
      const isLast = i === outputs.length - 1;
      if (isLast && m.kind === "video")
        addToLibrary(m, "generate", "scene clip", prov);
      else if (m.kind === "image")
        addToLibrary(m, "generate", `scene frame ${i + 1}`, prov);
    });
    return;
  }
  if (rec.kind === "generate_movie") {
    const prov = {
      ...(rec.prompt != null ? { prompt: rec.prompt } : {}),
      ...(rec.model != null ? { model: rec.model } : {}),
      groupId: rec.jobId,
      genKind: "generate_movie" as const,
    };
    // Derive `segment N frame M` labels from the LAST movie progress (persisted on
    // the record, so this is reload-safe): match each output frame to the segment
    // whose take contained it by uri. Falls back to a global `movie frame N` when a
    // frame isn't in the last progress snapshot; the assembled clip (video, LAST)
    // is labelled "movie".
    const labelByUri = new Map<string, string>();
    if (rec.progress && isMovieProgress(rec.progress)) {
      for (const seg of rec.progress.segments) {
        (seg.frames ?? []).forEach((f, m) => {
          labelByUri.set(f.uri, `segment ${seg.index + 1} frame ${m + 1}`);
        });
      }
    }
    outputs.forEach((m, i) => {
      const isLast = i === outputs.length - 1;
      if (isLast && m.kind === "video") addToLibrary(m, "generate", "movie", prov);
      else if (m.kind === "image")
        addToLibrary(m, "generate", labelByUri.get(m.uri) ?? `movie frame ${i + 1}`, prov);
    });
    return;
  }
  // audio_extract: intermediate (feeds the temporal editor) — never pushed.
}

// Interpret one poll response with the SAME parse each hook used, branched by
// kind. `done` records outputs + runs the single push; `failed` records the
// error; a null status = the bus forgot the job → expired; anything else just
// advances the status.
function applyPoll(rec: TrackedJob, raw: unknown): void {
  const parsed = jobRecordSchema.safeParse(raw);
  if (!parsed.success) return; // malformed — retry next tick (matches hooks)
  const jr = parsed.data;
  if (jr.status === "failed") {
    mutate(rec.jobId, {
      status: "failed",
      error: jr.result?.error?.message ?? failMsgFor(rec.kind),
      progress: null,
    });
  } else if (jr.status === "cancelled") {
    // Operator-stopped — terminal, but reported as "you stopped it", not a fault.
    mutate(rec.jobId, {
      status: "cancelled",
      error: jr.result?.error?.message ?? "cancelled",
      progress: null,
    });
  } else if (jr.status === "done") {
    const outputs = jr.result?.outputs ?? [];
    mutate(rec.jobId, {
      status: "done",
      outputs,
      progress: null,
      // The auto-archive destination, echoed on the terminal result (generation
      // kinds). Surfaced as the "Saved to …" line on the finished preview.
      project: jr.result?.project ?? null,
      ...(outputs.length === 0
        ? { error: `${rec.label} finished with no output.` }
        : {}),
    });
    pushOutputs(rec, outputs);
  } else if (jr.status == null) {
    // Server no longer knows this job id (e.g. reload of a forgotten job). Do NOT
    // crash and do NOT poll forever — mark expired and stop polling this record.
    mutate(rec.jobId, { expired: true });
  } else {
    // Non-terminal (queued/claimed/running/cancelling): advance status AND carry
    // the live progress so the descriptive readout + live frame gallery update.
    mutate(rec.jobId, { status: jr.status, progress: jr.progress ?? null });
  }
}

// One module-level tick: poll every active record, cap the overrun ones. A
// transport error is transient (matches the hooks) — skip and retry next tick.
async function pollTick(): Promise<void> {
  for (const rec of ensureLoaded()) {
    if (!isJobActive(rec)) continue;
    if (
      Date.now() - rec.startedAt >
      capFor(rec.kind, rec.nFrames, rec.chain, rec.segments)
    ) {
      // Cap reached — pause polling THIS record, but DON'T fail it.
      mutate(rec.jobId, { capped: true });
      continue;
    }
    const r = await request<unknown>(jobStatusUrl(rec.jobId), {
      meta: { specKey: rec.station, operation: opFor(rec.kind) },
    });
    if (!r.ok) continue; // transient — retry on the next tick
    applyPoll(rec, okValue(r));
  }
  disarmIfIdle();
}

// Arm the single interval while ≥1 record is active; a no-op if already armed or
// nothing is active. Fires an immediate tick (matches the retired hooks).
function armPolling(): void {
  if (typeof window === "undefined") return;
  if (pollTimer != null) return;
  if (!anyActive()) return;
  // Ordering guard for the canned demo: this module's initTracker() runs at IMPORT
  // time (before entry.tsx's installVideoDemo()), and on a reload it rehydrates any
  // persisted in-flight record and immediately polls it. If the fetch shim isn't up
  // yet, that first poll would hit the REAL backend (a 401 + a live wire hit — the
  // brochure must never do that). installDemoTransport is idempotent, so calling it
  // here guarantees the shim is installed before ANY canned poll fires, whatever the
  // module import order. No-op outside canned mode.
  if (isCanned()) installDemoTransport();
  pollTimer = window.setInterval(() => void pollTick(), POLL_MS);
  void pollTick();
}

// Clear the interval once no record is active (keeps it alive for others).
function disarmIfIdle(): void {
  if (pollTimer != null && !anyActive()) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

/** Fields the enqueue site hands the tracker right after it gets a job_id. */
export interface TrackJobInput {
  jobId: string;
  kind: JobKind;
  station: StationId;
  label: string;
  enqueuedAt: number;
  /** Stable per-row key for multi-job fan-out stations (e.g. the scene part key). */
  rowKey?: string;
  /** Library label to push on done (crop kinds); omit otherwise. */
  libLabel?: string | null;
  /** 1-based display index for multi-job stations. */
  index?: number;
  /** The crop region (crop / audio-crop) — kept so cards survive a reload. */
  region?: SpatialRegion | TemporalRegion | null;
  /**
   * Requested frame count — scales the poll cap. For generate_scene it's n_frames;
   * for generate_movie it's the timeline's TOTAL frame count (max end_frame).
   */
  nFrames?: number;
  /** Whether the job chains frames (sequential img2img) — adds poll time (scene/movie). */
  chain?: boolean;
  /** Segment count (goal-timeline length) for a generate_movie job — scales the poll cap. */
  segments?: number;
  /** Prompt summary that produced a generate_image/generate_scene/generate_movie job (provenance). */
  prompt?: string;
  /** Model id that produced a generate_image/generate_scene/generate_movie job (provenance). */
  model?: string;
}

/**
 * Register a freshly-enqueued job and ARM the polling loop. Called by each hook
 * immediately after its enqueue POST returns a job_id. De-dups by jobId.
 */
export function trackJob(input: TrackJobInput): void {
  if (!input.jobId) return;
  const current = ensureLoaded();
  if (current.some((rec) => rec.jobId === input.jobId)) return;
  const now = Date.now();
  const rec: TrackedJob = {
    jobId: input.jobId,
    rowKey: input.rowKey ?? null,
    kind: input.kind,
    station: input.station,
    label: input.label,
    enqueuedAt: input.enqueuedAt || now,
    startedAt: now,
    status: "queued",
    capped: false,
    expired: false,
    error: null,
    outputs: [],
    libLabel: input.libLabel ?? null,
    index: input.index ?? current.length + 1,
    region: input.region ?? null,
    nFrames: input.nFrames ?? null,
    chain: input.chain ?? null,
    segments: input.segments ?? null,
    prompt: input.prompt ?? null,
    model: input.model ?? null,
    progress: null,
    project: null,
  };
  cache = [...current, rec];
  writeStorage(cache);
  emit();
  armPolling();
}

/**
 * Cooperative cancel: fire-and-forget POST; the next poll observes the
 * 'cancelling' → 'cancelled' transition (a running scene honors it between
 * frames, so a long frame finishes first). Optimistically shows 'cancelling'
 * so the button gives immediate feedback.
 */
export function cancelJob(jobId: string): void {
  const rec = ensureLoaded().find((r) => r.jobId === jobId);
  if (!rec || !isJobActive(rec)) return;
  mutate(jobId, { status: "cancelling" });
  void request<unknown>(jobCancelUrl(jobId), {
    method: "POST",
    meta: { specKey: rec.station, operation: "job.cancel" },
  });
}

/**
 * Re-anchor the poll clock for a capped job and resume polling. Clears `capped`,
 * keeps the last real status. No-op if the id isn't a capped record.
 */
export function resumeJob(jobId: string): void {
  const current = ensureLoaded();
  let changed = false;
  const next = current.map((rec) => {
    if (rec.jobId !== jobId || !rec.capped) return rec;
    changed = true;
    return { ...rec, capped: false, startedAt: Date.now() };
  });
  if (!changed) return;
  cache = next;
  writeStorage(cache);
  emit();
  armPolling();
}

/** Resume EVERY capped job of one station (the no-key resume() form). */
export function resumeStation(station: StationId): void {
  const current = ensureLoaded();
  let changed = false;
  const now = Date.now();
  const next = current.map((rec) => {
    if (rec.station !== station || !rec.capped) return rec;
    changed = true;
    return { ...rec, capped: false, startedAt: now };
  });
  if (!changed) return;
  cache = next;
  writeStorage(cache);
  emit();
  armPolling();
}

/**
 * Forget a station's TERMINAL records (done / failed / expired). Called by a
 * hook's reset() when a fresh source is picked, so the station's result grid
 * starts clean — while any still-in-flight job of that station keeps polling and
 * stays visible (house policy: this only forgets local rows, never server work).
 */
export function dismissTerminal(station: StationId): void {
  const current = ensureLoaded();
  const next = current.filter(
    (rec) => !(rec.station === station && isJobTerminal(rec)),
  );
  if (next.length === current.length) return;
  cache = next;
  writeStorage(cache);
  emit();
}

/**
 * Forget EVERY station's TERMINAL records (done / failed / cancelled / expired) in
 * one call — the user-facing "Clear finished" broom for the workbench's process
 * list. Drops the station filter of dismissTerminal; any still-in-flight job (of
 * any station) keeps polling and stays visible (house policy: this only forgets
 * local rows, never server work).
 */
export function dismissAllTerminal(): void {
  const current = ensureLoaded();
  const next = current.filter((rec) => !isJobTerminal(rec));
  if (next.length === current.length) return;
  cache = next;
  writeStorage(cache);
  emit();
}

// Bind the cross-tab storage listener once (mirrors mediaLibrary). sessionStorage
// is per-tab so this mainly future-proofs the surface; if the key changes under
// us we re-sync + re-arm rather than serving a stale list.
function bindStorageOnce(): void {
  if (storageBound || typeof window === "undefined") return;
  storageBound = true;
  window.addEventListener("storage", (e) => {
    if (e.key !== null && e.key !== storageKey()) return;
    cache = readStorage();
    emit();
    armPolling();
  });
}

/** Subscribe to tracker changes. Returns an unsubscribe. */
export function subscribeJobTracker(listener: () => void): () => void {
  listeners.add(listener);
  bindStorageOnce();
  return () => {
    listeners.delete(listener);
  };
}

/** The current tracker snapshot (stable reference until it changes). */
export function getJobs(): TrackedJob[] {
  return ensureLoaded();
}

/**
 * React hook: the live tracker list, re-rendering on every mutation. Returns the
 * WHOLE list — do NOT filter here; consumers filter with useMemo in their own
 * body (filtering inside getSnapshot returns a fresh array each read → the
 * "getSnapshot should be cached" warning / an infinite render loop).
 */
export function useJobTracker(): TrackedJob[] {
  return useSyncExternalStore(subscribeJobTracker, getJobs, getJobs);
}

// Module init: rehydrate from sessionStorage and RESUME polling for every
// non-terminal record (fixes reload amnesia). Non-capped in-flight records are
// re-anchored (startedAt = now) so a reload gives them a fresh poll window rather
// than instantly tripping the cap; capped records are left capped (they show the
// resume affordance and DON'T auto-resume); expired/terminal records are left
// as-is. Runs once on import — the hooks + sidebar all import this module, so it
// always loads on boot.
function initTracker(): void {
  if (typeof window === "undefined") return;
  bindStorageOnce();
  const loaded = ensureLoaded();
  let mutated = false;
  const now = Date.now();
  const next = loaded.map((rec) => {
    if (isJobActive(rec)) {
      mutated = true;
      return { ...rec, startedAt: now };
    }
    return rec;
  });
  if (mutated) {
    cache = next;
    writeStorage(cache);
  }
  armPolling();
}

initTracker();
