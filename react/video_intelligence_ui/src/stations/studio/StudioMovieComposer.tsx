// The Studio MOVIE authoring surface (slice B2 — UI only).
//
// A studio movie is an ordered strip of REAL studio clips conjoined at splice
// points, like a single NLE timeline ROW: `[segment 0 | segment 1 | …]` (see
// video_intel/studio_movie_schema.py). This surface AUTHORS that strip and plays
// it back — it is the Movie sibling of StudioGenerateTab's single-clip Clip
// surface, mounted by the mode switch at the top of StudioGenerateSurface.
//
// V0/B3 BOUNDARY (stated plainly, mirroring the schema/runner headers):
//   • Authoring + playback + (B3) a VISUAL SPLICED ROW & branch-frame scrubber. After a
//     movie renders, StudioSplicedRow draws the cut (blocks sized to EFFECTIVE frames,
//     scissor notch per splice) and a click scrubs any segment to branch a new take from
//     a chosen frame; you can still TYPE a branch_frame per goal below (blank = the
//     parent's last frame). The take-tree UI (sibling divergence in place) is later — see
//     onBranchFromSegment's truncate note.
//   • The take-tree parent_segment_id field EXISTS in the schema (sibling
//     divergence is planned growth) but v0 authors a LINEAR CHAIN only — so we send
//     neither segment_id nor parent_segment_id and let the route auto-fill "seg_NN"
//     + the linear parent chain (video_routes.py video_studio_movie).
//   • Movie-level start_image (segment 0 renders i2v from a still instead of t2v) is now
//     WIRED (B3 stretch) behind a MOVIE START SWITCHER (operator ask 2026-07-12 — "any
//     library clip, image, or scene to start — or start fresh with a prompt"): the
//     operator picks segment 1's start from the session LIBRARY, UPLOADS a fresh still
//     (the original upload→ingest→MediaRef idiom), or clears it back to a FRESH PROMPT
//     (t2v). "Start from a CLIP" is shown but DISABLED with an honest tooltip: the
//     backend schema has no `start_clip` field, and there is no synchronous "give me
//     this clip's last frame" endpoint on the client — frame_extract is its own ASYNC
//     JOB (enqueue → poll → pick a frame from a paginated grid), so deriving a start
//     still from a clip client-side is not a small addition here. The button names the
//     backend gap instead of faking a still. Empty start_image = segment 0 t2v.
//   • Per-segment negative/model/steps/cfg are schema-overridable; v0 keeps those
//     movie-level (one negative field) and per-segment only what the strip needs
//     to read as a timeline: prompt, an optional seed, and (rows>0) a branch_frame.
//
// The request body this surface POSTs mirrors the route's field names EXACTLY
// (resolution{width,height,fps}, seed, vram_budget_gb, negative_prompt, goals:
// [{prompt, seed?, branch_frame?}]) — every optional dropped by JSON.stringify when
// unset. The job is polled on the SAME generic GET /video/jobs/<id> the studio flow
// uses; on done the assembled movie.mp4 (outputs[-1]) plays in the viewer idiom and
// the segment clips list below with their honest joint metadata.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
// Sub-tab input memory — see the useSessionState block at the top of the
// composer and the rationale in video/sessionForm.ts.
import {
  nextSessionSeq,
  useSessionRef,
  useSessionState,
} from "../../video/sessionForm";
import { createPortal } from "react-dom";
import { z } from "zod";
import {
  request,
  okValue,
  errorOf,
  describeAppError,
} from "../../transport/client";
import { hugpyConfig, mediaBytesUrl, jobStatusUrl } from "../../config";
import {
  mediaRefSchema,
  uploadResultSchema,
  jobErrorSchema,
  JOB_STATUSES,
  isTerminal,
  enqueueResultSchema,
  type MediaRef,
  type JobStatus,
} from "../../video/contract";
import { getSessionId } from "../../session";
import { addToLibrary, useMediaLibrary } from "../../video/mediaLibrary";
import { UnnamedWarnPanel, WarnUnnamedToggle, useWarnUnnamedPref } from "../../video/UnnamedWarn";
import { LengthRow } from "../../video/LengthRow";
import { GoalComposer } from "../GoalComposer";
import { usePromptAssist, type AssistExtras } from "../../video/usePromptAssist";
import { PromptAssistButtons } from "../../video/PromptAssistButtons";
// The SHARED prompt-card system (k88): toolbar, row shell, [Prompt][Negative]
// tabs and the settings expander this composer and the Movie timeline both
// render, plus the standard-negative composition helper.
import {
  AboutExpander,
  PromptCard,
  PromptCardSettings,
  PromptFieldTabs,
  PromptListToolbar,
  composeNegative,
} from "./promptCard";
// STUDIO SPREAD (STUDIO-SPREAD-SPEC §2) — the pure half: wire shapes, the ONE
// spread-request builder, the tolerant readers and the intent→action table. Kept
// beside the hook rather than inlined here so this component stays an editor.
import {
  assistModeFor,
  buildSpreadBody,
  detectionText,
  operationLabel,
  resolveOperation,
  segmentRef,
  spreadNoticeLines,
  type IdentityContextWire,
  type IntentResult,
  type PromptFieldMode,
  type SpreadGoalInput,
} from "../../video/spreadAssist";
// k121 COORDINATION REVIEW — the words-vs-knobs check that rides every spread.
// Reader + badge live in their own files so this component stays an editor.
import {
  knobPatch,
  readCoordinationReport,
  segmentReview,
  type CoordinationDecision,
  type CoordinationReport,
} from "../../video/coordinationReport";
import { CoordinationBadge, CoordinationSummary, decisionKey } from "./coordinationBadge";
// B3: the visual spliced row + branch-frame scrubber (the picture half). Pure math
// lives in movieTimeline.ts; both are new files beside this composer.
import { StudioSplicedRow, StudioPendingRow } from "./StudioSplicedRow";
// Movie start switcher (operator ask 2026-07-12): the SAME session-library
// thumbnail grid the Clip surface's start/reference/control pickers use, extracted
// into studioShared.tsx — see the "Movie start" section below.
import {
  LibraryImageGrid,
  STUDIO_MODEL_IDS,
  STUDIO_REAL_FLOOR_GB,
  STUDIO_SAFE_FILL_GB,
  bindsSyntheticTier,
} from "./studioShared";
// COMPATIBILITY AWARENESS (2026-07-29) — the same measured render table the Clip surface
// reads (one session-cached GET, shared via the hook's module singleton). It supplies the
// movie's generalized workflow presets, its geometry envelope, its budget floor and the
// model pins that can actually serve a segment.
import { useRenderPresets } from "../../video/useRenderPresets";
import {
  MOVIE_CAPABILITY,
  movieWorkflowPresets,
  movieBudgetGb,
  geometryForCapability,
  modelsForCapability,
  withinGeometry,
  suggestedBudgetGb,
} from "../../video/renderCompat";
// IDENTITY PROFILES (stage a): the shared "Identity" control block + the hook (the
// composer also reads the saved list so a template's `profile` slug can auto-attach).
import { IdentityProfileControls, type SelectedProfile } from "./IdentityProfileControls";
import { useIdentityProfiles } from "./useIdentityProfiles";
// CINEMA SESSIONS (k91): the durable movie sessions on the studio-movies root (resume /
// pause / watch what rendered), plus the shared live-pace line this composer reuses for
// its OWN in-flight render. Both live next door so there is one renderer for each.
// Cinema sessions moved to the Active display's Sessions panel (operator ask
// 2026-08-12) — this file keeps only the live-pace renderer for its own render.
import { MovieLiveProgress } from "./StudioMovieSessions";

// B3: the movie endpoint is now a promoted config literal (mirrors studioI2VEnqueueUrl),
// replacing the earlier inline `${hugpyConfig.apiBase}/video/studio/movie` derivation.
// jobStatusUrl for the poll is likewise imported from config.
const studioMovieUrl = hugpyConfig.studioMovieUrl;

// ── local wire schemas ──────────────────────────────────────────────────────
// The studio-movie job view does NOT fit contract.ts's movieProgressSchema /
// jobRecordSchema (those were shaped for the OTHER generate_movie flow, whose
// progress `current` is a flat JobProgress; the studio runner's `current` is a
// small {segment_id, …} dict). So we parse the studio-movie shapes with our own
// tolerant/passthrough schemas here, reusing the shared mediaRefSchema /
// jobErrorSchema / JOB_STATUSES / enqueueResultSchema from the contract. Every
// field beyond the ones we read is optional — a drifted-forward payload never
// breaks the parse.

// One node in the movie.json manifest's `segments` (the runner's seg_records).
const manifestSegmentSchema = z
  .object({
    index: z.number(),
    segment_id: z.string().nullable().optional(),
    parent_segment_id: z.string().nullable().optional(),
    prompt: z.string().nullable().optional(),
    capability: z.string().nullable().optional(),
    seed: z.number().nullable().optional(),
    // the AUTHORED branch value (may be null = "the parent's last frame").
    branch_frame: z.number().nullable().optional(),
    // the RESOLVED frame index into the PARENT (null for the root segment).
    resolved_branch: z.number().nullable().optional(),
    // how this segment was spliced onto its parent: "still" | "vace_extend" | "cut".
    joint_mode: z.string().nullable().optional(),
    frames: z.number().nullable().optional(),
    width: z.number().nullable().optional(),
    height: z.number().nullable().optional(),
    duration_s: z.number().nullable().optional(),
    resumed: z.boolean().nullable().optional(),
    status: z.string().nullable().optional(),
  })
  .passthrough();
type ManifestSegment = z.infer<typeof manifestSegmentSchema>;

const manifestJointSchema = z
  .object({
    parent_segment_id: z.string().nullable().optional(),
    child_segment_id: z.string().nullable().optional(),
    branch_frame: z.number().nullable().optional(),
    trim_frames: z.number().nullable().optional(),
    // splice mode: "still" | "vace_extend" | "cut" (drives the row notch label).
    mode: z.string().nullable().optional(),
  })
  .passthrough();

// The `movie` manifest block the runner echoes on JobResult (from movie.json).
const movieManifestSchema = z
  .object({
    kind: z.string().nullable().optional(),
    drift: z.string().nullable().optional(),
    fps: z.number().nullable().optional(),
    width: z.number().nullable().optional(),
    height: z.number().nullable().optional(),
    vram_budget_gb: z.number().nullable().optional(),
    segments: z.array(manifestSegmentSchema).optional(),
    joints: z.array(manifestJointSchema).optional(),
    assembly: z
      .object({
        movie: z.string().nullable().optional(),
        total_frames: z.number().nullable().optional(),
      })
      .passthrough()
      .nullable()
      .optional(),
    partial: z.boolean().nullable().optional(),
    segments_completed: z.number().nullable().optional(),
    segments_total: z.number().nullable().optional(),
  })
  .passthrough();

const movieJobResultSchema = z
  .object({
    ok: z.boolean(),
    outputs: z.array(mediaRefSchema).optional(),
    error: jobErrorSchema.nullable().optional(),
    movie: movieManifestSchema.nullable().optional(),
    project: z.record(z.unknown()).nullable().optional(),
  })
  .passthrough();
type MovieJobResult = z.infer<typeof movieJobResultSchema>;

// One segment's live entry in the nested progress (the runner's segments_meta).
const progressSegmentSchema = z
  .object({
    index: z.number(),
    segment_id: z.string().nullable().optional(),
    prompt: z.string().nullable().optional(),
    status: z.string().nullable().optional(),
    resumed: z.boolean().nullable().optional(),
    // k120 slice 2 — set when continuity refresh rewrote this segment's prompt:
    // the authored original + the runner's one-line note (what the previous
    // segment's closing frame actually showed, or why the rewrite was skipped).
    prompt_authored: z.string().nullable().optional(),
    refresh_note: z.string().nullable().optional(),
  })
  .passthrough();

const movieProgressSchema = z
  .object({
    stage: z.string().nullable().optional(),
    segment_done: z.number().nullable().optional(),
    segment_total: z.number().nullable().optional(),
    segments: z.array(progressSegmentSchema).optional(),
    // `current` is a small {segment_id, index?, prompt?, capability?, branch_frame?}
    // dict OR null — read defensively, never trusted to a fixed shape.
    current: z.record(z.unknown()).nullable().optional(),
    started_at: z.number().nullable().optional(),
    eta_s: z.number().nullable().optional(),
  })
  .passthrough();
type MovieProgress = z.infer<typeof movieProgressSchema>;

const movieJobSchema = z
  .object({
    job_id: z.string(),
    status: z.enum(JOB_STATUSES).nullable(),
    result: movieJobResultSchema.nullable().optional(),
    progress: movieProgressSchema.nullable().optional(),
  })
  .passthrough();

// ── the authored goal row (local editor state) ──────────────────────────────
// Numbers are kept as STRINGS so "blank = unset" is distinct from 0 (the clip
// surface's idiom): a blank per-goal seed lets the runner derive `seed + index`;
// a blank branch_frame means "the parent's last frame".
// How a segment (row >= 1) is spliced onto the one before it:
//   • "still"       — continue: condition i2v on ONE branch frame (motion NOT carried).
//   • "vace_extend" — extend motion: condition on the parent's trailing frames (VACE).
//   • "cut"         — scene cut: NO frame carry, a fresh render; the parent plays in full
//                     (in an identity movie the SUBJECT still carries via the references).
// Row 0 (the root) has no parent, so its `joint` is ignored.
type JointMode = "still" | "vace_extend" | "cut";

interface GoalRow {
  key: string;
  prompt: string;
  // Per-segment negative (operator addendum 2026-07-12). Blank ⇒ the goal body omits
  // `negative`, and the runner falls back to the movie-level negative (schema:
  // StudioMovieGoal.negative, None ⇒ movie default). Non-blank overrides it for this
  // segment only. Kept as a STRING so "blank = fall back" is distinct from an empty
  // override, matching the seed/branchFrame idiom.
  negative: string;
  // k88: [standard negative] tick — when on, the STANDARD artifact/junk set is
  // COMPOSED into this row's effective negative at the point a request payload
  // (or spread projection) is built. Client-side only; the wire is unchanged.
  stdNegative: boolean;
  seed: string;
  branchFrame: string;
  joint: JointMode;
  // FULL PER-SEGMENT KNOBS (operator ask 2026-08-12): the cinema route has
  // accepted per-goal model_id/steps/cfg/context_frames overrides all along
  // (StudioMovieGoal — "honored when set, not dead fields"); the editor just
  // never offered them. Same string idiom as seed/branchFrame: blank ⇒ the key
  // is dropped from the goal body and the segment inherits the movie-level
  // value (or the router/model default).
  modelId: string;
  steps: string;
  cfg: string;
  contextFrames: string;
  // per-segment CLIP LENGTH in frames (2026-08-13); blank = movie default.
  frames: string;
}

// The counter is SESSION-persisted (not a module `let`) now that the goal rows
// themselves survive a reload: a module counter restarts at 0 while the restored
// rows keep their old keys, so `goal_1` could be minted twice — and this key
// doubles as the wire `segment_id`. Same shape, just a counter that remembers.
// The real-model frame ceiling (Wan family: 81) — the spine clamps to this
// and snaps to 4k+1. Split-to-fit uses it to materialize CONTINUATION
// segments instead of letting a longer ask silently degrade.
const MODEL_FRAME_CEILING = 81;

function newGoalKey(): string {
  return `goal_${nextSessionSeq("studio.movie.goalKeySeq")}`;
}
// ── CINEMA PRODUCER (k120 slice 1) — POST /video/producer/plan reply shape.
// The server owns validation/repair; this guard only refuses a reply the row
// mapper below could not consume.
const producerPlanSchema = z.object({
  title: z.string().optional(),
  logline: z.string().optional(),
  total_seconds: z.number().optional(),
  segments: z
    .array(
      z.object({
        prompt: z.string(),
        negative: z.string().optional(),
        seconds: z.number(),
        joint: z.string(),
      }),
    )
    .min(1),
});

function blankGoal(joint: JointMode = "still"): GoalRow {
  return {
    key: newGoalKey(),
    prompt: "",
    negative: "",
    stdNegative: true,
    seed: "",
    branchFrame: "",
    joint,
    modelId: "",
    steps: "",
    cfg: "",
    contextFrames: "",
    frames: "",
  };
}

// ── the poll hook (local) ───────────────────────────────────────────────────
// StudioGenerateTab's Clip surface never polls — it enqueues and lets the durable
// /video/studio/clips catalog surface the clip. But that catalog lists ONLY
// studio_i2v jobs, so a movie's outputs come back on GET /video/jobs/<id>. This
// small hook mirrors the enqueue→track idiom (useGenerateMovieJob) but keeps its
// OWN poll loop against jobStatusUrl since no shared tracker owns this kind.
interface StudioMovieJobApi {
  jobId: string | null;
  status: JobStatus;
  progress: MovieProgress | null;
  result: MovieJobResult | null;
  /** Human message on enqueue error or a failed/cancelled job. */
  error: string | null;
  running: boolean;
  run: (body: unknown) => void;
  reset: () => void;
}

function useStudioMovieJob(): StudioMovieJobApi {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus>(null);
  const [progress, setProgress] = useState<MovieProgress | null>(null);
  const [result, setResult] = useState<MovieJobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const mounted = useRef(true);
  const timer = useRef<number | null>(null);
  const jobRef = useRef<string | null>(null);

  const stopPoll = useCallback(() => {
    if (timer.current != null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (timer.current != null) window.clearInterval(timer.current);
    };
  }, []);

  const poll = useCallback(async () => {
    const id = jobRef.current;
    if (!id) return;
    const res = await request<unknown>(jobStatusUrl(id), {
      meta: { specKey: "studio", operation: "studio.movie.status" },
    });
    if (!mounted.current) return;
    // A transient poll failure must NOT wipe a working readout — just wait for the
    // next tick (the terminal parse below stops the loop on its own).
    if (!res.ok) return;
    const parsed = movieJobSchema.safeParse(okValue(res));
    if (!parsed.success) return;
    const jr = parsed.data;
    setStatus(jr.status ?? null);
    if (jr.progress) setProgress(jr.progress);
    if (jr.result) setResult(jr.result);
    if (isTerminal(jr.status)) {
      stopPoll();
      setRunning(false);
      if (jr.status === "failed" || jr.status === "cancelled") {
        setError(jr.result?.error?.message ?? `Job ${jr.status}.`);
      }
    }
  }, [stopPoll]);

  const run = useCallback(
    (body: unknown) => {
      stopPoll();
      setJobId(null);
      jobRef.current = null;
      setStatus(null);
      setProgress(null);
      setResult(null);
      setError(null);
      setRunning(true);
      void request<unknown>(studioMovieUrl, {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "studio.movie.enqueue" },
      }).then((r) => {
        if (!mounted.current) return;
        if (!r.ok) {
          setError(describeAppError(errorOf(r)));
          setStatus("failed");
          setRunning(false);
          return;
        }
        const p = enqueueResultSchema.safeParse(okValue(r));
        if (!p.success) {
          setError("Malformed enqueue response.");
          setStatus("failed");
          setRunning(false);
          return;
        }
        const id = p.data.job_id;
        setJobId(id);
        jobRef.current = id;
        setStatus("queued");
        void poll();
        timer.current = window.setInterval(() => void poll(), 2500);
      });
    },
    [poll, stopPoll],
  );

  const reset = useCallback(() => {
    stopPoll();
    setJobId(null);
    jobRef.current = null;
    setStatus(null);
    setProgress(null);
    setResult(null);
    setError(null);
    setRunning(false);
  }, [stopPoll]);

  return { jobId, status, progress, result, error, running, run, reset };
}

// ── honest joint copy for one produced segment (uses the manifest, no invention) ─
// Root: "root · <capability>". A later segment: "branched from frame N · trimmed
// parent to N+1" (exactly the requested wording), with honest annotations when the
// authored branch was blank (parent's last frame) or the trim keeps the parent
// whole (no real cut).
function jointText(segs: ManifestSegment[], i: number): string {
  const seg = segs[i];
  if (!seg) return "";
  const cap = seg.capability ?? "t2v";
  if (i === 0) return `root segment · ${cap}`;
  const mode = seg.joint_mode ?? "still";
  // A "cut" carries no frame — the parent plays in full and the child is a fresh render
  // (id_lock in an identity movie, so the subject carries even though no pixels do).
  if (mode === "cut") {
    return `✂ scene cut · fresh ${cap} render — no frame carry (parent plays in full)`;
  }
  const rb = seg.resolved_branch;
  if (rb == null) return "branched from parent";
  const trim = rb + 1;
  const parentFrames = segs[i - 1]?.frames;
  const carry = mode === "vace_extend" ? "extended motion from" : "branched from";
  let s = `${carry} frame ${rb} · trimmed parent to ${trim}`;
  if (seg.branch_frame == null) s += " (blank branch = parent's last frame)";
  if (parentFrames != null && trim >= parentFrames) s += " — parent plays in full";
  return s;
}

function statusBadge(status: string | null | undefined, resumed?: boolean | null): string {
  if (resumed) return "resumed";
  return status ?? "pending";
}

// ── movie PRESETS — GENERALIZED WORKFLOWS, not canned scenes (operator ruling 2026-07-29)
//
// WHAT THIS REPLACES AND WHY. This slot used to hold a `TEMPLATES` const of two
// scene-specific prefills — "Beach & volleyball — character day out", carrying a baked-in
// subject ("a young woman in a light sundress"), a baked-in art direction ("anime
// illustration style"), a baked-in identity slug ("mira") and two baked-in scene prompts.
// Applying one REPLACED the operator's entire goal timeline with that content. Three
// problems, all of them the same problem:
//   • it answered ONE user's request rather than describing a workflow, so anyone whose
//     movie was not about a beach had to delete the preset's prose before starting;
//   • its numbers (480×480, budget 10) were hand-picked against a fleet fact and could
//     drift out from under it silently;
//   • and it clobbered the user's own session inputs, which is the defect class this
//     whole slice deletes.
//
// A preset here now names a WORKFLOW SHAPE and nothing about the content: which JOINT
// stitches the segments, what geometry the fleet has ratified for it, what that joint's
// binding really costs, and whether it has ever completed here. The CONTENT is the user's
// — their goal prompts, their identity, their start image — and `applyWorkflow` slots the
// workflow around all of it without touching any of it.
//
// SOURCED FROM THE BACKEND. The list is derived by `movieWorkflowPresets` from the
// ratified `movie-480p` RenderPreset: one row per joint mode the preset declares
// (`joints`), sized against the segment binding that joint actually uses (found via
// `composes`), and badged `proven` by the same rule the backend enforces on itself — a
// composite can never be more proven than the weakest thing it composes. Ratifying a
// fourth joint server-side makes a fourth workflow appear here with no frontend change.
//
// IDENTITY (operator 2026-07-12, unchanged and now the model for everything else):
// applying a preset NEVER clobbers an attached identity. That instinct was always right
// here; it is now the rule for every field.

// ── movie templates (RESTORED 2026-08-01, operator ask) ────────────────────────────
// The lightweight scene-PREFILL mechanism removed by e43d9e7 in favor of workflow
// presets. The operator wants BOTH: a workflow preset sets the SHAPE (joint/geometry/
// budget) and never touches prompts; a movie template prefills the SCENES themselves —
// goal prompts + joints + optional seed offset, resolution, budget, and an optional
// saved identity profile. Same rules as before: applying a template NEVER clobbers an
// already-attached identity (auto-attach only into an empty slot, from a SAVED profile
// the template names), and the movie START is left untouched.
interface MovieTemplate {
  id: string;
  name: string;
  hint: string;
  goals: Array<{ prompt: string; joint: JointMode; seed?: string }>;
  /** Prefill the movie resolution (the vace-1.3b id sweet spot is 480×480 / 832×480). */
  resolution?: { width: number; height: number };
  /** Prefill the VRAM budget field (a real id-capable model needs a real budget). */
  vramBudgetGb?: number;
  /** A saved identity-profile slug — auto-attached on apply when saved (else: nudge). */
  profile?: string;
}
const TEMPLATES: MovieTemplate[] = [
  {
    id: "character-short-two-scenes",
    name: "Character short — two scenes",
    hint:
      "Add one identity reference below (or pick a saved profile), then Generate: your character on a sunny beach, then a scene cut to playing volleyball.",
    goals: [
      { prompt: "on a sunny beach, gentle waves, warm golden-hour light", joint: "still" },
      { prompt: "playing volleyball on the sand, mid-jump, dynamic action", joint: "cut" },
    ],
  },
  {
    id: "beach-volleyball-character-day-out",
    name: "Beach & volleyball — character day out",
    hint:
      "One click: a young woman on a sunny beach, then a scene cut to beach volleyball. Attaches the saved “Mira” identity if you have one; otherwise add a reference (or pick a profile) below, then Generate.",
    resolution: { width: 480, height: 480 },
    vramBudgetGb: 10,
    profile: "mira",
    goals: [
      {
        prompt:
          "the same young woman in a light sundress on a sunny beach by the water, ocean waves and bright sky behind her, hair moving in the breeze, warm golden-hour light, anime illustration style, high detail",
        joint: "still",
      },
      {
        prompt:
          "the same young woman playing beach volleyball, mid-action leaping to spike the ball, sand kicking up, dynamic angle, sunny beach court, anime illustration style, high detail",
        joint: "cut",
        seed: "1000",
      },
    ],
  },
];

const JOINT_LABELS: Record<JointMode, string> = {
  still: "continue (still)",
  vace_extend: "extend motion (vace)",
  cut: "scene cut",
};

export interface StudioMovieComposerProps {
  /** The sidebar "Settings" tab host to portal the Movie-templates dropdown into (mirrors
   *  the Clip surface's knobGrid portal). null until the sidebar mounts its Settings panel. */
  settingsHost?: HTMLElement | null;
}

// ── the composer ────────────────────────────────────────────────────────────
export function StudioMovieComposer({ settingsHost }: StudioMovieComposerProps = {}) {
  // ── SUB-TAB INPUT MEMORY (operator ask 2026-08-06) ──────────────────────────
  // This composer unmounts on every Cinema→Clip switch and every workbench tab
  // switch, which used to throw away an entire written-out goal timeline. Every
  // OPERATOR-AUTHORED field below therefore uses `useSessionState` (same tuple as
  // useState, value outlives the mount, per-tab, session-scoped) — see
  // video/sessionForm.ts. Spread/assist bookkeeping, upload phases, errors and
  // in-flight job state stay on plain useState on purpose.
  //
  // Goal timeline — at least one row; row 0 is the root (t2v).
  const [goalsRaw, setGoals] = useSessionState<GoalRow[]>("studio.movie.goals", () => [
    blankGoal(),
    blankGoal(),
  ]);
  // Rows restored from a session written BEFORE the per-segment knob round
  // (2026-08-12) lack modelId/steps/cfg/contextFrames — normalize to the blank
  // ("inherit") value so `.trim()` never meets undefined. useMemo, not a write:
  // the stored rows heal on the next real edit.
  const goals = useMemo(
    () =>
      goalsRaw.map((g) => ({
        modelId: "",
        steps: "",
        cfg: "",
        contextFrames: "",
        frames: "",
        ...g,
      })),
    [goalsRaw],
  );

  // Movie-level knobs (mirror the clip surface's defaults + idiom where reusable).
  // Default 480×480: the wan2.1-vace-1.3b IDENTITY sweet spot — the only budget-fitting
  // id-capable model on this fleet tops out at 832×480, so a 512×512 default cannot serve
  // id_lock on it (no_capable_model). 832×480 widescreen is the other valid id envelope.
  // EXPLICIT SEGMENT LENGTH (operator 2026-08-13): movie-level default frames
  // per segment; a segment's own frames (below) overrides. Blank = each bound
  // model's default (81 real). Clamped + 4k+1-snapped by the spine.
  const [framesDefault, setFramesDefault] = useSessionState("studio.movie.frames", "");
  const [width, setWidth] = useSessionState("studio.movie.width", 480);
  const [height, setHeight] = useSessionState("studio.movie.height", 480);
  const [fps, setFps] = useSessionState("studio.movie.fps", 24);
  const [seed, setSeed] = useSessionState("studio.movie.seed", 0);
  // blank = autofit to the card's capacity
  const [vramBudget, setVramBudget] = useSessionState("studio.movie.vramBudget", "");
  const [negative, setNegative] = useSessionState("studio.movie.negative", "");
  const [project, setProject] = useSessionState("studio.movie.project", "");
  // MOVIE NAME (k91) — the spec's `title`, display-only on the wire and written into
  // movie.json's manifest, which is what the sessions panel above lists a movie BY. An
  // unnamed movie is still perfectly renderable (the route takes title=None), so a blank
  // name warns once and offers submit-anyway rather than blocking — but a root full of
  // "untitled · 7f3a91c2…" is exactly the state that made lost renders unrecoverable, so
  // the warning is worth the one extra click. Session-persisted like every other authored
  // field here.
  const [movieTitle, setMovieTitle] = useSessionState("studio.movie.title", "");
  // Movie-level MODEL PIN + sampler overrides (operator ask 2026-07-13: the Settings tab
  // wants "the model selection" + a wider range of options). All BLANK-first ⇒ router
  // default: modelId "" = the capability router picks by budget/resolution (an unroutable
  // pin comes back as errors-as-data, never a silent fallback); steps/cfg "" = the model
  // default. Kept as STRINGS so "blank = unset" is distinct from 0, matching the clip
  // surface's idiom. The movie route validates these at the movie level (studio_movie_schema
  // _check_steps_cfg: steps int [1,100], cfg number [0,20]) — mirrored in buildBody below.
  const [modelId, setModelId] = useSessionState("studio.movie.modelId", "");
  const [steps, setSteps] = useSessionState("studio.movie.steps", "");
  const [cfg, setCfg] = useSessionState("studio.movie.cfg", "");
  // MERGE-NOT-OVERWRITE bookkeeping — true once the OPERATOR has typed a budget. A
  // workflow preset then fills nothing over it. A ref, not state: it only ever gates a
  // write and must not cause a render. Session-scoped so it is restored WITH the
  // budget it guards (see the clip surface's matching note).
  const budgetTouched = useSessionRef("studio.movie.budgetTouched", false);
  // Advanced escape hatch for the model pin (mirrors the Clip surface): off by default so
  // the menu is the ratified set for a movie segment; on to reach every studio model id,
  // each labelled unverified rather than presented as equivalent.
  const [showAllModels, setShowAllModels] = useState(false);

  // ── the fleet's measured render table ──────────────────────────────────────────────
  // Session-cached and shared with the Clip surface. `renderLoaded` gates every filter
  // below so a slow discovery GET degrades to this composer's prior behavior rather than
  // to an empty picker.
  const {
    presets: renderPresets,
    loaded: renderLoaded,
    renderBox,
  } = useRenderPresets(true);
  const workflowPresets = useMemo(
    () => movieWorkflowPresets(renderPresets),
    [renderPresets],
  );
  const movieGeometry = renderLoaded
    ? geometryForCapability(renderPresets, MOVIE_CAPABILITY)
    : null;
  const movieModels = renderLoaded ? modelsForCapability(renderPresets, MOVIE_CAPABILITY) : [];

  const [formMsg, setFormMsg] = useState<string | null>(null);
  // B3: key of a just-appended branch goal to focus once it mounts (empty prompt).
  const [pendingFocusKey, setPendingFocusKey] = useState<string | null>(null);

  // B3 stretch: OPTIONAL movie-level start_image for SEGMENT 0. When set, the root
  // renders i2v from this still instead of t2v (schema field `start_image`). Reuses the
  // clip surface's upload→ingest→guard-kind idiom verbatim, importing only config /
  // contract / session — it touches NO forbidden file.
  const [startImage, setStartImage] = useSessionState<MediaRef | null>(
    "studio.movie.startImage",
    null,
  );
  const [startPhase, setStartPhase] = useState<"idle" | "uploading" | "ingesting">("idle");
  const [startError, setStartError] = useState<string | null>(null);
  const startBusy = startPhase !== "idle";
  // Movie start switcher (operator ask 2026-07-12): toggles the session-library
  // picker open/closed, mirroring the Clip surface's showLibPicker idiom.
  const [showStartLibPicker, setShowStartLibPicker] = useState(false);

  // IDENTITY LOCK: the movie-level subject reference image(s). When any are set the movie is
  // an IDENTITY MOVIE — every segment renders id_lock so the locked subject carries across
  // scene changes (the operator's "take that id and use it for a video"). Up to 4 (diffusers
  // 0.39 consumes each as a VACE reference latent). Reuses the same upload→ingest→guard-kind
  // idiom as the start-image picker; the `.uri`s are threaded into the enqueue.
  const [references, setReferences] = useSessionState<MediaRef[]>(
    "studio.movie.references",
    [],
  );
  const [refPhase, setRefPhase] = useState<"idle" | "uploading" | "ingesting">("idle");
  const [refError, setRefError] = useState<string | null>(null);
  const refBusy = refPhase !== "idle";
  const REF_MAX = 4;
  // IDENTITY PROFILES (stage a): when the identity came from a SAVED profile, this holds
  // it — the enqueue then sends `identity_profile:<slug>` (canonical) and omits raw refs.
  // Any ad-hoc edit (upload/remove) clears it → the set becomes an "unsaved identity".
  const [selectedProfile, setSelectedProfile] = useSessionState<SelectedProfile | null>(
    "studio.movie.selectedProfile",
    null,
  );
  // Profiles read for the restored movie TEMPLATES (2026-08-01): a template may name a
  // saved identity slug and auto-attach it on apply — that needs the saved list + a ref
  // resolver here. A one-off stale read is fine for template-apply (same as pre-e43d9e7).
  const { profiles: savedProfiles, resolveRefs: resolveProfileRefs } = useIdentityProfiles();
  // ⚠ (2026-07-29 note, now partially reversed by the templates restore above.) It existed for ONE reason: the old
  // scene template named a saved identity slug ("mira") and auto-attached it on apply. A
  // generalized workflow preset names no subject at all — the identity is the operator's
  // session input, attached through IdentityProfileControls below (which owns its own
  // instance of the hook) — so the composer no longer needs a second, staler copy of the
  // profile list to resolve a slug nobody typed.
  // A profile pick REPLACES the identity (a profile IS the whole identity); ad-hoc appends.
  const attachIdentityRefs = useCallback((refs: MediaRef[], replace: boolean) => {
    if (replace) {
      setReferences(refs.slice(0, REF_MAX));
    } else {
      setReferences((prev) => {
        const room = REF_MAX - prev.length;
        return room <= 0 ? prev : [...prev, ...refs.slice(0, room)];
      });
    }
  }, []);

  const job = useStudioMovieJob();

  // LLM PROMPT-ASSIST (Enhance / Generate) — the SAME shared hook + button cluster the
  // Generate + Clip surfaces use (POSTs promptAssistUrl). A movie has a prompt PER GOAL,
  // so the cluster is rendered under each goal's prompt below: Enhance enriches that
  // goal's text, Generate writes a fresh one for it. ONE hook instance guards the whole
  // composer to one in-flight assist at a time (across all rows), matching the doctrine;
  // context.kind = "movie" so the assistant leans into scene/motion phrasing.
  const assist = usePromptAssist({ kind: "movie", specKey: "studio" });

  // ── STUDIO SPREAD state (STUDIO-SPREAD-SPEC §2) ────────────────────────────
  //
  // WHAT THIS FIXES. Per-row Generate was actively INCOHERENT: the assist route
  // re-randomizes its steering axes on every call, so six rows produced six unrelated
  // worlds. A spread is ONE call that writes every SELECTED row against one shared
  // steering set, with the UNSELECTED rows sent as locked context — so the shots land
  // in the same movie instead of six of them.
  //
  // The three pieces of state that makes possible, all keyed by the goal row's own
  // `key` (which doubles as the wire `segment_id` — stable across reorder/insert, so a
  // reply can never be applied to the wrong row):
  //   • `selectedKeys` — which rows the group actions may rewrite. NOTHING else is
  //     ever touched: unselected prompts, every negative the user typed, identity,
  //     images, seeds and the budget are all out of scope by construction.
  //   • `fieldModes` — the per-field Auto · Direction · Scene prompt tri-state.
  //     An explicit choice SKIPS the router entirely (§1d.2).
  //   • `intents` — the router's last answer per row. A LABEL on a button, never an
  //     action: no classification ever starts a generation on its own.
  const [selectedKeys, setSelectedKeys] = useState<ReadonlySet<string>>(() => new Set());
  // Keyed by goal row key, so it rides along with the persisted `goals`.
  const [fieldModes, setFieldModes] = useSessionState<Record<string, PromptFieldMode>>(
    "studio.movie.fieldModes",
    {},
  );
  const [intents, setIntents] = useState<Record<string, IntentResult | null>>({});
  // Which rows are mid-classification (a tiny "…" beside the control, not a blocker).
  const [intentBusy, setIntentBusy] = useState<Record<string, boolean>>({});
  // The text each row was last classified FOR, so a blur that changed nothing is free.
  const intentText = useRef<Record<string, string>>({});
  // The overall request the spread writes against ("a chase through a night market").
  // Optional: with it blank the generator works from the rows themselves.
  const [movieQuery, setMovieQuery] = useSessionState("studio.movie.query", "");
  // The steering set's seed, echoed by the last spread. Re-sending it puts a follow-up
  // spread into the SAME world — how you re-roll two shots without leaving the movie.
  const [steeringSeed, setSteeringSeed] = useState<number | null>(null);
  const [pinSteering, setPinSteering] = useState(true);
  // The non-destructive result notice: what changed, what did NOT, every warning.
  // Dismissible; it never edits anything on its own.
  const [spreadNotice, setSpreadNotice] = useState<string[] | null>(null);
  // k121: the last spread's coordination review, and the proposals the operator
  // has already answered (accepted or rejected). Both are VIEW state — the
  // report is a fact about the last generate, and re-generating replaces it.
  const [coordination, setCoordination] = useState<CoordinationReport | null>(null);
  const [coordSettled, setCoordSettled] = useState<Set<string>>(() => new Set());

  const selectedCount = useMemo(
    () => goals.reduce((n, g) => (selectedKeys.has(g.key) ? n + 1 : n), 0),
    [goals, selectedKeys],
  );
  const allSelected = goals.length > 0 && selectedCount === goals.length;

  const toggleSelected = useCallback((key: string) => {
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);
  // Keyed off the LIVE row count, not the set's size: a removed row can leave a stale
  // key behind, and "select all" must mean the rows on screen, not that residue.
  const toggleSelectAll = useCallback(() => {
    setSelectedKeys(() => (allSelected ? new Set<string>() : new Set(goals.map((g) => g.key))));
  }, [allSelected, goals]);
  // Toolbar [standard negatives] (k88): tick [standard negative] ON for the
  // selected rows — or for EVERY row when nothing is selected. Set, not toggle:
  // the control is a bulk "add the standard set", and untick stays per-row.
  const onStandardNegatives = useCallback(() => {
    setGoals((prev) => {
      const anySelected = prev.some((g) => selectedKeys.has(g.key));
      return prev.map((g) =>
        !anySelected || selectedKeys.has(g.key) ? { ...g, stdNegative: true } : g,
      );
    });
  }, [selectedKeys]);

  // Movie start switcher's library picker: the SAME session-library image list the
  // Clip surface's start/reference/control pickers read (useMediaLibrary), filtered
  // to images (a movie start is a single still, like the i2v start image).
  const library = useMediaLibrary();
  const startLibraryImages = library.filter((it) => it.ref.kind === "image");

  // Upload → ingest → guard kind=image → hold the MediaRef (its `.uri` is threaded into
  // the enqueue as `start_image`). Mirrors StudioGenerateSurface.onPickImageFile.
  const onPickStartImage = useCallback(async (file: File) => {
    setStartError(null);
    setStartPhase("uploading");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("sid", getSessionId());
    const up = await request<unknown>(hugpyConfig.uploadUrl, {
      method: "POST",
      body: fd,
      meta: { specKey: "studio", operation: "studio.movie.start_image.upload" },
    });
    if (!up.ok) {
      setStartError(describeAppError(errorOf(up)));
      setStartPhase("idle");
      return;
    }
    const upParsed = uploadResultSchema.safeParse(okValue(up));
    if (!upParsed.success) {
      setStartError("Malformed upload response.");
      setStartPhase("idle");
      return;
    }
    setStartPhase("ingesting");
    const ing = await request<unknown>(hugpyConfig.videoIngestUrl, {
      method: "POST",
      body: JSON.stringify({ path: upParsed.data.path }),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "studio", operation: "studio.movie.start_image.ingest" },
    });
    if (!ing.ok) {
      setStartError(describeAppError(errorOf(ing)));
      setStartPhase("idle");
      return;
    }
    const media = mediaRefSchema.safeParse(okValue(ing));
    if (!media.success) {
      setStartError("Malformed ingest response.");
      setStartPhase("idle");
      return;
    }
    if (media.data.kind !== "image") {
      setStartError(`Ingested asset is "${media.data.kind}", not an image — pick a still.`);
      setStartPhase("idle");
      return;
    }
    setStartImage(media.data);
    setStartPhase("idle");
  }, []);

  // Upload → ingest → guard kind=image → APPEND a subject reference (identity lock). Mirrors
  // onPickStartImage; caps at REF_MAX (a clean caller error, never a silent drop).
  const onPickReferenceImage = useCallback(async (file: File) => {
    setRefError(null);
    setReferences((prev) => {
      if (prev.length >= REF_MAX) setRefError(`At most ${REF_MAX} reference images.`);
      return prev;
    });
    setRefPhase("uploading");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("sid", getSessionId());
    const up = await request<unknown>(hugpyConfig.uploadUrl, {
      method: "POST",
      body: fd,
      meta: { specKey: "studio", operation: "studio.movie.reference.upload" },
    });
    if (!up.ok) {
      setRefError(describeAppError(errorOf(up)));
      setRefPhase("idle");
      return;
    }
    const upParsed = uploadResultSchema.safeParse(okValue(up));
    if (!upParsed.success) {
      setRefError("Malformed upload response.");
      setRefPhase("idle");
      return;
    }
    setRefPhase("ingesting");
    const ing = await request<unknown>(hugpyConfig.videoIngestUrl, {
      method: "POST",
      body: JSON.stringify({ path: upParsed.data.path }),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "studio", operation: "studio.movie.reference.ingest" },
    });
    if (!ing.ok) {
      setRefError(describeAppError(errorOf(ing)));
      setRefPhase("idle");
      return;
    }
    const media = mediaRefSchema.safeParse(okValue(ing));
    if (!media.success) {
      setRefError("Malformed ingest response.");
      setRefPhase("idle");
      return;
    }
    if (media.data.kind !== "image") {
      setRefError(`Ingested asset is "${media.data.kind}", not an image — pick a still.`);
      setRefPhase("idle");
      return;
    }
    setReferences((prev) => (prev.length >= REF_MAX ? prev : [...prev, media.data]));
    setSelectedProfile(null); // an ad-hoc upload makes this an UNSAVED identity
    setRefPhase("idle");
  }, []);

  // ── APPLYING A WORKFLOW PRESET IS A MERGE (operator ruling 2026-07-29) ──
  //
  // The preset supplies the WORKFLOW SLOTS — the joint that stitches the segments, the
  // ratified geometry, and a budget sized for the binding this joint actually uses. The
  // CONTENT is the operator's and survives untouched:
  //
  //   • every goal PROMPT stays exactly as written (the old path replaced the whole
  //     timeline with the template's own scene prose);
  //   • every per-goal SEED stays;
  //   • the identity — references or a saved profile — is never clobbered (that rule was
  //     already right here and is now the rule for everything);
  //   • the movie START (start switcher) is untouched, as it always was;
  //   • rows are only ADDED, never removed: a workflow needs at least two segments to
  //     mean anything, so an empty timeline grows to two and a five-row timeline stays
  //     five. Losing a shot the operator wrote is not a preset's business.
  //
  // A budget the operator TYPED is likewise left alone (`budgetTouched`); a blank or
  // preset-filled one is re-sized for the new joint, because that IS a workflow slot.
  const applyWorkflow = useCallback(
    (id: string) => {
      const wf = workflowPresets.find((x) => x.id === id);
      if (!wf) return;
      const joint = wf.joint as JointMode;
      setGoals((prev) => {
        // Grow to the workflow's minimum, preserving every existing row verbatim.
        const rows = prev.length >= wf.minSegments
          ? prev.slice()
          : [...prev, ...Array.from({ length: wf.minSegments - prev.length }, () => blankGoal())];
        // The joint is the workflow. Row 0 is the ROOT and has no joint onto anything.
        return rows.map((g, i) => (i === 0 ? g : { ...g, joint }));
      });
      if (wf.geometry) {
        setWidth(wf.geometry.width);
        setHeight(wf.geometry.height);
        setFps(wf.geometry.fps);
      }
      if (!budgetTouched.current) {
        const need = movieBudgetGb(renderPresets, {
          joint,
          identity: references.length > 0 || selectedProfile != null,
        });
        if (need != null) setVramBudget(String(need));
      }
      setFormMsg(null);
    },
    [workflowPresets, renderPresets, references.length, selectedProfile],
  );

  // ── template PREFILL (restored 2026-08-01; from the sidebar Settings-tab dropdown) ──
  // Picking a template prefills EVERYTHING it names: the goal rows (prompt + joint + an
  // optional seed offset), and — when carried — the resolution and vram budget. The movie
  // START is deliberately left untouched (operator's established rule). IDENTITY: applying
  // a template NEVER clobbers an already-attached identity; only when NONE is attached AND
  // the template names a SAVED profile is that profile auto-attached — otherwise the
  // existing "attach a reference" nudge stands.
  const applyTemplate = useCallback((id: string) => {
    const t = TEMPLATES.find((x) => x.id === id);
    if (!t) return;
    setGoals(t.goals.map((g) => ({ ...blankGoal(g.joint), prompt: g.prompt, seed: g.seed ?? "" })));
    if (t.resolution) {
      setWidth(t.resolution.width);
      setHeight(t.resolution.height);
    }
    if (t.vramBudgetGb != null) {
      setVramBudget(String(t.vramBudgetGb));
      budgetTouched.current = false; // preset-filled, not typed — workflows may re-size it
    }
    setFormMsg(null);
    // Never clobber an attached identity. Auto-attach only into an EMPTY identity slot
    // and only from a SAVED profile the template names.
    const hasIdentity = references.length > 0 || selectedProfile != null;
    if (!hasIdentity && t.profile) {
      const p = savedProfiles.find((x) => x.slug === t.profile);
      if (p) {
        void resolveProfileRefs(p).then((refs) => {
          if (refs.length) {
            setReferences(refs.slice(0, REF_MAX));
            setSelectedProfile({ slug: p.slug, name: p.name });
          }
        });
      }
    }
  }, [references.length, selectedProfile, savedProfiles, resolveProfileRefs]);

  // ── goal-row editing ──
  const setGoalField = useCallback(
    (idx: number, patch: Partial<GoalRow>) => {
      setGoals((prev) => prev.map((g, i) => (i === idx ? { ...g, ...patch } : g)));
    },
    [],
  );

  // k121 ACCEPT / REJECT for a `proposed` coordination decision. A proposal is
  // anything that REMOVES coordination (dropping a carry, shortening a clip,
  // swapping an identity) or that spends real work (an IDENTITY_CAPTURE) — the
  // backend deliberately never applies those itself, so this is where a human
  // says yes. Both answers are recorded, so a rejected proposal stops asking
  // rather than nagging on every re-render.
  const acceptCoordination = useCallback(
    (d: CoordinationDecision) => {
      const patch = knobPatch(d);
      if (patch) {
        setGoals((prev) =>
          prev.map((g) => (g.key === d.segment_id ? { ...g, ...(patch as Partial<GoalRow>) } : g)),
        );
      }
      setCoordSettled((prev) => new Set(prev).add(decisionKey(d)));
    },
    [],
  );
  const rejectCoordination = useCallback((d: CoordinationDecision) => {
    setCoordSettled((prev) => new Set(prev).add(decisionKey(d)));
  }, []);

  // SPLIT-TO-FIT (operator 2026-08-13): a length ask beyond the model ceiling
  // becomes VISIBLE continuation segments — motion-carried (vace_extend),
  // length-distributed, each editable — never submit-time magic. The
  // continuation prompt is a plain, honest default; ✨ Generate can refine it.
  const splitGoalToFit = useCallback(
    (i: number) => {
      const g = goals[i];
      if (!g) return;
      const ask = Number(g.frames.trim() || framesDefault.trim());
      if (!Number.isInteger(ask) || ask <= MODEL_FRAME_CEILING) return;
      const n = Math.ceil(ask / MODEL_FRAME_CEILING);
      const per = Math.ceil(ask / n);
      const chunks: number[] = [];
      let left = ask;
      for (let k = 0; k < n; k++) {
        const c = Math.min(per, left);
        chunks.push(Math.max(5, c));
        left -= c;
      }
      const base = g.prompt.trim();
      const contPrompt = base
        ? `the scene continues seamlessly, same setting and motion — ${base.split(/[.,;\n]/)[0].trim().slice(0, 80)}`
        : "the scene continues seamlessly, same setting and motion";
      setGoals((prev) => {
        const next = [...prev];
        next[i] = { ...next[i], frames: String(chunks[0]) };
        const conts = chunks.slice(1).map((c) => ({
          ...blankGoal("vace_extend"),
          prompt: contPrompt,
          frames: String(c),
        }));
        next.splice(i + 1, 0, ...conts);
        return next;
      });
    },
    [goals, framesDefault, setGoals],
  );
  const addGoal = useCallback(() => setGoals((prev) => [...prev, blankGoal()]), []);
  const removeGoal = useCallback((idx: number) => {
    // Keep at least one row — a movie needs a root segment.
    setGoals((prev) => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== idx)));
  }, []);
  const moveGoal = useCallback((idx: number, dir: -1 | 1) => {
    setGoals((prev) => {
      const j = idx + dir;
      if (j < 0 || j >= prev.length) return prev;
      const next = prev.slice();
      const [it] = next.splice(idx, 1);
      next.splice(j, 0, it);
      return next;
    });
  }, []);

  // ── SEGMENTS STEPPER (spec §2) — say how many shots this movie has, up front ──
  // Materializes rows to reach N; shrinking to N PARKS the tail rows in a
  // session-scoped stash instead of deleting them (operator 2026-08-27: 3 edited
  // segments → count 1 → count 5 must bring the 3 edited rows back, not mint
  // blanks). Growing revives parked rows first — they keep their keys, so
  // per-key sidecars (ticks, field modes, intents) re-attach — and only then
  // mints blanks, which inherit the joint of the last row so a movie built as
  // "extend motion" stays that way as it grows. No upper cap: the count takes
  // any integer ≥ 1.
  const [goalStash, setGoalStash] = useSessionState<GoalRow[]>("studio.movie.goalStash", () => []);

  // ── 🎬 PRODUCER (k120 slice 1) — premise → /video/producer/plan → rows. ──
  // The plan's per-segment seconds convert to frames against the CURRENT movie
  // fps (frames is what rides the wire; fps is movie-level). Existing edited
  // rows are parked in the stash first, never deleted.
  const [premise, setPremise] = useSessionState<string>("studio.movie.premise", "");
  const [producerNote, setProducerNote] = useSessionState<string>(
    "studio.movie.producerNote",
    "",
  );
  const [producing, setProducing] = useState(false);
  const [produceError, setProduceError] = useState<string | null>(null);
  // k120 slice 2 — render-time continuity: the runner rewrites each next segment
  // from the previous segment's ACTUAL closing frame. Auto-enabled by a
  // successful Produce (a scripted movie wants it); always operator-untickable.
  const [continuityRefresh, setContinuityRefresh] = useSessionState<boolean>(
    "studio.movie.continuityRefresh",
    false,
  );
  const produceMovie = useCallback(async () => {
    const p = premise.trim();
    if (!p || producing) return;
    setProducing(true);
    setProduceError(null);
    const res = await request<unknown>(hugpyConfig.producerPlanUrl, {
      method: "POST",
      body: JSON.stringify({ premise: p }),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "studio", operation: "studio.producer.plan" },
    });
    setProducing(false);
    if (!res.ok) {
      setProduceError(describeAppError(errorOf(res)));
      return;
    }
    const parsed = producerPlanSchema.safeParse(okValue(res));
    if (!parsed.success) {
      setProduceError("The producer reply did not match the expected plan shape.");
      return;
    }
    const plan = parsed.data;
    const effFps = fps > 0 ? fps : 24;
    const rows = plan.segments.map((s, i) => {
      const joint: JointMode =
        i === 0
          ? "still"
          : s.joint === "cut" || s.joint === "still" || s.joint === "vace_extend"
            ? (s.joint as JointMode)
            : "vace_extend";
      return {
        ...blankGoal(joint),
        prompt: s.prompt,
        negative: s.negative ?? "",
        frames: String(Math.max(1, Math.round(s.seconds * effFps))),
      };
    });
    if (goals.some((g) => g.prompt.trim() !== "")) {
      setGoalStash((st) => [...goals, ...st]);
    }
    setGoals(rows);
    setProducerNote([plan.title, plan.logline].filter(Boolean).join(" — "));
    setContinuityRefresh(true);
  }, [premise, producing, goals, fps]);

  const setSegmentCount = useCallback(
    (n: number) => {
      if (!Number.isInteger(n) || n < 1 || n === goals.length) return;
      if (n < goals.length) {
        const cut = goals.slice(n);
        setGoalStash((st) => [...cut, ...st]);
        setGoals(goals.slice(0, n));
        return;
      }
      const revived = goalStash.slice(0, n - goals.length);
      if (revived.length) setGoalStash((st) => st.slice(revived.length));
      const joint: JointMode = goals[goals.length - 1]?.joint ?? "still";
      const fresh = Array.from({ length: n - goals.length - revived.length }, () =>
        blankGoal(joint),
      );
      setGoals([...goals, ...revived, ...fresh]);
    },
    [goals, goalStash],
  );

  // ── the row → wire projection ──
  // ONE place turns editor rows into typed segment references, so the per-row assist
  // calls, the spread and the negatives all describe a row identically. `isDirection`
  // is the RESOLVED per-field operation, which is what makes the user's own text ride
  // as a `direction` (an instruction to act on) rather than as a scene to enhance.
  const goalInputs = useMemo<SpreadGoalInput[]>(
    () =>
      goals.map((g, i) => {
        const op = resolveOperation(
          fieldModes[g.key] ?? "auto",
          intents[g.key] ?? null,
          g.prompt.trim() === "",
        );
        return {
          key: g.key,
          prompt: g.prompt,
          // The EFFECTIVE negative (user text + the standard set when ticked) —
          // the spread/assist context describes the row as it will render.
          negative: composeNegative(g.negative, g.stdNegative),
          seed: g.seed,
          branchFrame: g.branchFrame,
          joint: g.joint,
          index: i,
          isDirection: op === "generate_from_direction",
        };
      }),
    [goals, fieldModes, intents],
  );

  // The LOCKED identity block (§1e): sent ONLY for a SAVED profile, because the
  // backend requires a name and an unsaved pile of uploads has none — inventing one
  // ("the attached subject") is exactly the invention this block exists to prevent.
  const identityContext = useMemo<IdentityContextWire | undefined>(() => {
    if (!selectedProfile) return undefined;
    return {
      identity_id: selectedProfile.slug,
      name: selectedProfile.name,
      reference_asset_ids: references.map((r) => r.uri),
    };
  }, [selectedProfile, references]);

  // TYPED CONTEXT for one row (§1c) — this segment, its neighbours, the locked
  // identity. This is what `context.hint` was never able to carry and what nothing
  // was ever filling: the joint mode arrives as a sentence, the branch frame as
  // "starts from frame N of the shot before it".
  const rowContext = useCallback(
    (i: number): AssistExtras["context"] => {
      const ctx: NonNullable<AssistExtras["context"]> = {};
      const here = goalInputs[i];
      if (!here) return ctx;
      ctx.segment = segmentRef(here);
      if (i > 0) ctx.previous_segment = segmentRef(goalInputs[i - 1]);
      if (i < goalInputs.length - 1) ctx.next_segment = segmentRef(goalInputs[i + 1]);
      if (identityContext) ctx.identity_profile = identityContext;
      return ctx;
    },
    [goalInputs, identityContext],
  );

  // ── intent routing (§1d): classify on BLUR, never per keystroke ──
  // Skipped entirely when the user has picked an explicit mode (their choice wins) and
  // when the field is blank (blank is unambiguously "generate" — the backend
  // short-circuits it too, but not paying the round trip is better than being told).
  // Re-classifying identical text is also skipped: the backend caches by hash, which
  // is a latency nicety, not a licence to spam it.
  const classifyRow = useCallback(
    (key: string, text: string) => {
      if ((fieldModes[key] ?? "auto") !== "auto") return;
      const t = text.trim();
      if (t === "") {
        setIntents((prev) => (prev[key] == null ? prev : { ...prev, [key]: null }));
        intentText.current[key] = "";
        return;
      }
      if (intentText.current[key] === t) return;
      intentText.current[key] = t;
      setIntentBusy((prev) => ({ ...prev, [key]: true }));
      void assist.classifyIntent(t, "segment").then((res) => {
        setIntents((prev) => ({ ...prev, [key]: res }));
        setIntentBusy((prev) => ({ ...prev, [key]: false }));
      });
    },
    [assist, fieldModes],
  );

  // ── per-row assist, now context-aware ──
  // The row's RESOLVED operation only chooses which mode the primary button invokes;
  // both explicit buttons stay live underneath, so a router mistake costs one click,
  // never a paragraph of the user's work.
  // The last AUTO-derived movie title. A name the operator typed is sacred; a
  // name this ref minted may be replaced by the next Generate (operator
  // 2026-08-13: "purely a product of the same generation method").
  const autoTitleRef = useRef("");

  const runRowAssist = useCallback(
    (i: number, mode: "detail" | "generate") => {
      const g = goals[i];
      if (!g) return;
      void (async () => {
        let fresh = "";
        await assist.runAssist(mode, g.prompt, (p) => {
          fresh = p;
          setGoalField(i, { prompt: p });
        }, { context: rowContext(i) });
        if (mode !== "generate" || !fresh.trim()) return;
        // NEGATIVE RIDES ALONG (operator 2026-08-13): the generated exclusion
        // list APPENDS to this goal's typed negative, never clobbers it
        // (composeNegative dedupes the std terms at payload time).
        await assist.runNegative(
          { subject: fresh, extra: { context: rowContext(i) } },
          (neg) => {
            const cur = (g.negative ?? "").trim().replace(/,\s*$/, "");
            setGoalField(i, { negative: cur ? `${cur}, ${neg}` : neg });
          },
        );
        // NAME RIDES ALONG: derive a title from the generated prompt when the
        // movie has none — or when the current one was itself auto-derived.
        const t = movieTitle.trim();
        if (t === "" || t === autoTitleRef.current) {
          const derived = fresh
            .split(/[.,;\n]/)[0]
            .trim()
            .split(/\s+/)
            .slice(0, 7)
            .join(" ")
            .slice(0, 48);
          if (derived) {
            autoTitleRef.current = derived;
            setMovieTitle(derived);
          }
        }
      })();
      // The text is about to change, so the old classification no longer describes it.
      intentText.current[g.key] = "";
      setIntents((prev) => ({ ...prev, [g.key]: null }));
    },
    [assist, goals, rowContext, setGoalField, movieTitle, setMovieTitle],
  );

  // ── negative generation (§1b) — an exclusion list for THIS shot ──
  const runRowNegative = useCallback(
    (i: number) => {
      const g = goals[i];
      if (!g) return;
      void assist.runNegative(
        { subject: g.prompt, draft: g.negative, extra: { context: rowContext(i) } },
        (neg) => setGoalField(i, { negative: neg }),
      );
    },
    [assist, goals, rowContext, setGoalField],
  );

  // ── GROUP ACTION: one coherent spread across the selected rows (§1a) ──
  // ONE call. Unselected rows ride as LOCKED context so the generator can see the
  // whole timeline, and the reply may only land on rows whose segment_id came back —
  // both halves are enforced here as well as server-side, because "the checkbox means
  // something" is the property the whole feature rests on.
  const onSpreadSelected = useCallback(async () => {
    const body = buildSpreadBody({
      goals: goalInputs,
      selectedKeys,
      movieQuery,
      globalNegative: negative,
      steeringSeed: pinSteering ? steeringSeed : null,
      context: identityContext ? { identity_profile: identityContext } : undefined,
      // The only style_bible key we can fill WITHOUT inventing anything: the saved
      // identity's own name. Everything else in the bible is authored content this
      // surface does not collect, and guessing it would be the invention defect.
      styleBible: selectedProfile ? { subject: selectedProfile.name } : undefined,
    });
    if (!body) return;
    setSpreadNotice(null);
    const res = await assist.runSpread(body);
    if (!res) return;

    const bySeg = new Map(res.segments.map((s) => [s.segment_id, s]));
    const applied = goals
      .filter((g) => selectedKeys.has(g.key) && bySeg.has(g.key))
      .map((g) => g.key);
    setGoals((prev) =>
      prev.map((g) => {
        if (!selectedKeys.has(g.key)) return g;      // an unselected row is never touched
        const hit = bySeg.get(g.key);
        if (!hit) return g;                          // a row the generator skipped: unchanged
        // An explicit Generate on a SELECTED row is meant to replace that row's prose —
        // that is the point of selecting it. A blank returned negative still falls back
        // to what the user wrote rather than blanking it.
        //
        // k121: the row also carries the KNOBS the coordination review turned for
        // it (ratchet-safe ones only — the backend never removes carry on its own).
        // ``hit.knobs`` is MECHANICS: no prose from any other row reaches here.
        const k = hit.knobs ?? {};
        const applyKnobs: Partial<GoalRow> = {};
        if (typeof k.joint_mode === "string") {
          applyKnobs.joint = k.joint_mode as GoalRow["joint"];
        }
        if (typeof k.seed === "number") applyKnobs.seed = String(k.seed);
        if (typeof k.frames === "number") applyKnobs.frames = String(k.frames);
        if ("branch_frame" in k) {
          applyKnobs.branchFrame = k.branch_frame == null ? "" : String(k.branch_frame);
        }
        return {
          ...g,
          prompt: hit.prompt,
          negative: hit.negative || g.negative,
          ...applyKnobs,
        };
      }),
    );
    for (const key of applied) {
      intentText.current[key] = "";
    }
    setIntents((prev) => {
      const next = { ...prev };
      for (const key of applied) next[key] = null;
      return next;
    });
    if (res.steering_seed != null) setSteeringSeed(res.steering_seed);
    // k121: a fresh review replaces the old one, and the answered set resets with
    // it — a proposal answered against last generation's prose says nothing about
    // this one.
    setCoordination(readCoordinationReport(res.coordination));
    setCoordSettled(new Set());
    setSpreadNotice(spreadNoticeLines(res, applied));
  }, [
    assist,
    goalInputs,
    goals,
    identityContext,
    movieQuery,
    negative,
    pinSteering,
    selectedKeys,
    selectedProfile,
    steeringSeed,
  ]);

  // ── GROUP ACTION: negatives for the selected rows ──
  // Sequential per-row calls ON PURPOSE, unlike prompts: a negative is an exclusion
  // list for ONE shot, so there is no cross-row coherence to lose — and each row's
  // own prompt is the subject being negated. Stops at the first failure rather than
  // hammering a dead worker N times; the reason is on the shared assist error.
  const [negBatch, setNegBatch] = useState<{ done: number; total: number } | null>(null);
  const onNegativesSelected = useCallback(async () => {
    const rows = goals.map((g, i) => ({ g, i })).filter(({ g }) => selectedKeys.has(g.key));
    if (rows.length === 0) return;
    setSpreadNotice(null);
    let done = 0;
    setNegBatch({ done: 0, total: rows.length });
    for (const { g, i } of rows) {
      let wrote = false;
      await assist.runNegative(
        { subject: g.prompt, draft: g.negative, extra: { context: rowContext(i) } },
        (neg) => {
          wrote = true;
          setGoalField(i, { negative: neg });
        },
      );
      if (!wrote) break;
      done += 1;
      setNegBatch({ done, total: rows.length });
    }
    setNegBatch(null);
    setSpreadNotice([
      done === rows.length
        ? `${done} negative prompt(s) written.`
        : `${done} of ${rows.length} negative prompt(s) written — the rest are unchanged.`,
    ]);
  }, [assist, goals, rowContext, selectedKeys, setGoalField]);

  // ── B3 scrubber → append a BRANCH goal (the "Branch from frame N" action) ──
  // MID-ROW BRANCH CHOICE (option A — TRUNCATE, stated in the report + here): v0 authors
  // a LINEAR chain, so a branch is a new child appended after `segmentIndex`, conditioned
  // on `frame` of that segment (its parent). Branching from the LAST authored row is a
  // clean extension. Branching from a NON-final row cannot keep the old tail as a SIBLING
  // take in a linear timeline (that needs the schema's take-tree — the next slice), so we
  // TRUNCATE the authored rows after `segmentIndex` and re-continue from there. We chose
  // truncate over refuse so the operator's "divergence anywhere · the user's options
  // decide" works everywhere in v0; the scrubber's action LABEL names the dropped rows so
  // the cut is never silent (honesty doctrine). The render itself is a fresh enqueue —
  // studio jobs are IMMUTABLE, so this authors a NEW extended movie, never mutating the old.
  const onBranchFromSegment = useCallback((segmentIndex: number, frame: number) => {
    const key = newGoalKey();
    setGoals((prev) => {
      const kept = prev.slice(0, Math.min(prev.length, segmentIndex + 1));
      // A scrubber-branched take continues from the chosen frame — a "still" joint.
      return [
        ...kept,
        {
          key,
          prompt: "",
          negative: "",
          stdNegative: false,
          seed: "",
          branchFrame: String(frame),
          joint: "still" as JointMode,
        },
      ];
    });
    setPendingFocusKey(key);
    setFormMsg(null);
  }, []);

  // Focus the just-appended branch goal's prompt (and bring it into view) once mounted.
  useEffect(() => {
    if (!pendingFocusKey) return;
    // k88: the prompt pane now lives inside PromptFieldTabs — `${idPrefix}-prompt`.
    const el = document.getElementById(`vi-movie-${pendingFocusKey}-prompt`);
    if (el) {
      (el as HTMLTextAreaElement).focus();
      el.scrollIntoView({ block: "center", behavior: "smooth" });
    }
    setPendingFocusKey(null);
  }, [pendingFocusKey, goals]);

  // ── build the exact request body the route reads ──
  const buildBody = useCallback((titleOverride?: string): { body: Record<string, unknown> } | { err: string } => {
    if (goals.length < 1) return { err: "Add at least one goal (the root segment)." };
    for (let i = 0; i < goals.length; i++) {
      if (goals[i].prompt.trim() === "") {
        return { err: `Goal ${i + 1} needs a prompt (every segment renders its text).` };
      }
      const s = goals[i].seed.trim();
      if (s !== "" && !Number.isInteger(Number(s))) {
        return { err: `Goal ${i + 1} seed must be a whole number (or blank).` };
      }
      // Per-segment sampler overrides — the SAME bounds as the movie-level knobs
      // below (and the schema's _check_steps_cfg): blank = inherit.
      const gSteps = goals[i].steps.trim();
      if (gSteps !== "" && (!Number.isInteger(Number(gSteps)) || Number(gSteps) < 1 || Number(gSteps) > 100)) {
        return { err: `Goal ${i + 1} steps must be a whole number in [1, 100] (or blank to inherit).` };
      }
      const gCfg = goals[i].cfg.trim();
      if (gCfg !== "" && (Number.isNaN(Number(gCfg)) || Number(gCfg) < 0 || Number(gCfg) > 20)) {
        return { err: `Goal ${i + 1} CFG must be a number in [0, 20] (or blank to inherit).` };
      }
      const gFrames = goals[i].frames.trim();
      if (gFrames !== "" && (!Number.isInteger(Number(gFrames)) || Number(gFrames) < 1)) {
        return { err: `Goal ${i + 1} length must be a whole number of frames \u2265 1 (or blank to inherit).` };
      }
      const gCtx = goals[i].contextFrames.trim();
      if (gCtx !== "" && (!Number.isInteger(Number(gCtx)) || Number(gCtx) < 1)) {
        return { err: `Goal ${i + 1} context frames must be a whole number ≥ 1 (or blank to inherit).` };
      }
      // A "cut" carries no frame, so its branch-frame input is disabled + ignored; only
      // validate a branch frame for a still / vace_extend joint.
      if (i > 0 && goals[i].joint !== "cut") {
        const bf = goals[i].branchFrame.trim();
        if (bf !== "" && (!Number.isInteger(Number(bf)) || Number(bf) < 0)) {
          return {
            err: `Goal ${i + 1} branch frame must be a whole number ≥ 0 (or blank = last frame).`,
          };
        }
      }
    }
    for (const [name, val] of [["width", width], ["height", height], ["fps", fps]] as const) {
      if (!Number.isInteger(val) || val <= 0) {
        return { err: `${name} must be a positive whole number.` };
      }
    }
    const budgetNum = vramBudget.trim() === "" ? undefined : Number(vramBudget);
    if (budgetNum !== undefined && (Number.isNaN(budgetNum) || budgetNum <= 0)) {
      return { err: "VRAM budget must be a positive number (or blank for the route default)." };
    }
    // Movie-level sampler overrides — same bounds the clip surface (and the movie
    // schema's _check_steps_cfg) enforce: steps int [1,100], cfg number [0,20]. Blank
    // drops the key (model default).
    const framesDefNum = framesDefault.trim() === "" ? undefined : Number(framesDefault);
    if (framesDefNum !== undefined && (!Number.isInteger(framesDefNum) || framesDefNum < 1)) {
      return { err: "Segment length must be a whole number of frames \u2265 1 (or blank for model defaults)." };
    }
    const stepsNum = steps.trim() === "" ? undefined : Number(steps);
    if (
      stepsNum !== undefined &&
      (!Number.isInteger(stepsNum) || stepsNum < 1 || stepsNum > 100)
    ) {
      return { err: "Steps must be a whole number in [1, 100] (or blank for the model default)." };
    }
    const cfgNum = cfg.trim() === "" ? undefined : Number(cfg);
    if (cfgNum !== undefined && (Number.isNaN(cfgNum) || cfgNum < 0 || cfgNum > 20)) {
      return { err: "CFG must be a number in [0, 20] (or blank for the model default)." };
    }

    const goalBodies = goals.map((g, i) => {
      const node: Record<string, unknown> = { prompt: g.prompt };
      const s = g.seed.trim();
      if (s !== "") node.seed = Number(s);
      // Per-segment negative (operator addendum 2026-07-12): send `negative` ONLY when
      // the operator typed one — blank drops the key so the runner falls back to the
      // movie-level negative_prompt (schema: StudioMovieGoal.negative None ⇒ default).
      // k88: the standard set is COMPOSED in here when the row's tick is on, so an
      // empty user negative + a tick still sends the standard exclusions.
      const effNegative = composeNegative(g.negative, g.stdNegative);
      if (effNegative !== "") node.negative = effNegative;
      // Per-segment model/sampler overrides (2026-08-12): blank drops the key so
      // the segment inherits the movie-level pin/override (or the router/model
      // default) — the exact fallback chain StudioMovieGoal documents.
      if (g.modelId.trim() !== "") node.model_id = g.modelId.trim();
      if (g.steps.trim() !== "") node.steps = Number(g.steps);
      if (g.cfg.trim() !== "") node.cfg = Number(g.cfg);
      if (g.frames.trim() !== "") node.frames = Number(g.frames);
      // joint_mode + branch_frame are meaningless on the root (segment 0 has no parent).
      if (i > 0) {
        // Send a non-default joint_mode ("vace_extend" | "cut"); "still" is the route default.
        if (g.joint !== "still") node.joint_mode = g.joint;
        // branch_frame only for still / vace_extend (a cut carries no frame).
        if (g.joint !== "cut") {
          const bf = g.branchFrame.trim();
          if (bf !== "") node.branch_frame = Number(bf);
        }
        // context_frames is only consulted for a vace_extend joint (schema note) —
        // sending it otherwise would imply a knob that does nothing.
        if (g.joint === "vace_extend" && g.contextFrames.trim() !== "") {
          node.context_frames = Number(g.contextFrames);
        }
      }
      return node;
    });

    const body: Record<string, unknown> = {
      resolution: { width, height, fps },
      frames: framesDefNum,
      seed,
      vram_budget_gb: budgetNum,
      // Movie-level model pin + sampler overrides (blank ⇒ key dropped = router/model
      // default). The route reads these movie-level and applies them to every segment a
      // node doesn't override (studio_movie_schema: model_id/steps/cfg movie-level defaults).
      model_id: modelId.trim() ? modelId.trim() : undefined,
      steps: stepsNum,
      cfg: cfgNum,
      // negative_prompt (movie-wide) RETIRED from this surface 2026-08-13
      // (operator: deprecated artifact) — per-goal negatives + the standard
      // tick compose everything; the route treats absence as None cleanly.
      project: project.trim() ? project.trim() : undefined,
      // The movie's NAME (k91) — free text, display-only: it never changes what renders,
      // it is what the sessions list shows instead of a bare dir id. Blank drops the key
      // (the schema takes title=None and the session lists as "untitled").
      title: (titleOverride ?? movieTitle).trim() ? (titleOverride ?? movieTitle).trim() : undefined,
      // Optional root conditioning still — the route jail-resolves this abs uri and
      // renders segment 0 i2v (drops back to t2v when absent). Mirrors the clip surface.
      start_image: startImage ? startImage.uri : undefined,
      // IDENTITY (unified): a SAVED profile is the canonical identity — send its slug (the
      // route resolves it server-side; a later profile edit re-resolves). Non-empty identity
      // (either form) -> an identity movie (every segment renders id_lock so the subject
      // carries across scene changes). Raw reference_images ride only for an UNSAVED identity.
      identity_profile: selectedProfile ? selectedProfile.slug : undefined,
      reference_images:
        !selectedProfile && references.length ? references.map((r) => r.uri) : undefined,
      goals: goalBodies,
      // k120 slice 2 — PRODUCER CONTINUITY REFRESH: the runner rewrites each
      // non-root segment's prompt from a vision read of the frame the previous
      // segment actually ended on. Off drops the key (schema default False).
      continuity_refresh: continuityRefresh || undefined,
    };
    return { body };
  }, [goals, width, height, fps, seed, vramBudget, modelId, steps, cfg, negative, project, movieTitle, startImage, references, selectedProfile, continuityRefresh]);

  // NAME NUDGE (k91) — set when a submit was held back only because the movie has no
  // name. It is a warning, not a gate: the run row then offers "Generate without a
  // name", and `onGenerate(true)` skips the check. Cleared whenever a name is typed.
  const [nameWarn, setNameWarn] = useState(false);
  // The warning is an EXPLICIT PER-SECTION OPTION now (operator 2026-08-13),
  // DEFAULT OFF, remembered in localStorage — shared UnnamedWarn seam.
  const [warnUnnamed, setWarnUnnamed] = useWarnUnnamedPref("cinema");

  const onGenerate = useCallback(
    (skipNameCheck = false, titleOverride?: string) => {
      const built = buildBody(titleOverride);
      if ("err" in built) {
        setFormMsg(built.err);
        return;
      }
      // Warn ONCE per unnamed submit. An unnamed movie is still fully renderable — the
      // cost is later, in a sessions list where it reads as "untitled · 7f3a91c2…" and
      // nobody can tell which paused render is which.
      if (!skipNameCheck && warnUnnamed && movieTitle.trim() === "") {
        setNameWarn(true);
        setFormMsg(null);
        return;
      }
      setNameWarn(false);
      setFormMsg(null);
      if (titleOverride && titleOverride.trim()) setMovieTitle(titleOverride.trim());
      job.run(built.body);
    },
    [buildBody, job, movieTitle, setMovieTitle, warnUnnamed],
  );

  // SESSION LIBRARY PUSH (operator 2026-08-13): a finished movie's outputs
  // land in the Session Library IMMEDIATELY — every segment clip plus the
  // assembled movie (outputs[-1] when assembly produced one) — tagged
  // generate_movie so the sidebar's Videos section lists them newest-first.
  // addToLibrary de-dups by uri, so re-renders of the same result are no-ops.
  useEffect(() => {
    const r = job.result;
    if (!r || r.ok === false || !r.outputs || r.outputs.length === 0) return;
    for (const ref of r.outputs) {
      if (!ref?.uri || !ref?.asset_id) continue;
      addToLibrary(ref, "cinema", movieTitle.trim() || "movie", {
        genKind: "generate_movie",
        prompt: goals[0]?.prompt,
        groupId: job.jobId ?? undefined,
      });
    }
    // movieTitle/goals deliberately excluded: push once per RESULT, labelled
    // with whatever the fields held when it landed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.result, job.jobId]);

  // ── derive playback refs from the terminal result ──
  // Outputs = [segment clips…, movie.mp4 LAST] (the runner appends the assembled
  // movie only when assembly produced one). Pair each segment ref with its manifest
  // record for the joint metadata; the movie.mp4 is the trailing ref.
  const view = useMemo(() => {
    const result = job.result;
    if (!result) return null;
    const outputs: MediaRef[] = result.outputs ?? [];
    const segs = result.movie?.segments ?? [];
    const segCount = segs.length || outputs.length;
    const assembled = !!result.movie?.assembly?.movie && outputs.length > segCount;
    const movieRef = assembled ? outputs[outputs.length - 1] : null;
    const segmentRefs = outputs.slice(0, segCount);
    return { result, segs, segmentRefs, movieRef, assembled };
  }, [job.result]);

  const isDone = job.status === "done";
  const isFailed = job.status === "failed" || job.status === "cancelled";

  const disableSubmit = job.running;

  // ── SYNTHETIC-TIER HONESTY BANNER (operator ask 2026-07-12) ──
  // Same footgun as the Clip surface (see StudioGenerateTab's wouldBindSynthetic):
  // a blank/sub-floor budget silently binds synthetic-t2v (root has no start_image)
  // or synthetic-i2v (root has one) instead of a real render. IDENTITY MOVIES are
  // exempt — the moment `references` is non-empty every segment renders id_lock
  // (see the enqueue body below), and id_lock has no synthetic model registered at
  // all (studio/models_seed.py), so a too-low budget there fails the router
  // honestly instead of ever producing colored noise.
  //
  // ⚠ THE FLOOR IS MEASURED NOW (2026-07-29): it comes from the movie capability's own
  // `vram_envelope_gb` in the ratified table, not from the hand-mirrored
  // STUDIO_REAL_FLOOR_GB — which said 6 while the cheapest real row is 8.2, so a budget
  // of 7 bound the synthetic prover and this banner stayed quiet. The constant survives
  // only as the fallback for a page whose discovery GET has not landed.
  const hasIdentity = references.length > 0 || selectedProfile != null;
  const movieFloorGb =
    (renderLoaded ? suggestedBudgetGb(renderPresets, MOVIE_CAPABILITY) : null)
    ?? STUDIO_REAL_FLOOR_GB;
  // What THIS movie actually needs, given its joints and whether an identity is attached
  // (an identity movie renders EVERY segment id_lock, so it binds the VACE row whatever
  // the joint says). Offered as the one-click fill rather than a fixed 10.
  const movieNeedGb =
    (renderLoaded
      ? movieBudgetGb(renderPresets, {
          joint: goals[1]?.joint ?? "cut",
          identity: hasIdentity,
        })
      : null) ?? STUDIO_SAFE_FILL_GB;
  const movieGeomOutOfEnvelope =
    renderLoaded && !withinGeometry(renderPresets, MOVIE_CAPABILITY, width, height);
  const wouldBindSynthetic =
    references.length === 0 && bindsSyntheticTier(vramBudget, movieFloorGb);

  // Movie-mode SETTINGS panel — portaled into the sidebar Settings tab (the SAME host the
  // Clip surface's knobGrid uses), so the movie's generation options live on the shared
  // Settings surface exactly like the Clip surface's knobs (operator ask 2026-07-13: "the
  // Settings tab only consists of templates — I'd like a wide range of options including the
  // model selection"). ONE portaled node (a single `vi-studio-knobs` grid) grown from the
  // old templates-only node, mirroring the Clip knobGrid idiom — templates first, then the
  // model pin + sampler overrides + geometry/seed/budget/project. The movie-wide NEGATIVE
  // stays in the center column (a wide prompt textarea, kept beside the goals like the Clip
  // surface keeps its negative out of the knob grid).
  const settingsNode = (
    <div className="vi-studio-knobs">
      {/* ABOUT — the condensed what-is-this / how-to prose (k89), collapsed at the
          top of the Settings-tab content. Absorbs the old inline intro block and
          the "movie idea" explanation (one short functional line stays inline). */}
      <div className="vi-studio-knob-wide" style={{ flex: "1 1 100%" }}>
        <AboutExpander title="About Cinema">
          <p>
            Cinema renders an ordered strip of REAL studio clips — each goal below is
            one segment, spliced onto the last at its joint.
          </p>
          <p>
            Per segment, pick how it follows: continue (from one frame), extend
            motion (VACE), or a scene cut (a fresh render; the previous segment plays
            in full).
          </p>
          <p>
            Add an identity reference to carry the SAME subject across every scene —
            each segment then renders id_lock.
          </p>
          <p>
            The movie idea + ✨ Generate prompts write every ticked segment against
            ONE shared world; the steering-seed pin keeps a follow-up spread in that
            same world.
          </p>
          <p>
            v0 authors a linear chain; once rendered, the spliced row shows each
            joint and lets you scrub a segment to branch a new take.
          </p>
        </AboutExpander>
      </div>
      {/* WORKFLOW PRESETS — derived from the ratified movie preset's own joint modes.
          Each row is a SHAPE, not a scene: it sets the joint on your segments, the
          ratified geometry and a budget sized for the binding that joint really uses,
          and leaves every prompt, seed, identity and start image exactly as you left
          them. Unproven joints say so. */}
      {/* MOVIE TEMPLATES (restored 2026-08-01) — a SCENE prefill, distinct from the
          workflow presets below: a template writes the goal prompts/joints/seeds (and,
          when carried, resolution + budget + a saved identity), where a workflow only
          sets the shape. */}
      <div className="vi-knob vi-studio-knob-wide">
        <label htmlFor="vi-movie-scene-template">Movie templates (scenes)</label>
        <select
          id="vi-movie-scene-template"
          className="vi-knob-select"
          value=""
          disabled={disableSubmit}
          onChange={(e) => {
            const v = e.target.value;
            if (v) applyTemplate(v);
          }}
        >
          <option value="">Choose a template — prefills the scenes</option>
          {TEMPLATES.map((t) => (
            <option key={t.id} value={t.id} title={t.hint}>
              {t.name}
            </option>
          ))}
        </select>
        <span className="vi-knob-hint">
          Prefills the goal rows below (prompts, joints, seeds). An attached identity is
          never replaced — add your reference or pick a profile, then Generate.
        </span>
      </div>

      <div className="vi-knob vi-studio-knob-wide">
        <label htmlFor="vi-movie-template">Movie preset (workflow)</label>
        <select
          id="vi-movie-template"
          className="vi-knob-select"
          value=""
          disabled={disableSubmit || workflowPresets.length === 0}
          onChange={(e) => {
            const v = e.target.value;
            if (v) applyWorkflow(v);
          }}
        >
          <option value="">
            {renderLoaded
              ? "Choose a workflow — sets the joint, geometry and budget"
              : "Loading what this fleet can render…"}
          </option>
          {workflowPresets.map((w) => (
            <option key={w.id} value={w.id} title={w.hint}>
              {w.name}
              {w.proven ? "" : " · unproven"}
              {w.minBudgetGb != null ? ` · ≥${Math.ceil(w.minBudgetGb)} GB` : ""}
            </option>
          ))}
        </select>
        <span className="vi-knob-hint">
          A preset sets the WORKFLOW — which joint stitches your shots, the geometry this
          fleet has ratified, and a budget sized for that joint&apos;s binding. Your shot
          prompts, seeds, identity and start image are never touched; rows are only added
          if you have fewer than two. Rows marked “unproven” have a ratified path that has
          not yet produced a clip{renderBox ? ` on ${renderBox}` : ""}.
        </span>
      </div>

      {/* MODEL PIN — blank-first "Auto" (the capability router picks by budget + resolution);
          an explicit pin overrides for every segment a node doesn't itself override. Mirrors
          the Clip surface's model select (StudioGenerateTab knobGrid).
          2026-07-29: the menu is now the models a RATIFIED movie preset actually binds —
          the movie row's own model plus every model named by the segment presets it
          `composes` — instead of the unfiltered STUDIO_MODEL_IDS mirror. The escape hatch
          keeps the whole list reachable, labelled unverified. */}
      <div className="vi-knob vi-studio-knob-wide">
        <label htmlFor="vi-movie-model">model</label>
        <select
          id="vi-movie-model"
          className="vi-knob-select"
          value={modelId}
          disabled={disableSubmit}
          onChange={(e) => setModelId(e.target.value)}
        >
          <option value="">Auto — router picks by capability/budget</option>
          {movieModels.length > 0 && (
            <optgroup label="Bound by a ratified movie segment">
              {movieModels.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </optgroup>
          )}
          {(showAllModels || !renderLoaded || (modelId !== "" && !movieModels.includes(modelId))) && (
            <optgroup label={renderLoaded ? "Other studio models — unverified for a movie segment" : "Studio models"}>
              {STUDIO_MODEL_IDS.filter((id) => !movieModels.includes(id)).map((id) => (
                <option key={id} value={id}>
                  {id}
                  {renderLoaded ? " · unverified" : ""}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <span className="vi-knob-hint">
          Blank = the router picks per segment. An unroutable pin comes back as an error on
          the job, never a silent fallback.
        </span>
        {renderLoaded && (
          <label className="vi-knob-hint" style={{ display: "flex", gap: "0.35rem", alignItems: "center" }}>
            <input
              type="checkbox"
              checked={showAllModels}
              disabled={disableSubmit}
              onChange={(e) => setShowAllModels(e.target.checked)}
            />
            Advanced — show every studio model (off-menu pins are unverified for a movie segment)
          </label>
        )}
      </div>

      {/* SAMPLER overrides — blank = the model default. Same bounds the clip surface + the
          movie schema enforce (steps [1,100], cfg [0,20]). */}
      <div className="vi-knob">
        <label htmlFor="vi-movie-steps">steps</label>
        <input
          id="vi-movie-steps"
          className="vi-knob-input"
          type="number"
          min={1}
          max={100}
          value={steps}
          placeholder="model default"
          disabled={disableSubmit}
          onChange={(e) => setSteps(e.target.value)}
        />
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-movie-cfg">cfg</label>
        <input
          id="vi-movie-cfg"
          className="vi-knob-input"
          type="number"
          min={0}
          max={20}
          step={0.5}
          value={cfg}
          placeholder="model default"
          disabled={disableSubmit}
          onChange={(e) => setCfg(e.target.value)}
        />
      </div>

      {/* GEOMETRY — moved here from the center column so the Settings tab is the single home
          for the movie's generation options (like the Clip surface's knob grid). */}
      <div className="vi-knob">
        <label htmlFor="vi-movie-w">width (px)</label>
        <input
          id="vi-movie-w"
          className="vi-knob-input"
          type="number"
          min={64}
          step={8}
          value={width}
          disabled={disableSubmit}
          onChange={(e) => setWidth(Number(e.target.value) || 0)}
        />
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-movie-h">height (px)</label>
        <input
          id="vi-movie-h"
          className="vi-knob-input"
          type="number"
          min={64}
          step={8}
          value={height}
          disabled={disableSubmit}
          onChange={(e) => setHeight(Number(e.target.value) || 0)}
        />
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-movie-fps">fps</label>
        <input
          id="vi-movie-fps"
          className="vi-knob-input"
          type="number"
          min={1}
          max={60}
          value={fps}
          disabled={disableSubmit}
          title={
            movieGeometry
              ? `Ratified movie envelope: ${movieGeometry.width}×${movieGeometry.height} @ ${movieGeometry.fps}fps (“${movieGeometry.presetTitle}”).`
              : undefined
          }
          onChange={(e) => setFps(Number(e.target.value) || 0)}
        />
        {movieGeomOutOfEnvelope && (
          <span className="vi-knob-flag">
            {width}×{height} is outside the ratified movie envelope
            {movieGeometry ? ` (${movieGeometry.width}×${movieGeometry.height})` : ""} — no
            proven segment binding covers it and the router may refuse. An identity movie is
            the strictest case: every segment renders id_lock, whose only budget-fitting
            model tops out at 832×480.
          </span>
        )}
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-movie-seed">base seed</label>
        <input
          id="vi-movie-seed"
          className="vi-knob-input"
          type="number"
          min={0}
          value={seed}
          disabled={disableSubmit}
          onChange={(e) => setSeed(Number(e.target.value) || 0)}
        />
        <span className="vi-knob-hint">A segment with no seed uses base seed + its index.</span>
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-movie-vram">vram budget (GB)</label>
        <input
          id="vi-movie-vram"
          className="vi-knob-input"
          type="number"
          min={0}
          step={0.5}
          value={vramBudget}
          placeholder="route default"
          disabled={disableSubmit}
          onChange={(e) => {
            // Marks the field OPERATOR-OWNED: a workflow preset then fills nothing over
            // it. Clearing the field hands it back to the presets.
            budgetTouched.current = e.target.value.trim() !== "";
            setVramBudget(e.target.value);
          }}
        />
        <span className="vi-knob-hint">
          {renderLoaded
            ? `Blank = autofit to the card. This movie needs at least ${movieNeedGb} GB${hasIdentity ? " — an identity movie renders every segment id_lock, which binds the VACE row whatever the joint says" : ""}.`
            : "Blank = route default (sub-real binds the synthetic tier)."}
        </span>
      </div>
      <div className="vi-knob vi-studio-knob-wide">
        <label htmlFor="vi-movie-project">project name (optional)</label>
        <input
          id="vi-movie-project"
          className="vi-knob-input"
          type="text"
          value={project}
          placeholder="auto-named"
          disabled={disableSubmit}
          onChange={(e) => setProject(e.target.value)}
        />
      </div>
    </div>
  );

  return (
    <div className="vi-studio-generate">
      {/* The movie Settings panel lives in the left-sidebar Settings tab (portaled into
          settingsHost, like the Clip surface's knobGrid): templates + model pin + sampler
          + geometry/seed/budget/project; null until that panel mounts. */}
      {settingsHost ? createPortal(settingsNode, settingsHost) : null}
      {/* (The intro prose moved into the sidebar "About Cinema" expander — k89.) */}

      {/* ── CINEMA SESSIONS: relocated (2026-08-12). The recovery surface now lives in
             the Active display's Sessions panel (▤ Active drawer / the /active station),
             sectioned per surface (Scene · Movie · Clip · Cinema) — reachable from every
             station instead of only this composer. ── */}

      {/* ── MOVIE NAME — what this session will be called in the list above. Kept at the
             top of the authoring column (not in the sidebar with `project`) because it is
             the one field that decides whether a paused render is identifiable later. ── */}
      <section aria-label="Movie name" style={{ margin: "0.7rem 0 0.6rem" }}>
        <label
          htmlFor="vi-movie-title"
          className="vi-comfy-label"
          style={{ display: "block", marginBottom: "0.25rem", opacity: 0.85 }}
        >
          Movie name (optional, but names the session)
        </label>
        <input
          id="vi-movie-title"
          className="vi-knob-input"
          type="text"
          value={movieTitle}
          placeholder="e.g. Mira — beach day"
          disabled={disableSubmit}
          style={{ maxWidth: "26rem", width: "100%" }}
          onChange={(e) => {
            setMovieTitle(e.target.value);
            // Typing a name answers the nudge — clear it rather than making the operator
            // dismiss a warning about a field they just filled in.
            if (e.target.value.trim() !== "") setNameWarn(false);
          }}
        />
        <WarnUnnamedToggle checked={warnUnnamed} onChange={setWarnUnnamed} />
      </section>

      {/* ── MOVIE SETTINGS, IN-COLUMN (operator 2026-08-13: "not seeing model,
             fps, dimension inputs for cinema") — the geometry/model/sampler knobs
             live in the left-sidebar Settings tab via portal, which is easy to
             miss (and absent when no host mounts). This expander mirrors the
             SAME state — two views, one truth, like the Clip card. ── */}
      <PromptCardSettings label="movie settings — model · size · fps · steps · cfg · budget">
        <div className="vi-knob vi-studio-knob-wide" style={{ minWidth: "16rem" }}>
          <label htmlFor="vi-moviecol-model">model — whole movie</label>
          <select
            id="vi-moviecol-model"
            className="vi-knob-select"
            value={modelId}
            disabled={disableSubmit}
            onChange={(e) => setModelId(e.target.value)}
          >
            <option value="">Auto — router picks by capability/budget</option>
            {movieModels.length > 0 && (
              <optgroup label="Bound by a ratified movie segment">
                {movieModels.map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </optgroup>
            )}
            {(showAllModels || !renderLoaded || (modelId !== "" && !movieModels.includes(modelId))) && (
              <optgroup label={renderLoaded ? "Other studio models — unverified for a movie segment" : "Studio models"}>
                {STUDIO_MODEL_IDS.filter((id) => !movieModels.includes(id)).map((id) => (
                  <option key={id} value={id}>{id}{renderLoaded ? " · unverified" : ""}</option>
                ))}
              </optgroup>
            )}
          </select>
        </div>
        <div className="vi-knob" style={{ maxWidth: "7rem" }}>
          <label htmlFor="vi-moviecol-width">width</label>
          <input id="vi-moviecol-width" className="vi-knob-input" type="number" min={64} step={8}
                 value={width} onChange={(e) => setWidth(Number(e.target.value) || 0)} />
        </div>
        <div className="vi-knob" style={{ maxWidth: "7rem" }}>
          <label htmlFor="vi-moviecol-height">height</label>
          <input id="vi-moviecol-height" className="vi-knob-input" type="number" min={64} step={8}
                 value={height} onChange={(e) => setHeight(Number(e.target.value) || 0)} />
        </div>
        {/* LENGTH ROW (operator 2026-08-13): fps | frames/segment | seconds —
            any two set the third; dimmed = derived, last-touched wins. This is
            the MOVIE default; a segment's own length overrides it. */}
        <LengthRow
          idPrefix="vi-moviecol-len"
          fps={fps}
          onFps={(n) => setFps(n)}
          frames={framesDefault}
          onFrames={setFramesDefault}
          framesPlaceholder="model default (81) / segment"
        />
        <div className="vi-knob" style={{ maxWidth: "7rem" }}>
          <label htmlFor="vi-moviecol-steps">steps</label>
          <input id="vi-moviecol-steps" className="vi-knob-input" type="number" min={1} max={100}
                 value={steps} placeholder="model default" onChange={(e) => setSteps(e.target.value)} />
        </div>
        <div className="vi-knob" style={{ maxWidth: "7rem" }}>
          <label htmlFor="vi-moviecol-cfg">cfg</label>
          <input id="vi-moviecol-cfg" className="vi-knob-input" type="number" min={0} max={20} step={0.5}
                 value={cfg} placeholder="model default" onChange={(e) => setCfg(e.target.value)} />
        </div>
        <div className="vi-knob" style={{ maxWidth: "9rem" }}>
          <label htmlFor="vi-moviecol-vram">VRAM budget GB</label>
          <input id="vi-moviecol-vram" className="vi-knob-input" type="number" min={1}
                 value={vramBudget} placeholder="autofit" onChange={(e) => setVramBudget(e.target.value)} />
        </div>
      </PromptCardSettings>



      {/* ── the goal timeline (NLE-row order, top → bottom) ──
             Operator addendum 2026-07-12: each segment is now a DISTINCTLY CONTRASTED
             panel, reusing the Generate-station Movie-mode goal-card idiom verbatim —
             .vi-gen-parts (the list rhythm) → .vi-gen-part.vi-movie-goal (the bordered
             + var(--vi-surface-raised) panel that lifts off the black page) with a
             .vi-gen-part-head carrying a .vi-movie-badge chip, a .vi-gen-part-idx #N,
             the follows-hint, and .vi-gen-part-actions (auto-pushed reorder/remove). No
             new look invented — the same classes GenerateStation.tsx uses. */}
      <section aria-label="Goals" className="vi-studio-composer vi-movie-timeline">
        {/* ── 🎬 PRODUCER (k120 slice 1) — a premise in, a full script out: the
               plan decides segment count, per-segment prompt/negative/length and
               how each shot joins the previous one (vace_extend = continue from
               the last frame). Rows land below as ordinary editable rows; the
               previous rows are parked in the stash, not deleted. */}
        <div className="vi-knob" style={{ marginBottom: "0.6rem" }}>
          <label htmlFor="vi-movie-premise">🎬 producer</label>
          <div
            style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start", flexWrap: "wrap" }}
          >
            <textarea
              id="vi-movie-premise"
              className="vi-knob-input"
              rows={2}
              style={{ flex: "1 1 24rem", resize: "vertical" }}
              placeholder="Premise / logline — the producer writes the whole script: segments, prompts, lengths, joins"
              value={premise}
              onChange={(e) => setPremise(e.target.value)}
              disabled={producing}
            />
            <button
              type="button"
              className="vi-btn vi-btn-accent"
              disabled={producing || premise.trim() === ""}
              onClick={() => void produceMovie()}
              title="Generate a full script from the premise: segment count, per-segment prompts, negatives, lengths and joins. Current rows are parked in the stash."
            >
              {producing ? "Producing…" : "🎬 Produce"}
            </button>
          </div>
          <label
            className="vi-comfy-hint"
            style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}
            title="During the render, a vision model reads the frame each segment actually ended on and a text model rewrites the next segment's prompt to open from that state — true segment-to-segment continuity. Advisory: any failure keeps the authored prompt."
          >
            <input
              type="checkbox"
              checked={continuityRefresh}
              onChange={(e) => setContinuityRefresh(e.target.checked)}
            />
            continuity refresh — rewrite each next segment from the previous segment's rendered
            last frame
          </label>
          {producerNote ? <span className="vi-knob-hint">{producerNote}</span> : null}
          {produceError ? (
            <p className="vi-error" role="alert">
              {produceError}
            </p>
          ) : null}
        </div>
        {/* ── SEGMENT COUNT + GROUP ASSIST BAR (STUDIO-SPREAD-SPEC §2) ──
               Say how many shots the movie has, tick the ones you want written, and
               write them in ONE coherent pass. The count materializes or truncates
               rows; the group actions touch ONLY ticked rows — never a prompt you
               kept, never the identity, images, seeds or budget. */}
        <PromptListToolbar
          count={{
            id: "vi-movie-segment-count",
            label: "Segments",
            value: goals.length,
            min: 1,
            onChange: setSegmentCount,
            title:
              "How many shots this movie has. Lowering it parks rows from the END of the timeline; raising it brings them back before adding blanks.",
          }}
          noun="segment"
          selectedCount={selectedCount}
          itemCount={goals.length}
          allSelected={allSelected}
          onToggleSelectAll={toggleSelectAll}
          generateLabel={
            assist.assistBusy === "spread"
              ? "✨ Writing the spread…"
              : `✨ Generate prompts (${selectedCount} selected)`
          }
          generateTitle="ONE call writes every ticked segment against one shared world; the unticked ones ride along as locked context so the shots belong to the same movie."
          generateDisabledTitle="Tick the segments you want written."
          onGenerate={() => void onSpreadSelected()}
          negativesLabel={
            negBatch
              ? `✨ Negatives ${negBatch.done}/${negBatch.total}…`
              : `✨ Generate negatives (${selectedCount} selected)`
          }
          negativesTitle="Write an artifact/quality exclusion list for each ticked segment, from that segment's own prompt."
          onNegatives={() => void onNegativesSelected()}
          assistBusy={assist.assistBusy != null}
          onStandardNegatives={onStandardNegatives}
          standardNegativesTitle="Tick [standard negative] on for the selected segments (or every segment when none are selected) — the standard artifact/junk set is composed into their negatives at submit."
          disabled={disableSubmit}
        />

        {/* The overall request the spread writes against + the world pin. Both optional:
            blank, the generator works from the rows themselves; pinned, a follow-up
            spread re-rolls a couple of shots INTO THE SAME WORLD instead of a new one. */}
        <div className="vi-knob vi-studio-knob-wide" style={{ marginBottom: "0.6rem" }}>
          <label htmlFor="vi-movie-query">movie idea — the overall request (optional)</label>
          <input
            id="vi-movie-query"
            className="vi-knob-input"
            type="text"
            value={movieQuery}
            placeholder="e.g. a chase through a night market that ends on a rooftop"
            disabled={disableSubmit}
            onChange={(e) => setMovieQuery(e.target.value)}
          />
          {/* k89: the explanation moved into "About Cinema" — one functional line
              (+ the world pin, which needs point-of-use context) stays inline. */}
          <span className="vi-knob-hint">
            Steers “Generate prompts” — one shared world across the ticked segments.
            {steeringSeed != null && (
              <>
                {" "}
                <label style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                  <input
                    type="checkbox"
                    checked={pinSteering}
                    disabled={disableSubmit}
                    onChange={(e) => setPinSteering(e.target.checked)}
                  />
                  keep the last spread’s world (steering seed {steeringSeed})
                </label>
              </>
            )}
          </span>
        </div>

        {/* The spread's RESULT NOTICE — what changed, what did NOT, and every warning
            the backend surfaced (skipped rows, discarded unselected rows, invented
            identity attributes, a model that answered in place of the one requested).
            Dismissible and inert: it reports, it never edits. */}
        {spreadNotice && spreadNotice.length > 0 && (
          <div className="vi-input-callout" role="status" style={{ marginBottom: "0.6rem" }}>
            <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
              {spreadNotice.map((line, i) => (
                <li key={i} style={{ wordBreak: "break-word" }}>
                  {line}
                </li>
              ))}
            </ul>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              style={{ marginTop: "0.35rem" }}
              onClick={() => setSpreadNotice(null)}
            >
              ✕ Dismiss
            </button>
          </div>
        )}

        {/* k121: what the review DID to this set, in one line. Present even when
            nothing changed — "reviewed and fine" has to be visible, because the
            incident this exists for looked exactly like "nothing happened". */}
        {coordination && (
          <CoordinationSummary
            status={coordination.status}
            counts={coordination.counts}
            notes={coordination.notes}
            onDismiss={() => setCoordination(null)}
          />
        )}

        <ol className="vi-gen-parts">
          {goals.map((g, i) => (
            <PromptCard
              key={g.key}
              badge="segment"
              index={i}
              noun="segment"
              /* SELECTION — the group actions may rewrite ticked rows and nothing
                 else. An unticked row rides the spread as LOCKED context. */
              select={{
                checked: selectedKeys.has(g.key),
                onChange: () => toggleSelected(g.key),
                disabled: disableSubmit,
                title:
                  "Tick to let the group actions rewrite this segment. Unticked rows are held as locked context.",
                label: `Select segment ${i + 1} for group prompt assist`,
              }}
              /* The joint descriptor STAYS in the header (the joint SELECT moved
                 into the settings expander below — k88). */
              headerHint={
                <CoordinationBadge
                  review={segmentReview(coordination, g.key)}
                  onAccept={acceptCoordination}
                  onReject={rejectCoordination}
                  settled={coordSettled}
                  disabled={disableSubmit}
                >
                <span className="vi-comfy-hint" style={{ opacity: 0.75 }}>
                  {i === 0
                    ? references.length
                      ? "root · identity render (id_lock)"
                      : startImage
                        ? "root · image → video (from start image)"
                        : "root · text → video"
                    : g.joint === "cut"
                      ? "✂ scene cut · fresh render"
                      : g.joint === "vace_extend"
                        ? "extends motion from the previous segment"
                        : "continues from a frame of the previous segment"}
                </span>
                </CoordinationBadge>
              }
              canMoveUp={i > 0}
              canMoveDown={i < goals.length - 1}
              onMoveUp={() => moveGoal(i, -1)}
              onMoveDown={() => moveGoal(i, 1)}
              moveUpTitle="Move earlier"
              moveDownTitle="Move later"
              canRemove={goals.length > 1}
              onRemove={() => removeGoal(i)}
              removeTitle={
                goals.length <= 1 ? "A movie needs at least one segment" : "Remove segment"
              }
            >
              {/* k88 — [Prompt][Negative] tabs over ONE GoalComposer pane; the
                  [standard negative] tick rides inline with the tabs. The negative
                  pane keeps its own "✨ Generate negative" affordance (§1b). */}
              <PromptFieldTabs
                idPrefix={`vi-movie-${g.key}`}
                promptLabel={`prompt — segment ${i + 1}`}
                prompt={g.prompt}
                onPromptChange={(v) => setGoalField(i, { prompt: v })}
                onPromptBlur={() => classifyRow(g.key, g.prompt)}
                promptPlaceholder={
                  i === 0 ? "describe the opening shot" : "describe how this segment continues"
                }
                rows={2}
                negative={g.negative}
                onNegativeChange={(v) => setGoalField(i, { negative: v })}
                negativePlaceholder="what to avoid — blank = movie-level default"
                negativeHint="Blank falls back to the movie-level negative below; [standard negative] composes the standard set in either way."
                stdNegative={g.stdNegative}
                onStdNegativeChange={(v) => setGoalField(i, { stdNegative: v })}
                negativeExtras={
                  /* NEGATIVE ASSIST (§1b) — its OWN mode, not a prompt call with
                     different words. Seeded by this shot's own prompt as the
                     subject being negated. */
                  <div className="vi-prompt-assist">
                    <span className="vi-prompt-assist-lead" aria-hidden="true">
                      negative assist
                    </span>
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      disabled={assist.assistBusy != null || disableSubmit}
                      title="Write an artifact/quality exclusion list for this shot (replaces this field)."
                      onClick={() => runRowNegative(i)}
                    >
                      {assist.assistBusy === "negative" ? "✨ Writing…" : "✨ Generate negative"}
                    </button>
                  </div>
                }
              />
              {/* ── Auto · Direction · Scene prompt (spec §2/§1d) ──
                     What you type into a prompt box is one of two different things:
                     a SCENE ("a woman enters a red room" — enhance it) or a DIRECTION
                     ("make the framing wider and colder" — write a new prompt FROM it).
                     Word count is backwards on both of those, so in Auto a 3B router
                     classifies on BLUR and the answer LABELS the primary button below.
                     It never runs anything: a router mistake costs one click, never a
                     paragraph of your work. Picking Direction or Scene prompt yourself
                     skips the router entirely. */}
              <div className="vi-prompt-card-assist">
              {(() => {
                const mode: PromptFieldMode = fieldModes[g.key] ?? "auto";
                const intent = intents[g.key] ?? null;
                const blank = g.prompt.trim() === "";
                const op = resolveOperation(mode, intent, blank);
                const detected = mode === "auto" ? detectionText(intent, op) : "";
                return (
                  <div
                    className="vi-prompt-assist"
                    role="group"
                    aria-label={`How to treat segment ${i + 1}'s prompt text`}
                  >
                    <span className="vi-prompt-assist-lead" aria-hidden="true">
                      treat as
                    </span>
                    <select
                      className="vi-select vi-select-sm"
                      value={mode}
                      disabled={disableSubmit}
                      aria-label="Treat this text as a direction or as a scene prompt"
                      onChange={(e) => {
                        const next = e.target.value as PromptFieldMode;
                        setFieldModes((prev) => ({ ...prev, [g.key]: next }));
                        // Switching back to Auto asks the router about the CURRENT text.
                        if (next === "auto") classifyRow(g.key, g.prompt);
                      }}
                    >
                      <option value="auto">Auto</option>
                      <option value="direction">Direction</option>
                      <option value="scene_prompt">Scene prompt</option>
                    </select>
                    {op && (
                      <button
                        type="button"
                        className="vi-btn vi-btn-sm vi-btn-ghost"
                        disabled={assist.assistBusy != null || disableSubmit}
                        title={
                          op === "enhance_scene"
                            ? "Enrich this scene, preserving its subject and intent."
                            : op === "generate_from_direction"
                              ? "Write a new prompt that carries out this direction, using the surrounding shots as context."
                              : "Write a prompt for this shot from scratch."
                        }
                        onClick={() => runRowAssist(i, assistModeFor(op))}
                      >
                        ✨ {operationLabel(op)}
                      </button>
                    )}
                    {intentBusy[g.key] ? (
                      <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
                        reading…
                      </span>
                    ) : (
                      detected && (
                        <span className="vi-comfy-hint" style={{ opacity: 0.75 }}>
                          {detected}
                          {intent && !intent.degraded && intent.confidence > 0
                            ? ` (${Math.round(intent.confidence * 100)}%)`
                            : ""}
                        </span>
                      )
                    )}
                  </div>
                );
              })()}
              {/* PROMPT ASSIST for THIS segment — Enhance enriches this goal's prompt,
                  Generate writes a fresh one. The shared cluster + hook; one in-flight
                  assist at a time across all segments. Errors surface inline and never
                  clobber the goal's existing prompt. Both stay live whatever the router
                  said: the classification above is a default, not a lock.
                  Both now carry TYPED CONTEXT (§1c) — this shot, its neighbours, the
                  locked identity — which is what `context.hint` never carried. */}
              <PromptAssistButtons
                busy={assist.assistBusy}
                error={assist.assistError}
                onDismissError={assist.dismissError}
                onEnhance={() => runRowAssist(i, "detail")}
                onGenerate={() => runRowAssist(i, "generate")}
                canEnhance={g.prompt.trim() !== ""}
                models={assist.assistModels}
                model={assist.assistModel}
                onModelChange={assist.setAssistModel}
              />
              </div>

            {/* SEGMENT-1 INPUTS, IN-CARD (operator 2026-08-13): the movie-start
                still and the identity reference moved INTO the prompt component —
                the same per-part attach idiom Scene uses — decluttering the column.
                Wiring is byte-identical to the old standalone sections. */}
            {i === 0 && (
              <>
        {/* ── Movie start switcher (operator ask 2026-07-12, verbatim): "It needs to be
               choosable by the user: any library clip, image, or scene to start — or
               start fresh with a prompt to generate a clip. This static starting clip is
               not going to work if it cannot be overridden." Same visual row as the Clip
               surface's subject-clip switcher (.vi-studio-src-actions) — four honest ways
               to set (or clear) segment 1's start, none of them a hidden default. ── */}
        <section aria-label="Movie start" style={{ marginBottom: "0.6rem" }}>
          <p
            className="vi-comfy-label"
            style={{ marginBottom: "0.25rem", opacity: 0.85, cursor: "help" }}
            title="How segment 1 opens: pick a library image, use a library clip's last frame, upload a still, or leave unset to open from your first segment's prompt (text-to-video). Never a hidden default — whatever is set is shown here."
          >
            Movie start — segment 1 (optional) ⓘ
          </p>
          <div className="vi-studio-src-actions" role="group" aria-label="Choose how the movie starts">
            <button
              type="button"
              className="vi-btn vi-btn-sm"
              disabled={disableSubmit || startBusy}
              aria-expanded={showStartLibPicker}
              onClick={() => setShowStartLibPicker((v) => !v)}
            >
              {showStartLibPicker ? "Hide library" : `🖼 From library (${startLibraryImages.length})`}
            </button>
            {/* CLIP-AS-START FINDING (stated in the file header above): the movie schema
                has no `start_clip` field, and there is no synchronous "give me this
                clip's last frame" endpoint — frame_extract is its own async job (enqueue
                → poll → pick a frame), not a small addition here. Shown so the operator
                SEES the option exists, disabled so nothing dishonest is implied. */}
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              disabled
              title="Movie-from-clip needs the backend start_clip field — coming. A clip's last frame isn't a quick pick today (frame extraction is its own job); use a library image or upload a still instead."
            >
              🎬 From library (clip → last frame)
            </button>
            <label className="vi-btn vi-btn-sm vi-file-label">
              {startBusy ? "Working…" : "⇪ Upload still"}
              <input
                type="file"
                accept="image/*"
                className="vi-file-input"
                disabled={disableSubmit || startBusy}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void onPickStartImage(f);
                  e.target.value = "";
                }}
              />
            </label>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              disabled={disableSubmit || startBusy}
              title="Clear the start still — segment 1 opens from its own prompt instead (text-to-video)."
              onClick={() => {
                setStartImage(null);
                setShowStartLibPicker(false);
              }}
            >
              ✦ Fresh prompt
            </button>
          </div>

          {/* Current start state — ALWAYS visible, so the picked (or default) start is
              never implicit (the operator's exact complaint about the old static slot). */}
          {startImage ? (
            <div
              style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap", marginTop: "0.5rem" }}
            >
              <img
                src={mediaBytesUrl(startImage.uri)}
                alt="segment 1 start image"
                style={{
                  width: "5rem",
                  height: "5rem",
                  objectFit: "cover",
                  borderRadius: "0.4rem",
                  background: "#000",
                  border: "1px solid var(--vi-border)",
                }}
              />
              <span className="vi-comfy-hint" role="note" style={{ margin: 0 }}>
                Segment 1 renders image-to-video from this still.
              </span>
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost"
                disabled={disableSubmit || startBusy}
                onClick={() => setStartImage(null)}
              >
                ✕ Remove
              </button>
            </div>
          ) : (
            <p className="vi-comfy-hint" role="note" style={{ marginTop: "0.5rem" }}>
              No start still — text-to-video opening.
            </p>
          )}

          {startError && (
            <p className="vi-error" role="alert" style={{ marginTop: "0.3rem" }}>
              {startError}
            </p>
          )}

          {showStartLibPicker && (
            <LibraryImageGrid
              images={startLibraryImages}
              onPick={(ref) => {
                setStartImage(ref);
                setShowStartLibPicker(false);
              }}
            />
          )}
        </section>
        {/* ── IDENTITY reference(s): the locked subject carried across scene changes ── */}
        <section aria-label="Identity reference" style={{ marginBottom: "0.6rem" }}>
          <p
            className="vi-comfy-label"
            style={{ marginBottom: "0.25rem", opacity: 0.85, cursor: "help" }}
            title="Leave empty for a plain movie. Add a locked character/subject still (or pick a saved identity profile) and EVERY scene renders id_lock (Wan-VACE) so the same identity carries across every cut. Needs a VACE-capable GPU worker; a GPU-less box reports deps-missing."
          >
            Identity reference — carry a subject across scenes (optional) ⓘ
          </p>
          {/* The reference thumbnails + upload AND the identity-profile affordances read as
              ONE input block — matching how the Clip Generate tab embeds the same
              <IdentityProfileControls> inline, directly beneath its references
              (StudioGenerateTab.renderReferenceSlots), rather than letting it stand off as a
              separate island row. Presentation/placement only: the props + wiring below are
              byte-identical to before. */}
          {/* ONE ROW (operator 2026-08-13): thumbnails, upload AND the profile
              controls share a single wrapping flex row — no stacked button rows. */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
            <div style={{ display: "contents" }}>
              {references.map((r, i) => (
                <span key={r.uri} style={{ position: "relative", display: "inline-flex" }}>
                  <img
                    src={mediaBytesUrl(r.uri)}
                    alt={`identity reference ${i + 1}`}
                    style={{
                      width: "4rem",
                      height: "4rem",
                      objectFit: "cover",
                      borderRadius: "0.4rem",
                      background: "#000",
                      border: "1px solid var(--vi-border)",
                    }}
                  />
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-ghost"
                    disabled={disableSubmit || refBusy}
                    title="Remove this reference"
                    onClick={() => {
                      setReferences((prev) => prev.filter((x) => x.uri !== r.uri));
                      setSelectedProfile(null); // editing the set = an UNSAVED identity
                    }}
                    style={{ position: "absolute", top: -6, right: -6, padding: "0 0.3rem", lineHeight: 1.2 }}
                  >
                    ✕
                  </button>
                </span>
              ))}
              {references.length < REF_MAX && (
                <label className="vi-btn vi-btn-sm vi-btn-ghost vi-file-label">
                  {refBusy ? "Working…" : references.length ? "＋ Add reference" : "Upload a reference"}
                  <input
                    type="file"
                    accept="image/*"
                    className="vi-file-input"
                    disabled={disableSubmit || refBusy}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void onPickReferenceImage(f);
                      e.target.value = "";
                    }}
                  />
                </label>
              )}
            </div>
            {/* IDENTITY PROFILES (stage a): pick a saved identity profile or save the current
                references — the durable form of this identity. One additive row, embedded
                inline within the reference input block (matching the Clip Generate tab). */}
            <IdentityProfileControls
              refCount={references.length}
              currentRefUris={references.map((r) => r.uri)}
              selectedProfile={selectedProfile}
              onAttach={attachIdentityRefs}
              onSelectProfile={setSelectedProfile}
              maxRefs={REF_MAX}
              disabled={disableSubmit || refBusy}
            />
          </div>
          {/* RESOLUTION FOOTGUN (operator 2026-07-12): the only budget-fitting id-capable
              model (wan2.1-vace-1.3b) tops out at 832×480 — an identity movie above that fails
              no_capable_model. Warn LOUDLY with a one-click snap; never BLOCK (the backend's
              rejection is honest if the operator insists on a bigger box). */}
          {(references.length > 0 || selectedProfile != null) && (width > 832 || height > 480) && (
            <p className="vi-input-callout" role="note" style={{ marginTop: "0.4rem" }}>
              id-capable models on this fleet top out at 832×480 — this movie asks for {width}×{height}.{" "}
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-accent"
                onClick={() => {
                  setWidth(480);
                  setHeight(480);
                }}
                style={{ marginLeft: "0.3rem" }}
              >
                Set 480×480
              </button>
            </p>
          )}
          {references.length > 0 && (
            <span className="vi-comfy-hint" role="note" style={{ display: "block", marginTop: "0.3rem" }}
                  title="Every scene renders id_lock (Wan-VACE) so this subject carries across every cut. Needs a VACE-capable GPU worker.">
              Identity movie — id_lock every scene · {references.length}/{REF_MAX} reference(s)
            </span>
          )}
          {refError && (
            <p className="vi-error" role="alert" style={{ marginTop: "0.3rem" }}>
              {refError}
            </p>
          )}
        </section>
              </>
            )}
            {/* k88 — the shared settings expander (collapsed by default); the joint
                DESCRIPTOR stays visible in the card header above.
                2026-08-12: grew from seed-only to the FULL per-segment knob set the
                route has always accepted (StudioMovieGoal): model pin, steps, cfg,
                and (vace_extend rows) context frames. Every knob blank ⇒ the key is
                dropped and the segment inherits the movie-level value — Movie-tab
                (k92) parity, minus the share ticks (here the movie level IS the
                shared value, so "inherit" already says it). */}
            <PromptCardSettings
              label={
                i === 0
                  ? "settings — seed · model · steps · cfg · length"
                  : "settings — seed · model · steps · cfg · length · joint · branch"
              }
            >
              <div className="vi-knob" style={{ maxWidth: "10rem" }}>
                <label htmlFor={`vi-movie-seed-${g.key}`}>seed (optional)</label>
                <input
                  id={`vi-movie-seed-${g.key}`}
                  className="vi-knob-input"
                  type="number"
                  min={0}
                  value={g.seed}
                  placeholder="movie seed + index"
                  onChange={(e) => setGoalField(i, { seed: e.target.value })}
                />
              </div>
              <div className="vi-knob" style={{ maxWidth: "16rem" }}>
                <label htmlFor={`vi-movie-goalmodel-${g.key}`}>model — this segment</label>
                <select
                  id={`vi-movie-goalmodel-${g.key}`}
                  className="vi-knob-select"
                  value={g.modelId}
                  onChange={(e) => setGoalField(i, { modelId: e.target.value })}
                >
                  <option value="">
                    {modelId.trim() ? `Inherit — ${modelId.trim()}` : "Inherit — router picks"}
                  </option>
                  {movieModels.length > 0 && (
                    <optgroup label="Bound by a ratified movie segment">
                      {movieModels.map((id) => (
                        <option key={id} value={id}>
                          {id}
                        </option>
                      ))}
                    </optgroup>
                  )}
                  {(showAllModels || !renderLoaded || (g.modelId !== "" && !movieModels.includes(g.modelId))) && (
                    <optgroup label={renderLoaded ? "Other studio models — unverified for a movie segment" : "Studio models"}>
                      {STUDIO_MODEL_IDS.filter((id) => !movieModels.includes(id)).map((id) => (
                        <option key={id} value={id}>
                          {id}
                          {renderLoaded ? " · unverified" : ""}
                        </option>
                      ))}
                    </optgroup>
                  )}
                </select>
                <span className="vi-knob-hint">
                  Pin a model for THIS segment only. Inherit = the movie-level pin, else the
                  capability router. An unroutable pin errors on the job, never silently swaps.
                </span>
              </div>
              <div className="vi-knob" style={{ maxWidth: "8rem" }}>
                <label htmlFor={`vi-movie-goalsteps-${g.key}`}>steps</label>
                <input
                  id={`vi-movie-goalsteps-${g.key}`}
                  className="vi-knob-input"
                  type="number"
                  min={1}
                  max={100}
                  value={g.steps}
                  placeholder={steps.trim() || "inherit"}
                  onChange={(e) => setGoalField(i, { steps: e.target.value })}
                />
                <span className="vi-knob-hint">1–100 · blank = movie/model default</span>
              </div>
              <div className="vi-knob" style={{ maxWidth: "8rem" }}>
                <label htmlFor={`vi-movie-goalcfg-${g.key}`}>cfg</label>
                <input
                  id={`vi-movie-goalcfg-${g.key}`}
                  className="vi-knob-input"
                  type="number"
                  min={0}
                  max={20}
                  step={0.5}
                  value={g.cfg}
                  placeholder={cfg.trim() || "inherit"}
                  onChange={(e) => setGoalField(i, { cfg: e.target.value })}
                />
                <span className="vi-knob-hint">0–20 · blank = movie/model default</span>
              </div>
              {/* LENGTH ROW (per-goal): fps is MOVIE-level here (fixed/grey) —
                  frames<->seconds derive against it. Blank = movie default. */}
              <LengthRow
                idPrefix={`vi-movie-goal-len-${g.key}`}
                fps={fps}
                fpsFixedReason="Movie-level — one fps per movie; change it in movie settings above."
                frames={g.frames}
                onFrames={(v) => setGoalField(i, { frames: v })}
                framesPlaceholder={framesDefault.trim() || "movie default (81)"}
              />
              {Number(g.frames.trim() || framesDefault.trim()) > MODEL_FRAME_CEILING && (
                <div className="vi-knob" style={{ flexBasis: "100%" }}>
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm"
                    title="The model ceiling is 81 frames — materialize motion-carried continuation segments (vace_extend) so the movie honestly reaches this length. They appear in the timeline, fully editable."
                    onClick={() => splitGoalToFit(i)}
                  >
                    ✂ split into {Math.ceil(Number(g.frames.trim() || framesDefault.trim()) / MODEL_FRAME_CEILING)} segments to fit
                  </button>
                </div>
              )}
              {/* UNIFORM-CARD PLACEHOLDERS (operator 2026-08-13): geometry and
                  budget are MOVIE-level by server contract (one StudioMovieSpec
                  geometry per movie) — shown greyed with the reason, never
                  hidden, so every prompt card reads identically. */}
              <div className="vi-knob" style={{ maxWidth: "7rem" }}>
                <label htmlFor={`vi-movie-goalw-${g.key}`}>width</label>
                <input id={`vi-movie-goalw-${g.key}`} className="vi-knob-input" disabled value={width}
                       title="Movie-level — one geometry per movie (StudioMovieSpec); change it in movie settings above." />
              </div>
              <div className="vi-knob" style={{ maxWidth: "7rem" }}>
                <label htmlFor={`vi-movie-goalh-${g.key}`}>height</label>
                <input id={`vi-movie-goalh-${g.key}`} className="vi-knob-input" disabled value={height}
                       title="Movie-level — one geometry per movie (StudioMovieSpec); change it in movie settings above." />
              </div>
              <div className="vi-knob" style={{ maxWidth: "9rem" }}>
                <label htmlFor={`vi-movie-goalvram-${g.key}`}>VRAM budget GB</label>
                <input id={`vi-movie-goalvram-${g.key}`} className="vi-knob-input" disabled
                       value={vramBudget.trim() || "autofit"}
                       title="Movie-level — set in movie settings above (blank = autofit to the serving worker's free VRAM)." />
              </div>
              {i > 0 && (
                <>
                  <div className="vi-knob" style={{ maxWidth: "13rem" }}>
                    <label htmlFor={`vi-movie-joint-${g.key}`}>joint — how it follows</label>
                    <select
                      id={`vi-movie-joint-${g.key}`}
                      className="vi-knob-input"
                      value={g.joint}
                      onChange={(e) => setGoalField(i, { joint: e.target.value as JointMode })}
                    >
                      <option value="still">{JOINT_LABELS.still}</option>
                      <option value="vace_extend">{JOINT_LABELS.vace_extend}</option>
                      <option value="cut">{JOINT_LABELS.cut}</option>
                    </select>
                    <span className="vi-knob-hint">
                      {g.joint === "cut"
                        ? "Hard scene cut — the previous segment plays in full; this one is a fresh render (identity carries when a reference is set)."
                        : g.joint === "vace_extend"
                          ? "Carry motion across the join from the previous segment's trailing frames (needs a VACE GPU worker)."
                          : "Continue from ONE frame of the previous segment (motion not carried)."}
                    </span>
                  </div>
                  <div className="vi-knob" style={{ maxWidth: "16rem" }}>
                    <label htmlFor={`vi-movie-branch-${g.key}`}>
                      branch from frame — blank = last frame
                    </label>
                    <input
                      id={`vi-movie-branch-${g.key}`}
                      className="vi-knob-input"
                      type="number"
                      min={0}
                      value={g.joint === "cut" ? "" : g.branchFrame}
                      placeholder={g.joint === "cut" ? "n/a — a scene cut carries no frame" : "last frame"}
                      disabled={g.joint === "cut"}
                      title={g.joint === "cut" ? "A scene cut carries no frame — the previous segment plays in full." : undefined}
                      onChange={(e) => setGoalField(i, { branchFrame: e.target.value })}
                    />
                    <span className="vi-knob-hint">
                      {g.joint === "cut"
                        ? "A scene cut carries no frame — the previous segment plays in full."
                        : "Frame index into the previous segment's clip (0-based). Blank ⇒ its last frame (parent plays in full)."}
                    </span>
                  </div>
                  {g.joint === "vace_extend" && (
                    <div className="vi-knob" style={{ maxWidth: "10rem" }}>
                      <label htmlFor={`vi-movie-goalctx-${g.key}`}>context frames</label>
                      <input
                        id={`vi-movie-goalctx-${g.key}`}
                        className="vi-knob-input"
                        type="number"
                        min={1}
                        value={g.contextFrames}
                        placeholder="movie default"
                        onChange={(e) => setGoalField(i, { contextFrames: e.target.value })}
                      />
                      <span className="vi-knob-hint">
                        How many of the parent&apos;s trailing frames carry motion through this
                        VACE join. Blank = the movie-level default.
                      </span>
                    </div>
                  )}
                </>
              )}
            </PromptCardSettings>
          </PromptCard>
          ))}
        </ol>

        {/* Timeline foot — the Add-segment affordance, mirroring the reference
            .vi-movie-timeline-foot (Add goal + summary) in GenerateStation Movie mode. */}
        <div className="vi-movie-timeline-foot">
          <button type="button" className="vi-btn vi-btn-ghost" onClick={addGoal} disabled={disableSubmit}>
            ＋ Add segment
          </button>
          <span className="vi-movie-summary">
            {goals.length} segment{goals.length === 1 ? "" : "s"}
          </span>
        </div>
      </section>


      {/* ── SYNTHETIC-TIER HONESTY BANNER — loud, right above Generate movie, so the
             operator sees it at the moment it matters instead of the knob-grid
             footnote three fields up. Identity movies get the positive note instead
             (they can never bind synthetic — see wouldBindSynthetic above); a plain
             movie under the real floor gets the warning + one-click fill. ── */}
      {references.length > 0 ? (
        <p className="vi-comfy-hint" role="note" style={{ margin: "0 0 0.6rem" }}>
          ⓘ identity segments render on the real tier automatically.
        </p>
      ) : (
        wouldBindSynthetic && (
          <div
            className="vi-studio-required-callout"
            role="alert"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.6rem",
              flexWrap: "wrap",
              marginBottom: "0.6rem",
            }}
          >
            <span>
              ⚠ This will render the SYNTHETIC TEST tier (colored noise, no real model). Set VRAM
              budget ≥ {movieFloorGb} for a real render on the GPU worker
              {renderLoaded ? " — the measured envelope of the cheapest real segment binding" : ""}.
            </span>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-accent"
              disabled={disableSubmit}
              onClick={() => {
                budgetTouched.current = true;
                setVramBudget(String(movieNeedGb));
              }}
            >
              Set budget to {movieNeedGb}
            </button>
          </div>
        )
      )}

      {/* ── UNNAMED-MOVIE NUDGE — a warning with a way through, never a gate. Sits
             right above Generate because that is where the decision is made. ── */}
      {nameWarn && (
        <UnnamedWarnPanel
          noun="movie"
          onProceed={(name) => {
            setNameWarn(false);
            onGenerate(true, name);
          }}
          onDismiss={() => setNameWarn(false)}
        />
      )}

      {/* ── run row ── */}
      <div className="vi-station-footer vi-studio-run">
        <div className="vi-station-footer-main">
          {formMsg && (
            <p className="vi-knob-flag" role="alert" style={{ margin: 0 }}>
              {formMsg}
            </p>
          )}
          {job.status && !formMsg && (
            <p className="vi-knob-hint" role="status" style={{ margin: 0 }}>
              {job.running
                ? `Rendering — status: ${job.status}${
                    job.progress?.stage ? ` · ${job.progress.stage}` : ""
                  }`
                : `Status: ${job.status}`}
              {job.jobId ? ` · job ${job.jobId.slice(0, 8)}` : ""}
            </p>
          )}
          {/* LIVE PACE (k91) — the bus feed's folded fraction + "segment 2/14 · step
              18/32" + the freshest stage line, so the first minute of a long render
              already says how fast it is going. Mounted only while this movie runs (the
              component owns the poll), and it renders nothing the server didn't send.
              Wrapped in a WRAP ROW because the bar's shared classes are written for the
              Active-Processes row (`flex-basis: 100%`), and this footer column would
              read that basis as a HEIGHT — the exact trap app.css documents on
              `.vi-station-footer`. */}
          {job.running ? (
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                gap: "0.3rem 0.6rem",
                width: "100%",
                minWidth: 0,
              }}
            >
              <MovieLiveProgress jobId={job.jobId} />
            </div>
          ) : null}
        </div>
        <div className="vi-station-footer-run">
          <button
            type="button"
            className="vi-btn vi-btn-accent"
            disabled={disableSubmit}
            onClick={() => onGenerate()}
          >
            {job.running ? "Rendering…" : "Generate movie"}
          </button>
        </div>
      </div>

      {/* ── B3 pending spliced row: one block per authored goal, tinted by live
             per-segment progress (equal widths — effective frames aren't known until
             each clip exists; the sized row + scrubber arrive on done). ── */}
      {job.running && (
        <section aria-label="Rendering timeline" style={{ marginTop: "0.8rem" }}>
          <StudioPendingRow
            goalPrompts={goals.map((g) => g.prompt)}
            progressSegments={(job.progress?.segments ?? []).map((s) => ({
              index: s.index,
              status: s.status ?? undefined,
              resumed: s.resumed ?? undefined,
            }))}
          />
        </section>
      )}

      {/* ── live per-segment progress (render what the payload gives, honestly) ── */}
      {job.running && job.progress && (
        <section aria-label="Progress" style={{ marginTop: "0.8rem" }}>
          <p className="vi-comfy-label" style={{ marginBottom: "0.3rem" }}>
            {job.progress.stage ?? "working"} · segment {job.progress.segment_done ?? 0}/
            {job.progress.segment_total ?? goals.length}
            {job.progress.eta_s != null ? ` · ~${Math.round(job.progress.eta_s)}s left` : ""}
          </p>
          <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: "0.3rem" }}>
            {(job.progress.segments ?? []).map((s) => (
              <li
                key={s.index}
                style={{ display: "flex", gap: "0.6rem", alignItems: "baseline", fontSize: "0.85rem" }}
              >
                <span style={{ fontVariantNumeric: "tabular-nums", opacity: 0.7, minWidth: "5.5rem" }}>
                  seg {s.index + 1} · {statusBadge(s.status, s.resumed)}
                </span>
                <span style={{ opacity: 0.85, wordBreak: "break-word" }}>
                  {s.prompt ?? ""}
                  {s.refresh_note ? (
                    <span
                      className="vi-comfy-hint"
                      style={{ display: "block", opacity: 0.75 }}
                      title={
                        s.prompt_authored
                          ? `authored prompt: ${s.prompt_authored}`
                          : undefined
                      }
                    >
                      🎬 {s.refresh_note}
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ── failure: JobError as DATA (no toast-and-swallow) ── */}
      {isFailed && (
        <section aria-label="Movie error" style={{ marginTop: "0.8rem" }}>
          <p className="vi-error" role="alert">
            <strong>{job.status === "cancelled" ? "Cancelled" : "Failed"}</strong>
            {job.result?.error?.code ? ` [${job.result.error.code}]` : ""}
            {job.result?.error?.retryable ? " · retryable" : ""}
            {job.error ? <span style={{ display: "block" }}>{job.error}</span> : null}
          </p>
          {job.result?.movie && (
            <p className="vi-comfy-hint" role="note" style={{ marginTop: "0.3rem" }}>
              A partial movie may remain on disk:{" "}
              {job.result.movie.segments_completed ?? 0}/{job.result.movie.segments_total ?? goals.length}{" "}
              segment(s) completed
              {job.result.project && typeof (job.result.project as { dir?: unknown }).dir === "string"
                ? ` · ${(job.result.project as { dir: string }).dir}`
                : ""}
              .
            </p>
          )}
        </section>
      )}

      {/* ── done: play movie.mp4 in the viewer idiom, list segments below ── */}
      {isDone && view && (
        <section aria-label="Movie" style={{ marginTop: "0.9rem", display: "flex", flexDirection: "column", gap: "0.7rem" }}>
          {view.movieRef ? (
            <div className="vi-studio-viewer-wrap">
              <div className="vi-studio-viewer">
                <video
                  key={view.movieRef.uri}
                  className="vi-studio-clip-video"
                  controls
                  autoPlay
                  loop
                  playsInline
                  src={mediaBytesUrl(view.movieRef.uri)}
                />
              </div>
              <p className="vi-comfy-hint" role="note" style={{ marginTop: "0.3rem" }}>
                Assembled movie.mp4
                {view.result.movie?.assembly?.total_frames != null
                  ? ` · ${view.result.movie.assembly.total_frames} frames`
                  : ""}
                {" "}— {view.segmentRefs.length} segment(s) stitched at their splice points.
              </p>
            </div>
          ) : (
            <p className="vi-comfy-hint" role="note">
              Segments rendered but the assembled movie.mp4 is not in the outputs (assembly
              may not have produced one) — the segment clips are below.
            </p>
          )}

          {view.result.movie?.drift && (
            <p className="vi-comfy-hint" role="note" style={{ fontStyle: "italic", opacity: 0.8 }}>
              ⓘ {view.result.movie.drift}
            </p>
          )}

          {/* ── B3: the visual SPLICED ROW + branch-frame scrubber — the operator's
                 "spliced video conjoined at those points". Blocks sized to EFFECTIVE
                 frames (parent trimmed at its child's splice), scissor notch per join,
                 click a block to scrub + branch a new goal. Rendered ONCE (never in a
                 .map) so no `key` lands on the custom component. ── */}
          {view.segs.length > 0 && (
            <StudioSplicedRow
              segments={view.segs}
              joints={view.result.movie?.joints ?? []}
              segmentRefs={view.segmentRefs}
              fps={view.result.movie?.fps ?? 0}
              fpsFallback={fps}
              goalCount={goals.length}
              onBranch={onBranchFromSegment}
            />
          )}

          <div>
            <p className="vi-comfy-label" style={{ marginBottom: "0.35rem" }}>
              Segment clips
            </p>
            <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: "0.6rem" }}>
              {view.segmentRefs.map((ref, i) => {
                const seg = view.segs[i];
                return (
                  <li
                    key={ref.uri}
                    style={{ display: "flex", gap: "0.7rem", alignItems: "flex-start" }}
                  >
                    <video
                      src={`${mediaBytesUrl(ref.uri)}#t=0.1`}
                      controls
                      muted
                      playsInline
                      preload="metadata"
                      style={{
                        width: "12rem",
                        maxWidth: "40vw",
                        borderRadius: "0.4rem",
                        background: "#000",
                        border: "1px solid var(--vi-border)",
                      }}
                    />
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.15rem", fontSize: "0.85rem" }}>
                      <span style={{ fontWeight: 600 }}>
                        Segment {i + 1}
                        {seg?.segment_id ? ` · ${seg.segment_id}` : ""}
                        {seg?.resumed ? " · resumed" : ""}
                      </span>
                      <span style={{ opacity: 0.85 }}>{jointText(view.segs, i)}</span>
                      {seg?.frames != null && (
                        <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
                          {seg.frames} frames
                          {seg.width != null && seg.height != null ? ` · ${seg.width}×${seg.height}` : ""}
                          {seg.duration_s != null ? ` · ${Math.round(seg.duration_s)}s` : ""}
                        </span>
                      )}
                      {seg?.prompt && (
                        <span className="vi-comfy-hint" style={{ opacity: 0.7, wordBreak: "break-word" }}>
                          “{seg.prompt}”
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}
