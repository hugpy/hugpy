// STUDIO SPREAD — the pure half (STUDIO-SPREAD-SPEC §2).
//
// Everything here is a FUNCTION OF ITS INPUTS: wire shapes, request builders,
// tolerant response readers and the intent→action mapping. No React, no fetch, no
// component state. The transport lives in usePromptAssist (one in-flight assist,
// one error model); the composer keeps only its editor state. Splitting it this way
// keeps StudioMovieComposer from growing a second brain, and makes the two things
// most likely to be wrong — the wire shape and the intent routing table — readable
// in one screen.
//
// THE ONE RULE THAT SHAPES ALL OF IT (spec §1a/§1d): a classification is a
// LABEL, never an action, and a spread reply may only touch rows the user
// SELECTED. Every reader below is written so a malformed/partial/over-eager reply
// degrades to "change nothing and say so" rather than to "overwrite the movie".

/** How a segment is spliced onto the one before it — mirrors studio_movie_schema. */
export type SpreadJointMode = "still" | "vace_extend" | "cut";

/**
 * One typed segment reference (spec §1c). Field names mirror
 * `studio_movie_schema.StudioMovieGoal` EXACTLY so a row goes over the wire with no
 * translation layer to get wrong. Every field bar `segment_id` is optional; the
 * backend validates and renders them into the model preface as plain sentences.
 */
export interface SegmentRefWire {
  segment_id: string;
  prompt?: string;
  negative?: string;
  /** The user's INSTRUCTION for this row, when the router said "direction". */
  direction?: string;
  joint_mode?: SpreadJointMode;
  branch_frame?: number;
  seed?: number;
  index?: number;
}

/** The locked identity block (§1e). `name` is required by the backend validator. */
export interface IdentityContextWire {
  identity_id?: string;
  name: string;
  reference_asset_ids?: string[];
  locked_description?: string;
  do_not_invent?: string[];
}

/**
 * k93 §C — ONE reference media the assist is given as PRETEXT. The backend
 * samples ≤ 4 frames of a video (one description for an image), describes them
 * with a vision model and prepends a "Reference video: …" paragraph to the
 * assist prompt. `uri` is the library uri; `mime` "video/mp4" | "image/*".
 */
export interface AssistMediaWire {
  uri: string;
  mime: string;
  label?: string;
}

/** The typed `context` object shared by detail/generate/negative/spread. */
export interface AssistContextWire {
  kind?: "image" | "scene" | "movie";
  /** FREE-FORM user text only — structured row state rides the typed fields (§1c). */
  hint?: string;
  /** Optional reference media described server-side as pretext (k93 §C). */
  media?: AssistMediaWire;
  segment?: SegmentRefWire;
  previous_segment?: SegmentRefWire;
  next_segment?: SegmentRefWire;
  identity_profile?: IdentityContextWire;
}

/** The `mode:"spread"` request body (§1a), minus `model` (the hook adds that). */
export interface SpreadRequestWire {
  mode: "spread";
  movie_query?: string;
  style_bible?: Record<string, string>;
  fixed_segments: SegmentRefWire[];
  target_segments: SegmentRefWire[];
  global_negative?: string[];
  steering_seed?: number;
  context?: AssistContextWire;
}

/** One replacement row the generator wrote. Only TARGET ids ever appear here. */
export interface SpreadSegmentResult {
  segment_id: string;
  operation: string;
  prompt: string;
  negative: string;
  continuity_note: string;
  directions_used: unknown[];
  warnings: string[];
  /** k121: the knob values the coordination review APPLIED to this row
      (joint_mode / parent_segment_id / branch_frame / seed / frames /
      reference_images). Mechanics only — never prose (invariant 9). */
  knobs: Record<string, unknown>;
}

/** The parsed 200 body of a spread call. */
export interface SpreadResult {
  segments: SpreadSegmentResult[];
  missing_segments: string[];
  warnings: string[];
  invented_identity_attributes: string[];
  /** The shared steering set + its seed — echoed so a liked spread can be re-pinned. */
  steering: Record<string, unknown>;
  steering_seed: number | null;
  model: string | null;
  model_requested: string | null;
  model_resolved: string | null;
  /** k121: the words-vs-knobs review of the whole set. Read it with
      `readCoordinationReport` (video/coordinationReport.ts). */
  coordination: unknown;
}

// ── intent routing (§1d) ────────────────────────────────────────────────────
/** What the 3B router can say. `ambiguous` is also the degraded/failure value. */
export type IntentName = "empty" | "direction" | "scene_prompt" | "ambiguous";

/** The parsed /video/prompt/intent 200 body. This route ALWAYS returns 200. */
export interface IntentResult {
  intent: IntentName;
  operation: string | null;
  confidence: number;
  cached: boolean;
  degraded: boolean;
}

/** The per-field tri-state control: Auto · Direction · Scene prompt. */
export type PromptFieldMode = "auto" | "direction" | "scene_prompt";

/** The resolved action for a prompt field — null means "uncertain, show both". */
export type PromptOperation = "generate" | "enhance_scene" | "generate_from_direction";

/**
 * Resolve which action a row's primary button should invoke.
 *
 * Routing order is the spec's (§1d), in order: blank field → generate; an EXPLICIT
 * user mode → obey it (the router is never consulted); else the router's answer;
 * else — including `ambiguous`, a degraded router, or "not classified yet" — null,
 * which the UI renders as "Uncertain → choose an action" with BOTH buttons live.
 */
export function resolveOperation(
  mode: PromptFieldMode,
  intent: IntentResult | null,
  promptIsBlank: boolean,
): PromptOperation | null {
  if (promptIsBlank) return "generate";
  if (mode === "direction") return "generate_from_direction";
  if (mode === "scene_prompt") return "enhance_scene";
  if (!intent) return null;                       // not classified yet — no guess
  if (intent.intent === "empty") return "generate";
  if (intent.intent === "direction") return "generate_from_direction";
  if (intent.intent === "scene_prompt") return "enhance_scene";
  return null;                                    // ambiguous / degraded
}

/** Which /video/prompt/assist mode an operation invokes. */
export function assistModeFor(op: PromptOperation): "detail" | "generate" {
  return op === "enhance_scene" ? "detail" : "generate";
}

/** The button label for an operation (the detected verb the spec names). */
export function operationLabel(op: PromptOperation): string {
  if (op === "enhance_scene") return "Enhance prompt";
  if (op === "generate_from_direction") return "Generate from direction";
  return "Generate prompt";
}

/**
 * The honest one-line readout under a field in AUTO mode. Returns "" when there is
 * nothing to say yet (blank field, or no classification has been asked for).
 */
export function detectionText(
  intent: IntentResult | null,
  op: PromptOperation | null,
): string {
  if (!intent) return "";
  if (intent.intent === "empty") return "Empty → Generate prompt";
  if (op == null) {
    return intent.degraded
      ? "Router unavailable → choose an action"
      : "Uncertain → choose an action";
  }
  const what = intent.intent === "direction" ? "Direction" : "Scene prompt";
  return `Detected: ${what} → ${operationLabel(op)}`;
}

// ── request builders ────────────────────────────────────────────────────────
/** Drop blank strings / undefined so the wire carries only what the user actually set. */
function put<T extends object>(obj: T, key: keyof T & string, value: unknown): void {
  if (value === undefined || value === null) return;
  if (typeof value === "string" && value.trim() === "") return;
  (obj as Record<string, unknown>)[key] = typeof value === "string" ? value.trim() : value;
}

/** A composer goal row, in the shape this module needs (host-agnostic). */
export interface SpreadGoalInput {
  /** Stable per-row id — the composer's React key doubles as the segment_id. */
  key: string;
  prompt: string;
  negative: string;
  seed: string;
  branchFrame: string;
  joint: SpreadJointMode;
  /** Timeline position (0 = the root segment). */
  index: number;
  /** Set when the row's resolved operation is "generate from direction". */
  isDirection?: boolean;
}

/** Parse a blank-means-unset numeric editor string into an int, or undefined. */
function intOrUndef(raw: string): number | undefined {
  const s = raw.trim();
  if (s === "") return undefined;
  const n = Number(s);
  if (!Number.isInteger(n) || n < 0) return undefined;
  return n;
}

/**
 * One goal row → one typed segment reference.
 *
 * `branch_frame` is dropped for a `cut` (a cut carries no frame, and the backend
 * would render a sentence about a frame that does not exist). A row classified as a
 * DIRECTION sends its text under BOTH `prompt` (what is in the box now) and
 * `direction` (the instruction to act on) — the preface renders them distinctly.
 */
export function segmentRef(goal: SpreadGoalInput): SegmentRefWire {
  const ref: SegmentRefWire = { segment_id: goal.key };
  put(ref, "prompt", goal.prompt);
  put(ref, "negative", goal.negative);
  if (goal.isDirection) put(ref, "direction", goal.prompt);
  if (goal.index > 0) {
    ref.joint_mode = goal.joint;
    if (goal.joint !== "cut") {
      const bf = intOrUndef(goal.branchFrame);
      if (bf !== undefined) ref.branch_frame = bf;
    }
  }
  const seed = intOrUndef(goal.seed);
  if (seed !== undefined) ref.seed = seed;
  ref.index = goal.index;
  return ref;
}

export interface BuildSpreadOptions {
  goals: SpreadGoalInput[];
  /** Row keys the user SELECTED — these and only these may be rewritten. */
  selectedKeys: ReadonlySet<string>;
  /** The overall request ("a chase through a night market"). Optional. */
  movieQuery: string;
  /** The movie-wide negative, split into exclusion terms. */
  globalNegative: string;
  /** Pin a previous spread's world by reusing its steering seed. */
  steeringSeed?: number | null;
  context?: AssistContextWire;
  styleBible?: Record<string, string>;
}

/**
 * Build the ONE spread call (§1a): selected rows are `target_segments`, every other
 * row rides as a LOCKED `fixed_segment` so the generator can see the whole timeline
 * and write into it coherently. Returns null when nothing is selected — the caller
 * must not fire a call the backend would 400 for being empty.
 */
export function buildSpreadBody(opts: BuildSpreadOptions): SpreadRequestWire | null {
  const targets = opts.goals.filter((g) => opts.selectedKeys.has(g.key));
  if (targets.length === 0) return null;
  const fixed = opts.goals.filter((g) => !opts.selectedKeys.has(g.key));
  const body: SpreadRequestWire = {
    mode: "spread",
    fixed_segments: fixed.map(segmentRef),
    target_segments: targets.map(segmentRef),
  };
  put(body, "movie_query", opts.movieQuery);
  const negatives = opts.globalNegative
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "");
  if (negatives.length) body.global_negative = negatives;
  if (opts.steeringSeed != null && Number.isInteger(opts.steeringSeed)) {
    body.steering_seed = opts.steeringSeed;
  }
  if (opts.styleBible && Object.keys(opts.styleBible).length) {
    body.style_bible = opts.styleBible;
  }
  if (opts.context && Object.keys(opts.context).length) body.context = opts.context;
  return body;
}

// ── response readers (tolerant; a bad shape changes NOTHING) ────────────────
function strArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((x): x is string => typeof x === "string") : [];
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/**
 * Read a spread 200 body. Returns null if it is not a spread reply at all (the
 * caller then reports a malformed response rather than applying anything). Rows
 * without a usable `segment_id` + `prompt` are DROPPED, never defaulted.
 */
export function readSpreadResult(value: unknown): SpreadResult | null {
  if (value == null || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  if (!Array.isArray(v.segments)) return null;
  const segments: SpreadSegmentResult[] = [];
  for (const raw of v.segments) {
    if (raw == null || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    const id = str(r.segment_id).trim();
    const prompt = str(r.prompt).trim();
    if (!id || !prompt) continue;
    segments.push({
      segment_id: id,
      operation: str(r.operation) || "generate",
      prompt,
      negative: str(r.negative).trim(),
      continuity_note: str(r.continuity_note).trim(),
      directions_used: Array.isArray(r.directions_used) ? r.directions_used : [],
      warnings: strArray(r.warnings),
      knobs:
        r.knobs != null && typeof r.knobs === "object" && !Array.isArray(r.knobs)
          ? (r.knobs as Record<string, unknown>)
          : {},
    });
  }
  const seed = v.steering_seed;
  return {
    segments,
    missing_segments: strArray(v.missing_segments),
    warnings: strArray(v.warnings),
    invented_identity_attributes: strArray(v.invented_identity_attributes),
    steering:
      v.steering != null && typeof v.steering === "object"
        ? (v.steering as Record<string, unknown>)
        : {},
    steering_seed: typeof seed === "number" && Number.isInteger(seed) ? seed : null,
    model: typeof v.model === "string" ? v.model : null,
    model_requested: typeof v.model_requested === "string" ? v.model_requested : null,
    model_resolved: typeof v.model_resolved === "string" ? v.model_resolved : null,
    coordination: v.coordination ?? null,
  };
}

/**
 * Read an intent 200 body. Anything unrecognized reads as a DEGRADED `ambiguous`,
 * which the UI renders as "choose an action" — the same, safe, both-buttons state
 * a real router outage produces. There is no failure mode here that arms a button.
 */
export function readIntentResult(value: unknown): IntentResult {
  const degraded: IntentResult = {
    intent: "ambiguous",
    operation: null,
    confidence: 0,
    cached: false,
    degraded: true,
  };
  if (value == null || typeof value !== "object") return degraded;
  const v = value as Record<string, unknown>;
  const name = v.intent;
  if (name !== "empty" && name !== "direction" && name !== "scene_prompt" && name !== "ambiguous") {
    return degraded;
  }
  return {
    intent: name,
    operation: typeof v.operation === "string" ? v.operation : null,
    confidence: typeof v.confidence === "number" ? v.confidence : 0,
    cached: v.cached === true,
    degraded: v.degraded === true,
  };
}

/**
 * The non-destructive notice a spread produces: what changed, what did NOT, and
 * every warning the backend surfaced. Rendered as a dismissible panel — the spec's
 * "surface warnings/missing/invented non-destructively", i.e. never as an overwrite
 * and never swallowed.
 */
export function spreadNoticeLines(result: SpreadResult, appliedKeys: string[]): string[] {
  const lines: string[] = [];
  lines.push(
    appliedKeys.length === 1
      ? "1 segment rewritten."
      : `${appliedKeys.length} segments rewritten.`,
  );
  if (result.missing_segments.length) {
    lines.push(
      `The generator did not write ${result.missing_segments.length} selected ` +
        "row(s) — they are unchanged.",
    );
  }
  if (result.invented_identity_attributes.length) {
    lines.push(
      "⚠ Invented identity attributes reported: " +
        result.invented_identity_attributes.join(", ") +
        " — check these against your identity profile.",
    );
  }
  for (const w of result.warnings) lines.push(w);
  for (const s of result.segments) {
    for (const w of s.warnings) lines.push(`${s.segment_id}: ${w}`);
  }
  // PROVENANCE (§1f): say plainly when another model answered.
  if (
    result.model_requested &&
    result.model_resolved &&
    result.model_requested !== result.model_resolved
  ) {
    lines.push(
      `⚠ Answered by ${result.model_resolved}, not the requested ` +
        `${result.model_requested}.`,
    );
  }
  return lines;
}
