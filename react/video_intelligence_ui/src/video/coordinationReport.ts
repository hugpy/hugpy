// COORDINATION REPORT (k121) — the wire shape of the words-vs-knobs review.
//
// The backend kernel is `video_intel/prompt_coordination.py`. Every generated or
// enhanced video prompt path now returns a `coordination` block:
// `/video/prompt/assist` (spread + detail/generate) and `POST /video/studio/movie`.
//
// WHY THE UI HAS TO SHOW THIS. The operator's cinema session produced four
// prompts that all declared continuity — continuous scenes, a scene drawn from
// the previous scene's render, a recurring character — with every joint left at
// `cut`, no parent link, no lengthening and no identity. Nothing failed; it
// rendered the wrong film. The backend now turns the ratchet-safe knobs itself,
// but a knob turned invisibly is the same defect wearing the other hat, so every
// decision surfaces on the row it belongs to, with the operator's OWN WORDS as
// the evidence.
//
// Pure: no React, no fetch. Reader-shaped like `spreadAssist.readSpreadResult` —
// anything malformed is dropped, never defaulted into a false "ok".

/** What the review did about one knob. Worst-first, matching `STATUS_ORDER`. */
export type CoordinationStatus = "mismatch" | "proposed" | "set" | "ok";

/** Worst-first. Folding a segment's decisions into one badge picks the first hit. */
export const COORDINATION_STATUS_ORDER: CoordinationStatus[] = [
  "mismatch",
  "proposed",
  "set",
  "ok",
];

/** One claim the prose made. `evidence_quote` is verbatim from the row's text. */
export interface CoordinationExpectation {
  kind: string;
  segments: string[];
  evidence_quote: string;
  confidence: number;
  source: string;
  detail: Record<string, unknown>;
}

/** One knob decision — the diff a badge renders. */
export interface CoordinationDecision {
  segment_id: string;
  index: number;
  knob: string;
  current: unknown;
  proposed: unknown;
  status: CoordinationStatus;
  reason: string;
  confidence: number;
  blocking: boolean;
  evidence_quote: string;
  /** Set only for a work STEP that is not a knob (`IDENTITY_CAPTURE`). */
  step?: string;
  expectation?: CoordinationExpectation | null;
  detail?: Record<string, unknown>;
}

/** One row of the review. EVERY segment gets one, including the `ok` ones. */
export interface CoordinationSegment {
  segment_id: string;
  index: number;
  status: CoordinationStatus;
  decisions: CoordinationDecision[];
}

export interface CoordinationReport {
  version: string;
  status: CoordinationStatus;
  counts: Record<string, number>;
  llm_used: boolean;
  notes: string[];
  segments: CoordinationSegment[];
  expectations: CoordinationExpectation[];
  decisions: CoordinationDecision[];
}

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function num(v: unknown, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function statusOf(v: unknown): CoordinationStatus {
  const s = str(v);
  return (COORDINATION_STATUS_ORDER as string[]).includes(s)
    ? (s as CoordinationStatus)
    : "ok";
}

function dict(v: unknown): Record<string, unknown> {
  return v != null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
}

function readExpectation(v: unknown): CoordinationExpectation | null {
  const r = dict(v);
  const kind = str(r.kind);
  if (!kind) return null;
  return {
    kind,
    segments: Array.isArray(r.segments) ? r.segments.map(str).filter(Boolean) : [],
    evidence_quote: str(r.evidence_quote),
    confidence: num(r.confidence),
    source: str(r.source) || "rule",
    detail: dict(r.detail),
  };
}

function readDecision(v: unknown): CoordinationDecision | null {
  const r = dict(v);
  const segment_id = str(r.segment_id);
  const knob = str(r.knob);
  if (!segment_id || !knob) return null;
  const step = str(r.step);
  return {
    segment_id,
    index: num(r.index),
    knob,
    current: r.current,
    proposed: r.proposed,
    status: statusOf(r.status),
    reason: str(r.reason),
    confidence: num(r.confidence),
    blocking: r.blocking === true,
    evidence_quote: str(r.evidence_quote),
    ...(step ? { step } : {}),
    expectation: readExpectation(r.expectation),
    detail: dict(r.detail),
  };
}

/**
 * Read a `coordination` block. Returns null when it is not a report at all — the
 * caller then shows nothing rather than a badge built out of guesses.
 */
export function readCoordinationReport(value: unknown): CoordinationReport | null {
  const v = dict(value);
  if (!Array.isArray(v.segments) || !Array.isArray(v.decisions)) return null;
  const decisions = v.decisions
    .map(readDecision)
    .filter((d): d is CoordinationDecision => d != null);
  const segments: CoordinationSegment[] = [];
  for (const raw of v.segments) {
    const r = dict(raw);
    const segment_id = str(r.segment_id);
    if (!segment_id) continue;
    segments.push({
      segment_id,
      index: num(r.index),
      status: statusOf(r.status),
      decisions: Array.isArray(r.decisions)
        ? r.decisions.map(readDecision).filter((d): d is CoordinationDecision => d != null)
        : [],
    });
  }
  return {
    version: str(v.version),
    status: statusOf(v.status),
    counts: dict(v.counts) as Record<string, number>,
    llm_used: v.llm_used === true,
    notes: Array.isArray(v.notes) ? v.notes.map(str).filter(Boolean) : [],
    segments,
    expectations: Array.isArray(v.expectations)
      ? v.expectations
          .map(readExpectation)
          .filter((e): e is CoordinationExpectation => e != null)
      : [],
    decisions,
  };
}

/** The review row for one segment, or null when it was not reviewed. */
export function segmentReview(
  report: CoordinationReport | null,
  segmentId: string,
): CoordinationSegment | null {
  if (!report) return null;
  return report.segments.find((s) => s.segment_id === segmentId) ?? null;
}

/** The one-word badge label for a status. */
export function coordinationLabel(status: CoordinationStatus): string {
  switch (status) {
    case "mismatch":
      return "knobs ≠ words";
    case "proposed":
      return "knobs proposed";
    case "set":
      return "knobs set";
    default:
      return "knobs ok";
  }
}

/**
 * The existing assist-log badge palette, reused rather than reinvented:
 * `--err` red / `--warn` amber / `--ok` green / base dim. One badge language
 * across the console (`style/app.css` §.vi-assistlog-badge).
 */
export function coordinationClass(status: CoordinationStatus): string {
  switch (status) {
    case "mismatch":
      return "vi-assistlog-badge vi-assistlog-badge--err";
    case "proposed":
      return "vi-assistlog-badge vi-assistlog-badge--warn";
    case "set":
      return "vi-assistlog-badge vi-assistlog-badge--ok";
    default:
      return "vi-assistlog-badge";
  }
}

/** Human name for a knob, for the diff line. */
export function knobLabel(knob: string): string {
  switch (knob) {
    case "joint_mode":
      return "join";
    case "parent_segment_id":
      return "parent";
    case "branch_frame":
      return "branch frame";
    case "reference_images":
      return "identity refs";
    case "frames":
      return "clip length";
    default:
      return knob;
  }
}

/** Render a knob value for the diff line. Never throws on an odd shape. */
export function knobValue(v: unknown): string {
  if (v == null) return "—";
  if (Array.isArray(v)) return v.length ? `${v.length} ref(s)` : "—";
  if (typeof v === "object") {
    const o = v as Record<string, unknown>;
    if (typeof o.profile === "string") return `profile ${o.profile}`;
    if (typeof o.from_segment === "string") return `capture from ${o.from_segment}`;
    return "…";
  }
  return String(v);
}

/**
 * A knob decision projected onto the movie composer's own GoalRow field names.
 * `null` when the knob has no editable home on this surface (a parent pointer is
 * implicit in the composer's linear chain; an identity plan is a work step, not
 * a field), so accepting it can never write a field that does not exist.
 */
export function knobPatch(d: CoordinationDecision): Record<string, string> | null {
  switch (d.knob) {
    case "joint_mode":
      return typeof d.proposed === "string" ? { joint: d.proposed } : null;
    case "seed":
      return typeof d.proposed === "number" ? { seed: String(d.proposed) } : null;
    case "frames":
      return typeof d.proposed === "number" ? { frames: String(d.proposed) } : null;
    case "branch_frame":
      return { branchFrame: d.proposed == null ? "" : String(d.proposed) };
    default:
      return null;
  }
}
