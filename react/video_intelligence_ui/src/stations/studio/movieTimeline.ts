// Studio-movie TIMELINE MATH (slice B3) — the pure, JSX-free half of the spliced
// row. Given the terminal movie manifest's `segments` + `joints` (the honest
// non-destructive NLE ledger the runner writes; see studio_movie.py
// `_assemble_movie` / `_write_movie_json`), derive the strip the row draws:
//   • each block's EFFECTIVE frames in the assembly — a PARENT trimmed at its
//     child's splice contributes `trim_frames = branch_frame + 1`; the final
//     (leaf) segment plays its FULL `frames`. This mirrors the runner's
//     `contributions` list EXACTLY, so a block's width == its real share of the
//     stitched movie.mp4 (never the whole untrimmed clip).
//   • the splice markers that sit BETWEEN blocks — one per joint, labelled with
//     the frame N of the parent the child diverged at.
//
// Kept pure + presentation-free so the width/joint math is inspectable on its own
// (the row component only lays these numbers out). Structural param types
// (`RowSegment`/`RowJoint`) are subsets of the composer's zod-parsed manifest
// shapes, so the composer passes its `ManifestSegment[]` / joints straight in.

/** A subset of the movie manifest's per-segment node the row reads (see
 *  StudioMovieComposer's `manifestSegmentSchema`). */
export interface RowSegment {
  index: number;
  segment_id?: string | null;
  prompt?: string | null;
  capability?: string | null;
  branch_frame?: number | null;
  resolved_branch?: number | null;
  frames?: number | null;
  width?: number | null;
  height?: number | null;
  duration_s?: number | null;
  resumed?: boolean | null;
  status?: string | null;
}

/** A subset of the manifest's per-joint trim record (see `manifestJointSchema`). */
export interface RowJoint {
  parent_segment_id?: string | null;
  child_segment_id?: string | null;
  branch_frame?: number | null;
  trim_frames?: number | null;
  /** Splice mode the runner recorded: "still" | "vace_extend" | "cut". A "cut" carries no
   *  frame (branch_frame null, trim_frames == the parent's full length). */
  mode?: string | null;
}

/** One laid-out block of the spliced row. */
export interface RowBlock {
  index: number;
  segmentId: string | null;
  /** Frames this segment contributes to the assembled movie (trimmed or full). */
  effectiveFrames: number;
  /** The clip's real (untrimmed) frame count — the scrubber's slider bound. */
  fullFrames: number | null;
  /** Share of the assembled row, 0..100 (sums to ~100 across blocks). */
  widthPct: number;
  /** True when this parent is trimmed at a child's splice (a real cut happened). */
  trimmed: boolean;
  prompt: string | null;
  capability: string | null;
  status: string | null;
  resumed: boolean;
}

/** A splice marker sitting AFTER `afterIndex` (between two blocks). */
export interface RowSplice {
  afterIndex: number;
  childIndex: number;
  /** Frame N of the parent the child was conditioned on. null for a "cut" (no frame carry). */
  branchFrame: number | null;
  /** Frames of the parent kept in the assembly = branchFrame + 1 (the parent's FULL length
   *  for a "cut", which trims nothing). */
  trimFrames: number | null;
  /** Splice mode: "still" | "vace_extend" | "cut" — drives the notch label + tooltip. */
  mode: string;
}

export interface DerivedRow {
  blocks: RowBlock[];
  splices: RowSplice[];
  totalEffectiveFrames: number;
}

function num(v: number | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * Derive the spliced-row layout from the manifest's segments + joints.
 *
 * Effective frames per block (mirrors runners/studio_movie.py `_assemble_movie`):
 *   • a segment that is a JOINT PARENT keeps `trim_frames` (= branch_frame + 1);
 *   • the leaf/last segment keeps its FULL `frames`.
 * Widths are proportional to those effective counts. When no honest frame count
 * is available (e.g. a manifest without `frames`), blocks fall back to EQUAL
 * widths so the row still reads as a strip rather than collapsing.
 */
export function deriveRow(segments: RowSegment[], joints: RowJoint[]): DerivedRow {
  const segs = [...segments].sort((a, b) => a.index - b.index);

  // Trim lookup keyed by the joint's PARENT segment id (linear chain: one child
  // per parent). Also index each joint's child position for splice placement.
  const jointByParent = new Map<string, RowJoint>();
  for (const j of joints) {
    if (typeof j.parent_segment_id === "string") jointByParent.set(j.parent_segment_id, j);
  }
  const indexById = new Map<string, number>();
  segs.forEach((s, i) => {
    if (typeof s.segment_id === "string") indexById.set(s.segment_id, i);
  });

  const raw: number[] = segs.map((s) => {
    const joint = typeof s.segment_id === "string" ? jointByParent.get(s.segment_id) : undefined;
    const trim = num(joint?.trim_frames);
    if (trim != null && trim > 0) return trim;
    const full = num(s.frames);
    return full != null && full > 0 ? full : 0;
  });

  let total = raw.reduce((a, b) => a + b, 0);
  // Fallback: no honest counts anywhere → equal widths (still a legible strip).
  const equal = total <= 0;
  if (equal) total = segs.length || 1;

  const blocks: RowBlock[] = segs.map((s, i) => {
    const eff = equal ? 1 : raw[i];
    const joint = typeof s.segment_id === "string" ? jointByParent.get(s.segment_id) : undefined;
    return {
      index: s.index,
      segmentId: typeof s.segment_id === "string" ? s.segment_id : null,
      effectiveFrames: eff,
      fullFrames: num(s.frames),
      widthPct: (eff / total) * 100,
      // A "cut" parent plays in FULL (trim_frames == full) — it is NOT trimmed, so the
      // block shows no ✂ trim badge (the CUT is drawn as the splice notch below instead).
      trimmed: !!joint && joint.mode !== "cut" && num(joint.trim_frames) != null,
      prompt: typeof s.prompt === "string" ? s.prompt : null,
      capability: typeof s.capability === "string" ? s.capability : null,
      status: typeof s.status === "string" ? s.status : null,
      resumed: s.resumed === true,
    };
  });

  const splices: RowSplice[] = [];
  for (const j of joints) {
    const parentId = j.parent_segment_id;
    const childId = j.child_segment_id;
    if (typeof parentId !== "string" || typeof childId !== "string") continue;
    const afterIndex = indexById.get(parentId);
    const childIndex = indexById.get(childId);
    if (afterIndex == null || childIndex == null) continue;
    const mode = typeof j.mode === "string" ? j.mode : "still";
    const branchFrame = num(j.branch_frame);
    const trimFrames = num(j.trim_frames);
    // A "cut" carries NO frame (branch_frame null) but STILL draws a notch — a hard scene
    // cut. Every other splice needs a real branch frame (a null is a malformed record).
    if (mode === "cut") {
      splices.push({ afterIndex, childIndex, branchFrame: null, trimFrames, mode: "cut" });
      continue;
    }
    if (branchFrame == null) continue;
    splices.push({
      afterIndex,
      childIndex,
      branchFrame,
      trimFrames: trimFrames ?? branchFrame + 1,
      mode,
    });
  }
  splices.sort((a, b) => a.afterIndex - b.afterIndex);

  return { blocks, splices, totalEffectiveFrames: total };
}

/** Movie fps: the manifest value when present, else the composer's knob fallback. */
export function resolveFps(movieFps: number | null | undefined, fallback: number): number {
  const f = num(movieFps ?? null);
  return f != null && f > 0 ? f : fallback;
}

/** Clamp a frame index into a clip's [0, frames-1] range. */
export function clampFrame(frame: number, frames: number | null | undefined): number {
  const max = num(frames ?? null);
  const hi = max != null && max > 0 ? max - 1 : 0;
  if (!Number.isFinite(frame)) return 0;
  return Math.max(0, Math.min(hi, Math.round(frame)));
}

/** Seconds a frame index lands at, for seeking a <video> (frame / fps). */
export function frameToTime(frame: number, fps: number): number {
  const f = fps > 0 ? fps : 1;
  return frame / f;
}

/** Nearest frame index for a <video>'s currentTime (round to the grid). */
export function timeToFrame(time: number, fps: number): number {
  const f = fps > 0 ? fps : 1;
  return Math.round(time * f);
}
