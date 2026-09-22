// The VISUAL SPLICED ROW + branch-frame scrubber (slice B3) — the picture half of
// the operator's ask: "in the video edit row it displays a spliced video that can
// easily be conjoined at those points … the splice being the parent video |
// divergence, and the user's options decide."
//
// A studio movie is ONE NLE timeline row: `[segment 0 | segment 1 | …]`. This draws
// that row from the terminal manifest — each block sized to its EFFECTIVE frames in
// the assembly (a parent trimmed at its child's splice; the leaf full), with a
// scissors NOTCH at every join labelled with the parent frame the child diverged
// at. Clicking a rendered block opens a frame scrubber over THAT segment's clip; its
// one primary action appends a new branch goal to the composer's timeline.
//
// V1 BOUNDARIES (stated plainly; carried in the composer header too):
//   • The row reads the LAST render. Branching edits the AUTHORED goal timeline and
//     the user re-Generates — a studio job is IMMUTABLE, so extending a done movie
//     re-enqueues a NEW job (it never mutates the old one).
//   • v0 authors a LINEAR chain. The schema's take-tree (sibling divergence from one
//     parent) is planned growth; until then branching from a NON-final block cannot
//     keep the old tail as a sibling, so it TRUNCATES (see the composer's onBranch +
//     the destructive-note wiring below).
//
// Rendered ONCE by the composer (never inside a .map) so no `key` ever lands on this
// custom component — the repo's JSX typing rejects that (see StudioGenerateTab). All
// internal lists key on intrinsic elements.
import { useEffect, useRef, useState } from "react";
import { mediaBytesUrl } from "../../config";
import type { MediaRef } from "../../video/contract";
import {
  deriveRow,
  resolveFps,
  clampFrame,
  frameToTime,
  timeToFrame,
  type RowSegment,
  type RowJoint,
} from "./movieTimeline";

export interface StudioSplicedRowProps {
  /** Terminal manifest segments (index-aligned with `segmentRefs`). */
  segments: RowSegment[];
  /** Terminal manifest joints (per-splice trim ledger). */
  joints: RowJoint[];
  /** Per-segment clip MediaRefs, in order (outputs sans the trailing movie.mp4). */
  segmentRefs: MediaRef[];
  /** Movie fps from the manifest (falls back to the composer knob). */
  fps: number;
  fpsFallback: number;
  /** Authored goal-row count RIGHT NOW — clean append (last) vs truncate (mid-row). */
  goalCount: number;
  /** Append a branch goal after `segmentIndex`, conditioned on `frame` of its clip. */
  onBranch: (segmentIndex: number, frame: number) => void;
}

const BLOCK_MIN_H = "3.4rem";

export function StudioSplicedRow({
  segments,
  joints,
  segmentRefs,
  fps,
  fpsFallback,
  goalCount,
  onBranch,
}: StudioSplicedRowProps) {
  const effFps = resolveFps(fps, fpsFallback);
  const { blocks, splices } = deriveRow(segments, joints);
  const spliceAfter = new Map<number, (typeof splices)[number]>();
  splices.forEach((s) => spliceAfter.set(s.afterIndex, s));

  // Which block's scrubber is open (by segment index), and its live frame cursor.
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [frame, setFrame] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const openBlock = blocks.find((b) => b.index === openIndex) ?? null;
  const openRef = openIndex != null ? segmentRefs[openIndex] : undefined;
  const openFullFrames = openBlock?.fullFrames ?? null;

  // A fresh segment resets the cursor to frame 0.
  useEffect(() => {
    setFrame(0);
    const v = videoRef.current;
    if (v) {
      try {
        v.currentTime = 0;
      } catch {
        /* not yet seekable — the metadata load handles the first paint */
      }
    }
  }, [openIndex]);

  function seekTo(f: number) {
    const clamped = clampFrame(f, openFullFrames);
    setFrame(clamped);
    const v = videoRef.current;
    if (v) {
      try {
        v.currentTime = frameToTime(clamped, effFps);
      } catch {
        /* ignore — a not-yet-ready element seeks on its own metadata load */
      }
    }
  }

  // The native <video> scrubber is a second way to move the cursor: mirror its time
  // back into the frame readout/slider. We NEVER re-seek here (only explicit slider /
  // step handlers seek), so there is no feedback loop.
  function syncFromVideo() {
    const v = videoRef.current;
    if (!v) return;
    const f = clampFrame(timeToFrame(v.currentTime, effFps), openFullFrames);
    setFrame((cur) => (cur === f ? cur : f));
  }

  const maxFrame = openFullFrames != null && openFullFrames > 0 ? openFullFrames - 1 : 0;
  // Branching from a NON-final authored row cannot keep the old tail as a sibling in a
  // linear v0 — it truncates. Surface that in the action so the click is never silent.
  const droppedCount = openIndex != null ? Math.max(0, goalCount - (openIndex + 1)) : 0;
  const willTruncate = droppedCount > 0;

  return (
    <section aria-label="Spliced timeline row" style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <p className="vi-comfy-label" style={{ margin: 0 }}>
        Spliced timeline — {blocks.length} segment(s), conjoined at {splices.length} splice
        {splices.length === 1 ? "" : "s"}
      </p>

      {/* ── the row: blocks sized to EFFECTIVE frames, scissor notch at each join ── */}
      <div
        role="list"
        aria-label="Movie segments"
        style={{
          display: "flex",
          alignItems: "stretch",
          width: "100%",
          border: "1px solid var(--vi-border)",
          borderRadius: "0.5rem",
          overflow: "hidden",
          background: "#000",
        }}
      >
        {/* One flatMap over the blocks: each yields its block button, then (when the
            block is a joint parent) a splice NOTCH — flat siblings in the flex row, each
            keyed on its own intrinsic element (never on a custom component). */}
        {blocks.flatMap((b, i) => {
          const ref = segmentRefs[b.index];
          const isOpen = openIndex === b.index;
          const splice = spliceAfter.get(b.index);
          const pct = Math.max(2, b.widthPct); // never let a hard-trimmed block vanish
          const blockEl = (
            <button
              key={b.segmentId ?? `blk-${b.index}`}
              type="button"
              role="listitem"
              onClick={() => setOpenIndex((cur) => (cur === b.index ? null : b.index))}
              title={
                ref
                  ? `Segment ${b.index + 1} — scrub & branch (${b.effectiveFrames} frame${
                      b.effectiveFrames === 1 ? "" : "s"
                    } in the cut${b.trimmed ? ", trimmed at the splice" : ""})`
                  : `Segment ${b.index + 1} — no streamable clip`
              }
              aria-pressed={isOpen}
              disabled={!ref}
              style={{
                position: "relative",
                flex: `${pct} 1 0`,
                minWidth: 0,
                minHeight: BLOCK_MIN_H,
                padding: 0,
                margin: 0,
                border: "none",
                borderRight:
                  i < blocks.length - 1 && !splice ? "1px solid var(--vi-border)" : "none",
                cursor: ref ? "pointer" : "default",
                background: isOpen ? "var(--vi-accent)" : "transparent",
                outline: isOpen ? "2px solid var(--vi-accent)" : "none",
                outlineOffset: "-2px",
                overflow: "hidden",
              }}
            >
              {ref ? (
                <video
                  src={`${mediaBytesUrl(ref.uri)}#t=0.1`}
                  muted
                  playsInline
                  preload="metadata"
                  aria-hidden="true"
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    opacity: isOpen ? 0.55 : 0.8,
                  }}
                />
              ) : null}
              <span
                aria-hidden="true"
                style={{
                  position: "absolute",
                  left: 4,
                  top: 3,
                  fontSize: "0.7rem",
                  fontWeight: 700,
                  fontVariantNumeric: "tabular-nums",
                  color: "#fff",
                  textShadow: "0 1px 2px #000",
                  padding: "0 0.25rem",
                  borderRadius: "0.25rem",
                  background: "rgba(0,0,0,0.35)",
                }}
              >
                {b.index + 1}
                {b.trimmed ? " ✂" : ""}
              </span>
              <span
                aria-hidden="true"
                style={{
                  position: "absolute",
                  left: 4,
                  bottom: 3,
                  fontSize: "0.62rem",
                  fontVariantNumeric: "tabular-nums",
                  color: "#fff",
                  textShadow: "0 1px 2px #000",
                  opacity: 0.9,
                }}
              >
                {b.effectiveFrames}f
              </span>
            </button>
          );
          if (!splice) return [blockEl];
          // A "cut" splice is a HARD scene cut: the parent plays in FULL (no trim) and the
          // next segment is a fresh render — in an identity movie the SUBJECT carries across
          // the cut even though no frame does. Every other splice trims the parent at a frame.
          const isCut = splice.mode === "cut";
          const notchAria = isCut
            ? "Scene cut — identity carried, no frame carry"
            : `Splice — parent frame ${splice.branchFrame}`;
          const notchTitle = isCut
            ? "✂ scene cut — identity carried, no frame carry (the parent plays in full; the next segment is a fresh render)"
            : `✂ Splice at frame ${splice.branchFrame} — the parent is trimmed to ${splice.trimFrames} frame${
                splice.trimFrames === 1 ? "" : "s"
              }; the next segment diverges here.`;
          const notch = (
            <span
              key={`splice-${splice.afterIndex}-${splice.childIndex}`}
              role="separator"
              aria-label={notchAria}
              title={notchTitle}
              style={{
                flex: "0 0 auto",
                width: "1.15rem",
                alignSelf: "stretch",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background:
                  "repeating-linear-gradient(45deg, var(--vi-accent) 0 3px, transparent 3px 6px)",
                borderLeft: "1px solid var(--vi-border)",
                borderRight: "1px solid var(--vi-border)",
                color: "#fff",
                fontSize: "0.7rem",
                cursor: "help",
              }}
            >
              ✂
            </span>
          );
          return [blockEl, notch];
        })}
      </div>

      {/* ── the branch-frame scrubber (opens over the clicked segment's clip) ── */}
      {openBlock && openRef ? (
        <div
          style={{
            border: "1px solid var(--vi-border)",
            borderRadius: "0.5rem",
            padding: "0.6rem 0.7rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontWeight: 600 }}>
              Scrub segment {openBlock.index + 1}
              {openBlock.capability ? ` · ${openBlock.capability}` : ""}
            </span>
            <span style={{ flex: 1 }} />
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              onClick={() => setOpenIndex(null)}
              title="Close scrubber"
            >
              ✕
            </button>
          </div>

          <video
            key={openRef.uri}
            ref={videoRef}
            src={mediaBytesUrl(openRef.uri)}
            controls
            playsInline
            preload="metadata"
            onSeeked={syncFromVideo}
            onTimeUpdate={syncFromVideo}
            onLoadedMetadata={syncFromVideo}
            style={{
              width: "100%",
              maxHeight: "40vh",
              borderRadius: "0.4rem",
              background: "#000",
            }}
          />

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              onClick={() => seekTo(frame - 1)}
              disabled={frame <= 0}
              title="Step back one frame"
            >
              −1
            </button>
            <input
              type="range"
              min={0}
              max={maxFrame}
              step={1}
              value={Math.min(frame, maxFrame)}
              onChange={(e) => seekTo(Number(e.target.value))}
              disabled={maxFrame <= 0}
              aria-label="Frame"
              style={{ flex: "1 1 12rem", minWidth: "8rem" }}
            />
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              onClick={() => seekTo(frame + 1)}
              disabled={maxFrame > 0 && frame >= maxFrame}
              title="Step forward one frame"
            >
              +1
            </button>
            <span
              className="vi-knob-hint"
              style={{ fontVariantNumeric: "tabular-nums", minWidth: "9rem", textAlign: "right" }}
            >
              frame {frame}
              {openFullFrames != null ? ` / ${openFullFrames - 1}` : ""} · t=
              {frameToTime(frame, effFps).toFixed(2)}s
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
            <button
              type="button"
              className="vi-btn vi-btn-accent"
              onClick={() => onBranch(openBlock.index, frame)}
            >
              Branch from frame {frame}
              {willTruncate ? ` — replaces segment${droppedCount === 1 ? "" : "s"} ${openBlock.index + 2}+` : ""}
            </button>
            <span className="vi-comfy-hint" role="note" style={{ margin: 0 }}>
              {willTruncate
                ? `Adds a new segment branching from frame ${frame} of this clip and DROPS the ${droppedCount} authored segment(s) after it — a linear timeline can't hold the old tail as a sibling take (take-trees are the next slice). Then edit its prompt and re-Generate.`
                : `Adds a new segment branching i2v from frame ${frame} of this clip. Then edit its prompt and re-Generate — a studio job is immutable, so this renders a NEW extended movie.`}
            </span>
          </div>
        </div>
      ) : null}
    </section>
  );
}

// ── the PENDING row (shown WHILE a movie renders): one block per authored goal,
//    tinted by live per-segment progress. Equal widths — effective frames aren't
//    known until each clip exists. Rendered once by the composer, never in a .map. ──
export interface StudioPendingRowProps {
  goalPrompts: string[];
  /** Clean per-segment progress (the composer normalizes the zod-passthrough poll blob
   *  into this shape before passing it — keeps the row free of zod-internal typing). */
  progressSegments?: Array<{
    index: number;
    status?: string;
    resumed?: boolean;
  }>;
}

export function StudioPendingRow({ goalPrompts, progressSegments }: StudioPendingRowProps) {
  const statusByIndex = new Map<number, string>();
  (progressSegments ?? []).forEach((s) => {
    const label = s.resumed ? "resumed" : s.status ?? "";
    statusByIndex.set(s.index, label);
  });

  // Static tints only — the "generating" block reads as active via a brighter fill
  // (no CSS keyframe: app.css is read-only in this slice, so nothing animated is added).
  function tint(status: string): { bg: string; active: boolean } {
    if (status === "done" || status === "resumed") return { bg: "var(--vi-accent)", active: false };
    if (status === "generating" || status === "branching")
      return { bg: "rgba(138,180,255,0.5)", active: true };
    return { bg: "rgba(255,255,255,0.06)", active: false }; // pending / queued
  }

  return (
    <section aria-label="Spliced timeline row (rendering)" style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
      <p className="vi-comfy-label" style={{ margin: 0 }}>
        Spliced timeline — rendering {goalPrompts.length} segment(s)
      </p>
      <div
        role="list"
        style={{
          display: "flex",
          alignItems: "stretch",
          width: "100%",
          border: "1px solid var(--vi-border)",
          borderRadius: "0.5rem",
          overflow: "hidden",
        }}
      >
        {goalPrompts.map((prompt, i) => {
          const status = statusByIndex.get(i) ?? "";
          const { bg, active } = tint(status);
          return (
            <div
              key={`pending-${i}`}
              role="listitem"
              title={`Segment ${i + 1}${status ? ` · ${status}` : " · pending"}${prompt ? ` — ${prompt}` : ""}`}
              style={{
                position: "relative",
                flex: "1 1 0",
                minWidth: 0,
                minHeight: BLOCK_MIN_H,
                background: bg,
                borderRight: i < goalPrompts.length - 1 ? "1px solid var(--vi-border)" : "none",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.3rem",
                boxShadow: active ? "inset 0 0 0 2px var(--vi-accent)" : undefined,
              }}
            >
              <span
                style={{
                  fontSize: "0.7rem",
                  fontWeight: 700,
                  fontVariantNumeric: "tabular-nums",
                  color: "#fff",
                  textShadow: "0 1px 2px #000",
                }}
              >
                {i + 1}
              </span>
              <span style={{ fontSize: "0.62rem", color: "#fff", opacity: 0.85, textShadow: "0 1px 2px #000" }}>
                {status || "pending"}
              </span>
            </div>
          );
        })}
      </div>
      <p className="vi-comfy-hint" role="note" style={{ margin: 0 }}>
        Blocks fill as each segment renders; the spliced row with per-splice frame markers
        and the branch scrubber appear once the movie completes.
      </p>
    </section>
  );
}
