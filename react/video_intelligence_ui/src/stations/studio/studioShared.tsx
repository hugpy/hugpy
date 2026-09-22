// Shared internals of the Studio WORKSPACE (the sub-tabbed promotion of the old
// single-page Studio Clips station). This module holds the pieces BOTH sub-tabs
// (Generate, Library) read, so neither tab re-declares them and the wire schemas
// stay in one place:
//
//   • the tolerant zod schemas + types for the studio clip list and the per-clip
//     detail (manifest / requested-spec / error), extended for the new detail
//     `source` discriminator + bus timestamps the route now returns;
//   • the small display helpers (shortId / dims / whenText / fmtWhen);
//   • DetailRow + ClipDetailsPanel — the creation-parameters expander, carried
//     verbatim from the original station and taught to LABEL a job-record view as
//     REQUESTED (a failed/cancelled render bound no model, so its params are what
//     was asked for, not what ran);
//   • useStudioClips — the durable-catalog list + light background poll + the
//     Session-Library push, extracted from the station so the shell owns it once
//     and both tabs read the same list;
//   • the studio constants (the VACE v2v envelope, a curated model-pin list, and
//     the synthetic-vs-real preset test).
//
// Nothing here is new behavior — it is the original station's own code, factored
// so the workspace can grow two rich tabs without duplicating the contract.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import { hugpyConfig, studioClipDetailUrl, mediaBytesUrl } from "../../config";
import { addToLibrary, updateLibraryProvenance } from "../../video/mediaLibrary";
import type { LibraryItem } from "../../video/mediaLibrary";
import type { MediaRef } from "../../video/contract";
import type { StudioPreset } from "../../video/useStudioPresets";

// ── studio constants ───────────────────────────────────────────────────────

// v2v RESTYLE — the ONLY geometry the Wan 2.1 VACE 1.3B control model accepts:
// 832x480 LANDSCAPE @16fps at a 6 GB budget. The studio router rejects a
// portrait/oversized v2v request (RESOLUTION_UNSUPPORTED), so restyle mode LOCKS
// the form to this envelope; nothing dead-on-arrival can be enqueued.
export const VACE_W = 832;
export const VACE_H = 480;
export const VACE_FPS = 16;
export const VACE_BUDGET_GB = 6;

// ── honesty banner (operator ask 2026-07-12): "leaving VRAM budget blank silently
//    binds the SYNTHETIC tier" bit the operator repeatedly — the only prior hint was
//    a footnote nobody reads. These three pieces are the shared basis for a LOUD
//    pre-submit banner in both Studio surfaces (StudioGenerateTab's Clip mode,
//    StudioMovieComposer's Movie mode).
//
// ⚠ THE 0.5 PREDICTION IS GONE (operator ruling 2026-07-27). A blank budget used to
// mean "server substitutes 0.5", which sat under every real model's floor and bound the
// synthetic prover — so this file predicted 0.5 to raise the banner. Blank now means
// AUTOFIT TO THE CARD'S CAPACITY (the reservation engine evicts to free that room), and
// an unmeasurable worker REFUSES rather than falling back to a number. Predicting 0.5
// today would warn on every blank submit, which is the wrong lesson to teach.

// STUDIO_REAL_FLOOR_GB: the cheapest REAL model footprint among the capabilities
// that ALSO have a synthetic last-resort registered (t2v, i2v — see
// studio/models_seed.py: synthetic-t2v/synthetic-i2v are the only synthetic
// ModelConfigs, and the router's real_first score dimension means a real model
// ALWAYS wins when one fits — synthetic binds only when NO real model fits the
// budget). Reuses VACE_BUDGET_GB's number on purpose: wan2.1-vace-1.3b's 6GB INT8
// envelope is the cheapest real footprint in the whole catalog, so 6 is a single
// honest floor to advise regardless of which of the two synthetic-shadowed
// capabilities is active. v2v / id_lock are NOT in this club — models_seed.py
// registers no synthetic model for either capability, so a too-low budget there
// fails the router honestly (NO_CAPABLE_MODEL/VRAM_EXCEEDED) instead of silently
// rendering noise; the honesty banner has nothing to warn about on those shapes.
export const STUDIO_REAL_FLOOR_GB = VACE_BUDGET_GB;

// The banner's one-click fill target. 10 clears STUDIO_REAL_FLOOR_GB with headroom
// for either shadowed capability's cheapest real model (t2v ~5GB, i2v ~8GB via
// ltx-video-0.9.7-dev INT8) without hand-tuning per capability.
export const STUDIO_SAFE_FILL_GB = 10;

/** The budget the ROUTE would effectively see, or `null` when the operator has TYPED
 *  NOTHING USABLE — blank or garbled, both of which the server now resolves by sizing
 *  to the worker's GPU capacity (or refusing outright if it cannot read it).
 *
 *  `null` means "autofit — not predictable from here", NOT "zero". Callers must test it
 *  explicitly: `n < FLOOR` would be TRUE for null (null coerces to 0 in JS), which would
 *  raise the synthetic warning on every blank field — exactly the false alarm this
 *  return type exists to prevent. */
export function effectiveBudgetGb(raw: string): number | null {
  const t = raw.trim();
  if (t === "") return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

/** Would this budget bind the synthetic prover? Only a TYPED number under the real
 *  floor can; a blank field autofits to capacity server-side and is not a warning.
 *
 *  ⚠ `floorGb` (2026-07-29): the caller passes the MEASURED floor for the active
 *  capability — `vram_envelope_gb` off GET /video/render/presets, which is the exact
 *  number the router compares a budget against (`cfg.vram.fits(budget)` is
 *  `gb <= budget`). STUDIO_REAL_FLOOR_GB stays only as the fallback for a page whose
 *  discovery GET has not landed, and it is a KNOWN-LOW fallback: it says 6 while the
 *  cheapest real row in the ratified table is 8.2, so a budget of 7 binds synthetic and
 *  the un-parameterised form of this predicate said nothing. Deriving beats mirroring. */
export function bindsSyntheticTier(raw: string, floorGb: number = STUDIO_REAL_FLOOR_GB): boolean {
  const n = effectiveBudgetGb(raw);
  return n !== null && n < floorGb;
}

// A studio job does NOT pin a model by default — the capability router picks one
// from capability + budget + resolution. This is a SENSIBLE, curated list of the
// studio model ids an operator might PIN (from video_intel/studio/models_seed.py:
// the Wan tiers + the synthetic provers), offered blank-first ("auto"). It is
// advisory: an unroutable pin comes back as errors-as-data on the job, never a
// silent fallback.
//
// ⚠ THIS LIST IS NOW THE FALLBACK / ESCAPE HATCH, NOT THE MENU (2026-07-29). It is a
// hand-maintained mirror of a backend seed file and it is unfiltered: nothing in it
// knows which of these models can serve the capability the operator has chosen, so
// pinning wan2.1-t2v-1.3b under capability=v2v was offered as a first-class choice and
// only failed at render time. The pickers now build their menu from
// `modelsForCapability` over the MEASURED render-preset table and fall back to this list
// only while that discovery GET has not landed — plus keep it behind an explicit
// "show all models" toggle so an operator can still pin off-menu, labelled unverified.
export const STUDIO_MODEL_IDS: readonly string[] = [
  "wan2.1-t2v-1.3b",
  "wan2.1-i2v-14b-720p",
  // wan2.2-{t2v,i2v}-a14b removed 2026-08-13: battery-measured 22.2GiB peak
  // cannot complete on the fleet 24GB card; registry rows dropped too
  // (models_seed.py tombstone). Weights kept on the shared store for a
  // bigger-card future.
  "wan2.1-vace-1.3b",
  "wan2.1-vace-14b",
  "synthetic-i2v",
  "synthetic-t2v",
];

// A preset is a SYNTHETIC preview iff it ships the honesty badge (prompt_note) —
// the seed table sets it ONLY on the no-GPU procedural provers, whose frames are a
// pure function of seed + geometry (the prompt is recorded, not rendered). Real
// tiers leave it empty. This is the discriminator the Generate tab groups by.
export function isSyntheticPreset(p: StudioPreset): boolean {
  return typeof p.prompt_note === "string" && p.prompt_note.trim().length > 0;
}

// ── clip list schemas (tolerant/passthrough — assert only what we read) ──────

const clipOutputSchema = z
  .object({
    asset_id: z.string().nullable().optional(),
    uri: z.string().nullable().optional(),
    mime: z.string().nullable().optional(),
    width: z.number().nullable().optional(),
    height: z.number().nullable().optional(),
    duration_s: z.number().nullable().optional(),
  })
  .passthrough();

// Placement (Active Processes) — WHERE a clip's render physically executes.
// Additive + omit-when-unset on the wire; tolerant/passthrough like the rest.
const placementSchema = z
  .object({
    source: z.string().nullable().optional(),
    host: z.string().nullable().optional(),
    worker_id: z.string().nullable().optional(),
    gpu: z.string().nullable().optional(),
    process: z.string().nullable().optional(),
    reserved_bytes: z.number().nullable().optional(),
  })
  .passthrough();

// STAGE TIMELINE (the exhaustive per-process telemetry). One coarse-stage entry a
// render moved through — deduped server-side (a frame loop is one row with a running
// `count`). The terminal row also carries the outcome (code/message/retryable +
// failed_at_stage). RETAINED through terminal, so a failed row shows its full history.
const stageEntrySchema = z
  .object({
    stage: z.string(),
    ts: z.number().nullable().optional(),
    ts_last: z.number().nullable().optional(),
    count: z.number().nullable().optional(),
    detail: z.string().nullable().optional(),
    // terminal-entry extras
    code: z.string().nullable().optional(),
    message: z.string().nullable().optional(),
    retryable: z.boolean().nullable().optional(),
    failed_at_stage: z.string().nullable().optional(),
  })
  .passthrough();
export type StageEntry = z.infer<typeof stageEntrySchema>;

// Terminal FAILURE summary — the exact "where it's failing": the failing STAGE plus
// the error code/message/retryable. Present only on a failed/cancelled row.
const failureSchema = z
  .object({
    stage: z.string().nullable().optional(),
    code: z.string().nullable().optional(),
    message: z.string().nullable().optional(),
    retryable: z.boolean().nullable().optional(),
  })
  .passthrough();
export type JobFailure = z.infer<typeof failureSchema>;

export const clipSchema = z
  .object({
    job_id: z.string(),
    status: z.string().nullable().optional(),
    playable: z.boolean().nullable().optional(),
    created: z.number().nullable().optional(),
    updated: z.number().nullable().optional(),
    output: clipOutputSchema.nullable().optional(),
    // Live progress blob (carries the awaiting_capacity HOLD marker) + placement —
    // both additive; present only when the server sends them.
    progress: z.record(z.unknown()).nullable().optional(),
    placement: placementSchema.nullable().optional(),
    // Exhaustive per-process telemetry (all additive; retained through terminal):
    // the stage TIMELINE, the terminal FAILURE summary, the honest last-movement ts
    // (stall basis), and the current stage.
    stage_log: z.array(stageEntrySchema).nullable().optional(),
    failure: failureSchema.nullable().optional(),
    last_movement_ts: z.number().nullable().optional(),
    current_stage: z.string().nullable().optional(),
  })
  .passthrough();

export const clipsResponseSchema = z.object({ clips: z.array(clipSchema) }).passthrough();

export type Clip = z.infer<typeof clipSchema>;

/** Build a durable video MediaRef from a studio clip row, or null if it carries no
 *  streamable output (uri + asset_id). Used by the source-clip picker + "use as
 *  source" so both derive the same ref the enqueue body wants. */
export function clipToMediaRef(c: Clip): MediaRef | null {
  const o = c.output;
  if (!o || !o.uri || !o.asset_id) return null;
  return {
    asset_id: o.asset_id,
    kind: "video",
    uri: o.uri,
    mime: o.mime ?? "video/mp4",
    width: o.width ?? null,
    height: o.height ?? null,
    duration_s: o.duration_s ?? null,
  };
}

// GET /video/studio/clip/<id>/detail — the exact creation params for a row expander.
// `source` (new): "manifest" = the resolved params from the content-addressed
// manifest beside a produced clip; "job_record" = a failed/cancelled/running job
// wrote no manifest, so `spec` is the REQUESTED params and `error` the failure.
export const clipDetailSchema = z
  .object({
    job_id: z.string(),
    status: z.string().nullable().optional(),
    source: z.string().nullable().optional(),
    spec: z.record(z.unknown()).nullable().optional(),
    manifest: z
      .object({
        content_hash: z.string().nullable().optional(),
        model_id: z.string().nullable().optional(),
        precision: z.string().nullable().optional(),
        resolution: z
          .object({ width: z.number(), height: z.number(), fps: z.number() })
          .nullable()
          .optional(),
        duration_s: z.number().nullable().optional(),
        frames: z.number().nullable().optional(),
        seed: z.number().nullable().optional(),
        sampler: z.record(z.unknown()).nullable().optional(),
        prompt: z.string().nullable().optional(),
        negative_prompt: z.string().nullable().optional(),
        source_video: z.string().nullable().optional(),
      })
      .passthrough()
      .nullable()
      .optional(),
    error: z
      .object({
        code: z.string().nullable().optional(),
        message: z.string().nullable().optional(),
        retryable: z.boolean().nullable().optional(),
        // the failing STAGE (from the retained timeline) — where it broke.
        stage: z.string().nullable().optional(),
      })
      .passthrough()
      .nullable()
      .optional(),
    // Retained stage TIMELINE + current stage + last-movement ts (additive) so a
    // terminal Library row is inspectable with exactly what it did + where it stopped.
    stage_log: z.array(stageEntrySchema).nullable().optional(),
    current_stage: z.string().nullable().optional(),
    last_movement_ts: z.number().nullable().optional(),
    created: z.number().nullable().optional(),
    updated: z.number().nullable().optional(),
  })
  .passthrough();
export type ClipDetail = z.infer<typeof clipDetailSchema>;

// ── display helpers ──────────────────────────────────────────────────────────

export function shortId(id: string): string {
  return id.length > 10 ? `${id.slice(0, 8)}…` : id;
}

export function dims(c: Clip): string {
  const o = c.output;
  if (!o || o.width == null || o.height == null) return "";
  const d = o.duration_s != null ? ` · ${o.duration_s.toFixed(1)}s` : "";
  return `${o.width}×${o.height}${d}`;
}

export function whenText(c: Clip): string {
  const t = c.updated ?? c.created;
  if (t == null) return "";
  try {
    return new Date(t * 1000).toLocaleTimeString();
  } catch {
    return "";
  }
}

/** Absolute local date+time for a bus epoch-seconds timestamp (detail expander). */
export function fmtWhen(t: number | null | undefined): string {
  if (t == null) return "";
  try {
    return new Date(t * 1000).toLocaleString();
  } catch {
    return "";
  }
}

// ── the STAGE TIMELINE view (the exhaustive per-process expandable) ──────────

const TIMELINE_TERMINAL = new Set(["done", "failed", "cancelled"]);
// A render is "stalled" when the current stage has not moved for this long. Honest,
// tied to the CURRENT stage (server-computed last_movement_ts), not a bare row clock.
export const PROC_STALL_MS = 90_000;

/** "12s" / "3m 4s" / "1h 2m" since an epoch-seconds instant. */
export function agoText(tsSec: number | null | undefined, nowMs: number): string {
  if (tsSec == null) return "";
  const secs = Math.max(0, Math.floor(nowMs / 1000 - tsSec));
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  if (m < 60) return `${m}m ${secs % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

/** Is a job stalled? current stage silent past the window (never for a terminal job). */
export function isStalled(
  lastMovementTs: number | null | undefined,
  terminal: boolean,
  nowMs: number,
): boolean {
  if (terminal || lastMovementTs == null) return false;
  return nowMs / 1000 - lastMovementTs > PROC_STALL_MS / 1000;
}

/**
 * The exhaustive per-process detail: the STAGE TIMELINE (each stage with its time +
 * detail; the current stage highlighted, done stages checked), the CURRENT ACTIVITY
 * line, an honest STALL indicator tied to the current stage, and — the whole point —
 * on failure the exact failing STAGE + error code + message, prominently.
 *
 * No fabricated bars: this renders only what the server sent. `nowMs` is threaded in
 * (a ~1s tick from the parent) so the stall clock advances live without a poll.
 */
export function ProcessTimeline({
  stageLog,
  failure,
  currentStage,
  lastMovementTs,
  terminal = false,
  nowMs,
}: {
  stageLog: StageEntry[] | null | undefined;
  failure?: JobFailure | null;
  currentStage?: string | null;
  lastMovementTs?: number | null;
  terminal?: boolean;
  nowMs: number;
}) {
  const log = stageLog ?? [];
  const stalled = isStalled(lastMovementTs, terminal, nowMs);

  // The current stage's freshest detail = the "what it's doing right now" line.
  const currentEntry =
    !terminal && currentStage
      ? [...log].reverse().find((e) => e.stage === currentStage)
      : undefined;
  const currentDetail = currentEntry?.detail;

  if (log.length === 0 && !failure) {
    return (
      <p className="vi-timeline-empty">
        No stage history recorded yet — it appears here as the render moves through
        its stages.
      </p>
    );
  }

  return (
    <div className="vi-timeline">
      {/* FAILURE — the exact where-it-broke, first + loud. */}
      {failure && failure.code && (
        <div className="vi-timeline-failure" role="alert">
          <strong>
            Failed{failure.stage ? ` at ${failure.stage}` : ""} [{failure.code}]
          </strong>
          {failure.retryable ? <span className="vi-timeline-retry"> · retryable</span> : null}
          {failure.message ? <div className="vi-timeline-failmsg">{failure.message}</div> : null}
        </div>
      )}

      {/* CURRENT ACTIVITY — the live "frame 12/48" / "awaiting capacity …" line. */}
      {!terminal && (currentDetail || currentStage) && (
        <div className={`vi-timeline-current${stalled ? " vi-timeline-current-stalled" : ""}`}>
          <span className="vi-timeline-current-label">now</span>
          <span className="vi-timeline-current-text">
            {currentDetail || currentStage}
          </span>
          {stalled && lastMovementTs != null ? (
            <span className="vi-timeline-stall" title="No stage movement in a while">
              ⚠ no movement {agoText(lastMovementTs, nowMs)}
            </span>
          ) : lastMovementTs != null ? (
            <span className="vi-timeline-move">· {agoText(lastMovementTs, nowMs)} since last movement</span>
          ) : null}
        </div>
      )}

      {/* THE TIMELINE — every stage the render moved through, in order. */}
      <ol className="vi-timeline-list">
        {log.map((e, i) => {
          const isTerm = TIMELINE_TERMINAL.has(e.stage);
          const isCurrent = !terminal && !isTerm && e.stage === currentStage && i === log.length - 1;
          const failed = e.stage === "failed" || e.stage === "cancelled";
          const marker = failed ? "✕" : isTerm ? "✓" : isCurrent ? "●" : "✓";
          const cls = failed
            ? "vi-timeline-mark-fail"
            : isCurrent
            ? `vi-timeline-mark-current${stalled ? " vi-timeline-mark-stalled" : ""}`
            : "vi-timeline-mark-done";
          return (
            <li key={`${e.stage}-${i}`} className="vi-timeline-item">
              <span className={`vi-timeline-mark ${cls}`} aria-hidden>
                {marker}
              </span>
              <span className="vi-timeline-body">
                <span className="vi-timeline-stage">
                  {e.stage.replace(/_/g, " ")}
                  {e.count && e.count > 1 ? (
                    <span className="vi-timeline-count"> ×{e.count}</span>
                  ) : null}
                </span>
                {e.detail ? <span className="vi-timeline-detail">{e.detail}</span> : null}
              </span>
              {e.ts != null ? (
                <span className="vi-timeline-time" title={fmtWhen(e.ts_last ?? e.ts)}>
                  {agoText(e.ts_last ?? e.ts, nowMs)}
                </span>
              ) : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

// ── shared picker widgets ────────────────────────────────────────────────────

// A small library-thumbnail grid; `onPick` receives the chosen ref. Extracted
// MINIMALLY (operator ask 2026-07-12) out of the Clip surface's original inline
// `libraryGrid` closure — same markup, same empty-state copy, same keyed-intrinsic-
// <button> constraint (the repo's JSX typing rejects a `key` on a custom component,
// so each row stays a plain <button>) — so every studio image-library picker (the
// Clip surface's start/reference/control slots AND the Movie composer's start-image
// switcher) shares ONE implementation instead of three copies. `images` is the
// CALLER's already-kind-filtered list (each surface knows whether it wants image vs
// video items from its own useMediaLibrary() read).
export function LibraryImageGrid({
  images,
  onPick,
}: {
  images: LibraryItem[];
  onPick: (ref: MediaRef) => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "0.4rem",
        marginTop: "0.5rem",
        maxHeight: "12rem",
        overflowY: "auto",
      }}
    >
      {images.length === 0 ? (
        <p className="vi-comfy-hint">
          No images in the session library yet — upload one, or produce frames in the Frames tab.
        </p>
      ) : (
        images.map((it) => (
          <button
            key={it.ref.uri}
            type="button"
            className="vi-btn vi-btn-ghost"
            onClick={() => onPick(it.ref)}
            title={it.label ?? it.ref.asset_id}
            style={{ padding: 2, border: "none" }}
          >
            <img
              src={mediaBytesUrl(it.ref.uri)}
              alt={it.label ?? "library image"}
              style={{
                width: "4.5rem",
                height: "4.5rem",
                objectFit: "cover",
                borderRadius: "0.35rem",
                background: "#000",
              }}
            />
          </button>
        ))
      )}
    </div>
  );
}

// ── the creation-parameters expander (carried from the original station) ─────

// One label:value row. Skips null/empty so a compact panel only shows params that
// actually exist.
export function DetailRow({ label, value }: { label: string; value: unknown }) {
  if (value == null || value === "") return null;
  return (
    <div style={{ display: "flex", gap: "0.5rem", lineHeight: 1.5 }}>
      <span style={{ opacity: 0.7, minWidth: "7rem" }}>{label}</span>
      <span style={{ fontVariantNumeric: "tabular-nums", wordBreak: "break-word" }}>
        {String(value)}
      </span>
    </div>
  );
}

// Collapsible creation-parameters panel for one clip row. Shows the TRUE rendered
// params from the manifest (done) and, for a job with NO manifest (failed /
// cancelled / still running), the error prominently + the REQUESTED spec — clearly
// labeled as requested (not resolved), because that render never bound a model.
export function ClipDetailsPanel({
  detail,
  loading,
  error,
}: {
  detail: ClipDetail | undefined;
  loading: boolean;
  error: string | undefined;
}) {
  if (loading && !detail) return <p className="vi-comfy-hint">Loading details…</p>;
  if (error)
    return (
      <p className="vi-error" role="alert">
        {error}
      </p>
    );
  if (!detail) return null;
  const m = detail.manifest;
  const s = (detail.spec ?? {}) as Record<string, unknown>;
  const sampler = (m?.sampler ?? {}) as Record<string, unknown>;
  const res = m?.resolution;
  // "job_record" (or simply no manifest) = the params below are REQUESTED, not
  // resolved. Label them so an operator never reads a failed row's asked-for
  // numbers as what actually ran.
  const requested = !m || detail.source === "job_record";
  return (
    <div
      style={{
        marginTop: "0.4rem",
        padding: "0.5rem 0.6rem",
        borderRadius: "0.4rem",
        background: "rgba(127,127,127,0.08)",
        fontSize: "0.85rem",
      }}
    >
      {detail.error && (
        <div className="vi-error" role="alert" style={{ marginBottom: "0.4rem" }}>
          <strong>
            {detail.status === "cancelled" ? "Cancelled" : "Failed"} [
            {detail.error.code ?? "error"}]
          </strong>
          {detail.error.retryable ? " · retryable" : ""}
          {detail.error.message ? <div>{detail.error.message}</div> : null}
        </div>
      )}
      {m ? (
        <>
          <DetailRow label="model" value={m.model_id} />
          <DetailRow label="precision" value={m.precision} />
          <DetailRow
            label="resolution"
            value={res ? `${res.width}×${res.height} @ ${res.fps}fps` : null}
          />
          <DetailRow label="frames" value={m.frames} />
          <DetailRow label="duration" value={m.duration_s != null ? `${m.duration_s}s` : null} />
          <DetailRow label="seed" value={m.seed} />
          <DetailRow
            label="sampler"
            value={
              sampler.steps != null
                ? `${sampler.steps} steps · cfg ${sampler.cfg}${
                    sampler.shift != null ? ` · shift ${sampler.shift}` : ""
                  }`
                : null
            }
          />
          <DetailRow label="prompt" value={m.prompt} />
          <DetailRow label="negative" value={m.negative_prompt} />
          <DetailRow label="source" value={m.source_video} />
          <DetailRow label="hash" value={m.content_hash} />
        </>
      ) : (
        <>
          {/* No manifest: show the REQUESTED spec so failures are legible, plainly
              labeled as requested (the render bound no model). */}
          <p
            className="vi-comfy-hint"
            role="note"
            style={{ margin: "0 0 0.3rem", fontStyle: "italic", opacity: 0.85 }}
          >
            Requested parameters (this render did not complete — values were never
            resolved; blank steps/cfg = model default).
          </p>
          <DetailRow label="capability" value={s.capability} />
          <DetailRow
            label="resolution"
            value={s.width ? `${s.width}×${s.height} @ ${s.fps}fps` : null}
          />
          <DetailRow label="budget" value={s.vram_budget_gb != null ? `${s.vram_budget_gb} GB` : null} />
          <DetailRow label="seed" value={s.seed} />
          <DetailRow label="steps (req)" value={s.steps} />
          <DetailRow label="cfg (req)" value={s.cfg} />
          <DetailRow label="model (pin)" value={s.model_id} />
          <DetailRow label="prompt" value={s.prompt} />
          <DetailRow label="negative" value={s.negative} />
          <DetailRow label="start image" value={s.start_image} />
          <DetailRow label="source" value={s.source_video} />
        </>
      )}
      {/* Timestamps ride along in every case (from the bus row). */}
      <DetailRow label="created" value={fmtWhen(detail.created)} />
      <DetailRow label="updated" value={fmtWhen(detail.updated)} />
      {requested && !m && (
        <DetailRow label="source" value="job record (requested params)" />
      )}

      {/* STAGE TIMELINE — the retained per-stage history (what this render did + where
          it stopped). Present for any row the bus recorded a timeline for; a terminal
          failed/cancelled row shows the exact failing stage via the failure block. */}
      {detail.stage_log && detail.stage_log.length > 0 && (
        <div style={{ marginTop: "0.5rem" }}>
          <ProcessTimeline
            stageLog={detail.stage_log}
            failure={
              detail.error && detail.error.code
                ? {
                    stage: detail.error.stage ?? null,
                    code: detail.error.code ?? null,
                    message: detail.error.message ?? null,
                    retryable: detail.error.retryable ?? null,
                  }
                : null
            }
            currentStage={detail.current_stage}
            lastMovementTs={detail.last_movement_ts}
            terminal
            nowMs={Date.now()}
          />
        </div>
      )}
    </div>
  );
}

// ── the durable clip catalog: list + light poll + Session-Library push ───────

export interface StudioClipsState {
  clips: Clip[];
  loading: boolean;
  error: string | null;
  /** Force an immediate (foreground) reload of the clip list. */
  reload: () => void;
  /** Force a quiet (background) reload — used right after an enqueue/cancel. */
  refresh: () => void;
}

/**
 * Load the durable studio clip catalog (GET /video/studio/clips) with a light 6s
 * background poll so a freshly-produced clip appears without a manual refresh, and
 * push every playable clip into the Session Library (idempotent by uri) so the
 * studio's output is visible cross-station and itself "Send to Studio"-able. This
 * is the original station's `load()` extracted verbatim so the shell owns it once.
 */
export function useStudioClips(): StudioClipsState {
  const [clips, setClips] = useState<Clip[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const mounted = useRef(true);
  const inFlight = useRef(false);
  // Job ids whose per-clip detail we've already fetched for prompt/model back-fill —
  // so the back-fill runs ONCE per clip and never rides the 6s list poll.
  const detailFetched = useRef<Set<string>>(new Set());

  // Fill the REAL prompt (+ model) into the Session-Library item for a produced clip.
  // The clips-LIST DTO (GET /video/studio/clips) carries no prompt — it lives in the
  // bus row's spec_json and the clip's manifest.json, surfaced only by the per-clip
  // detail route. So each playable clip's detail is fetched ONCE (gated by
  // detailFetched, off the poll) and merged into its library entry, replacing the
  // "(no text prompt)" placeholder the group header showed. Best-effort: an app-error
  // response stays marked (no hammering); a thrown network error un-marks so a later
  // poll can retry.
  const backfillProvenance = useCallback(async (jobId: string, uri: string) => {
    if (detailFetched.current.has(jobId)) return;
    detailFetched.current.add(jobId);
    try {
      const res = await request<unknown>(studioClipDetailUrl(jobId), {
        meta: { specKey: "studio", operation: "studio.clip.detail" },
      });
      if (!mounted.current) return;
      if (!res.ok) return;
      const parsed = clipDetailSchema.safeParse(okValue(res));
      if (!parsed.success) return;
      const d = parsed.data;
      const rawSpecPrompt = d.spec?.prompt;
      const specPrompt =
        typeof rawSpecPrompt === "string" ? rawSpecPrompt : undefined;
      // Prefer the RESOLVED manifest prompt (what actually rendered); fall back to the
      // REQUESTED spec prompt for a clip whose manifest read didn't yield one.
      const prompt = d.manifest?.prompt ?? specPrompt;
      const model = d.manifest?.model_id ?? undefined;
      if (prompt == null && model == null) return;
      updateLibraryProvenance(uri, {
        ...(prompt != null ? { prompt } : {}),
        ...(model != null ? { model } : {}),
      });
    } catch {
      // Transient/network failure — allow a later poll to retry the back-fill.
      detailFetched.current.delete(jobId);
    }
  }, []);

  const load = useCallback(async (background: boolean) => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      // limit=200 (route max, 2026-08-13): the default 50-row window is
      // newest-first with NO status filter, so a burst of failed test/battery
      // jobs used to crowd every DONE clip out of the library. 200 + the
      // archive sweep keeps the render history actually visible.
      const res = await request<unknown>(`${hugpyConfig.studioClipsUrl}?limit=200`, {
        meta: { specKey: "studio", operation: "studio.clips.list" },
      });
      if (!mounted.current) return;
      if (!res.ok) {
        if (background) return; // a failed poll must not wipe a working list
        setLoading(false);
        setError(describeAppError(errorOf(res)));
        return;
      }
      const parsed = clipsResponseSchema.safeParse(okValue(res));
      if (!parsed.success) {
        if (background) return;
        setLoading(false);
        setError("Malformed studio clips response.");
        return;
      }
      const list = parsed.data.clips;
      setClips(list);
      setLoading(false);
      setError(null);
      // Produced studio clips join the Session Library (a video ref per clip,
      // grouped by its job id). addToLibrary de-dups by uri, so this idempotent
      // push on every poll never double-adds. Only playable clips carrying a real
      // uri + asset_id qualify.
      for (const c of list) {
        const o = c.output;
        if (!c.playable || !o || !o.uri || !o.asset_id) continue;
        const ref: MediaRef = {
          asset_id: o.asset_id,
          kind: "video",
          uri: o.uri,
          mime: o.mime ?? "video/mp4",
          width: o.width ?? null,
          height: o.height ?? null,
          duration_s: o.duration_s ?? null,
        };
        addToLibrary(ref, "studio", "studio clip", {
          groupId: c.job_id,
          genKind: "studio_i2v",
        });
        // Back-fill the real prompt/model (the list DTO has neither) so the Session
        // Library group shows the true prompt, not "(no text prompt)". Fire-and-forget,
        // deduped + gated inside backfillProvenance so it costs one detail fetch/clip.
        void backfillProvenance(c.job_id, o.uri);
      }
    } finally {
      inFlight.current = false;
    }
  }, [backfillProvenance]);

  useEffect(() => {
    mounted.current = true;
    void load(false);
    const iv = window.setInterval(() => void load(true), 6000);
    return () => {
      mounted.current = false;
      window.clearInterval(iv);
    };
  }, [load]);

  const reload = useCallback(() => void load(false), [load]);
  const refresh = useCallback(() => void load(true), [load]);

  return { clips, loading, error, reload, refresh };
}
