// A live view of every IN-FLIGHT media-bus job (studio / movie / identity /
// generate / crop / extract) WITH PLACEMENT — the console-wide "Active Processes"
// feed. Polls GET /video/jobs (the media CATALOG projection), which unlike the
// unified /llm/jobs feed carries a `placement` object: WHERE each render
// physically executes (ae · cuda:0 · P-studio), and surfaces the reservation
// `awaiting_capacity` HOLD phase + its shortfall reason + overtaken count.
//
// Honest rendering only: `progress` is projected ONLY when the server sends it —
// no fabricated bars. Cadence ~2s; a fetch error keeps the last-good list rather
// than flashing empty. Mounted-ref guarded so a late poll never setState after
// unmount, and the interval is cleared on unmount.
import { useCallback, useEffect, useRef, useState } from "react";
import { hugpyConfig } from "../config";
import { request, okValue } from "../transport/client";
// The stage-timeline + failure shapes are the SAME wire objects the studio tab
// already renders through `ProcessTimeline`; reusing its types (and, in the panel,
// the component itself) keeps one renderer for "what it did / where it broke"
// instead of a second, divergent one for the console-wide view.
import type { StageEntry, JobFailure } from "../stations/studio/studioShared";

const POLL_MS = 2000;
const IN_FLIGHT = new Set(["queued", "claimed", "running", "cancelling"]);

/** The numbers behind the bar (omit-when-unset on the wire). */
export interface ProgressDetail {
  segmentDone: number | null;
  segmentTotal: number | null;
  step: number | null;
  steps: number | null;
  fraction: number | null;
}

/** WHERE a job physically executes (omit-when-unset on the wire). */
export interface Placement {
  source: string | null; // "reservation" | "template"
  host: string | null; // box: "ae" | "central" | ...
  workerId: string | null;
  gpu: string | null; // device: "cuda:0" | null (cpu)
  process: string | null; // "P-studio" | "P-comfy" | "P-identity" | "ffmpeg" | ...
  reservedBytes: number | null;
}

/** The reservation HOLD marker (progress.phase === "awaiting_capacity"). */
export interface HoldInfo {
  reason: Record<string, unknown> | null;
  heldSince: number | null;
  overtaken: number;
}

/** One in-flight media-bus job, projected for the Active Processes panel. */
export interface MediaJob {
  id: string;
  name: string; // bus job kind
  status: string; // queued | claimed | running | cancelling
  createdAt: number | null; // epoch seconds
  principal: string | null;
  /** Effective phase chip: the reservation hold overrides the raw status. */
  phase: string; // queued | awaiting_capacity | running | claimed | cancelling
  hold: HoldInfo | null; // set only when phase === "awaiting_capacity"
  progress: number | null; // 0..1 when the server sends one, else null
  progressDetail: ProgressDetail | null; // the numbers behind the bar
  stage: string | null;
  placement: Placement | null;
  /** The server's stage timeline: a TAIL while live, the whole thing when terminal. */
  stageLog: StageEntry[];
  stageLogTotal: number; // how many entries exist server-side (tail elides the rest)
  /** The failure envelope of a terminal-failed job — rendered verbatim, never "failed". */
  failure: JobFailure | null;
  terminal: boolean; // done | failed | cancelled (only present with the history filter)
  progressedAt: number | null; // last REAL movement (the stall/stale basis)
  // ---- k117 LIFECYCLE CLOCK (video_intel/job_lifecycle.py) ----------------
  // The feed used to render `now - created` as if it were runtime, so a 9.4s
  // render created 946 minutes ago read as "946m · archiving". These are the
  // server's own facts, computed ONCE at the terminal transition and therefore
  // FROZEN: nothing below is a function of the browser's clock. Null on a row
  // from an older server (or a demo fixture) — see video/jobClock.ts for the
  // honest fallback.
  terminalAt: number | null; // when it ended (epoch seconds); null while live
  terminalStage: string | null; // the TRUE outcome: done | failed | cancelled
  atStage: string | null; // the live stage it was in when it ended ("archiving")
  startedAt: number | null; // when the run actually began
  queueWaitS: number | null; // time spent waiting, kept SEPARATE from runtime
  runS: number | null; // frozen runtime when terminal; time-so-far while running
  totalS: number | null; // queue + run, terminal only
  elapsedInStageS: number | null; // time in the CURRENT stage (live rows only)
  lastProgressAt: number | null; // last reported progress (epoch seconds)
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.trim() ? v : null;
}
function num(v: unknown): number | null {
  return typeof v === "number" && isFinite(v) ? v : null;
}

function parsePlacement(raw: unknown): Placement | null {
  if (!raw || typeof raw !== "object") return null;
  const p = raw as Record<string, unknown>;
  return {
    source: str(p.source),
    host: str(p.host),
    workerId: str(p.worker_id),
    gpu: str(p.gpu),
    process: str(p.process),
    reservedBytes: num(p.reserved_bytes),
  };
}

/** The timeline entries, kept in their WIRE shape for `ProcessTimeline`. */
function parseStageLog(raw: unknown): StageEntry[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (e): e is StageEntry =>
      !!e && typeof e === "object" && typeof (e as StageEntry).stage === "string",
  );
}

/**
 * The failure envelope, passed through UNTOUCHED. Whatever code/message the
 * backend produced (e.g. `deps_missing` with its pip remedy) is what the panel
 * renders — no summarising, no generic "failed".
 */
function parseFailure(raw: unknown): JobFailure | null {
  if (!raw || typeof raw !== "object") return null;
  const f = raw as JobFailure;
  return f.code || f.message ? f : null;
}

function parseProgressDetail(raw: unknown): ProgressDetail | null {
  if (!raw || typeof raw !== "object") return null;
  const d = raw as Record<string, unknown>;
  return {
    segmentDone: num(d.segment_done),
    segmentTotal: num(d.segment_total),
    step: num(d.step),
    steps: num(d.steps),
    fraction: num(d.fraction),
  };
}

function parseJobs(raw: unknown, includeTerminal: boolean): MediaJob[] {
  const jobs = (raw as { jobs?: unknown } | null)?.jobs;
  if (!Array.isArray(jobs)) return [];
  const out: MediaJob[] = [];
  for (const j of jobs) {
    if (!j || typeof j !== "object") continue;
    const row = j as Record<string, unknown>;
    const id = str(row.job_id);
    if (!id) continue;
    const status = str(row.status) ?? "";
    const terminal = !IN_FLIGHT.has(status);
    if (terminal && !includeTerminal) continue;
    const progress = (row.progress as Record<string, unknown> | null) ?? null;
    const phaseRaw =
      progress && typeof progress === "object" ? str(progress.phase) : null;
    const isHold = phaseRaw === "awaiting_capacity";
    let hold: HoldInfo | null = null;
    if (isHold && progress) {
      hold = {
        reason:
          progress.reason && typeof progress.reason === "object"
            ? (progress.reason as Record<string, unknown>)
            : null,
        heldSince: num(progress.held_since),
        overtaken: num(progress.overtaken) ?? 0,
      };
    }
    out.push({
      id,
      name: str(row.name) ?? "",
      status,
      createdAt: num(row.created),
      principal: str(row.principal),
      phase: isHold ? "awaiting_capacity" : status,
      hold,
      // Honest: only surface a numeric progress the server actually sent. The
      // server now computes it (segments done + the within-segment denoise
      // fraction) and sends null when the runner reports nothing measurable, so a
      // missing bar still means "unknown", never "zero".
      progress: num(row.progress_ratio),
      progressDetail: parseProgressDetail(row.progress_detail),
      stage: str(row.current_stage) ?? (progress ? str(progress.stage) : null),
      placement: parsePlacement(row.placement),
      stageLog: parseStageLog(row.stage_log),
      stageLogTotal: num(row.stage_log_total) ?? 0,
      failure: parseFailure(row.failure),
      terminal,
      progressedAt: num(row.progressed_at) ?? num(row.last_movement_ts),
      // k117 — read verbatim; never derived here (deriving is what lied).
      terminalAt: num(row.terminal_at),
      terminalStage: str(row.terminal_stage),
      atStage: str(row.at_stage),
      startedAt: num(row.started_at),
      queueWaitS: num(row.queue_wait_s),
      runS: num(row.run_s),
      totalS: num(row.total_s),
      elapsedInStageS: num(row.elapsed_in_stage_s),
      lastProgressAt: num(row.last_progress_at) ?? num(row.progressed_at),
    });
  }
  return out;
}

/**
 * The live in-flight media jobs + a manual `refetch` (fired right after a cancel
 * so the row disappears without waiting for the next poll tick). Polls only while
 * a consumer is mounted.
 *
 * `includeTerminal` asks the server for recent terminal rows too (?all=1) — the
 * panel's "history" filter, which is how a FAILED render (and its verbatim failure
 * envelope) stays inspectable after it leaves the in-flight set. The default feed
 * is live work only: the server hides in-flight rows abandoned by a dead process,
 * so days-old ghosts no longer crowd the panel.
 */
export function useMediaJobs(
  includeTerminal = false,
): { jobs: MediaJob[]; refetch: () => void } {
  const [jobs, setJobs] = useState<MediaJob[]>([]);
  const mounted = useRef(true);

  const load = useCallback(async () => {
    const url = includeTerminal
      ? `${hugpyConfig.mediaJobsUrl}?all=1`
      : hugpyConfig.mediaJobsUrl;
    const r = await request<unknown>(url, {
      meta: { specKey: "media", operation: "media.jobs.poll" },
    });
    if (!mounted.current) return;
    if (!r.ok) return; // transient — keep last-good, retry next tick
    setJobs(parseJobs(okValue(r), includeTerminal));
  }, [includeTerminal]);

  useEffect(() => {
    mounted.current = true;
    void load();
    const id = window.setInterval(() => void load(), POLL_MS);
    return () => {
      mounted.current = false;
      window.clearInterval(id);
    };
  }, [load]);

  return { jobs, refetch: load };
}

// ── shared render helpers (used by the console-wide panel + the studio tab) ──

/** "ae · cuda:0 · P-studio" — the placement one-liner, or "" when unknown. */
export function placementLine(p: Placement | null): string {
  if (!p) return "";
  const parts = [p.host, p.gpu ?? "cpu", p.process].filter(
    (x): x is string => typeof x === "string" && x.length > 0,
  );
  return parts.join(" · ");
}

/** True when the job runs OFF the central box (an external worker/service). */
export function isExternal(p: Placement | null): boolean {
  return !!p && !!p.host && p.host !== "central";
}

/** Human GiB for a reserved-bytes value, or "" when absent. */
export function reservedGib(p: Placement | null): string {
  if (!p || p.reservedBytes == null) return "";
  return `${(p.reservedBytes / 1024 ** 3).toFixed(1)} GB reserved`;
}

/**
 * "42% · segment 1/2 · step 12/30" — the label under the bar, built only from the
 * numbers the server actually sent. "" when there is nothing honest to say.
 */
export function progressLabel(job: MediaJob): string {
  const parts: string[] = [];
  if (job.progress != null) parts.push(`${Math.round(job.progress * 100)}%`);
  const d = job.progressDetail;
  if (d?.segmentTotal != null && d.segmentDone != null) {
    // 1-based for humans: while segment 0 renders, an operator reads "segment 1/2".
    const at = Math.min(d.segmentDone + 1, d.segmentTotal);
    parts.push(`segment ${at}/${d.segmentTotal}`);
  }
  if (d?.step != null && d.steps != null) parts.push(`step ${d.step}/${d.steps}`);
  return parts.join(" · ");
}

/**
 * The freshest timeline line — "segment 0/2 · t2v · worker rendering". This is the
 * one-line answer to "what is it doing", shown WITHOUT expanding the row, because
 * "no logs are ever shown" was the operator's complaint and a detail that needs a
 * click is still a detail nobody sees.
 */
export function latestEvent(job: MediaJob): string {
  for (let i = job.stageLog.length - 1; i >= 0; i -= 1) {
    const d = job.stageLog[i]?.detail;
    if (typeof d === "string" && d.trim()) return d;
  }
  return "";
}

/** Elapsed since an epoch-seconds start, as "12s" / "3m 4s". */
export function elapsedSince(createdAt: number | null): string {
  if (createdAt == null) return "";
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - createdAt));
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}
