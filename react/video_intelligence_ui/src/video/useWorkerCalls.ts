// A live view of EVERY in-flight worker call across ALL transports (web / v1 /
// discord / cli / media) — the fleet-wide feed the session jobTracker CANNOT see,
// because the tracker only knows jobs THIS browser enqueued. GET /llm/jobs is the
// authoritative unified feed; this hook polls it, keeps the in-flight rows
// (pending / processing / streaming), drops `download` provisioning (shown
// elsewhere, and there can be dozens), and hands the survivors to the shared Active
// Processes panel so an identity reconstruction (kind:"identity_reconstruction",
// transport:"media") — or any render from another station — shows up even though
// this browser never launched it.
//
// Cadence ~2s (the panel doesn't need the tracker's 800ms poll); a fetch error
// keeps the last-good list rather than flashing empty. Mounted-ref guarded so a
// late poll never setState after unmount, and the interval is cleared on unmount.
import { useCallback, useEffect, useRef, useState } from "react";
import { hugpyConfig } from "../config";
import { request, okValue } from "../transport/client";

/** Poll cadence for the unified worker-call feed. */
const POLL_MS = 2000;

/** In-flight statuses from the /llm/jobs feed (everything else is terminal/idle). */
const IN_FLIGHT = new Set(["pending", "processing", "streaming"]);

/** One in-flight worker call, projected from a /llm/jobs row for the panel. */
export interface WorkerCall {
  /** Server job id — the cancel handle + the dedup key against a TrackedJob.jobId. */
  id: string;
  /** Raw job kind (humanized for display in the row). */
  kind: string;
  /** In-flight status: pending | processing | streaming. */
  status: string;
  /** Transport that launched it — decides the cancel endpoint (media vs llm chat). */
  transport: string;
  /** Model label if the feed carries one (model_name preferred over model). */
  model: string | null;
  /** Serving worker name if present. */
  worker: string | null;
  /**
   * ms-epoch start anchor, derived from the server `elapsed` seconds at poll time —
   * the row's live clock reads (now - startedAt), so the elapsed ticks between polls
   * and re-aligns to the server's truth on every refresh.
   */
  startedAt: number;
  /**
   * Fractional progress 0..1 if the feed carries it, else null (unknown → the row
   * shows an indeterminate "working" bar rather than a fabricated percentage).
   */
  progress: number | null;
  /** Free-text status line the worker is currently emitting (if any). */
  message: string | null;
  /** Coarse pipeline stage label (e.g. "sampling", "encoding") if the feed carries it. */
  stage: string | null;
  /** Last ≤40 log lines, NEWEST LAST — the scrolling tail in the expanded panel. */
  logTail: string[];
  /** Server-asserted wedged flag: the render is not making progress. */
  stalled: boolean;
  /** True once a cancel was requested and the server acknowledged it. */
  cancelRequested: boolean;
  /**
   * ms-epoch of the last observed movement (server `updated_at`), or null. The row
   * derives "last movement Ns ago" as (now - updatedAt) and flips to a stalled look
   * when that exceeds the stall window, so a frozen render stops reading as live.
   */
  updatedAt: number | null;
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.trim() ? v : null;
}

/** Clamp an arbitrary progress reading into the renderable 0..1 band. */
function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

// A server timestamp → ms-epoch. Tolerates epoch SECONDS (the arm's usual wire
// form, ~1.7e9), epoch MILLISECONDS (~1.7e12), and ISO strings; anything else → null.
function toMs(v: unknown): number | null {
  if (typeof v === "number" && isFinite(v)) return v < 1e12 ? v * 1000 : v;
  if (typeof v === "string" && v.trim()) {
    const t = Date.parse(v);
    return isFinite(t) ? t : null;
  }
  return null;
}

// Project the /llm/jobs envelope into the in-flight WorkerCall rows. Defensive by
// construction (no zod schema for this feed, and strictNullChecks is off in the
// arm's tsconfig): a malformed row is skipped, never thrown.
function parseJobs(raw: unknown, at: number): WorkerCall[] {
  const jobs = (raw as { jobs?: unknown } | null)?.jobs;
  if (!Array.isArray(jobs)) return [];
  const out: WorkerCall[] = [];
  for (const j of jobs) {
    if (!j || typeof j !== "object") continue;
    const row = j as Record<string, unknown>;
    const id = str(row.id);
    if (!id) continue;
    const kind = str(row.kind) ?? "";
    if (kind === "download") continue; // provisioning — shown elsewhere
    const status = str(row.status) ?? "";
    if (!IN_FLIGHT.has(status)) continue;
    const elapsed =
      typeof row.elapsed === "number" && isFinite(row.elapsed) ? row.elapsed : 0;
    // NEW fields (a parallel backend change adds progress/stage/log_tail; until it
    // ships these read absent) — all optional, projected defensively so a bare
    // legacy row still renders, just without the richer expanded detail.
    const progress =
      typeof row.progress === "number" && isFinite(row.progress)
        ? clamp01(row.progress)
        : null;
    const logTail = Array.isArray(row.log_tail)
      ? row.log_tail.filter((l): l is string => typeof l === "string").slice(-40)
      : [];
    out.push({
      id,
      kind,
      status,
      transport: str(row.transport) ?? "",
      model: str(row.model_name) ?? str(row.model),
      worker: str(row.worker),
      startedAt: at - Math.max(0, elapsed) * 1000,
      progress,
      message: str(row.message),
      stage: str(row.stage),
      logTail,
      stalled: row.stalled === true,
      cancelRequested: row.cancel_requested === true,
      updatedAt: toMs(row.updated_at),
    });
  }
  return out;
}

/**
 * The live in-flight worker calls + a manual `refetch` (fired right after a cancel,
 * so the row disappears without waiting for the next poll tick). Polls only while a
 * consumer is mounted — the panel mounts it only on the Active Processes tab, so no
 * background polling when that tab is closed.
 */
export function useWorkerCalls(): { calls: WorkerCall[]; refetch: () => void } {
  const [calls, setCalls] = useState<WorkerCall[]>([]);
  const mounted = useRef(true);

  const load = useCallback(async () => {
    const r = await request<unknown>(hugpyConfig.llmJobsUrl, {
      meta: { specKey: "workers", operation: "worker.calls.poll" },
    });
    if (!mounted.current) return;
    if (!r.ok) return; // transient — keep last-good, retry next tick
    setCalls(parseJobs(okValue(r), Date.now()));
  }, []);

  useEffect(() => {
    mounted.current = true;
    void load();
    const id = window.setInterval(() => void load(), POLL_MS);
    return () => {
      mounted.current = false;
      window.clearInterval(id);
    };
  }, [load]);

  return { calls, refetch: load };
}
