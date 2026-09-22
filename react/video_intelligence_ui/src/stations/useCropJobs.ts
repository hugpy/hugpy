// The crop station's enqueue surface, now riding the session-wide job tracker.
// It still owns the request-body construction (one crop job per queued region)
// and POSTs each through request<T>() + config URLs; the moment it has a job_id
// it hands the job to jobTracker.trackJob and stops owning any polling or job
// state. Polling, the poll cap, the terminal parse, and the single library push
// all live in jobTracker now — so a crop enqueued here keeps running (and lands
// in the library) even after the user switches sub-tabs (which unmounts this
// station) or reloads the page.
//
// The public API to ImageCropStation is UNCHANGED: `{ jobs, runCrops, resume,
// reset }` where each job carries { key, index, region, jobId, status, output,
// error, capped, startedAt }. `jobs` is now DERIVED from the tracker (this
// station's crop records) merged with any still-enqueuing local placeholders.
import { useCallback, useMemo, useRef, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import {
  enqueueResultSchema,
  type CropJobRequest,
  type JobStatus,
  type MediaRef,
} from "../video/contract";
import {
  useJobTracker,
  trackJob,
  resumeJob,
  resumeStation,
  dismissTerminal,
  type TrackedJob as TrackerRecord,
} from "../video/jobTracker";
import type { SpatialRegion } from "../regions/types";

export interface TrackedJob {
  /** Stable local key (the server job id once enqueued). */
  key: string;
  /** 1-based position, for display. */
  index: number;
  /** The native-px region this job crops. */
  region: SpatialRegion;
  /** Server job id once enqueued, else null. */
  jobId: string | null;
  /** Latest polled status (null = enqueued but not yet claimed). */
  status: JobStatus;
  /** Result crop MediaRef on success. */
  output: MediaRef | null;
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Poll cap reached while non-terminal — NOT a failure (Phase 8). */
  capped: boolean;
  /** Enqueue timestamp — the anchor for the poll cap. */
  startedAt: number;
}

// A pre-enqueue placeholder: the brief window between the user clicking Run and
// the enqueue POST returning a job_id (or failing). Once a job_id arrives the
// tracker owns the job and the placeholder is dropped; a failed enqueue keeps its
// placeholder (as a failed row) since it never became a tracker record.
interface Pending {
  key: string;
  index: number;
  region: SpatialRegion;
  status: Extract<JobStatus, "queued" | "failed">;
  error: string | null;
  startedAt: number;
}

let _seq = 0;
function makeKey(): string {
  _seq += 1;
  return `job_${_seq.toString(36)}_${Date.now().toString(36)}`;
}

export interface CropJobsApi {
  jobs: TrackedJob[];
  /** Enqueue one crop job per region and begin polling (via the tracker). */
  runCrops: (regions: SpatialRegion[]) => void;
  /**
   * Re-anchor the poll clock for a capped job (or every capped crop when `key`
   * is omitted) and resume polling. Clears `capped`, keeps status.
   */
  resume: (key?: string) => void;
  /** Forget this station's finished jobs (called when a new source is ingested). */
  reset: () => void;
}

const STATION = "image-crop";

export function useCropJobs(source: MediaRef | null): CropJobsApi {
  const records = useJobTracker();
  const [pending, setPending] = useState<Pending[]>([]);

  // This station's crop records, newest-index last (stable via useMemo).
  const mine = useMemo(
    () => records.filter((r) => r.station === STATION && r.kind === "crop"),
    [records],
  );

  const mineRef = useRef<TrackerRecord[]>(mine);
  mineRef.current = mine;
  const pendingRef = useRef<Pending[]>(pending);
  pendingRef.current = pending;

  const jobs = useMemo<TrackedJob[]>(() => {
    const fromTracker: TrackedJob[] = mine.map((r) => ({
      key: r.jobId,
      index: r.index,
      region: (r.region as SpatialRegion) ?? { x: 0, y: 0, w: 0, h: 0 },
      jobId: r.jobId,
      status: r.status,
      output: r.outputs[0] ?? null,
      error: r.error,
      capped: r.capped,
      startedAt: r.startedAt,
    }));
    const fromPending: TrackedJob[] = pending.map((p) => ({
      key: p.key,
      index: p.index,
      region: p.region,
      jobId: null,
      status: p.status,
      output: null,
      error: p.error,
      capped: false,
      startedAt: p.startedAt,
    }));
    return [...fromTracker, ...fromPending].sort((a, b) => a.index - b.index);
  }, [mine, pending]);

  const reset = useCallback(() => {
    setPending([]);
    dismissTerminal(STATION);
  }, []);

  const resume = useCallback((key?: string) => {
    if (key != null) resumeJob(key);
    else resumeStation(STATION);
  }, []);

  const runCrops = useCallback(
    (regions: SpatialRegion[]) => {
      if (!source || regions.length === 0) return;
      // Continue the 1-based numbering past any existing/pending rows.
      const base = Math.max(
        0,
        ...mineRef.current.map((r) => r.index),
        ...pendingRef.current.map((p) => p.index),
      );

      regions.forEach((region, i) => {
        const key = makeKey();
        const index = base + i + 1;
        const enqueuedAt = Date.now();
        const label = `crop ${region.w}×${region.h}`;
        setPending((prev) => [
          ...prev,
          { key, index, region, status: "queued", error: null, startedAt: enqueuedAt },
        ]);

        const body: CropJobRequest = { source, spatial: region, temporal: null };
        void request<unknown>(hugpyConfig.cropEnqueueUrl, {
          method: "POST",
          body: JSON.stringify(body),
          headers: { "Content-Type": "application/json" },
          meta: { specKey: STATION, operation: "crop.enqueue" },
        }).then((r) => {
          if (!r.ok) {
            setPending((prev) =>
              prev.map((p) =>
                p.key === key
                  ? { ...p, status: "failed", error: describeAppError(errorOf(r)) }
                  : p,
              ),
            );
            return;
          }
          const parsed = enqueueResultSchema.safeParse(okValue(r));
          if (!parsed.success) {
            setPending((prev) =>
              prev.map((p) =>
                p.key === key
                  ? { ...p, status: "failed", error: "Malformed enqueue response." }
                  : p,
              ),
            );
            return;
          }
          // Hand the job to the durable tracker, then drop the local placeholder.
          trackJob({
            jobId: parsed.data.job_id,
            kind: "crop",
            station: STATION,
            label,
            libLabel: label,
            enqueuedAt,
            index,
            region,
          });
          setPending((prev) => prev.filter((p) => p.key !== key));
        });
      });
    },
    [source],
  );

  return { jobs, runCrops, resume, reset };
}
