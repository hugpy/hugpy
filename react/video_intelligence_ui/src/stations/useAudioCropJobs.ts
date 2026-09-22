// The Audio Crop station's enqueue surface — the temporal sibling of
// useCropJobs. It still owns the request-body construction (one TEMPORAL crop
// job per queued window, reusing the crop endpoint with spatial: null) and POSTs
// each through request<T>() + config URLs; the moment it has a job_id it hands
// the job to jobTracker.trackJob and stops owning any polling or job state.
// Polling, the poll cap, the terminal parse, and the single library push all
// live in jobTracker now — so a clip enqueued here keeps running (and lands in
// the library, origin "audio-crop") even after the user switches sub-tabs
// (which unmounts this station) or reloads the page.
//
// The public API to AudioCropStation is UNCHANGED: `{ jobs, runCrops, resume,
// reset }` where each job carries { key, index, region, jobId, status, output,
// error, capped, startedAt }. `jobs` is DERIVED from the tracker (this station's
// crop records) merged with any still-enqueuing local placeholders.
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
import type { TemporalRegion } from "../regions/types";

export interface TrackedAudioJob {
  /** Stable local key (the server job id once enqueued). */
  key: string;
  /** 1-based position, for display. */
  index: number;
  /** The temporal window (seconds) this job crops. */
  region: TemporalRegion;
  /** Server job id once enqueued, else null. */
  jobId: string | null;
  /** Latest polled status (null = enqueued but not yet claimed). */
  status: JobStatus;
  /** Result audio-clip MediaRef on success. */
  output: MediaRef | null;
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Poll cap reached while non-terminal — NOT a failure (Phase 8). */
  capped: boolean;
  /** Enqueue timestamp — the anchor for the poll cap. */
  startedAt: number;
}

// A pre-enqueue placeholder — see useCropJobs.ts for the rationale.
interface Pending {
  key: string;
  index: number;
  region: TemporalRegion;
  status: Extract<JobStatus, "queued" | "failed">;
  error: string | null;
  startedAt: number;
}

let _seq = 0;
function makeKey(): string {
  _seq += 1;
  return `aclip_${_seq.toString(36)}_${Date.now().toString(36)}`;
}

export interface AudioCropJobsApi {
  jobs: TrackedAudioJob[];
  /** Enqueue one temporal crop job per region and begin polling (via the tracker). */
  runCrops: (regions: TemporalRegion[]) => void;
  /**
   * Re-anchor the poll clock for a capped job (or every capped clip when `key`
   * is omitted) and resume polling. Clears `capped`, keeps status.
   */
  resume: (key?: string) => void;
  /** Forget this station's finished jobs (called when a new source is ingested). */
  reset: () => void;
}

const STATION = "audio-crop";

export function useAudioCropJobs(source: MediaRef | null): AudioCropJobsApi {
  const records = useJobTracker();
  const [pending, setPending] = useState<Pending[]>([]);

  const mine = useMemo(
    () => records.filter((r) => r.station === STATION && r.kind === "crop"),
    [records],
  );

  const mineRef = useRef<TrackerRecord[]>(mine);
  mineRef.current = mine;
  const pendingRef = useRef<Pending[]>(pending);
  pendingRef.current = pending;

  const jobs = useMemo<TrackedAudioJob[]>(() => {
    const fromTracker: TrackedAudioJob[] = mine.map((r) => ({
      key: r.jobId,
      index: r.index,
      region: (r.region as TemporalRegion) ?? { start_s: 0, end_s: 0 },
      jobId: r.jobId,
      status: r.status,
      output: r.outputs[0] ?? null,
      error: r.error,
      capped: r.capped,
      startedAt: r.startedAt,
    }));
    const fromPending: TrackedAudioJob[] = pending.map((p) => ({
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
    (regions: TemporalRegion[]) => {
      if (!source || regions.length === 0) return;
      const base = Math.max(
        0,
        ...mineRef.current.map((r) => r.index),
        ...pendingRef.current.map((p) => p.index),
      );

      regions.forEach((region, i) => {
        const key = makeKey();
        const index = base + i + 1;
        const enqueuedAt = Date.now();
        const label = `clip ${region.start_s.toFixed(2)}–${region.end_s.toFixed(2)}s`;
        setPending((prev) => [
          ...prev,
          { key, index, region, status: "queued", error: null, startedAt: enqueuedAt },
        ]);

        // Temporal audio crop reuses the crop endpoint — spatial is null.
        const body: CropJobRequest = { source, spatial: null, temporal: region };
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
