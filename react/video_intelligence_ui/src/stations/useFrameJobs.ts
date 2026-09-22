// The Frames station's enqueue surface, now riding the session-wide job tracker.
// frame-extract is ONE job that produces MANY frames. This hook still owns the
// request-body construction and POSTs it through request<T>() + config URLs; the
// moment it has a job_id it hands the job to jobTracker.trackJob and stops owning
// any polling or job state. Polling, the poll cap, the terminal parse, and the
// single "push every produced frame into the library" step all live in
// jobTracker now — so an extraction kicked off here keeps running (and its frames
// land in the shared library) even after the user switches sub-tabs (which
// unmounts this station) or reloads the page.
//
// The public API to FrameExtractStation is UNCHANGED:
// `{ jobId, status, frames, error, capped, running, run, resume, reset }`.
// Those values are DERIVED from the tracker (this station's newest frame-extract
// record) plus a little local state for the brief pre-enqueue window.
import { useCallback, useMemo, useRef, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import {
  enqueueResultSchema,
  type FrameExtractRequest,
  type JobStatus,
  type MediaRef,
} from "../video/contract";
import {
  useJobTracker,
  trackJob,
  resumeJob,
  dismissTerminal,
  isJobActive,
} from "../video/jobTracker";

export interface FrameJobsApi {
  /** Server job id once enqueued, else null. */
  jobId: string | null;
  /** Latest polled status (null = idle or enqueued-but-not-yet-claimed). */
  status: JobStatus;
  /** The produced frame MediaRefs (populated on done). */
  frames: MediaRef[];
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Poll cap reached while non-terminal (Phase 8) — NOT a failure. */
  capped: boolean;
  /** True while enqueuing or polling a non-terminal job. */
  running: boolean;
  /** Enqueue ONE frame-extract job and begin polling. Replaces any prior run. */
  run: (req: FrameExtractRequest) => void;
  /** Re-anchor the poll clock and resume polling a capped job. Clears `capped`. */
  resume: () => void;
  /** Forget this station's finished job (called when a new source is ingested). */
  reset: () => void;
}

const STATION = "frames";

export function useFrameJobs(): FrameJobsApi {
  const records = useJobTracker();
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [enqueuing, setEnqueuing] = useState<boolean>(false);
  const [enqueueError, setEnqueueError] = useState<string | null>(null);

  // The record this hook instance is watching: the one it just enqueued
  // (currentJobId), or — on a fresh mount after a tab switch / reload, when the
  // local pointer is gone — the newest frame-extract record for this station.
  const record = useMemo(() => {
    const mine = records.filter(
      (r) => r.station === STATION && r.kind === "frame_extract",
    );
    if (currentJobId) return mine.find((r) => r.jobId === currentJobId) ?? null;
    if (enqueuing) return null; // a fresh run is in flight — don't resurface the old one
    return mine.slice().sort((a, b) => b.enqueuedAt - a.enqueuedAt)[0] ?? null;
  }, [records, currentJobId, enqueuing]);

  const jobIdRef = useRef<string | null>(record?.jobId ?? null);
  jobIdRef.current = record?.jobId ?? null;

  const reset = useCallback(() => {
    setCurrentJobId(null);
    setEnqueuing(false);
    setEnqueueError(null);
    dismissTerminal(STATION);
  }, []);

  const resume = useCallback(() => {
    const id = jobIdRef.current;
    if (id) resumeJob(id);
  }, []);

  const run = useCallback((req: FrameExtractRequest) => {
    // Fresh run — hide any prior result and enqueue one job.
    setCurrentJobId(null);
    setEnqueueError(null);
    setEnqueuing(true);
    const enqueuedAt = Date.now();

    void request<unknown>(hugpyConfig.frameExtractEnqueueUrl, {
      method: "POST",
      body: JSON.stringify(req),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: STATION, operation: "frame_extract.enqueue" },
    }).then((r) => {
      setEnqueuing(false);
      if (!r.ok) {
        setEnqueueError(describeAppError(errorOf(r)));
        return;
      }
      const parsed = enqueueResultSchema.safeParse(okValue(r));
      if (!parsed.success) {
        setEnqueueError("Malformed enqueue response.");
        return;
      }
      const jobId = parsed.data.job_id;
      trackJob({
        jobId,
        kind: "frame_extract",
        station: STATION,
        label: `frames @ ${req.fps} fps`,
        enqueuedAt,
      });
      setCurrentJobId(jobId);
    });
  }, []);

  const status: JobStatus = enqueueError ? "failed" : record?.status ?? null;
  const error =
    enqueueError ??
    record?.error ??
    (record?.expired ? "This job is no longer tracked by the server." : null);

  return {
    jobId: record?.jobId ?? null,
    status,
    frames: record?.outputs ?? [],
    error,
    capped: record?.capped ?? false,
    running: enqueuing || (record != null && isJobActive(record)),
    run,
    resume,
    reset,
  };
}
