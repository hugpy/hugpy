// The Generate station's enqueue surface, now riding the session-wide job
// tracker. ONE ordered multimodal prompt → ONE generate job → ONE image. This
// hook still owns the request-body construction and POSTs it through request<T>()
// + config URLs; the moment it has a job_id it hands the job to
// jobTracker.trackJob and stops owning any polling or job state. Polling, the
// generate poll cap (GENERATE_POLL_CAP_MS), the terminal parse, and the single
// library push (origin "generate") all live in jobTracker now — so a generation
// kicked off here keeps running (and lands in the library) even after the user
// switches sub-tabs (which unmounts this station) or reloads the page.
//
// The public API to GenerateStation is UNCHANGED:
// `{ jobId, status, output, error, capped, running, run, resume, reset }`.
import { useCallback, useMemo, useRef, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import {
  enqueueResultSchema,
  promptSummary,
  isMovieProgress,
  type GenerateImageRequest,
  type JobStatus,
  type MediaRef,
  type JobProgress,
  type JobProject,
} from "../video/contract";
import {
  useJobTracker,
  trackJob,
  resumeJob,
  dismissTerminal,
  isJobActive,
} from "../video/jobTracker";

export interface GenerateJobApi {
  /** Server job id once enqueued, else null. */
  jobId: string | null;
  /** Latest polled status (null = idle or enqueued-but-not-yet-claimed). */
  status: JobStatus;
  /** The generated image MediaRef (populated on done). */
  output: MediaRef | null;
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Poll cap reached while non-terminal (Phase 8) — NOT a failure. */
  capped: boolean;
  /** True while enqueuing or polling a non-terminal job. */
  running: boolean;
  /** Live per-run progress while running (done/total, stage, step); null otherwise. */
  progress: JobProgress | null;
  /** Auto-archive project this run was saved into (populated on done); null otherwise. */
  project: JobProject | null;
  /** Enqueue ONE generate job and begin polling. Replaces any prior run. */
  run: (req: GenerateImageRequest) => void;
  /** Re-anchor the poll clock and resume polling a capped job. Clears `capped`. */
  resume: () => void;
  /** Forget this station's finished job (called when the composer is cleared). */
  reset: () => void;
}

const STATION = "generate";

export function useGenerateJob(): GenerateJobApi {
  const records = useJobTracker();
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [enqueuing, setEnqueuing] = useState<boolean>(false);
  const [enqueueError, setEnqueueError] = useState<string | null>(null);

  const record = useMemo(() => {
    const mine = records.filter(
      (r) => r.station === STATION && r.kind === "generate_image",
    );
    if (currentJobId) return mine.find((r) => r.jobId === currentJobId) ?? null;
    if (enqueuing) return null;
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

  const run = useCallback((req: GenerateImageRequest) => {
    setCurrentJobId(null);
    setEnqueueError(null);
    setEnqueuing(true);
    const enqueuedAt = Date.now();

    void request<unknown>(hugpyConfig.generateEnqueueUrl, {
      method: "POST",
      body: JSON.stringify(req),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: STATION, operation: "generate_image.enqueue" },
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
        kind: "generate_image",
        station: STATION,
        label: "generate image",
        enqueuedAt,
        prompt: promptSummary(req.parts),
        model: req.model_id,
      });
      setCurrentJobId(jobId);
    });
  }, []);

  const status: JobStatus = enqueueError ? "failed" : record?.status ?? null;
  const error =
    enqueueError ??
    record?.error ??
    (record?.expired ? "This job is no longer tracked by the server." : null);

  // A generate_image record only ever carries flat progress; narrow off the tracker's
  // union (never a nested movie blob for this kind).
  const progress: JobProgress | null =
    record?.progress && !isMovieProgress(record.progress) ? record.progress : null;

  return {
    jobId: record?.jobId ?? null,
    status,
    output: record?.outputs[0] ?? null,
    error,
    capped: record?.capped ?? false,
    running: enqueuing || (record != null && isJobActive(record)),
    progress,
    project: record?.project ?? null,
    run,
    resume,
    reset,
  };
}
