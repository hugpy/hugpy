// The Generate station's SCENE enqueue surface — Scene mode's sibling to
// useGenerateJob. ONE ordered multimodal prompt → ONE generate_scene job → N
// consecutive image frames + (optionally) one assembled mp4 clip. Cloned from
// useGenerateJob: this hook owns request-body construction and POSTs it through
// request<T>() + config URLs; the moment it has a job_id it hands the job to
// jobTracker.trackJob and stops owning any polling or job state. Polling, the
// n_frames-scaled scene poll cap (generateScenePollCapMs, plumbed via
// trackJob.nFrames → capFor), the terminal parse, and the centralized library
// push (frames + the clip, origin "generate") all live in jobTracker — so a scene
// kicked off here keeps running (and lands in the library) even after the user
// switches sub-tabs (which unmounts this station) or reloads the page.
//
// The ONLY shape difference from useGenerateJob is the output: Scene mode reads
// the FULL result.outputs array (N image refs, then the clip last), so this hook
// exposes `outputs: MediaRef[]` instead of a single `output`.
import { useCallback, useMemo, useRef, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import {
  enqueueResultSchema,
  promptSummary,
  isMovieProgress,
  type GenerateSceneRequest,
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

export interface GenerateSceneJobApi {
  /** Server job id once enqueued, else null. */
  jobId: string | null;
  /** Latest polled status (null = idle or enqueued-but-not-yet-claimed). */
  status: JobStatus;
  /** ALL produced MediaRefs (N image frames, then the clip last if assembled). */
  outputs: MediaRef[];
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Poll cap reached while non-terminal — NOT a failure. */
  capped: boolean;
  /** True while enqueuing or polling a non-terminal job. */
  running: boolean;
  /** Live per-run progress while running (done/total, stage, step, frames-so-far); null otherwise. */
  progress: JobProgress | null;
  /** Auto-archive project this scene was saved into (populated on done); null otherwise. */
  project: JobProject | null;
  /** Enqueue ONE generate_scene job and begin polling. Replaces any prior run. */
  run: (req: GenerateSceneRequest) => void;
  /** Re-anchor the poll clock and resume polling a capped job. Clears `capped`. */
  resume: () => void;
  /** Forget this station's finished job (called when the composer is cleared). */
  reset: () => void;
}

const STATION = "generate";

export function useGenerateSceneJob(): GenerateSceneJobApi {
  const records = useJobTracker();
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [enqueuing, setEnqueuing] = useState<boolean>(false);
  const [enqueueError, setEnqueueError] = useState<string | null>(null);

  const record = useMemo(() => {
    const mine = records.filter(
      (r) => r.station === STATION && r.kind === "generate_scene",
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

  const run = useCallback((req: GenerateSceneRequest) => {
    setCurrentJobId(null);
    setEnqueueError(null);
    setEnqueuing(true);
    const enqueuedAt = Date.now();

    void request<unknown>(hugpyConfig.generateSceneEnqueueUrl, {
      method: "POST",
      body: JSON.stringify(req),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: STATION, operation: "generate_scene.enqueue" },
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
        kind: "generate_scene",
        station: STATION,
        label: "generate scene",
        enqueuedAt,
        nFrames: req.n_frames,
        chain: req.chain,
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

  // A generate_scene record only ever carries flat progress; narrow off the tracker's
  // union (never a nested movie blob for this kind).
  const progress: JobProgress | null =
    record?.progress && !isMovieProgress(record.progress) ? record.progress : null;

  return {
    jobId: record?.jobId ?? null,
    status,
    outputs: record?.outputs ?? [],
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
