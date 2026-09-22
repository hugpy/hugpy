// The Generate station's MOVIE enqueue surface — Movie mode's sibling to
// useGenerateSceneJob. ONE goal timeline → ONE generate_movie job → N segments of
// consecutive frames + one assembled mp4 clip (LAST), with an opt-in vision
// director loop. Cloned from useGenerateSceneJob: this hook owns request-body
// construction and POSTs it through request<T>() + config URLs; the moment it has a
// job_id it hands the job to jobTracker.trackJob and stops owning any polling or
// job state. Polling, the segments×frames-scaled movie poll cap
// (generateMoviePollCapMs, plumbed via trackJob.segments/nFrames → capFor), the
// terminal parse, and the centralized library push (segment frames + the movie clip,
// origin "generate") all live in jobTracker — so a movie kicked off here keeps
// running (and lands in the library) even after the user switches sub-tabs (which
// unmounts this station) or reloads the page.
//
// The shape differences from useGenerateSceneJob: the request is a
// GenerateMovieRequest (a goal timeline + director knobs), and `progress` is the
// NESTED MovieProgress (segment strip + a current-segment view) rather than a flat
// JobProgress. Outputs are still the full result.outputs array (segment frames,
// then the movie.mp4 last).
import { useCallback, useMemo, useRef, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import {
  enqueueResultSchema,
  isMovieProgress,
  type GenerateMovieRequest,
  type JobStatus,
  type MediaRef,
  type MovieProgress,
  type JobProject,
} from "../video/contract";
import {
  useJobTracker,
  trackJob,
  resumeJob,
  dismissTerminal,
  isJobActive,
} from "../video/jobTracker";

export interface GenerateMovieJobApi {
  /** Server job id once enqueued, else null. */
  jobId: string | null;
  /** Latest polled status (null = idle or enqueued-but-not-yet-claimed). */
  status: JobStatus;
  /** ALL produced MediaRefs (segment image frames, then the movie clip last). */
  outputs: MediaRef[];
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Poll cap reached while non-terminal — NOT a failure. */
  capped: boolean;
  /** True while enqueuing or polling a non-terminal job. */
  running: boolean;
  /** Live NESTED movie progress while running (segment strip + current view); null otherwise. */
  progress: MovieProgress | null;
  /** Auto-archive project this movie was saved into (populated on done); null otherwise. */
  project: JobProject | null;
  /** Enqueue ONE generate_movie job and begin polling. Replaces any prior run. */
  run: (req: GenerateMovieRequest) => void;
  /** Re-anchor the poll clock and resume polling a capped job. Clears `capped`. */
  resume: () => void;
  /** Forget this station's finished job (called when the composer is cleared). */
  reset: () => void;
}

const STATION = "generate";

/** Flatten the goal-timeline prompts into one provenance summary string. */
function moviePromptSummary(req: GenerateMovieRequest): string {
  return req.goals
    .map((g) => g.prompt.trim())
    .filter((t) => t !== "")
    .join(" · ")
    .trim();
}

export function useGenerateMovieJob(): GenerateMovieJobApi {
  const records = useJobTracker();
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [enqueuing, setEnqueuing] = useState<boolean>(false);
  const [enqueueError, setEnqueueError] = useState<string | null>(null);

  const record = useMemo(() => {
    const mine = records.filter(
      (r) => r.station === STATION && r.kind === "generate_movie",
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

  const run = useCallback((req: GenerateMovieRequest) => {
    setCurrentJobId(null);
    setEnqueueError(null);
    setEnqueuing(true);
    const enqueuedAt = Date.now();

    void request<unknown>(hugpyConfig.generateMovieEnqueueUrl, {
      method: "POST",
      body: JSON.stringify(req),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: STATION, operation: "generate_movie.enqueue" },
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
      // total frame count = max end_frame across the (contiguous) timeline; the
      // poll cap scales off segments × total frames (see generateMoviePollCapMs).
      const framesTotal = req.goals.reduce(
        (mx, g) => Math.max(mx, g.end_frame),
        0,
      );
      trackJob({
        jobId,
        kind: "generate_movie",
        station: STATION,
        label: "generate movie",
        enqueuedAt,
        segments: req.goals.length,
        nFrames: framesTotal,
        chain: req.chain ?? false,
        prompt: moviePromptSummary(req),
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

  // The tracker stores progress as a union (flat OR nested); narrow to the movie
  // shape for this hook's consumers (a generate_movie record only ever carries a
  // movie blob, but the narrow keeps the type honest).
  const progress: MovieProgress | null =
    record?.progress && isMovieProgress(record.progress)
      ? record.progress
      : null;

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
