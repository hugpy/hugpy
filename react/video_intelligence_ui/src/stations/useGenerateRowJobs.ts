// The image TESTER's enqueue surface: N prompt rows -> N generate jobs, each
// with its OWN settings, tracked independently so they can be compared side by
// side.
//
// This is the multi-job sibling of useGenerateJob (ONE prompt -> ONE job). It
// rides the same session-wide jobTracker, so a tester run keeps going (and lands
// in the library) after the station unmounts or the page reloads — but it books
// its jobs under a DISTINCT station key.
//
// That station key matters. useGenerateJob resolves "my job" as the NEWEST
// record with station === "generate", so if tester rows were filed under the
// same key, every row would hijack the base composer's result tile and the last
// row to enqueue would masquerade as the single-prompt run. Filing tester jobs
// under "generate-tester" keeps the two surfaces from ever reading each other's
// records.
import { useCallback, useMemo, useState } from "react";
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
} from "../video/contract";
import {
  useJobTracker,
  trackJob,
  dismissTerminal,
  isJobActive,
} from "../video/jobTracker";

/** Live state of ONE tester row's job. */
export interface RowJobState {
  jobId: string | null;
  status: JobStatus;
  output: MediaRef | null;
  error: string | null;
  running: boolean;
  progress: JobProgress | null;
}

export interface GenerateRowJobsApi {
  /** Row id -> that row's job state. Rows never run are absent. */
  byRow: Record<string, RowJobState>;
  /** True while ANY row is enqueuing or polling. */
  anyRunning: boolean;
  /** Enqueue one job per row. Replaces the previous tester run. */
  runRows: (rows: { rowId: string; req: GenerateImageRequest }[]) => void;
  /** Forget this tester's finished jobs. */
  reset: () => void;
}

const STATION = "generate-tester";

const IDLE: RowJobState = {
  jobId: null,
  status: null,
  output: null,
  error: null,
  running: false,
  progress: null,
};

export function useGenerateRowJobs(): GenerateRowJobsApi {
  const records = useJobTracker();
  // rowId -> jobId for the CURRENT run. Rebuilt on every runRows so a re-run
  // never shows a stale row's previous image beside a fresh one.
  const [jobByRow, setJobByRow] = useState<Record<string, string>>({});
  // rowId -> enqueue-time failure (never reached a job_id, so the tracker has
  // nothing to report for it).
  const [enqueueErrors, setEnqueueErrors] = useState<Record<string, string>>({});
  const [enqueuing, setEnqueuing] = useState<Record<string, boolean>>({});

  const byRow = useMemo<Record<string, RowJobState>>(() => {
    const mine = new Map(
      records
        .filter((r) => r.station === STATION && r.kind === "generate_image")
        .map((r) => [r.jobId, r] as const),
    );
    const out: Record<string, RowJobState> = {};
    const rowIds = new Set([
      ...Object.keys(jobByRow),
      ...Object.keys(enqueueErrors),
      ...Object.keys(enqueuing),
    ]);
    for (const rowId of rowIds) {
      const enqErr = enqueueErrors[rowId];
      if (enqErr) {
        out[rowId] = { ...IDLE, status: "failed", error: enqErr };
        continue;
      }
      const jobId = jobByRow[rowId];
      const rec = jobId ? mine.get(jobId) : undefined;
      if (!rec) {
        // Enqueued but no job_id yet (or the tracker dropped it): running only
        // while the POST is genuinely in flight, so a lost record reads idle
        // rather than spinning forever.
        out[rowId] = { ...IDLE, running: !!enqueuing[rowId] };
        continue;
      }
      out[rowId] = {
        jobId: rec.jobId,
        status: rec.status,
        output: rec.outputs[0] ?? null,
        error:
          rec.error ??
          (rec.expired ? "This job is no longer tracked by the server." : null),
        running: isJobActive(rec),
        // generate_image only ever carries flat progress — narrow off the
        // tracker's union so a movie blob can never land here.
        progress:
          rec.progress && !isMovieProgress(rec.progress) ? rec.progress : null,
      };
    }
    return out;
  }, [records, jobByRow, enqueueErrors, enqueuing]);

  const anyRunning = useMemo(
    () => Object.keys(byRow).some((rowId) => byRow[rowId].running),
    [byRow],
  );

  const reset = useCallback(() => {
    setJobByRow({});
    setEnqueueErrors({});
    setEnqueuing({});
    dismissTerminal(STATION);
  }, []);

  const runRows = useCallback(
    (rows: { rowId: string; req: GenerateImageRequest }[]) => {
      if (rows.length === 0) return;
      // Clear the previous run up front so no row shows a stale result while the
      // new one enqueues.
      setJobByRow({});
      setEnqueueErrors({});
      setEnqueuing(Object.fromEntries(rows.map((r) => [r.rowId, true])));

      for (const { rowId, req } of rows) {
        const enqueuedAt = Date.now();
        void request<unknown>(hugpyConfig.generateEnqueueUrl, {
          method: "POST",
          body: JSON.stringify(req),
          headers: { "Content-Type": "application/json" },
          meta: { specKey: STATION, operation: "generate_image.enqueue" },
        }).then((r) => {
          setEnqueuing((prev) => ({ ...prev, [rowId]: false }));
          if (!r.ok) {
            setEnqueueErrors((prev) => ({
              ...prev,
              [rowId]: describeAppError(errorOf(r)),
            }));
            return;
          }
          const parsed = enqueueResultSchema.safeParse(okValue(r));
          if (!parsed.success) {
            setEnqueueErrors((prev) => ({
              ...prev,
              [rowId]: "Malformed enqueue response.",
            }));
            return;
          }
          const jobId = parsed.data.job_id;
          trackJob({
            jobId,
            kind: "generate_image",
            station: STATION,
            label: "tester row",
            enqueuedAt,
            prompt: promptSummary(req.parts),
            model: req.model_id,
          });
          setJobByRow((prev) => ({ ...prev, [rowId]: jobId }));
        });
      }
    },
    [],
  );

  return { byRow, anyRunning, runRows, reset };
}
