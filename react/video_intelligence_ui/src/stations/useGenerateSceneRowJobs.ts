// The Scene composer's PER-PART enqueue surface: N independent scene prompts ->
// N generate_scene jobs, each with its OWN settings, tracked independently so
// each prompt component produces and renders its own output ("2 prompts = 2
// outputs"). Rides the session-wide jobTracker (so a fan-out survives an unmount
// / reload) under a DISTINCT station key.
//
// SEQUENTIAL (operator 2026-08-10): the jobs run ONE AT A TIME — a component's
// generation must fully finish before the next starts. On a single GPU that both
// matches "test all models" reality (no concurrency benefit for big models) and
// avoids the VRAM contention / polite-load refusals that a simultaneous fan-out
// hit (a 17 GB model can't load while N others are mid-load). Not-yet-started
// rows show as "queued"; each output appears as its model completes.
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
} from "../video/contract";
import {
  useJobTracker,
  trackJob,
  resumeJob,
  dismissTerminal,
  isJobActive,
  type TrackedJob,
} from "../video/jobTracker";

/** Live state of ONE scene part's job. */
export interface SceneRowJobState {
  jobId: string | null;
  status: JobStatus;
  /** ALL produced MediaRefs (N image frames, then the clip last if assembled). */
  outputs: MediaRef[];
  error: string | null;
  running: boolean;
  /** Waiting its turn in the sequential queue (not yet submitted). */
  queued: boolean;
  /** Poll cap reached while non-terminal — NOT a failure; resume() re-anchors it. */
  capped: boolean;
  progress: JobProgress | null;
}

export interface GenerateSceneRowJobsApi {
  /** Part id -> that part's job state. Parts never run are absent. */
  byRow: Record<string, SceneRowJobState>;
  /** True while ANY part is queued, enqueuing, or polling. */
  anyRunning: boolean;
  /** Enqueue one scene job per part — SEQUENTIALLY. Replaces the previous fan-out. */
  runRows: (rows: { rowId: string; req: GenerateSceneRequest }[]) => void;
  /** Re-anchor the poll clock and resume polling a capped part's job. */
  resume: (rowId: string) => void;
  /** Forget this surface's finished jobs and stop any in-flight sequence. */
  reset: () => void;
}

const STATION = "generate-scene-parts";
const WAIT_CAP_MS = 20 * 60 * 1000; // don't let one wedged job stall the whole sweep

const IDLE: SceneRowJobState = {
  jobId: null,
  status: null,
  outputs: [],
  error: null,
  running: false,
  queued: false,
  capped: false,
  progress: null,
};

const sleep = (ms: number) => new Promise<void>((res) => setTimeout(res, ms));

export function useGenerateSceneRowJobs(): GenerateSceneRowJobsApi {
  const records = useJobTracker();
  // Latest tracker records, readable inside the async sequential loop (no stale
  // closure) so it can wait for a job to go terminal using the tracker's own poll.
  const recordsRef = useRef(records);
  recordsRef.current = records;

  const [jobByRow, setJobByRow] = useState<Record<string, string>>({});
  const [enqueueErrors, setEnqueueErrors] = useState<Record<string, string>>({});
  const [enqueuing, setEnqueuing] = useState<Record<string, boolean>>({});
  const [queued, setQueued] = useState<Record<string, boolean>>({});
  // Bumped on each runRows / reset so a superseded sequence stops itself.
  const runSeq = useRef(0);

  const byRow = useMemo<Record<string, SceneRowJobState>>(() => {
    const mineByJob = new Map(
      records
        .filter((r) => r.station === STATION && r.kind === "generate_scene")
        .map((r) => [r.jobId, r] as const),
    );
    const newestByRowKey = new Map<string, TrackedJob>();
    for (const r of records) {
      if (r.station !== STATION || r.kind !== "generate_scene" || !r.rowKey) continue;
      const prev = newestByRowKey.get(r.rowKey);
      if (!prev || r.enqueuedAt >= prev.enqueuedAt) newestByRowKey.set(r.rowKey, r);
    }
    const out: Record<string, SceneRowJobState> = {};
    const rowIds = new Set([
      ...newestByRowKey.keys(),
      ...Object.keys(jobByRow),
      ...Object.keys(enqueueErrors),
      ...Object.keys(enqueuing),
      ...Object.keys(queued),
    ]);
    for (const rowId of rowIds) {
      const enqErr = enqueueErrors[rowId];
      if (enqErr) {
        out[rowId] = { ...IDLE, status: "failed", error: enqErr };
        continue;
      }
      // Waiting its turn (sequential queue) and not yet submitted.
      if (queued[rowId] && !jobByRow[rowId]) {
        out[rowId] = { ...IDLE, queued: true, status: "queued" };
        continue;
      }
      // Enqueue window: submitted, no job_id yet — show running, keep Run disabled.
      if (enqueuing[rowId] && !jobByRow[rowId]) {
        out[rowId] = { ...IDLE, running: true };
        continue;
      }
      const jobId = jobByRow[rowId];
      const rec =
        (jobId ? mineByJob.get(jobId) : undefined) ?? newestByRowKey.get(rowId);
      if (!rec) {
        out[rowId] = { ...IDLE, running: !!enqueuing[rowId] };
        continue;
      }
      out[rowId] = {
        jobId: rec.jobId,
        status: rec.status,
        outputs: rec.outputs ?? [],
        error:
          rec.error ??
          (rec.expired ? "This job is no longer tracked by the server." : null),
        running: isJobActive(rec),
        queued: false,
        capped: rec.capped ?? false,
        progress:
          rec.progress && !isMovieProgress(rec.progress) ? rec.progress : null,
      };
    }
    return out;
  }, [records, jobByRow, enqueueErrors, enqueuing, queued]);

  const resume = useCallback(
    (rowId: string) => {
      const jobId = byRow[rowId]?.jobId;
      if (jobId) resumeJob(jobId);
    },
    [byRow],
  );

  const anyRunning = useMemo(
    () =>
      Object.values(byRow).some(
        (s: SceneRowJobState) => s.running || s.queued,
      ) || Object.keys(enqueuing).length > 0,
    [byRow, enqueuing],
  );

  const reset = useCallback(() => {
    runSeq.current += 1; // stop any in-flight sequence
    setJobByRow({});
    setEnqueueErrors({});
    setEnqueuing({});
    setQueued({});
    dismissTerminal(STATION);
  }, []);

  const runRows = useCallback(
    (rows: { rowId: string; req: GenerateSceneRequest }[]) => {
      if (rows.length === 0) return;
      const myRun = ++runSeq.current;
      setJobByRow({});
      setEnqueueErrors({});
      setEnqueuing({});
      // Everything starts queued; each row leaves the queue when its turn comes.
      setQueued(Object.fromEntries(rows.map((r) => [r.rowId, true])));

      const waitTerminal = async (jobId: string) => {
        const start = Date.now();
        while (Date.now() - start < WAIT_CAP_MS) {
          if (runSeq.current !== myRun) return; // superseded by a newer run/reset
          const rec = recordsRef.current.find(
            (r) => r.station === STATION && r.jobId === jobId,
          );
          if (
            rec &&
            (rec.status === "done" ||
              rec.status === "failed" ||
              rec.expired ||
              rec.capped)
          )
            return;
          await sleep(1200);
        }
      };

      void (async () => {
        for (const { rowId, req } of rows) {
          if (runSeq.current !== myRun) return;
          // This row's turn: leave the queue, mark enqueuing.
          setQueued((prev) => {
            const n = { ...prev };
            delete n[rowId];
            return n;
          });
          setEnqueuing((prev) => ({ ...prev, [rowId]: true }));
          const enqueuedAt = Date.now();
          let jobId: string | null = null;
          try {
            const r = await request<unknown>(hugpyConfig.generateSceneEnqueueUrl, {
              method: "POST",
              body: JSON.stringify(req),
              headers: { "Content-Type": "application/json" },
              meta: { specKey: STATION, operation: "generate_scene.enqueue" },
            });
            setEnqueuing((prev) => {
              const n = { ...prev };
              delete n[rowId];
              return n;
            });
            if (!r.ok) {
              setEnqueueErrors((prev) => ({
                ...prev,
                [rowId]: describeAppError(errorOf(r)),
              }));
              continue;
            }
            const parsed = enqueueResultSchema.safeParse(okValue(r));
            if (!parsed.success) {
              setEnqueueErrors((prev) => ({
                ...prev,
                [rowId]: "Malformed enqueue response.",
              }));
              continue;
            }
            jobId = parsed.data.job_id;
            trackJob({
              jobId,
              rowKey: rowId,
              kind: "generate_scene",
              station: STATION,
              label: "scene part",
              enqueuedAt,
              nFrames: req.n_frames,
              chain: req.chain,
              prompt: promptSummary(req.parts),
              model: req.model_id,
            });
            const jid = jobId;
            setJobByRow((prev) => ({ ...prev, [rowId]: jid }));
          } catch (e) {
            setEnqueuing((prev) => {
              const n = { ...prev };
              delete n[rowId];
              return n;
            });
            setEnqueueErrors((prev) => ({
              ...prev,
              [rowId]: e instanceof Error ? e.message : String(e),
            }));
            continue;
          }
          // SEQUENTIAL: block until this model's job is terminal before the next,
          // so each generation gets a clean GPU (no simultaneous-load contention).
          await waitTerminal(jobId);
        }
      })();
    },
    [],
  );

  return { byRow, anyRunning, runRows, resume, reset };
}
