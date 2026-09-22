// useVideoCropJobs — the Video Crop enqueue surface (Studio slice 2a). The
// spatial+temporal sibling of useCropJobs: it POSTs ONE crop job carrying BOTH a
// spatial bbox AND a temporal window, because the crop backend applies both in a
// SINGLE ffmpeg pass (`-ss <start> -i <in> -t <dur> -vf crop=w:h:x:y`) when the
// CropSpec carries both axes — so a video crop is one `/video/jobs/crop` job, NOT a
// two-job chain. (useCropJobs hardcodes `temporal: null`; useAudioCropJobs hardcodes
// `spatial: null`; this hook sets both.)
//
// Everything else mirrors useCropJobs: it owns only request-body construction; the
// moment it has a job_id it hands the job to jobTracker.trackJob and stops owning
// polling/job state (poll cadence, caps, terminal parse, the single library push all
// live in the tracker). `jobs` is DERIVED from the tracker (this station's crop
// records) merged with any still-enqueuing local placeholders.
//
// StationId: this hook rides the dedicated "video-crop" StationId (added to the
// closed union in jobTracker.ts as part of Studio slice 2b). It was modelled on
// useCropJobs (which rides "image-crop") and keeps the video crop's spatial region
// on the tracker record. The dedicated id is REQUIRED now that VideoCropCore is
// wired ALONGSIDE ImageCropCore on studio items: both filter `station==<id> &&
// kind==="crop"`, so a shared id would intermix their records — "video-crop" keeps
// each op's tracker rows (and Video Crop's library origin) cleanly separate.
import { useCallback, useMemo, useRef, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import { hugpyConfig } from "../../config";
import {
  enqueueResultSchema,
  type CropJobRequest,
  type JobStatus,
  type MediaRef,
} from "../../video/contract";
import {
  useJobTracker,
  trackJob,
  resumeJob,
  resumeStation,
  dismissTerminal,
  type TrackedJob as TrackerRecord,
} from "../../video/jobTracker";
import type { SpatialRegion, TemporalRegion } from "../../regions/types";

export interface TrackedVideoCropJob {
  /** Stable local key (the server job id once enqueued). */
  key: string;
  /** 1-based position, for display. */
  index: number;
  /** The native-px spatial box this job crops (kept on the tracker record). */
  spatial: SpatialRegion | null;
  /** The temporal window (seconds) this job trims — local only (see note below). */
  temporal: TemporalRegion | null;
  /** Server job id once enqueued, else null. */
  jobId: string | null;
  /** Latest polled status (null = enqueued but not yet claimed). */
  status: JobStatus;
  /** Result clip MediaRef on success. */
  output: MediaRef | null;
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Poll cap reached while non-terminal — NOT a failure. */
  capped: boolean;
  /** Enqueue timestamp — the anchor for the poll cap. */
  startedAt: number;
}

// A pre-enqueue placeholder — see useCropJobs.ts for the rationale. It carries BOTH
// axes; the tracker record it hands off to only carries the spatial region (its
// `region` field is a single Region), so `temporal` on a job is surfaced from the
// placeholder and is null once the tracker owns the row.
interface Pending {
  key: string;
  index: number;
  spatial: SpatialRegion | null;
  temporal: TemporalRegion | null;
  status: Extract<JobStatus, "queued" | "failed">;
  error: string | null;
  startedAt: number;
}

let _seq = 0;
function makeKey(): string {
  _seq += 1;
  return `vcrop_${_seq.toString(36)}_${Date.now().toString(36)}`;
}

export interface VideoCropJobsApi {
  jobs: TrackedVideoCropJob[];
  /** Enqueue ONE crop job carrying BOTH the spatial box and the temporal window. */
  runCrop: (spatial: SpatialRegion | null, temporal: TemporalRegion | null) => void;
  /**
   * Re-anchor the poll clock for a capped job (or every capped crop when `key`
   * is omitted) and resume polling. Clears `capped`, keeps status.
   */
  resume: (key?: string) => void;
  /** Forget this station's finished jobs (called when a new source is ingested). */
  reset: () => void;
}

const STATION = "video-crop";

export function useVideoCropJobs(source: MediaRef | null): VideoCropJobsApi {
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

  const jobs = useMemo<TrackedVideoCropJob[]>(() => {
    const fromTracker: TrackedVideoCropJob[] = mine.map((r) => ({
      key: r.jobId,
      index: r.index,
      spatial: (r.region as SpatialRegion) ?? null,
      temporal: null,
      jobId: r.jobId,
      status: r.status,
      output: r.outputs[0] ?? null,
      error: r.error,
      capped: r.capped,
      startedAt: r.startedAt,
    }));
    const fromPending: TrackedVideoCropJob[] = pending.map((p) => ({
      key: p.key,
      index: p.index,
      spatial: p.spatial,
      temporal: p.temporal,
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

  const runCrop = useCallback(
    (spatial: SpatialRegion | null, temporal: TemporalRegion | null) => {
      if (!source) return;
      // Continue the 1-based numbering past any existing/pending rows.
      const base = Math.max(
        0,
        ...mineRef.current.map((r) => r.index),
        ...pendingRef.current.map((p) => p.index),
      );
      const key = makeKey();
      const index = base + 1;
      const enqueuedAt = Date.now();
      const dims = spatial ? `${spatial.w}×${spatial.h}` : "full";
      const win = temporal
        ? ` ${temporal.start_s.toFixed(2)}–${temporal.end_s.toFixed(2)}s`
        : "";
      const label = `clip ${dims}${win}`;
      setPending((prev) => [
        ...prev,
        { key, index, spatial, temporal, status: "queued", error: null, startedAt: enqueuedAt },
      ]);

      // ONE crop job, BOTH axes — the backend applies spatial + temporal in a single
      // ffmpeg pass when the CropSpec carries both.
      const body: CropJobRequest = { source, spatial, temporal };
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
        // Hand the job to the durable tracker, then drop the local placeholder. The
        // tracker record carries the spatial region (single Region slot); the temporal
        // window lives only on the placeholder above.
        trackJob({
          jobId: parsed.data.job_id,
          kind: "crop",
          station: STATION,
          label,
          libLabel: label,
          enqueuedAt,
          index,
          region: spatial,
        });
        setPending((prev) => prev.filter((p) => p.key !== key));
      });
    },
    [source],
  );

  return { jobs, runCrop, resume, reset };
}
