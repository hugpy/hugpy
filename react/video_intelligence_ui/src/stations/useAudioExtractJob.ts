// The Audio Crop station's video→audio step, now riding the session-wide job
// tracker. When the ingested source is a VIDEO we pull its audio track out into
// a standalone audio MediaRef before the temporal editor can open. This hook
// still owns the request-body construction (fmt "wav") and POSTs it through
// request<T>() + config URLs; the moment it has a job_id it hands the job to
// jobTracker.trackJob and stops owning any polling or job state. Polling, the
// poll cap, and the terminal parse all live in jobTracker now — so the extract
// keeps running even after a sub-tab switch or a page reload.
//
// The extracted audio is INTERMEDIATE (it feeds the temporal editor), so the
// tracker deliberately does NOT push it to the media library — only the finished
// crops are (see useAudioCropJobs). It is still tracked/polled so the process is
// visible and survives tab switches.
//
// The public API to AudioCropStation is UNCHANGED:
// `{ jobId, status, output, error, capped, running, run, resume, reset }`.
import { useCallback, useMemo, useRef, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import {
  enqueueResultSchema,
  type AudioExtractRequest,
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

export interface AudioExtractJobApi {
  /** Server job id once enqueued, else null. */
  jobId: string | null;
  /** Latest polled status (null = idle or enqueued-but-not-yet-claimed). */
  status: JobStatus;
  /** The extracted audio MediaRef (populated on done). */
  output: MediaRef | null;
  /** Human message on failure (enqueue error or job error). */
  error: string | null;
  /** Poll cap reached while non-terminal (Phase 8) — NOT a failure. */
  capped: boolean;
  /** True while enqueuing or polling a non-terminal job. */
  running: boolean;
  /** Enqueue ONE audio_extract job for a video source (fmt "wav") and poll. */
  run: (source: MediaRef) => void;
  /** Re-anchor the poll clock and resume polling a capped job. Clears `capped`. */
  resume: () => void;
  /** Forget this station's finished extract (called when a new source is ingested). */
  reset: () => void;
}

const STATION = "audio-crop";

export function useAudioExtractJob(): AudioExtractJobApi {
  const records = useJobTracker();
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [enqueuing, setEnqueuing] = useState<boolean>(false);
  const [enqueueError, setEnqueueError] = useState<string | null>(null);

  // Only the audio_extract kind — the crop kind on this same station is owned by
  // useAudioCropJobs, so the two never cross-derive.
  const record = useMemo(() => {
    const mine = records.filter(
      (r) => r.station === STATION && r.kind === "audio_extract",
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
    // Only forget finished audio_extract rows — leave any in-flight crop of this
    // station alone (useAudioCropJobs.reset handles the crop rows).
    dismissTerminal(STATION);
  }, []);

  const resume = useCallback(() => {
    const id = jobIdRef.current;
    if (id) resumeJob(id);
  }, []);

  const run = useCallback((source: MediaRef) => {
    // Fresh run — default fmt "wav" (lossless, trivially decodable for the waveform).
    setCurrentJobId(null);
    setEnqueueError(null);
    setEnqueuing(true);
    const enqueuedAt = Date.now();

    const body: AudioExtractRequest = { source, fmt: "wav" };
    void request<unknown>(hugpyConfig.audioExtractEnqueueUrl, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: STATION, operation: "audio_extract.enqueue" },
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
        kind: "audio_extract",
        station: STATION,
        label: "extract audio track",
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
    output: record?.outputs[0] ?? null,
    error,
    capped: record?.capped ?? false,
    running: enqueuing || (record != null && isJobActive(record)),
    run,
    resume,
    reset,
  };
}
