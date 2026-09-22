// AudioCropCore — the reusable body of the Audio Crop station (Studio slice 2a).
// The station's "Components view" MINUS the upload bar: the video→audio extract
// step (useAudioExtractJob) + the TemporalRegionEditor + the useAudioCropJobs
// enqueue hook + the result grid, parameterised by a `source` ref (audio OR video).
//
// The video→audio extract PATH IS PRESERVED verbatim: when the `source` is a video
// the Core pulls its audio track out first and the editor works over the extracted
// audio; an audio source is edited directly. Behaviour is IDENTICAL to the
// pre-slice-2a AudioCropStation body: same `vi-station-components` shell, same
// extract progress affordance, same TemporalRegionEditor props, same run footer,
// same `vi-result-grid` + AudioJobCard. `onProduced` is an OPTIONAL slice-2b sink;
// when OMITTED (the station case) finished clips still land in the library via the
// tracker exactly as before.
//
// Source lifecycle: the station owned `extract.reset()` + `resetCrops()` + clearing
// the queued windows inside onPick/clearSource, and kicked `extract.run()` for a
// video. The Core now owns those, so it runs them whenever the `source` REFERENCE
// changes (a new pick or a clear). The initial mount is skipped (prevSource seed) so
// a remount after a sub-tab switch does NOT re-run the extract or wipe the durable
// tracker's finished clips.
import { useEffect, useRef, useState } from "react";
import { TemporalRegionEditor } from "../../regions/TemporalRegionEditor";
import type { TemporalRegion } from "../../regions/types";
import { useAudioExtractJob } from "../useAudioExtractJob";
import { useAudioCropJobs, type TrackedAudioJob } from "../useAudioCropJobs";
import { STILL_RUNNING_MESSAGE } from "../pollCaps";
import { mediaBytesUrl } from "../../config";
import type { MediaRef } from "../../video/contract";

export interface AudioCropCoreProps {
  /** Source audio or video, or null before one is loaded. */
  source: MediaRef | null;
  /** Multi-clip queue (station) vs single clip (inline-on-item, slice 2b). */
  multiRegion?: boolean;
  /** Additive (slice 2b): notified with each finished clip's ref + its window.
      Omitted by the station — the clip still lands in the library via the tracker. */
  onProduced?: (
    ref: MediaRef,
    meta: { spatial?: null; temporal?: TemporalRegion | null },
  ) => void;
}

export function AudioCropCore({
  source,
  multiRegion = true,
  onProduced,
}: AudioCropCoreProps) {
  const [queued, setQueued] = useState<TemporalRegion[]>([]);

  // Video sources are routed through the audio_extract chain first.
  const extract = useAudioExtractJob();

  // The audio ref the editor works over: an audio source directly, or the audio
  // track extracted from a video source once that job finishes.
  const audioSource: MediaRef | null =
    source?.kind === "audio"
      ? source
      : source?.kind === "video"
        ? extract.output
        : null;

  const { jobs, runCrops, resume, reset: resetCrops } = useAudioCropJobs(audioSource);

  const duration = audioSource?.duration_s ?? 0;

  // Reset the queue + extract + this station's tracker rows when a NEW source is
  // picked (or cleared), and kick the audio_extract chain for a video — keyed on the
  // source reference so it fires exactly on a pick/clear. The mount is skipped
  // (prevSource seed) so a remount after a tab switch neither re-runs the extract
  // (no duplicate job) nor wipes the durable tracker's finished clips.
  const prevSource = useRef<MediaRef | null>(source);
  useEffect(() => {
    if (prevSource.current === source) return;
    prevSource.current = source;
    setQueued([]);
    extract.reset();
    resetCrops();
    // A video needs its audio track pulled out before the editor can open.
    if (source?.kind === "video") extract.run(source);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  // Additive slice-2b sink: fire once per finished clip. Inert when onProduced is
  // omitted (the station case) — the library push is owned by the tracker regardless.
  const notified = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!onProduced) return;
    for (const j of jobs) {
      if (j.status === "done" && j.output && !notified.current.has(j.key)) {
        notified.current.add(j.key);
        onProduced(j.output, { spatial: null, temporal: j.region });
      }
    }
  }, [jobs, onProduced]);

  const isVideo = source?.kind === "video";
  const extracting = isVideo && !extract.output && extract.status !== "failed";

  return (
    <div className="vi-station-components">
      {/* Video → audio extract progress + still-running affordance. */}
      {isVideo && extracting && (
        <p className="vi-crop-hint">
          {extract.capped
            ? STILL_RUNNING_MESSAGE
            : `Extracting the audio track… ${extract.status ?? "queued"}`}
          {extract.capped && (
            <button
              type="button"
              className="vi-btn vi-btn-sm"
              onClick={extract.resume}
              style={{ marginLeft: "0.6rem" }}
            >
              Check again
            </button>
          )}
        </p>
      )}
      {isVideo && extract.error && (
        <p className="vi-error" role="alert">{extract.error}</p>
      )}

      {/* Audio view + crop/manipulation options (waveform + numeric
          windows) — ALWAYS rendered. With no audio loaded the editor shows
          its full chrome over a blank (flat) waveform; loading audio only
          fills the waveform. */}
      <TemporalRegionEditor
        source={audioSource}
        duration={duration}
        values={queued}
        onChange={(next) => setQueued(Array.isArray(next) ? next : [next])}
        multiRegion={multiRegion}
        numericInputs
      />

      {/* Run */}
      <div className="vi-station-footer">
        <div className="vi-station-footer-main">
          <span className="vi-crop-hint">
            Each queued window enqueues one temporal crop; clips land in the
            media library.
          </span>
        </div>
        <div className="vi-station-footer-run">
          <button
            type="button"
            className="vi-btn vi-btn-accent"
            disabled={!audioSource || queued.length === 0}
            onClick={() => runCrops(queued)}
          >
            Run crop{queued.length === 1 ? "" : "s"}
            {queued.length > 0 ? ` (${queued.length})` : ""}
          </button>
        </div>
      </div>

      {jobs.length > 0 && (
        <div className="vi-result-grid">
          {jobs.map((j) => (
            <AudioJobCard key={j.key} job={j} onResume={() => resume(j.key)} />
          ))}
        </div>
      )}
    </div>
  );
}

const STATUS_TEXT: Record<string, string> = {
  queued: "queued",
  claimed: "claimed",
  running: "running",
  done: "done",
  failed: "failed",
};

function AudioJobCard({
  job,
  onResume,
}: {
  job: TrackedAudioJob;
  onResume: () => void;
}) {
  const statusKey = job.status ?? "queued";
  const done = job.status === "done";
  const failed = job.status === "failed";
  return (
    <figure className="vi-result-card">
      <div className="vi-result-head">
        <span className="vi-region-idx">#{job.index}</span>
        <span className={`vi-status vi-status-${statusKey}`}>{STATUS_TEXT[statusKey]}</span>
        <code className="vi-region-dims">
          {job.region.start_s.toFixed(2)}–{job.region.end_s.toFixed(2)}s
        </code>
      </div>
      <div className="vi-result-body">
        {done && job.output ? (
          <div className="vi-audio-clip">
            <audio controls className="vi-audio-player" src={mediaBytesUrl(job.output.uri)} />
            <span className="vi-gen-note">in library</span>
          </div>
        ) : failed ? (
          <p className="vi-error">{job.error ?? "Audio-crop job failed."}</p>
        ) : job.capped ? (
          <div className="vi-result-pending">
            <p className="vi-crop-hint">{STILL_RUNNING_MESSAGE}</p>
            <button type="button" className="vi-btn vi-btn-sm" onClick={onResume}>
              Check again
            </button>
          </div>
        ) : (
          <div className="vi-result-pending" aria-hidden>
            {STATUS_TEXT[statusKey]}…
          </div>
        )}
      </div>
    </figure>
  );
}
