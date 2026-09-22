// VideoCropCore — the NEW op primitive of Studio slice 2a: a spatial+temporal crop
// of a VIDEO in ONE pass. It renders the SpatialRegionEditor (a native-px bbox over
// the video frame) AND the TemporalRegionEditor (a [start,end) window over the
// clip's timeline) together, and enqueues a SINGLE `/video/jobs/crop` job carrying
// BOTH axes via useVideoCropJobs — because the crop backend applies spatial + temporal
// in one ffmpeg pass when the CropSpec carries both (`-ss <start> -i <in> -t <dur> -vf
// crop=w:h:x:y`). It is NOT a two-job chain.
//
// This Core is additive: it is not wired into any station in slice 2a (no new
// user-facing surface — that is slice 2b). It exists as the reusable body 2b will
// mount inline-on-item. It follows the crop-core shape: a `source` ref + an OPTIONAL
// `onProduced` sink, the same `vi-station-components` shell + `vi-result-grid`.
//
// The result card is inlined onto an intrinsic <figure key=…> (the codebase's idiom
// for a keyed map — mirrors SpatialRegionEditor's <li key> and FrameExtractCore's
// <button key>), so this NEW file stays free of the bare-tsc `key`-prop artifact.
//
// Duration guard: a spatial crop needs the video's native width/height (always
// present for a video); a temporal trim needs a real duration. Mirroring how the
// Audio Crop station reads `duration = source.duration_s ?? 0`, when the source
// reports NO duration the Run is DISABLED (and a hint shown) rather than sending a
// bad interval to the backend. Both axes must be set to run (a video crop is, by
// definition, both a box and a window).
import { useEffect, useRef, useState } from "react";
import { SpatialRegionEditor } from "../../regions/SpatialRegionEditor";
import { TemporalRegionEditor } from "../../regions/TemporalRegionEditor";
import type { SpatialRegion, TemporalRegion } from "../../regions/types";
import { useVideoCropJobs } from "./useVideoCropJobs";
import { STILL_RUNNING_MESSAGE } from "../pollCaps";
import { mediaBytesUrl } from "../../config";
import type { MediaRef } from "../../video/contract";

const STATUS_TEXT: Record<string, string> = {
  queued: "queued",
  claimed: "claimed",
  running: "running",
  done: "done",
  failed: "failed",
};

export interface VideoCropCoreProps {
  /** Source video (kind==="video"), or null before one is loaded. */
  source: MediaRef | null;
  /** Additive (slice 2b): notified with each finished clip's ref + both axes.
      Omitted by callers — the clip still lands in the library via the tracker. */
  onProduced?: (
    ref: MediaRef,
    meta: { spatial?: SpatialRegion | null; temporal?: TemporalRegion | null },
  ) => void;
}

export function VideoCropCore({ source, onProduced }: VideoCropCoreProps) {
  // A video crop is a single {spatial, temporal} pair → ONE job.
  const [spatial, setSpatial] = useState<SpatialRegion | null>(null);
  const [temporal, setTemporal] = useState<TemporalRegion | null>(null);

  const { jobs, runCrop, resume, reset } = useVideoCropJobs(source);

  const duration = source?.duration_s ?? 0;
  const hasDuration = duration > 0;

  // Reset the working pair + this station's tracker rows when a NEW source is picked
  // (or cleared) — keyed on the source reference, mount skipped (prevSource seed).
  const prevSource = useRef<MediaRef | null>(source);
  useEffect(() => {
    if (prevSource.current === source) return;
    prevSource.current = source;
    setSpatial(null);
    setTemporal(null);
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  // Additive sink: fire once per finished clip. Inert when onProduced is omitted —
  // the library push is owned by the tracker regardless.
  const notified = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!onProduced) return;
    for (const j of jobs) {
      if (j.status === "done" && j.output && !notified.current.has(j.key)) {
        notified.current.add(j.key);
        onProduced(j.output, { spatial: j.spatial, temporal: j.temporal });
      }
    }
  }, [jobs, onProduced]);

  const canRun = !!source && !!spatial && !!temporal && hasDuration;

  return (
    <div className="vi-station-components">
      {/* Spatial box over the video frame (native px). Single-region: "Set crop". */}
      <SpatialRegionEditor
        source={source}
        onChange={(next) => setSpatial(Array.isArray(next) ? next[0] ?? null : next)}
        multiRegion={false}
        numericInputs
      />

      {/* Temporal window over the clip timeline. When the source reports no duration
          the editor cannot place a valid window — show a hint and gate Run. */}
      {hasDuration ? (
        <TemporalRegionEditor
          source={source}
          duration={duration}
          onChange={(next) => setTemporal(Array.isArray(next) ? next[0] ?? null : next)}
          multiRegion={false}
          numericInputs
        />
      ) : (
        <p className="vi-crop-hint">
          {source
            ? "This video reports no duration — temporal trimming is unavailable."
            : "Load a video to set a spatial box and a temporal window."}
        </p>
      )}

      {/* Run */}
      <div className="vi-station-footer">
        <div className="vi-station-footer-main">
          <span className="vi-crop-hint">
            One crop job trims the video to the box AND the window in a single pass.
          </span>
        </div>
        <div className="vi-station-footer-run">
          <button
            type="button"
            className="vi-btn vi-btn-accent"
            disabled={!canRun}
            onClick={() => runCrop(spatial, temporal)}
          >
            Run video crop
          </button>
        </div>
      </div>

      {jobs.length > 0 && (
        <div className="vi-result-grid">
          {jobs.map((j) => {
            const statusKey = j.status ?? "queued";
            const done = j.status === "done";
            const failed = j.status === "failed";
            const box = j.spatial
              ? `${j.spatial.w}×${j.spatial.h} @ ${j.spatial.x},${j.spatial.y}`
              : "full frame";
            const win = j.temporal
              ? ` · ${j.temporal.start_s.toFixed(2)}–${j.temporal.end_s.toFixed(2)}s`
              : "";
            return (
              <figure key={j.key} className="vi-result-card">
                <div className="vi-result-head">
                  <span className="vi-region-idx">#{j.index}</span>
                  <span className={`vi-status vi-status-${statusKey}`}>
                    {STATUS_TEXT[statusKey]}
                  </span>
                  <code className="vi-region-dims">
                    {box}
                    {win}
                  </code>
                </div>
                <div className="vi-result-body">
                  {done && j.output ? (
                    <video
                      controls
                      className="vi-result-img"
                      src={mediaBytesUrl(j.output.uri)}
                    />
                  ) : failed ? (
                    <p className="vi-error">{j.error ?? "Crop job failed."}</p>
                  ) : j.capped ? (
                    <div className="vi-result-pending">
                      <p className="vi-crop-hint">{STILL_RUNNING_MESSAGE}</p>
                      <button
                        type="button"
                        className="vi-btn vi-btn-sm"
                        onClick={() => resume(j.key)}
                      >
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
          })}
        </div>
      )}
    </div>
  );
}
