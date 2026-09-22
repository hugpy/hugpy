// ImageCropCore — the reusable body of the Image Crop station (Studio slice 2a).
// This is the station's "Components view" MINUS the upload bar: the
// SpatialRegionEditor + the useCropJobs enqueue hook + the result grid, all
// parameterised by a `source` ref. The station (or, in slice 2b, an inline-on-item
// surface) supplies the source; nothing here uploads or ingests.
//
// Behaviour is IDENTICAL to the pre-slice-2a ImageCropStation body: same
// `vi-station-components` shell, same SpatialRegionEditor props, same run footer,
// same `vi-result-grid` + JobCard. The one addition is `onProduced` — an OPTIONAL
// sink slice 2b wires to catch each finished crop; when it is OMITTED (the station
// case) the Core keeps the station's exact behaviour (crops land in the library via
// the tracker, the grid renders unchanged).
//
// Source lifecycle: the station owned `reset()` + clearing the queued regions
// inside onPick/clearSource. The Core now owns the `queued` state and the hook, so
// it resets both whenever the `source` REFERENCE changes (i.e. a new pick or a
// clear) — mirroring the station's per-pick reset. The initial mount is skipped
// (prevSource seed) so a remount after a sub-tab switch does NOT wipe the durable
// tracker's finished crops (they persist across tab switches by design).
import { useEffect, useRef, useState } from "react";
import { SpatialRegionEditor } from "../../regions/SpatialRegionEditor";
import type { SpatialRegion } from "../../regions/types";
import { useCropJobs, type TrackedJob } from "../useCropJobs";
import { STILL_RUNNING_MESSAGE } from "../pollCaps";
import { mediaBytesUrl } from "../../config";
import type { MediaRef } from "../../video/contract";

export interface ImageCropCoreProps {
  /** Source image (kind==="image"), or null before one is loaded. */
  source: MediaRef | null;
  /** Multi-crop queue (station) vs single crop (inline-on-item, slice 2b). */
  multiRegion?: boolean;
  /** Additive (slice 2b): notified with each finished crop's ref + its region.
      Omitted by the station — the crop still lands in the library via the tracker. */
  onProduced?: (
    ref: MediaRef,
    meta: { spatial?: SpatialRegion | null; temporal?: null },
  ) => void;
}

export function ImageCropCore({
  source,
  multiRegion = true,
  onProduced,
}: ImageCropCoreProps) {
  const [queued, setQueued] = useState<SpatialRegion[]>([]);
  const { jobs, runCrops, resume, reset } = useCropJobs(source);

  // Reset the queue + this station's tracker rows when a NEW source is picked (or
  // cleared) — keyed on the source reference so it fires exactly on a pick/clear,
  // never on an unrelated re-render. The mount is skipped (prevSource seed) so a
  // remount after a tab switch keeps the durable tracker's finished crops.
  const prevSource = useRef<MediaRef | null>(source);
  useEffect(() => {
    if (prevSource.current === source) return;
    prevSource.current = source;
    setQueued([]);
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  // Additive slice-2b sink: fire once per finished crop. Inert when onProduced is
  // omitted (the station case) — the library push is owned by the tracker regardless.
  const notified = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!onProduced) return;
    for (const j of jobs) {
      if (j.status === "done" && j.output && !notified.current.has(j.key)) {
        notified.current.add(j.key);
        onProduced(j.output, { spatial: j.region, temporal: null });
      }
    }
  }, [jobs, onProduced]);

  return (
    <div className="vi-station-components">
      {/* Image view + crop/manipulation options — ALWAYS rendered. With
          no image loaded the editor shows its full chrome over a blank
          canvas; loading an image only fills the canvas. */}
      <SpatialRegionEditor
        source={source}
        values={queued}
        onChange={(next) => setQueued(Array.isArray(next) ? next : [next])}
        multiRegion={multiRegion}
        numericInputs
      />

      {/* Run */}
      <div className="vi-station-footer">
        <div className="vi-station-footer-main">
          <span className="vi-crop-hint">
            Each queued box enqueues one crop job on the bus.
          </span>
        </div>
        <div className="vi-station-footer-run">
          <button
            type="button"
            className="vi-btn vi-btn-accent"
            disabled={!source || queued.length === 0}
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
            <JobCard key={j.key} job={j} onResume={() => resume(j.key)} />
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

function JobCard({ job, onResume }: { job: TrackedJob; onResume: () => void }) {
  const statusKey = job.status ?? "queued";
  const done = job.status === "done";
  const failed = job.status === "failed";
  return (
    <figure className="vi-result-card">
      <div className="vi-result-head">
        <span className="vi-region-idx">#{job.index}</span>
        <span className={`vi-status vi-status-${statusKey}`}>{STATUS_TEXT[statusKey]}</span>
        <code className="vi-region-dims">
          {job.region.w}×{job.region.h} @ {job.region.x},{job.region.y}
        </code>
      </div>
      <div className="vi-result-body">
        {done && job.output ? (
          <img
            className="vi-result-img"
            src={mediaBytesUrl(job.output.uri)}
            alt={`Crop ${job.index}`}
          />
        ) : failed ? (
          <p className="vi-error">{job.error ?? "Crop job failed."}</p>
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
