// ACTIVE PROCESSES — the console-wide, placement-aware view of every in-flight
// media-bus job (studio / movie / identity / generate / crop / extract), fed by
// GET /video/jobs. Promotes the studio-only StudioActiveProcesses pattern to the
// whole console and adds the two things that were invisible before:
//
//   • WHERE a job physically executes — the `placement` line ("ae · cuda:0 ·
//     P-studio"), with an "external" cue when it runs OFF the central box (an ae
//     worker, ComfyUI, or the identity-render service on :9750), plus reserved
//     VRAM when a GPU reservation is held.
//   • The reservation HOLD phase — a job that can't fit the card yet sits as
//     `awaiting_capacity` (with the shortfall reason on hover + the overtaken
//     count), not a fake "running".
//   • WHAT it is doing right now — a moving progress bar (segments done plus the
//     within-segment denoise fraction the runner reports) and the TAIL of the job's
//     stage timeline, so a running render shows its last few events instead of an
//     opaque "running" chip.
//   • WHY it broke — a terminal-failed job renders its failure envelope VERBATIM
//     (code + message + the stage it died in), never a generic "failed". Terminal
//     rows arrive only with the "Show finished" filter on.
//
// HONEST rendering only: no fabricated progress bars — a numeric progress/stage
// shows only when the server sends one. Cancel hits the same /video/jobs/<id>/
// cancel seam every station uses; a held job cancels exactly like any queued job.
import { useCallback, useEffect, useRef, useState } from "react";
import type { StationSpec } from "./types";
import { request } from "../transport/client";
import { jobCancelUrl } from "../config";
import { ProcessTimeline, shortId } from "./studio/studioShared";
import { SessionsPanel } from "./studio/StudioMovieSessions";
import {
  useMediaJobs,
  placementLine,
  isExternal,
  reservedGib,
  progressLabel,
  latestEvent,
  type MediaJob,
} from "../video/useMediaJobs";
// k117 — the honest clock/stage helpers. Kept OUT of this component so the
// studio's Sessions sections (which reuse JobRow) get the same fix for free.
import { jobClockParts, jobStageLabel, terminalAtStage } from "../video/jobClock";

function PhaseChip({ job }: { job: MediaJob }) {
  if (job.phase === "awaiting_capacity") {
    const reason = job.hold?.reason ? JSON.stringify(job.hold.reason) : "";
    const overtaken = job.hold?.overtaken ?? 0;
    return (
      <span
        className="vi-status vi-status-awaiting_capacity"
        title={
          reason
            ? `Held — waiting for GPU capacity. Shortfall: ${reason}`
            : "Held — waiting for GPU capacity"
        }
      >
        awaiting capacity{overtaken > 0 ? ` · jumped ${overtaken}×` : ""}
      </span>
    );
  }
  return <span className={`vi-status vi-status-${job.phase}`}>{job.phase}</span>;
}

/** The bar + its honest label. Renders nothing when the server sent no number. */
function ProgressBar({ job }: { job: MediaJob }) {
  if (job.progress == null) return null;
  const pct = Math.max(0, Math.min(100, job.progress * 100));
  return (
    <div className="vi-job-progress">
      <div className="vi-job-progress-track">
        <div className="vi-job-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="vi-job-progress-label">{progressLabel(job)}</span>
    </div>
  );
}

/** ONE job row's content (head, placement/meta, progress, timeline, Cancel) —
 *  extracted so the Sessions panel's per-surface sections (Scene/Movie/Clip)
 *  render feed jobs through the SAME renderer as the main list instead of a
 *  parallel copy. The `<li className="vi-studio-active-row">` wrapper (and its
 *  `key`) stays at the call site — this project's scoped tsconfig rejects a
 *  `key` prop on a locally-declared component (the SessionRowBody idiom). */
export function JobRow({
  job: j,
  expanded: isOpen,
  onToggle,
  nowMs,
  cancelBusy,
  onCancel,
}: {
  job: MediaJob;
  expanded: boolean;
  onToggle: () => void;
  nowMs: number;
  cancelBusy: boolean;
  onCancel: () => void;
}) {
  const line = placementLine(j.placement);
  const gib = reservedGib(j.placement);
  const external = isExternal(j.placement);
  const elided = j.stageLogTotal - j.stageLog.length;
  const event = latestEvent(j);
  return (
    <>
      <div className="vi-studio-active-head">
        <button
          type="button"
          className="vi-timeline-toggle"
          onClick={onToggle}
          aria-expanded={isOpen}
          title={isOpen ? "Hide detail" : "Show what it's doing"}
        >
          <span className="vi-timeline-caret" aria-hidden>
            {isOpen ? "▾" : "▸"}
          </span>
        </button>
        <PhaseChip job={j} />
        <span className="vi-studio-active-id">
          {j.name || "job"} · {shortId(j.id)}
        </span>
      </div>
      <div className="vi-studio-active-meta">
        {line ? (
          <span
            className={`vi-placement${external ? " vi-placement-external" : ""}`}
            title={external ? "Runs off the central box" : undefined}
          >
            {external ? "⇄ " : ""}
            {line}
            {external ? " · external" : ""}
          </span>
        ) : null}
        {gib ? <span>· {gib}</span> : null}
        {/* k117: the stage label and the clock, told honestly. A TERMINAL row
            says what it ENDED as (with the stage it was in as a parenthetical)
            and shows its FROZEN duration; a live row shows time-in-stage and
            queue-wait as separate chips. Both come from the server's lifecycle
            fields — nothing here counts off the browser clock, which is what
            made a 9.4s render read as "946m · archiving". */}
        {jobStageLabel(j) ? <span>· {jobStageLabel(j)}</span> : null}
        {terminalAtStage(j) ? (
          <span className="vi-knob-hint" title="the stage it was in when it ended">
            (in {terminalAtStage(j)})
          </span>
        ) : null}
        {jobClockParts(j).map((part) => (
          <span key={part}>· {part}</span>
        ))}
      </div>
      <ProgressBar job={j} />
      {event ? <p className="vi-job-latest">{event}</p> : null}
      {isOpen ? (
        <div className="vi-timeline-panel">
          {/* The SAME renderer the studio tab uses: the failure envelope
              verbatim (code · stage · message), the live "now" line, and
              every stage the render moved through. A live row carries a
              tail; a terminal one carries the whole sequence. */}
          <ProcessTimeline
            stageLog={j.stageLog}
            failure={j.failure}
            currentStage={j.stage}
            lastMovementTs={j.progressedAt}
            terminal={j.terminal}
            nowMs={nowMs}
          />
          {elided > 0 ? (
            <p className="vi-knob-hint">
              …{elided} earlier event{elided === 1 ? "" : "s"} — open the job for
              its full history.
            </p>
          ) : null}
        </div>
      ) : null}
      {j.terminal ? null : (
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          onClick={onCancel}
          disabled={cancelBusy}
        >
          {cancelBusy ? "Cancelling…" : "✕ Cancel"}
        </button>
      )}
    </>
  );
}

// `spec` is optional because this body has TWO mounts: the registry's /active
// station (which always passes one) and the sidebar's ActivePanel overlay
// (2026-08-06), which mounts it inside the shared drawer shell and has no spec
// to hand it. The prop was already unused here, so optional costs nothing and
// keeps ONE renderer for the feed instead of an overlay-only copy.
export function ActiveProcessesStation({ spec: _spec }: { spec?: StationSpec } = {}) {
  const [showFinished, setShowFinished] = useState(false);
  const { jobs, refetch } = useMediaJobs(showFinished);
  // Expanded rows + a 1s clock, exactly as the studio tab's sibling panel does —
  // ProcessTimeline needs `nowMs` to age its "…since last movement" line.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const toggle = useCallback((id: string) => {
    setExpanded((prev) => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  }, []);
  const [cancelBusy, setCancelBusy] = useState<Set<string>>(new Set());
  const mounted = useRef(true);

  const onCancel = useCallback(
    async (jobId: string) => {
      setCancelBusy((prev) => new Set(prev).add(jobId));
      try {
        await request<unknown>(jobCancelUrl(jobId), {
          method: "POST",
          meta: { specKey: "media", operation: "media.job.cancel" },
        });
        if (!mounted.current) return;
        refetch();
      } finally {
        if (mounted.current)
          setCancelBusy((prev) => {
            const n = new Set(prev);
            n.delete(jobId);
            return n;
          });
      }
    },
    [refetch],
  );

  return (
    <section className="station-card station-wide vi-studio-active">
      <h2>Active Processes</h2>
      <p className="vi-knob-hint">
        Every in-flight media build across the console, and WHERE it physically runs.
        A build marked <strong>external</strong> is executing off this box — on the{" "}
        ae worker, ComfyUI, or the identity-render service.
      </p>
      <label className="vi-knob-hint vi-job-filter">
        <input
          type="checkbox"
          checked={showFinished}
          onChange={(e) => setShowFinished(e.target.checked)}
        />{" "}
        Show finished — recent done/failed/cancelled builds, with each failure&apos;s
        full stage timeline and its exact error.
      </label>
      {jobs.length === 0 ? (
        <p className="vi-knob-hint">
          No active builds — enqueue a render and it appears here with its placement.
        </p>
      ) : (
        <ul className="vi-studio-active-list">
          {jobs.map((j) => (
            <li key={j.id} className="vi-studio-active-row">
              <JobRow
                job={j}
                expanded={expanded.has(j.id)}
                onToggle={() => toggle(j.id)}
                nowMs={nowMs}
                cancelBusy={cancelBusy.has(j.id)}
                onCancel={() => void onCancel(j.id)}
              />
            </li>
          ))}
        </ul>
      )}
      {/* SESSIONS (operator ask 2026-08-12): the recovery surface lives HERE now —
          relocated from the Cinema composer so "what can I pick back up" is visible
          from every station, split per generate surface. It reuses THIS station's
          feed + row renderer; only the Cinema section owns an extra poll (the
          durable sessions walk), and only while it is open. */}
      <SessionsPanel
        jobs={jobs}
        nowMs={nowMs}
        cancelBusy={cancelBusy}
        onCancel={(id) => void onCancel(id)}
      />
    </section>
  );
}
