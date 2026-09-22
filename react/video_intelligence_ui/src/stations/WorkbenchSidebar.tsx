// The static left sidebar of the Studio workbench, split into two switchable
// tabs (role="tablist" above the tab bodies):
//
//   • Active Processes — the session-wide "memory of processes": every in-flight
//     or recently-relevant tracked job (live status/spinner via isJobActive, a
//     Check-again for capped jobs, and a Cancel affordance wired to cancelJob).
//   • Session Library — the completed outputs. Generate-station outputs are
//     GROUPED by their producing run (expandable), each group annotated with the
//     prompt + model that produced it and offering Continue (extend a scene from
//     its last frame) / Replicate (re-stage prompt+model) — both wired through
//     the composer seam (requestStage), NOT a parallel generation path. Every
//     output row keeps the existing Add-to-prompt / assess / Remove affordances.
//     Non-generation media (frames, crops, uploads) list flat below the groups.
//
// This is still a VIEW over the library + tracker seams: it never enqueues jobs
// and never DELETES server media. "Remove" / "Clear all" forget LOCAL entries
// only (house policy: no deletion) — EXCEPT for a produced STUDIO clip (genKind
// "studio_i2v"), which is the one library-family item with real server identity
// (a job_id in the durable clips catalog). For those, Remove/Clear-all ARCHIVE it
// server side too (POST studioClipArchiveUrl — never-delete: the clip's bytes and
// bus row are untouched, only hidden from the catalog list, reversible via the
// matching unarchive route) — otherwise the clip's own list poll (studioShared.tsx
// useStudioClips, ~6s) re-adds it to this library on the very next tick, which is
// exactly the reported "removed clips just reappear" bug. Everything else (frames,
// crops, uploads, generate_image/scene/movie outputs) has no durable per-item
// catalog to resurrect FROM, so those stay a clean local forget. Only the server
// MediaRef is retained — pixels are never persisted — so "assess" opens the live
// server asset (valid for the session); Continue/Replicate work off the retained
// prompt+model regardless.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  useMediaLibrary,
  removeFromLibrary,
  clearLibrary,
  type LibraryItem,
  type LibraryGenKind,
} from "../video/mediaLibrary";
import {
  useJobTracker,
  resumeJob,
  cancelJob,
  isJobActive,
  isJobTerminal,
  dismissAllTerminal,
  type TrackedJob,
  type StationId,
} from "../video/jobTracker";
import { STILL_RUNNING_MESSAGE } from "./pollCaps";
import { requestAddPart, requestStage } from "../video/composerBridge";
import { requestStudioStage } from "../video/studioBridge";
import { useWorkerCalls, type WorkerCall } from "../video/useWorkerCalls";
import { useSidebarRegistry } from "./SidebarRegistry";
import { WorkbenchDrawer } from "./WorkbenchDrawer";
import { SharePanel } from "./SharePanel";
import {
  mediaBytesUrl,
  studioClipArchiveUrl,
  llmJobCancelUrl,
} from "../config";
import { request, errorOf, describeAppError } from "../transport/client";
import type { MediaRef } from "../video/contract";

/** Which sidebar tab is showing. Round 9: "settings" is guarded — it only appears
 *  when the mounted station has registered knobs into the sidebar registry. */
type SideTab = "processes" | "library" | "settings";

// Human station names for the process rows (mirrors the registry titles).
const STATION_NAME: Record<StationId, string> = {
  "image-crop": "Image Crop",
  "video-crop": "Video Crop",
  "audio-crop": "Audio Crop",
  frames: "Frames & Models",
  generate: "Generate",
  "generate-tester": "Generate (tester)",
};

// The route a process row links to. It is the station id by default (id === route
// path for the routed stations), but "video-crop" has NO route of its own — a video
// crop is an INLINE op on a studio item (Studio slice 2b), so its process row links
// to the studio workbench where that item lives instead of falling through to the
// default studio sub-tab.
const STATION_ROUTE: Partial<Record<StationId, string>> = {
  "video-crop": "studio-clips",
  // The tester is a surface INSIDE the Generate station, not a route of its own —
  // without this the default (id === path) would link a tester row to /generate-tester,
  // which resolves to nothing.
  "generate-tester": "generate",
};

// mm:ss elapsed since enqueue.
function fmtElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// One active/recent process row — the always-visible "memory of processes". It
// links to its station sub-tab; a capped row exposes the verbatim still-running
// copy + a Check-again button, and an active row exposes a Cancel button — both
// outside the link so they don't also navigate.
function ProcessRow({ job, now }: { job: TrackedJob; now: number }) {
  const statusKey = job.expired ? "failed" : job.status ?? "queued";
  const statusText = job.expired ? "unknown" : job.status ?? "queued";
  const active = isJobActive(job);
  return (
    <li className="vi-proc-item">
      <NavLink to={`/${STATION_ROUTE[job.station] ?? job.station}`} className="vi-proc-link">
        <div className="vi-proc-top">
          {active ? (
            <span className="vi-proc-spinner" aria-hidden />
          ) : (
            <span className="vi-proc-dot" aria-hidden />
          )}
          <span className="vi-proc-station">{STATION_NAME[job.station]}</span>
          <span className={`vi-status vi-status-${statusKey}`}>{statusText}</span>
          {active && job.progress && (
            <span className="vi-proc-progress">
              image {job.progress.done}/{job.progress.total}
            </span>
          )}
          <span className="vi-proc-elapsed">{fmtElapsed(now - job.enqueuedAt)}</span>
        </div>
        <span className="vi-proc-label">{job.label}</span>
      </NavLink>
      {active && (
        <div className="vi-proc-msg">
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-proc-btn"
            disabled={job.status === "cancelling"}
            onClick={() => cancelJob(job.jobId)}
            title="Stop this job — a frame already in flight finishes first"
          >
            {job.status === "cancelling" ? "Cancelling…" : "✕ Cancel"}
          </button>
        </div>
      )}
      {job.capped && (
        <div className="vi-proc-msg">
          <span>{STILL_RUNNING_MESSAGE}</span>
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-proc-btn"
            onClick={() => resumeJob(job.jobId)}
          >
            Check again
          </button>
        </div>
      )}
      {job.status === "failed" && job.error && (
        <span className="vi-proc-msg vi-proc-err">{job.error}</span>
      )}
      {job.expired && (
        <span className="vi-proc-msg">
          The server no longer tracks this job — it may have expired.
        </span>
      )}
    </li>
  );
}

// Friendly names for the unified /llm/jobs feed's known media kinds; anything
// unmapped falls through to its raw kind (never blank).
const WORKER_CALL_KIND_LABEL: Record<string, string> = {
  identity_reconstruction: "Identity render",
  studio_i2v: "Studio clip",
  generate_studio_movie: "Studio movie",
  generate_scene: "Scene",
  generate_movie: "Movie",
  generate_image: "Image",
};
function humanizeWorkerKind(kind: string): string {
  return WORKER_CALL_KIND_LABEL[kind] ?? (kind || "Worker call");
}

// A render is treated as STALLED (frozen, not merely slow) once it has gone this
// long with no observed movement — even if the server hasn't flagged it. This is
// the honest-status window: past it the row drops its live/green look for an amber
// stalled state, so a wedged render can never keep reading as active.
const WORKER_CALL_STALL_MS = 90_000;

// "12s" / "3m 04s" since some past moment — the expanded panel's relative
// last-movement clock (sibling of fmtElapsed, which counts UP from start).
function fmtAgo(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  return `${m}m ${(total % 60).toString().padStart(2, "0")}s`;
}

// The scrolling log tail in the expanded panel: a monospace <pre> capped by CSS
// max-height with its own scroll, pinned to the BOTTOM (newest line) whenever the
// tail changes — so a live render's newest output is always in view without the
// operator chasing it. Empty/absent tail degrades to a quiet "no log yet".
function WorkerCallLog({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLPreElement | null>(null);
  // Re-pin only when the content actually changes (count + newest line), not on
  // every parent re-render — otherwise a manual scroll-up would be yanked back
  // down every poll tick.
  const key = lines.length ? `${lines.length}:${lines[lines.length - 1]}` : "";
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [key]);
  if (lines.length === 0) {
    return <p className="vi-proc-log-empty">No log yet.</p>;
  }
  return (
    <pre className="vi-proc-log" ref={ref} aria-label="Recent log output">
      {lines.join("\n")}
    </pre>
  );
}

// One in-flight WORKER CALL row (from the fleet-wide /llm/jobs feed) — the calls
// this browser session did NOT launch (identity reconstruction, another station's
// render, another transport). Presentational sibling of ProcessRow: no NavLink
// (these jobs have no station route here), a unified hard-Cancel, and — unlike the
// old flat row that read green-active even when wedged — an EXPANDABLE panel with a
// live progress bar, current stage, an honest stalled state, and a scrolling log.
function WorkerCallRow({
  call,
  now,
  cancelling,
  expanded,
  onToggle,
  onCancel,
}: {
  call: WorkerCall;
  now: number;
  cancelling: boolean;
  expanded: boolean;
  onToggle: (id: string) => void;
  onCancel: (call: WorkerCall) => void;
}) {
  const meta = [call.model, call.worker].filter(Boolean).join(" · ");
  // Honest stalled signal: the server said so, OR movement has been silent past the
  // stall window. Recomputed off `now` (the ~1s panel tick), so a live render that
  // freezes flips to stalled on its own without a status change.
  const lastMoveMs = call.updatedAt != null ? Math.max(0, now - call.updatedAt) : null;
  const stalled =
    call.stalled || (lastMoveMs != null && lastMoveMs > WORKER_CALL_STALL_MS);
  const pct = call.progress != null ? Math.round(call.progress * 100) : null;
  // A requested-and-acknowledged cancel also latches the button into "cancelling…"
  // (survives across polls until the row drops), not just the local in-flight click.
  const cancelBusy = cancelling || call.cancelRequested;
  return (
    <li className="vi-proc-item vi-proc-item-worker">
      <div className="vi-proc-top">
        <button
          type="button"
          className="vi-proc-toggle"
          aria-expanded={expanded}
          aria-label={expanded ? "Collapse details" : "Expand — progress, stage, log"}
          onClick={() => onToggle(call.id)}
        >
          <span className="vi-proc-caret" aria-hidden>
            {expanded ? "▾" : "▸"}
          </span>
        </button>
        {stalled ? (
          <span className="vi-proc-stall-dot" aria-hidden />
        ) : (
          <span className="vi-proc-spinner" aria-hidden />
        )}
        <span className="vi-proc-station">{humanizeWorkerKind(call.kind)}</span>
        <span className={`vi-status vi-status-${call.status}`}>{call.status}</span>
        {stalled && <span className="vi-proc-stall-badge">stalled</span>}
        {pct != null && <span className="vi-proc-progress">{pct}%</span>}
        <span className="vi-proc-elapsed">{fmtElapsed(now - call.startedAt)}</span>
      </div>
      {meta && <span className="vi-proc-label">{meta}</span>}

      {expanded && (
        <div className="vi-proc-detail">
          {/* Progress bar: determinate when the feed gives a fraction, otherwise an
              indeterminate "working" sweep — never a fabricated percentage. Goes
              amber when stalled so a frozen bar doesn't read as healthy progress. */}
          <div className="vi-proc-progbar-row">
            <div
              className={`vi-proc-progbar${
                pct == null ? " vi-proc-progbar-indeterminate" : ""
              }${stalled ? " vi-proc-progbar-stalled" : ""}`}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              {...(pct != null ? { "aria-valuenow": pct } : {})}
            >
              <span
                className="vi-proc-progbar-fill"
                style={pct != null ? { width: `${pct}%` } : undefined}
              />
            </div>
            <span className="vi-proc-stage">
              {call.stage ? call.stage : pct == null ? "working…" : `${pct}%`}
            </span>
          </div>

          {call.message && <p className="vi-proc-detail-msg">{call.message}</p>}

          <div className="vi-proc-detail-meta">
            <span
              className={`vi-proc-lastmove${
                stalled ? " vi-proc-lastmove-stalled" : ""
              }`}
            >
              {lastMoveMs != null
                ? `last movement ${fmtAgo(lastMoveMs)} ago`
                : "last movement unknown"}
            </span>
          </div>

          <WorkerCallLog lines={call.logTail} />
        </div>
      )}

      <div className="vi-proc-msg">
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-proc-btn"
          disabled={cancelBusy}
          onClick={() => onCancel(call)}
          title="Stop this worker call — a step already in flight finishes first"
        >
          {cancelBusy ? "Cancelling…" : "✕ Cancel"}
        </button>
      </div>
    </li>
  );
}

// STUDIO SCOPE (operator, 2026-07-31): the studio workbench's Active Processes must
// list ONLY studio-originated work — NEVER a general fleet model call (a discord chat,
// a web /v1 completion, another transport's LLM call). Those must not be visible here
// and must not be cancellable from inside the studio. The fleet-wide /llm/jobs feed
// carries every transport, so we keep only rows that are RENDER/MEDIA work:
//   * transport === "media"  — the media/render bus (the authoritative signal: every
//     job on it is video-arm work — studio clips/movies, scene/image gen, identity
//     reconstruction/mesh/char360, mlt render), OR
//   * a known render/media job KIND (defensive fallback if a row's transport is blank).
// A general LLM call (transport web/v1/discord/cli, kind chat/completion/…) is dropped.
// The console-wide ActiveProcessesStation still shows the fleet superset — this scoping
// is the studio sidebar's alone.
const STUDIO_MEDIA_KINDS = new Set([
  "studio_i2v",
  "generate_studio_movie",
  "generate_scene",
  "generate_movie",
  "generate_image",
  "identity_reconstruction",
  "identity_mesh_build",
  "identity_video_extract",
  "mlt_render",
  "crop",
  "frame_extract",
  "audio_extract",
]);

function isStudioOriginated(c: WorkerCall): boolean {
  return c.transport === "media" || STUDIO_MEDIA_KINDS.has(c.kind);
}

/** Non-done tracked jobs, newest first — the Active Processes tab body. */
function activeJobsOf(records: TrackedJob[]): TrackedJob[] {
  return records
    .filter((r) => r.status !== "done")
    .slice()
    .sort((a, b) => b.enqueuedAt - a.enqueuedAt);
}

// The Active Processes tab body: every non-done tracked job, newest first. Done
// rows drop off once their artifact is in the library tab. Ticks ~1s so elapsed
// stays live while anything is running or paused-capped.
function ProcessesPanel() {
  const reg = useSidebarRegistry();
  const records = useJobTracker();
  const jobs = useMemo(() => activeJobsOf(records), [records]);

  // The fleet-wide in-flight worker calls (identity reconstruction, other stations'
  // renders, calls from any transport) — the work THIS browser session never
  // launched, so the session jobTracker can't see it. Merged below, DEDUPED against
  // the tracker: a /llm/jobs row whose id equals a TrackedJob.jobId is dropped (the
  // tracker row is richer — station link, progress, provenance — so it wins).
  const { calls, refetch } = useWorkerCalls();
  const trackerIds = useMemo(() => new Set(jobs.map((j) => j.jobId)), [jobs]);
  const workerCalls = useMemo(
    // Deduped against the session tracker AND scoped to studio-originated (media/
    // render) work only — a general fleet model call never surfaces or cancels here.
    () => calls.filter((c) => !trackerIds.has(c.id) && isStudioOriginated(c)),
    [calls, trackerIds],
  );

  // Terminal rows (failed / cancelled / expired) linger in this list after the
  // active ones drop off — this drives the opt-in "Clear finished" broom below.
  const hasTerminal = useMemo(() => jobs.some(isJobTerminal), [jobs]);

  // Per-id busy flag for the worker-call Cancel buttons (immediate feedback; the
  // next poll / the forced refetch drops the row once the server observes it).
  const [cancelling, setCancelling] = useState<Set<string>>(new Set());
  const onCancelWorkerCall = useCallback(
    async (call: WorkerCall) => {
      setCancelling((prev) => new Set(prev).add(call.id));
      // ONE unified hard-cancel for every transport: the feed's own cancel route
      // (POST /llm/jobs/<id>/cancel → {cancelled,status}) fans the stop out to the
      // right bus server-side, so the row no longer has to route media vs chat itself.
      await request<unknown>(llmJobCancelUrl(call.id), {
        method: "POST",
        meta: { specKey: "workers", operation: "worker.call.cancel" },
      });
      refetch();
      setCancelling((prev) => {
        const n = new Set(prev);
        n.delete(call.id);
        return n;
      });
    },
    [refetch],
  );

  // Per-id expanded state for the worker-call rows (progress / stage / stalled /
  // log). Keyed by job id so a row's panel stays open across poll ticks; a job that
  // finishes and drops from the feed just leaves a stale id in the set (harmless).
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggleExpanded = useCallback((id: string) => {
    setExpanded((prev) => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  }, []);

  // Tick ~1s while anything is live so every elapsed clock stays current: tracker
  // actives/capped OR any in-flight worker call (those are always live).
  const ticking =
    jobs.some((r) => isJobActive(r) || r.capped) || workerCalls.length > 0;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!ticking) {
      setNow(Date.now());
      return;
    }
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [ticking]);

  // Round 9: the mounted station (e.g. STUDIO — whose jobs are not in the jobTracker)
  // portals its own in-flight rows into this host, so the ONE Active Processes view
  // shows tracker jobs AND studio renders. The host is ALWAYS rendered so the portal
  // target is stable; the empty copy is suppressed when a station has portaled rows in.
  const extraHost = <div className="vi-proc-extra" ref={reg.setActiveExtraHost} />;

  const total = jobs.length + workerCalls.length;

  if (total === 0) {
    return (
      <>
        {!reg.activeExtraActive && (
          <p className="vi-lib-empty">
            No active processes — jobs you run appear here with live status until they
            finish, then their outputs land in the Session Library.
          </p>
        )}
        {extraHost}
      </>
    );
  }

  return (
    <>
      <div className="vi-proc-header">
        <h2 className="vi-proc-title">Active processes</h2>
        <span className="vi-proc-count">{total}</span>
        {/* Only shown when there's a finished/failed row to clear; forgets the
            LOCAL terminal rows only (never server work), and leaves every
            still-running job polling and visible. */}
        {hasTerminal && (
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            onClick={() => dismissAllTerminal()}
            title="Remove finished and failed items from this view — your saved files on disk are kept."
            aria-label="Remove finished and failed jobs from this list — still-running jobs stay, and your saved files on disk are kept."
          >
            Clear finished
          </button>
        )}
      </div>
      <ul className="vi-proc-list">
        {jobs.map((job) => (
          <ProcessRow key={job.jobId} job={job} now={now} />
        ))}
        {workerCalls.map((call) => (
          <WorkerCallRow
            key={call.id}
            call={call}
            now={now}
            cancelling={cancelling.has(call.id)}
            expanded={expanded.has(call.id)}
            onToggle={toggleExpanded}
            onCancel={onCancelWorkerCall}
          />
        ))}
      </ul>
      {extraHost}
    </>
  );
}

// The kind-driven media preview, shared verbatim by the library rows AND the
// collapsed group header (so the two thumbnail renderers can never drift apart):
// images get a lazy thumbnail, video gets a poster-like frame still (a muted,
// metadata-only <video> seeked to an early frame — there is no server-side poster
// to lean on; /video/media is Range-aware so only the seeked byte range is
// fetched), audio gets an inline player, and anything else (unknown) gets a cheap
// placeholder tile. `className` names the image/video tile wrapper so a compact
// caller (the group header, .vi-lib-group-thumb) can size it smaller while the
// rows keep the full-width .vi-lib-thumb. NOTE: the prop is `media` (not `ref`) —
// `ref` is a reserved prop name on React 18 function components and would not be
// forwarded as a normal prop.
function RefThumb({
  media,
  name,
  className = "vi-lib-thumb",
}: {
  media: MediaRef;
  name: string;
  className?: string;
}) {
  const kind = media.kind;
  if (kind === "image") {
    return (
      <div className={className}>
        <img src={mediaBytesUrl(media.uri)} loading="lazy" alt={name} />
      </div>
    );
  }
  if (kind === "video") {
    return (
      <div className={className}>
        <video
          src={`${mediaBytesUrl(media.uri)}#t=0.1`}
          muted
          playsInline
          preload="metadata"
          aria-label={name}
        />
      </div>
    );
  }
  if (kind === "audio") {
    return (
      <audio
        className="vi-audio-player"
        controls
        preload="none"
        src={mediaBytesUrl(media.uri)}
      />
    );
  }
  return (
    <div className="vi-lib-placeholder" aria-hidden>
      <span>{kind}</span>
    </div>
  );
}

// A single library row. Kind drives the preview: images get a lazy thumbnail,
// video gets a poster-like frame still (a muted metadata-only <video> seeked to an
// early frame — there is no server-side thumbnail to lean on), audio gets an inline
// player, and anything else (unknown) gets a cheap placeholder tile.
function LibraryRow({ it, onGenerate }: { it: LibraryItem; onGenerate: boolean }) {
  const navigate = useNavigate();
  const { ref } = it;
  // label wins, then origin, then a short asset-id stub (|| so blanks fall through).
  const name = it.label || it.origin || ref.asset_id.slice(0, 8);
  const kind = ref.kind;
  const canGenerate = kind === "image" || kind === "video";
  const canAssess = kind === "image" || kind === "video";
  // The B2 movie->studio chain: any VIDEO ref (a movie/scene mp4, or a produced
  // studio clip) can seed a studio i2v render that EXTENDS it from its last frame.
  const canSendToStudio = kind === "video";
  // A produced STUDIO clip is the one library item with real SERVER identity —
  // its job_id rides along as `groupId` (see studioShared.tsx's addToLibrary
  // call). Remove must archive that job server-side (see this file's header
  // note) instead of only forgetting the local entry, or the clip list's ~6s
  // poll re-adds it on the very next tick.
  const studioJobId = it.genKind === "studio_i2v" ? it.groupId : undefined;

  const dims =
    ref.width != null && ref.height != null ? `${ref.width}×${ref.height}` : null;
  const dur = ref.duration_s != null ? `${Math.round(ref.duration_s)}s` : null;

  // Optimistic remove for the studio-clip path: the row hides IMMEDIATELY (before
  // the archive call resolves) and only forgets the local entry once the server
  // confirms it archived — so a failed/slow archive un-hides the row with the
  // error shown, instead of the item silently vanishing while the server still
  // lists it (which would just re-trigger the same resurrection bug one poll
  // later). Non-studio items skip all of this — removeFromLibrary is a same-tick,
  // can't-fail local operation for them, exactly as before.
  const [hidden, setHidden] = useState(false);
  const [removeError, setRemoveError] = useState<string | null>(null);
  // In-place ("▸ Play here") playback for VIDEO rows — the same function as the
  // studio viewer's Play here (StudioViewer.tsx). When true, the poster still is
  // swapped for a real, audible, autoplaying inline <video> right here in the row
  // (no new browser tab). Local to the row: each row plays independently by design.
  const [playing, setPlaying] = useState(false);

  async function handleRemove() {
    if (!studioJobId) {
      removeFromLibrary(ref.uri);
      return;
    }
    setRemoveError(null);
    setHidden(true);
    const res = await request<unknown>(studioClipArchiveUrl(studioJobId), {
      method: "POST",
      meta: { specKey: "studio", operation: "studio.clip.archive" },
    });
    if (!res.ok) {
      setHidden(false);
      setRemoveError(describeAppError(errorOf(res)));
      return;
    }
    // Archived server-side (or already was — archive is idempotent): the clip is
    // gone from GET .../clips for good, so forgetting the local entry now is
    // permanent — no future poll can ever re-add it.
    removeFromLibrary(ref.uri);
  }

  if (hidden) return null;

  return (
    <li className="vi-lib-item">
      <div className="vi-lib-top">
        <span className="vi-lib-badge">{kind}</span>
        <span className="vi-lib-name" title={ref.uri}>
          {name}
        </span>
      </div>

      {/* Preview extracted into <RefThumb> (shared with the group header). Movie /
          scene / studio clips are video/mp4 and render a poster-like #t=0.1 still.
          When a VIDEO row is "▸ Play here"-playing, swap that still for a real
          inline player (audible, autoplaying) in the SAME slot + sizing wrapper
          (.vi-lib-thumb) so the row doesn't jump — no #t=0.1, not muted, so it
          actually plays with audio. */}
      {kind === "video" && playing ? (
        <div className="vi-lib-thumb">
          <video
            src={mediaBytesUrl(ref.uri)}
            controls
            autoPlay
            playsInline
            aria-label={name}
          />
        </div>
      ) : (
        <RefThumb media={ref} name={name} />
      )}

      <div className="vi-lib-meta">
        {kind === "image" && dims ? <span>{dims}</span> : null}
        {(kind === "audio" || kind === "video") && dur ? <span>{dur}</span> : null}
        <code>{ref.mime}</code>
      </div>

      <div className="vi-lib-actions">
        {kind === "video" && (
          <button
            type="button"
            className={`vi-btn vi-btn-sm${playing ? "" : " vi-btn-accent"}`}
            onClick={() => setPlaying((v) => !v)}
            title={
              playing
                ? "Stop inline playback"
                : "Play this clip here, in place (same as the studio viewer's Play here) — no new tab"
            }
          >
            {playing ? "■ Stop" : "▸ Play here"}
          </button>
        )}
        {canGenerate ? (
          onGenerate ? (
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-accent"
              onClick={() =>
                requestAddPart({
                  ref,
                  origin: it.origin,
                  ...(it.label != null ? { label: it.label } : {}),
                })
              }
            >
              Add to prompt
            </button>
          ) : (
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-accent"
              onClick={() => {
                // Stage the part FIRST — composerBridge BUFFERS it (Generate isn't
                // mounted yet) — THEN navigate. GenerateStation drains the buffer on
                // mount, so the item actually lands in the prompt instead of just
                // being "available in the picker" (which read as doing nothing). Same
                // effect as "Add to prompt", with the route hop.
                requestAddPart({
                  ref,
                  origin: it.origin,
                  ...(it.label != null ? { label: it.label } : {}),
                });
                navigate("/generate");
              }}
              title="Add this to the Generate prompt and open the Generate station."
            >
              Use in Generate
            </button>
          )
        ) : null}
        {canSendToStudio && (
          <>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-accent"
              onClick={() => {
                requestStudioStage({
                  sourceVideo: ref,
                  mode: "extend",
                  provenance: {
                    origin: it.origin,
                    prompt: it.prompt,
                    model: it.model,
                    genKind: it.genKind,
                    parentUri: ref.uri,
                  },
                });
                navigate("/studio-clips");
              }}
              title="Open Studio Clips staged to EXTEND this clip from its last frame (i2v, movie → studio chain)."
            >
              Send to Studio
            </button>
            <button
              type="button"
              className="vi-btn vi-btn-sm"
              onClick={() => {
                requestStudioStage({
                  sourceVideo: ref,
                  mode: "restyle",
                  provenance: {
                    origin: it.origin,
                    prompt: it.prompt,
                    model: it.model,
                    genKind: it.genKind,
                    parentUri: ref.uri,
                  },
                });
                navigate("/studio-clips");
              }}
              title="Open Studio Clips staged to RESTYLE this clip with a prompt (v2v · Wan VACE, 480p landscape)."
            >
              Restyle
            </button>
          </>
        )}
        {canAssess && (
          <a
            className="vi-btn vi-btn-sm vi-btn-ghost"
            href={mediaBytesUrl(ref.uri)}
            target="_blank"
            rel="noreferrer"
            title="Open this output at full resolution in a new tab (read-only)."
          >
            assess
          </a>
        )}
        <button
          type="button"
          className="vi-btn vi-btn-sm"
          onClick={() => void handleRemove()}
          title={
            studioJobId
              ? "Archive this clip on the server (never deleted, reversible) and remove it here — it won't come back."
              : "Remove from the library for good — tombstoned so it can't repopulate. The server asset is untouched (not deleted)."
          }
          aria-label={`Remove ${name} from the session library for good${
            studioJobId ? " (archives the clip on the server)" : " (won't repopulate)"
          }`}
        >
          Remove
        </button>
      </div>
      {removeError && (
        <p className="vi-lib-error" role="alert">
          Couldn't archive — {removeError} (row restored, try again)
        </p>
      )}
    </li>
  );
}

/** A WORK — a component/lineage whose iterations live inside it (t24). Everything
 *  in the library is a work now: a generation RUN (keyed by its producing job id,
 *  carrying prompt+model+genKind, so Continue/Replicate apply), OR a non-gen
 *  lineage (frames / crops / uploads / demo) keyed by ORIGIN. Uniform shape — a
 *  one-iteration work renders as a section exactly like a many-iteration one. */
interface WorkGroup {
  /** Stable key: the producing groupId when present, else a synthetic origin key. */
  key: string;
  /** True for a generation run (has groupId + genKind) — enables Continue/Replicate. */
  isGen: boolean;
  genKind: LibraryGenKind | null;
  prompt: string;
  model: string;
  origin: string;
  items: LibraryItem[];
  newestAt: number;
}

// Regroup the WHOLE library into work sections (t24 — invert the old
// groups-then-flat split). GROUPING KEY, strongest natural key present with NO
// data-shape change:
//   • a generation run  → `groupId` (the producing job id; already carried), so
//     each run is its own work with its prompt/model.
//   • everything else    → `origin` (frames / crops / uploads / demo). Non-gen
//     items carry no per-run id on the wire, so origin is the strongest lineage
//     key present — all frames form the "frames" work, all uploads the "uploads"
//     work, etc. (A per-extraction key would need a data-shape change, which this
//     slice forbids.) Prefixed "origin:" so it can never collide with a groupId.
// Both levels are freshest-first: works by their newest iteration, iterations
// within a work newest-first too.
function groupLibrary(items: LibraryItem[]): WorkGroup[] {
  const map = new Map<string, WorkGroup>();
  for (const it of items) {
    const isGen = !!(it.groupId && it.genKind);
    const key = isGen ? (it.groupId as string) : `origin:${it.origin || "other"}`;
    const g = map.get(key);
    if (g) {
      g.items.push(it);
      g.newestAt = Math.max(g.newestAt, it.addedAt);
      // A gen run keeps the provenance from its first-seen item; a non-gen work
      // has none. (All items under a groupId share prompt/model by construction.)
    } else {
      map.set(key, {
        key,
        isGen,
        genKind: isGen ? (it.genKind as LibraryGenKind) : null,
        prompt: it.prompt ?? "",
        model: it.model ?? "",
        origin: it.origin || "other",
        items: [it],
        newestAt: it.addedAt,
      });
    }
  }
  const works = [...map.values()].sort((a, b) => b.newestAt - a.newestAt);
  // Iterations newest-first within each work (both levels freshest-first).
  for (const w of works) w.items.sort((a, b) => b.addedAt - a.addedAt);
  return works;
}

// A human title for a non-gen work (the origin lineage). Gen works title
// themselves from prompt/model in the section header instead.
function originTitle(origin: string): string {
  switch (origin) {
    case "frames": return "Extracted frames";
    case "upload": return "Uploads";
    case "studio": return "Studio clips";
    case "demo": return "Demo assets";
    default: return origin ? origin[0].toUpperCase() + origin.slice(1) : "Other";
  }
}

// One WORK section (t24) — a collapsed-by-default row showing a representative
// summary (poster + title/prompt + iteration count), expandable to reveal THAT
// work's iterations (LibraryRow reused verbatim). Uniform for gen runs and non-gen
// lineages: a gen run adds the Continue/Replicate composer affordances + prompt/
// model header; a non-gen work (frames/uploads/…) shows its origin title. `open`
// is CONTROLLED by the panel (a keyed Set that survives the refresh cadence).
function WorkSection({
  group,
  open,
  onToggle,
  onGenerate,
}: {
  group: WorkGroup;
  open: boolean;
  onToggle: () => void;
  onGenerate: boolean;
}) {
  const navigate = useNavigate();
  const isScene = group.genKind === "generate_scene";
  // A studio i2v run: its output is a produced CLIP. The Generate-composer
  // affordances (Continue/Replicate) don't apply — the clip's chain affordance is
  // "Send to Studio", which lives on each video row (LibraryRow) below.
  const isStudio = group.genKind === "studio_i2v";
  const badge = group.isGen
    ? (isStudio ? "clip" : isScene ? "scene" : "image")
    : originTitle(group.origin).toLowerCase();
  // The header text: a gen run shows its prompt; a non-gen work shows its origin
  // title. Freshest-first means items[0] is the newest iteration.
  const headerText = group.isGen
    ? group.prompt.trim() || "(no text prompt)"
    : originTitle(group.origin);
  // Representative poster: the FRESHEST iteration that can render a still (image
  // or video) — items are newest-first, so the first renderable is the latest.
  // Reuses the same <RefThumb> the expanded rows use, so collapsed poster and row
  // still can never diverge.
  const thumb = group.items.find(
    (i) => i.ref.kind === "image" || i.ref.kind === "video",
  );
  const thumbName = thumb
    ? thumb.label || thumb.origin || thumb.ref.asset_id.slice(0, 8)
    : "";

  // Both affordances funnel through the SAME composer seam the Run path reads,
  // then navigate so the (possibly unmounted) Generate station mounts and drains
  // the staged request.
  function replicate() {
    requestStage({
      replace: true,
      mode: isScene ? "scene" : "image",
      ...(group.model ? { model: group.model } : {}),
      ...(group.prompt ? { text: group.prompt } : {}),
    });
    navigate("/generate");
  }
  function continueScene() {
    // Iterations are newest-first now; the "last frame" to continue from is the
    // FRESHEST image (items[0..]), so scan from the front for the first image.
    const last = group.items.find((i) => i.ref.kind === "image");
    requestStage({
      replace: true,
      mode: "scene",
      ...(group.model ? { model: group.model } : {}),
      ...(group.prompt ? { text: group.prompt } : {}),
      ...(last
        ? {
            parts: [
              { ref: last.ref, origin: "generate", label: "continue: last frame" },
            ],
          }
        : {}),
    });
    navigate("/generate");
  }

  return (
    <li className="vi-lib-group">
      <button
        type="button"
        className="vi-lib-group-toggle"
        aria-expanded={open}
        onClick={onToggle}
      >
        <span className="vi-lib-group-caret" aria-hidden>
          {open ? "▾" : "▸"}
        </span>
        <span className="vi-lib-badge">{badge}</span>
        {group.isGen && (
          <span className="vi-lib-group-model" title={`Model: ${group.model || "—"}`}>
            {group.model || "—"}
          </span>
        )}
        <span className="vi-lib-count">{group.items.length}</span>
      </button>

      {/* Poster + title/prompt read as the work's header: the representative thumb
          sits to the LEFT. The thumb lives OUTSIDE the toggle <button> so the
          expander's a11y/hit-target is untouched (no media nested in it). */}
      <div className="vi-lib-group-head">
        {thumb && (
          <RefThumb
            media={thumb.ref}
            name={thumbName}
            className="vi-lib-group-thumb"
          />
        )}
        <p className="vi-lib-group-prompt" title={headerText}>
          {headerText}
        </p>
      </div>

      {group.isGen && !isStudio && (
        <div className="vi-lib-actions">
          {isScene && (
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-accent"
              onClick={continueScene}
              title="Open Generate in Scene mode, staged from this scene's last frame to extend it."
            >
              Continue frames
            </button>
          )}
          <button
            type="button"
            className="vi-btn vi-btn-sm"
            onClick={replicate}
            title="Open Generate with this run's prompt + model staged."
          >
            Replicate prompt
          </button>
        </div>
      )}

      {open && (
        <ul className="vi-lib-list vi-lib-group-list">
          {group.items.map((it) => (
            <LibraryRow key={it.ref.uri} it={it} onGenerate={onGenerate} />
          ))}
        </ul>
      )}
    </li>
  );
}

// The Session Library tab body (t24): every entry is a WORK section — a collapsed
// row you expand to see that work's iterations. No flat rows; a one-iteration work
// renders as a section like any other (uniformity beats cleverness). Sections are
// collapsed by default (the library shows works, not churn) and freshest-first.
function LibraryPanel({
  items,
  onGenerate,
}: {
  items: LibraryItem[];
  onGenerate: boolean;
}) {
  // TYPE SECTIONS (operator 2026-08-13): the library splits into Images and
  // Videos (plus a rare Other for audio/unknown), each holding its work groups
  // newest-first — groupLibrary already sorts both levels freshest-first, so
  // the per-type ordering is "most recent on top" by construction.
  const sections = useMemo(() => {
    const imgs = items.filter((it) => it.ref.kind === "image");
    const vids = items.filter((it) => it.ref.kind === "video");
    const rest = items.filter((it) => it.ref.kind !== "image" && it.ref.kind !== "video");
    return [
      { id: "images", label: "🖼 Images", count: imgs.length, works: groupLibrary(imgs) },
      { id: "videos", label: "🎞 Videos", count: vids.length, works: groupLibrary(vids) },
      { id: "other", label: "📎 Other", count: rest.length, works: groupLibrary(rest) },
    ].filter((sec) => sec.count > 0);
  }, [items]);
  // Both type sections start OPEN (hiding freshly-landed work behind a closed
  // header is how "it never populates" reports happen); the caret remembers.
  const [closedTypes, setClosedTypes] = useState<Set<string>>(() => new Set());
  const toggleType = (id: string) =>
    setClosedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  // Expand state keyed by work key — the established discipline (a Set that
  // survives the library's refresh cadence: the works array is rebuilt on every
  // change, but the keys are stable, so an expanded work stays open across a
  // refresh). Default posture is COLLAPSED (empty set).
  const [openKeys, setOpenKeys] = useState<Set<string>>(() => new Set());
  const toggle = (key: string) =>
    setOpenKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  if (items.length === 0) {
    return (
      <p className="vi-lib-empty">
        No clips yet — produced crops, frames, and generations land here.
      </p>
    );
  }

  return (
    <>
      {sections.map((sec) => {
        const isOpen = !closedTypes.has(sec.id);
        return (
          <div key={sec.id} className="vi-lib-typegroup">
            <button
              type="button"
              className="vi-lib-group-toggle vi-lib-typegroup-head"
              aria-expanded={isOpen}
              onClick={() => toggleType(sec.id)}
            >
              <span className="vi-lib-group-caret" aria-hidden>
                {isOpen ? "▾" : "▸"}
              </span>{" "}
              {sec.label} <span className="vi-proc-count">{sec.count}</span>
            </button>
            {isOpen && (
              <ul className="vi-lib-list">
                {sec.works.map((w) => (
                  <WorkSection
                    key={`${sec.id}:${w.key}`}
                    group={w}
                    // Expand state is NAMESPACED by type section: origin-keyed works
                    // (uploads/frames/legacy history) can carry the SAME work key in
                    // both Images and Videos, and a shared key would open both.
                    open={openKeys.has(`${sec.id}:${w.key}`)}
                    onToggle={() => toggle(`${sec.id}:${w.key}`)}
                    onGenerate={onGenerate}
                  />
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </>
  );
}

export function WorkbenchSidebar() {
  const reg = useSidebarRegistry();
  const library = useMediaLibrary();
  const records = useJobTracker();
  const location = useLocation();
  const [tab, setTab] = useState<SideTab>("processes");

  // Compute the route gate ONCE here (not per row). Same segment convention the
  // shell uses: the first path segment === "generate" means the composer is live.
  const onGenerate =
    location.pathname.replace(/^\/+/, "").split("/")[0] === "generate";

  const activeCount = useMemo(() => activeJobsOf(records).length, [records]);
  const libraryCount = library.length;

  // "Clear all" wipes the WHOLE session library in one tick (unlike per-row
  // Remove, it has never been optimistic-with-restore, and a bulk operation has
  // no single row to un-hide on a partial failure) — but any STUDIO clip in that
  // list still needs the server-side archive, or it resurrects on the next clip
  // poll exactly like an un-wired single Remove would. Fire-and-forget (mirrors
  // jobTracker.cancelJob's idiom in this same arm): archive every studio job id
  // present, then clear locally immediately. A dropped archive call here just
  // means that one clip reappears on the next poll — recoverable via its own
  // Remove — never a stuck or corrupted library.
  function handleClearAll() {
    const studioJobIds = new Set(
      library
        .filter((it) => it.genKind === "studio_i2v" && it.groupId)
        .map((it) => it.groupId as string),
    );
    for (const jobId of studioJobIds) {
      void request<unknown>(studioClipArchiveUrl(jobId), {
        method: "POST",
        meta: { specKey: "studio", operation: "studio.clip.archive" },
      });
    }
    clearLibrary();
  }

  // Round 9: the guarded Settings tab. If the mounted station un-registers its
  // settings (e.g. you navigate off Studio) while the Settings tab is showing,
  // fall back to Active Processes so the sidebar is never on a vanished tab.
  const { settingsActive } = reg;
  useEffect(() => {
    if (!settingsActive && tab === "settings") setTab("processes");
  }, [settingsActive, tab]);

  return (
    <div className="vi-lib">
      {/* t52: the overlay affordances — always visible above the tab strip
          regardless of which sidebar tab is showing, so they survive tab
          switches exactly like the tabs themselves. This row is just toggles;
          the panels themselves expand into in-page overlays.

          2026-08-06 (operator): "having the 3 options as tab selectable within
          the overlay window would be a huge convenience". Console, ComfyUI and
          Active now share ONE drawer window with a tab strip in its header, so
          switching between them is a click instead of a close-and-reopen (the
          scrim covers this row while a drawer is open, so these buttons could
          never have done it). WorkbenchDrawer renders all three toggles — they
          stay one-per-destination DEEP LINKS into the shared window, matching
          this row's idiom (and Share, which is a separate modal). Each panel's
          own posture is untouched: the console frame is still unsandboxed
          first-party, the ComfyUI frame still sandboxed behind its scheme gate,
          and Active still polls only while its tab is the selected one. See
          WorkbenchDrawer.tsx. */}
      <div className="vi-side-toolbar">
        <WorkbenchDrawer />
        {/* k9: the Share affordance sits beside them. It renders ONLY for a
            console-authenticated session (it probes an operator-gated endpoint),
            so a share-link visitor never sees a working Share button. */}
        <SharePanel />
      </div>

      {/* Tab switcher — Active Processes | Session Library | (Settings). Reuses the
          segmented .vi-gen-mode control idiom, above the tab bodies. Settings only
          appears when a mounted station has registered knobs into it. */}
      <div className="vi-gen-mode vi-side-tabs" role="tablist" aria-label="Sidebar">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "processes"}
          className={`vi-gen-mode-btn${
            tab === "processes" ? " vi-gen-mode-btn-active" : ""
          }`}
          onClick={() => setTab("processes")}
        >
          Active Processes
          {activeCount > 0 && <span className="vi-side-tab-badge">{activeCount}</span>}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "library"}
          className={`vi-gen-mode-btn${
            tab === "library" ? " vi-gen-mode-btn-active" : ""
          }`}
          onClick={() => setTab("library")}
        >
          Session Library
          {libraryCount > 0 && (
            <span className="vi-side-tab-badge">{libraryCount}</span>
          )}
        </button>
        {settingsActive && (
          <button
            type="button"
            role="tab"
            aria-selected={tab === "settings"}
            className={`vi-gen-mode-btn${
              tab === "settings" ? " vi-gen-mode-btn-active" : ""
            }`}
            onClick={() => setTab("settings")}
          >
            Settings
          </button>
        )}
      </div>

      {tab === "processes" && (
        <section className="vi-proc" aria-label="Active processes">
          <ProcessesPanel />
        </section>
      )}
      {tab === "library" && (
        <section aria-label="Session library">
          <div className="vi-lib-header">
            <h2 className="vi-lib-title">Session library</h2>
            <span className="vi-lib-count">{libraryCount}</span>
            {libraryCount > 0 && (
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost"
                onClick={handleClearAll}
                title="Forget every local entry — server media is untouched, except produced studio clips, which are archived (never deleted, reversible)."
              >
                Clear all
              </button>
            )}
          </div>
          <LibraryPanel items={library} onGenerate={onGenerate} />
        </section>
      )}
      {/* Settings tab body — the DOM host the mounted station portals its knobs into.
          Only rendered when the tab is active AND a station is registered; the knob
          STATE lives in the station, so unmounting this host never loses input. */}
      {tab === "settings" && settingsActive && (
        <section
          aria-label="Settings"
          className="vi-side-settings"
          ref={reg.setSettingsHost}
        />
      )}
    </div>
  );
}
