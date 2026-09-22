// CINEMA SESSIONS PANEL — "which of these movies is waiting on me", answered from
// the studio-movies root instead of from a job id that may be long gone.
//
// A movie is ONE bus job that can run for hours; when that job ended (cancelled,
// reaped, or lost with the browser tab) its rendered segments stayed on disk and
// became unreachable — real GPU hours with no way to see or continue them. This panel
// is the handle on those dirs: every session with its status, how far it got, what it
// has ALREADY rendered (each done segment playable inline, as it lands), and the two
// actions — Resume (re-enqueue the persisted spec; the clips are the checkpoint) and
// Pause (cooperative cancel + "parked, not abandoned").
//
// Idioms borrowed rather than reinvented: the row/chip/caret shapes are
// StudioActiveProcesses', the progress bar + latest-event line are the console-wide
// Active Processes renderers (useMediaJobs' `progressLabel`/`latestEvent`), and the
// list hook mirrors useStudioClips. Nothing here fabricates a number: a bar appears
// only when the server sent one, and a disabled Resume always says WHY.
import { useEffect, useState } from "react";
import {
  useMovieSessions,
  segmentClipSrc,
  movieMediaSrc,
  isSessionLive,
  resumeBlockedReason,
  type MovieSession,
  type MovieSegment,
} from "../../video/useMovieSessions";
import {
  useMediaJobs,
  progressLabel,
  latestEvent,
  type MediaJob,
} from "../../video/useMediaJobs";
import { JobRow } from "../ActiveProcessesStation";
import { shortId, agoText, fmtWhen } from "./studioShared";

// Segment statuses that mean "this one is finished" — the rest are pending (dimmed)
// or failed (error shown verbatim).
const SEG_DONE = new Set(["done", "resumed"]);

/**
 * The live pace of ONE in-flight movie job, from the bus feed the studio arm already
 * polls: the folded 0..1 bar plus "segment 2/14 · step 18/32" and the freshest stage
 * line. This is what makes a 14-segment render legible in its first minute instead of
 * only when segment 1 lands.
 *
 * It owns its own `useMediaJobs` subscription and is therefore mounted ONLY while
 * something is actually running (a live session row, or the composer's own render) —
 * no in-flight movie, no poll. Shared by both call sites so there is one renderer for
 * "how fast is this going".
 */
export function MovieLiveProgress({ jobId }: { jobId: string | null }) {
  const { jobs } = useMediaJobs();
  const job = jobId ? jobs.find((j) => j.id === jobId) ?? null : null;
  if (!job) return null;
  const label = progressLabel(job);
  const event = latestEvent(job);
  const pct = job.progress == null ? null : Math.max(0, Math.min(100, job.progress * 100));
  return (
    <>
      {(pct != null || label) && (
        <div className="vi-job-progress">
          {pct != null && (
            <div className="vi-job-progress-track">
              <div className="vi-job-progress-fill" style={{ width: `${pct}%` }} />
            </div>
          )}
          {label ? <span className="vi-job-progress-label">{label}</span> : null}
        </div>
      )}
      {event ? <p className="vi-job-latest">{event}</p> : null}
    </>
  );
}

/** True for a segment that has not started — the spec's not-yet-rendered tail, which
 *  is listed (dimmed) so a 14-segment movie mid-render never reads as a 2-segment one. */
function isPending(seg: MovieSegment): boolean {
  const status = seg.status ?? "pending";
  return !SEG_DONE.has(status) && status !== "failed";
}

// NOTE ON `key`: the row bodies below are CONTENT, not the <li> itself — the list
// element (and its key) stays in the .map. This project's scoped tsconfig rejects a
// `key` prop on a locally-declared component (the same reason StudioSplicedRow is
// rendered once, never in a map), so keeping the key on the intrinsic element is the
// house idiom rather than a workaround invented here.

/** One segment's content: playable when its bytes exist, verbatim when it failed. */
function SegmentRowBody({ seg }: { seg: MovieSegment }) {
  const src = segmentClipSrc(seg);
  const status = seg.status ?? "pending";
  const failed = status === "failed";
  const pending = isPending(seg);
  return (
    <>
      {src ? (
        <video
          src={`${src}#t=0.1`}
          controls
          muted
          playsInline
          preload="metadata"
          style={{
            width: "10rem",
            maxWidth: "34vw",
            borderRadius: "0.4rem",
            background: "#000",
            border: "1px solid var(--vi-border)",
          }}
        />
      ) : null}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.15rem", minWidth: 0 }}>
        <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>
          Segment {seg.index + 1}
          <span className={`vi-status vi-status-${status}`} style={{ marginLeft: "0.5rem" }}>
            {status}
            {seg.resumed ? " · resumed" : ""}
          </span>
        </span>
        {failed && (
          // The runner's own envelope, unsummarised — a code + message an operator can
          // act on beats a generic "failed".
          <span className="vi-error" role="alert" style={{ fontSize: "0.8rem" }}>
            {seg.error?.code ? `[${seg.error.code}] ` : ""}
            {seg.error?.message ?? "failed (no message recorded)"}
          </span>
        )}
        {!src && !failed && !pending && (
          <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
            rendered, but its clip is not on disk right now
          </span>
        )}
        {(seg.frames != null || seg.duration_s != null) && (
          <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
            {seg.frames != null ? `${seg.frames} frames` : ""}
            {seg.duration_s != null ? `${seg.frames != null ? " · " : ""}${Math.round(seg.duration_s)}s` : ""}
          </span>
        )}
        {seg.prompt ? (
          <span className="vi-comfy-hint" style={{ opacity: 0.7, wordBreak: "break-word" }}>
            “{seg.prompt}”
          </span>
        ) : null}
      </div>
    </>
  );
}

/** One session's content — head line, meta line, its ONE action, live pace while it
 *  runs, and (expanded) the segment strip. The <li className="vi-studio-active-row">
 *  wrapper stays in the .map (see the key note above); these stay its direct children
 *  so the row's CSS (which orders the button and the full-width panel) still applies. */
function SessionRowBody({
  session,
  expanded,
  onToggle,
  busy,
  actionError,
  onPause,
  onResume,
}: {
  session: MovieSession;
  expanded: boolean;
  onToggle: () => void;
  busy: boolean;
  actionError: string | null;
  onPause: () => void;
  onResume: () => void;
}) {
  const s = session;
  const live = isSessionLive(s);
  const blocked = resumeBlockedReason(s);
  const status = s.status ?? "unknown";
  const done = s.segments_completed ?? 0;
  const total = s.segments_total ?? s.segments?.length ?? 0;
  const movieSrc = movieMediaSrc(s.movie);
  const geometry =
    s.width != null && s.height != null
      ? `${s.width}×${s.height}${s.fps != null ? ` · ${s.fps}fps` : ""}`
      : "";
  return (
    <>
      <div className="vi-studio-active-head">
        <button
          type="button"
          className="vi-timeline-toggle"
          onClick={onToggle}
          aria-expanded={expanded}
          title={expanded ? "Hide segments" : "Show what this movie has rendered"}
        >
          <span className="vi-timeline-caret" aria-hidden>
            {expanded ? "▾" : "▸"}
          </span>
        </button>
        <span className={`vi-status vi-status-${status}`}>{status}</span>
        <span className="vi-studio-active-id">
          {s.title?.trim() ? s.title : "untitled"} · {shortId(s.movie_id)}
        </span>
      </div>
      <div className="vi-studio-active-meta">
        <span>
          {done}/{total} segments
        </span>
        {geometry ? <span>· {geometry}</span> : null}
        {s.id_lock ? <span>· id_lock</span> : null}
        {s.project ? <span>· {s.project}</span> : null}
        {s.updated != null ? (
          <span title={fmtWhen(s.updated)}>· {agoText(s.updated, Date.now())} ago</span>
        ) : null}
      </div>
      {/* ONE action per row, and never both: a live movie gets Pause, everything else
          gets Resume. `status === "running"` is folded into the live test defensively —
          the route only reports running while the bus agrees, but a row that claims to
          be rendering must always be stoppable, and pause is idempotent anyway. */}
      {live || status === "running" ? (
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={busy}
          title="Stop this movie between segments and park it — everything already rendered stays on disk."
          onClick={onPause}
        >
          {busy ? "Pausing…" : "⏸ Pause"}
        </button>
      ) : (
        // The title rides the WRAPPER: a disabled button swallows pointer events in
        // most browsers, so a tooltip on it is a reason nobody can read.
        <span
          title={
            blocked ??
            "Re-enqueue this movie's spec — segments already rendered come back from the " +
              "clip store without touching a GPU, so it picks up where it stopped."
          }
        >
          <button
            type="button"
            className="vi-btn vi-btn-sm"
            disabled={busy || blocked != null}
            onClick={onResume}
          >
            {busy ? "Resuming…" : "▶ Resume"}
          </button>
        </span>
      )}
      {/* An UNFINISHED movie that cannot be resumed says so in the row itself, not only
          in a tooltip — that is the pre-feature (no spec.json) case, and it is the one
          an operator will otherwise keep clicking. A finished movie needs no excuse. */}
      {blocked && status !== "done" && !live ? (
        <p
          className="vi-knob-hint"
          style={{ flexBasis: "100%", margin: "0.2rem 0 0", wordBreak: "break-word" }}
        >
          {blocked}
        </p>
      ) : null}
      {/* Live pace for a running session — mounted only while it IS running, so a list
          of parked movies costs no job poll. */}
      {live ? <MovieLiveProgress jobId={s.job_id ?? null} /> : null}
      {actionError ? (
        // Verbatim: the 409 bodies ("no persisted spec.json…", "already running —
        // pause it first") are the whole explanation, and paraphrasing them would
        // hide which of the two happened.
        <p className="vi-error" role="alert" style={{ flexBasis: "100%", margin: "0.2rem 0 0" }}>
          {actionError}
        </p>
      ) : null}
      {expanded ? (
        <div className="vi-timeline-panel">
          {movieSrc ? (
            <div style={{ marginBottom: "0.6rem" }}>
              <p className="vi-comfy-label" style={{ marginBottom: "0.3rem" }}>
                Assembled movie
              </p>
              <video
                src={movieSrc}
                controls
                playsInline
                preload="metadata"
                style={{
                  width: "18rem",
                  maxWidth: "60vw",
                  borderRadius: "0.4rem",
                  background: "#000",
                  border: "1px solid var(--vi-border)",
                }}
              />
            </div>
          ) : null}
          {s.segments?.length ? (
            <ul
              style={{
                listStyle: "none",
                margin: 0,
                padding: 0,
                display: "flex",
                flexDirection: "column",
                gap: "0.6rem",
              }}
            >
              {s.segments.map((seg) => (
                <li
                  key={`${s.movie_id}:${seg.index}`}
                  style={{
                    display: "flex",
                    gap: "0.7rem",
                    alignItems: "flex-start",
                    // Not-yet-started segments are dimmed, not hidden: the strip shows
                    // the whole movie, so "how much is left" is visible at a glance.
                    opacity: isPending(seg) ? 0.55 : 1,
                  }}
                >
                  <SegmentRowBody seg={seg} />
                </li>
              ))}
            </ul>
          ) : (
            <p className="vi-knob-hint" style={{ margin: 0 }}>
              No segment records yet — this session was submitted but has not written one.
            </p>
          )}
        </div>
      ) : null}
    </>
  );
}

/** The panel body — split out so the poll it drives exists ONLY while the panel is
 *  open (mounting is the visibility signal; see the `active` gate below). */
export function SessionsBody() {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  // A hidden browser tab is not a visible panel: pause the walk of the movies root
  // while the console is in the background, and reload the moment it comes back.
  const [docVisible, setDocVisible] = useState(
    () => typeof document === "undefined" || document.visibilityState !== "hidden",
  );
  useEffect(() => {
    const onVis = () => setDocVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  const { sessions, loading, error, busy, actionError, pause, resume } =
    useMovieSessions(docVisible);

  const toggle = (movieId: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(movieId)) next.delete(movieId);
      else next.add(movieId);
      return next;
    });

  if (loading && sessions.length === 0) {
    return <p className="vi-knob-hint">Reading the movies root…</p>;
  }
  if (error && sessions.length === 0) {
    return (
      <p className="vi-error" role="alert">
        {error}
      </p>
    );
  }
  if (sessions.length === 0) {
    return (
      <p className="vi-knob-hint">
        No movie sessions on disk yet — generate one below and it appears here, resumable
        even if this tab goes away.
      </p>
    );
  }
  return (
    <ul className="vi-studio-active-list">
      {sessions.map((s) => (
        <li key={s.movie_id} className="vi-studio-active-row">
          <SessionRowBody
            session={s}
            expanded={expanded.has(s.movie_id)}
            onToggle={() => toggle(s.movie_id)}
            busy={busy.has(s.movie_id)}
            actionError={actionError[s.movie_id] ?? null}
            onPause={() => pause(s.movie_id)}
            onResume={() => resume(s.movie_id)}
          />
        </li>
      ))}
    </ul>
  );
}

// ── SESSIONS, PER GENERATE SURFACE (operator ask 2026-08-12) ────────────────
// The recovery panel moved from the Cinema composer into the Active display and
// grew one section per generate surface. Each section is the same question —
// "what work of this kind exists, and what can I do about it" — answered with
// what the server can actually offer today:
//   • Cinema — the durable movie sessions on disk (this file's original body):
//     resume, pause, watch what already rendered. The only tier with a
//     persisted spec.json, so the only one with true Resume.
//   • Scene / Movie / Clip — that surface's jobs from the SAME media-jobs feed
//     the Active list renders (and through the SAME JobRow renderer — no
//     parallel copy): live builds with progress + Cancel, finished ones when
//     the station's "Show finished" filter is on.
// When those tiers grow k91-style persisted sessions server-side, their
// sections upgrade in place — the frame does not change.
const SESSION_CATEGORIES: ReadonlyArray<{
  id: string;
  label: string;
  kinds: ReadonlyArray<string>;
  hint: string;
}> = [
  {
    id: "scene",
    label: "Scene",
    kinds: ["generate_scene", "generate_image"],
    hint: "scene & image fan-outs",
  },
  { id: "movie", label: "Movie", kinds: ["generate_movie"], hint: "goal-timeline movies" },
  { id: "clip", label: "Clip", kinds: ["studio_i2v"], hint: "single studio clips" },
  {
    id: "cinema",
    label: "Cinema",
    kinds: ["generate_studio_movie"],
    hint: "cinema movies",
  },
];

/** One surface's jobs from the shared feed, through the shared row renderer. */
function CategoryJobsBody({
  jobs,
  nowMs,
  cancelBusy,
  onCancel,
  label,
}: {
  jobs: MediaJob[];
  nowMs: number;
  cancelBusy: Set<string>;
  onCancel: (id: string) => void;
  label: string;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  if (jobs.length === 0) {
    return (
      <p className="vi-knob-hint">
        No {label.toLowerCase()} builds in the feed — live ones appear here with
        progress and Cancel; tick “Show finished” above for recent history.
      </p>
    );
  }
  return (
    <ul className="vi-studio-active-list">
      {jobs.map((j) => (
        <li key={j.id} className="vi-studio-active-row">
          <JobRow
            job={j}
            expanded={expanded.has(j.id)}
            onToggle={() => toggle(j.id)}
            nowMs={nowMs}
            cancelBusy={cancelBusy.has(j.id)}
            onCancel={() => onCancel(j.id)}
          />
        </li>
      ))}
    </ul>
  );
}

/**
 * The Sessions panel in the Active display: one collapsible section per generate
 * surface (Scene · Movie · Clip · Cinema). Cinema opens by default — the recovery
 * affordance is the point, and its durable-sessions poll runs only while its
 * section is open (the body unmounts when collapsed, which is what stops it).
 * The other sections cost nothing extra: they filter the feed the station
 * already holds.
 */
export function SessionsPanel({
  jobs,
  nowMs,
  cancelBusy,
  onCancel,
}: {
  jobs: MediaJob[];
  nowMs: number;
  cancelBusy: Set<string>;
  onCancel: (id: string) => void;
}) {
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const toggle = (id: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  return (
    <section className="vi-studio-active" aria-label="Sessions">
      <p className="vi-knob-headline" style={{ margin: "0.8rem 0 0.2rem" }}>
        Sessions{" "}
        <span className="vi-knob-headline-sub">
          previous & in-flight work, per surface
        </span>
      </p>
      {SESSION_CATEGORIES.map((cat) => {
        const catJobs = jobs.filter((j) => cat.kinds.includes(j.name));
        const isOpen = open.has(cat.id);
        return (
          <div key={cat.id}>
            <button
              type="button"
              className="vi-timeline-toggle"
              aria-expanded={isOpen}
              onClick={() => toggle(cat.id)}
              style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}
            >
              <span className="vi-timeline-caret" aria-hidden>
                {isOpen ? "▾" : "▸"}
              </span>
              <span className="vi-knob-headline" style={{ margin: 0 }}>
                {cat.label}{" "}
                <span className="vi-knob-headline-sub">
                  {cat.hint}
                  {catJobs.length > 0 ? ` · ${catJobs.length} in feed` : ""}
                </span>
              </span>
            </button>
            {isOpen ? (
              <CategoryJobsBody
                jobs={catJobs}
                nowMs={nowMs}
                cancelBusy={cancelBusy}
                onCancel={onCancel}
                label={cat.label}
              />
            ) : null}
          </div>
        );
      })}
    </section>
  );
}
