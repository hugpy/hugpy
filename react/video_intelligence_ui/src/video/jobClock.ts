/**
 * k117 — the feed's clock, told honestly.
 *
 * THE BUG the operator reported: the Active-Processes feed showed jobs at 875+
 * minutes in "loading", 25h in "awaiting_capacity", and DONE rows whose timers
 * kept ticking under a stale stage label ("archiving"). Keeper triage found the
 * server rows were CORRECT — the "done · 946m · archiving" row had really run
 * 9.4 seconds. Two client-side lies produced that display:
 *
 *   1. The row rendered `elapsedSince(createdAt)` — i.e. `now - created`, a
 *      ticking AGE — as if it were runtime. For a terminal row that number keeps
 *      counting for ever; for a live row it conflates a 25h queue wait with a
 *      9s render.
 *   2. The row rendered `current_stage` — the last IN-FLIGHT stage — on terminal
 *      rows, so a finished render was labelled "archiving" permanently.
 *
 * The server (video_intel/job_lifecycle.py) now sends the facts this file needs:
 * FROZEN `run_s`/`queue_wait_s`/`total_s`/`terminal_at` + the true
 * `terminal_stage` on terminal rows, and `elapsed_in_stage_s` +
 * `last_progress_at` (with queue wait kept separate) on live ones. Nothing here
 * derives a duration from `Date.now()` — that is the entire point.
 *
 * A row from an older server (or the demo fixtures) carries none of these, so
 * every helper degrades to the honest fallback: it says AGE, and labels it age,
 * rather than passing an age off as a runtime.
 */
import type { MediaJob } from "./useMediaJobs";

/** Seconds as "9.4s" / "3m 4s" / "1h 02m" — compact, never lossy about scale. */
export function fmtDuration(secs: number | null): string {
  if (secs == null || !isFinite(secs) || secs < 0) return "";
  if (secs < 10) return `${secs.toFixed(1)}s`;
  const s = Math.floor(secs);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  const h = Math.floor(s / 3600);
  return `${h}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/**
 * The stage label for a row. A TERMINAL row says what it ended as ("done",
 * "failed", "cancelled") — never the stage it happened to be in when it ended.
 * That stage is still available (`atStage`) and the expanded timeline shows it;
 * it is simply no longer allowed to masquerade as a current stage.
 */
export function jobStageLabel(job: MediaJob): string {
  const raw = job.terminal ? job.terminalStage ?? job.status : job.stage;
  return raw ? raw.replace(/_/g, " ") : "";
}

/**
 * The stage a TERMINAL job was in when it ended, for the secondary "(in
 * archiving)" note — empty when it adds nothing (a job that ended in no stage,
 * or whose last stage is the outcome itself).
 */
export function terminalAtStage(job: MediaJob): string {
  if (!job.terminal) return "";
  const at = job.atStage;
  if (!at || at === job.terminalStage || at === job.status) return "";
  return at.replace(/_/g, " ");
}

/**
 * The clock chips for one row, in display order.
 *
 * TERMINAL: exactly one FROZEN duration — how long it actually ran — plus the
 * queue wait when there was a meaningful one. Nothing ticks.
 *
 * LIVE: the two clocks kept APART, which is the whole fix: time in the CURRENT
 * stage, and the queue wait, as separate facts. A 9s render that waited 25h
 * reads as "queued 25h" + "in generating 9s", never as "875m".
 */
export function jobClockParts(job: MediaJob): string[] {
  const out: string[] = [];
  if (job.terminal) {
    if (job.runS != null) {
      out.push(`ran ${fmtDuration(job.runS)}`);
      // Only worth showing when the wait was a real part of the story.
      if (job.queueWaitS != null && job.queueWaitS >= 5)
        out.push(`queued ${fmtDuration(job.queueWaitS)}`);
      return out;
    }
    // No recorded start (a pre-k117 row whose runner wrote no stage entries):
    // the total span is known, the queue/run split is not. Say "took", which is
    // true of the whole span, rather than inventing a runtime.
    if (job.totalS != null) return [`took ${fmtDuration(job.totalS)}`];
    // Older server / demo fixture: say AGE, and say that it is age.
    const age = ageSeconds(job.createdAt);
    return age == null ? [] : [`${fmtDuration(age)} ago`];
  }
  if (job.elapsedInStageS != null && job.stage)
    out.push(`in ${job.stage.replace(/_/g, " ")} ${fmtDuration(job.elapsedInStageS)}`);
  if (job.queueWaitS != null && job.queueWaitS >= 1)
    out.push(`queued ${fmtDuration(job.queueWaitS)}`);
  if (job.runS != null && job.runS >= 1) out.push(`running ${fmtDuration(job.runS)}`);
  if (out.length) return out;
  const age = ageSeconds(job.createdAt);
  return age == null ? [] : [`${fmtDuration(age)} ago`];
}

function ageSeconds(createdAt: number | null): number | null {
  if (createdAt == null) return null;
  return Math.max(0, Date.now() / 1000 - createdAt);
}
