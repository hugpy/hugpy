// ACTIVE PROCESSES tab — the in-flight STUDIO jobs (queued / claimed / running) as a
// focused live panel. Each row is EXPANDABLE into the exhaustive per-process view:
// the stage TIMELINE (what it did + what it's doing now), the current-activity line,
// an honest stall indicator tied to the current stage, and — on failure — the exact
// failing stage + error code + message. This replaces the old "expands to a loading
// bar that never populates" with real, server-sent detail (no fabricated bars).
//
// SCOPE (operator, 2026-07-31): this panel is fed by GET /video/studio/clips, which is
// `WHERE name='studio_i2v'` — so it lists ONLY studio-originated renders, never general
// fleet model calls. Cancel here therefore only ever stops a studio render.
//
// The GET /video/studio/clips route projects `progress` (incl. the reservation
// awaiting_capacity HOLD marker), a `placement` object, and the new stage-timeline
// telemetry (stage_log / failure / current_stage / last_movement_ts). Terminal rows
// (done / failed / cancelled) stay in the Library tab, where their full timeline +
// failure detail is inspectable via the Details expander.
import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "../../transport/client";
import { jobCancelUrl } from "../../config";
import {
  shortId,
  ProcessTimeline,
  isStalled,
  type Clip,
  type StageEntry,
  type JobFailure,
} from "./studioShared";
import { placementLine, isExternal, reservedGib, type Placement } from "../../video/useMediaJobs";

const IN_FLIGHT = new Set(["queued", "claimed", "running", "cancelling"]);
const TERMINAL = new Set(["done", "failed", "cancelled"]);

function elapsed(c: Clip): string {
  const t = c.created ?? c.updated;
  if (t == null) return "";
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - t));
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  return `${m}m ${secs % 60}s`;
}

// The clip's `placement` rides the wire in snake_case; adapt it to the camelCase
// Placement the shared render helpers expect. null when the clip carries none.
function clipPlacement(c: Clip): Placement | null {
  const p = (c as { placement?: Record<string, unknown> | null }).placement;
  if (!p || typeof p !== "object") return null;
  const s = (v: unknown) => (typeof v === "string" && v ? v : null);
  const n = (v: unknown) => (typeof v === "number" && isFinite(v) ? v : null);
  return {
    source: s(p.source),
    host: s(p.host),
    workerId: s(p.worker_id),
    gpu: s(p.gpu),
    process: s(p.process),
    reservedBytes: n(p.reserved_bytes),
  };
}

// The effective phase chip: the reservation awaiting_capacity HOLD overrides the
// raw status (the clip's progress blob carries the marker), else the raw status.
function clipPhase(c: Clip): string {
  const prog = (c as { progress?: Record<string, unknown> | null }).progress;
  if (prog && typeof prog === "object" && prog.phase === "awaiting_capacity")
    return "awaiting_capacity";
  return c.status ?? "";
}

export function StudioActiveProcesses({
  clips,
  refresh,
}: {
  clips: Clip[];
  refresh: () => void;
}) {
  const [cancelBusy, setCancelBusy] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  // A ~1s tick so the stall clock + "since last movement" advance live (independent
  // of the 6s clip poll), without any fabricated progress.
  const [nowMs, setNowMs] = useState(() => Date.now());
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    const iv = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => {
      mounted.current = false;
      window.clearInterval(iv);
    };
  }, []);

  const toggle = useCallback((jobId: string) => {
    setExpanded((prev) => {
      const n = new Set(prev);
      if (n.has(jobId)) n.delete(jobId);
      else n.add(jobId);
      return n;
    });
  }, []);

  const onCancel = useCallback(
    async (jobId: string) => {
      setCancelBusy((prev) => new Set(prev).add(jobId));
      try {
        await request<unknown>(jobCancelUrl(jobId), {
          method: "POST",
          meta: { specKey: "studio", operation: "studio.clip.cancel" },
        });
        if (!mounted.current) return;
        refresh();
      } finally {
        if (mounted.current)
          setCancelBusy((prev) => {
            const n = new Set(prev);
            n.delete(jobId);
            return n;
          });
      }
    },
    [refresh],
  );

  const active = clips.filter((c) => IN_FLIGHT.has(c.status ?? ""));
  // Queue position: Nth among QUEUED jobs (oldest first) — a client-side, honest index.
  const queuePos = new Map<string, number>();
  active
    .filter((c) => (c.status ?? "") === "queued")
    .slice()
    .sort((a, b) => (a.created ?? 0) - (b.created ?? 0))
    .forEach((c, i) => queuePos.set(c.job_id, i + 1));

  return (
    <div className="vi-studio-active">
      <p className="vi-knob-headline">
        Active renders{" "}
        <span className="vi-knob-headline-sub">running &amp; queued studio jobs</span>
      </p>
      {active.length === 0 ? (
        <p className="vi-knob-hint">No active renders — enqueue one and it appears here.</p>
      ) : (
        <ul className="vi-studio-active-list">
          {active.map((c) => {
            const phase = clipPhase(c);
            const pos = queuePos.get(c.job_id);
            const el = elapsed(c);
            const pl = clipPlacement(c);
            const line = placementLine(pl);
            const gib = reservedGib(pl);
            const external = isExternal(pl);
            const phaseLabel =
              phase === "awaiting_capacity" ? "awaiting capacity" : phase;
            const isOpen = expanded.has(c.job_id);
            const terminal = TERMINAL.has(c.status ?? "");
            const stageLog = (c.stage_log ?? null) as StageEntry[] | null;
            const failure = (c.failure ?? null) as JobFailure | null;
            const stalled = isStalled(c.last_movement_ts, terminal, nowMs);
            return (
              <li key={c.job_id} className="vi-studio-active-row">
                <div className="vi-studio-active-head">
                  <button
                    type="button"
                    className="vi-timeline-toggle"
                    onClick={() => toggle(c.job_id)}
                    aria-expanded={isOpen}
                    title={isOpen ? "Hide detail" : "Show what it's doing"}
                  >
                    <span className="vi-timeline-caret" aria-hidden>
                      {isOpen ? "▾" : "▸"}
                    </span>
                  </button>
                  <span
                    className={`vi-status vi-status-${phase}`}
                    title={
                      phase === "awaiting_capacity"
                        ? "Held — waiting for GPU capacity"
                        : undefined
                    }
                  >
                    {phaseLabel}
                  </span>
                  {stalled ? <span className="vi-timeline-stall-badge">stalled</span> : null}
                  <span className="vi-studio-active-id">{shortId(c.job_id)}</span>
                </div>
                <div className="vi-studio-active-meta">
                  {line ? (
                    <span
                      className={`vi-placement${external ? " vi-placement-external" : ""}`}
                    >
                      {external ? "⇄ " : ""}
                      {line}
                      {external ? " · external" : ""}
                    </span>
                  ) : null}
                  {c.current_stage ? <span>· {c.current_stage.replace(/_/g, " ")}</span> : null}
                  {gib ? <span>· {gib}</span> : null}
                  {pos != null ? <span>· queue #{pos}</span> : null}
                  {el ? <span>· {el}</span> : null}
                </div>
                {isOpen ? (
                  <div className="vi-timeline-panel">
                    <ProcessTimeline
                      stageLog={stageLog}
                      failure={failure}
                      currentStage={c.current_stage}
                      lastMovementTs={c.last_movement_ts}
                      terminal={terminal}
                      nowMs={nowMs}
                    />
                  </div>
                ) : null}
                <button
                  type="button"
                  className="vi-btn vi-btn-sm vi-btn-ghost"
                  onClick={() => void onCancel(c.job_id)}
                  disabled={cancelBusy.has(c.job_id)}
                >
                  {cancelBusy.has(c.job_id) ? "Cancelling…" : "✕ Cancel"}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
