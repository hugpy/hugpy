// ToksPanel — the LIVE "tok/s" overview for the generate workbench, contributed
// as ONE TAB to the unified drawer (drawerShell.tsx → WorkbenchDrawer.tsx), the
// same window as Console / ComfyUI / Active and framed with the same
// `.vi-active-panel-body` chrome. It mirrors ActivePanel: a real component whose
// MOUNT starts the poll, so `keepMounted: false` IS the pause — the 2s cadence
// runs exactly when the operator is looking at this tab (see ActivePanel's note).
//
// Three compact sections, all fed by `useToks` (two same-origin `/api` endpoints
// polled together every 2s, error-tolerant, mount-gated):
//   1. Live tail        — the last ~25 recent samples, newest on top.
//   2. Per-worker strip  — recent grouped by worker → last tok/s + window avg.
//   3. Best-config board — /report's top ~10, best mean first.
// Every field is guarded: the server side is being built in parallel, so a
// missing number renders as "—", never a crash.
import { useEffect, useState } from "react";
import type { DrawerTab } from "./drawerShell";
import {
  useToks,
  rollupByWorker,
  tokAgo,
  TOKS_POLL_MS,
  type TokEntry,
  type TokGroup,
} from "../video/useToks";

const TAIL_LIMIT = 25;
const BOARD_LIMIT = 10;

/** A number to N decimals, or "—" when the server sent nothing honest. */
function fmt(v: number | null | undefined, digits = 1): string {
  return typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "—";
}

/** The ok/err badge for one sample (null ok → no badge, we just don't know). */
function OkBadge({ ok }: { ok: boolean | null }) {
  if (ok == null) return null;
  return (
    <span
      className={`vi-assistlog-badge ${ok ? "vi-assistlog-badge--ok" : "vi-assistlog-badge--err"}`}
      title={ok ? "completed ok" : "errored"}
    >
      {ok ? "ok" : "err"}
    </span>
  );
}

function LiveTail({ entries, nowMs }: { entries: TokEntry[]; nowMs: number }) {
  const tail = entries.slice(0, TAIL_LIMIT);
  return (
    <div>
      <h3 className="vi-toks-heading">Live tail</h3>
      {tail.length === 0 ? (
        <p className="vi-knob-hint">No samples yet — run an inference and it lands here.</p>
      ) : (
        <ul className="vi-assistlog-list vi-toks-list">
          {tail.map((e, i) => (
            <li
              key={`${e.workerId ?? "?"}-${e.ts ?? i}-${i}`}
              className="vi-assistlog-row vi-toks-row"
            >
              <span className="vi-toks-tok">{fmt(e.tokS)}<span className="vi-knob-hint"> tok/s</span></span>
              <span className="vi-toks-model" title={e.modelKey ?? undefined}>
                {e.modelKey ?? "—"}
              </span>
              <span className="vi-knob-hint vi-toks-worker">{e.workerName ?? e.workerId ?? "—"}</span>
              <span className="vi-knob-hint">ttft {fmt(e.ttftS, 2)}s</span>
              <span className="vi-knob-hint">{e.completionTokens ?? "—"} tok</span>
              {e.configKey ? <span className="vi-assistlog-chip">{e.configKey}</span> : null}
              <OkBadge ok={e.ok} />
              <span className="vi-knob-hint vi-toks-ago">{tokAgo(e.ts, nowMs)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function WorkerStrip({ entries }: { entries: TokEntry[] }) {
  const rows = rollupByWorker(entries);
  return (
    <div>
      <h3 className="vi-toks-heading">Per-worker</h3>
      {rows.length === 0 ? (
        <p className="vi-knob-hint">No workers reporting.</p>
      ) : (
        <div className="vi-toks-strip">
          {rows.map((w) => (
            <div key={w.workerId} className="vi-toks-worker-card">
              <div className="vi-toks-worker-name" title={w.modelKey ?? undefined}>
                {w.workerName}
              </div>
              <div className="vi-toks-worker-nums">
                <span className="vi-toks-tok">{fmt(w.lastTokS)}</span>
                <span className="vi-knob-hint"> now · avg {fmt(w.avgTokS)} · n{w.n}</span>
              </div>
              {w.modelKey ? <div className="vi-knob-hint vi-toks-worker-model">{w.modelKey}</div> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Leaderboard({ groups }: { groups: TokGroup[] }) {
  const top = groups.slice(0, BOARD_LIMIT);
  return (
    <div>
      <h3 className="vi-toks-heading">Best config</h3>
      {top.length === 0 ? (
        <p className="vi-knob-hint">No rollup yet — the report fills in as samples accrue.</p>
      ) : (
        <ul className="vi-assistlog-list vi-toks-list">
          {top.map((g, i) => (
            <li
              key={`${g.workerName ?? "?"}-${g.modelKey ?? "?"}-${g.configKey ?? i}`}
              className="vi-assistlog-row vi-toks-row"
            >
              <span className="vi-knob-hint vi-toks-rank">#{i + 1}</span>
              <span className="vi-toks-tok">{fmt(g.meanTokS)}<span className="vi-knob-hint"> mean</span></span>
              <span className="vi-knob-hint">p50 {fmt(g.p50TokS)}</span>
              <span className="vi-knob-hint">p95 {fmt(g.p95TokS)}</span>
              <span className="vi-toks-model">{g.modelKey ?? "—"}</span>
              <span className="vi-knob-hint vi-toks-worker">{g.workerName ?? "—"}</span>
              {g.configKey ? <span className="vi-assistlog-chip">{g.configKey}</span> : null}
              <span className="vi-knob-hint">n{g.n ?? "—"}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** The panel body — a station card with the three sections + a live header. */
export function ToksStation() {
  const { entries, groups, live } = useToks();
  // A 1s clock so the "Ns ago" column ages between the 2s polls, exactly as the
  // Active panel drives its ProcessTimeline.
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <section className="station-card station-wide vi-toks-station">
      <div className="vi-toks-header">
        <h2>tok/s</h2>
        <span className="vi-assistlog-conn vi-toks-cadence">
          <span
            className={`vi-assistlog-dot ${live ? "vi-assistlog-dot--live" : "vi-assistlog-dot--disconnected"}`}
            aria-hidden
          />
          {live ? "live" : "paused"} · every {Math.round(TOKS_POLL_MS / 1000)}s
        </span>
      </div>
      <p className="vi-knob-hint">
        Live inference throughput across the LLM workers — the newest samples, a
        per-worker last/avg strip, and the best-performing configs by mean tok/s.
      </p>
      <LiveTail entries={entries} nowMs={nowMs} />
      <WorkerStrip entries={entries} />
      <Leaderboard groups={groups} />
    </section>
  );
}

/** The tok/s tab in the unified drawer — mount starts the 2s poll, so
 *  `keepMounted: false` is the pause (see ActivePanel's polling note). */
export function toksDrawerTab(): DrawerTab<"toks"> {
  return {
    id: "toks",
    label: "tok/s",
    hint: "Open the live tok/s overview alongside this station — no tab switch",
    toggleClassName: "vi-toks-toggle",
    toggleLabel: "⚡ tok/s",
    ariaLabel: "Live tokens-per-second overview",
    panelClassName: "vi-toks-panel",
    keepMounted: false, // mount IS the pause — same policy as the Active tab
    body: (
      <div className="vi-active-panel-body">
        <ToksStation />
      </div>
    ),
  };
}
