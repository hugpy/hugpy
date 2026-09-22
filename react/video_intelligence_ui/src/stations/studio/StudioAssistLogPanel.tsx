// STUDIO-ASSIST LIVE LOG PANEL (operator directive, 2026-07-31).
//
// "a live log in the studio ui showing what each generate attempt actually
// returned — the raw model reply, what was stripped, and the outcome — so they
// can self-diagnose without asking the keeper each time."
//
// One row per generate attempt, newest first. Each row shows time, the model
// asked-for → the one that answered, the mode/kind, an outcome badge, and the
// elapsed time. The RAW model reply is the money view — collapsed by default,
// expanded on click — alongside the think-stripped `text`, the `reasoning` that
// was pulled out, and a "from reasoning" mark when the answer was salvaged from
// the monologue (the "returned only reasoning and no output" case). A parse
// failure ("did not return the JSON object the spread contract requires") shows
// the same raw reply so the operator sees exactly what the model sent.
//
// Presentation only: it reads the shared stream hook and holds no editor state.
// Styling lives in style/app.css under the vi-assistlog-* classes (the arm's
// convention — components carry no styling logic).
import { useMemo, useState } from "react";
import { useStudioAssistLog, type AssistAttempt } from "../../video/useStudioAssistLog";

const OUTCOME_LABEL: Record<string, string> = {
  served: "served",
  empty: "empty",
  parse_error: "parse error",
  worker_error: "worker error",
  resolve_error: "resolve error",
};

function outcomeClass(outcome?: string): string {
  switch (outcome) {
    case "served":
      return "vi-assistlog-badge vi-assistlog-badge--ok";
    case "empty":
    case "parse_error":
      return "vi-assistlog-badge vi-assistlog-badge--warn";
    case "worker_error":
    case "resolve_error":
      return "vi-assistlog-badge vi-assistlog-badge--err";
    default:
      return "vi-assistlog-badge";
  }
}

function fmtClock(ts?: number): string {
  if (!ts) return "";
  try {
    return new Date(ts * 1000).toLocaleTimeString();
  } catch {
    return "";
  }
}

function fmtMs(ms?: number): string {
  if (ms == null) return "";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function AttemptRow({ rec }: { rec: AssistAttempt }) {
  const [open, setOpen] = useState(false);
  const model =
    rec.model_resolved && rec.model_resolved !== rec.model_requested
      ? `${rec.model_requested ?? "?"} → ${rec.model_resolved}`
      : rec.model_requested ?? rec.model_resolved ?? "?";
  const modeKind = [rec.mode, rec.kind].filter(Boolean).join(" · ");
  const hasDetail = Boolean(rec.raw || rec.text || rec.reasoning || rec.error);

  return (
    <div className={`vi-assistlog-row vi-assistlog-row--${rec.outcome ?? "unknown"}`}>
      <button
        type="button"
        className="vi-assistlog-head"
        onClick={() => hasDetail && setOpen((o) => !o)}
        aria-expanded={open}
        title={hasDetail ? "Show the raw model reply" : undefined}
      >
        <span className="vi-assistlog-caret">{hasDetail ? (open ? "▾" : "▸") : "·"}</span>
        <span className="vi-assistlog-time">{fmtClock(rec.ts)}</span>
        <span className="vi-assistlog-model" title={model}>
          {model}
        </span>
        {modeKind && <span className="vi-assistlog-mode">{modeKind}</span>}
        {rec.from_reasoning && (
          <span
            className="vi-assistlog-chip"
            title="No prose came back — the prompt was salvaged from the model's reasoning"
          >
            from reasoning
          </span>
        )}
        <span className="vi-assistlog-spacer" />
        {rec.elapsed_ms != null && <span className="vi-assistlog-elapsed">{fmtMs(rec.elapsed_ms)}</span>}
        <span className={outcomeClass(rec.outcome)}>
          {OUTCOME_LABEL[rec.outcome ?? ""] ?? rec.outcome ?? "?"}
        </span>
      </button>

      {rec.error && !open && <div className="vi-assistlog-errline">{rec.error}</div>}

      {open && (
        <div className="vi-assistlog-detail">
          {rec.error && (
            <div className="vi-assistlog-field">
              <span className="vi-assistlog-label">error</span>
              <div className="vi-assistlog-errbox">{rec.error}</div>
            </div>
          )}
          <div className="vi-assistlog-field">
            <span className="vi-assistlog-label">
              raw reply{rec.raw_truncated ? " (truncated)" : ""}
            </span>
            <pre className="vi-assistlog-raw">
              {rec.raw != null && rec.raw !== "" ? rec.raw : <em>(no reply came back)</em>}
            </pre>
          </div>
          {rec.text && (
            <div className="vi-assistlog-field">
              <span className="vi-assistlog-label">stripped text (the prompt)</span>
              <pre className="vi-assistlog-text">{rec.text}</pre>
            </div>
          )}
          {rec.reasoning && (
            <div className="vi-assistlog-field">
              <span className="vi-assistlog-label">reasoning (stripped out)</span>
              <pre className="vi-assistlog-reasoning">{rec.reasoning}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function StudioAssistLogPanel() {
  const { attempts, conn, error, reconnect } = useStudioAssistLog();
  const [filter, setFilter] = useState("");
  const [collapsed, setCollapsed] = useState(false);

  const rows = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const out: AssistAttempt[] = [];
    for (let i = attempts.length - 1; i >= 0; i--) {
      const r = attempts[i];
      if (!q) {
        out.push(r);
        continue;
      }
      const hay = [
        r.mode,
        r.kind,
        r.outcome,
        r.model_requested,
        r.model_resolved,
        r.error,
        r.raw,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (hay.includes(q)) out.push(r);
    }
    return out;
  }, [attempts, filter]);

  return (
    <section className="vi-assistlog">
      <header className="vi-assistlog-toolbar">
        <button
          type="button"
          className="vi-assistlog-collapse"
          onClick={() => setCollapsed((c) => !c)}
          aria-expanded={!collapsed}
        >
          {collapsed ? "▸" : "▾"} Generation log
        </button>
        <span className={`vi-assistlog-dot vi-assistlog-dot--${conn}`} title={`stream ${conn}`} />
        <span className="vi-assistlog-conn">{conn}</span>
        <span className="vi-assistlog-spacer" />
        {!collapsed && (
          <input
            className="vi-assistlog-filter"
            value={filter}
            placeholder="filter by model, mode, outcome…"
            onChange={(e) => setFilter(e.target.value)}
          />
        )}
        {conn === "disconnected" && (
          <button type="button" className="vi-assistlog-reconnect" onClick={reconnect}>
            Reconnect
          </button>
        )}
        <span className="vi-assistlog-count">
          {rows.length} attempt{rows.length === 1 ? "" : "s"}
        </span>
      </header>

      {!collapsed && (
        <>
          {error && <div className="vi-assistlog-error">{error}</div>}
          {rows.length === 0 ? (
            <div className="vi-assistlog-empty">
              No generation attempts recorded yet. Each Enhance / Generate / Spread /
              Negative call lands here — the raw model reply, what was stripped, and the
              outcome.
            </div>
          ) : (
            <div className="vi-assistlog-list">
              {rows.map((r) => (
                <AttemptRow key={r.run_id ?? r._id ?? `${r.ts}-${r.seq}`} rec={r} />
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}

export default StudioAssistLogPanel;
