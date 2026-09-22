/**
 * or-k18 — tiny render helpers shared by the oracle panels. Nothing here
 * fetches; everything takes the wire object and prints it, including the raw
 * JSON behind a <details> so every intermediate object stays inspectable with
 * its provenance.
 */
import type { ReactNode } from "react";

import type { Gap } from "./useOracle";
import type { Scorecard } from "./oracleTypes";

export function pretty(value: unknown): string {
  try {
    return JSON.stringify(value ?? null, null, 2);
  } catch {
    return String(value ?? "");
  }
}

export function short(text: unknown, n = 14): string {
  const s = String(text ?? "");
  if (!s) return "—";
  return s.length > n ? `${s.slice(0, n)}…` : s;
}

/** ISO string → "2026-08-21 14:03:11" in the viewer's zone; unknown → "—". */
export function when(iso: unknown): string {
  if (typeof iso !== "string" || !iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { hour12: false });
}

export function ago(ms: number | null): string {
  if (ms == null) return "—";
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m ago`;
}

export function RawJson({ label, value, open = false }: { label: string; value: unknown; open?: boolean }) {
  return (
    <details className="vi-oracle-raw-toggle" open={open}>
      <summary>{label}</summary>
      <pre className="vi-oracle-raw">{pretty(value)}</pre>
    </details>
  );
}

export function GapNote({ gap, children }: { gap: Gap | null; children?: ReactNode }) {
  if (!gap) return null;
  return (
    <p className="vi-oracle-gap" role="status">
      {gap.message}
      {children}
    </p>
  );
}

export function StateChip({ state }: { state: string }) {
  const safe = state.replace(/[^a-z_]/gi, "_");
  return <span className={`vi-oracle-state vi-oracle-state-${safe}`}>{state || "—"}</span>;
}

export function Tag({ kind, children }: { kind?: "info" | "warn" | "alarm" | "ok" | "accent"; children: ReactNode }) {
  return <span className={`vi-oracle-tag${kind ? ` vi-oracle-tag-${kind}` : ""}`}>{children}</span>;
}

/** Confidence meter + checks + judges + disagreements. Renders nothing made up:
 *  a card without `confidence` shows "not reported", not a bar. */
export function ScorecardView({ card }: { card: Scorecard | null | undefined }) {
  if (!card) return <p className="vi-oracle-note">No scorecard on this node in the manifest.</p>;
  const conf = typeof card.confidence === "number" ? card.confidence : null;
  const disagreements = card.disagreements ?? [];
  return (
    <div>
      <div className="vi-oracle-row">
        <Tag kind={card.hard_pass ? "ok" : "alarm"}>{card.hard_pass ? "hard pass" : "hard fail"}</Tag>
        {card.repair_code ? <Tag kind="warn">repair: {card.repair_code}</Tag> : null}
      </div>
      <div className="vi-oracle-meter" style={{ marginTop: "0.4rem" }}>
        <span>confidence</span>
        {conf == null ? (
          <span className="vi-knob-hint">not reported</span>
        ) : (
          <>
            <div className="vi-oracle-meter-track" aria-hidden>
              <div className="vi-oracle-meter-fill" style={{ width: `${Math.round(conf * 100)}%` }} />
            </div>
            <code>{conf.toFixed(2)}</code>
          </>
        )}
      </div>
      {card.diagnosis ? <p className="vi-oracle-note">diagnosis: {card.diagnosis}</p> : null}
      {card.recommended_repair ? (
        <p className="vi-oracle-note">recommended repair: {card.recommended_repair}</p>
      ) : null}
      <h4 style={{ margin: "0.6rem 0 0.3rem" }} className="vi-oracle-state">
        disagreements ({disagreements.length})
      </h4>
      {disagreements.length ? (
        <ul className="vi-oracle-reasons">
          {disagreements.map((d, i) => (
            <li key={i}>{d}</li>
          ))}
        </ul>
      ) : (
        <p className="vi-oracle-note">none recorded between checks and judges</p>
      )}
      {(card.checks ?? []).length ? (
        <table className="vi-oracle-table" style={{ marginTop: "0.5rem" }}>
          <thead>
            <tr>
              <th>check</th>
              <th>kind</th>
              <th>value</th>
              <th>threshold</th>
              <th>pass</th>
            </tr>
          </thead>
          <tbody>
            {(card.checks ?? []).map((c, i) => (
              <tr key={`${c.name}-${i}`}>
                <td>{c.name}</td>
                <td>
                  <code>{c.kind}</code>
                </td>
                <td>
                  <code>{short(pretty(c.value), 40)}</code>
                </td>
                <td>
                  <code>{short(pretty(c.threshold), 24)}</code>
                </td>
                <td>
                  <Tag kind={c.passed ? "ok" : "alarm"}>{c.passed ? "yes" : "no"}</Tag>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {(card.judge_results ?? []).length ? (
        <ul className="vi-oracle-candidates" style={{ marginTop: "0.5rem" }}>
          {(card.judge_results ?? []).map((j, i) => (
            <li key={`${j.judge}-${i}`} className="vi-oracle-candidate">
              <div className="vi-oracle-candidate-head">
                <code>{j.judge}</code>
                <Tag kind={j.verdict === "pass" ? "ok" : "warn"}>{j.verdict}</Tag>
                {typeof j.score === "number" ? <code>score {j.score}</code> : <span className="vi-knob-hint">no score</span>}
              </div>
              {j.rationale ? <p className="vi-oracle-finding-action">{j.rationale}</p> : null}
            </li>
          ))}
        </ul>
      ) : null}
      <RawJson label="scorecard json" value={card} />
    </div>
  );
}
