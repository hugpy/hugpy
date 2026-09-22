// PER-SEGMENT COORDINATION BADGE (k121) — the review, on the row it is about.
//
// One small chip in the prompt card's existing `headerHint` slot. Collapsed it
// is a single status word; expanded it lists every decision on that segment as
//
//     [status] join: cut → vace_extend
//     “…the same shot continues, still running…”
//     the prose continues the previous shot, so the join is set to 'vace_extend'
//     [accept] [reject]           ← proposals only
//
// THE EVIDENCE QUOTE IS THE POINT. A badge that said "join changed" would be an
// assertion; a badge that quotes the operator's own sentence is falsifiable, and
// the operator can settle it in one glance — which is exactly what was missing
// when four prompts declared continuity and every joint stayed `cut`.
//
// Presentational only: values in, callbacks out. It owns one piece of ephemeral
// view state (open/closed) and nothing else. Colours reuse the assist-log badge
// palette (`coordinationClass`) rather than minting a fourth badge language.
import { useState, type ReactNode } from "react";
import {
  coordinationClass,
  coordinationLabel,
  knobLabel,
  knobValue,
  type CoordinationDecision,
  type CoordinationSegment,
} from "../../video/coordinationReport";

export interface CoordinationBadgeProps {
  /** This segment's review row. `null` ⇒ nothing rendered (never reviewed). */
  review: CoordinationSegment | null;
  /** Apply a `proposed` decision to the row. Absent ⇒ read-only badge. */
  onAccept?: (decision: CoordinationDecision) => void;
  /** Dismiss a `proposed` decision without applying it. */
  onReject?: (decision: CoordinationDecision) => void;
  /** Decision keys (`${segment_id}:${knob}`) the operator already answered. */
  settled?: Set<string>;
  disabled?: boolean;
  /** Extra content the host wants inside the same header line (the joint hint). */
  children?: ReactNode;
}

export function decisionKey(d: CoordinationDecision): string {
  return `${d.segment_id}:${d.knob}:${d.step ?? ""}`;
}

/** `ok` rows still render — a review that only spoke on failure would leave
 *  "reviewed and fine" and "never reviewed" looking identical, which is the
 *  original incident in miniature. */
export function CoordinationBadge({
  review,
  onAccept,
  onReject,
  settled,
  disabled,
  children,
}: CoordinationBadgeProps) {
  const [open, setOpen] = useState(false);
  if (!review) return <>{children}</>;

  const shown = review.decisions.filter((d) => d.status !== "ok" || open);
  const proposals = review.decisions.filter(
    (d) => d.status === "proposed" && !(settled?.has(decisionKey(d)) ?? false),
  );
  const label = coordinationLabel(review.status);
  const title =
    review.decisions.length === 0
      ? "Reviewed: nothing in this shot's words asks for a knob."
      : review.decisions
          .map((d) => `${d.status.toUpperCase()} ${knobLabel(d.knob)}: ${d.reason}`)
          .join("\n\n");

  return (
    <>
      {children}
      <button
        type="button"
        className={coordinationClass(review.status)}
        style={{ cursor: "pointer", background: "none" }}
        title={title}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {label}
        {proposals.length > 0 ? ` · ${proposals.length}?` : ""}
      </button>
      {open && (
        <ul
          className="vi-coord-list"
          style={{
            listStyle: "none",
            margin: "0.35rem 0 0",
            padding: 0,
            flexBasis: "100%",
            fontSize: "0.72rem",
          }}
        >
          {shown.length === 0 && (
            <li style={{ opacity: 0.7 }}>
              Reviewed — this shot's words ask for no knob.
            </li>
          )}
          {shown.map((d) => {
            const key = decisionKey(d);
            const answered = settled?.has(key) ?? false;
            return (
              <li key={key} style={{ margin: "0 0 0.4rem", lineHeight: 1.35 }}>
                <span
                  className={coordinationClass(d.status)}
                  style={{ marginRight: "0.35rem" }}
                >
                  {d.step ? d.step : d.status}
                </span>
                <strong>{knobLabel(d.knob)}</strong>{" "}
                <span style={{ opacity: 0.85 }}>
                  {knobValue(d.current)} → {knobValue(d.proposed)}
                </span>
                {d.evidence_quote && (
                  <div style={{ opacity: 0.8, fontStyle: "italic" }}>
                    “{d.evidence_quote}”
                  </div>
                )}
                <div style={{ opacity: 0.75 }}>{d.reason}</div>
                {d.status === "proposed" && !answered && (onAccept || onReject) && (
                  <div style={{ marginTop: "0.2rem" }}>
                    {onAccept && (
                      <button
                        type="button"
                        className="vi-btn vi-btn-sm vi-btn-ghost"
                        disabled={disabled}
                        onClick={() => onAccept(d)}
                      >
                        ✓ accept
                      </button>
                    )}
                    {onReject && (
                      <button
                        type="button"
                        className="vi-btn vi-btn-sm vi-btn-ghost"
                        disabled={disabled}
                        onClick={() => onReject(d)}
                      >
                        ✕ reject
                      </button>
                    )}
                  </div>
                )}
                {answered && <div style={{ opacity: 0.6 }}>answered</div>}
              </li>
            );
          })}
        </ul>
      )}
    </>
  );
}

export interface CoordinationSummaryProps {
  /** Report-level notes + counts, under the group toolbar. */
  status: string;
  counts: Record<string, number>;
  notes: string[];
  onDismiss?: () => void;
}

/** The aggregate line under the prompt toolbar: what the review did, in one row. */
export function CoordinationSummary({
  status,
  counts,
  notes,
  onDismiss,
}: CoordinationSummaryProps) {
  const bits: string[] = [];
  if (counts.set) bits.push(`${counts.set} knob(s) set`);
  if (counts.proposed) bits.push(`${counts.proposed} proposed`);
  if (counts.mismatch) bits.push(`${counts.mismatch} mismatch(es)`);
  if (counts.ok) bits.push(`${counts.ok} already correct`);
  return (
    <div className="vi-input-callout" role="status" style={{ marginBottom: "0.6rem" }}>
      <div>
        <span
          className={coordinationClass(status as never)}
          style={{ marginRight: "0.4rem" }}
        >
          coordination
        </span>
        {bits.length ? bits.join(" · ") : "reviewed — nothing to change"}
      </div>
      {notes.map((n, i) => (
        <div key={i} style={{ opacity: 0.8, marginTop: "0.2rem" }}>
          {n}
        </div>
      ))}
      {onDismiss && (
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          style={{ marginTop: "0.35rem" }}
          onClick={onDismiss}
        >
          ✕ Dismiss
        </button>
      )}
    </div>
  );
}
