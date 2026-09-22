/**
 * or-k18 — the steward report: the system checking itself (k113c).
 *
 * GET /api/oracle/steward is report-only. The "Re-run" button POSTs, which
 * re-audits AND applies bounded rebalancing to the live selector — the button
 * says so, because "re-run" that silently changes policy would be a lie.
 *
 * Findings are grouped by severity (alarm → warn → info). A `matrix_stale`
 * finding is promoted to its own banner regardless of severity: a stale
 * routing matrix means every selection below is reasoning from old evidence.
 */
import { useMemo } from "react";

import type { StewardFinding, StewardReport } from "./oracleTypes";
import type { StewardApi } from "./useOracle";
import { GapNote, RawJson, Tag, ago, when } from "./oracleShared";

const ORDER: Array<"alarm" | "warn" | "info"> = ["alarm", "warn", "info"];

function bucket(findings: StewardFinding[]): Record<string, StewardFinding[]> {
  const out: Record<string, StewardFinding[]> = { alarm: [], warn: [], info: [], other: [] };
  for (const f of findings) {
    const k = ORDER.includes(f.severity as "alarm") ? f.severity : "other";
    out[k].push(f);
  }
  return out;
}

function sevOf(f: StewardFinding): "alarm" | "warn" | "info" {
  return (ORDER.includes(f.severity as "alarm") ? f.severity : "info") as "alarm" | "warn" | "info";
}

/** The BODY of one finding row. The `<li key>` wrapper stays at the call site —
 *  this project's scoped tsconfig rejects a `key` prop on a locally-declared
 *  component (the JobRow idiom in ActiveProcessesStation). */
function FindingBody({ f }: { f: StewardFinding }) {
  return (
    <>
      <div className="vi-oracle-finding-head">
        <Tag kind={sevOf(f)}>{f.severity}</Tag>
        <Tag>{f.kind}</Tag>
        {f.capability ? <code>{f.capability}</code> : <span className="vi-knob-hint">fleet-wide</span>}
      </div>
      <div style={{ marginTop: "0.25rem" }}>{f.message}</div>
      {f.action ? <p className="vi-oracle-finding-action">action: {f.action}</p> : null}
      {f.evidence && Object.keys(f.evidence).length ? <RawJson label="evidence" value={f.evidence} /> : null}
    </>
  );
}

export function StewardPanel({ api, embedded }: { api: StewardApi; embedded?: StewardReport | null }) {
  const report = api.report;
  const findings = useMemo(() => report?.findings ?? [], [report]);
  const groups = useMemo(() => bucket(findings), [findings]);
  const stale = findings.filter((f) => f.kind === "matrix_stale");
  const busy = api.busy != null;

  return (
    <section className="vi-oracle-panel" aria-label="Steward report">
      <h3>
        Steward
        <span className="vi-oracle-sub">the selector audited from its own ledger — GET /api/oracle/steward</span>
      </h3>

      <div className="vi-oracle-row">
        <button type="button" className="vi-btn vi-btn-sm" disabled={busy} onClick={api.refresh}>
          {api.busy === "get" ? "Reading…" : "Refresh report"}
        </button>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-accent"
          disabled={busy}
          onClick={api.rerun}
          title="POST /api/oracle/steward — re-audits AND applies bounded rebalancing to the live selection policy"
        >
          {api.busy === "post" ? "Re-running…" : "Re-run + apply rebalancing"}
        </button>
        <span className="vi-knob-hint">
          last run (server): <code>{when(report?.at)}</code>
          {api.fetchedAt != null ? <> · fetched {ago(api.fetchedAt)} (browser clock)</> : null}
        </span>
      </div>

      <GapNote gap={api.gap} />

      {report ? (
        <>
          <div className="vi-oracle-row" style={{ marginTop: "0.6rem" }}>
            <Tag kind={report.ok ? "ok" : "alarm"}>{report.ok ? "no alarms" : "alarms present"}</Tag>
            {report.applied ? <Tag kind="accent">policy applied this call</Tag> : <Tag>report only</Tag>}
            {report.policy_changed ? <Tag kind="warn">policy changed</Tag> : null}
            {typeof report.ledger_rows === "number" ? <Tag>ledger rows: {report.ledger_rows}</Tag> : null}
            <Tag>
              {groups.alarm.length} alarm · {groups.warn.length} warn · {groups.info.length} info
            </Tag>
          </div>
          {report.summary ? <p className="vi-oracle-note">{report.summary}</p> : null}

          {stale.length ? (
            <div className="vi-oracle-stale" role="alert">
              <strong>Routing matrix is stale.</strong> Selections below reason from old evidence.
              <ul className="vi-oracle-reasons">
                {stale.map((f, i) => (
                  <li key={i}>
                    {f.message}
                    {f.action ? <> — {f.action}</> : null}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {ORDER.map((sev) =>
            groups[sev].length ? (
              <div key={sev}>
                <h4>
                  {sev} ({groups[sev].length})
                </h4>
                <ul className="vi-oracle-findings">
                  {groups[sev].map((f, i) => (
                    <li key={`${sev}-${f.kind}-${f.capability ?? ""}-${i}`} className={`vi-oracle-finding vi-oracle-finding-${sevOf(f)}`}>
                      <FindingBody f={f} />
                    </li>
                  ))}
                </ul>
              </div>
            ) : null,
          )}
          {groups.other.length ? (
            <div>
              <h4>unclassified severity ({groups.other.length})</h4>
              <ul className="vi-oracle-findings">
                {groups.other.map((f, i) => (
                  <li key={`o-${i}`} className="vi-oracle-finding vi-oracle-finding-info">
                    <FindingBody f={f} />
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {!findings.length ? (
            <p className="vi-oracle-note">The steward returned zero findings — a clean fleet normally still says so with numbers; treat an empty list as "nothing audited".</p>
          ) : null}

          {report.selection_policy ? <RawJson label="selection policy (live)" value={report.selection_policy} /> : null}
          {report.policy_after ? <RawJson label="policy after this call" value={report.policy_after} /> : null}
          <RawJson label="steward report json" value={report} />
        </>
      ) : !api.gap ? (
        <p className="vi-oracle-note">Reading the steward…</p>
      ) : null}

      {embedded ? (
        <>
          <h4>steward snapshot embedded in the selected run</h4>
          <p className="vi-oracle-note">
            This is the report the run itself recorded at the time (manifest.dag.steward), not the live one above.
            {embedded.summary ? <> Summary: {embedded.summary}</> : null}
          </p>
          <RawJson label="manifest.dag.steward" value={embedded} />
        </>
      ) : null}
    </section>
  );
}
