import React, { type ReactNode } from "react";
import styles from "../UtilityPage.module.css";
import { formatUnknown } from "./ResultHelpers";
import ResultActions from "./ResultActions";
import type { z } from "zod";
import type { BatchFileResultSchema, ChainStepResultSchema } from "../../../schemas";

type ChainStep = z.infer<typeof ChainStepResultSchema>;
type BatchRow = z.infer<typeof BatchFileResultSchema>;

// Phase 7 — render runChain's ChainStepResult[]: per-step ok/error, which step failed,
// and that step's error. runChain stops at the first failure, so any steps after the
// errored one are absent — shown as a "stopped here" note.
export default function ChainOutput({ steps }: { steps: ChainStep[] }) {
  const failedAt = steps.findIndex((s) => !s.ok);

  return (
    <div className={styles.prettyResult}>
      <ResultActions value={steps} label="chain" />
      <ol className={styles.chainList}>
        {steps.map((step, i) => (
          <li
            key={`${step.pageKey}-${i}`}
            className={`${styles.chainStep} ${
              step.ok ? styles.chainStepOk : styles.chainStepError
            }`}
          >
            <div className={styles.chainStepHeader}>
              <span className={styles.chainBadge}>
                {step.ok ? "✓ ok" : "✕ error"}
              </span>
              <strong>
                Step {i + 1}: {step.pageKey}
              </strong>
            </div>

            {step.ok ? (
              <details className={styles.rawResultDetails}>
                <summary>Result</summary>
                <pre className={styles.resultText}>{formatUnknown(step.data)}</pre>
              </details>
            ) : (
              <pre className={`${styles.resultText} ${styles.resultError}`}>
                {step.error ?? "Step failed."}
              </pre>
            )}
          </li>
        ))}
      </ol>

      {failedAt >= 0 && (
        <p className={styles.muted}>
          Chain stopped at step {failedAt + 1}; later steps did not run.
        </p>
      )}
    </div>
  );
}

// k65 — the other per-row run view: a Run over SEVERAL selected files. The Tool
// Console used to take the first file and drop the rest without a word; a
// multi-file selection now runs once per file and every file gets a row here —
// its own status and its own result, rendered by the tool's normal output
// renderer (passed in as `render`, so this component needs to know nothing about
// operations, and ExecutionOutput can delegate here without importing itself).
//
// Unlike a chain, a failing row does NOT stop the run: one bad file must not cost
// the user the other nine. Lives beside ChainOutput because it shares its list
// chrome and its "one row per unit of work" job.
function batchBadge(status: BatchRow["status"]): string {
  if (status === "ok") return "✓ ok";
  if (status === "error") return "✕ error";
  return status === "running" ? "… running" : "· queued";
}

export function BatchOutput({
  rows,
  render,
}: {
  rows: BatchRow[];
  render: (data: unknown) => ReactNode;
}) {
  const done = rows.filter((r) => r.status === "ok").length;
  const failed = rows.filter((r) => r.status === "error").length;
  const left = rows.filter((r) => r.status === "pending" || r.status === "running").length;

  return (
    <div className={styles.prettyResult}>
      <ResultActions value={rows} label="batch" />

      <p className={styles.muted}>
        {rows.length} files — {done} ok
        {failed > 0 ? `, ${failed} failed` : ""}
        {left > 0 ? `, ${left} to go` : ""}
      </p>

      <ol className={styles.chainList}>
        {rows.map((row, i) => (
          <li
            key={`${row.batchFile}-${i}`}
            className={`${styles.chainStep} ${
              row.status === "error"
                ? styles.chainStepError
                : row.status === "ok"
                  ? styles.chainStepOk
                  : ""
            }`}
          >
            <div className={styles.chainStepHeader}>
              <span className={styles.chainBadge}>{batchBadge(row.status)}</span>
              <strong>{row.batchFile}</strong>
            </div>

            {row.status === "ok" && render(row.data)}

            {row.status === "error" && (
              <pre className={`${styles.resultText} ${styles.resultError}`}>
                {row.error ?? "This file failed."}
              </pre>
            )}

            {row.status === "running" && <p className={styles.muted}>Working...</p>}
            {row.status === "pending" && <p className={styles.muted}>Waiting...</p>}
          </li>
        ))}
      </ol>

      {left > 0 && (
        <p className={styles.muted}>
          Files run one at a time; cancelling stops the ones that haven&apos;t started.
        </p>
      )}
    </div>
  );
}
