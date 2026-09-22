import React, { type ReactNode } from "react";
import type { Operation, PageSpec } from "../../../pages/pageSpec";
import { BatchFileResultArraySchema, ChainStepResultArraySchema } from "../../../schemas";
import { outputRenderers, GenericOutput } from "./outputRegistry";
import ChainOutput, { BatchOutput } from "./ChainOutput";

interface ExecutionOutputProps {
  result: unknown;
  spec: PageSpec;
  operation: Operation | "any";
}

/** The operation whose output shape we expect: the explicit selection, else the spec's
 *  first declared `produces`. */
function effectiveOperation(
  spec: PageSpec,
  operation: Operation | "any",
): Operation | null {
  if (operation !== "any") return operation;
  const produces = Array.isArray(spec.produces) ? spec.produces[0] : spec.produces;
  return produces ?? null;
}

/** One tool result, rendered as this spec/operation declares. */
function renderOne(
  result: unknown,
  spec: PageSpec,
  operation: Operation | "any",
): ReactNode {
  // Chain pages return ChainStepResult[] — render per-step status.
  if (spec.key.startsWith("chain:")) {
    const parsed = ChainStepResultArraySchema.safeParse(result);
    if (parsed.success) return <ChainOutput steps={parsed.data} />;
    // not chain-shaped (shouldn't happen) → fall through to generic
  }

  const op = effectiveOperation(spec, operation);
  const renderer = op ? outputRenderers.get(op) : undefined;

  return renderer ? renderer(result) : <GenericOutput result={result} />;
}

// Phase 7 — dispatch by DECLARED output (registry), not by sniffing the response shape.
export default function ExecutionOutput({
  result,
  spec,
  operation,
}: ExecutionOutputProps) {
  // k65 — a BATCH run (one row per selected file, live-updated) comes in as
  // BatchFileResult[] whatever the tool is; each row's own data is then rendered by
  // the tool's normal renderer. Checked FIRST and by the `batchFile` marker, so a
  // chain's steps (no such field) and ordinary object results are unaffected.
  const batch = BatchFileResultArraySchema.safeParse(result);
  if (batch.success && batch.data.length) {
    return (
      <BatchOutput
        rows={batch.data}
        render={(data) => renderOne(data, spec, operation)}
      />
    );
  }

  return <>{renderOne(result, spec, operation)}</>;
}
