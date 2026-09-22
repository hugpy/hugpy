import type { ChainStepResult } from "../chain/chainRuntime";

// Phase 8 — run lifecycle as observable state (queue over callbacks). The store lives at
// MODULE level so runs survive UtilityPage unmount (the page is keyed by spec.key and
// remounts on every tool switch). The UI reads this via useSyncExternalStore; an async
// submit writes here regardless of whether its originating component is still mounted, so
// switching away and back shows the (possibly still-running, possibly completed) run.

export type RunStatus = "running" | "ok" | "error" | "cancelled";

/**
 * One file of a BATCH run — a tool whose scalar file field was pointed at several
 * selected files runs once per file (operator ask 2026-08-04, k65). It IS a
 * ChainStepResult (so the existing live-progress plumbing, `setRunSteps` →
 * `RunRecord.steps` → ExecutionOutput, carries it unchanged) plus the file it ran
 * for and a status that can also be pending/running — a chain step only ever
 * appears once it has finished, a batch row is on screen before it starts.
 */
export type BatchFileStatus = "pending" | "running" | "ok" | "error";

export interface BatchFileResult extends ChainStepResult {
  /** Name of the uploaded file this row ran for (the row's label). */
  batchFile: string;
  status: BatchFileStatus;
}

export interface RunRecord {
  id: string;
  specKey: string;
  status: RunStatus;
  /** Final result for a non-chain run (on "ok"). */
  result?: unknown;
  /** Per-step results for a chain run — updated live as steps complete. */
  steps?: ChainStepResult[];
  /** Error message (on "error"). */
  error?: string;
  startedAt: number;
  endedAt?: number;
}

const runs = new Map<string, RunRecord>();
const latestBySpec = new Map<string, string>();
const controllers = new Map<string, AbortController>();
const listeners = new Set<() => void>();

function emit(): void {
  for (const l of listeners) l();
}

export function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

let _seq = 0;
function nextId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    _seq += 1;
    return `run_${_seq}`;
  }
}

// Records are replaced immutably so getSnapshot returns a stable reference for specs that
// didn't change — useSyncExternalStore relies on Object.is to skip re-renders.
function put(rec: RunRecord): void {
  runs.set(rec.id, rec);
  emit();
}

function running(id: string): RunRecord | undefined {
  const r = runs.get(id);
  return r && r.status === "running" ? r : undefined;
}

export function beginRun(specKey: string, controller: AbortController): string {
  const id = nextId();
  controllers.set(id, controller);
  latestBySpec.set(specKey, id);
  put({ id, specKey, status: "running", startedAt: Date.now() });
  return id;
}

// Also accepted while CANCELLED: a batch run writes one final snapshot after the
// user hits Cancel (k65), so the rows that were mid-flight stop reading "running"
// forever. A finished run (ok/error) is never rewritten.
export function setRunSteps(id: string, steps: ChainStepResult[]): void {
  const r = runs.get(id);
  if (r && (r.status === "running" || r.status === "cancelled")) {
    put({ ...r, steps: [...steps] });
  }
}

export function completeRun(
  id: string,
  result: unknown,
  steps?: ChainStepResult[],
): void {
  const r = running(id);
  if (!r) return; // already cancelled/failed — don't override
  controllers.delete(id);
  const nextSteps = steps ?? r.steps;
  put({
    ...r,
    status: "ok",
    result,
    ...(nextSteps ? { steps: nextSteps } : {}),
    endedAt: Date.now(),
  });
}

export function failRun(id: string, error: string): void {
  const r = running(id);
  if (!r) return;
  controllers.delete(id);
  put({ ...r, status: "error", error, endedAt: Date.now() });
}

export function cancelRun(id: string): void {
  const r = running(id);
  controllers.get(id)?.abort();
  controllers.delete(id);
  if (r) put({ ...r, status: "cancelled", endedAt: Date.now() });
}

export function getLatestForSpec(specKey: string): RunRecord | undefined {
  const id = latestBySpec.get(specKey);
  return id ? runs.get(id) : undefined;
}
