import type { AppError } from "./client";

// Phase 10 — observability at the transport boundary. Every request emits one structured
// event (request id, target, duration, outcome, + optional spec/op from the caller). A
// pluggable sink defaults to structured console logging and can be swapped for Sentry or
// any reporter via setTelemetrySink. In-memory metrics aggregate counts/error-rate/latency
// for a debug surface.

export type Outcome = "ok" | AppError["kind"];

export interface RequestEvent {
  requestId: string;
  url: string;
  method: string;
  outcome: Outcome;
  durationMs: number;
  status?: number;
  /** Caller context — the transport doesn't know these, so they're passed in. */
  specKey?: string;
  operation?: string;
  /** Server/Sentry correlation id, when present. */
  traceId?: string;
}

export interface TelemetrySink {
  record(event: RequestEvent): void;
}

// ---- metrics ----
const countByOp = new Map<string, number>();
const errorByKind = new Map<Outcome, number>();
const latencyByOp = new Map<string, number[]>();

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)] ?? 0;
}

export const metrics = {
  snapshot() {
    const latency: Record<string, { p50: number; p95: number; n: number }> = {};
    for (const [op, samples] of latencyByOp) {
      latency[op] = {
        p50: Math.round(percentile(samples, 50)),
        p95: Math.round(percentile(samples, 95)),
        n: samples.length,
      };
    }
    return {
      countByOp: Object.fromEntries(countByOp),
      errorByKind: Object.fromEntries(errorByKind),
      latency,
    };
  },
  reset() {
    countByOp.clear();
    errorByKind.clear();
    latencyByOp.clear();
  },
};

// ---- sink (default: structured console; swap for Sentry via setTelemetrySink) ----
const consoleSink: TelemetrySink = {
  record(event) {
    const line = { evt: "hugpy.request", ...event };
    if (event.outcome === "ok") {
      console.debug(line);
    } else {
      console.warn(line);
    }
  },
};

let sink: TelemetrySink = consoleSink;

export function setTelemetrySink(next: TelemetrySink): void {
  sink = next;
}

export function recordRequest(event: RequestEvent): void {
  const op = event.operation ?? event.specKey ?? event.method;
  countByOp.set(op, (countByOp.get(op) ?? 0) + 1);
  if (event.outcome !== "ok") {
    errorByKind.set(event.outcome, (errorByKind.get(event.outcome) ?? 0) + 1);
  }
  const samples = latencyByOp.get(op) ?? [];
  samples.push(event.durationMs);
  latencyByOp.set(op, samples);

  try {
    sink.record(event);
  } catch {
    // a telemetry failure must never break a request
  }
}
