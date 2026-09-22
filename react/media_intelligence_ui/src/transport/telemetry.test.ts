import { describe, it, expect, beforeEach } from "vitest";
import {
  recordRequest,
  metrics,
  setTelemetrySink,
  type RequestEvent,
} from "./telemetry";

function ev(over: Partial<RequestEvent>): RequestEvent {
  return {
    requestId: "r",
    url: "/x",
    method: "POST",
    outcome: "ok",
    durationMs: 10,
    ...over,
  };
}

describe("telemetry (Phase 10)", () => {
  beforeEach(() => metrics.reset());

  it("routes events to a pluggable sink (Sentry-ready)", () => {
    const seen: RequestEvent[] = [];
    setTelemetrySink({ record: (e) => seen.push(e) });
    recordRequest(ev({ operation: "transcribe" }));
    expect(seen).toHaveLength(1);
    expect(seen[0]?.operation).toBe("transcribe");
    setTelemetrySink({ record: () => {} }); // detach for other tests
  });

  it("counts by operation and error-rate by kind", () => {
    setTelemetrySink({ record: () => {} });
    recordRequest(ev({ operation: "summarize", outcome: "ok" }));
    recordRequest(ev({ operation: "summarize", outcome: "server" }));
    recordRequest(ev({ operation: "summarize", outcome: "timeout" }));
    const snap = metrics.snapshot();
    expect(snap.countByOp.summarize).toBe(3);
    expect(snap.errorByKind.server).toBe(1);
    expect(snap.errorByKind.timeout).toBe(1);
  });

  it("computes p50/p95 latency per op", () => {
    setTelemetrySink({ record: () => {} });
    for (const ms of [10, 20, 30, 40, 100]) {
      recordRequest(ev({ operation: "ocr", durationMs: ms }));
    }
    const snap = metrics.snapshot();
    expect(snap.latency.ocr.n).toBe(5);
    expect(snap.latency.ocr.p95).toBe(100);
    expect(snap.latency.ocr.p50).toBeGreaterThanOrEqual(20);
  });

  it("a throwing sink never breaks recording", () => {
    setTelemetrySink({
      record: () => {
        throw new Error("sink down");
      },
    });
    expect(() => recordRequest(ev({}))).not.toThrow();
    setTelemetrySink({ record: () => {} });
  });
});
