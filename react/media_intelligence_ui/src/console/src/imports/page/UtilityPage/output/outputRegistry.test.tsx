import { describe, it, expect } from "vitest";
import { type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { outputRenderers } from "./outputRegistry";
import { ChainStepResultArraySchema } from "../../../schemas";

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>);
}

describe("output registry (Phase 7 dispatch by declared op)", () => {
  it("registers renderers keyed by operation", () => {
    expect(outputRenderers.has("transcribe")).toBe(true);
    expect(outputRenderers.has("keywords")).toBe(true);
    // ops without a dedicated renderer fall back to GenericOutput in ExecutionOutput
    expect(outputRenderers.has("metadata")).toBe(false);
  });

  it("transcribe renderer renders the transcription shape", () => {
    const r = outputRenderers.get("transcribe");
    const out = html(r!({ text: "hello world", segments: [] }));
    expect(out).toContain("hello world");
    expect(out).not.toMatch(/unexpected shape/i);
  });

  it("transcribe renderer shows a TYPED error on an unexpected shape (no silent dump)", () => {
    const r = outputRenderers.get("transcribe");
    const out = html(r!({ totally: "different" }));
    expect(out).toMatch(/unexpected shape/i);
    expect(out).toContain("transcribe");
  });

  it("keywords renderer errors typed on a non-keyword shape", () => {
    const r = outputRenderers.get("keywords");
    const out = html(r!({ nope: 1 }));
    expect(out).toMatch(/unexpected shape/i);
  });
});

describe("chain results (Phase 7)", () => {
  it("validates a ChainStepResult[] and identifies the failed step", () => {
    const steps = [
      { pageKey: "a/x", ok: true, data: { text: "ok" } },
      { pageKey: "b/y", ok: false, data: null, error: "boom at step 2" },
    ];
    const parsed = ChainStepResultArraySchema.safeParse(steps);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.findIndex((s) => !s.ok)).toBe(1);
    }
  });

  it("rejects a non-chain payload", () => {
    expect(ChainStepResultArraySchema.safeParse({ not: "an array" }).success).toBe(
      false,
    );
  });
});
