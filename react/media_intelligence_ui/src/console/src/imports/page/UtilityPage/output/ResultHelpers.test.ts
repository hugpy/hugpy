import { describe, it, expect } from "vitest";
import {
  isRecord,
  unwrapResult,
  asStringArray,
  hasPdfReportShape,
  hasPdfTextShape,
  hasTranscriptionShape,
  hasKeywordShape,
  formatUnknown,
} from "./ResultHelpers";

// Phase 0 smoke test — proves the Vitest harness runs hugpy logic green, and seeds
// Phase 9's pure-logic coverage for the shape predicates ExecutionOutput dispatches on.
describe("ResultHelpers (Phase 0 smoke / Phase 9 seed)", () => {
  it("isRecord distinguishes plain objects from arrays/null", () => {
    expect(isRecord({})).toBe(true);
    expect(isRecord([])).toBe(false);
    expect(isRecord(null)).toBe(false);
  });

  it("unwrapResult peels a { result } envelope once", () => {
    expect(unwrapResult({ result: 7 })).toBe(7);
    expect(unwrapResult(7)).toBe(7);
  });

  it("asStringArray coerces via String() then drops falsy strings", () => {
    // NOTE latent quirk: String(null) === "null" (truthy), so null survives as the
    // literal "null"; only "" is dropped. Pinned here for Phase 9 to decide on.
    expect(asStringArray(["a", 1, "", null])).toEqual(["a", "1", "null"]);
    expect(asStringArray("nope")).toEqual([]);
  });

  it("shape predicates match their target shapes and reject others", () => {
    expect(hasPdfReportShape({ pages: [] })).toBe(true);
    expect(hasPdfTextShape([{ page_num: 1, text: "x" }])).toBe(true);
    expect(hasTranscriptionShape({ text: "x", segments: [] })).toBe(true);
    expect(hasKeywordShape({ primary: [] })).toBe(true);

    // A transcription payload must NOT also satisfy the pdf-report predicate.
    expect(hasPdfReportShape({ text: "x", segments: [] })).toBe(false);
  });

  it("formatUnknown stringifies non-strings, passes strings through", () => {
    expect(formatUnknown("hi")).toBe("hi");
    expect(formatUnknown(null)).toBe("");
    expect(formatUnknown({ a: 1 })).toContain("\"a\": 1");
  });
});
