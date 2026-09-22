import { describe, it, expect } from "vitest";
import {
  PageSpecSchema,
  MediaKindSchema,
  OperationSchema,
  isRecord,
  asFileArray,
  parseUploadResponse,
} from "./schemas";

// Phase 4 acceptance — the trust boundary parses, it does not assert.
describe("schemas (Phase 4 boundaries)", () => {
  const validPage = {
    key: "image/analyze",
    title: "Image Analysis",
    category: "image",
    path: "/image/analyze",
    fields: [{ name: "image_path", label: "Image", kind: "file" }],
    accepts: ["image"],
    produces: "analyze",
  };

  it("accepts a well-formed page spec", () => {
    expect(PageSpecSchema.safeParse(validPage).success).toBe(true);
  });

  it("rejects a malformed registry payload with structured issues (no crash)", () => {
    // missing required keys + wrong enum value
    const bad = { key: "x", title: "x", accepts: ["hologram"], produces: 7 };
    const r = PageSpecSchema.safeParse(bad);
    expect(r.success).toBe(false);
    if (!r.success) {
      expect(r.error.issues.length).toBeGreaterThan(0);
    }
  });

  it("array parse: one bad page fails the whole batch (degrade, don't inject garbage)", () => {
    const r = PageSpecSchema.array().safeParse([validPage, { nope: true }]);
    expect(r.success).toBe(false);
  });

  it("enum schemas reject unknown DOM values", () => {
    expect(MediaKindSchema.safeParse("image").success).toBe(true);
    expect(MediaKindSchema.safeParse("hologram").success).toBe(false);
    expect(OperationSchema.safeParse("summarize").success).toBe(true);
    expect(OperationSchema.safeParse("teleport").success).toBe(false);
  });

  it("isRecord guards objects only", () => {
    expect(isRecord({})).toBe(true);
    expect(isRecord([])).toBe(false);
    expect(isRecord(null)).toBe(false);
    expect(isRecord("x")).toBe(false);
  });

  it("asFileArray keeps only File instances, never throws on junk", () => {
    const f = new File(["a"], "a.txt");
    expect(asFileArray([f, "not-a-file", 3, null])).toEqual([f]);
    expect(asFileArray("nope")).toEqual([]);
    expect(asFileArray(undefined)).toEqual([]);
  });

  it("parseUploadResponse returns {} for non-objects, lens for objects", () => {
    expect(parseUploadResponse("nope")).toEqual({});
    expect(parseUploadResponse({ path: "/srv/x.png" }).path).toBe("/srv/x.png");
  });
});
