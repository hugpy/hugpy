import { describe, it, expect, beforeAll } from "vitest";
import { registerPage } from "../pages/pagesRegistry";
import { extractText, extractPaths, projectResultToInput } from "./chainRuntime";
import type { PageSpec } from "../pages/pageSpec";

describe("chainRuntime extraction (Phase 9)", () => {
  describe("extractText", () => {
    it("returns strings directly", () => {
      expect(extractText("hello")).toBe("hello");
    });
    it("pulls the first known text-bearing key", () => {
      expect(extractText({ summary: "S" })).toBe("S");
      expect(extractText({ transcript: "T" })).toBe("T");
    });
    it("recurses into result and joins results arrays", () => {
      expect(extractText({ result: { text: "deep" } })).toBe("deep");
      expect(extractText({ results: [{ text: "a" }, { text: "b" }] })).toBe("a\n\nb");
    });
    it("returns '' for unknown shapes", () => {
      expect(extractText({ nope: 1 })).toBe("");
      expect(extractText(42)).toBe("");
    });
  });

  describe("extractPaths", () => {
    it("reads paths[] (strings only)", () => {
      expect(extractPaths({ paths: ["a", 1, "b"] })).toEqual(["a", "b"]);
    });
    it("reads a single path", () => {
      expect(extractPaths({ path: "x" })).toEqual(["x"]);
    });
    it("recurses into result", () => {
      expect(extractPaths({ result: { paths: ["y"] } })).toEqual(["y"]);
    });
    it("returns [] for unknown shapes", () => {
      expect(extractPaths({})).toEqual([]);
    });
  });
});

describe("projectResultToInput (Phase 9)", () => {
  beforeAll(() => {
    registerPage({
      key: "proj/text",
      title: "t",
      category: "t",
      path: "/proj/text",
      fields: [],
      accepts: ["text"],
      produces: "summarize", // → text kind
    } as PageSpec);
    registerPage({
      key: "proj/audio",
      title: "a",
      category: "t",
      path: "/proj/audio",
      fields: [],
      accepts: ["audio"],
      produces: "extract_audio", // → audio kind (file)
    } as PageSpec);
  });

  it("projects a text-producing step to a text input", () => {
    const next = projectResultToInput("proj/text", { text: "carry" }, {});
    expect(next.inputMode).toBe("text");
    expect(next.text).toBe("carry");
  });

  it("projects a file-producing step to selected ids", () => {
    const next = projectResultToInput("proj/audio", { paths: ["/srv/out.wav"] }, {});
    expect(next.inputMode).toBe("file");
    expect(next.selectedIds).toEqual(["/srv/out.wav"]);
    expect(next.uploadedFiles[0]?.id).toBe("/srv/out.wav");
  });
});
