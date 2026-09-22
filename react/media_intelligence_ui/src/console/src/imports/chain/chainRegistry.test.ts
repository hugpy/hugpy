import { describe, it, expect, beforeAll } from "vitest";
import { registerPage } from "../pages/pagesRegistry";
import { validateChain } from "./chainRegistry";
import type { ChainSpec } from "./chainSpec";
import type { PageSpec } from "../pages/pageSpec";

function page(over: Partial<PageSpec> & Pick<PageSpec, "key" | "accepts" | "produces">): PageSpec {
  return {
    title: over.key,
    category: "t",
    path: `/${over.key}`,
    fields: [],
    ...over,
  };
}

// Registry is module-singleton per test file — register fixtures once.
beforeAll(() => {
  registerPage(page({ key: "c/textout", accepts: ["text"], produces: "summarize" })); // → text
  registerPage(page({ key: "c/wantstext", accepts: ["text"], produces: "summarize" }));
  registerPage(page({ key: "c/video", accepts: ["video"], produces: "download" })); // → video
  registerPage(page({ key: "c/multi", accepts: ["text"], produces: ["summarize", "keywords"] }));
});

function chain(steps: ChainSpec["steps"]): ChainSpec {
  return { key: "c1", title: "c1", category: "t", steps };
}

describe("validateChain (Phase 9)", () => {
  it("throws on an empty chain", () => {
    expect(() => validateChain(chain([]))).toThrow(/no steps/i);
  });

  it("accepts a compatible pipeline (text → text)", () => {
    expect(() =>
      validateChain(chain([{ pageKey: "c/textout" }, { pageKey: "c/wantstext" }])),
    ).not.toThrow();
  });

  it("throws on an accepts/produces mismatch (video → text-only)", () => {
    expect(() =>
      validateChain(chain([{ pageKey: "c/video" }, { pageKey: "c/wantstext" }])),
    ).toThrow(/accepts|produces/i);
  });

  it("throws when a multi-op step is not pinned via overrides.__op", () => {
    expect(() =>
      validateChain(chain([{ pageKey: "c/multi" }, { pageKey: "c/wantstext" }])),
    ).toThrow(/multiple ops/i);
  });

  it("accepts a pinned multi-op step", () => {
    expect(() =>
      validateChain(
        chain([
          { pageKey: "c/multi", overrides: { __op: "summarize" } },
          { pageKey: "c/wantstext" },
        ]),
      ),
    ).not.toThrow();
  });
});
