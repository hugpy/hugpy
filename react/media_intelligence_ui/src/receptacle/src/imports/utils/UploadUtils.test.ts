import { describe, it, expect } from "vitest";
import { extractUploadedFiles } from "./UploadUtils";

// Phase 9 — the defensive response-shape permutations extractUploadedFiles handles.
describe("extractUploadedFiles", () => {
  it("handles a bare array of items", () => {
    const out = extractUploadedFiles([{ path: "/srv/a.png", name: "a.png" }]);
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe("/srv/a.png");
    expect(out[0]?.name).toBe("a.png");
  });

  it("handles { files: [...] }, { uploadedFiles: [...] }, { results: [...] }", () => {
    expect(extractUploadedFiles({ files: [{ path: "/p1" }] })).toHaveLength(1);
    expect(extractUploadedFiles({ uploadedFiles: [{ path: "/p2" }] })).toHaveLength(1);
    expect(extractUploadedFiles({ results: [{ path: "/p3" }] })).toHaveLength(1);
  });

  it("handles a single object", () => {
    const out = extractUploadedFiles({ path: "/solo" });
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe("/solo");
  });

  it("prefers a server id over the path", () => {
    const out = extractUploadedFiles([{ id: "fid-1", path: "/srv/x" }]);
    expect(out[0]?.id).toBe("fid-1");
  });

  it("derives the name from the path tail when absent", () => {
    const out = extractUploadedFiles([{ path: "/srv/deep/clip.mp4" }]);
    expect(out[0]?.name).toBe("clip.mp4");
  });

  it("drops items with no usable id/path", () => {
    expect(extractUploadedFiles([{ nope: 1 }, { path: "/ok" }])).toHaveLength(1);
    expect(extractUploadedFiles(null)).toEqual([]);
  });
});
