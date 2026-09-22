import { describe, it, expect } from "vitest";
import { uploadResponseToUploadedFileRef } from "./UploadNormalize";

// Phase 6 — the client references an opaque id, never a raw server path.
describe("uploadResponseToUploadedFileRef (Phase 6 id model)", () => {
  it("prefers a server-issued opaque id over the path", () => {
    const ref = uploadResponseToUploadedFileRef({
      id: "f_abc123",
      path: "/srv/uploads/session/x.png",
      name: "x.png",
    });
    expect(ref.id).toBe("f_abc123");
    expect(ref.name).toBe("x.png");
    // Crucially, the raw path is NOT surfaced as the identity.
    expect(ref.id).not.toContain("/srv/");
  });

  it("falls back to the path as the handle when no id is present (interim)", () => {
    const ref = uploadResponseToUploadedFileRef({ path: "/srv/uploads/y.mp3" });
    expect(ref.id).toBe("/srv/uploads/y.mp3");
    expect(ref.name).toBe("y.mp3");
  });

  it("throws when the response has neither id nor path/url", () => {
    expect(() => uploadResponseToUploadedFileRef({ nope: true })).toThrow();
  });
});
