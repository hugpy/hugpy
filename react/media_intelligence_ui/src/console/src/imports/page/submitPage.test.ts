import { describe, it, expect, vi, beforeEach } from "vitest";
import type { MediaInputValue, FieldSpec, PageSpec } from "../pages/pageSpec";

// Mock the transport so submitPage's request is captured, not sent.
const requestMock = vi.fn();
vi.mock("./../../../../transport/client", () => ({
  request: (...args: unknown[]) => requestMock(...args),
}));

import {
  resolveSourcedField,
  buildSourcePayload,
  selectedUploadedFiles,
  selectedUploadedIds,
  isFieldVisible,
  submitPage,
  batchFilesFor,
  pinInputToFile,
} from "./submitPage";

function input(over: Partial<MediaInputValue> = {}): MediaInputValue {
  return {
    inputMode: "text",
    text: "",
    url: "",
    files: [],
    uploadedFiles: [],
    selectedIds: [],
    ...over,
  };
}

/** Two uploaded files, both ticked — the k65 batch case. */
function multiFileInput(): MediaInputValue {
  return input({
    inputMode: "file",
    uploadedFiles: [
      { name: "a", id: "id-a" },
      { name: "b", id: "id-b" },
    ],
    selectedIds: ["id-a", "id-b"],
  });
}

function field(over: Partial<FieldSpec> = {}): FieldSpec {
  return { name: "f", label: "F", kind: "text", ...over };
}

function page(over: Partial<PageSpec> = {}): PageSpec {
  return {
    key: "t/x",
    title: "T",
    category: "t",
    path: "/t/x",
    fields: [],
    accepts: ["text"],
    produces: "summarize",
    ...over,
  };
}

describe("resolveSourcedField (Phase 9 — all branches)", () => {
  it("text: ok when present, empty when blank", () => {
    expect(resolveSourcedField(field({ source: "text" }), input({ text: "hi" })).status).toBe("ok");
    expect(resolveSourcedField(field({ source: "text" }), input({ text: "  " })).status).toBe("empty");
  });

  it("url: ok when present, empty when blank", () => {
    expect(resolveSourcedField(field({ source: "url" }), input({ url: "some-url" })).status).toBe("ok");
    expect(resolveSourcedField(field({ source: "url" }), input()).status).toBe("empty");
  });

  it("selectedIds: ok with selected ids", () => {
    const r = resolveSourcedField(
      field({ source: "selectedIds" }),
      input({
        inputMode: "file",
        uploadedFiles: [{ name: "a", id: "id-a" }],
        selectedIds: ["id-a"],
      }),
    );
    expect(r.status).toBe("ok");
    expect(r.status === "ok" && r.value).toEqual(["id-a"]);
  });

  it("selectedIds: falls back to local file names when nothing uploaded-selected", () => {
    const f = new File(["x"], "local.png");
    const r = resolveSourcedField(
      field({ source: "selectedIds" }),
      input({ inputMode: "file", files: [f] }),
    );
    expect(r.status).toBe("ok");
    expect(r.status === "ok" && r.value).toEqual(["local.png"]);
  });

  it("selectedIds: empty when no files at all", () => {
    expect(
      resolveSourcedField(field({ source: "selectedIds" }), input({ inputMode: "file" })).status,
    ).toBe("empty");
  });

  // k65 — a multi-selection on a scalar file field is a BATCH (run once per file
  // by the tool layer), not the old "Select exactly one file" mismatch.
  it("selectedIds + scalar kind:file with >1 selected → ok (batch), summary says so", () => {
    const r = resolveSourcedField(
      field({ source: "selectedIds", kind: "file" }),
      multiFileInput(),
    );
    expect(r.status).toBe("ok");
    expect(r.status === "ok" && r.value).toEqual(["id-a", "id-b"]);
    expect(r.status === "ok" && r.summary).toContain("2 files");
    expect(r.status === "ok" && r.summary).toContain("one at a time");
  });
});

describe("batching a scalar file tool over a multi-file selection (k65)", () => {
  const filePage = () =>
    page({
      isUpload: true,
      fields: [{ name: "file", label: "File", kind: "file", source: "selectedIds" }],
    });

  it("batchFilesFor: [] for 0/1 files or a tool with no scalar file field", () => {
    expect(batchFilesFor(filePage(), input({ inputMode: "file" }))).toEqual([]);
    expect(
      batchFilesFor(
        filePage(),
        input({
          inputMode: "file",
          uploadedFiles: [{ name: "a", id: "id-a" }],
          selectedIds: ["id-a"],
        }),
      ),
    ).toEqual([]);
    expect(batchFilesFor(page(), multiFileInput())).toEqual([]);
  });

  it("batchFilesFor: every selected file, in order", () => {
    expect(batchFilesFor(filePage(), multiFileInput()).map((f) => f.id)).toEqual([
      "id-a",
      "id-b",
    ]);
  });

  it("pinInputToFile narrows the selection (and nothing else)", () => {
    const i = multiFileInput();
    const pinned = pinInputToFile(i, { name: "b", id: "id-b" });
    expect(pinned.selectedIds).toEqual(["id-b"]);
    expect(pinned.uploadedFiles).toEqual(i.uploadedFiles); // the library is untouched
    expect(i.selectedIds).toEqual(["id-a", "id-b"]);       // and the original isn't mutated
  });

  it("a pinned input submits exactly that one file", async () => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({ ok: true, value: {} });
    const i = multiFileInput();
    await submitPage(
      filePage(),
      pinInputToFile(i, { name: "b", id: "id-b" }),
      "pdf",
      "extract",
    );
    const body = JSON.parse(requestMock.mock.calls[0][1].body as string);
    expect(body.file).toBe("id-b");
  });

  it("submitPage THROWS rather than silently dropping files when handed >1", async () => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({ ok: true, value: {} });
    await expect(
      submitPage(filePage(), multiFileInput(), "pdf", "extract"),
    ).rejects.toThrow(/one file per call/);
    expect(requestMock).not.toHaveBeenCalled();
  });
});

describe("selected helpers + isFieldVisible + buildSourcePayload", () => {
  it("selectedUploadedFiles filters by id; selectedUploadedIds maps", () => {
    const i = input({
      inputMode: "file",
      uploadedFiles: [
        { name: "a", id: "id-a" },
        { name: "b", id: "id-b" },
      ],
      selectedIds: ["id-b"],
    });
    expect(selectedUploadedFiles(i).map((f) => f.id)).toEqual(["id-b"]);
    expect(selectedUploadedIds(i)).toEqual(["id-b"]);
  });

  it("isFieldVisible: sourced hidden, visibleWhen respected", () => {
    expect(isFieldVisible(field({ source: "text" }), {})).toBe(false);
    expect(isFieldVisible(field(), {})).toBe(true);
    const gated = field({ visibleWhen: { field: "mode", equals: "adv" } });
    expect(isFieldVisible(gated, { mode: "adv" })).toBe(true);
    expect(isFieldVisible(gated, { mode: "basic" })).toBe(false);
  });

  it("buildSourcePayload shapes by input mode", () => {
    expect(buildSourcePayload(input({ text: "hi" }), "text", "summarize")).toMatchObject({
      inputMode: "text",
      text: "hi",
    });
    expect(buildSourcePayload(input({ inputMode: "url", url: "u" }), "url", "any")).toMatchObject({
      inputMode: "url",
      url: "u",
    });
    const filePayload = buildSourcePayload(
      input({ inputMode: "file", uploadedFiles: [{ name: "a", id: "id-a" }], selectedIds: ["id-a"] }),
      "image",
      "analyze",
    );
    expect(filePayload).toMatchObject({ inputMode: "file" });
  });
});

describe("submitPage useMultipart decision matrix", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue({ ok: true, value: {} });
  });

  it("isUpload + local files + no uploaded paths → multipart FormData", async () => {
    const f = new File(["x"], "a.png");
    await submitPage(
      page({ isUpload: true, fields: [{ name: "img", label: "Img", kind: "file" }] }),
      input({ inputMode: "file", files: [f] }),
      "image",
      "analyze",
    );
    const sent = requestMock.mock.calls[0][1].body;
    expect(sent).toBeInstanceOf(FormData);
  });

  it("uploaded paths present → JSON (not multipart), source embedded", async () => {
    await submitPage(
      page({ isUpload: true }),
      input({
        inputMode: "file",
        uploadedFiles: [{ name: "a", id: "id-a" }],
        selectedIds: ["id-a"],
      }),
      "image",
      "analyze",
    );
    const sent = requestMock.mock.calls[0][1].body;
    expect(typeof sent).toBe("string");
    expect(JSON.parse(sent as string).source).toMatchObject({ inputMode: "file" });
  });

  it("non-upload page → JSON", async () => {
    await submitPage(page(), input({ text: "hi" }), "text", "summarize");
    expect(typeof requestMock.mock.calls[0][1].body).toBe("string");
  });
});
