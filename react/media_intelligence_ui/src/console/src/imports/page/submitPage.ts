import {
  API_BASE_HUGPY,
  type FieldResolution,
  type FieldSpec,
  type MediaInputValue,
  type MediaKind,
  type Operation,
  type PageSpec,
  type UploadedFileRef,
} from "./../pages/pageSpec";
import { asFileArray } from "./../schemas";
import { request, type Result } from "./../../../../transport/client";

export type PageValues = Record<
  string,
  string | number | boolean | File[] | string[] | null
>;

export function selectedUploadedFiles(input: MediaInputValue): UploadedFileRef[] {
  if (input.inputMode !== "file") return [];
  return input.uploadedFiles.filter((f) => input.selectedIds.includes(f.id));
}
export function selectedUploadedIds(input: MediaInputValue): string[] {
  return selectedUploadedFiles(input).map((f) => f.id);
}

export function buildSourcePayload(
  input: MediaInputValue,
  media: MediaKind,
  operation: Operation | "any",
) {
  if (input.inputMode === "text") {
    return {
      inputMode: "text",
      kind: media,
      media,
      operation,
      text: input.text,
    };
  }

  if (input.inputMode === "url") {
    return {
      inputMode: "url",
      kind: media,
      media,
      operation,
      url: input.url,
    };
  }

  const picked = selectedUploadedFiles(input);

  return {
    inputMode: "file",
    kind: media,
    media,
    operation,
    files: picked,
  };
}
export function resolveSourcedField(
  field: FieldSpec,
  input: MediaInputValue,
): FieldResolution {
  const src = field.source!;

  if (src === "text") {
    const v = input.text.trim();

    if (!v) {
      return {
        status: "empty",
        reason: "No text entered in the input panel.",
      };
    }

    const preview = v.length > 60 ? v.slice(0, 57) + "..." : v;

    return {
      status: "ok",
      value: v,
      summary: `${v.length} chars — "${preview}"`,
    };
  }

  if (src === "url") {
    const v = input.url.trim();

    if (!v) {
      return {
        status: "empty",
        reason: "No URL entered in the input panel.",
      };
    }

    return {
      status: "ok",
      value: v,
      summary: v,
    };
  }

if (src === "selectedIds") {
  const picked = selectedUploadedFiles(input);

  if (picked.length) {
    const names = picked.map((f) => f.name).join(", ");
    const ids = picked.map((f) => f.id);

    // A scalar `kind: "file"` field still takes ONE file per call — but selecting
    // several is no longer an error: it is a BATCH, run once per file by the tool
    // layer (operator ask 2026-08-04, k65; the old "Select exactly one file"
    // mismatch is gone, and the silent take-the-first it guarded is now a throw
    // in submitPage). Say so in the summary so the user knows what Run will do.
    const batched = field.kind === "file" && picked.length > 1;

    return {
      status: "ok",
      value: ids,
      summary:
        picked.length === 1
          ? names
          : `${picked.length} files — ${names}${batched ? " (run one at a time)" : ""}`,
    };
  }

  const localFiles = input.inputMode === "file" ? input.files ?? [] : [];

  if (localFiles.length) {
    const names = localFiles.map((f) => f.name).join(", ");

    return {
      status: "ok",
      value: localFiles.map((f) => f.name),
      summary:
        localFiles.length === 1
          ? names
          : `${localFiles.length} files — ${names}`,
    };
  }

  if (!input.uploadedFiles.length) {
    return { status: "empty", reason: "No files uploaded yet." };
  }

  return { status: "empty", reason: "Files uploaded but none selected." };
}
}

// ---- batching a scalar file tool over a multi-file selection (k65) ----------
//
// The /ml amenities are one-file-per-call by contract, but the receptacle lets the
// user tick SEVERAL uploaded files. Before k65 the extra files were silently
// dropped. They are now run one after another at the TOOL layer — the amenity
// contract is untouched; only the console loops. GPU amenities serialize anyway,
// so sequential is also what the hardware wants.

/** The scalar file field a tool feeds from the uploaded-file selection, if any. */
function batchFileField(spec: {
  readonly fields: readonly FieldSpec[];
}): FieldSpec | null {
  return (
    spec.fields.find((f) => f.source === "selectedIds" && f.kind === "file") ?? null
  );
}

/**
 * The files a Run should iterate over, or [] when this is an ordinary single
 * call (no scalar file field, or 0–1 files selected — which keeps the exact
 * pre-k65 code path).
 */
export function batchFilesFor(
  spec: { readonly fields: readonly FieldSpec[] },
  input: MediaInputValue,
): UploadedFileRef[] {
  if (!batchFileField(spec)) return [];
  const picked = selectedUploadedFiles(input);
  return picked.length > 1 ? picked : [];
}

/**
 * Narrow a media input to ONE selected file. This — rather than a field-level
 * `overrides` entry — is how a batch iteration pins the file: sourced fields are
 * resolved from the input, so narrowing the input keeps every derived value
 * consistent (the resolved field, the `source` payload, the multipart parts) with
 * the file actually being processed. An override could only have pinned the field.
 */
export function pinInputToFile(
  input: MediaInputValue,
  file: UploadedFileRef,
): MediaInputValue {
  return { ...input, selectedIds: [file.id] };
}

export function isFieldVisible(field: FieldSpec, values: PageValues): boolean {
  if (field.source) return false;
  if (!field.visibleWhen) return true;

  return values[field.visibleWhen.field] === field.visibleWhen.equals;
}

function appendScalarField(fd: FormData, name: string, value: unknown) {
  if (value === "" || value == null) return;

  if (typeof value === "boolean") {
    if (value) fd.append(name, "true");
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      fd.append(name, String(item));
    }
    return;
  }

  fd.append(name, String(value));
}

export async function submitPage(
  spec: PageSpec,
  input: MediaInputValue,
  media: MediaKind,
  operation: Operation | "any",
  values: PageValues = {},
  overrides: Record<string, unknown> = {},
  opts: { signal?: AbortSignal } = {},
): Promise<Result<unknown>> {
  const mergedValues: PageValues = {
    ...values,
    ...overrides,
  } as PageValues;

  const sourcePayload = buildSourcePayload(input, media, operation);
  const pickedUploadedFiles = selectedUploadedFiles(input);
  const pickedUploadedIds = pickedUploadedFiles.map((f) => f.id);

  /**
   * If the receptacle already uploaded files, prefer JSON/path-based execution.
   * Only use multipart when there are no uploaded server paths available.
   */
const hasUploadedPaths =
  input.inputMode === "file" && pickedUploadedIds.length > 0;

const hasLocalFiles =
  input.inputMode === "file" && (input.files?.length ?? 0) > 0;

const useMultipart =
  !!spec.isUpload && hasLocalFiles && !hasUploadedPaths;

  let body: BodyInit;
  let headers: Record<string, string> = {};

  if (useMultipart) {
    const fd = new FormData();

    fd.append("source", JSON.stringify(sourcePayload));
if (input.inputMode === "file") {
  for (const file of input.files ?? []) {
    fd.append("files", file);
  }
}
    for (const f of spec.fields) {
      if (f.source) continue;

      // Default fallback mirrors the JSON branch: carries `task` / `model_key` (and any
      // other spec-default field) into multipart submissions for upload-backed tasks.
      const v = mergedValues[f.name] ?? f.default;

      if (f.kind === "files" || f.kind === "file") {
        const fromField = asFileArray(v);
        const files = fromField.length ? fromField : input.files ?? [];

        for (const file of files) {
          fd.append("files", file);
        }
      } else if (f.kind === "checkbox") {
        if (v) fd.append(f.name, "true");
      } else {
        appendScalarField(fd, f.name, v);
      }
    }

    body = fd;
  } else {
const payload: Record<string, unknown> = {
  kind: media,
  media,
  operation,
  source: sourcePayload,
};


    for (const f of spec.fields) {
      if (f.source) {
  const r = resolveSourcedField(f, input);
  if (r.status === "ok") {
    if (f.source === "selectedIds" && f.kind === "file") {
      // Scalar file field: ONE file per call. A multi-selection is a batch, and
      // batching is the TOOL layer's job (UtilityPage pins this input to a single
      // file and calls submitPage once per file) — so more than one here means a
      // caller skipped that step. Throw rather than take [0]: silently processing
      // the first file and dropping the rest is the k65 bug, not a fallback.
      const ids = Array.isArray(r.value) ? r.value : [r.value];
      if (ids.length > 1) {
        throw new Error(
          `${f.label}: this tool takes one file per call (got ${ids.length}) — ` +
          `run the selection as a batch.`,
        );
      }
      payload[f.name] = ids[0];
    } else {
      payload[f.name] = r.value;
    }
  }
  continue;
}

      // Fall back to the spec's declared default when the form supplied nothing. This is
      // what carries the synthesized `task` / `model_key` routing fields (no `source`,
      // value lives only on the spec) into the POST /prompt body for server pages.
      const v = mergedValues[f.name] ?? f.default;

      if (f.kind === "number") {
        const n = Number(v);
        if (Number.isFinite(n)) payload[f.name] = n;
      } else if (f.kind === "checkbox") {
        payload[f.name] = !!v;
      } else if (v !== "" && v != null) {
        payload[f.name] = v;
      }
    }

    body = JSON.stringify(payload);
    headers = {
      "Content-Type": "application/json",
    };
  }

  const url = API_BASE_HUGPY.replace(/\/$/, "") + spec.path;

  // All transport concerns (auth, abort, timeout, error→Result mapping) live in the
  // client. POST is not retried (may have side effects).
  return request(url, {
    method: spec.method ?? "POST",
    body,
    headers,
    meta: { specKey: spec.key, operation: String(operation) },
    ...(opts.signal ? { signal: opts.signal } : {}),
    ...(spec.timeoutMs != null ? { timeoutMs: spec.timeoutMs } : {}),
  });
}