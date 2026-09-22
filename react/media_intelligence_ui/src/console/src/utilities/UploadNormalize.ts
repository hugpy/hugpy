import type { UploadedFileRef } from "./../imports/pages/pageSpec";
import { parseUploadResponse } from "./../imports/schemas";

function asCleanString(value: unknown): string {
  if (typeof value !== "string") return "";
  const trimmed = value.trim();
  if (!trimmed || trimmed === "[object Object]") return "";
  return trimmed;
}

function firstString(...values: unknown[]): string {
  for (const value of values) {
    const clean = asCleanString(value);
    if (clean) return clean;
  }
  return "";
}

function firstNumber(...values: unknown[]): number | undefined {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
      const n = Number(value);
      if (Number.isFinite(n)) return n;
    }
  }
  return undefined;
}

export function uploadResponseToUploadedFileRef(
  response: unknown,
  fallbackFile?: File,
): UploadedFileRef {
  const r = parseUploadResponse(response);

  // Prefer a server-issued opaque id; fall back to the path/url as the handle.
  const id = firstString(
    r.id,
    r.file_id,
    r.path,
    r.url,
    r.file_url,
    r.file?.id,
    r.file?.path,
    r.file?.url,
    r.file?.file_url,
    r.data?.id,
    r.data?.path,
    r.data?.url,
    r.data?.file_url,
    r.result?.id,
    r.result?.path,
    r.result?.url,
    r.result?.file_url,
  );

  if (!id) {
    throw new Error(
      "Upload response did not contain a usable id/path/url:\n\n" +
        JSON.stringify(response, null, 2),
    );
  }

  const name =
    firstString(
      r.name,
      r.filename,
      r.file?.name,
      r.file?.filename,
      r.data?.name,
      r.data?.filename,
      r.result?.name,
      r.result?.filename,
      fallbackFile?.name,
    ) || id.split("/").pop() || "uploaded-file";

  const type =
    firstString(
      r.type,
      r.file?.type,
      r.data?.type,
      r.result?.type,
      fallbackFile?.type,
    ) || undefined;

  const size = firstNumber(
    r.size,
    r.file?.size,
    r.data?.size,
    r.result?.size,
    fallbackFile?.size,
  );

  return {
    name,
    id,
    type,
    size,
  };
}
