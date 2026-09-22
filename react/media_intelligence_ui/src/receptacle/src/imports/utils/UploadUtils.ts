import type { UploadedFileRef } from "./../types";



export function extractUploadedFiles(data: any): UploadedFileRef[] {
  if (!data) return [];

  if (Array.isArray(data)) {
    return data.map(normalizeUploadedFile).filter(Boolean) as UploadedFileRef[];
  }

  if (Array.isArray(data.files)) {
    return data.files.map(normalizeUploadedFile).filter(Boolean) as UploadedFileRef[];
  }

  if (Array.isArray(data.uploadedFiles)) {
    return data.uploadedFiles.map(normalizeUploadedFile).filter(Boolean) as UploadedFileRef[];
  }

  if (Array.isArray(data.results)) {
    return data.results.map(normalizeUploadedFile).filter(Boolean) as UploadedFileRef[];
  }

  const single = normalizeUploadedFile(data);
  return single ? [single] : [];
}

function normalizeUploadedFile(item: any): UploadedFileRef | null {
  if (!item || typeof item !== "object") return null;

  // Prefer a server-issued opaque id; fall back to the upload path/url as the handle
  // until the server emits real ids. Either way the client treats it as opaque.
  const id =
    item.id ??
    item.file_id ??
    item.fileId ??
    item.path ??
    item.file_path ??
    item.filePath ??
    item.saved_path ??
    item.savedPath ??
    item.url ??
    item.location;

  if (!id) return null;

  const name =
    item.name ??
    item.filename ??
    item.file_name ??
    String(id).split("/").pop() ??
    "uploaded-file";

  return {
    name: String(name),
    id: String(id),
    type: item.type ?? item.mime_type ?? item.mimeType ?? item.media ?? "file",
  };
}
