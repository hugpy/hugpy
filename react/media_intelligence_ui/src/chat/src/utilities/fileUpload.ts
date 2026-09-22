// File attachments for the chat composer: upload to hugpy's /uploads and detect
// a coarse media category so the intermediary + tools can act on the file.
import { request, errorOf, okValue } from "../../../transport/client";
import { extractUploadedFiles } from "../../../receptacle/src/imports/utils/UploadUtils";
import type { UploadedFileRef } from "../../../receptacle/src/imports/types";
import { hugpyConfig } from "../../../config";
import { getSessionId } from "../../../session";

export async function uploadAttachments(
  files: File[],
): Promise<UploadedFileRef[]> {
  const refs: UploadedFileRef[] = [];
  // The backend (/uploads) takes ONE file per request under the field name
  // "file" (request.files.get("file")). Upload each separately, collect refs.
  for (const f of files) {
    const fd = new FormData();
    fd.append("file", f);
    fd.append("sid", getSessionId()); // tag the upload to this session for per-session wipe
    const res = await request(hugpyConfig.uploadUrl, {
      method: "POST",
      body: fd,
      meta: { specKey: "upload" },
    });
    if (!res.ok) {
      const e = errorOf(res) as { message?: string; status?: number; kind?: string };
      const msg =
        typeof e === "string"
          ? e
          : e?.message ??
            (e?.kind === "unauthorized" || e?.status === 401 || e?.status === 403
              ? "not authorized to upload (please sign in)"
              : "upload failed");
      throw new Error(msg);
    }
    refs.push(...extractUploadedFiles(okValue(res)));
  }
  return refs;
}

/**
 * Delete one uploaded file from the server store (user-initiated). Best-effort:
 * the caller removes the library entry regardless of the network result. POST +
 * query string (no custom header / JSON body) so it stays a "simple" request and
 * never triggers a CORS preflight in cross-origin live mode.
 */
export async function deleteUploadedFile(id: string): Promise<void> {
  if (!id) return;
  const base = hugpyConfig.apiBase.replace(/\/$/, "");
  const q = `id=${encodeURIComponent(id)}&sid=${encodeURIComponent(getSessionId())}`;
  try {
    await fetch(`${base}/session/file?${q}`, {
      method: "POST",
      credentials: hugpyConfig.withCredentials ? "include" : "same-origin",
      keepalive: true,
    });
  } catch {
    /* best-effort — the library entry is removed by the caller regardless */
  }
}

export type MediaCategory = "audio" | "image" | "video" | "pdf" | "document" | "text";

const EXT: Record<string, MediaCategory> = {
  mp3: "audio", wav: "audio", m4a: "audio", flac: "audio", ogg: "audio", aac: "audio",
  png: "image", jpg: "image", jpeg: "image", gif: "image", webp: "image", bmp: "image", svg: "image",
  mp4: "video", mov: "video", mkv: "video", webm: "video", avi: "video",
  pdf: "pdf",
  docx: "document", doc: "document",
  xlsx: "document", xls: "document", csv: "document",
  txt: "document", md: "document",
};

// Coarse category from mime then extension (the basis for the file-specific options).
export function detectCategory(ref: { name: string; type?: string }): MediaCategory {
  const t = (ref.type || "").toLowerCase();
  if (t.startsWith("audio/")) return "audio";
  if (t.startsWith("image/")) return "image";
  if (t.startsWith("video/")) return "video";
  if (t === "application/pdf") return "pdf";
  if (
    t.includes("officedocument") ||
    t.includes("ms-excel") ||
    t === "application/msword" ||
    t.startsWith("text/")
  )
    return "document";
  const ext = (ref.name.split(".").pop() || "").toLowerCase();
  return EXT[ext] ?? "text";
}

// ── attach-time honesty (operator ask 2026-08-04, k64) ────────────────────────
// The UI already knows, at attach time, which extensions the backend can
// genuinely read — so say so THEN, quietly, instead of letting the user submit
// and get a mushy failure back. `readable: false` means /ml/extract will decline
// this file outright; a `note` is a plain-language caveat worth showing either
// way. Never blocks attaching: the file still uploads and the user can still ask
// about it by name.
export interface FileReadability {
  readable: boolean;
  note?: string;
}

// Extensions the backend declines by name (media_extract._UNSUPPORTED_EXT). Kept
// in step with that map — the wording mirrors the server's so the attach-time
// caveat and the failure message tell the same story.
const UNREADABLE_EXT: Record<string, string> = {
  doc: "legacy .doc can't be read yet — save it as .docx and re-attach",
  xls: "legacy .xls can't be read yet — save it as .xlsx and re-attach",
  ppt: "legacy .ppt can't be read yet — save it as .pptx, or export a PDF",
};

export function readabilityFor(ref: { name: string; type?: string }): FileReadability {
  const ext = (ref.name.split(".").pop() || "").toLowerCase();
  const blocked = UNREADABLE_EXT[ext];
  if (blocked) return { readable: false, note: blocked };
  // Video rides the transcribe path: honest about what that actually covers, so
  // "analyze this video" doesn't quietly become "analyze this video's audio".
  if (detectCategory(ref) === "video") {
    return { readable: true, note: "only the audio track is transcribed — the visuals aren't analyzed yet" };
  }
  // Unknown/binary-ish extensions stay OPTIMISTIC: the backend's binary sniff is
  // the authority on whether they read, and guessing pessimistically here would
  // scare users off files that work.
  return { readable: true };
}

// A durable, viewable session object for an uploaded/dropped file. Accumulated in
// HugpyChat's `files` store (not cleared on submit), shown in-thread + in the
// sidebar, and previewed inline. Session-only: `blobUrl` is an in-tab object URL
// (lost on reload); `extractedText` is the full-text index for documents.
export interface StoredFile {
  id: string;            // server opaque upload id (the /uploads basename)
  name: string;
  type: string;          // mime
  kind: MediaCategory;
  size?: number;
  uploadedAt: string;    // ISO
  blobUrl?: string;      // URL.createObjectURL(original) — inline image/audio/video preview
  extractedText?: string;           // documents: full-text for search + preview
  extractStatus?: "idle" | "extracting" | "done" | "error" | "unsupported";
}

// A small glyph per kind — the chat avoids the (tiny) icon set for file chrome,
// matching AttachmentBar's text/✕ style.
export function fileKindGlyph(kind: string): string {
  switch (kind) {
    case "image": return "🖼";
    case "audio": return "🎵";
    case "video": return "🎬";
    case "pdf":
    case "document": return "📄";
    default: return "📎";
  }
}
