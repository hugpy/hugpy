// src/utilities/pageSpec.ts


export type FieldKind =
  | "text"
  | "textarea"
  | "number"
  | "checkbox"
  | "select"
  | "file"
  | "files";

export interface FieldVisibleWhen {
  field: string;
  equals: string | number | boolean;
}


export type FieldSource =
  | "text"            // <- MediaInputValue.text
  | "url"             // <- MediaInputValue.url
  | "selectedIds";    // <- selected uploaded-file ids, falling back to all uploadedFiles

export interface FieldSpec {
  name: string;
  label: string;
  kind: FieldKind;
  required?: boolean;
  default?: string | number | boolean;
  choices?: readonly string[];
  help?: string;
  accept?: string;
  visibleWhen?: FieldVisibleWhen;
  source?: FieldSource;          // NEW
}

// pageSpec.ts
export type FieldResolution =
  | { status: "ok"; value: string | string[]; summary: string }
  | { status: "empty"; reason: string }
  | { status: "mismatch"; reason: string };

// Phase 3: sourced from the config module (single source of truth for URLs).
// API_BASE_OCR / API_BASE_PDF were dead duplicates (Phase 1 audit) — removed.
import { hugpyConfig } from "../../../../config";
export const API_BASE_HUGPY = hugpyConfig.apiBase;
// pageSpec.ts
export type MediaKind = "text" | "audio" | "image" | "video" | "url" | "pdf" | "document";

export type Operation =
  | "summarize"   // A-I
  | "keywords"    // A-II
  | "metadata"    // A-III
  | "chat"        // A-IIII
  | "transcribe"  // B-I
  | "ocr"         // C-I
  | "analyze"     // C-II
  | "download"    // D-I
  | "extract_audio" // D-II
  | "extract"       // E-I — doc/url text extraction → text
  | "intelligence"  // F-I — full media-intelligence pipeline (extract+enrich) → structured doc record
  | "imagegen";     // G-I — text-to-image generation → image(s)

export interface PageSpec {
  key: string;
  title: string;
  category: string;
  path: string;
  method?: "POST" | "GET";
  isUpload?: boolean;
  fields: readonly FieldSpec[];
  submitLabel?: string;
  description?: string;
  resultKind?: "json" | "text" | "image-analysis";
  /** Per-tool timeout ceiling (ms); long inference (whisper/video) sets a higher one. */
  timeoutMs?: number;

  // NEW — declarative, not inferred
  accepts: readonly MediaKind[];
  produces: Operation[] | Operation;
}

// Single source of truth (Phase 2): InputMode / UploadedFileRef / MediaInputValue
// are OWNED by the receptacle (the producer of media-input values). Re-exported here
// so the console files that import them from this module keep working unchanged.
// Declared exactly once now — in receptacle/src/imports/types.ts.
export type {
  InputMode,
  UploadedFileRef,
  MediaInputValue,
} from "../../../../receptacle";

export const OPERATION_OUTPUT: Record<Operation, MediaKind> = {
  summarize: "text",
  keywords: "text",
  metadata: "text",
  chat: "text",
  transcribe: "text",
  ocr: "text",
  analyze: "text",
  download: "video",      // or: a new "file" kind, depending on your taste
  extract_audio: "audio",
  extract: "text",        // doc/url text extraction → text
  intelligence: "text",   // structured document-intelligence record (rendered specially)
  imagegen: "image",      // text-to-image generation → image(s)
};