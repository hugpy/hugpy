// mediaIntelligence.ts — the chat's media-intelligence BRIDGE.
//
// This is the "missing medium" between the chat UI and hugpy: instead of the old
// task-derivation (pick ONE tool per attachment and narrate its raw output), an
// attachment is run through the media_intelligence pipeline shape —
//
//     ingest(done: uploaded) → EXTRACT (ocr / transcribe / caption)
//                            → ENRICH  (summarize + keywords)
//
// — producing ONE typed DocumentIntelligence record the chat renders structurally.
//
// It mirrors the media_intelligence Python package's MediaPipeline / MediaItem so
// the same contract can later be served by the package's HTTP bridge
// (POST {apiBase}/media/analyze). When VITE_HUGPY_MEDIA_ANALYZE_URL is set the
// bridge calls that single endpoint; otherwise it orchestrates the existing
// hugpy /ml/* endpoints client-side. Either way the chat sees one result shape.
import "../../../console/src/imports/pages/pagesBuiltin"; // ensure /ml/* specs are registered
import { request, okValue, errorOf } from "../../../transport/client";
import { hugpyConfig } from "../../../config";
import { dispatchTool } from "./dispatch";
import { getPage } from "../../../console/src/imports/pages/pagesRegistry";
import type { MediaInputValue, MediaKind } from "../../../console/src/imports/pages/pageSpec";
import type { MediaCategory, UploadedFileRef } from "./fileUpload";

// One stage of the pipeline, recorded for transparency (like PipelineReport).
export interface DocStage {
  stage: "extract" | "transcribe" | "caption" | "summarize" | "keywords";
  status: "ok" | "skipped" | "error";
  detail?: string;
}

// The typed record the chat renders — the TS mirror of media_intelligence's
// MediaItem (extract + enrich outputs).
export interface DocumentIntelligence {
  source: string;
  kind: MediaKind;
  text?: string;
  pages?: { index?: number; text: string }[];
  transcript?: { text: string; segments?: unknown[]; language?: string };
  summary?: string;
  keywords?: unknown;
  caption?: string;
  stages: DocStage[];
  ok: boolean;
  // Partial-read reporting from the extractor (backend k65). A PDF whose page 4
  // threw still returns pages 1-3 plus a warning saying so — carried here so the
  // gap is REPORTED (panel + narration) instead of a short document passing for
  // a whole one. Absent when the format doesn't paginate.
  pagesTotal?: number;
  pagesExtracted?: number;
  warnings?: string[];
  meta?: { title?: string; author?: string; page_count?: number };
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// Pull the best text field out of a heterogeneous /ml/* response.
function pickText(data: unknown): string {
  if (typeof data === "string") return data.trim();
  if (Array.isArray(data)) {
    return data
      .map((p) => (isRecord(p) && typeof p.text === "string" ? p.text : ""))
      .filter(Boolean)
      .join("\n\n")
      .trim();
  }
  if (isRecord(data)) {
    for (const k of ["text", "transcription", "transcript", "summary",
                     "analysis", "caption", "result", "output", "response", "message"]) {
      const v = data[k];
      if (typeof v === "string" && v.trim()) return v.trim();
    }
    const nested = data.result ?? data.data;
    if (nested && nested !== data) return pickText(nested);
  }
  return "";
}

function pagesFrom(data: unknown): { index?: number; text: string }[] | undefined {
  const arr = Array.isArray(data) ? data : isRecord(data) && Array.isArray(data.pages) ? data.pages : null;
  if (!arr) return undefined;
  const pages = arr
    .map((p, i) =>
      isRecord(p) && typeof p.text === "string"
        ? { index: typeof p.page_num === "number" ? p.page_num : i, text: p.text }
        : null,
    )
    .filter((p): p is { index: number; text: string } => Boolean(p));
  return pages.length ? pages : undefined;
}

/** The extractor's partial-read report (`pages_total` / `pages_extracted` /
 *  `warnings` / `meta`), tolerant of it being absent. */
function coverageFrom(data: unknown): Partial<DocumentIntelligence> {
  const src = isRecord(data) && isRecord(data.result) ? data.result : data;
  if (!isRecord(src)) return {};
  const out: Partial<DocumentIntelligence> = {};
  if (typeof src.pages_total === "number") out.pagesTotal = src.pages_total;
  if (typeof src.pages_extracted === "number") out.pagesExtracted = src.pages_extracted;
  if (Array.isArray(src.warnings)) {
    const warnings = src.warnings.filter((w): w is string => typeof w === "string");
    if (warnings.length) out.warnings = warnings;
  }
  if (isRecord(src.meta)) {
    const m = src.meta;
    const meta: NonNullable<DocumentIntelligence["meta"]> = {};
    if (typeof m.title === "string") meta.title = m.title;
    if (typeof m.author === "string") meta.author = m.author;
    if (typeof m.page_count === "number") meta.page_count = m.page_count;
    if (Object.keys(meta).length) out.meta = meta;
  }
  return out;
}

/** "Extracted 3/5 pages; page 4 failed: …" — the honest one-liner about what this
 *  read did and didn't get. Empty when the read was complete (or unpaginated). */
export function coverageNoteFor(di: DocumentIntelligence): string {
  const parts: string[] = [];
  if (
    typeof di.pagesExtracted === "number" &&
    typeof di.pagesTotal === "number" &&
    di.pagesTotal > 0 &&
    di.pagesExtracted < di.pagesTotal
  ) {
    parts.push(`Extracted ${di.pagesExtracted}/${di.pagesTotal} pages`);
  }
  if (di.warnings?.length) parts.push(di.warnings.join("; "));
  return parts.join("; ");
}

// The extract op for each media category (mirrors MediaPipeline.extract dispatch).
const EXTRACT_BY_KIND: Partial<Record<MediaCategory, { specKey: string; stage: DocStage["stage"] }>> = {
  pdf:      { specKey: "ml/extract",    stage: "extract" },
  document: { specKey: "ml/extract",    stage: "extract" },
  audio:    { specKey: "ml/transcribe", stage: "transcribe" },
  video:    { specKey: "ml/transcribe", stage: "transcribe" },
  image:    { specKey: "ml/vision",     stage: "caption" },
  // Undetermined type (detectCategory's fallback) → default to just READING the
  // file via /ml/extract: it reads known text formats, and the backend now falls
  // back to a guarded UTF-8 read for unknown extensions too. So an unrecognized
  // file is read rather than rejected.
  text:     { specKey: "ml/extract",    stage: "extract" },
};

function asStageFailure(e: unknown): StageOutcome {
  return { data: null, error: e instanceof Error ? e.message : String(e) };
}

/**
 * The first thing that actually went wrong, as a sentence fit to show the user.
 * Prefers the EXTRACT/transcribe/caption stage (a file that wouldn't read is the
 * failure the operator cares about) and carries the backend's own wording —
 * "legacy binary .doc isn't supported — save it as .docx and re-attach" rather
 * than "couldn't analyze". Empty when nothing failed. (operator ask 2026-08-04, k64)
 */
export function failureReasonFor(di: DocumentIntelligence): string {
  const failed = di.stages.find((s) => s.status === "error");
  return failed?.detail?.trim() || "";
}

export function supportsIntelligence(kind: MediaCategory): boolean {
  return kind in EXTRACT_BY_KIND;
}

function fileInput(id: string, text = ""): MediaInputValue {
  return {
    inputMode: "file", text, url: "",
    files: [], uploadedFiles: [], selectedIds: [id],
  } as never;
}

function textInput(text: string): MediaInputValue {
  return {
    inputMode: "text", text, url: "",
    files: [], uploadedFiles: [], selectedIds: [],
  } as never;
}

// One stage call. Returns the stage's REASON on failure, not just null: the
// backend answers a decline with {ok:false, error:"<honest sentence>"} at HTTP
// 400, and that sentence is the only thing that tells the user WHY their file
// didn't read (operator ask 2026-08-04, k64). Swallowing it into `null` is what
// made failures mushy.
interface StageOutcome {
  data: unknown | null;
  error?: string;
}

async function runOne(
  specKey: string,
  input: MediaInputValue,
  signal?: AbortSignal,
): Promise<StageOutcome> {
  const spec = getPage(specKey);
  if (!spec) return { data: null, error: `no "${specKey}" tool is registered` };
  const res = await dispatchTool(spec, input, signal);
  if (res.ok) return { data: okValue(res) };
  const err = errorOf(res) as { message?: string; kind?: string };
  return {
    data: null,
    error: err?.message || (err?.kind ? `request ${err.kind}` : "the request failed"),
  };
}

// Server bridge: the media_intelligence Python package's HTTP endpoint, when deployed.
// Returns a DocumentIntelligence-shaped object or null if unavailable / wrong shape.
async function tryServerBridge(
  ref: UploadedFileRef,
  kind: MediaCategory,
  signal?: AbortSignal,
): Promise<DocumentIntelligence | null> {
  const url = hugpyConfig.mediaAnalyzeUrl;
  if (!url) return null;
  try {
    const res = await request<unknown>(url, {
      method: "POST",
      body: JSON.stringify({ file: ref.id, kind, source: ref.name }),
      headers: { "Content-Type": "application/json" },
      signal,
      meta: { specKey: "media/intelligence" },
    });
    if (!res.ok) return null;
    const data = okValue(res);
    const u = isRecord(data) && isRecord(data.result) ? data.result : data;
    if (isRecord(u) && (Array.isArray(u.stages) || typeof u.summary === "string" || typeof u.text === "string")) {
      return { source: ref.name, kind: kind as MediaKind, ok: true, stages: [], ...(u as object) } as DocumentIntelligence;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Run the media-intelligence pipeline for one attached file and return a typed
 * DocumentIntelligence record. Best-effort: per-stage failures are recorded in
 * `stages` rather than thrown, so the chat always has something to render.
 */
export async function runDocumentIntelligence(
  ref: UploadedFileRef,
  kind: MediaCategory,
  prompt: string,
  signal?: AbortSignal,
): Promise<DocumentIntelligence> {
  // Prefer the real media_intelligence HTTP bridge when configured.
  const server = await tryServerBridge(ref, kind, signal);
  if (server) return server;

  const di: DocumentIntelligence = { source: ref.name, kind: kind as MediaKind, stages: [], ok: false };
  const plan = EXTRACT_BY_KIND[kind];
  if (!plan) {
    di.stages.push({ stage: "extract", status: "error", detail: `no extractor for ${kind}` });
    return di;
  }

  // ── EXTRACT ──────────────────────────────────────────────────────────
  try {
    const visionPrompt = kind === "image"
      ? (prompt?.trim() || "Describe this image in detail.")
      : prompt;
    const { data: out, error: stageError } = await runOne(
      plan.specKey, fileInput(ref.id, visionPrompt), signal,
    );
    if (out == null) {
      di.stages.push({
        stage: plan.stage,
        status: "error",
        detail: stageError || "the extractor returned nothing",
      });
      return di;
    }
    if (plan.stage === "transcribe" && isRecord(out)) {
      const text = pickText(out);
      di.transcript = {
        text,
        segments: Array.isArray(out.segments) ? out.segments : undefined,
        language: typeof out.language === "string" ? out.language : undefined,
      };
      di.text = text;
    } else if (plan.stage === "caption") {
      di.caption = pickText(out);
      di.text = di.caption;
    } else {
      di.text = pickText(out);
      di.pages = pagesFrom(out);
      Object.assign(di, coverageFrom(out)); // pages_total / warnings / meta (k65)
    }
    di.stages.push({ stage: plan.stage, status: di.text ? "ok" : "error", detail: di.text ? undefined : "empty" });
  } catch (e) {
    di.stages.push({ stage: plan.stage, status: "error", detail: e instanceof Error ? e.message : String(e) });
    return di;
  }

  // ── ENRICH (summarize + keywords, in parallel over the extracted text) ─
  const text = di.text ?? "";
  if (text.trim().length < 12) {
    di.stages.push({ stage: "summarize", status: "skipped", detail: "too little text" });
    di.stages.push({ stage: "keywords", status: "skipped", detail: "too little text" });
    di.ok = Boolean(di.text || di.transcript || di.caption);
    return di;
  }

  const [summaryRes, keywordsRes] = await Promise.all([
    runOne("ml/summarize", textInput(text), signal).catch(asStageFailure),
    runOne("ml/keywords", textInput(text), signal).catch(asStageFailure),
  ]);

  const summary = pickText(summaryRes.data);
  if (summary) { di.summary = summary; di.stages.push({ stage: "summarize", status: "ok" }); }
  else di.stages.push({ stage: "summarize", status: "error", detail: summaryRes.error || "no summary" });

  const keywordsOut = keywordsRes.data;
  if (keywordsOut != null) {
    di.keywords = isRecord(keywordsOut) && isRecord(keywordsOut.result) ? keywordsOut.result : keywordsOut;
    di.stages.push({ stage: "keywords", status: "ok" });
  } else {
    di.stages.push({ stage: "keywords", status: "error", detail: keywordsRes.error || "no keywords" });
  }

  di.ok = Boolean(di.text || di.transcript || di.summary);
  return di;
}

// Compact context string handed to the narrator for the SHORT conversational
// lead-in (the structured panel carries the full detail).
//
// Options (k65 — several attachments per turn are now analyzed, not just the
// first): `label` tags the block so a multi-file context says WHICH file each
// part describes ("[file 1/2: notes.pdf]"), and `instruct: false` drops the
// trailing instruction so the caller can issue ONE instruction covering them all.
export function narrationContextFor(
  di: DocumentIntelligence,
  opts: { label?: string; instruct?: boolean } = {},
): string {
  const head = opts.label ? `[${opts.label}] ` : "";
  const lines: string[] = [`${head}Document intelligence for "${di.source}" (${di.kind}):`];
  if (di.summary) lines.push(`Summary: ${di.summary}`);
  else if (di.text) lines.push(`Extracted text (excerpt): ${di.text.slice(0, 800)}`);
  if (di.keywords) {
    const kw = di.keywords as Record<string, unknown>;
    const primary = Array.isArray(kw.primary) ? kw.primary : Array.isArray(kw) ? kw : [];
    if (primary.length) lines.push(`Key topics: ${primary.slice(0, 8).join(", ")}`);
  }
  if (di.meta?.title) lines.push(`Document title: ${di.meta.title}`);
  // A PARTIAL read must be said out loud — the user is owed "3 of 5 pages" rather
  // than a confident summary of a document we only half read (k65).
  const coverage = coverageNoteFor(di);
  if (coverage) {
    lines.push(
      `Incomplete read: ${coverage}. Mention this limitation — do not present the ` +
      `summary as covering the whole document.`,
    );
  }
  if (opts.instruct !== false) {
    lines.push(
      "\nGive a 1–2 sentence overview of what this file is and its key points. " +
      "Do NOT list everything — the full breakdown (summary, keywords, text) is shown to the user separately.",
    );
  }
  return lines.join("\n");
}
