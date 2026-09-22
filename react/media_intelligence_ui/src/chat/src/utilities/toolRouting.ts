// The LLM as authoritative intermediary. Two helpers:
//  - routeTool(): ask the selected model to pick ONE media tool (or none) for
//    the user's message, returning a validated registry key.
//  - serializeToolResult(): flatten a tool Result into text the narration model
//    can speak to (we never show raw output — the LLM presents it).
import { listPages } from "../../../console/src/imports/pages/pagesRegistry";
import type { PageSpec } from "../../../console/src/imports/pages/pageSpec";
import { generateChat } from "./api";
import type { ChatMessage } from "../imports";

export interface ToolRoute {
  specKey: string;
  reason: string;
}

// One-shot routing call (POST /prompt text-generation via the SELECTED model).
// Defensive: any failure or unrecognized key => null (just answer conversationally).
export async function routeTool(
  prompt: string,
  modelKey: string,
  requestId: string,
  signal?: AbortSignal,
  activeFile?: { name: string; kind: string } | null,
): Promise<ToolRoute | null> {
  const pages = listPages();
  if (pages.length === 0) return null;
  const catalog = pages.map((p) => `${p.key} — ${p.title}`).join("\n");

  // When a file is already uploaded/selected, tell the router — otherwise it
  // refuses image/audio tools ("unless the user clearly refers to a file"), which
  // is exactly why "transcribe this" on an uploaded audio file got a "please
  // upload a file" reply instead of running.
  const fileHint = activeFile
    ? `\n\nThe user already has a ${activeFile.kind} file "${activeFile.name}" uploaded and selected. ` +
      `If their message asks to process, analyze, transcribe, describe, or extract from it, ` +
      `pick the matching ${activeFile.kind} tool — do NOT ask them to upload a file.`
    : "";

  const system =
    "You route a user's request to ONE media-intelligence tool, or to NONE. " +
    "Default STRONGLY to none. Only pick a tool when the user CLEARLY asks for that " +
    "specific operation on real content. Greetings, questions, small talk, gibberish, " +
    "or anything ambiguous → empty. Never pick an image or audio tool unless the user " +
    "clearly refers to an image or audio file. " +
    'Reply with ONLY a JSON object, no prose: {"specKey":"<exact key, or empty>","reason":"<short>"}. ' +
    'The specKey MUST be the exact token BEFORE the “—” in a catalog line (e.g. "ml/summarize"), ' +
    'copied verbatim. Use "" when the user just wants to chat or no tool fits.\n\n' +
    `Catalog:\n${catalog}` + fileHint;

  const messages: ChatMessage[] = [
    { role: "system", content: system },
    { role: "user", content: prompt },
  ];

  try {
    const res = await generateChat(
      {
        request_id: requestId,
        model_key: modelKey,
        messages,
        max_new_tokens: 120,
        temperature: 0,
        do_sample: false,
      },
      signal,
    );
    const text = typeof res.text === "string" ? res.text : "";
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start < 0 || end < start) return null;
    const obj = JSON.parse(text.slice(start, end + 1)) as {
      specKey?: string;
      reason?: string;
    };
    const raw = (obj.specKey || "").trim();
    if (!raw) return null;
    // Resolve the model's answer to a REAL registry key: exact first, else the
    // last path segment / title / produces value (small models often reply with
    // "keywords" instead of "ml/keywords").
    const k = raw.toLowerCase();
    const match =
      pages.find((p) => p.key === raw) ??
      pages.find((p) => {
        const last = p.key.toLowerCase().split("/").pop();
        const prod = (Array.isArray(p.produces) ? p.produces : [p.produces])
          .map((x) => String(x).toLowerCase());
        return (
          p.key.toLowerCase() === k ||
          p.key.toLowerCase().endsWith("/" + k) ||
          last === k ||
          p.title.toLowerCase() === k ||
          prod.includes(k)
        );
      });
    if (!match) return null;
    return {
      specKey: match.key,
      reason: typeof obj.reason === "string" ? obj.reason : "",
    };
  } catch {
    return null;
  }
}

// Flatten common /ml Result shapes into narration-ready text. Never shows raw.
export function serializeToolResult(result: unknown, _spec: PageSpec): string {
  if (result == null) return "";
  if (typeof result === "string") return result;
  const r = result as Record<string, unknown>;
  if (typeof r.text === "string" && r.text.trim()) return r.text;
  // Transcription shape: the transcript may sit under segments[].text or
  // raw.whisper_result.text rather than a top-level `text`. Pull it so the
  // narrator speaks the transcript instead of "no audio file provided".
  const segs = Array.isArray(r.segments) ? (r.segments as Array<{ text?: unknown }>) : [];
  const segText = segs.map((s) => (typeof s?.text === "string" ? s.text : "")).join(" ").trim();
  if (segText) return segText;
  const raw = (r.raw && typeof r.raw === "object" ? r.raw : {}) as Record<string, unknown>;
  const wr = (raw.whisper_result && typeof raw.whisper_result === "object" ? raw.whisper_result : {}) as Record<string, unknown>;
  if (typeof wr.text === "string" && wr.text.trim()) return wr.text;
  const primary = Array.isArray(r.primary) ? (r.primary as string[]) : [];
  const combined = Array.isArray(r.combined) ? (r.combined as string[]) : [];
  if (primary.length || combined.length) {
    return `Extracted keywords: ${[...primary, ...combined].slice(0, 24).join(", ")}`;
  }
  if (Array.isArray(r.embeddings)) {
    return `Computed ${(r.embeddings as unknown[]).length} embedding vector(s).`;
  }
  if (typeof r.error === "string" && r.error) return `(tool error: ${r.error})`;
  try {
    return JSON.stringify(result).slice(0, 4000);
  } catch {
    return String(result);
  }
}
