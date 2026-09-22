// Flat dispatch for the chat arm. Build a flat JSON body from a tool's declared
// inputs and POST it to its NAMED endpoint (/api/ml/<task> or /api/prompt); the
// backend's execute_prompt → resolve does the actual routing. This replaces
// submitPage's field-spec machinery in the chat run path — the classic Tool
// Console still uses submitPage, so that path is left untouched.
import { request } from "../../../transport/client";
import { hugpyConfig } from "../../../config";
import type {
  PageSpec,
  MediaInputValue,
} from "../../../console/src/imports/pages/pageSpec";

// Map the gathered inputs onto the endpoint's declared fields:
//   • a selectedIds-sourced field gets the uploaded file path (file / image_path)
//   • a text-sourced field — or a free-text `prompt` (e.g. vision) — gets the text
//   • any field with a default is carried through (max_length, preset, task, …)
// Nothing here is task-specific: it reads the registry, so new tools just work.
function shapeBody(
  spec: PageSpec,
  text: string,
  filePath?: string,
  url?: string,
): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  for (const f of spec.fields ?? []) {
    if (f.source === "selectedIds" && filePath) body[f.name] = filePath;
    else if (f.source === "url" && url) body[f.name] = url;
    else if (f.source === "text" && text) body[f.name] = text;
    else if (f.name === "prompt" && text) body[f.name] = text;
    else if (f.default !== undefined) body[f.name] = f.default;
  }
  return body;
}

export async function dispatchTool(
  spec: PageSpec,
  input: MediaInputValue,
  signal?: AbortSignal,
) {
  const text = (input?.text ?? "").trim();
  const filePath = input?.selectedIds?.[0];
  const url = (input?.url ?? "").trim();
  const body = shapeBody(spec, text, filePath, url);
  // Heavy ML on CPU (Whisper transcription, video captioning) can run for minutes
  // on a real recording — lift the 120s default so longer media doesn't abort
  // mid-run and surface as "no file / couldn't complete".
  const slow = /transcribe|caption|video|embed/i.test(spec.key);
  return request(`${hugpyConfig.apiBase}${spec.path}`, {
    method: "POST",
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
    signal,
    timeoutMs: slow ? 600_000 : undefined,
    meta: { specKey: spec.key },
  });
}
