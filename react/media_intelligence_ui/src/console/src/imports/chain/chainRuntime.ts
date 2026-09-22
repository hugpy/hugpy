
import { getChain } from "./chainRegistry";
import { getPage } from "./../pages/pagesRegistry";
import  { type MediaInputValue, type MediaKind, type Operation, OPERATION_OUTPUT } from "./../pages/pageSpec";
import { isRecord, OperationSchema } from "./../schemas";
export interface ChainStepResult {
  pageKey: string;
  ok: boolean;
  data: unknown;
  nextInput?: MediaInputValue;
  error?: string;
}

function asArray<T>(value: T | T[]): T[] {
  return Array.isArray(value) ? value : [value];
}

export async function runChain(
  chainKey: string,
  initial: MediaInputValue,
  submitOne: (
    pageKey: string,
    input: MediaInputValue,
    overrides: Record<string, unknown>,
  ) => Promise<unknown>,
  onProgress?: (steps: ChainStepResult[]) => void,
): Promise<ChainStepResult[]> {
  const chain = getChain(chainKey);
  const results: ChainStepResult[] = [];

  let cursor = initial;

  for (const step of chain.steps) {
    try {
      const data = await submitOne(step.pageKey, cursor, step.overrides ?? {});
      const next = projectResultToInput(step.pageKey, data, step.overrides ?? {});

      results.push({
        pageKey: step.pageKey,
        ok: true,
        data,
        nextInput: next,
      });
      onProgress?.(results); // observable per-step status (Phase 8 run store)

      cursor = next;
    } catch (err) {
      results.push({
        pageKey: step.pageKey,
        ok: false,
        data: null,
        error: err instanceof Error ? err.message : String(err),
      });
      onProgress?.(results);

      break;
    }
  }

  return results;
}

export function projectResultToInput(
  pageKey: string,
  data: unknown,
  overrides: Record<string, unknown>,
): MediaInputValue {
  const page = getPage(pageKey);
  const ops = asArray(page.produces);

  const opOverride = OperationSchema.safeParse(overrides.__op);
  const op: Operation | undefined = opOverride.success ? opOverride.data : ops[0];

  // A chain step with no declared op is misconfigured — fall back to text projection.
  const kind: MediaKind = op ? OPERATION_OUTPUT[op] : "text";

  if (kind === "text") {
    return {
      inputMode: "text",
      text: extractText(data),
      url: "",
      files: [],
      uploadedFiles: [],
      selectedIds: [],
    };
  }

  if (kind === "audio" || kind === "video" || kind === "image") {
    const ids = extractPaths(data);

    return {
      inputMode: "file",
      text: "",
      url: "",
      files: [],
      uploadedFiles: ids.map((p) => ({
        name: p.split("/").pop() ?? p,
        id: p,
      })),
      selectedIds: ids,
    };
  }

  return {
    inputMode: "text",
    text: extractText(data),
    url: "",
    files: [],
    uploadedFiles: [],
    selectedIds: [],
  };
}

export function extractText(data: unknown): string {
  if (typeof data === "string") return data;

  if (isRecord(data)) {
    const o = data;

    for (const k of [
      "text",
      "summary",
      "result",
      "transcription",
      "transcript",
      "output",
      "response",
      "message",
    ]) {
      const val = o[k];
      if (typeof val === "string") return val;
    }

    if (isRecord(o.result)) {
      return extractText(o.result);
    }

    if (Array.isArray(o.results)) {
      return o.results.map(extractText).filter(Boolean).join("\n\n");
    }
  }

  return "";
}

export function extractPaths(data: unknown): string[] {
  if (isRecord(data)) {
    const o = data;

    if (Array.isArray(o.paths)) {
      return o.paths.filter((p): p is string => typeof p === "string");
    }

    if (typeof o.path === "string") {
      return [o.path];
    }

    if (isRecord(o.result)) {
      return extractPaths(o.result);
    }
  }

  return [];
}