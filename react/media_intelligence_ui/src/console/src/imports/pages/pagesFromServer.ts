// utilities/imports/pages/pagesFromServer.ts
//
// Server-driven pages from CURRENT hugpy (NOT the legacy pre-hugpy.ai API).
//
// The current-hugpy catalog lives at GET /prompt/tasks and returns:
//   { tasks: string[], defaults: Record<task, model_key> }
// — i.e. a flat list of task strings plus the default model_key per task. There is no
// per-page field registry on this surface, so we synthesize one PageSpec per task here
// (deriving fields/media/operation from the task name) and submit every page to the
// single unified POST /prompt endpoint (see submitPage.ts) as {task, model_key?, ...}.
//
// `defaults[task]` becomes the page's default model_key (carried on the synthesized
// `model_key` field so submitPage forwards it). If the catalog is unreachable or
// malformed, we keep the built-in pages only (graceful degrade — parse, don't assert).
import { z } from "zod";
import { registerPage } from "./pagesRegistry";
import type { FieldSpec, MediaKind, Operation, PageSpec } from "./pageSpec";
import { hugpyConfig } from "../../../../config";
import { request, errorOf, okValue } from "../../../../transport/client";
import { isRecord } from "../schemas";

const REGISTRY_ENDPOINT = hugpyConfig.registryUrl;

// Current hugpy's /prompt/tasks contract: { tasks, defaults }. Parse it as data at the
// trust boundary rather than asserting a shape onto network bytes.
const TaskCatalogSchema = z.object({
  tasks: z.array(z.string()),
  defaults: z.record(z.string()).optional(),
});

// Per-task UI hint: which media it accepts, the operation it produces, and the primary
// input field's name/kind/source. Tasks not listed fall back to a generic text page,
// so a NEW server task still produces a usable page instead of vanishing.
interface TaskHint {
  title: string;
  category: string;
  accepts: MediaKind;
  produces: Operation;
  /** Name of the primary input field the /prompt endpoint expects for this task. */
  inputName: string;
  inputKind: FieldSpec["kind"];
  inputSource: FieldSpec["source"];
  inputLabel: string;
}

const TASK_HINTS: Record<string, TaskHint> = {
  "text-generation": {
    title: "Text Generation",
    category: "text",
    accepts: "text",
    produces: "chat",
    inputName: "prompt",
    inputKind: "textarea",
    inputSource: "text",
    inputLabel: "Prompt",
  },
  "text2text-generation": {
    title: "Text-to-Text Generation",
    category: "text",
    accepts: "text",
    produces: "chat",
    inputName: "text",
    inputKind: "textarea",
    inputSource: "text",
    inputLabel: "Text",
  },
  "text-summarization": {
    title: "Summarize Text",
    category: "text",
    accepts: "text",
    produces: "summarize",
    inputName: "text",
    inputKind: "textarea",
    inputSource: "text",
    inputLabel: "Text",
  },
  "keyword-extraction": {
    title: "Extract Keywords",
    category: "text",
    accepts: "text",
    produces: "keywords",
    inputName: "text",
    inputKind: "textarea",
    inputSource: "text",
    inputLabel: "Text",
  },
  "sentence-similarity": {
    title: "Sentence Similarity",
    category: "text",
    accepts: "text",
    produces: "metadata",
    inputName: "text",
    inputKind: "textarea",
    inputSource: "text",
    inputLabel: "Text",
  },
  "feature-extraction": {
    title: "Feature Extraction",
    category: "text",
    accepts: "text",
    produces: "metadata",
    inputName: "text",
    inputKind: "textarea",
    inputSource: "text",
    inputLabel: "Text",
  },
  "image-text-to-text": {
    title: "Image Analysis",
    category: "image",
    accepts: "image",
    produces: "analyze",
    inputName: "image_path",
    inputKind: "file",
    inputSource: "selectedIds",
    inputLabel: "Image",
  },
  "automatic-speech-recognition": {
    title: "Audio Transcription",
    category: "audio",
    accepts: "audio",
    produces: "transcribe",
    inputName: "path",
    inputKind: "file",
    inputSource: "selectedIds",
    inputLabel: "Audio file",
  },
  "text-to-image": {
    title: "Text to Image",
    category: "image",
    accepts: "text",
    produces: "metadata",
    inputName: "prompt",
    inputKind: "textarea",
    inputSource: "text",
    inputLabel: "Prompt",
  },
};

// Media-intelligence amenities -> their dedicated, preordained /ml/<name>
// endpoint (the inverse of the Flask app's ML_TASKS map; the route fixes the
// task server-side and pins the reserved ML pool). Tasks absent here (text /
// LLM) keep riding the generic /prompt {task} verb.
const TASK_TO_ML: Record<string, string> = {
  "automatic-speech-recognition": "transcribe",
  "text-summarization": "summarize",
  "keyword-extraction": "keywords",
  "feature-extraction": "embed",
  "sentence-similarity": "similarity",
  "image-text-to-text": "vision",
  "text-to-image": "imagine",
};

function hintFor(task: string): TaskHint {
  return (
    TASK_HINTS[task] ?? {
      // Generic fallback: any unrecognized task takes free text and produces chat output.
      title: task,
      category: "text",
      accepts: "text",
      produces: "chat",
      inputName: "text",
      inputKind: "textarea",
      inputSource: "text",
      inputLabel: "Input",
    }
  );
}

/**
 * Build a PageSpec for one task. The synthesized `task` + `model_key` fields are NOT
 * user-sourced; they ride defaults and are forwarded verbatim by submitPage into the
 * /prompt body. The primary input field is user-facing (text/file).
 */
type Cap = { ready: boolean; extra: string | null };

function pageForTask(
  task: string,
  modelKey: string | undefined,
  cap?: Cap,
): PageSpec {
  const hint = hintFor(task);
  const isUpload = hint.inputKind === "file" || hint.inputKind === "files";

  // A not-ready amenity (its optional extra isn't installed server-side) keeps
  // its page but flags how to enable it, instead of only erroring on submit.
  const title =
    cap && cap.ready === false
      ? `${hint.title}${cap.extra ? ` · needs abstract_hugpy_dev[${cap.extra}]` : " · unavailable"}`
      : hint.title;

  // Amenity tasks post to their dedicated /ml/<name> endpoint (task fixed
  // server-side, reserved ML pool); everything else uses the generic /prompt.
  const mlName = TASK_TO_ML[task];
  const path = mlName ? `/ml/${mlName}` : "/prompt";

  const fields: FieldSpec[] = [
    // The generic /prompt path carries the task as a hidden field; the /ml/<name>
    // route fixes its task, so it's omitted there. model_key still rides the
    // default and is forwarded by submitPage.
    ...(mlName
      ? []
      : [{ name: "task", label: "Task", kind: "text" as const, default: task }]),
    ...(modelKey
      ? [
          {
            name: "model_key",
            label: "Model",
            kind: "text" as const,
            default: modelKey,
          },
        ]
      : []),
    {
      name: hint.inputName,
      label: hint.inputLabel,
      kind: hint.inputKind,
      required: true,
      source: hint.inputSource,
    },
  ];

  return {
    key: `task/${task}`,
    title,
    category: hint.category,
    path,
    method: "POST",
    ...(isUpload ? { isUpload: true } : {}),
    resultKind: "json",
    accepts: [hint.accepts],
    produces: hint.produces,
    fields,
  };
}

let _loaded: Promise<void> | null = null;

export function loadServerPages(): Promise<void> {
  if (_loaded) return _loaded;
  _loaded = (async () => {
    // Idempotent GET → the transport retries on 5xx/network with backoff.
    const res = await request(REGISTRY_ENDPOINT, {
      method: "GET",
      expect: "json",
      meta: { specKey: "registry" },
    });
    if (!res.ok) {
      console.error(
        "[hugpy] /prompt/tasks fetch failed; keeping built-in pages only.",
        errorOf(res),
      );
      return;
    }

    const body = okValue(res);

    // Parse, don't assert: a malformed catalog yields a structured, logged error and
    // a degraded-but-honest UI (built-in pages survive) instead of injecting garbage
    // that explodes three components later.
    const parsed = TaskCatalogSchema.safeParse(body);
    if (!parsed.success) {
      console.error(
        "[hugpy] /prompt/tasks returned an invalid payload; keeping built-in pages only.",
        parsed.error.issues,
      );
      return;
    }

    const { tasks } = parsed.data;
    const defaults = isRecord(parsed.data.defaults) ? parsed.data.defaults : {};

    // Best-effort capability probe (GET /ml): lets amenity tools show a clean
    // "needs abstract_hugpy_dev[X]" hint instead of only erroring on submit.
    const cap = new Map<string, Cap>();
    try {
      const mlRes = await request(`${hugpyConfig.apiBase}/ml`, {
        method: "GET",
        expect: "json",
        meta: { specKey: "ml" },
      });
      if (mlRes.ok) {
        const mlBody = okValue(mlRes) as {
          endpoints?: Record<
            string,
            { task?: string; ready?: boolean; extra?: string | null }
          >;
        };
        for (const v of Object.values(mlBody?.endpoints ?? {})) {
          if (v?.task) cap.set(v.task, { ready: !!v.ready, extra: v.extra ?? null });
        }
      }
    } catch {
      /* no hints if /ml is unreachable */
    }

    for (const task of tasks) {
      try {
        registerPage(pageForTask(task, defaults[task], cap.get(task)));
      } catch (e) {
        // duplicates are fine on hot-reload; log others
        if (!String(e).includes("already registered")) console.warn(e);
      }
    }
  })();
  return _loaded;
}
