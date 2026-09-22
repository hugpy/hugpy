// The demo transport: ONE window.fetch interceptor that backs both demo flavors.
//
//   CANNED (?demo=1)        → answer every hugpy call from fixtures.ts. The real
//                             chat shell + transport run unchanged; only the wire
//                             is faked. Streaming chat replays a scripted SSE
//                             conversation through a genuine ReadableStream.
//
//   LIVE (?demo=live&key=K) → pass through to the real backend, injecting
//                             `Authorization: Bearer K` (so a gated /ml backend
//                             accepts the call) and, if `&api=<base>` is set,
//                             rewriting same-origin /api calls to that base.
//
// Patching window.fetch (not a config hook) is deliberate: the arm's transport,
// the chat SSE client, and the model-list fetch ALL call the global fetch, so a
// single wrapper covers every path. Fully restorable on uninstall.

import { getDemoConfig } from "./mode";
import { hugpyConfig } from "../config";
import * as fx from "./fixtures";

const enc = new TextEncoder();

// ── request introspection ────────────────────────────────────────────────────
function pathOf(input: RequestInfo | URL): string {
  let url =
    typeof input === "string"
      ? input
      : input instanceof Request
        ? input.url
        : String((input as URL)?.toString?.() ?? input ?? "");
  if (/^https?:\/\//i.test(url)) {
    try {
      const u = new URL(url);
      url = u.pathname + u.search;
    } catch {
      /* keep raw */
    }
  }
  return url;
}

function methodOf(input: RequestInfo | URL, init?: RequestInit): string {
  return String(
    init?.method ??
      (input instanceof Request ? input.method : undefined) ??
      "GET",
  ).toUpperCase();
}

// The arm builds every URL as `${hugpyConfig.apiBase}${path}` (default "/api").
// Normalize a request path back to the contract-relative path so route matching
// is independent of the configured base.
function apiPrefix(): string {
  const b = hugpyConfig.apiBase || "/api";
  if (/^https?:\/\//i.test(b)) {
    try {
      return new URL(b).pathname.replace(/\/+$/, "");
    } catch {
      return "/api";
    }
  }
  return b.replace(/\/+$/, "");
}

function normalize(path: string): string {
  const pre = apiPrefix();
  if (pre && path.startsWith(pre)) return path.slice(pre.length) || "/";
  return path;
}

async function readBody(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<string> {
  if (init && typeof init.body === "string") return init.body;
  if (input instanceof Request) {
    try {
      return await input.clone().text();
    } catch {
      return "";
    }
  }
  return "";
}

function formDataOf(
  input: RequestInfo | URL,
  init?: RequestInit,
): FormData | null {
  if (init && init.body instanceof FormData) return init.body;
  return null;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// ── scripted streaming chat (genuine ReadableStream of SSE frames) ───────────
let _chatTurn = 0;

function chunkText(text: string): string[] {
  return text.match(/\S+\s*/g) ?? [text];
}

function sseStream(
  text: string,
  requestId: string,
  signal?: AbortSignal | null,
): ReadableStream<Uint8Array> {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const chunks = chunkText(text);

  return new ReadableStream<Uint8Array>({
    start(controller) {
      let i = 0;
      const send = (obj: unknown) => {
        try {
          controller.enqueue(enc.encode("data: " + JSON.stringify(obj) + "\n\n"));
        } catch {
          /* closed */
        }
      };
      const finish = () => {
        send({
          type: "done",
          request_id: requestId,
          input_tokens: 64,
          output_chunks: chunks.length,
          finish_reason: "stop",
        });
        try {
          controller.close();
        } catch {
          /* closed */
        }
      };
      const step = () => {
        if (cancelled) return;
        if (i >= chunks.length) return finish();
        send({ type: "token", request_id: requestId, text: chunks[i++] });
        timer = setTimeout(step, 16);
      };
      const onAbort = () => {
        cancelled = true;
        if (timer) clearTimeout(timer);
        try {
          controller.close();
        } catch {
          /* closed */
        }
      };
      if (signal) {
        if (signal.aborted) return onAbort();
        signal.addEventListener("abort", onAbort, { once: true });
      }
      // brief lead-in so the "thinking" state is visible before tokens flow
      timer = setTimeout(step, 140);
    },
    cancel() {
      cancelled = true;
      if (timer) clearTimeout(timer);
    },
  });
}

async function chatStreamResponse(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const bodyText = await readBody(input, init);
  let requestId = "demo";
  try {
    const b = JSON.parse(bodyText) as { request_id?: unknown };
    if (typeof b?.request_id === "string") requestId = b.request_id;
  } catch {
    /* keep default */
  }
  const reply = fx.CHAT_REPLIES[_chatTurn % fx.CHAT_REPLIES.length];
  _chatTurn += 1;
  const signal =
    init?.signal ?? (input instanceof Request ? input.signal : undefined);
  return new Response(sseStream(reply, requestId, signal), {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
    },
  });
}

// ── GET routes → canned fixtures ─────────────────────────────────────────────
function getRoute(path: string): unknown {
  if (path === "/version") return fx.VERSION;
  if (path === "/models") return fx.MODELS;
  if (path === "/prompt/tasks") return fx.PROMPT_TASKS;
  if (path === "/ml/gate") return fx.ML_GATE;
  return undefined;
}

// ── POST /prompt: routing vs. file-suggestion vs. plain generation ───────────
// One verb, three intents (toolRouting.routeTool / main.suggestForFile / the
// text/generate tool). Sniff the system prompt to answer each believably.
async function promptResponse(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<unknown> {
  const bodyText = await readBody(input, init);
  let requestId = "demo";
  let system = "";
  try {
    const b = JSON.parse(bodyText) as {
      request_id?: unknown;
      messages?: { role?: string; content?: string }[];
    };
    if (typeof b?.request_id === "string") requestId = b.request_id;
    system =
      (b?.messages ?? []).find((m) => m?.role === "system")?.content ?? "";
  } catch {
    /* keep defaults */
  }

  // routeTool → reply with the strict routing JSON, routing to NONE so the
  // canned demo just chats (no tool dependency).
  if (/route a user|ONE media-intelligence tool|specKey/i.test(system)) {
    return {
      ok: true,
      request_id: requestId,
      text: '{"specKey":"","reason":"conversational"}',
    };
  }
  // suggestForFile → a short, friendly nudge about the attached file.
  if (/suggest what you can do|just attached/i.test(system)) {
    return {
      ok: true,
      request_id: requestId,
      text: "I can transcribe it, summarize the contents, and pull out the key topics — want me to run the full analysis?",
    };
  }
  // anything else (e.g. the text/generate tool) → a generic helpful line.
  return {
    ok: true,
    request_id: requestId,
    text: fx.CHAT_REPLIES[0],
  };
}

// ── mutating routes → canned tool outputs ────────────────────────────────────
async function mutateRoute(
  method: string,
  path: string,
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<unknown> {
  if (method === "POST") {
    if (path === "/prompt") return promptResponse(input, init);
    if (path === "/ml/summarize") return fx.SUMMARY_RESULT;
    if (path === "/ml/keywords") return fx.KEYWORDS_RESULT;
    if (path === "/ml/extract") return fx.EXTRACT_RESULT;
    if (path === "/ml/transcribe") return fx.TRANSCRIBE_RESULT;
    if (path === "/ml/vision") return fx.VISION_RESULT;
    if (path === "/ml/embed") return fx.EMBED_RESULT;
    if (path === "/ml/imagine") return fx.IMAGINE_RESULT;
    if (path === "/ml/fetch") return fx.FETCH_RESULT;
    if (path === "/media/analyze") {
      const bodyText = await readBody(input, init);
      let kind = "document";
      let source = "your-file";
      try {
        const b = JSON.parse(bodyText) as { kind?: string; source?: string };
        if (typeof b?.kind === "string") kind = b.kind;
        if (typeof b?.source === "string") source = b.source;
      } catch {
        /* keep defaults */
      }
      return { ok: true, result: fx.cannedDI(kind, source) };
    }
    if (path === "/uploads") {
      const fd = formDataOf(input, init);
      // The chat composer posts under "file"; the classic receptacle posts
      // "files" — accept either so the canned ref keeps the real filename.
      const f = fd?.get("file") ?? fd?.get("files");
      if (f && typeof f === "object" && "name" in f) {
        const file = f as File;
        return {
          id: `demo-${file.name || "file"}`,
          name: file.name || "uploaded-file",
          type: file.type || "file",
        };
      }
      return { id: "demo-upload", name: "uploaded-file", type: "file" };
    }
    if (/^\/llm\/chat\/cancel\//.test(path)) return { ok: true };
  }
  if (method === "PUT" && path === "/ml/gate") return { require_key: false };
  return undefined;
}

async function cannedFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const path = normalize(pathOf(input));
  const method = methodOf(input, init);
  try {
    if (method === "POST" && path === "/chat/stream")
      return chatStreamResponse(input, init);

    if (method === "GET") {
      const g = getRoute(path);
      if (g !== undefined) return jsonResponse(g);
      // Unmatched same-origin API GET → an empty list never crashes a consumer.
      return jsonResponse([]);
    }

    const m = await mutateRoute(method, path, input, init);
    if (m !== undefined) return jsonResponse(m);
    return jsonResponse({ ok: true, demo: true });
  } catch (e) {
    return jsonResponse(
      { ok: false, error: "demo shim error", detail: String((e as Error)?.message ?? e) },
      500,
    );
  }
}

// ── live passthrough with key injection / base rewrite ───────────────────────
async function liveFetch(
  orig: typeof fetch,
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const cfg = getDemoConfig();
  if (!cfg.key && !cfg.apiBase) return orig(input, init);

  const pre = apiPrefix();
  // Normalize the same way cannedFetch does (pathOf for strings too), so an
  // absolute VITE_HUGPY_API_BASE (e.g. https://dev.hugpy.ai/api) still matches
  // and gets the Bearer key injected, not just same-origin "/api" builds.
  const reqPath = pathOf(input);
  const isApiCall = reqPath.startsWith(pre);

  if (!isApiCall) return orig(input, init);

  const headers = new Headers(
    init?.headers ?? (input instanceof Request ? input.headers : undefined),
  );
  if (cfg.key && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${cfg.key}`);
  }

  // Optional cross-origin rewrite for same-origin string URLs only (CORS is the
  // backend's responsibility; same-origin is the default, friction-free path).
  let target: RequestInfo | URL = input;
  if (cfg.apiBase && typeof input === "string" && input.startsWith(pre)) {
    target = cfg.apiBase + input.slice(pre.length);
  }

  return orig(target, { ...init, headers });
}

// ── install / uninstall (idempotent, restorable) ─────────────────────────────
let _installed = false;
let _original: typeof fetch | null = null;

export function installDemoTransport(): void {
  const cfg = getDemoConfig();
  if (_installed || cfg.mode === "off" || typeof window === "undefined") return;
  // Live mode with neither a key nor a base override needs no interception.
  if (cfg.mode === "live" && !cfg.key && !cfg.apiBase) return;

  _installed = true;
  _original = window.fetch.bind(window);
  const orig = _original;
  window.fetch =
    cfg.mode === "canned"
      ? ((input: RequestInfo | URL, init?: RequestInit) =>
          cannedFetch(input, init)) as typeof fetch
      : ((input: RequestInfo | URL, init?: RequestInit) =>
          liveFetch(orig, input, init)) as typeof fetch;
}

export function uninstallDemoTransport(): void {
  if (!_installed || !_original || typeof window === "undefined") return;
  window.fetch = _original;
  _installed = false;
}
