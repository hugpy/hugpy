// Phase 5 — the one transport all hugpy calls go through.
//
// Placed at the shared hugpy level (sibling of config.ts) rather than under console/
// so the lower-layer receptacle upload can use it too without a console⇄receptacle
// import cycle. Every call gets: auth (configurable credentials), an AbortSignal,
// a timeout ceiling, GET-only retry/backoff, a correlation id, and a typed
// Result<T, AppError> return — failures cross the boundary as data, never as throws.
import { hugpyConfig } from "../config";
import { recordRequest, type Outcome } from "./telemetry";

export type AppError =
  | { kind: "network"; message: string; requestId: string }
  | { kind: "timeout"; ms: number; requestId: string }
  | { kind: "aborted"; requestId: string }
  | { kind: "unauthorized"; status: number; requestId: string }
  | {
      kind: "server";
      status: number;
      message: string;
      traceId?: string;
      requestId: string;
    };

export type Result<T> =
  | { ok: true; value: T }
  | { ok: false; error: AppError };

export interface RequestOptions {
  method?: string;
  body?: BodyInit | null;
  headers?: Record<string, string>;
  /** Caller-owned abort (component unmount, explicit Cancel). */
  signal?: AbortSignal;
  /** Per-call timeout ceiling; defaults to DEFAULT_TIMEOUT_MS. */
  timeoutMs?: number;
  /** Extra retries — honored for idempotent GET/HEAD only. */
  retries?: number;
  /** Body parse hint; default tries JSON then falls back to text. */
  expect?: "json" | "text";
  /** Caller context for telemetry — the transport doesn't know these. */
  meta?: { specKey?: string; operation?: string };
}

const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_GET_RETRIES = 2;

let _counter = 0;
function newRequestId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    _counter += 1;
    const t = typeof performance !== "undefined" ? performance.now() : 0;
    return `req_${_counter.toString(36)}_${Math.floor(t).toString(36)}`;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function combineSignals(
  caller: AbortSignal | undefined,
  timeout: AbortSignal,
): AbortSignal {
  if (!caller) return timeout;
  const anyFn = (AbortSignal as { any?: (s: AbortSignal[]) => AbortSignal }).any;
  if (typeof anyFn === "function") return anyFn([caller, timeout]);

  const ctrl = new AbortController();
  if (caller.aborted || timeout.aborted) ctrl.abort();
  else {
    const onAbort = () => ctrl.abort();
    caller.addEventListener("abort", onAbort, { once: true });
    timeout.addEventListener("abort", onAbort, { once: true });
  }
  return ctrl.signal;
}

async function parseBody(res: Response, expect?: "json" | "text"): Promise<unknown> {
  const text = await res.text();
  if (expect === "text") return text;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function serverMessage(parsed: unknown, status: number): string {
  if (isRecord(parsed) && "error" in parsed) return String(parsed.error);
  if (typeof parsed === "string" && parsed.trim()) return parsed;
  return `HTTP ${status}`;
}

function backoff(attempt: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 250 * 2 ** attempt));
}

export async function request<T = unknown>(
  url: string,
  opts: RequestOptions = {},
): Promise<Result<T>> {
  const requestId = newRequestId();
  const method = (opts.method ?? "GET").toUpperCase();
  const idempotent = method === "GET" || method === "HEAD";
  const maxAttempts = idempotent
    ? Math.max(1, (opts.retries ?? DEFAULT_GET_RETRIES) + 1)
    : 1;
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const startedAt = typeof performance !== "undefined" ? performance.now() : 0;

  let lastError: AppError = {
    kind: "network",
    message: "no attempt made",
    requestId,
  };
  let final: Result<T> | null = null;
  let finalStatus: number | undefined;
  let finalTraceId: string | undefined;

  for (let attempt = 0; attempt < maxAttempts && !final; attempt++) {
    const timeoutCtrl = new AbortController();
    const timer = setTimeout(() => timeoutCtrl.abort(), timeoutMs);

    try {
      const res = await fetch(url, {
        method,
        body: opts.body ?? null,
        // Only attach the correlation header when explicitly enabled — a custom header
        // forces a CORS preflight that the upload endpoint may not answer.
        headers: {
          ...(hugpyConfig.sendRequestId ? { "X-Request-Id": requestId } : {}),
          ...(opts.headers ?? {}),
        },
        signal: combineSignals(opts.signal, timeoutCtrl.signal),
        credentials: hugpyConfig.withCredentials ? "include" : "same-origin",
      });
      clearTimeout(timer);
      finalStatus = res.status;

      if (res.status === 401 || res.status === 403) {
        final = {
          ok: false,
          error: { kind: "unauthorized", status: res.status, requestId },
        };
        break;
      }

      const parsed = await parseBody(res, opts.expect);

      if (!res.ok) {
        const traceId =
          res.headers.get("x-trace-id") ??
          res.headers.get("x-request-id") ??
          undefined;
        finalTraceId = traceId;
        const error: AppError = {
          kind: "server",
          status: res.status,
          message: serverMessage(parsed, res.status),
          ...(traceId ? { traceId } : {}),
          requestId,
        };
        if (idempotent && res.status >= 500 && attempt < maxAttempts - 1) {
          lastError = error;
          await backoff(attempt);
          continue;
        }
        final = { ok: false, error };
        break;
      }

      final = { ok: true, value: parsed as T };
    } catch (e) {
      clearTimeout(timer);

      // Caller aborted (unmount / Cancel) wins over timeout/network classification.
      if (opts.signal?.aborted) {
        final = { ok: false, error: { kind: "aborted", requestId } };
        break;
      }
      if (timeoutCtrl.signal.aborted) {
        lastError = { kind: "timeout", ms: timeoutMs, requestId };
      } else {
        lastError = {
          kind: "network",
          message: e instanceof Error ? e.message : String(e),
          requestId,
        };
      }
      if (idempotent && attempt < maxAttempts - 1) {
        await backoff(attempt);
        continue;
      }
      final = { ok: false, error: lastError };
    }
  }

  const result: Result<T> = final ?? { ok: false, error: lastError };
  const outcome: Outcome = result.ok ? "ok" : errorOf(result).kind;

  recordRequest({
    requestId,
    url,
    method,
    outcome,
    durationMs: Math.round(
      (typeof performance !== "undefined" ? performance.now() : 0) - startedAt,
    ),
    ...(finalStatus != null ? { status: finalStatus } : {}),
    ...(finalTraceId ? { traceId: finalTraceId } : {}),
    ...(opts.meta?.specKey ? { specKey: opts.meta.specKey } : {}),
    ...(opts.meta?.operation ? { operation: opts.meta.operation } : {}),
  });

  return result;
}

// Guarded Result accessors. TS only narrows discriminated unions with strictNullChecks,
// which is OFF in the app's root tsconfig — so call sites can't rely on `if (r.ok)` to
// narrow the negative branch. Each accessor centralizes a single cast; call only after
// checking `r.ok` (errorOf) / in the ok branch (okValue).
export function errorOf<T>(r: Result<T>): AppError {
  return (r as { error: AppError }).error;
}
export function okValue<T>(r: Result<T>): T {
  return (r as { value: T }).value;
}

/** Human-readable, kind-aware message for surfacing an AppError in the UI. */
export function describeAppError(error: AppError): string {
  switch (error.kind) {
    case "unauthorized":
      return "Your session has expired. Please sign in again.";
    case "timeout":
      return `The request timed out after ${Math.round(error.ms / 1000)}s.`;
    case "aborted":
      return "Cancelled.";
    case "network":
      return `Network error: ${error.message}`;
    case "server":
      return error.traceId
        ? `${error.message} (trace ${error.traceId})`
        : error.message;
  }
}
