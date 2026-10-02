// Shared fetch helper that always surfaces the API's real message.
//
// Reads the body once as text, then tries to parse it as JSON. If the backend
// returned an error payload ({error|detail|message}) or a non-JSON body (an
// HTML proxy page, an empty response from an unreachable backend, etc.), we
// throw an Error carrying that text — instead of letting `r.json()` blow up
// with the opaque "JSON.parse: unexpected character at line 1 column 1".
// Shared fetch helper that always surfaces the API's real message.
//
// Reads the body once as text, then tries to parse it as JSON. If the backend
// returned an error payload ({error|detail|message}) or a non-JSON body, we
// throw an Error carrying that text instead of letting `r.json()` fail with
// an opaque JSON.parse error.
//
// All requests go through `hugpyFetch`, which resolves relative "/api/..."
// paths against the configured baseUrl and merges configured auth headers.

import { hugpyFetch } from "./runtime/config";

export type ApiErrorPayload = {
  error?: string;
  detail?: string;
  message?: string;
  [key: string]: unknown;
};

export type UploadFileResponse = {
  path: string;
  name: string;
  size: number;
  [key: string]: unknown;
};

function getApiErrorMessage(data: unknown, fallback: string): string {
  if (data && typeof data === "object") {
    const payload = data as ApiErrorPayload;

    if (typeof payload.error === "string" && payload.error.trim()) {
      return payload.error;
    }

    // Typed worker-op errors: {error: {code, message, ...}} — show the code
    // and the recorded message, not a bare status line.
    if (payload.error && typeof payload.error === "object") {
      const e = payload.error as { code?: unknown; message?: unknown };
      const text = [e.code, e.message].filter(x => x != null && String(x).trim()).join(": ");
      if (text) return `${fallback} ${text}`;
    }

    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }

    if (typeof payload.message === "string" && payload.message.trim()) {
      return payload.message;
    }

    if (typeof payload.reason === "string" && payload.reason.trim()) {
      return payload.reason;
    }

    // No error/detail/message/reason field: the body itself is the record.
    try {
      return `${fallback}: ${JSON.stringify(data)}`;
    } catch {
      /* fall through */
    }
  }

  return fallback;
}

function parseJsonBody<T>(body: string, status: number): T {
  try {
    return JSON.parse(body) as T;
  } catch {
    throw new Error(body.trim() || `HTTP ${status} (empty response)`);
  }
}

// The console Help panel's context strip carries "the last error the console
// showed" — announced here as a window event (components/HelpPanel/helpBus.js).
function announceApiError(url: string, status: number, message: string): void {
  try {
    if (typeof window !== "undefined" && typeof CustomEvent !== "undefined") {
      window.dispatchEvent(new CustomEvent("hugpy:api-error", {
        detail: { url, status, message: String(message || "") },
      }));
    }
  } catch {
    /* never let error reporting break the caller */
  }
}

export async function fetchJson<T = unknown>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const r = await hugpyFetch(url, options);
  const body = await r.text();

  let data: T;
  try {
    data = parseJsonBody<T>(body, r.status);
  } catch (e) {
    announceApiError(url, r.status, (e as Error).message);
    throw e;
  }

  if (!r.ok) {
    const message = getApiErrorMessage(data, `HTTP ${r.status}`);
    announceApiError(url, r.status, message);
    throw new Error(message);
  }

  return data;
}

export async function uploadFile(file: File): Promise<UploadFileResponse> {
  const form = new FormData();
  form.append("file", file);

  return fetchJson<UploadFileResponse>("/api/uploads", {
    method: "POST",
    body: form,
  });
}