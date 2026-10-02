// Runtime configuration for the hugpy UI package.
//
// The console historically talked to the backend over *relative* URLs
// ("/api/...") and relied on the dev server / nginx to proxy "/api" to the
// Flask app. That works when the UI is served from the same origin as the API,
// but a package dropped into someone else's app has no such proxy.
//
// This module introduces a single, mutable runtime config (a process-wide
// singleton) plus URL resolvers that every call site goes through. The default
// keeps the historical behavior exactly: an empty baseUrl means "same origin,
// relative path", so the in-house console is unchanged. Consumers of the npm
// package call `configureHugpy({ baseUrl })` (or wrap their tree in
// `<HugpyProvider baseUrl=…>`) to point it at a real host.

export interface HugpyRuntimeConfig {
  /**
   * Origin of the hugpy API, e.g. "https://api.hugpy.ai".
   * Empty string = same-origin / relative paths (the default, unchanged
   * behavior for the bundled console behind its proxy).
   */
  baseUrl: string;
  /**
   * Fetch implementation to use for all API calls. Defaults to the global
   * fetch. Override to inject auth, retries, or an SSR-safe fetch.
   */
  fetch: typeof fetch;
  /**
   * Extra headers merged into every API request (e.g. an Authorization
   * bearer token). May be a function, evaluated per-request, for tokens that
   * rotate. Per-call headers take precedence over these.
   */
  headers?: HeadersInit | (() => HeadersInit | Promise<HeadersInit>);
  /**
   * Credentials mode for every request. Set to "include" for cookie-based
   * auth across origins.
   */
  credentials?: RequestCredentials;
}

const defaults: HugpyRuntimeConfig = {
  baseUrl: '',
  fetch: (...args: Parameters<typeof fetch>) => globalThis.fetch(...args),
};

let current: HugpyRuntimeConfig = { ...defaults };

/** Merge a partial config into the active runtime config. */
export function configureHugpy(partial: Partial<HugpyRuntimeConfig>): void {
  current = { ...current, ...partial };
}

/** Read the active runtime config. */
export function getHugpyConfig(): HugpyRuntimeConfig {
  return current;
}

/** Reset to defaults (relative, global fetch). Mainly for tests. */
export function resetHugpyConfig(): void {
  current = { ...defaults };
}

function stripTrailingSlash(s: string): string {
  return s.replace(/\/+$/, '');
}

const ABSOLUTE = /^[a-z][a-z0-9+.-]*:\/\//i;

/**
 * Resolve an API path to a fetchable URL. Relative paths ("/api/models") are
 * prefixed with the configured baseUrl; already-absolute URLs pass through.
 * With no baseUrl set, returns the path unchanged (same-origin).
 */
export function resolveApiUrl(path: string): string {
  if (ABSOLUTE.test(path)) return path;
  const base = stripTrailingSlash(current.baseUrl || '');
  if (!base) return path;
  return base + (path.startsWith('/') ? path : '/' + path);
}

/**
 * Absolute origin for display strings, EventSource, and <img> src — contexts
 * where a bare relative path is undesirable or impossible. Falls back to the
 * browser's window origin when no baseUrl is configured.
 */
export function resolveApiOrigin(): string {
  const base = stripTrailingSlash(current.baseUrl || '');
  if (base) return base;
  if (typeof window !== 'undefined' && window.location) return window.location.origin;
  return '';
}

/** Merge configured headers/credentials into a RequestInit (per-call wins). */
async function withHugpyRequest(init?: RequestInit): Promise<RequestInit> {
  const cfg = current;
  const headers = new Headers(init?.headers || {});
  if (cfg.headers) {
    const extra = typeof cfg.headers === 'function' ? await cfg.headers() : cfg.headers;
    new Headers(extra).forEach((value, key) => {
      if (!headers.has(key)) headers.set(key, value);
    });
  }
  const merged: RequestInit = { ...init, headers };
  if (cfg.credentials && merged.credentials == null) merged.credentials = cfg.credentials;
  return merged;
}

/**
 * The single fetch entry point for the whole UI: resolves the URL against the
 * configured baseUrl and merges configured headers/credentials. All raw
 * `fetch('/api/...')` call sites go through this.
 */
export async function hugpyFetch(path: string, init?: RequestInit): Promise<Response> {
  return current.fetch(resolveApiUrl(path), await withHugpyRequest(init));
}
