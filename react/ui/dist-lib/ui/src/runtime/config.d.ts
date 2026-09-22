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
/** Merge a partial config into the active runtime config. */
export declare function configureHugpy(partial: Partial<HugpyRuntimeConfig>): void;
/** Read the active runtime config. */
export declare function getHugpyConfig(): HugpyRuntimeConfig;
/** Reset to defaults (relative, global fetch). Mainly for tests. */
export declare function resetHugpyConfig(): void;
/**
 * Resolve an API path to a fetchable URL. Relative paths ("/api/models") are
 * prefixed with the configured baseUrl; already-absolute URLs pass through.
 * With no baseUrl set, returns the path unchanged (same-origin).
 */
export declare function resolveApiUrl(path: string): string;
/**
 * Absolute origin for display strings, EventSource, and <img> src — contexts
 * where a bare relative path is undesirable or impossible. Falls back to the
 * browser's window origin when no baseUrl is configured.
 */
export declare function resolveApiOrigin(): string;
/**
 * The single fetch entry point for the whole UI: resolves the URL against the
 * configured baseUrl and merges configured headers/credentials. All raw
 * `fetch('/api/...')` call sites go through this.
 */
export declare function hugpyFetch(path: string, init?: RequestInit): Promise<Response>;
