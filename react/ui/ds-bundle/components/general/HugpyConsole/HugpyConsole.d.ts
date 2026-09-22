import * as React from 'react';

/**
 * HugpyConsole — from @hugpy/ui@0.2.0.
 */
export interface HugpyConsoleProps {
  /** Origin of the hugpy API, e.g. "https://api.hugpy.ai". Empty string = same-origin / relative paths (the default, unchange */
  baseUrl?: string;
  /** Fetch implementation to use for all API calls. Defaults to the global fetch. Override to inject auth, retries, or an SSR */
  fetch?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  /** Extra headers merged into every API request (e.g. an Authorization bearer token). May be a function, evaluated per-reque */
  headers?: [string, string][] | Record<string, string> | Headers | (() => HeadersInit | Promise<HeadersInit>);
  /** Credentials mode for every request. Set to "include" for cookie-based auth across origins. */
  credentials?: "include" | "omit" | "same-origin";
}

export declare const HugpyConsole: React.ComponentType<HugpyConsoleProps>;
