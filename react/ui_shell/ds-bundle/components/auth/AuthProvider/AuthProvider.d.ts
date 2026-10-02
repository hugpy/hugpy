import * as React from 'react';

/**
 * AuthProvider — from @hugpy/ui@0.2.0.
 */
export interface AuthProviderProps {
  children?: React.ReactNode;
  /** Auth service origin. If omitted, resolved from GET /api/auth/config. */
  base?: string;
  /** Force a mode. If omitted, resolved from config. */
  mode?: "open" | "external";
  /** Override one or more endpoint paths. */
  endpoints?: Partial<AuthEndpoints>;
  /** Custom fetch (e.g. to inject headers). Defaults to global fetch. */
  fetch?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  /** Credentials mode for auth requests. Defaults to 'include' (cookie auth). */
  credentials?: "include" | "omit" | "same-origin";
}

export declare const AuthProvider: React.ComponentType<AuthProviderProps>;
