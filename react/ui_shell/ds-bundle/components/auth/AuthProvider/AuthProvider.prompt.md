AuthProvider from @hugpy/ui. Use via `window.HugpyUI.AuthProvider` (bundle loaded from the root `_ds_bundle.js`). Wrap the tree in `<DemoProvider>` (full provider chain in README.md — components read theme/i18n from that context).

## Props

```ts
interface AuthProviderProps {
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
```
