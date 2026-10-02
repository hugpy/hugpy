HugpyConsole from @hugpy/ui. Use via `window.HugpyUI.HugpyConsole` (bundle loaded from the root `_ds_bundle.js`). Wrap the tree in `<DemoProvider>` (full provider chain in README.md — components read theme/i18n from that context).

## Props

```ts
interface HugpyConsoleProps {
  /** Origin of the hugpy API, e.g. "https://api.hugpy.ai". Empty string = same-origin / relative paths (the default, unchange */
  baseUrl?: string;
  /** Fetch implementation to use for all API calls. Defaults to the global fetch. Override to inject auth, retries, or an SSR */
  fetch?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  /** Extra headers merged into every API request (e.g. an Authorization bearer token). May be a function, evaluated per-reque */
  headers?: [string, string][] | Record<string, string> | Headers | (() => HeadersInit | Promise<HeadersInit>);
  /** Credentials mode for every request. Set to "include" for cookie-based auth across origins. */
  credentials?: "include" | "omit" | "same-origin";
}
```
