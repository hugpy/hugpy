import { hugpyConfig } from "../../../config";

export type MlCap = { ready: boolean; extra: string | null };

// GET /ml -> { endpoints: { "/ml/<name>": { task, ready, extra } }, pool }.
// We key the map by the /ml PATH so a PageSpec (whose `path` is "/ml/<name>")
// looks up directly. find_spec on the server is cheap and never imports the
// heavy dep; on any failure we return an empty map and the UI degrades to
// all-runnable (graceful, matching the page console's prior behaviour).
export async function loadMlCapabilities(
  signal?: AbortSignal,
): Promise<Map<string, MlCap>> {
  const out = new Map<string, MlCap>();
  try {
    const res = await fetch(`${hugpyConfig.apiBase}/ml`, {
      credentials: hugpyConfig.withCredentials ? "include" : "same-origin",
      signal,
    });
    if (!res.ok) return out;
    const body = (await res.json()) as {
      endpoints?: Record<string, { ready?: boolean; extra?: string | null }>;
    };
    for (const [path, v] of Object.entries(body?.endpoints ?? {})) {
      out.set(path, { ready: !!v?.ready, extra: v?.extra ?? null });
    }
  } catch {
    /* /ml unreachable -> empty map -> all tools treated as runnable */
  }
  return out;
}

// A tool with no capability gate (e.g. /prompt text-gen) or one the server
// reports ready is runnable; otherwise it's gated behind an extra.
export function pageReady(
  specPath: string | undefined,
  caps: Map<string, MlCap>,
): MlCap {
  return (specPath && caps.get(specPath)) || { ready: true, extra: null };
}
