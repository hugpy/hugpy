// Shared ComfyUI-embed helpers — the ONE security gate every ComfyUI URL passes
// before it can become an iframe `src`, plus the embed-hazard classifier behind
// the inline warnings.
//
// Extracted from ComfyStation.tsx when ComfyUI became an in-page OVERLAY
// (ComfyPanel, mounted in the sidebar beside Console) so the panel and the
// station cannot drift apart on the part that matters. There must be exactly one
// sanitizer: a second, subtly-weaker copy is how a `javascript:` URL eventually
// reaches an iframe.
//
// ── WHY THIS EXISTS (do not delete the gate) ────────────────────────────────
// The ComfyUI endpoint is USER-EDITABLE and PERSISTED, so a hostile or mistyped
// value could carry a `javascript:` / `data:` / `blob:` / `file:` scheme. An
// unsanitized value handed to iframe src executes in the parent context. Every
// value — typed AND rehydrated from storage — goes through sanitizeEmbedUrl().
//
// This is also precisely why a ComfyUI iframe is NOT the same trust case as the
// console's: the console is first-party, same-origin, fixed-href content (see
// ConsolePanel's header), while this URL is user-supplied and foreign. The
// console frame is deliberately unsandboxed; a ComfyUI frame MUST stay sandboxed.

/** The ONLY schemes allowed to reach iframe src. Everything else — javascript:,
 *  data:, blob:, file:, vbscript:, mailto:, about:, … — is rejected so a
 *  persisted or typed hostile URI can never execute in the parent context. */
const ALLOWED_SCHEMES = new Set(["http:", "https:"]);

/**
 * Return `raw` as a normalized URL string ONLY if it parses as an absolute
 * http:// or https:// URL; otherwise null. Uses `new URL()` (so relative junk,
 * bare hosts, and unknown schemes throw or fail the allowlist) plus a strict
 * scheme allowlist. Call it on typed input AND on any persisted value.
 */
export function sanitizeEmbedUrl(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return null; // relative / malformed / scheme-less — never embeddable.
  }
  if (!ALLOWED_SCHEMES.has(parsed.protocol)) return null; // reject javascript:, data:, blob:, file:, …
  return parsed.href;
}

/** Embed-hazard classification of a (sanitized) URL, for the inline warning. */
export type EmbedRisk = "http" | "localhost" | null;

/**
 * Flag URLs that an HTTPS page cannot embed: http:// (mixed content) or a
 * loopback host (localhost / 127.0.0.1 / ::1 — not reachable/TLS-served from the
 * user's browser through the deployed HTTPS origin). Runs on the sanitized value,
 * so anything non-http/https is already null and returns no risk.
 */
export function embedRisk(raw: string): EmbedRisk {
  const safe = sanitizeEmbedUrl(raw);
  if (!safe) return null;
  const u = new URL(safe);
  if (u.protocol === "http:") return "http";
  const host = u.hostname.toLowerCase();
  const isLoopback =
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host === "127.0.0.1" ||
    host === "::1" ||
    host === "0.0.0.0";
  return isLoopback ? "localhost" : null;
}

/** The sandbox tokens a ComfyUI frame runs with. Permissive on purpose (ComfyUI
 *  needs scripts, same-origin, forms, downloads and popups to function) but
 *  NEVER absent: the URL is user-supplied and foreign, unlike the console's. */
export const COMFY_SANDBOX =
  "allow-scripts allow-same-origin allow-forms allow-downloads allow-popups allow-popups-to-escape-sandbox";

/** Session-persisted (per-tab) key for the entered endpoint URL. Shared so the
 *  panel and the station show the SAME endpoint instead of two half-configured
 *  ones. */
export const COMFY_URL_KEY = "vi.comfy.url.v1";
