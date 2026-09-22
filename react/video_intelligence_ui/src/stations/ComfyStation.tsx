// ComfyUI viewer station — a modular "viewer" station that EMBEDS an external
// ComfyUI endpoint in an <iframe>, with a user-editable, session-persisted URL.
// Sibling to the studio stations (Generate / Frames / crops), registered the
// same way (registry.ts, group:"studio").
//
// ── MIXED-CONTENT CONSTRAINT (read before touching the iframe src) ──────────
// The video UI is served over HTTPS (dev.hugpy.ai/video, llm.hugpy.ai/app/video).
// A page loaded over https:// CANNOT embed an http:// (or http://127.0.0.1:8188)
// resource — the browser silently blocks the mixed-content subframe and the user
// sees a blank frame with NO error. So the endpoint MUST be reached over HTTPS
// (e.g. https://comfy.hugpy.ai). This component detects http:// / localhost /
// 127.0.0.1 targets and warns that an HTTPS page will refuse to embed them.
//
// ── FRAME-ANCESTORS (proxy-side) ────────────────────────────────────────────
// Even over HTTPS, the endpoint must permit framing FROM the video-UI origin:
// the comfy.hugpy.ai proxy has to send `Content-Security-Policy: frame-ancestors`
// listing the video-UI origin (and must NOT send `X-Frame-Options: DENY`).
// ComfyUI itself has no auth — keep TLS + any auth at the proxy, not in this arm.
//
// ── SECURITY (scheme allowlist) ─────────────────────────────────────────────
// The URL is user-editable AND persisted, so a hostile/typo value could carry a
// `javascript:` / `data:` / `blob:` / `file:` scheme. We NEVER hand an unsanitized
// value to iframe src: sanitizeEmbedUrl() parses with `new URL()` and returns the
// href ONLY for http:/https:, else null — enforced on typed input AND on the
// value rehydrated from sessionStorage.
//
// That gate (and the sandbox token list, and the storage key) now live in
// ./comfyEmbed so this station and the ComfyPanel overlay share EXACTLY one
// implementation. A second, subtly-weaker copy is how a `javascript:` URL
// eventually reaches an iframe — do not re-inline them here.
//
// ── THIS STATION IS NO LONGER IN THE TAB STRIP (2026-08-04) ────────────────
// ComfyUI is reached from the sidebar overlay (ComfyPanel) now. The station is
// kept, `navHidden`, so an existing /comfy bookmark still resolves instead of
// bouncing to the default tab.
import { useState, type FormEvent } from "react";
import type { StationSpec } from "./types";
import { hugpyConfig } from "../config";
import {
  sanitizeEmbedUrl,
  embedRisk,
  COMFY_SANDBOX,
  COMFY_URL_KEY as URL_KEY,
} from "./comfyEmbed";

export { sanitizeEmbedUrl } from "./comfyEmbed";

// Read the persisted URL, falling back to the configured default (which itself
// defaults to "" — no secret is baked in). Best-effort: storage may be unavailable.
function readPersistedUrl(): string {
  try {
    const stored = sessionStorage.getItem(URL_KEY);
    if (stored != null) return stored;
  } catch {
    /* storage unavailable — fall back to config default */
  }
  return hugpyConfig.comfyUrl;
}

function persistUrl(value: string): void {
  try {
    sessionStorage.setItem(URL_KEY, value);
  } catch {
    /* best-effort — the in-memory state still drives this session */
  }
}

export function ComfyStation({ spec }: { spec: StationSpec }) {
  // `url` is the editable text field (persisted every keystroke). `loaded` is the
  // value actually committed to the iframe — updated on submit and initialized to
  // the persisted/default URL so a valid saved endpoint re-embeds on reload.
  const [url, setUrl] = useState<string>(() => readPersistedUrl());
  const [loaded, setLoaded] = useState<string>(() => readPersistedUrl());

  // Diagnose the CURRENT text field (immediate feedback while typing).
  const inputSanitized = sanitizeEmbedUrl(url);
  const schemeRejected = url.trim() !== "" && inputSanitized == null;

  // The iframe only ever sees a sanitized value; null → render the empty state.
  const embedSrc = sanitizeEmbedUrl(loaded);
  const risk = embedRisk(loaded);
  // Best target for the always-available "Open in new tab" affordance: the loaded
  // endpoint if valid, else whatever the user has currently typed (if valid).
  const openTarget = embedSrc ?? inputSanitized;
  const pageIsHttps =
    typeof window !== "undefined" && window.location.protocol === "https:";

  function onChange(next: string) {
    setUrl(next);
    persistUrl(next);
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoaded(url); // commit; embedSrc re-derives through sanitizeEmbedUrl()
  }

  return (
    <section className="station-wide vi-comfy" aria-label="ComfyUI viewer">
      <form className="vi-comfy-bar" onSubmit={onSubmit}>
        <label className="vi-comfy-field">
          <span className="vi-comfy-label">ComfyUI endpoint URL</span>
          <input
            className="vi-knob-input vi-comfy-input"
            type="url"
            inputMode="url"
            spellCheck={false}
            autoComplete="off"
            placeholder="https://comfy.hugpy.ai"
            value={url}
            onChange={(e) => onChange(e.target.value)}
            aria-invalid={schemeRejected || undefined}
          />
        </label>
        <div className="vi-comfy-bar-actions">
          <button type="submit" className="vi-btn vi-btn-accent" disabled={inputSanitized == null}>
            Load
          </button>
          {/* Always available: lets the user reach ComfyUI directly even when the
              frame is blocked (mixed content) or refuses embedding (X-Frame-Options). */}
          {openTarget ? (
            <a className="vi-btn vi-btn-ghost" href={openTarget} target="_blank" rel="noreferrer">
              Open in new tab ↗
            </a>
          ) : (
            <span className="vi-btn vi-btn-ghost vi-comfy-open-disabled" aria-disabled="true">
              Open in new tab ↗
            </span>
          )}
        </div>
      </form>

      {/* Scheme rejected (javascript:/data:/blob:/file:/relative/…): explicit and
          loud, since this is the security gate — the value never reaches the frame. */}
      {schemeRejected && (
        <p className="vi-error" role="alert">
          That URL isn’t an http:// or https:// address, so it won’t be embedded. Enter
          a full HTTPS URL (e.g. <code>https://comfy.hugpy.ai</code>).
        </p>
      )}

      {/* Mixed-content / loopback warning: an HTTPS page can’t frame these. */}
      {risk && (
        <p className="vi-comfy-warn" role="alert">
          {risk === "http" ? (
            <>
              This is an <code>http://</code> endpoint.{" "}
              {pageIsHttps
                ? "This page is HTTPS, so the browser blocks the frame as mixed content — it stays blank with no error."
                : "On the deployed HTTPS site (dev.hugpy.ai/video, llm.hugpy.ai/app/video) the browser will block this frame as mixed content."}{" "}
              Use an HTTPS endpoint such as <code>https://comfy.hugpy.ai</code>, or open it
              in a new tab.
            </>
          ) : (
            <>
              This points at a <code>localhost</code> / loopback host, which the browser
              can’t reach from an HTTPS page and won’t serve over TLS. Expose ComfyUI
              through an HTTPS proxy (e.g. <code>https://comfy.hugpy.ai</code>) instead.
            </>
          )}
        </p>
      )}

      {embedSrc ? (
        <div className="vi-comfy-frame">
          <iframe
            className="vi-comfy-iframe"
            src={embedSrc}
            title="ComfyUI"
            // Permissive on purpose: ComfyUI is a trusted internal tool that needs
            // scripts, same-origin, forms, downloads and popups to function. Shared
            // with ComfyPanel so the two frames can never diverge on sandboxing.
            sandbox={COMFY_SANDBOX}
            allow="clipboard-read; clipboard-write; fullscreen"
          />
        </div>
      ) : (
        <div className="vi-comfy-empty">
          <p className="vi-comfy-empty-title">No endpoint loaded</p>
          <p className="vi-comfy-empty-sub">
            Enter your ComfyUI endpoint above and press <strong>Load</strong> to embed it.
            It must be an HTTPS URL (e.g. <code>https://comfy.hugpy.ai</code>).
          </p>
        </div>
      )}

      {embedSrc && (
        <p className="vi-comfy-hint">
          If the frame stays blank, the endpoint is likely refusing to be embedded
          (<code>X-Frame-Options</code> / missing <code>frame-ancestors</code>). Open it in
          a new tab instead — the “Open in new tab” link above works regardless.
        </p>
      )}
    </section>
  );
}
