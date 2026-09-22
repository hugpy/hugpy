// ComfyPanel — ComfyUI in the shared expand-into-the-screen drawer, reachable
// from EVERY station without spending a slot in the studio tab strip (operator
// ask 2026-08-04: "have comfy UI as an overlay the same as console is in the
// left column").
//
// Since 2026-08-06 this file contributes ONE TAB to the single unified drawer
// (drawerShell.tsx → WorkbenchDrawer.tsx) rather than rendering a drawer of its
// own — so "the same as console" is now literally the same window. (Until
// 2026-08-04 that shell was copied into this file verbatim; then it was shared;
// now there is one instance of it.) Only the endpoint bar, the message strips
// and the frame below are Comfy-specific.
//
// ── WHERE IT DELIBERATELY DIFFERS FROM ConsolePanel (read before editing) ────
// ConsolePanel frames FIRST-PARTY, SAME-ORIGIN, FIXED-href content ("/console"),
// and is deliberately UNSANDBOXED because a sandbox without `allow-modals` would
// silently break the console's confirm()/alert() flows. Its header comment
// explicitly contrasts itself with "ComfyStation's user-supplied URL".
//
// This panel is that contrasting case. The ComfyUI endpoint is USER-EDITABLE and
// PERSISTED, i.e. attacker-influenceable and foreign-origin. So:
//   * every value passes sanitizeEmbedUrl() (http/https only) before it can ever
//     reach iframe src — typed input AND the value rehydrated from storage;
//   * the frame KEEPS its sandbox (COMFY_SANDBOX).
// Do NOT "simplify" this by copying ConsolePanel's unsandboxed iframe: that
// would hand a user-supplied foreign origin full same-origin-ish reign over this
// page. The shared gate lives in comfyEmbed.ts precisely so the two cannot drift.
// Sharing ONE drawer window with the console tab does not blur this: the two are
// distinct <iframe> elements rendered side by side (one hidden), each with its
// own sandbox attribute — never a single frame whose src is swapped.
//
// ── WHY THE FRAME MAY STILL BE BLANK (not this component's bug) ─────────────
// The video UI is HTTPS, so an http:// or loopback endpoint is blocked as mixed
// content with NO error, and any endpoint must also permit framing from this
// origin (`frame-ancestors`, no `X-Frame-Options: DENY`). Both cases are warned
// about inline, and "Open in new tab" always works regardless.
import { useState, type FormEvent } from "react";
import { hugpyConfig } from "../config";
import type { DrawerTab } from "./drawerShell";
import {
  sanitizeEmbedUrl,
  embedRisk,
  COMFY_SANDBOX,
  COMFY_URL_KEY,
} from "./comfyEmbed";

// The endpoint URL is session-persisted under the SAME key the station used, so
// an operator who already configured ComfyUI keeps their endpoint.
function readPersistedUrl(): string {
  try {
    const stored = sessionStorage.getItem(COMFY_URL_KEY);
    if (stored != null) return stored;
  } catch {
    /* storage unavailable — fall back to config default */
  }
  return hugpyConfig.comfyUrl;
}

function persistUrl(value: string): void {
  try {
    sessionStorage.setItem(COMFY_URL_KEY, value);
  } catch {
    /* best-effort — the in-memory state still drives this session */
  }
}

/**
 * ComfyUI's tab in the unified drawer.
 *
 * A HOOK, not a plain descriptor, because this tab owns state (the endpoint
 * field and the committed value). Hoisting that state to the always-mounted
 * drawer host — rather than to the frame's own body — is what lets a half-typed
 * endpoint survive a switch to the Console tab and back.
 */
export function useComfyDrawerTab(): DrawerTab<"comfy"> {
  const [url, setUrl] = useState<string>(() => readPersistedUrl());
  const [loaded, setLoaded] = useState<string>(() => readPersistedUrl());

  // Diagnose the CURRENT text field (immediate feedback while typing).
  const inputSanitized = sanitizeEmbedUrl(url);
  const schemeRejected = url.trim() !== "" && inputSanitized == null;
  // The iframe only ever sees a sanitized value; null → the empty state.
  const embedSrc = sanitizeEmbedUrl(loaded);
  const risk = embedRisk(loaded);
  const openTarget = embedSrc ?? inputSanitized;

  function onChangeUrl(next: string) {
    setUrl(next);
    persistUrl(next);
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoaded(url); // commit; embedSrc re-derives through sanitizeEmbedUrl()
  }

  return {
    id: "comfy",
    label: "ComfyUI",
    hint: "Open ComfyUI alongside this station — no tab switch",
    toggleClassName: "vi-comfy-toggle",
    toggleLabel: "▤ ComfyUI",
    ariaLabel: "ComfyUI",
    panelClassName: "vi-comfy-panel",
    // Stays in the DOM while another tab shows. This one matters most: a
    // ComfyUI workspace is a long-lived, unsaved graph — reloading the frame
    // every time the operator glanced at Active would lose work.
    keepMounted: true,
    headerExtra: (
      /* The endpoint is configurable here because it always was — the station
         owned this field, and dropping it would strand anyone whose ComfyUI is
         not at the configured default. It rides in the shared header, so it only
         appears while THIS tab is selected. */
      <form className="vi-comfy-panel-url" onSubmit={onSubmit}>
        <label className="vi-comfy-panel-url-label" htmlFor="vi-comfy-panel-url">
          Endpoint
        </label>
        <input
          id="vi-comfy-panel-url"
          className="vi-knob-input vi-comfy-input"
          type="url"
          inputMode="url"
          spellCheck={false}
          autoComplete="off"
          placeholder="https://comfy.hugpy.ai"
          value={url}
          onChange={(e) => onChangeUrl(e.target.value)}
          aria-invalid={schemeRejected || undefined}
        />
        <button
          type="submit"
          className="vi-btn vi-btn-sm vi-btn-accent"
          disabled={inputSanitized == null}
        >
          Load
        </button>
      </form>
    ),
    headerActions: openTarget ? (
      <a
        className="vi-btn vi-btn-sm vi-btn-ghost"
        href={openTarget}
        target="_blank"
        rel="noreferrer"
        title="Open ComfyUI in its own tab"
      >
        Open in new tab ↗
      </a>
    ) : (
      <span
        className="vi-btn vi-btn-sm vi-btn-ghost vi-comfy-open-disabled"
        aria-disabled="true"
      >
        Open in new tab ↗
      </span>
    ),
    body: (
      <>
        {/* Scheme rejected: loud, since this is the security gate — the value
            never reaches the frame. */}
        {schemeRejected && (
          <p className="vi-error vi-comfy-panel-msg" role="alert">
            That URL isn’t an http:// or https:// address, so it won’t be
            embedded. Enter a full HTTPS URL (e.g.{" "}
            <code>https://comfy.hugpy.ai</code>).
          </p>
        )}

        {risk && (
          <p className="vi-comfy-warn vi-comfy-panel-msg" role="alert">
            {risk === "http" ? (
              <>
                This is an <code>http://</code> endpoint. This page is HTTPS, so
                the browser blocks the frame as mixed content — it stays blank
                with no error. Use an HTTPS endpoint such as{" "}
                <code>https://comfy.hugpy.ai</code>, or open it in a new tab.
              </>
            ) : (
              <>
                This points at a <code>localhost</code> / loopback host, which the
                browser can’t reach from an HTTPS page. Expose ComfyUI through an
                HTTPS proxy (e.g. <code>https://comfy.hugpy.ai</code>) instead.
              </>
            )}
          </p>
        )}

        {embedSrc ? (
          <iframe
            className="vi-console-iframe"
            src={embedSrc}
            title="ComfyUI"
            // SANDBOXED on purpose — user-supplied, foreign origin. See the
            // header note; do not drop this to match ConsolePanel.
            sandbox={COMFY_SANDBOX}
            allow="clipboard-read; clipboard-write; fullscreen"
          />
        ) : (
          <div className="vi-comfy-empty vi-comfy-panel-empty">
            <p className="vi-comfy-empty-title">No endpoint loaded</p>
            <p className="vi-comfy-empty-sub">
              Enter your ComfyUI endpoint above and press <strong>Load</strong> to
              embed it. It must be an HTTPS URL (e.g.{" "}
              <code>https://comfy.hugpy.ai</code>).
            </p>
          </div>
        )}
      </>
    ),
  };
}
