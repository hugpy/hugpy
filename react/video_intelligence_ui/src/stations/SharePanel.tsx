// k9 — the SHARE affordance. An operator mints a video-scoped share link and
// hands it to an outside party, who can then drive the /video features WITHOUT a
// console login (the link carries a `?share=<key>` credential the SPA lifts on
// load — see src/share.ts). The minted key opens the /video surface ONLY; it can
// never reach the console/operator routes, and — crucially — never the mint route
// itself, so there is no key-minting-by-key.
//
// VISIBILITY IS AUTH-DECIDED, NOT GUESSED. This button renders only for a
// CONSOLE-AUTHENTICATED session. It probes the operator-gated list endpoint
// (GET /api/keys/video-share) once on mount: 200 ⇒ operator ⇒ show; 401 ⇒ a share
// party or anon ⇒ render nothing. So a share-principal session (which CAN pass the
// /video gate) never sees a working Share button — the probe 401s exactly as the
// mint POST would.
import { useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { request, okValue } from "../transport/client";
import { hugpyConfig, keysVideoShareRevokeUrl } from "../config";

interface ShareKey {
  id: string;
  label: string;
  prefix: string;
  created_at: number;
  expires_at: number | null;
  expired?: boolean;
  revoked?: boolean;
  last_used?: number | null;
}
interface MintResult extends ShareKey {
  key: string;
  url: string;
}

// Expiry choices for the mint form (days). 0 = a non-expiring link.
const TTL_OPTIONS: { label: string; days: number }[] = [
  { label: "7 days", days: 7 },
  { label: "30 days", days: 30 },
  { label: "90 days", days: 90 },
  { label: "No expiry", days: 0 },
];

function fmtExpiry(k: ShareKey): string {
  if (k.expired) return "expired";
  if (!k.expires_at) return "no expiry";
  const days = Math.max(0, Math.round((k.expires_at * 1000 - Date.now()) / 86_400_000));
  return days <= 0 ? "expiring" : `${days}d left`;
}

/** The full share link for a minted key — rebuilt from THIS origin so it's always
 *  correct regardless of how the backend saw its forwarded host. */
function linkFor(key: string): string {
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  return `${origin}/video/?share=${encodeURIComponent(key)}`;
}

export function SharePanel() {
  // null = probing (render nothing), false = not an operator (render nothing),
  // true = operator (render the button).
  const [authed, setAuthed] = useState<boolean | null>(null);
  const [open, setOpen] = useState(false);
  const [keys, setKeys] = useState<ShareKey[]>([]);
  const [label, setLabel] = useState("");
  const [ttl, setTtl] = useState(30);
  const [minting, setMinting] = useState(false);
  const [minted, setMinted] = useState<MintResult | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const res = await request<{ keys: ShareKey[] }>(hugpyConfig.keysVideoShareUrl, {
      method: "GET",
      meta: { specKey: "share", operation: "share.list" },
    });
    if (res.ok) {
      setAuthed(true);
      setKeys(okValue(res).keys ?? []);
    } else {
      // 401/403 ⇒ not an operator: keep the affordance hidden entirely.
      setAuthed(false);
    }
  }, []);

  // Probe once on mount to decide visibility (and seed the existing-links list).
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const mint = useCallback(async () => {
    setMinting(true);
    setError(null);
    setMinted(null);
    setCopied(false);
    const res = await request<MintResult>(hugpyConfig.keysVideoShareUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: label.trim(), ttl_days: ttl }),
      meta: { specKey: "share", operation: "share.mint" },
    });
    setMinting(false);
    if (!res.ok) {
      setError("Could not create a share link — are you still signed in to the console?");
      return;
    }
    const m = okValue(res);
    setMinted({ ...m, url: linkFor(m.key) });
    setLabel("");
    void refresh();
  }, [label, ttl, refresh]);

  const revoke = useCallback(
    async (id: string) => {
      const res = await request<unknown>(keysVideoShareRevokeUrl(id), {
        method: "DELETE",
        meta: { specKey: "share", operation: "share.revoke" },
      });
      if (res.ok) void refresh();
    },
    [refresh],
  );

  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }, []);

  const close = useCallback(() => {
    setOpen(false);
    setMinted(null);
    setError(null);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, close]);

  const activeKeys = useMemo(() => keys.filter((k) => !k.revoked), [keys]);

  // Hidden entirely until the probe confirms an operator session.
  if (authed !== true) return null;

  return (
    <>
      <button
        type="button"
        className="vi-share-toggle vi-btn vi-btn-sm"
        aria-expanded={open}
        onClick={() => setOpen(true)}
        title="Create a share link that lets someone use the video features without a console login"
      >
        ⤴ Share
      </button>

      {open &&
        createPortal(
          <>
            <button
              type="button"
              className="vi-share-scrim"
              aria-label="Close share links"
              onClick={close}
            />
            <div
              className="vi-share-panel"
              role="dialog"
              aria-modal="true"
              aria-label="Share links"
            >
              <div className="vi-share-bar">
                <span className="vi-share-title">Share the video features</span>
                <button
                  type="button"
                  className="vi-btn vi-btn-sm"
                  onClick={close}
                  aria-label="Close"
                  title="Close (Esc)"
                >
                  ✕
                </button>
              </div>

              <div className="vi-share-body">
                <p className="vi-share-hint">
                  A share link lets the person you send it to use the video features
                  here — no login. It works only for these video tools, nothing else
                  in the console. Revoke it any time.
                </p>

                <div className="vi-share-form">
                  <label className="vi-share-field">
                    <span>Label (optional)</span>
                    <input
                      type="text"
                      value={label}
                      placeholder="e.g. Alex — review cut"
                      onChange={(e) => setLabel(e.target.value)}
                    />
                  </label>
                  <label className="vi-share-field">
                    <span>Expires</span>
                    <select
                      value={ttl}
                      onChange={(e) => setTtl(Number(e.target.value))}
                    >
                      {TTL_OPTIONS.map((o) => (
                        <option key={o.days} value={o.days}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-accent"
                    disabled={minting}
                    onClick={() => void mint()}
                  >
                    {minting ? "Creating…" : "Create share link"}
                  </button>
                </div>

                {error && (
                  <p className="vi-share-error" role="alert">
                    {error}
                  </p>
                )}

                {minted && (
                  <div className="vi-share-minted">
                    <p className="vi-share-minted-note">
                      Copy this now — it won't be shown again.
                    </p>
                    <div className="vi-share-minted-row">
                      <input
                        className="vi-share-link"
                        type="text"
                        readOnly
                        value={minted.url}
                        onFocus={(e) => e.currentTarget.select()}
                      />
                      <button
                        type="button"
                        className="vi-btn vi-btn-sm vi-btn-accent"
                        onClick={() => void copy(minted.url)}
                      >
                        {copied ? "Copied ✓" : "Copy"}
                      </button>
                    </div>
                  </div>
                )}

                {activeKeys.length > 0 && (
                  <div className="vi-share-list">
                    <h3 className="vi-share-list-title">Active links</h3>
                    <ul>
                      {activeKeys.map((k) => (
                        <li key={k.id} className="vi-share-list-item">
                          <span className="vi-share-item-label">
                            {k.label || "unnamed"}
                          </span>
                          <code className="vi-share-item-prefix">{k.prefix}…</code>
                          <span
                            className={`vi-share-item-exp${
                              k.expired ? " vi-share-item-exp-dead" : ""
                            }`}
                          >
                            {fmtExpiry(k)}
                          </span>
                          <button
                            type="button"
                            className="vi-btn vi-btn-sm"
                            onClick={() => void revoke(k.id)}
                            title="Revoke this link — it stops working immediately"
                          >
                            Revoke
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </>,
          document.body,
        )}
    </>
  );
}
