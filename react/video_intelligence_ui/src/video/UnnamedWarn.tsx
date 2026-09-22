// UNNAMED-GENERATION WARNING (operator 2026-08-13): naming applies to EVERY
// generate section, and the warning is an EXPLICIT per-section OPTION,
// DEFAULT OFF, remembered per user (localStorage) — a 2-second image render
// must never suffer a 30-second warning unless its operator asked for one.
// ONE hook + toggle + panel for every section (house one-component rule);
// Cinema, Clip and the Generate station all mount these.
import { useEffect, useRef, useState, type ReactNode } from "react";

const PREF_PREFIX = "vi.warnUnnamed.v1";

/** localStorage-backed per-section preference; DEFAULT OFF. */
export function useWarnUnnamedPref(section: string): [boolean, (v: boolean) => void] {
  const key = `${PREF_PREFIX}:${section}`;
  const [on, setOn] = useState<boolean>(() => {
    try {
      return localStorage.getItem(key) === "1";
    } catch {
      return false;
    }
  });
  const set = (v: boolean) => {
    setOn(v);
    try {
      localStorage.setItem(key, v ? "1" : "0");
    } catch {
      /* best-effort — the in-memory state still applies this session */
    }
  };
  return [on, set];
}

/** The explicit option — one checkbox, same copy in every section. */
export function WarnUnnamedToggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label
      className="vi-knob-hint"
      style={{ display: "inline-flex", gap: "0.35rem", alignItems: "center" }}
      title="When on, generating without a name pauses with an inline name box for 30 seconds, then continues unnamed. Off = never interrupts. Remembered per section."
    >
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      warn when unnamed
    </label>
  );
}

/** First clause of a prompt as a derived name (≤7 words / 48 chars). */
export function deriveNameFrom(prompt: string): string {
  return prompt
    .split(/[.,;\n]/)[0]
    .trim()
    .split(/\s+/)
    .slice(0, 7)
    .join(" ")
    .slice(0, 48);
}

/**
 * The warning panel: inline name box + "Name it" + skip-with-countdown.
 * MOUNTING arms the expiry (default 30s); unmounting disarms it; onProceed
 * fires exactly once (undefined name = continue unnamed).
 */
export function UnnamedWarnPanel({
  noun,
  onProceed,
  onDismiss,
  seconds = 30,
}: {
  noun: string;
  onProceed: (name?: string) => void;
  onDismiss?: () => void;
  seconds?: number;
}) {
  const [draft, setDraft] = useState("");
  const [left, setLeft] = useState(seconds);
  const draftRef = useRef("");
  draftRef.current = draft;
  const firedRef = useRef(false);
  const proceedRef = useRef(onProceed);
  proceedRef.current = onProceed;
  const fire = (name?: string) => {
    if (firedRef.current) return;
    firedRef.current = true;
    proceedRef.current(name);
  };
  const fireRef = useRef(fire);
  fireRef.current = fire;
  useEffect(() => {
    const id = window.setInterval(() => {
      setLeft((l) => {
        if (l <= 1) {
          window.clearInterval(id);
          fireRef.current(draftRef.current.trim() || undefined);
          return 0;
        }
        return l - 1;
      });
    }, 1000);
    return () => window.clearInterval(id);
  }, []);
  return (
    <div
      className="vi-studio-required-callout"
      role="alert"
      style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap", marginBottom: "0.6rem" }}
    >
      <span>
        This {noun} has no name — unnamed work lists as “untitled” beside a bare id
        and is hard to tell apart later.
      </span>
      <input
        type="text"
        className="vi-knob-input"
        style={{ flex: "1 1 12rem", minWidth: "10rem" }}
        placeholder="name it here — or wait and it continues unnamed"
        value={draft}
        autoFocus
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && draft.trim()) fire(draft.trim());
        }}
      />
      <button type="button" className="vi-btn vi-btn-sm" disabled={!draft.trim()} onClick={() => fire(draft.trim())}>
        Name it &amp; generate
      </button>
      <button
        type="button"
        className="vi-btn vi-btn-sm vi-btn-ghost"
        title="Continues automatically when the timer runs out."
        onClick={() => fire(undefined)}
      >
        Generate without a name ({left}s)
      </button>
      {onDismiss && (
        <button type="button" className="vi-btn vi-btn-sm vi-btn-ghost" title="Cancel this submit." onClick={onDismiss}>
          ✕
        </button>
      )}
    </div>
  );
}
