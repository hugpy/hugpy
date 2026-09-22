// Shared GOAL-INPUT composer template (Round 7 addendum 2b/2c).
//
// Extracts the Movie mode's goal text-input idiom into ONE reusable, presentational
// component so the tiers share a single goal-input look: a .vi-knob-styled labeled text
// field (Movie's knob rhythm) with an OPTIONAL inline, chat-composer-style ATTACH TOOLBAR
// ("＋ image from library" / "＋ video") and removable attached-media chips.
//
// Presentational by design: it OWNS no picker or MediaRef logic. Each host wires the
// attach buttons to its OWN existing library pickers / MediaRef flows via callbacks
// (`attachments[].onClick`) and renders whatever it already tracks as `attached` chips.
// That keeps adoption per-mode and non-destabilizing — a mode adopts the idiom without
// surrendering its editor's state model.
//
// Adoption status (see the Round 7 report): the STUDIO surface adopts this for its
// prompt/negative. The image/scene (ordered-parts composer) and movie (goal/timeline)
// editors already carry richer, working attach models, so they MIRROR the visual rhythm
// and their full migration onto this component is flagged as a separate slice rather than
// forced (per the operator's "mirror + flag what would destabilize" guardrail).
import type { ReactNode } from "react";
import { mediaBytesUrl } from "../config";
import type { MediaRef } from "../video/contract";

/** One inline attach button (e.g. "＋ image from library" / "＋ video"). */
export interface GoalComposerAttach {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}

/** One attached-media chip (an already-picked MediaRef the host tracks). */
export interface GoalComposerAttached {
  ref: MediaRef;
  onRemove?: () => void;
  label?: string;
}

export interface GoalComposerProps {
  id?: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  rows?: number;
  disabled?: boolean;
  hint?: ReactNode;
  /** k93: keep the <label> for a11y but render it visually hidden (.vi-sr-only). */
  hideLabel?: boolean;
  /**
   * Fired when the textarea loses focus. Optional and inert unless wired: the studio
   * movie composer uses it to classify what was typed (scene vs direction) ON BLUR —
   * never per keystroke — so it can pre-arm the right assist button.
   */
  onBlur?: () => void;
  /** Inline chat-composer attach buttons. Omit/empty = a plain composer (no toolbar). */
  attachments?: GoalComposerAttach[];
  /** Attached media shown as removable thumbnail chips above the toolbar. */
  attached?: GoalComposerAttached[];
}

export function GoalComposer({
  id,
  label,
  value,
  onChange,
  placeholder,
  rows = 2,
  disabled = false,
  hint,
  hideLabel = false,
  onBlur,
  attachments,
  attached,
}: GoalComposerProps) {
  const labelClass = hideLabel ? "vi-sr-only" : undefined;
  const hasToolbar = !!attachments && attachments.length > 0;
  const hasChips = !!attached && attached.length > 0;
  return (
    <div className="vi-knob vi-goal-composer">
      {id ? (
        <label htmlFor={id} className={labelClass}>{label}</label>
      ) : (
        <label className={labelClass}>{label}</label>
      )}
      <div className="vi-goal-composer-field">
        <textarea
          id={id}
          className="vi-knob-input vi-goal-composer-input"
          rows={rows}
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          onBlur={onBlur}
        />
        {hasChips && (
          <div className="vi-goal-composer-chips">
            {attached!.map((a, i) => (
              <span key={`${a.ref.uri}-${i}`} className="vi-goal-composer-chip">
                {a.ref.kind === "image" ? (
                  <img
                    src={mediaBytesUrl(a.ref.uri)}
                    alt={a.label ?? "attachment"}
                    className="vi-goal-composer-chip-thumb"
                  />
                ) : (
                  <span className="vi-goal-composer-chip-kind">{a.ref.kind}</span>
                )}
                <span className="vi-goal-composer-chip-label">
                  {a.label ?? a.ref.asset_id.slice(0, 8)}
                </span>
                {a.onRemove && (
                  <button
                    type="button"
                    className="vi-goal-composer-chip-x"
                    onClick={a.onRemove}
                    aria-label="Remove attachment"
                    title="Remove"
                  >
                    ✕
                  </button>
                )}
              </span>
            ))}
          </div>
        )}
        {hasToolbar && (
          <div className="vi-goal-composer-toolbar">
            {attachments!.map((a, i) => (
              <button
                key={`${a.label}-${i}`}
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost vi-goal-composer-attach"
                onClick={a.onClick}
                disabled={a.disabled}
                title={a.title}
              >
                {a.label}
              </button>
            ))}
          </div>
        )}
      </div>
      {hint ? <span className="vi-knob-hint">{hint}</span> : null}
    </div>
  );
}
