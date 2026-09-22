// Compact "ⓘ" affordance for a knob's verbose explanation.
//
// The compacted knobs (width/height/steps/guidance/seed in the Generate rail)
// used to carry their explanation as permanently-visible copy under the input
// (`.vi-knob-hint`) — that copy IS the dead space the row-pairing reclaims. This
// component moves it behind a small info affordance instead of dropping it.
//
// ONE pattern covers both desktop and touch (which has no hover), via two
// INDEPENDENT reveal paths that don't fight each other:
//   - desktop hover / keyboard focus: pure CSS (`:hover` / `:focus-within` in
//     app.css) — no React state involved, so it can't be knocked closed by
//     anything else.
//   - click/tap: toggles `.vi-knob-info-open` via React state, driven ONLY by
//     onClick. This is the mobile-equivalent — touch has no hover, so tapping
//     the ⓘ is how the explanation gets discovered there; tap again to close.
//     (An earlier version also wired onMouseEnter/onMouseLeave into the same
//     state and that raced with the synthetic hover-compat events some touch
//     browsers fire around a tap, closing the popover right after it opened —
//     keeping the two paths independent avoids that.)
//
// Closing via tap/Escape also blurs the button. Without that, a tap leaves the
// button focused, `:focus-within` stays true, and the popover would keep
// showing even after `open` (and aria-expanded) flip back to false.
import { useRef, useState, type KeyboardEvent } from "react";

export function KnobInfo({ label, text }: { label: string; text: string }) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);

  function close() {
    setOpen(false);
    btnRef.current?.blur();
  }

  function onKeyDown(e: KeyboardEvent<HTMLButtonElement>) {
    if (e.key === "Escape") close();
  }

  return (
    <span className={`vi-knob-info${open ? " vi-knob-info-open" : ""}`}>
      <button
        ref={btnRef}
        type="button"
        className="vi-knob-info-btn"
        aria-label={`About ${label}`}
        aria-expanded={open}
        onClick={(e) => {
          // Don't let the click bubble to a document-level handler elsewhere
          // in the tree — there isn't one today, but this keeps the toggle
          // self-contained if one is added later.
          e.stopPropagation();
          if (open) close();
          else setOpen(true);
        }}
        onKeyDown={onKeyDown}
      >
        ⓘ
      </button>
      <span className="vi-knob-info-pop" role="tooltip" aria-hidden={!open}>
        {text}
      </span>
    </span>
  );
}
