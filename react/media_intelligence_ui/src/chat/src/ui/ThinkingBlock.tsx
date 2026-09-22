/*
 * ThinkingBlock.tsx — collapsible reasoning stream for <think> models.
 *
 * While the model is still inside its think block the panel is open and the
 * reasoning streams live (auto-following the bottom). The moment the answer
 * begins (</think> arrives) it collapses to a one-line "Thoughts" toggle.
 * A manual click pins the user's choice so auto-collapse never fights them.
 */

import { useEffect, useRef, useState } from "react";
import { Icon } from "./Icons";

interface ThinkingBlockProps {
  reasoning: string;
  /** True while the think block is still open (reasoning is streaming). */
  active: boolean;
}

export default function ThinkingBlock({
  reasoning,
  active,
}: ThinkingBlockProps): JSX.Element | null {
  // null = auto: open while reasoning streams, collapsed once the answer starts.
  const [pinned, setPinned] = useState<boolean | null>(null);
  const open = pinned ?? active;
  const bodyRef = useRef<HTMLDivElement>(null);

  // Follow the live stream — keep the newest reasoning in view.
  useEffect(() => {
    if (open && active && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [reasoning, open, active]);

  if (!reasoning && !active) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <button
        type="button"
        onClick={() => setPinned(!open)}
        aria-expanded={open}
        className="
          flex w-fit items-center gap-1.5 rounded-full px-2 py-1 -ml-2
          text-[13px] text-token-text-tertiary
          hover:bg-token-surface-hover hover:text-token-text-secondary
        "
      >
        <span
          className={`transition-transform duration-150 ${open ? "" : "-rotate-90"}`}
          aria-hidden
        >
          <Icon name="chevron-down" width={14} height={14} />
        </span>
        {active ? (
          <>
            <span>Thinking</span>
            <span className="typing-dots" aria-hidden>
              <span />
              <span />
              <span />
            </span>
          </>
        ) : (
          <span>Thoughts</span>
        )}
      </button>

      {open && (
        <div
          ref={bodyRef}
          className="
            max-h-56 overflow-y-auto rounded-xl px-3.5 py-2.5
            border-l-2 border-token-border-light
            text-[13px] leading-6 text-token-text-tertiary
            whitespace-pre-wrap break-words
          "
          style={{ background: "var(--bg-tertiary)" }}
        >
          {reasoning || "…"}
        </div>
      )}
    </div>
  );
}
