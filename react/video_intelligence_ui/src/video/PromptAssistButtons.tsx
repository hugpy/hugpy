// The PROMPT-ASSIST button cluster — the presentational half of usePromptAssist.
// Renders the same .vi-prompt-assist row GenerateStation shipped (an "prompt assist"
// lead + Enhance + Generate + a dismissible inline error), so every composer that
// adopts the shared hook gets an identical look. It owns NO state: the host wires
// the two clicks (Enhance passes its draft; Generate seeds from it) and passes the
// hook's busy/error values straight through.
import type { AssistBusyMode, AssistModel } from "./usePromptAssist";

export interface PromptAssistButtonsProps {
  /**
   * The active in-flight mode (from usePromptAssist), or null when idle. Typed as the
   * WIDER busy union so a host that also runs spread/negative calls can hand its
   * `assistBusy` straight through and have both buttons correctly disable — the two
   * labels below still key off "detail"/"generate" only.
   */
  busy: AssistBusyMode | null;
  /** The dismissible inline error (from usePromptAssist), or null. */
  error: string | null;
  /** Clear the inline error. */
  onDismissError: () => void;
  /** Fire Enhance ("detail") — enriches the current draft. Disabled when `canEnhance` is false. */
  onEnhance: () => void;
  /** Fire Generate ("generate") — writes a full prompt from scratch (seeded by any current text). */
  onGenerate: () => void;
  /** Enhance needs a non-empty draft; the host computes this from its own prompt text. */
  canEnhance: boolean;
  /**
   * The text generators the fleet can actually run (from usePromptAssist). OPTIONAL and
   * defaulting to empty: a host that hasn't adopted the picker renders exactly the old
   * cluster, so adoption stays a per-composer flip rather than a breaking change.
   */
  models?: AssistModel[];
  /** The chosen generator, or null for the fleet default. */
  model?: string | null;
  /** Choose a generator (null = "use the fleet default"). */
  onModelChange?: (model: string | null) => void;
}

export function PromptAssistButtons({
  busy,
  error,
  onDismissError,
  onEnhance,
  onGenerate,
  canEnhance,
  models = [],
  model = null,
  onModelChange,
}: PromptAssistButtonsProps) {
  // Only render the picker once discovery has returned something AND the host wired a
  // handler. An empty list means the fleet offered nothing (or discovery failed) — in
  // that case assist still works on the backend default, so showing an empty <select>
  // would advertise a choice that isn't there.
  const showPicker = onModelChange != null && models.length > 0;
  return (
    <div className="vi-prompt-assist">
      <span className="vi-prompt-assist-lead" aria-hidden="true">
        prompt assist
      </span>
      {showPicker && (
        <select
          className="vi-select vi-select-sm"
          value={model ?? ""}
          disabled={busy != null}
          onChange={(e) => onModelChange?.(e.target.value || null)}
          aria-label="Text generator used to enhance or generate prompts"
          title={
            "Which text generator writes the prompt. Thinking is suppressed for all of " +
            "them, so a reasoning model can't fill the box with its monologue."
          }
        >
          <option value="">fleet default</option>
          {models.map((m) => (
            // Not decoration (canonical STATE-MODEL.md #4): a non-ready model pays a
            // load on first use. `hot` = already on a worker drive (a t_load, seconds);
            // `cold` = central only (a download THEN a load). "cold" is reserved for
            // not-on-hot-drive, so it never mislabels a hot or loaded model.
            <option key={m.model} value={m.model}>
              {m.model}
              {m.state === "serving" || m.state === "loaded" ? ""
                : m.state === "hot" ? " · hot (loads)" : " · cold (downloads)"}
            </option>
          ))}
        </select>
      )}
      <button
        type="button"
        className="vi-btn vi-btn-sm vi-btn-ghost"
        onClick={onEnhance}
        disabled={!canEnhance || busy != null}
        title="Enrich your current prompt with more descriptive detail"
      >
        {busy === "detail" ? "✨ Enhancing…" : "✨ Enhance"}
      </button>
      <button
        type="button"
        className="vi-btn vi-btn-sm vi-btn-ghost"
        onClick={onGenerate}
        disabled={busy != null}
        title="Write a full prompt from scratch (uses any current text as a theme)"
      >
        {busy === "generate" ? "✨ Generating…" : "✨ Generate"}
      </button>
      {error && (
        <span className="vi-prompt-assist-error" role="alert">
          {error}
          <button
            type="button"
            className="vi-prompt-assist-dismiss"
            onClick={onDismissError}
            aria-label="Dismiss assist error"
            title="Dismiss"
          >
            ✕
          </button>
        </span>
      )}
    </div>
  );
}
