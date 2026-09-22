// SHARED PROMPT-CARD SYSTEM (task k88 — Studio declutter, part 1 of 2).
//
// ONE uniform structure for every Studio prompt list (Cinema + Movie now; Scene
// parts + Clip in the follow-up task k89):
//
//   toolbar:  [# count][select all][✨ Generate prompts][✨ Generate negatives][standard negatives]
//   row card: [badge #N]                                   [↑][↓][Split][✕]
//             [Prompt][Negative] tabs over ONE editor pane + [standard negative] tick
//             <GoalComposer pane for the active tab>
//             …host body (Prompt Assist row, attach rows)…
//             [settings ▸] expander hosting the row's tab-specific knobs
//
// Everything here is PRESENTATIONAL in the GoalComposer/PromptAssistButtons sense:
// no component owns editor state — values come in and callbacks go out, so each
// tab keeps its own immutable map/filter state model. The ONLY internal state is
// ephemeral VIEW state (which tab is showing, whether the settings expander is
// open — the latter via a native <details>), which is never data the host reads.
//
// Glyph ruling (one set everywhere): ↑ ↓ ✕ + a text "Split" — the Scene/Movie
// set. Cinema's ▲▼ are retired by its migration onto PromptCard.
import { useState, type ReactNode } from "react";
import { GoalComposer } from "../GoalComposer";

/**
 * The STANDARD negative set — the text GenerateStation's negative field opens
 * pre-filled with (operator 2026-07-13: distilled/painterly checkpoints reproduce
 * whole "framed gallery painting" objects from their training prior; suppressing
 * the frame + the usual junk by default makes the happy path a clean image).
 * MOVED here from GenerateStation's module-private DEFAULT_NEGATIVE so the
 * per-row [standard negative] checkbox and the station prefill read the same
 * source; GenerateStation imports it back (no behavior change there).
 */
export const STANDARD_NEGATIVE =
  "frame, picture frame, ornate frame, border, framed painting, gallery wall, watermark, signature, text";

/**
 * The effective negative for one row: the user's own negative, with the standard
 * set APPENDED (comma-joined, whitespace-normalized, term-deduped) when the row's
 * [standard negative] checkbox is on. Called ONLY at the point a request payload
 * is built — the wire schemas are untouched; composition is a client-side fold.
 */
export function composeNegative(negative: string, stdNegative: boolean): string {
  const user = negative.trim();
  if (!stdNegative) return user;
  const seen = new Set<string>();
  const terms: string[] = [];
  for (const raw of `${user},${STANDARD_NEGATIVE}`.split(",")) {
    const term = raw.trim().replace(/\s+/g, " ");
    if (term === "") continue;
    const dedupeKey = term.toLowerCase();
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    terms.push(term);
  }
  return terms.join(", ");
}

/** The standard set as individual lower-cased terms (the dedupe keys composeNegative uses). */
const STANDARD_NEGATIVE_TERMS: readonly string[] = STANDARD_NEGATIVE.split(",")
  .map((t) => t.trim().replace(/\s+/g, " ").toLowerCase())
  .filter((t) => t !== "");

/**
 * True when EVERY standard term is present in `negative` (k93 — the explicit
 * "std." tick reads its state from the visible field, so the checkbox can never
 * claim something the wire won't carry).
 */
export function hasStandardNegative(negative: string): boolean {
  const present = new Set(
    negative
      .split(",")
      .map((t) => t.trim().replace(/\s+/g, " ").toLowerCase())
      .filter((t) => t !== ""),
  );
  return STANDARD_NEGATIVE_TERMS.every((t) => present.has(t));
}

/**
 * Remove exactly the standard terms from `negative`, leaving every other term
 * (order preserved, whitespace normalized). The inverse of composeNegative(x, true).
 */
export function stripStandardNegative(negative: string): string {
  const std = new Set(STANDARD_NEGATIVE_TERMS);
  const kept: string[] = [];
  for (const raw of negative.split(",")) {
    const term = raw.trim().replace(/\s+/g, " ");
    if (term === "") continue;
    if (std.has(term.toLowerCase())) continue;
    kept.push(term);
  }
  return kept.join(", ");
}

/** Explicit std. tick semantics: ticked ⇒ insert the standard set into the field; unticked ⇒ remove it. */
export function applyStandardNegative(negative: string, on: boolean): string {
  return on ? composeNegative(negative, true) : stripStandardNegative(negative);
}

// ── PromptListToolbar ───────────────────────────────────────────────────────

/**
 * The k93 RANGE selector that replaces "Select all (n/m)" on the Scene toolbar:
 * 1-based `[start] - [end]` over the text rows + a select-all tick. Optional —
 * Movie/Cinema omit it and keep the original select-all label.
 */
export interface PromptListToolbarRange {
  /** 1-based inclusive bounds of the CURRENT selection, or null when nothing is selected. */
  start: number | null;
  end: number | null;
  /** The host re-selects rows `[start, end]` (already clamped by the toolbar). */
  onChange: (start: number, end: number) => void;
}

/** The optional "how many rows" count input at the toolbar's left edge. */
export interface PromptListToolbarCount {
  id: string;
  /** Knob label — "Segments" / "Goals". */
  label: string;
  value: number;
  min?: number;
  /** Omit for no upper bound. */
  max?: number;
  onChange: (n: number) => void;
  title?: string;
}

export interface PromptListToolbarProps {
  /** Omit to render the toolbar without the count knob. */
  count?: PromptListToolbarCount;
  /** "segment" / "goal" — used in aria labels only. */
  noun: string;
  selectedCount: number;
  itemCount: number;
  allSelected: boolean;
  onToggleSelectAll: () => void;
  /** Full button text, busy/progress label included — the host owns the wording. */
  generateLabel: string;
  generateTitle?: string;
  onGenerate: () => void;
  generateDisabledTitle?: string;
  negativesLabel: string;
  negativesTitle?: string;
  onNegatives: () => void;
  /** True while ANY assist call is in flight — disables both group actions. */
  assistBusy?: boolean;
  /**
   * Tick the [standard negative] checkbox ON for the selected rows (or every
   * row when nothing is selected) — the host implements that rule.
   */
  onStandardNegatives: () => void;
  standardNegativesTitle?: string;
  disabled?: boolean;

  // ── k93 group layout (all OPTIONAL — Movie/Cinema pass none and render the
  // original single-line toolbar byte-for-byte) ───────────────────────────────
  /** Rendered on its own line ABOVE the selection line (the "prompt assist model" select). */
  modelRow?: ReactNode;
  /** When given, replaces "Select all (n/m)" with `[start] - [end]  [ ] select all`. */
  range?: PromptListToolbarRange;
  /**
   * The `negatives: [on/off]` toggle. When given, the separate "✨ Generate
   * negatives" button is NOT rendered — the group actions write negatives
   * themselves when the toggle is on (the host implements that rule).
   */
  negativesOn?: boolean;
  onNegativesOnChange?: (on: boolean) => void;
  negativesOnTitle?: string;
  /**
   * The explicit `[ ] std.` checkbox (k93): ticked ⇒ the host INSERTS the
   * standard set into every selected row's Negative field; unticked ⇒ removes
   * it. When given, the "standard negatives" button is NOT rendered.
   */
  stdChecked?: boolean;
  onStdChange?: (on: boolean) => void;
  stdTitle?: string;
  /** The `✨ enhance (M selected)` group action; rendered only when `onEnhance` is given. */
  enhanceLabel?: string;
  enhanceTitle?: string;
  onEnhance?: () => void;
  /** Enhance is disabled when this is false (M = 0) — the host counts non-blank rows. */
  canEnhance?: boolean;
  /** Trailing content on the actions line (the `[+ video]` pretext attach). */
  actionsExtra?: ReactNode;
  /** Content rendered under the toolbar lines, full width (the expanded attach panel). */
  below?: ReactNode;
}

/** Draft-tolerant count field: the host state is authoritative, but while the
 * user is typing we show their raw draft — a controlled `value={count.value}`
 * used to swallow any keystroke the host rejected (typing "10" past an old max,
 * or clearing the field to retype, silently snapped back), which read as "the
 * input only takes N digits". Valid integers commit on every keystroke; the
 * draft resets to the host value on blur. */
function CountInput({ count, disabled }: { count: PromptListToolbarCount; disabled?: boolean }) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      id={count.id}
      className="vi-knob-input"
      type="number"
      min={count.min ?? 1}
      max={count.max}
      value={draft ?? count.value}
      disabled={disabled}
      title={count.title}
      onChange={(e) => {
        const raw = e.target.value;
        setDraft(raw);
        const n = Number(raw);
        if (raw.trim() !== "" && Number.isInteger(n) && n >= (count.min ?? 1)) count.onChange(n);
      }}
      onBlur={() => setDraft(null)}
    />
  );
}

export function PromptListToolbar({
  count,
  noun,
  selectedCount,
  itemCount,
  allSelected,
  onToggleSelectAll,
  generateLabel,
  generateTitle,
  onGenerate,
  generateDisabledTitle,
  negativesLabel,
  negativesTitle,
  onNegatives,
  assistBusy = false,
  onStandardNegatives,
  standardNegativesTitle,
  disabled = false,
  modelRow,
  range,
  negativesOn,
  onNegativesOnChange,
  negativesOnTitle,
  stdChecked,
  onStdChange,
  stdTitle,
  enhanceLabel,
  enhanceTitle,
  onEnhance,
  canEnhance = true,
  actionsExtra,
  below,
}: PromptListToolbarProps) {
  const actionsDisabled = disabled || assistBusy || selectedCount === 0;
  const grouped =
    modelRow != null ||
    range != null ||
    onNegativesOnChange != null ||
    onStdChange != null ||
    onEnhance != null ||
    actionsExtra != null ||
    below != null;

  // Clamp + drag rule (k93 §A.1): 1 ≤ start ≤ end ≤ itemCount; raising start above
  // end drags end up; lowering end below start drags start down.
  function setRangeStart(raw: string) {
    if (!range || itemCount === 0) return;
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    const s = Math.min(Math.max(1, Math.round(n)), itemCount);
    const e = Math.max(range.end ?? s, s);
    range.onChange(s, Math.min(e, itemCount));
  }
  function setRangeEnd(raw: string) {
    if (!range || itemCount === 0) return;
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    const e = Math.min(Math.max(1, Math.round(n)), itemCount);
    const s = Math.min(range.start ?? e, e);
    range.onChange(Math.max(s, 1), e);
  }

  const countKnob = count && (
    <div className="vi-knob vi-prompt-toolbar-count">
      <label htmlFor={count.id}>{count.label}</label>
      <CountInput count={count} disabled={disabled} />
    </div>
  );

  const selectAll = (
    <label className="vi-comfy-hint vi-prompt-toolbar-selectall">
      <input
        type="checkbox"
        checked={allSelected}
        disabled={disabled || itemCount === 0}
        onChange={onToggleSelectAll}
        aria-label={`Select every ${noun}`}
      />
      {range ? "select all" : `Select all (${selectedCount}/${itemCount})`}
    </label>
  );

  const generateBtn = (
    <button
      type="button"
      className="vi-btn vi-btn-sm vi-btn-accent"
      disabled={actionsDisabled}
      title={selectedCount === 0 ? (generateDisabledTitle ?? generateTitle) : generateTitle}
      onClick={onGenerate}
    >
      {generateLabel}
    </button>
  );

  if (!grouped) {
    return (
      <div
        className="vi-prompt-toolbar"
        role="group"
        aria-label={`${noun} count and group prompt assist`}
      >
        {countKnob}
        {selectAll}
        {generateBtn}
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={actionsDisabled}
          title={negativesTitle}
          onClick={onNegatives}
        >
          {negativesLabel}
        </button>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={disabled || itemCount === 0}
          title={
            standardNegativesTitle ??
            "Tick the [standard negative] checkbox on for the selected rows (or every row when none are selected)."
          }
          onClick={onStandardNegatives}
        >
          standard negatives
        </button>
      </div>
    );
  }

  // ── k93 three-line group layout ──────────────────────────────────────────
  //   prompt assist model  [select]
  //   [start] - [end]   [ ] select all    negatives: [on/off]  [ ] std.
  //   (✨ generate (N selected))  (✨ enhance (M selected))  [+ video]
  return (
    <div
      className="vi-prompt-toolbar vi-prompt-toolbar-group"
      role="group"
      aria-label={`${noun} count and group prompt assist`}
    >
      {modelRow && <div className="vi-prompt-toolbar-line">{modelRow}</div>}
      <div className="vi-prompt-toolbar-line">
        {countKnob}
        {range && (
          <span className="vi-prompt-toolbar-range" title={`1-based ${noun} range the group actions target (text ${noun}s only).`}>
            <input
              type="number"
              className="vi-knob-input vi-prompt-toolbar-range-input"
              min={1}
              max={Math.max(1, itemCount)}
              value={range.start ?? ""}
              placeholder="start"
              disabled={disabled || itemCount === 0 || allSelected}
              aria-label={`First ${noun} of the selection range`}
              onChange={(e) => setRangeStart(e.target.value)}
            />
            <span aria-hidden="true">-</span>
            <input
              type="number"
              className="vi-knob-input vi-prompt-toolbar-range-input"
              min={1}
              max={Math.max(1, itemCount)}
              value={range.end ?? ""}
              placeholder="end"
              disabled={disabled || itemCount === 0 || allSelected}
              aria-label={`Last ${noun} of the selection range`}
              onChange={(e) => setRangeEnd(e.target.value)}
            />
          </span>
        )}
        {selectAll}
        {onNegativesOnChange ? (
          <label
            className="vi-comfy-hint vi-prompt-toolbar-neg"
            title={
              negativesOnTitle ??
              "On: group generate / enhance also write each selected row's Negative field. Off: prompts only."
            }
          >
            negatives:
            <button
              type="button"
              className={`vi-btn vi-btn-sm vi-toggle${negativesOn ? " vi-toggle-on" : ""}`}
              role="switch"
              aria-checked={!!negativesOn}
              disabled={disabled}
              onClick={() => onNegativesOnChange(!negativesOn)}
            >
              {negativesOn ? "on" : "off"}
            </button>
          </label>
        ) : (
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={actionsDisabled}
            title={negativesTitle}
            onClick={onNegatives}
          >
            {negativesLabel}
          </button>
        )}
        {onStdChange ? (
          <label
            className="vi-comfy-hint vi-prompt-toolbar-std"
            title={
              stdTitle ??
              "Tick: insert the standard exclusion set into every selected row's Negative field (visible, sent as-is). Untick: remove exactly those terms."
            }
          >
            <input
              type="checkbox"
              checked={!!stdChecked}
              disabled={disabled || selectedCount === 0}
              onChange={(e) => onStdChange(e.target.checked)}
              aria-label="Insert the standard negative into the selected rows"
            />
            std.
          </label>
        ) : (
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={disabled || itemCount === 0}
            title={
              standardNegativesTitle ??
              "Tick the [standard negative] checkbox on for the selected rows (or every row when none are selected)."
            }
            onClick={onStandardNegatives}
          >
            standard negatives
          </button>
        )}
      </div>
      <div className="vi-prompt-toolbar-line">
        {generateBtn}
        {onEnhance && (
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={actionsDisabled || !canEnhance}
            title={enhanceTitle}
            onClick={onEnhance}
          >
            {enhanceLabel ?? "✨ enhance"}
          </button>
        )}
        {actionsExtra}
      </div>
      {below}
    </div>
  );
}

// ── PromptCard (the row shell) ──────────────────────────────────────────────

/** The optional leading select-checkbox slot (group-assist selection). */
export interface PromptCardSelect {
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
  title?: string;
  /** aria-label; defaults to "Select <noun> N". */
  label?: string;
}

export interface PromptCardProps {
  /** React's list key — consumed by React, never forwarded. Declared because this
      repo type-checks JSX attributes against props with no React types installed. */
  key?: string;
  /** Identifier badge text — "segment" / "goal". */
  badge: string;
  /** 0-based row position; rendered as "#N+1". */
  index: number;
  /** "segment" / "goal" — aria labels + button titles. */
  noun: string;
  select?: PromptCardSelect;
  /** Extra header content between the index and the action cluster (joint descriptor …). */
  headerHint?: ReactNode;
  /** Move/Remove render ONLY when their callback is given (k89: Clip's single card
      passes none, so its action cluster is empty rather than three dead buttons). */
  canMoveUp?: boolean;
  canMoveDown?: boolean;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  moveUpTitle?: string;
  moveDownTitle?: string;
  /** Split renders ONLY when this is given (Movie has it, Cinema does not). */
  onSplit?: () => void;
  splitTitle?: string;
  canRemove?: boolean;
  onRemove?: () => void;
  removeTitle?: string;
  disabled?: boolean;
  children: ReactNode;
}

export function PromptCard({
  badge,
  index,
  noun,
  select,
  headerHint,
  canMoveUp,
  canMoveDown,
  onMoveUp,
  onMoveDown,
  moveUpTitle,
  moveDownTitle,
  onSplit,
  splitTitle,
  canRemove,
  onRemove,
  removeTitle,
  disabled = false,
  children,
}: PromptCardProps) {
  return (
    <li className="vi-gen-part vi-movie-goal vi-prompt-card">
      <div className="vi-gen-part-head">
        {select && (
          <input
            type="checkbox"
            checked={select.checked}
            disabled={disabled || select.disabled}
            onChange={select.onChange}
            aria-label={select.label ?? `Select ${noun} ${index + 1}`}
            title={select.title}
          />
        )}
        <span className="vi-gen-badge vi-movie-badge">{badge}</span>
        <span className="vi-gen-part-idx">#{index + 1}</span>
        {headerHint}
        {(onMoveUp || onMoveDown || onSplit || onRemove) && (
        <div className="vi-gen-part-actions">
          {onMoveUp && (
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={disabled || !canMoveUp}
            onClick={onMoveUp}
            title={moveUpTitle ?? "Move earlier"}
            aria-label={`Move ${noun} earlier`}
          >
            ↑
          </button>
          )}
          {onMoveDown && (
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={disabled || !canMoveDown}
            onClick={onMoveDown}
            title={moveDownTitle ?? "Move later"}
            aria-label={`Move ${noun} later`}
          >
            ↓
          </button>
          )}
          {onSplit && (
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              disabled={disabled}
              onClick={onSplit}
              title={splitTitle ?? `Split this ${noun}`}
              aria-label={`Split ${noun}`}
            >
              Split
            </button>
          )}
          {onRemove && (
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={disabled || !canRemove}
            onClick={onRemove}
            title={removeTitle ?? `Remove ${noun}`}
            aria-label={`Remove ${noun}`}
          >
            ✕
          </button>
          )}
        </div>
        )}
      </div>
      {children}
    </li>
  );
}

// ── PromptFieldTabs ─────────────────────────────────────────────────────────

export interface PromptFieldTabsProps {
  /** DOM id namespace — the prompt pane is `${idPrefix}-prompt`, the negative `${idPrefix}-negative`. */
  idPrefix: string;
  promptLabel: string;
  prompt: string;
  onPromptChange: (v: string) => void;
  promptPlaceholder?: string;
  /** Blur hook for the prompt pane (Cinema's intent router classifies here). */
  onPromptBlur?: () => void;
  negativeLabel?: string;
  negative: string;
  onNegativeChange: (v: string) => void;
  negativePlaceholder?: string;
  negativeHint?: ReactNode;
  stdNegative: boolean;
  onStdNegativeChange: (v: boolean) => void;
  /**
   * When set, the Negative tab + the standard-negative tick render DISABLED with
   * this text as the tooltip (Movie: the wire carries ONE negative — row 0 only).
   */
  negativeDisabledNote?: string;
  rows?: number;
  disabled?: boolean;
  /** Rendered under the negative pane when it is active ("✨ Generate negative" row). */
  negativeExtras?: ReactNode;
  /** Rendered under the prompt pane when it is active. */
  promptExtras?: ReactNode;
  /**
   * k93: render the editor labels ("prompt" / "negative prompt (optional)")
   * visually hidden — the tabs already say it. aria-labels are kept.
   */
  hideLabels?: boolean;
  /**
   * k93: omit the tab-strip [standard negative] tick (the host renders an
   * explicit std. checkbox of its own in its assist row).
   */
  hideStdTick?: boolean;
  /** Tooltip on the Negative tab (Scene: "part #1's field is also the group negative"). */
  negativeTabTitle?: string;
}

export function PromptFieldTabs({
  idPrefix,
  promptLabel,
  prompt,
  onPromptChange,
  promptPlaceholder,
  onPromptBlur,
  negativeLabel = "negative prompt (optional)",
  negative,
  onNegativeChange,
  negativePlaceholder,
  negativeHint,
  stdNegative,
  onStdNegativeChange,
  negativeDisabledNote,
  rows = 2,
  disabled = false,
  negativeExtras,
  promptExtras,
  hideLabels = false,
  hideStdTick = false,
  negativeTabTitle,
}: PromptFieldTabsProps) {
  // Ephemeral VIEW state only — which pane shows. All editor data rides props.
  const [tab, setTab] = useState<"prompt" | "negative">("prompt");
  const negativeLocked = negativeDisabledNote != null;
  const active = negativeLocked ? "prompt" : tab;
  return (
    <div className="vi-prompt-tabbed">
      <div className="vi-prompt-tabs" role="tablist" aria-label="Prompt or negative prompt">
        <button
          type="button"
          role="tab"
          className="vi-prompt-tab"
          aria-selected={active === "prompt"}
          onClick={() => setTab("prompt")}
        >
          Prompt
        </button>
        <button
          type="button"
          role="tab"
          className="vi-prompt-tab"
          aria-selected={active === "negative"}
          disabled={negativeLocked}
          title={negativeDisabledNote ?? negativeTabTitle}
          onClick={() => setTab("negative")}
        >
          Negative
          {/* The content marker — tab state must not hide a written negative. */}
          {negative.trim() !== "" && <span className="vi-prompt-tab-dot" aria-label="has content" />}
        </button>
        {!hideStdTick && (
        <label
          className="vi-comfy-hint vi-prompt-tab-std"
          title={
            negativeDisabledNote ??
            "Append the standard artifact/junk exclusion set to this row's negative when the request is built."
          }
        >
          <input
            type="checkbox"
            checked={stdNegative}
            disabled={disabled || negativeLocked}
            onChange={(e) => onStdNegativeChange(e.target.checked)}
          />
          standard negative
        </label>
        )}
      </div>
      {active === "prompt" ? (
        <>
          <GoalComposer
            id={`${idPrefix}-prompt`}
            label={promptLabel}
            hideLabel={hideLabels}
            value={prompt}
            placeholder={promptPlaceholder}
            onChange={onPromptChange}
            onBlur={onPromptBlur}
            rows={rows}
            disabled={disabled}
          />
          {promptExtras}
        </>
      ) : (
        <>
          <GoalComposer
            id={`${idPrefix}-negative`}
            label={negativeLabel}
            hideLabel={hideLabels}
            value={negative}
            placeholder={negativePlaceholder}
            onChange={onNegativeChange}
            rows={rows}
            disabled={disabled}
            hint={negativeHint}
          />
          {negativeExtras}
        </>
      )}
    </div>
  );
}

// ── PromptCardSettings ──────────────────────────────────────────────────────

/**
 * The labelled settings expander at a card's foot — collapsed by default, styled
 * on the condensed knob-strip rhythm (see .vi-prompt-card-settings). A native
 * <details>, so the open/closed flag lives in the DOM, not in anyone's state.
 * Children are the row's tab-specific knobs (Cinema: seed/joint/branch; Movie:
 * the frame window).
 */
export function PromptCardSettings({
  label = "settings",
  children,
}: {
  label?: string;
  children: ReactNode;
}) {
  return (
    <details className="vi-prompt-card-settings">
      <summary className="vi-prompt-card-settings-summary">{label}</summary>
      <div className="vi-prompt-card-settings-body">{children}</div>
    </details>
  );
}

// ── AboutExpander ───────────────────────────────────────────────────────────

/**
 * The sidebar "About <tab>" expander (k89) — the condensed what-is-this / how-to
 * prose each Studio tab used to carry inline, folded into ONE collapsed-by-default
 * <details> at the top of that tab's Settings-tab content. Same native-<details>
 * rule as PromptCardSettings: the open flag lives in the DOM, never in state.
 * Children are short lines (<p>/<span>) — the body stacks them.
 */
export function AboutExpander({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <details className="vi-about-expander">
      <summary className="vi-about-summary">{title}</summary>
      <div className="vi-about-body">{children}</div>
    </details>
  );
}
