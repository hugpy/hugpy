/*
 * ComposerControls.tsx — Hugpy-specific sampling-knob strip.
 *
 * Now rendered OUTSIDE the composer surface, as a thin muted row above
 * the disclaimer. This keeps the composer pill clean (convo has no
 * footer cell) while still satisfying the "keep my controls visible
 * inline" rule — they're always on screen, never hidden in a popover.
 *
 * Layout: a single horizontal flex row. Small chip-style inputs that
 * blend into the secondary-text color so they don't pull focus from
 * the composer.
 */

import { CHAT_MODEL_OPTIONS, type ChatModelOption } from "./../imports";

interface ComposerControlsProps {
  loading: boolean;
  modelKey: string;
  /** Live model list from the hugpy registry; falls back to the built-in list. */
  modelOptions?: ChatModelOption[];
  maxNewTokens: number;
  maxAllowedTokens: number;
  temperature: number;
  topP: number;
  streamResponse: boolean;
  onModelKeyChange: (value: string) => void;
  onMaxNewTokensChange: (value: number) => void;
  onTemperatureChange: (value: number) => void;
  onTopPChange: (value: number) => void;
  onStreamResponseChange: (value: boolean) => void;
  onUseMaxTokens: () => void;
}

function clampNumber(
  value: number,
  min: number,
  max: number,
  fallback: number,
): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(Math.max(value, min), max);
}

export default function ComposerControls({
  loading,
  modelKey,
  modelOptions,
  maxNewTokens,
  maxAllowedTokens,
  temperature,
  topP,
  streamResponse,
  onModelKeyChange,
  onMaxNewTokensChange,
  onTemperatureChange,
  onTopPChange,
  onStreamResponseChange,
  onUseMaxTokens,
}: ComposerControlsProps): JSX.Element {
  const options =
    modelOptions && modelOptions.length ? modelOptions : CHAT_MODEL_OPTIONS;

  const safeMaxAllowedTokens =
    Number.isFinite(maxAllowedTokens) && maxAllowedTokens > 0
      ? maxAllowedTokens
      : 512;

  const safeMaxNewTokens =
    Number.isFinite(maxNewTokens) && maxNewTokens > 0
      ? maxNewTokens
      : safeMaxAllowedTokens;

  return (
    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 pt-2 text-[12px] text-token-text-tertiary">
      <label className="control-chip">
        <span className="control-chip-key">Model</span>
        <select
          className="control-chip-select"
          value={modelKey}
          disabled={loading}
          onChange={(event) => onModelKeyChange(event.target.value)}
        >
          {options.map((model) => (
            <option key={model.key} value={model.key}>
              {model.shortLabel ?? model.label}
            </option>
          ))}
        </select>
      </label>

      <span className="control-chip-sep" aria-hidden>·</span>

      <label className="control-chip">
        <span className="control-chip-key">Tokens</span>
        <input
          type="number"
          className="control-chip-input"
          min={1}
          max={safeMaxAllowedTokens}
          value={safeMaxNewTokens}
          disabled={loading}
          onChange={(event) => {
            const next = Number(event.target.value);
            onMaxNewTokensChange(
              clampNumber(next, 1, safeMaxAllowedTokens, safeMaxAllowedTokens),
            );
          }}
        />
        <button
          type="button"
          className="control-chip-link"
          disabled={loading}
          onClick={onUseMaxTokens}
        >
          max
        </button>
      </label>

      <span className="control-chip-sep" aria-hidden>·</span>

      <label className="control-chip">
        <span className="control-chip-key">Temp</span>
        <input
          type="number"
          className="control-chip-input"
          min={0}
          max={2}
          step={0.1}
          value={Number.isFinite(temperature) ? temperature : 0}
          disabled={loading}
          onChange={(event) => {
            const next = Number(event.target.value);
            onTemperatureChange(clampNumber(next, 0, 2, 0));
          }}
        />
      </label>

      <span className="control-chip-sep" aria-hidden>·</span>

      <label className="control-chip">
        <span className="control-chip-key">Top-P</span>
        <input
          type="number"
          className="control-chip-input"
          min={0}
          max={1}
          step={0.01}
          value={Number.isFinite(topP) ? topP : 0.95}
          disabled={loading}
          onChange={(event) => {
            const next = Number(event.target.value);
            onTopPChange(clampNumber(next, 0, 1, 0.95));
          }}
        />
      </label>

      <span className="control-chip-sep" aria-hidden>·</span>

      <label className="control-chip cursor-pointer">
        <input
          type="checkbox"
          className="control-chip-check"
          checked={streamResponse}
          disabled={loading}
          onChange={(event) => onStreamResponseChange(event.target.checked)}
        />
        <span className="control-chip-key">Stream</span>
      </label>
    </div>
  );
}
