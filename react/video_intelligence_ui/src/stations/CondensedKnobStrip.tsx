// CONDENSED KNOB STRIP + CARRY TICKS — extracted from GenerateStation.tsx (task
// k88) so the Scene/Clip prompt-card migration (k89) can consume the same strip
// without importing the whole station. Everything below moved VERBATIM (values,
// flags, ticks, layout); GenerateStation imports it back unchanged.
import type { ReactNode } from "react";
import { KnobInfo } from "./KnobInfo";
import type { CarrySettingKey, CarryTicks } from "../video/contract";

// Parse a text field into a number, or null when blank/invalid (mirrors the
// Frames station's helper — no silent coercion of garbage to 0). Exported so
// GenerateStation (its original home) keeps its single source.
export function toNumberOrNull(raw: string): number | null {
  const s = raw.trim();
  if (s === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

/** The five numeric knobs the condensed strip edits, as the live text they are. */
export interface KnobValues {
  width: string;
  height: string;
  steps: string;
  guidance: string;
  seed: string;
}

/**
 * The strip's validation flags, derived from the raw text — the SAME rules the
 * station's Run gate uses (widthValid/heightValid/stepsValid/guidanceValid/
 * seedValid + the multiple-of-8 warning), restated here so the component is
 * self-contained and every host (base composer, every tester row) flags identically.
 * Returned as a list so the strip itself stays ONE line and the flags stack under it.
 */
function knobFlags(v: KnobValues): string[] {
  const out: string[] = [];
  const w = toNumberOrNull(v.width);
  const h = toNumberOrNull(v.height);
  const st = toNumberOrNull(v.steps);
  const g = toNumberOrNull(v.guidance);
  const sd = toNumberOrNull(v.seed);
  const wOk = w != null && Number.isInteger(w) && w > 0;
  const hOk = h != null && Number.isInteger(h) && h > 0;
  if (!wOk) out.push("width: enter a positive integer.");
  else if (w % 8 !== 0) out.push("width: not a multiple of 8 — the model may round it.");
  if (!hOk) out.push("height: enter a positive integer.");
  else if (h % 8 !== 0) out.push("height: not a multiple of 8 — the model may round it.");
  if (!(st != null && Number.isInteger(st) && st > 0))
    out.push("steps: enter a positive integer.");
  if (!(g != null && g >= 0)) out.push("guidance: enter a number ≥ 0.");
  if (!(v.seed.trim() === "" || (sd != null && Number.isInteger(sd) && sd >= 0)))
    out.push("seed: enter a non-negative integer or leave blank.");
  return out;
}

/**
 * ONE carry tick — the small checkbox that rides in front of a setting's label
 * (operator ask 2026-08-04, k63). Checked = this setting is carried into the
 * component created from this one; unchecked = that component gets the mode
 * default instead. Deliberately label-less and title-only: the strip is condensed,
 * and a word of copy per tick would undo the condensing.
 */
export function CarryTick({
  id,
  setting,
  checked,
  onChange,
}: {
  id: string;
  setting: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <input
      id={id}
      type="checkbox"
      className="vi-carry-tick"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
      aria-label={`Carry ${setting} into the next component`}
      title={`Carry ${setting} into the component created from this one. Unticked = that component gets the default instead.`}
    />
  );
}

/**
 * CONDENSED KNOB STRIP (operator ask 2026-08-04, k63) — the generation knobs as
 * ONE inline line rather than a column of full-width blocks:
 *
 *     [model ▾]  ☑ size [512] × [512]  ☑ steps [4]  ☑ guidance [0]  ☑ seed [random]
 *
 * WHY a component and not just JSX: the tester rows must look and behave EXACTLY
 * like the base composer's settings, because the whole point of a row is "the same
 * generation with one thing changed". Two hand-written copies of the same seven
 * inputs is how they drift. Both hosts render this; there is one visual language.
 *
 * Deliberately presentational: it owns no state. Values come in, `onPatch` goes
 * out (the base host maps it onto the station setters, a tester row onto
 * patchTestRow), and the carry ticks are the host's state too. `lead`/`trail` are
 * the host-specific extremities — the model select cluster (which differs: the base
 * carries a refresh button + capability line, a row does not) and the row's inline
 * negative field.
 */
export function CondensedKnobStrip({
  idPrefix,
  values,
  onPatch,
  carry,
  onCarry,
  lead,
  trail,
  footer,
}: {
  /** DOM id namespace — must be unique per instance (ids collide across rows otherwise). */
  idPrefix: string;
  values: KnobValues;
  onPatch: (patch: Partial<KnobValues>) => void;
  /** Omit to render the strip without carry ticks (nothing is seeded from it). */
  carry?: CarryTicks;
  onCarry?: (key: CarrySettingKey, value: boolean) => void;
  lead?: ReactNode;
  trail?: ReactNode;
  footer?: ReactNode;
}) {
  const flags = knobFlags(values);
  const tick = (key: CarrySettingKey, setting: string) =>
    carry ? (
      <CarryTick
        id={`${idPrefix}-carry-${key}`}
        setting={setting}
        checked={carry[key]}
        onChange={(v) => onCarry?.(key, v)}
      />
    ) : null;

  return (
    <div className="vi-knob-strip">
      <div className="vi-knob-strip-line">
        {lead}

        {/* width × height read as ONE setting ("size"), so they pair behind one
            tick and one ⓘ — a literal × between two mini inputs. */}
        <span className="vi-knob-mini-group">
          {tick("size", "size")}
          <label className="vi-knob-mini-label" htmlFor={`${idPrefix}-width`}>
            size
          </label>
          <input
            id={`${idPrefix}-width`}
            type="text"
            inputMode="numeric"
            className="vi-knob-input vi-knob-mini"
            aria-label="width"
            value={values.width}
            onChange={(e) => onPatch({ width: e.target.value })}
          />
          <span className="vi-knob-mini-x" aria-hidden>
            ×
          </span>
          <input
            id={`${idPrefix}-height`}
            type="text"
            inputMode="numeric"
            className="vi-knob-input vi-knob-mini"
            aria-label="height"
            value={values.height}
            onChange={(e) => onPatch({ height: e.target.value })}
          />
          <KnobInfo
            label="size"
            text="Output width × height in pixels. Multiples of 8 avoid the model silently rounding them."
          />
        </span>

        <span className="vi-knob-mini-group">
          {tick("steps", "steps")}
          <label className="vi-knob-mini-label" htmlFor={`${idPrefix}-steps`}>
            steps
          </label>
          <input
            id={`${idPrefix}-steps`}
            type="text"
            inputMode="numeric"
            className="vi-knob-input vi-knob-mini"
            value={values.steps}
            onChange={(e) => onPatch({ steps: e.target.value })}
          />
          <KnobInfo label="steps" text="Denoising steps. sd-turbo runs well at ~4." />
        </span>

        <span className="vi-knob-mini-group">
          {tick("guidance", "guidance")}
          <label className="vi-knob-mini-label" htmlFor={`${idPrefix}-guidance`}>
            guidance
          </label>
          <input
            id={`${idPrefix}-guidance`}
            type="text"
            inputMode="decimal"
            className="vi-knob-input vi-knob-mini"
            value={values.guidance}
            onChange={(e) => onPatch({ guidance: e.target.value })}
          />
          <KnobInfo
            label="guidance"
            text="CFG scale. sd-turbo uses 0 (guidance-free)."
          />
        </span>

        <span className="vi-knob-mini-group">
          {tick("seed", "seed")}
          <label className="vi-knob-mini-label" htmlFor={`${idPrefix}-seed`}>
            seed
          </label>
          <input
            id={`${idPrefix}-seed`}
            type="text"
            inputMode="numeric"
            className="vi-knob-input vi-knob-mini vi-knob-mini-seed"
            placeholder="random"
            value={values.seed}
            onChange={(e) => onPatch({ seed: e.target.value })}
          />
          <KnobInfo label="seed" text="Optional. Blank = random each run." />
        </span>

        {trail}
      </div>

      {/* Flags stack BELOW the line so a warning never breaks the strip's single
          row. Same rules, same amber .vi-knob-flag as the rail knobs had. */}
      {flags.length > 0 && (
        <div className="vi-knob-strip-flags">
          {flags.map((f) => (
            <span className="vi-knob-flag" key={f}>
              {f}
            </span>
          ))}
        </div>
      )}
      {footer}
    </div>
  );
}
