// The RegionEditor contract (architecture map §7) — the shared abstraction every
// region editor in the arm implements. A region editor lets the user pick a slice
// of a MediaRef along ONE axis: spatial (a bbox, this phase's SpatialRegionEditor)
// or temporal (a time window, Phase 5's TemporalRegionEditor). Both editors take
// the same generic props shape and differ only in TRegion + how they paint, so the
// station wiring (queue a region, surface it via onChange) is identical for both.

/** A spatial bounding box in the source image's NATIVE pixel space (integers). */
export interface SpatialRegion {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A temporal window, in seconds — the axis the Phase 5 audio-crop station edits. */
export interface TemporalRegion {
  start_s: number;
  end_s: number;
}

/** Any region an editor manipulates. Widen the union as new axes land. */
export type Region = SpatialRegion | TemporalRegion;

/** Common shape of a one-tap starting-region button. */
export interface RegionPreset {
  /** Button label, e.g. "1:1", "16:9", "1024²". */
  label: string;
}

/**
 * A spatial preset: a labelled aspect ratio (width / height) or a fixed native
 * size. Applying one seeds the working bbox — the largest centered box of that
 * aspect that fits the source, or a box of the fixed `size` clamped to it.
 */
export interface SpatialPreset extends RegionPreset {
  /**
   * Aspect ratio as width / height (e.g. 1 for 1:1, 16 / 9 for 16:9). `null`
   * means "native" — take the source frame's own aspect (the whole image).
   */
  aspect: number | null;
  /**
   * Optional fixed native pixel size, for model-native inputs like 1024². When
   * present it wins over `aspect` for the seeded box (still clamped to the source).
   */
  size?: { w: number; h: number };
}

/**
 * A temporal preset: a one-tap starting window. `seconds` is a fixed window
 * length seeded from the current start (e.g. "1s", "5s"); omit it for "Full" (the
 * whole track). Mirrors SpatialPreset's role on the temporal axis — kept minimal.
 */
export interface TemporalPreset extends RegionPreset {
  /** Fixed window length in seconds; omit for the whole-track "Full" preset. */
  seconds?: number;
}

/**
 * Generic RegionEditor props. A single-region editor binds `value` + `onChange`;
 * a `multiRegion` editor binds `values` (the queue) + `onChange` with the updated
 * queue. onChange carries `TRegion` in single mode and `TRegion[]` in multi mode —
 * one prop so a station wires exactly one handler regardless of mode.
 *
 * Parameterised by the region shape AND the preset shape so the future
 * TemporalRegionEditor can slot in with its own preset type without touching this
 * contract.
 */
export interface RegionEditorProps<
  TRegion extends Region,
  TPreset extends RegionPreset = RegionPreset,
> {
  /** Current working region (single-region mode). */
  value?: TRegion | null;
  /** Queued regions (multi-region mode). */
  values?: TRegion[];
  /** Change surface: the working region (single) or the whole queue (multi). */
  onChange: (next: TRegion | TRegion[]) => void;
  /** One-tap starting-region buttons. */
  presets?: TPreset[];
  /** Render numeric field inputs bound two-way to the working region. */
  numericInputs?: boolean;
  /** Enable the add-to-queue affordance and render the queued list. */
  multiRegion?: boolean;
}
