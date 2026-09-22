// The first-class STUDIO ITEM (Studio redesign, slice 1) — the single "item" object
// the studio works over, replacing the loose bag of controlled `source` useState.
//
// A StudioItem has an IMMUTABLE half {source, provenance} and a MUTABLE half
// {working, ops}. `working` starts identical to `source`; later slices' same-kind
// ops (crop / frame-extract) REPLACE `working` (never `source`). Provenance is a
// forward-only DAG: each item records the `parentUri` of the ref it derived from,
// mirroring the produced-library provenance pattern in video/mediaLibrary.ts
// (LibraryItem + updateLibraryProvenance). The source — and the library item it came
// from — is never rewritten.
//
// LAYERING: StudioItemProvenance lives in video/mediaLibrary.ts (next to the library
// provenance types) so the video/ layer (studioBridge's StudioStageRequest) can name
// it without importing from stations/ — preserving the stations/ → video/ direction.
// It is re-exported here so studioItem.ts is the canonical StudioItem surface.
import type { MediaRef } from "../../video/contract";
import type { SpatialRegion, TemporalRegion } from "../../regions/types";
import type { StudioItemProvenance } from "../../video/mediaLibrary";

export type { StudioItemProvenance };

/** One recorded edit over an item's working ref. Same-kind ops replace `working`;
 *  the op keeps its own region + result so the ops list is a replayable history. */
export interface StudioItemOp {
  id: string;
  kind: "image_crop" | "video_crop" | "audio_crop" | "frame_extract";
  /** When the op ran (ms epoch). */
  at: number;
  spatial?: SpatialRegion | null;
  temporal?: TemporalRegion | null;
  result?: MediaRef;
}

/** A first-class studio item: immutable {source, provenance} + mutable {working, ops}. */
export interface StudioItem {
  /** Stable LOCAL identity (NOT the uri — a ref may legitimately repeat). */
  key: string;
  /** The origin ref. Never rewritten. */
  source: MediaRef;
  provenance: StudioItemProvenance;
  /** The current working ref — starts === source, replaced by same-kind ops. */
  working: MediaRef;
  ops: StudioItemOp[];
}

// Stable per-item local key. Mirrors the arm's fresh-id idiom (useCropJobs.makeKey /
// GenerateStation.partKey): a monotonic counter + a base-36 timestamp, so keys are
// unique within a session without a new dependency.
let _itemSeq = 0;
function itemKey(): string {
  _itemSeq += 1;
  return `item_${_itemSeq.toString(36)}_${Date.now().toString(36)}`;
}

/** Mint a fresh StudioItem from a source ref: working = source, ops = [], fresh key. */
export function makeStudioItem(
  source: MediaRef,
  provenance: StudioItemProvenance,
): StudioItem {
  return {
    key: itemKey(),
    source,
    provenance,
    working: source,
    ops: [],
  };
}

/** Apply an op — returns a NEW item with working = op.result and the op appended.
 *  A resultless op (not yet completed) leaves `working` unchanged so the field stays
 *  a valid MediaRef; the op is still recorded in the history. */
export function applyOp(item: StudioItem, op: StudioItemOp): StudioItem {
  return {
    ...item,
    working: op.result ?? item.working,
    ops: [...item.ops, op],
  };
}

/** Revert the working ref back to the immutable source — returns a NEW item.
 *  Forward-only: the ops history is preserved (source is never rewritten). */
export function revertItem(item: StudioItem): StudioItem {
  return {
    ...item,
    working: item.source,
  };
}
