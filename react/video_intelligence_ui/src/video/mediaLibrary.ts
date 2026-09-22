// The media library — a session-scoped, cross-station store of produced MediaRefs.
//
// This is the FieldSpec-propagation analog for the video-intelligence arm: a
// station (Frames today) pushes the frames the user picks in here, and a later
// station (Generate) reads them back as selectable prompt inputs. Backed by
// sessionStorage (per-tab, keyed by session id) so a page reload keeps the picks
// but a new tab/session starts clean. `clearLibrary` empties the SESSION list
// only — it NEVER deletes the underlying server assets (house policy: no deletion).
//
// The public surface below is FROZEN — the Generate station imports it verbatim.
import { useSyncExternalStore } from "react";
import { getSessionId } from "../session";
import { isCanned } from "../demo/mode";
import type { MediaRef } from "./contract";

/** The generation kinds whose outputs carry prompt/model provenance. */
export type LibraryGenKind =
  | "generate_image"
  | "generate_scene"
  | "generate_movie"
  | "studio_i2v";

export interface LibraryItem {
  /** The produced asset. */
  ref: MediaRef;
  /** Which station/flow produced it (e.g. "frames"). */
  origin: string;
  /** Optional human label (e.g. "frame 12"). */
  label?: string;
  /** When it landed in the library (ms epoch). */
  addedAt: number;
  /**
   * Generation provenance — present ONLY for Generate-station outputs (image &
   * scene). Captured client-side at generation time (the composer already holds
   * the prompt + model), so the session-library can annotate each output and
   * re-stage its prompt/model for Replicate / Continue. Only the server MediaRef
   * (not the pixels) is persisted, per the library's storage-quota policy.
   */
  prompt?: string;
  model?: string;
  /** Groups a single generation run's outputs together (the producing job id). */
  groupId?: string;
  /** The generation kind of the producing run — enables Continue for scenes. */
  genKind?: LibraryGenKind;
}

/** Optional generation provenance recorded alongside a produced MediaRef. */
export interface LibraryProvenance {
  prompt?: string;
  model?: string;
  groupId?: string;
  genKind?: LibraryGenKind;
}

/**
 * The immutable "where did this come from" half of a first-class StudioItem (see
 * stations/studio/studioItem.ts). Defined HERE — next to the library provenance
 * types — rather than in studioItem.ts so the video/ layer (studioBridge's
 * StudioStageRequest) can reference it WITHOUT importing from stations/, preserving
 * the stations/ → video/ dependency direction. studioItem.ts re-exports it as the
 * canonical StudioItem surface.
 */
export interface StudioItemProvenance {
  /** How this item entered the studio: "upload" | "library" | "send-to-studio" |
   *  "restyle" | "op:<kind>" | … */
  origin: string;
  prompt?: string;
  model?: string;
  genKind?: LibraryGenKind;
  /** uri of the ref this item derived from — the forward-only provenance DAG edge. */
  parentUri?: string;
}

const KEY_PREFIX = "vi.mediaLibrary.v1";

// Session-scoped key so two tabs (two sessions) keep independent libraries, and
// the persisted list travels with the same session id the uploads are tagged by.
//
// The canned demo (?demo=1) gets a `:demo`-SUFFIXED key (matching jobTracker) so
// the brochure's seeded library is hermetic — it never reads or writes the SAME
// tab's live-session library, and the tombstone key (derived below from
// storageKey) follows suit. isCanned() is URL-derived and available at module
// load, so the key is right from the first read regardless of import order.
function storageKey(): string {
  const suffix = isCanned() ? ":demo" : "";
  try {
    return `${KEY_PREFIX}:${getSessionId()}${suffix}`;
  } catch {
    return `${KEY_PREFIX}${suffix}`;
  }
}

// In-memory cache is the source of truth for getSnapshot — its reference only
// changes when the list actually changes, which keeps useSyncExternalStore stable
// (returning a fresh array every read would loop React 18).
let cache: LibraryItem[] | null = null;
const listeners = new Set<() => void>();
let storageBound = false;

function readStorage(): LibraryItem[] {
  try {
    const raw = sessionStorage.getItem(storageKey());
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Drop malformed entries defensively — a stored item must at least carry a uri.
    return parsed.filter(
      (it) =>
        it &&
        typeof it === "object" &&
        it.ref &&
        typeof it.ref === "object" &&
        typeof it.ref.uri === "string",
    ) as LibraryItem[];
  } catch {
    return [];
  }
}

function writeStorage(items: LibraryItem[]): void {
  try {
    sessionStorage.setItem(storageKey(), JSON.stringify(items));
  } catch {
    // storage unavailable / quota exceeded — the in-memory cache still serves this
    // session; persistence is best-effort.
  }
}

function ensureLoaded(): LibraryItem[] {
  if (cache == null) cache = readStorage();
  return cache;
}

function emit(): void {
  for (const l of listeners) l();
}

// ── Tombstones ─────────────────────────────────────────────────────────────
// A "Remove" that comes back on the next poll isn't a remove. The list has TWO
// re-populators that fire AFTER a local forget: jobTracker's terminal pushOutputs
// (re-runs on a reload / a re-poll of a done job) and the studio-clips list poll
// (re-adds every ~6s). addToLibrary's only guard was uri-dedup, which a *removed*
// uri no longer trips — so the item silently returned. Tombstones fix this at the
// single choke point: a removed uri is remembered and addToLibrary refuses to
// re-add it. Session-scoped (same key family as the list) so removal survives
// reloads within the session; studio clips ALSO archive server-side in the row
// handler, which stops the server from serving them across sessions too.
function tombstoneKey(): string {
  return `${storageKey()}:removed`;
}

let tombstones: Set<string> | null = null;

function ensureTombstones(): Set<string> {
  if (tombstones == null) {
    try {
      const raw = sessionStorage.getItem(tombstoneKey());
      const parsed = raw ? JSON.parse(raw) : [];
      tombstones = new Set(
        Array.isArray(parsed) ? parsed.filter((u) => typeof u === "string") : [],
      );
    } catch {
      tombstones = new Set();
    }
  }
  return tombstones;
}

function writeTombstones(): void {
  try {
    sessionStorage.setItem(tombstoneKey(), JSON.stringify([...ensureTombstones()]));
  } catch {
    // best-effort — the in-memory set still suppresses re-adds this session.
  }
}

/** Mark `uri` as removed so no re-populator can re-add it. */
function tombstone(uri: string): void {
  ensureTombstones().add(uri);
  writeTombstones();
}

/** Whether `uri` was explicitly removed (and must not be re-added). */
export function isRemoved(uri: string): boolean {
  return ensureTombstones().has(uri);
}

/** The current library snapshot (stable reference until it changes). */
export function getLibrary(): LibraryItem[] {
  return ensureLoaded();
}

/**
 * Add a produced MediaRef to the library. De-duplicates by `ref.uri` (no-op if
 * present). The optional `provenance` argument is additive (existing 3-arg
 * callers are unchanged) — it records the prompt/model/group that produced a
 * Generate-station output so the session-library can annotate + re-stage it.
 */
export function addToLibrary(
  ref: MediaRef,
  origin: string,
  label?: string,
  provenance?: LibraryProvenance,
): void {
  if (!ref || typeof ref.uri !== "string") return;
  // A uri the user explicitly removed stays gone — never let a re-populator
  // (jobTracker terminal push / studio-clips poll) resurrect it. See tombstones.
  if (isRemoved(ref.uri)) return;
  const current = ensureLoaded();
  if (current.some((it) => it.ref.uri === ref.uri)) return;
  const item: LibraryItem = { ref, origin, addedAt: Date.now() };
  if (label != null) item.label = label;
  if (provenance?.prompt != null) item.prompt = provenance.prompt;
  if (provenance?.model != null) item.model = provenance.model;
  if (provenance?.groupId != null) item.groupId = provenance.groupId;
  if (provenance?.genKind != null) item.genKind = provenance.genKind;
  cache = [...current, item];
  writeStorage(cache);
  emit();
}

/**
 * Merge generation provenance into an EXISTING library item (matched by `ref.uri`).
 * Additive back-fill for provenance discovered AFTER the ref was first added: the
 * studio clip catalog adds each produced clip on its list poll (keyed by uri, but the
 * clips-list DTO carries no prompt), then fills the REAL prompt/model in from the
 * per-clip detail fetch — this is that seam. Only writes a field the item does not
 * already carry (never clobbers a value captured at generation time), and no-ops if
 * the uri is absent or nothing would change. Never touches server assets.
 */
export function updateLibraryProvenance(
  uri: string,
  provenance: LibraryProvenance,
): void {
  if (typeof uri !== "string" || !provenance) return;
  const current = ensureLoaded();
  const idx = current.findIndex((it) => it.ref.uri === uri);
  if (idx < 0) return;
  const item = current[idx];
  const next: LibraryItem = { ...item };
  let changed = false;
  if (provenance.prompt != null && item.prompt == null) {
    next.prompt = provenance.prompt;
    changed = true;
  }
  if (provenance.model != null && item.model == null) {
    next.model = provenance.model;
    changed = true;
  }
  if (provenance.groupId != null && item.groupId == null) {
    next.groupId = provenance.groupId;
    changed = true;
  }
  if (provenance.genKind != null && item.genKind == null) {
    next.genKind = provenance.genKind;
    changed = true;
  }
  if (!changed) return;
  cache = [...current.slice(0, idx), next, ...current.slice(idx + 1)];
  writeStorage(cache);
  emit();
}

/**
 * Remove a SINGLE library entry by its `ref.uri`, PERMANENTLY for the session:
 * the uri is tombstoned so no re-populator (jobTracker terminal push / studio-clips
 * poll) can ever bring it back — a removal that repopulates isn't a removal. Does
 * NOT itself delete the underlying server asset (house policy: no deletion; studio
 * clips archive server-side in the row handler). Tombstone is recorded even when
 * the row is momentarily absent, so a remove that races a re-add still sticks.
 */
export function removeFromLibrary(uri: string): void {
  if (typeof uri !== "string") return;
  tombstone(uri); // record the removal FIRST so a concurrent re-add can't win.
  const current = ensureLoaded();
  const next = current.filter((it) => it.ref.uri !== uri);
  if (next.length === current.length) {
    emit(); // not in the list, but the tombstone changed — let subscribers re-render.
    return;
  }
  cache = next;
  writeStorage(cache);
  emit();
}

/**
 * Empty the SESSION library list. Tombstones every current uri so the studio-clips
 * poll (and any terminal re-push) can't repopulate what was just cleared — "Clear
 * all" that refills on the next tick isn't clearing. Does NOT delete server assets
 * (house policy: no deletion).
 */
export function clearLibrary(): void {
  for (const it of ensureLoaded()) tombstone(it.ref.uri);
  cache = [];
  try {
    sessionStorage.removeItem(storageKey());
  } catch {
    // ignore — cache is already cleared for this session.
  }
  emit();
}

// Bind the cross-tab storage listener once. sessionStorage is per-tab so this
// mainly future-proofs the surface, but if the key ever changes underneath us we
// re-sync the cache and notify rather than serving a stale list.
function bindStorageOnce(): void {
  if (storageBound || typeof window === "undefined") return;
  storageBound = true;
  window.addEventListener("storage", (e) => {
    if (e.key !== null && e.key !== storageKey()) return;
    cache = readStorage();
    emit();
  });
}

/** Subscribe to add/clear (and cross-tab storage) changes. Returns an unsubscribe. */
export function subscribeLibrary(listener: () => void): () => void {
  listeners.add(listener);
  bindStorageOnce();
  return () => {
    listeners.delete(listener);
  };
}

/** React hook: the live library list, re-rendering on every add/clear. */
export function useMediaLibrary(): LibraryItem[] {
  return useSyncExternalStore(subscribeLibrary, getLibrary, getLibrary);
}
