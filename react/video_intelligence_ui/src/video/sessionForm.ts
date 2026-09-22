// SESSION-SCOPED FORM MEMORY — "my inputs are still there when I come back".
//
// ── THE BUG THIS FIXES (operator ask 2026-08-06) ────────────────────────────
// Switching sub-tabs inside the STUDIO tab (Scene / Movie / Clip / Cinema) and
// switching workbench tabs (Studio ↔ Identities ↔ Frames …) both UNMOUNT the
// surface you were typing into: WorkbenchStation renders exactly one station
// body, and GenerateStation swaps the SectionTabs layout for StudioGenerateMode
// (and StudioGenerateSurface swaps its clip surface for StudioMovieComposer).
// Plain useState dies with the unmount, so a half-written prompt, a chosen
// model, a staged reference set — all gone for the price of a glance at another
// tab.
//
// ── WHY A STORE AND NOT KEEP-MOUNTED-BUT-HIDDEN ─────────────────────────────
// Rendering the inactive sub-tabs behind `hidden` would fix the SUB-tab half and
// nothing else — a main-tab switch unmounts the whole station regardless, and
// keeping every station mounted is not on the table. It would also double-mount
// the pieces that register into the shared sidebar (StudioPlane's
// registerSettings + its knob PORTAL both land in the ONE settingsHost node) and
// double the studio clip poll. So state is LIFTED out of the component lifetime
// instead — which is exactly the idiom this arm already uses for everything else
// that has to outlive a mount: mediaLibrary.ts and jobTracker.ts are both
// module-level stores persisted to sessionStorage under the session id.
//
// ── SHAPE ───────────────────────────────────────────────────────────────────
// A per-tab key/value store, session-id-scoped like mediaLibrary (so two tabs
// keep independent inputs and a new session starts clean), with a useState-shaped
// hook so a call site converts by swapping one word:
//
//     const [prompt, setPrompt] = useState("");
//     const [prompt, setPrompt] = useSessionState("studio.clip.prompt", "");
//
// In-memory is authoritative (synchronous, no JSON round-trip per keystroke);
// sessionStorage is a debounced mirror so a reload also keeps the inputs. If
// storage is unavailable (private mode, quota) the memory half still works — the
// feature degrades to "survives unmounts but not reloads", never to an error.
//
// ── WHAT BELONGS HERE, AND WHAT DOES NOT ────────────────────────────────────
// USER-AUTHORED INPUT only: prompts, knobs, picked models/presets, attached
// refs, which sub-tab you were on. NOT transient UI (open pickers, previews),
// NOT in-flight job/upload state, NOT anything unserializable (a Set/Map never
// survives JSON — leave those on useState). Restoring a stale "uploading…" or a
// half-open picker would be a worse bug than the one being fixed.
//
// Keys are global: two components sharing a key share the VALUE but not the
// React subscription (each holds its own useState), so only ever use one key per
// logical field, and never for two surfaces that are mounted at the same time.
import { useCallback, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { getSessionId } from "../session";
import { isCanned } from "../demo/mode";

const KEY_PREFIX = "vi.formMemory.v1";
const FLUSH_MS = 300;

// Session-scoped key so two tabs (two sessions) keep independent inputs, with the
// same `:demo`-suffix hermeticism mediaLibrary/jobTracker use so the canned
// brochure never reads or writes the live session's values.
function storageKey(): string {
  const suffix = isCanned() ? ":demo" : "";
  try {
    return `${KEY_PREFIX}:${getSessionId()}${suffix}`;
  } catch {
    return `${KEY_PREFIX}${suffix}`;
  }
}

function hydrate(): Map<string, unknown> {
  try {
    const raw = sessionStorage.getItem(storageKey());
    if (!raw) return new Map();
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return new Map();
    return new Map(Object.entries(parsed as Record<string, unknown>));
  } catch {
    return new Map(); // storage unavailable or corrupt — start empty, quietly
  }
}

// Hydrated ONCE at module load: every read after that is a Map lookup, so a
// keystroke never pays a JSON.parse.
const memory: Map<string, unknown> = hydrate();

let flushTimer: number | null = null;

function flush(): void {
  flushTimer = null;
  try {
    sessionStorage.setItem(
      storageKey(),
      JSON.stringify(Object.fromEntries(memory)),
    );
  } catch {
    /* quota/private mode — the in-memory half still carries this session */
  }
}

function scheduleFlush(): void {
  if (typeof window === "undefined") return;
  if (flushTimer != null) return;
  flushTimer = window.setTimeout(flush, FLUSH_MS);
}

function readValue<T>(key: string, fallback: T): T {
  return memory.has(key) ? (memory.get(key) as T) : fallback;
}

function writeValue(key: string, value: unknown): void {
  memory.set(key, value);
  scheduleFlush();
}

/**
 * useState, but the value outlives the component's mount for the whole browser
 * session (and a reload). Drop-in: same tuple, same updater-function support.
 *
 * `initial` is used only the FIRST time a key is seen this session; afterwards
 * the stored value wins, which is the entire point (a remount must not re-run
 * the default over what the operator typed).
 */
export function useSessionState<T>(
  key: string,
  initial: T | (() => T),
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    if (memory.has(key)) return memory.get(key) as T;
    const seed = typeof initial === "function" ? (initial as () => T)() : initial;
    writeValue(key, seed);
    return seed;
  });

  const set = useCallback<Dispatch<SetStateAction<T>>>(
    (action) => {
      setValue((prev) => {
        const next =
          typeof action === "function" ? (action as (p: T) => T)(prev) : action;
        // Writing from inside the updater keeps the store in step with the
        // functional-update form (setX(cur => …) is used all over these
        // surfaces). React may invoke an updater twice under StrictMode; the
        // write is idempotent for the same `prev`, so that is harmless.
        writeValue(key, next);
        return next;
      });
    },
    [key],
  );

  return [value, set];
}

/**
 * The next value of a session-persisted counter — for the module-level `let seq
 * = 0` row-key generators (`goal_3`, `ref_2`) that back rows now restored from
 * storage.
 *
 * Those counters reset to 0 on a page reload while the ROWS they keyed come back
 * from sessionStorage, so a freshly minted key could collide with a restored one
 * (React would then reconcile two different rows as the same element, and in the
 * movie composer the key doubles as the wire `segment_id`). Persisting the
 * counter is the smallest fix that keeps the existing key SHAPE intact.
 */
export function nextSessionSeq(key: string): number {
  const next = (readValue<number>(key, 0) ?? 0) + 1;
  writeValue(key, next);
  return next;
}

/**
 * A ref whose `.current` lives in the same session store — for the flags that
 * deliberately are NOT state because they must never cause a render (the
 * "operator has touched this field" marks that gate the studio's geometry/budget
 * autofill effects).
 *
 * These have to persist alongside the values they guard: restore a hand-typed
 * width but not the "the operator typed it" mark, and the autofill effect fires
 * on the next mount and clobbers exactly the value we just restored.
 */
export function useSessionRef<T>(key: string, initial: T): { current: T } {
  const holder = useRef<{ current: T } | null>(null);
  if (holder.current === null) {
    if (!memory.has(key)) writeValue(key, initial);
    const obj = {} as { current: T };
    Object.defineProperty(obj, "current", {
      get: () => readValue<T>(key, initial),
      set: (v: T) => writeValue(key, v),
      enumerable: true,
    });
    holder.current = obj;
  }
  return holder.current;
}
