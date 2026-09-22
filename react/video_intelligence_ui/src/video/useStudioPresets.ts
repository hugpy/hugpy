// The Studio Clips station's PRESET dropdown source. Fetches the curated studio
// clip presets (GET /video/studio/presets), validates them with a tolerant schema,
// and exposes the list + loading/error as data (never throws) — the same
// data-not-throws contract as useMoviePresets.ts, which it is cloned from.
//
// Two deliberate parities with useMoviePresets:
//   1. The wire is the {presets:[…]} envelope (matches /video/presets + /movie/presets).
//   2. The fetch is GATED behind an `active` flag so a caller can keep it idle until
//      the station is entered (and it stays cached after). The Studio Clips station
//      passes active=true (it only mounts when its tab is open).
//
// A studio preset pins a CAPABILITY ("i2v"/"t2v") + geometry + a routing
// `vram_budget_gb` (NOT a model_key — the studio router picks the model); picking one
// prefills the generate affordance and drives its enqueue.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";

// Tolerant/passthrough — assert only the fields the station reads; a preset may omit
// the advisory ones (the station keeps its current value). Kept INLINE (like
// StudioClipsStation's clipSchema) so the frozen contract module stays untouched.
export const studioPresetSchema = z
  .object({
    id: z.string(),
    name: z.string(),
    description: z.string().nullable().optional(),
    capability: z.string(),
    width: z.number(),
    height: z.number(),
    fps: z.number(),
    vram_budget_gb: z.number().nullable().optional(),
    seed: z.number().nullable().optional(),
    prompt: z.string().nullable().optional(),
    negative: z.string().nullable().optional(),
    recommended: z.string().nullable().optional(),
    // Slice (a) / v2v: a restyle preset (capability "v2v") is a video TRANSFORM — it
    // needs a staged source clip to mean anything. Drives the station's restyle flow.
    requires_source: z.boolean().nullable().optional(),
    // Identity lock (capability "id_lock"): the preset needs ≥1 REFERENCE image (the
    // subject to lock). Exact sibling of requires_source — same consumption pattern;
    // the surface gates the id_lock reference-slot flow on it (and on the capability).
    requires_reference: z.boolean().nullable().optional(),
    // UI-honesty badge: a note about the prompt for this preset. Set on SYNTHETIC
    // previews to state the prompt is recorded but NOT rendered (frames = seed +
    // geometry), so an evocative scaffold never misleads. "" / absent = no badge.
    prompt_note: z.string().nullable().optional(),
  })
  .passthrough();
export type StudioPreset = z.infer<typeof studioPresetSchema>;

// GET /video/studio/presets returns a {presets:[…]} envelope (matches /video/presets).
const studioPresetsResponseSchema = z.object({ presets: z.array(studioPresetSchema) });

export interface StudioPresetsState {
  presets: StudioPreset[];
  loading: boolean;
  error: string | null;
  /** Force an immediate re-fetch of the studio-preset list. */
  refresh: () => void;
}

/**
 * Load the curated studio clip presets. `active` gates the network call; while
 * inactive the hook reports loading:false / an empty list (nothing to show).
 */
export function useStudioPresets(active: boolean): StudioPresetsState {
  const [state, setState] = useState<Omit<StudioPresetsState, "refresh">>({
    presets: [],
    loading: false,
    error: null,
  });

  const mounted = useRef(true);
  const inFlight = useRef(false);
  const loadedRef = useRef(false);
  const ctrlRef = useRef<AbortController | null>(null);

  const load = useCallback(async (background: boolean) => {
    if (inFlight.current) return;
    inFlight.current = true;
    if (!background) setState((s) => ({ ...s, loading: true }));
    const signal = ctrlRef.current?.signal;
    try {
      const res = await request<unknown>(hugpyConfig.studioPresetsUrl, {
        signal,
        meta: { specKey: "studio", operation: "studio.presets.list" },
      });
      if (!mounted.current) return;

      if (!res.ok) {
        // A background refresh that fails must not wipe a working picker.
        if (background) return;
        setState({ presets: [], loading: false, error: describeAppError(errorOf(res)) });
        return;
      }
      const parsed = studioPresetsResponseSchema.safeParse(okValue(res));
      if (!parsed.success) {
        if (background) return;
        setState({ presets: [], loading: false, error: "Malformed studio presets response." });
        return;
      }
      loadedRef.current = true;
      setState({ presets: parsed.data.presets, loading: false, error: null });
    } finally {
      inFlight.current = false;
    }
  }, []);

  // Load once, the first time the station becomes active. Stays cached after.
  useEffect(() => {
    mounted.current = true;
    if (!active || loadedRef.current) return;
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    void load(false);
    return () => {
      mounted.current = false;
      ctrl.abort();
    };
  }, [active, load]);

  const refresh = useCallback(() => {
    void load(loadedRef.current);
  }, [load]);

  return { ...state, refresh };
}
