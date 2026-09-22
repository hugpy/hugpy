// The Movie-tab TEMPLATE dropdown source. Fetches the curated movie templates
// (GET /movie/presets), validates them with the tolerant moviePresetSchema, and
// exposes the list + loading/error as data (never throws) — the same
// data-not-throws contract as usePresets.ts, which the Generate station imports
// alongside it.
//
// Two deliberate differences from usePresets:
//   1. The wire is a BARE ARRAY of presets (not the {presets:[…]} envelope), each
//      item carrying its whole `goals` shot list — see moviePresetSchema.
//   2. The fetch is GATED behind an `active` flag: the Movie tab is one of three
//      Generate sub-modes, so the list only loads once the movie tab is entered
//      (and stays cached after). Passing active=false keeps it idle.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import { moviePresetSchema, type MoviePreset } from "./contract";

export interface MoviePresetsState {
  presets: MoviePreset[];
  loading: boolean;
  error: string | null;
  /** Force an immediate re-fetch of the movie-template list. */
  refresh: () => void;
}

// GET /movie/presets returns a {presets:[…]} envelope (matches /video/presets).
// Tolerant parse of the envelope; each item is a tolerant moviePresetSchema.
const moviePresetsResponseSchema = z.object({ presets: z.array(moviePresetSchema) });

/**
 * Load the curated movie templates. `active` gates the network call: pass
 * `mode === "movie"` so the list only fetches once the Movie tab is entered.
 * While inactive the hook reports loading:false / an empty list (nothing to show).
 */
export function useMoviePresets(active: boolean): MoviePresetsState {
  const [state, setState] = useState<Omit<MoviePresetsState, "refresh">>({
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
      const res = await request<unknown>(hugpyConfig.moviePresetsUrl, {
        signal,
        meta: { specKey: "generate", operation: "movie.presets.list" },
      });
      if (!mounted.current) return;

      if (!res.ok) {
        // A background refresh that fails must not wipe a working picker.
        if (background) return;
        setState({ presets: [], loading: false, error: describeAppError(errorOf(res)) });
        return;
      }
      const parsed = moviePresetsResponseSchema.safeParse(okValue(res));
      if (!parsed.success) {
        if (background) return;
        setState({ presets: [], loading: false, error: "Malformed movie templates response." });
        return;
      }
      loadedRef.current = true;
      setState({ presets: parsed.data.presets, loading: false, error: null });
    } finally {
      inFlight.current = false;
    }
  }, []);

  // Load once, the first time the movie tab becomes active. Stays cached after.
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
