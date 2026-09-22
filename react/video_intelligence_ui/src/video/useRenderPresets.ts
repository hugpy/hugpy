// GET /video/render/presets — WHAT THIS FLEET CAN ACTUALLY RENDER TODAY, read by the
// console instead of guessed at.
//
// This is the fourth and only MEASURED preset surface (video_intel/studio/presets.py:
// the eight ratified `RenderPreset` rows). The other three answer "what did we
// curate?"; this one answers "what will come back as pixels?" — and until 2026-07-29
// NOTHING in this SPA read it. The studio instead carried hand-maintained mirrors of
// backend facts (a static model-id list, a hand-curated capability union, a
// `STUDIO_REAL_FLOOR_GB = 6` that had drifted below every real row in the table), so a
// user could compose a request the server was always going to refuse and only find out
// at render time.
//
// WHAT THE PAYLOAD CARRIES, and why each half matters to a compatibility-aware UI:
//   • `presets[]`  — per row: capability, model, precision, geometry (width/height/fps),
//                    frame budget, `inputs[]` (what the CALLER must supply), `proven`
//                    (this exact path has produced pixels on this fleet), and the
//                    placement numbers (`vram_envelope_gb` = the minimum budget that can
//                    bind this model; `vram_need_gib` / `fits_render_box` = provenance).
//   • `unavailable[]` — every capability the enum declares that NO preset covers, with
//                    the measured blocker and the full refusal text, straight from the
//                    registry's own `capability_verdict`. A picker can therefore show
//                    the dead options DISABLED with the real reason instead of hiding
//                    the fleet's shape from the user.
//
// COURTESY LAYER, NEVER A GATE. The server-side refusal (POST /video/studio/i2v's
// capability gate) stays authoritative — everything derived from this hook only decides
// what to OFFER, never what to allow. If this fetch fails the surfaces fall back to
// their prior hand-maintained lists and the server still refuses what it always did.
//
// SESSION-CACHED. The table is frozen data derived at request time from a registry that
// only changes on deploy, so one fetch per page load is enough; the cache is a module
// singleton shared by every mount (the Clip surface and the Movie composer both read
// it) rather than a fetch per component.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";

// Tolerant/passthrough, like every other wire schema in this tree: assert only what we
// read, so a payload that grows a field never breaks the parse.
export const renderPresetSchema = z
  .object({
    id: z.string(),
    title: z.string(),
    description: z.string().nullable().optional(),
    /** The PRIMARY capability — what a caller asks for to reach this preset. */
    capability: z.string(),
    /** The full set (every ratified row serves exactly one today). */
    capabilities: z.array(z.string()).nullable().optional(),
    model: z.string(),
    framework: z.string().nullable().optional(),
    task: z.string().nullable().optional(),
    precision: z.string().nullable().optional(),
    /** Human geometry string; "source" on the two ffmpeg enhance rows. */
    geometry: z.string().nullable().optional(),
    // null on the enhance rows — they inherit the source clip's geometry.
    width: z.number().nullable().optional(),
    height: z.number().nullable().optional(),
    fps: z.number().nullable().optional(),
    default_frames: z.number().nullable().optional(),
    max_frames: z.number().nullable().optional(),
    /** What the CALLER must supply — the compatibility spine of this whole module. */
    inputs: z.array(z.string()).nullable().optional(),
    /** Checked claim: this exact path has produced pixels on this fleet. */
    proven: z.boolean().nullable().optional(),
    evidence: z.string().nullable().optional(),
    /** MOVIE ONLY: the preset ids a multi-segment render composes + its joint modes. */
    composes: z.array(z.string()).nullable().optional(),
    joints: z.array(z.string()).nullable().optional(),
    // Placement (added backend-side 2026-07-29). vram_envelope_gb is the MINIMUM
    // vram_budget_gb that can bind this row's model; the other two are provenance.
    vram_envelope_gb: z.number().nullable().optional(),
    vram_need_gib: z.number().nullable().optional(),
    fits_render_box: z.boolean().nullable().optional(),
  })
  .passthrough();
export type RenderPreset = z.infer<typeof renderPresetSchema>;

export const unavailableCapabilitySchema = z
  .object({
    capability: z.string(),
    /** The measured blocker, one clause — good tooltip material. */
    reason: z.string().nullable().optional(),
    /** The full user-facing refusal, including the menu of what IS available. */
    refusal: z.string().nullable().optional(),
  })
  .passthrough();
export type UnavailableCapability = z.infer<typeof unavailableCapabilitySchema>;

const renderPresetsResponseSchema = z
  .object({
    presets: z.array(renderPresetSchema),
    unavailable: z.array(unavailableCapabilitySchema).nullable().optional(),
    menu: z.string().nullable().optional(),
    /** Wan requires num_frames == 4k+1; the runner snaps down to the nearest. */
    frame_cadence: z.number().nullable().optional(),
    render_box: z.string().nullable().optional(),
    render_box_vram_gib: z.number().nullable().optional(),
  })
  .passthrough();
export type RenderPresetsPayload = z.infer<typeof renderPresetsResponseSchema>;

export interface RenderPresetsState {
  presets: RenderPreset[];
  unavailable: UnavailableCapability[];
  menu: string;
  frameCadence: number | null;
  renderBox: string;
  renderBoxVramGib: number | null;
  loading: boolean;
  error: string | null;
  /** True once a fetch has SUCCEEDED — the "is it safe to filter by this?" flag.
   *  Every consumer must fall back to its unfiltered list while this is false, so a
   *  slow or failed discovery call never presents an empty picker. */
  loaded: boolean;
  refresh: () => void;
}

// ── module-singleton session cache ──────────────────────────────────────────
// One fetch per page load, shared by every mount. `inFlight` coalesces concurrent
// first-mounts (the Clip surface and the Movie composer mount together) into one call.
let cached: RenderPresetsPayload | null = null;
let inFlight: Promise<void> | null = null;
const subscribers = new Set<() => void>();

function notify(): void {
  for (const fn of subscribers) fn();
}

async function fetchOnce(force: boolean): Promise<string | null> {
  if (cached && !force) return null;
  if (inFlight && !force) {
    await inFlight;
    return null;
  }
  let err: string | null = null;
  const p = (async () => {
    const res = await request<unknown>(hugpyConfig.renderPresetsUrl, {
      meta: { specKey: "studio", operation: "render.presets.list" },
    });
    if (!res.ok) {
      err = describeAppError(errorOf(res));
      return;
    }
    const parsed = renderPresetsResponseSchema.safeParse(okValue(res));
    if (!parsed.success) {
      err = "Malformed render-presets response.";
      return;
    }
    cached = parsed.data;
  })();
  inFlight = p;
  try {
    await p;
  } finally {
    inFlight = null;
    notify();
  }
  return err;
}

/** TEST/DEV seam: drop the session cache so the next mount re-fetches. */
export function resetRenderPresetsCache(): void {
  cached = null;
  inFlight = null;
}

/**
 * The fleet's measured render capability, as data. Never throws; a failed fetch
 * reports `loaded:false` + an error string and every consumer keeps its fallback.
 *
 * `active` gates the network call the same way useStudioPresets/useMoviePresets do,
 * so a surface that is not mounted costs nothing.
 */
export function useRenderPresets(active = true): RenderPresetsState {
  const [, bump] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    const rerender = () => {
      if (mounted.current) bump((n) => n + 1);
    };
    subscribers.add(rerender);
    return () => {
      mounted.current = false;
      subscribers.delete(rerender);
    };
  }, []);

  const run = useCallback(async (force: boolean) => {
    if (cached && !force) return;
    setLoading(true);
    const err = await fetchOnce(force);
    if (!mounted.current) return;
    setLoading(false);
    setError(err);
  }, []);

  useEffect(() => {
    if (!active) return;
    void run(false);
  }, [active, run]);

  const refresh = useCallback(() => void run(true), [run]);

  return {
    presets: cached?.presets ?? [],
    unavailable: cached?.unavailable ?? [],
    menu: cached?.menu ?? "",
    frameCadence: cached?.frame_cadence ?? null,
    renderBox: cached?.render_box ?? "",
    renderBoxVramGib: cached?.render_box_vram_gib ?? null,
    loading,
    error,
    loaded: cached != null,
    refresh,
  };
}
