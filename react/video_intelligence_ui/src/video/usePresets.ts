// The Generate-station preset dropdown source. Fetches the curated "ideal default
// loads" (GET /video/presets), validates them with a tolerant passthrough schema,
// and exposes the options + loading/error as data (never throws) — the exact same
// data-not-throws contract as useModels.ts, which the station imports alongside it.
//
// Presets are curated server-side and rarely change mid-session (unlike the model
// registry, which grows as ComfyUI checkpoints get adopted), so this hook does a
// single foreground load and a manual `refresh()` — it deliberately skips the
// focus/visibility/interval revalidation machinery useModels.ts needs.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";

// The knob defaults a preset prefills onto the Generate station. All optional +
// nullable: a preset may omit any field (the station keeps its current value), and
// the wire may carry extras we ignore (passthrough). Shared by the apply echo.
export interface PresetDefaults {
  strength?: number | null;
  steps?: number | null;
  guidance?: number | null;
  width?: number | null;
  height?: number | null;
  n_frames?: number | null;
  fps?: number | null;
  negative?: string | null;
}

export interface Preset {
  /** Opaque id — the path segment of the apply POST. */
  id: string;
  /** Human label for the dropdown + status line. */
  name: string;
  /** One-line description surfaced under the picker when selected. */
  description?: string | null;
  /** Generation sub-mode this preset targets ("text-to-image" | "edit-chain" | "img2img" | …). */
  mode: string;
  /** The model the station selects + the apply POST pre-warms. */
  model_key: string;
  /** Knob values to prefill. */
  defaults: PresetDefaults;
  /** Advisory placement hint (e.g. "gpu"). */
  recommended?: string | null;
}

export interface PresetsState {
  presets: Preset[];
  loading: boolean;
  error: string | null;
  /** Force an immediate re-fetch of the preset list. */
  refresh: () => void;
}

// Tolerant schemas — passthrough keeps unknown fields; we only assert what we read.
export const presetDefaultsSchema = z
  .object({
    strength: z.number().nullable().optional(),
    steps: z.number().nullable().optional(),
    guidance: z.number().nullable().optional(),
    width: z.number().nullable().optional(),
    height: z.number().nullable().optional(),
    n_frames: z.number().nullable().optional(),
    fps: z.number().nullable().optional(),
    negative: z.string().nullable().optional(),
  })
  .passthrough();

export const presetSchema = z
  .object({
    id: z.string(),
    name: z.string(),
    description: z.string().nullable().optional(),
    mode: z.string(),
    model_key: z.string(),
    defaults: presetDefaultsSchema,
    recommended: z.string().nullable().optional(),
  })
  .passthrough();

const presetsResponseSchema = z
  .object({ presets: z.array(presetSchema) })
  .passthrough();

// The per-id apply response — a discriminated pair sharing `ok`. On success it
// echoes the (possibly server-adjusted) model_key/mode/defaults the station honors
// as authoritative; on failure it carries {error:{code,message}}. Tolerant so a
// partial worker block or extra fields never fail the parse.
export const presetApplyResponseSchema = z
  .object({
    ok: z.boolean(),
    worker: z
      .object({ name: z.string().nullable().optional(), id: z.string().nullable().optional() })
      .passthrough()
      .nullable()
      .optional(),
    model_key: z.string().nullable().optional(),
    mode: z.string().nullable().optional(),
    defaults: presetDefaultsSchema.nullable().optional(),
    warming: z.boolean().nullable().optional(),
    error: z
      .object({
        code: z.string().nullable().optional(),
        message: z.string().nullable().optional(),
      })
      .passthrough()
      .nullable()
      .optional(),
  })
  .passthrough();

export function usePresets(): PresetsState {
  const [state, setState] = useState<Omit<PresetsState, "refresh">>({
    presets: [],
    loading: true,
    error: null,
  });

  const mounted = useRef(true);
  const inFlight = useRef(false);
  const ctrlRef = useRef<AbortController | null>(null);

  const load = useCallback(async (background: boolean) => {
    if (inFlight.current) return;
    inFlight.current = true;
    const signal = ctrlRef.current?.signal;
    try {
      const res = await request<unknown>(hugpyConfig.presetsUrl, {
        signal,
        meta: { specKey: "generate", operation: "presets.list" },
      });
      if (!mounted.current) return;

      if (!res.ok) {
        // A background refresh that fails must not wipe a working picker.
        if (background) return;
        setState({ presets: [], loading: false, error: describeAppError(errorOf(res)) });
        return;
      }
      const parsed = presetsResponseSchema.safeParse(okValue(res));
      if (!parsed.success) {
        if (background) return;
        setState({ presets: [], loading: false, error: "Malformed presets response." });
        return;
      }
      setState({ presets: parsed.data.presets, loading: false, error: null });
    } finally {
      inFlight.current = false;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    void load(false);
    return () => {
      mounted.current = false;
      ctrl.abort();
    };
  }, [load]);

  const refresh = useCallback(() => {
    void load(true);
  }, [load]);

  return { ...state, refresh };
}
