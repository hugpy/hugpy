// The shared image-model dropdown source. Fetches the model registry and the task
// defaults, filters to the GENERATION image tasks (text-to-image + image-to-image)
// off the FULL `tasks` capability list, and exposes the options + the default id +
// loading/error as data (never throws). Both the Frames station (config surface)
// and the later Generate station import this verbatim.
//
// Staleness fix: the registry is no longer fetched only once on mount. Models
// adopted MID-SESSION (e.g. ae ComfyUI checkpoints that appear once ComfyUI comes
// up) must enter the picker WITHOUT a full page reload, so the hook now
// stale-while-revalidates — it re-fetches on window focus / tab visibility, on a
// gentle background interval (skipped while the tab is hidden), and on demand via
// the exposed `refresh()`. Background revalidations never blank a working picker:
// `loading` flips only for the very first load, and a failed poll keeps the last
// good list. The public surface gained `refresh` (additive — existing consumers
// that destructure the four data fields are unaffected).
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";

export interface ModelOption {
  /** The model id sent to the backend. */
  id: string;
  /** Human label for the dropdown (hub_id when present, else the id). */
  label: string;
  /**
   * The model's FULL generation-capability list (from `capabilitiesOf`) — carried
   * through so the Generate station can drive REQUIRED inputs from it (see
   * `requiredInputsFor`). Previously discarded after the label was built.
   */
  tasks: string[];
  /**
   * Present but NOT pickable — a LoRA adapter or a pipeline component (an
   * encoder/VAE split). Rendered greyed with `reason` rather than offered
   * (it would refuse) or hidden (the operator then can't see the file they
   * downloaded). See the k61 incident note on GENERATION_IMAGE_TASKS.
   */
  disabled?: boolean;
  /** Why it can't be picked — shown as the option's title. */
  reason?: string;
}

export interface ImageModelsState {
  models: ModelOption[];
  defaultId: string | null;
  loading: boolean;
  error: string | null;
  /**
   * Force an immediate re-fetch of the registry (manual refresh affordance).
   * Runs as a background revalidation — it does not clear the current list.
   */
  refresh: () => void;
}

// Tolerant schemas — the registry carries many fields we ignore; passthrough keeps
// them and we only assert the three we read.
const modelEntrySchema = z
  .object({
    id: z.string(),
    task: z.string().nullable().optional(),
    // Full capability list (0.1.122+): `task` is only the PRIMARY — filtering
    // on it alone hid secondary capabilities (dual t2i+img2img models).
    tasks: z.array(z.string()).nullable().optional(),
    hub_id: z.string().nullable().optional(),
    // Why a present model still can't serve (0.1.x+, k61): adapters and
    // pipeline components are real files that are not servable on their own.
    adapter: z.boolean().nullable().optional(),
    serveable: z.boolean().nullable().optional(),
    unserveable_reason: z.string().nullable().optional(),
  })
  .passthrough();

const modelsResponseSchema = z
  .object({ data: z.array(modelEntrySchema) })
  .passthrough();

const promptTasksSchema = z
  .object({
    defaults: z.record(z.string()).nullable().optional(),
  })
  .passthrough();

const IMAGE_TASK = "text-to-image";
// Since the img2img go-live (map §12.5 follow-up), image-to-image models
// (Qwen-Image-Edit, flux-klein) are first-class generate targets — but they
// REQUIRE a start image (no text-only path), so their labels say so.
const IMG2IMG_TASK = "image-to-image";
// The full set of GENERATION / EDITING image tasks this workspace can drive: t2i
// produces from text, img2img edits a start image. Both stations consume exactly
// these. Analysis-only image tasks (classification, captioning, detection, …) are
// NOT generate targets, so they are deliberately excluded — a model must advertise
// one of these in its capability list to enter the picker.
const GENERATION_IMAGE_TASKS: readonly string[] = [IMAGE_TASK, IMG2IMG_TASK];

// Rows that are NOT generation targets but must still be explicable in an image
// picker (k61, 2026-07-31). The flux2 incident put a text-encoder COMPONENT and
// an image LoRA in front of the operator as if they were models: the encoder was
// selectable in image UIs where it can never serve, and the LoRA — left
// unclassified and defaulted to text-generation — refused with
// "supported: ['text-generation']". They now enter the list GREYED, carrying the
// backend's own reason, so the affordance says what the file is instead of
// pretending it can generate.
const ADAPTER_TASK = "adapter";
const COMPONENT_TASK = "pipeline-component";
const NEEDS_CLASSIFICATION_TASK = "needs-classification";
const EXPLAINED_NON_TARGETS: readonly string[] = [
  ADAPTER_TASK,
  COMPONENT_TASK,
  NEEDS_CLASSIFICATION_TASK,
];

function nonTargetReason(m: z.infer<typeof modelEntrySchema>): string {
  if (m.unserveable_reason) return m.unserveable_reason;
  const caps = capabilitiesOf(m);
  if (m.adapter || caps.includes(ADAPTER_TASK))
    return "LoRA adapter — apply it to a base image model; it can't generate on its own.";
  if (caps.includes(COMPONENT_TASK))
    return "Pipeline component (encoder/VAE) — served through its parent pipeline, not on its own.";
  return "Unclassified — no task derived from this model's files yet.";
}

// Gentle background revalidation cadence (ms). Cheap same-origin GET; only ONE
// picker instance is mounted at a time (studio sub-tabs share one WorkbenchStation),
// and polls are skipped while the tab is hidden.
const REFRESH_INTERVAL_MS = 45_000;

// Capability = the FULL `tasks` list; the single `task` (primary) is only the
// fallback for older payloads. Exported so task-specific pickers (e.g. the Movie
// director's judge-VLM select) can filter the registry the same way.
export function capabilitiesOf(m: z.infer<typeof modelEntrySchema>): string[] {
  return m.tasks?.length ? m.tasks : m.task ? [m.task] : [];
}

/**
 * The REQUIRED-input contract for a model, derived purely from its capability
 * `tasks[]`. The Generate station drives its explicit prompt/start-image controls
 * off this instead of leaving the user to hand-add parts. Truth table:
 *
 *   tasks includes…            → image
 *   ─────────────────────────────────────────
 *   text-to-image only         → "none"      (t2i can't take a start image)
 *   text-to-image + img2img    → "optional"  (start image ⇒ img2img edit)
 *   image-to-image only        → "required"  (edit model — no text-only path)
 *   empty / unrecognized       → "optional"  (safe default: text + optional image)
 *
 * `text` is always "required" — every generation path here needs a prompt.
 */
export function requiredInputsFor(
  tasks: string[],
): { text: "required"; image: "required" | "optional" | "none" } {
  const t2i = tasks.includes(IMAGE_TASK);
  const i2i = tasks.includes(IMG2IMG_TASK);
  let image: "required" | "optional" | "none";
  if (i2i && !t2i) image = "required"; // edit-only model
  else if (t2i && i2i) image = "optional"; // dual t2i + img2img
  else if (t2i && !i2i) image = "none"; // pure text-to-image
  else image = "optional"; // empty / unknown capabilities → safe default
  return { text: "required", image };
}

export function useImageModels(): ImageModelsState {
  const [state, setState] = useState<Omit<ImageModelsState, "refresh">>({
    models: [],
    defaultId: null,
    loading: true,
    error: null,
  });

  // `mounted` gates setState after unmount; `inFlight` collapses overlapping
  // revalidations (interval + focus + manual) into one; `ctrlRef` lets every
  // trigger — including a manual refresh — abort on unmount.
  const mounted = useRef(true);
  const inFlight = useRef(false);
  const ctrlRef = useRef<AbortController | null>(null);

  const load = useCallback(async (background: boolean) => {
    if (inFlight.current) return;
    inFlight.current = true;
    const signal = ctrlRef.current?.signal;
    try {
      const [mRes, tRes] = await Promise.all([
        request<unknown>(hugpyConfig.modelsUrl, {
          signal,
          meta: { specKey: "frames", operation: "models.list" },
        }),
        request<unknown>(hugpyConfig.promptTasksUrl, {
          signal,
          meta: { specKey: "frames", operation: "prompt.tasks" },
        }),
      ]);
      if (!mounted.current) return;

      if (!mRes.ok) {
        // A background revalidation that fails must NOT wipe a working picker —
        // keep the last good list; only the first (foreground) load surfaces it.
        if (background) return;
        setState({
          models: [],
          defaultId: null,
          loading: false,
          error: describeAppError(errorOf(mRes)),
        });
        return;
      }
      const mParsed = modelsResponseSchema.safeParse(okValue(mRes));
      if (!mParsed.success) {
        if (background) return;
        setState({
          models: [],
          defaultId: null,
          loading: false,
          error: "Malformed models response.",
        });
        return;
      }

      // Only a model that ADVERTISES a generation image task is pickable. Adapters,
      // pipeline components and unclassified rows are appended after them, greyed,
      // so they are explained rather than offered or hidden.
      const rows = mParsed.data.data.filter((m) => {
        const caps = capabilitiesOf(m);
        return (
          caps.some((t) => GENERATION_IMAGE_TASKS.includes(t)) ||
          m.adapter === true ||
          caps.some((t) => EXPLAINED_NON_TARGETS.includes(t))
        );
      });
      const models: ModelOption[] = rows
        .map((m) => {
          const caps = capabilitiesOf(m);
          const t2i = caps.includes(IMAGE_TASK);
          const i2i = caps.includes(IMG2IMG_TASK);
          if (!t2i && !i2i) {
            const reason = nonTargetReason(m);
            return {
              id: m.id,
              label: `${m.hub_id ?? m.id}  · not selectable`,
              tasks: caps,
              disabled: true,
              reason,
            };
          }
          const suffix = i2i && !t2i
            ? "  · img2img (needs a start image)"
            : i2i && t2i
              ? "  · t2i + img2img"
              : "";
          // Carry the full capability list through — the Generate station reads it
          // via `requiredInputsFor(...)` to drive its required prompt/start-image UI.
          return { id: m.id, label: (m.hub_id ?? m.id) + suffix, tasks: caps };
        })
        .sort((a, b) => Number(a.disabled ?? false) - Number(b.disabled ?? false));

      // Task defaults are best-effort — a failure there still yields a usable list.
      let defaultId: string | null = null;
      if (tRes.ok) {
        const tParsed = promptTasksSchema.safeParse(okValue(tRes));
        if (tParsed.success) {
          const d = tParsed.data.defaults?.[IMAGE_TASK];
          if (typeof d === "string") defaultId = d;
        }
      }
      // Guarantee a SELECTABLE default: fall back to the first pickable option
      // when the task default is absent, greyed, or not in the list at all.
      const pickable = models.filter((m) => !m.disabled);
      if (
        (!defaultId || !pickable.some((m) => m.id === defaultId)) &&
        pickable.length > 0
      ) {
        defaultId = pickable[0].id;
      }

      setState({ models, defaultId, loading: false, error: null });
    } finally {
      inFlight.current = false;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;

    // Foreground load (the only one that toggles `loading` / surfaces an error).
    void load(false);

    // Revalidate when the tab/window regains focus or becomes visible — a
    // returning operator sees models adopted while they were away, no reload.
    const revalidate = () => {
      if (document.visibilityState === "hidden") return;
      void load(true);
    };
    window.addEventListener("focus", revalidate);
    document.addEventListener("visibilitychange", revalidate);

    // Gentle interval backstop for the same-tab case (ComfyUI comes up while the
    // operator is looking at the station). Skipped while the tab is hidden.
    const timer = setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void load(true);
    }, REFRESH_INTERVAL_MS);

    return () => {
      mounted.current = false;
      ctrl.abort();
      window.removeEventListener("focus", revalidate);
      document.removeEventListener("visibilitychange", revalidate);
      clearInterval(timer);
    };
  }, [load]);

  const refresh = useCallback(() => {
    void load(true);
  }, [load]);

  return { ...state, refresh };
}

export interface TaskModelsState {
  models: ModelOption[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

/**
 * A registry list filtered to models whose capability `tasks[]` include ONE given
 * task — the generic sibling of useImageModels for non-generation pickers. The
 * Movie director's judge-model select uses it with "image-text-to-text" to surface
 * the vision-language models that can SCORE a segment. Same tolerant fetch +
 * stale-while-revalidate posture (focus / visibility / interval / manual refresh);
 * a failed background revalidation keeps the last good list. No task-defaults fetch
 * (there is no canonical default for an arbitrary task) — the caller supplies its
 * own default and keeps it selectable even when the list is empty.
 */
export function useModelsByTask(task: string): TaskModelsState {
  const [state, setState] = useState<Omit<TaskModelsState, "refresh">>({
    models: [],
    loading: true,
    error: null,
  });

  const mounted = useRef(true);
  const inFlight = useRef(false);
  const ctrlRef = useRef<AbortController | null>(null);

  const load = useCallback(
    async (background: boolean) => {
      if (inFlight.current) return;
      inFlight.current = true;
      const signal = ctrlRef.current?.signal;
      try {
        const mRes = await request<unknown>(hugpyConfig.modelsUrl, {
          signal,
          meta: { specKey: "generate", operation: "models.list" },
        });
        if (!mounted.current) return;
        if (!mRes.ok) {
          if (background) return;
          setState({ models: [], loading: false, error: describeAppError(errorOf(mRes)) });
          return;
        }
        const mParsed = modelsResponseSchema.safeParse(okValue(mRes));
        if (!mParsed.success) {
          if (background) return;
          setState({ models: [], loading: false, error: "Malformed models response." });
          return;
        }
        const models: ModelOption[] = mParsed.data.data
          .filter((m) => capabilitiesOf(m).includes(task))
          .map((m) => ({
            id: m.id,
            label: m.hub_id ?? m.id,
            tasks: capabilitiesOf(m),
          }));
        setState({ models, loading: false, error: null });
      } finally {
        inFlight.current = false;
      }
    },
    [task],
  );

  useEffect(() => {
    mounted.current = true;
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    void load(false);
    const revalidate = () => {
      if (document.visibilityState === "hidden") return;
      void load(true);
    };
    window.addEventListener("focus", revalidate);
    document.addEventListener("visibilitychange", revalidate);
    const timer = setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void load(true);
    }, REFRESH_INTERVAL_MS);
    return () => {
      mounted.current = false;
      ctrl.abort();
      window.removeEventListener("focus", revalidate);
      document.removeEventListener("visibilitychange", revalidate);
      clearInterval(timer);
    };
  }, [load]);

  const refresh = useCallback(() => {
    void load(true);
  }, [load]);

  return { ...state, refresh };
}
