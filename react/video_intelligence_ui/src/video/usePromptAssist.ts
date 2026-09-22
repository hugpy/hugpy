// Shared LLM PROMPT-ASSIST hook — the Enhance / Generate transport + busy/error
// model, extracted from GenerateStation so the Generate, Studio Clip, and Studio
// Movie composers all POST the SAME promptAssistUrl with ONE in-flight call at a
// time and identical error semantics.
//
// Presentational-by-callback, like GoalComposer: this hook owns NO editor state.
// Each host passes the current draft when it fires a call and an `apply` callback
// that drops the returned prompt wherever that host keeps its text (replace the
// primary text part / set the prompt string / patch a goal row). A failed call
// NEVER reaches `apply`, so a bad response can't clobber the existing prompt — it
// surfaces on `assistError` instead. The paired <PromptAssistButtons> renders the
// button cluster; keeping the two beside each other keeps adoption a small,
// non-destabilizing flip per composer.
import { useCallback, useEffect, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import {
  readIntentResult,
  readSpreadResult,
  type AssistContextWire,
  type IntentResult,
  type SpreadRequestWire,
  type SpreadResult,
} from "./spreadAssist";

/** The two assist actions: "detail" = Enhance (enrich a draft), "generate" = write fresh. */
export type AssistMode = "detail" | "generate";

/**
 * Every mode that can hold the ONE in-flight assist slot. The two whole-request
 * modes added by STUDIO-SPREAD-SPEC ("spread" = one call rewriting N selected rows,
 * "negative" = an exclusion list rather than prose) are NOT `AssistMode`, because
 * they take no `draft` and do not return a prompt for a single field — but they DO
 * share the concurrency gate and the error model, so they share the busy value.
 * Widening only the BUSY type keeps `runAssist`'s signature honest.
 */
export type AssistBusyMode = AssistMode | "spread" | "negative";

/** The active sub-mode, sent as `context.kind` so video modes get motion/camera phrasing. */
export type AssistKind = "image" | "scene" | "movie";

/** The `specKey` a host's telemetry rides under (matches the host's other calls). */
type SpecKey = "generate" | "studio";

export interface UsePromptAssistOptions {
  /** Sub-mode sent as `context.kind` (image | scene | movie). */
  kind: AssistKind;
  /** Telemetry spec key for the request meta (the host's own key). */
  specKey: SpecKey;
}

/** One selectable text generator, as offered by the backend's discovery route. */
export interface AssistModel {
  /** Catalog key sent back as `body.model`. */
  model: string;
  /** Seated on a worker right now (answers in ~1s) vs present but cold (first call loads it). */
  serving: boolean;
  /** Canonical residency (STATE-MODEL.md #4): serving|loaded = ready; hot = on a
   *  worker drive (pays a t_load); cold = central only (pays a download + load). */
  state?: "serving" | "loaded" | "hot" | "cold";
  framework?: string | null;
  /** True for the fleet default — used when the user has never chosen. */
  default?: boolean;
}

/**
 * The per-call typed extras (SPEC §1c). Purely ADDITIVE: a host that passes nothing
 * sends the byte-identical body it sent before, which is what keeps GenerateStation
 * and StudioGenerateTab compiling and behaving unchanged.
 */
export interface AssistExtras {
  /**
   * Typed row/identity state merged into `context` beside `kind`. This is the
   * carrier for structured state — `hint` stays free-form user text (§1c ruling).
   */
  context?: Omit<AssistContextWire, "kind">;
  /**
   * Per-CALL model override. When set, this generator writes THIS call's prompt
   * regardless of the hook-wide `assistModel` — the carrier for per-component
   * assist-model selection (each prompt component can pick its own). Omitted by
   * hosts that pass nothing, which then use the hook `assistModel` / fleet default.
   */
  model?: string | null;
}

export interface PromptAssistApi {
  /** The ACTIVE mode while a call is in flight, or null when idle. Blocks concurrency + drives labels. */
  assistBusy: AssistBusyMode | null;
  /** The dismissible inline notice; a failed call sets it and leaves the prompt intact. */
  assistError: string | null;
  /** Clear the inline error notice. */
  dismissError: () => void;
  /**
   * Fire an assist call. `draft` is the host's current prompt text (Enhance requires it;
   * Generate sends it as a loose theme). On success the returned prompt is handed to
   * `apply`; on ANY failure `apply` is never called and `assistError` is set instead.
   */
  runAssist: (
    mode: AssistMode,
    draft: string,
    apply: (prompt: string) => void,
    extra?: AssistExtras,
  ) => Promise<void>;
  /**
   * Generate a NEGATIVE prompt (mode "negative", §1b) — an artifact/quality exclusion
   * list, not prose. `subject` is what is being negated (the row's own prompt); a
   * failure never reaches `apply`, exactly like runAssist.
   */
  runNegative: (
    opts: { subject?: string; draft?: string; extra?: AssistExtras },
    apply: (negative: string) => void,
  ) => Promise<void>;
  /**
   * Run ONE coherent spread across the selected rows (mode "spread", §1a). Resolves
   * to the parsed result, or null when the call failed (the reason is on
   * `assistError`). NEVER N sequential per-row calls — that is the incoherence this
   * whole feature exists to delete.
   */
  runSpread: (body: SpreadRequestWire) => Promise<SpreadResult | null>;
  /**
   * Classify a field's text (§1d). Deliberately OUTSIDE the busy gate: this is called
   * on blur, must not be blocked by an in-flight generation, and must never surface as
   * an error — any failure reads as a degraded `ambiguous`, i.e. "show both actions".
   */
  classifyIntent: (text: string, scope?: "segment" | "movie") => Promise<IntentResult>;
  /** The text generators this fleet can actually run (empty until discovery returns). */
  assistModels: AssistModel[];
  /** The selected generator, or null to let the backend use the fleet default. */
  assistModel: string | null;
  /** Choose a generator; persisted so the choice survives a reload. */
  setAssistModel: (model: string | null) => void;
}

// The user's pick is a PREFERENCE, not app state: persist it in localStorage so it
// survives a reload, and share it across all three composers (Generate, Studio Clip,
// Studio Movie) — they each mount their own hook, so a module-level key is what makes
// the choice feel like one setting rather than three.
const ASSIST_MODEL_KEY = "hugpy.promptAssist.model";

function readStoredModel(): string | null {
  try {
    return window.localStorage.getItem(ASSIST_MODEL_KEY) || null;
  } catch {
    return null;                      // private mode / storage disabled — not fatal
  }
}

// Defensively pull `.prompt` (a non-empty string) out of the prompt-assist 200 body
// ({ prompt, model, kind }). Returns null for any malformed/empty shape so a bad
// response surfaces as a notice instead of clobbering the prompt with junk. (Lifted
// verbatim from GenerateStation's local helper so every host parses identically.)
function readAssistPrompt(value: unknown): string | null {
  if (value != null && typeof value === "object" && "prompt" in value) {
    const p = (value as { prompt?: unknown }).prompt;
    if (typeof p === "string" && p.trim() !== "") return p;
  }
  return null;
}

// Ticket flow (2026-08-28): a cold model load measured 6m12s on the live
// fleet, and a held browser connection dies long before that (ClientGone) —
// killing the load, which restarts on the next click, forever. POST with
// {async:true} returns {ticket} immediately and the generation keeps running
// server-side; we poll the ticket until the prompt lands. No connection has
// to survive the load. An old server without the ticket route answers the
// POST with the direct result, which passes through unchanged.
const TICKET_POLL_MS = 3_000;
const TICKET_CEILING_MS = 12 * 60_000;

async function assistCall(
  body: Record<string, unknown>,
  specKey: string | undefined,
  operation: string,
): Promise<ReturnType<typeof request<unknown>>> {
  const res = await request<unknown>(hugpyConfig.promptAssistUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, async: true }),
    timeoutMs: 30_000,
    meta: { specKey, operation },
  });
  if (!res.ok) return res;
  const value = okValue(res) as { ticket?: string; pending?: boolean } | null;
  const ticket = value?.ticket;
  if (!ticket) return res; // direct (non-ticket) result — serve as-is
  const deadline = Date.now() + TICKET_CEILING_MS;
  for (;;) {
    await new Promise((r) => setTimeout(r, TICKET_POLL_MS));
    const poll = await request<unknown>(
      `${hugpyConfig.promptAssistUrl}/ticket/${encodeURIComponent(ticket)}`,
      {
        method: "GET",
        timeoutMs: 15_000,
        meta: { specKey, operation: `${operation}.poll` },
      },
    );
    if (!poll.ok) return poll;
    const row = okValue(poll) as { pending?: boolean } | null;
    if (!row?.pending) return poll;
    if (Date.now() >= deadline) {
      return {
        ok: false,
        error: {
          kind: "timeout",
          message:
            "the assistant is still working after 12 minutes — the model may be stuck loading; try again",
          requestId: `assist-ticket-${ticket.slice(0, 8)}`,
        },
      } as Awaited<ReturnType<typeof request<unknown>>>;
    }
  }
}

export function usePromptAssist({ kind, specKey }: UsePromptAssistOptions): PromptAssistApi {
  // One in-flight call at a time: `assistBusy` holds the ACTIVE mode or null when idle —
  // it both blocks concurrent calls and drives which button shows its loading label.
  const [assistBusy, setAssistBusy] = useState<AssistBusyMode | null>(null);
  const [assistError, setAssistError] = useState<string | null>(null);
  const [assistModels, setAssistModels] = useState<AssistModel[]>([]);
  const [assistModel, setAssistModelState] = useState<string | null>(readStoredModel);

  // Discover the offerable generators once per mount. A failure is SILENT on purpose:
  // the picker simply stays empty and assist keeps working on the fleet default, so a
  // discovery hiccup can never block Enhance/Generate.
  useEffect(() => {
    let alive = true;
    (async () => {
      const res = await request<unknown>(hugpyConfig.promptAssistModelsUrl, {
        method: "GET",
        timeoutMs: 15_000,
        meta: { specKey, operation: "prompt.assist.models" },
      });
      if (!alive || !res.ok) return;
      const value = okValue(res);
      const rows =
        value != null && typeof value === "object" && Array.isArray((value as { models?: unknown }).models)
          ? ((value as { models: AssistModel[] }).models)
          : [];
      setAssistModels(rows);
      // Drop a stored pick the fleet no longer offers (model retired, worker gone) —
      // otherwise the UI shows a selection that would fail on use.
      setAssistModelState((current) =>
        current && !rows.some((r) => r.model === current) ? null : current,
      );
    })();
    return () => {
      alive = false;
    };
  }, [specKey]);

  const setAssistModel = useCallback((model: string | null) => {
    setAssistModelState(model);
    try {
      if (model) window.localStorage.setItem(ASSIST_MODEL_KEY, model);
      else window.localStorage.removeItem(ASSIST_MODEL_KEY);
    } catch {
      /* storage disabled — the choice still applies for this session */
    }
  }, []);

  const dismissError = useCallback(() => setAssistError(null), []);

  const runAssist = useCallback(
    async (
      mode: AssistMode,
      draft: string,
      apply: (prompt: string) => void,
      extra?: AssistExtras,
    ): Promise<void> => {
      if (assistBusy) return; // block concurrent calls
      const trimmed = draft.trim();
      if (mode === "detail" && trimmed === "") return; // Enhance needs a draft
      setAssistError(null);
      setAssistBusy(mode);

      // `model` is sent ONLY when the user has picked one; omitting it lets the backend
      // apply the fleet default (flux2-klein-9b-uncensored-text-encoder, served from
      // computron). The old comment here claimed "the backend picks the best resolvable
      // chat model" — it did not, it used one hardcoded key; the picker plus a resolved
      // default is what finally made that sentence true. The draft is omitted when blank
      // (Generate from a truly empty box) and context.kind carries the active sub-mode so
      // video modes get motion/camera phrasing.
      //
      // Thinking is suppressed server-side for EVERY generator (the assist route appends
      // a no-think directive and strips any <think>...</think> that survives), so picking
      // a reasoning model here cannot dump a monologue into the prompt box.
      //
      // TYPED CONTEXT (§1c): `extra.context` carries the STRUCTURED row state — this
      // segment, its neighbours, the locked identity — beside `kind`. The backend
      // renders joint modes into plain sentences and attaches the identity's
      // do-not-invent list, which is what stopped the generator making up wardrobes
      // for a bare character name. Omitted entirely by hosts that pass nothing, so
      // their request body is unchanged.
      const body: {
        mode: AssistMode;
        draft?: string;
        model?: string;
        context: AssistContextWire;
      } = {
        mode,
        ...(trimmed !== "" ? { draft: trimmed } : {}),
        ...((extra?.model ?? assistModel) ? { model: extra?.model ?? assistModel ?? undefined } : {}),
        context: { kind, ...(extra?.context ?? {}) },
      };

      // Ticket flow — see assistCall: the POST returns immediately and we poll,
      // so a cold model load (measured 6m12s) can finish without the browser
      // holding a connection through it.
      const res = await assistCall(body, specKey, `prompt.assist.${mode}`);
      if (!res.ok) {
        setAssistError(describeAppError(errorOf(res)));
        setAssistBusy(null);
        return;
      }
      const prompt = readAssistPrompt(okValue(res));
      if (prompt == null) {
        setAssistError("The assistant returned an empty prompt — try again.");
        setAssistBusy(null);
        return;
      }
      apply(prompt);
      setAssistBusy(null);
    },
    [assistBusy, kind, specKey, assistModel],
  );

  // ── mode "negative" (§1b) ────────────────────────────────────────────────
  // A negative prompt is an ARTIFACT/QUALITY EXCLUSION LIST, not prose — reusing the
  // scene system prompt returns a poem, which is why this is its own mode rather than
  // a `detail` call with different words. `prompt` and `negative` carry the same text
  // on the wire (no existing key moved); we read `prompt` via the shared reader.
  const runNegative = useCallback(
    async (
      opts: { subject?: string; draft?: string; extra?: AssistExtras },
      apply: (negative: string) => void,
    ): Promise<void> => {
      if (assistBusy) return;
      setAssistError(null);
      setAssistBusy("negative");
      const body: Record<string, unknown> = {
        mode: "negative",
        context: { kind, ...(opts.extra?.context ?? {}) },
      };
      if (opts.subject && opts.subject.trim()) body.subject = opts.subject.trim();
      if (opts.draft && opts.draft.trim()) body.draft = opts.draft.trim();
      const negModel = opts.extra?.model ?? assistModel;
      if (negModel) body.model = negModel;

      const res = await assistCall(body, specKey, "prompt.assist.negative");
      if (!res.ok) {
        setAssistError(describeAppError(errorOf(res)));
        setAssistBusy(null);
        return;
      }
      const text = readAssistPrompt(okValue(res));
      if (text == null) {
        setAssistError("The assistant returned an empty negative — try again.");
        setAssistBusy(null);
        return;
      }
      apply(text);
      setAssistBusy(null);
    },
    [assistBusy, kind, specKey, assistModel],
  );

  // ── mode "spread" (§1a) ──────────────────────────────────────────────────
  // ONE call for the whole selection. Returns the parsed result to the caller instead
  // of applying anything itself: only the host knows which editor rows the returned
  // segment_ids map to, and the "apply ONLY to rows that came back" rule is the whole
  // safety property. A 502 carries the raw reply — surfaced honestly by the shared
  // error mapper, never replaced with a fabricated segment.
  const runSpread = useCallback(
    async (body: SpreadRequestWire): Promise<SpreadResult | null> => {
      if (assistBusy) return null;
      setAssistError(null);
      setAssistBusy("spread");
      const payload = { ...body, ...(assistModel ? { model: assistModel } : {}) };
      // A spread writes N paragraphs in one generation, the longest assist call
      // there is — the ticket flow means a cold generator's load can't kill it.
      const res = await assistCall(payload, specKey, "prompt.assist.spread");
      if (!res.ok) {
        setAssistError(describeAppError(errorOf(res)));
        setAssistBusy(null);
        return null;
      }
      const parsed = readSpreadResult(okValue(res));
      setAssistBusy(null);
      if (parsed == null) {
        setAssistError("The assistant returned a malformed spread — nothing was changed.");
        return null;
      }
      if (parsed.segments.length === 0) {
        setAssistError("The assistant wrote no usable segments — nothing was changed.");
        return null;
      }
      return parsed;
    },
    [assistBusy, specKey, assistModel],
  );

  // ── the intent router (§1d) ──────────────────────────────────────────────
  // Outside the busy gate on purpose (see the API doc above) and outside the error
  // model too: this route ALWAYS returns 200, and a transport failure reads as a
  // degraded "ambiguous" — the UI then offers both actions, which is the correct
  // answer to "we don't know", not an error the user has to dismiss.
  const classifyIntent = useCallback(
    async (text: string, scope: "segment" | "movie" = "segment"): Promise<IntentResult> => {
      const res = await request<unknown>(hugpyConfig.promptIntentUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, scope }),
        timeoutMs: 30_000,
        meta: { specKey, operation: "prompt.intent" },
      });
      if (!res.ok) {
        return { intent: "ambiguous", operation: null, confidence: 0, cached: false, degraded: true };
      }
      return readIntentResult(okValue(res));
    },
    [specKey],
  );

  return {
    assistBusy,
    assistError,
    dismissError,
    runAssist,
    runNegative,
    runSpread,
    classifyIntent,
    assistModels,
    assistModel,
    setAssistModel,
  };
}
