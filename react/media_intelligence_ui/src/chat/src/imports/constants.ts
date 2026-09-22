import { hugpyConfig } from "../../../config";

// (legacy `DEFAULT_API_BASE` removed — base now comes from config.ts → /api.)

export type ChatModelOption = {
  key: string;
  label: string;
  shortLabel: string;
  maxNewTokens: number;
  defaultMaxNewTokens: number;
};

// ── token budgets ──────────────────────────────────────────────────────
// The /models entries don't carry a generation budget, only model_max_length
// (context window). We pick sane, uniform generation defaults so every model
// gets a usable slider without per-model tuning.
const DEFAULT_MAX_NEW_TOKENS = 65536;
const DEFAULT_DEFAULT_MAX_NEW_TOKENS = 8192;

// Tasks whose models can drive the chat composer. text-generation is plain
// chat; image-text-to-text (VL) models also accept a text turn.
const CHAT_TASKS = new Set(["text-generation", "image-text-to-text"]);

// ── built-in fallback ──────────────────────────────────────────────────
// Used ONLY when the live /models fetch fails, so the dropdown is never empty.
// Kept tiny on purpose; the runtime list replaces it once the fetch resolves.
export const FALLBACK_CHAT_MODEL_OPTIONS: ChatModelOption[] = [
  // 3B Instruct first → it's the DEFAULT (DEFAULT_MODEL = [0]). A general
  // instruct model is far less robotic as the intermediary/narrator than the
  // 1.5B *Coder*; it's also the server's own text-generation default.
  {
    key: "Qwen2.5-3B-Instruct-GGUF",
    label: "Qwen2.5-3B-Instruct-GGUF",
    shortLabel: "Qwen2.5-3B-Instruct-GGUF",
    maxNewTokens: DEFAULT_MAX_NEW_TOKENS,
    defaultMaxNewTokens: DEFAULT_DEFAULT_MAX_NEW_TOKENS,
  },
  {
    key: "Qwen2.5-Coder-1.5B-Instruct-GGUF",
    label: "Qwen2.5-Coder-1.5B-Instruct-GGUF",
    shortLabel: "Qwen2.5-Coder-1.5B-Instruct-GGUF",
    maxNewTokens: DEFAULT_MAX_NEW_TOKENS,
    defaultMaxNewTokens: DEFAULT_DEFAULT_MAX_NEW_TOKENS,
  },
];

// ── live registry (mutable singleton) ──────────────────────────────────
// Starts as the fallback list and is swapped for the fetched list on mount.
// getChatModelOption / getChatModelLabel resolve against this.
let CHAT_MODEL_OPTIONS_LIVE: ChatModelOption[] = [...FALLBACK_CHAT_MODEL_OPTIONS];

/** Back-compat export: the current model list (fallback until fetch resolves). */
export const CHAT_MODEL_OPTIONS: ChatModelOption[] = FALLBACK_CHAT_MODEL_OPTIONS;

export const DEFAULT_MODEL = FALLBACK_CHAT_MODEL_OPTIONS[0].key;

export function getChatModelOptions(): ChatModelOption[] {
  return CHAT_MODEL_OPTIONS_LIVE;
}

function setChatModelOptions(options: ChatModelOption[]): void {
  CHAT_MODEL_OPTIONS_LIVE = options.length
    ? options
    : [...FALLBACK_CHAT_MODEL_OPTIONS];
}

export function getChatModelOption(modelKey: string): ChatModelOption {
  return (
    CHAT_MODEL_OPTIONS_LIVE.find((model) => model.key === modelKey) ??
    CHAT_MODEL_OPTIONS_LIVE[0]
  );
}

export function getChatModelLabel(modelKey: string): string {
  return getChatModelOption(modelKey).key;
}

// ── /models → ChatModelOption ───────────────────────────────────────────
type ModelsEntry = {
  model_key?: unknown;
  primary_task?: unknown;
  name?: unknown;
  media?: unknown;
  media_default?: unknown;
};

function toOption(entry: ModelsEntry): ChatModelOption | null {
  const key = typeof entry.model_key === "string" ? entry.model_key : "";
  if (!key) return null;

  const label = typeof entry.name === "string" && entry.name ? entry.name : key;

  return {
    key,
    label,
    shortLabel: key,
    maxNewTokens: DEFAULT_MAX_NEW_TOKENS,
    defaultMaxNewTokens: DEFAULT_DEFAULT_MAX_NEW_TOKENS,
  };
}

/**
 * Fetch GET {apiBase}/models, keep chat-capable models (text-generation /
 * image-text-to-text), and map them to ChatModelOption. On success the live
 * registry is swapped so getChatModelOption/getChatModelLabel see the new list.
 *
 * Returns the resolved list (or the fallback list on any failure) so callers
 * can seed their UI state directly. Never throws — the dropdown must never be
 * empty.
 */
export async function fetchChatModelOptions(
  signal?: AbortSignal,
): Promise<ChatModelOption[]> {
  try {
    const response = await fetch(`${hugpyConfig.apiBase}/models`, {
      signal,
      ...(hugpyConfig.withCredentials ? { credentials: "include" } : {}),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = (await response.json()) as unknown;
    if (!Array.isArray(data)) throw new Error("unexpected /models shape");

    const chatCapable = (data as ModelsEntry[]).filter((entry) => {
      // Capability = ANY task in the list, not just the primary — a model
      // whose primary is e.g. image-text-to-text but also does
      // text-generation is chat-capable.
      const tasks = (entry as { tasks?: unknown })?.tasks;
      if (Array.isArray(tasks) && tasks.some((t) => typeof t === "string" && CHAT_TASKS.has(t))) {
        return true;
      }
      return (
        typeof entry?.primary_task === "string" &&
        CHAT_TASKS.has(entry.primary_task)
      );
    });

    // Honor the admin "Media" flag: offer only media-enabled chat models. If the
    // operator hasn't enabled any, fall back to every chat-capable model so the
    // dropdown never renders empty.
    const flagged = chatCapable.filter((entry) => entry.media === true);
    const chosen = flagged.length ? flagged : chatCapable;

    const options = chosen
      .map(toOption)
      .filter((option): option is ChatModelOption => option !== null);

    if (!options.length) throw new Error("no chat-capable models");

    // The server-designated default media model (POST /api/models/<key>/media-default)
    // comes back flagged as `media_default: true` on its entry — float it to the
    // front so it's first in the dropdown AND the initial selection. Concrete and
    // shared across clients, not a per-browser guess.
    const defaultEntry = chosen.find(
      (e) => (e as ModelsEntry).media_default === true,
    );
    const defKey =
      typeof defaultEntry?.model_key === "string" ? defaultEntry.model_key : "";
    const ordered = defKey ? withKeyFirst(options, defKey) : options;

    setChatModelOptions(ordered);
    return ordered;
  } catch (err) {
    console.warn("[fetchChatModelOptions] falling back to built-in list", err);
    return getChatModelOptions();
  }
}

// Move the option with the given key to the front, if present.
function withKeyFirst(options: ChatModelOption[], key: string): ChatModelOption[] {
  const i = options.findIndex((o) => o.key === key);
  if (i <= 0) return options;
  const copy = options.slice();
  const [d] = copy.splice(i, 1);
  copy.unshift(d);
  return copy;
}
