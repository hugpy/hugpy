// Client-side conversation history — PER SESSION (sessionStorage). The media arm
// has no server-side chat-history endpoint; Recents + Search live in the browser
// scoped to the tab session: they survive an in-session reload but are wiped when
// the tab/session closes (demo posture — nothing lingers across sessions).
// Degrades silently to in-memory when storage is unavailable.
import type { ChatEntry } from "../imports";
import { isCanned } from "../../../demo/mode";

export interface Conversation {
  id: string;
  title: string;
  updatedAt: string; // ISO; sortable lexicographically
  messages: ChatEntry[];
}

const KEY = "hugpy.media.conversations.v1";
const MAX = 50;

// One-time purge of any history persisted by the previous localStorage-backed
// build, so conversations from an earlier session can't survive into this one.
try {
  localStorage.removeItem(KEY);
} catch {
  // storage unavailable — nothing to purge.
}

export function deriveTitle(messages: ChatEntry[]): string {
  const firstUser = messages.find(
    (m) => (m.kind ?? "chat") === "chat" && (m.query ?? "").trim(),
  );
  const raw = (firstUser?.query ?? "").trim().replace(/\s+/g, " ");
  if (!raw) return "New chat";
  return raw.length > 60 ? raw.slice(0, 57) + "…" : raw;
}

// Tool-run results can be megabytes (base64 images) — well past the localStorage
// quota. Persist tool turns as lightweight markers (result/input stripped); the
// conversation text (the substance) is preserved, and panels re-render on a
// fresh run. `restored` tells ToolRunTurn not to try rendering an empty panel.
function lighten(messages: ChatEntry[]): ChatEntry[] {
  return messages.map((m) =>
    m.kind === "tool"
      ? { ...m, result: undefined, input: undefined, restored: true }
      : m,
  );
}

export function loadConversations(): Conversation[] {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return (parsed as Conversation[])
      .filter(
        (c) =>
          c &&
          typeof c.id === "string" &&
          typeof c.title === "string" &&
          Array.isArray(c.messages),
      )
      .sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1));
  } catch {
    return [];
  }
}

export function saveConversations(list: Conversation[]): void {
  // The canned demo is in-memory only: never let any save path (auto-save,
  // delete, or "Clear all") write the sample thread — or an empty list — over a
  // real visitor's localStorage history on the same origin.
  if (isCanned()) return;
  try {
    const trimmed = [...list]
      .sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1))
      .slice(0, MAX)
      .map((c) => ({ ...c, messages: lighten(c.messages) }));
    sessionStorage.setItem(KEY, JSON.stringify(trimmed));
  } catch {
    // quota exceeded / storage unavailable — degrade to in-memory silently.
  }
}

/** Move/insert `conv` to the front (most-recent-first). */
export function upsertConversation(
  list: Conversation[],
  conv: Conversation,
): Conversation[] {
  const next = list.filter((c) => c.id !== conv.id);
  next.unshift(conv);
  return next;
}
