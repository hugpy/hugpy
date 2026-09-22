import { hugpyConfig } from "../../../config";
import { parseThinking } from "./thinking";

export function formatClock(value?: string): string {
  if (!value) return "pending";

  return new Date(value).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function getApiBase(): string {
  // Single source of truth: config.ts (VITE_HUGPY_API_BASE or the configured default).
  // The chat module no longer carries its own duplicate base-URL literal.
  return hugpyConfig.apiBase;
}

export function nowIso(): string {
  return new Date().toISOString();
}

export function createChatId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function chatResponseToText(response: string | string[]): string {
  if (Array.isArray(response)) return response.join("\n");
  // Copy/share the answer only — <think> reasoning stays in the UI.
  return parseThinking(response).answer;
}

export async function copyToClipboard(text: string): Promise<void> {
  if (!text.trim()) return;

  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";

  document.body.appendChild(textarea);
  textarea.select();

  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }
}