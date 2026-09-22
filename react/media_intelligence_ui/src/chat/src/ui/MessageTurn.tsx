/*
 * MessageTurn.tsx — one user message + one assistant response.
 *
 * Convo's actual visual rules (this is the deviation from a bubble-style
 * chat that the previous version got wrong):
 *
 *   - The assistant message has NO bubble. No background, no border. The
 *     prose sits inline in the thread at full thread-width, like a doc.
 *
 *   - The user message IS a pill: right-aligned, soft `--message-surface`
 *     tint, rounded-3xl, max-width ~70% of the column.
 *
 *   - Action buttons (copy/share/edit) live BELOW the message, revealed
 *     on hover via `group-hover`. No always-on chrome — convo's restraint.
 *
 *   - A thin status line under the assistant message replaces the previous
 *     status-bordered bubble. Streaming/cancelled/error show as text, not
 *     as a colored frame.
 */

import { useState } from "react";
import type { ChatEntry } from "./../imports";
import {
  chatResponseToText,
  copyToClipboard,
  formatClock,
} from "./../utilities";
import { Icon } from "./Icons";
import ChatResponse from "./ChatResponse";
import ChatTurnActions from "./ChatTurnActions";
import { fileKindGlyph } from "../utilities/fileUpload";

interface MessageTurnProps {
  chat: ChatEntry;
  modelLabel: string;
  models?: { key: string; label: string }[];
  busy?: boolean;
  onEditQuery?: (chatId: string, newText: string) => void;
  onRegenerate?: (chatId: string, modelKey?: string) => void;
  onDelete?: (chatId: string) => void;
  onOpenFile?: (id: string) => void;
}

const STATUS_TEXT: Record<NonNullable<ChatEntry["status"]>, string | null> = {
  queued:    "Queued",
  thinking:  null,           // typing dots render instead
  streaming: null,           // tokens stream in place
  complete:  null,           // no status line — clean
  cancelled: "Stopped",
  error:     "Error",
};

export default function MessageTurn({
  chat,
  modelLabel,
  models = [],
  busy = false,
  onEditQuery,
  onRegenerate,
  onDelete,
  onOpenFile,
}: MessageTurnProps): JSX.Element | null {
  // Defensive: tool entries are routed to ToolRunTurn by Thread; never here.
  if (chat.kind === "tool") return null;
  const status = chat.status ?? "complete";
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState("");

  const currentModelLabel = chat.modelLabel || modelLabel;

  const hasCopyableResponse =
    Boolean(chat.response) && status !== "queued" && status !== "thinking";

  async function handleCopy() {
    await copyToClipboard(chatResponseToText(chat.response));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  async function handleShare() {
    const text = [
      `Prompt: ${chat.query}`,
      chatResponseToText(chat.response),
    ].join("\n\n");

    if (navigator.share) {
      await navigator.share({ title: `${currentModelLabel} response`, text });
      return;
    }

    await copyToClipboard(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  function startEditing() {
    setEditText(chat.query ?? "");
    setEditing(true);
  }
  function saveEdit() {
    const next = editText.trim();
    setEditing(false);
    if (next && next !== (chat.query ?? "").trim()) onEditQuery?.(chat.id, next);
  }

  // Provenance: the files this turn analyzed, the analysis it ran, and any link
  // in the prompt. Drives the (enabled-only-when-real) Sources button.
  const sources: { label: string; href?: string }[] = [];
  (chat.files ?? []).forEach((f) => sources.push({ label: f.name }));
  if (chat.toolUsed) sources.push({ label: `Analysis · ${chat.toolUsed}` });
  const urlInQuery = chat.query?.match(/https?:\/\/[^\s<>"']+/i)?.[0];
  if (urlInQuery) sources.push({ label: urlInQuery, href: urlInQuery });

  const statusText = STATUS_TEXT[status];

  return (
    <article
      className="flex flex-col gap-2 py-5"
      data-conversation-turn={chat.id}
    >
      {/* ─── USER MESSAGE (soft pill, right-aligned) ─────────────────── */}
      <div
        className="group/user-msg flex justify-end"
        data-message-author-role="user"
        data-message-id={`${chat.id}-user`}
      >
        <div className="flex max-w-[70%] flex-col items-end gap-1">
          {/* attachments used in this turn — click to preview */}
          {chat.files && chat.files.length > 0 && (
            <div className="flex flex-wrap justify-end gap-1.5">
              {chat.files.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => onOpenFile?.(f.id)}
                  title={f.name}
                  className="
                    flex max-w-[220px] items-center gap-1.5 rounded-xl
                    border border-token-border-light bg-token-surface-secondary
                    px-2.5 py-1 text-[12px] text-token-text-secondary
                    hover:bg-token-surface-hover
                  "
                >
                  <span aria-hidden>{fileKindGlyph(f.kind)}</span>
                  <span className="truncate">{f.name}</span>
                </button>
              ))}
            </div>
          )}
          {editing ? (
            <div className="flex w-full min-w-[280px] flex-col gap-2">
              <textarea
                autoFocus
                rows={Math.min(8, Math.max(2, editText.split("\n").length))}
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    saveEdit();
                  }
                  if (e.key === "Escape") setEditing(false);
                }}
                className="
                  w-full resize-none rounded-2xl px-4 py-2.5 text-[15px] leading-6
                  text-token-text-primary outline-none border border-token-border-light
                "
                style={{ background: "var(--message-surface)" }}
              />
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setEditing(false)}
                  className="rounded-full px-3 py-1 text-[13px] text-token-text-secondary hover:bg-token-surface-hover"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={saveEdit}
                  disabled={!editText.trim() || busy}
                  className="
                    rounded-full border border-token-border-light px-3 py-1 text-[13px]
                    text-token-text-primary hover:bg-token-surface-hover
                    disabled:opacity-40 disabled:cursor-not-allowed
                  "
                >
                  Send
                </button>
              </div>
            </div>
          ) : (
            <>
              <div
                className="
                  rounded-3xl px-4 py-2.5
                  text-[15px] leading-6
                  text-token-text-primary
                "
                style={{ background: "var(--message-surface)" }}
              >
                <p className="whitespace-pre-wrap">{chat.query}</p>
              </div>

              {/* hover-revealed actions on the user message */}
              <div
                className="
                  flex items-center gap-1
                  opacity-0 transition-opacity duration-150
                  group-hover/user-msg:opacity-100 focus-within:opacity-100
                "
              >
                <button
                  type="button"
                  onClick={startEditing}
                  disabled={!onEditQuery || busy}
                  aria-label="Edit message"
                  title="Edit"
                  className="
                    inline-flex h-7 w-7 items-center justify-center rounded-full
                    text-token-text-tertiary hover:bg-token-surface-hover
                    disabled:opacity-40 disabled:cursor-not-allowed
                  "
                >
                  <Icon name="edit" width={14} height={14} />
                </button>
                <span className="text-[11px] text-token-text-tertiary">
                  {formatClock(chat.queryStartedAt)}
                  {currentModelLabel && (
                    <span className="ml-1 opacity-80" title="model queried for this turn">
                      · {currentModelLabel}
                    </span>
                  )}
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* ─── ASSISTANT MESSAGE (no bubble — inline prose) ────────────── */}
      <div
        className="group/asst-msg flex flex-col gap-2"
        data-message-author-role="assistant"
        data-message-id={`${chat.id}-assistant`}
      >
        <ChatResponse chat={chat} modelLabel={currentModelLabel} />

        {chat.toolUsed && (
          <div className="text-[11px] text-token-text-tertiary opacity-70">
            ✦ analysis · {chat.toolUsed}
          </div>
        )}

        {/* status line — only shown for non-clean states */}
        {statusText && (
          <div className="text-[12px] text-token-text-tertiary">
            {statusText}
            {chat.responseFinishedAt && (
              <span className="ml-2">{formatClock(chat.responseFinishedAt)}</span>
            )}
          </div>
        )}

        {/* hover-revealed action row under the response */}
        {hasCopyableResponse && (
          <div
            className="
              opacity-0 transition-opacity duration-150
              group-hover/asst-msg:opacity-100 focus-within:opacity-100
            "
          >
            <ChatTurnActions
              copied={copied}
              canCopy={hasCopyableResponse}
              canShare={hasCopyableResponse}
              busy={busy}
              models={models}
              currentModelKey={chat.modelKey}
              sources={sources}
              onCopy={handleCopy}
              onShare={handleShare}
              onSwitchModel={(key) => onRegenerate?.(chat.id, key)}
              onRegenerate={() => onRegenerate?.(chat.id)}
              onDelete={() => onDelete?.(chat.id)}
            />
          </div>
        )}
      </div>
    </article>
  );
}
